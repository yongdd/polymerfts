#!/usr/bin/env python3
"""Finite grand-canonical ion densities with large compensating Donnan offsets.

Densities, thermodynamic operators and dynamics depend on exp(mu-u), and ln Q stays
finite where Q itself would under- or overflow. SCFT must reject a genuinely
non-finite state instead of reporting convergence. LM/Euler runs exercise both
noise streams and keep finite pending records. The ion densities are Python:
the first CPU platform runs them.
"""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, SCFT, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing
from polymerfts.utils.checkpoint import validate_checkpoint
from fts_checks import quiet


NX, LX = [4], [2.0]
FRACTIONS = {"A": 0.4, "B": 0.6}


def parameters(directory, platform, mu, method="lm", steps=4):
    return {
        "nx": NX, "lx": LX, "ds": 1.0, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 2.0}, "zeta_n": 10.0,
        "polymers": [{"volume_fraction": fraction,
                      "blocks": [{"type": name, "length": 1.0}]}
                     for name, fraction in FRACTIONS.items()],
        "smearing": {"type": "gaussian", "a_int": 0.1},
        "electrostatics": {"charges": {"A": 0.0, "B": 0.0}, "coupling": 1.0,
                           "fixed_charge": np.full(4, -1.0),
                           "ions": {"cation": {"valence": 1.0, "chemical_potential": mu,
                                                "smearing_length": 0.1}}},
        "platform": platform,
        "langevin": {"nbar": 10000.0, "dt": 0.001, "max_step": steps, "integrator": method},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 4, "pressure": True, "concentrations": True},
        "verbose_level": 0,
    }


def scft_parameters(directory, platform, mu):
    params = parameters(directory, platform, mu)
    params.pop("langevin")
    params.pop("recording")
    params.update(relax_cell=False, max_iter=2, tolerance=1e-8,
                  optimizer={"name": "am", "max_hist": 10, "start_error": 0.1,
                             "mix_min": 0.1, "mix_init": 0.1})
    return params


def check_log_partition(log_q, mu):
    """ln Q = -mu in the uniform potential u = mu."""
    assert np.imag(log_q) == 0
    np.testing.assert_allclose(np.real(log_q), -mu, rtol=0, atol=2e-12)


def uniform_operators(root, platform):
    for mu in (0.0, 700.0, 710.0, 1000.0, -710.0):
        sim = quiet(CLFTS, parameters(root, platform, mu), random_seed=7)
        es = sim.electrostatics
        psi = np.full(4, mu)
        es.psi = psi
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            densities, log_partitions = es.ion_densities()
            np.testing.assert_allclose(densities[0], 1.0, rtol=0, atol=2e-13)
            check_log_partition(log_partitions[0], mu)
            np.testing.assert_allclose(es.ion_number_densities(), [1.0], rtol=0, atol=2e-13)
            np.testing.assert_array_equal(es.ion_chemical_potentials(), [mu])
            np.testing.assert_allclose(es.hamiltonian(), -mu - 1.0, rtol=0, atol=3e-12)
            charge = es.charge_density({name: np.full(4, fraction) for name, fraction in FRACTIONS.items()})
            np.testing.assert_allclose(es.force(charge), 0.0, rtol=0, atol=2e-13)
            assert es.ion_partition_ratios() == [], "Grand-canonical ions have no partition pole"
            assert all("cation" not in name for name in sim.thermodynamics.partition_ratio_species)
            aux = sim.mpt.to_aux_fields(np.zeros((2, 4), dtype=np.complex128))
            phi, _ = quiet(sim.compute_concentrations, aux)
            for name, fraction in FRACTIONS.items():
                np.testing.assert_allclose(phi[name], fraction, rtol=0, atol=2e-13)
            for name in ("pressure", "pressure_virial"):
                value = getattr(sim.thermodynamics, name)(aux, phi)
                assert np.isfinite(value), (name, mu, value)
            assert quiet(sim.prop_solver.check_total_partition)

        mean = quiet(SCFT, scft_parameters(root, platform, mu))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            densities, log_partitions = mean.electrostatics.ion_densities(psi)
            np.testing.assert_allclose(densities[0], 1.0, rtol=0, atol=2e-13)
            check_log_partition(log_partitions[0], mu)
            np.testing.assert_allclose(mean.electrostatics.ion_number_densities(psi), [1.0], rtol=0, atol=2e-13)
            np.testing.assert_allclose(mean.electrostatics.free_energy(psi), -mu - 1.0, rtol=0, atol=3e-12)
        # The homogeneous root finder may visit overflowing brackets on its way
        # to the finite Donnan saddle; all returned physical quantities must be finite.
        with np.errstate(over="ignore"):
            result = quiet(mean.run, {name: np.zeros(4) for name in FRACTIONS},
                           initial_psi=psi, return_result=True)
        assert result.converged and np.isfinite(result.error_level)
        np.testing.assert_allclose(result.free_energy, 0.0, rtol=0, atol=3e-12)
        assert np.isfinite(result.free_energy_absolute)
        np.testing.assert_allclose(result.psi, psi, rtol=0, atol=2e-13)
        assert quiet(mean.prop_solver.check_total_partition)


def nonuniform_operators(root, platform):
    """Independent FFT oracle with nonuniform complex and real ion potentials."""
    k = 2 * np.pi * np.fft.fftfreq(4, d=LX[0] / 4)
    gamma = np.exp(-0.5 * 0.1 ** 2 * k ** 2)
    fluctuation = np.array([0.3, -0.2, 0.15, -0.25])
    for complex_fields in (False, True):
        delta = fluctuation + (1j * np.array([0.1, -0.2, 0.05, 0.05]) if complex_fields else 0.0)
        potential = np.fft.ifft(np.fft.fft(delta) * gamma)
        expected_density = np.exp(-potential)
        expected_number = np.mean(expected_density)
        laplacian = np.fft.ifft(-k ** 2 * np.fft.fft(delta))
        expected_energy_without_offset = 0.5 * np.mean(delta * laplacian) - np.mean(delta) - expected_number
        reference_pressure = None
        for mu in (0.0, 710.0, 1000.0, -710.0):
            if complex_fields:
                sim = quiet(CLFTS, parameters(root, platform, mu), random_seed=7)
                es = sim.electrostatics
                es.psi = mu + delta
                densities, _ = es.ion_densities()
                number, energy = es.ion_number_densities()[0], es.hamiltonian()
                pressure = es.pressure(1, 200.0) - mu
                if reference_pressure is None:
                    reference_pressure = pressure
                np.testing.assert_allclose(pressure, reference_pressure, rtol=0, atol=5e-10)
                assert np.isfinite(es.pressure_virial_correction(es.charge_density(
                    {name: np.full(4, fraction) for name, fraction in FRACTIONS.items()}), 1, 200.0))
            else:
                sim = quiet(SCFT, scft_parameters(root, platform, mu))
                es, psi = sim.electrostatics, mu + delta
                densities, _ = es.ion_densities(psi)
                number, energy = es.ion_number_densities(psi)[0], es.free_energy(psi)
            np.testing.assert_allclose(densities[0], expected_density, rtol=0, atol=5e-13)
            np.testing.assert_allclose(number, expected_number, rtol=0, atol=5e-13)
            np.testing.assert_allclose(energy + mu, expected_energy_without_offset, rtol=0, atol=5e-10)


class Observations:
    settings = {}

    def __init__(self):
        self.steps = []

    def __call__(self, sample):
        assert np.isfinite(sample["hamiltonian"])
        assert np.all(np.isfinite(sample["w_aux"]))
        for name, fraction in FRACTIONS.items():
            np.testing.assert_allclose(np.mean(sample["phi"][name]), fraction, rtol=0, atol=5e-8)
        self.steps.append(int(sample["step"]))

    def state_dict(self):
        return {"steps": list(self.steps)}

    def load_state_dict(self, state):
        self.steps = list(state["steps"])


def trajectory(root, platform, method, mu):
    observer = Observations()
    sim = quiet(CLFTS, parameters(root, platform, mu, method), random_seed=17,
                callbacks={"observations": observer})
    sim.draw_architectures = False
    initial = dict(zip(FRACTIONS, np.random.default_rng(3).normal(0.0, 5.0, (2, 4))))
    initial_psi = mu + np.array([0.3, -0.2, 0.15, -0.25])
    status = quiet(sim.run, initial, initial_psi=initial_psi)
    assert status["status"] == "completed" and status["last_completed_step"] == 4
    assert quiet(sim.prop_solver.check_total_partition)
    assert observer.steps == [1, 2, 3, 4]
    pending = loadmat(root / "fields_000002.mat", squeeze_me=True)
    validate_checkpoint(sim, pending)
    assert pending["record_sample_count"] == 2
    for key in ("record_H_history", "record_pressure_history"):
        assert np.size(pending[key]) == 2 and np.all(np.isfinite(pending[key])), key
    assert np.any(pending["normal_noise_prev"] != 0.0)
    if method == "lm":
        assert np.any(pending["psi_noise_prev"] != 0.0)
    else:
        np.testing.assert_array_equal(pending["psi_noise_prev"], np.zeros(4))
    recorded = loadmat(root / "dH_000004.mat", squeeze_me=True)
    keys = [key for key in recorded if key.startswith(("H_", "pressure_", "chemical_potential_", "number_density_"))]
    assert keys
    for key in keys:
        assert np.all(np.isfinite(recorded[key])), key


def true_overflow_rejected(root, platform):
    """At mu=1000, psi=0 the physical density exp(1000) itself is non-finite."""
    mean = quiet(SCFT, scft_parameters(root, platform, 1000.0))
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            quiet(mean.run, {name: np.zeros(4) for name in FRACTIONS},
                  initial_psi=np.zeros(4), return_result=True)
        except ValidationError as exc:
            assert "Non-finite SCFT" in str(exc), str(exc)
        else:
            raise AssertionError("A non-finite physical state must not return a converged SCFT result")


def main():
    # A fix in Python: one platform, a CPU one where the build has it
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return int(cuda_part_missing()) or 77
    platforms = [next((name for name in platforms if name.startswith("cpu")), platforms[0])]
    platform = platforms[0]
    with tempfile.TemporaryDirectory(prefix="polymerfts_gce_ion_range_") as directory:
        root = Path(directory)
        uniform_operators(root / "uniform", platform)
        nonuniform_operators(root / "nonuniform", platform)
        true_overflow_rejected(root / "invalid", platform)
        for method in ("lm", "euler"):
            for mu in (710.0, 1000.0):
                trajectory(root / f"{method}_{mu}", platform, method, mu)
        print(f"[PASS] {platform}: GCE ion range, finite SCFT states and LM/Euler runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
