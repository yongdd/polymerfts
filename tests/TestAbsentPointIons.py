#!/usr/bin/env python3
"""Absent canonical point ions at Q=0: observables, dynamics and checkpoint continuation."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, SCFT, PlatformSelector
from cuda_part import cuda_part_missing


PSI_ZERO_Q = 1j * np.array([0.0, 0.0, np.pi, -np.pi])


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def parameters(directory, platform, steps=4, concentration=0.0):
    params = {
        "nx": [4], "lx": [2.0], "ds": 1.0, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0}, "chi_n": {}, "zeta_n": 100.0,
        "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 1.0}]}],
        "smearing": {"type": "gaussian", "a_int": 0.1},
        "electrostatics": {"charges": {"A": 0.0}, "coupling": 1.0},
        "platform": platform, "langevin": {"nbar": 10000.0, "dt": 0.01, "max_step": steps},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 4, "pressure": True, "concentrations": True},
        "verbose_level": 0,
    }
    if concentration is not None:
        # A positive length whose multiplier rounds to one: the four finite weights cancel exactly.
        params["electrostatics"]["ions"] = {
            "unused": {"valence": 1.0, "concentration": concentration, "smearing_length": 1e-30}}
    return params


def make(params):
    sim = quiet(CLFTS, params, random_seed=7)
    sim.draw_architectures = False
    return sim


def absent_observables(root, platform):
    absent = make(parameters(root / "absent", platform))
    plain = make(parameters(root / "plain", platform, concentration=None))
    for sim in (absent, plain):
        sim.electrostatics.psi = PSI_ZERO_Q
    es, control = absent.electrostatics, plain.electrostatics
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise", over="raise"):
        warnings.simplefilter("error", RuntimeWarning)
        densities, partitions = es.ion_densities()
        assert partitions == [0j], "Keep the true zero partition for diagnostics"
        np.testing.assert_array_equal(densities[0], np.zeros(4))
        np.testing.assert_array_equal(es.ion_charge(), np.zeros(4))
        np.testing.assert_array_equal(es.ion_number_densities(), [0j])
        np.testing.assert_array_equal(es.ion_chemical_potentials(), [complex(-np.inf, 0.0)])
        assert es.ion_partition_ratios() == [0.0]
        assert np.isfinite(es.hamiltonian()) and es.hamiltonian() == control.hamiltonian()
        assert es.pressure(1, 200.0) == control.pressure(1, 200.0)
        w_aux = absent.mpt.to_aux_fields(np.zeros((1, 4), dtype=complex))
        phi, _ = quiet(absent.compute_concentrations, w_aux)
        plain_phi, _ = quiet(plain.compute_concentrations, w_aux)
        np.testing.assert_array_equal(absent.evaluate_force(w_aux), plain.evaluate_force(w_aux))
        for operator in ("pressure", "pressure_virial"):
            value = getattr(absent.thermodynamics, operator)(w_aux, phi)
            expected = getattr(plain.thermodynamics, operator)(w_aux, plain_phi)
            assert np.isfinite(value) and value == expected, operator


def compare_files(left, right, filename, prefixes):
    a, b = (loadmat(path / filename, squeeze_me=True) for path in (left, right))
    keys = [key for key in a if key.startswith(prefixes)]
    assert keys, filename
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


def run_and_restart(root, platform):
    paths = {name: root / name for name in ("full", "plain", "partial", "continued")}
    full = make(parameters(paths["full"], platform))
    plain = make(parameters(paths["plain"], platform, concentration=None))
    partial = make(parameters(paths["partial"], platform, steps=2))
    continued = make(parameters(paths["continued"], platform))
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise", over="raise"):
        warnings.simplefilter("error", RuntimeWarning)
        for sim in (full, plain, partial):
            status = quiet(sim.run, {"A": np.zeros(4)}, initial_psi=PSI_ZERO_Q)
            assert status["status"] == "completed"
        status = quiet(continued.continue_run, paths["partial"] / "fields_000002.mat")
        assert status["status"] == "completed"
    for other in ("plain", "continued"):
        compare_files(paths["full"], paths[other], "fields_000004.mat",
                      ("w_", "phi_", "normal_noise", "psi_", "rng_state"))
        compare_files(paths["full"], paths[other], "dH_000004.mat", ("H_", "pressure_"))
        compare_files(paths["full"], paths[other], "structure_function_000004.mat",
                      ("structure_function_", "charge_structure_function_", "concentration_A_"))
    compare_files(paths["full"], paths["continued"], "dH_000004.mat",
                  ("chemical_potential_", "partition_ratio_"))
    compare_files(paths["full"], paths["continued"], "structure_function_000004.mat", ("concentration_",))
    for name in ("full", "continued"):
        recorded = loadmat(paths[name] / "dH_000004.mat", squeeze_me=True)
        assert np.all(np.isneginf(recorded["chemical_potential_history_real"][:, 1]))
        np.testing.assert_array_equal(recorded["chemical_potential_history_imag"][:, 1], np.zeros(4))
        concentrations = loadmat(paths[name] / "structure_function_000004.mat", squeeze_me=True)
        for part in ("real", "imag"):
            np.testing.assert_array_equal(concentrations[f"concentration_unused_{part}"], np.zeros(4))


def occupied_controls(root, platform):
    params = parameters(root / "occupied", platform, concentration=0.1)
    params["electrostatics"]["screening_length"] = 1.0   # the occupied ion carries net charge
    sim = make(params)
    plain_params = parameters(root / "occupied_plain", platform, concentration=None)
    plain_params["electrostatics"]["screening_length"] = 1.0
    plain = make(plain_params)
    es = sim.electrostatics
    psi = np.array([-0.2, 0.1, 0.25, -0.15]) + 0.2 * PSI_ZERO_Q
    es.psi = plain.electrostatics.psi = psi
    weight = np.exp(-psi)
    q, density = np.mean(weight), 0.1 * weight / np.mean(weight)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        densities, partitions = es.ion_densities()
        np.testing.assert_allclose(partitions, [q], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(densities[0], density, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(es.ion_chemical_potentials(), [np.log(0.1) - np.log(q)], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(es.hamiltonian() - plain.electrostatics.hamiltonian(), -0.1 * np.log(q),
                                   rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(es.pressure(1, 200.0) - plain.electrostatics.pressure(1, 200.0), 0.1,
                                   rtol=1e-13, atol=1e-14)
    # An occupied canonical species really is singular at Q=0; the fix must not clamp its partition.
    es.psi = PSI_ZERO_Q
    with np.errstate(divide="ignore", invalid="ignore"):
        densities, partitions = es.ion_densities()
        assert partitions == [0j]
        assert not np.all(np.isfinite(densities[0]))
        assert not np.isfinite(es.hamiltonian())
        assert not np.isfinite(es.ion_chemical_potentials()[0])


def mean_field_underflow(root, platform):
    simulations = []
    for concentration in (0.0, None):
        params = parameters(root / "scft", platform, concentration=concentration)
        for key in ("langevin", "recording"):
            params.pop(key)
        params.update(relax_cell=False, max_iter=1, tolerance=1e-8,
                      optimizer={"name": "am", "max_hist": 10, "start_error": 0.1,
                                 "mix_min": 0.1, "mix_init": 0.1})
        params["electrostatics"]["screening_length"] = 1.0
        simulations.append(quiet(SCFT, params))
    absent, plain = simulations
    es, control = absent.electrostatics, plain.electrostatics
    psi = np.full(4, 1000.0)   # finite real potential with exp(-psi) = Q = 0 in double precision
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        warnings.simplefilter("error", RuntimeWarning)
        densities, partitions = es.ion_densities(psi)
        assert partitions == [0.0]
        np.testing.assert_array_equal(densities[0], np.zeros(4))
        np.testing.assert_array_equal(es.ion_chemical_potentials(psi), [-np.inf])
        np.testing.assert_array_equal(es.ion_number_densities(psi), [0.0])
        np.testing.assert_array_equal(es.charge_density({"A": np.ones(4)}, psi), np.zeros(4))
        assert np.isfinite(es.free_energy(psi)) and es.free_energy(psi) == control.free_energy(psi)
        k_sq = absent.smearing.k_sq
        args = (psi, absent.smearing.coefficients, k_sq, {0: -k_sq})
        np.testing.assert_array_equal(es.stress(*args), control.stress(*args))


def main():
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_absent_ions_") as directory:
        for platform in platforms:
            root = Path(directory) / platform
            absent_observables(root, platform)
            run_and_restart(root, platform)
            occupied_controls(root, platform)
            mean_field_underflow(root, platform)
            print(f"[PASS] {platform}: absent Q=0 ions, dynamics, restart and occupied controls")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
