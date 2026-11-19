#!/usr/bin/env python3
"""Canonical ion densities and ln Q in large uniform potentials, and a run in one.

The ion densities are Python: the first CPU platform runs them. The poles of an
absent and of an occupied ion are checked by TestAbsentPointIons.py.
"""

import copy
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, SCFT
from polymerfts.theory.electrostatics import _point_ion
from fts_checks import python_platform, quiet


def parameters(directory, platform, shift, steps=3):
    return {
        "nx": [4], "lx": [2.0], "ds": 1.0, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0}, "chi_n": {}, "zeta_n": 10.0,
        "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 1.0}]}],
        "smearing": {"type": "gaussian", "a_int": 0.1},
        "electrostatics": {
            "charges": {"A": 0.0}, "coupling": max(shift, 1.0), "screening_length": 1.0,
            "counterion": None,
            "ions": {"cation": {"valence": 1.0, "concentration": 1.0, "smearing_length": 0.1}}},
        "platform": platform, "verbose_level": 0,
        "langevin": {"nbar": 1e4, "dt": 0.001, "max_step": steps},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 2, "pressure": True, "concentrations": True},
    }


def uniform_operators(root, platform):
    for shift in (0.0, 700.0, 710.0, 720.0):
        params = parameters(root, platform, shift)
        sim = quiet(CLFTS, params, random_seed=7)
        es = sim.electrostatics
        psi = np.full(4, shift)
        es.psi = psi
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            density, log_partition = es.ion_densities()
            assert np.imag(log_partition[0]) == 0
            np.testing.assert_allclose(np.real(log_partition[0]), -shift, rtol=0, atol=2e-12)
            np.testing.assert_allclose(density[0], 1.0, rtol=0, atol=2e-14)
            np.testing.assert_allclose(es.ion_number_densities(), [1.0], rtol=0, atol=0)
            np.testing.assert_allclose(es.ion_chemical_potentials(), [shift], rtol=0, atol=2e-10)
            charge = es.charge_density({"A": np.ones(4)})
            np.testing.assert_allclose(charge, 1.0, rtol=0, atol=2e-14)
            np.testing.assert_allclose(es.force(charge), 1.0 - shift / max(shift, 1.0), rtol=0, atol=2e-14)
            expected_energy = shift - 0.5 * shift ** 2 / max(shift, 1.0)
            np.testing.assert_allclose(es.hamiltonian(), expected_energy, rtol=0, atol=2e-10)
            fields = sim.mpt.to_aux_fields({"A": np.zeros(4, dtype=complex)})
            phi, _ = quiet(sim.compute_concentrations, fields)
            for operator in ("pressure", "pressure_virial"):
                assert np.isfinite(getattr(sim.thermodynamics, operator)(fields, phi))
            assert quiet(sim.prop_solver.check_total_partition)

        mean_params = copy.deepcopy(params)
        for key in ("langevin", "recording", "verbose_level"):   # options of CL-FTS, which SCFT refuses
            mean_params.pop(key)
        mean_params.update(relax_cell=False, max_iter=1, tolerance=1e-8,
                           optimizer={"name": "am", "max_hist": 10, "start_error": 0.1,
                                      "mix_min": 0.1, "mix_init": 0.1})
        mean = quiet(SCFT, mean_params)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean.electrostatics.psi = psi
            density, log_partition = mean.electrostatics.ion_densities()
            np.testing.assert_allclose(density[0], 1.0, rtol=0, atol=2e-14)
            np.testing.assert_allclose(log_partition[0], -shift, rtol=0, atol=2e-12)
            np.testing.assert_allclose(mean.electrostatics.free_energy(), expected_energy - 1.0,
                                       rtol=0, atol=2e-10)


def shifted_weights():
    ion = {"concentration": 0.2, "chemical_potential": None}
    delta = np.array([0.3 + 0.1j, -0.2 - 0.2j, 0.15 + 0.05j, -0.25 + 0.05j])
    expected = 0.2 * np.exp(-delta) / np.mean(np.exp(-delta))
    log_mean = np.log(np.mean(np.exp(-delta)))
    for shift in (0.0, 700.0, 710.0, 720.0, 1000.0, -710.0):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            density, log_q, _ = _point_ion(ion, shift + delta, 1.0)
        np.testing.assert_allclose(density, expected, rtol=0, atol=2e-14)
        np.testing.assert_allclose(log_q, log_mean - shift, rtol=0, atol=1e-12)
    grand = {"concentration": None, "chemical_potential": 710.0}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        density, log_q, ratio = _point_ion(grand, 710.0 + delta, 1.0)
    np.testing.assert_allclose(density, np.exp(-delta), rtol=0, atol=1e-13)
    np.testing.assert_allclose(log_q, log_mean - 710.0, rtol=0, atol=1e-12)
    assert ratio is None


def trajectory(root, platform):
    sim = quiet(CLFTS, parameters(root, platform, 710.0), random_seed=17)
    sim.draw_architectures = False
    assert quiet(sim.run, {"A": np.zeros(4)}, initial_psi=np.full(4, 710.0))["status"] == "completed"
    assert quiet(sim.prop_solver.check_total_partition)
    assert loadmat(root / "fields_000001.mat", squeeze_me=True)["record_sample_count"] == 1
    recorded = loadmat(root / "dH_000002.mat", squeeze_me=True)
    keys = [key for key in recorded if key.startswith(("H_", "chemical_potential_", "pressure_"))]
    assert keys
    for key in keys:
        assert np.all(np.isfinite(recorded[key])), key


def main():
    shifted_weights()
    platform = python_platform()
    with tempfile.TemporaryDirectory(prefix="polymerfts_canonical_ions_") as directory:
        root = Path(directory)
        uniform_operators(root / "operators", platform)
        trajectory(root / "run", platform)
        print(f"[PASS] {platform}: canonical ion range and a run at a shift of 710")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
