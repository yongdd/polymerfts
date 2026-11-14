#!/usr/bin/env python3
"""Canonical ion densities at nonzero subnormal Q, physical poles, and exact restarts."""

import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, SCFT, PlatformSelector
from polymerfts.theory.electrostatics import _point_ion_density
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


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
            density, partition = es.ion_densities()
            assert 0 < partition[0].real < np.inf and partition[0].imag == 0
            np.testing.assert_allclose(partition[0].real, np.exp(-shift), rtol=2e-12, atol=0)
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
        mean_params.pop("langevin")
        mean_params.pop("recording")
        mean_params.update(relax_cell=False, max_iter=1, tolerance=1e-8,
                           optimizer={"name": "am", "max_hist": 10, "start_error": 0.1,
                                      "mix_min": 0.1, "mix_init": 0.1})
        mean = quiet(SCFT, mean_params)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            density, partition = mean.electrostatics.ion_densities(psi)
            np.testing.assert_allclose(density[0], 1.0, rtol=0, atol=2e-14)
            np.testing.assert_allclose(partition[0], np.exp(-shift), rtol=2e-12, atol=0)
            np.testing.assert_allclose(mean.electrostatics.free_energy(psi), expected_energy - 1.0,
                                       rtol=0, atol=2e-10)


def shifted_weights_and_poles():
    ion = {"concentration": 0.2, "chemical_potential": None}
    delta = np.array([0.3 + 0.1j, -0.2 - 0.2j, 0.15 + 0.05j, -0.25 + 0.05j])
    expected = 0.2 * np.exp(-delta) / np.mean(np.exp(-delta))
    for shift in (0.0, 700.0, 710.0, 720.0, 1000.0, -710.0):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            density, partition = _point_ion_density(ion, shift + delta, 1.0)
        np.testing.assert_allclose(density, expected, rtol=0, atol=2e-14)
        if shift == 1000.0:
            assert partition == 0.0, "Keep an underflowed diagnostic Q"
        if shift == -710.0:
            assert not np.isfinite(partition), "Keep an overflowing diagnostic Q"
    pole = 1j * np.array([0.0, 0.0, np.pi, -np.pi])
    with np.errstate(divide="ignore", invalid="ignore"):
        density, partition = _point_ion_density(ion, pole, 1.0)
    assert partition == 0.0 and not np.all(np.isfinite(density)), "An occupied species retains its true pole"
    absent = {**ion, "concentration": 0.0}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        density, partition = _point_ion_density(absent, pole, 1.0)
    assert partition == 0.0
    np.testing.assert_array_equal(density, np.zeros(4))
    grand = {"concentration": None, "chemical_potential": 710.0}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        density, partition = _point_ion_density(grand, 710.0 + delta, 1.0)
    np.testing.assert_allclose(density, np.exp(-delta), rtol=0, atol=1e-13)
    assert np.isfinite(partition) and partition.real > 0


def trajectory_and_restart(root, platform):
    paths = {name: root / name for name in ("full", "partial", "continued")}
    sims = {}
    for name, steps, seed in (("full", 3, 17), ("partial", 1, 17), ("continued", 3, 99)):
        sims[name] = quiet(CLFTS, parameters(paths[name], platform, 710.0, steps), random_seed=seed)
        sims[name].draw_architectures = False
    for name in ("full", "partial"):
        assert quiet(sims[name].run, {"A": np.zeros(4)}, initial_psi=np.full(4, 710.0))["status"] == "completed"
        assert quiet(sims[name].prop_solver.check_total_partition)
    checkpoint = paths["partial"] / "fields_000001.mat"
    assert loadmat(checkpoint, squeeze_me=True)["record_sample_count"] == 1
    assert quiet(sims["continued"].continue_run, checkpoint)["status"] == "completed"
    assert sims["full"].random_bg.state == sims["continued"].random_bg.state
    assert sims["full"].electrostatics.random_bg.state == sims["continued"].electrostatics.random_bg.state
    for filename in ("fields_000002.mat", "fields_000003.mat", "dH_000002.mat", "structure_function_000002.mat"):
        full, continued = (loadmat(paths[name] / filename, squeeze_me=True) for name in ("full", "continued"))
        keys = [key for key in full if key.startswith(("w_", "phi_", "psi_", "normal_noise", "rng_state", "record_",
                                                     "H_", "chemical_potential_", "pressure_", "structure_function_",
                                                     "charge_structure_function_", "concentration_"))]
        assert keys
        for key in keys:
            np.testing.assert_array_equal(full[key], continued[key], err_msg=f"{filename}: {key}")


def main():
    shifted_weights_and_poles()
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_canonical_ions_") as directory:
        for platform in platforms:
            root = Path(directory) / platform
            uniform_operators(root / "operators", platform)
            trajectory_and_restart(root / "restart", platform)
            print(f"[PASS] {platform}: canonical ion range, physical poles and exact restart")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
