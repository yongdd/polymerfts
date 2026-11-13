#!/usr/bin/env python3
"""Finite small field coefficients in H, its gradient and the LFTS uniform saddle.

Large finite compressibility must retain its quadratic term and uniform saddle
constraint. A nearly symmetric ternary interaction retains its small linear
term. Actual LM/Euler runs preserve RNG, previous noise and pending measurements
across checkpoints on every available backend.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, LFTS, PlatformSelector
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def parameters(directory, platform, zeta=1e8, method="lm", steps=4):
    return {
        "nx": [4], "lx": [2.0], "ds": 0.25, "chain_model": "continuous",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0}, "zeta_n": zeta,
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": platform,
        "langevin": {"nbar": 10000.0, "dt": 0.01, "max_step": steps, "integrator": method},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.1,
                       "mix_min": 0.1, "mix_init": 0.1},
        "saddle": {"max_iter": 200, "tolerance": 1e-9},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 4},
        "verbose_level": 0,
    }


def energy(sim, fields):
    """H against the complete polynomial and canonical log Q terms, without coefficient cutoffs."""
    phi, _ = quiet(sim.compute_concentrations, fields)
    partitions = [sim.prop_solver.get_partition_function(p)
                  for p in range(sim.prop_solver.get_n_polymer_types())]
    h = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, fields, partitions, sim.cb,
                                    include_const_term=True, phi=phi)
    exact = sim.mpt.h_const + np.sum(sim.mpt.h_coef_mu1 * np.mean(fields, axis=1)) \
        + np.sum(sim.mpt.h_coef_mu2 * np.mean(fields ** 2, axis=1))
    for p, q in enumerate(partitions):
        chain = sim.prop_solver.get_polymer(p)
        exact -= chain.get_volume_fraction() / chain.get_alpha() * np.log(q)
    return h, exact, phi


def quadratic_and_gradient(root, platform):
    for zeta in (1e6, 1e8):
        sim = quiet(CLFTS, parameters(root / f"quadratic_{zeta}", platform, zeta), random_seed=17)
        k = sim.mpt.aux_fields_imag_idx[-1]
        fields = np.zeros((2, 4), dtype=complex)
        fields[k] = 100.0
        h, exact, phi = energy(sim, fields)
        np.testing.assert_allclose(h, exact, rtol=0, atol=1e-11)
        if zeta == 1e8:
            assert 0 < abs(sim.mpt.h_coef_mu2[k]) < 1e-8
            assert abs(sim.mpt.h_coef_mu2[k] * np.mean(fields[k] ** 2)) > 4e-5
        gradient = -np.mean(sim.mpt.compute_func_deriv(fields, phi, [k]))
        delta = 0.1
        plus, minus = fields.copy(), fields.copy()
        plus[k] += delta
        minus[k] -= delta
        h_plus, exact_plus, _ = energy(sim, plus)
        h_minus, exact_minus, _ = energy(sim, minus)
        np.testing.assert_allclose((h_plus - h_minus) / (2 * delta), gradient, rtol=0, atol=1e-11)
        np.testing.assert_allclose((exact_plus - exact_minus) / (2 * delta), gradient, rtol=0, atol=1e-11)


def small_linear_term(root, platform):
    params = parameters(root / "linear", platform, zeta=100.0)
    params.update(segment_lengths={"A": 1.0, "B": 1.0, "C": 1.0},
                  chi_n={"A,B": 10.0, "A,C": 12.0 + 1e-7, "B,C": 12.0},
                  polymers=[{"volume_fraction": 1.0, "blocks": [
                      {"type": "A", "length": 0.25}, {"type": "B", "length": 0.25},
                      {"type": "C", "length": 0.5}]}])
    sim = quiet(CLFTS, params, random_seed=17)
    small = np.flatnonzero((np.abs(sim.mpt.h_coef_mu1) > 1e-12)
                          & (np.abs(sim.mpt.h_coef_mu1) < 1e-8))
    assert small.size, "The nearly symmetric interaction must supply a small finite linear coefficient"
    for sign in (-1, 1):
        fields = np.zeros((3, 4), dtype=complex)
        fields[small[0]] = sign * 100.0
        assert abs(sim.mpt.h_coef_mu1[small[0]] * np.mean(fields[small[0]])) > 1e-8
        h, exact, _ = energy(sim, fields)
        np.testing.assert_allclose(h, exact, rtol=0, atol=1e-10)


def uniform_saddle(root, platform):
    for zeta in (None, 1e6, 1e8):
        sim = quiet(LFTS, parameters(root / f"saddle_{zeta}", platform, zeta), random_seed=17)
        k = sim.mpt.aux_fields_imag_idx[-1]
        coefficient = sim.mpt.h_coef_mu2[k]
        if zeta is None:
            assert coefficient == 0.0, "The incompressible pressure keeps its exact structural zero"
            target = 0.0
        else:
            assert coefficient != 0.0
            target = -(sim.mpt.h_coef_mu1[k] + sim.mpt.matrix_a[:, k] @ sim.mean_concentrations()) / (2 * coefficient)
        fields = np.zeros((2, 4))
        fields[k] = 100.0
        phi, _, _, error = quiet(sim.saddle.solve, fields)
        assert error < sim.saddle.settings["tolerance"]
        np.testing.assert_allclose(np.mean(fields[k]), target, rtol=0, atol=1e-10)
        residual = sim.mpt.compute_func_deriv(fields, phi, sim.mpt.aux_fields_imag_idx)
        assert abs(np.mean(residual)) < sim.saddle.settings["tolerance"], residual


def compare_files(left, right, filename, prefixes):
    a, b = (loadmat(path / filename, squeeze_me=True) for path in (left, right))
    keys = [key for key in a if key.startswith(prefixes)]
    assert keys, filename
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


def dynamics_and_restart(root, platform):
    rng = np.random.default_rng(19)
    initial = {name: rng.normal(0.0, 5.0, 4) for name in ("A", "B")}
    for cls in (LFTS, CLFTS):
        for method in ("lm", "euler"):
            directory = root / f"{cls.__name__}_{method}"
            paths = {name: directory / name for name in ("full", "partial", "continued")}
            snapshots = {}

            def create(name, steps, seed):
                def observe(sample):
                    np.testing.assert_allclose(np.mean(sum(sample["phi"].values())), 1.0, rtol=0, atol=5e-8)
                    snapshots[name] = sample
                sim = quiet(cls, parameters(paths[name], platform, method=method, steps=steps),
                            random_seed=seed, callbacks={"observe": observe})
                sim.draw_architectures = False
                return sim

            full = create("full", 4, 17)
            partial = create("partial", 2, 17)
            continued = create("continued", 4, 99)
            for name, sim in (("full", full), ("partial", partial)):
                status = quiet(sim.run, initial)
                assert status["status"] == "completed"
                assert quiet(sim.prop_solver.check_total_partition)
            pending = loadmat(paths["partial"] / "fields_000002.mat", squeeze_me=True)
            assert pending["record_sample_count"] == 2
            assert np.size(pending["record_H_history"]) == 2
            status = quiet(continued.continue_run, paths["partial"] / "fields_000002.mat")
            assert status["status"] == "completed"
            assert quiet(continued.prop_solver.check_total_partition)
            assert full.random_bg.state == continued.random_bg.state
            for step in (3, 4):
                compare_files(paths["full"], paths["continued"], f"fields_{step:06d}.mat",
                              ("w_", "phi_", "normal_noise", "record_", "rng_state", "extension_state"))
            compare_files(paths["full"], paths["continued"], "dH_000004.mat",
                          ("H_", "dH_", "drift_", "chemical_potential_"))
            compare_files(paths["full"], paths["continued"], "structure_function_000004.mat", ("structure_function_",))
            # The accepted public run records H with every finite coefficient included.
            for name, sim in (("full", full), ("continued", continued)):
                sample = snapshots[name]
                h, exact, _ = energy(sim, sample["w_aux"])
                np.testing.assert_allclose(sample["hamiltonian"], exact, rtol=0, atol=1e-10)
                np.testing.assert_allclose(h, exact, rtol=0, atol=1e-10)


def main():
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_small_coefficients_") as directory:
        for platform in platforms:
            root = Path(directory) / platform
            quadratic_and_gradient(root, platform)
            small_linear_term(root, platform)
            uniform_saddle(root, platform)
            dynamics_and_restart(root, platform)
            print(f"[PASS] {platform}: small finite field coefficients, saddles and LM/Euler restarts")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
