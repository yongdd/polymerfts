#!/usr/bin/env python3
"""WTMD norms and derivatives stay finite for large ell and extreme field scales.

Compare against a logarithmic norm and directional finite differences, then run
actual LM/Euler trajectories with a deposited bias and pending checkpoint data.
All available backends exercise the norm, bias and exact continuation.
"""
import contextlib
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp

from polymerfts import LFTS, PlatformSelector, PropagatorSolver
from polymerfts.fts.wtmd import WTMD
from polymerfts.utils.checkpoint import validate_checkpoint
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def logarithmic_norm(wtmd, field):
    """Independent log-sum-exp oracle, with NumPy's FFT and half-spectrum weights."""
    magnitude = np.abs(np.fft.rfftn(np.reshape(field, wtmd.nx)))
    nonzero = (magnitude > 0) & (wtmd.fk > 0)
    if not np.any(nonzero):
        return 0.0
    terms = (wtmd.l * np.log(magnitude[nonzero]) + np.log(wtmd.fk[nonzero])
             + np.log(wtmd.wt[nonzero]))
    return np.exp(logsumexp(terms) / wtmd.l - np.log(wtmd.M))


class ExactUniformTransform:
    """Supply exact uniform coefficients; still use the platform's inverse FFT."""
    def __init__(self, nx, fourier):
        self.nx, self.fourier = tuple(nx), fourier

    def forward(self, field):
        field = np.asarray(field)
        assert np.all(field == field.flat[0])
        coefficients = np.zeros(self.nx[:-1] + (self.nx[-1] // 2 + 1,), dtype=np.complex128)
        coefficients[(0,) * len(self.nx)] = field.size * field.flat[0]
        return coefficients

    def backward(self, coefficients):
        if self.fourier is not None:
            return self.fourier.backward(coefficients)
        return np.fft.irfftn(coefficients, s=self.nx, axes=tuple(range(len(self.nx)))).ravel()


def norm_and_gradient(fourier=None):
    nx, lx = [4, 4, 5], [3.0, 3.0, 3.0]
    size = int(np.prod(nx))
    base = np.random.default_rng(3).normal(0.0, 5.0, size)
    direction = np.random.default_rng(9).normal(size=size)
    for ell in (1, 1.5, 4, 32, 256):
        wtmd = WTMD(nx, lx, 10000.0, np.array([-15.0, 0.0]), [0], l=ell, fourier=fourier)
        wtmd.up[:] = 0.7
        baseline_psi, baseline_force = None, None
        for scale in (1.0, 1e-100, 1e100):
            fields = np.stack([scale * base, np.zeros(size)])
            force = np.zeros((1, size))
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                psi = wtmd.compute_order_parameter(1, fields)
                quiet(wtmd.add_bias_to_langevin, psi, force)
            assert psi > 0 and np.isfinite(psi) and np.all(np.isfinite(force))
            np.testing.assert_allclose(psi, logarithmic_norm(wtmd, fields[0]), rtol=3e-13, atol=0)
            if baseline_psi is None:
                baseline_psi, baseline_force = psi, force.copy()
            else:
                np.testing.assert_allclose(psi / scale, baseline_psi, rtol=3e-13, atol=0)
                np.testing.assert_allclose(force, baseline_force, rtol=5e-12, atol=1e-13)

        # The force uses the continuum functional derivative: its grid mean
        # against a direction equals U'(psi) times the directional derivative.
        delta = 1e-5
        finite_difference = (logarithmic_norm(wtmd, base + delta * direction)
                             - logarithmic_norm(wtmd, base - delta * direction)) / (2 * delta)
        np.testing.assert_allclose(np.mean(baseline_force[0] * direction) / 0.7,
                                   finite_difference, rtol=3e-6, atol=1e-9)

        exact = WTMD(nx, lx, 10000.0, np.array([-15.0, 0.0]), [0], l=ell,
                     fourier=ExactUniformTransform(nx, fourier))
        exact.up[:] = 0.7
        for amplitude in (0.0, 2.0):
            fields = np.stack([np.full(size, amplitude), np.zeros(size)])
            force = np.zeros((1, size))
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                psi = wtmd.compute_order_parameter(1, fields)
                quiet(wtmd.add_bias_to_langevin, psi, force)
            np.testing.assert_allclose(psi, logarithmic_norm(wtmd, fields[0]), rtol=3e-13, atol=0)
            # An odd-length FFT can leave tiny nonzero modes of a constant
            # input. At ell=1 their phase has an O(1) derivative, and for
            # ell=1.5 it decays only as sqrt(magnitude). Compare the actual
            # coefficients against the original derivative with NumPy's IFFT.
            magnitude = np.abs(wtmd.w_aux_k)
            phase = np.divide(wtmd.w_aux_k, magnitude, out=np.zeros_like(wtmd.w_aux_k), where=magnitude > 0)
            if psi == 0:
                expected = np.zeros(size)
            else:
                coefficient = (magnitude / size / psi) ** (ell - 1.0) * phase * wtmd.fk
                expected = 0.7 * size * np.fft.irfftn(coefficient, s=nx, axes=(0, 1, 2)).ravel()
            np.testing.assert_allclose(force[0], expected, rtol=2e-12, atol=1e-14)

            # Exact zero coefficients separately exercise the zero subgradient
            # and analytic uniform derivative, without assuming FFT roundoff.
            force[:] = 0
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                psi = exact.compute_order_parameter(1, fields)
                quiet(exact.add_bias_to_langevin, psi, force)
            assert np.count_nonzero(exact.w_aux_k) == int(amplitude != 0)
            expected = 0.0 if amplitude == 0 else 0.7 * wtmd.fk[0, 0, 0] ** (1.0 / ell)
            np.testing.assert_allclose(force, expected, rtol=2e-12, atol=1e-14)


def parameters(directory, platform, ell, method, steps):
    return {"nx": [4, 4, 4], "lx": [3.0] * 3, "ds": 0.25, "chain_model": "continuous",
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
            "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                  {"type": "B", "length": 0.5}]}],
            "platform": platform,
            "langevin": {"nbar": 10000.0, "dt": 0.01, "max_step": steps, "integrator": method},
            "saddle": {"max_iter": 200, "tolerance": 1e-7},
            "compressor": {"name": "am", "max_hist": 20, "start_error": 0.1,
                           "mix_min": 0.1, "mix_init": 0.1},
            "wtmd": {"ell": ell, "update_freq": 3, "recording_period": 3, "dpsi": 0.1},
            "recording": {"dir": str(directory), "checkpoint_interval": 1,
                          "sample_interval": 1, "statistics_save_interval": 6}, "verbose_level": 0}


class Observations:
    def __init__(self, tolerance):
        self.settings = {"saddle_tolerance": tolerance}
        self.steps = []

    def __call__(self, sample):
        assert np.isfinite(sample["hamiltonian"])
        assert np.all(np.isfinite(sample["w_aux"]))
        error = float(sample["error_level"])
        assert np.isfinite(error) and error < self.settings["saddle_tolerance"]
        assert sample["status"]["noise_resamples"] == 0
        np.testing.assert_allclose(np.mean(sum(sample["phi"].values())), 1.0, rtol=0, atol=5e-8)
        self.steps.append(int(sample["step"]))

    def state_dict(self):
        return {"steps": list(self.steps)}

    def load_state_dict(self, state):
        self.steps = list(state["steps"])


def compare_files(left, right, filename, prefixes):
    a, b = (loadmat(path / filename, squeeze_me=True) for path in (left, right))
    keys = [key for key in a if key.startswith(prefixes) and key != "wtmd_params"]
    assert keys, filename
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


def trajectory_and_restart(root, platform, ell, method):
    paths = {name: root / name for name in ("full", "partial", "continued")}
    simulations, observers, bias_changes = {}, {}, {}

    def create(name, steps, seed):
        params = parameters(paths[name], platform, ell, method, steps)
        observer = Observations(params["saddle"]["tolerance"])
        sim = quiet(LFTS, params, random_seed=seed,
                    callbacks={"observations": observer})
        bias = sim.wtmd.add_bias_to_langevin
        changes = []

        def record_bias(psi, force):
            assert np.isfinite(psi)
            before = force.copy()
            bias(psi, force)
            assert np.all(np.isfinite(force))
            changes.append(float(np.max(np.abs(force - before))))

        sim.wtmd.add_bias_to_langevin = record_bias
        simulations[name], observers[name], bias_changes[name] = sim, observer, changes
        return sim

    full, partial, continued = (create(name, steps, seed) for name, steps, seed in
                                (("full", 6, 17), ("partial", 4, 17), ("continued", 6, 99)))
    initial = dict(zip(("A", "B"), np.random.default_rng(3).normal(0.0, 5.0, (2, 64))))
    for sim in (full, partial):
        status = quiet(sim.run, initial)
        # Retrying the initial saddle at the same fields is allowed; accepted
        # states must converge, and no failed attempt may select another noise.
        assert status["status"] == "completed" and status["noise_resamples"] == 0
        assert status["last_completed_step"] == sim.langevin["max_step"]
        assert quiet(sim.prop_solver.check_total_partition)
    checkpoint = paths["partial"] / "fields_000004.mat"
    pending = loadmat(checkpoint, squeeze_me=True)
    validate_checkpoint(continued, pending)
    assert pending["record_sample_count"] == 4
    assert np.size(pending["record_H_history"]) == 4
    assert np.size(pending["wtmd_order_parameter_history"]) == 1
    assert np.all(np.isfinite(pending["wtmd_order_parameter_history"]))
    assert np.any(pending["wtmd_unscaled_up"] != 0)
    assert max(bias_changes["partial"]) > 0.0, "The split must occur after a nonzero bias was applied"
    status = quiet(continued.continue_run, checkpoint)
    assert status["status"] == "completed" and status["noise_resamples"] == 0
    assert status["last_completed_step"] == 6
    assert status["saddle_failures"] == full.run_status["saddle_failures"]
    assert quiet(continued.prop_solver.check_total_partition)
    assert max(bias_changes["continued"]) > 0.0
    assert full.random_bg.state == continued.random_bg.state
    assert observers["full"].steps == observers["continued"].steps == list(range(1, 7))
    for step in (5, 6):
        data = loadmat(paths["continued"] / f"fields_{step:06d}.mat", squeeze_me=True)
        validate_checkpoint(continued, data)
        compare_files(paths["full"], paths["continued"], f"fields_{step:06d}.mat",
                      ("w_", "phi_", "normal_noise", "record_", "rng_state", "run_status", "extension_state", "wtmd_"))
    compare_files(paths["full"], paths["continued"], "dH_000006.mat", ("H_", "dH_", "drift_"))
    compare_files(paths["full"], paths["continued"], "structure_function_000006.mat", ("structure_function_",))
    compare_files(paths["full"], paths["continued"], "wtmd_statistics_000006.mat", ("u", "I", "psi", "dH_psi_"))


def main():
    norm_and_gradient()
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_wtmd_range_") as directory:
        for platform in platforms:
            solver = quiet(PropagatorSolver, nx=[4, 4, 5], lx=[3.0] * 3, ds=0.25,
                           bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                           chain_model="continuous", platform=platform)
            norm_and_gradient(solver.fourier)
            for ell in (4, 32, 256):
                for method in ("lm", "euler"):
                    trajectory_and_restart(Path(directory) / platform / f"{ell}_{method}", platform, ell, method)
            print(f"[PASS] {platform}: WTMD range, gradients, nonzero bias and LM/Euler restarts")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
