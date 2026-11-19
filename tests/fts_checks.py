"""Checks and inputs shared by the Python tests of simulation runs.

quiet() calls a function without what it prints, and silenced() discards what a with block
prints. python_platform() picks the one platform that runs a test of a fix made in Python
(cpu_platform() of cuda_part).
compare_files() compares a file that two runs wrote key by key. exact_restart() checks that a
run continued from the checkpoint of a shorter run ends where the uninterrupted run ends, bit for
bit on the same backend. expect_error() expects an exception. The inputs that several tests run:
diblock_parameters() and diblock_fields() (a 1D L-FTS or CL-FTS diblock melt), FailingLFTS (its
saddle search fails at chosen calls), compressible_parameters() (a 4-point diblock with zeta_n)
and scft_parameters() (an SCFT diblock on 8^3 points). The Langevin inputs are those of L-FTS,
with the saddle and compressor entries, of CL-FTS (cls=CLFTS), which would ignore them with a
warning, or the system part for SCFT (cls=SCFT), which would ignore the Langevin options (input_of).

The L-FTS and CL-FTS runs of a test that imports this module draw no chain architectures
(polymer_<id>.png): matplotlib takes about 0.1 s per run, more than a short run itself. A test of
the drawings sets draw_architectures = True on its simulation.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from polymerfts import SCFT, LFTS, CLFTS
from cuda_part import cpu_platform

LFTS.draw_architectures = CLFTS.draw_architectures = False


@contextlib.contextmanager
def silenced():
    """A with block whose printed output is discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def quiet(function, *args, **kwargs):
    """function(*args, **kwargs) with its printed output discarded."""
    with silenced():
        return function(*args, **kwargs)


# The one platform that runs a test of a fix made in Python
python_platform = cpu_platform


def expect_error(call, kind):
    """call() raises an exception of type kind."""
    try:
        call()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def compare_files(left, right, filename, prefixes, exclude=(), rtol=None):
    """The MAT files filename of the directories left and right hold the same keys starting
    with prefixes ("" takes every key) except those in exclude, and their values are equal bit
    for bit. With rtol, only the float and complex arrays are compared: they have the same
    shapes and differ by at most rtol times the largest magnitude of each array. The entries
    that loadmat adds (__header__, __version__, __globals__) are left out."""
    a, b = (loadmat(Path(directory) / filename, squeeze_me=True) for directory in (left, right))
    keys, other = ({key for key, value in data.items() if key.startswith(prefixes) and not key.startswith("__")
                    and key not in exclude
                    and (rtol is None or isinstance(value, np.ndarray) and value.dtype.kind in "fc")}
                   for data in (a, b))
    assert keys, f"{filename}: no key starts with {prefixes}"
    assert keys == other, f"{filename}: keys in one file only: {sorted(keys ^ other)}"
    for key in sorted(keys):
        if rtol is None:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")
            continue
        assert a[key].shape == b[key].shape, f"{filename}: {key} shapes {a[key].shape} {b[key].shape}"
        scale = np.max(np.abs(a[key]), initial=0.0)
        difference = np.max(np.abs(a[key] - b[key]), initial=0.0)
        assert difference <= rtol * scale, f"{filename}: {key} differs by {difference:.2e} (scale {scale:.2e})"


def recording_dir(sim):
    return Path(sim.recording["dir"])


def exact_restart(make, start, steps, split, files, inspect=None, exclude=(), same_directory=False):
    """An uninterrupted run against a run continued from the checkpoint of a shorter one.

    make(name, steps, seed) builds a simulation that records into its own directory: "full"
    runs all steps and "partial" the first split steps, both with seed 17, and "continued",
    built with seed 99, continues from the checkpoint of the partial run at step split; with
    same_directory it records into the directory of the partial run, where that run stopped
    (it is built as make("partial", steps, 99)). start(sim) begins a run and returns its status.
    inspect(checkpoint), if given, looks at that checkpoint before the continuation. The
    continued run must complete all steps with the random state of the full run, and each file
    of files ({filename: key prefixes}) of the two must agree bit for bit, but for the keys in
    exclude (compare_files). Returns the three simulations by name.
    """
    sims = {name: make("partial" if same_directory and name == "continued" else name, n, seed)
            for name, n, seed in (("full", steps, 17), ("partial", split, 17), ("continued", steps, 99))}
    for name in ("full", "partial"):
        status = quiet(start, sims[name])
        assert status["status"] == "completed", (name, status)
        assert quiet(sims[name].prop_solver.check_total_partition), name
    checkpoint = recording_dir(sims["partial"]) / f"fields_{split:06d}.mat"
    if inspect is not None:
        inspect(checkpoint)
    status = quiet(sims["continued"].continue_run, checkpoint)
    assert status["status"] == "completed" and status["last_completed_step"] == steps, status
    assert quiet(sims["continued"].prop_solver.check_total_partition)
    assert sims["continued"].random_bg.state == sims["full"].random_bg.state
    for filename, prefixes in files.items():
        compare_files(recording_dir(sims["full"]), recording_dir(sims["continued"]), filename, prefixes, exclude)
    return sims


def input_of(cls, params):
    """params for the simulation class cls, without the options of the other methods, which it would
    ignore with a warning: CL-FTS reads no saddle and compressor entries, SCFT neither them nor the
    Langevin options (an SCFT input also needs the optimizer and relax_cell)."""
    if issubclass(cls, (CLFTS, SCFT)):
        del params["compressor"], params["saddle"]
    if issubclass(cls, SCFT):
        del params["langevin"], params["recording"], params["verbose_level"]
    return params


def diblock_parameters(directory, steps=4, cls=LFTS):
    """An AB diblock melt in 1D (16 points), continuous chains, for L-FTS (the compressor and
    saddle entries), CL-FTS (cls=CLFTS) or SCFT (cls=SCFT, without the Langevin options), on
    cpu_platform(), recording into directory."""
    return input_of(cls, {
        "nx": [16], "lx": [3.0], "ds": 0.05, "chain_model": "continuous",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": cpu_platform(),
        "langevin": {"nbar": 10000, "dt": 1.0, "max_step": steps},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 200, "tolerance": 1e-6},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1, "statistics_save_interval": 2},
        "verbose_level": 0,
    })


def diblock_fields():
    """Random fields of standard deviation 5 for diblock_parameters."""
    rng = np.random.default_rng(3)
    return {m: rng.normal(0.0, 5.0, 16) for m in ("A", "B")}


class FailingLFTS(LFTS):
    """An L-FTS whose saddle search reports an infinite error at the calls in fail_calls."""
    fail_calls = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        solve, self.calls = self.saddle.solve, 0

        def failing_solve(w_aux):
            self.calls += 1
            phi, phi_smeared, h, iterations, error = solve(w_aux)
            return phi, phi_smeared, h, iterations, np.inf if self.calls in self.fail_calls else error
        self.saddle.solve = failing_solve


def compressible_parameters(directory, platform, zeta=1e8, method="lm", steps=4, cls=LFTS):
    """An AB diblock melt on 4 points with compressibility zeta_n, for L-FTS, CL-FTS (cls=CLFTS) or SCFT
    (cls=SCFT, without the Langevin options)."""
    return input_of(cls, {
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
    })


def scft_parameters():
    """An AB diblock melt for SCFT on 8^3 points in a fixed cell, on cpu_platform()."""
    return {
        "nx": [8, 8, 8],
        "lx": [4.0, 4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/10,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
        "platform": cpu_platform(),
        "relax_cell": False,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }
