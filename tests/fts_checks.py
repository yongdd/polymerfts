"""Checks shared by the Python tests of simulation runs.

quiet() calls a function without what it prints. python_platform() picks the one platform
that runs a test of a fix made in Python. compare_files() compares a file that two runs wrote
key by key. exact_restart() checks that a run continued from the checkpoint of a shorter run
ends where the uninterrupted run ends, bit for bit on the same backend.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    """function(*args, **kwargs) with its printed output discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def python_platform():
    """The one platform that runs a test of a fix made in Python: the first CPU platform of
    the build, else CUDA. Without a platform the test ends: it fails (exit 1) in a build with
    the CUDA backend but no visible device (cuda_part_missing), else it is skipped (exit 77)."""
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        raise SystemExit(int(cuda_part_missing()) or 77)
    return next((name for name in platforms if name.startswith("cpu")), platforms[0])


def compare_files(left, right, filename, prefixes, exclude=()):
    """The MAT files filename of the directories left and right hold the same keys starting
    with prefixes ("" takes every key) except those in exclude, and their values are equal bit
    for bit. The entries that loadmat adds (__header__, __version__, __globals__) are left out."""
    a, b = (loadmat(Path(directory) / filename, squeeze_me=True) for directory in (left, right))
    keys, other = ({key for key in data if key.startswith(prefixes) and not key.startswith("__")
                    and key not in exclude} for data in (a, b))
    assert keys, f"{filename}: no key starts with {prefixes}"
    assert keys == other, f"{filename}: keys in one file only: {sorted(keys ^ other)}"
    for key in sorted(keys):
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


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
