#!/usr/bin/env python3
"""Checkpoints written by main continue on this version.

tests/data/old_checkpoints holds the step-2 checkpoints (format 2) of an L-FTS and a CL-FTS run of four
steps in 1D (16 points), and of an L-FTS run of discrete chains, whose method main recorded as
"discrete" (now "ps"), written on cpu-fftw by main at 895d7e75 (package version 0.10.0) from the
fields and seed below. They carry that version's settings, random state, previous noise and the pending
samples of steps 1 and 2 (under its hashed record keys). Continued here to step 4, each must end where an
uninterrupted run of this version from the same fields and seed ends: with the same random state, the
samples of all four steps, and the same fields, noise and records within round-off (CL-FTS, 1e-10
relative) or within 100 times the saddle tolerance (L-FTS, whose saddle search may stop an iteration
earlier or later). On another machine, compiler or backend the uninterrupted run differs from the one
that wrote the fixtures at round-off, so equality is not required to be bitwise (on cpu-fftw of the
machine that wrote them both methods agree bit for bit, on cpu-mkl to 5e-14). The checks are Python: the
first CPU platform runs them.
"""

import os
from pathlib import Path
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, LFTS
from fts_checks import python_platform, quiet

DATA = Path(__file__).resolve().parent / "data" / "old_checkpoints"
SEED = 11
TOLERANCE = 1e-8   # the saddle tolerance of L-FTS


def parameters(cls, directory, platform, chain_model="continuous"):
    """The input of both versions; the recording directory and the platform may change."""
    params = {
        "nx": [16], "lx": [3.0], "ds": 0.05, "chain_model": chain_model, "platform": platform,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                         {"type": "B", "length": 0.5}]}],
        "langevin": {"nbar": 10000, "dt": 1.0 if cls is LFTS else 0.5, "max_step": 4},
        "recording": {"dir": str(directory), "checkpoint_interval": 2, "sample_interval": 1,
                      "statistics_save_interval": 4},
        "verbose_level": 0,
    }
    if cls is LFTS:
        params["compressor"] = {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01}
        params["saddle"] = {"max_iter": 200, "tolerance": TOLERANCE}
    return params


def initial_fields():
    x = (np.arange(16) + 0.5) / 16
    w = 5.0 * np.sin(2 * np.pi * x) + 2.0 * np.cos(6 * np.pi * x)
    return {"A": w, "B": -w}


def close_files(left, right, filename, rtol):
    """The float and complex arrays of the files filename of two directories have the same keys and
    shapes, and differ by at most rtol times the largest magnitude of each array."""
    a, b = ({key: value for key, value in loadmat(Path(directory) / filename, squeeze_me=True).items()
             if isinstance(value, np.ndarray) and value.dtype.kind in "fc"} for directory in (left, right))
    assert a and a.keys() == b.keys(), f"{filename}: keys in one file only: {sorted(a.keys() ^ b.keys())}"
    for key in sorted(a):
        assert a[key].shape == b[key].shape, f"{filename}: {key} shapes {a[key].shape} {b[key].shape}"
        scale = np.max(np.abs(a[key]), initial=0.0)
        difference = np.max(np.abs(a[key] - b[key]), initial=0.0)
        assert difference <= rtol * scale, f"{filename}: {key} differs by {difference:.2e} (scale {scale:.2e})"


def check(cls, platform, root, chain_model="continuous"):
    name = cls.__name__.lower() + ("_discrete" if chain_model == "discrete" else "")
    full = quiet(cls, parameters(cls, root / f"{name}_full", platform, chain_model), random_seed=SEED)
    assert quiet(full.run, initial_fields())["status"] == "completed"
    continued = quiet(cls, parameters(cls, root / f"{name}_continued", platform, chain_model), random_seed=99)
    status = quiet(continued.continue_run, DATA / f"{name}_fields_000002.mat")
    assert status["status"] == "completed" and status["last_completed_step"] == 4, status
    assert quiet(continued.prop_solver.check_total_partition)
    assert continued.random_bg.state == full.random_bg.state
    rtol = 1e-10 if cls is CLFTS else 100 * TOLERANCE
    for filename in ("fields_000004.mat", "dH_000004.mat", "structure_function_000004.mat"):
        close_files(full.recording["dir"], continued.recording["dir"], filename, rtol)
    samples = loadmat(Path(continued.recording["dir"]) / "dH_000004.mat", squeeze_me=True)["sample_count"]
    assert samples == 4, f"the continued run recorded {samples} samples: the old checkpoint's pending ones are lost"
    print(f"[PASS] {platform}: {cls.__name__} ({chain_model}) continues the step-2 checkpoint of main to step 4")


def main():
    platform = python_platform()
    with tempfile.TemporaryDirectory(prefix="polymerfts_old_checkpoints_") as directory:
        for cls in (LFTS, CLFTS):
            check(cls, platform, Path(directory))
        check(LFTS, platform, Path(directory), "discrete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
