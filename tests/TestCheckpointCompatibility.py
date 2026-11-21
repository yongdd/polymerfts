#!/usr/bin/env python3
"""Checkpoints written by main and by release 0.10.0 continue on this version.

tests/data/old_checkpoints holds the step-2 checkpoints (format 2) of an L-FTS and a CL-FTS run of four
steps in 1D (16 points), and of an L-FTS run of discrete chains, whose method main recorded as
"discrete" (now "ps"), written on cpu-fftw by main at 895d7e75 from the fields and seed below. They
carry that version's settings, random state, previous noise and the pending samples of steps 1 and 2
(under its hashed record keys). Its package version reads 0.10.0, but it does not descend from release
0.10.0 (5310fadc). clfts_v0100_fields_000002.mat is the step-2 checkpoint of the same CL-FTS run
written on cpu-fftw by release 0.10.0, which did not record the chemical potentials, the drift of every
step (record_drift_step_history, record_drift_step_start) and the entry "electrostatics" of the
settings. Continued here to step 4, each must end where an uninterrupted run of this version from the
same fields and seed ends: with the same random state, the samples of all four steps, and the same
fields, noise and records within round-off (CL-FTS, 1e-10 relative) or within 100 times the saddle
tolerance (L-FTS, whose saddle search may stop an iteration earlier or later). Continued from the
release, the chemical potentials of the samples of its checkpoint (steps 1 and 2) are NaN and those
of steps 3 and 4 are the uninterrupted run's, and the drifts of every step are recorded from the
restart (step 3) on. On another machine, compiler or backend the uninterrupted run differs from the
ones that wrote the fixtures at round-off, so equality is not required to be bitwise (on cpu-fftw of
the machine that wrote them the files of main agree bit for bit and that of the release to 2e-16, in
the structure functions whose sums it rounded differently; on cpu-mkl all agree to 5e-14). The checks
are Python: the first CPU platform runs them.
"""

import os
from pathlib import Path
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, LFTS
from fts_checks import compare_files, cpu_platform, quiet

DATA = Path(__file__).resolve().parent / "data" / "old_checkpoints"
SEED = 11
TOLERANCE = 1e-8   # the saddle tolerance of L-FTS
# The CL-FTS records of main that release 0.10.0 did not write into its checkpoints
NOT_IN_RELEASE = ("record_chemical_potential_history", "record_drift_step_history", "record_drift_step_start")
# The entries of the dH file that a run continued from the release fills otherwise (release_records)
RELEASE_GAPS = ("chemical_potential_history_real", "chemical_potential_history_imag", "drift_max_step_history")


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


def close(expected, actual, rtol):
    """Same shapes, and a difference of at most rtol times the largest magnitude of expected."""
    return (expected.shape == actual.shape and
            np.max(np.abs(expected - actual), initial=0.0) <= rtol * np.max(np.abs(expected), initial=0.0))


def release_records(full, continued, rtol):
    """The records of the dH files at step 4 that a checkpoint of release 0.10.0 lacks: continued from
    it, the chemical potentials of steps 1 and 2 are NaN and those of steps 3 and 4 the uninterrupted
    run's, and the drifts of every step are those of steps 3 and 4."""
    a, b = (loadmat(Path(directory) / "dH_000004.mat", squeeze_me=True) for directory in (full, continued))
    mu_full, mu = (data["chemical_potential_history_real"] + 1j * data["chemical_potential_history_imag"]
                   for data in (a, b))
    assert mu.shape == mu_full.shape and np.all(np.isnan(mu[:2])), f"chemical potentials of steps 1 and 2: {mu[:2]}"
    assert close(mu_full[2:], mu[2:], rtol), f"chemical potentials of steps 3 and 4: {mu[2:]}, not {mu_full[2:]}"
    drifts_full, drifts = a["drift_max_step_history"], b["drift_max_step_history"]
    assert close(drifts_full[2:], drifts, rtol), f"drifts of every step: {drifts}, uninterrupted {drifts_full}"


def check(cls, platform, root, chain_model="continuous", release=False):
    """Continues the step-2 checkpoint of main, with release that of release 0.10.0 (CL-FTS)."""
    name = cls.__name__.lower() + ("_discrete" if chain_model == "discrete" else "") + ("_v0100" if release else "")
    checkpoint = DATA / f"{name}_fields_000002.mat"
    if release:
        held = sorted(set(NOT_IN_RELEASE) & set(loadmat(checkpoint)))
        assert not held, f"{checkpoint.name} holds records that release 0.10.0 did not write: {held}"
    full = quiet(cls, parameters(cls, root / f"{name}_full", platform, chain_model), random_seed=SEED)
    assert quiet(full.run, initial_fields())["status"] == "completed"
    continued = quiet(cls, parameters(cls, root / f"{name}_continued", platform, chain_model), random_seed=99)
    status = quiet(continued.continue_run, checkpoint)
    assert status["status"] == "completed" and status["last_completed_step"] == 4, status
    assert quiet(continued.prop_solver.check_total_partition)
    assert continued.random_bg.state == full.random_bg.state
    rtol = 1e-10 if cls is CLFTS else 100 * TOLERANCE
    for filename in ("fields_000004.mat", "dH_000004.mat", "structure_function_000004.mat"):
        exclude = RELEASE_GAPS if release and filename == "dH_000004.mat" else ()
        compare_files(full.recording["dir"], continued.recording["dir"], filename, "", exclude=exclude, rtol=rtol)
    if release:
        release_records(full.recording["dir"], continued.recording["dir"], rtol)
    samples = loadmat(Path(continued.recording["dir"]) / "dH_000004.mat", squeeze_me=True)["sample_count"]
    assert samples == 4, f"the continued run recorded {samples} samples: the old checkpoint's pending ones are lost"
    writer = "release 0.10.0" if release else "main"
    print(f"[PASS] {platform}: {cls.__name__} ({chain_model}) continues the step-2 checkpoint of {writer} to step 4")


def main():
    platform = cpu_platform()
    with tempfile.TemporaryDirectory(prefix="polymerfts_old_checkpoints_") as directory:
        for cls in (LFTS, CLFTS):
            check(cls, platform, Path(directory))
        check(LFTS, platform, Path(directory), "discrete")
        check(CLFTS, platform, Path(directory), release=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
