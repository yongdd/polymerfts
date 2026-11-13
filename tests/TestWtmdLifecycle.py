"""WTMD state across fresh runs, manual continuations and checkpoint restarts."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import LFTS, PlatformSelector
from polymerfts.fts.wtmd import WTMD
from polymerfts.utils.checkpoint import validate_checkpoint
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def state(wtmd):
    return {"u": wtmd.u.copy(), "up": wtmd.up.copy(), "I0": wtmd.I0.copy(),
            "I1": {key: value.copy() for key, value in wtmd.I1.items()},
            "samples": list(wtmd.order_parameter_history),
            "dH": {key: list(values) for key, values in wtmd.dH_history.items()}}


def assert_state(wtmd, expected):
    for name in ("u", "up", "I0"):
        np.testing.assert_array_equal(getattr(wtmd, name), expected[name])
    assert wtmd.I1.keys() == expected["I1"].keys()
    for key in wtmd.I1:
        np.testing.assert_array_equal(wtmd.I1[key], expected["I1"][key])
    assert wtmd.order_parameter_history == expected["samples"]
    assert wtmd.dH_history == expected["dH"]


def constructor_reset():
    inputs = {"u": np.linspace(0.01, 0.07, 20), "up": np.linspace(-0.05, 0.02, 20),
              "I0": np.linspace(1.0, 2.0, 20), "I1": {"A,B": np.linspace(0.1, 0.9, 20)}}
    originals = copy.deepcopy(inputs)
    wtmd = quiet(WTMD, [4, 4, 4], [1.3, 1.7, 2.1], 17.0, np.array([-15.0, 0.0]), [0],
                 psi_max=2.0, dpsi=0.1, **inputs)
    expected = state(wtmd)
    for name in ("u", "up", "I0"):
        np.testing.assert_array_equal(inputs[name], originals[name])
        np.testing.assert_array_equal(expected[name], inputs[name] / (1 if name == "I0" else wtmd.CV))
        inputs[name][:] = -99.0   # caller-owned arrays must not change the initial snapshot
    np.testing.assert_array_equal(inputs["I1"]["A,B"], originals["I1"]["A,B"])
    np.testing.assert_array_equal(expected["I1"]["A,B"], inputs["I1"]["A,B"] / wtmd.CV)
    inputs["I1"]["A,B"][:] = -99.0
    for _ in range(2):
        wtmd.u[:] = 7.0
        wtmd.up[:] = 8.0
        wtmd.I0[:] = 9.0
        wtmd.I1["A,B"][:] = 10.0
        wtmd.I1["another pair"] = np.ones(wtmd.bins)
        wtmd.order_parameter_history.append(0.4)
        wtmd.dH_history["A,B"] = [0.5]
        wtmd.reset()
        assert_state(wtmd, expected)


def parameters(directory, platform, steps=2):
    return {"nx": [4, 4, 4], "lx": [2.0, 2.0, 2.0], "ds": 0.05, "chain_model": "continuous",
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
            "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                {"type": "B", "length": 0.5}]}],
            "platform": platform, "langevin": {"nbar": 10000, "dt": 0.2, "max_step": steps},
            "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
            "saddle": {"max_iter": 200, "tolerance": 1e-6}, "verbose_level": 0,
            "recording": {"dir": str(directory), "checkpoint_interval": 1,
                          "sample_interval": 1, "statistics_save_interval": 2},
            "wtmd": {"update_freq": 3, "recording_period": 100, "dpsi": 0.1}}


def fields():
    random = np.random.default_rng(11)
    return {name: random.normal(0.0, 5.0, 64) for name in ("A", "B")}


def checkpoint(sim, step):
    return Path(sim.recording["dir"]) / f"fields_{step:06d}.mat"


def compare_checkpoints(left, right):
    a, b = (loadmat(path, squeeze_me=True) for path in (left, right))
    keys = [key for key in a if key.startswith(("w_", "phi_", "normal_noise", "record_", "rng_state", "wtmd_"))
            and key != "wtmd_params"]
    assert keys
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def repeated_runs(root, platform):
    sim = quiet(LFTS, parameters(root / "reused", platform), random_seed=1)
    quiet(sim.run, fields())
    assert len(sim.wtmd.order_parameter_history) == 2
    for first_step in (None, 1):
        fresh = quiet(LFTS, parameters(root / f"fresh_{first_step}", platform), random_seed=99)
        # Fresh runs keep the current noise stream. Align the stream, not just the seed.
        fresh.random_bg.state = copy.deepcopy(sim.random_bg.state)
        quiet(sim.run, fields(), start_langevin_step=first_step)
        quiet(fresh.run, fields())
        assert len(sim.wtmd.order_parameter_history) == 2
        assert_state(sim.wtmd, state(fresh.wtmd))
        assert sim.random_bg.state == fresh.random_bg.state
        for step in (1, 2):
            data = loadmat(checkpoint(sim, step), squeeze_me=True)
            validate_checkpoint(sim, data)
            assert np.size(data["wtmd_order_parameter_history"]) == step
            compare_checkpoints(checkpoint(sim, step), checkpoint(fresh, step))
    return sim


def manual_continuation(sim):
    data = loadmat(checkpoint(sim, 2), squeeze_me=True)
    before = state(sim.wtmd)
    update_statistics = sim.wtmd.update_statistics
    pending_at_update = []

    def update():
        pending_at_update.append(len(sim.wtmd.order_parameter_history))
        for name in ("u", "up", "I0"):
            np.testing.assert_array_equal(getattr(sim.wtmd, name), before[name])
        update_statistics()

    sim.langevin["max_step"] = 3
    with patch.object(sim.wtmd, "update_statistics", side_effect=update):
        quiet(sim.run, {name: data["w_" + name] for name in sim.monomer_types},
              normal_noise_prev=data["normal_noise_prev"], start_langevin_step=3)
    assert pending_at_update == [3], "The hill must include both pending samples of the preceding run"
    assert not sim.wtmd.order_parameter_history
    assert not sim.wtmd.dH_history
    assert np.any(sim.wtmd.u != before["u"])
    # A manual continuation with no remaining steps must also keep the accumulated bias.
    before = state(sim.wtmd)
    rng = copy.deepcopy(sim.random_bg.state)
    data = loadmat(checkpoint(sim, 3), squeeze_me=True)
    quiet(sim.run, {name: data["w_" + name] for name in sim.monomer_types},
          normal_noise_prev=data["normal_noise_prev"], start_langevin_step=4)
    assert_state(sim.wtmd, before)
    assert sim.random_bg.state == rng


def checkpoint_restart(root, platform):
    full = quiet(LFTS, parameters(root / "full", platform, 4), random_seed=13)
    quiet(full.run, fields())
    partial = quiet(LFTS, parameters(root / "partial", platform, 2), random_seed=13)
    quiet(partial.run, fields())
    continued = quiet(LFTS, parameters(root / "continued", platform, 4), random_seed=99)
    quiet(continued.run, fields())   # the recipient already has a bias and pending samples
    quiet(continued.continue_run, checkpoint(partial, 2))
    assert_state(continued.wtmd, state(full.wtmd))
    assert continued.random_bg.state == full.random_bg.state
    compare_checkpoints(checkpoint(continued, 4), checkpoint(full, 4))

    # Restoring a checkpoint must not replace the constructor's initial state.
    fresh = quiet(LFTS, parameters(root / "fresh_after_restart", platform, 4), random_seed=99)
    fresh.random_bg.state = copy.deepcopy(continued.random_bg.state)
    quiet(continued.run, fields())
    quiet(fresh.run, fields())
    assert_state(continued.wtmd, state(fresh.wtmd))
    assert continued.random_bg.state == fresh.random_bg.state
    compare_checkpoints(checkpoint(continued, 4), checkpoint(fresh, 4))


def main():
    constructor_reset()
    print("[PASS] WTMD constructor state and reset copies")
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_wtmd_lifecycle_") as directory:
        root = Path(directory)
        for platform in platforms:
            sim = repeated_runs(root / platform, platform)
            manual_continuation(sim)
            checkpoint_restart(root / platform, platform)
            print(f"[PASS] {platform}: WTMD fresh runs, manual continuation and checkpoint restart")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
