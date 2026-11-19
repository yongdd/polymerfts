"""Reject corrupt checkpoints before mutating RNG, extension or WTMD state."""
import copy
import json
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, PlatformSelector, ValidationError
from polymerfts.utils.checkpoint import extension_state
from polymerfts.utils.common import atomic_savemat, input_hash, settings_json
from fts_checks import diblock_fields, diblock_parameters, expect_error, silenced


class Counter:
    def __init__(self, count=0):
        self.count = count

    def __call__(self, sample):
        self.count += 1

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]


def read_checkpoint(path):
    return {k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")}


def check_corruption(root, cls):
    original_dir = root / (cls.__name__ + "original")
    original = cls(diblock_parameters(original_dir, 1, cls), random_seed=11, callbacks={"count": Counter()})
    original.run(diblock_fields())
    data = read_checkpoint(original_dir / "fields_000001.mat")
    counter = Counter(7)
    candidate = cls(diblock_parameters(root / (cls.__name__ + "candidate"), cls=cls), random_seed=99,
                    callbacks={"count": counter})
    monomer_key = "w_A" if cls is LFTS else "w_A_real"
    corruptions = [
        ("missing-monomer", lambda d: d.pop(monomer_key)),
        ("fractional-step", lambda d: d.update(langevin_step=1.5)),
        ("nan-drift", lambda d: d.update(record_drift_history=np.array([np.nan]))),
        ("missing-drift", lambda d: d.pop("record_drift_history")),
        ("negative-drift", lambda d: d.update(record_drift_history=np.array([-1.0]))),
        ("fractional-count", lambda d: d.update(record_sample_count=1.5)),
        ("invalid-metadata", lambda d: d.update(simulation_metadata_json="[0]")),
        ("invalid-status", lambda d: d.update(run_status_json="{")),
        ("invalid-counter", lambda d: d.update(run_status_json=json.dumps(
            json.loads(str(d["run_status_json"])) | {"saddle_failures": "invalid"}))),
    ]
    before_rng = copy.deepcopy(candidate.random_bg.state)
    before_extension = extension_state(candidate)
    before_metadata = copy.deepcopy(candidate.metadata)
    for label, corrupt in corruptions:
        bad = copy.deepcopy(data)
        corrupt(bad)
        path = root / (cls.__name__ + "-" + label + ".mat")
        atomic_savemat(path, bad)
        expect_error(lambda: candidate.continue_run(path), ValidationError)
        assert candidate.random_bg.state == before_rng, (cls.__name__, label)
        assert extension_state(candidate) == before_extension, (cls.__name__, label)
        assert candidate.metadata == before_metadata, (cls.__name__, label)

    # Complete RNG JSON is sufficient; legacy scalar RNG keys are redundant.
    data.pop("random_state_state")
    data.pop("random_state_inc")
    path = root / (cls.__name__ + "-json-rng.mat")
    atomic_savemat(path, data)
    assert candidate.continue_run(path)["status"] == "completed"
    assert counter.count == 4


def check_unread_section_keys(root):
    """A checkpoint of an earlier release whose saddle and compressor held keys that no release read (the
    section key table refuses them now) continues without them; a stored key that is read still has to
    match."""
    original_dir = root / "unread_original"
    LFTS(diblock_parameters(original_dir, 1, LFTS), random_seed=11).run(diblock_fields())
    data = read_checkpoint(original_dir / "fields_000001.mat")

    def with_stored(change):
        stored = json.loads(str(data["checkpoint_settings_json"]))
        change(stored)
        edited = dict(data, checkpoint_settings_json=settings_json(stored), checkpoint_settings_hash=input_hash(stored))
        path = root / "unread.mat"
        atomic_savemat(path, edited)
        return path

    unread = with_stored(lambda s: (s["saddle"].update(max_retry=3), s["compressor"].update(mixing="fast")))
    candidate = LFTS(diblock_parameters(root / "unread_candidate", cls=LFTS), random_seed=99)
    assert candidate.continue_run(unread)["status"] == "completed"
    read = with_stored(lambda s: s["saddle"].update(tolerance=s["saddle"]["tolerance"] * 2))
    candidate = LFTS(diblock_parameters(root / "read_candidate", cls=LFTS), random_seed=99)
    expect_error(lambda: candidate.continue_run(read), ValidationError)


def check_wtmd(root):
    output = root / "wtmd"
    params = diblock_parameters(output)
    params.update(nx=[4, 4, 4], lx=[3.0, 3.0, 3.0],
                  wtmd={"update_freq": 3, "recording_period": 100, "dpsi": 0.1})
    sim = LFTS(params, random_seed=11)
    rng = np.random.default_rng(3)
    sim.run({name: rng.normal(0, 5, 64) for name in ("A", "B")})
    expected = copy.deepcopy(sim.wtmd.I1)
    saved_fields = read_checkpoint(output / "fields_000004.mat")
    checkpoint = output / "fields_000001.mat"  # no WTMD update, no I1 yet
    sim.continue_run(checkpoint)
    assert all(np.array_equal(sim.wtmd.I1[key], value) for key, value in expected.items())
    actual_fields = read_checkpoint(output / "fields_000004.mat")
    assert np.array_equal(saved_fields["w_aux"], actual_fields["w_aux"])

    data = read_checkpoint(output / "fields_000004.mat")
    before_rng = copy.deepcopy(sim.random_bg.state)
    before_u = sim.wtmd.u.copy()
    before_I1 = copy.deepcopy(sim.wtmd.I1)
    for key in ("wtmd_u", "wtmd_up", "wtmd_I0", "wtmd_I1_A_B", "wtmd_unscaled_up", "wtmd_unscaled_I1_A_B",
                "wtmd_order_parameter_history", "wtmd_dH_history_A_B"):
        bad = copy.deepcopy(data)
        bad.pop(key)
        path = root / (key + ".mat")
        atomic_savemat(path, bad)
        expect_error(lambda: sim.continue_run(path), ValidationError)
        assert sim.random_bg.state == before_rng
        assert np.array_equal(sim.wtmd.u, before_u)
        assert all(np.array_equal(sim.wtmd.I1[key], value) for key, value in before_I1.items())

    # An unversioned checkpoint of the WTMD before 2026-02-03 (no wtmd_I0) is refused before any state changes
    legacy = copy.deepcopy(data)
    legacy.pop("checkpoint_format_version")
    legacy.pop("wtmd_I0")
    path = root / "legacy_wtmd.mat"
    atomic_savemat(path, legacy)
    try:
        sim.continue_run(path, allow_legacy=True)
        raise AssertionError("A legacy WTMD checkpoint without wtmd_I0 was accepted")
    except ValidationError as exc:
        assert "wtmd_I0" in str(exc), exc
    assert sim.random_bg.state == before_rng
    assert np.array_equal(sim.wtmd.u, before_u)


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            with silenced():
                for cls in (LFTS, CLFTS):
                    check_corruption(Path(directory), cls)
                check_unread_section_keys(Path(directory))
                check_wtmd(Path(directory))
        finally:
            os.chdir(previous)
    print("Test passed, checkpoint validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
