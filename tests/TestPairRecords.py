"""Pair-labelled records preserve underscore names through MAT files and restarts.

The distinct pairs A,B_C and A_B,C used to overwrite each other. Exercise the
actual LFTS/CLFTS structure writers, the six interaction histories of four
monomers, and WTMD files and checkpoint restoration. Old, unambiguous WTMD
checkpoints remain readable; ambiguous old payloads and malformed new maps
must fail before mutating a simulation.
"""

import contextlib
import copy
import io
import itertools
import json
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, LFTS, PlatformSelector, ValidationError
from polymerfts.utils.checkpoint import (
    RunRecord,
    extension_state,
    pair_key_data,
    pair_suffix,
    recorded_pair_suffix,
)
from polymerfts.utils.common import atomic_savemat


NAMES = ("A", "A_B", "B_C", "C")


class Counter:
    def __init__(self, count=0):
        self.count = count

    def __call__(self, sample):
        self.count += 1

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]


def read_mat(path):
    return {key: value for key, value in loadmat(path, squeeze_me=True).items()
            if not key.startswith("__")}


def rejects(call):
    try:
        call()
    except ValidationError:
        return
    raise AssertionError("An invalid or ambiguous pair record was accepted")


def parameters(directory, platform, names=NAMES, steps=4, wtmd=False):
    result = {
        "nx": [4, 4, 4], "lx": [3.0, 3.0, 3.0], "ds": 0.05,
        "chain_model": "continuous", "platform": platform,
        "segment_lengths": {name: 1.0 for name in names},
        "chi_n": {",".join(pair): 15.0 for pair in itertools.combinations(names, 2)},
        "polymers": [{"volume_fraction": 1.0,
                      "blocks": [{"type": name, "length": 1.0 / len(names)} for name in names]}],
        "langevin": {"nbar": 10000, "dt": 0.01, "max_step": steps},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5,
                       "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 500, "tolerance": 1e-6},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 2},
        "verbose_level": 0,
    }
    if wtmd:
        result["wtmd"] = {"update_freq": 3, "recording_period": 3, "dpsi": 0.1}
    return result


def fields(names=NAMES):
    rng = np.random.default_rng(710)
    return {name: rng.normal(0.0, 5.0, 64) for name in names}


def check_helpers():
    pairs = [("A", "B"), ("A", "B_C"), ("A_B", "C"),
             ("pair", "A"), ("Long" * 15, "Name" * 15)]
    suffixes = [pair_suffix(pair) for pair in pairs]
    assert suffixes[0] == "A_B"
    assert len(set(suffixes)) == len(pairs)
    for pair, suffix in zip(pairs, suffixes):
        assert pair_suffix(tuple(reversed(pair))) == suffix
        assert pair_suffix(",".join(pair)) == suffix
        assert len(("structure_function_" + suffix + "_real").encode("ascii")) <= 63
    assert suffixes[1].startswith("pair__") and suffixes[2].startswith("pair__")
    assert suffixes[-1].startswith("pair__")
    data = pair_key_data(pairs)
    for pair, suffix in zip(pairs, suffixes):
        assert recorded_pair_suffix(data, pair) == suffix
    assert recorded_pair_suffix({}, ("A_B", "C")) == "A_B_C"

    malformed = [
        {"pair_key_format_version": 1},
        {"pair_keys_json": data["pair_keys_json"]},
        {"pair_key_format_version": 2, "pair_keys_json": data["pair_keys_json"]},
        {"pair_key_format_version": 1, "pair_keys_json": "{"},
        {"pair_key_format_version": 1, "pair_keys_json": "[]"},
        {"pair_key_format_version": 1, "pair_keys_json": "{}"},
    ]
    for bad in malformed:
        rejects(lambda bad=bad: recorded_pair_suffix(bad, pairs[1]))
    wrong = copy.deepcopy(data)
    mapping = json.loads(str(wrong["pair_keys_json"]))
    mapping[suffixes[1]] = ["A_B", "C"]
    wrong["pair_keys_json"] = json.dumps(mapping)
    rejects(lambda: recorded_pair_suffix(wrong, pairs[1]))


def check_statistics(root, platform, cls):
    output = root / (cls.__name__ + "_statistics")
    output.mkdir()
    sim = cls(parameters(output, platform), random_seed=11)
    pairs = list(itertools.combinations(sim.monomer_types, 2))
    spectrum_pairs = list(itertools.combinations_with_replacement(sim.monomer_types, 2))
    assert len(pairs) == 6 and len(spectrum_pairs) == 10
    complex_values = cls is CLFTS
    record = RunRecord({"dH " + ",".join(pair): 0 for pair in pairs},
                       sim.structure_functions.empty_sums(), complex_values=complex_values)
    expected_history = {}
    for index, pair in enumerate(pairs):
        value = 11.0 * (index + 1) + (1j * (index + 1) if complex_values else 0)
        expected_history[pair] = value
        record.add("dH " + ",".join(pair), value)
    record.sample_count = 1
    atomic_savemat(output / "dH.mat", record.statistics(), long_field_names=True)
    history = read_mat(output / "dH.mat")
    suffixes = {recorded_pair_suffix(history, pair) for pair in pairs}
    assert len(suffixes) == 6
    for pair, value in expected_history.items():
        key = "dH_history_" + recorded_pair_suffix(history, pair)
        actual = history[key + "_real"] + 1j * history[key + "_imag"] if complex_values else history[key]
        np.testing.assert_array_equal(actual, value)
    assert len([key for key in history if key.startswith("dH_history_")]) == (12 if complex_values else 6)

    expected_spectra = {}
    scale = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"])
    for index, pair in enumerate(spectrum_pairs):
        value = (index + 1) * (1.0 + 0.5j)
        record.sums[",".join(pair)][:] = value
        expected_spectra[pair] = value * scale
    sim.structure_functions.save(2, record)
    data = read_mat(output / "structure_function_000002.mat")
    assert len({recorded_pair_suffix(data, pair) for pair in spectrum_pairs}) == 10
    for pair, value in expected_spectra.items():
        key = "structure_function_" + recorded_pair_suffix(data, pair)
        actual = data[key + "_real"] + 1j * data[key + "_imag"] if complex_values else data[key]
        np.testing.assert_allclose(actual, value, rtol=1e-15, atol=0)
    assert len([key for key in data if key.startswith("structure_function_")]) == (20 if complex_values else 10)


def wtmd_state(sim):
    return {name: copy.deepcopy(getattr(sim.wtmd, name)) for name in
            ("u", "up", "I0", "I1", "order_parameter_history", "dH_history")}


def assert_wtmd_state(actual, expected):
    for name in expected:
        if isinstance(expected[name], dict):
            assert actual[name].keys() == expected[name].keys(), name
            for key in expected[name]:
                np.testing.assert_array_equal(actual[name][key], expected[name][key], err_msg=name + ": " + key)
        else:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def legacy_v2(data, pairs):
    """The actual old writer joined names with underscores, including overwrites."""
    legacy = copy.deepcopy(data)
    for pair in pairs:
        suffix = recorded_pair_suffix(data, pair)
        old_suffix = "_".join(sorted(pair.split(",")))
        for prefix in ("wtmd_I1_", "wtmd_unscaled_I1_", "wtmd_dH_history_"):
            if prefix + suffix in data:
                legacy.pop(prefix + suffix, None)
                legacy[prefix + old_suffix] = copy.deepcopy(data[prefix + suffix])
    legacy.pop("pair_key_format_version", None)
    legacy.pop("pair_keys_json", None)
    legacy["checkpoint_format_version"] = 2
    return legacy


def assert_rejected_without_mutation(sim, data, path):
    before_rng = copy.deepcopy(sim.random_bg.state)
    before_extensions = copy.deepcopy(extension_state(sim))
    before_metadata = copy.deepcopy(sim.metadata)
    before_wtmd = wtmd_state(sim)
    atomic_savemat(path, data, long_field_names=True)
    rejects(lambda: sim.continue_run(path))
    assert sim.random_bg.state == before_rng
    assert extension_state(sim) == before_extensions
    assert sim.metadata == before_metadata
    assert_wtmd_state(wtmd_state(sim), before_wtmd)


def check_wtmd(root, platform):
    full_dir = root / "wtmd_full"
    full = LFTS(parameters(full_dir, platform, wtmd=True), random_seed=21,
                callbacks={"counter": Counter()})
    assert full.run(fields())["status"] == "completed"
    saved = read_mat(full_dir / "fields_000004.mat")
    assert saved["checkpoint_format_version"] == 3
    expected = wtmd_state(full)
    assert len(expected["I1"]) == 6 and len(expected["dH_history"]) == 6
    assert len(expected["order_parameter_history"]) == 1
    for pair in full.chi_n:
        suffix = recorded_pair_suffix(saved, pair)
        np.testing.assert_array_equal(saved["wtmd_unscaled_I1_" + suffix], expected["I1"][pair])
        np.testing.assert_array_equal(np.atleast_1d(saved["wtmd_dH_history_" + suffix]),
                                      expected["dH_history"][pair])

    full.wtmd.write_data(root / "wtmd_statistics.mat")
    statistics = read_mat(root / "wtmd_statistics.mat")
    assert len([key for key in statistics if key.startswith("I1_")]) == 6
    assert len([key for key in statistics if key.startswith("dH_psi_")]) == 6
    for pair, values in expected["I1"].items():
        suffix = recorded_pair_suffix(statistics, pair)
        np.testing.assert_array_equal(statistics["I1_" + suffix], values * full.wtmd.CV)
        conditional = values * full.wtmd.CV
        nonzero = np.abs(full.wtmd.I0) > 0
        conditional[nonzero] /= full.wtmd.I0[nonzero]
        np.testing.assert_array_equal(statistics["dH_psi_" + suffix], conditional)

    continued_dir = root / "wtmd_continued"
    continued = LFTS(parameters(continued_dir, platform, wtmd=True), random_seed=99,
                     callbacks={"counter": Counter(17)})
    continued.continue_run(full_dir / "fields_000001.mat")
    assert_wtmd_state(wtmd_state(continued), expected)
    assert continued.random_bg.state == full.random_bg.state
    assert extension_state(continued) == extension_state(full)
    actual = read_mat(continued_dir / "fields_000004.mat")
    np.testing.assert_array_equal(actual["w_aux"], saved["w_aux"])

    candidate = LFTS(parameters(root / "wtmd_rejected", platform, wtmd=True), random_seed=47,
                     callbacks={"counter": Counter(17)})
    # At step 1 only pending derivatives exist; at step 3 only accumulated I1 exists.
    # Either old payload has lost one pair and must fail before any state is restored.
    for step, label in ((1, "pending"), (3, "integral")):
        old = legacy_v2(read_mat(full_dir / f"fields_{step:06d}.mat"), full.chi_n)
        assert_rejected_without_mutation(candidate, old, root / ("ambiguous_v2_" + label + ".mat"))

    corruptions = []
    no_mapping = copy.deepcopy(saved)
    no_mapping.pop("pair_key_format_version")
    no_mapping.pop("pair_keys_json")
    corruptions.append(("no_mapping", no_mapping))
    bad_json = copy.deepcopy(saved)
    bad_json["pair_keys_json"] = "{"
    corruptions.append(("bad_json", bad_json))
    missing_pair = copy.deepcopy(saved)
    mapping = json.loads(str(saved["pair_keys_json"]))
    mapping.pop(pair_suffix("A,B_C"))
    missing_pair["pair_keys_json"] = json.dumps(mapping)
    corruptions.append(("missing_pair", missing_pair))
    extra_pair = copy.deepcopy(saved)
    mapping = json.loads(str(saved["pair_keys_json"]))
    mapping[pair_suffix(("X", "Y"))] = ["X", "Y"]
    extra_pair["pair_keys_json"] = json.dumps(mapping)
    corruptions.append(("extra_pair", extra_pair))
    wrong_pair = copy.deepcopy(saved)
    mapping = json.loads(str(saved["pair_keys_json"]))
    mapping[pair_suffix("A,B_C")] = ["A_B", "C"]
    wrong_pair["pair_keys_json"] = json.dumps(mapping)
    corruptions.append(("wrong_pair", wrong_pair))
    for label, bad in corruptions:
        assert_rejected_without_mutation(candidate, bad, root / ("invalid_v3_" + label + ".mat"))

    # An underscore in a name alone is not ambiguous. Existing v2 files remain usable.
    names = ("A_B", "C")
    old_dir = root / "unambiguous_original"
    original = LFTS(parameters(old_dir, platform, names=names, wtmd=True), random_seed=31)
    original.run(fields(names))
    old = legacy_v2(read_mat(old_dir / "fields_000004.mat"), original.chi_n)
    old_path = root / "unambiguous_v2.mat"
    atomic_savemat(old_path, old, long_field_names=True)
    restored = LFTS(parameters(root / "unambiguous_restored", platform, names=names, wtmd=True), random_seed=73)
    assert restored.continue_run(old_path)["status"] == "completed"
    assert_wtmd_state(wtmd_state(restored), wtmd_state(original))
    assert restored.random_bg.state == original.random_bg.state


def main():
    check_helpers()
    platforms = [name for name in PlatformSelector.avail_platforms() if name.startswith("cpu")]
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_pair_records_") as directory:
        root = Path(directory)
        previous = Path.cwd()
        os.chdir(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for cls in (LFTS, CLFTS):
                    check_statistics(root, platforms[0], cls)
                check_wtmd(root, platforms[0])
        finally:
            os.chdir(previous)
    print("Pair records: statistics, WTMD files and versioned restarts preserve every pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
