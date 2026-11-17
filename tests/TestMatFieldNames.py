#!/usr/bin/env python3
"""Lossless MAT metadata for supported monomer names and exact FTS restarts.

Leading underscores and long alphabetic names remain valid physical inputs.
Check ordinary MAT structs stay readable, escaped keys decode without losing
data, and all SCFT/FTS/WTMD writers preserve those names and their fields.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat

from polymerfts import SCFT, LFTS, CLFTS
from polymerfts.utils.checkpoint import pair_suffix
from polymerfts.utils.common import atomic_savemat, input_hash
from polymerfts.utils.mat_fields import decode_mat_fields
from fts_checks import compare_files, exact_restart, python_platform, quiet


MARKER = "polymerfts_mat_keys_json"
NAMES = (("ordinary", "A"), ("underscore", "_A"), ("long", "A" * 70))


def read_mat(path):
    return decode_mat_fields(loadmat(path, simplify_cells=True))


def check_encoding(root):
    # A real user key deliberately occupies the first generated ID for _A.
    collision = "field_" + hashlib.sha256(b"_A").hexdigest()[:24]
    values = {"A": 1, "_A": 2, "2A": 3, "": 4, "\u03b1": 5, "L" * 70: 6,
              MARKER: "user value", collision: 7, "A,B": 8, "M" * 40: 9}
    array = np.arange(12.0).reshape(3, 4)
    cells = np.empty(2, dtype=object)
    cells[0], cells[1] = {"_left": 10}, {"_right": 11}
    unicode_key = "concentration_Na⁺_real"
    top_collision = "field_" + hashlib.sha256(unicode_key.encode()).hexdigest()[:24]
    original = {"metadata": values, "array": array, "cells": cells,
                "records": [{"_record": 12}, {"_record": 13}], "w_" + "L" * 70: array,
                unicode_key: array, "concentration_Cl⁻_imag": -array, top_collision: array + 1}
    for long_fields in (False, True):
        path = root / f"encoding_{long_fields}.mat"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            atomic_savemat(path, original, long_field_names=long_fields)
        raw = loadmat(path, simplify_cells=True)
        raw_fields = raw["metadata"]
        assert raw_fields[collision] == 7, "Generated ID overwrote a user field"
        assert all(len(key) <= (63 if long_fields else 31) for key in raw_fields)
        assert ("M" * 40 in raw_fields) == long_fields
        assert raw_fields["A"] == 1 and raw_fields["A,B"] == 8
        assert MARKER in raw and unicode_key not in raw
        assert all(key.isascii() for key in raw)
        np.testing.assert_array_equal(raw[top_collision], array + 1)
        restored = decode_mat_fields(raw)
        assert restored["metadata"] == values
        assert list(restored["cells"]) == [{"_left": 10}, {"_right": 11}]
        assert list(restored["records"]) == [{"_record": 12}, {"_record": 13}]
        np.testing.assert_array_equal(restored["array"], array)
        np.testing.assert_array_equal(restored["w_" + "L" * 70], array)
        np.testing.assert_array_equal(restored[unicode_key], array)
        np.testing.assert_array_equal(restored["concentration_Cl⁻_imag"], -array)
        unsqueezed = decode_mat_fields(loadmat(path))
        np.testing.assert_array_equal(unsqueezed[unicode_key], array)
        assert restored["array"] is raw["array"], "Decoder copied an ordinary numeric field"
    assert original["metadata"] == values and original["cells"][0] == {"_left": 10}
    np.testing.assert_array_equal(original["array"], np.arange(12.0).reshape(3, 4))
    assert decode_mat_fields({MARKER: 1.5, "A": 1}) == {MARKER: 1.5, "A": 1}
    # A rejected encoding must preserve an existing checkpoint and remove its
    # temporary file, just like a failed scipy write.
    path = root / "atomic.mat"
    atomic_savemat(path, {"value": array})
    previous = path.read_bytes()
    try:
        atomic_savemat(path, {"metadata": {1: 2, "1": 3}})
    except ValueError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("Keys colliding after string conversion were accepted")
    assert path.read_bytes() == previous and not list(root.glob(".atomic.mat.*.tmp"))
    for marker in ("{", "[]", '{"version":2,"keys":{"x":"_A"}}',
                   '{"version":1,"keys":{"missing":"_A"}}',
                   '{"version":1,"keys":{"x":"A"}}',
                   '{"version":1,"keys":{"x":"_A","y":"_A"}}'):
        malformed = {MARKER: marker, "x": 1, "y": 2, "A": 3}
        before = copy.deepcopy(malformed)
        try:
            decode_mat_fields(malformed)
        except ValueError as error:
            assert "MAT field-name mapping" in str(error)
        else:
            raise AssertionError("Malformed mapping was accepted")
        assert malformed == before


def parameters(platform, name):
    return {
        "nx": [4], "lx": [2.0], "ds": 0.25, "chain_model": "continuous",
        "segment_lengths": {name: 1.0, "B": 1.0}, "chi_n": {name + ",B": 3.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": name, "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": platform,
    }


def check_scft(root, platform, name):
    params = parameters(platform, name)
    params.update(relax_cell=False, max_iter=1, tolerance=1e-8,
                  optimizer={"name": "am", "max_hist": 5, "start_error": 0.01,
                             "mix_min": 0.1, "mix_init": 0.1})
    before = copy.deepcopy(params)
    initial = {monomer: np.zeros(4) for monomer in (name, "B")}
    sim = quiet(SCFT, params)
    result = quiet(sim.run, initial, return_result=True)
    assert result.converged and quiet(sim.prop_solver.check_total_partition)
    root.mkdir(parents=True)
    for suffix in ("json", "mat"):
        path = root / ("result." + suffix)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet(sim.save_results, path)
        saved = json.loads(path.read_text()) if suffix == "json" else read_mat(path)
        restored = saved["initial_params"]
        assert restored["segment_lengths"] == params["segment_lengths"]
        assert restored["chi_n"] == params["chi_n"]
        assert saved["chi_n"] == sim.chi_n
        for monomer in initial:
            np.testing.assert_array_equal(saved["w_" + monomer], result.w[monomer])
            np.testing.assert_array_equal(saved["phi_" + monomer], result.phi[monomer])
        # MAT cell/struct squeezing is independent of key encoding. Restore only
        # singleton grid/list dimensions to use the saved parameters directly.
        restored["nx"] = np.atleast_1d(restored["nx"]).astype(int).tolist()
        restored["lx"] = np.atleast_1d(restored["lx"]).astype(float).tolist()
        restored["relax_cell"] = bool(restored["relax_cell"])
        if isinstance(restored["polymers"], dict):
            restored["polymers"] = [restored["polymers"]]
        for polymer in restored["polymers"]:
            if isinstance(polymer["blocks"], np.ndarray):
                polymer["blocks"] = polymer["blocks"].tolist()
        reconstructed = quiet(SCFT, restored)
        again = quiet(reconstructed.run, initial, return_result=True)
        assert again.converged
        for monomer in initial:
            np.testing.assert_array_equal(again.phi[monomer], result.phi[monomer])
    raw = loadmat(root / "result.mat", simplify_cells=True)
    assert (MARKER in raw["initial_params"]["segment_lengths"]) == (name != "A")
    assert params == before


class Counter:
    def __init__(self, count=0):
        self.count = count

    def __call__(self, sample):
        self.count += 1

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]


def fts_parameters(directory, platform, name, cls, steps):
    params = parameters(platform, name)
    params.update(nx=[4, 4, 4], lx=[3.0, 3.0, 3.0], ds=0.05,
                  chi_n={name + ",B": 15.0}, verbose_level=0,
                  langevin={"nbar": 10000.0, "dt": 0.001, "max_step": steps},
                  compressor={"name": "am", "max_hist": 20, "start_error": 0.5,
                              "mix_min": 0.01, "mix_init": 0.01},
                  saddle={"max_iter": 500, "tolerance": 1e-8},
                  recording={"dir": str(directory), "checkpoint_interval": 1,
                             "sample_interval": 1, "statistics_save_interval": 3})
    if cls is LFTS:
        params["wtmd"] = {"update_freq": 3, "recording_period": 3, "dpsi": 0.1}
    else:
        del params["compressor"], params["saddle"]   # options of L-FTS, which CL-FTS refuses
    return params


def check_fts(root, platform, name, cls):
    def make(label, steps, seed):
        params = fts_parameters(root / label, platform, name, cls, steps)
        before = copy.deepcopy(params)
        sim = quiet(cls, params, random_seed=seed,
                    callbacks={"counter": Counter(17 if label == "continued" else 0)})
        sim.draw_architectures = False
        assert params == before
        built[label] = sim
        return sim

    def inspect(checkpoint):
        partial = read_mat(checkpoint)
        assert partial["record_sample_count"] == 2 and np.size(partial["record_H_history"]) == 2
        assert partial["initial_params"]["segment_lengths"] == {name: 1.0, "B": 1.0}
        assert partial["chi_n"] == built["partial"].chi_n
        settings = json.loads(partial["checkpoint_settings_json"])
        assert settings["segment_lengths"] == {name: 1.0, "B": 1.0}
        assert input_hash(settings) == partial["checkpoint_settings_hash"]
        if cls is LFTS:
            assert np.size(partial["wtmd_order_parameter_history"]) == 2

    built = {}
    initial = dict(zip((name, "B"), np.random.default_rng(19).normal(0.0, 5.0, (2, 64))))
    fields_keys = ("w_", "phi_", "normal_noise", "record_", "rng_state", "extension_state",
                   "wtmd_unscaled_", "wtmd_I0", "wtmd_order_parameter_history", "wtmd_dH_history_")
    sims = exact_restart(make, lambda sim: sim.run(initial), 4, 2, {
        "fields_000003.mat": fields_keys, "fields_000004.mat": fields_keys,
        "dH_000003.mat": ("H_", "dH_", "chemical_potential_"),
        "structure_function_000003.mat": ("structure_function_",)}, inspect)
    assert sims["continued"].metadata["random_seed"] == 17
    assert sims["continued"].metadata["initial_fields_hash"] == sims["full"].metadata["initial_fields_hash"]
    final = read_mat(root / "continued" / "fields_000004.mat")
    assert json.loads(final["extension_state_json"])["callbacks"]["counter"] == {"count": 4}
    assert final["record_sample_count"] == 1
    if cls is LFTS:
        assert np.size(final["wtmd_order_parameter_history"]) == 1
        for label in ("full", "continued"):
            quiet(sims[label].wtmd.write_data, root / label / "wtmd_final.mat")
            saved = read_mat(root / label / "wtmd_final.mat")
            for pair in sims[label].chi_n:
                suffix = pair_suffix(pair)
                np.testing.assert_array_equal(saved["I1_" + suffix],
                                              sims[label].wtmd.I1[pair] * sims[label].wtmd.CV)
        compare_files(root / "full", root / "continued", "wtmd_final.mat", ("I0", "I1_", "dH_psi_", "u", "up"))


def check_unicode_ions(root, platform):
    """Unicode ion concentrations survive statistics output and pending-sample restarts."""
    def make(label, steps, seed):
        params = fts_parameters(root / label, platform, "A", CLFTS, steps)
        params.update(nx=[4], lx=[2.0], ds=0.5, chain_model="discrete",
                      smearing={"type": "gaussian", "a_int": 0.1},
                      electrostatics={"charges": {"A": 0.0, "B": 0.0}, "coupling": 5.0,
                                     "counterion": None, "ions": {
                                         "Na⁺": {"valence": 1.0, "concentration": 0.01, "smearing_length": 0.1},
                                         "Cl⁻": {"valence": -1.0, "concentration": 0.01, "smearing_length": 0.1}}})
        params["recording"]["concentrations"] = True
        sim = quiet(CLFTS, params, random_seed=seed)
        sim.draw_architectures = False
        return sim

    def inspect(checkpoint):
        assert read_mat(checkpoint)["record_sample_count"] == 2

    initial = dict(zip(("A", "B"), np.random.default_rng(19).normal(0.0, 5.0, (2, 4))))
    fields_keys = ("w_", "phi_", "record_", "psi_", "normal_noise", "rng_state")
    sims = exact_restart(make, lambda sim: sim.run(initial), 4, 2,
                         {"fields_000003.mat": fields_keys, "fields_000004.mat": fields_keys}, inspect)
    assert sims["full"].electrostatics.random_bg.state == sims["continued"].electrostatics.random_bg.state
    filename = "structure_function_000003.mat"
    full, continued = (read_mat(root / label / filename) for label in ("full", "continued"))
    for ion in ("Na⁺", "Cl⁻"):
        for suffix in ("real", "imag"):
            key = f"concentration_{ion}_{suffix}"
            np.testing.assert_array_equal(continued[key], full[key])
        np.testing.assert_allclose(np.mean(full[f"concentration_{ion}_real"]), 0.01, rtol=0, atol=1e-14)
    assert MARKER in loadmat(root / "full" / filename, squeeze_me=True)


def main():
    with tempfile.TemporaryDirectory(prefix="polymerfts_mat_fields_") as directory:
        root = Path(directory)
        check_encoding(root)
        # The key encoding is Python: one platform runs the writers and restarts
        platform = python_platform()
        for label, name in NAMES:
            output = root / label
            check_scft(output / "scft", platform, name)
            for cls in (LFTS, CLFTS):
                check_fts(output / cls.__name__, platform, name, cls)
        check_unicode_ions(root / "unicode_ions", platform)
        print(f"[PASS] {platform}: MAT names, SCFT roundtrips, LFTS/CLFTS/WTMD restarts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
