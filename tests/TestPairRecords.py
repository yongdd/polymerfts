"""Monomer names in the keys of the measurement files, and the names that L-FTS and CL-FTS refuse.

The files name the measurements of a monomer pair by the two names joined with '_' (dH_history_A_B,
structure_function_A_B, I1_A_B, wtmd_I1_A_B). Names whose pairs would share such a key, as A,B_C and
A_B,C do, are refused when a simulation is constructed, and so is a monomer type named 'aux' in L-FTS,
whose field would take the key w_aux of the auxiliary fields. Underscores alone are no problem: the
statistics of four names keep their six and ten pairs apart, a WTMD run keeps its keys through its files
and an exact continuation, and CL-FTS (fields w_<type>_real/_imag) runs and continues with a type 'aux'.
"""

import contextlib
import copy
import io
import itertools
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, LFTS, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing
from polymerfts.utils.checkpoint import RunRecord, extension_state, pair_suffix
from polymerfts.utils.common import atomic_savemat


def read_mat(path):
    return {key: value for key, value in loadmat(path, squeeze_me=True).items() if not key.startswith("__")}


def parameters(directory, platform, names, steps=4, wtmd=False, nx=(4, 4, 4)):
    result = {
        "nx": list(nx), "lx": [3.0] * len(nx), "ds": 0.05,
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


def fields(names, size=64):
    rng = np.random.default_rng(710)
    return {name: rng.normal(0.0, 5.0, size) for name in names}


def refused(root, platform):
    for cls in (LFTS, CLFTS):
        try:
            cls(parameters(root / "refused", platform, ("A", "A_B", "B_C", "C")))
        except ValidationError as error:
            assert "A,B_C" in str(error) and "A_B,C" in str(error), str(error)
        else:
            raise AssertionError(f"{cls.__name__} accepted the pairs A,B_C and A_B,C with one key suffix")
    try:
        LFTS(parameters(root / "refused", platform, ("aux", "B")))
    except ValidationError as error:
        assert "w_aux" in str(error), str(error)
    else:
        raise AssertionError("L-FTS accepted a monomer type 'aux', whose field would be w_aux")


def check_statistics(root, platform, cls):
    output = root / (cls.__name__ + "_statistics")
    output.mkdir()
    names = ("A", "B_C", "D_E", "F")
    sim = cls(parameters(output, platform, names), random_seed=11)
    pairs = list(itertools.combinations(sim.monomer_types, 2))
    spectrum_pairs = list(itertools.combinations_with_replacement(sim.monomer_types, 2))
    assert len(pairs) == 6 and len(spectrum_pairs) == 10
    complex_values = cls is CLFTS
    record = RunRecord({"dH " + ",".join(pair): 0 for pair in pairs},
                       sim.structure_functions.empty_sums(), complex_values=complex_values)
    for index, pair in enumerate(pairs):
        record.add("dH " + ",".join(pair), 11.0 * (index + 1) + (1j * (index + 1) if complex_values else 0))
    record.sample_count = 1
    atomic_savemat(output / "dH.mat", record.statistics(), long_field_names=True)
    history = read_mat(output / "dH.mat")
    for index, pair in enumerate(pairs):
        key = "dH_history_" + "_".join(pair)
        actual = history[key + "_real"] + 1j * history[key + "_imag"] if complex_values else history[key]
        np.testing.assert_array_equal(actual, 11.0 * (index + 1) + (1j * (index + 1) if complex_values else 0))
    assert len([key for key in history if key.startswith("dH_history_")]) == (12 if complex_values else 6)

    scale = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"])
    for index, pair in enumerate(spectrum_pairs):
        record.sums[",".join(pair)][:] = (index + 1) * (1.0 + 0.5j)
    sim.structure_functions.save(2, record)
    data = read_mat(output / "structure_function_000002.mat")
    for index, pair in enumerate(spectrum_pairs):
        key = "structure_function_" + "_".join(pair)
        actual = data[key + "_real"] + 1j * data[key + "_imag"] if complex_values else data[key]
        np.testing.assert_allclose(actual, (index + 1) * (1.0 + 0.5j) * scale, rtol=1e-15, atol=0)
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


def check_wtmd(root, platform):
    names = ("A_B", "C")
    full_dir = root / "wtmd_full"
    full = LFTS(parameters(full_dir, platform, names, wtmd=True), random_seed=21)
    assert full.run(fields(names))["status"] == "completed"
    saved = read_mat(full_dir / "fields_000004.mat")
    expected = wtmd_state(full)
    assert list(expected["I1"]) == ["A_B,C"] and pair_suffix("A_B,C") == "A_B_C"
    np.testing.assert_array_equal(saved["wtmd_unscaled_I1_A_B_C"], expected["I1"]["A_B,C"])
    np.testing.assert_array_equal(np.atleast_1d(saved["wtmd_dH_history_A_B_C"]), expected["dH_history"]["A_B,C"])
    full.wtmd.write_data(root / "wtmd_statistics.mat")
    statistics = read_mat(root / "wtmd_statistics.mat")
    np.testing.assert_array_equal(statistics["I1_A_B_C"], expected["I1"]["A_B,C"] * full.wtmd.CV)
    conditional = expected["I1"]["A_B,C"] * full.wtmd.CV
    nonzero = np.abs(full.wtmd.I0) > 0
    conditional[nonzero] /= full.wtmd.I0[nonzero]
    np.testing.assert_array_equal(statistics["dH_psi_A_B_C"], conditional)

    continued = LFTS(parameters(root / "wtmd_continued", platform, names, wtmd=True), random_seed=99)
    assert continued.continue_run(full_dir / "fields_000001.mat")["status"] == "completed"
    assert_wtmd_state(wtmd_state(continued), expected)
    assert continued.random_bg.state == full.random_bg.state
    assert extension_state(continued) == extension_state(full)
    np.testing.assert_array_equal(read_mat(root / "wtmd_continued" / "fields_000004.mat")["w_aux"], saved["w_aux"])


def check_clfts_aux(root, platform):
    names = ("aux", "B")
    full = CLFTS(parameters(root / "aux_full", platform, names, nx=[8]), random_seed=5)
    assert full.run(fields(names, 8))["status"] == "completed"
    continued = CLFTS(parameters(root / "aux_continued", platform, names, nx=[8]), random_seed=99)
    assert continued.continue_run(root / "aux_full" / "fields_000002.mat")["status"] == "completed"
    a, b = (read_mat(root / name / "fields_000004.mat") for name in ("aux_full", "aux_continued"))
    assert a["w_aux"].shape == (2, 8) and "w_aux_real" in a and "w_aux_imag" in a
    for key in ("w_aux", "w_aux_real", "w_aux_imag", "w_B_real", "w_B_imag", "phi_aux_real", "normal_noise_prev"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def main():
    # Fixes in Python: one platform, a CPU one where the build has it
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return int(cuda_part_missing()) or 77
    platforms = [next((name for name in platforms if name.startswith("cpu")), platforms[0])]
    with tempfile.TemporaryDirectory(prefix="polymerfts_pair_records_") as directory:
        root = Path(directory)
        previous = Path.cwd()
        os.chdir(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                refused(root, platforms[0])
                for cls in (LFTS, CLFTS):
                    check_statistics(root, platforms[0], cls)
                check_wtmd(root, platforms[0])
                check_clfts_aux(root, platforms[0])
        finally:
            os.chdir(previous)
    print("Pair records: colliding names refused, distinct keys through statistics, WTMD files and restarts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
