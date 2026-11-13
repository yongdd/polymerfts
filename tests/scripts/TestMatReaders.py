"""Bundled MAT readers preserve names and field components without a backend."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[2]
MAT = runpy.run_path(str(ROOT / "src/python/utils/mat_fields.py"))
AUX = runpy.run_path(str(ROOT / "scripts/analysis/convert_into_auxiliary_fields.py"))
PSCF = runpy.run_path(str(ROOT / "scripts/pscf/pscf_converter.py"))
RENORM = runpy.run_path(str(ROOT / "scripts/analysis/renormalization.py"))
GUI = runpy.run_path(str(Path(__file__).with_name("TestSimulationScript.py")))
MODEL = runpy.run_path(str(Path(__file__).with_name("TestSimulationModel.py")))


def field_file(names, split, metadata):
    real = np.arange(len(names) * 24, dtype=float).reshape(len(names), 24) / 8
    fields = real + (1j * (real + 2) if split else 0)
    inverse = np.eye(len(names)) + .25
    data = {"monomer_types": names, "nx": [2, 3, 4], "lx": [3., 4., 5.],
            "matrix_a_inverse": inverse}
    if metadata:
        data["checkpoint_settings_json"] = json.dumps({
            "simulation": "CLFTS" if split else "LFTS", "angles": [90., 90., 90.]})
        data["simulation_metadata_json"] = json.dumps({"precision": "complex128" if split else "float64"})
    for name, field in zip(names, fields):
        if split:
            data["w_" + name + "_real"] = field.real
            data["w_" + name + "_imag"] = field.imag
        else:
            data["w_" + name] = field
    return data, fields, inverse


@pytest.mark.parametrize("names", (("A", "B"), ("A", "A_real"), ("A", "A_real", "A_imag")))
@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("metadata", (False, True))
def test_auxiliary_reader_uses_file_representation(names, split, metadata, tmp_path):
    data, fields, inverse = field_file(names, split, metadata)
    path = tmp_path / "fields.mat"
    savemat(path, data)
    loaded = loadmat(path, squeeze_me=True)
    decoded = MAT["read_monomer_fields"](loaded)
    for name, expected in zip(names, fields):
        np.testing.assert_array_equal(decoded[name], expected)
        # Selecting one species must still use all stored names to infer the format.
        np.testing.assert_array_equal(AUX["monomer_field"](loaded, name), expected)
    output = tmp_path / "aux.mat"
    assert AUX["main"]([str(path), "-o", str(output)]) == 0
    np.testing.assert_allclose(loadmat(output, squeeze_me=True)["w_aux"], inverse @ fields,
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode,split", (("SCFT", False), ("L-FTS", False), ("CL-FTS", True)))
@pytest.mark.parametrize("metadata", (False, True))
def test_generated_script_preserves_collision_names(mode, split, metadata, tmp_path, monkeypatch):
    names = ("A", "A_real", "A_imag")
    form = MODEL["values"]()
    form.update(mode=mode, initial="mat", mat_file="fields.mat")
    model = form["model"]
    model["box"]["nx"] = ["2", "3", "4"]
    mapping = dict(zip(("A", "B", "C"), names))
    for monomer in model["monomers"]:
        monomer["name"] = mapping[monomer["name"]]
    for polymer in model["polymers"]:
        for block in polymer["blocks"]:
            block["type"] = mapping[block["type"]]
    for interaction in model["interactions"]:
        for key in ("a", "b"):
            interaction[key] = mapping[interaction[key]]
    data, fields, _ = field_file(names, split, metadata)
    (tmp_path / "export").mkdir()
    savemat(tmp_path / "export/fields.mat", data)
    calls = GUI["execute_export"](form, tmp_path, monkeypatch)
    for name, expected in zip(names, fields):
        np.testing.assert_array_equal(calls["fields"][name].ravel(), expected)


def test_metadata_resolves_ambiguous_legacy_schema():
    data, fields, _ = field_file(("A", "B"), True, False)
    data.update(w_A=np.full(24, 10.), w_B=np.full(24, 20.))
    with pytest.raises(ValueError, match="Ambiguous"):
        MAT["read_monomer_fields"](data)
    data["checkpoint_settings_json"] = json.dumps({"simulation": "CLFTS"})
    for name, expected in zip(("A", "B"), fields):
        np.testing.assert_array_equal(MAT["read_monomer_fields"](data)[name], expected)
    data["checkpoint_settings_json"] = json.dumps({"simulation": "LFTS"})
    np.testing.assert_array_equal(MAT["read_monomer_fields"](data)["A"], data["w_A"])
    data["simulation_metadata_json"] = json.dumps({"precision": "complex128"})
    with pytest.raises(ValueError, match="disagrees"):
        MAT["read_monomer_fields"](data)


@pytest.mark.parametrize("problem", ("missing", "shape", "nonfinite", "complex_component"))
def test_split_collision_file_is_not_silently_reinterpreted(problem):
    data, _, _ = field_file(("A", "A_real"), True, True)
    if problem == "missing":
        del data["w_A_real_imag"]
    elif problem == "shape":
        data["w_A_real_imag"] = np.zeros(1)
    elif problem == "nonfinite":
        data["w_A_real_imag"] = np.full(24, np.nan)
    else:
        data["w_A_real_real"] = np.full(24, 1 + 2j)
    with pytest.raises(ValueError):
        MAT["read_monomer_fields"](data)


def parameter_file(name):
    params = {"nx": [2, 3, 4], "lx": [3., 4., 5.], "chain_model": "continuous", "ds": .1,
              "segment_lengths": {name: 1., "B": 1.}, "chi_n": {name + ",B": 12.},
              "polymers": [{"volume_fraction": 1., "blocks": [
                  {"type": name, "length": .4}, {"type": "B", "length": .6}]}],
              "langevin": {"nbar": 1e4, "dt": .01}}
    field = (np.arange(24, dtype=float) + .125).reshape(params["nx"])
    return {"initial_params": params, "nx": params["nx"], "lx": params["lx"],
            "chi_n": params["chi_n"], "w_" + name: field, "w_B": -field}


def write_encoded(path, data):
    savemat(path, MAT["_encode_mat_fields"](data, 63, top_level=True), long_field_names=True)


@pytest.mark.parametrize("name", ("A", "_A", "A" * 70))
def test_parameter_readers_restore_encoded_names(name, tmp_path, capsys):
    data = parameter_file(name)
    path = tmp_path / "fields.mat"
    write_encoded(path, data)
    read = PSCF["read_input"](path)
    assert read["initial_params"]["segment_lengths"] == data["initial_params"]["segment_lengths"]
    assert read["initial_params"]["chi_n"] == data["initial_params"]["chi_n"]
    # Full PSCF output must equal the same physical input supplied as ordinary JSON.
    plain = tmp_path / "fields.json"
    plain.write_text(json.dumps(data, default=lambda x: x.tolist()), encoding="utf-8")
    PSCF["export_files"](path, tmp_path / "from_mat")
    PSCF["export_files"](plain, tmp_path / "from_json")
    for filename in ("param", "command", "in/omega", "metadata.json"):
        assert (tmp_path / "from_mat" / filename).read_bytes() == (tmp_path / "from_json" / filename).read_bytes()
    settings = RENORM["read_simulation"](path)
    assert settings["segment_lengths"] == {name: 1., "B": 1.}
    assert settings["chi_n"] == {name + ",B": 12.}
    assert RENORM["main"]([str(path), "--quadrature", "16"]) == 0
    output = capsys.readouterr().out
    assert f"chi_b N ({name},B) = 12.0000000" in output
    assert "polymerfts_mat_keys_json" not in output and "field_" not in output


def test_numeric_node_names_and_invalid_mappings(tmp_path):
    data = parameter_file("_A")
    data["initial_params"]["polymers"][0]["initial_conditions"] = {"0": "graft"}
    path = tmp_path / "nodes.mat"
    write_encoded(path, data)
    assert PSCF["read_input"](path)["initial_params"]["polymers"]["initial_conditions"] == {"0": "graft"}
    data = parameter_file("A")
    data["initial_params"]["segment_lengths"]["polymerfts_mat_keys_json"] = '{"version": 9, "keys": {"A": "B"}}'
    savemat(path, data, long_field_names=True)
    for reader in (PSCF["read_input"], RENORM["read_simulation"]):
        with pytest.raises(ValueError, match="Invalid MAT field-name mapping"):
            reader(path)


def test_standalone_commands_do_not_import_backend(tmp_path):
    # An installed backend must not be needed just to decode a fields file.
    blocker = tmp_path / "polymerfts"
    blocker.mkdir()
    (blocker / "__init__.py").write_text('raise RuntimeError("Unexpected backend import")\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    path = tmp_path / "fields.mat"
    write_encoded(path, parameter_file("_A"))
    commands = [
        [ROOT / "scripts/pscf/pscf_converter.py", "export", path, "-o", tmp_path / "pscf"],
        [ROOT / "scripts/analysis/renormalization.py", path, "--quadrature", "16"],
    ]
    fields, _, _ = field_file(("A", "A_real"), True, True)
    split = tmp_path / "split.mat"
    savemat(split, fields)
    commands.append([ROOT / "scripts/analysis/convert_into_auxiliary_fields.py", split,
                     "-o", tmp_path / "aux.mat"])
    for command in commands:
        done = subprocess.run([sys.executable, *map(str, command)], cwd=tmp_path, env=env,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
