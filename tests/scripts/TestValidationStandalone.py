"""Input validation must work before polymerfts or its native extension is built."""

import copy
from pathlib import Path
import runpy
import sys

import pytest


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.setitem(sys.modules, "polymerfts", None)
    monkeypatch.setitem(sys.modules, "polymerfts._core", None)
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/python/utils/validation.py"))


def parameters(validator="validate_scft_params"):
    """The input of the validator, with the keys that its method reads: SCFT, L-FTS or CL-FTS (without the
    saddle and compressor of L-FTS); each method ignores the others."""
    params = {"nx": [4], "lx": [2.0], "ds": 1e-6, "chain_model": "discrete",
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 2.0},
              "polymers": [{"volume_fraction": 1.0,
                            "blocks": [{"type": "A", "length": 1.0009e-6},
                                       {"type": "B", "length": 1e-6}]}]}
    if validator == "validate_scft_params":
        params.update(optimizer={"name": "am", "max_hist": 5, "start_error": 0.01, "mix_min": 0.1, "mix_init": 0.1},
                      relax_cell=False)
        return params
    params.update(langevin={"nbar": 1e4, "dt": 0.01, "max_step": 1}, verbose_level=0,
                  recording={"dir": "unused", "checkpoint_interval": 1, "sample_interval": 1,
                             "statistics_save_interval": 1})
    if validator == "validate_lfts_params":
        params.update(saddle={"max_iter": 10, "tolerance": 1e-8}, compressor={"name": "lr"})
    return params


@pytest.mark.parametrize("validator", ("validate_scft_params", "validate_lfts_params", "validate_clfts_params"))
def test_discrete_validation_without_backend(validation, validator):
    params = parameters(validator)
    original = copy.deepcopy(params)
    validation[validator](params)
    assert validation["polymer_block_lengths"](params, params["polymers"][0]) == (1e-6, 1e-6)
    assert params == original
    params["polymers"][0]["blocks"][0]["length"] = 1.5e-6
    with pytest.raises(validation["ValidationError"], match=r"polymers\[0\].blocks\[0\].length.*integer multiple"):
        validation[validator](params)


@pytest.mark.parametrize("validator", ("validate_scft_params", "validate_clfts_params"))
def test_charge_validation_without_backend(validation, validator):
    params = parameters(validator)
    params["smearing"] = {"type": "gaussian", "a_int": 0.1}
    params["electrostatics"] = {"charges": {"A": 1.0, "B": -1.0}, "coupling": 1.0,
                               "ions": {"X": {"valence": -1.0, "concentration": 0.0,
                                              "smearing_length": 0.1}}, "counterion": "X"}
    validation[validator](params)
    assert validation["net_polymer_charge"](params) == 0.0
    assert validation["point_ion_concentrations"](params) == {"X": 0.0}


def test_positive_half_ties_round_up_without_backend(validation):
    params = parameters()
    params["ds"] = 1e-10
    # The absolute acceptance tolerance admits these half ties; ties-to-even
    # would choose two beads for the second block instead of the native three.
    for block, length in zip(params["polymers"][0]["blocks"], (0.5e-10, 2.5e-10)):
        block["length"] = length
    assert validation["polymer_block_lengths"](params, params["polymers"][0]) == (1e-10, 3e-10)
