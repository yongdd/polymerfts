"""Execute the WTMD notebook reader against legacy and mapped pair records."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_wtmd_statistics.ipynb'
CELL = next(''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
            if cell['cell_type'] == 'code' and 'from scipy.io import loadmat\n' in cell['source'])


def read_statistics(tmp_path, monkeypatch, pair, extra):
    data = {name: np.arange(4.) for name in ('psi_range', 'u', 'up', 'I0')}
    data.update(dT=1., dpsi=.1, **extra)
    savemat(tmp_path / 'wtmd_statistics_1000000.mat', data)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    cell = CELL.replace('type_pair = ("A", "B")', 'type_pair = ' + repr(pair))
    exec(compile(cell, str(NOTEBOOK) + ':reader', 'exec'), namespace)
    return namespace['dH_psi']


def test_legacy_pair(tmp_path, monkeypatch):
    expected = np.arange(4.)
    actual = read_statistics(tmp_path, monkeypatch, ('A', 'B'), {'dH_psi_A_B': expected})
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('pair, value', [(('A', 'B_C'), 1.), (('A_B', 'C'), 7.)])
def test_mapped_pairs_remain_distinct(tmp_path, monkeypatch, pair, value):
    extra = {'pair_key_format_version': 1,
             'pair_keys_json': json.dumps({'pair__first': ['A', 'B_C'], 'pair__second': ['A_B', 'C']}),
             'dH_psi_pair__first': np.full(4, 1.), 'dH_psi_pair__second': np.full(4, 7.)}
    np.testing.assert_array_equal(read_statistics(tmp_path, monkeypatch, pair, extra), np.full(4, value))


@pytest.mark.parametrize('metadata', [
    {'pair_key_format_version': 1},
    {'pair_keys_json': '{"A_B":["A","B"]}'},
    {'pair_key_format_version': 2, 'pair_keys_json': '{"A_B":["A","B"]}'},
    {'pair_key_format_version': 1, 'pair_keys_json': '{}'},
    {'pair_key_format_version': 1, 'pair_keys_json': '{"one":["A","B"],"two":["A","B"]}'},
])
def test_invalid_mapping_does_not_fall_back_to_legacy(tmp_path, monkeypatch, metadata):
    with pytest.raises(ValueError, match='pair-name mapping|no unique entry'):
        read_statistics(tmp_path, monkeypatch, ('A', 'B'), {'dH_psi_A_B': np.ones(4), **metadata})
