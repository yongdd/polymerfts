"""Execute the WTMD notebook reader on the pair records of a statistics file."""

import json
from pathlib import Path

import numpy as np
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


def test_pair(tmp_path, monkeypatch):
    expected = np.arange(4.)
    actual = read_statistics(tmp_path, monkeypatch, ('A', 'B'), {'dH_psi_A_B': expected})
    np.testing.assert_array_equal(actual, expected)


def test_pair_of_underscore_names(tmp_path, monkeypatch):
    expected = np.full(4, 7.)
    actual = read_statistics(tmp_path, monkeypatch, ('B_C', 'A'), {'dH_psi_A_B_C': expected})
    np.testing.assert_array_equal(actual, expected)
