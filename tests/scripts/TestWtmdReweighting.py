"""Execute the parameter and reader cells of the WTMD chi N reweighting notebook on a statistics file."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'scripts/analysis/plot_wtmd_chib_AB.ipynb'
CELLS = [''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
         if cell['cell_type'] == 'code']
RENORMALIZATION = runpy.run_path(str(ROOT / 'scripts/analysis/renormalization.py'))


def run_notebook(tmp_path, monkeypatch, settings):
    """The import, parameter and reader cells with the given parameter lines appended."""
    data_dir = tmp_path / 'data_simulation'
    data_dir.mkdir()
    data = {name: np.arange(4.) for name in ('psi_range', 'u', 'up', 'I0')}
    data.update(dT=1., dpsi=.1, nx=[16, 16, 16], lx=[4.36, 4.36, 4.36], nbar=1e4,
                dH_psi_A_B=np.full(4, 3.), dH_psi_A_C=np.full(4, 5.))
    savemat(data_dir / 'wtmd_statistics_001000.mat', data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.path', list(__import__('sys').path))
    namespace = {}
    parameters = CELLS[1] + '\n' + settings + f'\nlangevin_step = 1000\ndata_dir = "data_simulation"\npolymerfts_dir = {str(ROOT)!r}\n'
    for name, cell in (('imports', CELLS[0]), ('parameters', parameters), ('reader', CELLS[2])):
        exec(compile(cell, f'{NOTEBOOK}:{name}', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('chain_model, n_segments, zeta_n', [('discrete', 90, None), ('continuous', None, 50.)])
def test_z_inf_from_the_statistics_file(chain_model, n_segments, zeta_n, tmp_path, monkeypatch):
    settings = f'chain_model = {chain_model!r}\nn_segments = {n_segments!r}\nzeta_n = {zeta_n!r}'
    namespace = run_notebook(tmp_path, monkeypatch, settings)
    expected = RENORMALIZATION['z_inf'](chain_model, [16, 16, 16], [4.36] * 3, 1e4, n_segments=n_segments,
                                        zeta_n=zeta_n)
    assert namespace['z_inf'] == pytest.approx(expected, rel=1e-12)
    assert namespace['bare_dchiN'] == pytest.approx(namespace['dchiN'] / expected, rel=1e-12)
    np.testing.assert_array_equal(namespace['dH_psi'], np.full(4, 3.))


def test_own_z_inf_and_pair(tmp_path, monkeypatch):
    namespace = run_notebook(tmp_path, monkeypatch, 'z_inf = 0.5\ntype_pair = ("C", "A")')
    assert namespace['z_inf'] == 0.5
    np.testing.assert_array_equal(namespace['dH_psi'], np.full(4, 5.))
