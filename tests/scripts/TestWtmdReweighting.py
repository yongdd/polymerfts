"""Execute every cell of the WTMD chi N reweighting notebook on a statistics file and a fields file of the
same run: z_inf from the settings of the fields file, the reweighted free energy and the phase areas."""

import json
from pathlib import Path
import runpy
import warnings

import matplotlib
import numpy as np
import pytest
from scipy.io import savemat

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'scripts/analysis/plot_wtmd_chib_AB.ipynb'
CELLS = [''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
         if cell['cell_type'] == 'code']
RENORMALIZATION = runpy.run_path(str(ROOT / 'scripts/analysis/renormalization.py'))
NX, LX, NBAR = [16, 16, 16], [4.36, 4.36, 4.36], 1e4


def write_run(tmp_path, settings, statistics=None):
    """A statistics file with a double-well P(psi) and a fields file of the same run with the given settings."""
    data_dir = tmp_path / 'data_simulation'
    data_dir.mkdir()
    psi = np.linspace(0, 1, 50)
    data = dict(psi_range=psi, u=-np.sin(3*psi), up=-3*np.cos(3*psi), I0=np.exp(-(psi - .5)**2/.1), dT=5.,
                dpsi=psi[1] - psi[0], nx=NX, lx=LX, nbar=NBAR, dH_psi_A_B=psi**2, dH_psi_A_C=np.full(50, 5.))
    data.update(statistics or {})
    savemat(data_dir / 'wtmd_statistics_001000.mat', data)
    params = {'nx': NX, 'lx': LX, 'segment_lengths': {'A': 1., 'B': 1., 'C': 1.}, 'langevin': {'nbar': NBAR, 'dt': 1.},
              **settings}
    savemat(data_dir / 'fields_000500.mat', {'initial_params': params, 'nx': NX, 'lx': LX, 'nbar': NBAR,
                                             'chain_model': params['chain_model'], 'ds': params['ds']})


def run_notebook(tmp_path, monkeypatch, parameters=''):
    """Every code cell, with the given parameter lines appended to the parameter cell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.path', list(__import__('sys').path))
    cells = list(CELLS)
    cells[1] += (f'\nlangevin_step = 1000\ndata_dir = "data_simulation"\npolymerfts_dir = {str(ROOT)!r}\n'
                 + parameters)
    namespace = {}
    try:
        with warnings.catch_warnings():
            # plt.show() with Agg warns where a display could exist
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            for i, cell in enumerate(cells):
                exec(compile(cell, f'{NOTEBOOK}:cell {i + 1}', 'exec'), namespace)
    finally:
        plt.close('all')
    return namespace


@pytest.mark.parametrize('settings, z_inf_args', [
    ({'chain_model': 'discrete', 'ds': 1/90}, ('discrete', 90, None, None)),
    ({'chain_model': 'continuous', 'ds': 1/50, 'zeta_n': 50., 'smearing': {'type': 'gaussian', 'a_int': .05}},
     ('continuous', 50, 50., {'type': 'gaussian', 'a_int': .05}))])
def test_reweighting_with_z_inf_of_the_run(settings, z_inf_args, tmp_path, monkeypatch):
    write_run(tmp_path, settings)
    ns = run_notebook(tmp_path, monkeypatch)
    model, n_segments, zeta_n, smearing = z_inf_args
    z_inf = RENORMALIZATION['z_inf'](model, NX, LX, NBAR, n_segments=n_segments, zeta_n=zeta_n, smearing=smearing)
    assert ns['z_inf'] == pytest.approx(z_inf, rel=1e-12)
    np.testing.assert_array_equal(ns['dH_psi'], np.linspace(0, 1, 50)**2)
    # F = -(1 + 1/dT) U, shifted by dF/d(chi N) times the shift of the bare chi_b N
    np.testing.assert_allclose(ns['F'], 1.2*np.sin(3*np.linspace(0, 1, 50)), rtol=1e-14)
    np.testing.assert_allclose(ns['F_reweighted'], ns['F'] + ns['y_interp']*ns['dchiN']/z_inf, rtol=1e-12)
    # The two phases share the probability: areas of P(psi) summed over the bins of width dpsi
    assert 0 < ns['area1'] < 1 and ns['area1'] + ns['area2'] == pytest.approx(1, rel=1e-12)


def test_pair(tmp_path, monkeypatch):
    write_run(tmp_path, {'chain_model': 'discrete', 'ds': 1/90})
    ns = run_notebook(tmp_path, monkeypatch, 'type_pair = ("C", "A")')
    np.testing.assert_array_equal(ns['dH_psi'], np.full(50, 5.))


@pytest.mark.parametrize('settings, statistics, error, message', [
    ({'angles': [90., 90., 120.]}, {}, ValueError, 'orthogonal boxes'),
    ({'segment_lengths': {'A': 1., 'B': 1.2}}, {}, ValueError, 'conformationally symmetric'),
    ({}, {'nx': [16, 16, 32]}, ValueError, 'nx of'),
    ({}, {'nbar': 1e5}, ValueError, 'nbar of'),
    (None, {}, FileNotFoundError, r'No fields_\*.mat')])
def test_settings_of_another_run_or_none(settings, statistics, error, message, tmp_path, monkeypatch):
    write_run(tmp_path, {'chain_model': 'discrete', 'ds': 1/90, **(settings or {})}, statistics)
    if settings is None:
        (tmp_path / 'data_simulation' / 'fields_000500.mat').unlink()
    with pytest.raises(error, match=message):
        run_notebook(tmp_path, monkeypatch)
