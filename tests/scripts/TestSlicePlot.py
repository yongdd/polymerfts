"""Execute the cell of plot_2d_slice.ipynb on synthetic 3D results.

The slice is the plane of grid index slice_index along a, drawn with b along x and c at the angle
alpha. Each value fills the grid cell around its point, which the solvers place at (j + 1/2)/n along
b (j/n for a hexagonal-axes space group) and (k + 1/2)/n along c. The checks read what is drawn.
"""

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.io import savemat

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_2d_slice.ipynb'
CELL = ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][0]['source'])


def draw(tmp_path, monkeypatch, nx, lx, slice_index, layout='plain', **extra):
    """Write a 3D result, run the notebook's code on the plane slice_index; returns what it drew and phi_A."""
    phi = np.random.default_rng(7).random(np.prod(nx))
    fields = {'plain': {'phi_A': phi}, 'clfts': {'phi_A_real': phi, 'phi_A_imag': 0.1*phi},
              'struct': {'phi': {'A': phi}}}[layout]
    savemat(tmp_path / 'fields_002000.mat', {'nx': np.array(nx), 'lx': np.array(lx, dtype=float), **fields, **extra})
    monkeypatch.chdir(tmp_path)
    cell = CELL.replace('slice_index = 0 ', f'slice_index = {slice_index} ', 1)
    try:
        with warnings.catch_warnings():
            # fig.show() with Agg warns where a display could exist
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            namespace = {}
            exec(compile(cell, str(NOTEBOOK) + ':cell0', 'exec'), namespace)
        mesh = namespace['ax'].collections[0]
        drawn = (np.asarray(mesh.get_coordinates()), np.asarray(mesh.get_array()), namespace['ax'].get_aspect())
    finally:
        plt.close('all')
    return drawn, phi.reshape(nx)


@pytest.mark.parametrize('source, alpha, hall', [('angles', 75.0, None), ('checkpoint_settings_json', 110.0, None),
                                                 ('initial_params', 80.0, None), (None, 90.0, None),
                                                 ('angles', 90.0, 488), ('angles', 90.0, 459)])
def test_plane_cells_surround_the_grid_points(tmp_path, monkeypatch, capsys, source, alpha, hall):
    nx, lx = [4, 6, 5], [1.1, 1.3, 1.7]
    angles = [alpha, 95.0, 100.0] if hall is None else ([90.0, 90.0, 120.0] if hall == 488 else [75.0, 75.0, 75.0])
    alpha = angles[0]
    extra = {'angles': {'angles': np.array(angles)},
             'checkpoint_settings_json': {'checkpoint_settings_json': json.dumps({'angles': angles})},
             'initial_params': {'initial_params': {'nx': nx, 'angles': angles}},
             None: {}}[source]
    if hall is not None:
        extra['space_group_hall_number'] = hall
    if source is None:
        alpha = 90.0
    (corners, values, aspect), phi = draw(tmp_path, monkeypatch, nx, lx, 2, **extra)
    assert ('no lattice angles' in capsys.readouterr().out) == (source is None)
    offset_b = 0.0 if hall == 488 else 0.5      # 459 is R-3m on rhombohedral axes: cell centres
    j, k = np.meshgrid(np.arange(nx[1]), np.arange(nx[2]), indexing='ij')
    a = np.deg2rad(alpha)
    points = np.stack([(j + offset_b)/nx[1]*lx[1] + (k + 0.5)/nx[2]*lx[2]*np.cos(a),
                       (k + 0.5)/nx[2]*lx[2]*np.sin(a)], axis=-1)
    centres = (corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:])/4
    np.testing.assert_allclose(centres, points, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(values.reshape(nx[1], nx[2]), phi[2])
    assert aspect == 1.0


@pytest.mark.parametrize('layout', ['clfts', 'struct'])
def test_concentration_of_other_file_layouts(tmp_path, monkeypatch, layout):
    (_, values, _), phi = draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], 1, layout=layout,
                               angles=np.array([90.0, 90.0, 90.0]))
    np.testing.assert_array_equal(values.reshape(4, 5), phi[1])


def test_saved_angles_before_those_of_the_input(tmp_path, monkeypatch):
    # An SCFT result relaxed with relax_cell saves angles that differ from the input ones
    (corners, _, _), _ = draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], 0,
                              angles=np.array([70.0, 90.0, 90.0]),
                              initial_params={'nx': [3, 4, 5], 'angles': [80.0, 90.0, 90.0]})
    c = corners[0, -1] - corners[0, 0]          # the edge along c
    assert np.isclose(np.degrees(np.arctan2(c[1], c[0])), 70.0)


@pytest.mark.parametrize('slice_index', [-1, 3])
def test_slice_index_outside_the_grid(tmp_path, monkeypatch, slice_index):
    with pytest.raises(ValueError, match='slice_index'):
        draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], slice_index)
