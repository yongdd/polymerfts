"""Execute the cells of plot_2d_slice.ipynb on synthetic 3D results.

The slice is the plane of grid index slice_index along a, drawn with b along x and c at the angle
alpha, for every monomer type of the file. Each value fills the grid cell around its point, which the
solvers place at (j + 1/2)/n along b (j/n for a hexagonal-axes space group) and (k + 1/2)/n along c.
The checks read what is drawn.
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
CODE = '\n'.join(''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
                 if cell['cell_type'] == 'code')


def draw(tmp_path, monkeypatch, nx, lx, slice_index, layout='plain', types=('A',), **extra):
    """Write a 3D result, run the notebook's code on the plane slice_index.

    Returns {type: (corners, values, aspect)} of what it drew and {type: phi} written to the file.
    """
    rng = np.random.default_rng(7)
    phi = {name: rng.random(np.prod(nx)) for name in types}
    fields = {}
    for name, value in phi.items():
        fields.update({'plain': {'phi_' + name: value},
                       'clfts': {'phi_' + name + '_real': value, 'phi_' + name + '_imag': 0.1*value}}.get(layout, {}))
    if layout == 'struct':
        fields = {'phi': phi}
    savemat(tmp_path / 'fields_002000.mat', {'nx': np.array(nx), 'lx': np.array(lx, dtype=float), **fields, **extra})
    monkeypatch.chdir(tmp_path)
    code = CODE.replace('slice_index = 0 ', f'slice_index = {slice_index} ', 1)
    try:
        with warnings.catch_warnings():
            # fig.show() with Agg warns where a display could exist
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            namespace = {}
            exec(compile(code, str(NOTEBOOK), 'exec'), namespace)
        drawn = {}
        for ax in namespace['axes']:
            mesh = ax.collections[0]
            name = ax.get_title().split(',')[0].removeprefix('phi_')
            drawn[name] = (np.asarray(mesh.get_coordinates()), np.asarray(mesh.get_array()), ax.get_aspect())
    finally:
        plt.close('all')
    return drawn, {name: value.reshape(nx) for name, value in phi.items()}


# Hexagonal axes put the points along b at the nodes: SCFT results with a space group save its Hall
# number (488: P6_3/mmc; 459: R-3m on rhombohedral axes), which gives the axes
@pytest.mark.parametrize('source, angles, group, offset_b', [
    ('angles', [75.0, 95.0, 100.0], {}, 0.5), ('checkpoint_settings_json', [110.0, 95.0, 100.0], {}, 0.5),
    ('initial_params', [80.0, 95.0, 100.0], {}, 0.5), (None, [75.0, 95.0, 100.0], {}, 0.5),
    ('angles', [90.0, 90.0, 120.0], {'space_group_hall_number': 488}, 0.0),
    ('angles', [75.0, 75.0, 75.0], {'space_group_hall_number': 459}, 0.5)])
def test_plane_cells_surround_the_grid_points(tmp_path, monkeypatch, capsys, source, angles, group, offset_b):
    nx, lx = [4, 6, 5], [1.1, 1.3, 1.7]
    extra = {'angles': {'angles': np.array(angles)},
             'checkpoint_settings_json': {'checkpoint_settings_json': json.dumps({'angles': angles})},
             'initial_params': {'initial_params': {'nx': nx, 'angles': angles}},
             None: {}}[source]
    drawn, phi = draw(tmp_path, monkeypatch, nx, lx, 2, **extra, **group)
    corners, values, aspect = drawn['A']
    assert ('no lattice angles' in capsys.readouterr().out) == (source is None)
    j, k = np.meshgrid(np.arange(nx[1]), np.arange(nx[2]), indexing='ij')
    a = np.deg2rad(90.0 if source is None else angles[0])
    points = np.stack([(j + offset_b)/nx[1]*lx[1] + (k + 0.5)/nx[2]*lx[2]*np.cos(a),
                       (k + 0.5)/nx[2]*lx[2]*np.sin(a)], axis=-1)
    centres = (corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:])/4
    np.testing.assert_allclose(centres, points, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(values.reshape(nx[1], nx[2]), phi['A'][2])
    assert aspect == 1.0


@pytest.mark.parametrize('layout', ['clfts', 'struct'])
def test_concentration_of_other_file_layouts(tmp_path, monkeypatch, layout):
    drawn, phi = draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], 1, layout=layout,
                      angles=np.array([90.0, 90.0, 90.0]))
    np.testing.assert_array_equal(drawn['A'][1].reshape(4, 5), phi['A'][1])


@pytest.mark.parametrize('layout', ['plain', 'clfts', 'struct', 'listed'])
def test_every_monomer_type_of_a_file_without_A(tmp_path, monkeypatch, layout):
    # One plane for each type: the types the file lists, or those of its concentrations
    extra = {'monomer_types': ['B', 'C']} if layout == 'listed' else {}
    drawn, phi = draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], 2, layout=layout.replace('listed', 'plain'),
                      types=('B', 'C'), angles=np.array([90.0, 90.0, 90.0]), **extra)
    assert sorted(drawn) == ['B', 'C']
    for name in ('B', 'C'):
        np.testing.assert_array_equal(drawn[name][1].reshape(4, 5), phi[name][2])


def test_saved_angles_before_those_of_the_input(tmp_path, monkeypatch):
    # An SCFT result relaxed with relax_cell saves angles that differ from the input ones
    drawn, _ = draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], 0,
                    angles=np.array([70.0, 90.0, 90.0]),
                    initial_params={'nx': [3, 4, 5], 'angles': [80.0, 90.0, 90.0]})
    corners = drawn['A'][0]
    c = corners[0, -1] - corners[0, 0]          # the edge along c
    assert np.isclose(np.degrees(np.arctan2(c[1], c[0])), 70.0)


@pytest.mark.parametrize('slice_index', [-1, 3])
def test_slice_index_outside_the_grid(tmp_path, monkeypatch, slice_index):
    with pytest.raises(ValueError, match='slice_index'):
        draw(tmp_path, monkeypatch, [3, 4, 5], [1.0, 1.0, 1.0], slice_index)
