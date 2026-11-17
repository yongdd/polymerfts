"""Execute the cells of plot_2d_density.ipynb on synthetic 2D results.

Each value fills the grid cell around its point, which the solvers place at (i + 1/2)/n along each
lattice vector (a along x, b in the xy plane at the angle gamma). The checks read what is drawn: the
corners of the meshes, their values against the data written to the file, and the aspect. The
concentrations are read from SCFT and L-FTS files, CL-FTS files (the real parts) and the phi structure
of older files. The three density notebooks read files with the same first cell.
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


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_2d_density.ipynb'


def code_cells(notebook):
    return [''.join(cell['source']) for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']
            if cell['cell_type'] == 'code']


CODE = '\n'.join(code_cells(NOTEBOOK))


def draw(tmp_path, monkeypatch, nx, lx, layout='plain', **extra):
    """Write a 2D result where the notebook reads it, run the notebook's code and return what it drew.

    layout: 'plain' (phi_A, phi_B), 'clfts' (real and imaginary parts) or 'struct' (the phi
    structure of older files). Returns {name: (corners, values, aspect)} of the two plots and the
    concentrations written to the file.
    """
    rng = np.random.default_rng(5)
    phi = {name: rng.random(np.prod(nx)) for name in ('A', 'B')}
    if layout == 'plain':
        fields = {'phi_' + name: value for name, value in phi.items()}
    elif layout == 'clfts':
        fields = {}
        for name, value in phi.items():
            fields['phi_' + name + '_real'], fields['phi_' + name + '_imag'] = value, 0.1*value
    else:
        fields = {'phi': phi}
    (tmp_path / 'phases').mkdir()
    savemat(tmp_path / 'phases' / 'C2D.mat', {'nx': np.array(nx), 'lx': np.array(lx, dtype=float), **fields, **extra})
    monkeypatch.chdir(tmp_path)
    try:
        with warnings.catch_warnings():
            # fig.show() with Agg warns where a display could exist
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            namespace = {}
            exec(compile(CODE, str(NOTEBOOK), 'exec'), namespace)
        drawn = {}
        for ax in namespace['axes']:
            mesh = ax.collections[0]
            drawn[ax.get_title()] = (np.asarray(mesh.get_coordinates()), np.asarray(mesh.get_array()), ax.get_aspect())
    finally:
        plt.close('all')
    return drawn, phi


@pytest.mark.parametrize('source, gamma', [('angles', 120.0), ('checkpoint_settings_json', 75.0),
                                           ('initial_params [gamma]', 120.0), ('initial_params', 105.0),
                                           ('initial_params empty', 90.0), (None, 90.0)])
def test_cells_surround_the_grid_points(tmp_path, monkeypatch, capsys, source, gamma):
    nx, lx = [6, 4], [1.2, 1.5]
    extra = {'angles': {'angles': np.array([90.0, 90.0, gamma])},
             'checkpoint_settings_json': {'checkpoint_settings_json': json.dumps({'angles': [90.0, 90.0, gamma]})},
             'initial_params [gamma]': {'initial_params': {'nx': nx, 'angles': [gamma]}},
             'initial_params': {'initial_params': {'nx': nx, 'angles': [90.0, 90.0, gamma]}},
             'initial_params empty': {'initial_params': {'nx': nx, 'angles': []}},
             None: {}}[source]
    drawn, written = draw(tmp_path, monkeypatch, nx, lx, **extra)
    assert ('no lattice angles' in capsys.readouterr().out) == (source in (None, 'initial_params empty'))
    g = np.deg2rad(gamma)
    i, j = np.meshgrid(np.arange(nx[0]), np.arange(nx[1]), indexing='ij')
    points = np.stack([(i + 0.5)/nx[0]*lx[0] + (j + 0.5)/nx[1]*lx[1]*np.cos(g),
                       (j + 0.5)/nx[1]*lx[1]*np.sin(g)], axis=-1)
    for name in ('A', 'B'):
        corners, values, aspect = drawn['phi_' + name]
        centres = (corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:])/4
        np.testing.assert_allclose(centres, points, rtol=0, atol=1e-14)
        # The cell of grid point (i, j) carries the value written at (i, j), in the C order of the file
        np.testing.assert_array_equal(values.reshape(nx), written[name].reshape(nx))
        assert aspect == 1.0


@pytest.mark.parametrize('layout', ['plain', 'clfts', 'struct'])
def test_concentrations_of_every_file_layout(tmp_path, monkeypatch, layout):
    # CL-FTS: the real parts are drawn
    drawn, written = draw(tmp_path, monkeypatch, [5, 3], [1.0, 1.2], layout=layout,
                          angles=np.array([90.0, 90.0, 90.0]))
    for name in ('A', 'B'):
        np.testing.assert_array_equal(drawn['phi_' + name][1].reshape(5, 3), written[name].reshape(5, 3))


def test_every_monomer_type_of_the_file(tmp_path, monkeypatch):
    # Types A and C: one plot for each type the file lists
    nx = [5, 3]
    phi = {name: np.random.default_rng(9).random(15) + i for i, name in enumerate(('A', 'C'))}
    (tmp_path / 'phases').mkdir()
    savemat(tmp_path / 'phases' / 'C2D.mat', {'nx': np.array(nx), 'lx': np.array([1.0, 1.2]), 'angles': np.array([90.0, 90.0, 90.0]),
                                              'monomer_types': ['A', 'C'], 'phi_A': phi['A'], 'phi_C': phi['C']})
    monkeypatch.chdir(tmp_path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            namespace = {}
            exec(compile(CODE, str(NOTEBOOK), 'exec'), namespace)
        drawn = {ax.get_title(): np.asarray(ax.collections[0].get_array()) for ax in namespace['axes']}
    finally:
        plt.close('all')
    assert sorted(drawn) == ['phi_A', 'phi_C']
    for name in ('A', 'C'):
        np.testing.assert_array_equal(drawn['phi_' + name].reshape(nx), phi[name].reshape(nx))


@pytest.mark.parametrize('suffix', ['.json', '.yaml'])
def test_json_and_yaml_results(tmp_path, monkeypatch, suffix):
    # SCFT saves .json and .yaml results with lists, the angles and uses_hexagonal_axes
    nx, lx, gamma = [5, 3], [1.0, 1.2], 110.0
    phi = {name: np.random.default_rng(3).random(15) + i for i, name in enumerate(('X', 'Y'))}
    data = {'nx': nx, 'lx': lx, 'angles': [90.0, 90.0, gamma], 'uses_hexagonal_axes': False,
            'monomer_types': ['X', 'Y'], 'initial_params': {'angles': [90.0, 90.0, 60.0]},
            **{'phi_' + name: value.tolist() for name, value in phi.items()}}
    (tmp_path / 'phases').mkdir()
    text = json.dumps(data) if suffix == '.json' else pytest.importorskip('yaml').safe_dump(data)
    (tmp_path / 'phases' / ('C2D' + suffix)).write_text(text)
    monkeypatch.chdir(tmp_path)
    code = CODE.replace('"phases/C2D.mat"', f'"phases/C2D{suffix}"', 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*non-interactive', category=UserWarning)
            namespace = {}
            exec(compile(code, str(NOTEBOOK), 'exec'), namespace)
        drawn = {ax.get_title(): (np.asarray(ax.collections[0].get_coordinates()), np.asarray(ax.collections[0].get_array()))
                 for ax in namespace['axes']}
    finally:
        plt.close('all')
    assert sorted(drawn) == ['phi_X', 'phi_Y']
    for name in ('X', 'Y'):
        corners, values = drawn['phi_' + name]
        np.testing.assert_array_equal(values.reshape(nx), phi[name].reshape(nx))
        b = corners[0, -1] - corners[0, 0]      # the edge along b, at the saved angle gamma
        assert np.isclose(np.degrees(np.arctan2(b[1], b[0])), gamma)


def test_the_density_notebooks_read_files_with_the_same_cell():
    # Only the file name differs, so the checks of each notebook cover the reader of all three:
    # JSON and YAML here, hexagonal axes and SCFT results in TestSlicePlot and TestIsodensityPlot
    readers = []
    for name in ('plot_2d_density', 'plot_2d_slice', 'plot_3d_isodensity'):
        lines = code_cells(NOTEBOOK.with_name(name + '.ipynb'))[0].split('\n')
        assert sum(line.startswith('file_name = "') for line in lines) == 1
        readers.append([line for line in lines if not line.startswith('file_name = "')])
    assert readers[0] == readers[1] == readers[2]
