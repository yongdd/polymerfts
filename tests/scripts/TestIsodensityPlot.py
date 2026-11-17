"""Execute the cells of plot_3d_isodensity.ipynb on synthetic fields files.

The grid points of the solvers sit at the cell centres, (i + 1/2)/n along each lattice vector;
a hexagonal-axes space group samples a and b at the nodes, i/n (docs/theory/SpaceGroup.md,
PropagatorSolver.get_grid_points). A field written at those points must be drawn where it lives,
in the cell of its lattice vectors. The notebook draws one monomer type, the first of the file by default.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_3d_isodensity.ipynb'
CODE = '\n'.join(''.join(cell['source']) for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
                 if cell['cell_type'] == 'code')


def field(f):
    """A smooth periodic field of the fractional coordinates."""
    return 0.5 + 0.3*np.cos(2*np.pi*f[..., 0])*np.cos(2*np.pi*f[..., 1]) + 0.2*np.sin(2*np.pi*f[..., 2])


def draw(tmp_path, monkeypatch, nx, lx, offsets=(0.5, 0.5, 0.5), phi_key='phi_A', monomer_type=None, **extra):
    """Write the field at the grid points with the given offsets, then run the notebook's code."""
    f = np.stack(np.meshgrid(*[(np.arange(n) + o)/n for n, o in zip(nx, offsets)], indexing='ij'), axis=-1)
    phi = field(f).ravel()
    data = {'nx': np.array(nx), 'lx': np.array(lx, dtype=float), **extra}
    data[phi_key] = {'A': phi} if phi_key == 'phi' else phi
    savemat(tmp_path / 'fields_002000.mat', data)
    monkeypatch.chdir(tmp_path)
    code = CODE.replace('monomer_type = None ', f'monomer_type = {monomer_type!r} ', 1)
    result = {}
    exec(compile(code, str(NOTEBOOK), 'exec'), result)
    return result


def interpolation_error(result):
    """Largest difference between the drawn values and the field at the drawn points."""
    r = np.stack([result['X'], result['Y'], result['Z']], axis=-1)
    return np.max(np.abs(result['values'] - field(r @ np.linalg.inv(result['L']))))


@pytest.mark.parametrize('phi_key', ['phi_A', 'phi_A_real', 'phi'])
def test_orthogonal_cell_draws_the_grid_points(tmp_path, monkeypatch, capsys, phi_key):
    # phi_A_real: CL-FTS (its real part); phi: a MATLAB structure of older files
    nx, lx = [4, 3, 5], [1.0, 2.0, 3.0]
    result = draw(tmp_path, monkeypatch, nx, lx, phi_key=phi_key)
    assert 'no lattice angles' in capsys.readouterr().out
    expected = np.meshgrid(*[(np.arange(n) + 0.5)/n*l for n, l in zip(nx, lx)], indexing='ij')
    for axis, coordinate in zip(expected, ('X', 'Y', 'Z')):
        np.testing.assert_allclose(result[coordinate], axis, rtol=0, atol=1e-14)
    assert interpolation_error(result) < 1e-14
    # The outline: 12 edges between the corners of the box
    edges = result['edges']
    assert sum(row[0] is None for row in edges) == 12
    corners = {tuple(np.round(np.array(row, dtype=float), 12)) for row in edges if row[0] is not None}
    assert corners == {(i*lx[0], j*lx[1], k*lx[2]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}


@pytest.mark.parametrize('source', ['angles', 'checkpoint_settings_json', 'initial_params'])
def test_lattice_vectors_follow_the_angles_of_the_file(tmp_path, monkeypatch, source):
    # SCFT results, fields files and older fields files keep the angles in different places
    lx, angles = [1.3, 1.7, 2.1], [80.0, 95.0, 110.0]
    extra = {'angles': {'angles': np.array(angles)},
             'checkpoint_settings_json': {'checkpoint_settings_json': json.dumps({'angles': angles})},
             'initial_params': {'initial_params': {'nx': [8, 6, 4], 'angles': angles}}}[source]
    L = draw(tmp_path, monkeypatch, [8, 6, 4], lx, **extra)['L']
    np.testing.assert_allclose(np.linalg.norm(L, axis=1), lx, rtol=1e-14)
    between = [np.degrees(np.arccos(L[i] @ L[j]/(lx[i]*lx[j]))) for i, j in ((1, 2), (0, 2), (0, 1))]
    np.testing.assert_allclose(between, angles, rtol=1e-12)
    assert L[0, 1] == L[0, 2] == L[1, 2] == 0.0     # a along x, b in the xy plane


# Hexagonal axes put the points along a and b at the nodes: SCFT results with a space group save its
# Hall number, which gives the axes
@pytest.mark.parametrize('lx, angles, group, offsets', [
    ([1.3, 1.7, 2.1], [80.0, 95.0, 110.0], {}, (0.5, 0.5, 0.5)),     # triclinic, no group
    ([1.5, 1.5, 2.4], [90.0, 90.0, 120.0], {'space_group_hall_number': 488}, (0.0, 0.0, 0.5)),     # P6_3/mmc
    ([2.0, 2.0, 2.0], [75.0, 75.0, 75.0], {'space_group_hall_number': 459},
     (0.5, 0.5, 0.5)),                                                 # R-3m on rhombohedral axes
])
def test_non_orthogonal_cell_interpolates_the_periodic_field(tmp_path, monkeypatch, lx, angles, group, offsets):
    # The interpolation error is below 0.01; read with the other offsets along a and b, the field
    # is half a cell off, an error of about 0.04
    result = draw(tmp_path, monkeypatch, [24, 24, 24], lx, offsets=offsets, angles=np.array(angles), **group)
    assert interpolation_error(result) < 0.02
    assert result['X'].size <= 8*24**3


# The ends of the trigonal (430-461) and hexagonal (462-488) settings, and R settings on hexagonal
# (433, 460) and on rhombohedral axes (434, 461)
@pytest.mark.parametrize('hall, hexagonal', [(429, False), (430, True), (433, True), (434, False), (460, True),
                                             (461, False), (462, True), (488, True), (489, False)])
def test_hall_number_gives_the_axes_without_polymerfts(tmp_path, monkeypatch, hall, hexagonal):
    monkeypatch.setitem(sys.modules, 'polymerfts', None)        # import polymerfts fails
    result = draw(tmp_path, monkeypatch, [12, 12, 12], [1.5, 1.5, 2.4], offsets=(0.0, 0.0, 0.5),
                  angles=np.array([90.0, 90.0, 90.0]), space_group_hall_number=hall)
    assert result['hexagonal'] == hexagonal
    np.testing.assert_array_equal(result['offsets'], [0.0, 0.0, 0.5] if hexagonal else [0.5, 0.5, 0.5])


def test_hall_numbers_give_the_axes_of_the_solver(tmp_path, monkeypatch):
    # The reader (the first cell) of an SCFT result with each of the 530 Hall settings takes the
    # axes that polymerfts.SpaceGroup uses
    polymerfts = pytest.importorskip('polymerfts')
    reader = compile(''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][0]['source']),
                     str(NOTEBOOK), 'exec')
    monkeypatch.chdir(tmp_path)
    nx = [12, 12, 12]
    for hall in range(1, 531):
        savemat(tmp_path / 'fields_002000.mat', {'nx': np.array(nx), 'lx': np.ones(3), 'angles': np.full(3, 90.0),
                                                 'space_group_hall_number': hall, 'phi_A': np.zeros(np.prod(nx))})
        result = {}
        exec(reader, result)
        assert result['hexagonal'] == polymerfts.SpaceGroup(nx, hall).uses_hexagonal_axes(), hall


def test_large_box_around_a_long_tilted_cell_is_coarsened(tmp_path, monkeypatch, capsys):
    nx = [8, 8, 80]
    result = draw(tmp_path, monkeypatch, nx, [2.0, 2.0, 20.0], angles=np.array([60.0, 60.0, 90.0]))
    assert 'drawn with spacing' in capsys.readouterr().out
    assert result['X'].size <= 9*np.prod(nx)


def test_empty_input_angles_draw_an_orthogonal_cell(tmp_path, monkeypatch, capsys):
    result = draw(tmp_path, monkeypatch, [4, 3, 5], [1.0, 2.0, 3.0], initial_params={'nx': [4, 3, 5], 'angles': []})
    assert 'no lattice angles' in capsys.readouterr().out
    np.testing.assert_allclose(result['L'], np.diag([1.0, 2.0, 3.0]), rtol=0, atol=1e-15)


@pytest.mark.parametrize('monomer_type', [None, 'C'])
def test_file_without_A_draws_the_chosen_type(tmp_path, monkeypatch, monomer_type):
    # Types B and C: the first type of the file by default, or the type set in the notebook
    nx, lx = [4, 3, 5], [1.0, 2.0, 3.0]
    drawn, other = monomer_type or 'B', 'B' if monomer_type else 'C'
    result = draw(tmp_path, monkeypatch, nx, lx, phi_key='phi_' + drawn, monomer_type=monomer_type,
                  angles=np.array([90.0, 90.0, 90.0]), monomer_types=['B', 'C'], **{'phi_' + other: np.zeros(np.prod(nx))})
    assert result['monomer_type'] == drawn
    assert result['fig'].layout.title.text == 'phi_' + drawn
    assert interpolation_error(result) < 1e-14


@pytest.mark.parametrize('suffix', ['.mat', '.json'])
@pytest.mark.parametrize('symbol, hall, lx, angles', [('P6_3/mmc', 488, [2.0, 2.0, 3.0], [90.0, 90.0, 120.0]),
                                                      ('R-3m', 459, [2.0, 2.0, 2.0], [80.0, 80.0, 80.0])])
def test_scft_results_place_the_points_of_the_solver(tmp_path, monkeypatch, suffix, symbol, hall, lx, angles):
    # The angles and the Hall number saved by SCFT give the grid points of PropagatorSolver.get_grid_points
    polymerfts = pytest.importorskip('polymerfts')
    if NOTEBOOK.parents[2] not in Path(polymerfts.__file__).resolve().parents:
        pytest.skip('needs the staged build/python of this checkout on PYTHONPATH')
    platform = next((p for p in polymerfts.PlatformSelector.avail_platforms() if p.startswith('cpu')), None)
    if platform is None:
        pytest.skip('needs a CPU platform')
    nx = [12, 12, 12]
    params = {'nx': nx, 'lx': lx, 'angles': angles, 'chain_model': 'continuous', 'ds': 0.1, 'platform': platform,
              'segment_lengths': {'A': 1.0, 'B': 1.0}, 'chi_n': {'A,B': 15.0}, 'relax_cell': False,
              'polymers': [{'volume_fraction': 1.0, 'blocks': [{'type': 'A', 'length': 0.3}, {'type': 'B', 'length': 0.7}]}],
              'space_group': {'symbol': symbol, 'hall_number': hall}, 'max_iter': 1, 'tolerance': 1e-8,
              'optimizer': {'name': 'am', 'max_hist': 5, 'start_error': 0.01, 'mix_min': 0.1, 'mix_init': 0.1}}
    w = np.random.default_rng(11).normal(0.0, 5.0, nx)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = polymerfts.SCFT(params)
        sim.run(initial_fields={'A': w, 'B': -w})
        sim.save_results('fields_002000' + suffix)
    saved = (loadmat('fields_002000.mat', squeeze_me=True) if suffix == '.mat'
             else json.loads(Path('fields_002000.json').read_text()))
    assert int(saved['space_group_hall_number']) == hall and 'uses_hexagonal_axes' not in saved
    result = {}
    exec(compile(CODE.replace('"fields_002000.mat"', f'"fields_002000{suffix}"', 1), str(NOTEBOOK), 'exec'), result)
    np.testing.assert_array_equal(result['offsets'], [0.0, 0.0, 0.5] if hall == 488 else [0.5, 0.5, 0.5])
    f = np.stack(np.meshgrid(*[(np.arange(n) + o)/n for n, o in zip(nx, result['offsets'])], indexing='ij'), axis=-1)
    np.testing.assert_allclose(f @ result['L'], np.stack(sim.prop_solver.get_grid_points(), axis=-1), rtol=0, atol=1e-12)
