"""Execute the cell of plot_3d_isodensity.ipynb on synthetic fields files.

The grid points of the solvers sit at the cell centres, (i + 1/2)/n along each lattice vector;
a hexagonal-axes space group samples a and b at the nodes, i/n (docs/theory/SpaceGroup.md,
PropagatorSolver.get_grid_points). A field written at those points must be drawn where it lives,
in the cell of its lattice vectors.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_3d_isodensity.ipynb'
CELL = ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][0]['source'])


def field(f):
    """A smooth periodic field of the fractional coordinates."""
    return 0.5 + 0.3*np.cos(2*np.pi*f[..., 0])*np.cos(2*np.pi*f[..., 1]) + 0.2*np.sin(2*np.pi*f[..., 2])


def draw(tmp_path, monkeypatch, nx, lx, offsets=(0.5, 0.5, 0.5), phi_key='phi_A', **extra):
    """Write the field at the grid points with the given offsets, then run the notebook's code."""
    f = np.stack(np.meshgrid(*[(np.arange(n) + o)/n for n, o in zip(nx, offsets)], indexing='ij'), axis=-1)
    phi = field(f).ravel()
    data = {'nx': np.array(nx), 'lx': np.array(lx, dtype=float), **extra}
    data[phi_key] = {'A': phi} if phi_key == 'phi' else phi
    savemat(tmp_path / 'fields_002000.mat', data)
    monkeypatch.chdir(tmp_path)
    result = {}
    exec(compile(CELL, str(NOTEBOOK) + ':cell0', 'exec'), result)
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


@pytest.mark.parametrize('lx, angles, hall, offsets', [
    ([1.3, 1.7, 2.1], [80.0, 95.0, 110.0], None, (0.5, 0.5, 0.5)),     # triclinic, no group
    ([1.5, 1.5, 2.4], [90.0, 90.0, 120.0], 488, (0.0, 0.0, 0.5)),      # P6_3/mmc: a and b at the nodes
    ([2.0, 2.0, 2.0], [75.0, 75.0, 75.0], 459, (0.5, 0.5, 0.5)),       # R-3m on rhombohedral axes
])
def test_non_orthogonal_cell_interpolates_the_periodic_field(tmp_path, monkeypatch, lx, angles, hall, offsets):
    # The interpolation error is below 0.01; read with the other offsets along a and b, the field
    # is half a cell off, an error of about 0.04
    extra = {'angles': np.array(angles)}
    if hall is not None:
        extra['space_group_hall_number'] = hall
    result = draw(tmp_path, monkeypatch, [24, 24, 24], lx, offsets=offsets, **extra)
    assert interpolation_error(result) < 0.02
    assert result['X'].size <= 8*24**3


def test_large_box_around_a_long_tilted_cell_is_coarsened(tmp_path, monkeypatch, capsys):
    nx = [8, 8, 80]
    result = draw(tmp_path, monkeypatch, nx, [2.0, 2.0, 20.0], angles=np.array([60.0, 60.0, 90.0]))
    assert 'drawn with spacing' in capsys.readouterr().out
    assert result['X'].size <= 9*np.prod(nx)
