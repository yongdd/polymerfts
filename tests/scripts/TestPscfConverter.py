"""PSCF conversion format and CLI regressions; no compiled backend or simulation."""

import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / 'scripts/pscf/pscf_converter.py'
TOOL = runpy.run_path(str(CONVERTER))
# spglib 2.7+ warns on each call that its error handling will raise; the converter passes valid Hall numbers
pytestmark = pytest.mark.filterwarnings('ignore:Set OLD_ERROR_HANDLING:DeprecationWarning')


def params():
    return {'nx': [2, 3, 4], 'lx': [3., 4., 5.], 'chain_model': 'continuous',
            'ds': .01, 'segment_lengths': {'Zed': 1., 'long_B': 1.2},
            'chi_n': {'Zed,long_B': 20.}, 'relax_cell': True,
            'polymers': [{'volume_fraction': 1., 'blocks': [
                {'type': 'Zed', 'length': .4}, {'type': 'long_B', 'length': .6}]}],
            'optimizer': {'name': 'am', 'max_hist': 20}, 'max_iter': 1234, 'tolerance': 1e-7}


def result(p=None):
    p = params() if p is None else p
    n = math.prod(p['nx'])
    field = np.arange(n, dtype=float).reshape(p['nx']) + .125
    return {'initial_params': p, 'nx': p['nx'], 'lx': p['lx'],
            **{'w_' + name: (i + 1) * field for i, name in enumerate(p['segment_lengths'])}}


def rgrid(path, nx, values=None, lattice=None, cell=None, group='P_1', inline=False):
    dim = len(nx)
    lattice = lattice or {1: 'lamellar', 2: 'rectangular', 3: 'orthorhombic'}[dim]
    cell = cell if cell is not None else list(range(2, dim + 2))
    lines = ['format 1 0']
    for key, value in [('dim', str(dim)), ('crystal_system', lattice), ('N_cell_param', str(len(cell))),
                       ('cell_param', ' '.join(map(str, cell)))]:
        lines.extend([key + ' ' + value] if inline else [key, value])
    if group is not None:
        lines.extend(['group_name', "'" + group + "'"])
    lines.extend(['N_monomer 2', 'mesh ' + ' '.join(map(str, nx))])
    if values is None:
        # Explicit PSCF nested loops: x fastest, independently of numpy reshape.
        values = [sum(coord * 10**axis for axis, coord in enumerate(reversed(index)))
                  for index in itertools.product(*(range(n) for n in reversed(nx)))]
    lines.extend(f'{x} {-x}' for x in values)
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.mark.parametrize('nx', ([5], [3, 2], [2, 3, 4]))
@pytest.mark.parametrize('inline', (False, True))
def test_read_keeps_axis_order_and_first_index_fastest(tmp_path, nx, inline):
    path = rgrid(tmp_path / 'omega', nx, inline=inline)
    data = TOOL['read_rgrid'](path)
    assert data['nx'] == nx
    assert data['lx'] == list(range(2, len(nx) + 2))
    for index in np.ndindex(*nx):
        assert data['fields'][(0,) + index] == sum(coord * 10**axis for axis, coord in enumerate(index))


def test_write_order_is_independently_known(tmp_path):
    w = np.array([[0., 10., 20.], [1., 11., 21.]])
    path = tmp_path / 'omega'
    TOOL['write_rgrid'](path, [2, 3], 'rectangular', [3., 4.], np.stack([w, -w]))
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('mesh')) + 1
    numbers = np.loadtxt(lines[start:])
    np.testing.assert_array_equal(numbers[:, 0], [0, 1, 10, 11, 20, 21])
    np.testing.assert_array_equal(numbers[:, 1], -numbers[:, 0])


@pytest.mark.parametrize('lattice, dim, cell, lengths, angles', (
    ('lamellar', 1, [3], [3], [90, 90, 90]),
    ('square', 2, [3], [3, 3], [90, 90, 90]),
    ('rectangular', 2, [3, 4], [3, 4], [90, 90, 90]),
    ('hexagonal', 2, [3], [3, 3], [90, 90, 120]),
    ('rhombic', 2, [3, math.pi/3], [3, 3], [90, 90, 60]),
    ('oblique', 2, [3, 4, math.pi/3], [3, 4], [90, 90, 60]),
    ('cubic', 3, [3], [3, 3, 3], [90, 90, 90]),
    ('tetragonal', 3, [3, 4], [3, 3, 4], [90, 90, 90]),
    ('orthorhombic', 3, [3, 4, 5], [3, 4, 5], [90, 90, 90]),
    ('hexagonal', 3, [3, 4], [3, 3, 4], [90, 90, 120]),
    ('rhombohedral', 3, [3, math.pi/3], [3, 3, 3], [60, 60, 60]),
    ('monoclinic', 3, [3, 4, 5, math.pi/3], [3, 4, 5], [90, 60, 90]),
    ('triclinic', 3, [3, 4, 5, 0., 0., math.pi/2], [3, 4, 5], [90, 90, 90]),
))
def test_all_pscf_lattice_parameters(lattice, dim, cell, lengths, angles):
    lx, a = TOOL['lattice_geometry'](lattice, cell, dim)
    np.testing.assert_allclose(lx, lengths, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, angles, rtol=0, atol=1e-12)


def test_triclinic_euler_angles_preserve_the_metric():
    p = params()
    p.update(crystal_system='Triclinic', angles=[72., 83., 66.])
    lattice, cell = TOOL['choose_lattice'](p)
    assert lattice == 'triclinic'
    a, b, c, phi, theta, gamma = cell
    vectors = np.array([[a, 0, 0], [b*math.cos(gamma), b*math.sin(gamma), 0],
                        [c*math.sin(theta)*math.cos(phi), c*math.sin(theta)*math.sin(phi), c*math.cos(theta)]])
    alpha, beta, gamma = np.deg2rad(p['angles'])
    expected = np.array([[a*a, a*b*math.cos(gamma), a*c*math.cos(beta)],
                         [a*b*math.cos(gamma), b*b, b*c*math.cos(alpha)],
                         [a*c*math.cos(beta), b*c*math.cos(alpha), c*c]])
    np.testing.assert_allclose(vectors @ vectors.T, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('hall, symbol, lengths, angles, expected', (
    (530, 'Ia-3d', [3., 3., 3.], [90., 90., 90.], 'cubic'),
    (419, 'P4_2/mnm', [3., 3., 5.], [90., 90., 90.], 'tetragonal'),
    (488, 'P6_3/mmc', [3., 3., 5.], [90., 90., 120.], 'hexagonal'),
))
def test_space_group_identifies_cell_constraints(hall, symbol, lengths, angles, expected):
    pytest.importorskip('spglib')
    p = params()
    p.update(space_group={'symbol': symbol, 'hall_number': hall}, lx=lengths, angles=angles)
    assert TOOL['choose_lattice'](p)[0] == expected


def test_space_group_takes_precedence_over_explicit_crystal_system():
    pytest.importorskip('spglib')
    p = params()
    p.update(space_group={'symbol': 'Ia-3d', 'hall_number': 530},
             crystal_system='Orthorhombic', lx=[3., 3., 3.])
    assert TOOL['choose_lattice'](p)[0] == 'cubic'


@pytest.mark.parametrize('nx, angles', (([4], None), ([4], []), ([4], [90.]),
                                      ([4, 3], None), ([4, 3], []), ([4, 3, 2], None)))
def test_default_and_lower_dimensional_angles(nx, angles):
    p = params()
    p.update(nx=nx, lx=[3.] * len(nx), angles=angles)
    normalized, _, lattice, _ = TOOL['prepare_export'](result(p))
    assert normalized['angles'] == [90., 90., 90.]
    assert lattice == {1: 'lamellar', 2: 'rectangular', 3: 'orthorhombic'}[len(nx)]


def test_skewed_2d_fixed_angle_relaxation_is_not_changed():
    p = params()
    p.update(nx=[4, 3], lx=[3., 4.], angles=[60.], crystal_system='Orthorhombic')
    with pytest.raises(ValueError, match='fixed gamma cannot be transferred'):
        TOOL['prepare_export'](result(p))
    # With a fixed cell, the oblique metric can be transferred without changing its constraints.
    p['relax_cell'] = False
    assert TOOL['prepare_export'](result(p))[2] == 'oblique'


@pytest.mark.parametrize('key', ('initial_conditions', 'grafting_points'))
def test_grafting_and_custom_propagators_are_not_dropped(key):
    p = params()
    p['polymers'][0][key] = {0: 'G'}
    with pytest.raises(ValueError, match='initial propagator conditions'):
        TOOL['prepare_export'](result(p))


@pytest.mark.parametrize('hall, symbol, angles, relax, lattice', (
    (57, 'P2/m', [90., 100., 90.], True, 'monoclinic'), (58, 'P2/m', [90., 90., 100.], False, 'triclinic'),
    (5, 'P2', [100., 90., 90.], False, 'triclinic'), (58, 'P2/m', [90., 90., 100.], True, 'unique axis c'),
    (5, 'P2', [100., 90., 90.], True, 'unique axis a'),
    (42, 'Cc', [90., 100., 90.], True, 'monoclinic'), (48, 'Cc', [90., 90., 100.], True, 'unique axis c')))
def test_monoclinic_unique_axis(hall, symbol, angles, relax, lattice):
    # PSCF's monoclinic lattice has the unique axis b; the unique axes a and c keep a fixed cell as
    # triclinic and refuse box relaxation. Hall 42 and 48 are the cell choices -b1 and -c1 of Cc
    pytest.importorskip('spglib')
    p = params()
    p.update(angles=angles, relax_cell=relax, space_group={'symbol': symbol, 'hall_number': hall})
    if lattice.startswith('unique axis'):
        with pytest.raises(ValueError, match=lattice):
            TOOL['prepare_export'](result(p))
    else:
        assert TOOL['prepare_export'](result(p))[2] == lattice


@pytest.mark.parametrize('hall, message', ((0, 'positive'), (531, 'between 1 and 530')))
def test_hall_number_range(hall, message):
    pytest.importorskip('spglib')
    p = params()
    p['space_group'] = {'symbol': 'P1', 'hall_number': hall}
    with pytest.raises(ValueError, match=message):
        TOOL['choose_lattice'](p)


@pytest.mark.parametrize('group, outcome', (
    ({'symbol': 'Ia-3d'}, 'cubic'),                         # one Hall setting: the symbol suffices
    ({'symbol': ' Ia-3d\t'}, 'cubic'),                      # spaces and tabs around it are trimmed
    ({'symbol': 'P2/m'}, 'several Hall settings'),         # 57, 58, 59: the unique axis is ambiguous
    ({'symbol': 'P2/m', 'hall_number': 5}, 'not of P2/m'),  # Hall 5 is P2, not P2/m
    # SCFT takes spglib's symbols exactly: P6_3/mmc and Ia-3d
    ({'symbol': 'P63/mmc'}, 'Could not identify'), ({'symbol': 'I a -3 d'}, 'Could not identify'),
    ({'symbol': 'P63/mmc', 'hall_number': 488}, 'not of P63/mmc'),
))
def test_space_group_symbol_and_hall_number(group, outcome):
    pytest.importorskip('spglib')
    p = params()
    p.update(space_group=group, lx=[3., 3., 3.])
    if outcome == 'cubic':
        assert TOOL['choose_lattice'](p)[0] == 'cubic'
    else:
        with pytest.raises(ValueError, match=outcome):
            TOOL['choose_lattice'](p)


def test_invalid_space_group_symbol_has_a_clear_error():
    p = params()
    p['space_group'] = {'symbol': 123}
    with pytest.raises(ValueError, match='symbol must be a string'):
        TOOL['choose_lattice'](p)


def test_export_rejects_whitespace_in_monomer_identifiers():
    p = params()
    p['segment_lengths'] = {'two words': 1., 'long_B': 1.2}
    with pytest.raises(ValueError, match='must not contain whitespace'):
        TOOL['prepare_export'](result(p))


def test_explicit_lattice_does_not_require_spglib(monkeypatch):
    p = params()
    p['space_group'] = {'symbol': 'P1'}
    monkeypatch.setitem(sys.modules, 'spglib', None)
    assert TOOL['choose_lattice'](p, 'orthorhombic')[0] == 'orthorhombic'


def test_ambiguous_skewed_relaxation_is_not_silently_changed():
    p = params()
    p['angles'] = [72., 83., 66.]
    with pytest.raises(ValueError, match='angle constraints cannot be inferred'):
        TOOL['choose_lattice'](p)


@pytest.mark.parametrize('metadata', ([], {}, {'format': 'polymerfts-pscf-rgrid-1', 'nx': [2, 3], 'lattice': 'rectangular'}))
def test_invalid_column_metadata_is_rejected(tmp_path, metadata):
    source = rgrid(tmp_path / 'omega', [2, 3])
    meta = tmp_path / 'metadata.json'
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError): TOOL['import_files'](source, tmp_path / 'fields.mat', metadata=meta)
    assert not (tmp_path / 'fields.mat').exists()


@pytest.mark.parametrize('suffix', ('.mat', '.json', '.yaml'))
def test_result_export_import_preserves_fields_cell_and_custom_names(tmp_path, suffix):
    data = result()
    data['lx'] = [4., 5., 6.]  # Final relaxed cell differs from initial_params.
    source = tmp_path / ('source' + suffix)
    if suffix == '.mat': savemat(source, data, long_field_names=True)
    elif suffix == '.json': source.write_text(json.dumps(data, default=lambda x: x.tolist()))
    else:
        yaml = pytest.importorskip('yaml')
        source.write_text(yaml.safe_dump(json.loads(json.dumps(data, default=lambda x: x.tolist())), sort_keys=False))
    destination = tmp_path / 'pscf'
    TOOL['export_files'](source, destination)
    text = (destination / 'param').read_text()
    assert 'blocks[' in text and 'maxItr 1234' in text and 'isFlexible 1' in text
    assert 'groupName' not in text
    assert 'READ_W_RGRID in/omega' in (destination / 'command').read_text()
    omega = TOOL['read_rgrid'](destination / 'in/omega')
    assert omega['lx'] == [4., 5., 6.]
    target = tmp_path / 'imported.mat'
    TOOL['import_files'](destination / 'in/omega', target, metadata=destination / 'metadata.json')
    imported = loadmat(target, simplify_cells=True)
    assert list(np.atleast_1d(imported['monomer_types'])) == ['Zed', 'long_B']
    np.testing.assert_array_equal(imported['nx'], data['nx'])
    np.testing.assert_array_equal(imported['lx'], data['lx'])
    for name in ('Zed', 'long_B'):
        np.testing.assert_array_equal(imported['w_' + name], data['w_' + name].ravel())


@pytest.mark.parametrize('names', (
    ['A', 'B'],
    ['α', 'B'],
    ['α', 'α_real'],
    ['α', 'polymerfts_mat_keys_json'],
    ['α', 'field_' + hashlib.sha256('w_α'.encode()).hexdigest()[:24]],
))
def test_field_import_preserves_unicode_and_encoding_like_names(tmp_path, names):
    omega = rgrid(tmp_path / 'omega', [2, 3])
    rho = rgrid(tmp_path / 'rho', [2, 3], values=np.arange(6) / 10)
    output = TOOL['import_files'](omega, tmp_path / 'imported.mat', rho=rho, monomers=names)
    raw = loadmat(output, simplify_cells=True)
    restored = TOOL['_MAT']['decode_mat_fields'](raw)
    potentials = TOOL['_MAT']['read_monomer_fields'](raw)
    assert list(restored['monomer_types']) == names
    for kind, source in (('w_', omega), ('phi_', rho)):
        expected = TOOL['read_rgrid'](source)['fields']
        for i, name in enumerate(names):
            np.testing.assert_array_equal(restored[kind + name], expected[i].ravel())
            if name.isascii():
                np.testing.assert_array_equal(raw[kind + name], expected[i].ravel())
            if kind == 'w_':
                np.testing.assert_array_equal(potentials[name], expected[i].ravel())
    if all(name.isascii() for name in names):
        assert 'polymerfts_mat_keys_json' not in raw


@pytest.mark.parametrize('suffix', ('.mat', '.json', '.yaml'))
@pytest.mark.parametrize('separator', (',', '_', '/', ' '))
@pytest.mark.parametrize('boundary', ('periodic', 'Periodic', 'PeRiOdIc'))
def test_saved_result_option_spellings_export_identically(tmp_path, suffix, separator, boundary):
    p = params()
    p['chi_n'] = {separator.join(('long_B', 'Zed')): 20.}
    p['bc'] = [boundary] * 6
    data = result(p)
    # SCFT results retain the input spelling and also write canonical top-level chi_n.
    data['chi_n'] = {'Zed,long_B': 20.}
    before = json.dumps(data, default=lambda x: x.tolist())
    prepared, _, lattice, _ = TOOL['prepare_export'](data)
    text = TOOL['param_text'](prepared, lattice)
    assert json.dumps(data, default=lambda x: x.tolist()) == before
    assert '      0 1 20\n' in text

    source = tmp_path / ('source' + suffix)
    if suffix == '.mat':
        savemat(source, data, long_field_names=True)
    elif suffix == '.json':
        source.write_text(before)
    else:
        yaml = pytest.importorskip('yaml')
        source.write_text(yaml.safe_dump(json.loads(before), sort_keys=False))
    TOOL['export_files'](source, tmp_path / 'actual')

    control = tmp_path / 'control.json'
    control.write_text(json.dumps(result(), default=lambda x: x.tolist()))
    TOOL['export_files'](control, tmp_path / 'expected')
    for relative in ('param', 'command', 'in/omega', 'metadata.json'):
        assert (tmp_path / 'actual' / relative).read_bytes() == (tmp_path / 'expected' / relative).read_bytes()


@pytest.mark.parametrize('interactions', (
    {'Zed_long_B': 20., 'long_B/Zed': 20.},
    {'Zed,long_B': 20., 'long_BZed': 20.}, {'Zedlong_B': 20.},
    {'unknown,Zed': 20.}, {'Zed,Zed': 20.}, {'unknown': 20.}, {0: 20.},
    {'Zed_long_B': float('nan')}, {'Zed/long_B': float('inf')},
))
def test_invalid_interactions_still_fail_before_writing(tmp_path, interactions):
    p = params()
    p['chi_n'] = interactions
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(result(p), default=lambda x: x.tolist()))
    with pytest.raises(ValueError):
        TOOL['export_files'](source, tmp_path / 'pscf')
    assert not (tmp_path / 'pscf').exists()


@pytest.mark.parametrize('boundary', ('Reflecting', 'ABSORBING', None, 1))
def test_case_insensitive_periodic_check_keeps_rejecting_other_boundaries(boundary):
    p = params()
    p['bc'] = [boundary] * 6
    with pytest.raises(ValueError, match='periodic'):
        TOOL['prepare_export'](result(p))


def test_legacy_keys_and_branched_vertices_are_normalized(tmp_path):
    p = params()
    p['distinct_polymers'] = p.pop('polymers')
    p['box_is_altering'] = p.pop('relax_cell')
    p['scale_stress'] = .7
    p['distinct_polymers'][0]['blocks'] = [
        {'type': 'Zed', 'length': .2, 'v': 19, 'u': 90},
        {'type': 'long_B', 'length': .4, 'v': 90, 'u': 1234},
        {'type': 'Zed', 'length': .4, 'v': 90, 'u': 9999}]
    canonical, fields, lattice, cell = TOOL['prepare_export'](result(p))
    text = TOOL['param_text'](canonical, lattice)
    assert 'type branched' in text
    assert 'scaleStress 0.69999999999999996' in text
    block_lines = text.split('blocks[', 1)[1].split(']', 1)[0].splitlines()
    endpoints = [tuple(map(int, line.split()[-2:])) for line in block_lines if line.strip()]
    assert endpoints == [(0, 1), (1, 2), (1, 3)]


def test_matlab_singleton_polymer_and_block_structs(tmp_path):
    p = params()
    p['polymers'][0]['blocks'] = [{'type': 'Zed', 'length': 1.}]
    source = tmp_path / 'single.mat'
    savemat(source, result(p), long_field_names=True)
    TOOL['export_files'](source, tmp_path / 'pscf')
    assert 'nBlock 1' in (tmp_path / 'pscf/param').read_text()


# Langevin inputs: L-FTS requires saddle and compressor; CL-FTS runs without them, with zeta_n = 100
LANGEVIN = {'langevin': {'nbar': 1e4, 'dt': .01, 'max_step': 10}}
LFTS = {**LANGEVIN, 'saddle': {'max_iter': 100, 'tolerance': 1e-4}, 'compressor': {'name': 'lr'}}
GIBBS = {**LANGEVIN, 'gibbs_ensemble': {'cycles': 2, 'cycle_steps': 10, 'boxes': [
    {'volume_fractions': [1.]}, {'volume_fractions': [1.]}]}}


@pytest.mark.parametrize('suffix', ('.json', '.yaml'))
@pytest.mark.parametrize('model, message', ((LFTS, None), (LANGEVIN, 'compressible model'),
                                            ({**LFTS, 'zeta_n': 100.}, 'compressible model'), (GIBBS, 'two boxes')))
def test_langevin_parameter_files_export_one_incompressible_box(tmp_path, suffix, model, message):
    source = tmp_path / ('params' + suffix)
    p = {**params(), **model}
    source.write_text(json.dumps(p) if suffix == '.json' else pytest.importorskip('yaml').safe_dump(p))
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    if message is None:
        TOOL['export_files'](source, tmp_path / 'pscf', fields)
        assert (tmp_path / 'pscf/param').read_text() == TOOL['param_text'](params(), 'orthorhombic')
    else:
        with pytest.raises(ValueError, match=message):
            TOOL['export_files'](source, tmp_path / 'pscf', fields)
        assert not (tmp_path / 'pscf').exists()


def test_scft_result_with_langevin_settings_exports(tmp_path):
    # An SCFT run may share its params with a Langevin method: its result is one incompressible box
    data = result()
    data['initial_params'].update(LANGEVIN)
    source = tmp_path / 'scft.mat'
    savemat(source, data, long_field_names=True)
    TOOL['export_files'](source, tmp_path / 'pscf')
    assert (tmp_path / 'pscf/param').read_text() == TOOL['param_text'](params(), 'orthorhombic')


@pytest.mark.parametrize('simulation', ('LFTS', 'CLFTS', 'CLFTS as --fields'))
def test_langevin_fields_files(tmp_path, simulation):
    # A CL-FTS fields file, also of a box of a Gibbs ensemble, carries the zeta_n of its model and
    # its complex fields as w_A_real and w_A_imag (clfts.py save_simulation_data)
    data = result()
    if simulation == 'LFTS':
        data['initial_params'].update(LFTS)
    else:
        data['initial_params'].update(LANGEVIN)
        data['zeta_n'] = 100.
        for name in params()['segment_lengths']:
            w = data.pop('w_' + name)
            data['w_' + name + '_real'], data['w_' + name + '_imag'] = w, .5 * w
    source = tmp_path / 'fields.mat'
    savemat(source, data, long_field_names=True)
    if simulation == 'LFTS':
        TOOL['export_files'](source, tmp_path / 'pscf')
        assert (tmp_path / 'pscf/param').read_text() == TOOL['param_text'](params(), 'orthorhombic')
        return
    args = (source, tmp_path / 'pscf')
    if simulation == 'CLFTS as --fields':
        # The fields of an L-FTS parameter file are still CL-FTS fields, not a missing w_Zed
        args = (tmp_path / 'params.json', tmp_path / 'pscf', source)
        args[0].write_text(json.dumps({**params(), **LFTS}))
    with pytest.raises(ValueError, match='compressible model'):
        TOOL['export_files'](*args)
    assert not (tmp_path / 'pscf').exists()


@pytest.mark.parametrize('problem', ('discrete', 'complex', 'nonfinite', 'field_size', 'nx_mismatch',
                                   'missing_field', 'finite_compressibility', 'walls', 'random_block',
                                   'cycle', 'disconnected', 'partial_nodes', 'duplicate_chi', 'bad_cell',
                                   'electrostatics', 'grand_canonical', 'multiplicity'))
def test_invalid_exports_are_rejected_before_writing(tmp_path, problem):
    data = result()
    p = data['initial_params']
    if problem == 'discrete': p['chain_model'] = 'discrete'
    elif problem == 'complex': data['w_Zed'] = data['w_Zed'] + 1j
    elif problem == 'nonfinite': data['w_Zed'].flat[0] = np.nan
    elif problem == 'field_size': data['w_Zed'] = [1., 2.]
    elif problem == 'nx_mismatch': data['nx'] = [4, 3, 2]
    elif problem == 'missing_field': del data['w_Zed']
    elif problem == 'finite_compressibility': p['zeta_n'] = 100
    elif problem == 'walls': p['bc'] = ['reflecting']*6
    elif problem == 'random_block': p['polymers'][0]['blocks'][0]['monomer_fractions'] = {'Zed': 1.}
    elif problem == 'cycle': p['polymers'][0]['blocks'] = [
        {'type': 'Zed', 'length': .4, 'v': 0, 'u': 1}, {'type': 'long_B', 'length': .3, 'v': 1, 'u': 2},
        {'type': 'Zed', 'length': .3, 'v': 2, 'u': 0}]
    elif problem == 'disconnected': p['polymers'][0]['blocks'] = [
        {'type': 'Zed', 'length': .4, 'v': 0, 'u': 1}, {'type': 'long_B', 'length': .6, 'v': 2, 'u': 3}]
    elif problem == 'partial_nodes': p['polymers'][0]['blocks'][0]['v'] = 0
    elif problem == 'duplicate_chi': p['chi_n']['long_B,Zed'] = 20.
    elif problem == 'bad_cell': data['angles'] = [10., 10., 170.]
    elif problem == 'electrostatics': p['electrostatics'] = {'charges': {'Zed': .1, 'long_B': -.1}, 'coupling': 100.}
    elif problem == 'grand_canonical': p['polymers'][0]['chemical_potential'] = p['polymers'][0].pop('volume_fraction')
    elif problem == 'multiplicity': p['polymers'][0]['blocks'][0]['multiplicity'] = 2.5
    source = tmp_path / 'source.mat'
    savemat(source, data, long_field_names=True)
    with pytest.raises((ValueError, TypeError)): TOOL['export_files'](source, tmp_path / 'pscf')
    assert not (tmp_path / 'pscf').exists()


@pytest.mark.parametrize('problem', ('truncated', 'too_many_rows', 'columns', 'nonfinite', 'bad_count', 'bad_dim', 'basis'))
def test_malformed_pscf_fields_are_rejected(tmp_path, problem):
    path = rgrid(tmp_path / 'omega', [2, 3])
    text = path.read_text()
    if problem == 'truncated': text = '\n'.join(text.splitlines()[:-1]) + '\n'
    elif problem == 'too_many_rows': text += '10 -10\n'
    elif problem == 'columns': text = text.replace('10 -10', '10 -10 0')
    elif problem == 'nonfinite': text = text.replace('10 -10', 'nan -10')
    elif problem == 'bad_count': text = text.replace('N_cell_param\n2', 'N_cell_param\n9')
    elif problem == 'bad_dim': text = text.replace('dim\n2', 'dim\n4')
    elif problem == 'basis': text = text.replace('mesh 2 3', 'N_star 6')
    path.write_text(text)
    with pytest.raises(ValueError): TOOL['read_rgrid'](path)


def test_default_monomer_names_continue_after_z_with_letters(tmp_path):
    # Columns without names take A to Z, then AA, AB, ... as the simulation editor names its types:
    # the library takes letters and underscores only, so a name such as M27 could not be used
    omega = rgrid(tmp_path / 'omega', [2, 3])
    lines = omega.read_text().replace('N_monomer 2', 'N_monomer 28').splitlines()
    lines[lines.index('mesh 2 3') + 1:] = [' '.join(str(column) for column in range(28))] * 6
    omega.write_text('\n'.join(lines) + '\n')
    data = loadmat(TOOL['import_files'](omega, tmp_path / 'fields.mat'), simplify_cells=True)
    names = [*'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AA', 'AB']
    assert [str(name).strip() for name in np.atleast_1d(data['monomer_types'])] == names
    for column, name in enumerate(names):
        np.testing.assert_array_equal(data['w_' + name], np.full(6, column))
    assert [TOOL['default_monomer_name'](i) for i in (51, 52, 701, 702)] == ['AZ', 'BA', 'ZZ', 'AAA']


def test_omega_rho_headers_are_checked_independently(tmp_path):
    omega = rgrid(tmp_path / 'omega', [2, 3], inline=False)
    rho = rgrid(tmp_path / 'rho', [2, 3], inline=True)
    target = tmp_path / 'fields.mat'
    TOOL['import_files'](omega, target, rho=rho, monomers=['A1', 'B1'])
    data = loadmat(target, simplify_cells=True)
    np.testing.assert_array_equal(data['w_A1'], data['phi_A1'])
    rho.write_text(rho.read_text().replace('cell_param 2 3', 'cell_param 2 4'))
    with pytest.raises(ValueError, match='different grids, cells'): TOOL['import_files'](omega, tmp_path / 'bad.mat', rho=rho)
    assert not (tmp_path / 'bad.mat').exists()


def test_existing_outputs_are_preserved(tmp_path):
    source = tmp_path / 'source.mat'
    savemat(source, result(), long_field_names=True)
    output = tmp_path / 'pscf'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('keep me')
    with pytest.raises(FileExistsError): TOOL['export_files'](source, output)
    assert sentinel.read_text() == 'keep me'
    target = tmp_path / 'fields.mat'
    target.write_bytes(b'keep me')
    omega = rgrid(tmp_path / 'omega', [2, 3])
    with pytest.raises(FileExistsError): TOOL['import_files'](omega, target)
    assert target.read_bytes() == b'keep me'


def test_original_pscf_gyroid_fixture_can_be_imported(tmp_path):
    fixture = ROOT / 'tests/data/pscf'
    path = tmp_path / 'fields.mat'
    TOOL['import_files'](fixture / 'omega', path, rho=fixture / 'rho')
    data = loadmat(path, simplify_cells=True)
    assert list(data['nx']) == [32, 32, 32]
    np.testing.assert_allclose(data['lx'], [3.8891334563]*3, rtol=0, atol=1e-12)
    field = data['w_A'].reshape(32, 32, 32)
    assert field[1, 0, 0] == 2.146108919763847e+01
    lines = (fixture / 'omega').read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == 'mesh') + 2
    original = np.loadtxt(lines[start:])
    for x, y, z in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (7, 11, 13)):
        assert field[x, y, z] == original[x + 32 * (y + 32 * z), 0]
    assert data['phi_A'].size == 32**3


def test_command_line_example_and_error_exit(tmp_path):
    source = tmp_path / 'initial.mat'
    subprocess.run([sys.executable, str(ROOT / 'scripts/pscf/pscf_example.py'), '-o', str(source)], check=True, capture_output=True, timeout=15)
    output = tmp_path / 'pscf'
    subprocess.run([sys.executable, str(CONVERTER), 'export', str(source), '-o', str(output)], check=True, capture_output=True, timeout=15)
    subprocess.run([sys.executable, str(CONVERTER), 'import', str(output / 'in/omega'), '-o', str(tmp_path / 'roundtrip.mat'),
                    '--metadata', str(output / 'metadata.json')], check=True, capture_output=True, timeout=15)
    failed = subprocess.run([sys.executable, str(CONVERTER), 'export', str(source), '-o', str(output)], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=15)
    assert failed.returncode == 2 and 'Error:' in failed.stderr and 'Traceback' not in failed.stderr


def test_python_scripts_are_not_read(tmp_path):
    script = tmp_path / 'input.py'
    script.write_text('params = ' + repr(params()) + '\n')
    failed = subprocess.run([sys.executable, str(CONVERTER), 'export', str(script), '-o', str(tmp_path / 'pscf')],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
    assert failed.returncode == 2 and 'save_results' in failed.stderr and 'json.dump' in failed.stderr
    assert 'Traceback' not in failed.stderr and not (tmp_path / 'pscf').exists()
    source = tmp_path / 'params.json'
    source.write_text(json.dumps(params()))
    with pytest.raises(ValueError, match='Python scripts are not read'):
        TOOL['export_files'](source, tmp_path / 'pscf', script)


def test_bad_mat_file_has_a_clean_command_line_error(tmp_path):
    source = tmp_path / 'broken.mat'
    source.write_bytes(b'')
    target = tmp_path / 'pscf'
    failed = subprocess.run([sys.executable, str(CONVERTER), 'export', str(source), '-o', str(target)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
    assert failed.returncode == 2 and 'Cannot read MATLAB file' in failed.stderr
    assert 'Traceback' not in failed.stderr and not target.exists()
