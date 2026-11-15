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


@pytest.mark.filterwarnings('error::DeprecationWarning')
@pytest.mark.parametrize('hall, angles, relax, lattice', (
    (57, [90., 100., 90.], True, 'monoclinic'), (58, [90., 90., 100.], False, 'triclinic'),
    (5, [100., 90., 90.], False, 'triclinic'), (58, [90., 90., 100.], True, None)))
def test_monoclinic_unique_axis(hall, angles, relax, lattice, monkeypatch):
    # PSCF's monoclinic lattice has the unique axis b; the unique axes a and c keep a fixed cell as triclinic.
    # The spglib calls use its raising error handling (no DeprecationWarning) and restore the setting;
    # spglib's environment variable, which overrides that setting, is left out here
    monkeypatch.delenv('SPGLIB_OLD_ERROR_HANDLING', raising=False)
    spglib = pytest.importorskip('spglib')
    handling = getattr(getattr(spglib, 'error', None), 'OLD_ERROR_HANDLING', None)
    p = params()
    p.update(angles=angles, relax_cell=relax, space_group={'symbol': 'P2/m' if hall in (57, 58) else 'P2',
                                                           'hall_number': hall})
    if lattice is None:
        with pytest.raises(ValueError, match='unique axis c'):
            TOOL['prepare_export'](result(p))
    else:
        assert TOOL['prepare_export'](result(p))[2] == lattice
    assert getattr(getattr(spglib, 'error', None), 'OLD_ERROR_HANDLING', None) == handling


@pytest.mark.parametrize('hall, message', ((0, 'positive'), (531, 'between 1 and 530')))
def test_hall_number_range(hall, message):
    pytest.importorskip('spglib')
    p = params()
    p['space_group'] = {'symbol': 'P1', 'hall_number': hall}
    with pytest.raises(ValueError, match=message):
        TOOL['choose_lattice'](p)


@pytest.mark.parametrize('group, outcome', (
    ({'symbol': 'Ia-3d'}, 'cubic'),                         # one Hall setting: the symbol suffices
    ({'symbol': 'P2/m'}, 'several Hall settings'),         # 57, 58, 59: the unique axis is ambiguous
    ({'symbol': 'P2/m', 'hall_number': 5}, 'not of P2/m'),  # Hall 5 is P2, not P2/m
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


@pytest.mark.parametrize('names', (('A', 'B'), ('Zed', 'long_B')))
@pytest.mark.parametrize('reverse', (False, True))
def test_unique_compact_interaction_keys_remain_supported(names, reverse):
    p = params()
    p['segment_lengths'] = dict(zip(names, (1., 1.2)))
    for block, name in zip(p['polymers'][0]['blocks'], names):
        block['type'] = name
    p['chi_n'] = {''.join(names[::-1] if reverse else names): 20.}
    prepared, _, lattice, _ = TOOL['prepare_export'](result(p))
    assert '      0 1 20\n' in TOOL['param_text'](prepared, lattice)


@pytest.mark.parametrize('interactions', (
    {'Zed_long_B': 20., 'long_B/Zed': 20.},
    {'Zed,long_B': 20., 'long_BZed': 20.},
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


@pytest.mark.parametrize('names,pair,message', (
    (('A', 'AB', 'B', 'BB'), 'ABB', 'ambiguous'),
    (('A', 'B_C', 'A_B', 'C'), 'A_B_C', 'ambiguous'),
    # A unique compact interpretation must not override invalid separator syntax.
    (('A', 'A_'), 'A_A', 'self interaction'),
    (('A', 'B_C', 'A_B', 'C', 'A_'), 'A_B_C', 'ambiguous'),
))
def test_ambiguous_compact_and_legacy_interactions_are_rejected(names, pair, message):
    p = params()
    p['segment_lengths'] = dict.fromkeys(names, 1.)
    for block, name in zip(p['polymers'][0]['blocks'], names):
        block['type'] = name
    p['chi_n'] = {pair: 20.}
    prepared, _, lattice, _ = TOOL['prepare_export'](result(p))
    with pytest.raises(ValueError, match=message):
        TOOL['param_text'](prepared, lattice)


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


def test_static_python_parameters_do_not_execute_code(tmp_path):
    marker = tmp_path / 'executed'
    source = tmp_path / 'input.py'
    p = params()
    p['ds'] = '__DS__'
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nf = 100\nparams = " + repr(p).replace("'__DS__'", '1/f') + '\nraise RuntimeError("must not run")\n')
    assert TOOL['read_input'](source)['ds'] == .01
    assert not marker.exists()
    fields = tmp_path / 'fields.mat'
    savemat(fields, {k: v for k, v in result().items() if k.startswith('w_')})
    TOOL['export_files'](source, tmp_path / 'pscf', fields)
    assert not marker.exists()


@pytest.mark.parametrize('code', ('params = load_config("file.json")', 'params["nx"] = [3]', 'params.update(nx=[3])'))
def test_dynamic_python_params_are_rejected(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError): TOOL['read_input'](source)


@pytest.mark.parametrize('code', (
    'alias = params\nalias["ds"] = .02',
    'alias = params["polymers"][0]\nalias["volume_fraction"] = .5',
    'alias = params\nalias.update(ds=.02)',
    'alias = params\nalias = alias.pop("ds")',
    'alias = params\ndel alias["ds"]',
    'alias = params["nx"]\nalias += [8]',
    'alias = params["nx"]\nif True:\n    alias += [8]',
    'params["nx"] += [8]',
))
def test_mutations_through_python_aliases_are_rejected(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError, match='mutations|method calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', (
    'other, = [params]\nother["lx"] = [2., 4., 5.]',
    '[other] = [params]\nother["lx"] = [2., 4., 5.]',
    'other, *rest = [params, params]\nother["lx"] = [2., 4., 5.]',
    'for other in [params]:\n    other["lx"] = [2., 4., 5.]',
    'for other, in [(params,)]:\n    other["lx"] = [2., 4., 5.]',
    'for block in params["polymers"][0]["blocks"]:\n    block["length"] *= 2',
    'if True:\n    other = params\n    other["lx"] = [2., 4., 5.]',
    'def helper():\n    other, = [params]\n    other["lx"] = [2., 4., 5.]\nhelper()',
    '[other.update(lx=[2., 4., 5.]) for other in [params]]',
    'other = [params for _ in range(1)]\nother[0]["lx"] = [2., 4., 5.]',
    'other = {name: params for name in ["saved"]}\nother["saved"]["lx"] = [2., 4., 5.]',
    '(other := params)\nother["lx"] = [2., 4., 5.]',
    'match params:\n    case other:\n        other["lx"] = [2., 4., 5.]',
    'match [params]:\n    case [other]:\n        other["lx"] = [2., 4., 5.]',
    'match params:\n    case {"polymers": polymers}:\n        polymers[0]["volume_fraction"] = .5',
    'match [params]:\n    case [*others]:\n        others[0]["lx"] = [2., 4., 5.]',
    'match params:\n    case {**other}:\n        other["polymers"][0]["volume_fraction"] = .5',
    'def tweak(p=params):\n    p["ds"] = .02\ntweak()',
    'def tweak(*, p=params):\n    p["ds"] = .02\ntweak()',
    'tweak = lambda p=params: p.update(ds=.02)\ntweak()',
    'def tweak(p=dict(params)):\n    p["polymers"][0]["volume_fraction"] = .5\ntweak()',
    'def outer():\n    def tweak(p=params):\n        p["ds"] = .02\n    tweak()\nouter()',
    'def obtain():\n    return params\nother = obtain()\nother["ds"] = .02',
    'obtain = lambda: params\nother = obtain()\nother["ds"] = .02',
    'def obtain():\n    return list(params["polymers"])\nother = obtain()\nother[0]["volume_fraction"] = .5',
    'def obtain():\n    yield params\nother = next(obtain())\nother["ds"] = .02',
    'def obtain():\n    yield from [params]\nother = next(obtain())\nother["ds"] = .02',
    'for name, block in [("Zed", params["polymers"][0]["blocks"][0])]:\n    block["length"] = .5',
    'other = params["polymers"] + []\nother[0]["volume_fraction"] = .5',
    'other = [params["polymers"][index] for index in range(1)]\nother[0]["volume_fraction"] = .5',
    'import numpy as np\nother = np.reshape(np.array(params["polymers"]), -1)\nother[0]["volume_fraction"] = .5',
))
def test_export_rejects_unsupported_container_alias_bindings(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    destination = tmp_path / 'pscf'
    with pytest.raises(ValueError, match='alias bindings'):
        TOOL['export_files'](source, destination, fields)
    assert not destination.exists()


@pytest.mark.parametrize('definition, mutation', (
    ('def obtain():\n    return params', 'other = obtain()\nother["ds"] = .02'),
    ('def obtain():\n    other = params\n    return other', 'other = obtain()\nother["ds"] = .02'),
    ('obtain = lambda: params', 'other = obtain()\nother["ds"] = .02'),
    ('def obtain():\n    return list(params["polymers"])',
     'other = obtain()\nother[0]["volume_fraction"] = .5'),
    ('def obtain():\n    yield params', 'other = next(obtain())\nother["ds"] = .02'),
    ('def obtain():\n    yield from [params]', 'other = next(obtain())\nother["ds"] = .02'),
))
def test_export_rejects_aliases_returned_by_earlier_helpers(tmp_path, definition, mutation):
    source = tmp_path / 'input.py'
    source.write_text(definition + '\nparams = ' + repr(params()) + '\n' + mutation + '\n')
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    destination = tmp_path / 'pscf'
    with pytest.raises(ValueError, match='alias bindings'):
        TOOL['export_files'](source, destination, fields)
    assert not destination.exists()


@pytest.mark.parametrize('definition', (
    'def helper():\n    return params["ds"]',
    'def helper():\n    return list(params["nx"])',
    'helper = lambda: tuple(params["nx"])',
    'def helper():\n    yield params["ds"]\n    yield from params',
    'def helper(params=()):\n    return sorted(params)',
    'from polymerfts import SCFT\ndef helper():\n    return SCFT(params)',
))
def test_earlier_helpers_can_return_scalars_and_independent_copies(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text(definition + '\nparams = ' + repr(params()) + '\nhelper()\n')
    assert TOOL['read_input'](source) == params()


@pytest.mark.parametrize('code', (
    'first, second, third = params["lx"]',
    'for length in params["lx"]:\n    print(length)',
    'for name in params["segment_lengths"]:\n    print(name)',
    'names = [name for name in params["segment_lengths"]]',
    'if True:\n    copied = list(params["nx"])\n    copied[0] = 9',
    'def helper():\n    copied = list(params["nx"])\n    copied[0] = 9',
    'def helper(params):\n    other, = [params]\n    for alias in [params]:\n        pass',
    'match params["chain_model"]:\n    case name:\n        print(name)',
    'match params:\n    case _:\n        pass',
    'def helper(params):\n    match params:\n        case other:\n            pass',
    'def helper(value=params["ds"], *, count=2):\n    return value\nhelper()',
    'helper = lambda value=params["ds"]: value\nhelper()',
    'def helper(copied=list(params["nx"]), *, value=round(.4)):\n    copied[0] = 9',
    'def helper():\n    return tuple(params["nx"])',
    'helper = lambda: list(params["nx"])',
    'def helper(params):\n    return params',
    'helper = lambda params: params',
    'def outer(params):\n    def inner(p=params):\n        return p\n    return inner',
    'def helper(params):\n    return unknown(params)',
    'def helper():\n    yield params["ds"]\n    yield from params',
    # Unpacking that binds only numbers, and arithmetic or indexing on lists of numbers
    'positions = [[0., 0.], [.5, .5]]\nfor y, z in positions:\n    print(y + z)',
    'pairs = [("Zed", [1, 2])]\nfor name, (first, second) in pairs:\n    print(name, first)',
    'first, *rest = params["lx"]',
    'factors = [params["nx"][i] / 2 for i in range(3)]',
    'import numpy as np\nindex = np.round(np.array([.5, .5, .5]) * params["nx"]).astype(int)',
    'import numpy as np\nw = np.reshape(np.zeros(24), params["nx"])',
    'import numpy as np\nw = np.broadcast_to(np.zeros(1), params["nx"]).copy()',
    'merged = params["chi_n"] | {}',
    'same = params["nx"] == [2, 3, 4]',
    'present = "Zed" in params["segment_lengths"]',
    'total = 0\nfor n in params["nx"]:\n    total += n',
    'import numpy as np\nmask = np.zeros(3) == params["nx"]',
    'last = params["nx"][index]',
    # Comparisons of builtin values give a bool; identity calls no method
    'present = "polymers" in params',
    'if params["polymers"] is not None:\n    pass',
    'if params["polymers"][0]["blocks"] == []:\n    pass',
    'match params["chain_model"]:\n    case "continuous":\n        pass',
    'import numpy as np\ngrid = np.array(params["nx"], dtype=float)',
    'longest = max(params["lx"])',
    'longest = max(params["lx"], default=0)',
    'import numpy as np\nw = np.array(params["nx"], float)',
    'import numpy as np\na = np.zeros(3)\na += params["nx"]',
    'import numpy as np\nw = np.zeros(10)\nw[params["nx"]] = 1',
    'label = f"{params[\'nx\']}".replace(" ", "")',
    'import sys\nsys.argv = ["x"]',
    # The simulation constructors deep-copy params, also a copy handed straight to them
    'from polymerfts import lfts\nsimulation = lfts.LFTS(params=params, random_seed=1)',
    'from polymerfts import CLFTS\nsimulation = CLFTS(params=params)',
    # The value bound to a name is read with its old binding
    'import numpy as np\ncenter = np.array([.5, .5, .5])\ncenter = center * params["lx"]',
    'import numpy as np\na = np.zeros(3)\na += a * params["lx"]',
    # NumPy arithmetic of a new array with numbers makes another array
    'import numpy as np\ncenter = np.array([.5, .5, .5])\ncenter = center * params["lx"]\nshift = center - params["lx"]',
    'import numpy as np\ncenter = np.array([.5, .5, .5]) * params["lx"] + 1\ncenter = center % params["lx"]',
    'import numpy as np\na = np.zeros(3)\na += 1\nshift = a - params["lx"]',
    'def helper():\n    from polymerfts import PropagatorSolver\n    return PropagatorSolver',
    'def report():\n    import polymerfts\n    print(polymerfts.__version__)',
    'try:\n    import polymerfts\nexcept ImportError:\n    polymerfts = None',
    'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from polymerfts import SCFT\ndef run(calculation: "SCFT"):\n    pass',
    'from polymerfts import lfts\ndef check(simulation):\n    return isinstance(simulation, (lfts.LFTS,)) or type(simulation) is lfts.LFTS',
    'import numpy as np\nlengths = np.array([1., 1., 1.])\nlengths *= 2 * np.pi / np.sqrt(3)\nlengths += -0.5 * np.ones(3)\nbox = lengths * params["lx"]',
    'import numpy as np\ncenter = np.array([.5, .5, .5])\ncenter -= center.mean()\ncenter += np.linspace(0, .1, 3)\nposition = center * params["lx"]',
    # A local import of a name leaves its module-level import as it is
    'from polymerfts import SCFT\ncalculation = SCFT(params)\ndef describe(simulation):\n    from polymerfts import SCFT\n    return isinstance(simulation, SCFT)',
    'import polymerfts\ndef version():\n    import polymerfts.utils\n    return polymerfts.__version__\ncalculation = polymerfts.SCFT(params)',
    'def total():\n    return sum(lengths)\nlengths = [1, 2]\nprint(total())',
    # A class's own binding leaves the module import as it is
    'import numpy as np\nclass Plot:\n    if True:\n        np = None\nw = np.zeros(params["nx"])',
))
def test_scalar_iteration_and_local_parameter_bindings_stay_supported(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    assert TOOL['read_input'](source) == params()


@pytest.mark.parametrize('code, changes', (
    ('from polymerfts import SCFT\ncalculation = SCFT(dict(params, ds=.02))', {'ds': .02}),
    ('from polymerfts import scft\nfixed = dict(params, relax_cell=False)\ncalculation = scft.SCFT(params=fixed)',
     {'relax_cell': False}),
    ('from polymerfts import SCFT\nfirst = SCFT(params)\nsecond = SCFT(params=params)', {}),
    ('from polymerfts import scft\ndef main():\n    return scft.SCFT(params=params)\nif __name__ == "__main__":\n    main()', {}),
    # Annotations name the class without calling it
    ('from polymerfts import SCFT\ndef run() -> SCFT:\n    return SCFT(params)\nrun()', {}),
    ('from polymerfts import SCFT\ncalculation: SCFT = SCFT(params)', {}),
))
def test_export_describes_the_params_the_constructor_receives(tmp_path, code, changes):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    assert TOOL['read_input'](source) == {**params(), **changes}


def test_a_constructor_in_a_function_can_precede_params(tmp_path):
    # Until params is bound the function cannot run; afterwards it receives params.
    source = tmp_path / 'input.py'
    source.write_text('from polymerfts import scft\ndef main():\n    return scft.SCFT(params=params)\n'
                      + 'params = ' + repr(params()) + '\nif __name__ == "__main__":\n    main()\n')
    assert TOOL['read_input'](source) == params()


@pytest.mark.parametrize('code', (
    'from polymerfts import SCFT\nfirst = SCFT(params)\nsecond = SCFT(dict(params, ds=.02))',
    'from polymerfts import SCFT\ndef make():\n    return {}\ncalculation = SCFT(params=make())',
    'from polymerfts import SCFT\nfor chi in (10., 20.):\n    calculation = SCFT(dict(params, chi_n={"Zed,long_B": chi}))',
    # A function that runs where its argument cannot be read, and again where it can
    'from polymerfts import scft\ndef run():\n    return scft.SCFT(params=p)\nfor chi in (12., 14.):\n'
    '    p = {"ds": .01, "chi_n": {"Zed,long_B": chi}}\n    run()\np = params\nrun()',
    'from polymerfts import scft\ndef run():\n    return scft.SCFT(params=p)\nfor chi in (12., 14.):\n'
    '    p = {"ds": .01, "chi_n": {"Zed,long_B": chi}}\n    run()',
    'import atexit, json\nfrom polymerfts import scft\ndef run():\n    return scft.SCFT(params=p)\n'
    'atexit.register(run)\np = json.load(open("other.json"))',
    'import json\nfrom pathlib import Path\nfrom polymerfts import scft\n'
    'calculation = scft.SCFT(json.loads(Path(__file__).with_name("other.json").read_text()))',
))
def test_constructor_params_that_one_input_cannot_describe_are_refused(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError, match='different params|literal values|alias bindings|unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', (
    'from polymerfts import scft\ncalculation = scft.SCFT.from_file("gyroid.yaml", ds=.02)',
    'from polymerfts import SCFT\nfirst = SCFT(params)\nsecond = SCFT.from_file("gyroid.yaml", max_iter=500)',
    'from polymerfts import LFTS\ndef main():\n    return LFTS.from_file("lamella.yaml", random_seed=12)\nmain()',
    'from polymerfts.fts.clfts import CLFTS\nsimulation = CLFTS.from_file("lamella.yaml")',
    # An import in a function or under a condition, an instance, a bound method
    'if __name__ == "__main__":\n    from polymerfts import scft\n    calculation = scft.SCFT.from_file("gyroid.yaml")',
    'def main():\n    from polymerfts import SCFT\n    return SCFT.from_file("gyroid.yaml")\nmain()',
    'from polymerfts import SCFT\nfirst = SCFT(params)\nsecond = first.from_file("gyroid.yaml")',
    'import functools\nfrom polymerfts import SCFT\nload = functools.partial(SCFT.from_file, "gyroid.yaml")\nload()',
    # Nor can it tell another from_file from these
    'class Settings:\n    @classmethod\n    def from_file(cls, path):\n        return cls()\nsettings = Settings.from_file("run.json")',
))
def test_calculations_that_load_their_params_from_a_file_are_refused(tmp_path, code):
    # The module params is not what such a calculation receives.
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError, match='from_file'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', (
    # An import in a function or under a condition
    'def main():\n    from polymerfts import SCFT\n    return SCFT({"ds": .5})\nmain()',
    'if __name__ == "__main__":\n    from polymerfts import scft\n    calculation = scft.SCFT({"ds": .5})',
    'try:\n    from polymerfts.scft import SCFT as Solver\nexcept ImportError:\n    Solver = None\ncalculation = Solver({"ds": .5})',
    'def main():\n    import polymerfts.utils\n    return polymerfts.SCFT({"ds": .5})\nmain()',
    'if True:\n    import polymerfts as pf\ncalculation = pf.scft.SCFT({"ds": .5})',
    # The class or its module given to another name, a loop, a function or a subclass
    'from polymerfts import SCFT\nSolver = SCFT\ncalculation = Solver({"ds": .5})',
    'from polymerfts import SCFT, LFTS\nfor method in (SCFT, LFTS):\n    method.from_file("gyroid.yaml")',
    'import functools\nfrom polymerfts import scft\nmake = functools.partial(scft.SCFT, {"ds": .5})\nmake()',
    'from polymerfts import SCFT\nclass Mine(SCFT):\n    pass\ncalculation = Mine({"ds": .5})',
    'from polymerfts import scft\ndef build(module):\n    return module.SCFT({"ds": .5})\nbuild(scft)',
    'import polymerfts\nfor module in (polymerfts,):\n    calculation = module.SCFT({"ds": .5})',
    # Also with the bindings of the last statement, for code that runs after it
    'import atexit\nfrom polymerfts import SCFT\ndef check():\n    return isinstance(None, SCFT)\n'
    'atexit.register(check)\ndef isinstance(a, b):\n    return b({"ds": .5})',
))
def test_simulation_classes_the_converter_cannot_follow_are_refused(tmp_path, code):
    # Such a call can receive params other than the module's.
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError, match='module-level import'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('mutation', (
    'params["ds"] = .2',
    'params["ds"] += .1',
    'del params["ds"]',
    'params.update(ds=.2)',
    'params["nested"]["ds"] = .2',
    'params["nested"].update(ds=.2)',
))
@pytest.mark.parametrize('scope', ('argument', 'local', 'closure', 'nonlocal'))
def test_local_container_mutations_do_not_change_exported_params(tmp_path, mutation, scope):
    independent = '{"ds": .3, "nested": {"ds": .4}}'
    if scope == 'argument':
        code = 'def helper(params):\n    ' + mutation + '\nhelper(' + independent + ')'
    elif scope == 'local':
        code = 'def helper():\n    params = ' + independent + '\n    ' + mutation + '\nhelper()'
    else:
        declaration = '        nonlocal params\n' if scope == 'nonlocal' else ''
        code = ('def helper():\n    params = ' + independent + '\n    def inner():\n'
                + declaration + '        ' + mutation + '\n    inner()\nhelper()')
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    assert TOOL['read_input'](source) == params()
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    destination = TOOL['export_files'](source, tmp_path / 'pscf', fields)
    assert (destination / 'param').read_text() == TOOL['param_text'](params(), 'orthorhombic')


@pytest.mark.parametrize('mutation', ('lengths[0] = 9', 'del lengths[0]', 'lengths.append(9)'))
def test_local_containers_can_shadow_parameter_dependencies(tmp_path, mutation):
    source = tmp_path / 'input.py'
    source.write_text('lengths = [2, 3, 4]\nparams = {"nx": lengths}\n'
                      + 'def helper(lengths):\n    ' + mutation + '\nhelper([5, 6, 7])\n')
    assert TOOL['read_input'](source) == {'nx': [2, 3, 4]}


@pytest.mark.parametrize('code', (
    'def helper():\n    global params\n    params["ds"] = .2',
    'def helper():\n    global params\n    del params["ds"]',
    'def helper():\n    global params\n    params.update(ds=.2)',
    'def helper():\n    params["polymers"][0]["volume_fraction"] = .5',
    'def outer():\n    def inner():\n        params.update(ds=.2)',
    'def outer(params):\n    def inner():\n        global params\n        params.update(ds=.2)',
    'def helper(params=params):\n    params.update(ds=.2)',
    'alias = params\ndef helper():\n    alias.update(ds=.2)',
    'def outer():\n    alias = params\n    def inner():\n        nonlocal alias\n        alias.update(ds=.2)',
    'def helper(params):\n    params.update(ds=.2)\nhelper(params)',
    'def helper():\n    [item for params in [] for item in []]\n    params["ds"] = .2',
    'def helper():\n    (params for params in [])\n    params.update(ds=.2)',
    'class Helper:\n    params["ds"] = .2\n    params = {}',
    'class Helper:\n    params.update(ds=.2)\n    params = {}',
    'class Helper:\n    params: dict\n    params["ds"] = .2',
    'class Helper:\n    params = {}\n    del params\n    params.update(ds=.2)',
    'class Helper:\n    params = {}\n    def method(self):\n        params.update(ds=.2)',
    'class Helper:\n    params = {}\n    [params.update(ds=.2) for item in [0]]',
    'def outer():\n    seed = {}\n    class Helper:\n        params["ds"] = .2\n        params = {}',
    'def run(polymers):\n    return helper(dict(params, polymers=polymers))',
    # A parameter container given to code of the script through NumPy, an operator or a constructor
    'import numpy as np\nnp.reshape(params["polymers"], -1)[0]["volume_fraction"] = .5',
    'import numpy as np\nnp.reshape(params["polymers"], -1)[0].update(volume_fraction=.5)',
    'import numpy as np\nnp.broadcast_to(params["polymers"], (1,))[0]["volume_fraction"] = .5',
    'import numpy as np\nclass M:\n    def reshape(self, shape, order="C"):\n        shape[0] = 8\nnp.reshape(M(), params["nx"])',
    'import numpy as np\nclass M:\n    def reshape(self, shape, order="C"):\n        shape[0] = 8\nx = np.reshape(M(), params["nx"])',
    'class M:\n    def __rmul__(self, other):\n        other[0] = 8\n        return 0\nx = params["nx"] * M()',
    'class M:\n    def __or__(self, other):\n        other.append(5)\n        return 0\nprint(M() | params["nx"])',
    'import numpy as np\nx = np.array([unknown]) * params["polymers"]',
    'from polymerfts import LFTS\nsimulation = LFTS(params, callbacks={"x": params["polymers"]})',
    'from polymerfts import LFTS\nLFTS(params, callbacks={"m": params["polymers"].append})',
    # Names that shadow a module literal, and object arrays, which hand a dict on as it is
    'm = 2\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return 0\ndef f(m):\n    return m * params["nx"]\nf(M())',
    'm = 2\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return 0\ng = lambda m: m * params["nx"]\ng(M())',
    'm = 2\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return 0\nfor m in [M()]:\n    m * params["nx"]',
    'm = 2\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return 0\n[m * params["nx"] for m in [M()]]',
    'import numpy\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return 0\ndef f(np):\n    return np.array([1]) * params["nx"]\nf(M())',
    'import numpy as np\nclass M:\n    def __or__(self, other):\n        other["Zed,long_B"] = 99\n        return 0\nprint(np.array([M()]) | params["chi_n"])',
    # Arithmetic of an array with an object of the script can give that object
    'import numpy as np\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return self\nv = M() * np.ones(1)\nprint(v * params["nx"])',
    'import numpy as np\nclass M:\n    def __mul__(self, other):\n        other[0] = 99\n        return self\nv = np.array(M()) * 1\nprint(v * params["nx"])',
    'import numpy as np\nclass M:\n    def __array_ufunc__(self, *args, **kwargs):\n        return self\n'
    '    def __mul__(self, other):\n        other[0] = 99\n        return self\nv = np.zeros(1)\nv += M()\nprint(v * params["nx"])',
    # Comparisons, augmented assignments and subscript keys call methods of the script too
    'class M:\n    def __eq__(self, other):\n        other[0] = 99\n        return False\nprint(M() == params["nx"])',
    'class M:\n    def __contains__(self, item):\n        item[0] = 99\n        return False\nprint(params["nx"] in M())',
    'class M:\n    def __eq__(self, other):\n        other["volume_fraction"] = .5\n        return False\nprint(M() in params["polymers"])',
    'class M:\n    def __iadd__(self, other):\n        other[0] = 99\n        return self\nm = M()\nm += params["nx"]',
    'class M:\n    def __getitem__(self, key):\n        key[0] = 99\nprint(M()[params["nx"]])',
    # A constructor deep-copies only a literal params: a hook would receive the memo,
    # which keeps the original parameter containers
    'from polymerfts import SCFT\nclass X:\n    def __deepcopy__(self, memo):\n        memo[id(memo)][0]["nx"][0] = 99\n        return self\nSCFT(dict(params, extra=X()))',
    'from polymerfts import SCFT\nclass X:\n    def __getattr__(self, name):\n        return lambda memo: self\nSCFT(dict(params, extra=X()))',
    'from polymerfts import SCFT\ndef run(polymers):\n    return SCFT(dict(params, polymers=polymers))',
    'import copy\ncopy._deepcopy_dispatch[list] = print',
    'import copy\ndef f(x, memo):\n    return x\ngetattr(copy, "_deepcopy" + "_dispatch")[list] = f',
    'exec("params[\'ds\'] = .2")',
    'import sys\nsys.modules["copy"]._deepcopy_dispatch[list] = print',
    # match value patterns and max/min compare with an object of the script
    'class E:\n    def __eq__(self, other):\n        other[0] = 99\n        return False\nclass K:\n    v = E()\nmatch params["nx"]:\n    case K.v:\n        pass',
    'class E:\n    def __lt__(self, other):\n        other[0] = 99\n        return False\n    __gt__ = __lt__\nprint(max(params["nx"], E()))',
    'def f(x):\n    return 0\nprint(max(params["nx"], params["lx"], key=f))',
    'x = 1\nclass E:\n    def __lt__(self, other):\n        other[0] = 99\n        return False\n    __gt__ = __lt__\ndef f(x):\n    return max(params["nx"], x)\nf(E())',
    # A mutation through an expression that holds parameter containers
    '(params["polymers"] + [])[0]["volume_fraction"] = .5',
    '(params["polymers"] + [])[0].update(volume_fraction=.5)',
    # Reflection and module internals reached by reference, alias or method
    'import copy\ncopy._deepcopy_dispatch.update({list: print})',
    'import copy\nd = copy._deepcopy_dispatch\nd[list] = print',
    'import copy\ng = getattr\ng(copy, "_deepcopy" + "_dispatch")[list] = print',
    'import sys\nm = sys.modules["copy"]',
    # The script's own module
    'import __main__\n__main__.params["ds"] = .5',
    'from __main__ import params as same\nsame["ds"] = .5',
    'def tweak():\n    import __main__ as m\n    m.params["ds"] = .5\ntweak()',
    'def tweak():\n    from __main__ import params as same\n    same["ds"] = .5\ntweak()',
    'def tweak():\n    import sys as s\n    s.modules["__main__"].params["ds"] = .5\ntweak()',
    'def tweak():\n    from importlib import import_module as load\n    load("__main__").params["ds"] = .5\ntweak()',
    'def tweak():\n    from builtins import exec as run\n    run("params[\'ds\'] = .5")\ntweak()',
    # A class body reads a name it binds later from the module, also inside a function
    'lengths = params["lx"]\nclass Helper:\n    lengths += [2.]',
    'def outer():\n    params = {}\n    class Helper:\n        params["ds"] = .2\n        params = {}\nouter()',
    # After a binding inside a compound statement, := or nonlocal, the class's name is not the import
    *('from polymerfts.scft import SCFT\nclass Evil:\n    def __init__(self, p):\n        p["ds"] = .5\n'
      'def run():\n    SCFT = Evil\n    class C:\n        ' + binding + '\n        SCFT(params)\nrun()'
      for binding in ('(SCFT := Evil)', 'if True:\n            SCFT = Evil',
                      'for SCFT in [Evil]:\n            pass', 'nonlocal SCFT')),
    'from polymerfts.scft import SCFT\nclass Evil:\n    def __init__(self, p):\n        p["ds"] = .5\n'
    'class C:\n    if True:\n        SCFT = Evil\n    SCFT(params)',
))
def test_scope_handling_still_rejects_parameter_mutations(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    with pytest.raises(ValueError, match='mutations|method calls|alias bindings|unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', (
    'class Helper:\n    params = {}\n    params["ds"] = .2\n    params.update(ds=.3)',
    'class Helper:\n    params = {}\n    def method(self, params):\n        params.update(ds=.2)',
    '[params.update(ds=.2) for params in [{"ds": .3}]]',
    '{params.update(ds=.2) for params in [{"ds": .3}]}',
    '{index: params.update(ds=.2) for index in range(1) for params in [{"ds": .3}]}',
    'list(params.update(ds=.2) for params in [{"ds": .3}])',
    'def helper():\n    [params.update(ds=.2) for params in [{"ds": .3}]]\nhelper()',
    'class Helper:\n    lengths = [1.]\n    lengths += [2.]',
))
def test_class_and_comprehension_container_scopes(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + code + '\n')
    assert TOOL['read_input'](source) == params()


@pytest.mark.parametrize('argument, setup', (
    ('params', ''),
    ('params["chi_n"]', ''),
    ('alias', 'alias = params["chi_n"]\n'),
    ('wrapper', 'wrapper = {"nested": params}\n'),
    ('alias', 'alias = list(params["polymers"])\n'),
    ('alias', 'alias = params["polymers"][:]\n'),
    ('alias', 'alias = dict(config=params)\n'),
    ('np.min([params])', 'import numpy as np\n'),
    ('params["polymers"] + []', ''),
    ('config=params', ''),
    ('**params', ''),
))
def test_helper_arguments_cannot_hide_parameter_mutations(tmp_path, argument, setup):
    marker = tmp_path / 'executed'
    source = tmp_path / 'input.py'
    source.write_text('from pathlib import Path\nparams = ' + repr(params()) + '\n' + setup
                      + f'def helper(*args, **kwargs):\n    Path({str(marker)!r}).touch()\n'
                      + f'helper({argument})\n', encoding='utf-8')
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['export_files'](source, tmp_path / 'pscf', fields)
    assert not marker.exists() and not (tmp_path / 'pscf').exists()


@pytest.mark.parametrize('imports, constructor', (
    ('from polymerfts import scft', 'scft.SCFT'),
    ('from polymerfts import SCFT as MeanField', 'MeanField'),
    ('import polymerfts as pfts', 'pfts.SCFT'),
))
def test_known_scft_constructors_and_initial_field_shapes_are_safe(tmp_path, imports, constructor):
    source = tmp_path / 'input.py'
    source.write_text(imports + '\nimport numpy as np\nparams = ' + repr(params())
                      + '\nw = np.zeros(list(params["nx"]))\n'
                      + 'n = np.prod(params["nx"])\nprint(params)\n'
                      + 'noise = np.random.normal(0, 1, np.prod(params["nx"]))\n'
                      + 'noise = np.random.normal(0, 1, size=params["nx"])\n'
                      + 'grid = np.asarray(params["nx"])\n'
                      + 'loaded = np.asarray(initial["w_Zed"]).reshape(params["nx"])\n'
                      + f'calculation = {constructor}(params=params)\n'
                      + 'calculation.run(initial_fields={"Zed": w, "long_B": w})\n', encoding='utf-8')
    assert TOOL['read_input'](source) == params()
    assert TOOL['read_input'](ROOT / 'examples/block_copolymer/scft/Gyroid.py')['chi_n'] == {'A,B': 20}


SCFT_EXAMPLES_WITH_FIELD_SETUP = (
    'examples/block_copolymer/scft/phases/Cylinder2D.py',
    'examples/block_copolymer/scft/phases/HCP_Hexagonal.py',
    'examples/block_copolymer/scft/phases/PL_Hexagonal.py',
)


@pytest.mark.parametrize('example', (
    *SCFT_EXAMPLES_WITH_FIELD_SETUP,
    'examples/block_copolymer/lfts/Gyroid.py',
    'examples/block_copolymer/lfts/Lamella.py',
    'examples/block_copolymer/clfts/Lamella.py',
    'examples/blend/clfts/GibbsEnsemble.py',
))
def test_shipped_examples_are_read_despite_their_field_setup(example):
    # Loops over positions, arithmetic with new arrays and simulation constructors
    # around the literal params must not count as parameter aliases.
    assert 'nx' in TOOL['read_input'](ROOT / example)


def test_constructor_params_of_local_values_are_refused():
    # SCFT(dict(params, polymers=polymers)) with polymers built in a function: the deep
    # copy of a value the converter cannot read could run code of the script.
    with pytest.raises(ValueError, match='alias bindings|unsupported calls'):
        TOOL['read_input'](ROOT / 'examples/blend/scft/CanonicalCoexistence.py')


@pytest.mark.parametrize('example', SCFT_EXAMPLES_WITH_FIELD_SETUP)
def test_shipped_scft_examples_export(example):
    p = TOOL['read_input'](ROOT / example)
    fields = {'w_' + name: np.zeros(math.prod(p['nx'])) for name in p['segment_lengths']}
    TOOL['prepare_export'](p, fields)


@pytest.mark.parametrize('binding, call', (
    ('from polymerfts import SCFT\nSCFT = helper', 'SCFT'),
    ('from polymerfts import scft\nscft.SCFT = helper', 'scft.SCFT'),
))
def test_shadowed_constructor_is_an_unsupported_helper(tmp_path, binding, call):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params())
                      + '\ndef helper(config):\n    config["ds"] = .02\n' + binding + '\n'
                      + call + '(params)\n')
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('definition', (
    'def helper():\n    np = 0\n    n = 1',
    'class Helper:\n    np = 0\n    n = 1',
    'def helper(np):\n    np = 0\n    n = 1',
    'class Helper:\n    def method(self):\n        np = 0\n        n = 1',
))
def test_local_bindings_preserve_module_imports_and_constants(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text('import numpy as np\nn = 4\n' + definition
                      + '\nparams = {"nx": [n]}\nw = np.zeros(params["nx"])\n')
    assert TOOL['read_input'](source) == {'nx': [4]}


@pytest.mark.parametrize('definition', (
    'def helper():\n    global SCFT\n    SCFT = tweak',
    'class Helper:\n    global SCFT\n    SCFT = tweak',
))
def test_explicit_global_writes_still_invalidate_constructor_imports(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text('from polymerfts import SCFT\ndef tweak(p):\n    p["ds"] = .2\n'
                      + definition + '\nparams = {"ds": .1}\nSCFT(params)\n')
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('body', (
    'if True:\n    SCFT = tweak\n    SCFT(params)',
    'for i in [0]:\n    SCFT = tweak\n    SCFT(params)',
    'if True:\n    from other_module import SCFT\n    SCFT(params)',
))
def test_compound_statement_rebinding_is_checked_before_its_calls(tmp_path, body):
    source = tmp_path / 'input.py'
    source.write_text('from polymerfts import SCFT\ndef tweak(p):\n    p["ds"] = .2\n'
                      + 'params = {"ds": .1}\n' + body + '\n')
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('definition', (
    'def helper():\n    SCFT = tweak\n    SCFT(params)\nhelper()',
    'class Helper:\n    SCFT = tweak\n    SCFT(params)',
    'def helper(SCFT=tweak):\n    SCFT(params)\nhelper()',
    'def helper():\n    SCFT = tweak\n    def inner():\n        SCFT(params)\n    inner()\nhelper()',
    'def helper():\n    global SCFT\n    SCFT = tweak\n    SCFT(params)\nhelper()',
))
def test_local_constructor_aliases_cannot_escape_container_checks(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text('from polymerfts import SCFT\ndef tweak(p):\n    p["ds"] = .2\n'
                      + 'params = {"ds": .1}\n' + definition + '\n')
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('definition', (
    'def helper():\n    def inner():\n        SCFT = 0\n    return SCFT(params)',
    'class Helper:\n    SCFT = 0\n    def method(self):\n        return SCFT(params)',
))
def test_nested_scope_locals_do_not_shadow_a_global_constructor_call(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text('from polymerfts import SCFT\nparams = ' + repr(params()) + '\n' + definition + '\n')
    assert TOOL['read_input'](source) == params()


@pytest.mark.parametrize('expression', (
    'make_array().reshape(params["nx"])',
    'def helper():\n    np = other_library\n    np.asarray(initial).reshape(params["nx"])',
))
def test_arbitrary_reshape_methods_cannot_receive_parameter_containers(tmp_path, expression):
    source = tmp_path / 'input.py'
    source.write_text('import numpy as np\nparams = {"nx": [4]}\n' + expression + '\n')
    with pytest.raises(ValueError, match='unsupported calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', ('lengths.append(8)', 'lengths[0] = 8', 'del lengths[0]'))
def test_mutated_helper_containers_are_rejected(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('lengths = [2, 3, 4]\n' + code + '\nparams = {"nx": lengths}\n')
    with pytest.raises(ValueError, match='mutations|method calls'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', ('lengths.append(8)', 'lengths[0] = 8', 'del lengths[0]',
                                  'global lengths\n    lengths = [8]'))
def test_helpers_defined_before_the_containers_they_change_are_rejected(tmp_path, code):
    # The function runs after the list is defined, whatever the order of the definitions.
    source = tmp_path / 'input.py'
    source.write_text('def refine():\n    ' + code + '\nlengths = [2, 3, 4]\nparams = {"nx": lengths}\nrefine()\n')
    with pytest.raises(ValueError, match='mutations|method calls|statically'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('code', ('f = load_value()', 'if True:\n    f = .6', 'f += .2', 'f, g = .6, 1.',
                                'match .6:\n    case f:\n        pass',
                                'match [.6]:\n    case [*f]:\n        pass',
                                'match {"fraction": .6}:\n    case {**f}:\n        pass'))
def test_runtime_constants_do_not_reuse_stale_values(tmp_path, code):
    source = tmp_path / 'input.py'
    source.write_text('f = .4\n' + code + '\nparams = {"fraction": f}\n')
    with pytest.raises(ValueError, match='statically'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('binding', (
    'from math import pi as chi',
    'import math as chi',
    'import chi',
    'if True:\n    from math import pi as chi',
    'def helper():\n    global chi\n    from math import pi as chi\nhelper()',
    'def helper(value):\n    global chi\n    chi = value\nchi = 15.\nhelper(20.)',
))
def test_export_rejects_stale_constants_after_imports(tmp_path, binding):
    p = params()
    p['chi_n'] = {'Zed,long_B': '__CHI__'}
    source = tmp_path / 'input.py'
    source.write_text('chi = 20.\n' + binding + '\nparams = '
                      + repr(p).replace("'__CHI__'", 'chi') + '\n')
    fields = tmp_path / 'fields.mat'
    savemat(fields, {key: value for key, value in result().items() if key.startswith('w_')})
    destination = tmp_path / 'pscf'
    with pytest.raises(ValueError, match='statically'):
        TOOL['export_files'](source, destination, fields)
    assert not destination.exists()


@pytest.mark.parametrize('binding', ('from settings import params', 'import settings as params',
                                   'from settings import *'))
def test_imports_cannot_replace_an_existing_params_dictionary(tmp_path, binding):
    source = tmp_path / 'input.py'
    source.write_text('params = ' + repr(params()) + '\n' + binding + '\n')
    with pytest.raises(ValueError, match='assignments|Star imports'):
        TOOL['read_input'](source)


@pytest.mark.parametrize('definition', (
    'def helper():\n    from math import pi as chi',
    'def helper():\n    import math as chi',
    'class Helper:\n    from math import pi as chi',
))
def test_local_imports_do_not_invalidate_module_constants(tmp_path, definition):
    source = tmp_path / 'input.py'
    source.write_text('chi = 20.\n' + definition + '\nparams = {"chi": chi}\n')
    assert TOOL['read_input'](source) == {'chi': 20.}


def test_unchanged_literal_containers_and_indexing_are_supported(tmp_path):
    source = tmp_path / 'input.py'
    source.write_text('lengths = [2, 3, 4]\nparams = {"nx": lengths, "ds": 1 / lengths[0]}\n')
    assert TOOL['read_input'](source) == {'nx': [2, 3, 4], 'ds': .5}


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


def test_bad_mat_file_has_a_clean_command_line_error(tmp_path):
    source = tmp_path / 'broken.mat'
    source.write_bytes(b'')
    target = tmp_path / 'pscf'
    failed = subprocess.run([sys.executable, str(CONVERTER), 'export', str(source), '-o', str(target)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
    assert failed.returncode == 2 and 'Cannot read MATLAB file' in failed.stderr
    assert 'Traceback' not in failed.stderr and not target.exists()
