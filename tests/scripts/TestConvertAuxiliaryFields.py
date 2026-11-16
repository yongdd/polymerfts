"""scripts/analysis/convert_into_auxiliary_fields.py: w_aux = A^-1 w of fields files and SCFT results."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/analysis/convert_into_auxiliary_fields.py'
TOOL = runpy.run_path(str(SCRIPT))


def write(path, names, complex_fields=False, **extra):
    """A fields file with random monomer fields and A^-1; returns them."""
    rng = np.random.default_rng(11)
    w = rng.normal(size=(len(names), 24)) + (1j*rng.normal(size=(len(names), 24)) if complex_fields else 0)
    a_inverse = rng.normal(size=(len(names), len(names)))
    data = {'nx': np.array([2, 3, 4]), 'lx': np.array([1.0, 1.5, 2.0]), 'monomer_types': names,
            'matrix_a_inverse': a_inverse, **extra}
    for name, field in zip(names, w):
        if complex_fields:
            data['w_' + name + '_real'], data['w_' + name + '_imag'] = field.real, field.imag
        else:
            data['w_' + name] = field
    savemat(path, data)
    return w, a_inverse


@pytest.mark.parametrize('names, complex_fields', [(['A', 'B'], False), (['A', 'Bb', 'C'], False), (['A', 'B'], True)])
def test_auxiliary_fields_of_a_fields_file(tmp_path, names, complex_fields):
    # Names of unequal length come back padded from a MAT file; CL-FTS stores real and imaginary parts
    w, a_inverse = write(tmp_path / 'fields.mat', names, complex_fields, angles=np.array([90.0, 90.0, 120.0]))
    TOOL['main']([str(tmp_path / 'fields.mat'), '-o', str(tmp_path / 'w_aux.mat')])
    out = loadmat(tmp_path / 'w_aux.mat', squeeze_me=True)
    np.testing.assert_allclose(out['w_aux'], a_inverse @ w, rtol=1e-13, atol=1e-13)
    assert [str(name).strip() for name in out['monomer_types']] == names
    np.testing.assert_array_equal(out['nx'], [2, 3, 4])
    np.testing.assert_array_equal(out['lx'], [1.0, 1.5, 2.0])
    np.testing.assert_array_equal(out['angles'], [90.0, 90.0, 120.0])


def test_angles_of_a_fields_file_and_one_monomer_type(tmp_path):
    # Fields files keep the angles in their checkpoint settings; a single type with a long name
    # loads as a plain string
    w, a_inverse = write(tmp_path / 'fields.mat', ['Bb'],
                         checkpoint_settings_json=json.dumps({'angles': [80.0, 95.0, 110.0]}))
    TOOL['main']([str(tmp_path / 'fields.mat'), '-o', str(tmp_path / 'w_aux.mat')])
    out = loadmat(tmp_path / 'w_aux.mat', squeeze_me=True)
    np.testing.assert_allclose(np.ravel(out['w_aux']), np.ravel(a_inverse @ w), rtol=1e-13, atol=1e-13)
    assert str(out['monomer_types']).strip() == 'Bb'
    np.testing.assert_array_equal(out['angles'], [80.0, 95.0, 110.0])


def test_command_line_and_refusals(tmp_path):
    write(tmp_path / 'fields.mat', ['A', 'B'])
    done = subprocess.run([sys.executable, str(SCRIPT), 'fields.mat'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0 and (tmp_path / 'w_aux.mat').exists(), done.stderr
    # A^-1 of another number of monomer types, and a missing field
    savemat(tmp_path / 'bad.mat', {'monomer_types': ['A', 'B'], 'matrix_a_inverse': np.eye(3),
                                   'w_A': np.zeros(4), 'w_B': np.zeros(4)})
    with pytest.raises(ValueError, match='shape'):
        TOOL['auxiliary_fields'](loadmat(tmp_path / 'bad.mat', squeeze_me=True))
    savemat(tmp_path / 'missing.mat', {'monomer_types': ['A', 'B'], 'matrix_a_inverse': np.eye(2), 'w_A': np.zeros(4)})
    with pytest.raises(ValueError, match='w_B'):
        TOOL['auxiliary_fields'](loadmat(tmp_path / 'missing.mat', squeeze_me=True))
    # On the command line: a message and exit status 2, as the other scripts
    for name in ('missing.mat', 'absent.mat'):
        failed = subprocess.run([sys.executable, str(SCRIPT), name], cwd=tmp_path, capture_output=True, text=True)
        assert failed.returncode == 2 and failed.stderr.startswith('Error:'), failed.stderr
    (tmp_path / 'result.json').write_text('{}')
    failed = subprocess.run([sys.executable, str(SCRIPT), 'result.json'], cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode == 2 and failed.stderr.startswith('Error:'), failed.stderr
