#!/usr/bin/env python3
r"""Monomer potential fields to auxiliary fields, w_aux = A^-1 w.

    python scripts/analysis/convert_into_auxiliary_fields.py fields_000200.mat
    python scripts/analysis/convert_into_auxiliary_fields.py scft_result.mat -o w_aux.mat

Reads the monomer fields w_<type> (CL-FTS: w_<type>_real and w_<type>_imag), monomer_types and
matrix_a_inverse of a fields file or an SCFT result, and writes w_aux, one row per auxiliary field,
with nx, lx, the lattice angles (of an SCFT result, or of the checkpoint settings of a fields file)
and monomer_types. This is SymmetricPolymerTheory.to_aux_fields of the library
(docs/theory/PolymerFieldTheory.md); L-FTS and CL-FTS fields files also hold w_aux themselves.

Reference: D. Düchs, K. T. Delaney and G. H. Fredrickson, J. Chem. Phys. 141, 174103 (2014).
"""

import argparse
import json

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError


def monomer_field(data, name):
    """w of a monomer type: w_<name>, or the real and imaginary parts of CL-FTS."""
    if 'w_' + name in data:
        return np.ravel(data['w_' + name])
    if 'w_' + name + '_real' in data:
        return np.ravel(data['w_' + name + '_real']) + 1j*np.ravel(data['w_' + name + '_imag'])
    raise ValueError(f"The file holds no field w_{name}.")


def auxiliary_fields(data):
    """w_aux = A^-1 w and the monomer types, in the order of matrix_a_inverse."""
    names = [str(name).strip() for name in np.atleast_1d(data['monomer_types'])]
    a_inverse = np.atleast_2d(np.asarray(data['matrix_a_inverse'], dtype=float))
    if a_inverse.shape != (len(names), len(names)):
        raise ValueError(f"matrix_a_inverse has the shape {a_inverse.shape}, "
                         f"the file has {len(names)} monomer types {names}.")
    w = np.stack([monomer_field(data, name) for name in names])
    return a_inverse @ w, names


def lattice_angles(data):
    """The angles [alpha, beta, gamma]: saved by SCFT, in the checkpoint settings of fields files."""
    if 'angles' in data:
        return np.atleast_1d(data['angles'])
    if 'checkpoint_settings_json' in data:
        return np.asarray(json.loads(str(data['checkpoint_settings_json']))['angles'])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fields", help="fields_*.mat of L-FTS or CL-FTS, or an SCFT result .mat")
    parser.add_argument("-o", "--output", default="w_aux.mat", help="output file (default w_aux.mat)")
    args = parser.parse_args(argv)
    try:
        data = loadmat(args.fields, squeeze_me=True)
        w_aux, names = auxiliary_fields(data)
        result = {'w_aux': w_aux, 'monomer_types': names}
        for key in ('nx', 'lx'):
            if key in data:
                result[key] = data[key]
        angles = lattice_angles(data)
        if angles is not None:
            result['angles'] = angles
        savemat(args.output, result)
    except (ValueError, TypeError, KeyError, OSError, MatReadError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {len(names)} auxiliary fields of {names} to {args.output}")
    return 0


if __name__ == "__main__":
    main()
