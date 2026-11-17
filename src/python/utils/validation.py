"""Parameter checks for SCFT, L-FTS and CL-FTS.

The checks raise ValidationError (never assert, which disappears under
``python -O``) before any expensive C++ object is created.
"""

import copy
import difflib
import itertools
import math
import numbers
import warnings
from collections.abc import Mapping
from typing import Any, Dict
from os import PathLike

import numpy as np


class ValidationError(Exception):
    """Raised when a simulation parameter is missing or invalid."""
    pass


class NumericalError(RuntimeError):
    """The computation of a run failed: a numerical failure, not an input error.

    Non-finite residuals, fields, concentrations or Hamiltonians, lost material conservation and
    partition functions that are not positive or not consistent raise it. ``fields`` holds the
    monomer fields {type: array} of the failed state where the check has them, else None: SCFT
    gives them on the full grid, as SCFTResult.w, L-FTS and CL-FTS on their grid (complex in
    CL-FTS), the form run() takes as initial_fields.
    """

    def __init__(self, message, fields=None):
        super().__init__(message)
        self.fields = fields


class SaddlePointError(NumericalError):
    """A saddle-point search failed, and the Langevin run stopped before accepting the step;
    see ``sim.run_status``."""


def finite_values(values, label, dtype=np.float64):
    """values as a new flat array of dtype, float64 or complex128 (the caller's array stays as it is).

    Any number that converts to a finite float64 is accepted: booleans, integers of any size
    and floats, and with complex128 also complex numbers with finite parts. Strings, ragged
    lists, complex numbers for float64, NaN, infinities and numbers beyond float64 raise
    ValidationError.
    """
    complex_values = np.dtype(dtype) == np.complex128
    try:
        # A long double beyond float64 becomes inf, which the last check refuses
        with np.errstate(over="ignore"):
            array = np.asarray(values)   # a ragged list raises ValueError
            # Numbers that NumPy keeps as objects, such as Python integers beyond 64 bits (float() and
            # complex() raise OverflowError beyond float64); strings are not numbers
            number = numbers.Complex if complex_values else numbers.Real
            if array.dtype.kind == "O" and all(isinstance(value, number) for value in array.flat):
                array = np.array([complex(value) if complex_values else float(value) for value in array.flat])
            if array.dtype.kind in ("biufc" if complex_values else "biuf"):
                array = array.astype(dtype).reshape(-1)
                if np.all(np.isfinite(array)):
                    return array
    except (TypeError, ValueError, OverflowError):
        pass
    kind = "finite real or complex" if complex_values else "real, finite"
    raise ValidationError(f"{label} must contain {kind} values at float64 precision")


def initial_field_values(fields, monomer_types, size, dtype=np.float64):
    """The initial fields {monomer type: values} of a run as flat arrays of dtype with `size` values
    each (finite_values; the caller's arrays stay as they are). Keys of other names are left out."""
    if not isinstance(fields, Mapping):
        raise ValidationError("initial_fields must be a dict {monomer_type: field_array}")
    checked = {}
    for name in monomer_types:
        if name not in fields:
            raise ValidationError(f"initial_fields is missing monomer {name!r}")
        checked[name] = finite_values(fields[name], f"initial_fields[{name!r}]", dtype)
        if checked[name].size != size:
            raise ValidationError(f"initial_fields[{name!r}] needs {size} values, got {checked[name].size}")
    return checked


def previous_noise_values(noise, label, n_fields, n_grid, fields):
    """The previous noise of a run continued by hand (the noise of the Leimkuhler-Matthews step before the
    first step) as a flat float64 array of n_fields x n_grid real, finite values (finite_values; the
    caller's array stays as it is), or None without one. `fields` names the fields in the error."""
    if noise is None:
        return None
    noise = finite_values(noise, label)
    if noise.size != n_fields * n_grid:
        raise ValidationError(f"{label} needs {n_fields} x {n_grid} values ({fields} x grid points), got {noise.size}")
    return noise


def _rename_parameter(mapping, old, new, path=""):
    if old in mapping:
        if new in mapping:
            raise ValidationError(f"Specify only one of '{path}{old}' and '{path}{new}'")
        mapping[new] = mapping.pop(old)


def _normalize_initial_conditions(conditions, path):
    """Restore integer graft nodes from JSON/MAT object keys without merging nodes."""
    if not isinstance(conditions, dict):
        raise ValidationError(f"'{path}' must be a dict mapping integer node indices to labels, or None")
    normalized = {}
    for node, label in conditions.items():
        if isinstance(node, (int, np.integer)) and not isinstance(node, (bool, np.bool_)):
            index = int(node)
        elif (isinstance(node, str) and node.isascii() and node.isdecimal()
              and (node == "0" or not node.startswith("0"))):
            index = int(node)
        else:
            raise ValidationError(f"Node index {node!r} in '{path}' must be a non-negative integer "
                                  "or its canonical decimal string")
        if index < 0:
            raise ValidationError(f"Node index {node!r} in '{path}' must be non-negative")
        if index in normalized:
            raise ValidationError(f"Duplicate node index {index} in '{path}' after converting string keys")
        normalized[index] = label
    return normalized


def normalize_params(params):
    """Copy input parameters, translate legacy names and restore serialized graft nodes.

    Normalization stays at the input boundary so the simulation loops use only
    the documented names. Supplying both names is an error, even if equal.
    """
    if not isinstance(params, dict):
        raise ValidationError("Simulation parameters must be a dict")
    params = copy.deepcopy(params)
    for old, new in (("distinct_polymers", "polymers"),
                     ("box_is_altering", "relax_cell"),
                     ("scale_stress", "stress_scale"),
                     ("alpha_ds", "dynamic_stabilization")):
        _rename_parameter(params, old, new)
    polymers = params.get("polymers")
    if isinstance(polymers, list):
        for i, polymer in enumerate(polymers):
            if isinstance(polymer, dict):
                # Grand-canonical grafts are unsupported; report that domain
                # restriction before checking the syntax of their node keys.
                if "chemical_potential" not in polymer and polymer.get("initial_conditions") is not None:
                    polymer["initial_conditions"] = _normalize_initial_conditions(
                        polymer["initial_conditions"], f"polymers[{i}].initial_conditions")
                if isinstance(polymer.get("blocks"), list):
                    for j, block in enumerate(polymer["blocks"]):
                        if isinstance(block, dict):
                            _rename_parameter(block, "fraction", "monomer_fractions", f"polymers[{i}].blocks[{j}].")
    group = params.get("space_group")
    if isinstance(group, dict):
        _rename_parameter(group, "number", "hall_number", "space_group.")
    recording = params.get("recording")
    if isinstance(recording, dict):
        for old, new in (("recording_period", "checkpoint_interval"),
                         ("sf_computing_period", "sample_interval"),
                         ("sf_recording_period", "statistics_save_interval")):
            _rename_parameter(recording, old, new, "recording.")
    return params


# Largest block multiplicity, as MAX_BLOCK_MULTIPLICITY in Polymer.h (the native check)
MAX_BLOCK_MULTIPLICITY = 1.0e6

# Largest |βμ| of a grand-canonical polymer. Its chains enter as e^(βμ) times their unnormalized
# density and partition function, which are of order one for fields of zero mean. e^(βμ) overflows
# a double above βμ = 709.78 and is subnormal (losing precision) below -708.40; 700 keeps it a
# normal double with a margin of about e^8 for that factor.
MAX_POLYMER_CHEMICAL_POTENTIAL = 700.0


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValidationError(f"'{name}' must be a finite {qualifier}number, got {value!r}")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValidationError(f"'{name}' must be an integer >= {minimum}, got {value!r}")


def validate_monomer_fractions(fractions, name="monomer_fractions"):
    """Check random-block fractions without changing their physical values.

    Use the absolute sum tolerance of the system's polymer volume fractions and
    homogeneous thermodynamics; NumPy's default relative tolerance is too loose.
    """
    if not isinstance(fractions, dict) or not fractions:
        raise ValidationError(f"'{name}' must be a non-empty dict")
    for monomer, fraction in fractions.items():
        _finite_number(fraction, f"{name}.{monomer}", nonnegative=True)
    total = sum(fractions.values())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
        raise ValidationError(f"'{name}' must sum to 1.0, got {total}")


def _discrete_contour_length(contour_length: float, ds: float) -> float:
    """Mirror ContourLengthMapping::discrete_contour_length without a compiled backend.

    Standalone script tools load this module before the package is built. Keep
    the half-integer snapping, positive tie rounding and segment limit identical
    to the native implementation; TestDiscreteComposition checks their agreement.
    """
    contour_length, ds = float(contour_length), float(ds)
    if not (contour_length > 0 and math.isfinite(contour_length) and ds > 0 and math.isfinite(ds)):
        raise ValueError("Contour length and contour step must be positive and finite.")
    ratio = contour_length / ds
    # Native segment counts use a 32-bit signed int on the supported platforms.
    if not math.isfinite(ratio) or ratio > 2**31 - 1:
        raise ValueError("The block contour length requires more segments than supported.")
    rounded_to_half = math.floor(2 * ratio + 0.5) / 2
    if abs(ratio - rounded_to_half) < 1e-9:
        ratio = rounded_to_half
    # Python round() uses ties-to-even; native std::lround rounds positive ties up.
    n_segment = max(1, math.floor(ratio + 0.5))
    expected_length = n_segment * ds
    if abs(contour_length - expected_length) > 1e-9:
        raise ValueError(f"For discrete chain model, block contour_length ({contour_length:g}) must be an integer "
                         f"multiple of ds ({ds:g}). Closest valid value: {expected_length:g}")
    return expected_length


def polymer_block_lengths(params: Dict[str, Any], polymer: Dict[str, Any]) -> tuple:
    """Effective contour lengths in input block order, without changing input metadata.

    Continuous chains retain their lengths. Discrete chains follow the native
    acceptance and snapping rules of Molecules.add_polymer without importing it.
    """
    if str(params["chain_model"]).lower() != "discrete":
        return tuple(block["length"] for block in polymer["blocks"])
    lengths = []
    for j, block in enumerate(polymer["blocks"]):
        try:
            lengths.append(_discrete_contour_length(block["length"], params["ds"]))
        except (OverflowError, TypeError, ValueError) as exc:
            index = next((i for i, candidate in enumerate(params.get("polymers", ()))
                          if candidate is polymer), None)
            path = f"polymers[{index}]." if index is not None else ""
            raise ValidationError(f"'{path}blocks[{j}].length': {exc}") from exc
    return tuple(lengths)


def _check_block_numbering(polymer: Dict[str, Any]) -> None:
    """The vertices v and u are given in all blocks of a chain or in none."""
    numbered = "v" in polymer["blocks"][0]
    for block in polymer["blocks"]:
        for vertex in ("v", "u"):
            if (vertex in block) != numbered:
                raise ValidationError(f"Index {vertex} should exist in all blocks, "
                                      "or it should not exist in all blocks for each polymer.")


def polymer_block_copies(polymer: Dict[str, Any]) -> tuple:
    """Copies of each block in one chain, in input block order.

    A block (v, u) with 'multiplicity' m stands for m copies of itself and of every
    block on its u side, attached at v. The copies of a block are the product of the
    multiplicities that replicate it, as Block.n_copies of the native polymer.
    """
    blocks = polymer["blocks"]
    copies = [1.0] * len(blocks)
    if all(block.get("multiplicity", 1) == 1 for block in blocks):
        return tuple(copies)
    _check_block_numbering(polymer)
    numbered = "v" in blocks[0]
    edges = [(block["v"], block["u"]) if numbered else (j, j + 1) for j, block in enumerate(blocks)]
    neighbors = {}
    for j, (v, u) in enumerate(edges):
        neighbors.setdefault(v, []).append((u, j))
        neighbors.setdefault(u, []).append((v, j))
    for i, block in enumerate(blocks):
        multiplicity = float(block.get("multiplicity", 1))
        if multiplicity == 1:
            continue
        v, u = edges[i]
        replicated, to_visit = {i}, [(u, v)]
        while to_visit:
            node, reached_from = to_visit.pop()
            for next_node, j in neighbors[node]:
                if next_node != reached_from and j not in replicated:
                    replicated.add(j)
                    to_visit.append((next_node, node))
        for j in replicated:
            copies[j] *= multiplicity
    return tuple(copies)


def polymer_block_copy_lengths(params: Dict[str, Any], polymer: Dict[str, Any]) -> tuple:
    """Contour length of all copies of each block (length times copies), in input block order.

    Their sum is alpha, the length of one chain relative to the reference chain.
    """
    return tuple(length * copies for length, copies in zip(polymer_block_lengths(params, polymer),
                                                          polymer_block_copies(polymer)))


# An angle the cell fixes lies within this many radians of its value: the orthogonality test of the
# native box (FIXED_ANGLE_TOLERANCE in ComputationBox.h), which SpaceGroup.check_cell applies too
FIXED_ANGLE_TOLERANCE = 1e-10


# The crystal systems SCFT relaxes without a space group in each dimension (rows of CRYSTAL_SYSTEMS in
# scft/cell_relaxation.py)
SCFT_CRYSTAL_SYSTEMS = {1: ("Orthorhombic",),
                        2: ("Orthorhombic", "Hexagonal2D", "Oblique2D"),
                        3: ("Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Monoclinic", "Triclinic")}


def angle_fixed_at(degrees, fixed=90.0):
    """Whether an angle in degrees is the angle `fixed` (in degrees) for the native box and
    SpaceGroup.check_cell: within FIXED_ANGLE_TOLERANCE rad, in their arithmetic."""
    return abs(degrees * (math.pi / 180.0) - fixed * (math.pi / 180.0)) < FIXED_ANGLE_TOLERANCE


def _check_angles(angles, dim: int) -> None:
    """Lattice angles [alpha, beta, gamma] in degrees; a 2D box may give gamma alone, and the angles
    of the axes that a 1D or 2D box lacks are 90 degrees (the native box checks the cell itself)."""
    if angles is None:
        return
    if np.ndim(angles) != 1:
        raise ValidationError("angles must be a list of angles in degrees.")
    if dim == 3 and len(angles) != 3:
        raise ValidationError("angles must be [alpha, beta, gamma] (three entries) for a 3D box.")
    if dim < 3:
        if len(angles) not in (0, 1, 3):
            raise ValidationError("angles must have 0, 1 or 3 entries for a 1D or 2D box.")
        fixed = angles if dim == 1 else (angles[:2] if len(angles) == 3 else [])
        if not all(angle_fixed_at(angle) for angle in np.asarray(fixed, dtype=float)):
            raise ValidationError("A 1D box requires all angles to be 90 degrees; "
                                  "a 2D box requires alpha and beta to be 90 degrees.")


def validate_system_params(params: Dict[str, Any]) -> None:
    """Check the parameters shared by SCFT, L-FTS and CL-FTS, with the names of normalize_params.

    Required: nx, lx, ds, chain_model, segment_lengths, chi_n, polymers; optional: angles, platform, zeta_n.
    The first problem found raises ValidationError with a message naming the key.
    """
    for key in ("nx", "lx", "ds", "chain_model", "segment_lengths", "chi_n", "polymers"):
        if key not in params:
            raise ValidationError(f"Missing required parameter: '{key}'")

    nx, lx = params["nx"], params["lx"]
    if not isinstance(nx, (list, tuple, np.ndarray)) or not isinstance(lx, (list, tuple, np.ndarray)):
        raise ValidationError("'nx' and 'lx' must be lists")
    if len(nx) not in (1, 2, 3):
        raise ValidationError("'nx' must have 1, 2, or 3 dimensions")
    for n in nx:
        _integer(n, "nx")
    if len(lx) != len(nx):
        raise ValidationError(f"'lx' must have {len(nx)} elements, got {len(lx)}")
    for length in lx:
        _finite_number(length, "lx", positive=True)

    _check_angles(params.get("angles"), len(nx))

    ds = params["ds"]
    _finite_number(ds, "ds", positive=True)
    if ds > 1:
        raise ValidationError(f"'ds' must be a number in (0, 1], got {ds}")
    if params["chain_model"] not in ("continuous", "discrete"):
        raise ValidationError(f"'chain_model' must be 'continuous' or 'discrete', got '{params['chain_model']}'")

    segment_lengths = params["segment_lengths"]
    if not isinstance(segment_lengths, dict) or len(segment_lengths) == 0:
        raise ValidationError("'segment_lengths' must be a non-empty dict {monomer_type: length}")
    for monomer, length in segment_lengths.items():
        if not isinstance(monomer, str) or not monomer:
            raise ValidationError("Monomer names must be non-empty strings")
        if "," in monomer:
            raise ValidationError(f"Monomer name {monomer!r} must not contain ',', which separates the two names "
                                  "of a pair in 'chi_n'")
        _finite_number(length, f"segment_lengths.{monomer}", positive=True)

    if not isinstance(params["chi_n"], dict):
        raise ValidationError("'chi_n' must be a dict, e.g. {'A,B': 15.0}")
    for pair, value in params["chi_n"].items():
        _finite_number(value, f"chi_n.{pair}")

    polymers = params["polymers"]
    if not isinstance(polymers, list) or len(polymers) == 0:
        raise ValidationError("'polymers' must be a non-empty list")
    total_volume_fraction = 0.0
    for i, polymer in enumerate(polymers):
        if not isinstance(polymer, dict) or "blocks" not in polymer or \
                ("volume_fraction" in polymer) == ("chemical_potential" in polymer):
            raise ValidationError(f"polymers[{i}] must be a dict with 'volume_fraction' and 'blocks' (or, in "
                                  "CL-FTS, 'chemical_potential' in place of 'volume_fraction')")
        if "chemical_potential" in polymer:
            # Grand canonical: the chemical potential fixes the number of chains (CL-FTS checks the rest)
            mu = polymer["chemical_potential"]
            _finite_number(mu, f"polymers[{i}].chemical_potential")
            if abs(mu) > MAX_POLYMER_CHEMICAL_POTENTIAL:
                raise ValidationError(
                    f"'polymers[{i}].chemical_potential' must lie in [-{MAX_POLYMER_CHEMICAL_POTENTIAL:g}, "
                    f"{MAX_POLYMER_CHEMICAL_POTENTIAL:g}], got {mu!r}: the fugacity e^(chemical_potential) "
                    "must be a normal double (it overflows above 709.78 and is subnormal below -708.40)")
        else:
            vf = polymer["volume_fraction"]
            _finite_number(vf, f"polymers[{i}].volume_fraction", nonnegative=True)
            if vf > 1:
                raise ValidationError(f"polymers[{i}]['volume_fraction'] must be in [0, 1], got {vf}")
            total_volume_fraction += vf
        if not isinstance(polymer["blocks"], list) or len(polymer["blocks"]) == 0:
            raise ValidationError(f"polymers[{i}]['blocks'] must be a non-empty list")
        for j, block in enumerate(polymer["blocks"]):
            name = f"polymers[{i}].blocks[{j}]"
            if not isinstance(block, dict) or not isinstance(block.get("type"), str) or "length" not in block:
                raise ValidationError(f"'{name}' needs 'type' and 'length'")
            _finite_number(block["length"], name + ".length", positive=True)
            for vertex in ("v", "u"):
                if vertex in block:
                    _integer(block[vertex], name + "." + vertex, minimum=0)
            if "multiplicity" in block:
                _finite_number(block["multiplicity"], name + ".multiplicity")
                if not 1 <= block["multiplicity"] <= MAX_BLOCK_MULTIPLICITY:
                    raise ValidationError(f"'{name}.multiplicity' must be a number from 1 to 10^6, "
                                          f"got {block['multiplicity']!r}")
            if "monomer_fractions" in block:
                fractions = block["monomer_fractions"]
                validate_monomer_fractions(fractions, name + ".monomer_fractions")
                for monomer in fractions:
                    if monomer not in segment_lengths:
                        raise ValidationError(f"Unknown monomer '{monomer}' in '{name}.monomer_fractions'")
        _check_block_numbering(polymer)
    if any("chemical_potential" in polymer for polymer in polymers):
        if total_volume_fraction > 1.0 + 1e-10:
            raise ValidationError(f"Sum of the volume fractions of the canonical polymers must not exceed 1.0, "
                                  f"got {total_volume_fraction}")
    elif abs(total_volume_fraction - 1.0) > 1e-10:
        raise ValidationError(f"Sum of volume fractions must be 1.0, got {total_volume_fraction}")

    if "platform" in params and params["platform"] not in ("auto", "cuda", "cpu-fftw", "cpu-mkl"):
        raise ValidationError(f"'platform' must be 'auto', 'cuda', 'cpu-fftw' or 'cpu-mkl', got '{params['platform']}'")
    if params.get("zeta_n") is not None:
        _finite_number(params["zeta_n"], "zeta_n", nonnegative=True)


def parse_chi_n(
    chi_n_params: Dict[str, float],
    segment_lengths: Dict[str, float]
) -> Dict[str, float]:
    """Parse and validate Flory-Huggins interaction parameters.

    Parameters
    ----------
    chi_n_params : dict
        Dictionary of chi_n values with monomer pair keys (e.g., "A,B": 15.0).
    segment_lengths : dict
        Dictionary of segment lengths for validation.

    Returns
    -------
    dict
        Validated and normalized chi_n dictionary with sorted monomer pair keys.

    Raises
    ------
    ValidationError
        If monomer types are invalid, self-interactions are specified,
        or duplicate parameters are found.
    """
    chi_n = {}
    for monomer_pair_str, chin_value in chi_n_params.items():
        if not isinstance(monomer_pair_str, str):
            raise ValidationError("A chi_n key must name two monomer types, for example 'A,B'.")
        if ',' in monomer_pair_str:
            monomer_pair = [name.strip() for name in monomer_pair_str.split(',')]
            if len(monomer_pair) != 2:
                raise ValidationError(f"chi_n key '{monomer_pair_str}' must name exactly two monomer types.")
        else:
            # Keep legacy pair separators without splitting underscores inside names.
            pairs = [(a, b) for a in segment_lengths for b in segment_lengths
                     if any(monomer_pair_str == a + sep + b for sep in (' ', '_', '/'))]
            if len(pairs) != 1:
                raise ValidationError(f"chi_n key '{monomer_pair_str}' is invalid or ambiguous; use 'type1,type2'.")
            monomer_pair = list(pairs[0])
        if monomer_pair[0] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'.")
        if monomer_pair[1] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'.")
        if monomer_pair[0] == monomer_pair[1]:
            raise ValidationError(f"Do not add self interaction parameter, {monomer_pair_str}.")
        monomer_pair.sort()
        sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
        if sorted_monomer_pair in chi_n:
            raise ValidationError(f"There are duplicated χN ({sorted_monomer_pair}) parameters.")
        chi_n[sorted_monomer_pair] = chin_value
    return chi_n


def _validate_smearing(params: Dict[str, Any], complex_fields: bool) -> None:
    """The `smearing` option; per-type Gaussian lengths need complex fields (CL-FTS)."""
    smearing = params.get("smearing")
    if smearing is None:
        return
    if not isinstance(smearing, dict):
        raise ValidationError("'smearing' must be a dict or None")
    kind = smearing.get("type", "gaussian")
    if kind not in ("gaussian", "sigmoidal"):
        raise ValidationError("Unknown smearing type")
    if kind == "gaussian" and isinstance(smearing.get("a_int"), dict):
        # One smearing length per monomer type (CL-FTS)
        if not complex_fields:
            raise ValidationError("Per-type 'smearing.a_int' is supported in CL-FTS only")
        if set(smearing["a_int"]) != set(params["segment_lengths"]):
            raise ValidationError("Per-type 'smearing.a_int' needs one length for each monomer type")
        if _has_random_copolymer(params):
            raise ValidationError("Per-type 'smearing.a_int' does not support random copolymers")
        for name, a_int in smearing["a_int"].items():
            _finite_number(a_int, f"smearing.a_int.{name}", nonnegative=True)
    else:
        keys = ("a_int",) if kind == "gaussian" else ("k_int", "dk_int")
        for key in keys:
            _finite_number(smearing.get(key, 5.0 if key == "dk_int" else None), "smearing." + key,
                           positive=key != "a_int", nonnegative=key == "a_int")


def _numerical_method(params: Dict[str, Any]) -> tuple:
    """The numerical method of the input as the factories resolve it, and its capabilities in the method
    registry: (None, None) for a name the registry does not know (PropagatorSolver names the valid ones),
    and without the native extension (the input is also checked before it is built)."""
    try:
        from .. import _core
    except ImportError:
        return None, None
    try:
        name = _core.resolve_numerical_method(params["chain_model"], str(params.get("numerical_method") or "").lower())
    except RuntimeError:
        return None, None
    return name, _core.numerical_methods()[name]


# The top-level keys of the input that each method reads, in its constructor and the objects it
# builds, with the names of normalize_params. check_method_keys warns about the options of the other
# methods, which a method ignores, and refuses the keys no method reads (a misspelled key)
_SYSTEM_KEYS = {"nx", "lx", "angles", "bc", "ds", "chain_model", "segment_lengths", "chi_n", "polymers", "zeta_n",
                "platform", "numerical_method", "reduce_memory", "aggregate_propagator_computation", "smearing"}
_LANGEVIN_KEYS = {"langevin", "recording", "verbose_level"}
METHOD_KEYS = {
    "SCFT": _SYSTEM_KEYS | {"optimizer", "relax_cell", "max_iter", "tolerance", "stress_scale", "stress_interval",
                           "crystal_system", "space_group", "axis_ordering", "electrostatics", "polymer_figure_dir",
                           "fts_consistent_energy"},
    "L-FTS": _SYSTEM_KEYS | _LANGEVIN_KEYS | {"saddle", "compressor", "wtmd"},
    "CL-FTS": _SYSTEM_KEYS | _LANGEVIN_KEYS | {"dynamic_stabilization", "electrostatics"},
}
# Why no method reads a key that users give
_UNREAD_KEY_HINTS = {"mask": "impenetrable regions are available through PropagatorSolver(mask=...) only",
                     "gibbs_ensemble": "a Gibbs-ensemble input runs with GibbsEnsemble(params)"}


def check_method_keys(params: Dict[str, Any], method: str) -> None:
    """The top-level keys that the method ("SCFT", "L-FTS" or "CL-FTS") does not read (METHOD_KEYS):
    the options of another method are ignored with one warning that lists them, so that one input serves
    SCFT and the Langevin methods; a key that no method reads (a misspelled or unsupported option)
    raises ValidationError naming it and the closest key. params has the names of normalize_params."""
    known = sorted(set().union(*METHOD_KEYS.values()))
    ignored = []
    for key, value in params.items():
        if key in METHOD_KEYS[method] or value is None:   # None asks for nothing
            continue
        readers = [other for other, keys in METHOD_KEYS.items() if key in keys]
        if not readers:
            hint = _UNREAD_KEY_HINTS.get(key)
            if hint is None:
                close = difflib.get_close_matches(str(key), known, n=1)
                hint = (f"did you mean '{close[0]}'?" if close
                        else "no method reads it (docs/references/Parameters.md lists the keys)")
            raise ValidationError(f"{method} does not take a '{key}' parameter: {hint}")
        if value is not None:
            ignored.append(f"{key} ({' and '.join(readers)})")
    if ignored:
        # stacklevel: the caller of the simulation class (check_method_keys < validate_* < the constructor)
        warnings.warn(f"{method} ignores the options of other methods: {', '.join(ignored)}", stacklevel=4)


def validate_scft_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the SCFT options before constructing numerical objects; returns the copy with the
    names of normalize_params that the simulation uses."""
    params = normalize_params(params)
    check_method_keys(params, "SCFT")
    validate_system_params(params)
    group = params.get("space_group")
    if group is not None:
        if not isinstance(group, dict) or not isinstance(group.get("symbol"), str) or not group["symbol"]:
            raise ValidationError("'space_group' needs a dict with the Hermann-Mauguin 'symbol' (and optionally "
                                  "'hall_number'), e.g. {'symbol': 'Ia-3d'}")
        hall = group.get("hall_number")
        if hall is not None and (isinstance(hall, (bool, np.bool_)) or not isinstance(hall, (int, np.integer))
                                 or not 1 <= hall <= 530):
            raise ValidationError(f"'space_group.hall_number' must be an integer from 1 to 530, got {hall!r}")
        # 'electrostatics' needs the smearing (validate_electrostatics), so this refuses it too
        if params.get("smearing") is not None:
            raise ValidationError("'smearing' (and 'electrostatics', which needs it) is not available with "
                                  "'space_group': the smearing works on full-grid fields, the fields of a space "
                                  "group live in its reduced basis")
    for key in ("optimizer", "relax_cell"):
        if key not in params:
            raise ValidationError(f"Missing required parameter for SCFT: '{key}'")
    figure_dir = params.get("polymer_figure_dir")
    if figure_dir is not None and not isinstance(figure_dir, (str, PathLike)):
        raise ValidationError(f"'polymer_figure_dir' must be a directory path or None, got {figure_dir!r}")
    electrostatics = params.get("electrostatics")
    if electrostatics is not None:
        # The mean field of the Coulomb interaction (ψ from the Poisson equation); SCFT has no N̄
        if isinstance(electrostatics, dict) and "bjerrum_length" in electrostatics:
            raise ValidationError("SCFT has no N̄ for 'electrostatics.bjerrum_length': give 'coupling' "
                                  "(E = 4 pi l_B rho0 N R0^2)")
        if isinstance(electrostatics, dict) and "psi_dt_scaling" in electrostatics:
            raise ValidationError("'electrostatics.psi_dt_scaling' is a CL-FTS option")
    # Before validate_electrostatics, which reads the smearing length of the charges
    _validate_smearing(params, complex_fields=False)
    if electrostatics is not None:
        validate_electrostatics(params)
        if not electrostatics["coupling"] > 0:
            raise ValidationError("SCFT needs a positive 'electrostatics.coupling'; leave out 'electrostatics' "
                                  "for uncharged chains")
    _reject_grand_canonical(params, "SCFT")
    if params.get("stress_interval", 1) != "adaptive":
        _integer(params.get("stress_interval", 1), "stress_interval")
    optimizer = params["optimizer"]
    if not isinstance(optimizer, dict) or optimizer.get("name") != "am":
        raise ValidationError("optimizer.name must be 'am' (Anderson mixing)")
    _integer(optimizer.get("max_hist"), "optimizer.max_hist", minimum=0)
    for key in ("start_error", "mix_min", "mix_init"):
        _finite_number(optimizer.get(key), "optimizer." + key, nonnegative=True)
    _finite_number(params.get("tolerance", 1e-8), "tolerance", positive=True)
    _finite_number(params.get("stress_scale", 1), "stress_scale", positive=True)
    _integer(params.get("max_iter", 2000), "max_iter")
    # A crystal system given without a space group exists in the dimension (a group fixes its own)
    crystal_system = params.get("crystal_system")
    valid_systems = SCFT_CRYSTAL_SYSTEMS[len(params["nx"])]
    if group is None and crystal_system is not None and crystal_system not in valid_systems:
        raise ValidationError(f"'crystal_system' {crystal_system!r} is not valid for a {len(params['nx'])}D box; "
                              "choose from " + ", ".join(valid_systems))
    if params["relax_cell"]:
        # Box relaxation needs the stress, which not every numerical method computes
        method, capabilities = _numerical_method(params)
        if capabilities is not None and not capabilities["supports_stress"]:
            raise ValidationError(f"'relax_cell' needs the stress; '{method}' computes none")
        # Box relaxation with electrostatics is limited to periodic boxes (CellRelaxation.stress also
        # holds in closed boxes)
        if electrostatics is not None and any(str(b).lower() != "periodic" for b in params.get("bc") or []):
            raise ValidationError("'relax_cell' with 'electrostatics' needs a periodic box")
    # Preserve the ordering of option checks, while rejecting invalid discrete
    # lengths before any solver is constructed (including uncharged systems).
    for polymer in params["polymers"]:
        polymer_block_lengths(params, polymer)
    return params


def validate_lfts_params(params: Dict[str, Any], *, complex_fields=False) -> Dict[str, Any]:
    """Validate the shared Langevin options before constructing numerical objects; returns the copy with
    the names of normalize_params that the simulation uses."""
    params = normalize_params(params)
    # 'electrostatics' defines the system (its Hamiltonian), not an option of the method: L-FTS, which has
    # no electrostatic potential, refuses it rather than simulating a neutral system
    if not complex_fields and params.get("electrostatics") is not None:
        raise ValidationError("'electrostatics' is supported in SCFT and CL-FTS only: L-FTS has no "
                              "electrostatic potential and would simulate a neutral system; remove it")
    check_method_keys(params, "CL-FTS" if complex_fields else "L-FTS")
    validate_system_params(params)
    # The files name the measurements of a monomer pair by the two names joined with '_'
    # (structure_function_A_B, dH_history_A_B, I1_A_B), which must differ for distinct pairs
    # (A,B_C and A_B,C do not); L-FTS writes the auxiliary fields as w_aux next to the fields w_<type>
    pairs = {}
    for pair in itertools.combinations_with_replacement(sorted(params["segment_lengths"]), 2):
        pairs.setdefault("_".join(pair), []).append(",".join(pair))
    for suffix, same in pairs.items():
        if len(same) > 1:
            raise ValidationError(f"The monomer pairs {same} would share the key suffix '{suffix}' of the "
                                  "measurement files; rename a monomer type")
    if not complex_fields and "aux" in params["segment_lengths"]:
        raise ValidationError("L-FTS writes the auxiliary fields as 'w_aux': rename the monomer type 'aux'")
    langevin = params.get("langevin", None)
    if not isinstance(langevin, dict):
        raise ValidationError("Missing required parameter: 'langevin' (dict with 'nbar', 'dt', 'max_step')")
    for key in ("nbar", "dt"):
        _finite_number(langevin.get(key), "langevin." + key, positive=True)
    _integer(langevin.get("max_step"), "langevin.max_step", minimum=0)
    recording = params.get("recording")
    if not isinstance(recording, dict) or not isinstance(recording.get("dir"), (str, PathLike)):
        raise ValidationError("'recording' needs a 'dir' path")
    for key in ("checkpoint_interval", "sample_interval", "statistics_save_interval"):
        _integer(recording.get(key), "recording." + key)
    for key in ("pressure", "concentrations"):
        if key in recording:
            if not isinstance(recording[key], (bool, np.bool_)):
                raise ValidationError(f"'recording.{key}' must be True or False")
            if recording[key] and not complex_fields:
                raise ValidationError(f"'recording.{key}' is supported in CL-FTS only")
    _integer(params.get("verbose_level"), "verbose_level", minimum=0)
    if params["verbose_level"] > 2:
        raise ValidationError("'verbose_level' must be 0, 1 or 2")
    if complex_fields:
        # A non-integer multiplicity raises complex propagators to a real power, which has no unique value.
        for i, polymer in enumerate(params["polymers"]):
            for j, block in enumerate(polymer["blocks"]):
                if not float(block.get("multiplicity", 1)).is_integer():
                    raise ValidationError(f"'polymers[{i}].blocks[{j}].multiplicity' must be an integer in CL-FTS "
                                          f"(complex fields), got {block['multiplicity']!r}")
    if not complex_fields:
        _reject_grand_canonical(params, "L-FTS")
    else:
        for i, polymer in enumerate(params["polymers"]):
            if "chemical_potential" in polymer and (polymer.get("initial_conditions") or _has_random_copolymer(
                    {"polymers": [polymer]})):
                raise ValidationError(f"polymers[{i}]: grand-canonical polymers cannot be grafted or random copolymers")
    # All fields of CL-FTS fluctuate: it solves no saddle point (and ignores 'saddle' and 'compressor')
    if not complex_fields:
        saddle = params.get("saddle")
        if not isinstance(saddle, dict):
            raise ValidationError("Missing required dict: 'saddle'")
        _integer(saddle.get("max_iter"), "saddle.max_iter")
        _finite_number(saddle.get("tolerance"), "saddle.tolerance", positive=True)
        _integer(saddle.get("max_retries", 5), "saddle.max_retries", minimum=0)
        if saddle.get("failure_policy", "retry") not in ("retry", "abort", "resample"):
            raise ValidationError("saddle.failure_policy must be 'retry', 'abort' or 'resample'")
        compressor = params.get("compressor")
        if not isinstance(compressor, dict) or compressor.get("name") not in ("am", "lr", "lram"):
            raise ValidationError("compressor.name must be 'am', 'lr' or 'lram'")
        if compressor["name"] in ("am", "lram"):
            _integer(compressor.get("max_hist"), "compressor.max_hist", minimum=0)
            for key in ("start_error", "mix_min", "mix_init"):
                _finite_number(compressor.get(key), "compressor." + key, nonnegative=True)
    # The propagators of grafted chains start from q_init, which only SCFT.run takes
    for i, polymer in enumerate(params["polymers"]):
        if polymer.get("initial_conditions"):
            raise ValidationError(f"polymers[{i}]: L-FTS and CL-FTS need free chains; grafted chains "
                                  "(initial_conditions) need q_init, which only SCFT takes")
    # The options of one of the two methods, checked where they are read (the other method ignores
    # them, check_method_keys): dynamical stabilization and electrostatics of CL-FTS, WTMD of L-FTS
    if complex_fields and "dynamic_stabilization" in params:
        _finite_number(params["dynamic_stabilization"], "dynamic_stabilization", nonnegative=True)
    _validate_smearing(params, complex_fields)
    if complex_fields and params.get("electrostatics") is not None:
        validate_electrostatics(params)
    wtmd = params.get("wtmd")
    if not complex_fields and wtmd is not None:
        if not isinstance(wtmd, dict):
            raise ValidationError("'wtmd' must be a dict or None")
        for key in ("update_freq", "recording_period"):
            _integer(wtmd.get(key, 1000 if key == "update_freq" else 100000), "wtmd." + key)
        for key, default in (("ell", 4), ("kc", 6.02), ("delta_t", 5.0), ("sigma_psi", 0.16), ("dpsi", 1e-3)):
            _finite_number(wtmd.get(key, default), "wtmd." + key, positive=True)
        if wtmd.get("ell", 4) < 1:
            raise ValidationError("wtmd.ell must be at least 1")
        lo, hi = wtmd.get("psi_min", 0.0), wtmd.get("psi_max", 10.0)
        _finite_number(lo, "wtmd.psi_min")
        _finite_number(hi, "wtmd.psi_max")
        bins = (hi - lo) / wtmd.get("dpsi", 1e-3)
        if hi <= lo or bins < 2:
            raise ValidationError("WTMD range must contain at least two bins")
        # The histogram, the Gaussian kernels and the integrals of the statistics all use the bin width
        if abs(bins - round(bins)) > 1e-9 * bins:
            raise ValidationError(f"wtmd.dpsi must divide psi_max - psi_min into a whole number of bins "
                                  f"(dpsi = {wtmd.get('dpsi', 1e-3)!r}, psi_max - psi_min = {hi - lo!r})")
    for polymer in params["polymers"]:
        polymer_block_lengths(params, polymer)
    return params


def _has_random_copolymer(params):
    return any(isinstance(block, dict) and "monomer_fractions" in block
               for polymer in params.get("polymers", []) for block in polymer.get("blocks", []))


def _reject_grand_canonical(params: Dict[str, Any], method: str) -> None:
    if any("chemical_potential" in polymer for polymer in params["polymers"]):
        raise ValidationError(f"Grand-canonical polymers ('chemical_potential') are supported in CL-FTS only, "
                              f"not in {method}")


def validate_electrostatics(params: Dict[str, Any]) -> None:
    """Charges, coupling and electroneutrality of the CL-FTS 'electrostatics' input."""
    electrostatics = params["electrostatics"]
    if not isinstance(electrostatics, dict):
        raise ValidationError("'electrostatics' must be a dict or None")
    unknown = set(electrostatics) - {"charges", "bjerrum_length", "coupling", "psi_dt_scaling", "screening_length",
                                     "ions", "counterion", "n_segments", "fixed_charge"}
    if unknown:
        raise ValidationError(f"'electrostatics' has unknown keys: {sorted(unknown)}")
    charges = electrostatics.get("charges")
    if not isinstance(charges, dict) or not charges:
        raise ValidationError("'electrostatics.charges' must be a dict {monomer type: valence}")
    unknown = set(charges) - set(params["segment_lengths"])
    if unknown:
        raise ValidationError(f"'electrostatics.charges' has unknown monomer types: {sorted(unknown)}")
    for name, valence in charges.items():
        _finite_number(valence, f"electrostatics.charges.{name}")
    if ("bjerrum_length" in electrostatics) == ("coupling" in electrostatics):
        raise ValidationError("Give exactly one of 'electrostatics.bjerrum_length' (l_B/R0) and "
                              "'electrostatics.coupling' (E)")
    for key in ("bjerrum_length", "coupling"):
        if key in electrostatics:
            _finite_number(electrostatics[key], "electrostatics." + key, nonnegative=True)
    # E = 4π (l_B/R₀) N² N̄^(1/2) and the point ions need the number of segments N of the reference
    # chain. Discrete chains have N = 1/ds; ds of continuous chains is a contour step, so N is given
    ions = electrostatics.get("ions") or {}
    if "n_segments" in electrostatics:
        _finite_number(electrostatics["n_segments"], "electrostatics.n_segments", positive=True)
    if str(params["chain_model"]).lower() == "discrete":
        n_segments = 1.0 / params["ds"]
        if "n_segments" in electrostatics and abs(electrostatics["n_segments"] - n_segments) > 1e-9 * n_segments:
            raise ValidationError(f"'electrostatics.n_segments' of discrete chains is 1/ds = {n_segments:g}")
        if ("bjerrum_length" in electrostatics or ions) and abs(n_segments - round(n_segments)) > 1e-9:
            raise ValidationError("'electrostatics.bjerrum_length' and 'ions' need an integer 1/ds; give 'coupling' "
                                  "and no point ions instead")
    elif "n_segments" not in electrostatics:
        if "bjerrum_length" in electrostatics:
            raise ValidationError("'electrostatics.bjerrum_length' needs discrete chains (N = 1/ds) or the number of "
                                  "segments N of the reference chain, 'electrostatics.n_segments'; or give 'coupling' "
                                  "(E = 4 pi l_B rho0 N R0^2)")
        if ions:
            raise ValidationError("'electrostatics.ions' need the number of segments N of the reference chain: give "
                                  "'electrostatics.n_segments' for continuous chains")
    # Point ions: no volume, no chi, only their smeared charge
    if not isinstance(ions, dict):
        raise ValidationError("'electrostatics.ions' must be a dict "
                              "{ion name: {valence, concentration, smearing_length}}")
    for name, ion in ions.items():
        if not isinstance(name, str) or name in params["segment_lengths"]:
            raise ValidationError(f"Ion name {name!r} must be a string that is not a monomer type")
        keys = set(ion) if isinstance(ion, dict) else set()
        allowed = {"valence", "concentration", "chemical_potential", "smearing_length"}
        if not isinstance(ion, dict) or keys - allowed or not {"valence", "smearing_length"} <= keys or \
                ("concentration" in ion) == ("chemical_potential" in ion):
            raise ValidationError(f"'electrostatics.ions.{name}' needs 'valence', 'smearing_length' and one of "
                                  "'concentration' and 'chemical_potential' (grand canonical)")
        _finite_number(ion["valence"], f"electrostatics.ions.{name}.valence")
        if ion["valence"] == 0:
            raise ValidationError(f"'electrostatics.ions.{name}.valence' must not be zero")
        if "concentration" in ion:
            _finite_number(ion["concentration"], f"electrostatics.ions.{name}.concentration", nonnegative=True)
        else:
            _finite_number(ion["chemical_potential"], f"electrostatics.ions.{name}.chemical_potential")
        _finite_number(ion["smearing_length"], f"electrostatics.ions.{name}.smearing_length", positive=True)
    counterion = electrostatics.get("counterion", "")
    if counterion not in ("", None) and (counterion not in ions or "concentration" not in ions[counterion]):
        raise ValidationError(f"'electrostatics.counterion' {counterion!r} is not one of the canonical 'ions' "
                              "(with a concentration)")
    _finite_number(electrostatics.get("psi_dt_scaling", 1.0), "electrostatics.psi_dt_scaling", positive=True)
    if electrostatics.get("screening_length") is not None:
        _finite_number(electrostatics["screening_length"], "electrostatics.screening_length", positive=True)
    # Point charges have a grid-dependent self-energy; the charges are smeared
    smearing = params.get("smearing")
    if smearing is None:
        raise ValidationError("'electrostatics' needs 'smearing' (the smeared charge distribution)")
    if smearing.get("type", "gaussian") == "gaussian":
        a_int = smearing.get("a_int")
        lengths = a_int if isinstance(a_int, dict) else dict.fromkeys(charges, a_int)
        point_charges = sorted(name for name, valence in charges.items() if valence != 0 and not lengths[name] > 0)
        if point_charges:
            raise ValidationError(f"'electrostatics' needs a positive smearing length 'a_int' for the charged "
                                  f"types {point_charges}; a_int = 0 leaves point charges")
    if electrostatics.get("fixed_charge") is not None:
        fixed = np.asarray(electrostatics["fixed_charge"])
        if fixed.size != int(np.prod(params["nx"])) or not np.all(np.isfinite(fixed)) or np.iscomplexobj(fixed):
            raise ValidationError("'electrostatics.fixed_charge' needs one real, finite charge density per grid point")
    if _has_random_copolymer(params):
        raise ValidationError("'electrostatics' does not support random copolymers")
    # Electroneutrality: Σ_i z_i φ_i + Σ_s z_s c_s = 0 with the overall segment fraction φ_i of each type
    # and the point ions, counterions included. An implicit electrolyte (screening_length) screens a net
    # charge, so a screened system need not be neutral; with charged grand-canonical species the uniform
    # mode of ψ neutralizes the system
    concentrations = point_ion_concentrations(params)
    if electrostatics.get("screening_length") is not None:
        return
    if open_charges(params):
        # The Donnan potential neutralizes the canonical charge only with grand-canonical
        # species of the opposite sign (of both signs for a neutral canonical part)
        canonical = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
        signs = {np.sign(ion["valence"]) for ion in ions.values() if "chemical_potential" in ion}
        for polymer in params["polymers"]:
            if "chemical_potential" in polymer:
                charge = _chain_charge(params, polymer, charges)
                if charge != 0.0:
                    signs.add(np.sign(charge))
        needed = {1.0, -1.0} if abs(canonical) <= 1e-14 else {-np.sign(canonical)}
        if not needed <= signs:
            raise ValidationError(f"The grand-canonical species cannot neutralize the system (canonical net charge "
                                  f"{canonical:.3e}): without screening they need charges of "
                                  + ("both signs" if len(needed) == 2 else "the opposite sign"))
        return
    total = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
    if abs(total) > 1e-10:
        raise ValidationError(f"The system is not electroneutral: sum of z_i phi_i = {total:.3e}"
                              + ("; add point ions ('electrostatics.ions') as counterions" if not ions else ""))


def reference_segments(params: Dict[str, Any], default: Any = None) -> Any:
    """Segments N of the reference chain: 1/ds of discrete chains, given for continuous chains."""
    if str(params["chain_model"]).lower() == "discrete":
        return round(1.0 / params["ds"])
    return (params.get("electrostatics") or {}).get("n_segments", default)


def _chain_charge(params: Dict[str, Any], polymer: Dict[str, Any], charges: Dict[str, float]) -> float:
    """Net charge of a chain per segment of the reference chain, Σ_blocks z_i α_i (zero within round-off)."""
    lengths = polymer_block_copy_lengths(params, polymer)
    charge = sum(charges.get(block["type"], 0.0) * length for block, length in zip(polymer["blocks"], lengths))
    scale = sum(abs(charges.get(block["type"], 0.0)) * length for block, length in zip(polymer["blocks"], lengths))
    return 0.0 if abs(charge) <= 1e-12 * scale else charge


def open_charges(params: Dict[str, Any]) -> bool:
    """True if a grand-canonical species (polymer or point ion) carries a net charge."""
    electrostatics = params["electrostatics"]
    charges = electrostatics["charges"]
    if any("chemical_potential" in ion for ion in (electrostatics.get("ions") or {}).values()):
        return True
    # The uniform mode of ψ does not change the weight of a chain without net charge
    return any("chemical_potential" in polymer and _chain_charge(params, polymer, charges) != 0.0
               for polymer in params["polymers"])


def net_polymer_charge(params: Dict[str, Any]) -> float:
    """Σ_i z_i φ_i of the canonical polymers and the mean fixed charge density, the net charge per
    segment of the reference density without the point ions."""
    charges = params["electrostatics"]["charges"]
    fixed = params["electrostatics"].get("fixed_charge")
    total = 0.0 if fixed is None else float(np.mean(fixed))
    for polymer in params["polymers"]:
        if "chemical_potential" in polymer:
            continue
        lengths = polymer_block_copy_lengths(params, polymer)
        alpha = sum(lengths)
        for block, length in zip(polymer["blocks"], lengths):
            total += charges.get(block["type"], 0.0) * polymer["volume_fraction"] * length / alpha
    return total


def point_ion_concentrations(params: Dict[str, Any]) -> Dict[str, float]:
    """Concentration of each canonical point ion (per segment of the reference density), counterions included.

    The given concentrations are added salt; grand-canonical ions (chemical_potential) have none.
    The counterion species, 'electrostatics.counterion' or the only canonical ion species whose
    valence has the opposite sign of the net charge, receives the concentration that neutralizes
    the canonical polymers and the given ions; 'counterion': None, or no such species, adds none
    (the electroneutrality check of validate_electrostatics then applies).
    """
    electrostatics = params["electrostatics"]
    ions = {name: ion for name, ion in (electrostatics.get("ions") or {}).items() if "concentration" in ion}
    concentrations = {name: float(ion["concentration"]) for name, ion in ions.items()}
    total = net_polymer_charge(params) + sum(ions[name]["valence"] * c for name, c in concentrations.items())
    counterion = electrostatics.get("counterion", "")
    if not ions or counterion is None or abs(total) <= 1e-14:
        return concentrations
    if counterion == "":
        candidates = [name for name, ion in ions.items() if ion["valence"] * total < 0]
        if not candidates:
            return concentrations
        if len(candidates) > 1:
            raise ValidationError(f"The net charge {total:.3e} has {len(candidates)} ion species of opposite sign; "
                                  "name the counterion with 'electrostatics.counterion' (None: no counterions)")
        counterion = candidates[0]
    added = -total / ions[counterion]["valence"]
    if added < 0:
        raise ValidationError(f"The counterion {counterion!r} has the sign of the net charge {total:.3e}")
    concentrations[counterion] += added
    return concentrations


def validate_gibbs_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """The `gibbs_ensemble` section of a Gibbs-ensemble CL-FTS input (gibbs_ensemble.py); returns the
    copy with the names of normalize_params."""
    params = normalize_params(params)
    gibbs = params.get("gibbs_ensemble")
    if not isinstance(gibbs, dict):
        raise ValidationError("Missing required dict: 'gibbs_ensemble'")
    unknown = set(gibbs) - {"boxes", "cycles", "cycle_steps", "exchange_rate", "skip_steps"}
    if unknown:
        raise ValidationError(f"'gibbs_ensemble' has unknown keys: {sorted(unknown)}")
    _integer(gibbs.get("cycles"), "gibbs_ensemble.cycles", minimum=1)
    _integer(gibbs.get("cycle_steps"), "gibbs_ensemble.cycle_steps", minimum=1)
    _finite_number(gibbs.get("exchange_rate", 0.2), "gibbs_ensemble.exchange_rate", positive=True)
    _integer(gibbs.get("skip_steps", 0), "gibbs_ensemble.skip_steps", minimum=0)
    if gibbs.get("skip_steps", 0) >= gibbs["cycle_steps"]:
        raise ValidationError("'gibbs_ensemble.skip_steps' must be smaller than 'cycle_steps'")
    recording = params.get("recording")
    if not isinstance(recording, dict) or "dir" not in recording:
        raise ValidationError("The Gibbs ensemble needs 'recording.dir' for its boxes and history")
    _integer(recording.get("sample_interval", 1), "recording.sample_interval")
    if gibbs["cycle_steps"] % recording.get("sample_interval", 1) != 0:
        raise ValidationError("'gibbs_ensemble.cycle_steps' must be a multiple of 'recording.sample_interval'")
    if any(str(b).lower() != "periodic" for b in params.get("bc") or []):
        raise ValidationError("The Gibbs ensemble needs periodic boxes (the osmotic pressure)")
    polymers = params.get("polymers") or []
    for i, polymer in enumerate(polymers):
        if "chemical_potential" in polymer or polymer.get("initial_conditions"):
            raise ValidationError(f"polymers[{i}]: the Gibbs ensemble exchanges free canonical chains; grand-canonical "
                                  "and grafted polymers are not supported")
    electrostatics = params.get("electrostatics") or {}
    if electrostatics.get("fixed_charge") is not None:
        raise ValidationError("The Gibbs ensemble does not support 'electrostatics.fixed_charge' "
                              "(the amounts and the neutrality of the boxes do not count it)")
    canonical_ions = [name for name, ion in (electrostatics.get("ions") or {}).items()
                      if isinstance(ion, dict) and "concentration" in ion]
    if canonical_ions and str(params.get("chain_model")).lower() != "discrete" and "n_segments" not in electrostatics:
        raise ValidationError("'electrostatics.ions' need the number of segments N of the reference chain: give "
                              "'electrostatics.n_segments' (continuous chains)")
    for key in ("nx", "lx"):
        if key not in params:
            raise ValidationError(f"Missing required parameter: '{key}'")
    nx, lx = params["nx"], params["lx"]
    if not isinstance(nx, (list, tuple, np.ndarray)) or not isinstance(lx, (list, tuple, np.ndarray)) or len(lx) != len(nx):
        raise ValidationError("'lx', the cell of both boxes, needs one length per dimension of 'nx'")
    _check_angles(params.get("angles"), len(nx))   # before the volumes of the boxes are computed
    for length in lx:
        _finite_number(length, "lx", positive=True)
    boxes = gibbs.get("boxes")
    if not isinstance(boxes, list) or len(boxes) != 2:
        raise ValidationError("'gibbs_ensemble.boxes' must be a list of two dicts")
    for b, box in enumerate(boxes):
        name = f"gibbs_ensemble.boxes[{b}]"
        if not isinstance(box, dict) or set(box) - {"volume", "volume_fractions", "ion_concentrations"} or \
                "volume_fractions" not in box:
            raise ValidationError(f"'{name}' must be a dict with 'volume_fractions' and optionally 'volume' and "
                                  "'ion_concentrations'; 'lx' is the cell of both boxes")
        if "volume" in box:
            _finite_number(box["volume"], name + ".volume", positive=True)
        fractions = box["volume_fractions"]
        if not isinstance(fractions, (list, tuple)) or len(fractions) != len(polymers):
            raise ValidationError(f"'{name}.volume_fractions' needs one value per polymer")
        for value in fractions:
            _finite_number(value, name + ".volume_fractions", nonnegative=True)
        if abs(sum(fractions) - 1.0) > 1e-10:
            raise ValidationError(f"'{name}.volume_fractions' must sum to 1, got {sum(fractions)}")
        concentrations = box.get("ion_concentrations")
        if concentrations is not None:
            if not isinstance(concentrations, dict) or set(concentrations) != set(canonical_ions):
                raise ValidationError(f"'{name}.ion_concentrations' needs the concentration of every canonical point "
                                      f"ion {canonical_ions}")
            for ion, value in concentrations.items():
                _finite_number(value, f"{name}.ion_concentrations.{ion}", nonnegative=True)
    return params
