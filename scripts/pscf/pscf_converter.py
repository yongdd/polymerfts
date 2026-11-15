#!/usr/bin/env python3
"""Convert polymerfts and PSCF periodic SCFT inputs without running simulations.

    python scripts/pscf/pscf_converter.py export fields.mat -o pscf-input
    python scripts/pscf/pscf_converter.py import pscf-input/out/omega --rho pscf-input/out/rho \
        --metadata pscf-input/metadata.json -o imported.mat
    python scripts/pscf/pscf_converter.py import-script pscf-input/param \
        --command pscf-input/command --metadata pscf-input/metadata.json -o polymerfts-job

PSCF reference: https://dmorse.github.io/pscfpp-man/user_field_rgrid_page.html
"""

import argparse
import ast
import builtins
import copy
import json
import math
from pathlib import Path
from pprint import pformat
import runpy
import shlex
import shutil
import string

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError


# Reuse input aliases and validation without importing the compiled package.
_VALIDATION = runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/python/utils/validation.py"))
_MAT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/python/utils/mat_fields.py"))


def real(value):
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return float(str(value).replace("D", "e").replace("d", "e"))


def vector(value, label, *, integer=False):
    data = np.asarray(value)
    if np.iscomplexobj(data) or data.dtype.kind == "b":
        raise ValueError(f"{label} must contain real numbers.")
    data = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(data).all() or (data <= 0).any():
        raise ValueError(f"{label} must contain finite positive numbers.")
    if integer and not np.equal(data, np.floor(data)).all():
        raise ValueError(f"{label} must contain integers.")
    return [int(v) if integer else float(v) for v in data]


def validate_cell(nx, lx, angles):
    nx, lx = vector(nx, "nx", integer=True), vector(lx, "lx")
    if len(nx) not in (1, 2, 3) or len(lx) != len(nx):
        raise ValueError("nx and lx must have matching dimensions between 1 and 3.")
    angles = [90.] * 3 if angles is None else vector(angles, "angles")
    if not angles and len(nx) < 3:
        angles = [90.] * 3
    if len(angles) == 1 and len(nx) < 3:
        angles = [90., 90., angles[0]]
    if len(angles) != 3 or any(a >= 180 for a in angles):
        raise ValueError("angles must contain alpha, beta and gamma in degrees, between 0 and 180.")
    if len(nx) < 3 and not np.allclose(angles[:2], 90, rtol=0, atol=1e-9):
        raise ValueError("1D/2D cells require alpha = beta = 90 degrees.")
    if len(nx) == 1 and not math.isclose(angles[2], 90, abs_tol=1e-9):
        raise ValueError("1D cells require all angles to be 90 degrees.")
    ca, cb, cg = np.cos(np.deg2rad(angles))
    if 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg <= 1e-12:
        raise ValueError("Cell angles define a degenerate or impossible cell.")
    return nx, lx, angles


def lattice_geometry(lattice, cell, dim):
    """PSCF cell parameters -> polymerfts lengths and crystallographic angles.

    PSCF triclinic parameters are a,b,c,phi,theta,gamma (angles in radians).
    Its c vector has Cartesian direction (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
    """
    lattice = lattice.lower()
    counts = {1: {"lamellar": 1},
              2: {"square": 1, "rectangular": 2, "hexagonal": 1, "rhombic": 2, "oblique": 3},
              3: {"cubic": 1, "tetragonal": 2, "orthorhombic": 3, "hexagonal": 2,
                  "rhombohedral": 2, "monoclinic": 4, "triclinic": 6}}
    if dim not in counts or lattice not in counts[dim] or len(cell) != counts[dim][lattice]:
        raise ValueError(f"Invalid {dim}D lattice or cell parameter count: {lattice} ({len(cell)}).")
    if not np.isfinite(cell).all():
        raise ValueError("Cell parameters must be finite.")
    p, angles = list(cell), [90., 90., 90.]
    if dim == 1:
        lx = [p[0]]
    elif dim == 2:
        lx = [p[0], p[0]] if lattice in ("square", "hexagonal", "rhombic") else p[:2]
        if lattice == "hexagonal": angles[2] = 120.
        elif lattice == "rhombic": angles[2] = math.degrees(p[1])
        elif lattice == "oblique": angles[2] = math.degrees(p[2])
    else:
        if lattice == "cubic": lx = [p[0]] * 3
        elif lattice in ("tetragonal", "hexagonal"): lx = [p[0], p[0], p[1]]
        elif lattice == "rhombohedral": lx = [p[0]] * 3
        else: lx = p[:3]
        if lattice == "hexagonal": angles[2] = 120.
        elif lattice == "rhombohedral": angles = [math.degrees(p[1])] * 3
        elif lattice == "monoclinic": angles[1] = math.degrees(p[3])
        elif lattice == "triclinic":
            phi, theta, gamma = p[3:]
            # Require the conventional right-handed representation.
            if not 0 < gamma < math.pi or not 0 <= theta < math.pi / 2:
                raise ValueError("Triclinic gamma must be in (0, pi), theta in [0, pi/2).")
            cb = math.sin(theta) * math.cos(phi)
            ca = math.sin(theta) * math.cos(phi - gamma)
            angles = [math.degrees(math.acos(np.clip(ca, -1, 1))),
                      math.degrees(math.acos(np.clip(cb, -1, 1))), math.degrees(gamma)]
    _, lx, angles = validate_cell([1] * dim, lx, angles)
    return lx, angles


def spglib_call(spglib, function, *args):
    """Call spglib with its raising error handling, which spglib 2.7+ asks for (with a
    DeprecationWarning otherwise) and makes the only one in 3.0, without changing the
    handling for other code. spglib's SPGLIB_OLD_ERROR_HANDLING variable takes precedence."""
    errors = getattr(spglib, "error", None)
    old = getattr(errors, "OLD_ERROR_HANDLING", None)
    if old is not None:
        errors.OLD_ERROR_HANDLING = False
    try:
        return function(*args)
    finally:
        if old is not None:
            errors.OLD_ERROR_HANDLING = old


def choose_lattice(params, override=None):
    nx, lx, angles = validate_cell(params["nx"], params["lx"], params.get("angles", [90.] * 3))
    dim = len(nx)
    system = params.get("crystal_system")
    if system is not None and not isinstance(system, str):
        raise ValueError("crystal_system must be a string.")
    if params.get("space_group") is not None and not isinstance(params["space_group"], dict):
        raise ValueError("space_group must be a dictionary.")
    if params.get("space_group") and not override:
        try:
            import spglib
        except ImportError:
            raise ValueError("Space-group inputs need Python spglib, or an explicit --lattice.") from None
        group = params["space_group"]
        symbol = group.get("symbol")
        if symbol is not None and not isinstance(symbol, str):
            raise ValueError("space_group.symbol must be a string.")
        # The Hall settings of the symbol; like SCFT, a symbol with several needs the hall_number
        key = None if symbol is None else symbol.replace(" ", "").replace("_", "")
        settings = [] if key is None else [h for h in range(1, 531) if spglib_call(
            spglib, spglib.get_spacegroup_type, h).international_short.replace("_", "") == key]
        hall = group.get("hall_number")
        if hall is not None:
            numbers = vector(hall, "hall_number", integer=True)
            if len(numbers) != 1 or numbers[0] > 530:
                raise ValueError("hall_number must be an integer between 1 and 530.")
            hall = numbers[0]
            if key is not None and hall not in settings:
                raise ValueError(f"space_group.hall_number {hall} is a setting of "
                                 f"{spglib_call(spglib, spglib.get_spacegroup_type, hall).international_short}, "
                                 f"not of {symbol}.")
        elif len(settings) > 1:
            raise ValueError(f"space_group.symbol {symbol} has several Hall settings ({', '.join(map(str, settings))}); "
                             "give space_group.hall_number, as SCFT requires, or --lattice.")
        else:
            hall = settings[0] if settings else None
        if hall is None:
            raise ValueError("Could not identify the space group. Specify --lattice explicitly.")
        sg = spglib_call(spglib, spglib.get_spacegroup_type, hall)
        number = sg.number
        system = ("Triclinic" if number <= 2 else "Monoclinic" if number <= 15 else
                  "Orthorhombic" if number <= 74 else "Tetragonal" if number <= 142 else
                  "Trigonal" if number <= 167 else "Hexagonal" if number <= 194 else "Cubic")
        if system == "Monoclinic":
            # PSCF's monoclinic lattice frees beta, the angle of the unique axis b. A setting with
            # the unique axis a or c keeps its fixed cell as a triclinic lattice; relaxing it would
            # free every angle in PSCF. The unique axis is the 2-fold axis or mirror normal, which
            # every operation keeps in place (R[u][u] = det R without mixing it with other axes)
            rotations = spglib_call(spglib, spglib.get_symmetry_from_database, hall)["rotations"]
            unique = next(u for u in range(3) if all(
                round(np.linalg.det(r)) == r[u][u] and all(r[u][k] == 0 and r[k][u] == 0 for k in range(3) if k != u)
                for r in rotations))
            if unique != 1:
                if params.get("relax_cell", False):
                    raise ValueError(f"PSCF's monoclinic lattice relaxes beta (unique axis b), but space group "
                                     f"{sg.international_short} (Hall {hall}) has the unique axis {'abc'[unique]}. "
                                     "Export a fixed cell, use the unique-axis-b setting of the group, or give "
                                     "--lattice triclinic (PSCF then relaxes every angle).")
                system = "Triclinic"
    if override:
        lattice = override.lower()
    elif dim == 1:
        lattice = "lamellar"
    elif dim == 2:
        if (system == "Orthorhombic" and params.get("relax_cell", False)
                and not math.isclose(angles[2], 90, abs_tol=1e-9)):
            raise ValueError("Flexible skewed 2D cells with fixed gamma cannot be transferred; use relax_cell=False or choose --lattice explicitly.")
        lattice = ("hexagonal" if system == "Hexagonal2D" else "oblique" if system == "Oblique2D" else
                   "rectangular" if math.isclose(angles[2], 90, abs_tol=1e-9) else
                   "hexagonal" if math.isclose(angles[2], 120, abs_tol=1e-9) and math.isclose(lx[0], lx[1]) else "oblique")
    else:
        if (not system and not override and params.get("relax_cell", False)
                and not np.allclose(angles, 90, rtol=0, atol=1e-9)):
            raise ValueError("Flexible skewed 3D inputs need an explicit crystal_system or --lattice; angle constraints cannot be inferred.")
        lattice = (system or ("Orthorhombic" if np.allclose(angles, 90, rtol=0, atol=1e-9) else "Triclinic")).lower()
        if lattice == "trigonal":
            lattice = "hexagonal" if np.allclose(angles, [90, 90, 120]) else "rhombohedral"
    if lattice in ("lamellar", "square", "cubic"): cell = lx[:1]
    elif lattice in ("tetragonal", "hexagonal"): cell = lx[:1] if dim == 2 else [lx[0], lx[2]]
    elif lattice in ("rectangular", "orthorhombic"): cell = lx[:]
    elif lattice == "rhombohedral": cell = [lx[0], math.radians(angles[0])]
    elif lattice == "monoclinic": cell = lx[:] + [math.radians(angles[1])]
    elif lattice == "rhombic": cell = [lx[0], math.radians(angles[2])]
    elif lattice == "oblique": cell = lx[:] + [math.radians(angles[2])]
    elif lattice == "triclinic":
        ca, cb, cg = np.cos(np.deg2rad(angles))
        sg = math.sin(math.radians(angles[2]))
        cy = (ca - cb * cg) / sg
        cz = math.sqrt(max(0., 1 - cb * cb - cy * cy))
        theta = math.atan2(math.hypot(cb, cy), cz)
        phi = math.atan2(cy, cb) if theta > 1e-14 else 0.
        cell = lx[:] + [phi, theta, math.radians(angles[2])]
    else:
        raise ValueError(f"Unsupported lattice: {lattice}.")
    check_lx, check_angles = lattice_geometry(lattice, cell, dim)
    if not np.allclose(check_lx, lx, rtol=1e-8, atol=1e-10) or not np.allclose(check_angles, angles, rtol=0, atol=1e-8):
        raise ValueError(f"Cell geometry does not match the requested lattice {lattice}.")
    return lattice, cell


def read_rgrid(path):
    """Read a format-1.0 r-grid header and independent, strictly sized field data."""
    with Path(path).open(encoding="utf-8") as handle:
        def line():
            for raw in handle:
                tokens = shlex.split(raw, comments=True)
                if tokens:
                    return tokens
            raise ValueError("Truncated PSCF field header.")

        def entry(label, count, tokens=None):
            tokens = line() if tokens is None else tokens
            if tokens[0] != label:
                raise ValueError(f"Expected '{label}' in r-grid header; found '{tokens[0]}'. Export with WRITE_W_RGRID/WRITE_C_RGRID.")
            values = tokens[1:]
            while len(values) < count:
                values.extend(line())
            if len(values) != count:
                raise ValueError(f"Header '{label}' needs {count} values.")
            return values

        if entry("format", 2) != ["1", "0"]:
            raise ValueError("Only PSCF r-grid format 1 0 is supported.")
        dim = int(entry("dim", 1)[0])
        if dim not in (1, 2, 3):
            raise ValueError("PSCF dimension must be 1, 2 or 3.")
        lattice = entry("crystal_system", 1)[0].lower()
        count = int(entry("N_cell_param", 1)[0])
        if not 1 <= count <= 6:
            raise ValueError("Invalid N_cell_param.")
        cell = [real(v) for v in entry("cell_param", count)]
        tokens, group = line(), ""
        if tokens[0] == "group_name":
            group = entry("group_name", 1, tokens)[0]
            tokens = line()
        n_monomer = int(entry("N_monomer", 1, tokens)[0])
        if n_monomer < 1:
            raise ValueError("N_monomer must be positive.")
        tokens = line()
        if tokens[0] not in ("mesh", "ngrid"):
            raise ValueError("Only full real-space r-grid fields with mesh/ngrid are supported.")
        nx = vector(entry(tokens[0], dim, tokens), "mesh", integer=True)
        lx, angles = lattice_geometry(lattice, cell, dim)
        data = np.loadtxt(handle, ndmin=2, converters=real)
    if data.shape != (math.prod(nx), n_monomer):
        raise ValueError(f"{path}: expected {math.prod(nx)} rows and {n_monomer} columns; got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: fields must be finite real values.")
    # PSCF's first index is fastest; keep physical axes and lengths unchanged.
    fields = np.stack([data[:, i].reshape(nx, order="F") for i in range(n_monomer)])
    return dict(nx=nx, lx=lx, angles=angles, lattice=lattice, cell=cell,
                group_name=group, fields=fields)


def write_rgrid(path, nx, lattice, cell, fields):
    lattice_geometry(lattice, cell, len(nx))
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(f"format 1 0\ndim {len(nx)}\ncrystal_system {lattice}\nN_cell_param {len(cell)}\n")
        handle.write("cell_param " + " ".join(format(x, ".17g") for x in cell) + "\n")
        handle.write(f"N_monomer {len(fields)}\nmesh " + " ".join(map(str, nx)) + "\n")
        np.savetxt(handle, np.column_stack([field.ravel(order="F") for field in fields]), fmt="%.17e")


def read_python_params(path):
    """Read simple module-level parameter expressions, never execute user Python."""
    symbols, params = {}, None
    imports, rebound = {}, set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Reflection reaches objects and modules by names built at run time, which a
    # reading of the source cannot follow.
    reflection = {"getattr", "setattr", "delattr", "vars", "globals", "locals", "exec", "eval",
                  "compile", "__import__", "importlib.import_module"}
    untraceable_error = ("SCFT, LFTS, CLFTS and GibbsEnsemble are followed through module-level import statements "
                         "and calls by those names; reached through an import in a function or under a condition, "
                         "or given to another name, a function or a loop, they can receive params the converter "
                         "does not see.")
    reflection_error = ("Reflection and module internals (getattr, exec, sys.modules, __main__, copy._...) may "
                        "cause parameter mutations that a reading of the script cannot follow; use a saved SCFT result.")

    def callable_name(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id if node.id not in rebound else "")
        if isinstance(node, ast.Attribute):
            owner = callable_name(node.value)
            return owner + "." + node.attr if owner else ""
        return ""

    # These calls consume literal configuration values without modifying them.
    # Resolve imported names, rather than trusting an arbitrary function named SCFT.
    # The simulation constructors deep-copy their params argument; they keep the
    # others (callbacks, an integrator) as given, so those are checked as usual.
    simulations = {"polymerfts.SCFT", "polymerfts.scft.SCFT", "polymerfts.scft.scft.SCFT",
                   "polymerfts.LFTS", "polymerfts.lfts.LFTS", "polymerfts.fts.lfts.LFTS",
                   "polymerfts.CLFTS", "polymerfts.clfts.CLFTS", "polymerfts.fts.clfts.CLFTS",
                   "polymerfts.GibbsEnsemble", "polymerfts.fts.gibbs_ensemble.GibbsEnsemble"}
    # Simulation classes and the modules that hold one
    simulation_modules = {name.rsplit(".", i)[0] for name in simulations for i in range(1, name.count(".") + 1)}
    traced = simulations | simulation_modules
    # How each name, attribute or call appears: the function of a call, the owner of an
    # attribute, in an annotation, or a type in isinstance, issubclass or an `is` test
    called, owners, annotated, type_tests = set(), set(), set(), {}
    # What an import in a function, a class or a compound statement binds each name to;
    # the converter does not resolve these names
    untraced = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called.add(id(node.func))
            if isinstance(node.func, ast.Name) and len(node.args) == 2:
                types = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
                type_tests.update((id(part), node.func) for part in types)
        elif isinstance(node, ast.Attribute):
            owners.add(id(node.value))
        elif isinstance(node, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            type_tests.update((id(part), None) for part in (node.left, *node.comparators))
        annotations = ([node.annotation] if isinstance(node, (ast.arg, ast.AnnAssign)) else
                       [node.returns] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [])
        annotated.update(id(part) for annotation in annotations if annotation is not None
                         for part in ast.walk(annotation))
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body \
                and not (isinstance(node, ast.ImportFrom) and node.level):
            for alias in node.names:
                if isinstance(node, ast.ImportFrom):
                    name, target = alias.asname or alias.name, f"{node.module}.{alias.name}"
                else:  # import a.b binds a
                    name = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else name
                untraced.setdefault(name, set()).add(target)
    # The names, attributes and calls the rules below can concern, found once
    imported = {name for statement in tree.body if isinstance(statement, (ast.Import, ast.ImportFrom))
                for alias in statement.names
                for name in [alias.asname or (alias.name.split(".")[0] if isinstance(statement, ast.Import)
                                              else alias.name)]}
    may_hide, reach_untraced, from_files, main_imports = set(), set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "__main__" in (
                [(node.module or "").partition(".")[0]] if isinstance(node, ast.ImportFrom)
                else [alias.name.partition(".")[0] for alias in node.names]):
            main_imports.add(id(node))
        if not (isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load)):
            continue
        if isinstance(node, ast.Attribute) and node.attr == "from_file":
            from_files.add(id(node))
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in untraced:
            reach_untraced.add(id(node))
        if (isinstance(root, ast.Name) and (root.id in imported or root.id in untraced)
                and id(node) not in owners and id(node) not in annotated):
            may_hide.add(id(node))
    # Shape-only functions; on an array from array_makers they run ndarray code.
    shape_readers = {"numpy.reshape", "numpy.broadcast_to"}
    # Without `like=` these return a base-class ndarray (or a float), whose arithmetic
    # and reshape pass no operand or shape list on to code of the script.
    array_makers = {*("numpy." + name for name in ("array", "asarray", "zeros", "ones", "empty", "full")),
                    *("numpy.random." + name for name in ("normal", "uniform", "random"))}
    readers = {"print", "len", "sum", "min", "max",
               "range", "list", "tuple", "dict",
               *("numpy." + name for name in ("zeros", "ones", "empty", "full", "array", "asarray", "prod", "sum", "mean", "min", "max")),
               *("numpy.random." + name for name in ("normal", "uniform", "random"))}
    numeric_readers = {"sum", "min", "max", *("numpy." + name for name in ("array", "asarray", "prod", "sum", "mean", "min", "max"))}
    numpy_math = {"numpy." + name for name in ("sqrt", "cbrt", "square", "exp", "log", "sin", "cos", "tan",
                                               "abs", "absolute", "linspace", "arange")}

    def numeric(value):
        return (isinstance(value, (int, float, complex))
                or isinstance(value, (list, tuple)) and all(numeric(item) for item in value))

    def flat_scalars(value, mappings=True):
        # An element of such a container is immutable, and arithmetic or a slice
        # makes a new container: neither can alias a parameter container.
        scalar = (int, float, complex, str, type(None))
        if isinstance(value, scalar):
            return True
        if isinstance(value, dict):
            items = value.values() if mappings else None
        else:
            items = value if isinstance(value, (list, tuple)) else None
        return items is not None and all(isinstance(item, scalar) for item in items)

    # The checks revisit every function body at each later statement; cache the
    # names in each node of the tree, which stays alive while it is read.
    name_cache = {}

    def names_in(node):
        found = name_cache.get(id(node))
        if found is None:
            found = frozenset((node.id,) if isinstance(node, ast.Name) else ()).union(
                *(names_in(child) for child in ast.iter_child_nodes(node)))
            name_cache[id(node)] = found
        return found

    # `untrusted` holds the names whose module literal may not be their value here:
    # locals of a function or comprehension, and names a compound statement rebinds.
    def uses(node, untrusted):
        return not names_in(node).isdisjoint(untrusted)

    def evaluates_to_flat_scalars(node, untrusted=frozenset(), mappings=True):
        if not uses(node, untrusted):
            try:
                return flat_scalars(evaluate(node), mappings)
            except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                pass
        # An element or slice of a flat container is a scalar or a new list, whatever the index.
        return isinstance(node, ast.Subscript) and evaluates_to_flat_scalars(node.value, untrusted)

    def new_array(node, untrusted=frozenset()):
        return (isinstance(node, ast.Call) and not uses(node.func, untrusted)
                and callable_name(node.func) in array_makers
                and all(keyword.arg in ("dtype", "size") for keyword in node.keywords))

    def is_array(node, untrusted):
        # A new array, or a module name whose last binding was one
        return new_array(node, untrusted) or (isinstance(node, ast.Name) and node.id in array_names
                                              and node.id not in untrusted)

    def array_result(node, untrusted):
        """A new array, or NumPy arithmetic of an array with a list of numbers, which makes another.

        With scalars only, a 0-d array gives an element, which can be any object, and
        matrix multiplication of two vectors gives a scalar. A list makes the result an
        array of at least one dimension, and so does such an array as an operand.
        """
        if new_array(node, untrusted):
            return True
        if not isinstance(node, ast.BinOp) or isinstance(node.op, ast.MatMult):
            return False
        parts = (node.left, node.right)
        made = [array_result(part, untrusted) and isinstance(part, ast.BinOp) for part in parts]
        if not all(done or builtin_operand(part, untrusted) for done, part in zip(made, parts)):
            return False
        if not any(done or is_array(part, untrusted) for done, part in zip(made, parts)):
            return False
        return any(made) or any(evaluates_to_list(part, untrusted) for part in parts)

    def numerical(node, untrusted):
        """Arithmetic of numbers, arrays and NumPy math, which runs no method of the script."""
        if isinstance(node, ast.BinOp):
            return numerical(node.left, untrusted) and numerical(node.right, untrusted)
        if isinstance(node, ast.UnaryOp):
            return numerical(node.operand, untrusted)
        if isinstance(node, ast.Attribute) and not uses(node, untrusted):
            return callable_name(node) in ("numpy.pi", "numpy.e")
        if (isinstance(node, ast.Call) and not node.keywords and not uses(node.func, untrusted)
                and callable_name(node.func) in numpy_math):
            return all(numerical(argument, untrusted) for argument in node.args)
        if (isinstance(node, ast.Call) and not node.keywords and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("mean", "sum", "min", "max", "std") and is_array(node.func.value, untrusted)):
            return all(numerical(argument, untrusted) for argument in node.args)  # such as center.mean()
        return builtin_operand(node, untrusted)

    def evaluates_to_list(node, untrusted):
        if uses(node, untrusted):
            return False
        try:
            return isinstance(evaluate(node), (list, tuple))
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
            return False

    def builtin_operand(node, untrusted):
        """Arithmetic with this operand runs builtin or NumPy code, not a method of the script."""
        if is_array(node, untrusted):
            return True
        if uses(node, untrusted):
            return False
        try:
            evaluate(node)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
            return evaluates_to_flat_scalars(node)
        return True

    def operands(node):
        """The operands of an operator and those it hands to a method of another operand."""
        if isinstance(node, ast.BinOp):
            return [node.left, node.right], [node.left, node.right]
        if isinstance(node, ast.Compare):  # __eq__, __lt__, __contains__, ...
            if all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                return None, None  # identity calls no method
            return [node.left, *node.comparators], [node.left, *node.comparators]
        if isinstance(node, ast.AugAssign):  # target.__iadd__(value), ...
            return [node.target, node.value], [node.value]
        if isinstance(node, ast.Subscript):  # value.__getitem__(key), __setitem__, __delitem__
            return [node.value, node.slice], [node.slice]
        return None, None

    def inert_operation(parts, containers, untrusted, result=False):
        # A parameter container given to an operator of another type reaches that type's
        # method (__rmul__, __eq__, __getitem__, ...) as it is; builtin operators run no
        # code of the script. An object array hands each element's method the other
        # operand converted to an array: a list of scalars arrives as numbers, a dict as
        # itself. With result, the new object (a concatenation, a merged dict) must not
        # hold a container of the operands either.
        if not all(builtin_operand(part, untrusted) for part in parts):
            return False
        arrays = any(is_array(part, untrusted) for part in parts)
        if not (arrays or result):
            return True
        return all(evaluates_to_flat_scalars(part, untrusted, not arrays) for part in parts
                   if references_container(part, containers, untrusted))

    def params_argument(node):
        return (node.args[0] if node.args and not isinstance(node.args[0], ast.Starred)
                else next((keyword.value for keyword in node.keywords if keyword.arg == "params"), None))

    def simulation_params(node, untrusted):
        """The params argument of a simulation constructor, when it is a literal value.

        Its deep copy then meets only builtin objects. Any other object could bring a
        copy hook of the script, and __deepcopy__ receives the memo, which keeps the
        original parameter containers.
        """
        argument = params_argument(node)
        if argument is None or uses(argument, untrusted):
            return None
        try:
            evaluate(argument)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
            return None
        return argument

    def simulation_extras(node, untrusted):
        trusted = simulation_params(node, untrusted)
        return [argument for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
                if argument is not trusted]

    def plain(value):
        # A constant or a dotted name such as float or np.float64: no call, no container
        return all(isinstance(part, (ast.Constant, ast.Name, ast.Attribute, ast.Load)) for part in ast.walk(value))

    def independent_result(node, untrusted=frozenset()):
        name = callable_name(node.func)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"
                and isinstance(node.func.value, ast.Call)
                and callable_name(node.func.value.func) in {"numpy.asarray", "numpy.array", "numpy.zeros",
                                                           "numpy.ones", "numpy.empty", "numpy.full"}):
            return True  # the shape cannot alias the array produced by this call
        if name in {"print", "len", "range",
                    "numpy.zeros", "numpy.ones", "numpy.empty", "numpy.random.normal",
                    "numpy.random.uniform", "numpy.random.random"}:
            return True
        if name in numeric_readers and node.args and not uses(node, untrusted):
            # max and min compare all their arguments and pass them to key=: every argument
            # must be numbers. NumPy reads the first and dispatches on out= and like=; its
            # other arguments may be plain values such as a dtype.
            try:
                if name.startswith("numpy."):
                    return (numeric(evaluate(node.args[0])) and all(plain(argument) for argument in node.args[1:])
                            and all(keyword.arg not in (None, "out", "like", "where") and plain(keyword.value)
                                    for keyword in node.keywords))
                return (all(numeric(evaluate(argument)) for argument in node.args)
                        and all(keyword.arg in ("default", "start") and numeric(evaluate(keyword.value))
                                for keyword in node.keywords))
            except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                pass
        return False

    def references_container(node, containers, untrusted=frozenset()):
        if names_in(node).isdisjoint(containers):
            return False  # each finding below is the name of a container inside node
        if isinstance(node, ast.Lambda):
            # Defaults and the result are checked in their own lexical scopes.
            return False
        # A random-field initializer often receives np.prod(params["nx"]). Its
        # scalar result cannot give the initializer access to the original list.
        if isinstance(node, ast.Call) and independent_result(node, untrusted):
            return False
        if isinstance(node, ast.JoinedStr):
            return False  # an f-string makes a new str
        if isinstance(node, ast.Call) and callable_name(node.func) in simulations:
            # The simulation holds a deep copy of params, and its other arguments as given.
            return any(references_container(argument, containers, untrusted)
                       for argument in simulation_extras(node, untrusted))
        if (isinstance(node, ast.Call) and callable_name(node.func) in shape_readers
                and node.args and new_array(node.args[0], untrusted)):
            return references_container(node.args[0], containers, untrusted)  # a view; the shape is not kept
        if isinstance(node, ast.BinOp) and inert_operation(operands(node)[0], containers, untrusted, result=True):
            return False  # a new object without a container of the operands
        if isinstance(node, ast.Compare) and (operands(node)[0] is None
                                              or inert_operation(operands(node)[0], containers, untrusted)):
            return False  # a bool, or a new array of them
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # The iterator bindings are checked separately by scoped_nodes. Only
            # output expressions can put a known container into the new result.
            outputs = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            bound = comprehension_bindings(node)
            return any(references_container(output, containers - bound, untrusted | bound) for output in outputs)
        if isinstance(node, ast.Subscript):
            try:
                if not uses(node, untrusted) and not isinstance(evaluate(node), (dict, list, tuple)):
                    return False
            except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                pass
            # An index unknown statically still selects a scalar, or slices a new list.
            if evaluates_to_flat_scalars(node.value, untrusted):
                return False
        return ((isinstance(node, ast.Name) and node.id in containers)
                or any(references_container(child, containers, untrusted) for child in ast.iter_child_nodes(node)))

    def container_ids(value):
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        else:
            return set()
        return {id(value)}.union(*(container_ids(child) for child in children))

    def bound_values(target, value):
        """Values that assigning value to target gives to names: unpacking binds the items."""
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (dict, list, tuple)):
            items = list(value)
            if (len(items) == len(target.elts)
                    and not any(isinstance(element, ast.Starred) for element in target.elts)):
                return [bound for element, item in zip(target.elts, items)
                        for bound in bound_values(element, item)]
            return items  # a starred name gets a new list of the remaining items
        return [value]

    def evaluate(node):
        if isinstance(node, ast.Name):
            # Keep literal containers shared so aliases cannot conceal mutations.
            return symbols[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(v) for v in node.elts]
        if isinstance(node, ast.Dict):
            return {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Slice):
            return slice(*(None if value is None else evaluate(value) for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            container, index = evaluate(node.value), evaluate(node.slice)
            if isinstance(container, (dict, list, tuple)) and isinstance(index, (str, int, slice)):
                return container[index]
        if isinstance(node, ast.Call):
            name = callable_name(node.func)
            constructor = {"list": list, "tuple": tuple, "dict": dict}.get(name)
            if constructor is not None and len(node.args) <= 1 and (not node.keywords or name == "dict"):
                # Shallow copies must retain their nested aliases in symbols.
                value = constructor(*(evaluate(arg) for arg in node.args))
                for keyword in node.keywords:
                    if keyword.arg is None:
                        value.update(evaluate(keyword.value))
                    else:
                        value[keyword.arg] = evaluate(keyword.value)
                return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            a, b = evaluate(node.left), evaluate(node.right)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError("Only numerical arithmetic is supported.")
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.Pow) and abs(b) <= 100: return a ** b
        raise ValueError("Use literal params with simple arithmetic, or a saved SCFT result/configuration file.")

    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def comprehension_values(node):
        return (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)

    def comprehension_bindings(node):
        return {part.id for generator in node.generators for part in ast.walk(generator.target)
                if isinstance(part, ast.Name)}

    def scope_nodes(node):
        yield node
        if isinstance(node, comprehensions):
            # Iteration targets belong to the comprehension, not its surrounding
            # function. Assignment expressions in its values still bind outside.
            for generator in node.generators:
                for child in (generator.iter, *generator.ifs):
                    yield from scope_nodes(child)
            for child in comprehension_values(node):
                yield from scope_nodes(child)
        elif not isinstance(node, definitions):
            for child in ast.iter_child_nodes(node):
                yield from scope_nodes(child)

    def bound_names(node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return {node.id}
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            return {node.name}
        if isinstance(node, ast.MatchMapping) and node.rest is not None:
            return {node.rest}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {node.name}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                    for alias in node.names}
        return set()

    def scope_info(node):
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        nodes = [part for item in body for part in scope_nodes(item)]
        declared = {name for part in nodes if isinstance(part, ast.Global) for name in part.names}
        local = set().union(*(bound_names(part) for part in nodes))
        if not isinstance(node, ast.ClassDef):
            local.update(arg.arg for arg in ast.walk(node.args) if isinstance(arg, ast.arg))
        local.difference_update(declared)
        return body, local, declared

    def scoped_nodes(node, hidden=frozenset(), closure=frozenset()):
        """Nodes with names shadowed in their lexical scope; never execute their bodies."""
        if isinstance(node, comprehensions):
            yield node, hidden
            # Like a nested function, the body cannot see class-local names.
            inner = closure | comprehension_bindings(node)
            for index, generator in enumerate(node.generators):
                # Only the first iterable is evaluated in the enclosing scope.
                iterator_scope = hidden if index == 0 else inner
                yield generator, iterator_scope
                yield from scoped_nodes(generator.iter, iterator_scope, closure if index == 0 else inner)
                yield from scoped_nodes(generator.target, inner, inner)
                for condition in generator.ifs:
                    yield from scoped_nodes(condition, inner, inner)
            for value in comprehension_values(node):
                yield from scoped_nodes(value, inner, inner)
            return
        if isinstance(node, definitions):
            body, local, declared = scope_info(node)
            for child in ast.iter_child_nodes(node):
                if child not in body:
                    yield from scoped_nodes(child, hidden, closure)
            if isinstance(node, ast.ClassDef):
                # A name the class body binds is looked up in the class namespace and
                # then the module, not in an enclosing function, so before its binding
                # it is the module's. Other names resolve through the enclosing scope.
                # Only definite, preceding bindings hide a name; methods do not close
                # over the class namespace. A nonlocal name stays the enclosing function's.
                nonlocal_names = {name for item in body for part in scope_nodes(item)
                                  if isinstance(part, ast.Nonlocal) for name in part.names}
                inner = closure - declared - (local - nonlocal_names)
                for item in body:
                    yield from scoped_nodes(item, inner, closure)
                    if (isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                            and not (isinstance(item, ast.AnnAssign) and item.value is None)):
                        targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                        bindings = set().union(*(bound_names(part) for target in targets for part in ast.walk(target)))
                        inner = inner | (bindings - declared)
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                        inner = inner | (bound_names(item) - declared)
                    elif isinstance(item, ast.Delete):
                        inner = inner - set().union(*(bound_names(part) for part in ast.walk(item)))
                return
            inner = (closure - declared) | local
            yield node, inner  # function bodies resolve their globals when called
            for item in body:
                yield from scoped_nodes(item, inner, inner)
            return
        yield node, hidden
        for child in ast.iter_child_nodes(node):
            yield from scoped_nodes(child, hidden, closure)

    def module_bindings(statement, hidden=frozenset()):
        """Names changed in the module namespace, including explicit global writes.

        A function or class has its own local bindings. Inspect its body for
        declared globals and mutations of imported objects, without treating a
        local variable called np or SCFT as a change to the module import.
        hidden holds the local names of a function whose body statement this is.
        """

        assigned = set()

        def visit(node, hidden):
            assigned.update(bound_names(node) - hidden)
            if isinstance(node, comprehensions):
                inner = hidden | comprehension_bindings(node)
                for index, generator in enumerate(node.generators):
                    visit(generator.iter, hidden if index == 0 else inner)
                    visit(generator.target, inner)
                    for condition in generator.ifs:
                        visit(condition, inner)
                for value in comprehension_values(node):
                    visit(value, inner)
                return
            if isinstance(node, definitions):
                body, local, declared = scope_info(node)
                inner_hidden = (hidden | local) - declared
                # Defaults, decorators and class bases are evaluated in the containing scope.
                for child in ast.iter_child_nodes(node):
                    if child not in body:
                        visit(child, hidden)
                for item in body:
                    visit(item, inner_hidden)
                return
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                owner = node.value
                while isinstance(owner, (ast.Attribute, ast.Subscript)):
                    owner = owner.value
                if isinstance(owner, ast.Name) and owner.id not in hidden and owner.id in imports:
                    if imports[owner.id].split(".")[0] in ("copy", "copyreg"):
                        raise ValueError("Changes to the copy module may cause parameter mutations when a "
                                         "simulation constructor copies params; use a saved SCFT result.")
                    assigned.add(owner.id)
            for child in ast.iter_child_nodes(node):
                visit(child, hidden)

        visit(statement, set(hidden))
        return assigned

    def unsettled_class_names(node):
        """Names a class body binds other than by one of its own statements.

        After a binding inside if, for, while, try, with or match, by := or del, the
        name may be the class's or still the module's: its module value cannot be
        trusted, and a module container under that name is still checked.
        """
        names = set()
        for item in node.body:
            if isinstance(item, (ast.Assign, ast.AugAssign)) or (isinstance(item, ast.AnnAssign) and item.value is not None):
                targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                definite = {id(part) for target in targets for part in ast.walk(target)}
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                definite = {id(item)}
            else:
                definite = set()
            names.update(name for part in scope_nodes(item) if id(part) not in definite for name in bound_names(part))
        return names

    # For each node in the scope of a class body, the names that body binds unsettled
    class_unsettled = {}
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        names = frozenset(unsettled_class_names(cls))
        for part in (part for item in cls.body for part in scope_nodes(item)) if names else ():
            class_unsettled[id(part)] = class_unsettled.get(id(part), frozenset()) | names
    # Names a module has before its first statement
    predefined = set(dir(builtins)) | {"__file__", "__cached__", "__builtins__", "__annotations__"}

    def resolved_name(node, hidden, unsettled):
        owner = node
        while isinstance(owner, ast.Attribute):
            owner = owner.value
        if isinstance(owner, ast.Name) and (owner.id in hidden or owner.id in unsettled):
            return ""
        return callable_name(node)

    def call_name(node, hidden, unsettled):
        return resolved_name(node.func, hidden, unsettled)

    def loads_params(node, hidden, unsettled):
        """SCFT.from_file and the like, reached through any name the converter cannot resolve:
        an import in a function or under a condition, an alias, a subclass, an instance."""
        if id(node) not in from_files:
            return False
        owner = resolved_name(node.value, hidden, unsettled)
        return not owner or owner == "polymerfts" or owner.startswith("polymerfts.")

    def hides_simulation(node, hidden, unsettled):
        """A simulation class or its module where the converter cannot follow the calls.

        A name that an import in a function or under a condition binds is not resolved,
        and the class or module given to another name, a loop or a function can be
        called there with any params. Annotations, isinstance and `is` only name it.
        """
        if id(node) not in may_hide:
            return False  # a name nothing imports, an owner of an attribute or an annotation
        if id(node) in type_tests and (type_tests[id(node)] is None or resolved_name(
                type_tests[id(node)], hidden, unsettled) in ("isinstance", "issubclass")):
            return False
        if id(node) not in called and resolved_name(node, hidden, unsettled) in traced:
            return True
        return any(path in traced for path in untraced_paths(node, hidden, unsettled))

    def untraced_paths(node, hidden, unsettled):
        """What a name the converter does not resolve may reach through those imports."""
        if id(node) not in reach_untraced:
            return []
        root, attributes = node, []
        while isinstance(root, ast.Attribute):
            attributes.insert(0, root.attr)
            root = root.value
        if resolved_name(root, hidden, unsettled):
            return []
        return [".".join([target, *attributes]) for target in untraced.get(root.id, ())]

    def reflective(name):
        return (name in reflection or name.removeprefix("builtins.") in reflection
                or name in ("__builtins__", "sys.modules") or name.startswith("copy._")
                or name.partition(".")[0] == "__main__")  # the script's own module

    def record_construction(node, hidden, untrusted, bound, from_file=False):
        """Record the params a simulation constructor receives here.

        Nothing is recorded while a global it reads is unbound: the call would stop
        with a NameError before it constructs anything. A constructor that loads its
        params from a file records `loaded` in place of a value.
        """
        argument = None if from_file else simulation_params(node, untrusted)
        given = node if from_file else params_argument(node)
        if argument is not None:
            constructed.setdefault(id(node), []).append(evaluate(argument))
        elif given is None or not (names_in(given) - hidden - predefined - bound):
            constructed.setdefault(id(node), []).append(loaded if from_file else None)

    deferred_bodies = []
    array_names = set()  # module names bound to a new base-class ndarray
    runtime_globals = set()  # module names that a function body binds whenever it runs
    ever_bound = set()  # module names bound by an earlier statement
    constructed = {}  # each simulation constructor: the params values it is seen to receive
    loaded = object()  # stands for params that a constructor loads from a file
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                imports[alias.asname or alias.name] = (statement.module or "") + "." + alias.name
        assigned = module_bindings(statement)
        for node in ast.walk(statement):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in bound_names(node) & assigned:
                    if name == "*":
                        raise ValueError("Star imports may rebind parameter values; use explicit imports "
                                         "or saved SCFT result/JSON/YAML parameters.")
                    if name == "params" and params is not None:
                        raise ValueError("Runtime params assignments are not supported; use a saved SCFT result.")
                    # Imported values are not evaluated. A previous literal with
                    # the same name must not stand in for the new binding.
                    symbols.pop(name, None)
        # We do not execute control flow. Any binding it might change must be
        # untrusted before looking at calls later in that same compound statement.
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom,
                                      ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for name in assigned:
                imports.pop(name, None)
            rebound.update(assigned)
        containers = {name for name, value in symbols.items() if isinstance(value, (dict, list, tuple))}
        known_containers = set().union(*(container_ids(symbols[name]) for name in containers))
        # Check escapes before unsupported assignments discard their old symbols.
        # Passing a parameter container to arbitrary Python can mutate any nested
        # value; observing its AST cannot establish the configuration afterwards.
        nodes = list(scoped_nodes(statement))
        current = {id(node) for node, _ in nodes}  # the rest are earlier function bodies
        # An assignment evaluates its value, and an augmented one also its target, with
        # the old bindings of the names it binds, unless an assignment expression (:=)
        # rebinds one first. A lambda or generator in the value runs later.
        settled, before_binding = set(), set()
        if (isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and statement.value is not None
                and not any(isinstance(part, ast.NamedExpr) for part in ast.walk(statement))):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            settled = set().union(*(bound_names(part) for target in targets for part in ast.walk(target)))
            pending = [statement.value]
            before_binding.add(id(statement))
            while pending:
                part = pending.pop()
                before_binding.add(id(part))
                if not isinstance(part, (ast.Lambda, ast.GeneratorExp)):
                    pending.extend(ast.iter_child_nodes(part))
        # A function can read a global container that is defined after it.
        # Revisit its body as later module bindings become known.
        for node, hidden in [*deferred_bodies, *nodes]:
            # Only direct module assignments to names are evaluated below. Other
            # bindings must not hide an alias that later mutates a known container.
            # Iterating a dictionary yields its scalar keys; numeric shape loops
            # and the generated script's monomer-name comprehension remain safe.
            # targets: what the value (or each iterated item) is assigned to; None binds it whole.
            expression, iterated, targets = None, False, [None]
            expressions = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                expression = node.value
                if node is statement and all(isinstance(target, ast.Name) for target in targets):
                    try:
                        evaluate(expression)
                    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                        pass
                    else:
                        expression = None  # this assignment retains its aliases in symbols below
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                expression, iterated, targets = node.iter, True, [node.target]
            elif isinstance(node, ast.NamedExpr):
                expression = node.value
            elif isinstance(node, ast.arguments):
                # Defaults bind before entering the function's scope, including
                # defaults of lambdas and keyword-only arguments.
                expressions = [value for value in [*node.defaults, *node.kw_defaults] if value is not None]
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                expression, iterated = node.value, isinstance(node, ast.YieldFrom)
            elif isinstance(node, ast.Lambda):
                expression = node.body
            elif isinstance(node, ast.Match) and any(
                    bound_names(part) for case in node.cases for part in ast.walk(case.pattern)):
                expression = node.subject
            if expression is not None:
                expressions.append(expression)
            here = class_unsettled.get(id(node), frozenset())
            untrusted = hidden | here | (assigned - settled if id(node) in before_binding else assigned)
            for expression in expressions:
                if not references_container(expression, containers - hidden, untrusted):
                    continue
                try:
                    value = evaluate(expression)
                except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                    aliases = True
                else:
                    items = list(value) if iterated and isinstance(value, (dict, list, tuple)) else [value]
                    values = [bound for item in items for target in targets for bound in bound_values(target, item)]
                    aliases = any(container_ids(item) & known_containers for item in values)
                if aliases:
                    raise ValueError("Runtime container alias bindings may cause parameter mutations; "
                                     "use direct assignments or saved SCFT result/JSON/YAML parameters.")
            if isinstance(node, ast.Attribute) or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                                                   and node.id not in hidden):
                # Also a reference, such as g = getattr, or a table reached through a module
                if reflective(callable_name(node)):
                    raise ValueError(reflection_error)
            if id(node) in main_imports:
                raise ValueError(reflection_error)  # also under another name in a function
            if id(node) in reach_untraced and any(reflective(path) for path in untraced_paths(node, hidden, here)):
                raise ValueError(reflection_error)  # such as import sys as s in a function
            if id(node) in from_files and loads_params(node, hidden, here):
                bound = ever_bound | (assigned if id(node) in current else assigned - settled)
                record_construction(node, hidden, untrusted, bound | here, from_file=True)
            if id(node) in may_hide and hides_simulation(node, hidden, here):
                raise ValueError(untraceable_error)
            parts, given = operands(node)
            if isinstance(node, ast.Match) and references_container(node.subject, containers - hidden, untrusted):
                # A value pattern compares the subject with its value (__eq__), a class
                # pattern runs isinstance, which a metaclass of the script can define.
                for part in (part for case in node.cases for part in ast.walk(case.pattern)):
                    if ((isinstance(part, ast.MatchValue) and not builtin_operand(part.value, untrusted))
                            or (isinstance(part, ast.MatchClass) and not (
                                isinstance(part.cls, ast.Name) and part.cls.id not in untrusted
                                and callable_name(part.cls) in ("int", "float", "complex", "str", "bool",
                                                                "list", "tuple", "dict")))):
                        raise ValueError("Runtime params containers given to operators of unsupported objects may "
                                         "cause mutations; use a saved SCFT result or JSON/YAML parameters.")
            if parts is not None and any(references_container(part, containers - hidden, untrusted) for part in given) \
                    and not inert_operation(parts, containers - hidden, untrusted):
                raise ValueError("Runtime params containers given to operators of unsupported objects may cause "
                                 "mutations; use a saved SCFT result or JSON/YAML parameters.")
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node, hidden, here)
            safe = name in readers
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"
                    and isinstance(node.func.value, ast.Call)):
                # Generated SCFT scripts reshape a loaded NumPy field to params["nx"].
                # Trust this shape-only operation on a known NumPy constructor,
                # not an arbitrary object's method with the same name.
                safe = call_name(node.func.value, hidden, here) in {"numpy.asarray", "numpy.array", "numpy.zeros",
                                                                    "numpy.ones", "numpy.empty", "numpy.full"}
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if name in simulations:
                arguments = simulation_extras(node, untrusted)  # a literal params is deep-copied
                # The export describes the calculation the script runs: what the constructor
                # receives, as seen here (a function body is seen again at later statements).
                # An earlier function body can run during this statement after any of its
                # bindings except the targets of a plain assignment, which are bound last.
                bound = ever_bound | (assigned if id(node) in current else assigned - settled)
                record_construction(node, hidden, untrusted, bound | here)
            if name in numeric_readers:
                safe = independent_result(node, untrusted)
            elif name in ("list", "tuple", "dict"):
                try:
                    evaluate(node)
                except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                    safe = False
            elif name in shape_readers:
                # ndarray.reshape and broadcast_to read the shape; another first
                # argument could pass the shape list to its own reshape method.
                safe = bool(node.args) and new_array(node.args[0], untrusted)
            elif name == "numpy.full":
                # Object fill values can retain a reference to a parameter dict.
                fill = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "fill_value"), None)
                safe = fill is None or not references_container(fill, containers - hidden, untrusted)
            if safe:
                continue
            for argument in arguments:
                if isinstance(argument, ast.Starred):
                    argument = argument.value
                try:
                    value = evaluate(argument)
                except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                    escaped = references_container(argument, containers - hidden, untrusted)
                else:
                    escaped = isinstance(value, (dict, list, tuple)) and references_container(argument, containers - hidden, untrusted)
                if escaped:
                    raise ValueError("Runtime params container arguments to unsupported calls may cause mutations; "
                                     "use a saved SCFT result or JSON/YAML parameters.")
        names = []
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "params" in names and params is not None:
                raise ValueError("Repeated params assignments are not supported; use a saved SCFT result.")
            try:
                value = evaluate(statement.value)
            except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError):
                if "params" in names:
                    raise ValueError("Could not read params statically. Use a saved SCFT result or JSON/YAML parameters.") from None
                # Never retain a stale constant after an unsupported assignment.
                for name in names:
                    symbols.pop(name, None)
            else:
                for name in names:
                    symbols[name] = value
                if "params" in names:
                    params = value
        # Parameter mutations are ambiguous without executing the script.
        containers.update(name for name, value in symbols.items() if isinstance(value, (dict, list, tuple)))
        for node, hidden in nodes:
            if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
                for name in (bound_names(node) & assigned) - hidden:
                    if name == "params":
                        raise ValueError("Runtime params assignments are not supported; use a saved SCFT result.")
                    # Pattern captures store names as strings, without Name(Store)
                    # nodes. Do not reuse a pre-match scalar or container binding.
                    symbols.pop(name, None)
            if (isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                    and node.id not in names and node.id in assigned and node.id not in hidden):
                if node.id == "params":
                    raise ValueError("Runtime params assignments are not supported; use a saved SCFT result.")
                if isinstance(node.ctx, ast.Store) and node.id in containers:
                    raise ValueError("Runtime container mutations are not supported; use a saved SCFT result.")
                # Assignments inside control flow and tuple unpacking are not evaluated.
                symbols.pop(node.id, None)
        # A function body may run at any later statement, so it is checked against the
        # containers known at each of them, also when it is defined before them.
        for node, hidden in [*deferred_bodies, *nodes]:
            if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                    and node.target.id not in hidden and node.target.id in containers):
                # Also in a class body before its own binding: += changes the module's list
                raise ValueError("Runtime container mutations are not supported; use a saved SCFT result.")
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                target = node
                while isinstance(target, ast.Subscript): target = target.value
                if (isinstance(target, ast.Name) and target.id not in hidden
                        and (target.id == "params" or target.id in containers)):
                    raise ValueError("Runtime params mutations are not supported; use a saved SCFT result.")
                # An expression, such as (params["polymers"] + [])[0], can hold the containers
                if not isinstance(target, ast.Name) and references_container(node.value, containers - hidden,
                                                                              hidden | assigned | class_unsettled.get(id(node), frozenset())):
                    raise ValueError("Runtime params mutations are not supported; use a saved SCFT result.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                while isinstance(target, ast.Subscript): target = target.value
                if (isinstance(target, ast.Name) and target.id not in hidden
                        and (target.id == "params" or target.id in containers)):
                    raise ValueError("Runtime params method calls are not supported; use a saved SCFT result.")
                if not isinstance(target, ast.Name) and references_container(node.func.value, containers - hidden,
                                                                              hidden | assigned | class_unsettled.get(id(node), frozenset())):
                    raise ValueError("Runtime params method calls are not supported; use a saved SCFT result.")
        for node, hidden in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for item in node.body:
                    deferred_bodies.extend(scoped_nodes(item, hidden, hidden))
            elif isinstance(node, ast.Lambda):
                deferred_bodies.append((node, hidden))
                deferred_bodies.extend(scoped_nodes(node.body, hidden, hidden))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # A global the body binds (global x) can change whenever the function runs
                body, local, _ = scope_info(node)
                runtime_globals.update(*(module_bindings(item, local) for item in body))
        # A plain assignment of a new array, also center = center * params["lx"], binds one.
        # ndarray.__iadd__ and the like keep the array with a NumPy or builtin operand; an
        # object of the script can override them (__array_ufunc__) and become the binding.
        # Any other binding ends it.
        made = (isinstance(statement, ast.Assign) and all(isinstance(target, ast.Name) for target in statement.targets)
                and array_result(statement.value, assigned - settled))
        kept = (isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name)
                and numerical(statement.value, assigned - settled))
        if not kept:
            array_names.difference_update(assigned)
        if made:
            array_names.update(target.id for target in statement.targets)
        # Neither a later literal, an import nor a new array settles such a global
        for name in runtime_globals:
            symbols.pop(name, None)
            imports.pop(name, None)
        array_names.difference_update(runtime_globals)
        rebound.update(runtime_globals)
        ever_bound.update(assigned)
        # An imported safe reader can be shadowed by a later assignment or definition.
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue  # The direct import above establishes its new binding.
        for name in assigned:
            imports.pop(name, None)
        rebound.update(assigned)
    # A function body can also run after the last statement (atexit, or a module importing this one).
    for node, hidden in deferred_bodies:
        here = class_unsettled.get(id(node), frozenset())
        if isinstance(node, ast.Call) and call_name(node, hidden, here) in simulations:
            record_construction(node, hidden, hidden | here, ever_bound | here)
        elif loads_params(node, hidden, here):
            record_construction(node, hidden, hidden | here, ever_bound | here, from_file=True)
        if hides_simulation(node, hidden, here):
            raise ValueError(untraceable_error)  # with the bindings of the last statement
    if not isinstance(params, dict):
        raise ValueError("The Python file needs a module-level params dictionary.")
    if not constructed:
        return params
    # A constructor in a function body may run at any later statement: every value it
    # can receive there must be readable and the same, and so must those of all constructors.
    received = [value for seen in constructed.values() for value in seen]
    if any(value is loaded for value in received):
        raise ValueError("SCFT.from_file, LFTS.from_file and CLFTS.from_file, and a from_file of an object the "
                         "converter cannot identify, load their params when the script runs; export the parameter "
                         "file, with the overrides applied, or a saved SCFT result.")
    if not all(isinstance(value, dict) for value in received):
        raise ValueError("The params argument of SCFT, LFTS, CLFTS or GibbsEnsemble must be a dictionary "
                         "written with literal values, such as params or dict(params, ds=.02); "
                         "use a saved SCFT result.")
    if any(value != received[0] for value in received):
        raise ValueError("The script runs calculations with different params, which one PSCF input cannot "
                         "describe; export one calculation from its saved SCFT result.")
    return received[0]


def read_input(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        try:
            return _MAT['decode_mat_fields'](loadmat(path, simplify_cells=True))
        except (MatReadError, IndexError, NotImplementedError) as exc:
            raise ValueError(f"Cannot read MATLAB file {path}: {exc}. Use a SciPy-compatible MAT file (not HDF5/v7.3).") from None
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ValueError("YAML input requires PyYAML.") from None
        with path.open(encoding="utf-8") as handle:
            try:
                return yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML input: {exc}") from None
    if suffix == ".py":
        return read_python_params(path)
    raise ValueError("Input must be a .mat result, .json/.yaml configuration/result, or a literal .py parameter file.")


def prepare_export(source, field_source=None, lattice=None):
    if not isinstance(source, dict):
        raise ValueError("Input must contain a parameter dictionary.")
    params = copy.deepcopy(source.get("initial_params", source.get("params", source)))
    if not isinstance(params, dict):
        raise ValueError("initial_params must be a parameter dictionary.")
    # simplify_cells turns singleton MATLAB structure arrays into dictionaries.
    for key in ("polymers", "distinct_polymers"):
        if key in params:
            polymers = params[key]
            if isinstance(polymers, dict): polymers = [polymers]
            if isinstance(polymers, np.ndarray): polymers = polymers.tolist()
            if not isinstance(polymers, list):
                raise ValueError("polymers must be a list of polymer definitions.")
            params[key] = polymers
            for polymer in polymers:
                if not isinstance(polymer, dict):
                    raise ValueError("Every polymer must be a dictionary.")
                if isinstance(polymer.get("blocks"), dict): polymer["blocks"] = [polymer["blocks"]]
    try:
        params = _VALIDATION["normalize_params"](params)
    except _VALIDATION["ValidationError"] as exc:
        raise ValueError(str(exc)) from None
    data = field_source if field_source is not None else source
    if not isinstance(data, dict):
        raise ValueError("Field input must be a dictionary.")
    if "nx" in data and "nx" in params and vector(data["nx"], "nx", integer=True) != vector(params["nx"], "nx", integer=True):
        raise ValueError("Field-file nx does not match the parameter grid.")
    # Results contain the final relaxed cell, while initial_params may describe its initial size.
    for key in ("nx", "lx", "angles"):
        if key in data: params[key] = data[key]
    params["nx"], params["lx"], params["angles"] = validate_cell(params["nx"], params["lx"], params.get("angles", [90.] * 3))
    try:
        _VALIDATION["validate_system_params"](params)
    except _VALIDATION["ValidationError"] as exc:
        raise ValueError(str(exc)) from None
    if params["chain_model"] != "continuous":
        raise ValueError("Export supports continuous Gaussian chains only; discrete chains are not silently approximated.")
    if params.get("zeta_n") is not None:
        raise ValueError("Finite-compressibility models are not supported by this incompressible PSCF export.")
    if params.get("mask") is not None or params.get("smearing") is not None:
        raise ValueError("Masks and interaction smearing cannot be transferred to this PSCF input.")
    if params.get("electrostatics") is not None:
        raise ValueError("Electrostatics cannot be transferred to this PSCF input.")
    bc = params.get("bc")
    if bc is not None and (not isinstance(bc, (list, tuple, np.ndarray))
            or len(bc) != 2 * len(params["nx"])
            or any(not isinstance(value, str) or value.lower() != "periodic" for value in bc)):
        raise ValueError("Export supports one periodic boundary condition for each cell face only.")
    if params.get("axis_ordering") is not None:
        raise ValueError("Use a saved SCFT result with normalized axes instead of axis_ordering parameters.")
    names = list(params["segment_lengths"])
    if any(any(c.isspace() for c in name) for name in names):
        raise ValueError("Monomer names must not contain whitespace.")
    fields = []
    for name in names:
        if "w_" + name not in data:
            raise ValueError(f"Missing w_{name}. Provide a saved result, or use --fields with the parameter file.")
        field = np.asarray(data["w_" + name])
        if np.iscomplexobj(field) or not np.isfinite(field).all() or field.size != math.prod(params["nx"]):
            raise ValueError(f"w_{name} must contain one finite real value per grid point.")
        fields.append(field.astype(float).reshape(params["nx"]))
    for polymer in params["polymers"]:
        if any(polymer.get(key) is not None for key in ("initial_conditions", "grafting_points")):
            raise ValueError("Grafted chains and custom initial propagator conditions are not supported by this PSCF export.")
        if "chemical_potential" in polymer:
            raise ValueError("Grand-canonical polymers (chemical_potential) are not supported by this canonical PSCF export.")
        blocks = polymer["blocks"]
        explicit = any("v" in b or "u" in b for b in blocks)
        edges, parent = set(), {}
        def root(v):
            parent.setdefault(v, v)
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for block in blocks:
            if "monomer_fractions" in block:
                raise ValueError("Random-copolymer blocks are not supported by this PSCF export.")
            if block.get("multiplicity", 1) != 1:
                raise ValueError("Block multiplicities are not supported by this PSCF export; "
                                 "write integer copies as separate blocks.")
            if block["type"] not in names:
                raise ValueError(f"Unknown block monomer: {block['type']}.")
            if explicit:
                if "v" not in block or "u" not in block:
                    raise ValueError("All branched blocks must specify both v and u.")
                v, u = block["v"], block["u"]
                pair = tuple(sorted((v, u)))
                if v == u or pair in edges or root(v) == root(u):
                    raise ValueError("Polymer blocks must form a connected tree without repeated edges or rings.")
                edges.add(pair)
                parent[root(v)] = root(u)
        if explicit and len({root(v) for v in parent}) != 1:
            raise ValueError("All blocks of a branched polymer must be connected.")
    lattice, cell = choose_lattice(params, lattice)
    return params, np.stack(fields), lattice, cell


def param_text(params, lattice):
    names = list(params["segment_lengths"])
    indices = {name: i for i, name in enumerate(names)}
    lines = ["System{", "  Mixture{", f"    nMonomer {len(names)}", "    monomers["]
    lines += [f"      {params['segment_lengths'][name]:.17g}" for name in names]
    lines += ["    ]", f"    nPolymer {len(params['polymers'])}"]
    for polymer in params["polymers"]:
        blocks = polymer["blocks"]
        branched = "v" in blocks[0]
        vertices = {v: i for i, v in enumerate(sorted({b[k] for b in blocks for k in ("v", "u")}))} if branched else {}
        lines += ["    Polymer{", f"      type {'branched' if branched else 'linear'}", f"      nBlock {len(blocks)}", "      blocks["]
        for block in blocks:
            text = f"        {indices[block['type']]} {block['length']:.17g}"
            if branched: text += f" {vertices[block['v']]} {vertices[block['u']]}"
            lines.append(text)
        lines += ["      ]", f"      phi {polymer['volume_fraction']:.17g}", "    }"]
    lines += [f"    ds {params['ds']:.17g}", "  }", "  Interaction{", "    chi("]
    interactions = {}
    for pair, chi in params["chi_n"].items():
        try:
            normalized = _VALIDATION["parse_chi_n"]({pair: chi}, params["segment_lengths"])
            a, b = next(iter(normalized)).split(",")
        except _VALIDATION["ValidationError"] as exc:
            # Older converter inputs also accepted concatenated names (AB).
            # A recognized separator syntax must retain its validation error,
            # including self-interactions and ambiguous names.
            if (not isinstance(pair, str) or "," in pair
                    or any(pair == a + separator + b for a in names for b in names
                           for separator in (" ", "_", "/"))):
                raise ValueError(str(exc)) from None
            candidates = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
                          if pair in (a+b, b+a)]
            if len(candidates) != 1:
                raise ValueError(str(exc)) from None
            a, b = candidates[0]
        key = tuple(sorted((indices[a], indices[b])))
        if key in interactions: raise ValueError(f"Repeated interaction: {pair}.")
        interactions[key] = chi
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lines.append(f"      {i} {j} {interactions.get((i, j), 0.):.17g}")
    lines += ["    )", "  }", "  Domain{", "    mesh " + " ".join(map(str, params["nx"])), f"    lattice {lattice}", "  }", "  AmIteratorGrid{"]
    optimizer = params.get("optimizer", {})
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer must be a dictionary.")
    epsilon, maximum, history = params.get("tolerance", 1e-8), params.get("max_iter", 5000), optimizer.get("max_hist", 20)
    for value, label, integer in ((epsilon, "tolerance", False), (maximum, "max_iter", True), (history, "max_hist", True)):
        checked = vector(value, label, integer=integer)
        if len(checked) != 1: raise ValueError(f"{label} must be scalar.")
    flexible = params.get("relax_cell", False)
    if not isinstance(flexible, (bool, np.bool_, int, np.integer)) or flexible not in (0, 1):
        raise ValueError("relax_cell must be a boolean.")
    stress = params.get("stress_scale", 1.)
    vector(stress, "stress_scale")
    lines += [f"    epsilon {epsilon:.17g}", f"    maxItr {int(maximum)}", f"    maxHist {int(history)}",
              f"    isFlexible {int(flexible)}", f"    scaleStress {stress:.17g}", "  }", "}"]
    return "\n".join(lines) + "\n"


def export_files(input_path, output_dir, fields_path=None, lattice=None):
    source = read_input(input_path)
    fields = read_input(fields_path) if fields_path else None
    params, fields, lattice, cell = prepare_export(source, fields, lattice)
    text = param_text(params, lattice)
    output_dir = Path(output_dir)
    # Require a new directory, so no existing inputs/results can be replaced.
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        (output_dir / "in").mkdir()
        (output_dir / "out").mkdir()
        write_rgrid(output_dir / "in/omega", params["nx"], lattice, cell, fields)
        (output_dir / "param").write_text(text, encoding="utf-8")
        (output_dir / "command").write_text("READ_W_RGRID in/omega\nITERATE\nWRITE_W_RGRID out/omega\nWRITE_C_RGRID out/rho\nWRITE_THERMO out/thermo\nFINISH\n", encoding="utf-8")
        metadata = dict(format="polymerfts-pscf-rgrid-1", monomer_types=list(params["segment_lengths"]),
                        nx=params["nx"], lattice=lattice, chain_model="continuous")
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # This directory was exclusively created by this invocation.
        shutil.rmtree(output_dir)
        raise
    return output_dir


def field_identity(data, monomers=None, metadata=None, chain_model=None):
    """Resolve a field's column names without writing anything."""
    n_monomer = len(data["fields"])
    if metadata:
        info = json.loads(Path(metadata).read_text(encoding="utf-8"))
        if (not isinstance(info, dict) or info.get("format") != "polymerfts-pscf-rgrid-1"
                or info.get("nx") != data["nx"] or info.get("lattice") != data["lattice"]):
            raise ValueError("Metadata format or grid does not match the PSCF fields.")
        if monomers is None: monomers = info.get("monomer_types", [])
        if chain_model is None: chain_model = info.get("chain_model")
    if monomers is None:
        monomers = [string.ascii_uppercase[i] if i < 26 else f"M{i+1}" for i in range(n_monomer)]
    if (not isinstance(monomers, (list, tuple)) or len(monomers) != n_monomer or
            any(not isinstance(name, str) or not name or any(c.isspace() for c in name) for name in monomers)
            or len(set(monomers)) != n_monomer):
        raise ValueError("Provide one unique, nonempty monomer name per field column.")
    if chain_model is not None and chain_model not in ("continuous", "discrete"):
        raise ValueError("chain_model must be continuous or discrete.")
    return list(monomers), chain_model


def import_files(omega, output, rho=None, monomers=None, metadata=None, chain_model=None):
    data = read_rgrid(omega)
    monomers, chain_model = field_identity(data, monomers, metadata, chain_model)
    densities = read_rgrid(rho) if rho else None
    if densities:
        if (densities["nx"] != data["nx"] or densities["lattice"] != data["lattice"] or
                densities["fields"].shape != data["fields"].shape or
                not np.allclose(densities["cell"], data["cell"], rtol=1e-10, atol=1e-12) or
                densities["group_name"] != data["group_name"]):
            raise ValueError("Omega and rho headers describe different grids, cells, groups or monomer counts.")
    system = ({"hexagonal": "Hexagonal2D", "oblique": "Oblique2D", "rhombic": "Oblique2D"}.get(data["lattice"], "Orthorhombic") if len(data["nx"]) < 3 else
              "Triclinic" if data["lattice"] == "rhombohedral" else data["lattice"].title())
    result = dict(dim=len(data["nx"]), nx=data["nx"], lx=data["lx"], angles=data["angles"],
                  crystal_system=system, pscf_lattice=data["lattice"], pscf_group_name=data["group_name"],
                  monomer_types=np.asarray(monomers, dtype=object))
    if chain_model is not None: result["chain_model"] = chain_model
    for i, name in enumerate(monomers):
        result["w_" + name] = data["fields"][i].ravel(order="C")
        if densities: result["phi_" + name] = densities["fields"][i].ravel(order="C")
    output = Path(output)
    if output.suffix.lower() != ".mat": raise ValueError("Import output must have a .mat extension.")
    with output.open("xb") as handle:
        try:
            encoded = _MAT['_encode_mat_fields'](result, 63, top_level=True)
            savemat(handle, encoded, long_field_names=True, do_compression=True)
        except Exception:
            output.unlink()
            raise
    return output


def parse_param(text):
    """Read PSCF's labeled blocks, arrays and sparse matrices, without execution."""
    lexer = shlex.shlex(text, posix=True, punctuation_chars="{}[]()")
    lexer.whitespace_split, lexer.commenters = True, "#"
    delimiters = "{}[]()"
    tokens = []
    for token in lexer:
        tokens.extend(list(token) if token and all(c in delimiters for c in token) else [token])
    position = 0

    def numeric(token):
        try:
            real(token)
            return True
        except ValueError:
            return False

    def block(closing=None, depth=0):
        nonlocal position
        if depth > 16:
            raise ValueError("PSCF parameter nesting is too deep.")
        entries = {}
        while position < len(tokens):
            label = tokens[position]
            position += 1
            if label == closing:
                return entries
            if label in delimiters or numeric(label) or position == len(tokens):
                raise ValueError(f"Malformed PSCF parameter near {label!r}.")
            first = tokens[position]
            position += 1
            if first == "{":
                kind, value = "block", block("}", depth + 1)
            elif first in ("[", "("):
                kind, value = first, []
                end = "]" if first == "[" else ")"
                while position < len(tokens) and tokens[position] not in delimiters:
                    value.append(tokens[position])
                    position += 1
                if position == len(tokens) or tokens[position] != end:
                    raise ValueError(f"Unclosed or nested PSCF array/matrix: {label}.")
                position += 1
            else:
                if first in delimiters:
                    raise ValueError(f"Missing PSCF parameter value: {label}.")
                kind, value = "value", [first]
                while position < len(tokens) and numeric(tokens[position]):
                    value.append(tokens[position])
                    position += 1
            entries.setdefault(label, []).append((kind, value))
        if closing:
            raise ValueError(f"Missing closing {closing} in PSCF parameters.")
        return entries

    return block()


def param_scope(entries, allowed, label, repeated=()):
    unknown = set(entries) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported PSCF {label} entries: {', '.join(sorted(unknown))}.")
    for name, values in entries.items():
        if len(values) != 1 and name not in repeated:
            raise ValueError(f"Repeated PSCF parameter: {label}.{name}.")
    return entries


def param_value(entries, label, kind="value", default=None):
    if label not in entries:
        if default is not None:
            return default
        raise ValueError(f"Missing PSCF parameter: {label}.")
    actual, value = entries[label][0]
    if actual != kind:
        raise ValueError(f"Wrong PSCF parameter format: {label}.")
    return value


def param_scalar(entries, label, default=None, *, integer=False, minimum=None, text=False):
    values = param_value(entries, label, default=[str(default)] if default is not None else None)
    if len(values) != 1:
        raise ValueError(f"PSCF {label} must be scalar.")
    if text:
        return values[0]
    value = real(values[0])
    if (not math.isfinite(value) or (integer and value != int(value))
            or (minimum is not None and value < minimum)):
        raise ValueError(f"Invalid PSCF {label}: {values[0]}.")
    return int(value) if integer else value


def pscf_index(token, size, label):
    value = real(token)
    if not math.isfinite(value) or value != int(value) or not 0 <= value < size:
        raise ValueError(f"Invalid PSCF {label}: {token}.")
    return int(value)


def script_params(text, data, names):
    root = param_scope(parse_param(text), ("System",), "root")
    system = param_scope(param_value(root, "System", "block"),
                         ("polymerModel", "Mixture", "Interaction", "Domain", "AmIteratorGrid"), "System")
    if param_scalar(system, "polymerModel", "thread", text=True) != "thread":
        raise ValueError("Script import supports continuous (thread) chains only; bead chains are unsupported.")
    mixture = param_scope(param_value(system, "Mixture", "block"),
                          ("nMonomer", "monomers", "nPolymer", "Polymer", "nSolvent", "ds", "vMonomer"),
                          "Mixture", repeated=("Polymer",))
    count = param_scalar(mixture, "nMonomer", integer=True, minimum=1)
    if count != len(names):
        raise ValueError("PSCF nMonomer does not match the initial fields.")
    if count < 2:
        raise ValueError("Runnable incompressible SCFT scripts require at least two monomer types; "
                         "use field-only import for a single-monomer PSCF result.")
    if any(any(c not in string.ascii_letters + "_" for c in name) for name in names):
        raise ValueError("Runnable script monomer names require ASCII letters and underscores only; provide compatible --monomers names.")
    segments = vector([real(x) for x in param_value(mixture, "monomers", "[")], "monomers")
    if len(segments) != count:
        raise ValueError("PSCF monomers array must contain nMonomer segment lengths.")
    if param_scalar(mixture, "nSolvent", 0, integer=True, minimum=0) != 0:
        raise ValueError("PSCF solvents are not supported by script import.")
    if param_scalar(mixture, "vMonomer", 1.) != 1.:
        raise ValueError("Script import requires PSCF vMonomer = 1 in reference-chain units.")
    n_polymer = param_scalar(mixture, "nPolymer", integer=True, minimum=1)
    definitions = mixture.get("Polymer", [])
    if len(definitions) != n_polymer:
        raise ValueError("PSCF nPolymer does not match the number of Polymer blocks.")
    polymers = []
    for kind, definition in definitions:
        if kind != "block":
            raise ValueError("PSCF Polymer must be a block.")
        definition = param_scope(definition, ("type", "nBlock", "blocks", "phi"), "Polymer")
        topology = param_scalar(definition, "type", "linear", text=True)
        if topology not in ("linear", "branched"):
            raise ValueError(f"Unsupported PSCF polymer type: {topology}.")
        n_block = param_scalar(definition, "nBlock", integer=True, minimum=1)
        raw = param_value(definition, "blocks", "[")
        width = 2 if topology == "linear" else 4
        if len(raw) != n_block * width:
            raise ValueError("PSCF blocks array does not match nBlock and polymer type.")
        blocks = []
        for start in range(0, len(raw), width):
            row = raw[start:start + width]
            block = dict(type=names[pscf_index(row[0], count, "block monomer")], length=real(row[1]))
            if topology == "branched":
                # A PSCF tree with nBlock edges has nBlock+1 numbered vertices.
                block.update(v=pscf_index(row[2], n_block + 1, "vertex"),
                             u=pscf_index(row[3], n_block + 1, "vertex"))
            blocks.append(block)
        polymers.append(dict(volume_fraction=param_scalar(definition, "phi"), blocks=blocks))
    interaction = param_scope(param_value(system, "Interaction", "block"), ("chi",), "Interaction")
    chi = param_value(interaction, "chi", "(")
    if len(chi) % 3:
        raise ValueError("PSCF chi must contain index-index-value triples.")
    pairs = {}
    for start in range(0, len(chi), 3):
        i, j = (pscf_index(chi[start + k], count, "chi index") for k in (0, 1))
        value = real(chi[start + 2])
        key = tuple(sorted((i, j)))
        if not math.isfinite(value) or key in pairs or (i == j and value != 0):
            raise ValueError("PSCF chi contains repeated pairs, nonfinite values or a nonzero diagonal.")
        pairs[key] = value
    domain = param_scope(param_value(system, "Domain", "block"), ("mesh", "lattice", "groupName"), "Domain")
    nx = vector([real(x) for x in param_value(domain, "mesh")], "mesh", integer=True)
    lattice = param_scalar(domain, "lattice", text=True).lower()
    if nx != data["nx"] or lattice != data["lattice"]:
        raise ValueError("PSCF Domain mesh/lattice does not match the initial fields.")
    group = param_scalar(domain, "groupName", "", text=True)
    iterator = param_scope(param_value(system, "AmIteratorGrid", "block"),
                           ("epsilon", "maxItr", "maxHist", "isFlexible", "flexibleParams", "scaleStress"),
                           "AmIteratorGrid")
    flexible = param_scalar(iterator, "isFlexible", 1, integer=True, minimum=0)
    if flexible not in (0, 1):
        raise ValueError("PSCF isFlexible must be 0 or 1.")
    if "flexibleParams" in iterator:
        flags = [pscf_index(x, 2, "flexibleParams flag") for x in param_value(iterator, "flexibleParams", "[")]
        if not flexible or len(flags) != len(data["cell"]):
            raise ValueError("PSCF flexibleParams requires isFlexible=1 and one flag per cell parameter.")
        if any(flags) and not all(flags):
            raise ValueError("Partially flexible PSCF cells cannot be transferred to polymerfts.")
        flexible = int(all(flags))
    dim = len(nx)
    if flexible and ((dim == 2 and lattice in ("square", "rhombic")) or lattice == "rhombohedral"):
        raise ValueError(f"Flexible {lattice} cells cannot preserve their constraints in polymerfts; use a fixed cell.")
    crystal = ({"hexagonal": "Hexagonal2D", "oblique": "Oblique2D", "rhombic": "Oblique2D"}.get(lattice, "Orthorhombic")
               if dim < 3 else "Triclinic" if lattice == "rhombohedral" else lattice.title())
    params = dict(nx=nx, lx=data["lx"], angles=data["angles"], chain_model="continuous",
                  numerical_method="rqm4", ds=param_scalar(mixture, "ds"),
                  segment_lengths=dict(zip(names, segments)),
                  chi_n={f"{names[i]},{names[j]}": pairs.get((i, j), 0.) for i in range(count) for j in range(i + 1, count)},
                  polymers=polymers, crystal_system=crystal, relax_cell=bool(flexible),
                  stress_scale=param_scalar(iterator, "scaleStress", 10.),
                  tolerance=param_scalar(iterator, "epsilon"), max_iter=param_scalar(iterator, "maxItr", 200, integer=True, minimum=1),
                  optimizer=dict(name="am", max_hist=param_scalar(iterator, "maxHist", 50, integer=True, minimum=0),
                                 start_error=.01, mix_min=.1, mix_init=.1))
    if params["tolerance"] <= 0 or params["stress_scale"] <= 0:
        raise ValueError("PSCF epsilon and scaleStress must be positive.")
    source = {"params": params, **{"w_" + name: data["fields"][i] for i, name in enumerate(names)}}
    # Reuse the backend-free physical model and connected-tree checks.
    params, _, _, _ = prepare_export(source, lattice=lattice)
    return params, group


def script_commands(text):
    """Accept one read/iterate/save calculation; validate before creating files."""
    tokens = shlex.split(text, comments=True)
    commands, outputs, omega = [], [], None
    stage, position = "read", 0
    writes = {"WRITE_W_RGRID", "WRITE_C_RGRID", "WRITE_THERMO", "WRITE_PARAM"}
    while position < len(tokens):
        name = tokens[position]
        position += 1
        argument = None
        if name == "READ_W_RGRID" or name in writes:
            if position == len(tokens):
                raise ValueError(f"Missing argument for PSCF command {name}.")
            argument = tokens[position]
            position += 1
        if name == "READ_W_RGRID" and stage == "read":
            omega, stage = argument, "iterate"
        elif name == "ITERATE" and stage == "iterate":
            stage = "write"
        elif name in writes and stage == "write":
            source = Path(argument)
            # anchor: a root or a drive; on Windows "/tmp/x" is rooted on the current drive
            if (not argument or source.anchor or ".." in source.parts or "\\" in argument
                    or argument.endswith("/") or source == Path(".")):
                raise ValueError(f"PSCF output must be a relative file path without '..': {argument!r}.")
            outputs.append({"command": name, "source": argument, "target": source.as_posix() + ".mat"})
        elif name == "FINISH" and stage == "write" and position == len(tokens):
            stage = "done"
        else:
            raise ValueError(f"Unsupported PSCF command or sequence: {name}. Expected READ_W_RGRID, ITERATE, optional writes, FINISH.")
        commands.append({"command": name, **({"argument": argument} if argument is not None else {})})
    if stage != "done":
        raise ValueError("PSCF command file must end with FINISH after one READ_W_RGRID and ITERATE.")
    paths = [Path(item["target"]) for item in outputs]
    if any(a in b.parents for a in paths for b in paths):
        raise ValueError("PSCF output paths conflict as files and directories.")
    return omega, commands, outputs


def simulation_text(params, outputs):
    targets = list(dict.fromkeys(item["target"] for item in outputs)) or ["fields.mat"]
    return '''#!/usr/bin/env python3
"""SCFT calculation converted from PSCF; see conversion.json for the mapping."""
import os
from pathlib import Path

# Set thread defaults before importing the numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("OMP_STACKSIZE", "1G")

import numpy as np
from scipy.io import loadmat
from polymerfts import scft

params = ''' + pformat(params, sort_dicts=False, width=100) + '''


def main():
    directory = Path(__file__).resolve().parent
    initial = loadmat(directory / "initial.mat", simplify_cells=True)
    initial_fields = {name: np.asarray(initial["w_" + name]).reshape(params["nx"])
                      for name in params["segment_lengths"]}
    # Keep the results beside this script.
    results = directory / "results"
    results.mkdir(exist_ok=False)
    calculation = scft.SCFT(params=params)
    result = calculation.run(initial_fields=initial_fields, return_result=True)
    if not result.converged:
        raise RuntimeError("SCFT did not converge; inspect the log and simulation parameters.")
    for relative in ''' + repr(targets) + ''':
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        calculation.save_results(str(path))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
'''


def import_script(param, command, output_dir, *, input_dir=None, monomers=None, metadata=None, platform="auto"):
    param, command = Path(param), Path(command)
    omega, commands, outputs = script_commands(command.read_text(encoding="utf-8"))
    base = Path(input_dir) if input_dir is not None else param.parent
    omega = base / omega
    data = read_rgrid(omega)
    names, model = field_identity(data, monomers, metadata)
    if model not in (None, "continuous"):
        raise ValueError("Script import requires continuous chains in the field metadata.")
    params, group = script_params(param.read_text(encoding="utf-8"), data, names)
    # polymerfts rejects cells flatter than V = 0.1 abc (field import and export keep them);
    # the small margin keeps rounding from letting through a cell that it rejects
    ca, cb, cg = np.cos(np.deg2rad(params["angles"]))
    if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg < 0.1 * 0.1 * (1 + 1e-9):
        raise ValueError("The PSCF cell has a volume below 0.1 abc, which polymerfts rejects; "
                         "convert the reduced (Niggli) cell of the lattice, spanned by its shortest vectors.")
    if platform not in ("auto", "cuda", "cpu-fftw", "cpu-mkl"):
        raise ValueError(f"Unknown platform: {platform}.")
    if platform != "auto":
        params["platform"] = platform
    manifest = dict(format="polymerfts-pscf-script-1", source_param=str(param.resolve()),
                    source_command=str(command.resolve()), source_fields=str(omega.resolve()),
                    pscf_group_name=group, field_group_name=data["group_name"], commands=commands,
                    outputs=outputs, default_output=None if outputs else "fields.mat",
                    notes=["Full-grid SCFT without imposed space-group symmetry.",
                           "All mapped outputs are full polymerfts .mat snapshots, not PSCF files.",
                           "epsilon, history, iteration limit and stress scale are starting settings; solver norms, mixing and discretization differ.",
                           "Only converged calculations save results; an existing results directory is never reused."])
    output_dir = Path(output_dir)
    script = simulation_text(params, outputs)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        import_files(omega, output_dir / "initial.mat", monomers=names, chain_model="continuous")
        (output_dir / "simulation.py").write_text(script, encoding="utf-8")
        (output_dir / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="operation", required=True)
    export = commands.add_parser("export", help="Write a PSCF periodic SCFT input directory")
    export.add_argument("input", type=Path, help="Saved SCFT result, or parameters with --fields")
    export.add_argument("-o", "--output", type=Path, required=True, help="New output directory")
    export.add_argument("--fields", type=Path, help="Separate .mat/.json/.yaml file containing w_<monomer>")
    export.add_argument("--lattice", help="Explicit PSCF lattice (validated against the cell geometry)")
    load = commands.add_parser("import", help="Read PSCF real-space grid fields into .mat")
    load.add_argument("omega", type=Path)
    load.add_argument("-o", "--output", type=Path, required=True)
    load.add_argument("--rho", type=Path, help="Optional density r-grid file")
    load.add_argument("--monomers", nargs="+", help="Monomer names in PSCF column order (default: A, B, ...)")
    load.add_argument("--metadata", type=Path, help="Use the column mapping from an exported metadata.json")
    load.add_argument("--chain-model", choices=("continuous", "discrete"), help="Optional source model; r-grid headers do not encode it")
    script = commands.add_parser("import-script", help="Convert PSCF param/command/r-grid inputs to a runnable SCFT script")
    script.add_argument("param", type=Path)
    script.add_argument("--command", type=Path, required=True, help="PSCF command file for one SCFT calculation")
    script.add_argument("-o", "--output", type=Path, required=True, help="New directory for simulation.py and bundled initial fields")
    script.add_argument("--input-dir", type=Path, help="PSCF working directory for READ_W_RGRID (default: param's directory)")
    script.add_argument("--monomers", nargs="+", help="ASCII letter/underscore monomer names in PSCF column order (default: A, B, ...)")
    script.add_argument("--metadata", type=Path, help="Column mapping from an exported metadata.json")
    script.add_argument("--platform", choices=("auto", "cpu-mkl", "cpu-fftw", "cuda"), default="auto", help="Backend used when running the generated script")
    args = parser.parse_args(argv)
    try:
        if args.operation == "export":
            path = export_files(args.input, args.output, args.fields, args.lattice)
            print(f"Written: {path}/param, command, in/omega, metadata.json")
            print("Uses AmIteratorGrid without imposed space-group symmetry; simulations are run separately.")
        elif args.operation == "import":
            path = import_files(args.omega, args.output, args.rho, args.monomers, args.metadata, args.chain_model)
            print(f"Written: {path} (initial fields; not a Langevin restart checkpoint)")
        else:
            path = import_script(args.param, args.command, args.output, input_dir=args.input_dir,
                                 monomers=args.monomers, metadata=args.metadata, platform=args.platform)
            print(f"Written: {path}/simulation.py, initial.mat, conversion.json")
            print("Run simulation.py separately with polymerfts installed; see conversion.json for output mappings.")
    except (ValueError, TypeError, KeyError, OSError, SyntaxError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    main()
