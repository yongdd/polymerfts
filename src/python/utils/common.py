"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import itertools
import inspect
import json
import hashlib
import os
import platform
import secrets
import subprocess
from functools import lru_cache
from types import MappingProxyType
from pathlib import Path
import numpy as np

from .mat_fields import _encode_mat_fields


def numpy_scalar(value):
    """Convert a NumPy scalar to a Python scalar, including extended precision.

    NumPy's longdouble/clongdouble.item() still returns a NumPy scalar. Portable
    output uses Python's double precision; reject finite overflow/underflow
    instead of recording a different non-finite or zero parameter.
    """
    result = value.item()
    if isinstance(result, np.floating):
        result = float(result)
        if np.isfinite(value) and (not np.isfinite(result) or (result == 0 and value != 0)):
            raise ValueError("NumPy floating-point scalar is outside the Python float range")
    elif isinstance(result, np.complexfloating):
        result = complex(numpy_scalar(result.real), numpy_scalar(result.imag))
    return result


def json_ready(value, strict=True):
    """Convert parameter dictionaries to portable JSON values.

    Strict JSON has no NaN or infinity: a non-finite float becomes {"nonfinite_float": "-inf"}.
    With strict=False it stays a float, for YAML, which has .nan and .inf.
    """
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist(), strict)
    if isinstance(value, np.generic):
        return json_ready(numpy_scalar(value), strict)
    if strict and isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v, strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v, strict) for v in value]
    if isinstance(value, complex):
        return {"real": json_ready(value.real, strict), "imag": json_ready(value.imag, strict)}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def mat_ready(value):
    """Convert a parameter dictionary for scipy.io.savemat, which has no None: an option
    given as None (its default) is left out, and paths become strings."""
    if isinstance(value, np.generic):
        return mat_ready(numpy_scalar(value))
    if isinstance(value, dict):
        return {str(k): mat_ready(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [np.empty(0) if v is None else mat_ready(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value



def settings_json(settings):
    return json.dumps(json_ready(settings), sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_hash(value):
    return hashlib.sha256(settings_json(value).encode()).hexdigest()


def fields_hash(fields):
    digest = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        digest.update(settings_json([name, array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _package_hashes():
    from .. import _core
    source = hashlib.sha256()
    package = Path(_core.__file__).parent  # Not __file__: a copy of this module may lie anywhere
    for path in sorted(package.rglob("*.py")):
        source.update(path.relative_to(package).as_posix().encode())
        source.update(path.read_bytes())
    native = hashlib.sha256()
    with open(_core.__file__, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            native.update(chunk)
    return {"python_source_sha256": source.hexdigest(), "native_binary_sha256": native.hexdigest()}


def _simulation_source(sim):
    """The file that defines the class of sim and its hash when it lies outside the package: a modified
    copy of scft.py (QuickStart) or a subclass in a script, which python_source_sha256 does not cover."""
    from .. import _core
    try:
        path = Path(inspect.getsourcefile(type(sim))).resolve()
        if path.is_relative_to(Path(_core.__file__).parent.resolve()):
            return {}
        return {"simulation_source": str(path), "simulation_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except (TypeError, OSError):   # a class without a source file, such as one defined interactively
        return {}


def _new_file_next_to(path, mode):
    """Open a new hidden file in the directory of path for writing; return it and its name.

    Unlike tempfile.mkstemp (mode 0600), the umask sets its permissions, as for a
    file written directly to path, so results stay readable as the user's other files.
    """
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    return os.fdopen(fd, mode), temporary


def _atomic_write(path, mode, write):
    """Write a file with write(stream) to a new file next to path, on the disk, that then replaces
    path: a failed write leaves an existing file as it was."""
    path = Path(path)
    stream, temporary = _new_file_next_to(path, mode)
    try:
        with stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_savemat(path, data, **kwargs):
    """Atomically write a MAT file, reversibly escaping unsupported struct keys."""
    from scipy.io import savemat
    encoded = _encode_mat_fields(data, 63 if kwargs.get("long_field_names", False) else 31, top_level=True)
    _atomic_write(path, "wb", lambda stream: savemat(stream, encoded, **kwargs))


def atomic_write_json(path, value):
    """Atomically write value as strict, indented JSON (json_ready) with a final newline."""
    def write(stream):
        json.dump(json_ready(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    _atomic_write(path, "w", write)


def atomic_write_text(path, text):
    """Atomically write a text file."""
    _atomic_write(path, "w", lambda stream: stream.write(text))


def simulation_metadata(sim, random_seed=None):
    """Build provenance comes from the loaded extension, never the current cwd."""
    import scipy
    from .. import _core
    solver = sim.prop_solver
    metadata = {
        "schema_version": 1, "build": dict(getattr(_core, "build_info", {})),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "system": platform.platform(), "machine": platform.machine(),
        "backend": solver.platform, "numerical_method": solver.numerical_method,
        "chain_model": solver.chain_model, "precision": "complex128" if solver.field_type == "complex" else "float64",
        "reduce_memory": solver.reduce_memory,
        "threads": {key: os.environ.get(key, "") for key in
                    ("OMP_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS")},
        "random_seed": random_seed, "input_hash": input_hash(sim.params),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **_package_hashes(), **_simulation_source(sim),
    }
    try:
        metadata["cpu"] = next(line.split(":", 1)[1].strip() for line in
                               Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        metadata["cpu"] = platform.processor()
    if solver.platform == "cuda":
        command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            command.append("--id=" + visible)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
            metadata["gpu"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            metadata["gpu"] = "unavailable"
    if hasattr(_core, "runtime_info"):
        metadata["libraries"] = _core.runtime_info(solver.platform == "cuda")
    return json_ready(metadata)


def notify_callbacks(sim, step, w_aux, phi, hamiltonian, drift_max, error_level, step_seconds):
    """Observers receive detached, read-only snapshots of accepted steps only."""
    callbacks = [cb for cb in sim.callbacks.values() if step % getattr(cb, "interval", 1) == 0]
    if not callbacks:
        return

    def frozen(array):
        # A bytes-backed array cannot be made writeable, even with setflags.
        array = np.asarray(array)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

    sample = MappingProxyType({"step": step, "dt": sim.langevin["dt"],
                               "time": step * sim.langevin["dt"],
                               "w_aux": frozen(w_aux), "w": frozen(sim.mpt.to_monomer_fields(w_aux)),
                               "phi": MappingProxyType({m: frozen(v) for m, v in phi.items()}),
                               "hamiltonian": hamiltonian, "drift_max": drift_max,
                               "error_level": frozen(error_level), "step_seconds": step_seconds,
                               "status": MappingProxyType(dict(sim.run_status))})
    for callback in callbacks:
        callback(sample)



def display_system(sim):
    """Prints the box, the chains and the interactions of a simulation."""
    solver = sim.prop_solver
    print("---------- Simulation Parameters ----------")
    print("Platform:", solver.platform)
    print("Box Dimension: %d" % (solver.dim))
    print("Nx:", solver.nx)
    print("Lx [a, b, c]:", solver.get_lx())
    print("Angles [alpha, beta, gamma]:", list(solver.get_angles_degrees()))
    print("dx:", solver.get_dx())
    print("Volume: %f" % (solver.get_volume()))

    print("Chain model: %s" % (sim.chain_model))
    print("Segment lengths:\n\t", list(sim.segment_lengths.items()))
    print("Conformational asymmetry (epsilon):")
    for monomer_pair in itertools.combinations(sim.monomer_types, 2):
        print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1],
                               sim.segment_lengths[monomer_pair[0]] / sim.segment_lengths[monomer_pair[1]]))

    print("χN:")
    for key in sim.chi_n:
        print("\t%s: %f" % (key, sim.chi_n[key]))
    if sim.zeta_n is not None:
        print("zeta_n: %f" % (sim.zeta_n))
    solver.display_architectures()


def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
    """Calculate Langevin noise strength for L-FTS simulations.

    Computes the standard deviation of Gaussian noise to be added at each
    Langevin step, ensuring correct statistical weighting of field configurations
    in the field-theoretic ensemble.

    Parameters
    ----------
    langevin_nbar : float
        Average polymerization index N̄ for the Langevin scheme. Related to
        the prefactor of the Hamiltonian. Typical values: 1-100.
    langevin_dt : float
        Langevin time step size Δt. Controls the magnitude of field updates.
        Typical values: 0.001-0.1.
    n_grids : int
        Total number of grid points M (product of nx dimensions).
    volume : float
        System volume V in units of (a_Ref * N_Ref^0.5)³.

    Returns
    -------
    sigma : float
        Standard deviation of Gaussian noise for each field component.
    """
    return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))
