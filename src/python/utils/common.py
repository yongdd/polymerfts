"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import itertools
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

from .validation import ValidationError, parse_chi_n  # noqa: F401 (parse_chi_n: former utils API)


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


def json_ready(value):
    """Convert parameter dictionaries to portable JSON values."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(numpy_scalar(value))
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, complex):
        return {"real": json_ready(value.real), "imag": json_ready(value.imag)}
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


def _new_file_next_to(path, mode):
    """Open a new hidden file in the directory of path for writing; return it and its name.

    Unlike tempfile.mkstemp (mode 0600), the umask sets its permissions, as for a
    file written directly to path, so results stay readable as the user's other files.
    """
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    return os.fdopen(fd, mode), temporary


def atomic_savemat(path, data, **kwargs):
    """Replace a MAT file only after a complete write in the same directory."""
    from scipy.io import savemat
    path = Path(path)
    stream, temporary = _new_file_next_to(path, "wb")
    try:
        with stream:
            savemat(stream, data, **kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_run_status(sim):
    """Persist completion/failure separately from periodic field checkpoints."""
    path = Path(sim.recording["dir"])
    path.mkdir(parents=True, exist_ok=True)
    stream, temporary = _new_file_next_to(path / "run_status.json", "w")
    try:
        with stream:
            json.dump(json_ready({**sim.run_status, "metadata": sim.metadata}), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path / "run_status.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def report_partition_consistency(prop_solver):
    """Print the forward/backward partition error of each polymer; True if all are within tolerance."""
    from .. import _core
    errors = prop_solver.partition_errors()
    print("Relative contour partition error of each polymer: " +
          ", ".join(f"{p}: {error:.3e}" for p, error in enumerate(errors)))
    return all(np.isfinite(error) and error <= _core.PARTITION_TOLERANCE for error in errors)


def check_partition_functions(prop_solver, state):
    """Reject an accepted state whose partition functions are not positive and finite.

    With real fields Q is a sum of positive Boltzmann weights. Intermediate iterates can
    still give Q <= 0 (the Richardson extrapolation of rqm4 or Crank-Nicolson steps on a
    coarse grid in strong fields), so only accepted states are checked: the final SCFT
    fields and an L-FTS saddle point whose Hamiltonian, which contains log Q, is not
    finite. Complex fields (CL-FTS) have no sign to check.
    """
    if prop_solver.field_type != "real":
        return
    for p in range(prop_solver.get_n_polymer_types()):
        Q = prop_solver.get_partition_function(p)
        if not (np.isfinite(Q) and Q > 0):
            raise ValidationError(
                f"The partition function of polymer {p} is {Q:.6e} at {state}, but with real fields it must "
                "be positive and finite. Either the discretization is too coarse for the fields (refine the "
                "grid or the contour step), or the propagators fell below the round-off level of the spectral "
                "transforms, which is relative to their largest value on the grid (for example strong fields "
                "next to absorbing walls or a grafted source).")


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
        **_package_hashes(),
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


class DriftTailSummary:
    """Distribution of the largest drift of each step, summarized at the end of a CL-FTS run.

    Complex Langevin averages need the distribution of the drift magnitude to fall off
    exponentially or faster (Nagata, Nishimura & Shimasaki, Phys. Rev. D 94, 114515 (2016)).
    The counts stay in memory, twenty bins per decade (the printed median and maximum are
    bin edges); the step values themselves are recorded as drift_max_step_history and
    psi_drift_max_step_history. The summary gives
    numbers, not a verdict: a finite run cannot show an exponential tail, and the weight of
    a power-law tail sets the size of the bias.
    """
    EDGES = 10.0 ** (np.arange(-120, 241) / 20)

    def __init__(self, labels):
        self.labels = list(labels)
        self.counts = np.zeros((len(self.labels), self.EDGES.size + 1), dtype=np.int64)

    def add(self, values):
        self.counts[np.arange(len(self.labels)), np.searchsorted(self.EDGES, values)] += 1

    def report(self):
        steps = int(self.counts[0].sum()) if self.labels else 0
        if steps == 0:
            return
        print(f"Largest drift per step ({steps} steps of this run); S(u) = P(drift > u) at 10 and 100 "
              "times the median, and the log-log slope of S between them (scripts/analysis/drift_tail.py):")
        for label, counts in zip(self.labels, self.counts):
            beyond = steps - np.cumsum(counts)          # steps above EDGES[k]
            k_median = int(np.searchsorted(-beyond, -steps / 2))
            if k_median >= self.EDGES.size:
                continue
            median = self.EDGES[k_median]
            if median <= self.EDGES[0]:
                print(f"  {label}: no drift")
                continue
            k10 = min(np.searchsorted(self.EDGES, 10 * median), self.EDGES.size - 1)
            k100 = min(np.searchsorted(self.EDGES, 100 * median), self.EDGES.size - 1)
            s10, s100 = beyond[k10] / steps, beyond[k100] / steps
            slope = f"{np.log(s100 / s10) / np.log(self.EDGES[k100] / self.EDGES[k10]):.2f}" \
                if beyond[k100] >= 20 else "-"
            largest = self.EDGES[min(np.flatnonzero(counts).max(), self.EDGES.size - 1)]
            print(f"  {label}: median ~{median:.3g}, maximum < {largest:.3g}, "
                  f"S(10x) {s10:.3g}, S(100x) {s100:.3g}, slope {slope}")


class PartitionRatioSummary:
    """Smallest |Q_s|/<|e^(-u_s)|> of each canonical single-site species (ThermodynamicOperators.partition_ratios) over
    a run, and the fractions of steps below 0.1 and 0.01. The step values are recorded as
    partition_ratio_step_history. Like the drift summary it gives numbers, not a verdict: a ratio
    near 0 means Q_s came close to its zero, the pole of -n_s ln Q_s."""
    THRESHOLDS = (0.1, 0.01)

    def __init__(self, labels):
        self.labels = list(labels)
        self.steps = 0
        self.smallest = np.full(len(self.labels), np.inf)
        self.below = np.zeros((len(self.labels), len(self.THRESHOLDS)), dtype=np.int64)

    def add(self, ratios):
        self.steps += 1
        self.smallest = np.minimum(self.smallest, ratios)
        self.below += np.asarray(ratios)[:, None] < np.asarray(self.THRESHOLDS)[None, :]

    def report(self):
        if self.steps == 0 or not self.labels:
            return
        print(f"Partition ratio |Q_s|/<|e^(-u_s)|> of the single-site species ({self.steps} steps of this run; "
              "near 0: close to the pole of -n_s ln Q_s):")
        for label, smallest, below in zip(self.labels, self.smallest, self.below):
            print(f"  {label}: minimum {smallest:.3g}, steps below 0.1 {below[0] / self.steps:.3g}, "
                  f"below 0.01 {below[1] / self.steps:.3g}")


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



def display_system(sim, chain_model):
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

    print("Chain model: %s" % (chain_model))
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
