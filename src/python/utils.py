"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import ast
import copy
import itertools
import json
import operator
import warnings
import hashlib
import os
import platform
import secrets
import subprocess
from functools import lru_cache
from types import MappingProxyType
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import yaml

from .validation import ValidationError, parse_chi_n  # noqa: F401 (parse_chi_n: former utils API)


def json_ready(value):
    """Convert parameter dictionaries to portable JSON values."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def mat_ready(value):
    """Convert a parameter dictionary for scipy.io.savemat, which has no None: an option
    given as None (its default) is left out, and paths become strings."""
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
    from . import _core
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
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
    from . import _core
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
    from . import _core
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


# =============================================================================
# Polymer architecture drawing (SCFT into polymer_figure_dir, LFTS/CLFTS into the recording directory)
# =============================================================================

DEFAULT_MONOMER_COLORS: List[str] = [
    "red", "blue", "green", "cyan", "magenta", "yellow"
]


def create_monomer_color_dict(
    segment_lengths: Dict[str, float],
    colors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a color dictionary for monomer types.

    Parameters
    ----------
    segment_lengths : dict
        Dictionary mapping monomer types to their statistical segment lengths.
    colors : list of str, optional
        List of color names to use. Defaults to DEFAULT_MONOMER_COLORS.

    Returns
    -------
    dict
        Dictionary mapping monomer types to colors.
    """
    if colors is None:
        colors = DEFAULT_MONOMER_COLORS

    # Colors beyond the list come from a fixed local stream: the same in every run, and the
    # global NumPy random state stays untouched
    extra_colors = np.random.default_rng(0)
    dict_color = {}
    for count, monomer_type in enumerate(segment_lengths.keys()):
        if count < len(colors):
            dict_color[monomer_type] = colors[count]
        else:
            dict_color[monomer_type] = extra_colors.random(3)

    return dict_color


def draw_polymer_architectures(polymers, ds, segment_lengths, prefix="polymer", directory="."):
    """Draw every polymer of a simulation to ``<directory>/<prefix>_<id>.png``.

    SCFT draws them into ``polymer_figure_dir`` when the input gives one; LFTS and CLFTS
    draw them into the recording directory when a run starts. To draw them elsewhere:
    ``draw_polymer_architectures(calc.polymers, calc.ds, calc.segment_lengths, directory=...)``.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    dict_color = create_monomer_color_dict(segment_lengths)
    for idx, polymer in enumerate(polymers):
        draw_polymer_architecture(polymer, idx, ds, dict_color,
                                  output_file=os.path.join(directory, f"{prefix}_{idx}.png"))


def draw_polymer_architecture(
    polymer: Dict[str, Any],
    polymer_id: int,
    ds: float,
    dict_color: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100
) -> None:
    """Draw polymer chain architecture as a graph.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary containing 'blocks_input' with block definitions.
    polymer_id : int
        Identifier for the polymer (used in title and filename).
    ds : float
        Contour step size for calculating segment numbers.
    dict_color : dict
        Dictionary mapping monomer types to colors.
    output_file : str, optional
        Output filename. Defaults to "polymer_{id}.png".
    figsize : tuple of int
        Figure size in inches. Defaults to (8, 8).
    dpi : int
        Resolution in dots per inch. Defaults to 100.
    """
    # Make a graph
    import networkx as nx               # imported here: only the drawing needs them
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for block in polymer["blocks_input"]:
        monomer_type = block[0]
        length = round(block[1] / ds)
        v = block[2]
        u = block[3]
        G.add_edge(v, u, weight=length, monomer_type=monomer_type)

    # Set node colors
    color_map = []
    for node in G:
        if len(G.edges(node)) == 1:
            color_map.append('yellow')
        else:
            color_map.append('gray')

    labels = nx.get_edge_attributes(G, 'weight')
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
    except ImportError:
        # pygraphviz not available, use spring_layout as fallback
        pos = nx.spring_layout(G)
    edge_colors = [dict_color[G[u][v]['monomer_type']] for u, v in G.edges()]

    plt.figure(figsize=figsize)
    title = f"Polymer ID: {polymer_id:2d},"
    title += f"\nColors of monomers: {dict_color},"
    title += "\nColor of chain ends: 'yellow',"
    title += "\nColor of junctions: 'gray',"
    title += "\nPlease note that the length of each edge is not proportional to the number of monomers in this image."
    plt.title(title)
    nx.draw(G, pos, node_color=color_map, edge_color=edge_colors, width=4, with_labels=True)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=labels, rotate=False,
        bbox=dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.5)
    )

    if output_file is None:
        output_file = f"polymer_{polymer_id:01d}.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def process_polymer_blocks(
    polymer: Dict[str, Any]
) -> List[List[Any]]:
    """Process polymer block definitions into blocks_input format.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary containing 'blocks' list.

    Returns
    -------
    list
        Processed blocks_input list with [type, length, v, u] format.

    Raises
    ------
    ValidationError
        If block vertex indices are inconsistent.
    """
    blocks_input = []
    has_node_number = "v" not in polymer["blocks"][0]

    for block in polymer["blocks"]:
        if has_node_number:
            if "v" in block:
                raise ValidationError(
                    "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            if "u" in block:
                raise ValidationError(
                    "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input) + 1])
        else:
            if "v" not in block:
                raise ValidationError(
                    "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            if "u" not in block:
                raise ValidationError(
                    "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            blocks_input.append([block["type"], block["length"], block["v"], block["u"]])

    return blocks_input


def process_random_copolymer(
    polymer: Dict[str, Any],
    segment_lengths: Dict[str, float]
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """Process random copolymer definitions.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary potentially containing random copolymer definition.
    segment_lengths : dict
        Current segment lengths dictionary.

    Returns
    -------
    tuple or None
        If random copolymer: (type_string, statistical_segment_length, fraction_dict)
        Otherwise: None

    Raises
    ------
    ValidationError
        If random copolymer definition is invalid.
    """
    # Check if this is a random copolymer
    is_random = any("monomer_fractions" in block for block in polymer["blocks"])
    if not is_random:
        return None

    if len(polymer["blocks"]) != 1:
        raise ValidationError("Only single block random copolymer is allowed.")

    statistical_segment_length = 0
    total_random_fraction = 0
    for monomer_type in polymer["blocks"][0]["monomer_fractions"]:
        statistical_segment_length += (
            segment_lengths[monomer_type] ** 2 * polymer["blocks"][0]["monomer_fractions"][monomer_type]
        )
        total_random_fraction += polymer["blocks"][0]["monomer_fractions"][monomer_type]
    statistical_segment_length = np.sqrt(statistical_segment_length)

    if not np.isclose(total_random_fraction, 1.0):
        raise ValidationError("The sum of volume fractions of random copolymer must be equal to 1.")

    random_type_string = polymer["blocks"][0]["type"]
    if random_type_string in segment_lengths:
        raise ValidationError(
            f"The name of random copolymer '{random_type_string}' is already used as a type "
            "in 'segment_lengths' or other random copolymer"
        )

    return (random_type_string, statistical_segment_length, polymer["blocks"][0]["monomer_fractions"])


# =============================================================================
# Langevin noise
# =============================================================================
def read_chains(params):
    """Monomer types, segment lengths, polymers with their blocks, χN and random copolymers of the input."""
    monomer_types = sorted(list(params["segment_lengths"].keys()))
    segment_lengths = copy.deepcopy(params["segment_lengths"])
    polymers = copy.deepcopy(params["polymers"])

    # Flory-Huggins parameters chi N; pairs the user did not give do not interact
    chi_n = parse_chi_n(params["chi_n"], segment_lengths)
    for pair in itertools.combinations(monomer_types, 2):
        key = ",".join(sorted(pair))
        if key not in chi_n:
            chi_n[key] = 0.0

    # Polymer chains: block connectivity, then random copolymers (one effective monomer type each)
    for polymer in polymers:
        polymer["blocks_input"] = process_polymer_blocks(polymer)
    random_fraction = {}
    for polymer in polymers:
        result = process_random_copolymer(polymer, segment_lengths)
        if result is not None:
            random_type, effective_segment_length, fraction = result
            polymer["block_monomer_types"] = [random_type]
            segment_lengths[random_type] = effective_segment_length
            random_fraction[random_type] = fraction
    return monomer_types, segment_lengths, polymers, chi_n, random_fraction


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


def mean_concentrations(sim):
    """Volume-averaged concentration of each monomer type from the chain compositions.

    mean(phi_i) = sum_p (f_p / alpha_p) sum_{blocks b of p} length_b * (fraction of i in b).
    Use the native block lengths, including accepted discrete lengths snapped
    to integer multiples of ds, to match the concentration normalization.
    """
    phi_mean = np.zeros(len(sim.monomer_types))
    for p in range(sim.prop_solver.get_n_polymer_types()):
        chain = sim.prop_solver.get_polymer(p)
        for block in chain.get_blocks():
            weight = chain.get_volume_fraction() * block.contour_length / chain.get_alpha()
            fractions = sim.random_fraction.get(block.monomer_type, {block.monomer_type: 1.0})
            for monomer_type, fraction in fractions.items():
                phi_mean[sim.monomer_types.index(monomer_type)] += weight * fraction
    return phi_mean


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

# =============================================================================
# Field compressors for the saddle-point search of the imaginary fields
# =============================================================================


# =============================================================================
# Parameter files (YAML / JSON)
# =============================================================================

class ConfigError(Exception):
    """Exception raised for configuration file errors."""
    pass


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load simulation parameters from a configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) formats.
    Automatically detects format from file extension.

    String values that use numpy, such as ``"4.0*np.sqrt(3)/2"``, are
    evaluated: numbers, ``+ - * / **``, parentheses, ``np.pi``, ``np.e`` and
    the elementwise functions in ``CONFIG_NP_FUNCTIONS`` are allowed, and
    nothing else is executed. Any other string containing ``np.`` is kept as
    it is, with a warning.

    Parameters
    ----------
    path : str or Path
        Path to configuration file.

    Returns
    -------
    dict
        Simulation parameters dictionary.

    Raises
    ------
    ConfigError
        If file format is not supported or file cannot be read.
    FileNotFoundError
        If configuration file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    try:
        with open(path, 'r') as f:
            if suffix in ('.yaml', '.yml'):
                params = yaml.safe_load(f)
            elif suffix == '.json':
                params = json.load(f)
            else:
                raise ConfigError(
                    f"Unsupported configuration file format: {suffix}. "
                    "Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax in {path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON syntax in {path}: {e}")

    if params is None:
        raise ConfigError(f"Configuration file is empty: {path}")

    if not isinstance(params, dict):
        raise ConfigError(f"Configuration must be a dictionary, got {type(params).__name__}")

    # Process special values (e.g., numpy expressions)
    params = _process_config_values(params)

    return params


# Numpy names a configuration expression may use
CONFIG_NP_CONSTANTS = ("pi", "e")
CONFIG_NP_FUNCTIONS = ("sqrt", "cbrt", "exp", "log", "log10", "sin", "cos", "tan",
                       "arcsin", "arccos", "arctan", "deg2rad", "rad2deg", "abs")
_CONFIG_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval_config_expression(text: str) -> float:
    """Value of an arithmetic expression such as "4.0*np.sqrt(3)/2".

    Evaluates the parsed expression node by node instead of calling eval, so
    only numbers, + - * / **, unary signs, np.pi, np.e and the functions in
    CONFIG_NP_FUNCTIONS (one argument each) can appear, and every intermediate
    value must be a finite real number. Raises ValueError otherwise.
    """
    def is_np(node):
        return isinstance(node, ast.Name) and node.id in ("np", "numpy")

    def finite_real(x, node):
        # (-8)**0.5 is complex in Python, np.sqrt(-1) is nan, np.log(0) is -inf
        if isinstance(x, complex) or not np.isfinite(x):
            raise ValueError(f"'{ast.unparse(node)}' is not a finite real number")
        return float(x)

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _CONFIG_OPERATORS:
            return finite_real(_CONFIG_OPERATORS[type(node.op)](value(node.left), value(node.right)), node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return value(node.operand) if isinstance(node.op, ast.UAdd) else -value(node.operand)
        if isinstance(node, ast.Attribute) and is_np(node.value) and node.attr in CONFIG_NP_CONSTANTS:
            return float(getattr(np, node.attr))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_np(node.func.value)
                and node.func.attr in CONFIG_NP_FUNCTIONS and len(node.args) == 1 and not node.keywords):
            return finite_real(getattr(np, node.func.attr)(value(node.args[0])), node)
        raise ValueError(f"'{ast.unparse(node)}' is not allowed in a configuration expression")

    with np.errstate(all="ignore"):
        return value(ast.parse(text.strip(), mode="eval").body)


def _process_config_values(obj: Any) -> Any:
    """Process configuration values, evaluating numpy expressions.

    Handles:
    - String expressions like "4.0*np.sqrt(3)/2" for lx (see _eval_config_expression)
    - Nested dictionaries and lists

    Parameters
    ----------
    obj : Any
        Configuration value to process.

    Returns
    -------
    Any
        Processed value.
    """
    if isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    elif isinstance(obj, str):
        if 'np.' in obj or 'numpy.' in obj:
            try:
                return _eval_config_expression(obj)
            except (SyntaxError, ValueError, ArithmeticError, RecursionError, MemoryError) as e:
                # A string that LOOKS like a numpy expression but cannot be
                # evaluated is almost certainly a typo in the config; passing
                # it through silently would surface later as a confusing
                # type error. Warn so the user can fix the expression.
                warnings.warn(
                    f"Config value {obj!r} looks like a numpy expression but "
                    f"was not evaluated ({e}); using the raw string instead.")
                return obj
        return obj
    else:
        return obj


# =============================================================================
# VTK output for ParaView
# =============================================================================

def save_vtk(
    path: Union[str, Path],
    fields: Union[np.ndarray, Dict[str, np.ndarray]],
    lx: List[float],
    field_name: str = "phi",
    nx: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
) -> None:
    """Save fields as a legacy VTK file for ParaView.

    The grid points sit at the cell centres, (i + 1/2) * lx / nx along each
    lattice vector, as in the solvers, so the outline ParaView draws through
    the points lies half a cell inside the box (the walls of a non-periodic
    box are at 0 and lx).

    Parameters
    ----------
    path : str or Path
        Output file path (should end with .vtk).
    fields : np.ndarray or dict
        One real field, or {name: field} to store several fields in one file
        (for example SCFTResult.phi). A field has the grid shape (1D, 2D or 3D),
        or is flat when nx is given.
    lx : list of float
        Box lengths, one per dimension.
    field_name : str
        Name of the field when a single array is given (default: "phi").
    nx : list of int, optional
        Grid size, needed for flat arrays.
    angles : list of float, optional
        Lattice angles [alpha, beta, gamma] in degrees, or [gamma] for a 2D
        cell. A non-orthogonal cell is written as a structured grid with
        explicit point coordinates.
    """
    if not isinstance(fields, dict):
        fields = {field_name: fields}
    if not fields:
        raise ValueError("No field to save")

    if nx is None:
        nx = list(np.shape(next(iter(fields.values()))))
    nx = [int(n) for n in nx]
    dim = len(nx)
    if not 1 <= dim <= 3 or len(lx) != dim:
        raise ValueError(f"nx and lx must have the same length 1, 2 or 3, got nx={nx}, lx={list(lx)}")
    n_points = int(np.prod(nx))

    # Fields in VTK point order (x fastest); names without spaces
    data = {}
    for name, field in fields.items():
        field = np.asarray(field)
        if np.iscomplexobj(field):
            raise ValueError(f"Field '{name}' is complex; VTK stores real values "
                             "(save the real and imaginary parts separately)")
        if field.size != n_points:
            raise ValueError(f"Field '{name}' has {field.size} values, the grid {nx} has {n_points}")
        data[str(name).replace(" ", "_")] = np.reshape(field, nx).ravel(order="F")

    # Pad to three dimensions
    nx3 = nx + [1] * (3 - dim)
    lx3 = [float(x) for x in lx] + [1.0] * (3 - dim)
    if angles is not None and len(angles) == 1:
        angles = [90.0, 90.0, angles[0]]          # 2D: [gamma]
    if angles is not None and len(angles) != 3:
        raise ValueError(f"angles must be [alpha, beta, gamma] or [gamma], got {list(angles)}")
    alpha, beta, gamma = np.deg2rad(angles) if angles is not None else (np.pi/2,)*3
    orthogonal = np.allclose([alpha, beta, gamma], np.pi/2, atol=1e-10)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Polymer field data\n")
        f.write("ASCII\n")
        if orthogonal:
            spacing = [lx3[d]/nx3[d] for d in range(3)]
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write("DIMENSIONS %d %d %d\n" % tuple(nx3))
            f.write("ORIGIN %.10g %.10g %.10g\n" % tuple(0.5*h for h in spacing))
            f.write("SPACING %.10g %.10g %.10g\n" % tuple(spacing))
        else:
            # Lattice vectors as in ComputationBox: a along x, b in the xy plane
            a = lx3[0]*np.array([1.0, 0.0, 0.0])
            b = lx3[1]*np.array([np.cos(gamma), np.sin(gamma), 0.0])
            cx = np.cos(beta)
            cy = (np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)
            c = lx3[2]*np.array([cx, cy, np.sqrt(max(1.0 - cx*cx - cy*cy, 0.0))])
            frac = [(np.arange(nx3[d]) + 0.5)/nx3[d] for d in range(3)]
            fz, fy, fx = np.meshgrid(frac[2], frac[1], frac[0], indexing="ij")   # x fastest after ravel
            points = np.outer(fx.ravel(), a) + np.outer(fy.ravel(), b) + np.outer(fz.ravel(), c)
            f.write("DATASET STRUCTURED_GRID\n")
            f.write("DIMENSIONS %d %d %d\n" % tuple(nx3))
            f.write("POINTS %d double\n" % n_points)
            np.savetxt(f, points, fmt="%.10g")
        f.write("POINT_DATA %d\n" % n_points)
        for name, values in data.items():
            f.write("SCALARS %s float 1\n" % name)
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, values, fmt="%.7g")
