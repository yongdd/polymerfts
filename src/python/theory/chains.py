"""The polymer chains of a simulation: reading them from the input, their concentrations, their drawings.

read_chains() reads the input. chain_fields() and solve_chains() give the concentrations of the monomer
types in given fields, with what SCFT, LFTS and CLFTS have in common; each of them adds to this what
is its own, such as the electrostatic potential or the grand-canonical chains.
"""

import copy
import itertools
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from ..utils.common import report_partition_consistency
from ..utils.validation import ValidationError, parse_chi_n

DEFAULT_MONOMER_COLORS: List[str] = [
    "red", "blue", "green", "cyan", "magenta", "yellow"
]


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
        Processed blocks_input list with [type, length, v, u] format, and the
        multiplicity as a fifth entry for blocks that have one.

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
        if "multiplicity" in block:
            blocks_input[-1].append(block["multiplicity"])

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


def chain_fields(sim, w):
    """The field of each monomer type in w and of each random copolymer, the mean of its monomer types."""
    w_input = {sim.monomer_types[i]: w[i] for i in range(len(sim.monomer_types))}
    for random_polymer_name, random_fraction in sim.random_fraction.items():
        w_input[random_polymer_name] = np.zeros(sim.prop_solver.n_grid, dtype=np.result_type(np.asarray(w[0]), np.float64))
        for monomer_type, fraction in random_fraction.items():
            w_input[random_polymer_name] += w_input[monomer_type] * fraction
    return w_input


def solve_chains(sim, w_input, q_init=None):
    """The concentrations of the monomer types in the fields w_input, and the time they took.

    The fields are smeared before they enter the Boltzmann weights of the propagators (Matsen et al.,
    J. Chem. Phys. 164, 014905 (2026)), and each random copolymer adds to its monomer types.
    """
    elapsed_time = {}
    w_input_for_propagator = sim.smearing.apply_to_dict(w_input)

    # For the given fields, compute propagators
    time_solver_start = time.time()
    sim.prop_solver.compute_propagators(w_input_for_propagator, q_init=q_init)
    elapsed_time["solver"] = time.time() - time_solver_start

    # Compute concentrations for each monomer type
    time_phi_start = time.time()
    phi = {}
    sim.prop_solver.compute_concentrations()
    for monomer_type in sim.monomer_types:
        phi[monomer_type] = sim.prop_solver.get_concentration(monomer_type)

    # Add random copolymer concentration to each monomer type
    for random_polymer_name, random_fraction in sim.random_fraction.items():
        phi[random_polymer_name] = sim.prop_solver.get_concentration(random_polymer_name)
        for monomer_type, fraction in random_fraction.items():
            phi[monomer_type] += phi[random_polymer_name] * fraction
    elapsed_time["phi"] = time.time() - time_phi_start
    return phi, elapsed_time


def check_partition_once(sim):
    """Partition consistency check (forward vs backward propagators), once per run."""
    if not getattr(sim, "_partition_checked", False):
        sim._partition_checked = True
        if not report_partition_consistency(sim.prop_solver):
            raise ValidationError("Partition consistency check failed (forward/backward mismatch).")


def mean_concentrations(sim):
    """Volume-averaged concentration of each monomer type from the chain compositions.

    mean(phi_i) = sum_p (f_p / alpha_p) sum_{blocks b of p} copies_b * length_b * (fraction of i in b).
    Use the native block lengths, including accepted discrete lengths snapped
    to integer multiples of ds, to match the concentration normalization.
    """
    phi_mean = np.zeros(len(sim.monomer_types))
    for p in range(sim.prop_solver.get_n_polymer_types()):
        chain = sim.prop_solver.get_polymer(p)
        for block in chain.get_blocks():
            weight = chain.get_volume_fraction() * block.n_copies * block.contour_length / chain.get_alpha()
            fractions = sim.random_fraction.get(block.monomer_type, {block.monomer_type: 1.0})
            for monomer_type, fraction in fractions.items():
                phi_mean[sim.monomer_types.index(monomer_type)] += weight * fraction
    return phi_mean


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
    labels = {}
    multiplied_nodes = set()    # nodes where copies of a block (a fifth entry, its multiplicity) hang
    for block in polymer["blocks_input"]:
        monomer_type, contour_length, v, u = block[:4]
        length = round(contour_length / ds)
        labels[(v, u)] = str(length)
        if len(block) > 4 and block[4] != 1:
            labels[(v, u)] += f" ×{block[4]:.12g}"
            multiplied_nodes.add(v)
        G.add_edge(v, u, weight=length, monomer_type=monomer_type)

    # Set node colors; a chain end that carries copies is a junction
    color_map = []
    for node in G:
        if len(G.edges(node)) == 1 and node not in multiplied_nodes:
            color_map.append('yellow')
        else:
            color_map.append('gray')

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
    if multiplied_nodes:
        title += "\n'×m' on a block (v, u): m copies of it and of everything on its u side, attached at v,"
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
