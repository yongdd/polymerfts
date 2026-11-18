"""What SCFT returns and writes: its result classes, the chemical potentials and the grand potential of the
final state of a run, the result files and the final report."""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import yaml

from ..utils.common import atomic_savemat, atomic_write_json, atomic_write_text, json_ready, mat_ready, simulation_metadata


@dataclass
class SCFTResult:
    """Result object for SCFT simulations.

    Encapsulates all outputs from an SCFT simulation run, providing
    a clean interface for accessing results.

    Attributes
    ----------
    converged : bool
        Whether the simulation converged within tolerance.
    free_energy : float
        Free energy relative to the homogeneous state at the same composition,
        in units of rho_0 V / N. Zero for a disordered bulk state.
    error_level : float
        Final convergence error (L2 norm of field residuals).
    iterations : int
        Number of iterations performed.
    phi : dict
        Monomer concentration fields {monomer_type: array} on the full grid
        (expanded from the reduced basis when a space group is used).
    w : dict
        Potential fields {monomer_type: array} on the full grid. They can be
        passed back to SCFT.run as initial_fields, for example to continue a
        parameter sweep from the previous solution.
    partition_functions : list
        Single-chain partition functions for each polymer type.
    stress : np.ndarray or None
        Stress tensor components (if relax_cell was True).
    lx : list
        Final box dimensions.
    angles : list or None
        Final lattice angles [alpha, beta, gamma] in degrees for 2D and 3D,
        including fixed angles. None in 1D.
    nx : list
        Grid dimensions.
    monomer_types : list
        List of monomer type labels.
    params : dict
        Original simulation parameters.
    free_energy_homogeneous : float
        Full homogeneous Flory-Huggins value at the same composition.
    free_energy_absolute : float
        Total canonical Helmholtz free energy, beta F_total / (rho_0 V / N),
        including ideal mixing entropy. Equals free_energy + free_energy_homogeneous.
        Grand-canonical point ions contribute -e^(beta mu_s) Q_s (a semi-grand free energy).
    chemical_potentials : list or None
        beta mu_p per polymer molecule, with mean(xi) = 0 for incompressible
        systems. Not available for grafted chains or supplied initial propagators.
    grand_potential : float or None
        free_energy_absolute - sum_p chemical_potentials[p] * phi_p / alpha_p, in the
        same pressure convention; canonical point ions also subtract
        sum_s N c_s ion_chemical_potentials[s]. Use coexistence_quantities() for phase
        comparisons of systems without electrostatics.
    polymer_lengths, polymer_volume_fractions : list or None
        Relative chain lengths alpha_p and polymer volume fractions phi_p.
    psi : np.ndarray or None
        Electrostatic potential ψ on the full grid (with `electrostatics`); it can be passed
        back to SCFT.run as initial_psi.
    ion_chemical_potentials : list or None
        βμ_s of the point ions, in the order of `electrostatics.ions`.
    """
    converged: bool
    free_energy: float
    error_level: float
    iterations: int
    phi: Dict[str, np.ndarray]
    w: Dict[str, np.ndarray]
    partition_functions: List[float]
    lx: List[float]
    nx: List[int]
    monomer_types: List[str]
    params: Dict[str, Any]
    stress: Optional[np.ndarray] = None
    angles: Optional[List[float]] = None
    free_energy_homogeneous: Optional[float] = None
    chemical_potentials: Optional[List[float]] = None
    grand_potential: Optional[float] = None
    polymer_lengths: Optional[List[float]] = None
    polymer_volume_fractions: Optional[List[float]] = None
    free_energy_absolute: Optional[float] = None
    psi: Optional[np.ndarray] = None
    ion_chemical_potentials: Optional[List[float]] = None

    def coexistence_quantities(self, reference_polymer=0):
        """Return exchange chemical potentials and the reduced grand potential.

        Requires a converged, incompressible, periodic bulk solution. The
        reference polymer's chemical potential is set to zero by a pressure shift.
        """
        from .phase_equilibrium import coexistence_quantities
        return coexistence_quantities(self, reference_polymer)

    def get_concentration(self, monomer_type: str) -> np.ndarray:
        """Get concentration field for a monomer type.

        Parameters
        ----------
        monomer_type : str
            Monomer type label (e.g., "A", "B").

        Returns
        -------
        np.ndarray
            Concentration field for the specified monomer.

        Raises
        ------
        KeyError
            If monomer type not found.
        """
        return self.phi[monomer_type]


@dataclass
class IterationInfo:
    """Information about a single SCFT iteration.

    Used for progress callbacks to provide iteration details.

    Attributes
    ----------
    iteration : int
        Current iteration number.
    error_level : float
        Current convergence error.
    energy : float
        Current free energy.
    mass_error : float
        Mass conservation error.
    partition_functions : list
        Partition functions for each polymer.
    lx : list or None
        Current box dimensions (if relax_cell).
    angles : list or None
        Current lattice angles [alpha, beta, gamma] in degrees for 2D and 3D,
        including fixed angles. None in 1D.
    """
    iteration: int
    error_level: float
    energy: float
    mass_error: float
    partition_functions: List[float]
    lx: Optional[List[float]] = None
    angles: Optional[List[float]] = None


def save_results(sim, path):
    """Save SCFT results to file in MATLAB, JSON, or YAML format.

    Exports the computed fields, concentrations, and simulation parameters
    to a file. The format is determined by the file extension (.mat, .json,
    or .yaml).

    Parameters
    ----------
    path : str
        Output file path. Extension determines format:

        - '.mat' : MATLAB format (scipy.io.savemat)
        - '.json' : JSON format (human-readable, larger files)
        - '.yaml' : YAML format (human-readable, structured)

    Raises
    ------
    ValueError
        If file extension is not .mat, .json, or .yaml.
    """
    # Make a dictionary for data
    m_dic = {"initial_params": sim.params,
        "dim":sim.prop_solver.dim, "nx":sim.prop_solver.nx, "lx":sim.prop_solver.get_lx(),
        "angles":list(sim.prop_solver.get_angles_degrees()),
        "monomer_types":sim.monomer_types, "chi_n":dict(sim.chi_n), "chain_model":sim.chain_model, "ds":sim.ds,
        "eigenvalues": sim.mpt.eigenvalues, "matrix_a": sim.mpt.matrix_a, "matrix_a_inverse": sim.mpt.matrix_a_inv}
    metadata = simulation_metadata(sim)
    metadata.update(execution_status="converged" if sim.error_level < sim.tolerance else "not_converged",
                    iterations=sim.iter)
    m_dic["simulation_metadata_json"] = json.dumps(metadata)

    m_dic.update(free_energy=sim.free_energy,
                 free_energy_absolute=sim.free_energy_absolute,
                 free_energy_homogeneous=sim.free_energy_homogeneous,
                 partition_functions=sim.partition_functions,
                 polymer_lengths=sim.polymer_lengths,
                 polymer_volume_fractions=sim.polymer_volume_fractions)
    if sim.chemical_potentials is not None:
        m_dic["chemical_potentials"] = sim.chemical_potentials
        m_dic["grand_potential"] = sim.grand_potential

    if sim.sg is not None:
        m_dic["space_group_symbol"] = sim.sg.get_spacegroup_symbol()
        m_dic["space_group_hall_number"] = sim.sg.get_hall_number()

    # Add w fields to the dictionary (expand to full grid if space group is set)
    for i, name in enumerate(sim.monomer_types):
        m_dic["w_" + name] = sim._to_full_grid(sim.w[i])

    # Add concentrations to the dictionary (expand to full grid if space group is set)
    for name in sim.monomer_types:
        m_dic["phi_" + name] = sim._to_full_grid(sim.phi[name])

    # Electrostatic potential and the chemical potentials of the point ions
    if sim.electrostatics is not None:
        m_dic["psi"] = sim._to_full_grid(sim.electrostatics.psi)
        if sim.ion_chemical_potentials is not None:
            m_dic["ion_chemical_potentials"] = sim.ion_chemical_potentials

    # Get input file extension
    _, file_extension = os.path.splitext(path)

    # A failure keeps an existing file: each format is written to a new file that then replaces it.
    # The converters (mat_ready, json_ready) copy the nested parameters, so sim.params stays as it is.

    # Save data in matlab format (MAT files have no None: options given as None are left out)
    if file_extension == ".mat":
        atomic_savemat(path, mat_ready(m_dic), long_field_names=True, do_compression=True)

    # Save data in json format (strict JSON: a non-finite number, such as the chemical
    # potential -inf of a species with volume fraction 0, becomes {"nonfinite_float": "-inf"})
    elif file_extension == ".json":
        atomic_write_json(path, m_dic)

    # Save data in yaml format (YAML has .inf: a non-finite number stays a number)
    elif file_extension == ".yaml":
        atomic_write_text(path, yaml.safe_dump(json_ready(m_dic, strict=False), default_flow_style=None, width=90,
                                               sort_keys=False))
    else:
        raise ValueError("Invalid output file extension. Use .mat, .json, or .yaml.")


def ion_chemical_potentials(sim, phi):
    """With electrostatics, prints the Poisson residual of the final ψ of SCFT.run (concentrations phi) and
    returns βμ_s of the point ions, which it prints with the ions per reference chain; None without
    point ions."""
    es = sim.electrostatics
    if es is None:
        return None
    print("Poisson residual |c - (k² + κ²)ψ/E| relative to the charge density: %.3e"
          % es.poisson_error(sim.smearing.apply_to_dict(phi)))
    if not es.ions:
        return None
    mu = es.ion_chemical_potentials()
    print("Point ions: chemical potentials βμ", mu, "ions per reference chain", es.ion_number_densities())
    return mu


def chemical_potentials(sim, q_init):
    """βμ_p of the chains and the grand potential of the final state of SCFT.run, which it prints; (None,
    None) for grafted chains (initial propagators), which have no chemical potential."""
    if q_init is not None or any(p.get("initial_conditions") for p in sim.polymers):
        return None, None
    mu = sim.compute_chemical_potentials()
    present = sim.chain_densities > 0.0
    grand_potential = float(sim.free_energy_absolute - np.dot(sim.chain_densities[present], np.array(mu)[present]))
    # Canonical point ions add -N c_s βμ_s; the term of grand-canonical ions is grand already
    es = sim.electrostatics
    if es is not None:
        for ion, ion_mu in zip(es.ions, sim.ion_chemical_potentials or []):
            if ion["concentration"] is not None and ion["concentration"] > 0:
                grand_potential -= es.n_segments * ion["concentration"] * ion_mu
    print("Chemical potentials βμ (mean ξ = 0 for incompressible systems):", mu)
    print("Grand potential in the same pressure convention: %.9f" % grand_potential)
    return mu, grand_potential


def scft_result(sim):
    """The SCFTResult of the last SCFT.run, on the full grid."""
    es = sim.electrostatics
    return SCFTResult(
        converged=sim.error_level < sim.tolerance,
        free_energy=sim.free_energy,
        error_level=sim.error_level,
        iterations=sim.iter,
        phi={name: sim._to_full_grid(sim.phi[name]) for name in sim.monomer_types},
        w={name: sim._to_full_grid(sim.w[i]) for i, name in enumerate(sim.monomer_types)},
        partition_functions=list(sim.partition_functions),
        lx=list(sim.prop_solver.get_lx()),
        nx=list(sim.prop_solver.nx),
        monomer_types=sim.monomer_types,
        params=sim.params,
        stress=sim.cell.stress_array if sim.relax_cell else None,
        angles=list(sim.prop_solver.get_angles_degrees()) if sim.prop_solver.dim > 1 else None,
        free_energy_homogeneous=sim.free_energy_homogeneous,
        free_energy_absolute=sim.free_energy_absolute,
        chemical_potentials=sim.chemical_potentials,
        grand_potential=sim.grand_potential,
        polymer_lengths=list(sim.polymer_lengths),
        polymer_volume_fractions=list(sim.polymer_volume_fractions),
        psi=None if es is None else sim._to_full_grid(es.psi),
        ion_chemical_potentials=sim.ion_chemical_potentials
    )


def iteration_info(sim, iter, error_level, energy_total, mass_error, total_partitions):
    return IterationInfo(
        iteration=iter,
        error_level=error_level,
        energy=energy_total,
        mass_error=mass_error,
        partition_functions=total_partitions,
        lx=list(sim.prop_solver.get_lx()) if sim.relax_cell else None,
        angles=list(sim.prop_solver.get_angles_degrees()) if sim.prop_solver.dim > 1 else None
    )


def print_free_energies(sim, energy_total, energy_absolute):
    print("Free energy relative to the homogeneous state: %.9f" % energy_total)
    print("Full canonical free energy βF/(ρ₀V/N): %.9f" % energy_absolute)
    print("Homogeneous Flory-Huggins free energy: %.9f" % sim.free_energy_homogeneous)
    print("Relative free energy divided by the number of chains of each type:")
    for p, concentration in enumerate(sim.chain_densities):
        if concentration > 0.0:
            print("\tβΔF/n_%d : %12.7f" % (p, energy_total / concentration))
