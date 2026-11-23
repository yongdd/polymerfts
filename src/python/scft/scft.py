import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Union

from ..theory.symmetric_formulation import SymmetricPolymerTheory
from ..theory.chains import chain_solver, partition_functions
from ..utils.validation import (with_overrides, validate_scft_params, ValidationError, NumericalError,
                         point_ion_concentrations, open_charges, reference_segments, finite_values,
                         initial_field_values)
from ..utils.common import (
    report_partition_consistency,
    display_system,
    check_partition_functions,
)
from .cell_relaxation import CellRelaxation
from .grid import FieldGrid
from .scft_results import (SCFTResult, IterationInfo, scft_result, iteration_info, print_free_energies, save_results,
                           ion_chemical_potentials, chemical_potentials)
from ..theory.chains import read_chains, mean_concentrations, draw_polymer_architectures, chain_fields, solve_chains, check_partition_once
from ..utils.file_formats import load_config
from ..theory.smearing import smearing_of
from ..theory.electrostatics import MeanFieldElectrostatics


class SCFT:
    """Self-consistent field theory (SCFT) of polymer melts and solutions.

    Finds the saddle point of the field-theoretic Hamiltonian with Anderson
    mixing for arbitrary polymer architectures (linear, branched, random
    copolymers, mixtures), optionally relaxing the box (and the angles of
    non-orthogonal cells) and using space-group symmetry.

    Parameters
    ----------
    params : dict
        Simulation parameters; every key is documented in
        docs/references/Parameters.md. Required: nx, lx, chain_model, ds,
        segment_lengths, chi_n, polymers, optimizer, relax_cell.

    Attributes
    ----------
    prop_solver : PropagatorSolver
        Propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    grid : FieldGrid
        The axis order of the input, the space group (also sg) and the bases of the fields.
    phi, w, free_energy, free_energy_absolute, error_level, iter
        Set by run(): concentrations {type: array}, fields (M, n_grid),
        relative and full CE free energies, final error and iteration count.

    Examples
    --------
    >>> calc = SCFT(params)                       # params: see docs/references/Parameters.md
    >>> calc.run(initial_fields={"A": w_A, "B": w_B})
    >>> calc.save_results("fields.mat")
    >>> calc.free_energy, calc.phi["A"], calc.w[0]

    Complete scripts: examples/block_copolymer/scft/ (Lamella3D.py, Gyroid.py, phases/),
    examples/blend/scft/ and examples/charged/scft/.
    """
    def __init__(self, params):

        # Validate input parameters (a copy with the current names)
        params = validate_scft_params(params)

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)

        # Monomer types, segment lengths, polymer chains, χN and random copolymers
        self.monomer_types, self.segment_lengths, self.polymers, self.chi_n, self.random_fraction = read_chains(params)

        # Draw the chain architectures (polymer_<id>.png) only into a directory the input names,
        # so that constructing a calculation writes no files
        if params.get("polymer_figure_dir") is not None:
            draw_polymer_architectures(self.polymers, params["ds"], self.segment_lengths,
                                       directory=params["polymer_figure_dir"])

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

        # χN matrix in the order of monomer_types
        M = len(self.monomer_types)
        self.matrix_chi = self.mpt.matrix_chi
        # Incompressible SCFT needs at least two monomer types. A singular χN (a zero pair in a
        # three-monomer system, chemically identical monomers) is fine, and so is 1ᵀ(χN)⁻¹1 = 0,
        # where the auxiliary field of a zero eigenvalue of the projected χN has a linear term:
        # it acts as a uniform field, and the free energy is computed from the monomer fields
        if self.zeta_n is None and M < 2:
            raise ValidationError("Incompressible SCFT needs at least two monomer types.")

        # Field residual (see the run loop): the composition part is projected with
        # P⊥ = I - eeᵀ/M, and the pressure part is scaled by κ = max(max|λ|, 1)/2 with λ
        # the eigenvalues of P⊥ χN P⊥, the size of the composition-pressure coupling
        # (χN/2 for AB)
        self.matrix_q_perp = np.identity(M) - np.ones((M, M))/M
        projected_chi = self.matrix_q_perp @ self.matrix_chi @ self.matrix_q_perp
        self.pressure_kappa = max(np.max(np.abs(np.linalg.eigvalsh(projected_chi))), 1.0)/2

        # The grid of the fields (grid.py): nx, lx, bc and angles of the input in the standard axis order
        # [a, b, c] (axis_ordering), and the space group with its basis, before the solver that works in it
        self.grid = FieldGrid(self, params)
        self.axis_ordering, self.sg = self.grid.axis_ordering, self.grid.sg

        # Lattice angles [alpha, beta, gamma] in degrees
        angles = params.get("angles", None)

        # Propagator solver with the polymers: the C++ computation box, molecules and solver live inside it
        self.prop_solver = chain_solver(params, self.segment_lengths, self.polymers, space_group=self.sg)

        # Computation box as a public attribute (the propagator computation is created on first use)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Smearing of the interactions, on full-grid fields (validate_scft_params refuses it with a space group)
        self.smearing = smearing_of(params, self.prop_solver)

        # Electrostatics: the charged types feel z_i ψ, and ψ (self.electrostatics.psi) solves the Poisson
        # equation with the charges of the chains and the point ions (electrostatics.py, MeanFieldElectrostatics)
        self.electrostatics = None
        if params.get("electrostatics") is not None:
            self._setup_electrostatics(params)

        # The lattice parameters that vary with relax_cell
        self.cell = CellRelaxation(self, params, angles)

        # Anderson mixing (C++ class) for the saddle-point iteration of one vector: the fields (in the
        # working basis of the grid), ψ with electrostatics and the lattice parameters that relax
        n_var = M * self.grid.n_irreducible + self.cell.n_parameters
        if self.electrostatics is not None:
            n_var += self.prop_solver.n_grid
        optimizer = params["optimizer"]
        self.field_optimizer = self.prop_solver.create_anderson_mixing(n_var,
            optimizer["max_hist"],     # maximum number of history
            optimizer["start_error"],  # when switch to AM from simple mixing
            optimizer["mix_min"],      # minimum mixing rate of simple mixing
            optimizer["mix_init"])     # initial mixing rate of simple mixing

        display_system(self, params["chain_model"])
        print("Pressure scale of the field residual, κ: %f" % self.pressure_kappa)
        self.prop_solver.display_statistics()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.relax_cell = params["relax_cell"]

        self.max_iter = params.get("max_iter", 2000)       # maximum number of iterations
        self.tolerance = params.get("tolerance", 1e-8)     # stop when the self-consistency error is below this

    def _setup_electrostatics(self, params):
        """The Poisson equation for ψ and what it needs of the input."""
        # A fixed charge is given in the axis order of nx, like the initial fields
        if params["electrostatics"].get("fixed_charge") is not None:
            params["electrostatics"]["fixed_charge"] = self.grid.to_standard_axes(params["electrostatics"]["fixed_charge"])
        self.electrostatics = MeanFieldElectrostatics(self, params["electrostatics"], reference_segments(params),
                                                      ion_concentrations=point_ion_concentrations(params),
                                                      open_charges=open_charges(params))
        es = self.electrostatics
        es.psi = np.zeros(self.prop_solver.n_grid)
        print("Electrostatics: charges", es.charges, "coupling E = %g" % es.coupling
              + ("" if es.screening_length is None else ", screening length %g" % es.screening_length))
        for ion in es.ions:
            amount = ("chemical potential %g (grand canonical)" % ion["chemical_potential"] if ion["concentration"] is None
                      else "concentration %g (with counterions)" % ion["concentration"])
            print("Point ion %s: valence %g, %s, smearing length %g"
                  % (ion["name"], ion["valence"], amount, ion["smearing_length"]))

    # =========================================================================
    # Class Methods
    # =========================================================================

    @classmethod
    def from_file(cls, path: Union[str, Path], **overrides) -> "SCFT":
        """Create SCFT instance from configuration file.

        Loads parameters from YAML or JSON configuration file and
        creates an SCFT instance. Supports parameter overrides.

        Parameters
        ----------
        path : str or Path
            Path to configuration file (.yaml, .yml, or .json).
        **overrides : Any
            Parameter values to override from the file.

        Returns
        -------
        SCFT
            Configured SCFT instance.
        """
        params = with_overrides(load_config(path), overrides)
        return cls(params)

    def compute_concentrations(self, w, q_init=None, stress=False):
        """Compute monomer concentration fields for given potential fields.

        NOTE: unlike LFTS/CLFTS.compute_concentrations (which take AUXILIARY
        fields), this method takes MONOMER potential fields directly.

        Solves the modified diffusion equations to compute chain propagators,
        then integrates over chain contours to obtain monomer concentration
        distributions. Handles both block copolymers and random copolymers.
        q_init optionally supplies initial propagator values for grafted
        polymers (passed through to PropagatorSolver.compute_propagators).
        With stress the propagators also give their stress, in the same pass
        (PropagatorSolver.compute_concentrations_and_stress), which the cell
        relaxation reads (CellRelaxation.stress_error).

        Parameters
        ----------
        w : ndarray
            Potential fields, shape (M, total_grid) where M is the number
            of monomer types. Each row contains the field for one monomer type.

        Returns
        -------
        phi : dict
            Monomer concentration fields, {monomer_type: concentration_array}.
            Each array has length total_grid.
        elapsed_time : dict
            Timing information:

            - 'solver' : float
                Time (seconds) for propagator computation.
            - 'phi' : float
                Time (seconds) for concentration integration.
        """
        # The fields of the optimizer (in the irreducible basis with a space group) in the basis of the solver
        w_reduced = self.grid.to_solver_basis(w)

        # The fields of the chains: charged types feel w_i + z_i ψ with the current ψ
        w_input = chain_fields(self, w_reduced)
        if self.electrostatics is not None:
            self.electrostatics.add_potential(w_input)

        phi, elapsed_time = solve_chains(self, w_input, q_init, stress=stress)
        check_partition_once(self, lambda: self.grid.full_grid_fields(w))

        # The concentrations in the basis of the optimizer
        return self.grid.to_working_basis(phi), elapsed_time

    def save_results(self, path):
        """Saves the results to a MATLAB, JSON or YAML file, by the extension of path (scft_results.py)."""
        save_results(self, path)

    def mean_concentrations(self):
        """Volume-averaged concentration of each monomer type from the chain compositions."""
        return mean_concentrations(self)

    # =========================================================================
    # run() helpers (used more than once, or long enough to stand on their own)
    # =========================================================================

    def _pin_field_means(self, w):
        """Normalize the field means in place (in the working basis of the grid).

        Compressible systems have no gauge freedom: the means are always pinned to the
        saddle point, mean(w_i) = sum_j chi_ij mean(phi_j) + xi with
        xi = zeta (sum_j mean(phi_j) - 1) and the mean concentrations of the chain compositions
        (mean_concentrations; the residual leaves the means out).
        Incompressible fields are shifted to zero mean (the uniform modes do not change the
        concentrations, and the free energy does not depend on them).
        """
        target = np.zeros(len(self.monomer_types))
        if self.zeta_n is not None:
            phi_mean = self.mean_concentrations()
            target = self.matrix_chi @ phi_mean + self.zeta_n * (np.sum(phi_mean) - 1.0)
        for i in range(len(self.monomer_types)):
            w[i] += target[i] - self.grid.mean(w[i])

    def compute_chemical_potentials(self, w=None, partition_functions=None):
        """Restore the saddle-point field means and return beta mu per chain.

        w and partition_functions must describe the same CE state; by default
        use the final fields and Q saved by run(). w may also be a dict of fields,
        such as SCFTResult.w (on the full grid with a space group). No propagators
        are recomputed.
        Incompressible: mean(xi) = 0. Compressible: mean(xi) = zeta_n (sum(phi)-1).
        """
        if w is None:
            w = self.w
        if isinstance(w, dict):
            w = [w[m] for m in self.monomer_types]
        if partition_functions is None:
            partition_functions = self.partition_functions
        phi_mean = self.mean_concentrations()
        target = self.matrix_chi @ phi_mean
        if self.zeta_n is not None:
            target += self.zeta_n * (np.sum(phi_mean) - 1.0)
        # A field on the full grid (SCFTResult.w with a space group): the mean of its grid points
        means = [float(np.mean(field)) if self.sg is not None and np.size(field) == self.prop_solver.total_grid
                 else self.grid.mean(field) for field in w]
        shift = target - np.array(means)
        mu = []
        for p in range(self.prop_solver.get_n_polymer_types()):
            chain = self.prop_solver.get_polymer(p)
            concentration = chain.get_volume_fraction() / chain.get_alpha()
            if concentration == 0.0:
                mu.append(float("-inf"))
                continue
            field_shift = 0.0
            for block in chain.get_blocks():
                fractions = self.random_fraction.get(block.monomer_type, {block.monomer_type: 1.0})
                for monomer, fraction in fractions.items():
                    field_shift += block.n_copies * block.contour_length * fraction * shift[self.monomer_types.index(monomer)]
            mu.append(float(np.log(concentration) - np.log(partition_functions[p]) + field_shift))
        return mu

    def run(
        self,
        initial_fields: Dict[str, np.ndarray],
        q_init: Optional[Dict] = None,
        on_iteration: Optional[Callable[[IterationInfo], Optional[bool]]] = None,
        return_result: bool = False,
        initial_psi: Optional[np.ndarray] = None
    ) -> Optional[SCFTResult]:
        """Run SCFT iteration to find saddle point of the Hamiltonian.

        Iteratively solves the self-consistent field equations until convergence
        or maximum iterations reached. At each iteration:

        1. Computes monomer concentrations φ for current fields w
        2. Calculates self-consistency error and free energy
        3. Updates fields using optimizer (Anderson mixing)
        4. Optionally adjusts box size to minimize stress (if relax_cell=True)

        The iteration continues until the error_level falls below tolerance or
        max_iter is reached.

        Parameters
        ----------
        initial_fields : dict
            Initial potential fields, {monomer_type: field_array}.
            Each field_array should have shape matching nx (can be flattened or
            shaped). Typical initialization: small random perturbations around zero.
        q_init : dict, optional
            Initial propagator values for grafted polymers (default: None).
            Format: {propagator_key: value_array}. Full-grid arrays follow the
            input axis_ordering. Arrays in the solver's active reduced basis are
            also accepted; they already use standard axes and are not transposed.
            Both representations must have the full space-group symmetry when a
            group is used.
        on_iteration : callable, optional
            Callback function called after each iteration.
            Signature: on_iteration(info: IterationInfo) -> Optional[bool]
            If callback returns False or another false value, such as the
            NumPy False of a comparison, iteration stops early; None continues.
            Use for monitoring, checkpointing, or custom stopping criteria.
        return_result : bool, optional
            If True, return an SCFTResult object with the fields on the full
            grid (default: False). The results are also stored as attributes.
        initial_psi : ndarray, optional
            Initial electrostatic potential ψ, one real value per grid point (only with
            `electrostatics`); by default the uniform ψ₀ of the homogeneous state (zero for a
            neutral canonical system without screening).

        Attributes Set
        --------------
        phi : dict
            Final monomer concentration fields {monomer_type: array}.
        w : ndarray
            Final potential fields, shape (M, total_grid).
        free_energy : float
            Free energy relative to the homogeneous state at the same composition.
        free_energy_absolute : float
            Full canonical free energy, beta F_total / (rho_0 V / N).
        error_level : float
            Final convergence error (L2 norm of field residuals).
        iter : int
            Number of iterations performed.

        Raises
        ------
        ValidationError
            Invalid initial_fields, q_init or initial_psi, before the run changes any state.
        NumericalError
            A non-finite residual or error level, a failed partition consistency check, or a final
            partition function that is not positive and finite; its `fields` are the fields of
            that iteration.
        """

        # ψ of SCFT is real (the mean field of the Coulomb interaction)
        if initial_psi is not None:
            if self.electrostatics is None:
                raise ValidationError("initial_psi needs 'electrostatics'")
            initial_psi = finite_values(initial_psi, "initial_psi")
            if initial_psi.size != self.prop_solver.n_grid:
                raise ValidationError("initial_psi must hold one real, finite value per grid point")

        # Reject invalid input before resetting the optimizer or changing the run state: the initial fields
        # in the working basis of the grid with the field means of the saddle point, and the propagators
        # that grafted chains start from
        initial_fields = initial_field_values(initial_fields, self.monomer_types, self.prop_solver.total_grid)
        w = self.grid.initial_fields(initial_fields)
        q_init = self.grid.initial_propagators(q_init)
        self._pin_field_means(w)

        # Assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # Reset optimizer
        self.field_optimizer.reset_count()

        print("---------- Run ----------")
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        print(", box size" if self.relax_cell else "")

        # ---- 1. Initial fields and ψ (the Poisson equation, electrostatics.py) ----
        es = self.electrostatics
        energy_electrostatic_homogeneous = 0.0 if es is None else es.start(initial_psi, q_init)

        # Check the partition consistency on the first field of this run
        self._partition_checked = False

        # ---- 2. Iteration ----
        self._homogeneous_state(energy_electrostatic_homogeneous)

        time_iter_start = time.time()
        for iter in range(1, self.max_iter+1):

            # Concentrations for the current fields, and with relax_cell the stress of the propagators
            phi, runtime_info = self.compute_concentrations(w, q_init=q_init, stress=self.relax_cell)

            # Deviation of the total density from 1
            phi_total = sum(phi[m] for m in self.monomer_types)
            phi_diff = phi_total - 1.0

            # Concentrations the fields couple to (smeared with smearing), phi_array[i] =
            # phi[monomer_types[i]], shape (M, n_grid)
            phi_smeared = self.smearing.apply_to_dict(phi)
            phi_array = np.array([phi_smeared[m] for m in self.monomer_types])

            # Free energy: the reported one is relative to the homogeneous state; thermodynamic
            # transforms and comparisons across compositions use the full F
            total_partitions = partition_functions(self)
            energy_absolute = self._free_energy(w, phi_array, total_partitions)
            energy_total = energy_absolute - self.free_energy_homogeneous

            # Residuals of the fields and of the Poisson equation for ψ, and their size
            w_diff = self._field_residual(w, phi_array)
            psi_diff = None if es is None else es.residual(es.charge_density(phi_smeared))
            if not np.all(np.isfinite(w_diff)) or (psi_diff is not None and not np.all(np.isfinite(psi_diff))):
                raise NumericalError(f"Non-finite SCFT residual at iteration {iter}", self.grid.full_grid_fields(w))
            old_error_level = error_level
            field_error_level = self._error_level(w, w_diff, psi_diff)
            mass_error = self.grid.mean(phi_diff)

            # With relax_cell, the stress along the lattice parameters that relax counts towards the error
            error_level = field_error_level + self.cell.stress_error(w, phi)
            if not np.isfinite(error_level):
                raise NumericalError(f"Non-finite SCFT error level at iteration {iter}", self.grid.full_grid_fields(w))

            # Print this iteration
            self._print_iteration(iter, mass_error, total_partitions, energy_total, error_level)

            # Call progress callback if provided
            if on_iteration is not None:
                keep_going = on_iteration(iteration_info(self, iter, error_level, energy_total, mass_error, total_partitions))
                if keep_going is not None and not keep_going:
                    print(f"Iteration stopped by callback at iteration {iter}")
                    break

            # Conditions to end the iteration
            if error_level < self.tolerance or iter == self.max_iter:
                break

            # New fields, ψ and cell
            w = self._next_fields(w, w_diff, psi_diff, old_error_level, error_level)

        self.cell.check_tied_lengths(error_level)

        # The propagators of the last evaluated fields: Q must be positive for log Q in the free energy
        # (some discretizations have non-positive partitions at intermediate iterates, which may recover)
        check_partition_functions(self.prop_solver, f"the end of the SCFT iteration ({iter} iterations)",
                                  lambda: self.grid.full_grid_fields(w))

        # Print iteration timing
        time_iter_total = time.time() - time_iter_start
        print("Iteration time: %.3f seconds (%.4f sec/iter, %d iterations)" % (time_iter_total, time_iter_total/iter, iter))

        print_free_energies(self, energy_total, energy_absolute)

        # Final partition consistency check (only once at the end)
        if not report_partition_consistency(self.prop_solver):
            print("Warning: Final partition consistency check failed.")

        # Keep the final state
        self.phi, self.w = phi, w
        self.free_energy, self.free_energy_absolute = energy_total, energy_absolute
        self.error_level, self.iter = error_level, iter
        self.partition_functions = list(total_partitions)

        # The chemical potentials of the point ions and of the chains, and the grand potential (scft_results.py)
        self.ion_chemical_potentials = ion_chemical_potentials(self, phi)
        self.chemical_potentials, self.grand_potential = chemical_potentials(self, q_init)

        # Return SCFTResult if requested
        return scft_result(self) if return_result else None

    def _homogeneous_state(self, energy_electrostatic_homogeneous):
        """The chain densities and the free energy of the homogeneous state."""
        # Ideal mixing term is constant during a CE run. Define c log(c) = 0 at c = 0.
        self.polymer_lengths = [self.prop_solver.get_polymer(p).get_alpha()
                                for p in range(self.prop_solver.get_n_polymer_types())]
        self.polymer_volume_fractions = [self.prop_solver.get_polymer(p).get_volume_fraction()
                                         for p in range(self.prop_solver.get_n_polymer_types())]
        self.chain_densities = np.array(self.polymer_volume_fractions) / self.polymer_lengths
        present = self.chain_densities > 0.0
        self.ideal_energy = float(np.sum(self.chain_densities[present] * (np.log(self.chain_densities[present]) - 1.0)))
        phi_mean = self.mean_concentrations()
        self.free_energy_homogeneous = self.ideal_energy + 0.5*float(phi_mean @ self.matrix_chi @ phi_mean) \
            + energy_electrostatic_homogeneous

    def _free_energy(self, w, phi_array, total_partitions):
        """Full canonical free energy of the fields."""
        M, es = len(self.monomer_types), self.electrostatics
        # Full CE free energy: Σ_p c_p[ln(c_p/Q_p)-1] + <½ φᵀ χN φ - w·φ>
        # (+ compressibility penalty). Uniform field shifts cancel between ln Q and w·φ.
        # At the saddle point F - self.ideal_energy is the Hamiltonian recorded by L-FTS.
        energy_total = self.ideal_energy
        for p, concentration in enumerate(self.chain_densities):
            if concentration > 0.0:
                energy_total -= concentration*np.log(total_partitions[p])
        for i in range(M):
            for j in range(i+1, M):
                if self.matrix_chi[i, j] != 0.0:
                    energy_total += self.matrix_chi[i, j]*self.grid.mean(phi_array[i]*phi_array[j])
            energy_total -= self.grid.mean(w[i]*phi_array[i])
        if self.zeta_n is not None:
            energy_total += 0.5*self.zeta_n*self.grid.mean((np.sum(phi_array, axis=0) - 1.0)**2)
        if es is not None:
            energy_total += es.free_energy()
        return energy_total

    def _field_residual(self, w, phi_array):
        """Residual of the SCFT equations for the fields, with zero means."""
        M = len(self.monomer_types)
        # The SCFT equations are w = χφ + ξe with Σφ = 1 (incompressible), or w = χφ + ζ(Σφ - 1)e
        # (compressible), e = (1, ..., 1). Composition part: P⊥(χφ - w), the residual of the
        # exchange fields of the symmetric theory. Pressure part along e: κ(Σφ - 1), which raises
        # the pressure where the density is too high; κ = max(max|λ|, 1)/2 follows the coupling
        # of the composition and the pressure (for AB κ = χN/2). Compressible: the stiff ζ term
        # lies along e, so that part of E = χφ + ζ(Σφ - 1)e - w is scaled by κ/(κ + ζ), which
        # gives the incompressible residual for ζ → ∞ and leaves E unscaled for ζ → 0
        e_column = np.ones((M, 1))
        if self.zeta_n is None:
            w_diff = self.matrix_q_perp @ (self.matrix_chi @ phi_array - w) \
                + self.pressure_kappa * e_column * (np.sum(phi_array, axis=0) - 1.0)[None, :]
        else:
            residual = self.matrix_chi @ phi_array + self.zeta_n * (np.sum(phi_array, axis=0) - 1.0) - w
            composition = self.matrix_q_perp @ residual
            w_diff = composition + (self.pressure_kappa / (self.pressure_kappa + self.zeta_n)) * (residual - composition)

        # Keep the level of functional derivatives
        for i in range(M):
            w_diff[i] -= self.grid.mean(w_diff[i])
        return w_diff

    def _error_level(self, w, w_diff, psi_diff):
        """Relative distance between the input and output fields,
        sqrt( sum_i <w_diff_i^2> / (1 + sum_i <w_i^2>) ), ψ included."""
        M, es = len(self.monomer_types), self.electrostatics
        error_level = 0.0
        error_normal = 1.0  # add 1.0 to prevent divergence
        for i in range(M):
            error_level += self.grid.inner_product(w_diff[i], w_diff[i])
            error_normal += self.grid.inner_product(w[i], w[i])
        if es is not None:
            # The uniform ψ₀ (Donnan potential) shifts with the chemical potentials of the
            # ions, so the scale of ψ is that of its variation
            psi_variation = es.psi - self.grid.mean(es.psi)
            error_level += self.grid.inner_product(psi_diff, psi_diff)
            error_normal += self.grid.inner_product(psi_variation, psi_variation)
        return np.sqrt(error_level/error_normal)

    def _print_iteration(self, iter, mass_error, total_partitions, energy_total, error_level):
        print("%8d %12.3E " % (iter, mass_error), end=" [ ")
        for Q in total_partitions:
            print("%13.7E " % (Q), end=" ")
        if self.relax_cell:
            print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
            print("lx=[", ",".join(["%10.7f" % (x) for x in self.prop_solver.get_lx()]), "]", end="")
            if len(self.cell.angles_reduced_indices) > 0:
                print(" angles=[", ",".join(["%7.2f" % (x) for x in self.prop_solver.get_angles_degrees()]), "]", end="")
            print()
        else:
            print("] %15.9f %15.7E " % (energy_total, error_level))

    def _next_fields(self, w, w_diff, psi_diff, old_error_level, error_level):
        """Simple and Anderson mixing of one vector: the fields (in the working basis of the grid), ψ and,
        with relax_cell, the lattice parameters that relax (cell_relaxation.py)."""
        M, es = len(self.monomer_types), self.electrostatics
        psi, psi_diff = (es.psi, psi_diff) if es is not None else (np.array([]), np.array([]))
        current = np.concatenate((w.flatten(), psi, self.cell.parameters()))
        residual = np.concatenate((w_diff.flatten(), psi_diff, self.cell.residual()))
        new = self.field_optimizer.calculate_new_fields(current, residual, old_error_level, error_level)

        # The new cell, fields and ψ
        self.cell.apply(new[w.size + psi.size:])
        w = np.reshape(new[:w.size], (M, -1))
        if es is not None:
            psi = np.array(new[w.size:w.size + psi.size])
            if es.zero_uniform_mode:
                psi -= np.mean(psi)
            es.psi = psi

        # Normalize field means
        self._pin_field_means(w)
        return w
