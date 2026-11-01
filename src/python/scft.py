import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Union

from . import _core
from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .validation import (normalize_params, validate_scft_params, ValidationError,
                         point_ion_concentrations, open_charges, reference_segments)
from .utils import (
    report_partition_consistency,
    display_system,
    check_partition_functions,
)
from .cell_relaxation import CellRelaxation
from .scft_results import SCFTResult, IterationInfo, scft_result, iteration_info, print_free_energies, save_results
from .chains import read_chains, mean_concentrations, draw_polymer_architectures, chain_fields, solve_chains, check_partition_once
from .file_formats import load_config
from .smearing import smearing_of
from .electrostatics import MeanFieldElectrostatics

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")

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
    @staticmethod
    def _apply_axis_ordering(params):
        """Reorder nx/lx/bc/angles in ``params`` (in place) from the user's axis ordering
        to the standard [a, b, c]. Returns (axis_ordering, perm); both None when the
        option is absent. perm[i] is the position of standard axis i in the user's arrays.
        """
        axis_ordering = params.get("axis_ordering", None)
        if axis_ordering is None:
            return None, None
        dim = len(params["nx"])
        labels = ["a", "b", "c"][:dim]
        if len(axis_ordering) != dim:
            raise ValueError(f"axis_ordering length ({len(axis_ordering)}) must match dimension ({dim})")
        if sorted(axis_ordering) != labels:
            raise ValueError(f"axis_ordering must contain exactly {labels}, got {axis_ordering}")
        perm = [axis_ordering.index(label) for label in labels]
        params["nx"] = [params["nx"][perm[i]] for i in range(dim)]
        params["lx"] = [params["lx"][perm[i]] for i in range(dim)]
        bc = params.get("bc", None)
        if bc is not None and len(bc) == 2 * dim:
            # bc = [low, high] of each axis in the order of nx
            params["bc"] = [bc[2 * perm[i] + side] for i in range(dim) for side in (0, 1)]
        angles = params.get("angles", None)
        if angles is not None and dim == 3:
            # angles = [alpha, beta, gamma] are the angles between axes (b,c), (a,c), (a,b);
            # the new angle between axes i and j is the old angle between perm[i] and perm[j]
            angle_index = {(1, 2): 0, (0, 2): 1, (0, 1): 2}
            params["angles"] = [angles[angle_index[tuple(sorted((perm[i], perm[j])))]]
                                for (i, j) in ((1, 2), (0, 2), (0, 1))]
        print(f"Axis ordering: {axis_ordering} -> reordered to [a, b, c]")
        print(f"  nx: {params['nx']}, lx: {params['lx']}")
        if params.get("bc", None) is not None:
            print(f"  bc: {params['bc']}")
        if params.get("angles", None) is not None:
            print(f"  angles: {params['angles']}")
        return axis_ordering, perm

    def __init__(self, params):

        # Validate input parameters
        params = normalize_params(params)
        validate_scft_params(params)

        # Handle axis ordering (convert user ordering to standard [a, b, c]); this
        # rewrites nx/lx/bc/angles in `params`, so the key is dropped from the copy that
        # save_results() writes as "initial_params" (it must not be applied twice)
        self.axis_ordering, self._axis_perm = self._apply_axis_ordering(params)
        params.pop("axis_ordering", None)

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)
        if self.zeta_n is not None and self.zeta_n < 0:
            raise ValidationError(f"zeta_n must not be negative, got {self.zeta_n}.")

        # Monomer types, segment lengths, polymer chains, χN and random copolymers
        self.monomer_types, self.segment_lengths, self.polymers, self.chi_n, self.random_fraction = read_chains(params)

        # Draw the chain architectures (polymer_<id>.png) only into a directory the input names,
        # so that constructing a calculation writes no files
        if params.get("polymer_figure_dir") is not None:
            draw_polymer_architectures(self.polymers, params["ds"], self.segment_lengths,
                                       directory=params["polymer_figure_dir"])

        # Boundary conditions (default: periodic) and lattice angles [alpha, beta, gamma] in degrees
        bc = params.get("bc", None)
        if bc is None:
            bc = ["periodic"] * (2 * len(params["nx"]))
        angles = params.get("angles", None)
        if angles is not None and len(params["nx"]) == 3 and len(angles) != 3:
            raise ValidationError("angles must be [alpha, beta, gamma] (three entries) for a 3D box.")
        if angles is not None and len(params["nx"]) < 3:
            dim = len(params["nx"])
            if len(angles) not in (0, 1, 3):
                raise ValidationError("angles must have 0, 1 or 3 entries for a 1D or 2D box.")
            fixed = angles if dim == 1 else (angles[:2] if len(angles) == 3 else [])
            if not np.all(np.isclose(fixed, 90.0, rtol=0.0, atol=1e-10)):
                raise ValidationError("A 1D box requires all angles to be 90 degrees; "
                                      "a 2D box requires alpha and beta to be 90 degrees.")

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

        # Create space group object early (before PropagatorSolver) if specified
        self.sg = self._create_space_group(params, angles) if params.get("space_group", None) is not None else None

        # Propagator solver: the C++ computation box, molecules and solver live inside it
        self.prop_solver = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", None),   # None: "rqm4"
            platform=params.get("platform", "auto"),   # auto: CPU for 1D, CUDA for 2D/3D when available
            reduce_memory=params.get("reduce_memory", False),
            space_group=self.sg,
            angles=angles,
            aggregate_propagator_computation=params.get("aggregate_propagator_computation", True),
        )
        # Box relaxation needs the stress, which the real-space method (cn-adi2) does not compute
        if (params["relax_cell"]
                and PropagatorSolver.numerical_methods()[self.prop_solver.numerical_method]["solver_type"] != "pseudospectral"):
            raise ValidationError(f"'relax_cell' needs a pseudo-spectral method; '{self.prop_solver.numerical_method}' "
                                  "computes no stress")
        for polymer in self.polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))

        # Computation box as a public attribute (this creates the C++ objects)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Initialize smearing (must be after self.cb is set)
        if params.get("smearing", None) is not None and self.sg is not None:
            # Smearing operates on full-grid fields, but with a space group the
            # fields live in the reduced basis; Smearing.apply would reshape a
            # reduced-basis array to the full grid and crash (or corrupt data).
            raise NotImplementedError(
                "'smearing' is not supported together with 'space_group'. "
                "Disable one of the two options.")
        self.smearing = smearing_of(params, self.prop_solver)

        # Electrostatics: the charged types feel z_i ψ, and ψ solves the Poisson equation with the
        # charges of the chains and the point ions (electrostatics.py, MeanFieldElectrostatics)
        self.electrostatics = None
        self.psi = None
        if params.get("electrostatics") is not None:
            self._setup_electrostatics(params)

        # The lattice parameters that vary with relax_cell, and the size of the optimizer
        self.cell = CellRelaxation(self, params, angles)
        n_var = self.cell.n_variables

        # Anderson mixing (C++ class) for the saddle-point iteration
        optimizer = params["optimizer"]
        if optimizer["name"] != "am":
            raise ValueError(f"Invalid optimizer name: {optimizer['name']}. Only 'am' (Anderson mixing) is available.")
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

        # Stress computation interval: an integer (every N iterations) or
        # "adaptive" (more often while the error is high, see run()).
        self.stress_interval = params.get("stress_interval", 1)

        self.max_iter = params.get("max_iter", 2000)       # maximum number of iterations
        if self.max_iter < 1:
            raise ValidationError(f"'max_iter' must be at least 1, got {self.max_iter}")
        self.tolerance = params.get("tolerance", 1e-8)     # stop when the self-consistency error is below this

        # The fields of an incompressible run have zero means (the free energy, computed from
        # the monomer fields, does not depend on them); fts_consistent_energy, which pinned
        # them to the saddle point, has no effect any more
        if "fts_consistent_energy" in params:
            print("Warning: 'fts_consistent_energy' has no effect: the free energy does not depend on the "
                  "field means, and incompressible fields have zero means.")

    def _create_space_group(self, params, angles):
        """The space group of the input, with the physical basis that the cell allows."""
        if params["space_group"].get("hall_number") is not None:
            sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"], params["space_group"]["hall_number"])
        else:
            sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"])
        # Auto-enable physical basis when possible (m3 → pmmm → z-mirror).
        # NOTE: reuse the (possibly axis_ordering-reordered) local `angles`;
        # missing angles means an orthogonal cell for classification, while
        # `angles` itself stays None so the computation box is not
        # needlessly recreated below.
        try:
            alpha, beta, gamma = angles if angles is not None else (90.0, 90.0, 90.0)
        except Exception:
            alpha, beta, gamma = 90.0, 90.0, 90.0
        # The test of the native box and solvers (1e-10 rad): a looser one could enable a
        # basis whose CrysFFT the solver then refuses
        def right_angle(angle):
            return abs(angle * (np.pi / 180.0) - np.pi / 2.0) < 1e-10
        is_orthogonal = right_angle(alpha) and right_angle(beta) and right_angle(gamma)
        z_axis_orthogonal = right_angle(alpha) and right_angle(beta)

        if is_orthogonal:
            # Fall back down the basis hierarchy when a basis is not
            # applicable to this group/grid (the C++ side throws
            # RuntimeError). Other exception types indicate real bugs
            # and must propagate.
            # (the m3 basis needs the recursive 3m CrysFFT: even grid, nz >= 16
            # and a compatible translation parity)
            try:
                if not sg.m3_crysfft_feasible():
                    raise RuntimeError("recursive 3m CrysFFT is not feasible on this grid")
                sg.enable_m3_physical_basis()
            except RuntimeError:
                try:
                    sg.enable_pmmm_physical_basis()
                except RuntimeError:
                    pass  # keep the irreducible (full-group) basis
        elif z_axis_orthogonal:
            # Hexagonal/trigonal (hexagonal-axes) space groups use a hybrid
            # orbit convention (node-centered x,y; cell-centered z), making
            # all in-plane symmetry maps exact on the grid. The z-mirror
            # physical basis (cell-centered z fold) remains valid.
            try:
                sg.enable_z_mirror_physical_basis()
            except RuntimeError:
                pass  # keep the irreducible (full-group) basis

        # The cell must belong to the crystal system of the space group. Otherwise
        # the symmetry operations are not isometries of the box and the run would
        # silently converge to a constrained, unphysical state. The native check
        # (SpaceGroup::check_cell) is the one the computation box applies later.
        lx_sg = [float(v) for v in params["lx"]]
        angles_sg = [float(v) for v in angles] if angles is not None else [90.0, 90.0, 90.0]
        try:
            sg.check_cell(lx_sg, angles_sg)
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from None
        if (sg.get_crystal_system() == "Trigonal" and not sg.uses_hexagonal_axes()
                and params.get("relax_cell", False)):
            # a = b = c and alpha = beta = gamma would have to vary together
            raise ValidationError("Box relaxation is not supported for rhombohedral-axes settings; "
                                  "use the hexagonal-axes setting of the space group.")
        sg.display_info()
        return sg

    def _setup_electrostatics(self, params):
        """The Poisson equation for ψ and what it needs of the input."""
        # A fixed charge is given in the axis order of nx, like the initial fields
        if self._axis_perm is not None and params["electrostatics"].get("fixed_charge") is not None:
            params["electrostatics"]["fixed_charge"] = self._to_standard_axes(params["electrostatics"]["fixed_charge"])
        self.electrostatics = MeanFieldElectrostatics(params["electrostatics"], self.monomer_types, reference_segments(params),
                                                      self.prop_solver.fourier,
                                                      ion_concentrations=point_ion_concentrations(params),
                                                      open_charges=open_charges(params))
        es = self.electrostatics
        self.psi = np.zeros(self.prop_solver.n_grid)
        # Box relaxation with electrostatics is limited to periodic boxes (CellRelaxation.stress also
        # holds in closed boxes)
        if params["relax_cell"] and not self.prop_solver.fourier.periodic:
            raise ValidationError("'relax_cell' with 'electrostatics' needs a periodic box")
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
        params = normalize_params(load_config(path))
        params.update(normalize_params(overrides))
        return cls(params)

    def compute_concentrations(self, w, q_init=None):
        """Compute monomer concentration fields for given potential fields.

        NOTE: unlike LFTS/CLFTS.compute_concentrations (which take AUXILIARY
        fields), this method takes MONOMER potential fields directly.

        Solves the modified diffusion equations to compute chain propagators,
        then integrates over chain contours to obtain monomer concentration
        distributions. Handles both block copolymers and random copolymers.
        q_init optionally supplies initial propagator values for grafted
        polymers (passed through to PropagatorSolver.compute_propagators).

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
        # Field optimizer operates on irreducible grid when space group is set.
        if self.sg is not None:
            w = self.sg.irreducible_to_reduced(w)

        # The fields of the chains: charged types feel w_i + z_i ψ with the current ψ (self.psi)
        w_input = chain_fields(self, w)
        if self.electrostatics is not None:
            self.electrostatics.add_potential(w_input, self.psi)

        phi, elapsed_time = solve_chains(self, w_input, q_init)
        check_partition_once(self)

        # Convert concentrations back to irreducible grid for optimizer.
        if self.sg is not None:
            phi = {name: self.sg.reduced_to_irreducible(phi[name]) for name in phi}

        return phi, elapsed_time

    def _to_standard_axes(self, field):
        """Transpose a full-grid array given in the user's axis order (axis_ordering) to [a, b, c]."""
        user_nx = [0] * len(self._axis_perm)
        for i, axis in enumerate(self._axis_perm):
            user_nx[axis] = self.prop_solver.nx[i]
        return np.ravel(np.transpose(np.reshape(np.asarray(field), user_nx), self._axis_perm))

    def _to_full_grid(self, field: np.ndarray) -> np.ndarray:
        """Expand a field of the working basis to the full grid (a copy without a space group)."""
        if self.sg is None:
            return np.array(field, dtype=np.float64)
        # self.w and self.phi are stored in the irreducible basis. With a physical
        # basis the reduced ordering differs even when the sizes coincide (mmm
        # orthorhombic groups with the m3 basis), so the conversion is gated on
        # the basis, not on the sizes.
        if self.sg.physical_basis_name() != "irreducible":
            field = self.sg.irreducible_to_reduced(field)
        return self.sg.from_reduced_basis(field)

    # =========================================================================
    # run() helpers (used more than once, or long enough to stand on their own)
    # =========================================================================

    def save_results(self, path):
        """Saves the results to a MATLAB, JSON or YAML file, by the extension of path (scft_results.py)."""
        save_results(self, path)

    def mean_concentrations(self):
        """Volume-averaged concentration of each monomer type from the chain compositions."""
        return mean_concentrations(self)

    def _pin_field_means(self, w, chi_phi_mean, xi=0.0):
        """Normalize the field means in place (mean() works in the reduced basis
        when a space group is set).

        Compressible systems have no gauge freedom: the means are always pinned to the
        saddle point, mean(w_i) = sum_j chi_ij mean(phi_j) + xi with
        xi = zeta (mean(sum_j phi_j) - 1) (the residual leaves the means out).
        Incompressible fields are shifted to zero mean (the uniform modes do not change the
        concentrations, and the free energy does not depend on them).
        """
        M = len(self.monomer_types)
        if self.zeta_n is not None:
            for i in range(M):
                w[i] += chi_phi_mean[i] + xi - self.prop_solver.mean(w[i], basis="irreducible")
        else:
            for i in range(M):
                w[i] -= self.prop_solver.mean(w[i], basis="irreducible")

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
                 else self.prop_solver.mean(field, basis="irreducible") for field in w]
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
                    field_shift += block.contour_length * fraction * shift[self.monomer_types.index(monomer)]
            mu.append(float(np.log(concentration) - np.log(partition_functions[p]) + field_shift))
        return mu

    def _polymer_charge(self):
        """Net charge Σ_i z_i <φ_i> of the polymers and the mean fixed charge, per segment."""
        es = self.electrostatics
        fixed = 0.0 if es.fixed_charge is None else float(np.mean(es.fixed_charge))
        return fixed + float(sum(es.charges[m] * phi for m, phi in zip(self.monomer_types, self.mean_concentrations())))

    def _charge_response(self, psi0, q_init=None):
        """Response D(k) = -∂c(k)/∂ψ(k) of the charges in the homogeneous state with the uniform ψ₀.

        The uniform fields do not couple different modes of the transform (Fourier modes, or the
        cosine modes of a box with walls), so central differences of the charge density for a
        kick of ψ at one grid point, divided by the transform of the kick (nonzero at every mode),
        give D(k) at all k from two computations of the propagators. The uniform fields w = 0
        suffice: a uniform field does not change the concentrations of canonical chains.
        Grafted chains (q_init) are not homogeneous; D(k) then only approximates the response,
        which is enough for the preconditioner of the residual (its zero is the Poisson equation),
        and a negative estimate is set to zero.
        """
        es, n_grid = self.electrostatics, self.prop_solver.n_grid
        w = np.zeros((len(self.monomer_types), n_grid))
        epsilon = 1e-3
        charges = []
        for sign in (1.0, -1.0):
            self.psi = np.full(n_grid, psi0)
            self.psi[0] += sign * epsilon
            phi, _ = self.compute_concentrations(w, q_init=q_init)
            charges.append(es.charge_density(self.smearing.apply_to_dict(phi), self.psi))
        kick = np.zeros(n_grid)
        kick[0] = 2.0 * epsilon
        fourier = self.prop_solver.fourier
        response = -np.real(fourier.forward(charges[0] - charges[1]) / fourier.forward(kick))
        return response if q_init is None else np.maximum(response, 0.0)

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
            Format: {propagator_key: value_array}. Rarely used; for advanced
            applications like brush layers.
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
        """

        M = len(self.monomer_types)

        # ψ of SCFT is real (the mean field of the Coulomb interaction)
        if initial_psi is not None:
            if self.electrostatics is None:
                raise ValidationError("initial_psi needs 'electrostatics'")
            if np.size(initial_psi) != self.prop_solver.n_grid or np.iscomplexobj(initial_psi) \
                    or not np.all(np.isfinite(initial_psi)):
                raise ValidationError("initial_psi must hold one real, finite value per grid point")

        # Assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # Reset optimizer
        self.field_optimizer.reset_count()

        print("---------- Run ----------")
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        print(", box size" if self.relax_cell else "")

        # ---- 1. Initial fields and ψ ----
        w, q_init = self._initial_fields(initial_fields, q_init)
        es = self.electrostatics
        energy_electrostatic_homogeneous = 0.0
        if es is not None:
            energy_electrostatic_homogeneous = self._start_psi(initial_psi, q_init)

        # Check the partition consistency on the first field of this run
        self._partition_checked = False

        # ---- 2. Iteration ----
        stress_array = None
        self._homogeneous_state(energy_electrostatic_homogeneous)

        time_iter_start = time.time()
        for iter in range(1, self.max_iter+1):

            # Concentrations for the current fields
            phi, runtime_info = self.compute_concentrations(w, q_init=q_init)

            # Deviation of the total density from 1
            phi_total = sum(phi[m] for m in self.monomer_types)
            phi_diff = phi_total - 1.0

            # Concentrations the fields couple to (smeared with smearing), phi_array[i] =
            # phi[monomer_types[i]], shape (M, n_grid)
            phi_smeared = self.smearing.apply_to_dict(phi)
            phi_array = np.array([phi_smeared[m] for m in self.monomer_types])

            # Free energy: the reported one is relative to the homogeneous state; thermodynamic
            # transforms and comparisons across compositions use the full F
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            energy_absolute = self._free_energy(w, phi_array, total_partitions)
            energy_total = energy_absolute - self.free_energy_homogeneous

            # Residuals of the fields and of the Poisson equation for ψ, and their size
            w_diff = self._field_residual(w, phi_array)
            psi_diff = None if es is None else es.residual(es.charge_density(phi_smeared, self.psi), self.psi)
            old_error_level = error_level
            error_level = self._error_level(w, w_diff, psi_diff)
            field_error_level = error_level
            mass_error = self.prop_solver.mean(phi_diff, basis="irreducible")

            if self.relax_cell:
                # The stress costs ~30 % of an iteration: compute it every stress_interval
                # iterations ("adaptive": every 1, 2 or 4 as the error drops) and reuse it in between
                if self.stress_interval == "adaptive":
                    stress_interval = 1 if error_level > 1e-2 else (2 if error_level > 1e-4 else 4)
                else:
                    stress_interval = self.stress_interval
                stress_is_current = (iter == 1) or (iter % stress_interval == 0) or iter == self.max_iter
                if stress_is_current:
                    stress_array = self.cell.stress(w, phi)
                # The stress along the optimized lattice parameters counts towards the error
                error_level += self.cell.error(stress_array)
                # A cached stress can guide updates, but cannot certify convergence.
                if not stress_is_current and error_level < self.tolerance:
                    stress_array = self.cell.stress(w, phi)
                    stress_is_current = True
                    error_level = field_error_level + self.cell.error(stress_array)

            # Print this iteration
            self._print_iteration(iter, mass_error, total_partitions, energy_total, error_level)

            # Call progress callback if provided
            if on_iteration is not None:
                keep_going = on_iteration(iteration_info(self, iter, error_level, energy_total, mass_error, total_partitions))
                if keep_going is not None and not keep_going:
                    if self.relax_cell and not stress_is_current:
                        stress_array = self.cell.stress(w, phi)
                        error_level = field_error_level + self.cell.error(stress_array)
                    print(f"Iteration stopped by callback at iteration {iter}")
                    break

            # Conditions to end the iteration
            if error_level < self.tolerance or iter == self.max_iter:
                break

            # New fields, ψ and cell
            w = self._next_fields(w, w_diff, psi_diff, stress_array, old_error_level, error_level, phi_array, phi_total)

        self.cell.check_tied_lengths(stress_array, error_level)

        # The propagators of the last evaluated fields: Q must be positive for log Q in the free energy
        check_partition_functions(self.prop_solver, f"the end of the SCFT iteration ({iter} iterations)")

        # Print iteration timing
        time_iter_total = time.time() - time_iter_start
        print("Iteration time: %.3f seconds (%.4f sec/iter, %d iterations)" % (time_iter_total, time_iter_total/iter, iter))

        print_free_energies(self, energy_total, energy_absolute)

        # Final partition consistency check (only once at the end)
        if not report_partition_consistency(self.prop_solver):
            print("Warning: Final partition consistency check failed.")

        # Store phi and w
        self.phi = phi
        self.w = w
        self.ion_chemical_potentials = None
        if es is not None:
            print("Poisson residual |c - (k² + κ²)ψ/E| relative to the charge density: %.3e"
                  % es.poisson_error(self.smearing.apply_to_dict(phi), self.psi))
            if es.ions:
                self.ion_chemical_potentials = es.ion_chemical_potentials(self.psi)
                print("Point ions: chemical potentials βμ", self.ion_chemical_potentials,
                      "ions per reference chain", es.ion_number_densities(self.psi))

        # Store free energy
        self.free_energy = energy_total
        self.free_energy_absolute = energy_absolute
        self.error_level = error_level
        self.iter = iter
        self.partition_functions = list(total_partitions)
        self.chemical_potentials = None
        self.grand_potential = None
        if q_init is None and not any(p.get("initial_conditions") for p in self.polymers):
            self.chemical_potentials = self.compute_chemical_potentials()
            present = self.chain_densities > 0.0
            self.grand_potential = float(energy_absolute - np.dot(self.chain_densities[present],
                                                                 np.array(self.chemical_potentials)[present]))
            # Canonical point ions add -N c_s βμ_s; the term of grand-canonical ions is grand already
            if es is not None:
                for ion, mu in zip(es.ions, self.ion_chemical_potentials or []):
                    if ion["concentration"] is not None and ion["concentration"] > 0:
                        self.grand_potential -= es.n_segments * ion["concentration"] * mu
            print("Chemical potentials βμ (mean ξ = 0 for incompressible systems):", self.chemical_potentials)
            print("Grand potential in the same pressure convention: %.9f" % self.grand_potential)

        # Return SCFTResult if requested
        if return_result:
            return scft_result(self, phi, w, energy_total, error_level, iter, total_partitions, stress_array)
        return None

    def _initial_fields(self, initial_fields, q_init):
        """The initial fields as the working array of shape (M, n_grid), and the initial propagators."""
        M = len(self.monomer_types)

        # With axis_ordering the arrays are given in the axis order of nx and lx and are
        # transposed to the standard [a, b, c] order (the results are in that order)
        if self._axis_perm is not None:
            initial_fields = {m: self._to_standard_axes(f) for m, f in initial_fields.items()}
            if q_init is not None:
                q_init = {label: self._to_standard_axes(f) for label, f in q_init.items()}

        # With a space group the fields are symmetrized and converted to the irreducible basis.
        if self.sg is not None:
            w_full = np.zeros([M, self.prop_solver.total_grid], dtype=np.float64)
            for i in range(M):
                w_full[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.total_grid)
            w_full = self.sg.symmetrize(w_full)
            w = self.sg.to_reduced_basis(w_full)
            w = self.sg.reduced_to_irreducible(w)
        else:
            w = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            for i in range(M):
                w[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.n_grid)

        # Normalize the initial field means. In a compressible model the target is
        # chi N . mean(phi) + xi from the chain compositions (see mean_concentrations)
        if self.zeta_n is not None:
            phi_mean = self.mean_concentrations()
            xi = 0.0 if self.zeta_n is None else self.zeta_n * (np.sum(phi_mean) - 1.0)
            self._pin_field_means(w, self.matrix_chi @ phi_mean, xi)
        else:
            self._pin_field_means(w, None)
        return w, q_init

    def _start_psi(self, initial_psi, q_init):
        """ψ: the Newton residual of the Poisson equation needs the charge response of the homogeneous
        state (MeanFieldElectrostatics), which also gives the initial ψ. Returns the electrostatic
        free energy of the homogeneous state."""
        es = self.electrostatics
        energy_electrostatic_homogeneous = 0.0
        if es is not None:
            es.set_box()
            psi0, energy_electrostatic_homogeneous = es.homogeneous(self._polymer_charge())
            es.set_response(self._charge_response(psi0, q_init))
            if initial_psi is None:
                self.psi = np.full(self.prop_solver.n_grid, psi0)
            else:
                if self._axis_perm is not None:
                    initial_psi = self._to_standard_axes(initial_psi)
                self.psi = np.array(np.reshape(initial_psi, self.prop_solver.n_grid), dtype=np.float64)
                if es.zero_uniform_mode:
                    self.psi -= np.mean(self.psi)
        return energy_electrostatic_homogeneous

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
                    energy_total += self.matrix_chi[i, j]*self.prop_solver.mean(phi_array[i]*phi_array[j], basis="irreducible")
            energy_total -= self.prop_solver.mean(w[i]*phi_array[i], basis="irreducible")
        if self.zeta_n is not None:
            energy_total += 0.5*self.zeta_n*self.prop_solver.mean((np.sum(phi_array, axis=0) - 1.0)**2, basis="irreducible")
        if es is not None:
            energy_total += es.free_energy(self.psi)
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
            w_diff[i] -= self.prop_solver.mean(w_diff[i], basis="irreducible")
        return w_diff

    def _error_level(self, w, w_diff, psi_diff):
        """Relative distance between the input and output fields,
        sqrt( sum_i <w_diff_i^2> / (1 + sum_i <w_i^2>) ), ψ included."""
        M, es = len(self.monomer_types), self.electrostatics
        error_level = 0.0
        error_normal = 1.0  # add 1.0 to prevent divergence
        for i in range(M):
            error_level += self.prop_solver.inner_product(w_diff[i], w_diff[i], basis="irreducible")
            error_normal += self.prop_solver.inner_product(w[i], w[i], basis="irreducible")
        if es is not None:
            # The uniform ψ₀ (Donnan potential) shifts with the chemical potentials of the
            # ions, so the scale of ψ is that of its variation
            psi_variation = self.psi - self.prop_solver.mean(self.psi, basis="irreducible")
            error_level += self.prop_solver.inner_product(psi_diff, psi_diff, basis="irreducible")
            error_normal += self.prop_solver.inner_product(psi_variation, psi_variation, basis="irreducible")
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

    def _next_fields(self, w, w_diff, psi_diff, stress_array, old_error_level, error_level, phi_array, phi_total):
        """Simple and Anderson mixing of the fields (irreducible basis with a space group), ψ and the cell."""
        M, es = len(self.monomer_types), self.electrostatics
        # Calculate new fields using simple and Anderson mixing, ψ after the fields
        # Note: w and w_diff are in irreducible basis when space group is set
        fields, fields_diff = w.flatten(), w_diff.flatten()
        if es is not None:
            fields, fields_diff = np.concatenate((fields, self.psi)), np.concatenate((fields_diff, psi_diff))
        if self.relax_cell:
            fields = self.cell.update(fields, fields_diff, stress_array, old_error_level, error_level)
        else:
            fields = self.field_optimizer.calculate_new_fields(fields, fields_diff, old_error_level, error_level)
        w = np.reshape(fields[:w.size], (M, -1))
        if es is not None:
            self.psi = np.array(fields[w.size:])
            if es.zero_uniform_mode:
                self.psi -= np.mean(self.psi)

        # Normalize field means
        if self.zeta_n is not None:
            # mean(φ_i) = f_i exactly by chain normalization, so the target is constant
            chi_phi_mean = np.array([self.prop_solver.mean((self.matrix_chi @ phi_array)[i], basis="irreducible") for i in range(M)])
            xi = 0.0 if self.zeta_n is None else self.zeta_n * (self.prop_solver.mean(phi_total, basis="irreducible") - 1.0)
            self._pin_field_means(w, chi_phi_mean, xi)
        else:
            self._pin_field_means(w, None)
        return w
