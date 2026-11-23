"""The grid of the SCFT fields: the axis order of the input, the space group and the bases of the fields.

SCFT keeps a field in one of three representations. The input gives full-grid arrays in the axis order
of nx (axis_ordering), which become arrays in the standard order [a, b, c]. With a space group the
Anderson mixing iterates the fields in the irreducible basis of the group (the working basis; without a
group it is the whole grid), and the propagator solver takes them in its active basis: a physical basis
(m3, Pmmm or z-mirror) where the group and the grid allow one, else the irreducible basis itself. The
results go back to the full grid. See docs/theory/SpaceGroup.md.
"""

import weakref

import numpy as np

from .. import _core
from ..utils.validation import ValidationError, finite_values


class FieldGrid:
    """The axis order of the input and the space group of an SCFT calculation, and the bases of its fields.

    Built before the propagator solver, which works in the basis of the space group: the constructor
    reorders nx, lx, bc and angles of `params` to [a, b, c] and creates the space group with its basis.
    The solver is then reached through self.sim. initial_fields() and initial_propagators() take the
    arrays of SCFT.run, to_solver_basis() and to_working_basis() convert between the optimizer and the
    solver, mean() and inner_product() work in the working basis, and to_full_grid() expands a field.

    Attributes
    ----------
    axis_ordering : list or None
        The axis order of the input (None: [a, b, c]).
    sg : SpaceGroup or None
        The space group, with its basis enabled.
    n_irreducible : int
        Points of the working basis: of the irreducible basis of the group, or of the grid.
    """

    def __init__(self, sim, params):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once

        # Reorder the arrays of the input to the standard [a, b, c] (validate_scft_params has checked
        # axis_ordering); the key is dropped from `params`, whose copy save_results() writes as
        # "initial_params", so that it is not applied twice
        self.axis_ordering, self._axis_perm = self._apply_axis_ordering(params)
        params.pop("axis_ordering", None)
        self.nx = list(params["nx"])

        # The space group with its basis, if the input gives one
        self.sg = self._create_space_group(params) if params.get("space_group", None) is not None else None
        self.n_irreducible = int(np.prod(self.nx)) if self.sg is None else self.sg.get_n_reduced_basis_full()

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

    def _create_space_group(self, params):
        """The space group of the input, with the physical basis that the cell allows."""
        if params["space_group"].get("hall_number") is not None:
            sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"], params["space_group"]["hall_number"])
        else:
            sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"])

        # The cell must belong to the crystal system of the space group. Otherwise
        # the symmetry operations are not isometries of the box and the run would
        # silently converge to a constrained, unphysical state. The native check
        # (SpaceGroup::check_cell) is the one the computation box applies later.
        # The angles (reordered by axis_ordering) are absent for an orthogonal cell.
        angles = params.get("angles", None)
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

        # The physical basis (m3, Pmmm or z-mirror) from the angles the group fixes, which the cell keeps
        # in box relaxation; else the irreducible basis (SpaceGroup::enable_best_physical_basis)
        sg.enable_best_physical_basis()
        sg.display_info()
        return sg

    def to_standard_axes(self, field):
        """A full-grid array given in the user's axis order (axis_ordering) transposed to [a, b, c] and
        flattened; the array itself without axis_ordering."""
        if self._axis_perm is None:
            return field
        user_nx = [0] * len(self._axis_perm)
        for i, axis in enumerate(self._axis_perm):
            user_nx[axis] = self.nx[i]
        return np.ravel(np.transpose(np.reshape(np.asarray(field), user_nx), self._axis_perm))

    def initial_fields(self, initial_fields):
        """The initial fields {monomer type: full-grid array in the axis order of nx} of SCFT.run (checked:
        finite_values) as the working array of shape (M, n): symmetrized and in the irreducible basis with a
        space group."""
        sim = self.sim
        M, total_grid = len(sim.monomer_types), sim.prop_solver.total_grid
        initial_fields = {m: self.to_standard_axes(f) for m, f in initial_fields.items()}
        if self.sg is None:
            w = np.zeros([M, sim.prop_solver.n_grid], dtype=np.float64)
            for i in range(M):
                w[i, :] = np.reshape(initial_fields[sim.monomer_types[i]], sim.prop_solver.n_grid)
            return w
        w_full = np.zeros([M, total_grid], dtype=np.float64)
        for i in range(M):
            w_full[i, :] = np.reshape(initial_fields[sim.monomer_types[i]], total_grid)
        w_full = self.sg.symmetrize(w_full)
        w = self.sg.to_reduced_basis(w_full)
        return self.sg.reduced_to_irreducible(w)

    def initial_propagators(self, q_init):
        """Prepare graft profiles without changing them or the caller's arrays.

        Full-grid inputs follow axis_ordering and are converted to the solver's
        active reduced basis after checking the full space-group symmetry.
        Existing reduced inputs already use the standard solver basis, even with
        axis_ordering; their full-grid expansion must have the same symmetry.
        If both sizes coincide, the full-grid interpretation wins.
        Unlike an initial field guess, a graft profile defines the model, so it
        must not be silently averaged over symmetry orbits.
        """
        if q_init is None:
            return None
        prepared = {}
        full_size, reduced_size = self.sim.prop_solver.total_grid, self.sim.prop_solver.n_grid
        for label, values in q_init.items():
            profile = finite_values(values, f"q_init[{label!r}]")
            is_full = profile.size == full_size
            if is_full:
                profile = self.to_standard_axes(profile)
            elif self.sg is None or profile.size != reduced_size:
                sizes = (f"{full_size} full-grid or {reduced_size} reduced-basis values"
                         if self.sg is not None else f"{full_size} grid values")
                raise ValidationError(f"q_init[{label!r}] needs {sizes}, got {profile.size}")
            if self.sg is not None:
                full_profile = profile if is_full else self.sg.from_reduced_basis(profile)
                symmetric = self.sg.symmetrize(full_profile)
                tolerance = 1e-12 * np.max(np.abs(full_profile))
                if not np.allclose(full_profile, symmetric, rtol=0.0, atol=tolerance):
                    raise ValidationError(f"q_init[{label!r}] must have the symmetry of 'space_group'")
                if is_full:
                    profile = self.sg.to_reduced_basis(profile)
            prepared[label] = profile
        return prepared

    def to_solver_basis(self, w):
        """The fields w (M, n) of the working basis in the active basis of the solver."""
        return w if self.sg is None else self.sg.irreducible_to_reduced(w)

    def to_working_basis(self, phi):
        """The concentrations {type: array} of the solver in the working basis of the optimizer."""
        if self.sg is None:
            return phi
        return {name: self.sg.reduced_to_irreducible(phi[name]) for name in phi}

    def mean(self, field):
        """Spatial mean of a field of the working basis."""
        return self.sim.prop_solver.mean(field, basis="irreducible")

    def inner_product(self, field1, field2):
        """Inner product <field1 field2> of two fields of the working basis."""
        return self.sim.prop_solver.inner_product(field1, field2, basis="irreducible")

    def to_full_grid(self, field: np.ndarray) -> np.ndarray:
        """Expand a field of the working basis to the full grid (a copy without a space group)."""
        if self.sg is None:
            return np.array(field, dtype=np.float64)
        return self.sg.from_reduced_basis(self.sg.irreducible_to_reduced(field))

    def full_grid_fields(self, w):
        """The fields w (M, n) of the working basis as {monomer type: array on the full grid}, as SCFTResult.w."""
        return {name: self.to_full_grid(w[i]) for i, name in enumerate(self.sim.monomer_types)}
