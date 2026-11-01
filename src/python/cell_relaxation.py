"""Relaxation of the simulation cell in SCFT: which lattice parameters vary, and the stress that moves them.

See docs/theory/SelfConsistentFieldTheory.md.
"""

import weakref

import numpy as np

from .validation import ValidationError


# Which lattice parameters may vary during box relaxation (relax_cell=True):
#   lx_reduced : indices of the independent box lengths among [a, b, c]
#   lx_full    : for each of a, b, c, the independent length it follows
#   angles     : indices of the free angles among [alpha, beta, gamma]
# "Orthorhombic" frees every box length in any dimension and is handled in the constructor.
# To add a crystal system, add one row here (and, if it needs averaged stress
# components like a = b, one case in CellRelaxation.update).
CRYSTAL_SYSTEMS = {
    "Orthorhombic": {"lx_reduced": None,      "lx_full": None,      "angles": []},
    "Tetragonal":   {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # a = b != c
    "Cubic":        {"lx_reduced": [0],       "lx_full": [0, 0, 0], "angles": []},         # a = b = c
    "Hexagonal":    {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # a = b != c, gamma = 120
    "Trigonal":     {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # hexagonal axes (space group only)
    "Monoclinic":   {"lx_reduced": [0, 1, 2], "lx_full": [0, 1, 2], "angles": [1]},        # beta free (a space group: see below)
    "Triclinic":    {"lx_reduced": [0, 1, 2], "lx_full": [0, 1, 2], "angles": [0, 1, 2]},  # all free
    "Hexagonal2D":  {"lx_reduced": [0],       "lx_full": [0, 0],    "angles": []},         # a = b, gamma = 120
    "Oblique2D":    {"lx_reduced": [0, 1],    "lx_full": [0, 1],    "angles": [2]},        # gamma free
}


class CellRelaxation:
    """Which lattice parameters of an SCFT calculation vary, and their update from the stress.

    stress() computes the stress of the fields, error() its size along the lattice parameters that
    vary, update() moves the fields and the cell together, and check_tied_lengths() reports a tie of
    the crystal system that the structure does not satisfy. n_variables is the size of the optimizer.
    """

    def __init__(self, sim, params, angles):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.monomer_types, self.sg = sim.monomer_types, sim.sg
        self.prop_solver, self.smearing = sim.prop_solver, sim.smearing
        self.electrostatics = sim.electrostatics
        self.stress_scale = params.get("stress_scale", 1)
        self.n_variables = self._choose_lattice_parameters(params, angles)

    def _choose_lattice_parameters(self, params, angles):
        """Which lattice parameters the iteration varies; returns the number of variables of the optimizer."""
        # Which lattice parameters may vary during box relaxation (table CRYSTAL_SYSTEMS)
        dim = len(params["nx"])
        if self.sg is not None:
            # The space group fixes the crystal system (and which angles vary, below)
            crystal_system = self.sg.get_crystal_system()
            n_field_var = len(self.monomer_types) * self.sg.get_n_reduced_basis_full()
        else:
            # The user chooses; a 2D cell without "crystal_system" is classified from gamma
            if params.get("crystal_system", None) is None and dim == 2 and angles is not None and len(angles) > 0:
                gamma = angles[0] if len(angles) == 1 else angles[2]
                if abs(gamma - 120.0) < 1e-6:
                    crystal_system = "Hexagonal2D"
                elif abs(gamma - 90.0) > 1e-6:
                    crystal_system = "Oblique2D"
                else:
                    crystal_system = "Orthorhombic"
            else:
                crystal_system = params.get("crystal_system", None) or "Orthorhombic"

            valid_systems = {1: ["Orthorhombic"],
                             2: ["Orthorhombic", "Hexagonal2D", "Oblique2D"],
                             3: ["Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Monoclinic", "Triclinic"]}[dim]
            if crystal_system not in valid_systems:
                raise ValueError(f"Crystal system '{crystal_system}' is not valid for a {dim}D box. " +
                    "Choose from: " + ", ".join(valid_systems) + ".")

            # Relaxation never changes the fixed angles, and its first step overwrites the tied
            # lengths with the first one (a jump of the cell), so with relax_cell a crystal system
            # that is given (or Hexagonal2D, classified from gamma) must describe the cell
            if params["relax_cell"] and (params.get("crystal_system", None) is not None
                                         or crystal_system == "Hexagonal2D"):
                lx_given = [float(v) for v in params["lx"]]
                if angles is None or len(angles) == 0:
                    cell_angles = [90.0, 90.0, 90.0]
                elif dim == 2:
                    cell_angles = [90.0, 90.0, float(angles[0] if len(angles) == 1 else angles[2])]
                else:
                    cell_angles = [float(v) for v in angles]
                tied = CRYSTAL_SYSTEMS[crystal_system]["lx_full"] or list(range(dim))
                fixed = {"Hexagonal": [90, 90, 120], "Hexagonal2D": [None, None, 120], "Monoclinic": [90, None, 90],
                         "Triclinic": [None] * 3, "Oblique2D": [None] * 3}.get(crystal_system, [90, 90, 90])
                rule = {"Orthorhombic": "angles of 90 degrees", "Tetragonal": "a = b and angles of 90 degrees",
                        "Cubic": "a = b = c and angles of 90 degrees",
                        "Hexagonal": "a = b, alpha = beta = 90 and gamma = 120 degrees",
                        "Hexagonal2D": "a = b and gamma = 120 degrees",
                        "Monoclinic": "alpha = gamma = 90 degrees"}.get(crystal_system)
                if (any(abs(lx_given[i] - lx_given[tied.index(tied[i])]) > 1e-6 * lx_given[i] for i in range(dim))
                        or (dim > 1 and any(f is not None and abs(c - f) > 1e-6 for f, c in zip(fixed, cell_angles)))):
                    raise ValidationError(f"crystal_system '{crystal_system}' needs a cell with {rule}; got lx = "
                                          f"{lx_given} and angles = {cell_angles} (set lx and angles to such a cell"
                                          + (", or crystal_system 'Oblique2D')" if dim == 2 else ")"))
            n_field_var = len(self.monomer_types) * np.prod(params["nx"])
            # ψ is iterated with the fields
            if self.electrostatics is not None:
                n_field_var += int(np.prod(params["nx"]))
        self.crystal_system = crystal_system

        row = CRYSTAL_SYSTEMS[crystal_system]
        if crystal_system == "Orthorhombic":    # every box length is free, in any dimension
            self.lx_reduced_indices = list(range(dim))
            self.lx_full_indices = list(range(dim))
        else:
            self.lx_reduced_indices = list(row["lx_reduced"])
            self.lx_full_indices = list(row["lx_full"])
        if self.sg is None:
            self.angles_reduced_indices = list(row["angles"])
        else:
            # Angles vary only in monoclinic and triclinic groups: those whose change keeps every
            # symmetry operation an isometry of the cell (SpaceGroup.check_cell), the angle between
            # the two axes perpendicular to the unique axis of a monoclinic setting (beta for the
            # unique axis b, gamma for c, alpha for a) and all three in a triclinic group
            self.angles_reduced_indices = []
            if crystal_system in ("Monoclinic", "Triclinic"):
                cell_lx = list(self.prop_solver.get_lx())
                cell_angles = list(self.prop_solver.get_angles_degrees())
                for i in range(3):
                    trial = list(cell_angles)
                    trial[i] += 1.0
                    try:
                        self.sg.check_cell(cell_lx, trial)
                        self.angles_reduced_indices.append(i)
                    except RuntimeError:
                        pass
        self.angles_full_indices = list(self.angles_reduced_indices)

        # Number of variables the optimizer adjusts: the fields, plus the free box lengths and angles
        if params["relax_cell"]:
            n_var = n_field_var + len(self.lx_reduced_indices) + len(self.angles_reduced_indices)
        else:
            n_var = n_field_var

        # Enable off-diagonal (cross-term) stress computation when angle
        # optimization is active. Without this explicit opt-in, a run started
        # at exactly 90-degree angles would compute zero angle derivatives
        # (the orthogonal-box stress fast path skips the V12/V13/V23 cross
        # terms) and the angles would never move away from 90 degrees.
        if len(self.angles_reduced_indices) > 0:
            self.prop_solver.set_force_off_diagonal_stress(True)

        # Stress array layout: 3D [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta, dH/dalpha],
        # 2D [dH/dL1, dH/dL2, dH/dgamma, 0, 0, 0]; the angle index in [alpha, beta, gamma]
        # maps to the stress index through angle_stress_map
        angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
        self.angle_stress_indices = [angle_stress_map[i] for i in self.angles_reduced_indices if i in angle_stress_map]

        # Lengths tied by the crystal system (a = b, or a = b = c) move together, along the sum
        # of their stresses. Whether each stress vanishes is checked at the end.
        self.tied_groups = [[i for i in range(dim) if self.lx_full_indices[i] == j]
                            for j in range(len(self.lx_reduced_indices))]
        return n_var

    def update(self, fields, fields_diff, stress_array, old_error_level, error_level):
        """One optimizer step on the concatenated (fields, box lengths, angles)
        vector; fields are the flattened fields (and ψ). Applies the new lattice
        parameters to the solver and returns the new flattened fields. Length
        changes are limited to 10 % and angle changes to 5 degrees per step.
        """
        dim = self.prop_solver.dim

        # -dH/dL for the independent box lengths; lengths tied by the crystal
        # system (a = b, or a = b = c) use the averaged derivative
        dlx_full = -stress_array[0:3]
        crystal_system = self.crystal_system
        if crystal_system == "Hexagonal2D":
            dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1])])
        elif crystal_system in ("Tetragonal", "Hexagonal", "Trigonal") and len(self.lx_reduced_indices) == 2:
            dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1]), dlx_full[2]])
        elif crystal_system == "Cubic":
            dlx_reduced = np.array([np.mean(dlx_full)])
        else:
            dlx_reduced = dlx_full[self.lx_reduced_indices]

        # Stress array layout: 3D [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta, dH/dalpha],
        # 2D [dH/dL1, dH/dL2, dH/dgamma, 0, 0, 0]. get_stress() returns +dH/d(parameter)
        # for angles as well as lengths (the d(cos gamma)/d gamma = -sin gamma factor is
        # already applied in the C++ compute_stress post-processing), so gradient
        # descent uses the same negative sign as for the box lengths.
        angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
        dangle = -np.array([stress_array[angle_stress_map[i]] for i in self.angles_reduced_indices if i in angle_stress_map])

        current_lx = np.array(self.prop_solver.get_lx())[self.lx_reduced_indices]
        current_angles = np.array(self.prop_solver.get_angles_degrees())[self.angles_reduced_indices] if len(self.angles_reduced_indices) > 0 else np.array([])

        # Angles enter the optimizer in radians, the unit of their residual -dH/dθ
        am_current = np.concatenate((fields, current_lx, np.deg2rad(current_angles)))
        am_diff = np.concatenate((fields_diff,
                                 self.stress_scale*dlx_reduced,
                                 self.stress_scale*dangle))
        am_new = self.sim.field_optimizer.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

        fields = am_new[0:len(fields)]

        n_lx = len(self.lx_reduced_indices)
        n_angles = len(self.angles_reduced_indices)

        # Restricting |dLx| to be less than 10 % of Lx
        old_lx = current_lx
        new_lx = np.array(am_new[-(n_lx + n_angles):-(n_angles)] if n_angles > 0 else am_new[-n_lx:])
        new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
        new_lx = (1 + new_dlx)*old_lx

        if n_angles > 0:
            old_angles = current_angles
            new_angles = np.rad2deg(np.array(am_new[-n_angles:]))
            new_dangle = np.clip(new_angles - old_angles, -5.0, 5.0)
            new_angles = old_angles + new_dangle

            full_angles = list(self.prop_solver.get_angles_degrees())
            for i, idx in enumerate(self.angles_full_indices):
                full_angles[idx] = new_angles[i]
            self.prop_solver.set_lattice_parameters(list(np.array(new_lx)[self.lx_full_indices]), full_angles)
        else:
            self.prop_solver.set_lx(list(np.array(new_lx)[self.lx_full_indices]))

        # The smearing kernel Γ(k) and the Coulomb kernel are functions of the physical wavenumber
        self.smearing.set_box(self.prop_solver.get_lx(), self.prop_solver.get_angles_degrees())
        if self.electrostatics is not None:
            self.electrostatics.set_box()
        return fields

    def stress(self, w, phi):
        """Lattice derivatives of the free energy at the fixed fields w (and ψ) of the last
        compute_concentrations, whose concentrations are phi, in the layout of get_stress().

        At the saddle point they are the derivatives of the free energy. Besides the stress of
        the propagators they hold the box dependence of the smearing kernel Γ(k), at the fields
        w_i + z_i ψ the chains feel, and with electrostatics that of the Coulomb kernel and the
        smearing of the point ions.
        """
        dim = self.prop_solver.dim
        self.prop_solver.compute_stress()
        stress = np.array(self.prop_solver.get_stress())
        if self.smearing.enabled:
            es = self.electrostatics
            felt = w if es is None else [w[i] + es.charges[m] * self.sim.psi for i, m in enumerate(self.monomer_types)]
            smearing_stress = self.smearing.stress(felt, [phi[m] for m in self.monomer_types])
            if es is not None:
                smearing_stress += es.stress(self.sim.psi, self.smearing.coefficients, self.smearing.k_sq,
                                             self.smearing.lattice_derivatives_k_sq())
            # The angles of an orthogonal box do not vary unless the crystal system lets them
            angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
            if self.smearing.orthogonal and not any(i in angle_stress_map for i in self.angles_reduced_indices):
                smearing_stress[dim:] = 0.0
            stress += smearing_stress[:len(stress)]
        return stress

    def error(self, stress):
        """Size of the stress along the optimized lattice parameters: a tied group counts as the sum of
        its stresses divided by sqrt(group size), the norm of its stresses when they are equal."""
        lengths = [np.sum(stress[group]) / np.sqrt(len(group)) for group in self.tied_groups]
        return np.linalg.norm(lengths + list(stress[self.angle_stress_indices]))

    def check_tied_lengths(self, stress_array, error_level):
        # The tied lengths relax only along the sum of their stresses. Without a space group the
        # structure need not have the symmetry of the tie, and then the stresses of the tied lengths
        # do not vanish separately at the converged tied cell: there is no stress-free cell with
        # this crystal system.
        if self.sim.relax_cell and self.sg is None and stress_array is not None and error_level < self.sim.tolerance:
            unequal = [stress_array[group] - np.mean(stress_array[group]) for group in self.tied_groups if len(group) > 1]
            if unequal and np.linalg.norm(np.concatenate(unequal)) >= self.sim.tolerance:
                tie = "a = b = c" if self.crystal_system == "Cubic" else "a = b"
                free_system = {"Hexagonal": "Triclinic", "Hexagonal2D": "Oblique2D"}.get(self.crystal_system, "Orthorhombic")
                print(f"Warning: the box-length stresses {stress_array[:self.prop_solver.dim]} do not vanish separately. The "
                      f"structure does not satisfy the {tie} condition of crystal_system '{self.crystal_system}'; "
                      f"relax it with crystal_system '{free_system}' to find a cell where every stress vanishes.")
