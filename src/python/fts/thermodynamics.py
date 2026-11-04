"""Thermodynamic operators of CL-FTS: chemical potentials, pressure and partition ratios.

ThermodynamicOperators says which of them a run records, and the run loop adds them to its RunRecord.
See docs/theory/ComplexLangevinSimulation.md.
"""

import warnings
import weakref

import numpy as np

from ..utils.validation import ValidationError


class ThermodynamicOperators:
    """The thermodynamic operators of a CLFTS simulation, and which of them it records."""

    def __init__(self, sim):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.monomer_types, self.mpt = sim.monomer_types, sim.mpt
        self.prop_solver, self.smearing = sim.prop_solver, sim.smearing
        self.electrostatics = sim.electrostatics

        # Thermodynamic operators, recorded with H: the chemical potential of each species,
        # βμ_p = ln(φ_p/α_p) - ln Q_p per chain (validation rejects grafted chains), and with
        # recording.pressure the osmotic pressure (pressure_virial())
        chains = [self.prop_solver.get_polymer(p) for p in range(self.prop_solver.get_n_polymer_types())]
        self.chain_densities = np.array([chain.get_volume_fraction() / chain.get_alpha() for chain in chains])
        # One chemical potential per polymer, then one per point ion; with grand-canonical species the
        # number of chains or ions per reference chain of every species is recorded too
        self.n_chemical_potentials = len(self.chain_densities) + (len(self.electrostatics.ions) if self.electrostatics else 0)
        self.record_number_densities = bool(self.sim.grand_canonical) or (
            self.electrostatics is not None and any(ion["concentration"] is None for ion in self.electrostatics.ions))
        self.record_chemical_potentials = not any(polymer.get("initial_conditions") for polymer in self.sim.polymers)
        self.record_pressure = bool(self.sim.recording.get("pressure", False))
        if self.record_pressure:
            if not self.record_chemical_potentials:
                raise ValidationError("'recording.pressure' needs free chains (no initial_conditions)")
            if any(b != "periodic" for b in self.prop_solver.bc):
                raise ValidationError("'recording.pressure' needs a periodic box")
            if not self.prop_solver.numerical_methods()[self.prop_solver.numerical_method]["supports_stress"]:
                raise ValidationError(f"'recording.pressure' needs the stress; "
                                      f"'{self.prop_solver.numerical_method}' computes none")
            # The O(ds²) contour error of rk2 in the stress, and so in the pressure, is large where one
            # contour step b √ds spans more than the smearing length, the shortest length on which the
            # smeared fields vary (docs/theory/ComplexLangevinSimulation.md, Section 7); without smearing
            # they vary down to the grid. The types are those of the blocks, so a random block counts
            # with its effective segment length
            if self.sim.chain_model.lower() == "continuous" and self.prop_solver.numerical_method == "rk2":
                coarse = []
                for name in sorted({block["type"] for polymer in self.sim.polymers for block in polymer["blocks"]}):
                    if not self.smearing.enabled:
                        length = 0.0
                    elif self.smearing.type == "gaussian":
                        a_int = self.smearing.params["a_int"]
                        length = a_int.get(name, 0.0) if isinstance(a_int, dict) else a_int
                    else:
                        length = 1.0 / self.smearing.params["k_int"]
                    step = self.sim.segment_lengths[name] * np.sqrt(self.sim.ds)
                    if step > length:
                        coarse.append(f"{name}: step {step:.3g} > smearing length {length:.3g}")
                if coarse:
                    warnings.warn("With rk2 the recorded pressure carries an O(ds^2) contour error that is large "
                                  "when one contour step b sqrt(ds) exceeds the smearing length ("
                                  + ", ".join(coarse) + ", in R0); use numerical_method 'rqm4' or a smaller ds")
        # Canonical single-site species, discrete chains of one segment and point ions, whose ratio
        # |Q_s|/<|e^(-u_s)|> is recorded every step (partition_ratios())
        self._single_site_polymers = [
            p for p, polymer in enumerate(self.sim.polymers)
            if self.sim.chain_model.lower() == "discrete" and p not in self.sim.grand_canonical
            and not polymer.get("initial_conditions") and len(polymer["blocks"]) == 1
            and polymer["blocks"][0]["type"] in self.monomer_types
            and np.isclose(polymer["blocks"][0]["length"], self.sim.ds)]
        self.partition_ratio_species = [f"polymer {p} ({self.sim.polymers[p]['blocks'][0]['type']})"
                                        for p in self._single_site_polymers]
        if self.electrostatics is not None:
            self.partition_ratio_species += [f"ion {ion['name']}" for ion in self.electrostatics.ions
                                             if ion["concentration"] is not None]

    def chemical_potentials(self, total_partitions):
        """Chemical potential operator βμ_p = ln(φ_p/α_p) - ln Q_p of each species (complex).

        H depends on the number of chains n_p only through -n_p ln Q_p, and the ideal term of
        the chains gives ln(n_p/V), so ∂ln Z/∂n_p yields this operator; its average is the
        chemical potential per chain in the convention of SCFT.compute_chemical_potentials,
        which it equals at the saddle point. Its real part does not depend on the branch of
        the logarithm. An absent canonical species has -inf; grand-canonical species have
        their imposed chemical potentials, including at Q_p = 0. The point ions follow the polymers,
        βμ_s = ln(N c_s) - ln Q_s (Electrostatics.ion_chemical_potentials).
        """
        values = np.full(self.chain_densities.shape, -np.inf, dtype=np.complex128)
        occupied_canonical = self.chain_densities > 0
        occupied_canonical[self.sim.grand_canonical] = False
        partitions = np.asarray(total_partitions, dtype=np.complex128)
        values[occupied_canonical] = (np.log(self.chain_densities[occupied_canonical])
                                      - np.log(partitions[occupied_canonical]))
        values[self.sim.grand_canonical] = self.sim.polymer_chemical_potentials[self.sim.grand_canonical]
        if self.electrostatics is not None and self.electrostatics.ions:
            values = np.append(values, self.electrostatics.ion_chemical_potentials())
        return values

    def number_densities(self, total_partitions):
        """Chains (or ions) per reference chain of each species, n_p/n_ref: φ_p/α_p of canonical
        polymers, e^(βμ_p) Q_p of grand-canonical ones, then the point ions (complex operators)."""
        values = np.asarray(self.chain_densities, dtype=np.complex128).copy()
        values[self.sim.grand_canonical] = self.sim.fugacities[self.sim.grand_canonical] * np.asarray(total_partitions)[self.sim.grand_canonical]
        if self.electrostatics is not None and self.electrostatics.ions:
            values = np.append(values, self.electrostatics.ion_number_densities())
        return values

    def grand_canonical_hamiltonian(self, total_partitions):
        """-Σ_p e^(βμ_p) Q_p of the grand-canonical chains, per reference chain (complex)."""
        return -complex(np.sum(self.sim.fugacities * np.asarray(total_partitions, dtype=np.complex128)))

    def partition_ratios(self, w_aux):
        """|Q_s|/<|e^(-u_s)|> of each canonical single-site species (partition_ratio_species).

        Such a species has Q_s = <e^(-u_s)>, the grid mean of the Boltzmann weight of its one site:
        e^(-w_t ds) with the smeared field w_t (+ z_t ψ) of its type for a discrete chain of one
        segment, e^(-z_s (Γ_s * ψ)/N) for a point ion. The ratio is 1 for real fields and falls
        towards 0 as the complex weights cancel in the mean. -n_s ln Q_s has a pole at Q_s = 0,
        where the force n_s e^(-u_s)/Q_s of the species grows without bound. Grand-canonical
        species enter H as -e^(βμ_s) Q_s, without a pole, and are left out.
        """
        ratios = []
        if self._single_site_polymers:
            w = self.mpt.to_monomer_fields_dict(w_aux)
            if self.electrostatics is not None:
                self.electrostatics.add_potential(w)
            types = {self.sim.polymers[p]["blocks"][0]["type"] for p in self._single_site_polymers}
            w = self.smearing.apply_to_dict({t: w[t] for t in types})
            for p in self._single_site_polymers:
                u = self.sim.ds * w[self.sim.polymers[p]["blocks"][0]["type"]]
                weight = np.exp(-(u - np.min(np.real(u))))   # the ratio does not change with a real shift of u
                ratios.append(min(1.0, float(np.abs(np.mean(weight)) / np.mean(np.abs(weight)))))   # 1 up to rounding
        if self.electrostatics is not None:
            ratios += self.electrostatics.ion_partition_ratios()
        return np.array(ratios)

    def pressure(self, w_aux, phi):
        """Osmotic pressure operator βΠ R₀³/C of the auxiliary fields w_aux (complex).

        The propagators of the solver must be those of w_aux, as after compute_concentrations,
        and phi are its concentrations. βΠ = ∂ln Z/∂V at fixed numbers of canonical chains and
        ions and fixed chemical potentials of grand-canonical species, with the grid values of
        the fields held fixed while the box is dilated in its shape (k² ∝ V^(-2/d)). The terms,
        per C/R₀³ = ρ₀/N:
        - the ideal gas of the chains, Σ_p n_p/n_ref: φ_p/α_p of canonical polymers and
          e^(βμ_p) Q_p of grand-canonical ones (from their term -e^(βμ_p) Q_p of H);
        - the conformations, -(1/d) Σ_a L_a ∂h/∂L_a with h = -Σ_p (φ_p/α_p) ln Q_p - Σ_p e^(βμ_p) Q_p
          over the canonical and the grand-canonical polymers (the solver stress, at fixed
          smeared fields);
        - the smearing functions, which follow the box: -Σ_i <φ_i ((∂Γ_i/∂ln V) * (w_i + z_i ψ))>;
        - the field terms of H, which scale with V: -(Σ_i <h2_i Ω_i² + h1_i Ω_i> + h_const);
        - the Gaussian normalization of each auxiliary field, n_grid/2 per C V/R₀³ chains, which
          removes the grid-dependent part of the field term;
        - ψ and the point ions: their terms of H and the normalization of ψ (Electrostatics.pressure).
        At the homogeneous saddle point this is the Flory-Huggins osmotic pressure
        Σ_p φ_p/α_p + φᵀXφ/2 + (ζN/2)((Σφ)² - 1) as C → ∞, plus E q²/(2κ²) for a screened system
        with the net charge q = Σ_i z_i φ_i per segment (ψ₀ = E q/κ² there).
        """
        M, dim, es = len(self.monomer_types), self.prop_solver.dim, self.electrostatics
        n_chains = np.sqrt(self.sim.langevin["nbar"]) * self.prop_solver.get_volume()   # C V/R₀³
        total_partitions = [self.prop_solver.get_partition_function(p) for p in range(len(self.sim.polymers))]
        value = complex(np.sum(self.number_densities(total_partitions)[:len(self.sim.polymers)]))

        self.prop_solver.compute_stress()
        stress, lx = np.asarray(self.prop_solver.get_stress()), self.prop_solver.get_lx()
        if self.sim.grand_canonical:
            # h = -Σ_p e^(βμ_p) Q_p of the grand-canonical chains: dh/dθ = -Σ_p e^(βμ_p) dQ_p/dθ
            stress = stress + np.asarray(self.prop_solver.get_stress_gce(self.sim.fugacities))
        value -= sum(lx[a] * stress[a] for a in range(dim)) / dim

        if self.smearing.enabled:
            w = self.mpt.to_monomer_fields_dict(w_aux)
            if es is not None:
                es.add_potential(w)
            for name in self.monomer_types:
                value -= np.mean(phi[name] * self.smearing.dilation(w[name], name))

        value -= sum(self.mpt.h_coef_mu2[i] * np.mean(w_aux[i] ** 2) + self.mpt.h_coef_mu1[i] * np.mean(w_aux[i])
                     for i in range(M)) + self.mpt.h_const
        value += 0.5 * (self.mpt.R + self.mpt.I) * self.prop_solver.n_grid / n_chains
        if es is not None:
            value += es.pressure(dim, n_chains)
        return value

    def pressure_virial(self, w_aux, phi):
        """Virial form of the osmotic pressure operator: pressure() plus terms of zero average.

        Every auxiliary field Ω_i whose Gaussian integral pressure() normalizes obeys the
        Schwinger-Dyson identity <C V/R₀³ mean(Ω_i ∂H/∂Ω_i)> = n_grid (integration by parts over
        its n_grid grid values; ∂H/∂Ω_i is compute_func_deriv without its sign flip). Adding
        (1/2)[mean(Ω_i ∂H/∂Ω_i) - n_grid/(C V/R₀³)] for each field replaces the field term
        -(h2_i Ω_i² + h1_i Ω_i) and the normalization n_grid/(2 C V/R₀³) by
        -h1_i Ω_i/2 + (Ω_i/2) Σ_j A_ji φ_j: the same average, without the Gaussian fluctuation
        of order sqrt(n_grid/2)/(C V/R₀³) per field and sample. ψ gets the same treatment mode by
        mode (Electrostatics.pressure_virial_correction). CLFTS records this operator.

        The identities hold for exact CL sampling of H; an O(δτ) bias of the integrator or the
        stabilization term of dynamical stabilization, which is not part of H, can make the added
        terms average to a small non-zero value.
        """
        M, es = len(self.monomer_types), self.electrostatics
        n_chains = np.sqrt(self.sim.langevin["nbar"]) * self.prop_solver.get_volume()   # C V/R₀³
        value = self.pressure(w_aux, phi)
        phi_smeared = self.smearing.apply_to_dict(phi)
        deriv = self.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
        for i in self.mpt.aux_fields_real_idx:
            value += 0.5 * (np.mean(w_aux[i] * deriv[i]) - self.prop_solver.n_grid / n_chains)
        for i in self.mpt.aux_fields_imag_idx:
            value += 0.5 * (-np.mean(w_aux[i] * deriv[i]) - self.prop_solver.n_grid / n_chains)
        if es is not None:
            value += es.pressure_virial_correction(es.charge_density(phi_smeared), self.prop_solver.dim, n_chains)
        return value
