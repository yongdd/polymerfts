"""Structure functions of L-FTS and CL-FTS: their samples and their files.

Each sample adds <μ_i(k) φ_j(-k)> to the sums of the RunRecord, and the files hold S(k) of the
densities: the sums over the samples, scaled and with the smearing divided out. L-FTS uses the half
spectrum of real fields, CL-FTS the full spectrum.
"""

import itertools
import json
import os
import weakref

import numpy as np

from .checkpoint import system_data
from .utils import atomic_savemat


class LftsStructureFunctions:
    """The structure functions of an LFTS simulation: empty_sums(), add() a sample and save() a file."""

    def __init__(self, sim):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.monomer_types, self.mpt = sim.monomer_types, sim.mpt
        self.prop_solver, self.smearing = sim.prop_solver, sim.smearing
        self.periodic_box = sim._periodic_box

    def empty_sums(self):
        """Sums of <u(k) phi(-k)> on the half spectrum of rfftn."""
        half_spectrum = list(self.prop_solver.nx[:-1]) + [self.prop_solver.nx[-1]//2 + 1]
        return {",".join(pair): np.zeros(half_spectrum, np.complex128)
                for pair in itertools.combinations_with_replacement(self.monomer_types, 2)}

    def _transform(self, fields):
        """rfftn of each field (n_fields, n_grid) on the grid: with the platform FFT
        (PropagatorSolver.fourier) in a periodic box, with numpy.fft otherwise."""
        fields = np.reshape(fields, (-1, self.prop_solver.n_grid))
        if self.periodic_box:
            return self.prop_solver.fourier.forward(fields)
        nx = tuple(self.prop_solver.nx)
        return np.fft.rfftn(np.reshape(fields, (-1,) + nx), axes=tuple(range(1, len(nx) + 1)))

    def add(self, w_aux, phi, sf_average):
        """Add the current state to the sums of the structure functions."""
        M = len(self.monomer_types)
        # Perform Fourier transforms (all densities in one call, all fields in another).
        # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
        # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>
        kept = [k for k in range(M-1) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
        phi_k = self._transform([phi[name] for name in self.monomer_types])/self.prop_solver.n_grid
        w_k = self._transform(w_aux[kept])/self.prop_solver.n_grid if kept else None
        mu_fourier = {}
        phi_fourier = {}
        for i in range(M):
            key = self.monomer_types[i]
            phi_fourier[key] = phi_k[i]
            mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
            for j, k in enumerate(kept):
                mu_fourier[key] += w_k[j]*self.mpt.matrix_a[i,k]/self.mpt.eigenvalues[k]
        # Accumulate S_ij(K), assuming that <mu(k)>*<phi(-k)> is zero
        for key in sf_average:
            monomer_pair = sorted(key.split(","))
            sf_average[key] += mu_fourier[monomer_pair[0]]* np.conj( phi_fourier[monomer_pair[1]])

    def save(self, langevin_step, record):
        """Write the means of the interval to structure_function_NNNNNN.mat."""
        # Make a dictionary for chi_n
        sf_average, sf_sample_count = record.sums, record.sample_count
        mdic = system_data(self.sim)
        # Add structure functions to the dictionary
        for key in sf_average:
            sf_average[key] *= 1.0/max(sf_sample_count, 1)* \
                    self.prop_solver.get_volume()*np.sqrt(self.sim.langevin["nbar"])
            # With smearing the fields couple to the smeared densities, so <μ φ> = Γ(k) S(k):
            # divide by Γ to record S(k) of the densities (modes the smeared interaction
            # does not see, Γ < 1e-12, are recorded as 0). With walls Γ lives in the cosine
            # basis while S(k) is a Fourier transform, so it is recorded as it is.
            if self.smearing.enabled and self.smearing.periodic:
                gamma = self.smearing.fourier[..., :self.prop_solver.nx[-1]//2+1]
                sf_average[key] = np.divide(sf_average[key], gamma, out=np.zeros_like(sf_average[key]), where=gamma > 1e-12)
            monomer_pair = sorted(key.split(","))
            mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1]] = sf_average[key]
        mdic["sample_count"] = sf_sample_count
        mdic["simulation_metadata_json"] = json.dumps(self.sim.metadata)
        atomic_savemat(os.path.join(self.sim.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)


class ClftsStructureFunctions:
    """The structure functions of a CLFTS simulation: empty_sums(), add() a sample and save() a file."""

    def __init__(self, sim):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.monomer_types, self.mpt = sim.monomer_types, sim.mpt
        self.prop_solver, self.smearing = sim.prop_solver, sim.smearing
        self.electrostatics = sim.electrostatics
        self.periodic_box = sim._periodic_box

        # Structure functions: composition modes with a zero eigenvalue carry no auxiliary field,
        # but ψ measures the charge direction z, <((k² + κ²)/E) ψ(k) φ_j(-k)> = Σ_i z_i <Γ_i φ_i φ_j>.
        # Its part outside the eigenvectors of the auxiliary fields, u = P_0 z (P_0: projector onto
        # the zero-eigenvalue modes), completes the estimator (docs/theory/StructureFunctionEstimators.md,
        # Section 5).
        # The estimators with ψ need a periodic box: with walls ψ lives in the cosine basis, while the
        # structure functions are Fourier transforms
        self.psi_probe = None
        if self.electrostatics is not None and self.electrostatics.coupling > 0 and self.periodic_box:
            M = len(self.monomer_types)
            kept = [k for k in range(M) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
            projector = np.eye(M) - sum(np.outer(self.mpt.matrix_a[:, k], self.mpt.matrix_a[:, k]) for k in kept) / M
            u = projector @ np.array([self.electrostatics.charges[name] for name in self.monomer_types])
            if u @ u > 1e-12:
                self.psi_probe = u / (u @ u)

    def empty_sums(self):
        """Sums of <u(k) phi(-k)> on the full spectrum of complex fields, and of the charge structure function."""
        nx = self.prop_solver.nx
        sums = {",".join(pair): np.zeros(nx, np.complex128)
                for pair in itertools.combinations_with_replacement(self.monomer_types, 2)}
        # Charge structure function (with Coulomb coupling): sums of the mixed form M = ((k² + κ²)/E) ψ c,
        # of the zero-mean difference D = (field-field form) - M and of their products, for a combination
        # M + w D of least variance (docs/theory/StructureFunctionEstimators.md, Section 6)
        if self.electrostatics is not None and self.electrostatics.coupling > 0 and self.periodic_box:
            sums |= {key: np.zeros(nx, np.complex128) for key in self.electrostatics.CHARGE_SUMS}
        return sums

    def _transform(self, fields):
        """fftn of each field (n_fields, n_grid) on the grid: with the platform FFT
        (PropagatorSolver.fourier) in a periodic box, with numpy.fft otherwise."""
        fields = np.reshape(fields, (-1, self.prop_solver.n_grid))
        if self.periodic_box:
            return self.prop_solver.fourier.forward(fields)
        nx = tuple(self.prop_solver.nx)
        return np.fft.fftn(np.reshape(fields, (-1,) + nx), axes=tuple(range(1, len(nx) + 1)))

    def add(self, w_aux, phi, sums):
        """Add the current state to the sums of the structure functions."""
        M = len(self.monomer_types)
        # Perform Fourier transforms (full FFT for complex fields).
        # Shifting the Gaussian integral over Ω_k gives <Ω_k φ_j> = (λ_k/M) Σ_l A_lk <φ_l φ_j>,
        # and A Aᵀ = M I, so μ_i = Σ_k A_ik Ω_k/λ_k gives <μ_i φ_j> = <δφ_i δφ_j>. All fields
        # fluctuate in CL-FTS; only the incompressible pressure field (λ = 0) is left out.
        # The fields are complex, so φ(-k) is not the conjugate of φ(k):
        # φ(-k) = conj(FFT[conj φ](k)).
        # All densities are transformed in one call, all fields in another
        kept = [k for k in range(M) if not np.isclose(self.mpt.eigenvalues[k], 0.0)]
        phi_mk = np.conj(self._transform(np.conj([phi[name] for name in self.monomer_types]))) / self.prop_solver.n_grid
        w_k = self._transform(w_aux[kept]) / self.prop_solver.n_grid if kept else None
        mu_fourier = {}
        phi_fourier = {}   # φ(-k)
        for i in range(M):
            key = self.monomer_types[i]
            phi_fourier[key] = phi_mk[i]
            mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
            for j, k in enumerate(kept):
                mu_fourier[key] += w_k[j] * self.mpt.matrix_a[i, k] / self.mpt.eigenvalues[k]
        # Zero-eigenvalue modes along u: μ_i += u_i [((k² + κ²)/E) ψ(k) - Σ_l z_l μ_l(k)] / |u|²
        if self.psi_probe is not None:
            es = self.electrostatics
            probe = es.psi_k * es.screened_k_sq / es.coupling / self.prop_solver.n_grid
            # ψ measures the charge of the point ions and the fixed charge as well: remove them
            if es.ions:
                probe -= np.reshape(self._transform(es.ion_charge())[0], probe.shape) / self.prop_solver.n_grid
            if es.fixed_charge is not None:
                probe -= np.reshape(self._transform(es.fixed_charge)[0], probe.shape) / self.prop_solver.n_grid
            # k = 0 is not a structure function in the canonical ensemble; with screening
            # κ²ψ(0)/E would carry a net charge there
            probe[(0,) * len(self.prop_solver.nx)] = 0.0
            for name in self.monomer_types:
                probe -= es.charges[name] * mu_fourier[name]
            for i, name in enumerate(self.monomer_types):
                mu_fourier[name] = mu_fourier[name] + self.psi_probe[i] * probe
        # Accumulate S_ij(K) = <μ_i(k) φ_j(-k)>
        for pair in itertools.combinations_with_replacement(self.monomer_types, 2):
            sums[",".join(pair)] += mu_fourier[pair[0]] * phi_fourier[pair[1]]
        if "charge_mixed" in sums:
            es = self.electrostatics
            n_grid, nx = self.prop_solver.n_grid, self.prop_solver.nx
            norm = self.prop_solver.get_volume() * np.sqrt(self.sim.langevin["nbar"])
            psi_k = es.psi_k / n_grid
            psi_mk = es.psi_minus_k() / n_grid
            # c(-k) = Σ_i z_i Γ_i(k) φ_i(-k) from the φ_i(-k) above (Γ_i is even in k), and the
            # smeared charge of the point ions
            charge_mk = sum(z * self.smearing.kernel(name) * phi_fourier[name]
                            for name, z in es.charges.items() if z != 0.0)
            if es.ions:
                ion_mk = np.conj(self._transform(np.conj(es.ion_charge()))[0]) / n_grid
                charge_mk = charge_mk + np.reshape(ion_mk, np.shape(psi_k))
            if es.fixed_charge is not None:
                fixed_mk = np.conj(self._transform(es.fixed_charge)[0]) / n_grid
                charge_mk = charge_mk + np.reshape(fixed_mk, np.shape(psi_k))
            rate = es.screened_k_sq / es.coupling
            with np.errstate(divide="ignore", invalid="ignore"):
                mixed = rate * norm * psi_k * charge_mk
                difference = rate ** 2 * (norm * psi_k * psi_mk + 1.0 / rate) - mixed
            mixed[(0,) * len(nx)] = 0.0
            difference[(0,) * len(nx)] = 0.0
            sums["charge_mixed"] += mixed
            sums["charge_difference"] += difference
            sums["charge_mixed_difference"] += mixed.real * difference.real
            sums["charge_difference_squared"] += difference.real ** 2

    def save(self, langevin_step, record):
        """Writes structure_function_<step>.mat: S(k), the charge structure function and the mean concentrations."""
        sums, sf_sample_count = record.sums, record.sample_count
        # Make a dictionary for chi_n
        mdic = system_data(self.sim)
        # Add structure functions to the dictionary
        for monomer_pair in itertools.combinations_with_replacement(self.monomer_types, 2):
            sf_scaled = (
                sums[",".join(monomer_pair)] *
                1.0 / max(sf_sample_count, 1) *
                self.prop_solver.get_volume() * np.sqrt(self.sim.langevin["nbar"])
            )
            # With smearing the fields couple to the smeared densities, so <μ_i φ_j> = Γ_i(k) S_ij(k):
            # divide by Γ_i to record S(k) of the densities (modes the smeared interaction
            # does not see, Γ < 1e-12, are recorded as 0). With walls Γ lives in the cosine
            # basis while S(k) is a Fourier transform, so it is recorded as it is.
            if self.smearing.enabled and self.smearing.periodic:
                gamma = self.smearing.kernel(monomer_pair[0])
                sf_scaled = np.divide(sf_scaled, gamma, out=np.zeros_like(sf_scaled), where=gamma > 1e-12)
            mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_real"] = np.real(sf_scaled)
            mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1] + "_imag"] = np.imag(sf_scaled)
        # Structure function of the smeared charge density c = Σ_i z_i Γ_i φ_i: <M> + w <D>
        # with the weight w = -Cov(M, D)/Var(D) of least variance at each wavevector (w = 0:
        # the mixed form, w = 1: the field-field form). Both forms are recorded too: in the
        # Gaussian regime the combination follows from the linear response of the chains
        # and does not show how well ψ is sampled (docs/theory/StructureFunctionEstimators.md)
        if "charge_mixed" in sums:
            count = max(sf_sample_count, 1)
            mixed = sums["charge_mixed"] / count
            difference = sums["charge_difference"] / count
            covariance = sums["charge_mixed_difference"].real / count - mixed.real * difference.real
            variance = sums["charge_difference_squared"].real / count - difference.real ** 2
            weight = np.divide(-covariance, variance, out=np.zeros_like(variance), where=variance > 0)
            s_charge = mixed + weight * difference
            mdic["charge_structure_function_real"] = np.real(s_charge)
            mdic["charge_structure_function_imag"] = np.imag(s_charge)
            mdic["charge_structure_function_weight"] = weight
            for form, s_form in (("mixed", mixed), ("field_field", mixed + difference)):
                mdic["charge_structure_function_" + form + "_real"] = np.real(s_form)
                mdic["charge_structure_function_" + form + "_imag"] = np.imag(s_form)
        # Mean concentrations on the grid, shaped like nx
        for name in self.sim.concentration_names:
            mean = np.reshape(sums["concentration_" + name], self.prop_solver.nx) / max(sums["concentration steps"][0], 1)
            mdic["concentration_" + name + "_real"] = np.real(mean)
            mdic["concentration_" + name + "_imag"] = np.imag(mean)
        if self.sim.concentration_names:
            mdic["concentration_step_count"] = int(sums["concentration steps"][0])
        mdic["sample_count"] = sf_sample_count
        mdic["simulation_metadata_json"] = json.dumps(self.sim.metadata)
        atomic_savemat(
            os.path.join(self.sim.recording["dir"], "structure_function_%06d.mat" % langevin_step),
            mdic, long_field_names=True, do_compression=True
        )
