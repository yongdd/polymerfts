"""Complex Langevin field-theoretic simulation (CL-FTS).

All auxiliary fields are complex and fluctuate: real noise on the exchange
fields, imaginary noise on the pressure field, with optional dynamical
stabilization (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024)),
smearing (Matsen et al., J. Chem. Phys. 164, 014905 (2026)) and smeared
Coulomb interactions through a fluctuating electrostatic potential
(electrostatics.py). Parameters are documented in
docs/references/Parameters.md; complete scripts are examples/block_copolymer/clfts/Lamella.py,
examples/charged/clfts/ and examples/blend/clfts/GibbsEnsemble.py.
"""

import os
import time
import warnings
import copy
import contextlib
import numpy as np
from .integrators import checked_fields
from .force_jacobian import force_jacobian, symmetric_jacobian
from .thermodynamics import ThermodynamicOperators
from .diagnostics import RunDiagnostics

from ..theory.symmetric_formulation import SymmetricPolymerTheory
from ..theory.chains import (read_chains, chain_solver, chain_fields, solve_chains, check_partition_once,
                             prepare_run, chain_hamiltonian)
from ..utils.file_formats import load_config
from ..theory.smearing import smearing_of
from .structure_functions import ClftsStructureFunctions
from ..theory.electrostatics import Electrostatics, display_point_ions
from ..utils.validation import (with_overrides, validate_clfts_params, ValidationError, NumericalError,
                         point_ion_concentrations, open_charges, reference_segments, finite_values,
                         initial_field_values, previous_noise_values, first_langevin_step)
from ..utils.common import (
    display_system,
    calculate_sigma,
    atomic_savemat,
    simulation_metadata,
    notify_callbacks,
)
from ..utils.checkpoint import (
    RunRecord,
    save_statistics,
    fields_data,
    load_checkpoint,
    recording_run_status,
    initialize_extensions,
    extension_state,
    restore_extensions,
)


class CLFTS:
    """Complex Langevin field-theoretic simulation.

    Parameters
    ----------
    params : dict
        Simulation parameters; see docs/references/Parameters.md (CL-FTS
        section). Required: the SCFT system keys plus langevin, recording,
        verbose_level. zeta_n defaults to 100.
    random_seed : int, optional
        Seed of the PCG64 noise generator (None: non-reproducible).
    integrator : object, optional
        Langevin step implementation; otherwise langevin.integrator (default "lm").
        See docs/internals/Extensions.md for the step and checkpoint contracts.
    callbacks : dict, optional
        Stable names to callable observers of accepted, read-only step snapshots.

    Attributes
    ----------
    prop_solver : PropagatorSolver
        Complex-field propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    alpha_ds : float
        Dynamical stabilization constant.

    Examples
    --------
    >>> sim = CLFTS(params)
    >>> sim.run(initial_fields={"A": w_A, "B": w_B})
    """

    # run() draws the chain architectures into the recording directory unless this is False (on the
    # object, or on the class for every simulation built afterwards)
    draw_architectures = True

    @classmethod
    def from_file(cls, path, random_seed=None, *, integrator=None, callbacks=None, **overrides):
        """Create a simulation from a YAML or JSON parameter file.

        Reads the file with load_config, applies the keyword overrides
        and calls the constructor, as SCFT.from_file does.
        """
        params = with_overrides(load_config(path), overrides)
        return cls(params, random_seed=random_seed, integrator=integrator, callbacks=callbacks)

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):
        """Initialize the Complex Langevin FTS simulation."""

        # Validate input parameters (a copy with the current names)
        params = validate_clfts_params(params)

        # Compressibility parameter; complex Langevin needs a finite one (default: high compressibility)
        self.zeta_n = params.get("zeta_n", 100.0)

        # Monomer types, segment lengths, polymer chains, χN and random copolymers
        self.monomer_types, self.segment_lengths, self.polymers, self.chi_n, self.random_fraction = read_chains(params)

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

        # Complex-valued propagator solver (complex Langevin fields) with the polymers. Grand-canonical
        # polymers (chemical_potential βμ_p) get the volume fraction 0: their chains, e^(βμ_p) Q_p per
        # reference chain, come from get_concentration_gce (compute_concentrations)
        self.prop_solver = chain_solver(params, self.segment_lengths, self.polymers, field_type="complex")
        self.polymer_chemical_potentials = np.array(
            [polymer.get("chemical_potential", np.nan) for polymer in self.polymers], dtype=np.float64)
        # Fugacities e^(βμ_p), zero for canonical chains (validation.py bounds |βμ_p| so that they
        # are normal doubles)
        self.fugacities = np.where(np.isnan(self.polymer_chemical_potentials), 0.0,
                                   np.exp(self.polymer_chemical_potentials))
        self.grand_canonical = [p for p, polymer in enumerate(self.polymers) if "chemical_potential" in polymer]
        self.canonical_fraction = sum(polymer.get("volume_fraction", 0.0) for polymer in self.polymers)

        # Computation box as a public attribute (display_statistics() below creates the propagator computation)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Settings of the run: the Langevin step with the standard deviation of its noise, the scaling
        # dt_scaling[i] of dH/dw_aux[i], the dynamical stabilization, what is printed and what is recorded
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin["sigma"] = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"],
                                                 np.prod(params["nx"]), self.prop_solver.get_volume())
        self.dt_scaling = self.mpt.dt_scaling()
        # Dynamical stabilization constant for complex Langevin simulations
        # This prevents runaway solutions ("hot spots") by adding iαds·Im[W-] to exchange field forces.
        # "dynamic_stabilization" is the combined coefficient αds (as in Willis & Matsen 2024);
        # normalize_params has renamed its alias "alpha_ds", kept for compatibility with reference codes.
        # Typical values: 0.01-0.1 for N̄ ≥ 10^5 (Willis & Matsen 2024)
        self.alpha_ds = params.get("dynamic_stabilization", 0.0)
        self.verbose_level = params["verbose_level"]
        self.recording = params["recording"].copy()

        # Random bit generator of the noise (random_seed None: non-reproducible)
        self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)

        # Smearing of the interactions
        self.smearing = smearing_of(params, self.prop_solver)
        self._periodic_box = all(b == "periodic" for b in self.prop_solver.bc_lower)   # S(k) transforms with fftn

        # CL-FTS has no WTMD bias
        self.wtmd = None

        # Smeared Coulomb interactions (optional): the electrostatic potential ψ, felt by
        # charged types as z_i ψ, fluctuates next to the auxiliary fields (electrostatics.py)
        self.electrostatics = None
        if params.get("electrostatics", None) is not None:
            self._setup_electrostatics(params)
        for p in self.grand_canonical:
            print("Polymer %d: grand canonical, chemical potential %g" % (p, self.polymer_chemical_potentials[p]))

        # Structure functions and thermodynamic operators
        self.structure_functions = ClftsStructureFunctions(self)
        self.thermodynamics = ThermodynamicOperators(self)

        display_system(self)

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (self.langevin["sigma"]))
        print("Scaling factor of delta tau N for each field:", self.dt_scaling)
        print("Dynamical stabilization constant: %f" % (self.alpha_ds))
        print("Random Number Generator:", self.random_bg.state)

        self.prop_solver.display_statistics()

        # The metadata of the files, and the integrator and the callbacks (docs/internals/Extensions.md)
        self.metadata = simulation_metadata(self, random_seed)
        initialize_extensions(self, integrator if integrator is not None else self.langevin.get("integrator", "lm"), callbacks)
        self._prepare_integrator()
        self.run_status = {"status": "initialized", "last_completed_step": 0}
        # The state of the integrator and the callbacks after the setup, which a fresh run restores: a
        # stateful integrator may hold the response that _prepare_integrator gave it
        self._initial_extension_state = extension_state(self)

    def _setup_electrostatics(self, params):
        """The electrostatic potential ψ and what the structure functions need of it."""
        self.electrostatics = Electrostatics(params["electrostatics"], self.monomer_types, self.smearing,
                                             reference_segments(params), self.langevin, self.random_bg.jumped(),
                                             fourier=self.prop_solver.fourier,
                                             ion_concentrations=point_ion_concentrations(params),
                                             open_charges=open_charges(params))
        print("Electrostatics: charges", self.electrostatics.charges,
              "coupling E = %g, psi_dt_scaling = %g" % (self.electrostatics.coupling, self.electrostatics.psi_dt_scaling))
        display_point_ions(self.electrostatics.ions)

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

        NOTE: this takes AUXILIARY fields (exchange-mapped), unlike
        SCFT.compute_concentrations which takes monomer potential fields.

        Parameters
        ----------
        w_aux : numpy.ndarray
            Complex auxiliary fields array of shape (M, n_grid).

        Returns
        -------
        phi : dict
            Dictionary of concentration fields for each monomer type.
        elapsed_time : dict
            Dictionary of timing information.
        """
        # The fields of the chains: charged types feel the electrostatic potential, w_i + z_i ψ
        w_input = chain_fields(self, self.mpt.to_monomer_fields(w_aux))
        if self.electrostatics is not None:
            self.electrostatics.add_potential(w_input)

        phi, elapsed_time = solve_chains(self, w_input)

        # Material conservation of the canonical chains: mean(sum phi) stays at their total volume
        # fraction, 1 without grand-canonical chains (a numerical blow-up shows up here first)
        mass_error = self.prop_solver.mean(sum(phi[m] for m in self.monomer_types) - self.canonical_fraction)
        if not abs(mass_error) <= 5e-8:
            raise NumericalError("Material conservation failed: mean(sum(phi)) differs from the total volume "
                                 f"fraction of the canonical chains by {mass_error:.3e}",
                                 self.mpt.to_monomer_fields_dict(w_aux))

        # Grand-canonical chains: e^(βμ_p) times their unnormalized segment densities
        for p in self.grand_canonical:
            for monomer_type in self.monomer_types:
                phi[monomer_type] = phi[monomer_type] + self.prop_solver.get_concentration_gce(
                    self.fugacities[p], p, monomer_type)

        check_partition_once(self, lambda: self.mpt.to_monomer_fields_dict(w_aux))
        return phi, elapsed_time

    def _prepare_integrator(self):
        """What the integrator needs of this simulation: the noise scheme of ψ and the linear response."""
        # ψ takes the noise scheme the integrator declares (electrostatics.py): the average of
        # two noises like Leimkuhler-Matthews, or fresh noise like Euler
        if self.electrostatics is not None:
            averaged_noise = getattr(self.integrator, "averaged_noise", None)
            if not isinstance(averaged_noise, (bool, np.bool_)):
                raise ValidationError("With 'electrostatics' the integrator must declare averaged_noise: True if "
                                      "it averages the current and the previous noise (Leimkuhler-Matthews), "
                                      "False if it takes fresh noise")
            self.electrostatics.averaged_noise = bool(averaged_noise)

        # An integrator that declares linear_response (the built-in "etd") integrates the linear
        # response of the homogeneous state exactly; it gets the Jacobian of the forces and, with
        # Coulomb coupling, also advances ψ as the last field. It holds the response of this
        # simulation, so each simulation keeps its own copy of the integrator
        self._joint_psi = False
        if getattr(self.integrator, "linear_response", False):
            if any(b != "periodic" for b in self.prop_solver.bc_lower):
                raise ValidationError(f"The integrator '{self.integrator.name}' (linear_response) needs a periodic box")
            self.integrator = copy.deepcopy(self.integrator)
            self._joint_psi = self.electrostatics is not None and self.electrostatics.coupling > 0
            mobility, direction = self._integrated_mobility(), self._integrated_noise_direction()
            jacobian = symmetric_jacobian(self, force_jacobian(self, self._joint_psi), direction, mobility)
            self.integrator.set_linear_response(jacobian, dt=self.langevin["dt"], mobility=mobility,
                                                noise_direction=direction, fourier=self.prop_solver.fourier)

    def _integrated_mobility(self):
        """Mobilities of the fields the integrator advances: dt_scaling, and that of ψ when joint."""
        if self._joint_psi:
            return np.append(self.dt_scaling, self.electrostatics.psi_dt_scaling)
        return self.dt_scaling.copy()

    def _integrated_noise_direction(self):
        """Real noise for the real-type auxiliary fields, imaginary for the others (and ψ)."""
        direction = np.array([1.0 if i in self.mpt.aux_fields_real_idx else 1j for i in range(len(self.monomer_types))])
        return np.append(direction, 1j) if self._joint_psi else direction

    def evaluate_force(self, fields):
        """The forces at the intermediate fields of a step (the evaluate_force of the integrator).

        `fields` holds the auxiliary fields, and with electrostatics may hold ψ as one more row (when the
        integrator advances ψ, linear_response). The forces are δH/δw of the auxiliary fields from the
        smeared concentrations, with the stabilization (_stabilized), and -δH/δψ in the row of ψ.
        compute_func_deriv transforms the concentrations through the matrix A: for an AB diblock the
        exchange field w- gets phi_A - phi_B and the pressure field w+ gets phi_A + phi_B (Matsen et al.,
        J. Chem. Phys. 164, 014905 (2026)).
        """
        fields = np.asarray(fields, dtype=np.complex128)
        M, es = len(self.monomer_types), self.electrostatics
        with_psi = es is not None and fields.shape == (M + 1, self.prop_solver.n_grid)
        if not with_psi and fields.shape != (M, self.prop_solver.n_grid):
            raise ValidationError("Intermediate CLFTS fields have an invalid shape")
        if not np.all(np.isfinite(fields)):
            raise NumericalError("Intermediate CLFTS fields have non-finite values",
                                 self.mpt.to_monomer_fields_dict(fields[:M]))
        with es.trial(fields[M]) if with_psi else contextlib.nullcontext():
            phi, _ = self.compute_concentrations(fields[:M])
            phi_smeared = self.smearing.apply_to_dict(phi)
            force = self._stabilized(fields[:M], self.mpt.compute_func_deriv(fields[:M], phi_smeared, range(M)))
            if with_psi:
                force = np.vstack([force, -es.force(es.charge_density(phi_smeared))])
        if not np.all(np.isfinite(force)):
            raise NumericalError("Non-finite intermediate CLFTS force", self.mpt.to_monomer_fields_dict(fields[:M]))
        return force

    def save_simulation_data(self, path, w_aux, phi, langevin_step, normal_noise_prev, record):
        """Save CL-FTS simulation checkpoint data to MATLAB file.

        Parameters
        ----------
        path : str
            Output file path.
        w_aux : numpy.ndarray
            Auxiliary fields; the file holds them and their monomer potential fields.
        phi : dict
            Concentration fields.
        langevin_step : int
            Current Langevin step number.
        normal_noise_prev : numpy.ndarray
            Previous noise array for Leimkuhler-Matthews integration.
        record : RunRecord
            The measurements of the run since its last statistics file.
        """
        # Make dictionary for data
        mdic = fields_data(self, w_aux, langevin_step, normal_noise_prev, record)
        mdic["zeta_n"] = self.zeta_n

        # Add w fields to the dictionary (complex values)
        w = self.mpt.to_monomer_fields(w_aux)
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name + "_real"] = np.real(w[i])
            mdic["w_" + name + "_imag"] = np.imag(w[i])

        # Add concentrations to the dictionary (complex values)
        for name in self.monomer_types:
            mdic["phi_" + name + "_real"] = np.real(phi[name])
            mdic["phi_" + name + "_imag"] = np.imag(phi[name])

        # Electrostatic potential and its noise stream
        if self.electrostatics is not None:
            mdic.update(self.electrostatics.checkpoint_data())

        # Save data with matlab format
        atomic_savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name, *, allow_legacy=False):
        """Continue simulation from a saved checkpoint.

        Parameters
        ----------
        file_name : str
            Path to the checkpoint file.
        """
        # Load data
        load_data = load_checkpoint(self, file_name, allow_legacy)
        if self.electrostatics is not None:
            self.electrostatics.restore(load_data)

        # Make initial_fields (complex)
        initial_fields = {}
        for name in self.monomer_types:
            w_real = np.array(load_data["w_" + name + "_real"])
            w_imag = np.array(load_data["w_" + name + "_imag"])
            initial_fields[name] = w_real + 1j * w_imag

        # Run
        return self.run(
            initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"] + 1, _restart_data=load_data
        )

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, initial_psi=None,
            psi_noise_prev=None, _restart_data=None):
        """Run complex Langevin dynamics and return a status dict; errors are persisted and raised."""
        # The initial fields and ψ (real or complex finite numbers) and the previous noises of the fields
        # and ψ (real) are checked before the run starts
        fields = initial_field_values(initial_fields, self.monomer_types, self.prop_solver.n_grid, np.complex128)
        if initial_psi is not None:
            if self.electrostatics is None:
                raise ValidationError("initial_psi needs 'electrostatics'")
            initial_psi = finite_values(initial_psi, "initial_psi", np.complex128)
            if initial_psi.size != self.prop_solver.n_grid:
                raise ValidationError("initial_psi must hold one finite value per grid point")
            # Without Coulomb coupling ψ does not move, so a nonzero ψ would be a fixed potential
            if self.electrostatics.coupling <= 0 and np.any(initial_psi != 0):
                raise ValidationError("initial_psi must be zero without Coulomb coupling (coupling E = 0)")
        if psi_noise_prev is not None and initial_psi is None:
            raise ValidationError("psi_noise_prev needs initial_psi")
        psi_noise_prev = previous_noise_values(psi_noise_prev, "psi_noise_prev", 1, self.prop_solver.n_grid,
                                               "electrostatic potential")[0]
        normal_noise_prev = previous_noise_values(normal_noise_prev, "normal_noise_prev", len(self.monomer_types),
                                                  self.prop_solver.n_grid, "auxiliary fields")
        start_langevin_step = first_langevin_step(start_langevin_step)
        # CL-FTS solves no saddle points: a SaddlePointError (of an extension) fails the run like other errors
        with recording_run_status(self, start_langevin_step, fields, _restart_data, saddle_search=False):
            if _restart_data is None:
                restore_extensions(self)
            self._run(fields, normal_noise_prev, start_langevin_step, initial_psi=initial_psi,
                      psi_noise_prev=psi_noise_prev, _restart_data=_restart_data)
        return dict(self.run_status)

    def _run(self, initial_fields, normal_noise_prev, start_langevin_step, *, initial_psi, psi_noise_prev,
             _restart_data):
        """Run the Complex Langevin FTS simulation with the arguments that run() has checked.

        Parameters
        ----------
        initial_fields : dict
            Initial potential fields for each monomer type (complex arrays).
        normal_noise_prev : numpy.ndarray
            Previous noise of the auxiliary fields, shape (M, n_grid), for continuing a simulation.
        start_langevin_step : int
            Starting step number.
        initial_psi : numpy.ndarray or None
            Initial electrostatic potential (with `electrostatics`), e.g. psi_real + 1j psi_imag
            of a fields file; zero if None. A continuation by hand does not restore the
            random-number streams; continue_run() continues a checkpoint exactly.
        psi_noise_prev : numpy.ndarray
            Previous noise of ψ (psi_noise_prev of a fields file) for the Leimkuhler-Matthews step.
        """
        print("---------- Run ----------")

        # The number of components
        M = len(self.monomer_types)

        # The architecture drawings and the complex auxiliary fields to start from (chains.prepare_run)
        w_aux, restored = prepare_run(self, initial_fields, _restart_data, np.complex128)

        # The electrostatic potential: continue_run() has restored it with its noise stream
        if self.electrostatics is not None and _restart_data is None:
            self._start_psi(initial_psi, psi_noise_prev, start_langevin_step)

        # Measurements of the run, with the pending ones of a checkpoint
        record = self._start_record()
        record.restore(_restart_data, start_langevin_step)

        # Summaries of this run printed at the end (diagnostics.py): the distribution of the largest drift
        # of every step and the partition ratios of the single-site species
        diagnostics = RunDiagnostics(self)

        # Init timers
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step

        # Concentrations of the initial fields. Afterwards they are computed right after
        # each update, so that the measurements of a step (H, dH/dχN, S(k), saved φ) and
        # the forces of the next step use the concentrations of the same fields.
        # Smearing implements finite-range interactions: φ_Γ = FFT⁻¹[FFT[φ]·Γ(k)]
        # Reference: Matsen et al., J. Chem. Phys. 164, 014905 (2026)
        if restored:
            phi = {m: (np.asarray(_restart_data["phi_" + m + "_real"]) +
                       1j * np.asarray(_restart_data["phi_" + m + "_imag"])).reshape(self.prop_solver.n_grid)
                   for m in self.monomer_types}
        else:
            phi, _ = self.compute_concentrations(w_aux=w_aux)
        phi_for_force = self.smearing.apply_to_dict(phi)

        # Charge density of the current state and the force of ψ, for the drift record, the Poisson
        # residual of the progress line and the next step
        es = self.electrostatics
        charge_density = None if es is None else es.charge_density(phi_for_force)
        psi_force = None if es is None else es.force(charge_density)

        # δH/δw of the current state (compute_func_deriv), for the forces of the next step and the
        # progress line
        h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))

        for langevin_step in range(start_langevin_step, self.langevin["max_step"] + 1):
            step_start = time.perf_counter()

            # Forces on all fields and the largest drift of each
            w_lambda = self._stabilized(w_aux, h_deriv)
            field_drift_max = np.max(np.abs(w_lambda * self.dt_scaling[:, None]), axis=1)
            drift_max = float(np.max(field_drift_max))
            record.add("drift_step_history", field_drift_max)

            # Largest drift of ψ, s |δH/δψ|, recorded apart from drift_max: its linear part ∇²ψ/E
            # is integrated exactly and would dominate the maximum at large k
            psi_drift_max = None
            if es is not None:
                psi_drift_max = float(np.max(np.abs(es.psi_dt_scaling * psi_force)))
                record.add("psi_drift_step_history", psi_drift_max)

            # |Q_s|/<|e^(-u_s)|> of the single-site species, and the summaries of the drifts and the ratios
            ratios = None
            if self.thermodynamics.partition_ratio_species:
                ratios = self.thermodynamics.partition_ratios(w_aux)
                record.add("partition_ratio_step_history", ratios)
            diagnostics.add(field_drift_max, psi_drift_max, ratios)

            # One Langevin step of the fields and ψ
            noise = self.random.normal(0.0, self.langevin["sigma"], [M, self.prop_solver.n_grid])
            w_aux = self._advance_fields(w_aux, w_lambda, noise, normal_noise_prev, psi_force, charge_density,
                                         langevin_step)
            normal_noise_prev = noise

            # Concentrations and partition functions of the updated fields (the step has checked that the
            # fields are finite, checked_fields; the grand-canonical chains add to the concentrations)
            phi, _ = self.compute_concentrations(w_aux=w_aux)
            phi_for_force = self.smearing.apply_to_dict(phi)
            if not all(np.all(np.isfinite(v)) for v in phi.values()):
                raise NumericalError(f"Non-finite concentrations at step {langevin_step}",
                                     self.mpt.to_monomer_fields_dict(w_aux))
            if es is not None:
                charge_density = es.charge_density(phi_for_force)
                psi_force = es.force(charge_density)

            # Hamiltonian of the updated fields
            hamiltonian, total_partitions = self._hamiltonian(w_aux, phi_for_force)
            if not np.isfinite(hamiltonian):
                raise NumericalError(f"Non-finite Hamiltonian at step {langevin_step}", self.mpt.to_monomer_fields_dict(w_aux))
            self.run_status["last_completed_step"] = langevin_step

            # δH/δw of the updated fields and the progress line (functional derivatives with the smeared φ,
            # for consistency)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_force, range(M))
            error_level_array = np.std(h_deriv, axis=1)
            if self.verbose_level >= 1:
                self._print_progress(langevin_step, h_deriv, total_partitions, hamiltonian, error_level_array, psi_force)

            # Mean concentrations of recording.concentrations: every step, since φ and c_s of the accepted
            # state are at hand and the fast modes of ψ decorrelate within a few steps
            self.structure_functions.add_concentrations(phi, record.sums)

            # Measurements of the accepted state: H, the largest drifts, dH/dχN, the thermodynamic
            # operators and S(k)
            if langevin_step % self.recording["sample_interval"] == 0:
                record.sample_count += 1
                record.add("H_history", hamiltonian)
                record.add("drift_history", drift_max)
                if es is not None:
                    record.add("psi_drift_history", psi_drift_max)
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=phi_for_force)
                for key in self.chi_n:
                    record.add("dH " + key, dH[key])
                record.add("chemical_potential_history", self.thermodynamics.chemical_potentials(total_partitions))
                if self.thermodynamics.record_number_densities:
                    record.add("number_density_history", self.thermodynamics.number_densities(total_partitions))
                if self.thermodynamics.record_pressure:
                    record.add("pressure_history", self.thermodynamics.pressure_virial(w_aux, phi, h_deriv, charge_density))
                self.structure_functions.add(w_aux, phi, record.sums)

            # Statistics files: H and dH/dχN, then S(k) and the mean concentrations
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                save_statistics(self, langevin_step, record, diagnostics.statistics_entries)

            # Observe an accepted state before saving its callback/integrator state.
            notify_callbacks(self, langevin_step, w_aux, phi, hamiltonian, drift_max,
                             error_level_array, time.perf_counter() - step_start)

            # Save simulation data
            if langevin_step % self.recording["checkpoint_interval"] == 0:
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % langevin_step),
                    w_aux=w_aux, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev,
                    record=record
                )

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration / n_steps))
        diagnostics.report()

    def _start_psi(self, initial_psi, psi_noise_prev, first_step):
        """ψ of a new run: initial_psi or zero, with the given previous noise (zero without one)."""
        es = self.electrostatics
        if initial_psi is not None:
            es.psi = initial_psi
        else:
            if first_step > 1 and es.coupling > 0:
                warnings.warn("The run continues without initial_psi: the electrostatic potential restarts "
                              "at zero and relaxes again", RuntimeWarning)
            es.psi = np.zeros_like(es.psi)
        # run() has checked psi_noise_prev and made it a flat array of its own (previous_noise_values)
        es.previous_noise = psi_noise_prev

    def _stabilized(self, w_aux, h_deriv):
        """The forces of the fields w_aux from their δH/δw (compute_func_deriv). Dynamical stabilization
        (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024)) adds iαds·N·Im[W] to the real-type fields,
        against runaway solutions in the imaginary direction; self.alpha_ds is the combined coefficient αds·N.
        """
        forces = h_deriv.copy()
        if self.alpha_ds > 0:
            for i in self.mpt.aux_fields_real_idx:
                forces[i] += 1j * self.alpha_ds * np.imag(w_aux[i])
        return forces

    def _advance_fields(self, w_aux, forces, noise, noise_prev, psi_force, charge_density, langevin_step):
        """One step of the integrator; returns the new auxiliary fields and moves ψ.

        Multistage methods can evaluate new forces. ψ comes from the step of the integrator or,
        from the charge density of the state before the step, a step of its own with its own noise stream.
        """
        M, es = len(self.monomer_types), self.electrostatics
        fields = w_aux
        if self._joint_psi:
            # The integrator advances ψ with the auxiliary fields (linear_response): ψ is the last
            # field, with the force -δH/δψ, its mobility and imaginary noise from its own stream
            psi_noise = es.draw_noise()
            fields, forces = np.vstack([w_aux, es.psi]), np.vstack([forces, -psi_force])
            noise, noise_prev = np.vstack([noise, psi_noise]), np.vstack([noise_prev, es.previous_noise])
        proposed = self.integrator.step(fields.copy(), forces.copy(), dt=self.langevin["dt"],
                                        mobility=self._integrated_mobility(), noise=noise.copy(),
                                        previous_noise=noise_prev.copy(),
                                        noise_direction=self._integrated_noise_direction(),
                                        evaluate_force=self.evaluate_force)
        proposed = checked_fields(proposed, fields)
        if not np.array_equal(proposed[:M][self.dt_scaling == 0], w_aux[self.dt_scaling == 0]):
            raise ValidationError("An integrator must preserve fields with zero mobility")
        if es is not None:
            if self._joint_psi:
                es.accept(proposed[M], psi_noise)
            else:
                es.step(charge_density)
            if not np.all(np.isfinite(es.psi)):
                raise NumericalError(f"Non-finite electrostatic potential at step {langevin_step}",
                                     self.mpt.to_monomer_fields_dict(proposed[:M]))
        return proposed[:M].copy()

    def _hamiltonian(self, w_aux, phi_smeared):
        """H of the fields, with the electrostatic and grand-canonical terms; returns H and the partition
        functions of the chains."""
        hamiltonian, total_partitions = chain_hamiltonian(self, w_aux, phi_smeared)
        if self.electrostatics is not None:
            hamiltonian += self.electrostatics.hamiltonian()
        if self.grand_canonical:
            hamiltonian += self.thermodynamics.grand_canonical_hamiltonian(total_partitions)
        return hamiltonian, total_partitions

    def _print_progress(self, langevin_step, h_deriv, total_partitions, hamiltonian, error_level_array, psi_force):
        """One line per step: mass error, partition functions, H, error levels and the Poisson residual of ψ."""
        mass_error = self.prop_solver.mean(h_deriv[len(self.monomer_types) - 1])
        print(f"{langevin_step:8d} {mass_error.real:+.3E}{mass_error.imag:+.3E}j", end="  [ ")
        for Q in total_partitions:
            print(f"{Q.real:.6E}{Q.imag:+.6E}j", end=" ")
        print(f"] {hamiltonian.real:+.9E}{hamiltonian.imag:+.9E}j  [", end="")
        for error_level in error_level_array:
            print(f" {error_level:.3E}", end="")
        if psi_force is not None:
            print(f" {np.std(psi_force):.3E}", end="")
        print(" ]")

    def _start_record(self):
        """What a run records (checkpoint.RunRecord)."""
        # Per sample: H, the largest drifts, dH/dχN and the thermodynamic operators (the chemical
        # potentials and the numbers have a column per species). Per step: the largest drift of each
        # auxiliary field and of ψ, and the partition ratios
        series = {"H_history": 0, "drift_history": 0, "drift_step_history": len(self.monomer_types)}
        series |= {"dH " + key: 0 for key in self.chi_n}
        if self.electrostatics is not None:
            series |= {"psi_drift_history": 0, "psi_drift_step_history": 0}
        if self.thermodynamics.partition_ratio_species:
            series["partition_ratio_step_history"] = len(self.thermodynamics.partition_ratio_species)
        series["chemical_potential_history"] = self.thermodynamics.n_chemical_potentials
        if self.thermodynamics.record_number_densities:
            series["number_density_history"] = self.thermodynamics.n_chemical_potentials
        if self.thermodynamics.record_pressure:
            series["pressure_history"] = 0

        # Sums of the structure functions, and with recording.concentrations of the mean concentrations
        return RunRecord(series, self.structure_functions.empty_sums(), complex_values=True)
