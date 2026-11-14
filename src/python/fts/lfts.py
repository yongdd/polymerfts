import os
import time
import pathlib
import copy
import json
import numpy as np
from ..theory.symmetric_formulation import SymmetricPolymerTheory
from ..theory.propagator_solver import PropagatorSolver
from .integrators import checked_fields
from ..utils.validation import normalize_params, validate_lfts_params, ValidationError, SaddlePointError
from ..utils.common import (
    display_system,
    calculate_sigma,
    atomic_savemat,
    simulation_metadata,
    write_run_status,
    notify_callbacks,
)
from ..utils.checkpoint import (
    fields_data,
    auxiliary_field_key,
    load_checkpoint,
    start_run_status,
    RunRecord,
    initialize_extensions,
    restore_extensions,
    pair_suffix,
    checkpoint_pair_suffix,
)
from ..theory.chains import read_chains, mean_concentrations, draw_polymer_architectures, chain_fields, solve_chains, check_partition_once
from ..utils.file_formats import load_config
from .saddle import SaddlePointSearch
from ..theory.smearing import smearing_of
from .structure_functions import LftsStructureFunctions
from .wtmd import WTMD

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")


class LFTS:
    """Langevin field-theoretic simulation (L-FTS) of polymer melts and solutions.

    Samples the field-theoretic partition function with Langevin dynamics
    (Leimkuhler-Matthews scheme) of the real auxiliary fields, while the
    imaginary fields are compressed to their saddle point every step with
    Anderson mixing, linear response, or both (LRAM). Records fields,
    structure functions and d H / d chi N at the requested periods.

    Parameters
    ----------
    params : dict
        Simulation parameters; every key is documented in
        docs/references/Parameters.md. Required: the SCFT system keys plus
        langevin, compressor, saddle, recording, verbose_level.
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
        Propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    langevin, recording : dict
        Copies of the corresponding parameter dictionaries (langevin also holds 'sigma').
    saddle : SaddlePointSearch
        The saddle point of the imaginary fields; saddle.settings is the input "saddle".
    structure_functions : LftsStructureFunctions
        The samples and files of S(k).

    Examples
    --------
    >>> sim = LFTS(params, random_seed=12345)
    >>> sim.run(initial_fields={"A": w_A, "B": w_B})   # writes data/fields_*.mat, structure_function_*.mat
    >>> sim.continue_run("data/fields_001000.mat")

    Complete scripts: examples/block_copolymer/lfts/ (Lamella.py, Gyroid.py) and
    examples/blend/lfts/MixtureBlockRandom.py.
    """
    @classmethod
    def from_file(cls, path, random_seed=None, *, integrator=None, callbacks=None, **overrides):
        """Create a simulation from a YAML or JSON parameter file.

        Reads the file with load_config, applies the keyword overrides
        and calls the constructor, as SCFT.from_file does.
        """
        params = normalize_params(load_config(path))
        params.update(normalize_params(overrides))
        return cls(params, random_seed=random_seed, integrator=integrator, callbacks=callbacks)

    def __init__(self, params, random_seed=None, *, integrator=None, callbacks=None):

        # Validate input parameters
        params = normalize_params(params)
        validate_lfts_params(params)
        for key in ("compressor", "saddle", "recording", "verbose_level"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for LFTS: '{key}'")
        if params["verbose_level"] not in (0, 1, 2):
            raise ValidationError(f"'verbose_level' must be 0, 1 or 2, got {params['verbose_level']}")

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)

        # Monomer types, segment lengths, polymer chains, χN and random copolymers
        self.monomer_types, self.segment_lengths, self.polymers, self.chi_n, self.random_fraction = read_chains(params)

        # Boundary conditions (default: periodic)
        bc = params.get("bc", None)
        if bc is None:
            bc = ["periodic"] * (2 * len(params["nx"]))

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

        # The number of monomer types
        M = len(self.monomer_types)

        # Propagator solver: the C++ computation box, molecules and solver live inside it
        self.prop_solver = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", None),   # None: "rqm4", or "ps" for discrete chains
            platform=params.get("platform", "auto"),   # auto: CPU for 1D, CUDA for 2D/3D when available
            reduce_memory=params.get("reduce_memory", False),
            angles=params.get("angles", None),
            aggregate_propagator_computation=params.get("aggregate_propagator_computation", True),
        )
        for polymer in self.polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))

        # Computation box as a public attribute (this creates the C++ objects)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Initialize smearing (must be after self.cb is set)
        self.smearing = smearing_of(params, self.prop_solver)
        self._periodic_box = all(b.lower() == "periodic" for b in self.prop_solver.bc)   # WTMD and S(k) transform with rfftn

        # Well-Tempered Metadynamics (WTMD), if the input asks for it
        self.wtmd = self._create_wtmd(params)

        # The saddle point of the imaginary fields, with its compressor
        self.saddle = SaddlePointSearch(self, params)

        # Structure functions
        self.structure_functions = LftsStructureFunctions(self)

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), self.prop_solver.get_volume())

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        self.dt_scaling = np.ones(M)
        for i in range(M-1):
            if i not in self.mpt.aux_fields_dropped_idx:
                self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues[:M-1]))
        self.dt_scaling[self.mpt.aux_fields_dropped_idx] = 0.0

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        display_system(self, params["chain_model"])

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        self.prop_solver.display_statistics()

        #  Save internal variables
        self.params = copy.deepcopy(params)
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.recording = params["recording"].copy()
        self.metadata = simulation_metadata(self, random_seed)
        initialize_extensions(self, integrator if integrator is not None else self.langevin.get("integrator", "lm"), callbacks)
        # The linear response of the homogeneous state (the "etd" integrator) is measured by CLFTS only
        if getattr(self.integrator, "linear_response", False):
            raise ValidationError(f"The integrator '{self.integrator.name}' (linear_response) is supported in CL-FTS only")
        self.run_status = {"status": "initialized", "last_completed_step": 0}

    def _create_wtmd(self, params):
        # Well-Tempered Metadynamics (WTMD), if parameters are provided. The WTMD class
        # follows deep-langevin-fts; the input names differ: "ell" and "delta_t" here for its
        # "l" and "dT"
        wtmd_params = params.get("wtmd", None)
        if wtmd_params is not None:
            wtmd = WTMD(
                nx=list(self.cb.get_nx()),
                lx=list(self.cb.get_lx()),
                angles=self.prop_solver.get_angles_degrees(),
                nbar=params["langevin"]["nbar"],
                eigenvalues=self.mpt.eigenvalues,
                real_fields_idx=self.mpt.aux_fields_real_idx,
                l=wtmd_params.get("ell", 4),
                kc=wtmd_params.get("kc", 6.02),
                dT=wtmd_params.get("delta_t", 5.0),
                sigma_psi=wtmd_params.get("sigma_psi", 0.16),
                psi_min=wtmd_params.get("psi_min", 0.0),
                psi_max=wtmd_params.get("psi_max", 10.0),
                dpsi=wtmd_params.get("dpsi", 1e-3),
                update_freq=wtmd_params.get("update_freq", 1000),
                recording_period=wtmd_params.get("recording_period", 100000),
                fourier=self.prop_solver.fourier if self._periodic_box else None,
            )
            print("WTMD enabled: l=%d, sigma_psi=%.4f, dT=%.4f, kc=%.4f, exchange_idx=%d, langevin_idx=%d" % (
                wtmd.l, wtmd.sigma_psi, wtmd.dT, wtmd.kc, wtmd.exchange_idx, wtmd.langevin_idx))
            return wtmd
        return None

    def compute_concentrations(self, w_aux):
        """Compute monomer concentration fields from auxiliary fields.

        NOTE: this takes AUXILIARY fields (exchange-mapped), unlike
        SCFT.compute_concentrations which takes monomer potential fields.

        Converts auxiliary fields to monomer potential fields via inverse
        field transformation, then computes chain propagators and monomer
        concentrations.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid) where M is
            the number of auxiliary fields (= number of monomer types).

        Returns
        -------
        phi : dict
            Monomer concentration fields, {monomer_type: concentration_array}.
        elapsed_time : dict
            Timing information:

            - 'solver' : float
                Time (seconds) for propagator computation.
            - 'phi' : float
                Time (seconds) for concentration integration.
        """
        phi, elapsed_time = solve_chains(self, chain_fields(self, self.mpt.to_monomer_fields(w_aux)))

        # Material conservation: mean(sum phi) must stay 1 (a numerical blow-up shows up here first)
        mass_error = self.prop_solver.mean(sum(phi[m] for m in self.monomer_types) - 1.0)
        if not abs(mass_error) <= 5e-8:
            raise ValidationError(f"Material conservation failed: mean(sum(phi) - 1) = {mass_error:.3e}")

        check_partition_once(self)
        return phi, elapsed_time

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev, w_aux=None):
        """Save L-FTS simulation checkpoint data to MATLAB file.

        Saves the current state of the simulation including fields, concentrations,
        random generator state, and all parameters needed to continue the simulation.

        Parameters
        ----------
        path : str
            Output file path (should end with '.mat').
        w : ndarray
            Current monomer potential fields, shape (M, total_grid).
        phi : dict
            Current monomer concentration fields, {monomer_type: array}.
        langevin_step : int
            Current Langevin iteration number.
        normal_noise_prev : ndarray
            Previous Langevin noise (needed for Leimkuhler-Matthews method),
            shape (R, total_grid) where R is number of real auxiliary fields.
        """

        # Make dictionary for data
        mdic = fields_data(self, w, langevin_step, normal_noise_prev, w_aux)

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = phi[name]

        # Add WTMD state if enabled (same format as deep-langevin-fts)
        if self.wtmd is not None:
            mdic["wtmd_u"] = self.wtmd.u * self.wtmd.CV
            mdic["wtmd_up"] = self.wtmd.up * self.wtmd.CV
            mdic["wtmd_I0"] = self.wtmd.I0
            # The same arrays without the factor CV, which a continued run restores exactly:
            # (x*CV)/CV can differ from x in the last bit
            mdic["wtmd_unscaled_u"] = self.wtmd.u
            mdic["wtmd_unscaled_up"] = self.wtmd.up
            # I1 and the samples stored since the last update, so that a continued run
            # carries the same statistics as an uninterrupted one
            for key, value in self.wtmd.I1.items():
                suffix = pair_suffix(key)
                mdic["wtmd_I1_" + suffix] = value * self.wtmd.CV
                mdic["wtmd_unscaled_I1_" + suffix] = value
            mdic["wtmd_order_parameter_history"] = np.array(self.wtmd.order_parameter_history, dtype=np.float64)
            for key, values in self.wtmd.dH_history.items():
                mdic["wtmd_dH_history_" + pair_suffix(key)] = np.array(values, dtype=np.float64)
            mdic["wtmd_params"] = {
                "l": self.wtmd.l,
                "sigma_psi": self.wtmd.sigma_psi,
                "dT": self.wtmd.dT,
                "kc": self.wtmd.kc,
                "psi_min": self.wtmd.psi_min,
                "psi_max": self.wtmd.psi_max,
                "dpsi": self.wtmd.dpsi,
                "bins": self.wtmd.bins,
                "update_freq": self.wtmd.update_freq,
                "recording_period": self.wtmd.recording_period,
            }

        # Save data with matlab format
        atomic_savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name, *, allow_legacy=False):
        """Continue L-FTS simulation from a saved checkpoint file.

        Loads simulation state from a MATLAB .mat file (created by save_simulation_data)
        and continues the Langevin dynamics from where it left off, preserving the
        random number generator state for reproducibility.

        Parameters
        ----------
        file_name : str
            Path to checkpoint file (MATLAB .mat format created by save_simulation_data).
        """

        # Load_data
        load_data = load_checkpoint(self, file_name, allow_legacy)

        # Restore WTMD state if available: the arrays without the factor CV, or for an older
        # checkpoint the scaled ones divided by CV (as in deep-langevin-fts)
        if self.wtmd is not None and "wtmd_u" in load_data:
            def bias(name):
                if "wtmd_unscaled_u" in load_data:
                    return np.array(load_data["wtmd_unscaled_" + name])
                return np.array(load_data["wtmd_" + name]) / self.wtmd.CV
            self.wtmd.u = bias("u")
            self.wtmd.up = bias("up")
            if "wtmd_I0" in load_data:
                self.wtmd.I0 = np.array(load_data["wtmd_I0"])
            n_pending = np.atleast_1d(load_data.get("wtmd_order_parameter_history", np.zeros(0))).size
            self.wtmd.order_parameter_history = list(np.atleast_1d(load_data["wtmd_order_parameter_history"])) if n_pending > 0 else []
            self.wtmd.I1 = {}
            self.wtmd.dH_history = {}
            for key in self.chi_n:
                name = checkpoint_pair_suffix(load_data, key)
                if "wtmd_I1_" + name in load_data:
                    self.wtmd.I1[key] = bias("I1_" + name)
                if n_pending > 0 and "wtmd_dH_history_" + name in load_data:
                    self.wtmd.dH_history[key] = list(np.atleast_1d(load_data["wtmd_dH_history_" + name]))
            print("Restored WTMD state")

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(load_data["w_" + name])

        # Run
        return self.run(initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1, _restart_data=load_data)

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, _restart_data=None):
        """Run Langevin dynamics, returning a status dict; failures raise and persist run_status.json.

        ``saddle.failure_policy='retry'`` (default) retains the proposed real fields
        and noise while continuing the saddle solve. ``'abort'`` stops immediately;
        ``'resample'`` explicitly opts into the former noise-selection policy.
        ``saddle.max_retries`` defaults to five. Failed steps are never recorded.
        """
        # The histogram, the Gaussian kernels and the integrals all use the bin width dpsi.
        # Checked unless WTMD statistics are restored, so that a run started with another
        # dpsi still continues
        if self.wtmd is not None and (_restart_data is None or "wtmd_u" not in _restart_data):
            bins = (self.wtmd.psi_max - self.wtmd.psi_min) / self.wtmd.dpsi
            if abs(bins - round(bins)) > 1e-9 * bins:
                raise ValidationError("wtmd.dpsi must divide psi_max - psi_min into a whole number of bins")
        start_run_status(self, start_langevin_step, initial_fields, _restart_data, saddle_failures=0, noise_resamples=0)
        write_run_status(self)
        try:
            if _restart_data is None:
                restore_extensions(self)
                if self.wtmd is not None and start_langevin_step in (None, 1):
                    self.wtmd.reset()
            self._run(initial_fields, normal_noise_prev, start_langevin_step, _restart_data=_restart_data)
            self.run_status["status"] = "completed"
        except SaddlePointError as exc:
            self.run_status.update(status="aborted_saddle_failure", reason=str(exc))
            raise
        except (KeyboardInterrupt, SystemExit):
            self.run_status.update(status="interrupted", reason="Run interrupted")
            raise
        except Exception as exc:
            self.run_status.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_run_status(self)
        return dict(self.run_status)

    def _run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None, *, _restart_data=None):
        """Run Langevin Field-Theoretic Simulation (L-FTS).

        Performs Langevin dynamics to sample compositional fluctuations in the
        field-theoretic representation of the polymer system. At each Langevin
        step:

        1. Computes functional derivative δH/δw for real auxiliary fields
        2. Updates real fields via Leimkuhler-Matthews method with Gaussian noise
        3. Finds saddle point of imaginary auxiliary fields via compression
        4. Records fields, Hamiltonian, and structure function data
        5. Saves checkpoint periodically

        The simulation continues for langevin.max_step iterations.

        Parameters
        ----------
        initial_fields : dict
            Initial monomer potential fields, {monomer_type: field_array}.
            Typically initialized from SCFT saddle point for faster equilibration.
            Each array has length total_grid.
        normal_noise_prev : ndarray, optional
            Previous Langevin noise array for Leimkuhler-Matthews method,
            shape (R, total_grid) where R = number of real auxiliary fields.
            Default: None (initialized as zeros). Used when continuing from checkpoint.
        start_langevin_step : int, optional
            Starting Langevin iteration number. Default: 1.
            Used when continuing from checkpoint; a value above 1 assumes the
            fields are already at the saddle point, as a checkpoint stores them.
        """

        print("---------- Run  ----------")

        # The number of components
        M = len(self.monomer_types)

        # The number of real fields
        R = len(self.mpt.aux_fields_real_idx)

        # Check the partition consistency on the first field of this run
        # (the LR/LRAM setup already called compute_concentrations on w = 0)
        self._partition_checked = False

        # Simulation data directory, with the drawings of the chain architectures (polymer_<id>.png)
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)
        draw_polymer_architectures(self.polymers, self.ds, self.segment_lengths, directory=self.recording["dir"])

        # Reshape initial fields
        w = np.array([np.reshape(initial_fields[m], self.prop_solver.n_grid)
                      for m in self.monomer_types])
            
        # Convert monomer potential fields into auxiliary potential fields
        w_aux = self.mpt.to_aux_fields(w)
        self.mpt.pin_dropped_fields(w_aux)
        saved_aux = None if _restart_data is None else _restart_data.get(auxiliary_field_key(_restart_data))
        if saved_aux is not None:
            w_aux = np.asarray(saved_aux, dtype=np.float64).reshape(M, self.prop_solver.n_grid).copy()
        if not np.all(np.isfinite(w_aux)):
            raise ValidationError("Initial fields must be finite")

        if start_langevin_step is None:
            start_langevin_step = 1

        # Find the saddle point of the initial fields. A continued run loads fields
        # that are already at the saddle point; solving again would change the
        # trajectory slightly, so the loop's own solve is the first one then.
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        if start_langevin_step == 1:
            phi = self.saddle.first_point(w_aux)
        elif saved_aux is not None:
            phi = {m: np.asarray(_restart_data["phi_" + m]).reshape(self.prop_solver.n_grid).copy()
                   for m in self.monomer_types}
        else:
            phi, _ = self.compute_concentrations(w_aux)

        # Measurements of the run, with the pending ones of a checkpoint
        series = {"H_history": 0, "drift_history": 0} | {"dH " + key: 0 for key in self.chi_n}
        record = RunRecord(series, self.structure_functions.empty_sums())
        record.restore(_restart_data, start_langevin_step)

        # Create an empty array for field update algorithm
        # (a checkpoint loaded with squeeze_me drops the leading axis when R == 1)
        if normal_noise_prev is None:
            normal_noise_prev = np.zeros([R, self.prop_solver.n_grid], dtype=np.float64)
        else:
            normal_noise_prev = np.reshape(np.asarray(normal_noise_prev, dtype=np.float64), [R, self.prop_solver.n_grid])


        # Init timers
        self.saddle.iterations = 0
        self._intermediate_saddle_iterations = 0
        time_start = time.time()

        # Langevin iteration begins here
        langevin_step = start_langevin_step - 1   # stays when max_step < start_langevin_step
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            step_start = time.perf_counter()
            if self.verbose_level >= 1:
                print("Langevin step: ", langevin_step)

            # Forces on the real fields, with the bias of WTMD (same as deep-langevin-fts)
            phi_for_force = self.smearing.apply_to_dict(phi)
            w_lambda = self.mpt.compute_func_deriv(w_aux, phi_for_force, self.mpt.aux_fields_real_idx)
            if self.wtmd is not None:
                psi = self.wtmd.compute_order_parameter(langevin_step, w_aux)
                self.wtmd.add_bias_to_langevin(psi, w_lambda)
            drift_max = float(np.max(np.abs(w_lambda * self.dt_scaling[self.mpt.aux_fields_real_idx, None]))) if R else 0.0

            # One Langevin step of the real fields and the saddle point of the imaginary fields
            w_aux_before = w_aux
            w_aux, phi, hamiltonian, error_level, normal_noise_prev = self.saddle.advance(
                w_aux_before, w_lambda, normal_noise_prev, langevin_step)
            self.run_status["last_completed_step"] = langevin_step

            # WTMD: order parameter with dH/dχN of the same fields (those before the update, at which
            # psi and the bias were computed), the update of the bias and its statistics files
            if self.wtmd is not None:
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux_before, phi=phi_for_force, wtmd=True)
                self.wtmd.store_order_parameter(psi, dH)
                if langevin_step % self.wtmd.update_freq == 0:
                    self.wtmd.update_statistics()
                if langevin_step % self.wtmd.recording_period == 0:
                    self.wtmd.write_data(os.path.join(self.recording["dir"], "wtmd_statistics_%06d.mat" % (langevin_step)))

            # Measurements of the accepted state: H, the largest drift, dH/dχN and S(k)
            if langevin_step % self.recording["sample_interval"] == 0:
                record.sample_count += 1
                record.add("H_history", hamiltonian)
                record.add("drift_history", drift_max)
                dH = self.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=self.smearing.apply_to_dict(phi))
                for key in self.chi_n:
                    record.add("dH " + key, dH[key])
                self.structure_functions.add(w_aux, phi, record.sums)

            # Statistics files: H and dH/dχN, then S(k)
            if langevin_step % self.recording["statistics_save_interval"] == 0:
                mdic = record.statistics()
                mdic["simulation_metadata_json"] = json.dumps(self.metadata)
                atomic_savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % (langevin_step)), mdic,
                               long_field_names=True, do_compression=True)
                self.structure_functions.save(langevin_step, record)
                record.clear(langevin_step + 1)

            # Observe an accepted state before saving its callback/integrator state.
            notify_callbacks(self, langevin_step, w_aux, phi, hamiltonian, drift_max,
                             error_level, time.perf_counter() - step_start)

            # Save simulation data
            if langevin_step % self.recording["checkpoint_interval"] == 0:
                self._recording_state = record
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev, w_aux=w_aux)

        print("Cumulative failed saddle-point solves: %d times" % self.run_status["saddle_failures"])

        # Estimate execution time
        time_duration = time.time() - time_start

        n_steps = max(langevin_step + 1 - start_langevin_step, 1)
        print("total time: %f, time per step: %f" % (time_duration, time_duration/n_steps))
        total_saddle_iter = self.saddle.iterations + self._intermediate_saddle_iterations
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/n_steps))

    def evaluate_force(self, fields, *, w_aux, step):
        """dH/dw at intermediate active fields, with a saddle solve and fixed bias.

        No noise is drawn and no observations or WTMD statistics are recorded.
        ``w_aux`` supplies the imaginary-field warm start of this proposal.
        """
        trial = w_aux.copy()
        indices = self.mpt.aux_fields_real_idx
        trial[indices] = checked_fields(fields, trial[indices])
        for attempt in range(self.saddle.settings.get("max_retries", 5) + 1):
            phi, hamiltonian, iterations, error = self.saddle.solve(trial)
            self._intermediate_saddle_iterations += iterations
            if not np.isfinite(hamiltonian) or not np.all(np.isfinite(trial)):
                raise ValidationError("Non-finite intermediate LFTS state")
            if np.isfinite(error) and error < self.saddle.settings["tolerance"]:
                break
            self.run_status["saddle_failures"] += 1
            if self.saddle.settings.get("failure_policy", "retry") == "abort":
                break
        else:
            raise SaddlePointError(f"Intermediate saddle failed at step {step} (error={error})")
        if not np.isfinite(error) or error >= self.saddle.settings["tolerance"]:
            raise SaddlePointError(f"Intermediate saddle failed at step {step} (error={error})")
        force = self.mpt.compute_func_deriv(trial, self.smearing.apply_to_dict(phi), indices)
        if self.wtmd is not None:
            psi = self.wtmd.compute_order_parameter(step, trial)
            self.wtmd.add_bias_to_langevin(psi, force)
        if not np.all(np.isfinite(force)):
            raise ValidationError("Non-finite intermediate LFTS force")
        return force

    def mean_concentrations(self):
        """Volume-averaged concentration of each monomer type from the chain compositions."""
        return mean_concentrations(self)
