"""The saddle point of the imaginary fields in L-FTS.

LR, LRAM and Anderson mixing are the compressors. SaddlePointSearch uses them: it finds the saddle
point of given fields, and takes a Langevin step with its retries.
"""

import copy
import warnings
import weakref

import numpy as np

from ..utils.checkpoint import extension_object_state
from .integrators import checked_fields
from ..utils.common import check_partition_functions
from ..utils.validation import ValidationError, SaddlePointError, NumericalError


class LR:
    """Linear Response (LR) field compressor for L-FTS.

    The response of -δH/δw to the imaginary field near w = 0 is diagonal in the Fourier
    basis of the box, -δH/δw ≈ J(k) (w* - w)(k), with J(k) measured once from a small
    kick (SaddlePointSearch). One LR step divides the residual by J(k):
    w_new = w + F⁻¹[F(-δH/δw) / J(k)], which lands on the saddle point w* where the
    response is linear. No matrix is inverted: 1/J(k) is a stored Fourier multiplier.

    Parameters
    ----------
    jk : ndarray
        J(k) in the coefficient shape of fourier, measured by finite difference of δH/δw.
    fourier : FourierTransform
        PropagatorSolver.fourier of real fields (FFT along periodic axes, cosine series
        along reflecting walls), which applies 1/J(k) (on the GPU with CUDA).

    Attributes
    ----------
    jk : ndarray
        J(k), a copy of the given values.
    """
    def __init__(self, jk, fourier):
        self.jk = jk.copy()
        self._fourier = fourier
        self._inverse = fourier.set_multiplier(1.0 / self.jk)

    def reset_count(self,):
        """Reset compressor state (no-op for LR).

        LR is stateless, so this method does nothing. Included for
        compatibility with Anderson Mixing interface.
        """
        pass

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using linear response method.

        Performs one linear response update to move toward the saddle point.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (not used by LR, kept for interface compatibility).
        error_level : float
            Current error level (not used by LR, kept for interface compatibility).

        Returns
        -------
        w_new : ndarray
            Updated field values, same shape as w_current.
        """
        w_diff = self._fourier.apply([(self._inverse, np.reshape(negative_h_deriv, -1))])
        return w_current + np.reshape(w_diff, np.shape(w_current))


class LRAM:
    """Linear Response + Anderson Mixing (LRAM) field compressor.

    Combines Linear Response for coarse optimization with Anderson Mixing for
    final refinement. This hybrid approach is more robust than LR alone while
    being faster than pure Anderson Mixing.

    The algorithm:
    1. Use LR to get approximate update
    2. Use AM to refine the LR result

    This combines the speed of LR with the robustness of AM.

    Parameters
    ----------
    lr : LR
        Linear Response compressor instance.
    am : AndersonMixing
        Anderson Mixing instance (from C++ backend).

    Attributes
    ----------
    lr : LR
        Linear Response compressor.
    am : AndersonMixing
        Anderson Mixing optimizer.
    """
    def __init__(self, lr, am):
        self.lr = lr
        self.am = am

    def reset_count(self,):
        """Reset Anderson Mixing history.

        Clears the AM history buffer. Should be called when starting a new
        saddle point search (e.g., at beginning of each Langevin step).
        """
        self.am.reset_count()

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using LRAM hybrid method.

        Performs Linear Response followed by Anderson Mixing refinement.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (used by AM for mixing parameter adjustment).
        error_level : float
            Current error level (used by AM for mixing parameter adjustment).

        Returns
        -------
        w_new : ndarray
            Updated field values after LR + AM, same shape as w_current.
        """
        w_lr_old = np.reshape(w_current, (1, -1))
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), (1, -1))
        w_diff = w_lr_new - w_lr_old
        w_new = self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level)

        return np.reshape(w_new, np.shape(w_current))


class SaddlePointSearch:
    """The saddle point of the imaginary fields of an LFTS simulation.

    solve() finds the saddle point of given fields, and solve_with_retries() goes on with a failed search
    as the input allows. first_point() finds that of the initial fields, advance() takes one Langevin
    step of the real fields and finds the saddle point of the result, and evaluate_force() gives the
    forces at the intermediate fields of a step. `compressor` is Anderson mixing, LR or LRAM, `settings`
    the input "saddle" and `iterations` the number of compressor iterations of the run.
    """

    def __init__(self, sim, params):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.settings = params["saddle"].copy()
        self.mpt, self.smearing, self.wtmd = sim.mpt, sim.smearing, sim.wtmd
        self.prop_solver, self.cb = sim.prop_solver, sim.cb
        self.monomer_types, self.zeta_n = sim.monomer_types, sim.zeta_n
        self.iterations = 0
        self.compressor = self._create_compressor(params)

    def _create_compressor(self, params):
        """Anderson mixing, linear response or both, with the response measured at w = 0."""
        if not self.mpt.aux_fields_imag_idx:
            return None   # The ideal reference limit has no fields to compress.
        M = len(self.monomer_types)

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["compressor"]["name"] == "am" or params["compressor"]["name"] == "lram":
            n_am = int(len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]))
            am = self.prop_solver.create_anderson_mixing(
                n_am,                                                      # the number of variables
                params["compressor"]["max_hist"],                          # maximum number of history
                params["compressor"]["start_error"],                       # when switch to AM from simple mixing
                params["compressor"]["mix_min"],                           # minimum mixing rate of simple mixing
                params["compressor"]["mix_init"])                          # initial mixing rate of simple mixing

        # Fields Relaxation using Linear Response Method
        if params["compressor"]["name"] == "lr" or params["compressor"]["name"] == "lram":
            if len(self.mpt.aux_fields_imag_idx) != 1:
                raise ValidationError("Currently, LR methods are not working for imaginary-valued auxiliary fields.")
            # The response J to a kick at w = 0 is diagonal in the Fourier basis of the box
            # (PropagatorSolver.fourier): the FFT along periodic axes, the cosine series along
            # reflecting walls (the box is half of a periodic box of twice its size, with the
            # mirror image of the kick). Next to an absorbing wall the concentrations at w = 0
            # are depleted, and no Fourier basis makes J diagonal
            if "absorbing" in self.prop_solver.bc_lower:
                raise ValidationError("compressor 'lr' and 'lram' need periodic axes or reflecting walls; "
                                      "use 'am' in a box with absorbing walls")

            # Baseline: h_deriv at w=0
            w_aux_zero = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            phi_zero, _ = self.sim.compute_concentrations(w_aux_zero)
            h_deriv_zero = self.mpt.compute_func_deriv(w_aux_zero, self.smearing.apply_to_dict(phi_zero), self.mpt.aux_fields_imag_idx)

            # Perturbed: h_deriv at w with small perturbation
            w_aux_perturbed = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            w_aux_perturbed[M-1,0] = 1e-3 # add a small perturbation at the pressure field

            phi_perturbed, _ = self.sim.compute_concentrations(w_aux_perturbed)
            h_deriv_perturbed = self.mpt.compute_func_deriv(w_aux_perturbed, self.smearing.apply_to_dict(phi_perturbed), self.mpt.aux_fields_imag_idx)

            # jk = (h(w_pert) - h(0)) / w_pert  (correct finite difference)
            h_deriv_diff = h_deriv_perturbed - h_deriv_zero
            fourier = self.prop_solver.fourier
            jk_numeric = np.real(fourier.forward(np.reshape(h_deriv_diff, -1))/fourier.forward(w_aux_perturbed[M-1]))
            # Incompressible: a uniform shift of the pressure field changes no concentration, so
            # J(0) = 0 and the value above is rounding, exactly zero on some small grids. J(0) = 1
            # keeps 1/J finite, and the gauge fix of the saddle search removes the uniform mode
            if self.zeta_n is None:
                jk_numeric.flat[0] = 1.0
            lr = LR(jk_numeric, fourier=fourier)

        if params["compressor"]["name"] == "am":
            return am
        elif params["compressor"]["name"] == "lr":
            return lr
        return LRAM(lr, am)   # "lram" (validation.py accepts these three)

    def first_point(self, w_aux):
        """Saddle point of the initial fields, found in place; returns their concentrations."""
        phi, _, _ = self.solve_with_retries(w_aux, "the initial saddle point")
        return phi

    def advance(self, w_aux_start, forces, noise_prev, langevin_step):
        """One Langevin step of the real fields, then the saddle point of the imaginary fields.

        A failed saddle search goes on from where it stopped ("retry", solve_with_retries), or the step
        is drawn again ("resample", which can change the sampled distribution). Returns the new fields
        w_aux, their concentrations and Hamiltonian, the error level and the noise of the step.
        """
        R = len(self.mpt.aux_fields_real_idx)
        real = self.mpt.aux_fields_real_idx

        # Retrying the solver must not select the Langevin noise by convergence: only "resample" draws
        # the step again, each time from the state of the integrator before the step
        resample = self.settings.get("failure_policy", "retry") == "resample"
        draws = self.settings.get("max_retries", 5) + 1 if resample else 1
        integrator_state = extension_object_state(self.sim.integrator)
        for draw in range(draws):
            noise = self.sim.random.normal(0.0, self.sim.langevin["sigma"], [R, self.prop_solver.n_grid])
            if draw:
                self.sim.run_status["noise_resamples"] += 1
                warnings.warn("Saddle failure: resampling noise can change the sampled distribution", RuntimeWarning)
            if hasattr(self.sim.integrator, "load_state_dict"):
                self.sim.integrator.load_state_dict(copy.deepcopy(integrator_state))
            try:
                w_aux = w_aux_start.copy()
                active = w_aux[real]
                proposed = self.sim.integrator.step(
                    active.copy(), forces.copy(), dt=self.sim.langevin["dt"],
                    mobility=self.sim.dt_scaling[real].copy(),
                    noise=noise.copy(), previous_noise=noise_prev.copy(),
                    noise_direction=np.ones(R),
                    evaluate_force=lambda fields: self.evaluate_force(fields, w_aux_start, langevin_step))
                w_aux[real] = checked_fields(proposed, active)

                # Find saddle point of the pressure field (one search per draw with "resample")
                phi, hamiltonian, error_level = self.solve_with_retries(
                    w_aux, f"the saddle point of step {langevin_step}", retry=not resample)
                return w_aux, phi, hamiltonian, error_level, noise
            except SaddlePointError:
                if hasattr(self.sim.integrator, "load_state_dict"):
                    self.sim.integrator.load_state_dict(copy.deepcopy(integrator_state))
                if draw == draws - 1:
                    raise

    def evaluate_force(self, fields, w_aux, langevin_step):
        """dH/dw of the real fields at the intermediate fields of a step (the evaluate_force of the
        integrator), with the saddle point of the imaginary fields found from those of w_aux and the
        bias of WTMD as it stands. No noise is drawn and nothing is recorded."""
        trial = w_aux.copy()
        real = self.mpt.aux_fields_real_idx
        trial[real] = checked_fields(fields, trial[real])
        phi, _, _ = self.solve_with_retries(trial, f"an intermediate saddle point of step {langevin_step}")
        force = self.mpt.compute_func_deriv(trial, self.smearing.apply_to_dict(phi), real)
        if self.wtmd is not None:
            self.wtmd.add_bias(trial, force)
        if not np.all(np.isfinite(force)):
            raise NumericalError("Non-finite intermediate LFTS force", self.mpt.to_monomer_fields_dict(trial))
        return force

    def solve_with_retries(self, w_aux, where, retry=True):
        """The saddle point of w_aux, found in place; returns its concentrations, Hamiltonian and error level.

        A search that does not converge goes on from where it stopped, saddle.max_retries times at most:
        not with failure_policy "abort" or retry=False, nor from fields that are not finite. `where` names
        the saddle point in the errors: a Hamiltonian that is not finite raises NumericalError, which names
        a partition function that is not positive as its cause, and a search that fails SaddlePointError.
        """
        retries = self.settings.get("max_retries", 5)
        if not retry or self.settings.get("failure_policy", "retry") == "abort":
            retries = 0
        for attempt in range(retries + 1):
            phi, hamiltonian, iterations, error_level = self.solve(w_aux)
            self.iterations += iterations
            if not np.isfinite(hamiltonian):
                # H contains log Q: name a partition function that is not positive as the cause
                fields = self.mpt.to_monomer_fields_dict(w_aux)
                check_partition_functions(self.prop_solver, where, fields)
                raise NumericalError(f"Non-finite Hamiltonian at {where}", fields)
            if np.isfinite(error_level) and error_level < self.settings["tolerance"]:
                return phi, hamiltonian, error_level
            self.sim.run_status["saddle_failures"] += 1
            if not np.all(np.isfinite(w_aux)):
                break
        raise SaddlePointError(f"The search for {where} failed after {attempt + 1} attempts (error={error_level})",
                               self.mpt.to_monomer_fields_dict(w_aux))

    def solve(self, w_aux):
        """Find saddle point of imaginary auxiliary fields.

        Iteratively compresses imaginary-valued auxiliary fields to their saddle
        point values while keeping real-valued fields fixed. This is performed at
        each Langevin step to ensure imaginary fields remain at their saddle point.

        The saddle point satisfies δH/δw_imag = 0, enforcing incompressibility and
        other constraints.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).
            Real auxiliary fields (indices in aux_fields_real_idx) remain fixed.
            Imaginary fields (indices in aux_fields_imag_idx) are optimized.

        Returns
        -------
        phi : dict
            Final monomer concentration fields {monomer_type: array}.
        hamiltonian : float
            Hamiltonian value at converged saddle point.
        saddle_iter : int
            Number of iterations performed to find saddle point.
        error_level : float
            Final convergence error (max standard deviation of δH/δw_imag).
        """

        # The number of components
        M = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        if I == 0:
            # There is no saddle search, but the run still needs the concentrations and H of this state.
            phi, _ = self.sim.compute_concentrations(w_aux)
            total_partitions = [self.prop_solver.get_partition_function(p)
                                for p in range(self.prop_solver.get_n_polymer_types())]
            hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb,
                                                      include_const_term=True, phi=self.smearing.apply_to_dict(phi))
            return phi, hamiltonian, 0, 0.0

        # Assign large initial value for error
        error_level = 1e20

        # Reset compressor
        self.compressor.reset_count()

        # The uniform mode of each imaginary field (except the incompressible pressure
        # field, a gauge freedom) has the analytic saddle value
        # mean(Ω_k) = -(c1_k + Σ_j A_jk mean(φ_j))/(2 c2_k), mean(φ) being fixed by the
        # chain compositions. It is pinned in every iteration, so the concentrations and
        # partition functions of the last iteration belong to the returned fields; the
        # convergence test (standard deviation) does not see this mode.
        phi_mean = self.sim.mean_concentrations()
        mean_targets = {}
        for k in self.mpt.aux_fields_imag_idx:
            # A finite bulk modulus still fixes this mode, however small its reciprocal coefficient.
            if self.mpt.h_coef_mu2[k] != 0.0:
                mean_targets[k] = -(self.mpt.h_coef_mu1[k] + np.dot(self.mpt.matrix_a[:, k], phi_mean)) / (2*self.mpt.h_coef_mu2[k])

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.settings["max_iter"]+1):

            for k, target in mean_targets.items():
                w_aux[k] += target - self.cb.mean(w_aux[k])

            # Compute total concentrations
            phi, _ = self.sim.compute_concentrations(w_aux)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields
            phi_for_saddle = self.smearing.apply_to_dict(phi)
            h_deriv = self.mpt.compute_func_deriv(w_aux, phi_for_saddle, self.mpt.aux_fields_imag_idx)

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Hamiltonian on the last iteration (every iteration when verbose_level == 2)
            is_last = error_level < self.settings["tolerance"] or saddle_iter == self.settings["max_iter"]
            if self.sim.verbose_level == 2 or is_last:
                total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
                hamiltonian = self.mpt.compute_hamiltonian(self.prop_solver.molecules, w_aux, total_partitions, self.cb,
                                                          include_const_term=True, phi=phi_for_saddle)

            # Print iteration # and error levels
            if self.sim.verbose_level == 2 or (self.sim.verbose_level == 1 and is_last):
                mass_error = self.cb.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for Q in total_partitions:
                    print("%13.7E " % (Q), end=" ")
                print("] %15.9f   [" % (hamiltonian), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("]")

            # Conditions to end the iteration
            if error_level < self.settings["tolerance"]:
                break

            # Scaling h_deriv
            for count, i in enumerate(self.mpt.aux_fields_imag_idx):
                h_deriv[count] *= self.sim.dt_scaling[i]

            # Calculate new fields using compressor (AM, LR, LRAM)
            w_aux[self.mpt.aux_fields_imag_idx] = np.reshape(self.compressor.calculate_new_fields(w_aux[self.mpt.aux_fields_imag_idx], -h_deriv, old_error_level, error_level), [I, self.prop_solver.n_grid])

        # Incompressible: gauge fix of the pressure field to mean 0 (H and φ do not change)
        if self.zeta_n is None:
            w_aux[M-1] -= self.cb.mean(w_aux[M-1])

        return phi, hamiltonian, saddle_iter, error_level
