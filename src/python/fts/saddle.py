"""The saddle point of the imaginary fields in L-FTS.

LR, LRAM and Anderson mixing are the compressors. SaddlePointSearch uses them: it finds the saddle
point of given fields, and takes a Langevin step with its retries.
"""

import copy
import warnings
import weakref

import numpy as np
import scipy.fft

from ..utils.checkpoint import extension_object_state
from .integrators import checked_fields
from ..utils.common import check_partition_functions
from ..utils.validation import ValidationError, SaddlePointError


def _axis_transform(field, axis_bc, inverse=False):
    """Orthonormal transform of each axis: FFT ("periodic") or cosine series (DCT-II, "wall"), or its inverse."""
    for axis, kind in enumerate(axis_bc):
        if kind == "periodic":
            field = (scipy.fft.ifft if inverse else scipy.fft.fft)(field, axis=axis, norm="ortho")
        else:
            field = (scipy.fft.idct if inverse else scipy.fft.dct)(field, type=2, axis=axis, norm="ortho")
    return field


class LR:
    """Linear Response (LR) field compressor for L-FTS.

    Uses the local Hessian (second derivative matrix) of the Hamiltonian
    to find the saddle point of imaginary auxiliary fields via linear response
    theory. This is faster than iterative methods but requires computing and
    inverting the Hessian matrix.

    The update is: w_new = w_old - J^(-1) · δH/δw, where J is the Jacobian
    (Hessian) in Fourier space.

    Parameters
    ----------
    nx : list of int
        Grid dimensions [nx, ny, nz].
    lx : list of float
        Box dimensions in units of a_Ref * N_Ref^(1/2).
    jk : ndarray
        Jacobian (Hessian) matrix in Fourier space. Precomputed from
        finite difference of δH/δw. Shape: the coefficients of fourier, or nx with axis_bc.
    fourier : FourierTransform, optional
        PropagatorSolver.fourier of real fields (FFT in a periodic box, cosine series in
        a box with reflecting walls), which applies 1/J(k) (on the GPU with CUDA).
    axis_bc : list of str, optional
        "periodic" or "wall" for each axis of a box that mixes periodic axes and
        reflecting walls: FFT and cosine series (orthonormal) with scipy.fft.
        Give either fourier or axis_bc.

    Attributes
    ----------
    nx : list of int
        Grid dimensions.
    lx : list of float
        Box dimensions.
    jk : ndarray
        Jacobian in Fourier space.
    """
    def __init__(self, nx, lx, jk, fourier=None, axis_bc=None):
        if (fourier is None) == (axis_bc is None):
            raise ValidationError("LR needs either fourier or axis_bc.")
        self.nx = nx
        self.lx = lx
        self.jk = jk.copy()
        self._fourier = fourier
        self._inverse = None if fourier is None else fourier.set_multiplier(1.0 / self.jk)
        self.axis_bc = axis_bc

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
        if self._fourier is not None:
            w_diff = self._fourier.apply([(self._inverse, np.reshape(negative_h_deriv, -1))])
        else:
            w_diff_k = _axis_transform(np.reshape(negative_h_deriv, self.nx), self.axis_bc)/self.jk
            w_diff = np.real(_axis_transform(w_diff_k, self.axis_bc, inverse=True))
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
        nx = self.lr.nx

        w_lr_old = np.reshape(w_current, [1, np.prod(nx)])
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), [1, np.prod(nx)])
        w_diff = w_lr_new - w_lr_old
        w_new = self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level)

        return np.reshape(w_new, np.shape(w_current))


class SaddlePointSearch:
    """The saddle point of the imaginary fields of an LFTS simulation.

    solve() finds the saddle point of given fields, first_point() that of the initial fields with
    the retries of the input, and advance() takes one Langevin step of the real fields and finds
    the saddle point of the result. `compressor` is Anderson mixing, LR or LRAM, and `settings`
    the input "saddle".
    """

    def __init__(self, sim, params):
        self.sim = weakref.proxy(sim)   # no cycle: a dropped simulation frees its solver at once
        self.settings = params["saddle"].copy()
        self.mpt, self.smearing = sim.mpt, sim.smearing
        self.prop_solver, self.cb = sim.prop_solver, sim.cb
        self.monomer_types, self.zeta_n = sim.monomer_types, sim.zeta_n
        self.iterations = 0
        self.compressor = self._create_compressor(params)

    def _create_compressor(self, params):
        """Anderson mixing, linear response or both, with the response measured at w = 0."""
        M = len(self.monomer_types)
        periodic_sides = [b.lower() == "periodic" for b in self.prop_solver.bc]
        mixed_box = any(periodic_sides) and not all(periodic_sides)

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
            # The response J to a kick at w = 0 is diagonal in the Fourier basis of the box:
            # the FFT in a periodic box, the cosine series along reflecting walls (the box is
            # half of a periodic box of twice its size, with the mirror image of the kick).
            # Next to an absorbing wall the concentrations at w = 0 are depleted, and no
            # Fourier basis makes J diagonal
            if any(b.lower() == "absorbing" for b in self.prop_solver.bc):
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
            nx = self.prop_solver.nx
            fourier, axis_bc = None, None
            if mixed_box:
                # Periodic axes and reflecting walls: FFT and cosine series with scipy.fft
                axis_bc = ["periodic" if periodic else "wall" for periodic in periodic_sides[::2]]
                jk_numeric = np.real(_axis_transform(np.reshape(h_deriv_diff, nx), axis_bc)
                                     /_axis_transform(np.reshape(w_aux_perturbed[M-1], nx), axis_bc))
            else:
                # The FFT or the cosine series of PropagatorSolver.fourier
                fourier = self.prop_solver.fourier
                jk_numeric = np.real(fourier.forward(np.reshape(h_deriv_diff, -1))/fourier.forward(w_aux_perturbed[M-1]))
            # Incompressible: a uniform shift of the pressure field changes no concentration, so
            # J(0) = 0 and the value above is rounding, exactly zero on some small grids. J(0) = 1
            # keeps 1/J finite, and the gauge fix of the saddle search removes the uniform mode
            if self.zeta_n is None:
                jk_numeric.flat[0] = 1.0
            lr = LR(nx, self.prop_solver.get_lx(), jk_numeric, fourier=fourier, axis_bc=axis_bc)

        if params["compressor"]["name"] == "am":
            return am
        elif params["compressor"]["name"] == "lr":
            return lr
        elif params["compressor"]["name"] == "lram":
            return LRAM(lr, am)
        else:
            raise ValidationError(f"compressor['name'] must be 'am', 'lr' or 'lram', got '{params['compressor']['name']}'")

    def first_point(self, w_aux):
        """Saddle point of the initial fields, found in place; returns their concentrations."""
        for attempt in range(self.settings.get("max_retries", 5) + 1):
            phi, hamiltonian, _, error_level = self.solve(w_aux=w_aux)
            if not np.isfinite(hamiltonian):
                # H contains log Q: name a partition function that is not positive as the cause
                check_partition_functions(self.prop_solver, "the initial saddle point")
                raise ValidationError("Non-finite initial Hamiltonian")
            if np.isfinite(error_level) and error_level < self.settings["tolerance"]:
                break
            self.sim.run_status["saddle_failures"] += 1
            if self.settings.get("failure_policy", "retry") == "abort":
                break
        if not np.isfinite(error_level) or error_level >= self.settings["tolerance"]:
            raise SaddlePointError(f"Initial saddle point failed after {attempt + 1} attempts (error={error_level})")
        return phi

    def advance(self, w_aux_start, forces, noise_prev, langevin_step):
        """One Langevin step of the real fields, then the saddle point of the imaginary fields.

        A failed saddle search goes on from where it stopped ("retry"), or the step is drawn again
        ("resample", which can change the sampled distribution). Returns the new fields w_aux, their
        concentrations and Hamiltonian, the error level and the noise of the step.
        """
        R = len(self.mpt.aux_fields_real_idx)

        # Retrying the solver must not select the Langevin noise by convergence.
        policy = self.settings.get("failure_policy", "retry")
        integrator_state = extension_object_state(self.sim.integrator)
        for attempt in range(self.settings.get("max_retries", 5) + 1):
            if attempt == 0 or policy == "resample":
                w_aux = w_aux_start.copy()
                noise = self.sim.random.normal(0.0, self.sim.langevin["sigma"], [R, self.prop_solver.n_grid])
                if attempt:
                    self.sim.run_status["noise_resamples"] += 1
                    warnings.warn("Saddle failure: resampling noise can change the sampled distribution", RuntimeWarning)
                if hasattr(self.sim.integrator, "load_state_dict"):
                    self.sim.integrator.load_state_dict(copy.deepcopy(integrator_state))
                active = w_aux[self.mpt.aux_fields_real_idx]
                try:
                    proposed = self.sim.integrator.step(
                        active.copy(), forces.copy(), dt=self.sim.langevin["dt"],
                        mobility=self.sim.dt_scaling[self.mpt.aux_fields_real_idx].copy(),
                        noise=noise.copy(), previous_noise=noise_prev.copy(),
                        noise_direction=np.ones(R),
                        evaluate_force=lambda fields: self.sim.evaluate_force(fields, w_aux=w_aux_start, step=langevin_step))
                except SaddlePointError:
                    if hasattr(self.sim.integrator, "load_state_dict"):
                        self.sim.integrator.load_state_dict(copy.deepcopy(integrator_state))
                    if policy == "resample" and attempt < self.settings.get("max_retries", 5):
                        continue
                    raise
                w_aux[self.mpt.aux_fields_real_idx] = checked_fields(proposed, active)

            # Find saddle point of the pressure field
            phi, hamiltonian, saddle_iter, error_level = self.solve(w_aux=w_aux)
            if not np.isfinite(hamiltonian):
                # H contains log Q: name a partition function that is not positive as the cause
                check_partition_functions(self.prop_solver, f"the saddle point of step {langevin_step}")
                raise ValidationError(f"Non-finite Hamiltonian at step {langevin_step}")
            self.iterations += saddle_iter
            saddle_failed = not np.isfinite(error_level) or error_level >= self.settings["tolerance"]
            if not saddle_failed:
                break
            self.sim.run_status["saddle_failures"] += 1
            if policy == "abort" or not np.all(np.isfinite(w_aux)):
                break
        if saddle_failed:
            if hasattr(self.sim.integrator, "load_state_dict"):
                self.sim.integrator.load_state_dict(copy.deepcopy(integrator_state))
            raise SaddlePointError(f"Saddle point failed at step {langevin_step} after {attempt + 1} attempts (error={error_level})")
        return w_aux, phi, hamiltonian, error_level, noise

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
            if not np.isclose(self.mpt.h_coef_mu2[k], 0.0):
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
                                                          include_const_term=True, phi=self.smearing.apply_to_dict(phi))

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
