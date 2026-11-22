"""The Jacobian of the forces of a CL-FTS simulation at the homogeneous state: its linear response.

An integrator that declares ``linear_response`` (the built-in "etd", integrators.py) integrates
the linear part of the forces exactly. Before the run CLFTS measures the Jacobian of its forces at
the homogeneous state (force_jacobian) and passes its symmetric part (symmetric_jacobian) to the
integrator's set_linear_response. CLFTS._prepare_integrator gives them what the integrator advances:
whether ψ moves with the auxiliary fields (joint_psi), and the mobilities and noise directions of the
integrated fields.
"""

import contextlib

import numpy as np

from ..utils.validation import ValidationError


def force_jacobian(sim, joint_psi):
    """Jacobian J(k) = dF/dw of the forces of the integrated fields at the homogeneous state.

    The fields are the auxiliary fields and, when the integrator advances it (joint_psi), ψ with the
    force -δH/δψ, so that every field steps as w - δt s F. A kick iε at one grid point of
    field j has the same Fourier amplitude iε at every wavevector; the concentrations
    respond linearly and in a homogeneous periodic box do not mix wavevectors, so the
    imaginary part of the change of the forces divided by ε gives column j of J(k) at all k
    at once (complex-step derivative, error O(ε²)). The forces of the unkicked state are
    subtracted: transforms of real data, for example FFTs of sizes with large prime factors,
    leave imaginary rounding errors of order 1e-15 that a division by ε alone would amplify.
    With ε = 1e-6 the error O(ε²) is about 1e-12 and the rounding error of J below 1e-7 of its
    size, even on large grids. The
    homogeneous state is w = 0 with ψ = 0: a uniform shift of the fields does not change
    the concentrations of canonical chains. Grand-canonical chains have the density
    e^(βμ_p) α_p there, not that of the homogeneous saddle point; J only sets the linear
    part that ETD integrates exactly. Dynamical stabilization is not part of J.
    """
    M, n_grid, nx = len(sim.monomer_types), sim.prop_solver.n_grid, sim.prop_solver.nx
    es = sim.electrostatics
    m = M + int(joint_psi)
    epsilon = 1e-6

    def forces(w_aux, psi):
        with es.trial(psi) if es is not None else contextlib.nullcontext():
            phi, _ = sim.compute_concentrations(w_aux)
            phi_smeared = sim.smearing.apply_to_dict(phi)
            force = sim.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
            if joint_psi:
                force = np.vstack([force, -es.charge_density(phi_smeared)])
        return force

    base = forces(np.zeros((M, n_grid), dtype=np.complex128), np.zeros(n_grid, dtype=np.complex128))
    jacobian = np.zeros((m, m) + tuple(nx), dtype=np.complex128)
    for j in range(m):
        if j < M and sim.dt_scaling[j] == 0:
            continue
        w_aux = np.zeros((M, n_grid), dtype=np.complex128)
        psi = np.zeros(n_grid, dtype=np.complex128)
        if j < M:
            w_aux[j, 0] = 1j * epsilon
        else:
            psi[0] = 1j * epsilon
        change = forces(w_aux, psi) - base
        for i in range(m):
            jacobian[i, j] = np.fft.fftn(np.reshape(np.imag(change[i]) / epsilon, nx))
    if joint_psi:
        jacobian[M, M] += es.screened_k_sq / es.coupling
    return jacobian


def symmetric_jacobian(sim, jacobian, noise_direction, mobility):
    """Symmetric part of the measured Jacobian in the variables w/n of the noise directions n.

    noise_direction and mobility are those of the integrated fields; ψ is the last of them when the
    Jacobian has a row more than the auxiliary fields (force_jacobian with joint_psi).

    K = N⁻¹ J N holds second derivatives of H and is complex symmetric. The forces of
    continuous chains are the gradient of the discretized H only up to the contour error, so
    the measured K is symmetric only to O(ds²) (rk2 and rqm4 alike), notably where the
    segment lengths differ; the linear response takes its symmetric part. A relative
    asymmetry above 1e-2 among the fields that move means a far too coarse contour step or
    a wrong measurement. The diagonal (k² + κ²)/E of a jointly advanced ψ can exceed the
    auxiliary entries by orders of magnitude, so the auxiliary block is measured on its own
    scale and the coupling of ψ to it on the geometric mean of the auxiliary scale and the
    measured charge response of ψ (its diagonal without the analytic (k² + κ²)/E).
    """
    m, M = jacobian.shape[0], len(sim.monomer_types)
    joint_psi = m > M
    ratio = (noise_direction[None, :] / noise_direction[:, None]).reshape((m, m) + (1,) * (jacobian.ndim - 2))
    K = jacobian * ratio
    aux = np.flatnonzero(mobility[:M] > 0)
    K_aux = K[np.ix_(aux, aux)]
    aux_scale = max(np.max(np.abs(K_aux)), 1e-300) if aux.size else 1e-300
    asymmetry = np.max(np.abs(K_aux - np.swapaxes(K_aux, 0, 1))) / aux_scale if aux.size else 0.0
    if joint_psi and aux.size:
        es = sim.electrostatics
        psi_scale = max(np.max(np.abs(K[M, M] - es.screened_k_sq / es.coupling)), 1e-300)
        asymmetry = max(asymmetry, np.max(np.abs(K[aux, M] - K[M, aux])) / np.sqrt(aux_scale * psi_scale))
    if asymmetry > 1e-2:
        raise ValidationError(f"The Jacobian of the forces is not symmetric (relative asymmetry {asymmetry:.1e}); "
                              "with continuous chains use a smaller ds")
    if asymmetry > 1e-5:
        print(f"Jacobian of the forces: relative asymmetry {asymmetry:.1e} from the contour step, symmetrized")
    return 0.5 * (K + np.swapaxes(K, 0, 1)) / ratio
