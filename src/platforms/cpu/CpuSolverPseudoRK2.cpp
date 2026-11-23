/**
 * @file CpuSolverPseudoRK2.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators using RK2.
 *
 * Implements the operator splitting pseudo-spectral method with
 * RK2 (Rasmussen-Kalosakas 2nd-order) for continuous Gaussian chains.
 * Supports all boundary conditions (periodic, reflecting, absorbing).
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation dq/ds = (b^2/6)nabla^2 q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT -> multiply by exp(-b^2 k^2 ds/6) -> IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step, while RQM4 combines full and half steps
 * via Richardson extrapolation. RK2 is faster (2 FFTs vs 6 FFTs) but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoRK2<double>: Real fields with r2c FFT
 * - CpuSolverPseudoRK2<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see ComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuSolverPseudoRK2.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRK2<T>::CpuSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, SpaceGroup* space_group, FFTBackend backend)
    : CpuSolverPseudoBase<T>(cb, molecules, space_group, /*with_half=*/false)   // no Richardson extrapolation
{
    // Initialize shared components (FFT, Pseudo, etc.)
    this->init_shared(backend);
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRK2<T>::update_dw(std::map<std::string, const T*> w_input)
{
    // exp(-w·ds/2), in the reduced basis on the CrysFFT path
    this->compute_exp_dw(w_input, /*reduced_basis_tables=*/this->use_crysfft(), false, 0.5, 0.0);
}

//------------------------------------------------------------------------------
// Advance propagator (RK2 - single full step)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRK2<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    // Boltzmann factor exp(-w·ds/2) of this contour step
    const T* _exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();

    if (this->use_crysfft())
    {
        // [real-only feature] The CrysFFT steps fold real fields on the physical sub-grid (init_crysfft())
        if constexpr (!std::is_same_v<T, double>)
        {
            throw_with_line_number("CrysFFT path is only supported for real fields.");
        }
        else
        {
            // exp(-w·ds/2) exp(b²ds/6 ∇²) exp(-w·ds/2) in the reduced basis
            const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
            this->crysfft_diffuse(q_in, q_out, coeff_full, _exp_dw, nullptr, _exp_dw);
            this->multiply_mask(q_out, q_mask);
            return;
        }
    }

    const int M = this->cb->get_total_grid();
    const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);

    // Full-grid input/output (expanded from / reduced to the basis with a space group) and
    // the spectrum, work arrays of this thread (called from OpenMP parallel regions). The
    // step runs in fft_out: the transform reads it before the inverse transform writes the
    // result there
    typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
    T* fft_in = this->expand_to_full_grid(q_in, work.q_full);
    T* fft_out = this->full_grid_output(q_out, work.q_full);
    std::vector<double>& k_q = work.k_1;
    k_q.resize(this->spectrum_size());

    // Step 1: exp(-w·ds/2) * q_in
    for (int i = 0; i < M; ++i)
        fft_out[i] = _exp_dw[i] * fft_in[i];

    // Step 2-5: Forward FFT -> multiply by Boltzmann factor -> Backward FFT -> exp(-w·ds/2) * result
    this->transform_forward(fft_out, k_q.data());
    this->transform_backward(k_q.data(), _boltz_bond, fft_out, _exp_dw);

    // Reduce to the basis (with a space group) and multiply the mask
    this->store_full_grid(fft_out, q_out, q_mask);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRK2);
