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
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(backend);
    }
    catch (std::exception& exc)
    {
        // Nothing to free here: the FFT and Pseudo objects created by
        // init_shared() are owning members of CpuSolverPseudoBase
        throw_without_line_number(exc.what());
    }
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
    try
    {
        const int M_full = this->cb->get_total_grid();
        const int M_reduced = this->n_basis_;
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // For periodic BC, coefficient array is actually complex (interleaved real/imag)
        // so allocate extra space
        int coeff_size = this->complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;

        // The CrysFFT path works on the physical grid and returns the reduced basis;
        // the other paths work on the full grid
        const bool crysfft_path = this->use_crysfft();

        // Work array, kept per thread (called from OpenMP parallel regions)
        typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
        std::vector<double>& k_q_in = work.k_1;
        if (!crysfft_path)
            k_q_in.resize(coeff_size);

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();

        // Determine input/output pointers based on space group
        T* fft_in = q_in;
        T* fft_out = q_out;

        // Work arrays of this thread for expand/reduce (called from OpenMP parallel regions);
        // the CrysFFT path reads and writes the reduced basis directly
        std::vector<T>& q_full_in_local = work.q_full_in;
        std::vector<T>& q_full_out_local = work.q_full_out;

        if (this->space_group_ != nullptr && !crysfft_path)
        {
            q_full_in_local.resize(M_full);
            q_full_out_local.resize(M_full);

            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);

            fft_in = q_full_in_local.data();
            fft_out = q_full_out_local.data();
        }

        // ===== RK2: Single full step =====
        if (crysfft_path)
        {
            // Steps 1-5 on the CrysFFT physical grid, from and into the reduced basis (real fields
            // only, init_crysfft)
            if constexpr (std::is_same_v<T, double>)
            {
                auto [phys_in, phys_out] = this->crysfft_grids(work);
                const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
                this->fill_crysfft_from_reduced(q_in, phys_in);
                this->crysfft_step(phys_in, phys_out, coeff_full, _exp_dw, _exp_dw);
                this->reduce_crysfft_to_reduced(phys_out, q_out);
            }
        }
        else
        {
            const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);

            // Step 1: exp(-w·ds/2) * q_in
            for (int i = 0; i < M_full; ++i)
                fft_out[i] = _exp_dw[i] * fft_in[i];

            // Step 2-5: Forward FFT -> multiply by Boltzmann factor -> Backward FFT -> exp(-w·ds/2) * result
            this->transform_forward(fft_out, k_q_in.data());
            this->transform_backward(k_q_in.data(), _boltz_bond, fft_out, _exp_dw);

            // Reduce full grid → reduced basis
            if (this->space_group_ != nullptr)
            {
                if constexpr (std::is_same_v<T, double>)
                    this->space_group_->to_reduced_basis(fft_out, q_out, 1);
            }
        }

        // Apply the mask on the output. The computation box stores the mask in the
        // basis of the propagators (reduced with a space group, full otherwise), so it
        // is applied after the reduction; the mask is symmetric, so this equals masking
        // on the full grid.
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M_reduced; ++i)
                q_out[i] *= q_mask[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRK2);
