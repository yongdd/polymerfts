/**
 * @file CpuSolverPseudoRQM4.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators using RQM4.
 *
 * Implements the operator splitting pseudo-spectral method with
 * RQM4 (Ranjan-Qin-Morse 4th-order using Richardson extrapolation)
 * for continuous Gaussian chains. Supports all boundary conditions
 * (periodic, reflecting, absorbing).
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation dq/ds = (b^2/6)nabla^2 q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT -> multiply by exp(-b^2 k^2 ds/6) -> IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * Combines one full step (ds) and two half-steps (ds/2 each):
 *     q_out = (4·q_half - q_full) / 3
 *
 * This yields 4th-order accuracy in ds.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoRQM4<double>: Real fields with r2c FFT
 * - CpuSolverPseudoRQM4<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see ComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuSolverPseudoRQM4.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRQM4<T>::CpuSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, SpaceGroup* space_group, FFTBackend backend)
    : CpuSolverPseudoBase<T>(cb, molecules, space_group, /*with_half=*/true)
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
void CpuSolverPseudoRQM4<T>::update_dw(std::map<std::string, const T*> w_input)
{
    // exp(-w·ds/2) and exp(-w·ds/4), in the reduced basis on the CrysFFT path
    this->compute_exp_dw(w_input, /*reduced_basis_tables=*/this->use_crysfft(), true, 0.5, 0.25);
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::advance_propagator(
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
        const int M_out = crysfft_path ? M_reduced : M_full;

        // Work arrays, kept per thread (called from OpenMP parallel regions)
        typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
        std::vector<T>& q_out1 = work.q_1;
        std::vector<T>& q_out2 = work.q_2;
        std::vector<double>& k_q_in1 = work.k_1;
        std::vector<double>& k_q_in2 = work.k_2;
        q_out1.resize(M_out);
        q_out2.resize(M_out);
        if (!crysfft_path)
        {
            k_q_in1.resize(coeff_size);
            k_q_in2.resize(coeff_size);
        }

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();
        const T* _exp_dw_half = this->exp_dw_half.at(ds_index).at(monomer_type).data();

        // Determine input pointer (expand reduced basis if space_group is set;
        // the CrysFFT path reads the reduced basis directly)
        T* fft_in = q_in;

        // Work array of this thread for expand (called from OpenMP parallel regions)
        std::vector<T>& q_full_in_local = work.q_full_in;

        if (this->space_group_ != nullptr && !crysfft_path)
        {
            q_full_in_local.resize(M_full);
            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);
            fft_in = q_full_in_local.data();
        }

        // ===== Step 1: one full step (q_out1); Step 2: two half steps (q_out2) =====
        if (crysfft_path)
        {
            // On the CrysFFT physical grid, from and into the reduced basis (real fields only, init_crysfft)
            if constexpr (std::is_same_v<T, double>)
            {
                auto [phys_in, phys_out] = this->crysfft_grids(work);

                // Full step: exp(-w·ds/2) exp(b²ds/6 ∇²) exp(-w·ds/2)
                const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
                this->fill_crysfft_from_reduced(q_in, phys_in);
                this->crysfft_step(phys_in, phys_out, coeff_full, _exp_dw, _exp_dw);
                this->reduce_crysfft_to_reduced(phys_out, q_out1.data());

                // Two half steps (exp(-w·ds/4) outside, exp(-w·ds/2) between them)
                const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, ds_index, true);
                this->fill_crysfft_from_reduced(q_in, phys_in);
                this->crysfft_step(phys_in, phys_out, coeff_half, _exp_dw_half, _exp_dw);   // first half step
                this->crysfft_step(phys_out, phys_in, coeff_half, nullptr, _exp_dw_half);   // second half step
                this->reduce_crysfft_to_reduced(phys_in, q_out2.data());
            }
        }
        else
        {
            const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
            const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, ds_index);

            // The full step and the first half step start from the same input
            for (int i = 0; i < M_full; ++i)
            {
                q_out1[i] = _exp_dw[i] * fft_in[i];
                q_out2[i] = _exp_dw_half[i] * fft_in[i];
            }

            // Full step (its last factor exp(-w·ds/2) comes with the extrapolation)
            this->transform_forward(q_out1.data(), k_q_in1.data());
            this->transform_backward(k_q_in1.data(), _boltz_bond, q_out1.data(), nullptr);

            // Two half steps, exp(-w·ds/2) between them (the last exp(-w·ds/4) comes with the extrapolation)
            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->transform_backward(k_q_in2.data(), _boltz_bond_half, q_out2.data(), _exp_dw);
            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->transform_backward(k_q_in2.data(), _boltz_bond_half, q_out2.data(), nullptr);
        }

        // ===== RQM4: Richardson extrapolation =====
        // Into q_out, unless a full-grid result is reduced to the basis of a space group afterwards
        const bool reduce_full_grid = this->space_group_ != nullptr && !crysfft_path;
        T* q_richardson = reduce_full_grid ? q_out2.data() : q_out;
        if (crysfft_path)
        {
            for (int i = 0; i < M_out; ++i)
                q_richardson[i] = (4.0 * q_out2[i] - q_out1[i]) / 3.0;
        }
        else
        {
            for (int i = 0; i < M_out; ++i)
            {
                const T q_full = q_out1[i] * _exp_dw[i];
                const T q_half = q_out2[i] * _exp_dw_half[i];
                q_richardson[i] = (4.0 * q_half - q_full) / 3.0;
            }
        }

        // Reduce full grid → reduced basis
        if (reduce_full_grid)
        {
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->to_reduced_basis(q_out2.data(), q_out, 1);
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
INSTANTIATE_CLASS(CpuSolverPseudoRQM4);
