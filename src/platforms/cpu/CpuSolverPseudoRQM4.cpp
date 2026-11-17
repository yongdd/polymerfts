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
CpuSolverPseudoRQM4<T>::CpuSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules, backend);

        const int M = cb->get_total_grid();

        // Ensure ContourLengthMapping is finalized before using it
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw vectors for each ds_index and monomer type
        // Also register local_ds values with Pseudo for boltz_bond computation
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            this->pseudo->add_ds_value(ds_idx, local_ds);
            this->register_ds_value(ds_idx, local_ds);

            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->exp_dw[ds_idx][monomer_type].resize(M);
                this->exp_dw_half[ds_idx][monomer_type].resize(M);
            }
        }

        // Finalize ds values to allocate boltz_bond arrays
        // (update_laplacian_operator will compute the actual values)
        this->pseudo->finalize_ds_values();
        this->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        // Nothing to free here: the FFT and Pseudo objects created by
        // init_shared() are owning members of CpuSolverPseudoBase
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRQM4<T>::~CpuSolverPseudoRQM4()
{
    // exp_dw and exp_dw_half vectors are automatically cleaned up;
    // the shared components (FFT, Pseudo) are owning members of the base
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::update_dw(std::map<std::string, const T*> w_input)
{
    // Fields on the grid the Boltzmann factors live on (full grid, or the
    // reduced basis on the CrysFFT path); M_use is the size of each exp_dw array
    std::map<std::string, std::vector<T>> w_full_cache;
    std::map<std::string, const T*> w_use;
    const int M_use = this->expand_w_for_update_dw(w_input, w_full_cache, w_use);

    // Get unique ds values from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    // Compute exp_dw for each ds_index and monomer type
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for (const auto& item : w_use)
        {
            const std::string& monomer_type = item.first;
            const T* w = item.second;

            if (!this->exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            std::vector<T>& exp_dw_vec = this->exp_dw.at(ds_idx).at(monomer_type);
            std::vector<T>& exp_dw_half_vec = this->exp_dw_half.at(ds_idx).at(monomer_type);

            exp_dw_vec.resize(M_use);
            exp_dw_half_vec.resize(M_use);

            for (int i = 0; i < M_use; ++i)
            {
                exp_dw_vec[i] = std::exp(-w[i] * local_ds * 0.5);
                exp_dw_half_vec[i] = std::exp(-w[i] * local_ds * 0.25);
            }
        }
    }
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
        const int M_reduced = (this->space_group_ != nullptr)
            ? this->space_group_->get_n_reduced_basis()
            : M_full;
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // For periodic BC, coefficient array is actually complex (interleaved real/imag)
        // so allocate extra space
        int coeff_size = this->complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;

        // The CrysFFT path works on the physical grid and returns the reduced basis;
        // the other paths work on the full grid
        const bool crysfft_path = this->use_crysfft() && this->space_group_ != nullptr;
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

        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // Determine input pointer (expand reduced basis if space_group is set;
        // the CrysFFT path reads the reduced basis directly)
        T* fft_in = q_in;

        // Thread-local buffer for expand (called from OpenMP parallel regions)
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
            // Full step
            for (int i = 0; i < M_full; ++i)
                q_out1[i] = _exp_dw[i] * fft_in[i];

            this->transform_forward(q_out1.data(), k_q_in1.data());
            this->multiply_fourier_coeffs(k_q_in1.data(), _boltz_bond, M_COMPLEX);
            this->transform_backward(k_q_in1.data(), q_out1.data());

            for (int i = 0; i < M_full; ++i)
                q_out1[i] *= _exp_dw[i];

            // First half step
            for (int i = 0; i < M_full; ++i)
                q_out2[i] = _exp_dw_half[i] * fft_in[i];

            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
            this->transform_backward(k_q_in2.data(), q_out2.data());

            for (int i = 0; i < M_full; ++i)
                q_out2[i] *= _exp_dw[i];

            // Second half step
            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
            this->transform_backward(k_q_in2.data(), q_out2.data());

            for (int i = 0; i < M_full; ++i)
                q_out2[i] *= _exp_dw_half[i];
        }

        // ===== RQM4: Richardson extrapolation =====
        // Into q_out, unless a full-grid result is reduced to the basis of a space group afterwards
        const bool reduce_full_grid = this->space_group_ != nullptr && !crysfft_path;
        T* q_richardson = reduce_full_grid ? q_out2.data() : q_out;
        for (int i = 0; i < M_out; ++i)
            q_richardson[i] = (4.0 * q_out2[i] - q_out1[i]) / 3.0;

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
