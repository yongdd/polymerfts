/**
 * @file CpuSolverPseudoDiscrete.cpp
 * @brief CPU pseudo-spectral solver for discrete chain propagators.
 *
 * Implements the pseudo-spectral method for discrete chain models using the
 * Chapman-Kolmogorov integral equation. Supports all boundary conditions
 * (periodic, reflecting, absorbing).
 *
 * **Chapman-Kolmogorov Equation (N-1 Bond Model):**
 *
 * For discrete chains, the propagator satisfies:
 *     q(r, i+1) = exp(-w(r)*ds) * integral g(r-r') q(r', i) dr'
 *
 * In Fourier space:
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
 * See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * Half-bond steps (ĝ^(1/2)(k) = exp(-b²|k|²ds/12)) are used only at
 * chain ends and junction points.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoDiscrete<double>: Real fields
 * - CpuSolverPseudoDiscrete<std::complex<double>>: Complex fields
 *
 * @see ComputationDiscrete for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuSolverPseudoDiscrete.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoDiscrete<T>::CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules, backend);

        const int M = cb->get_total_grid();

        // Create exp_dw vectors for each ds_index and monomer type (init_shared()
        // finalized the ContourLengthMapping)
        const int n_unique_ds = molecules->get_contour_length_mapping().get_n_unique_ds();
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->exp_dw[ds_idx][monomer_type].resize(M);
            }
        }
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
CpuSolverPseudoDiscrete<T>::~CpuSolverPseudoDiscrete()
{
    // exp_dw vectors are automatically cleaned up;
    // the shared components (FFT, Pseudo) are owning members of the base
}

//------------------------------------------------------------------------------
// Get stress Boltzmann bond factor
//------------------------------------------------------------------------------
template <typename T>
const double* CpuSolverPseudoDiscrete<T>::get_stress_boltz_bond(
    std::string monomer_type, bool is_half_bond_length) const
{
    // Discrete chains include boltz_bond factor in stress computation
    // Discrete chains always use ds_index=0 (global ds)
    if (is_half_bond_length)
        return this->pseudo->get_boltz_bond_half(monomer_type, 0);
    else
        return this->pseudo->get_boltz_bond(monomer_type, 0);
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::update_dw(std::map<std::string, const T*> w_input)
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
            exp_dw_vec.resize(M_use);

            for (int i = 0; i < M_use; ++i)
                exp_dw_vec[i] = std::exp(-w[i] * local_ds);
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    try
    {
        const int M_full = this->cb->get_total_grid();
        const int M_reduced = (this->space_group_ != nullptr)
            ? this->space_group_->get_n_reduced_basis()
            : M_full;
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // The CrysFFT path works on the physical grid and returns the reduced basis;
        // the other paths work on the full grid
        const bool crysfft_path = this->use_crysfft() && this->space_group_ != nullptr;

        // Work array, kept per thread (called from OpenMP parallel regions)
        int coeff_size = this->complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;
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
            // Reduced basis input/output: expand input, reduce output
            q_full_in_local.resize(M_full);
            q_full_out_local.resize(M_full);

            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);

            fft_in = q_full_in_local.data();
            fft_out = q_full_out_local.data();
        }

        if (crysfft_path)
        {
            // Bond function ĝ(k), then exp(-w*ds), on the CrysFFT physical grid, from and into the
            // reduced basis (real fields only, init_crysfft)
            if constexpr (std::is_same_v<T, double>)
            {
                auto [phys_in, phys_out] = this->crysfft_grids(work);
                const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
                this->fill_crysfft_from_reduced(q_in, phys_in);
                this->crysfft_step(phys_in, phys_out, coeff_full, nullptr, _exp_dw);
                this->reduce_crysfft_to_reduced(phys_out, q_out);
            }
        }
        else
        {
            // Forward transform -> multiply by bond function ĝ(k) -> Backward transform
            // -> exp(-w*ds) in real space
            const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
            this->transform_forward(fft_in, k_q_in.data());
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

//------------------------------------------------------------------------------
// Advance propagator half bond step
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    T* q_in, T* q_out, std::string monomer_type)
{
    try
    {
        const int M_full = this->cb->get_total_grid();
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // The CrysFFT path works on the physical grid and returns the reduced basis;
        // the other paths work on the full grid
        const bool crysfft_path = this->use_crysfft() && this->space_group_ != nullptr;

        // Work array, kept per thread (called from OpenMP parallel regions)
        int coeff_size = this->complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;
        typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
        std::vector<double>& k_q_in = work.k_1;
        if (!crysfft_path)
            k_q_in.resize(coeff_size);

        // Determine input/output pointers based on space group
        T* fft_in = q_in;
        T* fft_out = q_out;

        // Work arrays of this thread for expand/reduce (called from OpenMP parallel regions);
        // the CrysFFT path reads and writes the reduced basis directly
        std::vector<T>& q_full_in_local = work.q_full_in;
        std::vector<T>& q_full_out_local = work.q_full_out;

        if (this->space_group_ != nullptr && !crysfft_path)
        {
            // Reduced basis input/output: expand input, reduce output
            q_full_in_local.resize(M_full);
            q_full_out_local.resize(M_full);

            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);

            fft_in = q_full_in_local.data();
            fft_out = q_full_out_local.data();
        }

        if (crysfft_path)
        {
            // Half-bond function ĝ^(1/2)(k) on the CrysFFT physical grid, from and into the reduced
            // basis (real fields only, init_crysfft)
            if constexpr (std::is_same_v<T, double>)
            {
                auto [phys_in, phys_out] = this->crysfft_grids(work);
                const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, 0, true);
                this->fill_crysfft_from_reduced(q_in, phys_in);
                this->crysfft_step(phys_in, phys_out, coeff_half, nullptr, nullptr);
                this->reduce_crysfft_to_reduced(phys_out, q_out);
            }
        }
        else
        {
            // Forward transform -> multiply by half-bond function ĝ^(1/2)(k) -> Backward transform
            // (discrete chains always use ds_index=0, the global ds)
            const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, 0);
            this->transform_forward(fft_in, k_q_in.data());
            this->transform_backward(k_q_in.data(), _boltz_bond_half, fft_out, nullptr);

            // Reduce full grid → reduced basis
            if (this->space_group_ != nullptr)
            {
                if constexpr (std::is_same_v<T, double>)
                    this->space_group_->to_reduced_basis(fft_out, q_out, 1);
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoDiscrete);
