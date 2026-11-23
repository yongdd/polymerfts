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
CpuSolverPseudoDiscrete<T>::CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, SpaceGroup* space_group, FFTBackend backend)
    : CpuSolverPseudoBase<T>(cb, molecules, space_group, /*with_half=*/false)
{
    // Initialize shared components (FFT, Pseudo, etc.)
    this->init_shared(backend);
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
    // exp(-w·ds), in the reduced basis on the CrysFFT path
    this->compute_exp_dw(w_input, /*reduced_basis_tables=*/this->use_crysfft(), false, 1.0, 0.0);
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    // Boltzmann factor exp(-w·ds) of this contour step
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
            // Bond function ĝ(k), then exp(-w*ds), in the reduced basis
            const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
            this->crysfft_diffuse(q_in, q_out, coeff_full, nullptr, nullptr, _exp_dw);
            this->multiply_mask(q_out, q_mask);
            return;
        }
    }

    const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);

    // Full-grid input/output (expanded from / reduced to the basis with a space group) and
    // the spectrum, work arrays of this thread (called from OpenMP parallel regions); the
    // forward transform reads fft_in before the backward transform writes fft_out
    typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
    T* fft_in = this->expand_to_full_grid(q_in, work.q_full);
    T* fft_out = this->full_grid_output(q_out, work.q_full);
    std::vector<double>& k_q = work.k_1;
    k_q.resize(this->spectrum_size());

    // Forward transform -> multiply by bond function ĝ(k) -> Backward transform
    // -> exp(-w*ds) in real space
    this->transform_forward(fft_in, k_q.data());
    this->transform_backward(k_q.data(), _boltz_bond, fft_out, _exp_dw);

    // Reduce to the basis (with a space group) and multiply the mask
    this->store_full_grid(fft_out, q_out, q_mask);
}

//------------------------------------------------------------------------------
// Advance propagator half bond step
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    T* q_in, T* q_out, std::string monomer_type)
{
    if (this->use_crysfft())
    {
        // [real-only feature] The CrysFFT steps fold real fields on the physical sub-grid (init_crysfft())
        if constexpr (!std::is_same_v<T, double>)
        {
            throw_with_line_number("CrysFFT path is only supported for real fields.");
        }
        else
        {
            // Half-bond function ĝ^(1/2)(k) in the reduced basis
            const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, 0, true);
            this->crysfft_diffuse(q_in, q_out, coeff_half, nullptr, nullptr, nullptr);
            return;
        }
    }

    // Discrete chains always use ds_index=0 (the global ds)
    const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, 0);

    // Full-grid input/output (expanded from / reduced to the basis with a space group) and
    // the spectrum, work arrays of this thread (called from OpenMP parallel regions)
    typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
    T* fft_in = this->expand_to_full_grid(q_in, work.q_full);
    T* fft_out = this->full_grid_output(q_out, work.q_full);
    std::vector<double>& k_q = work.k_1;
    k_q.resize(this->spectrum_size());

    // Forward transform -> multiply by half-bond function ĝ^(1/2)(k) -> Backward transform
    this->transform_forward(fft_in, k_q.data());
    this->transform_backward(k_q.data(), _boltz_bond_half, fft_out, nullptr);

    // Reduce to the basis (with a space group; no mask on the half bond)
    this->store_full_grid(fft_out, q_out, nullptr);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoDiscrete);
