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
        // Boltzmann factors exp(-w·ds/2) and exp(-w·ds/4) of this contour step
        const T* _exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();
        const T* _exp_dw_half = this->exp_dw_half.at(ds_index).at(monomer_type).data();

        // The results of the full step and of the two half steps, work arrays of this
        // thread (called from OpenMP parallel regions)
        typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
        std::vector<T>& q_out1 = work.q_1;
        std::vector<T>& q_out2 = work.q_2;

        if (this->use_crysfft())
        {
            // [real-only feature] The CrysFFT steps fold real fields on the physical sub-grid (init_crysfft())
            if constexpr (!std::is_same_v<T, double>)
            {
                throw_with_line_number("CrysFFT path is only supported for real fields.");
            }
            else
            {
                const int M = this->n_basis_;
                const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
                const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, ds_index, true);
                q_out1.resize(M);
                q_out2.resize(M);

                // Full step exp(-w·ds/2) exp(b²ds/6 ∇²) exp(-w·ds/2), and two half steps with
                // exp(-w·ds/4) outside and exp(-w·ds/2) between them, in the reduced basis
                this->crysfft_diffuse(q_in, q_out1.data(), coeff_full, _exp_dw, nullptr, _exp_dw);
                this->crysfft_diffuse(q_in, q_out2.data(), coeff_half, _exp_dw_half, _exp_dw, _exp_dw_half);

                // ===== RQM4: Richardson extrapolation =====
                for (int i = 0; i < M; ++i)
                    q_out[i] = (4.0 * q_out2[i] - q_out1[i]) / 3.0;
                this->multiply_mask(q_out, q_mask);
                return;
            }
        }

        const int M = this->cb->get_total_grid();
        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // Full-grid input/output (expanded from / reduced to the basis with a space group) and
        // the spectra; the Richardson extrapolation writes fft_out after the steps have read fft_in
        T* fft_in = this->expand_to_full_grid(q_in, work.q_full);
        T* fft_out = this->full_grid_output(q_out, work.q_full);
        std::vector<double>& k_q_in1 = work.k_1;
        std::vector<double>& k_q_in2 = work.k_2;
        q_out1.resize(M);
        q_out2.resize(M);
        k_q_in1.resize(this->spectrum_size());
        k_q_in2.resize(this->spectrum_size());

        // ===== Step 1: one full step (q_out1); Step 2: two half steps (q_out2) =====
        // The full step and the first half step start from the same input
        for (int i = 0; i < M; ++i)
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

        // ===== RQM4: Richardson extrapolation =====
        for (int i = 0; i < M; ++i)
        {
            const T q_full = q_out1[i] * _exp_dw[i];
            const T q_half = q_out2[i] * _exp_dw_half[i];
            fft_out[i] = (4.0 * q_half - q_full) / 3.0;
        }

        // Reduce to the basis (with a space group) and multiply the mask
        this->store_full_grid(fft_out, q_out, q_mask);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRQM4);
