/**
 * @file CudaSolverPseudoRQM4.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RQM4.
 *
 * Implements RQM4 (Ranjan-Qin-Morse 4th-order using Richardson extrapolation)
 * for advancing chain propagators using cuFFT for Fourier transforms and
 * multiple CUDA streams for concurrent computation.
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * The propagator is advanced using two step sizes combined for 4th-order:
 * - Full step ds: O(ds²) accuracy
 * - Two half steps ds/2: O(ds²) accuracy
 * - Combination: (4*q_half - q_full)/3 gives O(ds⁴) accuracy
 *
 * **cuFFT Plans:**
 *
 * - cuFFT plans and DCT/DST objects are owned by CudaSolverPseudoBase
 *   (fft_forward / fft_backward / fft_)
 * - Stream-associated for concurrent execution
 *
 * **Stress Calculation:**
 *
 * CudaSolverPseudoBase::compute_single_segment_stress() (shared with RK2 and
 * the discrete chain) computes dQ/dL by multiplying propagator products in
 * Fourier space by weighted basis functions; continuous chains have no bond
 * factor in it.
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoRQM4<double>: Real field solver
 * - CudaSolverPseudoRQM4<std::complex<double>>: Complex field solver
 *
 * @see CudaPseudo for Boltzmann factors
 * @see ComputationContinuous for the order of the propagator computation
 */

#include <iostream>
#include <cmath>
#include <array>

#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaSolverPseudoRQM4.h"
#include "CudaCrysFFT.h"

template <typename T>
CudaSolverPseudoRQM4<T>::CudaSolverPseudoRQM4(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group,
    const int* d_reduced_basis_indices,
    const int* d_full_to_reduced_map)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams,
                              space_group, d_reduced_basis_indices, d_full_to_reduced_map, /*max_fft_batch=*/2,
                              /*with_exp_dw_half=*/true,
                              /*complex_requires_periodic=*/true)
{
    try{
        // The CrysFFT objects and the work buffers (allocate_work()) of the path the space group chooses
        this->init_path();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}
template <typename T>
CudaSolverPseudoRQM4<T>::~CudaSolverPseudoRQM4() = default;

template <typename T>
void CudaSolverPseudoRQM4<T>::allocate_work(int n_field, bool)
{
    const int M = cb->get_total_grid();
    for(int i=0; i<n_streams; i++)
    {
        // The full step and the two half steps (both paths)
        provide(d_q_step_1_one[i], n_field);
        provide(d_q_step_2_one[i], n_field);
        // The full-grid steps with periodic BC transform the full step and the first
        // half step in one batch
        if (!use_crysfft_ && is_periodic_)
            provide(d_q_step_1_two[i], 2*static_cast<size_t>(M));
    }
}

template <typename T>
void CudaSolverPseudoRQM4<T>::update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w)
{
    // exp(-w ds/2) for the full step and exp(-w ds/4) for the half step
    this->update_dw_impl(d_w, true, 0.5, 0.25);
}
// Advance propagator using RQM4
template <typename T>
void CudaSolverPseudoRQM4<T>::advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = this->total_complex_grid();

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw.at(ds_index).at(monomer_type);
        CuDeviceData<T> *_d_exp_dw_half = this->d_exp_dw_half.at(ds_index).at(monomer_type);
        if (use_crysfft_)
        {
            // [real-only feature] Space-group reduced basis and CrysFFT are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (!std::is_same_v<T, double>)
            {
                throw_with_line_number("CrysFFT path is only supported for real fields.");
            }
            else
            {
                const double coeff_full = this->crysfft_coeff(monomer_type, ds_index, 6.0);
                const double coeff_half = this->crysfft_coeff(monomer_type, ds_index, 12.0);

                double* d_full_step = reinterpret_cast<double*>(d_q_step_1_one[STREAM].get());
                double* d_half_step = reinterpret_cast<double*>(d_q_step_2_one[STREAM].get());

                if (crysfft_identity_map_)
                {
                    // Full step and two half steps; their last factors exp(-w*ds/2)
                    // and exp(-w*ds/4) and the mask go into the Richardson combination
                    this->crysfft_diffuse(STREAM, d_q_in, d_full_step, coeff_full, _d_exp_dw, nullptr, nullptr);
                    this->crysfft_diffuse(STREAM, d_q_in, d_half_step, coeff_half, _d_exp_dw_half, _d_exp_dw, nullptr);

                    ker_lin_comb_exp<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                        d_q_out,
                        d_half_step, _d_exp_dw_half, 4.0/3.0,
                        d_full_step, _d_exp_dw, -1.0/3.0,
                        d_q_mask, crysfft_physical_size_);
                    CUDA_CHECK_LAUNCH();
                    return;
                }

                // ===== Full step and two half steps on the reduced basis =====
                this->crysfft_diffuse(STREAM, d_q_in, d_full_step, coeff_full, _d_exp_dw, nullptr, _d_exp_dw);
                this->crysfft_diffuse(STREAM, d_q_in, d_half_step, coeff_half, _d_exp_dw_half, _d_exp_dw, _d_exp_dw_half);

                // ===== Richardson extrapolation on reduced basis =====
                ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                    d_q_out, 4.0/3.0, d_half_step, -1.0/3.0, d_full_step, n_basis_);
                CUDA_CHECK_LAUNCH();

                this->multiply_mask(STREAM, d_q_out, d_q_mask);
                return;
            }
        }

        // Bond factors of the full grid (the Pseudo tables exist on this path only)
        const double* _d_boltz_bond      = pseudo->get_boltz_bond     (monomer_type, ds_index);
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // Full-grid input/output (expanded from / reduced to the basis with a space group)
        CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
        CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

        // The last factor exp(-w*ds/4) of the two half steps (with the FFT
        // normalization) is applied inside the Richardson combination below
        double half_step_scale = 1.0;
        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use batched cuFFT with complex coefficients
            // ============================================================

            // step 1/2: Evaluate exp(-w*ds/2) in real space
            // step 1/4: Evaluate exp(-w*ds/4) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                (d_q_step_1_two[STREAM].get() + 0), fft_in, _d_exp_dw,
                (d_q_step_1_two[STREAM].get() + M), fft_in, _d_exp_dw_half, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Execute a forward FFT
            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_1_two[STREAM], d_qk_[STREAM], 2);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                (d_qk_[STREAM].get() + 0),         _d_boltz_bond,
                (d_qk_[STREAM].get() + M_COMPLEX), _d_boltz_bond_half, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Execute a backward FFT
            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk_[STREAM], d_q_step_1_two[STREAM], 2);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Evaluate exp(-w*ds/2) in real space
            // step 1/4: Evaluate exp(-w*ds/2) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                d_q_step_1_one[STREAM], (d_q_step_1_two[STREAM].get() + 0), _d_exp_dw,
                d_q_step_2_one[STREAM], (d_q_step_1_two[STREAM].get() + M), _d_exp_dw, 1.0/static_cast<double>(M), M);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_2_one[STREAM], d_qk_[STREAM], 1);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_qk_[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk_[STREAM], d_q_step_2_one[STREAM], 1);
            CUDA_CHECK_LAUNCH();

            // step 1/4: exp(-w*ds/4) and 1/M follow in the Richardson combination
            half_step_scale = 1.0/static_cast<double>(M);
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // ============================================================
            // Each stream has its own CudaFFT object with independent work buffers;
            // transforms and kernels are issued on this propagator's stream.

            // Cast device pointers for FFT interface
            T* d_q_step_1 = reinterpret_cast<T*>(d_q_step_1_one[STREAM].get());
            T* d_q_step_2 = reinterpret_cast<T*>(d_q_step_2_one[STREAM].get());

            // ===== Step 1: Full step =====
            // Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_1_one[STREAM], fft_in, _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // Forward transform (DCT/DST)
            fft_[STREAM]->forward_stream(d_q_step_1, d_rk_1_[STREAM], streams[STREAM]);

            // Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_rk_1_[STREAM], d_rk_1_[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward_stream(d_rk_1_[STREAM], d_q_step_1, streams[STREAM]);

            // Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_1_one[STREAM], d_q_step_1_one[STREAM], _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // ===== Step 2: Two half steps =====
            // First half step
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_2_one[STREAM], fft_in, _d_exp_dw_half, 1.0, M);
            CUDA_CHECK_LAUNCH();

            fft_[STREAM]->forward_stream(d_q_step_2, d_rk_2_[STREAM], streams[STREAM]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_rk_2_[STREAM], d_rk_2_[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            fft_[STREAM]->backward_stream(d_rk_2_[STREAM], d_q_step_2, streams[STREAM]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // Second half step
            fft_[STREAM]->forward_stream(d_q_step_2, d_rk_2_[STREAM], streams[STREAM]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_rk_2_[STREAM], d_rk_2_[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            fft_[STREAM]->backward_stream(d_rk_2_[STREAM], d_q_step_2, streams[STREAM]);
            // exp(-w*ds/4) follows in the Richardson combination
        }

        // ===== RQM4: Richardson extrapolation =====
        // q = 4/3 q(ds/2 twice) - 1/3 q(ds), the half-step result taking its last
        // factor exp(-w*ds/4) (and the FFT normalization) here
        ker_multi_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
            fft_out, 4.0/3.0, d_q_step_2_one[STREAM], _d_exp_dw_half, half_step_scale,
            -1.0/3.0, d_q_step_1_one[STREAM], M);
        CUDA_CHECK_LAUNCH();

        // Reduce to the basis (with a space group) and multiply the mask
        this->store_full_grid(STREAM, fft_out, d_q_out, d_q_mask);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoRQM4<double>;
template class CudaSolverPseudoRQM4<std::complex<double>>;
