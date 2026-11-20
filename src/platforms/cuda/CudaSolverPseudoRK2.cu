/**
 * @file CudaSolverPseudoRK2.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements RK2 (Rasmussen-Kalosakas 2nd-order) for advancing chain propagators
 * using cuFFT for Fourier transforms and multiple CUDA streams for concurrent
 * computation.
 *
 * **RK2 (Operator Splitting):**
 *
 * Single full step with Strang splitting:
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step (2 FFTs) while RQM4 combines full and half
 * steps via Richardson extrapolation (6 FFTs). RK2 is faster but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoRK2<double>: Real field solver
 * - CudaSolverPseudoRK2<std::complex<double>>: Complex field solver
 *
 * @see CudaPseudo for Boltzmann factors
 * @see ComputationContinuous for the order of the propagator computation
 */

#include <iostream>
#include <cmath>
#include <array>

#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaCrysFFT.h"

template <typename T>
CudaSolverPseudoRK2<T>::CudaSolverPseudoRK2(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group,
    const int* d_reduced_basis_indices,
    const int* d_full_to_reduced_map)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams,
                              space_group, d_reduced_basis_indices, d_full_to_reduced_map, /*max_fft_batch=*/1,
                              /*with_exp_dw_half=*/false,
                              /*complex_requires_periodic=*/true)
{
    try{
        // The CrysFFT objects and the work buffers of the path the space group chooses
        this->init_path();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}

template <typename T>
CudaSolverPseudoRK2<T>::~CudaSolverPseudoRK2() = default;

template <typename T>
void CudaSolverPseudoRK2<T>::update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w)
{
    // exp(-w ds/2): the half-step factor applied on both sides of the diffusion step, in the
    // basis of the steps
    this->compute_exp_dw(d_w, /*reduced_basis_tables=*/use_crysfft_, false, 0.5, 0.0);
}

// Advance propagator using RK2 (single full step)
template <typename T>
void CudaSolverPseudoRK2<T>::advance_propagator(
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

                this->crysfft_diffuse(STREAM, d_q_in, d_q_out, coeff_full, _d_exp_dw, nullptr, _d_exp_dw);
                this->multiply_mask(STREAM, d_q_out, d_q_mask);
                return;
            }
        }

        // Bond factors of the full grid (the Pseudo tables exist on this path only)
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, ds_index);

        // Full-grid input/output (expanded from / reduced to the basis with a space group).
        // The step runs in fft_out: the transform reads it before the inverse transform
        // writes the result there
        CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
        CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use cuFFT with complex coefficients
            // ============================================================

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                fft_out, fft_in, _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // Step 2: Execute a forward FFT
            fft_forward(STREAM, fft_out, d_qk_[STREAM], 1);
            CUDA_CHECK_LAUNCH();

            // Step 3: Multiply exp(-k^2 ds/6) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                d_qk_[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // Step 4: Execute a backward FFT
            fft_backward(STREAM, d_qk_[STREAM], fft_out, 1);
            CUDA_CHECK_LAUNCH();

            // Step 5: Evaluate exp(-w*ds/2) in real space and normalize
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // ============================================================

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(fft_out, fft_in, _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // Step 2: Forward transform (DCT/DST)
            fft_[STREAM]->forward_stream(reinterpret_cast<T*>(fft_out), d_rk_1_[STREAM], streams[STREAM]);

            // Step 3: Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_rk_1_[STREAM], d_rk_1_[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // Step 4: Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward_stream(d_rk_1_[STREAM], reinterpret_cast<T*>(fft_out), streams[STREAM]);

            // Step 5: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(fft_out, fft_out, _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();
        }

        // Reduce to the basis (with a space group) and multiply the mask
        this->store_full_grid(STREAM, fft_out, d_q_out, d_q_mask);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoRK2<double>;
template class CudaSolverPseudoRK2<std::complex<double>>;
