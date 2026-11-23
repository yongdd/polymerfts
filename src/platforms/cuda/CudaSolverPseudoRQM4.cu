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
 * - cuFFT plans and the cosine and sine series are owned by CudaSolverPseudoBase
 *   (fft_forward / fft_backward for the batch of periodic BC, bond_step() otherwise)
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
#include "CudaDeviceOps.cuh"
#include "CudaSolverPseudoRQM4.h"
#include "CudaCrysFFT.h"

namespace
{
/// dst = a*(scale*src1*w1) + b*src2: ker_multi(src1, w1, scale) followed by
/// ker_lin_comb in one pass, with the same arithmetic (the Richardson step)
__global__ void ker_multi_lin_comb(double* dst, double a, const double* src1, const double* w1, double scale,
                                   double b, const double* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        const double t = scale * src1[i] * w1[i];
        dst[i] = a * t + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, const cuDoubleComplex* w1, double scale,
                                   double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Same arithmetic as the complex ker_multi
        const double tx = scale * (src1[i].x * w1[i].x - src1[i].y * w1[i].y);
        const double ty = scale * (src1[i].x * w1[i].y + src1[i].y * w1[i].x);
        dst[i].x = a * tx + b * src2[i].x;
        dst[i].y = a * ty + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

/// dst = (a*(src1*exp1) + b*(src2*exp2)) * mask: the Richardson step of the CrysFFT identity map
/// (real fields; the complex solver instantiates it without running it)
template <typename D>
__global__ void ker_lin_comb_exp(
    D* dst,
    const D* src1, const D* exp1, double a,
    const D* src2, const D* exp2, double b,
    const double* mask, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        D val = added(scaled(a, multiplied(src1[i], exp1[i])), scaled(b, multiplied(src2[i], exp2[i])));
        if (mask != nullptr)
            val = scaled(mask[i], val);
        dst[i] = val;
        i += blockDim.x * gridDim.x;
    }
}

/// Two fields times their factors in one pass: the full and the first half step of the batch
__global__ void ker_multi_exp_dw_two(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i] = a * src1[i] * exp_dw1[i];
        dst2[i] = a * src2[i] * exp_dw2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_exp_dw_two(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use temporary variables to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst1, temp_dst2;

        temp_dst1.x = a * (src1[i].x * exp_dw1[i].x - src1[i].y * exp_dw1[i].y);
        temp_dst1.y = a * (src1[i].x * exp_dw1[i].y + src1[i].y * exp_dw1[i].x);

        temp_dst2.x = a * (src2[i].x * exp_dw2[i].x - src2[i].y * exp_dw2[i].y);
        temp_dst2.y = a * (src2[i].x * exp_dw2[i].y + src2[i].y * exp_dw2[i].x);

        dst1[i] = temp_dst1;
        dst2[i] = temp_dst2;

        i += blockDim.x * gridDim.x;
    }
}

/// The spectra of the batch times their bond factors
__global__ void ker_complex_real_multi_bond_two(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x *= boltz_bond1[i];
        dst1[i].y *= boltz_bond1[i];
        dst2[i].x *= boltz_bond2[i];
        dst2[i].y *= boltz_bond2[i];
        i += blockDim.x * gridDim.x;
    }
}
}  // namespace

template <typename T>
CudaSolverPseudoRQM4<T>::CudaSolverPseudoRQM4(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, space_group, /*max_fft_batch=*/2,
                              /*with_exp_dw_half=*/true)
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
void CudaSolverPseudoRQM4<T>::allocate_work(int n_field)
{
    // The full-grid steps with periodic BC transform the full step and the first half
    // step in one batch, and the full step stays in the first half of the batch
    const bool batched = !use_crysfft_ && is_periodic_;
    const int M = cb->get_total_grid();
    for(int i=0; i<n_streams; i++)
    {
        // The two half steps (both paths), and the full step outside the batch
        provide(d_q_step_2_one[i], n_field);
        if (batched)
            provide(d_q_step_1_two[i], 2*static_cast<size_t>(M));
        else
            provide(d_q_step_1_one[i], n_field);
    }
}

template <typename T>
void CudaSolverPseudoRQM4<T>::update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w)
{
    // exp(-w ds/2) for the full step and exp(-w ds/4) for the half step, in the basis of the steps
    this->compute_exp_dw(d_w, /*reduced_basis_tables=*/use_crysfft_, true, 0.5, 0.25);
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
            const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
            const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, ds_index, true);

            CuDeviceData<T>* d_full_step = d_q_step_1_one[STREAM];
            CuDeviceData<T>* d_half_step = d_q_step_2_one[STREAM];

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

        // Bond factors of the full grid (the Pseudo tables exist on this path only)
        const double* _d_boltz_bond      = pseudo->get_boltz_bond     (monomer_type, ds_index);
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // Full-grid input/output (expanded from / reduced to the basis with a space group)
        CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
        CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

        // The full step: the first half of the batch (periodic BC), or d_q_step_1_one
        CuDeviceData<T> *d_full_step = is_periodic_ ? d_q_step_1_two[STREAM].get() : d_q_step_1_one[STREAM].get();

        // The last factor exp(-w*ds/4) of the two half steps (with the FFT
        // normalization) is applied inside the Richardson combination below
        double half_step_scale = 1.0;
        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use batched cuFFT with complex coefficients
            // ============================================================
            // The spectra of the batch, one after the other in the base's scratch
            cuDoubleComplex* d_qk = reinterpret_cast<cuDoubleComplex*>(spectrum(STREAM, 0));

            // step 1/2: Evaluate exp(-w*ds/2) in real space
            // step 1/4: Evaluate exp(-w*ds/4) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                (d_q_step_1_two[STREAM].get() + 0), fft_in, _d_exp_dw,
                (d_q_step_1_two[STREAM].get() + M), fft_in, _d_exp_dw_half, 1.0, M);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Execute a forward FFT
            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_1_two[STREAM], d_qk, 2);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                (d_qk + 0),         _d_boltz_bond,
                (d_qk + M_COMPLEX), _d_boltz_bond_half, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Execute a backward FFT
            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk, d_q_step_1_two[STREAM], 2);
            CUDA_CHECK_LAUNCH();

            // step 1/2: Evaluate exp(-w*ds/2) in real space (in place)
            // step 1/4: Evaluate exp(-w*ds/2) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                d_full_step, d_full_step, _d_exp_dw,
                d_q_step_2_one[STREAM], (d_q_step_1_two[STREAM].get() + M), _d_exp_dw, 1.0/static_cast<double>(M), M);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_2_one[STREAM], d_qk, 1);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_qk, _d_boltz_bond_half, 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();

            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk, d_q_step_2_one[STREAM], 1);
            CUDA_CHECK_LAUNCH();

            // step 1/4: exp(-w*ds/4) and 1/M follow in the Richardson combination
            half_step_scale = 1.0/static_cast<double>(M);
        }
        else
        {
            // ============================================================
            // Walls: one field at a time through the cosine and sine series
            // ============================================================

            // ===== Step 1: Full step =====
            // exp(-w*ds/2), exp(-k^2 ds/6) in Fourier space, exp(-w*ds/2) with 1/M
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_1_one[STREAM], fft_in, _d_exp_dw, 1.0, M);
            CUDA_CHECK_LAUNCH();
            this->bond_step(STREAM, d_q_step_1_one[STREAM], d_q_step_1_one[STREAM], _d_boltz_bond, 1.0);
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                d_q_step_1_one[STREAM], d_q_step_1_one[STREAM], _d_exp_dw, 1.0/static_cast<double>(M), M);
            CUDA_CHECK_LAUNCH();

            // ===== Step 2: Two half steps =====
            // First half step: exp(-w*ds/4), exp(-k^2 ds/12), exp(-w*ds/2) with 1/M
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q_step_2_one[STREAM], fft_in, _d_exp_dw_half, 1.0, M);
            CUDA_CHECK_LAUNCH();
            this->bond_step(STREAM, d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_boltz_bond_half, 1.0);
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw, 1.0/static_cast<double>(M), M);
            CUDA_CHECK_LAUNCH();

            // Second half step: exp(-k^2 ds/12); exp(-w*ds/4) and 1/M follow in the
            // Richardson combination
            this->bond_step(STREAM, d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_boltz_bond_half, 1.0);
            half_step_scale = 1.0/static_cast<double>(M);
        }

        // ===== RQM4: Richardson extrapolation =====
        // q = 4/3 q(ds/2 twice) - 1/3 q(ds), the half-step result taking its last
        // factor exp(-w*ds/4) (and the FFT normalization) here
        ker_multi_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
            fft_out, 4.0/3.0, d_q_step_2_one[STREAM], _d_exp_dw_half, half_step_scale,
            -1.0/3.0, d_full_step, M);
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
