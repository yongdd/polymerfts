/**
 * @file CudaSolverPseudoRK2.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements RK2 (Rasmussen-Kalosakas 2nd-order) for advancing chain propagators
 * using cuFFT (periodic BC) or the cosine and sine series (walls) and multiple CUDA
 * streams for concurrent computation.
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
    SpaceGroup* space_group)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, space_group, /*max_fft_batch=*/1,
                              /*with_exp_dw_half=*/false)
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

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw.at(ds_index).at(monomer_type);
        if (use_crysfft_)
        {
            const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);

            this->crysfft_diffuse(STREAM, d_q_in, d_q_out, coeff_full, _d_exp_dw, nullptr, _d_exp_dw);
            this->multiply_mask(STREAM, d_q_out, d_q_mask);
            return;
        }

        // Bond factors of the full grid (the Pseudo tables exist on this path only)
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, ds_index);

        // Full-grid input/output (expanded from / reduced to the basis with a space group).
        // The step runs in fft_out: the transform reads it before the inverse transform
        // writes the result there
        CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
        CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

        // Step 1: Evaluate exp(-w*ds/2) in real space
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
            fft_out, fft_in, _d_exp_dw, 1.0, M);
        CUDA_CHECK_LAUNCH();

        // Steps 2-4: the forward transform, exp(-k^2 ds/6) in Fourier space and the
        // backward transform (cuFFT for periodic BC, the cosine and sine series for walls)
        this->bond_step(STREAM, fft_out, fft_out, _d_boltz_bond, 1.0);

        // Step 5: Evaluate exp(-w*ds/2) in real space and normalize
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
            fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
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
template class CudaSolverPseudoRK2<double>;
template class CudaSolverPseudoRK2<std::complex<double>>;
