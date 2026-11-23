/**
 * @file CudaSolverPseudoDiscrete.cu
 * @brief CUDA pseudo-spectral solver for discrete chain model.
 *
 * Implements propagator advancement for discrete chains using the
 * Chapman-Kolmogorov integral equation via cuFFT.
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
 * **Methods:**
 *
 * - advance_propagator(): Full segment step (bond convolution + full-segment weight)
 * - advance_propagator_half_bond_step(): Half bond convolution at chain ends/junctions
 * - get_stress_boltz_bond(): Bond factor of the shared stress computation
 *   (CudaSolverPseudoBase::compute_single_segment_stress())
 *
 * **cuFFT Plans:**
 *
 * - cuFFT plans and the cosine and sine series are owned by CudaSolverPseudoBase,
 *   whose bond_step() runs either
 * - Stream-associated for concurrent execution
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoDiscrete<double>: Real field solver
 * - CudaSolverPseudoDiscrete<std::complex<double>>: Complex field solver
 *
 * @see ComputationDiscrete for the order of the discrete-chain computation
 */

#include <iostream>
#include <cmath>
#include <array>

#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaSolverPseudoDiscrete.h"
#include "CudaCrysFFTBase.h"

template <typename T>
CudaSolverPseudoDiscrete<T>::CudaSolverPseudoDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, space_group, /*max_fft_batch=*/1,
                              /*with_exp_dw_half=*/false)
{
    // The CrysFFT objects and the work buffers of the path the space group chooses
    this->init_path();
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w)
{
    // exp(-w ds): one full segment per discrete bond step, in the basis of the steps, and in the
    // basis of the propagators when the steps take the full grid of a space group
    this->compute_exp_dw(d_w, /*reduced_basis_tables=*/use_crysfft_, false, 1.0, 0.0, /*basis_copy=*/true);
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
    std::string monomer_type,
    double *d_q_mask, int ds_index)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int M = cb->get_total_grid();

    // Get Boltzmann factors for the correct ds_index
    CuDeviceData<T> *_d_exp_dw = this->d_exp_dw.at(ds_index).at(monomer_type);
    if (use_crysfft_)
    {
        const double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);

        // Bond convolution, then the full-segment weight exp(-w*ds)
        this->crysfft_diffuse(STREAM, d_q_in, d_q_out, coeff_full, nullptr, nullptr, _d_exp_dw);
        this->multiply_mask(STREAM, d_q_out, d_q_mask);
        return;
    }

    // Bond factors of the full grid (the Pseudo tables exist on this path only)
    const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, ds_index);

    // Full-grid input/output (expanded from / reduced to the basis with a space group)
    CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
    CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

    // The forward transform, exp(-k^2 ds/6) in Fourier space and the backward transform
    // (cuFFT for periodic BC, the cosine and sine series for walls); fft_in is kept
    this->bond_step(STREAM, fft_in, fft_out, _d_boltz_bond, 1.0);

    // Evaluate exp(-w*ds) in real space (1/M normalization of the unnormalized transforms)
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
    CUDA_CHECK_LAUNCH();

    // Reduce to the basis (with a space group) and multiply the mask
    this->store_full_grid(STREAM, fft_out, d_q_out, d_q_mask);
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out, std::string monomer_type)
{
    const int M = cb->get_total_grid();
    if (use_crysfft_)
    {
        const double coeff_half = this->get_effective_diffusion_coeff(monomer_type, 0, true);

        this->crysfft_diffuse(STREAM, d_q_in, d_q_out, coeff_half, nullptr, nullptr, nullptr);
        return;
    }

    // Discrete chains always use ds_index=0 (global ds); the Pseudo tables exist on the full-grid path only
    const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, 0);

    // Full-grid input/output (expanded from / reduced to the basis with a space group)
    CuDeviceData<T> *fft_in = this->expand_to_full_grid(STREAM, d_q_in);
    CuDeviceData<T> *fft_out = this->full_grid_output(STREAM, d_q_out);

    // Multiply exp(-k^2 ds/12) in Fourier space with the 1/M normalization
    this->bond_step(STREAM, fft_in, fft_out, _d_boltz_bond_half, 1.0/static_cast<double>(M));

    // Reduce full grid → reduced basis (no mask on the half bond)
    this->store_full_grid(STREAM, fft_out, d_q_out, nullptr);
}

template <typename T>
const double* CudaSolverPseudoDiscrete<T>::get_stress_boltz_bond(
    const std::string& monomer_type, bool is_half_bond_length) const
{
    // Discrete chains include the Boltzmann bond factor exp(-k²·b²·ds/6) in the
    // stress integrand; they always use ds_index = 0 (global ds)
    if (is_half_bond_length)
        return pseudo->get_boltz_bond_half(monomer_type, 0);
    return pseudo->get_boltz_bond(monomer_type, 0);
}

// Explicit template instantiation
template class CudaSolverPseudoDiscrete<double>;
template class CudaSolverPseudoDiscrete<std::complex<double>>;
