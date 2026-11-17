/**
 * @file CudaCrysFFT.cu
 * @brief CUDA implementation of crystallographic FFT using DCT-II/III.
 *
 * Uses CudaRealTransform for DCT-II (forward) and DCT-III (backward).
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include "CudaCrysFFT.h"
#include "CudaRealTransform.h"
#include "CudaCommon.h"
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <cstdlib>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// CUDA Kernels
//------------------------------------------------------------------------------

/**
 * @brief Apply Boltzmann factor element-wise.
 *
 * Note: Boltzmann factor includes pre-multiplied normalization (1/M_logical)
 * to eliminate a separate normalize kernel.
 */
__global__ void kernel_apply_boltzmann_crys(
    double* __restrict__ d_data,
    const double* __restrict__ d_boltz,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < M) {
        d_data[idx] *= d_boltz[idx];
        idx += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Apply raw multiplier with explicit normalization factor.
 */
__global__ void kernel_apply_multiplier_crys(
    double* __restrict__ d_data,
    const double* __restrict__ d_multiplier,
    double norm,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < M) {
        d_data[idx] *= d_multiplier[idx] * norm;
        idx += blockDim.x * gridDim.x;
    }
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
CudaCrysFFT::CudaCrysFFT(std::array<int, 3> nx_logical)
    : nx_logical_(nx_logical)
{
    // The owning members release what was acquired if a step below throws.
    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
            throw_with_line_number("CudaCrysFFT requires even grid dimensions.");
    }

    // Physical grid: (N/2) in each dimension
    nx_physical_ = {
        nx_logical_[0] / 2,
        nx_logical_[1] / 2,
        nx_logical_[2] / 2
    };

    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

    // Normalization: 1/M_logical for round-trip
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    // Create CudaRealTransform objects for DCT-II and DCT-III
    dct_forward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_2);
    dct_backward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_3);
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFT::clone() const
{
    return std::make_unique<CudaCrysFFT>(nx_logical_);
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
CudaCrysFFT::~CudaCrysFFT() = default;   // the owning members release the transforms and buffers

//------------------------------------------------------------------------------
// Boltzmann factors
//------------------------------------------------------------------------------
void CudaCrysFFT::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                  double ds, cudaStream_t stream) const
{
    // Compute on host, then copy to device
    std::vector<double> h_boltz(M_physical_);

    double Lx = cell[0];
    double Ly = cell[1];
    double Lz = cell[2];

    int Nx2 = nx_physical_[0];
    int Ny2 = nx_physical_[1];
    int Nz2 = nx_physical_[2];

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double kx = ix * 2.0 * M_PI / Lx;
        double kx2 = kx * kx;

        for (int iy = 0; iy < Ny2; ++iy)
        {
            double ky = iy * 2.0 * M_PI / Ly;
            double ky2 = ky * ky;

            for (int iz = 0; iz < Nz2; ++iz)
            {
                double kz = iz * 2.0 * M_PI / Lz;
                double kz2 = kz * kz;

                // Pre-multiply normalization factor (1/M_logical) into Boltzmann
                // This eliminates a separate normalize kernel in diffusion()
                h_boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds) * norm_factor_;
            }
        }
    }

    upload_table(table, h_boltz, stream);
}

//------------------------------------------------------------------------------
// Apply diffusion operator
//------------------------------------------------------------------------------
void CudaCrysFFT::diffusion(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                            cudaStream_t stream)
{
    set_stream(stream);
    int threads = 256;
    int blocks = grid_blocks(M_physical_, threads);

    static std::atomic<bool> profile_once{false};
    static const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_PMMM") != nullptr);
    const bool profile_this = do_profile && !profile_once.exchange(true);
    // Created only when profiling; CudaEvent destroys them on every exit
    std::vector<CudaEvent> ev(profile_this ? 5 : 0);
    if (profile_this)
    {
        CUDA_CHECK(cudaEventRecord(ev[0], stream_));
    }

    // Determine working buffer to minimize memory copies
    // - If in-place (d_q_in == d_q_out): work directly on input
    // - If out-of-place: copy to output once, then work on output
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;

    if (d_q_in != d_q_out)
    {
        // Single copy at start (instead of two copies before/after)
        CUDA_CHECK(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[1], stream_));

    // Step 1: Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[2], stream_));

    // Step 2: Apply Boltzmann factor (in-place)
    kernel_apply_boltzmann_crys<<<blocks, threads, 0, stream_>>>(d_work, table.get(), M_physical_);
    CUDA_CHECK_LAUNCH();

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[3], stream_));

    // Step 3: Backward DCT-III (in-place)
    // Normalization is already included in Boltzmann factor (pre-multiplied in build_boltzmann)
    dct_backward_->execute(d_work);

    if (profile_this)
    {
        CUDA_CHECK(cudaEventRecord(ev[4], stream_));
        CUDA_CHECK(cudaEventSynchronize(ev[4]));
        float t_copy = 0.0f, t_fwd = 0.0f, t_boltz = 0.0f, t_bwd = 0.0f, t_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t_copy, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&t_fwd, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&t_boltz, ev[2], ev[3]));
        CUDA_CHECK(cudaEventElapsedTime(&t_bwd, ev[3], ev[4]));
        CUDA_CHECK(cudaEventElapsedTime(&t_total, ev[0], ev[4]));

        std::cout << "[CrysFFT-Pmmm profile] M_phys=" << M_physical_
                  << " copy(ms)=" << t_copy
                  << " fwd(ms)=" << t_fwd
                  << " boltz(ms)=" << t_boltz
                  << " bwd(ms)=" << t_bwd
                  << " total(ms)=" << t_total << std::endl;

    }
}

void CudaCrysFFT::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    if (dct_forward_)
        dct_forward_->set_stream(stream_);
    if (dct_backward_)
        dct_backward_->set_stream(stream_);
}

//------------------------------------------------------------------------------
// Apply arbitrary k-space multiplier (for stress computation)
//------------------------------------------------------------------------------
void CudaCrysFFT::apply_multiplier(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                                   cudaStream_t stream)
{
    set_stream(stream);
    int threads = 256;
    int blocks = grid_blocks(M_physical_, threads);

    // Same in-place strategy as diffusion(): work directly on the output buffer
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    // Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    // Apply raw multiplier with round-trip normalization
    kernel_apply_multiplier_crys<<<blocks, threads, 0, stream_>>>(d_work, table.get(), norm_factor_, M_physical_);
    CUDA_CHECK_LAUNCH();

    // Backward DCT-III (in-place)
    dct_backward_->execute(d_work);
}

//------------------------------------------------------------------------------
// Stress multiplier exp(-k²·coeff)·Σ k_d² over an axis mask on the physical DCT grid
//------------------------------------------------------------------------------
void CudaCrysFFT::build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                   int axes, double coeff, cudaStream_t stream) const
{
    if (axes < 1 || axes > 7)
        throw_with_line_number("CudaCrysFFT::build_multiplier: axes must be a mask of 1 (x), 2 (y), 4 (z).");

    const double factor_x = 2.0 * M_PI / cell[0];
    const double factor_y = 2.0 * M_PI / cell[1];
    const double factor_z = 2.0 * M_PI / cell[2];

    std::vector<double> h_mult(M_physical_);
    int idx = 0;
    for (int ix = 0; ix < nx_physical_[0]; ++ix)
    {
        const double kx = ix * factor_x;
        const double kx2 = kx * kx;
        for (int iy = 0; iy < nx_physical_[1]; ++iy)
        {
            const double ky = iy * factor_y;
            const double ky2 = ky * ky;
            for (int iz = 0; iz < nx_physical_[2]; ++iz)
            {
                const double kz = iz * factor_z;
                const double kz2 = kz * kz;
                const double axis_k2 = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
                // Without the bond factor (coeff 0, continuous chains) the multiplier is Σ k_d² alone
                h_mult[idx++] = (coeff == 0.0) ? axis_k2 : std::exp(-(kx2 + ky2 + kz2) * coeff) * axis_k2;
            }
        }
    }
    upload_table(table, h_mult, stream);
}
