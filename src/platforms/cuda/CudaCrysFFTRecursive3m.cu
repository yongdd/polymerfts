/**
 * @file CudaCrysFFTRecursive3m.cu
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 */

#include "CudaCrysFFTRecursive3m.h"

#include "CudaCommon.h"
#include "CrysFFTTables.h"
#include "Exception.h"

#include <cstdint>

namespace {

/// True when p meets cuFFT's alignment (a cufftDoubleComplex, 16 bytes)
bool cufft_aligned(const void* p)
{
    return reinterpret_cast<std::uintptr_t>(p) % alignof(cufftDoubleComplex) == 0;
}

__global__ void ker_apply_k_3m(
    cufftDoubleComplex* __restrict__ dst,
    const cufftDoubleComplex* __restrict__ src,
    const double* __restrict__ k_re0,
    const double* __restrict__ k_re1,
    const double* __restrict__ k_re2,
    const double* __restrict__ k_re3,
    const double* __restrict__ k_re4,
    const double* __restrict__ k_re5,
    const double* __restrict__ k_re6,
    const double* __restrict__ k_re7,
    const double* __restrict__ k_im0,
    const double* __restrict__ k_im1,
    const double* __restrict__ k_im2,
    const double* __restrict__ k_im3,
    const double* __restrict__ k_im4,
    const double* __restrict__ k_im5,
    const double* __restrict__ k_im6,
    const double* __restrict__ k_im7,
    int Nx2, int Ny2, int Nz2, int Nz2c)
{
    const int total = Nx2 * Ny2 * Nz2c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        int iz = idx % Nz2c;
        int tmp = idx / Nz2c;
        int iy = tmp % Ny2;
        int ix = tmp / Ny2;

        int ix_m = (Nx2 - ix) % Nx2;
        int iy_m = (Ny2 - iy) % Ny2;

        size_t base_c = static_cast<size_t>(ix * Ny2 + iy) * Nz2c;
        size_t base_k = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
        size_t base_c_110 = static_cast<size_t>(ix_m * Ny2 + iy_m) * Nz2c;
        size_t base_c_010 = static_cast<size_t>(ix * Ny2 + iy_m) * Nz2c;
        size_t base_c_100 = static_cast<size_t>(ix_m * Ny2 + iy) * Nz2c;
        size_t idx_k = base_k + iz;

        cufftDoubleComplex s000 = src[base_c + iz];
        cufftDoubleComplex s110 = src[base_c_110 + iz];
        cufftDoubleComplex s010 = src[base_c_010 + iz];
        cufftDoubleComplex s100 = src[base_c_100 + iz];

        double out_re =
            s000.x * k_re0[idx_k] - s000.y * k_im7[idx_k] +
            s110.x * k_re6[idx_k] - s110.y * k_im1[idx_k] +
            s010.x * k_re2[idx_k] - s010.y * k_im5[idx_k] +
            s100.x * k_re4[idx_k] - s100.y * k_im3[idx_k];

        double out_im =
            s000.x * k_im0[idx_k] + s000.y * k_re7[idx_k] +
            s110.x * k_im6[idx_k] + s110.y * k_re1[idx_k] +
            s010.x * k_im2[idx_k] + s010.y * k_re5[idx_k] +
            s100.x * k_im4[idx_k] + s100.y * k_re3[idx_k];

        dst[base_c + iz].x = out_re;
        dst[base_c + iz].y = out_im;

        idx += blockDim.x * gridDim.x;
    }
}

}  // namespace

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 9> translational_part)
    : tables_(std::make_shared<const Recursive3mTables>(nx_logical, translational_part))
{
    nx_physical_ = tables_->nx_physical();
    M_physical_ = tables_->M_physical();
    M_complex_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);

    d_step1_.allocate(M_complex_);
    d_step2_.allocate(M_complex_);

    init_plans();
}

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(const CudaCrysFFTRecursive3m& other)
    : tables_(other.tables_),
      nx_physical_(other.nx_physical_),
      M_physical_(other.M_physical_),
      M_complex_(other.M_complex_)
{
    d_step1_.allocate(M_complex_);
    d_step2_.allocate(M_complex_);
    init_plans();
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTRecursive3m::clone() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTRecursive3m(*this));
}

void CudaCrysFFTRecursive3m::init_plans()
{
    if (plan_r2c_.create(cufftPlan3d, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_D2Z) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT D2Z plan for CudaCrysFFTRecursive3m.");
    if (plan_c2r_.create(cufftPlan3d, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_Z2D) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT Z2D plan for CudaCrysFFTRecursive3m.");
}

void CudaCrysFFTRecursive3m::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                             double coeff, cudaStream_t stream) const
{
    // The 16 arrays of the table one after another (stride M_physical)
    std::vector<double> h_table;
    tables_->boltzmann(h_table, M_physical_, cell, coeff);
    upload_table(table, h_table, stream);
}

void CudaCrysFFTRecursive3m::build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                              int axes, double coeff, cudaStream_t stream) const
{
    std::vector<double> h_table;
    tables_->multiplier(h_table, M_physical_, cell, axes, coeff);
    upload_table(table, h_table, stream);
}

void CudaCrysFFTRecursive3m::diffusion(
    double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table, cudaStream_t stream)
{
    use_stream(stream);

    // The out-of-place D2Z keeps d_q_in, and the Z2D reads only d_step2_, so an
    // out-of-place call preserves its input without a copy; d_q_out may alias d_q_in.
    // cuFFT needs the arrays it reads and writes aligned to a cufftDoubleComplex, as
    // the work buffers and propagators are.
    if (!cufft_aligned(d_q_in) || !cufft_aligned(d_q_out))
        throw_with_line_number("CudaCrysFFTRecursive3m: the input and output must be aligned to a cufftDoubleComplex.");
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_, d_q_in, d_step1_));

    const int threads = 256;
    const int blocks = grid_blocks(M_complex_, threads);

    const double* k = table.get();
    const size_t M = M_physical_;
    ker_apply_k_3m<<<blocks, threads, 0, stream_>>>(
        d_step2_, d_step1_,
        k, k + M, k + 2 * M, k + 3 * M, k + 4 * M, k + 5 * M, k + 6 * M, k + 7 * M,
        k + 8 * M, k + 9 * M, k + 10 * M, k + 11 * M, k + 12 * M, k + 13 * M, k + 14 * M, k + 15 * M,
        nx_physical_[0], nx_physical_[1], nx_physical_[2], nx_physical_[2] / 2 + 1);
    CUDA_CHECK_LAUNCH();

    CUFFT_CHECK(cufftExecZ2D(plan_c2r_, d_step2_, d_q_out));
}

void CudaCrysFFTRecursive3m::apply_multiplier(
    double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table, cudaStream_t stream)
{
    diffusion(d_q_in, d_q_out, table, stream);
}

void CudaCrysFFTRecursive3m::use_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
}
