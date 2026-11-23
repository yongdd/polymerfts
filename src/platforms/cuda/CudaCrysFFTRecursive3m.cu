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

// The k-space product of the fold on the half spectrum (Nz2c = Nz2/2 + 1 stored z modes).
// `table` holds 16 arrays of M values: the real parts re_j of the eight folded octants
// at table + j*M, then their imaginary parts im_j at table + (8 + j)*M, rows of Nz2 values.
__global__ void ker_apply_k_3m(
    cufftDoubleComplex* __restrict__ dst,
    const cufftDoubleComplex* __restrict__ src,
    const double* __restrict__ table, size_t M,
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
        const double* k = table + base_k + iz;
        auto re = [k, M](int j) { return k[j * M]; };
        auto im = [k, M](int j) { return k[(8 + j) * M]; };

        cufftDoubleComplex s000 = src[base_c + iz];
        cufftDoubleComplex s110 = src[base_c_110 + iz];
        cufftDoubleComplex s010 = src[base_c_010 + iz];
        cufftDoubleComplex s100 = src[base_c_100 + iz];

        double out_re =
            s000.x * re(0) - s000.y * im(7) +
            s110.x * re(6) - s110.y * im(1) +
            s010.x * re(2) - s010.y * im(5) +
            s100.x * re(4) - s100.y * im(3);

        double out_im =
            s000.x * im(0) + s000.y * re(7) +
            s110.x * im(6) + s110.y * re(1) +
            s010.x * im(2) + s010.y * re(5) +
            s100.x * im(4) + s100.y * re(3);

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
    const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table, cudaStream_t stream)
{
    use_stream(stream);

    // The out-of-place D2Z keeps d_q_in (cuFFT takes it as non-const all the same), and
    // the Z2D reads only d_step2_, so an out-of-place call preserves its input without a
    // copy; d_q_out may alias d_q_in. cuFFT needs the arrays it reads and writes aligned
    // to a cufftDoubleComplex, as the work buffers and propagators are.
    if (!cufft_aligned(d_q_in) || !cufft_aligned(d_q_out))
        throw_with_line_number("CudaCrysFFTRecursive3m: the input and output must be aligned to a cufftDoubleComplex.");
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_, const_cast<double*>(d_q_in), d_step1_));

    const int threads = 256;
    const int blocks = grid_blocks(M_complex_, threads);

    ker_apply_k_3m<<<blocks, threads, 0, stream_>>>(
        d_step2_, d_step1_, table.get(), static_cast<size_t>(M_physical_),
        nx_physical_[0], nx_physical_[1], nx_physical_[2], nx_physical_[2] / 2 + 1);
    CUDA_CHECK_LAUNCH();

    CUFFT_CHECK(cufftExecZ2D(plan_c2r_, d_step2_, d_q_out));
}

void CudaCrysFFTRecursive3m::use_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
}
