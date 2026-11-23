/**
 * @file CudaCrysFFTObliqueZ.cu
 * @brief CUDA implementation of z-mirror CrysFFT (DCT-z + FFT-xy as one
 *        permuted real 3D FFT, see CudaCrysFFTObliqueZ.h).
 */

#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include "CudaCrysFFTObliqueZ.h"
#include "CudaCommon.h"
#include "CrysFFTObliqueZTable.h"
#include "Exception.h"

namespace {

// Makhoul's permutation of each z line of length n: v[m] = x[2m] for
// m < (n+1)/2 and v[n-1-j] = x[2j+1]
__global__ void ker_permute_z(const double* __restrict__ in, double* __restrict__ out, int total, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int m = idx % n;
        const int src = (m < (n + 1) / 2) ? 2 * m : 2 * (n - 1 - m) + 1;
        out[idx] = in[idx - m + src];
        idx += blockDim.x * gridDim.x;
    }
}

// Inverse of ker_permute_z
__global__ void ker_unpermute_z(const double* __restrict__ in, double* __restrict__ out, int total, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int i = idx % n;
        const int m = (i % 2 == 0) ? i / 2 : n - 1 - i / 2;
        out[idx] = in[idx - i + m];
        idx += blockDim.x * gridDim.x;
    }
}

// S p + D w conj(p_m) for one entry, with f = (S, D)
__device__ inline cufftDoubleComplex obliquez_mode(
    cufftDoubleComplex p, cufftDoubleComplex p_m, double2 f, double2 w)
{
    // w * conj(p_m)
    const double wc_re = w.x * p_m.x + w.y * p_m.y;
    const double wc_im = w.y * p_m.x - w.x * p_m.y;

    cufftDoubleComplex out;
    out.x = f.x * p.x + f.y * wc_re;
    out.y = f.x * p.y + f.y * wc_im;
    return out;
}

// P'(k) = S(k) P(k) + D(k) w(kz) conj(P(-kx, -ky, kz)) on the half spectrum
// (nz_c = n/2 + 1 stored kz), with sd = (S, D) and w = (cos, sin)(pi kz/n).
// In place: k and its mirror (-kx, -ky, kz) need each other's old value, so the
// thread of the lower index of each pair reads both before it writes both; an
// entry that is its own mirror is updated once.
__global__ void ker_obliquez_spectral(
    cufftDoubleComplex* __restrict__ spectrum,
    const double2* __restrict__ sd,
    const double2* __restrict__ twiddle,
    int nx, int ny, int nz_c)
{
    const int total = nx * ny * nz_c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int kz = idx % nz_c;
        const int tmp = idx / nz_c;
        const int iy = tmp % ny;
        const int ix = tmp / ny;
        const int ix_m = (nx - ix) % nx;
        const int iy_m = (ny - iy) % ny;
        const int idx_m = (ix_m * ny + iy_m) * nz_c + kz;

        if (idx <= idx_m)
        {
            const cufftDoubleComplex p = spectrum[idx];
            const cufftDoubleComplex p_m = spectrum[idx_m];
            const double2 w = twiddle[kz];
            spectrum[idx] = obliquez_mode(p, p_m, sd[idx], w);
            if (idx_m != idx)
                spectrum[idx_m] = obliquez_mode(p_m, p, sd[idx_m], w);
        }

        idx += blockDim.x * gridDim.x;
    }
}

}  // namespace

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------
CudaCrysFFTObliqueZ::CudaCrysFFTObliqueZ(std::array<int, 3> nx_logical)
    : nx_logical_(nx_logical)
{
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] <= 0)
            throw_with_line_number("CudaCrysFFTObliqueZ requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("CudaCrysFFTObliqueZ requires even Nz.");

    nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    M_spectral_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    init_transforms();
}

CudaCrysFFTObliqueZ::CudaCrysFFTObliqueZ(const CudaCrysFFTObliqueZ& other)
    : nx_logical_(other.nx_logical_),
      nx_physical_(other.nx_physical_),
      M_logical_(other.M_logical_),
      M_physical_(other.M_physical_),
      M_spectral_(other.M_spectral_),
      d_twiddle_(other.d_twiddle_),
      norm_factor_(other.norm_factor_)
{
    init_transforms();
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTObliqueZ::clone() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTObliqueZ(*this));
}

void CudaCrysFFTObliqueZ::init_transforms()
{
    const int n = nx_physical_[2];
    if (!d_twiddle_)
    {
        // (cos, sin) pairs, the layout of std::complex<double> (shared with the CPU engine)
        const std::vector<std::complex<double>> h_twiddle = obliquez_twiddle(n);
        auto d_twiddle = std::make_shared<CudaDeviceBuffer<double>>(2 * h_twiddle.size());
        CUDA_CHECK(cudaMemcpy(*d_twiddle, h_twiddle.data(), sizeof(std::complex<double>) * h_twiddle.size(),
                              cudaMemcpyHostToDevice));
        d_twiddle_ = std::move(d_twiddle);
    }

    d_real_.allocate(M_physical_);
    d_spectrum_.allocate(M_spectral_);

    if (plan_fwd_.create(cufftPlan3d, nx_physical_[0], nx_physical_[1], n, CUFFT_D2Z) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create its cuFFT D2Z plan.");
    if (plan_bwd_.create(cufftPlan3d, nx_physical_[0], nx_physical_[1], n, CUFFT_Z2D) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create its cuFFT Z2D plan.");
    CUFFT_CHECK(cufftSetStream(plan_fwd_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_bwd_, stream_));
}

//------------------------------------------------------------------------------
// Boltzmann table
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                          double ds, cudaStream_t stream) const
{
    // The (S, D) pairs of the half spectrum, with the normalization (shared with the CPU engine)
    std::vector<double> h_table;
    build_obliquez_boltzmann(h_table, nx_physical_, cell, ds, norm_factor_);
    upload_table(table, h_table, stream);
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::use_stream(cudaStream_t stream)
{
    // The plans already run on stream_ (set at construction or by the last call)
    if (stream == stream_)
        return;
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(plan_fwd_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_bwd_, stream_));
}

void CudaCrysFFTObliqueZ::diffusion(const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                                    cudaStream_t stream)
{
    use_stream(stream);

    const int threads = 256;
    const int n = nx_physical_[2];

    // The input is only read, so an out-of-place call preserves it
    ker_permute_z<<<grid_blocks(M_physical_, threads), threads, 0, stream_>>>(d_q_in, d_real_, M_physical_, n);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_, d_real_, d_spectrum_));
    // Boltzmann factors in the table include the normalization
    ker_obliquez_spectral<<<grid_blocks(M_spectral_, threads), threads, 0, stream_>>>(
        d_spectrum_, reinterpret_cast<const double2*>(table.get()),
        reinterpret_cast<const double2*>(d_twiddle_->get()), nx_physical_[0], nx_physical_[1], n / 2 + 1);
    CUDA_CHECK_LAUNCH();
    // Z2D overwrites its input d_spectrum_, which the next step recomputes
    CUFFT_CHECK(cufftExecZ2D(plan_bwd_, d_spectrum_, d_real_));
    ker_unpermute_z<<<grid_blocks(M_physical_, threads), threads, 0, stream_>>>(d_real_, d_q_out, M_physical_, n);
    CUDA_CHECK_LAUNCH();
}
