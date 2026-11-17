/**
 * @file CudaCrysFFTRecursive3m.cu
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 */

#include "CudaCrysFFTRecursive3m.h"

#include "CudaCommon.h"
#include "Exception.h"
#include "Recursive3mFoldParity.h"
#include "Recursive3mHelpers.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <tuple>

// Host-side fold, twiddles and multipliers are shared with the CPU engine
using namespace crys_recursive_3m;

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
    : nx_logical_(nx_logical),
      translational_part_(translational_part)
{
    try
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0 || (nx_logical_[d] % 2) != 0)
            {
                throw_with_line_number("CudaCrysFFTRecursive3m requires even, positive grid dimensions.");
            }
        }
        // The kernels here are size-agnostic (ker_apply_k_3m iterates the packed
        // z range Nz2c = Nz2/2+1 with exact bounds; cuFFT D2Z/Z2D handles any
        // length). The nz/2 >= 2 bound mirrors the guard in
        // CrysFFTRecursive3mBase.h so CPU and CUDA stay constructible for exactly
        // the same grids.
        if ((nx_logical_[2] / 2) < 2)
        {
            throw_with_line_number("CudaCrysFFTRecursive3m requires Nz/2 >= 2.");
        }

        // The fold pairs twiddle r_p with the octant-transformed Boltzmann array
        // S_{psi(p)}, where psi depends on the parities of the m3 generator
        // translations in grid units (see Recursive3mFoldParity.h). When psi is
        // not a bijection the even-index physical basis does not exist on this
        // grid and the fold cannot represent the group; CrysFFTSelector rejects
        // such grids, but guard direct construction.
        if (!compute_m3_fold_permutation(translational_part_, nx_logical_, fold_perm_))
        {
            throw_with_line_number("CudaCrysFFTRecursive3m: m3 translations are "
                "incompatible with this grid (translation-parity permutation is "
                "not a bijection; no even-index physical basis exists).");
        }

        nx_physical_ = { nx_logical_[0] / 2, nx_logical_[1] / 2, nx_logical_[2] / 2 };
        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        M_complex_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);

        d_step1_.allocate(M_complex_);
        d_step2_.allocate(M_complex_);

        init_plans();
        twiddles_ = generate_twiddle_factors();
    }
    catch (...)
    {
        release();
        throw;
    }
}

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(const CudaCrysFFTRecursive3m& other)
    : nx_logical_(other.nx_logical_),
      nx_physical_(other.nx_physical_),
      M_logical_(other.M_logical_),
      M_physical_(other.M_physical_),
      M_complex_(other.M_complex_),
      translational_part_(other.translational_part_),
      fold_perm_(other.fold_perm_),
      twiddles_(other.twiddles_)
{
    try
    {
        d_step1_.allocate(M_complex_);
        d_step2_.allocate(M_complex_);
        init_plans();
    }
    catch (...)
    {
        release();
        throw;
    }
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTRecursive3m::clone() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTRecursive3m(*this));
}

CudaCrysFFTRecursive3m::~CudaCrysFFTRecursive3m()
{
    release();
}

// Idempotent; tolerates a partly built object (null buffers, plans not created).
void CudaCrysFFTRecursive3m::release()
{
    // The buffers are owning; the cuFFT plans need the explicit free
    twiddles_.reset();
    d_step1_.release();
    d_step2_.release();

    free_plans();
}

void CudaCrysFFTRecursive3m::init_plans()
{
    if (plans_initialized_)
        return;
    if (count_cufft_plan(cufftPlan3d(&plan_r2c_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_D2Z)) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT D2Z plan for CudaCrysFFTRecursive3m.");
    if (count_cufft_plan(cufftPlan3d(&plan_c2r_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_Z2D)) != CUFFT_SUCCESS)
    {
        // plans_initialized_ covers both plans; destroy the first one here
        release_cufft_plan(plan_r2c_);
        throw_with_line_number("Failed to create cuFFT Z2D plan for CudaCrysFFTRecursive3m.");
    }
    plans_initialized_ = true;
}

void CudaCrysFFTRecursive3m::free_plans()
{
    if (!plans_initialized_)
        return;
    release_cufft_plan(plan_r2c_);
    release_cufft_plan(plan_c2r_);
    plans_initialized_ = false;
}

std::shared_ptr<const CudaCrysFFTRecursive3m::Twiddles> CudaCrysFFTRecursive3m::generate_twiddle_factors() const
{
    auto twiddles = std::make_shared<Twiddles>();
    auto& r_re = twiddles->re;
    auto& r_im = twiddles->im;
    for (int i = 0; i < 8; ++i)
    {
        r_re[i].assign(M_physical_, 0.0);
        r_im[i].assign(M_physical_, 0.0);
    }

    auto operations = generate_m3_operations(translational_part_, nx_logical_);
    std::vector<std::tuple<double*, double*, std::array<double, 3>>> twiddle;
    for (size_t i = 0; i < operations.size(); ++i)
    {
        twiddle.emplace_back(r_re[i + 1].data(), r_im[i + 1].data(), operations[i]);
    }

    std::fill(r_re[0].begin(), r_re[0].end(), 1.0);
    std::fill(r_im[0].begin(), r_im[0].end(), 0.0);

    twiddle_factor(twiddle, nx_physical_);
    return twiddles;
}

void CudaCrysFFTRecursive3m::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                             double coeff, cudaStream_t stream) const
{
    const std::vector<double> tempmat = logical_k_table(nx_logical_, cell, [coeff](double kx2, double ky2, double kz2) {
        return std::exp(-(kx2 + ky2 + kz2) * coeff);
    });
    build_table_from_tempmat(table, tempmat, stream);
}

void CudaCrysFFTRecursive3m::build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                              int axes, double coeff, cudaStream_t stream) const
{
    if (axes < 1 || axes > 7)
        throw_with_line_number("CudaCrysFFTRecursive3m::build_multiplier: axes must be a mask of 1 (x), 2 (y), 4 (z).");
    const bool with_bond = (coeff != 0.0);   // continuous chains: Σ k_d² alone
    const std::vector<double> tempmat = logical_k_table(nx_logical_, cell, [axes, with_bond, coeff](double kx2, double ky2, double kz2) {
        const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
        return with_bond ? std::exp(-(kx2 + ky2 + kz2) * coeff) * k_axes : k_axes;
    });
    build_table_from_tempmat(table, tempmat, stream);
}

void CudaCrysFFTRecursive3m::build_table_from_tempmat(
    CudaDeviceBuffer<double>& table, const std::vector<double>& tempmat, cudaStream_t stream) const
{
    std::array<std::vector<double>, 8> k_split;
    std::array<double*, 8> k_split_ptr{};
    for (int i = 0; i < 8; ++i)
    {
        k_split[i].assign(M_physical_, 0.0);
        k_split_ptr[i] = k_split[i].data();
    }

    mat_split(tempmat.data(), k_split_ptr, nx_logical_);

    std::array<int, 3> halfsize = nx_physical_;
    constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
    add_sub_by_seq(k_split_ptr, seq0, halfsize);
    add_sub_by_seq(k_split_ptr, seq1, halfsize);
    add_sub_by_seq(k_split_ptr, seq2, halfsize);

    // The table on the host: the eight real factor arrays, then the eight imaginary ones
    std::vector<double> k_host(16 * static_cast<size_t>(M_physical_), 0.0);
    std::array<double*, 8> k_re_ptr{};
    std::array<double*, 8> k_im_ptr{};
    std::array<const double*, 8> r_re_ptr{};
    std::array<const double*, 8> r_im_ptr{};
    std::array<const double*, 8> k_split_perm_ptr{};

    for (int i = 0; i < 8; ++i)
    {
        k_re_ptr[i] = k_host.data() + static_cast<size_t>(i) * M_physical_;
        k_im_ptr[i] = k_host.data() + static_cast<size_t>(8 + i) * M_physical_;
        r_re_ptr[i] = twiddles_->re[i].data();
        r_im_ptr[i] = twiddles_->im[i].data();
        // Twiddle r_p pairs with S_{fold_perm_[p]} (translation-parity
        // permutation; identity when all generator translations are even in
        // grid units).
        k_split_perm_ptr[i] = k_split[fold_perm_[i]].data();
    }

    constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
    mul_add_sub_by_seq(k_re_ptr, k_im_ptr, r_re_ptr, r_im_ptr, k_split_perm_ptr, seq3, halfsize);

    upload_table(table, k_host, stream);
}

void CudaCrysFFTRecursive3m::diffusion(
    double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table, cudaStream_t stream)
{
    use_stream(stream);
    const size_t bytes = sizeof(double) * M_physical_;

    // The out-of-place D2Z keeps d_q_in, and the Z2D reads only d_step2_, so an
    // out-of-place call preserves its input without a copy; d_q_out may alias d_q_in.
    // cuFFT needs the arrays it reads and writes aligned to a cufftDoubleComplex. An input
    // at an odd offset into a larger array (the second propagator of a stress pair of an
    // odd basis size) is not: it is copied to d_step2_, free until the kernel below writes
    // it, and transformed there. The outputs (work buffers and propagators) are aligned.
    if (!cufft_aligned(d_q_out))
        throw_with_line_number("CudaCrysFFTRecursive3m: the output must be aligned to a cufftDoubleComplex.");
    double* d_src = d_q_in;
    if (!cufft_aligned(d_q_in))
    {
        d_src = reinterpret_cast<double*>(d_step2_.get());
        CUDA_CHECK(cudaMemcpyAsync(d_src, d_q_in, bytes, cudaMemcpyDeviceToDevice, stream_));
    }
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_, d_src, d_step1_));

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
