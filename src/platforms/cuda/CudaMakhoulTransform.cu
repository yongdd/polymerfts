/**
 * @file CudaMakhoulTransform.cu
 * @brief DFT along periodic axes and cosine/sine series along walls through one cuFFT of the grid.
 *
 * The kernels index the grid with int: a grid has at most MAX_GRID_POINTS points (GridLimits.h).
 *
 * @see CudaMakhoulTransform.h for the algorithm
 */

#include <algorithm>
#include <array>
#include <complex>
#include <numbers>
#include <string>
#include <cuComplex.h>

#include "CudaMakhoulTransform.h"
#include "GridLimits.h"

namespace
{
/// Sample j at place p of a reordered wall axis of n points: even samples forward, odd samples backward
__device__ inline int makhoul_sample(int p, int n)
{
    return (p < (n + 1) / 2) ? 2 * p : 2 * (n - 1 - p) + 1;
}

/// Place of sample j on a reordered wall axis of n points (inverse of makhoul_sample)
__device__ inline int makhoul_place(int j, int n)
{
    return (j % 2 == 0) ? j / 2 : n - 1 - j / 2;
}

__device__ inline double scaled(double x, double s) { return s * x; }
__device__ inline cuDoubleComplex scaled(cuDoubleComplex x, double s) { return make_cuDoubleComplex(s * x.x, s * x.y); }

/// Grid index (i0, i1, i2) of a flat index on a grid of shape n
__device__ inline void unflatten(int idx, const int n[3], int i[3])
{
    i[2] = idx % n[2];
    i[1] = (idx / n[2]) % n[1];
    i[0] = idx / (n[2] * n[1]);
}

/// samples(p) = scale * sign * field(sample of p): the reordering of every wall axis
template <typename D>
__global__ void ker_makhoul_reorder(const D* __restrict__ field, D* __restrict__ samples, CudaMakhoulAxes a,
                                    double scale, int n_grid)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_grid; idx += blockDim.x * gridDim.x)
    {
        int p[3];
        unflatten(idx, a.n, p);
        double factor = scale;
        int j[3];
        #pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            j[d] = a.wall[d] ? makhoul_sample(p[d], a.n[d]) : p[d];
            if (a.sine[d] && j[d] % 2 == 1)
                factor = -factor;
        }
        samples[idx] = scaled(field[(j[0] * a.n[1] + j[1]) * a.n[2] + j[2]], factor);
    }
}

/// field(j) = scale * sign * samples(place of j): the inverse of ker_makhoul_reorder (up to scale)
template <typename D>
__global__ void ker_makhoul_restore(const D* __restrict__ samples, D* __restrict__ field, CudaMakhoulAxes a,
                                    double scale, int n_grid)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_grid; idx += blockDim.x * gridDim.x)
    {
        int j[3];
        unflatten(idx, a.n, j);
        double factor = scale;
        int p[3];
        #pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            p[d] = a.wall[d] ? makhoul_place(j[d], a.n[d]) : j[d];
            if (a.sine[d] && j[d] % 2 == 1)
                factor = -factor;
        }
        field[idx] = scaled(samples[(p[0] * a.n[1] + p[1]) * a.n[2] + p[2]], factor);
    }
}

/// Mode V(i) of the cuFFT of the samples. D2Z holds the half spectrum of the last axis;
/// the other half of the FFT of real samples is V(-i) = conj V(i).
template <bool REAL>
__device__ inline cuDoubleComplex makhoul_mode(const cuDoubleComplex* modes, const CudaMakhoulAxes& a,
                                               int i0, int i1, int i2)
{
    if (!REAL)
        return modes[(i0 * a.n[1] + i1) * a.n[2] + i2];
    const int half = a.n[2] / 2 + 1;
    if (i2 < half)
        return modes[(i0 * a.n[1] + i1) * half + i2];
    const int m0 = (a.n[0] - i0) % a.n[0];
    const int m1 = (a.n[1] - i1) % a.n[1];
    return cuConj(modes[(m0 * a.n[1] + m1) * half + (a.n[2] - i2)]);
}

/// Coefficient C(c), or for a real field the Hermitian part (C(c) + conj C(-c))/2, -c reversing
/// only the periodic indices: the coefficients of the real part of the field
template <bool REAL>
__device__ inline cuDoubleComplex makhoul_coefficient(const cuDoubleComplex* coefficients, const CudaMakhoulAxes& a,
                                                      int c0, int c1, int c2)
{
    const cuDoubleComplex x = coefficients[(c0 * a.n[1] + c1) * a.n[2] + c2];
    if (!REAL)
        return x;
    const int m0 = a.wall[0] ? c0 : (a.n[0] - c0) % a.n[0];
    const int m1 = a.wall[1] ? c1 : (a.n[1] - c1) % a.n[1];
    const int m2 = a.wall[2] ? c2 : (a.n[2] - c2) % a.n[2];
    const cuDoubleComplex y = coefficients[(m0 * a.n[1] + m1) * a.n[2] + m2];
    return make_cuDoubleComplex(0.5 * (x.x + y.x), 0.5 * (x.y - y.y));
}

/// Forward: every coefficient from its modes. Along a wall axis the coefficient of mode k
/// (stored at k, or at n-1-k on a sine axis) is W^k V_k + W^-k V_{-k}; a periodic axis keeps V_k.
/// The loops over the two terms of each axis unroll, so the terms stay in registers.
template <bool REAL>
__global__ void ker_makhoul_combine(const cuDoubleComplex* __restrict__ modes, cuDoubleComplex* __restrict__ coefficients,
                                    CudaMakhoulAxes a, int n_grid)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_grid; idx += blockDim.x * gridDim.x)
    {
        int c[3];
        unflatten(idx, a.n, c);
        int index[3][2];
        cuDoubleComplex factor[3][2];
        #pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            const int n = a.n[d];
            const int k = a.sine[d] ? n - 1 - c[d] : c[d];
            index[d][0] = k;
            index[d][1] = (n - k) % n;
            factor[d][0] = a.wall[d] ? a.w[d][k] : make_cuDoubleComplex(1.0, 0.0);
            factor[d][1] = cuConj(factor[d][0]);
        }
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        #pragma unroll
        for (int t0 = 0; t0 < 2; ++t0)
            #pragma unroll
            for (int t1 = 0; t1 < 2; ++t1)
                #pragma unroll
                for (int t2 = 0; t2 < 2; ++t2)
                {
                    // A periodic axis has one term
                    if ((t0 == 1 && !a.wall[0]) || (t1 == 1 && !a.wall[1]) || (t2 == 1 && !a.wall[2]))
                        continue;
                    cuDoubleComplex v = makhoul_mode<REAL>(modes, a, index[0][t0], index[1][t1], index[2][t2]);
                    if (a.wall[0])
                        v = cuCmul(factor[0][t0], v);
                    if (a.wall[1])
                        v = cuCmul(factor[1][t1], v);
                    if (a.wall[2])
                        v = cuCmul(factor[2][t2], v);
                    sum = cuCadd(sum, v);
                }
        coefficients[idx] = sum;
    }
}

/// Backward: every mode from its coefficients (a real field: the half spectrum of the last axis,
/// from the Hermitian part). Along a wall axis V_k = W^-k (X_k - i X_{n-k}), X_n = 0, twice the
/// inverse of the forward step; a periodic axis keeps X_k.
template <bool REAL>
__global__ void ker_makhoul_separate(const cuDoubleComplex* __restrict__ coefficients, cuDoubleComplex* __restrict__ modes,
                                     CudaMakhoulAxes a, int n_modes)
{
    const int shape[3] = {a.n[0], a.n[1], REAL ? a.n[2] / 2 + 1 : a.n[2]};
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_modes; idx += blockDim.x * gridDim.x)
    {
        int k[3];
        unflatten(idx, shape, k);
        int index[3][2];
        cuDoubleComplex factor[3][2];
        bool two_terms[3];
        #pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            // The coefficient of mode k lies at k, or at n-1-k on a sine axis
            const int n = a.n[d];
            index[d][0] = a.sine[d] ? n - 1 - k[d] : k[d];
            index[d][1] = a.sine[d] ? k[d] - 1 : n - k[d];
            const cuDoubleComplex w_inverse = a.wall[d] ? cuConj(a.w[d][k[d]]) : make_cuDoubleComplex(1.0, 0.0);
            factor[d][0] = w_inverse;
            factor[d][1] = make_cuDoubleComplex(w_inverse.y, -w_inverse.x);   // -i W^-k
            two_terms[d] = a.wall[d] && k[d] != 0;
        }
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        #pragma unroll
        for (int t0 = 0; t0 < 2; ++t0)
            #pragma unroll
            for (int t1 = 0; t1 < 2; ++t1)
                #pragma unroll
                for (int t2 = 0; t2 < 2; ++t2)
                {
                    if ((t0 == 1 && !two_terms[0]) || (t1 == 1 && !two_terms[1]) || (t2 == 1 && !two_terms[2]))
                        continue;
                    cuDoubleComplex v = makhoul_coefficient<REAL>(coefficients, a, index[0][t0], index[1][t1], index[2][t2]);
                    if (a.wall[0])
                        v = cuCmul(factor[0][t0], v);
                    if (a.wall[1])
                        v = cuCmul(factor[1][t1], v);
                    if (a.wall[2])
                        v = cuCmul(factor[2][t2], v);
                    sum = cuCadd(sum, v);
                }
        modes[idx] = sum;
    }
}
}  // namespace

template <typename T>
CudaMakhoulTransform<T>::CudaMakhoulTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc)
{
    const int rank = static_cast<int>(nx.size());
    if (rank < 1 || rank > 3 || bc.size() != nx.size())
        throw_with_line_number("CudaMakhoulTransform takes one to three axes, each periodic, cosine or sine.");
    if (std::none_of(bc.begin(), bc.end(), [](BoundaryCondition b) { return b == BoundaryCondition::PERIODIC; }))
        throw_with_line_number("CudaMakhoulTransform needs a periodic axis: walls alone have real coefficients.");
    n_grid_ = checked_grid_points(nx);   // the kernels index the grid with int

    // Axes padded to three in front; W^k on the wall axes, one table after the other
    std::vector<std::complex<double>> w;
    std::array<size_t, 3> w_offset{};
    for (int d = 0; d < 3; ++d)
    {
        const int a = d - (3 - rank);
        const int n = (a < 0) ? 1 : nx[a];
        axes_.n[d] = n;
        axes_.wall[d] = a >= 0 && bc[a] != BoundaryCondition::PERIODIC;
        axes_.sine[d] = a >= 0 && bc[a] == BoundaryCondition::ABSORBING;
        w_offset[d] = w.size();
        if (axes_.wall[d])
            for (int k = 0; k < n; ++k)
                w.push_back(std::polar(1.0, -k * std::numbers::pi / (2.0 * n)));
    }
    n_modes_ = REAL ? n_grid_ / axes_.n[2] * (axes_.n[2] / 2 + 1) : n_grid_;

    d_w_.allocate(w.size());
    CUDA_CHECK(cudaMemcpy(d_w_, w.data(), sizeof(cuDoubleComplex) * w.size(), cudaMemcpyHostToDevice));
    for (int d = 0; d < 3; ++d)
        axes_.w[d] = axes_.wall[d] ? d_w_.get() + w_offset[d] : nullptr;

    d_modes_.allocate(n_modes_);
    if constexpr (REAL)
        d_samples_.allocate(n_grid_);

    // Unnormalized periodic FFT of the grid as given (row-major, rank axes). The forward
    // and inverse plans of a real field never run at the same time: one work area serves both.
    std::vector<int> n(nx.begin(), nx.end());
    size_t work_bytes = 0;
    auto make_plan = [&](CufftPlan& plan, cufftType type) {
        CUFFT_CHECK(plan.create_without_work_area(work_bytes, cufftMakePlanMany, rank, n.data(),
                                                  nullptr, 1, 0, nullptr, 1, 0, type, 1));
    };
    make_plan(plan_, REAL ? CUFFT_D2Z : CUFFT_Z2Z);
    if constexpr (REAL)
        make_plan(plan_inverse_, CUFFT_Z2D);
    if (work_bytes > 0)
    {
        d_work_.allocate((work_bytes + sizeof(double) - 1) / sizeof(double));
        CUFFT_CHECK(cufftSetWorkArea(plan_, d_work_.get()));
        if constexpr (REAL)
            CUFFT_CHECK(cufftSetWorkArea(plan_inverse_, d_work_.get()));
    }
}

template <typename T>
void CudaMakhoulTransform<T>::forward(const DeviceT* d_in, cuDoubleComplex* d_out, double scale, cudaStream_t stream)
{
    const int n_threads = CudaCommon::get_instance().get_n_threads();
    const int n_blocks = grid_blocks(n_grid_, n_threads);
    CUFFT_CHECK(cufftSetStream(plan_, stream));
    if constexpr (REAL)
    {
        ker_makhoul_reorder<<<n_blocks, n_threads, 0, stream>>>(d_in, d_samples_.get(), axes_, scale, n_grid_);
        CUDA_CHECK_LAUNCH();
        CUFFT_CHECK(cufftExecD2Z(plan_, d_samples_.get(), d_modes_.get()));
    }
    else
    {
        // The samples of a complex field are transformed in place
        ker_makhoul_reorder<<<n_blocks, n_threads, 0, stream>>>(d_in, d_modes_.get(), axes_, scale, n_grid_);
        CUDA_CHECK_LAUNCH();
        CUFFT_CHECK(cufftExecZ2Z(plan_, d_modes_.get(), d_modes_.get(), CUFFT_FORWARD));
    }
    ker_makhoul_combine<REAL><<<n_blocks, n_threads, 0, stream>>>(d_modes_.get(), d_out, axes_, n_grid_);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaMakhoulTransform<T>::backward(const cuDoubleComplex* d_in, DeviceT* d_out, double scale, cudaStream_t stream)
{
    const int n_threads = CudaCommon::get_instance().get_n_threads();
    ker_makhoul_separate<REAL><<<grid_blocks(n_modes_, n_threads), n_threads, 0, stream>>>(
        d_in, d_modes_.get(), axes_, n_modes_);
    CUDA_CHECK_LAUNCH();
    const int n_blocks = grid_blocks(n_grid_, n_threads);
    if constexpr (REAL)
    {
        // Z2D overwrites the modes, which are not needed again
        CUFFT_CHECK(cufftSetStream(plan_inverse_, stream));
        CUFFT_CHECK(cufftExecZ2D(plan_inverse_, d_modes_.get(), d_samples_.get()));
        ker_makhoul_restore<<<n_blocks, n_threads, 0, stream>>>(d_samples_.get(), d_out, axes_, scale, n_grid_);
    }
    else
    {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
        CUFFT_CHECK(cufftExecZ2Z(plan_, d_modes_.get(), d_modes_.get(), CUFFT_INVERSE));
        ker_makhoul_restore<<<n_blocks, n_threads, 0, stream>>>(d_modes_.get(), d_out, axes_, scale, n_grid_);
    }
    CUDA_CHECK_LAUNCH();
}

template class CudaMakhoulTransform<double>;
template class CudaMakhoulTransform<std::complex<double>>;
