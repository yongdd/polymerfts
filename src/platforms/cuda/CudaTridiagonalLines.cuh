#ifndef POLYMERFTS_CUDA_TRIDIAGONAL_LINES_CUH_
#define POLYMERFTS_CUDA_TRIDIAGONAL_LINES_CUH_

// Kernels that solve tridiagonal lines with the factors of factorize_tridiagonal()
// (CudaSolverCNADI.h). They are templates defined here, so that every translation unit
// that launches them instantiates its own: since CUDA 13 the host stub of a __global__
// template has internal linkage, and an instantiation in another file cannot be launched.

// One line of M values with the given stride: the sweeps over the factors (see factorize_tridiagonal())
template <bool PERIODIC>
__device__ __forceinline__ void solve_line(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* d, double* x, const int stride)
{
    const double* lower = table;
    const double* inv = table + M;
    const double* c = table + 2 * M;
    double y = d[0] * __ldg(&inv[0]);
    x[0] = y;
    for (int i = 1; i < M; i++)
    {
        y = (d[i * stride] - __ldg(&lower[i]) * y) * __ldg(&inv[i]);
        x[i * stride] = y;
    }
    for (int i = M - 2; i >= 0; i--)
    {
        y = x[i * stride] - __ldg(&c[i]) * y;
        x[i * stride] = y;
    }
    if constexpr (PERIODIC)
    {
        const double* q = table + 3 * M;
        const double value = (x[0] + v * x[(M - 1) * stride]) / denom;
        for (int i = 0; i < M; i++)
            x[i * stride] -= __ldg(&q[i]) * value;
    }
}

/**
 * @brief CUDA kernel: the REPEAT lines of M points of a grid axis with stride INTERVAL, one per thread.
 *
 * In a row-major grid the points of a line along an axis are INTERVAL apart (the product of the
 * sizes of the axes after it); line n starts at (n / INTERVAL) * INTERVAL * M + n % INTERVAL, the
 * outer index n / INTERVAL running over the axes before it and the inner one n % INTERVAL over the
 * axes after it. The start is computed as n + (n / INTERVAL) * INTERVAL * (M - 1), with one
 * division: the form with a remainder made the wall lines 17 % slower (A10, 64^3). Solves d_x
 * from the right-hand side d_d with the factors in table. d_d and d_x are __restrict__: they must
 * not overlap (tridiagonal_lines_shared() solves in place, in shared memory).
 */
template <bool PERIODIC>
__global__ void tridiagonal_lines(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x, const int REPEAT, const int INTERVAL)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < REPEAT; n += blockDim.x * gridDim.x)
    {
        const int offset = n + (n / INTERVAL) * (INTERVAL * (M - 1));
        solve_line<PERIODIC>(table, M, v, denom, d_d + offset, d_x + offset, INTERVAL);
    }
}

/**
 * @brief CUDA kernel: REPEAT contiguous systems of size M (system k at k*M), staged in shared memory.
 *
 * A block of blockDim.x threads copies as many systems into shared memory with coalesced reads,
 * each thread solves one there, and the block writes them back. Rows hold an odd number of values
 * so that the threads of a warp use different banks: blockDim.x * (M or M+1) doubles of dynamic
 * shared memory.
 */
template <bool PERIODIC>
__global__ void tridiagonal_lines_shared(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x, const int REPEAT)
{
    extern __shared__ double s_lines[];
    const int rows = blockDim.x;
    const int pitch = (M % 2 == 1) ? M : M + 1;
    for (long first = static_cast<long>(blockIdx.x) * rows; first < REPEAT; first += static_cast<long>(gridDim.x) * rows)
    {
        const int count = static_cast<int>(min(static_cast<long>(rows), REPEAT - first));
        // Neighbouring threads read neighbouring values of the block's lines
        for (int k = threadIdx.x; k < count * M; k += rows)
            s_lines[(k / M) * pitch + k % M] = d_d[first * M + k];
        __syncthreads();
        if (threadIdx.x < count)
            solve_line<PERIODIC>(table, M, v, denom, s_lines + threadIdx.x * pitch, s_lines + threadIdx.x * pitch, 1);
        __syncthreads();
        for (int k = threadIdx.x; k < count * M; k += rows)
            d_x[first * M + k] = s_lines[(k / M) * pitch + k % M];
        __syncthreads();
    }
}

#endif
