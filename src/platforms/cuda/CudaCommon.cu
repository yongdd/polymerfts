/**
 * @file CudaCommon.cu
 * @brief CUDA common utilities and kernel implementations.
 *
 * Provides GPU configuration management and a comprehensive library of
 * CUDA kernels for element-wise array operations used throughout the
 * SCFT/L-FTS computations.
 *
 * **Configuration:**
 *
 * - CudaCommon singleton manages GPU blocks/threads configuration
 * - Environment variables: LFTS_GPU_NUM_BLOCKS, LFTS_GPU_NUM_THREADS
 * - Default: 1024 blocks, 256 threads per block
 *
 * **Kernel Categories:**
 *
 * - Exponential: ker_exp (dst = a*exp(b*src))
 * - Multiplication: ker_multi (element-wise products)
 * - Division: ker_divide (element-wise division)
 * - Linear combination: ker_lin_comb
 * - Complex operations: ker_multi_complex_conjugate, ker_multi_complex_real
 *
 * **Type Conversion:**
 *
 * - stdToCuDoubleComplex: std::complex to CUDA conversion
 *
 * @see CudaCommon for singleton interface
 */

#include <iostream>
#include <cstdlib>
#include <string>

#include "CudaCommon.h"

void report_cuda_error(cudaError_t code, const char *file, int line, const char *func)
{
    // A failed API call (e.g. cudaMalloc) also leaves its error in the per-thread
    // slot; clear it so the next launch check does not report it again. Sticky
    // errors of a broken context stay, as they must.
    (void)cudaGetLastError();
    std::string file_and_line("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">");
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
}

void report_cuda_allocation_error(cudaError_t code, size_t bytes, const char *kind,
                                  const std::source_location& where, bool injected)
{
    if (!injected)
        (void)cudaGetLastError();   // as above (an injected failure made no CUDA call)
    std::string what = std::string(injected ? "injected failure (test hook) of the allocation"
                                            : "allocation")
        + " of " + std::to_string(bytes) + " bytes of " + kind + " memory; File: \""
        + where.file_name() + "\", line: " + std::to_string(where.line())
        + ", function <" + where.function_name() + ">";
    throw thrust::system_error(code, thrust::cuda_category(), what);
}

cuDoubleComplex stdToCuDoubleComplex(const std::complex<double>& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

namespace
{
    // The integer value of a launch environment variable, or default_value when it is unset
    int launch_variable(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return default_value;
        std::size_t end = 0;
        int result = 0;
        try
        {
            result = std::stoi(value, &end);
        }
        catch (const std::exception&)
        {
            end = 0;
        }
        if (end == 0 || value[end] != '\0')
            throw_with_line_number(std::string(name) + " must be an integer, got \"" + value + "\".");
        return result;
    }
}

CudaCommon::CudaCommon()
{
    // 1024 blocks of 256 threads: 3-6% faster than 256 blocks at 64^3 on an A10
    // (grid-stride kernels and CUB reductions, so the results do not depend on it)
    this->n_blocks  = launch_variable("LFTS_GPU_NUM_BLOCKS", 1024);
    this->n_threads = launch_variable("LFTS_GPU_NUM_THREADS", 256);
    // The grid-stride kernels rely on at most MAX_GRID_THREADS threads (see
    // GridLimits.h); 1024 threads per block and 2^20 blocks are within the
    // limits of every supported device
    if (n_blocks < 1 || n_threads < 1 || n_threads > 1024 ||
        static_cast<long long>(n_blocks) * n_threads > MAX_GRID_THREADS)
        throw_with_line_number("The CUDA launch of " + std::to_string(n_blocks) + " blocks x " +
            std::to_string(n_threads) + " threads is not supported: at most 1024 threads per block and " +
            std::to_string(MAX_GRID_THREADS) + " threads in total (LFTS_GPU_NUM_BLOCKS, LFTS_GPU_NUM_THREADS).");
}
int CudaCommon::get_n_blocks()
{
    return n_blocks;
}
int CudaCommon::get_n_threads()
{
    return n_threads;
}

__global__ void ker_exp(double* dst, const double* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * exp(exp_b * src[i]);
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_exp(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a * exp(exp_b * src[i].x) * cos(exp_b * src[i].y);
        temp_dst.y = a * exp(exp_b * src[i].x) * sin(exp_b * src[i].y);
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(double* dst, const double* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x * src2[i];
        dst[i].y = a * src1[i].y * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y);
        temp_dst.y = a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x);
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide(double* dst, const double* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] / src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        double denom = src2[i].x * src2[i].x + src2[i].y * src2[i].y;
        temp_dst.x = a * (src1[i].x * src2[i].x + src1[i].y * src2[i].y) / denom;
        temp_dst.y = a * (src1[i].y * src2[i].x - src1[i].x * src2[i].y) / denom;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide_masked(double* dst, const double* src1, const double* src2, double a, const double* mask, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = mask[i] == 0.0 ? 0.0 : a * src1[i] / src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide_masked(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2,
                                 double a, const double* mask, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        if (mask[i] == 0.0)
            dst[i] = make_cuDoubleComplex(0.0, 0.0);
        else
        {
            // Preserve ker_divide's arithmetic and its support for aliased inputs.
            cuDoubleComplex temp_dst;
            double denom = src2[i].x * src2[i].x + src2[i].y * src2[i].y;
            temp_dst.x = a * (src1[i].x * src2[i].x + src1[i].y * src2[i].y) / denom;
            temp_dst.y = a * (src1[i].y * src2[i].x - src1[i].x * src2[i].y) / denom;
            dst[i] = temp_dst;
        }
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x + b * src2[i].x;
        dst[i].y = a * src1[i].y + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a.x * src1[i].x - a.y * src1[i].y + b * src2[i].x;
        temp_dst.y = a.x * src1[i].y + a.y * src1[i].x + b * src2[i].y;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const double* src, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * dst[i].x * src[i];
        dst[i].y = a * dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_conjugate(double* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_conjugate(
    cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        temp_dst.y = src1[i].x * src2[i].y - src1[i].y * src2[i].x;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}
