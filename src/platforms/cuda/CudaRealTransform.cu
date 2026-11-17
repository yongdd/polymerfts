/**
 * @file CudaRealTransform.cu
 * @brief Unified CUDA Real Transform (DCT/DST) implementation.
 *
 * DCT and DST types 2 and 3 for 1D, 2D, and 3D arrays.
 */

#include "CudaRealTransform.h"
#include <initializer_list>
#include "CudaCommon.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// DCT lookup table helpers (optional sin/cos precompute)
//==============================================================================

static void allocate_dct2_lut(int N, CudaDeviceBuffer<double>& d_sin, CudaDeviceBuffer<double>& d_cos)
{
    if (N <= 0)
        return;

    std::vector<double> h_sin(N, 0.0);
    std::vector<double> h_cos(N, 0.0);
    for (int i = 0; i < N; ++i)
    {
        double angle = i * M_PI / (2.0 * static_cast<double>(N));
        h_sin[i] = std::sin(angle);
        h_cos[i] = std::cos(angle);
    }

    d_sin.allocate(N);
    d_cos.allocate(N);
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
}

static void allocate_dct3_lut(int N, CudaDeviceBuffer<double>& d_sin, CudaDeviceBuffer<double>& d_cos)
{
    if (N <= 0)
        return;

    int half = N / 2;
    std::vector<double> h_sin(half + 1, 0.0);
    std::vector<double> h_cos(half + 1, 0.0);
    for (int i = 1; i <= half; ++i)
    {
        double angle = i * M_PI / (2.0 * static_cast<double>(N));
        h_sin[i] = std::sin(angle);
        h_cos[i] = std::cos(angle);
    }

    d_sin.allocate(half + 1);
    d_cos.allocate(half + 1);
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), sizeof(double) * (half + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), sizeof(double) * (half + 1), cudaMemcpyHostToDevice));
}

//==============================================================================
// Helper functions
//==============================================================================

/// Number of doubles that hold `bytes` bytes (work buffers sized in bytes)
static size_t doubles_for_bytes(size_t bytes)
{
    return (bytes + sizeof(double) - 1) / sizeof(double);
}

static int pow2roundup(int x)
{
    if (x < 0) return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

static int getFFTSize(CudaTransformType type, int N)
{
    switch (type) {
        case CUDA_DCT_2:
        case CUDA_DCT_3:
            return N;
        case CUDA_DST_2:
        case CUDA_DST_3:
            return N + 2;  // FCT/FST: FFT size N, buffer N+2
        default:
            return N;
    }
}

// Round-trip scale of one axis of N points: DCT-3(DCT-2(x)) = DST-3(DST-2(x)) = 2N x
static double getNormFactor(CudaTransformType, int N)
{
    return 2.0 * N;
}

//==============================================================================
// 1D DCT Kernels
//==============================================================================

// Packs the N real samples into the N/2+1 complex slots of the half-length FFT.
// Out of place: the packed spectrum occupies up to N+2 doubles, more than the
// caller's N-double buffer holds.
__global__ void kernel_dct2_preOp(const double* __restrict__ in, double* __restrict__ out, int N)
{
    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        if (i == 0) {
            out[0] = in[0];
            out[1] = 0;
        } else if (i == N / 2 && N % 2 == 0) {
            // Nyquist slot of an even-length sequence. For odd N the slot N/2 is
            // an ordinary pair (2k = N-1) and takes the general branch below.
            out[N] = in[N - 1];
            out[N + 1] = 0;
        } else {
            out[i * 2] = (in[i * 2] + in[i * 2 - 1]) / 2;
            out[i * 2 + 1] = -((in[i * 2 - 1] - in[i * 2]) / 2);
        }
    }
}

__global__ void kernel_dct2_postOp(double* data, int N,
                                   const double* __restrict__ sin_lut,
                                   const double* __restrict__ cos_lut)
{
    extern __shared__ double sh_in[];

    // Stage every sample (N may be odd, so the input is not split into halves)
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sh_in[i] = data[i];
    __syncthreads();

    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        if (i == 0) {
            data[0] = sh_in[0] * 2;
        } else {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2 * N), &sina, &cosa);
            }
            double Ta = sh_in[i] + sh_in[N - i];
            double Tb = sh_in[i] - sh_in[N - i];

            data[i] = (Ta * cosa + Tb * sina);
            data[N - i] = (Ta * sina - Tb * cosa);
        }
    }
}

__global__ void kernel_dct3_preOp(double* data, int N,
                                  const double* __restrict__ sin_lut,
                                  const double* __restrict__ cos_lut)
{
    // Thread i owns the pair (i+1, N-i-1)
    for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
        double sina, cosa;
        int k = i + 1;
        if (sin_lut && cos_lut) {
            sina = sin_lut[k];
            cosa = cos_lut[k];
        } else {
            sincos(k * M_PI / (2 * N), &sina, &cosa);
        }
        double Ta = data[i + 1] + data[N - i - 1];
        double Tb = data[i + 1] - data[N - i - 1];

        data[i + 1] = Ta * sina + Tb * cosa;
        data[N - i - 1] = Ta * cosa - Tb * sina;
    }
}

__global__ void kernel_dct3_postOp(double* data, int N)
{
    extern __shared__ double sh_in[];

    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        sh_in[i] = data[i];
        sh_in[i + N / 2 + 1] = data[i + N / 2 + 1];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        if (i == 0) {
            data[0] = sh_in[0];
        } else {
            data[2 * i - 1] = (sh_in[i * 2] - sh_in[i * 2 + 1]);
            if (i * 2 < N) {
                data[2 * i] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
            }
        }
    }
}

//==============================================================================
// 1D DST Kernels (FCT/FST style, Makhoul 1980)
//==============================================================================

// DST-2: Uses Z2D FFT (FST algorithm)
// For FFTW N points: input at buffer[1..N], output at buffer[1..N]
__global__ void kernel_dst2_preOp(double* pin, int N)
{
    extern __shared__ double sh_in[];

    // Load from pin[i+1] (FST reads from offset 1)
    for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
        sh_in[i] = pin[i + 1];
        sh_in[i + N / 2] = pin[i + N / 2 + 1];
    }
    __syncthreads();

    // Create complex for Z2D FFT
    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        if (i == 0) {
            pin[0] = sh_in[0];
            pin[1] = 0.0;
        } else if (i == N / 2) {
            pin[N] = -sh_in[N - 1];
            pin[N + 1] = 0.0;
        } else {
            pin[i * 2] = (sh_in[i * 2] - sh_in[i * 2 - 1]) / 2.0;
            pin[i * 2 + 1] = -((sh_in[i * 2] + sh_in[i * 2 - 1]) / 2.0);
        }
    }
}

__global__ void kernel_dst2_postOp(double* pin, int N,
                                   const double* __restrict__ sin_lut,
                                   const double* __restrict__ cos_lut)
{
    extern __shared__ double sh_in[];

    // Load FFT output
    for (int i = threadIdx.x; i < N / 2; i += blockDim.x) {
        sh_in[i] = pin[i];
        sh_in[i + N / 2] = pin[i + N / 2];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        if (i != 0) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * N), &sina, &cosa);
            }
            double Ta = sh_in[i] + sh_in[N - i];
            double Tb = sh_in[i] - sh_in[N - i];
            pin[i] = (Ta * sina + Tb * cosa);
            pin[N - i] = (Ta * cosa - Tb * sina);
        } else {
            pin[0] = 0.0;
            pin[N] = sh_in[0] * 2.0;
        }
    }
}

// DST-3: Uses D2Z FFT (FST algorithm)
// For FFTW N points: input at buffer[0..N-1], output at buffer[1..N]
__global__ void kernel_dst3_preOp(double* pin, int N,
                                  const double* __restrict__ sin_lut,
                                  const double* __restrict__ cos_lut)
{
    // Thread i owns the pair (i, N-i), so no barrier is needed between its reads and writes
    for (int i = threadIdx.x; i < N / 2 + 1; i += blockDim.x) {
        double sina, cosa;
        if (sin_lut && cos_lut) {
            sina = sin_lut[i];
            cosa = cos_lut[i];
        } else {
            sincos(i * M_PI / (2.0 * N), &sina, &cosa);
        }
        double Ta = pin[i] + pin[N - i];
        double Tb = pin[i] - pin[N - i];
        pin[i] = Ta * cosa + Tb * sina;
        pin[N - i] = Ta * sina - Tb * cosa;
    }
}

__global__ void kernel_dst3_postOp(double* pin, int N)
{
    extern __shared__ double sh_in[];
    const int L = N;

    // Load FFT output (complex: L/2+1 complex values)
    for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
        sh_in[i] = pin[i];
        sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
    }
    __syncthreads();

    // Negate imaginary parts (FST)
    for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
        if (i != 0)
            sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
    }
    __syncthreads();

    // Output - no /2 factor, to match the FFTW unnormalized convention
    for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
        if (i == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
            if (i * 2 + 1 < L + 1) {
                pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
            }
        }
    }
}

//==============================================================================
// 2D DCT Kernels
//==============================================================================

// DCT-2 2D kernels
__global__ void kernel_dct2_2d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * (Ny / 2 + 1);

    while (idx < total) {
        int iy_half = idx % (Ny / 2 + 1);
        int ix = idx / (Ny / 2 + 1);

        int out_idx = ix * (Ny / 2 + 1) + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[ix * Ny];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2 && Ny % 2 == 0) {  // even N only; odd N: ordinary pair below
            out[out_idx].x = in[ix * Ny + Ny - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ix * Ny + 2 * iy_half;
            out[out_idx].x = (in[in_idx] + in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx - 1] - in[in_idx]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_2d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        if (iy == 0) {
            out[idx] = in[ix * Ny] * 2.0;
        } else {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[iy];
                cosa = cos_lut[iy];
            } else {
                sincos(iy * M_PI / (2.0 * Ny), &sina, &cosa);
            }
            double Ta = in[ix * Ny + iy] + in[ix * Ny + (Ny - iy)];
            double Tb = in[ix * Ny + iy] - in[ix * Ny + (Ny - iy)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_2d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2 + 1);

    while (idx < total) {
        int ix_half = idx % (Nx / 2 + 1);
        int iy = idx / (Nx / 2 + 1);

        int out_idx = iy * (Nx / 2 + 1) + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[iy];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2 && Nx % 2 == 0) {  // even N only; odd N: ordinary pair below
            out[out_idx].x = in[(Nx - 1) * Ny + iy];
            out[out_idx].y = 0.0;
        } else {
            int ix_even = 2 * ix_half;
            int ix_odd = ix_even - 1;
            out[out_idx].x = (in[ix_even * Ny + iy] + in[ix_odd * Ny + iy]) / 2.0;
            out[out_idx].y = -((in[ix_odd * Ny + iy] - in[ix_even * Ny + iy]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_2d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        if (ix == 0) {
            out[idx] = in[iy * Nx] * 2.0;
        } else {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[ix];
                cosa = cos_lut[ix];
            } else {
                sincos(ix * M_PI / (2.0 * Nx), &sina, &cosa);
            }
            double Ta = in[iy * Nx + ix] + in[iy * Nx + (Nx - ix)];
            double Tb = in[iy * Nx + ix] - in[iy * Nx + (Nx - ix)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

// DCT-3 2D kernels
__global__ void kernel_dct3_2d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * (Ny / 2);

    while (idx < total) {
        int iy = idx % (Ny / 2);
        int ix = idx / (Ny / 2);

        int base = ix * Ny;
        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;

        double sina, cosa;
        if (sin_lut && cos_lut) {
            sina = sin_lut[iy1];
            cosa = cos_lut[iy1];
        } else {
            sincos(iy1 * M_PI / (2.0 * Ny), &sina, &cosa);
        }
        double Ta = data[base + iy1] + data[base + iy2];
        double Tb = data[base + iy1] - data[base + iy2];

        data[base + iy1] = Ta * sina + Tb * cosa;
        data[base + iy2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_2d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_base = ix * (Ny / 2 + 1);

        if (iy == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iy + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iy % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_2d_preOp_x_strided(
    double* __restrict__ data,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2);

    while (idx < total) {
        int ix = idx % (Nx / 2);
        int iy = idx / (Nx / 2);

        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        int idx1 = ix1 * Ny + iy;
        int idx2 = ix2 * Ny + iy;

        double sina, cosa;
        if (sin_lut && cos_lut) {
            sina = sin_lut[ix1];
            cosa = cos_lut[ix1];
        } else {
            sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        }
        double Ta = data[idx1] + data[idx2];
        double Tb = data[idx1] - data[idx2];

        data[idx1] = Ta * sina + Tb * cosa;
        data[idx2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_2d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_base = iy * (Nx / 2 + 1);

        if (ix == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (ix + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (ix % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

//==============================================================================
// 2D DST Kernels
//==============================================================================

// DST-2 2D kernels (FST algorithm, Martucci 1994)
// DST-2 uses Z2D FFT with preOp/postOp
__global__ void kernel_dst2_2d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * (Ny / 2 + 1);

    while (idx < total) {
        int iy_half = idx % (Ny / 2 + 1);
        int ix = idx / (Ny / 2 + 1);

        int out_idx = ix * (Ny / 2 + 1) + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[ix * Ny];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2) {
            out[out_idx].x = -in[ix * Ny + Ny - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ix * Ny + 2 * iy_half;
            // DST-2: z[k] = ((x[2k] - x[2k-1])/2, -((x[2k] + x[2k-1])/2))
            out[out_idx].x = (in[in_idx] - in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx] + in[in_idx - 1]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_2d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        // FST index k = iy + 1 (mapping FFTW index iy to FST index k)
        // FST output[1..N] maps to FFTW output[0..N-1]
        int k = iy + 1;

        if (k == Ny) {
            // Last element: FFTW[Ny-1] = 2 * in[0]
            out[idx] = 2.0 * in[ix * Ny];
        } else if (k <= Ny / 2) {
            // First half: apply twiddle formula
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k];
                cosa = cos_lut[k];
            } else {
                sincos(k * M_PI / (2.0 * Ny), &sina, &cosa);
            }
            double Ta = in[ix * Ny + k] + in[ix * Ny + Ny - k];
            double Tb = in[ix * Ny + k] - in[ix * Ny + Ny - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            // Second half: use symmetric formula
            int k_mirror = Ny - k;
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k_mirror];
                cosa = cos_lut[k_mirror];
            } else {
                sincos(k_mirror * M_PI / (2.0 * Ny), &sina, &cosa);
            }
            double Ta = in[ix * Ny + k_mirror] + in[ix * Ny + k];
            double Tb = in[ix * Ny + k_mirror] - in[ix * Ny + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_2d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2 + 1);

    while (idx < total) {
        int ix_half = idx % (Nx / 2 + 1);
        int iy = idx / (Nx / 2 + 1);

        int out_idx = iy * (Nx / 2 + 1) + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[0 * Ny + iy];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = -in[(Nx - 1) * Ny + iy];
            out[out_idx].y = 0.0;
        } else {
            int ix_in = 2 * ix_half;
            // DST-2: z[k] = ((x[2k] - x[2k-1])/2, -((x[2k] + x[2k-1])/2))
            out[out_idx].x = (in[ix_in * Ny + iy] - in[(ix_in - 1) * Ny + iy]) / 2.0;
            out[out_idx].y = -((in[ix_in * Ny + iy] + in[(ix_in - 1) * Ny + iy]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_2d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        // FST index k = ix + 1 (mapping FFTW index ix to FST index k)
        // Input is in [Ny][Nx] transposed layout
        int k = ix + 1;

        if (k == Nx) {
            // Last element: FFTW[Nx-1] = 2 * in[0]
            out[idx] = 2.0 * in[iy * Nx];
        } else if (k <= Nx / 2) {
            // First half: apply twiddle formula
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k];
                cosa = cos_lut[k];
            } else {
                sincos(k * M_PI / (2.0 * Nx), &sina, &cosa);
            }
            double Ta = in[iy * Nx + k] + in[iy * Nx + Nx - k];
            double Tb = in[iy * Nx + k] - in[iy * Nx + Nx - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            // Second half: use symmetric formula
            int k_mirror = Nx - k;
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k_mirror];
                cosa = cos_lut[k_mirror];
            } else {
                sincos(k_mirror * M_PI / (2.0 * Nx), &sina, &cosa);
            }
            double Ta = in[iy * Nx + k_mirror] + in[iy * Nx + k];
            double Tb = in[iy * Nx + k_mirror] - in[iy * Nx + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }

        idx += blockDim.x * gridDim.x;
    }
}

// DST-3 2D kernels (FST algorithm with padded buffer)
// Buffer layout per row: [0, x0, x1, ..., x_{N-1}, 0] (N+2 elements)
// This matches the 1D FST approach (Martucci 1994)

// Setup padded buffer for Y-direction DST-3 - matches 1D layout
// Data at positions 1..Ny, boundary at position 0 (zero)
__global__ void kernel_dst3_2d_setup_padded_y(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Ny + 2;  // Padded row stride
    int total = Nx * stride;

    while (idx < total) {
        int pos = idx % stride;
        int ix = idx / stride;

        if (pos == 0 || pos == Ny + 1) {
            // Boundary at position 0, padding at Ny+1
            work[idx] = 0.0;
        } else {
            // Data at positions 1..Ny
            work[idx] = in[ix * Ny + (pos - 1)];
        }

        idx += blockDim.x * gridDim.x;
    }
}

// PreOp for Y-direction DST-3 - FCT/FST style (Makhoul 1980)
// Data at positions 0..Ny-1, boundary at position Ny (=0)
// Each block handles one row
__global__ void kernel_dst3_2d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    // One row of Ny samples (stride Ny + 2) per block iteration
    const int L = Ny;
    for (int row = blockIdx.x; row < Nx; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Thread i owns the pair (i, L-i)
        for (int i = threadIdx.x; i <= L / 2; i += blockDim.x) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * L), &sina, &cosa);
            }
            double Ta = pin[i] + pin[L - i];
            double Tb = pin[i] - pin[L - i];
            pin[i] = Ta * cosa + Tb * sina;
            pin[L - i] = Ta * sina - Tb * cosa;
        }
    }
}

// PostOp for Y-direction DST-3 - matches 1D implementation
// Each block handles one row, uses shared memory to avoid race conditions
// No /2 factor - matches FFTW unnormalized convention (same as 1D kernel)
__global__ void kernel_dst3_2d_postOp_y(
    double* __restrict__ data,
    int Nx, int Ny)
{
    extern __shared__ double sh_in[];

    // One row of Ny samples (stride Ny + 2) per block iteration
    const int L = Ny;
    for (int row = blockIdx.x; row < Nx; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Load FFT output (complex: L/2+1 complex values)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            sh_in[i] = pin[i];
            sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
        }
        __syncthreads();

        // Negate imaginary parts (FST)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i != 0)
                sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
        }
        __syncthreads();

        // Output - no /2 factor, to match the FFTW unnormalized convention
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i == 0) {
                pin[0] = 0.0;
                pin[1] = sh_in[0];
            } else {
                pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
                if (i * 2 + 1 < L + 1) {
                    pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
                }
            }
        }
        __syncthreads();   // the next row reuses sh_in
    }
}

// Copy result back from padded buffer
__global__ void kernel_dst3_2d_copy_back_y(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;
        int stride = Ny + 2;

        out[idx] = work[ix * stride + iy + 1];  // Skip padding at position 0

        idx += blockDim.x * gridDim.x;
    }
}

// X-direction kernels - matches 1D layout
// Data at positions 1..Nx, boundary at position 0 (zero)
__global__ void kernel_dst3_2d_setup_padded_x(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nx + 2;
    int total = Ny * stride;

    while (idx < total) {
        int pos = idx % stride;
        int iy = idx / stride;

        if (pos == 0 || pos == Nx + 1) {
            // Boundary at position 0, padding at Nx+1
            work[idx] = 0.0;
        } else {
            // Data at positions 1..Nx, transpose from [Nx][Ny] to [Ny][Nx+2]
            work[idx] = in[(pos - 1) * Ny + iy];
        }

        idx += blockDim.x * gridDim.x;
    }
}

// PreOp for X-direction DST-3 - FCT/FST style (Makhoul 1980)
// Each block handles one row
__global__ void kernel_dst3_2d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    // One row of Nx samples (stride Nx + 2) per block iteration
    const int L = Nx;
    for (int row = blockIdx.x; row < Ny; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Thread i owns the pair (i, L-i)
        for (int i = threadIdx.x; i <= L / 2; i += blockDim.x) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * L), &sina, &cosa);
            }
            double Ta = pin[i] + pin[L - i];
            double Tb = pin[i] - pin[L - i];
            pin[i] = Ta * cosa + Tb * sina;
            pin[L - i] = Ta * sina - Tb * cosa;
        }
    }
}

// PostOp for X-direction DST-3 - matches 1D implementation
// Each block handles one row, uses shared memory to avoid race conditions
// No /2 factor - matches FFTW unnormalized convention (same as 1D kernel)
__global__ void kernel_dst3_2d_postOp_x(
    double* __restrict__ data,
    int Nx, int Ny)
{
    extern __shared__ double sh_in[];

    // One row of Nx samples (stride Nx + 2) per block iteration
    const int L = Nx;
    for (int row = blockIdx.x; row < Ny; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Load FFT output (complex: L/2+1 complex values)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            sh_in[i] = pin[i];
            sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
        }
        __syncthreads();

        // Negate imaginary parts (FST)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i != 0)
                sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
        }
        __syncthreads();

        // Output - no /2 factor, to match the FFTW unnormalized convention
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i == 0) {
                pin[0] = 0.0;
                pin[1] = sh_in[0];
            } else {
                pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
                if (i * 2 + 1 < L + 1) {
                    pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
                }
            }
        }
        __syncthreads();   // the next row reuses sh_in
    }
}

__global__ void kernel_dst3_2d_copy_back_x(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    while (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;
        int stride = Nx + 2;

        // Output needs to be in [Nx][Ny] layout
        out[idx] = work[iy * stride + ix + 1];

        idx += blockDim.x * gridDim.x;
    }
}

//==============================================================================
// 3D DCT Kernels
// Data layout: [Nx][Ny][Nz] with linear index = (ix * Ny + iy) * Nz + iz
//==============================================================================
//------------------------------------------------------------------------------
// 3D DCT-2 Kernels
//------------------------------------------------------------------------------

__global__ void kernel_dct2_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_z = Nz / 2 + 1;
    int total = Nx * Ny * half_z;

    while (idx < total) {
        int iz_half = idx % half_z;
        int iy = (idx / half_z) % Ny;
        int ix = idx / (half_z * Ny);

        int out_idx = (ix * Ny + iy) * half_z + iz_half;
        int in_base = (ix * Ny + iy) * Nz;

        if (iz_half == 0) {
            out[out_idx].x = in[in_base];
            out[out_idx].y = 0.0;
        } else if (iz_half == Nz / 2 && Nz % 2 == 0) {  // even N only; odd N: ordinary pair below
            out[out_idx].x = in[in_base + Nz - 1];
            out[out_idx].y = 0.0;
        } else {
            int iz_even = 2 * iz_half;
            int iz_odd = iz_even - 1;
            out[out_idx].x = (in[in_base + iz_even] + in[in_base + iz_odd]) / 2.0;
            out[out_idx].y = -((in[in_base + iz_odd] - in[in_base + iz_even]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_z(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Ny + iy) * Nz;

        if (iz == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(iz * M_PI / (2.0 * Nz), &sina, &cosa);
            double Ta = in[in_base + iz] + in[in_base + (Nz - iz)];
            double Tb = in[in_base + iz] - in[in_base + (Nz - iz)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_z_lut(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Ny + iy) * Nz;

        if (iz == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina = sin_lut[iz];
            double cosa = cos_lut[iz];
            double Ta = in[in_base + iz] + in[in_base + (Nz - iz)];
            double Tb = in[in_base + iz] - in[in_base + (Nz - iz)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_y = Ny / 2 + 1;
    int total = Nx * Nz * half_y;

    while (idx < total) {
        int iy_half = idx % half_y;
        int iz = (idx / half_y) % Nz;
        int ix = idx / (half_y * Nz);

        int out_idx = (ix * Nz + iz) * half_y + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[(ix * Ny) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2 && Ny % 2 == 0) {  // even N only; odd N: ordinary pair below
            out[out_idx].x = in[(ix * Ny + Ny - 1) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int iy_even = 2 * iy_half;
            int iy_odd = iy_even - 1;
            double v_even = in[(ix * Ny + iy_even) * Nz + iz];
            double v_odd = in[(ix * Ny + iy_odd) * Nz + iz];
            out[out_idx].x = (v_even + v_odd) / 2.0;
            out[out_idx].y = -((v_odd - v_even) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Nz + iz) * Ny;

        if (iy == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(iy * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[in_base + iy] + in[in_base + (Ny - iy)];
            double Tb = in[in_base + iy] - in[in_base + (Ny - iy)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_y_lut(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Nz + iz) * Ny;

        if (iy == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina = sin_lut[iy];
            double cosa = cos_lut[iy];
            double Ta = in[in_base + iy] + in[in_base + (Ny - iy)];
            double Tb = in[in_base + iy] - in[in_base + (Ny - iy)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_x = Nx / 2 + 1;
    int total = Ny * Nz * half_x;

    while (idx < total) {
        int ix_half = idx % half_x;
        int iz = (idx / half_x) % Nz;
        int iy = idx / (half_x * Nz);

        int out_idx = (iy * Nz + iz) * half_x + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[iy * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2 && Nx % 2 == 0) {  // even N only; odd N: ordinary pair below
            out[out_idx].x = in[((Nx - 1) * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int ix_even = 2 * ix_half;
            int ix_odd = ix_even - 1;
            double v_even = in[(ix_even * Ny + iy) * Nz + iz];
            double v_odd = in[(ix_odd * Ny + iy) * Nz + iz];
            out[out_idx].x = (v_even + v_odd) / 2.0;
            out[out_idx].y = -((v_odd - v_even) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (iy * Nz + iz) * Nx;

        if (ix == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(ix * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[in_base + ix] + in[in_base + (Nx - ix)];
            double Tb = in[in_base + ix] - in[in_base + (Nx - ix)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct2_3d_postOp_x_lut(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (iy * Nz + iz) * Nx;

        if (ix == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina = sin_lut[ix];
            double cosa = cos_lut[ix];
            double Ta = in[in_base + ix] + in[in_base + (Nx - ix)];
            double Tb = in[in_base + ix] - in[in_base + (Nx - ix)];
            out[idx] = Ta * cosa + Tb * sina;
        }

        idx += blockDim.x * gridDim.x;
    }
}

//------------------------------------------------------------------------------
// 3D DCT-3 Kernels
//------------------------------------------------------------------------------

__global__ void kernel_dct3_3d_preOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * (Nz / 2);

    while (idx < total) {
        int iz = idx % (Nz / 2);
        int iy = (idx / (Nz / 2)) % Ny;
        int ix = idx / ((Nz / 2) * Ny);

        int base = (ix * Ny + iy) * Nz;
        int iz1 = iz + 1;
        int iz2 = Nz - iz - 1;

        double sina, cosa;
        sincos(iz1 * M_PI / (2.0 * Nz), &sina, &cosa);
        double Ta = data[base + iz1] + data[base + iz2];
        double Tb = data[base + iz1] - data[base + iz2];

        data[base + iz1] = Ta * sina + Tb * cosa;
        data[base + iz2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_z_lut(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * (Nz / 2);

    while (idx < total) {
        int iz = idx % (Nz / 2);
        int iy = (idx / (Nz / 2)) % Ny;
        int ix = idx / ((Nz / 2) * Ny);

        int base = (ix * Ny + iy) * Nz;
        int iz1 = iz + 1;
        int iz2 = Nz - iz - 1;

        double sina = sin_lut[iz1];
        double cosa = cos_lut[iz1];
        double Ta = data[base + iz1] + data[base + iz2];
        double Tb = data[base + iz1] - data[base + iz2];

        data[base + iz1] = Ta * sina + Tb * cosa;
        data[base + iz2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_z = Nz / 2 + 1;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Ny + iy) * half_z;

        if (iz == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iz + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iz % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_y_transpose(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        out[(ix * Nz + iz) * Ny + iy] = in[(ix * Ny + iy) * Nz + iz];

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2);

    while (idx < total) {
        int iy = idx % (Ny / 2);
        int iz = (idx / (Ny / 2)) % Nz;
        int ix = idx / ((Ny / 2) * Nz);

        int base = (ix * Nz + iz) * Ny;
        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;

        double sina, cosa;
        sincos(iy1 * M_PI / (2.0 * Ny), &sina, &cosa);
        double Ta = data[base + iy1] + data[base + iy2];
        double Tb = data[base + iy1] - data[base + iy2];

        data[base + iy1] = Ta * sina + Tb * cosa;
        data[base + iy2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_y_lut(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2);

    while (idx < total) {
        int iy = idx % (Ny / 2);
        int iz = (idx / (Ny / 2)) % Nz;
        int ix = idx / ((Ny / 2) * Nz);

        int base = (ix * Nz + iz) * Ny;
        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;

        double sina = sin_lut[iy1];
        double cosa = cos_lut[iy1];
        double Ta = data[base + iy1] + data[base + iy2];
        double Tb = data[base + iy1] - data[base + iy2];

        data[base + iy1] = Ta * sina + Tb * cosa;
        data[base + iy2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_y = Ny / 2 + 1;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Nz + iz) * half_y;

        if (iy == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iy + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iy % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_x_strided(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2);

    while (idx < total) {
        int ix = idx % (Nx / 2);
        int iz = (idx / (Nx / 2)) % Nz;
        int iy = idx / ((Nx / 2) * Nz);

        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        int idx1 = (ix1 * Ny + iy) * Nz + iz;
        int idx2 = (ix2 * Ny + iy) * Nz + iz;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[idx1] + data[idx2];
        double Tb = data[idx1] - data[idx2];

        data[idx1] = Ta * sina + Tb * cosa;
        data[idx2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_preOp_x_strided_lut(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2);

    while (idx < total) {
        int ix = idx % (Nx / 2);
        int iz = (idx / (Nx / 2)) % Nz;
        int iy = idx / ((Nx / 2) * Nz);

        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        int idx1 = (ix1 * Ny + iy) * Nz + iz;
        int idx2 = (ix2 * Ny + iy) * Nz + iz;

        double sina = sin_lut[ix1];
        double cosa = cos_lut[ix1];
        double Ta = data[idx1] + data[idx2];
        double Tb = data[idx1] - data[idx2];

        data[idx1] = Ta * sina + Tb * cosa;
        data[idx2] = Ta * cosa - Tb * sina;

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dct3_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_x = Nx / 2 + 1;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (iy * Nz + iz) * half_x;

        if (ix == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (ix + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (ix % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

//==============================================================================
// 3D DST Kernels
//==============================================================================

//------------------------------------------------------------------------------
// 3D DST-2 Kernels
//------------------------------------------------------------------------------

// 3D DST-2 Z-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * (Nz / 2 + 1);

    while (idx < total) {
        int iz_half = idx % (Nz / 2 + 1);
        int ixy = idx / (Nz / 2 + 1);
        int out_idx = ixy * (Nz / 2 + 1) + iz_half;

        if (iz_half == 0) {
            out[out_idx].x = in[ixy * Nz];
            out[out_idx].y = 0.0;
        } else if (iz_half == Nz / 2) {
            out[out_idx].x = -in[ixy * Nz + Nz - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ixy * Nz + 2 * iz_half;
            out[out_idx].x = (in[in_idx] - in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx] + in[in_idx - 1]) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_3d_postOp_z(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int ixy = idx / Nz;
        int k = iz + 1;

        if (k == Nz) {
            out[idx] = 2.0 * in[ixy * Nz];
        } else if (k <= Nz / 2) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k];
                cosa = cos_lut[k];
            } else {
                sincos(k * M_PI / (2.0 * Nz), &sina, &cosa);
            }
            double Ta = in[ixy * Nz + k] + in[ixy * Nz + Nz - k];
            double Tb = in[ixy * Nz + k] - in[ixy * Nz + Nz - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Nz - k;
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k_mirror];
                cosa = cos_lut[k_mirror];
            } else {
                sincos(k_mirror * M_PI / (2.0 * Nz), &sina, &cosa);
            }
            double Ta = in[ixy * Nz + k_mirror] + in[ixy * Nz + k];
            double Tb = in[ixy * Nz + k_mirror] - in[ixy * Nz + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }

        idx += blockDim.x * gridDim.x;
    }
}

// 3D DST-2 Y-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2 + 1);

    while (idx < total) {
        int iy_half = idx % (Ny / 2 + 1);
        int ixz = idx / (Ny / 2 + 1);
        int iz = ixz % Nz;
        int ix = ixz / Nz;
        int out_idx = ixz * (Ny / 2 + 1) + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[(ix * Ny) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2) {
            out[out_idx].x = -in[(ix * Ny + Ny - 1) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int iy_in = 2 * iy_half;
            double v1 = in[(ix * Ny + iy_in) * Nz + iz];
            double v0 = in[(ix * Ny + iy_in - 1) * Nz + iz];
            out[out_idx].x = (v1 - v0) / 2.0;
            out[out_idx].y = -((v1 + v0) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_3d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int ixz = ix * Nz + iz;
        int k = iy + 1;

        if (k == Ny) {
            out[idx] = 2.0 * in[ixz * Ny];
        } else if (k <= Ny / 2) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k];
                cosa = cos_lut[k];
            } else {
                sincos(k * M_PI / (2.0 * Ny), &sina, &cosa);
            }
            double Ta = in[ixz * Ny + k] + in[ixz * Ny + Ny - k];
            double Tb = in[ixz * Ny + k] - in[ixz * Ny + Ny - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Ny - k;
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k_mirror];
                cosa = cos_lut[k_mirror];
            } else {
                sincos(k_mirror * M_PI / (2.0 * Ny), &sina, &cosa);
            }
            double Ta = in[ixz * Ny + k_mirror] + in[ixz * Ny + k];
            double Tb = in[ixz * Ny + k_mirror] - in[ixz * Ny + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }

        idx += blockDim.x * gridDim.x;
    }
}

// 3D DST-2 X-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2 + 1);

    while (idx < total) {
        int ix_half = idx % (Nx / 2 + 1);
        int iyz = idx / (Nx / 2 + 1);
        int iz = iyz % Nz;
        int iy = iyz / Nz;
        int out_idx = iyz * (Nx / 2 + 1) + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[(0 * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = -in[((Nx - 1) * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int ix_in = 2 * ix_half;
            double v1 = in[(ix_in * Ny + iy) * Nz + iz];
            double v0 = in[((ix_in - 1) * Ny + iy) * Nz + iz];
            out[out_idx].x = (v1 - v0) / 2.0;
            out[out_idx].y = -((v1 + v0) / 2.0);
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_dst2_3d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int iyz = iy * Nz + iz;
        int k = ix + 1;

        if (k == Nx) {
            out[idx] = 2.0 * in[iyz * Nx];
        } else if (k <= Nx / 2) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k];
                cosa = cos_lut[k];
            } else {
                sincos(k * M_PI / (2.0 * Nx), &sina, &cosa);
            }
            double Ta = in[iyz * Nx + k] + in[iyz * Nx + Nx - k];
            double Tb = in[iyz * Nx + k] - in[iyz * Nx + Nx - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Nx - k;
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[k_mirror];
                cosa = cos_lut[k_mirror];
            } else {
                sincos(k_mirror * M_PI / (2.0 * Nx), &sina, &cosa);
            }
            double Ta = in[iyz * Nx + k_mirror] + in[iyz * Nx + k];
            double Tb = in[iyz * Nx + k_mirror] - in[iyz * Nx + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }

        idx += blockDim.x * gridDim.x;
    }
}

//------------------------------------------------------------------------------
// 3D DST-3 Kernels - FCT/FST style (Makhoul 1980)
//------------------------------------------------------------------------------

// Setup padded buffer for Z-direction: [0, x0, x1, ..., x_{Nz-1}, 0]
__global__ void kernel_dst3_3d_setup_z(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nz + 2;
    int total = Nx * Ny * stride;

    while (idx < total) {
        int pos = idx % stride;
        int ixy = idx / stride;

        if (pos == 0 || pos == Nz + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[ixy * Nz + (pos - 1)];
        }

        idx += blockDim.x * gridDim.x;
    }
}

// PreOp twiddle factor for Z-direction - each XY point processes independently
__global__ void kernel_dst3_3d_preOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    // One row of Nz samples (stride Nz + 2) per block iteration
    const int L = Nz;
    for (int row = blockIdx.x; row < Nx * Ny; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Thread i owns the pair (i, L-i)
        for (int i = threadIdx.x; i <= L / 2; i += blockDim.x) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * L), &sina, &cosa);
            }
            double Ta = pin[i] + pin[L - i];
            double Tb = pin[i] - pin[L - i];
            pin[i] = Ta * cosa + Tb * sina;
            pin[L - i] = Ta * sina - Tb * cosa;
        }
    }
}

// PostOp for Z-direction - extract DST-3 result
__global__ void kernel_dst3_3d_postOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];

    // One row of Nz samples (stride Nz + 2) per block iteration
    const int L = Nz;
    for (int row = blockIdx.x; row < Nx * Ny; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Load FFT output (complex: L/2+1 complex values)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            sh_in[i] = pin[i];
            sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
        }
        __syncthreads();

        // Negate imaginary parts (FST)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i != 0)
                sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
        }
        __syncthreads();

        // Output - no /2 factor, to match the FFTW unnormalized convention
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i == 0) {
                pin[0] = 0.0;
                pin[1] = sh_in[0];
            } else {
                pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
                if (i * 2 + 1 < L + 1) {
                    pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
                }
            }
        }
        __syncthreads();   // the next row reuses sh_in
    }
}

// Copy result back from padded buffer
__global__ void kernel_dst3_3d_copy_z(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int ixy = idx / Nz;
        out[idx] = work[ixy * (Nz + 2) + iz + 1];

        idx += blockDim.x * gridDim.x;
    }
}

// Setup padded buffer for Y-direction: [0, x0, x1, ..., x_{Ny-1}, 0]
__global__ void kernel_dst3_3d_setup_y(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Ny + 2;
    int total = Nx * Nz * stride;

    while (idx < total) {
        int pos = idx % stride;
        int ixz = idx / stride;
        int iz = ixz % Nz;
        int ix = ixz / Nz;

        if (pos == 0 || pos == Ny + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[(ix * Ny + (pos - 1)) * Nz + iz];
        }

        idx += blockDim.x * gridDim.x;
    }
}

// PreOp twiddle factor for Y-direction
__global__ void kernel_dst3_3d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    // One row of Ny samples (stride Ny + 2) per block iteration
    const int L = Ny;
    for (int row = blockIdx.x; row < Nx * Nz; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Thread i owns the pair (i, L-i)
        for (int i = threadIdx.x; i <= L / 2; i += blockDim.x) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * L), &sina, &cosa);
            }
            double Ta = pin[i] + pin[L - i];
            double Tb = pin[i] - pin[L - i];
            pin[i] = Ta * cosa + Tb * sina;
            pin[L - i] = Ta * sina - Tb * cosa;
        }
    }
}

// PostOp for Y-direction
__global__ void kernel_dst3_3d_postOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];

    // One row of Ny samples (stride Ny + 2) per block iteration
    const int L = Ny;
    for (int row = blockIdx.x; row < Nx * Nz; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Load FFT output (complex: L/2+1 complex values)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            sh_in[i] = pin[i];
            sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
        }
        __syncthreads();

        // Negate imaginary parts (FST)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i != 0)
                sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
        }
        __syncthreads();

        // Output - no /2 factor, to match the FFTW unnormalized convention
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i == 0) {
                pin[0] = 0.0;
                pin[1] = sh_in[0];
            } else {
                pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
                if (i * 2 + 1 < L + 1) {
                    pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
                }
            }
        }
        __syncthreads();   // the next row reuses sh_in
    }
}

// Copy result back for Y-direction
__global__ void kernel_dst3_3d_copy_y(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int ixz = ix * Nz + iz;
        out[idx] = work[ixz * (Ny + 2) + iy + 1];

        idx += blockDim.x * gridDim.x;
    }
}

// Setup padded buffer for X-direction: [0, x0, x1, ..., x_{Nx-1}, 0]
__global__ void kernel_dst3_3d_setup_x(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nx + 2;
    int total = Ny * Nz * stride;

    while (idx < total) {
        int pos = idx % stride;
        int iyz = idx / stride;
        int iz = iyz % Nz;
        int iy = iyz / Nz;

        if (pos == 0 || pos == Nx + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[((pos - 1) * Ny + iy) * Nz + iz];
        }

        idx += blockDim.x * gridDim.x;
    }
}

// PreOp twiddle factor for X-direction
__global__ void kernel_dst3_3d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny, int Nz,
    const double* __restrict__ sin_lut,
    const double* __restrict__ cos_lut)
{
    // One row of Nx samples (stride Nx + 2) per block iteration
    const int L = Nx;
    for (int row = blockIdx.x; row < Ny * Nz; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Thread i owns the pair (i, L-i)
        for (int i = threadIdx.x; i <= L / 2; i += blockDim.x) {
            double sina, cosa;
            if (sin_lut && cos_lut) {
                sina = sin_lut[i];
                cosa = cos_lut[i];
            } else {
                sincos(i * M_PI / (2.0 * L), &sina, &cosa);
            }
            double Ta = pin[i] + pin[L - i];
            double Tb = pin[i] - pin[L - i];
            pin[i] = Ta * cosa + Tb * sina;
            pin[L - i] = Ta * sina - Tb * cosa;
        }
    }
}

// PostOp for X-direction
__global__ void kernel_dst3_3d_postOp_x(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];

    // One row of Nx samples (stride Nx + 2) per block iteration
    const int L = Nx;
    for (int row = blockIdx.x; row < Ny * Nz; row += gridDim.x) {
        double* pin = data + row * (L + 2);
        // Load FFT output (complex: L/2+1 complex values)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            sh_in[i] = pin[i];
            sh_in[i + L / 2 + 1] = pin[i + L / 2 + 1];
        }
        __syncthreads();

        // Negate imaginary parts (FST)
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i != 0)
                sh_in[i * 2 + 1] = -sh_in[i * 2 + 1];
        }
        __syncthreads();

        // Output - no /2 factor, to match the FFTW unnormalized convention
        for (int i = threadIdx.x; i < L / 2 + 1; i += blockDim.x) {
            if (i == 0) {
                pin[0] = 0.0;
                pin[1] = sh_in[0];
            } else {
                pin[2 * i] = (sh_in[i * 2 + 1] - sh_in[i * 2]);
                if (i * 2 + 1 < L + 1) {
                    pin[2 * i + 1] = (sh_in[i * 2] + sh_in[i * 2 + 1]);
                }
            }
        }
        __syncthreads();   // the next row reuses sh_in
    }
}

// Copy result back for X-direction
__global__ void kernel_dst3_3d_copy_x(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    while (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int iyz = iy * Nz + iz;
        out[idx] = work[iyz * (Nx + 2) + ix + 1];

        idx += blockDim.x * gridDim.x;
    }
}

//==============================================================================
// CudaRealTransform1D Implementation
//==============================================================================


//------------------------------------------------------------------------------
// Supported transform lengths, verified against FFTW (tests/TestCudaDCT.cu and
// tests/TestCudaDST.cu, including odd lengths): DCT-2/3 and the 1D DST-3 accept any
// length; the DST-2 kernels (all dimensions) and the 2D/3D DST-3 kernels assume even
// lengths.
// Unsupported lengths are refused here instead of returning wrong values (nothing in
// the library reaches them: CudaFFT rejects odd absorbing axes, its odd reflecting
// axes take the DCT-2/3 of any length, and the CrysFFT uses DCT-2/3 only).
//------------------------------------------------------------------------------
static const char* transform_type_name(CudaTransformType type)
{
    switch (type)
    {
        case CUDA_DCT_2: return "DCT-2";
        case CUDA_DCT_3: return "DCT-3";
        case CUDA_DST_2: return "DST-2";
        case CUDA_DST_3: return "DST-3";
    }
    return "unknown";
}

static void check_transform_length(CudaTransformType type, int N, const char* axis, int dim)
{
    const bool even = (N % 2 == 0);
    const bool unsupported =
        (type == CUDA_DST_2 && !even) ||
        (type == CUDA_DST_3 && dim >= 2 && !even);
    if (!unsupported)
        return;
    std::string msg = std::string("CudaRealTransform: ") + transform_type_name(type) + " does not support an " +
                      (even ? "even" : "odd") + " length in " + std::to_string(dim) + "D; got " + axis + " = " +
                      std::to_string(N);
    throw std::runtime_error(msg);
}

// Shared memory the 1D kernels of a transform stage one line in (see execute()): the DCT-2
// post-op reads N values and the DCT-3 post-op N + 2
static size_t shared_bytes_1d(CudaTransformType type, int N)
{
    switch (type) {
        case CUDA_DCT_2: return sizeof(double) * N;
        case CUDA_DCT_3: return sizeof(double) * (N + 2);
        case CUDA_DST_2: return sizeof(double) * N;
        case CUDA_DST_3: return sizeof(double) * (N + 2);
        default:         return 0;
    }
}

// The 2D/3D DST-3 kernels (executeX/Y/Z_DST3) stage one line, N+2 values, in the
// shared memory of a block; the other 2D/3D kernels use none
static void check_shared_line(CudaTransformType type, int N, const char* axis, int dim)
{
    if (type != CUDA_DST_3)
        return;
    int device = 0, max_shared = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device));
    const size_t needed = sizeof(double) * (static_cast<size_t>(N) + 2);
    if (needed > static_cast<size_t>(max_shared))
        throw std::runtime_error("CudaRealTransform: the " + std::to_string(dim) + "D DST-3 (absorbing axis) of " +
            axis + " = " + std::to_string(N) + " points stages " + std::to_string(needed) +
            " bytes in shared memory, more than the " + std::to_string(max_shared) +
            " bytes a block may use on this GPU; use a CPU platform for this grid (platform \"auto\" does).");
}

CudaRealTransform1D::CudaRealTransform1D(int N, CudaTransformType type)
    : N_(N), type_(type), initialized_(false)
{
    check_transform_length(type_, N_, "N", 1);
    {
        int device = 0, max_shared = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device));
        const size_t needed = shared_bytes_1d(type_, N_);
        if (needed > static_cast<size_t>(max_shared))
            throw std::runtime_error("CudaRealTransform: the 1D " + std::string(transform_type_name(type_)) +
                " of " + std::to_string(N_) + " points stages " + std::to_string(needed) +
                " bytes in shared memory, more than the " + std::to_string(max_shared) +
                " bytes a block may use on this GPU; use a CPU platform for this 1D grid "
                "(the default platform for 1D).");
    }
    int fftSize = getFFTSize(type_, N_);

    // Allocate buffers
    int complexSize = fftSize / 2 + 1;
    // DCT-3 runs its D2Z into d_work_ (complexSize complex values)
    int workSize = std::max(fftSize, 2 * complexSize);
    d_work_.allocate(workSize);
    d_x1_.allocate(2 * static_cast<size_t>(complexSize));   // complex values

    // Create FFT plan
    if (type_ == CUDA_DCT_2) {
        if (plan_.create(cufftPlan1d, N_, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan");
        }
    } else if (type_ == CUDA_DCT_3) {
        if (plan_.create(cufftPlan1d, N_, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan");
        }
    } else if (type_ == CUDA_DST_2) {
        // DST-2: Z2D FFT of size N (FST algorithm)
        if (plan_.create(cufftPlan1d, N_, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for DST-2");
        }
    } else if (type_ == CUDA_DST_3) {
        // DST-3: D2Z FFT of size N (FST algorithm)
        if (plan_.create(cufftPlan1d, N_, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for DST-3");
        }
    } else {
        throw std::runtime_error("Unsupported transform type");
    }

    if (type_ == CUDA_DCT_2 || type_ == CUDA_DST_2 || type_ == CUDA_DST_3)
        allocate_dct2_lut(N_, dct2_sin_, dct2_cos_);
    if (type_ == CUDA_DCT_3)
        allocate_dct3_lut(N_, dct3_sin_, dct3_cos_);

    initialized_ = true;
}

void CudaRealTransform1D::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    if (initialized_)
    {
        if (cufftSetStream(plan_, stream_) != CUFFT_SUCCESS)
        {
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform1D");
        }
    }
}

void CudaRealTransform1D::execute(double* d_data, cudaStream_t stream)
{
    set_stream(stream);
    execute(d_data);
}

void CudaRealTransform1D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform1D not initialized");
    }

    int nThread = pow2roundup(N_);

    switch (type_) {
        case CUDA_DCT_2: {
            // packed spectrum (N/2+1 complex) into d_x1_, which has room for it
            kernel_dct2_preOp<<<1, std::min(nThread, 256), 0, stream_>>>(d_data, d_x1_, N_);
            CUDA_CHECK_LAUNCH();
            CUFFT_CHECK(cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_x1_.get()), d_work_));
            kernel_dct2_postOp<<<1, std::min(nThread, 256), shared_bytes_1d(type_, N_), stream_>>>(
                d_work_, N_, dct2_sin_, dct2_cos_);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpyAsync(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            break;
        }
        case CUDA_DCT_3: {
            kernel_dct3_preOp<<<1, std::min(nThread, 256), 0, stream_>>>(d_data, N_, dct3_sin_, dct3_cos_);
            CUDA_CHECK_LAUNCH();
            CUFFT_CHECK(cufftExecD2Z(plan_, d_data, reinterpret_cast<cufftDoubleComplex*>(d_work_.get())));
            kernel_dct3_postOp<<<1, std::min(nThread, 256), shared_bytes_1d(type_, N_), stream_>>>(d_work_, N_);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpyAsync(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            break;
        }
        case CUDA_DST_2: {
            // FCT/FST style (Makhoul 1980): input at buffer[1..N], output at buffer[1..N]
            // Setup buffer: [0, x0, x1, ..., x_{N-1}, 0]
            CUDA_CHECK(cudaMemsetAsync(d_work_, 0, sizeof(double) * (N_ + 2), stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_work_ + 1, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            // PreOp (reads from buffer[1..N]) - needs N/2+1 threads
            kernel_dst2_preOp<<<1, std::min(N_ / 2 + 1, 256), N_ * sizeof(double), stream_>>>(d_work_, N_);
            CUDA_CHECK_LAUNCH();
            // Z2D FFT of size N
            CUFFT_CHECK(cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_.get()), d_work_));
            // PostOp
            kernel_dst2_postOp<<<1, std::min(N_ / 2 + 1, 256), N_ * sizeof(double), stream_>>>(
                d_work_, N_, dct2_sin_, dct2_cos_);
            CUDA_CHECK_LAUNCH();
            // Output mapping: result[0..N-1] = buffer[1..N]
            CUDA_CHECK(cudaMemcpyAsync(d_data, d_work_ + 1, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            break;
        }
        case CUDA_DST_3: {
            // FCT/FST style (Makhoul 1980): input at buffer[1..N] (buffer[0]=0), output at buffer[1..N]
            // Setup buffer: [0, x0, x1, ..., x_{N-1}, 0]
            CUDA_CHECK(cudaMemsetAsync(d_work_, 0, sizeof(double) * (N_ + 2), stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_work_ + 1, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            // PreOp (reads buffer[0..N])
            kernel_dst3_preOp<<<1, std::min(N_ / 2 + 1, 256), 0, stream_>>>(d_work_, N_, dct2_sin_, dct2_cos_);
            CUDA_CHECK_LAUNCH();
            // D2Z FFT of size N
            CUFFT_CHECK(cufftExecD2Z(plan_, d_work_, reinterpret_cast<cufftDoubleComplex*>(d_work_.get())));
            // PostOp
            kernel_dst3_postOp<<<1, std::min(N_ / 2 + 1, 256), (N_ + 2) * sizeof(double), stream_>>>(d_work_, N_);
            CUDA_CHECK_LAUNCH();
            // Output mapping: result[0..N-1] = buffer[1..N]
            CUDA_CHECK(cudaMemcpyAsync(d_data, d_work_ + 1, sizeof(double) * N_, cudaMemcpyDeviceToDevice, stream_));
            break;
        }
        default:
            throw std::runtime_error("Unsupported transform type");
    }
}

int CudaRealTransform1D::get_size() const
{
    return N_;
}

double CudaRealTransform1D::get_normalization() const
{
    return getNormFactor(type_, N_);
}

//==============================================================================
// CudaRealTransform2D Implementation
//==============================================================================

CudaRealTransform2D::CudaRealTransform2D(int Nx, int Ny, CudaTransformType type)
    : CudaRealTransform2D(Nx, Ny, type, type)
{
}

CudaRealTransform2D::CudaRealTransform2D(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y)
    : Nx_(Nx), Ny_(Ny), type_x_(type_x), type_y_(type_y), initialized_(false)
{
    check_transform_length(type_x_, Nx_, "Nx", 2);
    check_transform_length(type_y_, Ny_, "Ny", 2);
    check_shared_line(type_x_, Nx_, "Nx", 2);
    check_shared_line(type_y_, Ny_, "Ny", 2);

    init();
    set_stream(stream_);
}

//------------------------------------------------------------------------------
// Per-axis cuFFT plan for the DCT/DST types 2 and 3.
//
// All axes use the same plan layout: the transform of length N is batched `batch`
// times over contiguous rows. Only the DCT-3 input layout differs per axis (the X
// axis of the 3D/2D transforms reads the strided layout produced by the preceding
// axes), so its input stride and distance are passed in.
//------------------------------------------------------------------------------
static void create_axis_plan(
    CufftPlan& plan, CudaTransformType type, int N, int batch,
    const char* axis, int dct3_istride, int dct3_idist)
{
    auto fail = [axis](const char* kind, const char* suffix) {
        throw std::runtime_error(std::string("Failed to create cuFFT ") + kind + " plan for " + axis + suffix);
    };

    if (type == CUDA_DCT_2) {
        int n[1] = {N};
        int inembed[1] = {N / 2 + 1};
        int onembed[1] = {N};
        if (plan.create(cufftPlanMany, 1, n, inembed, 1, N / 2 + 1,
                        onembed, 1, N, CUFFT_Z2D, batch) != CUFFT_SUCCESS)
            fail("Z2D", "");
    } else if (type == CUDA_DCT_3) {
        int n[1] = {N};
        int inembed[1] = {N};
        int onembed[1] = {N / 2 + 1};
        if (plan.create(cufftPlanMany, 1, n, inembed, dct3_istride, dct3_idist,
                        onembed, 1, N / 2 + 1, CUFFT_D2Z, batch) != CUFFT_SUCCESS)
            fail("D2Z", "");
    } else if (type == CUDA_DST_2) {
        // DST-2: Z2D FFT of size N - FCT/FST style (Makhoul 1980)
        int n[1] = {N};
        int inembed[1] = {N / 2 + 1};
        int onembed[1] = {N};
        if (plan.create(cufftPlanMany, 1, n, inembed, 1, N / 2 + 1,
                        onembed, 1, N, CUFFT_Z2D, batch) != CUFFT_SUCCESS)
            fail("Z2D", " (DST-2)");
    } else if (type == CUDA_DST_3) {
        // DST-3: D2Z FFT of size N - FCT/FST style (Makhoul 1980) with a padded
        // buffer of N+2 elements per row (in-place, interleaved output)
        int n[1] = {N};
        int inembed[1] = {N + 2};
        int onembed[1] = {N / 2 + 1};
        if (plan.create(cufftPlanMany, 1, n, inembed, 1, N + 2,
                        onembed, 1, N / 2 + 1, CUFFT_D2Z, batch) != CUFFT_SUCCESS)
            fail("D2Z", " (DST-3)");
    } else {
        throw std::runtime_error(std::string("Unsupported transform type for ") + axis);
    }
}

void CudaRealTransform2D::init()
{
    M_ = Nx_ * Ny_;

    // Calculate FFT sizes
    int fftSize_y = getFFTSize(type_y_, Ny_);
    int fftSize_x = getFFTSize(type_x_, Nx_);

    // Calculate buffer sizes
    size_t work_size_y = sizeof(double) * Nx_ * fftSize_y;
    size_t work_size_x = sizeof(double) * Ny_ * fftSize_x;

    size_t complex_size_y = sizeof(cufftDoubleComplex) * Nx_ * (fftSize_y / 2 + 1);
    size_t complex_size_x = sizeof(cufftDoubleComplex) * Ny_ * (fftSize_x / 2 + 1);

    // Allocate buffers
    d_work_.allocate(doubles_for_bytes(std::max(work_size_y, work_size_x)));
    d_temp_.allocate(M_);
    d_x1_.allocate(doubles_for_bytes(std::max(complex_size_y, complex_size_x)));

    // Per-axis plans. The X axis DCT-3 reads the y-major layout directly,
    // hence its strided input (istride = Ny, idist = 1).
    create_axis_plan(plan_y_, type_y_, Ny_, Nx_, "Y", 1, Ny_);
    create_axis_plan(plan_x_, type_x_, Nx_, Ny_, "X", Ny_, 1);

    if (type_x_ == CUDA_DCT_2 || type_x_ == CUDA_DST_2 || type_x_ == CUDA_DST_3)
        allocate_dct2_lut(Nx_, dct2_sin_x_, dct2_cos_x_);
    if (type_x_ == CUDA_DCT_3)
        allocate_dct3_lut(Nx_, dct3_sin_x_, dct3_cos_x_);

    if (type_y_ == CUDA_DCT_2 || type_y_ == CUDA_DST_2 || type_y_ == CUDA_DST_3)
        allocate_dct2_lut(Ny_, dct2_sin_y_, dct2_cos_y_);
    if (type_y_ == CUDA_DCT_3)
        allocate_dct3_lut(Ny_, dct3_sin_y_, dct3_cos_y_);

    initialized_ = true;
    set_stream(stream_);
}

void CudaRealTransform2D::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    if (initialized_)
    {
        if (cufftSetStream(plan_x_, stream_) != CUFFT_SUCCESS)
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform2D (plan_x)");
        if (cufftSetStream(plan_y_, stream_) != CUFFT_SUCCESS)
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform2D (plan_y)");
    }
}

void CudaRealTransform2D::execute(double* d_data, cudaStream_t stream)
{
    set_stream(stream);
    execute(d_data);
}

void CudaRealTransform2D::executeY_DCT2(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dct2_2d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_data, d_complex, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_y_, d_complex, d_work_));
    kernel_dct2_2d_postOp_y<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_temp_, Nx_, Ny_, dct2_sin_y_, dct2_cos_y_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeY_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    CUDA_CHECK(cudaMemcpyAsync(d_temp_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice, stream_));

    kernel_dct3_2d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(
        d_temp_, Nx_, Ny_, dct3_sin_y_, dct3_cos_y_);
    CUDA_CHECK_LAUNCH();

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_y_, d_temp_, d_complex));

    kernel_dct3_2d_postOp_y<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_complex, d_temp_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeY_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dst2_2d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_data, d_complex, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_y_, d_complex, d_work_));
    kernel_dst2_2d_postOp_y<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_temp_, Nx_, Ny_, dct2_sin_y_, dct2_cos_y_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeY_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980) with padded buffer:
    // 1. Setup padded buffer [Nx][Ny+2] with boundary zeros
    // 2. PreOp (in-place on padded buffer)
    // 3. D2Z FFT (in-place)
    // 4. PostOp (in-place with shared memory)
    // 5. Copy results back

    int blockSize = 256;
    int stride = Ny_ + 2;

    // Step 1: Setup padded buffer
    int total_setup = Nx_ * stride;
    int numBlocks_setup = grid_blocks(total_setup, blockSize);
    kernel_dst3_2d_setup_padded_y<<<numBlocks_setup, blockSize, 0, stream_>>>(d_data, d_work_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();

    // Step 2: PreOp on padded buffer (each block handles one row)
    kernel_dst3_2d_preOp_y<<<Nx_, std::min(Ny_ / 2 + 1, 256), 0, stream_>>>(
        d_work_, Nx_, Ny_, dct2_sin_y_, dct2_cos_y_);
    CUDA_CHECK_LAUNCH();

    // Step 3: D2Z FFT in-place on padded buffer
    CUFFT_CHECK(cufftExecD2Z(plan_y_, d_work_, reinterpret_cast<cufftDoubleComplex*>(d_work_.get())));

    // Step 4: PostOp in-place (each block handles one row, needs shared memory)
    size_t sharedSize = sizeof(double) * (Ny_ + 2);
    kernel_dst3_2d_postOp_y<<<Nx_, std::min(Ny_ / 2 + 1, 256), sharedSize, stream_>>>(d_work_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();

    // Step 5: Copy results back to d_temp_
    int total_copy = Nx_ * Ny_;
    int numBlocks_copy = grid_blocks(total_copy, blockSize);
    kernel_dst3_2d_copy_back_y<<<numBlocks_copy, blockSize, 0, stream_>>>(d_work_, d_temp_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeX_DCT2(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Ny_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dct2_2d_preOp_x<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_x_, d_complex, d_work_));
    kernel_dct2_2d_postOp_x<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_data, Nx_, Ny_, dct2_sin_x_, dct2_cos_x_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeX_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Ny_ * (Nx_ / 2);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    kernel_dct3_2d_preOp_x_strided<<<numBlocks_preOp, blockSize, 0, stream_>>>(
        d_temp_, Nx_, Ny_, dct3_sin_x_, dct3_cos_x_);
    CUDA_CHECK_LAUNCH();

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_x_, d_temp_, d_complex));

    kernel_dct3_2d_postOp_x<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_complex, d_data, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeX_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Ny_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = grid_blocks(total_postOp, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dst2_2d_preOp_x<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_x_, d_complex, d_work_));
    kernel_dst2_2d_postOp_x<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_data, Nx_, Ny_, dct2_sin_x_, dct2_cos_x_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::executeX_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980) with padded buffer:
    // 1. Setup padded buffer [Ny][Nx+2] with boundary zeros and transpose from d_temp_
    // 2. PreOp (in-place on padded buffer)
    // 3. D2Z FFT (in-place)
    // 4. PostOp (in-place with shared memory)
    // 5. Copy results back with transpose

    int blockSize = 256;
    int stride = Nx_ + 2;

    // Step 1: Setup padded buffer with transpose
    int total_setup = Ny_ * stride;
    int numBlocks_setup = grid_blocks(total_setup, blockSize);
    kernel_dst3_2d_setup_padded_x<<<numBlocks_setup, blockSize, 0, stream_>>>(d_temp_, d_work_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();

    // Step 2: PreOp on padded buffer (each block handles one row)
    kernel_dst3_2d_preOp_x<<<Ny_, std::min(Nx_ / 2 + 1, 256), 0, stream_>>>(
        d_work_, Nx_, Ny_, dct2_sin_x_, dct2_cos_x_);
    CUDA_CHECK_LAUNCH();

    // Step 3: D2Z FFT in-place on padded buffer
    CUFFT_CHECK(cufftExecD2Z(plan_x_, d_work_, reinterpret_cast<cufftDoubleComplex*>(d_work_.get())));

    // Step 4: PostOp in-place (each block handles one row, needs shared memory)
    size_t sharedSize = sizeof(double) * (Nx_ + 2);
    kernel_dst3_2d_postOp_x<<<Ny_, std::min(Nx_ / 2 + 1, 256), sharedSize, stream_>>>(d_work_, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();

    // Step 5: Copy results back to d_data with transpose
    int total_copy = Nx_ * Ny_;
    int numBlocks_copy = grid_blocks(total_copy, blockSize);
    kernel_dst3_2d_copy_back_x<<<numBlocks_copy, blockSize, 0, stream_>>>(d_work_, d_data, Nx_, Ny_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform2D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform2D not initialized");
    }

    // Y dimension transform
    switch (type_y_) {
        case CUDA_DCT_2: executeY_DCT2(d_data); break;
        case CUDA_DCT_3: executeY_DCT3(d_data); break;
        case CUDA_DST_2: executeY_DST2(d_data); break;
        case CUDA_DST_3: executeY_DST3(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Y dimension");
    }

    // X dimension transform
    switch (type_x_) {
        case CUDA_DCT_2: executeX_DCT2(d_data); break;
        case CUDA_DCT_3: executeX_DCT3(d_data); break;
        case CUDA_DST_2: executeX_DST2(d_data); break;
        case CUDA_DST_3: executeX_DST3(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for X dimension");
    }
}

void CudaRealTransform2D::get_dims(int& nx, int& ny) const
{
    nx = Nx_;
    ny = Ny_;
}

void CudaRealTransform2D::get_types(CudaTransformType& type_x, CudaTransformType& type_y) const
{
    type_x = type_x_;
    type_y = type_y_;
}

double CudaRealTransform2D::get_normalization() const
{
    return getNormFactor(type_x_, Nx_) * getNormFactor(type_y_, Ny_);
}

//==============================================================================
// CudaRealTransform3D Implementation
//==============================================================================

CudaRealTransform3D::CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type)
    : CudaRealTransform3D(Nx, Ny, Nz, type, type, type)
{
}

CudaRealTransform3D::CudaRealTransform3D(int Nx, int Ny, int Nz,
                                         CudaTransformType type_x,
                                         CudaTransformType type_y,
                                         CudaTransformType type_z)
    : Nx_(Nx), Ny_(Ny), Nz_(Nz), type_x_(type_x), type_y_(type_y), type_z_(type_z),
      initialized_(false)
{
    check_transform_length(type_x_, Nx_, "Nx", 3);
    check_transform_length(type_y_, Ny_, "Ny", 3);
    check_transform_length(type_z_, Nz_, "Nz", 3);
    check_shared_line(type_x_, Nx_, "Nx", 3);
    check_shared_line(type_y_, Ny_, "Ny", 3);
    check_shared_line(type_z_, Nz_, "Nz", 3);

    init();
    set_stream(stream_);
}

void CudaRealTransform3D::init()
{
    // Per-dimension processing
    M_ = Nx_ * Ny_ * Nz_;

    // Calculate FFT sizes
    int fftSize_z = getFFTSize(type_z_, Nz_);
    int fftSize_y = getFFTSize(type_y_, Ny_);
    int fftSize_x = getFFTSize(type_x_, Nx_);

    // Calculate buffer sizes
    size_t work_size_z = sizeof(double) * Nx_ * Ny_ * fftSize_z;
    size_t work_size_y = sizeof(double) * Nx_ * Nz_ * fftSize_y;
    size_t work_size_x = sizeof(double) * Ny_ * Nz_ * fftSize_x;

    size_t max_work = std::max({work_size_z, work_size_y, work_size_x});

    size_t complex_size_z = sizeof(cufftDoubleComplex) * Nx_ * Ny_ * (fftSize_z / 2 + 1);
    size_t complex_size_y = sizeof(cufftDoubleComplex) * Nx_ * Nz_ * (fftSize_y / 2 + 1);
    size_t complex_size_x = sizeof(cufftDoubleComplex) * Ny_ * Nz_ * (fftSize_x / 2 + 1);

    size_t max_complex = std::max({complex_size_z, complex_size_y, complex_size_x});

    // Allocate buffers
    d_work_.allocate(doubles_for_bytes(max_work));
    d_temp_.allocate(M_);
    d_x1_.allocate(doubles_for_bytes(max_complex));

    // Per-axis plans. The X axis DCT-3 reads the (y,z)-major layout directly,
    // hence its strided input (istride = Ny*Nz, idist = 1).
    create_axis_plan(plan_z_, type_z_, Nz_, Nx_ * Ny_, "Z", 1, Nz_);
    create_axis_plan(plan_y_, type_y_, Ny_, Nx_ * Nz_, "Y", 1, Ny_);
    create_axis_plan(plan_x_, type_x_, Nx_, Ny_ * Nz_, "X", Ny_ * Nz_, 1);

    if (type_x_ == CUDA_DCT_2 || type_x_ == CUDA_DST_2 || type_x_ == CUDA_DST_3)
        allocate_dct2_lut(Nx_, dct2_sin_x_, dct2_cos_x_);
    if (type_x_ == CUDA_DCT_3)
        allocate_dct3_lut(Nx_, dct3_sin_x_, dct3_cos_x_);
    if (type_y_ == CUDA_DCT_2 || type_y_ == CUDA_DST_2 || type_y_ == CUDA_DST_3)
        allocate_dct2_lut(Ny_, dct2_sin_y_, dct2_cos_y_);
    if (type_y_ == CUDA_DCT_3)
        allocate_dct3_lut(Ny_, dct3_sin_y_, dct3_cos_y_);
    if (type_z_ == CUDA_DCT_2 || type_z_ == CUDA_DST_2 || type_z_ == CUDA_DST_3)
        allocate_dct2_lut(Nz_, dct2_sin_z_, dct2_cos_z_);
    if (type_z_ == CUDA_DCT_3)
        allocate_dct3_lut(Nz_, dct3_sin_z_, dct3_cos_z_);

    initialized_ = true;
}

void CudaRealTransform3D::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    if (initialized_)
    {
        if (cufftSetStream(plan_x_, stream_) != CUFFT_SUCCESS)
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform3D (plan_x)");
        if (cufftSetStream(plan_y_, stream_) != CUFFT_SUCCESS)
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform3D (plan_y)");
        if (cufftSetStream(plan_z_, stream_) != CUFFT_SUCCESS)
            throw std::runtime_error("Failed to set cuFFT stream for CudaRealTransform3D (plan_z)");
    }
}

void CudaRealTransform3D::execute(double* d_data, cudaStream_t stream)
{
    set_stream(stream);
    execute(d_data);
}

//------------------------------------------------------------------------------
// 3D Z dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeZ_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_z = Nz_ / 2 + 1;
    int total_preOp = Nx_ * Ny_ * half_z;
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dct2_3d_preOp_z<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_data, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_z_, d_complex, d_work_));
    if (!dct2_sin_z_.empty() && !dct2_cos_z_.empty())
        kernel_dct2_3d_postOp_z_lut<<<numBlocks_postOp, blockSize, 0, stream_>>>(
            d_work_, d_temp_, Nx_, Ny_, Nz_, dct2_sin_z_, dct2_cos_z_);
    else
        kernel_dct2_3d_postOp_z<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeZ_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * Ny_ * (Nz_ / 2);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    CUDA_CHECK(cudaMemcpyAsync(d_temp_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice, stream_));

    if (!dct3_sin_z_.empty() && !dct3_cos_z_.empty())
        kernel_dct3_3d_preOp_z_lut<<<numBlocks_preOp, blockSize, 0, stream_>>>(
            d_temp_, Nx_, Ny_, Nz_, dct3_sin_z_, dct3_cos_z_);
    else
        kernel_dct3_3d_preOp_z<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_z_, d_temp_, d_complex));

    kernel_dct3_3d_postOp_z<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_complex, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeZ_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * Ny_ * (Nz_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dst2_3d_preOp_z<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_data, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_z_, d_complex, d_work_));
    kernel_dst2_3d_postOp_z<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_temp_, Nx_, Ny_, Nz_, dct2_sin_z_, dct2_cos_z_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeZ_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Nx_ * Ny_;
    int stride = Nz_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = grid_blocks(total_setup, blockSize);
    int numBlocks_copy = grid_blocks(M_, blockSize);

    // Setup padded buffer: [0, x0, x1, ..., x_{Nz-1}, 0] for each XY point
    kernel_dst3_3d_setup_z<<<numBlocks_setup, blockSize, 0, stream_>>>(d_data, d_work_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // PreOp twiddle factors - 1 block per XY point, Nz/2+1 threads per block
    kernel_dst3_3d_preOp_z<<<batch, std::min(Nz_ / 2 + 1, 256), 0, stream_>>>(
        d_work_, Nx_, Ny_, Nz_, dct2_sin_z_, dct2_cos_z_);
    CUDA_CHECK_LAUNCH();

    // D2Z FFT
    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_z_, d_work_, d_complex));

    // PostOp - 1 block per XY point, with shared memory
    kernel_dst3_3d_postOp_z<<<batch, std::min(Nz_ / 2 + 1, 256), sizeof(double) * (Nz_ + 2), stream_>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_z<<<numBlocks_copy, blockSize, 0, stream_>>>((double*)d_complex, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

//------------------------------------------------------------------------------
// 3D Y dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeY_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_y = Ny_ / 2 + 1;
    int total_preOp = Nx_ * Nz_ * half_y;
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dct2_3d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_y_, d_complex, d_work_));
    if (!dct2_sin_y_.empty() && !dct2_cos_y_.empty())
        kernel_dct2_3d_postOp_y_lut<<<numBlocks_postOp, blockSize, 0, stream_>>>(
            d_work_, d_temp_, Nx_, Ny_, Nz_, dct2_sin_y_, dct2_cos_y_);
    else
        kernel_dct2_3d_postOp_y<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeY_DCT3(double* d_data)
{
    int blockSize = 256;
    int numBlocks_M = grid_blocks(M_, blockSize);
    int total_preOp = Nx_ * Nz_ * (Ny_ / 2);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_transpose = grid_blocks(M_, blockSize);

    // Transpose (x,y,z) -> (x,z,y) into d_work_ for contiguous Y FFT
    kernel_dct3_3d_preOp_y_transpose<<<numBlocks_transpose, blockSize, 0, stream_>>>(
        d_temp_, d_work_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    if (!dct3_sin_y_.empty() && !dct3_cos_y_.empty())
        kernel_dct3_3d_preOp_y_lut<<<numBlocks_preOp, blockSize, 0, stream_>>>(
            d_work_, Nx_, Ny_, Nz_, dct3_sin_y_, dct3_cos_y_);
    else
        kernel_dct3_3d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_work_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_y_, d_work_, d_complex));

    kernel_dct3_3d_postOp_y<<<numBlocks_M, blockSize, 0, stream_>>>(d_complex, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeY_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp (to complex) -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * Nz_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dst2_3d_preOp_y<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_y_, d_complex, d_work_));
    kernel_dst2_3d_postOp_y<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_temp_, Nx_, Ny_, Nz_, dct2_sin_y_, dct2_cos_y_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeY_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Nx_ * Nz_;
    int stride = Ny_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = grid_blocks(total_setup, blockSize);
    int numBlocks_copy = grid_blocks(M_, blockSize);

    // Setup padded buffer: [0, x0, x1, ..., x_{Ny-1}, 0] for each XZ point
    kernel_dst3_3d_setup_y<<<numBlocks_setup, blockSize, 0, stream_>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // PreOp twiddle factors - 1 block per XZ point, Ny/2+1 threads per block
    kernel_dst3_3d_preOp_y<<<batch, std::min(Ny_ / 2 + 1, 256), 0, stream_>>>(
        d_work_, Nx_, Ny_, Nz_, dct2_sin_y_, dct2_cos_y_);
    CUDA_CHECK_LAUNCH();

    // D2Z FFT
    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_y_, d_work_, d_complex));

    // PostOp - 1 block per XZ point, with shared memory
    kernel_dst3_3d_postOp_y<<<batch, std::min(Ny_ / 2 + 1, 256), sizeof(double) * (Ny_ + 2), stream_>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_y<<<numBlocks_copy, blockSize, 0, stream_>>>((double*)d_complex, d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

//------------------------------------------------------------------------------
// 3D X dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeX_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_x = Nx_ / 2 + 1;
    int total_preOp = Ny_ * Nz_ * half_x;
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dct2_3d_preOp_x<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_x_, d_complex, d_work_));
    if (!dct2_sin_x_.empty() && !dct2_cos_x_.empty())
        kernel_dct2_3d_postOp_x_lut<<<numBlocks_postOp, blockSize, 0, stream_>>>(
            d_work_, d_data, Nx_, Ny_, Nz_, dct2_sin_x_, dct2_cos_x_);
    else
        kernel_dct2_3d_postOp_x<<<numBlocks_postOp, blockSize, 0, stream_>>>(d_work_, d_data, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeX_DCT3(double* d_data)
{
    int blockSize = 256;
    int numBlocks_M = grid_blocks(M_, blockSize);
    int total_preOp = Ny_ * Nz_ * (Nx_ / 2);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);

    if (!dct3_sin_x_.empty() && !dct3_cos_x_.empty())
        kernel_dct3_3d_preOp_x_strided_lut<<<numBlocks_preOp, blockSize, 0, stream_>>>(
            d_temp_, Nx_, Ny_, Nz_, dct3_sin_x_, dct3_cos_x_);
    else
        kernel_dct3_3d_preOp_x_strided<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_x_, d_temp_, d_complex));

    kernel_dct3_3d_postOp_x<<<numBlocks_M, blockSize, 0, stream_>>>(d_complex, d_data, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeX_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp (to complex) -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Ny_ * Nz_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = grid_blocks(total_preOp, blockSize);
    int numBlocks_postOp = grid_blocks(M_, blockSize);

    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());

    kernel_dst2_3d_preOp_x<<<numBlocks_preOp, blockSize, 0, stream_>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
    CUFFT_CHECK(cufftExecZ2D(plan_x_, d_complex, d_work_));
    kernel_dst2_3d_postOp_x<<<numBlocks_postOp, blockSize, 0, stream_>>>(
        d_work_, d_data, Nx_, Ny_, Nz_, dct2_sin_x_, dct2_cos_x_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::executeX_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Ny_ * Nz_;
    int stride = Nx_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = grid_blocks(total_setup, blockSize);
    int numBlocks_copy = grid_blocks(M_, blockSize);

    // Setup padded buffer: [0, x0, x1, ..., x_{Nx-1}, 0] for each YZ point
    kernel_dst3_3d_setup_x<<<numBlocks_setup, blockSize, 0, stream_>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // PreOp twiddle factors - 1 block per YZ point, Nx/2+1 threads per block
    kernel_dst3_3d_preOp_x<<<batch, std::min(Nx_ / 2 + 1, 256), 0, stream_>>>(
        d_work_, Nx_, Ny_, Nz_, dct2_sin_x_, dct2_cos_x_);
    CUDA_CHECK_LAUNCH();

    // D2Z FFT
    cufftDoubleComplex* d_complex = reinterpret_cast<cufftDoubleComplex*>(d_x1_.get());
    CUFFT_CHECK(cufftExecD2Z(plan_x_, d_work_, d_complex));

    // PostOp - 1 block per YZ point, with shared memory
    kernel_dst3_3d_postOp_x<<<batch, std::min(Nx_ / 2 + 1, 256), sizeof(double) * (Nx_ + 2), stream_>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_x<<<numBlocks_copy, blockSize, 0, stream_>>>((double*)d_complex, d_data, Nx_, Ny_, Nz_);
    CUDA_CHECK_LAUNCH();
}

void CudaRealTransform3D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform3D not initialized");
    }

    // Z dimension transform
    switch (type_z_) {
        case CUDA_DCT_2: executeZ_DCT2(d_data); break;
        case CUDA_DCT_3: executeZ_DCT3(d_data); break;
        case CUDA_DST_2: executeZ_DST2(d_data); break;
        case CUDA_DST_3: executeZ_DST3(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Z dimension");
    }

    // Y dimension transform
    switch (type_y_) {
        case CUDA_DCT_2: executeY_DCT2(d_data); break;
        case CUDA_DCT_3: executeY_DCT3(d_data); break;
        case CUDA_DST_2: executeY_DST2(d_data); break;
        case CUDA_DST_3: executeY_DST3(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Y dimension");
    }

    // X dimension transform
    switch (type_x_) {
        case CUDA_DCT_2: executeX_DCT2(d_data); break;
        case CUDA_DCT_3: executeX_DCT3(d_data); break;
        case CUDA_DST_2: executeX_DST2(d_data); break;
        case CUDA_DST_3: executeX_DST3(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for X dimension");
    }
}

void CudaRealTransform3D::get_dims(int& nx, int& ny, int& nz) const
{
    nx = Nx_;
    ny = Ny_;
    nz = Nz_;
}

void CudaRealTransform3D::get_types(CudaTransformType& type_x,
                                     CudaTransformType& type_y,
                                     CudaTransformType& type_z) const
{
    type_x = type_x_;
    type_y = type_y_;
    type_z = type_z_;
}

double CudaRealTransform3D::get_normalization() const
{
    return getNormFactor(type_x_, Nx_) * getNormFactor(type_y_, Ny_) * getNormFactor(type_z_, Nz_);
}
