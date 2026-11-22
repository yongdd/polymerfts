#ifndef POLYMERFTS_CUDA_DEVICE_OPS_CUH_
#define POLYMERFTS_CUDA_DEVICE_OPS_CUH_

// Device helpers of the kernels that take real (double) and complex (cuDoubleComplex) fields
// in one spelling, and the gather through an index map. The gather kernel is a template defined
// here, so that every translation unit that launches it instantiates its own: since CUDA 13 the
// host stub of a __global__ template has internal linkage, and an instantiation in another file
// cannot be launched.

#include <cuComplex.h>

__device__ inline double zero_of(double) { return 0.0; }
__device__ inline cuDoubleComplex zero_of(cuDoubleComplex) { return make_cuDoubleComplex(0.0, 0.0); }

/// a · x
__device__ inline double scaled(double a, double x) { return a * x; }
__device__ inline cuDoubleComplex scaled(double a, cuDoubleComplex x) { return make_cuDoubleComplex(a * x.x, a * x.y); }

/// x + y
__device__ inline double added(double x, double y) { return x + y; }
__device__ inline cuDoubleComplex added(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x + y.x, x.y + y.y); }

/// x · y
__device__ inline double multiplied(double x, double y) { return x * y; }
__device__ inline cuDoubleComplex multiplied(cuDoubleComplex x, cuDoubleComplex y) { return cuCmul(x, y); }

/**
 * @brief dst[i] = src[index[i]] for i < n: a gather through an index map, such as the expansion of
 *        a reduced basis to the full grid, the reduction back and the map k -> -k of a spectrum.
 *
 * dst and src must not overlap (an in-place gather through a permutation would race).
 */
template <typename D>
__global__ void ker_gather(D* dst, const D* src, const int* index, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n)
    {
        dst[i] = src[index[i]];
        i += blockDim.x * gridDim.x;
    }
}

#endif  // POLYMERFTS_CUDA_DEVICE_OPS_CUH_
