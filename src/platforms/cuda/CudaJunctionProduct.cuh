#ifndef POLYMERFTS_CUDA_JUNCTION_PRODUCT_CUH_
#define POLYMERFTS_CUDA_JUNCTION_PRODUCT_CUH_

#include <cuComplex.h>
#include "JunctionProduct.h"

namespace junction_product
{
// State is allocated once per backend stream. Checkpoint sources can use the
// existing staging buffer; the state survives until the last source arrives.
static __global__ void accumulate(double* dst, Real* state, const double* src,
                                  double power, bool first, bool last, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        Real product = first ? Real{} : state[i];
        product.multiply(src[i], power);
        if (last)
            dst[i] = product.value();
        else
            state[i] = product;
    }
}

static __global__ void accumulate(cuDoubleComplex* dst, Complex* state, const cuDoubleComplex* src,
                                  int power, bool first, bool last, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        Complex product = first ? Complex{} : state[i];
        product.multiply(src[i].x, src[i].y, power);
        if (last)
        {
            const scaled_product::Complex result = product.value();
            dst[i] = make_cuDoubleComplex(result.real, result.imag);
        }
        else
            state[i] = product;
    }
}
} // namespace junction_product

#endif
