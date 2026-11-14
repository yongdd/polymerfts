#ifndef POLYMERFTS_CUDA_SCALED_PRODUCT_CUH_
#define POLYMERFTS_CUDA_SCALED_PRODUCT_CUH_

#include <cuComplex.h>
#include "ScaledProduct.h"

namespace scaled_product
{
__device__ inline cuDoubleComplex divide(cuDoubleComplex a, cuDoubleComplex b,
                                         cuDoubleComplex divisor, double weight = 1.0)
{
    if ((divisor.x == 0.0 && divisor.y == 0.0) || !finite(a.x) || !finite(a.y)
        || !finite(b.x) || !finite(b.y) || !finite(divisor.x) || !finite(divisor.y) || !finite(weight)
        || (ordinary(a.x) && ordinary(a.y) && ordinary(b.x) && ordinary(b.y)
            && ordinary(divisor.x) && ordinary(divisor.y) && ordinary(weight)))
    {
        const cuDoubleComplex product = cuCmul(a, b);
        return cuCdiv(make_cuDoubleComplex(weight * product.x, weight * product.y), divisor);
    }
    const Complex result = divide_finite(a.x, a.y, b.x, b.y, divisor.x, divisor.y, weight);
    return make_cuDoubleComplex(result.real, result.imag);
}

__device__ inline cuDoubleComplex divide(cuDoubleComplex a, cuDoubleComplex b,
                                         cuDoubleComplex divisor, Number weight)
{
    if ((divisor.x == 0.0 && divisor.y == 0.0) || !finite(a.x) || !finite(a.y)
        || !finite(b.x) || !finite(b.y) || !finite(divisor.x) || !finite(divisor.y))
    {
        const cuDoubleComplex multiplier = cuCdiv(
            make_cuDoubleComplex(scalbn(weight.mantissa, weight.exponent), 0.0), divisor);
        return cuCmul(cuCmul(a, b), multiplier);
    }
    const Complex result = divide_finite(a.x, a.y, b.x, b.y, divisor.x, divisor.y, weight);
    return make_cuDoubleComplex(result.real, result.imag);
}
} // namespace scaled_product

#endif
