#ifndef POLYMERFTS_CUDA_SCALED_PRODUCT_CUH_
#define POLYMERFTS_CUDA_SCALED_PRODUCT_CUH_

#include <cuComplex.h>
#include "ScaledProduct.h"
#include "DensityAccumulator.h"

namespace scaled_product
{
__device__ inline cuDoubleComplex divide(cuDoubleComplex a, cuDoubleComplex b,
                                         cuDoubleComplex divisor, double weight = 1.0)
{
    if ((divisor.x == 0.0 && divisor.y == 0.0) || !scaled_product::finite(a.x) || !scaled_product::finite(a.y)
        || !scaled_product::finite(b.x) || !scaled_product::finite(b.y) || !scaled_product::finite(divisor.x) || !scaled_product::finite(divisor.y) || !scaled_product::finite(weight)
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
    if ((divisor.x == 0.0 && divisor.y == 0.0) || !scaled_product::finite(a.x) || !scaled_product::finite(a.y)
        || !scaled_product::finite(b.x) || !scaled_product::finite(b.y) || !scaled_product::finite(divisor.x) || !scaled_product::finite(divisor.y))
    {
        const cuDoubleComplex multiplier = cuCdiv(
            make_cuDoubleComplex(scalbn(weight.mantissa, weight.exponent), 0.0), divisor);
        return cuCmul(cuCmul(a, b), multiplier);
    }
    const Complex result = divide_finite(a.x, a.y, b.x, b.y, divisor.x, divisor.y, weight);
    return make_cuDoubleComplex(result.real, result.imag);
}
} // namespace scaled_product

namespace density_accumulation
{
__device__ inline void accumulate(cuDoubleComplex& value, cuDoubleComplex& exponent,
                                  cuDoubleComplex a, cuDoubleComplex b, double weight,
                                  const cuDoubleComplex* divisor)
{
    using namespace scaled_product;
    const cuDoubleComplex w = divisor == nullptr ? make_cuDoubleComplex(1.0, 0.0) : *divisor;
    const bool regular = exponent.x == 0.0 && exponent.y == 0.0
        && ordinary(value.x) && ordinary(value.y) && ordinary(a.x) && ordinary(a.y)
        && ordinary(b.x) && ordinary(b.y) && ordinary(w.x) && ordinary(w.y) && ordinary(weight);
    const bool invalid = (w.x == 0.0 && w.y == 0.0) || !scaled_product::finite(a.x) || !scaled_product::finite(a.y)
        || !scaled_product::finite(b.x) || !scaled_product::finite(b.y) || !scaled_product::finite(w.x) || !scaled_product::finite(w.y) || !scaled_product::finite(weight);
    if (regular || invalid)
    {
        cuDoubleComplex term;
        if (divisor == nullptr)
        {
            // Keep the existing continuous CUDA product's operation order.
            term.x = weight * (a.x * b.x - a.y * b.y);
            term.y = weight * (a.x * b.y + a.y * b.x);
        }
        else
            term = scaled_product::divide(a, b, w, weight);
        if (exponent.x == 0.0 && exponent.y == 0.0)
        {
            value.x += term.x;
            value.y += term.y;
        }
        else
        {
            add(value.x, exponent.x, split(term.x));
            add(value.y, exponent.y, split(term.y));
        }
        return;
    }
    const ComplexNumber term = product(a.x, a.y, b.x, b.y, w.x, w.y, weight);
    add(value.x, exponent.x, term.real);
    add(value.y, exponent.y, term.imag);
}
} // namespace density_accumulation

#endif
