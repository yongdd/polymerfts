#ifndef POLYMERFTS_CUDA_JUNCTION_PRODUCT_CUH_
#define POLYMERFTS_CUDA_JUNCTION_PRODUCT_CUH_

#include <cuComplex.h>
#include "JunctionProduct.h"

namespace junction_product
{
/// Sources of a junction product in one launch (passed as kernel parameters)
constexpr int MAX_SOURCES_PER_LAUNCH = 32;

template <typename TD>
struct Sources
{
    const TD* q[MAX_SOURCES_PER_LAUNCH];
    double power[MAX_SOURCES_PER_LAUNCH];
    int n;
};

// The accumulator of a junction that takes several launches: its value lies in q
// and its binary exponent and scaled flag in a scratch field, which holds two
// doubles per point (state[i] and state[size + i], or the two parts of a complex).
__device__ inline void load(Real& product, const double* q, const double* state, int i, int size)
{
    product.current = q[i];
    product.exponent = state[i];
    product.scaled = state[size + i] != 0.0;
}

__device__ inline void store(const Real& product, double* q, double* state, int i, int size)
{
    q[i] = product.current;
    state[i] = product.exponent;
    state[size + i] = product.scaled ? 1.0 : 0.0;
}

__device__ inline void load(Complex& product, const cuDoubleComplex* q, const cuDoubleComplex* state, int i, int)
{
    product.current_real = q[i].x;
    product.current_imag = q[i].y;
    product.exponent = state[i].x;
    product.scaled = state[i].y != 0.0;
}

__device__ inline void store(const Complex& product, cuDoubleComplex* q, cuDoubleComplex* state, int i, int)
{
    q[i] = make_cuDoubleComplex(product.current_real, product.current_imag);
    state[i] = make_cuDoubleComplex(product.exponent, product.scaled ? 1.0 : 0.0);
}

__device__ inline void multiply(Real& product, double source, double power)
{
    product.multiply(source, power);
}

__device__ inline void multiply(Complex& product, cuDoubleComplex source, double power)
{
    // Complex powers are integers (integer_power, checked when the computation is built)
    product.multiply(source.x, source.y, static_cast<int>(power));
}

__device__ inline double result(const Real& product)
{
    return product.value();
}

__device__ inline cuDoubleComplex result(const Complex& product)
{
    const ComplexValue value = product.value();
    return make_cuDoubleComplex(value.real, value.imag);
}

/// q = Π_k q_k^power_k over the sources of one launch, in their order, with the
/// accumulator in registers. With first false the launch continues the accumulator
/// that the previous one left in (q, state); with last false it leaves it there.
template <typename TD>
__global__ void ker_junction_product(TD* q, Sources<TD> sources, TD* state, bool first, bool last, int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        std::conditional_t<std::is_same_v<TD, double>, Real, Complex> product;
        if (!first)
            load(product, q, state, i, size);
        for (int k = 0; k < sources.n; ++k)
            multiply(product, sources.q[k][i], sources.power[k]);
        if (last)
            q[i] = result(product);
        else
            store(product, q, state, i, size);
    }
}
} // namespace junction_product

#endif
