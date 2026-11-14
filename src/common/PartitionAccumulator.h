#ifndef POLYMERFTS_PARTITION_ACCUMULATOR_H_
#define POLYMERFTS_PARTITION_ACCUMULATOR_H_

#include "DensityAccumulator.h"
#include <type_traits>

#ifdef __CUDACC__
#define POLYMERFTS_PARTITION_HD __host__ __device__
#else
#define POLYMERFTS_PARTITION_HD
#endif

namespace partition_accumulation
{
// A normalized partition can be finite when its unnormalized integral is not.
// Keep both components scaled through the grid reduction, including signed
// sums, and restore their exponents only after volume and copy normalization.
struct Sum
{
    scaled_product::Number real{0.0, 0};
    scaled_product::Number imag{0.0, 0};
};

struct Add
{
    POLYMERFTS_PARTITION_HD Sum operator()(Sum a, Sum b) const
    {
        return {scaled_product::add(a.real, b.real), scaled_product::add(a.imag, b.imag)};
    }
};

POLYMERFTS_PARTITION_HD inline Sum product(
    double ar, double ai, double br, double bi, double wr, double wi,
    double count, double total_grid, double copies, bool absolute)
{
    using namespace scaled_product;
    if (count == 0.0)
        return {};  // An excluded point must not evaluate an unused 0/0.
    const Number weight = ratio(ratio(split(count), split(total_grid)), split(copies));
    const auto value = density_accumulation::product(ar, ai, br, bi, wr, wi, 1.0);
    Sum term{multiply(value.real, weight), multiply(value.imag, weight)};
    if (absolute)
    {
        if (term.real.mantissa == 0.0 && term.imag.mantissa == 0.0)
            return {};
        const int exponent = term.real.mantissa == 0.0 ? term.imag.exponent
            : term.imag.mantissa == 0.0 ? term.real.exponent
            : term.real.exponent > term.imag.exponent ? term.real.exponent : term.imag.exponent;
        Number magnitude = split(hypot(scalbn(term.real.mantissa, term.real.exponent - exponent),
                                       scalbn(term.imag.mantissa, term.imag.exponent - exponent)));
        magnitude.exponent += exponent;
        return {magnitude, {0.0, 0}};
    }
    return term;
}

// A zero or small component can hide underflow; recompute it rather than
// mistaking it for a vanishing physical partition or imaginary component.
inline bool ordinary_result(double value)
{
    return value != 0.0 && scaled_product::ordinary(value);
}
inline bool ordinary_result(std::complex<double> value)
{
    return ordinary_result(value.real()) && ordinary_result(value.imag());
}

template <typename T>
T value(Sum sum)
{
    if constexpr (std::is_same_v<T, double>)
        return std::scalbn(sum.real.mantissa, sum.real.exponent);
    else
        return {std::scalbn(sum.real.mantissa, sum.real.exponent),
                std::scalbn(sum.imag.mantissa, sum.imag.exponent)};
}
} // namespace partition_accumulation

#undef POLYMERFTS_PARTITION_HD
#endif
