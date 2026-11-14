#ifndef POLYMERFTS_DENSITY_NORMALIZATION_H_
#define POLYMERFTS_DENSITY_NORMALIZATION_H_

#include "ScaledProduct.h"

// Physical density = raw_density * factor / partition. Keep the real factor
// as a mantissa/exponent because population/Q or fugacity*copies can overflow
// even when the final density is finite. A cached multiplier retains the
// original inexpensive arithmetic when its inputs and the raw density are in
// the ordinary range.
template <typename T>
struct DensityNormalization
{
    scaled_product::Number factor;
    T partition;
    T multiplier;
    bool multiplier_available;
};

template <typename T>
DensityNormalization<T> make_density_normalization(
    double first, double second, double copies, double scalar_divisor, T partition)
{
    using namespace scaled_product;
    const Number factor = multiply(ratio(multiply(split(first), split(second)), split(scalar_divisor)),
                                   split(copies));
    DensityNormalization<T> result{factor, partition, T(0.0), false};
    if (factor.mantissa == 0.0)
    {
        result.multiplier_available = true;
        return result;
    }
    // Bound the inputs before materializing a coefficient. In particular,
    // never evaluate population/Q for subnormal Q just to test its result.
    if (ordinary(first) && ordinary(second) && ordinary(copies) && ordinary(scalar_divisor)
        && partition != T(0.0) && ordinary(std::real(partition)) && ordinary(std::imag(partition)))
    {
        const T multiplier = (first * second / scalar_divisor * copies) / partition;
        if (ordinary(std::real(multiplier)) && ordinary(std::imag(multiplier)))
        {
            result.multiplier = multiplier;
            result.multiplier_available = true;
        }
    }
    return result;
}

template <typename T>
T apply_density_normalization(T density, const DensityNormalization<T>& normalization)
{
    if (normalization.factor.mantissa == 0.0)
        return T(0.0);
    if (normalization.multiplier_available
        && scaled_product::ordinary(std::real(density)) && scaled_product::ordinary(std::imag(density)))
        return density * normalization.multiplier;
    return scaled_product::divide(density, T(1.0), normalization.partition, normalization.factor);
}

#endif
