#ifndef POLYMERFTS_LOG_FUGACITY_H_
#define POLYMERFTS_LOG_FUGACITY_H_

#include <cmath>
#include <limits>

#include "DensityNormalization.h"
#include "Exception.h"

namespace log_fugacity
{
inline void validate(double value)
{
    if (!std::isfinite(value) && value != -std::numeric_limits<double>::infinity())
        throw_with_line_number("log_fugacity must be finite or -inf (zero activity).");
}

// Store exp(mu) before combining it with a density or partition. Its binary
// exponent need not fit the range of a double. -inf denotes an absent species.
inline scaled_product::Number factor(double value)
{
    validate(value);
    if (value == -std::numeric_limits<double>::infinity())
        return {0.0, 0};
    constexpr double log_two = 0.693147180559945309417232121458176568;
    constexpr double log_two_tail = 2.319046813846299558417771099e-17;
    constexpr int exponent_limit = std::numeric_limits<int>::max() / 4;
    // Beyond this bound, multiplying by any finite double or a stored contour
    // integral can only overflow or underflow (or remain exactly zero). Keep
    // room for subsequent exponent arithmetic, without casting a huge double.
    if (value >= exponent_limit * log_two)
        return {0.5, exponent_limit};
    if (value <= -exponent_limit * log_two)
        return {0.5, -exponent_limit};
    const int exponent = static_cast<int>(std::floor(value / log_two));
    const double remainder = std::fma(-static_cast<double>(exponent), log_two, value)
                           - static_cast<double>(exponent) * log_two_tail;
    auto result = scaled_product::split(std::exp(remainder));
    result.exponent += exponent;
    return result;
}

template <typename T>
DensityNormalization<T> normalization(double value, double step = 1.0, double copies = 1.0)
{
    validate(value);
    // The usual chemical-potential range retains the existing coefficient
    // arithmetic. Wider values are exponentiated only after combining ranges.
    if (value >= -128.0 && value <= 128.0)
        return make_density_normalization(std::exp(value), step, copies, 1.0, T(1.0));
    const auto coefficient = scaled_product::multiply(
        scaled_product::multiply(factor(value), scaled_product::split(step)),
        scaled_product::split(copies));
    return {coefficient, T(1.0), T(0.0), coefficient.mantissa == 0.0};
}
} // namespace log_fugacity

#endif
