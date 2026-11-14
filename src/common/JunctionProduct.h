#ifndef POLYMERFTS_JUNCTION_PRODUCT_H_
#define POLYMERFTS_JUNCTION_PRODUCT_H_

#include <cmath>
#include <complex>
#include <type_traits>

#include "ScaledProduct.h"

#ifdef __CUDACC__
#define POLYMERFTS_JUNCTION_HD __host__ __device__
#else
#define POLYMERFTS_JUNCTION_HD
#endif

// A junction is one product, even when its sources arrive in separate GPU
// launches. Do not materialize a source power or a partial product whose range
// may be restored by a later source. Real and imaginary components have their
// own exponents: a common complex scale would discard a small component.
namespace junction_product
{
struct Number
{
    double mantissa;
    // Integer powers of two, stored as two doubles to avoid overflowing an int
    // or losing normalization carries for large block multiplicities.
    double exponent;
    double exponent_correction = 0.0;
};

POLYMERFTS_JUNCTION_HD inline Number split(double value)
{
    int exponent = 0;
    const double mantissa = frexp(value, &exponent);
    return {mantissa, static_cast<double>(exponent)};
}

POLYMERFTS_JUNCTION_HD inline void shift(Number& number, double exponent, double correction = 0.0)
{
    // TwoSum keeps the low part when adding a unit normalization carry to an
    // exponent above 2^53; opposite large exponents can then cancel exactly.
    const double sum = number.exponent + exponent;
    if (!scaled_product::finite(sum))
    {
        number.exponent = sum;
        number.exponent_correction = 0.0;
        return;
    }
    const double part = sum - number.exponent;
    const double error = (number.exponent - (sum - part)) + (exponent - part);
    const double remainder = (number.exponent_correction + correction) + error;
    number.exponent = sum + remainder;
    number.exponent_correction = remainder - (number.exponent - sum);
}

POLYMERFTS_JUNCTION_HD inline Number multiply(Number a, Number b)
{
    Number result = split(a.mantissa * b.mantissa);
    shift(result, a.exponent, a.exponent_correction);
    shift(result, b.exponent, b.exponent_correction);
    return result;
}

POLYMERFTS_JUNCTION_HD inline double value(Number number)
{
    if (number.mantissa == 0.0 || !scaled_product::finite(number.mantissa))
        return number.mantissa;
    const double exponent = number.exponent + number.exponent_correction;
    if (exponent > 1024.0)
        return copysign(HUGE_VAL, number.mantissa);
    if (exponent < -1074.0)
        return copysign(0.0, number.mantissa);
    if (!scaled_product::finite(exponent))
        return exponent; // indeterminate exponent: propagate NaN, never cast it
    return scalbn(number.mantissa, static_cast<int>(exponent));
}

POLYMERFTS_JUNCTION_HD inline double exponent_difference(Number a, Number b)
{
    shift(a, -b.exponent, -b.exponent_correction);
    return a.exponent + a.exponent_correction;
}

POLYMERFTS_JUNCTION_HD inline Number add(Number a, Number b)
{
    if (a.mantissa == 0.0)
        return b;
    if (b.mantissa == 0.0)
        return a;
    if (!scaled_product::finite(a.exponent) || !scaled_product::finite(b.exponent))
        return split(value(a) + value(b));
    const Number reference = a.exponent > b.exponent
        || (a.exponent == b.exponent && a.exponent_correction > b.exponent_correction) ? a : b;
    const double da = exponent_difference(a, reference), db = exponent_difference(b, reference);
    Number result = split((da < -1074.0 ? 0.0 : scalbn(a.mantissa, static_cast<int>(da)))
                        + (db < -1074.0 ? 0.0 : scalbn(b.mantissa, static_cast<int>(db))));
    shift(result, reference.exponent, reference.exponent_correction);
    return result;
}

POLYMERFTS_JUNCTION_HD inline Number negate(Number a)
{
    a.mantissa = -a.mantissa;
    return a;
}

POLYMERFTS_JUNCTION_HD inline bool integer_power(double power)
{
    return power >= 0.0 && power <= 1.0e6 && power == floor(power);
}

POLYMERFTS_JUNCTION_HD inline Number power(Number base, unsigned int exponent)
{
    Number result = split(1.0);
    while (exponent != 0)
    {
        if (exponent & 1U)
            result = multiply(result, base);
        exponent >>= 1U;
        if (exponent != 0)
            base = multiply(base, base);
    }
    return result;
}

POLYMERFTS_JUNCTION_HD inline Number power(double base, double exponent)
{
    if (integer_power(exponent))
        return power(split(base), static_cast<unsigned int>(exponent));
    if (base == 0.0 || !scaled_product::finite(base) || !scaled_product::finite(exponent)
        || (base < 0.0 && exponent != floor(exponent)))
        return split(pow(base, exponent));
    // A real multiplicity need not be small or integral. Separate the integer
    // exponent before exp2; forming pow(base, exponent) could overflow first.
    const double binary = log2(fabs(base)) * exponent;
    if (!scaled_product::finite(binary))
        return split(pow(base, exponent));
    const double whole = floor(binary);
    Number result = split(exp2(binary - whole));
    shift(result, whole);
    if (base < 0.0 && fmod(exponent, 2.0) != 0.0)
        result.mantissa = -result.mantissa;
    return result;
}

POLYMERFTS_JUNCTION_HD inline bool ordinary(double component)
{
    const double magnitude = fabs(component);
    return magnitude == 0.0 || (magnitude >= 0x1p-64 && magnitude <= 0x1p64);
}

POLYMERFTS_JUNCTION_HD inline bool ordinary_power(double source, double exponent)
{
    if (exponent < 0.0 || !scaled_product::finite(exponent))
        return false;
    if (ordinary(source) && exponent <= 4.0)
        return true;
    if (source == 0.0)
        return true;
    // Only unusual multiplicities need a logarithm to certify the old path.
    // log2(source) is evaluated as a whole: splitting log2(mantissa)+exponent
    // would lose the small logarithm of a source close to one.
    return scaled_product::finite(source) && fabs(log2(fabs(source)) * exponent) <= 256.0;
}

struct Real
{
    double current = 1.0;
    Number product = {0.5, 1.0};
    bool scaled = false;

    POLYMERFTS_JUNCTION_HD void multiply(double source, double exponent)
    {
        if (exponent == 0.0)
            return;
        // Up to four ordinary factors and the previous ordinary product stay
        // far from either end of the double range. Retain the original order.
        if (!scaled && ordinary(current) && ordinary_power(source, exponent))
        {
#ifdef __CUDA_ARCH__
            if (integer_power(exponent))
                for (int n = 0; n < static_cast<int>(exponent); ++n)
                    current *= source;
            else
                current *= pow(source, exponent);
#else
            current *= std::pow(source, exponent);
#endif
            return;
        }
        if (!scaled)
        {
            product = split(current);
            scaled = true;
        }
        product = junction_product::multiply(product, power(source, exponent));
    }

    POLYMERFTS_JUNCTION_HD double value() const
    {
        return scaled ? junction_product::value(product) : current;
    }
};

struct ComplexNumber
{
    Number real;
    Number imag;
};

POLYMERFTS_JUNCTION_HD inline ComplexNumber multiply(ComplexNumber a, ComplexNumber b)
{
    return {add(multiply(a.real, b.real), negate(multiply(a.imag, b.imag))),
            add(multiply(a.real, b.imag), multiply(a.imag, b.real))};
}

POLYMERFTS_JUNCTION_HD inline ComplexNumber power(ComplexNumber base, unsigned int exponent)
{
    ComplexNumber result = {split(1.0), split(0.0)};
    while (exponent != 0)
    {
        if (exponent & 1U)
            result = multiply(result, base);
        exponent >>= 1U;
        if (exponent != 0)
            base = multiply(base, base);
    }
    return result;
}

struct Complex
{
    double current_real = 1.0;
    double current_imag = 0.0;
    ComplexNumber product = {{0.5, 1.0}, {0.0, 0.0}};
    bool scaled = false;

    // The caller validates complex powers with is_integer_power, as before.
    POLYMERFTS_JUNCTION_HD void multiply(double real, double imag, int exponent)
    {
        if (exponent == 0)
            return;
        if (!scaled && ordinary(current_real) && ordinary(current_imag)
            && ordinary(real) && ordinary(imag) && exponent >= 0 && exponent <= 4)
        {
#ifdef __CUDA_ARCH__
            for (int n = 0; n < exponent; ++n)
            {
                const double next_real = current_real * real - current_imag * imag;
                current_imag = current_real * imag + current_imag * real;
                current_real = next_real;
            }
#else
            const std::complex<double> next = std::complex<double>(current_real, current_imag)
                                           * std::pow(std::complex<double>(real, imag), exponent);
            current_real = next.real();
            current_imag = next.imag();
#endif
            return;
        }
        if (!scaled)
        {
            product = {split(current_real), split(current_imag)};
            scaled = true;
        }
        product = junction_product::multiply(product,
            power(ComplexNumber{split(real), split(imag)}, static_cast<unsigned int>(exponent)));
    }

    POLYMERFTS_JUNCTION_HD scaled_product::Complex value() const
    {
        return scaled ? scaled_product::Complex{junction_product::value(product.real), junction_product::value(product.imag)}
                      : scaled_product::Complex{current_real, current_imag};
    }
};

template <typename T>
using Accumulator = std::conditional_t<std::is_same_v<T, double>, Real, Complex>;
} // namespace junction_product

#undef POLYMERFTS_JUNCTION_HD
#endif
