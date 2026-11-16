#ifndef POLYMERFTS_JUNCTION_PRODUCT_H_
#define POLYMERFTS_JUNCTION_PRODUCT_H_

#include <cmath>
#include <complex>
#include <type_traits>

#ifdef __CUDACC__
#define POLYMERFTS_JUNCTION_HD __host__ __device__
#else
#define POLYMERFTS_JUNCTION_HD
#endif

/// Largest block multiplicity (Polymer). The extended-range junction products below then
/// keep exact binary exponents in a double, and complex fields take every integer
/// multiplicity (junction_product::integer_power).
constexpr double MAX_BLOCK_MULTIPLICITY = 1.0e6;

// A junction is one product: in a star with n arms in a strong field q_A^(n-1) can
// overflow while the product with the other arms is finite. An accumulator keeps the
// ordinary arithmetic while its factors stay in an ordinary range and continues with
// a mantissa and a binary exponent otherwise. A block multiplicity is at most
// MAX_BLOCK_MULTIPLICITY = 10^6, so the exponents stay far below 2^53 and a double
// holds them exactly.
namespace junction_product
{
POLYMERFTS_JUNCTION_HD inline bool finite(double value)
{
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

/// mantissa * 2^exponent with |mantissa| in [0.5, 1), zero or not finite
struct Number
{
    double mantissa;
    double exponent;
};

POLYMERFTS_JUNCTION_HD inline Number split(double value)
{
    int exponent = 0;
    const double mantissa = frexp(value, &exponent);
    return {mantissa, static_cast<double>(exponent)};
}

POLYMERFTS_JUNCTION_HD inline Number multiply(Number a, Number b)
{
    Number result = split(a.mantissa * b.mantissa);
    result.exponent += a.exponent + b.exponent;
    return result;
}

/// mantissa * 2^exponent rounded to a double: beyond 2^±2200 every mantissa
/// overflows or underflows, so the exponent is clamped before it becomes an int.
POLYMERFTS_JUNCTION_HD inline double scale(double mantissa, double exponent)
{
    return scalbn(mantissa, static_cast<int>(fmin(fmax(exponent, -2200.0), 2200.0)));
}

/// True if a junction power is an integer in [0, MAX_BLOCK_MULTIPLICITY]: an integer block
/// multiplicity takes the same operation as explicit arms. Other powers (a non-integer
/// multiplicity, or identical copied branches whose powers sum beyond the limit) need real
/// propagators: a computation with complex fields refuses them when it is built (ComputationCommon).
POLYMERFTS_JUNCTION_HD inline bool integer_power(double power)
{
    return power >= 0.0 && power <= MAX_BLOCK_MULTIPLICITY && power == floor(power);
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
    if (base == 0.0 || !finite(base) || !finite(exponent)
        || (base < 0.0 && exponent != floor(exponent)))
        return split(pow(base, exponent));
    // Separate the integer exponent before exp2: pow(base, exponent) could overflow first.
    const double binary = log2(fabs(base)) * exponent;
    const double whole = floor(binary);
    Number result = split(exp2(binary - whole));
    result.exponent += whole;
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
    if (exponent < 0.0 || !finite(exponent))
        return false;
    if (ordinary(source) && exponent <= 4.0)
        return true;
    if (source == 0.0)
        return true;
    // Only unusual multiplicities need a logarithm to certify the old path.
    // log2(source) is evaluated as a whole: splitting log2(mantissa)+exponent
    // would lose the small logarithm of a source close to one.
    return finite(source) && fabs(log2(fabs(source)) * exponent) <= 256.0;
}

/// Product of real powers. Once scaled, current is the mantissa of the product.
struct Real
{
    double current = 1.0;
    double exponent = 0.0;
    bool scaled = false;

    POLYMERFTS_JUNCTION_HD void multiply(double source, double power)
    {
        if (power == 0.0)
            return;
        // Up to four ordinary factors and the previous ordinary product stay
        // far from either end of the double range. Retain the original order.
        if (!scaled && ordinary(current) && ordinary_power(source, power))
        {
#ifdef __CUDA_ARCH__
            // An integer power of at most 4 as successive multiplications (as the complex
            // product), every other power by pow: a multiplicity of 10^6 costs one call
            if (power <= 4.0 && integer_power(power))
                for (int n = 0; n < static_cast<int>(power); ++n)
                    current *= source;
            else
                current *= pow(source, power);
#else
            current *= std::pow(source, power);
#endif
            return;
        }
        const Number product = junction_product::multiply(
            scaled ? Number{current, exponent} : split(current), junction_product::power(source, power));
        current = product.mantissa;
        exponent = product.exponent;
        scaled = true;
    }

    POLYMERFTS_JUNCTION_HD double value() const
    {
        return scaled && current != 0.0 && finite(current) ? scale(current, exponent) : current;
    }
};

/// (real + i imag) * 2^exponent with one exponent for both parts: max(|real|, |imag|)
/// lies in [0.5, 1) unless both are zero or one is not finite. A part smaller than
/// 2^-1074 of the other is lost, as in any complex product of doubles.
struct ComplexNumber
{
    double real;
    double imag;
    double exponent;
};

POLYMERFTS_JUNCTION_HD inline ComplexNumber normalize(double real, double imag, double exponent)
{
    const double magnitude = fmax(fabs(real), fabs(imag));
    if (magnitude == 0.0 || !finite(magnitude))
        return {real, imag, exponent};
    int shift = 0;
    frexp(magnitude, &shift);
    return {scalbn(real, -shift), scalbn(imag, -shift), exponent + shift};
}

POLYMERFTS_JUNCTION_HD inline ComplexNumber multiply(ComplexNumber a, ComplexNumber b)
{
    return normalize(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real,
                     a.exponent + b.exponent);
}

POLYMERFTS_JUNCTION_HD inline ComplexNumber power(ComplexNumber base, unsigned int exponent)
{
    ComplexNumber result = {1.0, 0.0, 0.0};
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

/// The real and imaginary parts of a complex junction product.
struct ComplexValue
{
    double real;
    double imag;
};

/// Product of integer powers of complex numbers (the caller checks integer_power).
/// Once scaled, (current_real, current_imag) is the mantissa of the product.
struct Complex
{
    double current_real = 1.0;
    double current_imag = 0.0;
    double exponent = 0.0;
    bool scaled = false;

    POLYMERFTS_JUNCTION_HD void multiply(double real, double imag, int power)
    {
        if (power == 0)
            return;
        if (!scaled && ordinary(current_real) && ordinary(current_imag)
            && ordinary(real) && ordinary(imag) && power >= 0 && power <= 4)
        {
#ifdef __CUDA_ARCH__
            for (int n = 0; n < power; ++n)
            {
                const double next_real = current_real * real - current_imag * imag;
                current_imag = current_real * imag + current_imag * real;
                current_real = next_real;
            }
#else
            const std::complex<double> next = std::complex<double>(current_real, current_imag)
                                           * std::pow(std::complex<double>(real, imag), power);
            current_real = next.real();
            current_imag = next.imag();
#endif
            return;
        }
        const ComplexNumber product = junction_product::multiply(
            scaled ? ComplexNumber{current_real, current_imag, exponent} : normalize(current_real, current_imag, 0.0),
            junction_product::power(normalize(real, imag, 0.0), static_cast<unsigned int>(power)));
        current_real = product.real;
        current_imag = product.imag;
        exponent = product.exponent;
        scaled = true;
    }

    POLYMERFTS_JUNCTION_HD ComplexValue value() const
    {
        if (!scaled)
            return {current_real, current_imag};
        return {scale(current_real, exponent), scale(current_imag, exponent)};
    }
};

template <typename T>
using Accumulator = std::conditional_t<std::is_same_v<T, double>, Real, Complex>;
} // namespace junction_product

#undef POLYMERFTS_JUNCTION_HD
#endif
