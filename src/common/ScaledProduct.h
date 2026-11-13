#ifndef POLYMERFTS_SCALED_PRODUCT_H_
#define POLYMERFTS_SCALED_PRODUCT_H_

#include <cmath>
#include <complex>

#ifdef __CUDACC__
#define POLYMERFTS_PRODUCT_HD __host__ __device__
#else
#define POLYMERFTS_PRODUCT_HD
#endif

// A shared bead occurs in both propagators of a discrete-chain contour cut.
// Keep exponents separate until weight*a*b/divisor is complete: a*b itself
// need not be representable even when every input and the result are finite.
namespace scaled_product
{
struct Number
{
    double mantissa;
    int exponent;
};

POLYMERFTS_PRODUCT_HD inline Number split(double value)
{
    int exponent = 0;
    const double mantissa = frexp(value, &exponent);
    return {mantissa, exponent};
}

POLYMERFTS_PRODUCT_HD inline Number multiply(Number a, Number b)
{
    Number result = split(a.mantissa * b.mantissa);
    result.exponent += a.exponent + b.exponent;
    return result;
}

POLYMERFTS_PRODUCT_HD inline Number add(Number a, Number b)
{
    if (a.mantissa == 0.0)
        return b;
    if (b.mantissa == 0.0)
        return a;
    const int exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    Number result = split(scalbn(a.mantissa, a.exponent - exponent)
                        + scalbn(b.mantissa, b.exponent - exponent));
    result.exponent += exponent;
    return result;
}

POLYMERFTS_PRODUCT_HD inline Number negate(Number a)
{
    a.mantissa = -a.mantissa;
    return a;
}

POLYMERFTS_PRODUCT_HD inline double quotient(Number numerator, Number denominator)
{
    return scalbn(numerator.mantissa / denominator.mantissa,
                  numerator.exponent - denominator.exponent);
}

POLYMERFTS_PRODUCT_HD inline bool finite(double value)
{
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// In this range every intermediate of the ordinary weighted complex product
// and quotient is normal (apart from exact cancellation). Retain the cheap
// arithmetic for ordinary fields, and scale only the wider-range cases.
POLYMERFTS_PRODUCT_HD inline bool ordinary(double value)
{
    const double magnitude = fabs(value);
    return magnitude == 0.0 || (magnitude >= 0x1p-200 && magnitude <= 0x1p200);
}

POLYMERFTS_PRODUCT_HD inline double divide(double a, double b, double divisor, double weight = 1.0)
{
    if (divisor == 0.0 || !finite(a) || !finite(b) || !finite(divisor) || !finite(weight)
        || (ordinary(a) && ordinary(b) && ordinary(divisor) && ordinary(weight)))
        return weight * a * b / divisor;
    return quotient(multiply(multiply(split(weight), split(a)), split(b)), split(divisor));
}

struct Complex
{
    double real;
    double imag;
};

// Finite operands and a nonzero complex divisor only. Each real/imaginary
// product has its own exponent, so a small component is not discarded just
// because the other component of the same complex input is large.
POLYMERFTS_PRODUCT_HD inline Complex divide_finite(
    double ar, double ai, double br, double bi, double wr, double wi, double weight)
{
    const Number a_real = split(ar), a_imag = split(ai);
    const Number b_real = split(br), b_imag = split(bi);
    const Number w_real = split(wr), w_imag = split(wi);
    const Number product_real = add(multiply(a_real, b_real), negate(multiply(a_imag, b_imag)));
    const Number product_imag = add(multiply(a_real, b_imag), multiply(a_imag, b_real));
    const Number norm = add(multiply(w_real, w_real), multiply(w_imag, w_imag));
    const Number real_numerator = add(multiply(product_real, w_real), multiply(product_imag, w_imag));
    const Number imag_numerator = add(multiply(product_imag, w_real), negate(multiply(product_real, w_imag)));
    const Number factor = split(weight);
    return {quotient(multiply(real_numerator, factor), norm),
            quotient(multiply(imag_numerator, factor), norm)};
}

inline std::complex<double> divide(std::complex<double> a, std::complex<double> b,
                                    std::complex<double> divisor, double weight = 1.0)
{
    if (divisor == std::complex<double>(0.0) || !finite(a.real()) || !finite(a.imag())
        || !finite(b.real()) || !finite(b.imag()) || !finite(divisor.real()) || !finite(divisor.imag())
        || !finite(weight)
        || (ordinary(a.real()) && ordinary(a.imag()) && ordinary(b.real()) && ordinary(b.imag())
            && ordinary(divisor.real()) && ordinary(divisor.imag()) && ordinary(weight)))
        return weight * a * b / divisor;
    const Complex result = divide_finite(a.real(), a.imag(), b.real(), b.imag(),
                                         divisor.real(), divisor.imag(), weight);
    return {result.real, result.imag};
}
} // namespace scaled_product

#undef POLYMERFTS_PRODUCT_HD
#endif
