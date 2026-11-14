#ifndef POLYMERFTS_DENSITY_ACCUMULATOR_H_
#define POLYMERFTS_DENSITY_ACCUMULATOR_H_

#include "ScaledProduct.h"

#ifdef __CUDACC__
#define POLYMERFTS_DENSITY_HD __host__ __device__
#else
#define POLYMERFTS_DENSITY_HD
#endif

// A block integral can exceed the field type even though its normalized
// density is finite. Keep each component's exponent separately until the
// ensemble normalization. Exponent zero also represents ordinary, unscaled
// doubles, preserving the usual arithmetic without frexp/scalbn in that path.
namespace density_accumulation
{
using scaled_product::Number;

POLYMERFTS_DENSITY_HD inline Number stored(double value, double exponent)
{
    return exponent == 0.0 ? scaled_product::split(value)
                           : Number{value, static_cast<int>(exponent)};
}

POLYMERFTS_DENSITY_HD inline void add(double& value, double& exponent, Number term)
{
    const Number sum = scaled_product::add(stored(value, exponent), term);
    value = sum.mantissa;
    exponent = value == 0.0 ? 0.0 : static_cast<double>(sum.exponent);
}

POLYMERFTS_DENSITY_HD inline Number product(double a, double b, double divisor, double weight)
{
    using namespace scaled_product;
    return ratio(multiply(multiply(split(weight), split(a)), split(b)), split(divisor));
}

struct ComplexNumber
{
    Number real;
    Number imag;
};

POLYMERFTS_DENSITY_HD inline ComplexNumber product(
    double ar, double ai, double br, double bi, double wr, double wi, double weight)
{
    using namespace scaled_product;
    const Number a_real = split(ar), a_imag = split(ai);
    const Number b_real = split(br), b_imag = split(bi);
    const Number w_real = split(wr), w_imag = split(wi);
    const Number real = scaled_product::add(multiply(a_real, b_real), negate(multiply(a_imag, b_imag)));
    const Number imag = scaled_product::add(multiply(a_real, b_imag), multiply(a_imag, b_real));
    const Number norm = scaled_product::add(multiply(w_real, w_real), multiply(w_imag, w_imag));
    const Number factor = split(weight);
    return {ratio(multiply(scaled_product::add(multiply(real, w_real), multiply(imag, w_imag)), factor), norm),
            ratio(multiply(scaled_product::add(multiply(imag, w_real), negate(multiply(real, w_imag))), factor), norm)};
}

// Normalize components before restoring either exponent. In particular a
// tiny imaginary component must survive beside a very large real component.
POLYMERFTS_DENSITY_HD inline scaled_product::Complex normalize(
    Number real, Number imag, double wr, double wi, Number factor)
{
    using namespace scaled_product;
    const Number w_real = split(wr), w_imag = split(wi);
    const Number norm = scaled_product::add(multiply(w_real, w_real), multiply(w_imag, w_imag));
    return {quotient(multiply(scaled_product::add(multiply(real, w_real), multiply(imag, w_imag)), factor), norm),
            quotient(multiply(scaled_product::add(multiply(imag, w_real), negate(multiply(real, w_imag))), factor), norm)};
}

POLYMERFTS_DENSITY_HD inline void accumulate(
    double& value, double& exponent, double a, double b, double weight, double divisor = 1.0)
{
    using namespace scaled_product;
    if ((exponent == 0.0 && ordinary(value) && ordinary(a) && ordinary(b)
         && ordinary(weight) && ordinary(divisor))
        || divisor == 0.0 || !scaled_product::finite(a) || !scaled_product::finite(b) || !scaled_product::finite(weight) || !scaled_product::finite(divisor))
    {
        // Invalid inputs keep their ordinary IEEE result, without losing the
        // existing accumulator if it already has a separate exponent.
        const double term = weight * a * b / divisor;
        if (exponent == 0.0)
            value += term;
        else
            add(value, exponent, split(term));
        return;
    }
    add(value, exponent, product(a, b, divisor, weight));
}

inline void accumulate(std::complex<double>& value, std::complex<double>& exponent,
                       std::complex<double> a, std::complex<double> b, double weight,
                       const std::complex<double>* divisor)
{
    using namespace scaled_product;
    const std::complex<double> w = divisor == nullptr ? std::complex<double>(1.0) : *divisor;
    const bool regular = exponent == std::complex<double>(0.0)
        && ordinary(value.real()) && ordinary(value.imag())
        && ordinary(a.real()) && ordinary(a.imag()) && ordinary(b.real()) && ordinary(b.imag())
        && ordinary(w.real()) && ordinary(w.imag()) && ordinary(weight);
    const bool invalid = w == std::complex<double>(0.0) || !scaled_product::finite(a.real()) || !scaled_product::finite(a.imag())
        || !scaled_product::finite(b.real()) || !scaled_product::finite(b.imag()) || !scaled_product::finite(w.real()) || !scaled_product::finite(w.imag()) || !scaled_product::finite(weight);
    if (regular || invalid)
    {
        const std::complex<double> term = divisor == nullptr ? weight * a * b : weight * a * b / w;
        if (exponent == std::complex<double>(0.0))
            value += term;
        else
        {
            double vr = value.real(), vi = value.imag(), er = exponent.real(), ei = exponent.imag();
            add(vr, er, split(term.real()));
            add(vi, ei, split(term.imag()));
            value = {vr, vi};
            exponent = {er, ei};
        }
        return;
    }
    const ComplexNumber term = product(a.real(), a.imag(), b.real(), b.imag(), w.real(), w.imag(), weight);
    double vr = value.real(), vi = value.imag(), er = exponent.real(), ei = exponent.imag();
    add(vr, er, term.real);
    add(vi, ei, term.imag);
    value = {vr, vi};
    exponent = {er, ei};
}
} // namespace density_accumulation

#undef POLYMERFTS_DENSITY_HD
#endif
