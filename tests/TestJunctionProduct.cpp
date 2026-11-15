/** A junction keeps the range of its entire product, including complex components. */
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "JunctionProduct.h"

namespace
{
void check(double actual, double expected, double tolerance = 2e-13)
{
    if (!std::isfinite(actual) || (expected == 0.0 ? actual != 0.0
            : actual == 0.0 || std::abs(actual / expected - 1.0) > tolerance))
        throw std::runtime_error("A junction product lost a finite component.");
}

void check(junction_product::Complex actual, std::complex<double> expected)
{
    const auto result = actual.value();
    check(result.real, expected.real());
    check(result.imag, expected.imag());
}
} // namespace

int main()
{
    try
    {
        const double large = std::exp(500.0), small = std::exp(-500.0);
        for (const double first : {large, small})
        {
            const double second = first == large ? small : large;
            junction_product::Real product;
            product.multiply(first, 2.0);
            product.multiply(second, 1.0);
            check(product.value(), first);
            product.multiply(second, 1.0);
            check(product.value(), 1.0);
        }
        junction_product::Real signed_product;
        signed_product.multiply(-large, 3.0);
        signed_product.multiply(small, 2.0);
        check(signed_product.value(), -large);

        // A real noninteger multiplicity follows pow's real domain, without
        // materializing either an overflowing or an underflowing source power.
        junction_product::Real fractional;
        fractional.multiply(large, 2.5);
        fractional.multiply(small, 1.5);
        check(fractional.value(), large, 4e-13);
        junction_product::Real invalid;
        invalid.multiply(-large, 2.5);
        if (!std::isnan(invalid.value()))
            throw std::runtime_error("A noninteger power of a negative real source lost its domain error.");

        junction_product::Real zero;
        zero.multiply(large, 2.0);
        zero.multiply(0.0, 0.5);
        zero.multiply(small, 1.0);
        check(zero.value(), 0.0);
        junction_product::Real unit;
        unit.multiply(0.0, 0.0);
        unit.multiply(std::numeric_limits<double>::quiet_NaN(), 0.0);
        check(unit.value(), 1.0);

        // Multiplicities above 10^6 take the real-power path. Their binary
        // exponents can exceed INT_MAX even though the final product is one.
        junction_product::Real large_power;
        large_power.multiply(0x1p300, 1e8);
        large_power.multiply(0x1p-300, 1e8);
        check(large_power.value(), 1.0);
        for (const double base : {1.0, 1.0 + 0x1p-40, 1.0 - 0x1p-40})
        {
            junction_product::Real near_one;
            near_one.multiply(base, 1e8 + 0.5);
            check(near_one.value(), std::pow(base, 1e8 + 0.5));
        }
        for (const double exponent : {1e20, 1e300})
        {
            junction_product::Real cancelling;
            cancelling.multiply(2.0, exponent);
            cancelling.multiply(0.5, exponent);
            check(cancelling.value(), 1.0);
            // Complex component addition aligns both exponent parts, including
            // a one-bit difference invisible in the high part alone.
            const auto full = junction_product::power(2.0, exponent);
            auto half = full;
            junction_product::shift(half, -1.0);
            const auto sum = junction_product::add(full, half);
            check(junction_product::value(junction_product::multiply(sum,
                  junction_product::power(0.5, exponent))), 1.5);
            check(junction_product::value(junction_product::add(full,
                  junction_product::negate(full))), 0.0);
        }
        for (const double exponent : {1e300, 1e308})
        {
            junction_product::Real unrepresentable;
            unrepresentable.multiply(1e300, exponent);
            if (!std::isinf(unrepresentable.value()))
                throw std::runtime_error("An unrepresentable final junction product was made finite.");
        }

        for (const double edge : {std::numeric_limits<double>::denorm_min(),
                                  std::numeric_limits<double>::max()})
        {
            junction_product::Real product;
            product.multiply(edge, 1.0);
            check(product.value(), edge);
        }

        junction_product::Complex independent;
        independent.multiply(1e150, 1e-200, 1);
        independent.multiply(1e200, 0.0, 1);
        independent.multiply(1e-200, 0.0, 1);
        check(independent, {1e150, 1e-200});
        junction_product::Complex phase;
        phase.multiply(0.0, large, 2);
        phase.multiply(small, 0.0, 1);
        check(phase, {-large, 0.0});
        phase.multiply(0.0, small, 1);
        check(phase, {0.0, -1.0});
        junction_product::Complex complex_zero;
        complex_zero.multiply(large, 0.0, 2);
        complex_zero.multiply(0.0, 0.0, 1);
        check(complex_zero, {0.0, 0.0});

        // The ordinary CPU path keeps its previous arithmetic exactly. The inputs are
        // volatile so that the reference is computed at run time, as the junction is:
        // GCC 13 and later fold the constant complex expression with another rounding.
        volatile double re_1 = 1.2, im_1 = 0.1, re_2 = -0.7, im_2 = 0.2;
        junction_product::Real ordinary;
        ordinary.multiply(re_1, 3.0);
        ordinary.multiply(re_2, 2.0);
        if (ordinary.value() != std::pow(re_1, 3.0) * std::pow(re_2, 2.0))
            throw std::runtime_error("The ordinary real junction arithmetic changed.");
        junction_product::Complex ordinary_complex;
        ordinary_complex.multiply(re_1, im_1, 3);
        ordinary_complex.multiply(re_2, im_2, 2);
        const auto result = ordinary_complex.value();
        // MSVC's complex constructor takes const references, which cannot bind volatile.
        const double r_1 = re_1, i_1 = im_1, r_2 = re_2, i_2 = im_2;
        const auto expected = std::pow(std::complex<double>(r_1, i_1), 3)
                            * std::pow(std::complex<double>(r_2, i_2), 2);
        if (result.real != expected.real() || result.imag != expected.imag())
            throw std::runtime_error("The ordinary complex junction arithmetic changed.");

        std::cout << "Junction products: finite range, independent complex components and domains passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
