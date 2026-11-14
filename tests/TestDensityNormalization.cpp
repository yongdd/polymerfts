/** Density normalization preserves finite results across scalar/partition ranges. */
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "DensityNormalization.h"

void check(double actual, double expected)
{
    if (!std::isfinite(actual) || (expected == 0.0 ? actual != 0.0
            : actual == 0.0 || std::abs(actual / expected - 1.0) > 2e-13))
        throw std::runtime_error("Density normalization lost a finite component.");
}

void check(std::complex<double> actual, std::complex<double> expected)
{
    check(actual.real(), expected.real());
    check(actual.imag(), expected.imag());
}

int main()
{
    try
    {
        const double tiny_q = std::exp(-720.0);
        check(apply_density_normalization(tiny_q,
              make_density_normalization(1.0, 0.5, 1.0, 1.0, tiny_q)), 0.5);
        const double q = std::exp(-708.0);
        check(apply_density_normalization(q,
              make_density_normalization(1.0, 1e308, 3.0, 1.0, 1.0)), 3.0 * (1e308 * q));
        // A short contour step and small population multiply below the double
        // range, but dividing by the chain length restores a finite population.
        check(apply_density_normalization(1.0,
              make_density_normalization(1e-200, 1e-200, 1.0, 1e-200, 1.0)), 1e-200);
        using Complex = std::complex<double>;
        const Complex phase(0.8, 0.6);
        const Complex complex_q = tiny_q * phase;
        check(apply_density_normalization(complex_q,
              make_density_normalization(1.0, 0.5, 1.0, 1.0, complex_q)), Complex(0.5, 0.0));
        check(apply_density_normalization(Complex(1e150, 1e-200),
              make_density_normalization(1.0, 1e100, 1.0, 1.0, Complex(1e100, 0.0))),
              Complex(1e150, 1e-200));

        const double nan = std::numeric_limits<double>::quiet_NaN();
        check(apply_density_normalization(nan,
              make_density_normalization(1.0, 0.0, 3.0, 1.0, 0.0)), 0.0);
        check(apply_density_normalization(Complex(nan, nan),
              make_density_normalization(1.0, 0.0, 3.0, 1.0, Complex(0.0, 0.0))), Complex(0.0));
        // Occupied species remain singular at Q=0; do not replace it by epsilon.
        for (double density : {0.0, 1.0})
        {
            const double singular = apply_density_normalization(density,
                make_density_normalization(1.0, 0.5, 1.0, 1.0, 0.0));
            if (std::isfinite(singular))
                throw std::runtime_error("An occupied zero partition was regularized.");
        }
        std::cout << "Finite density normalization: canonical, GCE, complex and zero population passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
