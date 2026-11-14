/** Log activities are combined with finite densities before restoring range. */
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "LogFugacity.h"

namespace
{
void check(double value, double expected, double tolerance = 2e-13)
{
    if (!std::isfinite(value) || (expected == 0.0 ? value != 0.0
            : std::abs(value / expected - 1.0) > tolerance))
        throw std::runtime_error("A logarithmic activity lost a finite result.");
}
}

int main()
{
    try
    {
        for (double mu : {0.0, 100.0, 700.0, 710.0, 720.0})
            check(apply_density_normalization(std::exp(-mu), log_fugacity::normalization<double>(mu)),
                  1.0, 3e-11); // the input exp(-720) is itself subnormal
        check(apply_density_normalization(std::exp(700.0), log_fugacity::normalization<double>(-750.0)),
              std::exp(-50.0));
        const auto normalization = log_fugacity::normalization<std::complex<double>>(710.0, 0.5, 3.0);
        const auto complex_value = apply_density_normalization(
            std::complex<double>(0.8, -0.6) * std::exp(-710.0), normalization);
        check(complex_value.real(), 1.2);
        check(complex_value.imag(), -0.9);

        const double infinity = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        check(apply_density_normalization(nan, log_fugacity::normalization<double>(-infinity)), 0.0);
        check(apply_density_normalization(0.0, log_fugacity::normalization<double>(1e308)), 0.0);
        check(apply_density_normalization(std::numeric_limits<double>::max(),
              log_fugacity::normalization<double>(-1e308)), 0.0);
        if (!std::isinf(apply_density_normalization(1.0, log_fugacity::normalization<double>(1e308))))
            throw std::runtime_error("An unrepresentable physical activity was clipped.");
        for (double invalid : {infinity, nan})
        {
            bool rejected = false;
            try { log_fugacity::normalization<double>(invalid); }
            catch (const Exception&) { rejected = true; }
            if (!rejected)
                throw std::runtime_error("An invalid logarithmic activity was accepted.");
        }
        std::cout << "Log fugacity: finite range, complex phase, zero weight and invalid inputs passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
