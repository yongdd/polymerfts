#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

#include "PartitionAccumulator.h"

static void close(double value, double expected)
{
    if (!std::isfinite(value) || (expected == 0.0 ? value != 0.0 : std::abs(value/expected - 1.0) > 2e-14))
        throw std::runtime_error("Scaled partition reduction lost a representable result.");
}

int main()
{
    try
    {
        using namespace partition_accumulation;
        // The raw pair and the sum both exceed double range. Aggregation and
        // grid normalization belong inside the scaled reduction.
        Sum sum;
        for (int i = 0; i < 8; ++i)
            sum = Add{}(sum, product(1e308, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0, 8.0, 8.0, false));
        close(value<double>(sum), 5e307);

        // A shared bead must be removed before a tiny product is rounded to zero.
        sum = product(1e-200, 0.0, 1e-200, 0.0, 1e-200, 0.0, 1.0, 1.0, 1.0, false);
        close(value<double>(sum), 1e-200);

        // Components retain separate exponents, including after complex division.
        sum = product(1e200, 1e-200, 1e100, 0.0, 1e100, 0.0, 1.0, 1.0, 1.0, false);
        const auto complex_value = value<std::complex<double>>(sum);
        close(complex_value.real(), 1e200);
        close(complex_value.imag(), 1e-200);

        // Partial sums outside double range must still cancel to exactly zero.
        Sum positive = product(1e308, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, false);
        Sum negative = product(-1e308, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, false);
        close(value<double>(Add{}(positive, negative)), 0.0);
        close(value<double>(product(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, true)), 0.0);

        // The absolute-integrand scale receives the same normalization before hypot.
        sum = product(1e308, 1e308, 1.0, 0.0, 1.0, 0.0, 1.0, 4.0, 1.0, true);
        close(value<double>(sum), std::sqrt(2.0)*2.5e307);
        // Each normalized term lies below the smallest double, but the sum
        // is representable. A zero imaginary component must not set its scale.
        sum = {};
        for (int i = 0; i < 8; ++i)
            sum = Add{}(sum, product(1e-308, 0.0, 1e-15, 0.0, 1.0, 0.0, 1.0, 8.0, 1.0, true));
        close(value<double>(sum), 1e-323);
        std::cout << "Scaled partition reduction passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
