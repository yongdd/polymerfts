#include <cmath>
#include <complex>
#include <iostream>
#include <limits>

#include "DensityNormalization.h"

int main()
{
    using Complex = std::complex<double>;
    int failures = 0;
    auto check = [&](bool condition, const char* name)
    {
        std::cout << (condition ? "PASS " : "FAIL ") << name << '\n';
        failures += !condition;
    };

    // Neither the individual term nor its sum fits double, but the density does.
    double value = 0.0, exponent = 0.0;
    for (int i = 0; i < 128; ++i)
        density_accumulation::accumulate(value, exponent, 0x1p600, 0x1p500, 1.0);
    auto norm = make_density_normalization(0x1p-207, 1.0, 1.0, 1.0, 1.0);
    check(apply_density_normalization(value, norm, exponent) == 0x1p900,
          "large contour sum is normalized before restoring its exponent");

    // Scaling every term down by the contour count would erase these terms.
    value = exponent = 0.0;
    for (int i = 0; i < 128; ++i)
        density_accumulation::accumulate(value, exponent, 0x1p-700, 0x1p-700, 1.0);
    norm = make_density_normalization(0x1p1000, 1.0, 1.0, 1.0, 1.0);
    check(apply_density_normalization(value, norm, exponent) == 0x1p-393,
          "sub-double contour terms survive normalization");

    Complex complex_value = 0.0, complex_exponent = 0.0;
    for (int i = 0; i < 128; ++i)
        density_accumulation::accumulate(complex_value, complex_exponent,
            Complex(0x1p600, 0x1p-600), Complex(0x1p500), 1.0, nullptr);
    const auto complex_norm = make_density_normalization(0x1p-207, 1.0, 1.0, 1.0, Complex(1.0));
    check(apply_density_normalization(complex_value, complex_norm, complex_exponent)
              == Complex(0x1p900, 0x1p-300),
          "complex components retain independent exponents across 1200 bits");

    // Complex division rotates the independently stored components as well.
    const auto rotated_norm = make_density_normalization(0x1p-207, 1.0, 1.0, 1.0, Complex(0.0, 1.0));
    check(apply_density_normalization(complex_value, rotated_norm, complex_exponent)
              == Complex(0x1p-300, -0x1p900),
          "complex partition normalization preserves both component ranges");

    value = exponent = 0.0;
    density_accumulation::accumulate(value, exponent, 0x1p600, 0x1p600, 1.0);
    density_accumulation::accumulate(value, exponent, -0x1p600, 0x1p600, 1.0);
    density_accumulation::accumulate(value, exponent, 1.0, 1.0, 1.0);
    norm = make_density_normalization(1.0, 1.0, 1.0, 1.0, 1.0);
    check(apply_density_normalization(value, norm, exponent) == 1.0,
          "opposite out-of-range terms cancel before a finite contribution");

    value = exponent = 0.0;
    density_accumulation::accumulate(value, exponent, 0x1p900, 0x1p900, 1.0, 0x1p800);
    norm = make_density_normalization(0x1p-1000, 1.0, 1.0, 1.0, 1.0);
    check(apply_density_normalization(value, norm, exponent) == 1.0,
          "shared-bead division is included in the scaled term");

    const auto absent = make_density_normalization(0.0, 1.0, 1.0, 1.0, 0.0);
    check(apply_density_normalization(std::numeric_limits<double>::infinity(), absent, 1200.0) == 0.0,
          "zero population remains zero even with a nonfinite unused integral");

    // Ordinary arithmetic is the previous sequence, with no changes to rounding.
    value = exponent = 0.0;
    double reference = 0.0;
    for (int i = 1; i <= 100; ++i)
    {
        const double a = 0.125 * i, b = 1.0 / i, weight = 4.0 / 3.0;
        reference += weight * a * b;
        density_accumulation::accumulate(value, exponent, a, b, weight);
    }
    check(value == reference && exponent == 0.0, "ordinary accumulations retain their arithmetic");
    return failures == 0 ? 0 : 1;
}
