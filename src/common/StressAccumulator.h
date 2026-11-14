#ifndef POLYMERFTS_STRESS_ACCUMULATOR_H_
#define POLYMERFTS_STRESS_ACCUMULATOR_H_

#include <array>
#include <type_traits>

#include "DensityNormalization.h"

#ifdef __CUDACC__
#define POLYMERFTS_STRESS_HD __host__ __device__
#else
#define POLYMERFTS_STRESS_HD
#endif

namespace stress_accumulation
{
// A stress integral need not fit in a double before division by Q. As for
// densities, exponent zero keeps the original unscaled arithmetic. Real and
// imaginary components have independent exponents after the FFT contraction.
struct Component
{
    double value;
    double exponent;
};

POLYMERFTS_STRESS_HD inline Component encode(scaled_product::Number number)
{
    return {number.mantissa, number.mantissa == 0.0 ? 0.0 : static_cast<double>(number.exponent)};
}

POLYMERFTS_STRESS_HD inline Component shifted(double value, int shift)
{
    if (shift == 0)
        return {value, 0.0};
    auto number = scaled_product::split(value);
    number.exponent += shift;
    return encode(number);
}

POLYMERFTS_STRESS_HD inline Component add(Component a, Component b)
{
    if (a.exponent == 0.0 && b.exponent == 0.0
        && scaled_product::ordinary(a.value) && scaled_product::ordinary(b.value))
        return {a.value + b.value, 0.0};
    return encode(scaled_product::add(density_accumulation::stored(a.value, a.exponent),
                                     density_accumulation::stored(b.value, b.exponent)));
}

POLYMERFTS_STRESS_HD inline Component multiply(Component a, double factor)
{
    if (a.exponent == 0.0 && scaled_product::ordinary(a.value) && scaled_product::ordinary(factor))
#ifdef __CUDA_ARCH__
        // Match the separately rounded multiplication in the host contour
        // accumulator used by reduce-memory computations.
        return {__dmul_rn(a.value, factor), 0.0};
#else
        return {a.value * factor, 0.0};
#endif
    return encode(scaled_product::multiply(density_accumulation::stored(a.value, a.exponent),
                                          scaled_product::split(factor)));
}

POLYMERFTS_STRESS_HD inline Component divide(Component a, double divisor)
{
    if (a.exponent == 0.0 && scaled_product::ordinary(a.value) && scaled_product::ordinary(divisor))
        return {a.value / divisor, 0.0};
    return encode(scaled_product::ratio(density_accumulation::stored(a.value, a.exponent),
                                       scaled_product::split(divisor)));
}

// Leave ordinary propagators bitwise unchanged. Outside this range, scaling
// each complete field by a power of two protects both its FFT and the product
// of the two transforms. The inverse scale accompanies the stress integral.
POLYMERFTS_STRESS_HD inline int input_shift(double maximum)
{
    if (scaled_product::ordinary(maximum) || !scaled_product::finite(maximum))
        return 0;
    int exponent = 0;
    frexp(maximum, &exponent);
    return exponent;
}

// A common scale preserves a complex phase, but must not turn a smaller
// nonzero component into a subnormal (or zero). Such pairs are contracted as
// four real bilinear forms, each with its own input exponents.
POLYMERFTS_STRESS_HD inline bool needs_component_split(double real_maximum, double imag_maximum)
{
    const double maximum = real_maximum > imag_maximum ? real_maximum : imag_maximum;
    const double minimum = real_maximum < imag_maximum ? real_maximum : imag_maximum;
    const int shift = input_shift(maximum);
    return shift > 0 && minimum > 0.0 && scalbn(minimum, -shift) < 0x1p-1022;
}

// Round the product before addition on CUDA as the reduce-memory host
// accumulator does. Otherwise full storage can contract the two operations
// and differ by an ulp even when all propagators are identical.
POLYMERFTS_STRESS_HD inline Component accumulate_segment(
    Component accumulated, double segment, int shift, double weight)
{
    if (accumulated.exponent == 0.0 && shift == 0
        && scaled_product::ordinary(accumulated.value) && scaled_product::ordinary(segment)
        && scaled_product::ordinary(weight))
#ifdef __CUDA_ARCH__
        return {__dadd_rn(accumulated.value, __dmul_rn(weight, segment)), 0.0};
#else
        return {accumulated.value + weight * segment, 0.0};
#endif
    return add(accumulated, multiply(shifted(segment, shift), weight));
}
} // namespace stress_accumulation

template <typename T>
struct StressValue
{
    T value{};
    T exponent{};

    static StressValue shifted(T value, int shift)
    {
        const auto real = stress_accumulation::shifted(std::real(value), shift);
        if constexpr (std::is_same_v<T, double>)
            return {real.value, real.exponent};
        else
        {
            const auto imag = stress_accumulation::shifted(std::imag(value), shift);
            return {{real.value, imag.value}, {real.exponent, imag.exponent}};
        }
    }

    template <typename Operation>
    StressValue componentwise(Operation operation) const
    {
        const auto real = operation(stress_accumulation::Component{std::real(value), std::real(exponent)}, false);
        if constexpr (std::is_same_v<T, double>)
            return {real.value, real.exponent};
        else
        {
            const auto imag = operation(stress_accumulation::Component{std::imag(value), std::imag(exponent)}, true);
            return {{real.value, imag.value}, {real.exponent, imag.exponent}};
        }
    }

    StressValue operator+(const StressValue& other) const
    {
        return componentwise([&](auto component, bool imag) {
            return stress_accumulation::add(component,
                {imag ? std::imag(other.value) : std::real(other.value),
                 imag ? std::imag(other.exponent) : std::real(other.exponent)});
        });
    }
    StressValue& operator+=(const StressValue& other) { return *this = *this + other; }
    StressValue operator*(double factor) const
    {
        return componentwise([&](auto component, bool) { return stress_accumulation::multiply(component, factor); });
    }
    StressValue& operator*=(double factor) { return *this = *this * factor; }
    StressValue operator/(double divisor) const
    {
        return componentwise([&](auto component, bool) { return stress_accumulation::divide(component, divisor); });
    }
    T normalized(const DensityNormalization<T>& normalization) const
    {
        return apply_density_normalization(value, normalization, exponent);
    }
};

template <typename T>
StressValue<T> operator*(double factor, const StressValue<T>& value) { return value * factor; }

template <typename T>
using StressAccumulator = std::array<StressValue<T>, 6>;

#undef POLYMERFTS_STRESS_HD
#endif
