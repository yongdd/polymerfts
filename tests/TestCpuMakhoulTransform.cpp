/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestCpuMakhoulTransform.cpp
 * @brief DCT and DST of MakhoulFFT against their O(N^2) definitions.
 *
 * createFFT() runs the DCT and DST with either library as one reordered real
 * FFT of the grid (MakhoulFFT over MakhoulTransform). For every combination of
 * reflecting and absorbing axes, on grids of one to three axes with even, odd
 * and unit lengths, real and complex fields of standard deviation 5 must match
 * the definition axis by axis:
 *
 * - forward: REDFT10 or RODFT10 / 2 per axis
 * - backward: REDFT01 or RODFT01 / n per axis
 *
 * Forward and backward transforms also run in place (CrysFFTPmmm relies on
 * it) and concurrently, with their thread-local scratch, on one object, and must
 * give the same results as out of place and serially. The backward transform with
 * a factor per coefficient and one per grid point, each given or not, which
 * MakhoulTransform applies inside its loops, must give the values of separate
 * loops around backward() bit for bit, out of place and in place. The definitions
 * and the mixed grids are those of TestMakhoulDefinitions.h; the error maxima keep
 * a NaN (TestNanSafe.h).
 *
 * Boxes that mix periodic axes and walls (CpuFourierTransform) run
 * MakhoulTransform over the real FFT of either library with periodic axes. For
 * every mix of periodic, reflecting and absorbing axes on grids of two and three
 * axes with at least one periodic axis and one wall, with even, odd and unit
 * lengths, real and complex fields of standard deviation 5 must match the
 * definition axis by axis:
 *
 * - forward: the DFT exp(-2 pi i m j / n) along periodic axes, REDFT10 or RODFT10 / 2 along walls
 * - backward: the inverse DFT / n along periodic axes, REDFT01 or RODFT01 / n along walls,
 *   from random complex coefficients without symmetry; a real field is the real part
 *
 * and concurrent calls must give the serial results.
 */

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Exception.h"
#include "ComputationBox.h"
#include "FFTFactory.h"
#include "TestMakhoulDefinitions.h"

std::vector<double> definition(std::vector<double> data, const std::vector<int>& nx, int sine_mask, bool forward)
{
    for (size_t axis = 0; axis < nx.size(); ++axis)
        apply_axis(data, nx, axis, sine_mask >> axis & 1, forward);
    return data;
}

double max_difference(const std::vector<double>& a, const std::vector<double>& b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = max_or_nan(worst, std::abs(a[i] - b[i]));
    return worst;
}

/// Largest error relative to the largest value, of a field of one type on one grid
template <typename T, int DIM>
double check_grid(FFTBackend backend, std::array<int, DIM> nx, int sine_mask, std::mt19937& rng)
{
    const bool is_complex = !std::is_same<T, double>::value;
    const int parts = is_complex ? 2 : 1;
    int total = 1;
    std::array<BoundaryCondition, DIM> bc;
    for (int d = 0; d < DIM; ++d)
    {
        total *= nx[d];
        bc[d] = (sine_mask >> d & 1) ? BoundaryCondition::ABSORBING : BoundaryCondition::REFLECTING;
    }
    const std::vector<int> axes(nx.begin(), nx.end());
    const std::unique_ptr<FFT<T>> transform = createFFT<T>(axes, std::vector<BoundaryCondition>(bc.begin(), bc.end()), backend);
    FFT<T>& fft = *transform;

    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> field(parts * total), coefficient(parts * total), result(parts * total);
    for (double& x : field)
        x = normal(rng);

    double worst = 0.0;
    for (bool forward : {true, false})
    {
        // Interleaved (re, im) pairs for a complex field
        fft.forward(reinterpret_cast<T*>(field.data()), coefficient.data());
        if (forward)
            result = coefficient;
        else
            fft.backward(field.data(), reinterpret_cast<T*>(result.data()));
        for (int part = 0; part < parts; ++part)
        {
            std::vector<double> in(total), out(total);
            for (int i = 0; i < total; ++i)
            {
                in[i] = field[parts * i + part];
                out[i] = result[parts * i + part];
            }
            const std::vector<double> expected = definition(in, axes, sine_mask, forward);
            worst = max_or_nan(worst, max_difference(out, expected) / max_abs_or_nan(expected));
        }
    }

    // Round trip
    fft.forward(reinterpret_cast<T*>(field.data()), coefficient.data());
    fft.backward(coefficient.data(), reinterpret_cast<T*>(result.data()));
    worst = max_or_nan(worst, max_difference(result, field) / max_abs_or_nan(field));

    // The same round trip in place
    std::vector<double> in_place = field;
    fft.forward(reinterpret_cast<T*>(in_place.data()), in_place.data());
    worst = max_or_nan(worst, max_difference(in_place, coefficient) / max_abs_or_nan(coefficient));
    fft.backward(in_place.data(), reinterpret_cast<T*>(in_place.data()));
    worst = max_or_nan(worst, max_difference(in_place, result) / max_abs_or_nan(result));

    // The backward transform with factors: the values of separate loops around backward(), as the
    // default of FFT<T> computes them, bit for bit (factors from a generator of their own, so the
    // fields of the other checks stay those of the shared one)
    {
        std::mt19937 factor_rng(17 + total);
        std::normal_distribution<double> factor_normal(0.0, 0.5);
        std::vector<double> k_factor(total), r_factor(parts * total);
        for (int i = 0; i < total; ++i)
            k_factor[i] = std::exp(-0.37 * (i % 11));
        for (double& x : r_factor)
            x = std::exp(factor_normal(factor_rng));
        const T* r_field = reinterpret_cast<const T*>(r_factor.data());
        for (bool with_k : {false, true})
        {
            for (bool with_r : {false, true})
            {
                std::vector<double> separate = coefficient, expected(parts * total);
                T* separate_coefficients = reinterpret_cast<T*>(separate.data());
                T* expected_field = reinterpret_cast<T*>(expected.data());
                if (with_k)
                    for (int i = 0; i < total; ++i)
                        separate_coefficients[i] *= k_factor[i];
                fft.backward(separate.data(), expected_field);
                if (with_r)
                    for (int i = 0; i < total; ++i)
                        expected_field[i] *= r_field[i];

                std::vector<double> fused = coefficient, out(parts * total), in_place_fused = coefficient;
                fft.backward_with_factors(fused.data(), with_k ? k_factor.data() : nullptr,
                                          reinterpret_cast<T*>(out.data()), with_r ? r_field : nullptr);
                fft.backward_with_factors(in_place_fused.data(), with_k ? k_factor.data() : nullptr,
                                          reinterpret_cast<T*>(in_place_fused.data()), with_r ? r_field : nullptr);
                if (out != expected || in_place_fused != expected)
                {
                    std::cout << "  backward_with_factors differs from separate loops (k_factor " << with_k
                              << ", r_factor " << with_r << ", " << (is_complex ? "complex" : "real") << " field, grid";
                    for (int n : nx)
                        std::cout << " " << n;
                    std::cout << ", sine axes " << sine_mask << ")" << std::endl;
                    worst = max_or_nan(worst, 1.0);
                }
            }
        }
    }

#ifdef _OPENMP
    // Concurrent transforms of one object, one field per thread
    const int threads = 4;
    std::vector<std::vector<double>> fields(threads, std::vector<double>(parts * total));
    std::vector<std::vector<double>> serial(threads, std::vector<double>(2 * parts * total));
    std::vector<std::vector<double>> parallel(threads, std::vector<double>(2 * parts * total));
    for (int t = 0; t < threads; ++t)
    {
        for (double& x : fields[t])
            x = normal(rng);
        fft.forward(reinterpret_cast<T*>(fields[t].data()), serial[t].data());
        fft.backward(fields[t].data(), reinterpret_cast<T*>(serial[t].data() + parts * total));
    }
    #pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (int t = 0; t < threads; ++t)
    {
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            fft.forward(reinterpret_cast<T*>(fields[t].data()), parallel[t].data());
            fft.backward(fields[t].data(), reinterpret_cast<T*>(parallel[t].data() + parts * total));
        }
    }
    for (int t = 0; t < threads; ++t)
        worst = max_or_nan(worst, max_difference(parallel[t], serial[t]) / max_abs_or_nan(serial[t]));
#endif
    return worst;
}

template <typename T>
double check_all_grids(FFTBackend backend, std::mt19937& rng)
{
    double worst = 0.0;
    for (int n : {1, 2, 3, 4, 5, 7, 8, 16, 17})
        for (int mask = 0; mask < 2; ++mask)
            worst = max_or_nan(worst, check_grid<T, 1>(backend, {n}, mask, rng));
    for (std::array<int, 2> nx : std::vector<std::array<int, 2>>{{1, 1}, {1, 6}, {6, 1}, {2, 2}, {4, 6}, {5, 7}, {3, 8}, {7, 2}, {12, 10}})
        for (int mask = 0; mask < 4; ++mask)
            worst = max_or_nan(worst, check_grid<T, 2>(backend, nx, mask, rng));
    for (std::array<int, 3> nx : std::vector<std::array<int, 3>>{{1, 1, 1}, {2, 2, 2}, {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {4, 1, 6}, {1, 5, 6}, {9, 8, 7}, {16, 12, 10}})
        for (int mask = 0; mask < 8; ++mask)
            worst = max_or_nan(worst, check_grid<T, 3>(backend, nx, mask, rng));
    return worst;
}

int check_library(const char* name, FFTBackend backend, std::mt19937& rng)
{
    const double tolerance = 1e-13;
    const double worst_real = check_all_grids<double>(backend, rng);
    const double worst_complex = check_all_grids<std::complex<double>>(backend, rng);
    std::cout << name << ": real " << worst_real << ", complex " << worst_complex << std::endl;
    return (worst_real < tolerance && worst_complex < tolerance) ? 0 : 1;
}

//------------------------------------------------------------------------------
// Boxes that mix periodic axes and walls
//------------------------------------------------------------------------------
/// Largest error relative to the largest value, of a real or complex field on one mixed grid
double check_mixed_grid(FFTBackend backend, const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                        bool complex_field, std::mt19937& rng)
{
    int total = 1;
    int walls = 0;
    for (size_t d = 0; d < nx.size(); ++d)
    {
        total *= nx[d];
        walls += bc[d] != BoundaryCondition::PERIODIC;
    }
    const MakhoulTransform transform(nx, bc, createRealFFT(nx, backend));
    // As CpuFourierTransform: REDFT10 and RODFT10 are twice the series
    const double forward_scale = 1.0 / (1 << walls);
    const double backward_scale = 1.0 / total;

    // A real field is stored in the real parts; the coefficients have no symmetry
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<Complex> field(total), coefficients(total);
    for (Complex& x : field)
        x = complex_field ? Complex(normal(rng), normal(rng)) : Complex(normal(rng), 0.0);
    for (Complex& x : coefficients)
        x = Complex(normal(rng), normal(rng));

    auto forward = [&](const std::vector<Complex>& in, std::vector<Complex>& out) {
        if (complex_field)
        {
            transform.forward(in.data(), out.data(), forward_scale);
            return;
        }
        std::vector<double> real(total);
        for (int i = 0; i < total; ++i)
            real[i] = in[i].real();
        transform.forward(real.data(), out.data(), forward_scale);
    };
    auto backward = [&](const std::vector<Complex>& in, std::vector<Complex>& out) {
        if (complex_field)
        {
            transform.backward(in.data(), out.data(), backward_scale);
            return;
        }
        std::vector<double> real(total);
        transform.backward(in.data(), real.data(), backward_scale);
        for (int i = 0; i < total; ++i)
            out[i] = real[i];
    };

    std::vector<Complex> result(total), round_trip(total);
    forward(field, result);
    std::vector<Complex> expected = mixed_definition(field, nx, bc, true);
    double worst = max_difference(result, expected) / max_abs(expected);

    backward(result, round_trip);
    worst = max_or_nan(worst, max_difference(round_trip, field) / max_abs(field));

    backward(coefficients, result);
    expected = mixed_definition(coefficients, nx, bc, false);
    if (!complex_field)
        for (Complex& x : expected)
            x = x.real();
    worst = max_or_nan(worst, max_difference(result, expected) / max_abs(expected));

#ifdef _OPENMP
    // Concurrent transforms of one object, one field per thread
    const int threads = 4;
    std::vector<std::vector<Complex>> fields(threads, std::vector<Complex>(total));
    std::vector<std::vector<Complex>> serial(threads, std::vector<Complex>(total));
    std::vector<std::vector<Complex>> parallel(threads, std::vector<Complex>(total));
    for (int t = 0; t < threads; ++t)
    {
        for (Complex& x : fields[t])
            x = complex_field ? Complex(normal(rng), normal(rng)) : Complex(normal(rng), 0.0);
        std::vector<Complex> modes(total);
        forward(fields[t], modes);
        backward(modes, serial[t]);
    }
    #pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (int t = 0; t < threads; ++t)
    {
        std::vector<Complex> modes(total);
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            forward(fields[t], modes);
            backward(modes, parallel[t]);
        }
    }
    for (int t = 0; t < threads; ++t)
        worst = max_or_nan(worst, max_difference(parallel[t], serial[t]) / max_abs(serial[t]));
#endif
    return worst;
}

int check_mixed(const char* name, FFTBackend backend, std::mt19937& rng)
{
    const double tolerance = 1e-13;
    double worst_real = 0.0;
    double worst_complex = 0.0;
    int grids = 0;
    for (const std::vector<int>& nx : mixed_grids())
    {
        for (const std::vector<BoundaryCondition>& bc : mixed_conditions(static_cast<int>(nx.size())))
        {
            worst_real = max_or_nan(worst_real, check_mixed_grid(backend, nx, bc, false, rng));
            worst_complex = max_or_nan(worst_complex, check_mixed_grid(backend, nx, bc, true, rng));
            ++grids;
        }
    }
    std::cout << name << " with periodic axes (" << grids << " grids): real " << worst_real
              << ", complex " << worst_complex << std::endl;
    return (worst_real < tolerance && worst_complex < tolerance) ? 0 : 1;
}

int main()
{
    try
    {
        std::mt19937 rng(11);
        int failures = 0;
#ifdef USE_CPU_FFTW
        failures += check_library("MakhoulFFT over FftwRealFFT", FFTBackend::FFTW, rng);
#endif
#ifdef USE_CPU_MKL
        failures += check_library("MakhoulFFT over MklRealFFT", FFTBackend::MKL, rng);
#endif
#ifdef USE_CPU_FFTW
        failures += check_mixed("MakhoulTransform over FftwRealFFT", FFTBackend::FFTW, rng);
#endif
#ifdef USE_CPU_MKL
        failures += check_mixed("MakhoulTransform over MklRealFFT", FFTBackend::MKL, rng);
#endif
        return failures == 0 ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
