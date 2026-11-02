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
 * @brief DCT and DST of FftwFFT and MklFFT against their O(N^2) definitions.
 *
 * Both libraries run the DCT and DST as one reordered real FFT of the grid
 * (MakhoulTransform). For every combination of reflecting and absorbing axes,
 * on grids of one to three axes with even, odd and unit lengths, real and
 * complex fields of standard deviation 5 must match the definition axis by axis:
 *
 * - forward: REDFT10 or RODFT10 / 2 per axis
 * - backward: REDFT01 or RODFT01 / n per axis
 *
 * Forward and backward transforms also run concurrently, with their thread-local
 * scratch, on one object and must give the same results as serially. The error
 * maxima keep a NaN (TestNanSafe.h).
 */

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Exception.h"
#include "ComputationBox.h"
#include "TestNanSafe.h"
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif
#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

/// One axis of the definition: apply to every line of `data` along `axis`.
void apply_axis(std::vector<double>& data, const std::vector<int>& nx, int axis, bool sine, bool forward)
{
    const double pi = std::numbers::pi;
    const int n = nx[axis];
    int stride = 1;
    for (size_t d = axis + 1; d < nx.size(); ++d)
        stride *= nx[d];
    std::vector<double> line(n);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
        if ((i / stride) % n != 0)
            continue;
        for (int j = 0; j < n; ++j)
            line[j] = data[i + j * stride];
        for (int k = 0; k < n; ++k)
        {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
            {
                if (forward && !sine)
                    sum += line[j] * std::cos(pi * (j + 0.5) * k / n);
                else if (forward)
                    sum += line[j] * std::sin(pi * (j + 0.5) * (k + 1) / n);
                else if (!sine && j > 0)
                    sum += line[j] * 2.0 * std::cos(pi * j * (k + 0.5) / n);
                else if (!sine)
                    sum += line[j];
                else if (j < n - 1)
                    sum += line[j] * 2.0 * std::sin(pi * (j + 1) * (k + 0.5) / n);
                else
                    sum += line[j] * (k % 2 == 0 ? 1.0 : -1.0);
            }
            // The factor 2 of REDFT10 and RODFT10 cancels against the / 2 of forward
            data[i + k * stride] = forward ? sum : sum / n;
        }
    }
}

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
template <template <typename, int> class Transform, typename T, int DIM>
double check_grid(std::array<int, DIM> nx, int sine_mask, std::mt19937& rng)
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
    Transform<T, DIM> fft(nx, bc);

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

template <template <typename, int> class Transform, typename T>
double check_all_grids(std::mt19937& rng)
{
    double worst = 0.0;
    for (int n : {1, 2, 3, 4, 5, 7, 8, 16, 17})
        for (int mask = 0; mask < 2; ++mask)
            worst = max_or_nan(worst, check_grid<Transform, T, 1>({n}, mask, rng));
    for (std::array<int, 2> nx : std::vector<std::array<int, 2>>{{1, 1}, {1, 6}, {6, 1}, {2, 2}, {4, 6}, {5, 7}, {3, 8}, {7, 2}, {12, 10}})
        for (int mask = 0; mask < 4; ++mask)
            worst = max_or_nan(worst, check_grid<Transform, T, 2>(nx, mask, rng));
    for (std::array<int, 3> nx : std::vector<std::array<int, 3>>{{1, 1, 1}, {2, 2, 2}, {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {4, 1, 6}, {1, 5, 6}, {9, 8, 7}, {16, 12, 10}})
        for (int mask = 0; mask < 8; ++mask)
            worst = max_or_nan(worst, check_grid<Transform, T, 3>(nx, mask, rng));
    return worst;
}

template <template <typename, int> class Transform>
int check_library(const char* name, std::mt19937& rng)
{
    const double tolerance = 1e-13;
    const double worst_real = check_all_grids<Transform, double>(rng);
    const double worst_complex = check_all_grids<Transform, std::complex<double>>(rng);
    std::cout << name << ": real " << worst_real << ", complex " << worst_complex << std::endl;
    return (worst_real < tolerance && worst_complex < tolerance) ? 0 : 1;
}

int main()
{
    try
    {
        std::mt19937 rng(11);
        int failures = 0;
#ifdef USE_CPU_FFTW
        failures += check_library<FftwFFT>("FftwFFT", rng);
#endif
#ifdef USE_CPU_MKL
        failures += check_library<MklFFT>("MklFFT", rng);
#endif
        return failures == 0 ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
