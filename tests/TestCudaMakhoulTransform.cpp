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
 * @file TestCudaMakhoulTransform.cpp
 * @brief CudaMakhoulTransform against the O(N^2) definitions, the CPU MakhoulTransform
 *        and FFTW's REDFT/RODFT transforms; the round trips of the CUDA transforms.
 *
 * The CUDA wall boxes (the propagator solvers and CudaFourierTransform), the Pmmm
 * CrysFFT and the boxes that mix periodic axes and walls (CudaFourierTransform) run
 * one cuFFT of the grid with Makhoul's reordering and combining on the wall axes.
 * Real and complex fields of standard deviation 5 must match the definition axis by
 * axis, as TestCpuMakhoulTransform checks the CPU version:
 *
 * - forward: the DFT exp(-2 pi i m j / n) along periodic axes, REDFT10 or RODFT10 / 2 along walls
 * - backward: the inverse DFT / n along periodic axes, REDFT01 or RODFT01 / n along walls,
 *   from random coefficients: complex without symmetry when a periodic axis is present or the
 *   field is complex (a real field is the real part), real for a real field between walls
 * - backward(forward(f)) = f
 *
 * on these grids:
 *
 * - every mix of periodic, reflecting and absorbing axes on grids of two and three axes
 *   with at least one periodic axis and one wall, with even, odd and unit lengths, and a
 *   few larger grids (the mixed grids of TestMakhoulDefinitions.h);
 * - walls only, every mix of reflecting and absorbing axes on the grids of one to three
 *   axes of TestCpuMakhoulTransform and a few larger ones, whose real fields have real
 *   coefficients.
 *
 * The forward and backward transforms keep their inputs, and with a CPU backend
 * they agree with MakhoulTransform over its real FFT (the largest relative
 * difference is printed). The backward transform with a factor per coefficient
 * equals the backward transform of the products bit for bit. The error maxima keep a
 * NaN (TestNanSafe.h).
 *
 * On three grids of more than two million points, whose real half spectrum also
 * has more than MAX_GRID_THREADS modes, every grid-stride loop of the kernels takes
 * more than one stride. There the definition is summed for the corners and 16
 * random points of every result, and with a CPU backend MakhoulTransform checks
 * the whole results.
 *
 * Taken with scale 1 between walls, the forward transform is FFTW's unnormalized
 * REDFT10 / RODFT10 per axis (the type-2 DCT along reflecting, the type-2 DST along
 * absorbing walls) and the backward transform REDFT01 / RODFT01 (type 3), which a
 * backward scale of 1/(2 n) per axis makes the inverse. Each of the following runs
 * is counted, and the test asserts their number:
 *
 * - odd wall lengths (1D 11, 3D 6x4x3, 2D 6x5) against direct sums of these series, the
 *   round trip, and with a CPU backend MakhoulTransform;
 * - round trips of the ramp 1..12 and of uniform random fields along both walls in 1D,
 *   8x6 and 6x4x3, and the round trip of the periodic CudaFourierTransform (cuFFT);
 * - with FFTW: the transforms of one family (DCT or DST) and of mixed families per axis
 *   against fftw_plan_r2r on grids of one to three axes up to 32^3, lengths of every
 *   parity and 1D lengths beyond 6144 points included, and the type-2/type-3 round trips
 *   of both libraries. A CUDA build without FFTW runs the other parts.
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ComputationBox.h"
#include "CudaCommon.h"
#include "CudaFourierTransform.h"
#include "CudaMakhoulTransform.h"
#include "Exception.h"
#include "TestChecks.h"
#include "TestMakhoulDefinitions.h"
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#include "FFTFactory.h"
#include "MakhoulTransform.h"
#endif
#ifdef USE_CPU_FFTW
#include <fftw3.h>
#endif

/// The factor of input point j in output point k along one axis, as in mixed_definition(), with
/// the angle reduced exactly to one period: on long axes pi (j + 1/2) k / n reaches thousands of
/// radians, and its rounding alone would cost 1e-13 in a sum over millions of points
Complex series_factor(BoundaryCondition kind, int n, int j, int k, bool forward)
{
    const double pi = std::numbers::pi;
    if (kind == BoundaryCondition::PERIODIC)
    {
        const Complex f = std::polar(1.0, (forward ? -2.0 : 2.0) * pi * static_cast<double>(j * k % n) / n);
        return forward ? f : f / static_cast<double>(n);
    }
    // The wall series at angles pi m / (2 n), m taken modulo the period 4 n
    const long long period = 4LL * n;
    const auto cos_at = [&](long long m) { return std::cos(pi * static_cast<double>(m % period) / (2.0 * n)); };
    const auto sin_at = [&](long long m) { return std::sin(pi * static_cast<double>(m % period) / (2.0 * n)); };
    const bool sine = kind == BoundaryCondition::ABSORBING;
    double f;
    if (forward && !sine)
        f = cos_at((2LL * j + 1) * k);                // cos(pi (j + 1/2) k / n)
    else if (forward)
        f = sin_at((2LL * j + 1) * (k + 1));          // sin(pi (j + 1/2) (k + 1) / n)
    else if (!sine && j > 0)
        f = 2.0 * cos_at(j * (2LL * k + 1));          // 2 cos(pi j (k + 1/2) / n)
    else if (!sine)
        f = 1.0;
    else if (j < n - 1)
        f = 2.0 * sin_at((j + 1) * (2LL * k + 1));    // 2 sin(pi (j + 1) (k + 1/2) / n)
    else
        f = k % 2 == 0 ? 1.0 : -1.0;
    return forward ? f : f / n;
}

/// The definition at one point of the result, for grids too large for mixed_definition(): the sum
/// over the whole input, along the last axis first so that the partial sums stay short
Complex definition_at(const std::vector<Complex>& data, const std::vector<int>& nx,
                      const std::vector<BoundaryCondition>& bc, bool forward, int index)
{
    // Axes padded to three in front
    std::array<int, 3> n{1, 1, 1}, k{0, 0, 0};
    std::array<BoundaryCondition, 3> kind{BoundaryCondition::PERIODIC, BoundaryCondition::PERIODIC,
                                          BoundaryCondition::PERIODIC};
    const int dim = static_cast<int>(nx.size());
    for (int a = dim - 1, rest = index; a >= 0; --a)
    {
        const int d = a + 3 - dim;
        n[d] = nx[a];
        kind[d] = bc[a];
        k[d] = rest % n[d];
        rest /= n[d];
    }
    std::array<std::vector<Complex>, 3> factor;
    for (int d = 0; d < 3; ++d)
        for (int j = 0; j < n[d]; ++j)
            factor[d].push_back(series_factor(kind[d], n[d], j, k[d], forward));

    Complex sum = 0.0;
    for (int j0 = 0; j0 < n[0]; ++j0)
    {
        Complex sum1 = 0.0;
        for (int j1 = 0; j1 < n[1]; ++j1)
        {
            const Complex* line = data.data() + (static_cast<size_t>(j0) * n[1] + j1) * n[2];
            Complex sum2 = 0.0;
            for (int j2 = 0; j2 < n[2]; ++j2)
                sum2 += line[j2] * factor[2][j2];
            sum1 += sum2 * factor[1][j1];
        }
        sum += sum1 * factor[0][j0];
    }
    return sum;
}

/// Largest difference of `result` from the definition applied to `input`, relative to the largest
/// value of the definition (its real part for a real result) at the points checked: the whole grid,
/// or with `samples` > 0 the corners and `samples` random points, for grids too large for
/// mixed_definition()
double definition_error(const std::vector<Complex>& result, const std::vector<Complex>& input,
                        const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, bool forward,
                        bool real_result, int samples, std::mt19937& rng)
{
    if (samples == 0)
    {
        std::vector<Complex> expected = mixed_definition(input, nx, bc, forward);
        if (real_result)
            for (Complex& x : expected)
                x = x.real();
        return max_difference(result, expected) / max_abs(expected);
    }
    const int dim = static_cast<int>(nx.size());
    std::vector<int> points;
    for (int corner = 0; corner < (1 << dim); ++corner)
    {
        int index = 0;
        for (int d = 0; d < dim; ++d)
            index = index * nx[d] + (((corner >> d) & 1) ? nx[d] - 1 : 0);
        points.push_back(index);
    }
    std::uniform_int_distribution<int> any_point(0, static_cast<int>(result.size()) - 1);
    for (int s = 0; s < samples; ++s)
        points.push_back(any_point(rng));

    double worst = 0.0, largest = 0.0;
    for (const int index : points)
    {
        Complex expected = definition_at(input, nx, bc, forward, index);
        if (real_result)
            expected = expected.real();
        worst = max_or_nan(worst, std::abs(result[index] - expected));
        largest = max_or_nan(largest, std::abs(expected));
    }
    return worst / largest;
}

/// Largest errors relative to the largest value of a real or complex field on one grid
struct GridErrors
{
    double definition = 0.0;   ///< forward, backward and round trip against the definition
    double cpu = 0.0;          ///< forward and backward against MakhoulTransform
    bool inputs_kept = true;
    bool k_factor_exact = true;   ///< backward with a factor per coefficient = backward of the products
};

/// The CUDA transform of host arrays: a real field lives in the real parts of `in`, and so do
/// real coefficients (a real field between walls only)
template <typename T>
struct DeviceTransform
{
    using DeviceT = CuDeviceData<T>;
    CudaMakhoulTransform<T> transform;
    int total;
    CudaDeviceBuffer<DeviceT> d_field;
    CudaDeviceBuffer<cuDoubleComplex> d_coefficients;
    CudaDeviceBuffer<double> d_k_factor;

    DeviceTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, int n)
        : transform(nx, bc), total(n), d_field(n), d_coefficients(n), d_k_factor(n) {}

    /// Upload, transform and download; `in_after` receives the device input after the call
    void forward(const std::vector<Complex>& in, std::vector<Complex>& out, double scale, std::vector<Complex>& in_after)
    {
        upload_field(in);
        transform.forward(d_field, reinterpret_cast<double*>(d_coefficients.get()), scale, 0);
        download_coefficients(out);
        download_field(in_after);
    }

    /// The backward transform of in, times k_factor per coefficient unless it is empty
    void backward(const std::vector<Complex>& in, std::vector<Complex>& out, double scale, std::vector<Complex>& in_after,
                  const std::vector<double>& k_factor = {})
    {
        upload_coefficients(in);
        if (!k_factor.empty())
            CUDA_CHECK(cudaMemcpy(d_k_factor, k_factor.data(), sizeof(double) * total, cudaMemcpyHostToDevice));
        transform.backward(reinterpret_cast<const double*>(d_coefficients.get()), k_factor.empty() ? nullptr : d_k_factor.get(),
                           d_field, scale, 0);
        download_field(out);
        download_coefficients(in_after);
    }

private:
    void upload_coefficients(const std::vector<Complex>& in)
    {
        if (transform.real_coefficients())
        {
            std::vector<double> real(total);
            for (int i = 0; i < total; ++i)
                real[i] = in[i].real();
            CUDA_CHECK(cudaMemcpy(d_coefficients, real.data(), sizeof(double) * total, cudaMemcpyHostToDevice));
        }
        else
            CUDA_CHECK(cudaMemcpy(d_coefficients, in.data(), sizeof(Complex) * total, cudaMemcpyHostToDevice));
    }

    void download_coefficients(std::vector<Complex>& out)
    {
        if (transform.real_coefficients())
        {
            std::vector<double> real(total);
            CUDA_CHECK(cudaMemcpy(real.data(), d_coefficients, sizeof(double) * total, cudaMemcpyDeviceToHost));
            for (int i = 0; i < total; ++i)
                out[i] = real[i];
        }
        else
            CUDA_CHECK(cudaMemcpy(out.data(), d_coefficients, sizeof(Complex) * total, cudaMemcpyDeviceToHost));
    }

    void upload_field(const std::vector<Complex>& in)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            std::vector<double> real(total);
            for (int i = 0; i < total; ++i)
                real[i] = in[i].real();
            CUDA_CHECK(cudaMemcpy(d_field, real.data(), sizeof(double) * total, cudaMemcpyHostToDevice));
        }
        else
            CUDA_CHECK(cudaMemcpy(d_field, in.data(), sizeof(Complex) * total, cudaMemcpyHostToDevice));
    }

    void download_field(std::vector<Complex>& out)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            std::vector<double> real(total);
            CUDA_CHECK(cudaMemcpy(real.data(), d_field, sizeof(double) * total, cudaMemcpyDeviceToHost));
            for (int i = 0; i < total; ++i)
                out[i] = real[i];
        }
        else
            CUDA_CHECK(cudaMemcpy(out.data(), d_field, sizeof(Complex) * total, cudaMemcpyDeviceToHost));
    }
};

#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
using CpuReference = MakhoulTransform;

/// MakhoulTransform of a grid, built once for its real and complex fields (FFTW measures its plans)
std::unique_ptr<CpuReference> cpu_reference(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc)
{
#ifdef USE_CPU_FFTW
    const FFTBackend backend = FFTBackend::FFTW;
#else
    const FFTBackend backend = FFTBackend::MKL;
#endif
    return std::make_unique<MakhoulTransform>(nx, bc, createRealFFT(nx, backend));
}
#else
struct CpuReference {};   ///< without a CPU backend there is nothing to compare with

std::unique_ptr<CpuReference> cpu_reference(const std::vector<int>&, const std::vector<BoundaryCondition>&)
{
    return nullptr;
}
#endif

/// One grid against the definition (on `samples` random points and the corners when > 0) and
/// MakhoulTransform
template <typename T>
GridErrors check_grid(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, std::mt19937& rng,
                      [[maybe_unused]] const CpuReference* cpu, int samples = 0)
{
    constexpr bool complex_field = !std::is_same_v<T, double>;
    int total = 1;
    int walls = 0;
    for (size_t d = 0; d < nx.size(); ++d)
    {
        total *= nx[d];
        walls += bc[d] != BoundaryCondition::PERIODIC;
    }
    [[maybe_unused]] const bool walls_only = walls == static_cast<int>(nx.size());
    DeviceTransform<T> gpu(nx, bc, total);
    // As CudaFourierTransform: REDFT10 and RODFT10 are twice the series
    const double forward_scale = 1.0 / (1 << walls);
    const double backward_scale = 1.0 / total;

    // A real field is stored in the real parts; the coefficients have no symmetry, and are real
    // for a real field between walls
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<Complex> field(total), coefficients(total);
    for (Complex& x : field)
        x = complex_field ? Complex(normal(rng), normal(rng)) : Complex(normal(rng), 0.0);
    for (Complex& x : coefficients)
        x = Complex(normal(rng), normal(rng));
    if (gpu.transform.real_coefficients())
        for (Complex& x : coefficients)
            x = x.real();

    GridErrors errors;
    std::vector<Complex> forward(total), backward(total), round_trip(total), kept(total);
    gpu.forward(field, forward, forward_scale, kept);
    errors.inputs_kept = errors.inputs_kept && kept == field;
    errors.definition = definition_error(forward, field, nx, bc, true, false, samples, rng);

    gpu.backward(forward, round_trip, backward_scale, kept);
    errors.inputs_kept = errors.inputs_kept && kept == forward;
    errors.definition = max_or_nan(errors.definition, max_difference(round_trip, field) / max_abs(field));

    gpu.backward(coefficients, backward, backward_scale, kept);
    errors.inputs_kept = errors.inputs_kept && kept == coefficients;
    errors.definition = max_or_nan(errors.definition, definition_error(backward, coefficients, nx, bc, false,
                                                                       !complex_field, samples, rng));

    // A factor per coefficient: the backward transform of the products, bit for bit (the factors
    // come from a generator of their own, so that the fields of the other grids stay those of rng)
    std::mt19937 factor_rng(static_cast<unsigned>(total));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> k_factor(total);
    for (double& k : k_factor)
        k = uniform(factor_rng);
    std::vector<Complex> products(total), with_factor(total), of_products(total);
    for (int i = 0; i < total; ++i)
        products[i] = Complex(coefficients[i].real() * k_factor[i], coefficients[i].imag() * k_factor[i]);
    gpu.backward(coefficients, with_factor, backward_scale, kept, k_factor);
    errors.inputs_kept = errors.inputs_kept && kept == coefficients;
    gpu.backward(products, of_products, backward_scale, kept);
    errors.k_factor_exact = with_factor == of_products;

#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
    std::vector<Complex> cpu_forward(total), cpu_backward(total);
    std::vector<double> real_in(total), real_out(total);
    if (walls_only)
    {
        // The real coefficients of MakhoulTransform, (re, im) pairs for a complex field
        std::vector<double> pairs(2 * total);
        if (complex_field)
        {
            cpu->forward(field.data(), pairs.data(), forward_scale);
            for (int i = 0; i < total; ++i)
                cpu_forward[i] = Complex(pairs[2 * i], pairs[2 * i + 1]);
            for (int i = 0; i < total; ++i)
            {
                pairs[2 * i] = coefficients[i].real();
                pairs[2 * i + 1] = coefficients[i].imag();
            }
            cpu->backward(pairs.data(), cpu_backward.data(), backward_scale);
        }
        else
        {
            for (int i = 0; i < total; ++i)
                real_in[i] = field[i].real();
            cpu->forward(real_in.data(), pairs.data(), forward_scale);
            for (int i = 0; i < total; ++i)
                cpu_forward[i] = pairs[i];
            for (int i = 0; i < total; ++i)
                pairs[i] = coefficients[i].real();
            cpu->backward(pairs.data(), real_out.data(), backward_scale);
            for (int i = 0; i < total; ++i)
                cpu_backward[i] = real_out[i];
        }
    }
    else if (complex_field)
    {
        cpu->forward(field.data(), cpu_forward.data(), forward_scale);
        cpu->backward(coefficients.data(), cpu_backward.data(), backward_scale);
    }
    else
    {
        for (int i = 0; i < total; ++i)
            real_in[i] = field[i].real();
        cpu->forward(real_in.data(), cpu_forward.data(), forward_scale);
        cpu->backward(coefficients.data(), real_out.data(), backward_scale);
        for (int i = 0; i < total; ++i)
            cpu_backward[i] = real_out[i];
    }
    errors.cpu = max_or_nan(max_difference(forward, cpu_forward) / max_abs(cpu_forward),
                            max_difference(backward, cpu_backward) / max_abs(cpu_backward));
#endif
    return errors;
}

//==============================================================================
// The unnormalized wall transforms: REDFT10/RODFT10 forward, REDFT01/RODFT01 backward
//==============================================================================
using BC = BoundaryCondition;

/// The CUDA transform of a real field between walls: forward (REDFT10 or RODFT10 per axis,
/// the type 2) or backward (REDFT01 or RODFT01, the type 3), times `scale`
std::vector<double> wall_transform(const std::vector<int>& nx, const std::vector<BC>& bc, bool forward,
                                   const std::vector<double>& in, double scale = 1.0)
{
    CudaMakhoulTransform<double> transform(nx, bc);
    CudaDeviceBuffer<double> d_in(in.size()), d_out(in.size());
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), sizeof(double) * in.size(), cudaMemcpyHostToDevice));
    if (forward)
        transform.forward(d_in, d_out, scale, 0);
    else
        transform.backward(d_in, nullptr, d_out, scale, 0);
    std::vector<double> out(in.size());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost));
    return out;
}

/// The backward transform of forward coefficients with the scale 1/(2 n) per axis: the inverse
std::vector<double> wall_inverse(const std::vector<int>& nx, const std::vector<BC>& bc,
                                 const std::vector<double>& coefficients)
{
    double backward_scale = 1.0;
    for (int n : nx)
        backward_scale /= 2.0 * n;
    return wall_transform(nx, bc, false, coefficients, backward_scale);
}

/// max |a - b|, NaN if a value is NaN
double max_abs_difference(const std::vector<double>& a, const std::vector<double>& b)
{
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        error = max_or_nan(error, std::abs(a[i] - b[i]));
    return error;
}

/// REDFT10 (reflecting) or RODFT10 (absorbing) of x along one axis by its sum, axes padded to
/// three in front (a padded axis has the factor 1)
std::vector<double> wall_direct_sum(const std::vector<int>& nx, const std::vector<BC>& bc, const std::vector<double>& x)
{
    const auto basis = [](BC kind, int k, int n, int len) {
        const double pi = std::numbers::pi;
        return (kind == BC::REFLECTING) ? 2.0 * std::cos(pi * k * (2 * n + 1) / (2.0 * len))
                                        : 2.0 * std::sin(pi * (k + 1) * (2 * n + 1) / (2.0 * len));
    };
    const int pad = 3 - static_cast<int>(nx.size());
    std::array<int, 3> n{1, 1, 1};
    std::array<BC, 3> kind{BC::REFLECTING, BC::REFLECTING, BC::REFLECTING};
    for (size_t d = 0; d < nx.size(); ++d)
    {
        n[d + pad] = nx[d];
        kind[d + pad] = bc[d];
    }
    const auto factor = [&](int d, int k, int i) { return d < pad ? 1.0 : basis(kind[d], k, i, n[d]); };
    std::vector<double> ref(x.size(), 0.0);
    for (int k0 = 0; k0 < n[0]; ++k0)
        for (int k1 = 0; k1 < n[1]; ++k1)
            for (int k2 = 0; k2 < n[2]; ++k2)
            {
                double sum = 0.0;
                for (int i0 = 0; i0 < n[0]; ++i0)
                    for (int i1 = 0; i1 < n[1]; ++i1)
                        for (int i2 = 0; i2 < n[2]; ++i2)
                            sum += x[(i0 * n[1] + i1) * n[2] + i2]
                                 * factor(0, k0, i0) * factor(1, k1, i1) * factor(2, k2, i2);
                ref[(k0 * n[1] + k1) * n[2] + k2] = sum;
            }
    return ref;
}

/// Odd lengths along the walls: the forward transform against the direct sums of the series and,
/// with a CPU backend, against MakhoulTransform; the round trip. One normal(0, 5) stream of
/// mt19937(7) gives the fields of the grids in this order.
void check_odd_walls(Checks& checks, int& expected_runs)
{
    const std::vector<std::tuple<std::string, std::vector<int>, std::vector<BC>>> grids = {
        {"1D 11 R", {11}, {BC::REFLECTING}},
        {"3D 6x4x3 R/A/R", {6, 4, 3}, {BC::REFLECTING, BC::ABSORBING, BC::REFLECTING}},
        {"2D 6x5 R/A", {6, 5}, {BC::REFLECTING, BC::ABSORBING}},
    };
    expected_runs += static_cast<int>(grids.size());
    std::mt19937 gen0(7);
    std::normal_distribution<double> normal(0.0, 5.0);
    for (const auto& [name, nx, bc] : grids)
    {
        checks.begin_run("Odd walls, " + name);
        int total = 1;
        for (int n : nx)
            total *= n;
        std::vector<double> x(total);
        for (double& v : x)
            v = normal(gen0);
        const std::vector<double> ref = wall_direct_sum(nx, bc, x);
        const std::vector<double> coefficients = wall_transform(nx, bc, true, x);
        double scale = 0.0;
        for (double v : ref)
            scale = std::max(scale, std::abs(v));
        checks.close("Forward against the direct sum (relative)", max_abs_difference(coefficients, ref) / scale, 1e-12);
        checks.close("Round trip", max_abs_difference(wall_inverse(nx, bc, coefficients), x), 1e-12);
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::vector<double> cpu(total);
        cpu_reference(nx, bc)->forward(x.data(), cpu.data(), 1.0);
        checks.close("Forward against MakhoulTransform (relative)", max_abs_difference(coefficients, cpu) / scale, 1e-13);
#endif
        checks.end_run();
    }
}

/// backward(forward(x)) = x along reflecting and absorbing walls, and the periodic round trip of
/// CudaFourierTransform
void check_round_trips(Checks& checks, int& expected_runs)
{
    // The ramp 1..12, then one uniform(0, 1) stream of mt19937(42) drawn in this order: 12 values
    // for both walls, 48 for the 8x6 grid, 72 for the 6x4x3 grid
    std::vector<double> ramp(12);
    for (int i = 0; i < 12; ++i)
        ramp[i] = i + 1;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::vector<double> random_line(12), random_2d(8 * 6), random_3d(6 * 4 * 3);
    for (std::vector<double>* values : {&random_line, &random_2d, &random_3d})
        for (double& v : *values)
            v = dist(gen);

    const std::vector<std::tuple<std::string, std::vector<int>, std::vector<BC>, const std::vector<double>*>> fields = {
        {"DCT-2/3, 1..12", {12}, {BC::REFLECTING}, &ramp},
        {"DST-2/3, 1..12", {12}, {BC::ABSORBING}, &ramp},
        {"DCT-2/3, random 12", {12}, {BC::REFLECTING}, &random_line},
        {"DST-2/3, random 12", {12}, {BC::ABSORBING}, &random_line},
        {"2D 8x6 R/A, random", {8, 6}, {BC::REFLECTING, BC::ABSORBING}, &random_2d},
        {"3D 6x4x3 R/A/R, random", {6, 4, 3}, {BC::REFLECTING, BC::ABSORBING, BC::REFLECTING}, &random_3d},
    };
    expected_runs += static_cast<int>(fields.size()) + 1;
    for (const auto& [name, nx, bc, x] : fields)
    {
        checks.begin_run("Round trip, " + name);
        const std::vector<double> back = wall_inverse(nx, bc, wall_transform(nx, bc, true, *x));
        checks.close("Round trip", max_abs_difference(back, *x), 1e-7);
        checks.end_run();
    }

    // The periodic transform of CudaFourierTransform (cuFFT D2Z/Z2D)
    checks.begin_run("Round trip, periodic CudaFourierTransform 8x6x10");
    const std::vector<int> nx_p = {8, 6, 10};
    const int MP = nx_p[0] * nx_p[1] * nx_p[2];
    ComputationBox<double> cb(nx_p, {1.0, 1.0, 1.0}, std::vector<std::string>(6, "periodic"));
    CudaFourierTransform<double> fft_p(&cb, std::vector<BC>(3, BC::PERIODIC));
    std::vector<double> data_p(MP), data_p_back(MP);
    for (int i = 0; i < MP; ++i)
        data_p[i] = std::sin(0.37 * i) + 0.1 * i / MP;
    std::vector<double> coefficients(2 * nx_p[0] * nx_p[1] * (nx_p[2] / 2 + 1));
    fft_p.forward(data_p.data(), 1, coefficients.data());
    fft_p.backward(coefficients.data(), 1, data_p_back.data());
    checks.close("Round trip", max_abs_difference(data_p_back, data_p), 1e-12);
    checks.end_run();
}

#ifdef USE_CPU_FFTW
//==============================================================================
// Against FFTW's r2r transforms
//==============================================================================
/// A family of transforms: the cosine (DCT) or the sine (DST) transforms of types 2 and 3
struct Family
{
    const char* name;
    bool cosine;
    fftw_r2r_kind kind2, kind3;

    fftw_r2r_kind kind(int order) const { return order == 2 ? kind2 : kind3; }
    BC wall() const { return cosine ? BC::REFLECTING : BC::ABSORBING; }
    /// The function of the family: cos for the DCT, sin for the DST
    double g(double x) const { return cosine ? cos(x) : sin(x); }
};

static const Family DCT{"DCT", true, FFTW_REDFT10, FFTW_REDFT01};
static const Family DST{"DST", false, FFTW_RODFT10, FFTW_RODFT01};

/// One check: a grid (1 to 3 axes), the family of each axis and the order (2 or 3) of the transform
struct Case
{
    std::vector<int> n;
    std::vector<const Family*> family;
    int order;
};

/// The cases of grids of one family and one type
static std::vector<Case> of_type(const Family& f, int order, const std::vector<std::vector<int>>& grids)
{
    std::vector<Case> cases;
    for (const std::vector<int>& n : grids)
        cases.push_back({n, std::vector<const Family*>(n.size(), &f), order});
    return cases;
}

static bool one_family(const Case& c)
{
    for (const Family* f : c.family)
        if (f != c.family[0])
            return false;
    return true;
}

/// 1D input: sin(pi u) + g(2 pi u)/2 at the points u of the type, (i + 1/2)/N for type 2,
/// i/N (DCT) or (i + 1)/(N + 1) (DST) for type 3
static double input_1d(const Family& f, int order, int i, int N)
{
    if (order == 2)
        return sin(std::numbers::pi * (i + 0.5) / N) + 0.5 * f.g(2 * std::numbers::pi * (i + 0.5) / N);
    if (f.cosine)
        return sin(std::numbers::pi * i / N) + 0.5 * f.g(2 * std::numbers::pi * i / N);
    return sin(std::numbers::pi * (i + 1) / (N + 1)) + 0.5 * f.g(2 * std::numbers::pi * (i + 1) / (N + 1));
}

/// 2D and 3D input of one family: a Gaussian exp(-r^2/50) for the DCT (r from the points
/// i + 1/2 for type 2, i for type 3), a product of sines for the DST (at (i + 1/2)/N for
/// type 2, (i + 1)/(N + 1) for type 3)
static double input_one_type(const Family& f, int order, const int* i, const std::vector<int>& n)
{
    const int dim = static_cast<int>(n.size());
    if (f.cosine)
    {
        double r2 = 0.0;
        for (int d = 0; d < dim; ++d)
        {
            const double x = (order == 2) ? i[d] + 0.5 : i[d];
            r2 += x * x;
        }
        return exp(-r2 / 50.0);
    }
    double value = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        const double x = (order == 2) ? (i[d] + 0.5) / n[d] : (i[d] + 1.0) / (n[d] + 1);
        value *= sin(std::numbers::pi * x);
    }
    return value;
}

/// Input of mixed families, with x = (i + 1/2)/N on every axis: the product over the axes of
/// sin(pi x) + g(2 pi x)/2, g the function of the axis
static double input_mixed(const Case& c, const int* i)
{
    double value = 1.0;
    for (size_t d = 0; d < c.n.size(); ++d)
    {
        const double x = (i[d] + 0.5) / c.n[d];
        value *= sin(std::numbers::pi * x) + 0.5 * c.family[d]->g(2 * std::numbers::pi * x);
    }
    return value;
}

static std::vector<double> input_of(const Case& c)
{
    const int dim = static_cast<int>(c.n.size());
    int M = 1;
    for (int nd : c.n)
        M *= nd;
    std::vector<double> input(M);
    for (int m = 0; m < M; ++m)
    {
        // Row-major index: the last axis is contiguous
        int i[3] = {0, 0, 0};
        int rest = m;
        for (int d = dim - 1; d >= 0; --d)
        {
            i[d] = rest % c.n[d];
            rest /= c.n[d];
        }
        if (dim == 1)
            input[m] = input_1d(*c.family[0], c.order, i[0], c.n[0]);
        else if (one_family(c))
            input[m] = input_one_type(*c.family[0], c.order, i, c.n);
        else
            input[m] = input_mixed(c, i);
    }
    return input;
}

static std::vector<double> fftw_transform(const Case& c, std::vector<double> data)
{
    fftw_r2r_kind kinds[3];
    for (size_t d = 0; d < c.n.size(); ++d)
        kinds[d] = c.family[d]->kind(c.order);
    fftw_plan plan = fftw_plan_r2r(static_cast<int>(c.n.size()), c.n.data(), data.data(), data.data(),
                                   kinds, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    return data;
}

/// The CUDA transform of a case with scale 1: forward for type 2, backward for type 3
static std::vector<double> cuda_transform(const Case& c, const std::vector<double>& data)
{
    std::vector<BC> bc;
    for (const Family* f : c.family)
        bc.push_back(f->wall());
    return wall_transform(c.n, bc, c.order == 2, data);
}

/// max |a - b| / max |b|, NaN if a value is NaN
static double relative_error(const std::vector<double>& a, const std::vector<double>& b)
{
    return max_abs_difference(a, b) / max_abs_or_nan(b);
}

static std::string describe(const Case& c)
{
    std::string text = std::to_string(c.n.size()) + "D ";
    for (size_t d = 0; d < c.n.size(); ++d)
        text += (d > 0 ? " x " : "") + std::string(c.family[d]->name) + "-" + std::to_string(c.order);
    text += ":";
    for (size_t d = 0; d < c.n.size(); ++d)
        text += (d > 0 ? " x " : " ") + std::to_string(c.n[d]);
    return text;
}

/// The cases of mixed families per axis, of both types
static std::vector<Case> mixed_cases()
{
    std::vector<Case> cases;
    for (int order : {2, 3})
    {
        cases.push_back({{16, 16}, {&DCT, &DST}, order});
        cases.push_back({{16, 16}, {&DST, &DCT}, order});
        cases.push_back({{8, 8, 8}, {&DCT, &DST, &DCT}, order});
        cases.push_back({{8, 8, 8}, {&DST, &DCT, &DST}, order});
        cases.push_back({{8, 8, 8}, {&DST, &DST, &DCT}, order});
        // Odd lengths along both families
        cases.push_back({{7, 9}, {&DCT, &DST}, order});
        cases.push_back({{5, 7, 9}, {&DST, &DCT, &DST}, order});
    }
    return cases;
}

static std::vector<Case> join(std::vector<std::vector<Case>> lists)
{
    std::vector<Case> all;
    for (const auto& list : lists)
        all.insert(all.end(), list.begin(), list.end());
    return all;
}

/// The CUDA transforms of every case against FFTW, and the type-2/type-3 round trips of both
void check_against_fftw(Checks& checks, int& expected_runs)
{
    const std::vector<Case> cases = join({
        // DCT (FFTW REDFT10/01). 1D: odd lengths, and beyond the 6144 points of the earlier
        // shared-memory transforms
        of_type(DCT, 2, {{16}, {32}, {64}, {15}, {31}, {5001}, {6144}, {6145}}),
        of_type(DCT, 3, {{16}, {32}, {64}, {15}, {5001}, {6142}, {6143}}),
        of_type(DCT, 2, {{8, 8}, {16, 16}, {32, 32}, {6, 5}, {5, 6}, {7, 9}}),
        of_type(DCT, 3, {{8, 8}, {16, 16}, {32, 32}, {7, 9}, {6, 5, 7}}),
        // (6, 6, 5): odd physical z of the logical 12x12x10 CrysFFT grid
        of_type(DCT, 2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 6, 5}, {6, 5, 6}, {5, 6, 6}, {5, 7, 9}}),
        of_type(DCT, 3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {5, 7, 9}}),
        // DST (FFTW RODFT10/01): the odd lengths of every type and dimension, which the earlier
        // transforms refused
        of_type(DST, 2, {{16}, {32}, {64}, {15}}),
        of_type(DST, 3, {{16}, {32}, {64}, {15}}),
        of_type(DST, 2, {{8, 8}, {16, 16}, {32, 32}, {6, 5}}),
        of_type(DST, 3, {{8, 8}, {16, 16}, {32, 32}, {5, 6}}),
        of_type(DST, 2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 6, 5}}),
        of_type(DST, 3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 5, 6}}),
        // DCT and DST per axis
        mixed_cases()});
    const std::vector<std::pair<const Family*, int>> round_trips = {{&DCT, 32}, {&DCT, 15}, {&DST, 32}, {&DST, 15}};
    expected_runs += static_cast<int>(cases.size() + round_trips.size());

    for (const Case& c : cases)
    {
        checks.begin_run("Against FFTW, " + describe(c));
        const std::vector<double> input = input_of(c);
        checks.close("Relative", relative_error(cuda_transform(c, input), fftw_transform(c, input)), 1e-10);
        checks.end_run();
    }

    // Type 2 then type 3 recovers the input times 2N, with FFTW and with CUDA
    for (const auto& [f, N] : round_trips)
    {
        checks.begin_run(std::string(f->name) + "-2 + " + f->name + "-3 round trip, N = " + std::to_string(N));
        const Case forward{{N}, {f}, 2}, backward{{N}, {f}, 3};
        const std::vector<double> input = input_of(forward);
        for (const bool cuda : {false, true})
        {
            std::vector<double> result = cuda ? cuda_transform(backward, cuda_transform(forward, input))
                                              : fftw_transform(backward, fftw_transform(forward, input));
            const double norm = 1.0 / (2.0 * N);
            for (double& v : result)
                v *= norm;
            checks.close(cuda ? "CUDA, relative" : "FFTW, relative", relative_error(result, input), 1e-12);
        }
        checks.end_run();
    }
}
#endif

int main()
{
    try
    {
        const double tolerance = 1e-13;
        std::mt19937 rng(17);
        GridErrors worst_real, worst_complex;
        int grids = 0;
        auto add = [](GridErrors& worst, const GridErrors& e) {
            worst.definition = max_or_nan(worst.definition, e.definition);
            worst.cpu = max_or_nan(worst.cpu, e.cpu);
            worst.inputs_kept = worst.inputs_kept && e.inputs_kept;
            worst.k_factor_exact = worst.k_factor_exact && e.k_factor_exact;
        };
        // The grids of TestCpuMakhoulTransform and a few larger ones, whose kernels loop over several blocks
        std::vector<std::vector<int>> grid_list = mixed_grids();
        grid_list.insert(grid_list.end(), {{96, 75}, {24, 20, 33}, {32, 31, 16}});
        for (const std::vector<int>& nx : grid_list)
        {
            for (const std::vector<BoundaryCondition>& bc : mixed_conditions(static_cast<int>(nx.size())))
            {
                const auto cpu = cpu_reference(nx, bc);
                add(worst_real, check_grid<double>(nx, bc, rng, cpu.get()));
                add(worst_complex, check_grid<Complex>(nx, bc, rng, cpu.get()));
                ++grids;
            }
        }
        std::cout << "CudaMakhoulTransform (" << grids << " grids) against the definitions: real "
                  << worst_real.definition << ", complex " << worst_complex.definition << std::endl;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::cout << "CudaMakhoulTransform against MakhoulTransform: real " << worst_real.cpu << ", complex "
                  << worst_complex.cpu << std::endl;
#endif

        // Walls only: the grids of TestCpuMakhoulTransform with every mix of reflecting and absorbing
        // axes, and a few larger ones whose kernels loop over several blocks (fields from a generator
        // of their own)
        std::mt19937 wall_rng(23);
        GridErrors walls_real, walls_complex;
        int wall_grids = 0;
        const std::vector<std::vector<int>> wall_grid_list = {
            {1}, {2}, {3}, {4}, {5}, {7}, {8}, {16}, {17},
            {1, 1}, {1, 6}, {6, 1}, {2, 2}, {4, 6}, {5, 7}, {3, 8}, {7, 2}, {12, 10},
            {1, 1, 1}, {2, 2, 2}, {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {4, 1, 6}, {1, 5, 6}, {9, 8, 7},
            {16, 12, 10}, {96, 75}, {24, 20, 33}, {32, 31, 16}};
        for (const std::vector<int>& nx : wall_grid_list)
        {
            const int dim = static_cast<int>(nx.size());
            for (int mask = 0; mask < (1 << dim); ++mask)
            {
                std::vector<BoundaryCondition> bc(dim);
                for (int d = 0; d < dim; ++d)
                    bc[d] = ((mask >> d) & 1) ? BoundaryCondition::ABSORBING : BoundaryCondition::REFLECTING;
                const auto cpu = cpu_reference(nx, bc);
                add(walls_real, check_grid<double>(nx, bc, wall_rng, cpu.get()));
                add(walls_complex, check_grid<Complex>(nx, bc, wall_rng, cpu.get()));
                ++wall_grids;
            }
        }
        std::cout << "CudaMakhoulTransform, walls only (" << wall_grids << " grids), against the definitions: real "
                  << walls_real.definition << ", complex " << walls_complex.definition << std::endl;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::cout << "  against MakhoulTransform: real " << walls_real.cpu << ", complex " << walls_complex.cpu
                  << std::endl;
#endif

        // Grids whose real half spectrum, too, has more than MAX_GRID_THREADS modes: every
        // grid-stride loop takes more than one stride (the fields still come from rng). Both walls,
        // even and odd, last axes periodic and wall; lengths with small factors keep the measured
        // FFTW plans short.
        using BC = BoundaryCondition;
        GridErrors large_real, large_complex;
        bool beyond_one_stride = true;
        for (const auto& [nx, bc] : std::vector<std::pair<std::vector<int>, std::vector<BoundaryCondition>>>{
                 {{1215, 1755}, {BC::REFLECTING, BC::PERIODIC}},
                 {{135, 128, 125}, {BC::PERIODIC, BC::ABSORBING, BC::REFLECTING}},
                 {{125, 135, 128}, {BC::ABSORBING, BC::REFLECTING, BC::ABSORBING}}})
        {
            long long points = 1;
            for (const int n : nx)
                points *= n;
            beyond_one_stride = beyond_one_stride && points / nx.back() * (nx.back() / 2 + 1) > MAX_GRID_THREADS;
            const auto cpu = cpu_reference(nx, bc);
            add(large_real, check_grid<double>(nx, bc, rng, cpu.get(), 16));
            add(large_complex, check_grid<Complex>(nx, bc, rng, cpu.get(), 16));
        }
        std::cout << "CudaMakhoulTransform (3 grids beyond one stride: " << (beyond_one_stride ? "yes" : "no")
                  << ") against the definition on samples: real " << large_real.definition << ", complex "
                  << large_complex.definition << std::endl;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::cout << "  against MakhoulTransform: real " << large_real.cpu << ", complex " << large_complex.cpu
                  << std::endl;
#endif
        const bool inputs_kept = worst_real.inputs_kept && worst_complex.inputs_kept && walls_real.inputs_kept
                                 && walls_complex.inputs_kept && large_real.inputs_kept && large_complex.inputs_kept;
        std::cout << "Inputs kept: " << (inputs_kept ? "yes" : "no") << std::endl;
        const bool k_factor_exact = worst_real.k_factor_exact && worst_complex.k_factor_exact
                                    && walls_real.k_factor_exact && walls_complex.k_factor_exact
                                    && large_real.k_factor_exact && large_complex.k_factor_exact;
        std::cout << "Backward with factors equals the backward of the products bit for bit: "
                  << (k_factor_exact ? "yes" : "no") << std::endl;

        const bool ok = worst_real.definition < tolerance && worst_complex.definition < tolerance
                        && worst_real.cpu < tolerance && worst_complex.cpu < tolerance
                        && walls_real.definition < tolerance && walls_complex.definition < tolerance
                        && walls_real.cpu < tolerance && walls_complex.cpu < tolerance
                        && large_real.definition < tolerance && large_complex.definition < tolerance
                        && large_real.cpu < tolerance && large_complex.cpu < tolerance
                        && beyond_one_stride && inputs_kept && k_factor_exact;
        std::cout << "Against the definitions and MakhoulTransform: " << (ok ? "PASSED" : "FAILED") << std::endl;

        // The unnormalized transforms between walls and the round trips, each one counted run
        Checks checks;
        int expected_runs = 0;
        check_odd_walls(checks, expected_runs);
        check_round_trips(checks, expected_runs);
#ifdef USE_CPU_FFTW
        check_against_fftw(checks, expected_runs);
#endif
        const bool runs_ok = checks.passed(expected_runs);
        return ok && runs_ok ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
