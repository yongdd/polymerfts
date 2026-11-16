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
 * @file TestCudaMakhoulTransform.cu
 * @brief CudaMakhoulTransform against the O(N^2) definitions and the CPU MakhoulTransform.
 *
 * Boxes that mix periodic axes and walls (CudaFourierTransform) run one cuFFT of
 * the grid with Makhoul's reordering and combining on the wall axes. For every mix
 * of periodic, reflecting and absorbing axes on grids of two and three axes with at
 * least one periodic axis and one wall, with even, odd and unit lengths, and on a
 * few larger grids, real and complex fields of standard deviation 5 must match the
 * definition axis by axis, as TestCpuMakhoulTransform checks the CPU version:
 *
 * - forward: the DFT exp(-2 pi i m j / n) along periodic axes, REDFT10 or RODFT10 / 2 along walls
 * - backward: the inverse DFT / n along periodic axes, REDFT01 or RODFT01 / n along walls,
 *   from random complex coefficients without symmetry; a real field is the real part
 * - backward(forward(f)) = f
 *
 * The forward and backward transforms keep their inputs, and with a CPU backend
 * they agree with MakhoulTransform over its real FFT (the largest relative
 * difference is printed). The error maxima keep a NaN (TestNanSafe.h).
 *
 * On two grids of more than two million points, whose real half spectrum also
 * has more than MAX_GRID_THREADS modes, every grid-stride loop of the kernels takes
 * more than one stride. There the definition is summed for the corners and 16
 * random points of every result, and with a CPU backend MakhoulTransform checks
 * the whole results.
 */

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "CudaCommon.h"
#include "CudaMakhoulTransform.h"
#include "Exception.h"
#include "TestNanSafe.h"
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#include "FFTFactory.h"
#include "MakhoulTransform.h"
#endif

using Complex = std::complex<double>;

/// One wall axis of the definition: apply to every line of `data` along `axis`.
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

/// The definition axis by axis: the DFT along periodic axes, apply_axis() on the parts along walls
std::vector<Complex> mixed_definition(std::vector<Complex> data, const std::vector<int>& nx,
                                      const std::vector<BoundaryCondition>& bc, bool forward)
{
    const double pi = std::numbers::pi;
    const int size = static_cast<int>(data.size());
    for (int axis = 0; axis < static_cast<int>(nx.size()); ++axis)
    {
        if (bc[axis] != BoundaryCondition::PERIODIC)
        {
            // The series are real: the real and imaginary parts in turn
            std::vector<double> re(size), im(size);
            for (int i = 0; i < size; ++i)
            {
                re[i] = data[i].real();
                im[i] = data[i].imag();
            }
            const bool sine = bc[axis] == BoundaryCondition::ABSORBING;
            apply_axis(re, nx, axis, sine, forward);
            apply_axis(im, nx, axis, sine, forward);
            for (int i = 0; i < size; ++i)
                data[i] = Complex(re[i], im[i]);
            continue;
        }
        const int n = nx[axis];
        int stride = 1;
        for (size_t d = axis + 1; d < nx.size(); ++d)
            stride *= nx[d];
        std::vector<Complex> line(n);
        for (int i = 0; i < size; ++i)
        {
            if ((i / stride) % n != 0)
                continue;
            for (int j = 0; j < n; ++j)
                line[j] = data[i + j * stride];
            for (int k = 0; k < n; ++k)
            {
                Complex sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += line[j] * std::polar(1.0, (forward ? -2.0 : 2.0) * pi * static_cast<double>(j * k % n) / n);
                data[i + k * stride] = forward ? sum : sum / static_cast<double>(n);
            }
        }
    }
    return data;
}

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

double max_difference(const std::vector<Complex>& a, const std::vector<Complex>& b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = max_or_nan(worst, std::abs(a[i] - b[i]));
    return worst;
}

double max_abs(const std::vector<Complex>& a)
{
    double worst = 0.0;
    for (const Complex& x : a)
        worst = max_or_nan(worst, std::abs(x));
    return worst;
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
};

/// The CUDA transform of host arrays: a real field lives in the real parts of `in`
template <typename T>
struct DeviceTransform
{
    using DeviceT = CuDeviceData<T>;
    CudaMakhoulTransform<T> transform;
    int total;
    CudaDeviceBuffer<DeviceT> d_field;
    CudaDeviceBuffer<cuDoubleComplex> d_coefficients;

    DeviceTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, int n)
        : transform(nx, bc), total(n), d_field(n), d_coefficients(n) {}

    /// Upload, transform and download; `in_after` receives the device input after the call
    void forward(const std::vector<Complex>& in, std::vector<Complex>& out, double scale, std::vector<Complex>& in_after)
    {
        upload_field(in);
        transform.forward(d_field, d_coefficients, scale, 0);
        CUDA_CHECK(cudaMemcpy(out.data(), d_coefficients, sizeof(Complex) * total, cudaMemcpyDeviceToHost));
        download_field(in_after);
    }

    void backward(const std::vector<Complex>& in, std::vector<Complex>& out, double scale, std::vector<Complex>& in_after)
    {
        CUDA_CHECK(cudaMemcpy(d_coefficients, in.data(), sizeof(Complex) * total, cudaMemcpyHostToDevice));
        transform.backward(d_coefficients, d_field, scale, 0);
        download_field(out);
        CUDA_CHECK(cudaMemcpy(in_after.data(), d_coefficients, sizeof(Complex) * total, cudaMemcpyDeviceToHost));
    }

private:
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
    DeviceTransform<T> gpu(nx, bc, total);
    // As CudaFourierTransform: REDFT10 and RODFT10 are twice the series
    const double forward_scale = 1.0 / (1 << walls);
    const double backward_scale = 1.0 / total;

    // A real field is stored in the real parts; the coefficients have no symmetry
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<Complex> field(total), coefficients(total);
    for (Complex& x : field)
        x = complex_field ? Complex(normal(rng), normal(rng)) : Complex(normal(rng), 0.0);
    for (Complex& x : coefficients)
        x = Complex(normal(rng), normal(rng));

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

#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
    std::vector<Complex> cpu_forward(total), cpu_backward(total);
    std::vector<double> real_in(total), real_out(total);
    if (complex_field)
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

int main()
{
    try
    {
        const double tolerance = 1e-13;
        const BoundaryCondition kinds[3] = {BoundaryCondition::PERIODIC, BoundaryCondition::REFLECTING,
                                            BoundaryCondition::ABSORBING};
        std::mt19937 rng(17);
        GridErrors worst_real, worst_complex;
        int grids = 0;
        auto add = [](GridErrors& worst, const GridErrors& e) {
            worst.definition = max_or_nan(worst.definition, e.definition);
            worst.cpu = max_or_nan(worst.cpu, e.cpu);
            worst.inputs_kept = worst.inputs_kept && e.inputs_kept;
        };
        // The grids of TestCpuMakhoulTransform and a few larger ones, whose kernels loop over several blocks
        for (const std::vector<int>& nx : std::vector<std::vector<int>>{
                 {2, 3}, {4, 6}, {5, 7}, {8, 1}, {1, 8}, {6, 6}, {7, 2},
                 {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {1, 5, 6}, {9, 8, 7}, {6, 5, 4},
                 {96, 75}, {24, 20, 33}, {32, 31, 16}})
        {
            const int dim = static_cast<int>(nx.size());
            const int mixes = dim == 2 ? 9 : 27;
            for (int mix = 0; mix < mixes; ++mix)
            {
                std::vector<BoundaryCondition> bc(dim);
                int periodic = 0;
                for (int d = 0, m = mix; d < dim; ++d, m /= 3)
                {
                    bc[d] = kinds[m % 3];
                    periodic += bc[d] == BoundaryCondition::PERIODIC;
                }
                if (periodic == 0 || periodic == dim)
                    continue;
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

        // Grids whose real half spectrum, too, has more than MAX_GRID_THREADS modes: every
        // grid-stride loop takes more than one stride (the fields still come from rng). Both walls,
        // even and odd, last axes periodic and wall; lengths with small factors keep the measured
        // FFTW plans short.
        using BC = BoundaryCondition;
        GridErrors large_real, large_complex;
        bool beyond_one_stride = true;
        for (const auto& [nx, bc] : std::vector<std::pair<std::vector<int>, std::vector<BoundaryCondition>>>{
                 {{1215, 1755}, {BC::REFLECTING, BC::PERIODIC}},
                 {{135, 128, 125}, {BC::PERIODIC, BC::ABSORBING, BC::REFLECTING}}})
        {
            long long points = 1;
            for (const int n : nx)
                points *= n;
            beyond_one_stride = beyond_one_stride && points / nx.back() * (nx.back() / 2 + 1) > MAX_GRID_THREADS;
            const auto cpu = cpu_reference(nx, bc);
            add(large_real, check_grid<double>(nx, bc, rng, cpu.get(), 16));
            add(large_complex, check_grid<Complex>(nx, bc, rng, cpu.get(), 16));
        }
        std::cout << "CudaMakhoulTransform (2 grids beyond one stride: " << (beyond_one_stride ? "yes" : "no")
                  << ") against the definition on samples: real " << large_real.definition << ", complex "
                  << large_complex.definition << std::endl;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::cout << "  against MakhoulTransform: real " << large_real.cpu << ", complex " << large_complex.cpu
                  << std::endl;
#endif
        const bool inputs_kept = worst_real.inputs_kept && worst_complex.inputs_kept && large_real.inputs_kept
                                 && large_complex.inputs_kept;
        std::cout << "Inputs kept: " << (inputs_kept ? "yes" : "no") << std::endl;

        // A box without a periodic axis has real coefficients: refused
        bool refused = false;
        try
        {
            CudaMakhoulTransform<double> walls({4, 6}, {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING});
        }
        catch (std::exception&)
        {
            refused = true;
        }
        std::cout << "Walls only refused: " << (refused ? "yes" : "no") << std::endl;

        const bool ok = worst_real.definition < tolerance && worst_complex.definition < tolerance
                        && worst_real.cpu < tolerance && worst_complex.cpu < tolerance
                        && large_real.definition < tolerance && large_complex.definition < tolerance
                        && large_real.cpu < tolerance && large_complex.cpu < tolerance
                        && beyond_one_stride && inputs_kept && refused;
        return ok ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
