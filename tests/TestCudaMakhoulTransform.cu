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

template <typename T>
GridErrors check_grid(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, std::mt19937& rng)
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
    std::vector<Complex> expected = mixed_definition(field, nx, bc, true);
    errors.definition = max_difference(forward, expected) / max_abs(expected);

    gpu.backward(forward, round_trip, backward_scale, kept);
    errors.inputs_kept = errors.inputs_kept && kept == forward;
    errors.definition = max_or_nan(errors.definition, max_difference(round_trip, field) / max_abs(field));

    gpu.backward(coefficients, backward, backward_scale, kept);
    errors.inputs_kept = errors.inputs_kept && kept == coefficients;
    expected = mixed_definition(coefficients, nx, bc, false);
    if (!complex_field)
        for (Complex& x : expected)
            x = x.real();
    errors.definition = max_or_nan(errors.definition, max_difference(backward, expected) / max_abs(expected));

#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#ifdef USE_CPU_FFTW
    const FFTBackend backend = FFTBackend::FFTW;
#else
    const FFTBackend backend = FFTBackend::MKL;
#endif
    const MakhoulTransform cpu(nx, bc, createRealFFT(nx, backend));
    std::vector<Complex> cpu_forward(total), cpu_backward(total);
    std::vector<double> real_in(total), real_out(total);
    if (complex_field)
    {
        cpu.forward(field.data(), cpu_forward.data(), forward_scale);
        cpu.backward(coefficients.data(), cpu_backward.data(), backward_scale);
    }
    else
    {
        for (int i = 0; i < total; ++i)
            real_in[i] = field[i].real();
        cpu.forward(real_in.data(), cpu_forward.data(), forward_scale);
        cpu.backward(coefficients.data(), real_out.data(), backward_scale);
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
                add(worst_real, check_grid<double>(nx, bc, rng));
                add(worst_complex, check_grid<Complex>(nx, bc, rng));
                ++grids;
            }
        }
        std::cout << "CudaMakhoulTransform (" << grids << " grids) against the definitions: real "
                  << worst_real.definition << ", complex " << worst_complex.definition << std::endl;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
        std::cout << "CudaMakhoulTransform against MakhoulTransform: real " << worst_real.cpu << ", complex "
                  << worst_complex.cpu << std::endl;
#endif
        std::cout << "Inputs kept: " << (worst_real.inputs_kept && worst_complex.inputs_kept ? "yes" : "no") << std::endl;

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
                        && worst_real.inputs_kept && worst_complex.inputs_kept && refused;
        return ok ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
