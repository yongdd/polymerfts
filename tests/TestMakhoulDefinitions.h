/**
 * @file TestMakhoulDefinitions.h
 * @brief The O(N^2) definitions that TestCpuMakhoulTransform and TestCudaMakhoulTransform
 *        compare the transforms with, and the boxes that mix periodic axes and walls.
 *
 * - forward: the DFT exp(-2 pi i m j / n) along periodic axes, REDFT10 or RODFT10 / 2 along walls
 * - backward: the inverse DFT / n along periodic axes, REDFT01 or RODFT01 / n along walls
 *
 * The error maxima keep a NaN (TestNanSafe.h).
 */

#ifndef TEST_MAKHOUL_DEFINITIONS_H_
#define TEST_MAKHOUL_DEFINITIONS_H_

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "ComputationBox.h"
#include "TestNanSafe.h"

using Complex = std::complex<double>;

/// One wall axis of the definition: apply to every line of `data` along `axis`.
inline void apply_axis(std::vector<double>& data, const std::vector<int>& nx, int axis, bool sine, bool forward)
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
inline std::vector<Complex> mixed_definition(std::vector<Complex> data, const std::vector<int>& nx,
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

inline double max_difference(const std::vector<Complex>& a, const std::vector<Complex>& b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = max_or_nan(worst, std::abs(a[i] - b[i]));
    return worst;
}

inline double max_abs(const std::vector<Complex>& a)
{
    double worst = 0.0;
    for (const Complex& x : a)
        worst = max_or_nan(worst, std::abs(x));
    return worst;
}

/// Grids of two and three axes with even, odd and unit lengths for the boxes that mix periodic axes and walls
inline std::vector<std::vector<int>> mixed_grids()
{
    return {{2, 3}, {4, 6}, {5, 7}, {8, 1}, {1, 8}, {6, 6}, {7, 2},
            {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {1, 5, 6}, {9, 8, 7}, {6, 5, 4}};
}

/// Every mix of periodic, reflecting and absorbing axes with at least one periodic axis and one wall,
/// the first axis changing fastest
inline std::vector<std::vector<BoundaryCondition>> mixed_conditions(int dim)
{
    const BoundaryCondition kinds[3] = {BoundaryCondition::PERIODIC, BoundaryCondition::REFLECTING,
                                        BoundaryCondition::ABSORBING};
    std::vector<std::vector<BoundaryCondition>> mixes;
    const int count = dim == 2 ? 9 : 27;
    for (int mix = 0; mix < count; ++mix)
    {
        std::vector<BoundaryCondition> bc(dim);
        int periodic = 0;
        for (int d = 0, m = mix; d < dim; ++d, m /= 3)
        {
            bc[d] = kinds[m % 3];
            periodic += bc[d] == BoundaryCondition::PERIODIC;
        }
        if (periodic > 0 && periodic < dim)
            mixes.push_back(bc);
    }
    return mixes;
}

#endif
