#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaCommon.h"

namespace
{
    struct DivisionCase
    {
        const char* name;
        cuDoubleComplex numerator;
        cuDoubleComplex divisor;
        cuDoubleComplex expected;
        double factor = 1.0;
        bool singular = false;
    };

    bool close_component(double actual, double expected)
    {
        if (!std::isfinite(actual))
            return false;
        if (expected == 0.0)
            return actual == 0.0;
        // A relative comparison also checks tiny/subnormal components: an
        // absolute floor would incorrectly accept zero in place of them.
        return actual != 0.0 && std::abs(actual / expected - 1.0) < 2e-14;
    }

    void check_case(const DivisionCase& test)
    {
        CudaDeviceBuffer<cuDoubleComplex> numerator(1), divisor(1), output(1);
        CudaDeviceBuffer<double> mask(1);
        constexpr int threads = 128;
        for (int masking = 0; masking < 3; ++masking)
        {
            const double weight = masking == 2 ? 0.0 : 1.0;
            CUDA_CHECK(cudaMemcpy(mask.get(), &weight, sizeof(weight), cudaMemcpyHostToDevice));
            // Exercise distinct output and aliasing with either complete input.
            for (int alias = 0; alias < 3; ++alias)
            {
                CUDA_CHECK(cudaMemcpy(numerator.get(), &test.numerator, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(divisor.get(), &test.divisor, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
                cuDoubleComplex* destination = alias == 1 ? numerator.get() : alias == 2 ? divisor.get() : output.get();
                if (masking == 0)
                    ker_divide<<<grid_blocks(1, threads), threads>>>(destination, numerator.get(), divisor.get(), test.factor, 1);
                else
                    ker_divide_masked<<<grid_blocks(1, threads), threads>>>(destination, numerator.get(), divisor.get(), test.factor, mask.get(), 1);
                CUDA_CHECK_LAUNCH();
                cuDoubleComplex actual;
                CUDA_CHECK(cudaMemcpy(&actual, destination, sizeof(actual), cudaMemcpyDeviceToHost));
                const bool correct = masking == 2 ? actual.x == 0.0 && actual.y == 0.0
                    : test.singular ? !std::isfinite(actual.x) || !std::isfinite(actual.y)
                    : close_component(actual.x, test.expected.x) && close_component(actual.y, test.expected.y);
                if (!correct)
                {
                    std::cerr << test.name << ": mask=" << masking << ", alias=" << alias
                              << ", actual=(" << actual.x << ", " << actual.y << ")"
                              << ", expected=(" << test.expected.x << ", " << test.expected.y << ")\n";
                    throw std::runtime_error("Complex division range/alias check failed");
                }
            }
        }
    }
}

int main()
{
    try
    {
        const double tiny = std::numeric_limits<double>::denorm_min();
        std::vector<DivisionCase> cases = {
            {"finite quotient despite numerator/scale overflow", {1e308, 0.0}, {0.5, 0.5}, {1e308, -1e308}},
            {"finite quotient despite numerator sum overflow", {1e308, 1e308}, {1.0, 1.0}, {1e308, 0.0}},
            {"large divisor components", {1e308, 1e308}, {1e308, 1e308}, {1.0, 0.0}},
            {"subnormal real divisor", {4.0 * tiny, 2.0 * tiny}, {2.0 * tiny, 0.0}, {2.0, 1.0}},
            {"subnormal complex divisor", {2.0 * tiny, 0.0}, {tiny, tiny}, {1.0, -1.0}},
            {"subnormal quotient", {tiny, tiny}, {1.0, 0.0}, {tiny, tiny}},
            {"different numerator component scales", {1e150, 1e-200}, {1e100, 0.0}, {1e50, 1e-300}},
            {"different divisor component scales", {1e150, 0.0}, {1e100, 1e-220}, {1e50, -1e-270}},
            {"ordinary complex quotient", {3.0, 4.0}, {1.0, -2.0}, {-1.0, 2.0}},
            {"negative factor", {3.0, 4.0}, {1.0, -2.0}, {0.5, -1.0}, -0.5},
            {"factor rescues overflow", {1e308, 0.0}, {1e-100, 0.0}, {1e308, 0.0}, 1e-100},
            {"factor rescues underflow", {1e-308, 0.0}, {1e100, 0.0}, {1e-308, 0.0}, 1e100},
            {"zero numerator", {0.0, 0.0}, {0.5, -0.7}, {0.0, 0.0}},
            {"zero factor", {1e308, 1e308}, {tiny, tiny}, {0.0, 0.0}, 0.0},
            {"accessible zero divisor", {1.0, 2.0}, {0.0, 0.0}, {0.0, 0.0}, 1.0, true},
            {"accessible zero over zero", {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, 1.0, true},
            {"zero factor does not regularize zero divisor", {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, 0.0, true},
        };
        for (double shift : {-250.0, 250.0})
        {
            const double q = std::exp(-shift);
            cases.push_back({"discrete real bead weight", {q * q, 0.0}, {q, 0.0}, {q, 0.0}});
            cases.push_back({"discrete complex bead weight", {0.75 * q * q, q * q}, {q, 0.5 * q}, {q, 0.5 * q}});
        }
        for (const DivisionCase& test : cases)
            check_case(test);
        std::cout << "CUDA complex division: finite range, factors, masks and aliases passed\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return -1;
    }
}
