/**
 * @file TestCudaDCTDST.cu
 * @brief Test the CUDA DCT and DST of types 2 and 3 against FFTW.
 *
 * Both families of CudaRealTransform run the same checks on their own inputs and
 * grids: the 1D, 2D and 3D transforms of one type and of mixed types per axis against
 * FFTW (REDFT10/01 for the DCT, RODFT10/01 for the DST), the type-2/type-3 round trip,
 * and the lengths the constructors accept and refuse. The DCT takes odd lengths on
 * every axis; the DST takes them only in the 1D DST-3.
 */

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fftw3.h>

#include "../src/platforms/cuda/CudaRealTransform.h"
#include "TestNanSafe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// A family of transforms: the cosine (DCT) or the sine (DST) transforms of types 2 and 3
struct Family
{
    const char* name;
    bool cosine;
    CudaTransformType type2, type3;
    fftw_r2r_kind kind2, kind3;

    CudaTransformType type(int order) const { return order == 2 ? type2 : type3; }
    fftw_r2r_kind kind(int order) const { return order == 2 ? kind2 : kind3; }
    /// The function of the family: cos for the DCT, sin for the DST
    double g(double x) const { return cosine ? cos(x) : sin(x); }
};

static const Family DCT{"DCT", true, CUDA_DCT_2, CUDA_DCT_3, FFTW_REDFT10, FFTW_REDFT01};
static const Family DST{"DST", false, CUDA_DST_2, CUDA_DST_3, FFTW_RODFT10, FFTW_RODFT01};

/// One check: a grid (1 to 3 axes) and the order (2 or 3) of the transform along each axis
struct Case
{
    std::vector<int> n;
    std::vector<int> order;
};

/// The cases of grids of one type
static std::vector<Case> of_type(int order, const std::vector<std::vector<int>>& grids)
{
    std::vector<Case> cases;
    for (const std::vector<int>& n : grids)
        cases.push_back({n, std::vector<int>(n.size(), order)});
    return cases;
}

static bool one_type(const std::vector<int>& order)
{
    for (int o : order)
        if (o != order[0])
            return false;
    return true;
}

//==============================================================================
// Inputs
//==============================================================================
/// 1D input: sin(pi u) + g(2 pi u)/2 at the points u of the type, (i + 1/2)/N for type 2,
/// i/N (DCT) or (i + 1)/(N + 1) (DST) for type 3
static double input_1d(const Family& f, int order, int i, int N)
{
    if (order == 2)
        return sin(M_PI * (i + 0.5) / N) + 0.5 * f.g(2 * M_PI * (i + 0.5) / N);
    if (f.cosine)
        return sin(M_PI * i / N) + 0.5 * f.g(2 * M_PI * i / N);
    return sin(M_PI * (i + 1) / (N + 1)) + 0.5 * f.g(2 * M_PI * (i + 1) / (N + 1));
}

/// 2D and 3D input of one type: a Gaussian exp(-r^2/50) for the DCT (r from the points
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
        value *= sin(M_PI * x);
    }
    return value;
}

/// Input of mixed types, with x = (i + 1/2)/N on every axis: sin(pi x) g(pi y) + g(2 pi x) sin(2 pi y)/2
/// in 2D, sin(pi x) g(pi y) sin(pi z) in 3D
static double input_mixed(const Family& f, const int* i, const std::vector<int>& n)
{
    if (n.size() == 2)
        return sin(M_PI * (i[0] + 0.5) / n[0]) * f.g(M_PI * (i[1] + 0.5) / n[1])
             + 0.5 * f.g(2 * M_PI * (i[0] + 0.5) / n[0]) * sin(2 * M_PI * (i[1] + 0.5) / n[1]);
    return sin(M_PI * (i[0] + 0.5) / n[0])
         * f.g(M_PI * (i[1] + 0.5) / n[1])
         * sin(M_PI * (i[2] + 0.5) / n[2]);
}

static std::vector<double> input_of(const Family& f, const Case& c)
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
            input[m] = input_1d(f, c.order[0], i[0], c.n[0]);
        else if (one_type(c.order))
            input[m] = input_one_type(f, c.order[0], i, c.n);
        else
            input[m] = input_mixed(f, i, c.n);
    }
    return input;
}

//==============================================================================
// Transforms
//==============================================================================
static std::vector<double> fftw_transform(const Family& f, const Case& c, std::vector<double> data)
{
    fftw_r2r_kind kinds[3];
    for (size_t d = 0; d < c.n.size(); ++d)
        kinds[d] = f.kind(c.order[d]);
    fftw_plan plan = fftw_plan_r2r(static_cast<int>(c.n.size()), c.n.data(), data.data(), data.data(),
                                   kinds, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    return data;
}

/// The transform of CudaRealTransform, in place on the device; a grid of one type takes the
/// constructor of one type
static std::vector<double> cuda_transform(const Family& f, const Case& c, std::vector<double> data)
{
    CudaDeviceBuffer<double> d_data(data.size());
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(double) * data.size(), cudaMemcpyHostToDevice));
    const std::vector<int>& n = c.n;
    const CudaTransformType t0 = f.type(c.order[0]);
    if (n.size() == 1)
    {
        CudaRealTransform1D transform(n[0], t0);
        transform.execute(d_data);
    }
    else if (n.size() == 2 && one_type(c.order))
    {
        CudaRealTransform2D transform(n[0], n[1], t0);
        transform.execute(d_data);
    }
    else if (n.size() == 2)
    {
        CudaRealTransform2D transform(n[0], n[1], t0, f.type(c.order[1]));
        transform.execute(d_data);
    }
    else if (one_type(c.order))
    {
        CudaRealTransform3D transform(n[0], n[1], n[2], t0);
        transform.execute(d_data);
    }
    else
    {
        CudaRealTransform3D transform(n[0], n[1], n[2], t0, f.type(c.order[1]), f.type(c.order[2]));
        transform.execute(d_data);
    }
    CUDA_CHECK(cudaMemcpy(data.data(), d_data, sizeof(double) * data.size(), cudaMemcpyDeviceToHost));
    return data;
}

/// max |a - b| / max |b|, NaN if a value is NaN
static double relative_error(const std::vector<double>& a, const std::vector<double>& b)
{
    double max_err = 0.0;
    for (size_t i = 0; i < b.size(); ++i)
        max_err = max_or_nan(max_err, std::abs(a[i] - b[i]));
    return max_err / max_abs_or_nan(b);
}

static std::string describe(const Family& f, const Case& c)
{
    std::string text = std::to_string(c.n.size()) + "D ";
    for (size_t d = 0; d < c.n.size(); ++d)
        text += (d > 0 ? " x " : "") + std::string(f.name) + "-" + std::to_string(c.order[d]);
    text += ":";
    for (size_t d = 0; d < c.n.size(); ++d)
        text += (d > 0 ? " x " : " ") + std::to_string(c.n[d]);
    return text;
}

//==============================================================================
// Checks
//==============================================================================
/// The CUDA transform of a case against FFTW
static bool check_transform(const Family& f, const Case& c)
{
    const std::vector<double> input = input_of(f, c);
    const double rel_err = relative_error(cuda_transform(f, c, input), fftw_transform(f, c, input));
    const bool passed = rel_err < 1e-10;
    printf("%-40s relative error %.6e  %s\n", describe(f, c).c_str(), rel_err, passed ? "PASSED" : "FAILED");
    return passed;
}

/// Type 2 then type 3 recovers the input times 2N, with FFTW and with CUDA
static bool check_round_trip(const Family& f, int N)
{
    const Case forward{{N}, {2}}, backward{{N}, {3}};
    const std::vector<double> input = input_of(f, forward);
    bool passed = true;
    for (const bool cuda : {false, true})
    {
        std::vector<double> result = cuda ? cuda_transform(f, backward, cuda_transform(f, forward, input))
                                          : fftw_transform(f, backward, fftw_transform(f, forward, input));
        const double norm = 1.0 / (2.0 * N);
        for (double& v : result)
            v *= norm;
        const double rel_err = relative_error(result, input);
        const bool ok = rel_err < 1e-12;
        printf("%s-2 + %s-3 round trip, N = %d (%s): relative error %.6e  %s\n",
               f.name, f.name, N, cuda ? "CUDA" : "FFTW", rel_err, ok ? "PASSED" : "FAILED");
        passed = passed && ok;
    }
    return passed;
}

/// A transform shape: ny = 0 for 1D, nz = 0 for 2D
struct Shape
{
    int nx, ny, nz;
    CudaTransformType type;
};

static void construct(const Shape& s)
{
    if (s.ny == 0)
    {
        CudaRealTransform1D transform(s.nx, s.type);
    }
    else if (s.nz == 0)
    {
        CudaRealTransform2D transform(s.nx, s.ny, s.type);
    }
    else
    {
        CudaRealTransform3D transform(s.nx, s.ny, s.nz, s.type);
    }
}

/// Supported lengths are accepted; unsupported ones are refused in the constructor, not
/// computed wrongly
static bool check_lengths(const Family& f, const std::vector<Shape>& accepted, const std::vector<Shape>& rejected)
{
    bool ok = true;
    for (const Shape& s : accepted)
    {
        try { construct(s); }
        catch (const std::exception& e)
        {
            printf("  %s (%d,%d,%d), a supported length, was rejected: %s\n", getTransformName(s.type), s.nx, s.ny, s.nz, e.what());
            ok = false;
        }
    }
    for (const Shape& s : rejected)
    {
        bool threw = false;
        try { construct(s); }
        catch (const std::exception&) { threw = true; }
        if (!threw)
        {
            printf("  %s (%d,%d,%d), an unsupported length, was NOT rejected\n", getTransformName(s.type), s.nx, s.ny, s.nz);
            ok = false;
        }
    }
    printf("%s length checks: %s\n", f.name, ok ? "PASSED" : "FAILED");
    return ok;
}

static bool check_family(const Family& f, const std::vector<Case>& cases)
{
    bool passed = true;
    for (const Case& c : cases)
        passed = check_transform(f, c) && passed;
    return passed;
}

/// The cases of mixed types per axis, the same for both families
static std::vector<Case> mixed_cases()
{
    return {
        {{16, 16}, {2, 3}}, {{16, 16}, {3, 2}},
        {{8, 8, 8}, {2, 3, 2}}, {{8, 8, 8}, {3, 2, 3}}, {{8, 8, 8}, {3, 3, 2}},
    };
}

static std::vector<Case> join(std::vector<std::vector<Case>> lists)
{
    std::vector<Case> all;
    for (const auto& list : lists)
        all.insert(all.end(), list.begin(), list.end());
    return all;
}

int main()
{
    printf("========== CUDA DCT and DST vs FFTW ==========\n");
    bool all_passed = true;
    try
    {
        // DCT: odd lengths on every axis
        printf("--- DCT (FFTW REDFT10/01) ---\n");
        all_passed &= check_family(DCT, join({
            // 1D: odd lengths (slot N/2 is an ordinary pair), and beyond 4096 points (the shared
            // memory holds N values, not 2^ceil(log2 N)) up to the largest that fits
            of_type(2, {{16}, {32}, {64}, {15}, {31}, {5001}, {6144}}),
            of_type(3, {{16}, {32}, {64}, {15}, {5001}, {6142}}),
            of_type(2, {{8, 8}, {16, 16}, {32, 32}, {6, 5}, {5, 6}, {7, 9}}),
            of_type(3, {{8, 8}, {16, 16}, {32, 32}, {7, 9}}),
            // (6, 6, 5): odd physical z of the logical 12x12x10 CrysFFT grid
            of_type(2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 6, 5}, {6, 5, 6}, {5, 6, 6}, {5, 7, 9}}),
            of_type(3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {5, 7, 9}}),
            mixed_cases()}));
        // The 1D DCT-2/3 stage N and N + 2 values in the 48 KB of shared memory of a block:
        // 6144 and 6142 points fit, one more does not
        all_passed &= check_lengths(DCT,
            {{15, 0, 0, CUDA_DCT_2}, {6, 5, 7, CUDA_DCT_3}, {6144, 0, 0, CUDA_DCT_2}, {6142, 0, 0, CUDA_DCT_3}},
            {{6145, 0, 0, CUDA_DCT_2}, {6143, 0, 0, CUDA_DCT_3}});
        all_passed &= check_round_trip(DCT, 32);
        all_passed &= check_round_trip(DCT, 15);

        // DST: odd lengths only in the 1D DST-3; DST-2 and the 2D/3D DST-3 kernels assume even lengths
        printf("--- DST (FFTW RODFT10/01) ---\n");
        all_passed &= check_family(DST, join({
            of_type(2, {{16}, {32}, {64}}),
            of_type(3, {{16}, {32}, {64}, {15}}),
            of_type(2, {{8, 8}, {16, 16}, {32, 32}}),
            of_type(3, {{8, 8}, {16, 16}, {32, 32}}),
            of_type(2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}}),
            of_type(3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}}),
            mixed_cases()}));
        all_passed &= check_lengths(DST,
            {{15, 0, 0, CUDA_DST_3}},
            {{15, 0, 0, CUDA_DST_2}, {6, 5, 0, CUDA_DST_2}, {5, 6, 0, CUDA_DST_3},
             {6, 6, 5, CUDA_DST_2}, {6, 5, 6, CUDA_DST_3}});
        all_passed &= check_round_trip(DST, 32);
    }
    catch (const std::exception& e)
    {
        printf("Unexpected exception: %s\n", e.what());
        all_passed = false;
    }

    printf("=== %s ===\n", all_passed ? "All tests PASSED!" : "Some tests FAILED!");
    return all_passed ? 0 : 1;
}
