/**
 * @file TestCudaDCTDST.cu
 * @brief Test the CUDA DCT and DST of types 2 and 3 against FFTW.
 *
 * The CUDA cosine and sine transforms of the wall boxes and the Pmmm CrysFFT run
 * through CudaMakhoulTransform: with scale 1 its forward transform is the type 2
 * (FFTW's REDFT10 / RODFT10) and its backward transform the type 3 (REDFT01 /
 * RODFT01) along every axis, the cosine or the sine transform per axis. Both
 * families run the same checks on their own inputs and grids: the 1D, 2D and 3D
 * transforms of one family and of mixed families per axis against FFTW, the
 * type-2/type-3 round trip, and lengths of every parity on every axis, also the
 * 1D lengths beyond the 6144 points the earlier transforms took.
 */

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fftw3.h>

#include "CudaCommon.h"
#include "CudaMakhoulTransform.h"
#include "TestNanSafe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// A family of transforms: the cosine (DCT) or the sine (DST) transforms of types 2 and 3
struct Family
{
    const char* name;
    bool cosine;
    fftw_r2r_kind kind2, kind3;

    fftw_r2r_kind kind(int order) const { return order == 2 ? kind2 : kind3; }
    BoundaryCondition wall() const { return cosine ? BoundaryCondition::REFLECTING : BoundaryCondition::ABSORBING; }
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
        value *= sin(M_PI * x);
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
        value *= sin(M_PI * x) + 0.5 * c.family[d]->g(2 * M_PI * x);
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

//==============================================================================
// Transforms
//==============================================================================
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

/// The transform of CudaMakhoulTransform on the device: the forward transform (type 2) or the
/// backward transform (type 3), both with scale 1
static std::vector<double> cuda_transform(const Case& c, std::vector<double> data)
{
    std::vector<BoundaryCondition> bc;
    for (const Family* f : c.family)
        bc.push_back(f->wall());
    CudaMakhoulTransform<double> transform(c.n, bc);
    CudaDeviceBuffer<double> d_in(data.size()), d_out(data.size());
    CUDA_CHECK(cudaMemcpy(d_in, data.data(), sizeof(double) * data.size(), cudaMemcpyHostToDevice));
    if (c.order == 2)
        transform.forward(d_in, d_out, 1.0, 0);
    else
        transform.backward(d_in, nullptr, d_out, 1.0, 0);
    CUDA_CHECK(cudaMemcpy(data.data(), d_out, sizeof(double) * data.size(), cudaMemcpyDeviceToHost));
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

//==============================================================================
// Checks
//==============================================================================
/// The CUDA transform of a case against FFTW
static bool check_transform(const Case& c)
{
    const std::vector<double> input = input_of(c);
    const double rel_err = relative_error(cuda_transform(c, input), fftw_transform(c, input));
    const bool passed = rel_err < 1e-10;
    printf("%-40s relative error %.6e  %s\n", describe(c).c_str(), rel_err, passed ? "PASSED" : "FAILED");
    return passed;
}

/// Type 2 then type 3 recovers the input times 2N, with FFTW and with CUDA
static bool check_round_trip(const Family& f, int N)
{
    const Case forward{{N}, {&f}, 2}, backward{{N}, {&f}, 3};
    const std::vector<double> input = input_of(forward);
    bool passed = true;
    for (const bool cuda : {false, true})
    {
        std::vector<double> result = cuda ? cuda_transform(backward, cuda_transform(forward, input))
                                          : fftw_transform(backward, fftw_transform(forward, input));
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

static bool check_cases(const std::vector<Case>& cases)
{
    bool passed = true;
    for (const Case& c : cases)
        passed = check_transform(c) && passed;
    return passed;
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

int main()
{
    printf("========== CUDA DCT and DST vs FFTW ==========\n");
    bool all_passed = true;
    try
    {
        printf("--- DCT (FFTW REDFT10/01) ---\n");
        all_passed &= check_cases(join({
            // 1D: odd lengths, and beyond the 6144 points of the earlier shared-memory transforms
            of_type(DCT, 2, {{16}, {32}, {64}, {15}, {31}, {5001}, {6144}, {6145}}),
            of_type(DCT, 3, {{16}, {32}, {64}, {15}, {5001}, {6142}, {6143}}),
            of_type(DCT, 2, {{8, 8}, {16, 16}, {32, 32}, {6, 5}, {5, 6}, {7, 9}}),
            of_type(DCT, 3, {{8, 8}, {16, 16}, {32, 32}, {7, 9}, {6, 5, 7}}),
            // (6, 6, 5): odd physical z of the logical 12x12x10 CrysFFT grid
            of_type(DCT, 2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 6, 5}, {6, 5, 6}, {5, 6, 6}, {5, 7, 9}}),
            of_type(DCT, 3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {5, 7, 9}})}));
        all_passed &= check_round_trip(DCT, 32);
        all_passed &= check_round_trip(DCT, 15);

        // DST: the odd lengths of every type and dimension, which the earlier transforms refused
        printf("--- DST (FFTW RODFT10/01) ---\n");
        all_passed &= check_cases(join({
            of_type(DST, 2, {{16}, {32}, {64}, {15}}),
            of_type(DST, 3, {{16}, {32}, {64}, {15}}),
            of_type(DST, 2, {{8, 8}, {16, 16}, {32, 32}, {6, 5}}),
            of_type(DST, 3, {{8, 8}, {16, 16}, {32, 32}, {5, 6}}),
            of_type(DST, 2, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 6, 5}}),
            of_type(DST, 3, {{8, 8, 8}, {16, 16, 16}, {32, 32, 32}, {6, 5, 6}})}));
        all_passed &= check_round_trip(DST, 32);
        all_passed &= check_round_trip(DST, 15);

        printf("--- DCT and DST per axis ---\n");
        all_passed &= check_cases(mixed_cases());
    }
    catch (const std::exception& e)
    {
        printf("Unexpected exception: %s\n", e.what());
        all_passed = false;
    }

    printf("=== %s ===\n", all_passed ? "All tests PASSED!" : "Some tests FAILED!");
    return all_passed ? 0 : 1;
}
