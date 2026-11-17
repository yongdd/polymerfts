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
 * @file TestCrysFFTSharedTables.cu
 * @brief CUDA CrysFFT objects of different streams share one table per key.
 *
 * A solver keeps one CrysFFT object per stream and owns the k-space tables,
 * one per coefficient for all streams, which it rebuilds in place when the cell
 * changes. For the Pmmm DCT, recursive 3m and ObliqueZ engines, with `a` and its
 * clone `b` on two streams:
 *
 * 1. b diffuses with the table a built.
 * 2. b rebuilds that table in place for a new cell (same buffer, no
 *    allocation); a and b then give the diffusion of the new cell.
 * 3. b uses a table a built for a new coefficient of the new cell, and a one b
 *    built.
 * 4. A stress multiplier rebuilt in place by b is that of the new cell
 *    (ObliqueZ has none and refuses to build one).
 *
 * "Equal" means a relative difference of at most 1e-14 against a fresh object
 * with fresh tables.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"

namespace {

constexpr double TOLERANCE = 1e-14;
constexpr double PI = 3.14159265358979323846;

using Make = std::function<std::unique_ptr<CudaCrysFFTBase>()>;

std::vector<double> to_host(const double* d, int n)
{
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(double) * n, cudaMemcpyDeviceToHost));
    return h;
}

double relative_difference(const std::vector<double>& a, const std::vector<double>& b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        num = std::max(num, std::abs(a[i] - b[i]));
        den = std::max(den, std::abs(b[i]));
    }
    return num / den;
}

bool check(const char* engine, const char* what, double err)
{
    const bool ok = std::isfinite(err) && err <= TOLERANCE;
    std::printf("  %-12s %-52s %.1e  %s\n", engine, what, err, ok ? "ok" : "FAIL");
    return ok;
}

bool check_flag(const char* engine, const char* what, bool ok)
{
    std::printf("  %-12s %-52s %s\n", engine, what, ok ? "ok" : "FAIL");
    return ok;
}

/// diffusion() of `crysfft` with `table` on d_in, on the host
std::vector<double> diffuse(CudaCrysFFTBase& crysfft, const CudaDeviceBuffer<double>& table,
                            double* d_in, double* d_out, int M_phys, cudaStream_t stream)
{
    crysfft.diffusion(d_in, d_out, table, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    return to_host(d_out, M_phys);
}

/// apply_multiplier() of `crysfft` with `table` on d_in, on the host
std::vector<double> multiply(CudaCrysFFTBase& crysfft, const CudaDeviceBuffer<double>& table,
                             double* d_in, double* d_out, int M_phys, cudaStream_t stream)
{
    crysfft.apply_multiplier(d_in, d_out, table, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    return to_host(d_out, M_phys);
}

/// mult_axes: axes of the stress multiplier checked (0: the engine has none)
bool run(const char* engine, const Make& make, int mult_axes, int M_phys,
         const std::array<double, 6>& cell_a, const std::array<double, 6>& cell_b)
{
    const double ds1 = 0.01, ds2 = 0.015, ds3 = 0.0075, mult_coeff = 0.02;

    std::mt19937 rng(17);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> q(M_phys);
    for (double& v : q)
        v = std::exp(0.2 * normal(rng));
    CudaDeviceBuffer<double> d_q(M_phys), d_out(M_phys);
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    // References: a fresh object with fresh tables for each cell
    auto reference = [&](const std::array<double, 6>& cell, double ds) {
        std::unique_ptr<CudaCrysFFTBase> fresh = make();
        CudaDeviceBuffer<double> table;
        fresh->build_boltzmann(table, cell, ds, 0);
        return diffuse(*fresh, table, d_q, d_out, M_phys, 0);
    };
    const std::vector<double> ref_a1 = reference(cell_a, ds1);
    const std::vector<double> ref_b1 = reference(cell_b, ds1);
    const std::vector<double> ref_b2 = reference(cell_b, ds2);
    const std::vector<double> ref_b3 = reference(cell_b, ds3);
    std::vector<double> ref_mult_b;
    if (mult_axes != 0)
    {
        std::unique_ptr<CudaCrysFFTBase> fresh = make();
        CudaDeviceBuffer<double> table;
        fresh->build_multiplier(table, cell_b, mult_axes, mult_coeff, 0);
        ref_mult_b = multiply(*fresh, table, d_q, d_out, M_phys, 0);
    }

    cudaStream_t stream_a{}, stream_b{};
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));
    std::unique_ptr<CudaCrysFFTBase> a = make();
    std::unique_ptr<CudaCrysFFTBase> b = a->clone();
    bool ok = true;

    // 1. b uses the table a built
    CudaDeviceBuffer<double> table_1, multiplier;
    a->build_boltzmann(table_1, cell_a, ds1, stream_a);
    if (mult_axes != 0)
        a->build_multiplier(multiplier, cell_a, mult_axes, mult_coeff, stream_a);
    ok = check(engine, "a, old cell", relative_difference(diffuse(*a, table_1, d_q, d_out, M_phys, stream_a), ref_a1)) && ok;
    ok = check(engine, "b with the table a built", relative_difference(diffuse(*b, table_1, d_q, d_out, M_phys, stream_b), ref_a1)) && ok;

    // 2. b rebuilds the tables in place for the new cell
    const double* table_1_buffer = table_1.get();
    const double* multiplier_buffer = multiplier.get();
    const long long live = CudaDeviceMemory::counters().live_count.load();
    b->build_boltzmann(table_1, cell_b, ds1, stream_b);
    if (mult_axes != 0)
        b->build_multiplier(multiplier, cell_b, mult_axes, mult_coeff, stream_b);
    const bool in_place = table_1.get() == table_1_buffer && multiplier.get() == multiplier_buffer
                       && CudaDeviceMemory::counters().live_count.load() == live;
    ok = check_flag(engine, "tables rebuilt in place, no allocation", in_place) && ok;
    ok = check(engine, "a, rebuilt table, new cell", relative_difference(diffuse(*a, table_1, d_q, d_out, M_phys, stream_a), ref_b1)) && ok;
    ok = check(engine, "b, rebuilt table, new cell", relative_difference(diffuse(*b, table_1, d_q, d_out, M_phys, stream_b), ref_b1)) && ok;

    // 3. Tables of new coefficients, built by one object and used by the other
    CudaDeviceBuffer<double> table_2, table_3;
    a->build_boltzmann(table_2, cell_b, ds2, stream_a);
    b->build_boltzmann(table_3, cell_b, ds3, stream_b);
    ok = check(engine, "b with a new table a built", relative_difference(diffuse(*b, table_2, d_q, d_out, M_phys, stream_b), ref_b2)) && ok;
    ok = check(engine, "a with a new table b built", relative_difference(diffuse(*a, table_3, d_q, d_out, M_phys, stream_a), ref_b3)) && ok;

    // 4. The stress multiplier of the new cell
    if (mult_axes != 0)
    {
        ok = check(engine, "a's stress multiplier, new cell",
                   relative_difference(multiply(*a, multiplier, d_q, d_out, M_phys, stream_a), ref_mult_b)) && ok;
    }
    else
    {
        bool threw = false;
        try
        {
            CudaDeviceBuffer<double> table;
            a->build_multiplier(table, cell_b, 7, 0.0, stream_a);
        }
        catch (std::exception&)
        {
            threw = true;
        }
        ok = check_flag(engine, "no stress multipliers", threw) && ok;
    }

    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));
    return ok;
}

}  // namespace

int main()
{
    try
    {
        bool ok = true;
        const double right = PI / 2;

        // Pmmm DCT: physical grid (Nx/2, Ny/2, Nz/2)
        {
            const std::array<int, 3> nx = {8, 6, 10};
            const Make make = [nx] { return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFT(nx)); };
            ok = run("Pmmm DCT", make, 3, 4 * 3 * 5,
                     {1.5, 1.3, 1.7, right, right, right}, {1.62, 1.25, 1.81, right, right, right}) && ok;
        }

        // Recursive 3m (no translations): physical grid (Nx/2, Ny/2, Nz/2)
        {
            const std::array<int, 3> nx = {8, 8, 8};
            const std::array<double, 9> translations = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            const Make make = [nx, translations] {
                return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTRecursive3m(nx, translations));
            };
            ok = run("Recursive 3m", make, 1, 4 * 4 * 4,
                     {1.6, 1.6, 1.6, right, right, right}, {1.75, 1.75, 1.75, right, right, right}) && ok;
        }

        // ObliqueZ: physical grid (Nx, Ny, Nz/2), gamma changes with the cell
        {
            const std::array<int, 3> nx = {8, 6, 8};
            const Make make = [nx] { return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTObliqueZ(nx)); };
            ok = run("ObliqueZ", make, 0, 8 * 6 * 4,
                     {1.5, 1.3, 1.2, right, right, 100.0 * PI / 180.0},
                     {1.55, 1.28, 1.22, right, right, 104.0 * PI / 180.0}) && ok;
        }

        if (!ok)
        {
            std::printf("TestCrysFFTSharedTables FAILED\n");
            return -1;
        }
        std::printf("TestCrysFFTSharedTables passed\n");
        return 0;
    }
    catch (std::exception& exc)
    {
        std::printf("Unexpected exception: %s\n", exc.what());
        return -1;
    }
}
