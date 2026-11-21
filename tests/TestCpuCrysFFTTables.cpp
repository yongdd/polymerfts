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
 * @file TestCpuCrysFFTTables.cpp
 * @brief The CPU CrysFFT engines run on tables their caller owns.
 *
 * A solver keeps one engine for all its threads and owns the k-space tables,
 * one per coefficient, which it rebuilds in place when the cell changes. For the
 * Pmmm DCT, recursive 3m and ObliqueZ engines of each CPU library, on fields of
 * standard deviation 5:
 *
 * 1. Eight threads call one engine at once, each with one of two Boltzmann
 *    tables, twenty times: every result equals the serial one and the input is
 *    kept.
 * 2. A table rebuilt for a new cell keeps its buffer and gives the diffusion
 *    of a fresh engine with a fresh table of that cell.
 * 3. A stress multiplier rebuilt in place equals a fresh one (Pmmm and
 *    recursive 3m); ObliqueZ has none and refuses to build one. Each engine
 *    reports which (has_multipliers()).
 * 4. A table of another size is refused.
 * 5. The z work arrays of the MKL ObliqueZ engine, which every engine on a thread
 *    shares, fit each grid just transformed: 24x24x24, then 16x16x64 and
 *    32x32x16, which have the same Nx*Ny*Nz but need 4352 and 5120 spectrum
 *    values. An undersized array would only show as heap corruption, which need
 *    not crash.
 *
 * "Equal" means a relative difference of at most 1e-14 (the thread and fresh
 * engine results run the same plans on other scratch arrays).
 */

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "CpuCrysFFT.h"
#include "CrysFFTPmmm.h"
#ifdef USE_CPU_FFTW
#include "FftwCrysFFTRecursive3m.h"
#include "FftwCrysFFTObliqueZ.h"
#endif
#ifdef USE_CPU_MKL
#include "MklCrysFFTRecursive3m.h"
#include "MklCrysFFTObliqueZ.h"
#endif
#include "TestNanSafe.h"

namespace {

constexpr double TOLERANCE = 1e-14;
constexpr int THREADS = 8;
constexpr int REPEATS = 20;

using Make = std::function<std::unique_ptr<CpuCrysFFT>()>;

double relative_difference(const std::vector<double>& a, const std::vector<double>& b)
{
    double num = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        num = max_or_nan(num, std::abs(a[i] - b[i]));
    return num / max_abs_or_nan(b);
}

bool check(const std::string& engine, const std::string& what, bool ok)
{
    std::cout << "  " << engine << ": " << what << (ok ? "  ok" : "  FAIL") << std::endl;
    return ok;
}

/// A relative difference within TOLERANCE (a NaN fails)
bool check_difference(const std::string& engine, const std::string& what, double err)
{
    return check(engine, what + " " + std::to_string(err), err <= TOLERANCE);
}

std::vector<double> diffused(const CpuCrysFFT& engine, const std::vector<double>& q, const std::vector<double>& table)
{
    std::vector<double> out(q.size());
    engine.diffusion(q.data(), out.data(), table);
    return out;
}

std::vector<double> multiplied(const CpuCrysFFT& engine, const std::vector<double>& q, const std::vector<double>& table)
{
    std::vector<double> out(q.size());
    engine.apply_multiplier(q.data(), out.data(), table);
    return out;
}

bool check_engine(const std::string& name, const Make& make, std::array<double, 6> cell, bool has_multipliers,
                  std::mt19937_64& rng)
{
    bool ok = true;
    const std::unique_ptr<CpuCrysFFT> engine = make();
    const int M = engine->get_M_physical();
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> q(M);
    for (double& v : q)
        v = normal(rng);
    const std::vector<double> q_kept = q;

    // 1. One engine, two tables, eight threads at once
    const std::array<double, 2> coeffs = {0.004, 0.011};
    std::array<std::vector<double>, 2> tables;
    for (int c = 0; c < 2; ++c)
        engine->build_boltzmann(tables[c], cell, coeffs[c]);
    const std::array<std::vector<double>, 2> serial = {diffused(*engine, q, tables[0]), diffused(*engine, q, tables[1])};
    std::vector<std::vector<double>> parallel(THREADS, std::vector<double>(M));
    #pragma omp parallel for num_threads(THREADS) schedule(static, 1)
    for (int t = 0; t < THREADS; ++t)
        for (int repeat = 0; repeat < REPEATS; ++repeat)
            engine->diffusion(q.data(), parallel[t].data(), tables[(t + repeat) % 2]);
    double err = 0.0;
    for (int t = 0; t < THREADS; ++t)
        err = max_or_nan(err, relative_difference(parallel[t], serial[(t + REPEATS - 1) % 2]));
    ok &= check_difference(name, "threads against serial", err);
    ok &= check(name, "input kept", q == q_kept);

    // 2. Cell change: the table is rebuilt in place
    std::array<double, 6> new_cell = cell;
    for (int d = 0; d < 3; ++d)
        new_cell[d] *= 1.0 + 0.05 * (d + 1);
    const double* buffer = tables[0].data();
    engine->build_boltzmann(tables[0], new_cell, coeffs[0]);
    const std::unique_ptr<CpuCrysFFT> fresh = make();
    std::vector<double> fresh_table;
    fresh->build_boltzmann(fresh_table, new_cell, coeffs[0]);
    ok &= check(name, "table rebuilt in place", tables[0].data() == buffer);
    err = relative_difference(diffused(*engine, q, tables[0]), diffused(*fresh, q, fresh_table));
    ok &= check_difference(name, "diffusion of the new cell against a fresh engine", err);

    // 3. Stress multipliers, which the engine reports (the solver's stress fast path asks it)
    ok &= check(name, "has_multipliers()", engine->has_multipliers() == has_multipliers);
    if (has_multipliers)
    {
        std::vector<double> multiplier;
        engine->build_multiplier(multiplier, cell, 1, coeffs[1]);
        const double* multiplier_buffer = multiplier.data();
        engine->build_multiplier(multiplier, new_cell, 1, coeffs[1]);
        std::vector<double> fresh_multiplier;
        fresh->build_multiplier(fresh_multiplier, new_cell, 1, coeffs[1]);
        ok &= check(name, "multiplier rebuilt in place", multiplier.data() == multiplier_buffer);
        err = relative_difference(multiplied(*engine, q, multiplier), multiplied(*fresh, q, fresh_multiplier));
        ok &= check_difference(name, "multiplier of the new cell against a fresh engine", err);
    }
    else
    {
        bool refused = false;
        try
        {
            std::vector<double> multiplier;
            engine->build_multiplier(multiplier, cell, 1, 0.0);
        }
        catch (const std::exception&)
        {
            refused = true;
        }
        ok &= check(name, "no stress multipliers", refused);
    }

    // 4. A table of another size
    bool refused = false;
    try
    {
        std::vector<double> out(M);
        engine->diffusion(q.data(), out.data(), std::vector<double>(tables[0].size() + 1, 1.0));
    }
    catch (const std::exception&)
    {
        refused = true;
    }
    ok &= check(name, "table of another size refused", refused);
    return ok;
}

#ifdef USE_CPU_MKL
/// 5. The z work arrays of this thread after a diffusion of the MKL ObliqueZ engine of grid nx
bool check_mkl_z_work(const std::array<int, 3>& nx, const std::array<double, 6>& cell, std::mt19937_64& rng)
{
    const MklCrysFFTObliqueZ engine(nx);
    std::vector<double> table;
    engine.build_boltzmann(table, cell, 0.004);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> q(engine.get_M_physical());
    for (double& v : q)
        v = normal(rng);
    diffused(engine, q, table);

    const std::array<size_t, 2> sizes = MklCrysFFTObliqueZ::thread_z_work_sizes();
    const size_t need_real = static_cast<size_t>(nx[0]) * nx[1] * (nx[2] / 2);
    const size_t need_freq = static_cast<size_t>(nx[0]) * nx[1] * (nx[2] / 4 + 1);
    const std::string grid = std::to_string(nx[0]) + "x" + std::to_string(nx[1]) + "x" + std::to_string(nx[2]);
    return check("MKL ObliqueZ", "z work arrays of " + grid + ": " + std::to_string(sizes[0]) + " real, "
                 + std::to_string(sizes[1]) + " complex (need " + std::to_string(need_real) + ", "
                 + std::to_string(need_freq) + ")", sizes[0] >= need_real && sizes[1] >= need_freq);
}
#endif

}  // namespace

int main()
{
    try
    {
        std::mt19937_64 rng(5);
        const double right = std::numbers::pi / 2.0;
        const std::array<double, 6> orthorhombic = {2.0, 1.7, 1.5, right, right, right};
        const std::array<double, 6> hexagonal = {4.0, 4.0, 3.0, right, right, 2.0 * std::numbers::pi / 3.0};
        const std::array<int, 3> nx = {16, 12, 8};
        const std::array<int, 3> nx_hex = {12, 12, 8};
        bool ok = true;
#ifdef USE_CPU_FFTW
        ok &= check_engine("FFTW Pmmm", [&] { return std::make_unique<CrysFFTPmmm>(nx, FFTBackend::FFTW); },
                           orthorhombic, true, rng);
        ok &= check_engine("FFTW 3m", [&] { return std::make_unique<FftwCrysFFTRecursive3m>(nx); },
                           orthorhombic, true, rng);
        ok &= check_engine("FFTW ObliqueZ", [&] { return std::make_unique<FftwCrysFFTObliqueZ>(nx_hex); },
                           hexagonal, false, rng);
#endif
#ifdef USE_CPU_MKL
        ok &= check_engine("MKL Pmmm", [&] { return std::make_unique<CrysFFTPmmm>(nx, FFTBackend::MKL); },
                           orthorhombic, true, rng);
        ok &= check_engine("MKL 3m", [&] { return std::make_unique<MklCrysFFTRecursive3m>(nx); },
                           orthorhombic, true, rng);
        ok &= check_engine("MKL ObliqueZ", [&] { return std::make_unique<MklCrysFFTObliqueZ>(nx_hex); },
                           hexagonal, false, rng);
        for (const std::array<int, 3>& grid : {std::array<int, 3>{24, 24, 24}, std::array<int, 3>{16, 16, 64},
                                               std::array<int, 3>{32, 32, 16}})
            ok &= check_mkl_z_work(grid, hexagonal, rng);
#endif
        std::cout << (ok ? "TestCpuCrysFFTTables passed" : "TestCpuCrysFFTTables FAILED") << std::endl;
        return ok ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
