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
 * @file TestCudaCrysTablesAllocation.cu
 * @brief The CUDA CrysFFT paths allocate no tables while computing propagators.
 *
 * The solver owns the CrysFFT tables, one per coefficient for all streams. It
 * builds the Boltzmann tables when the computation is created, the stress
 * multipliers in the first compute_stress() before its segment loop, and
 * rebuilds all of them in place when the cell changes (update_laplacian_operator);
 * the loops only look them up. For each CrysFFT mode (recursive 3m, Pmmm DCT,
 * ObliqueZ) and each pseudo-spectral method:
 *
 * - The first compute_propagators()/compute_concentrations() may make only the
 *   counted allocations that the same system makes with the group P1, which has
 *   no CrysFFT path; the allocation test hook fails any further one. The solver
 *   allocates its work with the computation, so the P1 call allocates only per
 *   field: the device copy of each field and, for discrete chains, its Boltzmann
 *   factor in the basis, at most two per monomer type, which is checked first so
 *   that the budget cannot grow by solver work the reference does in that call.
 * - A cell change keeps the device allocations (count and bytes), and after it
 *   these calls may allocate nothing.
 * - A second compute_stress() allocates nothing and gives the same stress.
 * - After a second cell change, which rebuilds the stress multipliers too,
 *   compute_stress() allocates nothing either.
 * - After each change, the results (Q, and the stress after the second) equal
 *   those of a computation created for the new cell.
 *
 * The tables are one per coefficient for all streams: a second bond length (B
 * of 1.3 instead of 1.0, the same polymer) adds the tables of its coefficients
 * (and, for discrete chains, stress multipliers), and these add the same device
 * bytes with four streams as with one. Tables kept per stream would add four
 * times as many.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "AbstractFactory.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "PlatformSelector.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "SpaceGroup.h"

struct Case
{
    std::string name;
    std::string chain_model;
    std::string method;
    std::vector<int> nx;
    std::vector<double> lx;
    std::vector<double> angles;   ///< empty: orthogonal
    std::string sg_symbol;
    int sg_hall;
    std::string basis;            ///< "m3", "pmmm" or "z-mirror"
};

struct Computation
{
    std::unique_ptr<AbstractFactory<double>> factory;
    std::unique_ptr<SpaceGroup> sg;
    std::unique_ptr<ComputationBox<double>> cb;
    std::unique_ptr<Molecules> molecules;
    std::unique_ptr<PropagatorComputationOptimizer> optimizer;
    std::unique_ptr<PropagatorComputation<double>> computation;
    std::vector<double> w_a, w_b;
};

/// The computation of a case, or with p1 the same system with the group P1; b_B is the bond length of B
static Computation create(const Case& c, const std::vector<double>& lx, bool p1 = false, double b_B = 1.3)
{
    Computation s;
    s.factory = PlatformSelector::create_factory_real("cuda", false);
    s.sg = p1 ? std::make_unique<SpaceGroup>(c.nx, "P1", 1) : std::make_unique<SpaceGroup>(c.nx, c.sg_symbol, c.sg_hall);
    if (!p1 && c.basis == "m3")       s.sg->enable_m3_physical_basis();
    if (!p1 && c.basis == "pmmm")     s.sg->enable_pmmm_physical_basis();
    if (!p1 && c.basis == "z-mirror") s.sg->enable_z_mirror_physical_basis();
    const std::vector<std::string> periodic;
    s.cb = c.angles.empty()
        ? s.factory->create_computation_box(c.nx, lx, periodic)
        : s.factory->create_computation_box(c.nx, lx, periodic, c.angles);
    s.molecules = s.factory->create_molecules_information(c.chain_model, 0.05, {{"A", 1.0}, {"B", b_B}});
    s.molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
    s.optimizer = s.factory->create_propagator_computation_optimizer(s.molecules.get(), true);
    s.computation = s.factory->create_propagator_computation(
        s.cb.get(), s.molecules.get(), s.optimizer.get(), c.method, s.sg.get());

    std::mt19937 rng(43);
    std::normal_distribution<double> normal(0.0, 5.0);
    s.w_a.resize(s.cb->get_n_basis());
    s.w_b.resize(s.cb->get_n_basis());
    for (size_t i = 0; i < s.w_a.size(); ++i)
    {
        s.w_a[i] = normal(rng);
        s.w_b[i] = normal(rng);
    }
    return s;
}

static void compute(Computation& s)
{
    s.computation->compute_propagators({{"A", s.w_a.data()}, {"B", s.w_b.data()}});
    s.computation->compute_concentrations();
}

/// Number of counted allocations made by compute()
static long long count_allocations(Computation& s)
{
    const long long start = 1LL << 40;
    cuda_buffer_fail_countdown().store(start);
    compute(s);
    const long long n = start - cuda_buffer_fail_countdown().load();
    cuda_buffer_fail_countdown().store(-1);
    return n;
}

/// f(); false if it makes more than `allowed` counted allocations
template <typename F>
static bool with_allocations(const char* what, long long allowed, F&& f)
{
    bool ok = true;
    cuda_buffer_fail_countdown().store(allowed);
    try
    {
        f();
    }
    catch (std::exception& exc)
    {
        printf("  %s: %s\n", what, exc.what());
        ok = false;
    }
    cuda_buffer_fail_countdown().store(-1);
    return ok;
}

/// New box lengths and update_laplacian_operator(); false if the device allocations change
static bool change_cell(Computation& s, const std::vector<double>& lx)
{
    const CudaBufferCounters& counters = CudaDeviceMemory::counters();
    const long long count = counters.live_count.load(), bytes = counters.live_bytes.load();
    s.cb->set_lx(lx);
    s.computation->update_laplacian_operator();
    const bool same = counters.live_count.load() == count && counters.live_bytes.load() == bytes;
    if (!same)
        printf("  The cell change changed the device allocations: %lld -> %lld (%lld -> %lld bytes)\n",
               count, counters.live_count.load(), bytes, counters.live_bytes.load());
    return same;
}

#ifdef _WIN32
// The CRT environment that getenv reads; an empty value removes the variable
static void set_env(const char* name, const char* value) { _putenv_s(name, value); }
static void unset_env(const char* name) { _putenv_s(name, ""); }
#else
static void set_env(const char* name, const char* value) { setenv(name, value, 1); }
static void unset_env(const char* name) { unsetenv(name); }
#endif

/// Device bytes held by a computation of the case with n_streams streams (OMP_NUM_THREADS when it is
/// created) and bond length b_B after compute() and compute_stress()
static long long device_bytes(const Case& c, int n_streams, double b_B)
{
    const char* previous = std::getenv("OMP_NUM_THREADS");
    const std::string saved = previous ? previous : "";
    set_env("OMP_NUM_THREADS", std::to_string(n_streams).c_str());
    const long long start = CudaDeviceMemory::counters().live_bytes.load();
    long long bytes = 0;
    {
        Computation s = create(c, c.lx, /*p1=*/false, b_B);
        compute(s);
        s.computation->compute_stress();
        bytes = CudaDeviceMemory::counters().live_bytes.load() - start;
    }
    if (previous)
        set_env("OMP_NUM_THREADS", saved.c_str());
    else
        unset_env("OMP_NUM_THREADS");
    return bytes;
}

/// The tables of a second bond length add the same bytes with one stream and with four
static bool tables_shared_by_streams(const Case& c)
{
    const long long added_1 = device_bytes(c, 1, 1.3) - device_bytes(c, 1, 1.0);
    const long long added_4 = device_bytes(c, 4, 1.3) - device_bytes(c, 4, 1.0);
    const bool ok = added_1 > 0 && added_4 == added_1;
    if (!ok)
        printf("  The tables of a second bond length take %lld bytes with one stream, %lld with four\n",
               added_1, added_4);
    return ok;
}

/// |a - b| <= 1e-13 |b| componentwise relative to the largest |b|
static bool same_values(const char* what, const std::vector<double>& a, const std::vector<double>& b)
{
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < b.size(); ++i)
    {
        diff = std::max(diff, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(b[i]));
    }
    const bool same = a.size() == b.size() && diff <= 1e-13 * scale;
    if (!same)
        printf("  %s: difference %.3e (scale %.3e)\n", what, diff, scale);
    return same;
}

int main()
{
    std::vector<Case> cases = {
        {"rqm4, Im-3m m3 basis (recursive 3m)", "continuous", "rqm4", {16, 16, 16}, {1.9, 1.9, 1.9}, {}, "Im-3m", 529, "m3"},
        {"rk2, Im-3m m3 basis (recursive 3m)", "continuous", "rk2", {16, 16, 16}, {1.9, 1.9, 1.9}, {}, "Im-3m", 529, "m3"},
        {"discrete, Pmmm basis (Pmmm DCT)", "discrete", "rqm4", {16, 16, 16}, {2.0, 1.8, 1.6}, {}, "Pmmm", -1, "pmmm"},
        {"rqm4, Pmmm basis (Pmmm DCT)", "continuous", "rqm4", {16, 16, 16}, {2.0, 1.8, 1.6}, {}, "Pmmm", -1, "pmmm"},
        {"rk2, P6_3/mmc z-mirror basis (ObliqueZ)", "continuous", "rk2", {12, 12, 16}, {1.8, 1.8, 2.4}, {90.0, 90.0, 120.0}, "P6_3/mmc", 488, "z-mirror"},
        {"discrete, P6_3/mmc z-mirror basis (ObliqueZ)", "discrete", "rqm4", {12, 12, 16}, {1.8, 1.8, 2.4}, {90.0, 90.0, 120.0}, "P6_3/mmc", 488, "z-mirror"},
    };

    bool all_ok = true;
    try
    {
        for (const Case& c : cases)
        {
            Computation p1 = create(c, c.lx, /*p1=*/true);
            const long long first_call = count_allocations(p1);
            const long long per_field = 2 * 2;   // two monomer types
            bool ok = first_call <= per_field;
            if (!ok)
                printf("  The first call with P1 makes %lld allocations, more than the %lld of its fields\n",
                       first_call, per_field);

            Computation s = create(c, c.lx);
            ok = with_allocations("The first call", first_call, [&] { compute(s); }) && ok;

            // A new cell: the tables are rebuilt in place by update_laplacian_operator()
            std::vector<double> new_lx = c.lx;
            for (double& l : new_lx)
                l *= 1.07;
            ok = change_cell(s, new_lx) && ok;
            ok = with_allocations("The call after the cell change", 0, [&] { compute(s); }) && ok;
            ok = s.computation->check_total_partition() && ok;

            // Stress: multipliers before the segment loop, once
            s.computation->compute_stress();
            const std::vector<double> stress = s.computation->get_stress();
            ok = with_allocations("The second compute_stress()", 0, [&] { s.computation->compute_stress(); }) && ok;
            if (s.computation->get_stress() != stress)
            {
                printf("  The second compute_stress() changed the stress\n");
                ok = false;
            }

            Computation fresh = create(c, new_lx);
            compute(fresh);
            const double q = s.computation->get_total_partition(0);
            const double q_fresh = fresh.computation->get_total_partition(0);
            const bool same = std::abs(q - q_fresh) <= 1e-13 * std::abs(q_fresh);
            if (!same)
                printf("  Q after the cell change %.15e, fresh %.15e\n", q, q_fresh);
            ok = same && ok;

            // A second cell: the stress multipliers are rebuilt in place as well
            std::vector<double> lx_2 = c.lx;
            for (double& l : lx_2)
                l *= 0.96;
            ok = change_cell(s, lx_2) && ok;
            ok = with_allocations("The calls after the second cell change", 0, [&] {
                compute(s);
                s.computation->compute_stress();
            }) && ok;
            Computation fresh_2 = create(c, lx_2);
            compute(fresh_2);
            fresh_2.computation->compute_stress();
            ok = same_values("Q after the second cell change", {s.computation->get_total_partition(0)},
                             {fresh_2.computation->get_total_partition(0)}) && ok;
            ok = same_values("Stress after the second cell change", s.computation->get_stress(),
                             fresh_2.computation->get_stress()) && ok;

            ok = tables_shared_by_streams(c) && ok;

            printf("%s (%lld allocations in the first call): %s\n", c.name.c_str(), first_call, ok ? "ok" : "FAILED");
            all_ok = all_ok && ok;
        }
    }
    catch (std::exception& exc)
    {
        printf("Unexpected exception: %s\n", exc.what());
        return -1;
    }

    if (!all_ok)
    {
        printf("TestCudaCrysTablesAllocation FAILED\n");
        return -1;
    }
    printf("TestCudaCrysTablesAllocation passed\n");
    return 0;
}
