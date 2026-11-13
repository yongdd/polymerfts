/** Failed CUDA rs operator uploads must remain unusable until a complete refresh. */
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "AbstractFactory.h"
#include "CudaCommon.h"
#include "PlatformSelector.h"

namespace
{
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Call>
void rejects(Call call, const std::string& expected)
{
    try
    {
        call();
    }
    catch (const std::exception& error)
    {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "Unexpected refusal: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("A computation used an incomplete rs operator update.");
}

struct AllocationFailure
{
    explicit AllocationFailure(long long index) { cuda_buffer_fail_countdown().store(index); }
    ~AllocationFailure() { cuda_buffer_fail_countdown().store(-1); }
};

template <typename T>
struct Result
{
    T partition;
    std::vector<T> propagator, concentration, stress;
    std::vector<double> errors;
};

template <typename T>
bool same_bits(const std::vector<T>& left, const std::vector<T>& right)
{
    return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0;
}

template <typename T>
void require_same(const Result<T>& actual, const Result<T>& expected)
{
    require(std::memcmp(&actual.partition, &expected.partition, sizeof(T)) == 0,
            "Recovered partition differs from a fresh solver.");
    require(same_bits(actual.propagator, expected.propagator), "Recovered propagator differs from a fresh solver.");
    require(same_bits(actual.concentration, expected.concentration), "Recovered concentration differs from a fresh solver.");
    require(same_bits(actual.stress, expected.stress), "Recovered stress differs from a fresh solver.");
    require(same_bits(actual.errors, expected.errors), "Recovered partition errors differ from a fresh solver.");
}

template <typename T>
struct System
{
    std::unique_ptr<AbstractFactory<T>> factory;
    std::unique_ptr<Molecules> molecules;
    std::unique_ptr<PropagatorComputationOptimizer> optimizer;
    std::unique_ptr<ComputationBox<T>> box;
    std::unique_ptr<PropagatorComputation<T>> solver;
    std::vector<T> field;
    int segments;

    System(bool reduced, double length, int n_segments = 10) : segments(n_segments)
    {
        if constexpr (std::is_same_v<T, double>)
            factory = PlatformSelector::create_factory_real("cuda", reduced);
        else
            factory = PlatformSelector::create_factory_complex("cuda", reduced);
        molecules = factory->create_molecules_information("discrete", 0.1, {{"A", 1.0}});
        molecules->add_polymer(1.0, {{"A", 0.1 * segments, 0, 1}});
        optimizer = factory->create_propagator_computation_optimizer(molecules.get(), false);
        box = factory->create_computation_box({64}, {length}, {}, nullptr);
        solver = factory->create_propagator_computation(box.get(), molecules.get(), optimizer.get(), "rs");
        field.resize(64);
        for (int i = 0; i < 64; ++i)
        {
            const double x = 2.0 * std::numbers::pi * (i + 0.5) / 64;
            if constexpr (std::is_same_v<T, double>)
                field[i] = std::sqrt(2.0) * 5.0 * std::cos(x);
            else
                field[i] = {std::sqrt(2.0) * 5.0 * std::cos(x), 0.2 * std::sin(2.0 * x)};
        }
    }

    T partition()
    {
        solver->compute_propagators({{"A", field.data()}});
        return solver->get_total_partition(0);
    }

    Result<T> compute()
    {
        Result<T> result;
        result.partition = partition();
        result.propagator.resize(64);
        solver->get_chain_propagator(result.propagator.data(), 0, 0, 1, segments);
        solver->compute_concentrations();
        result.concentration.resize(64);
        solver->get_total_concentration("A", result.concentration.data());
        require(std::abs(box->mean(result.concentration.data()) - T(1.0)) < 2e-12,
                "Material is not conserved after refreshing the operators.");
        require(solver->check_total_partition(), "Partitions are inconsistent after refreshing the operators.");
        result.errors = solver->compute_partition_errors();
        solver->compute_stress();
        result.stress = solver->get_stress();
        return result;
    }

    void require_unusable()
    {
        rejects([&] { solver->get_total_partition(0); }, "Propagators not computed");
        rejects([&] { solver->compute_concentrations(); }, "Propagators not computed");
        rejects([&] { solver->compute_stress(); }, "Propagators not computed");
        rejects([&] { solver->get_stress(); }, "Propagators not computed");
        rejects([&] { solver->check_total_partition(); }, "Propagators not computed");
        // A repeated compute attempt must not clear the operator refusal.
        for (int attempt = 0; attempt < 2; ++attempt)
            rejects([&] { partition(); }, "rs operator update has not completed");
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

template <typename T>
Result<T> check_recovery(bool reduced)
{
    constexpr double old_length = 2.0, new_length = 4.0;
    System<T> fresh(reduced, new_length);
    const Result<T> expected = fresh.compute();

    // Sweep every allocation until the first update for which the hook does
    // not fire. This includes both kernel uploads and the last ghost buffers.
    int failures = 0;
    bool completed = false;
    for (int failure_index = 0; failure_index < 64; ++failure_index)
    {
        System<T> changed(reduced, old_length);
        (void)changed.compute();
        changed.box->set_lx({new_length});
        bool failed = false, hook_fired = false;
        {
            AllocationFailure injection(failure_index);
            try
            {
                changed.solver->update_laplacian_operator();
            }
            catch (const std::exception& error)
            {
                require(std::string(error.what()).find("injected failure (test hook)") != std::string::npos,
                        "Operator update failed for an unexpected reason: " + std::string(error.what()));
                failed = true;
            }
            hook_fired = cuda_buffer_fail_countdown().load() < 0;
        }
        require(failed == hook_fired, "The update did not report the injected allocation failure.");
        if (failed)
        {
            ++failures;
            changed.require_unusable();
            changed.solver->update_laplacian_operator();
        }
        require_same(changed.compute(), expected);
        if (!failed)
        {
            completed = true;
            break;
        }
    }
    require(completed && failures >= 3, "The update allocation sweep did not cover early, middle and late failures.");

    // A one-bead chain does not call convolve(); field preparation must also
    // enforce the refusal instead of accepting this special case.
    System<T> bead(reduced, old_length, 1), fresh_bead(reduced, new_length, 1);
    (void)bead.compute();
    bead.box->set_lx({new_length});
    {
        AllocationFailure injection(0);
        rejects([&] { bead.solver->update_laplacian_operator(); }, "injected failure (test hook)");
    }
    bead.require_unusable();
    bead.solver->update_laplacian_operator();
    require_same(bead.compute(), fresh_bead.compute());

    // Independent thermodynamic control: stress is -d(log Q)/dL for this chain.
    constexpr double h = 1e-5;
    System<T> plus(reduced, new_length + h), minus(reduced, new_length - h);
    const T finite_difference = -(std::log(plus.partition()) - std::log(minus.partition())) / (2.0 * h);
    require(std::abs(finite_difference - expected.stress[0]) < 2e-9,
            "Fresh rs stress disagrees with the partition finite difference.");
    std::cout << "CUDA rs recovery: " << (std::is_same_v<T, double> ? "real" : "complex")
              << ", reduced=" << reduced << ", " << failures << " allocation failures passed\n";
    return expected;
}
}

int main()
{
    try
    {
        const auto real_full = check_recovery<double>(false);
        const auto real_reduced = check_recovery<double>(true);
        require_same(real_reduced, real_full);
        const auto complex_full = check_recovery<std::complex<double>>(false);
        const auto complex_reduced = check_recovery<std::complex<double>>(true);
        require_same(complex_reduced, complex_full);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return -1;
    }
}
