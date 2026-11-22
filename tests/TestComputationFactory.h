/**
 * @file TestComputationFactory.h
 * @brief The propagator computations of a test on every available platform.
 *
 * The C++ tests build their molecules directly and get the box and the propagator
 * computation from the platform factory, the construction path of `PropagatorSolver`.
 *
 * PlatformSelector::avail_platforms() lists cpu-fftw, cpu-mkl and cuda in this order,
 * cuda only when a device is visible. A test that runs the computations of
 * make_platform_runs() therefore ends with `return cuda_part_missing() ? 1 : 0;`
 * (TestCudaPart.h): a build with the CUDA backend tested without a GPU runs its CPU
 * parts and fails.
 */

#ifndef TEST_COMPUTATION_FACTORY_H_
#define TEST_COMPUTATION_FACTORY_H_

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "ComputationBox.h"
#include "Molecules.h"
#include "PlatformSelector.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "TestCudaPart.h"

/// A propagator computation of a test and the box it uses, on one platform
template <typename T>
struct PlatformRun
{
    std::string name;       ///< "cpu-mkl", "cpu-mkl, reduce_memory, aggregated", ...
    std::string platform;   ///< "cpu-fftw", "cpu-mkl" or "cuda"
    bool reduce_memory = false;
    std::unique_ptr<ComputationBox<T>> cb;
    /// Declared after its box, so destroyed before it: a computation uses its box until then
    std::unique_ptr<PropagatorComputation<T>> solver;

    PlatformRun() = default;
    PlatformRun(PlatformRun&&) = default;
    /// No assignment: it would replace the box while the old computation still uses it
    PlatformRun& operator=(PlatformRun&&) = delete;
};

/**
 * @brief The propagator computation of one platform and memory mode.
 *
 * The box and the molecules, and the optimizer, outlive the computation.
 *
 * @param platform         "cpu-fftw", "cpu-mkl" or "cuda"
 * @param reduce_memory    Store only checkpoints (reduce-memory mode)
 * @param bc               Boundary conditions, two per axis, or {} for a periodic box
 * @param numerical_method A method of the chain model ("rqm4", "rk2" or "cn-adi2"; "ps" or
 *                         "rs"), or "" for its default
 * @param mask             Accessible fraction of each grid point, or nullptr
 */
template <typename T>
PlatformRun<T> make_platform_run(
    const std::string& platform, bool reduce_memory,
    const std::vector<int>& nx, const std::vector<double>& lx, const std::vector<std::string>& bc,
    Molecules* molecules, PropagatorComputationOptimizer* optimizer,
    const std::string& numerical_method = "", const double* mask = nullptr)
{
    std::unique_ptr<AbstractFactory<T>> factory = PlatformSelector::create_factory<T>(platform, reduce_memory);
    PlatformRun<T> run;
    run.name = platform + (reduce_memory ? ", reduce_memory" : "") + (optimizer->use_aggregation() ? ", aggregated" : "");
    run.platform = platform;
    run.reduce_memory = reduce_memory;
    run.cb = factory->create_computation_box(nx, lx, bc, mask);
    run.solver = factory->create_propagator_computation(run.cb.get(), molecules, optimizer, numerical_method);
    return run;
}

/**
 * @brief The propagator computations of a test on every available platform.
 *
 * In the order of avail_platforms(), each platform in the memory modes given (full
 * memory, then reduced memory), and each mode with every optimizer given (without,
 * then with aggregation, say).
 */
template <typename T>
std::vector<PlatformRun<T>> make_platform_runs(
    const std::vector<int>& nx, const std::vector<double>& lx, const std::vector<std::string>& bc,
    Molecules* molecules, const std::vector<PropagatorComputationOptimizer*>& optimizers,
    const std::string& numerical_method = "", const std::vector<bool>& memory_modes = {false, true})
{
    std::vector<PlatformRun<T>> runs;
    for (const std::string& platform : PlatformSelector::avail_platforms())
        for (bool reduce_memory : memory_modes)
            for (PropagatorComputationOptimizer* optimizer : optimizers)
                runs.push_back(make_platform_run<T>(platform, reduce_memory, nx, lx, bc,
                                                    molecules, optimizer, numerical_method));
    return runs;
}

#endif
