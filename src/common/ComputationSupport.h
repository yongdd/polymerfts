/**
 * @file ComputationSupport.h
 * @brief Two small helpers of the computations: exceptions out of OpenMP
 *        loops (ParallelErrors) and the owner of backend memory (BackendMemory).
 *
 * @see PropagatorComputation, which uses both
 */

#ifndef COMPUTATION_SUPPORT_H_
#define COMPUTATION_SUPPORT_H_

#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include "ComputationBackend.h"

/**
 * @brief Carries an exception out of an OpenMP parallel loop.
 *
 * An exception cannot leave an OpenMP parallel region; the program would be
 * terminated. run() catches it inside the loop body and rethrow() rethrows
 * the first one after the loop, so that, for example, running out of GPU
 * memory during a computation reaches Python as an exception.
 */
class ParallelErrors
{
private:
    std::exception_ptr first;
    std::mutex mutex;

public:
    template <typename Body>
    void run(Body&& body)
    {
        try
        {
            body();
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!first)
                first = std::current_exception();
        }
    }

    void rethrow()
    {
        if (first)
            std::rethrow_exception(first);
    }
};

/**
 * @brief The owner of the fields and checkpoints of one computation in backend memory.
 *
 * Every array it allocates is freed when it is destroyed. A computation declares it
 * after its backend, so the arrays go before the backend (and its CUDA streams),
 * also when a constructor throws after allocating some of them.
 */
template <typename T>
class BackendMemory
{
private:
    const std::unique_ptr<ComputationBackend<T>>& backend;
    std::vector<BackendArray<T>*> arrays;
    std::vector<CheckpointArray<T>*> checkpoints;

public:
    explicit BackendMemory(const std::unique_ptr<ComputationBackend<T>>& backend) : backend(backend) {}
    BackendMemory(const BackendMemory&) = delete;
    BackendMemory& operator=(const BackendMemory&) = delete;
    ~BackendMemory()
    {
        for (BackendArray<T>* a : arrays)
            backend->deallocate(a);
        for (CheckpointArray<T>* a : checkpoints)
            backend->deallocate_checkpoint(a);
    }

    /// A field of get_n_basis() elements, set to zero
    BackendArray<T>* allocate()
    {
        arrays.push_back(nullptr);  // room first, so that no allocation is lost
        try
        {
            arrays.back() = backend->allocate();
        }
        catch (...)
        {
            arrays.pop_back();
            throw;
        }
        return arrays.back();
    }

    /// A checkpoint of get_n_basis() elements (contents undefined)
    CheckpointArray<T>* allocate_checkpoint()
    {
        checkpoints.push_back(nullptr);
        try
        {
            checkpoints.back() = backend->allocate_checkpoint();
        }
        catch (...)
        {
            checkpoints.pop_back();
            throw;
        }
        return checkpoints.back();
    }
};

#endif
