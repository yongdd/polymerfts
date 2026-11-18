/**
 * @file ComputationReduceMemoryContinuous.h
 * @brief Propagator computation for continuous chains, reduced memory.
 *
 * Same physics and arithmetic as ComputationContinuous, but q(r, s) is kept
 * only at checkpoints: s = 0, every checkpoint_interval segments, and the
 * segments other propagators start from. Every other segment is recomputed
 * from the last checkpoint before it when the concentrations, the stress,
 * or a propagator are needed. The solvers are deterministic, so the
 * recomputed segments equal the stored ones bit for bit, and the sums over
 * segment pairs run in the same order as in full storage.
 *
 * Memory: the checkpoints (backend checkpoint memory, pinned host memory on
 * CUDA) plus checkpoint_interval + 4 work fields
 * (see ComputationReduceMemoryBase).
 *
 * @see ComputationContinuous for the full-storage computation
 */

#ifndef COMPUTATION_REDUCE_MEMORY_CONTINUOUS_H_
#define COMPUTATION_REDUCE_MEMORY_CONTINUOUS_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationReduceMemoryBase.h"

template <typename T>
class ComputationReduceMemoryContinuous : public ComputationReduceMemoryBase<T>
{
private:
    /// A job of the scheduler with what it reads, resolved once: the schedule and the
    /// checkpoints are fixed after construction, and keys grow with the size of a polymer
    struct Job
    {
        const std::string* key;       ///< Propagator key: '{' grafting point, '[' aggregated, '(' junction
        const ComputationEdge* edge;  ///< Its monomer type and dependencies
        int n_segment_from;           ///< The job computes q(n_segment_from+1) .. q(n_segment_to)
        int n_segment_to;
        int ds_index;                 ///< Contour step of the propagator
        std::vector<std::pair<CheckpointArray<T>*, double>> sources;  ///< (q_i, n_i) of a junction or aggregation
        std::vector<CheckpointArray<T>*> checkpoints;  ///< Checkpoint of q(n_segment_from + i), or nullptr

        /// The checkpoint of q(n), n = n_segment_from..n_segment_to, or nullptr
        CheckpointArray<T>* checkpoint(int n) const { return checkpoints[n - n_segment_from]; }
    };
    std::vector<std::vector<Job>> schedule;  ///< The scheduler's jobs, computed one at a time

    /// One segment per polymer for its partition function: (polymer, key, n, key†, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, double>> single_partition_segment;

protected:
    int propagator_ds_index(const std::string& key) const override;

public:
    ComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules* molecules,
                                      PropagatorComputationOptimizer* propagator_computation_optimizer,
                                      const NumericalMethodInfo& method, SpaceGroup* space_group,
                                      const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
