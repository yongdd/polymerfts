/**
 * @file ComputationReduceMemoryContinuous.h
 * @brief Propagator computation for continuous chains, reduced memory.
 *
 * Same physics and arithmetic as ComputationContinuous, but q(r, s) is kept
 * only at checkpoints: s = 0, every checkpoint_interval segments, and the
 * segments other propagators start from. Every other segment is recomputed
 * from the last checkpoint before it when the concentrations, the stress,
 * or a propagator are needed; compute_concentrations_and_stress() walks the
 * segments once for both. The solvers are deterministic, so the
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
    /// A propagator with what it reads, resolved once: the checkpoints are fixed after
    /// construction, and keys grow with the size of a polymer
    struct Job
    {
        const std::string* key;       ///< Propagator key: '{' grafting point, '[' aggregated, '(' junction
        const ComputationEdge* edge;  ///< Its monomer type, dependencies and max_n_segment
        int ds_index;                 ///< Contour step of the propagator
        std::vector<std::pair<CheckpointArray<T>*, double>> sources;  ///< (q_i, n_i) of a junction or aggregation
        std::vector<CheckpointArray<T>*> checkpoints;  ///< Checkpoint of q(n), or nullptr
    };
    /// The propagators one at a time, each after the ones it depends on (the optimizer's order)
    std::vector<Job> jobs;

    /// One segment per polymer for its partition function: (polymer, key, n, key†, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, double>> single_partition_segment;

protected:
    int propagator_ds_index(const std::string& key) const override;
    void compute_blocks(bool with_concentrations, bool with_stress) override;

public:
    ComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules* molecules,
                                      PropagatorComputationOptimizer* propagator_computation_optimizer,
                                      const NumericalMethodInfo& method, SpaceGroup* space_group,
                                      const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
