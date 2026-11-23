/**
 * @file ComputationReduceMemoryDiscrete.h
 * @brief Propagator computation for discrete chains, reduced memory.
 *
 * Same physics and arithmetic as ComputationDiscrete, but q(r, n) is kept
 * only at checkpoints (see ComputationReduceMemoryBase) and recomputed from
 * the last checkpoint before it when needed; the sums over segment pairs run
 * in the same order as in full storage. compute_concentrations_and_stress()
 * walks the segments once for both.
 *
 * The half-bond states q(r, n+1/2) that junctions need are always kept, as
 * half_checkpoint(key, n): the junction value a junction propagator starts
 * from (n = 0) and the states half a bond past the segments in junction_ends.
 * A junction reads only these, so the segments its branches end at need no
 * checkpoint of their own.
 *
 * @see ComputationDiscrete for the full-storage computation
 */

#ifndef COMPUTATION_REDUCE_MEMORY_DISCRETE_H_
#define COMPUTATION_REDUCE_MEMORY_DISCRETE_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationReduceMemoryBase.h"

template <typename T>
class ComputationReduceMemoryDiscrete : public ComputationReduceMemoryBase<T>
{
private:
    /// q(r, n+1/2) of propagator key: n = 0 of a junction propagator and the junction ends
    /// (in the backend memory of PropagatorComputation)
    typename ComputationReduceMemoryBase<T>::CheckpointMap half_checkpoint;

    /// A propagator with what it reads, resolved once: the checkpoints are fixed after
    /// construction, and keys grow with the size of a polymer
    struct Job
    {
        const ComputationEdge* edge;  ///< Its start, monomer type, dependencies and max_n_segment
        /// (q_i, n_i) of an aggregation (segments) or of a junction (half-bond states)
        std::vector<std::pair<CheckpointArray<T>*, double>> sources;
        std::vector<CheckpointArray<T>*> checkpoints;       ///< Checkpoint of q(n), or nullptr
        std::vector<CheckpointArray<T>*> half_checkpoints;  ///< Half-bond state q(n+1/2), or nullptr
    };
    /// The propagators one at a time, each after the ones it depends on (the optimizer's order)
    std::vector<Job> jobs;

    /// One segment per polymer for its partition function: (polymer, key, n, key†, monomer type, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, std::string, double>> single_partition_segment;

protected:
    void compute_blocks(bool with_concentrations, bool with_stress) override;

public:
    ComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules* molecules,
                                    PropagatorComputationOptimizer* propagator_computation_optimizer,
                                    const NumericalMethodInfo& method,
                                    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
