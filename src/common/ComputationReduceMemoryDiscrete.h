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
 * half_checkpoint(key, n): the junction value a propagator starts from
 * (n = 0) and the states half a bond past the segments in junction_ends.
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
    /// q(r, n+1/2) of propagator key, for n = 0 and the junction ends
    std::map<std::tuple<std::string, int>, CheckpointArray<T>*> half_checkpoint;

    /// A job of the scheduler with what it reads, resolved once: the schedule and the
    /// checkpoints are fixed after construction, and keys grow with the size of a polymer
    struct Job
    {
        const std::string* key;       ///< Propagator key: '{' grafting point, '[' aggregated, '(' junction
        const ComputationEdge* edge;  ///< Its monomer type and dependencies
        int n_segment_from;           ///< The job computes the segments n_segment_from+1 .. n_segment_to
        int n_segment_to;
        /// (q_i, n_i) of an aggregation (half-bond states at a junction, else segments) or of a
        /// junction (half-bond states)
        std::vector<std::pair<CheckpointArray<T>*, double>> sources;
        std::vector<CheckpointArray<T>*> checkpoints;       ///< Checkpoint of q(n_segment_from + i), or nullptr
        std::vector<CheckpointArray<T>*> half_checkpoints;  ///< Half-bond state after it, or nullptr

        /// The checkpoint of q(n), n = n_segment_from..n_segment_to, or nullptr
        CheckpointArray<T>* checkpoint(int n) const { return checkpoints[n - n_segment_from]; }
        /// The half-bond state q(n+1/2), or nullptr
        CheckpointArray<T>* half_checkpoint(int n) const { return half_checkpoints[n - n_segment_from]; }
    };
    std::vector<std::vector<Job>> schedule;  ///< The scheduler's jobs, computed one at a time

    /// One segment per polymer for its partition function: (polymer, key, n, key†, monomer type, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, std::string, double>> single_partition_segment;

    void add_half_checkpoint(const std::string& key, int n);
    void free_half_checkpoints();
    CheckpointArray<T>* half_checkpoint_at(const std::string& key, int n);

    /// The concentrations, the stress or both, in one walk along each block.
    void compute_blocks(bool with_concentrations, bool with_stress);

protected:
    int propagator_ds_index(const std::string&) const override { return 0; }

public:
    ComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules* molecules,
                                    PropagatorComputationOptimizer* propagator_computation_optimizer,
                                    const NumericalMethodInfo& method, SpaceGroup* space_group,
                                    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);
    ~ComputationReduceMemoryDiscrete();

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void compute_concentrations_and_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
