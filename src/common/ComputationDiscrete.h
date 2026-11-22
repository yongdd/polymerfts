/**
 * @file ComputationDiscrete.h
 * @brief Propagator computation for discrete (bead-spring) chains, full storage.
 *
 * Solves the Chapman-Kolmogorov recursion
 *
 *     q(r, n+1) = exp(-w(r) ds) ∫ g(r-r') q(r', n) dr'
 *
 * where g is the bond function (Park et al., J. Chem. Phys. 150, 234901
 * (2019)), keeping q(r, n) at every segment, and sums the concentrations
 * and stress over the segments:
 *
 *     φ(r) = (1/Q) Σ_n q(r,n) q†(r,N+1-n) / exp(-w(r) ds)
 *
 * The division removes the Boltzmann factor of segment n, which both
 * propagators contain.
 *
 * **Half bonds.** A junction sits between segments, half a bond away from
 * each of them. The propagators that meet there are stored half a bond past
 * their last segment, q(r, n+1/2) = g^(1/2) * q(r, n), in
 * propagator_half_steps[key][n] for every n in the optimizer's junction_ends;
 * propagator_half_steps[key][0] of a junction propagator holds the junction
 * value it starts from. An aggregated propagator starts from a sum of
 * segments (the optimizer slices to at least one segment).
 *
 * The class is platform neutral: the grid work goes through a
 * ComputationBackend, which the platform factories (CpuFactory, CudaFactory)
 * pass in.
 *
 * **Propagator storage:** propagator[key][n] for n = 1..max_n_segment
 * (segment n; index 0 is unused).
 *
 * @see ComputationCommon for the shared concentration accessors
 * @see ComputationBackend for the platform operations
 */

#ifndef COMPUTATION_DISCRETE_H_
#define COMPUTATION_DISCRETE_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationCommon.h"

template <typename T>
class ComputationDiscrete : public ComputationCommon<T>
{
private:
    // In the backend memory of ComputationCommon
    std::map<std::string, std::vector<BackendArray<T>*>> propagator;             ///< q(r, n), n = 1..max_n_segment
    std::map<std::string, std::vector<BackendArray<T>*>> propagator_half_steps;  ///< q(r, n+1/2) where needed, else nullptr

    /// A job of the scheduler with what it reads, resolved once: the schedule and the
    /// storage are fixed after construction, and keys grow with the size of a polymer
    struct Job
    {
        const ComputationEdge* edge;  ///< Its start, monomer type and dependencies
        BackendArray<T>** q;          ///< propagator[key]
        BackendArray<T>** q_half;     ///< propagator_half_steps[key]
        int n_segment_from;           ///< The job computes the segments n_segment_from+1 .. n_segment_to
        int n_segment_to;
        /// (q_i, n_i) of an aggregation (segments) or of a junction (half-bond states)
        std::vector<std::pair<BackendArray<T>*, double>> sources;
    };
    std::vector<std::vector<Job>> schedule;  ///< The scheduler's groups of concurrent jobs

    /// One segment per polymer for its partition function: (polymer, q, q†, monomer type, n_aggregated)
    std::vector<std::tuple<int, BackendArray<T>*, BackendArray<T>*, std::string, double>> single_partition_segment;

public:
    ComputationDiscrete(ComputationBox<T>* cb, Molecules* molecules,
                        PropagatorComputationOptimizer* propagator_computation_optimizer,
                        const NumericalMethodInfo& method,
                        const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
