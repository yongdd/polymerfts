/**
 * @file ComputationContinuous.h
 * @brief Propagator computation for continuous (Gaussian) chains, full storage.
 *
 * Solves the modified diffusion equation
 *
 *     ∂q/∂s = (b²/6)∇²q - w(r)q
 *
 * for every propagator the optimizer selected, keeping q(r, s) at every
 * contour step, and integrates the concentrations and stress along the
 * contour with Simpson's rule:
 *
 *     φ(r) = (1/Q) ∫ q(r,s) q†(r,N-s) ds
 *
 * The class is platform neutral: the grid work goes through a
 * ComputationBackend, which the platform factories (CpuFactory, CudaFactory)
 * pass in.
 *
 * **Propagator storage:** propagator[key][n] for n = 0..max_n_segment, where
 * n = 0 is the initial condition.
 *
 * @see PropagatorComputation for the shared concentration accessors
 * @see ComputationBackend for the platform operations
 */

#ifndef COMPUTATION_CONTINUOUS_H_
#define COMPUTATION_CONTINUOUS_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "PropagatorComputation.h"

template <typename T>
class ComputationContinuous : public PropagatorComputation<T>
{
private:
    /// propagator[key][n], n = 0..max_n_segment (in the backend memory of PropagatorComputation)
    std::map<std::string, std::vector<BackendArray<T>*>> propagator;

    /// A job of the scheduler with what it reads, resolved once: the schedule and the
    /// storage are fixed after construction, and keys grow with the size of a polymer
    struct Job
    {
        const ComputationEdge* edge;  ///< Its start, monomer type, contour step and dependencies
        BackendArray<T>** q;          ///< propagator[key]
        int n_segment_from;           ///< The job computes q(n_segment_from+1) .. q(n_segment_to)
        int n_segment_to;
        std::vector<std::pair<BackendArray<T>*, double>> sources;  ///< (q_i, n_i) of a junction or aggregation
    };
    std::vector<std::vector<Job>> schedule;  ///< The scheduler's groups of concurrent jobs

    /// One segment per polymer for its partition function: (polymer, q, q†, n_aggregated)
    std::vector<std::tuple<int, BackendArray<T>*, BackendArray<T>*, double>> single_partition_segment;

public:
    /**
     * @param method         The numerical method (see PropagatorComputation)
     * @param create_backend Creates the platform backend (see PropagatorComputation)
     */
    ComputationContinuous(ComputationBox<T>* cb, Molecules* molecules,
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
