/**
 * @file ComputationContinuous.cpp
 * @brief Propagator computation for continuous chains, full storage.
 */

#include <cmath>
#include <memory>

#include "ComputationContinuous.h"
#include "Scheduler.h"
#include "SimpsonRule.h"

template <typename T>
ComputationContinuous<T>::ComputationContinuous(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& method,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer, method, create_backend)
{
    // q(r, s) at every contour step of every propagator
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        std::vector<BackendArray<T>*>& q = propagator[key];
        for (int n = 0; n <= edge.max_n_segment; n++)
            q.push_back(this->memory.allocate());
    }

    // The jobs of the scheduler: the propagator, its storage and the sources of a junction.
    // A job points to the edge of the optimizer, which outlives the scheduler.
    const auto& propagators = this->propagator_computation_optimizer->get_computation_propagators();
    Scheduler scheduler(propagators, this->n_streams);
    for (const auto& parallel_job : scheduler.get_schedule())
    {
        std::vector<Job>& jobs = schedule.emplace_back();
        for (const auto& [job_key, n_segment_from, n_segment_to] : parallel_job)
        {
            const ComputationEdge& edge = propagators.at(job_key);
            std::vector<std::pair<BackendArray<T>*, double>> sources;
            for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                sources.push_back({propagator.at(sub_key)[sub_n_segment], sub_n_repeated});
            jobs.push_back({&edge, propagator.at(job_key).data(), n_segment_from, n_segment_to, std::move(sources)});
        }
    }

    // One segment of each polymer gives its partition function
    for (const auto& it : this->first_block_of_each_polymer())
    {
        const auto& [p, key_left, key_right] = it->first;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
        const double n_aggregated = block.n_copies/block.n_repeated;
        single_partition_segment.push_back(std::make_tuple(
            p,
            propagator.at(key_left)[block.n_segment_left],  // q
            propagator.at(key_right)[0],                    // q†
            n_aggregated));
    }
}

template <typename T>
void ComputationContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->set_fields(w_input, q_init);

    // The scheduler groups the propagators so that the propagators of one
    // group only depend on propagators of earlier groups
    for (const std::vector<Job>& parallel_jobs : schedule)
    {
        this->run_concurrently(parallel_jobs.size(), [&](const int STREAM, const size_t j)
        {
            const Job& job = parallel_jobs[j];
            const ComputationEdge& edge = *job.edge;
            BackendArray<T>** q = job.q;

            // Initial condition q(r, 0)
            if (job.n_segment_from == 0)
            {
                switch (edge.start)
                {
                case PropagatorStart::FREE_END:
                    this->backend->set_unity(STREAM, q[0]);
                    break;
                case PropagatorStart::GRAFTED:
                    this->backend->copy_from_host(STREAM, q[0], q_init.at(edge.graft_label));
                    break;
                case PropagatorStart::AGGREGATED:
                    this->backend->junction_sum(STREAM, q[0], job.sources);
                    break;
                case PropagatorStart::JUNCTION:  // branches meeting at the start
                    this->backend->junction_product(STREAM, q[0], job.sources);
                    break;
                }
                this->backend->apply_mask(STREAM, q[0]);
            }

            // q(r, n+1) from q(r, n)
            for (int n = job.n_segment_from; n < job.n_segment_to; n++)
                this->backend->advance_propagator(STREAM, q[n], q[n+1], edge.monomer_type, edge.ds_index);
        });
    }

    // Partition function of each polymer: Q = ∫ q(r,s) q†(r,N-s) dr / V
    for (const auto& [p, q_left, q_right, n_aggregated] : single_partition_segment)
        this->single_polymer_partitions[p] = this->backend->inner_product(q_left, q_right)
                                             /(n_aggregated*this->cb->get_volume());
    this->check_real_power_partitions();

    this->propagators_computed_ = true;
}

template <typename T>
void ComputationContinuous<T>::compute_concentrations()
{
    this->require_propagators();
    try
    {
        // Concentration of each block (several blocks when propagators are aggregated)
        this->run_concurrently(this->phi_block_iterators.size(), [&](const int STREAM, const size_t b)
        {
            const auto it = this->phi_block_iterators[b];
            const auto& [p, key_left, key_right] = it->first;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;

            // A block without segments has no concentration
            this->backend->set_zero(STREAM, it->second);
            if (N_RIGHT == 0)
                return;

            // φ = Σ_n c_n q(N_LEFT-n) q†(n), Simpson's rule over the segments of the block
            const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
            BackendArray<T>** q_left  = propagator.at(key_left).data();
            BackendArray<T>** q_right = propagator.at(key_right).data();
            std::vector<SegmentPair<T>> pairs;
            for (int n = 0; n <= N_RIGHT; n++)
                pairs.push_back({q_left[N_LEFT-n], q_right[n], simpson[n]});
            this->backend->add_segment_products(STREAM, it->second, pairs);

            // Getters normalize the integral for the requested ensemble.
        });

        this->compute_solvent_concentrations();
        this->concentrations_computed_ = true;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::compute_stress()
{
    this->require_propagators();
    try
    {
        this->check_stress_supported();
        this->backend->prepare_stress();

        // Stress of each block: Simpson's rule over the segment stresses
        std::vector<std::array<T, 6>> stress_of_block(this->phi_block.size());
        this->run_concurrently(this->phi_block_iterators.size(), [&](const int STREAM, const size_t b)
        {
            const auto it = this->phi_block_iterators[b];
            const auto& [p, key_left, key_right] = it->first;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;

            stress_of_block[b] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (N_RIGHT == 0)
                return;

            // Every segment contributes, the chain ends included: q(r,0) = 1 has no
            // stress only for periodic or reflecting walls, not for absorbing
            // walls, masks or grafting points.
            const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
            BackendArray<T>** q_left  = propagator.at(key_left).data();
            BackendArray<T>** q_right = propagator.at(key_right).data();
            std::vector<SegmentPair<T>> pairs;
            for (int n = 0; n <= N_RIGHT; n++)
                pairs.push_back({q_left[N_LEFT-n], q_right[n], simpson[n]*block.n_repeated});
            stress_of_block[b] = this->backend->block_stress(STREAM, pairs, block.monomer_type);
            this->scale_by_contour_step(stress_of_block[b], key_right);
        });

        // Sum the blocks of each polymer
        this->set_stress_from_blocks(stress_of_block);
        this->stress_computed_ = true;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    const std::string& key = this->chain_propagator_key(polymer, v, u, n, 0);
    this->backend->copy_to_host(q_out, propagator.at(key)[n]);
}

template <typename T>
std::vector<double> ComputationContinuous<T>::compute_partition_errors()
{
    this->require_propagators();
    // Q must be the same at every segment of every block
    typename PropagatorComputation<T>::ContourPartitions partitions(this->molecules->get_n_polymer_types());
    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        BackendArray<T>** q_left  = propagator.at(key_left).data();
        BackendArray<T>** q_right = propagator.at(key_right).data();
        for (int n = 0; n <= block.n_segment_right; n++)
            this->add_partition_sample(partitions, p, block, q_left[block.n_segment_left-n], q_right[n], nullptr);
    }
    return this->partition_errors(partitions);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationContinuous);
