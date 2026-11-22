/**
 * @file ComputationReduceMemoryContinuous.cpp
 * @brief Propagator computation for continuous chains, reduced memory.
 */

#include <algorithm>
#include <memory>
#include <cmath>

#include "ComputationReduceMemoryContinuous.h"
#include "SimpsonRule.h"

template <typename T>
ComputationReduceMemoryContinuous<T>::ComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& method,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer, method,
                                     /*first_segment=*/0, create_backend)
{
    // The start of every propagator and the segments other propagators start from
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        this->add_checkpoint(key, 0);
        for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
            this->add_checkpoint(sub_key, sub_n_segment);
    }
    this->add_regular_checkpoints();

    // One segment of each polymer gives its partition function
    for (const auto& it : this->first_block_of_each_polymer())
    {
        const auto& [p, key_left, key_right] = it->first;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
        const double n_aggregated = block.n_copies/block.n_repeated;
        this->add_checkpoint(key_left, block.n_segment_left);
        this->add_checkpoint(key_right, 0);
        single_partition_segment.push_back(std::make_tuple(
            p, key_left, block.n_segment_left, key_right, n_aggregated));
    }

    // The propagators, with the checkpoints they store and join
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        std::vector<std::pair<CheckpointArray<T>*, double>> sources;
        for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
            sources.push_back({this->checkpoint_at(sub_key, sub_n_segment), sub_n_repeated});
        jobs.push_back({&edge, std::move(sources), this->stored_segments(this->checkpoint, key, 0, edge.max_n_segment)});
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->set_fields(w_input, q_init);

    // The propagators one at a time
    for (const Job& job : jobs)
    {
        const ComputationEdge& edge = *job.edge;

        // q(r, 0)
        switch (edge.start)
        {
        case PropagatorStart::FREE_END:
            this->backend->set_unity(0, this->q_work[0]);
            break;
        case PropagatorStart::GRAFTED:
            this->backend->copy_from_host(0, this->q_work[0], q_init.at(edge.graft_label));
            break;
        case PropagatorStart::AGGREGATED:
            this->backend->junction_sum(0, this->q_work[0], job.sources);
            break;
        case PropagatorStart::JUNCTION:  // branches meeting at the start
            this->backend->junction_product(0, this->q_work[0], job.sources);
            break;
        }
        this->backend->apply_mask(0, this->q_work[0]);
        this->backend->store_checkpoint(0, job.checkpoints[0], this->q_work[0]);

        // q(r, n+1) from q(r, n), keeping the checkpointed segments
        int cur = 0;
        for (int n = 0; n < edge.max_n_segment; n++)
        {
            this->backend->advance_propagator(0, this->q_work[cur], this->q_work[1-cur], edge.monomer_type, edge.ds_index);
            cur = 1-cur;
            if (CheckpointArray<T>* stored = job.checkpoints[n+1])
                this->backend->store_checkpoint(0, stored, this->q_work[cur]);
        }
    }
    this->backend->synchronize();

    // Partition function of each polymer: Q = ∫ q(r,s) q†(r,N-s) dr / V
    for (const auto& [p, key_left, n_left, key_right, n_aggregated] : single_partition_segment)
    {
        this->backend->load_checkpoint(0, this->q_work[0], this->checkpoint_at(key_left, n_left));
        this->backend->load_checkpoint(0, this->q_work[1], this->checkpoint_at(key_right, 0));
        this->backend->synchronize();
        this->single_polymer_partitions[p] = this->backend->inner_product(this->q_work[0], this->q_work[1])
                                             /(n_aggregated*this->cb->get_volume());
    }
    this->check_real_power_partitions();

    this->propagators_computed_ = true;
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::compute_blocks(bool with_concentrations, bool with_stress)
{
    this->check_box_space_group();
    this->require_propagators();
    try
    {
        if (with_stress)
        {
            this->check_stress_supported();
            this->backend->prepare_stress();
        }

        // Concentration and stress of each block (several blocks when propagators are aggregated)
        std::vector<std::array<T, 6>> stress_of_block;
        for (const auto& [key, phi] : this->phi_block)
        {
            const auto& [p, key_left, key_right] = key;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;

            // A block without segments has no concentration and no stress
            if (with_concentrations)
                this->backend->set_zero(0, phi);
            std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (N_RIGHT > 0)
            {
                // φ = Σ_n c_n q(N_LEFT-n) q†(n) and the stress Σ_n c_n σ(q(N_LEFT-n), q†(n)),
                // Simpson's rule over the segments of the block; the getters normalize φ for
                // the requested ensemble. Every segment contributes to the stress, the chain
                // ends included (see ComputationContinuous).
                const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
                this->for_each_segment_pair(key_left, key_right, N_LEFT, 0, N_RIGHT,
                    [&](int n, BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>*)
                    {
                        if (with_concentrations)
                            this->backend->add_segment_products(0, phi, {{q, q_dagger, simpson[n]}});
                        if (with_stress)
                        {
                            const std::array<T, 6> segment = this->backend->block_stress(
                                0, {{q, q_dagger, simpson[n]*block.n_repeated}}, block.monomer_type);
                            for (int d = 0; d < 6; d++)
                                stress[d] += segment[d];
                        }
                    });
                this->scale_by_contour_step(stress, key_right);
            }
            stress_of_block.push_back(stress);
        }
        this->backend->synchronize();

        if (with_concentrations)
        {
            this->compute_solvent_concentrations();
            this->concentrations_computed_ = true;
        }
        if (with_stress)
        {
            // Sum the blocks of each polymer
            this->set_stress_from_blocks(stress_of_block);
            this->stress_computed_ = true;
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    const std::string& key = this->chain_propagator_key(polymer, v, u, n, 0);
    this->recompute_segments(key, n, 1);
    this->backend->synchronize();
    this->backend->copy_to_host(q_out, this->q_recal[0]);
}

template <typename T>
std::vector<double> ComputationReduceMemoryContinuous<T>::compute_partition_errors()
{
    this->check_box_space_group();
    this->require_propagators();
    // Q must be the same at every segment of every block
    typename ComputationCommon<T>::ContourPartitions partitions(this->molecules->get_n_polymer_types());
    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        this->for_each_segment_pair(key_left, key_right, block.n_segment_left, 0, block.n_segment_right,
            [&](int, BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>*)
            {
                this->add_partition_sample(partitions, p, block, q, q_dagger, nullptr);
            });
    }
    return this->partition_errors(partitions);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationReduceMemoryContinuous);
