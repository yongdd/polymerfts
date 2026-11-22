/**
 * @file ComputationDiscrete.cpp
 * @brief Propagator computation for discrete chains, full storage.
 */

#include <cmath>
#include <memory>

#include "ComputationDiscrete.h"
#include "Scheduler.h"

template <typename T>
ComputationDiscrete<T>::ComputationDiscrete(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& method,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationCommon<T>(cb, molecules, propagator_computation_optimizer, method, create_backend)
{
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        // Segments 1..max_n_segment, e.g. max_n_segment = 5:
        //   O--O--O--O--O
        //   1  2  3  4  5
        std::vector<BackendArray<T>*>& q = propagator[key];
        q.push_back(nullptr);
        for (int n = 1; n <= edge.max_n_segment; n++)
            q.push_back(this->memory.allocate());

        // Half-bond states: the junction value a junction propagator starts from
        // (n = 0) and the states half a bond past the segments where junctions attach
        std::vector<BackendArray<T>*>& q_half = propagator_half_steps[key];
        q_half.push_back(edge.start == PropagatorStart::JUNCTION ? this->memory.allocate() : nullptr);
        for (int n = 1; n <= edge.max_n_segment; n++)
            q_half.push_back(edge.junction_ends.contains(n) ? this->memory.allocate() : nullptr);
    }

    // The jobs of the scheduler: the propagator and the sources of a junction or aggregation.
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
            {
                // A junction joins half-bond states, an aggregation sliced segments
                sources.push_back({edge.start == PropagatorStart::JUNCTION ? propagator_half_steps.at(sub_key)[sub_n_segment]
                                                                           : propagator.at(sub_key)[sub_n_segment], sub_n_repeated});
            }
            jobs.push_back({&edge, propagator.at(job_key).data(), propagator_half_steps.at(job_key).data(),
                            n_segment_from, n_segment_to, std::move(sources)});
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
            propagator.at(key_right)[1],                    // q†
            block.monomer_type,
            n_aggregated));
    }
}

template <typename T>
void ComputationDiscrete<T>::compute_propagators(
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
            const std::string& monomer_type = edge.monomer_type;
            BackendArray<T>** q      = job.q;
            BackendArray<T>** q_half = job.q_half;

            // First segment q(r, 1)
            if (job.n_segment_from == 0)
            {
                switch (edge.start)
                {
                case PropagatorStart::FREE_END:    // chain end: q(1) = exp(-w ds)
                    this->backend->set_unity(STREAM, q[1]);
                    this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                    break;
                case PropagatorStart::GRAFTED:     // grafting point: q(1) = exp(-w ds) q_init
                    this->backend->copy_from_host(STREAM, q[1], q_init.at(edge.graft_label));
                    this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                    break;
                case PropagatorStart::AGGREGATED:
                    // Sum of the sliced segments, then a full bond and the next segment
                    this->backend->junction_sum(STREAM, q[1], job.sources);
                    this->backend->advance_propagator(STREAM, q[1], q[1], monomer_type, 0);
                    break;
                case PropagatorStart::JUNCTION:
                    // Junction value: product of the half-bond states of the branches
                    //     A
                    //     |
                    // O - . - B      . junction, - half bonds, O first segment
                    //     |
                    //     C
                    this->backend->junction_product(STREAM, q_half[0], job.sources);

                    // Half a bond and the first segment
                    if (job.n_segment_to > 0)
                    {
                        this->backend->advance_propagator_half_bond_step(STREAM, q_half[0], q[1], monomer_type);
                        this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                    }
                    break;
                }
            }
            if (job.n_segment_to == 0)
                return;

            int n_from = job.n_segment_from;
            if (n_from == 0)
            {
                this->backend->apply_mask(STREAM, q[1]);
                if (q_half[1] != nullptr)
                    this->backend->advance_propagator_half_bond_step(STREAM, q[1], q_half[1], monomer_type);
                n_from = 1;
            }

            // q(r, n+1) from q(r, n); discrete chains use the global ds (ds_index 0)
            for (int n = n_from; n < job.n_segment_to; n++)
                this->backend->advance_propagator(STREAM, q[n], q[n+1], monomer_type, 0);

            // q(r, n+1/2) where a junction attaches
            for (int n = n_from; n < job.n_segment_to; n++)
                if (q_half[n+1] != nullptr)
                    this->backend->advance_propagator_half_bond_step(STREAM, q[n+1], q_half[n+1], monomer_type);
        });
    }

    // Partition function of each polymer: Q = ∫ q(r,n) q†(r,N+1-n) / exp(-w ds) dr / V
    for (const auto& [p, q_left, q_right, monomer_type, n_aggregated] : single_partition_segment)
        this->single_polymer_partitions[p] = this->backend->inner_product_inverse_weight(
            q_left, q_right, this->backend->boltzmann_factor(monomer_type))
            /(n_aggregated*this->cb->get_volume());
    this->check_real_power_partitions();

    this->propagators_computed_ = true;
}

template <typename T>
void ComputationDiscrete<T>::compute_concentrations()
{
    this->check_box_space_group();
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

            // φ = Σ_n q(N_LEFT+1-n) q†(n) / exp(-w ds), n = 1..N_RIGHT
            BackendArray<T>** q_left  = propagator.at(key_left).data();
            BackendArray<T>** q_right = propagator.at(key_right).data();
            std::vector<SegmentPair<T>> pairs;
            for (int n = 1; n <= N_RIGHT; n++)
                pairs.push_back({q_left[N_LEFT-n+1], q_right[n], 1.0});
            this->backend->add_segment_products(STREAM, it->second, pairs);

            // Remove the shared bead factor; getters apply ensemble normalization.
            this->backend->divide_by_boltzmann_factor(STREAM, it->second,
                                                      this->backend->boltzmann_factor(block.monomer_type));
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
void ComputationDiscrete<T>::compute_stress()
{
    this->check_box_space_group();
    this->require_propagators();
    try
    {
        this->check_stress_supported();
        this->backend->prepare_stress();

        // Stress of each block: sum over its bonds
        std::vector<std::array<T, 6>> stress_of_block(this->phi_block.size());
        this->run_concurrently(this->phi_block_iterators.size(), [&](const int STREAM, const size_t b)
        {
            const auto it = this->phi_block_iterators[b];
            const auto& [p, key_left, key_right] = it->first;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;
            const bool left_is_junction  = this->propagator_computation_optimizer->get_computation_propagator(key_left).start == PropagatorStart::JUNCTION;
            const bool right_is_junction = this->propagator_computation_optimizer->get_computation_propagator(key_right).start == PropagatorStart::JUNCTION;
            BackendArray<T>** q_left  = propagator.at(key_left).data();
            BackendArray<T>** q_right = propagator.at(key_right).data();

            // Bond n joins q(N_LEFT-n) and q†(n). The bonds at the two ends are
            // half bonds to a junction; free ends and aggregation points have none.
            std::vector<SegmentPair<T>> pairs;
            for (int n = 0; n <= N_RIGHT; n++)
            {
                if (n == N_LEFT)                          // half bond at the left end (v)
                {
                    if (left_is_junction)
                        pairs.push_back({propagator_half_steps.at(key_left)[0], q_right[N_RIGHT],
                                         block.n_repeated, true});
                }
                else if (n == 0)                          // half bond at the right end (u)
                {
                    if (right_is_junction)
                        pairs.push_back({q_left[N_LEFT], propagator_half_steps.at(key_right)[0],
                                         block.n_repeated, true});
                }
                else                                      // full bond within the block
                {
                    pairs.push_back({q_left[N_LEFT-n], q_right[n], block.n_repeated, false});
                }
            }
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
void ComputationDiscrete<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    const std::string& key = this->chain_propagator_key(polymer, v, u, n, 1);
    this->backend->copy_to_host(q_out, propagator.at(key)[n]);
}

template <typename T>
std::vector<double> ComputationDiscrete<T>::compute_partition_errors()
{
    this->check_box_space_group();
    this->require_propagators();
    // Q must be the same at every segment of every block
    typename ComputationCommon<T>::ContourPartitions partitions(this->molecules->get_n_polymer_types());
    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        BackendArray<T>* exp_dw = this->backend->boltzmann_factor(block.monomer_type);
        BackendArray<T>** q_left  = propagator.at(key_left).data();
        BackendArray<T>** q_right = propagator.at(key_right).data();
        for (int n = 1; n <= block.n_segment_right; n++)
            this->add_partition_sample(partitions, p, block, q_left[block.n_segment_left-n+1], q_right[n], exp_dw);
    }
    return this->partition_errors(partitions);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationDiscrete);
