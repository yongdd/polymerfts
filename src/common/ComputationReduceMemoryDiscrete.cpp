/**
 * @file ComputationReduceMemoryDiscrete.cpp
 * @brief Propagator computation for discrete chains, reduced memory.
 */

#include <cmath>
#include <memory>

#include "ComputationReduceMemoryDiscrete.h"
#include "PropagatorCode.h"

template <typename T>
ComputationReduceMemoryDiscrete<T>::ComputationReduceMemoryDiscrete(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& method,
    SpaceGroup* space_group,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer, method, space_group,
                                     /*first_segment=*/1, create_backend)
{
    try
    {
        // The first segment of every propagator, the segments an aggregation
        // sums, the junction values and the half-bond states at the junction ends
        for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        {
            this->add_checkpoint(key, 1);
            // A junction joins the half-bond states of its branches (junction_ends below)
            // and keeps its junction value; an aggregation sums sliced segments
            if (key[0] == '(')
                add_half_checkpoint(key, 0);
            else
                for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                    this->add_checkpoint(sub_key, sub_n_segment);
            for (int n : edge.junction_ends)
                add_half_checkpoint(key, n);
        }
        this->add_regular_checkpoints();

        // One segment of each polymer gives its partition function
        int current_p = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            const int p = std::get<0>(key);
            if (p != current_p)
                continue;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const double n_aggregated = block.n_copies/block.n_repeated;
            this->add_checkpoint(std::get<1>(key), block.n_segment_left);
            this->add_checkpoint(std::get<2>(key), 1);
            single_partition_segment.push_back(std::make_tuple(
                p, std::get<1>(key), block.n_segment_left, std::get<2>(key), block.monomer_type, n_aggregated));
            current_p++;
        }

        // The propagators, with the checkpoints they store and join
        for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::vector<std::pair<CheckpointArray<T>*, double>> sources;
            for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                sources.push_back({key[0] == '(' ? half_checkpoint_at(sub_key, sub_n_segment)
                                                 : this->checkpoint_at(sub_key, sub_n_segment), sub_n_repeated});
            jobs.push_back({&key, &edge, std::move(sources),
                            this->stored_segments(this->checkpoint, key, 0, edge.max_n_segment),
                            this->stored_segments(half_checkpoint, key, 0, edge.max_n_segment)});
        }
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object; the
        // base class frees the checkpoints it holds
        free_half_checkpoints();
        throw_without_line_number(exc.what());
    }
}

template <typename T>
ComputationReduceMemoryDiscrete<T>::~ComputationReduceMemoryDiscrete()
{
    free_half_checkpoints();
}

template <typename T>
void ComputationReduceMemoryDiscrete<T>::free_half_checkpoints()
{
    for (const auto& item : half_checkpoint)
        this->backend->deallocate_checkpoint(item.second);
    half_checkpoint.clear();
}

template <typename T>
void ComputationReduceMemoryDiscrete<T>::add_half_checkpoint(const std::string& key, int n)
{
    if (!half_checkpoint.contains(std::make_tuple(key, n)))
        half_checkpoint[std::make_tuple(key, n)] = this->backend->allocate_checkpoint();
}

template <typename T>
CheckpointArray<T>* ComputationReduceMemoryDiscrete<T>::half_checkpoint_at(const std::string& key, int n)
{
    auto it = half_checkpoint.find(std::make_tuple(key, n));
    if (it == half_checkpoint.end())
        throw_with_line_number("No half-bond checkpoint at segment " + std::to_string(n) + " of propagator '" + key + "'.");
    return it->second;
}

template <typename T>
void ComputationReduceMemoryDiscrete<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->set_fields(w_input, q_init);

    // The propagators one at a time
    for (const Job& job : jobs)
    {
        const std::string& key = *job.key;
        const ComputationEdge& edge = *job.edge;
        const std::string& monomer_type = edge.monomer_type;
        BackendArray<T>** q = this->q_work;  // q[cur] holds the current segment
        int cur = 0;

        // First segment q(r, 1)
        if (edge.deps.size() == 0)  // chain end: q(1) = exp(-w ds) q_init
        {
            if (key[0] == '{')      // grafting point
                this->backend->copy_from_host(0, q[0], q_init.at(PropagatorCode::get_q_input_idx_from_key(key)));
            else                    // free chain end
                this->backend->set_unity(0, q[0]);
            this->backend->multiply_boltzmann_factor(0, q[0], monomer_type);
        }
        else if (key[0] == '[')     // aggregated propagators
        {
            // Sum of the sliced segments, then a full bond and the next segment
            this->backend->junction_sum(0, q[0], job.sources);
            this->backend->advance_propagator(0, q[0], q[0], monomer_type, 0);
        }
        else if (key[0] == '(')     // junction of branches
        {
            // Junction value: product of the half-bond states of the branches
            this->backend->junction_product(0, q[0], job.sources);
            this->backend->store_checkpoint(0, job.half_checkpoints[0], q[0]);

            // Half a bond and the first segment
            if (edge.max_n_segment > 0)
            {
                this->backend->advance_propagator_half_bond_step(0, q[0], q[1], monomer_type);
                this->backend->multiply_boltzmann_factor(0, q[1], monomer_type);
                cur = 1;
            }
        }
        if (edge.max_n_segment == 0)
            continue;

        this->backend->apply_mask(0, q[cur]);
        this->backend->store_checkpoint(0, job.checkpoints[1], q[cur]);
        if (CheckpointArray<T>* half = job.half_checkpoints[1])
        {
            this->backend->advance_propagator_half_bond_step(0, q[cur], q[1-cur], monomer_type);
            this->backend->store_checkpoint(0, half, q[1-cur]);
        }

        // q(r, n+1) from q(r, n), keeping the checkpointed segments and
        // the half-bond states where a junction attaches
        for (int n = 1; n < edge.max_n_segment; n++)
        {
            this->backend->advance_propagator(0, q[cur], q[1-cur], monomer_type, 0);
            cur = 1-cur;
            if (CheckpointArray<T>* stored = job.checkpoints[n+1])
                this->backend->store_checkpoint(0, stored, q[cur]);
            if (CheckpointArray<T>* half = job.half_checkpoints[n+1])
            {
                this->backend->advance_propagator_half_bond_step(0, q[cur], q[1-cur], monomer_type);
                this->backend->store_checkpoint(0, half, q[1-cur]);
            }
        }
    }
    this->backend->synchronize();

    // Partition function of each polymer: Q = ∫ q(r,n) q†(r,N+1-n) / exp(-w ds) dr / V
    for (const auto& [p, key_left, n_left, key_right, monomer_type, n_aggregated] : single_partition_segment)
    {
        this->backend->load_checkpoint(0, this->q_work[0], this->checkpoint_at(key_left, n_left));
        this->backend->load_checkpoint(0, this->q_work[1], this->checkpoint_at(key_right, 1));
        this->backend->synchronize();
        this->single_polymer_partitions[p] = this->backend->inner_product_inverse_weight(
            this->q_work[0], this->q_work[1], this->backend->boltzmann_factor(monomer_type))
            /(n_aggregated*this->cb->get_volume());
    }
    this->check_real_power_partitions();

    this->propagators_computed_ = true;
}

template <typename T>
void ComputationReduceMemoryDiscrete<T>::advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u)
{
    this->check_box_space_group();
    // The step uses the fields and the cell of the last compute_propagators()
    this->require_propagators();
    const std::string monomer_type = this->molecules->get_polymer(p).get_block(v, u).monomer_type;

    this->backend->copy_from_host(0, this->q_work[0], q_init);
    this->backend->advance_propagator(0, this->q_work[0], this->q_work[1], monomer_type, 0);
    this->backend->synchronize();
    this->backend->copy_to_host(q_out, this->q_work[1]);
}

template <typename T>
void ComputationReduceMemoryDiscrete<T>::compute_blocks(bool with_concentrations, bool with_stress)
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
            const double weight = block.n_repeated;
            const bool left_is_leaf  = this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0;
            const bool right_is_leaf = this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0;

            if (with_concentrations)
                this->backend->set_zero(0, phi);
            std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            auto add_bond = [&](BackendArray<T>* q, BackendArray<T>* q_dagger, bool is_half_bond_length)
            {
                const std::array<T, 6> bond = this->backend->block_stress(
                    0, {{q, q_dagger, weight, is_half_bond_length}}, block.monomer_type);
                for (int d = 0; d < 6; d++)
                    stress[d] += bond[d];
            };

            // φ = Σ_n q(N_LEFT+1-n) q†(n) / exp(-w ds), n = 1..N_RIGHT. Bond n joins
            // q(N_LEFT-n) and q†(n), n = 0..N_RIGHT (see ComputationDiscrete): the walk
            // meets bond n at segment n+1, which holds the same q, with q†(n) as the
            // previous q†; it goes one segment past N_RIGHT for the last bond. The
            // bonds at the two ends are half bonds to a junction; free ends and
            // aggregation points have none.
            this->for_each_segment_pair(key_left, key_right, N_LEFT+1, 1, N_RIGHT,
                [&](int n, BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>* q_dagger_previous)
                {
                    if (with_concentrations && n <= N_RIGHT)
                        this->backend->add_segment_products(0, phi, {{q, q_dagger, 1.0}});
                    if (!with_stress)
                        return;
                    if (n == 1)                   // bond 0: half bond at the right end, a junction
                    {
                        if (key_right[0] == '(' && !right_is_leaf)
                        {
                            this->backend->load_checkpoint(0, this->q_work[0], half_checkpoint_at(key_right, 0));
                            add_bond(q, this->q_work[0], true);
                        }
                    }
                    else if (n - 1 == N_LEFT)     // bond N_LEFT: half bond at the left end
                    {
                        if (!left_is_leaf)
                        {
                            this->backend->load_checkpoint(0, this->q_work[0], half_checkpoint_at(key_left, 0));
                            add_bond(this->q_work[0], q_dagger_previous, true);
                        }
                    }
                    else                          // bond n-1 within the block
                        add_bond(q, q_dagger_previous, false);
                }, with_stress);

            if (with_concentrations)
            {
                // Remove the shared bead factor; getters apply ensemble normalization.
                this->backend->divide_by_boltzmann_factor(0, phi, this->backend->boltzmann_factor(block.monomer_type));
            }
            this->scale_by_contour_step(stress, key_right);
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
void ComputationReduceMemoryDiscrete<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    this->check_box_space_group();
    this->require_propagators();
    // For debugging and testing: the propagator of one block and direction
    Polymer& pc = this->molecules->get_polymer(polymer);
    const std::string dep = pc.get_propagator_key(v, u);

    if (!this->propagator_computation_optimizer->get_computation_propagators().contains(dep))
        throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

    const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
    if (n < 1 || n > N_RIGHT)
        throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

    this->recompute_segments(dep, n, 1);
    this->backend->synchronize();
    this->backend->copy_to_host(q_out, this->q_recal[0]);
}

template <typename T>
std::vector<double> ComputationReduceMemoryDiscrete<T>::compute_partition_errors()
{
    this->check_box_space_group();
    this->require_propagators();
    // Q must be the same at every segment of every block
    const int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions(n_polymer_types);
    std::vector<std::vector<double>> absolute_integrals(n_polymer_types);

    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        const double n_propagators = block.n_copies;
        BackendArray<T>* exp_dw = this->backend->boltzmann_factor(block.monomer_type);

        if (block.n_segment_right > 0)
            this->for_each_segment_pair(key_left, key_right, block.n_segment_left+1, 1, block.n_segment_right,
                [&](int, BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>*)
                {
                    T total_partition = this->backend->inner_product_inverse_weight(q, q_dagger, exp_dw)
                                        *(block.n_repeated/this->cb->get_volume());
                    total_partition /= n_propagators;
                    total_partitions[p].push_back(total_partition);
                    if (this->molecules->get_polymer(p).is_absent())
                    {
                        double integral = this->backend->absolute_inner_product(q, q_dagger, exp_dw)
                                          *(block.n_repeated/this->cb->get_volume());
                        integral /= n_propagators;
                        absolute_integrals[p].push_back(integral);
                    }
                });
    }

    return this->partition_errors_from_contours(total_partitions, absolute_integrals);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationReduceMemoryDiscrete);
