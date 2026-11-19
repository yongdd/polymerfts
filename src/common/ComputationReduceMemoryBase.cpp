/**
 * @file ComputationReduceMemoryBase.cpp
 * @brief Checkpoint storage shared by the reduce-memory propagator computations.
 */

#include <algorithm>
#include <memory>
#include <cmath>
#include <iterator>

#include "ComputationReduceMemoryBase.h"

template <typename T>
ComputationReduceMemoryBase<T>::ComputationReduceMemoryBase(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& numerical_method,
    SpaceGroup* space_group,
    int first_segment,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationCommon<T>(cb, molecules, propagator_computation_optimizer, numerical_method, space_group, create_backend),
      first_segment(first_segment), checkpoint_interval(1)
{
    try
    {
        // Every scheduled job starts from a stored segment
        for (const auto& parallel_job : this->sc->get_schedule())
            for (const auto& [key, n_segment_from, n_segment_to] : parallel_job)
                if (n_segment_from >= first_segment)
                    add_checkpoint(key, n_segment_from);

        int total_n_segment = 0;
        int max_n_segment_right = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            total_n_segment += std::max(block.n_segment_left, block.n_segment_right);
            max_n_segment_right = std::max(max_n_segment_right, block.n_segment_right);
        }
        checkpoint_interval = std::max(1, static_cast<int>(std::ceil(2.0*std::sqrt(static_cast<double>(total_n_segment)))));

        // A chunk of for_each_segment_pair() never spans more than one block
        const int n_recal = std::min(checkpoint_interval, max_n_segment_right + 1);
        for (int i = 0; i < n_recal; i++)
            q_recal.push_back(this->backend->allocate());
        for (int i = 0; i < 2; i++)
        {
            q_work[i]  = this->backend->allocate();
            q_right[i] = this->backend->allocate();
        }
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object
        free_storage();
        throw_without_line_number(exc.what());
    }
}

template <typename T>
ComputationReduceMemoryBase<T>::~ComputationReduceMemoryBase()
{
    free_storage();
}

template <typename T>
void ComputationReduceMemoryBase<T>::free_storage()
{
    for (const auto& item : checkpoint)
        this->backend->deallocate_checkpoint(item.second);
    checkpoint.clear();
    for (BackendArray<T>* q : q_recal)
        this->backend->deallocate(q);
    q_recal.clear();
    for (int i = 0; i < 2; i++)
    {
        if (q_work[i] != nullptr)
            this->backend->deallocate(q_work[i]);
        if (q_right[i] != nullptr)
            this->backend->deallocate(q_right[i]);
        q_work[i]  = nullptr;
        q_right[i] = nullptr;
    }
}

template <typename T>
void ComputationReduceMemoryBase<T>::add_checkpoint(const std::string& key, int n)
{
    if (!checkpoint.contains(std::make_tuple(key, n)))
        checkpoint[std::make_tuple(key, n)] = this->backend->allocate_checkpoint();
}

template <typename T>
CheckpointArray<T>* ComputationReduceMemoryBase<T>::checkpoint_at(const std::string& key, int n)
{
    auto it = checkpoint.find(std::make_tuple(key, n));
    if (it == checkpoint.end())
        throw_with_line_number("No checkpoint at segment " + std::to_string(n) + " of propagator '" + key + "'.");
    return it->second;
}

template <typename T>
std::vector<CheckpointArray<T>*> ComputationReduceMemoryBase<T>::stored_segments(
    const std::map<std::tuple<std::string, int>, CheckpointArray<T>*>& stored,
    const std::string& key, int n_first, int n_last)
{
    std::vector<CheckpointArray<T>*> segments(n_last - n_first + 1, nullptr);
    for (auto it = stored.lower_bound(std::make_tuple(key, n_first));
         it != stored.end() && std::get<0>(it->first) == key && std::get<1>(it->first) <= n_last; ++it)
        segments[std::get<1>(it->first) - n_first] = it->second;
    return segments;
}

template <typename T>
void ComputationReduceMemoryBase<T>::add_regular_checkpoints()
{
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        for (int n = checkpoint_interval; n < edge.max_n_segment; n += checkpoint_interval)
            add_checkpoint(key, n);
}

template <typename T>
void ComputationReduceMemoryBase<T>::recompute_segments(const std::string& key, int n_first, int count)
{
    const std::string& monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;
    const int ds_index = propagator_ds_index(key);

    // Start from the last checkpoint at or before n_first, the entry before the first one past
    // (key, n_first); the first segment of a propagator always has one
    auto last_checkpoint = checkpoint.upper_bound(std::make_tuple(key, n_first));
    if (last_checkpoint == checkpoint.begin() || std::get<0>(std::prev(last_checkpoint)->first) != key)
        throw_with_line_number("No checkpoint at or before segment " + std::to_string(n_first) + " of propagator '" + key + "'.");
    --last_checkpoint;
    const int n_check = std::get<1>(last_checkpoint->first);

    if (n_check == n_first)
        this->backend->load_checkpoint(0, q_recal[0], last_checkpoint->second);
    else
    {
        this->backend->load_checkpoint(0, q_work[0], last_checkpoint->second);
        int cur = 0;
        for (int n = n_check; n < n_first-1; n++)
        {
            this->backend->advance_propagator(0, q_work[cur], q_work[1-cur], monomer_type, ds_index);
            cur = 1-cur;
        }
        this->backend->advance_propagator(0, q_work[cur], q_recal[0], monomer_type, ds_index);
    }
    for (int i = 1; i < count; i++)
        this->backend->advance_propagator(0, q_recal[i-1], q_recal[i], monomer_type, ds_index);
}

template <typename T>
void ComputationReduceMemoryBase<T>::for_each_segment_pair(
    const std::string& key_left, const std::string& key_right,
    int left_offset, int n_first, int n_last, const SegmentVisit& visit, bool past_last)
{
    const std::string& monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key_right).monomer_type;
    const int ds_index = propagator_ds_index(key_right);
    // The visits of q; q† ends at n_last
    const int n_last_q = past_last ? n_last + 1 : n_last;

    this->backend->load_checkpoint(0, q_right[0], checkpoint_at(key_right, n_first));
    int r = 0;
    for (int n_start = n_first; n_start <= n_last_q; )
    {
        // q at segments segment_first..segment_last = left_offset-n_start, so that
        // q_recal[segment - segment_first] holds q(segment). The chunk starts at the regular
        // checkpoint of q below segment_last (or at the first segment the block needs), so
        // recompute_segments() loads it instead of advancing to it; below the first
        // segment, the rest of the block is one chunk without q.
        const int segment_last  = left_offset - n_start;
        const int segment_first = std::max({first_segment, left_offset - n_last_q,
                                            segment_last / checkpoint_interval * checkpoint_interval});
        const int n_end = (segment_last >= first_segment) ? left_offset - segment_first : n_last_q;
        if (segment_last >= first_segment)
            recompute_segments(key_left, segment_first, segment_last - segment_first + 1);

        for (int n = n_start; n <= n_end; n++)
        {
            // q†(n-1) stays in the other buffer of the ping-pong pair
            BackendArray<T>* q_dagger_previous = nullptr;
            if (n > n_first)
            {
                q_dagger_previous = q_right[r];
                if (n <= n_last)
                {
                    this->backend->advance_propagator(0, q_right[r], q_right[1-r], monomer_type, ds_index);
                    r = 1-r;
                }
            }
            const int segment = left_offset - n;
            visit(n, (segment >= first_segment) ? q_recal[segment - segment_first] : nullptr,
                  (n <= n_last) ? q_right[r] : nullptr, q_dagger_previous);
        }
        n_start = n_end + 1;
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationReduceMemoryBase);
