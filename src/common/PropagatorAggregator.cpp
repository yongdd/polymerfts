/**
 * @file PropagatorAggregator.cpp
 * @brief Implementation of propagator aggregation algorithms.
 *
 * Implements chain-model-specific algorithms for identifying and merging
 * equivalent propagator computations to reduce computational cost.
 *
 * @see PropagatorAggregator.h for class documentation
 */

#include <iostream>
#include <algorithm>
#include <set>
#include <utility>

#include "PropagatorAggregator.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorCode.h"

namespace
{
/*
 * The block of the aggregated propagator that sums the sliced propagators of slices and continues
 * for n_segment segments, and its key, which edges gets. Its key "[k1n1:c1,k2n2,...]M+ds" lists
 * the slices in reverse ComparePropagatorKey order: each key without "+ds", the segment it was
 * sliced to (n_segment_right in blocks) and its copies when more than one. The slices all
 * continue blocks of one left key, so they share the monomer type and ds index.
 */
std::pair<std::string, ComputationBlock> aggregated_block(
    const std::map<std::string, ComputationBlock, ComparePropagatorKey>& slices,
    const std::map<std::string, ComputationBlock>& blocks,
    int n_segment,
    std::map<std::string, ComputationEdge>& edges)
{
    ComputationEdge aggregated;
    aggregated.start = PropagatorStart::AGGREGATED;
    aggregated.monomer_type = slices.begin()->second.monomer_type;
    aggregated.ds_index = edges.at(slices.rbegin()->first).ds_index;

    ComputationBlock block;
    block.monomer_type = aggregated.monomer_type;
    block.n_segment_right = n_segment;
    block.n_segment_left = n_segment;
    block.n_repeated = 1;

    std::string key = "[";
    for (auto it = slices.rbegin(); it != slices.rend(); it++)
    {
        const ComputationBlock& slice = blocks.at(it->first);
        if (it != slices.rbegin())
            key += ",";
        key += PropagatorCode::dependency_code(it->first, slice.n_segment_right);
        if (slice.n_repeated > 1)
            key += ":" + PropagatorCode::format_count(slice.n_repeated);
        aggregated.deps.push_back(std::make_tuple(it->first, slice.n_segment_right, slice.n_repeated));
        block.v_u.insert(block.v_u.end(), slice.v_u.begin(), slice.v_u.end());
    }
    key += "]" + aggregated.monomer_type + "+" + std::to_string(aggregated.ds_index);
    edges.emplace(key, std::move(aggregated));
    return {key, block};
}
}  // namespace

std::map<std::string, ComputationBlock> PropagatorAggregator::aggregate_continuous_chain(
    std::map<std::string, ComputationBlock> set_I, std::map<std::string, ComputationEdge>& edges)
{
    // Example) Continuous chain aggregation
    //
    // Input (key format: DK+ds_index):
    //   n_segment_right, key,       n_segment_left, n_repeated
    //   6,               (C)B+0,    6,              1
    //   4,               (D)B+0,    4,              3
    //   4,               (E)B+0,    4,              2
    //   2,               (F)B+0,    2,              1
    //
    //      ↓   Aggregation (sliced propagators: n_segment=0)
    //
    // Output:
    //   6,               (C)B+0,              6,  1
    //   4 → 0,           (D)B+0,              4,  3  (sliced)
    //   4 → 0,           (E)B+0,              4,  2  (sliced)
    //   4,               [(D)B0:3,(E)B0:2]B+0, 4,  1  (aggregated)
    //   2,               (F)B+0,              2,  1

    // Collect unique n_segment values
    std::set<int> set_n_compute;
    for (const auto& item : set_I)
        set_n_compute.insert(item.second.n_segment_right);

    // Aggregate propagators with same n_segment
    for (const int n_segment_current : set_n_compute)
    {
        // Collect propagators with this n_segment
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for (const auto& item : set_I)
        {
            if (item.second.n_segment_right == n_segment_current)
                set_S[item.first] = item.second;
        }

        // Skip if nothing to aggregate
        if (set_S.size() == 1)
            continue;

        // Set sliced propagators to n_segment=0
        for (const auto& item : set_S)
            set_I[item.first].n_segment_right = 0;

        // Add aggregated block: [dep1,dep2,...]monomer+ds_index
        auto [aggregated_key, block] = aggregated_block(set_S, set_I, n_segment_current, edges);
        set_I[aggregated_key] = block;
    }
    return set_I;
}

std::map<std::string, ComputationBlock> PropagatorAggregator::aggregate_discrete_chain(
    std::map<std::string, ComputationBlock> set_I, std::map<std::string, ComputationEdge>& edges)
{
    // Example) Discrete chain aggregation (hierarchical merging)
    //
    // Input (key format: DK+ds_index, uses n_segment directly):
    //   n_segment_right, key,       n_segment_left, n_repeated
    //   6,               (C)B+0,    6,              1
    //   4,               (D)B+0,    4,              3
    //   4,               (E)B+0,    4,              2
    //   2,               (F)B+0,    2,              1
    //
    //      ↓   Aggregation (minimum_n_segment=1 for discrete chains)
    //
    // Output:
    //   6,               (C)B+0,              6,  1
    //   4 → 1,           (D)B+0,              4,  3  (sliced to n_segment=1)
    //   4 → 1,           (E)B+0,              4,  2  (sliced to n_segment=1)
    //   3,               [(D)B1:3,(E)B1:2]B+0, 3,  1  (aggregated, n_segment=4-1=3)
    //   2,               (F)B+0,              2,  1

    constexpr int MIN_N_SEGMENT = 1;  // Discrete chains require at least 1 segment

    // set_F: final result, set_I: working set for hierarchical merging
    std::map<std::string, ComputationBlock> set_F = set_I;

    // Remove propagators with n_segment <= 1 from working set (can't be aggregated further)
    for (const auto& item : set_F)
    {
        if (item.second.n_segment_right <= MIN_N_SEGMENT)
            set_I.erase(item.first);
    }

    // Hierarchical merging: repeatedly merge propagators with largest n_segment values
    while (set_I.size() > 1)
    {
        // Find 2nd largest n_segment
        std::vector<int> n_segments;
        for (const auto& item : set_I)
            n_segments.push_back(item.second.n_segment_right);
        std::sort(n_segments.rbegin(), n_segments.rend());
        int n_segment_threshold = n_segments[1];

        // Collect propagators with n_segment >= threshold
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for (const auto& item : set_I)
        {
            if (item.second.n_segment_right >= n_segment_threshold)
                set_S[item.first] = item.second;
        }

        // Slice: reduce n_segment_right by (threshold - 1)
        int slice_amount = n_segment_threshold - MIN_N_SEGMENT;
        for (const auto& item : set_S)
            set_F[item.first].n_segment_right -= slice_amount;

        // Add aggregated block to both sets: [dep1,dep2,...]monomer+ds_index
        auto [aggregated_key, block] = aggregated_block(set_S, set_F, n_segment_threshold - MIN_N_SEGMENT, edges);
        set_F[aggregated_key] = block;
        set_I[aggregated_key] = block;

        // Remove merged propagators from working set
        for (const auto& item : set_S)
            set_I.erase(item.first);
    }

    return set_F;
}
