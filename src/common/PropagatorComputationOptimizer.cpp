/**
 * @file PropagatorComputationOptimizer.cpp
 * @brief Implementation of propagator computation optimizer.
 *
 * Analyzes polymer architectures to identify equivalent propagator computations
 * that can be shared across chains or within branched structures. This
 * optimization significantly reduces computational cost for complex mixtures.
 *
 * **Optimization Strategy:**
 *
 * The optimizer identifies two types of redundancy:
 *
 * 1. **Intra-polymer aggregation**: Identical branches within a single chain
 *    (e.g., star polymers with identical arms) can share propagator computation
 *
 * 2. **Inter-polymer sharing**: Different polymer species with identical
 *    sub-structures can reuse computed propagators
 *
 * **Data Structures:**
 *
 * - computation_propagators: Map of propagator keys to computation metadata
 * - computation_blocks: Map of (polymer_id, left_key, right_key) to block info
 *
 * **Algorithm Reference:**
 *
 * See *J. Chem. Theory Comput.* **2025**, 21, 3676 for the optimization algorithm.
 *
 * @see PropagatorCode for key generation
 * @see Scheduler for parallel execution scheduling
 */

#include <iostream>
#include <cctype>
#include <cmath>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "PropagatorComputationOptimizer.h"
#include "PropagatorAggregator.h"
#include "Molecules.h"
#include "Polymer.h"
#include "Exception.h"
#include "ValidationUtils.h"

namespace
{
/// The computational cost reduction of the optimization in percent, rounded to two decimals;
/// zero when there is nothing to reduce (chains of one discrete segment need no steps)
double reduction_percent(int total_steps, int reduced_steps)
{
    double percent = 0.0;
    if (total_steps > 0)
        percent = 100*(1.0 - static_cast<double>(reduced_steps)/static_cast<double>(total_steps));
    return std::round(percent*100)/100;
}
}  // namespace

/**
 * @brief Compare propagator keys for ordered storage.
 *
 * Keys are compared first by height (nesting depth), then lexicographically.
 * This ordering ensures that dependencies are processed before dependents.
 *
 * @param str1 First propagator key
 * @param str2 Second propagator key
 * @return true if str1 should come before str2
 */
bool ComparePropagatorKey::operator()(const std::string& str1, const std::string& str2) const
{
    // First compare heights
    int height_str1 = PropagatorCode::get_height_from_key(str1);
    int height_str2 = PropagatorCode::get_height_from_key(str2);

    if (height_str1 < height_str2)
        return true;
    else if(height_str1 > height_str2)
        return false;

    // Second compare their strings
    return str1 > str2;
}

/**
 * @brief Construct optimizer for given molecular system.
 *
 * Analyzes all polymer species and builds optimized computation schedule.
 * If aggregation is enabled, identifies equivalent propagators.
 *
 * @param molecules                       Molecular system to optimize
 * @param aggregate_propagator_computation Enable branch aggregation
 *
 * @throws Exception if no polymers in the system
 */
PropagatorComputationOptimizer::PropagatorComputationOptimizer(Molecules* molecules, bool aggregate_propagator_computation)
{
    if (molecules == nullptr)
        throw_with_line_number("A propagator computation optimizer needs molecules (null pointer).");
    if(molecules->get_n_polymer_types() == 0)
        throw_with_line_number("There is no chain. Add polymers first.");

    // The keys use the length and ds indices of the contour lengths of all polymers
    molecules->finalize_contour_length_mapping();

    // The mapping writes the length and ds indices into the keys
    this->contour_length_mapping = &molecules->get_contour_length_mapping();
    this->molecules = molecules;

    this->aggregate_propagator_computation = aggregate_propagator_computation;
    this->continuous = (molecules->get_model_name() == "continuous");  // Molecules takes "continuous" or "discrete"
    for(int p=0; p<molecules->get_n_polymer_types();p++)
    {
        add_polymer(molecules->get_polymer(p), p);
    }
}

/**
 * @brief Add a polymer to the optimization analysis.
 *
 * Processes the polymer's block structure and identifies computation blocks.
 * If aggregation is enabled, merges equivalent branches.
 *
 * **Steps:**
 *
 * 1. Write the propagator keys of all edges
 * 2. Apply aggregation algorithm (if enabled)
 * 3. Update global computation_blocks and computation_propagators maps
 *
 * @param pc         Polymer to add
 * @param polymer_id Index of polymer in the system
 */
void PropagatorComputationOptimizer::add_polymer(Polymer& pc, int polymer_id)
{
    // The key of each directed edge and what each key computes; aggregation records the keys it writes
    std::map<std::string, ComputationEdge> edges;
    std::map<std::pair<int, int>, std::string>& keys = propagator_keys.emplace_back();
    PropagatorCode::generate_keys(pc, *contour_length_mapping, keys, edges);

    // Temporary map for the new polymer
    std::map<std::string, std::map<std::string, ComputationBlock>> computation_blocks_new_polymer;
    std::map<std::tuple<int, int>, std::string> v_u_to_right_key;

    // Find computation_blocks in new_polymer
    std::vector<Block> blocks = pc.get_blocks();
    for(size_t b=0; b<blocks.size(); b++)
    {
        int v = blocks[b].v;
        int u = blocks[b].u;
        std::string key_left  = keys.at(std::make_pair(v, u));
        std::string key_right = keys.at(std::make_pair(u, v));

        if (key_left < key_right){
            key_left.swap(key_right);
            std::swap(v,u);
        }

        computation_blocks_new_polymer[key_left][key_right].monomer_type = blocks[b].monomer_type;
        computation_blocks_new_polymer[key_left][key_right].n_segment_right = blocks[b].n_segment;
        computation_blocks_new_polymer[key_left][key_right].n_segment_left = blocks[b].n_segment;
        computation_blocks_new_polymer[key_left][key_right].v_u.push_back(std::make_tuple(v,u));
        // Equal blocks share the entry: its weight is the copies of all of them
        computation_blocks_new_polymer[key_left][key_right].n_repeated += blocks[b].n_copies;

        v_u_to_right_key[std::make_tuple(v,u)] = key_right;
    }

    // Aggregation. Its re-keying of downstream blocks (substitute_right_keys) assumes one copy of
    // each branch at a junction, so polymers with block multiplicities are computed without it.
    bool has_multiplicity = false;
    for(const auto& block : blocks)
        has_multiplicity = has_multiplicity || (block.multiplicity != 1.0);
    if (this->aggregate_propagator_computation && !has_multiplicity)
    {
        // Aggregated keys (initially empty)
        std::map<std::string, std::vector<std::string>> aggregated_blocks;

        // Find aggregated branches in computation_blocks_new_polymer
        for(auto& item : computation_blocks_new_polymer)
        {
            // Left key and right keys
            std::string left_key = item.first;
            std::map<std::string, ComputationBlock> right_keys = item.second;


            // Aggregate propagators for given left key
            std::map<std::string, ComputationBlock> set_I;
            if (continuous)
                set_I = PropagatorAggregator::aggregate_continuous_chain(right_keys, edges);
            else
                set_I = PropagatorAggregator::aggregate_discrete_chain(right_keys, edges);

            // Replace the second map of computation_blocks_new_polymer with 'set_I'
            computation_blocks_new_polymer[left_key] = set_I;

            // Keys whose downstream blocks must be re-keyed: the new aggregated keys and the
            // keys substituted here earlier, as long as they still cover the whole block. A
            // substituted key that this aggregation sliced is computed only up to the slice;
            // the aggregated key that took over the rest carries its downstream instead.
            std::vector<std::string> full_length_keys;
            for(const auto& key : aggregated_blocks[left_key])
            {
                auto it = set_I.find(key);
                if(it != set_I.end() && it->second.n_segment_right == it->second.n_segment_left)
                    full_length_keys.push_back(key);
            }
            for(auto& item : set_I)
            {
                if(edges.at(item.first).start == PropagatorStart::AGGREGATED
                    && item.second.n_segment_right == item.second.n_segment_left)
                    full_length_keys.push_back(item.first);
            }
            aggregated_blocks[left_key] = full_length_keys;

            // Remove right keys from other left keys related to aggregated keys, and create new right keys
            substitute_right_keys(
                pc, keys, v_u_to_right_key,
                computation_blocks_new_polymer,
                aggregated_blocks, edges, left_key);
        }
    }

    // Add results to computation_blocks and computation_propagators
    for(const auto& v_item : computation_blocks_new_polymer)
    {
        for(const auto& u_item : v_item.second)
        {
            std::string key_v = v_item.first;
            std::string key_u = u_item.first;

            int n_segment_right = u_item.second.n_segment_right;
            int n_segment_left = u_item.second.n_segment_left;
            double n_repeated = u_item.second.n_repeated;
            double n_copies = 0.0;
            for(const auto& [v, u] : u_item.second.v_u)
                n_copies += pc.get_block(v, u).n_copies;

            // Add blocks
            auto key = std::make_tuple(polymer_id, key_v, key_u);

            computation_blocks[key].monomer_type = edges.at(key_v).monomer_type;
            computation_blocks[key].n_segment_right = n_segment_right;
            computation_blocks[key].n_segment_left = n_segment_left;
            computation_blocks[key].v_u = u_item.second.v_u;
            computation_blocks[key].n_repeated = n_repeated;
            computation_blocks[key].n_copies = n_copies;

            // Update propagators. A propagator that starts from others (a junction or an
            // aggregation, height above zero) makes the other one of the block end there.
            const bool is_junction_left  = !edges.at(key_v).deps.empty();
            const bool is_junction_right = !edges.at(key_u).deps.empty();
            add_propagator_use(key_v, edges.at(key_v), n_segment_left,  is_junction_right);
            add_propagator_use(key_u, edges.at(key_u), n_segment_right, is_junction_left);
        }
    }
}

/** @brief Check if aggregation is enabled. */
bool PropagatorComputationOptimizer::use_aggregation() const
{
    return aggregate_propagator_computation;
}

/** @brief Check whether the schedule was built from this Molecules object. */
bool PropagatorComputationOptimizer::is_built_from(const Molecules* molecules) const
{
    return this->molecules == molecules;
}

/**
 * @brief Update right keys after aggregation.
 *
 * When branches are aggregated, the keys of dependent propagators must
 * be updated to reference the new aggregated key.
 *
 * @param pc                              Polymer graph
 * @param keys                            The key of each directed edge of the polymer
 * @param v_u_to_right_key               Map from (v,u) to its initial right key; the entry is
 *                                       removed when the block is substituted
 * @param computation_blocks_new_polymer Block map to update
 * @param aggregated_blocks              Keys per left key whose downstream blocks are re-keyed:
 *                                       full-length aggregated keys and substituted keys
 * @param edges                          What each key computes; gets the new keys
 * @param left_key                       Left key being processed
 *
 * @throws Exception if the right key of a block would be substituted twice
 */
void PropagatorComputationOptimizer::substitute_right_keys(
    Polymer& pc,
    const std::map<std::pair<int, int>, std::string>& keys,
    std::map<std::tuple<int, int>, std::string>& v_u_to_right_key,
    std::map<std::string, std::map<std::string, ComputationBlock>> & computation_blocks_new_polymer,
    std::map<std::string, std::vector<std::string>>& aggregated_blocks,
    std::map<std::string, ComputationEdge>& edges,
    std::string left_key)
{
    for(auto& aggregated_key : aggregated_blocks[left_key])
    {
        auto& computation_block = computation_blocks_new_polymer[left_key][aggregated_key];
        const ComputationEdge& aggregated = edges.at(aggregated_key);

        // For each v_u
        for(auto& v_u : computation_block.v_u)
        {
            // (u) ----- (v) ----- (j)
            //               ----- (k)
            int v = std::get<0>(v_u);
            int u = std::get<1>(v_u);

            auto neighbor_nodes_v = pc.get_adjacent_nodes()[v];
            // Remove 'u' from 'neighbor_nodes_v'
            neighbor_nodes_v.erase(std::remove(neighbor_nodes_v.begin(), neighbor_nodes_v.end(), u), neighbor_nodes_v.end());

            // For each neighbor_node of v
            for(auto& j : neighbor_nodes_v)
            {
                const std::string& dep_j = keys.at(std::make_pair(j, v));

                // The new right key of block (j, v) starts where the aggregated propagator and the
                // other branches k meet: "(" + the aggregated code + the sorted codes of the others
                // + ")" + monomer type + "+" + ds index. A code ends with the length index of its
                // block (continuous chains) or its number of segments (discrete chains).
                ComputationEdge substituted;
                substituted.start = PropagatorStart::JUNCTION;
                substituted.monomer_type = pc.get_block(j,v).monomer_type;
                substituted.ds_index = contour_length_mapping->get_ds_index(pc.get_block(j,v).contour_length);

                const int n_aggregated = computation_block.n_segment_right;
                std::string new_u_key = "(" + PropagatorCode::dependency_code(aggregated_key, continuous
                    ? contour_length_mapping->get_length_index_from_n_segment(n_aggregated, aggregated.ds_index)
                    : n_aggregated);
                substituted.deps.push_back(std::make_tuple(aggregated_key, n_aggregated, 1.0));

                // (code, dependency) of the other branches
                std::vector<std::pair<std::string, std::tuple<std::string, int, double>>> branches;
                for(auto& k : neighbor_nodes_v)
                {
                    if (k != j)
                    {
                        const std::string& key_k = keys.at(std::make_pair(k, v));
                        const Block& block_k = pc.get_block(k,v);
                        if (continuous)
                            branches.push_back({PropagatorCode::dependency_code(key_k, contour_length_mapping->get_length_index(block_k.contour_length)),
                                                std::make_tuple(key_k, contour_length_mapping->get_n_segment(block_k.contour_length), 1.0)});
                        else
                            branches.push_back({PropagatorCode::dependency_code(key_k, block_k.n_segment),
                                                std::make_tuple(key_k, block_k.n_segment, 1.0)});
                    }
                }
                std::sort(branches.begin(), branches.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                for(auto& [code, dep] : branches)
                {
                    new_u_key += code;
                    substituted.deps.push_back(dep);
                }
                new_u_key += ")" + substituted.monomer_type + "+" + std::to_string(substituted.ds_index);
                edges.emplace(new_u_key, std::move(substituted));

                // Remove the initial right key of block (j,v). A block is substituted at most
                // once; a second substitution would leave both of its right keys in the table
                // and count its concentration twice, so it stops with an error instead.
                auto initial_right_key = v_u_to_right_key.find(std::make_tuple(j,v));
                if (initial_right_key == v_u_to_right_key.end())
                    throw_with_line_number("Propagator aggregation would substitute the right key of block ("
                        + std::to_string(j) + ", " + std::to_string(v) + ") twice. "
                        + "Set aggregate_propagator_computation to False and report this polymer as a bug.");
                computation_blocks_new_polymer[dep_j].erase(initial_right_key->second);
                v_u_to_right_key.erase(initial_right_key);

                // Add new key
                if (computation_blocks_new_polymer[dep_j].find(new_u_key) == computation_blocks_new_polymer[dep_j].end())
                {
                    computation_blocks_new_polymer[dep_j][new_u_key].monomer_type = pc.get_block(j,v).monomer_type;
                    computation_blocks_new_polymer[dep_j][new_u_key].n_segment_right = pc.get_block(j,v).n_segment;
                    computation_blocks_new_polymer[dep_j][new_u_key].n_segment_left = pc.get_block(j,v).n_segment;
                    computation_blocks_new_polymer[dep_j][new_u_key].v_u.push_back(std::make_tuple(j,v));

                    if (aggregated.start == PropagatorStart::AGGREGATED)
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated = 1;
                    else
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated = computation_block.n_repeated;

                    aggregated_blocks[dep_j].push_back(new_u_key);
                }
                else
                {
                    computation_blocks_new_polymer[dep_j][new_u_key].v_u.push_back(std::make_tuple(j,v));
                    int v0 = std::get<1>(computation_blocks_new_polymer[dep_j][new_u_key].v_u[0]);
                    if(v0 == v)
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated += computation_block.n_repeated;
                }
            }
        }
    }
}

/**
 * @brief Add a use of a propagator to computation_propagators.
 *
 * A new key enters with what its builder recorded; every use extends the
 * maximum segment count and adds its junction end.
 *
 * @param key             Propagator key
 * @param built           What the key computes (PropagatorCode, PropagatorAggregator, substitute_right_keys)
 * @param new_n_segment   Segment count for this usage
 * @param is_junction_end Whether this ends at a junction
 */
void PropagatorComputationOptimizer::add_propagator_use(
    const std::string& key, const ComputationEdge& built, int new_n_segment, bool is_junction_end)
{
    auto [it, inserted] = computation_propagators.try_emplace(key, built);
    ComputationEdge& edge = it->second;
    if (inserted)
    {
        edge.max_n_segment = new_n_segment;
        edge.height = PropagatorCode::get_height_from_key(key);
    }
    else if (edge.max_n_segment < new_n_segment)
        edge.max_n_segment = new_n_segment;
    if (is_junction_end)
        edge.junction_ends.insert(new_n_segment);
}

/**
 * @brief Check if a node is a junction point.
 *
 * Junctions have more than one adjacent node (branching points).
 * Chain ends have exactly one adjacent node, unless copies of their block hang there.
 *
 * @param pc   Polymer graph
 * @param node Node index to check
 * @return true if node is a junction
 */
bool PropagatorComputationOptimizer::is_junction(Polymer& pc, int node)
{
    // The node is an end of a block, so it has a neighbor
    const auto& neighbors = pc.get_adjacent_nodes().at(node);
    if (neighbors.size() > 1)
        return true;
    // A chain end where the copies of its block hang (multiplicity) joins them
    return pc.get_branch_multiplicity(node, neighbors[0]) != 1.0;
}

/** @brief Get number of unique propagator computations. */
int PropagatorComputationOptimizer::get_n_computation_propagator_codes() const
{
    return computation_propagators.size();
}

/** @brief Get the key of the propagator that leaves v along the block (v, u) of a polymer. */
const std::string& PropagatorComputationOptimizer::get_propagator_key(int polymer, int v, int u) const
{
    validation::require_index(polymer, static_cast<int>(propagator_keys.size()), "Polymer index");
    const auto& keys = propagator_keys[polymer];
    validation::require_key(keys, std::make_pair(v, u),
        "the blocks of polymer " + std::to_string(polymer) + " for (" + std::to_string(v) + ", " + std::to_string(u) + ")");
    return keys.at(std::make_pair(v, u));
}

/** @brief Get all computation propagators. */
std::map<std::string, ComputationEdge, ComparePropagatorKey>& PropagatorComputationOptimizer::get_computation_propagators()
{
    return computation_propagators;
}

/** @brief Get computation propagator by key. */
ComputationEdge& PropagatorComputationOptimizer::get_computation_propagator(const std::string& key)
{
    auto it = computation_propagators.find(key);
    if (it == computation_propagators.end())
        throw_with_line_number("Key '" + key + "' not found in computation_propagators");
    return it->second;
}

/** @brief Get all computation blocks. */
std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& PropagatorComputationOptimizer::get_computation_blocks()
{
    return computation_blocks;
}

/** @brief Get computation block by (polymer_id, left_key, right_key). */
ComputationBlock& PropagatorComputationOptimizer::get_computation_block(const std::tuple<int, std::string, std::string>& key)
{
    auto it = computation_blocks.find(key);
    if (it == computation_blocks.end())
        throw_with_line_number("Key not found in computation_blocks for (" + std::to_string(std::get<0>(key)) + ", "
            + std::get<1>(key) + ", " + std::get<2>(key) + ")");
    return it->second;
}

/**
 * @brief Print computation blocks for debugging.
 *
 * Displays contour length mapping followed by all blocks with their
 * aggregation status, junction flags, segment counts, and (v,u) node pairs.
 */
void PropagatorComputationOptimizer::display_blocks() const
{
    // Print contour length mapping
    if (contour_length_mapping != nullptr && contour_length_mapping->finalized())
    {
        contour_length_mapping->print_mapping();
    }

    // Print blocks
    std::cout << "--------------- Blocks ---------------" << std::endl;
    std::cout << "Polymer id, left key:\n\taggregated, (left, right) is_junction, (left, right) n_segment, right key, n_repeat, {v, u} list" << std::endl;

    const int MAX_PRINT_LENGTH = 500;
    std::tuple<int, std::string> v_tuple = std::make_tuple(-1, "");

    for(const auto& item : computation_blocks)
    {
        // Print polymer id, left key
        const std::string v_string = std::get<1>(item.first);
        if (v_tuple != std::make_tuple(std::get<0>(item.first), v_string))
        {
            std::cout << std::endl << std::to_string(std::get<0>(item.first)) + ", ";
            if (v_string.size() <= MAX_PRINT_LENGTH)
                std::cout << v_string;
            else
                std::cout << v_string.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted>, " ;
            std::cout << ":" << std::endl;
            v_tuple = std::make_tuple(std::get<0>(item.first), v_string);
        }

        // Print if aggregated
        const std::string u_string = std::get<2>(item.first);
        std::cout << "\t ";
        if (u_string.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";

        // Print is_junction (left, right)
        std::cout << "(";
        if (computation_propagators.at(v_string).height > 0)
            std::cout << "O, ";
        else
            std::cout << "X, ";

        if (computation_propagators.at(u_string).height > 0)
            std::cout << "O), ";
        else
            std::cout << "X), ";

        // Print n_segment (left, right)
        std::cout << "(" + std::to_string(item.second.n_segment_left) + ", " + std::to_string(item.second.n_segment_right) + "), ";

        // Print right key
        if (u_string.size() <= MAX_PRINT_LENGTH)
            std::cout << u_string;
        else
            std::cout << u_string.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted>" ;

        // Print n_repeat
        std::cout << ", " + PropagatorCode::format_count(item.second.n_repeated);

        // Print v_u list
        for(const auto& v_u : item.second.v_u)
        {
            std::cout << ", {"
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + "}";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Steps to compute the propagators: both directions of every block of every
 *        polymer (without the optimization), and the computation propagators (with it).
 */
std::pair<int, int> PropagatorComputationOptimizer::count_steps() const
{
    // Without the optimization: the segments of every block of every polymer, twice. A discrete
    // block of n segments takes n-1 bonds and a half bond to each junction at its ends.
    int total_mde_steps_without_reduction = 0;
    for(int p = 0; p < molecules->get_n_polymer_types(); p++)
    {
        Polymer& pc = molecules->get_polymer(p);
        int total_segment_number = 0;
        for(const Block& block : pc.get_blocks())
        {
            if (continuous)
                total_segment_number += block.n_segment;
            else
            {
                total_segment_number += block.n_segment-1;
                if(is_junction(pc, block.v))
                    total_segment_number ++;
                if(is_junction(pc, block.u))
                    total_segment_number ++;
            }
        }
        total_mde_steps_without_reduction += 2*total_segment_number;
    }

    // With it: the steps of the computation propagators
    int reduced_mde_steps = 0;
    for(const auto& item : computation_propagators)
    {
        if (continuous)
            reduced_mde_steps += item.second.max_n_segment;
        else
        {
            reduced_mde_steps += item.second.max_n_segment-1;
            reduced_mde_steps += item.second.junction_ends.size();
            if (item.second.deps.size() > 0)
                reduced_mde_steps++;
        }
    }
    return {total_mde_steps_without_reduction, reduced_mde_steps};
}

/**
 * @brief Print only the summary statistics (no detailed propagator/block info).
 *
 * Shows total computational steps before and after optimization,
 * and the efficiency gain percentage.
 */
void PropagatorComputationOptimizer::display_statistics() const
{
    const auto [total_mde_steps_without_reduction, reduced_mde_steps] = count_steps();

    if (continuous)
        std::cout << "Propagator solver: total MDE steps = " << total_mde_steps_without_reduction;
    else
        std::cout << "Propagator solver: total integral equation steps = " << total_mde_steps_without_reduction;

    std::cout << ", after optimization = " << reduced_mde_steps;
    std::cout << ", reduction = " << reduction_percent(total_mde_steps_without_reduction, reduced_mde_steps) << " %" << std::endl;
}

/**
 * @brief Print propagator list with optimization statistics.
 *
 * Shows each propagator's height, aggregation status, max segment count,
 * and dependencies. Also reports total computational cost reduction.
 */
void PropagatorComputationOptimizer::display_propagators() const
{
    // Print propagators
    const auto [total_mde_steps_without_reduction, reduced_mde_steps] = count_steps();

    std::cout << "--------------- Propagators ---------------" << std::endl;
    std::cout << "Key:\n\theight, aggregated, max_n_segment, # dependencies, junction_ends" << std::endl;

    for(const auto& item : computation_propagators)
    {
        const int MAX_PRINT_LENGTH = 500;

        if (item.first.size() <= MAX_PRINT_LENGTH)
            std::cout << item.first;
        else
            std::cout << item.first.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted> " ;

        std::cout << ":\n\t ";
        std::cout << item.second.height << ", ";
        if (item.first.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";

        // Print max_n_segment
        std::cout << item.second.max_n_segment << ", ";

        // Print number of dependency
        std::cout << item.second.deps.size() << ", ";

        // Print indices for junction_ends
        std::cout << "{";
        for (auto it = item.second.junction_ends.begin(); it != item.second.junction_ends.end(); ++it)
        {
            std::cout << *it;
            if (std::next(it) != item.second.junction_ends.end()) {
                std::cout << ", ";
            }
        }
        std::cout << "}, "<< std::endl;
    }
    if (continuous)
        std::cout << "Total number of modified diffusion equation steps (time complexity) to compute propagators: " << total_mde_steps_without_reduction << std::endl;
    else
        std::cout << "Total number of integral equation steps (time complexity) to compute propagators: " << total_mde_steps_without_reduction << std::endl;
    std::cout << "Total number of steps after optimizing computation : " << reduced_mde_steps << std::endl;
    std::cout << "Computational cost reduction (higher is better) : "
              << reduction_percent(total_mde_steps_without_reduction, reduced_mde_steps) << " %" << std::endl;
}
