/**
 * @file PropagatorCode.cpp
 * @brief Implementation of propagator key generation.
 *
 * Writes the string keys that identify propagator computations, and the
 * ComputationEdge of each key. Equal keys detect equivalent propagators across
 * the blocks of a polymer and across the polymers of a mixture.
 *
 * **Key Format:**
 *
 * Propagator keys encode the path from chain ends to a given edge:
 * - Simple chain end: "A+0" (monomer type A, ds index 0)
 * - Junction: "(A3B1)C+0" (two branches meeting, then a C block)
 * - Aggregated: "[A0:2,B0]C+0" (identical branches aggregated, PropagatorAggregator)
 * - Custom q_init: "{init_key}A+0" (custom initial condition)
 *
 * **Key Components:**
 *
 * - Monomer type: Letters and underscores (e.g., "A", "B_long")
 * - Length index: Integer after the monomer type of a branch code
 * - Parentheses (): Junction point merging multiple branches
 * - Brackets []: Aggregated computation of identical branches
 * - Braces {}: Custom initial condition reference
 * - Colon: Repetition count (e.g., "A3:3" = three identical A3 branches). A block
 *   multiplicity gives real counts ("A3:4.6"): the power of the branch in a
 *   junction product, or its weight in an aggregated sum
 *
 * @see PropagatorComputationOptimizer for optimization using these keys
 * @see Polymer for chain topology
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <tuple>
#include <utility>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "ContourLengthMapping.h"
#include "Exception.h"
#include "ValidationUtils.h"

std::string PropagatorCode::format_count(double count)
{
    // Shortest fixed-point text that reads back to the same double: "3", "4.6"
    char buffer[400];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), count, std::chars_format::fixed);
    if (result.ec != std::errc() || !(count > 0.0))
        throw_with_line_number("Invalid branch count " + std::to_string(count) + " in a propagator key.");
    return std::string(buffer, result.ptr);
}

std::string PropagatorCode::dependency_code(const std::string& key, int number)
{
    // Every key ends with "+M"; monomer types and labels have no '+'
    return key.substr(0, key.rfind('+')) + std::to_string(number);
}

namespace
{
// The code and key of the propagator that leaves a node along a block
struct EdgeCode
{
    std::string code;  ///< "DKN": the key without "+M", then the length index of the block
    std::string key;   ///< "DK+M"
};

// A branch that arrives at a junction: its propagator, its copies there and the segment it ends at
struct Branch
{
    const EdgeCode* code;
    double count;
    int n_segment;
};

// Junction part "(code1code2:2...)" of the branches meeting at a node, each with its number of
// copies there, and the dependencies it stands for, in the same order. Equal codes merge and add
// their copies (in the order of the sorted counts); a count of one is not written.
std::string merge_branches(std::vector<Branch> branches, std::vector<std::tuple<std::string, int, double>>& deps)
{
    std::sort(branches.begin(), branches.end(), [](const Branch& a, const Branch& b)
        { return std::tie(a.code->code, a.count) < std::tie(b.code->code, b.count); });
    std::string code = "(";
    for (size_t i = 0; i < branches.size(); )
    {
        double count = 0.0;
        size_t j = i;
        for (; j < branches.size() && branches[j].code->code == branches[i].code->code; j++)
            count += branches[j].count;
        code += branches[i].code->code;
        if (count != 1.0)
            code += ":" + PropagatorCode::format_count(count);
        deps.push_back(std::make_tuple(branches[i].code->key, branches[i].n_segment, count));
        i = j;
    }
    return code + ")";
}

/*
 * Code and key of the propagator that leaves in_node along the block (in_node, out_node), built
 * from the chain ends with a heap-backed traversal and memoized per directed edge. The
 * ComputationEdge of each new key goes to edges.
 *
 * The codes of the branches that arrive at in_node, sorted and with their copies there
 * (Polymer::get_branch_multiplicity), form the junction part "(...)"; if the copies of the
 * block itself hang at in_node, its other copies join as one more branch. A chain end without
 * branches starts with its custom initial condition "{label}", if it has one. The monomer type
 * of the block follows, then the length index of the block (code) or its ds index (key).
 */
const EdgeCode& generate_edge_code(
    std::map<std::pair<int, int>, EdgeCode>& memory,
    std::map<std::string, ComputationEdge>& edges,
    Polymer& pc, const ContourLengthMapping& mapping,
    int in_node, int out_node)
{
    const auto requested_edge = std::make_pair(in_node, out_node);
    if (validation::contains(memory, requested_edge))
        return memory.at(requested_edge);

    // A linear multiblock chain can have one dependency per block. Keep its
    // traversal frames on the heap instead of exhausting the process stack.
    struct Frame
    {
        int in_node;
        int out_node;
        size_t next_neighbor = 0;
        bool copies_checked = false;
    };
    std::vector<Frame> pending{{in_node, out_node}};
    while (!pending.empty())
    {
        Frame& frame = pending.back();
        const auto edge = std::make_pair(frame.in_node, frame.out_node);
        if (validation::contains(memory, edge))
        {
            pending.pop_back();
            continue;
        }

        const auto& neighbors = pc.get_adjacent_nodes().at(frame.in_node);
        bool waiting = false;
        while (frame.next_neighbor < neighbors.size())
        {
            const int neighbor = neighbors[frame.next_neighbor++];
            if (neighbor != frame.out_node
                && !validation::contains(memory, std::make_pair(neighbor, frame.in_node)))
            {
                pending.push_back({neighbor, frame.in_node});
                waiting = true;
                break;  // push_back may have invalidated frame
            }
        }
        if (waiting)
            continue;

        // Resolve the other copies after the adjacent branches, as in the
        // original traversal. The polymer's validated multiplicities make
        // these directed dependencies acyclic as well.
        const double copies_out = pc.get_branch_multiplicity(frame.in_node, frame.out_node);
        if (!frame.copies_checked)
        {
            frame.copies_checked = true;
            if (copies_out != 1.0
                && !validation::contains(memory, std::make_pair(frame.out_node, frame.in_node)))
            {
                pending.push_back({frame.out_node, frame.in_node});
                continue;  // do not use frame after growing pending
            }
        }

        // All dependencies are memoized now. Preserve the original branch
        // order and merging so every generated key remains identical.
        auto n_segment_of = [&](int v, int u)
        {
            return mapping.get_n_segment(pc.get_blocks()[pc.get_block_index_from_edge(v, u)].contour_length);
        };
        std::vector<Branch> branches;
        for (const int neighbor : neighbors)
            if (neighbor != frame.out_node)
                branches.push_back({&memory.at(std::make_pair(neighbor, frame.in_node)),
                    pc.get_branch_multiplicity(frame.in_node, neighbor), n_segment_of(neighbor, frame.in_node)});
        if (copies_out != 1.0)
            branches.push_back({&memory.at(std::make_pair(frame.out_node, frame.in_node)), copies_out - 1.0,
                n_segment_of(frame.out_node, frame.in_node)});

        const Block& block = pc.get_blocks()[pc.get_block_index_from_edge(frame.in_node, frame.out_node)];
        ComputationEdge propagator;
        propagator.monomer_type = block.monomer_type;
        propagator.ds_index = mapping.get_ds_index(block.contour_length);
        std::string code;
        if (!branches.empty())
        {
            propagator.start = PropagatorStart::JUNCTION;
            code = merge_branches(branches, propagator.deps);
        }
        else if (validation::contains(pc.get_chain_end_to_q_init(), frame.in_node))
        {
            propagator.start = PropagatorStart::GRAFTED;
            propagator.graft_label = pc.get_chain_end_to_q_init().at(frame.in_node);
            code = "{" + propagator.graft_label + "}";
        }
        code += block.monomer_type;

        std::string key = code + "+" + std::to_string(propagator.ds_index);
        code += std::to_string(mapping.get_length_index(block.contour_length));
        edges.emplace(key, std::move(propagator));
        memory.emplace(edge, EdgeCode{std::move(code), std::move(key)});
        pending.pop_back();
    }
    return memory.at(requested_edge);
}
}  // namespace

/**
 * @brief Write the keys of both directions of every block of a polymer.
 *
 * Traverses the polymer graph and writes a key for each directed edge (v→u),
 * memoizing the codes of shared sub-paths.
 *
 * @param pc        Polymer graph to process
 * @param mapping   Contour length mapping (must be finalized)
 * @param edge_keys The key of each directed edge
 * @param edges     The ComputationEdge of each key
 */
void PropagatorCode::generate_keys(
    Polymer& pc, const ContourLengthMapping& mapping,
    std::map<std::pair<int, int>, std::string>& edge_keys,
    std::map<std::string, ComputationEdge>& edges)
{
    std::map<std::pair<int, int>, EdgeCode> memory;
    for (const Block& block : pc.get_blocks())
    {
        edge_keys[std::make_pair(block.v, block.u)] = generate_edge_code(memory, edges, pc, mapping, block.v, block.u).key;
        edge_keys[std::make_pair(block.u, block.v)] = generate_edge_code(memory, edges, pc, mapping, block.u, block.v).key;
    }
}

/**
 * @brief Get nesting depth (height) of a propagator key.
 *
 * Counts the number of opening parentheses/brackets at the start.
 * Height 0 = chain end, Height 1+ = junction with merged branches.
 *
 * **Examples:**
 *
 * - "A+0" → height 0 (chain end)
 * - "(A10)B+0" → height 1 (one junction)
 * - "((A10)B5)C+0" → height 2 (nested junctions)
 *
 * @param key Propagator key
 * @return Nesting depth (0 for chain ends)
 */
int PropagatorCode::get_height_from_key(const std::string& key)
{
    int height_count = 0;
    for(size_t i=0; i<key.size();i++)
    {
        if (key[i] == '[' || key[i] == '(')
            height_count++;
        else
            break;
    }
    return height_count;
}
