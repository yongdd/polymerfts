/**
 * @file PropagatorCode.cpp
 * @brief Implementation of propagator key generation and parsing.
 *
 * Provides static utility functions for generating unique string codes
 * that identify propagator computations. These codes encode the polymer
 * topology and enable detection of equivalent propagators across different
 * chains in a mixture.
 *
 * **Code Format:**
 *
 * Propagator keys encode the path from chain ends to a given edge:
 * - Simple chain end: "A10" (monomer type A, 10 segments)
 * - Junction: "(A10B5)C8" (two branches meeting, then C block)
 * - Aggregated: "[A10:2,B5]C8" (two identical A branches aggregated)
 * - Custom q_init: "{init_key}A10" (custom initial condition)
 *
 * **Key Components:**
 *
 * - Monomer type: Single letter or string (e.g., "A", "B_long")
 * - Segment count: Integer following monomer type
 * - Parentheses (): Junction point merging multiple branches
 * - Brackets []: Aggregated computation of identical branches
 * - Braces {}: Custom initial condition reference
 * - Colon: Repetition count (e.g., "A10:3" = three identical A10 branches). A block
 *   multiplicity gives real counts ("A10:4.6"): the power of the branch in a
 *   junction product, or its weight in an aggregated sum
 *
 * @see PropagatorComputationOptimizer for optimization using these codes
 * @see Polymer for chain topology
 */

#include <iostream>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <charconv>
#include <locale>
#include <sstream>
#include <stack>
#include <set>

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

namespace
{
// A branch count written by format_count, read independently of the C locale
double parse_count(const std::string& text)
{
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    double count = 0.0;
    stream >> count;
    if (stream.fail() || !stream.eof() || !(count > 0.0))
        throw_with_line_number("Invalid branch count \"" + text + "\" in a propagator key.");
    return count;
}

// Junction part "(code1code2:2...)" of the branches meeting at a node, each with its number of
// copies there. Equal codes merge and add their copies; a count of one is not written.
std::string merge_branch_codes(std::vector<std::pair<std::string, double>> branches)
{
    std::sort(branches.begin(), branches.end());
    std::string code = "(";
    for (size_t i = 0; i < branches.size(); )
    {
        double count = 0.0;
        size_t j = i;
        for (; j < branches.size() && branches[j].first == branches[i].first; j++)
            count += branches[j].second;
        code += branches[i].first;
        if (count != 1.0)
            code += ":" + PropagatorCode::format_count(count);
        i = j;
    }
    return code + ")";
}

/*
 * Code of the propagator that leaves in_node along the block (in_node, out_node), built
 * recursively from the chain ends and memoized per directed edge.
 *
 * The codes of the branches that arrive at in_node, sorted and with their copies there
 * (Polymer::get_branch_multiplicity), form the junction part "(...)"; if the copies of the
 * block itself hang at in_node, its other copies join as one more branch. A chain end without
 * branches starts with its custom initial condition "{label}", if it has one. The monomer type
 * of the block follows, then its length: the length index of the mapping if one is given,
 * else the number of segments.
 */
std::string generate_edge_code(
    std::map<std::pair<int, int>, std::string>& memory,
    Polymer& pc,
    std::map<int, std::string>& chain_end_to_q_init,
    const ContourLengthMapping* mapping,
    int in_node, int out_node)
{
    const auto edge = std::make_pair(in_node, out_node);
    if (validation::contains(memory, edge))
        return memory[edge];

    // Branches meeting at in_node, each with its number of copies there
    std::vector<std::pair<std::string, double>> branches;
    for (const int neighbor : pc.get_adjacent_nodes()[in_node])
    {
        if (neighbor != out_node)
            branches.push_back({
                generate_edge_code(memory, pc, chain_end_to_q_init, mapping, neighbor, in_node),
                pc.get_branch_multiplicity(in_node, neighbor)});
    }
    // The other copies of the branch that this propagator enters
    const double copies_out = pc.get_branch_multiplicity(in_node, out_node);
    if (copies_out != 1.0)
        branches.push_back({
            generate_edge_code(memory, pc, chain_end_to_q_init, mapping, out_node, in_node),
            copies_out - 1.0});

    std::string code;
    if (!branches.empty())
        code = merge_branch_codes(branches);
    else if (validation::contains(chain_end_to_q_init, in_node))
        code = "{" + chain_end_to_q_init[in_node] + "}";

    const Block& block = pc.get_blocks()[pc.get_block_index_from_edge(in_node, out_node)];
    code += block.monomer_type;
    code += std::to_string(mapping != nullptr ? mapping->get_length_index(block.contour_length)
                                              : block.n_segment);

    memory[edge] = code;
    return code;
}

// Codes of both directions of every block, as (from node, to node, code)
std::vector<std::tuple<int, int, std::string>> generate_all_codes(
    Polymer& pc, std::map<int, std::string>& chain_end_to_q_init, const ContourLengthMapping* mapping)
{
    std::vector<std::tuple<int, int, std::string>> propagator_codes;
    std::map<std::pair<int, int>, std::string> memory;
    for (const Block& block : pc.get_blocks())
    {
        propagator_codes.push_back(std::make_tuple(block.v, block.u,
            generate_edge_code(memory, pc, chain_end_to_q_init, mapping, block.v, block.u)));
        propagator_codes.push_back(std::make_tuple(block.u, block.v,
            generate_edge_code(memory, pc, chain_end_to_q_init, mapping, block.u, block.v)));
    }
    return propagator_codes;
}
}

/**
 * @brief Generate propagator codes for all edges in a polymer.
 *
 * Traverses the polymer graph and generates unique string codes for
 * each directed edge (v→u). Uses memoization to avoid recomputing
 * codes for shared sub-paths.
 *
 * @param pc                  Polymer graph to process
 * @param chain_end_to_q_init Map of chain ends to custom initial conditions
 *
 * @return Vector of (v, u, code) tuples for each directed edge
 */
std::vector<std::tuple<int, int, std::string>> PropagatorCode::generate_codes(
    Polymer& pc, std::map<int, std::string>& chain_end_to_q_init)
{
    return generate_all_codes(pc, chain_end_to_q_init, nullptr);
}

/**
 * @brief Generate propagator codes using length index mapping.
 *
 * Traverses the polymer graph and generates unique string codes for
 * each directed edge (v→u), using ContourLengthMapping to convert
 * contour_length to integer indices.
 *
 * @param pc                  Polymer graph to process
 * @param chain_end_to_q_init Map of chain ends to custom initial conditions
 * @param mapping             Contour length mapping (must be finalized)
 *
 * @return Vector of (v, u, code) tuples for each directed edge
 */
std::vector<std::tuple<int, int, std::string>> PropagatorCode::generate_codes_with_mapping(
    Polymer& pc, std::map<int, std::string>& chain_end_to_q_init,
    const ContourLengthMapping& mapping)
{
    return generate_all_codes(pc, chain_end_to_q_init, &mapping);
}

/**
 * @brief Extract propagator key from full code (legacy version without ds_index).
 *
 * Strips trailing length_index digits to get DK format.
 * For codes like "A3", returns "A".
 * For complex codes like "(A1B2)C3", returns "(A1B2)C".
 *
 * @param code Full propagator code (format: DKN)
 * @return Key in DK format (without ds_index)
 */
std::string PropagatorCode::get_key_from_code(std::string code)
{
    // Strip trailing digits (length_index) to get DK. A monomer type has no
    // digits but may end with an underscore.
    int pos = 0;
    for (int i = code.size() - 1; i >= 0; i--)
    {
        if (!isdigit(code[i]))
        {
            pos = i + 1;
            break;
        }
    }
    return code.substr(0, pos);
}

/**
 * @brief Extract propagator key from full code with ds_index computation.
 *
 * Converts code format DKN to key format DK+M by:
 * 1. Extracting length_index N from the code
 * 2. Looking up ds_index M from the mapping
 * 3. Returning DK+M format
 *
 * For code "A3", returns "A+1" (if length_index 3 maps to ds_index 1).
 * For complex code "(A1B2)C3", returns "(A1B2)C+1".
 *
 * @param code Full propagator code (format: DKN)
 * @param mapping ContourLengthMapping for ds_index lookup
 * @return Key in DK+M format
 */
std::string PropagatorCode::get_key_from_code(std::string code, const ContourLengthMapping& mapping)
{
    // Find the position after the last non-digit character (end of DK). A
    // monomer type has no digits but may end with an underscore.
    int pos = 0;
    for (int i = code.size() - 1; i >= 0; i--)
    {
        if (!isdigit(code[i]))
        {
            pos = i + 1;
            break;
        }
    }

    // Extract DK part and length_index
    std::string dk_part = code.substr(0, pos);
    int length_index = std::stoi(code.substr(pos));

    // Look up ds_index from length_index via mapping
    double contour_length = mapping.get_length_from_index(length_index);
    int ds_index = mapping.get_ds_index(contour_length);

    // Return DK+M format
    return dk_part + "+" + std::to_string(ds_index);
}

/**
 * @brief Parse dependency information from a propagator key.
 *
 * Extracts the sub-propagators that must be computed before this one.
 * Parses the nested structure to find immediate dependencies.
 *
 * **Example:**
 *
 * Key "(A10B5:2)C" has dependencies:
 * - ("A", 10, 1) - one A propagator with 10 segments
 * - ("B", 5, 2) - two identical B propagators with 5 segments
 *
 * For aggregated keys (DKN format inside brackets):
 * Key "[(A)B3,(C)D2]E+1" has dependencies:
 * - ("(A)B", 3, 1) - propagator (A)B up to 3 segments
 * - ("(C)D", 2, 1) - propagator (C)D up to 2 segments
 *
 * @param key Propagator key to parse
 * @return Vector of (sub_key, n_segment, n_repeated) tuples
 */
std::vector<std::tuple<std::string, int, double>> PropagatorCode::get_deps_from_key(std::string key)
{
    // sub_key, sub_n_segment, sub_n_repeated
    std::vector<std::tuple<std::string, int, double>> sub_deps;
    std::string sub_key;
    int sub_n_segment = 0;    // always assigned in state 2 before use; initialised for -Wmaybe-uninitialized
    double sub_n_repeated = 1.0;

    int pos_start = 1; // start from 1 to remove open parenthesis
    std::stack<char> stack_brace;
    int state = 1;
    // States: 1 = reading key, 2 = reading n_segment, 3 = reading n_repeated

    // First pass: check if key uses ; separator format (has ; at depth 1)
    // If so, use ; for ALL deps; otherwise use digit-based parsing
    bool use_semicolon_format = false;
    {
        std::stack<char> temp_stack;
        bool in_label = false;
        for(size_t i=0; i<key.size(); i++)
        {
            // Labels of custom initial conditions, "{label}", are opaque
            if(in_label)
            {
                if(key[i] == '}')
                    in_label = false;
                continue;
            }
            if(key[i] == '{')
                in_label = true;
            else if(key[i] == '(' || key[i] == '[')
                temp_stack.push(key[i]);
            else if(key[i] == ')' || key[i] == ']')
                temp_stack.pop();
            else if(key[i] == ';' && temp_stack.size() == 1)
            {
                use_semicolon_format = true;
                break;
            }
        }
    }

    bool in_label = false;
    for(size_t i=0; i<key.size();i++)
    {
        // Labels of custom initial conditions, "{label}", are part of the sub-key:
        // their characters (digits included) are not segment numbers. The opening
        // brace itself goes through the parser below (it ends a preceding number).
        if(in_label)
        {
            if(key[i] == '}')
                in_label = false;
            continue;
        }
        if(stack_brace.size() == 1)
        {
            // For keys using ; separator format: look for ';' to find end of key
            if(state==1 && key[i]==';')
            {
                // Remove a comma at start
                if (key[pos_start] == ',')
                    pos_start += 1;

                sub_key = key.substr(pos_start, i-pos_start);
                state = 2;
                pos_start = i+1; // skip the ';'
            }
            // For non-aggregated keys: look for first digit after alpha (length_index)
            // Only use digit-based parsing if the key doesn't use ; format
            else if(state==1 && isdigit(key[i]) && !use_semicolon_format)
            {
                // Remove a comma
                if (key[pos_start] == ',')
                    pos_start += 1;

                sub_key = key.substr(pos_start, i-pos_start);

                state = 2;
                pos_start = i;
            }
            // It was reading n_segment (or length_index) and have found a ':'
            else if(state==2 && key[i]==':')
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));

                state = 3;
                pos_start = i+1;
            }
            // It was reading n_segment (or length_index) and have found a non-digit
            else if(state==2 && !isdigit(key[i]))
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));

                sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, 1.0));

                state = 1;
                pos_start = i;
            }
            // It was reading n_repeated (digits, and a decimal point for a real count) and
            // have found another character
            else if(state==3 && !isdigit(key[i]) && key[i] != '.')
            {
                sub_n_repeated = parse_count(key.substr(pos_start, i-pos_start));

                sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, sub_n_repeated));
                state = 1;
                pos_start = i;
            }
        }
        if(key[i] == '{')
        {
            in_label = true;
        }
        else if(key[i] == '(' || key[i] == '[')
        {
            stack_brace.push(key[i]);
        }
        else if(key[i] == ')' || key[i] == ']')
        {
            if((stack_brace.top() == '(' && key[i] == ')') ||
               (stack_brace.top() == '[' && key[i] == ']'))
            {
                stack_brace.pop();
            }
            else
            {
                throw_with_line_number("Failed to match parentheses or braces.");
            }
        }
    }
    return sub_deps;
}

/**
 * @brief Remove outer monomer type from key, returning inner content.
 *
 * For keys like "(A10B5)C", returns "A10B5" (content inside parentheses).
 * For simple keys like "A", returns empty string.
 *
 * @param key Propagator key
 * @return Inner content without outer monomer type, or empty if no nesting
 */
std::string PropagatorCode::remove_monomer_type_from_key(std::string key)
{
    if (key[0] != '[' && key[0] != '(' && key[0] != '{')
    {
        return "";
    }
    else
    {
        int brace_depth = 0;
        int species_idx = 0;
        for(size_t i=0; i<key.size();i++)
        {
            if (key[i] == '[' || key[i] == '(' || key[i] == '{')
            {
                brace_depth++;
            }
            else if (key[i] == ']' || key[i] == ')' || key[i] == '}')
            {
                brace_depth--;
                if (brace_depth == 0)
                {
                    species_idx=i;
                    break;
                }
            }
        }
        return key.substr(1, species_idx-1);
    }
}

/**
 * @brief Extract monomer type from the end of a propagator key.
 *
 * Returns the monomer type string at the outermost level of the key.
 * For key in DKM format "(A1B2)C0", returns "C".
 * For simple key "A0", returns "A".
 *
 * @param key Propagator key in DKM format
 * @return Monomer type string
 */
std::string PropagatorCode::get_monomer_type_from_key(std::string key)
{
    // First, strip the +M suffix if present (new format: DK+M)
    size_t plus_pos = key.rfind('+');
    if (plus_pos != std::string::npos)
        key = key.substr(0, plus_pos);

    // Find position after last closing bracket
    int pos_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        if(key[i] == ')' || key[i] == ']' || key[i] == '}')
        {
            pos_start=i+1;
            break;
        }
    }
    // Return the monomer type (everything after last closing bracket, or entire key)
    return key.substr(pos_start, key.size()-pos_start);
}

/**
 * @brief Extract ds_index from the end of a propagator key.
 *
 * For key in DK+M format "(A1B2)C+1", returns 1.
 * For simple key "A+2", returns 2.
 * Returns -1 if no ds_index is found (no '+' separator).
 * Note: ds_index uses 1-based indexing.
 *
 * @param key Propagator key in DK+M format
 * @return ds_index value (1-based), or -1 if not present
 */
int PropagatorCode::get_ds_index_from_key(std::string key)
{
    // Find the plus sign that separates DK from M
    size_t plus_pos = key.rfind('+');
    if (plus_pos == std::string::npos || plus_pos >= key.size() - 1)
        return -1;

    // Extract and parse the ds_index after '+'
    std::string ds_index_str = key.substr(plus_pos + 1);
    return std::stoi(ds_index_str);
}

/**
 * @brief Extract custom initial condition identifier from key.
 *
 * For keys starting with "{...}", extracts the identifier inside braces.
 * Used when chain ends have custom initial conditions (q_init).
 *
 * @param key Propagator key starting with "{"
 * @return Initial condition identifier
 * @throws Exception if key doesn't start with "{"
 */
std::string PropagatorCode::get_q_input_idx_from_key(std::string key)
{
    if (key[0] != '{')
        throw_with_line_number("There is no related initial condition in key (" + key + ").");

    int pos_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        if(key[i] == '}')
        {
            pos_start=i;
            break;
        }
    }
    return key.substr(1, pos_start-1);
}

/**
 * @brief Get nesting depth (height) of a propagator key.
 *
 * Counts the number of opening parentheses/brackets at the start.
 * Height 0 = chain end, Height 1+ = junction with merged branches.
 *
 * **Examples:**
 *
 * - "A" → height 0 (chain end)
 * - "(A10)B" → height 1 (one junction)
 * - "((A10)B5)C" → height 2 (nested junctions)
 *
 * @param key Propagator key
 * @return Nesting depth (0 for chain ends)
 */
int PropagatorCode::get_height_from_key(std::string key)
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