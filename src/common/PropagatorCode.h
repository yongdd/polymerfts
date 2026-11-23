/**
 * @file PropagatorCode.h
 * @brief Writes the propagator keys of a polymer and records what each key computes.
 *
 * A propagator key is a string that identifies a propagator: equal keys of
 * different blocks or polymers are one computation. For example:
 * - "A+0" - A-type propagator from a free end (ds index 0)
 * - "(B2)A+0" - A-type propagator that starts where a B branch of length index 2 ends
 * - "(B2C1)A+0" - Junction: A-type propagator starting where B and C branches meet
 * - "{wall}A+0" - A-type propagator from the grafting point labelled "wall"
 *
 * A key is written once, where its propagator is built, together with its
 * ComputationEdge: the monomer type, the ds index, how it starts and the
 * propagators it starts from. The computations read these fields; a key is
 * never parsed back. Besides its identity a key gives only its height
 * (get_height_from_key), which orders the propagators.
 *
 * **Dynamic Programming:**
 *
 * For branched polymers, the same sub-propagator may be needed by multiple
 * parent propagators. Memoization ensures each unique propagator is computed
 * only once, significantly speeding up calculations for complex architectures.
 *
 * This optimization is described in:
 * J. Chem. Theory Comput. 2025, 21, 3676.
 *
 * @see Polymer for chain topology definition
 * @see PropagatorComputationOptimizer for using keys in scheduling
 * @see docs/internals/PropagatorSystem.md for the key format
 */

#ifndef PROPAGATOR_CODE_H_
#define PROPAGATOR_CODE_H_

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "Polymer.h"

// Forward declaration
class ContourLengthMapping;

/// How a propagator starts: q(r, 0) of a continuous chain, q(r, 1) of a discrete chain
enum class PropagatorStart
{
    FREE_END,    ///< q = 1 at a free chain end
    GRAFTED,     ///< q = q_init[graft_label] at a grafted chain end
    JUNCTION,    ///< Product of the branches that meet at its start, key "(...)..."
    AGGREGATED,  ///< Weighted sum of sliced propagators that continue one block, key "[...]..."
};

/**
 * @brief One propagator: what its key computes and how far its users need it.
 *
 * The first five fields are recorded where the key is written (PropagatorCode,
 * PropagatorAggregator, PropagatorComputationOptimizer::substitute_right_keys);
 * the optimizer adds the others from the blocks that use the propagator.
 */
struct ComputationEdge
{
    std::string monomer_type;                          ///< Monomer type of the block it runs along
    int ds_index = 0;                                  ///< Contour step index (ContourLengthMapping)
    PropagatorStart start = PropagatorStart::FREE_END; ///< Its initial condition
    std::string graft_label;                           ///< q_init label of a GRAFTED propagator
    /// The propagators it starts from, in the order of its key, as <key, n_segment, n_repeated>:
    /// the segment of the dependency that enters (a junction of discrete chains reads the
    /// half-bond state there), and its power in a junction product or its weight in an
    /// aggregated sum (block multiplicities make it real)
    std::vector<std::tuple<std::string, int, double>> deps;
    int height = 0;               ///< get_height_from_key(): orders the propagators and the scheduler's groups
    int max_n_segment = 0;        ///< The last segment its users need
    std::set<int> junction_ends;  ///< Segments where a junction attaches (discrete chains: half-bond states)
};

/**
 * @namespace PropagatorCode
 * @brief The functions that write propagator keys.
 *
 * **Key Format:** `{INIT}` or `(DEPS)`, then `TYPE`, then `+M`
 * - {INIT}: Initial condition label of a grafted end
 * - (DEPS): The sorted codes of the branches that meet at the start, concatenated;
 *   "B2:3" stands for three identical B2 branches. The code of a branch is its key
 *   without "+M", followed by the length index of its block (ContourLengthMapping)
 * - TYPE: Monomer type (e.g., "A", "B")
 * - M: ds index of the block
 *
 * See docs/internals/PropagatorSystem.md for aggregated keys "[...]".
 *
 * **Height:**
 *
 * The "height" of a propagator is the number of brackets that open its key:
 * - Free end propagators have height 0
 * - Junction propagators have height = max(child heights) + 1
 * Height is used for scheduling independent propagators in parallel.
 */
namespace PropagatorCode
{
/**
 * @brief Write the keys of a polymer from the finalized contour length mapping.
 *
 * Traverses the polymer graph from the chain ends, memoized per directed edge,
 * and records the ComputationEdge of every key it writes.
 *
 * @param pc        Polymer to write the keys of
 * @param mapping   Finalized contour length mapping
 * @param edge_keys Gets the key of each directed edge (v, u): the propagator that
 *                  leaves v along the block (v, u)
 * @param edges     Gets the ComputationEdge of each key
 *
 * @example
 * @code
 * std::map<std::pair<int, int>, std::string> edge_keys;
 * std::map<std::string, ComputationEdge> edges;
 * PropagatorCode::generate_keys(polymer, molecules.get_contour_length_mapping(), edge_keys, edges);
 * @endcode
 */
void generate_keys(Polymer& pc, const ContourLengthMapping& mapping,
                   std::map<std::pair<int, int>, std::string>& edge_keys,
                   std::map<std::string, ComputationEdge>& edges);

/**
 * @brief The code of a propagator as written inside another key.
 * @param key    Propagator key "DK+M"
 * @param number Length index (continuous chains) or number of segments written after it
 * @return "DK" followed by number
 */
std::string dependency_code(const std::string& key, int number);

/**
 * @brief Text of a branch count in a key, "n" of "A10:n".
 *
 * Integers keep their integer form ("3"); other counts, from block
 * multiplicities, take the shortest decimal that reads back to the same
 * double ("4.6"). Equal counts give equal keys.
 *
 * @param count Number of copies (> 0)
 * @return Count text without exponent or sign
 */
std::string format_count(double count);

/**
 * @brief Get height (tree depth) from key.
 * @param key Propagator key
 * @return Height: 0 for leaf nodes, max(child heights)+1 otherwise
 */
int get_height_from_key(const std::string& key);
}  // namespace PropagatorCode
#endif
