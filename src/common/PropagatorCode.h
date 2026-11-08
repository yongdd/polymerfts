/**
 * @file PropagatorCode.h
 * @brief Generates dependency codes for efficient propagator computation.
 *
 * This header provides the PropagatorCode class which generates string-based
 * codes that encode the computational dependencies between chain propagators.
 * These codes enable efficient scheduling of propagator computations using
 * dynamic programming (memoization) to avoid redundant calculations.
 *
 * **Propagator Codes:**
 *
 * A propagator code is a string that uniquely identifies a propagator and
 * encodes its dependencies. For example:
 * - "A10" - A-type propagator of length 10 from a free end
 * - "A10(B20)" - A-type propagator depending on B-type sub-propagator
 * - "A10(B20,C15)" - Junction: A-type depending on both B and C branches
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
 * @see PropagatorComputationOptimizer for using codes in scheduling
 *
 * @example
 * @code
 * // Generate propagator codes for a polymer
 * Polymer& polymer = molecules.get_polymer(0);
 * std::map<int, std::string> q_init;  // empty for free ends
 *
 * auto codes = PropagatorCode::generate_codes(polymer, q_init);
 *
 * // Each entry is (from_vertex, to_vertex, code_string)
 * for (auto& [v, u, code] : codes) {
 *     std::cout << "Edge " << v << "->" << u << ": " << code << std::endl;
 * }
 *
 * // Parse a code to get dependencies
 * std::string key = PropagatorCode::get_key_from_code(code);
 * auto deps = PropagatorCode::get_deps_from_key(key);
 * std::string monomer = PropagatorCode::get_monomer_type_from_key(key);
 * @endcode
 */

#ifndef PROPAGATOR_CODE_H_
#define PROPAGATOR_CODE_H_

#include <string>
#include <vector>
#include <map>
#include "Polymer.h"

// Forward declaration
class ContourLengthMapping;

/**
 * @class PropagatorCode
 * @brief Static utility class for generating and parsing propagator codes.
 *
 * All methods are static; the class should not be instantiated.
 * Propagator codes are strings that encode:
 * - Monomer type of the propagator
 * - Number of contour steps
 * - Dependencies on other propagators (for branched polymers)
 * - Initial condition type (for grafted chains)
 *
 * **Code Format:**
 *
 * The code format is: `TYPE` `N_SEGMENT` `(DEPS)` `[INIT]`
 * - TYPE: Monomer type (e.g., "A", "B")
 * - N_SEGMENT: Number of contour steps
 * - (DEPS): Optional comma-separated list of dependency codes
 * - [INIT]: Optional initial condition label for grafted ends
 *
 * **Height:**
 *
 * The "height" of a propagator is the depth in the dependency tree:
 * - Free end propagators have height 0
 * - Junction propagators have height = max(child heights) + 1
 * Height is used for scheduling independent propagators in parallel.
 */
class PropagatorCode
{
public:
    /**
     * @brief Generate all propagator codes for a polymer.
     *
     * Traverses the polymer graph and generates codes for all propagators
     * needed to compute partition functions and concentrations.
     *
     * @param pc               Polymer to generate codes for
     * @param chain_end_to_q_init Map of chain end vertices to initial condition labels
     * @return Vector of (from_vertex, to_vertex, code) tuples
     *
     * @note Uses top-down dynamic programming (memoization) to avoid
     *       redundant code generation for shared sub-propagators.
     *
     * @example
     * @code
     * Polymer& p = molecules.get_polymer(0);
     * std::map<int, std::string> q_init;
     * auto codes = PropagatorCode::generate_codes(p, q_init);
     * @endcode
     */
    static std::vector<std::tuple<int, int, std::string>> generate_codes(Polymer& pc, std::map<int, std::string>& chain_end_to_q_init);

    /**
     * @brief Generate propagator codes using length index mapping.
     *
     * This version uses ContourLengthMapping to generate codes with integer
     * indices representing block lengths, rather than segment counts.
     *
     * @param pc                  Polymer to generate codes for
     * @param chain_end_to_q_init Map of chain end vertices to initial condition labels
     * @param mapping             Contour length to index mapping (must be finalized)
     * @return Vector of (from_vertex, to_vertex, code) tuples
     */
    static std::vector<std::tuple<int, int, std::string>> generate_codes_with_mapping(
        Polymer& pc,
        std::map<int, std::string>& chain_end_to_q_init,
        const ContourLengthMapping& mapping);

    /**
     * @brief Extract the key (unique identifier) from a full code.
     * @param code Full propagator code string (format: DKN)
     * @return Key string in DK format (without ds_index)
     */
    static std::string get_key_from_code(std::string code);

    /**
     * @brief Extract key from code with ds_index computation.
     * @param code Full propagator code string (format: DKN)
     * @param mapping ContourLengthMapping for ds_index lookup
     * @return Key string in DK+M format (with ds_index)
     */
    static std::string get_key_from_code(std::string code, const ContourLengthMapping& mapping);

    /**
     * @brief Parse dependencies from a propagator key.
     *
     * @param key Propagator key string
     * @return Vector of (dependency_key, n_segment, n_repeated) tuples; n_repeated is the
     *         power of the dependency in a junction product or its weight in an aggregated sum
     *
     * @example
     * @code
     * // For key "A30(B20,C10)"
     * auto deps = PropagatorCode::get_deps_from_key(key);
     * // deps contains entries for B20 and C10 dependencies
     * @endcode
     */
    static std::vector<std::tuple<std::string, int, double>> get_deps_from_key(std::string key);

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
    static std::string format_count(double count);

    /**
     * @brief Remove monomer type prefix from key.
     * @param key Propagator key
     * @return Key with monomer type removed
     */
    static std::string remove_monomer_type_from_key(std::string key);

    /**
     * @brief Extract monomer type from key.
     * @param key Propagator key in DK+M format (e.g., "(A3B3)C+1")
     * @return Monomer type string (e.g., "A", "B", "C")
     */
    static std::string get_monomer_type_from_key(std::string key);

    /**
     * @brief Extract ds_index from key.
     * @param key Propagator key in DK+M format (e.g., "(A3B3)C+1")
     * @return ds_index value (e.g., 1), or -1 if not present
     */
    static int get_ds_index_from_key(std::string key);

    /**
     * @brief Extract initial condition label from key.
     * @param key Propagator key
     * @return Initial condition label, or empty string if none
     */
    static std::string get_q_input_idx_from_key(std::string key);

    /**
     * @brief Get height (tree depth) from key.
     * @param key Propagator key
     * @return Height: 0 for leaf nodes, max(child heights)+1 otherwise
     */
    static int get_height_from_key(std::string key);

};
#endif
