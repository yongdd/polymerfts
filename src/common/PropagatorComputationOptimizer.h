/*----------------------------------------------------------
* This class defines optimizer to avoid redundent computation of chain propagators
*-----------------------------------------------------------*/

#ifndef PROPAGATOR_COMPUTATION_OPTIMIZER_H_
#define PROPAGATOR_COMPUTATION_OPTIMIZER_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>

#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorAggregator.h"

struct ComputationBlock{
    std::string monomer_type;  // monomer_type

    // When the 'aggregate_propagator_computation' is on, one block can be sliced to smaller block pieces.
    // For example, suppose the block is composed of 5'A' monomers. -> -A-A-A-A-A-
    // , and this block is sliced into 3'A' monomers and 2'A' monomers -> -A-A-A-,  -A-A-
    // For the first slice, n_segment_right, and n_segment_left are 2, and 5, respectively.
    // For the second slice, n_segment_right, and n_segment_left are 3, and 3, respectively.
    // If the 'aggregate_propagator_computation' is off, original block is not sliced to smaller block pieces.
    // In this case, n_segment_right, and n_segment_left are 5, and 5, respectively.
    int n_segment_left = 0;
    int n_segment_right = 0;
    // Weight of this entry's ∫ q q† in the concentration: the copies of the blocks it holds, or
    // one when its right propagator is an aggregated sum that carries them
    double n_repeated = 0.0;
    // Copies of the blocks in v_u (the sum of their Block::n_copies): n_repeated ∫ q q† dr is
    // n_copies times the integral of one block
    double n_copies = 0.0;
    std::vector<std::tuple<int ,int>> v_u; // node pairs <v, u> of the blocks it holds
};

/* This stucture defines comparison function for branched key */
struct ComparePropagatorKey
{
    bool operator()(const std::string& str1, const std::string& str2) const;
};

class PropagatorComputationOptimizer
{
private:
    std::string model_name; // "continuous": continuous standard Gaussian model
                            // "discrete": discrete bead-spring model
    bool aggregate_propagator_computation; // compute multiple propagators using property of linearity of the diffusion equation.

    // The contour length mapping, whose length and ds indices the keys contain
    const ContourLengthMapping* contour_length_mapping;

    // The Molecules this schedule was built from; its polymer indices size the computation's arrays
    const Molecules* molecules;

    // The key of each directed edge (v, u) of each polymer: the propagator that leaves v along the block (v, u)
    std::vector<std::map<std::pair<int, int>, std::string>> propagator_keys;

    // set{key: (polymer id, key_left, key_right) (assert(key_left <= key_right))}
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock> computation_blocks;

    // dictionary{key:non-duplicated unique propagator_codes, value: ComputationEdge}
    std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators;

    // Total segment number
    std::vector<int> total_segment_numbers;

    // Substitute right keys of lower left key with aggregated keys; the new keys are recorded in edges
    void substitute_right_keys(
        Polymer& pc,
        const std::map<std::pair<int, int>, std::string>& keys,
        std::map<std::tuple<int, int>, std::string>& v_u_to_right_key,
        std::map<std::string, std::map<std::string, ComputationBlock>> & computation_blocks_new_polymer,
        std::map<std::string, std::vector<std::string>>& aggregated_blocks,
        std::map<std::string, ComputationEdge>& edges,
        std::string left_key);

    // Add a use of the propagator 'key', which 'built' describes: it is added if new, and its
    // max_n_segment grows to new_n_segment
    void add_propagator_use(const std::string& key, const ComputationEdge& built,
                            int new_n_segment, bool is_junction_end);

    bool is_junction(Polymer& pc, int node);

    // MDE (or integral equation) steps of the polymers without and with the optimization
    std::pair<int, int> count_steps() const;

    // Add the polymers in their order in Molecules (the constructor does)
    void add_polymer(Polymer& pc, int polymer_id);

public:
    PropagatorComputationOptimizer(Molecules* molecules, bool aggregate_propagator_computation);

    // Get information of computation propagators and blocks
    bool use_aggregation() const;
    bool is_built_from(const Molecules* molecules) const;
    int get_n_computation_propagator_codes() const;
    // The key of the propagator that leaves v along the block (v, u) of a polymer
    const std::string& get_propagator_key(int polymer, int v, int u) const;
    std::map<std::string, ComputationEdge, ComparePropagatorKey>& get_computation_propagators();
    ComputationEdge& get_computation_propagator(const std::string& key);
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& get_computation_blocks();
    ComputationBlock& get_computation_block(const std::tuple<int, std::string, std::string>& key);

    // Display
    void display_statistics() const;
    void display_propagators() const;
    void display_blocks() const;
};
#endif
