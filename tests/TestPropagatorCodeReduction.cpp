/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestPropagatorCodeReduction.cpp
 * @brief The dependencies recorded for aggregated and re-keyed propagators of discrete chains.
 *
 * Discrete chains write the sliced segments of an aggregated propagator into its key
 * ("[...B1,...B1]B+0"), and a propagator re-keyed after an aggregation starts from the
 * aggregated propagator at its full length ("([...]B7A2)B+0"). The optimizer records the
 * dependencies of each key where it writes the key: the key of each dependency, the
 * segment it is read at and its weight or power, in the order of the key.
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "Polymer.h"

using Deps = std::vector<std::tuple<std::string, int, double>>;

// Whether the propagator key of the optimizer has this start, monomer type and dependencies
static bool edge_is(PropagatorComputationOptimizer& optimizer, const std::string& key,
                    PropagatorStart start, const std::string& monomer_type, const Deps& deps)
{
    const ComputationEdge& edge = optimizer.get_computation_propagator(key);
    const bool ok = edge.start == start && edge.monomer_type == monomer_type && edge.deps == deps;
    std::cout << (ok ? "  ok: " : "  FAILED: ") << key << std::endl;
    for (const auto& [dep_key, dep_n_segment, dep_n_repeated] : edge.deps)
        std::cout << "\t" << dep_key << ", " << dep_n_segment << ", " << PropagatorCode::format_count(dep_n_repeated) << std::endl;
    return ok;
}

int main()
{
    try
    {
        std::vector<BlockInput> blocks =
        {
            {"A", 0.2, 0, 1},
            {"A", 0.2, 1, 2},
            {"A", 0.2, 2, 3},
            {"A", 0.2, 3, 4},
            {"A", 0.2, 4, 5},
            {"B", 0.8, 1, 6},
            {"B", 0.8, 2, 7},
            {"B", 0.4, 3, 8},
            {"B", 0.2, 4, 9},
            {"B", 0.8, 6,10},
            {"B", 0.8, 7,12},
            {"A", 0.2, 6,11},
            {"A", 0.2, 7,13},
        };

        Molecules molecules("discrete", 0.1, {{"A",1.0}, {"B",1.0}});
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer propagator_computation_optimizer(&molecules, false);

        // Display all blocks and branches
        propagator_computation_optimizer.display_blocks();
        propagator_computation_optimizer.display_propagators();

        bool ok = true;
        std::cout << "-------test---------------------------" << std::endl;
        {
            // The same polymer with aggregation (length indices: 1 = 0.2, 2 = 0.4, 3 = 0.8)
            Molecules aggregated_molecules("discrete", 0.1, {{"A",1.0}, {"B",1.0}});
            aggregated_molecules.add_polymer(1.0, blocks, {});
            PropagatorComputationOptimizer optimizer(&aggregated_molecules, true);
            optimizer.display_propagators();

            // Two propagators sliced to their first segment
            const std::string aggregated = "[((((A1B1)A1B2)A1(A1B3)B3)A1A1)B1,(((A1B1)A1B2)A1((A1B3)B3A1)A1)B1]B+0";
            ok &= edge_is(optimizer, aggregated, PropagatorStart::AGGREGATED, "B",
                {{"((((A1B1)A1B2)A1(A1B3)B3)A1A1)B+0", 1, 1.0}, {"(((A1B1)A1B2)A1((A1B3)B3A1)A1)B+0", 1, 1.0}});

            // Re-keyed after the aggregation: the aggregated propagator at its 7 segments, then the
            // other branch at its last segment (a code ends with its number of segments here)
            const std::string rekeyed = "(" + aggregated.substr(0, aggregated.size() - 2) + "7A2)B+0";
            ok &= edge_is(optimizer, rekeyed, PropagatorStart::JUNCTION, "B", {{aggregated, 7, 1.0}, {"A+0", 2, 1.0}});
            ok &= edge_is(optimizer, "(" + aggregated.substr(0, aggregated.size() - 2) + "7B8)A+0", PropagatorStart::JUNCTION, "A",
                {{aggregated, 7, 1.0}, {"B+0", 8, 1.0}});

            // Aggregations nested in a hierarchy: an aggregated propagator sliced into the next one
            const std::string inner = "[" + rekeyed.substr(0, rekeyed.size() - 2) + "5,((((A1B3)B3A1)A1(A1B3)B3)A1(A1B1)A1)B1]B+0";
            ok &= edge_is(optimizer, inner, PropagatorStart::AGGREGATED, "B",
                {{rekeyed, 5, 1.0}, {"((((A1B3)B3A1)A1(A1B3)B3)A1(A1B1)A1)B+0", 1, 1.0}});
            ok &= edge_is(optimizer, "[" + inner.substr(0, inner.size() - 2) + "2,(((((A1B3)B3A1)A1(A1B3)B3)A1B2)A1A1)B1]B+0",
                PropagatorStart::AGGREGATED, "B",
                {{inner, 2, 1.0}, {"(((((A1B3)B3A1)A1(A1B3)B3)A1B2)A1A1)B+0", 1, 1.0}});
        }
        {
            // Identical side chains: an aggregated propagator weights the copies of a branch
            Molecules side_chains("discrete", 0.1, {{"A",1.0}, {"B",1.0}});
            side_chains.add_polymer(1.0, {{"A",0.3,1,2}, {"A",0.3,2,3}, {"B",0.3,1,4}, {"B",0.3,2,5}, {"B",0.3,3,6}}, {});
            PropagatorComputationOptimizer optimizer(&side_chains, true);
            ok &= edge_is(optimizer, "[(((B1)A1B1)A1)B1:2,((B1)A1:2)B1]B+0", PropagatorStart::AGGREGATED, "B",
                {{"(((B1)A1B1)A1)B+0", 1, 2.0}, {"((B1)A1:2)B+0", 1, 1.0}});
            ok &= edge_is(optimizer, "((B1)A1:2)B+0", PropagatorStart::JUNCTION, "B", {{"(B1)A+0", 3, 2.0}});
        }
        {
            // Real counts of block multiplicities: the branch powers of a junction
            Molecules multiplicities("discrete", 0.1, {{"A",1.0}, {"B",1.0}});
            multiplicities.add_polymer(1.0, {{"A", 0.2, 0, 1, 4.6}, {"B", 0.4, 1, 3}, {"B", 0.4, 0, 2, 5.6}, {"A", 0.2, 0, 4}}, {});
            PropagatorComputationOptimizer optimizer(&multiplicities, false);
            const std::string key = optimizer.get_propagator_key(0, 0, 4);
            ok &= key == "((B2)A1:4.6B2:5.6)A+0";
            ok &= edge_is(optimizer, key, PropagatorStart::JUNCTION, "A", {{"(B2)A+0", 2, 4.6}, {"B+0", 4, 5.6}});
        }
        if(PropagatorCode::format_count(5.6) != "5.6" || PropagatorCode::format_count(3.0) != "3"
            || PropagatorCode::format_count(5.6 - 1.0) != "4.6")
            return -1;

        return ok ? 0 : -1;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
