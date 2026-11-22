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

#include <iostream>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "Polymer.h"

int main()
{
    try
    {
        // The native entry point needs the same guard as the Python binding.
        bool rejected_null = false;
        try
        {
            PropagatorComputationOptimizer(nullptr, false);
        }
        catch (const std::exception&)
        {
            rejected_null = true;
        }
        if (!rejected_null)
            throw std::runtime_error("A null optimizer molecules pointer was accepted.");

        std::vector<BlockInput> blocks = {
            {"A",0.4,0,1},
            {"A",1.2,0,2},
            {"B",1.2,0,5},
            {"B",0.9,0,6},
            {"A",0.9,1,4},
            {"A",1.2,1,15},
            {"B",1.2,2,3},
            {"A",0.9,2,7},
            {"B",1.2,2,10},
            {"B",1.2,3,14},
            {"A",0.9,4,8},
            {"A",1.2,4,9},
            {"B",1.2,7,19},
            {"A",0.9,8,13},
            {"B",1.2,9,12},
            {"A",1.2,9,16},
            {"A",1.2,10,11},
            {"B",1.2,13,17},
            {"A",1.2,13,18}};

        Molecules molecules("Continuous", 0.1, {{"A",1.0}, {"B",2.0}});
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer propagator_computation_optimizer(&molecules, false);

        // Display contour length mapping, blocks, and propagators
        propagator_computation_optimizer.display_blocks();
        propagator_computation_optimizer.display_propagators();

        // What a key computes is recorded where the key is written: its monomer type, its
        // start and its dependencies <key, n_segment, n_repeated> in the order of the key
        // (length indices 3, 2, 1 are 12, 9 and 4 segments)
        using Deps = std::vector<std::tuple<std::string, int, double>>;
        auto edge_is = [&](const std::string& key, const std::string& monomer_type, PropagatorStart start, const Deps& deps)
        {
            const ComputationEdge& edge = propagator_computation_optimizer.get_computation_propagator(key);
            const bool ok = edge.monomer_type == monomer_type && edge.start == start && edge.deps == deps
                && edge.ds_index == 0 && edge.graft_label.empty();
            if (!ok)
                std::cout << "Unexpected edge of " << key << std::endl;
            return ok;
        };
        if (!edge_is("B+0", "B", PropagatorStart::FREE_END, {}))
            return -1;
        if (!edge_is("(A3B3)A+0", "A", PropagatorStart::JUNCTION, {{"A+0", 12, 1.0}, {"B+0", 12, 1.0}}))
            return -1;
        if (!edge_is("((A3)B3(B3)A2(B3)B3)A+0", "A", PropagatorStart::JUNCTION,
                     {{"(A3)B+0", 12, 1.0}, {"(B3)A+0", 9, 1.0}, {"(B3)B+0", 12, 1.0}}))
            return -1;
        if (!edge_is("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)B3)A+0", "A", PropagatorStart::JUNCTION,
                     {{"(((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A+0", 12, 1.0}, {"(A3)B+0", 12, 1.0}, {"(B3)B+0", 12, 1.0}}))
            return -1;

        // Print the propagators and their dependencies
        std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators = propagator_computation_optimizer.get_computation_propagators();
        for(const auto& item : computation_propagators)
        {
            std::cout << item.first << ":\n\t";
            std::cout << "{max_n_segment: " << item.second.max_n_segment << ",\n\tsub_deps: [";
            for(const auto& [dep_key, dep_n_segment, dep_n_repeated] : item.second.deps)
                std::cout << dep_key << ":" << dep_n_segment << ", " ;
            std::cout << "]}" << std::endl;
        }

        // Every dependency is a propagator of the table, computed at least as far as it is read
        for(const auto& item : computation_propagators)
            for(const auto& [dep_key, dep_n_segment, dep_n_repeated] : item.second.deps)
                if(!computation_propagators.contains(dep_key)
                    || computation_propagators.at(dep_key).max_n_segment < dep_n_segment)
                    return -1;

        // Check size of sub_deps dictionary
        if(propagator_computation_optimizer.get_n_computation_propagator_codes() != 29)
            return -1;

        // Check max_n_segment (keys now use length_index: 3=12 segs, 2=9 segs, 1=4 segs)
        // Keys now include ds_index with + separator (e.g., "B+0" instead of "B")
        if(propagator_computation_optimizer.get_computation_propagator("B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(B3)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(B3)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(A3B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(A3)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((A3B3)A2)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((A3)B3(B3)A2(B3)B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((A3B3)A2)A2(A3B3)A3)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((A3)B3(B3)A2(B3)B3)A3B2B3)A+0").max_n_segment != 4)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3B3)A2)A2(A3B3)A3)A2A3)A+0").max_n_segment != 4)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3)B3(B3)A2(B3)B3)A3B2B3)A1(((A3B3)A2)A2(A3B3)A3)A2)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1((A3)B3(B3)A2(B3)B3)A3B2)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1((A3)B3(B3)A2(B3)B3)A3B3)B+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(B3)A2(B3)B3)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)B3)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)A2)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A+0").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A3B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A3A3)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(B3)A2(B3)B3)B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)B3)A2)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)A2)B3)B+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A2B3)A+0").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A2A3)B+0").max_n_segment != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
