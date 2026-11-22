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
 * @file TestCompareBranchKey.cpp
 * @brief The order of the propagator keys (ComparePropagatorKey).
 *
 * The optimizer keeps its propagators in this order: by height (the brackets that
 * open a key), then in descending string order. The reduced-memory computations
 * compute the propagators one at a time in this order and the scheduler groups them
 * by height, so every propagator must come after the propagators it starts from.
 * Checked on a fixed set of keys and on the propagators of branched polymers,
 * aggregated or not, of both chain models.
 */

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"

static bool check(bool condition, const std::string& what)
{
    std::cout << (condition ? "  ok: " : "  FAILED: ") << what << std::endl;
    return condition;
}

int main()
{
    try
    {
        bool ok = true;

        // Height first, then descending string order ('{' > letters > '[' > '(', "AA" > "A")
        const std::vector<std::string> expected = {
            "{G}A+0", "b+0", "a+0", "B+0", "AA+0", "A+0",
            "[A0,B0]C+0", "(B1)A+0", "(A1)B+0",
            "([A0,B0]C1)D+0", "((A1)B1)C+0"};
        std::map<std::string, int, ComparePropagatorKey> keys;
        for (auto it = expected.rbegin(); it != expected.rend(); ++it)
            keys[*it] = 0;
        std::vector<std::string> order;
        for (const auto& item : keys)
            order.push_back(item.first);
        for (const auto& key : order)
            std::cout << "    " << key << std::endl;
        ok &= check(order == expected, "keys in the order of height, then of descending strings");

        const ComparePropagatorKey less;
        bool strict = true;
        for (const auto& a : expected)
        {
            strict = strict && !less(a, a);
            for (const auto& b : expected)
                strict = strict && !(a != b && less(a, b) == less(b, a));
        }
        ok &= check(strict, "irreflexive and asymmetric, every two keys ordered");

        // Every propagator comes after the propagators it starts from, with a lower height
        const std::vector<BlockInput> star = {{"A",0.3,0,1}, {"A",0.3,0,2}, {"A",0.3,0,3}, {"B",0.3,1,4}, {"B",0.3,1,5},
                                              {"B",0.3,2,6}, {"B",0.3,2,7}, {"B",0.3,3,8}, {"B",0.3,3,9}, {"A",0.2,4,10},
                                              {"A",0.2,5,11}, {"A",0.2,6,12}, {"A",0.2,7,13}, {"A",0.2,8,14}, {"A",0.2,9,15}};
        const std::vector<BlockInput> comb = {{"A",0.2,0,1}, {"A",0.2,1,2}, {"A",0.2,2,3}, {"A",0.2,3,4}, {"A",0.2,4,5},
                                              {"B",0.8,1,6}, {"B",0.8,2,7}, {"B",0.4,3,8}, {"B",0.2,4,9}, {"B",0.8,6,10},
                                              {"B",0.8,7,12}, {"A",0.2,6,11}, {"A",0.2,7,13}};
        const std::vector<BlockInput> side_chains = {{"A",0.3,1,2}, {"A",0.3,2,3}, {"B",0.3,1,4}, {"B",0.3,2,5}, {"B",0.3,3,6},
                                                     {"A",0.3,4,7}, {"A",0.3,5,8}, {"A",0.3,6,9}, {"B",0.3,7,10},
                                                     {"B",0.3,8,11}, {"B",0.3,9,12}};
        int n_pairs = 0;
        bool after_dependencies = true;
        for (const std::string model : {"continuous", "discrete"})
        {
            for (bool aggregate : {false, true})
            {
                Molecules molecules(model, 0.1, {{"A",1.0}, {"B",1.0}});
                molecules.add_polymer(0.4, star, {{10, "G"}});
                molecules.add_polymer(0.3, comb, {});
                molecules.add_polymer(0.3, side_chains, {});
                PropagatorComputationOptimizer optimizer(&molecules, aggregate);
                for (const auto& [key, edge] : optimizer.get_computation_propagators())
                {
                    for (const auto& [dep_key, dep_n_segment, dep_n_repeated] : edge.deps)
                    {
                        n_pairs++;
                        if (!less(dep_key, key) || optimizer.get_computation_propagator(dep_key).height >= edge.height)
                        {
                            after_dependencies = false;
                            std::cout << "    " << key << " is not after " << dep_key << std::endl;
                        }
                    }
                }
            }
        }
        ok &= check(n_pairs > 0 && after_dependencies,
            "every propagator after its " + std::to_string(n_pairs) + " dependencies");

        return ok ? 0 : -1;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
