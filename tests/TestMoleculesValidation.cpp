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
 * @file TestMoleculesValidation.cpp
 * @brief Input checks of Molecules, Polymer and the propagator keys.
 *
 * - A monomer name may end with an underscore: the keys and the branch codes
 *   (the name, then a length index) are written and recorded with that name.
 * - Out-of-range solvent and block indices throw instead of reading past the end.
 * - A rejected polymer registers no contour lengths in the mapping.
 * - NaN lengths are rejected like non-positive ones.
 * - A polymer without blocks is rejected instead of reading an empty vertex list.
 * - The blocks must form a tree: loops (also through parallel blocks or a block
 *   from a node to itself) and disconnected blocks are rejected; trees with any
 *   non-negative node labels are accepted.
 * - A chain of one discrete segment needs no propagator steps: the statistics
 *   print a reduction of 0 %, not 0/0.
 */

#include <iostream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "ContourLengthMapping.h"
#include "ValidationUtils.h"

static bool throws(const std::function<void()>& call)
{
    try
    {
        call();
    }
    catch (std::exception&)
    {
        return true;
    }
    return false;
}

// True if the call throws an exception whose message contains the given text
static bool throws_with(const std::function<void()>& call, const std::string& text)
{
    try
    {
        call();
    }
    catch (std::exception& exc)
    {
        return std::string(exc.what()).find(text) != std::string::npos;
    }
    return false;
}

static bool check(bool condition, const std::string& what)
{
    std::cout << (condition ? "  ok: " : "  FAILED: ") << what << std::endl;
    return condition;
}

int main()
{
    bool ok = true;
    try
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();

        std::cout << "=== Monomer names ending with an underscore ===" << std::endl;
        {
            // A linear chain and a star with two identical arms (aggregated keys)
            Molecules molecules("continuous", 0.1, {{"A_", 1.0}, {"B", 1.0}, {"_", 1.5}});
            molecules.add_polymer(0.5, {{"A_", 0.5, 0, 1}, {"B", 0.5, 1, 2}}, {});
            molecules.add_polymer(0.5, {{"A_", 0.4, 0, 1}, {"A_", 0.4, 0, 2}, {"_", 0.3, 0, 3}}, {});
            PropagatorComputationOptimizer optimizer(&molecules, true);

            bool types_ok = !optimizer.get_computation_propagators().empty();
            for (const auto& [key, edge] : optimizer.get_computation_propagators())
                types_ok = types_ok && validation::contains(molecules.get_bond_lengths(), edge.monomer_type)
                    && key.ends_with(edge.monomer_type + "+" + std::to_string(edge.ds_index));
            ok &= check(types_ok, "every propagator key ends with its known monomer type and ds index");

            // The code of a branch is its key without "+M", then the length index of its block
            const ContourLengthMapping& mapping = molecules.get_contour_length_mapping();
            const std::string arm = "A_+" + std::to_string(mapping.get_ds_index(0.4));
            const std::string junction = "(A_" + std::to_string(mapping.get_length_index(0.4)) + ":2)_+"
                + std::to_string(mapping.get_ds_index(0.3));
            ok &= check(optimizer.get_propagator_key(1, 1, 0) == arm, "key " + arm + " of a free arm");
            ok &= check(optimizer.get_propagator_key(1, 0, 3) == junction, "key " + junction + " of the arms meeting");
            const ComputationEdge& edge = optimizer.get_computation_propagator(junction);
            ok &= check(edge.start == PropagatorStart::JUNCTION && edge.monomer_type == "_"
                && edge.deps == std::vector<std::tuple<std::string, int, double>>{{arm, 4, 2.0}},
                "the two A_ arms of 4 segments start " + junction);
        }
        {
            // Labels of grafted ends are opaque names: their digits are no segment numbers
            Molecules molecules("discrete", 0.1, {{"A_", 1.0}, {"B", 1.0}});
            molecules.add_polymer(1.0, {{"A_", 0.4, 0, 1}, {"B", 0.2, 1, 2}, {"A_", 0.4, 1, 3}}, {{0, "G1"}, {2, "L10"}});
            PropagatorComputationOptimizer optimizer(&molecules, false);
            const std::string key = optimizer.get_propagator_key(0, 1, 3);
            const ComputationEdge& edge = optimizer.get_computation_propagator(key);
            const ComputationEdge& graft = optimizer.get_computation_propagator("{L10}B+0");
            ok &= check(key == "({G1}A_2{L10}B1)A_+0" && edge.start == PropagatorStart::JUNCTION
                && edge.deps == std::vector<std::tuple<std::string, int, double>>{{"{G1}A_+0", 4, 1.0}, {"{L10}B+0", 2, 1.0}}
                && graft.start == PropagatorStart::GRAFTED && graft.graft_label == "L10" && graft.deps.empty(),
                "junction " + key + " of the grafted ends G1 and L10");
        }

        std::cout << "=== Index checks ===" << std::endl;
        {
            Molecules molecules("continuous", 0.1, {{"A", 1.0}, {"B", 1.0}});
            molecules.add_polymer(0.8, {{"A", 1.0, 0, 1}}, {});
            molecules.add_solvent(0.2, "B");
            ok &= check(std::get<1>(molecules.get_solvent(0)) == "B", "solvent 0");
            ok &= check(throws([&]{ molecules.get_solvent(1); }), "solvent 1 of 1 throws");
            ok &= check(throws([&]{ molecules.get_solvent(-1); }), "solvent -1 throws");

            Polymer& polymer = molecules.get_polymer(0);
            ok &= check(polymer.get_n_segment(0) == 10, "segments of block 0");
            ok &= check(throws([&]{ polymer.get_n_segment(1); }), "block 1 of 1 throws");
            ok &= check(throws([&]{ polymer.get_n_segment(-1); }), "block -1 throws");
        }

        std::cout << "=== Rejected polymers leave no contour lengths ===" << std::endl;
        {
            Molecules molecules("continuous", 0.05, {{"A", 1.0}, {"B", 1.0}});
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 0.37, 0, 1}, {"C", 0.5, 1, 2}}, {}); }),
                "a polymer with an unknown monomer type is rejected");
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 0.37, 0, 1}, {"B", nan, 1, 2}}, {}); }),
                "a polymer with a NaN contour length is rejected");
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {}, {}); }),
                "a polymer without blocks is rejected");
            molecules.add_polymer(1.0, {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}}, {});
            molecules.finalize_contour_length_mapping();
            const ContourLengthMapping& mapping = molecules.get_contour_length_mapping();
            ok &= check(molecules.get_n_polymer_types() == 1, "one polymer type");
            ok &= check(mapping.get_n_unique_lengths() == 1 && mapping.get_n_unique_ds() == 1,
                "only the accepted polymer's contour length and ds are mapped");
        }

        std::cout << "=== Polymer topology ===" << std::endl;
        {
            Molecules molecules("continuous", 0.1, {{"A", 1.0}, {"B", 1.0}});
            auto add = [&](std::vector<BlockInput> blocks) { molecules.add_polymer(0.1, blocks, {}); };

            // Trees, also with node labels that are not consecutive
            ok &= check(!throws([&]{ add({{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}}); }), "a linear chain is accepted");
            ok &= check(!throws([&]{ add({{"A", 0.4, 0, 1}, {"A", 0.4, 0, 2}, {"B", 0.3, 0, 3}}); }),
                "a three-arm star is accepted");
            ok &= check(!throws([&]{ add({{"A", 0.3, 10, 20}, {"B", 0.3, 35, 20}, {"A", 0.2, 20, 7},
                {"B", 0.2, 7, 4}}); }), "a branched polymer with arbitrary node labels is accepted");

            // Loops: a block closing a triangle, a loop between later blocks, two blocks
            // between the same nodes and a block from a node to itself
            ok &= check(throws_with([&]{ add({{"A", 0.3, 0, 1}, {"B", 0.3, 1, 2}, {"A", 0.3, 2, 0}}); },
                "Block 2 (nodes 2 and 0) closes a loop"), "a three-block ring is rejected at its closing block");
            ok &= check(throws_with([&]{ add({{"A", 0.2, 0, 1}, {"B", 0.2, 1, 2}, {"A", 0.2, 2, 3},
                {"B", 0.2, 3, 1}, {"A", 0.2, 3, 4}}); }, "Block 3 (nodes 3 and 1) closes a loop"),
                "a loop between later blocks is rejected at its closing block");
            ok &= check(throws([&]{ add({{"A", 0.5, 0, 1}, {"B", 0.5, 1, 0}}); }), "a two-block ring is rejected");
            ok &= check(throws([&]{ add({{"A", 0.5, 0, 1}, {"B", 0.5, 2, 2}}); }),
                "a block from a node to itself is rejected");

            // Disconnected blocks: every node that cannot be reached from the first one is named
            ok &= check(throws_with([&]{ add({{"A", 0.5, 0, 1}, {"B", 0.5, 2, 3}, {"A", 0.5, 3, 5}}); },
                "There is no route from node 0 to nodes: 2, 3, 5."), "disconnected blocks are rejected");

            ok &= check(molecules.get_n_polymer_types() == 3, "only the three trees were added");
        }

        std::cout << "=== Statistics of chains without propagator steps ===" << std::endl;
        {
            Molecules molecules("discrete", 1.0, {{"A", 1.0}});
            molecules.add_polymer(1.0, {{"A", 1.0, 0, 1}}, {});
            PropagatorComputationOptimizer optimizer(&molecules, true);
            std::ostringstream printed;
            std::streambuf* original = std::cout.rdbuf(printed.rdbuf());
            optimizer.display_statistics();
            std::cout.rdbuf(original);
            ok &= check(printed.str().find("reduction = 0 %") != std::string::npos,
                "a one-segment discrete chain prints a reduction of 0 % (" + printed.str().substr(0, printed.str().size() - 1) + ")");
        }

        std::cout << "=== NaN values ===" << std::endl;
        {
            Molecules molecules("continuous", 0.1, {{"A", nan}});
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 1.0, 0, 1}}, {}); }), "a NaN bond length is rejected");
        }
        // A NaN or infinite contour step used to give one segment per block and a
        // plausible partition function
        const double inf = std::numeric_limits<double>::infinity();
        for (const std::string model : {"continuous", "discrete"})
        {
            for (double ds : {nan, inf, -0.1, 0.0})
                ok &= check(throws([&]{ Molecules molecules(model, ds, {{"A", 1.0}}); }),
                    model + " chains reject ds = " + std::to_string(ds));
        }
        ok &= check(throws([&]{ ContourLengthMapping mapping(0.1, nan); }), "a NaN mapping tolerance is rejected");
        {
            ContourLengthMapping mapping(0.1);
            ok &= check(throws([&]{ mapping.add_block(nan); }) && throws([&]{ mapping.add_block(inf); }),
                "NaN and infinite contour lengths are rejected by the mapping");
        }
        ok &= check(throws([&]{ validation::require_positive(nan, "x"); }), "require_positive(NaN) throws");
        ok &= check(throws([&]{ validation::require_non_negative(nan, "x"); }), "require_non_negative(NaN) throws");
        ok &= check(throws([&]{ validation::require_in_range(nan, 0.0, 1.0, "x"); }), "require_in_range(NaN) throws");
        ok &= check(throws([&]{ validation::require_all_positive(std::vector<double>{1.0, nan}, "x"); }),
            "require_all_positive({1, NaN}) throws");
        ok &= check(!throws([&]{
            validation::require_positive(0.5, "x");
            validation::require_positive(1, "x");
            validation::require_non_negative(0, "x");
            validation::require_in_range(1.0, 0.0, 1.0, "x");
            validation::require_all_positive(std::vector<double>{1.0, 2.0}, "x");
        }), "valid values pass");
        ok &= check(throws([&]{ validation::require_positive(0, "x"); })
            && throws([&]{ validation::require_non_negative(-1, "x"); })
            && throws([&]{ validation::require_in_range(1.5, 0.0, 1.0, "x"); }),
            "out-of-range values throw");
    }
    catch (std::exception& exc)
    {
        std::cout << "Unexpected exception: " << exc.what() << std::endl;
        return -1;
    }

    if (!ok)
    {
        std::cout << "TestMoleculesValidation FAILED" << std::endl;
        return -1;
    }
    std::cout << "TestMoleculesValidation passed" << std::endl;
    return 0;
}
