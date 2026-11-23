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
 * @file TestContourLengthMapping.cpp
 * @brief Unit test for ContourLengthMapping class.
 *
 * Tests the mapping of floating-point contour lengths and local Δs values
 * to integer indices. Each topic is one function; every topic runs, and the
 * test fails if one of them does.
 */

#include <iostream>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <limits>

#include "Molecules.h"
#include "ContourLengthMapping.h"

namespace {

const std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};

/// Prints why a check of a topic failed; false
bool fail(const std::string& why)
{
    std::cout << "FAILED: " << why << std::endl;
    return false;
}

/// Lengths of one global ds: duplicates merged, indices in sorted order, segment counts and local ds
bool check_basic_mapping()
{
    // Test with global_ds = 0.01
    ContourLengthMapping mapping(0.01);

    // Add some block lengths
    mapping.add_block(0.3);   // 30 segments, local_ds = 0.01
    mapping.add_block(0.5);   // 50 segments, local_ds = 0.01
    mapping.add_block(0.505); // 51 segments, local_ds = 0.505/51 ≈ 0.00990196
    mapping.add_block(0.7);   // 70 segments, local_ds = 0.01
    mapping.add_block(0.3);   // duplicate - should be deduplicated

    mapping.finalize();
    mapping.print_mapping();

    // Verify unique lengths
    if (mapping.get_n_unique_lengths() != 4)
        return fail("Expected 4 unique lengths, got " + std::to_string(mapping.get_n_unique_lengths()));

    // Verify index lookups
    int idx_03 = mapping.get_length_index(0.3);
    int idx_05 = mapping.get_length_index(0.5);
    int idx_0505 = mapping.get_length_index(0.505);
    int idx_07 = mapping.get_length_index(0.7);

    std::cout << "Index for 0.3: " << idx_03 << std::endl;
    std::cout << "Index for 0.5: " << idx_05 << std::endl;
    std::cout << "Index for 0.505: " << idx_0505 << std::endl;
    std::cout << "Index for 0.7: " << idx_07 << std::endl;

    // Indices should be 1, 2, 3, 4 (sorted order)
    if (idx_03 != 1 || idx_05 != 2 || idx_0505 != 3 || idx_07 != 4)
        return fail("Incorrect length indices");

    // Verify n_segment values
    if (mapping.get_n_segment(0.3) != 30)
        return fail("n_segment(0.3) should be 30, got " + std::to_string(mapping.get_n_segment(0.3)));
    if (mapping.get_n_segment(0.505) != 51)
        return fail("n_segment(0.505) should be 51, got " + std::to_string(mapping.get_n_segment(0.505)));

    // Verify local_ds values
    double local_ds_03 = mapping.get_ds_from_index(mapping.get_ds_index(0.3));
    double local_ds_0505 = mapping.get_ds_from_index(mapping.get_ds_index(0.505));

    std::cout << "Local ds for 0.3: " << local_ds_03 << std::endl;
    std::cout << "Local ds for 0.505: " << local_ds_0505 << std::endl;

    if (!(std::abs(local_ds_03 - 0.01) <= 1e-10))
        return fail("local_ds(0.3) should be 0.01");
    if (!(std::abs(local_ds_0505 - 0.505 / 51.0) <= 1e-10))
        return fail("local_ds(0.505) should be 0.505/51");

    // Verify ds index (0.3, 0.5, 0.7 all have local_ds = 0.01)
    int ds_idx_03 = mapping.get_ds_index(0.3);
    int ds_idx_05 = mapping.get_ds_index(0.5);
    int ds_idx_0505 = mapping.get_ds_index(0.505);
    int ds_idx_07 = mapping.get_ds_index(0.7);

    std::cout << "DS index for 0.3: " << ds_idx_03 << std::endl;
    std::cout << "DS index for 0.5: " << ds_idx_05 << std::endl;
    std::cout << "DS index for 0.505: " << ds_idx_0505 << std::endl;
    std::cout << "DS index for 0.7: " << ds_idx_07 << std::endl;

    // 0.3, 0.5, 0.7 should have the same ds_index (0.01)
    // 0.505 should have a different ds_index
    if (ds_idx_03 != ds_idx_05 || ds_idx_03 != ds_idx_07)
        return fail("0.3, 0.5, 0.7 should have same ds_index");
    if (ds_idx_0505 == ds_idx_03)
        return fail("0.505 should have different ds_index from 0.3");
    return true;
}

/// The mapping that Molecules builds from the blocks of its polymers
bool check_molecules_mapping()
{
    // Use continuous model since we test non-integer block lengths (0.505)
    Molecules molecules("continuous", 0.01, bond_lengths);

    // Add a polymer with various block lengths
    std::vector<BlockInput> blocks1 = {
        {"A", 0.25, 0, 1},  // 25 segments
        {"B", 0.505, 1, 2}  // 51 segments (non-integer ratio with ds, allowed for continuous)
    };
    molecules.add_polymer(0.5, blocks1);

    // Add another polymer
    std::vector<BlockInput> blocks2 = {
        {"A", 0.25, 0, 1},  // same as above - should deduplicate
        {"B", 0.75, 1, 2}   // 75 segments
    };
    molecules.add_polymer(0.5, blocks2);

    // Finalize the mapping
    molecules.finalize_contour_length_mapping();

    const auto& mol_mapping = molecules.get_contour_length_mapping();
    mol_mapping.print_mapping();

    // Should have 3 unique lengths: 0.25, 0.505, 0.75
    if (mol_mapping.get_n_unique_lengths() != 3)
        return fail("Expected 3 unique lengths, got " + std::to_string(mol_mapping.get_n_unique_lengths()));
    return true;
}

/// Lengths that differ by round-off are one length
bool check_machine_precision()
{
    ContourLengthMapping mapping3(0.01);

    // Add values that differ only by machine precision
    mapping3.add_block(0.5);
    mapping3.add_block(0.5 + 1e-14);  // should be treated as same
    mapping3.add_block(0.5 - 1e-14);  // should be treated as same
    mapping3.add_block(0.5 + 1e-8);   // should be different

    mapping3.finalize();
    mapping3.print_mapping();

    // Should have 2 unique lengths (0.5 and 0.5+1e-8)
    if (mapping3.get_n_unique_lengths() != 2)
        return fail("Expected 2 unique lengths (machine precision test), got "
                    + std::to_string(mapping3.get_n_unique_lengths()));
    return true;
}

/// The local ds of each ds index, and each length index found again from (n_segment, ds index)
bool check_index_lookups()
{
    ContourLengthMapping mapping4(0.01);
    mapping4.add_block(0.3);
    mapping4.add_block(0.505);
    mapping4.add_block(0.7);
    mapping4.finalize();

    // Indices 1, 2, 3 in sorted order
    if (mapping4.get_length_index(0.3) != 1 || mapping4.get_length_index(0.505) != 2 ||
        mapping4.get_length_index(0.7) != 3)
        return fail("Incorrect length indices");

    // Reverse mapping for ds: 0.01 (0.3 and 0.7) and 0.505/51
    int n_unique_ds = mapping4.get_n_unique_ds();
    std::cout << "Number of unique ds values: " << n_unique_ds << std::endl;

    for (int i = 0; i < n_unique_ds; ++i)
    {
        double ds_val = mapping4.get_ds_from_index(i);
        std::cout << "DS Index " << i << " -> ds: " << ds_val << std::endl;
    }
    if (n_unique_ds != 2)
        return fail("Expected 2 unique ds values, got " + std::to_string(n_unique_ds));

    // Round trip: length -> (n_segment, ds index) -> length index
    for (double length : {0.3, 0.505, 0.7})
        if (mapping4.get_length_index_from_n_segment(mapping4.get_n_segment(length), mapping4.get_ds_index(length))
            != mapping4.get_length_index(length))
            return fail("Round-trip failed for length " + std::to_string(length));

    std::cout << "Round-trip verification passed!" << std::endl;
    return true;
}

/// Lengths near a half-integer number of steps round alike on both sides
bool check_half_integer_rounding()
{
    // Test case: contour_length/ds close to 3.5 (mentioned in issue)
    // Values like 3.4999999999 and 3.5000000001 should both round to 4
    double ds5 = 1.0;
    ContourLengthMapping mapping5(ds5);

    // Add values that are very close to 3.5*ds but on different sides
    double len_below = 3.5 * ds5 - 1e-12;  // Just below 3.5
    double len_above = 3.5 * ds5 + 1e-12;  // Just above 3.5
    double len_exact = 3.5 * ds5;          // Exactly 3.5

    mapping5.add_block(len_below);
    mapping5.add_block(len_above);
    mapping5.add_block(len_exact);

    mapping5.finalize();
    mapping5.print_mapping();

    // All three should have the same n_segment (4, since 3.5 rounds away from zero)
    int n_seg_below = mapping5.get_n_segment(len_below);
    int n_seg_above = mapping5.get_n_segment(len_above);
    int n_seg_exact = mapping5.get_n_segment(len_exact);

    std::cout << "n_segment for len_below (3.5-1e-12): " << n_seg_below << std::endl;
    std::cout << "n_segment for len_above (3.5+1e-12): " << n_seg_above << std::endl;
    std::cout << "n_segment for len_exact (3.5): " << n_seg_exact << std::endl;

    // Verify all are 4 (std::lround(3.5) = 4)
    if (n_seg_below != 4 || n_seg_above != 4 || n_seg_exact != 4)
        return fail("Values near 3.5 should all round to 4");

    // Test another boundary: values near 2.5*ds
    ContourLengthMapping mapping5b(ds5);
    double len_2_5_below = 2.5 * ds5 - 1e-12;
    double len_2_5_above = 2.5 * ds5 + 1e-12;
    mapping5b.add_block(len_2_5_below);
    mapping5b.add_block(len_2_5_above);
    mapping5b.finalize();

    int n_seg_2_5_below = mapping5b.get_n_segment(len_2_5_below);
    int n_seg_2_5_above = mapping5b.get_n_segment(len_2_5_above);

    std::cout << "n_segment for 2.5-1e-12: " << n_seg_2_5_below << std::endl;
    std::cout << "n_segment for 2.5+1e-12: " << n_seg_2_5_above << std::endl;

    if (n_seg_2_5_below != 3 || n_seg_2_5_above != 3)
        return fail("Values near 2.5 should all round to 3");
    return true;
}

/// Lengths of different segment counts or local ds stay apart; a short positive length is not zero
bool check_distinct_lengths()
{
    // Distinct segment counts must never share a length, even with an
    // explicitly loose comparison tolerance near a rounding boundary.
    ContourLengthMapping boundary(0.01, 1e-10);
    const double lower = 0.495 - 2e-11;
    const double upper = 0.495 + 2e-11;
    boundary.add_block(lower);
    boundary.add_block(upper);
    boundary.finalize();
    if (boundary.get_n_unique_lengths() != 2 || boundary.get_n_segment(lower) != 49 ||
        boundary.get_n_segment(upper) != 50 ||
        boundary.get_length_index(lower) == boundary.get_length_index(upper))
        return fail("lengths with different segment counts were merged");

    // A fixed absolute tolerance for local ds loses significant contour
    // length after hundreds of steps, and breaks inverse aggregation lookups.
    ContourLengthMapping precise(0.001);
    for (double length : {0.1, 0.4, 0.50000004})
        precise.add_block(length);
    precise.finalize();
    if (precise.get_n_unique_ds() != 2 || precise.get_ds_index(0.4) == precise.get_ds_index(0.50000004))
        return fail("distinct local contour steps were merged");
    for (double length : {0.1, 0.4, 0.50000004})
        if (precise.get_length_index_from_n_segment(precise.get_n_segment(length), precise.get_ds_index(length))
            != precise.get_length_index(length))
            return fail("segment-count / ds-index inverse lookup");

    // Positive short blocks are not the zero-length aggregation sentinel.
    ContourLengthMapping short_block(1e-11);
    short_block.add_block(5e-11);
    short_block.finalize();
    if (short_block.get_length_index(5e-11) == 0 || short_block.get_n_segment(5e-11) != 5)
        return fail("a positive short contour was treated as zero");
    return true;
}

/// Discrete chains: the scalar canonicalizer and molecule construction accept and reject the same lengths
bool check_discrete_lengths()
{
    Molecules discrete("discrete", 0.01, bond_lengths);
    discrete.add_polymer(1.0, {{"A", 0.5, 0, 1}, {"B", 0.5 + 4e-10, 1, 2}});
    discrete.finalize_contour_length_mapping();
    if (discrete.get_contour_length_mapping().get_n_unique_ds() != 1 ||
        discrete.get_polymer(0).get_alpha() != 1.0)
        return fail("accepted discrete roundoff changed the global contour step");

    // The shared scalar canonicalizer and molecule construction must choose
    // exactly the same lengths, including tiny steps and half-integer snaps.
    double accepted_upper = 0.5 + 1e-9;
    while (accepted_upper - 0.5 > 1e-9)
        accepted_upper = std::nextafter(accepted_upper, 0.5);
    double accepted_lower = 0.5 - 1e-9;
    while (0.5 - accepted_lower > 1e-9)
        accepted_lower = std::nextafter(accepted_lower, 0.5);
    const std::vector<std::array<double, 3>> accepted_discrete = {
        {0.5 + 4e-10, 0.01, 0.5},
        {0.5 - 4e-10, 0.01, 0.5},
        {accepted_upper, 0.01, 0.5},
        {accepted_lower, 0.01, 0.5},
        {1.0009e-6, 1e-6, 1e-6},
        {0.25e-9, 1e-9, 1e-9},
        {1.5e-9 - 5e-19, 1e-9, 2e-9},
        {1.5e-9 - 2e-18, 1e-9, 1e-9},
    };
    for (const auto& [length, step, expected] : accepted_discrete)
    {
        const double canonical = ContourLengthMapping::discrete_contour_length(length, step);
        Molecules molecule("discrete", step, bond_lengths);
        molecule.add_polymer(1.0, {{"A", length, 0, 1}});
        if (canonical != expected || molecule.get_polymer(0).get_alpha() != canonical ||
            molecule.get_polymer(0).get_blocks()[0].contour_length != canonical)
            return fail("discrete canonicalizer differs from molecule length");
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<std::array<double, 2>> rejected_discrete = {
        {std::nextafter(accepted_upper, inf), 0.01},
        {std::nextafter(accepted_lower, 0.0), 0.01},
        {0.495, 0.01}, {0.0, 0.01}, {-0.1, 0.01}, {nan, 0.01}, {inf, 0.01},
        {0.5, 0.0}, {0.5, -0.01}, {0.5, nan}, {0.5, inf}, {1.0, 1e-12},
    };
    for (const auto& [length, step] : rejected_discrete)
    {
        bool scalar_rejected = false, molecule_rejected = false;
        try { ContourLengthMapping::discrete_contour_length(length, step); }
        catch (const std::exception&) { scalar_rejected = true; }
        try
        {
            Molecules molecule("discrete", step, bond_lengths);
            molecule.add_polymer(1.0, {{"A", length, 0, 1}});
        }
        catch (const std::exception&) { molecule_rejected = true; }
        if (!scalar_rejected || !molecule_rejected)
            return fail("invalid discrete length accepted by one construction path");
    }
    return true;
}

} // namespace

int main()
{
    const std::vector<std::pair<std::string, bool (*)()>> topics = {
        {"Basic ContourLengthMapping", check_basic_mapping},
        {"Integration with Molecules", check_molecules_mapping},
        {"Machine precision comparison", check_machine_precision},
        {"Index lookups", check_index_lookups},
        {"Half-integer boundary robustness", check_half_integer_rounding},
        {"Distinct lengths and local steps", check_distinct_lengths},
        {"Discrete contour lengths", check_discrete_lengths},
    };
    int n_failed = 0;
    for (const auto& [name, check] : topics)
    {
        std::cout << "=== " << name << " ===" << std::endl;
        bool passed = false;
        try
        {
            passed = check();
        }
        catch (std::exception& exc)
        {
            std::cout << exc.what() << std::endl;
        }
        std::cout << name << (passed ? " PASSED" : " FAILED") << std::endl << std::endl;
        if (!passed)
            ++n_failed;
    }
    std::cout << topics.size() << " topics, " << n_failed << " failed" << std::endl;
    return n_failed == 0 ? 0 : -1;
}
