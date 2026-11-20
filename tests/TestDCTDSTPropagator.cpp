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
 * @file TestDCTDSTPropagator.cpp
 * @brief One pseudo-spectral step with reflecting (DCT-2) and absorbing (DST-2) boundaries.
 *
 * The propagator solver must give the analytical DCT-2 and DST-2 diffusion of a
 * Gaussian:
 * - 1D domain with L=4.0, N=12 grid points (cell-centred)
 * - Reflecting boundary conditions (Neumann BC), then absorbing ones (Dirichlet
 *   BC, u=0 at the boundaries)
 * - Zero potential field (w=0)
 * - Single homopolymer block of one step (ds=0.1, b=1.0)
 * - Gaussian initial condition (sigma=0.5) centered at L/2
 *
 * Tests are run on all available platforms with both standard and
 * memory-saving modes.
 *
 * The pseudo-spectral method for the diffusion equation:
 *     dq/ds = (b^2/6) nabla^2 q - w*q
 *
 * With w=0 the solution is:
 *     q_hat(k, s+ds) = q_hat(k, s) * exp(-b^2/6 * k^2 * ds)
 *
 * where q_hat are the DCT-2 (reflecting) or DST-2 (absorbing) coefficients,
 * k = n*pi/L for the DCT-2 and k = (n+1)*pi/L for the DST-2. The expected values
 * are this step computed independently of the library.
 *
 * Reflecting walls conserve the mass; absorbing walls do not: the field "leaks
 * out" at the boundaries where u=0 is enforced.
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
#include <string>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "TestCudaPart.h"
#include "TestNanSafe.h"

// Test parameters
const int N = 12;
const double L = 4.0;
const double dx = L / N;
const double ds = 0.1;
const double sigma = 0.5;
const double b = 1.0;

// Expected values (full precision)
const std::vector<double> expected_q_init = {
    0.0012038599948282,  0.01110899653824231, 0.06572852861653045,
    0.2493522087772961,  0.6065306597126334,  0.9459594689067654,
    0.9459594689067655,  0.6065306597126334,  0.24935220877729644,
    0.06572852861653053, 0.01110899653824231, 0.00120385999482821
};

/// The walls of one case and the step they give
struct WallCase
{
    std::string name;                   ///< "DCT-2" or "DST-2"
    std::string boundary;               ///< "reflecting" or "absorbing"
    std::vector<double> expected_q_out; ///< q after one step
};

const std::vector<WallCase> wall_cases = {
    {"DCT-2", "reflecting", {
        0.00265976695175955, 0.01771721116200349, 0.0850483724745189,
        0.27580183068075836, 0.6042556834496476,  0.894400857827608,
        0.894400857827608,   0.6042556834496476,  0.2758018306807586,
        0.085048372474519,   0.01771721116200349, 0.00265976695175955
    }},
    // The mass decreases slightly through the absorbing walls
    {"DST-2", "absorbing", {
        0.00229854560729656, 0.01772114433056498, 0.08504620746747918,
        0.27580273305699987, 0.6042552886190555,  0.8944009714708175,
        0.8944009714708175,  0.6042552886190556,  0.2758027330570002,
        0.08504620746747932, 0.01772114433056499, 0.00229854560729657
    }},
};

/**
 * Run the step of one wall case for a specific platform and memory mode.
 *
 * @param wall Reflecting (DCT-2) or absorbing (DST-2) walls
 * @param platform Platform name ("cuda", "cpu-fftw" or "cpu-mkl")
 * @param reduce_memory Enable memory-saving mode
 * @return 0 on success, -1 on failure
 */
int run_tests(const WallCase& wall, const std::string& platform, bool reduce_memory)
{
    std::cout << std::fixed << std::setprecision(17);

    std::cout << "\n============================================================" << std::endl;
    std::cout << wall.name << " (" << wall.boundary << " BC)" << std::endl;
    std::cout << "Platform: " << platform << std::endl;
    std::cout << "Memory saving mode: " << (reduce_memory ? "ON" : "OFF") << std::endl;
    std::cout << "============================================================" << std::endl;

    // Create factory
    std::unique_ptr<AbstractFactory<double>> factory = PlatformSelector::create_factory<double>(platform, reduce_memory);
    factory->display_info();

    // Staggered grid (half-sample symmetric)
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
    {
        x[i] = (i + 0.5) * dx;
    }

    // Gaussian initial condition centered at L/2
    std::vector<double> q_init(N);
    for (int i = 0; i < N; ++i)
    {
        q_init[i] = std::exp(-std::pow(x[i] - L/2, 2) / (2 * sigma * sigma));
    }

    std::cout << "\nParameters:" << std::endl;
    std::cout << "  N = " << N << std::endl;
    std::cout << "  L = " << L << std::endl;
    std::cout << "  dx = " << dx << std::endl;
    std::cout << "  ds = " << ds << std::endl;
    std::cout << "  sigma = " << sigma << std::endl;
    std::cout << "  b = " << b << std::endl;

    std::cout << "\nInitial condition (Gaussian):" << std::endl;
    std::cout << "  q_init = [";
    for (int i = 0; i < N; ++i)
    {
        std::cout << q_init[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify the initial condition against the expected values
    double init_error = 0.0;
    for (int i = 0; i < N; ++i)
    {
        init_error = max_or_nan(init_error, std::abs(q_init[i] - expected_q_init[i]));
    }
    std::cout << "  Init error: " << std::scientific << init_error << std::fixed << std::endl;

    // Zero potential field
    std::vector<double> w_zero(N, 0.0);

    // Create molecules for homopolymer
    std::map<std::string, double> bond_lengths = {{"A", b}};
    std::unique_ptr<Molecules> molecules = factory->create_molecules_information("Continuous", ds, bond_lengths);

    // Single A block with 1 step (ds length)
    // Use custom initial condition "G" at vertex 0
    std::vector<BlockInput> blocks = {
        {"A", ds, 0, 1}   // A block: 1 step
    };
    std::map<int, std::string> chain_end_to_q_init = {{0, "G"}};  // vertex 0 -> "G"
    molecules->add_polymer(1.0, blocks, chain_end_to_q_init);

    std::cout << "Polymer: Homopolymer (A block)" << std::endl;
    std::cout << "  A block length: " << ds << " (1 step)" << std::endl;
    std::cout << "  Initial condition: Gaussian at vertex 0" << std::endl;

    // Create computation box with the walls of the case
    std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box(
        {N}, {L}, {wall.boundary, wall.boundary});

    // Create propagator optimizer and solver
    auto optimizer = std::make_unique<PropagatorComputationOptimizer>(molecules.get(), false);
    std::unique_ptr<PropagatorComputation<double>> solver = factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4");

    // Compute propagators with zero field and custom initial condition
    solver->compute_propagators({{"A", w_zero.data()}}, {{"G", q_init.data()}});

    // Get propagator after A block (step 1)
    std::vector<double> q_out(N);
    solver->get_chain_propagator(q_out.data(), 0, 0, 1, 1);

    std::cout << "\nPropagator after 1 step (A block):" << std::endl;
    std::cout << "  q_out = [";
    for (int i = 0; i < N; ++i)
    {
        std::cout << q_out[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    double sum_out = 0.0;
    for (int i = 0; i < N; ++i)
        sum_out += q_out[i];
    std::cout << "  sum(q_out) = " << sum_out << std::endl;

    // Compare with the expected values
    double max_error = 0.0;
    for (int i = 0; i < N; ++i)
    {
        max_error = max_or_nan(max_error, std::abs(q_out[i] - wall.expected_q_out[i]));
    }
    std::cout << "\nMax error against the expected values: " << std::scientific << max_error << std::fixed << std::endl;

    // Verify symmetry
    double sym_error = 0.0;
    for (int i = 0; i < N/2; ++i)
    {
        sym_error = max_or_nan(sym_error, std::abs(q_out[i] - q_out[N-1-i]));
    }
    std::cout << "Symmetry error: " << std::scientific << sym_error << std::fixed << std::endl;

    // The mass: conserved with reflecting walls, decreasing through absorbing ones
    double initial_mass = 0.0;
    for (int i = 0; i < N; ++i)
        initial_mass += q_init[i] * dx;

    double final_mass = 0.0;
    for (int i = 0; i < N; ++i)
        final_mass += q_out[i] * dx;

    std::cout << "\nMass:" << std::endl;
    std::cout << "  Initial mass: " << initial_mass << std::endl;
    std::cout << "  Final mass:   " << final_mass << std::endl;
    bool mass_ok;
    if (wall.boundary == "reflecting")
    {
        double mass_error = std::abs(final_mass - initial_mass) / initial_mass;
        std::cout << "  Relative error: " << std::scientific << mass_error << std::fixed << std::endl;
        mass_ok = mass_error < 1e-6;
    }
    else
    {
        double mass_ratio = final_mass / initial_mass;
        std::cout << "  Mass ratio: " << mass_ratio << " (expected ~0.9998)" << std::endl;
        mass_ok = (mass_ratio > 0.99) && (mass_ratio < 1.0);
    }

    // The initial condition and the q(x) = q(L-x) symmetry are exact up to round-off
    bool test_passed = mass_ok && (max_error < 1e-6)
                       && (init_error <= 1e-12) && (sym_error <= 1e-12);
    std::cout << "\nTest (Homopolymer " << wall.name << ") for " << platform;
    if (reduce_memory) std::cout << " (memory-saving)";
    std::cout << ": " << (test_passed ? "PASSED" : "FAILED") << std::endl;

    return test_passed ? 0 : -1;
}

int main()
{
    try
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "Test: DCT-2 and DST-2 Propagator with Reflecting and Absorbing Boundaries" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Testing on all available platforms with standard and" << std::endl;
        std::cout << "memory-saving modes." << std::endl;

        int result = 0;
        int test_count = 0;
        int pass_count = 0;

        // Get available platforms
        std::vector<std::string> platforms = PlatformSelector::avail_platforms();

        std::cout << "\nAvailable platforms: ";
        for (const auto& p : platforms)
            std::cout << p << " ";
        std::cout << std::endl;

        // Run each wall case on each platform with both memory modes
        for (const WallCase& wall : wall_cases)
        {
            for (const std::string& platform : platforms)
            {
                for (bool reduce_memory : {false, true})
                {
                    test_count++;
                    if (run_tests(wall, platform, reduce_memory) == 0)
                        pass_count++;
                    else
                        result = -1;
                }
            }
        }

        //=======================================================================
        // Final Summary
        //=======================================================================
        std::cout << "\n============================================================" << std::endl;
        std::cout << "FINAL TEST SUMMARY" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Tests passed: " << pass_count << "/" << test_count << std::endl;

        if (result == 0)
        {
            std::cout << "All tests PASSED!" << std::endl;
        }
        else
        {
            std::cout << "Some tests FAILED!" << std::endl;
        }
        std::cout << "============================================================" << std::endl;

        return result != 0 ? result : (cuda_part_missing() ? 1 : 0);
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
