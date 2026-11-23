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
 * @file TestScft.cpp
 * @brief Twenty SCFT iterations of an AB diblock (continuous chains, rqm4) in a 1D, 2D
 *        and 3D box on every platform.
 *
 * The fields start from short-wavelength cosines. The 1D and 2D runs relax toward the
 * homogeneous state: every iteration conserves the mass and the composition and passes
 * check_total_partition(), the free energy approaches its value zero in this
 * zero-mean-field convention, and the residual falls below 0.1. The 3D run must reach
 * its stored residual after twenty iterations.
 *
 * Without an argument the test runs the 1D and 2D boxes; `--3d` runs the 3D box, a
 * CTest entry of its own (TestScft3D, numerical).
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "TestAbScft.h"
#include "TestChecks.h"
#include "TestCudaPart.h"

namespace {

/// A box and the checks of its run
struct ScftCase
{
    std::string name;
    double f;
    double chi_n;
    std::vector<int> nx;
    std::vector<double> lx;
    double ds;
    /// CUDA runs four streams whatever OMP_NUM_THREADS is: the two propagators of the
    /// diblock run at once, and the transforms of 263 points need a cuFFT work area,
    /// which each stream must have for itself
    bool cuda_four_streams;
    /// The mass, check_total_partition() and the mean concentrations at every iteration
    bool check_each_iteration;
    /// |free energy| at the end, the homogeneous state; none for 3D
    std::optional<double> energy_tolerance;
    /// The residual at the end: within 1e-4 of the stored value, or <= 0.1 without one
    std::optional<double> error_level_ref;
};

const std::vector<ScftCase> cases_1d_2d =
{
    {"1D", 0.3, 25.0, {263}, {4.0}, 1.0/50, true, true, 5e-6, std::nullopt},
    {"2D", 2.0/7.0, 25.0, {49, 63}, {4.0,3.0}, 1.0/49, false, true, 1e-7, std::nullopt},
};
const std::vector<ScftCase> cases_3d =
{
    {"3D", 2.0/7.0, 20.0, {31,49,23}, {4.0,3.0,2.0}, 1.0/49, false, false, std::nullopt, 0.011080773},
};

/// Twenty iterations of a case on one platform
void run_case(const ScftCase& c, const std::string& platform, Checks& checks)
{
    // Math constants
    const double PI = std::numbers::pi;
    // Chrono timer
    std::chrono::system_clock::time_point chrono_start, chrono_end;
    std::chrono::duration<double> time_duration;

    std::streamsize default_precision = std::cout.precision();
    int idx;
    double sum;

    int max_scft_iter = 20;
    double tolerance = 1e-9;
    double f = c.f;
    double chi_n = c.chi_n;
    std::string chain_model = "Continuous";  // with "rqm4" below; discrete chains take "ps" or "rs"

    int am_max_hist = 20;
    double am_start_error = 8e-1;
    double am_mix_min = 0.1;
    double am_mix_init = 0.1;

    std::vector<BlockInput> blocks =
    {
        {"A",    f, 0, 1},
        {"B",1.0-f, 1, 2},
    };

    bool reduce_memory=false;

    checks.begin_run(c.name + ", " + platform);
    std::optional<ScopedStreams> streams;
    if (c.cuda_four_streams && platform == "cuda")
        streams.emplace("4");
    std::unique_ptr<AbstractFactory<double>> factory = PlatformSelector::create_factory<double>(platform, reduce_memory);
    factory->display_info();

    // Create instances and assign to the variables of base classes for the dynamic binding
    std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box(c.nx, c.lx, {});
    std::unique_ptr<Molecules> molecules = factory->create_molecules_information(chain_model, c.ds, {{"A",1.0}, {"B",1.0}});
    molecules->add_polymer(1.0, blocks, {});
    auto propagator_computation_optimizer = std::make_unique<PropagatorComputationOptimizer>(molecules.get(), false);
    std::unique_ptr<PropagatorComputation<double>> solver = factory->create_propagator_computation(cb.get(), molecules.get(), propagator_computation_optimizer.get(), "rqm4");

    const int M = cb->get_total_grid();
    int am_n_var = 2*M;  // A and B
    std::unique_ptr<AndersonMixing<double>> am = factory->create_anderson_mixing(am_n_var,
                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

    // -------------- Print simulation parameters ------------
    std::cout << std::setprecision(default_precision);
    std::cout << std::boolalpha;
    std::cout<< "---------- Simulation Parameters ----------" << std::endl;
    std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
    std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
    std::cout << "chi_n, f: " << chi_n << " " << f << " "  << std::endl;
    std::cout << "Nx, Lx, dx:";
    for (int d = 0; d < cb->get_dim(); d++)
        std::cout << " " << cb->get_nx(d) << ", " << cb->get_lx(d) << ", " << cb->get_dx(d) << ";";
    std::cout << std::endl;
    sum = 0.0;
    for(int i=0; i<M; i++)
        sum += cb->get_dv(i);
    std::cout << "volume, sum(dv):  " << cb->get_volume() << " " << sum << std::endl;

    propagator_computation_optimizer->display_blocks();
    propagator_computation_optimizer->display_propagators();

    //-------------- Allocate array ------------
    std::vector<double> w(M*2), phi_a(M), phi_b(M);

    //-------------- Setup fields ------------
    // cos(2 pi i/4.68) cos(2 pi j/3.48) cos(2 pi k/2.74)/10 on the grid (i, j, k), the axes in
    // front of a 1D or 2D box being one point (i = 0, then j = 0)
    std::cout<< "w_a and w_b are initialized to a given test fields." << std::endl;
    std::vector<int> nx_3d(3 - cb->get_dim(), 1);
    for (int d = 0; d < cb->get_dim(); d++)
        nx_3d.push_back(cb->get_nx(d));
    for(int i=0; i<nx_3d[0]; i++)
        for(int j=0; j<nx_3d[1]; j++)
            for(int k=0; k<nx_3d[2]; k++)
            {
                idx = i*nx_3d[1]*nx_3d[2] + j*nx_3d[2] + k;
                phi_a[idx]= cos(2.0*PI*i/4.68)*cos(2.0*PI*j/3.48)*cos(2.0*PI*k/2.74)*0.1;
            }

    for(int i=0; i<M; i++)
    {
        phi_b[i] = 1.0 - phi_a[i];
        w[i]   = chi_n*phi_b[i];
        w[i+M] = chi_n*phi_a[i];
    }

    // Keep the level of field value
    cb->zero_mean(&w[0]);
    cb->zero_mean(&w[M]);

    //------------------ run ----------------------
    std::cout<< "---------- Run ----------" << std::endl;
    std::cout<< "iteration, mass error, total partitions, total energy, error level" << std::endl;
    chrono_start = std::chrono::system_clock::now();
    AbScft scft(cb.get(), solver.get(), molecules.get(), chi_n);
    for(int iter=0; iter<max_scft_iter; iter++)
    {
        scft.step(w.data());

        // Print iteration # and error levels and check the mass conservation
        sum = scft.mass_error();
        if (c.check_each_iteration)
        {
            const bool ok = std::abs(sum) <= 1e-10 && solver->check_total_partition()
                         && std::abs(cb->mean(scft.phi_a.data()) - f) <= 1e-10
                         && std::abs(cb->mean(scft.phi_b.data()) - (1.0-f)) <= 1e-10;
            if (!checks.that("Iteration " + std::to_string(iter) + ": mass, check_total_partition and composition", ok))
                break;
        }
        std::cout<< std::setw(8) << iter;
        std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
        std::cout<< "\t[" << std::setprecision(7) << std::scientific << solver->get_total_partition(0);
        for(int p=1; p<molecules->get_n_polymer_types(); p++)
            std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << solver->get_total_partition(p);
        std::cout<< "]";
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << scft.energy;
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << scft.error_level << std::endl;

        // Conditions to end the iteration
        if(scft.error_level < tolerance) break;
        scft.mix(am.get(), w.data());
    }

    // Estimate execution time
    chrono_end = std::chrono::system_clock::now();
    time_duration = chrono_end - chrono_start;
    std::cout<< "total time, time per step: ";
    std::cout<< time_duration.count() << " " << time_duration.count()/max_scft_iter << std::endl;
    std::cout << std::setprecision(default_precision) << std::defaultfloat;

    // These short-wavelength initial fields relax toward the homogeneous state, whose free
    // energy in this zero-mean-field convention is zero. Check the analytic energy as well
    // as the residual of this short run.
    if (c.energy_tolerance)
        checks.close("Free energy", std::abs(scft.energy), *c.energy_tolerance);

    if (c.error_level_ref)
        checks.close("Residual", std::abs(scft.error_level-*c.error_level_ref), 1e-4);
    else    // SCFT converged reasonably: the residual is small after 20 iterations
        checks.close("Residual", scft.error_level, 0.1);
    checks.end_run();
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const bool run_3d = (argc > 1 && std::string(argv[1]) == "--3d");
        const std::vector<ScftCase>& cases = run_3d ? cases_3d : cases_1d_2d;
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();

        Checks checks;
        for (const ScftCase& c : cases)
            for (const std::string& platform : avail_platforms)
                run_case(c, platform, checks);

        // Each case on every platform
        const int expected_runs = static_cast<int>(cases.size() * avail_platforms.size());
        if (!checks.passed(expected_runs))
            return -1;
        return cuda_part_missing() ? 1 : 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
