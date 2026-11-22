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
 * @file TestAggregation.cpp
 * @brief Aggregated propagators of branched architectures against the plain computation.
 *
 * Three architectures, a branched polymer of 27 blocks, a blend of three dendritic
 * polymers and a blend of four side-chain polymers, run three box-relaxing SCFT
 * iterations from gyroid fields on a 16^3 grid. For each architecture and chain model the
 * free energy must agree, within a standard deviation of 1e-7, on every platform in full
 * and reduced memory, without and with aggregated propagators.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
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
#include "TestChecks.h"
#include "TestCudaPart.h"

namespace {

/// A polymer of an architecture
struct PolymerInput
{
    double volume_fraction;
    std::vector<BlockInput> blocks;
};

/// An architecture: its segment lengths and polymers
struct Architecture
{
    std::string name;
    std::map<std::string, double> bond_lengths;
    std::vector<PolymerInput> polymers;
};

const double f = 0.2;

const std::vector<Architecture> architectures =
{
    {"Branched (27 blocks)", {{"A",1.0}, {"B",1.0}},
    {
        {1.0,
        {
            {"A",  f, 0, 1},
            {"A",  f, 1, 2},
            {"A",  f, 2, 3},
            {"A",  f, 3, 4},
            {"A",  f, 4, 5},
            {"A",  f, 5, 6},
            {"A",  f, 6, 7},
            {"A",  f, 7, 8},
            {"B", 0.5, 1, 9},
            {"B", 0.4, 2,10},
            {"B", 0.8, 3,11},
            {"B", 0.3, 4,12},
            {"B", 0.4, 5,13},
            {"B", 0.1, 6,14},
            {"B", 0.5, 7,15},
            {"B", 0.8, 9,16},
            {"B", 0.8,10,17},
            {"B", 0.8,11,18},
            {"A",   f, 9,19},
            {"A",   f,10,20},
            {"A",   f,11,21},
            {"B", 0.8, 9,22},
            {"B", 0.8,10,23},
            {"B", 0.1,11,24},
            {"B", 0.3, 4,25},
            {"B", 0.1, 6,27},
            {"B", 0.5, 7,28},
        }},
    }},
    {"Dendritic blend", {{"A",1.0}, {"B",0.7}},
    {
        {0.5,
        {
            {"A",0.3,0,1},
            {"A",0.3,0,2},
            {"A",0.3,0,3},
            {"B",0.3,1,4},
            {"B",0.3,1,5},
            {"B",0.3,1,6},
            {"B",0.3,2,7},
            {"B",0.3,2,8},
            {"B",0.3,2,9},
            {"B",0.3,3,10},
            {"B",0.3,3,11},
            {"B",0.3,3,12},
            {"A",0.2,4,13},
            {"A",0.2,4,14},
            {"A",0.2,4,15},
            {"A",0.2,5,16},
            {"A",0.2,5,17},
            {"A",0.2,5,18},
            {"A",0.2,6,19},
            {"A",0.2,6,20},
            {"A",0.2,6,21},
            {"A",0.2,7,22},
            {"A",0.2,7,23},
            {"A",0.2,7,24},
            {"A",0.2,8,25},
            {"A",0.2,8,26},
            {"A",0.2,8,27},
            {"A",0.2,9,28},
            {"A",0.2,9,29},
            {"A",0.2,9,30},
            {"A",0.2,10,31},
            {"A",0.2,10,32},
            {"A",0.2,10,33},
            {"A",0.2,11,34},
            {"A",0.2,11,35},
            {"A",0.2,11,36},
            {"A",0.2,12,37},
            {"A",0.2,12,38},
            {"A",0.2,12,39},
        }},
        {0.3,
        {
            {"A",0.3,0,1},
            {"B",0.3,1,4},
            {"B",0.3,1,5},
            {"B",0.3,1,6},
            {"B",0.2,0,10},
            {"B",0.3,0,11},
            {"B",0.4,0,12},
            {"A",0.2,4,13},
            {"A",0.2,4,14},
            {"A",0.2,4,15},
            {"A",0.2,5,16},
            {"A",0.2,5,17},
            {"A",0.2,5,18},
            {"A",0.2,6,19},
            {"A",0.2,6,20},
            {"A",0.2,6,21},
            {"A",0.2,0,22},
            {"A",0.3,0,23},
            {"A",0.4,0,24},
            {"A",0.2,10,31},
            {"A",0.2,10,32},
            {"A",0.2,10,33},
            {"A",0.2,11,34},
            {"A",0.2,11,35},
            {"A",0.2,11,36},
            {"A",0.2,12,37},
            {"A",0.2,12,38},
            {"A",0.2,12,39},
        }},
        {0.2,
        {
            {"B",0.3,40,41},
            {"B",0.3,41,42},
            {"B",0.3,42,43},
            {"A",0.3,40,1},
            {"A",0.3,41,2},
            {"A",0.3,42,3},
            {"B",0.3,1,4},
            {"B",0.3,1,5},
            {"B",0.3,1,6},
            {"B",0.3,2,7},
            {"B",0.3,2,8},
            {"B",0.3,2,9},
            {"B",0.3,3,10},
            {"B",0.3,3,11},
            {"B",0.3,3,12},
            {"A",0.2,4,13},
            {"A",0.2,4,14},
            {"A",0.2,4,15},
            {"A",0.2,5,16},
            {"A",0.2,5,17},
            {"A",0.2,5,18},
            {"A",0.2,6,19},
            {"A",0.2,6,20},
            {"A",0.2,6,21},
            {"A",0.2,7,22},
            {"A",0.2,7,23},
            {"A",0.2,7,24},
            {"A",0.2,8,25},
            {"A",0.2,8,26},
            {"A",0.2,8,27},
            {"A",0.2,9,28},
            {"A",0.2,9,29},
            {"A",0.2,9,30},
            {"A",0.2,10,31},
            {"A",0.2,10,32},
            {"A",0.2,10,33},
            {"A",0.2,11,34},
            {"A",0.2,11,35},
            {"A",0.2,11,36},
            {"A",0.2,12,37},
            {"A",0.2,12,38},
            {"A",0.2,12,39},
        }},
    }},
    {"Side chains", {{"A",1.0}, {"B",0.7}},
    {
        // Depth 1
        {0.2,
        {
            {"A",0.3,1,2},
            {"A",0.3,2,3},

            {"B",0.3,1,4},
            {"B",0.3,2,5},
            {"B",0.3,3,6},
        }},
        // Depth 2
        {0.2,
        {
            {"A",0.3,1,2},
            {"A",0.3,2,3},

            {"B",0.3,1,4},
            {"B",0.3,2,5},
            {"B",0.3,3,6},

            {"A",0.3,4,7},
            {"A",0.3,5,8},
            {"A",0.3,6,9},

            {"A",0.3,4,10},
            {"A",0.3,5,11},
            {"A",0.3,6,12},
        }},
        // Depth 3
        {0.2,
        {
            {"A",0.3,1,2},
            {"A",0.3,2,3},

            {"B",0.3,1,4},
            {"B",0.3,2,5},
            {"B",0.3,3,6},

            // Repeat 1
            {"A",0.3,4,7},
            {"A",0.3,5,8},
            {"A",0.3,6,9},

            {"B",0.3,7,10},
            {"B",0.3,8,11},
            {"B",0.3,9,12},

            // Repeat 2
            {"A",0.3,4,13},
            {"A",0.3,5,14},
            {"A",0.3,6,15},

            {"B",0.3,13,16},
            {"B",0.3,14,17},
            {"B",0.3,15,18},
        }},
        // Mixed Depth
        {0.2,
        {
            {"A",0.3,1,2},
            {"A",0.3,2,3},
            {"A",0.3,3,23},

            // Depth 1
            {"B",0.3,1,4},

            // Depth 2
            {"A",0.3,2,20},
            {"B",0.3,20,5},

            // Depth 3
            {"B",0.3,3,21},
            {"A",0.3,21,22},
            {"B",0.3,22,6},

            // Repeat 1
            {"A",0.3,4,7},
            {"A",0.3,5,8},
            {"A",0.3,6,9},

            {"B",0.3,7,10},
            {"B",0.3,8,11},
            {"B",0.3,9,12},

            // Repeat 2
            {"A",0.3,4,13},
            {"A",0.3,5,14},
            {"A",0.3,6,15},

            {"B",0.3,13,16},
            {"B",0.3,14,17},
            {"B",0.3,15,18},
        }},
    }},
};

/// Three SCFT iterations of an architecture on one platform; the free energy at the end
double run_scft(const Architecture& architecture, const std::string& chain_model, const std::string& platform,
                bool aggregate_propagator_computation, bool reduce_memory, Checks& checks)
{
    // Math constants
    const double PI = std::numbers::pi;

    double energy_total;
    // error_level = variable to check convergence of the iteration
    double error_level, old_error_level;
    std::streamsize default_precision = std::cout.precision();
    int idx;
    double sum;

    int max_scft_iter = 3;
    double tolerance = 1e-9;

    double chi_n = 15.0;
    std::vector<int> nx = {16,16,16};
    std::vector<double> lx = {2.9,2.9,2.9};
    std::vector<double> lx_backup = lx;
    double ds = 1.0/10;

    const int M = nx[0]*nx[1]*nx[2];

    int am_n_var = 2*M+nx.size();  // A and B
    int am_max_hist = 20;
    double am_start_error = 1e-2;
    double am_mix_min = 0.02;
    double am_mix_init = 0.02;

    //-------------- allocate array ------------
    std::vector<double> w(2*M+nx.size());
    std::vector<double> w_out(2*M+nx.size());
    std::vector<double> w_diff(2*M+nx.size());
    std::vector<double> xi(M);
    std::vector<double> phi_a(M);
    std::vector<double> phi_b(M);
    std::vector<double> w_plus(M);
    std::vector<double> w_minus(M);

    std::unique_ptr<AbstractFactory<double>> factory = PlatformSelector::create_factory<double>(platform, reduce_memory);

    // Create instances and assign to the variables of base classes for the dynamic binding
    std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box(nx, lx_backup, {});
    std::unique_ptr<Molecules> molecules = factory->create_molecules_information(chain_model, ds, architecture.bond_lengths);
    for (const PolymerInput& polymer : architecture.polymers)
        molecules->add_polymer(polymer.volume_fraction, polymer.blocks, {});
    auto propagator_computation_optimizer = std::make_unique<PropagatorComputationOptimizer>(molecules.get(), aggregate_propagator_computation);
    std::unique_ptr<PropagatorComputation<double>> solver = factory->create_propagator_computation(cb.get(), molecules.get(), propagator_computation_optimizer.get(), "");
    std::unique_ptr<AndersonMixing<double>> am = factory->create_anderson_mixing(am_n_var,
                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

    // -------------- Print simulation parameters ------------
    std::cout << std::setprecision(default_precision);
    std::cout << std::boolalpha;
    std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
    std::cout << "Platform: " << platform << std::endl;
    std::cout << "Using Aggregation: " << aggregate_propagator_computation << std::endl;
    std::cout << "Reduce Memory Usage: " << reduce_memory << std::endl;

    // Display branches
    #ifndef NDEBUG
    propagator_computation_optimizer->display_blocks();
    #endif
    propagator_computation_optimizer->display_propagators();

    // w_a and w_b are initialized to a gyroid
    double xx, yy, zz, c1, c2;
    for(int i=0; i<nx[0]; i++)
    {
        xx = (i+1)*2*PI/nx[0];
        for(int j=0; j<nx[1]; j++)
        {
            yy = (j+1)*2*PI/nx[1];
            for(int k=0; k<nx[2]; k++)
            {
                zz = (k+1)*2*PI/nx[2];
                c1 = sqrt(8.0/3.0)*(cos(xx)*sin(yy)*sin(2.0*zz) +
                    cos(yy)*sin(zz)*sin(2.0*xx)+cos(zz)*sin(xx)*sin(2.0*yy));
                c2 = sqrt(4.0/3.0)*(cos(2.0*xx)*cos(2.0*yy)+
                    cos(2.0*yy)*cos(2.0*zz)+cos(2.0*zz)*cos(2.0*xx));
                idx = i*nx[1]*nx[2] + j*nx[2] + k;
                w[idx] = -0.3164*c1 +0.1074*c2;
                w[idx+M] = 0.3164*c1 -0.1074*c2;
            }
        }
    }

    // Keep the level of field value
    cb->zero_mean(&w[0]);
    cb->zero_mean(&w[M]);

    // Assign large initial value for the energy and error
    energy_total = 1.0e20;
    error_level = 1.0e20;

    //------------------ Run ----------------------
    // Iteration begins here
    for(int iter=0; iter<max_scft_iter; iter++)
    {
        // For the given fields find the polymer statistics
        solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
        solver->compute_concentrations();
        solver->get_total_concentration("A", phi_a.data());
        solver->get_total_concentration("B", phi_b.data());

        // Check the total partition function
        if (iter == 0 && !checks.that("check_total_partition", solver->check_total_partition()))
            break;

        // Compute stress
        solver->compute_stress();
        std::vector<double> stress = solver->get_stress();

        // Calculate the total energy
        for(int i=0; i<M; i++)
        {
            w_minus[i] = (w[i]-w[i+M])/2;
            w_plus[i]  = (w[i]+w[i+M])/2;
        }

        energy_total = cb->inner_product(w_minus.data(),w_minus.data())/chi_n/cb->get_volume();
        energy_total -= cb->integral(w_plus.data())/cb->get_volume();
        for(int p=0; p<molecules->get_n_polymer_types(); p++){
            Polymer& pc = molecules->get_polymer(p);
            energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
        }

        for(int i=0; i<M; i++)
        {
            // Calculate pressure field for the new field calculation
            xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
            // Calculate output fields
            w_out[i]   = chi_n*phi_b[i] + xi[i];
            w_out[i+M] = chi_n*phi_a[i] + xi[i];
        }
        cb->zero_mean(&w_out[0]);
        cb->zero_mean(&w_out[M]);

        // Error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level;
        for(int i=0; i<2*M; i++)
            w_diff[i] = w_out[i]- w[i];
        error_level = sqrt(cb->multi_inner_product(2,w_diff.data(),w_diff.data())/
                        (cb->multi_inner_product(2,w.data(),w.data())+1.0));
        error_level += sqrt(stress[0]*stress[0] + stress[1]*stress[1] + stress[2]*stress[2]);

        // Print iteration # and error levels and check the mass conservation
        sum = (cb->integral(phi_a.data()) + cb->integral(phi_b.data()))/cb->get_volume() - 1.0;
        std::cout<< std::setw(8) << iter;
        std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
        std::cout<< "\t[" << std::setprecision(7) << std::scientific << solver->get_total_partition(0);
        for(int p=1; p<molecules->get_n_polymer_types(); p++)
            std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << solver->get_total_partition(p);
        std::cout<< "]";
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << error_level << std::endl;

        // Conditions to end the iteration
        if(error_level < tolerance) break;

        // Calculate new fields using simple and Anderson mixing  //w_new, w_current, w_diff
        for(int d=0; d<cb->get_dim(); d++)
        {
            w[2*M+d] = cb->get_lx(d);
            w_diff[2*M+d] = -stress[d];
        }
        am->calculate_new_fields(w.data(), w.data(), w_diff.data(), old_error_level, error_level);

        // Update lx
        for(int d=0; d<cb->get_dim(); d++)
            lx[d] = w[2*M+d];

        cb->set_lx(lx);
        solver->update_laplacian_operator();
    }
    std::cout << std::setprecision(default_precision) << std::defaultfloat;
    return energy_total;
}

}  // namespace

int main()
{
    try
    {
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<bool> aggregate_propagator_computations = {false, true};
        std::vector<bool> reduce_memorys = {false, true};

        Checks checks;
        for (const Architecture& architecture : architectures)
        {
            for(std::string chain_model : chain_models)
            {
                std::vector<double> energy_total_list;
                for(std::string platform : avail_platforms)
                {
                    for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                    {
                        for(bool reduce_memory : reduce_memorys)
                        {
                            checks.begin_run(architecture.name + ", " + chain_model + ", " + platform
                                             + (aggregate_propagator_computation ? ", aggregated" : "")
                                             + (reduce_memory ? ", reduce_memory" : ""));
                            energy_total_list.push_back(run_scft(architecture, chain_model, platform,
                                                                 aggregate_propagator_computation, reduce_memory, checks));
                            checks.end_run();
                        }
                    }
                }
                checks.close(architecture.name + ", " + chain_model + ": std. of the free energy",
                             standard_deviation(energy_total_list), 1e-7);
            }
        }
        // Each architecture and chain model on every platform in two memory modes, without and with aggregation
        const int expected_runs = static_cast<int>(architectures.size() * chain_models.size() * avail_platforms.size() * 2 * 2);
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
