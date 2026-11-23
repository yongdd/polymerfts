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
 * @file TestPseudoSolvent3D.cpp
 * @brief A solvent against a homopolymer of one segment.
 *
 * A blend of AB diblocks (volume fraction 0.6, f = 0.5, ds = 1/100) with A homopolymers
 * of length 0.01, a single segment (0.4), against the same diblocks with an A solvent
 * (0.4), on a 5x4x3 grid of a periodic 4x3x2 box: the partition function and the
 * concentration of the short homopolymer equal those of the solvent, and the stress of
 * the two blends agree. Continuous chains (rqm4) and discrete chains (ps, their default)
 * each run on every platform in full and reduced memory, in the stored real fields and
 * in random complex fields (seed 42); the complex computation computes no stress.
 */

#include <complex>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "TestChecks.h"
#include "TestComputationFactory.h"

namespace {

const int II{5};
const int JJ{4};
const int KK{3};
const int M{II*JJ*KK};
const int NN{100};
const double f = 0.5;
const double Lx = 4.0;
const double Ly = 3.0;
const double Lz = 2.0;

// The real fields
const double w_a[M] =
{
    0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
    0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
    0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
    0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
    0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
    0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
    0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
    0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
    0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
    0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
    0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
    0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
};
const double w_b[M] =
{
    0.113822903e-1,0.330673934e+0,0.270138412e+0,0.669606774e+0,0.885344778e-1,
    0.604752856e+0,0.890062293e+0,0.328557615e+0,0.965824739e+0,0.865399960e+0,
    0.698893686e+0,0.857947305e+0,0.594897904e+0,0.248187208e+0,0.155686710e+0,
    0.116803898e+0,0.711146609e+0,0.107610460e+0,0.143034307e+0,0.123131521e+0,
    0.230387237e+0,0.516274641e+0,0.562366089e-1,0.491449746e+0,0.746656140e+0,
    0.296108614e+0,0.424987667e+0,0.651538750e+0,0.116745920e+0,0.567790110e+0,
    0.954487190e+0,0.802476927e-1,0.440223916e+0,0.843025420e+0,0.612864528e+0,
    0.571893767e+0,0.759625605e+0,0.872255004e+0,0.935065364e+0,0.635565347e+0,
    0.373711972e-2,0.860683468e+0,0.186492706e+0,0.267880995e+0,0.579305501e+0,
    0.693549226e+0,0.613843845e+0,0.259811620e-1,0.848915465e+0,0.766111508e+0,
    0.872008750e+0,0.116289041e+0,0.917713893e+0,0.710076955e+0,0.442712526e+0,
    0.516722213e+0,0.253395805e+0,0.472950065e-1,0.152934959e+0,0.292486174e+0
};

/// The fields of type T: the stored fields, or random complex fields
template <typename T>
void make_fields(std::vector<T>& w_a_t, std::vector<T>& w_b_t)
{
    if constexpr (std::is_same_v<T, double>)
    {
        w_a_t.assign(w_a, w_a + M);
        w_b_t.assign(w_b, w_b + M);
    }
    else
    {
        w_a_t.resize(M);
        w_b_t.resize(M);
        std::mt19937 rng(42); // fixed seed
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for(int i=0; i<M; i++)
        {
            w_a_t[i] = T{dist(rng), dist(rng)};
            w_b_t[i] = T{dist(rng), dist(rng)};
        }
    }
}

/// A chain model with fields of type T on every platform and memory mode
template <typename T>
void run_case(const std::string& chain_model, const std::string& method, Checks& checks)
{
    std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};
    std::vector<BlockInput> blocks_1 =
    {
        {"A",    f, 0, 1},
        {"B",1.0-f, 1, 2},
    };
    std::vector<BlockInput> blocks_2 =
    {
        {"A", 0.01, 0, 1},
    };

    // The blend with the short homopolymer (polymer 1) and the one with the solvent (solvent 0)
    Molecules molecules_1(chain_model, 1.0/NN, bond_lengths);
    molecules_1.add_polymer(0.6, blocks_1, {});
    molecules_1.add_polymer(0.4, blocks_2, {});
    Molecules molecules_2(chain_model, 1.0/NN, bond_lengths);
    molecules_2.add_polymer(0.6, blocks_1, {});
    molecules_2.add_solvent(0.4, "A");

    PropagatorComputationOptimizer optimizer_1(&molecules_1, false);
    optimizer_1.display_blocks();
    optimizer_1.display_propagators();
    PropagatorComputationOptimizer optimizer_2(&molecules_2, false);
    optimizer_2.display_blocks();
    optimizer_2.display_propagators();

    std::vector<T> w_a_t, w_b_t;
    make_fields(w_a_t, w_b_t);
    const std::string type = std::is_same_v<T, double> ? "real" : "complex";

    std::vector<PlatformRun<T>> runs_1 = make_platform_runs<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}, &molecules_1, {&optimizer_1}, method);
    std::vector<PlatformRun<T>> runs_2 = make_platform_runs<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}, &molecules_2, {&optimizer_2}, method);
    for(size_t n=0; n<runs_1.size(); n++)
    {
        checks.begin_run(chain_model + ", " + type + ", " + runs_1[n].name);
        PropagatorComputation<T>* solver_1 = runs_1[n].solver.get();
        PropagatorComputation<T>* solver_2 = runs_2[n].solver.get();
        std::vector<T> phi_a(M), phi_solvent(M);

        const int p = 1;
        const int s = 0;
        solver_1->compute_propagators({{"A",w_a_t.data()},{"B",w_b_t.data()}},{});
        solver_1->compute_concentrations();
        solver_1->get_total_concentration(p, "A", phi_a.data());

        solver_2->compute_propagators({{"A",w_a_t.data()},{"B",w_b_t.data()}},{});
        solver_2->compute_concentrations();
        solver_2->get_solvent_concentration(s, phi_solvent.data());

        checks.that("check_total_partition (homopolymer)", solver_1->check_total_partition());
        checks.that("check_total_partition (solvent)", solver_2->check_total_partition());

        T QQ_1 = solver_1->get_total_partition(p);
        T QQ_2 = solver_2->get_solvent_partition(s);
        checks.close("Total partition", std::abs((QQ_1-QQ_2)/(Lx*Ly*Lz)), 1e-7);
        checks.close("Segment concentration A", max_distance(phi_a.data(), phi_solvent.data(), M), 1e-4);

        if constexpr (std::is_same_v<T, double>)
        {
            solver_1->compute_stress();
            std::vector<double> stress_1 = solver_1->get_stress();
            std::cout << "  Stress_1: " << stress_1[0] << ", " << stress_1[1] << ", " << stress_1[2] << std::endl;
            solver_2->compute_stress();
            std::vector<double> stress_2 = solver_2->get_stress();
            std::cout << "  Stress_2: " << stress_2[0] << ", " << stress_2[1] << ", " << stress_2[2] << std::endl;
            for(int i=0; i<3; i++)
                checks.close("Stress[" + std::to_string(i) + "]", std::abs(stress_1[i] - stress_2[i]), 1e-7);
        }
        checks.end_run();
    }
}

}  // namespace

int main()
{
    try
    {
        Checks checks;
        for (const std::string chain_model : {"Continuous", "Discrete"})
        {
            const std::string method = (chain_model == "Continuous") ? "rqm4" : "";
            run_case<double>(chain_model, method, checks);
            run_case<std::complex<double>>(chain_model, method, checks);
        }
        // Two chain models with real and complex fields, on every platform in two memory modes
        const int expected_runs = static_cast<int>(2 * 2 * PlatformSelector::avail_platforms().size() * 2);
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
