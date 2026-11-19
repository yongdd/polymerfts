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
 * @file TestCompareRealSpacePseudoSpectral.cpp
 * @brief Compare real-space and pseudo-spectral methods for non-periodic BCs.
 *
 * This test verifies that both numerical methods produce consistent results
 * for propagator computation with non-periodic boundary conditions:
 *
 * - Pseudo-spectral: DCT (reflecting) / DST (absorbing) transforms
 * - Real-space: Crank-Nicolson finite differences with ADI splitting
 *
 * Both methods should converge to the same solution as grid resolution
 * increases. The pseudo-spectral method has higher accuracy (spectral
 * convergence) while the real-space method is more flexible.
 *
 * Tests cover:
 * - 1D reflecting, absorbing, and mixed BCs
 * - 2D reflecting, absorbing, and mixed BCs
 * - 3D reflecting, absorbing, and mixed BCs
 *
 * An axis that reflects at one end and absorbs at the other, which the pseudo-spectral
 * method cannot take, is checked against its exact oracle: the half of the box doubled
 * across the reflecting wall, absorbing at both ends, with the mirrored input.
 *
 * Platform support:
 * - CPU (FFTW): Both pseudo-spectral (DCT/DST) and real-space
 * - CUDA: Real-space only for non-periodic BCs (pseudo-spectral uses FFT only)
 *
 * The test compares:
 * - CPU pseudo-spectral vs CPU real-space
 * - CUDA real-space vs CPU pseudo-spectral (to verify CUDA implementation)
 */

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"

#include "TestComputationFactory.h"
#include "TestNanSafe.h"
#include "ComputationBox.h"

#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include <cuda_runtime.h>
// A CUDA build tested on a node without a visible GPU fails, like the TestCuda*
// tests: a skipped CUDA part would read as a passed CUDA test.
[[maybe_unused]] static bool cuda_device_visible()
{
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0)
    {
        std::cout << "FAILED: this build has the CUDA backend, but no CUDA device is visible." << std::endl;
        std::exit(1);
    }
    return true;
}
#endif

/**
 * @brief Check if boundary conditions are mixed (different on each side of a direction).
 *
 * Pseudo-spectral methods require matching BCs on both sides of each direction
 * because FFT/DCT/DST transforms apply to the entire direction.
 */
bool has_mixed_bc(const std::vector<std::string>& bc)
{
    int dim = bc.size() / 2;
    for (int d = 0; d < dim; ++d)
    {
        if (bc[2*d] != bc[2*d + 1])
            return true;
    }
    return false;
}

/**
 * @brief Compare the real-space propagator of a box whose one axis reflects at one end and
 *        absorbs at the other with its exact oracle.
 *
 * The cell-centred reflecting wall is the mirror plane of the box doubled across it, which
 * absorbs at both ends: next to the wall the finite-difference Laplacian reads the mirror image
 * of a point as its ghost value, and the doubled box keeps a mirror-symmetric input symmetric.
 * The propagator of the mixed box therefore equals its half of the doubled box with the mirrored
 * input, up to the rounding of the larger tridiagonal solves.
 *
 * @param q_mixed Propagator of the mixed box after n_steps steps from q_init
 * @return true if they agree to 1e-12 of the largest value
 */
template <typename T>
bool mirror_oracle_passed(const std::string& platform, const std::vector<int>& nx, const std::vector<double>& lx,
                          const std::vector<std::string>& bc, const std::vector<double>& q_init,
                          const std::vector<double>& q_mixed, int n_steps, Molecules& molecules,
                          PropagatorComputationOptimizer& prop_opt)
{
    const int dim = nx.size();
    int axis = -1;
    int n_mixed = 0;
    for (int d = 0; d < dim; ++d)
    {
        if (bc[2*d] != bc[2*d + 1])
        {
            axis = d;
            ++n_mixed;
        }
    }
    if (n_mixed != 1 || std::count(bc.begin() + 2*axis, bc.begin() + 2*axis + 2, std::string("reflecting")) != 1
        || std::count(bc.begin() + 2*axis, bc.begin() + 2*axis + 2, std::string("absorbing")) != 1)
    {
        std::cout << "FAILED (the mirror oracle takes one axis reflecting at one end and absorbing at the other)" << std::endl;
        return false;
    }

    // The doubled box: the mixed axis twice as long, absorbing at both ends, the mixed box in the
    // half next to its absorbing end (outer x n x inner points)
    const bool reflecting_low = bc[2*axis] == "reflecting";
    std::vector<int> nx_2 = nx;
    std::vector<double> lx_2 = lx;
    std::vector<std::string> bc_2 = bc;
    nx_2[axis] *= 2;
    lx_2[axis] *= 2.0;
    bc_2[2*axis] = "absorbing";
    bc_2[2*axis + 1] = "absorbing";
    const int n = nx[axis];
    int outer = 1;
    int inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= nx[d];
    for (int d = axis + 1; d < dim; ++d)
        inner *= nx[d];
    auto mixed_index = [&](int o, int i, int r) { return (static_cast<long>(o) * n + i) * inner + r; };
    auto doubled_index = [&](int o, int i, int r) { return (static_cast<long>(o) * 2 * n + i) * inner + r; };
    const int offset = reflecting_low ? n : 0;

    // Point i of the mixed box and its mirror image
    std::vector<double> q_2(2 * q_init.size()), q_out(2 * q_init.size());
    for (int o = 0; o < outer; ++o)
        for (int i = 0; i < n; ++i)
            for (int r = 0; r < inner; ++r)
            {
                q_2[doubled_index(o, offset + i, r)] = q_init[mixed_index(o, i, r)];
                q_2[doubled_index(o, reflecting_low ? n - 1 - i : 2 * n - 1 - i, r)] = q_init[mixed_index(o, i, r)];
            }

    ComputationBox<T>* cb_2 = new ComputationBox<T>(nx_2, lx_2, bc_2);
    PropagatorComputation<T>* solver_2 = make_propagator_computation<T>(platform, false, cb_2, &molecules, &prop_opt, "cn-adi2");
    std::vector<double> w_2(q_2.size(), 0.0);
    solver_2->compute_propagators({{"A", w_2.data()}}, {});
    for (int step = 0; step < n_steps; ++step)
    {
        solver_2->advance_propagator_single_segment(q_2.data(), q_out.data(), 0, 0, 1);
        std::swap(q_2, q_out);
    }
    delete solver_2;
    delete cb_2;

    double max_diff = 0.0;
    for (int o = 0; o < outer; ++o)
        for (int i = 0; i < n; ++i)
            for (int r = 0; r < inner; ++r)
            {
                const double d = std::abs(q_mixed[mixed_index(o, i, r)] - q_2[doubled_index(o, offset + i, r)]);
                if (std::isnan(d) || d > max_diff)
                    max_diff = d;   // a NaN stays
            }
    const double rel_err = max_diff / max_or_nan(q_mixed);
    if (!(rel_err <= 1e-12))
    {
        std::cout << "FAILED (rel_err=" << rel_err << " against the doubled box)" << std::endl;
        return false;
    }
    std::cout << "PASSED (rel_err=" << rel_err << " against the doubled box)" << std::endl;
    return true;
}

/**
 * @brief Run a single comparison test between pseudo-spectral and real-space methods.
 *
 * @param test_name   Name of the test for output
 * @param nx          Grid dimensions
 * @param lx          Box dimensions
 * @param bc          Boundary conditions
 * @param n_steps     Number of propagator steps
 * @param tolerance   Maximum allowed relative error
 * @return true if test passed, false otherwise
 *
 * Note: For mixed BCs (different on each side of a direction), only real-space
 * is tested since pseudo-spectral doesn't support mixed BCs; it is compared with
 * the doubled box of mirror_oracle_passed().
 */
template <typename T>
bool run_comparison_test(
    const std::string& test_name,
    const std::vector<int>& nx,
    const std::vector<double>& lx,
    const std::vector<std::string>& bc,
    int n_steps,
    double tolerance)
{
    const double ds = 0.01;
    int dim = nx.size();
    int M = 1;
    for (int i = 0; i < dim; ++i)
        M *= nx[i];

    std::map<std::string, double> bond_lengths = {{"A", 1.0}};
    std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};

    Molecules molecules("Continuous", ds, bond_lengths);
    molecules.add_polymer(1.0, blocks, {});
    PropagatorComputationOptimizer prop_opt(&molecules, false);

    // Create Gaussian initial condition
    std::vector<double> q_init(M);
    if (dim == 1)
    {
        double dx = lx[0] / nx[0];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            q_init[i] = std::exp(-std::pow(x - lx[0]/2, 2) / (2 * 0.4 * 0.4));
        }
    }
    else if (dim == 2)
    {
        double dx = lx[0] / nx[0];
        double dy = lx[1] / nx[1];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < nx[1]; ++j)
            {
                double y = (j + 0.5) * dy;
                int idx = i * nx[1] + j;
                q_init[idx] = std::exp(-((x - lx[0]/2)*(x - lx[0]/2) + (y - lx[1]/2)*(y - lx[1]/2)) / (2 * 0.4 * 0.4));
            }
        }
    }
    else if (dim == 3)
    {
        double dx = lx[0] / nx[0];
        double dy = lx[1] / nx[1];
        double dz = lx[2] / nx[2];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < nx[1]; ++j)
            {
                double y = (j + 0.5) * dy;
                for (int k = 0; k < nx[2]; ++k)
                {
                    double z = (k + 0.5) * dz;
                    int idx = (i * nx[1] + j) * nx[2] + k;
                    q_init[idx] = std::exp(-((x - lx[0]/2)*(x - lx[0]/2) + (y - lx[1]/2)*(y - lx[1]/2) + (z - lx[2]/2)*(z - lx[2]/2)) / (2 * 0.5 * 0.5));
                }
            }
        }
    }

    std::vector<double> w(M, 0.0);
    std::vector<double> q_ps(M), q_rs(M);
    std::vector<double> q_out(M);
    double max_diff, rel_err;
    bool all_passed = true;

    // Check if BCs are mixed (different on each side of a direction)
    // Pseudo-spectral doesn't support mixed BCs
    bool mixed_bc = has_mixed_bc(bc);

    // ====================================================================
    // CPU Tests: Compare pseudo-spectral vs real-space
    // Both methods support non-periodic BCs on CPU (DCT/DST for pseudo-spectral)
    // For mixed BCs, only test real-space (pseudo-spectral not supported)
    // ====================================================================
#ifdef USE_CPU_FFTW
    std::vector<double> q_cpu_ref(M);  // Reference for CUDA comparison

    if (mixed_bc)
    {
        // Mixed BCs: only test real-space, against the doubled box
        std::cout << "  [CPU-FFTW RS, mirror oracle] " << std::flush;
        ComputationBox<T>* cb_rs = new ComputationBox<T>(nx, lx, bc);
        PropagatorComputation<T>* solver_rs = make_propagator_computation<T>("cpu-fftw", false, cb_rs, &molecules, &prop_opt, "cn-adi2");

        // Run real-space solver
        solver_rs->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), 0, 0, 1);
            std::swap(q_rs, q_out);
        }

        if (!mirror_oracle_passed<T>("cpu-fftw", nx, lx, bc, q_init, q_rs, n_steps, molecules, prop_opt))
            all_passed = false;

        q_cpu_ref = q_rs;  // Use CPU real-space as reference for CUDA

        delete solver_rs;
        delete cb_rs;
    }
    else
    {
        // Matching BCs: compare pseudo-spectral vs real-space
        std::cout << "  [CPU-FFTW PS vs RS] " << std::flush;
        ComputationBox<T>* cb_ps = new ComputationBox<T>(nx, lx, bc);
        ComputationBox<T>* cb_rs = new ComputationBox<T>(nx, lx, bc);

        PropagatorComputation<T>* solver_ps = make_propagator_computation<T>("cpu-fftw", false, cb_ps, &molecules, &prop_opt, "rqm4");
        PropagatorComputation<T>* solver_rs = make_propagator_computation<T>("cpu-fftw", false, cb_rs, &molecules, &prop_opt, "cn-adi2");

        // Run pseudo-spectral solver
        solver_ps->compute_propagators({{"A", w.data()}}, {});
        q_ps = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_ps->advance_propagator_single_segment(q_ps.data(), q_out.data(), 0, 0, 1);
            std::swap(q_ps, q_out);
        }

        // Run real-space solver
        solver_rs->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), 0, 0, 1);
            std::swap(q_rs, q_out);
        }

        max_diff = 0.0;
        for (int i = 0; i < M; ++i)
            { const double d = std::abs(q_ps[i] - q_rs[i]); if (std::isnan(d) || d > max_diff) max_diff = d; }  // a NaN stays
        rel_err = max_diff / (max_or_nan(q_ps));

        if (!(rel_err <= tolerance))
        {
            std::cout << "FAILED (rel_err=" << rel_err << ")" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << "PASSED (rel_err=" << rel_err << ")" << std::endl;
        }

        q_cpu_ref = q_ps;  // Use CPU pseudo-spectral as reference for CUDA

        delete solver_ps;
        delete solver_rs;
        delete cb_ps;
        delete cb_rs;
    }
#endif

#ifdef USE_CPU_MKL
#ifndef USE_CPU_FFTW
    std::vector<double> q_cpu_ref(M);  // Reference for CUDA comparison
#endif

    if (mixed_bc)
    {
        // Mixed BCs: only test real-space, against the doubled box
        std::cout << "  [CPU-MKL RS, mirror oracle] " << std::flush;
        ComputationBox<T>* cb_rs_mkl = new ComputationBox<T>(nx, lx, bc);
        PropagatorComputation<T>* solver_rs_mkl = make_propagator_computation<T>("cpu-mkl", false, cb_rs_mkl, &molecules, &prop_opt, "cn-adi2");

        // Run real-space solver
        solver_rs_mkl->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs_mkl->advance_propagator_single_segment(q_rs.data(), q_out.data(), 0, 0, 1);
            std::swap(q_rs, q_out);
        }

        if (!mirror_oracle_passed<T>("cpu-mkl", nx, lx, bc, q_init, q_rs, n_steps, molecules, prop_opt))
            all_passed = false;

#ifndef USE_CPU_FFTW
        q_cpu_ref = q_rs;  // Use CPU real-space as reference for CUDA
#endif

        delete solver_rs_mkl;
        delete cb_rs_mkl;
    }
    else
    {
        // Matching BCs: compare pseudo-spectral vs real-space
        std::cout << "  [CPU-MKL PS vs RS] " << std::flush;
        ComputationBox<T>* cb_ps_mkl = new ComputationBox<T>(nx, lx, bc);
        ComputationBox<T>* cb_rs_mkl = new ComputationBox<T>(nx, lx, bc);

        PropagatorComputation<T>* solver_ps_mkl = make_propagator_computation<T>("cpu-mkl", false, cb_ps_mkl, &molecules, &prop_opt, "rqm4");
        PropagatorComputation<T>* solver_rs_mkl = make_propagator_computation<T>("cpu-mkl", false, cb_rs_mkl, &molecules, &prop_opt, "cn-adi2");

        // Run pseudo-spectral solver
        solver_ps_mkl->compute_propagators({{"A", w.data()}}, {});
        q_ps = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_ps_mkl->advance_propagator_single_segment(q_ps.data(), q_out.data(), 0, 0, 1);
            std::swap(q_ps, q_out);
        }

        // Run real-space solver
        solver_rs_mkl->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs_mkl->advance_propagator_single_segment(q_rs.data(), q_out.data(), 0, 0, 1);
            std::swap(q_rs, q_out);
        }

        max_diff = 0.0;
        for (int i = 0; i < M; ++i)
            { const double d = std::abs(q_ps[i] - q_rs[i]); if (std::isnan(d) || d > max_diff) max_diff = d; }  // a NaN stays
        rel_err = max_diff / (max_or_nan(q_ps));

        if (!(rel_err <= tolerance))
        {
            std::cout << "FAILED (rel_err=" << rel_err << ")" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << "PASSED (rel_err=" << rel_err << ")" << std::endl;
        }

#ifndef USE_CPU_FFTW
        q_cpu_ref = q_ps;  // Use CPU pseudo-spectral as reference for CUDA
#endif

        delete solver_ps_mkl;
        delete solver_rs_mkl;
        delete cb_ps_mkl;
        delete cb_rs_mkl;
    }
#endif

    // ====================================================================
    // CUDA Tests: Compare CUDA real-space vs CPU reference
    // For matching BCs: compare with CPU pseudo-spectral
    // For mixed BCs: compare with CPU real-space
    // ====================================================================
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#ifdef USE_CUDA
    if (cuda_device_visible())
    {
        std::cout << "  [CUDA RS vs CPU " << (mixed_bc ? "RS" : "PS") << "] " << std::flush;
        {
            CudaComputationBox<T>* cb_cuda_rs = new CudaComputationBox<T>(nx, lx, bc);
            PropagatorComputation<T>* solver_cuda_rs = make_propagator_computation<T>("cuda", false, cb_cuda_rs, &molecules, &prop_opt, "cn-adi2");

            // Run CUDA real-space solver
            solver_cuda_rs->compute_propagators({{"A", w.data()}}, {});
            q_rs = q_init;
            for (int step = 0; step < n_steps; ++step)
            {
                solver_cuda_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), 0, 0, 1);
                std::swap(q_rs, q_out);
            }

            // Compare CUDA real-space result with CPU reference
            max_diff = 0.0;
            for (int i = 0; i < M; ++i)
                { const double d = std::abs(q_cpu_ref[i] - q_rs[i]); if (std::isnan(d) || d > max_diff) max_diff = d; }  // a NaN stays
            rel_err = max_diff / (max_or_nan(q_cpu_ref));

            // For mixed BCs comparing RS vs RS, use tighter tolerance
            double cuda_tol = mixed_bc ? 1e-10 : tolerance;
            if (!(rel_err <= cuda_tol))
            {
                std::cout << "FAILED (rel_err=" << rel_err << ")" << std::endl;
                all_passed = false;
            }
            else
            {
                std::cout << "PASSED (rel_err=" << rel_err << ")" << std::endl;
            }

            delete solver_cuda_rs;
            delete cb_cuda_rs;
        }
    }
#endif
#endif

    return all_passed;
}

int main()
{
    try
    {
        bool all_passed = true;

        // Tolerance for comparison (real-space is 2nd order, pseudo-spectral is spectral)
        const double tol_1d = 0.01;   // 1% relative error for 1D
        const double tol_2d = 0.02;   // 2% relative error for 2D
        const double tol_3d = 0.05;   // 5% relative error for 3D

        const int n_steps_1d = 30;
        const int n_steps_2d = 30;
        const int n_steps_3d = 20;

        //=======================================================================
        // Test 1: 1D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 1: 1D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "1D Reflecting BC",
            {64}, {4.0},
            {"reflecting", "reflecting"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 2: 1D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 2: 1D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "1D Absorbing BC",
            {64}, {4.0},
            {"absorbing", "absorbing"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 3: 2D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 3: 2D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "2D Reflecting BC",
            {32, 24}, {4.0, 3.0},
            {"reflecting", "reflecting", "reflecting", "reflecting"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 4: 2D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 4: 2D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "2D Absorbing BC",
            {32, 24}, {4.0, 3.0},
            {"absorbing", "absorbing", "absorbing", "absorbing"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 5: 1D Mixed BC (Reflecting left, Absorbing right)
        //=======================================================================
        std::cout << "\nTest 5: 1D Mixed BC (Reflect-Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "1D Mixed BC",
            {64}, {4.0},
            {"reflecting", "absorbing"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 6: 2D Mixed BC (Reflecting in X, Absorbing in Y)
        //=======================================================================
        std::cout << "\nTest 6: 2D Mixed BC (X:Reflect, Y:Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "2D Mixed BC X:Reflect Y:Absorb",
            {32, 24}, {4.0, 3.0},
            {"reflecting", "reflecting", "absorbing", "absorbing"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 7: 2D Mixed BC (Absorbing in X, Reflecting in Y)
        //=======================================================================
        std::cout << "\nTest 7: 2D Mixed BC (X:Absorb, Y:Reflect)" << std::endl;
        if (!run_comparison_test<double>(
            "2D Mixed BC X:Absorb Y:Reflect",
            {32, 24}, {4.0, 3.0},
            {"absorbing", "absorbing", "reflecting", "reflecting"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 8: 3D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 8: 3D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "3D Reflecting BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"reflecting", "reflecting", "reflecting", "reflecting", "reflecting", "reflecting"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Test 9: 3D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 9: 3D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "3D Absorbing BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Test 10: 3D Mixed BC (Reflecting in X/Y, Absorbing in Z)
        //=======================================================================
        std::cout << "\nTest 10: 3D Mixed BC (X:Reflect, Y:Reflect, Z:Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "3D Mixed BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Final Summary
        //=======================================================================
        if (all_passed)
        {
            std::cout << "\nAll tests passed!" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "\nSome tests FAILED!" << std::endl;
            return -1;
        }
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
