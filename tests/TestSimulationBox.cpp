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
#include <complex>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "AbstractFactory.h"
#include "GridLimits.h"
#include "PlatformSelector.h"
#include "TestCudaPart.h"
#ifdef USE_CUDA
#include <algorithm>
#include "CudaComputationBox.h"
#include "SpaceGroup.h"
#include "TestNanSafe.h"
#endif

template <typename T>
void check_overflow_grid(const std::string& platform)
{
    std::unique_ptr<AbstractFactory<T>> factory;
    if constexpr (std::is_same_v<T, double>)
        factory = PlatformSelector::create_factory_real(platform, false);
    else
        factory = PlatformSelector::create_factory_complex(platform, false);
    // Exercise the C++ constructors directly, without the Python size guard.
    for (const bool angles : {false, true})
    {
        try
        {
            if (angles)
                factory->create_computation_box({1 << 30, 1 << 30, 16}, {1.0, 1.0, 1.0}, {}, {90.0, 90.0, 90.0});
            else
                factory->create_computation_box({1 << 30, 1 << 30, 16}, {1.0, 1.0, 1.0}, {});
        }
        catch (const Exception& exc)
        {
            if (std::string(exc.what()).find("32-bit") == std::string::npos)
                throw;
            continue;
        }
        throw std::runtime_error("A grid whose product overflows 64 bits was accepted");
    }
}

// The angles of the axes a 1D or 2D box lacks hold at 90 degrees within
// FIXED_ANGLE_TOLERANCE in radians: 1e-9 degrees off is 1.7e-11 rad (accepted),
// 1e-8 degrees is 1.7e-10 rad (refused)
void check_absent_axis_angles(const std::string& platform)
{
    auto factory = PlatformSelector::create_factory_real(platform, false);
    auto accepted = [&](std::vector<int> nx, std::vector<double> lx, std::vector<double> angles) {
        try
        {
            factory->create_computation_box(nx, lx, {}, angles);
            return true;
        }
        catch (const Exception& exc)
        {
            if (std::string(exc.what()).find("90 degrees") == std::string::npos)
                throw;
            return false;
        }
    };
    if (!accepted({4, 4}, {1.0, 1.0}, {90.0 + 1e-9, 90.0, 120.0}) || accepted({4, 4}, {1.0, 1.0}, {90.0 + 1e-8, 90.0, 120.0})
        || !accepted({4}, {1.0}, {90.0, 90.0 - 1e-9, 90.0}) || accepted({4}, {1.0}, {90.0, 90.0 - 1e-8, 90.0}))
        throw std::runtime_error("The angles of absent axes are not compared with 1e-10 rad");
}

// Host reductions of a 1x2x2 box against reference values
static bool check_host_reductions(ComputationBox<double>& cb)
{
    double sum_w;
    double w[] {-0.81747, -2.02147, -3.52469, 1.43482};
    double g[] {2.82006, -0.944886, -1.37078, -2.47236};
    double h[] {4.40394, 1.71801, 2.73214, -0.499087};
    double gm[] {1.21997, 0.130716, -4.524, -1.09946, 1.79612, 0.367613, -0.670658, 1.53048};
    double hm[] {0.443984, 1.55078, 3.26028, -1.07704, 1.37609, -2.33879, -4.88014, 4.23021};

    cb.zero_mean(w);
    sum_w = 0.0;
    for(int i=0; i<cb.get_total_grid(); i++)
        sum_w += w[i];

    std::cout<< "nx: " << cb.get_nx(0) <<" "<< cb.get_nx(1) <<" "<< cb.get_nx(2) << std::endl;
    std::cout<< "lx: " << cb.get_lx(0) <<" "<< cb.get_lx(1) <<" "<< cb.get_lx(2) << std::endl;
    std::cout<< "dx: " << cb.get_dx(0) <<" "<< cb.get_dx(1) <<" "<< cb.get_dx(2) << std::endl;

    std::cout<< "inner product: "<< cb.inner_product(g,h) << std::endl;
    std::cout<< "multi inner product: "<< cb.multi_inner_product(2, gm,hm) << std::endl;
    std::cout<< "sum(w): " << sum_w << std::endl;

    if( !(std::abs(cb.inner_product(g,h)-81.553611) <= 1e-5))
        return false;
    if( !(std::abs(cb.multi_inner_product(2,gm,hm)+14.391321) <= 1e-5))
        return false;
    if( !(std::abs(sum_w) <= 1e-7))
        return false;
    return true;
}

#ifdef USE_CUDA
// Device reductions of a box against its host ones, for two fields of length n
// (the full grid, or the reduced basis with a space group); returns the largest
// difference
static double device_host_difference(CudaComputationBox<double>& cb, int n)
{
    std::vector<double> g(n), h(n);
    for (int i = 0; i < n; i++)
    {
        g[i] = std::sin(0.7*i + 0.3) + 0.5;
        h[i] = std::cos(1.3*i) - 0.2;
    }
    double *d_g, *d_h;
    CUDA_CHECK(cudaMalloc((void**)&d_g, sizeof(double)*n));
    CUDA_CHECK(cudaMalloc((void**)&d_h, sizeof(double)*n));
    CUDA_CHECK(cudaMemcpy(d_g, g.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, h.data(), sizeof(double)*n, cudaMemcpyHostToDevice));

    double diff = std::abs(cb.integral_device(d_g) - cb.integral(g.data()));
    diff = max_or_nan(diff, std::abs(cb.inner_product_device(d_g, d_h) - cb.inner_product(g.data(), h.data())));

    cudaFree(d_g);
    cudaFree(d_h);
    return diff;
}

// Device reductions = host reductions, on the full grid and on the reduced
// basis of Im-3m with a mask of that symmetry (cavities at the corner and the
// body center).
static bool check_device_reductions()
{
    CudaComputationBox<double> cb({1,2,2}, {4.5,3.5,2.5}, {});
    double diff_full = device_host_difference(cb, cb.get_total_grid());
    std::cout<< "device - host reductions, full grid: " << diff_full << std::endl;
    if( !(diff_full <= 1e-10))
        return false;

    const int n = 16;
    std::vector<double> mask(n*n*n, 1.0);
    for(int i=0; i<n*n*n; i++)
    {
        const int idx[3] {i/(n*n), (i/n)%n, i%n};
        for(double c : {0.0, 0.5})
        {
            double dist2 = 0.0;
            for(int d=0; d<3; d++)
            {
                double x = std::abs((idx[d] + 0.5)/n - c);
                x = std::min(x, 1.0 - x);
                dist2 += x*x;
            }
            if (dist2 < 0.15*0.15)
                mask[i] = 0.0;
        }
    }
    CudaComputationBox<double> cb_sg({n,n,n}, {3.0,3.0,3.0}, {}, mask.data());
    SpaceGroup sg({n,n,n}, "Im-3m", 529);
    cb_sg.set_space_group(&sg);
    double diff_sg = device_host_difference(cb_sg, cb_sg.get_n_basis());
    std::cout<< "device - host reductions, Im-3m reduced basis with a mask: " << diff_sg << std::endl;
    return diff_sg <= 1e-10;
}
#endif

int main()
{
    try
    {
        // The count itself accepts the boundary, without allocating a huge box.
        if (checked_grid_points({static_cast<int>(MAX_GRID_POINTS)}) != MAX_GRID_POINTS)
            return -1;
        for (const std::string& platform : PlatformSelector::avail_platforms())
        {
            std::cout << "Platform: " << platform << std::endl;
            check_overflow_grid<double>(platform);
            check_overflow_grid<std::complex<double>>(platform);
            check_absent_axis_angles(platform);
            auto factory = PlatformSelector::create_factory_real(platform, false);
            std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box({1,2,2}, {4.5,3.5,2.5}, {});
            if (!check_host_reductions(*cb))
                return -1;
#ifdef USE_CUDA
            if (platform == "cuda" && !check_device_reductions())
                return -1;
#endif
        }
        return cuda_part_missing() ? 1 : 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
