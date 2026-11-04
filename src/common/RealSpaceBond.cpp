/**
 * @file RealSpaceBond.cpp
 * @brief Sampled Gaussian bond kernels and ghost tables of the "rs" solvers.
 */

#include <algorithm>
#include <cmath>

#include "Exception.h"
#include "RealSpaceBond.h"

namespace
{
// The kernel ends where exp(-x^2/2) falls below 1e-40
constexpr double kernel_cut = 13.6;

// Weights of the Gaussian of width s sampled with spacing dx over -half..half, summed to one
std::vector<double> sampled(double s, double dx, int half)
{
    std::vector<double> g(half + 1);
    double sum = 0.0;
    for (int d = 0; d <= half; ++d)
    {
        const double x = d * dx / s;
        g[d] = std::exp(-0.5 * x * x);
        sum += (d == 0 ? 1.0 : 2.0) * g[d];
    }
    for (double& v : g)
        v /= sum;
    return g;
}

double variance_of(const std::vector<double>& g, double dx)
{
    double v = 0.0;
    for (std::size_t d = 1; d < g.size(); ++d)
        v += 2.0 * g[d] * d * d * dx * dx;
    return v;
}

// The kernel of one axis and its derivative with respect to the grid spacing
struct Axis
{
    RealSpaceBond::Kernel g;
    RealSpaceBond::Kernel dg_ddx;
};

Axis gaussian_axis(double variance, double dx)
{
    if (variance <= 0.0)
        return {{1.0}, {0.0}};

    // Width whose sampled kernel has the variance: the variance on the grid grows with
    // the width, so bisect between zero and a width beyond it
    double lo = 0.0;
    double hi = std::sqrt(variance) + dx;
    auto grid_variance = [&](double s, int half) { return variance_of(sampled(s, dx, half), dx); };
    int half = static_cast<int>(std::ceil(kernel_cut * hi / dx));
    while (grid_variance(hi, half) < variance)
    {
        hi *= 2.0;
        half = static_cast<int>(std::ceil(kernel_cut * hi / dx));
    }
    for (int iter = 0; iter < 200 && hi - lo > 1e-15 * hi; ++iter)
    {
        const double mid = 0.5 * (lo + hi);
        (grid_variance(mid, half) < variance ? lo : hi) = mid;
    }
    const double s = 0.5 * (lo + hi);
    Axis axis;
    axis.g = sampled(s, dx, std::max(1, static_cast<int>(std::ceil(kernel_cut * s / dx))));

    // Derivative of g_d = exp(-u_d/2)/Z, u_d = (d dx/s)^2, along the width s(dx) that keeps the
    // variance s^2 <u> on the grid fixed: dg/ddx = dg/ddx|_s + dg/ds|_dx ds/ddx, where
    // ds/ddx = -(2 V - s^2 var(u)) / (dx s var(u)) from the derivatives of V = s^2 <u>
    double mean_u = 0.0, mean_u2 = 0.0;
    for (std::size_t d = 0; d < axis.g.size(); ++d)
    {
        const double u = (d * dx / s) * (d * dx / s);
        const double w = (d == 0 ? 1.0 : 2.0) * axis.g[d];
        mean_u += w * u;
        mean_u2 += w * u * u;
    }
    const double var_u = mean_u2 - mean_u * mean_u;
    const double ds_ddx = var_u > 0.0 ? -(2.0 * s * s * mean_u - s * s * var_u) / (dx * s * var_u) : 0.0;
    axis.dg_ddx.resize(axis.g.size());
    for (std::size_t d = 0; d < axis.g.size(); ++d)
    {
        const double u = (d * dx / s) * (d * dx / s);
        axis.dg_ddx[d] = axis.g[d] * (mean_u - u) / dx + axis.g[d] * (u - mean_u) / s * ds_ddx;
    }
    return axis;
}
}  // namespace

RealSpaceBond::Kernels RealSpaceBond::gaussian(double variance, const std::vector<double>& dx)
{
    Kernels kernels = {Kernel{1.0}, Kernel{1.0}, Kernel{1.0}};
    for (std::size_t d = 0; d < dx.size(); ++d)
        kernels[d] = gaussian_axis(variance, dx[d]).g;
    return kernels;
}

std::array<RealSpaceBond::Kernels, 3> RealSpaceBond::length_derivatives(
    double variance, const std::vector<double>& dx, const std::vector<double>& lx)
{
    const Kernels kernels = gaussian(variance, dx);
    std::array<Kernels, 3> derivatives = {kernels, kernels, kernels};
    for (std::size_t d = 0; d < dx.size(); ++d)
    {
        // The grid spacing is L/n, so d/dL = (dx/L) d/ddx
        derivatives[d][d] = gaussian_axis(variance, dx[d]).dg_ddx;
        for (double& v : derivatives[d][d])
            v *= dx[d] / lx[d];
    }
    return derivatives;
}

RealSpaceBond::Ghosts RealSpaceBond::ghosts(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                                            const std::vector<const Kernels*>& kernels)
{
    if (bc.size() != 2 * nx.size())
        throw_with_line_number("Expected two boundary conditions per axis.");

    Ghosts ghosts;
    for (std::size_t d = 0; d < nx.size(); ++d)
    {
        const int n = nx[d];
        const BoundaryCondition low = bc[2 * d];
        const BoundaryCondition high = bc[2 * d + 1];
        if ((low == BoundaryCondition::PERIODIC) != (high == BoundaryCondition::PERIODIC))
            throw_with_line_number("A periodic axis must be periodic at both ends.");

        int halo = 0;
        for (const Kernels* k : kernels)
            halo = std::max(halo, static_cast<int>((*k)[d].size()) - 1);
        ghosts.halo[d] = halo;
        ghosts.index[d].resize(n + 2 * halo);
        ghosts.sign[d].resize(n + 2 * halo);

        for (int p = 0; p < n + 2 * halo; ++p)
        {
            long j = p - halo;
            double sign = 1.0;
            if (low == BoundaryCondition::PERIODIC)
                j = ((j % n) + n) % n;
            // Mirror about the cell faces at -1/2 and n-1/2 until the point is inside
            while (j < 0 || j >= n)
            {
                if (j < 0)
                {
                    j = -1 - j;
                    if (low == BoundaryCondition::ABSORBING)
                        sign = -sign;
                }
                else
                {
                    j = 2L * n - 1 - j;
                    if (high == BoundaryCondition::ABSORBING)
                        sign = -sign;
                }
            }
            ghosts.index[d][p] = static_cast<int>(j);
            ghosts.sign[d][p] = sign;
        }
    }
    return ghosts;
}
