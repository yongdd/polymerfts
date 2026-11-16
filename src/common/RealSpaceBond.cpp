/**
 * @file RealSpaceBond.cpp
 * @brief Sampled Gaussian bond kernels and ghost tables of the "rs" solvers.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

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
    double width = 0.0;
    double mean_u = 0.0;
    double var_u = 0.0;
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
    axis.width = s;
    axis.mean_u = mean_u;
    axis.var_u = var_u;
    const double ds_ddx = var_u > 0.0 ? -(2.0 * s * s * mean_u - s * s * var_u) / (dx * s * var_u) : 0.0;
    axis.dg_ddx.resize(axis.g.size());
    for (std::size_t d = 0; d < axis.g.size(); ++d)
    {
        const double u = (d * dx / s) * (d * dx / s);
        axis.dg_ddx[d] = axis.g[d] * (mean_u - u) / dx + axis.g[d] * (u - mean_u) / s * ds_ddx;
    }
    return axis;
}

// Evaluate sin(pi*numerator/denominator) after exact integer range reduction.
// This also keeps a point near either wall equally accurate.
double sine_fraction(long long numerator, long long denominator)
{
    numerator %= 2 * denominator;
    if (numerator < 0)
        numerator += 2 * denominator;
    double sign = 1.0;
    if (numerator > denominator)
    {
        numerator -= denominator;
        sign = -1.0;
    }
    if (numerator > denominator / 2)
        numerator = denominator - numerator;
    return sign * static_cast<double>(std::sin(std::numbers::pi_v<long double>
                                               * numerator / denominator));
}

// The Fourier symbol of the infinite sampled Gaussian, evaluated with its
// Poisson sum. Directly transforming the truncated stencil would retain its
// absolute roundoff/tail error instead of the tiny surviving eigenvalue.
std::pair<long double, long double> theta_sum(long double width_sq, long double theta)
{
    const long double pi = std::numbers::pi_v<long double>;
    const int extent = static_cast<int>(std::ceil(std::sqrt(200.0L / width_sq) / (2 * pi))) + 1;
    long double sum = 0.0L, moment = 0.0L;
    for (int k = -extent; k <= extent; ++k)
    {
        const long double frequency = theta + 2 * pi * k;
        const long double squared = frequency * frequency;
        const long double weight = std::exp(-0.5L * width_sq * (squared - theta * theta));
        sum += weight;
        moment += squared * weight;
    }
    return {sum, moment / sum};
}

// The folded kernel of a narrow absorbing axis or, with `dlength`, its derivative with respect
// to the box length: the n x n matrix whose entry (i, j) is sum_m weights[m] basis[m][i] basis[m][j]
// over the retained eigenmodes, summed in the order of the modes. Empty where ghost points carry
// the axis.
RealSpaceBond::Folded folded_axis(const Axis& axis, double dx, double length, int n,
                                  BoundaryCondition low, BoundaryCondition high, bool dlength)
{
    using BC = BoundaryCondition;
    if (axis.width == 0.0 || (low != BC::ABSORBING && high != BC::ABSORBING))
        return {};
    const bool both = low == BC::ABSORBING && high == BC::ABSORBING;
    const long double pi = std::numbers::pi_v<long double>;
    const long double width = static_cast<long double>(axis.width) / dx;
    const long double width_sq = width * width;
    const long double first_theta = pi / (both ? n : 2.0L * n);
    const long double first_exponent = 0.5L * width_sq * first_theta * first_theta;
    // Here the first eigenfunction dominates every folded entry. For two
    // absorbing walls, the sum of higher/first bounds is at most
    // sum_{m>=2} m^2 exp(-(m^2-1)) < 0.203; mixed walls are smaller still.
    // Forming each entry before multiplying q preserves positivity and local
    // relative precision, unlike transforming q and reconstructing it later.
    if (first_exponent < 1.0L)
        return {};

    std::vector<double> weights;   // per mode
    std::vector<double> basis;     // per mode, n values
    const auto [denominator, denominator_moment] = theta_sum(width_sq, 0.0L);
    // V = s^2 <u> stays fixed as L varies. This includes the sampled-width
    // correction even when an axis has only one or two grid points.
    const long double dwidth_sq = -4.0L * width_sq * axis.mean_u / (length * axis.var_u);
    for (int mode = 0; mode < n; ++mode)
    {
        const long double theta = pi * (both ? mode + 1.0L : mode + 0.5L) / n;
        const long double exponent = 0.5L * width_sq * theta * theta;
        // The omitted modes are below 1e-41 of the first mode, including their
        // derivative factors; there are at most ten retained modes.
        if (exponent - first_exponent > 112.0L)
            break;
        const auto [numerator, numerator_moment] = theta_sum(width_sq, theta);
        const long double eigenvalue = std::exp(-exponent) * numerator / denominator;
        const long double norm = both && mode == n - 1 ? n : 0.5L * n;
        weights.push_back(static_cast<double>(dlength
            ? -0.5L * dwidth_sq * (numerator_moment - denominator_moment) * eigenvalue / norm
            : eigenvalue / norm));
        const long long denominator_phase = (both ? 2LL : 4LL) * n;
        for (int i = 0; i < n; ++i)
        {
            long long numerator_phase = (both ? mode + 1LL : 2LL * mode + 1) * (2LL * i + 1);
            if (low == BC::REFLECTING)
                numerator_phase = denominator_phase / 2 - numerator_phase;
            basis.push_back(sine_fraction(numerator_phase, denominator_phase));
        }
    }

    // Each entry is formed whole, once per box, before it multiplies q
    RealSpaceBond::Folded matrix(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double entry = 0.0;
            for (std::size_t mode = 0; mode < weights.size(); ++mode)
                entry += weights[mode] * basis[mode * n + i] * basis[mode * n + j];
            matrix[static_cast<std::size_t>(i) * n + j] = entry;
        }
    return matrix;
}
}  // namespace

RealSpaceBond::Kernels RealSpaceBond::gaussian(double variance, const std::vector<double>& dx,
    const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc)
{
    Kernels kernels;
    for (std::size_t d = 0; d < dx.size(); ++d)
    {
        Axis axis = gaussian_axis(variance, dx[d]);
        kernels[d] = axis.g;
        kernels.folded[d] = folded_axis(axis, dx[d], dx[d] * nx[d], nx[d], bc[2*d], bc[2*d+1], false);
    }
    return kernels;
}

std::array<RealSpaceBond::Kernels, 3> RealSpaceBond::length_derivatives(
    double variance, const std::vector<double>& dx, const std::vector<double>& lx,
    const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc)
{
    const Kernels kernels = gaussian(variance, dx, nx, bc);
    std::array<Kernels, 3> derivatives = {kernels, kernels, kernels};
    for (std::size_t d = 0; d < dx.size(); ++d)
    {
        // The grid spacing is L/n, so d/dL = (dx/L) d/ddx
        Axis axis = gaussian_axis(variance, dx[d]);
        derivatives[d][d] = axis.dg_ddx;
        derivatives[d].folded[d] = folded_axis(axis, dx[d], lx[d], nx[d], bc[2*d], bc[2*d+1], true);
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
            if (k->folded[d].empty())
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
