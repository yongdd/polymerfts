/**
 * @file RealSpaceBond.h
 * @brief Bond kernels and ghost points of the real-space discrete-chain solvers ("rs").
 *
 * A Gaussian bond of mean-square length b^2 ds (in units of R0^2) has the variance
 * b^2 ds / 3 along each axis of an orthogonal box. It is sampled on the grid and
 * cut where it falls below 1e-40 of its peak (13.6 standard deviations), and its
 * width is set so that the sampled kernel has exactly that variance: on grids that
 * resolve the bond the correction is negligible, and on coarse grids it keeps the
 * size of the chain. The kernels are summed to one. Their derivatives with respect to the box
 * lengths, which give the stress, include the change of that width.
 *
 * Ghost points carry the boundary conditions: the point j of an axis of n points
 * is folded back into 0..n-1, with a sign that turns negative on each reflection at
 * an absorbing wall. Folding repeats, so a kernel may be longer than the box.
 */

#ifndef REAL_SPACE_BOND_H_
#define REAL_SPACE_BOND_H_

#include <array>
#include <vector>

#include "ComputationBox.h"

namespace RealSpaceBond
{
    /// Weights g_0..g_H of one axis; g_{-d} = g_d
    using Kernel = std::vector<double>;
    /// One kernel per axis; axes the grid does not have hold {1}
    using Kernels = std::array<Kernel, 3>;

    /// Source and sign of the points -halo..n-1+halo of each axis
    struct Ghosts
    {
        std::array<int, 3> halo{};
        std::array<std::vector<int>, 3> index;
        std::array<std::vector<double>, 3> sign;
    };

    /// Kernels of a bond with the given variance along each axis
    Kernels gaussian(double variance, const std::vector<double>& dx);

    /// For each axis a, the kernels of the bond with the one of axis a replaced by its derivative with
    /// respect to the box length L_a (the grid points move with the box; the variance stays fixed)
    std::array<Kernels, 3> length_derivatives(double variance, const std::vector<double>& dx,
                                              const std::vector<double>& lx);

    /// Tables for halos as wide as the longest kernel of each axis
    Ghosts ghosts(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                  const std::vector<const Kernels*>& kernels);
}

#endif
