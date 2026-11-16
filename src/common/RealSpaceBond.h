/**
 * @file RealSpaceBond.h
 * @brief Bond kernels and ghost points of the real-space discrete-chain solvers ("rs").
 *
 * A Gaussian bond of mean-square length b^2 ds (in units of R0^2) has the variance
 * b^2 ds / 3 along each axis of an orthogonal box. It is sampled on the grid and
 * cut where it falls below 1e-40 of its peak (13.6 standard deviations), and its
 * width is set so that the sampled kernel has exactly that variance: on grids that
 * resolve the bond the correction is negligible, and on coarse grids it keeps the
 * size of the chain. These free-space stencils sum to one. Their derivatives with respect to the box
 * lengths, which give the stress, include the change of that width.
 *
 * Ghost points carry the boundary conditions: the point j of an axis of n points
 * is folded back into 0..n-1, with a sign that turns negative on each reflection at
 * an absorbing wall. Folding repeats, so a kernel may be longer than the box.
 * Narrow absorbing axes use a compact sine/cosine expansion of the folded
 * kernel to avoid cancellation between images. Its eigenvalues come from the
 * Poisson sum of the same variance-corrected sampled Gaussian without the tail
 * cutoff; the absorbing folded rows need not sum to one. Their box-length
 * derivatives include the width correction as well. The expansion is summed
 * into the dense n x n matrix of the folded kernel once per box.
 *
 * bonds() builds all of this for the molecules in a box, once per box, for the
 * CPU and CUDA solvers alike.
 */

#ifndef REAL_SPACE_BOND_H_
#define REAL_SPACE_BOND_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include "ComputationBox.h"

class Molecules;

namespace RealSpaceBond
{
    /// Weights g_0..g_H of one axis; g_{-d} = g_d
    using Kernel = std::vector<double>;
    /// The folded kernel of a narrow absorbing axis of n points: the n x n matrix of its
    /// entries, (i, j) at i*n + j; empty on the other axes
    using Folded = std::vector<double>;

    /// Stencils on ordinary axes; the folded matrix on narrow absorbing axes.
    struct Kernels
    {
        std::array<Kernel, 3> stencil{Kernel{1.0}, Kernel{1.0}, Kernel{1.0}};
        std::array<Folded, 3> folded;
        Kernel& operator[](std::size_t d) { return stencil[d]; }
        const Kernel& operator[](std::size_t d) const { return stencil[d]; }
    };

    /// The kernel of a bond along one axis differentiated with respect to the box length of that
    /// axis: a stencil, or the folded matrix on a narrow absorbing axis. A convolution with it takes
    /// the kernels of the bond along the other axes.
    struct Derivative
    {
        int axis = 0;
        Kernel stencil;
        Folded folded;
    };

    /// Source and sign of the points -halo..n-1+halo of each axis
    struct Ghosts
    {
        std::array<int, 3> halo{};
        std::array<std::vector<int>, 3> index;
        std::array<std::vector<double>, 3> sign;
    };

    /// Kernels of a bond with the given variance along each axis
    Kernels gaussian(double variance, const std::vector<double>& dx,
                     const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc);

    /// For each axis a of the box, the derivative of the bond's kernel along a with respect to the
    /// box length L_a (the grid points move with the box; the variance stays fixed)
    std::vector<Derivative> length_derivatives(double variance, const std::vector<double>& dx,
                                               const std::vector<double>& lx, const std::vector<int>& nx,
                                               const std::vector<BoundaryCondition>& bc);

    /// Tables for halos as wide as the longest kernel of each axis
    Ghosts ghosts(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                  const std::vector<const Kernels*>& kernels);

    /// The bonds of every monomer type in one box
    struct Bonds
    {
        /// Kernels of the full bonds, per ds index and monomer type
        std::map<int, std::map<std::string, Kernels>> full;
        /// Kernels of the half bonds at chain ends and junctions (global ds), per monomer type
        std::map<std::string, Kernels> half;
        /// For the stress: the kernels of the full and half bonds (global ds) differentiated with
        /// respect to the box length, per monomer type and axis
        std::map<std::string, std::vector<Derivative>> full_dlength, half_dlength;
        /// Source and sign of the points beyond each end of each axis, for the longest kernels
        Ghosts ghosts;
    };

    /// The bonds of the molecules in an orthogonal box: a bond of contour ds has the variance
    /// b^2 ds / 3 along each axis, a half bond half of it
    template <typename T>
    Bonds bonds(const Molecules& molecules, const ComputationBox<T>& box);
}

#endif
