/**
 * @file CpuSolverCNADI.h
 * @brief CN-ADI finite difference solver for continuous chains on CPU.
 *
 * This header provides CpuSolverCNADI, which implements the CN-ADI
 * (Crank-Nicolson Alternating Direction Implicit) finite difference method
 * for solving the modified diffusion equation.
 * Unlike pseudo-spectral methods, this supports non-periodic boundary conditions.
 *
 * **Numerical Method:**
 *
 * Uses CN-ADI (implicit) scheme with operator splitting:
 *
 *     (1 + A_x/2)(1 + A_y/2)(1 + A_z/2) q^(n+1) =
 *     (1 - A_x/2)(1 - A_y/2)(1 - A_z/2) q^(n)
 *
 * where A_α = -(b²ds/12) ∂²/∂α² + (w·ds/6).
 *
 * **Accuracy:**
 *
 * This solver implements CN-ADI2 (2nd order accuracy in ds).
 *
 * **ADI (Alternating Direction Implicit):**
 *
 * Each spatial direction is solved implicitly in sequence, resulting in
 * a series of tridiagonal systems that are efficiently solvable.
 *
 * **Supported Boundary Conditions:**
 *
 * - Periodic: q(0) = q(L)
 * - Reflecting (Neumann): ∂q/∂n = 0 at boundary
 * - Absorbing (Dirichlet): q = 0 at boundary
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverPseudoRQM4 for spectral method (periodic only)
 * @see FiniteDifference for coefficient generation
 */

#ifndef CPU_SOLVER_CNADI_H_
#define CPU_SOLVER_CNADI_H_

#include <map>
#include <string>
#include <vector>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"
#include "SpaceGroup.h"

/**
 * @class TridiagonalLines
 * @brief A tridiagonal matrix and the parts of its Thomas solve that depend only on the matrix.
 *
 * Row i reads l[i] x[i-1] + d[i] x[i] + h[i] x[i+1] = r[i]. A periodic (cyclic)
 * matrix also couples x[0] to x[n-1] through l[0] and x[n-1] to x[0] through
 * h[n-1]; it is solved with the Sherman-Morrison correction, whose vector q does
 * not depend on the right-hand side either. The constructor computes the pivots,
 * the upper diagonal over the pivots and, for a cyclic matrix, q and the
 * denominator of the correction, so that a solve is the two sweeps over the
 * right-hand side.
 */
class TridiagonalLines
{
private:
    std::vector<double> l_, d_, h_;   ///< lower, main and upper diagonal
    bool periodic_;
    std::vector<double> den_;         ///< pivots of the forward sweep
    std::vector<double> c_star_;      ///< h[i] over the pivot of row i (n-1 values)
    std::vector<double> q_;           ///< cyclic: solution of the Sherman-Morrison system
    double l0_gamma_ = 0.0;           ///< cyclic: l[0] / gamma
    double q_den_ = 1.0;              ///< cyclic: 1 + q[0] + l[0] / gamma q[n-1]

public:
    /// Factorize the matrix of the diagonals l, d, h (n >= 2 rows), cyclic when periodic.
    TridiagonalLines(std::vector<double> l, std::vector<double> d, std::vector<double> h, bool periodic);

    int size() const { return static_cast<int>(d_.size()); }
    const double* lower() const { return l_.data(); }
    const double* diagonal() const { return d_.data(); }
    const double* upper() const { return h_.data(); }

    /**
     * @brief Solve `count` lines at once.
     *
     * Element i of line c has its right-hand side at rhs[i*count + c] and its
     * solution at x[i*stride + c]; x may be rhs when stride == count.
     * `correction` holds count values (cyclic matrices only).
     */
    void solve(const double* rhs, double* x, long stride, long count, double* correction) const;
};

/**
 * @class CpuSolverCNADI
 * @brief CPU solver using CN-ADI (Crank-Nicolson ADI).
 *
 * Implements the CN-ADI scheme for solving the modified diffusion equation
 * with various boundary conditions. Provides CN-ADI2 (2nd order).
 *
 * **Sweeps:** a step solves one tridiagonal system per grid line along each axis
 * in turn (TridiagonalLines, one per ds value, monomer type and axis). The lines
 * of an axis are solved together, plane by plane: the right-hand side of a plane
 * spanned by the axis and the contiguous (last) axis is formed row by row, and
 * then all its lines are swept at once with the contiguous axis innermost. The
 * lines along the contiguous axis itself are solved in planes spanned by it and
 * the axis before it, stored transposed. The neighbour of every grid position
 * along every axis is tabulated with the boundary conditions applied.
 *
 * @note Stress calculation is not implemented for CN-ADI.
 *
 * @example
 * @code
 * CpuSolverCNADI solver(cb, molecules);
 * solver.update_dw(w_fields);
 *
 * // Advance with reflecting boundaries
 * solver.advance_propagator(q_in, q_out, "A", nullptr, 0);
 * @endcode
 */
class CpuSolverCNADI : public CpuSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box for grid/boundary info
    Molecules *molecules;         ///< Molecules container
    SpaceGroup* space_group_;     ///< Space group pointer (nullptr if not used)

    int dim_;                                    ///< Number of axes
    std::vector<int> nx_;                        ///< Grid points per axis
    std::vector<std::vector<int>> lower_;        ///< Neighbour below each position of each axis (boundary conditions applied)
    std::vector<std::vector<int>> upper_;        ///< Neighbour above each position of each axis

    /// Tridiagonal matrices of the axes: [ds_index][monomer_type][axis]
    std::map<int, std::map<std::string, std::vector<TridiagonalLines>>> matrices_;

    /// Work arrays of one calling thread (PerThread)
    struct WorkArrays
    {
        std::vector<double> q_full;          ///< full-grid propagator (space group)
        std::vector<double> q_exp;           ///< exp(-w ds/2) q_in
        std::vector<double> q_star, q_dstar; ///< results of the sweeps before the last
        std::vector<double> plane;           ///< right-hand sides of the lines of one plane
        std::vector<double> correction;      ///< cyclic matrices: correction factor of each line of a plane
    };
    PerThread<WorkArrays> work_arrays_;   ///< Work arrays of each thread

    /// Work arrays of the calling thread, sized for the grid
    WorkArrays& work_arrays() const;

    /// One ADI step of the 1D, 2D or 3D grid: q_out = exp_dw · ADI(w.q_exp), with w.q_exp = exp_dw · q_in.
    void step_1d(const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& w) const;
    void step_2d(const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& w) const;
    void step_3d(const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& w) const;

public:
    /**
     * @brief Construct CN-ADI solver.
     *
     * Builds the neighbour tables and the tridiagonal matrices of every axis
     * for each ds value and monomer type.
     *
     * @param cb            Computation box with boundary conditions
     * @param molecules     Molecules container with monomer types
     */
    CpuSolverCNADI(ComputationBox<double>* cb, Molecules *molecules);

    ~CpuSolverCNADI() override = default;

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, q_in/q_out are in reduced basis and the solver expands/reduces
     * internally around finite-difference operations.
     *
     * @param sg Space group pointer (nullptr to disable)
     */
    void set_space_group(SpaceGroup* sg) override { space_group_ = sg; }

    /**
     * @brief Update finite difference coefficients.
     *
     * Recomputes the tridiagonal matrices and their factors when box dimensions change.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update the Boltzmann factors exp(-w ds/2) from potential fields.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const double*> w_input) override;

    /**
     * @brief Advance propagator by one contour step.
     *
     * Uses ADI scheme appropriate for the grid dimensionality.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator (may be q_in)
     * @param monomer_type Monomer type
     * @param q_mask      Optional mask, in the basis of the propagators
     * @param ds_index    Index for the ds value (0-based) from ContourLengthMapping
     */
    void advance_propagator(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask, int ds_index) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     *
     * Empty implementation - continuous chains don't use half-bond steps.
     */
    void advance_propagator_half_bond_step(double *, double *, std::string) override {};

    /**
     * @brief Compute stress from one segment.
     *
     * @warning Not implemented for CN-ADI: throws.
     */
    std::vector<double> compute_single_segment_stress(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
