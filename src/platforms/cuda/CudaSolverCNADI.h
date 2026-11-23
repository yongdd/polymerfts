/**
 * @file CudaSolverCNADI.h
 * @brief GPU solver using CN-ADI finite difference.
 *
 * This header provides CudaSolverCNADI, the CUDA implementation
 * of the CN-ADI (Crank-Nicolson Alternating Direction Implicit) finite
 * difference method for propagator computation. Uses parallel tridiagonal
 * solves on GPU.
 *
 * **CN-ADI Method:**
 *
 * Implicit time-stepping with 2nd-order accuracy (CN-ADI2):
 *     (I - ds/2 L) q(n+1) = (I + ds/2 L) q(n)
 *
 * **ADI Splitting:**
 *
 * Multi-dimensional problems split into 1D tridiagonal systems, one sweep per axis
 * (x, then y, then z). Each sweep after the first corrects the right-hand side
 * with the result of the sweep before it.
 *
 * **GPU Parallelization:**
 *
 * The lines of an axis are solved independently in parallel: one line per
 * thread, or, along the contiguous axis, a warp of lines staged in shared memory.
 *
 * **Boundary Conditions:**
 *
 * Supports non-periodic boundaries (reflecting, absorbing)
 * which are not available in pseudo-spectral methods.
 *
 * @see CudaSolver for the abstract interface
 * @see CpuSolverCNADI for CPU version
 * @see FiniteDifference for coefficient computation
 */

#ifndef CUDA_SOLVER_CNADI_H_
#define CUDA_SOLVER_CNADI_H_

#include <array>
#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "SpaceGroup.h"

/**
 * @brief Thomas factors of the constant tridiagonal matrix of a line, computed once on the host.
 *
 * table holds four arrays of M values: the lower diagonal, the inverse pivots, the
 * modified upper diagonal c* and, for a periodic line, the Sherman-Morrison vector q.
 * A periodic line folds its corners into the first and last pivots; v and denom are the
 * scalars of the correction x -= q (x_0 + v x_{M-1}) / denom.
 */
struct TridiagonalFactors
{
    std::vector<double> table;
    double v = 0.0;
    double denom = 1.0;
};

/// Factors of the matrix with lower diagonal l, diagonal d and upper diagonal h (M >= 2)
TridiagonalFactors factorize_tridiagonal(const double* l, const double* d, const double* h, int M, bool periodic);

/**
 * @class CudaSolverCNADI
 * @brief GPU solver using CN-ADI (Crank-Nicolson ADI) finite difference.
 *
 * Implements propagator advancement using ADI (Alternating Direction Implicit)
 * method with parallel tridiagonal solves on GPU. Supports non-periodic
 * boundary conditions.
 *
 * Each axis has one matrix per ds and monomer type (AxisMatrix): its diagonals, which
 * the explicit half of a step reads, and its Thomas factors, computed once per box and
 * ds, which the implicit sweep along the axis reads. The lines of an axis are found
 * from its stride in the row-major grid.
 *
 * **Workspace Arrays:**
 *
 * Per-stream workspace for tridiagonal solves:
 * - d_temp: Right-hand side of a sweep
 * - d_q_star: Result of a sweep before the last (2D and 3D)
 *
 * @note Only supports double type (not complex) as CN-ADI method
 *       is primarily for continuous chains with non-periodic BCs.
 */
class CudaSolverCNADI : public CudaSolver<double>
{
private:
    int dim;                          ///< Dimensionality (1, 2, or 3)
    std::array<int, 3> n_axis_{};     ///< Points on each axis
    std::array<int, 3> stride_{};     ///< Distance in memory of neighbouring points of each axis
    std::array<bool, 3> periodic_{};  ///< Periodic axes (ComputationBox pairs the two ends)

    /// The tridiagonal matrix of one axis for one ds and monomer type (device)
    struct AxisMatrix
    {
        CudaDeviceBuffer<double> l;        ///< Lower diagonal
        CudaDeviceBuffer<double> d;        ///< Main diagonal
        CudaDeviceBuffer<double> h;        ///< Upper diagonal
        CudaDeviceBuffer<double> factors;  ///< Thomas factors (TridiagonalFactors::table)
        double v = 0.0;                    ///< Periodic: Sherman-Morrison scalars (TridiagonalFactors)
        double denom = 1.0;
    };
    /// The matrices of the axes: [ds_index][monomer_type][axis]
    std::map<int, std::map<std::string, std::array<AxisMatrix, 3>>> matrices_;

    /// @name Tridiagonal Solver Workspace (per stream)
    /// @{
    CudaDeviceBuffer<double> d_temp[MAX_STREAMS];    ///< Right-hand side of a sweep
    CudaDeviceBuffer<double> d_q_star[MAX_STREAMS];  ///< Result of a sweep before the last (2D, 3D)
    /// @}

    int max_lines_shmem_;   ///< Dynamic shared memory allowed for tridiagonal_lines_shared()

    /**
     * @brief Solve the lines of an axis on the stream: x = A^-1 rhs with the matrix of the axis.
     *
     * Contiguous lines go through tridiagonal_lines_shared(), the others, and lines too long
     * for the shared memory, through tridiagonal_lines().
     */
    void solve_lines(const int STREAM, const AxisMatrix& matrix, int axis, const double *d_rhs, double *d_x);

public:
    /**
     * @brief Construct GPU CN-ADI solver.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param space_group         Space group of the reduced basis, or nullptr: input and
     *                            output are then in its basis, expanded and reduced around
     *                            the finite-difference steps
     */
    CudaSolverCNADI(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS],
        SpaceGroup* space_group);

    /// Update finite difference coefficients for new box.
    void update_laplacian_operator() override;

    /// Update Boltzmann factors from potential fields.
    void update_dw(const std::map<std::string, const double*>& d_w) override;

    /**
     * @brief Advance propagator by one contour step.
     *
     * Uses ADI method: X-sweep → Y-sweep → Z-sweep (for 3D).
     *
     * @param STREAM       Stream index
     * @param d_q_in       Input propagator (device)
     * @param d_q_out      Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask     Optional mask (device)
     * @param ds_index     Index for ds value (0-based) from ContourLengthMapping
     */
    void advance_propagator(
        const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /// Half-bond step (not applicable for CN-ADI continuous).
    void advance_propagator_half_bond_step(
        const int, double *, double *, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     *
     * @param STREAM            Stream index
     * @param d_q_1             Forward propagator (device)
     * @param d_q_2             Backward propagator (device)
     * @param d_segment_stress  Output stress (device)
     * @param monomer_type      Monomer type
     * @param is_half_bond_length Ignored for CN-ADI
     */
    void compute_single_segment_stress(
        const int STREAM,
        double *d_q_1, double *d_q_2, double *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
