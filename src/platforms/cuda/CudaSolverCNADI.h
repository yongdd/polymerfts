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
 * Multi-dimensional problems split into 1D tridiagonal systems:
 * - 2D: X-sweep → Y-sweep
 * - 3D: X-sweep → Y-sweep → Z-sweep
 *
 * **GPU Parallelization:**
 *
 * Each tridiagonal system solved independently in parallel:
 * - 2D: ny systems for X-sweep, nx systems for Y-sweep
 * - 3D: ny×nz, nx×nz, nx×ny systems respectively
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

/// Device function: max of two integers.
__device__ int d_max_of_two(int x, int y);

/// Device function: min of two integers.
__device__ int d_min_of_two(int x, int y);

/**
 * @brief CUDA kernel: 1D Crank-Nicolson RHS computation.
 *
 * Computes the right-hand side (I + ds/2 Lx) q for 1D problems.
 *
 * @param bc_xl  Boundary condition at x-low
 * @param bc_xh  Boundary condition at x-high
 * @param d_xl   Lower diagonal coefficients
 * @param d_xd   Main diagonal coefficients
 * @param d_xh   Upper diagonal coefficients
 * @param d_q_out Output RHS vector
 * @param d_q_in  Input propagator
 * @param M       Grid size
 */
__global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 2D ADI step 1 (X-direction).
 *
 * Computes RHS for X-sweep: (I + ds/2 Lx) q
 *
 * @param bc_xl, bc_xh  X-direction boundary conditions
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_xl, d_xd, d_xh  X-direction tridiagonal coefficients
 * @param I             X grid size
 * @param d_yl, d_yd, d_yh  Y-direction tridiagonal coefficients
 * @param J             Y grid size
 * @param d_q_out       Output RHS
 * @param d_q_in        Input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 2D ADI step 2 (Y-direction).
 *
 * Computes RHS for Y-sweep using intermediate solution q*.
 *
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_yl, d_yd, d_yh  Y-direction tridiagonal coefficients
 * @param J             Y grid size
 * @param d_q_out       Output RHS
 * @param d_q_star      Intermediate solution from step 1
 * @param d_q_in        Original input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 1 (X-direction).
 *
 * First ADI sweep in X-direction for 3D problems.
 *
 * @param bc_xl, bc_xh  X-direction boundary conditions
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param bc_zl, bc_zh  Z-direction boundary conditions
 * @param d_xl, d_xd, d_xh  X-direction coefficients
 * @param I             X grid size
 * @param d_yl, d_yd, d_yh  Y-direction coefficients
 * @param J             Y grid size
 * @param d_zl, d_zd, d_zh  Z-direction coefficients
 * @param K             Z grid size
 * @param d_q_out       Output RHS
 * @param d_q_in        Input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 2 (Y-direction).
 *
 * Second ADI sweep in Y-direction using intermediate q*.
 *
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_yl, d_yd, d_yh  Y-direction coefficients
 * @param J, K          Y and Z grid sizes
 * @param d_q_out       Output RHS
 * @param d_q_star      Intermediate from step 1
 * @param d_q_in        Original input
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 3 (Z-direction).
 *
 * Final ADI sweep in Z-direction using intermediate q**.
 *
 * @param bc_zl, bc_zh  Z-direction boundary conditions
 * @param d_zl, d_zd, d_zh  Z-direction coefficients
 * @param J, K          Y and Z grid sizes
 * @param d_q_out       Output RHS
 * @param d_q_dstar     Intermediate from step 2
 * @param d_q_in        Original input
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M);

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
 * @brief CUDA kernel: REPEAT systems of size M, one per thread, starting at d_offset with stride INTERVAL.
 *
 * Solves d_x from the right-hand side d_d (which may be the same array) with the factors in table.
 */
template <bool PERIODIC>
__global__ void tridiagonal_lines(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x,
    const int* __restrict__ d_offset, const int REPEAT, const int INTERVAL);

/**
 * @brief CUDA kernel: REPEAT contiguous systems of size M (system k at k*M), staged in shared memory.
 *
 * A block of blockDim.x threads copies as many systems into shared memory with coalesced reads,
 * each thread solves one there, and the block writes them back. Rows hold an odd number of values
 * so that the threads of a warp use different banks: blockDim.x * (M or M+1) doubles of dynamic
 * shared memory.
 */
template <bool PERIODIC>
__global__ void tridiagonal_lines_shared(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x, const int REPEAT);

/**
 * @class CudaSolverCNADI
 * @brief GPU solver using CN-ADI (Crank-Nicolson ADI) finite difference.
 *
 * Implements propagator advancement using ADI (Alternating Direction Implicit)
 * method with parallel tridiagonal solves on GPU. Supports non-periodic
 * boundary conditions.
 *
 * **Tridiagonal Systems:**
 *
 * For each direction (x, y, z), stores coefficients:
 * - d_xl: Lower diagonal (subdiagonal)
 * - d_xd: Main diagonal
 * - d_xh: Upper diagonal (superdiagonal)
 *
 * **Workspace Arrays:**
 *
 * Per-stream workspace for tridiagonal solves:
 * - d_q_star: Intermediate ADI solution (after first sweep)
 * - d_q_dstar: Second intermediate (3D only)
 *
 * The Thomas factors of each axis (d_factors) are computed once per box and ds.
 *
 * **Offset Arrays:**
 *
 * Index mappings for parallel system solving:
 * - d_offset_xy, d_offset_yz, d_offset_xz: 3D sweeps
 * - d_offset_x, d_offset_y: 2D sweeps
 *
 * @note Only supports double type (not complex) as CN-ADI method
 *       is primarily for continuous chains with non-periodic BCs.
 */
class CudaSolverCNADI : public CudaSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box
    Molecules *molecules;        ///< Molecules container

    int dim;                     ///< Dimensionality (1, 2, or 3)

    int n_streams;               ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name Space group support (reduced basis)
    /// @{
    SpaceGroup* space_group_;           ///< Space group pointer (nullptr if not used)
    int* d_reduced_basis_indices_;      ///< Device array: reduced → full index mapping (lent by the backend)
    int* d_full_to_reduced_map_;        ///< Device array: full → reduced index mapping (lent by the backend)
    int n_basis_;                       ///< Number of reduced basis points
    CudaDeviceBuffer<double> d_q_full_in_[MAX_STREAMS];  ///< Work buffer: full grid input (per stream)
    CudaDeviceBuffer<double> d_q_full_out_[MAX_STREAMS]; ///< Work buffer: full grid output (per stream)
    /// @}

    /// @name Tridiagonal Coefficients per ds_index per monomer_type
    /// Nested map structure: [ds_index][monomer_type] -> coefficient array (device)
    /// @{
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_xl;  ///< X-direction lower diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_xd;  ///< X-direction main diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_xh;  ///< X-direction upper diagonal

    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_yl;  ///< Y-direction lower diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_yd;  ///< Y-direction main diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_yh;  ///< Y-direction upper diagonal

    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_zl;  ///< Z-direction lower diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_zd;  ///< Z-direction main diagonal
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_zh;  ///< Z-direction upper diagonal
    /// @}


    /// @name Tridiagonal Solver Workspace (per stream)
    /// @{
    CudaDeviceBuffer<double> d_q_star[MAX_STREAMS];  ///< First intermediate solution
    CudaDeviceBuffer<double> d_q_dstar[MAX_STREAMS];  ///< Second intermediate (3D)
    CudaDeviceBuffer<double> d_temp[MAX_STREAMS];  ///< General temporary
    /// @}

    /// @name Index Offset Arrays
    /// @{
    CudaDeviceBuffer<int> d_offset_xy;  ///< 3D: XY-plane offsets
    CudaDeviceBuffer<int> d_offset_yz;  ///< 3D: YZ-plane offsets
    CudaDeviceBuffer<int> d_offset_xz;  ///< 3D: XZ-plane offsets
    CudaDeviceBuffer<int> d_offset_x;   ///< 2D: X-direction offsets
    CudaDeviceBuffer<int> d_offset_y;   ///< 2D: Y-direction offsets
    CudaDeviceBuffer<int> d_offset;     ///< 1D: System offsets
    /// @}

    /// Thomas factors of one axis on the device (see TridiagonalFactors)
    struct DeviceFactors
    {
        CudaDeviceBuffer<double> table;
        double v = 0.0;
        double denom = 1.0;
        bool periodic = false;
    };
    /// Factors of the x, y and z lines per ds_index and monomer type
    std::map<int, std::map<std::string, std::array<DeviceFactors, 3>>> d_factors;

    int max_lines_shmem_;   ///< Dynamic shared memory allowed for tridiagonal_lines_shared()

    /**
     * @brief Solve REPEAT systems of size M of one axis on the stream.
     *
     * Contiguous systems (INTERVAL 1) go through tridiagonal_lines_shared(), the others, and
     * systems too long for the shared memory, through tridiagonal_lines().
     */
    void solve_lines(
        const int STREAM, const DeviceFactors& factors, const double *d_d, double *d_x,
        const int *d_offset, const int REPEAT, const int INTERVAL, const int M);


    /**
     * @brief 3D ADI propagator advancement.
     * @param ds_index Index for ds value (0-based) from ContourLengthMapping
     */
    void advance_propagator_3d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

    /**
     * @brief 2D ADI propagator advancement.
     * @param ds_index Index for ds value (0-based) from ContourLengthMapping
     */
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

    /**
     * @brief 1D Crank-Nicolson propagator advancement.
     * @param ds_index Index for ds value (0-based) from ContourLengthMapping
     */
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

public:
    /**
     * @brief Construct GPU CN-ADI solver.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Checkpointing mode
     */
    CudaSolverCNADI(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /// Destructor. Frees GPU resources.
    ~CudaSolverCNADI();

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, input/output are in reduced basis and the solver expands/reduces
     * internally around finite-difference operations.
     */
    void set_space_group(
        SpaceGroup* sg,
        int* d_reduced_basis_indices,
        int* d_full_to_reduced_map,
        int n_basis) override;

    /// Update finite difference coefficients for new box.
    void update_laplacian_operator() override;

    /// Update Boltzmann factors from potential fields.
    void update_dw(std::string device, std::map<std::string, const double*> w_input) override;

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
     * @param d_q_pair          Propagator product (device)
     * @param d_segment_stress  Output stress (device)
     * @param monomer_type      Monomer type
     * @param is_half_bond_length Ignored for CN-ADI
     */
    void compute_single_segment_stress(
        const int STREAM,
        double *d_q_pair, double *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
