/**
 * @file ComputationBox.h
 * @brief Defines the computational domain (grid and box) for polymer simulations.
 *
 * This header provides the ComputationBox class which manages the discrete spatial
 * grid used in polymer field theory simulations. It defines grid dimensions, box
 * lengths, boundary conditions, and provides methods for computing integrals and
 * inner products over the simulation domain.
 *
 * The class supports 1D, 2D, and 3D simulations with various boundary conditions:
 * - Periodic: Standard periodic boundary conditions (for pseudo-spectral method)
 * - Reflecting: Zero-flux Neumann boundary conditions (for real-space method)
 * - Absorbing: Zero-value Dirichlet boundary conditions (for real-space method)
 *
 * **Crystal System Support:**
 *
 * Supports all crystal systems through lattice angles:
 * - Cubic/Tetragonal/Orthorhombic: α = β = γ = 90°
 * - Monoclinic: α = γ = 90°, β ≠ 90°
 * - Hexagonal/Trigonal: α = β = 90°, γ = 120°
 * - Triclinic: Arbitrary angles
 *
 * For non-orthogonal systems, lattice vectors and reciprocal metric tensor
 * are computed for use in Fourier-space wavenumber calculations.
 *
 * @note All lengths are in units of a_Ref * N_Ref^(1/2), where a_Ref is the
 *       reference statistical segment length and N_Ref is the reference
 *       polymerization index.
 *
 * @see CudaComputationBox for the device versions of the CUDA platform
 *
 * @example
 * @code
 * // Create a 3D computation box with 32^3 grid points
 * std::vector<int> nx = {32, 32, 32};
 * std::vector<double> lx = {4.0, 4.0, 4.0};
 * std::vector<std::string> bc = {"periodic", "periodic", "periodic",
 *                                 "periodic", "periodic", "periodic"};
 *
 * std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box(nx, lx, bc);
 *
 * // Compute volume integral of a field
 * double integral_phi = cb->integral(phi);
 *
 * // Compute inner product <g|h> = integral(g * h)
 * double ip = cb->inner_product(g, h);
 *
 * // Create hexagonal box (γ = 120°)
 * std::vector<double> angles = {90.0, 90.0, 120.0};
 * std::unique_ptr<ComputationBox<double>> hex_cb = factory->create_computation_box(nx, lx, bc, angles);
 * @endcode
 */
#ifndef COMPUTATION_BOX_H_
#define COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>
#include <cmath>
#include <complex>
#include <numbers>

#include "Exception.h"

// Forward declaration
class SpaceGroup;

/// An angle fixed at 90 degrees (or at another value a space group fixes) holds when it lies
/// within this many radians of it: the box is then orthogonal (is_orthogonal()), a CrysFFT
/// engine that needs the angle applies, and SpaceGroup::check_cell accepts the cell.
constexpr double FIXED_ANGLE_TOLERANCE = 1e-10;

/// Whether an angle given in degrees lies within FIXED_ANGLE_TOLERANCE of `radians`
/// (false if it is not finite), the one test of a fixed angle
inline bool angle_is_fixed_at(double degrees, double radians)
{
    return std::abs(degrees * (std::numbers::pi / 180.0) - radians) < FIXED_ANGLE_TOLERANCE;
}

/**
 * @enum BoundaryCondition
 * @brief Types of boundary conditions supported by the simulation.
 *
 * One condition per face; a periodic face needs a periodic opposite face, and walls
 * need an orthogonal cell. The pseudo-spectral methods take the same condition on
 * both faces of a direction and every direction periodic (FFT) or none (DCT/DST:
 * reflecting and absorbing directions may coexist); the real-space methods
 * (cn-adi2, rs) also mix periodic and non-periodic directions.
 */
enum class BoundaryCondition
{
    PERIODIC,    ///< Periodic BC: f(0) = f(L), f'(0) = f'(L). Used with FFT-based solvers.
    REFLECTING,  ///< Reflecting (Neumann) BC: df/dn = 0 at boundary. Models impenetrable walls.
    ABSORBING,   ///< Absorbing (Dirichlet) BC: f = 0 at boundary. Models chain-absorbing surfaces.
};

/**
 * @class ComputationBox
 * @brief Manages the computational domain grid and provides integration methods.
 *
 * This template class defines the spatial discretization for polymer field theory
 * simulations. It stores grid dimensions (nx), box lengths (lx), and boundary
 * conditions. The class provides methods for computing volume integrals and
 * inner products, which are fundamental operations in SCFT and L-FTS calculations.
 *
 * @tparam T Numeric type for field values (double or std::complex<double>)
 *
 * **Grid Layout:**
 *
 * For a 3D box with dimensions nx = [Nx, Ny, Nz]:
 * - Total grid points: M = Nx * Ny * Nz
 * - Grid spacing: dx[i] = lx[i] / nx[i]
 * - Volume element: dV = dx[0] * dx[1] * dx[2] (V/M in an oblique cell)
 *
 * **Memory Layout:**
 *
 * Fields are stored as 1D arrays in row-major order (C-style):
 * index = iz + Nz * (iy + Ny * ix)
 *
 * **Units:**
 *
 * - Length unit: a_Ref * N_Ref^(1/2)
 * - Volume unit: (a_Ref * N_Ref^(1/2))^dim
 *
 * @note The CPU platforms use this class directly; CudaComputationBox adds the
 *       device versions of its operations.
 */
template <typename T>
class ComputationBox
{
protected:
    int dim;                      ///< Spatial dimension (1, 2, or 3)
    std::vector<int> nx;          ///< Number of grid points in each direction [Nx, Ny, Nz]
    std::vector<double> lx;       ///< Box lengths in each direction [Lx, Ly, Lz] (units: a_Ref*N^0.5)
    std::vector<double> dx;       ///< Grid spacing in each direction [dx, dy, dz] = lx/nx
    int total_grid;               ///< Total number of grid points M = Nx*Ny*Nz
    std::vector<double> mask;     ///< The mask in the reduced basis of a space group (empty otherwise; get_mask() then returns mask_full_)
    std::vector<double> mask_full_; ///< Mask for impenetrable regions (0=blocked, 1=accessible) on the full grid, as given; empty if no mask. dv and the reduced mask start from it
    double dv_base_;              ///< Volume element of a grid point: Π dx[d] in an orthogonal cell, V/M in an oblique one
    std::vector<double> dv;       ///< Volume element weights for integration (dv_base_ times the mask)
    double volume;                ///< Total system volume V = a·(b×c) for general lattice
    double accessible_volume;     ///< Accessible volume (excluding masked regions)

    /**
     * @brief Lattice angles [alpha, beta, gamma] in radians.
     *
     * Crystal system convention:
     * - alpha: angle between b and c lattice vectors
     * - beta:  angle between a and c lattice vectors
     * - gamma: angle between a and b lattice vectors
     *
     * Default: [π/2, π/2, π/2] for orthogonal systems.
     */
    std::vector<double> angles_;

    /**
     * @brief 3x3 lattice vector matrix in column-major order.
     *
     * Stores [a|b|c] where a, b, c are lattice vectors.
     * Layout: [a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z]
     *
     * Standard parameterization:
     * - a = (lx[0], 0, 0)
     * - b = (lx[1]*cos(γ), lx[1]*sin(γ), 0)
     * - c = (c_x, c_y, c_z) computed from angles
     */
    std::array<double, 9> lattice_vec_;

    /**
     * @brief Symmetric reciprocal metric tensor (6 components).
     *
     * G*_ij = e*_i · e*_j where e* are reciprocal basis vectors.
     * Layout: [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
     *
     * Used in wavenumber calculation: |k|² = G*_ij k_i k_j
     */
    std::array<double, 6> recip_metric_;

    /**
     * @brief Whether the lattice is orthogonal (all angles = 90°).
     *
     * Orthogonal systems can use simplified wavenumber calculation:
     * |k|² = kx² + ky² + kz² (no cross terms)
     */
    bool is_orthogonal_;

    /// Whether alpha and beta are 90° (within FIXED_ANGLE_TOLERANCE): the c axis is
    /// perpendicular to the a-b plane, whatever gamma is
    bool is_z_axis_orthogonal_;

    /**
     * @brief Boundary conditions for each face.
     *
     * Order: [x_low, x_high, y_low, y_high, z_low, z_high]
     * - For 1D: only indices 0,1 are used
     * - For 2D: indices 0-3 are used
     * - For 3D: all 6 indices are used
     */
    std::vector<BoundaryCondition> bc;

    // ==================== Space Group Support ====================

    /**
     * @brief The space group of the reduced basis, fixed when the box is made (nullptr: none).
     *
     * With a group, integral(), inner_product(), etc. operate on reduced basis
     * arrays of size n_basis instead of full grid arrays.
     */
    SpaceGroup* space_group_;

    /**
     * @brief Cached orbit counts from space group.
     *
     * orbit_counts[i] is the multiplicity of the i-th reduced-basis point.
     */
    std::vector<int> orbit_counts_;

    /**
     * @brief Integration weights of the reduced-basis points.
     *
     * basis_weights_[i] = orbit_counts_[i] * mask[i] (the orbit counts without a
     * mask), so that the reduced-basis integrals equal the full-grid ones, whose
     * dv carries the mask.
     */
    std::vector<double> basis_weights_;

    /**
     * @brief Number of reduced-basis points.
     *
     * When space_group_ is nullptr, this equals total_grid.
     */
    int n_basis_;

    /// Computations of this box whose numerical method requires an orthogonal cell
    int orthogonal_cell_holders_ = 0;

private:
    /**
     * @brief Grid spacing, lattice and reciprocal vectors, metric tensor, volume
     *        elements and accessible volume of the current lx and angles.
     *
     * The cell geometry is cell_geometry() (ReciprocalMetric.h). The volume element
     * dv_base_ is Π dx[d] in an orthogonal cell and V/M in an oblique one; dv is
     * dv_base_ times the mask. The constructor and every cell change take this one
     * update, so a box moved to a cell equals a box created there, bit for bit.
     */
    void update_cell();

    /**
     * @brief Take the space group of the box: check it against the box (real fields,
     *        walls, grid, cell, mask symmetry) and build its reduced basis (basis size,
     *        orbit counts, reduced mask and integration weights).
     *
     * The sizes and maps are snapshots of the group, so its physical basis is
     * frozen from now on (SpaceGroup::mark_in_use()).
     */
    void use_space_group(SpaceGroup* sg);

public:
    /**
     * @brief Keep the cell orthogonal while a computation needs it.
     *
     * A computation whose numerical method requires an orthogonal cell
     * (cn-adi2, rs) holds it from the end of its construction until it is
     * destroyed, and so does a Fourier transform with wall axes. While it is
     * held, set_lattice_parameters() rejects other angles before changing the box.
     */
    void hold_orthogonal_cell() { orthogonal_cell_holders_++; }
    void release_orthogonal_cell() noexcept { orthogonal_cell_holders_--; }

    /**
     * @brief Construct a ComputationBox with specified grid and box parameters.
     *
     * @param nx   Number of grid points in each direction, e.g., {32, 32, 32}
     * @param lx   Box lengths in each direction, e.g., {4.0, 4.0, 4.0}
     * @param bc   Boundary condition strings: {"periodic", "reflecting", "absorbing"}
     *             Order: [x_low, x_high, y_low, y_high, z_low, z_high]
     * @param mask Optional mask array defining impenetrable regions.
     *             mask[i]=0 means grid point i is inside an obstacle.
     *             mask[i]=1 means grid point i is accessible.
     *             If nullptr, all points are accessible.
     *
     * @throws Exception if nx and lx have different sizes or invalid BC strings
     *
     * @example
     * @code
     * // 64x64x64 periodic box
     * std::vector<int> nx = {64, 64, 64};
     * std::vector<double> lx = {8.0, 8.0, 8.0};
     * std::vector<std::string> bc = {"periodic", "periodic", "periodic",
     *                                 "periodic", "periodic", "periodic"};
     * ComputationBox<double> cb(nx, lx, bc);
     * @endcode
     */
    ComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);

    /**
     * @brief Construct a ComputationBox with lattice angles for non-orthogonal systems.
     *
     * @param nx     Number of grid points in each direction, e.g., {32, 32, 32}
     * @param lx     Box lengths (lattice constants) in each direction, e.g., {4.0, 4.0, 4.0}
     * @param bc     Boundary condition strings
     * @param angles Lattice angles [alpha, beta, gamma] in DEGREES
     *               - alpha: angle between b and c vectors
     *               - beta:  angle between a and c vectors
     *               - gamma: angle between a and b vectors
     * @param mask   Optional mask array for impenetrable regions
     * @param space_group Optional space group (real fields only): the fields of the box
     *               are in its reduced basis for the whole life of the box. It must
     *               outlive the box, fit its grid, cell, boundary conditions and mask,
     *               and its physical basis is frozen (SpaceGroup::mark_in_use())
     *
     * @throws Exception if angles are invalid (must be > 0 and < 180 and form a cell of
     *         volume at least 0.1 abc), or if the space group does not fit the box
     *
     * @example
     * @code
     * // Hexagonal box (gamma = 120°)
     * std::vector<int> nx = {32, 32, 32};
     * std::vector<double> lx = {4.0, 4.0, 6.0};
     * std::vector<std::string> bc = {"periodic", "periodic", "periodic",
     *                                 "periodic", "periodic", "periodic"};
     * std::vector<double> angles = {90.0, 90.0, 120.0};
     * ComputationBox<double> cb(nx, lx, bc, angles);
     * @endcode
     */
    ComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
                   std::vector<double> angles, const double* mask=nullptr, SpaceGroup* space_group=nullptr);

    /// Virtual destructor.
    virtual ~ComputationBox();

    /**
     * @brief Get spatial dimension.
     * @return Dimension (1, 2, or 3)
     */
    int get_dim() const;

    /**
     * @brief Get grid dimensions vector.
     * @return Vector of grid points [Nx, Ny, Nz]
     */
    std::vector<int> get_nx() const;

    /**
     * @brief Get grid dimension in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Number of grid points in direction i
     */
    int get_nx(int i) const;

    /**
     * @brief Get box lengths vector.
     * @return Vector of box lengths [Lx, Ly, Lz] in units of a_Ref*N^0.5
     */
    std::vector<double> get_lx() const;

    /**
     * @brief Get box length in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Box length in direction i
     */
    double get_lx(int i) const;

    /**
     * @brief Get grid spacing vector.
     * @return Vector of grid spacings [dx, dy, dz] = lx/nx
     */
    std::vector<double> get_dx() const;

    /**
     * @brief Get grid spacing in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Grid spacing in direction i
     */
    double get_dx(int i) const;

    /**
     * @brief Get volume element weight at grid point.
     * @param i Grid point index
     * @return Volume element dV[i] (includes mask weighting)
     */
    double get_dv(int i) const;

    /**
     * @brief Get the volume element of a grid point without the mask.
     * @return Π dx[d] in an orthogonal cell, V/M in an oblique one: the weight of
     *         every integral (get_dv() is it times the mask)
     */
    double get_dv_base() const;

    /**
     * @brief Get total number of grid points.
     * @return M = Nx*Ny*Nz
     */
    int get_total_grid() const;

    /**
     * @brief Get total system volume.
     * @return V = Lx*Ly*Lz in units of (a_Ref*N^0.5)^dim
     */
    double get_volume() const;

    /**
     * @brief Get accessible volume (excluding masked regions).
     * @return Accessible volume = sum of dV over unmasked points
     */
    double get_accessible_volume() const;

    /**
     * @brief Get pointer to mask array.
     * @return Const pointer to mask array (nullptr if no mask)
     */
    const double* get_mask() const;

    /**
     * @brief Get all boundary conditions.
     * @return Vector of BoundaryCondition enums
     */
    const std::vector<BoundaryCondition> get_boundary_conditions() const;

    /**
     * @brief Get boundary condition for specific face.
     * @param i Face index (0=x_low, 1=x_high, 2=y_low, ...)
     * @return BoundaryCondition enum value
     */
    BoundaryCondition get_boundary_condition(int i) const;

    /**
     * @brief Get lattice angles in radians.
     * @return Vector [alpha, beta, gamma] in radians
     */
    std::vector<double> get_angles() const;

    /**
     * @brief Get lattice angles in degrees.
     * @return Vector [alpha, beta, gamma] in degrees
     */
    std::vector<double> get_angles_degrees() const;

    /**
     * @brief Get lattice vectors (3x3 matrix).
     * @return Array [a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z] (column-major)
     */
    const std::array<double, 9>& get_lattice_vec() const;

    /**
     * @brief Get reciprocal metric tensor.
     * @return Array [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22] (symmetric)
     */
    const std::array<double, 6>& get_recip_metric() const;

    /**
     * @brief Check if the lattice is orthogonal.
     * @return True if all angles are 90° (within FIXED_ANGLE_TOLERANCE), false otherwise
     */
    bool is_orthogonal() const;

    /**
     * @brief Check if the c axis is perpendicular to the a-b plane.
     * @return True if alpha and beta are 90° (within FIXED_ANGLE_TOLERANCE); gamma is free
     */
    bool is_z_axis_orthogonal() const;

    // ==================== Space Group Methods ====================

    /**
     * @brief Get the space group of the box.
     *
     * With a space group, integral(), inner_product(), multi_inner_product(),
     * and zero_mean() operate on reduced basis arrays (size n_basis) instead
     * of full grid arrays (size total_grid).
     *
     * @return Pointer to SpaceGroup, or nullptr if the box has none
     */
    SpaceGroup* get_space_group() const;

    /**
     * @brief Get the number of grid points for field operations.
     *
     * Returns n_basis if space group is set, otherwise total_grid.
     * Use this to determine the expected array size for integral(), inner_product(), etc.
     *
     * @return Number of grid points (n_basis or total_grid)
     */
    int get_n_basis() const;

    /**
     * @brief Get the orbit counts for reduced basis weighting.
     * @return Const reference to orbit_counts vector (empty if no space group)
     */
    const std::vector<int>& get_orbit_counts() const;

    // ==================== Field Operations ====================

    /**
     * @brief Compute volume integral of a field.
     *
     * When space group is NOT set:
     *   Computes: integral(g) = sum_i g[i] * dV[i]
     *   Array size: total_grid
     *
     * When space group IS set (reduced basis mode):
     *   Computes: integral(g) = dv_base * sum_i basis_weights[i] * g[i]
     *   (basis_weights = orbit counts times the mask)
     *   Array size: n_basis
     *
     * @param g Field array (length: get_n_basis())
     * @return Volume integral of g
     */
    T integral(const T *g);

    /**
     * @brief Compute mean (spatial average) of a field.
     *
     * Computes: mean(g) = integral(g) / volume
     *
     * @param g Field array (length: get_n_basis())
     * @return Spatial average of g
     */
    T mean(const T *g);

    /**
     * @brief Compute inner product of two fields.
     *
     * When space group is NOT set:
     *   Computes: <g|h> = sum_i g[i] * h[i] * dV[i]
     *   Array size: total_grid
     *
     * When space group IS set (reduced basis mode):
     *   Computes: <g|h> = dv_base * sum_i basis_weights[i] * g[i] * h[i]
     *   Array size: n_basis
     *
     * @param g First field array (length: get_n_basis())
     * @param h Second field array (length: get_n_basis())
     * @return Inner product <g|h>
     */
    T inner_product(const T *g, const T *h);

    /**
     * @brief Compute weighted inner product with inverse weight.
     *
     * Computes: sum_i g[i] * h[i] / w[i] * dV[i]
     *
     * When space group is set, operates on reduced basis arrays.
     *
     * @param g First field array (length: get_n_basis())
     * @param h Second field array (length: get_n_basis())
     * @param w Weight field array (must be non-zero everywhere)
     * @return Weighted inner product
     */
    T inner_product_inverse_weight(const T *g, const T *h, const T *w);

    /**
     * @brief Integral of an absolute product.
     *
     * Computes: sum_i |g[i] * h[i] / d[i]| * dV[i] with the weights of integral()
     * (no division without a divisor); a masked point adds zero whatever its
     * values. The scale of a partition integral that need not vanish when a
     * complex partition cancels.
     *
     * @param g       First field array (length: get_n_basis())
     * @param h       Second field array (length: get_n_basis())
     * @param divisor Divisor array, or nullptr
     * @return Integral value
     */
    double absolute_inner_product(const T *g, const T *h, const T *divisor = nullptr);

    /**
     * @brief Compute inner product for multiple field components.
     *
     * When space group is NOT set:
     *   Array size per component: total_grid
     *
     * When space group IS set:
     *   Array size per component: n_basis
     *
     * @param n_comp Number of field components
     * @param g First multi-component field (length: n_comp * get_n_basis())
     * @param h Second multi-component field (length: n_comp * get_n_basis())
     * @return Sum of inner products over all components
     */
    T multi_inner_product(int n_comp, const T *g, const T *h);

    /**
     * @brief Remove the spatial mean from a field (make it zero-mean).
     *
     * Modifies g in-place: g[i] -= integral(g) / accessible volume, so that
     * integral(g) and mean(g) vanish afterwards. Without a mask this is
     * g[i] -= mean(g); with one, masked points shift with the rest.
     *
     * When space group is set, uses the mean weighted with the basis weights
     * (orbit counts times the mask).
     *
     * @param g Field array (length: get_n_basis(), modified in-place)
     */
    void zero_mean(T *g);

    /**
     * @brief Update box lengths only (for box relaxation in SCFT).
     *
     * Updates lx, dx, dv, volume, and recomputes lattice/reciprocal vectors.
     * Used when optimizing box dimensions to minimize free energy.
     *
     * @param new_lx New box lengths [Lx, Ly, Lz]
     *
     * @note After calling set_lx(), you should also update the Laplacian
     *       operator in the propagator solver.
     */
    virtual void set_lx(std::vector<double> new_lx);

    /**
     * @brief Update lattice parameters - box lengths and angles (for box relaxation in SCFT).
     *
     * Updates lx, angles, dx, dv, volume, lattice vectors, reciprocal
     * vectors, and metric tensor. Used when optimizing both box dimensions
     * and angles to minimize free energy.
     *
     * @param new_lx     New box lengths [Lx, Ly, Lz]
     * @param new_angles New lattice angles [alpha, beta, gamma] in DEGREES
     *
     * @note After calling set_lattice_parameters(), you should also update the Laplacian
     *       operator in the propagator solver.
     *
     * @throws Exception before changing the box for invalid angles (including a
     *         cell of volume below 0.1 abc), for a
     *         non-orthogonal cell with walls or while hold_orthogonal_cell()
     *         is in effect, and for a cell that does not match the space group
     *         (whose fixed angles pass the orthogonality test of this box)
     */
    virtual void set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles);
};
#endif
