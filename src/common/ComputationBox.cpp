/**
 * @file ComputationBox.cpp
 * @brief Implementation of ComputationBox simulation domain.
 *
 * Provides the implementation for managing the computational
 * grid, including grid dimensions, volume elements, boundary conditions,
 * and integral operations, on host memory (the CPU platforms use it directly;
 * CudaComputationBox adds device versions).
 *
 * **Grid Structure:**
 *
 * - Supports 1D, 2D, and 3D grids
 * - Grid spacing: dx[d] = lx[d] / nx[d]
 * - Volume element: dv[i] = Π dx[d] × mask[i] (V/M × mask[i] in an oblique cell)
 *
 * **Boundary Conditions:**
 *
 * Each dimension has two boundaries (low/high):
 * - PERIODIC: Wrap-around (default, required for pseudo-spectral)
 * - REFLECTING: Zero derivative (Neumann)
 * - ABSORBING: Zero value (Dirichlet)
 *
 * **Template Instantiations:**
 *
 * - ComputationBox<double>: Real-valued fields
 * - ComputationBox<std::complex<double>>: Complex-valued fields
 */

#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <map>
#include <complex>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <type_traits>

#include "ComputationBox.h"
#include "GridLimits.h"
#include "ReciprocalMetric.h"
#include "SpaceGroup.h"
#include "ValidationUtils.h"

namespace {
    /// Tolerance for comparing mask values to 0.0 or 1.0
    constexpr double MASK_TOLERANCE = 1e-7;

    /// An angle in degrees lies within FIXED_ANGLE_TOLERANCE radians of 90 degrees
    bool is_right_angle(double degrees)
    {
        return angle_is_fixed_at(degrees, std::numbers::pi / 2.0);
    }

    /**
     * @brief Expand dimension-appropriate angles to full 3-element array.
     *
     * Accepts angles based on dimensionality:
     * - 1D: empty or {90} (no angles needed)
     * - 2D: {gamma} (only gamma matters, alpha=beta=90°)
     * - 3D: {alpha, beta, gamma}
     *
     * @param angles Input angles (0, 1, or 3 elements)
     * @param dim    Simulation dimensionality (1, 2, or 3)
     * @return Expanded 3-element angles array [alpha, beta, gamma]
     */
    std::vector<double> expand_angles(const std::vector<double>& angles, int dim)
    {
        std::vector<double> full_angles = {90.0, 90.0, 90.0};  // Default orthogonal

        if (angles.empty())
        {
            // No angles specified: orthogonal system
            return full_angles;
        }
        else if (angles.size() == 1)
        {
            if (dim == 1 && is_right_angle(angles[0]))
                return full_angles;
            // Single angle: gamma only (for 2D)
            if (dim != 2)
                throw_with_line_number("Single-angle specification is only valid for 2D systems")
            full_angles[2] = angles[0];  // gamma
            return full_angles;
        }
        else if (angles.size() == 3)
        {
            // Angles involving absent axes must not enter the reciprocal metric.
            for (int i = 0; i < (dim == 1 ? 3 : dim == 2 ? 2 : 0); ++i)
                if (!is_right_angle(angles[i]))
                    throw_with_line_number("A 1D box requires all angles to be 90 degrees; "
                        "a 2D box requires alpha and beta to be 90 degrees.");
            return angles;
        }
        else
        {
            throw_with_line_number("angles must have 0, 1 (gamma for 2D), or 3 elements, but got " +
                std::to_string(angles.size()))
        }
    }

    /// True when a boundary is a wall (reflecting or absorbing)
    bool has_wall(const std::vector<BoundaryCondition>& bc)
    {
        return std::any_of(bc.begin(), bc.end(),
            [](BoundaryCondition b) { return b != BoundaryCondition::PERIODIC; });
    }

    /**
     * @brief Reject walls in a non-orthogonal cell.
     *
     * The wall solvers (DCT/DST pseudo-spectral and CN-ADI) treat each axis
     * separately and have no cross terms of the metric; the pseudo-spectral
     * ones would silently compute an orthogonal box.
     */
    void require_orthogonal_with_walls(const std::vector<BoundaryCondition>& bc, bool orthogonal)
    {
        if (has_wall(bc) && !orthogonal)
            throw_with_line_number("Reflecting and absorbing boundary conditions need an orthogonal box "
                "(all angles 90 degrees).")
    }

    /**
     * @brief Reject angles that do not form a unit cell or form a nearly flat one.
     *
     * The cell volume is V = abc sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ).
     * A lattice with a cell of V < MIN_VOLUME_RATIO abc also has a reduced (Niggli)
     * cell, spanned by its shortest vectors, with angles between 60 and 120 degrees
     * and V >= abc/sqrt(2). The flat cell is rejected: the alias search of its
     * oblique Fourier modes (MinimalFormAlias.h) spans about abc/V periods along
     * each axis, and the volume formula loses its precision when a cosine rounds
     * to +-1. A rhombohedral space group fixes the cell to its rhombohedral axes;
     * its hexagonal-axes setting describes the same structure.
     */
    constexpr double MIN_VOLUME_RATIO = 0.1;

    void require_cell_volume(const std::vector<double>& full_angles)
    {
        const double deg_to_rad = std::numbers::pi / 180.0;
        const double ca = std::cos(full_angles[0] * deg_to_rad);
        const double cb = std::cos(full_angles[1] * deg_to_rad);
        const double cg = std::cos(full_angles[2] * deg_to_rad);
        const double ratio_sq = 1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg;
        const std::string combination = "(alpha=" + std::to_string(full_angles[0]) +
            ", beta=" + std::to_string(full_angles[1]) + ", gamma=" + std::to_string(full_angles[2]) + ") degrees";
        if (ratio_sq <= 0.0)
            throw_with_line_number("Invalid lattice angles: they do not form a valid unit cell. The combination " +
                combination + " is geometrically impossible.");
        const double ratio = std::sqrt(ratio_sq);
        if (ratio < MIN_VOLUME_RATIO)
        {
            // All digits, so that a ratio just below the bound never prints as the bound
            std::ostringstream value;
            value << std::setprecision(std::numeric_limits<double>::max_digits10) << ratio;
            std::ostringstream message;
            message << "The lattice angles " << combination << " give a cell volume of " << value.str()
                    << " abc, below the minimum of " << MIN_VOLUME_RATIO << " abc. "
                    << "Use the reduced (Niggli) cell of the lattice, spanned by its shortest vectors, or for a "
                    << "rhombohedral space group its hexagonal-axes setting.";
            throw_with_line_number(message.str());
        }
    }
}

/**
 * @brief Construct computation box with given grid and boundaries (an orthogonal cell).
 *
 * @param new_nx Grid points per dimension [nx, ny, nz]
 * @param new_lx Box lengths per dimension [Lx, Ly, Lz]
 * @param bc     Boundary conditions (2 per dimension: low, high)
 * @param mask   Optional mask for impenetrable regions (nullptr = no mask)
 *
 * @throws Exception if dimensions mismatch or invalid parameters
 */
template <typename T>
ComputationBox<T>::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx,
    std::vector<std::string> bc, const double* mask)
    : ComputationBox(std::move(new_nx), std::move(new_lx), std::move(bc), std::vector<double>{}, mask)
{
}
//----------------- Constructor with angles -----------------------------
/**
 * @brief Construct computation box with lattice angles for non-orthogonal systems.
 *
 * @param new_nx    Grid points per dimension [nx, ny, nz]
 * @param new_lx    Box lengths (lattice constants) per dimension [a, b, c]
 * @param bc        Boundary conditions (2 per dimension: low, high)
 * @param angles    Lattice angles [alpha, beta, gamma] in DEGREES (empty: all 90 degrees)
 * @param mask      Optional mask for impenetrable regions (nullptr = no mask)
 * @param space_group Optional space group of the reduced basis (nullptr = full grid)
 *
 * @throws Exception if dimensions mismatch, invalid parameters, invalid angles, or a
 *         space group that does not fit the box
 */
template <typename T>
ComputationBox<T>::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx,
    std::vector<std::string> bc, std::vector<double> angles, const double* mask, SpaceGroup* space_group)
{
    validation::require_same_size(new_nx.size(), new_lx.size(), "nx", "lx");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
    {
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()))
    }
    validation::require_all_positive(new_nx, "nx");
    validation::require_all_positive(new_lx, "lx");

    try
    {
        this->dim = new_nx.size();
        nx = new_nx;
        lx = new_lx;

        const int DIM = this->dim;

        // Expand angles to full 3-element array (handles 1-element gamma for 2D)
        std::vector<double> full_angles = expand_angles(angles, DIM);

        // Validate angles
        for (int i = 0; i < 3; i++) {
            validation::require_in_range(full_angles[i], 0.0 + 1e-10, 180.0 - 1e-10,
                "angles[" + std::to_string(i) + "]");
        }
        require_cell_volume(full_angles);

        // Convert angles from degrees to radians and store
        angles_.resize(3);
        const double deg_to_rad = std::numbers::pi / 180.0;
        for (int d = 0; d < 3; d++)
            angles_[d] = full_angles[d] * deg_to_rad;

        // Orthogonal: all angles are 90 degrees within tolerance
        is_orthogonal_ = std::all_of(full_angles.begin(), full_angles.end(), is_right_angle);
        is_z_axis_orthogonal_ = is_right_angle(full_angles[0]) && is_right_angle(full_angles[1]);

        // Check before multiplication: three valid int dimensions can overflow even 64 bits.
        total_grid = checked_grid_points(nx);

        // Mask
        // Penetrable region == 1.0
        // Impenetrable region == 0.0
        if (mask != nullptr)
        {
            mask_full_.assign(mask, mask + total_grid);
            validation::validate_mask(mask_full_.data(), total_grid, MASK_TOLERANCE);
        }
        // else: mask remains empty (no mask)

        // Grid interval, lattice vectors, reciprocal lattice, metric tensor and volume elements
        update_cell();

        // Set boundary conditions
        if((unsigned int) 2*DIM != bc.size() && 0 != bc.size())
        {
            throw_with_line_number(
                "We expect 0 or " + std::to_string(2*DIM) + " boundary conditions, but we get " + std::to_string(bc.size()) +
                ". For each dimension, two boundary conditions (top and bottom) are required.");
        }

        if(bc.size() == 0)
        {
            // Default is periodic boundary
            for(int i=0; i<2*DIM; i++)
                this->bc.push_back(BoundaryCondition::PERIODIC);
        }
        else
        {
            for(int i=0; i<2*DIM; i+=2)
            {
                std::string bc_name_l = validation::to_lower(bc[i]);
                std::string bc_name_h = validation::to_lower(bc[i+1]);

                if((bc_name_l == "periodic" && bc_name_h != "periodic") ||
                   (bc_name_l != "periodic" && bc_name_h == "periodic"))
                {
                   throw_with_line_number(bc_name_l + " and "  + bc_name_h + " are an invalid boundary condition combination. " +
                    + "If one side imposes a periodic boundary condition, the other side must also be a periodic boundary condition.")
                }
            }

            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = validation::to_lower(bc[i]);

                if(bc_name == "periodic")
                    this->bc.push_back(BoundaryCondition::PERIODIC);
                else if(bc_name == "reflecting")
                    this->bc.push_back(BoundaryCondition::REFLECTING);
                else if(bc_name == "absorbing")
                    this->bc.push_back(BoundaryCondition::ABSORBING);
                else
                {
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']")
                }
            }
        }
        require_orthogonal_with_walls(this->bc, is_orthogonal_);

        // The full grid, or the reduced basis of the space group for the whole life of the box
        space_group_ = nullptr;
        n_basis_ = total_grid;
        if (space_group != nullptr)
            use_space_group(space_group);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//----------------- Destructor -----------------------------
template <typename T>
ComputationBox<T>::~ComputationBox()
{
    // Vectors handle memory cleanup automatically
}

//----------------- Cell -----------------------------
/**
 * @brief Grid spacing, cell geometry, volume elements and accessible volume of the
 *        current lx and angles.
 *
 * The constructor, set_lx and set_lattice_parameters all take this one update, so a
 * box moved to a cell equals one created there bit for bit. 1D and 2D cells use the
 * 3D formalism with unit lengths on the absent axes. The volume element is
 * dv_base_ = Π dx[d] in an orthogonal cell and the cell volume over the grid points
 * V/M in an oblique one (the two agree to round-off for right angles); dv carries
 * the full-grid mask (`mask` is in the reduced basis while a space group is set).
 */
template <typename T>
void ComputationBox<T>::update_cell()
{
    // Grid interval
    dx.resize(dim);
    for (int d = 0; d < dim; d++)
        dx[d] = lx[d] / nx[d];

    // Lattice vectors, cell volume, reciprocal lattice and metric tensor
    const CellGeometry geometry = cell_geometry({lx[0], (dim >= 2) ? lx[1] : 1.0, (dim >= 3) ? lx[2] : 1.0,
                                                 angles_[0], angles_[1], angles_[2]});
    volume = geometry.volume;
    lattice_vec_ = geometry.lattice_vec;
    recip_metric_ = geometry.recip_metric;

    // Volume element of a grid point, the weight of every integral, then with the mask
    if (is_orthogonal_)
    {
        dv_base_ = 1.0;
        for (int d = 0; d < dim; d++)
            dv_base_ *= dx[d];
    }
    else
    {
        dv_base_ = volume / total_grid;
    }
    dv.assign(total_grid, dv_base_);
    if (!mask_full_.empty())
        for (int i = 0; i < total_grid; i++)
            dv[i] *= mask_full_[i];

    accessible_volume = 0.0;
    for (int i = 0; i < total_grid; i++)
        accessible_volume += dv[i];
}

//----------------- get methods-------------------------------------
template <typename T>
int ComputationBox<T>::get_dim() const
{
    return dim;
}
template <typename T>
int ComputationBox<T>::get_nx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return nx[i];
}
template <typename T>
double ComputationBox<T>::get_lx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return lx[i];
}
template <typename T>
double ComputationBox<T>::get_dx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return dx[i];
}
template <typename T>
std::vector<int> ComputationBox<T>::get_nx() const
{
    return nx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_lx() const
{
    return lx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_dx() const
{
    return dx;
}
template <typename T>
double ComputationBox<T>::get_dv(int i) const
{
    return dv[i];
}
template <typename T>
double ComputationBox<T>::get_dv_base() const
{
    return dv_base_;
}
template <typename T>
int ComputationBox<T>::get_total_grid() const
{
    return total_grid;
}
template <typename T>
double ComputationBox<T>::get_volume() const
{
    return volume;
}
template <typename T>
double ComputationBox<T>::get_accessible_volume() const
{
    return accessible_volume;
}
template <typename T>
const double* ComputationBox<T>::get_mask() const
{
    // In the basis of the fields: reduced while a space group is set
    const std::vector<double>& active = (space_group_ != nullptr) ? mask : mask_full_;
    return active.empty() ? nullptr : active.data();
}
template <typename T>
const std::vector<BoundaryCondition> ComputationBox<T>::get_boundary_conditions() const
{
    return bc;
}
template <typename T>
BoundaryCondition ComputationBox<T>::get_boundary_condition(int i) const
{
    return bc[i];
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles() const
{
    return angles_;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles_degrees() const
{
    const double rad_to_deg = 180.0 / std::numbers::pi;
    std::vector<double> angles_deg(3);
    for (int d = 0; d < 3; d++)
        angles_deg[d] = angles_[d] * rad_to_deg;
    return angles_deg;
}
template <typename T>
const std::array<double, 9>& ComputationBox<T>::get_lattice_vec() const
{
    return lattice_vec_;
}
template <typename T>
const std::array<double, 6>& ComputationBox<T>::get_recip_metric() const
{
    return recip_metric_;
}
template <typename T>
bool ComputationBox<T>::is_orthogonal() const
{
    return is_orthogonal_;
}
template <typename T>
bool ComputationBox<T>::is_z_axis_orthogonal() const
{
    return is_z_axis_orthogonal_;
}
//----------------- set methods-------------------------------------
template <typename T>
void ComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    validation::require_same_size(new_lx.size(), (size_t)dim, "new lx", "dim");
    validation::require_all_positive(new_lx, "new lx");
    if (space_group_ != nullptr)
        space_group_->check_cell(new_lx, get_angles_degrees());

    lx = new_lx;
    update_cell();
}

template <typename T>
void ComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    validation::require_same_size(new_lx.size(), (size_t)dim, "new lx", "dim");
    validation::require_all_positive(new_lx, "new lx");

    // Expand angles to full 3-element array (handles 1-element gamma for 2D)
    std::vector<double> full_angles = expand_angles(new_angles, dim);

    // Validate angles
    for (int i = 0; i < 3; i++) {
        validation::require_in_range(full_angles[i], 0.0 + 1e-10, 180.0 - 1e-10,
            "angles[" + std::to_string(i) + "]");
    }
    const bool orthogonal = std::all_of(full_angles.begin(), full_angles.end(), is_right_angle);
    require_orthogonal_with_walls(bc, orthogonal);
    if (orthogonal_cell_holders_ > 0 && !orthogonal)
        throw_with_line_number("A computation of this box uses a numerical method that requires an "
            "orthogonal cell (all angles 90 degrees).");
    if (space_group_ != nullptr)
        space_group_->check_cell(new_lx, full_angles);

    // Validate that the angles form a unit cell before anything changes
    require_cell_volume(full_angles);
    const double deg_to_rad = std::numbers::pi / 180.0;

    lx = new_lx;

    // Convert angles from degrees to radians and store
    for (int d = 0; d < 3; d++)
        angles_[d] = full_angles[d] * deg_to_rad;

    is_orthogonal_ = orthogonal;
    is_z_axis_orthogonal_ = is_right_angle(full_angles[0]) && is_right_angle(full_angles[1]);
    update_cell();
}
// ==================== Space Group Methods ====================

template <typename T>
void ComputationBox<T>::use_space_group(SpaceGroup* sg)
{
    // The reduced basis holds the symmetric part of real fields only
    if constexpr (!std::is_same_v<T, double>)
        throw_with_line_number("Space group reduced basis is only supported for real fields.");

    // Walls do not commute with the group's translations, rotations and glides
    if (has_wall(bc))
        throw_with_line_number("Space group symmetry needs periodic boundary conditions in all directions.")

    // The group's maps index the grid it was built for
    if (sg->get_nx() != nx)
    {
        auto join = [](const std::vector<int>& v) {
            std::string text;
            for (size_t d = 0; d < v.size(); ++d)
                text += (d ? "x" : "") + std::to_string(v[d]);
            return text;
        };
        throw_with_line_number("The space group was built for a " + join(sg->get_nx())
            + " grid, but the computation box has " + join(nx) + ".");
    }
    sg->check_cell(lx, get_angles_degrees());

    // The reduced basis stores one value per orbit: a mask without the symmetry of
    // the group would silently become another one
    if (!mask_full_.empty() && !sg->is_invariant(mask_full_.data(), MASK_TOLERANCE))
        throw_with_line_number("The mask is not invariant under the space group; "
            "a mask used with a space group must have the symmetry of the group.")

    // The basis size, the orbit counts, the reduced mask and the integration weights
    // (orbit counts times the mask), so that the reduced-basis integrals equal the
    // full-grid ones, whose dv carries the mask
    space_group_ = sg;
    orbit_counts_ = sg->get_orbit_counts();
    n_basis_ = sg->get_n_reduced_basis();
    basis_weights_.assign(orbit_counts_.begin(), orbit_counts_.end());
    if (!mask_full_.empty())
    {
        mask.resize(n_basis_);
        sg->to_reduced_basis(mask_full_.data(), mask.data(), 1);
        for (size_t i = 0; i < basis_weights_.size(); i++)
            basis_weights_[i] *= mask[i];
    }

    // The sizes and maps above are snapshots: freeze the physical basis
    sg->mark_in_use();
}

template <typename T>
SpaceGroup* ComputationBox<T>::get_space_group() const
{
    return space_group_;
}

template <typename T>
int ComputationBox<T>::get_n_basis() const
{
    return n_basis_;
}

template <typename T>
const std::vector<int>& ComputationBox<T>::get_orbit_counts() const
{
    return orbit_counts_;
}

// ==================== Field Operations ====================

/**
 * @brief Compute volume integral of a field.
 *
 * When space_group_ is set, operates on reduced basis (size n_basis).
 * Otherwise, operates on full grid (size total_grid).
 */
template <typename T>
T ComputationBox<T>::integral(const T *g)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        for (int i = 0; i < n_basis_; i++)
            sum += static_cast<T>(basis_weights_[i]) * g[i];
        return sum * static_cast<T>(dv_base_);
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i];
        return sum;
    }
}

/**
 * @brief Compute mean (spatial average) of a field.
 */
template <typename T>
T ComputationBox<T>::mean(const T *g)
{
    return integral(g) / static_cast<T>(volume);
}

/**
 * @brief Compute inner product of two fields.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::inner_product(const T *g, const T *h)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        for (int i = 0; i < n_basis_; i++)
            sum += static_cast<T>(basis_weights_[i]) * g[i] * h[i];
        return sum * static_cast<T>(dv_base_);
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i] * h[i];
        return sum;
    }
}

/**
 * @brief Compute weighted inner product with inverse weight.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::inner_product_inverse_weight(const T *g, const T *h, const T *w)
{
    // A masked point adds nothing, whatever w is there; the mask is tested once, outside the sums
    const double* active_mask = get_mask();
    auto sum = [active_mask](int n, auto term)
    {
        T total{0.0};
        if (active_mask == nullptr)
            for (int i = 0; i < n; i++)
                total += term(i);
        else
            for (int i = 0; i < n; i++)
                if (active_mask[i] != 0.0)
                    total += term(i);
        return total;
    };

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        return sum(n_basis_, [&](int i) { return static_cast<T>(basis_weights_[i]) * g[i] * h[i] / w[i]; })
               * static_cast<T>(dv_base_);
    }
    else
    {
        // Full grid mode
        return sum(total_grid, [&](int i) { return dv[i] * g[i] * h[i] / w[i]; });
    }
}

/**
 * @brief Integral of |g h / divisor| with the weights of integral().
 */
template <typename T>
double ComputationBox<T>::absolute_inner_product(const T *g, const T *h, const T *divisor)
{
    const double* active_mask = get_mask();
    const bool reduced = space_group_ != nullptr;
    const int n = reduced ? n_basis_ : total_grid;
    const double* weight = reduced ? basis_weights_.data() : dv.data();
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        if (active_mask != nullptr && active_mask[i] == 0.0)
            continue;
        T product = g[i] * h[i];
        if (divisor != nullptr)
            product /= divisor[i];
        sum += weight[i] * std::abs(product);
    }
    return reduced ? sum * dv_base_ : sum;
}

/**
 * @brief Compute inner product for multiple field components.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::multi_inner_product(int n_comp, const T *g, const T *h)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        for (int n = 0; n < n_comp; n++)
        {
            const size_t offset = static_cast<size_t>(n) * n_basis_;
            for (int i = 0; i < n_basis_; i++)
                sum += static_cast<T>(basis_weights_[i]) * g[offset + i] * h[offset + i];
        }
        return sum * static_cast<T>(dv_base_);
    }
    else
    {
        // Full grid mode
        for (int n = 0; n < n_comp; n++)
        {
            // n*total_grid may exceed int for large grids; offset in 64 bits
            const size_t offset = static_cast<size_t>(n) * total_grid;
            for (int i = 0; i < total_grid; i++)
                sum += dv[i] * g[offset + i] * h[offset + i];
        }
        return sum;
    }
}

/**
 * @brief Remove the spatial mean from a field (make it zero-mean).
 *
 * Subtracts the mean over the accessible volume, so that integral() and mean() of
 * the result vanish also with a mask. When space_group_ is set, operates on reduced
 * basis using weighted mean.
 */
template <typename T>
void ComputationBox<T>::zero_mean(T *g)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode: the weighted sum over the accessible grid points, as in the
        // full-grid branch (the weights are the orbit sizes times the mask)
        double accessible_points = 0.0;
        for (int i = 0; i < n_basis_; i++)
        {
            sum += static_cast<T>(basis_weights_[i]) * g[i];
            accessible_points += basis_weights_[i];
        }
        T mean = sum / static_cast<T>(mask_full_.empty() ? static_cast<double>(total_grid) : accessible_points);

        for (int i = 0; i < n_basis_; i++)
            g[i] -= mean;
    }
    else
    {
        // Full grid mode: Σ dv g over the accessible volume (dv is zero at masked points)
        const double norm = mask_full_.empty() ? volume : accessible_volume;
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i];
        for (int i = 0; i < total_grid; i++)
            g[i] -= sum / norm;
    }
}

// Explicit template instantiation
template class ComputationBox<double>;
template class ComputationBox<std::complex<double>>;
