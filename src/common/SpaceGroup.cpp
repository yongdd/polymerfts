/**
 * @file SpaceGroup.cpp
 * @brief Implementation of SpaceGroup class using spglib for symmetry operations.
 */

#include "SpaceGroup.h"
#include "ComputationBox.h"
#include "GridLimits.h"
#include "Exception.h"
#include "CrysFFTBasisMapping.h"
#include "CrysFFTSelector.h"
#include <cstring>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <numbers>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <limits>

extern "C" {
#include <spglib.h>
}

namespace {
/// Whether a rotation (lattice coordinates) is diagonal
bool is_diagonal(const std::array<std::array<int, 3>, 3>& R)
{
    return R[0][1] == 0 && R[0][2] == 0 && R[1][0] == 0 && R[1][2] == 0 && R[2][0] == 0 && R[2][1] == 0;
}

/// Whether a rotation (lattice coordinates) is diag(sign)
bool is_diagonal_with(const std::array<std::array<int, 3>, 3>& R, const std::array<int, 3>& sign)
{
    return is_diagonal(R) && R[0][0] == sign[0] && R[1][1] == sign[1] && R[2][2] == sign[2];
}

/// Whether a translation component is a lattice translation (an integer within tol)
bool is_zero_mod1(double v, double tol)
{
    return std::fabs(v - std::round(v)) < tol;
}

/// Whether every component of a translation is a lattice translation
bool is_lattice_translation(const std::array<double, 3>& t, double tol)
{
    return is_zero_mod1(t[0], tol) && is_zero_mod1(t[1], tol) && is_zero_mod1(t[2], tol);
}

/// A translation component mapped to [-0.5, 0.5), for comparing the norms of translations
double wrap_half(double v)
{
    return v - std::floor(v + 0.5);
}

/// The rotations of the mirrors perpendicular to z, y and x (the order of get_m3_translations)
constexpr std::array<std::array<int, 3>, 3> MIRRORS = {{{1, 1, -1}, {1, -1, 1}, {-1, 1, 1}}};

/// Index of the shortest translation (components taken in [-1/2, 1/2)): the first one
/// unless a later one is shorter by more than tol; -1 for none.
int shortest_translation(const std::vector<std::array<double, 3>>& translations, double tol)
{
    int best = -1;
    double best_norm = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < translations.size(); ++k)
    {
        double norm = 0.0;
        for (double v : translations[k])
            norm += wrap_half(v) * wrap_half(v);
        if (best < 0 || norm + tol < best_norm)
        {
            best = static_cast<int>(k);
            best_norm = norm;
        }
    }
    return best;
}
}  // namespace

SpaceGroup::SpaceGroup(std::vector<int> nx, int hall_number)
{
    initialize(std::move(nx), hall_number);
}

SpaceGroup::SpaceGroup(std::vector<int> nx, const std::string& symbol, int hall_number)
{
    // Find Hall numbers for the symbol
    std::vector<int> hall_numbers = hall_numbers_from_symbol(symbol);

    if (hall_numbers.empty())
    {
        throw_with_line_number("No Hall numbers found for ITA symbol '" + symbol + "'");
    }

    if (hall_number == -1)
    {
        // No Hall number specified
        if (hall_numbers.size() > 1)
        {
            std::string msg = "Multiple Hall numbers found for symbol '" + symbol + "': [";
            for (size_t i = 0; i < hall_numbers.size(); ++i)
            {
                if (i > 0) msg += ", ";
                msg += std::to_string(hall_numbers[i]);
            }
            msg += "]. Please specify one.";
            throw_with_line_number(msg);
        }
        hall_number = hall_numbers[0];
    }
    else
    {
        // Verify provided Hall number is valid for this symbol
        if (std::find(hall_numbers.begin(), hall_numbers.end(), hall_number) == hall_numbers.end())
        {
            std::string msg = "Hall number " + std::to_string(hall_number) +
                " is not valid for symbol '" + symbol + "'. Valid Hall numbers: [";
            for (size_t i = 0; i < hall_numbers.size(); ++i)
            {
                if (i > 0) msg += ", ";
                msg += std::to_string(hall_numbers[i]);
            }
            msg += "]";
            throw_with_line_number(msg);
        }
    }

    initialize(std::move(nx), hall_number);
}

std::vector<int> SpaceGroup::hall_numbers_from_symbol(const std::string& symbol)
{
    std::vector<int> result;
    std::string symbol_trimmed = symbol;

    // Trim whitespace
    symbol_trimmed.erase(0, symbol_trimmed.find_first_not_of(" \t"));
    symbol_trimmed.erase(symbol_trimmed.find_last_not_of(" \t") + 1);

    for (int h = 1; h <= 530; ++h)
    {
        SpglibSpacegroupType spg_type = spg_get_spacegroup_type(h);
        std::string spg_symbol = spg_type.international_short;

        // Trim whitespace from spglib symbol
        spg_symbol.erase(0, spg_symbol.find_first_not_of(" \t"));
        spg_symbol.erase(spg_symbol.find_last_not_of(" \t") + 1);

        if (spg_symbol == symbol_trimmed)
        {
            result.push_back(h);
        }
    }

    return result;
}

void SpaceGroup::initialize(std::vector<int> nx, int hall_number)
{
    nx_ = std::move(nx);
    hall_number_ = hall_number;

    if (nx_.size() != 3)
    {
        throw_with_line_number("SpaceGroup requires 3D grid dimensions");
    }
    for (int n : nx_)
        if (n <= 0)
            throw_with_line_number("SpaceGroup grid dimensions must be positive.");

    if (hall_number_ < 1 || hall_number_ > 530)
    {
        throw_with_line_number("Hall number must be between 1 and 530, got " +
            std::to_string(hall_number_));
    }

    // Compute total grid size, in 64 bits: the space group is created before the
    // computation box and must reject the same grids (see GridLimits.h)
    long long n_points = 1;
    for (int n : nx_)
    {
        n_points *= n;
        if (n_points > MAX_GRID_POINTS)
            throw_with_line_number("The grid exceeds " + std::to_string(MAX_GRID_POINTS) +
                " points, the limit for the 32-bit grid indices.");
    }
    total_grid_ = static_cast<int>(n_points);

    // Get space group info from spglib
    SpglibSpacegroupType spg_type = spg_get_spacegroup_type(hall_number_);
    spacegroup_number_ = spg_type.number;
    spacegroup_symbol_ = spg_type.international_short;

    // Determine crystal system
    determine_crystal_system();

    // Validate grid for crystal system (e.g., cubic requires nx=ny=nz)
    validate_grid();

    // Get symmetry operations (needed for grid compatibility check)
    get_symmetry_operations();

    // Validate grid compatibility with symmetry operations
    validate_grid_compatibility();

    // The operations as exact maps of the grid, and the irreducible orbits they make
    build_grid_maps();
    find_irreducible_orbits();
}

void SpaceGroup::display_info() const
{
    std::cout << "---------- Space Group ----------" << std::endl;
    std::cout << "Hall number: " << hall_number_ << std::endl;
    std::cout << "International space group number: " << spacegroup_number_ << std::endl;
    std::cout << "Symbol: " << spacegroup_symbol_ << std::endl;
    std::cout << "Crystal system: " << crystal_system_ << std::endl;
    std::cout << "Number of symmetry operations: " << n_symmetry_ops_ << std::endl;
    std::cout << "Original mesh size: " << total_grid_ << std::endl;
    std::cout << "Reduced basis size (irreducible): " << get_n_reduced_basis_full() << std::endl;
    // The Pmmm and m3 physical bases are an octant of an even grid
    const bool even_grid = (nx_[0] % 2 == 0 && nx_[1] % 2 == 0 && nx_[2] % 2 == 0);
    const std::string octant = std::to_string((nx_[0] / 2) * (nx_[1] / 2) * (nx_[2] / 2));
    std::array<double, 9> m3_part{};
    const bool has_m3 = even_grid && get_m3_translations(m3_part);
    std::cout << "Pmmm physical basis size (if enabled): " << (even_grid && has_mirror_planes_xyz() ? octant : "n/a")
              << std::endl;
    std::cout << "M3 physical basis size (if enabled): " << (has_m3 ? octant : "n/a") << std::endl;
}

void SpaceGroup::validate_grid()
{
    if (crystal_system_ == "Cubic")
    {
        if (!(nx_[0] == nx_[1] && nx_[1] == nx_[2]))
        {
            throw_with_line_number("Cubic crystal system requires nx = ny = nz. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
    else if (crystal_system_ == "Tetragonal" || crystal_system_ == "Trigonal")
    {
        if (nx_[0] != nx_[1])
        {
            throw_with_line_number(crystal_system_ + " crystal system requires nx = ny. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
    else if (crystal_system_ == "Hexagonal")
    {
        if (nx_[0] != nx_[1])
        {
            throw_with_line_number("Hexagonal crystal system requires nx = ny. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
}

std::array<int, 3> SpaceGroup::get_required_grid_divisors() const
{
    // Per axis: for every translation component t = p/q (in lowest terms) of a
    // symmetry operation along that axis, the grid size of the axis must be divisible
    // by q (a 1/2 translation along z alone, as in P6_3/mmc, constrains only nz).

    auto get_denominator = [](double t) -> int {
        t = wrap01(t);

        // Check common fractions with tolerance
        const double tol = 1e-6;

        // Check for 0
        if (std::abs(t) < tol || std::abs(t - 1.0) < tol) return 1;

        // Check for 1/2
        if (std::abs(t - 0.5) < tol) return 2;

        // Check for 1/3, 2/3
        if (std::abs(t - 1.0/3.0) < tol || std::abs(t - 2.0/3.0) < tol) return 3;

        // Check for 1/4, 3/4
        if (std::abs(t - 0.25) < tol || std::abs(t - 0.75) < tol) return 4;

        // Check for 1/6, 5/6
        if (std::abs(t - 1.0/6.0) < tol || std::abs(t - 5.0/6.0) < tol) return 6;

        // Check for 1/8, 3/8, 5/8, 7/8
        for (int n = 1; n <= 7; n += 2) {
            if (std::abs(t - n/8.0) < tol) return 8;
        }

        // Check for 1/12, 5/12, 7/12, 11/12
        for (int n : {1, 5, 7, 11}) {
            if (std::abs(t - n/12.0) < tol) return 12;
        }

        return 1;  // Unknown fraction, assume no special requirement
    };

    std::array<int, 3> result = {1, 1, 1};

    for (int op = 0; op < n_symmetry_ops_; ++op)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            int denom = get_denominator(translations_[op][dim]);
            result[dim] = std::lcm(result[dim], denom);
        }
    }

    return result;
}

void SpaceGroup::validate_grid_compatibility()
{
    const std::array<int, 3> divisors = get_required_grid_divisors();

    std::vector<int> incompatible;
    for (int dim = 0; dim < 3; ++dim)
    {
        if (divisors[dim] > 1 && nx_[dim] % divisors[dim] != 0)
            incompatible.push_back(dim);
    }
    if (incompatible.empty())
        return;

    const char* axis_names[] = {"x", "y", "z"};
    std::string msg = "Grid [" + std::to_string(nx_[0]) + ", " +
        std::to_string(nx_[1]) + ", " + std::to_string(nx_[2]) +
        "] is incompatible with space group " + spacegroup_symbol_ +
        " (No. " + std::to_string(spacegroup_number_) + ").\n";
    msg += "The translation parts of the symmetry operations require:";
    for (int dim : incompatible)
    {
        msg += std::string(" ") + axis_names[dim] + "=" + std::to_string(nx_[dim]) +
            " (a multiple of " + std::to_string(divisors[dim]) + ")";
    }

    // Suggest valid grid sizes
    msg += "\nSuggested grid: [";
    for (int dim = 0; dim < 3; ++dim)
    {
        if (dim > 0) msg += ", ";
        const int d = divisors[dim];
        if (d <= 1 || nx_[dim] % d == 0)
        {
            msg += std::to_string(nx_[dim]);
        }
        else
        {
            int lower = (nx_[dim] / d) * d;
            int upper = lower + d;
            int suggested = (upper - nx_[dim] < nx_[dim] - lower || lower == 0) ? upper : lower;
            msg += std::to_string(suggested);
        }
    }
    msg += "]";

    throw_with_line_number(msg);
}

void SpaceGroup::determine_crystal_system()
{
    if (spacegroup_number_ >= 1 && spacegroup_number_ <= 2)
        crystal_system_ = "Triclinic";
    else if (spacegroup_number_ >= 3 && spacegroup_number_ <= 15)
        crystal_system_ = "Monoclinic";
    else if (spacegroup_number_ >= 16 && spacegroup_number_ <= 74)
        crystal_system_ = "Orthorhombic";
    else if (spacegroup_number_ >= 75 && spacegroup_number_ <= 142)
        crystal_system_ = "Tetragonal";
    else if (spacegroup_number_ >= 143 && spacegroup_number_ <= 167)
        crystal_system_ = "Trigonal";
    else if (spacegroup_number_ >= 168 && spacegroup_number_ <= 194)
        crystal_system_ = "Hexagonal";
    else if (spacegroup_number_ >= 195 && spacegroup_number_ <= 230)
        crystal_system_ = "Cubic";
    else
        throw_with_line_number("Invalid space group number: " + std::to_string(spacegroup_number_));
}

void SpaceGroup::get_symmetry_operations()
{
    // Get symmetry operations from spglib database using Hall number
    // Maximum possible symmetry operations is 192 (for Fm-3m)
    constexpr int max_size = 192;
    int rotations[max_size][3][3];
    double translations[max_size][3];

    int n_ops = spg_get_symmetry_from_database(rotations, translations, hall_number_);

    if (n_ops == 0)
    {
        throw_with_line_number("Failed to get symmetry operations for Hall number " +
            std::to_string(hall_number_));
    }

    n_symmetry_ops_ = n_ops;

    // Store symmetry operations
    rotations_.resize(n_ops);
    translations_.resize(n_ops);

    for (int i = 0; i < n_ops; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                rotations_[i][j][k] = rotations[i][j][k];
            }
            translations_[i][j] = translations[i][j];
        }
    }
}

void SpaceGroup::build_grid_maps()
{
    // Grid point i of an axis with N points sits at (i + o/2)/N: o = 1 at the cell
    // centers, o = 0 at the nodes. Hexagonal-axes settings put x and y on the nodes:
    // an in-plane row such as x' = x - y annihilates a half-cell offset and maps cell
    // centers onto cell boundaries, while every operation of these settings is a 2x2
    // in-plane block (+) (+/-1 on z). All other axes and settings are cell-centered.
    //
    // In half-index units u = 2i + o the operation x' = R x + t reads
    //     u'_r = sum_k R_rk u_k (N_r/N_k) + 2 N_r t_r,
    // so with equal sizes on the axes R mixes it is an integer map of the indices,
    //     i'_r = sum_k R_rk i_k + c_r (mod N_r),  c_r = (sum_k R_rk o_k + 2 N_r t_r - o_r) / 2,
    // provided 2 N_r t_r and c_r are integers: a grid point then lands on a grid point.
    // validate_grid() and validate_grid_compatibility() ensure this for every setting
    // except a rhombohedral-axes setting with unequal sizes, which is refused here.
    const bool hexagonal_axes = uses_hexagonal_axes();
    const std::array<int, 3> o = {hexagonal_axes ? 0 : 1, hexagonal_axes ? 0 : 1, 1};

    grid_maps_.resize(n_symmetry_ops_);
    for (int op = 0; op < n_symmetry_ops_; ++op)
    {
        const auto& R = rotations_[op];
        bool exact = true;
        for (int r = 0; r < 3; ++r)
        {
            long long numerator = -o[r];
            for (int k = 0; k < 3; ++k)
            {
                if (R[r][k] != 0 && nx_[k] != nx_[r])
                    exact = false;
                numerator += static_cast<long long>(R[r][k]) * o[k];
            }
            const double shift = 2.0 * static_cast<double>(nx_[r]) * translations_[op][r];
            const long long shift_int = std::llround(shift);
            numerator += shift_int;
            if (std::abs(shift - static_cast<double>(shift_int)) > 1e-6 || numerator % 2 != 0)
                exact = false;
            const long long c = (numerator / 2) % nx_[r];
            grid_maps_[op].R[r] = R[r];
            grid_maps_[op].c[r] = static_cast<int>(c < 0 ? c + nx_[r] : c);
        }
        if (!exact)
        {
            throw_with_line_number(
                "Symmetry operation " + std::to_string(op) + " does not map the grid (cell-centered; "
                "node-centered x and y in a hexagonal-axes setting) onto itself. This grid is "
                "incompatible with the space group under the current sampling convention; for "
                "rhombohedral-axes settings use Nx = Ny = Nz.");
        }
    }
}

std::array<int, 3> SpaceGroup::grid_indices(int point) const
{
    return {point / (nx_[1] * nx_[2]), (point / nx_[2]) % nx_[1], point % nx_[2]};
}

int SpaceGroup::map_point(const GridMap& op, const std::array<int, 3>& i) const
{
    int image = 0;
    for (int r = 0; r < 3; ++r)
    {
        // 0 <= c_r < N_r, and R mixes only axes of size N_r with entries 0 and +-1:
        // v lies within a few periods of [0, N_r)
        long long v = op.c[r];
        for (int k = 0; k < 3; ++k)
            v += static_cast<long long>(op.R[r][k]) * i[k];
        while (v < 0)
            v += nx_[r];
        while (v >= nx_[r])
            v -= nx_[r];
        image = image * nx_[r] + static_cast<int>(v);
    }
    return image;
}

void SpaceGroup::find_irreducible_orbits()
{
    // The first point of an orbit in flat order represents it; its images under all
    // operations are the orbit
    orbit_of_.assign(total_grid_, -1);
    std::vector<int> representatives;
    for (int point = 0; point < total_grid_; ++point)
    {
        if (orbit_of_[point] >= 0)
            continue;
        const int orbit = static_cast<int>(representatives.size());
        representatives.push_back(point);
        const std::array<int, 3> i = grid_indices(point);
        for (const GridMap& op : grid_maps_)
            orbit_of_[map_point(op, i)] = orbit;
    }
    n_orbits_ = static_cast<int>(representatives.size());
    orbit_sizes_.assign(n_orbits_, 0);
    for (int point = 0; point < total_grid_; ++point)
        orbit_sizes_[orbit_of_[point]] += 1;

    // Without a physical basis the active basis is the irreducible one
    n_basis_ = n_orbits_;
    basis_points_ = std::move(representatives);
    basis_of_ = orbit_of_;
    basis_counts_ = orbit_sizes_;
    basis_to_orbit_.clear();
}

void SpaceGroup::to_reduced_basis(const double* full_field, double* reduced_field, int n_fields) const
{
    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = full_field + static_cast<size_t>(f) * total_grid_;
        double* dst = reduced_field + static_cast<size_t>(f) * n_basis_;

        for (int i = 0; i < n_basis_; ++i)
        {
            dst[i] = src[basis_points_[i]];
        }
    }
}

void SpaceGroup::reduced_to_irreducible(const double* reduced_field, double* irreducible_field, int n_fields) const
{
    // Without a physical basis the reduced basis is the irreducible one
    if (physical_basis_ == PhysicalBasis::None)
    {
        const int n_reduced = n_basis_;
        for (int f = 0; f < n_fields; ++f)
        {
            std::memcpy(irreducible_field + static_cast<size_t>(f) * n_reduced,
                        reduced_field + static_cast<size_t>(f) * n_reduced,
                        sizeof(double) * n_reduced);
        }
        return;
    }

    const int n_reduced = n_basis_;
    const int n_irreducible = n_orbits_;

    std::vector<double> orbit_sums(n_irreducible, 0.0);
    std::vector<int> orbit_weights(n_irreducible, 0);

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = reduced_field + static_cast<size_t>(f) * n_reduced;
        double* dst = irreducible_field + static_cast<size_t>(f) * n_irreducible;

        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);
        std::fill(orbit_weights.begin(), orbit_weights.end(), 0);

        for (int i = 0; i < n_reduced; ++i)
        {
            const int ridx = basis_to_orbit_[i];
            const int weight = basis_counts_[i];
            orbit_sums[ridx] += static_cast<double>(weight) * src[i];
            orbit_weights[ridx] += weight;
        }

        for (int i = 0; i < n_irreducible; ++i)
        {
            if (orbit_weights[i] > 0)
                dst[i] = orbit_sums[i] / static_cast<double>(orbit_weights[i]);
            else
                dst[i] = 0.0;
        }
    }
}

void SpaceGroup::irreducible_to_reduced(const double* irreducible_field, double* reduced_field, int n_fields) const
{
    // Without a physical basis the reduced basis is the irreducible one
    if (physical_basis_ == PhysicalBasis::None)
    {
        const int n_reduced = n_basis_;
        for (int f = 0; f < n_fields; ++f)
        {
            std::memcpy(reduced_field + static_cast<size_t>(f) * n_reduced,
                        irreducible_field + static_cast<size_t>(f) * n_reduced,
                        sizeof(double) * n_reduced);
        }
        return;
    }

    const int n_reduced = n_basis_;
    const int n_irreducible = n_orbits_;

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = irreducible_field + static_cast<size_t>(f) * n_irreducible;
        double* dst = reduced_field + static_cast<size_t>(f) * n_reduced;

        for (int i = 0; i < n_reduced; ++i)
        {
            dst[i] = src[basis_to_orbit_[i]];
        }
    }
}

void SpaceGroup::from_reduced_basis(const double* reduced_field, double* full_field, int n_fields) const
{
    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = reduced_field + static_cast<size_t>(f) * n_basis_;
        double* dst = full_field + static_cast<size_t>(f) * total_grid_;

        for (int i = 0; i < total_grid_; ++i)
        {
            dst[i] = src[basis_of_[i]];
        }
    }
}

bool SpaceGroup::is_invariant(const double* full_field, double tolerance) const
{
    // Orbits of the full group (kept from the constructor; a physical basis replaces
    // only the active basis)
    std::vector<double> value(n_orbits_, 0.0);
    std::vector<char> seen(n_orbits_, 0);
    for (int i = 0; i < total_grid_; i++)
    {
        const int orbit = orbit_of_[i];
        if (!seen[orbit])
        {
            seen[orbit] = 1;
            value[orbit] = full_field[i];
        }
        else if (std::abs(full_field[i] - value[orbit]) > tolerance)
            return false;
    }
    return true;
}

void SpaceGroup::symmetrize(double* field, int n_fields) const
{
    // Temporary buffer for orbit sums
    std::vector<double> orbit_sums(n_orbits_);

    for (int f = 0; f < n_fields; ++f)
    {
        double* data = field + static_cast<size_t>(f) * total_grid_;

        // Reset orbit sums
        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);

        // Accumulate values by orbit
        for (int i = 0; i < total_grid_; ++i)
        {
            orbit_sums[orbit_of_[i]] += data[i];
        }

        // Compute averages and broadcast back
        for (int i = 0; i < total_grid_; ++i)
        {
            int orbit_idx = orbit_of_[i];
            data[i] = orbit_sums[orbit_idx] / orbit_sizes_[orbit_idx];
        }
    }
}

void SpaceGroup::symmetrize(const double* field_in, double* field_out, int n_fields) const
{
    // Copy input to output first
    std::memcpy(field_out, field_in, sizeof(double) * n_fields * total_grid_);

    // Then symmetrize in-place
    symmetrize(field_out, n_fields);
}

std::vector<std::array<double, 3>> SpaceGroup::translations_of(const std::array<int, 3>& sign) const
{
    std::vector<std::array<double, 3>> result;
    for (int op = 0; op < n_symmetry_ops_; ++op)
    {
        if (is_diagonal_with(rotations_[op], sign))
        {
            const auto& t = translations_[op];
            result.push_back({wrap01(t[0]), wrap01(t[1]), wrap01(t[2])});
        }
    }
    return result;
}

bool SpaceGroup::has_mirror_planes_xyz(double tol) const
{
    // The mirrors perpendicular to x, y and z through the origin (lattice translations)
    for (const auto& sign : MIRRORS)
    {
        const auto translations = translations_of(sign);
        if (std::none_of(translations.begin(), translations.end(),
                         [tol](const auto& t) { return is_lattice_translation(t, tol); }))
            return false;
    }
    return true;
}

bool SpaceGroup::get_m3_translations(std::array<double, 9>& g, double tol) const
{
    // The shortest translation of each mirror (z, y, x)
    bool found_all = true;
    for (int m = 0; m < 3; ++m)
    {
        const auto translations = translations_of(MIRRORS[m]);
        const int k = shortest_translation(translations, tol);
        if (k < 0)
        {
            found_all = false;
            continue;
        }
        for (int d = 0; d < 3; ++d)
            g[3 * m + d] = translations[k][d];
    }
    return found_all;
}

void SpaceGroup::get_pure_translations(std::vector<std::array<double, 3>>& translations) const
{
    translations = translations_of({1, 1, 1});
}

bool SpaceGroup::get_z_mirror_translation(double& t_z, double tol) const
{
    // The shortest (x, y, -z) + t with lattice translations along x and y
    auto translations = translations_of(MIRRORS[0]);
    std::erase_if(translations, [tol](const auto& t) { return !is_zero_mod1(t[0], tol) || !is_zero_mod1(t[1], tol); });
    const int k = shortest_translation(translations, tol);
    if (k < 0)
        return false;
    t_z = translations[k][2];
    return true;
}

int SpaceGroup::z_mirror_half_grid_shift(std::string* why) const
{
    auto none = [why](const std::string& reason) {
        if (why != nullptr)
            *why = reason;
        return -1;
    };
    const int Nz = nx_[2];
    if ((Nz % 2) != 0)
        return none("Z-mirror physical basis requires even Nz.");
    double t_z = 0.0;
    if (!get_z_mirror_translation(t_z))
        return none("Space group does not contain (x,y,-z)+t mirror for z-physical basis.");

    // Only t_z = 0 or 1/2 (within tolerance)
    const double tol = 1e-8;
    const double t = (std::fabs(t_z - 1.0) < tol) ? 0.0 : t_z;
    if (std::fabs(t) < tol)
        return 0;
    if (std::fabs(t - 0.5) < tol)
    {
        if ((Nz % 4) != 0)
            return none("Z-mirror physical basis with t_z=1/2 requires Nz divisible by 4.");
        return Nz / 4;
    }
    return none("Unsupported z-mirror translation (only 0 or 1/2 are supported).");
}

template <typename Keep>
std::vector<SpaceGroup::GridMap> SpaceGroup::grid_maps_where(Keep keep) const
{
    std::vector<GridMap> result;
    for (int op = 0; op < n_symmetry_ops_; ++op)
        if (keep(rotations_[op], translations_[op]))
            result.push_back(grid_maps_[op]);
    return result;
}

void SpaceGroup::check_basis_not_frozen() const
{
    if (in_use_)
        throw_with_line_number("The physical basis cannot be changed after a computation box or solver was created with this SpaceGroup (their reduced-basis maps are snapshots); create a new SpaceGroup.");
}

void SpaceGroup::adopt_physical_basis(PhysicalBasis basis, const CrysFFTBasisMapping& grid,
                                      const std::vector<GridMap>& folds)
{
    // Built in locals and committed at the end, so that a failure leaves the group as it was
    std::vector<int> grid_index(total_grid_, -1);
    for (int b = 0; b < grid.M_phys; ++b)
        grid_index[grid.phys_to_full[b]] = b;

    std::vector<int> basis_of(total_grid_);
    std::vector<int> basis_counts(grid.M_phys, 0);
    for (int point = 0; point < total_grid_; ++point)
    {
        int b = grid_index[point];
        if (b < 0)
        {
            const std::array<int, 3> i = grid_indices(point);
            for (size_t k = 0; b < 0 && k < folds.size(); ++k)
                b = grid_index[map_point(folds[k], i)];
        }
        if (b < 0)
        {
            const char* name = basis == PhysicalBasis::M3 ? "M3" : (basis == PhysicalBasis::Pmmm ? "Pmmm" : "Z-mirror");
            throw_with_line_number(std::string("Failed to map grid point to ") + name + " physical basis.");
        }
        basis_of[point] = b;
        basis_counts[b] += 1;
    }

    n_basis_ = grid.M_phys;
    basis_points_ = grid.phys_to_full;
    basis_of_ = std::move(basis_of);
    basis_counts_ = std::move(basis_counts);
    basis_to_orbit_ = grid.phys_to_reduced;
    physical_basis_ = basis;
}

void SpaceGroup::enable_pmmm_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::Pmmm)
        return;
    check_basis_not_frozen();

    if ((nx_[0] % 2) != 0 || (nx_[1] % 2) != 0 || (nx_[2] % 2) != 0)
        throw_with_line_number("Pmmm physical basis requires even grid dimensions.");

    if (!has_mirror_planes_xyz())
        throw_with_line_number("Space group does not contain mirror planes along x, y, z (Pmmm).");

    // The octant at consecutive indices; the mirrors through the origin fold the grid onto it
    const std::array<int, 3> nx3 = {nx_[0], nx_[1], nx_[2]};
    adopt_physical_basis(PhysicalBasis::Pmmm,
        build_crysfft_mapping_octant(nx3, orbit_of_, n_orbits_, /*even_indices=*/false),
        grid_maps_where([](const auto& R, const auto& t) { return is_diagonal(R) && is_lattice_translation(t, 1e-10); }));
}

bool SpaceGroup::m3_crysfft_feasible() const
{
    std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
    const std::array<int, 3> nx3 = {nx_[0], nx_[1], nx_[2]};
    return crysfft_recursive_3m_feasible(this, nx3, trans_part);
}

void SpaceGroup::enable_m3_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::M3)
        return;
    check_basis_not_frozen();

    if ((nx_[0] % 2) != 0 || (nx_[1] % 2) != 0 || (nx_[2] % 2) != 0)
        throw_with_line_number("M3 physical basis requires even grid dimensions.");

    std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
    if (!get_m3_translations(trans_part))
        throw_with_line_number("Space group does not contain 3m translations for M3 physical basis.");

    // The octant at even indices; a point takes the first diagonal operation that maps it there
    const std::array<int, 3> nx3 = {nx_[0], nx_[1], nx_[2]};
    adopt_physical_basis(PhysicalBasis::M3,
        build_crysfft_mapping_octant(nx3, orbit_of_, n_orbits_, /*even_indices=*/true),
        grid_maps_where([](const auto& R, const auto&) { return is_diagonal(R); }));
}

void SpaceGroup::enable_z_mirror_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::ZMirror)
        return;
    check_basis_not_frozen();

    std::string why;
    const int shift = z_mirror_half_grid_shift(&why);
    if (shift < 0)
        throw_with_line_number(why);

    // The z half starting at the shift; the (x, y, -z) + t mirror of the shift folds the grid onto it
    double t_z = 0.0;
    get_z_mirror_translation(t_z);
    const std::array<int, 3> nx3 = {nx_[0], nx_[1], nx_[2]};
    adopt_physical_basis(PhysicalBasis::ZMirror,
        build_crysfft_mapping_z_mirror(nx3, orbit_of_, n_orbits_, shift),
        grid_maps_where([t_z](const auto& R, const auto& t) {
            return is_diagonal_with(R, MIRRORS[0]) && is_zero_mod1(t[0], 1e-10) && is_zero_mod1(t[1], 1e-10)
                   && is_zero_mod1(wrap01(t[2]) - t_z, 1e-10);
        }));
}

void SpaceGroup::check_cell(const std::vector<double>& lx, const std::vector<double>& angles_deg) const
{
    if (lx.size() != 3 || angles_deg.size() != 3)
        return;
    // Lengths and free angles agree to a relative 1e-6. An angle the group fixes
    // passes the orthogonality test of ComputationBox (angle_is_fixed_at, 1e-10 rad):
    // a solver chooses its CrysFFT engine for the angles of its box once, and every
    // cell accepted later keeps the angles that engine needs.
    auto same = [](double a, double b) { return std::abs(a - b) <= 1e-6 * std::max({std::abs(a), std::abs(b), 1.0}); };
    const std::array<double, 3> fixed = fixed_angles();
    bool angles_fixed = true;
    for (int i = 0; i < 3; ++i)
        if (!std::isnan(fixed[i]))
            angles_fixed = angles_fixed && angle_is_fixed_at(angles_deg[i], fixed[i] * (std::numbers::pi / 180.0));

    bool ok = true;
    std::string need;
    if (crystal_system_ == "Cubic")
    {
        ok = same(lx[0], lx[1]) && same(lx[0], lx[2]) && angles_fixed;
        need = "lx[0] = lx[1] = lx[2] and an orthogonal cell (angles [90, 90, 90])";
    }
    else if (crystal_system_ == "Tetragonal")
    {
        ok = same(lx[0], lx[1]) && angles_fixed;
        need = "lx[0] = lx[1] and an orthogonal cell (angles [90, 90, 90])";
    }
    else if (crystal_system_ == "Orthorhombic")
    {
        ok = angles_fixed;
        need = "an orthogonal cell (angles [90, 90, 90])";
    }
    else if (crystal_system_ == "Hexagonal" || (crystal_system_ == "Trigonal" && uses_hexagonal_axes()))
    {
        ok = same(lx[0], lx[1]) && angles_fixed;
        need = "lx[0] = lx[1] and angles [90, 90, 120]";
    }
    else if (crystal_system_ == "Trigonal")
    {
        ok = same(lx[0], lx[1]) && same(lx[0], lx[2]) && same(angles_deg[0], angles_deg[1])
             && same(angles_deg[0], angles_deg[2]);
        need = "lx[0] = lx[1] = lx[2] and alpha = beta = gamma (rhombohedral-axes setting)";
    }
    else
    {
        // Monoclinic and triclinic rotations are diagonal in lattice coordinates (every
        // Hall setting of groups 1-15), so a rotation preserves the cell metric exactly
        // when the angles between the axes it reverses relative to each other are right:
        // the angles fixed_angles() names, at the unique axis of a monoclinic group.
        // Triclinic groups fix no angle.
        ok = angles_fixed;
        const char* names[3] = {"alpha", "beta", "gamma"};
        std::string right_angles;
        for (int i = 0; i < 3; ++i)
            if (!std::isnan(fixed[i]))
                right_angles += std::string(right_angles.empty() ? "" : " = ") + names[i];
        need = "the unique axis perpendicular to the other two (" + right_angles + " = 90)";
    }
    if (!ok)
    {
        std::ostringstream got;
        got << std::setprecision(15) << "lx = [" << lx[0] << ", " << lx[1] << ", " << lx[2] << "], angles = ["
            << angles_deg[0] << ", " << angles_deg[1] << ", " << angles_deg[2] << "]";
        throw_with_line_number("The cell does not match the " + crystal_system_ + " crystal system of space group " +
            spacegroup_symbol_ + ": it needs " + need + " (the angles the group fixes to 1e-10 rad, lengths to "
            "a relative 1e-6). Got " + got.str() + ".");
    }
}

std::array<double, 3> SpaceGroup::fixed_angles() const
{
    const double free = std::numeric_limits<double>::quiet_NaN();
    if (crystal_system_ == "Cubic" || crystal_system_ == "Tetragonal" || crystal_system_ == "Orthorhombic")
        return {90.0, 90.0, 90.0};
    if (crystal_system_ == "Hexagonal" || (crystal_system_ == "Trigonal" && uses_hexagonal_axes()))
        return {90.0, 90.0, 120.0};
    std::array<double, 3> result = {free, free, free};
    if (crystal_system_ == "Trigonal")
        return result;
    // Monoclinic and triclinic rotations are diagonal in lattice coordinates. One that
    // reverses exactly one of two axes preserves the metric only when they are
    // perpendicular: the angles at the unique axis of a monoclinic group.
    for (const auto& R : rotations_)
        for (int i = 0; i < 3; ++i)
            if (R[(i + 1) % 3][(i + 1) % 3] * R[(i + 2) % 3][(i + 2) % 3] < 0)
                result[i] = 90.0;
    return result;
}

bool SpaceGroup::uses_hexagonal_axes() const
{
    if (crystal_system_ != "Hexagonal" && crystal_system_ != "Trigonal")
        return false;
    for (const auto& R : rotations_)
    {
        if (R[0][2] != 0 || R[1][2] != 0 || R[2][0] != 0 || R[2][1] != 0 || std::abs(R[2][2]) != 1)
            return false;   // 3-fold about [111]: rhombohedral axes
    }
    return true;
}

std::string SpaceGroup::physical_basis_name() const
{
    switch (physical_basis_)
    {
        case PhysicalBasis::Pmmm:    return "pmmm";
        case PhysicalBasis::M3:      return "m3";
        case PhysicalBasis::ZMirror: return "z-mirror";
        default:                     return "irreducible";
    }
}
