/**
 * @file Pseudo.cpp
 * @brief The Fourier-space tables of the pseudo-spectral solvers for one cell.
 *
 * Supports all boundary conditions and non-orthogonal crystal systems:
 * - PERIODIC: FFT with recip_metric for non-orthogonal lattices
 * - REFLECTING (DCT): k = π*n/L, n = 0, 1, ..., N-1
 * - ABSORBING (DST): k = π*(n+1)/L, n = 0, 1, ..., N-1
 *
 * The axes of 1D and 2D boxes are padded to three at the end (padded_grid), which
 * keeps the row-major index of every mode.
 *
 * @see Pseudo.h for class documentation
 */

#include <algorithm>
#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <complex>
#include <utility>
#include "Pseudo.h"
#include "MinimalFormAlias.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
Pseudo<T>::Pseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx,
    std::map<int, double> ds_values)
    : bond_lengths(std::move(bond_lengths)), bc(std::move(bc)), nx(std::move(nx)), ds_values(std::move(ds_values))
{
    int total_grid = 1;
    for (int n : this->nx)
        total_grid *= n;
    total_complex_grid = complex_grid_size(total_grid, this->nx.back(), is_all_periodic());

    // The Boltzmann tables of every contour step and monomer type follow at the first update()
    for (const auto& [ds_idx, local_ds] : this->ds_values)
        if (ds_idx < 0)
            throw_with_line_number("ds_index must be >= 0, got " + std::to_string(ds_idx));
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
Pseudo<T>::~Pseudo() = default;   // the arrays are std::vector members

//------------------------------------------------------------------------------
// Check if all BCs are periodic
//------------------------------------------------------------------------------
template <typename T>
bool Pseudo<T>::is_all_periodic() const
{
    for (const auto& b : bc)
    {
        if (b != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Getters
//------------------------------------------------------------------------------
template <typename T>
int Pseudo<T>::get_total_complex_grid() const
{
    return total_complex_grid;
}

template <typename T>
double* Pseudo<T>::get_boltz_bond(const std::string& monomer_type, int ds_index)
{
    // Called from the propagator threads: find(), not operator[] (which may insert)
    auto ds_entry = boltz_bond.find(ds_index);
    if (ds_entry == boltz_bond.end())
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " has no Boltzmann bond table (none before the first update()).");
    auto entry = ds_entry->second.find(monomer_type);
    if (entry == ds_entry->second.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in boltz_bond[" + std::to_string(ds_index) + "].");
    return entry->second.data();
}

template <typename T>
double* Pseudo<T>::get_boltz_bond_half(const std::string& monomer_type, int ds_index)
{
    // Called from the propagator threads: find(), not operator[] (which may insert)
    auto ds_entry = boltz_bond_half.find(ds_index);
    if (ds_entry == boltz_bond_half.end())
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " has no Boltzmann bond table (none before the first update()).");
    auto entry = ds_entry->second.find(monomer_type);
    if (entry == ds_entry->second.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in boltz_bond_half[" + std::to_string(ds_index) + "].");
    return entry->second.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_x()
{
    return fourier_basis_x.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_y()
{
    return fourier_basis_y.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_z()
{
    return fourier_basis_z.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_xy()
{
    return fourier_basis_xy.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_xz()
{
    return fourier_basis_xz.data();
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_yz()
{
    return fourier_basis_yz.data();
}

template <typename T>
const int* Pseudo<T>::get_negative_frequency_mapping()
{
    return negative_k_idx.empty() ? nullptr : negative_k_idx.data();
}

//------------------------------------------------------------------------------
// Size of the spectrum
//------------------------------------------------------------------------------
template <typename T>
int Pseudo<T>::complex_grid_size(int n_grid, int n_last, bool all_periodic)
{
    // Periodic BC: the r2c FFT of a real field keeps n/2 + 1 values along the last axis
    if constexpr (std::is_same<T, double>::value)
    {
        if (all_periodic)
            return n_grid / n_last * (n_last / 2 + 1);
    }
    return n_grid;   // c2c FFT (complex fields), or DCT/DST coefficients (non-periodic BC)
}

//------------------------------------------------------------------------------
// The tables of a cell
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update(const std::vector<double>& dx, const std::array<double, 6>& recip_metric,
                       bool with_stress_basis)
{
    // The stress basis while it is computed, of the box axes: v_d² of each axis and, for
    // periodic BC (walls need an orthogonal cell), v_a v_b of each pair of axes
    const int DIM = nx.size();
    const int n_pairs = is_all_periodic() ? DIM * (DIM - 1) / 2 : 0;
    auto keep_if = [&](std::vector<double>& table, bool used) {
        if (used)
            table.resize(total_complex_grid);
        else
            std::vector<double>().swap(table);
    };
    std::vector<double>* diagonal[3] = {&fourier_basis_x, &fourier_basis_y, &fourier_basis_z};
    std::vector<double>* pairs[3] = {&fourier_basis_xy, &fourier_basis_xz, &fourier_basis_yz};
    for (int i = 0; i < 3; ++i)
    {
        keep_if(*diagonal[i], with_stress_basis && i < DIM);
        keep_if(*pairs[i], with_stress_basis && i < n_pairs);
    }
    if (with_stress_basis)
        update_negative_frequency_mapping();
    else
        std::vector<int>().swap(negative_k_idx);

    // The stress basis, and f = |k|²/(4π²) of every mode in the layout of the Boltzmann factors
    const std::vector<double> forms = is_all_periodic() ? update_periodic(recip_metric, with_stress_basis)
                                                         : update_nonperiodic(dx, with_stress_basis);

    // The Boltzmann tables of every contour step and monomer type: exp(-b² |k|² ds / 6), |k|² = 4π² f
    const double PI = std::numbers::pi;
    for (const auto& [ds_idx, local_ds] : ds_values)
    {
        for (const auto& [monomer_type, bond_length] : bond_lengths)
        {
            std::vector<double>& full = boltz_bond[ds_idx][monomer_type];
            std::vector<double>& half = boltz_bond_half[ds_idx][monomer_type];
            full.resize(total_complex_grid);
            half.resize(total_complex_grid);
            double bond_length_sq = bond_length * bond_length;
            double prefactor = -bond_length_sq * 4.0 * PI * PI * local_ds / 6.0;
            for (size_t idx = 0; idx < forms.size(); ++idx)
            {
                const double mag_q2 = prefactor * forms[idx];
                full[idx] = std::exp(mag_q2);
                half[idx] = std::exp(mag_q2 / 2.0);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Periodic BC (with recip_metric support)
//------------------------------------------------------------------------------
// Oblique cells use the alias representative of least reciprocal-metric form
// (first-Brillouin-zone representative; see MinimalFormAlias.h). The
// componentwise principal-zone choice (|m_d| <= N_d/2) is NOT invariant under
// point-group operations that mix axes with a nonzero metric cross term: e.g.
// the hexagonal rotation (kx,ky) -> (kx+ky,-kx) can push kx+ky out of the
// zone, and re-aliasing changes kx^2 + kx*ky + ky^2. The minimal-form
// representative is group-invariant: symmetry operations permute alias classes
// and preserve the form, so the minimum over the class is preserved. This makes
// the pseudo-spectral evolution operator exactly symmetric under the crystal
// point group (required for space-group reduced-basis computations to match
// full-grid computations). For orthogonal cells the form is separable and the
// componentwise representative is already minimal, so only cells with a
// nonzero off-diagonal metric take this path (keeping orthogonal results
// bit-exact).

namespace {
/**
 * @brief The periodic stress basis: v⊗v of the deformation vector v = 2π g⁻¹ m of each
 *        mode m, times the weight of the mode in the sum over the spectrum.
 *
 * `tables` holds the table of each component of v⊗v (x, y, z, xy, xz, yz; null for the
 * components a box does not keep):
 *   v₁ = 2π (G₁₁m₁ + G₁₂m₂ + G₁₃m₃)
 *   v₂ = 2π (G₁₂m₁ + G₂₂m₂ + G₂₃m₃)
 *   v₃ = 2π (G₁₃m₁ + G₂₃m₂ + G₃₃m₃)
 * The modes are those of `shape` (the r2c half spectrum for real fields, whose
 * interior modes along the last box axis stand for a conjugate pair: weight 2).
 *
 * @see docs/theory/StressTensor.md for derivation
 */
template <typename T>
void fill_periodic_stress_basis(
    const std::array<double*, 6>& tables, const std::array<int, 3>& grid,
    const std::array<int, 3>& shape, const IndexMetric& G, int last_axis)
{
    // The Boltzmann factor depends only on the minimum of the form over the aliases
    // m + N t, but v⊗v differs between representatives that tie for the minimum: the
    // Brillouin-zone boundary of an oblique cell (e.g. 2 m1 + m2 = N in a hexagonal
    // cell) and the Nyquist planes (m = N/2 and -N/2) of any cell. The products are
    // averaged over the tied representatives (tied_alias_moments), which keeps the
    // stress invariant under the point group (sigma_a = sigma_b in a hexagonal cell);
    // for two tied representatives the average is the central difference of the minimum.
    // The same moments give FourierTransform::get_alias_moments().
    for_each_mode_tied_moments(grid, shape, G, 2.0 * std::numbers::pi,
        [&](long long idx, const std::array<int, 3>& m, const std::array<double, 6>& vv) {
            double weight = 1.0;
            if constexpr (std::is_same<T, double>::value)
                weight = (m[last_axis] != 0 && 2 * m[last_axis] != grid[last_axis]) ? 2.0 : 1.0;
            for (int c = 0; c < 6; ++c)
                if (tables[c] != nullptr)
                    tables[c][idx] = weight * vv[c];
        });
}

/// Index i of the wavenumber of mode m (0 <= m < n) of an axis with separable
/// wavenumbers: k = πi/L on walls, i = m for the cosine series (REFLECTING) and
/// m + 1 for the sine series (ABSORBING), and k = 2πi/L on periodic axes, i the
/// signed index of the FFT
int separable_wavenumber_index(int m, int n, BoundaryCondition bc)
{
    if (bc == BoundaryCondition::PERIODIC)
        return centered_index(m, n);
    return (bc == BoundaryCondition::ABSORBING) ? m + 1 : m;
}
}  // namespace

std::vector<double> separable_k_squared(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                                        const std::vector<double>& lx)
{
    // The wavenumbers of each axis; the axes padded to three at the end have k = 0
    const double PI = std::numbers::pi;
    const int dim = nx.size();
    const std::array<int, 3> tnx = padded_grid(nx);
    std::array<std::vector<double>, 3> k_axis;
    for (int d = 0; d < 3; ++d)
    {
        k_axis[d].assign(tnx[d], 0.0);
        if (d >= dim)
            continue;
        const double scale = (bc[d] == BoundaryCondition::PERIODIC) ? 2.0 * PI : PI;
        for (int m = 0; m < tnx[d]; ++m)
            k_axis[d][m] = scale * separable_wavenumber_index(m, tnx[d], bc[d]) / lx[d];
    }
    std::vector<double> k_sq(static_cast<size_t>(tnx[0]) * tnx[1] * tnx[2]);
    for (int i = 0; i < tnx[0]; ++i)
        for (int j = 0; j < tnx[1]; ++j)
            for (int k = 0; k < tnx[2]; ++k)
                k_sq[(static_cast<size_t>(i) * tnx[1] + j) * tnx[2] + k] =
                    k_axis[0][i] * k_axis[0][i] + k_axis[1][j] * k_axis[1][j] + k_axis[2][k] * k_axis[2][k];
    return k_sq;
}

template <typename T>
std::vector<double> Pseudo<T>::update_periodic(const std::array<double, 6>& recip_metric, bool with_stress_basis)
{
    const int DIM = nx.size();

    // The modes of the spectrum: the r2c half spectrum for real fields
    std::vector<int> shape = nx;
    if constexpr (std::is_same<T, double>::value)
        shape.back() = shape.back() / 2 + 1;
    const std::array<int, 3> grid = padded_grid(nx), modes = padded_grid(shape);
    const IndexMetric G = index_metric_of(recip_metric, DIM);

    if (with_stress_basis)
    {
        // The tables update() keeps
        std::vector<double>* basis[6] = {&fourier_basis_x, &fourier_basis_y, &fourier_basis_z,
                                         &fourier_basis_xy, &fourier_basis_xz, &fourier_basis_yz};
        std::array<double*, 6> tables{};
        for (int c = 0; c < 6; ++c)
            tables[c] = basis[c]->empty() ? nullptr : basis[c]->data();
        fill_periodic_stress_basis<T>(tables, grid, modes, G, DIM - 1);
    }

    if (ds_values.empty())
        return {};

    // f = mᵀ G m of the alias of least form (first-Brillouin-zone representative;
    // see MinimalFormAlias.h), |k|² = 4π² f. One search serves every contour step
    // and monomer type.
    const IndexForm form{G};
    std::vector<double> forms(total_complex_grid);
    for_each_mode_min_alias(grid, modes, G, form,
        [&](long long idx, const std::array<int, 3>& u) { forms[idx] = form(u[0], u[1], u[2]); });
    return forms;
}

//------------------------------------------------------------------------------
// Reflecting and absorbing BC (orthogonal cells, no cross terms)
//------------------------------------------------------------------------------
template <typename T>
std::vector<double> Pseudo<T>::update_nonperiodic(const std::vector<double>& dx, bool with_stress_basis)
{
    const double PI = std::numbers::pi;
    const int DIM = nx.size();
    std::vector<double> lx(DIM);
    for (int d = 0; d < DIM; ++d)
        lx[d] = nx[d] * dx[d];

    if (with_stress_basis)
    {
        // Deformation vector factors v² = k²/L² (units 1/L⁴) of the wavenumbers
        // k = πi/L of each wall axis (separable_k_squared), consistent with the
        // periodic-BC path where v = 2π g⁻¹ m and g⁻¹ = 1/L² for orthogonal boxes.
        //
        // Per-mode Parseval weight, folded into the basis tables (mirroring the
        // periodic path, which folds the r2c conjugate-pair factor 2 into the
        // tables). With the textbook DCT-II/DST-II coefficients F(m) = Σ_j f_j φ_m(x_j)
        // used by the FFT<T> real interface, the discrete Parseval relation per
        // dimension is Σ_j f_j g_j = (1/n) Σ_m w(m) F(m) G(m) with
        //   DCT-II (reflecting): w = 1 for m = 0,      w = 2 otherwise
        //   DST-II (absorbing):  w = 1 for k = m+1 = n, w = 2 otherwise
        // (the special basis functions cos(0) and sin(πn x/L) have squared norm n
        // on the cell-centered grid instead of n/2). Composed dimensions multiply
        // their weights; the axes padded to three at the end have one mode of weight 1.
        const std::array<int, 3> tnx = padded_grid(nx);
        std::array<std::vector<double>, 3> weight, v_sq;
        for (int d = 0; d < 3; ++d)
        {
            weight[d].assign(tnx[d], 1.0);
            if (d >= DIM)
                continue;
            const double xfactor = PI * PI / (lx[d] * lx[d] * lx[d] * lx[d]);
            const int special = (bc[d] == BoundaryCondition::REFLECTING) ? 0 : nx[d] - 1;
            for (int m = 0; m < nx[d]; ++m)
            {
                const int i = separable_wavenumber_index(m, nx[d], bc[d]);
                weight[d][m] = (m == special) ? 1.0 : 2.0;
                v_sq[d].push_back(i * i * xfactor);
            }
        }

        // The tables of the box axes (update())
        double* basis[3] = {fourier_basis_x.data(), fourier_basis_y.data(), fourier_basis_z.data()};
        for (int i = 0; i < tnx[0]; ++i)
            for (int j = 0; j < tnx[1]; ++j)
                for (int k = 0; k < tnx[2]; ++k)
                {
                    const int m[3] = {i, j, k};
                    const double w = weight[0][i] * weight[1][j] * weight[2][k];
                    const size_t idx = (static_cast<size_t>(i) * tnx[1] + j) * tnx[2] + k;
                    for (int d = 0; d < DIM; ++d)
                        basis[d][idx] = w * v_sq[d][m[d]];
                }
    }

    if (ds_values.empty())
        return {};

    // f = k²/(4π²) of the separable wavenumbers
    std::vector<double> forms = separable_k_squared(nx, bc, lx);
    const double four_pi_sq = 4.0 * PI * PI;
    for (double& f : forms)
        f /= four_pi_sq;
    return forms;
}

//------------------------------------------------------------------------------
// Negative frequency mapping (only for periodic BC with complex fields)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_negative_frequency_mapping() {}

template <>
void Pseudo<std::complex<double>>::update_negative_frequency_mapping()
{
    // A map of the grid, made once
    if (!is_all_periodic() || !negative_k_idx.empty())
        return;
    negative_k_idx.resize(total_complex_grid);

    const std::array<int, 3> n = padded_grid(nx);
    int idx = 0;
    for (int i = 0; i < n[0]; i++)
    {
        const int i_neg = (i == 0) ? 0 : n[0] - i;
        for (int j = 0; j < n[1]; j++)
        {
            const int j_neg = (j == 0) ? 0 : n[1] - j;
            for (int k = 0; k < n[2]; k++)
            {
                const int k_neg = (k == 0) ? 0 : n[2] - k;
                negative_k_idx[idx++] = (i_neg * n[1] + j_neg) * n[2] + k_neg;
            }
        }
    }
}

// Explicit template instantiation
template class Pseudo<double>;
template class Pseudo<std::complex<double>>;
