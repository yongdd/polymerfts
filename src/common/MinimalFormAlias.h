/**
 * @file MinimalFormAlias.h
 * @brief Exact search for the alias of a Fourier mode that minimizes the
 *        reciprocal-metric form (first-Brillouin-zone representative), and the
 *        loops over a periodic grid that use it.
 *
 * On a periodic grid a mode m is defined modulo the grid sizes N, and |k|² must
 * use the alias u = m + (a N1, b N2, c N3) of least form f(u) = uᵀ G u, which
 * is invariant under the point group. A window of one period around the
 * centered representative finds it only when the lattice of the periods is
 * reduced in the metric. On oblique cells with strongly anisotropic grid
 * spacing (e.g. 4x32 points on a 120-degree cell with equal lengths) it misses
 * the minimum, and the factors of k and -k then differ. The form of the centered
 * representative bounds the search instead: every alias with a form not above
 * it satisfies |u_d| <= sqrt(f(m) (G⁻¹)_dd), so enumerating that box is exact.
 * Within each coupled component, aliases are visited in the same (a, b, c)
 * order as the former one-period search.
 *
 * Numerically orthogonal components (normalized coupling at most 1e-12) are
 * minimized separately. Their bounds and tie tolerances use only their own
 * forms, so an arbitrarily short orthogonal axis cannot inflate a plane's
 * search or hide differences between its aliases in floating-point addition.
 * AliasSearch finds the components once per grid; the loops over a grid reuse it.
 * It also finds λ₁², the least form of a nonzero period vector of each component:
 * a mode with 4 f(m) < λ₁² is its own unique alias of least form (every other alias
 * u = m + N t has |u| ≥ λ₁ - |m| > |m|), so most modes need no search.
 * Within a component the box spans about abc/V times the ratio of the
 * component's grid spacings in periods along each axis; ComputationBox refuses
 * nearly flat oblique cells (V < 0.1 abc).
 *
 * Axes with N_d <= 1 or a zero diagonal metric (the padded axes of 1D and 2D
 * grids) are not searched.
 */

#ifndef MINIMAL_FORM_ALIAS_H_
#define MINIMAL_FORM_ALIAS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "Exception.h"

/// Reciprocal metric in the index space of the search.
struct IndexMetric
{
    double g11, g22, g33, g12, g13, g23;
};

/// Whether two reciprocal axes are coupled beyond orthogonal-angle roundoff.
/// Normalize by this pair's lengths: a short, unrelated third axis must not
/// hide an oblique plane. Separate divisions avoid forming gii * gjj.
inline bool reciprocal_axes_coupled(double gii, double gjj, double gij)
{
    return gii > 0.0 && gjj > 0.0 &&
        std::abs(gij) / std::sqrt(gii) / std::sqrt(gjj) > 1e-12;
}

inline bool index_metric_has_coupling(const IndexMetric& g)
{
    return reciprocal_axes_coupled(g.g11, g.g22, g.g12) ||
           reciprocal_axes_coupled(g.g11, g.g33, g.g13) ||
           reciprocal_axes_coupled(g.g22, g.g33, g.g23);
}

/// The reciprocal metric of a box ([G11, G12, G13, G22, G23, G33] of its dim
/// axes, ComputationBox::get_recip_metric()), padded to three axes at the end.
/// Padding with unsearched axes keeps the row-major index of every mode.
inline IndexMetric index_metric_of(const std::array<double, 6>& G, int dim)
{
    if (dim == 3)
        return {G[0], G[3], G[5], G[1], G[2], G[4]};
    if (dim == 2)
        return {G[0], G[3], 0.0, G[1], 0.0, 0.0};
    return {G[0], 0.0, 0.0, 0.0, 0.0, 0.0};
}

/// f(u) = uᵀ G u, evaluated in the order of the propagator solvers (|k|² = 4π² f)
struct IndexForm
{
    IndexMetric g;
    double operator()(double a, double b, double c) const
    {
        return g.g11 * a * a + g.g22 * b * b + g.g33 * c * c + 2.0 * (g.g12 * a * b + g.g13 * a * c + g.g23 * b * c);
    }
};

/// Centered (signed) index of loop index i on an axis of n points
inline int centered_index(int i, int n) { return (i > n / 2) ? i - n : i; }

/// Grid sizes of 1 to 3 axes, padded to three axes at the end like index_metric_of()
inline std::array<int, 3> padded_grid(const std::vector<int>& n)
{
    return {n[0], n.size() > 1 ? n[1] : 1, n.size() > 2 ? n[2] : 1};
}

namespace min_form_alias_detail
{
/// Connected components of the searched axes. Numerically orthogonal axes
/// have independent alias minima. Searching each component separately keeps
/// a large constant form on one axis out of another component's search bound
/// and tie tolerance (e.g. a thin z direction above an oblique xy plane).
inline int components(const std::array<int, 3>& n, const IndexMetric& g, std::array<int, 3>& masks)
{
    const double G[3][3] = {{g.g11, g.g12, g.g13}, {g.g12, g.g22, g.g23}, {g.g13, g.g23, g.g33}};
    int remaining = 0;
    for (int d = 0; d < 3; ++d)
        if (n[d] > 1 && G[d][d] > 0.0)
            remaining |= 1 << d;
    int count = 0;
    while (remaining != 0)
    {
        int mask = remaining & -remaining;
        int previous;
        do
        {
            previous = mask;
            for (int i = 0; i < 3; ++i)
                if (mask & (1 << i))
                    for (int j = 0; j < 3; ++j)
                        if ((remaining & (1 << j)) && reciprocal_axes_coupled(G[i][i], G[j][j], G[i][j]))
                            mask |= 1 << j;
        } while (mask != previous);
        masks[count++] = mask;
        remaining &= ~mask;
    }
    return count;
}

/// Ranges of the period shifts (a, b, c) that can hold an alias whose form does
/// not exceed f_center, the form of m itself.
inline void search_box(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g,
    double f_center, std::array<int, 3>& lo, std::array<int, 3>& hi)
{
    const double G[3][3] = {{g.g11, g.g12, g.g13}, {g.g12, g.g22, g.g23}, {g.g13, g.g23, g.g33}};
    int active[3];
    int n_active = 0;
    for (int d = 0; d < 3; ++d)
    {
        lo[d] = hi[d] = 0;
        if (n[d] > 1 && G[d][d] > 0.0)
            active[n_active++] = d;
    }
    if (n_active == 0 || !(f_center > 0.0))
        return;

    // Diagonal of the inverse of the metric restricted to the searched axes
    double inv_diag[3] = {0.0, 0.0, 0.0};
    if (n_active == 1)
    {
        inv_diag[0] = 1.0 / G[active[0]][active[0]];
    }
    else if (n_active == 2)
    {
        const double p = G[active[0]][active[0]], q = G[active[0]][active[1]], r = G[active[1]][active[1]];
        const double det = p * r - q * q;
        inv_diag[0] = r / det;
        inv_diag[1] = p / det;
    }
    else
    {
        const double det = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[1][2])
                         - G[0][1] * (G[0][1] * G[2][2] - G[1][2] * G[0][2])
                         + G[0][2] * (G[0][1] * G[1][2] - G[1][1] * G[0][2]);
        inv_diag[0] = (G[1][1] * G[2][2] - G[1][2] * G[1][2]) / det;
        inv_diag[1] = (G[0][0] * G[2][2] - G[0][2] * G[0][2]) / det;
        inv_diag[2] = (G[0][0] * G[1][1] - G[0][1] * G[0][1]) / det;
    }

    for (int i = 0; i < n_active; ++i)
    {
        const int d = active[i];
        // A cell of V >= 0.1 abc (ComputationBox) has a positive-definite metric with a
        // positive, finite inverse diagonal; a window of a few periods would not be exact
        if (!(inv_diag[i] > 0.0) || !std::isfinite(inv_diag[i]))
            throw_with_line_number("The reciprocal metric of the alias search is degenerate "
                "(not positive definite): the cell is flat.");
        // A small slack only adds candidates, so rounding cannot drop an alias
        const double reach = std::sqrt(f_center * inv_diag[i]) * (1.0 + 1e-9) + 1e-9;
        lo[d] = static_cast<int>(std::ceil((-reach - m[d]) / n[d]));
        hi[d] = static_cast<int>(std::floor((reach - m[d]) / n[d]));
        if (lo[d] > 0) lo[d] = 0;
        if (hi[d] < 0) hi[d] = 0;
    }
}

/// Replace m by its alias of least form on a grid of one coupled component; a mode whose
/// form is below `unique_below` (see unique_alias_bound) is its own least alias
template <typename Form>
inline void component_min_alias(
    std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g, Form&& form, double unique_below)
{
    const double f_center = form(m[0], m[1], m[2]);
    if (f_center < unique_below)
        return;
    std::array<int, 3> lo, hi;
    min_form_alias_detail::search_box(m, n, g, f_center, lo, hi);

    std::array<int, 3> best_rep = m;
    double best = f_center;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                if (a == 0 && b == 0 && c == 0)
                    continue;
                const int t1 = m[0] + a * n[0];
                const int t2 = m[1] + b * n[1];
                const int t3 = m[2] + c * n[2];
                const double f = form(t1, t2, t3);
                if (f < best)
                {
                    best = f;
                    best_rep = {t1, t2, t3};
                }
            }
    m = best_rep;
}

/**
 * @brief A quarter of λ₁², the least form of a nonzero period vector N t of the
 *        searched axes, reduced by 1e-9; zero without a searched axis.
 *
 * A mode m with f(m) below it is the unique alias of least form: every other alias
 * u = m + N t has |u| ≥ λ₁ - |m| > |m| (1 + 1e-9), so f(u) exceeds f(m) by far more
 * than the tie tolerances (1e-10) and the rounding of the forms.
 */
inline double unique_alias_bound(const std::array<int, 3>& n, const IndexMetric& g)
{
    const double G[3] = {g.g11, g.g22, g.g33};
    // The least form of a period along one axis bounds λ₁² from above
    double bound = 0.0;
    for (int d = 0; d < 3; ++d)
        if (n[d] > 1 && G[d] > 0.0)
        {
            const double f_axis = G[d] * n[d] * n[d];
            bound = (bound == 0.0) ? f_axis : std::min(bound, f_axis);
        }
    if (bound == 0.0)
        return 0.0;
    // Every period vector N t of a form not above the bound lies in the box of the aliases of mode 0
    const IndexForm form{g};
    std::array<int, 3> lo, hi;
    search_box({0, 0, 0}, n, g, bound, lo, hi);
    double least = bound;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
                if (a != 0 || b != 0 || c != 0)
                    least = std::min(least, form(static_cast<double>(a) * n[0], static_cast<double>(b) * n[1],
                                                 static_cast<double>(c) * n[2]));
    return 0.25 * least * (1.0 - 1e-9);
}
}  // namespace min_form_alias_detail

/// One coupled component of a grid's alias search: its axes (bit d of mask for
/// axis d), and the grid sizes and metric with the other axes not searched.
struct AliasComponent
{
    int mask = 0;
    std::array<int, 3> n{};
    IndexMetric g{};
    double unique_below = 0.0;   ///< modes of a smaller form are their own unique least alias (unique_alias_bound)

    /// The indices of m on this component's axes, zero elsewhere
    std::array<int, 3> part(const std::array<int, 3>& m) const
    {
        return {mask & 1 ? m[0] : 0, mask & 2 ? m[1] : 0, mask & 4 ? m[2] : 0};
    }
};

/// What the alias search needs of a grid, found once per grid rather than per
/// mode: whether its metric couples axes, and its coupled components.
struct AliasSearch
{
    std::array<int, 3> n;
    IndexMetric g;
    bool coupled;  ///< index_metric_has_coupling(g); otherwise the centered representative is minimal
    int n_components = 0;
    std::array<AliasComponent, 3> components{};

    AliasSearch(const std::array<int, 3>& n_, const IndexMetric& g_)
        : n(n_), g(g_), coupled(index_metric_has_coupling(g_))
    {
        std::array<int, 3> masks{};
        n_components = min_form_alias_detail::components(n, g, masks);
        for (int i = 0; i < n_components; ++i)
        {
            const int mask = masks[i];
            components[i] = {mask, {mask & 1 ? n[0] : 1, mask & 2 ? n[1] : 1, mask & 4 ? n[2] : 1},
                             {mask & 1 ? g.g11 : 0.0, mask & 2 ? g.g22 : 0.0, mask & 4 ? g.g33 : 0.0,
                              (mask & 3) == 3 ? g.g12 : 0.0, (mask & 5) == 5 ? g.g13 : 0.0,
                              (mask & 6) == 6 ? g.g23 : 0.0}};
            components[i].unique_below = min_form_alias_detail::unique_alias_bound(components[i].n, components[i].g);
        }
    }

    /// unique_below of the whole grid when it has one component (zero without one)
    double grid_unique_below() const { return n_components == 1 ? components[0].unique_below : 0.0; }
};

/**
 * @brief Replace m by its alias of least form.
 *
 * `form(u1, u2, u3)` evaluates the form; callers pass their own expression so
 * that its rounding is unchanged. In each coupled component the aliases are
 * visited in (a, b, c) order and replace the current choice only when their
 * form is strictly smaller, so its centered representative is kept on ties.
 */
template <typename Form>
inline void min_form_alias_rep(std::array<int, 3>& m, const AliasSearch& search, Form&& form)
{
    if (search.n_components <= 1)
    {
        min_form_alias_detail::component_min_alias(m, search.n, search.g, form, search.grid_unique_below());
        return;
    }
    for (int i = 0; i < search.n_components; ++i)
    {
        const AliasComponent& component = search.components[i];
        std::array<int, 3> sub_m = component.part(m);
        min_form_alias_detail::component_min_alias(sub_m, component.n, component.g, form, component.unique_below);
        for (int d = 0; d < 3; ++d)
            if (component.mask & (1 << d))
                m[d] = sub_m[d];
    }
}

/// min_form_alias_rep for a single mode of a grid
template <typename Form>
inline void min_form_alias_rep(
    std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g, Form&& form)
{
    min_form_alias_rep(m, AliasSearch(n, g), form);
}

/**
 * @brief Visit, in (a, b, c) order, every least-form alias of m; returns the
 *        number visited. Each independent metric component has tolerance
 *        rel_tol * its minimum, so a large orthogonal contribution cannot
 *        turn nonminimal aliases of another component into ties.
 */
template <typename Form, typename Visit>
inline int for_each_min_form_alias(
    const std::array<int, 3>& m, const AliasSearch& search, Form&& form, double rel_tol, Visit&& visit)
{
    const std::array<int, 3>& n = search.n;
    if (search.n_components > 1)
    {
        std::array<int, 3> lo{0, 0, 0}, hi{0, 0, 0};
        std::array<double, 3> limits{};
        for (int i = 0; i < search.n_components; ++i)
        {
            const AliasComponent& component = search.components[i];
            std::array<int, 3> sub_m = component.part(m), sub_lo{0, 0, 0}, sub_hi{0, 0, 0};
            const double sub_center = form(sub_m[0], sub_m[1], sub_m[2]);
            if (!(sub_center < component.unique_below))   // otherwise sub_m is its component's only tie
            {
                min_form_alias_detail::search_box(sub_m, component.n, component.g, sub_center, sub_lo, sub_hi);
                min_form_alias_detail::component_min_alias(sub_m, component.n, component.g, form, 0.0);
            }
            const double best = form(sub_m[0], sub_m[1], sub_m[2]);
            limits[i] = best + rel_tol * best;
            for (int d = 0; d < 3; ++d)
                if (component.mask & (1 << d))
                {
                    lo[d] = sub_lo[d];
                    hi[d] = sub_hi[d];
                }
        }
        int count = 0;
        for (int a = lo[0]; a <= hi[0]; ++a)
            for (int b = lo[1]; b <= hi[1]; ++b)
                for (int c = lo[2]; c <= hi[2]; ++c)
                {
                    const int t1 = m[0] + a * n[0], t2 = m[1] + b * n[1], t3 = m[2] + c * n[2];
                    bool tied = true;
                    for (int i = 0; i < search.n_components && tied; ++i)
                    {
                        const int mask = search.components[i].mask;
                        tied = form(mask & 1 ? t1 : 0, mask & 2 ? t2 : 0, mask & 4 ? t3 : 0) <= limits[i];
                    }
                    if (tied)
                    {
                        visit(t1, t2, t3);
                        ++count;
                    }
                }
        return count;
    }
    const double f_center = form(m[0], m[1], m[2]);
    if (f_center < search.grid_unique_below())
    {
        visit(m[0], m[1], m[2]);
        return 1;
    }
    std::array<int, 3> lo, hi;
    min_form_alias_detail::search_box(m, n, search.g, f_center, lo, hi);

    double best = f_center;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                const double f = form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]);
                if (f < best)
                    best = f;
            }

    const double tol = rel_tol * best;
    int count = 0;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                const int t1 = m[0] + a * n[0];
                const int t2 = m[1] + b * n[1];
                const int t3 = m[2] + c * n[2];
                if (form(t1, t2, t3) > best + tol)
                    continue;
                visit(t1, t2, t3);
                ++count;
            }
    return count;
}

/// for_each_min_form_alias for a single mode of a grid
template <typename Form, typename Visit>
inline int for_each_min_form_alias(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g,
    Form&& form, double rel_tol, Visit&& visit)
{
    return for_each_min_form_alias(m, AliasSearch(n, g), form, rel_tol, visit);
}

/**
 * @brief Visit every mode of a periodic grid of n points with its alias of
 *        least form: visit(index, u), in the row-major order of `shape`.
 *
 * `shape` is n or, for the r2c half spectrum, n with its last axis of more
 * than one point cut to n/2 + 1; the modes have centered indices. Cells
 * without coupled axes keep the centered representative, which is minimal
 * there. `form` is the caller's expression, as for min_form_alias_rep.
 */
template <typename Form, typename Visit>
inline void for_each_mode_min_alias(
    const std::array<int, 3>& n, const std::array<int, 3>& shape, const IndexMetric& g,
    Form&& form, Visit&& visit)
{
    const AliasSearch search(n, g);
    long long index = 0;
    for (int i = 0; i < shape[0]; ++i)
        for (int j = 0; j < shape[1]; ++j)
            for (int k = 0; k < shape[2]; ++k)
            {
                std::array<int, 3> u = {centered_index(i, n[0]), centered_index(j, n[1]), centered_index(k, n[2])};
                if (search.coupled)
                    min_form_alias_rep(u, search, form);
                visit(index++, u);
            }
}

/**
 * @brief <v_a v_b> with v = scale · G u, averaged over the aliases u of m tied
 *        for the least form, in the order xx, yy, zz, xy, xz, yz.
 *
 * Lattice derivatives of |k|² differ between tied aliases: on the Brillouin-zone
 * boundary of an oblique cell and on the Nyquist planes (m_d = N_d/2) of any
 * cell. Their average is the symmetric derivative of the minimum, invariant
 * under the point group. Without coupled axes the ties are those of the
 * Nyquist planes, where -N_d/2 ties with N_d/2.
 */
inline std::array<double, 6> tied_alias_moments(const std::array<int, 3>& m, const AliasSearch& search, double scale)
{
    const std::array<int, 3>& n = search.n;
    const IndexMetric& g = search.g;
    std::array<double, 6> sum{};
    auto add = [&](int a, int b, int c) {
        const double v1 = scale * (g.g11 * a + g.g12 * b + g.g13 * c);
        const double v2 = scale * (g.g12 * a + g.g22 * b + g.g23 * c);
        const double v3 = scale * (g.g13 * a + g.g23 * b + g.g33 * c);
        sum[0] += v1 * v1; sum[1] += v2 * v2; sum[2] += v3 * v3;
        sum[3] += v1 * v2; sum[4] += v1 * v3; sum[5] += v2 * v3;
    };
    int count = 0;
    if (search.coupled)
    {
        count = for_each_min_form_alias(m, search, IndexForm{g}, 1e-10, add);
    }
    else
    {
        const int s[3] = {n[0] > 1 && 2 * m[0] == n[0], n[1] > 1 && 2 * m[1] == n[1], n[2] > 1 && 2 * m[2] == n[2]};
        for (int a = 0; a <= s[0]; ++a)
            for (int b = 0; b <= s[1]; ++b)
                for (int c = 0; c <= s[2]; ++c, ++count)
                    add(m[0] - a * n[0], m[1] - b * n[1], m[2] - c * n[2]);
    }
    for (double& value : sum)
        value /= count;
    return sum;
}

/**
 * @brief |k|² = 4π² f(u) of every mode of a periodic grid of n points, u its alias
 *        of least form, in the row-major order of `shape` (see for_each_mode_min_alias):
 *        the wavenumbers of the propagator solvers (Pseudo).
 */
inline void alias_k_squared(
    const std::array<int, 3>& n, const std::array<int, 3>& shape, const IndexMetric& g, double* k_squared)
{
    const double PI = std::numbers::pi;
    const IndexForm form{g};
    for_each_mode_min_alias(n, shape, g, form,
        [&](long long index, const std::array<int, 3>& u) { k_squared[index] = 4.0 * PI * PI * form(u[0], u[1], u[2]); });
}

/**
 * @brief tied_alias_moments (scale 1) of every mode of a periodic grid of n points, in the
 *        row-major order of `shape`: dim × dim arrays, <v_a v_b> of mode i at
 *        moments[(a dim + b) n_modes + i], n_modes the number of modes of `shape`.
 */
inline void alias_moments(
    const std::array<int, 3>& n, const std::array<int, 3>& shape, const IndexMetric& g, int dim, double* moments)
{
    const AliasSearch search(n, g);
    const long long n_modes = static_cast<long long>(shape[0]) * shape[1] * shape[2];
    long long index = 0;
    for (int i = 0; i < shape[0]; ++i)
        for (int j = 0; j < shape[1]; ++j)
            for (int k = 0; k < shape[2]; ++k, ++index)
            {
                const std::array<double, 6> v = tied_alias_moments(
                    {centered_index(i, n[0]), centered_index(j, n[1]), centered_index(k, n[2])}, search, 1.0);
                const double product[3][3] = {{v[0], v[3], v[4]}, {v[3], v[1], v[5]}, {v[4], v[5], v[2]}};
                for (int a = 0; a < dim; ++a)
                    for (int b = 0; b < dim; ++b)
                        moments[(a * dim + b) * n_modes + index] = product[a][b];
            }
}

#endif  // MINIMAL_FORM_ALIAS_H_
