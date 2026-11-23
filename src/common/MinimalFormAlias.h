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
 * The least alias and the tied aliases of a mode are those of its parts on each
 * component, so the loops over a grid of several components (a cell with
 * α = β = 90°: the xy plane and z) search each component's modes once,
 * N1 N2 + N3 searches instead of N1 N2 N3, and combine them per mode.
 * AliasSearch also finds λ₁², the least form of a nonzero period vector of each
 * component: a mode with 4 f(m) < λ₁² is its own unique alias of least form (every
 * other alias u = m + N t has |u| ≥ λ₁ - |m| > |m|), so most modes need no search.
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

/// Σ_d strides[d] (m_d mod n_d): the row-major position of the centered mode m (|m_d| < n_d)
inline long long mode_position(const std::array<int, 3>& m, const std::array<int, 3>& n,
                               const std::array<long long, 3>& strides)
{
    long long position = 0;
    for (int d = 0; d < 3; ++d)
        position += strides[d] * (m[d] < 0 ? m[d] + n[d] : m[d]);
    return position;
}

/// Grid sizes of 1 to 3 axes, padded to three axes at the end like index_metric_of()
inline std::array<int, 3> padded_grid(const std::vector<int>& n)
{
    return {n[0], n.size() > 1 ? n[1] : 1, n.size() > 2 ? n[2] : 1};
}

/// Visit the centered modes m of a periodic grid of n points: visit(index, m) in the
/// row-major order of `shape` (n or, for the r2c half spectrum, n with its last axis
/// of more than one point cut to n/2 + 1)
template <typename Visit>
inline void for_each_centered_mode(const std::array<int, 3>& n, const std::array<int, 3>& shape, Visit&& visit)
{
    long long index = 0;
    for (int i = 0; i < shape[0]; ++i)
        for (int j = 0; j < shape[1]; ++j)
            for (int k = 0; k < shape[2]; ++k)
                visit(index++, std::array<int, 3>{centered_index(i, n[0]), centered_index(j, n[1]),
                                                  centered_index(k, n[2])});
}

namespace min_form_alias_detail
{
/// Relative difference of the forms of aliases tied for the least form (tied_alias_moments)
inline constexpr double tie_tolerance = 1e-10;

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

/// What search_box() needs of a grid's metric, found once per grid: the searched
/// axes and the diagonal of the inverse of the metric restricted to them.
struct BoxMetric
{
    int n_active = 0;
    std::array<int, 3> active{};
    std::array<double, 3> inv_diag{};
};

inline BoxMetric box_metric(const std::array<int, 3>& n, const IndexMetric& g)
{
    const double G[3][3] = {{g.g11, g.g12, g.g13}, {g.g12, g.g22, g.g23}, {g.g13, g.g23, g.g33}};
    BoxMetric box;
    for (int d = 0; d < 3; ++d)
        if (n[d] > 1 && G[d][d] > 0.0)
            box.active[box.n_active++] = d;
    const std::array<int, 3>& a = box.active;
    if (box.n_active == 1)
    {
        box.inv_diag[0] = 1.0 / G[a[0]][a[0]];
    }
    else if (box.n_active == 2)
    {
        const double p = G[a[0]][a[0]], q = G[a[0]][a[1]], r = G[a[1]][a[1]];
        const double det = p * r - q * q;
        box.inv_diag[0] = r / det;
        box.inv_diag[1] = p / det;
    }
    else if (box.n_active == 3)
    {
        const double det = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[1][2])
                         - G[0][1] * (G[0][1] * G[2][2] - G[1][2] * G[0][2])
                         + G[0][2] * (G[0][1] * G[1][2] - G[1][1] * G[0][2]);
        box.inv_diag[0] = (G[1][1] * G[2][2] - G[1][2] * G[1][2]) / det;
        box.inv_diag[1] = (G[0][0] * G[2][2] - G[0][2] * G[0][2]) / det;
        box.inv_diag[2] = (G[0][0] * G[1][1] - G[0][1] * G[0][1]) / det;
    }
    // A cell of V >= 0.1 abc (ComputationBox) has a positive-definite metric with a
    // positive, finite inverse diagonal; a window of a few periods would not be exact
    for (int i = 0; i < box.n_active; ++i)
        if (!(box.inv_diag[i] > 0.0) || !std::isfinite(box.inv_diag[i]))
            throw_with_line_number("The reciprocal metric of the alias search is degenerate "
                "(not positive definite): the cell is flat.");
    return box;
}

/// Ranges of the period shifts (a, b, c) that can hold an alias whose form does
/// not exceed f_center, the form of m itself.
inline void search_box(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const BoxMetric& box,
    double f_center, std::array<int, 3>& lo, std::array<int, 3>& hi)
{
    lo = {0, 0, 0};
    hi = {0, 0, 0};
    if (!(f_center > 0.0))
        return;
    for (int i = 0; i < box.n_active; ++i)
    {
        const int d = box.active[i];
        // A small slack only adds candidates, so rounding cannot drop an alias
        const double reach = std::sqrt(f_center * box.inv_diag[i]) * (1.0 + 1e-9) + 1e-9;
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
    std::array<int, 3>& m, const std::array<int, 3>& n, const BoxMetric& box, Form&& form, double unique_below)
{
    const double f_center = form(m[0], m[1], m[2]);
    if (f_center < unique_below)
        return;
    std::array<int, 3> lo, hi;
    search_box(m, n, box, f_center, lo, hi);

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
inline double unique_alias_bound(const std::array<int, 3>& n, const IndexMetric& g, const BoxMetric& box)
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
    search_box({0, 0, 0}, n, box, bound, lo, hi);
    double least = bound;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
                if (a != 0 || b != 0 || c != 0)
                    least = std::min(least, form(static_cast<double>(a) * n[0], static_cast<double>(b) * n[1],
                                                 static_cast<double>(c) * n[2]));
    return 0.25 * least * (1.0 - 1e-9);
}

/// <v_a v_b> with v = scale · G u, averaged over the aliases u added
struct MomentSum
{
    const IndexMetric& g;
    double scale;
    std::array<double, 6> sum{};
    int count = 0;

    void add(int a, int b, int c)
    {
        const double v1 = scale * (g.g11 * a + g.g12 * b + g.g13 * c);
        const double v2 = scale * (g.g12 * a + g.g22 * b + g.g23 * c);
        const double v3 = scale * (g.g13 * a + g.g23 * b + g.g33 * c);
        sum[0] += v1 * v1; sum[1] += v2 * v2; sum[2] += v3 * v3;
        sum[3] += v1 * v2; sum[4] += v1 * v3; sum[5] += v2 * v3;
        ++count;
    }

    std::array<double, 6> mean() const
    {
        std::array<double, 6> result = sum;
        if (count > 1)   // most modes have a single alias of least form
            for (double& value : result)
                value /= count;
        return result;
    }
};
}  // namespace min_form_alias_detail

/// One coupled component of a grid's alias search: its axes (bit d of mask for
/// axis d), and the grid sizes and metric with the other axes not searched.
struct AliasComponent
{
    int mask = 0;
    std::array<int, 3> n{};
    IndexMetric g{};
    min_form_alias_detail::BoxMetric box{};   ///< what search_box() needs of n and g
    double unique_below = 0.0;   ///< modes of a smaller form are their own unique least alias (unique_alias_bound)

    /// The indices of m on this component's axes, zero elsewhere
    std::array<int, 3> part(const std::array<int, 3>& m) const
    {
        return {mask & 1 ? m[0] : 0, mask & 2 ? m[1] : 0, mask & 4 ? m[2] : 0};
    }

    /// Visit this component's modes of a grid of `shape` (the parts of its centered modes) in row-major order
    template <typename Visit>
    void for_each_mode(const std::array<int, 3>& shape, Visit&& visit) const
    {
        const std::array<int, 3> sub_shape = {mask & 1 ? shape[0] : 1, mask & 2 ? shape[1] : 1, mask & 4 ? shape[2] : 1};
        for_each_centered_mode(n, sub_shape, [&](long long, const std::array<int, 3>& sub_m) { visit(sub_m); });
    }

    /// Strides of the position of a mode's part in the order of for_each_mode(shape): the
    /// part of the centered mode m is at mode_position(m, n, strides), zero off this component
    std::array<long long, 3> mode_strides(const std::array<int, 3>& shape) const
    {
        std::array<long long, 3> strides{0, 0, 0};
        long long stride = 1;
        for (int d = 2; d >= 0; --d)
            if (mask & (1 << d))
            {
                strides[d] = stride;
                stride *= shape[d];
            }
        return strides;
    }

    /// Visit the period shifts (a, b, c) of the aliases of sub_m (a mode of this component) tied
    /// for its least form, in (a, b, c) order: those within rel_tol times the least form.
    /// Returns the number visited.
    template <typename Form, typename Visit>
    int for_each_tie(const std::array<int, 3>& sub_m, Form&& form, double rel_tol, Visit&& visit) const
    {
        const double f_center = form(sub_m[0], sub_m[1], sub_m[2]);
        std::array<int, 3> lo{0, 0, 0}, hi{0, 0, 0};
        if (!(f_center < unique_below))   // otherwise sub_m is its own unique least alias
            min_form_alias_detail::search_box(sub_m, n, box, f_center, lo, hi);
        if (lo == hi)
        {
            visit(0, 0, 0);
            return 1;
        }
        double best = f_center;
        for (int a = lo[0]; a <= hi[0]; ++a)
            for (int b = lo[1]; b <= hi[1]; ++b)
                for (int c = lo[2]; c <= hi[2]; ++c)
                    best = std::min(best, form(sub_m[0] + a * n[0], sub_m[1] + b * n[1], sub_m[2] + c * n[2]));
        const double limit = best + rel_tol * best;
        int count = 0;
        for (int a = lo[0]; a <= hi[0]; ++a)
            for (int b = lo[1]; b <= hi[1]; ++b)
                for (int c = lo[2]; c <= hi[2]; ++c)
                    if (form(sub_m[0] + a * n[0], sub_m[1] + b * n[1], sub_m[2] + c * n[2]) <= limit)
                    {
                        visit(a, b, c);
                        ++count;
                    }
        return count;
    }

    /// for_each_tie appending the shifts to `ties`
    template <typename Form>
    void append_ties(const std::array<int, 3>& sub_m, Form&& form, double rel_tol,
                     std::vector<std::array<int, 3>>& ties) const
    {
        for_each_tie(sub_m, form, rel_tol, [&](int a, int b, int c) { ties.push_back({a, b, c}); });
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
            AliasComponent& component = components[i];
            const int mask = masks[i];
            component.mask = mask;
            component.n = {mask & 1 ? n[0] : 1, mask & 2 ? n[1] : 1, mask & 4 ? n[2] : 1};
            component.g = {mask & 1 ? g.g11 : 0.0, mask & 2 ? g.g22 : 0.0, mask & 4 ? g.g33 : 0.0,
                           (mask & 3) == 3 ? g.g12 : 0.0, (mask & 5) == 5 ? g.g13 : 0.0,
                           (mask & 6) == 6 ? g.g23 : 0.0};
            component.box = min_form_alias_detail::box_metric(component.n, component.g);
            component.unique_below = min_form_alias_detail::unique_alias_bound(component.n, component.g, component.box);
        }
    }
};

/**
 * @brief Visit, in (a, b, c) order, the sums t0 + t1 of the tied period shifts of the two
 *        components of a grid (t0: ties0[0..n0), the first component; t1: ties1[0..n1)).
 *
 * Nested loops, the first component outside. When the second component lies between the
 * axes of the first ({x, z} and {y}, the only such pair), the ties of {x, z} with the same
 * shift a form the inner loop instead.
 */
template <typename Visit>
inline void for_each_tie_combination(const AliasSearch& search, const std::array<int, 3>* ties0, long long n0,
                                     const std::array<int, 3>* ties1, long long n1, Visit&& visit)
{
    if (n0 == 1 && n1 == 1)   // one alias of least form (most modes)
    {
        visit(ties0[0][0] + ties1[0][0], ties0[0][1] + ties1[0][1], ties0[0][2] + ties1[0][2]);
        return;
    }
    const bool interleaved = search.components[0].mask == 5 && search.components[1].mask == 2;
    for (long long j0 = 0; j0 < n0;)
    {
        long long run_end = j0 + 1;
        while (interleaved && run_end < n0 && ties0[run_end][0] == ties0[j0][0])
            ++run_end;
        for (long long j1 = 0; j1 < n1; ++j1)
            for (long long j = j0; j < run_end; ++j)
                visit(ties0[j][0] + ties1[j1][0], ties0[j][1] + ties1[j1][1], ties0[j][2] + ties1[j1][2]);
        j0 = run_end;
    }
}

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
    for (int i = 0; i < search.n_components; ++i)
    {
        const AliasComponent& component = search.components[i];
        std::array<int, 3> sub_m = component.part(m);
        min_form_alias_detail::component_min_alias(sub_m, component.n, component.box, form, component.unique_below);
        for (int d = 0; d < 3; ++d)
            if (component.mask & (1 << d))
                m[d] = sub_m[d];
    }
}

/**
 * @brief Visit, in (a, b, c) order, every least-form alias of m on a grid of at
 *        most two metric components (a grid of three throws; tied_alias_moments
 *        finds their ties on the Nyquist planes); returns the number visited.
 *        Each independent metric component has tolerance
 *        rel_tol * its minimum, so a large orthogonal contribution cannot
 *        turn nonminimal aliases of another component into ties.
 */
template <typename Form, typename Visit>
inline int for_each_min_form_alias(
    const std::array<int, 3>& m, const AliasSearch& search, Form&& form, double rel_tol, Visit&& visit)
{
    const std::array<int, 3>& n = search.n;
    auto visit_shift = [&](int a, int b, int c) { visit(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]); };
    if (search.n_components == 0)
    {
        visit(m[0], m[1], m[2]);
        return 1;
    }
    if (search.n_components == 1)
        return search.components[0].for_each_tie(search.components[0].part(m), form, rel_tol, visit_shift);

    // Two components on a coupled grid (three would need three mutually uncoupled axes,
    // whose ties tied_alias_moments finds on the Nyquist planes): the pairs of their ties
    if (search.n_components > 2)
        throw_with_line_number("for_each_min_form_alias needs a grid of at most two components.");
    std::array<std::vector<std::array<int, 3>>, 2> ties;
    for (int i = 0; i < 2; ++i)
        search.components[i].append_ties(search.components[i].part(m), form, rel_tol, ties[i]);
    int count = 0;
    for_each_tie_combination(search, ties[0].data(), ties[0].size(), ties[1].data(), ties[1].size(),
        [&](int a, int b, int c) {
            visit_shift(a, b, c);
            ++count;
        });
    return count;
}

/**
 * @brief Visit every mode of a periodic grid of n points with its alias of
 *        least form: visit(index, u), in the row-major order of `shape`.
 *
 * `shape` is n or, for the r2c half spectrum, n with its last axis of more
 * than one point cut to n/2 + 1; the modes have centered indices. Cells
 * without coupled axes keep the centered representative, which is minimal
 * there. `form` is the caller's expression, as for min_form_alias_rep. On a
 * grid of several components each component's modes are searched once.
 */
template <typename Form, typename Visit>
inline void for_each_mode_min_alias(
    const std::array<int, 3>& n, const std::array<int, 3>& shape, const IndexMetric& g,
    Form&& form, Visit&& visit)
{
    const AliasSearch search(n, g);
    if (!search.coupled)
    {
        for_each_centered_mode(n, shape, visit);
        return;
    }
    if (search.n_components < 2)
    {
        for_each_centered_mode(n, shape, [&](long long index, std::array<int, 3> u) {
            min_form_alias_rep(u, search, form);
            visit(index, u);
        });
        return;
    }
    // The least alias of every mode of each component, combined per mode as in min_form_alias_rep
    // (two components: three would need three mutually uncoupled axes)
    std::array<std::vector<std::array<int, 3>>, 2> least;
    std::array<std::array<long long, 3>, 2> strides;
    for (int i = 0; i < 2; ++i)
    {
        const AliasComponent& component = search.components[i];
        component.for_each_mode(shape, [&](std::array<int, 3> sub_m) {
            min_form_alias_detail::component_min_alias(sub_m, component.n, component.box, form, component.unique_below);
            least[i].push_back(sub_m);
        });
        strides[i] = component.mode_strides(shape);
    }
    for_each_centered_mode(n, shape, [&](long long index, const std::array<int, 3>& m) {
        std::array<int, 3> u = m;
        for (int i = 0; i < 2; ++i)
        {
            const AliasComponent& component = search.components[i];
            const std::array<int, 3>& sub_u = least[i][mode_position(m, n, strides[i])];
            for (int d = 0; d < 3; ++d)
                if (component.mask & (1 << d))
                    u[d] = sub_u[d];
        }
        visit(index, u);
    });
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
    min_form_alias_detail::MomentSum moments{search.g, scale};
    if (search.coupled)
    {
        for_each_min_form_alias(m, search, IndexForm{search.g}, min_form_alias_detail::tie_tolerance,
                                [&](int a, int b, int c) { moments.add(a, b, c); });
    }
    else
    {
        const int s[3] = {n[0] > 1 && 2 * m[0] == n[0], n[1] > 1 && 2 * m[1] == n[1], n[2] > 1 && 2 * m[2] == n[2]};
        for (int a = 0; a <= s[0]; ++a)
            for (int b = 0; b <= s[1]; ++b)
                for (int c = 0; c <= s[2]; ++c)
                    moments.add(m[0] - a * n[0], m[1] - b * n[1], m[2] - c * n[2]);
    }
    return moments.mean();
}

/**
 * @brief tied_alias_moments of every mode of a periodic grid of n points:
 *        visit(index, m, moments) in the row-major order of `shape` (see
 *        for_each_mode_min_alias), m the centered mode.
 *
 * On a grid of several components the aliases of m tied for the least form are
 * the combinations of the ties of its parts, so each component's modes are
 * searched once. for_each_tie_combination visits them as for_each_min_form_alias
 * does, which gives the sums of tied_alias_moments to the bit.
 */
template <typename Visit>
inline void for_each_mode_tied_moments(
    const std::array<int, 3>& n, const std::array<int, 3>& shape, const IndexMetric& g, double scale, Visit&& visit)
{
    const AliasSearch search(n, g);
    if (!search.coupled || search.n_components < 2)
    {
        for_each_centered_mode(n, shape, [&](long long index, const std::array<int, 3>& m) {
            visit(index, m, tied_alias_moments(m, search, scale));
        });
        return;
    }
    // A coupled grid has two components here (three would need three mutually uncoupled
    // axes). The tied period shifts of every mode of each component: those of its mode s
    // are shifts[i][first[i][s]] to shifts[i][first[i][s + 1] - 1].
    const IndexForm form{g};
    std::array<std::vector<std::array<int, 3>>, 2> shifts;
    std::array<std::vector<long long>, 2> first;
    std::array<std::array<long long, 3>, 2> strides;
    for (int i = 0; i < 2; ++i)
    {
        const AliasComponent& component = search.components[i];
        first[i].push_back(0);
        component.for_each_mode(shape, [&](const std::array<int, 3>& sub_m) {
            component.append_ties(sub_m, form, min_form_alias_detail::tie_tolerance, shifts[i]);
            first[i].push_back(static_cast<long long>(shifts[i].size()));
        });
        strides[i] = component.mode_strides(shape);
    }
    for_each_centered_mode(n, shape, [&](long long index, const std::array<int, 3>& m) {
        const long long s0 = mode_position(m, n, strides[0]);
        const long long s1 = mode_position(m, n, strides[1]);
        const long long n0 = first[0][s0 + 1] - first[0][s0], n1 = first[1][s1 + 1] - first[1][s1];
        min_form_alias_detail::MomentSum moments{g, scale};
        for_each_tie_combination(search, shifts[0].data() + first[0][s0], n0, shifts[1].data() + first[1][s1], n1,
            [&](int a, int b, int c) { moments.add(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]); });
        visit(index, m, moments.mean());
    });
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
    const long long n_modes = static_cast<long long>(shape[0]) * shape[1] * shape[2];
    for_each_mode_tied_moments(n, shape, g, 1.0,
        [&](long long index, const std::array<int, 3>&, const std::array<double, 6>& v) {
            const double product[3][3] = {{v[0], v[3], v[4]}, {v[3], v[1], v[5]}, {v[4], v[5], v[2]}};
            for (int a = 0; a < dim; ++a)
                for (int b = 0; b < dim; ++b)
                    moments[(a * dim + b) * n_modes + index] = product[a][b];
        });
}

#endif  // MINIMAL_FORM_ALIAS_H_
