/**
 * @file MinimalFormAlias.h
 * @brief Exact search for the alias of a Fourier mode that minimizes the
 *        reciprocal-metric form (first-Brillouin-zone representative).
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
 *
 * Axes with N_d <= 1 or a zero diagonal metric (the padded axes of 1D and 2D
 * grids) are not searched.
 */

#ifndef MINIMAL_FORM_ALIAS_H_
#define MINIMAL_FORM_ALIAS_H_

#include <array>
#include <cmath>

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

inline void component_parameters(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g, int mask,
    std::array<int, 3>& sub_m, std::array<int, 3>& sub_n, IndexMetric& sub_g)
{
    for (int d = 0; d < 3; ++d)
    {
        sub_m[d] = (mask & (1 << d)) ? m[d] : 0;
        sub_n[d] = (mask & (1 << d)) ? n[d] : 1;
    }
    sub_g = {mask & 1 ? g.g11 : 0.0, mask & 2 ? g.g22 : 0.0, mask & 4 ? g.g33 : 0.0,
             (mask & 3) == 3 ? g.g12 : 0.0, (mask & 5) == 5 ? g.g13 : 0.0,
             (mask & 6) == 6 ? g.g23 : 0.0};
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
        if (!(inv_diag[i] > 0.0) || !std::isfinite(inv_diag[i]))
        {
            // Degenerate metric: fall back to one period on each searched axis
            for (int j = 0; j < n_active; ++j)
            {
                lo[active[j]] = -1;
                hi[active[j]] = 1;
            }
            return;
        }
        // A small slack only adds candidates, so rounding cannot drop an alias
        const double reach = std::sqrt(f_center * inv_diag[i]) * (1.0 + 1e-9) + 1e-9;
        lo[d] = static_cast<int>(std::ceil((-reach - m[d]) / n[d]));
        hi[d] = static_cast<int>(std::floor((reach - m[d]) / n[d]));
        if (lo[d] > 0) lo[d] = 0;
        if (hi[d] < 0) hi[d] = 0;
    }
}
}  // namespace min_form_alias_detail

/**
 * @brief Replace m by its alias of least form.
 *
 * `form(u1, u2, u3)` evaluates the form; callers pass their own expression so
 * that its rounding is unchanged. In each coupled component the aliases are
 * visited in (a, b, c) order and replace the current choice only when their
 * form is strictly smaller, so its centered representative is kept on ties.
 */
template <typename Form>
inline void min_form_alias_rep(
    std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g, Form&& form)
{
    std::array<int, 3> masks{};
    const int n_components = min_form_alias_detail::components(n, g, masks);
    if (n_components > 1)
    {
        for (int i = 0; i < n_components; ++i)
        {
            std::array<int, 3> sub_m, sub_n;
            IndexMetric sub_g;
            min_form_alias_detail::component_parameters(m, n, g, masks[i], sub_m, sub_n, sub_g);
            min_form_alias_rep(sub_m, sub_n, sub_g, form);
            for (int d = 0; d < 3; ++d)
                if (masks[i] & (1 << d))
                    m[d] = sub_m[d];
        }
        return;
    }
    const double f_center = form(m[0], m[1], m[2]);
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
 * @brief Visit, in (a, b, c) order, every least-form alias of m; returns the
 *        number visited. Each independent metric component has tolerance
 *        rel_tol * its minimum, so a large orthogonal contribution cannot
 *        turn nonminimal aliases of another component into ties.
 */
template <typename Form, typename Visit>
inline int for_each_min_form_alias(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g,
    Form&& form, double rel_tol, Visit&& visit)
{
    std::array<int, 3> masks{};
    const int n_components = min_form_alias_detail::components(n, g, masks);
    if (n_components > 1)
    {
        std::array<int, 3> lo{0, 0, 0}, hi{0, 0, 0};
        std::array<double, 3> limits{};
        for (int i = 0; i < n_components; ++i)
        {
            std::array<int, 3> sub_m, sub_n, sub_lo, sub_hi;
            IndexMetric sub_g;
            min_form_alias_detail::component_parameters(m, n, g, masks[i], sub_m, sub_n, sub_g);
            min_form_alias_detail::search_box(sub_m, sub_n, sub_g, form(sub_m[0], sub_m[1], sub_m[2]), sub_lo, sub_hi);
            min_form_alias_rep(sub_m, sub_n, sub_g, form);
            const double best = form(sub_m[0], sub_m[1], sub_m[2]);
            limits[i] = best + rel_tol * best;
            for (int d = 0; d < 3; ++d)
                if (masks[i] & (1 << d))
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
                    for (int i = 0; i < n_components && tied; ++i)
                        tied = form(masks[i] & 1 ? t1 : 0, masks[i] & 2 ? t2 : 0, masks[i] & 4 ? t3 : 0) <= limits[i];
                    if (tied)
                    {
                        visit(t1, t2, t3);
                        ++count;
                    }
                }
        return count;
    }
    const double f_center = form(m[0], m[1], m[2]);
    std::array<int, 3> lo, hi;
    min_form_alias_detail::search_box(m, n, g, f_center, lo, hi);

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

#endif  // MINIMAL_FORM_ALIAS_H_
