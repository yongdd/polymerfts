/**
 * @file TestMinimalFormAlias.cpp
 * @brief The alias search of MinimalFormAlias.h against a brute-force search.
 *
 * For random oblique (2D) and triclinic (3D) metrics, including strongly
 * anisotropic grid spacing, every mode's alias of least form must match a wide
 * brute-force window, the least form must be the same for m and -m (which the
 * CUDA ObliqueZ spectral kernel needs), and for_each_min_form_alias must visit
 * the same number of tied aliases. On cells with nearly isotropic spacing the
 * representative must equal the one of the former one-period window, so that
 * results do not change there. The case that motivated the exact search, 4x32
 * points on a 120-degree cell with equal lengths, must defeat that window.
 * Near-tie cells, whose shortest period lies off the axes and whose even grids
 * hold modes tied exactly with an alias, check the early exit of the search
 * against the brute force, with the form summed in another order by the caller.
 * On cells of two components (an axis orthogonal to the other two) the least
 * aliases and the tie counts, which combine the ties of each component, must
 * match the brute force too, and the loops over every mode, which search each
 * component's modes once, must give the per-mode least aliases and tied-alias
 * moments to the bit.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "MinimalFormAlias.h"

namespace {

struct Form
{
    IndexMetric g;
    double operator()(double a, double b, double c) const
    {
        return g.g11 * a * a + g.g22 * b * b + g.g33 * c * c + 2.0 * (g.g12 * a * b + g.g13 * a * c + g.g23 * b * c);
    }
};

// The same form summed in another order, as a caller may evaluate it (the ObliqueZ
// engines do): its rounding differs from IndexForm's, with which the search computes
// its bound
struct SwappedForm
{
    IndexMetric g;
    double operator()(double a, double b, double c) const
    {
        return 2.0 * (g.g23 * b * c + g.g13 * a * c + g.g12 * a * b) + g.g33 * c * c + g.g22 * b * b + g.g11 * a * a;
    }
};

int centered(int i, int n) { return (i > n / 2) ? i - n : i; }

// Reciprocal metric (index space) of a cell with lengths L and angles in radians
IndexMetric metric_of(double a, double b, double c, double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), cb = std::cos(beta), cg = std::cos(gamma), sg = std::sin(gamma);
    const double vol = a * b * c * std::sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg);
    const double A[3][3] = {{a, 0.0, 0.0}, {b * cg, b * sg, 0.0},
                            {c * cb, c * (ca - cb * cg) / sg, vol / (a * b * sg)}};
    double R[3][3];   // rows: reciprocal vectors (inverse transpose of A)
    for (int i = 0; i < 3; ++i)
    {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        R[i][0] = (A[j][1] * A[k][2] - A[j][2] * A[k][1]) / vol;
        R[i][1] = (A[j][2] * A[k][0] - A[j][0] * A[k][2]) / vol;
        R[i][2] = (A[j][0] * A[k][1] - A[j][1] * A[k][0]) / vol;
    }
    auto dot = [&](int i, int j) { return R[i][0] * R[j][0] + R[i][1] * R[j][1] + R[i][2] * R[j][2]; };
    return {dot(0, 0), dot(1, 1), dot(2, 2), dot(0, 1), dot(0, 2), dot(1, 2)};
}

// Former window: one period on every axis with N > 1
std::array<int, 3> one_period_rep(std::array<int, 3> m, const std::array<int, 3>& n, const Form& form)
{
    const int s[3] = {n[0] > 1, n[1] > 1, n[2] > 1};
    std::array<int, 3> best_rep = m;
    double best = form(m[0], m[1], m[2]);
    for (int a = -s[0]; a <= s[0]; ++a)
        for (int b = -s[1]; b <= s[1]; ++b)
            for (int c = -s[2]; c <= s[2]; ++c)
            {
                if (a == 0 && b == 0 && c == 0)
                    continue;
                const std::array<int, 3> t = {m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]};
                const double f = form(t[0], t[1], t[2]);
                if (f < best)
                {
                    best = f;
                    best_rep = t;
                }
            }
    return best_rep;
}

struct Stats
{
    long modes = 0, wrong_min = 0, odd = 0, wrong_ties = 0, window_differs = 0, window_misses = 0, tied = 0;
};

// The search with the caller's form LibForm against a brute-force window
template <typename LibForm = Form>
void check_grid(const std::array<int, 3>& n, const IndexMetric& g, bool expect_window, Stats& s)
{
    const Form form{g};
    const LibForm lib_form{g};
    const int W = 16;   // brute-force window in periods
    for (int i = 0; i < n[0]; ++i)
        for (int j = 0; j < n[1]; ++j)
            for (int k = 0; k < n[2]; ++k)
            {
                const std::array<int, 3> m = {centered(i, n[0]), centered(j, n[1]), centered(k, n[2])};
                double brute = form(m[0], m[1], m[2]);
                const int w[3] = {n[0] > 1 ? W : 0, n[1] > 1 ? W : 0, n[2] > 1 ? W : 0};
                for (int a = -w[0]; a <= w[0]; ++a)
                    for (int b = -w[1]; b <= w[1]; ++b)
                        for (int c = -w[2]; c <= w[2]; ++c)
                            brute = std::min(brute, form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]));
                int brute_ties = 0;
                for (int a = -w[0]; a <= w[0]; ++a)
                    for (int b = -w[1]; b <= w[1]; ++b)
                        for (int c = -w[2]; c <= w[2]; ++c)
                            if (form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]) <= brute + 1e-10 * brute)
                                ++brute_ties;

                std::array<int, 3> rep = m;
                min_form_alias_rep(rep, n, g, lib_form);
                const double f = form(rep[0], rep[1], rep[2]);
                std::array<int, 3> rep_neg = {centered((n[0] - i) % n[0], n[0]), centered((n[1] - j) % n[1], n[1]),
                                              centered((n[2] - k) % n[2], n[2])};
                min_form_alias_rep(rep_neg, n, g, lib_form);
                const double f_neg = form(rep_neg[0], rep_neg[1], rep_neg[2]);
                const int ties = for_each_min_form_alias(m, n, g, lib_form, 1e-10, [](int, int, int) {});
                const std::array<int, 3> window = one_period_rep(m, n, form);

                ++s.modes;
                s.tied += (brute_ties > 1);
                s.wrong_min += (std::abs(f - brute) > 1e-12 * std::max(brute, 1e-300));
                s.odd += (std::abs(f - f_neg) > 1e-12 * std::max(f, 1e-300));
                s.wrong_ties += (ties != brute_ties);
                if (expect_window)
                    s.window_differs += (window != rep);
                s.window_misses += (form(window[0], window[1], window[2]) > brute * (1.0 + 1e-12));
            }
}

bool check_separable_thin_cells()
{
    bool ok = true;
    // Every orientation of a coupled plane beside a short orthogonal axis.
    // Its 1e20 contribution must neither enlarge the plane's search nor hide
    // its O(1) form differences/ties in floating-point addition.
    for (int thin = 0; thin < 3; ++thin)
    {
        const int x = (thin + 1) % 3, y = (thin + 2) % 3;
        double G[3][3]{};
        G[x][x] = G[y][y] = 1.0;
        G[thin][thin] = 1e20;
        G[x][y] = G[y][x] = 0.5;
        const IndexMetric g{G[0][0], G[1][1], G[2][2], G[0][1], G[0][2], G[1][2]};
        const Form form{g};
        std::array<int, 3> n{}, m{};
        n[x] = n[y] = 8; n[thin] = 4;
        m[x] = m[y] = 3; m[thin] = 1;
        int evaluations = 0;
        auto counted_form = [&](int a, int b, int c) {
            ++evaluations;
            return form(a, b, c);
        };
        min_form_alias_rep(m, n, g, counted_form);
        const double plane_min = m[x] * m[x] + m[y] * m[y] + m[x] * m[y];
        ok &= index_metric_has_coupling(g) && plane_min == 19.0 && m[thin] == 1 && evaluations < 200;

        // Three aliases tie in the hexagonal plane; two tie on z's Nyquist
        // plane. The Cartesian product has six, regardless of the z scale.
        n[x] = n[y] = 6;
        m[x] = m[y] = m[thin] = 2;
        evaluations = 0;
        bool values_ok = true;
        const int ties = for_each_min_form_alias(m, n, g, counted_form, 1e-10, [&](int a, int b, int c) {
            const std::array<int, 3> v{a, b, c};
            values_ok &= v[x] * v[x] + v[y] * v[y] + v[x] * v[y] == 12 && std::abs(v[thin]) == 2;
        });
        ok &= ties == 6 && values_ok && evaluations < 200;
    }
    // A pair's normalized coupling is independent of axis length scaling.
    ok &= reciprocal_axes_coupled(1e200, 1e-200, 0.5);
    ok &= !reciprocal_axes_coupled(1e200, 1e-200, 6e-17);
    std::printf("separable thin cells: %s\n", ok ? "passed" : "FAILED");
    return ok;
}

// The loops over every mode of a grid of two components (cells with an axis
// orthogonal to the other two) search each component's modes once and combine
// them per mode. They must give the per-mode results of min_form_alias_rep and
// tied_alias_moments to the bit, on the full grid and on the r2c half spectrum,
// also when the components interleave ({x, z} and {y}: the ties are visited in
// (a, b, c) order).
bool check_component_grid_loops()
{
    const double deg = 3.14159265358979323846 / 180.0;
    struct Cell
    {
        std::array<int, 3> n;
        double a, b, c, alpha, beta, gamma;
        const char* name;
    };
    const Cell cells[] = {
        {{24, 24, 20}, 4.0, 4.0, 3.0, 90, 90, 120, "hexagonal: {x, y} and {z}"},
        {{17, 17, 15}, 4.0, 4.0, 3.0, 90, 90, 120, "odd hexagonal"},
        {{30, 26, 20}, 3.0, 3.1, 2.0, 90, 90, 100, "monoclinic, unique axis c"},
        {{30, 26, 20}, 3.0, 3.1, 2.0, 90, 105, 90, "monoclinic, unique axis b: {x, z} and {y}"},
        {{16, 32, 8}, 4.0, 4.0, 4.0, 90, 120, 90, "anisotropic spacing: {x, z} and {y}"},
        {{20, 26, 30}, 2.0, 3.1, 3.0, 100, 90, 90, "alpha != 90: {x} and {y, z}"},
        {{32, 32, 16}, 4.0, 4.0, 0.3, 90, 90, 120, "thin z"},
    };
    bool ok = true;
    for (const Cell& cell : cells)
    {
        const std::array<int, 3>& n = cell.n;
        const IndexMetric g = metric_of(cell.a, cell.b, cell.c, cell.alpha * deg, cell.beta * deg, cell.gamma * deg);
        const AliasSearch search(n, g);
        const Form form{g};
        long long modes = 0, wrong_rep = 0, wrong_moments = 0, tied = 0;
        for (const std::array<int, 3>& shape : {n, std::array<int, 3>{n[0], n[1], n[2] / 2 + 1}})
        {
            std::vector<std::array<int, 3>> reps;
            for_each_mode_min_alias(n, shape, g, form,
                [&](long long, const std::array<int, 3>& u) { reps.push_back(u); });
            std::vector<std::array<double, 6>> moments;
            for_each_mode_tied_moments(n, shape, g, 2.5,
                [&](long long, const std::array<int, 3>&, const std::array<double, 6>& v) { moments.push_back(v); });
            long long index = 0;
            for (int i = 0; i < shape[0]; ++i)
                for (int j = 0; j < shape[1]; ++j)
                    for (int k = 0; k < shape[2]; ++k, ++index)
                    {
                        const std::array<int, 3> m = {centered(i, n[0]), centered(j, n[1]), centered(k, n[2])};
                        std::array<int, 3> rep = m;
                        min_form_alias_rep(rep, search, form);
                        const std::array<double, 6> v = tied_alias_moments(m, search, 2.5);
                        ++modes;
                        tied += for_each_min_form_alias(m, search, IndexForm{g}, 1e-10, [](int, int, int) {}) > 1;
                        wrong_rep += (reps[index] != rep);
                        wrong_moments += std::memcmp(moments[index].data(), v.data(), sizeof(v)) != 0;
                    }
        }
        const bool cell_ok = search.coupled && search.n_components == 2 && tied > 0 && wrong_rep == 0
                             && wrong_moments == 0;
        std::printf("grid loops, %s: %lld modes (%lld tied), least alias differs %lld, moments differ %lld%s\n",
                    cell.name, modes, tied, wrong_rep, wrong_moments, cell_ok ? "" : "  FAILED");
        ok &= cell_ok;
    }
    return ok;
}

}  // namespace

int main()
{
    const double deg = 3.14159265358979323846 / 180.0;
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    bool ok = true;

    // 1) Random oblique in-plane cells (alpha = beta = 90) and triclinic cells,
    //    with grid spacings up to a factor 8 apart
    Stats random_cells;
    for (int trial = 0; trial < 60; ++trial)
    {
        const bool triclinic = (trial % 3 == 2);
        const std::array<int, 3> n = {2 + static_cast<int>(uni(rng) * 30), 2 + static_cast<int>(uni(rng) * 30),
                                      triclinic ? 2 + static_cast<int>(uni(rng) * 10) : 1};
        const double gamma = (50.0 + 80.0 * uni(rng)) * deg;
        const double alpha = triclinic ? (70.0 + 40.0 * uni(rng)) * deg : 90.0 * deg;
        const double beta = triclinic ? (70.0 + 40.0 * uni(rng)) * deg : 90.0 * deg;
        IndexMetric g = metric_of(1.0 + 7.0 * uni(rng), 1.0 + 7.0 * uni(rng), 1.0 + 3.0 * uni(rng), alpha, beta, gamma);
        if (!triclinic)
            g.g33 = g.g13 = g.g23 = 0.0;   // in-plane search, as in ObliqueZ
        check_grid(n, g, false, random_cells);
    }
    std::printf("random cells: %ld modes, least form wrong %ld, f(m) != f(-m) %ld, tie count wrong %ld "
                "(one-period window misses the minimum on %ld)\n",
                random_cells.modes, random_cells.wrong_min, random_cells.odd, random_cells.wrong_ties,
                random_cells.window_misses);
    ok &= (random_cells.wrong_min == 0 && random_cells.odd == 0 && random_cells.wrong_ties == 0);

    // 2) Cells with nearly isotropic spacing: same representative as the former window
    Stats isotropic;
    const std::array<double, 4> gammas = {60.0, 90.0 + 1e-6, 100.0, 120.0};
    for (double gam : gammas)
    {
        check_grid({24, 24, 1}, metric_of(2.0, 2.0, 1.0, 90 * deg, 90 * deg, gam * deg), true, isotropic);
        check_grid({12, 10, 1}, metric_of(1.5, 1.3, 1.0, 90 * deg, 90 * deg, gam * deg), true, isotropic);
        check_grid({16, 16, 12}, metric_of(2.0, 2.0, 1.6, 80 * deg, 95 * deg, gam * deg), true, isotropic);
    }
    std::printf("isotropic spacing: %ld modes, representative differs from the one-period window %ld, "
                "least form wrong %ld\n", isotropic.modes, isotropic.window_differs, isotropic.wrong_min);
    ok &= (isotropic.window_differs == 0 && isotropic.wrong_min == 0);

    // 3) 4x32 points on a 120-degree cell with equal lengths: the window fails, the search does not
    Stats aniso;
    IndexMetric g_hex = metric_of(4.0, 4.0, 2.0, 90 * deg, 90 * deg, 120 * deg);
    g_hex.g33 = g_hex.g13 = g_hex.g23 = 0.0;
    check_grid({4, 32, 1}, g_hex, false, aniso);
    std::printf("4x32 on a 4x4 120-degree cell: window misses %ld of %ld modes; search: least form wrong %ld, "
                "f(m) != f(-m) %ld\n", aniso.window_misses, aniso.modes, aniso.wrong_min, aniso.odd);
    ok &= (aniso.window_misses > 0 && aniso.wrong_min == 0 && aniso.odd == 0);

    // 4) Near-tie cells for the early exit of the search (a mode with 4 f(m) below
    //    lambda_1^2, the least form of a period, by a margin, is its own unique least
    //    alias). Angles far from 90 degrees put the shortest period N t off the axes, so
    //    a bound from the axis periods alone would exceed lambda_1, and even grids hold the
    //    modes m = N t / 2, tied exactly with m - N t. The caller's form sums in another
    //    order than the search's bound (SwappedForm), so their roundings differ at these ties.
    Stats near_tie;
    for (const double gam : {35.0, 45.0, 55.0, 125.0, 135.0, 145.0})
        for (const double b_len : {2.0, 2.3, 1.7, 2.9, 1.3, 3.7})
        {
            IndexMetric g_plane = metric_of(2.0, b_len, 1.0, 90 * deg, 90 * deg, gam * deg);
            g_plane.g33 = g_plane.g13 = g_plane.g23 = 0.0;
            check_grid<SwappedForm>({12, 12, 1}, g_plane, false, near_tie);
            check_grid<SwappedForm>({16, 14, 1}, g_plane, false, near_tie);
            check_grid<SwappedForm>({8, 18, 1}, g_plane, false, near_tie);
            if (b_len == 2.0 || b_len == 2.3)
                check_grid<SwappedForm>({10, 10, 6}, metric_of(2.0, b_len, 1.7, 80 * deg, 100 * deg, gam * deg),
                                        false, near_tie);
        }
    std::printf("near-tie cells: %ld modes (%ld tied), least form wrong %ld, f(m) != f(-m) %ld, "
                "tie count wrong %ld\n", near_tie.modes, near_tie.tied, near_tie.wrong_min, near_tie.odd,
                near_tie.wrong_ties);
    ok &= (near_tie.tied > 0 && near_tie.wrong_min == 0 && near_tie.odd == 0 && near_tie.wrong_ties == 0);

    // 5) Two-component cells (an axis orthogonal to the other two) against the brute force
    Stats two_components;
    bool all_two_components = true;
    auto check_two_components = [&](const std::array<int, 3>& n, const IndexMetric& g) {
        all_two_components &= AliasSearch(n, g).n_components == 2;
        check_grid(n, g, false, two_components);
    };
    check_two_components({12, 12, 8}, metric_of(4.0, 4.0, 3.0, 90 * deg, 90 * deg, 120 * deg));
    check_two_components({9, 9, 7}, metric_of(4.0, 4.0, 3.0, 90 * deg, 90 * deg, 120 * deg));
    check_two_components({10, 8, 12}, metric_of(3.0, 3.1, 2.0, 90 * deg, 105 * deg, 90 * deg));
    check_two_components({4, 16, 6}, metric_of(4.0, 4.0, 1.0, 90 * deg, 90 * deg, 120 * deg));
    std::printf("two-component cells: %ld modes (%ld tied), least form wrong %ld, f(m) != f(-m) %ld, "
                "tie count wrong %ld\n", two_components.modes, two_components.tied, two_components.wrong_min,
                two_components.odd, two_components.wrong_ties);
    ok &= (all_two_components && two_components.tied > 0 && two_components.wrong_min == 0 && two_components.odd == 0
           && two_components.wrong_ties == 0);

    ok &= check_separable_thin_cells();
    ok &= check_component_grid_loops();

    std::printf(ok ? "\nAll passed!\n" : "\nFAILED!\n");
    return ok ? 0 : 1;
}
