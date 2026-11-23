/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestCrysFFTSelector.cpp
 * @brief The CrysFFT engine that plan_crysfft chooses (CrysFFTSelector.h) and the
 *        physical basis of SpaceGroup::enable_best_physical_basis().
 *
 * The space-group tests compare CrysFFT runs with full-grid runs, which agree
 * whatever CrysFFT is chosen; this test pins the choice itself:
 * - Oblique cells with alpha = beta = 90 degrees take ObliqueZ on the
 *   irreducible basis (P2/m, P6/mmm), with the glide shift nz/4 for P6_3/mmc;
 *   a z mirror at z = 1/4 needs nz divisible by 4.
 * - alpha != 90 degrees, non-periodic boxes, 2D grids and no space group give
 *   no CrysFFT.
 * - Orthogonal cells: Recursive3m for Im-3m and Pmmm when nz/2 >= 8 (the m3
 *   fold), the Pmmm DCT below that, and ObliqueZ on a grid with an odd nx or ny
 *   where the group has a z mirror.
 * - Physical bases enabled by hand must match the cell and the grid of their engine.
 * - The basis SCFT enables: m3 or Pmmm on orthogonal cells (never z-mirror),
 *   z-mirror with alpha = beta = 90 and gamma free or 120 degrees.
 */

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "CrysFFTSelector.h"
#include "SpaceGroup.h"

static bool check(bool condition, const std::string& what)
{
    std::cout << (condition ? "  ok: " : "  FAILED: ") << what << std::endl;
    return condition;
}

static bool throws(const std::function<void()>& call)
{
    try
    {
        call();
    }
    catch (std::exception&)
    {
        return true;
    }
    return false;
}

static const char* name(CrysFFTEngine engine)
{
    switch (engine)
    {
    case CrysFFTEngine::None: return "None";
    case CrysFFTEngine::Recursive3m: return "Recursive3m";
    case CrysFFTEngine::Pmmm: return "Pmmm";
    case CrysFFTEngine::ObliqueZ: return "ObliqueZ";
    }
    return "?";
}

// plan_crysfft for a 3D periodic box; orthogonal: all angles 90 degrees,
// z_orthogonal: alpha = beta = 90 degrees
static CrysFFTPlan select(const SpaceGroup& sg, std::array<int, 3> nx, bool orthogonal, bool z_orthogonal)
{
    return plan_crysfft(&sg, nx, 3, true, orthogonal, z_orthogonal);
}

static bool expect(const CrysFFTPlan& plan, CrysFFTEngine engine, const std::string& what)
{
    return check(plan.engine == engine, what + ": " + name(plan.engine) + " (expected " + name(engine) + ")");
}

// The z index of the first point of an ObliqueZ physical grid: the shift of its half along z
static int z_shift(const CrysFFTPlan& plan)
{
    return plan.map.phys_to_full.empty() ? -1 : plan.map.phys_to_full[0];
}

int main()
{
    bool ok = true;
    try
    {
        std::cout << "=== Oblique cells, irreducible basis ===" << std::endl;
        {
            // Monoclinic, unique axis c: the mirror (x, y, -z), gamma = 100 degrees
            SpaceGroup p2m({11, 10, 8}, "P2/m", 58);
            const auto oblique = select(p2m, {11, 10, 8}, false, true);
            ok &= expect(oblique, CrysFFTEngine::ObliqueZ, "P2/m 11x10x8, gamma != 90");
            ok &= check(z_shift(oblique) == 0 && oblique.map.covers_all_reduced && oblique.map.M_phys == 11 * 10 * 4,
                "P2/m: z mirror at z = 0, the lower half of the grid reaches every orbit");
            ok &= expect(select(p2m, {11, 10, 8}, false, false), CrysFFTEngine::None, "P2/m with alpha != 90");
        }
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            const auto hexagonal = select(p6mmm, {12, 12, 8}, false, true);
            ok &= expect(hexagonal, CrysFFTEngine::ObliqueZ, "P6/mmm 12x12x8, gamma = 120");
            ok &= check(z_shift(hexagonal) == 0, "P6/mmm: z mirror at z = 0");
        }
        {
            // The z mirror of P6_3/mmc lies at z = 1/4: shift nz/4 on the grid
            SpaceGroup p63mmc({12, 12, 8}, "P6_3/mmc", 488);
            const auto glide = select(p63mmc, {12, 12, 8}, false, true);
            ok &= expect(glide, CrysFFTEngine::ObliqueZ, "P6_3/mmc 12x12x8");
            double t_z = 0.0;
            ok &= check(z_shift(glide) == 2 && p63mmc.get_z_mirror_translation(t_z) && std::abs(t_z - 0.5) < 1e-12,
                "P6_3/mmc: shift nz/4 = 2 for the translation 1/2");
            SpaceGroup p63mmc_10({12, 12, 10}, "P6_3/mmc", 488);
            const auto odd_quarter = select(p63mmc_10, {12, 12, 10}, false, true);
            ok &= expect(odd_quarter, CrysFFTEngine::None, "P6_3/mmc 12x12x10 (nz/4 not an integer)");
            ok &= check(p63mmc_10.z_mirror_half_grid_shift() < 0, "P6_3/mmc 12x12x10: no ObliqueZ half grid");
            ok &= check(throws([&]{ p63mmc_10.enable_z_mirror_physical_basis(); }),
                "P6_3/mmc 12x12x10: no z-mirror basis either (the same shift rule)");
        }
        {
            // The shift rule of the z-mirror basis and ObliqueZ (SpaceGroup::z_mirror_half_grid_shift)
            std::string why;
            SpaceGroup p6mmm_odd({12, 12, 9}, "P6/mmm");
            ok &= check(p6mmm_odd.z_mirror_half_grid_shift(&why) == -1 && why.find("even Nz") != std::string::npos,
                "P6/mmm 12x12x9: no half grid along an odd z (" + why + ")");
            ok &= expect(select(p6mmm_odd, {12, 12, 9}, false, true), CrysFFTEngine::None, "P6/mmm 12x12x9: no ObliqueZ");
            SpaceGroup p63mmc_10({12, 12, 10}, "P6_3/mmc", 488);
            ok &= check(p63mmc_10.z_mirror_half_grid_shift(&why) == -1 && why.find("divisible by 4") != std::string::npos,
                "P6_3/mmc 12x12x10: the translation 1/2 needs nz divisible by 4 (" + why + ")");
            SpaceGroup p_1({8, 8, 8}, "P-1");
            ok &= check(p_1.z_mirror_half_grid_shift(&why) == -1 && why.find("does not contain") != std::string::npos,
                "P-1: no z mirror (" + why + ")");
            SpaceGroup p63mmc_16({12, 12, 16}, "P6_3/mmc", 488);
            ok &= check(p63mmc_16.z_mirror_half_grid_shift() == 4, "P6_3/mmc 12x12x16: shift nz/4 = 4");
        }

        std::cout << "=== Orthogonal cells ===" << std::endl;
        {
            SpaceGroup im3m({16, 16, 16}, "Im-3m", 529);
            ok &= expect(select(im3m, {16, 16, 16}, true, true), CrysFFTEngine::Recursive3m, "Im-3m 16^3 (nz/2 = 8)");
            SpaceGroup pmmm({32, 24, 16}, "Pmmm");
            ok &= expect(select(pmmm, {32, 24, 16}, true, true), CrysFFTEngine::Recursive3m, "Pmmm 32x24x16 (nz/2 = 8)");
            SpaceGroup pmmm_small({12, 12, 10}, "Pmmm");
            const auto dct = select(pmmm_small, {12, 12, 10}, true, true);
            ok &= expect(dct, CrysFFTEngine::Pmmm, "Pmmm 12x12x10 (nz/2 = 5, below the m3 fold)");
            ok &= check(dct.map.M_phys == 6 * 6 * 5 && dct.map.covers_all_reduced,
                "Pmmm 12x12x10: the octant of the Pmmm DCT reaches every orbit");
        }
        {
            // An odd nx or ny leaves no octant, so a z mirror takes ObliqueZ on the
            // irreducible basis as on an oblique cell
            SpaceGroup pmmm_odd({11, 10, 8}, "Pmmm");
            const auto odd = select(pmmm_odd, {11, 10, 8}, true, true);
            ok &= expect(odd, CrysFFTEngine::ObliqueZ, "Pmmm 11x10x8 (odd nx)");
            ok &= check(z_shift(odd) == 0 && odd.map.covers_all_reduced && odd.map.M_phys == 11 * 10 * 4,
                "Pmmm 11x10x8: z mirror at z = 0, the lower half reaches every orbit");
            SpaceGroup pmcm_odd({11, 10, 8}, 243);   // Pmcm: the z mirror (x, y, -z + 1/2) at z = 1/4
            const auto glide_odd = select(pmcm_odd, {11, 10, 8}, true, true);
            ok &= expect(glide_odd, CrysFFTEngine::ObliqueZ, "Pmcm (Hall 243) 11x10x8 (odd nx)");
            ok &= check(z_shift(glide_odd) == 2 && glide_odd.map.covers_all_reduced,
                "Pmcm 11x10x8: the half grid shifted by nz/4 = 2 reaches every orbit");
            SpaceGroup p4mmm_odd({9, 9, 8}, "P4/mmm");
            ok &= expect(select(p4mmm_odd, {9, 9, 8}, true, true), CrysFFTEngine::ObliqueZ, "P4/mmm 9x9x8 (odd nx = ny)");
            SpaceGroup pmmm_odd_z({12, 12, 9}, "Pmmm");
            ok &= expect(select(pmmm_odd_z, {12, 12, 9}, true, true), CrysFFTEngine::None, "Pmmm 12x12x9 (odd nz: no half grid)");
            SpaceGroup pmm2_odd({11, 10, 8}, "Pmm2");
            ok &= expect(select(pmm2_odd, {11, 10, 8}, true, true), CrysFFTEngine::None, "Pmm2 11x10x8 (no z mirror)");
        }

        std::cout << "=== No CrysFFT ===" << std::endl;
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            ok &= check(plan_crysfft(&p6mmm, {12, 12, 8}, 3, false, false, true).engine == CrysFFTEngine::None,
                "non-periodic box");
            ok &= check(plan_crysfft(&p6mmm, {12, 12, 8}, 2, true, false, true).engine == CrysFFTEngine::None,
                "2D grid");
            ok &= check(plan_crysfft(nullptr, {12, 12, 8}, 3, true, false, true).engine == CrysFFTEngine::None,
                "no space group");
        }

        std::cout << "=== Physical bases must match the cell ===" << std::endl;
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            p6mmm.enable_z_mirror_physical_basis();
            const auto z_basis = select(p6mmm, {12, 12, 8}, false, true);
            ok &= expect(z_basis, CrysFFTEngine::ObliqueZ, "P6/mmm z-mirror basis");
            ok &= check(z_basis.map.identity, "P6/mmm z-mirror basis: the basis is the ObliqueZ grid");
            ok &= check(throws([&]{ select(p6mmm, {12, 12, 8}, false, false); }),
                "the z-mirror basis rejects alpha != 90");
            SpaceGroup pmmm({12, 12, 10}, "Pmmm");
            pmmm.enable_pmmm_physical_basis();
            ok &= expect(select(pmmm, {12, 12, 10}, true, true), CrysFFTEngine::Pmmm, "Pmmm basis");
            ok &= check(throws([&]{ select(pmmm, {12, 12, 10}, false, true); }),
                "the Pmmm basis rejects an oblique cell");
            // The basis takes only its own engine: a z-mirror basis on an orthogonal cell
            // of a group with the Pmmm mirrors runs ObliqueZ
            SpaceGroup pmmm_z({12, 12, 10}, "Pmmm");
            pmmm_z.enable_z_mirror_physical_basis();
            ok &= expect(select(pmmm_z, {12, 12, 10}, true, true), CrysFFTEngine::ObliqueZ, "Pmmm z-mirror basis");
            // ... and must fit its grid: the recursive 3m engine needs nz/2 >= 2, so the
            // m3 basis is refused on 8x8x2 when it is enabled
            SpaceGroup pmmm_flat({8, 8, 2}, "Pmmm");
            ok &= check(throws([&]{ pmmm_flat.enable_m3_physical_basis(); }),
                "an m3 basis that the recursive 3m engine does not take (nz/2 = 1) is refused");
        }

        std::cout << "=== An enabled m3 basis below the nz/2 >= 8 heuristic ===" << std::endl;
        {
            // The heuristic chooses bases; an m3 basis already enabled needs the
            // recursive 3m CrysFFT, which is exact for nz/2 >= 2
            for (int n : {8, 12})
            {
                SpaceGroup im3m({n, n, n}, "Im-3m", 529);
                const std::string grid = "Im-3m " + std::to_string(n) + "^3";
                ok &= check(!im3m.m3_crysfft_feasible(), grid + ": m3_crysfft_feasible() keeps the heuristic");
                ok &= expect(select(im3m, {n, n, n}, true, true), CrysFFTEngine::Pmmm, grid + ", irreducible basis");
                im3m.enable_m3_physical_basis();
                ok &= expect(select(im3m, {n, n, n}, true, true), CrysFFTEngine::Recursive3m, grid + ", m3 basis");
            }
        }

        std::cout << "=== The basis SCFT enables (enable_best_physical_basis) ===" << std::endl;
        {
            struct Case { std::vector<int> nx; const char* symbol; int hall; CrysFFTEngine basis; const char* what; };
            const std::vector<Case> cases = {
                {{16, 16, 16}, "Im-3m", 529, CrysFFTEngine::Recursive3m, "Im-3m 16^3: m3"},
                {{12, 12, 12}, "Im-3m", 529, CrysFFTEngine::Pmmm, "Im-3m 12^3 (nz/2 = 6): Pmmm"},
                {{12, 12, 10}, "Pmmm", -1, CrysFFTEngine::Pmmm, "Pmmm 12x12x10: Pmmm"},
                {{11, 10, 8}, "Pmmm", -1, CrysFFTEngine::None, "Pmmm 11x10x8: irreducible (no z-mirror basis on orthogonal cells)"},
                {{12, 12, 8}, "P6/mmm", -1, CrysFFTEngine::ObliqueZ, "P6/mmm 12x12x8: z-mirror"},
                {{12, 12, 10}, "P6_3/mmc", 488, CrysFFTEngine::None, "P6_3/mmc 12x12x10: irreducible (nz/4 not an integer)"},
                {{11, 10, 8}, "P2/m", 58, CrysFFTEngine::ObliqueZ, "P2/m (unique axis c) 11x10x8: z-mirror"},
                {{8, 8, 8}, "P2/m", 57, CrysFFTEngine::None, "P2/m (unique axis b) 8^3: irreducible"},
                {{8, 8, 8}, "P-1", -1, CrysFFTEngine::None, "P-1 8^3: irreducible"}};
            for (const Case& c : cases)
            {
                SpaceGroup sg(c.nx, c.symbol, c.hall);
                const CrysFFTEngine basis = sg.enable_best_physical_basis();
                ok &= check(basis == c.basis && sg.physical_basis() == c.basis,
                    std::string(c.what) + " (" + sg.physical_basis_name() + ")");
            }
        }
    }
    catch (std::exception& exc)
    {
        std::cout << "Unexpected exception: " << exc.what() << std::endl;
        return -1;
    }

    if (!ok)
    {
        std::cout << "TestCrysFFTSelector FAILED" << std::endl;
        return -1;
    }
    std::cout << "TestCrysFFTSelector passed" << std::endl;
    return 0;
}
