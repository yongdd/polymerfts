/**
 * @file PseudoSolverHelpers.h
 * @brief Host helpers shared by the CPU and CUDA pseudo-spectral solver bases
 *        (CpuSolverPseudoBase, CudaSolverPseudoBase).
 *
 * The boundary conditions a pseudo-spectral box takes, the bond and step
 * coefficients of a monomer type, and the cell and the keys of the CrysFFT
 * tables: the Boltzmann table of each step coefficient and the stress
 * multiplier of each axis group and bond. The solvers keep their tables and
 * build them with the engine of their platform.
 */

#ifndef PSEUDO_SOLVER_HELPERS_H_
#define PSEUDO_SOLVER_HELPERS_H_

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ComputationBox.h"
#include "Exception.h"
#include "Molecules.h"
#include "SpaceGroup.h"

/**
 * @brief Whether every face of a pseudo-spectral box is periodic.
 *
 * Throws unless both faces of each direction have the same boundary condition:
 * the transforms (FFT, cosine and sine series) take whole directions.
 *
 * @param bc  Boundary conditions of the faces, two per direction
 * @param dim Number of directions
 */
inline bool pseudo_spectral_periodic(const std::vector<BoundaryCondition>& bc, int dim)
{
    for (int d = 0; d < dim; ++d)
    {
        if (bc[2 * d] != bc[2 * d + 1])
        {
            throw_with_line_number("Pseudo-spectral method requires matching boundary conditions on both sides of each direction. "
                "Direction " + std::to_string(d) + " has mismatched BCs. Use real-space method for mixed BCs.");
        }
    }
    return std::all_of(bc.begin(), bc.end(), [](BoundaryCondition b) { return b == BoundaryCondition::PERIODIC; });
}

/// b² of a monomer type, halved for a half bond (the stress integrand and the key of its multiplier).
inline double stress_bond_length_sq(const Molecules& molecules, const std::string& monomer_type, bool is_half_bond_length)
{
    const auto& bond_lengths = molecules.get_bond_lengths();
    auto bond_it = bond_lengths.find(monomer_type);
    if (bond_it == bond_lengths.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");

    double bond_length_sq = bond_it->second * bond_it->second;
    if (is_half_bond_length)
        bond_length_sq *= 0.5;
    return bond_length_sq;
}

/**
 * @brief The diffusion coefficient b² ds / 6 of a monomer type and contour step, the
 *        coefficient of a step exp(coeff ∇²) and the key of its CrysFFT Boltzmann table.
 *
 * @param ds_index  Index of the contour step (ContourLengthMapping)
 * @param half_step The half step: ds / 2
 */
inline double effective_diffusion_coeff(const Molecules& molecules, const std::string& monomer_type, int ds_index,
                                        bool half_step)
{
    double ds = molecules.get_contour_length_mapping().get_ds_from_index(ds_index);
    if (half_step)
        ds *= 0.5;
    return stress_bond_length_sq(molecules, monomer_type, false) * ds / 6.0;
}

/// Cell parameters [Lx, Ly, Lz, alpha, beta, gamma] of a box, those of the CrysFFT tables.
template <typename T>
std::array<double, 6> crysfft_cell(const ComputationBox<T>& cb)
{
    const auto lx = cb.get_lx();
    const auto angles = cb.get_angles();
    return {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};
}

/**
 * @brief The step coefficients of the CrysFFT Boltzmann tables: those of each block's
 *        monomer type and contour step, not of every combination (a recursive 3m table
 *        holds 16 physical-grid arrays).
 *
 * @param block_coeffs Callable (monomer_type, ds_index) -> the coefficients of the steps
 *                     a method takes for a block (the solver's crysfft_block_coeffs())
 */
template <typename BlockCoeffs>
std::set<double> crysfft_step_coeffs(Molecules& molecules, BlockCoeffs&& block_coeffs)
{
    std::set<double> coeffs;
    const ContourLengthMapping& mapping = molecules.get_contour_length_mapping();
    for (int p = 0; p < molecules.get_n_polymer_types(); ++p)
        for (const Block& block : molecules.get_polymer(p).get_blocks())
            for (double coeff : block_coeffs(block.monomer_type, mapping.get_ds_index(block.contour_length)))
                coeffs.insert(coeff);
    return coeffs;
}

/**
 * @brief Axis groups of the CrysFFT stress fast path, as bit masks (1: x, 2: y, 4: z).
 *
 * A per-axis multiplier k_d² is not invariant under point-group operations
 * that permute axes (cubic: x, y, z; tetragonal: x, y), so on the reduced basis
 * only its sum over such a group of axes is well defined. The stress routines
 * apply one multiplier Σ_{d in group} k_d² per group and give every axis of the
 * group the group sum divided by the group size; the kernel q1·q2 is symmetric,
 * so this is the exact value of each diagonal component.
 */
inline std::vector<int> crysfft_stress_axis_groups(const std::string& crystal_system)
{
    if (crystal_system == "Cubic")
        return {7};
    if (crystal_system == "Tetragonal")
        return {3, 4};
    return {1, 2, 4};
}

/**
 * @brief The keys of the stress multipliers of a bond: per axis group of the crystal system
 *        (crysfft_stress_axis_groups()), (axes, b²·ds/6), or (axes, 0) without the bond factor.
 *
 * Discrete chains carry the bond factor ĝ(k) = exp(-b² k² Δs / 6) in the stress integrand.
 */
inline std::vector<std::pair<int, double>> crysfft_stress_keys(
    const Molecules& molecules, const SpaceGroup& space_group, double bond_length_sq, bool include_bond_factor)
{
    const double coeff = include_bond_factor ? bond_length_sq * molecules.get_global_ds() / 6.0 : 0.0;
    std::vector<std::pair<int, double>> keys;
    for (int axes : crysfft_stress_axis_groups(space_group.get_crystal_system()))
        keys.emplace_back(axes, coeff);
    return keys;
}

/**
 * @brief The keys of every stress multiplier of the molecules: those of the full bonds of
 *        each block's monomer type and, with the bond factor (discrete chains), of the half bonds.
 */
inline std::set<std::pair<int, double>> crysfft_multiplier_keys(
    Molecules& molecules, const SpaceGroup& space_group, bool include_bond_factor)
{
    std::set<std::string> monomer_types;
    for (int p = 0; p < molecules.get_n_polymer_types(); ++p)
        for (const Block& block : molecules.get_polymer(p).get_blocks())
            monomer_types.insert(block.monomer_type);

    std::set<std::pair<int, double>> keys;
    for (const std::string& monomer_type : monomer_types)
    {
        for (const bool half : {false, true})
        {
            if (half && !include_bond_factor)
                continue;   // continuous chains: one multiplier per axis group
            for (const auto& key : crysfft_stress_keys(molecules, space_group,
                     stress_bond_length_sq(molecules, monomer_type, half), include_bond_factor))
                keys.insert(key);
        }
    }
    return keys;
}

#endif  // PSEUDO_SOLVER_HELPERS_H_
