/**
 * @file Pseudo.h
 * @brief The Fourier-space tables of the pseudo-spectral solvers for one cell.
 *
 * This header provides the Pseudo class for pseudo-spectral method
 * supporting periodic, reflecting, and absorbing boundary conditions,
 * as well as non-orthogonal crystal systems via reciprocal metric tensor.
 *
 * **Wavenumber Conventions:**
 *
 * - PERIODIC: k = 2π*n/L (n = 0, 1, ..., N/2, -N/2+1, ..., -1)
 * - REFLECTING (DCT): k = π*n/L (n = 0, 1, ..., N-1)
 * - ABSORBING (DST): k = π*(n+1)/L (n = 0, 1, ..., N-1)
 *
 * **Boltzmann Bond Factors:**
 *
 * Both chain models use the same mathematical formula but with different
 * physical interpretations:
 *
 * boltz_bond[k] = exp(-b²*|k|²*ds/6)
 *
 * - **Continuous chains**: This is the diffusion propagator from solving
 *   the modified diffusion equation ∂q/∂s = (b²/6)∇²q - wq.
 *
 * - **Discrete chains**: This is the bond function ĝ(k) = exp(-a²|k|²/6)
 *   from the Chapman-Kolmogorov integral equation. With the convention
 *   a² = b²ds (where a is the segment length and ds = 1/N), it takes
 *   the same form as the continuous case. See Park et al. J. Chem. Phys.
 *   150, 234901 (2019) for details.
 *
 * For non-orthogonal systems with periodic BC:
 * |k|² = g^{-1}_ij k_i k_j where g^{-1} is the reciprocal metric tensor.
 *
 * **Stress Calculation using Deformation Vector:**
 *
 * For stress calculation, we use the deformation vector v = 2π g⁻¹ m
 * where g⁻¹ is the inverse metric tensor and m is the Miller index vector.
 * The deformation vector components vᵢ have units of 1/L² (not 1/L like
 * Cartesian wavevector components).
 *
 * The stress arrays store v⊗v dyad product components in Voigt notation:
 * - Index 0: V₁₁ = Σ(kernel × v₁²) → drives L₁ optimization
 * - Index 1: V₂₂ = Σ(kernel × v₂²) → drives L₂ optimization
 * - Index 2: V₃₃ = Σ(kernel × v₃²) → drives L₃ optimization
 * - Index 3: V₁₂ = Σ(kernel × v₁v₂) → drives γ optimization
 * - Index 4: V₁₃ = Σ(kernel × v₁v₃) → drives β optimization
 * - Index 5: V₂₃ = Σ(kernel × v₂v₃) → drives α optimization
 *
 * For 2D: [V₁₁, V₂₂, V₁₂, 0, 0, 0]. For 1D: only index 0 used.
 *
 * @see CpuSolverPseudoRQM4 for continuous chain solver
 * @see CpuSolverPseudoDiscrete for discrete chain solver
 * @see docs/theory/StressTensor.md for detailed derivation
 */

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "Exception.h"
#include "ComputationBox.h"

/**
 * @class Pseudo
 * @brief The Fourier-space tables of one cell for the pseudo-spectral method.
 *
 * Periodic boundary conditions (FFT, with the reciprocal metric of an oblique
 * cell) on every axis, or reflecting (DCT) and absorbing (DST) ones on every
 * axis. The grid, boundary conditions, bond lengths and contour steps are fixed
 * at construction; update() computes the tables of a cell.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Array Storage** (row-major in box axis order, like the fields):
 *
 * - Periodic BC with r2c FFT: size = Nx × Ny × (Nz/2+1)
 * - Periodic BC with c2c FFT: size = Nx × Ny × Nz
 * - Non-periodic BC (DCT/DST): size = Nx × Ny × Nz (real-to-real)
 */
template <typename T>
class Pseudo
{
protected:
    std::map<std::string, double> bond_lengths;  ///< Segment lengths for each type
    std::vector<BoundaryCondition> bc;       ///< Boundary conditions per dimension
    std::vector<int> nx;                     ///< Grid dimensions [Nx, Ny, Nz]
    std::map<int, double> ds_values;         ///< Contour step of each ds_index with Boltzmann tables
    int total_complex_grid;                  ///< Number of spectral coefficients of a field

    /**
     * @brief Boltzmann bond factors of a full and a half step: [ds_index][type][idx].
     *
     * boltz_bond = exp(-|k|² b² ds / 6), boltz_bond_half = exp(-|k|² b² ds / 12), where
     * k depends on the boundary conditions: the diffusion propagator of continuous
     * chains, the bond function (and half-bond function at chain ends) of discrete ones.
     */
    std::map<int, std::map<std::string, std::vector<double>>> boltz_bond;
    std::map<int, std::map<std::string, std::vector<double>>> boltz_bond_half;

    /**
     * @brief The stress basis: v⊗v of the deformation vector v = 2π g⁻¹ m of each mode,
     *        with the weight of the mode in the sum over the spectrum.
     *
     * fourier_basis_x[m] = v₁², _y = v₂², _z = v₃² along the box axes (1D and 2D boxes
     * use the first one or two); the cross terms _xy = v₁v₂, _xz = v₁v₃, _yz = v₂v₃
     * drive the angles of oblique cells and exist for periodic BC only (walls need an
     * orthogonal cell). Non-periodic BC use k² = (π n / L)² of their wavenumbers.
     * vᵢ has units of 1/L², not 1/L like the Cartesian wavevector.
     */
    std::vector<double> fourier_basis_x;
    std::vector<double> fourier_basis_y;
    std::vector<double> fourier_basis_z;
    std::vector<double> fourier_basis_xy;
    std::vector<double> fourier_basis_xz;
    std::vector<double> fourier_basis_yz;

    std::vector<int> negative_k_idx;  ///< Index of -k for each k (complex fields, periodic BC; with the stress basis)

    /// Check if all BCs are periodic.
    bool is_all_periodic() const;

    /// The tables of periodic BC (the alias of least form of each mode on oblique cells).
    void update_periodic(const std::array<double, 6>& recip_metric, bool with_stress_basis);

    /// The tables of reflecting and absorbing BC (orthogonal cells).
    void update_nonperiodic(const std::vector<double>& dx, bool with_stress_basis);

    /// Fill negative_k_idx (complex fields with periodic BC).
    void update_negative_frequency_mapping();

public:
    /**
     * @brief The tables of a grid, filled by update().
     *
     * @param bond_lengths Segment lengths: {type: b}
     * @param bc           One boundary condition per dimension: all periodic, or none
     * @param nx           Grid dimensions
     * @param ds_values    Contour step of each ds_index whose Boltzmann bond factors are
     *                     kept; none when empty (a solver that only needs the stress basis)
     */
    Pseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx,
        std::map<int, double> ds_values);

    /// Virtual destructor.
    virtual ~Pseudo();

    /**
     * @brief Get total complex grid size.
     * @return Total grid size for Fourier coefficients
     */
    int get_total_complex_grid() const;

    /**
     * @brief Values of the spectrum of a field on a grid of n_grid points with
     *        n_last points along its last axis: the r2c half spectrum (n_last/2 + 1
     *        values along that axis) for real fields with periodic BCs on every axis,
     *        the full grid otherwise (c2c FFT, or DCT/DST coefficients).
     *
     * The rule of get_total_complex_grid(), for solvers that need the size without
     * the tables.
     */
    static int complex_grid_size(int n_grid, int n_last, bool all_periodic);

    /**
     * @brief Get Boltzmann factor for full step for given ds_index.
     *
     * @param monomer_type Monomer type (e.g., "A", "B")
     * @param ds_index     0-based index of the contour step (ContourLengthMapping)
     * @return Pointer to Boltzmann factor array
     */
    virtual double* get_boltz_bond(const std::string& monomer_type, int ds_index);

    /**
     * @brief Get Boltzmann factor for half step for given ds_index.
     *
     * @param monomer_type Monomer type (e.g., "A", "B")
     * @param ds_index     0-based index of the contour step (ContourLengthMapping)
     * @return Pointer to Boltzmann factor array
     */
    virtual double* get_boltz_bond_half(const std::string& monomer_type, int ds_index);

    /// Get x-direction Fourier basis for stress.
    virtual const double* get_fourier_basis_x();

    /// Get y-direction Fourier basis for stress.
    virtual const double* get_fourier_basis_y();

    /// Get z-direction Fourier basis for stress.
    virtual const double* get_fourier_basis_z();

    /// Get xy cross-term Fourier basis for stress (nullptr for non-periodic BC).
    virtual const double* get_fourier_basis_xy();

    /// Get xz cross-term Fourier basis for stress (nullptr for non-periodic BC).
    virtual const double* get_fourier_basis_xz();

    /// Get yz cross-term Fourier basis for stress (nullptr for non-periodic BC).
    virtual const double* get_fourier_basis_yz();

    /// Get negative frequency index mapping.
    virtual const int* get_negative_frequency_mapping();

    /**
     * @brief Compute the tables of a cell.
     *
     * The Boltzmann bond factors of every contour step and monomer type, and, with
     * with_stress_basis, the stress basis (its tables are released otherwise).
     *
     * @param dx                Grid spacings (the lengths of non-periodic axes are nx·dx)
     * @param recip_metric      Reciprocal metric tensor (periodic BC)
     *                          [g^{-1}_11, g^{-1}_12, g^{-1}_13, g^{-1}_22, g^{-1}_23, g^{-1}_33]
     * @param with_stress_basis Whether to compute the stress basis
     */
    virtual void update(const std::vector<double>& dx, const std::array<double, 6>& recip_metric,
                        bool with_stress_basis = true);
};

#endif
