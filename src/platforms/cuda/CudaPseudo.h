/**
 * @file CudaPseudo.h
 * @brief GPU implementation of pseudo-spectral method utilities.
 *
 * This header provides CudaPseudo, the GPU-specific implementation
 * of Pseudo that stores Boltzmann bond factors and Fourier basis vectors
 * in GPU device memory.
 *
 * **GPU Memory Contents:**
 *
 * - d_boltz_bond: exp(-b²|k|²ds/6) for full step propagation
 *     - Continuous chains: diffusion propagator
 *     - Discrete chains: bond function ĝ(k)
 * - d_boltz_bond_half: exp(-b²|k|²ds/12) for half steps
 *     - Continuous chains: half-step diffusion
 *     - Discrete chains: half-bond function ĝ^(1/2)(k)
 * - d_fourier_basis_*: stress basis, v⊗v of the deformation vector v of each mode
 * - d_negative_k_idx: Index mapping for Hermitian symmetry
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * The stress is stored as a 6-component array:
 * - Index 0: σ₁ → drives L₁ optimization
 * - Index 1: σ₂ → drives L₂ optimization
 * - Index 2: σ₃ → drives L₃ optimization
 * - Index 3: σ₁₂ → drives γ (angle between a₁ and a₂) optimization
 * - Index 4: σ₁₃ → drives β (angle between a₁ and a₃) optimization
 * - Index 5: σ₂₃ → drives α (angle between a₂ and a₃) optimization
 *
 * @see Pseudo for the abstract interface
 * @see CudaSolverPseudoRQM4 for usage in continuous chains
 * @see CudaSolverPseudoDiscrete for usage in discrete chains
 * @see docs/theory/StressTensor.md for detailed derivation
 */

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Pseudo.h"
#include "CudaCommon.h"
#include "Exception.h"
#include "ComputationBox.h"

/**
 * @class CudaPseudo
 * @brief GPU implementation of pseudo-spectral utilities.
 *
 * Stores pseudo-spectral operators in GPU device memory for efficient
 * access during propagator computation. update() computes the host tables
 * of a cell and copies them to the device, allocating the device arrays at
 * its first call (and releasing those of a table it no longer computes); the
 * getters return the device arrays.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Boltzmann Bond Factors:**
 *
 * Stored in Fourier space for efficient multiplication. Same formula
 * for both chain models but different physical interpretation:
 *
 * - boltz_bond[k] = exp(-b²|k|²ds/6)
 *     - Continuous: diffusion propagator
 *     - Discrete: bond function ĝ(k) from Chapman-Kolmogorov equation
 * - boltz_bond_half[k] = exp(-b²|k|²ds/12)
 *     - Continuous: half-step diffusion
 *     - Discrete: half-bond function ĝ^(1/2)(k) for chain ends
 *
 * **Stress Basis:**
 *
 * v⊗v of the deformation vector v = 2π g⁻¹ m of each mode, with the mode's weight
 * in the sum over the spectrum (see Pseudo), when update() computes it
 * (with_stress_basis):
 * - d_fourier_basis_x = v₁², _y = v₂², _z = v₃²
 * - d_fourier_basis_xy = v₁v₂, _xz = v₁v₃, _yz = v₂v₃ (periodic boxes)
 */
template <typename T>
class CudaPseudo : public Pseudo<T>
{
private:
    /// @name Stress Calculation Arrays (v⊗v Diagonal)
    /// @{
    CudaDeviceBuffer<double> d_fourier_basis_x;  ///< v₁² in device memory
    CudaDeviceBuffer<double> d_fourier_basis_y;  ///< v₂² in device memory
    CudaDeviceBuffer<double> d_fourier_basis_z;  ///< v₃² in device memory
    /// @}

    /// @name Stress Calculation Arrays (v⊗v Off-diagonal, periodic BC)
    /// @{
    CudaDeviceBuffer<double> d_fourier_basis_xy;  ///< v₁ v₂ in device memory
    CudaDeviceBuffer<double> d_fourier_basis_xz;  ///< v₁ v₃ in device memory
    CudaDeviceBuffer<double> d_fourier_basis_yz;  ///< v₂ v₃ in device memory
    /// @}

    CudaDeviceBuffer<int> d_negative_k_idx;  ///< Index map for negative frequencies (device)

    /// @name Pseudo-Spectral Operators (per ds_index)
    /// @{
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_boltz_bond;       ///< Full bond diffusion (device) [ds_index][monomer_type]
    std::map<int, std::map<std::string, CudaDeviceBuffer<double>>> d_boltz_bond_half;  ///< Half bond diffusion (device) [ds_index][monomer_type]
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral utility (see Pseudo::Pseudo).
     *
     * The tables follow at the first update().
     *
     * @param bond_lengths Relative statistical segment lengths by monomer type (not squared)
     * @param bc           One boundary condition per dimension
     * @param nx           Grid dimensions
     * @param ds_values    Contour step of each ds_index with Boltzmann tables (none when empty)
     */
    CudaPseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc, std::vector<int> nx,
        std::map<int, double> ds_values);

    /// Destructor. Frees GPU memory.
    ~CudaPseudo();

    /**
     * @brief Get full bond factor (device pointer).
     * @param monomer_type Monomer type
     * @param ds_index Index for ds value (0-based)
     * @return Device pointer to exp(-b²|k|²ds/6)
     */
    double* get_boltz_bond(const std::string& monomer_type, int ds_index) override { return d_boltz_bond.at(ds_index).at(monomer_type).get(); };

    /**
     * @brief Get half bond factor (device pointer).
     * @param monomer_type Monomer type
     * @param ds_index Index for ds value (0-based)
     * @return Device pointer to exp(-b²|k|²ds/12)
     */
    double* get_boltz_bond_half(const std::string& monomer_type, int ds_index) override { return d_boltz_bond_half.at(ds_index).at(monomer_type).get();};

    /// Get x-direction Fourier basis (device pointer).
    const double* get_fourier_basis_x() override { return d_fourier_basis_x;};

    /// Get y-direction Fourier basis (device pointer).
    const double* get_fourier_basis_y() override { return d_fourier_basis_y;};

    /// Get z-direction Fourier basis (device pointer).
    const double* get_fourier_basis_z() override { return d_fourier_basis_z;};

    /// Get negative frequency mapping (device pointer).
    const int* get_negative_frequency_mapping() override { return d_negative_k_idx;};

    /// Get xy cross-term Fourier basis (device pointer).
    const double* get_fourier_basis_xy() override { return d_fourier_basis_xy;};

    /// Get xz cross-term Fourier basis (device pointer).
    const double* get_fourier_basis_xz() override { return d_fourier_basis_xz;};

    /// Get yz cross-term Fourier basis (device pointer).
    const double* get_fourier_basis_yz() override { return d_fourier_basis_yz;};

    /**
     * @brief Compute the tables of a cell (Pseudo::update) and copy them to the device.
     *
     * @param dx                Grid spacings
     * @param recip_metric      Reciprocal metric tensor
     * @param with_stress_basis Whether to compute the stress basis
     */
    void update(const std::vector<double>& dx, const std::array<double, 6>& recip_metric,
                bool with_stress_basis = true) override;
};
#endif
