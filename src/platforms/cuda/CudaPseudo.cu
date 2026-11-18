/**
 * @file CudaPseudo.cu
 * @brief CUDA implementation of pseudo-spectral method base class.
 *
 * Extends Pseudo base class with GPU device memory for Boltzmann bond
 * factors and Fourier basis arrays used in stress calculations.
 *
 * **Device Arrays:**
 *
 * - d_boltz_bond: exp(-b²|k|²ds/6) for full step propagation
 *     - Continuous chains: diffusion propagator
 *     - Discrete chains: bond function ĝ(k)
 * - d_boltz_bond_half: exp(-b²|k|²ds/12) for half step
 *     - Continuous chains: half-step diffusion
 *     - Discrete chains: half-bond function ĝ^(1/2)(k)
 * - d_fourier_basis_x/y/z: Fourier basis for stress computation
 * - d_negative_k_idx: Mapping for conjugate symmetry (complex fields)
 *
 * **Update Mechanism:**
 *
 * update() computes the host tables of a cell and copies them to device
 * memory, allocating the device arrays at its first call. The solvers create
 * this object when a full-grid path first needs it: one whose propagators and
 * stress take the CrysFFT path holds none of these full-grid arrays.
 *
 * **Template Instantiations:**
 *
 * - CudaPseudo<double>: Real field computations
 * - CudaPseudo<std::complex<double>>: Complex field computations
 *
 * @see Pseudo for base class interface
 * @see CudaSolverPseudoRQM4 for continuous chain solver
 * @see CudaSolverPseudoDiscrete for discrete chain solver
 */

#include <iostream>
#include <cmath>
#include <numbers>

#include "CudaCommon.h"
#include "CudaPseudo.h"

//----------------- Constructor -----------------------------
template <typename T>
CudaPseudo<T>::CudaPseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx,
    std::map<int, double> ds_values)
        : Pseudo<T>(bond_lengths, bc, nx, ds_values)
{
    // The device arrays are owning buffers, allocated by the first update()
}
//----------------- Destructor -----------------------------
template <typename T>
CudaPseudo<T>::~CudaPseudo() = default;

namespace {
/// Copy a host table into `slot`, allocated first when it does not hold as many values
/// (an empty table releases the slot)
template <typename V>
void upload_table(CudaDeviceBuffer<V>& slot, const std::vector<V>& h_values)
{
    if (slot.size() != h_values.size())
        slot.allocate(h_values.size());
    if (!h_values.empty())
        CUDA_CHECK(cudaMemcpy(slot, h_values.data(), sizeof(V) * h_values.size(), cudaMemcpyHostToDevice));
}
}  // namespace

//----------------- Update -----------------------------
template <typename T>
void CudaPseudo<T>::update(const std::vector<double>& dx, const std::array<double, 6>& recip_metric,
                           bool with_stress_basis)
{
    Pseudo<T>::update(dx, recip_metric, with_stress_basis);

    // Boltzmann bond factors of every ds_index and monomer type
    for (const auto& [ds_idx, by_type] : this->boltz_bond)
        for (const auto& [monomer_type, table] : by_type)
            upload_table(d_boltz_bond[ds_idx][monomer_type], table);
    for (const auto& [ds_idx, by_type] : this->boltz_bond_half)
        for (const auto& [monomer_type, table] : by_type)
            upload_table(d_boltz_bond_half[ds_idx][monomer_type], table);

    // The stress basis (empty without it; no cross terms for non-periodic BC) and the
    // negative-frequency map of complex fields with periodic BC
    upload_table(d_fourier_basis_x, this->fourier_basis_x);
    upload_table(d_fourier_basis_y, this->fourier_basis_y);
    upload_table(d_fourier_basis_z, this->fourier_basis_z);
    upload_table(d_fourier_basis_xy, this->fourier_basis_xy);
    upload_table(d_fourier_basis_xz, this->fourier_basis_xz);
    upload_table(d_fourier_basis_yz, this->fourier_basis_yz);
    upload_table(d_negative_k_idx, this->negative_k_idx);
}

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
