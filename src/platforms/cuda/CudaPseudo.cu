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
 * update() recalculates the Fourier space operators for the current box and
 * copies them to device memory, allocating the device arrays at its first
 * call: a solver whose propagators and stress take the CrysFFT path never
 * calls it and holds none of these full-grid arrays.
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
    std::vector<int> nx, std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
        : Pseudo<T>(bond_lengths, bc, nx, dx, recip_metric, recip_vec)
{
    // The device arrays are owning buffers, allocated by the first update()
}
//----------------- Destructor -----------------------------
template <typename T>
CudaPseudo<T>::~CudaPseudo() = default;

namespace {
/// Copy a host table of n values into `slot`, allocated first when it does not hold n values
void upload_table(CudaDeviceBuffer<double>& slot, const double* h_values, int n)
{
    if (slot.size() != static_cast<size_t>(n))
        slot.allocate(n);
    CUDA_CHECK(cudaMemcpy(slot, h_values, sizeof(double) * n, cudaMemcpyHostToDevice));
}
}  // namespace

template <typename T>
void CudaPseudo<T>::upload_boltz_bond()
{
    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Upload all ds_index values: boltz_bond[ds_index][monomer_type]
    for (const auto& ds_entry : this->boltz_bond)
        for (const auto& item : ds_entry.second)
            upload_table(d_boltz_bond[ds_entry.first][item.first], item.second.data(), M_COMPLEX);

    for (const auto& ds_entry : this->boltz_bond_half)
        for (const auto& item : ds_entry.second)
            upload_table(d_boltz_bond_half[ds_entry.first][item.first], item.second.data(), M_COMPLEX);
}

//----------------- Upload Fourier basis to GPU -----------------------------
template <typename T>
void CudaPseudo<T>::upload_fourier_basis()
{
    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Diagonal terms
    upload_table(d_fourier_basis_x, this->fourier_basis_x.data(), M_COMPLEX);
    upload_table(d_fourier_basis_y, this->fourier_basis_y.data(), M_COMPLEX);
    upload_table(d_fourier_basis_z, this->fourier_basis_z.data(), M_COMPLEX);
    // Cross-terms for non-orthogonal systems
    upload_table(d_fourier_basis_xy, this->fourier_basis_xy.data(), M_COMPLEX);
    upload_table(d_fourier_basis_xz, this->fourier_basis_xz.data(), M_COMPLEX);
    upload_table(d_fourier_basis_yz, this->fourier_basis_yz.data(), M_COMPLEX);

    // negative_k_idx for complex fields with periodic BC
    if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        // The negative-frequency mapping exists only for periodic BCs
        // (conjugate symmetry of the full complex FFT); for non-periodic
        // boxes the host pointer is nullptr, nothing is uploaded and
        // get_negative_frequency_mapping() returns nullptr on both sides.
        const int* negative_k_idx = Pseudo<T>::get_negative_frequency_mapping();
        if (negative_k_idx != nullptr)
        {
            if (d_negative_k_idx.size() != static_cast<size_t>(M_COMPLEX))
                d_negative_k_idx.allocate(M_COMPLEX);
            CUDA_CHECK(cudaMemcpy(d_negative_k_idx, negative_k_idx,
                                       sizeof(int) * M_COMPLEX, cudaMemcpyHostToDevice));
        }
    }
}

//----------------- Update -----------------------------
template <typename T>
void CudaPseudo<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
{
    Pseudo<T>::update(bc, bond_lengths, dx, recip_metric, recip_vec);

    // Upload updated data to GPU
    upload_boltz_bond();
    upload_fourier_basis();
}

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
