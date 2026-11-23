/**
 * @file CudaCrysFFTPmmm.h
 * @brief CUDA Crystallographic FFT using DCT-II/III for Pmmm symmetry.
 *
 * The DCT-II/III of the physical grid run through CudaMakhoulTransform, as
 * CrysFFTPmmm runs them through MakhoulTransform on the CPU; the tables come from
 * CrysFFTTables.h.
 *
 * **Grid size:**
 *
 * Logical grid: Nx × Ny × Nz (full simulation box, must be even)
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2) (1/8 of logical grid)
 *
 * **Reference:**
 *
 * Qiang & Li, "Accelerated pseudo-spectral method of self-consistent field
 * theory via crystallographic fast Fourier transform", Macromolecules (2020)
 */

#ifndef CUDA_CRYS_FFT_PMMM_H_
#define CUDA_CRYS_FFT_PMMM_H_

#include <array>
#include <memory>
#include <cuda_runtime.h>

#include "CudaCommon.h"
#include "CudaCrysFFTBase.h"
#include "CudaMakhoulTransform.h"

/**
 * @class CudaCrysFFTPmmm
 * @brief CUDA crystallographic FFT using DCT-II/III.
 *
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2). Its tables hold M_physical values
 * with the round-trip normalization 1/M_logical: the Boltzmann factors and the
 * stress multipliers.
 */
class CudaCrysFFTPmmm : public CudaCrysFFTBase
{
private:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    /// DCT-II (FFTW's REDFT10) and DCT-III (REDFT01) of the physical grid. Their round
    /// trip gains 2n per dimension, which norm_factor_ in the tables removes.
    std::unique_ptr<CudaMakhoulTransform<double>> dct_;
    CudaDeviceBuffer<double> d_coefficients_;   ///< DCT-II coefficients of the physical grid

    // Normalization factor
    double norm_factor_;

public:
    /**
     * @brief Construct CudaCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     */
    explicit CudaCrysFFTPmmm(std::array<int, 3> nx_logical);

    std::unique_ptr<CudaCrysFFTBase> clone() const override;
    void build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                         double coeff, cudaStream_t stream) const override;

    /// DCT-II -> the table (normalization included) -> DCT-III.
    void diffusion(const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                   cudaStream_t stream) override;

    void build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff, cudaStream_t stream) const override;
};

#endif  // CUDA_CRYS_FFT_PMMM_H_
