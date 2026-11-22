/**
 * @file CrysFFTPmmm.h
 * @brief Crystallographic FFT with Pmmm symmetry on the CPU (FFTW or MKL).
 *
 * The DCT-II/III of the physical grid run through MakhoulTransform over the
 * periodic real FFT of the selected library (FftwRealFFT or MklRealFFT). The
 * tables hold M_physical values without the round-trip normalization 1/M_logical,
 * which the transforms apply.
 */

#ifndef CRYS_FFT_PMMM_H_
#define CRYS_FFT_PMMM_H_

#include <array>
#include <memory>
#include <vector>

#include "CpuCrysFFT.h"
#include "CrysFFTTables.h"
#include "Exception.h"
#include "FFTFactory.h"
#include "MakhoulTransform.h"

/**
 * @class CrysFFTPmmm
 * @brief Pmmm crystallographic FFT: diffusion through the DCT-II/III of the physical grid.
 */
class CrysFFTPmmm : public CpuCrysFFT
{
private:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // Normalization factor for DCT-II/III round trip: 1/M_logical
    double norm_factor_;

    /// DCT-II (FFTW's REDFT10) and DCT-III (REDFT01) of the physical grid. Their
    /// round trip gains 2n per dimension, which norm_factor_ removes.
    std::unique_ptr<MakhoulTransform> dct_;

    /// q_out = DCT-III[ multiplier · DCT-II[q_in] ] / M_logical
    void apply_spectral(const double* q_in, double* q_out, const std::vector<double>& multiplier) const
    {
        check_table(multiplier, M_physical_);
        // DCT-II into q_out, then DCT-III of the coefficients times the multiplier, with
        // normalization, in place (MakhoulTransform reads its whole input before it writes)
        dct_->forward(q_in, q_out, 1.0);
        dct_->backward_with_factors(q_out, multiplier.data(), q_out, nullptr, norm_factor_);
    }

public:
    /**
     * @brief Construct CrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param backend    Library of the real FFT under the DCT (FFTW or MKL)
     */
    CrysFFTPmmm(std::array<int, 3> nx_logical, FFTBackend backend)
        : nx_logical_(nx_logical)
    {
        // Validate even grid sizes
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] % 2 != 0)
            {
                throw_with_line_number("CrysFFTPmmm requires even grid dimensions. "
                    "Dimension " + std::to_string(d) + " has size " + std::to_string(nx_logical_[d]));
            }
            if (nx_logical_[d] <= 0)
            {
                throw_with_line_number("CrysFFTPmmm requires positive grid dimensions.");
            }
        }

        // Physical grid: (N/2) in each dimension
        nx_physical_ = {
            nx_logical_[0] / 2,
            nx_logical_[1] / 2,
            nx_logical_[2] / 2
        };

        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

        // DCT-II/III normalization: 1 / M_logical for round-trip
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);

        const std::vector<int> nx(nx_physical_.begin(), nx_physical_.end());
        const std::vector<BoundaryCondition> cosine(3, BoundaryCondition::REFLECTING);
        dct_ = std::make_unique<MakhoulTransform>(nx, cosine, createRealFFT(nx, backend));
    }

    int get_M_physical() const override { return M_physical_; }

    /// exp(-k² ds) on the physical grid, k_d = 2π m_d / L_d with m_d < N_d/2 (CrysFFTTables.h).
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double ds) const override
    {
        build_pmmm_boltzmann(table, nx_physical_, cell, ds);
    }

    /**
     * @brief Apply diffusion operator using DCT-II/III.
     *
     * Computes: q_out = DCT-III[ exp(-k²ds) * DCT-II[q_in] ]
     */
    void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        apply_spectral(q_in, q_out, table);
    }

    bool has_multipliers() const override { return true; }

    /// Σ_{d in axes} k_d², times exp(-k² coeff) when coeff is not 0 (the bond
    /// factor of discrete chains), on the physical grid (CrysFFTTables.h).
    void build_multiplier(std::vector<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff) const override
    {
        build_pmmm_multiplier(table, nx_physical_, cell, axes, coeff);
    }

    /// q_out = DCT-III[ multiplier · DCT-II[q_in] ], as diffusion().
    void apply_multiplier(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        apply_spectral(q_in, q_out, table);
    }
};

#endif  // CRYS_FFT_PMMM_H_
