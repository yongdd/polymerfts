/**
 * @file CudaCrysFFTPmmm.cu
 * @brief CUDA implementation of crystallographic FFT using DCT-II/III.
 *
 * The DCT-II and DCT-III of the physical grid run through CudaMakhoulTransform; the
 * Boltzmann factor or multiplier of each coefficient is applied as the DCT-III reads it.
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include <string>
#include <vector>

#include "CudaCrysFFTPmmm.h"
#include "CrysFFTTables.h"

CudaCrysFFTPmmm::CudaCrysFFTPmmm(std::array<int, 3> nx_logical)
    : nx_logical_(nx_logical)
{
    // The owning members release what was acquired if a step below throws.
    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
            throw_with_line_number("CudaCrysFFTPmmm requires even grid dimensions.");
    }

    // Physical grid: (N/2) in each dimension
    nx_physical_ = {
        nx_logical_[0] / 2,
        nx_logical_[1] / 2,
        nx_logical_[2] / 2
    };

    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

    // Normalization: 1/M_logical for round-trip
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    const std::vector<int> nx(nx_physical_.begin(), nx_physical_.end());
    dct_ = std::make_unique<CudaMakhoulTransform<double>>(nx, std::vector<BoundaryCondition>(3, BoundaryCondition::REFLECTING));
    d_coefficients_.allocate(M_physical_);
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTPmmm::clone() const
{
    return std::make_unique<CudaCrysFFTPmmm>(nx_logical_);
}

//------------------------------------------------------------------------------
// Tables, computed on the host with the normalization (1/M_logical) of the
// round trip, which the diffusion's result then needs no separate scaling for
//------------------------------------------------------------------------------
void CudaCrysFFTPmmm::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                      double coeff, cudaStream_t stream) const
{
    std::vector<double> h_boltz;
    pmmm_table(h_boltz, nx_physical_, cell, [this, coeff](double kx2, double ky2, double kz2) {
        return crysfft_boltzmann_value(coeff, kx2, ky2, kz2) * norm_factor_;
    });
    upload_table(table, h_boltz, stream);
}

/// Stress multiplier exp(-k²·coeff)·Σ k_d² over an axis mask on the physical DCT grid
void CudaCrysFFTPmmm::build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                       int axes, double coeff, cudaStream_t stream) const
{
    check_crysfft_axes(axes);
    std::vector<double> h_mult;
    pmmm_table(h_mult, nx_physical_, cell, [this, axes, coeff](double kx2, double ky2, double kz2) {
        return crysfft_multiplier_value(axes, coeff, kx2, ky2, kz2) * norm_factor_;
    });
    upload_table(table, h_mult, stream);
}

//------------------------------------------------------------------------------
// Spectral step: DCT-II, the table, DCT-III
//------------------------------------------------------------------------------
void CudaCrysFFTPmmm::diffusion(const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                                cudaStream_t stream)
{
    if (static_cast<int>(table.size()) != M_physical_)
        throw_with_line_number("The CrysFFT table has " + std::to_string(table.size()) + " values for a physical grid of "
                               + std::to_string(M_physical_) + " points.");
    // The DCT-II keeps d_q_in and the DCT-III reads only the coefficients: d_q_out may be d_q_in
    dct_->forward(d_q_in, d_coefficients_.get(), 1.0, stream);
    dct_->backward(d_coefficients_.get(), table.get(), d_q_out, 1.0, stream);
}
