/**
 * @file CudaCrysFFT.cu
 * @brief CUDA implementation of crystallographic FFT using DCT-II/III.
 *
 * The DCT-II and DCT-III of the physical grid run through CudaMakhoulTransform; the
 * Boltzmann factor or multiplier of each coefficient is applied as the DCT-III reads it.
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include <string>
#include <vector>

#include "CudaCrysFFT.h"
#include "CrysFFTTables.h"

CudaCrysFFT::CudaCrysFFT(std::array<int, 3> nx_logical)
    : nx_logical_(nx_logical)
{
    // The owning members release what was acquired if a step below throws.
    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
            throw_with_line_number("CudaCrysFFT requires even grid dimensions.");
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

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFT::clone() const
{
    return std::make_unique<CudaCrysFFT>(nx_logical_);
}

//------------------------------------------------------------------------------
// Boltzmann factors
//------------------------------------------------------------------------------
void CudaCrysFFT::build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                  double ds, cudaStream_t stream) const
{
    // Compute on host, then copy to device
    std::vector<double> h_boltz;
    build_pmmm_boltzmann(h_boltz, nx_physical_, cell, ds);

    // Pre-multiply normalization factor (1/M_logical) into Boltzmann
    // This eliminates a separate scaling of the diffusion's result
    for (double& value : h_boltz)
        value *= norm_factor_;

    upload_table(table, h_boltz, stream);
}

//------------------------------------------------------------------------------
// Spectral step: DCT-II, the table, DCT-III
//------------------------------------------------------------------------------
void CudaCrysFFT::apply_spectral(const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                                 double scale, cudaStream_t stream)
{
    if (static_cast<int>(table.size()) != M_physical_)
        throw_with_line_number("The CrysFFT table has " + std::to_string(table.size()) + " values for a physical grid of "
                               + std::to_string(M_physical_) + " points.");
    // The DCT-II keeps d_q_in and the DCT-III reads only the coefficients: d_q_out may be d_q_in
    dct_->forward(d_q_in, d_coefficients_.get(), 1.0, stream);
    dct_->backward(d_coefficients_.get(), table.get(), d_q_out, scale, stream);
}

void CudaCrysFFT::diffusion(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                            cudaStream_t stream)
{
    // Normalization is already included in Boltzmann factor (pre-multiplied in build_boltzmann)
    apply_spectral(d_q_in, d_q_out, table, 1.0, stream);
}

void CudaCrysFFT::apply_multiplier(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                                   cudaStream_t stream)
{
    // Raw multiplier with the round-trip normalization
    apply_spectral(d_q_in, d_q_out, table, norm_factor_, stream);
}

//------------------------------------------------------------------------------
// Stress multiplier exp(-k²·coeff)·Σ k_d² over an axis mask on the physical DCT grid
//------------------------------------------------------------------------------
void CudaCrysFFT::build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                   int axes, double coeff, cudaStream_t stream) const
{
    std::vector<double> h_mult;
    build_pmmm_multiplier(h_mult, nx_physical_, cell, axes, coeff);
    upload_table(table, h_mult, stream);
}
