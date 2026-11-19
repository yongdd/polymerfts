/**
 * @file FftwCrysFFTObliqueZ.h
 * @brief FFTW implementation of crystallographic FFT for z-mirror symmetry.
 *
 * Uses native FFTW3 DCT-II/III (REDFT10/REDFT01).
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see MklCrysFFTObliqueZ for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_OBLIQUEZ_H_
#define FFTW_CRYS_FFT_OBLIQUEZ_H_

#include <fftw3.h>
#include <vector>
#include "CrysFFTObliqueZBase.h"

class FftwCrysFFTObliqueZ : public CrysFFTObliqueZBase
{
private:
    using Base = CrysFFTObliqueZBase;

    fftw_plan plan_dct_forward_z_{nullptr};
    fftw_plan plan_dct_backward_z_{nullptr};
    fftw_plan plan_fft_forward_xy_{nullptr};
    fftw_plan plan_fft_backward_xy_{nullptr};

    void initFFTPlans();

    /// Destroy the plans (also after a failed construction); idempotent.
    void release();

    /// DCT-II along z (REDFT10)
    void dct2_execute(const double* in, double* out) const;
    /// DCT-III along z (REDFT01)
    void dct3_execute(const double* in, double* out) const;
    /// DCT-II(z) -> r2c FFT(xy) -> multiply -> c2r FFT(xy) -> DCT-III(z) -> normalize.
    void apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const;

public:
    explicit FftwCrysFFTObliqueZ(std::array<int, 3> nx_logical);

    ~FftwCrysFFTObliqueZ();

    /// q_out = exp(ds ∇²) q_in with the table of build_boltzmann(ds) (M_complex values).
    void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const override;
};

#endif  // FFTW_CRYS_FFT_OBLIQUEZ_H_
