/**
 * @file FftwCrysFFTRecursive3m.h
 * @brief Recursive crystallographic FFT (2x2y2z) over the real FFT of FFTW.
 *
 * @see CrysFFTRecursive3mBase for the algorithm
 * @see MklCrysFFTRecursive3m for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_RECURSIVE_3M_H_
#define FFTW_CRYS_FFT_RECURSIVE_3M_H_

#include <fftw3.h>
#include "CrysFFTRecursive3mBase.h"

class FftwCrysFFTRecursive3m : public CrysFFTRecursive3mBase
{
private:
    fftw_plan plan_forward_ = nullptr;
    fftw_plan plan_backward_ = nullptr;

    void init_fft_plans();

    /// Destroy the plans (also after a failed construction); idempotent.
    void release();

    /// The plans run on the caller's fields when they have the alignment of the planning
    /// arrays (the r2c transform keeps its input), and otherwise on a copy.
    void forward(const double* q, std::complex<double>* spectrum) const override;
    void backward(std::complex<double>* spectrum, double* q) const override;

public:
    explicit FftwCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~FftwCrysFFTRecursive3m();
};

#endif  // FFTW_CRYS_FFT_RECURSIVE_3M_H_
