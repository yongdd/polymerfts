/**
 * @file MklCrysFFTRecursive3m.h
 * @brief Recursive crystallographic FFT (2x2y2z) over the real FFT of MKL.
 *
 * @see CrysFFTRecursive3mBase for the algorithm
 * @see FftwCrysFFTRecursive3m for FFTW implementation
 */

#ifndef MKL_CRYS_FFT_RECURSIVE_3M_H_
#define MKL_CRYS_FFT_RECURSIVE_3M_H_

#include <complex>
#include "mkl_dfti.h"
#include "CrysFFTRecursive3mBase.h"

class MklCrysFFTRecursive3m : public CrysFFTRecursive3mBase
{
private:
    DFTI_DESCRIPTOR_HANDLE plan_forward_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE plan_backward_ = nullptr;

    void init_fft_plans();

    /// Free the DFTI descriptors (also after a failed construction); idempotent.
    void release();

    /// Out of place on the caller's fields; q is kept.
    void forward(const double* q, std::complex<double>* spectrum) const override;
    void backward(std::complex<double>* spectrum, double* q) const override;

public:
    explicit MklCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~MklCrysFFTRecursive3m();
};

#endif  // MKL_CRYS_FFT_RECURSIVE_3M_H_
