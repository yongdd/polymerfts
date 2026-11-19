/**
 * @file FftwFFT.h
 * @brief FFTW3 implementation of the periodic FFT and of the real FFT under MakhoulTransform.
 *
 * FftwFFT is the periodic FFT of the FFT<T> interface (r2c/c2r for real
 * fields, c2c for complex fields); the complex array of a real field has
 * size Nx × Ny × (Nz/2+1) for Hermitian symmetry. Boxes of reflecting and
 * absorbing axes run MakhoulFFT over FftwRealFFT instead (createFFT()).
 *
 * @see FFT for the abstract interface
 * @see MakhoulTransform for the DCT and DST
 * @see CudaFFT for GPU implementation
 */

#ifndef FFTW_FFT_H_
#define FFTW_FFT_H_

#include <array>
#include <memory>
#include <vector>
#include <complex>
#include <fftw3.h>
#include "FFT.h"
#include "MakhoulTransform.h"

/**
 * @brief Whether FFTW's new-array execute functions may run a plan on the array p.
 *
 * They take arrays of the alignment of the planning arrays, which come from
 * fftw_malloc, for which fftw_alignment_of() is 0. The scratch of the transforms
 * (AlignedScratch) starts on a 64-byte boundary and has that alignment as well.
 */
inline bool has_fftw_plan_alignment(const void* p)
{
    return fftw_alignment_of(static_cast<double*>(const_cast<void*>(p))) == 0;
}

/// Deleter of a planning array of fftw_malloc held by a std::unique_ptr
struct FftwFree
{
    void operator()(void* p) const { fftw_free(p); }
};

/**
 * @class FftwRealFFT
 * @brief FFTW's unnormalized real FFT of a grid, the FFT under MakhoulTransform.
 *
 * The plans run on the scratch of MakhoulTransform through FFTW's new-array
 * execute functions, without copies; the arrays they were planned on are freed.
 */
class FftwRealFFT : public MakhoulTransform::RealFFT
{
private:
    fftw_plan forward_ = nullptr;    ///< r2c; may overwrite its input
    fftw_plan backward_ = nullptr;   ///< c2r; overwrites its input

    void release();

public:
    /// Plans of the real FFT of a row-major grid nx (one to three axes)
    explicit FftwRealFFT(const std::vector<int>& nx);
    ~FftwRealFFT() override;
    FftwRealFFT(const FftwRealFFT&) = delete;
    FftwRealFFT& operator=(const FftwRealFFT&) = delete;

    void forward(double* samples, std::complex<double>* modes) const override;
    void backward(std::complex<double>* modes, double* samples) const override;
};

/**
 * @class FftwFFT
 * @brief FFTW3-based periodic FFT.
 *
 * @tparam T   Input type for forward transform (double or std::complex<double>)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 *
 * **Thread Safety:**
 *
 * FFTW plan execution is thread-safe, but plan creation/destruction is not.
 * All plans are created in the constructor and destroyed in the destructor.
 *
 * **Arrays of the transforms:**
 *
 * The plans run on the caller's arrays through FFTW's new-array execute
 * functions when FFTW allows it: the array has the alignment of the planning
 * arrays (fftw_alignment_of() is 0) and the input and output do not overlap.
 * Otherwise, as for a packed field after an odd number of grid points, the
 * array goes through thread-local scratch that starts on a 64-byte boundary
 * (AlignedScratch), which has that alignment too; a call takes scratch only for
 * the arrays it copies. The input of the complex-to-real transform
 * goes through the scratch, since that transform overwrites it, unless the
 * caller gives it up (backward_with_factors(cdata, nullptr, rdata, r_factor)). The same plan
 * thus always runs on arrays of one alignment class, and the result does not
 * depend on where the caller's arrays lie.
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so backward(forward(x)) = x.
 */
template <typename T, int DIM>
class FftwFFT : public FFT<T>
{
private:
    std::array<int, DIM> nx_;                      ///< Grid dimensions
    int total_grid_;                               ///< Total real grid size
    int total_complex_grid_;                       ///< Total complex grid size

    // FFTW plans, made on fftw_malloc arrays that are freed after planning
    fftw_plan plan_forward_ = nullptr;             ///< Forward (r2c or c2c); keeps its input
    fftw_plan plan_backward_ = nullptr;            ///< Backward (c2r, which overwrites its input, or c2c)

    /// Destroy the plans (also after a failed construction); idempotent.
    void release();

    /// Forward transform: modes = FFT of field (half spectrum for a real field)
    void periodic_forward(const T* field, std::complex<double>* modes) const;

    /**
     * @brief Backward transform with the 1/N normalization: field = r_factor · IFFT of (k_factor · modes).
     *
     * k_factor and r_factor may be nullptr. A k_factor is applied while the modes are copied
     * into the scratch; the modes themselves are overwritten only when keep_modes is false.
     */
    void periodic_backward(std::complex<double>* modes, const double* k_factor, T* field, const T* r_factor,
                           bool keep_modes) const;

    /// Initialize the FFTW plans.
    void initPeriodicFFT();

public:
    /**
     * @brief Construct the FFTW FFT of a periodic grid.
     *
     * @param nx Grid dimensions [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D
     */
    FftwFFT(std::array<int, DIM> nx);

    /// Destructor. Frees the FFTW plans.
    ~FftwFFT();

    /**
     * @brief Perform forward FFT (real coefficient interface).
     *
     * @param rdata Input: real array of size total_grid
     * @param cdata Output: interleaved real/imaginary coefficients (2 × total_complex_grid doubles)
     */
    void forward(T *rdata, double *cdata) override;

    /**
     * @brief Perform backward FFT with the 1/N normalization (real coefficient interface).
     *
     * @param cdata Input: interleaved real/imaginary coefficients
     * @param rdata Output: real array
     */
    void backward(double *cdata, T *rdata) override;

    /**
     * @brief rdata = r_factor · backward(k_factor · cdata) (see FFT<T>).
     *
     * A k_factor is multiplied into the scratch, and the 1/N and r_factor are
     * applied in one pass; without a k_factor the complex-to-real transform
     * runs on cdata when FFTW can, overwriting it.
     */
    void backward_with_factors(double *cdata, const double *k_factor, T *rdata, const T *r_factor) override;
};

#endif
