/**
 * @file MklFFT.h
 * @brief Intel MKL implementation of the periodic FFT and of the real FFT under MakhoulTransform.
 *
 * MklFFT is the periodic FFT of the FFT<T> interface through MKL's DFTI
 * (r2c/c2r for real fields, c2c for complex fields); the complex array of a
 * real field has size Nx * Ny * (Nz/2+1) for Hermitian symmetry. Boxes of
 * reflecting and absorbing axes run MakhoulFFT over MklRealFFT instead
 * (createFFT()).
 *
 * @see FFT for the abstract interface
 * @see MakhoulTransform for the DCT and DST
 * @see CudaFFT for GPU implementation
 */

#ifndef MKL_FFT_H_
#define MKL_FFT_H_

#include <array>
#include <memory>
#include <vector>
#include <complex>
#include "FFT.h"
#include "MakhoulTransform.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

/**
 * @class MklRealFFT
 * @brief MKL's unnormalized real FFT of a grid, the FFT under MakhoulTransform.
 *
 * Out-of-place DFTI descriptors without a backward scale, run on the scratch of
 * MakhoulTransform.
 */
class MklRealFFT : public MakhoulTransform::RealFFT
{
private:
    DFTI_DESCRIPTOR_HANDLE forward_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE backward_ = nullptr;

    void release();

public:
    /// Descriptors of the real FFT of a row-major grid nx (one to three axes)
    explicit MklRealFFT(const std::vector<int>& nx);
    ~MklRealFFT() override;
    MklRealFFT(const MklRealFFT&) = delete;
    MklRealFFT& operator=(const MklRealFFT&) = delete;

    void forward(double* samples, std::complex<double>* modes) const override;
    void backward(std::complex<double>* modes, double* samples) const override;
};

/**
 * @class MklFFT
 * @brief Intel MKL-based periodic FFT.
 *
 * Out-of-place DFTI descriptors that run on the caller's arrays; the backward
 * transform includes the 1/N normalization.
 *
 * @tparam T   Input type for forward transform (double or std::complex<double>)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 */
template <typename T, int DIM>
class MklFFT : public FFT<T>
{
private:
    // MKL DFTI descriptors
    DFTI_DESCRIPTOR_HANDLE hand_forward_ = nullptr;   ///< Forward (r2c or c2c) transform
    DFTI_DESCRIPTOR_HANDLE hand_backward_ = nullptr;  ///< Backward (c2r or c2c) transform
    int total_grid_;                                  ///< Total real grid size
    int total_complex_grid_;                          ///< Total complex grid size

    /// Free the DFTI descriptors (also after a failed construction); idempotent.
    void release();

public:
    /**
     * @brief Construct the MKL FFT of a periodic grid.
     *
     * @param nx Grid dimensions [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D
     */
    MklFFT(std::array<int, DIM> nx);

    /// Destructor. Frees the MKL DFTI descriptors.
    ~MklFFT();

    /**
     * @brief Perform forward FFT (real coefficient interface).
     *
     * @param rdata Input: real array of size total_grid
     * @param cdata Output: interleaved real/imaginary coefficients (2 * total_complex_grid doubles)
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
     * Multiplies cdata by k_factor in place, runs backward() and multiplies
     * rdata by r_factor.
     */
    void backward_with_factors(double *cdata, const double *k_factor, T *rdata, const T *r_factor) override;
};

#endif
