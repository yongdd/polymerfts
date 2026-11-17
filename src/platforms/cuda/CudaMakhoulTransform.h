/**
 * @file CudaMakhoulTransform.h
 * @brief DFT along periodic axes and cosine/sine series along walls through one cuFFT of the grid.
 *
 * The CUDA counterpart of MakhoulTransform with periodic axes, for boxes that mix
 * periodic axes and walls (CudaFourierTransform); docs/internals/FFTImplementation.md,
 * Section 2.5, gives the derivation.
 *
 * Each wall axis is reordered as in Makhoul's cosine transform: the even samples
 * forward, the odd samples backward, the odd samples of a sine axis negated. One
 * periodic cuFFT of the reordered grid follows, D2Z for a real field and Z2Z for
 * a complex one, which leaves the DFT V on the periodic axes. Along a wall axis of
 * N points the cosine transform of a complex line is
 *
 *     X_k = W^k V_k + W^-k V_{-k},    W^k = exp(-i pi k / (2N)),
 *
 * and a sine axis takes its coefficients in reverse order. One kernel forms each
 * coefficient from its 2^w modes (w wall axes), reading the half of a real
 * field's spectrum that D2Z does not hold as V(-k) = conj V(k). The backward
 * transform runs these steps in reverse, V_k = W^-k (X_k - i X_{N-k}) per wall
 * axis (twice the inverse, as REDFT01), from the Hermitian part
 * (C(k) + conj C(-k))/2 of a real field's coefficients, -k reversing only the
 * periodic indices; a real field is the real part of the inverse.
 *
 * The scratch (the reordered samples and their modes, a complex field's samples
 * transformed in place) and the cuFFT plans, which share one work area, are made
 * once by the constructor. An object runs one transform at a time, on the stream
 * of each call.
 */

#ifndef CUDA_MAKHOUL_TRANSFORM_H_
#define CUDA_MAKHOUL_TRANSFORM_H_

#include <complex>
#include <type_traits>
#include <vector>
#include <cufft.h>

#include "ComputationBox.h"
#include "CudaCommon.h"

/// Axes of a CudaMakhoulTransform, padded to three in front, passed to its kernels by value
struct CudaMakhoulAxes
{
    int n[3];                      ///< grid points of each axis (1 on padding axes)
    bool wall[3];                  ///< cosine or sine axis
    bool sine[3];                  ///< sine axis
    const cuDoubleComplex* w[3];   ///< W^k = exp(-i pi k / (2n)) for k < n on wall axes (device)
};

/**
 * @class CudaMakhoulTransform
 * @brief One cuFFT of the grid with Makhoul's reordering and combining on the wall axes.
 *
 * @tparam T Field type (double or std::complex<double>); the coefficients are complex.
 */
template <typename T>
class CudaMakhoulTransform
{
public:
    using DeviceT = CuDeviceData<T>;

    /**
     * @param nx Grid dimensions (one to three axes)
     * @param bc Transform of each axis: PERIODIC (DFT), REFLECTING (cosine) or
     *           ABSORBING (sine); at least one axis is periodic
     */
    CudaMakhoulTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc);

    CudaMakhoulTransform(const CudaMakhoulTransform&) = delete;
    CudaMakhoulTransform& operator=(const CudaMakhoulTransform&) = delete;

    /**
     * @brief out = scale * (unnormalized DFT along the periodic axes, REDFT10/RODFT10 along the walls) of in.
     *
     * @param d_in   Field on the device (kept)
     * @param d_out  Coefficients on the device, shape nx, 16-byte aligned, distinct from d_in
     * @param scale  Factor applied to the samples
     * @param stream Stream of the kernels and the cuFFT call
     */
    void forward(const DeviceT* d_in, cuDoubleComplex* d_out, double scale, cudaStream_t stream);

    /**
     * @brief out = scale * (unnormalized inverse DFT, REDFT01/RODFT01) of in; a real field is its real part.
     *
     * @param d_in   Coefficients on the device, shape nx (kept)
     * @param d_out  Field on the device, distinct from d_in
     * @param scale  Factor applied to the samples
     * @param stream Stream of the kernels and the cuFFT call
     */
    void backward(const cuDoubleComplex* d_in, DeviceT* d_out, double scale, cudaStream_t stream);

private:
    static constexpr bool REAL = std::is_same_v<T, double>;

    CudaMakhoulAxes axes_{};
    int n_grid_ = 0;    ///< grid points
    int n_modes_ = 0;   ///< modes of the cuFFT: the half spectrum of the last axis for a real field

    CudaDeviceBuffer<double> d_work_;             ///< work area of the cuFFT plans, shared by both
    CudaDeviceBuffer<cuDoubleComplex> d_w_;       ///< W^k of the three axes, one after the other
    CudaDeviceBuffer<double> d_samples_;          ///< reordered samples of a real field
    CudaDeviceBuffer<cuDoubleComplex> d_modes_;   ///< modes (a complex field: also its samples, in place)

    // Declared after their work area, so they are destroyed before it
    CufftPlan plan_;           ///< D2Z (real field) or Z2Z (complex field, both directions)
    CufftPlan plan_inverse_;   ///< Z2D (real field)
};

#endif
