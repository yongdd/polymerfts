/**
 * @file MakhoulTransform.h
 * @brief DCT-II/III and DST-II/III of a whole grid through one periodic real FFT.
 *
 * Used by FftwFFT and MklFFT (reflecting and absorbing boundaries) and by
 * FftwCrysFFTPmmm and MklCrysFFTPmmm.
 *
 * Makhoul (1980) turns a DCT-II of length n into a real FFT of length n: store
 * the even samples forward and the odd samples backward, transform to V, and
 * rotate with W^k = exp(-i pi k / (2n)):
 *
 *     X_k = 2 Re(W^k V_k).
 *
 * Reordering every axis gives one real FFT of the whole grid, the transform
 * both libraries run fastest (docs/internals/FFTImplementation.md, Section 8.2).
 * On a complex line, which is what the real FFT of the last axis leaves on the
 * other axes, the cosine transform is W^k V_k + W^-k V_{-k}, so the modes k and
 * n-k of axes 0 and 1 combine in pairs,
 *
 *     Y_k = W^k V_k + W^-k V_{-k},    Y_{n-k} = i (W^k V_k - W^-k V_{-k}),
 *
 * where mode 0 gains 2 and mode n/2 gains sqrt(2). The last axis, which holds
 * only the half spectrum, then gives real coefficients as in one dimension.
 * DCT-III runs these steps backward.
 *
 * A DST is a DCT of the samples with alternating signs and has its coefficients
 * in reverse order, RODFT10[x]_k = REDFT10[(-1)^j x_j]_{n-1-k}, so sine axes
 * only change the two tables that place the samples and the coefficients.
 *
 * The transforms are real-linear: a complex field transforms its real and
 * imaginary parts in turn, read from and written to every second double.
 */

#ifndef MAKHOUL_TRANSFORM_H_
#define MAKHOUL_TRANSFORM_H_

#include <array>
#include <complex>
#include <memory>
#include <utility>
#include <vector>

/**
 * @class MakhoulTransform
 * @brief Multi-dimensional cosine and sine transforms in FFTW's conventions.
 *
 * Execution is thread-safe; the scratch arrays are thread-local.
 */
class MakhoulTransform
{
public:
    /**
     * @brief The periodic real FFT of the reordered grid, run on the scratch of MakhoulTransform.
     *
     * Both directions are unnormalized and may overwrite their input, which the
     * transform no longer needs: FFTW's complex-to-real transform does. The
     * arrays start on a 64-byte boundary.
     */
    class RealFFT
    {
    public:
        virtual ~RealFFT() = default;
        /// modes = FFT of samples (nx[0] x ... x (nx[last]/2+1) modes)
        virtual void forward(double* samples, std::complex<double>* modes) const = 0;
        /// samples = unnormalized inverse FFT of modes
        virtual void backward(std::complex<double>* modes, double* samples) const = 0;
    };

private:
    int rank_;                 ///< Number of axes
    std::array<int, 3> nx_;    ///< Grid dimensions; the leading axes of a lower rank have length 1
    int half_;                 ///< Length of the half spectrum of the last axis
    int total_grid_;           ///< Total number of samples
    int total_modes_;          ///< Total number of complex modes of the real FFT
    std::unique_ptr<RealFFT> fft_;                        ///< Periodic real FFT of the reordered grid

    std::array<std::vector<int>, 3> sample_;              ///< Sample stored at each place of the reordered axis
    std::array<std::vector<double>, 3> sign_;             ///< Sign of that sample: -1 for odd samples of a sine axis
    std::array<std::vector<int>, 3> coefficient_;         ///< Where the coefficient of mode k of an axis is stored
    std::array<std::vector<std::complex<double>>, 3> w_;  ///< W^k for k = 0..n/2

    /// Thread-local scratch: the reordered samples and their modes, each from a 64-byte boundary.
    std::pair<double*, std::complex<double>*> thread_buffers() const;

    /// The transforms of one real field whose values are STRIDE doubles apart.
    template <int STRIDE> void forward_part(const double* in, double* out, double scale) const;
    template <int STRIDE> void backward_part(const double* in, double* out, double scale) const;

public:
    /**
     * @brief Set up the transform of one grid.
     *
     * @param nx   Grid dimensions (one to three axes)
     * @param sine For each axis, true for a sine transform, false for a cosine transform
     * @param fft  Periodic real FFT of the grid nx
     */
    MakhoulTransform(const std::vector<int>& nx, const std::vector<bool>& sine, std::unique_ptr<RealFFT> fft);

    /// out = scale * FFTW's unnormalized REDFT10/RODFT10 of in (in may be out).
    void forward(const double* in, double* out, double scale) const;

    /// The forward transform of the real and of the imaginary part of in, stored as (re, im) pairs (in may be out).
    void forward(const std::complex<double>* in, double* out, double scale) const;

    /// out = scale * FFTW's unnormalized REDFT01/RODFT01 of in (in may be out).
    void backward(const double* in, double* out, double scale) const;

    /// The backward transform of the (re, im) pairs of in into the real and imaginary parts of out (in may be out).
    void backward(const double* in, std::complex<double>* out, double scale) const;
};

#endif  // MAKHOUL_TRANSFORM_H_
