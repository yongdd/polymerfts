/**
 * @file CpuFourierTransform.h
 * @brief FourierTransform with the CPU transforms (FFTW or MKL).
 *
 * The fields of a call are transformed concurrently (OpenMP over fields, the
 * transforms themselves single-threaded as in the propagator solvers) and the
 * multipliers are applied with OpenMP over wavevectors. One thread uses a
 * transform object at a time: backward() and apply() work in its member arrays.
 *
 * A box that mixes periodic and wall axes runs MakhoulTransform with periodic
 * axes: one real FFT of the grid, the DFT along the periodic axes and the
 * cosine or sine series along the walls.
 *
 * @see FourierTransform for the interface and layouts
 */

#ifndef CPU_FOURIER_TRANSFORM_H_
#define CPU_FOURIER_TRANSFORM_H_

#include <complex>
#include <memory>
#include <vector>

#include "AlignedScratch.h"
#include "FFT.h"
#include "FourierTransform.h"
#include "MakhoulTransform.h"

/// Multiplier values in host memory
template <typename T>
class CpuFourierMultiplier : public FourierMultiplier<T>
{
public:
    CpuFourierMultiplier(const FourierTransform<T>* owner, int n_out, int n_in, bool diagonal)
        : FourierMultiplier<T>(owner, n_out, n_in, diagonal) {}

    std::vector<std::complex<double>> complex_values;   ///< complex coefficients
    std::vector<double> real_values;                    ///< real coefficients
};

template <typename T>
class CpuFourierTransform : public FourierTransform<T>
{
public:
    /**
     * @param cb      Computation box; must outlive the transform
     * @param bc      Transform of each axis (see FourierTransform)
     * @param backend FFTW or MKL
     */
    CpuFourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc, FFTBackend backend);

protected:
    std::unique_ptr<FourierMultiplier<T>> store_multiplier(
        const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal) override;
    void apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out) override;
    void forward_impl(const T* fields, int n_fields, double* coefficients) override;
    void backward_impl(const double* coefficients, int n_fields, T* fields) override;
    /// The coefficients copied into coefficients_out_, made Hermitian there and transformed
    /// from that copy, which the transform may overwrite (FFTW's complex-to-real transform runs on it)
    void backward_hermitian_impl(const double* coefficients, int n_fields, T* fields) override;

private:
    std::unique_ptr<FFT<T>> fft_;                     ///< periodic or wall boxes
    std::unique_ptr<MakhoulTransform> mixed_fft_;     ///< boxes that mix periodic and wall axes
    // From 64-byte boundaries, as the scratch of the transforms, which then run on them
    AlignedScratch coefficients_in_;    ///< coefficients of the inputs of a term
    AlignedScratch coefficients_out_;   ///< accumulated output coefficients; the copy of backward()

    /// The inverse transforms of n_fields fields, OpenMP over fields; with give_up the
    /// transforms may overwrite the coefficients (FFT<T>::backward_with_factors() without factors)
    void backward_fields(double* coefficients, int n_fields, T* fields, bool give_up);
};

#endif
