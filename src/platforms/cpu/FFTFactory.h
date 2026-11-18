/**
 * @file FFTFactory.h
 * @brief Factory functions for creating FFT objects.
 *
 * createFFT() returns the periodic FFT of the FFTW or MKL backend (FftwFFT or
 * MklFFT) for a periodic box, and MakhoulFFT, the cosine and sine transforms
 * through MakhoulTransform over the real FFT of that backend, for a box of
 * reflecting and absorbing axes.
 *
 * **Usage:**
 *
 * @code
 * std::unique_ptr<FFT<double>> fft = createFFT<double, 3>(nx, bc, FFTBackend::FFTW);
 * auto fft_mkl = createFFT<double, 3>(nx, bc, FFTBackend::MKL);
 * @endcode
 *
 * @see FFT for the abstract interface
 * @see FftwFFT for FFTW implementation
 * @see MklFFT for MKL implementation
 */

#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <vector>
#include "FFT.h"
#include "ComputationBox.h"
#include "MakhoulTransform.h"

#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

/**
 * @brief Create the unnormalized real FFT of a grid that runs under MakhoulTransform.
 *
 * @param nx      Grid dimensions (one to three axes)
 * @param backend FFT backend to use (FFTW or MKL)
 *
 * @return Owning pointer to FftwRealFFT or MklRealFFT
 *
 * @throws Exception if the requested backend is not available
 */
inline std::unique_ptr<MakhoulTransform::RealFFT> createRealFFT(const std::vector<int>& nx, FFTBackend backend)
{
#ifdef USE_CPU_MKL
    if (backend == FFTBackend::MKL)
        return std::make_unique<MklRealFFT>(nx);
#endif

#ifdef USE_CPU_FFTW
    if (backend == FFTBackend::FFTW)
        return std::make_unique<FftwRealFFT>(nx);
#endif

    (void)nx;
    (void)backend;
    throw_with_line_number("Requested FFT backend is not available. "
        "Compile with POLYMERFTS_USE_FFTW=ON or POLYMERFTS_USE_MKL=ON.");
}

/**
 * @class MakhoulFFT
 * @brief DCT-II/III and DST-II/III of a box of reflecting and absorbing axes in the FFT<T>
 *        interface: MakhoulTransform over the real FFT of either backend.
 *
 * The coefficients are real (the double* interface): forward() gives FFTW's REDFT10 or
 * RODFT10 / 2 per axis, the cosine or sine series, and backward() their inverse with the
 * 1/N normalization. A complex field (CL-FTS) has the coefficients of its real and imaginary
 * parts as (re, im) pairs, the periodic complex layout used by the solvers. The transforms
 * may run in place and concurrently (MakhoulTransform).
 */
template <typename T>
class MakhoulFFT : public FFT<T>
{
private:
    MakhoulTransform makhoul_;
    double forward_scale_ = 1.0;   ///< 1/2 per axis: REDFT10 and RODFT10 are twice the series
    double backward_scale_;        ///< 1/N

public:
    /// @param nx Grid dimensions; @param bc REFLECTING or ABSORBING per axis; @param backend FFTW or MKL
    MakhoulFFT(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, FFTBackend backend)
        : makhoul_(nx, bc, createRealFFT(nx, backend))
    {
        int total_grid = 1;
        for (int n : nx)
        {
            total_grid *= n;
            forward_scale_ *= 0.5;
        }
        backward_scale_ = 1.0 / static_cast<double>(total_grid);
        // One coefficient per grid point: real, or a (re, im) pair of a complex field
        this->set_sizes(total_grid, total_grid, std::is_same_v<T, std::complex<double>>);
    }

    void forward(T* rdata, double* cdata) override { makhoul_.forward(rdata, cdata, forward_scale_); }
    void backward(double* cdata, T* rdata) override { makhoul_.backward(cdata, rdata, backward_scale_); }

    void forward(T*, std::complex<double>*) override
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                               "Use forward(T*, double*) for non-periodic BC.");
    }
    void backward(std::complex<double>*, T*) override
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                               "Use backward(double*, T*) for non-periodic BC.");
    }
};

/**
 * @brief Create an FFT object with the specified backend.
 *
 * A periodic box gets the FFT of the backend (FftwFFT or MklFFT), a box of
 * reflecting and absorbing axes MakhoulFFT over the real FFT of the backend.
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param bc      Boundary conditions per dimension: all periodic or none
 * @param backend FFT backend to use (FFTW or MKL)
 *
 * @return Owning pointer to the new FFT object
 *
 * @throws Exception if periodic and non-periodic axes are mixed or the requested backend is not available
 */
template <typename T, int DIM>
std::unique_ptr<FFT<T>> createFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc, FFTBackend backend)
{
    const auto periodic = std::count(bc.begin(), bc.end(), BoundaryCondition::PERIODIC);
    if (periodic == 0)
        return std::make_unique<MakhoulFFT<T>>(std::vector<int>(nx.begin(), nx.end()),
                                               std::vector<BoundaryCondition>(bc.begin(), bc.end()), backend);
    if (periodic != DIM)
        throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                               "Use all periodic or all non-periodic (reflecting/absorbing).");

#ifdef USE_CPU_MKL
    if (backend == FFTBackend::MKL)
        return std::make_unique<MklFFT<T, DIM>>(nx);
#endif

#ifdef USE_CPU_FFTW
    if (backend == FFTBackend::FFTW)
        return std::make_unique<FftwFFT<T, DIM>>(nx);
#endif

    (void)backend;
    throw_with_line_number("Requested FFT backend is not available. "
        "Compile with POLYMERFTS_USE_FFTW=ON or POLYMERFTS_USE_MKL=ON.");
}

/**
 * @brief Create an FFT object with default (periodic) boundary conditions.
 *
 * Convenience overload that creates an FFT with all periodic boundaries.
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param backend FFT backend to use (FFTW or MKL)
 *
 * @return Owning pointer to the new FFT object
 */
template <typename T, int DIM>
std::unique_ptr<FFT<T>> createFFT(std::array<int, DIM> nx, FFTBackend backend)
{
    std::array<BoundaryCondition, DIM> bc;
    bc.fill(BoundaryCondition::PERIODIC);
    return createFFT<T, DIM>(nx, bc, backend);
}

#endif  // FFT_FACTORY_H_
