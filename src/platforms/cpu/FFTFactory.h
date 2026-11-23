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
 * std::unique_ptr<FFT<double>> fft = createFFT<double>(nx, bc, FFTBackend::FFTW);
 * auto fft_mkl = createFFT<double>(nx, bc, FFTBackend::MKL);
 * @endcode
 *
 * @see FFT for the abstract interface
 * @see FftwFFT for FFTW implementation
 * @see MklFFT for MKL implementation
 */

#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <algorithm>
#include <memory>
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
 * may run in place and concurrently (MakhoulTransform), and backward_with_factors() applies
 * its factors inside the loops of MakhoulTransform.
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
    }

    void forward(T* rdata, double* cdata) override { makhoul_.forward(rdata, cdata, forward_scale_); }
    void backward(double* cdata, T* rdata) override { makhoul_.backward(cdata, rdata, backward_scale_); }

    /// One k_factor per coefficient (a (re, im) pair of a complex field); cdata is kept
    void backward_with_factors(double* cdata, const double* k_factor, T* rdata, const T* r_factor) override
    {
        makhoul_.backward_with_factors(cdata, k_factor, rdata, r_factor, backward_scale_);
    }
};

/**
 * @brief Create an FFT object with the specified backend.
 *
 * A periodic box gets the FFT of the backend (FftwFFT or MklFFT), a box of
 * reflecting and absorbing axes MakhoulFFT over the real FFT of the backend.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * @param nx      Grid dimensions (one to three axes)
 * @param bc      Boundary condition of each axis: all periodic or none
 * @param backend FFT backend to use (FFTW or MKL)
 *
 * @return Owning pointer to the new FFT object
 *
 * @throws Exception if periodic and non-periodic axes are mixed or the requested backend is not available
 */
template <typename T>
std::unique_ptr<FFT<T>> createFFT(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                                  FFTBackend backend)
{
    if (bc.size() != nx.size())
        throw_with_line_number("createFFT needs one boundary condition per axis.");
    const size_t periodic = static_cast<size_t>(std::count(bc.begin(), bc.end(), BoundaryCondition::PERIODIC));
    if (periodic == 0)
        return std::make_unique<MakhoulFFT<T>>(nx, bc, backend);
    if (periodic != nx.size())
        throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                               "Use all periodic or all non-periodic (reflecting/absorbing).");

#ifdef USE_CPU_MKL
    if (backend == FFTBackend::MKL)
        return std::make_unique<MklFFT<T>>(nx);
#endif

#ifdef USE_CPU_FFTW
    if (backend == FFTBackend::FFTW)
        return std::make_unique<FftwFFT<T>>(nx);
#endif

    (void)backend;
    throw_with_line_number("Requested FFT backend is not available. "
        "Compile with POLYMERFTS_USE_FFTW=ON or POLYMERFTS_USE_MKL=ON.");
}

#endif  // FFT_FACTORY_H_
