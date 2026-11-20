/**
 * @file FFT.h
 * @brief Abstract interface for spectral transforms (FFT, DCT, DST).
 *
 * This header defines the FFT abstract base class that supports:
 *
 * - PERIODIC (FFT): Interleaved real/imaginary complex coefficients
 * - REFLECTING (DCT): Real coefficients
 * - ABSORBING (DST): Real coefficients
 *
 * **Interfaces:**
 *
 * - `forward(T*, double*)` / `backward(double*, T*)`: all BCs; the complex coefficients of a
 *   periodic axis are stored as interleaved doubles
 * - `backward_with_factors(double*, k_factor, T*, r_factor)`: the backward transform with a factor
 *   on the coefficients and one on the grid, as a propagator step takes it
 *
 * **Design Rationale:**
 *
 * This allows storing transform objects of different dimensions (1D, 2D, 3D)
 * in a single FFT<T>* pointer without void* casts and if-else dispatch.
 * Since all simulations run in fixed dimensions, a single pointer suffices.
 *
 * @see FftwFFT for FFTW implementation
 * @see CudaFFT for CUDA implementation
 */

#ifndef FFT_H_
#define FFT_H_

#include <complex>
#include "Exception.h"

/**
 * @enum FFTBackend
 * @brief Available FFT backend implementations.
 *
 * Two backends are available for CPU:
 * - FFTW: FFTW3 library with O(N log N) for all transform types (GPL license)
 * - MKL: Intel MKL with O(N log N) for FFT and FFT-based DCT/DST (proprietary)
 */
enum class FFTBackend
{
    FFTW,  ///< FFTW3 library (O(N log N) for all transform types)
    MKL    ///< Intel MKL (O(N log N) for all transform types)
};

/**
 * @class FFT
 * @brief Abstract base class for spectral transforms.
 *
 * Provides a dimension-independent interface for forward/backward transforms
 * including FFT (periodic), DCT (reflecting), and DST (absorbing).
 *
 * @tparam T Input/output type (double or std::complex<double>)
 *
 * **Grid Size Convention:**
 *
 * For a real array of size Nx × Ny × Nz, the complex array has size
 * Nx × Ny × (Nz/2 + 1) due to Hermitian symmetry.
 *
 * **FFT Convention:**
 *
 * - Forward: Real-to-complex transform (r2c)
 * - Backward: Complex-to-real transform (c2r) with normalization
 *
 * The backward transform includes the 1/N normalization factor so that
 * backward(forward(x)) = x (up to numerical precision).
 *
 * @example
 * @code
 * FFT<double>* fft = ...;
 * double* real_data = ...;
 * std::complex<double>* complex_data = ...;   // periodic: Nx × Ny × (Nz/2 + 1) values
 * fft->forward(real_data, reinterpret_cast<double*>(complex_data));
 * fft->backward(reinterpret_cast<double*>(complex_data), real_data);
 * @endcode
 */
template <typename T>
class FFT
{
public:
    /// Virtual destructor.
    virtual ~FFT() {}

    /**
     * @brief Perform forward transform (real coefficient interface).
     *
     * The CPU implementations take all boundary conditions; CudaFFT, which the
     * CUDA solvers build for walls only, refuses periodic boxes.
     * - PERIODIC: real-to-complex FFT (output is interleaved real/imag)
     * - REFLECTING: DCT-II
     * - ABSORBING: DST-II
     *
     * @param rdata Input: real-space data
     * @param cdata Output: spectral coefficients (double* for all BCs)
     */
    virtual void forward(T* rdata, double* cdata) = 0;

    /**
     * @brief Perform backward transform (real coefficient interface).
     *
     * Boundary conditions as for forward():
     * - PERIODIC: complex-to-real IFFT with 1/N normalization
     * - REFLECTING: DCT-III with normalization
     * - ABSORBING: DST-III with normalization
     *
     * @param cdata Input: spectral coefficients
     * @param rdata Output: real-space data
     */
    virtual void backward(double* cdata, T* rdata) = 0;

    /**
     * @brief Backward transform of the coefficients times k_factor, times r_factor:
     *        rdata = r_factor · backward(k_factor · cdata).
     *
     * k_factor holds one value per coefficient of backward(double*, T*) (a complex
     * coefficient, or a real one of the cosine and sine series), r_factor one per
     * grid point; either may be nullptr. The products are those of separate loops
     * around backward(), so the result is the same bit for bit. cdata may be
     * overwritten. The CPU transforms implement it (FftwFFT, MklFFT, MakhoulFFT);
     * the default refuses, for CudaFFT, whose solvers run their own kernels. It has
     * a name of its own: an overload of backward() would be hidden in every
     * transform that overrides backward().
     */
    virtual void backward_with_factors(double* /*cdata*/, const double* /*k_factor*/, T* /*rdata*/,
                                       const T* /*r_factor*/)
    {
        throw_with_line_number("This transform has no backward_with_factors().");
    }
};

#endif
