/**
 * @file FftwFFT.h
 * @brief FFTW3 implementation of spectral transforms with mixed BC support.
 *
 * This header provides FftwFFT, which implements spectral transforms using
 * FFTW3 and supports all boundary conditions (FFT/DCT/DST).
 *
 * **Supported Boundary Conditions:**
 *
 * - PERIODIC: Standard FFT (real-to-complex) using FFTW r2c/c2r
 * - REFLECTING: DCT-II forward, DCT-III backward (Neumann BC)
 * - ABSORBING: DST-II forward, DST-III backward (Dirichlet BC)
 *
 * **Memory Layout:**
 *
 * - Periodic BC: Complex array has size Nx × Ny × (Nz/2+1) for Hermitian symmetry
 * - Non-periodic BC: "Complex" array has same size as real array (all real coefficients)
 *
 * @see FFT for the abstract interface
 * @see MakhoulTransform for the DCT and DST
 * @see FftwFFT for FFTW implementation
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
#include "ComputationBox.h"
#include "MakhoulTransform.h"

/**
 * @class FftwFFT
 * @brief FFTW3-based FFT with support for all boundary conditions.
 *
 * Uses FFTW3 for periodic FFT and MakhoulTransform over that FFT for non-periodic BC.
 *
 * @tparam T   Input type for forward transform (typically double)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 *
 * **Thread Safety:**
 *
 * FFTW plan execution is thread-safe, but plan creation/destruction is not.
 * All plans are created in the constructor and destroyed in the destructor.
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 */
template <typename T, int DIM>
class FftwFFT : public FFT<T>
{
private:
    std::array<int, DIM> nx_;                      ///< Grid dimensions
    std::array<BoundaryCondition, DIM> bc_;        ///< Boundary conditions per dimension
    int total_grid_;                               ///< Total real grid size
    int total_complex_grid_;                       ///< Total "complex" grid size
    bool is_all_periodic_;                         ///< True if all BCs are periodic

    // FFTW plans for periodic FFT
    fftw_plan plan_forward_ = nullptr;             ///< Forward (r2c) transform
    fftw_plan plan_backward_ = nullptr;            ///< Backward (c2r) transform

    /// DCT/DST of the grid through one real FFT (non-periodic BC only)
    std::unique_ptr<MakhoulTransform> makhoul_;

    // Buffers for FFTW (FFTW requires aligned memory)
    double* work_buffer_ = nullptr;                ///< Work buffer for transforms
    fftw_complex* complex_buffer_ = nullptr;       ///< Complex buffer for r2c/c2r

    /// Destroy the plans and free the buffers (also after a failed construction); idempotent.
    void release();

    /// Initialize FFTW plans for periodic FFT.
    void initPeriodicFFT();

    /// Set up the DCT/DST of the grid for non-periodic BC.
    void initNonPeriodicFFT();

    /// Compute complex grid size based on dimensions and input type.
    void computeComplexGridSize();

public:
    /**
     * @brief Construct FFTW FFT object for given grid dimensions (periodic BC).
     *
     * For backward compatibility, creates an FFT with all periodic boundaries.
     *
     * @param nx Grid dimensions [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D
     */
    FftwFFT(std::array<int, DIM> nx);

    /**
     * @brief Construct FFTW FFT object with specified boundary conditions.
     *
     * @param nx Grid dimensions
     * @param bc Boundary conditions per dimension
     */
    FftwFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc);

    /// Destructor. Frees FFTW plans and buffers.
    ~FftwFFT();

    /**
     * @brief Check if all boundary conditions are periodic.
     * @return true if all periodic, false otherwise
     */
    bool is_all_periodic() const { return is_all_periodic_; }

    /**
     * @brief Get total "complex" grid size.
     *
     * For periodic: Nx × Ny × (Nz/2+1)
     * For non-periodic: same as real grid size
     *
     * @return Total complex grid size
     */
    int get_total_complex_grid() const { return total_complex_grid_; }

    /**
     * @brief Get total real grid size.
     * @return Total grid size
     */
    int get_total_grid() const { return total_grid_; }

    /**
     * @brief Get boundary condition for a dimension.
     * @param dim Dimension index
     * @return Boundary condition
     */
    BoundaryCondition get_bc(int dim) const { return bc_[dim]; }

    /**
     * @brief Perform forward FFT (complex output interface).
     *
     * Implements FFT<T> interface for backward compatibility.
     *
     * @param rdata Input: real array
     * @param cdata Output: complex array
     *
     * @throws Exception if any dimension is non-periodic
     */
    void forward(T *rdata, std::complex<double> *cdata) override;

    /**
     * @brief Perform backward FFT (complex input interface).
     *
     * Implements FFT<T> interface for backward compatibility.
     *
     * @param cdata Input: complex array
     * @param rdata Output: real array
     *
     * @throws Exception if any dimension is non-periodic
     */
    void backward(std::complex<double> *cdata, T *rdata) override;

    /**
     * @brief Perform forward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: real-to-complex FFT (output is interleaved real/imag)
     * - REFLECTING: DCT-II
     * - ABSORBING: DST-II
     *
     * @param rdata Input: real array of size total_grid
     * @param cdata Output: coefficients (size total_complex_grid)
     */
    void forward(T *rdata, double *cdata) override;

    /**
     * @brief Perform backward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: complex-to-real IFFT with 1/N normalization
     * - REFLECTING: DCT-III with normalization
     * - ABSORBING: DST-III with normalization
     *
     * @param cdata Input: Fourier coefficients
     * @param rdata Output: real array
     */
    void backward(double *cdata, T *rdata) override;
};

#endif
