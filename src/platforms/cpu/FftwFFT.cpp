/**
 * @file FftwFFT.cpp
 * @brief FFTW3 implementation of spectral transforms with mixed BC support.
 *
 * Provides Fast Fourier Transform functionality using FFTW3 for both
 * periodic and non-periodic (reflecting/absorbing) boundary conditions.
 *
 * **Transform Types by Boundary Condition:**
 *
 * - PERIODIC: FFTW r2c/c2r transforms
 * - REFLECTING: DCT-II (forward), DCT-III (backward) through one real FFT of the grid
 * - ABSORBING: DST-II (forward), DST-III (backward) through one real FFT of the grid
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 *
 * @see FftwFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include "FftwFFT.h"

//------------------------------------------------------------------------------
// Compute complex grid size based on periodicity and input type
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::computeComplexGridSize()
{
    if (is_all_periodic_)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            // r2c FFT: exploit Hermitian symmetry
            if constexpr (DIM == 3)
                total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
            else if constexpr (DIM == 2)
                total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
            else
                total_complex_grid_ = nx_[0] / 2 + 1;
        }
        else
        {
            // c2c FFT: full complex grid
            total_complex_grid_ = total_grid_;
        }
    }
    else
    {
        // DCT/DST: real-to-real, same size
        total_complex_grid_ = total_grid_;
    }
}

//------------------------------------------------------------------------------
// Constructor (periodic BC, backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::FftwFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true)
{
    // Initialize all BCs to periodic
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    // Compute grid sizes
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    try
    {
        computeComplexGridSize();
        initPeriodicFFT();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Constructor (with BC specification)
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::FftwFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc)
{
    // Compute total grid size
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // Check if all periodic
    is_all_periodic_ = true;
    for (int d = 0; d < DIM; ++d)
    {
        if (bc_[d] != BoundaryCondition::PERIODIC)
        {
            is_all_periodic_ = false;
            break;
        }
    }

    // Require either all periodic or all non-periodic (no mixed)
    if (!is_all_periodic_)
    {
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] == BoundaryCondition::PERIODIC)
            {
                throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                      "Use all periodic or all non-periodic (reflecting/absorbing).");
            }
        }
    }

    computeComplexGridSize();

    try
    {
        if (is_all_periodic_)
            initPeriodicFFT();
        else
            initNonPeriodicFFT();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::~FftwFFT()
{
    release();
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::release()
{
    if (plan_forward_ != nullptr)
        fftw_destroy_plan(plan_forward_);
    if (plan_backward_ != nullptr)
        fftw_destroy_plan(plan_backward_);
    plan_forward_ = nullptr;
    plan_backward_ = nullptr;

    if (work_buffer_ != nullptr)
        fftw_free(work_buffer_);
    if (complex_buffer_ != nullptr)
        fftw_free(complex_buffer_);
    work_buffer_ = nullptr;
    complex_buffer_ = nullptr;
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (FFTW r2c/c2r)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::initPeriodicFFT()
{
    try
    {
        // Allocate aligned buffers
        // For complex input (T = std::complex<double>), work_buffer_ needs
        // to hold complex values for c2c transform, so allocate 2x the size
        if constexpr (std::is_same<T, double>::value)
            work_buffer_ = fftw_alloc_real(total_grid_);
        else
            work_buffer_ = fftw_alloc_real(total_grid_ * 2);  // For complex c2c FFT
        complex_buffer_ = fftw_alloc_complex(total_complex_grid_);

        if (work_buffer_ == nullptr || complex_buffer_ == nullptr)
            throw_with_line_number("FFTW memory allocation failed");

        // Create FFTW plans
        // FFTW_MEASURE: runs actual transforms to find optimal algorithm
        // This takes longer to create plans but results in faster execution
        unsigned int flags = FFTW_MEASURE;

        if constexpr (std::is_same<T, double>::value)
        {
            if (DIM == 1)
            {
                plan_forward_ = fftw_plan_dft_r2c_1d(nx_[0], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_1d(nx_[0], complex_buffer_, work_buffer_, flags);
            }
            else if (DIM == 2)
            {
                plan_forward_ = fftw_plan_dft_r2c_2d(nx_[0], nx_[1], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_2d(nx_[0], nx_[1], complex_buffer_, work_buffer_, flags);
            }
            else // DIM == 3
            {
                plan_forward_ = fftw_plan_dft_r2c_3d(nx_[0], nx_[1], nx_[2], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_3d(nx_[0], nx_[1], nx_[2], complex_buffer_, work_buffer_, flags);
            }
        }
        else // Complex input (c2c transform)
        {
            int n[DIM];
            for (int d = 0; d < DIM; ++d)
                n[d] = nx_[d];

            plan_forward_ = fftw_plan_dft(DIM, n,
                reinterpret_cast<fftw_complex*>(work_buffer_),
                complex_buffer_, FFTW_FORWARD, flags);
            plan_backward_ = fftw_plan_dft(DIM, n,
                complex_buffer_,
                reinterpret_cast<fftw_complex*>(work_buffer_),
                FFTW_BACKWARD, flags);
        }

        if (plan_forward_ == nullptr || plan_backward_ == nullptr)
            throw_with_line_number("FFTW plan creation failed");
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Initialize non-periodic FFT (DCT/DST through one real FFT of the grid)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::initNonPeriodicFFT()
{
    std::vector<bool> sine(DIM);
    for (int d = 0; d < DIM; ++d)
        sine[d] = bc_[d] == BoundaryCondition::ABSORBING;
    makhoul_ = std::make_unique<MakhoulTransform>(std::vector<int>(nx_.begin(), nx_.end()), sine,
                                                  std::make_unique<FftwFFT<double, DIM>>(nx_));
}

//------------------------------------------------------------------------------
// Forward (complex output interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                              "Use forward(T*, double*) for non-periodic BC.");
    }

    // Thread-safe FFT using thread-local buffers
    thread_local std::vector<double> work_local;

    if constexpr (std::is_same<T, double>::value)
    {
        if (static_cast<int>(work_local.size()) < total_grid_)
            work_local.resize(total_grid_);

        std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(double));
        fftw_execute_dft_r2c(plan_forward_, work_local.data(),
                             reinterpret_cast<fftw_complex*>(cdata));
    }
    else
    {
        if (static_cast<int>(work_local.size()) < total_grid_ * 2)
            work_local.resize(total_grid_ * 2);

        std::copy_n(rdata, total_grid_, reinterpret_cast<std::complex<double>*>(work_local.data()));
        fftw_execute_dft(plan_forward_,
                         reinterpret_cast<fftw_complex*>(work_local.data()),
                         reinterpret_cast<fftw_complex*>(cdata));
    }
}

//------------------------------------------------------------------------------
// Backward (complex input interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                              "Use backward(double*, T*) for non-periodic BC.");
    }

    // Thread-safe IFFT using thread-local buffers
    // CRITICAL: fftw_execute_dft_c2r destroys its input
    thread_local std::vector<double> work_local;
    thread_local std::vector<std::complex<double>> complex_local;

    double scale = 1.0 / static_cast<double>(total_grid_);

    if constexpr (std::is_same<T, double>::value)
    {
        if (static_cast<int>(work_local.size()) < total_grid_)
            work_local.resize(total_grid_);
        if (static_cast<int>(complex_local.size()) < total_complex_grid_)
            complex_local.resize(total_complex_grid_);

        std::copy_n(cdata, total_complex_grid_, complex_local.data());
        fftw_execute_dft_c2r(plan_backward_,
                             reinterpret_cast<fftw_complex*>(complex_local.data()),
                             work_local.data());
        for (int i = 0; i < total_grid_; ++i)
            rdata[i] = work_local[i] * scale;
    }
    else
    {
        if (static_cast<int>(work_local.size()) < total_grid_ * 2)
            work_local.resize(total_grid_ * 2);
        if (static_cast<int>(complex_local.size()) < total_complex_grid_)
            complex_local.resize(total_complex_grid_);

        std::copy_n(cdata, total_complex_grid_, complex_local.data());
        fftw_execute_dft(plan_backward_,
                         reinterpret_cast<fftw_complex*>(complex_local.data()),
                         reinterpret_cast<fftw_complex*>(work_local.data()));
        std::complex<double>* work_complex = reinterpret_cast<std::complex<double>*>(work_local.data());
        for (int i = 0; i < total_grid_; ++i)
            rdata[i] = work_complex[i] * scale;
    }
}

//------------------------------------------------------------------------------
// Forward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    if (is_all_periodic_)
    {
        // Thread-safe FFT using thread-local aligned buffers
        // Each thread gets its own work and complex buffers to avoid race conditions
        // when multiple threads call FFT simultaneously
        thread_local std::vector<double> work_local;
        thread_local std::vector<std::complex<double>> complex_local;

        if constexpr (std::is_same<T, double>::value)
        {
            if (static_cast<int>(work_local.size()) < total_grid_)
                work_local.resize(total_grid_);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // r2c: copy to local buffer, execute with new-array interface, copy output
            std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(double));
            fftw_execute_dft_r2c(plan_forward_, work_local.data(),
                                 reinterpret_cast<fftw_complex*>(complex_local.data()));
            std::copy_n(complex_local.data(), total_complex_grid_, reinterpret_cast<std::complex<double>*>(cdata));
        }
        else
        {
            // c2c case - both work and complex are full grid size
            if (static_cast<int>(work_local.size()) < total_grid_ * 2)
                work_local.resize(total_grid_ * 2);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2c: copy to local buffer, execute with new-array interface, copy output
            std::copy_n(rdata, total_grid_, reinterpret_cast<std::complex<double>*>(work_local.data()));
            fftw_execute_dft(plan_forward_,
                             reinterpret_cast<fftw_complex*>(work_local.data()),
                             reinterpret_cast<fftw_complex*>(complex_local.data()));
            std::copy_n(complex_local.data(), total_complex_grid_, reinterpret_cast<std::complex<double>*>(cdata));
        }
    }
    else
    {
        // FFTW's DCT-II/DST-II are 2x the standard definition per dimension
        const double scale = 1.0 / (1 << DIM);

        if constexpr (std::is_same<T, double>::value)
        {
            makhoul_->forward(rdata, cdata, scale);
        }
        else
        {
            // Real-linear transforms: for complex fields (CL-FTS) transform
            // the real and imaginary parts independently and store the
            // coefficients interleaved as complex pairs (matching the
            // periodic complex layout used by the solvers).
            // Thread-local buffer for thread safety in OpenMP parallel regions
            thread_local std::vector<double> work;
            if (static_cast<int>(work.size()) < total_grid_)
                work.resize(total_grid_);

            for (int part = 0; part < 2; ++part)
            {
                for (int i = 0; i < total_grid_; ++i)
                    work[i] = (part == 0) ? std::real(rdata[i]) : std::imag(rdata[i]);
                makhoul_->forward(work.data(), work.data(), scale);
                for (int i = 0; i < total_complex_grid_; ++i)
                    cdata[2 * i + part] = work[i];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Backward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    if (is_all_periodic_)
    {
        // Thread-safe IFFT using thread-local aligned buffers
        // CRITICAL: fftw_execute_dft_c2r destroys its input, so we must copy first
        thread_local std::vector<double> work_local;
        thread_local std::vector<std::complex<double>> complex_local;

        double scale = 1.0 / static_cast<double>(total_grid_);

        if constexpr (std::is_same<T, double>::value)
        {
            if (static_cast<int>(work_local.size()) < total_grid_)
                work_local.resize(total_grid_);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2r: copy input to local buffer (preserving original), execute, copy with scaling
            std::copy_n(reinterpret_cast<const std::complex<double>*>(cdata), total_complex_grid_, complex_local.data());
            fftw_execute_dft_c2r(plan_backward_,
                                 reinterpret_cast<fftw_complex*>(complex_local.data()),
                                 work_local.data());
            for (int i = 0; i < total_grid_; ++i)
                rdata[i] = work_local[i] * scale;
        }
        else
        {
            // c2c case
            if (static_cast<int>(work_local.size()) < total_grid_ * 2)
                work_local.resize(total_grid_ * 2);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2c: copy input to local buffer, execute, copy with scaling
            std::copy_n(reinterpret_cast<const std::complex<double>*>(cdata), total_complex_grid_, complex_local.data());
            fftw_execute_dft(plan_backward_,
                             reinterpret_cast<fftw_complex*>(complex_local.data()),
                             reinterpret_cast<fftw_complex*>(work_local.data()));
            std::complex<double>* work_complex = reinterpret_cast<std::complex<double>*>(work_local.data());
            for (int i = 0; i < total_grid_; ++i)
                rdata[i] = work_complex[i] * scale;
        }
    }
    else
    {
        const double scale = 1.0 / static_cast<double>(total_grid_);

        if constexpr (std::is_same<T, double>::value)
        {
            makhoul_->backward(cdata, rdata, scale);
        }
        else
        {
            // Complex fields: coefficients arrive interleaved (see forward);
            // inverse-transform the real and imaginary parts independently.
            thread_local std::vector<double> work;
            if (static_cast<int>(work.size()) < total_grid_)
                work.resize(total_grid_);

            double* out = reinterpret_cast<double*>(rdata);
            for (int part = 0; part < 2; ++part)
            {
                for (int i = 0; i < total_complex_grid_; ++i)
                    work[i] = cdata[2 * i + part];
                makhoul_->backward(work.data(), work.data(), scale);
                for (int i = 0; i < total_grid_; ++i)
                    out[2 * i + part] = work[i];
            }
        }
    }
}

// Explicit template instantiations
template class FftwFFT<double, 1>;
template class FftwFFT<double, 2>;
template class FftwFFT<double, 3>;
template class FftwFFT<std::complex<double>, 1>;
template class FftwFFT<std::complex<double>, 2>;
template class FftwFFT<std::complex<double>, 3>;
