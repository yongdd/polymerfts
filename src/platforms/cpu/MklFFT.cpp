/**
 * @file MklFFT.cpp
 * @brief Intel MKL FFT implementation with O(N log N) DCT/DST via FFT.
 *
 * Provides Fast Fourier Transform functionality using Intel MKL DFTI for
 * all boundary conditions with O(N log N) complexity.
 *
 * **Transform Types by Boundary Condition:**
 *
 * - PERIODIC: MKL DFTI r2c/c2r transforms
 * - REFLECTING: DCT-II/III through one real FFT of the grid (MakhoulTransform)
 * - ABSORBING: DST-II/III through one real FFT of the grid (MakhoulTransform)
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 *
 * @see MklFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "MklFFT.h"
#include "MklDftiLayout.h"
#include "MklStatusCheck.h"

//------------------------------------------------------------------------------
// Constructor (periodic BC, backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true)
{
    // Initialize all BCs to periodic
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    // Compute grid sizes
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // Compute complex grid size for r2c FFT
    if constexpr (std::is_same<T, double>::value)
    {
        if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else
            total_complex_grid_ = nx_[0] / 2 + 1;
    }
    else
    {
        total_complex_grid_ = total_grid_;
    }

    try
    {
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
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc)
{
    try
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

        // For simplicity, require either all periodic or all non-periodic
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

        // Compute complex grid size
        if (is_all_periodic_)
        {
            if constexpr (std::is_same<T, double>::value)
            {
                if (DIM == 3)
                    total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
                else if (DIM == 2)
                    total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
                else
                    total_complex_grid_ = nx_[0] / 2 + 1;
            }
            else
            {
                total_complex_grid_ = total_grid_;
            }
            initPeriodicFFT();
        }
        else
        {
            // DCT/DST: real-to-real, same size
            total_complex_grid_ = total_grid_;
            initNonPeriodicFFT();
        }
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::~MklFFT()
{
    release();
}

template <typename T, int DIM>
void MklFFT<T, DIM>::release()
{
    if (hand_forward_ != nullptr)
        DftiFreeDescriptor(&hand_forward_);
    if (hand_backward_ != nullptr)
        DftiFreeDescriptor(&hand_backward_);
    hand_forward_ = nullptr;
    hand_backward_ = nullptr;
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (MKL DFTI)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::initPeriodicFFT()
{
    try
    {
        MKL_LONG NX[DIM];
        for (int i = 0; i < DIM; i++)
            NX[i] = nx_[i];

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[DIM + 1] = {0};
        MKL_LONG cs[DIM + 1] = {0};
        rs[DIM] = 1;
        cs[DIM] = 1;

        // Determine if we're using real or complex data type
        constexpr bool is_real = std::is_same<T, double>::value;
        constexpr DFTI_CONFIG_VALUE dtype = is_real ? DFTI_REAL : DFTI_COMPLEX;

        // Calculate strides for real domain
        for (int i = DIM - 1; i > 0; i--)
            rs[i] = rs[i + 1] * nx_[i];

        // Calculate strides for complex domain
        if constexpr (std::is_same<T, double>::value)
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * (i == DIM - 1 ? nx_[i] / 2 + 1 : nx_[i]);
        }
        else
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * nx_[i];
        }

        if (DIM == 1)
        {
            mkl_dfti_check(DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, 1, NX[0]), "DftiCreateDescriptor (forward)");
            mkl_dfti_check(DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, 1, NX[0]), "DftiCreateDescriptor (backward)");
        }
        else
        {
            mkl_dfti_check(DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, DIM, NX), "DftiCreateDescriptor (forward)");
            mkl_dfti_check(DftiSetValue(hand_forward_, mkl_dfti::forward_input_strides, rs), "DftiSetValue (forward input strides)");
            mkl_dfti_check(DftiSetValue(hand_forward_, mkl_dfti::forward_output_strides, cs), "DftiSetValue (forward output strides)");

            mkl_dfti_check(DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, DIM, NX), "DftiCreateDescriptor (backward)");
            mkl_dfti_check(DftiSetValue(hand_backward_, mkl_dfti::backward_input_strides, cs), "DftiSetValue (backward input strides)");
            mkl_dfti_check(DftiSetValue(hand_backward_, mkl_dfti::backward_output_strides, rs), "DftiSetValue (backward output strides)");
        }

        mkl_dfti_check(DftiSetValue(hand_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, forward)");
        mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, backward)");

        if (is_real)
        {
            mkl_dfti_check(DftiSetValue(hand_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, forward)");
            mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, backward)");
        }

        mkl_dfti_check(DftiCommitDescriptor(hand_forward_), "DftiCommitDescriptor (forward)");
        mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(total_grid_)), "DftiSetValue(DFTI_BACKWARD_SCALE, backward)");
        mkl_dfti_check(DftiCommitDescriptor(hand_backward_), "DftiCommitDescriptor (backward)");
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
void MklFFT<T, DIM>::initNonPeriodicFFT()
{
    std::vector<bool> sine(DIM);
    for (int d = 0; d < DIM; ++d)
        sine[d] = bc_[d] == BoundaryCondition::ABSORBING;
    makhoul_ = std::make_unique<MakhoulTransform>(std::vector<int>(nx_.begin(), nx_.end()), sine,
                                                  std::make_unique<MklFFT<double, DIM>>(nx_));
}

//------------------------------------------------------------------------------
// Forward (complex output interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                              "Use forward(T*, double*) for non-periodic BC.");
    }

    mkl_dfti_check(DftiComputeForward(hand_forward_, rdata, cdata), "DftiComputeForward (periodic)");
}

//------------------------------------------------------------------------------
// Backward (complex input interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                              "Use backward(double*, T*) for non-periodic BC.");
    }

    mkl_dfti_check(DftiComputeBackward(hand_backward_, cdata, rdata), "DftiComputeBackward (periodic)");
}

//------------------------------------------------------------------------------
// Forward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            mkl_dfti_check(DftiComputeForward(hand_forward_, rdata, cdata_complex), "DftiComputeForward (periodic)");
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
                // The DCT/DST is a real-linear operator: for complex fields
                // (CL-FTS) transform the real and imaginary parts independently
                // and store the coefficients interleaved as complex pairs
                // (cdata must then hold 2*total_complex_grid_ doubles, matching
                // the periodic complex layout used by the solvers).
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
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            mkl_dfti_check(DftiComputeBackward(hand_backward_, cdata_complex, rdata), "DftiComputeBackward (periodic)");
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
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
#include "TemplateInstantiations.h"
INSTANTIATE_FFT_CLASS(MklFFT);
