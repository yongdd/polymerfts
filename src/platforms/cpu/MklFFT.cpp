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

namespace
{
/**
 * @brief Out-of-place DFTI descriptors of a row-major grid nx (one to three axes).
 *
 * A real FFT stores the half spectrum of the last axis as complex values
 * (Nx x ... x (Nz/2+1)); a complex FFT stores the whole grid. The backward
 * transform is multiplied by backward_scale. The caller frees the handles,
 * also after a failure.
 */
void create_descriptors(const std::vector<int>& nx, bool real, double backward_scale,
                        DFTI_DESCRIPTOR_HANDLE& forward, DFTI_DESCRIPTOR_HANDLE& backward)
{
    const int rank = static_cast<int>(nx.size());
    std::vector<MKL_LONG> lengths(nx.begin(), nx.end());

    // Strides of the real and of the complex domain
    std::vector<MKL_LONG> rs(rank + 1, 0);
    std::vector<MKL_LONG> cs(rank + 1, 0);
    rs[rank] = 1;
    cs[rank] = 1;
    for (int i = rank - 1; i > 0; i--)
    {
        rs[i] = rs[i + 1] * nx[i];
        cs[i] = cs[i + 1] * (real && i == rank - 1 ? nx[i] / 2 + 1 : nx[i]);
    }

    const DFTI_CONFIG_VALUE domain = real ? DFTI_REAL : DFTI_COMPLEX;
    if (rank == 1)
    {
        mkl_dfti_check(DftiCreateDescriptor(&forward, DFTI_DOUBLE, domain, 1, lengths[0]), "DftiCreateDescriptor (forward)");
        mkl_dfti_check(DftiCreateDescriptor(&backward, DFTI_DOUBLE, domain, 1, lengths[0]), "DftiCreateDescriptor (backward)");
    }
    else
    {
        mkl_dfti_check(DftiCreateDescriptor(&forward, DFTI_DOUBLE, domain, rank, lengths.data()), "DftiCreateDescriptor (forward)");
        mkl_dfti_check(DftiSetValue(forward, mkl_dfti::forward_input_strides, rs.data()), "DftiSetValue (forward input strides)");
        mkl_dfti_check(DftiSetValue(forward, mkl_dfti::forward_output_strides, cs.data()), "DftiSetValue (forward output strides)");

        mkl_dfti_check(DftiCreateDescriptor(&backward, DFTI_DOUBLE, domain, rank, lengths.data()), "DftiCreateDescriptor (backward)");
        mkl_dfti_check(DftiSetValue(backward, mkl_dfti::backward_input_strides, cs.data()), "DftiSetValue (backward input strides)");
        mkl_dfti_check(DftiSetValue(backward, mkl_dfti::backward_output_strides, rs.data()), "DftiSetValue (backward output strides)");
    }

    mkl_dfti_check(DftiSetValue(forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, forward)");
    mkl_dfti_check(DftiSetValue(backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, backward)");
    if (real)
    {
        mkl_dfti_check(DftiSetValue(forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, forward)");
        mkl_dfti_check(DftiSetValue(backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, backward)");
    }

    mkl_dfti_check(DftiCommitDescriptor(forward), "DftiCommitDescriptor (forward)");
    mkl_dfti_check(DftiSetValue(backward, DFTI_BACKWARD_SCALE, backward_scale), "DftiSetValue(DFTI_BACKWARD_SCALE, backward)");
    mkl_dfti_check(DftiCommitDescriptor(backward), "DftiCommitDescriptor (backward)");
}
}  // namespace

//------------------------------------------------------------------------------
// The real FFT under MakhoulTransform
//------------------------------------------------------------------------------
MklRealFFT::MklRealFFT(const std::vector<int>& nx)
{
    try
    {
        create_descriptors(nx, true, 1.0, forward_, backward_);
    }
    catch (...)
    {
        release();
        throw;
    }
}

MklRealFFT::~MklRealFFT()
{
    release();
}

void MklRealFFT::release()
{
    if (forward_ != nullptr)
        DftiFreeDescriptor(&forward_);
    if (backward_ != nullptr)
        DftiFreeDescriptor(&backward_);
    forward_ = nullptr;
    backward_ = nullptr;
}

void MklRealFFT::forward(double* samples, std::complex<double>* modes) const
{
    mkl_dfti_check(DftiComputeForward(forward_, samples, modes), "DftiComputeForward (DCT/DST)");
}

void MklRealFFT::backward(std::complex<double>* modes, double* samples) const
{
    mkl_dfti_check(DftiComputeBackward(backward_, modes, samples), "DftiComputeBackward (DCT/DST)");
}

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
        // The backward transform includes the 1/N of the FFT<T> interface
        create_descriptors(std::vector<int>(nx_.begin(), nx_.end()), std::is_same<T, double>::value,
                           1.0 / static_cast<double>(total_grid_), hand_forward_, hand_backward_);
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
    const std::vector<int> nx(nx_.begin(), nx_.end());
    const std::vector<BoundaryCondition> bc(bc_.begin(), bc_.end());
    makhoul_ = std::make_unique<MakhoulTransform>(nx, bc, std::make_unique<MklRealFFT>(nx));
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
            // FFTW's DCT-II/DST-II are 2x the standard definition per dimension. A complex
            // field (CL-FTS) gets the coefficients of its real and imaginary parts as
            // (re, im) pairs, the periodic complex layout used by the solvers.
            makhoul_->forward(rdata, cdata, 1.0 / (1 << DIM));
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
            // A complex field takes its coefficients as (re, im) pairs (see forward)
            makhoul_->backward(cdata, rdata, 1.0 / static_cast<double>(total_grid_));
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
