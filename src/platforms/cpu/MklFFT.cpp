/**
 * @file MklFFT.cpp
 * @brief Intel MKL periodic FFT (DFTI r2c/c2r, c2c) and the real FFT under MakhoulTransform.
 *
 * Backward transforms of MklFFT include 1/N scaling so backward(forward(x)) = x.
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
    mkl_dfti_check(DftiComputeForward(forward_, samples, modes), "DftiComputeForward (MklRealFFT)");
}

void MklRealFFT::backward(std::complex<double>* modes, double* samples) const
{
    mkl_dfti_check(DftiComputeBackward(backward_, modes, samples), "DftiComputeBackward (MklRealFFT)");
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
MklFFT<T>::MklFFT(const std::vector<int>& nx)
{
    total_grid_ = 1;
    for (int n : nx)
        total_grid_ *= n;

    // r2c: the half spectrum of the last axis (Hermitian symmetry); c2c: the full grid
    if constexpr (std::is_same<T, double>::value)
        total_complex_grid_ = total_grid_ / nx.back() * (nx.back() / 2 + 1);
    else
        total_complex_grid_ = total_grid_;

    try
    {
        // The backward transform includes the 1/N of the FFT<T> interface
        create_descriptors(nx, std::is_same<T, double>::value,
                           1.0 / static_cast<double>(total_grid_), hand_forward_, hand_backward_);
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
template <typename T>
MklFFT<T>::~MklFFT()
{
    release();
}

template <typename T>
void MklFFT<T>::release()
{
    if (hand_forward_ != nullptr)
        DftiFreeDescriptor(&hand_forward_);
    if (hand_backward_ != nullptr)
        DftiFreeDescriptor(&hand_backward_);
    hand_forward_ = nullptr;
    hand_backward_ = nullptr;
}

//------------------------------------------------------------------------------
// FFT<T> interface: the complex coefficients as interleaved doubles
//------------------------------------------------------------------------------
template <typename T>
void MklFFT<T>::forward(T *rdata, double *cdata)
{
    mkl_dfti_check(DftiComputeForward(hand_forward_, rdata, reinterpret_cast<std::complex<double>*>(cdata)),
                   "DftiComputeForward (periodic)");
}

template <typename T>
void MklFFT<T>::backward(double *cdata, T *rdata)
{
    mkl_dfti_check(DftiComputeBackward(hand_backward_, reinterpret_cast<std::complex<double>*>(cdata), rdata),
                   "DftiComputeBackward (periodic)");
}

template <typename T>
void MklFFT<T>::backward_with_factors(double *cdata, const double *k_factor, T *rdata, const T *r_factor)
{
    if (k_factor != nullptr)
    {
        std::complex<double>* modes = reinterpret_cast<std::complex<double>*>(cdata);
        for (int i = 0; i < total_complex_grid_; ++i)
            modes[i] *= k_factor[i];
    }
    backward(cdata, rdata);
    if (r_factor != nullptr)
        for (int i = 0; i < total_grid_; ++i)
            rdata[i] *= r_factor[i];
}

// Explicit template instantiations
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(MklFFT);
