/**
 * @file MklCrysFFTRecursive3m.cpp
 * @brief Recursive crystallographic FFT (2x2y2z) over the real FFT of MKL.
 */

#include "MklCrysFFTRecursive3m.h"
#include "MklDftiLayout.h"
#include "MklStatusCheck.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MklCrysFFTRecursive3m::MklCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 9> translational_part)
    : CrysFFTRecursive3mBase(nx_logical, translational_part)
{
    try
    {
        init_fft_plans();
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
MklCrysFFTRecursive3m::~MklCrysFFTRecursive3m()
{
    release();
}

void MklCrysFFTRecursive3m::release()
{
    if (plan_forward_) DftiFreeDescriptor(&plan_forward_);
    if (plan_backward_) DftiFreeDescriptor(&plan_backward_);
    plan_forward_ = nullptr;
    plan_backward_ = nullptr;
}

//------------------------------------------------------------------------------
// Initialize MKL DFTI descriptors
//------------------------------------------------------------------------------
void MklCrysFFTRecursive3m::init_fft_plans()
{
    MKL_LONG dims[3] = { nx_physical_[0], nx_physical_[1], nx_physical_[2] };
    MKL_LONG strides_real[4] = { 0, nx_physical_[1] * nx_physical_[2], nx_physical_[2], 1 };
    MKL_LONG strides_complex[4] = { 0, nx_physical_[1] * nx_physical_[2], nx_physical_[2], 1 };

    // Forward r2c plan
    mkl_dfti_check(DftiCreateDescriptor(&plan_forward_, DFTI_DOUBLE, DFTI_REAL, 3, dims), "DftiCreateDescriptor (3m forward)");
    mkl_dfti_check(DftiSetValue(plan_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, 3m forward)");
    mkl_dfti_check(DftiSetValue(plan_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, 3m forward)");
    mkl_dfti_check(DftiSetValue(plan_forward_, mkl_dfti::forward_input_strides, strides_real), "DftiSetValue (3m forward input strides)");
    mkl_dfti_check(DftiSetValue(plan_forward_, mkl_dfti::forward_output_strides, strides_complex), "DftiSetValue (3m forward output strides)");
    mkl_dfti_check(DftiCommitDescriptor(plan_forward_), "DftiCommitDescriptor (3m forward)");

    // Backward c2r plan
    mkl_dfti_check(DftiCreateDescriptor(&plan_backward_, DFTI_DOUBLE, DFTI_REAL, 3, dims), "DftiCreateDescriptor (3m backward)");
    mkl_dfti_check(DftiSetValue(plan_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, 3m backward)");
    mkl_dfti_check(DftiSetValue(plan_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, 3m backward)");
    mkl_dfti_check(DftiSetValue(plan_backward_, mkl_dfti::backward_input_strides, strides_complex), "DftiSetValue (3m backward input strides)");
    mkl_dfti_check(DftiSetValue(plan_backward_, mkl_dfti::backward_output_strides, strides_real), "DftiSetValue (3m backward output strides)");
    mkl_dfti_check(DftiCommitDescriptor(plan_backward_), "DftiCommitDescriptor (3m backward)");
}

//------------------------------------------------------------------------------
// The real FFT of the physical grid
//------------------------------------------------------------------------------
void MklCrysFFTRecursive3m::forward(const double* q, std::complex<double>* spectrum) const
{
    mkl_dfti_check(DftiComputeForward(plan_forward_, const_cast<double*>(q), spectrum), "DftiComputeForward (3m)");
}

void MklCrysFFTRecursive3m::backward(std::complex<double>* spectrum, double* q) const
{
    mkl_dfti_check(DftiComputeBackward(plan_backward_, spectrum, q), "DftiComputeBackward (3m)");
}
