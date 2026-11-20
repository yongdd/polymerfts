/**
 * @file FftwCrysFFTRecursive3m.cpp
 * @brief Recursive crystallographic FFT (2x2y2z) over the real FFT of FFTW.
 */

#include "FftwCrysFFTRecursive3m.h"
#include <cstring>
#include <memory>

#include "AlignedScratch.h"
#include "FftwFFT.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
FftwCrysFFTRecursive3m::FftwCrysFFTRecursive3m(
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
FftwCrysFFTRecursive3m::~FftwCrysFFTRecursive3m()
{
    release();
}

void FftwCrysFFTRecursive3m::release()
{
    if (plan_forward_) fftw_destroy_plan(plan_forward_);
    if (plan_backward_) fftw_destroy_plan(plan_backward_);
    plan_forward_ = nullptr;
    plan_backward_ = nullptr;
}

//------------------------------------------------------------------------------
// Initialize FFT plans
//------------------------------------------------------------------------------
void FftwCrysFFTRecursive3m::init_fft_plans()
{
    std::unique_ptr<double, FftwFree> dummy_in(
        static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
    std::unique_ptr<fftw_complex, FftwFree> dummy_out(
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_physical_)));
    // FFTW_PATIENT runs trial transforms on these arrays
    if (!dummy_in || !dummy_out)
        throw_with_line_number("Failed to allocate the FFTW planning buffers of FftwCrysFFTRecursive3m.");

    fftw_iodim dims[3];
    dims[0].n = nx_physical_[0];
    dims[0].is = nx_physical_[1] * nx_physical_[2];
    dims[0].os = nx_physical_[1] * nx_physical_[2];
    dims[1].n = nx_physical_[1];
    dims[1].is = nx_physical_[2];
    dims[1].os = nx_physical_[2];
    dims[2].n = nx_physical_[2];
    dims[2].is = 1;
    dims[2].os = 1;

    plan_forward_ = fftw_plan_guru_dft_r2c(
        3, dims, 0, nullptr, dummy_in.get(), dummy_out.get(), FFTW_PATIENT);
    plan_backward_ = fftw_plan_guru_dft_c2r(
        3, dims, 0, nullptr, dummy_out.get(), dummy_in.get(), FFTW_PATIENT);

    if (!plan_forward_ || !plan_backward_)
    {
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTRecursive3m.");
    }
}

//------------------------------------------------------------------------------
// The real FFT of the physical grid, on the caller's field or on a copy of this
// thread from a 64-byte boundary (AlignedScratch), which has the alignment of the
// planning arrays that the new-array execute functions need
//------------------------------------------------------------------------------
namespace
{
thread_local AlignedScratch copy_scratch;
}  // namespace

void FftwCrysFFTRecursive3m::forward(const double* q, std::complex<double>* spectrum) const
{
    const double* r2c_in = q;
    if (!has_fftw_plan_alignment(q))
    {
        double* copy = copy_scratch.get(M_physical_);
        std::memcpy(copy, q, sizeof(double) * M_physical_);
        r2c_in = copy;
    }
    fftw_execute_dft_r2c(plan_forward_, const_cast<double*>(r2c_in), reinterpret_cast<fftw_complex*>(spectrum));
}

void FftwCrysFFTRecursive3m::backward(std::complex<double>* spectrum, double* q) const
{
    if (has_fftw_plan_alignment(q))
    {
        fftw_execute_dft_c2r(plan_backward_, reinterpret_cast<fftw_complex*>(spectrum), q);
    }
    else
    {
        double* copy = copy_scratch.get(M_physical_);
        fftw_execute_dft_c2r(plan_backward_, reinterpret_cast<fftw_complex*>(spectrum), copy);
        std::memcpy(q, copy, sizeof(double) * M_physical_);
    }
}
