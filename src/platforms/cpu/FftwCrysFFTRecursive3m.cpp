/**
 * @file FftwCrysFFTRecursive3m.cpp
 * @brief FFTW implementation of recursive crystallographic FFT (2x2y2z).
 */

#include "FftwCrysFFTRecursive3m.h"
#include <algorithm>
#include <cstring>
#include <memory>

#include "AlignedScratch.h"
#include "FftwFFT.h"

namespace {

constexpr size_t kAlignment = 64;

inline size_t align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
FftwCrysFFTRecursive3m::FftwCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 9> translational_part)
    : Base(nx_logical, translational_part)
{
    try
    {
        init_fft_plans();
        generate_twiddle_factors();
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
// Apply FFT with a folded table (Boltzmann factors or stress multiplier)
//------------------------------------------------------------------------------
void FftwCrysFFTRecursive3m::apply_with_table(
    const double* table, const double* q_in, double* q_out) const
{
    // Work arrays of this thread, from 64-byte boundaries (AlignedScratch), which have the
    // alignment of the planning arrays that the new-array execute functions need. The
    // transforms run on the caller's fields when they have it (the r2c transform keeps its
    // input) and otherwise on a copy.
    thread_local AlignedScratch copy_scratch, step1_scratch, step2_scratch;
    const size_t complex_doubles = 2 * static_cast<size_t>(M_physical_);
    fftw_complex* step1 = reinterpret_cast<fftw_complex*>(step1_scratch.get(complex_doubles));
    fftw_complex* step2 = reinterpret_cast<fftw_complex*>(step2_scratch.get(complex_doubles));

    const double* r2c_in = q_in;
    if (!has_fftw_plan_alignment(q_in))
    {
        double* copy = copy_scratch.get(M_physical_);
        std::memcpy(copy, q_in, sizeof(double) * M_physical_);
        r2c_in = copy;
    }

    // Forward r2c FFT
    fftw_execute_dft_r2c(plan_forward_, const_cast<double*>(r2c_in), step1);

    const int Nx2 = nx_physical_[0];
    const int Ny2 = nx_physical_[1];
    const int Nz2 = nx_physical_[2];
    // Pad the packed z count for vectorization, but clamp to the row stride
    // Nz2 so the loop never overruns the Nx2*Ny2*Nz2-sized buffers. The clamp
    // still covers the packed range [0, Nz2/2] for any Nz2 >= 2.
    const size_t Nz_qua = std::min(
        align_up(static_cast<size_t>(Nz2 / 2 + 1), kAlignment / 8),
        static_cast<size_t>(Nz2));

    // The real parts of the eight folded octants, then their imaginary parts
    std::array<const double*, 8> k_re, k_im;
    for (int i = 0; i < 8; ++i)
    {
        k_re[i] = table + i * table_stride_;
        k_im[i] = table + (8 + i) * table_stride_;
    }

    for (int ix = 0; ix < Nx2; ++ix)
    {
        for (int iy = 0; iy < Ny2; ++iy)
        {
            const size_t base_shift = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
            const fftw_complex* src000 = step1 + base_shift;
            const fftw_complex* src110 = step1 + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const fftw_complex* src010 = step1 + (static_cast<size_t>(ix) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const fftw_complex* src100 = step1 + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + iy) * Nz2;
            fftw_complex* dst = step2 + base_shift;

            for (size_t iz = 0; iz < Nz_qua; ++iz)
            {
                size_t idx = base_shift + iz;
                dst[iz][0]
                    = src000[iz][0] * k_re[0][idx] - src000[iz][1] * k_im[7][idx]
                    + src110[iz][0] * k_re[6][idx] - src110[iz][1] * k_im[1][idx]
                    + src010[iz][0] * k_re[2][idx] - src010[iz][1] * k_im[5][idx]
                    + src100[iz][0] * k_re[4][idx] - src100[iz][1] * k_im[3][idx];

                dst[iz][1]
                    = src000[iz][0] * k_im[0][idx] + src000[iz][1] * k_re[7][idx]
                    + src110[iz][0] * k_im[6][idx] + src110[iz][1] * k_re[1][idx]
                    + src010[iz][0] * k_im[2][idx] + src010[iz][1] * k_re[5][idx]
                    + src100[iz][0] * k_im[4][idx] + src100[iz][1] * k_re[3][idx];
            }
        }
    }

    // Backward c2r FFT
    if (has_fftw_plan_alignment(q_out))
    {
        fftw_execute_dft_c2r(plan_backward_, step2, q_out);
    }
    else
    {
        double* copy = copy_scratch.get(M_physical_);
        fftw_execute_dft_c2r(plan_backward_, step2, copy);
        std::memcpy(q_out, copy, sizeof(double) * M_physical_);
    }
}
