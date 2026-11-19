/**
 * @file FftwCrysFFTObliqueZ.cpp
 * @brief FFTW implementation of crystallographic FFT for z-mirror symmetry.
 */

#include <cstring>
#include <memory>

#include "AlignedScratch.h"
#include "FftwCrysFFTObliqueZ.h"
#include "FftwFFT.h"

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
FftwCrysFFTObliqueZ::FftwCrysFFTObliqueZ(std::array<int, 3> nx_logical)
    : Base(nx_logical)
{
    try
    {
        initFFTPlans();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

FftwCrysFFTObliqueZ::~FftwCrysFFTObliqueZ()
{
    release();
}

void FftwCrysFFTObliqueZ::release()
{
    if (plan_dct_forward_z_) fftw_destroy_plan(plan_dct_forward_z_);
    if (plan_dct_backward_z_) fftw_destroy_plan(plan_dct_backward_z_);
    if (plan_fft_forward_xy_) fftw_destroy_plan(plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) fftw_destroy_plan(plan_fft_backward_xy_);
    plan_dct_forward_z_ = plan_dct_backward_z_ = nullptr;
    plan_fft_forward_xy_ = plan_fft_backward_xy_ = nullptr;
}

//------------------------------------------------------------------------------
// FFTW plan initialization
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::initFFTPlans()
{
    const unsigned plan_flags = FFTW_MEASURE;

    // Planning arrays only, freed on return: the transforms run on the
    // thread-local scratch of apply_spectral_multiplier() and on the caller's
    // fields of the same alignment (new-array execute)
    std::unique_ptr<double, FftwFree> io_buffer(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
    std::unique_ptr<double, FftwFree> temp_buffer(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
    std::unique_ptr<fftw_complex, FftwFree> complex_buffer(
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_)));

    if (!io_buffer || !temp_buffer || !complex_buffer)
        throw_with_line_number("Failed to allocate FFTW buffers for FftwCrysFFTObliqueZ.");

    // DCT-II/III along Z for each (x,y) column
    int n_z[1] = {nx_physical_[2]};
    int howmany_z = nx_physical_[0] * nx_physical_[1];
    int inembed_z[1] = {n_z[0]};
    int onembed_z[1] = {n_z[0]};
    int istride_z = 1;
    int ostride_z = 1;
    int idist_z = n_z[0];
    int odist_z = n_z[0];

    fftw_r2r_kind kind_forward[1] = {FFTW_REDFT10};
    fftw_r2r_kind kind_backward[1] = {FFTW_REDFT01};

    plan_dct_forward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        io_buffer.get(), inembed_z, istride_z, idist_z,
        temp_buffer.get(), onembed_z, ostride_z, odist_z,
        kind_forward, plan_flags);

    plan_dct_backward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        temp_buffer.get(), inembed_z, istride_z, idist_z,
        io_buffer.get(), onembed_z, ostride_z, odist_z,
        kind_backward, plan_flags);

    // Real-to-complex FFT along X,Y for each Z slice (batched, strided)
    int n_xy[2] = {nx_logical_[0], nx_logical_[1]};
    int howmany_xy = nx_physical_[2];
    int inembed_xy[2] = {n_xy[0], n_xy[1]};
    int onembed_xy[2] = {n_xy[0], n_xy[1] / 2 + 1};
    int istride_xy = nx_physical_[2];
    int ostride_xy = nx_physical_[2];
    int idist_xy = 1;
    int odist_xy = 1;

    plan_fft_forward_xy_ = fftw_plan_many_dft_r2c(
        2, n_xy, howmany_xy,
        temp_buffer.get(), inembed_xy, istride_xy, idist_xy,
        complex_buffer.get(), onembed_xy, ostride_xy, odist_xy,
        plan_flags);

    plan_fft_backward_xy_ = fftw_plan_many_dft_c2r(
        2, n_xy, howmany_xy,
        complex_buffer.get(), onembed_xy, ostride_xy, odist_xy,
        temp_buffer.get(), inembed_xy, istride_xy, idist_xy,
        plan_flags);

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTObliqueZ.");
}

//------------------------------------------------------------------------------
// z-direction DCT helpers and the steps of diffusion()
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::dct2_execute(const double* in, double* out) const
{
    fftw_execute_r2r(plan_dct_forward_z_, const_cast<double*>(in), out);
}

void FftwCrysFFTObliqueZ::dct3_execute(const double* in, double* out) const
{
    fftw_execute_r2r(plan_dct_backward_z_, const_cast<double*>(in), out);
}

void FftwCrysFFTObliqueZ::apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const
{
    // Work arrays of this thread, from 64-byte boundaries (AlignedScratch), which have the
    // alignment of the planning arrays that the new-array execute functions need
    thread_local AlignedScratch copy_scratch, real_scratch, complex_scratch;
    double* real_tmp = real_scratch.get(M_physical_);
    fftw_complex* spectrum = reinterpret_cast<fftw_complex*>(complex_scratch.get(2 * static_cast<size_t>(M_complex_)));

    // The z transforms run on the caller's fields when they have that alignment (the r2r
    // transforms keep their input), otherwise on a copy
    const double* dct_in = q_in;
    if (!has_fftw_plan_alignment(q_in))
    {
        double* copy = copy_scratch.get(M_physical_);
        std::memcpy(copy, q_in, sizeof(double) * M_physical_);
        dct_in = copy;
    }

    dct2_execute(dct_in, real_tmp);
    fftw_execute_dft_r2c(plan_fft_forward_xy_, real_tmp, spectrum);

    for (int i = 0; i < M_complex_; ++i)
    {
        spectrum[i][0] *= multiplier[i];
        spectrum[i][1] *= multiplier[i];
    }

    fftw_execute_dft_c2r(plan_fft_backward_xy_, spectrum, real_tmp);
    double* dct_out = has_fftw_plan_alignment(q_out) ? q_out : copy_scratch.get(M_physical_);
    dct3_execute(real_tmp, dct_out);

    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = dct_out[i] * norm_factor_;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const
{
    check_table(table, M_complex_);
    const double* boltz = table.data();

    apply_spectral_multiplier(q_in, q_out, boltz);
}
