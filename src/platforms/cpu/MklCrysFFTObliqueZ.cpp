/**
 * @file MklCrysFFTObliqueZ.cpp
 * @brief MKL implementation of crystallographic FFT for z-mirror symmetry.
 *
 * MKL has no native DCT: the DCT-II/III of a column of N points goes through a real
 * FFT of the same length N (Makhoul 1980). The 2D FFT runs on contiguous z planes.
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see FftwCrysFFTObliqueZ for FFTW implementation
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "MklCrysFFTObliqueZ.h"
#include "MklDftiLayout.h"
#include "MklStatusCheck.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
/// Work arrays of the z DCT steps, one set per thread and shared by every
/// instance on that thread. Each array is sized from its own requirement: two
/// grids with the same Nx*Ny*Nz but a different Nz need the same real size and
/// different spectrum sizes.
struct ZWork
{
    std::vector<double> z_reordered;              ///< reordered columns, howmany * N
    std::vector<std::complex<double>> z_freq;     ///< their spectra, howmany * (N/2 + 1)
};
thread_local ZWork z_work;

ZWork& z_work_for(size_t n_reordered, size_t n_freq)
{
    if (z_work.z_reordered.size() != n_reordered)
        z_work.z_reordered.resize(n_reordered);
    if (z_work.z_freq.size() != n_freq)
        z_work.z_freq.resize(n_freq);
    return z_work;
}
}  // namespace

std::array<size_t, 2> MklCrysFFTObliqueZ::thread_z_work_sizes()
{
    return {z_work.z_reordered.size(), z_work.z_freq.size()};
}

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
MklCrysFFTObliqueZ::MklCrysFFTObliqueZ(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> trans_part)
    : Base(nx_logical, cell_para, trans_part)
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

MklCrysFFTObliqueZ::~MklCrysFFTObliqueZ()
{
    release();
}

void MklCrysFFTObliqueZ::release()
{
    if (plan_dct_forward_z_) DftiFreeDescriptor(&plan_dct_forward_z_);
    if (plan_dct_backward_z_) DftiFreeDescriptor(&plan_dct_backward_z_);
    if (plan_fft_forward_xy_) DftiFreeDescriptor(&plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) DftiFreeDescriptor(&plan_fft_backward_xy_);
    plan_dct_forward_z_ = plan_dct_backward_z_ = nullptr;
    plan_fft_forward_xy_ = plan_fft_backward_xy_ = nullptr;
}

//------------------------------------------------------------------------------
// MKL plan initialization
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::initFFTPlans()
{
    // DCT via FFT: real 1D FFT of length N = Nz/2 for each (x,y) column
    // Data layout: [Nx*Ny][N] with contiguous z-columns
    const int N = nx_physical_[2];
    const int howmany_z = nx_physical_[0] * nx_physical_[1];
    const MKL_LONG N_long = N;

    mkl_dfti_check(DftiCreateDescriptor(&plan_dct_forward_z_, DFTI_DOUBLE, DFTI_REAL, 1, N_long), "DftiCreateDescriptor (z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, mkl_dfti::forward_input_distance, N_long), "DftiSetValue (z forward input distance)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, mkl_dfti::forward_output_distance, static_cast<MKL_LONG>(N / 2 + 1)), "DftiSetValue (z forward output distance)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, z forward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_dct_forward_z_), "DftiCommitDescriptor (z forward)");

    mkl_dfti_check(DftiCreateDescriptor(&plan_dct_backward_z_, DFTI_DOUBLE, DFTI_REAL, 1, N_long), "DftiCreateDescriptor (z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, mkl_dfti::backward_input_distance, static_cast<MKL_LONG>(N / 2 + 1)), "DftiSetValue (z backward input distance)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, mkl_dfti::backward_output_distance, N_long), "DftiSetValue (z backward output distance)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(N)), "DftiSetValue (z backward scale)");
    mkl_dfti_check(DftiCommitDescriptor(plan_dct_backward_z_), "DftiCommitDescriptor (z backward)");

    // Real 2D FFT along X,Y for each Z plane, on the contiguous layout [N][Nx][Ny]:
    // on strided planes [Nx][Ny][N] MKL takes longer than the two transpositions add
    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];
    MKL_LONG dims_xy[2] = { static_cast<MKL_LONG>(Nx), static_cast<MKL_LONG>(Ny) };
    const MKL_LONG howmany_xy = N;
    const MKL_LONG distance_real = static_cast<MKL_LONG>(Nx) * Ny;
    const MKL_LONG distance_complex = static_cast<MKL_LONG>(Nx) * (Ny / 2 + 1);
    MKL_LONG real_strides_xy[3] = {0, static_cast<MKL_LONG>(Ny), 1};
    MKL_LONG complex_strides_xy[3] = {0, static_cast<MKL_LONG>(Ny / 2 + 1), 1};

    mkl_dfti_check(DftiCreateDescriptor(&plan_fft_forward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy), "DftiCreateDescriptor (xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_NUMBER_OF_TRANSFORMS, howmany_xy), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, mkl_dfti::forward_input_distance, distance_real), "DftiSetValue (xy forward input distance)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, mkl_dfti::forward_output_distance, distance_complex), "DftiSetValue (xy forward output distance)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, mkl_dfti::forward_input_strides, real_strides_xy), "DftiSetValue (xy forward input strides)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, mkl_dfti::forward_output_strides, complex_strides_xy), "DftiSetValue (xy forward output strides)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, xy forward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_fft_forward_xy_), "DftiCommitDescriptor (xy forward)");

    mkl_dfti_check(DftiCreateDescriptor(&plan_fft_backward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy), "DftiCreateDescriptor (xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_NUMBER_OF_TRANSFORMS, howmany_xy), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, mkl_dfti::backward_input_distance, distance_complex), "DftiSetValue (xy backward input distance)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, mkl_dfti::backward_output_distance, distance_real), "DftiSetValue (xy backward output distance)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, mkl_dfti::backward_input_strides, complex_strides_xy), "DftiSetValue (xy backward input strides)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, mkl_dfti::backward_output_strides, real_strides_xy), "DftiSetValue (xy backward output strides)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(distance_real)), "DftiSetValue (xy backward scale)");
    mkl_dfti_check(DftiCommitDescriptor(plan_fft_backward_xy_), "DftiCommitDescriptor (xy backward)");

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create MKL DFTI plans for MklCrysFFTObliqueZ.");

    // Precompute DCT twiddle factors
    dct_cos_.resize(N / 2 + 1);
    dct_sin_.resize(N / 2 + 1);
    const double inv_2N = 1.0 / (2.0 * static_cast<double>(N));
    for (int k = 0; k <= N / 2; ++k)
    {
        const double theta = M_PI * static_cast<double>(k) * inv_2N;
        dct_cos_[k] = std::cos(theta);
        dct_sin_[k] = std::sin(theta);
    }
}

//------------------------------------------------------------------------------
// DCT-II forward along Z through a real FFT of the same length (Makhoul 1980)
// DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))
// With v = (x[0], x[2], ..., x[3], x[1]) and its FFT V, W[k] = exp(-i*pi*k/(2N)):
// X[k] = Re(W[k] V[k]) and X[N-k] = -Im(W[k] V[k])
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::applyDCT2ForwardZ(const double* in, double* out) const
{
    const int N = nx_physical_[2];
    const int howmany = nx_logical_[0] * nx_logical_[1];
    const int freq_stride = N / 2 + 1;

    ZWork& buffers = z_work_for(static_cast<size_t>(howmany) * N, static_cast<size_t>(howmany) * freq_stride);

    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const double* x = in + static_cast<size_t>(ixy) * N;
        double* v = buffers.z_reordered.data() + static_cast<size_t>(ixy) * N;
        for (int j = 0; 2 * j < N; ++j)
            v[j] = x[2 * j];
        for (int j = 0; 2 * j + 1 < N; ++j)
            v[N - 1 - j] = x[2 * j + 1];
    }

    // Batched FFT along z
    mkl_dfti_check(DftiComputeForward(plan_dct_forward_z_, buffers.z_reordered.data(), buffers.z_freq.data()), "DftiComputeForward (z DCT-II)");

    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const std::complex<double>* V = buffers.z_freq.data() + static_cast<size_t>(ixy) * freq_stride;
        double* X = out + static_cast<size_t>(ixy) * N;
        X[0] = V[0].real();
        for (int k = 1; k <= N / 2; ++k)
        {
            X[k] = V[k].real() * dct_cos_[k] + V[k].imag() * dct_sin_[k];
            X[N - k] = V[k].real() * dct_sin_[k] - V[k].imag() * dct_cos_[k];
        }
    }
}

//------------------------------------------------------------------------------
// DCT-III backward along Z, the inverse of the DCT-II above:
// x[n] = (X[0] + 2 sum_{k=1}^{N-1} X[k] * cos(pi*k*(2n+1)/(2N))) / N
// V[k] = exp(i*pi*k/(2N)) (X[k] - i X[N-k]), and the scaled inverse FFT gives v
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::applyDCT3BackwardZ(const double* in, double* out) const
{
    const int N = nx_physical_[2];
    const int howmany = nx_logical_[0] * nx_logical_[1];
    const int freq_stride = N / 2 + 1;

    ZWork& buffers = z_work_for(static_cast<size_t>(howmany) * N, static_cast<size_t>(howmany) * freq_stride);

    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const double* X = in + static_cast<size_t>(ixy) * N;
        std::complex<double>* V = buffers.z_freq.data() + static_cast<size_t>(ixy) * freq_stride;
        V[0] = X[0];
        for (int k = 1; k <= N / 2; ++k)
        {
            V[k] = std::complex<double>(X[k] * dct_cos_[k] + X[N - k] * dct_sin_[k],
                                        X[k] * dct_sin_[k] - X[N - k] * dct_cos_[k]);
        }
    }

    // Batched backward FFT along z
    mkl_dfti_check(DftiComputeBackward(plan_dct_backward_z_, buffers.z_freq.data(), buffers.z_reordered.data()), "DftiComputeBackward (z DCT-III)");

    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const double* v = buffers.z_reordered.data() + static_cast<size_t>(ixy) * N;
        double* x = out + static_cast<size_t>(ixy) * N;
        for (int j = 0; 2 * j < N; ++j)
            x[2 * j] = v[j];
        for (int j = 0; 2 * j + 1 < N; ++j)
            x[2 * j + 1] = v[N - 1 - j];
    }
}

//------------------------------------------------------------------------------
// Shared pipeline of diffusion() and apply_multiplier():
// DCT-II(z) -> to z planes -> FFT(xy) -> multiply -> IFFT(xy) -> from z planes -> DCT-III(z)
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const
{
    const int N = nx_physical_[2];
    const int n_real = nx_logical_[0] * nx_logical_[1];
    const int n_complex = nx_logical_[0] * (nx_logical_[1] / 2 + 1);

    struct ThreadBuffers
    {
        std::unique_ptr<double[]> columns;   // [Nx][Ny][N]
        std::unique_ptr<double[]> planes;    // [N][Nx][Ny]
        std::unique_ptr<std::complex<double>[]> complex;   // [N][Nx][Ny/2 + 1]
        int size_real = 0;
        int size_complex = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size_real != M_physical_ || buffers.size_complex != M_complex_)
    {
        buffers.size_real = buffers.size_complex = 0;
        buffers.columns = std::make_unique<double[]>(M_physical_);
        buffers.planes = std::make_unique<double[]>(M_physical_);
        buffers.complex = std::make_unique<std::complex<double>[]>(M_complex_);
        buffers.size_real = M_physical_;
        buffers.size_complex = M_complex_;
    }
    double* columns = buffers.columns.get();
    double* planes = buffers.planes.get();
    std::complex<double>* complex = buffers.complex.get();

    // Step 1: DCT-II along Z
    applyDCT2ForwardZ(q_in, columns);

    // Step 2: Batched 2D FFT along XY of the z planes
    for (int iz = 0; iz < N; ++iz)
        for (int i = 0; i < n_real; ++i)
            planes[iz * n_real + i] = columns[i * N + iz];
    mkl_dfti_check(DftiComputeForward(plan_fft_forward_xy_, planes, complex), "DftiComputeForward (xy)");

    // Step 3: Apply the spectral multiplier (Boltzmann factor or stress multiplier),
    // stored with z running fastest
    for (int iz = 0; iz < N; ++iz)
        for (int i = 0; i < n_complex; ++i)
            complex[iz * n_complex + i] *= multiplier[i * N + iz];

    // Step 4: Batched 2D IFFT along XY, scaled by 1/(Nx*Ny)
    mkl_dfti_check(DftiComputeBackward(plan_fft_backward_xy_, complex, planes), "DftiComputeBackward (xy)");
    for (int iz = 0; iz < N; ++iz)
        for (int i = 0; i < n_real; ++i)
            columns[i * N + iz] = planes[iz * n_real + i];

    // Step 5: DCT-III along Z, the inverse of step 1
    applyDCT3BackwardZ(columns, q_out);
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::diffusion(double* q_in, double* q_out)
{
    const double* boltz = current_boltzmann();
    if (!boltz)
        throw_with_line_number("MklCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    apply_spectral_multiplier(q_in, q_out, boltz);
}

void MklCrysFFTObliqueZ::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    apply_spectral_multiplier(q_in, q_out, multiplier);
}
