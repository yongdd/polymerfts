/**
 * @file MklRealTransform.cpp
 * @brief MKL real-to-real DCT/DST implementation using FFT and TT interface.
 *
 * Uses FFT-based O(N log N) algorithms, and Intel MKL Trigonometric
 * Transform (TT) routines for DCT-IV and DST-IV.
 *
 * MKL TT to FFTW mapping (empirically verified):
 *   - STAGGERED2_COSINE forward × N = DCT-IV (FFTW REDFT11)
 *   - STAGGERED2_SINE forward × N   = DST-IV (FFTW RODFT11)
 *
 * FFT-based O(N log N) implementations:
 *   - DCT-I: via 2(N-1) point FFT symmetric extension
 *   - DST-I: via 2(N+1) point FFT antisymmetric extension
 *   - DCT-II, DCT-III, DST-II, DST-III: N point FFT, MakhoulTransform (Makhoul 1980)
 */

#include "MklRealTransform.h"

#include <complex>
#include <cstring>
#include <string>
#include <vector>

#include "mkl_dfti.h"
#include "mkl_trig_transforms.h"
#include "Exception.h"
#include "MakhoulTransform.h"
#include "MklFFT.h"

//==============================================================================
// FFT-based O(N log N) implementations
//==============================================================================

/**
 * @brief DCT-I via FFT: O(N log N) using 2(N-1) point symmetric extension
 */
static void fft_dct1(double* data, int N, const MklRealFFT& fft)
{
    int M = 2 * (N - 1);

    thread_local std::vector<double> extended;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(extended.size()) < M)
    {
        extended.resize(M);
        fft_out.resize(M / 2 + 1);
    }

    // Create symmetric extension: [x0, x1, ..., x_{N-1}, x_{N-2}, ..., x_1]
    for (int i = 0; i < N; ++i)
        extended[i] = data[i];
    for (int i = 1; i < N - 1; ++i)
        extended[N - 1 + i] = data[N - 1 - i];

    fft.forward(extended.data(), fft_out.data());

    // Extract DCT-I coefficients (real parts of first N FFT coefficients)
    for (int k = 0; k < N; ++k)
        data[k] = fft_out[k].real();
}

/**
 * @brief DST-I via FFT: O(N log N) using 2(N+1) point antisymmetric extension
 */
static void fft_dst1(double* data, int N, const MklRealFFT& fft)
{
    int M = 2 * (N + 1);

    thread_local std::vector<double> extended;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(extended.size()) < M)
    {
        extended.resize(M);
        fft_out.resize(M / 2 + 1);
    }

    // Create antisymmetric extension: [0, x0, x1, ..., x_{N-1}, 0, -x_{N-1}, ..., -x_0]
    extended[0] = 0.0;
    for (int i = 0; i < N; ++i)
        extended[i + 1] = data[i];
    extended[N + 1] = 0.0;
    for (int i = 0; i < N; ++i)
        extended[N + 2 + i] = -data[N - 1 - i];

    fft.forward(extended.data(), fft_out.data());

    // Extract DST-I coefficients: Y[k] = -Im(X[k+1])
    for (int k = 0; k < N; ++k)
        data[k] = -fft_out[k + 1].imag();
}

//==============================================================================
// MKL TT Wrapper Classes
//==============================================================================

/**
 * @brief Wrapper for MKL TT transforms.
 *
 * Supported transforms:
 *   - DCT-IV: STAGGERED2_COSINE forward × N
 *   - DST-IV: STAGGERED2_SINE forward × N
 */
class MklTTHandle
{
public:
    MklTTHandle(int N, MklTransformType type)
        : N_(N), scale_(static_cast<double>(N))
    {
        // Determine MKL TT type
        switch (type)
        {
            case MKL_DCT_4:
                tt_type_ = MKL_STAGGERED2_COSINE_TRANSFORM;
                break;
            case MKL_DST_4:
                tt_type_ = MKL_STAGGERED2_SINE_TRANSFORM;
                break;
            default:
                throw_with_line_number("Transform type not supported by MKL TT: " +
                                       std::string(getTransformName(type)));
        }

        // Allocate work buffer
        work_.resize(N + 16);

        // dpar size requirement from MKL docs
        dpar_.resize(10 * N + 100);

        MKL_INT n = N_;
        MKL_INT stat = 0;

        // Initialize TT parameters
        d_init_trig_transform(&n, &tt_type_, ipar_, dpar_.data(), &stat);
        if (stat != 0)
            throw_with_line_number("d_init_trig_transform failed, stat: " + std::to_string(stat));

        // Commit the transform (creates handle_). The destructor does not run
        // for a partly constructed object, so free a handle a failed commit left.
        d_commit_trig_transform(work_.data(), &handle_, ipar_, dpar_.data(), &stat);
        if (stat != 0)
        {
            release();
            throw_with_line_number("d_commit_trig_transform failed, stat: " + std::to_string(stat));
        }
    }

    ~MklTTHandle()
    {
        release();
    }

    void execute(double* data)
    {
        // Copy data to work buffer (MKL TT works in-place)
        std::memcpy(work_.data(), data, N_ * sizeof(double));

        MKL_INT stat = 0;
        d_forward_trig_transform(work_.data(), &handle_, ipar_, dpar_.data(), &stat);
        if (stat != 0)
            throw_with_line_number("MKL TT transform failed, stat: " + std::to_string(stat));

        // Apply scaling to match FFTW output
        for (int i = 0; i < N_; ++i)
            data[i] = work_[i] * scale_;
    }

private:
    int N_;
    MKL_INT tt_type_;
    double scale_;
    DFTI_DESCRIPTOR_HANDLE handle_{nullptr};
    MKL_INT ipar_[128]{};
    std::vector<double> dpar_;
    std::vector<double> work_;

    /// Free the TT handle (also after a failed commit); idempotent.
    void release()
    {
        if (handle_ != nullptr)
        {
            MKL_INT stat = 0;
            free_trig_transform(&handle_, ipar_, &stat);
            handle_ = nullptr;
        }
    }
};

//==============================================================================
// MklRealTransform::Line
//==============================================================================

/// The transform of a line of n contiguous values, in place
class MklRealTransform::Line
{
public:
    Line(int n, MklTransformType type) : n_(n), type_(type)
    {
        switch (type)
        {
            case MKL_DCT_1:
                extension_fft_ = std::make_unique<MklRealFFT>(std::vector<int>{2 * (n - 1)});
                break;
            case MKL_DST_1:
                extension_fft_ = std::make_unique<MklRealFFT>(std::vector<int>{2 * (n + 1)});
                break;
            case MKL_DCT_2:
            case MKL_DCT_3:
            case MKL_DST_2:
            case MKL_DST_3:
            {
                const BoundaryCondition axis = (type == MKL_DCT_2 || type == MKL_DCT_3)
                    ? BoundaryCondition::REFLECTING : BoundaryCondition::ABSORBING;
                makhoul_ = std::make_unique<MakhoulTransform>(std::vector<int>{n}, std::vector<BoundaryCondition>{axis},
                                                              std::make_unique<MklRealFFT>(std::vector<int>{n}));
                break;
            }
            case MKL_DCT_4:
            case MKL_DST_4:
                tt_ = std::make_unique<MklTTHandle>(n, type);
                break;
            default:
                throw_with_line_number("Unknown transform type " + std::to_string(static_cast<int>(type)) + ".");
        }
    }

    void execute(double* line)
    {
        switch (type_)
        {
            case MKL_DCT_1: fft_dct1(line, n_, *extension_fft_); break;
            case MKL_DST_1: fft_dst1(line, n_, *extension_fft_); break;
            case MKL_DCT_2:
            case MKL_DST_2: makhoul_->forward(line, line, 1.0); break;    // REDFT10, RODFT10
            case MKL_DCT_3:
            case MKL_DST_3: makhoul_->backward(line, line, 1.0); break;   // REDFT01, RODFT01
            default:        tt_->execute(line); break;
        }
    }

private:
    int n_;
    MklTransformType type_;
    std::unique_ptr<MklRealFFT> extension_fft_;   ///< DCT-I, DST-I: the real FFT of the extended line
    std::unique_ptr<MakhoulTransform> makhoul_;   ///< DCT-II/III, DST-II/III
    std::unique_ptr<MklTTHandle> tt_;             ///< DCT-IV, DST-IV
};

//==============================================================================
// MklRealTransform
//==============================================================================

MklRealTransform::MklRealTransform(const std::vector<int>& nx, const std::vector<MklTransformType>& types)
    : nx_(nx)
{
    if (nx.empty() || nx.size() > 3 || types.size() != nx.size())
        throw_with_line_number("MklRealTransform takes one to three axes and one transform type per axis.");
    for (size_t d = 0; d < nx.size(); ++d)
    {
        if (nx[d] < 2)
            throw_with_line_number("Transform sizes must be >= 2");
        lines_.push_back(std::make_unique<Line>(nx[d], types[d]));
    }
}

MklRealTransform::~MklRealTransform() = default;

void MklRealTransform::execute(double* data)
{
    const int rank = static_cast<int>(nx_.size());
    thread_local std::vector<double> line;

    // The lines of the last axis first, then those of the axes before it
    for (int d = rank - 1; d >= 0; --d)
    {
        const int n = nx_[d];
        int stride = 1;
        for (int e = d + 1; e < rank; ++e)
            stride *= nx_[e];
        int num_lines = 1;
        for (int e = 0; e < d; ++e)
            num_lines *= nx_[e];
        if (static_cast<int>(line.size()) < n)
            line.resize(n);

        for (int batch = 0; batch < num_lines; ++batch)
        {
            for (int s = 0; s < stride; ++s)
            {
                double* first = data + batch * n * stride + s;
                for (int j = 0; j < n; ++j)
                    line[j] = first[j * stride];
                lines_[d]->execute(line.data());
                for (int j = 0; j < n; ++j)
                    first[j * stride] = line[j];
            }
        }
    }
}
