/**
 * @file MakhoulTransform.cpp
 * @brief DCT-II/III and DST-II/III of a whole grid through one periodic real FFT.
 *
 * @see MakhoulTransform.h for the algorithm
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include "Exception.h"
#include "MakhoulTransform.h"

namespace
{
using Mode = std::complex<double>;

/**
 * @brief Forward step of one axis on blocks of `count` modes, block k at modes + k * count.
 *
 * With a = W^k V_k and b = W^-k V_{-k}: Y_k = a + b and Y_{n-k} = i (a - b).
 * Spelled out because the complex product of the standard library checks for NaN.
 */
void combine_modes(Mode* modes, int n, int count, const Mode* w)
{
    for (int i = 0; i < count; ++i)
        modes[i] *= 2.0;
    for (int k = 1; 2 * k < n; ++k)
    {
        Mode* v_k = modes + k * count;
        Mode* v_nk = modes + (n - k) * count;
        const double c = w[k].real();
        const double s = w[k].imag();
        for (int i = 0; i < count; ++i)
        {
            const double a_re = v_k[i].real() * c - v_k[i].imag() * s;
            const double a_im = v_k[i].real() * s + v_k[i].imag() * c;
            const double b_re = v_nk[i].real() * c + v_nk[i].imag() * s;
            const double b_im = v_nk[i].imag() * c - v_nk[i].real() * s;
            v_k[i] = Mode(a_re + b_re, a_im + b_im);
            v_nk[i] = Mode(b_im - a_im, a_re - b_re);
        }
    }
    if (n % 2 == 0)
    {
        Mode* middle = modes + n / 2 * count;
        for (int i = 0; i < count; ++i)
            middle[i] *= std::numbers::sqrt2;
    }
}

/**
 * @brief Backward step of one axis, twice the inverse of combine_modes:
 *        V_k = W^-k (Y_k - i Y_{n-k}) and V_{-k} = W^k (Y_k + i Y_{n-k}).
 */
void separate_modes(Mode* modes, int n, int count, const Mode* w)
{
    for (int k = 1; 2 * k < n; ++k)
    {
        Mode* y_k = modes + k * count;
        Mode* y_nk = modes + (n - k) * count;
        const double c = w[k].real();
        const double s = w[k].imag();
        for (int i = 0; i < count; ++i)
        {
            const double a_re = y_k[i].real() + y_nk[i].imag();
            const double a_im = y_k[i].imag() - y_nk[i].real();
            const double b_re = y_k[i].real() - y_nk[i].imag();
            const double b_im = y_k[i].imag() + y_nk[i].real();
            y_k[i] = Mode(a_re * c + a_im * s, a_im * c - a_re * s);
            y_nk[i] = Mode(b_re * c - b_im * s, b_re * s + b_im * c);
        }
    }
    if (n % 2 == 0)
    {
        Mode* middle = modes + n / 2 * count;
        for (int i = 0; i < count; ++i)
            middle[i] *= std::numbers::sqrt2;
    }
}
}  // namespace

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MakhoulTransform::MakhoulTransform(const std::vector<int>& nx, const std::vector<bool>& sine, std::unique_ptr<RealFFT> fft)
    : rank_(static_cast<int>(nx.size())), nx_{1, 1, 1}, fft_(std::move(fft))
{
    if (rank_ < 1 || rank_ > 3 || sine.size() != nx.size())
        throw_with_line_number("MakhoulTransform takes one to three axes, each cosine or sine.");

    std::array<bool, 3> sine_axis = {false, false, false};
    for (int d = 0; d < rank_; ++d)
    {
        nx_[3 - rank_ + d] = nx[d];
        sine_axis[3 - rank_ + d] = sine[d];
    }
    half_ = nx_[2] / 2 + 1;
    total_grid_ = nx_[0] * nx_[1] * nx_[2];
    total_modes_ = nx_[0] * nx_[1] * half_;

    for (int d = 0; d < 3; ++d)
    {
        const int n = nx_[d];
        sample_[d].resize(n);
        sign_[d].resize(n);
        coefficient_[d].resize(n);
        w_[d].resize(n / 2 + 1);
        for (int j = 0; j < n; ++j)
        {
            // Even samples forward from the first place, odd samples backward from the last
            const int place = j % 2 == 0 ? j / 2 : n - 1 - j / 2;
            sample_[d][place] = j;
            sign_[d][place] = (sine_axis[d] && j % 2 == 1) ? -1.0 : 1.0;
            coefficient_[d][j] = sine_axis[d] ? n - 1 - j : j;
        }
        for (int k = 0; k <= n / 2; ++k)
            w_[d][k] = std::polar(1.0, -k * std::numbers::pi / (2.0 * n));
    }
}

//------------------------------------------------------------------------------
// Thread-local scratch
//------------------------------------------------------------------------------
std::pair<double*, std::complex<double>*> MakhoulTransform::thread_buffers() const
{
    // One array per thread. The samples and the modes start on 64-byte boundaries,
    // so a library that picks its code by alignment runs the same code on every thread.
    constexpr std::size_t line = 64 / sizeof(double);
    const std::size_t samples = (static_cast<std::size_t>(total_grid_) + line - 1) / line * line;
    const std::size_t size = samples + 2 * static_cast<std::size_t>(total_modes_) + line;
    thread_local std::vector<double> scratch;
    if (scratch.size() < size)
        scratch.resize(size);
    const auto address = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* base = scratch.data() + (64 - address % 64) % 64 / sizeof(double);
    return {base, reinterpret_cast<std::complex<double>*>(base + samples)};
}

//------------------------------------------------------------------------------
// REDFT10 / RODFT10 of one real field, its values STRIDE doubles apart
//------------------------------------------------------------------------------
template <int STRIDE>
void MakhoulTransform::forward_part(const double* in, double* out, double scale) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* sample = sample_[2].data();
    const double* sign = sign_[2].data();
    const int* coefficient = coefficient_[2].data();
    const Mode* w = w_[2].data();
    auto [samples, modes] = thread_buffers();

    double* v = samples;
    for (int i0 = 0; i0 < n0; ++i0)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            const double* x = in + static_cast<std::ptrdiff_t>(STRIDE) * (sample_[0][i0] * n1 + sample_[1][i1]) * n2;
            const double line_scale = scale * sign_[0][i0] * sign_[1][i1];
            for (int i2 = 0; i2 < n2; ++i2)
                v[i2] = line_scale * sign[i2] * x[STRIDE * sample[i2]];
            v += n2;
        }
    }

    fft_->forward(samples, modes);

    const int plane = n1 * half_;
    if (rank_ == 3)
        combine_modes(modes, n0, plane, w_[0].data());
    if (rank_ >= 2)
    {
        for (int k0 = 0; k0 < n0; ++k0)
            combine_modes(modes + k0 * plane, n1, half_, w_[1].data());
    }

    for (int k0 = 0; k0 < n0; ++k0)
    {
        for (int k1 = 0; k1 < n1; ++k1)
        {
            const Mode* y = modes + (k0 * n1 + k1) * half_;
            double* x = out + static_cast<std::ptrdiff_t>(STRIDE) * (coefficient_[0][k0] * n1 + coefficient_[1][k1]) * n2;
            x[STRIDE * coefficient[0]] = 2.0 * y[0].real();
            for (int k = 1; 2 * k < n2; ++k)
            {
                x[STRIDE * coefficient[k]] = 2.0 * (y[k].real() * w[k].real() - y[k].imag() * w[k].imag());
                x[STRIDE * coefficient[n2 - k]] = -2.0 * (y[k].real() * w[k].imag() + y[k].imag() * w[k].real());
            }
            if (n2 % 2 == 0)
                x[STRIDE * coefficient[n2 / 2]] = std::numbers::sqrt2 * y[n2 / 2].real();
        }
    }
}

//------------------------------------------------------------------------------
// REDFT01 / RODFT01 of one real field, its values STRIDE doubles apart
//------------------------------------------------------------------------------
template <int STRIDE>
void MakhoulTransform::backward_part(const double* in, double* out, double scale) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* sample = sample_[2].data();
    const double* sign = sign_[2].data();
    const int* coefficient = coefficient_[2].data();
    const Mode* w = w_[2].data();
    auto [samples, modes] = thread_buffers();

    for (int k0 = 0; k0 < n0; ++k0)
    {
        for (int k1 = 0; k1 < n1; ++k1)
        {
            Mode* y = modes + (k0 * n1 + k1) * half_;
            const double* x = in + static_cast<std::ptrdiff_t>(STRIDE) * (coefficient_[0][k0] * n1 + coefficient_[1][k1]) * n2;
            y[0] = x[STRIDE * coefficient[0]];
            for (int k = 1; 2 * k < n2; ++k)
            {
                // W^-k (X_k - i X_{n-k})
                const double a = x[STRIDE * coefficient[k]];
                const double b = x[STRIDE * coefficient[n2 - k]];
                y[k] = Mode(a * w[k].real() - b * w[k].imag(), -a * w[k].imag() - b * w[k].real());
            }
            if (n2 % 2 == 0)
                y[n2 / 2] = std::numbers::sqrt2 * x[STRIDE * coefficient[n2 / 2]];
        }
    }

    const int plane = n1 * half_;
    if (rank_ >= 2)
    {
        for (int k0 = 0; k0 < n0; ++k0)
            separate_modes(modes + k0 * plane, n1, half_, w_[1].data());
    }
    if (rank_ == 3)
        separate_modes(modes, n0, plane, w_[0].data());

    // Unnormalized, as REDFT01 and RODFT01
    fft_->backward(modes, samples);

    const double* v = samples;
    for (int i0 = 0; i0 < n0; ++i0)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            double* x = out + static_cast<std::ptrdiff_t>(STRIDE) * (sample_[0][i0] * n1 + sample_[1][i1]) * n2;
            const double line_scale = scale * sign_[0][i0] * sign_[1][i1];
            for (int i2 = 0; i2 < n2; ++i2)
                x[STRIDE * sample[i2]] = line_scale * sign[i2] * v[i2];
            v += n2;
        }
    }
}

//------------------------------------------------------------------------------
// Real and complex fields. Each part is read whole into the scratch before its
// coefficients are written, so in may be out; the real part writes only the
// even doubles, which the imaginary part does not read.
//------------------------------------------------------------------------------
void MakhoulTransform::forward(const double* in, double* out, double scale) const
{
    forward_part<1>(in, out, scale);
}

void MakhoulTransform::forward(const std::complex<double>* in, double* out, double scale) const
{
    const double* parts = reinterpret_cast<const double*>(in);
    forward_part<2>(parts, out, scale);
    forward_part<2>(parts + 1, out + 1, scale);
}

void MakhoulTransform::backward(const double* in, double* out, double scale) const
{
    backward_part<1>(in, out, scale);
}

void MakhoulTransform::backward(const double* in, std::complex<double>* out, double scale) const
{
    double* parts = reinterpret_cast<double*>(out);
    backward_part<2>(in, parts, scale);
    backward_part<2>(in + 1, parts + 1, scale);
}
