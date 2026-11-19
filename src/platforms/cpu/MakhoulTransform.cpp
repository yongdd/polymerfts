/**
 * @file MakhoulTransform.cpp
 * @brief DCT-II/III and DST-II/III of a whole grid through one periodic real FFT.
 *
 * @see MakhoulTransform.h for the algorithm
 */

#include <cmath>
#include <cstddef>
#include <numbers>
#include "AlignedScratch.h"
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

/// Coefficient j of a line x (values STRIDE doubles apart), times its factor kx[j] when K_FACTOR:
/// the product of a separate loop before the transform
template <int STRIDE, bool K_FACTOR>
inline double coefficient_value(const double* x, const double* kx, int j)
{
    if constexpr (K_FACTOR)
        return x[STRIDE * j] * kx[j];
    else
        return x[STRIDE * j];
}

/// x = value, or x += i * value for the imaginary part of a complex field
inline void store(Mode& x, double re, double im, bool imaginary)
{
    if (imaginary)
        x = Mode(x.real() - im, x.imag() + re);
    else
        x = Mode(re, im);
}

/// Hermitian part (x + conj x_partner) / 2, or that of -i x for the imaginary part of a complex field
inline Mode hermitian_part(const Mode& x, const Mode& x_partner, bool imaginary)
{
    if (imaginary)
        return Mode(0.5 * (x.imag() + x_partner.imag()), 0.5 * (x_partner.real() - x.real()));
    return Mode(0.5 * (x.real() + x_partner.real()), 0.5 * (x.imag() - x_partner.imag()));
}
}  // namespace

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MakhoulTransform::MakhoulTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc, std::unique_ptr<RealFFT> fft)
    : rank_(static_cast<int>(nx.size())), nx_{1, 1, 1}, complex_coefficients_(false), fft_(std::move(fft)),
      wall_{false, false, false}
{
    if (rank_ < 1 || rank_ > 3 || bc.size() != nx.size())
        throw_with_line_number("MakhoulTransform takes one to three axes, each periodic, cosine or sine.");

    std::array<bool, 3> sine_axis = {false, false, false};
    for (int d = 0; d < rank_; ++d)
    {
        nx_[3 - rank_ + d] = nx[d];
        wall_[3 - rank_ + d] = bc[d] != BoundaryCondition::PERIODIC;
        sine_axis[3 - rank_ + d] = bc[d] == BoundaryCondition::ABSORBING;
        complex_coefficients_ = complex_coefficients_ || bc[d] == BoundaryCondition::PERIODIC;
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
        partner_[d].resize(n);
        w_[d].resize(n / 2 + 1);
        for (int j = 0; j < n; ++j)
        {
            if (wall_[d])
            {
                // Even samples forward from the first place, odd samples backward from the last
                const int place = j % 2 == 0 ? j / 2 : n - 1 - j / 2;
                sample_[d][place] = j;
                sign_[d][place] = (sine_axis[d] && j % 2 == 1) ? -1.0 : 1.0;
                coefficient_[d][j] = sine_axis[d] ? n - 1 - j : j;
                partner_[d][j] = j;
            }
            else
            {
                sample_[d][j] = j;
                sign_[d][j] = 1.0;
                coefficient_[d][j] = j;
                partner_[d][j] = (n - j) % n;
            }
        }
        for (int k = 0; k <= n / 2; ++k)
            w_[d][k] = std::polar(1.0, -k * std::numbers::pi / (2.0 * n));
    }
}

void MakhoulTransform::check_coefficients(bool complex_coefficients) const
{
    if (complex_coefficients != complex_coefficients_)
        throw_with_line_number(complex_coefficients_
            ? "MakhoulTransform: a grid with periodic axes has complex coefficients."
            : "MakhoulTransform: a grid of walls only has real coefficients.");
}

//------------------------------------------------------------------------------
// Thread-local scratch
//------------------------------------------------------------------------------
std::pair<double*, std::complex<double>*> MakhoulTransform::thread_buffers() const
{
    // One pair per thread. The samples and the modes start on 64-byte boundaries,
    // so a library that picks its code by alignment runs the same code on every thread.
    thread_local AlignedScratch samples, modes;
    return {samples.get(total_grid_), reinterpret_cast<Mode*>(modes.get(2 * static_cast<std::size_t>(total_modes_)))};
}

//------------------------------------------------------------------------------
// Steps shared by real and complex coefficients
//------------------------------------------------------------------------------
template <int STRIDE>
void MakhoulTransform::reorder(const double* in, double* samples, double scale) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* sample = sample_[2].data();
    const double* sign = sign_[2].data();

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
}

template <int STRIDE, bool R_FACTOR>
void MakhoulTransform::restore(const double* samples, double* out, const double* r_factor, double scale) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* sample = sample_[2].data();
    const double* sign = sign_[2].data();

    const double* v = samples;
    for (int i0 = 0; i0 < n0; ++i0)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            const std::ptrdiff_t line = static_cast<std::ptrdiff_t>(sample_[0][i0] * n1 + sample_[1][i1]) * n2;
            double* x = out + STRIDE * line;
            const double line_scale = scale * sign_[0][i0] * sign_[1][i1];
            if constexpr (R_FACTOR)
            {
                // The product of a separate loop after the transform
                const double* r = r_factor + line;
                for (int i2 = 0; i2 < n2; ++i2)
                    x[STRIDE * sample[i2]] = line_scale * sign[i2] * v[i2] * r[sample[i2]];
            }
            else
            {
                for (int i2 = 0; i2 < n2; ++i2)
                    x[STRIDE * sample[i2]] = line_scale * sign[i2] * v[i2];
            }
            v += n2;
        }
    }
}

void MakhoulTransform::combine_leading_axes(Mode* modes) const
{
    const int plane = nx_[1] * half_;
    if (wall_[0])
        combine_modes(modes, nx_[0], plane, w_[0].data());
    if (wall_[1])
    {
        for (int k0 = 0; k0 < nx_[0]; ++k0)
            combine_modes(modes + k0 * plane, nx_[1], half_, w_[1].data());
    }
}

void MakhoulTransform::separate_leading_axes(Mode* modes) const
{
    const int plane = nx_[1] * half_;
    if (wall_[1])
    {
        for (int k0 = 0; k0 < nx_[0]; ++k0)
            separate_modes(modes + k0 * plane, nx_[1], half_, w_[1].data());
    }
    if (wall_[0])
        separate_modes(modes, nx_[0], plane, w_[0].data());
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
    const int* coefficient = coefficient_[2].data();
    const Mode* w = w_[2].data();
    auto [samples, modes] = thread_buffers();

    reorder<STRIDE>(in, samples, scale);
    fft_->forward(samples, modes);
    combine_leading_axes(modes);

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
template <int STRIDE, bool K_FACTOR>
void MakhoulTransform::read_coefficients(const double* in, const double* k_factor, Mode* modes) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* coefficient = coefficient_[2].data();
    const Mode* w = w_[2].data();

    for (int k0 = 0; k0 < n0; ++k0)
    {
        for (int k1 = 0; k1 < n1; ++k1)
        {
            Mode* y = modes + (k0 * n1 + k1) * half_;
            const std::ptrdiff_t line = static_cast<std::ptrdiff_t>(coefficient_[0][k0] * n1 + coefficient_[1][k1]) * n2;
            const double* x = in + STRIDE * line;
            const double* kx = K_FACTOR ? k_factor + line : nullptr;
            y[0] = coefficient_value<STRIDE, K_FACTOR>(x, kx, coefficient[0]);
            for (int k = 1; 2 * k < n2; ++k)
            {
                // W^-k (X_k - i X_{n-k})
                const double a = coefficient_value<STRIDE, K_FACTOR>(x, kx, coefficient[k]);
                const double b = coefficient_value<STRIDE, K_FACTOR>(x, kx, coefficient[n2 - k]);
                y[k] = Mode(a * w[k].real() - b * w[k].imag(), -a * w[k].imag() - b * w[k].real());
            }
            if (n2 % 2 == 0)
                y[n2 / 2] = std::numbers::sqrt2 * coefficient_value<STRIDE, K_FACTOR>(x, kx, coefficient[n2 / 2]);
        }
    }
}

template <int STRIDE>
void MakhoulTransform::backward_part(const double* in, const double* k_factor, double* out,
                                     const double* r_factor, double scale) const
{
    auto [samples, modes] = thread_buffers();
    if (k_factor == nullptr)
        read_coefficients<STRIDE, false>(in, nullptr, modes);
    else
        read_coefficients<STRIDE, true>(in, k_factor, modes);

    separate_leading_axes(modes);
    // Unnormalized, as REDFT01 and RODFT01
    fft_->backward(modes, samples);
    if (r_factor == nullptr)
        restore<STRIDE, false>(samples, out, nullptr, scale);
    else
        restore<STRIDE, true>(samples, out, r_factor, scale);
}

//------------------------------------------------------------------------------
// Complex coefficients (periodic axes among the walls) of one real field
//------------------------------------------------------------------------------
template <int STRIDE>
void MakhoulTransform::forward_complex_part(const double* in, Mode* out, double scale, bool imaginary) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* coefficient = coefficient_[2].data();
    const Mode* w = w_[2].data();
    auto [samples, modes] = thread_buffers();

    reorder<STRIDE>(in, samples, scale);
    fft_->forward(samples, modes);
    combine_leading_axes(modes);

    for (int k0 = 0; k0 < n0; ++k0)
    {
        for (int k1 = 0; k1 < n1; ++k1)
        {
            // Y(k) and Y(-k) on the periodic leading axes; the wall indices are coefficients
            const Mode* y = modes + (k0 * n1 + k1) * half_;
            const Mode* y_partner = modes + (partner_[0][k0] * n1 + partner_[1][k1]) * half_;
            Mode* x = out + static_cast<std::ptrdiff_t>(coefficient_[0][k0] * n1 + coefficient_[1][k1]) * n2;
            if (wall_[2])
            {
                store(x[coefficient[0]], 2.0 * y[0].real(), 2.0 * y[0].imag(), imaginary);
                for (int k = 1; 2 * k < n2; ++k)
                {
                    // a = W^k Y(k_p, k), b = W^-k Y(k_p, -k) = conj(W^k Y(-k_p, k))
                    const double c = w[k].real();
                    const double s = w[k].imag();
                    const double a_re = y[k].real() * c - y[k].imag() * s;
                    const double a_im = y[k].real() * s + y[k].imag() * c;
                    const double b_re = y_partner[k].real() * c - y_partner[k].imag() * s;
                    const double b_im = -(y_partner[k].real() * s + y_partner[k].imag() * c);
                    store(x[coefficient[k]], a_re + b_re, a_im + b_im, imaginary);
                    store(x[coefficient[n2 - k]], b_im - a_im, a_re - b_re, imaginary);
                }
                if (n2 % 2 == 0)
                    store(x[coefficient[n2 / 2]], std::numbers::sqrt2 * y[n2 / 2].real(),
                          std::numbers::sqrt2 * y[n2 / 2].imag(), imaginary);
            }
            else
            {
                // The upper half of a periodic last axis: Y(k_p, k) = conj Y(-k_p, n - k)
                for (int k = 0; k < half_; ++k)
                    store(x[k], y[k].real(), y[k].imag(), imaginary);
                for (int k = half_; k < n2; ++k)
                    store(x[k], y_partner[n2 - k].real(), -y_partner[n2 - k].imag(), imaginary);
            }
        }
    }
}

template <int STRIDE>
void MakhoulTransform::backward_complex_part(const Mode* in, double* out, double scale, bool imaginary) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const int* coefficient = coefficient_[2].data();
    const int* partner = partner_[2].data();
    const Mode* w = w_[2].data();
    auto [samples, modes] = thread_buffers();

    for (int k0 = 0; k0 < n0; ++k0)
    {
        for (int k1 = 0; k1 < n1; ++k1)
        {
            Mode* y = modes + (k0 * n1 + k1) * half_;
            const Mode* x = in + static_cast<std::ptrdiff_t>(coefficient_[0][k0] * n1 + coefficient_[1][k1]) * n2;
            const Mode* x_partner = in + static_cast<std::ptrdiff_t>(
                coefficient_[0][partner_[0][k0]] * n1 + coefficient_[1][partner_[1][k1]]) * n2;
            if (wall_[2])
            {
                y[0] = hermitian_part(x[coefficient[0]], x_partner[coefficient[0]], imaginary);
                for (int k = 1; 2 * k < n2; ++k)
                {
                    // W^-k (H_k - i H_{n-k})
                    const Mode a = hermitian_part(x[coefficient[k]], x_partner[coefficient[k]], imaginary);
                    const Mode b = hermitian_part(x[coefficient[n2 - k]], x_partner[coefficient[n2 - k]], imaginary);
                    const double c = w[k].real();
                    const double s = w[k].imag();
                    const double d_re = a.real() + b.imag();
                    const double d_im = a.imag() - b.real();
                    y[k] = Mode(d_re * c + d_im * s, d_im * c - d_re * s);
                }
                if (n2 % 2 == 0)
                    y[n2 / 2] = std::numbers::sqrt2 * hermitian_part(x[coefficient[n2 / 2]], x_partner[coefficient[n2 / 2]], imaginary);
            }
            else
            {
                for (int k = 0; k < half_; ++k)
                    y[k] = hermitian_part(x[k], x_partner[partner[k]], imaginary);
            }
        }
    }

    separate_leading_axes(modes);
    fft_->backward(modes, samples);
    restore<STRIDE, false>(samples, out, nullptr, scale);
}

//------------------------------------------------------------------------------
// Real and complex fields. Each part is read whole into the scratch before its
// coefficients are written, so in may be out; the real part writes only the
// even doubles, which the imaginary part does not read.
//------------------------------------------------------------------------------
void MakhoulTransform::forward(const double* in, double* out, double scale) const
{
    check_coefficients(false);
    forward_part<1>(in, out, scale);
}

void MakhoulTransform::forward(const std::complex<double>* in, double* out, double scale) const
{
    check_coefficients(false);
    const double* parts = reinterpret_cast<const double*>(in);
    forward_part<2>(parts, out, scale);
    forward_part<2>(parts + 1, out + 1, scale);
}

void MakhoulTransform::backward(const double* in, double* out, double scale) const
{
    backward_with_factors(in, nullptr, out, nullptr, scale);
}

void MakhoulTransform::backward(const double* in, std::complex<double>* out, double scale) const
{
    backward_with_factors(in, nullptr, out, nullptr, scale);
}

void MakhoulTransform::backward_with_factors(const double* in, const double* k_factor, double* out,
                                             const double* r_factor, double scale) const
{
    check_coefficients(false);
    backward_part<1>(in, k_factor, out, r_factor, scale);
}

void MakhoulTransform::backward_with_factors(const double* in, const double* k_factor, std::complex<double>* out,
                                             const std::complex<double>* r_factor, double scale) const
{
    check_coefficients(false);
    double* parts = reinterpret_cast<double*>(out);
    backward_part<2>(in, k_factor, parts, nullptr, scale);
    backward_part<2>(in + 1, k_factor, parts + 1, nullptr, scale);
    // A complex factor mixes the two parts
    if (r_factor != nullptr)
        for (int i = 0; i < total_grid_; ++i)
            out[i] *= r_factor[i];
}

//------------------------------------------------------------------------------
// Complex coefficients: C = C[Re f] + i C[Im f], and back from the Hermitian
// parts of C and of -i C. The imaginary part reads in again, so in and out are
// distinct.
//------------------------------------------------------------------------------
void MakhoulTransform::forward(const double* in, std::complex<double>* out, double scale) const
{
    check_coefficients(true);
    forward_complex_part<1>(in, out, scale, false);
}

void MakhoulTransform::forward(const std::complex<double>* in, std::complex<double>* out, double scale) const
{
    check_coefficients(true);
    const double* parts = reinterpret_cast<const double*>(in);
    forward_complex_part<2>(parts, out, scale, false);
    forward_complex_part<2>(parts + 1, out, scale, true);
}

void MakhoulTransform::backward(const std::complex<double>* in, double* out, double scale) const
{
    check_coefficients(true);
    backward_complex_part<1>(in, out, scale, false);
}

void MakhoulTransform::backward(const std::complex<double>* in, std::complex<double>* out, double scale) const
{
    check_coefficients(true);
    double* parts = reinterpret_cast<double*>(out);
    backward_complex_part<2>(in, parts, scale, false);
    backward_complex_part<2>(in, parts + 1, scale, true);
}
