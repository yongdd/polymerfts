/**
 * @file CpuFourierTransform.cpp
 * @brief Fourier multipliers with the FFTW or MKL transforms.
 */

#include <algorithm>
#include <array>
#include <type_traits>

#include "ComputationCommon.h"   // ParallelErrors
#include "CpuFourierTransform.h"
#include "FFTFactory.h"

namespace
{
/// out_i (+)= Σ_j M_ij · in_j at every coefficient (diagonal: out_i (+)= M · in_i)
template <typename C>
void multiply(C* out, const C* m, const C* in, int n_out, int n_in, bool diagonal,
              long long n, bool accumulate)
{
    #pragma omp parallel for
    for (long long k = 0; k < n; ++k)
    {
        for (int i = 0; i < n_out; ++i)
        {
            C sum;
            if (diagonal)
            {
                sum = m[k] * in[i * n + k];
            }
            else
            {
                sum = C(0.0);
                for (int j = 0; j < n_in; ++j)
                    sum += m[(static_cast<long long>(i) * n_in + j) * n + k] * in[j * n + k];
            }
            out[i * n + k] = accumulate ? out[i * n + k] + sum : sum;
        }
    }
}
}  // namespace

template <typename T>
CpuFourierTransform<T>::CpuFourierTransform(
    ComputationBox<T>* cb, std::vector<BoundaryCondition> bc, FFTBackend backend)
    : FourierTransform<T>(cb, bc)
{
    const std::vector<int> nx = cb->get_nx();
    if (this->mixed_)
    {
        // Complex FFT of the periodic mirror image (a mixed box has at least two axes)
        const auto& m = this->mirror_axes_;
        if (this->dim_ == 3)
            mirror_fft_ = createFFT<std::complex<double>, 3>({m[0].n_extended, m[1].n_extended, m[2].n_extended}, backend);
        else
            mirror_fft_ = createFFT<std::complex<double>, 2>({m[1].n_extended, m[2].n_extended}, backend);
    }
    else if (this->dim_ == 3)
        fft_ = createFFT<T, 3>({nx[0], nx[1], nx[2]}, {bc[0], bc[1], bc[2]}, backend);
    else if (this->dim_ == 2)
        fft_ = createFFT<T, 2>({nx[0], nx[1]}, {bc[0], bc[1]}, backend);
    else if (this->dim_ == 1)
        fft_ = createFFT<T, 1>({nx[0]}, {bc[0]}, backend);
    else
        throw_with_line_number("A Fourier transform supports 1D, 2D or 3D grids, got dim = "
                               + std::to_string(this->dim_) + ".");
}

template <typename T>
void CpuFourierTransform<T>::mirror_forward(const T* field, std::complex<double>* coefficients) const
{
    const auto& a = this->mirror_axes_;
    const long long e1 = a[1].n_extended, e2 = a[2].n_extended;
    const long long n1 = a[1].n, n2 = a[2].n;
    // Reused per thread: the fields and propagators are transformed concurrently
    thread_local std::vector<std::complex<double>> image, modes;
    image.resize(this->n_extended_);
    modes.resize(this->n_extended_);
    for (int i = 0; i < a[0].n_extended; ++i)
        for (int j = 0; j < e1; ++j)
            for (int k = 0; k < e2; ++k)
            {
                const double sign = a[0].extend_sign[i] * a[1].extend_sign[j] * a[2].extend_sign[k];
                const long long source = (a[0].extend_source[i] * n1 + a[1].extend_source[j]) * n2 + a[2].extend_source[k];
                image[(i * e1 + j) * e2 + k] = sign * std::complex<double>(field[source]);
            }
    mirror_fft_->forward(image.data(), modes.data());
    for (int i = 0; i < a[0].n; ++i)
        for (int j = 0; j < n1; ++j)
            for (int k = 0; k < n2; ++k)
            {
                const long long mode = (a[0].extract_source[i] * e1 + a[1].extract_source[j]) * e2 + a[2].extract_source[k];
                coefficients[(i * n1 + j) * n2 + k] =
                    a[0].extract_factor[i] * a[1].extract_factor[j] * a[2].extract_factor[k] * modes[mode];
            }
}

template <typename T>
void CpuFourierTransform<T>::mirror_backward(const std::complex<double>* coefficients, T* field) const
{
    const auto& a = this->mirror_axes_;
    const long long e1 = a[1].n_extended, e2 = a[2].n_extended;
    const long long n1 = a[1].n, n2 = a[2].n;
    // Reused per thread: the fields and propagators are transformed concurrently
    thread_local std::vector<std::complex<double>> image, modes;
    image.resize(this->n_extended_);
    modes.resize(this->n_extended_);
    for (int i = 0; i < a[0].n_extended; ++i)
        for (int j = 0; j < e1; ++j)
            for (int k = 0; k < e2; ++k)
            {
                const int si = a[0].fill_source[i], sj = a[1].fill_source[j], sk = a[2].fill_source[k];
                modes[(i * e1 + j) * e2 + k] = (si < 0 || sj < 0 || sk < 0) ? std::complex<double>(0.0)
                    : a[0].fill_factor[i] * a[1].fill_factor[j] * a[2].fill_factor[k] * coefficients[(si * n1 + sj) * n2 + sk];
            }
    mirror_fft_->backward(modes.data(), image.data());
    // The field is the first N points of every extended axis
    for (int i = 0; i < a[0].n; ++i)
        for (int j = 0; j < n1; ++j)
            for (int k = 0; k < n2; ++k)
            {
                const std::complex<double> value = image[(i * e1 + j) * e2 + k];
                if constexpr (std::is_same_v<T, double>)
                    field[(i * n1 + j) * n2 + k] = value.real();
                else
                    field[(i * n1 + j) * n2 + k] = value;
            }
}

template <typename T>
std::unique_ptr<FourierMultiplier<T>> CpuFourierTransform<T>::store_multiplier(
    const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal)
{
    auto multiplier = std::make_unique<CpuFourierMultiplier<T>>(this, n_out, n_in, diagonal);
    if (this->complex_coefficients_)
    {
        multiplier->complex_values.assign(values, values + n_values);
    }
    else
    {
        multiplier->real_values.resize(n_values);
        for (long long i = 0; i < n_values; ++i)
            multiplier->real_values[i] = values[i].real();
    }
    return multiplier;
}

template <typename T>
void CpuFourierTransform<T>::forward_impl(const T* fields, int n_fields, double* coefficients)
{
    const long long n_grid = this->n_grid_;
    const long long stride = this->coefficient_stride();
    ParallelErrors errors;
    // The transforms copy their input into thread-local buffers, so the
    // caller's fields are not modified
    #pragma omp parallel for
    for (int f = 0; f < n_fields; ++f)
        errors.run([&] {
            if (this->mixed_)
                mirror_forward(fields + f * n_grid, reinterpret_cast<std::complex<double>*>(coefficients + f * stride));
            else
                fft_->forward(const_cast<T*>(fields + f * n_grid), coefficients + f * stride);
        });
    errors.rethrow();
}

template <typename T>
void CpuFourierTransform<T>::backward_impl(const double* coefficients, int n_fields, T* fields)
{
    const long long n_grid = this->n_grid_;
    const long long stride = this->coefficient_stride();
    ParallelErrors errors;
    // The inverse transforms work on thread-local copies of their input
    #pragma omp parallel for
    for (int f = 0; f < n_fields; ++f)
        errors.run([&] {
            if (this->mixed_)
                mirror_backward(reinterpret_cast<const std::complex<double>*>(coefficients + f * stride), fields + f * n_grid);
            else
                fft_->backward(const_cast<double*>(coefficients + f * stride), fields + f * n_grid);
        });
    errors.rethrow();
}

template <typename T>
void CpuFourierTransform<T>::apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out)
{
    const long long n = this->n_coefficients_;
    const long long stride = this->coefficient_stride();
    int max_in = 0;
    for (const FourierTerm<T>& term : terms)
        max_in = std::max(max_in, term.n_fields);
    coefficients_in_.resize(max_in * stride);
    coefficients_out_.resize(n_out * stride);

    for (size_t t = 0; t < terms.size(); ++t)
    {
        const FourierTerm<T>& term = terms[t];
        const auto* m = static_cast<const CpuFourierMultiplier<T>*>(term.multiplier);
        forward_impl(term.fields, term.n_fields, coefficients_in_.data());
        if (this->complex_coefficients_)
        {
            multiply(reinterpret_cast<std::complex<double>*>(coefficients_out_.data()), m->complex_values.data(),
                     reinterpret_cast<const std::complex<double>*>(coefficients_in_.data()),
                     n_out, m->get_n_in(), m->is_diagonal(), n, t > 0);
        }
        else
        {
            multiply(coefficients_out_.data(), m->real_values.data(), coefficients_in_.data(),
                     n_out, m->get_n_in(), m->is_diagonal(), n, t > 0);
        }
    }
    backward_impl(coefficients_out_.data(), n_out, out);
}

template class CpuFourierTransform<double>;
template class CpuFourierTransform<std::complex<double>>;
