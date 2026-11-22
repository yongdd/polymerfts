/**
 * @file CpuFourierTransform.cpp
 * @brief Fourier multipliers with the FFTW or MKL transforms.
 */

#include <algorithm>

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
        // The DFT along the periodic axes and the series along the walls, through one real
        // FFT of the grid (a mixed box has at least two axes)
        mixed_fft_ = std::make_unique<MakhoulTransform>(nx, bc, createRealFFT(nx, backend));
    }
    else
    {
        fft_ = createFFT<T>(nx, bc, backend);
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
    // The forward transforms keep their input, so the caller's fields are not modified
    #pragma omp parallel for
    for (int f = 0; f < n_fields; ++f)
        errors.run([&] {
            if (this->mixed_)
                mixed_fft_->forward(fields + f * n_grid, reinterpret_cast<std::complex<double>*>(coefficients + f * stride),
                                    this->mixed_forward_scale_);
            else
                fft_->forward(const_cast<T*>(fields + f * n_grid), coefficients + f * stride);
        });
    errors.rethrow();
}

template <typename T>
void CpuFourierTransform<T>::backward_impl(const double* coefficients, int n_fields, T* fields)
{
    // The inverse transforms keep the coefficients: MakhoulTransform (mixed boxes and walls) and
    // FFTW's complex-to-real transform, which overwrites its input, read them into thread-local
    // scratch, and MKL and FFTW's complex-to-complex transform run out of place on them
    backward_fields(const_cast<double*>(coefficients), n_fields, fields, false);
}

template <typename T>
void CpuFourierTransform<T>::backward_hermitian_impl(const double* coefficients, int n_fields, T* fields)
{
    // A copy of the coefficients, made Hermitian, which the transforms may then overwrite
    const long long n = n_fields * this->coefficient_stride();
    double* copy = coefficients_out_.get(n);
    std::copy_n(coefficients, n, copy);
    this->make_hermitian(reinterpret_cast<std::complex<double>*>(copy), n_fields);
    backward_fields(copy, n_fields, fields, true);
}

template <typename T>
void CpuFourierTransform<T>::backward_fields(double* coefficients, int n_fields, T* fields, bool give_up)
{
    const long long n_grid = this->n_grid_;
    const long long stride = this->coefficient_stride();
    ParallelErrors errors;
    // A real field of a mixed box is the real part of the inverse
    #pragma omp parallel for
    for (int f = 0; f < n_fields; ++f)
        errors.run([&] {
            if (this->mixed_)
                mixed_fft_->backward(reinterpret_cast<const std::complex<double>*>(coefficients + f * stride),
                                     fields + f * n_grid, 1.0 / static_cast<double>(n_grid));
            else if (give_up)
                fft_->backward_with_factors(coefficients + f * stride, nullptr, fields + f * n_grid, nullptr);
            else
                fft_->backward(coefficients + f * stride, fields + f * n_grid);
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
    double* coefficients_in = coefficients_in_.get(max_in * stride);
    double* coefficients_out = coefficients_out_.get(n_out * stride);

    for (size_t t = 0; t < terms.size(); ++t)
    {
        const FourierTerm<T>& term = terms[t];
        const auto* m = static_cast<const CpuFourierMultiplier<T>*>(term.multiplier);
        forward_impl(term.fields, term.n_fields, coefficients_in);
        if (this->complex_coefficients_)
        {
            multiply(reinterpret_cast<std::complex<double>*>(coefficients_out), m->complex_values.data(),
                     reinterpret_cast<const std::complex<double>*>(coefficients_in),
                     n_out, m->get_n_in(), m->is_diagonal(), n, t > 0);
        }
        else
        {
            multiply(coefficients_out, m->real_values.data(), coefficients_in,
                     n_out, m->get_n_in(), m->is_diagonal(), n, t > 0);
        }
    }
    backward_fields(coefficients_out, n_out, out, true);
}

template class CpuFourierTransform<double>;
template class CpuFourierTransform<std::complex<double>>;
