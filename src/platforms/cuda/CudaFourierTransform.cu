/**
 * @file CudaFourierTransform.cu
 * @brief Fourier multipliers with cuFFT and the CUDA cosine/sine transforms.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cuComplex.h>

#include "CudaDeviceOps.cuh"
#include "CudaFourierTransform.h"
#include "CudaTypeDispatch.h"

namespace
{
/// out_i (+)= Σ_j M_ij · in_j at every coefficient (diagonal: out_i (+)= M · in_i).
/// The n coefficients of a field are `pitch` apart in `in` and `out`; m is packed.
template <typename C>
__global__ void ker_fourier_multiply(C* out, const C* m, const C* in, int n_out, int n_in,
                                     bool diagonal, long long n, long long pitch, bool accumulate)
{
    for (long long k = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; k < n;
         k += static_cast<long long>(blockDim.x) * gridDim.x)
    {
        for (int i = 0; i < n_out; ++i)
        {
            C sum;
            if (diagonal)
            {
                sum = multiplied(m[k], in[i * pitch + k]);
            }
            else
            {
                sum = zero_of(sum);
                for (int j = 0; j < n_in; ++j)
                    sum = added(sum, multiplied(m[(static_cast<long long>(i) * n_in + j) * n + k], in[j * pitch + k]));
            }
            out[i * pitch + k] = accumulate ? added(out[i * pitch + k], sum) : sum;
        }
    }
}

/// x *= factor at every element (the normalization of a periodic box's inverse cuFFT)
__global__ void ker_fourier_scale(double* x, double factor, long long n)
{
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(blockDim.x) * gridDim.x)
        x[i] *= factor;
}

/// Allocate at least `count` elements, keeping a large enough buffer
template <typename X>
void reserve(CudaDeviceBuffer<X>& buffer, long long count)
{
    if (static_cast<long long>(buffer.size()) < count)
        buffer.allocate(count);
}
}  // namespace

template <typename T>
CudaFourierTransform<T>::CudaFourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc)
    : FourierTransform<T>(cb, bc)
{
    std::vector<int> nx = cb->get_nx();
    if (this->dim_ < 1 || this->dim_ > 3)
        throw_with_line_number("A Fourier transform supports 1D, 2D or 3D grids, got dim = "
                               + std::to_string(this->dim_) + ".");
    if (this->periodic_)
    {
        // The unnormalized FFT of the grid, both directions; backward() divides by the grid
        CUFFT_CHECK(plan_forward_.create(cufftPlanMany, this->dim_, nx.data(), nullptr, 1, 0, nullptr, 1, 0,
                                         CufftPlanType<T>::forward, 1));
        CUFFT_CHECK(plan_backward_.create(cufftPlanMany, this->dim_, nx.data(), nullptr, 1, 0, nullptr, 1, 0,
                                          CufftPlanType<T>::backward, 1));
    }
    else
    {
        // The series along the walls and the DFT along the periodic axes, through one cuFFT of the grid
        series_ = std::make_unique<CudaMakhoulTransform<T>>(nx, bc);
    }
    // cuFFT needs 16-byte aligned real data: give every field an even number of doubles
    const long long doubles_per_value = static_cast<long long>(sizeof(T) / sizeof(double));
    const long long field_doubles = this->n_grid_ * doubles_per_value;
    field_pitch_ = (field_doubles + field_doubles % 2) / doubles_per_value;
    coefficient_pitch_ = this->coefficient_stride() + this->coefficient_stride() % 2;
    CUDA_CHECK(stream_.create());
}

template <typename T>
std::unique_ptr<FourierMultiplier<T>> CudaFourierTransform<T>::store_multiplier(
    const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal)
{
    auto multiplier = std::make_unique<CudaFourierMultiplier<T>>(this, n_out, n_in, diagonal);
    if (this->complex_coefficients_)
    {
        multiplier->complex_values.allocate(n_values);
        CUDA_CHECK(cudaMemcpy(multiplier->complex_values, values, sizeof(cuDoubleComplex) * n_values,
                              cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<double> real_values(n_values);
        for (long long i = 0; i < n_values; ++i)
            real_values[i] = values[i].real();
        multiplier->real_values.allocate(n_values);
        CUDA_CHECK(cudaMemcpy(multiplier->real_values, real_values.data(), sizeof(double) * n_values,
                              cudaMemcpyHostToDevice));
    }
    return multiplier;
}

template <typename T>
void CudaFourierTransform<T>::upload_fields(const T* fields, int n_fields)
{
    // The fields are packed on the host and field_pitch_ apart on the device
    const long long n_grid = this->n_grid_;
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
        CUDA_CHECK(cudaMemcpyAsync(d_fields + f * field_pitch_, fields + f * n_grid, sizeof(T) * n_grid,
                                   cudaMemcpyHostToDevice, stream_));
}

template <typename T>
void CudaFourierTransform<T>::download_fields(T* fields, int n_fields)
{
    const long long n_grid = this->n_grid_;
    const T* d_fields = reinterpret_cast<const T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
        CUDA_CHECK(cudaMemcpyAsync(fields + f * n_grid, d_fields + f * field_pitch_, sizeof(T) * n_grid,
                                   cudaMemcpyDeviceToHost, stream_));
}

template <typename T>
void CudaFourierTransform<T>::copy_coefficients(double* destination, const double* source, int n_fields,
                                                bool to_device)
{
    // Packed on the host, coefficient_pitch_ apart on the device
    const long long stride = this->coefficient_stride();
    for (int f = 0; f < n_fields; ++f)
    {
        if (to_device)
            CUDA_CHECK(cudaMemcpyAsync(destination + f * coefficient_pitch_, source + f * stride,
                                       sizeof(double) * stride, cudaMemcpyHostToDevice, stream_));
        else
            CUDA_CHECK(cudaMemcpyAsync(destination + f * stride, source + f * coefficient_pitch_,
                                       sizeof(double) * stride, cudaMemcpyDeviceToHost, stream_));
    }
}

template <typename T>
void CudaFourierTransform<T>::transform_forward(int n_fields, double* d_coefficients)
{
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
    {
        // The coefficients of a field start an even number of doubles apart: 16-byte aligned
        DeviceT* d_field = reinterpret_cast<DeviceT*>(d_fields + f * field_pitch_);
        double* d_field_coefficients = d_coefficients + f * coefficient_pitch_;
        if (this->periodic_)
        {
            cuDoubleComplex* d_spectrum = reinterpret_cast<cuDoubleComplex*>(d_field_coefficients);
            CUFFT_CHECK(cufftSetStream(plan_forward_, stream_));
            CUFFT_CHECK(cufft_exec_forward(plan_forward_, d_field, d_spectrum));
        }
        else
        {
            // The series are half of FFTW's REDFT10/RODFT10 per wall axis (mixed_forward_scale_
            // holds 1/2 per wall axis, also in a box of walls only)
            series_->forward(d_field, d_field_coefficients, this->mixed_forward_scale_, stream_);
        }
    }
}

template <typename T>
void CudaFourierTransform<T>::transform_backward(double* d_coefficients, int n_fields)
{
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    const double normalization = 1.0 / static_cast<double>(this->n_grid_);
    for (int f = 0; f < n_fields; ++f)
    {
        DeviceT* d_field = reinterpret_cast<DeviceT*>(d_fields + f * field_pitch_);
        double* d_field_coefficients = d_coefficients + f * coefficient_pitch_;
        if (this->periodic_)
        {
            // Z2D may overwrite its input even out of place; the coefficients are scratch
            // (see the declaration), so they are transformed without a copy
            cuDoubleComplex* d_spectrum = reinterpret_cast<cuDoubleComplex*>(d_field_coefficients);
            CUFFT_CHECK(cufftSetStream(plan_backward_, stream_));
            CUFFT_CHECK(cufft_exec_backward(plan_backward_, d_spectrum, d_field));
            const long long n_doubles = static_cast<long long>(sizeof(T) / sizeof(double)) * this->n_grid_;
            const int n_threads = CudaCommon::get_instance().get_n_threads();
            ker_fourier_scale<<<grid_blocks(n_doubles, n_threads), n_threads, 0, stream_>>>(
                reinterpret_cast<double*>(d_field), normalization, n_doubles);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            // A real field with periodic axes is the real part of the inverse
            series_->backward(d_field_coefficients, nullptr, d_field, normalization, stream_);
        }
    }
}

template <typename T>
void CudaFourierTransform<T>::forward_impl(const T* fields, int n_fields, double* coefficients)
{
    reserve(d_fields_, n_fields * field_pitch_);
    reserve(d_coefficients_in_, n_fields * coefficient_pitch_);
    upload_fields(fields, n_fields);
    transform_forward(n_fields, d_coefficients_in_);
    copy_coefficients(coefficients, d_coefficients_in_, n_fields, false);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename T>
void CudaFourierTransform<T>::backward_impl(const double* coefficients, int n_fields, T* fields)
{
    reserve(d_coefficients_out_, n_fields * coefficient_pitch_);
    reserve(d_fields_, n_fields * field_pitch_);
    copy_coefficients(d_coefficients_out_, coefficients, n_fields, true);
    transform_backward(d_coefficients_out_, n_fields);
    download_fields(fields, n_fields);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename T>
void CudaFourierTransform<T>::apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out)
{
    const long long n = this->n_coefficients_;
    // Pitch in coefficients: complex coefficients take two doubles each
    const long long pitch = this->complex_coefficients_ ? coefficient_pitch_ / 2 : coefficient_pitch_;
    int max_in = 0;
    for (const FourierTerm<T>& term : terms)
        max_in = std::max(max_in, term.n_fields);
    // Reserved before any work is queued: a reallocation frees memory in use
    reserve(d_fields_, std::max(max_in, n_out) * field_pitch_);
    reserve(d_coefficients_in_, max_in * coefficient_pitch_);
    reserve(d_coefficients_out_, n_out * coefficient_pitch_);

    const int n_threads = CudaCommon::get_instance().get_n_threads();
    const int n_blocks = grid_blocks(n, n_threads);
    for (size_t t = 0; t < terms.size(); ++t)
    {
        const FourierTerm<T>& term = terms[t];
        const auto* m = static_cast<const CudaFourierMultiplier<T>*>(term.multiplier);
        upload_fields(term.fields, term.n_fields);
        transform_forward(term.n_fields, d_coefficients_in_);
        if (this->complex_coefficients_)
        {
            ker_fourier_multiply<<<n_blocks, n_threads, 0, stream_>>>(
                reinterpret_cast<cuDoubleComplex*>(d_coefficients_out_.get()), m->complex_values.get(),
                reinterpret_cast<const cuDoubleComplex*>(d_coefficients_in_.get()),
                n_out, m->get_n_in(), m->is_diagonal(), n, pitch, t > 0);
        }
        else
        {
            ker_fourier_multiply<<<n_blocks, n_threads, 0, stream_>>>(
                d_coefficients_out_.get(), m->real_values.get(), d_coefficients_in_.get(),
                n_out, m->get_n_in(), m->is_diagonal(), n, pitch, t > 0);
        }
        CUDA_CHECK_LAUNCH();
    }
    transform_backward(d_coefficients_out_, n_out);
    download_fields(out, n_out);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template class CudaFourierTransform<double>;
template class CudaFourierTransform<std::complex<double>>;
