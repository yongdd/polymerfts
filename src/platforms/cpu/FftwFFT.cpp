/**
 * @file FftwFFT.cpp
 * @brief FFTW3 periodic FFT (r2c/c2r, c2c) and the real FFT under MakhoulTransform.
 *
 * Backward transforms of FftwFFT include 1/N scaling so backward(forward(x)) = x.
 *
 * @see FftwFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "FftwFFT.h"

namespace
{
/// FFTW's new-array execute functions take arrays of the alignment of the planning
/// arrays; those came from fftw_malloc, for which fftw_alignment_of() is 0.
bool has_plan_alignment(const void* p)
{
    return fftw_alignment_of(static_cast<double*>(const_cast<void*>(p))) == 0;
}

bool overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x < y + b_bytes && y < x + a_bytes;
}

/// Thread-local scratch for a field and its modes of the periodic transforms, each
/// from a 64-byte boundary, which has the alignment of fftw_malloc as well
std::pair<double*, double*> thread_scratch(std::size_t field_doubles, std::size_t mode_doubles)
{
    constexpr std::size_t line = 64 / sizeof(double);
    const std::size_t field = (field_doubles + line - 1) / line * line;
    const std::size_t size = field + mode_doubles + line;
    thread_local std::vector<double> scratch;
    if (scratch.size() < size)
        scratch.resize(size);
    const auto address = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* base = scratch.data() + (64 - address % 64) % 64 / sizeof(double);
    return {base, base + field};
}
}  // namespace

//------------------------------------------------------------------------------
// The real FFT under MakhoulTransform
//------------------------------------------------------------------------------
FftwRealFFT::FftwRealFFT(const std::vector<int>& nx)
{
    int total = 1;
    for (int n : nx)
        total *= n;
    const int modes = total / nx.back() * (nx.back() / 2 + 1);

    // Planning arrays only: the plans run on other arrays of the same alignment
    double* samples = fftw_alloc_real(total);
    fftw_complex* spectrum = fftw_alloc_complex(modes);
    if (samples != nullptr && spectrum != nullptr)
    {
        const int rank = static_cast<int>(nx.size());
        forward_ = fftw_plan_dft_r2c(rank, nx.data(), samples, spectrum, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        backward_ = fftw_plan_dft_c2r(rank, nx.data(), spectrum, samples, FFTW_MEASURE);
    }
    if (samples != nullptr)
        fftw_free(samples);
    if (spectrum != nullptr)
        fftw_free(spectrum);
    if (forward_ == nullptr || backward_ == nullptr)
    {
        release();
        throw_with_line_number("FFTW plan creation failed");
    }
}

FftwRealFFT::~FftwRealFFT()
{
    release();
}

void FftwRealFFT::release()
{
    if (forward_ != nullptr)
        fftw_destroy_plan(forward_);
    if (backward_ != nullptr)
        fftw_destroy_plan(backward_);
    forward_ = nullptr;
    backward_ = nullptr;
}

void FftwRealFFT::forward(double* samples, std::complex<double>* modes) const
{
    fftw_execute_dft_r2c(forward_, samples, reinterpret_cast<fftw_complex*>(modes));
}

void FftwRealFFT::backward(std::complex<double>* modes, double* samples) const
{
    fftw_execute_dft_c2r(backward_, reinterpret_cast<fftw_complex*>(modes), samples);
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::FftwFFT(std::array<int, DIM> nx)
    : nx_(nx)
{
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // r2c: the half spectrum of the last axis (Hermitian symmetry); c2c: the full grid
    if constexpr (std::is_same<T, double>::value)
        total_complex_grid_ = total_grid_ / nx_[DIM - 1] * (nx_[DIM - 1] / 2 + 1);
    else
        total_complex_grid_ = total_grid_;

    try
    {
        initPeriodicFFT();
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
template <typename T, int DIM>
FftwFFT<T, DIM>::~FftwFFT()
{
    release();
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::release()
{
    if (plan_forward_ != nullptr)
        fftw_destroy_plan(plan_forward_);
    if (plan_backward_ != nullptr)
        fftw_destroy_plan(plan_backward_);
    plan_forward_ = nullptr;
    plan_backward_ = nullptr;
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (FFTW r2c/c2r, or c2c for complex fields)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::initPeriodicFFT()
{
    // Planning arrays only: the plans run on other arrays of the same alignment
    // (see the class description). A complex field (c2c) takes two doubles per point.
    constexpr std::size_t doubles_per_point = sizeof(T) / sizeof(double);
    double* field = fftw_alloc_real(doubles_per_point * total_grid_);
    fftw_complex* modes = fftw_alloc_complex(total_complex_grid_);

    if (field != nullptr && modes != nullptr)
    {
        // FFTW_MEASURE runs actual transforms to find the fastest algorithm: slower
        // planning, faster execution. Out of place, the forward plans keep their input.
        const unsigned int flags = FFTW_MEASURE;
        if constexpr (std::is_same<T, double>::value)
        {
            if (DIM == 1)
            {
                plan_forward_ = fftw_plan_dft_r2c_1d(nx_[0], field, modes, flags);
                plan_backward_ = fftw_plan_dft_c2r_1d(nx_[0], modes, field, flags);
            }
            else if (DIM == 2)
            {
                plan_forward_ = fftw_plan_dft_r2c_2d(nx_[0], nx_[1], field, modes, flags);
                plan_backward_ = fftw_plan_dft_c2r_2d(nx_[0], nx_[1], modes, field, flags);
            }
            else // DIM == 3
            {
                plan_forward_ = fftw_plan_dft_r2c_3d(nx_[0], nx_[1], nx_[2], field, modes, flags);
                plan_backward_ = fftw_plan_dft_c2r_3d(nx_[0], nx_[1], nx_[2], modes, field, flags);
            }
        }
        else // Complex input (c2c transform)
        {
            int n[DIM];
            for (int d = 0; d < DIM; ++d)
                n[d] = nx_[d];

            plan_forward_ = fftw_plan_dft(DIM, n, reinterpret_cast<fftw_complex*>(field), modes, FFTW_FORWARD, flags);
            plan_backward_ = fftw_plan_dft(DIM, n, modes, reinterpret_cast<fftw_complex*>(field), FFTW_BACKWARD, flags);
        }
    }
    const bool allocated = field != nullptr && modes != nullptr;
    if (field != nullptr)
        fftw_free(field);
    if (modes != nullptr)
        fftw_free(modes);

    if (!allocated)
        throw_with_line_number("FFTW memory allocation failed");
    if (plan_forward_ == nullptr || plan_backward_ == nullptr)
        throw_with_line_number("FFTW plan creation failed");
}

//------------------------------------------------------------------------------
// Periodic transforms on the caller's arrays or on thread-local scratch
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::periodic_forward(const T* field, std::complex<double>* modes) const
{
    constexpr std::size_t doubles_per_point = sizeof(T) / sizeof(double);
    const std::size_t field_bytes = sizeof(T) * total_grid_;
    const std::size_t modes_bytes = sizeof(std::complex<double>) * total_complex_grid_;

    // The forward plans keep their input: it is copied only when FFTW cannot run
    // on it or when it overlaps the output
    const bool field_direct = has_plan_alignment(field) && !overlap(field, field_bytes, modes, modes_bytes);
    const bool modes_direct = has_plan_alignment(modes);
    double* field_scratch = nullptr;
    double* modes_scratch = nullptr;
    if (!field_direct || !modes_direct)
        std::tie(field_scratch, modes_scratch) = thread_scratch(doubles_per_point * total_grid_, 2 * static_cast<std::size_t>(total_complex_grid_));

    T* in = const_cast<T*>(field);
    if (!field_direct)
    {
        in = reinterpret_cast<T*>(field_scratch);
        std::copy_n(field, total_grid_, in);
    }
    std::complex<double>* out = modes_direct ? modes : reinterpret_cast<std::complex<double>*>(modes_scratch);

    if constexpr (std::is_same<T, double>::value)
        fftw_execute_dft_r2c(plan_forward_, in, reinterpret_cast<fftw_complex*>(out));
    else
        fftw_execute_dft(plan_forward_, reinterpret_cast<fftw_complex*>(in), reinterpret_cast<fftw_complex*>(out));

    if (!modes_direct)
        std::copy_n(out, total_complex_grid_, modes);
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::periodic_backward(std::complex<double>* modes, const double* k_factor, T* field,
                                        const T* r_factor, bool keep_modes) const
{
    constexpr bool real_field = std::is_same<T, double>::value;
    constexpr std::size_t doubles_per_point = sizeof(T) / sizeof(double);
    const std::size_t field_bytes = sizeof(T) * total_grid_;
    const std::size_t modes_bytes = sizeof(std::complex<double>) * total_complex_grid_;
    const double scale = 1.0 / static_cast<double>(total_grid_);

    // The complex-to-real transform overwrites its input, which therefore goes through
    // the scratch unless the caller gives it up; the complex-to-complex transform keeps
    // its input. A k_factor is multiplied into the scratch.
    const bool may_run_on_modes = (!real_field || !keep_modes) && k_factor == nullptr;
    const bool modes_direct = may_run_on_modes && has_plan_alignment(modes) && !overlap(modes, modes_bytes, field, field_bytes);
    const bool field_direct = has_plan_alignment(field);
    double* field_scratch = nullptr;
    double* modes_scratch = nullptr;
    if (!modes_direct || !field_direct)
        std::tie(field_scratch, modes_scratch) = thread_scratch(doubles_per_point * total_grid_, 2 * static_cast<std::size_t>(total_complex_grid_));

    std::complex<double>* in = modes;
    if (!modes_direct)
    {
        in = reinterpret_cast<std::complex<double>*>(modes_scratch);
        if (k_factor != nullptr)
            for (int i = 0; i < total_complex_grid_; ++i)
                in[i] = modes[i] * k_factor[i];
        else
            std::copy_n(modes, total_complex_grid_, in);
    }
    T* out = field_direct ? field : reinterpret_cast<T*>(field_scratch);

    if constexpr (real_field)
        fftw_execute_dft_c2r(plan_backward_, reinterpret_cast<fftw_complex*>(in), out);
    else
        fftw_execute_dft(plan_backward_, reinterpret_cast<fftw_complex*>(in), reinterpret_cast<fftw_complex*>(out));

    // The 1/N of the FFT<T> interface and r_factor (in place when FFTW wrote into the field)
    if (r_factor != nullptr)
        for (int i = 0; i < total_grid_; ++i)
            field[i] = out[i] * scale * r_factor[i];
    else
        for (int i = 0; i < total_grid_; ++i)
            field[i] = out[i] * scale;
}

//------------------------------------------------------------------------------
// FFT<T> interface: complex coefficients, or the same as interleaved doubles
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    periodic_forward(rdata, cdata);
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    periodic_backward(cdata, nullptr, rdata, nullptr, true);
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    periodic_forward(rdata, reinterpret_cast<std::complex<double>*>(cdata));
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    periodic_backward(reinterpret_cast<std::complex<double>*>(cdata), nullptr, rdata, nullptr, true);
}

template <typename T, int DIM>
void FftwFFT<T, DIM>::backward_with_factors(double *cdata, const double *k_factor, T *rdata, const T *r_factor)
{
    periodic_backward(reinterpret_cast<std::complex<double>*>(cdata), k_factor, rdata, r_factor, false);
}

// Explicit template instantiations
template class FftwFFT<double, 1>;
template class FftwFFT<double, 2>;
template class FftwFFT<double, 3>;
template class FftwFFT<std::complex<double>, 1>;
template class FftwFFT<std::complex<double>, 2>;
template class FftwFFT<std::complex<double>, 3>;
