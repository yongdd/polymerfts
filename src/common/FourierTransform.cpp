/**
 * @file FourierTransform.cpp
 * @brief Coefficient layout, |k|² and argument checks shared by the platforms.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>
#include <type_traits>

#include "MinimalFormAlias.h"
#include "FourierTransform.h"

template <typename T>
FourierTransform<T>::FourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc)
    : cb_(cb), bc_(bc)
{
    if (cb == nullptr)
        throw_with_line_number("A Fourier transform needs a computation box.");
    dim_ = cb->get_dim();
    if (static_cast<int>(bc_.size()) != dim_)
        throw_with_line_number("A Fourier transform needs one transform per axis: got "
            + std::to_string(bc_.size()) + " for " + std::to_string(dim_) + " axes.");

    const int n_periodic = static_cast<int>(std::count(bc_.begin(), bc_.end(), BoundaryCondition::PERIODIC));
    periodic_ = n_periodic == dim_;
    mixed_ = n_periodic > 0 && !periodic_;
    for (BoundaryCondition b : bc_)
        if (b != BoundaryCondition::PERIODIC)
            mixed_forward_scale_ *= 0.5;
    if (!periodic_ && !cb->is_orthogonal())
        throw_with_line_number("The cosine and sine series need an orthogonal box.");

    const std::vector<int> nx = cb->get_nx();
    n_grid_ = cb->get_total_grid();
    coefficient_shape_ = nx;
    if (periodic_ && std::is_same_v<T, double>)
        coefficient_shape_[dim_ - 1] = nx[dim_ - 1] / 2 + 1;   // r2c half spectrum
    n_coefficients_ = 1;
    for (int n : coefficient_shape_)
        n_coefficients_ *= n;
    complex_coefficients_ = periodic_ || mixed_ || !std::is_same_v<T, double>;

    // The separable wavenumbers of the cosine and sine series hold only in an
    // orthogonal cell: keep the cell orthogonal while the transform exists
    if (!periodic_)
    {
        cb->hold_orthogonal_cell();
        holds_orthogonal_cell_ = true;
    }
}

template <typename T>
FourierTransform<T>::~FourierTransform()
{
    if (holds_orthogonal_cell_)
        cb_->release_orthogonal_cell();
}

template <typename T>
void FourierTransform<T>::make_hermitian(std::complex<double>* values, long long n_blocks) const
{
    if (!periodic_ || !std::is_same_v<T, double>)
        return;
    // Planes of the half spectrum that hold both m and -m: last index 0 and,
    // for an even length, n/2. There M(m) and M(-m) become (M(m) + conj M(-m))/2
    // and its conjugate, and the self-conjugate modes become real.
    const std::vector<int> nx = cb_->get_nx();
    std::array<int, 3> tcoef = {1, 1, 1};
    for (int d = 0; d < dim_; ++d)
        tcoef[3 - dim_ + d] = coefficient_shape_[d];
    const int n_last = nx[dim_ - 1];
    std::vector<int> planes = {0};
    if (n_last % 2 == 0 && n_last > 1)
        planes.push_back(n_last / 2);

    for (long long b = 0; b < n_blocks; ++b)
    {
        std::complex<double>* block = values + b * n_coefficients_;
        for (int l : planes)
        {
            for (int i = 0; i < tcoef[0]; ++i)
            {
                for (int j = 0; j < tcoef[1]; ++j)
                {
                    const int i_neg = (tcoef[0] - i) % tcoef[0];
                    const int j_neg = (tcoef[1] - j) % tcoef[1];
                    const long long idx = (static_cast<long long>(i) * tcoef[1] + j) * tcoef[2] + l;
                    const long long idx_neg = (static_cast<long long>(i_neg) * tcoef[1] + j_neg) * tcoef[2] + l;
                    if (idx_neg < idx)
                        continue;   // the pair was done at its partner
                    // Each component of the mean can be finite even if the
                    // sum overflows. midpoint also retains subnormal values
                    // that would be lost by halving both operands first.
                    const std::complex<double> average{
                        std::midpoint(block[idx].real(), block[idx_neg].real()),
                        std::midpoint(block[idx].imag(), -block[idx_neg].imag())};
                    block[idx] = average;
                    block[idx_neg] = std::conj(average);
                }
            }
        }
    }
}

template <typename T>
std::vector<double> FourierTransform<T>::get_k_squared() const
{
    const double PI = std::numbers::pi;
    const std::vector<int> nx = cb_->get_nx();
    std::vector<double> k_sq(n_coefficients_);

    if (!periodic_)
    {
        // Separable wavenumbers (orthogonal cell): k = πm/L (cosine), π(m+1)/L (sine)
        // or 2πm/L with the signed index m of the FFT (periodic axes of a mixed box);
        // the axes padded to three at the end have k = 0
        const std::array<int, 3> tnx = padded_grid(nx);
        const std::vector<double> lx = cb_->get_lx();
        std::array<std::vector<double>, 3> k_axis;
        for (int d = 0; d < 3; ++d)
        {
            k_axis[d].assign(tnx[d], 0.0);
            if (d >= dim_)
                continue;
            for (int m = 0; m < tnx[d]; ++m)
            {
                if (bc_[d] == BoundaryCondition::PERIODIC)
                    k_axis[d][m] = 2.0 * PI * ((m > tnx[d] / 2) ? m - tnx[d] : m) / lx[d];
                else
                    k_axis[d][m] = PI * (m + (bc_[d] == BoundaryCondition::ABSORBING ? 1 : 0)) / lx[d];
            }
        }
        for (int i = 0; i < tnx[0]; ++i)
            for (int j = 0; j < tnx[1]; ++j)
                for (int k = 0; k < tnx[2]; ++k)
                    k_sq[(static_cast<long long>(i) * tnx[1] + j) * tnx[2] + k] =
                        k_axis[0][i] * k_axis[0][i] + k_axis[1][j] * k_axis[1][j] + k_axis[2][k] * k_axis[2][k];
        return k_sq;
    }

    // |k|² = 4π² uᵀ g⁻¹ u of the alias u of least form, as in the propagator solvers (Pseudo)
    alias_k_squared(padded_grid(nx), padded_grid(coefficient_shape_), index_metric_of(cb_->get_recip_metric(), dim_),
                    k_sq.data());
    return k_sq;
}

template <typename T>
std::vector<double> FourierTransform<T>::get_alias_moments() const
{
    if (!periodic_)
        throw_with_line_number("The alias moments need a periodic box.");
    std::vector<double> moments(static_cast<size_t>(dim_) * dim_ * n_coefficients_);
    alias_moments(padded_grid(cb_->get_nx()), padded_grid(coefficient_shape_),
                  index_metric_of(cb_->get_recip_metric(), dim_), dim_, moments.data());
    return moments;
}

template <typename T>
std::unique_ptr<FourierMultiplier<T>> FourierTransform<T>::create_multiplier(
    const std::complex<double>* values, int n_out, int n_in, bool diagonal)
{
    if (diagonal && (n_out != 1 || n_in != 1))
        throw_with_line_number("A diagonal multiplier has n_out = n_in = 1.");
    if (n_out < 1 || n_in < 1)
        throw_with_line_number("A multiplier needs at least one input and one output field.");

    const long long n_values = static_cast<long long>(n_out) * n_in * n_coefficients_;
    for (long long i = 0; i < n_values; ++i)
    {
        if (!std::isfinite(values[i].real()) || !std::isfinite(values[i].imag()))
            throw_with_line_number("A multiplier must be finite.");
        if (!complex_coefficients_ && values[i].imag() != 0.0)
            throw_with_line_number("Real fields with wall axes have real coefficients: the multiplier must be real.");
    }
    // Real fields keep the real part of the operator, the same on every backend
    std::vector<std::complex<double>> hermitian(values, values + n_values);
    make_hermitian(hermitian.data(), static_cast<long long>(n_out) * n_in);
    return store_multiplier(hermitian.data(), n_values, n_out, n_in, diagonal);
}

template <typename T>
void FourierTransform<T>::apply(const std::vector<FourierTerm<T>>& terms, int n_out, T* out)
{
    if (terms.empty())
        throw_with_line_number("apply() needs at least one term.");
    for (const FourierTerm<T>& term : terms)
    {
        const FourierMultiplier<T>* m = term.multiplier;
        if (m == nullptr || m->get_owner() != this)
            throw_with_line_number("A multiplier is used only with the transform that created it.");
        if (m->is_diagonal())
        {
            if (term.n_fields != n_out)
                throw_with_line_number("A diagonal multiplier gives one output field per input field: got "
                    + std::to_string(term.n_fields) + " inputs for " + std::to_string(n_out) + " outputs.");
        }
        else
        {
            if (term.n_fields != m->get_n_in())
                throw_with_line_number("The multiplier takes " + std::to_string(m->get_n_in())
                    + " input fields, got " + std::to_string(term.n_fields) + ".");
            if (m->get_n_out() != n_out)
                throw_with_line_number("Every term must give the same number of output fields: "
                    + std::to_string(m->get_n_out()) + " != " + std::to_string(n_out) + ".");
        }
    }
    apply_impl(terms, n_out, out);
}

template <typename T>
void FourierTransform<T>::forward(const T* fields, int n_fields, double* coefficients)
{
    if (n_fields < 1)
        throw_with_line_number("forward() needs at least one field.");
    forward_impl(fields, n_fields, coefficients);
}

template <typename T>
void FourierTransform<T>::backward(const double* coefficients, int n_fields, T* fields)
{
    if (n_fields < 1)
        throw_with_line_number("backward() needs at least one field.");
    // The inverse r2c transforms read the planes of make_hermitian() differently
    if (periodic_ && std::is_same_v<T, double>)
        backward_hermitian_impl(coefficients, n_fields, fields);
    else
        backward_impl(coefficients, n_fields, fields);
}

template <typename T>
void FourierTransform<T>::backward_hermitian_impl(const double* coefficients, int n_fields, T* fields)
{
    std::vector<std::complex<double>> hermitian(
        reinterpret_cast<const std::complex<double>*>(coefficients),
        reinterpret_cast<const std::complex<double>*>(coefficients) + static_cast<long long>(n_fields) * n_coefficients_);
    make_hermitian(hermitian.data(), n_fields);
    backward_impl(reinterpret_cast<const double*>(hermitian.data()), n_fields, fields);
}

template class FourierTransform<double>;
template class FourierTransform<std::complex<double>>;
