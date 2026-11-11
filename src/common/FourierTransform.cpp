/**
 * @file FourierTransform.cpp
 * @brief Coefficient layout, |k|² and argument checks shared by the platforms.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>
#include <type_traits>

#include "GridLimits.h"
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
    if (mixed_)
        set_mirror_axes(nx);

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
void FourierTransform<T>::set_mirror_axes(const std::vector<int>& nx)
{
    const double PI = std::numbers::pi;
    const std::complex<double> I(0.0, 1.0);
    n_extended_ = 1;
    for (int d = 0; d < 3; ++d)
    {
        MirrorAxis& axis = mirror_axes_[d];
        const int a = d - (3 - dim_);
        const BoundaryCondition bc = (a < 0) ? BoundaryCondition::PERIODIC : bc_[a];
        const int n = (a < 0) ? 1 : nx[a];
        const bool wall = bc != BoundaryCondition::PERIODIC;
        const bool sine = bc == BoundaryCondition::ABSORBING;
        axis.n = n;
        axis.n_extended = wall ? 2 * n : n;
        n_extended_ *= axis.n_extended;
        const int n2 = axis.n_extended;
        axis.extend_source.resize(n2);
        axis.extend_sign.resize(n2);
        axis.fill_source.assign(n2, -1);
        axis.fill_factor.assign(n2, 0.0);
        axis.extract_source.resize(n);
        axis.extract_factor.resize(n);
        if (!wall)
        {
            for (int j = 0; j < n; ++j)
            {
                axis.extend_source[j] = j;
                axis.extend_sign[j] = 1.0;
                axis.extract_source[j] = j;
                axis.extract_factor[j] = 1.0;
                axis.fill_source[j] = j;
                axis.fill_factor[j] = 1.0;
            }
            continue;
        }
        // Mirror image f_{2N-1-j} = ±f_j: even for the cosine series, odd for the sine series
        for (int j = 0; j < n2; ++j)
        {
            axis.extend_source[j] = (j < n) ? j : n2 - 1 - j;
            axis.extend_sign[j] = (j < n || !sine) ? 1.0 : -1.0;
        }
        // Its mode m is 2 e^{iπm/2N} C(m) (cosine) or -2i e^{iπm/2N} S(m) (sine, S(m) the
        // coefficient m - 1), with C(2N-m) = -C(m) and S(2N-m) = S(m)
        auto phase = [&](int m) { return std::polar(1.0, PI * m / n2); };
        for (int m = 0; m < n; ++m)
        {
            const int mode = sine ? m + 1 : m;
            axis.extract_source[m] = mode;
            axis.extract_factor[m] = (sine ? 0.5 * I : 0.5) * std::conj(phase(mode));
        }
        for (int j = 0; j < n2; ++j)
        {
            if (!sine)
            {
                if (j < n)
                {
                    axis.fill_source[j] = j;
                    axis.fill_factor[j] = 2.0 * phase(j);
                }
                else if (j > n)
                {
                    axis.fill_source[j] = n2 - j;
                    axis.fill_factor[j] = 2.0 * std::conj(phase(n2 - j));
                }
            }
            else
            {
                if (j >= 1 && j <= n)
                {
                    axis.fill_source[j] = j - 1;
                    axis.fill_factor[j] = -2.0 * I * phase(j);
                }
                else if (j > n)
                {
                    axis.fill_source[j] = n2 - j - 1;
                    axis.fill_factor[j] = 2.0 * I * std::conj(phase(n2 - j));
                }
            }
        }
    }
    // The FFTs and kernels index the doubles of a complex field with int
    if (n_extended_ > MAX_GRID_POINTS)
        throw_with_line_number("The mirror image of this box (" + std::to_string(n_extended_)
            + " points, twice the grid along every wall axis) exceeds the largest supported grid ("
            + std::to_string(MAX_GRID_POINTS) + " points).");
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
                    const std::complex<double> average = 0.5 * (block[idx] + std::conj(block[idx_neg]));
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

    // Pad to 3D in front, as the propagator solvers do (Pseudo.cpp)
    std::array<int, 3> tnx = {1, 1, 1};
    std::array<int, 3> tcoef = {1, 1, 1};
    for (int d = 0; d < dim_; ++d)
    {
        tnx[3 - dim_ + d] = nx[d];
        tcoef[3 - dim_ + d] = coefficient_shape_[d];
    }

    if (!periodic_)
    {
        // Separable wavenumbers (orthogonal cell): k = πm/L (cosine), π(m+1)/L (sine)
        // or 2πm/L with the signed index m of the FFT (periodic axes of a mixed box)
        const std::vector<double> lx = cb_->get_lx();
        std::array<std::vector<double>, 3> k_axis;
        for (int d = 0; d < 3; ++d)
        {
            k_axis[d].assign(tnx[d], 0.0);
            const int a = d - (3 - dim_);
            if (a < 0)
                continue;
            for (int m = 0; m < tnx[d]; ++m)
            {
                if (bc_[a] == BoundaryCondition::PERIODIC)
                    k_axis[d][m] = 2.0 * PI * ((m > tnx[d] / 2) ? m - tnx[d] : m) / lx[a];
                else
                    k_axis[d][m] = PI * (m + (bc_[a] == BoundaryCondition::ABSORBING ? 1 : 0)) / lx[a];
            }
        }
        for (int i = 0; i < tnx[0]; ++i)
            for (int j = 0; j < tnx[1]; ++j)
                for (int k = 0; k < tnx[2]; ++k)
                    k_sq[(static_cast<long long>(i) * tnx[1] + j) * tnx[2] + k] =
                        k_axis[0][i] * k_axis[0][i] + k_axis[1][j] * k_axis[1][j] + k_axis[2][k] * k_axis[2][k];
        return k_sq;
    }

    // |k|² = 4π² mᵀ g⁻¹ m with the reciprocal metric of the box, as in
    // Pseudo<T>::update_boltz_bond_periodic_for_ds_index()
    const std::array<double, 6>& recip_metric = cb_->get_recip_metric();
    std::array<double, 3> Gd;
    double Gij, Gik, Gjk;
    if (dim_ == 3) {
        Gd = {recip_metric[0], recip_metric[3], recip_metric[5]};
        Gij = recip_metric[1]; Gik = recip_metric[2]; Gjk = recip_metric[4];
    } else if (dim_ == 2) {
        Gd = {0.0, recip_metric[0], recip_metric[3]};
        Gij = 0.0; Gik = 0.0; Gjk = recip_metric[1];
    } else {
        Gd = {0.0, 0.0, recip_metric[0]};
        Gij = 0.0; Gik = 0.0; Gjk = 0.0;
    }
    const double G_diag_max = std::max({std::abs(Gd[0]), std::abs(Gd[1]), std::abs(Gd[2])});
    const bool oblique = (std::abs(Gij) > 1e-12 * G_diag_max ||
                          std::abs(Gik) > 1e-12 * G_diag_max ||
                          std::abs(Gjk) > 1e-12 * G_diag_max);
    const IndexMetric index_metric{Gd[0], Gd[1], Gd[2], Gij, Gik, Gjk};
    auto form = [&](double a, double b, double c) {
        return Gd[0] * a * a + Gd[1] * b * b + Gd[2] * c * c +
               2.0 * (Gij * a * b + Gik * a * c + Gjk * b * c);
    };
    auto signed_index = [](int i, int n) { return (i > n / 2) ? i - n : i; };

    for (int i = 0; i < tcoef[0]; ++i)
    {
        for (int j = 0; j < tcoef[1]; ++j)
        {
            for (int k = 0; k < tcoef[2]; ++k)
            {
                std::array<int, 3> m = {signed_index(i, tnx[0]), signed_index(j, tnx[1]), signed_index(k, tnx[2])};
                if (oblique)
                    min_form_alias_rep(m, tnx, index_metric, form);
                k_sq[(static_cast<long long>(i) * tcoef[1] + j) * tcoef[2] + k] =
                    4.0 * PI * PI * form(m[0], m[1], m[2]);
            }
        }
    }
    return k_sq;
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
    if (periodic_ && std::is_same_v<T, double>)
    {
        // The inverse r2c transforms read the planes of make_hermitian() differently
        std::vector<std::complex<double>> hermitian(
            reinterpret_cast<const std::complex<double>*>(coefficients),
            reinterpret_cast<const std::complex<double>*>(coefficients) + static_cast<long long>(n_fields) * n_coefficients_);
        make_hermitian(hermitian.data(), n_fields);
        backward_impl(reinterpret_cast<const double*>(hermitian.data()), n_fields, fields);
        return;
    }
    backward_impl(coefficients, n_fields, fields);
}

template class FourierTransform<double>;
template class FourierTransform<std::complex<double>>;
