/**
 * @file CpuComputationBackend.cpp
 * @brief CPU implementation of ComputationBackend.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuComputationBackend.h"
#include "Exception.h"
#include "NumericalMethods.h"
#include "ScaledProduct.h"
#include "PartitionAccumulator.h"
#include "JunctionProduct.h"
#include "CpuSolverCNADI.h"
#include "CpuSolverPseudoDiscrete.h"
#include "CpuSolverPseudoRK2.h"
#include "CpuSolverPseudoRQM4.h"
#include "CpuSolverRealSpaceDiscrete.h"

// Only GCC vectorizes these loops as `omp simd`. MSVC accepts the pragma only with
// /openmp:experimental, and Clang-based compilers (clang, clang-cl, icx) cannot
// vectorize them and warn with -Wpass-failed, an error under -Werror. Elsewhere they
// stay ordinary loops; their reductions give the same values in either form.
#if defined(__GNUC__) && !defined(__clang__)
#define POLYMERFTS_PRAGMA(text) _Pragma(#text)
#define POLYMERFTS_OMP_SIMD(clauses) POLYMERFTS_PRAGMA(omp simd clauses)
#else
#define POLYMERFTS_OMP_SIMD(clauses)
#endif

template <typename T>
CpuComputationBackend<T>::CpuComputationBackend(
    ComputationBox<T>* cb, Molecules* molecules,
    const NumericalMethodInfo& method_info,
    FFTBackend fft_backend, SpaceGroup* space_group, int max_n_streams)
    : cb(cb), space_group(space_group), chain_model(molecules->get_model_name())
{
    n_basis = cb->get_n_basis();
    n_streams = std::min(this->n_streams_from_env(/*default_n_streams=*/4), max_n_streams);
    stress_pairs.resize(n_streams);

    switch (method_info.id)
    {
    case NumericalMethodId::PS:
        solver = std::make_unique<CpuSolverPseudoDiscrete<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::RS:
        solver = std::make_unique<CpuSolverRealSpaceDiscrete<T>>(cb, molecules);
        break;
    case NumericalMethodId::RQM4:
        solver = std::make_unique<CpuSolverPseudoRQM4<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::RK2:
        solver = std::make_unique<CpuSolverPseudoRK2<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::CNADI2:
        // The factory rejects this combination first (validate_numerical_method);
        // without a solver the calls below would dereference null
        if constexpr (std::is_same_v<T, double>)
            solver = std::make_unique<CpuSolverCNADI>(cb, molecules);
        else
            throw_with_line_number("The cn-adi2 method is only available for real fields.");
        break;
    }

    // The solver expands/reduces around its FFTs when a space group is set
    // (the owning member releases the solver if this throws)
    if (space_group != nullptr)
        solver->set_space_group(space_group);
    solver->update_laplacian_operator();
}

template <typename T>
CpuComputationBackend<T>::~CpuComputationBackend() = default;

// ==================== Memory ====================

template <typename T>
BackendArray<T>* CpuComputationBackend<T>::allocate()
{
    return reinterpret_cast<BackendArray<T>*>(new T[n_basis]());
}

template <typename T>
void CpuComputationBackend<T>::deallocate(BackendArray<T>* a)
{
    delete[] raw(a);
}

template <typename T>
void CpuComputationBackend<T>::copy_from_host(int, BackendArray<T>* dst, const T* src)
{
    std::copy(src, src + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::copy_to_host(T* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, dst);
}

template <typename T>
CheckpointArray<T>* CpuComputationBackend<T>::allocate_checkpoint()
{
    return reinterpret_cast<CheckpointArray<T>*>(new T[n_basis]);
}

template <typename T>
void CpuComputationBackend<T>::deallocate_checkpoint(CheckpointArray<T>* a)
{
    delete[] raw(a);
}

template <typename T>
void CpuComputationBackend<T>::store_checkpoint(int, CheckpointArray<T>* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::load_checkpoint(int, BackendArray<T>* dst, CheckpointArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

// ==================== Solver ====================

template <typename T>
void CpuComputationBackend<T>::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

template <typename T>
void CpuComputationBackend<T>::update_dw(std::map<std::string, const T*> w_input)
{
    solver->update_dw(w_input);

    // The discrete solver keeps exp(-w ds) on the full grid unless its CrysFFT
    // path stores it in the reduced basis already
    if (chain_model == "discrete" && space_group != nullptr)
    {
        for (const auto& [monomer_type, exp_dw_solver] : solver->exp_dw.at(0))
        {
            std::vector<T>& reduced = exp_dw_reduced[monomer_type];
            reduced.resize(n_basis);
            if (static_cast<int>(exp_dw_solver.size()) == n_basis)
                std::copy(exp_dw_solver.begin(), exp_dw_solver.end(), reduced.begin());
            else if constexpr (std::is_same_v<T, double>)
                space_group->to_reduced_basis(exp_dw_solver.data(), reduced.data(), 1);
            else
                throw_with_line_number("Discrete chains with a space group require real fields.");
        }
    }
}

template <typename T>
void CpuComputationBackend<T>::set_force_off_diagonal_stress(bool force)
{
    solver->set_force_off_diagonal_stress(force);
}

template <typename T>
void CpuComputationBackend<T>::advance_propagator(
    int, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type, int ds_index)
{
    solver->advance_propagator(raw(q_in), raw(q_out), monomer_type, cb->get_mask(), ds_index);
}

template <typename T>
void CpuComputationBackend<T>::advance_propagator_half_bond_step(
    int, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type)
{
    solver->advance_propagator_half_bond_step(raw(q_in), raw(q_out), monomer_type);
}

template <typename T>
BackendArray<T>* CpuComputationBackend<T>::boltzmann_factor(const std::string& monomer_type)
{
    if (space_group != nullptr)
        return reinterpret_cast<BackendArray<T>*>(exp_dw_reduced.at(monomer_type).data());
    return reinterpret_cast<BackendArray<T>*>(solver->exp_dw.at(0).at(monomer_type).data());
}

// ==================== Propagator initial conditions ====================

template <typename T>
void CpuComputationBackend<T>::set_unity(int, BackendArray<T>* q)
{
    std::fill(raw(q), raw(q) + n_basis, T(1.0));
}

template <typename T>
void CpuComputationBackend<T>::set_zero(int, BackendArray<T>* a)
{
    std::fill(raw(a), raw(a) + n_basis, T(0.0));
}

template <typename T>
void CpuComputationBackend<T>::copy(int, BackendArray<T>* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::sum_sources(T* q, const std::vector<std::pair<const T*, double>>& sources)
{
    for (int i = 0; i < n_basis; i++)
        q[i] = 0.0;
    for (const auto& [src, n_repeated] : sources)
        for (int i = 0; i < n_basis; i++)
            q[i] += src[i]*n_repeated;
}

template <typename T>
void CpuComputationBackend<T>::multiply_sources(T* q, const std::vector<std::pair<const T*, double>>& sources)
{
    if constexpr (!std::is_same_v<T, double>)
    {
        for (const auto& source : sources)
            if (!is_integer_power(source.second))
                throw_with_line_number(complex_real_power_message(source.second));
    }
    for (int i = 0; i < n_basis; i++)
    {
        junction_product::Accumulator<T> product;
        for (const auto& [src, n_repeated] : sources)
        {
            if constexpr (std::is_same_v<T, double>)
                product.multiply(src[i], n_repeated);
            else
                product.multiply(src[i].real(), src[i].imag(), static_cast<int>(n_repeated));
        }
        if constexpr (std::is_same_v<T, double>)
            q[i] = product.value();
        else
        {
            const auto value = product.value();
            q[i] = T(value.real, value.imag);
        }
    }
}

template <typename T>
void CpuComputationBackend<T>::junction_sum(
    int, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, double>>& sources)
{
    std::vector<std::pair<const T*, double>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    sum_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_product(
    int, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, double>>& sources)
{
    std::vector<std::pair<const T*, double>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    multiply_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_sum(
    int, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, double>>& sources)
{
    std::vector<std::pair<const T*, double>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    sum_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_product(
    int, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, double>>& sources)
{
    std::vector<std::pair<const T*, double>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    multiply_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::apply_mask(int, BackendArray<T>* q)
{
    const double* mask = cb->get_mask();
    if (mask == nullptr)
        return;
    T* _q = raw(q);
    for (int i = 0; i < n_basis; i++)
        _q[i] *= mask[i];
}

template <typename T>
void CpuComputationBackend<T>::multiply_boltzmann_factor(int, BackendArray<T>* q, const std::string& monomer_type)
{
    T* _q = raw(q);
    const T* exp_dw = raw(boltzmann_factor(monomer_type));
    for (int i = 0; i < n_basis; i++)
        _q[i] *= exp_dw[i];
}

// ==================== Statistics ====================

template <typename T>
T CpuComputationBackend<T>::inner_product(BackendArray<T>* a, BackendArray<T>* b)
{
    return cb->inner_product(raw(a), raw(b));
}

template <typename T>
T CpuComputationBackend<T>::partition_product(
    BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* divisor, double copies)
{
    return cb->mean_product(raw(a), raw(b), divisor == nullptr ? nullptr : raw(divisor), copies);
}

template <typename T>
double CpuComputationBackend<T>::absolute_partition_product(
    BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* divisor, double copies)
{
    const double integral_value = absolute_inner_product(a, b, divisor);
    const double result = integral_value / (copies * cb->get_volume());
    if (partition_accumulation::ordinary_result(integral_value)
        && partition_accumulation::ordinary_result(result))
        return result;
    return std::real(cb->mean_product(raw(a), raw(b),
        divisor == nullptr ? nullptr : raw(divisor), copies, true));
}

template <typename T>
T CpuComputationBackend<T>::inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w)
{
    return cb->inner_product_inverse_weight(raw(a), raw(b), raw(w));
}

template <typename T>
double CpuComputationBackend<T>::absolute_inner_product(
    BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* divisor)
{
    if (divisor != nullptr)
    {
        // Include the integration weight before restoring the product's
        // exponent, including the volume element in a reduced basis.
        const double* mask = cb->get_mask();
        const double dv = cb->get_volume() / cb->get_total_grid();
        const auto& orbits = cb->get_orbit_counts();
        double sum = 0.0;
        for (int i = 0; i < n_basis; ++i)
            if (mask == nullptr || mask[i] != 0.0)
            {
                const double weight = space_group != nullptr ? orbits[i] * dv : cb->get_dv(i);
                sum += std::abs(scaled_product::divide(raw(a)[i], raw(b)[i], raw(divisor)[i], weight));
            }
        return sum;
    }
    if (absolute_product.empty())
        absolute_product.resize(n_basis);
    const double* mask = cb->get_mask();
    for (int i = 0; i < n_basis; ++i)
    {
        if (mask != nullptr && mask[i] == 0.0)
        {
            absolute_product[i] = T(0.0);
            continue;
        }
        T product = raw(a)[i] * raw(b)[i];
        absolute_product[i] = std::abs(product);
    }
    return std::real(cb->integral(absolute_product.data()));
}

template <typename T>
void CpuComputationBackend<T>::add_segment_products(
    int, BackendArray<T>* phi, const std::vector<SegmentPair<T>>& pairs, BackendArray<T>* divisor,
    BackendArray<T>* exponent)
{
    T* _phi = raw(phi);
    if (exponent != nullptr)
    {
        T* _exponent = raw(exponent);
        const T* _divisor = divisor == nullptr ? nullptr : raw(divisor);
        const double* mask = divisor == nullptr ? nullptr : cb->get_mask();
        for (const SegmentPair<T>& pair : pairs)
        {
            const T* q_1 = raw(pair.q_1);
            const T* q_2 = raw(pair.q_2);
            const double weight = pair.weight;
            // Classify a cache-sized tile with a SIMD reduction, then retain
            // the original vectorizable arithmetic for ordinary fields. A
            // per-point branch containing frexp/scalbn prevents that loop
            // from vectorizing even when none of its slow paths are taken.
            constexpr int TILE = 256;
            const auto ordinary_value = [](T value)
            {
                const auto component = [](double x)
                {
                    const double magnitude = std::abs(x);
                    return static_cast<int>(magnitude == 0.0)
                         | (static_cast<int>(magnitude >= 0x1p-200)
                          & static_cast<int>(magnitude <= 0x1p200));
                };
                return component(std::real(value)) & component(std::imag(value));
            };
            for (int begin = 0, end = 0; begin < n_basis; begin = end)
            {
                end = begin + std::min(TILE, n_basis - begin);
                int regular = scaled_product::ordinary(weight);
                if (_divisor == nullptr)
                {
                    if constexpr (std::is_same_v<T, double>)
                    {
                        double minimum = 1.0, maximum = 1.0, maximum_exponent = 0.0;
                        POLYMERFTS_OMP_SIMD(reduction(min:minimum) reduction(max:maximum,maximum_exponent))
                        for (int i = begin; i < end; ++i)
                        {
                            const double a = std::abs(q_1[i]), b = std::abs(q_2[i]);
                            minimum = std::min(minimum, std::min(a, b));
                            maximum = std::max(maximum, std::max(a, b));
                            maximum_exponent = std::max(maximum_exponent, std::abs(_exponent[i]));
                        }
                        regular &= static_cast<int>(minimum >= 0x1p-200 && maximum <= 0x1p200
                                                    && maximum_exponent == 0.0);
                        // Zero operands are safe, but must not hide a tiny
                        // nonzero operand elsewhere in the same tile.
                        if (minimum == 0.0 && maximum <= 0x1p200 && maximum_exponent == 0.0)
                        {
                            regular = scaled_product::ordinary(weight);
                            POLYMERFTS_OMP_SIMD(reduction(&:regular))
                            for (int i = begin; i < end; ++i)
                                regular &= ordinary_value(q_1[i]) & ordinary_value(q_2[i]);
                        }
                        // The bounded nonzero products lie in [2^-600,2^600].
                        // Adding one to any finite unscaled accumulator is
                        // safe: near DBL_MAX half an ulp is at least 2^970.
                        // No lower bound on that accumulator is needed either.
                        // A NaN operand may be ignored by min/max, but its
                        // product stays NaN with the original IEEE arithmetic;
                        // it cannot conceal an exceptional finite product.
                    }
                    else
                    {
                        POLYMERFTS_OMP_SIMD(reduction(&:regular))
                        for (int i = begin; i < end; ++i)
                            regular &= static_cast<int>(_exponent[i] == T(0.0)) & ordinary_value(_phi[i])
                                     & ordinary_value(q_1[i]) & ordinary_value(q_2[i]);
                    }
                }
                else
                {
                    POLYMERFTS_OMP_SIMD(reduction(&:regular))
                    for (int i = begin; i < end; ++i)
                    {
                        const int safe = static_cast<int>(_exponent[i] == T(0.0)) & ordinary_value(_phi[i])
                                       & ordinary_value(q_1[i]) & ordinary_value(q_2[i])
                                       & ordinary_value(_divisor[i])
                                       & static_cast<int>(_divisor[i] != T(0.0));
                        regular &= safe | static_cast<int>(mask != nullptr && mask[i] == 0.0);
                    }
                }
                if (regular)
                {
                    if (_divisor == nullptr)
                        for (int i = begin; i < end; ++i)
                            _phi[i] += weight * q_1[i] * q_2[i];
                    else
                        for (int i = begin; i < end; ++i)
                            if (mask == nullptr || mask[i] != 0.0)
                                _phi[i] += weight * q_1[i] * q_2[i] / _divisor[i];
                    continue;
                }
                for (int i = begin; i < end; ++i)
                {
                    if (mask != nullptr && mask[i] == 0.0)
                        continue;
                    if constexpr (std::is_same_v<T, double>)
                        density_accumulation::accumulate(_phi[i], _exponent[i], q_1[i], q_2[i], weight,
                                                         _divisor == nullptr ? 1.0 : _divisor[i]);
                    else
                        density_accumulation::accumulate(_phi[i], _exponent[i], q_1[i], q_2[i], weight,
                                                         _divisor == nullptr ? nullptr : &_divisor[i]);
                }
            }
        }
        return;
    }
    if (divisor != nullptr)
    {
        const T* _divisor = raw(divisor);
        const double* mask = cb->get_mask();
        for (const SegmentPair<T>& pair : pairs)
        {
            const T* q_1 = raw(pair.q_1);
            const T* q_2 = raw(pair.q_2);
            for (int i = 0; i < n_basis; ++i)
                if (mask == nullptr || mask[i] != 0.0)
                    _phi[i] += scaled_product::divide(q_1[i], q_2[i], _divisor[i], pair.weight);
        }
        return;
    }
    for (const SegmentPair<T>& pair : pairs)
    {
        const T* q_1 = raw(pair.q_1);
        const T* q_2 = raw(pair.q_2);
        const double weight = pair.weight;
        for (int i = 0; i < n_basis; i++)
            _phi[i] += weight*q_1[i]*q_2[i];
    }
}

template <typename T>
void CpuComputationBackend<T>::normalize_concentration(int, BackendArray<T>* phi, T norm, BackendArray<T>* divisor)
{
    T* _phi = raw(phi);
    if (divisor != nullptr)
    {
        const T* _divisor = raw(divisor);
        const double* mask = cb->get_mask();
        for (int i = 0; i < n_basis; i++)
        {
            // A masked bead has zero density, including when exp(-w ds) underflows.
            if (mask != nullptr && mask[i] == 0.0)
                _phi[i] = T(0.0);
            else
                _phi[i] /= _divisor[i];
        }
    }
    for (int i = 0; i < n_basis; i++)
        _phi[i] *= norm;
}

template <typename T>
StressAccumulator<T> CpuComputationBackend<T>::block_stress(
    int stream, const std::vector<SegmentPair<T>>& pairs, const std::string& monomer_type)
{
    StressAccumulator<T> stress{};
    for (const SegmentPair<T>& pair : pairs)
    {
        T* original[2] = {raw(pair.q_1), raw(pair.q_2)};
        double maxima[2][2] = {};
        bool split_components = false;
        for (int side = 0; side < 2; ++side)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                maxima[side][0] = std::max(maxima[side][0], std::abs(std::real(original[side][i])));
                if constexpr (!std::is_same_v<T, double>)
                    maxima[side][1] = std::max(maxima[side][1], std::abs(std::imag(original[side][i])));
            }
            split_components = split_components || stress_accumulation::needs_component_split(
                maxima[side][0], maxima[side][1]);
        }

        // Ordinary complex pairs retain their two FFTs. A rare pair with a
        // component too small for common scaling uses Re/Re, Re/Im, Im/Re,
        // Im/Im contractions; their signs/phases are restored after the FFT.
        StressAccumulator<T> pair_stress{};
        for (int contraction = 0; contraction < (split_components ? 4 : 1); ++contraction)
        {
            T* inputs[2] = {original[0], original[1]};
            int shift = 0;
            for (int side = 0; side < 2; ++side)
            {
                const int component = side == 0 ? contraction / 2 : contraction % 2;
                const double maximum = split_components ? maxima[side][component]
                    : std::max(maxima[side][0], maxima[side][1]);
                const int exponent = stress_accumulation::input_shift(maximum);
                shift += exponent;
                if (exponent != 0 || split_components)
                {
                    auto& scaled = stress_pairs[stream][side];
                    scaled.resize(n_basis);
                    for (int i = 0; i < n_basis; ++i)
                    {
                        if constexpr (std::is_same_v<T, double>)
                            scaled[i] = std::scalbn(original[side][i], -exponent);
                        else if (split_components)
                            scaled[i] = T(std::scalbn(component == 0 ? original[side][i].real()
                                                                 : original[side][i].imag(), -exponent), 0.0);
                        else
                            scaled[i] = {std::scalbn(original[side][i].real(), -exponent),
                                         std::scalbn(original[side][i].imag(), -exponent)};
                    }
                    inputs[side] = scaled.data();
                }
            }
            // The solver expands/reduces internally when a space group is set.
            const std::vector<T> segment_stress = solver->compute_single_segment_stress(
                inputs[0], inputs[1], monomer_type, pair.is_half_bond_length);
            for (int d = 0; d < 6; d++)
            {
                T segment = segment_stress[d];
                if constexpr (!std::is_same_v<T, double>)
                    if (split_components)
                        segment = (contraction == 1 || contraction == 2)
                            ? T(0.0, segment.real()) : T((contraction == 3 ? -1.0 : 1.0) * segment.real(), 0.0);
                pair_stress[d] += StressValue<T>::shifted(segment, shift);
            }
        }
        for (int d = 0; d < 6; ++d)
            stress[d] += pair_stress[d] * pair.weight;
    }
    return stress;
}

template <typename T>
T CpuComputationBackend<T>::solvent_concentration(
    BackendArray<T>* phi, BackendArray<T>* w, double ds, double volume_fraction)
{
    T* _phi = raw(phi);
    const T* _w = raw(w);
    for (int i = 0; i < n_basis; i++)
        _phi[i] = exp(-_w[i] * ds);

    const T partition = cb->mean(_phi);
    // Keep Q for diagnostics, but an absent species has zero concentration even
    // when its Boltzmann factors cancel (Q = 0) or are non-finite.
    if (volume_fraction == 0.0)
    {
        std::fill(_phi, _phi + n_basis, T(0.0));
        return partition;
    }
    const auto norm = make_density_normalization(volume_fraction, 1.0, 1.0, 1.0, partition);
    for (int i = 0; i < n_basis; i++)
        _phi[i] = apply_density_normalization(_phi[i], norm);

    // No solvent inside impenetrable regions (the mask is in the field basis)
    const double* mask = cb->get_mask();
    if (mask != nullptr)
        for (int i = 0; i < n_basis; i++)
            _phi[i] *= mask[i];
    return partition;
}

template <typename T>
void CpuComputationBackend<T>::sum_to_host(
    T* out, const std::vector<BackendArray<T>*>& arrays,
    const std::vector<DensityNormalization<T>>& weights,
    const std::vector<BackendArray<T>*>& exponents)
{
    for (int i = 0; i < n_basis; i++)
        out[i] = 0.0;
    for (size_t a = 0; a < arrays.size(); a++)
    {
        const T* _a = raw(arrays[a]);
        const T* exponent = exponents.empty() || exponents[a] == nullptr ? nullptr : raw(exponents[a]);
        const auto& weight = weights[a];
        for (int i = 0; i < n_basis; i++)
            out[i] += apply_density_normalization(_a[i], weight, exponent == nullptr ? T(0.0) : exponent[i]);
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationBackend);
