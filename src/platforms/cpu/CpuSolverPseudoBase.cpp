/**
 * @file CpuSolverPseudoBase.cpp
 * @brief Implementation of common pseudo-spectral solver functionality.
 *
 * Implements shared code of the three pseudo-spectral solvers (CpuSolverPseudoRQM4,
 * CpuSolverPseudoRK2 and CpuSolverPseudoDiscrete), including:
 *
 * - FFT object initialization and cleanup
 * - The CrysFFT diffusion of reduced-basis propagators
 * - Laplacian operator updates
 * - Stress computation with chain-model-specific factors
 *
 * Transform dispatch goes through the FFT<T> interface (FftwFFT, MklFFT or MakhoulFFT).
 *
 * @see CpuSolverPseudoBase.h for interface documentation
 */

#include <iostream>
#include <array>
#include <cmath>
#include <complex>
#include <type_traits>
#include <algorithm>

#include "CpuSolverPseudoBase.h"
#include "CrysFFTSelector.h"
#include "CrysFFTObliqueZ.h"
#include "CrysFFTPmmm.h"
#include "CrysFFTRecursive3m.h"
#include "PseudoSolverHelpers.h"

namespace
{
/**
 * @brief stress[slot[c]] += Σ_i kernel(i) basis[c][i] for the NC components c, each a
 *        sequential sum from zero in one pass over the coefficients.
 *
 * NC is a compile-time constant, so the sums stay in registers; the caller sums the
 * cross terms only when it needs them (see compute_single_segment_stress).
 */
template <int NC, typename T, typename Kernel>
void accumulate_dyads(int n_coeffs, Kernel&& kernel, const std::array<const double*, 6>& basis,
                      const std::array<int, 6>& slot, std::vector<T>& stress)
{
    std::array<T, NC> v;
    v.fill(T{0.0});
    for (int i = 0; i < n_coeffs; ++i)
    {
        const T coeff = kernel(i);
        for (int c = 0; c < NC; ++c)
            v[c] += coeff * basis[c][i];
    }
    for (int c = 0; c < NC; ++c)
        stress[slot[c]] += v[c];
}
}  // namespace

//------------------------------------------------------------------------------
// Initialize shared components
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::init_shared(FFTBackend backend)
{
    this->fft_backend_ = backend;

    // Both faces of each direction need the same BC: the transforms take whole directions
    auto bc_vec = cb->get_boundary_conditions();
    dim_ = cb->get_dim();
    is_periodic_ = pseudo_spectral_periodic(bc_vec, dim_);

    // The transform of the grid, one boundary condition per dimension (all BCs and backends)
    std::vector<BoundaryCondition> bc_per_dim;
    for (int d = 0; d < dim_; ++d)
        bc_per_dim.push_back(bc_vec[2 * d]);
    fft_ = createFFT<T>(cb->get_nx(), bc_per_dim, backend);

    // The Fourier-space tables of every distinct contour step (the CpuSolver constructor
    // finalized the ContourLengthMapping); update_laplacian_operator() computes them for the cell
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
        ds_values_[ds_idx] = mapping.get_ds_from_index(ds_idx);
    pseudo = std::make_unique<Pseudo<T>>(molecules->get_bond_lengths(), bc_per_dim, cb->get_nx(), ds_values_);

    // With a space group the propagators live in its reduced basis, and the solver expands
    // and reduces around its full-grid steps. [real-only feature] The CrysFFT fast paths
    // fold real fields on the physical sub-grid (the factory refuses complex fields with
    // a space group).
    if (space_group_ != nullptr)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            if (dim_ == 3)
                init_crysfft();
        }
        // The CrysFFT maps above are snapshots: freeze the physical basis
        space_group_->mark_in_use();
    }

    // The tables of the cell
    update_laplacian_operator();
}

//------------------------------------------------------------------------------
// Space group / CrysFFT setup
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::init_crysfft()
{
    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    // Engine choice and its basis map, shared with the CUDA solver
    const CrysFFTPlan plan = plan_crysfft(
        space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), cb->is_z_axis_orthogonal());
    if (plan.engine == CrysFFTEngine::None)
        return;

    // The physical-grid -> reduced-basis map, not needed when the reduced basis is the
    // physical grid (a physical basis has the identity map, see plan_crysfft())
    crysfft_identity_map_ = plan.map.identity;
    if (!crysfft_identity_map_)
        crysfft_reduced_indices_ = plan.map.phys_to_reduced;

    // The engines run over the real FFT of the selected library
    if (plan.engine == CrysFFTEngine::Recursive3m)
        crysfft_ = std::make_unique<CrysFFTRecursive3m>(nx_logical, plan.m3_translations, fft_backend_);
    else if (plan.engine == CrysFFTEngine::Pmmm)
        crysfft_ = std::make_unique<CrysFFTPmmm>(nx_logical, fft_backend_);
    else
        crysfft_ = std::make_unique<CrysFFTObliqueZ>(nx_logical, fft_backend_);

    // The Boltzmann tables of the steps of the blocks, which update_laplacian_operator()
    // builds for the cell
    const auto block_coeffs = [this](const std::string& monomer_type, int ds_index) {
        return crysfft_block_coeffs(monomer_type, ds_index);
    };
    for (double coeff : crysfft_step_coeffs(*molecules, block_coeffs))
        crysfft_boltz_.try_emplace(coeff);
}

//------------------------------------------------------------------------------
// CrysFFT diffusion of reduced-basis propagators
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::crysfft_diffuse(const double* q_in, double* q_out, double coeff,
                                             const double* pre, const double* mid, const double* post) const
{
    const auto boltz_it = crysfft_boltz_.find(coeff);
    if (boltz_it == crysfft_boltz_.end())
        throw_with_line_number("No CrysFFT table was built for the step coefficient " + std::to_string(coeff) +
            " (see crysfft_block_coeffs()).");
    const std::vector<double>& boltz = boltz_it->second;
    const int M_phys = crysfft_->get_M_physical();

    if (crysfft_identity_map_)
    {
        // The physical grid is the reduced basis: work in q_out, in place
        if (pre != nullptr)
        {
            for (int i = 0; i < M_phys; ++i)
                q_out[i] = q_in[i] * pre[i];
            crysfft_->diffusion(q_out, q_out, boltz);
        }
        else
        {
            crysfft_->diffusion(q_in, q_out, boltz);   // keeps q_in
        }
        if (mid != nullptr)
        {
            for (int i = 0; i < M_phys; ++i)
                q_out[i] *= mid[i];
            crysfft_->diffusion(q_out, q_out, boltz);
        }
        if (post != nullptr)
            for (int i = 0; i < M_phys; ++i)
                q_out[i] *= post[i];
        return;
    }

    // Gather onto this thread's physical grid, diffuse there in place and scatter back:
    // physical-grid point i holds reduced-basis point index[i]
    const int* index = crysfft_reduced_indices_.data();
    std::vector<double>& phys_vector = work_arrays().phys_in;
    phys_vector.resize(M_phys);
    double* phys = phys_vector.data();
    if (pre != nullptr)
        for (int i = 0; i < M_phys; ++i)
            phys[i] = q_in[index[i]] * pre[index[i]];
    else
        for (int i = 0; i < M_phys; ++i)
            phys[i] = q_in[index[i]];

    crysfft_->diffusion(phys, phys, boltz);
    if (mid != nullptr)
    {
        for (int i = 0; i < M_phys; ++i)
            phys[i] *= mid[index[i]];
        crysfft_->diffusion(phys, phys, boltz);
    }
    if (post != nullptr)
        for (int i = 0; i < M_phys; ++i)
            phys[i] *= post[index[i]];

    // The CrysFFT path is taken only when the physical grid reaches every reduced-basis
    // point (plan_crysfft()), so the scatter writes all of q_out
    for (int i = 0; i < M_phys; ++i)
        q_out[index[i]] = phys[i];
}

template <typename T>
void CpuSolverPseudoBase<T>::update_crysfft_tables()
{
    // In place, while no thread runs a propagator or stress loop
    const std::array<double, 6> cell = crysfft_cell(*cb);
    for (auto& [coeff, table] : crysfft_boltz_)
        crysfft_->build_boltzmann(table, cell, coeff);
    for (auto& [key, table] : crysfft_multipliers_)
        crysfft_->build_multiplier(table, cell, key.first, key.second);
}

//------------------------------------------------------------------------------
// Update Laplacian operator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::update_laplacian_operator()
{
    try
    {
        // The CrysFFT tables are rebuilt now, and so are the Pseudo tables when a
        // full-grid path (propagator steps or stress) uses them, that is unless the stress
        // takes the CrysFFT fast path, which implies the CrysFFT steps; otherwise
        // prepare_stress() computes them for the cell in use if the off-diagonal flag
        // turns the stress to the full grid
        pseudo_current_ = false;
        if constexpr (std::is_same_v<T, double>)
        {
            if (use_crysfft())
                update_crysfft_tables();
        }
        if (!crysfft_stress_fast_path())
            update_pseudo();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::update_pseudo()
{
    // The stress basis once a stress is computed (prepare_stress()), as on CUDA
    pseudo->update(cb->get_dx(), cb->get_recip_metric(), stress_basis_);
    pseudo_current_ = true;
}

//------------------------------------------------------------------------------
// Compute single segment stress
//------------------------------------------------------------------------------
template <typename T>
bool CpuSolverPseudoBase<T>::crysfft_stress_fast_path() const
{
    // Only the engines with stress multipliers, Pmmm and recursive 3m, have a stress
    // fast path (ObliqueZ has none). The fast path computes the diagonal sums V₁₁, V₂₂,
    // V₃₃ only, which is fine because the eligible crystal systems are orthogonal and
    // never optimize angles. If angle derivatives are forced (force_off_diagonal_stress_),
    // the standard path computes the cross terms instead of returning zeros for them.
    // (An engine exists only for a space group in a 3D periodic box, see init_crysfft().)
    return crysfft_ != nullptr && crysfft_->has_multipliers() && cb->is_orthogonal()
        && !this->force_off_diagonal_stress_;
}

template <typename T>
void CpuSolverPseudoBase<T>::prepare_stress()
{
    // The Pseudo tables of the cell when the off-diagonal flag has turned the stress
    // to the full grid since the last cell change (see update_laplacian_operator()),
    // with the stress basis from the first stress on
    if (!stress_basis_)
    {
        stress_basis_ = true;
        pseudo_current_ = false;
    }
    if (!crysfft_stress_fast_path() && !pseudo_current_)
        update_pseudo();

    // [real-only feature] CrysFFT folds real fields on the physical sub-grid.
    if constexpr (std::is_same_v<T, double>)
    {
        if (!crysfft_stress_fast_path())
            return;

        // The multipliers of each block's monomer type, full and (discrete chains)
        // half bonds, one per axis group for all threads: the missing ones are built
        // here, before the stress loop, and update_crysfft_tables() rebuilds them
        // for a new cell
        const std::array<double, 6> cell = crysfft_cell(*cb);
        for (const auto& key : crysfft_multiplier_keys(*molecules, *space_group_, stress_has_bond_factor()))
        {
            if (crysfft_multipliers_.contains(key))
                continue;
            std::vector<double> table;
            crysfft_->build_multiplier(table, cell, key.first, key.second);
            crysfft_multipliers_.emplace(key, std::move(table));
        }
    }
}

template <typename T>
bool CpuSolverPseudoBase<T>::compute_stress_crysfft(
    const T* q_1, const T* q_2, double bond_length_sq, std::vector<T>& stress)
{
    // [real-only feature] CrysFFT folds real fields on the physical sub-grid.
    if constexpr (!std::is_same_v<T, double>)
    {
        (void)q_1; (void)q_2; (void)bond_length_sq; (void)stress;
        return false;
    }
    else
    {
        if (!crysfft_stress_fast_path())
            return false;

        const int M_full = cb->get_total_grid();
        const int M_reduced = n_basis_;
        const int M_phys = crysfft_->get_M_physical();
        const auto& orbit_counts = space_group_->get_orbit_counts();

        WorkArrays& work = work_arrays();
        std::vector<double>& phys_q2 = work.phys_in;
        std::vector<double>& phys_tmp = work.phys_out;
        phys_q2.resize(M_phys);
        phys_tmp.resize(M_phys);

        // q_2 on the physical grid, once for all axis groups (diffusion keeps its
        // input); q_1 is summed in the reduced basis. Physical-grid point i holds
        // reduced-basis point crysfft_reduced_indices_[i], or i with the identity map
        const int* index = crysfft_reduced_indices_.data();
        if (crysfft_identity_map_)
            std::copy(q_2, q_2 + M_phys, phys_q2.begin());
        else
            for (int i = 0; i < M_phys; ++i)
                phys_q2[i] = q_2[index[i]];

        // sums[d] = Σ_r n_orbit(r) q1(r) [F⁻¹ k_d² ĝ F q2](r) over the reduced basis, one
        // multiplier Σ_{d in group} k_d² per group of equivalent axes, shared equally by
        // the axes of the group (see crysfft_stress_axis_groups), built by prepare_stress()
        // with the bond factor for discrete chains
        double sums[3] = {0.0, 0.0, 0.0};
        for (const auto& key : crysfft_stress_keys(*molecules, *space_group_, bond_length_sq, stress_has_bond_factor()))
        {
            const int axes = key.first;
            const auto multiplier = crysfft_multipliers_.find(key);
            if (multiplier == crysfft_multipliers_.end())
                throw_with_line_number("The CrysFFT stress multipliers of b^2 = " + std::to_string(bond_length_sq) +
                    " were not built (see prepare_stress()).");
            crysfft_->diffusion(phys_q2.data(), phys_tmp.data(), multiplier->second);

            // Back to the reduced basis: the physical grid itself with the identity map, else
            // a scatter, which writes every reduced-basis point (plan_crysfft())
            const double* reduced_tmp = phys_tmp.data();
            if (!crysfft_identity_map_)
            {
                work.reduced.resize(M_reduced);
                for (int i = 0; i < M_phys; ++i)
                    work.reduced[index[i]] = phys_tmp[i];
                reduced_tmp = work.reduced.data();
            }
            double sum = 0.0;
            for (int i = 0; i < M_reduced; ++i)
                sum += static_cast<double>(orbit_counts[i]) * q_1[i] * reduced_tmp[i];
            const int n_axes = ((axes & 1) != 0) + ((axes & 2) != 0) + ((axes & 4) != 0);
            for (int d = 0; d < 3; ++d)
                if (axes & (1 << d))
                    sums[d] = sum / n_axes;
        }

        // The CrysFFT multipliers are Cartesian k_d² = (2πm_d/L_d)², while the
        // standard path accumulates deformation-vector components v_d² = k_d²/L_d²
        // (see fill_periodic_stress_basis in Pseudo.cpp). Divide by L_d²
        // so both paths return the same V_dd.
        const std::vector<double> lx = cb->get_lx();
        for (int d = 0; d < 3; ++d)
        {
            const double inv_L2 = 1.0 / (lx[d] * lx[d]);
            stress[d] = bond_length_sq * M_full * sums[d] * inv_L2;
        }
        return true;
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::accumulate_stress(
    const double* qk_1, const double* qk_2, double bond_length_sq,
    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const
{
    const int M_COMPLEX = pseudo->get_total_complex_grid();
    const std::complex<double>* qk_1_complex = reinterpret_cast<const std::complex<double>*>(qk_1);
    const std::complex<double>* qk_2_complex = reinterpret_cast<const std::complex<double>*>(qk_2);
    const int* negative_k_idx = pseudo->get_negative_frequency_mapping();

    // v⊗v dyad components (deformation vector v = 2π g⁻¹ m, units 1/L²): the diagonal ones of
    // the box axes (periodic 1D: the Miller index is mapped to m1, so the stress is in
    // fourier_basis_x), then the cross terms of a non-orthogonal periodic box (see the note in
    // compute_single_segment_stress), V₁₂ stored in index 2 in 2D; walls need an orthogonal cell
    std::array<const double*, 6> basis{};
    std::array<int, 6> slot{};
    int n_components = 0;
    const std::array<const double*, 3> diagonal = {
        pseudo->get_fourier_basis_x(), pseudo->get_fourier_basis_y(), pseudo->get_fourier_basis_z()};
    for (int d = 0; d < dim_; ++d)
    {
        basis[n_components] = diagonal[d];
        slot[n_components++] = d;
    }
    if (is_periodic_ && !is_orthogonal && dim_ >= 2)
    {
        basis[n_components] = pseudo->get_fourier_basis_xy();   // → dH/dγ
        slot[n_components++] = (dim_ == 2) ? 2 : 3;
        if (dim_ == 3)
        {
            basis[n_components] = pseudo->get_fourier_basis_xz();   // → dH/dβ
            slot[n_components++] = 4;
            basis[n_components] = pseudo->get_fourier_basis_yz();   // → dH/dα
            slot[n_components++] = 5;
        }
    }

    // [type-specific algorithm] Periodic boxes: real fields store the Hermitian half-spectrum,
    // so the conjugate product is formed directly; complex fields need the -k map. Walls: real
    // DCT/DST coefficients, for complex fields those of the real and imaginary parts as complex
    // pairs, whose product needs no conjugate (the basis functions are real). The per-mode
    // Parseval weights are folded into the fourier_basis tables (Pseudo), so no flat factor is
    // applied here.
    auto accumulate = [&](auto&& kernel) {
        switch (n_components)
        {
            case 1: accumulate_dyads<1, T>(M_COMPLEX, kernel, basis, slot, stress); break;
            case 2: accumulate_dyads<2, T>(M_COMPLEX, kernel, basis, slot, stress); break;
            case 3: accumulate_dyads<3, T>(M_COMPLEX, kernel, basis, slot, stress); break;
            case 6: accumulate_dyads<6, T>(M_COMPLEX, kernel, basis, slot, stress); break;
            default: throw_with_line_number("Unsupported number of stress components: " + std::to_string(n_components));
        }
    };
    if (is_periodic_)
    {
        accumulate([&](int i) -> T {
            const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
            if constexpr (std::is_same<T, double>::value)
                return bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
            else
                return bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[negative_k_idx[i]];
        });
    }
    else
    {
        accumulate([&](int i) -> T {
            const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
            if constexpr (std::is_same<T, double>::value)
                return bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];
            else
                return bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[i];
        });
    }
}

template <typename T>
const double* CpuSolverPseudoBase<T>::get_stress_boltz_bond(
    std::string /*monomer_type*/, bool /*is_half_bond_length*/) const
{
    return nullptr;   // continuous chains: the bond factor is already in the propagator
}

/**
 * @brief Compute stress contribution from a single bond/segment.
 *
 * Uses v⊗v dyad product components stored in fourier_basis arrays, where
 * v = 2π g⁻¹ m is the deformation vector (units 1/L², NOT the Cartesian
 * wavevector — see fill_periodic_stress_basis in Pseudo.cpp):
 * - fourier_basis_x = v₁², fourier_basis_y = v₂², fourier_basis_z = v₃²
 * - fourier_basis_xy = v₁v₂, fourier_basis_xz = v₁v₃, fourier_basis_yz = v₂v₃
 *
 * Returns the accumulated sums [V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃] where
 * V_ij = Σ_k kernel(k)·v_i·v_j. The caller (compute_stress in the
 * computation classes) converts these to lattice-parameter derivatives
 * [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα] via the metric tensor.
 *
 * @see docs/theory/StressTensor.md for the derivation
 */
template <typename T>
std::vector<T> CpuSolverPseudoBase<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int N_STRESS = 6;  // v⊗v dyad sums: V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃
        const double bond_length_sq = stress_bond_length_sq(*molecules, monomer_type, is_half_bond_length);

        std::vector<T> stress(N_STRESS, 0.0);

        // CrysFFT fast path (reduced grid, no full-grid FFT and no Pseudo tables)
        if (compute_stress_crysfft(q_1, q_2, bond_length_sq, stress))
            return stress;

        // Chain-model-specific Boltzmann bond factor (nullptr for continuous chains)
        const double* _boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);

        WorkArrays& work = work_arrays();
        std::vector<double>& qk_1 = work.k_1;
        std::vector<double>& qk_2 = work.k_2;
        qk_1.resize(spectrum_size());
        qk_2.resize(spectrum_size());

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-term sums (V₁₂, V₁₃, V₂₃)
        // are not computed: angle optimization is inactive for orthogonal crystal
        // systems, so those slots are simply left at zero. (For generic asymmetric
        // fields the true values need not vanish, but they are never consumed.)
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        //
        // Explicit opt-in: force_off_diagonal_stress_ (set when angle optimization
        // is active) disables the skip so that angle derivatives are available
        // even when the box is at exactly 90° angles. The skip remains the default.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal() && !this->force_off_diagonal_stress_;

        // Transform to Fourier space, with a space group from the full grid of this thread
        // (this function runs in the OpenMP block loops of compute_stress), expanded from
        // the reduced basis for one propagator after the other, and accumulate
        // Σ_k kernel(k) v_i v_j
        transform_forward(this->expand_to_full_grid(q_1, work.q_full), qk_1.data());
        transform_forward(this->expand_to_full_grid(q_2, work.q_full), qk_2.data());
        accumulate_stress(qk_1.data(), qk_2.data(), bond_length_sq, _boltz_bond, is_orthogonal, stress);

        return stress;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoBase);
