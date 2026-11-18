/**
 * @file CpuSolverPseudoBase.cpp
 * @brief Implementation of common pseudo-spectral solver functionality.
 *
 * Implements shared code between CpuSolverPseudoRQM4 and
 * CpuSolverPseudoDiscrete, including:
 *
 * - FFT object initialization and cleanup
 * - Laplacian operator updates
 * - Stress computation with chain-model-specific factors
 *
 * Transform dispatch goes through the FFT<T> interface (FftwFFT, MklFFT or MakhoulFFT).
 *
 * @see CpuSolverPseudoBase.h for interface documentation
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <set>

#include "CpuSolverPseudoBase.h"
#include "CrysFFTSelector.h"
#include "CrysFFTBasisMapping.h"
#include "CrysFFTPmmm.h"
#ifdef USE_CPU_FFTW
#include "FftwCrysFFTRecursive3m.h"
#include "FftwCrysFFTObliqueZ.h"
#endif
#ifdef USE_CPU_MKL
#include "MklCrysFFTRecursive3m.h"
#include "MklCrysFFTObliqueZ.h"
#endif

//------------------------------------------------------------------------------
// Initialize shared components
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::init_shared(ComputationBox<T>* cb, Molecules* molecules, FFTBackend backend)
{
    this->cb = cb;
    this->molecules = molecules;
    this->chain_model = molecules->get_model_name();
    this->fft_backend_ = backend;

    // Check if all BCs are periodic
    auto bc_vec = cb->get_boundary_conditions();

    // Validate that both sides of each direction have matching BC
    // (required for pseudo-spectral: FFT/DCT/DST apply to entire direction)
    int dim = cb->get_dim();
    for (int d = 0; d < dim; ++d)
    {
        if (bc_vec[2*d] != bc_vec[2*d + 1])
        {
            throw_with_line_number("Pseudo-spectral method requires matching boundary conditions on both sides of each direction. "
                "Direction " + std::to_string(d) + " has mismatched BCs. Use real-space method for mixed BCs.");
        }
    }
    is_periodic_ = true;
    for (const auto& b : bc_vec)
    {
        if (b != BoundaryCondition::PERIODIC)
        {
            is_periodic_ = false;
            break;
        }
    }

    // Create FFT object using factory (handles all BCs and backends)
    // All simulations are conducted in fixed dimensions
    dim_ = cb->get_dim();

    // Extract one BC per dimension and create appropriate FFT
    if (dim_ == 3)
    {
        std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
        fft_ = createFFT<T, 3>(nx_arr, bc_arr, backend);
    }
    else if (dim_ == 2)
    {
        std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
        std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
        fft_ = createFFT<T, 2>(nx_arr, bc_arr, backend);
    }
    else if (dim_ == 1)
    {
        std::array<int, 1> nx_arr = {cb->get_nx(0)};
        std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
        fft_ = createFFT<T, 1>(nx_arr, bc_arr, backend);
    }
    else
    {
        throw_with_line_number("Pseudo-spectral solver supports 1D, 2D, or 3D only, got dim = "
                               + std::to_string(dim_) + ".");
    }

    // The Fourier-space tables of every distinct contour step, one boundary condition per
    // dimension; update_laplacian_operator() computes them for the cell
    molecules->finalize_contour_length_mapping();
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
        ds_values_[ds_idx] = mapping.get_ds_from_index(ds_idx);
    std::vector<BoundaryCondition> bc_per_dim;
    for (int d = 0; d < dim_; ++d)
        bc_per_dim.push_back(bc_vec[2 * d]);
    pseudo = std::make_unique<Pseudo<T>>(molecules->get_bond_lengths(), bc_per_dim, cb->get_nx(), ds_values_);
}

//------------------------------------------------------------------------------
// Space group / CrysFFT setup
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::reset_crysfft()
{
    crysfft_.reset();
    crysfft_boltz_.clear();
    crysfft_multipliers_.clear();
    crysfft_reduced_indices_.clear();
    crysfft_identity_map_ = false;
}

template <typename T>
void CpuSolverPseudoBase<T>::adopt_crysfft_mapping(const CrysFFTBasisMapping& map, bool physical_basis_enabled)
{
    if (physical_basis_enabled)
    {
        // The fields already live on the physical grid: no index map needed.
        crysfft_reduced_indices_.clear();
        crysfft_identity_map_ = true;
    }
    else
    {
        crysfft_reduced_indices_ = map.phys_to_reduced;
        crysfft_identity_map_ = map.identity;
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::set_space_group(SpaceGroup* sg)
{
    space_group_ = sg;

    if (sg == nullptr)
    {
        if constexpr (std::is_same_v<T, double>)
            reset_crysfft();
        return;
    }

    // [real-only feature] The CrysFFT fast paths fold real fields on the
    // physical sub-grid; complex fields always use the full-grid reduced basis.
    if constexpr (std::is_same_v<T, double>)
    {
        reset_crysfft();
        if (dim_ == 3)
            init_crysfft();
    }

    // The CrysFFT maps above are snapshots: freeze the physical basis
    sg->mark_in_use();
}

template <typename T>
void CpuSolverPseudoBase<T>::init_crysfft()
{
    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    // Engine choice and its basis map, shared with the CUDA solver
    const CrysFFTPlan plan = plan_crysfft(
        space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), cb->is_z_axis_orthogonal());
    if (plan.mode == CrysFFTChoice::None)
        return;

    adopt_crysfft_mapping(plan.map, plan.physical_basis);

    if (plan.mode == CrysFFTChoice::Recursive3m)
    {
#ifdef USE_CPU_MKL
        if (fft_backend_ == FFTBackend::MKL)
            crysfft_ = std::make_unique<MklCrysFFTRecursive3m>(nx_logical, plan.m3_translations);
#endif
#ifdef USE_CPU_FFTW
        if (fft_backend_ == FFTBackend::FFTW)
            crysfft_ = std::make_unique<FftwCrysFFTRecursive3m>(nx_logical, plan.m3_translations);
#endif
    }
    else if (plan.mode == CrysFFTChoice::PmmmDct)
    {
        crysfft_ = std::make_unique<CrysFFTPmmm>(nx_logical, fft_backend_);
    }
    else if (plan.mode == CrysFFTChoice::ObliqueZ)
    {
#ifdef USE_CPU_MKL
        if (fft_backend_ == FFTBackend::MKL)
            crysfft_ = std::make_unique<MklCrysFFTObliqueZ>(nx_logical);
#endif
#ifdef USE_CPU_FFTW
        if (fft_backend_ == FFTBackend::FFTW)
            crysfft_ = std::make_unique<FftwCrysFFTObliqueZ>(nx_logical);
#endif
    }
    if (!crysfft_)
        throw_with_line_number("No CrysFFT engine was built for the selected FFT library.");

    // The Boltzmann tables of the steps of each block's monomer type and contour step
    // (not of every combination: a recursive 3m table holds 16 physical-grid arrays)
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    for (int p = 0; p < molecules->get_n_polymer_types(); ++p)
        for (const Block& block : molecules->get_polymer(p).get_blocks())
            for (double coeff : crysfft_block_coeffs(block.monomer_type, mapping.get_ds_index(block.contour_length)))
                crysfft_boltz_.try_emplace(coeff);
    update_crysfft_tables();
}

//------------------------------------------------------------------------------
// ds value helpers
//------------------------------------------------------------------------------
template <typename T>
double CpuSolverPseudoBase<T>::get_ds_value(int ds_index) const
{
    auto it = ds_values_.find(ds_index);
    if (it == ds_values_.end())
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " not registered in solver.");
    return it->second;
}

template <typename T>
double CpuSolverPseudoBase<T>::get_effective_diffusion_coeff(
    const std::string& monomer_type, int ds_index, bool half_step) const
{
    double ds = get_ds_value(ds_index);
    if (half_step)
        ds *= 0.5;

    auto& bond_lengths = molecules->get_bond_lengths();
    auto it = bond_lengths.find(monomer_type);
    if (it == bond_lengths.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");
    double bond_length_sq = it->second * it->second;
    return bond_length_sq * ds / 6.0;
}

//------------------------------------------------------------------------------
// update_dw() helper: fields on the grid the Boltzmann factors live on
//------------------------------------------------------------------------------
template <typename T>
int CpuSolverPseudoBase<T>::expand_w_for_update_dw(
    const std::map<std::string, const T*>& w_input,
    std::map<std::string, std::vector<T>>& w_full_cache,
    std::map<std::string, const T*>& w_use) const
{
    const int M_full = cb->get_total_grid();
    const int M_reduced = (space_group_ != nullptr)
        ? space_group_->get_n_reduced_basis()
        : M_full;
    const bool use_reduced_basis = (space_group_ != nullptr);
    const bool use_crysfft_path = (use_crysfft() && space_group_ != nullptr);

    w_full_cache.clear();
    w_use.clear();
    if (use_reduced_basis && !use_crysfft_path)
    {
        // Standard FFT path: exp_dw lives on the full grid, so expand the
        // reduced-basis fields. [real-only feature]
        if constexpr (std::is_same_v<T, double>)
        {
            for (const auto& item : w_input)
            {
                const std::string& monomer_type = item.first;
                std::vector<T>& w_full = w_full_cache[monomer_type];
                w_full.resize(M_full);
                space_group_->from_reduced_basis(item.second, w_full.data(), 1);
                w_use[monomer_type] = w_full.data();
            }
        }
        else
        {
            throw_with_line_number("Space group reduced basis is only supported for real fields.");
        }
    }
    else
    {
        // No space group (full grid), or CrysFFT path (exp_dw stays in the reduced basis)
        for (const auto& item : w_input)
            w_use[item.first] = item.second;
    }

    return use_crysfft_path ? M_reduced : M_full;
}

//------------------------------------------------------------------------------
// CrysFFT grid helpers (reduced basis <-> physical grid)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::fill_crysfft_from_reduced(const double* reduced_in, double* phys_out) const
{
    if (!use_crysfft())
        throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");

    const int M_phys = get_crysfft_physical_size();
    if (crysfft_identity_map_)
    {
        std::memcpy(phys_out, reduced_in, sizeof(double) * M_phys);
        return;
    }
    for (int i = 0; i < M_phys; ++i)
    {
        phys_out[i] = reduced_in[crysfft_reduced_indices_[i]];
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::reduce_crysfft_to_reduced(const double* phys_in, double* reduced_out) const
{
    if (!use_crysfft())
        throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");

    const int M_phys = get_crysfft_physical_size();
    if (crysfft_identity_map_)
    {
        std::memcpy(reduced_out, phys_in, sizeof(double) * M_phys);
        return;
    }

    // The CrysFFT path is taken only when the physical grid reaches every reduced-basis
    // point (crysfft_mapping_usable), so the scatter writes all of reduced_out
    for (int i = 0; i < M_phys; ++i)
    {
        reduced_out[crysfft_reduced_indices_[i]] = phys_in[i];
    }
}

//------------------------------------------------------------------------------
// CrysFFT helpers
//------------------------------------------------------------------------------
template <typename T>
int CpuSolverPseudoBase<T>::get_crysfft_physical_size() const
{
    if (!crysfft_)
        throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
    return crysfft_->get_M_physical();
}

template <typename T>
std::pair<double*, double*> CpuSolverPseudoBase<T>::crysfft_grids(WorkArrays& work) const
{
    const int M_phys = get_crysfft_physical_size();
    if (static_cast<int>(work.phys_in.size()) != M_phys)
    {
        work.phys_in.resize(M_phys);
        work.phys_out.resize(M_phys);
    }
    return {work.phys_in.data(), work.phys_out.data()};
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_step(double* q_in, double* q_out, double coeff,
                                          const double* pre, const double* post) const
{
    const auto boltz = crysfft_boltz_.find(coeff);
    if (boltz == crysfft_boltz_.end())
        throw_with_line_number("No CrysFFT table was built for the step coefficient " + std::to_string(coeff) +
            " (see crysfft_block_coeffs()).");

    // The factors live in the reduced basis: physical-grid point i holds reduced-basis point
    // crysfft_reduced_indices_[i], or i with the identity map
    const int M_phys = get_crysfft_physical_size();
    const int* reduced_index = crysfft_identity_map_ ? nullptr : crysfft_reduced_indices_.data();
    if (pre != nullptr)
        for (int i = 0; i < M_phys; ++i)
            q_in[i] *= pre[reduced_index ? reduced_index[i] : i];
    crysfft_->diffusion(q_in, q_out, boltz->second);
    if (post != nullptr)
        for (int i = 0; i < M_phys; ++i)
            q_out[i] *= post[reduced_index ? reduced_index[i] : i];
}

template <typename T>
std::array<double, 6> CpuSolverPseudoBase<T>::crysfft_cell() const
{
    const auto lx = cb->get_lx();
    const auto angles = cb->get_angles();
    return {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};
}

template <typename T>
void CpuSolverPseudoBase<T>::update_crysfft_tables()
{
    // In place, while no thread runs a propagator or stress loop
    const std::array<double, 6> cell = crysfft_cell();
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
        // The CrysFFT tables are rebuilt now, and so are the Pseudo tables when the
        // full-grid path uses them; otherwise prepare_stress() computes them for the
        // cell in use if the off-diagonal flag turns the stress to the full grid
        pseudo_current_ = false;
        if constexpr (std::is_same_v<T, double>)
        {
            if (use_crysfft())
                update_crysfft_tables();
        }
        if (full_grid_path())
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
    pseudo->update(cb->get_dx(), cb->get_recip_metric());
    pseudo_current_ = true;
}

template <typename T>
bool CpuSolverPseudoBase<T>::full_grid_path() const
{
    // The stress fast path implies the CrysFFT propagator steps
    return !crysfft_stress_fast_path();
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
    return crysfft_ != nullptr && crysfft_->has_multipliers()
        && space_group_ != nullptr && is_periodic_ && dim_ == 3 && cb->is_orthogonal()
        && !this->force_off_diagonal_stress_;
}

template <typename T>
double CpuSolverPseudoBase<T>::stress_bond_length_sq(const std::string& monomer_type, bool is_half_bond_length) const
{
    const auto& bond_lengths = this->molecules->get_bond_lengths();
    auto bond_it = bond_lengths.find(monomer_type);
    if (bond_it == bond_lengths.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");

    double bond_length_sq = bond_it->second * bond_it->second;
    if (is_half_bond_length)
        bond_length_sq *= 0.5;
    return bond_length_sq;
}

template <typename T>
std::vector<std::pair<int, double>> CpuSolverPseudoBase<T>::crysfft_stress_keys(
    double bond_length_sq, bool include_bond_factor) const
{
    // Discrete chains carry the bond factor ĝ(k) = exp(-b² k² Δs / 6)
    const double coeff = include_bond_factor ? bond_length_sq * molecules->get_global_ds() / 6.0 : 0.0;
    std::vector<std::pair<int, double>> keys;
    for (int axes : crysfft_stress_axis_groups(space_group_->get_crystal_system()))
        keys.emplace_back(axes, coeff);
    return keys;
}

template <typename T>
void CpuSolverPseudoBase<T>::prepare_stress()
{
    // The Pseudo tables of the cell when the off-diagonal flag has turned the stress
    // to the full grid since the last cell change (see full_grid_path())
    if (full_grid_path() && !pseudo_current_)
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
        const std::array<double, 6> cell = crysfft_cell();
        std::set<std::string> monomer_types;
        for (int p = 0; p < molecules->get_n_polymer_types(); ++p)
            for (const Block& block : molecules->get_polymer(p).get_blocks())
                monomer_types.insert(block.monomer_type);
        for (const std::string& monomer_type : monomer_types)
        {
            for (const bool half : {false, true})
            {
                const bool include_bond_factor = (get_stress_boltz_bond(monomer_type, half) != nullptr);
                if (half && !include_bond_factor)
                    continue;   // continuous chains: one multiplier per axis group
                for (const auto& key : crysfft_stress_keys(stress_bond_length_sq(monomer_type, half), include_bond_factor))
                {
                    if (crysfft_multipliers_.contains(key))
                        continue;
                    std::vector<double> table;
                    crysfft_->build_multiplier(table, cell, key.first, key.second);
                    crysfft_multipliers_.emplace(key, std::move(table));
                }
            }
        }
    }
}

template <typename T>
bool CpuSolverPseudoBase<T>::compute_stress_crysfft(
    const T* q_1, const T* q_2, double bond_length_sq, const double* boltz_bond, std::vector<T>& stress)
{
    // [real-only feature] CrysFFT folds real fields on the physical sub-grid.
    if constexpr (!std::is_same_v<T, double>)
    {
        (void)q_1; (void)q_2; (void)bond_length_sq; (void)boltz_bond; (void)stress;
        return false;
    }
    else
    {
        if (!crysfft_stress_fast_path())
            return false;

        const int M_full = cb->get_total_grid();
        const int M_reduced = space_group_->get_n_reduced_basis();
        const int M_phys = get_crysfft_physical_size();
        const auto& orbit_counts = space_group_->get_orbit_counts();

        WorkArrays& work = work_arrays();
        std::vector<double>& phys_q2 = work.phys_1;
        std::vector<double>& phys_tmp = work.phys_2;
        std::vector<double>& reduced_tmp = work.reduced;
        if (static_cast<int>(phys_q2.size()) != M_phys)
        {
            phys_q2.resize(M_phys);
            phys_tmp.resize(M_phys);
        }
        if (static_cast<int>(reduced_tmp.size()) != M_reduced)
            reduced_tmp.resize(M_reduced);

        // Reduced -> physical grid (q_1 is summed in the reduced basis)
        fill_crysfft_from_reduced(q_2, phys_q2.data());

        // sums[d] = Σ_r n_orbit(r) q1(r) [F⁻¹ k_d² ĝ F q2](r) over the reduced basis, one
        // multiplier Σ_{d in group} k_d² per group of equivalent axes, shared equally by
        // the axes of the group (see crysfft_stress_axis_groups), built by prepare_stress()
        // with the bond factor for discrete chains
        double sums[3] = {0.0, 0.0, 0.0};
        for (const auto& key : crysfft_stress_keys(bond_length_sq, boltz_bond != nullptr))
        {
            const int axes = key.first;
            const auto multiplier = crysfft_multipliers_.find(key);
            if (multiplier == crysfft_multipliers_.end())
                throw_with_line_number("The CrysFFT stress multipliers of b^2 = " + std::to_string(bond_length_sq) +
                    " were not built (see prepare_stress()).");
            crysfft_->apply_multiplier(phys_q2.data(), phys_tmp.data(), multiplier->second);
            reduce_crysfft_to_reduced(phys_tmp.data(), reduced_tmp.data());
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
        // (see update_weighted_fourier_basis_periodic_impl in Pseudo.cpp). Divide by L_d²
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
void CpuSolverPseudoBase<T>::accumulate_stress_periodic(
    const double* qk_1, const double* qk_2, double bond_length_sq,
    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const
{
    const int DIM = dim_;
    const int M_COMPLEX = pseudo->get_total_complex_grid();
    const std::complex<double>* qk_1_complex = reinterpret_cast<const std::complex<double>*>(qk_1);
    const std::complex<double>* qk_2_complex = reinterpret_cast<const std::complex<double>*>(qk_2);
    const int* negative_k_idx = pseudo->get_negative_frequency_mapping();

    // v⊗v dyad components (deformation vector v = 2π g⁻¹ m, units 1/L²)
    const double* kk_xx = pseudo->get_fourier_basis_x();
    const double* kk_yy = pseudo->get_fourier_basis_y();
    const double* kk_zz = pseudo->get_fourier_basis_z();
    const double* kk_xy = pseudo->get_fourier_basis_xy();
    const double* kk_xz = pseudo->get_fourier_basis_xz();
    const double* kk_yz = pseudo->get_fourier_basis_yz();

    // [type-specific algorithm] Real fields store the Hermitian half-spectrum,
    // so the conjugate product is formed directly; complex fields need the -k map.
    auto kernel = [&](int i) -> T
    {
        const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
        if constexpr (std::is_same<T, double>::value)
            return bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
        else
            return bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[negative_k_idx[i]];
    };

    if (DIM == 3)
    {
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const T coeff = kernel(i);
            stress[0] += coeff * kk_xx[i];  // V₁₁
            stress[1] += coeff * kk_yy[i];  // V₂₂
            stress[2] += coeff * kk_zz[i];  // V₃₃
        }
        // Cross terms only for non-orthogonal boxes (see the note in compute_single_segment_stress)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                const T coeff = kernel(i);
                stress[3] += coeff * kk_xy[i];  // V₁₂ (→ dH/dγ)
                stress[4] += coeff * kk_xz[i];  // V₁₃ (→ dH/dβ)
                stress[5] += coeff * kk_yz[i];  // V₂₃ (→ dH/dα)
            }
        }
    }
    else if (DIM == 2)
    {
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const T coeff = kernel(i);
            stress[0] += coeff * kk_xx[i];  // V₁₁
            stress[1] += coeff * kk_yy[i];  // V₂₂
        }
        // Cross term only for non-orthogonal boxes (as in 3D and on the CUDA side)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                stress[2] += kernel(i) * kk_xy[i];  // V₁₂ (stored in index 2 for 2D)
        }
    }
    else if (DIM == 1)
    {
        // Periodic 1D: the Miller index is mapped to m1, so the stress is in fourier_basis_x
        for (int i = 0; i < M_COMPLEX; ++i)
            stress[0] += kernel(i) * kk_xx[i];
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::accumulate_stress_nonperiodic(
    const double* qk_1, const double* qk_2, double bond_length_sq,
    const double* boltz_bond, std::vector<T>& stress) const
{
    const int DIM = dim_;
    const int M_COMPLEX = pseudo->get_total_complex_grid();
    const double* kk_xx = pseudo->get_fourier_basis_x();
    const double* kk_yy = pseudo->get_fourier_basis_y();
    const double* kk_zz = pseudo->get_fourier_basis_z();

    // Real coefficients from DCT/DST; for complex fields those of the real and
    // imaginary parts as complex pairs, whose product needs no conjugate (the
    // basis functions are real). The per-mode Parseval weights (2 for
    // interior modes, 1 for the DCT m=0 / DST k=n special modes, composed
    // across dimensions) are folded into the fourier_basis tables by
    // Pseudo::update_nonperiodic, mirroring the periodic
    // path, so no flat factor is applied here.
    const std::complex<double>* qk_1_complex = reinterpret_cast<const std::complex<double>*>(qk_1);
    const std::complex<double>* qk_2_complex = reinterpret_cast<const std::complex<double>*>(qk_2);
    auto kernel = [&](int i) -> T
    {
        const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
        if constexpr (std::is_same<T, double>::value)
            return bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];
        else
            return bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[i];
    };

    // V₁₁, V₂₂, V₃₃ of the box axes; the cross terms are zero, since walls need an orthogonal cell
    for (int i = 0; i < M_COMPLEX; ++i)
    {
        const T coeff = kernel(i);
        stress[0] += coeff * kk_xx[i];
        if (DIM >= 2)
            stress[1] += coeff * kk_yy[i];
        if (DIM == 3)
            stress[2] += coeff * kk_zz[i];
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
 * wavevector — see update_weighted_fourier_basis_periodic_impl in Pseudo.cpp):
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
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        const double bond_length_sq = stress_bond_length_sq(monomer_type, is_half_bond_length);

        std::vector<T> stress(N_STRESS, 0.0);

        // Chain-model-specific Boltzmann bond factor (nullptr for continuous chains)
        const double* _boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);

        // CrysFFT fast path (reduced grid, no full-grid FFT)
        if (compute_stress_crysfft(q_1, q_2, bond_length_sq, _boltz_bond, stress))
            return stress;

        const int coeff_size = complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;
        WorkArrays& work = work_arrays();
        std::vector<double>& qk_1 = work.k_1;
        std::vector<double>& qk_2 = work.k_2;
        qk_1.resize(coeff_size);
        qk_2.resize(coeff_size);

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

        // Expand reduced basis → full grid. Per-thread work arrays: this function
        // is called from OpenMP parallel regions (block loops in compute_stress),
        // so no shared scratch space may be used here.
        T* q_1_full = q_1;
        T* q_2_full = q_2;
        std::vector<T>& q_full_1_local = work.pair_1;
        std::vector<T>& q_full_2_local = work.pair_2;
        if (space_group_ != nullptr)
        {
            const int M_full = cb->get_total_grid();
            if (static_cast<int>(q_full_1_local.size()) != M_full)
            {
                q_full_1_local.resize(M_full);
                q_full_2_local.resize(M_full);
            }
            if constexpr (std::is_same<T, double>::value)
            {
                space_group_->from_reduced_basis(q_1, q_full_1_local.data(), 1);
                space_group_->from_reduced_basis(q_2, q_full_2_local.data(), 1);
            }
            else // complex<double>: expand as 2 interleaved real fields
            {
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_1),
                    reinterpret_cast<double*>(q_full_1_local.data()), 2);
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_2),
                    reinterpret_cast<double*>(q_full_2_local.data()), 2);
            }
            q_1_full = q_full_1_local.data();
            q_2_full = q_full_2_local.data();
        }

        // Transform to Fourier space and accumulate Σ_k kernel(k) v_i v_j
        transform_forward(q_1_full, qk_1.data());
        transform_forward(q_2_full, qk_2.data());
        if (is_periodic_)
            accumulate_stress_periodic(qk_1.data(), qk_2.data(), bond_length_sq, _boltz_bond, is_orthogonal, stress);
        else
            accumulate_stress_nonperiodic(qk_1.data(), qk_2.data(), bond_length_sq, _boltz_bond, stress);

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
