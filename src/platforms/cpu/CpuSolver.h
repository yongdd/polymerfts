/**
 * @file CpuSolver.h
 * @brief Abstract interface for CPU propagator solvers.
 *
 * This header defines the CpuSolver class, which provides a common interface
 * for different propagator solving methods on CPU. The implementations are:
 *
 * - **Pseudo-spectral methods** (rqm4, rk2, ps): Use FFT for diffusion (continuous)
 *   or bond convolution (discrete) in Fourier space
 * - **Real-space methods**: Finite differences with the Crank-Nicolson ADI scheme
 *   (cn-adi2, continuous chains), and the bond of discrete chains summed on the grid (rs)
 *
 * **Chain Models:**
 *
 * - **Continuous chains**: Solve the modified diffusion equation
 *       ∂q/∂s = (b²/6) ∇²q - w(r) q
 *
 * - **Discrete chains**: Solve the Chapman-Kolmogorov integral equation
 *       q(r, n+1) = exp(-w(r)*ds) * ∫ g(r-r') q(r', n) dr'
 *   where g(r) is the bond function. See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * @see CpuSolverPseudoRQM4 for continuous chain pseudo-spectral
 * @see CpuSolverPseudoDiscrete for discrete chain pseudo-spectral
 * @see CpuSolverCNADI for finite difference method
 * @see CpuSolverRealSpaceDiscrete for the bond summed on the grid
 */

#ifndef CPU_SOLVER_H_
#define CPU_SOLVER_H_

#include <cmath>
#include <string>
#include <type_traits>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "SpaceGroup.h"

/**
 * @brief Work arrays W of each calling thread, owned by a solver and released with it.
 *
 * The propagators run in OpenMP threads, so every thread needs its own arrays
 * (thread_local arrays would stay allocated until the threads end). A step
 * function or the stress routine uses them from start to end of one call;
 * nothing nests, so one W per thread is enough.
 */
template <typename W>
class PerThread
{
private:
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::thread::id, W> items_;

public:
    /// The W of the calling thread, created on first use (a reference into a node map stays valid)
    W& get() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_[std::this_thread::get_id()];
    }
};

/**
 * @class CpuSolver
 * @brief Abstract base class for CPU-based propagator solvers.
 *
 * Defines the interface for advancing chain propagators and computing
 * stress. Derived classes implement specific numerical methods.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Operator Splitting (Continuous Chains):**
 *
 * For continuous chains (modified diffusion equation):
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-b²|k|²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Operator Splitting (Discrete Chains):**
 *
 * For discrete chains (Chapman-Kolmogorov equation):
 *     q(n+1) = B^(1/2) · exp(-w·ds) · B^(1/2) · q(n)
 *
 * where B^(1/2) is half-bond convolution with ĝ^(1/2)(k) = exp(-b²|k|²ds/12).
 *
 * **Common Usage Pattern:**
 *
 * @code
 * // Create solver (via derived class), with the tables of the cell; space_group may be nullptr
 * std::unique_ptr<CpuSolver<double>> solver =
 *     std::make_unique<CpuSolverPseudoRQM4<double>>(cb, molecules, space_group, FFTBackend::FFTW);
 *
 * // Tables of a new cell
 * solver->update_laplacian_operator();
 *
 * // Update when fields change
 * solver->update_dw(w_fields);
 *
 * // Advance propagator (contour step of index 0)
 * solver->advance_propagator(q_in, q_out, "A", mask, 0);
 * @endcode
 */
template <typename T>
class CpuSolver
{
public:
    /**
     * @brief Force off-diagonal (cross-term) stress accumulation for orthogonal boxes.
     *
     * By default the stress computation skips V₁₂/V₁₃/V₂₃ for orthogonal
     * boxes (performance optimization). Set to true when angle optimization
     * is active so that angle derivatives are available even when the box
     * starts at exactly 90° angles. Forwarded from
     * PropagatorComputation::set_force_off_diagonal_stress().
     */
    bool force_off_diagonal_stress_ = false;

    /// Virtual destructor.
    virtual ~CpuSolver() = default;

    /**
     * @brief Set the force-off-diagonal-stress flag (see force_off_diagonal_stress_).
     * @param force true to always accumulate cross-term stress sums
     */
    void set_force_off_diagonal_stress(bool force) { force_off_diagonal_stress_ = force; }

    /**
     * @brief Update Laplacian operator for changed box dimensions.
     *
     * Called when box size changes (e.g., during stress relaxation).
     * Recomputes FFT wavenumbers k² or finite difference coefficients.
     *
     * For pseudo-spectral: Updates exp(-b²|k|²ds/6) in Fourier space
     *     - Continuous chains: diffusion propagator
     *     - Discrete chains: bond function ĝ(k)
     * For real-space: Updates tridiagonal matrix coefficients
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Update Boltzmann factors from new potential fields.
     *
     * Computes exp_dw and exp_dw_half from input potential fields.
     * Must be called whenever w fields change (each SCFT iteration).
     *
     * @param w_input Map of potential fields by monomer type
     *                Key: monomer type (e.g., "A", "B")
     *                Value: Pointer to potential field array (size n_grid)
     *
     * @example
     * @code
     * std::map<std::string, const double*> w_fields;
     * w_fields["A"] = w_A;
     * w_fields["B"] = w_B;
     * solver->update_dw(w_fields);
     * @endcode
     */
    virtual void update_dw(std::map<std::string, const T*> w_input) = 0;

    /**
     * @brief Advance propagator by one contour step.
     *
     * Solves the modified diffusion equation for one step ds:
     *
     *     q(r, s+ds) = Propagator[ q(r, s), w(r) ]
     *
     * With a space group (given to the constructor), input/output are in its reduced
     * basis and the solver handles expand/reduce around FFT operations internally.
     *
     * @param q_in        Input propagator at contour s (size n_grid or n_basis)
     * @param q_out       Output propagator at contour s+ds (size n_grid or n_basis)
     * @param monomer_type Monomer type determining segment length and field
     * @param q_mask      Optional mask for impenetrable regions (nullptr if none)
     * @param ds_index    Index of the contour step ds of the block (0-based, from ContourLengthMapping)
     *
     * @note For continuous chains, ds is the contour discretization parameter.
     *       For discrete chains, this advances by one segment.
     *
     * @example
     * @code
     * // Advance the propagator by one A segment with the contour step of index 0
     * solver->advance_propagator(q_current, q_next, "A", nullptr, 0);
     * @endcode
     */
    virtual void advance_propagator(
        T *q_in, T *q_out, std::string monomer_type, const double *q_mask, int ds_index) = 0;

    /**
     * @brief Advance propagator by half bond step (discrete chain model).
     *
     * Used in discrete chain model where the full propagator step is split:
     *
     *     q(n+1) = B^(1/2) · A · B^(1/2) · q(n)
     *
     * where B^(1/2) is the half-bond convolution with ĝ^(1/2)(k) = exp(-b²|k|²ds/12).
     *
     * With a space group (given to the constructor), input/output are in its reduced
     * basis and the solver handles expand/reduce around FFT operations internally.
     *
     * @param q_in        Input propagator (size n_grid or n_basis)
     * @param q_out       Output propagator (size n_grid or n_basis)
     * @param monomer_type Monomer type for segment length
     *
     * @note Only meaningful for discrete chain model. Empty implementation
     *       for continuous chain solvers.
     */
    virtual void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) = 0;

    /**
     * @brief Compute stress contribution from a single segment.
     *
     * Calculates the stress tensor contribution from correlating two
     * propagators (forward and backward) at a single contour point:
     *
     *     σ_αβ = ∫ q₁(r) · (∂/∂ε_αβ)[Propagator] · q₂(r) dr
     *
     * @param q_1                First propagator (typically forward)
     * @param q_2                Second propagator (typically backward)
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Whether using half bond length (discrete model)
     *
     * @return v⊗v dyad sums [V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃] (deformation-vector basis)
     *         Full 6-component stress tensor for non-orthogonal systems.
     *
     * @note Used in box size optimization to find stress-free unit cell.
     */
    virtual std::vector<T> compute_single_segment_stress(
        T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) = 0;

    /**
     * @brief Build what compute_single_segment_stress() needs before the stress
     *        loop runs its blocks in parallel (the CrysFFT stress multipliers of
     *        the pseudo-spectral solvers); the default does nothing.
     */
    virtual void prepare_stress() {}

    /**
     * @brief The Boltzmann factor exp(-w ds) of one full segment of the discrete chains,
     *        at ds index 0, in the basis of the propagators; valid after update_dw().
     *
     * exp_dw itself, unless the tables are on the full grid with a space group: then the
     * copy in the reduced basis that compute_exp_dw() keeps for the discrete chains.
     */
    const T* boltzmann_factor(const std::string& monomer_type) const;

protected:
    ComputationBox<T>* cb;       ///< Computation box
    Molecules* molecules;        ///< Molecules container
    SpaceGroup* space_group_;    ///< Space group of the reduced basis of the propagators, or nullptr
    int n_basis_;                ///< Values of a propagator: the reduced basis, or the full grid without a group

    /**
     * @brief Boltzmann factor of a step: exp(-w(r)·ds·factor) per ds_index.
     *
     * Key: (ds_index, monomer_type)
     * Value: exp(-w[i]*ds*factor), factor 1 for the discrete chains (ps, rs) and 1/2 for
     * rqm4, rk2 and cn-adi2 (see compute_exp_dw()); n_basis_ values when the CrysFFT steps
     * run in the reduced basis, otherwise one per grid point.
     *
     * ds_index is 0-based (from ContourLengthMapping).
     */
    std::map<int, std::map<std::string, std::vector<T>>> exp_dw;

    /**
     * @brief Boltzmann factor of a half step of rqm4: exp(-w(r)·ds/4) per ds_index.
     *
     * Key: (ds_index, monomer_type)
     * Value: exp(-w[i]*ds/4), laid out as exp_dw.
     *
     * ds_index is 0-based (from ContourLengthMapping).
     * Used by the two half steps of the Richardson extrapolation.
     */
    std::map<int, std::map<std::string, std::vector<T>>> exp_dw_half;

    /**
     * @brief The shared context of a solver.
     *
     * Enters exp_dw and, with with_half, exp_dw_half for every contour step and monomer
     * type, which compute_exp_dw() sizes for the basis of the tables (finalizing the
     * ContourLengthMapping).
     *
     * @param cb          Computation box
     * @param molecules   Molecules container
     * @param space_group Space group of the reduced basis of the propagators, or nullptr (full grid)
     * @param with_half   Also exp_dw_half
     */
    CpuSolver(ComputationBox<T>* cb, Molecules* molecules, SpaceGroup* space_group, bool with_half);

    /**
     * @brief The Boltzmann factors of the fields at every contour step, the body of update_dw().
     *
     * exp_dw[ds] = exp(-w·ds·factor) and, with with_half, exp_dw_half[ds] =
     * exp(-w·ds·factor_half) at every contour step of the chains. Tables on the full grid
     * take the fields of a space group expanded from the reduced basis (real fields only).
     *
     * Serial use only: called from update_dw(), never from the OpenMP propagator loops.
     *
     * @param w_input              Fields by monomer type, in the basis of the propagators
     * @param reduced_basis_tables The tables are in the reduced basis (the CrysFFT steps), not on the full grid
     * @param with_half            Also exp_dw_half
     * @param factor               Factor of ds in exp_dw
     * @param factor_half          Factor of ds in exp_dw_half
     */
    void compute_exp_dw(const std::map<std::string, const T*>& w_input, bool reduced_basis_tables,
                        bool with_half, double factor, double factor_half);

    /// @name The reduced basis and the mask around the full-grid steps (as in CudaSolver)
    /// `full` is a work array of the calling thread, which these functions size for the grid.
    /// @{
    /// q itself without a space group; otherwise its expansion from the reduced basis into full
    T* expand_to_full_grid(T* q, std::vector<T>& full) const;

    /// Where a step writes its full-grid result: q_out itself without a space group, otherwise
    /// full, which may hold the step's expanded input when the step reads it before it writes
    T* full_grid_output(T* q_out, std::vector<T>& full) const;

    /// Store the full-grid result full in q_out, reduced to the basis with a space group
    /// (full is q_out without one), then multiply the mask
    void store_full_grid(const T* full, T* q_out, const double* q_mask) const;

    /// q *= mask in the basis of the propagators; nothing for nullptr
    void multiply_mask(T* q, const double* q_mask) const;
    /// @}

private:
    /// Discrete chains with a space group and tables on the full grid: exp(-w ds) at ds index 0
    /// in the reduced basis, by monomer type (boltzmann_factor())
    std::map<std::string, std::vector<T>> exp_dw_reduced_;
};

template <typename T>
CpuSolver<T>::CpuSolver(ComputationBox<T>* cb, Molecules* molecules, SpaceGroup* space_group, bool with_half)
    : cb(cb), molecules(molecules), space_group_(space_group)
{
    n_basis_ =(space_group != nullptr) ? space_group->get_n_reduced_basis() : cb->get_total_grid();

    // The tables only: compute_exp_dw() sizes them, n_basis_ values on the CrysFFT path
    molecules->finalize_contour_length_mapping();
    const int n_unique_ds = molecules->get_contour_length_mapping().get_n_unique_ds();
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        for (const auto& item : molecules->get_bond_lengths())
        {
            exp_dw[ds_idx][item.first];
            if (with_half)
                exp_dw_half[ds_idx][item.first];
        }
    }
}

template <typename T>
void CpuSolver<T>::compute_exp_dw(const std::map<std::string, const T*>& w_input, bool reduced_basis_tables,
                                  bool with_half, double factor, double factor_half)
{
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    const int n_unique_ds = mapping.get_n_unique_ds();
    // Values of a table; the fields of a space group are expanded from the reduced basis
    // into tables on the full grid
    const int n_table = reduced_basis_tables ? n_basis_ : cb->get_total_grid();
    const bool expand = (space_group_ != nullptr) && !reduced_basis_tables;
    // Discrete chains keep exp(-w ds) at ds index 0 in the basis of the propagators as well
    // (boltzmann_factor()), from the fields of the basis: the values of the full-grid table
    // at the points of the basis
    const bool keep_reduced = expand && molecules->get_model_name() == "discrete";

    std::vector<T> w_full;
    for (const auto& [monomer_type, w_input_monomer] : w_input)
    {
        const T* w = w_input_monomer;
        if (expand)
        {
            // [real-only feature] The reduced basis of a space group is implemented for real fields
            if constexpr (std::is_same_v<T, double>)
            {
                w_full.resize(cb->get_total_grid());
                space_group_->from_reduced_basis(w_input_monomer, w_full.data(), 1);
                w = w_full.data();
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            if (!exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            const double local_ds = mapping.get_ds_from_index(ds_idx);
            std::vector<T>& table = exp_dw.at(ds_idx).at(monomer_type);
            table.resize(n_table);
            for (int i = 0; i < n_table; ++i)
                table[i] = std::exp(-w[i] * local_ds * factor);
            if (with_half)
            {
                std::vector<T>& table_half = exp_dw_half.at(ds_idx).at(monomer_type);
                table_half.resize(n_table);
                for (int i = 0; i < n_table; ++i)
                    table_half[i] = std::exp(-w[i] * local_ds * factor_half);
            }
        }
        if (keep_reduced)
        {
            const double ds = mapping.get_ds_from_index(0);
            std::vector<T>& reduced = exp_dw_reduced_[monomer_type];
            reduced.resize(n_basis_);
            for (int i = 0; i < n_basis_; ++i)
                reduced[i] = std::exp(-w_input_monomer[i] * ds * factor);
        }
    }
}

template <typename T>
const T* CpuSolver<T>::boltzmann_factor(const std::string& monomer_type) const
{
    // exp_dw of the continuous chains holds exp(-w ds/2)
    if (molecules->get_model_name() != "discrete")
        throw_with_line_number("The Boltzmann factor of a full segment belongs to the discrete chains.");
    // The reduced copy exists exactly when compute_exp_dw() keeps one
    const bool reduced_copy = !exp_dw_reduced_.empty();
    return (reduced_copy ? exp_dw_reduced_ : exp_dw.at(0)).at(monomer_type).data();
}

template <typename T>
T* CpuSolver<T>::expand_to_full_grid(T* q, std::vector<T>& full) const
{
    if (space_group_ == nullptr)
        return q;
    // [real-only feature] The reduced basis of a space group is implemented for real fields
    if constexpr (std::is_same_v<T, double>)
    {
        full.resize(cb->get_total_grid());
        space_group_->from_reduced_basis(q, full.data(), 1);
        return full.data();
    }
    throw_with_line_number("Space group reduced basis is only supported for real fields.");
}

template <typename T>
T* CpuSolver<T>::full_grid_output(T* q_out, std::vector<T>& full) const
{
    if (space_group_ == nullptr)
        return q_out;
    full.resize(cb->get_total_grid());
    return full.data();
}

template <typename T>
void CpuSolver<T>::store_full_grid(const T* full, T* q_out, const double* q_mask) const
{
    if (space_group_ != nullptr)
    {
        // [real-only feature] The reduced basis of a space group is implemented for real fields
        if constexpr (std::is_same_v<T, double>)
            space_group_->to_reduced_basis(full, q_out, 1);
        else
            throw_with_line_number("Space group reduced basis is only supported for real fields.");
    }
    // The mask is stored in the basis of q_out, so it applies after the reduction; it is
    // symmetric, so this equals masking on the full grid
    multiply_mask(q_out, q_mask);
}

template <typename T>
void CpuSolver<T>::multiply_mask(T* q, const double* q_mask) const
{
    if (q_mask == nullptr)
        return;
    for (int i = 0; i < n_basis_; ++i)
        q[i] *= q_mask[i];
}
#endif
