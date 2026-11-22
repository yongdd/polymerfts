/**
 * @file PropagatorComputation.h
 * @brief Abstract interface for computing chain propagators in polymer field theory.
 *
 * This header defines the PropagatorComputation class, which is the central
 * computational engine for polymer field theory simulations. It computes
 * chain propagators q(r,s) that describe the statistical weight of polymer
 * chain configurations, and from these derives partition functions and
 * monomer concentrations.
 *
 * **Chain Propagators:**
 *
 * For a continuous Gaussian chain, the propagator q(r,s) satisfies the
 * modified diffusion equation:
 *
 *   dq/ds = (a²/6) ∇²q - w(r) q
 *
 * where:
 * - a is the statistical segment length
 * - w(r) is the external potential field
 * - s is the contour variable (0 to N)
 *
 * Initial condition: q(r,0) = 1 for free ends, or specified for grafted chains.
 *
 * **Partition Functions:**
 *
 * The single-chain partition function is:
 *   Q = (1/V) ∫ q(r,N) dr
 *
 * **Concentration Calculation:**
 *
 * Monomer concentrations are computed by integrating propagator products:
 *   φ(r) = (φ_total/Q) ∫₀ᴺ q(r,s) q†(r,N-s) ds
 *
 * where q† is the complementary propagator from the other chain end.
 *
 * @see Pseudo for the pseudo-spectral solver implementation
 * @see PropagatorComputationOptimizer for computation scheduling
 *
 * @example
 * @code
 * // Create solver
 * std::unique_ptr<PropagatorComputation<double>> solver = factory->create_propagator_computation(
 *     cb.get(), molecules.get(), optimizer.get(), "rqm4");
 *
 * // Compute propagators with given potential fields
 * std::map<std::string, const double*> w = {{"A", w_A}, {"B", w_B}};
 * solver->compute_propagators(w);
 *
 * // Compute concentrations
 * solver->compute_concentrations();
 *
 * // Get concentration field for monomer type A
 * double* phi_A = new double[total_grid];
 * solver->get_total_concentration("A", phi_A);
 *
 * // Get partition function
 * double Q = solver->get_total_partition(0);
 * @endcode
 */

#ifndef PROPAGATOR_COMPUTATION_H_
#define PROPAGATOR_COMPUTATION_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "SpaceGroup.h"
#include "Exception.h"
#include "PartitionFunctionCheck.h"

/// Relative tolerance for contour partition consistency (including complex phase).
constexpr double PARTITION_TOLERANCE = 1e-7;

/**
 * @class PropagatorComputation
 * @brief Abstract base class for chain propagator computation.
 *
 * This class defines the interface for computing chain propagators and
 * derived quantities (partition functions, concentrations, stress).
 * Platform-specific implementations (CPU, CUDA) inherit from this class.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Computation Pipeline:**
 *
 * 1. compute_propagators(w): Solve diffusion equations for all propagators
 * 2. compute_concentrations(): Integrate propagator products to get φ(r)
 *    (compute_concentrations_and_stress() also gives the stress)
 * 3. get_total_concentration(): Retrieve computed concentration fields
 *
 * **Memory Management:**
 *
 * The class allocates internal storage for propagators and concentrations.
 * Memory can be reduced using the reduce_memory option,
 * which stores only checkpoints and recomputes on demand.
 *
 * **Parallel Computation:**
 *
 * With full storage, independent propagators are computed concurrently in the
 * groups of a Scheduler: one OpenMP thread each on the CPU, one stream each on
 * CUDA. With reduced memory they are computed one at a time.
 *
 * @see ComputationContinuous, ComputationDiscrete and the ComputationReduceMemory
 *      classes for implementations (platform neutral, see ComputationBackend)
 */
template <typename T>
class PropagatorComputation
{
protected:
    /// Reject use of buffers whose basis no longer matches the shared box.
    void check_box_space_group() const;
    ComputationBox<T>* cb;                                      ///< Computation box (grid and integration)
    Molecules *molecules;                                       ///< Polymer and solvent definitions
    PropagatorComputationOptimizer *propagator_computation_optimizer;  ///< Computation schedule

    std::vector<T> single_polymer_partitions;  ///< Partition functions Q[p] for each polymer type p
    std::vector<T> single_solvent_partitions;  ///< Partition functions Q_s[s] for each solvent type s

    /**
     * @brief Lattice-parameter derivative contributions from each polymer.
     *
     * After compute_stress(), dq_dl[p] holds the per-polymer derivatives
     * [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα] (deformation-vector
     * approach; see docs/theory/StressTensor.md).
     * For 2D: [dH/dL₁, dH/dL₂, dH/dγ, 0, 0, 0]. For 1D: only index 0.
     * Used for box relaxation in SCFT.
     */
    std::vector<std::array<T,6>> dq_dl;

    /**
     * @brief Space group for reduced basis representation (optional).
     *
     * When set, fields and concentrations can be represented using only
     * irreducible mesh points. Set to nullptr when not using symmetry.
     */
    SpaceGroup* space_group_;

    /**
     * @brief What has been computed for the current fields and box.
     *
     * A new compute_propagators() and update_laplacian_operator() clear all
     * three; each computation sets its own flag when it completes. The
     * getters throw instead of returning values of other fields or another
     * box (or the zeros and uninitialized arrays before the first run).
     */
    bool propagators_computed_ = false;
    bool concentrations_computed_ = false;
    bool stress_computed_ = false;

    void clear_computed();
    void require_propagators() const;
    void require_concentrations() const;
    void require_stress() const;

    /// Whether this computation holds its box orthogonal (hold_orthogonal_cell())
    bool holds_orthogonal_cell_ = false;

public:
    /**
     * @brief Construct a PropagatorComputation solver.
     *
     * @param cb           Computation box managing grid and FFT
     * @param molecules    Polymer/solvent definitions
     * @param propagator_computation_optimizer Computation schedule optimizer
     */
    PropagatorComputation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer);

    /// Virtual destructor.
    virtual ~PropagatorComputation();

    /**
     * @brief Keep the box orthogonal for the lifetime of this computation.
     *
     * The factory calls it once for a numerical method that requires an
     * orthogonal cell; see ComputationBox::hold_orthogonal_cell().
     */
    void hold_orthogonal_cell();

    /**
     * @brief Get the computation box.
     * @return Pointer to ComputationBox
     */
    ComputationBox<T>* get_cb() const { check_box_space_group(); return this->cb; }

    /**
     * @brief Get number of blocks in a polymer.
     * @param polymer Polymer index
     * @return Number of blocks (edges) in polymer graph
     */
    int get_n_blocks(int polymer) const { Polymer& pc = this->molecules->get_polymer(polymer); return pc.get_n_blocks();};

    /**
     * @brief Update Laplacian operator after box size change.
     *
     * Must be called after ComputationBox::set_lx() or set_lattice_parameters()
     * to update the Fourier-space Laplacian for the new box dimensions.
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Compute all chain propagators.
     *
     * Solves the modified diffusion equation for all propagators needed
     * to compute partition functions and concentrations. Uses dynamic
     * programming to avoid redundant computations for branched polymers.
     *
     * @param w_block Map from monomer type to potential field w(r)
     * @param q_init  Optional: map from label to initial condition q(r,0)
     *                for grafted chain ends
     *
     * @note w_block must contain entries for all monomer types.
     * @note For free ends, q_init is empty and q(r,0) = 1 is used.
     *
     * @example
     * @code
     * std::map<std::string, const double*> w = {{"A", w_A}, {"B", w_B}};
     * solver->compute_propagators(w);
     *
     * // For grafted chain with "surface" initial condition
     * std::map<std::string, const double*> q_init = {{"surface", grafting_density}};
     * solver->compute_propagators(w, q_init);
     * @endcode
     */
    virtual void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) = 0;

    /**
     * @brief Advance a propagator by one step of block (v, u).
     *
     * One step of the numerical method with the fields and the cell of the last
     * compute_propagators(): for continuous chains one contour step ds of the
     * block, for discrete chains one bond and segment, q_out = exp(-w ds) (g * q_init).
     *
     * @param q_init      Input propagator at contour position s
     * @param q_out       Output propagator at contour position s+ds
     * @param p           Polymer index
     * @param v           Starting vertex of the block
     * @param u           Ending vertex of the block
     *
     * @note A hook for tests and debugging; compute_propagators() does not call it.
     */
    virtual void advance_propagator_single_segment(T* q_init, T *q_out, int p, int v, int u) = 0;

    /**
     * @brief Compute monomer concentrations from propagators.
     *
     * Integrates propagator products along the chain contour to compute
     * concentration fields φ(r) for each monomer type.
     *
     * @pre compute_propagators() must be called first.
     *
     * @note Continuous chains integrate the contour with Simpson's rule; discrete
     *       chains sum over the segments.
     */
    virtual void compute_concentrations() = 0;

    /**
     * @brief Compute stress tensor from propagators.
     *
     * Computes the lattice-parameter derivatives of each polymer for box
     * relaxation (dq_dl); retrieve their sum with get_stress() or get_stress_gce().
     *
     * @pre compute_propagators() must be called first.
     */
    virtual void compute_stress() = 0;

    /**
     * @brief Compute the concentrations and the stress of the current propagators.
     *
     * Gives the results of compute_concentrations() followed by compute_stress(),
     * bit for bit. With reduced memory, one walk along each block gives both, so the
     * propagators are recomputed from the checkpoints once instead of twice.
     *
     * @pre compute_propagators() must be called first.
     */
    virtual void compute_concentrations_and_stress() = 0;

    /**
     * @brief Get single-chain partition function.
     *
     * @param polymer Polymer index (0 to n_polymer_types-1)
     * @return Q = (1/V) ∫ q(r,N) dr
     *
     * @pre compute_propagators() must be called first.
     */
    virtual T get_total_partition(int polymer) = 0;

    /**
     * @brief Get chain propagator at specific contour position.
     *
     * @param q_out   Output array of length total_grid
     * @param polymer Polymer index
     * @param v       Starting vertex of edge
     * @param u       Ending vertex of edge
     * @param n       Contour index (continuous chains 0 to n_segment, discrete chains 1 to n_segment)
     *
     * @note Returns q(r,s) propagating from vertex v toward u at step n.
     */
    virtual void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) = 0;

    /**
     * @brief Get total concentration for a monomer type (all polymers).
     *
     * @param monomer_type Monomer type label (e.g., "A")
     * @param phi         Output array of length total_grid
     *
     * @pre compute_concentrations() must be called first.
     */
    virtual void get_total_concentration(std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get concentration for a monomer type from specific polymer.
     *
     * @param polymer     Polymer index
     * @param monomer_type Monomer type label
     * @param phi         Output array of length total_grid
     */
    virtual void get_total_concentration(int polymer, std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get block-resolved concentration for a polymer.
     *
     * Returns φ for each block concatenated: [φ_block0, φ_block1, ...]. A block with copies
     * (Block::n_copies, from multiplicities) gives the concentration of all its copies.
     *
     * @param polymer Polymer index
     * @param phi     Output array of length (n_blocks * total_grid)
     */
    virtual void get_block_concentration(int polymer, T *phi) = 0;

    /**
     * @brief Get solvent partition function.
     * @param s Solvent index
     * @return Q_s = (1/V) ∫ exp(-w(r) ds) dr, with the global contour step ds
     */
    virtual T get_solvent_partition(int s) = 0;

    /**
     * @brief Get solvent concentration field.
     * @param s   Solvent index
     * @param phi Output array of length total_grid
     */
    virtual void get_solvent_concentration(int s, T *phi) = 0;

    /**
     * @brief Get the lattice-parameter derivatives of H for all polymers combined.
     *
     * @return [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα]; for 2D
     *         [dH/dL₁, dH/dL₂, dH/dγ, 0, 0, 0], for 1D only index 0
     *         (see docs/theory/StressTensor.md)
     *
     * @pre compute_stress() must be called first.
     */
    std::vector<T> get_stress();

    /**
     * @brief Get concentration for grand canonical ensemble.
     *
     * Scales concentration by fugacity for open systems.
     *
     * @param fugacity    Chemical fugacity z = exp(μ/kT)
     * @param polymer     Polymer index
     * @param monomer_type Monomer type label
     * @param phi         Output concentration field
     */
    virtual void get_total_concentration_gce  (double fugacity, int polymer, std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get stress for grand canonical ensemble.
     *
     * @param fugacities Vector of fugacities for each polymer type
     * @return Stress vector weighted by fugacities
     */
    std::vector<T> get_stress_gce(std::vector<double> fugacities);

    /**
     * @brief Relative partition function error of each polymer.
     *
     * Every contour pairing must give the same Q. For positive canonical
     * volume fraction, normalize deviations by the largest magnitude of the
     * reference and contour partitions; zero reference Q is invalid. For zero
     * canonical weight, normalize by the largest integral of the absolute
     * contour integrand, including the discrete inverse Boltzmann weight and
     * the same volume, mask and aggregation factors. This permits cancellation
     * at Q = 0 without amplifying roundoff. Non-finite partitions or scales give
     * infinite error. Every polymer is evaluated and nothing is printed.
     *
     * @return One error per polymer type
     *
     * @note Small deviations are normal due to numerical discretization.
     */
    virtual std::vector<double> compute_partition_errors() = 0;

    /**
     * @brief Verify partition function consistency.
     *
     * @param tolerance Largest accepted relative error (PARTITION_TOLERANCE by default)
     * @return true if every error of compute_partition_errors() is finite and at
     *         most tolerance
     */
    bool check_total_partition(double tolerance = PARTITION_TOLERANCE);

    // ==================== Space Group / Reduced Basis Methods ====================

    /**
     * @brief Get the space group, fixed at construction.
     *
     * The backend buffers and solvers use its basis; the computation refuses its
     * results while its box holds another group.
     *
     * @return Pointer to SpaceGroup, or nullptr if not set
     */
    SpaceGroup* get_space_group() const { return space_group_; }

    /**
     * @brief Force off-diagonal (cross-term) stress computation for orthogonal boxes.
     *
     * By default (false), the stress computation skips the cross-term sums
     * (V₁₂, V₁₃, V₂₃) for orthogonal boxes, because angle optimization is
     * normally inactive for orthogonal crystal systems. When angle optimization
     * starts at exactly 90° angles (e.g., Monoclinic/Triclinic/Oblique2D runs
     * initialized orthogonal), the angle derivatives would then be zero and the
     * angles would never move. Platform implementations forward this flag to
     * their propagator solvers.
     *
     * @param force true to always accumulate V₁₂/V₁₃/V₂₃; false (default)
     *              keeps the orthogonal-box skip.
     */
    virtual void set_force_off_diagonal_stress(bool force) = 0;
};
#endif
