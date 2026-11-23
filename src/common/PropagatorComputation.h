/**
 * @file PropagatorComputation.h
 * @brief The propagator computation of polymer field theory: its interface and what the computation classes share.
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

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationBackend.h"
#include "ComputationBox.h"
#include "ComputationSupport.h"
#include "Exception.h"
#include "Molecules.h"
#include "NumericalMethods.h"
#include "PartitionFunctionCheck.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"

/// Relative tolerance for contour partition consistency (including complex phase).
constexpr double PARTITION_TOLERANCE = 1e-7;

/**
 * @class PropagatorComputation
 * @brief The propagator computation: chain propagators, partition functions,
 *        concentrations and stress.
 *
 * This class defines the interface for computing chain propagators and
 * derived quantities (partition functions, concentrations, stress), and holds
 * what every chain model and storage mode shares: the platform backend, the
 * backend memory, the block and solvent concentrations, the fields kept for
 * the solvents, the accessors that read them, and the bookkeeping of the
 * computations (partition-function samples, the chain propagator getter,
 * concurrent jobs). The derived classes ComputationContinuous,
 * ComputationDiscrete and their ComputationReduceMemory variants own the
 * propagator storage, the order of the propagators and the compute_*
 * algorithms; AbstractFactory::create_propagator_computation() creates them.
 * They are platform neutral: every grid operation goes through a ComputationBackend.
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
 * The fields are in the basis of the box: the reduced basis of its space group
 * if it has one, otherwise the full grid.
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
 * @see ComputationBackend for the platform operations
 */
template <typename T>
class PropagatorComputation
{
protected:
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

    NumericalMethodInfo numerical_method;            ///< The numerical method (check_stress_supported())
    std::unique_ptr<ComputationBackend<T>> backend;  ///< Platform operations
    BackendMemory<T> memory;                         ///< Every field and checkpoint (freed before backend)
    int n_streams;                   ///< Number of propagators computed concurrently

    using BlockMap = std::map<std::tuple<int, std::string, std::string>, BackendArray<T>*>;

    /// Unnormalized segment-product integral of each block; getters apply the
    /// canonical or fugacity normalization, including when volume fraction is zero.
    BlockMap phi_block;

    /// Indexed access in map order. The block map is fixed after construction.
    std::vector<typename BlockMap::const_iterator> phi_block_iterators;

    std::vector<BackendArray<T>*> phi_solvent;     ///< Concentration of each solvent
    std::map<std::string, BackendArray<T>*> w;     ///< Fields of the last compute_propagators()
    std::set<std::string> w_input_types;           ///< Monomer types given to the last compute_propagators()

    /**
     * With complex fields every junction power must be an integer of at most 10^6
     * (junction_product::integer_power): the constructor refuses other powers.
     * The computation works in the basis of the box. A numerical method that
     * requires an orthogonal cell holds the box orthogonal from the end of this
     * constructor until the computation is destroyed (ComputationBox::hold_orthogonal_cell()).
     *
     * @param cb               Computation box (grid, basis and integration)
     * @param molecules        Polymer and solvent definitions
     * @param propagator_computation_optimizer Computation schedule, built from molecules
     * @param numerical_method The numerical method of the solver
     * @param create_backend   Creates the platform backend, whose solvers size
     *                         their buffers from the basis of the box.
     */
    PropagatorComputation(ComputationBox<T>* cb, Molecules* molecules,
                          PropagatorComputationOptimizer* propagator_computation_optimizer,
                          const NumericalMethodInfo& numerical_method,
                          const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    /**
     * @brief Check the inputs of compute_propagators(), keep a copy of the
     *        fields (solvents need them) and update the Boltzmann factors.
     *
     * Checked here, before any parallel loop: every monomer type has a field
     * and every grafting point has its q_init.
     */
    void set_fields(const std::map<std::string, const T*>& w_input,
                    const std::map<std::string, const T*>& q_init);

    /// Solvent concentrations and partition functions from the stored fields.
    void compute_solvent_concentrations();

    /**
     * @brief Throw if a non-integer block multiplicity met a negative propagator.
     *
     * A real power of a negative value is NaN, which reaches the partition
     * function of the polymer, so only Q is checked, once per compute_propagators().
     */
    void check_real_power_partitions() const;

    /// Contour step ds of the block that propagator `key` belongs to.
    double contour_step(const std::string& key) const;

    /// Normalization of a block concentration, φ = norm · Σ_n c_n q q†:
    /// norm = ds · volume fraction / alpha · n_repeated / Q.
    T concentration_norm(const std::tuple<int, std::string, std::string>& block_key) const;

    /// Stress of a block: the sum over its segments times the contour step ds.
    void scale_by_contour_step(std::array<T, 6>& stress, const std::string& key_right) const;

    /**
     * @brief Sum the stresses of the blocks of each polymer into dq_dl and
     *        convert them to lattice derivatives.
     * @param stress_of_block One entry per block, in the order of phi_block.
     */
    void set_stress_from_blocks(const std::vector<std::array<T, 6>>& stress_of_block);

    /**
     * @brief Turn the summed v⊗v components in dq_dl into lattice-parameter
     *        derivatives [dH/dL1, dH/dL2, dH/dL3, dH/dγ, dH/dβ, dH/dα].
     * @see docs/theory/StressTensor.md
     */
    void convert_stress_to_lattice_derivatives();

    /// Throw if the numerical method computes no stress.
    void check_stress_supported() const;

    /**
     * @brief body(stream, i) for i = 0..n-1, concurrently on n_streams streams.
     *
     * Synchronizes the backend after the loop and rethrows the first exception of
     * a body (ParallelErrors).
     */
    void run_concurrently(size_t n, const std::function<void(int, size_t)>& body);

    /// The first block of each polymer in phi_block: one segment of it gives Q of the polymer
    std::vector<typename BlockMap::const_iterator> first_block_of_each_polymer() const;

    /**
     * @brief The key of the propagator of one block and direction, for get_chain_propagator().
     *
     * Checks that the propagators are computed, that the propagator exists (not
     * aggregated away) and that n lies in [first_segment, max_n_segment].
     */
    const std::string& chain_propagator_key(int polymer, int v, int u, int n, int first_segment);

    /// Q at every segment of every block, by polymer, for compute_partition_errors()
    struct ContourPartitions
    {
        std::vector<std::vector<T>> values;
        std::vector<std::vector<double>> absolute_integrals;  ///< Absent polymers: the scale of the integrands
        explicit ContourPartitions(int n_polymers) : values(n_polymers), absolute_integrals(n_polymers) {}
    };

    /**
     * @brief Add Q = ∫ q q† / bead_factor dr / V of one segment of a block, per propagator.
     * @param bead_factor exp(-w ds) of discrete chains, which both propagators carry;
     *                    nullptr for continuous chains
     */
    void add_partition_sample(ContourPartitions& partitions, int p, const ComputationBlock& block,
                              BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>* bead_factor);

    /// Compare canonical Q relatively and zero-weight species on the scale of
    /// their absolute contour integrands (which need not cancel when Q does).
    std::vector<double> partition_errors(const ContourPartitions& partitions) const;

private:
    /**
     * @brief The block concentrations of one monomer type and their weights, for sum_to_host().
     *
     * The blocks with segments of polymer p (of every polymer when p is -1), weighted
     * by concentration_norm() or, with a fugacity, by fugacity · ds · n_repeated. Without a
     * fugacity an absent species adds nothing, also when its unused propagators are not
     * finite; an occupied one whose Q overflows keeps its non-finite density.
     */
    void collect_block_concentrations(int p, const std::string& monomer_type, std::optional<double> fugacity,
                                      std::vector<BackendArray<T>*>& arrays, std::vector<T>& weights) const;

public:
    /// Virtual destructor; releases the orthogonal cell of the box (see the constructor).
    virtual ~PropagatorComputation();

    /**
     * @brief Get the computation box.
     * @return Pointer to ComputationBox
     */
    ComputationBox<T>* get_cb() const { return this->cb; }

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
    void update_laplacian_operator();

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
    void advance_propagator_single_segment(T* q_init, T *q_out, int p, int v, int u);

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
     * propagators are recomputed from the checkpoints once instead of twice (the
     * reduced-memory classes override this function).
     *
     * @pre compute_propagators() must be called first.
     */
    virtual void compute_concentrations_and_stress();

    /**
     * @brief Get single-chain partition function.
     *
     * @param polymer Polymer index (0 to n_polymer_types-1)
     * @return Q = (1/V) ∫ q(r,N) dr
     *
     * @pre compute_propagators() must be called first.
     */
    T get_total_partition(int polymer);

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
    void get_total_concentration(std::string monomer_type, T *phi);

    /**
     * @brief Get concentration for a monomer type from specific polymer.
     *
     * @param polymer     Polymer index
     * @param monomer_type Monomer type label
     * @param phi         Output array of length total_grid
     */
    void get_total_concentration(int polymer, std::string monomer_type, T *phi);

    /**
     * @brief Get block-resolved concentration for a polymer.
     *
     * Returns φ for each block concatenated: [φ_block0, φ_block1, ...]. A block with copies
     * (Block::n_copies, from multiplicities) gives the concentration of all its copies.
     *
     * @param polymer Polymer index
     * @param phi     Output array of length (n_blocks * total_grid)
     */
    void get_block_concentration(int polymer, T *phi);

    /**
     * @brief Get solvent partition function.
     * @param s Solvent index
     * @return Q_s = (1/V) ∫ exp(-w(r) ds) dr, with the global contour step ds
     */
    T get_solvent_partition(int s);

    /**
     * @brief Get solvent concentration field.
     * @param s   Solvent index
     * @param phi Output array of length total_grid
     */
    void get_solvent_concentration(int s, T *phi);

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
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi);

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

    /**
     * @brief Force off-diagonal (cross-term) stress computation for orthogonal boxes.
     *
     * By default (false), the stress computation skips the cross-term sums
     * (V₁₂, V₁₃, V₂₃) for orthogonal boxes, because angle optimization is
     * normally inactive for orthogonal crystal systems. When angle optimization
     * starts at exactly 90° angles (e.g., Monoclinic/Triclinic/Oblique2D runs
     * initialized orthogonal), the angle derivatives would then be zero and the
     * angles would never move. The backend forwards this flag to its
     * propagator solvers.
     *
     * @param force true to always accumulate V₁₂/V₁₃/V₂₃; false (default)
     *              keeps the orthogonal-box skip.
     */
    void set_force_off_diagonal_stress(bool force);
};
#endif
