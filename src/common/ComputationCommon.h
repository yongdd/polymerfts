/**
 * @file ComputationCommon.h
 * @brief Platform-neutral base of the propagator computation classes.
 *
 * Holds what every chain model and storage mode shares: the platform backend,
 * the backend memory, the block and solvent concentrations, the fields kept
 * for the solvents, the accessors that read them, and the bookkeeping of the
 * four computations (partition-function samples, the chain propagator getter,
 * concurrent jobs). The derived classes own the propagator storage, the order
 * of the propagators and the compute_* algorithms.
 *
 * @see ComputationBackend for the platform interface
 * @see ComputationContinuous for the continuous-chain computation
 */

#ifndef COMPUTATION_COMMON_H_
#define COMPUTATION_COMMON_H_

#include <array>
#include <memory>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationBackend.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "NumericalMethods.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"

/**
 * @brief Carries an exception out of an OpenMP parallel loop.
 *
 * An exception cannot leave an OpenMP parallel region; the program would be
 * terminated. run() catches it inside the loop body and rethrow() rethrows
 * the first one after the loop, so that, for example, running out of GPU
 * memory during a computation reaches Python as an exception.
 */
class ParallelErrors
{
private:
    std::exception_ptr first;
    std::mutex mutex;

public:
    template <typename Body>
    void run(Body&& body)
    {
        try
        {
            body();
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!first)
                first = std::current_exception();
        }
    }

    void rethrow()
    {
        if (first)
            std::rethrow_exception(first);
    }
};

/**
 * @brief The owner of the fields and checkpoints of one computation in backend memory.
 *
 * Every array it allocates is freed when it is destroyed. A computation declares it
 * after its backend, so the arrays go before the backend (and its CUDA streams),
 * also when a constructor throws after allocating some of them.
 */
template <typename T>
class BackendMemory
{
private:
    const std::unique_ptr<ComputationBackend<T>>& backend;
    std::vector<BackendArray<T>*> arrays;
    std::vector<CheckpointArray<T>*> checkpoints;

public:
    explicit BackendMemory(const std::unique_ptr<ComputationBackend<T>>& backend) : backend(backend) {}
    BackendMemory(const BackendMemory&) = delete;
    BackendMemory& operator=(const BackendMemory&) = delete;
    ~BackendMemory()
    {
        for (BackendArray<T>* a : arrays)
            backend->deallocate(a);
        for (CheckpointArray<T>* a : checkpoints)
            backend->deallocate_checkpoint(a);
    }

    /// A field of get_n_basis() elements, set to zero
    BackendArray<T>* allocate()
    {
        arrays.push_back(nullptr);  // room first, so that no allocation is lost
        try
        {
            arrays.back() = backend->allocate();
        }
        catch (...)
        {
            arrays.pop_back();
            throw;
        }
        return arrays.back();
    }

    /// A checkpoint of get_n_basis() elements (contents undefined)
    CheckpointArray<T>* allocate_checkpoint()
    {
        checkpoints.push_back(nullptr);
        try
        {
            checkpoints.back() = backend->allocate_checkpoint();
        }
        catch (...)
        {
            checkpoints.pop_back();
            throw;
        }
        return checkpoints.back();
    }
};

template <typename T>
class ComputationCommon : public PropagatorComputation<T>
{
protected:
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
     * The space group of the computation is the one of the box (the factory's
     * SpaceGroupGuard attaches it for the construction).
     *
     * @param numerical_method The numerical method of the solver
     * @param create_backend   Called after the space group is set on the box,
     *                         because the solvers size their buffers from it.
     */
    ComputationCommon(ComputationBox<T>* cb, Molecules* molecules,
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
    void update_laplacian_operator() override;
    void set_force_off_diagonal_stress(bool force) override;

    /// One step of the propagator of block (v, u) of polymer p from q_init, in two temporary fields
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;

    /// The two computations in turn; the reduced-memory classes do both in one walk.
    void compute_concentrations_and_stress() override;

    T get_total_partition(int polymer) override;
    T get_solvent_partition(int s) override;
    void get_total_concentration(std::string monomer_type, T* phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T* phi) override;
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T* phi) override;
    void get_block_concentration(int polymer, T* phi) override;
    void get_solvent_concentration(int s, T* phi) override;
};

#endif
