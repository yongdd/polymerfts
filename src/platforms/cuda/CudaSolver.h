/**
 * @file CudaSolver.h
 * @brief Abstract interface for GPU propagator solvers.
 *
 * This header defines CudaSolver, the base class for GPU-based propagator
 * solvers. It provides a common interface for pseudo-spectral and real-space
 * methods implemented using CUDA.
 *
 * **Multi-Stream Architecture:**
 *
 * All solver methods take a STREAM parameter for concurrent execution:
 * - Up to MAX_STREAMS propagators can be computed simultaneously
 * - Each stream has independent cuFFT plans and workspace
 *
 * **Device Memory:**
 *
 * All input/output arrays are device pointers (CuDeviceData<T>).
 * No host-device transfers occur within solver methods. With a space group,
 * which the solvers take in their constructors with the backend's device index
 * maps, the propagators are in its reduced basis: the base class expands them to
 * the full grid around the steps, reduces the results back and multiplies the mask.
 *
 * @see CudaSolverPseudoRQM4 for continuous chain pseudo-spectral
 * @see CudaSolverPseudoDiscrete for discrete chain pseudo-spectral
 * @see CudaSolverCNADI for finite difference method
 */

#ifndef CUDA_SOLVER_H_
#define CUDA_SOLVER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaCommon.h"

class SpaceGroup;  // Forward declaration

/**
 * @class CudaSolver
 * @brief Abstract base class for GPU propagator solvers.
 *
 * Defines the interface for GPU-accelerated propagator advancement
 * and stress computation. Uses CUDA streams for concurrent execution.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Boltzmann Factors:**
 *
 * Like CpuSolver, stores precomputed exp(-w·ds) factors in GPU memory:
 * - d_exp_dw: Full segment Boltzmann factor (per ds_index for local ds support)
 * - d_exp_dw_half: Half segment Boltzmann factor (per ds_index)
 *
 * **Shared context:** the computation box, the molecules, the backend's streams,
 * the space group with the backend's index maps, and the full-grid buffers of a
 * reduced-basis run, with the expansion, reduction and mask around the steps.
 */
template <typename T>
class CudaSolver
{
public:
    /**
     * @brief Full segment Boltzmann factor: exp(-w·ds) on GPU.
     *
     * Outer key: ds_index (0-based)
     * Inner key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<int, std::map<std::string, CudaDeviceBuffer<CuDeviceData<T>>>> d_exp_dw;

    /**
     * @brief Half segment Boltzmann factor: exp(-w·ds/2) on GPU.
     *
     * Outer key: ds_index (0-based)
     * Inner key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<int, std::map<std::string, CudaDeviceBuffer<CuDeviceData<T>>>> d_exp_dw_half;

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
    virtual ~CudaSolver() = default;

    /**
     * @brief Set the force-off-diagonal-stress flag (see force_off_diagonal_stress_).
     * @param force true to always accumulate cross-term stress sums
     */
    void set_force_off_diagonal_stress(bool force) { force_off_diagonal_stress_ = force; }

    /**
     * @brief Update Laplacian operator for new box dimensions.
     *
     * Recomputes Fourier-space operators or finite difference coefficients.
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * @param d_w Device fields by monomer type, in the basis of the propagators
     *            (the reduced basis with a space group)
     */
    virtual void update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w) = 0;

    /**
     * @brief Advance propagator by one contour step on GPU.
     *
     * @param STREAM      CUDA stream index (0 to MAX_STREAMS-1)
     * @param d_q_in      Input propagator (device pointer)
     * @param d_q_out     Output propagator (device pointer)
     * @param monomer_type Monomer type for Boltzmann factor
     * @param d_q_mask    Optional mask for impenetrable regions (device)
     * @param ds_index    Index for ds value (0-based, for per-block ds support)
     */
    virtual void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) = 0;

    /**
     * @brief Advance propagator by half bond step (discrete chains).
     *
     * @param STREAM      CUDA stream index
     * @param q_in        Input propagator (device)
     * @param q_out       Output propagator (device)
     * @param monomer_type Monomer type
     */
    virtual void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *q_in, CuDeviceData<T> *q_out, std::string monomer_type) = 0;

    /**
     * @brief Compute stress contribution from one segment.
     *
     * @param STREAM           CUDA stream index
     * @param d_q_1            Forward propagator of the segment (device)
     * @param d_q_2            Backward propagator of the segment (device)
     * @param d_segment_stress Output stress array (device)
     * @param monomer_type     Monomer type
     * @param is_half_bond_length Whether using half bond (discrete)
     */
    virtual void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_1, CuDeviceData<T> *d_q_2, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) = 0;

    /// Build the tables compute_single_segment_stress() needs, before the streams
    /// compute the segment stresses (default: nothing).
    virtual void prepare_stress() {}

protected:
    ComputationBox<T>* cb;                   ///< Computation box
    Molecules* molecules;                    ///< Molecules container
    int n_streams;                           ///< Number of parallel streams
    cudaStream_t streams[MAX_STREAMS] = {};  ///< CUDA stream per stream index (the backend's)

    /// @name Space group (reduced basis)
    /// With a space group the propagators, fields and mask are in its reduced basis: the
    /// steps expand their input to the full grid and reduce their result back.
    /// @{
    SpaceGroup* space_group_;                ///< nullptr without a space group
    const int* d_reduced_basis_indices_;     ///< Device map reduced -> full index (lent by the backend; nullptr without a group)
    const int* d_full_to_reduced_map_;       ///< Device map full -> reduced index (lent by the backend; nullptr without a group)
    int n_basis_;                            ///< Values of a propagator: the reduced basis, or the full grid without a group
    /// The full grid of a step, per stream (provide_full_grid()): the input expanded from
    /// the reduced basis, which a step turns into its result in place
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_[MAX_STREAMS];
    /// A second full grid per stream, for steps that cannot run in place (the rs bonds)
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_out_[MAX_STREAMS];
    /// @}

    /**
     * @brief The shared context of a solver.
     *
     * @param cb                      Computation box
     * @param molecules               Molecules container
     * @param n_streams               Number of parallel streams
     * @param streams                 The backend's CUDA streams
     * @param space_group             Space group of the reduced basis, or nullptr (real fields only)
     * @param d_reduced_basis_indices Device map reduced -> full index (lent by the backend; nullptr without a group)
     * @param d_full_to_reduced_map   Device map full -> reduced index (lent by the backend; nullptr without a group)
     */
    CudaSolver(ComputationBox<T>* cb, Molecules* molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
               SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map);

    /// With a space group, allocate d_q_full_ and, with separate_output, d_q_full_out_ for
    /// every stream unless they are; nothing without a group
    void provide_full_grid(bool separate_output);

    /// d_q itself without a space group; otherwise its expansion from the reduced basis into
    /// d_full, by default d_q_full_[STREAM]
    CuDeviceData<T>* expand_to_full_grid(int STREAM, CuDeviceData<T>* d_q, CuDeviceData<T>* d_full = nullptr);

    /// Where a step writes its full-grid result: d_q_out itself without a space group,
    /// otherwise d_q_full_[STREAM], whose expanded input the step reads before it writes
    CuDeviceData<T>* full_grid_output(int STREAM, CuDeviceData<T>* d_q_out);

    /// Store the full-grid result d_full in d_q_out, reduced to the basis with a space group
    /// (d_full is d_q_out without one), then multiply the mask
    void store_full_grid(int STREAM, const CuDeviceData<T>* d_full, CuDeviceData<T>* d_q_out, const double* d_q_mask);

    /// d_q *= mask in the basis of the propagators; nothing for nullptr
    void multiply_mask(int STREAM, CuDeviceData<T>* d_q, const double* d_q_mask);

    /**
     * @brief The Boltzmann factors of the fields at every contour step, the body of update_dw().
     *
     * d_exp_dw[ds] = exp(-factor·ds·w) and, if with_half, d_exp_dw_half[ds] =
     * exp(-factor_half·ds·w) at every contour step of the chains; then waits for the device.
     * Tables on the full grid take the fields of a space group expanded from the reduced
     * basis, staged in the table written last.
     *
     * @param d_w                  Fields by monomer type (device), in the basis of the propagators
     * @param reduced_basis_tables The tables are in the reduced basis (the CrysFFT steps), not on the full grid
     * @param with_half            Also d_exp_dw_half
     * @param factor               Factor of ds in d_exp_dw
     * @param factor_half          Factor of ds in d_exp_dw_half
     */
    void compute_exp_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w, bool reduced_basis_tables,
                        bool with_half, double factor, double factor_half);
};
#endif
