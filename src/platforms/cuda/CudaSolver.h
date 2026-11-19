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
 * maps, the propagators are in its reduced basis.
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
    virtual ~CudaSolver() {};

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
    /**
     * @brief The Boltzmann factors of the fields at every contour step, the body of update_dw().
     *
     * d_exp_dw[ds] = exp(-factor·ds·w) and, if with_half, d_exp_dw_half[ds] =
     * exp(-factor_half·ds·w), n_table values each; then waits for the device.
     *
     * @param d_w               Fields by monomer type (device), in the basis of the propagators
     * @param mapping           Contour steps of the chains (ds of each ds index)
     * @param n_table           Values of a table: the full grid, or the reduced basis on the CrysFFT path
     * @param d_full_to_reduced nullptr when the fields come in the basis of the tables; otherwise
     *                          they come in the reduced basis of a space group and are expanded to
     *                          the full grid of the tables through this map (device), staged in the
     *                          table written last
     * @param with_half         Also d_exp_dw_half
     * @param factor            Factor of ds in d_exp_dw
     * @param factor_half       Factor of ds in d_exp_dw_half
     */
    void compute_exp_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w,
                        const ContourLengthMapping& mapping, int n_table, const int* d_full_to_reduced,
                        bool with_half, double factor, double factor_half);
};
#endif
