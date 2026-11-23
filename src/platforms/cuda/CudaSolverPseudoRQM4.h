/**
 * @file CudaSolverPseudoRQM4.h
 * @brief GPU pseudo-spectral solver for continuous chain model using RQM4.
 *
 * This header provides CudaSolverPseudoRQM4, the CUDA implementation
 * of the pseudo-spectral method with RQM4 (Ranjan-Qin-Morse 4th-order using
 * Richardson extrapolation) for continuous Gaussian chains.
 *
 * **GPU Architecture:**
 *
 * - Multiple CUDA streams for concurrent propagator computation
 * - cuFFT for GPU-accelerated FFT operations
 * - Custom CUDA kernels for element-wise operations
 *
 * **Stream Organization:**
 *
 * Each stream index has one CUDA stream (streams[STREAM]), on which the
 * kernels and transforms of its propagators run.
 *
 * **cuFFT Plans:**
 *
 * The cuFFT plans of periodic boxes (batch 2 for the full step with the first half
 * step, batch 1 for the second half step) and the cosine and sine series of wall
 * boxes are owned by CudaSolverPseudoBase
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * - Index 0-2: Diagonal components (σ₁, σ₂, σ₃) for length optimization
 * - Index 3-5: Off-diagonal components (σ₁₂, σ₁₃, σ₂₃) for angle optimization
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoDiscrete for discrete chain version
 * @see CpuSolverPseudoRQM4 for CPU version
 * @see docs/theory/StressTensor.md for detailed derivation
 */

#ifndef CUDA_SOLVER_PSEUDO_RQM4_H_
#define CUDA_SOLVER_PSEUDO_RQM4_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaSolverPseudoBase.h"
#include "CudaCommon.h"
#include "CudaCrysFFTBase.h"

class SpaceGroup;  // Forward declaration
class CudaCrysFFTBase;  // Forward declaration

/**
 * @class CudaSolverPseudoRQM4
 * @brief GPU pseudo-spectral solver for continuous Gaussian chains using RQM4.
 *
 * Implements operator splitting with RQM4 (4th-order Richardson extrapolation)
 * on GPU using cuFFT and custom CUDA kernels.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * Same algorithm as CPU version:
 *     q(s+ds) = (4/3) q^(ds/2,ds/2) - (1/3) q^(ds)
 *
 * On a periodic full grid the full step and the first half step take their transforms
 * together, as one cuFFT batch of two fields; the second half step follows alone. Walls
 * (bond_step()) and the CrysFFT path take one field at a time.
 *
 * **Memory per Stream:**
 *
 * - d_q_step_2_one: the two half steps (the reduced basis on the CrysFFT path)
 * - d_q_step_1_two: the full step and the first half step side by side for the batched
 *   cuFFT (2*M; full-grid steps with periodic BC), the full step staying in the first half
 * - d_q_step_1_one: the full step of the other paths (the reduced basis on the CrysFFT path)
 *
 * The spectra go to the base's scratch, which the stress shares.
 */
template <typename T>
class CudaSolverPseudoRQM4 : public CudaSolverPseudoBase<T>
{
private:
    using Base = CudaSolverPseudoBase<T>;
    using Base::cb; using Base::pseudo;
    using Base::n_streams; using Base::is_periodic_; using Base::streams;
    using Base::n_basis_; using Base::use_crysfft_; using Base::crysfft_identity_map_;
    using Base::crysfft_physical_size_;
    using Base::fft_forward; using Base::fft_backward; using Base::spectrum;

    /// @name Workspace Arrays (per stream)
    /// @{
    CudaDeviceBuffer<CuDeviceData<T>> d_q_step_1_one[MAX_STREAMS];      ///< The full step (not batched)
    CudaDeviceBuffer<CuDeviceData<T>> d_q_step_2_one[MAX_STREAMS];      ///< The two half steps
    CudaDeviceBuffer<CuDeviceData<T>> d_q_step_1_two[MAX_STREAMS];      ///< The full step and the first half step, batched
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for continuous chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param space_group         Space group of the reduced basis, or nullptr
     */
    CudaSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
                            SpaceGroup* space_group);

    /**
     * @brief Update Boltzmann factors.
     * @param d_w Potential fields (device)
     */
    void update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w) override;

    /**
     * @brief Advance propagator by one step using RQM4.
     *
     * @param STREAM      Stream index for concurrent execution
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask (device, nullptr if none)
     * @param ds_index    Index for ds value (0-based, for per-block ds support)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /// Half-bond step (empty for continuous chains).
    void advance_propagator_half_bond_step(
        const int, CuDeviceData<T> *, CuDeviceData<T> *, std::string) override {};

    void allocate_work(int n_field) override;
    /// CrysFFT steps of a block: a full step and two half steps
    std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const override
    {
        return {this->get_effective_diffusion_coeff(monomer_type, ds_index, false), this->get_effective_diffusion_coeff(monomer_type, ds_index, true)};
    }
};
#endif
