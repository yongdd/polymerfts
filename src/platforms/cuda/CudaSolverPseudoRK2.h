/**
 * @file CudaSolverPseudoRK2.h
 * @brief GPU pseudo-spectral solver for continuous chain model using RK2.
 *
 * This header provides CudaSolverPseudoRK2, the CUDA implementation
 * of the pseudo-spectral method with RK2 (Rasmussen-Kalosakas 2nd-order)
 * for continuous Gaussian chains.
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
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step (2 FFTs) while RQM4 combines full and half
 * steps via Richardson extrapolation (6 FFTs). RK2 is faster but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoRQM4 for 4th-order version
 * @see CpuSolverPseudoRK2 for CPU version
 */

#ifndef CUDA_SOLVER_PSEUDO_RK2_H_
#define CUDA_SOLVER_PSEUDO_RK2_H_

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
#include "CudaFFT.h"
#include "CudaCrysFFT.h"
#include "FFT.h"
#include "SpaceGroup.h"

class CudaCrysFFTBase;

/**
 * @class CudaSolverPseudoRK2
 * @brief GPU pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements operator splitting with RK2 (2nd-order Rasmussen-Kalosakas method)
 * on GPU using cuFFT and custom CUDA kernels.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **RK2 (Operator Splitting):**
 *
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Memory per Stream:**
 *
 * - d_q_work: real-space workspace of advance_propagator() on the full grid (M); the
 *   spectra go to the base's scratch (d_qk_ or d_rk_1_), which the stress shares
 */
template <typename T>
class CudaSolverPseudoRK2 : public CudaSolverPseudoBase<T>
{
private:
    using Base = CudaSolverPseudoBase<T>;
    using Base::cb; using Base::molecules; using Base::pseudo;
    using Base::n_streams; using Base::is_periodic_; using Base::streams;

    using Base::space_group_; using Base::d_reduced_basis_indices_; using Base::d_full_to_reduced_map_;
    using Base::n_basis_; using Base::d_q_full_in_; using Base::d_q_full_out_;
    using Base::use_crysfft_; using Base::crysfft_identity_map_;
    using Base::crysfft_physical_size_;
    using Base::d_crysfft_phys_to_reduced_; using Base::d_crysfft_reduced_to_phys_;
    using Base::crysfft_; using Base::d_crysfft_in_; using Base::d_crysfft_out_;
    using Base::fft_; using Base::fft_forward; using Base::fft_backward;
    using Base::d_qk_; using Base::d_rk_1_; using Base::d_rk_2_;

    /// @name Workspace Arrays (per stream)
    /// @{
    CudaDeviceBuffer<CuDeviceData<T>> d_q_work[MAX_STREAMS];            ///< Real-space workspace (M)
    /// @}


public:
    /**
     * @brief Construct GPU pseudo-spectral solver for continuous chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param space_group             Space group of the reduced basis, or nullptr
     * @param d_reduced_basis_indices Device map reduced -> full index (lent by the backend)
     * @param d_full_to_reduced_map   Device map full -> reduced index (lent by the backend)
     */
    CudaSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
                           SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map);

    /// Destructor. Frees the workspaces (the base class destroys the cuFFT plans).
    ~CudaSolverPseudoRK2();

    /**
     * @brief Update Boltzmann factors.
     * @param d_w Potential fields (device)
     */
    void update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w) override;

    /**
     * @brief Advance propagator by one step using RK2.
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

    void allocate_work(int n_field, bool full_grid) override;
    /// CrysFFT steps of a block: full steps only
    std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const override
    {
        return {this->crysfft_coeff(monomer_type, ds_index, 6.0)};
    }
};
#endif
