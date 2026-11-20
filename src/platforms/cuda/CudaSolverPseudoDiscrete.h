/**
 * @file CudaSolverPseudoDiscrete.h
 * @brief GPU pseudo-spectral solver for discrete chain model.
 *
 * This header provides CudaSolverPseudoDiscrete, the CUDA implementation
 * of the pseudo-spectral method for discrete chain propagators using
 * the Chapman-Kolmogorov integral equation.
 *
 * **Chapman-Kolmogorov Equation (N-1 Bond Model):**
 *
 * For discrete chains, the propagator satisfies:
 *     q(r, i+1) = exp(-w(r)*ds) * integral g(r-r') q(r', i) dr'
 *
 * In Fourier space:
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
 * See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * Half-bond steps (ĝ^(1/2)(k) = exp(-b²|k|²ds/12)) are used only at
 * chain ends and junction points.
 *
 * **GPU Resources:**
 *
 * Same stream and cuFFT organization as continuous solver,
 * but uses bond convolution operators instead of diffusion.
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoRQM4 for continuous chain version
 * @see CpuSolverPseudoDiscrete for CPU version
 */

#ifndef CUDA_SOLVER_PSEUDO_DISCRETE_H_
#define CUDA_SOLVER_PSEUDO_DISCRETE_H_

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

class SpaceGroup;  // Forward declaration
class CudaCrysFFTBase;  // Forward declaration

/**
 * @class CudaSolverPseudoDiscrete
 * @brief GPU pseudo-spectral solver for discrete chain model.
 *
 * Implements discrete chain propagator updates using the Chapman-Kolmogorov
 * integral equation on the GPU.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **N-1 Bond Model:**
 *
 * Full segment step: q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 * - ĝ(k) = exp(-b²|k|²ds/6) is the full bond function
 * - exp(-w*ds) is the full-segment Boltzmann factor
 */
template <typename T>
class CudaSolverPseudoDiscrete : public CudaSolverPseudoBase<T>
{
private:
    using Base = CudaSolverPseudoBase<T>;
    using Base::cb; using Base::pseudo; using Base::streams;
    using Base::use_crysfft_;
    using Base::fft_; using Base::fft_forward; using Base::fft_backward;
    using Base::d_qk_; using Base::d_rk_1_;

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for discrete chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param space_group             Space group of the reduced basis, or nullptr
     * @param d_reduced_basis_indices Device map reduced -> full index (lent by the backend)
     * @param d_full_to_reduced_map   Device map full -> reduced index (lent by the backend)
     */
    CudaSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
                                SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map);

    /// Destructor. Frees GPU resources.
    ~CudaSolverPseudoDiscrete();

    /**
     * @brief True when d_exp_dw is stored in the reduced basis (size n_basis).
     *
     * In CrysFFT mode the solver keeps exp(-w*ds) in the reduced basis;
     * otherwise (standard FFT with space group) it is stored on the full grid.
     * Callers that consume exp_dw in the reduced basis must convert when
     * this returns false.
     */
    bool exp_dw_stored_reduced() const { return use_crysfft_; }

    /// Update Boltzmann factors.
    void update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w) override;

    /**
     * @brief Advance propagator by one full segment.
     *
     * Computes: q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
     * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
     *
     * @param STREAM      Stream index
     * @param d_q_in      Input propagator q(i)
     * @param d_q_out     Output propagator q(i+1)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask
     * @param ds_index    Index for ds value (discrete chains use ds_index=0)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /**
     * @brief Advance by half bond step only.
     *
     * Computes: q' = FFT^-1[ ĝ^(1/2)(k) * FFT[q] ]
     * where ĝ^(1/2)(k) = exp(-b²|k|²ds/12) is the half-bond function.
     *
     * Used at chain ends and junction points.
     *
     * @param STREAM      Stream index
     * @param d_q_in      Input propagator
     * @param d_q_out     Output after half-bond convolution
     * @param monomer_type Monomer type
     */
    void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type) override;

    /// CrysFFT steps of a block: its bonds, and half bonds, which always take the
    /// global ds (index 0; see advance_propagator_half_bond_step())
    std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const override
    {
        return {this->crysfft_coeff(monomer_type, ds_index, 6.0), this->crysfft_coeff(monomer_type, 0, 12.0)};
    }
    /// Boltzmann bond factor of the stress integrand (see CudaSolverPseudoBase)
    const double* get_stress_boltz_bond(
        const std::string& monomer_type, bool is_half_bond_length) const override;
    bool stress_has_bond_factor() const override { return true; }
};
#endif
