/**
 * @file CudaSolverPseudoBase.h
 * @brief Shared state and operations of the CUDA pseudo-spectral solvers.
 *
 * CudaSolverPseudoRQM4, CudaSolverPseudoRK2 and CudaSolverPseudoDiscrete differ
 * only in how they advance a propagator (and, for the discrete model, in the
 * bond factor used by the stress). Everything else lives here, mirroring
 * CpuSolverPseudoBase on the CPU side:
 *
 * - computation box, Pseudo helper and stream handles,
 * - the spectral transforms: cuFFT plans for periodic boundaries
 *   (fft_forward / fft_backward, batch of 1 or 2 fields; the plans of a stream
 *   share one work area) and per-stream DCT/DST objects (fft_) for
 *   non-periodic boundaries,
 * - the stress reduction workspace,
 * - the space-group reduced basis and the CrysFFT fast path.
 *
 * Derived classes own only their propagator work buffers, because the number
 * and size of those buffers is what distinguishes the three methods. Every
 * resource is held by an owning member (CudaDeviceBuffer, CufftPlan,
 * std::unique_ptr), so the solvers have no cleanup code and leave nothing behind
 * when a constructor throws.
 *
 * Nothing the chosen path does not use is allocated: the space group, a
 * constructor argument, decides it. The constructor of each derived solver ends
 * with init_path(), which sets up the CrysFFT path of the group and allocates the
 * buffers and plans of the path (provide_work()); prepare_stress() calls
 * provide_work() again before its loop, which allocates what the stress needs
 * when the force-off-diagonal flag has turned it to the full grid since. With the
 * Pmmm or recursive 3m CrysFFT and its stress fast path the solver holds no
 * full-grid transform, work buffer or Pseudo table.
 */

#ifndef CUDA_SOLVER_PSEUDO_BASE_H_
#define CUDA_SOLVER_PSEUDO_BASE_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <cufft.h>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaPseudo.h"
#include "FFT.h"
#include "CudaFFT.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaCrysFFT.h"   // CudaCrysFFTBase, CudaCrysFFTMode
#include "CudaCrysFFTStress.h"

class SpaceGroup;
struct CrysFFTBasisMapping;

template <typename T>
class CudaSolverPseudoBase : public CudaSolver<T>
{
protected:
    ComputationBox<T>* cb;        ///< Computation box
    Molecules *molecules;         ///< Molecules container
    /// Pseudo-spectral tables of the full grid (Fourier basis, Boltzmann bond factors),
    /// created when a full-grid path first needs them (update_pseudo())
    std::unique_ptr<CudaPseudo<T>> pseudo;
    int n_streams;                ///< Number of parallel streams
    int dim_;                     ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;            ///< True if all BCs are periodic
    cudaStream_t streams[MAX_STREAMS];  ///< CUDA stream per stream index

    // Spectral transforms of the full-grid path (created by init_fft())
    int max_fft_batch_;                      ///< Largest batch a derived solver transforms at once (1 or 2)
    CufftPlan plan_for_[MAX_STREAMS][2];     ///< Periodic BC: forward plans, index = batch - 1
    CufftPlan plan_bak_[MAX_STREAMS][2];     ///< Periodic BC: backward plans, index = batch - 1
    CudaDeviceBuffer<double> d_fft_work_[MAX_STREAMS];   ///< Periodic BC: the one cuFFT work area of a stream's plans
    std::unique_ptr<CudaFFTBase<T>> fft_[MAX_STREAMS];       ///< Non-periodic BC: per-stream DCT/DST transform objects (issued on streams[i])

    // Stress reduction workspace (per stream). The CrysFFT fast path sums its
    // reduced-basis integrand in d_stress_sum with the CUB scratch d_temp_storage
    // (allocated by prepare_stress()); the full-grid path forms the integrand in
    // d_q_multi and sums it by blocks into d_stress_partial_ (see provide_work())
    size_t temp_storage_bytes[MAX_STREAMS];
    CudaDeviceBuffer<unsigned char> d_temp_storage[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_sum[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_sum_out[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_q_multi[MAX_STREAMS];
    CudaDeviceBuffer<cuDoubleComplex> d_qk_conj_[MAX_STREAMS];   ///< q2(-k) gather of the stress (complex fields; nullptr for real)
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_partial_[MAX_STREAMS];   ///< Per-block partial sums of the stress components

    // Space group symmetry (reduced basis)
    SpaceGroup* space_group_;                        ///< Space group pointer (nullptr if not used)
    const int* d_reduced_basis_indices_;             ///< Device array: reduced -> full index mapping (lent by the backend)
    const int* d_full_to_reduced_map_;               ///< Device array: full -> reduced index mapping (lent by the backend)
    int n_basis_;                                    ///< Number of reduced basis points (0 without a space group)
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_in_[MAX_STREAMS];      ///< Work buffer: full grid input (per stream; full-grid path)
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_out_[MAX_STREAMS];     ///< Work buffer: full grid output (per stream; full-grid path)

    // CrysFFT fast path (crystallographic FFT on the physical sub-grid)
    bool use_crysfft_;                               ///< True when the CrysFFT path is enabled (a space group, 3D, periodic, real fields)
    CudaCrysFFTMode crysfft_mode_;
    bool crysfft_identity_map_;                      ///< True when reduced basis matches physical grid
    int crysfft_physical_size_;
    CudaDeviceBuffer<int> d_crysfft_phys_to_reduced_;   ///< Device map: physical index -> reduced index
    CudaDeviceBuffer<int> d_crysfft_reduced_to_phys_;   ///< Device map: reduced index -> physical index
    CudaDeviceBuffer<int> d_crysfft_orbit_counts_;      ///< Device orbit counts (stress fast path)
    std::unique_ptr<CudaCrysFFTBase> crysfft_[MAX_STREAMS];   ///< Per-stream CrysFFT objects
    CudaDeviceBuffer<double> d_crysfft_in_[MAX_STREAMS];              ///< Per-stream physical grid input
    CudaDeviceBuffer<double> d_crysfft_out_[MAX_STREAMS];             ///< Per-stream physical grid output
    // k-space tables of the current cell, one per key for all streams: built before the
    // propagator and stress loops, which only look them up, and rebuilt in place when
    // the cell changes (update_crysfft_tables())
    std::map<double, CudaDeviceBuffer<double>> crysfft_boltz_;   ///< Boltzmann factors per step coefficient
    std::map<std::pair<int, double>, CudaDeviceBuffer<double>> crysfft_multipliers_;   ///< Stress multipliers per (axis group, bond coefficient)

    /**
     * The derived constructor ends with init_path(), which needs its overrides.
     *
     * @param space_group               Space group of the reduced basis, or nullptr
     * @param d_reduced_basis_indices   Device map reduced -> full index (lent by the backend; nullptr without a group)
     * @param d_full_to_reduced_map     Device map full -> reduced index (lent by the backend; nullptr without a group)
     * @param max_fft_batch             Largest number of fields the derived solver
     *                                  transforms in one cuFFT call (1 or 2).
     * @param with_exp_dw_half          Also allocate d_exp_dw_half (half-step Boltzmann factors).
     * @param complex_requires_periodic Reject complex fields with non-periodic BCs.
     */
    CudaSolverPseudoBase(
        ComputationBox<T>* cb, Molecules *molecules, int n_streams,
        cudaStream_t streams[MAX_STREAMS],
        SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map,
        int max_fft_batch, bool with_exp_dw_half, bool complex_requires_periodic);

    /**
     * @brief Set up the CrysFFT path of the space group (3D, periodic, real fields)
     *        and allocate the work of the path (provide_work()).
     *
     * Called at the end of the derived constructors: it uses their
     * crysfft_block_coeffs() and allocate_work().
     */
    void init_path();

    /**
     * @brief Allocate what the path in use needs and is missing.
     *
     * The exp(-w ds) tables (reduced basis on the CrysFFT path, full grid
     * otherwise) and, when full_grid_path(), the full-grid transforms, stress
     * workspace, reduced-basis expansion buffers and Pseudo tables of the current
     * cell; then the derived solver's buffers (allocate_work()). Buffers are kept
     * once allocated; one of another size is replaced. Called by init_path() and,
     * as the force-off-diagonal flag can turn the stress to the full grid, by
     * prepare_stress() before its loop.
     */
    void provide_work();

    /// Values of the spectrum of a full-grid field, as Pseudo::get_total_complex_grid()
    /// counts them (the r2c half spectrum for real fields with periodic BCs, the full
    /// grid otherwise), without the Pseudo tables
    int total_complex_grid() const;

    /// Forward FFT of `batch` consecutive full-grid fields on stream `stream` (periodic BC only).
    void fft_forward(int stream, CuDeviceData<T>* d_in, cuDoubleComplex* d_out, int batch = 1);
    /// Backward (unnormalized) FFT, inverse of fft_forward() (periodic BC only).
    void fft_backward(int stream, cuDoubleComplex* d_in, CuDeviceData<T>* d_out, int batch = 1);

    /**
     * @brief Full-grid input of a spectral step: d_q_in itself, or its expansion
     *        from the reduced basis into d_q_full_in_ when a space group is set.
     */
    CuDeviceData<T>* expand_to_full_grid(int STREAM, CuDeviceData<T>* d_q_in);

    /// Full-grid output of a spectral step: d_q_out itself, or d_q_full_out_ when a space group is set.
    CuDeviceData<T>* full_grid_output(int STREAM, CuDeviceData<T>* d_q_out);

    /**
     * @brief Store the full-grid result d_full in d_q_out, reducing it to the
     *        basis when a space group is set, then apply the mask (if any).
     */
    void store_full_grid(int STREAM, CuDeviceData<T>* d_full, CuDeviceData<T>* d_q_out, const double* d_q_mask);

    /// d_q *= mask on the propagator basis (reduced basis with a space group); no-op for nullptr.
    void multiply_mask(int STREAM, CuDeviceData<T>* d_q, const double* d_q_mask);

    /**
     * @brief CrysFFT diffusion of reduced-basis propagators on the physical sub-grid.
     *
     * d_q_out = post · D · (mid · D ·) pre · d_q_in with D = exp(coeff ∇²): the
     * factors are reduced-basis Boltzmann factors, nullptr to skip one, and a
     * non-null mid adds the second diffusion step. d_q_out may alias d_q_in.
     */
    void crysfft_diffuse(int STREAM, double* d_q_in, double* d_q_out, double coeff,
                         const double* pre, const double* mid, const double* post);

    /**
     * @brief Shared body of update_dw(): d_exp_dw = exp(-factor*ds*w) and,
     *        if with_half, d_exp_dw_half = exp(-factor_half*ds*w).
     */
    void update_dw_impl(
        const std::map<std::string, const CuDeviceData<T>*>& d_w,
        bool with_half, double factor, double factor_half);

    /**
     * @brief Device scratch a derived solver lends to compute_single_segment_stress().
     *
     * Buffers of the derived solver that are free while one segment's stress is
     * computed: qk_pair takes the forward FFTs of the two propagators
     * (2*M_COMPLEX), rk_1/rk_2 their DCT/DST coefficients (non-periodic BC; nullptr
     * when periodic): M_COMPLEX doubles each for real fields, 2*M_COMPLEX for complex
     * fields, whose real and imaginary parts come as interleaved pairs. With a space
     * group each propagator is expanded into d_q_full_in_ before its transform.
     */
    struct StressScratch
    {
        cuDoubleComplex* qk_pair;
        double* rk_1;
        double* rk_2;
    };
    virtual StressScratch stress_scratch(int STREAM) = 0;

    /**
     * @brief Boltzmann bond factor multiplied into the stress integrand (full-grid path).
     *
     * The default (continuous chains) returns nullptr; the discrete chain
     * overrides it with exp(-k^2 b^2 ds/6) (half-bond variant when requested).
     */
    virtual const double* get_stress_boltz_bond(
        const std::string& monomer_type, bool is_half_bond_length) const;

    /**
     * @brief True unless the Pmmm or recursive 3m CrysFFT takes the propagator
     *        steps and its fast path the segment stress, so that no full-grid
     *        transform, buffer or Pseudo table is used.
     *
     * ObliqueZ runs its stress on the full grid; the force-off-diagonal flag and
     * a non-orthogonal cell turn the stress fast path off.
     */
    bool full_grid_path() const;

    /**
     * @brief Allocate the derived solver's work buffers that are missing or of another size.
     *
     * @param n_field   Values of a field in the propagator steps (n_basis_ on the
     *                  CrysFFT path, the full grid otherwise)
     * @param full_grid Also the buffers of the full-grid path (its FFT workspaces
     *                  and the stress scratch of stress_scratch())
     */
    virtual void allocate_work(int n_field, bool full_grid) = 0;

    /// True when the stress integrand carries the Boltzmann bond factor (discrete chains).
    virtual bool stress_has_bond_factor() const { return false; }

    /**
     * @brief CrysFFT diffusion coefficient b²·Δs/divisor of one propagator step.
     *
     * @param divisor 6 for a full step, 12 for a half step
     */
    double crysfft_coeff(const std::string& monomer_type, int ds_index, double divisor) const;

    /**
     * @brief Coefficients (see crysfft_coeff()) of the CrysFFT steps that the
     *        method takes for a block of this monomer type and contour step.
     *
     * Their Boltzmann tables are built for every block when the CrysFFT objects
     * are created and when the cell changes, so the propagator loop only selects
     * them; crysfft_diffuse() rejects a coefficient whose table was not built.
     */
    virtual std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const = 0;

private:
    bool pseudo_current_ = false;            ///< pseudo's tables are those of the current cell (see update_pseudo())
    bool stress_basis_ = false;              ///< pseudo computes the stress basis (from the first prepare_stress())

    /// Create the missing cuFFT plans (periodic BC) or DCT/DST objects (non-periodic BC).
    void init_fft();
    /// Compute and upload the Pseudo tables (Boltzmann bond factors, Fourier basis) of the
    /// current cell, creating them at the first call.
    void update_pseudo();
    /// Select and build the CrysFFT fast path for the current space group (3D periodic, real fields).
    void init_crysfft();
    /// Upload the index maps and create the per-stream CrysFFT objects for `mode`.
    void setup_crysfft(CudaCrysFFTMode mode, const CrysFFTBasisMapping& map,
                       const std::array<double, 9>& m3_translations);
    /// Rebuild every CrysFFT table for the current cell.
    void update_crysfft_tables();
    /// Cell parameters [Lx, Ly, Lz, alpha, beta, gamma] of the computation box.
    std::array<double, 6> crysfft_cell() const;
    /// Keys of the stress multipliers of a bond: per axis group, (axes, b²·ds/6 or 0 without the bond factor).
    std::vector<std::pair<int, double>> crysfft_stress_keys(double bond_length_sq, bool include_bond_factor) const;
    /// True when the segment stress takes the CrysFFT fast path (Pmmm or 3m, orthogonal 3D cell).
    bool crysfft_stress_fast_path() const;
    /// b² of the monomer type, halved for a half bond (the stress integrand and its multiplier key).
    double stress_bond_length_sq(const std::string& monomer_type, bool is_half_bond_length) const;
    /// Arguments of the CrysFFT stress fast path on this stream.
    CudaCrysFFTStressArgs crysfft_stress_args(int STREAM, double bond_length_sq, bool include_bond_factor);

public:
    virtual ~CudaSolverPseudoBase();

    /**
     * @brief Stress contribution of one segment: V_ij = sum_k q1(k) q2(-k) v_i v_j.
     *
     * Shared by RQM4, RK2 and the discrete chain; they differ only in the
     * Boltzmann bond factor (get_stress_boltz_bond()) and in the scratch
     * buffers they lend (stress_scratch()). Paths: CrysFFT fast path for
     * orthogonal 3D reduced-basis runs, cuFFT for periodic BC, DCT/DST
     * for non-periodic BC (real fields). Cross terms are accumulated only for
     * non-orthogonal boxes or when force_off_diagonal_stress_ is set.
     *
     * @param STREAM           Stream index for concurrent execution
     * @param d_q_1            Forward propagator (device, in the basis of the propagators)
     * @param d_q_2            Backward propagator (device, in the basis of the propagators)
     * @param d_segment_stress Output: [V11, V22, V33, V12, V13, V23] (device)
     * @param monomer_type     Monomer type (bond length)
     * @param is_half_bond_length Half-bond contribution (discrete chain)
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_1, CuDeviceData<T> *d_q_2, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;

    /// A new cell: rebuilds the CrysFFT tables, and the Pseudo tables when the full-grid
    /// path is in use (otherwise prepare_stress() brings them to the cell if the flag
    /// turns the stress to the full grid).
    void update_laplacian_operator() override;

    /// Provide the stress path before the segment loop: the full-grid work, or the
    /// fast-path workspace and the CrysFFT stress multipliers of every block.
    void prepare_stress() override;
};

#endif  // CUDA_SOLVER_PSEUDO_BASE_H_
