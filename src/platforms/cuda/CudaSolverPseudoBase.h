/**
 * @file CudaSolverPseudoBase.h
 * @brief Shared state and operations of the CUDA pseudo-spectral solvers.
 *
 * CudaSolverPseudoRQM4, CudaSolverPseudoRK2 and CudaSolverPseudoDiscrete differ
 * only in how they advance a propagator (and, for the discrete model, in the
 * bond factor used by the stress). Everything else lives here, mirroring
 * CpuSolverPseudoBase on the CPU side:
 *
 * - the Pseudo helper (the computation box, streams and space group are
 *   CudaSolver's),
 * - the spectral transforms of the full grid: cuFFT plans for periodic boundaries
 *   (fft_forward / fft_backward, batch of 1 or 2 fields; the plans of a stream
 *   share one work area) and the cosine and sine series of CudaMakhoulTransform
 *   for walls, behind one bond step (bond_step()) and one forward transform
 *   (transform_forward()), the counterparts of transform_forward() and
 *   transform_backward() on the CPU,
 * - the spectral scratch of the full-grid path, which the propagator steps and
 *   the segment stress share,
 * - the stress reduction workspace,
 * - the CrysFFT fast path of a space group.
 *
 * The work of a stream is one StreamWork, as the CPU solvers keep one set of
 * work arrays per thread. Derived classes add only their steps and, for RQM4, the
 * buffers of its full and half steps. Every resource is held by an owning member
 * (CudaDeviceBuffer, CufftPlan, std::unique_ptr), so the solvers have no cleanup
 * code and leave nothing behind when a constructor throws.
 *
 * Nothing the chosen path does not use is allocated: the space group, a
 * constructor argument, decides it. The constructor of each derived solver ends
 * with init_path(), which sets up the CrysFFT path of the group and allocates the
 * buffers and plans of the path (provide_work()); prepare_stress() calls
 * provide_work() again before its loop, which allocates the stress workspace and
 * what the stress needs when the force-off-diagonal flag has turned it to the full
 * grid since. A run without a stress holds no stress workspace. With the Pmmm or
 * recursive 3m CrysFFT and its stress fast path the solver holds no full-grid
 * transform, work buffer or Pseudo table.
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
#include "CudaMakhoulTransform.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaCrysFFT.h"   // CudaCrysFFTBase
#include "PseudoSolverHelpers.h"
#include "SpaceGroup.h"   // CrysFFTEngine

struct CrysFFTPlan;

template <typename T>
class CudaSolverPseudoBase : public CudaSolver<T>
{
protected:
    // The shared context (CudaSolver)
    using CudaSolver<T>::cb; using CudaSolver<T>::molecules;
    using CudaSolver<T>::n_streams; using CudaSolver<T>::streams;
    using CudaSolver<T>::space_group_; using CudaSolver<T>::n_basis_;
    using CudaSolver<T>::expand_to_full_grid;

    /// Pseudo-spectral tables of the full grid (Fourier basis, Boltzmann bond factors),
    /// created when a full-grid path first needs them (update_pseudo())
    std::unique_ptr<CudaPseudo<T>> pseudo;
    int dim_;                     ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;            ///< True if all BCs are periodic

    /**
     * The work of one stream (see provide_work()). The kernels and transforms of a
     * propagator run on its stream, so the streams share no buffer.
     */
    struct StreamWork
    {
        /// @name Transforms of the full-grid path (init_fft())
        /// @{
        CudaDeviceBuffer<double> d_fft_work;   ///< Periodic BC: the one cuFFT work area of the plans
        CufftPlan plan_forward[2];             ///< Periodic BC: forward plans, index = batch - 1
        CufftPlan plan_backward[2];            ///< Periodic BC: backward plans, index = batch - 1
        std::unique_ptr<CudaMakhoulTransform<T>> series;   ///< Walls: the cosine and sine series
        /// @}

        /// Spectral scratch of the full-grid path: one or two spectra of spectrum_doubles()
        /// doubles, the second for the RQM4 half step (a batch of two) or the second
        /// propagator of the stress. The propagator steps and the segment stress share it:
        /// they run one after another on the stream
        CudaDeviceBuffer<double> d_spectra;

        /// @name Stress reduction workspace, from the first prepare_stress() on
        /// The CrysFFT fast path sums its reduced-basis integrand in d_stress_sum with the
        /// CUB scratch d_temp_storage; the full-grid path forms the integrand in d_q_multi
        /// and sums it by blocks into d_stress_partial
        /// @{
        size_t temp_storage_bytes = 0;
        CudaDeviceBuffer<unsigned char> d_temp_storage;
        CudaDeviceBuffer<double> d_stress_sum;                ///< CrysFFT fast path (real fields): the integrand
        CudaDeviceBuffer<CuDeviceData<T>> d_q_multi;
        CudaDeviceBuffer<cuDoubleComplex> d_qk_conj;          ///< q2(-k) gather of the stress (complex fields, periodic BC)
        CudaDeviceBuffer<CuDeviceData<T>> d_stress_partial;   ///< Per-block partial sums of the stress components
        /// @}

        /// @name CrysFFT path: the transform of the stream and its physical-grid buffers
        /// @{
        std::unique_ptr<CudaCrysFFTBase> crysfft;
        CudaDeviceBuffer<double> d_crysfft_in;
        CudaDeviceBuffer<double> d_crysfft_out;
        /// @}
    };
    StreamWork work_[MAX_STREAMS];

    int max_fft_batch_;   ///< Largest batch a derived solver transforms at once (1 or 2)

    // CrysFFT fast path (crystallographic FFT on the physical sub-grid)
    bool use_crysfft_;                               ///< True when the CrysFFT path is enabled (a space group, 3D, periodic, real fields)
    CrysFFTEngine crysfft_engine_;                   ///< The engine of the CrysFFT path (plan_crysfft())
    bool crysfft_identity_map_;                      ///< True when reduced basis matches physical grid
    int crysfft_physical_size_;
    CudaDeviceBuffer<int> d_crysfft_phys_to_reduced_;   ///< Device map: physical index -> reduced index
    CudaDeviceBuffer<int> d_crysfft_reduced_to_phys_;   ///< Device map: reduced index -> physical index
    CudaDeviceBuffer<int> d_crysfft_orbit_counts_;      ///< Device orbit counts (stress fast path)
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
     */
    CudaSolverPseudoBase(
        ComputationBox<T>* cb, Molecules *molecules, int n_streams,
        cudaStream_t streams[MAX_STREAMS],
        SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map,
        int max_fft_batch, bool with_exp_dw_half);

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
     * otherwise) and, when full_grid_path(), the full-grid transforms, spectral
     * scratch, reduced-basis expansion buffer (d_q_full_) and Pseudo tables of the
     * current cell, with the stress workspace from the first prepare_stress() on;
     * then the derived solver's buffers (allocate_work()). Buffers are kept once
     * allocated; one of another size is replaced. Called by init_path() and,
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

    /// Doubles of one spectrum of a full-grid field: the complex spectrum of the FFT (periodic
    /// BC), or the cosine and sine series of the walls (real, or (re, im) pairs of a complex field)
    size_t spectrum_doubles() const;
    /// Spectrum i (0 or 1) of the spectral scratch of a stream
    double* spectrum(int STREAM, int i) { return work_[STREAM].d_spectra.get() + i * spectrum_doubles(); }

    /**
     * @brief The spectrum of a full-grid field on its stream: the FFT (periodic BC) or the
     *        cosine and sine series of the walls, half of FFTW's REDFT10/RODFT10 per axis as
     *        on the CPU (MakhoulFFT), which the stress tables take.
     *
     * @param STREAM Stream index
     * @param d_in   Full-grid field (kept)
     * @param d_out  Its spectrum (spectrum())
     */
    void transform_forward(int STREAM, CuDeviceData<T>* d_in, double* d_out);

    /**
     * @brief The bond convolution of a full-grid field on its stream, without the
     *        normalization 1/M: d_out = M · F⁻¹[scale · boltz · F[d_in]].
     *
     * Periodic BC: the unnormalized cuFFT, the spectrum times scale · boltz. Walls: the
     * cosine and sine series (transform_forward()) and scale times REDFT01/RODFT01 of the
     * coefficients times boltz, M times their inverse, the factors applied as the
     * coefficients are read. The caller multiplies 1/M into its next kernel or passes it as
     * scale, so periodic and wall boxes take one sequence of steps. The spectrum goes to
     * spectrum(STREAM, 0); d_out may be d_in.
     *
     * @param STREAM Stream index
     * @param d_in   Full-grid field
     * @param d_out  Result
     * @param boltz  Bond factor of each spectral value (a Pseudo table)
     * @param scale  Factor of the spectrum
     */
    void bond_step(int STREAM, CuDeviceData<T>* d_in, CuDeviceData<T>* d_out, const double* boltz, double scale);

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
     * @brief Allocate the derived solver's work buffers that are missing or of another size
     *        (default: none).
     *
     * @param n_field Values of a field in the propagator steps (n_basis_ on the
     *                CrysFFT path, the full grid otherwise)
     */
    virtual void allocate_work([[maybe_unused]] int n_field) {}

    /// True when the stress integrand carries the Boltzmann bond factor (discrete chains).
    virtual bool stress_has_bond_factor() const { return false; }

    /// The diffusion coefficient b² ds / 6 of a step, ds / 2 for a half step (effective_diffusion_coeff()).
    double get_effective_diffusion_coeff(const std::string& monomer_type, int ds_index, bool half_step) const
    {
        return effective_diffusion_coeff(*molecules, monomer_type, ds_index, half_step);
    }

    /**
     * @brief Coefficients (get_effective_diffusion_coeff()) of the CrysFFT steps that the
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

    /// Create the missing cuFFT plans (periodic BC) or cosine and sine series (walls) of every stream.
    void init_fft();
    /// Compute and upload the Pseudo tables (Boltzmann bond factors, Fourier basis) of the
    /// current cell, creating them at the first call.
    void update_pseudo();
    /// Select and build the CrysFFT fast path for the current space group (3D periodic, real fields).
    void init_crysfft();
    /// Upload the index maps of the plan and create the per-stream CrysFFT objects of its engine.
    void setup_crysfft(const CrysFFTPlan& plan);
    /// Rebuild every CrysFFT table for the current cell.
    void update_crysfft_tables();
    /// True when the segment stress takes the CrysFFT fast path (Pmmm or 3m, orthogonal 3D cell).
    bool crysfft_stress_fast_path() const;
    /**
     * @brief Segment stress of the CrysFFT fast path: V₁₁, V₂₂, V₃₃ into d_segment_stress[0..2].
     *
     * Mirrors the CPU fast path in CpuSolverPseudoBase::compute_single_segment_stress():
     * q2 on the physical grid, one k-space multiplier per group of equivalent axes
     * (Σ_{d in group} k_d², times exp(-k²·b²·ds/6) for discrete chains), back to the
     * reduced basis, the orbit-weighted sum with q1, and the group sum shared equally by
     * the axes of the group, scaled by b²·M_full/L_d² as in the full-grid path. Only the
     * diagonal is written: the path serves orthogonal cells, whose cross terms the caller
     * does not read.
     */
    void crysfft_segment_stress(int STREAM, double bond_length_sq,
                                const double* d_q_1, const double* d_q_2, double* d_segment_stress);

public:
    virtual ~CudaSolverPseudoBase();

    /**
     * @brief Stress contribution of one segment: V_ij = sum_k q1(k) q2(-k) v_i v_j.
     *
     * Shared by RQM4, RK2 and the discrete chain; they differ only in the
     * Boltzmann bond factor (get_stress_boltz_bond()). Paths: CrysFFT fast path for
     * orthogonal 3D reduced-basis runs, otherwise the spectra of transform_forward()
     * (cuFFT for periodic BC, the cosine and sine series for walls). Cross terms are
     * accumulated only for non-orthogonal boxes or when force_off_diagonal_stress_ is set.
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
