/**
 * @file CpuSolverPseudoBase.h
 * @brief Common base class for CPU pseudo-spectral solvers.
 *
 * This header provides CpuSolverPseudoBase, a common base class that
 * consolidates shared functionality between CpuSolverPseudoRQM4
 * and CpuSolverPseudoDiscrete, including:
 *
 * - FFT object management (creation/destruction)
 * - Pseudo-spectral operator helper
 * - Boundary condition handling
 * - Transform dispatch methods
 * - Laplacian operator updates
 * - Stress computation (with customizable coefficient function)
 *
 * **Design Decision:**
 *
 * All simulations are conducted in fixed dimensions (1D, 2D, or 3D).
 * The FFT object is stored as `std::unique_ptr<FFT<T>> fft_` which points to FftwFFT<T, DIM>
 * or MklFFT<T, DIM> (periodic, by backend and dimensionality) or MakhoulFFT<T> (walls). This avoids
 * maintaining separate fft_1d, fft_2d, fft_3d pointers when only one is ever
 * used per solver.
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * - Index 0-2: Diagonal components (σ₁, σ₂, σ₃) for length optimization
 * - Index 3-5: Off-diagonal components (σ₁₂, σ₁₃, σ₂₃) for angle optimization
 *
 * @see CpuSolverPseudoRQM4 for continuous chain implementation
 * @see CpuSolverPseudoDiscrete for discrete chain implementation
 * @see docs/theory/StressTensor.md for detailed derivation
 */

#ifndef CPU_SOLVER_PSEUDO_BASE_H_
#define CPU_SOLVER_PSEUDO_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <complex>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFTFactory.h"
#include "SpaceGroup.h"
#include "CrysFFTBasisMapping.h"
#include "CpuCrysFFT.h"

/**
 * @class CpuSolverPseudoBase
 * @brief Common base class for CPU pseudo-spectral propagator solvers.
 *
 * Consolidates shared code between continuous and discrete chain solvers,
 * including FFT management, transform dispatch, and stress computation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Shared Functionality:**
 *
 * - FFT object creation/destruction with dimension dispatch
 * - Forward/backward transform methods
 * - Laplacian operator updates
 * - Stress computation framework
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - update_dw(): Different Boltzmann factor formulas
 * - advance_propagator(): Different propagation algorithms
 * - advance_propagator_half_bond_step(): Discrete-specific
 * - get_stress_boltz_bond() (optional): bond factor of the stress integrand; the base returns nullptr
 */
template <typename T>
class CpuSolverPseudoBase : public CpuSolver<T>
{
protected:
    /**
     * @brief Work arrays of one calling thread.
     *
     * The propagators run in OpenMP threads, so every thread needs its own
     * arrays. They are owned by the solver and released with it (thread_local
     * arrays would stay allocated until the threads end). Each step function
     * and the stress routine use them from start to end of one call; nothing
     * nests, so one set per thread is enough.
     */
    struct WorkArrays
    {
        std::vector<T> q_1, q_2;                     ///< Step results (RQM4 full step and half steps)
        std::vector<T> q_full_in, q_full_out;        ///< Full-grid copies of reduced-basis propagators
        std::vector<double> k_1, k_2;                ///< Spectral coefficients
        std::vector<double> phys_in, phys_out;       ///< CrysFFT physical grid
        std::vector<T> pair_1, pair_2;               ///< Stress: full-grid propagator pair
        std::vector<double> phys_1, phys_2;          ///< Stress: CrysFFT physical grid
        std::vector<double> reduced;                 ///< Stress: reduced-basis sum
    };

    /// Work arrays of the calling thread (created on first use)
    WorkArrays& work_arrays() const
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        return work_arrays_[std::this_thread::get_id()];
    }

    ComputationBox<T>* cb;       ///< Computation box for grid info
    Molecules* molecules;        ///< Molecules container
    std::string chain_model;     ///< Chain model identifier

    /**
     * @brief Spectral transform object for FFT/DCT/DST (handles all BCs).
     *
     * Points to FftwFFT<T, DIM> or MklFFT<T, DIM> (periodic) based on the selected
     * backend, where DIM is 1, 2, or 3 based on the simulation dimension, or to
     * MakhoulFFT<T> over the real FFT of that backend (walls).
     * All simulations are conducted in fixed dimensions, so only one
     * transform object is needed per solver instance.
     *
     * Uses an FFT<T> base class pointer to enable polymorphic calls without
     * dimension-specific casting or backend-specific dispatch.
     */
    std::unique_ptr<FFT<T>> fft_;

    FFTBackend fft_backend_;  ///< Selected FFT backend (FFTW or MKL)

    /// Fourier-space tables of the full grid (Boltzmann bond factors, stress basis), computed for
    /// the cell when a full-grid path uses them (update_pseudo())
    std::unique_ptr<Pseudo<T>> pseudo;

    bool is_periodic_;           ///< True if all BCs are periodic
    int dim_;                    ///< Number of dimensions (1, 2, or 3)

    /// Space group for reduced basis operations (nullptr if not used).
    SpaceGroup* space_group_;

    /// The CrysFFT engine (Pmmm DCT, recursive 3m or ObliqueZ of the selected
    /// library), called by all threads at once; nullptr without one
    std::unique_ptr<CpuCrysFFT> crysfft_;
    std::map<int, double> ds_values_;

    // k-space tables of the current cell, one per key for all threads: built before
    // the propagator and stress loops, which only read them, and rebuilt in place
    // when the cell changes (update_crysfft_tables())
    std::map<double, std::vector<double>> crysfft_boltz_;   ///< Boltzmann factors per step coefficient
    std::map<std::pair<int, double>, std::vector<double>> crysfft_multipliers_;   ///< Stress multipliers per (axis group, bond coefficient)

    std::vector<int> crysfft_reduced_indices_;   ///< Physical-grid index -> reduced-basis index (empty with the identity map)
    bool crysfft_identity_map_ = false;  ///< True when reduced basis matches physical grid

    bool use_crysfft() const { return crysfft_ != nullptr; }

    int get_crysfft_physical_size() const;

    /// The two physical-grid arrays of the calling thread's CrysFFT steps
    /// (WorkArrays::phys_in, phys_out), sized for the physical grid.
    std::pair<double*, double*> crysfft_grids(WorkArrays& work) const;

    /**
     * @brief One CrysFFT step on the physical grid: q_out = post · exp(coeff ∇²) (pre · q_in).
     *
     * `pre` and `post` are factors in the reduced basis (exp_dw) or nullptr; `pre`
     * multiplies q_in in place. The Boltzmann table of coeff, which
     * crysfft_block_coeffs() must list, is looked up, never built.
     */
    void crysfft_step(double* q_in, double* q_out, double coeff, const double* pre, const double* post) const;

    /**
     * @brief Coefficients (get_effective_diffusion_coeff()) of the CrysFFT steps
     *        that the method takes for a block of this monomer type and contour step.
     *
     * Their Boltzmann tables are built for every block when the CrysFFT path is
     * set up and rebuilt when the cell changes, so the propagator loop only looks
     * them up; crysfft_step() rejects a coefficient whose table was not built.
     */
    virtual std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const = 0;

    /**
     * @brief Perform forward transform.
     *
     * Delegates to fft_->forward() which handles all boundary conditions.
     *
     * @param rdata Real-space data input
     * @param cdata Fourier/DCT coefficient output
     */
    void transform_forward(T* rdata, double* cdata) { fft_->forward(rdata, cdata); }

    /**
     * @brief Perform backward transform.
     *
     * Delegates to fft_->backward() which handles all boundary conditions.
     *
     * @param cdata Fourier/DCT coefficient input
     * @param rdata Real-space data output
     */
    void transform_backward(double* cdata, T* rdata) { fft_->backward(cdata, rdata); }

    /**
     * @brief Backward transform with factors: rdata = r_factor · backward(k_factor · cdata)
     *        (FFT<T>::backward_with_factors()).
     *
     * k_factor (a Boltzmann bond factor, one value per coefficient) and r_factor
     * (one value per grid point) may be nullptr; cdata may be overwritten. The
     * result equals separate loops around transform_backward() bit for bit.
     */
    void transform_backward(double* cdata, const double* k_factor, T* rdata, const T* r_factor)
    {
        fft_->backward_with_factors(cdata, k_factor, rdata, r_factor);
    }

    /**
     * @brief Whether spectral coefficients are stored as complex pairs.
     *
     * Periodic BCs always use complex coefficients. For complex fields
     * (CL-FTS) the DCT/DST coefficients of the real and imaginary parts
     * are stored interleaved as complex pairs as well — the transforms
     * are real-linear, so each part transforms independently.
     */
    bool complex_coeffs() const
    {
        if constexpr (std::is_same<T, std::complex<double>>::value)
            return true;
        else
            return is_periodic_;
    }

    /**
     * @brief Initialize shared components in constructor.
     *
     * Sets up:
     * - Boundary condition checking (is_periodic_)
     * - FFT object creation (fft_) using specified backend
     * - The contour steps (ds_values_) and the Fourier-space tables (pseudo), which
     *   update_laplacian_operator() computes for the cell (the owner calls it after
     *   set_space_group(), as CpuComputationBackend does)
     *
     * @param cb Computation box
     * @param molecules Molecules container
     * @param backend FFT backend to use (FFTW or MKL)
     */
    void init_shared(ComputationBox<T>* cb, Molecules* molecules, FFTBackend backend);


    /**
     * @brief Get bond factor for stress computation.
     *
     * Returns the appropriate multiplier for stress coefficient:
     * - Continuous (default): Returns nullptr (no bond factor in stress)
     * - Discrete (override): Returns bond function ĝ(k) or half-bond ĝ^(1/2)(k)
     *
     * @param monomer_type Monomer type
     * @param is_half_bond_length Whether using half bond length
     * @return Pointer to bond function array, or nullptr if not applicable
     */
    virtual const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const;

    // ---- space group / CrysFFT setup (set_space_group) ----------------------
    /// Drop the CrysFFT engine, its tables and the index map.
    void reset_crysfft();
    /// Select and build the CrysFFT fast path for the current space group (3D periodic,
    /// real fields), with the Boltzmann tables of every block.
    void init_crysfft();
    /// Rebuild every CrysFFT table in place for the current cell.
    void update_crysfft_tables();
    /// Cell parameters [Lx, Ly, Lz, alpha, beta, gamma] of the computation box.
    std::array<double, 6> crysfft_cell() const;
    /// Store the physical-grid <-> reduced-basis map of the chosen CrysFFT (see plan_crysfft()).
    void adopt_crysfft_mapping(const CrysFFTBasisMapping& map, bool physical_basis_enabled);

    // ---- update_dw() helper --------------------------------------------------
    /**
     * @brief Bring the update_dw() input fields onto the grid the Boltzmann factors live on.
     *
     * With a space group and the standard FFT path the reduced-basis fields are
     * expanded to the full grid into `w_full_cache` (real fields only; complex
     * fields throw). Without a space group, or on the CrysFFT path where exp_dw
     * stays in the reduced basis, the input pointers are used as they are.
     * `w_use` maps each monomer type to the field pointer to read, and the
     * return value is the number of grid points each exp_dw array must hold.
     *
     * Serial use only: called from update_dw(), never from the OpenMP
     * propagator loops.
     */
    int expand_w_for_update_dw(const std::map<std::string, const T*>& w_input,
                               std::map<std::string, std::vector<T>>& w_full_cache,
                               std::map<std::string, const T*>& w_use) const;

    // ---- compute_single_segment_stress() helpers ----------------------------
    /// True when the segment stress takes the CrysFFT fast path (an engine with stress
    /// multipliers: Pmmm or recursive 3m; orthogonal 3D periodic cell, no angle optimization).
    bool crysfft_stress_fast_path() const;
    /// b² of the monomer type, halved for a half bond (the stress integrand and its multiplier key).
    double stress_bond_length_sq(const std::string& monomer_type, bool is_half_bond_length) const;
    /// Keys of the stress multipliers of a bond: per axis group, (axes, b²·ds/6 or 0 without the bond factor).
    std::vector<std::pair<int, double>> crysfft_stress_keys(double bond_length_sq, bool include_bond_factor) const;
    /**
     * @brief Diagonal stress sums through the CrysFFT fast path.
     *
     * Applies when crysfft_stress_fast_path() holds, with the multipliers of
     * prepare_stress(). Returns false when the fast path does not apply;
     * `stress` is then untouched.
     */
    bool compute_stress_crysfft(const T* q_1, const T* q_2, double bond_length_sq,
                                const double* boltz_bond, std::vector<T>& stress);
    /// Accumulate Σ_k kernel(k) v_i v_j from complex Fourier coefficients (periodic BC).
    void accumulate_stress_periodic(const double* qk_1, const double* qk_2, double bond_length_sq,
                                    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const;
    /// Accumulate Σ_k kernel(k) v_i² from DCT/DST coefficients (non-periodic BC, orthogonal cells; complex
    /// pairs for complex fields).
    void accumulate_stress_nonperiodic(const double* qk_1, const double* qk_2, double bond_length_sq,
                                       const double* boltz_bond, std::vector<T>& stress) const;

public:
    /// Default constructor.
    CpuSolverPseudoBase() : is_periodic_(true), dim_(0), space_group_(nullptr) {}

    /// Virtual destructor. Frees the shared components (FFT, Pseudo).
    virtual ~CpuSolverPseudoBase() = default;   // fft_ and pseudo are owning members

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, propagator input/output uses reduced basis and the solver
     * internally handles expand/reduce around FFT operations.
     *
     * @param sg Space group pointer (nullptr to disable reduced basis)
     */
    void set_space_group(SpaceGroup* sg) override;

    /// Get the contour step size for a ds index.
    double get_ds_value(int ds_index) const;

    /// Compute effective diffusion coefficient (b^2 * ds / 6).
    double get_effective_diffusion_coeff(
        const std::string& monomer_type, int ds_index, bool half_step) const;

    /**
     * @brief Update Fourier-space operators.
     *
     * Recomputes exp(-b²|k|²ds/6) for each monomer type when box
     * dimensions change. Required after box size updates.
     *
     * - Continuous chains: diffusion propagator
     * - Discrete chains: bond function ĝ(k)
     *
     * Shared by all three pseudo-spectral solvers (none overrides it).
     */
    void update_laplacian_operator() override;

    /**
     * @brief Build the CrysFFT stress multipliers of the blocks' monomer types
     *        that are missing, before the stress loop (CrysFFT fast path only).
     *
     * update_laplacian_operator() rebuilds them for a new cell; the stress loop
     * only looks them up.
     */
    void prepare_stress() override;

protected:
    /**
     * @brief Copy a reduced-basis field onto the CrysFFT physical grid, and back.
     *
     * Index-map copies (crysfft_reduced_indices_).
     */
    void fill_crysfft_from_reduced(const double* reduced_in, double* phys_out) const;
    void reduce_crysfft_to_reduced(const double* phys_in, double* reduced_out) const;

    /**
     * @brief Compute stress contribution from a single segment.
     *
     * Computes ∂ln(Q)/∂θ in Fourier space by multiplying forward and backward
     * propagators with weighted basis functions. Cross-term corrections for
     * non-orthogonal boxes are included.
     *
     * Chain model difference:
     * - Continuous: Φ(k) = 1 (bond factor already in propagator)
     * - Discrete: Φ(k) = exp(-b²k²Δs/6) or exp(-b²k²Δs/12) for half-bond
     *
     * @param q_1                Forward propagator
     * @param q_2                Backward propagator
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Whether using half bond (discrete model)
     *
     * @return Stress components in Voigt notation:
     *         [σ₁, σ₂, σ₃, σ₁₂, σ₁₃, σ₂₃] (6 components)
     *         For 2D: [σ₁, σ₂, σ₁₂, 0, 0, 0]
     *         For 1D: only index 0 is meaningful
     */
    std::vector<T> compute_single_segment_stress(
        T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length) override;

private:
    bool pseudo_current_ = false;   ///< pseudo's tables are those of the current cell (see update_pseudo())

    /// Compute the Pseudo tables of the current cell.
    void update_pseudo();
    /// Whether a full-grid path (propagator steps or stress) uses the Pseudo tables: not when the
    /// CrysFFT steps propagate and the stress takes the CrysFFT fast path.
    bool full_grid_path() const;

    mutable std::mutex work_mutex_;
    mutable std::unordered_map<std::thread::id, WorkArrays> work_arrays_;   ///< Per-thread work arrays (see WorkArrays)
};

#endif
