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
 * The FFT object is stored as `std::unique_ptr<FFT<T>> fft_` which points to FftwFFT<T>
 * or MklFFT<T> (periodic, by backend) or MakhoulFFT<T> (walls), the transform of the grid
 * of any dimension.
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
#include <utility>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFTFactory.h"
#include "SpaceGroup.h"
#include "CpuCrysFFT.h"
#include "PseudoSolverHelpers.h"

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
 * - FFT object creation/destruction
 * - Forward/backward transform methods
 * - Laplacian operator updates
 * - Stress computation framework
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - update_dw(): the factors of ds of the Boltzmann factors (CpuSolver::compute_exp_dw())
 * - advance_propagator(): Different propagation algorithms
 * - advance_propagator_half_bond_step(): Discrete-specific
 * - get_stress_boltz_bond() (optional): bond factor of the stress integrand; the base returns nullptr
 */
template <typename T>
class CpuSolverPseudoBase : public CpuSolver<T>
{
protected:
    /// Work arrays of one calling thread (PerThread)
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
    WorkArrays& work_arrays() const { return work_arrays_.get(); }

    using CpuSolver<T>::cb;
    using CpuSolver<T>::molecules;
    using CpuSolver<T>::space_group_;
    using CpuSolver<T>::n_basis_;

    /**
     * @brief Spectral transform object for FFT/DCT/DST (handles all BCs).
     *
     * Points to FftwFFT<T> or MklFFT<T> (periodic) based on the selected
     * backend, or to MakhoulFFT<T> over the real FFT of that backend (walls)
     * (createFFT()). Uses an FFT<T> base class pointer to enable polymorphic
     * calls without backend-specific dispatch.
     */
    std::unique_ptr<FFT<T>> fft_;

    FFTBackend fft_backend_;  ///< Selected FFT backend (FFTW or MKL)

    /// Fourier-space tables of the full grid (Boltzmann bond factors, stress basis), computed for
    /// the cell when a full-grid path uses them (update_pseudo())
    std::unique_ptr<Pseudo<T>> pseudo;

    bool is_periodic_ = true;    ///< True if all BCs are periodic
    int dim_ = 0;                ///< Number of dimensions (1, 2, or 3)

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
     * @brief The shared context of a solver (CpuSolver: the box, the molecules, the space
     *        group and exp_dw); the derived constructor then calls init_shared().
     *
     * @param with_half Also exp_dw_half (RQM4)
     */
    CpuSolverPseudoBase(ComputationBox<T>* cb, Molecules* molecules, SpaceGroup* space_group, bool with_half)
        : CpuSolver<T>(cb, molecules, space_group, with_half) {}

    /**
     * @brief Initialize the shared components, from the derived constructor.
     *
     * Sets up:
     * - Boundary condition checking (is_periodic_)
     * - FFT object creation (fft_) using specified backend
     * - The contour steps (ds_values_) and the Fourier-space tables (pseudo)
     * - With a space group, for real fields, the CrysFFT fast path (init_crysfft())
     * - The tables of the cell (update_laplacian_operator())
     *
     * @param backend FFT backend to use (FFTW or MKL)
     */
    void init_shared(FFTBackend backend);

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

    /// True when the stress integrand carries the Boltzmann bond factor (discrete chains).
    virtual bool stress_has_bond_factor() const { return false; }

    // ---- space group / CrysFFT setup (init_shared) ---------------------------
    /// Select and build the CrysFFT fast path for the space group (3D periodic, real
    /// fields), with a Boltzmann table of every block, which update_laplacian_operator()
    /// builds for the cell.
    void init_crysfft();
    /// Rebuild every CrysFFT table in place for the current cell.
    void update_crysfft_tables();

    // ---- compute_single_segment_stress() helpers ----------------------------
    /// True when the segment stress takes the CrysFFT fast path (an engine with stress
    /// multipliers: Pmmm or recursive 3m; orthogonal 3D periodic cell, no angle optimization).
    bool crysfft_stress_fast_path() const;
    /**
     * @brief Diagonal stress sums through the CrysFFT fast path.
     *
     * Applies when crysfft_stress_fast_path() holds, with the multipliers of
     * prepare_stress(). Returns false when the fast path does not apply;
     * `stress` is then untouched.
     */
    bool compute_stress_crysfft(const T* q_1, const T* q_2, double bond_length_sq, std::vector<T>& stress);
    /**
     * @brief Accumulate the v⊗v dyad sums V_ij = Σ_k kernel(k) v_i v_j of the coefficients
     *        of two propagators.
     *
     * Periodic boxes have complex Fourier coefficients; boxes of walls have the
     * real DCT/DST coefficients (complex pairs for complex fields) and an
     * orthogonal cell, so only their diagonal sums. The cross terms of a periodic
     * box are summed only when is_orthogonal is false.
     */
    void accumulate_stress(const double* qk_1, const double* qk_2, double bond_length_sq,
                           const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const;

public:
    /// Virtual destructor. Frees the shared components (FFT, Pseudo).
    virtual ~CpuSolverPseudoBase() = default;   // fft_ and pseudo are owning members

    /// The diffusion coefficient b² ds / 6 of a step, ds / 2 for a half step (effective_diffusion_coeff()).
    double get_effective_diffusion_coeff(const std::string& monomer_type, int ds_index, bool half_step) const
    {
        return effective_diffusion_coeff(*molecules, monomer_type, ds_index, half_step);
    }

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
    bool stress_basis_ = false;     ///< pseudo computes the stress basis (from the first prepare_stress())

    /// Compute the Pseudo tables of the current cell.
    void update_pseudo();
    /// Whether a full-grid path (propagator steps or stress) uses the Pseudo tables: not when the
    /// CrysFFT steps propagate and the stress takes the CrysFFT fast path.
    bool full_grid_path() const;

    PerThread<WorkArrays> work_arrays_;   ///< Work arrays of each thread
};

#endif
