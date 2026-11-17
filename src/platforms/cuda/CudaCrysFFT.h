/**
 * @file CudaCrysFFT.h
 * @brief CUDA Crystallographic FFT using DCT-II/III for Pmmm symmetry.
 *
 * DCT-II/III is implemented via CudaRealTransform (FCT algorithm).
 *
 * **Grid size:**
 *
 * Logical grid: Nx × Ny × Nz (full simulation box, must be even)
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2) (1/8 of logical grid)
 *
 * **Reference:**
 *
 * Qiang & Li, "Accelerated pseudo-spectral method of self-consistent field
 * theory via crystallographic fast Fourier transform", Macromolecules (2020)
 */

#ifndef CUDA_CRYS_FFT_H_
#define CUDA_CRYS_FFT_H_

#include <array>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include "CudaCommon.h"

// Forward declaration
class CudaRealTransform3D;

/// CrysFFT mode selection for CUDA solvers.
enum class CudaCrysFFTMode
{
    None,
    Recursive3m,
    PmmmDct,
    ObliqueZ
};

/**
 * @class CudaCrysFFTBase
 * @brief Abstract base for CUDA crystallographic FFT implementations.
 *
 * An object holds the plans and work buffers of one stream; a solver keeps one
 * per stream (clone()). The k-space tables (Boltzmann factors, stress
 * multipliers) depend only on the grid, the cell and their coefficient, so the
 * solver owns one table per coefficient for all streams: it builds them with
 * build_boltzmann() and build_multiplier() outside its propagator and stress
 * loops and passes them to diffusion() and apply_multiplier().
 */
class CudaCrysFFTBase
{
public:
    virtual ~CudaCrysFFTBase() = default;

    /// An object of the same grid for another stream, with plans and work buffers of
    /// its own. The recursive 3m and ObliqueZ clones share the read-only twiddle
    /// factors of the original; a Pmmm clone is a new object (it has no such data).
    virtual std::unique_ptr<CudaCrysFFTBase> clone() const = 0;

    /// Write the Boltzmann factors exp(-k² coeff) of `cell` into `table`
    /// (allocated when its size differs). The upload runs on `stream` and is
    /// complete on return; no kernel may be reading the table meanwhile.
    virtual void build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                 double coeff, cudaStream_t stream) const = 0;

    /// d_q_out = exp(coeff ∇²) d_q_in on the physical grid, with the table of
    /// build_boltzmann(coeff), on `stream`. d_q_out may alias d_q_in; otherwise
    /// d_q_in is kept. Either may lie at any offset into a larger array: the engines
    /// run cuFFT on their own buffers or stage an array not aligned for it.
    virtual void diffusion(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                           cudaStream_t stream) = 0;

    /// Write the stress multiplier exp(-k² coeff)·Σ_{d in axes} k_d² (axes: bit
    /// mask 1 = x, 2 = y, 4 = z; coeff 0: no bond factor) into `table`, as
    /// build_boltzmann() does. Pmmm and recursive 3m only.
    virtual void build_multiplier(CudaDeviceBuffer<double>&, const std::array<double, 6>&, int, double,
                                  cudaStream_t) const
    {
        throw_with_line_number("This CrysFFT engine has no stress multipliers.");
    }

    /// d_q_out = multiplier · d_q_in with a table of build_multiplier(), as diffusion().
    virtual void apply_multiplier(double*, double*, const CudaDeviceBuffer<double>&, cudaStream_t)
    {
        throw_with_line_number("This CrysFFT engine has no stress multipliers.");
    }

protected:
    /// Copy `h` into `table` (allocated when its size differs) on `stream`, complete on return.
    static void upload_table(CudaDeviceBuffer<double>& table, const std::vector<double>& h, cudaStream_t stream)
    {
        if (table.size() != h.size())
            table.allocate(h.size());
        CUDA_CHECK(cudaMemcpyAsync(table, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));   // h may go out of scope; other streams read the table next
    }
};

/**
 * @class CudaCrysFFT
 * @brief CUDA crystallographic FFT using DCT-II/III.
 *
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2). Its tables hold M_physical values:
 * the Boltzmann factors with the round-trip normalization 1/M_logical, the stress
 * multipliers without it.
 */
class CudaCrysFFT : public CudaCrysFFTBase
{
private:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // CudaRealTransform objects for DCT-II (forward) and DCT-III (backward)
    std::unique_ptr<CudaRealTransform3D> dct_forward_;
    std::unique_ptr<CudaRealTransform3D> dct_backward_;

    // Normalization factor
    double norm_factor_;
    cudaStream_t stream_{0};      ///< Stream the transforms run on (the last call's)

    /// Run the DCTs and kernels on `stream` (diffusion() and apply_multiplier() call it).
    void use_stream(cudaStream_t stream);

public:
    /**
     * @brief Construct CudaCrysFFT for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     */
    explicit CudaCrysFFT(std::array<int, 3> nx_logical);

    ~CudaCrysFFT();

    std::unique_ptr<CudaCrysFFTBase> clone() const override;
    void build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                         double coeff, cudaStream_t stream) const override;

    /// Diffusion with DCT-II -> Boltzmann factors -> DCT-III.
    void diffusion(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                   cudaStream_t stream) override;

    void build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff, cudaStream_t stream) const override;

    /**
     * @brief Apply a k-space multiplier: DCT-II -> multiply -> DCT-III.
     *
     * Round-trip normalization (1/M_logical) is applied here, so the table holds
     * raw values, mirroring the CPU CrysFFTPmmm::apply_multiplier convention.
     */
    void apply_multiplier(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                          cudaStream_t stream) override;

    // Getters
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif
