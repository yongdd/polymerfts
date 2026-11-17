/**
 * @file CudaCrysFFTRecursive3m.h
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 *
 * This implements the generalized 3m symmetry using only cuFFT and
 * precomputed k-matrix factors. It mirrors the CPU FftwCrysFFTRecursive3m
 * algorithm but runs fully on GPU.
 */

#ifndef CUDA_CRYS_FFT_RECURSIVE_3M_H_
#define CUDA_CRYS_FFT_RECURSIVE_3M_H_

#include <array>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"

/**
 * @class CudaCrysFFTRecursive3m
 * @brief Recursive 3m CrysFFT; physical grid (Nx/2) × (Ny/2) × (Nz/2).
 *
 * A table (Boltzmann factors or a stress multiplier, round-trip normalization
 * included) holds the eight real and the eight imaginary k-space factors of the
 * fold, 16 arrays of M_physical values one after another.
 */
class CudaCrysFFTRecursive3m : public CudaCrysFFTBase
{
public:
    CudaCrysFFTRecursive3m(std::array<int, 3> nx_logical, std::array<double, 9> translational_part);

    ~CudaCrysFFTRecursive3m() override;

    std::unique_ptr<CudaCrysFFTBase> clone() const override;
    void build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                         double coeff, cudaStream_t stream) const override;
    void diffusion(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                   cudaStream_t stream) override;
    void build_multiplier(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff, cudaStream_t stream) const override;
    /// The same D2Z -> k-multiply -> Z2D pipeline as diffusion() (the input is kept when out of place).
    void apply_multiplier(double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                          cudaStream_t stream) override;

private:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    std::array<double, 9> translational_part_;

    /// Pairing permutation of the fold: twiddle r_p multiplies the
    /// octant-transformed Boltzmann array S_{fold_perm_[p]} (identity when
    /// all generator translations are an even number of grid cells).
    /// See Recursive3mFoldParity.h for the derivation.
    std::array<int, 8> fold_perm_{};

    /// Twiddle factors of the fold, fixed by the grid and the translations (shared with the clones)
    struct Twiddles
    {
        std::array<std::vector<double>, 8> re;
        std::array<std::vector<double>, 8> im;
    };
    std::shared_ptr<const Twiddles> twiddles_;

    cufftHandle plan_r2c_{};
    cufftHandle plan_c2r_{};
    bool plans_initialized_{false};

    CudaDeviceBuffer<cufftDoubleComplex> d_step1_;
    CudaDeviceBuffer<cufftDoubleComplex> d_step2_;

    cudaStream_t stream_{0};   ///< Stream the plans run on (the last call's)

    /// Clone of `other` sharing its twiddle factors (see clone()).
    explicit CudaCrysFFTRecursive3m(const CudaCrysFFTRecursive3m& other);

    void init_plans();
    void free_plans();
    /// Run the cuFFT plans and kernels on `stream` (diffusion() calls it every time).
    void use_stream(cudaStream_t stream);
    std::shared_ptr<const Twiddles> generate_twiddle_factors() const;
    /// Fold the logical k-space array tempmat into the 16 factor arrays of a table.
    void build_table_from_tempmat(CudaDeviceBuffer<double>& table, const std::vector<double>& tempmat,
                                  cudaStream_t stream) const;
    /// Free everything acquired so far (idempotent; also used when the constructor throws).
    void release();
};

#endif
