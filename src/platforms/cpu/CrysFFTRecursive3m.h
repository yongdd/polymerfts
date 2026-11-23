/**
 * @file CrysFFTRecursive3m.h
 * @brief Recursive crystallographic FFT (2x2y2z) on the CPU (FFTW or MKL).
 *
 * Implements the recursive 2x2y2z algorithm described in
 * "Accelerated pseudo-spectral method of self-consistent field theory via
 * crystallographic fast Fourier transform", Macromolecules (2020), over the
 * periodic real FFT of the selected library (FftwRealFFT or MklRealFFT), as
 * CudaCrysFFTRecursive3m runs it over cuFFT.
 */

#ifndef CRYS_FFT_RECURSIVE_3M_H_
#define CRYS_FFT_RECURSIVE_3M_H_

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "AlignedScratch.h"
#include "CpuCrysFFT.h"
#include "CrysFFTTables.h"
#include "FFTFactory.h"
#include "MakhoulTransform.h"

/**
 * @class CrysFFTRecursive3m
 * @brief Recursive crystallographic FFT on the physical grid (Nx/2, Ny/2, Nz/2).
 *
 * Builds the folded k-space tables (build_boltzmann(), build_multiplier()) with the
 * fold of Recursive3mTables (CrysFFTTables.h, shared with the CUDA engine) and
 * applies them in diffusion(). A table holds 16 arrays of M_physical values, the
 * real parts of the eight folded octants and then their imaginary parts,
 * table_stride_ values apart.
 */
class CrysFFTRecursive3m : public CpuCrysFFT
{
private:
    /// The grid checks, the twiddle factors and the fold of the tables
    Recursive3mTables tables_;
    std::array<int, 3> nx_physical_;   ///< (Nx/2, Ny/2, Nz/2)
    int M_physical_;
    int M_spectral_;                   ///< Nx/2 * Ny/2 * (Nz/4 + 1): half spectrum of the physical grid

    /// Values from one array of a table to the next: M_physical rounded up to whole
    /// 64-byte lines, plus one line. The multiply loop reads all 16 arrays at the same
    /// index; with M_physical a power of two and no extra line they would all fall
    /// into one cache set.
    size_t table_stride_;

    std::unique_ptr<MakhoulTransform::RealFFT> fft_;   ///< Real FFT of the physical grid

public:
    /**
     * @param nx_logical   Logical grid dimensions (even, Nz/2 >= 2)
     * @param translations Generator translations of the m3 mirrors (CrysFFTPlan::m3_translations)
     * @param backend      Library of the real FFT (FFTW or MKL)
     */
    CrysFFTRecursive3m(std::array<int, 3> nx_logical, std::array<double, 9> translations, FFTBackend backend)
        : tables_(nx_logical, translations),
          nx_physical_(tables_.nx_physical()),
          M_physical_(tables_.M_physical()),
          M_spectral_(nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1)),
          table_stride_((static_cast<size_t>(M_physical_) + 7) / 8 * 8 + 8),
          fft_(createRealFFT(std::vector<int>(nx_physical_.begin(), nx_physical_.end()), backend))
    {
    }

    int get_M_physical() const override { return M_physical_; }

    /// Folded table of exp(-k² coeff) for diffusion().
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double coeff) const override
    {
        tables_.boltzmann(table, table_stride_, cell, coeff);
    }

    /// q_out = F^-1[ table F[q_in] ] with a table of build_boltzmann() or build_multiplier()
    /// (16 arrays, table_stride_ values apart); q_in is kept (q_out may be q_in).
    void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        check_table(table, 16 * table_stride_);

        // Work arrays of this thread, from 64-byte boundaries, as the real FFT needs; it
        // may overwrite its input, so the field is transformed from a copy
        thread_local AlignedScratch real_scratch, spectrum_1, spectrum_2;
        double* real = real_scratch.get(M_physical_);
        std::complex<double>* step1 = reinterpret_cast<std::complex<double>*>(
            spectrum_1.get(2 * static_cast<size_t>(M_spectral_)));
        std::complex<double>* step2 = reinterpret_cast<std::complex<double>*>(
            spectrum_2.get(2 * static_cast<size_t>(M_spectral_)));

        std::copy(q_in, q_in + M_physical_, real);
        fft_->forward(real, step1);

        // The half spectrum has rows of Nz2/2 + 1 modes, the table rows of Nz2 values
        const int Nx2 = nx_physical_[0];
        const int Ny2 = nx_physical_[1];
        const int Nz2 = nx_physical_[2];
        const int Nz2c = Nz2 / 2 + 1;

        // The real parts of the eight folded octants, then their imaginary parts
        std::array<const double*, 8> k_re, k_im;
        for (int i = 0; i < 8; ++i)
        {
            k_re[i] = table.data() + i * table_stride_;
            k_im[i] = table.data() + (8 + i) * table_stride_;
        }

        for (int ix = 0; ix < Nx2; ++ix)
        {
            const int ix_m = (Nx2 - ix) % Nx2;
            for (int iy = 0; iy < Ny2; ++iy)
            {
                const int iy_m = (Ny2 - iy) % Ny2;
                const size_t base_k = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
                const std::complex<double>* src000 = step1 + static_cast<size_t>(ix * Ny2 + iy) * Nz2c;
                const std::complex<double>* src110 = step1 + static_cast<size_t>(ix_m * Ny2 + iy_m) * Nz2c;
                const std::complex<double>* src010 = step1 + static_cast<size_t>(ix * Ny2 + iy_m) * Nz2c;
                const std::complex<double>* src100 = step1 + static_cast<size_t>(ix_m * Ny2 + iy) * Nz2c;
                std::complex<double>* dst = step2 + static_cast<size_t>(ix * Ny2 + iy) * Nz2c;

                for (int iz = 0; iz < Nz2c; ++iz)
                {
                    const size_t idx = base_k + iz;
                    const double src000_re = src000[iz].real();
                    const double src000_im = src000[iz].imag();
                    const double src110_re = src110[iz].real();
                    const double src110_im = src110[iz].imag();
                    const double src010_re = src010[iz].real();
                    const double src010_im = src010[iz].imag();
                    const double src100_re = src100[iz].real();
                    const double src100_im = src100[iz].imag();

                    const double dst_re
                        = src000_re * k_re[0][idx] - src000_im * k_im[7][idx]
                        + src110_re * k_re[6][idx] - src110_im * k_im[1][idx]
                        + src010_re * k_re[2][idx] - src010_im * k_im[5][idx]
                        + src100_re * k_re[4][idx] - src100_im * k_im[3][idx];

                    const double dst_im
                        = src000_re * k_im[0][idx] + src000_im * k_re[7][idx]
                        + src110_re * k_im[6][idx] + src110_im * k_re[1][idx]
                        + src010_re * k_im[2][idx] + src010_im * k_re[5][idx]
                        + src100_re * k_im[4][idx] + src100_im * k_re[3][idx];

                    dst[iz] = std::complex<double>(dst_re, dst_im);
                }
            }
        }

        // The inverse FFT overwrites step2, which the next call recomputes
        fft_->backward(step2, real);
        std::copy(real, real + M_physical_, q_out);
    }

    bool has_multipliers() const override { return true; }

    /// Folded table of Σ_{d in axes} k_d², times exp(-k² coeff) when coeff is not
    /// 0 (the bond factor of discrete chains), for diffusion().
    void build_multiplier(std::vector<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff) const override
    {
        tables_.multiplier(table, table_stride_, cell, axes, coeff);
    }
};

#endif  // CRYS_FFT_RECURSIVE_3M_H_
