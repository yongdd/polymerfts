/**
 * @file CrysFFTRecursive3mBase.h
 * @brief Recursive crystallographic FFT (2x2y2z) over the real FFT of a library.
 *
 * Implements the recursive 2x2y2z algorithm described in
 * "Accelerated pseudo-spectral method of self-consistent field theory via
 * crystallographic fast Fourier transform", Macromolecules (2020).
 *
 * @see FftwCrysFFTRecursive3m for the real FFT of FFTW
 * @see MklCrysFFTRecursive3m for the real FFT of MKL
 */

#ifndef CRYS_FFT_RECURSIVE_3M_BASE_H_
#define CRYS_FFT_RECURSIVE_3M_BASE_H_

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "AlignedScratch.h"
#include "CpuCrysFFT.h"
#include "CrysFFTTables.h"

/**
 * @class CrysFFTRecursive3mBase
 * @brief Recursive crystallographic FFT on the physical grid.
 *
 * Builds the folded k-space tables (build_boltzmann(), build_multiplier()) with the
 * fold of Recursive3mTables (CrysFFTTables.h, shared with the CUDA engine) and
 * applies them (diffusion(), apply_multiplier()). A table holds 16 arrays of
 * M_physical values, the real parts of the eight folded octants and then their
 * imaginary parts, table_stride_ values apart.
 *
 * The derived classes of the FFTW and MKL libraries hold the plans of the real FFT
 * of the physical grid, which they run in forward() and backward(): the spectrum
 * keeps the strides of the grid (Nz2 complex values per row, of which the first
 * Nz2/2 + 1 are set) and starts on a 64-byte boundary (AlignedScratch).
 */
class CrysFFTRecursive3mBase : public CpuCrysFFT
{
protected:
    /// The grid checks, the twiddle factors and the fold of the tables
    Recursive3mTables tables_;
    std::array<int, 3> nx_physical_;   ///< (Nx/2, Ny/2, Nz/2)
    int M_physical_;

    /// Values from one array of a table to the next: M_physical rounded up to whole
    /// 64-byte lines, plus one line. The multiply loop reads all 16 arrays at the same
    /// index; with M_physical a power of two and no extra line they would all fall
    /// into one cache set.
    size_t table_stride_ = 0;

    /// spectrum = unnormalized real FFT of q (q is kept), in the strides of the grid
    virtual void forward(const double* q, std::complex<double>* spectrum) const = 0;
    /// q = unnormalized inverse real FFT of spectrum (spectrum may be overwritten)
    virtual void backward(std::complex<double>* spectrum, double* q) const = 0;

    /// q_out = F^-1[ table F[q_in] ] with a table of build_boltzmann() or
    /// build_multiplier() (16 arrays, table_stride_ values apart); q_in is kept.
    void apply_with_table(const double* table, const double* q_in, double* q_out) const
    {
        // The two spectra of this thread, from 64-byte boundaries (AlignedScratch)
        thread_local AlignedScratch spectrum_1, spectrum_2;
        std::complex<double>* step1 = reinterpret_cast<std::complex<double>*>(spectrum_1.get(2 * static_cast<size_t>(M_physical_)));
        std::complex<double>* step2 = reinterpret_cast<std::complex<double>*>(spectrum_2.get(2 * static_cast<size_t>(M_physical_)));

        forward(q_in, step1);

        const int Nx2 = nx_physical_[0];
        const int Ny2 = nx_physical_[1];
        const int Nz2 = nx_physical_[2];
        // Pad the packed z count to whole 64-byte lines for vectorization, but clamp
        // to the row stride Nz2 so the loop never overruns the Nx2*Ny2*Nz2-sized
        // buffers. The clamp still covers the packed range [0, Nz2/2] for any Nz2 >= 2.
        constexpr size_t line = 64 / sizeof(double);
        const size_t Nz_qua = std::min((static_cast<size_t>(Nz2 / 2 + 1) + line - 1) / line * line,
                                       static_cast<size_t>(Nz2));

        // The real parts of the eight folded octants, then their imaginary parts
        std::array<const double*, 8> k_re, k_im;
        for (int i = 0; i < 8; ++i)
        {
            k_re[i] = table + i * table_stride_;
            k_im[i] = table + (8 + i) * table_stride_;
        }

        for (int ix = 0; ix < Nx2; ++ix)
        {
            for (int iy = 0; iy < Ny2; ++iy)
            {
                const size_t base_shift = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
                const std::complex<double>* src000 = step1 + base_shift;
                const std::complex<double>* src110 = step1 + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
                const std::complex<double>* src010 = step1 + (static_cast<size_t>(ix) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
                const std::complex<double>* src100 = step1 + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + iy) * Nz2;
                std::complex<double>* dst = step2 + base_shift;

                for (size_t iz = 0; iz < Nz_qua; ++iz)
                {
                    size_t idx = base_shift + iz;
                    double src000_re = src000[iz].real();
                    double src000_im = src000[iz].imag();
                    double src110_re = src110[iz].real();
                    double src110_im = src110[iz].imag();
                    double src010_re = src010[iz].real();
                    double src010_im = src010[iz].imag();
                    double src100_re = src100[iz].real();
                    double src100_im = src100[iz].imag();

                    double dst_re
                        = src000_re * k_re[0][idx] - src000_im * k_im[7][idx]
                        + src110_re * k_re[6][idx] - src110_im * k_im[1][idx]
                        + src010_re * k_re[2][idx] - src010_im * k_im[5][idx]
                        + src100_re * k_re[4][idx] - src100_im * k_im[3][idx];

                    double dst_im
                        = src000_re * k_im[0][idx] + src000_im * k_re[7][idx]
                        + src110_re * k_im[6][idx] + src110_im * k_re[1][idx]
                        + src010_re * k_im[2][idx] + src010_im * k_re[5][idx]
                        + src100_re * k_im[4][idx] + src100_im * k_re[3][idx];

                    dst[iz] = std::complex<double>(dst_re, dst_im);
                }
            }
        }

        backward(step2, q_out);
    }

    /// The grids and the twiddle factors; the derived class makes the plans of its library.
    CrysFFTRecursive3mBase(std::array<int, 3> nx_logical, std::array<double, 9> translational_part)
        : tables_(nx_logical, translational_part),
          nx_physical_(tables_.nx_physical()),
          M_physical_(tables_.M_physical()),
          table_stride_((static_cast<size_t>(M_physical_) + 7) / 8 * 8 + 8)
    {
    }

public:
    int get_M_physical() const override { return M_physical_; }

    /// Folded table of exp(-k² coeff) for diffusion().
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double coeff) const override
    {
        tables_.boltzmann(table, table_stride_, cell, coeff);
    }

    void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        check_table(table, 16 * table_stride_);
        apply_with_table(table.data(), q_in, q_out);
    }

    bool has_multipliers() const override { return true; }

    /// Folded table of Σ_{d in axes} k_d², times exp(-k² coeff) when coeff is not
    /// 0 (the bond factor of discrete chains), for apply_multiplier().
    void build_multiplier(std::vector<double>& table, const std::array<double, 6>& cell,
                          int axes, double coeff) const override
    {
        tables_.multiplier(table, table_stride_, cell, axes, coeff);
    }

    void apply_multiplier(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        check_table(table, 16 * table_stride_);
        apply_with_table(table.data(), q_in, q_out);
    }
};

#endif  // CRYS_FFT_RECURSIVE_3M_BASE_H_
