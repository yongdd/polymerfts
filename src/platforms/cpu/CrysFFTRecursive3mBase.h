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
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "AlignedScratch.h"
#include "CpuCrysFFT.h"
#include "Exception.h"
#include "Recursive3mFoldParity.h"
#include "Recursive3mHelpers.h"

/**
 * @class CrysFFTRecursive3mBase
 * @brief Recursive crystallographic FFT on the physical grid.
 *
 * Holds the twiddle factors, builds the folded k-space tables (build_boltzmann(),
 * build_multiplier()) and applies them (diffusion(), apply_multiplier()). A table
 * holds 16 arrays of M_physical values, the real parts of the eight folded octants
 * and then their imaginary parts, table_stride_ values apart.
 *
 * The derived classes of the FFTW and MKL libraries hold the plans of the real FFT
 * of the physical grid, which they run in forward() and backward(): the spectrum
 * keeps the strides of the grid (Nz2 complex values per row, of which the first
 * Nz2/2 + 1 are set) and starts on a 64-byte boundary (AlignedScratch).
 */
class CrysFFTRecursive3mBase : public CpuCrysFFT
{
protected:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_physical_;
    std::array<double, 9> translational_part_;

    /// Pairing permutation of the fold: twiddle r_p multiplies the
    /// octant-transformed Boltzmann array S_{fold_perm_[p]} (identity when
    /// all generator translations are an even number of grid cells).
    /// See Recursive3mFoldParity.h for the derivation.
    std::array<int, 8> fold_perm_{};

    std::array<std::unique_ptr<double[]>, 8> r_re_;
    std::array<std::unique_ptr<double[]>, 8> r_im_;

    /// Values from one array of a table to the next: M_physical rounded up to whole
    /// 64-byte lines, plus one line. The multiply loop reads all 16 arrays at the same
    /// index; with M_physical a power of two and no extra line they would all fall
    /// into one cache set.
    size_t table_stride_ = 0;

    /// spectrum = unnormalized real FFT of q (q is kept), in the strides of the grid
    virtual void forward(const double* q, std::complex<double>* spectrum) const = 0;
    /// q = unnormalized inverse real FFT of spectrum (spectrum may be overwritten)
    virtual void backward(std::complex<double>* spectrum, double* q) const = 0;

    void generate_twiddle_factors()
    {
        using namespace crys_recursive_3m;

        for (int i = 0; i < 8; ++i)
        {
            r_re_[i] = std::make_unique<double[]>(M_physical_);
            r_im_[i] = std::make_unique<double[]>(M_physical_);
        }

        // First twiddle factor is identity
        std::fill_n(r_re_[0].get(), M_physical_, 1.0);
        std::fill_n(r_im_[0].get(), M_physical_, 0.0);

        // Generate remaining twiddle factors
        auto ops = generate_m3_operations(translational_part_, nx_logical_);
        std::vector<std::tuple<double*, double*, std::array<double, 3>>> tasks;
        for (size_t i = 0; i < ops.size(); ++i)
        {
            tasks.emplace_back(r_re_[i + 1].get(), r_im_[i + 1].get(), ops[i]);
        }
        twiddle_factor(tasks, nx_physical_);
    }

    /**
     * @brief Write the folded 2x2y2z table of a per-mode value into `table`.
     *
     * Shared body of build_boltzmann() and build_multiplier(). Tabulates the
     * squared wavevector components kx^2, ky^2, kz^2 of the logical grid of
     * `cell`, fills the full grid with value_at(kx2, ky2, kz2) / M_logical,
     * splits it into 8 octants, applies the 2x2y2z butterflies and multiplies
     * with the (parity-permuted) twiddle factors.
     *
     * @tparam ValueFn Callable with signature double(double kx2, double ky2, double kz2).
     * @param value_at Per-mode value before the 1/M_logical normalization.
     */
    template <typename ValueFn>
    void build_table(std::vector<double>& table, const std::array<double, 6>& cell, ValueFn&& value_at) const
    {
        using namespace crys_recursive_3m;

        table.assign(16 * table_stride_, 0.0);

        // value_at(kx^2, ky^2, kz^2) / M_logical on the logical grid
        std::vector<double> tempmat = logical_k_table(nx_logical_, cell, value_at);

        // Split into 8 octants
        std::array<std::unique_ptr<double[]>, 8> k_split;
        for (int i = 0; i < 8; ++i)
        {
            k_split[i] = std::make_unique<double[]>(M_physical_);
        }

        std::array<double*, 8> k_split_ptr;
        for (int i = 0; i < 8; ++i) k_split_ptr[i] = k_split[i].get();

        mat_split(tempmat.data(), k_split_ptr, nx_logical_);

        // Apply 2x2y2z decomposition
        std::array<int, 3> halfsize = nx_physical_;
        constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
        add_sub_by_seq(k_split_ptr, seq0, halfsize);
        add_sub_by_seq(k_split_ptr, seq1, halfsize);
        add_sub_by_seq(k_split_ptr, seq2, halfsize);

        // Multiply with twiddle factors. Twiddle r_p pairs with
        // S_{fold_perm_[p]} (translation-parity permutation; identity when all
        // generator translations are even in grid units).
        constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
        std::array<double*, 8> table_re_ptr, table_im_ptr;
        std::array<const double*, 8> r_re_ptr, r_im_ptr, k_split_perm_ptr;
        for (int i = 0; i < 8; ++i)
        {
            table_re_ptr[i] = table.data() + i * table_stride_;
            table_im_ptr[i] = table.data() + (8 + i) * table_stride_;
            r_re_ptr[i] = r_re_[i].get();
            r_im_ptr[i] = r_im_[i].get();
            k_split_perm_ptr[i] = k_split[fold_perm_[i]].get();
        }

        mul_add_sub_by_seq(
            table_re_ptr, table_im_ptr,
            r_re_ptr, r_im_ptr,
            k_split_perm_ptr,
            seq3, halfsize);
    }

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
    CrysFFTRecursive3mBase(
        std::array<int, 3> nx_logical,
        std::array<double, 9> translational_part)
        : nx_logical_(nx_logical),
          translational_part_(translational_part)
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] % 2 != 0)
            {
                throw_with_line_number("CrysFFTRecursive3m requires even grid dimensions.");
            }
            if (nx_logical_[d] <= 0)
            {
                throw_with_line_number("CrysFFTRecursive3m requires positive grid dimensions.");
            }
        }

        // The k-space multiply loop pads the packed z iteration count to whole
        // 64-byte lines, clamped to Nz2, so any Nz2 = nz/2 >= 2 is safe (the clamp
        // keeps the padded loop inside each row while still covering the packed
        // range [0, Nz2/2]).
        if (nx_logical_[2] / 2 < 2)
        {
            throw_with_line_number("CrysFFTRecursive3m requires Nz/2 >= 2.");
        }

        // The fold pairs twiddle r_p with the octant-transformed Boltzmann
        // array S_{psi(p)}, where psi depends on the parities of the m3
        // generator translations in grid units (see Recursive3mFoldParity.h).
        // When psi is not a bijection the even-index physical basis does not
        // exist on this grid and the fold cannot represent the group;
        // CrysFFTSelector rejects such grids, but guard direct construction.
        if (!compute_m3_fold_permutation(translational_part_, nx_logical_, fold_perm_))
        {
            throw_with_line_number("CrysFFTRecursive3m: m3 translations are "
                "incompatible with this grid (translation-parity permutation "
                "is not a bijection; no even-index physical basis exists).");
        }

        nx_physical_ = {
            nx_logical_[0] / 2,
            nx_logical_[1] / 2,
            nx_logical_[2] / 2
        };

        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        table_stride_ = (static_cast<size_t>(M_physical_) + 7) / 8 * 8 + 8;

        generate_twiddle_factors();
    }

public:
    int get_M_physical() const override { return M_physical_; }

    /// Folded table of exp(-k² coeff) for diffusion().
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double coeff) const override
    {
        build_table(table, cell, [coeff](double kx2, double ky2, double kz2) {
            return std::exp(-(kx2 + ky2 + kz2) * coeff);
        });
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
        if (axes < 1 || axes > 7)
            throw_with_line_number("CrysFFTRecursive3m::build_multiplier: axes must be a mask of 1 (x), 2 (y), 4 (z).");
        const bool bond = (coeff != 0.0);
        build_table(table, cell, [axes, bond, coeff](double kx2, double ky2, double kz2) {
            const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
            return bond ? std::exp(-(kx2 + ky2 + kz2) * coeff) * k_axes : k_axes;
        });
    }

    void apply_multiplier(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        check_table(table, 16 * table_stride_);
        apply_with_table(table.data(), q_in, q_out);
    }
};

#endif  // CRYS_FFT_RECURSIVE_3M_BASE_H_
