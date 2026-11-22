/**
 * @file CrysFFTTables.h
 * @brief k-space tables of the Pmmm and recursive 3m crystallographic FFTs, built on the
 *        host for the CPU (CrysFFTPmmm, CrysFFTRecursive3mBase) and CUDA (CudaCrysFFT,
 *        CudaCrysFFTRecursive3m) engines.
 *
 * A table holds the value of each mode of the physical grid: the Boltzmann factors
 * exp(-k² coeff) of a step, or a stress multiplier Σ_{d in axes} k_d² (times the
 * bond factor of discrete chains). The Pmmm physical grid holds N_d/2 points per
 * axis, and its cosine series has the wavenumbers k_d = 2π m_d / L_d with
 * m_d < N_d/2; a Pmmm table holds one value per point of the physical grid, z
 * fastest. The recursive 3m engines fold the values of the logical grid into the
 * 16 arrays of a table (Recursive3mTables).
 */

#ifndef CRYS_FFT_TABLES_H_
#define CRYS_FFT_TABLES_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <tuple>
#include <utility>
#include <vector>

#include "Exception.h"
#include "Recursive3mFoldParity.h"
#include "Recursive3mHelpers.h"

/// The Boltzmann factor exp(-k² coeff) of a mode with squared wavenumber components kx², ky², kz².
inline double crysfft_boltzmann_value(double coeff, double kx2, double ky2, double kz2)
{
    return std::exp(-(kx2 + ky2 + kz2) * coeff);
}

/// Throw unless `axes` is a mask of the axes 1 (x), 2 (y) and 4 (z) of a stress multiplier.
inline void check_crysfft_axes(int axes)
{
    if (axes < 1 || axes > 7)
        throw_with_line_number("A CrysFFT stress multiplier takes a mask of the axes 1 (x), 2 (y) and 4 (z).");
}

/**
 * @brief The stress multiplier Σ_{d in axes} k_d² of a mode, times exp(-k² coeff) when
 *        coeff is not 0 (the bond factor of discrete chains).
 *
 * @param axes  Bit mask of the axes: 1 (x), 2 (y), 4 (z)
 * @param coeff b² Δs / 6 of the bond factor, or 0 without it
 */
inline double crysfft_multiplier_value(int axes, double coeff, double kx2, double ky2, double kz2)
{
    const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
    return (coeff != 0.0) ? std::exp(-(kx2 + ky2 + kz2) * coeff) * k_axes : k_axes;
}

/**
 * @brief The Boltzmann factors exp(-k² ds) of the Pmmm physical grid.
 *
 * @param table       Resized to the physical grid
 * @param nx_physical Physical grid (N/2 per axis)
 * @param cell        Cell [Lx, Ly, Lz, alpha, beta, gamma] (orthogonal)
 * @param ds          Step coefficient (b² Δs / 6 of a full step)
 */
inline void build_pmmm_boltzmann(std::vector<double>& table, const std::array<int, 3>& nx_physical,
                                 const std::array<double, 6>& cell, double ds)
{
    table.assign(static_cast<size_t>(nx_physical[0]) * nx_physical[1] * nx_physical[2], 0.0);

    const double Lx = cell[0];
    const double Ly = cell[1];
    const double Lz = cell[2];
    const double PI = std::numbers::pi;

    int idx = 0;
    for (int ix = 0; ix < nx_physical[0]; ++ix)
    {
        double kx = ix * 2.0 * PI / Lx;
        double kx2 = kx * kx;

        for (int iy = 0; iy < nx_physical[1]; ++iy)
        {
            double ky = iy * 2.0 * PI / Ly;
            double ky2 = ky * ky;

            for (int iz = 0; iz < nx_physical[2]; ++iz)
            {
                double kz = iz * 2.0 * PI / Lz;
                double kz2 = kz * kz;

                table[idx++] = crysfft_boltzmann_value(ds, kx2, ky2, kz2);
            }
        }
    }
}

/**
 * @brief The stress multipliers crysfft_multiplier_value() of the Pmmm physical grid.
 *
 * @param table       Resized to the physical grid
 * @param nx_physical Physical grid (N/2 per axis)
 * @param cell        Cell [Lx, Ly, Lz, alpha, beta, gamma] (orthogonal)
 * @param axes        Bit mask of the axes: 1 (x), 2 (y), 4 (z)
 * @param coeff       b² Δs / 6 of the bond factor, or 0 without it
 */
inline void build_pmmm_multiplier(std::vector<double>& table, const std::array<int, 3>& nx_physical,
                                  const std::array<double, 6>& cell, int axes, double coeff)
{
    check_crysfft_axes(axes);
    table.assign(static_cast<size_t>(nx_physical[0]) * nx_physical[1] * nx_physical[2], 0.0);

    const double factor_x = 2.0 * std::numbers::pi / cell[0];
    const double factor_y = 2.0 * std::numbers::pi / cell[1];
    const double factor_z = 2.0 * std::numbers::pi / cell[2];

    int idx = 0;
    for (int ix = 0; ix < nx_physical[0]; ++ix)
    {
        const double kx = ix * factor_x;
        const double kx2 = kx * kx;
        for (int iy = 0; iy < nx_physical[1]; ++iy)
        {
            const double ky = iy * factor_y;
            const double ky2 = ky * ky;
            for (int iz = 0; iz < nx_physical[2]; ++iz)
            {
                const double kz = iz * factor_z;
                const double kz2 = kz * kz;
                table[idx++] = crysfft_multiplier_value(axes, coeff, kx2, ky2, kz2);
            }
        }
    }
}

/**
 * @class Recursive3mTables
 * @brief The host part of the recursive 2x2y2z (3m) CrysFFT, shared by the CPU
 *        (CrysFFTRecursive3mBase) and CUDA (CudaCrysFFTRecursive3m) engines.
 *
 * Checks the grid, holds the pairing permutation and the twiddle factors of the
 * fold, and folds the values of the logical grid into a table. A table holds 16
 * arrays of M_physical values, `stride` values apart: the real parts of the eight
 * folded octants, then their imaginary parts. The algorithm is described in
 * "Accelerated pseudo-spectral method of self-consistent field theory via
 * crystallographic fast Fourier transform", Macromolecules (2020).
 */
class Recursive3mTables
{
public:
    /**
     * @param nx_logical   Logical grid (even sizes, Nz/2 >= 2)
     * @param translations Generator translations of the m3 mirrors (CrysFFTPlan::m3_translations)
     */
    Recursive3mTables(const std::array<int, 3>& nx_logical, const std::array<double, 9>& translations)
        : nx_logical_(nx_logical)
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0 || nx_logical_[d] % 2 != 0)
                throw_with_line_number("The recursive 3m CrysFFT requires even, positive grid dimensions.");
        }
        // The CPU multiply loop pads the packed z count to whole 64-byte lines, clamped
        // to the row stride Nz/2, which covers the packed range [0, Nz/4] for any
        // Nz/2 >= 2; the CUDA kernels take any size. Both platforms take the same grids.
        if (nx_logical_[2] / 2 < 2)
            throw_with_line_number("The recursive 3m CrysFFT requires Nz/2 >= 2.");

        // The fold pairs twiddle r_p with the octant-transformed array S_{psi(p)}, where
        // psi depends on the parities of the m3 generator translations in grid units
        // (Recursive3mFoldParity.h). When psi is not a bijection the even-index physical
        // basis does not exist on this grid and the fold cannot represent the group;
        // plan_crysfft() rejects such grids, but guard direct construction.
        if (!compute_m3_fold_permutation(translations, nx_logical_, fold_perm_))
        {
            throw_with_line_number("The m3 translations are incompatible with the grid of the recursive 3m CrysFFT "
                "(translation-parity permutation is not a bijection; no even-index physical basis exists).");
        }

        nx_physical_ = {nx_logical_[0] / 2, nx_logical_[1] / 2, nx_logical_[2] / 2};
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

        // Twiddle factors: the identity, then those of the seven non-identity operations
        for (int i = 0; i < 8; ++i)
        {
            twiddle_re_[i].assign(M_physical_, 0.0);
            twiddle_im_[i].assign(M_physical_, 0.0);
        }
        std::fill(twiddle_re_[0].begin(), twiddle_re_[0].end(), 1.0);
        const auto operations = crys_recursive_3m::generate_m3_operations(translations, nx_logical_);
        std::vector<std::tuple<double*, double*, std::array<double, 3>>> tasks;
        for (size_t i = 0; i < operations.size(); ++i)
            tasks.emplace_back(twiddle_re_[i + 1].data(), twiddle_im_[i + 1].data(), operations[i]);
        crys_recursive_3m::twiddle_factor(tasks, nx_physical_);
    }

    const std::array<int, 3>& nx_physical() const { return nx_physical_; }
    int M_physical() const { return M_physical_; }

    /// The folded Boltzmann factors crysfft_boltzmann_value(coeff) / M_logical (see fold_table()).
    void boltzmann(std::vector<double>& table, size_t stride, const std::array<double, 6>& cell, double coeff) const
    {
        fold_table(table, stride, cell, [coeff](double kx2, double ky2, double kz2) {
            return crysfft_boltzmann_value(coeff, kx2, ky2, kz2);
        });
    }

    /// The folded stress multiplier crysfft_multiplier_value(axes, coeff) / M_logical (see fold_table()).
    void multiplier(std::vector<double>& table, size_t stride, const std::array<double, 6>& cell,
                    int axes, double coeff) const
    {
        check_crysfft_axes(axes);
        fold_table(table, stride, cell, [axes, coeff](double kx2, double ky2, double kz2) {
            return crysfft_multiplier_value(axes, coeff, kx2, ky2, kz2);
        });
    }

    /**
     * @brief Write the folded table of a per-mode value into `table`.
     *
     * Fills the logical grid of `cell` with value_at(kx², ky², kz²) / M_logical
     * (logical_k_table()), splits it into its eight octants, applies the 2x2y2z
     * butterflies and multiplies with the parity-permuted twiddle factors.
     *
     * @param table    Resized to 16 arrays `stride` values apart, zero between them
     * @param stride   Values from one array to the next (at least M_physical)
     * @param value_at Callable double(double kx2, double ky2, double kz2)
     */
    template <typename ValueAt>
    void fold_table(std::vector<double>& table, size_t stride, const std::array<double, 6>& cell,
                    ValueAt&& value_at) const
    {
        using namespace crys_recursive_3m;
        table.assign(16 * stride, 0.0);

        // value_at(kx^2, ky^2, kz^2) / M_logical on the logical grid, split into 8 octants
        const std::vector<double> tempmat = logical_k_table(nx_logical_, cell, value_at);
        std::array<std::vector<double>, 8> k_split;
        std::array<double*, 8> k_split_ptr{};
        for (int i = 0; i < 8; ++i)
        {
            k_split[i].assign(M_physical_, 0.0);
            k_split_ptr[i] = k_split[i].data();
        }
        mat_split(tempmat.data(), k_split_ptr, nx_logical_);

        // The 2x2y2z decomposition
        constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
        add_sub_by_seq(k_split_ptr, seq0, nx_physical_);
        add_sub_by_seq(k_split_ptr, seq1, nx_physical_);
        add_sub_by_seq(k_split_ptr, seq2, nx_physical_);

        // The twiddle factors: r_p pairs with S_{fold_perm_[p]} (translation-parity
        // permutation; identity when all generator translations are even in grid units)
        constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
        std::array<double*, 8> table_re{}, table_im{};
        std::array<const double*, 8> r_re{}, r_im{}, k_split_perm{};
        for (int i = 0; i < 8; ++i)
        {
            table_re[i] = table.data() + i * stride;
            table_im[i] = table.data() + (8 + i) * stride;
            r_re[i] = twiddle_re_[i].data();
            r_im[i] = twiddle_im_[i].data();
            k_split_perm[i] = k_split[fold_perm_[i]].data();
        }
        mul_add_sub_by_seq(table_re, table_im, r_re, r_im, k_split_perm, seq3, nx_physical_);
    }

private:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_{};
    int M_physical_ = 0;
    /// Pairing permutation of the fold: twiddle r_p multiplies the octant-transformed array
    /// S_{fold_perm_[p]} (identity when all generator translations are an even number of grid
    /// cells; Recursive3mFoldParity.h)
    std::array<int, 8> fold_perm_{};
    std::array<std::vector<double>, 8> twiddle_re_;   ///< cos of the twiddle phases, one array per operation
    std::array<std::vector<double>, 8> twiddle_im_;   ///< sin of the twiddle phases
};

#endif  // CRYS_FFT_TABLES_H_
