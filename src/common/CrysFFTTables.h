/**
 * @file CrysFFTTables.h
 * @brief k-space tables of the Pmmm crystallographic FFT, built on the host for the CPU
 *        (CrysFFTPmmm) and CUDA (CudaCrysFFT) engines.
 *
 * The Pmmm physical grid holds N_d/2 points per axis, and its cosine series has the
 * wavenumbers k_d = 2π m_d / L_d with m_d < N_d/2. A table holds one value per point of
 * the physical grid, z fastest.
 */

#ifndef CRYS_FFT_TABLES_H_
#define CRYS_FFT_TABLES_H_

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "Exception.h"

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

                table[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds);
            }
        }
    }
}

/**
 * @brief The stress multipliers Σ_{d in axes} k_d² of the Pmmm physical grid, times
 *        exp(-k² coeff) when coeff is not 0 (the bond factor of discrete chains).
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
    if (axes < 1 || axes > 7)
        throw_with_line_number("A Pmmm stress multiplier takes a mask of the axes 1 (x), 2 (y) and 4 (z).");
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
                const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0)
                                    + ((axes & 4) ? kz2 : 0.0);
                const double k2 = kx2 + ky2 + kz2;
                table[idx++] = (coeff != 0.0) ? std::exp(-k2 * coeff) * k_axes : k_axes;
            }
        }
    }
}

#endif  // CRYS_FFT_TABLES_H_
