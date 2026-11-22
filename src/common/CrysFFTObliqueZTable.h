/**
 * @file CrysFFTObliqueZTable.h
 * @brief The Boltzmann table of the ObliqueZ CrysFFT, built once for the CPU and CUDA engines.
 *
 * ObliqueZ (a z-mirror, alpha = beta = 90 degrees, gamma arbitrary) stores the
 * physical grid (Nx, Ny, n) with n = Nz/2. The DCT-II along z and the FFT along
 * x, y run as one real FFT of that grid after Makhoul's z permutation (even
 * samples forward, odd samples backward). One pass over the half spectrum
 * (kz < n/2 + 1) then forms the DCT coefficients of the pair kz, n - kz,
 * applies their Boltzmann factors B and returns to the half spectrum of the
 * inverse permutation:
 *
 *   P'(k) = S(k) P(k) + D(k) e^{i pi kz/n} conj(P(-kx, -ky, kz)),
 *
 * with S = B(kz) + B(n - kz), D = B(kz) - B(n - kz) and B(n) = 0. This needs
 * B to be even in (kx, ky), which holds for alpha = beta = 90 degrees.
 * CrysFFTObliqueZ (CPU) and CudaCrysFFTObliqueZ run this algorithm with the
 * table of build_obliquez_boltzmann().
 */

#ifndef CRYS_FFT_OBLIQUEZ_TABLE_H_
#define CRYS_FFT_OBLIQUEZ_TABLE_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "MinimalFormAlias.h"
#include "ReciprocalMetric.h"

/**
 * @brief Visit the in-plane modes of the ObliqueZ grid (Nx, Ny) with their
 *        Boltzmann factors: visit(index, boltzmann) in the row-major order of
 *        the shape (Nx, ny_shape), where boltzmann(m3) = exp(-k² ds) of the mode
 *        with z index m3.
 *
 * ny_shape is Ny, or Ny/2 + 1 for the half spectrum of an in-plane real FFT.
 * For oblique in-plane cells (gamma != 90), the in-plane alias representative
 * minimizes the metric form, matching the invariant (first-Brillouin-zone)
 * convention used by Pseudo for the standard full-grid path. This keeps the
 * ObliqueZ diffusion operator exactly symmetric under the hexagonal point group
 * and mode by mode consistent with the full-grid multiplier. The minimal form
 * is the same for (kx, ky) and (-kx, -ky), which makes the factors even in
 * (kx, ky), as the spectral update requires; the search is exact
 * (MinimalFormAlias.h), since a window of one period misses the minimum on
 * cells with strongly anisotropic grid spacing. The z (DCT) index is a true
 * non-negative frequency and needs no re-aliasing (ObliqueZ requires
 * alpha = beta = 90, i.e. G13 = G23 = 0). Rectangular cells, where gamma = 90
 * degrees gives cos(pi/2) ~ 6e-17, keep the centered representative
 * (reciprocal_axes_coupled() is a relative-tolerance gate).
 */
template <typename Visit>
inline void for_each_obliquez_mode(int Nx, int Ny, int ny_shape, const std::array<double, 6>& cell, double ds,
                                   Visit&& visit)
{
    const std::array<double, 6> recip_metric = cell_geometry(cell).recip_metric;
    const double G11 = recip_metric[0];
    const double G12 = recip_metric[1];
    const double G13 = recip_metric[2];
    const double G22 = recip_metric[3];
    const double G23 = recip_metric[4];
    const double G33 = recip_metric[5];
    const double factor = 4.0 * std::numbers::pi * std::numbers::pi;

    const IndexMetric metric_xy{G11, G22, 0.0, G12, 0.0, 0.0};
    auto form_xy = [&](double a, double b, double) {
        return G11 * a * a + 2.0 * G12 * a * b + G22 * b * b;
    };
    for_each_mode_min_alias({Nx, Ny, 1}, {Nx, ny_shape, 1}, metric_xy, form_xy,
        [&](long long index, const std::array<int, 3>& rep) {
            const int m1 = rep[0], m2 = rep[1];
            visit(index, [&](int m3) {
                const double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                                 + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                return std::exp(-factor * gmm * ds);
            });
        });
}

/**
 * @brief Write the (S, D) pairs of the half spectrum of the physical grid
 *        nx_physical = (Nx, Ny, n) into `table`: 2 Nx Ny (n/2 + 1) values, kz
 *        running fastest, B = exp(-k² ds) times norm_factor (the 1/M_logical of
 *        the round trip). In place when `table` has that size already.
 */
inline void build_obliquez_boltzmann(std::vector<double>& table, const std::array<int, 3>& nx_physical,
                                     const std::array<double, 6>& cell, double ds, double norm_factor)
{
    const int n = nx_physical[2];
    const int nz_c = n / 2 + 1;
    table.resize(2 * static_cast<size_t>(nx_physical[0]) * nx_physical[1] * nz_c);

    // (S, D) = (B(kz) + B(n - kz), B(kz) - B(n - kz)) with B(n) = 0, per half-spectrum entry
    for_each_obliquez_mode(nx_physical[0], nx_physical[1], nx_physical[1], cell, ds,
        [&](long long ixy, auto&& boltzmann) {
            for (int kz = 0; kz < nz_c; ++kz)
            {
                const size_t idx = static_cast<size_t>(ixy) * nz_c + kz;
                const double b = boltzmann(kz) * norm_factor;
                const double b_mirror = (kz == 0) ? 0.0 : boltzmann(n - kz) * norm_factor;
                table[2 * idx] = b + b_mirror;
                table[2 * idx + 1] = b - b_mirror;
            }
        });
}

#endif  // CRYS_FFT_OBLIQUEZ_TABLE_H_
