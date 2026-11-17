/**
 * @file CrysFFTObliqueZBase.h
 * @brief Base class for crystallographic FFT with z-mirror symmetry.
 *
 * Implements the ObliqueZ algorithm for space groups with a z-mirror
 * operation (t_z = 0 or 1/2) and alpha=beta=90° (gamma arbitrary).
 *
 * @see FftwCrysFFTObliqueZ for FFTW implementation
 * @see MklCrysFFTObliqueZ for MKL implementation
 */

#ifndef CRYS_FFT_OBLIQUEZ_BASE_H_
#define CRYS_FFT_OBLIQUEZ_BASE_H_

#include <array>
#include <cmath>
#include <vector>

#include "CpuCrysFFT.h"
#include "Exception.h"
#include "MinimalFormAlias.h"
#include "ReciprocalMetric.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class CrysFFTObliqueZBase
 * @brief Base class for ObliqueZ crystallographic FFT.
 *
 * Provides the grid sizes and the Boltzmann factors of a cell (build_boltzmann(),
 * from the reciprocal metric tensor). The derived classes (FftwCrysFFTObliqueZ,
 * MklCrysFFTObliqueZ) provide the transforms: initFFTPlans(), diffusion().
 * ObliqueZ has no stress multipliers.
 */
class CrysFFTObliqueZBase : public CpuCrysFFT
{
protected:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    double norm_factor_{1.0};

public:
    explicit CrysFFTObliqueZBase(std::array<int, 3> nx_logical)
        : nx_logical_(nx_logical)
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0)
                throw_with_line_number("CrysFFTObliqueZ requires positive grid dimensions.");
        }
        if (nx_logical_[2] % 2 != 0)
            throw_with_line_number("CrysFFTObliqueZ requires even Nz.");

        nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        M_complex_ = nx_logical_[0] * (nx_logical_[1] / 2 + 1) * nx_physical_[2];
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);
    }

    /// exp(-k² ds) of the half spectrum, M_complex values with z running fastest.
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double ds) const override
    {
        table.assign(M_complex_, 0.0);

        const std::array<double, 6> recip_metric = cell_geometry(cell).recip_metric;
        const double G11 = recip_metric[0];
        const double G12 = recip_metric[1];
        const double G13 = recip_metric[2];
        const double G22 = recip_metric[3];
        const double G23 = recip_metric[4];
        const double G33 = recip_metric[5];
        const double factor = 4.0 * M_PI * M_PI;

        // For oblique in-plane cells (gamma != 90), pick the in-plane alias
        // representative minimizing the metric form, matching the invariant
        // (first-Brillouin-zone) convention used by Pseudo for the standard
        // full-grid path. This keeps the ObliqueZ diffusion operator exactly
        // symmetric under the hexagonal point group and mode-by-mode
        // consistent with the full-grid multiplier. The z (DCT) index is a
        // true non-negative frequency and needs no re-aliasing (ObliqueZ
        // requires alpha = beta = 90, i.e. G13 = G23 = 0). Rectangular cells,
        // where gamma = 90 degrees gives cos(pi/2) ~ 6e-17, keep the centered
        // representative (index_metric_has_coupling).
        const std::array<int, 3> grid_xy = {nx_logical_[0], nx_logical_[1], 1};
        const IndexMetric metric_xy{G11, G22, 0.0, G12, 0.0, 0.0};
        auto form_xy = [&](double a, double b, double) {
            return G11 * a * a + 2.0 * G12 * a * b + G22 * b * b;
        };

        const int Nyh = nx_logical_[1] / 2 + 1;
        const int Nz2 = nx_physical_[2];
        for_each_mode_min_alias(grid_xy, {nx_logical_[0], Nyh, 1}, metric_xy, form_xy,
            [&](long long ixy, const std::array<int, 3>& rep) {
                const int m1 = rep[0], m2 = rep[1];
                for (int iz = 0; iz < Nz2; ++iz)
                {
                    int m3 = iz;
                    double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                               + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                    double k2 = factor * gmm;
                    table[ixy * Nz2 + iz] = std::exp(-k2 * ds);
                }
            });
    }

    int get_M_physical() const override { return M_physical_; }
};

#endif  // CRYS_FFT_OBLIQUEZ_BASE_H_
