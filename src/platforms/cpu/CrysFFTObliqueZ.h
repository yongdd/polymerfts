/**
 * @file CrysFFTObliqueZ.h
 * @brief Crystallographic FFT with a z-mirror on the CPU (FFTW or MKL), gamma arbitrary.
 *
 * The algorithm of CudaCrysFFTObliqueZ (CrysFFTObliqueZTable.h): the DCT-II along
 * z and the FFT along x, y run as one periodic real FFT of the physical grid
 * (Nx, Ny, Nz/2) after Makhoul's z permutation, through the real FFT of the
 * selected library (FftwRealFFT or MklRealFFT), and one pass over the half
 * spectrum applies the Boltzmann factors of the pair kz, Nz/2 - kz.
 */

#ifndef CRYS_FFT_OBLIQUEZ_H_
#define CRYS_FFT_OBLIQUEZ_H_

#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <vector>

#include "AlignedScratch.h"
#include "CpuCrysFFT.h"
#include "CrysFFTObliqueZTable.h"
#include "Exception.h"
#include "FFTFactory.h"
#include "MakhoulTransform.h"

/**
 * @class CrysFFTObliqueZ
 * @brief ObliqueZ crystallographic FFT: diffusion through one real FFT of the permuted physical grid.
 *
 * Its tables hold the (S, D) pairs of the half spectrum, with the round-trip
 * normalization 1/M_logical (build_obliquez_boltzmann()). ObliqueZ has no stress
 * multipliers.
 */
class CrysFFTObliqueZ : public CpuCrysFFT
{
private:
    std::array<int, 3> nx_physical_;   ///< (Nx, Ny, n = Nz/2)
    int M_physical_;                   ///< Nx * Ny * n
    int M_spectral_;                   ///< Nx * Ny * (n/2 + 1): half spectrum of the physical grid
    double norm_factor_;               ///< 1/M_logical

    std::vector<std::complex<double>> twiddle_;          ///< e^{i pi kz/n} per stored kz
    std::unique_ptr<MakhoulTransform::RealFFT> fft_;     ///< Real FFT of the physical grid

public:
    /**
     * @param nx_logical Logical grid dimensions (Nz even)
     * @param backend    Library of the real FFT (FFTW or MKL)
     */
    CrysFFTObliqueZ(std::array<int, 3> nx_logical, FFTBackend backend)
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical[d] <= 0)
                throw_with_line_number("CrysFFTObliqueZ requires positive grid dimensions.");
        }
        if (nx_logical[2] % 2 != 0)
            throw_with_line_number("CrysFFTObliqueZ requires even Nz.");

        nx_physical_ = {nx_logical[0], nx_logical[1], nx_logical[2] / 2};
        const int n = nx_physical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * n;
        M_spectral_ = nx_physical_[0] * nx_physical_[1] * (n / 2 + 1);
        norm_factor_ = 1.0 / static_cast<double>(nx_logical[0] * nx_logical[1] * nx_logical[2]);

        for (int kz = 0; kz < n / 2 + 1; ++kz)
        {
            const double angle = std::numbers::pi * kz / n;
            twiddle_.emplace_back(std::cos(angle), std::sin(angle));
        }
        fft_ = createRealFFT(std::vector<int>(nx_physical_.begin(), nx_physical_.end()), backend);
    }

    int get_M_physical() const override { return M_physical_; }

    /// The (S, D) pairs of the half spectrum for exp(-k² ds), 2 M_spectral values.
    void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell, double ds) const override
    {
        build_obliquez_boltzmann(table, nx_physical_, cell, ds, norm_factor_);
    }

    /// q_out = exp(ds ∇²) q_in with the table of build_boltzmann(ds); q_in is only read.
    void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const override
    {
        check_table(table, 2 * static_cast<size_t>(M_spectral_));
        const int Nx = nx_physical_[0];
        const int Ny = nx_physical_[1];
        const int n = nx_physical_[2];
        const int nz_c = n / 2 + 1;

        // Work arrays of this thread, from 64-byte boundaries, as the real FFT needs
        thread_local AlignedScratch real_scratch, spectrum_scratch;
        double* real = real_scratch.get(M_physical_);
        std::complex<double>* spectrum = reinterpret_cast<std::complex<double>*>(
            spectrum_scratch.get(2 * static_cast<size_t>(M_spectral_)));

        // Makhoul's permutation of each z line: v[m] = x[2m], v[n-1-j] = x[2j+1]
        for (int line = 0; line < Nx * Ny; ++line)
        {
            const double* x = q_in + static_cast<size_t>(line) * n;
            double* v = real + static_cast<size_t>(line) * n;
            for (int m = 0; 2 * m < n; ++m)
                v[m] = x[2 * m];
            for (int j = 0; 2 * j + 1 < n; ++j)
                v[n - 1 - j] = x[2 * j + 1];
        }
        fft_->forward(real, spectrum);

        // P'(k) = S P(k) + D w conj(P(-kx, -ky, kz)), with (S, D) of the table (with the
        // normalization) and w = e^{i pi kz/n}. In place: a line and its mirror need each
        // other's old values, so the lower line of each pair updates both; a line that is
        // its own mirror is updated once.
        const double* sd = table.data();
        auto mode = [](std::complex<double> p, std::complex<double> p_m, const double* f, std::complex<double> w) {
            // w * conj(p_m)
            const double wc_re = w.real() * p_m.real() + w.imag() * p_m.imag();
            const double wc_im = w.imag() * p_m.real() - w.real() * p_m.imag();
            return std::complex<double>(f[0] * p.real() + f[1] * wc_re, f[0] * p.imag() + f[1] * wc_im);
        };
        for (int ix = 0; ix < Nx; ++ix)
        {
            const int ix_m = (Nx - ix) % Nx;
            for (int iy = 0; iy < Ny; ++iy)
            {
                const int iy_m = (Ny - iy) % Ny;
                const size_t line = static_cast<size_t>(ix * Ny + iy) * nz_c;
                const size_t line_m = static_cast<size_t>(ix_m * Ny + iy_m) * nz_c;
                if (line > line_m)
                    continue;
                for (int kz = 0; kz < nz_c; ++kz)
                {
                    const std::complex<double> p = spectrum[line + kz];
                    const std::complex<double> p_m = spectrum[line_m + kz];
                    spectrum[line + kz] = mode(p, p_m, sd + 2 * (line + kz), twiddle_[kz]);
                    if (line_m != line)
                        spectrum[line_m + kz] = mode(p_m, p, sd + 2 * (line_m + kz), twiddle_[kz]);
                }
            }
        }

        // The inverse FFT overwrites the spectrum, which the next call recomputes
        fft_->backward(spectrum, real);
        for (int line = 0; line < Nx * Ny; ++line)
        {
            const double* v = real + static_cast<size_t>(line) * n;
            double* x = q_out + static_cast<size_t>(line) * n;
            for (int m = 0; 2 * m < n; ++m)
                x[2 * m] = v[m];
            for (int j = 0; 2 * j + 1 < n; ++j)
                x[2 * j + 1] = v[n - 1 - j];
        }
    }
};

#endif  // CRYS_FFT_OBLIQUEZ_H_
