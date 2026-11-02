/**
 * @file CrysFFTPmmmBase.h
 * @brief Base class for crystallographic FFT with Pmmm symmetry.
 *
 * Provides the DCT-based Pmmm symmetry FFT. Derived classes (FFTW, MKL) supply
 * the periodic real FFT under the DCT (MakhoulTransform).
 *
 * **Common Functionality:**
 *
 * - Grid dimension validation and setup
 * - Boltzmann factors shared by the calling threads (CrysFFTSharedTables.h)
 * - Cell parameter management
 * - Contour step management
 *
 * @see FftwCrysFFTPmmm for FFTW implementation
 * @see MklCrysFFTPmmm for MKL implementation
 */

#ifndef CRYS_FFT_PMMM_BASE_H_
#define CRYS_FFT_PMMM_BASE_H_

#include <array>
#include <cstring>
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "CrysFFTSharedTables.h"
#include "Exception.h"
#include "MakhoulTransform.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class CrysFFTPmmmBase
 * @brief Base class for Pmmm crystallographic FFT.
 */
class CrysFFTPmmmBase
{
protected:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // Cell parameters (Lx, Ly, Lz, alpha, beta, gamma)
    std::array<double, 6> cell_para_;

    // Normalization factor for DCT-II/III round trip: 1/M_logical
    double norm_factor_;

    /// Boltzmann factors per ds, shared by the threads that call the engine
    SharedTableCache<double, std::unique_ptr<double[]>> boltzmann_;

    /// DCT-II (FFTW's REDFT10) and DCT-III (REDFT01) of the physical grid; the
    /// derived class sets it. Their round trip gains 2n per dimension, which
    /// norm_factor_ removes.
    std::unique_ptr<MakhoulTransform> dct_;

    /// Generate Boltzmann factors for given ds.
    std::unique_ptr<double[]> generateBoltzmann(double ds) const
    {
        auto boltz = std::make_unique<double[]>(M_physical_);

        double Lx = cell_para_[0];
        double Ly = cell_para_[1];
        double Lz = cell_para_[2];

        int Nx2 = nx_physical_[0];
        int Ny2 = nx_physical_[1];
        int Nz2 = nx_physical_[2];

        const double PI = M_PI;

        int idx = 0;
        for (int ix = 0; ix < Nx2; ++ix)
        {
            double kx = ix * 2.0 * PI / Lx;
            double kx2 = kx * kx;

            for (int iy = 0; iy < Ny2; ++iy)
            {
                double ky = iy * 2.0 * PI / Ly;
                double ky2 = ky * ky;

                for (int iz = 0; iz < Nz2; ++iz)
                {
                    double kz = iz * 2.0 * PI / Lz;
                    double kz2 = kz * kz;

                    boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds);
                }
            }
        }

        return boltz;
    }

    /// Invalidate all Boltzmann factor caches.
    void freeBoltzmann()
    {
        boltzmann_.invalidate();
    }

    /// DCT coefficients of the physical grid, one array per thread.
    double* thread_buffer() const
    {
        thread_local std::vector<double> io;
        if (static_cast<int>(io.size()) < M_physical_)
            io.resize(M_physical_);
        return io.data();
    }

public:
    /**
     * @brief Construct CrysFFTPmmmBase for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     */
    CrysFFTPmmmBase(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para)
        : nx_logical_(nx_logical),
          cell_para_(cell_para)
    {
        // Validate even grid sizes
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] % 2 != 0)
            {
                throw_with_line_number("CrysFFTPmmm requires even grid dimensions. "
                    "Dimension " + std::to_string(d) + " has size " + std::to_string(nx_logical_[d]));
            }
            if (nx_logical_[d] <= 0)
            {
                throw_with_line_number("CrysFFTPmmm requires positive grid dimensions.");
            }
        }

        // Physical grid: (N/2) in each dimension
        nx_physical_ = {
            nx_logical_[0] / 2,
            nx_logical_[1] / 2,
            nx_logical_[2] / 2
        };

        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

        // DCT-II/III normalization: 1 / M_logical for round-trip
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);
    }

    /// Virtual destructor.
    virtual ~CrysFFTPmmmBase() = default;

    /// Update cell parameters.
    void set_cell_para(const std::array<double, 6>& cell_para)
    {
        if (cell_para[0] == cell_para_[0] &&
            cell_para[1] == cell_para_[1] &&
            cell_para[2] == cell_para_[2])
        {
            return;
        }

        cell_para_ = cell_para;
        freeBoltzmann();
    }

    /// Set contour step size and prepare Boltzmann factors.
    void set_contour_step(double ds)
    {
        boltzmann_.select(ds, [&] { return generateBoltzmann(ds); });
    }

    /**
     * @brief Apply diffusion operator using DCT-II/III.
     *
     * Computes: q_out = DCT-III[ exp(-k²ds) * DCT-II[q_in] ]
     */
    void diffusion(double* q_in, double* q_out)
    {
        const std::unique_ptr<double[]>* boltz = boltzmann_.current();
        if (!boltz)
        {
            throw_with_line_number("CrysFFTPmmm::set_contour_step must be called before diffusion().");
        }

        double* io_local = thread_buffer();

        // Step 1: Forward DCT-II
        dct_->forward(q_in, io_local, 1.0);

        // Step 2: Apply Boltzmann factor
        for (int i = 0; i < M_physical_; ++i)
            io_local[i] *= (*boltz)[i];

        // Step 3: Backward DCT-III with normalization
        dct_->backward(io_local, q_out, norm_factor_);
    }

    /// Apply custom multiplier in Fourier space.
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
    {
        double* io_local = thread_buffer();

        // Step 1: Forward DCT-II
        dct_->forward(q_in, io_local, 1.0);

        // Step 2: Apply multiplier
        for (int i = 0; i < M_physical_; ++i)
            io_local[i] *= multiplier[i];

        // Step 3: Backward DCT-III with normalization
        dct_->backward(io_local, q_out, norm_factor_);
    }

    // Accessors
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif  // CRYS_FFT_PMMM_BASE_H_
