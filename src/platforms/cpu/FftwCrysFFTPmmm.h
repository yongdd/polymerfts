/**
 * @file FftwCrysFFTPmmm.h
 * @brief FFTW implementation of crystallographic FFT for Pmmm symmetry.
 *
 * @see CrysFFTPmmmBase for common functionality
 * @see MklCrysFFTPmmm for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_PMMM_H_
#define FFTW_CRYS_FFT_PMMM_H_

#include <array>
#include "CrysFFTPmmmBase.h"

/**
 * @class FftwCrysFFTPmmm
 * @brief FFTW-based crystallographic FFT: the DCT-II/III of CrysFFTPmmmBase on the real FFT of FFTW.
 */
class FftwCrysFFTPmmm : public CrysFFTPmmmBase
{
public:
    /**
     * @brief Construct FftwCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     * @param trans_part (Ignored, kept for API compatibility)
     */
    FftwCrysFFTPmmm(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});
};

#endif  // FFTW_CRYS_FFT_PMMM_H_
