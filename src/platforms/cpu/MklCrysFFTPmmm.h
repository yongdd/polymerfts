/**
 * @file MklCrysFFTPmmm.h
 * @brief MKL implementation of crystallographic FFT for Pmmm symmetry.
 *
 * @see CrysFFTPmmmBase for common functionality
 * @see FftwCrysFFTPmmm for FFTW implementation
 */

#ifndef MKL_CRYS_FFT_PMMM_H_
#define MKL_CRYS_FFT_PMMM_H_

#include <array>
#include "CrysFFTPmmmBase.h"

/**
 * @class MklCrysFFTPmmm
 * @brief MKL-based crystallographic FFT: the DCT-II/III of CrysFFTPmmmBase on the real FFT of MKL.
 */
class MklCrysFFTPmmm : public CrysFFTPmmmBase
{
public:
    /**
     * @brief Construct MklCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     * @param trans_part (Ignored, kept for API compatibility)
     */
    MklCrysFFTPmmm(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});
};

#endif  // MKL_CRYS_FFT_PMMM_H_
