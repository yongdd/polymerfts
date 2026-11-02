/**
 * @file FftwCrysFFTPmmm.cpp
 * @brief FFTW implementation of crystallographic FFT for Pmmm symmetry.
 */

#include "FftwCrysFFTPmmm.h"
#include "FftwFFT.h"

FftwCrysFFTPmmm::FftwCrysFFTPmmm(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part */)
    : CrysFFTPmmmBase(nx_logical, cell_para)
{
    dct_ = std::make_unique<MakhoulTransform>(std::vector<int>(nx_physical_.begin(), nx_physical_.end()),
                                              std::vector<bool>(3, false),
                                              std::make_unique<FftwFFT<double, 3>>(nx_physical_));
}
