/**
 * @file CpuFactory.cpp
 * @brief Implementation of the factory of the CPU platforms.
 *
 * Provides the abstract factory implementation for creating CPU-based
 * computational objects with the transforms of FFTW3 or Intel MKL. This factory
 * creates all platform-specific classes needed for SCFT/L-FTS simulations.
 *
 * **Created Objects:**
 *
 * - ComputationBox: Grid and integration on CPU
 * - CpuAndersonMixing: CPU-optimized field mixing
 * - CpuComputationBackend with the transforms of the FFT backend, around which
 *   AbstractFactory::create_propagator_computation() builds the computation
 *
 * **Template Instantiations:**
 *
 * - CpuFactory<double>: Factory for real field simulations
 * - CpuFactory<std::complex<double>>: Factory for complex field simulations
 *
 * @see AbstractFactory for factory interface
 * @see PlatformSelector for factory creation
 */

#include <cstdio>
#include <iostream>
#include <array>
#include <vector>
#include <string>

#ifdef USE_CPU_FFTW
#include <fftw3.h>
#endif
#ifdef USE_CPU_MKL
#include "mkl_service.h"
#endif

#include "ComputationBox.h"
#include "CpuComputationBackend.h"
#include "CpuAndersonMixing.h"
#include "CpuFourierTransform.h"
#include "CpuFactory.h"

/**
 * @brief Construct a CPU factory with optional checkpointing.
 *
 * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
 * @param backend       FFT library of the transforms
 */
template <typename T>
CpuFactory<T>::CpuFactory(bool reduce_memory, FFTBackend backend) : backend_(backend)
{
    this->reduce_memory = reduce_memory;
#ifdef USE_CPU_MKL
    if (backend_ == FFTBackend::MKL)
        mkl_set_num_threads(1);  // Disable MKL internal threading (parallelism at propagator level)
#endif
}

template <typename T>
std::unique_ptr<ComputationBox<T>> CpuFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double *mask)
{
    return std::make_unique<ComputationBox<T>>(nx, lx, bc, mask);
}

template <typename T>
std::unique_ptr<ComputationBox<T>> CpuFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double *mask)
{
    return std::make_unique<ComputationBox<T>>(nx, lx, bc, angles, mask);
}

template <typename T>
std::unique_ptr<ComputationBackend<T>> CpuFactory<T>::create_backend(
    ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
    SpaceGroup* space_group, int max_n_streams)
{
    return std::make_unique<CpuComputationBackend<T>>(cb, molecules, method_info, backend_, space_group, max_n_streams);
}

template <typename T>
std::unique_ptr<AndersonMixing<T>> CpuFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return std::make_unique<CpuAndersonMixing<T>>(
        n_var, max_hist, start_error, mix_min, mix_init);
}

template <typename T>
std::unique_ptr<FourierTransform<T>> CpuFactory<T>::create_fourier_transform(
    ComputationBox<T>* cb, std::vector<BoundaryCondition> bc)
{
    return std::make_unique<CpuFourierTransform<T>>(cb, bc, backend_);
}

template <typename T>
void CpuFactory<T>::display_info()
{
#ifdef USE_CPU_FFTW
    if (backend_ == FFTBackend::FFTW)
    {
        std::cout << "==================== FFTW Version ====================" << std::endl;
        std::cout << "Version:                 " << fftw_version << std::endl;
        std::cout << "================================================================" << std::endl;
    }
#endif
#ifdef USE_CPU_MKL
    if (backend_ == FFTBackend::MKL)
    {
        MKLVersion Version;

        mkl_get_version(&Version);
        std::cout<< "==================== MKL Version ====================" << std::endl;
        printf("Major version:           %d\n",Version.MajorVersion);
        printf("Minor version:           %d\n",Version.MinorVersion);
        printf("Update version:          %d\n",Version.UpdateVersion);
        printf("Product status:          %s\n",Version.ProductStatus);
        printf("Build:                   %s\n",Version.Build);
        printf("Platform:                %s\n",Version.Platform);
        printf("Processor optimization:  %s\n",Version.Processor);
        printf("================================================================\n");
    }
#endif
}

// Explicit template instantiation
template class CpuFactory<double>;
template class CpuFactory<std::complex<double>>;
