/**
 * @file PlatformSelector.h
 * @brief Platform detection and factory instantiation for multi-backend support.
 *
 * This header provides the PlatformSelector class which detects available
 * computational platforms and creates the appropriate AbstractFactory instance.
 * It abstracts away the platform selection logic, allowing client code to
 * work with a unified interface regardless of whether CPU or GPU is used.
 *
 * **Supported Platforms:**
 *
 * - **cpu-fftw**: FFTW-based CPU implementation
 *   - Uses OpenMP for parallelization
 *   - Available when compiled with FFTW support
 *
 * - **cpu-mkl**: Intel MKL-based CPU implementation
 *   - Uses OpenMP for parallelization
 *   - Available when compiled with MKL support
 *
 * - **cuda**: NVIDIA CUDA GPU implementation
 *   - Uses cuFFT for FFT operations
 *   - Requires CUDA-capable GPU
 *   - Available when compiled with CUDA support
 *
 * @see AbstractFactory for the factory interface
 * @see CpuFactory for CPU implementation details (FFTW or MKL transforms)
 * @see CudaFactory for GPU implementation details
 *
 * @example
 * @code
 * // Check available platforms
 * std::vector<std::string> platforms = PlatformSelector::avail_platforms();
 * for (const auto& p : platforms) {
 *     std::cout << "Available: " << p << std::endl;
 * }
 * // Output of a build with all three backends on a GPU node: "cpu-fftw", "cpu-mkl", "cuda"
 *
 * // Create factory for CUDA platform
 * auto factory = PlatformSelector::create_factory<double>("cuda", false);
 *
 * // Create factory for CPU with memory reduction (only propagator checkpoints are stored)
 * auto factory_cpu = PlatformSelector::create_factory<double>("cpu-fftw", true);
 *
 * // Create factory for complex fields (e.g. complex Langevin simulations)
 * auto factory_complex = PlatformSelector::create_factory<std::complex<double>>("cuda", false);
 * @endcode
 */

#ifndef PLATFORM_SELECTOR_H_
#define PLATFORM_SELECTOR_H_

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include "AbstractFactory.h"

/**
 * @class PlatformSelector
 * @brief Static utility class for platform detection and factory creation.
 *
 * PlatformSelector provides static methods to:
 * 1. Query which computational platforms are available
 * 2. Create AbstractFactory instances for a specified platform
 *
 * **Platform Selection Strategy:**
 *
 * The Python interface (PropagatorSolver with platform "auto") selects the platform:
 * - 2D/3D simulations: "cuda" if available, otherwise the first CPU backend
 * - 1D simulations: the first CPU backend (GPU overhead not worthwhile)
 *
 * **Build Configuration:**
 *
 * Available platforms depend on CMake build configuration:
 * - CUDA support: Requires CUDA Toolkit and compatible GPU
 * - FFTW support: Requires FFTW library
 * - MKL support: Requires Intel MKL
 *
 * @note This class has only static methods; do not instantiate it.
 */
class PlatformSelector
{
public:
    /**
     * @brief Get list of available computational platforms.
     *
     * Returns a vector of platform name strings that can be passed to
     * create_factory().
     *
     * @return The platforms of this build in the order "cpu-fftw", "cpu-mkl", "cuda";
     *         "cuda" only when a GPU is visible
     *
     * @throws Exception if no platform is available
     *
     * @example
     * @code
     * auto platforms = PlatformSelector::avail_platforms();
     * if (std::find(platforms.begin(), platforms.end(), "cuda") != platforms.end()) {
     *     std::cout << "CUDA is available" << std::endl;
     * }
     * @endcode
     */
    static std::vector<std::string> avail_platforms();

    /**
     * @brief Create an AbstractFactory for fields of type T.
     *
     * @tparam T double for real fields (SCFT and L-FTS) or std::complex<double>
     *           for complex fields (e.g. CL-FTS)
     *
     * @param platform      Platform name: "cuda", "cpu-mkl", or "cpu-fftw"
     * @param reduce_memory If true, store only propagator checkpoints instead
     *                      of full histories, recomputing as needed.
     *                      Reduces memory usage but increases computation time.
     *
     * @return Owning pointer to the created factory
     *
     * @throws Exception if platform is not available or invalid
     *
     * @example
     * @code
     * // Create CUDA factory
     * auto factory = PlatformSelector::create_factory<double>("cuda", false);
     *
     * // Create solver with specific numerical method
     * auto solver = factory->create_propagator_computation(cb.get(), mols.get(), optimizer.get(), "rqm4");
     * // Or for real-space methods:
     * // auto solver = factory->create_propagator_computation(cb.get(), mols.get(), optimizer.get(), "cn-adi2");
     * @endcode
     */
    template <typename T>
    static std::unique_ptr<AbstractFactory<T>> create_factory(const std::string& platform, bool reduce_memory);
};

#endif
