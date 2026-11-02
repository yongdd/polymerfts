/**
 * @file FftwFactory.h
 * @brief Concrete factory for FFTW3-based CPU implementations.
 *
 * This header provides the FftwFactory class, which is the concrete implementation
 * of AbstractFactory for CPU-based computations using FFTW3 (Fastest Fourier
 * Transform in the West). It creates CPU-optimized versions of all simulation objects.
 *
 * **Platform Features:**
 *
 * - Uses FFTW3 for FFT operations (O(N log N) for all transform types)
 * - OpenMP parallelization for multi-core CPUs
 * - Supports both continuous and discrete chain models
 * - Supports pseudo-spectral and real-space solvers
 * - DCT/DST through MakhoulTransform over FFTW real FFT plans
 *
 * **Performance Notes:**
 *
 * - O(N log N) DCT/DST through FFTW's real FFT
 * - Whether FFTW or MKL is faster depends on the processor
 *   (docs/internals/FFTImplementation.md, Section 8.2)
 *
 * **License Note:**
 *
 * FFTW3 is licensed under GPL. Using this factory requires compliance with GPL terms.
 *
 * @see AbstractFactory for the factory interface
 * @see FftwFactory for FFTW implementation
 * @see PlatformSelector for factory instantiation
 *
 * @example
 * @code
 * // Create FFTW factory
 * FftwFactory<double> factory(false);  // reduce_memory=false for speed
 *
 * // Create simulation objects (each create function returns a std::unique_ptr)
 * auto cb = factory.create_computation_box(nx, lx, bc);
 * auto mols = factory.create_molecules_information("continuous", 0.01, bonds);
 * mols->add_polymer(1.0, blocks);
 * auto optimizer = factory.create_propagator_computation_optimizer(mols.get(), true);
 * auto solver = factory.create_propagator_computation(cb.get(), mols.get(), optimizer.get(), "rqm4");
 *
 * // Display platform info
 * factory.display_info();
 * @endcode
 */

#ifndef FFTW_FACTORY_H_
#define FFTW_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

/**
 * @class FftwFactory
 * @brief Factory for creating FFTW3-based CPU simulation objects.
 *
 * Implements AbstractFactory to create CPU-optimized objects using FFTW3.
 * All created objects operate on host memory and use OpenMP for parallelization.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Created Object Types:**
 *
 * - CpuComputationBox: Grid and integration on CPU
 * - Propagator computations with the CPU backend (FFTW transforms)
 * - CpuAndersonMixing: Iteration accelerator
 *
 * @note The reduce_memory flag enables checkpointing mode for memory-constrained
 *       simulations at the cost of increased computation time.
 */
template <typename T>
class FftwFactory : public AbstractFactory<T>
{
public :
    /**
     * @brief Construct an FFTW factory.
     * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
     */
    FftwFactory(bool reduce_memory);

    /**
     * @brief Create a CPU computation box.
     * @param nx   Grid dimensions
     * @param lx   Box lengths
     * @param bc   Boundary conditions
     * @param mask Optional mask for impenetrable regions
     * @return CpuComputationBox instance
     */
    std::unique_ptr<ComputationBox<T>> create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        const double* mask=nullptr) override;

    /**
     * @brief Create a CPU computation box with lattice angles.
     * @param nx     Grid dimensions
     * @param lx     Box lengths
     * @param bc     Boundary conditions
     * @param angles Lattice angles [alpha, beta, gamma] in degrees
     * @param mask   Optional mask for impenetrable regions
     * @return CpuComputationBox instance
     */
    std::unique_ptr<ComputationBox<T>> create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) override;

    /**
     * @brief Create CPU Anderson mixing optimizer.
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     * @return CpuAndersonMixing instance
     */
    std::unique_ptr<AndersonMixing<T>> create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    /**
     * @brief Create a CPU (FFTW) Fourier transform.
     * @param cb Computation box
     * @param bc Transform of each axis
     * @return CpuFourierTransform instance
     */
    std::unique_ptr<FourierTransform<T>> create_fourier_transform(
        ComputationBox<T>* cb, std::vector<BoundaryCondition> bc) override;

    /**
     * @brief Display CPU platform information.
     *
     * Prints FFTW version and other configuration details.
     */
    void display_info() override;

protected:
    /// CPU backend with FFTW transforms (see AbstractFactory::create_backend())
    std::unique_ptr<ComputationBackend<T>> create_backend(
        ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
        SpaceGroup* space_group, int max_n_streams) override;
};
#endif
