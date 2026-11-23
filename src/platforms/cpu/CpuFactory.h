/**
 * @file CpuFactory.h
 * @brief Concrete factory for the CPU platforms (cpu-fftw and cpu-mkl).
 *
 * This header provides the CpuFactory class, the implementation of
 * AbstractFactory for computations on the host. The two CPU platforms differ
 * only in the FFT library of their transforms, the FFTBackend the factory
 * passes to the objects it creates:
 *
 * - cpu-fftw: FFTW3. FFTW3 is licensed under GPL; distributing binaries built
 *   with it requires compliance with GPL terms.
 * - cpu-mkl: Intel MKL. The factory turns MKL's internal threading off: the
 *   parallelism is at the propagator level.
 *
 * **Platform Features:**
 *
 * - OpenMP parallelization over propagators
 * - Continuous and discrete chain models, pseudo-spectral and real-space solvers
 * - DCT/DST through MakhoulTransform over the real FFT of the backend
 * - Whether FFTW or MKL is faster depends on the processor
 *   (docs/internals/FFTImplementation.md, Section 8.2)
 *
 * @see AbstractFactory for the factory interface
 * @see CudaFactory for GPU implementation
 * @see PlatformSelector for factory instantiation
 *
 * @example
 * @code
 * // Create the factory of the cpu-fftw platform
 * CpuFactory<double> factory(false, FFTBackend::FFTW);  // reduce_memory=false for speed
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

#ifndef CPU_FACTORY_H_
#define CPU_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "FFT.h"  // FFTBackend

/**
 * @class CpuFactory
 * @brief Factory for creating CPU simulation objects with FFTW3 or Intel MKL transforms.
 *
 * All created objects operate on host memory and use OpenMP for parallelization.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Created Object Types:**
 *
 * - ComputationBox: Grid and integration on CPU
 * - Propagator computations with the CPU backend (transforms of the FFT backend)
 * - CpuAndersonMixing: Iteration accelerator
 * - CpuFourierTransform: Fourier multipliers of whole fields
 *
 * @note The reduce_memory flag enables checkpointing mode for memory-constrained
 *       simulations at the cost of increased computation time.
 */
template <typename T>
class CpuFactory : public AbstractFactory<T>
{
public :
    /**
     * @brief Construct a CPU factory.
     * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
     * @param backend       FFT library of the transforms (one the build includes)
     */
    CpuFactory(bool reduce_memory, FFTBackend backend);

    /**
     * @brief Create a CPU computation box.
     * @param nx   Grid dimensions
     * @param lx   Box lengths
     * @param bc   Boundary conditions
     * @param mask Optional mask for impenetrable regions
     * @return ComputationBox instance
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
     * @param space_group Optional space group of the reduced basis (see ComputationBox)
     * @return ComputationBox instance
     */
    std::unique_ptr<ComputationBox<T>> create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr,
        SpaceGroup* space_group=nullptr) override;

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
     * @brief Create a CPU Fourier transform with the transforms of the FFT backend.
     * @param cb Computation box
     * @param bc Transform of each axis
     * @return CpuFourierTransform instance
     */
    std::unique_ptr<FourierTransform<T>> create_fourier_transform(
        ComputationBox<T>* cb, std::vector<BoundaryCondition> bc) override;

    /**
     * @brief Display CPU platform information.
     *
     * Prints the version of the FFT library (and, for MKL, its configuration).
     */
    void display_info() override;

protected:
    /// CPU backend with the transforms of the FFT backend (see AbstractFactory::create_backend())
    std::unique_ptr<ComputationBackend<T>> create_backend(
        ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
        SpaceGroup* space_group, int max_n_streams) override;

private:
    FFTBackend backend_;   ///< FFT library of the transforms
};
#endif
