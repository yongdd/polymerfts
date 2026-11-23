/**
 * @file CudaComputationBox.cu
 * @brief CUDA implementation of computation box with GPU operations.
 *
 * Extends ComputationBox with GPU-accelerated integral and inner product
 * operations using CUB parallel reductions.
 *
 * **GPU Operations:**
 *
 * - integral_device(): Volume-weighted sum on GPU
 * - inner_product_device(): <f,g> = ∫ f(r)g(r) dv
 * - absolute_inner_product_device(): ∫ |f(r)g(r)/d(r)| dv
 *
 * **CUB Reductions:**
 *
 * Uses CUB library for high-performance parallel reductions:
 * - DeviceReduce::Sum for real fields
 * - Custom ComplexSumOp for complex fields
 *
 * **Memory Layout:**
 *
 * - d_dv: Device copy of volume elements
 * - d_sum, d_sum_out: Temporary arrays for reduction
 * - d_temp_storage: CUB workspace
 *
 * **Template Instantiations:**
 *
 * - CudaComputationBox<double>: Real fields
 * - CudaComputationBox<std::complex<double>>: Complex fields
 *
 * @see ComputationBox for base class interface
 */

#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "CudaComputationBox.h"
#include "CudaCommon.h"
#include "CudaTypeDispatch.h"
#include "SpaceGroup.h"

namespace {

/// out = |a b / divisor| weight (|a b| weight without a divisor); zero where the weight
/// is zero (a masked point), whatever a, b and the divisor are there
__global__ void ker_absolute_product(double* out, const double* a, const double* b, const double* divisor,
                                     const double* weights, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (weights[i] == 0.0)
            out[i] = 0.0;
        else
        {
            double product = a[i] * b[i];
            if (divisor != nullptr)
                product /= divisor[i];
            out[i] = fabs(product) * weights[i];
        }
    }
}

__global__ void ker_absolute_product(cuDoubleComplex* out, const cuDoubleComplex* a, const cuDoubleComplex* b,
                                     const cuDoubleComplex* divisor, const double* weights, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (weights[i] == 0.0)
            out[i] = make_cuDoubleComplex(0.0, 0.0);
        else
        {
            cuDoubleComplex product = cuCmul(a[i], b[i]);
            if (divisor != nullptr)
                product = cuCdiv(product, divisor[i]);
            out[i] = make_cuDoubleComplex(cuCabs(product) * weights[i], 0.0);
        }
    }
}

} // namespace

/**
 * @brief Construct CUDA computation box with device memory.
 *
 * @param nx   Grid points per dimension
 * @param lx   Box lengths per dimension
 * @param bc   Boundary conditions
 * @param mask Optional mask for impenetrable regions
 */
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox<T>(nx, lx, bc, mask)
{
    initialize();   // the owning buffers release what was acquired if this throws
}
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double* mask, SpaceGroup* space_group)
    : ComputationBox<T>(nx, lx, bc, angles, mask, space_group)
{
    initialize();   // the owning buffers release what was acquired if this throws
}
template <typename T>
void CudaComputationBox<T>::initialize()
{
    d_dv.allocate(this->total_grid);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
    if (this->space_group_ != nullptr)
    {
        const std::vector<double>& weights = this->basis_weights_;
        d_basis_weights_.allocate(weights.size());
        CUDA_CHECK(cudaMemcpy(d_basis_weights_, weights.data(), sizeof(double)*weights.size(), cudaMemcpyHostToDevice));
    }

    // Allocate memory for cub reduction sum
    d_sum.allocate(this->total_grid);
    d_sum_out.allocate(1);

    // Determine temporary storage size for cub reduction (the first call only queries the size)
    CUDA_CHECK(cuda_reduce_sum(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, 0));
    d_temp_storage.allocate(temp_storage_bytes);
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::reduce_sum_to_host(int n)
{
    T sum{0.0};
    CUDA_CHECK(cuda_reduce_sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), n, 0));
    CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//----------------- Destructor -----------------------------
template <typename T>
CudaComputationBox<T>::~CudaComputationBox() = default;   // the owning buffers release themselves
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    ComputationBox<T>::set_lx(new_lx);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    ComputationBox<T>::set_lattice_parameters(new_lx, new_angles);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
typename ComputationBox<T>::IntegralWeights CudaComputationBox<T>::device_integral_weights() const
{
    // The reduced-basis weights (orbit counts times the mask) with a space group, dv otherwise
    typename ComputationBox<T>::IntegralWeights weights = this->integral_weights();
    weights.weight = (this->space_group_ != nullptr) ? d_basis_weights_.get() : d_dv.get();
    return weights;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::integral_device(const CuDeviceData<T> *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const auto weights = device_integral_weights();

    // scale Σ_i g[i] weight[i]
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, weights.weight, 1.0, weights.n);
    CUDA_CHECK_LAUNCH();
    return reduce_sum_to_host(weights.n) * weights.scale;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::mean_device(const CuDeviceData<T> *d_g)
{
    return integral_device(d_g) / static_cast<T>(this->volume);
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const auto weights = device_integral_weights();

    // scale Σ_i g[i] h[i] weight[i]
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, weights.n);
    CUDA_CHECK_LAUNCH();
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, weights.weight, 1.0, weights.n);
    CUDA_CHECK_LAUNCH();
    return reduce_sum_to_host(weights.n) * weights.scale;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_inverse_weight_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h, const CuDeviceData<T>* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const auto weights = device_integral_weights();
    const bool reduced = this->space_group_ != nullptr;
    auto multiply_weights = [&]() {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, weights.weight, 1.0, weights.n);
        CUDA_CHECK_LAUNCH();
    };

    // scale Σ_i g[i] h[i] weight[i] / w[i], nothing at a masked point (zero weight) whatever w
    // is there. The full grid multiplies by its weights before the division, the reduced
    // basis after it.
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, weights.n);
    CUDA_CHECK_LAUNCH();
    if (!reduced)
        multiply_weights();
    if (this->get_mask() != nullptr)
        ker_divide_masked<<<grid_blocks(weights.n, N_THREADS), N_THREADS>>>(d_sum, d_sum, d_w, 1.0, weights.weight, weights.n);
    else
        ker_divide<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, weights.n);
    CUDA_CHECK_LAUNCH();
    if (reduced)
        multiply_weights();
    return reduce_sum_to_host(weights.n) * weights.scale;
}
//-----------------------------------------------------------
template <typename T>
double CudaComputationBox<T>::absolute_inner_product_device(
    const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h, const CuDeviceData<T>* d_divisor)
{
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const auto weights = device_integral_weights();

    ker_absolute_product<<<grid_blocks(weights.n, N_THREADS), N_THREADS>>>(
        d_sum, d_g, d_h, d_divisor, weights.weight, weights.n);
    CUDA_CHECK_LAUNCH();
    return std::real(reduce_sum_to_host(weights.n)) * weights.scale;
}

// Explicit template instantiation

template class CudaComputationBox<double>;
template class CudaComputationBox<std::complex<double>>;
