/**
 * @file CudaAndersonMixing.h
 * @brief GPU implementation of Anderson Mixing algorithm.
 *
 * This header provides CudaAndersonMixing, the GPU-specific implementation
 * of Anderson Mixing for accelerating SCFT convergence. The field updates
 * and inner products run on the GPU.
 *
 * **History:**
 *
 * The field and residual histories (max_hist + 1 arrays of n_var values each)
 * are kept in GPU device memory, or in memory-saving mode in pinned host
 * memory: each entry is then copied to the device when it is used, which
 * trades GPU memory for PCIe transfers. Both give the same new fields, bit for bit.
 *
 * - Device history: 2 (max_hist + 1) + 4 arrays of n_var values on the GPU
 * - Pinned history: 6 arrays of n_var values on the GPU
 *
 * @see AndersonMixing for the algorithm description
 * @see CpuAndersonMixing for CPU implementation
 */

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "AndersonMixing.h"
#include <memory>
#include <vector>
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

/**
 * @class CudaAndersonMixing
 * @brief GPU-accelerated Anderson Mixing for SCFT iteration.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * - w_hist, w_deriv_hist: histories of the fields and residuals (device or pinned host memory)
 * - cb_w_deriv_dots: cached inner products of the residuals (CPU)
 * - d_w_current, d_w_deriv: the current field and residual on the GPU, which serve
 *   as the newest history entries
 * - d_stage_w, d_stage_deriv: a pinned history entry copied to the GPU (pinned history only)
 *
 * @example
 * @code
 * CudaAndersonMixing<double> am(n_grid, 20, 1e-1, 0.01, 0.1, false);
 *
 * // In SCFT loop
 * am.calculate_new_fields(w_new, w_current, w_deriv, old_err, err);
 * @endcode
 */
template <typename T>
class CudaAndersonMixing : public AndersonMixing<T>
{
private:
    std::unique_ptr<CudaCircularBuffer<T>> w_hist;         ///< Field history (device or pinned host memory)
    std::unique_ptr<CudaCircularBuffer<T>> w_deriv_hist;   ///< Residual history (device or pinned host memory)
    std::unique_ptr<CircularBuffer<T>> cb_w_deriv_dots;    ///< Cached dot products (CPU)
    std::vector<T> w_deriv_dots;                            ///< Current residual dots

    /// @name Least-squares solve arrays (on CPU)
    /// @{
    std::vector<T> u_nm_storage;   ///< max_hist x max_hist, row-major
    std::vector<T*> u_nm;          ///< Row pointers into u_nm_storage (find_an takes T**)
    std::vector<T> v_n;     ///< RHS vector
    std::vector<T> a_n;     ///< Solution coefficients
    /// @}

    /// @name Temporary GPU arrays
    /// @{
    CudaDeviceBuffer<CuDeviceData<T>> d_w_current;    ///< Current field on GPU
    CudaDeviceBuffer<CuDeviceData<T>> d_w_new;        ///< New field on GPU
    CudaDeviceBuffer<CuDeviceData<T>> d_w_deriv;      ///< Residual on GPU
    CudaDeviceBuffer<CuDeviceData<T>> d_sum;          ///< Reduction temp
    CudaDeviceBuffer<CuDeviceData<T>> d_stage_w;      ///< Pinned history: a field entry on GPU
    CudaDeviceBuffer<CuDeviceData<T>> d_stage_deriv;  ///< Pinned history: a residual entry on GPU
    /// @}

    /// @name CUB reduction storage
    /// @{
    size_t temp_storage_bytes = 0;
    CudaDeviceBuffer<unsigned char> d_temp_storage;
    CudaDeviceBuffer<CuDeviceData<T>> d_sum_out;
    /// @}

    /// History entry n on the GPU: the entry itself, or with a pinned history its copy in d_stage
    const CuDeviceData<T>* entry_on_device(CudaCircularBuffer<T>& hist, int n,
                                           CudaDeviceBuffer<CuDeviceData<T>>& d_stage);

public:
    /**
     * @brief Construct GPU Anderson mixing optimizer.
     *
     * @param n_var          Number of field variables (n_grid)
     * @param max_hist       Maximum history length
     * @param start_error    Error threshold for switching to Anderson mixing
     * @param mix_min        Minimum mixing parameter
     * @param mix_init       Initial mixing parameter
     * @param pinned_history Keep the histories in pinned host memory (memory-saving mode)
     */
    CudaAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init, bool pinned_history);

    /// Destructor. Frees GPU and pinned memory.
    ~CudaAndersonMixing();

    /// Reset history and iteration counter.
    void reset_count() override;

    /**
     * @brief Compute next iterate using Anderson mixing.
     *
     * @param w_new         Output: new field values (host)
     * @param w_current     Input: current field values (host)
     * @param w_deriv       Input: residual (host)
     * @param old_error_level Previous error level
     * @param error_level   Current error level
     */
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
