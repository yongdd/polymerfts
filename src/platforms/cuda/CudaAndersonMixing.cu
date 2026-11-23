/**
 * @file CudaAndersonMixing.cu
 * @brief CUDA implementation of Anderson mixing algorithm.
 *
 * Provides GPU-accelerated Anderson mixing for SCFT convergence using
 * device memory and CUB parallel reductions.
 *
 * **GPU Optimization:**
 *
 * - Field history in CudaCircularBuffer (device memory, or pinned host memory
 *   in memory-saving mode)
 * - Inner products computed with CUB reduction kernels
 * - The history terms of the new field added in one pass over the field
 *   (up to ANDERSON_TERMS_PER_LAUNCH terms per launch)
 *
 * **Algorithm:**
 *
 * 1. Store w_i and d_i history
 * 2. Compute U matrix inner products using CUB reduction
 * 3. Solve least-squares on CPU (small system)
 * 4. Apply mixing coefficients with CUDA kernels
 *
 * **Template Instantiations:**
 *
 * - CudaAndersonMixing<double>: Real field mixing
 * - CudaAndersonMixing<std::complex<double>>: Complex field mixing
 *
 * @see CudaCircularBuffer for the history storage
 * @see AndersonMixing for algorithm description
 */

#include <algorithm>
#include "CudaCommon.h"
#include "CudaTypeDispatch.h"
#include "CudaComputationBox.h"
#include "CudaCircularBuffer.h"
#include "CudaAndersonMixing.h"

namespace {

/// History terms of the new field added in one launch (passed as kernel parameters)
constexpr int ANDERSON_TERMS_PER_LAUNCH = 16;

template <typename TD>
struct AndersonTerms
{
    const TD* w[ANDERSON_TERMS_PER_LAUNCH];   ///< Older fields w_k
    const TD* d[ANDERSON_TERMS_PER_LAUNCH];   ///< Their residuals d_k
    TD a[ANDERSON_TERMS_PER_LAUNCH];          ///< a_k
    TD b[ANDERSON_TERMS_PER_LAUNCH];          ///< -a_k
    int n;
};

// acc += a x + b y
__device__ inline void add_lin_comb(double& acc, double a, double x, double b, double y)
{
    acc += a * x + b * y;
}
__device__ inline void add_lin_comb(cuDoubleComplex& acc, cuDoubleComplex a, cuDoubleComplex x,
                                    cuDoubleComplex b, cuDoubleComplex y)
{
    cuDoubleComplex t;
    t.x = a.x * x.x - a.y * x.y + b.x * y.x - b.y * y.y;
    t.y = a.x * x.y + a.y * x.x + b.x * y.y + b.y * y.x;
    acc.x += t.x;
    acc.y += t.y;
}

// w_new += Σ_k a_k w_k + b_k w_1 + a_k d_k + b_k d_1, term after term in the order of the
// history: the same operations as one launch per term, with w_new read and written once
template <typename TD>
__global__ void ker_add_history_terms(TD* w_new, AndersonTerms<TD> t, const TD* w_1, const TD* d_1, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        TD acc = w_new[i];
        const TD w1 = w_1[i];
        const TD d1 = d_1[i];
        for (int k = 0; k < t.n; ++k)
        {
            add_lin_comb(acc, t.a[k], t.w[k][i], t.b[k], w1);
            add_lin_comb(acc, t.a[k], t.d[k][i], t.b[k], d1);
        }
        w_new[i] = acc;
    }
}

/// Host scalar to the device scalar type.
inline double to_device_scalar(double value) { return value; }
inline cuDoubleComplex to_device_scalar(const std::complex<double>& value) { return stdToCuDoubleComplex(value); }

}  // namespace

/**
 * @brief Construct CUDA Anderson mixing with GPU buffers.
 *
 * @param n_var          Number of field variables
 * @param max_hist       Maximum history length
 * @param start_error    Error threshold to begin Anderson mixing
 * @param mix_min        Minimum mixing parameter
 * @param mix_init       Initial mixing parameter
 * @param pinned_history Keep the histories in pinned host memory
 */
template <typename T>
CudaAndersonMixing<T>::CudaAndersonMixing(
    int n_var, int max_hist, double start_error,
    double mix_min,   double mix_init, bool pinned_history)
    :AndersonMixing<T>(n_var, max_hist, start_error, mix_min, mix_init)
{
    // Number of anderson mixing steps, increases from 0 to this->max_hist
    this->n_anderson = -1;
    // Record history of w and w_deriv in GPU device memory or pinned host memory
    w_hist = std::make_unique<CudaCircularBuffer<T>>(this->max_hist+1, this->n_var, pinned_history);
    w_deriv_hist = std::make_unique<CudaCircularBuffer<T>>(this->max_hist+1, this->n_var, pinned_history);
    // Record history of inner_product product of w_deriv in CPU host memory
    cb_w_deriv_dots = std::make_unique<CircularBuffer<T>>(this->max_hist+1, this->max_hist+1);

    // define arrays for anderson mixing
    u_nm_storage.assign(static_cast<size_t>(this->max_hist) * this->max_hist, T());
    u_nm.resize(this->max_hist);
    for(int i=0; i<this->max_hist; i++)
        u_nm[i] = u_nm_storage.data() + static_cast<size_t>(i) * this->max_hist;
    v_n.assign(this->max_hist, T());
    a_n.assign(this->max_hist, T());
    w_deriv_dots.assign(this->max_hist+1, T());

    // Fields arrays
    d_w_current.allocate(this->n_var);
    d_w_new.allocate(this->n_var);
    d_w_deriv.allocate(this->n_var);
    d_sum.allocate(this->n_var);
    d_sum_out.allocate(1);
    if (pinned_history)
    {
        d_stage_w.allocate(this->n_var);
        d_stage_deriv.allocate(this->n_var);
    }

    // Allocate memory for cub reduction sum (the first call only queries the size)
    temp_storage_bytes = 0;
    CUDA_CHECK(cuda_reduce_sum(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var, 0));
    d_temp_storage.allocate(temp_storage_bytes);

    // Reset_count
    reset_count();
}
template <typename T>
CudaAndersonMixing<T>::~CudaAndersonMixing() = default;
template <typename T>
void CudaAndersonMixing<T>::reset_count()
{
    /* initialize mixing parameter */
    this->mix = this->mix_init;
    /* number of anderson mixing steps, increases from 0 to this->max_hist */
    this->n_anderson = -1;

    w_hist->reset();
    w_deriv_hist->reset();
    cb_w_deriv_dots->reset();
}
template <typename T>
const CuDeviceData<T>* CudaAndersonMixing<T>::entry_on_device(
    CudaCircularBuffer<T>& hist, int n, CudaDeviceBuffer<CuDeviceData<T>>& d_stage)
{
    if (!hist.on_host())
        return hist.get_array(n);
    CUDA_CHECK(cudaMemcpy(d_stage, hist.get_array(n), sizeof(T)*this->n_var, cudaMemcpyHostToDevice));
    return d_stage;
}
template <typename T>
void CudaAndersonMixing<T>::calculate_new_fields(
    T *w_new,
    T *w_current,
    T *w_deriv,
    double old_error_level,
    double error_level)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    CUDA_CHECK(cudaMemcpy(d_w_deriv,   w_deriv,   sizeof(T)*this->n_var, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_current, w_current, sizeof(T)*this->n_var, cudaMemcpyHostToDevice));

    // Condition to start anderson mixing
    if(error_level < this->start_error || this->n_anderson >= 0)
        this->n_anderson = this->n_anderson + 1;
    if( this->n_anderson >= 0 )
    {
        // Number of histories to use for anderson mixing
        this->n_anderson = std::min(this->max_hist, this->n_anderson);

        // store the input and output field (the memory is used in a periodic way): a
        // pinned history takes them from the host, a device history from the device
        if (w_hist->on_host())
        {
            w_hist->insert(reinterpret_cast<const CuDeviceData<T>*>(w_current));
            w_deriv_hist->insert(reinterpret_cast<const CuDeviceData<T>*>(w_deriv));
        }
        else
        {
            w_hist->insert(d_w_current);
            w_deriv_hist->insert(d_w_deriv);
        }

        // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
        // (the newest residual, i = 0, is d_w_deriv)
        for(int i=0; i<=this->n_anderson; i++)
        {
            const CuDeviceData<T>* d_w_deriv_i = (i == 0) ? d_w_deriv.get() : entry_on_device(*w_deriv_hist, i, d_stage_deriv);
            if constexpr (std::is_same_v<T, double>)
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_w_deriv_i, 1.0, this->n_var);
            else
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_w_deriv_i, this->n_var);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cuda_reduce_sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var, 0));
            CUDA_CHECK(cudaMemcpy(&w_deriv_dots[i], d_sum_out, sizeof(T),cudaMemcpyDeviceToHost));
        }

        cb_w_deriv_dots->insert(w_deriv_dots.data());
    }

    // Conditions to apply the simple mixing method
    if( this->n_anderson <= 0 )
    {
        // dynamically change mixing parameter
        if (old_error_level < error_level)
            this->mix = std::max(this->mix*0.7, this->mix_min);
        else
            this->mix = this->mix*1.01;

        // Make a simple mixing of input and output fields for the next iteration
        ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_current, this->mix, d_w_deriv, this->n_var);
        CUDA_CHECK_LAUNCH();
    }
    else
    {
        this->build_normal_equations(*cb_w_deriv_dots, u_nm.data(), v_n.data(), this->n_anderson);
        this->find_an(u_nm.data(), v_n.data(), a_n.data(), this->n_anderson);

        // Calculate the new field: w_new = w_1 + d_1 + Σ_k a_k (w_k - w_1 + d_k - d_1) over the
        // older entries k, where the newest field and residual w_1 and d_1 are d_w_current and
        // d_w_deriv. A pinned history copies one older entry at a time to the device.
        ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_current, 1.0, d_w_deriv, this->n_var);
        CUDA_CHECK_LAUNCH();
        const int terms_per_launch = w_hist->on_host() ? 1 : ANDERSON_TERMS_PER_LAUNCH;
        for (int first = 0; first < this->n_anderson; first += terms_per_launch)
        {
            AndersonTerms<CuDeviceData<T>> terms{};
            terms.n = std::min(terms_per_launch, this->n_anderson - first);
            for (int k = 0; k < terms.n; ++k)
            {
                const int i = first + k;
                terms.w[k] = entry_on_device(*w_hist, i+1, d_stage_w);
                terms.d[k] = entry_on_device(*w_deriv_hist, i+1, d_stage_deriv);
                terms.a[k] = to_device_scalar(a_n[i]);
                terms.b[k] = to_device_scalar(-a_n[i]);
            }
            ker_add_history_terms<<<N_BLOCKS, N_THREADS>>>(d_w_new.get(), terms, d_w_current.get(), d_w_deriv.get(), this->n_var);
            CUDA_CHECK_LAUNCH();
        }
    }
    CUDA_CHECK(cudaMemcpy(w_new, d_w_new, sizeof(T)*this->n_var, cudaMemcpyDeviceToHost));
}

// Explicit template instantiation
template class CudaAndersonMixing<double>;
template class CudaAndersonMixing<std::complex<double>>;
