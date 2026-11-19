/**
 * @file CudaCircularBuffer.h
 * @brief Ring buffer of arrays in GPU device memory or in pinned host memory.
 *
 * CudaAndersonMixing keeps the history of its fields and residuals in two of
 * these: in device memory, or in memory-saving mode in pinned (page-locked)
 * host memory, from which it copies an entry to the device when it uses it.
 *
 * **Usage Pattern:**
 *
 * @code
 * CudaCircularBuffer<double> buffer(10, 1024, false);  // 10 device arrays of 1024 elements
 *
 * buffer.insert(d_new_field);              // copies the array in (device or host memory)
 * double* d_prev = buffer.get_array(0);    // most recent
 * double* d_prev2 = buffer.get_array(1);   // second most recent
 * @endcode
 *
 * @see CircularBuffer for the CPU version
 * @see CudaAndersonMixing for its use
 */

#ifndef CUDA_CIRCULAR_BUFFER_H_
#define CUDA_CIRCULAR_BUFFER_H_

#include <vector>
#include "CudaCommon.h"

/**
 * @class CudaCircularBuffer
 * @brief Ring buffer of `length` arrays of `width` values, in device or pinned host memory.
 *
 * When full, the newest array overwrites the oldest.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 */
template <typename T>
class CudaCircularBuffer
{
private:
    int length;    ///< Maximum number of arrays (history depth)
    int width;     ///< Size of each array (n_grid)
    int start;     ///< Index of oldest element
    int n_items;   ///< Current number of items stored
    std::vector<CudaDeviceBuffer<CuDeviceData<T>>> device_elems;  ///< One device array per slot (device history)
    std::vector<CudaPinnedBuffer<CuDeviceData<T>>> host_elems;    ///< One pinned host array per slot (host history)

    /// The array of slot i (device or pinned host memory)
    CuDeviceData<T>* slot(int i) { return on_host() ? host_elems[i].get() : device_elems[i].get(); }

public:
    /**
     * @brief Allocate `length` arrays of `width` zeros.
     *
     * @param length  Maximum number of arrays to store
     * @param width   Size of each array (number of elements)
     * @param on_host Keep the arrays in pinned host memory instead of device memory
     */
    CudaCircularBuffer(int length, int width, bool on_host);

    /// Clear the buffer (the memory is kept).
    void reset();

    /**
     * @brief Copy an array in as the newest, overwriting the oldest when the buffer is full.
     *
     * @param new_arr Source array of `width` values, in device or host memory
     */
    void insert(const CuDeviceData<T>* new_arr);

    /**
     * @brief Array n of the history.
     *
     * @param n History index (0 = most recent, 1 = second most recent, ...)
     * @return Device pointer, or pinned host pointer when on_host()
     */
    CuDeviceData<T>* get_array(int n);

    /// True when the arrays are in pinned host memory.
    bool on_host() const { return !host_elems.empty(); }
};

#endif
