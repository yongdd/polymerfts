/**
 * @file CudaCircularBuffer.cu
 * @brief Ring buffer of arrays in GPU device memory or in pinned host memory.
 *
 * See CudaCircularBuffer.h.
 */

#include <algorithm>

#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

template<typename T>
CudaCircularBuffer<T>::CudaCircularBuffer(int length, int width, bool on_host)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    if (on_host)
    {
        host_elems.resize(length);
        for (auto& elem : host_elems)
        {
            elem.allocate(width);
            std::fill_n(elem.get(), width, CuDeviceData<T>{});
        }
    }
    else
    {
        device_elems.resize(length);
        for (auto& elem : device_elems)
        {
            elem.allocate(width);
            CUDA_CHECK(cudaMemset(elem, 0, sizeof(T)*width));
        }
    }
}
template<typename T>
void CudaCircularBuffer<T>::reset()
{
    start = 0;
    n_items = 0;
}
template<typename T>
void CudaCircularBuffer<T>::insert(const CuDeviceData<T>* new_arr)
{
    int i = (start+n_items)%length;
    // The direction follows from the addresses (device or host memory on either side)
    CUDA_CHECK(cudaMemcpy(slot(i), new_arr, sizeof(T)*width, cudaMemcpyDefault));
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}
template<typename T>
CuDeviceData<T>* CudaCircularBuffer<T>::get_array(int n)
{
    return slot((start+n_items-n-1+length)%length);
}

// Explicit template instantiation
template class CudaCircularBuffer<double>;
template class CudaCircularBuffer<std::complex<double>>;
