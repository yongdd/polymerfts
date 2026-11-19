/**
 * @file CudaSolver.cu
 * @brief Boltzmann factors of the fields, shared by the CUDA propagator solvers.
 *
 * See CudaSolver.h.
 */

#include <complex>
#include <string>

#include "CudaSolver.h"

template <typename T>
void CudaSolver<T>::compute_exp_dw(
    const std::map<std::string, const CuDeviceData<T>*>& d_w_input,
    const ContourLengthMapping& mapping, int n_table, const int* d_full_to_reduced,
    bool with_half, double factor, double factor_half)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int n_unique_ds = mapping.get_n_unique_ds();

    for (const auto& [monomer_type, d_w_input_monomer] : d_w_input)
    {
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const auto tables = d_exp_dw.find(ds_idx);
            if (tables == d_exp_dw.end() || !tables->second.contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");
        }

        // The field of the tables: the input itself, or its expansion from the reduced
        // basis into the table written last below, which is then computed in place
        const CuDeviceData<T>* d_w = d_w_input_monomer;
        if (d_full_to_reduced != nullptr)
        {
            CuDeviceData<T>* d_stage = (with_half ? d_exp_dw_half : d_exp_dw)
                                       .at(n_unique_ds - 1).at(monomer_type).get();
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_stage, d_w_input_monomer, d_full_to_reduced, n_table);
            CUDA_CHECK_LAUNCH();
            d_w = d_stage;
        }

        // d_exp_dw = exp(-w * local_ds * factor); d_exp_dw_half = exp(-w * local_ds * factor_half)
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const double local_ds = mapping.get_ds_from_index(ds_idx);
            ker_exp<<<N_BLOCKS, N_THREADS>>>(d_exp_dw.at(ds_idx).at(monomer_type).get(),
                                             d_w, 1.0, -factor*local_ds, n_table);
            CUDA_CHECK_LAUNCH();
            if (with_half)
            {
                ker_exp<<<N_BLOCKS, N_THREADS>>>(d_exp_dw_half.at(ds_idx).at(monomer_type).get(),
                                                 d_w, 1.0, -factor_half*local_ds, n_table);
                CUDA_CHECK_LAUNCH();
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiation
template class CudaSolver<double>;
template class CudaSolver<std::complex<double>>;
