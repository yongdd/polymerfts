/**
 * @file CudaSolver.cu
 * @brief The context shared by the CUDA propagator solvers: the reduced basis around
 *        the steps and the Boltzmann factors of the fields.
 *
 * See CudaSolver.h.
 */

#include <complex>
#include <string>
#include <type_traits>

#include "CudaDeviceOps.cuh"
#include "CudaSolver.h"
#include "SpaceGroup.h"

template <typename T>
CudaSolver<T>::CudaSolver(
    ComputationBox<T>* cb, Molecules* molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group)
    : cb(cb), molecules(molecules), n_streams(n_streams), space_group_(space_group)
{
    // [real-only feature] The reduced basis of a space group is implemented for real fields
    if (space_group != nullptr && !std::is_same_v<T, double>)
        throw_with_line_number("Space group reduced basis is only supported for real fields.");
    for (int i = 0; i < n_streams; i++)
        this->streams[i] = streams[i];
    n_basis_ = (space_group != nullptr) ? space_group->get_n_reduced_basis() : cb->get_total_grid();

    // The index maps between the full grid and the reduced basis
    if (space_group != nullptr)
    {
        const int M = cb->get_total_grid();
        d_reduced_basis_indices_.allocate(n_basis_);
        CUDA_CHECK(cudaMemcpy(d_reduced_basis_indices_, space_group->get_reduced_basis_indices().data(),
                              sizeof(int)*n_basis_, cudaMemcpyHostToDevice));
        d_full_to_reduced_map_.allocate(M);
        CUDA_CHECK(cudaMemcpy(d_full_to_reduced_map_, space_group->get_full_to_reduced_map().data(),
                              sizeof(int)*M, cudaMemcpyHostToDevice));
    }
}

template <typename T>
const CuDeviceData<T>* CudaSolver<T>::boltzmann_factor(const std::string& monomer_type) const
{
    // d_exp_dw of the continuous chains holds exp(-w ds/2)
    if (molecules->get_model_name() != "discrete")
        throw_with_line_number("The Boltzmann factor of a full segment belongs to the discrete chains.");
    // The copy in the reduced basis exists exactly when compute_exp_dw() keeps one (the tables
    // on the full grid of a space group)
    if (!d_boltzmann_factor_.empty())
        return d_boltzmann_factor_.at(monomer_type).get();
    return d_exp_dw.at(0).at(monomer_type).get();
}

template <typename T>
void CudaSolver<T>::provide_full_grid(bool separate_output)
{
    if (space_group_ == nullptr)
        return;
    const int M = cb->get_total_grid();
    for (int i = 0; i < n_streams; i++)
    {
        provide(d_q_full_[i], M);
        if (separate_output)
            provide(d_q_full_out_[i], M);
    }
}

template <typename T>
CuDeviceData<T>* CudaSolver<T>::expand_to_full_grid(int STREAM, CuDeviceData<T>* d_q, CuDeviceData<T>* d_full)
{
    if (space_group_ == nullptr)
        return d_q;
    if (d_full == nullptr)
        d_full = d_q_full_[STREAM];
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_gather<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
        d_full, d_q, d_full_to_reduced_map_, cb->get_total_grid());
    CUDA_CHECK_LAUNCH();
    return d_full;
}

template <typename T>
CuDeviceData<T>* CudaSolver<T>::full_grid_output(int STREAM, CuDeviceData<T>* d_q_out)
{
    return (space_group_ != nullptr) ? d_q_full_[STREAM].get() : d_q_out;
}

template <typename T>
void CudaSolver<T>::store_full_grid(
    int STREAM, const CuDeviceData<T>* d_full, CuDeviceData<T>* d_q_out, const double* d_q_mask)
{
    if (space_group_ != nullptr)
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        ker_gather<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
            d_q_out, d_full, d_reduced_basis_indices_, n_basis_);
        CUDA_CHECK_LAUNCH();
    }
    // The mask is stored in the basis of d_q_out, so it applies after the reduction
    multiply_mask(STREAM, d_q_out, d_q_mask);
}

template <typename T>
void CudaSolver<T>::multiply_mask(int STREAM, CuDeviceData<T>* d_q, const double* d_q_mask)
{
    if (d_q_mask == nullptr)
        return;
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_q, d_q, d_q_mask, 1.0, n_basis_);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaSolver<T>::compute_exp_dw(
    const std::map<std::string, const CuDeviceData<T>*>& d_w_input, bool reduced_basis_tables,
    bool with_half, double factor, double factor_half, bool basis_copy)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    const int n_unique_ds = mapping.get_n_unique_ds();
    // Values of a table; the fields of a space group are expanded from the reduced basis
    // into tables on the full grid
    const int n_table = reduced_basis_tables ? n_basis_ : cb->get_total_grid();
    const bool expand = (space_group_ != nullptr) && !reduced_basis_tables;

    for (const auto& [monomer_type, d_w_input_monomer] : d_w_input)
    {
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const auto tables = d_exp_dw.find(ds_idx);
            if (tables == d_exp_dw.end() || !tables->second.contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");
        }

        // exp(-w ds) of ds index 0 also in the reduced basis, from the field in that basis:
        // the values of the table below at the points of the basis
        if (expand && basis_copy)
        {
            auto copy = d_boltzmann_factor_.find(monomer_type);
            if (copy == d_boltzmann_factor_.end())
            {
                // Enter the buffer only once it exists, so a failed allocation can be retried
                copy = d_boltzmann_factor_.emplace(monomer_type, CudaDeviceBuffer<CuDeviceData<T>>(n_basis_)).first;
            }
            ker_exp<<<N_BLOCKS, N_THREADS>>>(copy->second.get(), d_w_input_monomer, 1.0,
                                             -factor*mapping.get_ds_from_index(0), n_basis_);
            CUDA_CHECK_LAUNCH();
        }

        // The field of the tables: the input itself, or its expansion from the reduced
        // basis into the table written last below, which is then computed in place
        const CuDeviceData<T>* d_w = d_w_input_monomer;
        if (expand)
        {
            CuDeviceData<T>* d_stage = (with_half ? d_exp_dw_half : d_exp_dw)
                                       .at(n_unique_ds - 1).at(monomer_type).get();
            ker_gather<<<N_BLOCKS, N_THREADS>>>(d_stage, d_w_input_monomer, d_full_to_reduced_map_, n_table);
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
