/**
 * @file CudaSolverPseudoBase.cu
 * @brief Shared implementation of the CUDA pseudo-spectral solvers.
 *
 * See CudaSolverPseudoBase.h.
 */

#include <algorithm>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>

#include "CudaSolverPseudoBase.h"
#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaDeviceOps.cuh"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"
#include "CrysFFTSelector.h"
#include "PseudoSolverHelpers.h"
#include "SpaceGroup.h"

namespace {

// Stress dyad sums V_c = Σ_k factor·integrand(k)·basis_c(k) for up to six components
// in one pass over k: each block writes its partial sums, and a one-block kernel adds
// the partials in block order, so the result does not depend on scheduling.
constexpr int STRESS_THREADS = 256;      // threads per block of both stress kernels
constexpr int STRESS_MAX_BLOCKS = 1024;  // capacity of the partial-sum buffer

struct StressBases { const double* basis[6]; };

// Same product as ker_multi(dst, integrand, basis, factor)
__device__ inline double stress_term(double integrand, double basis, double factor) { return factor * integrand * basis; }
__device__ inline cuDoubleComplex stress_term(cuDoubleComplex integrand, double basis, double factor)
{
    return make_cuDoubleComplex(factor * integrand.x * basis, factor * integrand.y * basis);
}

// Warp sum (all 32 lanes take part; lane 0 gets the total)
__device__ inline double stress_warp_sum(double v)
{
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__device__ inline cuDoubleComplex stress_warp_sum(cuDoubleComplex v)
{
    return make_cuDoubleComplex(stress_warp_sum(v.x), stress_warp_sum(v.y));
}

// Block sum of NC per-thread values (blockDim.x == STRESS_THREADS); thread 0 gets the totals
template <typename TD, int NC>
__device__ inline void stress_block_sum(TD (&acc)[NC])
{
    __shared__ TD sh[NC][STRESS_THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int c = 0; c < NC; ++c)
    {
        acc[c] = stress_warp_sum(acc[c]);
        if (lane == 0)
            sh[c][warp] = acc[c];
    }
    __syncthreads();
    if (warp == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            acc[c] = stress_warp_sum(lane < STRESS_THREADS / 32 ? sh[c][lane] : zero_of(TD{}));
    }
}

template <typename TD, int NC>
__global__ void ker_stress_dyad_partials(TD* partial, const TD* integrand, StressBases bases, double factor, int M)
{
    TD acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c)
        acc[c] = zero_of(TD{});
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        const TD v = integrand[i];
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            acc[c] = added(acc[c], stress_term(v, bases.basis[c][i], factor));
        i += blockDim.x * gridDim.x;
    }
    stress_block_sum<TD, NC>(acc);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            partial[blockIdx.x * NC + c] = acc[c];
    }
}

template <typename TD, int NC>
__global__ void ker_stress_dyad_finalize(TD* out, const TD* partial, int n_partial)
{
    TD acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c)
        acc[c] = zero_of(TD{});
    for (int b = threadIdx.x; b < n_partial; b += blockDim.x)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            acc[c] = added(acc[c], partial[b * NC + c]);
    }
    stress_block_sum<TD, NC>(acc);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            out[c] = acc[c];
    }
}

// out[0..NC-1] = Σ_i factor·integrand[i]·basis_c[i]: block partials, then their sum in block order
template <typename TD, int NC>
void launch_stress_dyads(TD* out, TD* partial, const TD* integrand, const StressBases& bases,
                         double factor, int M, cudaStream_t stream)
{
    // About two elements per thread, at most STRESS_MAX_BLOCKS blocks
    const int n_partial = std::max(1, std::min(STRESS_MAX_BLOCKS, (M + 2 * STRESS_THREADS - 1) / (2 * STRESS_THREADS)));
    ker_stress_dyad_partials<TD, NC><<<n_partial, STRESS_THREADS, 0, stream>>>(partial, integrand, bases, factor, M);
    CUDA_CHECK_LAUNCH();
    ker_stress_dyad_finalize<TD, NC><<<1, STRESS_THREADS, 0, stream>>>(out, partial, n_partial);
    CUDA_CHECK_LAUNCH();
}

template <typename TD>
void launch_stress_dyads(TD* out, TD* partial, const TD* integrand, const StressBases& bases, int n_comp,
                         double factor, int M, cudaStream_t stream)
{
    switch (n_comp)
    {
        case 1: launch_stress_dyads<TD, 1>(out, partial, integrand, bases, factor, M, stream); break;
        case 2: launch_stress_dyads<TD, 2>(out, partial, integrand, bases, factor, M, stream); break;
        case 3: launch_stress_dyads<TD, 3>(out, partial, integrand, bases, factor, M, stream); break;
        case 6: launch_stress_dyads<TD, 6>(out, partial, integrand, bases, factor, M, stream); break;
        default: throw_with_line_number("Unsupported number of stress components: " + std::to_string(n_comp));
    }
}

// CrysFFT diffusion on the physical grid of a reduced basis: a reduced-basis factor through the
// physical -> reduced map, and the orbit weights of the stress fast path
__global__ void ker_expand_reduced_basis_multi(
    double* dst, const double* src, const double* reduced_multiplier,
    const int* phys_to_reduced_map, const int n_phys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_phys)
    {
        int ridx = phys_to_reduced_map[i];
        dst[i] = src[ridx] * reduced_multiplier[ridx];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_map(
    double* dst, const double* reduced_multiplier, const int* phys_to_reduced_map, const int n_phys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_phys)
    {
        dst[i] *= reduced_multiplier[phys_to_reduced_map[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_weight(
    double* dst, const double* src, const int* weights, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n)
    {
        dst[i] = src[i] * static_cast<double>(weights[i]);
        i += blockDim.x * gridDim.x;
    }
}

// CrysFFT stress fast path: d_stress[g] holds the sum of axis group g (see
// crysfft_stress_axis_groups; masks 1: x, 2: y, 4: z); every axis of a group gets the
// group sum divided by the group size, times its scale b²·M_full/L_d² (single thread)
__global__ void ker_crysfft_stress_finalize(
    double* d_stress, int n_groups, int mask0, int mask1, int mask2, double s0, double s1, double s2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    const int masks[3] = {mask0, mask1, mask2};
    double group_sum[3] = {0.0, 0.0, 0.0};
    for (int g = 0; g < n_groups; ++g)
        group_sum[g] = d_stress[g];

    double v[3] = {0.0, 0.0, 0.0};
    for (int g = 0; g < n_groups; ++g)
    {
        const int n_axes = ((masks[g] & 1) != 0) + ((masks[g] & 2) != 0) + ((masks[g] & 4) != 0);
        for (int d = 0; d < 3; ++d)
            if (masks[g] & (1 << d))
                v[d] = group_sum[g] / n_axes;
    }

    d_stress[0] = v[0] * s0;
    d_stress[1] = v[1] * s1;
    d_stress[2] = v[2] * s2;
}

}  // namespace

template <typename T>
CudaSolverPseudoBase<T>::CudaSolverPseudoBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group,
    const int* d_reduced_basis_indices,
    const int* d_full_to_reduced_map,
    int max_fft_batch,
    bool with_exp_dw_half)
    : CudaSolver<T>(cb, molecules, n_streams, streams, space_group, d_reduced_basis_indices, d_full_to_reduced_map)
{
    this->dim_ = cb->get_dim();
    this->max_fft_batch_ = max_fft_batch;

    // The device buffers, cuFFT plans and transform objects are owning members,
    // released automatically (also when a constructor throws)
    use_crysfft_ = false;
    crysfft_engine_ = CrysFFTEngine::None;
    crysfft_identity_map_ = false;
    crysfft_physical_size_ = 0;

    // Both faces of each direction need the same BC: the transforms take whole directions
    is_periodic_ = pseudo_spectral_periodic(cb->get_boundary_conditions(), dim_);
    // The Boltzmann factor buffers exp(-w ds) per distinct contour step and
    // monomer type; provide_work() allocates them for the field basis of the path
    molecules->finalize_contour_length_mapping();
    const int n_unique_ds = molecules->get_contour_length_mapping().get_n_unique_ds();
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        for(const auto& item: molecules->get_bond_lengths())
        {
            const std::string& monomer_type = item.first;
            this->d_exp_dw[ds_idx][monomer_type];
            if (with_exp_dw_half)
                this->d_exp_dw_half[ds_idx][monomer_type];
        }
    }
    // The CrysFFT objects and the work of the path follow in init_path(), at the
    // end of the derived constructor, and the workspace of the CrysFFT stress fast
    // path in prepare_stress()
}

template <typename T>
CudaSolverPseudoBase<T>::~CudaSolverPseudoBase() = default;

template <typename T>
void CudaSolverPseudoBase<T>::init_fft()
{
    // Creates what is missing (empty plans and series tell), so a call that
    // failed part way can be repeated
    if (is_periodic_)
    {
        // Forward and backward plans of each batch size up to full_grid_fft_batch() per stream;
        // they run one after another on their stream and share one work area.
        // cufftMakePlanMany reads dim_ entries of total_grid
        const int NRANK = dim_;
        int total_grid[3] = {0, 0, 0};
        for (int d = 0; d < dim_; ++d)
            total_grid[d] = cb->get_nx(d);
        const int n_batch = full_grid_fft_batch();

        for (int i = 0; i < n_streams; i++)
        {
            // The plans of a stream are moved in last, once all of them are ready
            if (!work_[i].plan_forward[0].empty())
                continue;
            CufftPlan forward[2], backward[2];
            size_t work_bytes = 0;
            for (int b = 0; b < n_batch; b++)
            {
                CUFFT_CHECK(forward[b].create_without_work_area(work_bytes, cufftMakePlanMany, NRANK, total_grid,
                    nullptr, 1, 0, nullptr, 1, 0, CufftPlanType<T>::forward, b + 1));
                CUFFT_CHECK(backward[b].create_without_work_area(work_bytes, cufftMakePlanMany, NRANK, total_grid,
                    nullptr, 1, 0, nullptr, 1, 0, CufftPlanType<T>::backward, b + 1));
            }
            if (work_bytes > 0)
                provide(work_[i].d_fft_work, (work_bytes + sizeof(double) - 1) / sizeof(double));
            for (int b = 0; b < n_batch; b++)
            {
                for (CufftPlan* plan : {&forward[b], &backward[b]})
                {
                    if (work_bytes > 0)
                        CUFFT_CHECK(cufftSetWorkArea(*plan, work_[i].d_fft_work.get()));
                    CUFFT_CHECK(cufftSetStream(*plan, streams[i]));
                }
            }
            for (int b = 0; b < n_batch; b++)
            {
                work_[i].plan_forward[b] = std::move(forward[b]);
                work_[i].plan_backward[b] = std::move(backward[b]);
            }
        }
    }
    else
    {
        // The cosine and sine series of each stream: each has its own scratch, so the
        // streams never race on it. One BC per dimension (both faces match).
        const auto bc_faces = cb->get_boundary_conditions();
        std::vector<BoundaryCondition> bc;
        for (int d = 0; d < dim_; ++d)
            bc.push_back(bc_faces[2 * d]);
        for (int i = 0; i < n_streams; i++)
            if (!work_[i].series)
                work_[i].series = std::make_unique<CudaMakhoulTransform<T>>(cb->get_nx(), bc);
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::fft_forward(int stream, CuDeviceData<T>* d_in, cuDoubleComplex* d_out, int batch)
{
    CUFFT_CHECK(cufft_exec_forward(work_[stream].plan_forward[batch - 1], d_in, d_out));
}

template <typename T>
void CudaSolverPseudoBase<T>::fft_backward(int stream, cuDoubleComplex* d_in, CuDeviceData<T>* d_out, int batch)
{
    CUFFT_CHECK(cufft_exec_backward(work_[stream].plan_backward[batch - 1], d_in, d_out));
}

template <typename T>
size_t CudaSolverPseudoBase<T>::spectrum_doubles() const
{
    // The complex spectrum of the FFT, or one series coefficient (two doubles for a complex
    // field) per grid point
    if (is_periodic_)
        return 2 * static_cast<size_t>(total_complex_grid());
    return (sizeof(T) / sizeof(double)) * static_cast<size_t>(cb->get_total_grid());
}

template <typename T>
void CudaSolverPseudoBase<T>::transform_forward(int STREAM, CuDeviceData<T>* d_in, double* d_out)
{
    if (is_periodic_)
    {
        fft_forward(STREAM, d_in, reinterpret_cast<cuDoubleComplex*>(d_out), 1);
        return;
    }
    // The cosine and sine series are half of REDFT10/RODFT10 per axis, as MakhoulFFT's on the CPU
    work_[STREAM].series->forward(d_in, d_out, std::ldexp(1.0, -dim_), streams[STREAM]);
}

template <typename T>
void CudaSolverPseudoBase<T>::bond_step(
    int STREAM, CuDeviceData<T>* d_in, CuDeviceData<T>* d_out, const double* boltz, double scale)
{
    double* d_spectrum = spectrum(STREAM, 0);
    transform_forward(STREAM, d_in, d_spectrum);
    if (is_periodic_)
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        cuDoubleComplex* d_qk = reinterpret_cast<cuDoubleComplex*>(d_spectrum);
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(d_qk, boltz, scale, total_complex_grid());
        CUDA_CHECK_LAUNCH();
        fft_backward(STREAM, d_qk, d_out, 1);
        return;
    }
    // REDFT01/RODFT01 of the series: M times their inverse, unnormalized as the inverse FFT
    work_[STREAM].series->backward(d_spectrum, boltz, d_out, scale, streams[STREAM]);
}

template <typename T>
void CudaSolverPseudoBase<T>::crysfft_diffuse(
    int STREAM, double* d_q_in, double* d_q_out, double coeff,
    const double* pre, const double* mid, const double* post)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M_phys = crysfft_physical_size_;
    const cudaStream_t stream = streams[STREAM];
    CudaCrysFFTBase* crysfft = work_[STREAM].crysfft.get();
    const auto boltz_it = crysfft_boltz_.find(coeff);
    if (boltz_it == crysfft_boltz_.end())
        throw_with_line_number("No CrysFFT table was built for the step coefficient " + std::to_string(coeff) +
            " (see crysfft_block_coeffs()).");
    const CudaDeviceBuffer<double>& boltz = boltz_it->second;

    if (crysfft_identity_map_)
    {
        // The physical grid is the reduced basis: work in d_q_out, in place (an
        // out-of-place diffusion would first copy its input into d_q_out)
        if (pre != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_in, pre, 1.0, M_phys);
            CUDA_CHECK_LAUNCH();
            crysfft->diffusion(d_q_out, d_q_out, boltz, stream);
        }
        else
        {
            crysfft->diffusion(d_q_in, d_q_out, boltz, stream);   // preserves d_q_in when out of place
        }
        if (mid != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_out, mid, 1.0, M_phys);
            CUDA_CHECK_LAUNCH();
            crysfft->diffusion(d_q_out, d_q_out, boltz, stream);
        }
        if (post != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_out, post, 1.0, M_phys);
            CUDA_CHECK_LAUNCH();
        }
        return;
    }

    // Expand to the physical grid, diffuse there in place and reduce back
    double* d_work = work_[STREAM].d_crysfft_in;
    if (pre != nullptr)
        ker_expand_reduced_basis_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_work, d_q_in, pre, d_crysfft_phys_to_reduced_, M_phys);
    else
        ker_gather<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_work, d_q_in, d_crysfft_phys_to_reduced_, M_phys);
    CUDA_CHECK_LAUNCH();

    crysfft->diffusion(d_work, d_work, boltz, stream);
    if (mid != nullptr)
    {
        ker_multi_map<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work, mid, d_crysfft_phys_to_reduced_, M_phys);
        CUDA_CHECK_LAUNCH();
        crysfft->diffusion(d_work, d_work, boltz, stream);
    }
    if (post != nullptr)
    {
        ker_multi_map<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work, post, d_crysfft_phys_to_reduced_, M_phys);
        CUDA_CHECK_LAUNCH();
    }

    ker_gather<<<N_BLOCKS, N_THREADS, 0, stream>>>(
        d_q_out, d_work, d_crysfft_reduced_to_phys_, n_basis_);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaSolverPseudoBase<T>::init_path()
{
    // [type-specific algorithm] The CrysFFT fast path folds real fields on the
    // physical sub-grid; complex fields always use the full-grid reduced basis.
    if constexpr (std::is_same_v<T, double>)
    {
        if (space_group_ != nullptr && dim_ == 3 && is_periodic_)
            init_crysfft();
    }

    // The backend's maps and the CrysFFT setup above are snapshots: freeze the physical basis
    if (space_group_ != nullptr)
        space_group_->mark_in_use();

    // The work of the path, before the first computation
    provide_work();
}

template <typename T>
void CudaSolverPseudoBase<T>::init_crysfft()
{
    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    // Engine choice and its basis map, shared with the CPU solver
    const CrysFFTPlan plan = plan_crysfft(
        space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), cb->is_z_axis_orthogonal());
    if (plan.engine != CrysFFTEngine::None)
        setup_crysfft(plan);
}

template <typename T>
void CudaSolverPseudoBase<T>::setup_crysfft(const CrysFFTPlan& plan)
{
    const CrysFFTBasisMapping& map = plan.map;
    crysfft_physical_size_ = map.M_phys;
    crysfft_identity_map_ = map.identity;

    // Index maps are needed only when the physical grid is not the reduced basis itself
    if (!crysfft_identity_map_)
    {
        d_crysfft_phys_to_reduced_.allocate(map.M_phys);
        d_crysfft_reduced_to_phys_.allocate(map.M_reduced);
        CUDA_CHECK(cudaMemcpy(d_crysfft_phys_to_reduced_, map.phys_to_reduced.data(),
                                   sizeof(int) * map.M_phys, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_crysfft_reduced_to_phys_, map.reduced_to_phys.data(),
                                   sizeof(int) * map.M_reduced, cudaMemcpyHostToDevice));
    }

    // Orbit counts for the CrysFFT stress fast path
    const auto& orbit_counts = space_group_->get_orbit_counts();
    d_crysfft_orbit_counts_.allocate(map.M_reduced);
    CUDA_CHECK(cudaMemcpy(d_crysfft_orbit_counts_, orbit_counts.data(),
                               sizeof(int) * map.M_reduced, cudaMemcpyHostToDevice));

    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};

    for (int i = 0; i < n_streams; i++)
    {
        // One transform per stream (plans and work buffers); the tables are shared
        if (i > 0)
            work_[i].crysfft = work_[0].crysfft->clone();
        else if (plan.engine == CrysFFTEngine::Recursive3m)
            work_[i].crysfft = std::make_unique<CudaCrysFFTRecursive3m>(nx_logical, plan.m3_translations);
        else if (plan.engine == CrysFFTEngine::ObliqueZ)
            work_[i].crysfft = std::make_unique<CudaCrysFFTObliqueZ>(nx_logical);
        else
            work_[i].crysfft = std::make_unique<CudaCrysFFT>(nx_logical);

        work_[i].d_crysfft_in.allocate(map.M_phys);
        work_[i].d_crysfft_out.allocate(map.M_phys);
    }
    crysfft_engine_ = plan.engine;
    use_crysfft_ = true;

    // The Boltzmann tables of the steps of the blocks, built here for the cell
    const auto block_coeffs = [this](const std::string& monomer_type, int ds_index) {
        return crysfft_block_coeffs(monomer_type, ds_index);
    };
    for (double coeff : crysfft_step_coeffs(*molecules, block_coeffs))
        crysfft_boltz_.try_emplace(coeff);
    update_crysfft_tables();
}

template <typename T>
void CudaSolverPseudoBase<T>::update_crysfft_tables()
{
    // In place: the kernels of every stream that may still read the tables finish first
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::array<double, 6> cell = crysfft_cell(*cb);
    for (auto& [coeff, table] : crysfft_boltz_)
        work_[0].crysfft->build_boltzmann(table, cell, coeff, streams[0]);
    for (auto& [key, table] : crysfft_multipliers_)
        work_[0].crysfft->build_multiplier(table, cell, key.first, key.second, streams[0]);
}

template <typename T>
void CudaSolverPseudoBase<T>::update_laplacian_operator()
{
    try{
        // The CrysFFT tables are rebuilt now, and so are the Pseudo tables when the
        // full-grid path uses them; otherwise prepare_stress() computes them for the
        // cell in use if the off-diagonal flag turns the stress to the full grid
        pseudo_current_ = false;
        if (use_crysfft_)
            update_crysfft_tables();
        if (full_grid_path())
            update_pseudo();
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::update_pseudo()
{
    if (!pseudo)
    {
        // The first full-grid use. The Boltzmann bond factors of every distinct contour
        // step when the full-grid propagator steps or the discrete stress use them (the
        // CrysFFT steps of continuous chains do not); face BCs squeezed to one per
        // dimension (both faces match).
        const auto bc_faces = cb->get_boundary_conditions();
        std::vector<BoundaryCondition> bc_dim;
        for (int d = 0; d < dim_; ++d)
            bc_dim.push_back(bc_faces[2 * d]);
        std::map<int, double> ds_values;
        if (!use_crysfft_ || stress_has_bond_factor())
        {
            const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
            for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
                ds_values[ds_idx] = mapping.get_ds_from_index(ds_idx);
        }
        pseudo = std::make_unique<CudaPseudo<T>>(molecules->get_bond_lengths(), bc_dim, cb->get_nx(), ds_values);
    }
    // The stress basis once a stress is computed (prepare_stress())
    pseudo->update(cb->get_dx(), cb->get_recip_metric(), stress_basis_);
    pseudo_current_ = true;
}

template <typename T>
bool CudaSolverPseudoBase<T>::full_grid_path() const
{
    // The stress fast path implies the CrysFFT propagator steps
    return !crysfft_stress_fast_path();
}

template <typename T>
int CudaSolverPseudoBase<T>::full_grid_fft_batch() const
{
    return use_crysfft_ ? 1 : max_fft_batch_;
}

template <typename T>
int CudaSolverPseudoBase<T>::total_complex_grid() const
{
    return Pseudo<T>::complex_grid_size(cb->get_total_grid(), cb->get_nx(dim_ - 1), is_periodic_);
}

template <typename T>
void CudaSolverPseudoBase<T>::provide_work()
{
    const int M = cb->get_total_grid();
    // Values of a field in the propagator steps: the reduced basis on the CrysFFT path
    const int n_field = use_crysfft_ ? n_basis_ : M;

    // exp(-w ds) per contour step and monomer type (compute_exp_dw() stages a
    // reduced-basis field expanded to the full grid in one of them)
    for (auto* tables : {&this->d_exp_dw, &this->d_exp_dw_half})
        for (auto& [ds_idx, by_type] : *tables)
            for (auto& [monomer_type, table] : by_type)
                provide(table, n_field);

    if (full_grid_path())
    {
        // Transforms, spectral scratch and reduced-basis expansion buffer of the full grid,
        // with the stress workspace from the first prepare_stress() on
        init_fft();
        const int M_COMPLEX = total_complex_grid();
        // The second spectrum of the scratch holds the RQM4 half step of the full-grid steps (a batch
        // of two, periodic BC) or the second propagator of the stress: the other steps, the CrysFFT
        // steps among them, transform one field at a time, and their solvers hold it only from the
        // first prepare_stress() on
        const size_t n_spectra = ((full_grid_fft_batch() == 2 && is_periodic_) || stress_basis_) ? 2 : 1;
        for (int i = 0; i < n_streams; i++)
        {
            StreamWork& work = work_[i];
            if (stress_basis_)
            {
                provide(work.d_q_multi, M_COMPLEX);
                provide(work.d_stress_partial, 6*STRESS_MAX_BLOCKS);
            }
            // The spectral scratch of the steps and the stress. The second spectrum comes with the
            // first stress: the larger buffer is allocated before the steps' one is released, so a
            // failed allocation leaves the steps theirs
            const size_t spectra_size = n_spectra*spectrum_doubles();
            if (work.d_spectra.size() != spectra_size)
                work.d_spectra = CudaDeviceBuffer<double>(spectra_size);
            // Complex fields with periodic BC gather q2(-k) for the stress apart
            if constexpr (std::is_same<T, std::complex<double>>::value)
                if (stress_basis_ && is_periodic_)
                    provide(work.d_qk_conj, M_COMPLEX);
        }
        // One full grid per stream: the steps work in place in the expanded input
        this->provide_full_grid(/*separate_output=*/false);
        if (!pseudo_current_)
            update_pseudo();
    }
    allocate_work(n_field);
}

//------------------------------------------------------------------------------
// CrysFFT stress fast path
//------------------------------------------------------------------------------
template <typename T>
bool CudaSolverPseudoBase<T>::crysfft_stress_fast_path() const
{
    // The fast path computes only the diagonal sums (V₁₁, V₂₂, V₃₃). Crystal systems
    // eligible for CrysFFT are orthogonal and never optimize angles; with the
    // force-off-diagonal flag (angle optimization active) the standard path,
    // which computes the cross terms, takes over. (use_crysfft_ implies a 3D
    // periodic box with a space group.)
    return use_crysfft_
        && this->cb->is_orthogonal()
        && !this->force_off_diagonal_stress_
        && (crysfft_engine_ == CrysFFTEngine::Pmmm || crysfft_engine_ == CrysFFTEngine::Recursive3m);
}

template <typename T>
void CudaSolverPseudoBase<T>::crysfft_segment_stress(
    int STREAM, double bond_length_sq, const double* d_q_1, const double* d_q_2, double* d_segment_stress)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const cudaStream_t stream = streams[STREAM];
    StreamWork& work = work_[STREAM];
    CudaCrysFFTBase* crysfft = work.crysfft.get();
    double* d_reduced = work.d_stress_sum;

    // q2 on the physical grid, once for all axis groups: an out-of-place apply_multiplier
    // preserves its input (the Pmmm engine's DCT-II and the recursive 3m engine's
    // out-of-place D2Z only read it)
    const double* d_q2_phys = d_q_2;
    if (!crysfft_identity_map_)
    {
        ker_gather<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            work.d_crysfft_in.get(), d_q_2, d_crysfft_phys_to_reduced_, crysfft_physical_size_);
        CUDA_CHECK_LAUNCH();
        d_q2_phys = work.d_crysfft_in;
    }

    // d_segment_stress[g] = Σ_r n_orbit(r) q1(r) [F⁻¹ m_g F q2](r) over the reduced basis, with
    // the multiplier m_g of axis group g built by prepare_stress()
    const std::vector<std::pair<int, double>> keys =
        crysfft_stress_keys(*molecules, *space_group_, bond_length_sq, stress_has_bond_factor());
    int masks[3] = {0, 0, 0};
    for (size_t g = 0; g < keys.size(); ++g)
    {
        const auto multiplier = crysfft_multipliers_.find(keys[g]);
        if (multiplier == crysfft_multipliers_.end())
            throw_with_line_number("The CrysFFT stress multipliers of b^2 = " + std::to_string(bond_length_sq) +
                " were not built (see prepare_stress()).");
        masks[g] = keys[g].first;
        crysfft->apply_multiplier(const_cast<double*>(d_q2_phys), work.d_crysfft_out, multiplier->second, stream);

        // Back to the reduced basis
        const double* d_tmp_reduced = work.d_crysfft_out;
        if (!crysfft_identity_map_)
        {
            ker_gather<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_reduced, work.d_crysfft_out.get(), d_crysfft_reduced_to_phys_, n_basis_);
            CUDA_CHECK_LAUNCH();
            d_tmp_reduced = d_reduced;
        }

        // Orbit-weighted sum with q1
        ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_reduced, d_tmp_reduced, d_q_1, 1.0, n_basis_);
        CUDA_CHECK_LAUNCH();
        ker_multi_weight<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_reduced, d_reduced, d_crysfft_orbit_counts_, n_basis_);
        CUDA_CHECK_LAUNCH();
        size_t bytes = work.temp_storage_bytes;
        CUDA_CHECK(cuda_reduce_sum(work.d_temp_storage.get(), bytes, d_reduced, &d_segment_stress[g], n_basis_, stream));
        CUDA_CHECK_LAUNCH();
    }

    // The multipliers are Cartesian k_d² = (2πm_d/L_d)², while the full-grid path accumulates
    // deformation-vector components v_d² = k_d²/L_d²: divide each sum by L_d² so both paths
    // return the same V_dd, and scale by b²·M_full (FFT normalization convention)
    const std::vector<double> lx = cb->get_lx();
    const int M_full = cb->get_total_grid();
    const double s0 = bond_length_sq * M_full / (lx[0] * lx[0]);
    const double s1 = bond_length_sq * M_full / (lx[1] * lx[1]);
    const double s2 = bond_length_sq * M_full / (lx[2] * lx[2]);
    ker_crysfft_stress_finalize<<<1, 1, 0, stream>>>(
        d_segment_stress, static_cast<int>(keys.size()), masks[0], masks[1], masks[2], s0, s1, s2);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaSolverPseudoBase<T>::prepare_stress()
{
    try
    {
        // The full-grid work when the force-off-diagonal flag has turned the stress to the
        // full grid since the last call (see full_grid_path()), and its Pseudo tables of the
        // cell, with the stress basis from the first stress on
        if (!stress_basis_)
        {
            stress_basis_ = true;
            pseudo_current_ = false;
        }
        provide_work();

        // [real-only feature] Space-group reduced basis and CrysFFT are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same_v<T, double>)
        {
            if (!crysfft_stress_fast_path())
                return;

            // Per stream: the reduced-basis integrand and the CUB scratch of its sum
            for (int i = 0; i < n_streams; i++)
            {
                StreamWork& work = work_[i];
                provide(work.d_stress_sum, n_basis_);
                // the first call only queries the size of the CUB scratch (it touches no array)
                size_t bytes = 0;
                CUDA_CHECK(cuda_reduce_sum(nullptr, bytes, work.d_stress_sum.get(), static_cast<double*>(nullptr), n_basis_, streams[i]));
                provide(work.d_temp_storage, bytes);
                work.temp_storage_bytes = bytes;
            }

            // The multipliers of each block's monomer type, full and (discrete chains)
            // half bonds, one per axis group for all streams: the missing ones are built
            // here, before the segment loop, and update_crysfft_tables() rebuilds them
            // for a new cell
            const std::array<double, 6> cell = crysfft_cell(*cb);
            for (const auto& key : crysfft_multiplier_keys(*molecules, *space_group_, stress_has_bond_factor()))
            {
                if (crysfft_multipliers_.contains(key))
                    continue;
                CudaDeviceBuffer<double> table;
                work_[0].crysfft->build_multiplier(table, cell, key.first, key.second, streams[0]);
                crysfft_multipliers_.emplace(key, std::move(table));
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Stress of one segment (shared by RQM4, RK2 and the discrete chain)
//------------------------------------------------------------------------------
template <typename T>
const double* CudaSolverPseudoBase<T>::get_stress_boltz_bond(
    [[maybe_unused]] const std::string& monomer_type, [[maybe_unused]] bool is_half_bond_length) const
{
    return nullptr;   // continuous chains: the bond factor is already in the propagator
}

template <typename T>
void CudaSolverPseudoBase<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_1, CuDeviceData<T> *d_q_2, CuDeviceData<T> *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M_COMPLEX = this->total_complex_grid();

        const double bond_length_sq = stress_bond_length_sq(*molecules, monomer_type, is_half_bond_length);

        // =====================================================================
        // CrysFFT stress fast path (reduced grid, no full-grid FFT)
        // Mirrors CpuSolverPseudoBase::compute_single_segment_stress.
        // ObliqueZ and all other cases fall through to the standard path.
        // =====================================================================
        // [real-only feature] Space-group reduced basis and CrysFFT are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same_v<T, double>)
        {
            // NOTE: The CrysFFT fast path computes only the diagonal sums
            // (V₁₁, V₂₂, V₃₃). This is fine because crystal systems eligible
            // for CrysFFT are orthogonal and never optimize angles. If the
            // force-off-diagonal flag is set (angle optimization active),
            // fall back to the standard path below — which can compute the
            // cross terms — instead of silently returning zeros for them.
            if (crysfft_stress_fast_path())
            {
                crysfft_segment_stress(STREAM, bond_length_sq, d_q_1, d_q_2, d_segment_stress);
                return;
            }
        }

        // Chain-model-specific Boltzmann bond factor (nullptr for continuous chains).
        // The discrete chain multiplies exp(-k^2 b^2 ds/6) * b^2 into the integrand
        // and weights the basis tables with 1; continuous chains weight them with b^2.
        const double* _d_boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);
        const double basis_factor = (_d_boltz_bond != nullptr) ? 1.0 : bond_length_sq;

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const double* _d_fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _d_fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _d_fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _d_negative_k_idx = pseudo->get_negative_frequency_mapping();

        // The spectra of both propagators. The transforms take the full grid: with a space group
        // (no CrysFFT fast path) each propagator is expanded from the reduced basis into
        // d_q_full_, which its transform reads before the next expansion (both run on this stream)
        StreamWork& work = work_[STREAM];
        double* qk_1 = spectrum(STREAM, 0);
        double* qk_2 = spectrum(STREAM, 1);
        transform_forward(STREAM, expand_to_full_grid(STREAM, d_q_1), qk_1);
        transform_forward(STREAM, expand_to_full_grid(STREAM, d_q_2), qk_2);
        if (is_periodic_)
        {
            const cuDoubleComplex* k_1 = reinterpret_cast<const cuDoubleComplex*>(qk_1);
            const cuDoubleComplex* k_2 = reinterpret_cast<const cuDoubleComplex*>(qk_2);
            // [type-specific algorithm] Real fields store the Hermitian half-spectrum, so the
            // conjugate product is formed directly; complex fields need the -k index map.
            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(work.d_q_multi, k_1, k_2, M_COMPLEX);
            else
            {
                // Gather q2(-k) into a separate buffer: an in-place gather through the
                // k -> -k map (an involution) would race.
                ker_gather<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(work.d_qk_conj.get(), k_2, _d_negative_k_idx, M_COMPLEX);
                CUDA_CHECK_LAUNCH();
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(work.d_q_multi, k_1, work.d_qk_conj, 1.0, M_COMPLEX);
            }
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            // Walls: the coefficients of the cosine and sine series are real for real fields;
            // for complex fields those of the real and imaginary parts come as complex pairs,
            // whose product needs no conjugate (the basis functions are real). The per-mode
            // Parseval weights (2 interior / 1 special modes) are folded into the fourier_basis
            // tables by Pseudo::update_nonperiodic.
            // [type-specific algorithm] The series coefficients are real for real fields and interleaved complex pairs for complex fields.
            if constexpr (std::is_same<T, double>::value)
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(work.d_q_multi, qk_1, qk_2, 1.0, M_COMPLEX);
            else
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(
                    work.d_q_multi, reinterpret_cast<const cuDoubleComplex*>(qk_1),
                    reinterpret_cast<const cuDoubleComplex*>(qk_2), 1.0, M_COMPLEX);
            CUDA_CHECK_LAUNCH();
        }

        // Discrete chains: Boltzmann bond factor exp(-k²·b²·ds/6) and b² (already
        // halved for half-bond steps) into the integrand
        if (_d_boltz_bond != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM]>>>(work.d_q_multi, work.d_q_multi, _d_boltz_bond, bond_length_sq, M_COMPLEX);
            CUDA_CHECK_LAUNCH();
        }

        // fourier_basis arrays contain v⊗v dyad components of the deformation
        // vector v = 2π g⁻¹ m (units 1/L², NOT the Cartesian wavevector):
        // _d_fourier_basis_x = v₁², _d_fourier_basis_xy = v₁v₂, etc.
        // The accumulated sums V_ij = Σ kernel·v_i·v_j are converted to
        // lattice-parameter derivatives in compute_stress().
        // No cross-term corrections needed.

        // For orthogonal boxes (all angles = 90°) the cross-term sums (V₁₂, V₁₃, V₂₃) are
        // not computed: angle optimization is inactive for orthogonal crystal systems, so
        // those slots are never read, and the one pass over k below reads three basis
        // tables instead of six. force_off_diagonal_stress_ (set when angle optimization
        // is active) computes them also at exactly 90°, so that the angle derivatives are
        // available. Walls need an orthogonal cell: the non-periodic tables have no cross terms.
        const bool is_orthogonal = (this->cb->is_orthogonal() && !this->force_off_diagonal_stress_) || !is_periodic_;

        // Dyad components V_c = Σ_k integrand(k)·basis_c(k)·basis_factor, written to
        // d_segment_stress[0..n_comp-1]
        StressBases bases{};
        int n_comp = 0;
        auto add_component = [&](const double* d_basis) { bases.basis[n_comp++] = d_basis; };
        if ( DIM == 3 )
        {
            add_component(_d_fourier_basis_x);   // V₁₁
            add_component(_d_fourier_basis_y);   // V₂₂
            add_component(_d_fourier_basis_z);   // V₃₃

            // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
            if (!is_orthogonal)
            {
                add_component(_d_fourier_basis_xy);   // V₁₂
                add_component(_d_fourier_basis_xz);   // V₁₃
                add_component(_d_fourier_basis_yz);   // V₂₃
            }
        }
        if ( DIM == 2 )
        {
            add_component(_d_fourier_basis_x);   // V₁₁
            add_component(_d_fourier_basis_y);   // V₂₂

            // V₁₂: only compute for non-orthogonal boxes
            if (!is_orthogonal)
                add_component(_d_fourier_basis_xy);
        }
        if ( DIM == 1 )
            add_component(_d_fourier_basis_x);

        // One pass over k for all components, then the block partials in block order
        launch_stress_dyads(d_segment_stress, work.d_stress_partial.get(), work.d_q_multi.get(), bases, n_comp,
                            basis_factor, M_COMPLEX, streams[STREAM]);
        CUDA_CHECK_LAUNCH();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoBase<double>;
template class CudaSolverPseudoBase<std::complex<double>>;
