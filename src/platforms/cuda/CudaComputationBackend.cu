/**
 * @file CudaComputationBackend.cu
 * @brief CUDA implementation of ComputationBackend.
 */

#include <algorithm>
#include <complex>
#include <type_traits>
#include <utility>
#include <cuComplex.h>

#include "CudaComputationBackend.h"
#include "Exception.h"
#include "NumericalMethods.h"
#include "CudaSolverCNADI.h"
#include "CudaSolverPseudoDiscrete.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaSolverPseudoRQM4.h"
#include "CudaSolverRealSpaceDiscrete.h"
#include "CudaTypeDispatch.h"
#include "CudaJunctionProduct.cuh"

namespace {

/// Host scalar to the device scalar type.
template <typename T>
CuDeviceData<T> to_device_scalar(T value)
{
    if constexpr (std::is_same_v<T, double>)
        return value;
    else
        return stdToCuDoubleComplex(value);
}

/// Segment pairs added to a concentration in one launch (passed as kernel parameters)
constexpr int MAX_PAIRS_PER_LAUNCH = 32;

template <typename TD>
struct SegmentProducts
{
    const TD* q_1[MAX_PAIRS_PER_LAUNCH];
    const TD* q_2[MAX_PAIRS_PER_LAUNCH];
    double weight[MAX_PAIRS_PER_LAUNCH];
    int n;
};

// phi += Σ_k weight_k q_1k q_2k in the order of the pairs, one pair after another
// (phi += weight q_1 q_2), but phi is read and written once per launch
__global__ void ker_add_segment_products(double* phi, SegmentProducts<double> p, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        double acc = phi[i];
        for (int k = 0; k < p.n; ++k)
            acc += p.weight[k] * p.q_1[k][i] * p.q_2[k][i];
        phi[i] = acc;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_segment_products(cuDoubleComplex* phi, SegmentProducts<cuDoubleComplex> p, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        cuDoubleComplex acc = phi[i];
        for (int k = 0; k < p.n; ++k)
        {
            const cuDoubleComplex a = p.q_1[k][i];
            const cuDoubleComplex b = p.q_2[k][i];
            const double w = p.weight[k];
            const double tx = w * (a.x * b.x - a.y * b.y);
            const double ty = w * (a.x * b.y + a.y * b.x);
            acc.x += tx;
            acc.y += ty;
        }
        phi[i] = acc;
        i += blockDim.x * gridDim.x;
    }
}

}  // namespace

template <typename T>
CudaComputationBackend<T>::CudaComputationBackend(
    ComputationBox<T>* cb_in, Molecules* molecules,
    const NumericalMethodInfo& method_info,
    SpaceGroup* space_group, int max_n_streams)
    : cb(dynamic_cast<CudaComputationBox<T>*>(cb_in)), n_streams(0), n_basis(0)
{
    // The members own what they hold: if a step below throws, what was allocated is released
    // (the solver before the streams, see the header)
    if (cb == nullptr)
        throw_with_line_number("The CUDA backend requires a CudaComputationBox.");

    n_basis = cb->get_n_basis();

    // The streams, lent to the solver
    n_streams = std::min({this->n_streams_from_env(/*default_n_streams=*/MAX_STREAMS), MAX_STREAMS, max_n_streams});
    cudaStream_t stream_handles[MAX_STREAMS] = {};
    for (int i = 0; i < n_streams; i++)
    {
        CUDA_CHECK(streams[i].create());
        stream_handles[i] = streams[i];
    }

    // The solver of the method, with the space group (nullptr without one): it holds the
    // index maps of the group, expands and reduces around its full-grid steps and allocates
    // the work of its path, so that the computations allocate no solver work
    switch (method_info.id)
    {
    case NumericalMethodId::PS:
        solver = std::make_unique<CudaSolverPseudoDiscrete<T>>(cb, molecules, n_streams, stream_handles, space_group);
        break;
    case NumericalMethodId::RS:
        solver = std::make_unique<CudaSolverRealSpaceDiscrete<T>>(cb, molecules, n_streams, stream_handles, space_group);
        break;
    case NumericalMethodId::RQM4:
        solver = std::make_unique<CudaSolverPseudoRQM4<T>>(cb, molecules, n_streams, stream_handles, space_group);
        break;
    case NumericalMethodId::RK2:
        solver = std::make_unique<CudaSolverPseudoRK2<T>>(cb, molecules, n_streams, stream_handles, space_group);
        break;
    case NumericalMethodId::CNADI2:
        if constexpr (std::is_same_v<T, double>)
            solver = std::make_unique<CudaSolverCNADI>(cb, molecules, n_streams, stream_handles, space_group);
        else
            throw_with_line_number("The cn-adi2 method is only available for real fields.");
        break;
    default:
        throw_with_line_number("The CUDA backend has no solver for the numerical method \"" +
                               std::string(method_info.name) + "\".");
    }

    // Ones for free chain ends
    std::vector<T> unity(n_basis, T(1.0));
    d_q_unity.allocate(n_basis);
    CUDA_CHECK(cudaMemcpy(d_q_unity, unity.data(), sizeof(T)*n_basis, cudaMemcpyHostToDevice));

    // Mask (the box stores it in the reduced basis of its space group)
    if (cb->get_mask() != nullptr)
    {
        d_q_mask.allocate(n_basis);
        CUDA_CHECK(cudaMemcpy(d_q_mask, cb->get_mask(), sizeof(double)*n_basis, cudaMemcpyHostToDevice));
    }

    d_sum.allocate(n_basis);
    // The accumulator of a junction product over several launches of device-resident
    // sources: only a vertex with more arms than one launch takes needs it (checkpoint
    // products allocate it with their staging, allocate_checkpoint())
    int max_degree = 0;
    for (int p = 0; p < molecules->get_n_polymer_types(); ++p)
    {
        std::map<int, int> degree;
        for (const Block& block : molecules->get_polymer(p).get_blocks())
            max_degree = std::max({max_degree, ++degree[block.v], ++degree[block.u]});
    }
    if (max_degree > junction_product::MAX_SOURCES_PER_LAUNCH)
        for (int i = 0; i < n_streams; i++)
            d_junction_state[i].allocate(2*static_cast<size_t>(n_basis));
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        d_segment_stress[i].allocate(6);
        d_block_stress[i].allocate(6);
        h_block_stress[i].allocate(6);
    }
}

template <typename T>
void CudaComputationBackend<T>::synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== Memory ====================

// The propagator and checkpoint arrays are raw handles owned by the
// computations; they are counted like CudaBuffer allocations, so the tests'
// live counters and failure hook cover them as well.
template <typename T>
BackendArray<T>* CudaComputationBackend<T>::allocate()
{
    const size_t bytes = sizeof(T)*n_basis;
    void* d_a = counted_cuda_allocate<CudaDeviceMemory>(bytes);
    const cudaError_t err = cudaMemset(d_a, 0, bytes);
    if (err != cudaSuccess)
    {
        counted_cuda_release<CudaDeviceMemory>(d_a, bytes);
        CUDA_CHECK(err);
    }
    return wrap(static_cast<CuDeviceData<T>*>(d_a));
}

template <typename T>
void CudaComputationBackend<T>::deallocate(BackendArray<T>* a)
{
    counted_cuda_release<CudaDeviceMemory>(raw(a), sizeof(T)*n_basis);
}

template <typename T>
void CudaComputationBackend<T>::copy_from_host(int stream, BackendArray<T>* dst, const T* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), src, sizeof(T)*n_basis, cudaMemcpyHostToDevice, streams[stream]));
}

template <typename T>
void CudaComputationBackend<T>::copy_to_host(T* dst, BackendArray<T>* src)
{
    CUDA_CHECK(cudaMemcpy(dst, raw(src), sizeof(T)*n_basis, cudaMemcpyDeviceToHost));
}

template <typename T>
CheckpointArray<T>* CudaComputationBackend<T>::allocate_checkpoint()
{
    // Products and sums of checkpoints stage them on the device, two per launch
    for (int i = 0; i < n_streams; i++)
    {
        provide(d_junction_stage[i], 2*static_cast<size_t>(n_basis));
        provide(d_junction_state[i], 2*static_cast<size_t>(n_basis));
    }
    void* h_a = counted_cuda_allocate<CudaPinnedMemory>(sizeof(T)*n_basis);
    return reinterpret_cast<CheckpointArray<T>*>(h_a);
}

template <typename T>
void CudaComputationBackend<T>::deallocate_checkpoint(CheckpointArray<T>* a)
{
    counted_cuda_release<CudaPinnedMemory>(raw(a), sizeof(T)*n_basis);
}

template <typename T>
void CudaComputationBackend<T>::store_checkpoint(int stream, CheckpointArray<T>* dst, BackendArray<T>* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), raw(src), sizeof(T)*n_basis, cudaMemcpyDeviceToHost, streams[stream]));
}

template <typename T>
void CudaComputationBackend<T>::load_checkpoint(int stream, BackendArray<T>* dst, CheckpointArray<T>* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), raw(src), sizeof(T)*n_basis, cudaMemcpyHostToDevice, streams[stream]));
}

// ==================== Solver ====================

template <typename T>
void CudaComputationBackend<T>::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

template <typename T>
void CudaComputationBackend<T>::update_dw(const std::map<std::string, BackendArray<T>*>& w)
{
    // The fields that set_fields uploaded
    std::map<std::string, const CuDeviceData<T>*> d_w;
    for (const auto& [monomer_type, w_monomer] : w)
        d_w[monomer_type] = raw(w_monomer);
    solver->update_dw(d_w);
}

template <typename T>
void CudaComputationBackend<T>::set_force_off_diagonal_stress(bool force)
{
    solver->set_force_off_diagonal_stress(force);
}

template <typename T>
void CudaComputationBackend<T>::advance_propagator(
    int stream, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type, int ds_index)
{
    solver->advance_propagator(stream, raw(q_in), raw(q_out), monomer_type, d_q_mask, ds_index);
}

template <typename T>
void CudaComputationBackend<T>::advance_propagator_half_bond_step(
    int stream, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type)
{
    solver->advance_propagator_half_bond_step(stream, raw(q_in), raw(q_out), monomer_type);
}

template <typename T>
BackendArray<T>* CudaComputationBackend<T>::boltzmann_factor(const std::string& monomer_type)
{
    // exp(-w ds) of the global ds in the propagator basis, kept by the solver; the computations
    // only read it
    return wrap(const_cast<CuDeviceData<T>*>(solver->boltzmann_factor(monomer_type)));
}

// ==================== Propagator initial conditions ====================

template <typename T>
void CudaComputationBackend<T>::set_unity(int stream, BackendArray<T>* q)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(q), d_q_unity, sizeof(T)*n_basis,
                                    cudaMemcpyDeviceToDevice, streams[stream]));
}

template <typename T>
void CudaComputationBackend<T>::set_zero(int stream, BackendArray<T>* a)
{
    CUDA_CHECK(cudaMemsetAsync(raw(a), 0, sizeof(T)*n_basis, streams[stream]));
}

template <typename T>
template <typename Source>
void CudaComputationBackend<T>::multiply_sources(
    int stream, BackendArray<T>* q, const std::vector<std::pair<Source*, double>>& sources)
{
    // Device-resident sources: up to MAX_SOURCES_PER_LAUNCH per launch. Checkpoints: two per
    // launch, staged in d_junction_stage
    constexpr bool staged = std::is_same_v<Source, CheckpointArray<T>>;
    const size_t batch_size = staged ? 2 : junction_product::MAX_SOURCES_PER_LAUNCH;
    // A product over several launches needs the accumulator, which the constructor allocates
    // for a vertex of more arms than one launch takes (and allocate_checkpoint() for checkpoints)
    if (sources.size() > batch_size && d_junction_state[stream].empty())
        throw_with_line_number("A junction product of " + std::to_string(sources.size()) +
            " sources has no accumulator: no vertex has that many arms.");
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    for (size_t begin = 0; begin < sources.size(); begin += batch_size)
    {
        junction_product::Sources<CuDeviceData<T>> batch{};
        batch.n = static_cast<int>(std::min(batch_size, sources.size() - begin));
        for (int k = 0; k < batch.n; ++k)
        {
            const auto& [source, power] = sources[begin + k];
            if constexpr (staged)
            {
                CuDeviceData<T>* d_source = d_junction_stage[stream].get() + k*n_basis;
                CUDA_CHECK(cudaMemcpyAsync(d_source, raw(source), sizeof(T)*n_basis,
                                           cudaMemcpyHostToDevice, streams[stream]));
                batch.q[k] = d_source;
            }
            else
                batch.q[k] = raw(source);
            batch.power[k] = power;
        }
        // A product over several launches keeps its accumulator in q and in d_junction_state
        junction_product::ker_junction_product<<<grid_blocks(n_basis, N_THREADS), N_THREADS, 0, streams[stream]>>>(
            raw(q), batch, reinterpret_cast<CuDeviceData<T>*>(d_junction_state[stream].get()),
            begin == 0, begin + batch.n == sources.size(), n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_sum(
    int stream, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, double>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);

    CUDA_CHECK(cudaMemsetAsync(d_q, 0, sizeof(T)*n_basis, streams[stream]));
    for (const auto& [src, n_repeated] : sources)
    {
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[stream]>>>(
            d_q, 1.0, d_q, n_repeated, raw(src), n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_product(
    int stream, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, double>>& sources)
{
    multiply_sources(stream, q, sources);
}

template <typename T>
void CudaComputationBackend<T>::junction_sum(
    int stream, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, double>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);
    // Each checkpoint is staged in d_junction_stage
    CuDeviceData<T>* d_stage = d_junction_stage[stream];

    CUDA_CHECK(cudaMemsetAsync(d_q, 0, sizeof(T)*n_basis, streams[stream]));
    for (const auto& [src, n_repeated] : sources)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_stage, raw(src), sizeof(T)*n_basis,
                                        cudaMemcpyHostToDevice, streams[stream]));
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[stream]>>>(
            d_q, 1.0, d_q, n_repeated, d_stage, n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_product(
    int stream, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, double>>& sources)
{
    multiply_sources(stream, q, sources);
}

template <typename T>
void CudaComputationBackend<T>::apply_mask(int stream, BackendArray<T>* q)
{
    if (d_q_mask.empty())
        return;
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream]>>>(raw(q), raw(q), d_q_mask, 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaComputationBackend<T>::multiply_boltzmann_factor(int stream, BackendArray<T>* q, const std::string& monomer_type)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream]>>>(
        raw(q), raw(q), raw(boltzmann_factor(monomer_type)), 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
}

// ==================== Statistics ====================

template <typename T>
T CudaComputationBackend<T>::inner_product(BackendArray<T>* a, BackendArray<T>* b)
{
    return cb->inner_product_device(raw(a), raw(b));
}

template <typename T>
T CudaComputationBackend<T>::inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w)
{
    return cb->inner_product_inverse_weight_device(raw(a), raw(b), raw(w));
}

template <typename T>
double CudaComputationBackend<T>::absolute_inner_product(
    BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* divisor)
{
    return cb->absolute_inner_product_device(raw(a), raw(b), divisor == nullptr ? nullptr : raw(divisor));
}

template <typename T>
void CudaComputationBackend<T>::add_segment_products(
    int stream, BackendArray<T>* phi, const std::vector<SegmentPair<T>>& pairs)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    // Up to MAX_PAIRS_PER_LAUNCH pairs per launch, in order
    SegmentProducts<CuDeviceData<T>> chunk{};
    for (size_t first = 0; first < pairs.size(); first += MAX_PAIRS_PER_LAUNCH)
    {
        chunk.n = static_cast<int>(std::min<size_t>(MAX_PAIRS_PER_LAUNCH, pairs.size() - first));
        for (int k = 0; k < chunk.n; ++k)
        {
            chunk.q_1[k] = raw(pairs[first + k].q_1);
            chunk.q_2[k] = raw(pairs[first + k].q_2);
            chunk.weight[k] = pairs[first + k].weight;
        }
        ker_add_segment_products<<<N_BLOCKS, N_THREADS, 0, streams[stream]>>>(raw(phi), chunk, n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::divide_by_boltzmann_factor(int stream, BackendArray<T>* phi, BackendArray<T>* boltzmann_factor)
{
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_phi = raw(phi);
    cudaStream_t s = streams[stream];
    if (!d_q_mask.empty())
        ker_divide_masked<<<grid_blocks(n_basis, N_THREADS), N_THREADS, 0, s>>>(d_phi, d_phi, raw(boltzmann_factor), 1.0, d_q_mask, n_basis);
    else
        ker_divide<<<grid_blocks(n_basis, N_THREADS), N_THREADS, 0, s>>>(d_phi, d_phi, raw(boltzmann_factor), 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaComputationBackend<T>::prepare_stress()
{
    solver->prepare_stress();
}

template <typename T>
std::array<T, 6> CudaComputationBackend<T>::block_stress(
    int stream, const std::vector<SegmentPair<T>>& pairs, const std::string& monomer_type)
{
    std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (pairs.empty())
        return stress;

    // All six components, as on the CPU: those the solver does not compute stay zero
    cudaStream_t s = streams[stream];
    CuDeviceData<T>* d_segment = d_segment_stress[stream];
    CuDeviceData<T>* d_block   = d_block_stress[stream];
    CUDA_CHECK(cudaMemsetAsync(d_segment, 0, sizeof(T)*6, s));
    CUDA_CHECK(cudaMemsetAsync(d_block,   0, sizeof(T)*6, s));

    // The pairs are computed on this stream (also the reduce-memory recomputation)
    for (const SegmentPair<T>& pair : pairs)
    {
        // block += weight * segment, summed on the device
        solver->compute_single_segment_stress(stream, raw(pair.q_1), raw(pair.q_2), d_segment,
                                              monomer_type, pair.is_half_bond_length);
        ker_lin_comb<<<1, 32, 0, s>>>(d_block, 1.0, d_block, pair.weight, d_segment, 6);
        CUDA_CHECK_LAUNCH();
    }

    // One copy per block; waiting on this stream only lets the other streams run on
    CUDA_CHECK(cudaMemcpyAsync(h_block_stress[stream], d_block, sizeof(T)*6,
                                    cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    for (int d = 0; d < 6; d++)
        stress[d] = h_block_stress[stream][d];
    return stress;
}

template <typename T>
T CudaComputationBackend<T>::solvent_concentration(
    BackendArray<T>* phi, BackendArray<T>* w, double ds, double volume_fraction)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_phi = raw(phi);

    ker_exp<<<N_BLOCKS, N_THREADS>>>(d_phi, raw(w), 1.0, -ds, n_basis);
    CUDA_CHECK_LAUNCH();

    const T partition = cb->mean_device(d_phi);

    // Keep Q for diagnostics, but an absent species has zero concentration even
    // when its Boltzmann factors cancel (Q = 0) or are non-finite.
    if (volume_fraction == 0.0)
    {
        CUDA_CHECK(cudaMemsetAsync(d_phi, 0, sizeof(T)*n_basis, 0));
        return partition;
    }

    // The same host arithmetic as the CPU backend
    const CuDeviceData<T> norm = to_device_scalar(volume_fraction / partition);
    ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, norm, d_phi, 0.0, d_phi, n_basis);
    CUDA_CHECK_LAUNCH();

    // No solvent inside impenetrable regions (the mask is in the field basis)
    if (!d_q_mask.empty())
    {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_q_mask, 1.0, n_basis);
        CUDA_CHECK_LAUNCH();
    }
    return partition;
}

template <typename T>
void CudaComputationBackend<T>::sum_to_host(
    T* out, const std::vector<BackendArray<T>*>& arrays, const std::vector<T>& weights)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(T)*n_basis));
    for (size_t a = 0; a < arrays.size(); a++)
    {
        ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_sum, to_device_scalar(weights[a]), raw(arrays[a]), 1.0, d_sum, n_basis);
        CUDA_CHECK_LAUNCH();
    }
    CUDA_CHECK(cudaMemcpy(out, d_sum, sizeof(T)*n_basis, cudaMemcpyDeviceToHost));
}

// Explicit template instantiation
template class CudaComputationBackend<double>;
template class CudaComputationBackend<std::complex<double>>;
