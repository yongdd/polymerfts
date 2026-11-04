/**
 * @file CudaSolverRealSpaceDiscrete.cu
 * @brief CUDA real-space solver for discrete chain propagators (numerical method "rs").
 *
 * One bond is one stencil pass per axis,
 *
 *     out_i = g_0 in_i + sum_{d=1..H} g_d (s_{i-d} in_{j(i-d)} + s_{i+d} in_{j(i+d)}),
 *
 * where j and s are the source and sign of a point in the ghost tables.
 *
 * **Template Instantiations:**
 *
 * - CudaSolverRealSpaceDiscrete<double>: Real fields
 * - CudaSolverRealSpaceDiscrete<std::complex<double>>: Complex fields
 */

#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

#include "CudaSolverRealSpaceDiscrete.h"
#include "SpaceGroup.h"

namespace
{
__device__ inline double scaled(double a, double x) { return a * x; }
__device__ inline cuDoubleComplex scaled(double a, cuDoubleComplex x) { return make_cuDoubleComplex(a * x.x, a * x.y); }
__device__ inline double added(double x, double y) { return x + y; }
__device__ inline cuDoubleComplex added(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x + y.x, x.y + y.y); }
__device__ inline double multiplied(double x, double y) { return x * y; }
__device__ inline cuDoubleComplex multiplied(cuDoubleComplex x, cuDoubleComplex y) { return cuCmul(x, y); }

// The last pass of a bond multiplies by exp(-w ds) and the mask (null when absent)
template <typename D>
__device__ inline D finished(D sum, long m, const D* __restrict__ factor, const double* __restrict__ mask)
{
    if (factor != nullptr)
        sum = multiplied(sum, factor[m]);
    if (mask != nullptr)
        sum = scaled(mask[m], sum);
    return sum;
}

// Sum of a b over the grid in a fixed order: partial sums of the blocks, then their sum times factor
constexpr int reduce_threads = 256;
constexpr int reduce_blocks = 256;
__device__ inline double zero_of(double) { return 0.0; }
__device__ inline cuDoubleComplex zero_of(cuDoubleComplex) { return make_cuDoubleComplex(0.0, 0.0); }

template <typename D>
__global__ void ker_dot_partial(const D* __restrict__ a, const D* __restrict__ b, D* __restrict__ partial, const long n)
{
    __shared__ D cache[reduce_threads];
    D sum = zero_of(D{});
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long>(blockDim.x) * gridDim.x)
        sum = added(sum, multiplied(a[i], b[i]));
    cache[threadIdx.x] = sum;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2)
    {
        if (threadIdx.x < half)
            cache[threadIdx.x] = added(cache[threadIdx.x], cache[threadIdx.x + half]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = cache[0];
}

template <typename D>
__global__ void ker_sum_partial(const D* __restrict__ partial, const int count, const double factor, D* __restrict__ out)
{
    D sum = zero_of(D{});
    for (int i = 0; i < count; ++i)
        sum = added(sum, partial[i]);
    *out = scaled(factor, sum);
}

// A tile holds the weights, then the values from a 16-byte boundary
constexpr int max_shared_bytes = 40 * 1024;
__host__ __device__ inline int weights_bytes(int H) { return ((H + 1) * static_cast<int>(sizeof(double)) + 15) / 16 * 16; }

// One stencil pass along an axis of n points with `inner` contiguous values per point. A block of
// columns x rows threads copies `points` points of the axis with their halos, signed by the ghost tables,
// for `columns` consecutive inner values (coalesced) into shared memory once, and each thread sums the
// stencils of points of one inner value from there. Along the contiguous axis a tile has one column.
// index and sign are centred on point 0 of the axis.
template <typename D>
__global__ void ker_bond_pass(const D* __restrict__ in, D* __restrict__ out, const int n, const long inner,
                              const long outer, const double* __restrict__ g, const int H,
                              const int* __restrict__ index, const double* __restrict__ sign, const int points,
                              const D* __restrict__ factor, const double* __restrict__ mask)
{
    extern __shared__ __align__(16) unsigned char rs_shared[];
    double* weights = reinterpret_cast<double*>(rs_shared);
    D* tile = reinterpret_cast<D*>(rs_shared + weights_bytes(H));
    const int columns = blockDim.x;
    const int rows = blockDim.y;
    const int column = threadIdx.x;
    for (int k = threadIdx.y * columns + column; k <= H; k += columns * rows)
        weights[k] = g[k];

    const long tiles_i = (n + points - 1) / points;
    const long tiles_j = (inner + columns - 1) / columns;
    for (long b = blockIdx.x; b < outer * tiles_i * tiles_j; b += gridDim.x)
    {
        const long j = (b % tiles_j) * columns + column;
        const int start = static_cast<int>((b / tiles_j) % tiles_i) * points;
        const long first = (b / (tiles_j * tiles_i)) * n;   // first point of this block of the axis
        const D* a = in + first * inner + j;
        __syncthreads();
        if (j < inner)
        {
            for (int r = threadIdx.y; r < points + 2 * H && start - H + r < n + H; r += rows)
                tile[r * columns + column] = scaled(sign[start - H + r], a[index[start - H + r] * inner]);
        }
        __syncthreads();
        if (j < inner)
        {
            for (int r = threadIdx.y; r < points && start + r < n; r += rows)
            {
                const D* c = tile + (r + H) * columns + column;
                D sum = scaled(weights[0], c[0]);
                for (int d = 1; d <= H; ++d)
                    sum = added(sum, scaled(weights[d], added(c[-d * columns], c[d * columns])));
                const long m = (first + start + r) * inner + j;
                out[m] = finished(sum, m, factor, mask);
            }
        }
    }
}
}  // namespace

template <typename T>
CudaSolverRealSpaceDiscrete<T>::CudaSolverRealSpaceDiscrete(
    ComputationBox<T>* cb, Molecules* molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
    : cb(cb), molecules(molecules), n_streams(n_streams)
{
    try
    {
        if (molecules->get_model_name() != "discrete")
            throw_with_line_number("The rs method is for discrete chains.");
        if (!cb->is_orthogonal())
            throw_with_line_number("The rs method requires an orthogonal computation box.");

        for (int i = 0; i < n_streams; ++i)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        const int M = cb->get_total_grid();
        n_basis_ = M;
        molecules->finalize_contour_length_mapping();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
            for (const auto& item : molecules->get_bond_lengths())
                this->d_exp_dw[ds_idx][item.first].allocate(M);
        for (int i = 0; i < n_streams; ++i)
        {
            d_work_a[i].allocate(M);
            d_work_b[i].allocate(M);
        }

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::set_space_group(
    SpaceGroup* sg, int* d_reduced_basis_indices, int* d_full_to_reduced_map, int n_basis)
{
    if (sg != nullptr && !std::is_same_v<T, double>)
        throw_with_line_number("Space groups need real fields.");
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = sg != nullptr ? n_basis : cb->get_total_grid();
    if (sg != nullptr)
    {
        for (int i = 0; i < n_streams; ++i)
        {
            if (d_q_full_in[i].empty())
                d_q_full_in[i].allocate(cb->get_total_grid());
            if (d_q_full_out[i].empty())
                d_q_full_out[i].allocate(cb->get_total_grid());
        }
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::upload(const RealSpaceBond::Kernels& kernels, DeviceKernels& device)
{
    std::vector<double> weights;
    for (int d = 0; d < 3; ++d)
    {
        device.offset[d] = static_cast<int>(weights.size());
        device.half[d] = static_cast<int>(kernels[d].size()) - 1;
        weights.insert(weights.end(), kernels[d].begin(), kernels[d].end());
    }
    if (device.weights.size() != weights.size())
        device.weights.allocate(weights.size());
    CUDA_CHECK(cudaMemcpy(device.weights, weights.data(), sizeof(double) * weights.size(), cudaMemcpyHostToDevice));
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::update_laplacian_operator()
{
    try
    {
        if (!cb->is_orthogonal())
            throw_with_line_number("The rs method requires an orthogonal computation box.");

        // A bond of contour ds has the variance b^2 ds / 3 along each axis, a half bond half of it
        const std::vector<double> dx = cb->get_dx();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        const double ds = mapping.get_ds_from_index(0);
        std::map<int, std::map<std::string, RealSpaceBond::Kernels>> bond;
        std::map<std::string, RealSpaceBond::Kernels> bond_half;
        std::vector<const RealSpaceBond::Kernels*> all;
        for (const auto& [monomer_type, b] : molecules->get_bond_lengths())
        {
            for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
            {
                bond[ds_idx][monomer_type] = RealSpaceBond::gaussian(b * b * mapping.get_ds_from_index(ds_idx) / 3.0, dx);
                all.push_back(&bond[ds_idx][monomer_type]);
                upload(bond[ds_idx][monomer_type], d_bond[ds_idx][monomer_type]);
            }
            bond_half[monomer_type] = RealSpaceBond::gaussian(b * b * ds / 6.0, dx);
            all.push_back(&bond_half[monomer_type]);
            upload(bond_half[monomer_type], d_bond_half[monomer_type]);
            const std::array<RealSpaceBond::Kernels, 3> dlength = RealSpaceBond::length_derivatives(b * b * ds / 3.0, dx, cb->get_lx());
            const std::array<RealSpaceBond::Kernels, 3> half_dlength = RealSpaceBond::length_derivatives(b * b * ds / 6.0, dx, cb->get_lx());
            for (int a = 0; a < 3; ++a)
            {
                upload(dlength[a], d_bond_dlength[monomer_type][a]);
                upload(half_dlength[a], d_bond_half_dlength[monomer_type][a]);
            }
        }

        const RealSpaceBond::Ghosts ghosts = RealSpaceBond::ghosts(cb->get_nx(), cb->get_boundary_conditions(), all);
        halo = ghosts.halo;
        for (int d = 0; d < cb->get_dim(); ++d)
        {
            const size_t count = ghosts.index[d].size();
            if (d_ghost_index[d].size() != count)
            {
                d_ghost_index[d].allocate(count);
                d_ghost_sign[d].allocate(count);
            }
            CUDA_CHECK(cudaMemcpy(d_ghost_index[d], ghosts.index[d].data(), sizeof(int) * count, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ghost_sign[d], ghosts.sign[d].data(), sizeof(double) * count, cudaMemcpyHostToDevice));
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    try
    {
        if (device != "gpu" && device != "cpu")
            throw_with_line_number("Invalid device \"" + device + "\".");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        const int n_unique_ds = mapping.get_n_unique_ds();

        for (const auto& [monomer_type, w] : w_input)
        {
            for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
                if (!this->d_exp_dw[ds_idx].contains(monomer_type))
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

            // The field on the device: the input itself, or a copy in the table of the last ds,
            // which is computed last (in place)
            // With a space group the field comes in its reduced basis
            const CuDeviceData<T>* d_w = reinterpret_cast<const CuDeviceData<T>*>(w);
            CuDeviceData<T>* d_stage = this->d_exp_dw.at(n_unique_ds - 1).at(monomer_type);
            if (device == "cpu")
            {
                const T* w_full = w;
                std::vector<T> w_host;
                if constexpr (std::is_same_v<T, double>)
                {
                    if (space_group_ != nullptr)
                    {
                        w_host.resize(M);
                        space_group_->from_reduced_basis(w, w_host.data(), 1);
                        w_full = w_host.data();
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_stage, w_full, sizeof(T) * M, cudaMemcpyHostToDevice));
                d_w = d_stage;
            }
            else if (space_group_ != nullptr)
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_stage, d_w, d_full_to_reduced_map_, M);
                CUDA_CHECK_LAUNCH();
                d_w = d_stage;
            }
            // exp(-w ds): one full segment per bond
            for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
            {
                ker_exp<<<N_BLOCKS, N_THREADS>>>(this->d_exp_dw.at(ds_idx).at(monomer_type).get(), d_w, 1.0,
                                                 -mapping.get_ds_from_index(ds_idx), M);
                CUDA_CHECK_LAUNCH();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::convolve(
    int STREAM, const CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out, const DeviceKernels& kernels,
    const CuDeviceData<T>* d_factor, const double* d_mask)
{
    using D = CuDeviceData<T>;
    const std::vector<int> nx = cb->get_nx();
    const int dim = static_cast<int>(nx.size());
    const long M = cb->get_total_grid();
    cudaStream_t stream = streams[STREAM][0];

    // The contiguous (last) axis first; the first axis writes q_out and multiplies by the factor and mask
    const D* src = d_q_in;
    if (d_q_in == d_q_out)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_work_b[STREAM], d_q_in, sizeof(T) * M, cudaMemcpyDeviceToDevice, stream));
        src = d_work_b[STREAM];
    }
    D* dst = d_work_a[STREAM];
    for (int axis = dim - 1; axis >= 0; --axis)
    {
        const int n = nx[axis];
        long inner = 1;
        for (int d = axis + 1; d < dim; ++d)
            inner *= nx[d];
        const long outer = M / (inner * n);
        const D* factor = nullptr;
        const double* mask = nullptr;
        if (axis == 0)
        {
            dst = d_q_out;
            factor = d_factor;
            mask = d_mask;
        }
        const double* g = kernels.weights.get() + kernels.offset[axis];
        const int H = kernels.half[axis];
        const int* index = d_ghost_index[axis].get() + halo[axis];
        const double* sign = d_ghost_sign[axis].get() + halo[axis];

        // The widest tile the shared memory of a block holds with this kernel; along the contiguous
        // axis a block has a thread per point of a short axis
        const int threads = inner == 1 ? std::min(256, (n + 31) / 32 * 32) : 256;
        int columns = inner == 1 ? 1 : 32;
        int points = inner == 1 ? threads : 64;
        auto bytes = [&] { return weights_bytes(H) + (points + 2 * H) * columns * static_cast<int>(sizeof(D)); };
        while (bytes() > max_shared_bytes && points > 8)
            points /= 2;
        while (bytes() > max_shared_bytes && columns > 1)
            columns /= 2;
        if (bytes() > max_shared_bytes)
            throw_with_line_number("The bond spans more grid points than the shared memory of a block holds; "
                                   "use a coarser grid.");
        const long tiles = outer * ((n + points - 1) / points) * ((inner + columns - 1) / columns);
        ker_bond_pass<<<grid_blocks(tiles * threads, threads), dim3(columns, threads / columns), bytes(), stream>>>(
            src, dst, n, inner, outer, g, H, index, sign, points, factor, mask);
        CUDA_CHECK_LAUNCH();
        src = dst;
        dst = (dst == d_work_a[STREAM].get()) ? d_work_b[STREAM].get() : d_work_a[STREAM].get();
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::advance_propagator(
    const int STREAM, CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out, std::string monomer_type,
    double* d_q_mask, int ds_index)
{
    try
    {
        const auto& kernels = d_bond.at(ds_index).at(monomer_type);
        const CuDeviceData<T>* exp_dw = this->d_exp_dw.at(ds_index).at(monomer_type).get();
        if (space_group_ == nullptr)
        {
            convolve(STREAM, d_q_in, d_q_out, kernels, exp_dw, d_q_mask);
            return;
        }
        // From and back to the reduced basis, where the mask is
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        cudaStream_t stream = streams[STREAM][0];
        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_full_in[STREAM], d_q_in, d_full_to_reduced_map_, M);
        CUDA_CHECK_LAUNCH();
        convolve(STREAM, d_q_full_in[STREAM], d_q_full_out[STREAM], kernels, exp_dw, nullptr);
        ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_full_out[STREAM], d_reduced_basis_indices_, n_basis_);
        CUDA_CHECK_LAUNCH();
        if (d_q_mask != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_out, d_q_mask, 1.0, n_basis_);
            CUDA_CHECK_LAUNCH();
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::advance_propagator_half_bond_step(
    const int STREAM, CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out, std::string monomer_type)
{
    try
    {
        if (space_group_ == nullptr)
        {
            convolve(STREAM, d_q_in, d_q_out, d_bond_half.at(monomer_type), nullptr, nullptr);
            return;
        }
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        cudaStream_t stream = streams[STREAM][0];
        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_full_in[STREAM], d_q_in, d_full_to_reduced_map_, M);
        CUDA_CHECK_LAUNCH();
        convolve(STREAM, d_q_full_in[STREAM], d_q_full_out[STREAM], d_bond_half.at(monomer_type), nullptr, nullptr);
        ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_out, d_q_full_out[STREAM], d_reduced_basis_indices_, n_basis_);
        CUDA_CHECK_LAUNCH();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::prepare_stress()
{
    const int M = cb->get_total_grid();
    for (int i = 0; i < n_streams; ++i)
    {
        if (d_work_stress[i].empty())
            d_work_stress[i].allocate(M);
        if (d_partial_sums[i].empty())
            d_partial_sums[i].allocate(reduce_blocks);
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::compute_single_segment_stress(
    const int STREAM, CuDeviceData<T>* d_q_pair, CuDeviceData<T>* d_segment_stress, std::string monomer_type,
    bool is_half_bond_length)
{
    try
    {
        // The cell stays orthogonal (the factory holds it so), so there are no shear components
        if (this->force_off_diagonal_stress_)
            throw_with_line_number("The rs method keeps the cell orthogonal and computes no shear stress.");

        // V_aa = M (3 / ds) / L_a sum_r q_1 (dG/dL_a q_2), as in CpuSolverRealSpaceDiscrete
        const long M = cb->get_total_grid();
        const double ds = molecules->get_contour_length_mapping().get_ds_from_index(0);
        const auto& kernels = (is_half_bond_length ? d_bond_half_dlength : d_bond_dlength).at(monomer_type);
        cudaStream_t stream = streams[STREAM][0];
        // The pair is [q_1 | q_2] in the basis of the propagators; with a space group on the full grid
        const CuDeviceData<T>* q_1 = d_q_pair;
        const CuDeviceData<T>* q_2 = d_q_pair + n_basis_;
        if (space_group_ != nullptr)
        {
            const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
            const int N_THREADS = CudaCommon::get_instance().get_n_threads();
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_full_out[STREAM], q_1, d_full_to_reduced_map_, M);
            CUDA_CHECK_LAUNCH();
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_q_full_in[STREAM], q_2, d_full_to_reduced_map_, M);
            CUDA_CHECK_LAUNCH();
            q_1 = d_q_full_out[STREAM];
            q_2 = d_q_full_in[STREAM];
        }
        for (int a = 0; a < cb->get_dim(); ++a)
        {
            convolve(STREAM, q_2, d_work_stress[STREAM], kernels[a], nullptr, nullptr);
            ker_dot_partial<<<reduce_blocks, reduce_threads, 0, stream>>>(
                q_1, d_work_stress[STREAM].get(), d_partial_sums[STREAM].get(), M);
            CUDA_CHECK_LAUNCH();
            ker_sum_partial<<<1, 1, 0, stream>>>(d_partial_sums[STREAM].get(), reduce_blocks,
                                                 M * 3.0 / (ds * cb->get_lx(a)), d_segment_stress + a);
            CUDA_CHECK_LAUNCH();
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template class CudaSolverRealSpaceDiscrete<double>;
template class CudaSolverRealSpaceDiscrete<std::complex<double>>;
