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
#include <vector>

#include "CudaDeviceOps.cuh"
#include "CudaSolverRealSpaceDiscrete.h"

namespace
{

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

// The tile of a pass along an axis of n points with `inner` contiguous values per point and a kernel
// of half width H: the widest the shared memory of a block holds; along the contiguous axis a block
// has a thread per point of a short axis. bytes exceeds max_shared_bytes when no tile fits.
struct Tile
{
    int threads, columns, points, bytes;
};

template <typename D>
Tile tile_of(int n, long inner, int H)
{
    Tile tile;
    tile.threads = inner == 1 ? std::min(256, (n + 31) / 32 * 32) : 256;
    tile.columns = inner == 1 ? 1 : 32;
    tile.points = inner == 1 ? tile.threads : 64;
    auto bytes = [&] { return weights_bytes(H) + (tile.points + 2 * H) * tile.columns * static_cast<int>(sizeof(D)); };
    while (bytes() > max_shared_bytes && tile.points > 8)
        tile.points /= 2;
    while (bytes() > max_shared_bytes && tile.columns > 1)
        tile.columns /= 2;
    tile.bytes = bytes();
    return tile;
}

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

// A narrow absorbing axis: out_i = sum_j F_ij in_j with the folded matrix F of the
// axis, built on the host once per box (RealSpaceBond); each output is summed in a
// fixed order.
template <typename D>
__global__ void ker_folded_bond_pass(const D* __restrict__ in, D* __restrict__ out,
    const int n, const long inner, const long count, const double* __restrict__ folded,
    const D* __restrict__ factor, const double* __restrict__ mask)
{
    for (long m = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
         m < count; m += static_cast<long>(blockDim.x) * gridDim.x)
    {
        const int i = static_cast<int>((m / inner) % n);
        const long first = m - i * inner;
        const double* row = folded + static_cast<long>(i) * n;
        D sum = zero_of(D{});
        for (int j = 0; j < n; ++j)
            sum = added(sum, scaled(row[j], in[first + j * inner]));
        out[m] = finished(sum, m, factor, mask);
    }
}
}  // namespace

template <typename T>
CudaSolverRealSpaceDiscrete<T>::CudaSolverRealSpaceDiscrete(
    ComputationBox<T>* cb, Molecules* molecules, int n_streams, cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group, const int* d_reduced_basis_indices, const int* d_full_to_reduced_map)
    : CudaSolver<T>(cb, molecules, n_streams, streams, space_group, d_reduced_basis_indices, d_full_to_reduced_map)
{
    try
    {
        if (molecules->get_model_name() != "discrete")
            throw_with_line_number("The rs method is for discrete chains.");

        const int M = cb->get_total_grid();
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
        // The full grid around the bonds of a reduced-basis run: the input and, as a bond
        // does not run in place, the output
        this->provide_full_grid(/*separate_output=*/true);

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::upload(const RealSpaceBond::Kernel& stencil, const RealSpaceBond::Folded& folded,
                                            DeviceAxis& device)
{
    // A folded axis needs no stencil
    const std::size_t stencil_size = folded.empty() ? stencil.size() : 0;
    if (device.stencil.size() != stencil_size)
        device.stencil.allocate(stencil_size);
    if (stencil_size > 0)
        CUDA_CHECK(cudaMemcpy(device.stencil, stencil.data(), sizeof(double) * stencil_size, cudaMemcpyHostToDevice));
    if (device.folded.size() != folded.size())
        device.folded.allocate(folded.size());
    if (!folded.empty())
        CUDA_CHECK(cudaMemcpy(device.folded, folded.data(), sizeof(double) * folded.size(), cudaMemcpyHostToDevice));
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::update_laplacian_operator()
{
    // Uploads replace live buffers in place. If any allocation/copy fails,
    // neither propagation nor stress may use the mixture of old and new tables.
    refusal_ = "The rs operator update has not completed. Call update_laplacian_operator() successfully before computing again.";
    try
    {
        // The box is orthogonal: the factory refuses a skewed box for rs and holds its angles
        const RealSpaceBond::Bonds bonds = RealSpaceBond::bonds(*molecules, *cb);
        const RealSpaceBond::Ghosts& ghosts = bonds.ghosts;
        const auto nx = cb->get_nx();

        // Every pass must fit the shared memory of a block: the longest kernel of an axis (its halo;
        // the derivatives have the lengths of the bonds) is checked before the solver changes. The box
        // has changed already, so a box that does not fit refuses every bond until one fits again,
        // instead of leaving the kernels of the previous box in use.
        for (int d = 0; d < cb->get_dim(); ++d)
        {
            long inner = 1;
            for (int e = d + 1; e < cb->get_dim(); ++e)
                inner *= nx[e];
            if (tile_of<CuDeviceData<T>>(nx[d], inner, ghosts.halo[d]).bytes > max_shared_bytes)
            {
                refusal_ = "The bond spans " + std::to_string(2 * ghosts.halo[d] + 1) + " grid points along axis " +
                           std::to_string(d) + ", more than the shared memory of a block holds; use a coarser grid.";
                throw_with_line_number(refusal_);
            }
        }

        // The kernel of each axis of the box; a derivative holds that of its own axis only
        const int dim = cb->get_dim();
        for (const auto& [ds_idx, kernels] : bonds.full)
            for (const auto& [monomer_type, k] : kernels)
                for (int d = 0; d < dim; ++d)
                    upload(k[d], k.folded[d], d_bond[ds_idx][monomer_type][d]);
        for (const auto& [monomer_type, k] : bonds.half)
        {
            for (int d = 0; d < dim; ++d)
            {
                const RealSpaceBond::Derivative& full = bonds.full_dlength.at(monomer_type)[d];
                const RealSpaceBond::Derivative& half = bonds.half_dlength.at(monomer_type)[d];
                upload(k[d], k.folded[d], d_bond_half[monomer_type][d]);
                upload(full.stencil, full.folded, d_bond_dlength[monomer_type][d]);
                upload(half.stencil, half.folded, d_bond_half_dlength[monomer_type][d]);
            }
        }
        halo = ghosts.halo;
        for (int d = 0; d < cb->get_dim(); ++d)
        {
            const size_t count = ghosts.index[d].size();
            if (d_ghost_index[d].size() != count)
                d_ghost_index[d].allocate(count);
            // A previous attempt may have allocated the index but failed while
            // replacing the sign buffer. Each buffer must be retried separately.
            if (d_ghost_sign[d].size() != count)
                d_ghost_sign[d].allocate(count);
            CUDA_CHECK(cudaMemcpy(d_ghost_index[d], ghosts.index[d].data(), sizeof(int) * count, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ghost_sign[d], ghosts.sign[d].data(), sizeof(double) * count, cudaMemcpyHostToDevice));
        }
        refusal_.clear();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::update_dw(const std::map<std::string, const CuDeviceData<T>*>& d_w)
{
    try
    {
        // Even a one-bead chain (with no convolution) needs a completed update.
        if (!refusal_.empty())
            throw_with_line_number(refusal_);
        // exp(-w ds) on the full grid: one full segment per bond
        this->compute_exp_dw(d_w, /*reduced_basis_tables=*/false, false, 1.0, 0.0);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::convolve(
    int STREAM, const CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out, const DeviceKernels& kernels,
    const CuDeviceData<T>* d_factor, const double* d_mask, int derivative_axis, const DeviceAxis* derivative)
{
    using D = CuDeviceData<T>;
    if (!refusal_.empty())
        throw_with_line_number(refusal_);
    const std::vector<int> nx = cb->get_nx();
    const int dim = static_cast<int>(nx.size());
    const long M = cb->get_total_grid();
    cudaStream_t stream = streams[STREAM];

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
        const DeviceAxis& kernel = axis == derivative_axis ? *derivative : kernels[axis];
        if (!kernel.folded.empty())
        {
            ker_folded_bond_pass<<<grid_blocks(M, 256), 256, 0, stream>>>(src, dst, n, inner, M,
                kernel.folded.get(), factor, mask);
        }
        else
        {
            const double* g = kernel.stencil.get();
            const int H = static_cast<int>(kernel.stencil.size()) - 1;
            const int* index = d_ghost_index[axis].get() + halo[axis];
            const double* sign = d_ghost_sign[axis].get() + halo[axis];
            // The tile fits: update_laplacian_operator() refuses kernels that do not.
            const Tile tile = tile_of<D>(n, inner, H);
            const long tiles = outer * ((n + tile.points - 1) / tile.points) * ((inner + tile.columns - 1) / tile.columns);
            ker_bond_pass<<<grid_blocks(tiles * tile.threads, tile.threads), dim3(tile.columns, tile.threads / tile.columns),
                            tile.bytes, stream>>>(src, dst, n, inner, outer, g, H, index, sign, tile.points, factor, mask);
        }
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
        convolve(STREAM, this->expand_to_full_grid(STREAM, d_q_in), d_q_full_out_[STREAM], kernels, exp_dw, nullptr);
        this->store_full_grid(STREAM, d_q_full_out_[STREAM], d_q_out, d_q_mask);
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
        convolve(STREAM, this->expand_to_full_grid(STREAM, d_q_in), d_q_full_out_[STREAM],
                 d_bond_half.at(monomer_type), nullptr, nullptr);
        this->store_full_grid(STREAM, d_q_full_out_[STREAM], d_q_out, nullptr);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverRealSpaceDiscrete<T>::prepare_stress()
{
    if (!refusal_.empty())
        throw_with_line_number(refusal_);
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
    const int STREAM, CuDeviceData<T>* d_q_1, CuDeviceData<T>* d_q_2, CuDeviceData<T>* d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        // The cell stays orthogonal (the factory holds it so), so there are no shear components
        if (this->force_off_diagonal_stress_)
            throw_with_line_number("The rs method keeps the cell orthogonal and computes no shear stress.");

        // V_aa = M (3 / ds) / L_a sum_r q_1 (dG/dL_a q_2), as in CpuSolverRealSpaceDiscrete
        const long M = cb->get_total_grid();
        const double ds = molecules->get_contour_length_mapping().get_ds_from_index(0);
        // The bond of the global ds (index 0), whose kernel along axis a is replaced by its derivative
        const DeviceKernels& kernels = is_half_bond_length ? d_bond_half.at(monomer_type) : d_bond.at(0).at(monomer_type);
        const auto& derivatives = (is_half_bond_length ? d_bond_half_dlength : d_bond_dlength).at(monomer_type);
        cudaStream_t stream = streams[STREAM];
        // The pair on the full grid, expanded from the reduced basis with a space group
        const CuDeviceData<T>* q_1 = this->expand_to_full_grid(STREAM, d_q_1, d_q_full_out_[STREAM]);
        const CuDeviceData<T>* q_2 = this->expand_to_full_grid(STREAM, d_q_2);
        for (int a = 0; a < cb->get_dim(); ++a)
        {
            convolve(STREAM, q_2, d_work_stress[STREAM], kernels, nullptr, nullptr, a, &derivatives[a]);
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
