/**
 * @file CudaSolverCNADI.cu
 * @brief CUDA solver using CN-ADI method.
 *
 * Implements propagator advancement for continuous chains using finite
 * differences instead of FFT. Supports non-periodic boundary conditions
 * (reflecting, absorbing) and uses ADI splitting with GPU-accelerated
 * tridiagonal solvers.
 *
 * **CN-ADI (Crank-Nicolson ADI):**
 *
 * The modified diffusion equation is split dimensionally: the explicit half of
 * a step gives the right-hand side of the first sweep, and one tridiagonal solve
 * follows per axis (x, y, z); each sweep after the first corrects the right-hand
 * side with the result of the sweep before it.
 *
 * **Tridiagonal Solvers:**
 *
 * The matrices do not change from step to step (the field is split off), so
 * factorize_tridiagonal() computes the Thomas factors of each axis once, with the
 * Sherman-Morrison vector of periodic axes, and a solve is two sweeps over them.
 * - tridiagonal_lines_shared(): contiguous lines, a warp of them per block, staged in
 *   shared memory so that both the reads and the writes coalesce
 * - tridiagonal_lines(): the other axes, whose neighbouring values belong to neighbouring
 *   threads, one line per thread (and lines too long for the shared memory)
 *
 * **Boundary Conditions:**
 *
 * - PERIODIC: Cyclic boundary with Sherman-Morrison correction
 * - REFLECTING: Zero flux (Neumann) boundary
 * - ABSORBING: Zero value (Dirichlet) boundary
 *
 * **Note:** Stress computation not yet supported for CN-ADI method.
 *
 * @see FiniteDifference for matrix construction
 * @see ComputationContinuous for integration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <array>

#include "CudaSolverCNADI.h"
#include "CudaTridiagonalLines.cuh"

namespace {

// The line solvers below give one thread to each grid line, so a launch sized
// for whole grids (CudaCommon blocks x threads) would leave most blocks idle:
// a 64^3 grid has 4096 lines per direction. Launch one 64-thread block per 64 lines.
constexpr int LINE_THREADS = 64;
inline int line_blocks(int n_lines) { return (n_lines + LINE_THREADS - 1)/LINE_THREADS; }

/// An axis of the row-major grid as the kernels read it (passed by value): the diagonals
/// of its matrix, its points and the distance in memory between neighbouring points
struct AdiAxis
{
    const double* l;   ///< Lower diagonal
    const double* d;   ///< Main diagonal
    const double* h;   ///< Upper diagonal
    int n;             ///< Points on the axis
    int stride;        ///< Distance in memory of neighbouring points
    bool periodic;     ///< Periodic axis
};

/// Index on the axis of grid point p
__device__ __forceinline__ int axis_index(const AdiAxis& a, int p)
{
    return (p / a.stride) % a.n;
}

/// The grid point next to p (index i on the axis) one step below (step -1) or above (+1)
/// along the axis: wrapped around a periodic axis, p itself beyond a wall (the wall rows of
/// the matrix hold the boundary condition)
__device__ __forceinline__ int neighbour(const AdiAxis& a, int p, int i, int step)
{
    int j = i + step;
    if (j < 0)
        j = a.periodic ? a.n - 1 : 0;
    else if (j >= a.n)
        j = a.periodic ? 0 : a.n - 1;
    return p + (j - i) * a.stride;
}

// Each kernel below finds the neighbours of a point before it evaluates the point's
// expression. Where nvcc contracts a product into an FMA depends on the code between the
// product and its sum, and so does the last bit of the result: index code interleaved with
// the terms changes the bits.

// The explicit half of a step, the right-hand side of the first sweep (along x), in 1D
__global__ void adi_rhs_1d(AdiAxis x, double *d_q_out, const double *d_q_in, const int M)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += blockDim.x * gridDim.x)
    {
        const int i = n;
        const int im = neighbour(x, n, i, -1), ip = neighbour(x, n, i, +1);
        d_q_out[n] = (2.0-x.d[i])*d_q_in[n] - x.l[i]*d_q_in[im] - x.h[i]*d_q_in[ip];
    }
}

// The explicit half of a step in 2D
__global__ void adi_rhs_2d(AdiAxis x, AdiAxis y, double *d_q_out, const double *d_q_in, const int M)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += blockDim.x * gridDim.x)
    {
        const int i = n / x.stride, j = n % y.n;   // y is contiguous
        const int im = neighbour(x, n, i, -1), ip = neighbour(x, n, i, +1);
        const int jm = neighbour(y, n, j, -1), jp = neighbour(y, n, j, +1);
        d_q_out[n] = 2.0*((2.0-0.5*x.d[i]-y.d[j])*d_q_in[n] - y.l[j]*d_q_in[jm] - y.h[j]*d_q_in[jp])
                   - x.l[i]*d_q_in[im] - x.h[i]*d_q_in[ip];
    }
}

// The explicit half of a step in 3D
__global__ void adi_rhs_3d(AdiAxis x, AdiAxis y, AdiAxis z, double *d_q_out, const double *d_q_in, const int M)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += blockDim.x * gridDim.x)
    {
        const int i = n / x.stride, j = (n / y.stride) % y.n, k = n % z.n;   // z is contiguous
        const int im = neighbour(x, n, i, -1), ip = neighbour(x, n, i, +1);
        const int jm = neighbour(y, n, j, -1), jp = neighbour(y, n, j, +1);
        const int km = neighbour(z, n, k, -1), kp = neighbour(z, n, k, +1);
        d_q_out[n] = 2.0*((3.0-0.5*x.d[i]-y.d[j]-z.d[k])*d_q_in[n]
                - z.l[k]*d_q_in[km] - z.h[k]*d_q_in[kp] - y.l[j]*d_q_in[jm] - y.h[j]*d_q_in[jp])
                - x.l[i]*d_q_in[im] - x.h[i]*d_q_in[ip];
    }
}

// The right-hand side of the sweep along an axis after the first: the result of the sweep
// before it (d_q_star) corrected by the matrix of the axis applied to the step's input
__global__ void adi_correction(AdiAxis a, double *d_q_out, const double *d_q_star, const double *d_q_in, const int M)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += blockDim.x * gridDim.x)
    {
        const int i = axis_index(a, n);
        const int im = neighbour(a, n, i, -1), ip = neighbour(a, n, i, +1);
        d_q_out[n] = d_q_star[n] + (a.d[i]-1.0)*d_q_in[n] + a.l[i]*d_q_in[im] + a.h[i]*d_q_in[ip];
    }
}

}  // namespace

CudaSolverCNADI::CudaSolverCNADI(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS],
    SpaceGroup* space_group)
    : CudaSolver<double>(cb, molecules, n_streams, streams, space_group)
{
    if(molecules->get_model_name() != "continuous")
        throw_with_line_number("Real-space method only support 'continuous' chain model.");

    if(!cb->is_orthogonal())
        throw_with_line_number("Real-space method only supports orthogonal unit cells. "
                               "Use pseudo-spectral method (chain_model='continuous') for non-orthogonal systems.");

    // The tridiagonal solvers need at least two points on every axis
    for (int d = 0; d < cb->get_dim(); d++)
        if (cb->get_nx(d) < 2)
            throw_with_line_number("Real-space method needs at least 2 grid points on every axis, got nx["
                                   + std::to_string(d) + "] = " + std::to_string(cb->get_nx(d)) + ".");

    const int M = cb->get_total_grid();
    dim = cb->get_dim();

    // The axes of the row-major grid (the last one is contiguous)
    const std::vector<BoundaryCondition> bc = cb->get_boundary_conditions();
    int stride = 1;
    for (int a = dim - 1; a >= 0; --a)
    {
        n_axis_[a] = cb->get_nx(a);
        stride_[a] = stride;
        periodic_[a] = bc[2*a] == BoundaryCondition::PERIODIC;
        stride *= n_axis_[a];
    }

    // Ensure ContourLengthMapping is finalized
    molecules->finalize_contour_length_mapping();
    const int n_unique_ds = molecules->get_contour_length_mapping().get_n_unique_ds();

    // exp(-w ds/2) and the matrices of the axes for each ds_index and monomer type
    // (map slots are owning buffers: a failed allocation leaves the slot empty)
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        for(const auto& item: molecules->get_bond_lengths())
        {
            d_exp_dw[ds_idx][item.first].allocate(M);
            std::array<AxisMatrix, 3>& matrices = matrices_[ds_idx][item.first];
            for (int a = 0; a < dim; ++a)
            {
                matrices[a].l.allocate(n_axis_[a]);
                matrices[a].d.allocate(n_axis_[a]);
                matrices[a].h.allocate(n_axis_[a]);
                matrices[a].factors.allocate(4*static_cast<size_t>(n_axis_[a]));
            }
        }
    }

    for(int i=0; i<n_streams; i++)
    {
        d_temp[i].allocate(M);
        if (dim > 1)
            d_q_star[i].allocate(M);
    }

    // The full grid around the steps of a reduced-basis run, in which they work in place
    provide_full_grid(/*separate_output=*/false);

    // Allow tridiagonal_lines_shared() up to 96000 bytes of dynamic shared memory, or the
    // device's opt-in limit when that is smaller (see solve_lines())
    int device = 0, optin_shmem = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&optin_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    max_lines_shmem_ = std::min(96000, optin_shmem);
    CUDA_CHECK(cudaFuncSetAttribute(tridiagonal_lines_shared<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_lines_shmem_));
    CUDA_CHECK(cudaFuncSetAttribute(tridiagonal_lines_shared<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_lines_shmem_));

    update_laplacian_operator();
}

void CudaSolverCNADI::update_laplacian_operator()
{
    if (!cb->is_orthogonal())
        throw_with_line_number("CN-ADI requires an orthogonal computation box.");

    // The diagonals of the x, y and z axes (an axis the box does not have keeps one point)
    std::array<std::vector<double>, 3> l, d, h;
    for (int a = 0; a < 3; ++a)
    {
        const size_t n = a < dim ? n_axis_[a] : 1;
        l[a].resize(n);
        d[a].resize(n);
        h[a].resize(n);
    }
    auto upload = [](CudaDeviceBuffer<double>& dst, const std::vector<double>& src)
    {
        CUDA_CHECK(cudaMemcpy(dst, src.data(), sizeof(double)*src.size(), cudaMemcpyHostToDevice));
    };

    // Coefficients for each unique ds value from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            double bond_length_sq = item.second*item.second;

            // Full-step coefficients (local_ds)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                l[0].data(), d[0].data(), h[0].data(),
                l[1].data(), d[1].data(), h[1].data(),
                l[2].data(), d[2].data(), h[2].data(),
                bond_length_sq, local_ds);

            // The matrices do not change from step to step: factorize them once
            std::array<AxisMatrix, 3>& matrices = matrices_.at(ds_idx).at(item.first);
            for (int a = 0; a < dim; ++a)
            {
                const TridiagonalFactors f = factorize_tridiagonal(l[a].data(), d[a].data(), h[a].data(),
                                                                   n_axis_[a], periodic_[a]);
                upload(matrices[a].l, l[a]);
                upload(matrices[a].d, d[a]);
                upload(matrices[a].h, h[a]);
                upload(matrices[a].factors, f.table);
                matrices[a].v = f.v;
                matrices[a].denom = f.denom;
            }
        }
    }
}
void CudaSolverCNADI::update_dw(const std::map<std::string, const double*>& d_w)
{
    if (!cb->is_orthogonal())
        throw_with_line_number("CN-ADI requires an orthogonal computation box.");
    // exp(-w ds/2) on the full grid, on both sides of the implicit step
    compute_exp_dw(d_w, /*reduced_basis_tables=*/false, false, 0.5, 0.0);
}
void CudaSolverCNADI::advance_propagator(
    const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask, int ds_index)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M = this->cb->get_total_grid();
    const cudaStream_t stream = streams[STREAM];

    // Full-grid input/output (expanded from / reduced to the basis with a space group;
    // the steps then work in place in the expanded input)
    double *d_q_in_full = expand_to_full_grid(STREAM, d_q_in);
    double *q = full_grid_output(STREAM, d_q_out);

    // CN-ADI2: single full step only (2nd order accuracy)
    const double *_d_exp_dw = d_exp_dw.at(ds_index).at(monomer_type);
    const std::array<AxisMatrix, 3>& matrices = matrices_.at(ds_index).at(monomer_type);
    std::array<AdiAxis, 3> axes{};
    for (int a = 0; a < dim; ++a)
        axes[a] = {matrices[a].l, matrices[a].d, matrices[a].h, n_axis_[a], stride_[a], periodic_[a]};

    // Apply exp(-w*local_ds/2) at start
    ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(q, d_q_in_full, _d_exp_dw, 1.0, M);
    CUDA_CHECK_LAUNCH();

    // Diffusion with full-step coefficients, in place in q: the explicit half of the step
    // is the right-hand side of the first sweep ...
    if (dim == 3)
        adi_rhs_3d<<<N_BLOCKS, N_THREADS, 0, stream>>>(axes[0], axes[1], axes[2], d_temp[STREAM], q, M);
    else if (dim == 2)
        adi_rhs_2d<<<N_BLOCKS, N_THREADS, 0, stream>>>(axes[0], axes[1], d_temp[STREAM], q, M);
    else
        adi_rhs_1d<<<N_BLOCKS, N_THREADS, 0, stream>>>(axes[0], d_temp[STREAM], q, M);
    CUDA_CHECK_LAUNCH();

    // ... then one implicit sweep per axis: each after the first corrects its right-hand
    // side with the result of the sweep before it, and the last writes q
    for (int a = 0; a < dim; ++a)
    {
        if (a > 0)
        {
            adi_correction<<<N_BLOCKS, N_THREADS, 0, stream>>>(axes[a], d_temp[STREAM], d_q_star[STREAM], q, M);
            CUDA_CHECK_LAUNCH();
        }
        solve_lines(STREAM, matrices[a], a, d_temp[STREAM], a == dim - 1 ? q : d_q_star[STREAM].get());
    }

    // Apply exp(-w*local_ds/2) at end
    ker_multi<<<N_BLOCKS, N_THREADS, 0, stream>>>(q, q, _d_exp_dw, 1.0, M);
    CUDA_CHECK_LAUNCH();

    // Reduce to the basis (with a space group) and multiply the mask
    store_full_grid(STREAM, q, d_q_out, d_q_mask);
}
void CudaSolverCNADI::solve_lines(const int STREAM, const AxisMatrix& matrix, int axis, const double *d_rhs, double *d_x)
{
    // The lines of the axis: n_lines lines of M points, stride apart. Contiguous lines go
    // through shared memory, a warp of lines per block; lines along other axes, whose
    // neighbouring values belong to neighbouring threads, and lines too long for the shared
    // memory are solved one per thread in place
    const int M = n_axis_[axis];
    const int stride = stride_[axis];
    const int n_lines = this->cb->get_total_grid() / M;
    const int pitch = (M % 2 == 1) ? M : M + 1;
    const int rows = std::min(32, max_lines_shmem_ / (pitch * static_cast<int>(sizeof(double))));
    cudaStream_t stream = streams[STREAM];
    if (stride == 1 && rows >= 1)
    {
        const size_t shmem = static_cast<size_t>(rows) * pitch * sizeof(double);
        const int blocks = (n_lines + rows - 1) / rows;
        if (periodic_[axis])
            tridiagonal_lines_shared<true><<<blocks, rows, shmem, stream>>>(
                matrix.factors, M, matrix.v, matrix.denom, d_rhs, d_x, n_lines);
        else
            tridiagonal_lines_shared<false><<<blocks, rows, shmem, stream>>>(
                matrix.factors, M, matrix.v, matrix.denom, d_rhs, d_x, n_lines);
    }
    else if (periodic_[axis])
        tridiagonal_lines<true><<<line_blocks(n_lines), LINE_THREADS, 0, stream>>>(
            matrix.factors, M, matrix.v, matrix.denom, d_rhs, d_x, n_lines, stride);
    else
        tridiagonal_lines<false><<<line_blocks(n_lines), LINE_THREADS, 0, stream>>>(
            matrix.factors, M, matrix.v, matrix.denom, d_rhs, d_x, n_lines, stride);
    CUDA_CHECK_LAUNCH();
}
void CudaSolverCNADI::compute_single_segment_stress(
    [[maybe_unused]] const int STREAM,
    [[maybe_unused]] double *d_q_1, [[maybe_unused]] double *d_q_2, [[maybe_unused]] double *d_segment_stress,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    throw_with_line_number("Currently, the CN-ADI method does not support stress computation.");
}

// Thomas factors of a line: forward and backward sweeps over the factors, then for a periodic
// line the Sherman-Morrison correction
TridiagonalFactors factorize_tridiagonal(const double* l, const double* d, const double* h, int M, bool periodic)
{
    TridiagonalFactors f;
    f.table.assign(4 * static_cast<size_t>(M), 0.0);
    double* lower = f.table.data();
    double* inv = lower + M;
    double* c = lower + 2 * M;
    double* q = lower + 3 * M;
    std::copy(l, l + M, lower);
    if (!periodic)
    {
        inv[0] = 1.0 / d[0];
        c[0] = h[0] * inv[0];
        for (int i = 1; i < M; i++)
        {
            inv[i] = 1.0 / (d[i] - l[i] * c[i - 1]);
            if (i < M - 1)
                c[i] = h[i] * inv[i];
        }
        return f;
    }
    // The corners enter the first and last pivots; q solves the matrix for the corner vector
    const double gamma = -d[0];
    inv[0] = 1.0 / (d[0] - gamma);
    c[0] = h[0] * inv[0];
    q[0] = gamma * inv[0];
    for (int i = 1; i < M - 1; i++)
    {
        inv[i] = 1.0 / (d[i] - l[i] * c[i - 1]);
        c[i] = h[i] * inv[i];
        q[i] = (-l[i] * q[i - 1]) * inv[i];
    }
    inv[M - 1] = 1.0 / (d[M - 1] - h[M - 1] * l[0] / gamma - l[M - 1] * c[M - 2]);
    q[M - 1] = (h[M - 1] - l[M - 1] * q[M - 2]) * inv[M - 1];
    for (int i = M - 2; i >= 0; i--)
        q[i] -= c[i] * q[i + 1];
    f.v = l[0] / gamma;
    f.denom = 1.0 + q[0] + f.v * q[M - 1];
    return f;
}
