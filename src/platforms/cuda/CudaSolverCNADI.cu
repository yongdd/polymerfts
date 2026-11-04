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
 * The modified diffusion equation is split dimensionally:
 * - 3D: Three sequential tridiagonal solves (x, y, z directions)
 * - 2D: Two tridiagonal solves
 * - 1D: Single tridiagonal solve
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

// The line solvers below give one thread to each grid line, so a launch sized
// for whole grids (CudaCommon blocks x threads) would leave most blocks idle:
// a 64^3 grid has 4096 lines per direction. Launch one 64-thread block per 64 lines.
constexpr int LINE_THREADS = 64;
static inline int line_blocks(int n_lines) { return (n_lines + LINE_THREADS - 1)/LINE_THREADS; }

CudaSolverCNADI::CudaSolverCNADI(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->n_streams = n_streams;
        this->space_group_ = nullptr;
        this->d_reduced_basis_indices_ = nullptr;
        this->d_full_to_reduced_map_ = nullptr;
        this->n_basis_ = 0;

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
        const int DIM = cb->get_dim();
        this->dim = DIM;
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        else if (DIM == 2)
            nx = {cb->get_nx(0), cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {cb->get_nx(0), 1, 1};

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Ensure ContourLengthMapping is finalized
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Map slots are owning buffers: a failed allocation leaves the slot empty
        // Create exp_dw and tridiagonal coefficients for each ds_index
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                d_exp_dw[ds_idx][monomer_type].allocate(M);

                d_xl[ds_idx][monomer_type].allocate(nx[0]);
                d_xd[ds_idx][monomer_type].allocate(nx[0]);
                d_xh[ds_idx][monomer_type].allocate(nx[0]);

                d_yl[ds_idx][monomer_type].allocate(nx[1]);
                d_yd[ds_idx][monomer_type].allocate(nx[1]);
                d_yh[ds_idx][monomer_type].allocate(nx[1]);

                d_zl[ds_idx][monomer_type].allocate(nx[2]);
                d_zd[ds_idx][monomer_type].allocate(nx[2]);
                d_zh[ds_idx][monomer_type].allocate(nx[2]);
            }
        }

        if(DIM == 3)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
                d_q_dstar[i].allocate(M);
                d_temp[i].allocate(M);
            }

            std::vector<int> offset_xy(nx[0]*nx[1]);
            std::vector<int> offset_yz(nx[1]*nx[2]);
            std::vector<int> offset_xz(nx[0]*nx[2]);
            int count;

            count = 0;
            for(int i=0;i<nx[0];i++)
                for(int j=0;j<nx[1];j++)
                    offset_xy[count++] = i*nx[1]*nx[2] + j*nx[2];

            count = 0;
            for(int j=0;j<nx[1];j++)
                for(int k=0;k<nx[2];k++)
                    offset_yz[count++] = j*nx[2] + k;

            count = 0;
            for(int i=0;i<nx[0];i++)
                for(int k=0;k<nx[2];k++)
                    offset_xz[count++] = i*nx[1]*nx[2] + k;

            d_offset_xy.allocate(nx[0]*nx[1]);
            d_offset_yz.allocate(nx[1]*nx[2]);
            d_offset_xz.allocate(nx[0]*nx[2]);

            CUDA_CHECK(cudaMemcpy(d_offset_xy, offset_xy.data(), sizeof(int)*nx[0]*nx[1], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_yz, offset_yz.data(), sizeof(int)*nx[1]*nx[2], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_xz, offset_xz.data(), sizeof(int)*nx[0]*nx[2], cudaMemcpyHostToDevice));
        }
        else if(DIM == 2)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
                d_temp[i].allocate(M);
            }

            std::vector<int> offset_x(nx[0]);
            std::vector<int> offset_y(nx[1]);

            for(int i=0;i<nx[0];i++)
                offset_x[i] = i*nx[1];

            for(int j=0;j<nx[1];j++)
                offset_y[j] = j;

            d_offset_x.allocate(nx[0]);
            d_offset_y.allocate(nx[1]);

            CUDA_CHECK(cudaMemcpy(d_offset_x, offset_x.data(), sizeof(int)*nx[0], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_y, offset_y.data(), sizeof(int)*nx[1], cudaMemcpyHostToDevice));
        }
        else if(DIM == 1)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
            }

            d_offset.allocate(1);
            CUDA_CHECK(cudaMemset(d_offset, 0, sizeof(int)));
        }

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
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}
CudaSolverCNADI::~CudaSolverCNADI() = default;


void CudaSolverCNADI::set_space_group(
    SpaceGroup* sg,
    int* d_reduced_basis_indices,
    int* d_full_to_reduced_map,
    int n_basis)
{
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = n_basis;

    const int M = cb->get_total_grid();
    if (space_group_ != nullptr)
    {
        for (int i = 0; i < n_streams; i++)
        {
            if (d_q_full_in_[i].empty())
                d_q_full_in_[i].allocate(M);
            if (d_q_full_out_[i].empty())
                d_q_full_out_[i].allocate(M);
        }
    }
}
void CudaSolverCNADI::update_laplacian_operator()
{
    try
    {
        if (!cb->is_orthogonal())
            throw_with_line_number("CN-ADI requires an orthogonal computation box.");

        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {this->cb->get_nx(0), this->cb->get_nx(1), this->cb->get_nx(2)};
        else if (DIM == 2)
            nx = {this->cb->get_nx(0), this->cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {this->cb->get_nx(0), 1, 1};

        std::vector<double> xl(nx[0]), xd(nx[0]), xh(nx[0]);
        std::vector<double> yl(nx[1]), yd(nx[1]), yh(nx[1]);
        std::vector<double> zl(nx[2]), zd(nx[2]), zh(nx[2]);

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Compute coefficients for each unique ds value
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: this->molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                double bond_length_sq = item.second*item.second;

                // Full-step coefficients (local_ds)
                FiniteDifference::get_laplacian_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    xl.data(), xd.data(), xh.data(),
                    yl.data(), yd.data(), yh.data(),
                    zl.data(), zd.data(), zh.data(),
                    bond_length_sq, local_ds);

                CUDA_CHECK(cudaMemcpy(d_xl.at(ds_idx).at(monomer_type), xl.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_xd.at(ds_idx).at(monomer_type), xd.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_xh.at(ds_idx).at(monomer_type), xh.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(d_yl.at(ds_idx).at(monomer_type), yl.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_yd.at(ds_idx).at(monomer_type), yd.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_yh.at(ds_idx).at(monomer_type), yh.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(d_zl.at(ds_idx).at(monomer_type), zl.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_zd.at(ds_idx).at(monomer_type), zd.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_zh.at(ds_idx).at(monomer_type), zh.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));

                // The matrices do not change from step to step: factorize them once
                const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();
                const std::array<const double*, 3> lower = {xl.data(), yl.data(), zl.data()};
                const std::array<const double*, 3> diagonal = {xd.data(), yd.data(), zd.data()};
                const std::array<const double*, 3> upper = {xh.data(), yh.data(), zh.data()};
                for (int axis = 0; axis < DIM; ++axis)
                {
                    const TridiagonalFactors f = factorize_tridiagonal(lower[axis], diagonal[axis], upper[axis],
                        nx[axis], bc[2*axis] == BoundaryCondition::PERIODIC);
                    DeviceFactors& device_factors = d_factors[ds_idx][monomer_type][axis];
                    if (device_factors.table.size() != f.table.size())
                        device_factors.table.allocate(f.table.size());
                    CUDA_CHECK(cudaMemcpy(device_factors.table, f.table.data(), sizeof(double)*f.table.size(), cudaMemcpyHostToDevice));
                    device_factors.v = f.v;
                    device_factors.denom = f.denom;
                    device_factors.periodic = bc[2*axis] == BoundaryCondition::PERIODIC;
                }

            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    try{
        if (!cb->is_orthogonal())
            throw_with_line_number("CN-ADI requires an orthogonal computation box.");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (space_group_ != nullptr);

        if (device != "gpu" && device != "cpu")
            throw_with_line_number("Invalid device \"" + device + "\".");

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        for(const auto& item: w_input)
        {
            const std::string& monomer_type = item.first;
            const double *w = item.second;

            for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
            {
                if (d_exp_dw[ds_idx].find(monomer_type) == d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");
            }

            // The full-grid field on the device: the input itself, or a copy in the
            // table of the last ds, which is computed last (in place)
            double* d_stage = d_exp_dw.at(n_unique_ds - 1).at(monomer_type);
            const double* d_w = w;
            if (use_reduced_basis && device == "gpu")
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_stage, w, d_full_to_reduced_map_, M);
                CUDA_CHECK_LAUNCH();
                d_w = d_stage;
            }
            else if (device == "cpu")
            {
                const double* w_full = w;
                std::vector<double> w_full_host;
                if (use_reduced_basis)
                {
                    w_full_host.resize(M);
                    space_group_->from_reduced_basis(w, w_full_host.data(), 1);
                    w_full = w_full_host.data();
                }
                CUDA_CHECK(cudaMemcpyAsync(d_stage, w_full, sizeof(double)*M, cudaMemcpyHostToDevice));
                d_w = d_stage;
            }

            // d_exp_dw = exp(-w*local_ds/2) for each unique ds value
            for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
            {
                const double local_ds = mapping.get_ds_from_index(ds_idx);
                ker_exp<<<N_BLOCKS, N_THREADS>>>(
                    d_exp_dw.at(ds_idx).at(monomer_type).get(), d_w, 1.0, -0.50*local_ds, M);
                CUDA_CHECK_LAUNCH();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator(
    const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();
        const bool use_reduced_basis = (space_group_ != nullptr);

        double *d_q_in_full = d_q_in;
        double *d_q_out_full = d_q_out;

        if (use_reduced_basis)
        {
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
            CUDA_CHECK_LAUNCH();
            d_q_in_full = d_q_full_in_[STREAM];
            d_q_out_full = d_q_full_out_[STREAM];
        }

        // CN-ADI2: single full step only (2nd order accuracy)
        double *_d_exp_dw = d_exp_dw.at(ds_index).at(monomer_type);

        // Apply exp(-w*local_ds/2) at start
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out_full, d_q_in_full, _d_exp_dw, 1.0, M);
        CUDA_CHECK_LAUNCH();

        // Diffusion with full-step coefficients
        if(DIM == 3)
            advance_propagator_3d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);
        else if(DIM == 2)
            advance_propagator_2d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);
        else if(DIM == 1)
            advance_propagator_1d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);

        // Apply exp(-w*local_ds/2) at end
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out_full, d_q_out_full, _d_exp_dw, 1.0, M);
        CUDA_CHECK_LAUNCH();

        if (use_reduced_basis)
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_out_full, d_reduced_basis_indices_, n_basis_);
            CUDA_CHECK_LAUNCH();
        }

        // Multiply mask (stored in the reduced basis when a space group is set)
        if (d_q_mask != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_out, d_q_mask, 1.0, use_reduced_basis ? n_basis_ : M);
            CUDA_CHECK_LAUNCH();
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator_3d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);
        const auto& factors = d_factors.at(ds_index).at(monomer_type);

        double *_d_yl = d_yl.at(ds_index).at(monomer_type);
        double *_d_yd = d_yd.at(ds_index).at(monomer_type);
        double *_d_yh = d_yh.at(ds_index).at(monomer_type);

        double *_d_zl = d_zl.at(ds_index).at(monomer_type);
        double *_d_zd = d_zd.at(ds_index).at(monomer_type);
        double *_d_zh = d_zh.at(ds_index).at(monomer_type);

        // Calculate q_star
        compute_crank_3d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], 
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            _d_zl, _d_zd, _d_zh, nx[2],
            d_temp[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        solve_lines(STREAM, factors[0], d_temp[STREAM], d_q_star[STREAM], d_offset_yz, nx[1]*nx[2], nx[1]*nx[2], nx[0]);

        // Calculate q_dstar
        compute_crank_3d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3], 
            _d_yl, _d_yd, _d_yh, nx[1], nx[2],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        solve_lines(STREAM, factors[1], d_temp[STREAM], d_q_dstar[STREAM], d_offset_xz, nx[0]*nx[2], nx[2], nx[1]);

        // Calculate q^(n+1)
        compute_crank_3d_step_3<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[4], bc[5], 
            _d_zl, _d_zd, _d_zh, nx[1], nx[2],
            d_temp[STREAM], d_q_dstar[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        solve_lines(STREAM, factors[2], d_temp[STREAM], d_q_out, d_offset_xy, nx[0]*nx[1], 1, nx[2]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator_2d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);
        const auto& factors = d_factors.at(ds_index).at(monomer_type);

        double *_d_yl = d_yl.at(ds_index).at(monomer_type);
        double *_d_yd = d_yd.at(ds_index).at(monomer_type);
        double *_d_yh = d_yh.at(ds_index).at(monomer_type);

        // Calculate q_star
        compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        // CUDA_CHECK(cudaMemcpy(d_q_out, d_q_star, sizeof(double)*M, cudaMemcpyDeviceToDevice));

        solve_lines(STREAM, factors[0], d_temp[STREAM], d_q_star[STREAM], d_offset_y, nx[1], nx[1], nx[0]);

        // Calculate q_dstar
        compute_crank_2d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        solve_lines(STREAM, factors[1], d_temp[STREAM], d_q_out, d_offset_x, nx[0], 1, nx[1]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator_1d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);
        const auto& factors = d_factors.at(ds_index).at(monomer_type);

        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, nx[0]);
        CUDA_CHECK_LAUNCH();

        solve_lines(STREAM, factors[0], d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::solve_lines(
    const int STREAM, const DeviceFactors& factors, const double *d_d, double *d_x, const int *d_offset,
    const int REPEAT, const int INTERVAL, const int M)
{
    // Contiguous lines (one after another, M apart) go through shared memory, a warp of lines per
    // block; lines along other axes, whose neighbouring values belong to neighbouring threads,
    // and lines too long for the shared memory are solved one per thread in place
    const int pitch = (M % 2 == 1) ? M : M + 1;
    const int rows = std::min(32, max_lines_shmem_ / (pitch * static_cast<int>(sizeof(double))));
    cudaStream_t stream = streams[STREAM][0];
    if (INTERVAL == 1 && rows >= 1)
    {
        const size_t shmem = static_cast<size_t>(rows) * pitch * sizeof(double);
        const int blocks = (REPEAT + rows - 1) / rows;
        if (factors.periodic)
            tridiagonal_lines_shared<true><<<blocks, rows, shmem, stream>>>(
                factors.table, M, factors.v, factors.denom, d_d, d_x, REPEAT);
        else
            tridiagonal_lines_shared<false><<<blocks, rows, shmem, stream>>>(
                factors.table, M, factors.v, factors.denom, d_d, d_x, REPEAT);
    }
    else if (factors.periodic)
        tridiagonal_lines<true><<<line_blocks(REPEAT), LINE_THREADS, 0, stream>>>(
            factors.table, M, factors.v, factors.denom, d_d, d_x, d_offset, REPEAT, INTERVAL);
    else
        tridiagonal_lines<false><<<line_blocks(REPEAT), LINE_THREADS, 0, stream>>>(
            factors.table, M, factors.v, factors.denom, d_d, d_x, d_offset, REPEAT, INTERVAL);
    CUDA_CHECK_LAUNCH();
}
void CudaSolverCNADI::compute_single_segment_stress(
    [[maybe_unused]] const int STREAM,
    [[maybe_unused]] double *d_q_pair, [[maybe_unused]] double *d_segment_stress,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try
    {
        throw_with_line_number("Currently, the CN-ADI method does not support stress computation.");
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

__device__ int d_max_of_two(int x, int y)
{
   return (x > y) ? x : y;
}
__device__ int d_min_of_two(int x, int y)
{
   return (x < y) ? x : y;
}

__global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip, jm, jp, km, kp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (I+i-1) % I;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % I;
        else
            ip = d_min_of_two(I-1,i+1);

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        if (bc_zl == BoundaryCondition::PERIODIC)
            km = (K+k-1) % K;
        else
            km = d_max_of_two(0,k-1);
        if (bc_zh == BoundaryCondition::PERIODIC)
            kp = (k+1) % K;
        else
            kp = d_min_of_two(K-1,k+1);

        int im_j_k = im*J*K + j*K + k;
        int ip_j_k = ip*J*K + j*K + k;
        int i_jm_k = i*J*K + jm*K + k;
        int i_jp_k = i*J*K + jp*K + k;
        int i_j_km = i*J*K + j*K + km;
        int i_j_kp = i*J*K + j*K + kp;

        d_q_out[n] = 2.0*((3.0-0.5*d_xd[i]-d_yd[j]-d_zd[k])*d_q_in[n]
                - d_zl[k]*d_q_in[i_j_km] - d_zh[k]*d_q_in[i_j_kp]
                - d_yl[j]*d_q_in[i_jm_k] - d_yh[j]*d_q_in[i_jp_k])
                - d_xl[i]*d_q_in[im_j_k] - d_xh[i]*d_q_in[ip_j_k];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M)
{
    int jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm_k = i*J*K + jm*K + k;
        int i_jp_k = i*J*K + jp*K + k;

        d_q_out[n] = d_q_star[n] + (d_yd[j]-1.0)*d_q_in[n]
            + d_yl[j]*d_q_in[i_jm_k] + d_yh[j]*d_q_in[i_jp_k];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M)
{
    int km, kp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_zl == BoundaryCondition::PERIODIC)
            km = (K+k-1) % K;
        else
            km = d_max_of_two(0,k-1);
        if (bc_zh == BoundaryCondition::PERIODIC)
            kp = (k+1) % K;
        else
            kp = d_min_of_two(K-1,k+1);

        int i_j_km = i*J*K + j*K + km;
        int i_j_kp = i*J*K + j*K + kp;

        d_q_out[n] = d_q_dstar[n] + (d_zd[k]-1.0)*d_q_in[n]
            + d_zl[k]*d_q_in[i_j_km] + d_zh[k]*d_q_in[i_j_kp];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip, jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / J;
        int j = n % J;

        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (I+i-1) % I;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % I;
        else
            ip = d_min_of_two(I-1,i+1);

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm = i*J + jm;
        int i_jp = i*J + jp;
        int im_j = im*J + j;
        int ip_j = ip*J + j;

        d_q_out[n] = 2.0*((2.0-0.5*d_xd[i]-d_yd[j])*d_q_in[n]
                   - d_yl[j]*d_q_in[i_jm] - d_yh[j]*d_q_in[i_jp])
                   - d_xl[i]*d_q_in[im_j] - d_xh[i]*d_q_in[ip_j];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M)
{
    int jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n/J;
        int j = n%J;

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm = i*J + jm;
        int i_jp = i*J + jp;

        d_q_out[n] = d_q_star[n] + (d_yd[j]-1.0)*d_q_in[n]
            + d_yl[j]*d_q_in[i_jm] + d_yh[j]*d_q_in[i_jp];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < M)
    {
        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (M+i-1) % M;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % M;
        else
            ip = d_min_of_two(M-1,i+1);

        // B part of Ax=B matrix equation
        d_q_out[i] = (2.0-d_xd[i])*d_q_in[i] - d_xl[i]*d_q_in[im] - d_xh[i]*d_q_in[ip];

        i += blockDim.x * gridDim.x;
    }
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

// One line of M values with the given stride: the sweeps over the factors (see factorize_tridiagonal())
template <bool PERIODIC>
__device__ __forceinline__ void solve_line(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* d, double* x, const int stride)
{
    const double* lower = table;
    const double* inv = table + M;
    const double* c = table + 2 * M;
    double y = d[0] * __ldg(&inv[0]);
    x[0] = y;
    for (int i = 1; i < M; i++)
    {
        y = (d[i * stride] - __ldg(&lower[i]) * y) * __ldg(&inv[i]);
        x[i * stride] = y;
    }
    for (int i = M - 2; i >= 0; i--)
    {
        y = x[i * stride] - __ldg(&c[i]) * y;
        x[i * stride] = y;
    }
    if constexpr (PERIODIC)
    {
        const double* q = table + 3 * M;
        const double value = (x[0] + v * x[(M - 1) * stride]) / denom;
        for (int i = 0; i < M; i++)
            x[i * stride] -= __ldg(&q[i]) * value;
    }
}

template <bool PERIODIC>
__global__ void tridiagonal_lines(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x,
    const int* __restrict__ d_offset, const int REPEAT, const int INTERVAL)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < REPEAT; n += blockDim.x * gridDim.x)
        solve_line<PERIODIC>(table, M, v, denom, d_d + d_offset[n], d_x + d_offset[n], INTERVAL);
}

template <bool PERIODIC>
__global__ void tridiagonal_lines_shared(
    const double* __restrict__ table, const int M, const double v, const double denom,
    const double* __restrict__ d_d, double* __restrict__ d_x, const int REPEAT)
{
    extern __shared__ double s_lines[];
    const int rows = blockDim.x;
    const int pitch = (M % 2 == 1) ? M : M + 1;
    for (long first = static_cast<long>(blockIdx.x) * rows; first < REPEAT; first += static_cast<long>(gridDim.x) * rows)
    {
        const int count = static_cast<int>(min(static_cast<long>(rows), REPEAT - first));
        // Neighbouring threads read neighbouring values of the block's lines
        for (int k = threadIdx.x; k < count * M; k += rows)
            s_lines[(k / M) * pitch + k % M] = d_d[first * M + k];
        __syncthreads();
        if (threadIdx.x < count)
            solve_line<PERIODIC>(table, M, v, denom, s_lines + threadIdx.x * pitch, s_lines + threadIdx.x * pitch, 1);
        __syncthreads();
        for (int k = threadIdx.x; k < count * M; k += rows)
            d_x[first * M + k] = s_lines[(k / M) * pitch + k % M];
        __syncthreads();
    }
}

template __global__ void tridiagonal_lines<false>(const double* __restrict__, const int, const double, const double,
    const double* __restrict__, double* __restrict__, const int* __restrict__, const int, const int);
template __global__ void tridiagonal_lines<true>(const double* __restrict__, const int, const double, const double,
    const double* __restrict__, double* __restrict__, const int* __restrict__, const int, const int);
template __global__ void tridiagonal_lines_shared<false>(const double* __restrict__, const int, const double, const double,
    const double* __restrict__, double* __restrict__, const int);
template __global__ void tridiagonal_lines_shared<true>(const double* __restrict__, const int, const double, const double,
    const double* __restrict__, double* __restrict__, const int);
