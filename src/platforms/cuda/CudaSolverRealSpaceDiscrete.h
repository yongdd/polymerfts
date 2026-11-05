/**
 * @file CudaSolverRealSpaceDiscrete.h
 * @brief CUDA real-space solver for discrete chain propagators (numerical method "rs").
 *
 * The CUDA counterpart of CpuSolverRealSpaceDiscrete: the Gaussian bond is summed on
 * the grid, one axis after another, with the boundary conditions in ghost tables
 * (RealSpaceBond). A block copies a tile of the grid with its halos into shared memory
 * once, signed by the ghost tables, and each thread sums the stencil of one output point
 * from there; the last pass also multiplies by exp(-w ds) and the mask.
 *
 * The stress of a bond is sum_r q_1 (dG/dL_a q_2), with dG/dL_a the bond whose kernel along
 * axis a is differentiated with respect to the box length; the sum is reduced in a fixed order.
 * With a space group the propagators stay in its reduced basis and each bond expands them to
 * the full grid.
 *
 * @see CpuSolverRealSpaceDiscrete for the algorithm
 */

#ifndef CUDA_SOLVER_REAL_SPACE_DISCRETE_H_
#define CUDA_SOLVER_REAL_SPACE_DISCRETE_H_

#include <array>
#include <map>
#include <string>

#include "ComputationBox.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Exception.h"
#include "Molecules.h"
#include "RealSpaceBond.h"

template <typename T>
class CudaSolverRealSpaceDiscrete : public CudaSolver<T>
{
private:
    /// Kernels of the three axes on the device, one after another
    struct DeviceKernels
    {
        CudaDeviceBuffer<double> weights;
        std::array<int, 3> offset{};
        std::array<int, 3> half{};
    };

    ComputationBox<T>* cb;
    Molecules* molecules;
    int n_streams;
    SpaceGroup* space_group_ = nullptr;          ///< propagators in its reduced basis when set
    int* d_reduced_basis_indices_ = nullptr;     ///< reduced -> full index (lent by the backend)
    int* d_full_to_reduced_map_ = nullptr;       ///< full -> reduced index (lent by the backend)
    int n_basis_ = 0;                            ///< points of the propagators
    cudaStream_t streams[MAX_STREAMS][2];

    std::map<int, std::map<std::string, DeviceKernels>> d_bond;   ///< full bonds per ds index and monomer type
    std::map<std::string, DeviceKernels> d_bond_half;             ///< half bonds per monomer type
    /// Full and half bonds with the kernel of one axis differentiated with respect to the box length
    std::map<std::string, std::array<DeviceKernels, 3>> d_bond_dlength, d_bond_half_dlength;
    std::array<int, 3> halo{};
    std::string refusal_;   ///< Why the bonds of the current box do not fit a block (empty when they do)
    std::array<CudaDeviceBuffer<int>, 3> d_ghost_index;
    std::array<CudaDeviceBuffer<double>, 3> d_ghost_sign;

    CudaDeviceBuffer<CuDeviceData<T>> d_work_a[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_work_b[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_in[MAX_STREAMS];     ///< full grid with a space group
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_out[MAX_STREAMS];    ///< full grid with a space group
    CudaDeviceBuffer<CuDeviceData<T>> d_work_stress[MAX_STREAMS];   ///< dG/dL q_2 (allocated by prepare_stress())
    CudaDeviceBuffer<CuDeviceData<T>> d_partial_sums[MAX_STREAMS];  ///< partial sums of the stress

    static void upload(const RealSpaceBond::Kernels& kernels, DeviceKernels& device);
    /// q_out = g * q_in, one axis after another; the last pass multiplies by d_factor and d_mask when given
    void convolve(int STREAM, const CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out, const DeviceKernels& kernels,
                  const CuDeviceData<T>* d_factor, const double* d_mask);

public:
    CudaSolverRealSpaceDiscrete(ComputationBox<T>* cb, Molecules* molecules, int n_streams,
                                cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);
    ~CudaSolverRealSpaceDiscrete() override = default;

    void set_space_group(SpaceGroup* sg, int* d_reduced_basis_indices, int* d_full_to_reduced_map, int n_basis) override;
    void update_laplacian_operator() override;
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;
    void advance_propagator(const int STREAM, CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out,
                            std::string monomer_type, double* d_q_mask, int ds_index) override;
    void advance_propagator_half_bond_step(const int STREAM, CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out,
                                           std::string monomer_type) override;
    void prepare_stress() override;
    void compute_single_segment_stress(const int STREAM, CuDeviceData<T>* d_q_pair, CuDeviceData<T>* d_segment_stress,
                                       std::string monomer_type, bool is_half_bond_length) override;
};

#endif
