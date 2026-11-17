/**
 * @file CudaCrysFFTStress.cu
 * @brief Shared CrysFFT stress fast path for CUDA pseudo-spectral solvers.
 *
 * See CudaCrysFFTStress.h for the algorithm description. This mirrors the
 * CPU implementation in CpuSolverPseudoBase::compute_single_segment_stress.
 */

#include "CudaCrysFFTStress.h"

#include <string>

#include <cub/device/device_reduce.cuh>

#include "CrysFFTBasisMapping.h"
#include "CudaCommon.h"
#include "SpaceGroup.h"

namespace {

/**
 * @brief Spread the group sums over their axes and apply the final scaling (single thread).
 *
 * d_stress[g] holds the sum for axis group g (see crysfft_stress_axis_groups);
 * every axis of a group gets the group sum divided by the group size.
 *
 * @param d_stress Device array holding the group sums in [0..n_groups-1]
 * @param masks    Axis masks of the groups (1: x, 2: y, 4: z)
 * @param s0,s1,s2 Final scale factors b²·M_full/L_d²
 */
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
    // NOTE: only the three diagonal components are written (the fast path
    // serves orthogonal cells, whose cross terms are structurally zero).
    // Some callers allocate d_stress with exactly 3 elements, so indices
    // 3..5 must NOT be touched here; callers that read 6 components must
    // zero-initialize their buffer.
}

}  // namespace

void cuda_crysfft_compute_single_segment_stress(
    const CudaCrysFFTStressArgs& args,
    const double* d_q1_reduced,
    const double* d_q2_reduced,
    double* d_segment_stress)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // ===== Step 1: Fill physical grid from reduced basis (q2) =====
    // Loop-invariant: an out-of-place apply_multiplier preserves its input (the
    // Pmmm engine copies it to the output buffer before its in-place DCTs, the
    // recursive 3m engine's out-of-place D2Z only reads it), so expand once for all axes.
    const double* d_q2_phys;
    if (args.identity_map)
    {
        d_q2_phys = d_q2_reduced;
    }
    else
    {
        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
            args.d_phys_work, d_q2_reduced, args.d_phys_to_reduced, args.M_phys);
        CUDA_CHECK_LAUNCH();
        d_q2_phys = args.d_phys_work;
    }

    // One multiplier Σ_{d in group} k_d² per group of equivalent axes; the group
    // sums go to d_segment_stress[0..n_groups-1] (see crysfft_stress_axis_groups)
    const std::vector<int> groups = crysfft_stress_axis_groups(args.space_group->get_crystal_system());
    for (size_t g = 0; g < groups.size(); ++g)
    {
        // ===== Step 2: Apply the group multiplier (input is preserved) =====
        args.crysfft->apply_multiplier(const_cast<double*>(d_q2_phys), args.d_phys_out, *args.multipliers[g], args.stream);

        // ===== Step 3: Reduce physical grid back to reduced basis =====
        const double* d_tmp_reduced;
        if (args.identity_map)
        {
            d_tmp_reduced = args.d_phys_out;
        }
        else
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
                args.d_reduce_buf, args.d_phys_out, args.d_reduced_to_phys, args.n_basis);
            CUDA_CHECK_LAUNCH();
            d_tmp_reduced = args.d_reduce_buf;
        }

        // ===== Step 4: Orbit-weighted dot product with q1 =====
        ker_multi<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
            args.d_reduce_buf, d_tmp_reduced, d_q1_reduced, 1.0, args.n_basis);
        CUDA_CHECK_LAUNCH();
        ker_multi_weight<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
            args.d_reduce_buf, args.d_reduce_buf, args.d_orbit_counts, args.n_basis);
        CUDA_CHECK_LAUNCH();

        size_t temp_bytes = args.cub_temp_bytes;
        CUDA_CHECK(cub::DeviceReduce::Sum(args.d_cub_temp, temp_bytes,
                               args.d_reduce_buf, &d_segment_stress[g], args.n_basis, args.stream));
        CUDA_CHECK_LAUNCH();
    }

    // ===== Step 5: Spread the group sums over their axes and scale =====
    // The multipliers above are weighted by Cartesian k_d² = (2πm_d/L_d)²,
    // while the standard path accumulates deformation-vector components
    // v_d² = k_d²/L_d². Divide each sum by L_d² so both paths return the
    // same V_dd, and scale by b²·M_full (FFT normalization convention).
    int masks[3] = {0, 0, 0};
    for (size_t g = 0; g < groups.size(); ++g)
        masks[g] = groups[g];

    const double s0 = args.bond_length_sq * args.M_full / (args.lx[0] * args.lx[0]);
    const double s1 = args.bond_length_sq * args.M_full / (args.lx[1] * args.lx[1]);
    const double s2 = args.bond_length_sq * args.M_full / (args.lx[2] * args.lx[2]);

    ker_crysfft_stress_finalize<<<1, 1, 0, args.stream>>>(
        d_segment_stress, static_cast<int>(groups.size()), masks[0], masks[1], masks[2], s0, s1, s2);
    CUDA_CHECK_LAUNCH();
}
