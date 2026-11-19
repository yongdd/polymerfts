/**
 * @file CpuSolverRealSpaceDiscrete.h
 * @brief CPU real-space solver for discrete chain propagators (numerical method "rs").
 *
 * The bond convolution of the Chapman-Kolmogorov equation,
 *
 *     q(r, i+1) = exp(-w(r)*ds) * sum_r' g(r - r') q(r', i),
 *
 * is summed on the grid instead of being taken through FFTs. The Gaussian bond
 * g factorizes into one kernel per axis of an orthogonal box, so a bond is three
 * one-dimensional stencils. Positive real-space weights preserve each value of
 * q's relative precision however small it is. An FFT
 * leaves an error of about 1e-16 of the largest value at every point, which the
 * Boltzmann factor amplifies when a field drops by tens of kT per chain where q is
 * small (grafted brushes, charged layers).
 *
 * The walls enter through ghost points: periodic axes wrap, reflecting walls mirror
 * the grid about the cell face (the symmetry of the DCT-II), absorbing walls mirror
 * it with the opposite sign (DST-II). Each end of an axis has its own condition.
 * Narrow absorbing axes instead multiply by the dense matrix of the folded kernel,
 * whose entries come from a compact eigenfunction expansion once per box, avoiding
 * cancellation between ghost images.
 *
 * The stress of a bond is sum_r q_1 (dG/dL_a q_2), with dG/dL_a the bond whose kernel along
 * axis a is differentiated with respect to the box length. With a space group the propagators
 * are kept in its reduced basis and expanded to the full grid for each bond: the Gaussian of an
 * orthogonal cell commutes with the operations of its group.
 *
 * @see CpuSolverPseudoDiscrete for the FFT version ("ps")
 */

#ifndef CPU_SOLVER_REAL_SPACE_DISCRETE_H_
#define CPU_SOLVER_REAL_SPACE_DISCRETE_H_

#include <map>
#include <string>
#include <vector>

#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Exception.h"
#include "Molecules.h"
#include "RealSpaceBond.h"

template <typename T>
class CpuSolverRealSpaceDiscrete : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;
    Molecules* molecules;
    SpaceGroup* space_group_ = nullptr;   ///< propagators in its reduced basis when set

    /// Kernels of the full and half bonds, their length derivatives and the ghost tables of the box
    RealSpaceBond::Bonds bonds;

    /// Work arrays of one calling thread (PerThread)
    struct WorkArrays
    {
        std::vector<T> full_1, full_2;   ///< full-grid propagators (space group): the input and output, or the stress pair
        std::vector<T> dq;               ///< stress: the derivative of the bond applied to q_2
        std::vector<T> pass_a, pass_b;   ///< convolve(): results of the axis passes
        std::vector<T> line;             ///< convolve(): a line with its halo
    };
    PerThread<WorkArrays> work_arrays_;   ///< Work arrays of each thread

    /// Work arrays of the calling thread (created on first use)
    WorkArrays& work_arrays() const { return work_arrays_.get(); }

    /// q_out = g * q_in, one axis after another, on the full grid; a derivative replaces the kernel
    /// of axis derivative_axis
    void convolve(const T* q_in, T* q_out, const RealSpaceBond::Kernels& kernels, WorkArrays& work,
                  int derivative_axis = -1, const RealSpaceBond::Derivative* derivative = nullptr) const;
    /// q on the full grid: q itself, or with a space group its expansion into buffer
    const T* expanded(const T* q, std::vector<T>& buffer) const;
    /// The full-grid q_full in the basis of the propagators (a copy of the reduced points with a space group)
    void reduced(const T* q_full, T* q) const;

public:
    CpuSolverRealSpaceDiscrete(ComputationBox<T>* cb, Molecules* molecules);
    ~CpuSolverRealSpaceDiscrete() override = default;

    void set_space_group(SpaceGroup* sg) override;
    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const T*> w_input) override;
    void advance_propagator(T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index) override;
    void advance_propagator_half_bond_step(T* q_in, T* q_out, std::string monomer_type) override;
    std::vector<T> compute_single_segment_stress(T* q_1, T* q_2, std::string monomer_type,
                                                 bool is_half_bond_length) override;
};

#endif
