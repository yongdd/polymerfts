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
 * one-dimensional stencils. Every term of the sums is a positive weight times q,
 * so each value of q keeps its own relative precision however small it is. An FFT
 * leaves an error of about 1e-16 of the largest value at every point, which the
 * Boltzmann factor amplifies when a field drops by tens of kT per chain where q is
 * small (grafted brushes, charged layers).
 *
 * The walls enter through ghost points: periodic axes wrap, reflecting walls mirror
 * the grid about the cell face (the symmetry of the DCT-II), absorbing walls mirror
 * it with the opposite sign (DST-II). Each end of an axis has its own condition.
 *
 * The stress of a bond is sum_r q_1 (dG/dL_a q_2), with dG/dL_a the bond whose kernel along
 * axis a is differentiated with respect to the box length. Space groups are not available.
 *
 * @see CpuSolverPseudoDiscrete for the FFT version ("ps")
 */

#ifndef CPU_SOLVER_REAL_SPACE_DISCRETE_H_
#define CPU_SOLVER_REAL_SPACE_DISCRETE_H_

#include <array>
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

    /// Kernels of the full bonds, per ds index and monomer type
    std::map<int, std::map<std::string, RealSpaceBond::Kernels>> bond;
    /// Kernels of the half bonds at chain ends and junctions (global ds)
    std::map<std::string, RealSpaceBond::Kernels> bond_half;
    /// For the stress: the full and half bonds with the kernel of one axis differentiated
    /// with respect to the box length, per monomer type and axis (global ds)
    std::map<std::string, std::array<RealSpaceBond::Kernels, 3>> bond_dlength, bond_half_dlength;
    /// Source and sign of the points beyond each end of each axis
    RealSpaceBond::Ghosts ghosts;

    /// q_out = g * q_in, one axis after another
    void convolve(const T* q_in, T* q_out, const RealSpaceBond::Kernels& kernels) const;

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
