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
 * axis a is differentiated with respect to the box length. With a space group the propagators
 * are kept in its reduced basis and expanded to the full grid for each bond: the Gaussian of an
 * orthogonal cell commutes with the operations of its group.
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
    SpaceGroup* space_group_ = nullptr;   ///< propagators in its reduced basis when set

    /// Kernels of the full bonds, per ds index and monomer type
    std::map<int, std::map<std::string, RealSpaceBond::Kernels>> bond;
    /// Kernels of the half bonds at chain ends and junctions (global ds)
    std::map<std::string, RealSpaceBond::Kernels> bond_half;
    /// For the stress: the full and half bonds with the kernel of one axis differentiated
    /// with respect to the box length, per monomer type and axis (global ds)
    std::map<std::string, std::array<RealSpaceBond::Kernels, 3>> bond_dlength, bond_half_dlength;
    /// Source and sign of the points beyond each end of each axis
    RealSpaceBond::Ghosts ghosts;

    /// q_out = g * q_in, one axis after another, on the full grid
    void convolve(const T* q_in, T* q_out, const RealSpaceBond::Kernels& kernels) const;
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
