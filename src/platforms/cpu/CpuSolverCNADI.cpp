/**
 * @file CpuSolverCNADI.cpp
 * @brief CPU CN-ADI solver for chain propagators.
 *
 * Implements the CN-ADI (Crank-Nicolson Alternating Direction Implicit)
 * finite difference method for solving the modified diffusion equation
 * in real space.
 *
 * **CN-ADI Scheme:**
 *
 * Semi-implicit discretization:
 *     (I - ds/2·L) q^(n+1) = (I + ds/2·L) q^n
 *
 * where L is the discrete Laplacian operator.
 *
 * **ADI Splitting (3D):**
 *
 * The 3D solve is split into three 1D tridiagonal solves:
 * 1. X-direction sweep: compute q*
 * 2. Y-direction sweep: compute q**
 * 3. Z-direction sweep: compute q^(n+1)
 *
 * Each sweep solves all lines of a plane together (see CpuSolverCNADI): the
 * right-hand side of the plane first, then the Thomas sweeps over the factors
 * of TridiagonalLines with the lines innermost.
 *
 * **Boundary Conditions:**
 *
 * Supports periodic, reflecting (Neumann), and absorbing (Dirichlet)
 * boundaries. Uses cyclic Thomas algorithm for periodic boundaries.
 *
 * **Limitation:**
 *
 * Only supports continuous chain model (not discrete).
 * Stress computation not implemented.
 *
 * @see FiniteDifference for matrix coefficient computation
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "CpuSolverCNADI.h"

//------------------------------------------------------------------------------
// TridiagonalLines
//------------------------------------------------------------------------------
TridiagonalLines::TridiagonalLines(std::vector<double> l, std::vector<double> d, std::vector<double> h, bool periodic)
    : l_(std::move(l)), d_(std::move(d)), h_(std::move(h)), periodic_(periodic)
{
    const int n = size();
    if (n < 2 || l_.size() != d_.size() || h_.size() != d_.size())
        throw_with_line_number("A tridiagonal matrix needs at least two rows and three diagonals of one length.");

    // The steps of the Thomas algorithm that do not involve the right-hand side, in its order
    den_.resize(n);
    c_star_.resize(n - 1);
    if (!periodic_)
    {
        den_[0] = d_[0];
        for (int i = 1; i < n; ++i)
        {
            c_star_[i - 1] = h_[i - 1] / den_[i - 1];
            den_[i] = d_[i] - l_[i] * c_star_[i - 1];
        }
        return;
    }

    // Cyclic: the Sherman-Morrison correction with u = (gamma, 0, ..., h[n-1]); a negative
    // diagonal-sized gamma avoids cancellation when diffusion is small
    const double gamma = -d_[0];
    q_.resize(n);
    den_[0] = d_[0] - gamma;
    q_[0] = gamma / den_[0];
    for (int i = 1; i < n - 1; ++i)
    {
        c_star_[i - 1] = h_[i - 1] / den_[i - 1];
        den_[i] = d_[i] - l_[i] * c_star_[i - 1];
        q_[i] = (-l_[i] * q_[i - 1]) / den_[i];
    }
    c_star_[n - 2] = h_[n - 2] / den_[n - 2];
    den_[n - 1] = d_[n - 1] - h_[n - 1] * l_[0] / gamma - l_[n - 1] * c_star_[n - 2];
    q_[n - 1] = (h_[n - 1] - l_[n - 1] * q_[n - 2]) / den_[n - 1];
    for (int i = n - 2; i >= 0; i--)
        q_[i] = q_[i] - c_star_[i] * q_[i + 1];
    l0_gamma_ = l_[0] / gamma;
    q_den_ = 1.0 + q_[0] + l0_gamma_ * q_[n - 1];
}

void TridiagonalLines::solve(const double* rhs, double* x, long stride, long count, double* correction) const
{
    const int n = size();

    // The sweeps over `lines` lines, the lines innermost; one line (1D) is compiled
    // without the inner loops
    auto sweeps = [&](auto lines) {
        // Forward sweep
        for (long c = 0; c < lines; ++c)
            x[c] = rhs[c] / den_[0];
        for (int i = 1; i < n; ++i)
        {
            const double l = l_[i];
            const double den = den_[i];
            const double* r = rhs + i * count;
            const double* x_prev = x + (i - 1) * stride;
            double* x_i = x + i * stride;
            for (long c = 0; c < lines; ++c)
                x_i[c] = (r[c] - l * x_prev[c]) / den;
        }

        // Backward substitution
        for (int i = n - 2; i >= 0; i--)
        {
            const double c_star = c_star_[i];
            const double* x_next = x + (i + 1) * stride;
            double* x_i = x + i * stride;
            for (long c = 0; c < lines; ++c)
                x_i[c] = x_i[c] - c_star * x_next[c];
        }

        if (!periodic_)
            return;

        // Sherman-Morrison correction of the cyclic matrix
        const double* x_last = x + (n - 1) * stride;
        for (long c = 0; c < lines; ++c)
            correction[c] = (x[c] + l0_gamma_ * x_last[c]) / q_den_;
        for (int i = 0; i < n; ++i)
        {
            const double q = q_[i];
            double* x_i = x + i * stride;
            for (long c = 0; c < lines; ++c)
                x_i[c] = x_i[c] - q * correction[c];
        }
    };
    if (count == 1)
        sweeps(std::integral_constant<long, 1>{});
    else
        sweeps(count);
}

namespace
{
/// f(k, k_lower, k_upper) for k = 0..n-1 (n >= 2): the neighbours of the tables at both
/// ends of the axis and k-1, k+1 in between, where every table has them
template <typename F>
inline void along_axis(int n, const int* lower, const int* upper, F f)
{
    f(0, lower[0], upper[0]);
    for (int k = 1; k < n - 1; ++k)
        f(k, k - 1, k + 1);
    f(n - 1, lower[n - 1], upper[n - 1]);
}

/**
 * The sweep along the contiguous axis (n points): n_planes planes of `count` grid rows.
 * rhs_row(row, r) writes the right-hand side of grid row `row` (points row·n to row·n + n - 1)
 * to r[k·count], k = 0..n-1; each plane is solved transposed and in place, the rows innermost,
 * and q_out = exp_dw · solution.
 */
template <typename RhsRow>
void sweep_contiguous(const TridiagonalLines& m, long n_planes, int count, RhsRow rhs_row,
                      const double* exp_dw, double* q_out, double* plane, double* correction)
{
    const int n = m.size();
    for (long p = 0; p < n_planes; ++p)
    {
        for (int c = 0; c < count; ++c)
            rhs_row(p * count + c, plane + c);
        m.solve(plane, plane, count, count, correction);
        for (int c = 0; c < count; ++c)
        {
            const long row = (p * count + c) * n;
            for (int k = 0; k < n; ++k)
                q_out[row + k] = plane[static_cast<long>(k) * count + c] * exp_dw[row + k];
        }
    }
}

/// The last sweep of a 2D or 3D grid, along the contiguous axis: the right-hand side
/// s + (d - 1) a + l a_- + h a_+ of the previous sweep's result s and a = exp_dw · q_in
void last_sweep(const TridiagonalLines& m, const int* lower, const int* upper, const double* s, const double* a,
                const double* exp_dw, double* q_out, long n_planes, int count, double* plane, double* correction)
{
    const int n = m.size();
    const double* l = m.lower();
    const double* d = m.diagonal();
    const double* h = m.upper();
    sweep_contiguous(m, n_planes, count, [&](long row, double* r) {
        const double* s_row = s + row * n;
        const double* a_row = a + row * n;
        along_axis(n, lower, upper, [&](int k, int km, int kp) {
            r[static_cast<long>(k) * count] = s_row[k] + (d[k]-1.0)*a_row[k] + l[k]*a_row[km] + h[k]*a_row[kp];
        });
    }, exp_dw, q_out, plane, correction);
}
}  // namespace

//------------------------------------------------------------------------------
// CpuSolverCNADI
//------------------------------------------------------------------------------
/**
 * @brief Construct CN-ADI solver for continuous chains.
 *
 * Builds the neighbour tables of the axes and the tridiagonal matrices for each
 * ds value and monomer type.
 *
 * @param cb            Computation box for grid and boundary info
 * @param molecules     Molecular system (must use continuous model)
 * @throws Exception if discrete chain model is specified
 */
CpuSolverCNADI::CpuSolverCNADI(ComputationBox<double>* cb, Molecules *molecules)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->space_group_ = nullptr;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Real-space method only support 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("Real-space method only supports orthogonal unit cells. "
                                   "Use pseudo-spectral method (chain_model='continuous') for non-orthogonal systems.");

        dim_ = cb->get_dim();
        nx_ = cb->get_nx();

        // The tridiagonal solvers need at least two points on every axis
        for (int d = 0; d < dim_; d++)
            if (nx_[d] < 2)
                throw_with_line_number("Real-space method needs at least 2 grid points on every axis, got nx["
                                       + std::to_string(d) + "] = " + std::to_string(nx_[d]) + ".");

        // Neighbours along each axis: a periodic side wraps around; at a wall the
        // point itself, whose coefficient FiniteDifference sets to zero
        const std::vector<BoundaryCondition> bc = cb->get_boundary_conditions();
        lower_.resize(dim_);
        upper_.resize(dim_);
        for (int d = 0; d < dim_; ++d)
        {
            const int n = nx_[d];
            lower_[d].resize(n);
            upper_[d].resize(n);
            for (int i = 0; i < n; ++i)
            {
                lower_[d][i] = (bc[2*d] == BoundaryCondition::PERIODIC) ? (n + i - 1) % n : std::max(0, i - 1);
                upper_[d][i] = (bc[2*d+1] == BoundaryCondition::PERIODIC) ? (i + 1) % n : std::min(n - 1, i + 1);
            }
        }

        // Boltzmann factors for each unique ds value of the ContourLengthMapping
        const int M = cb->get_total_grid();
        molecules->finalize_contour_length_mapping();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
            for (const auto& item : molecules->get_bond_lengths())
                exp_dw[ds_idx][item.first].resize(M);

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverCNADI::update_laplacian_operator()
{
    if (!cb->is_orthogonal())
        throw_with_line_number("CN-ADI requires an orthogonal computation box.");

    const std::vector<BoundaryCondition> bc = cb->get_boundary_conditions();
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            const std::string& monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            // Lower, main and upper diagonal of each axis (an axis the grid does not
            // have gets one unused entry)
            std::array<std::vector<double>, 9> diagonals;
            for (int a = 0; a < 3; ++a)
                for (int k = 0; k < 3; ++k)
                    diagonals[3*a + k].resize(a < dim_ ? nx_[a] : 1);
            FiniteDifference::get_laplacian_matrix(
                bc, nx_, this->cb->get_dx(),
                diagonals[0].data(), diagonals[1].data(), diagonals[2].data(),
                diagonals[3].data(), diagonals[4].data(), diagonals[5].data(),
                diagonals[6].data(), diagonals[7].data(), diagonals[8].data(),
                bond_length_sq, local_ds);

            std::vector<TridiagonalLines> axes;
            for (int a = 0; a < dim_; ++a)
                axes.emplace_back(std::move(diagonals[3*a]), std::move(diagonals[3*a + 1]),
                                  std::move(diagonals[3*a + 2]), bc[2*a] == BoundaryCondition::PERIODIC);
            matrices_[ds_idx].insert_or_assign(monomer_type, std::move(axes));
        }
    }
}

void CpuSolverCNADI::update_dw(std::map<std::string, const double*> w_input)
{
    if (!cb->is_orthogonal())
        throw_with_line_number("CN-ADI requires an orthogonal computation box.");

    const int M = this->cb->get_total_grid();
    const bool use_reduced_basis = (this->space_group_ != nullptr);

    std::map<std::string, std::vector<double>> w_full_cache;
    if (use_reduced_basis)
    {
        for (const auto& item : w_input)
        {
            const std::string& monomer_type = item.first;
            w_full_cache[monomer_type].resize(M);
            this->space_group_->from_reduced_basis(item.second, w_full_cache[monomer_type].data(), 1);
        }
    }

    // Get unique ds values from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    // Compute exp_dw for each unique ds value
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for(const auto& item: w_input)
        {
            const std::string& monomer_type = item.first;
            const double *w = use_reduced_basis ? w_full_cache[monomer_type].data() : item.second;

            if( !exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            std::vector<double>& exp_dw_vec = exp_dw.at(ds_idx).at(monomer_type);

            for(int i=0; i<M; i++)
            {
                // Full step: exp(-w*local_ds/2) for symmetric splitting
                exp_dw_vec[i] = exp(-w[i]*local_ds*0.5);
            }
        }
    }
}

CpuSolverCNADI::WorkArrays& CpuSolverCNADI::work_arrays() const
{
    WorkArrays& work = work_arrays_.get();
    const size_t M = cb->get_total_grid();
    if (work.q_exp.size() != M)
    {
        // Planes: 1D the line; 2D the grid (y sweep, transposed); 3D (x, z) of the x sweep and (z, y) of the z sweep.
        // Lines per plane: 1D one; 2D n_y (x sweep) and n_x (y sweep); 3D n_z (x, y sweeps) and n_y (z sweep)
        size_t plane = nx_[0];
        size_t lines = 1;
        if (dim_ == 2)
        {
            plane = M;
            lines = std::max(nx_[0], nx_[1]);
        }
        else if (dim_ == 3)
        {
            plane = static_cast<size_t>(std::max(nx_[0], nx_[1])) * nx_[2];
            lines = std::max(nx_[1], nx_[2]);
        }
        work.q_exp.resize(M);
        work.q_star.resize(dim_ >= 2 ? M : 0);
        work.q_dstar.resize(dim_ == 3 ? M : 0);
        work.plane.resize(plane);
        work.correction.resize(lines);
    }
    return work;
}

void CpuSolverCNADI::advance_propagator(
    double *q_in, double *q_out, std::string monomer_type, const double *q_mask, int ds_index)
{
    try
    {
        const int M = this->cb->get_total_grid();
        WorkArrays& work = work_arrays();
        const double* _exp_dw = exp_dw.at(ds_index).at(monomer_type).data();   // exp(-w*local_ds/2)
        const std::vector<TridiagonalLines>& matrices = matrices_.at(ds_index).at(monomer_type);

        // On the full grid: with a space group from and back to its reduced basis
        const double* q_full_in = q_in;
        double* q_full_out = q_out;
        if (this->space_group_ != nullptr)
        {
            work.q_full.resize(M);
            this->space_group_->from_reduced_basis(q_in, work.q_full.data(), 1);
            q_full_in = work.q_full.data();
            q_full_out = work.q_full.data();
        }

        // CN-ADI2: exp(-w*h/2), the ADI diffusion step, exp(-w*h/2)
        for (int i = 0; i < M; i++)
            work.q_exp[i] = _exp_dw[i] * q_full_in[i];
        if (dim_ == 3)
            step_3d(matrices, _exp_dw, q_full_out, work);
        else if (dim_ == 2)
            step_2d(matrices, _exp_dw, q_full_out, work);
        else
            step_1d(matrices, _exp_dw, q_full_out, work);

        int n_out = M;
        if (this->space_group_ != nullptr)
        {
            this->space_group_->to_reduced_basis(q_full_out, q_out, 1);
            n_out = this->space_group_->get_n_reduced_basis();
        }

        // The box stores the mask in the basis of the propagators (reduced with a space
        // group); it is symmetric, so masking after the reduction equals masking the full grid
        if (q_mask != nullptr)
            for (int i = 0; i < n_out; i++)
                q_out[i] *= q_mask[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// ============================================================================
// ADI steps of the 1D, 2D and 3D grids with the matrices of one (ds, monomer
// type) pair: a = exp_dw q_in (work.q_exp) in, q_out = exp_dw · ADI(a) out
// ============================================================================
void CpuSolverCNADI::step_1d(
    const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& work) const
{
    const int n0 = nx_[0];
    const double *xl = m[0].lower(), *xd = m[0].diagonal(), *xh = m[0].upper();
    const double* a = work.q_exp.data();

    // One line: (1 + A_x/2) q = (1 - A_x/2) a
    sweep_contiguous(m[0], 1, 1, [&](long, double* r) {
        along_axis(n0, lower_[0].data(), upper_[0].data(), [&](int i, int im, int ip) {
            r[i] = (2.0-xd[i])*a[i] - xl[i]*a[im] - xh[i]*a[ip];
        });
    }, exp_dw, q_out, work.plane.data(), work.correction.data());
}

void CpuSolverCNADI::step_2d(
    const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& work) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const double *xl = m[0].lower(), *xd = m[0].diagonal(), *xh = m[0].upper();
    const double *yl = m[1].lower(), *yd = m[1].diagonal(), *yh = m[1].upper();
    const int *im = lower_[0].data(), *ip = upper_[0].data();
    const double* a = work.q_exp.data();
    double* q_star = work.q_star.data();

    // x sweep: the grid is one plane; its right-hand side in q_star, row by row along y,
    // then its n_y lines along x solved together in place
    for (int i = 0; i < n0; i++)
    {
        const double* a_i = a + static_cast<long>(i) * n1;
        const double* a_im = a + static_cast<long>(im[i]) * n1;
        const double* a_ip = a + static_cast<long>(ip[i]) * n1;
        const double c = 2.0 - 0.5*xd[i];
        double* r = q_star + static_cast<long>(i) * n1;
        along_axis(n1, lower_[1].data(), upper_[1].data(), [&](int j, int jm, int jp) {
            r[j] = 2.0*((c-yd[j])*a_i[j] - yl[j]*a_i[jm] - yh[j]*a_i[jp]) - xl[i]*a_im[j] - xh[i]*a_ip[j];
        });
    }
    m[0].solve(q_star, q_star, n1, n1, work.correction.data());

    // y sweep along the contiguous axis, the grid as one plane of n_x lines
    last_sweep(m[1], lower_[1].data(), upper_[1].data(), q_star, a, exp_dw, q_out, 1, n0,
               work.plane.data(), work.correction.data());
}

void CpuSolverCNADI::step_3d(
    const std::vector<TridiagonalLines>& m, const double* exp_dw, double* q_out, WorkArrays& work) const
{
    const int n0 = nx_[0];
    const int n1 = nx_[1];
    const int n2 = nx_[2];
    const long P = static_cast<long>(n1) * n2;
    const double *xl = m[0].lower(), *xd = m[0].diagonal(), *xh = m[0].upper();
    const double *yl = m[1].lower(), *yd = m[1].diagonal(), *yh = m[1].upper();
    const double *zl = m[2].lower(), *zd = m[2].diagonal(), *zh = m[2].upper();
    const int *im = lower_[0].data(), *ip = upper_[0].data();
    const int *jm = lower_[1].data(), *jp = upper_[1].data();
    const double* a = work.q_exp.data();
    double* q_star = work.q_star.data();
    double* q_dstar = work.q_dstar.data();
    double* plane = work.plane.data();

    // x sweep: for each j the plane (x, z), row by row along z, then its n_z lines along x
    for (int j = 0; j < n1; j++)
    {
        for (int i = 0; i < n0; i++)
        {
            const double* a_ij = a + i * P + static_cast<long>(j) * n2;
            const double* a_imj = a + im[i] * P + static_cast<long>(j) * n2;
            const double* a_ipj = a + ip[i] * P + static_cast<long>(j) * n2;
            const double* a_ijm = a + i * P + static_cast<long>(jm[j]) * n2;
            const double* a_ijp = a + i * P + static_cast<long>(jp[j]) * n2;
            const double c = 3.0 - 0.5*xd[i] - yd[j];
            double* r = plane + static_cast<long>(i) * n2;
            along_axis(n2, lower_[2].data(), upper_[2].data(), [&](int k, int km, int kp) {
                r[k] = 2.0*((c-zd[k])*a_ij[k]
                        - zl[k]*a_ij[km] - zh[k]*a_ij[kp]
                        - yl[j]*a_ijm[k] - yh[j]*a_ijp[k])
                        - xl[i]*a_imj[k] - xh[i]*a_ipj[k];
            });
        }
        m[0].solve(plane, q_star + static_cast<long>(j) * n2, P, n2, work.correction.data());
    }

    // y sweep: for each i the plane (y, z), contiguous in q_dstar: its right-hand side there,
    // row by row along z, then its n_z lines along y in place
    for (int i = 0; i < n0; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            const long row = i * P + static_cast<long>(j) * n2;
            const double* s = q_star + row;
            const double* a_ij = a + row;
            const double* a_ijm = a + i * P + static_cast<long>(jm[j]) * n2;
            const double* a_ijp = a + i * P + static_cast<long>(jp[j]) * n2;
            const double d_1 = yd[j]-1.0;
            double* r = q_dstar + row;
            for (int k = 0; k < n2; k++)
                r[k] = s[k] + d_1*a_ij[k] + yl[j]*a_ijm[k] + yh[j]*a_ijp[k];
        }
        m[1].solve(q_dstar + i * P, q_dstar + i * P, n2, n2, work.correction.data());
    }

    // z sweep along the contiguous axis: for each i the plane (y, z) of n_y lines
    last_sweep(m[2], lower_[2].data(), upper_[2].data(), q_dstar, a, exp_dw, q_out, n0, n1,
               plane, work.correction.data());
}

std::vector<double> CpuSolverCNADI::compute_single_segment_stress(
    double*, double*, std::string, bool)
{
    throw_with_line_number("Currently, the CN-ADI method does not support stress computation.");
}
