/**
 * @file CpuSolverRealSpaceDiscrete.cpp
 * @brief CPU real-space solver for discrete chain propagators (numerical method "rs").
 *
 * One bond is three passes of a symmetric stencil, one per axis:
 *
 *     out_i = g_0 in_i + sum_{d=1..H} g_d (in_{i-d} + in_{i+d}),
 *
 * with the points beyond the box taken from the ghost tables. The pass along the
 * contiguous axis copies each line with its halo into a buffer; the other passes
 * work on whole planes of contiguous values, which the compiler vectorizes.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverRealSpaceDiscrete<double>: Real fields
 * - CpuSolverRealSpaceDiscrete<std::complex<double>>: Complex fields
 */

#include <algorithm>
#include <cmath>
#include <complex>

#include "CpuSolverRealSpaceDiscrete.h"

namespace
{
// out = g * in along one axis of n points; `inner` contiguous values per point, `outer` blocks of n points
template <typename T>
void pass_strided(const T* in, T* out, int n, long inner, long outer, const RealSpaceBond::Kernel& g,
                  const int* index, const double* sign)
{
    const int H = static_cast<int>(g.size()) - 1;
    for (long o = 0; o < outer; ++o)
    {
        const T* a = in + o * n * inner;
        T* b = out + o * n * inner;
        for (int i = 0; i < n; ++i)
        {
            T* bi = b + i * inner;
            const T* ai = a + i * inner;
            for (long j = 0; j < inner; ++j)
                bi[j] = g[0] * ai[j];
            for (int d = 1; d <= H; ++d)
            {
                // index and sign are centred on point 0 of the axis
                const T* l = a + static_cast<long>(index[i - d]) * inner;
                const T* r = a + static_cast<long>(index[i + d]) * inner;
                const double wl = g[d] * sign[i - d];
                const double wr = g[d] * sign[i + d];
                for (long j = 0; j < inner; ++j)
                    bi[j] += wl * l[j] + wr * r[j];
            }
        }
    }
}

// Along the contiguous axis: each line with its halo in a buffer, then the stencil over the line
template <typename T>
void pass_contiguous(const T* in, T* out, int n, long lines, const RealSpaceBond::Kernel& g,
                     const int* index, const double* sign, std::vector<T>& line)
{
    const int H = static_cast<int>(g.size()) - 1;
    line.resize(n + 2 * H);
    for (long l = 0; l < lines; ++l)
    {
        const T* a = in + l * n;
        T* b = out + l * n;
        for (int p = -H; p < n + H; ++p)
            line[p + H] = sign[p] * a[index[p]];
        const T* c = line.data() + H;
        for (int i = 0; i < n; ++i)
            b[i] = g[0] * c[i];
        for (int d = 1; d <= H; ++d)
        {
            const double w = g[d];
            for (int i = 0; i < n; ++i)
                b[i] += w * (c[i - d] + c[i + d]);
        }
    }
}
}  // namespace

template <typename T>
CpuSolverRealSpaceDiscrete<T>::CpuSolverRealSpaceDiscrete(ComputationBox<T>* cb, Molecules* molecules)
    : cb(cb), molecules(molecules)
{
    try
    {
        if (molecules->get_model_name() != "discrete")
            throw_with_line_number("The rs method is for discrete chains.");
        if (!cb->is_orthogonal())
            throw_with_line_number("The rs method requires an orthogonal computation box.");

        const int M = cb->get_total_grid();
        molecules->finalize_contour_length_mapping();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
            for (const auto& item : molecules->get_bond_lengths())
                this->exp_dw[ds_idx][item.first].resize(M);

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::set_space_group(SpaceGroup* sg)
{
    if (sg != nullptr)
        throw_with_line_number("The rs method does not support space groups.");
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::update_laplacian_operator()
{
    try
    {
        if (!cb->is_orthogonal())
            throw_with_line_number("The rs method requires an orthogonal computation box.");

        // A bond of contour ds has the variance b^2 ds / 3 along each axis, a half bond half of it
        const std::vector<double> dx = cb->get_dx();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        const double ds = mapping.get_ds_from_index(0);
        std::vector<const RealSpaceBond::Kernels*> all;
        for (const auto& [monomer_type, b] : molecules->get_bond_lengths())
        {
            for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
            {
                bond[ds_idx][monomer_type] = RealSpaceBond::gaussian(b * b * mapping.get_ds_from_index(ds_idx) / 3.0, dx);
                all.push_back(&bond[ds_idx][monomer_type]);
            }
            bond_half[monomer_type] = RealSpaceBond::gaussian(b * b * ds / 6.0, dx);
            all.push_back(&bond_half[monomer_type]);
            bond_dlength[monomer_type] = RealSpaceBond::length_derivatives(b * b * ds / 3.0, dx, cb->get_lx());
            bond_half_dlength[monomer_type] = RealSpaceBond::length_derivatives(b * b * ds / 6.0, dx, cb->get_lx());
        }
        ghosts = RealSpaceBond::ghosts(cb->get_nx(), cb->get_boundary_conditions(), all);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M = cb->get_total_grid();
    const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
    for (int ds_idx = 0; ds_idx < mapping.get_n_unique_ds(); ++ds_idx)
    {
        const double local_ds = mapping.get_ds_from_index(ds_idx);
        for (const auto& [monomer_type, w] : w_input)
        {
            if (!this->exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");
            std::vector<T>& exp_dw = this->exp_dw.at(ds_idx).at(monomer_type);
            for (int i = 0; i < M; ++i)
                exp_dw[i] = std::exp(-w[i] * local_ds);
        }
    }
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::convolve(const T* q_in, T* q_out, const RealSpaceBond::Kernels& kernels) const
{
    const std::vector<int> nx = cb->get_nx();
    const int dim = static_cast<int>(nx.size());
    const long M = cb->get_total_grid();

    // Scratch kept per thread (propagators advance in parallel)
    thread_local std::vector<T> work_a, work_b, line;
    work_a.resize(M);
    work_b.resize(M);

    // The contiguous (last) axis first, from q_in; then the others, the first one into q_out
    const T* src = q_in;
    if (q_in == q_out)
    {
        std::copy(q_in, q_in + M, work_b.begin());
        src = work_b.data();
    }
    T* dst = work_a.data();
    for (int axis = dim - 1; axis >= 0; --axis)
    {
        const int n = nx[axis];
        long inner = 1, outer = 1;
        for (int d = axis + 1; d < dim; ++d)
            inner *= nx[d];
        for (int d = 0; d < axis; ++d)
            outer *= nx[d];
        if (axis == 0)
            dst = q_out;
        const int* index = ghosts.index[axis].data() + ghosts.halo[axis];
        const double* sign = ghosts.sign[axis].data() + ghosts.halo[axis];
        if (inner == 1)
            pass_contiguous(src, dst, n, outer, kernels[axis], index, sign, line);
        else
            pass_strided(src, dst, n, inner, outer, kernels[axis], index, sign);
        src = dst;
        dst = (dst == work_a.data()) ? work_b.data() : work_a.data();
    }
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    try
    {
        const int M = cb->get_total_grid();
        const T* exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();
        convolve(q_in, q_out, bond.at(ds_index).at(monomer_type));
        for (int i = 0; i < M; ++i)
            q_out[i] *= exp_dw[i];
        if (q_mask != nullptr)
            for (int i = 0; i < M; ++i)
                q_out[i] *= q_mask[i];
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CpuSolverRealSpaceDiscrete<T>::advance_propagator_half_bond_step(T* q_in, T* q_out, std::string monomer_type)
{
    try
    {
        convolve(q_in, q_out, bond_half.at(monomer_type));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
std::vector<T> CpuSolverRealSpaceDiscrete<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        // The cell stays orthogonal (the factory holds it so), so there are no shear components
        if (this->force_off_diagonal_stress_)
            throw_with_line_number("The rs method keeps the cell orthogonal and computes no shear stress.");

        // The sums V_aa = sum_k b^2 g(k) q_1 q_2 k_a^2 / L_a^2 of the pseudo-spectral solvers: in real
        // space k_a^2 g(k) is (L_a / sigma^2) dG/dL_a, with sigma^2 the variance of the (half) bond
        // and b^2 the squared bond length (halved for a half bond), so
        // V_aa = M (b^2 / sigma^2) / L_a sum_r q_1 (dG/dL_a q_2) = M (3 / ds) / L_a sum_r q_1 (dG/dL_a q_2)
        const int M = cb->get_total_grid();
        const int dim = cb->get_dim();
        const double ds = molecules->get_contour_length_mapping().get_ds_from_index(0);
        const auto& kernels = (is_half_bond_length ? bond_half_dlength : bond_dlength).at(monomer_type);
        thread_local std::vector<T> dq;
        dq.resize(M);
        std::vector<T> stress(6, T(0.0));
        for (int a = 0; a < dim; ++a)
        {
            convolve(q_2, dq.data(), kernels[a]);
            T sum = 0.0;
            for (int i = 0; i < M; ++i)
                sum += q_1[i] * dq[i];
            stress[a] = sum * (M * 3.0 / (ds * cb->get_lx(a)));
        }
        return stress;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverRealSpaceDiscrete);
