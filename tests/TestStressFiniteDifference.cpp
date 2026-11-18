/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

// The stress against the finite difference of the free energy at fixed fields.
//
// get_stress() returns dH/dθ for the box lengths and angles θ. At fixed field
// values on the grid
//     H(θ) = -Σ_p (φ_p/α_p) ln Q_p(θ)
// up to the field terms, which are grid means and do not depend on the cell.
// Each row of the table in main() is one cell (1D, 2D or 3D, orthogonal or
// oblique), chain model and numerical method (rqm4, rk2, ps), real or complex
// fields, polymer set and memory mode. Every row runs on every available
// platform, and every length and angle derivative of the cell must match the
// central difference of H within the tolerance of the row.
//
// The fields are smooth: the modes with |m| <= 2 along each axis, random
// amplitudes and phases, a standard deviation of 5 kT (2 kT for the imaginary
// part). High-frequency content would loosen the check: where two aliases of a
// mode tie in |k| the central difference converges only as O(h)
// (TestStressAliasTies.py), and the contour error of continuous chains grows.
// Discrete chains (ps) have the exact derivative of their ln Q, so only the
// finite difference limits them. Continuous chains carry the contour error of
// the method (the Simpson sum over the segments against the derivative of ln Q
// of the same steps), O(ds^4) for rqm4 and O(ds^2) for rk2.

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "TestCudaPart.h"

namespace
{

struct Chain
{
    double volume_fraction;
    std::vector<BlockInput> blocks;
};

struct StressCase
{
    std::string name;
    std::vector<int> nx;
    std::vector<double> lx;
    std::vector<double> angles;                 // alpha, beta, gamma in degrees
    std::string chain_model;
    std::string method;
    bool complex_fields;
    double ds;
    std::map<std::string, double> bond_lengths;
    std::vector<Chain> chains;
    bool aggregate;
    bool reduce_memory;
    double tolerance;                           // on |stress - dH/dθ| / |dH/dθ| of each parameter
};

// A box length or angle and its index in get_stress()
struct CellParameter
{
    std::string name;
    int stress_index;
    bool is_angle;
    int index;                                  // in lx, or in angles (alpha, beta, gamma)
};

// Steps of the central differences: lengths in units of R0, angles in radians
constexpr double LENGTH_STEP = 1e-5;
constexpr double ANGLE_STEP = 1e-5;
// The fields: standard deviation of the real part, and of the imaginary part of complex fields
constexpr double FIELD_STD = 5.0;
constexpr double FIELD_STD_IMAG = 2.0;

std::vector<CellParameter> cell_parameters(const StressCase& c)
{
    const int dim = c.nx.size();
    bool oblique = false;
    for (double angle : c.angles)
        oblique = oblique || angle != 90.0;

    std::vector<CellParameter> params;
    for (int d = 0; d < dim; ++d)
        params.push_back({"L" + std::to_string(d + 1), d, false, d});
    if (oblique && dim == 2)
        params.push_back({"gamma", 2, true, 2});
    if (oblique && dim == 3)
    {
        params.push_back({"gamma", 3, true, 2});
        params.push_back({"beta", 4, true, 1});
        params.push_back({"alpha", 5, true, 0});
    }
    return params;
}

// Sum of the modes with |m_d| <= 2 on every axis, random amplitudes and phases,
// scaled to the standard deviation `std`. The values depend on the grid indices
// only, so they stay fixed when the cell changes.
std::vector<double> smooth_field(const std::vector<int>& nx, std::mt19937& engine, double std)
{
    const int K = 2;
    const int dim = nx.size();
    int M = 1;
    for (int n : nx)
        M *= n;

    std::uniform_real_distribution<double> amplitude(-1.0, 1.0);
    std::uniform_real_distribution<double> phase(0.0, 2.0*std::numbers::pi);
    std::vector<double> w(M, 0.0);
    std::vector<int> m(dim, -K);
    while (true)
    {
        bool zero_mode = true;
        for (int d = 0; d < dim; ++d)
            zero_mode = zero_mode && m[d] == 0;
        if (!zero_mode)
        {
            const double a = amplitude(engine);
            const double p = phase(engine);
            for (int i = 0; i < M; ++i)
            {
                // grid index along each axis (last axis fastest)
                double x = 0.0;
                int rest = i;
                for (int d = dim - 1; d >= 0; --d)
                {
                    x += static_cast<double>(m[d]) * (rest % nx[d]) / nx[d];
                    rest /= nx[d];
                }
                w[i] += a*std::cos(2.0*std::numbers::pi*x + p);
            }
        }
        // next mode in [-K, K]^dim
        int d = dim - 1;
        while (d >= 0 && m[d] == K)
            m[d--] = -K;
        if (d < 0)
            break;
        m[d]++;
    }

    double sum_sq = 0.0;
    for (double v : w)
        sum_sq += v*v;
    const double scale = std / std::sqrt(sum_sq / M);
    for (double& v : w)
        v *= scale;
    return w;
}

template <typename T>
std::vector<T> smooth_field(const std::vector<int>& nx, std::mt19937& engine)
{
    std::vector<double> re = smooth_field(nx, engine, FIELD_STD);
    if constexpr (std::is_same_v<T, double>)
        return re;
    else
    {
        std::vector<double> im = smooth_field(nx, engine, FIELD_STD_IMAG);
        std::vector<T> w(re.size());
        for (size_t i = 0; i < re.size(); ++i)
            w[i] = T(re[i], im[i]);
        return w;
    }
}

template <typename T>
std::unique_ptr<AbstractFactory<T>> create_factory(const std::string& platform, bool reduce_memory)
{
    if constexpr (std::is_same_v<T, double>)
        return PlatformSelector::create_factory_real(platform, reduce_memory);
    else
        return PlatformSelector::create_factory_complex(platform, reduce_memory);
}

// Checks one row on one platform; prints each parameter and returns whether all pass
template <typename T>
bool check_stress(const StressCase& c, const std::string& platform)
{
    std::unique_ptr<AbstractFactory<T>> factory = create_factory<T>(platform, c.reduce_memory);
    std::unique_ptr<ComputationBox<T>> cb = factory->create_computation_box(c.nx, c.lx, {}, c.angles);
    std::unique_ptr<Molecules> molecules = factory->create_molecules_information(c.chain_model, c.ds, c.bond_lengths);
    for (const Chain& chain : c.chains)
        molecules->add_polymer(chain.volume_fraction, chain.blocks, {});
    auto optimizer = std::make_unique<PropagatorComputationOptimizer>(molecules.get(), c.aggregate);
    std::unique_ptr<PropagatorComputation<T>> solver =
        factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), c.method);

    // One field per monomer type, the same on every platform
    std::mt19937 engine(42);
    std::map<std::string, std::vector<T>> w;
    std::map<std::string, const T*> w_input;
    for (const auto& item : c.bond_lengths)
        w[item.first] = smooth_field<T>(c.nx, engine);
    for (const auto& item : w)
        w_input[item.first] = item.second.data();

    solver->compute_propagators(w_input, {});
    solver->compute_stress();
    const std::vector<T> stress = solver->get_stress();

    // ln Q_p of every polymer in a cell
    const int n_polymers = molecules->get_n_polymer_types();
    auto partitions = [&](const std::vector<double>& lx, const std::vector<double>& angles)
    {
        cb->set_lattice_parameters(lx, angles);
        solver->update_laplacian_operator();
        solver->compute_propagators(w_input, {});
        std::vector<T> q(n_polymers);
        for (int p = 0; p < n_polymers; ++p)
            q[p] = solver->get_total_partition(p);
        return q;
    };

    bool ok = true;
    for (const CellParameter& param : cell_parameters(c))
    {
        std::vector<double> lx_plus = c.lx, lx_minus = c.lx;
        std::vector<double> angles_plus = c.angles, angles_minus = c.angles;
        double step;
        if (param.is_angle)
        {
            step = ANGLE_STEP;
            angles_plus[param.index] += step*180.0/std::numbers::pi;
            angles_minus[param.index] -= step*180.0/std::numbers::pi;
        }
        else
        {
            step = LENGTH_STEP;
            lx_plus[param.index] += step;
            lx_minus[param.index] -= step;
        }
        const std::vector<T> q_plus = partitions(lx_plus, angles_plus);
        const std::vector<T> q_minus = partitions(lx_minus, angles_minus);

        // dH/dθ = -Σ_p (φ_p/α_p) [ln Q_p(θ+h) - ln Q_p(θ-h)] / 2h; the logarithm of the
        // ratio keeps the complex logarithm on one branch
        T dh_dtheta = 0.0;
        for (int p = 0; p < n_polymers; ++p)
        {
            Polymer& pc = molecules->get_polymer(p);
            dh_dtheta -= pc.get_volume_fraction()/pc.get_alpha()*std::log(q_plus[p]/q_minus[p]);
        }
        dh_dtheta /= 2.0*step;

        const T value = stress[param.stress_index];
        const double error = std::abs(value - dh_dtheta)/std::abs(dh_dtheta);
        const bool pass = error <= c.tolerance;    // false for NaN
        ok = ok && pass;
        std::cout << "    " << std::left << std::setw(6) << param.name << std::right
                  << " stress " << std::setw(26) << value << "  finite difference " << std::setw(26) << dh_dtheta
                  << "  rel. error " << std::scientific << std::setprecision(2) << error
                  << std::defaultfloat << std::setprecision(10) << (pass ? "" : "  FAILED") << std::endl;
    }
    return ok;
}

}  // namespace

int main()
{
    try
    {
        const std::vector<double> right_angles = {90.0, 90.0, 90.0};
        const std::map<std::string, double> bonds_ab = {{"A", 1.0}, {"B", 1.2}};
        auto diblock = [](double f) { return std::vector<Chain>{{1.0, {{"A", f, 0, 1}, {"B", 1.0 - f, 1, 2}}}}; };
        // Two copies of one diblock: the stress sums over the polymers
        const std::vector<Chain> two_diblocks = {
            {0.5, {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}}},
            {0.5, {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}}}};
        // Two branched polymers, each with a junction of three blocks
        const std::vector<Chain> branched = {
            {0.3, {{"A", 0.2, 0, 1}, {"A", 0.4, 0, 2}, {"A", 0.3, 1, 3}, {"A", 0.4, 1, 4}}},
            {0.7, {{"A", 0.2, 0, 1}, {"A", 0.4, 0, 3}, {"A", 0.3, 1, 2}, {"A", 0.4, 1, 4}}}};

        // The methods, their contour steps and their tolerances. The largest mismatches of
        // the rows below: rqm4 2.1e-6, ps 4e-9 (the finite difference itself), rk2 6.9e-4
        // (2.8e-3 at ds = 0.01, falling as ds^2)
        struct Method
        {
            std::string chain_model;
            std::string name;
            double ds;
            double tolerance;
        };
        const std::vector<Method> methods = {
            {"continuous", "rqm4", 0.01, 2e-5},
            {"continuous", "rk2", 0.005, 5e-3},
            {"discrete", "ps", 0.01, 1e-7}};

        // name, nx, lx, angles, chain model, method, complex fields, ds, bond lengths, polymers,
        // aggregation, reduce memory, tolerance
        std::vector<StressCase> cases;
        for (const Method& m : methods)
        {
            const std::string& model = m.chain_model;
            cases.push_back({"1D", {64}, {3.3}, right_angles, model, m.name, false, m.ds, bonds_ab,
                             diblock(0.5), true, false, m.tolerance});
            cases.push_back({"1D complex", {64}, {3.3}, right_angles, model, m.name, true, m.ds, bonds_ab,
                             two_diblocks, true, false, m.tolerance});
            cases.push_back({"2D oblique", {33, 29}, {3.2, 4.1}, {90.0, 90.0, 115.0}, model, m.name, false, m.ds,
                             bonds_ab, diblock(0.25), true, false, m.tolerance});
            cases.push_back({"2D complex", {32, 28}, {3.2, 3.4}, right_angles, model, m.name, true, m.ds, bonds_ab,
                             diblock(0.4), true, false, m.tolerance});
            cases.push_back({"2D complex oblique", {32, 28}, {3.2, 3.4}, {90.0, 90.0, 105.0}, model, m.name, true,
                             m.ds, bonds_ab, diblock(0.4), true, false, m.tolerance});
            cases.push_back({"3D triclinic", {23, 27, 25}, {3.2, 3.3, 3.4}, {83.0, 95.0, 98.0}, model, m.name,
                             false, m.ds, bonds_ab, diblock(0.36), true, false, m.tolerance});
        }
        // The branched chains in every aggregation and memory mode
        for (bool aggregate : {false, true})
            for (bool reduce_memory : {false, true})
                cases.push_back({"1D branched", {10}, {1.0}, right_angles, "discrete", "ps", false, 0.1,
                                 {{"A", 1.0}}, branched, aggregate, reduce_memory, 1e-7});

        const std::vector<std::string> platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> failed;
        const auto start = std::chrono::steady_clock::now();
        for (const StressCase& c : cases)
        {
            for (const std::string& platform : platforms)
            {
                const auto case_start = std::chrono::steady_clock::now();
                std::cout << std::setprecision(10) << c.name << ", " << c.chain_model << " " << c.method
                          << " ds " << c.ds << ", " << platform << (c.aggregate ? "" : ", no aggregation")
                          << (c.reduce_memory ? ", reduce memory" : "") << std::endl;
                const bool ok = c.complex_fields ? check_stress<std::complex<double>>(c, platform)
                                                 : check_stress<double>(c, platform);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - case_start).count();
                std::cout << "    (" << std::fixed << std::setprecision(2) << seconds << " s)" << std::defaultfloat << std::endl;
                if (!ok)
                    failed.push_back(c.name + ", " + c.method + ", " + platform +
                                     (c.aggregate ? "" : ", no aggregation") + (c.reduce_memory ? ", reduce memory" : ""));
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << cases.size()*platforms.size() << " checks in " << std::fixed << std::setprecision(1)
                  << seconds << " s" << std::endl;

        if (!failed.empty())
        {
            std::cout << "FAILED (" << failed.size() << "):" << std::endl;
            for (const std::string& name : failed)
                std::cout << "    " << name << std::endl;
            return 1;
        }
        std::cout << "Test passed!" << std::endl;
        return cuda_part_missing() ? 1 : 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return 1;
    }
}
