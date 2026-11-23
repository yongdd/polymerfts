/**
 * @file TestAbScft.h
 * @brief The SCFT iteration of an incompressible AB melt that the C++ SCFT tests run.
 *
 * step(w) takes the fields w = (w_A, w_B) and computes the concentrations, the free energy per
 * chain in the zero-mean-field convention,
 *     F = <w_-^2>/chi_n - <w_+> - sum_p phi_p/alpha_p ln Q_p,    w_+- = (w_A +- w_B)/2,
 * the output fields w_out = (chi_n phi_B + xi, chi_n phi_A + xi), both of zero mean, with the
 * pressure field xi = (w_A + w_B - chi_n)/2, and the residual
 *     sqrt(<w_out - w, w_out - w> / (<w, w> + 1)).
 * With box relaxation (TestAggregation) w holds the box lengths after the two fields: the norm
 * of the stress adds to the residual, and mix() moves the lengths against the stress with the
 * fields. The tests print the iterations and keep their own checks.
 */

#ifndef TEST_AB_SCFT_H_
#define TEST_AB_SCFT_H_

#include <cmath>
#include <vector>

#include "AndersonMixing.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputation.h"

/// The SCFT iteration of an AB melt in a box
struct AbScft
{
    ComputationBox<double>* cb;
    PropagatorComputation<double>* solver;
    Molecules* molecules;
    double chi_n;
    bool relax_box;                    ///< The stress joins the residual and the box lengths the mixing

    std::vector<double> phi_a, phi_b;  ///< Concentrations of the last step
    std::vector<double> w_diff;        ///< w_out - w of the last step (then minus the stress)
    std::vector<double> stress;        ///< Stress of the last step, with box relaxation
    double energy = 1.0e20;            ///< Free energy per chain of the last step
    double error_level = 1.0e20;       ///< Residual of the last step
    double old_error_level = 1.0e20;   ///< Residual of the step before

    AbScft(ComputationBox<double>* cb, PropagatorComputation<double>* solver, Molecules* molecules,
           double chi_n, bool relax_box = false)
        : cb(cb), solver(solver), molecules(molecules), chi_n(chi_n), relax_box(relax_box),
          phi_a(cb->get_total_grid()), phi_b(cb->get_total_grid()),
          w_diff(2*cb->get_total_grid() + (relax_box ? cb->get_dim() : 0)),
          w_out(2*cb->get_total_grid()), xi(cb->get_total_grid()),
          w_plus(cb->get_total_grid()), w_minus(cb->get_total_grid())
    {
    }

    /// The concentrations, free energy and residual of the fields w
    void step(const double* w)
    {
        const int M = cb->get_total_grid();

        // For the given fields find the polymer statistics
        solver->compute_propagators({{"A", &w[0]}, {"B", &w[M]}}, {});
        solver->compute_concentrations();
        solver->get_total_concentration("A", phi_a.data());
        solver->get_total_concentration("B", phi_b.data());
        if (relax_box)
        {
            solver->compute_stress();
            stress = solver->get_stress();
        }

        // The free energy
        for (int i = 0; i < M; i++)
        {
            w_minus[i] = (w[i]-w[i+M])/2;
            w_plus[i]  = (w[i]+w[i+M])/2;
        }
        energy = cb->inner_product(w_minus.data(), w_minus.data())/chi_n/cb->get_volume();
        energy -= cb->integral(w_plus.data())/cb->get_volume();
        for (int p = 0; p < molecules->get_n_polymer_types(); p++)
        {
            Polymer& pc = molecules->get_polymer(p);
            energy -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
        }

        // The output fields, with the pressure field
        for (int i = 0; i < M; i++)
        {
            xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
            w_out[i]   = chi_n*phi_b[i] + xi[i];
            w_out[i+M] = chi_n*phi_a[i] + xi[i];
        }
        cb->zero_mean(&w_out[0]);
        cb->zero_mean(&w_out[M]);

        // The residual: the "relative distance" between the input and output fields
        old_error_level = error_level;
        for (int i = 0; i < 2*M; i++)
            w_diff[i] = w_out[i] - w[i];
        error_level = sqrt(cb->multi_inner_product(2, w_diff.data(), w_diff.data())/
                           (cb->multi_inner_product(2, w, w) + 1.0));
        if (relax_box)
        {
            double stress_sq = 0.0;
            for (int d = 0; d < cb->get_dim(); d++)
                stress_sq += stress[d]*stress[d];
            error_level += sqrt(stress_sq);
        }
    }

    /// (integral of phi_A + phi_B)/V - 1 of the last step
    double mass_error()
    {
        return (cb->integral(phi_a.data()) + cb->integral(phi_b.data()))/cb->get_volume() - 1.0;
    }

    /// Anderson mixing of the fields w, and of the box lengths with box relaxation
    void mix(AndersonMixing<double>* am, double* w)
    {
        const int M = cb->get_total_grid();
        if (relax_box)
            for (int d = 0; d < cb->get_dim(); d++)
            {
                w[2*M+d] = cb->get_lx(d);
                w_diff[2*M+d] = -stress[d];
            }
        am->calculate_new_fields(w, w, w_diff.data(), old_error_level, error_level);
        if (relax_box)
        {
            std::vector<double> lx(w + 2*M, w + 2*M + cb->get_dim());
            cb->set_lx(lx);
            solver->update_laplacian_operator();
        }
    }

private:
    std::vector<double> w_out, xi, w_plus, w_minus;
};

#endif
