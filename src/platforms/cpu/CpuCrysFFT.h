/**
 * @file CpuCrysFFT.h
 * @brief Interface of the CPU crystallographic FFT engines.
 *
 * An engine (CrysFFTPmmm, CrysFFTObliqueZ, {Fftw,Mkl}CrysFFTRecursive3m)
 * holds the plans of one grid. The OpenMP threads of the propagator and stress
 * loops call one engine at once, each on its own thread-local scratch. The
 * k-space tables (Boltzmann factors, stress multipliers) depend only on the
 * grid, the cell and a coefficient, so the solver owns them
 * (CpuSolverPseudoBase): it builds them with build_boltzmann() and
 * build_multiplier() outside its loops and passes them to diffusion() and
 * apply_multiplier(), which only read them.
 */

#ifndef CPU_CRYS_FFT_H_
#define CPU_CRYS_FFT_H_

#include <array>
#include <string>
#include <vector>

#include "Exception.h"

class CpuCrysFFT
{
public:
    virtual ~CpuCrysFFT() = default;

    /// Points of the physical grid, the size of the fields the transforms take.
    virtual int get_M_physical() const = 0;

    /// Write the Boltzmann factors exp(-k² coeff) of `cell` [Lx, Ly, Lz, alpha,
    /// beta, gamma] into `table`, in place when it has the size already.
    virtual void build_boltzmann(std::vector<double>& table, const std::array<double, 6>& cell,
                                 double coeff) const = 0;

    /// q_out = exp(coeff ∇²) q_in on the physical grid, with the table of
    /// build_boltzmann(coeff); q_in is kept (q_out may be q_in).
    virtual void diffusion(const double* q_in, double* q_out, const std::vector<double>& table) const = 0;

    /// Whether the engine has stress multipliers (build_multiplier(), apply_multiplier()):
    /// Pmmm and recursive 3m.
    virtual bool has_multipliers() const { return false; }

    /// Write the stress multiplier exp(-k² coeff)·Σ_{d in axes} k_d² (axes: bit
    /// mask 1 = x, 2 = y, 4 = z; coeff 0: no bond factor) into `table`, as
    /// build_boltzmann() does. Pmmm and recursive 3m only (has_multipliers()).
    virtual void build_multiplier(std::vector<double>&, const std::array<double, 6>&, int, double) const
    {
        throw_with_line_number("This CrysFFT engine has no stress multipliers.");
    }

    /// q_out = multiplier · q_in with a table of build_multiplier(), as diffusion().
    virtual void apply_multiplier(const double*, double*, const std::vector<double>&) const
    {
        throw_with_line_number("This CrysFFT engine has no stress multipliers.");
    }

protected:
    /// Refuse a table of another size (another engine's, or one never built).
    static void check_table(const std::vector<double>& table, size_t size)
    {
        if (table.size() != size)
            throw_with_line_number("A CrysFFT table has " + std::to_string(table.size()) + " values, the engine needs "
                + std::to_string(size) + ".");
    }
};

#endif  // CPU_CRYS_FFT_H_
