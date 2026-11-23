/**
 * @file CudaCrysFFTBase.h
 * @brief Interface of the CUDA crystallographic FFT engines.
 *
 * An engine (CudaCrysFFTPmmm, CudaCrysFFTRecursive3m, CudaCrysFFTObliqueZ) holds
 * the plans and work buffers of one stream; a solver keeps one per stream
 * (clone()). The k-space tables (Boltzmann factors, stress multipliers) depend
 * only on the grid, the cell and their coefficient, so the solver owns one table
 * per coefficient for all streams: it builds them with build_boltzmann() and
 * build_multiplier() outside its propagator and stress loops and passes them to
 * diffusion(), which only reads them.
 */

#ifndef CUDA_CRYS_FFT_BASE_H_
#define CUDA_CRYS_FFT_BASE_H_

#include <array>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include "CudaCommon.h"
#include "Exception.h"

class CudaCrysFFTBase
{
public:
    virtual ~CudaCrysFFTBase() = default;

    /// An object of the same grid for another stream, with plans and work buffers of
    /// its own. The recursive 3m and ObliqueZ clones share the read-only twiddle
    /// factors of the original; a Pmmm clone is a new object (it has no such data).
    virtual std::unique_ptr<CudaCrysFFTBase> clone() const = 0;

    /// Write the Boltzmann factors exp(-k² coeff) of `cell` into `table`
    /// (allocated when its size differs). The upload runs on `stream` and is
    /// complete on return; no kernel may be reading the table meanwhile.
    virtual void build_boltzmann(CudaDeviceBuffer<double>& table, const std::array<double, 6>& cell,
                                 double coeff, cudaStream_t stream) const = 0;

    /// d_q_out = exp(coeff ∇²) d_q_in on the physical grid with the table of
    /// build_boltzmann(coeff), or the stress multiplier times d_q_in in k-space with
    /// a table of build_multiplier(), on `stream`. d_q_out may alias d_q_in;
    /// otherwise d_q_in is kept. Both must be aligned to a cufftDoubleComplex, as
    /// the work buffers and propagators are (the recursive 3m engine runs cuFFT on
    /// them and refuses another).
    virtual void diffusion(const double* d_q_in, double* d_q_out, const CudaDeviceBuffer<double>& table,
                           cudaStream_t stream) = 0;

    /// Write the stress multiplier exp(-k² coeff)·Σ_{d in axes} k_d² (axes: bit
    /// mask 1 = x, 2 = y, 4 = z; coeff 0: no bond factor) into `table`, as
    /// build_boltzmann() does, for diffusion(). Pmmm and recursive 3m only.
    virtual void build_multiplier(CudaDeviceBuffer<double>&, const std::array<double, 6>&, int, double,
                                  cudaStream_t) const
    {
        throw_with_line_number("This CrysFFT engine has no stress multipliers.");
    }

protected:
    /// Copy `h` into `table` (allocated when its size differs) on `stream`, complete on return.
    static void upload_table(CudaDeviceBuffer<double>& table, const std::vector<double>& h, cudaStream_t stream)
    {
        if (table.size() != h.size())
            table.allocate(h.size());
        CUDA_CHECK(cudaMemcpyAsync(table, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));   // h may go out of scope; other streams read the table next
    }
};

#endif  // CUDA_CRYS_FFT_BASE_H_
