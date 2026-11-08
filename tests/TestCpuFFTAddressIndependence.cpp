/*******************************************************************************
 * The result of a CPU transform must not depend on where the caller's arrays
 * lie in memory. The full- and reduced-memory computations hand different
 * arrays to the same transform (stored propagators against recomputed ones,
 * and the stress transforms write into arrays allocated per call), and they
 * must agree bit for bit on the same backend.
 *
 * Each transform runs from and into arrays placed 0, 16, 32 and 48 bytes after
 * a 64-byte boundary (the placements of heap arrays), for real and complex
 * fields, periodic and non-periodic grids. Every placement must give the
 * values of the first one. A failure prints which placements of the input
 * (rows) and the output (columns) give which result.
 *
 * Periodic grids also check the complex-coefficient interface: the DCT/DST of
 * the walls (MakhoulTransform) runs through it on per-thread buffers, whose
 * placement differs from one thread to the next. MKL runs one thread inside
 * each transform, as MklFactory sets it for the solvers.
 ******************************************************************************/

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ComputationBox.h"
#include "FFT.h"
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif
#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

namespace
{

const std::array<int, 4> OFFSETS_BYTES = {0, 16, 32, 48};

// Doubles from a 64-byte boundary, with room for the largest offset
class PlacedArray
{
public:
    explicit PlacedArray(int n_doubles) : storage_(n_doubles + 16)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
        base_ = storage_.data() + ((64 - address % 64) % 64) / sizeof(double);
    }
    double* at(int offset_bytes) { return base_ + offset_bytes / static_cast<int>(sizeof(double)); }

private:
    std::vector<double> storage_;
    double* base_;
};

struct Placements
{
    // Result class of each (input, output) placement; class 0 is the first placement
    std::array<std::array<int, 4>, 4> result_class{};
    int n_results = 1;
    double max_diff = 0.0;
};

// Runs transform(in, out) for every placement, with the input copied from source
// each time (a backward transform may overwrite its input)
template <typename Transform>
Placements run_placements(const std::vector<double>& source, int n_out, Transform transform)
{
    const int n_in = static_cast<int>(source.size());
    PlacedArray in(n_in), out(n_out);
    std::vector<std::vector<double>> results;
    Placements p;

    for (int i = 0; i < 4; i++)
    {
        for (int o = 0; o < 4; o++)
        {
            double* in_ptr = in.at(OFFSETS_BYTES[i]);
            double* out_ptr = out.at(OFFSETS_BYTES[o]);
            for (int k = 0; k < n_in; k++)
                in_ptr[k] = source[k];
            for (int k = 0; k < n_out; k++)
                out_ptr[k] = 0.0;

            transform(in_ptr, out_ptr);

            std::vector<double> result(out_ptr, out_ptr + n_out);
            int found = -1;
            for (size_t r = 0; r < results.size() && found < 0; r++)
            {
                bool same = true;
                for (int k = 0; k < n_out && same; k++)
                    same = (result[k] == results[r][k]);
                if (same)
                    found = static_cast<int>(r);
            }
            if (found < 0)
            {
                if (!results.empty())
                    for (int k = 0; k < n_out; k++)
                    {
                        const double diff = std::abs(result[k] - results[0][k]);
                        if (std::isnan(diff) || diff > p.max_diff)  // a NaN stays
                            p.max_diff = diff;
                    }
                found = static_cast<int>(results.size());
                results.push_back(std::move(result));
            }
            p.result_class[i][o] = found;
        }
    }
    p.n_results = static_cast<int>(results.size());
    return p;
}

bool report(const std::string& label, const Placements& p)
{
    const bool ok = (p.n_results == 1);
    std::cout << (ok ? "    [PASS] " : "    [FAIL] ") << label << ": 16 placements, "
              << p.n_results << (p.n_results == 1 ? " result" : " results");
    if (!ok)
        std::cout << ", max |diff| " << std::scientific << std::setprecision(1) << p.max_diff
                  << std::defaultfloat;
    std::cout << std::endl;
    if (!ok)
    {
        std::cout << "        result by input offset (rows) and output offset (columns), bytes:"
                  << std::endl << "              ";
        for (int o = 0; o < 4; o++)
            std::cout << std::setw(4) << OFFSETS_BYTES[o];
        std::cout << std::endl;
        for (int i = 0; i < 4; i++)
        {
            std::cout << "          " << std::setw(4) << OFFSETS_BYTES[i];
            for (int o = 0; o < 4; o++)
                std::cout << std::setw(4) << p.result_class[i][o];
            std::cout << std::endl;
        }
    }
    return ok;
}

template <typename T, int DIM>
std::vector<std::pair<std::string, std::unique_ptr<FFT<T>>>> make_ffts(
    std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
{
    std::vector<std::pair<std::string, std::unique_ptr<FFT<T>>>> ffts;
#ifdef USE_CPU_FFTW
    ffts.emplace_back("fftw", std::make_unique<FftwFFT<T, DIM>>(nx, bc));
#endif
#ifdef USE_CPU_MKL
    ffts.emplace_back("mkl", std::make_unique<MklFFT<T, DIM>>(nx, bc));
#endif
    return ffts;
}

template <typename T, int DIM>
bool check_grid(std::array<int, DIM> nx, BoundaryCondition boundary, const std::string& bc_name)
{
    constexpr bool IS_COMPLEX = std::is_same<T, std::complex<double>>::value;
    std::array<BoundaryCondition, DIM> bc;
    bc.fill(boundary);

    int n_grid = 1;
    std::string grid_name;
    for (int d = 0; d < DIM; d++)
    {
        n_grid *= nx[d];
        grid_name += (d > 0 ? "x" : "") + std::to_string(nx[d]);
    }
    // Doubles of the field and of its coefficients (FFT.h): a periodic real field
    // has nx[0]...(nx[DIM-1]/2+1) complex coefficients, a complex field n_grid;
    // a non-periodic field has as many coefficients as grid points.
    const int n_field = IS_COMPLEX ? 2 * n_grid : n_grid;
    int n_coeff;
    if (boundary == BoundaryCondition::PERIODIC)
        n_coeff = IS_COMPLEX ? 2 * n_grid : 2 * (n_grid / nx[DIM - 1]) * (nx[DIM - 1] / 2 + 1);
    else
        n_coeff = n_field;

    std::mt19937 rng(1234 + n_grid);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> field(n_field);
    for (double& v : field)
        v = normal(rng);

    bool ok = true;
    for (const auto& entry : make_ffts<T, DIM>(nx, bc))
    {
        const std::string& backend = entry.first;
        FFT<T>* fft = entry.second.get();
        const std::string label = backend + (IS_COMPLEX ? " complex " : " real ") + bc_name + " " + grid_name;

        std::vector<double> coeff(n_coeff);
        fft->forward(reinterpret_cast<T*>(field.data()), coeff.data());

        ok &= report(label + " forward", run_placements(field, n_coeff,
            [&](double* in, double* out) { fft->forward(reinterpret_cast<T*>(in), out); }));
        ok &= report(label + " backward", run_placements(coeff, n_field,
            [&](double* in, double* out) { fft->backward(in, reinterpret_cast<T*>(out)); }));
        if (boundary == BoundaryCondition::PERIODIC)
        {
            ok &= report(label + " forward, complex interface", run_placements(field, n_coeff,
                [&](double* in, double* out)
                { fft->forward(reinterpret_cast<T*>(in), reinterpret_cast<std::complex<double>*>(out)); }));
            ok &= report(label + " backward, complex interface", run_placements(coeff, n_field,
                [&](double* in, double* out)
                { fft->backward(reinterpret_cast<std::complex<double>*>(in), reinterpret_cast<T*>(out)); }));
        }
    }
    return ok;
}

template <typename T>
bool check_type()
{
    bool ok = true;
    // The grid of TestBlockMultiplicity, then odd, larger and lower-dimensional grids
    ok &= check_grid<T, 3>({12, 10, 8}, BoundaryCondition::PERIODIC, "periodic");
    ok &= check_grid<T, 3>({12, 10, 8}, BoundaryCondition::REFLECTING, "reflecting");
    ok &= check_grid<T, 3>({12, 10, 8}, BoundaryCondition::ABSORBING, "absorbing");
    ok &= check_grid<T, 3>({15, 9, 7}, BoundaryCondition::PERIODIC, "periodic");
    ok &= check_grid<T, 3>({32, 32, 32}, BoundaryCondition::PERIODIC, "periodic");
    ok &= check_grid<T, 2>({24, 20}, BoundaryCondition::PERIODIC, "periodic");
    ok &= check_grid<T, 1>({64}, BoundaryCondition::PERIODIC, "periodic");
    return ok;
}

}  // namespace

int main()
{
    try
    {
#ifdef USE_CPU_MKL
        mkl_set_num_threads(1);  // As MklFactory does: no threading inside a transform
#endif
        std::cout << "Transforms from and into arrays 0, 16, 32 and 48 bytes after a 64-byte boundary"
                  << std::endl;
        bool ok = check_type<double>();
        ok &= check_type<std::complex<double>>();
        std::cout << (ok ? "All placements give the same result" : "Some placements give different results")
                  << std::endl;
        return ok ? 0 : 1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return 1;
    }
}
