/*******************************************************************************
 * The result of a CPU transform must not depend on where the caller's arrays
 * lie in memory. The full- and reduced-memory computations hand different
 * arrays to the same transform (stored propagators against recomputed ones,
 * and the stress transforms write into arrays allocated per call), and they
 * must agree bit for bit on the same backend.
 *
 * The periodic transforms run on the caller's arrays: MklFFT passes them to
 * DftiCompute, and MKL may pick a code path by their alignment; FftwFFT runs its
 * plans on them through FFTW's new-array execute functions when they have the
 * alignment of the planning arrays, and on per-thread scratch of that alignment
 * when they do not (an array 8 bytes off, as a packed field after an odd number
 * of grid points) or when FFTW cannot run on them (the input of the
 * complex-to-real transform, which FFTW overwrites). The DCT/DST of the walls
 * (MakhoulTransform) runs on per-thread scratch that starts on a 64-byte
 * boundary, with either library.
 *
 * Each periodic transform runs from and into arrays placed 0, 16, 32 and 48
 * bytes after a 64-byte boundary (the placements of heap arrays), and for FFTW
 * also 8, 24, 40 and 56 bytes after it, for real and complex fields on even and
 * odd grids, through the real-coefficient and the complex-coefficient interface.
 * Every placement must give the values of the first one, and the input must stay
 * unchanged. A failure prints which placements of the input (rows) and the
 * output (columns) give which result. FftwFFT must also give these values when
 * its input and output overlap. MKL runs one thread inside each transform, as
 * MklFactory sets it for the solvers. A build without either library skips its
 * part, and a build without both skips the test.
 ******************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ComputationBox.h"
#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

#if defined(USE_CPU_MKL) || defined(USE_CPU_FFTW)
namespace
{

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
    std::vector<int> offsets;
    // Result class of each (input, output) placement; class 0 is the first placement
    std::vector<std::vector<int>> result_class;
    int n_results = 1;
    double max_diff = 0.0;
    bool input_kept = true;
};

// Runs transform(in, out) for every placement, with the input copied from source
// each time (a backward transform may overwrite its input)
template <typename Transform>
Placements run_placements(const std::vector<int>& offsets, const std::vector<double>& source, int n_out,
                          Transform transform)
{
    const int n_in = static_cast<int>(source.size());
    const int n_offsets = static_cast<int>(offsets.size());
    PlacedArray in(n_in), out(n_out);
    std::vector<std::vector<double>> results;
    Placements p;
    p.offsets = offsets;
    p.result_class.assign(n_offsets, std::vector<int>(n_offsets, 0));

    for (int i = 0; i < n_offsets; i++)
    {
        for (int o = 0; o < n_offsets; o++)
        {
            double* in_ptr = in.at(offsets[i]);
            double* out_ptr = out.at(offsets[o]);
            for (int k = 0; k < n_in; k++)
                in_ptr[k] = source[k];
            for (int k = 0; k < n_out; k++)
                out_ptr[k] = 0.0;

            transform(in_ptr, out_ptr);

            for (int k = 0; k < n_in; k++)
                p.input_kept = p.input_kept && (in_ptr[k] == source[k]);
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

bool report(const std::string& label, const Placements& p, bool check_input)
{
    const int n_offsets = static_cast<int>(p.offsets.size());
    const bool ok = p.n_results == 1 && (!check_input || p.input_kept);
    std::cout << (ok ? "    [PASS] " : "    [FAIL] ") << label << ": " << n_offsets * n_offsets << " placements, "
              << p.n_results << (p.n_results == 1 ? " result" : " results");
    if (p.n_results != 1)
        std::cout << ", max |diff| " << std::scientific << std::setprecision(1) << p.max_diff
                  << std::defaultfloat;
    if (check_input && !p.input_kept)
        std::cout << ", the input changed";
    std::cout << std::endl;
    if (p.n_results != 1)
    {
        std::cout << "        result by input offset (rows) and output offset (columns), bytes:"
                  << std::endl << "              ";
        for (int o = 0; o < n_offsets; o++)
            std::cout << std::setw(4) << p.offsets[o];
        std::cout << std::endl;
        for (int i = 0; i < n_offsets; i++)
        {
            std::cout << "          " << std::setw(4) << p.offsets[i];
            for (int o = 0; o < n_offsets; o++)
                std::cout << std::setw(4) << p.result_class[i][o];
            std::cout << std::endl;
        }
    }
    return ok;
}

// The same values in place: input and output start at one 8-byte-aligned address
template <typename Transform>
bool report_in_place(const std::string& label, const std::vector<double>& source, const std::vector<double>& expected,
                     Transform transform)
{
    std::vector<double> data(std::max(source.size(), expected.size()) + 1, 0.0);
    bool ok = true;
    for (int shift : {0, 1})
    {
        double* ptr = data.data() + shift;
        std::copy(source.begin(), source.end(), ptr);
        transform(ptr, ptr);
        for (size_t k = 0; k < expected.size(); k++)
            ok = ok && (ptr[k] == expected[k]);
    }
    std::cout << (ok ? "    [PASS] " : "    [FAIL] ") << label << ": in place" << std::endl;
    return ok;
}

// Library: MklFFT or FftwFFT; the FFTW plans also run on arrays 8 bytes off and in place
template <template <typename, int> class Library, typename T, int DIM>
bool check_grid(const std::string& library, std::array<int, DIM> nx)
{
    constexpr bool IS_COMPLEX = std::is_same<T, std::complex<double>>::value;
    const bool fftw = library == "fftw";
    const std::vector<int> offsets = fftw ? std::vector<int>{0, 8, 16, 24, 32, 40, 48, 56}
                                          : std::vector<int>{0, 16, 32, 48};
    std::array<BoundaryCondition, DIM> bc;
    bc.fill(BoundaryCondition::PERIODIC);

    int n_grid = 1;
    std::string grid_name;
    for (int d = 0; d < DIM; d++)
    {
        n_grid *= nx[d];
        grid_name += (d > 0 ? "x" : "") + std::to_string(nx[d]);
    }
    // Doubles of the field and of its coefficients (FFT.h): a real field has
    // nx[0]...(nx[DIM-1]/2+1) complex coefficients, a complex field n_grid
    const int n_field = IS_COMPLEX ? 2 * n_grid : n_grid;
    const int n_coeff = IS_COMPLEX ? 2 * n_grid : 2 * (n_grid / nx[DIM - 1]) * (nx[DIM - 1] / 2 + 1);

    std::mt19937 rng(1234 + n_grid);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> field(n_field);
    for (double& v : field)
        v = normal(rng);

    Library<T, DIM> fft(nx, bc);
    const std::string label = library + (IS_COMPLEX ? " complex " : " real ") + "periodic " + grid_name;

    std::vector<double> coeff(n_coeff), back(n_field);
    fft.forward(reinterpret_cast<T*>(field.data()), coeff.data());
    fft.backward(coeff.data(), reinterpret_cast<T*>(back.data()));

    // MKL keeps the input of its out-of-place transforms (mkl_c2r, review of 2026-11-16);
    // FftwFFT copies the input that FFTW would overwrite
    bool ok = true;
    ok &= report(label + " forward", run_placements(offsets, field, n_coeff,
        [&](double* in, double* out) { fft.forward(reinterpret_cast<T*>(in), out); }), fftw);
    ok &= report(label + " backward", run_placements(offsets, coeff, n_field,
        [&](double* in, double* out) { fft.backward(in, reinterpret_cast<T*>(out)); }), fftw);
    ok &= report(label + " forward, complex interface", run_placements(offsets, field, n_coeff,
        [&](double* in, double* out)
        { fft.forward(reinterpret_cast<T*>(in), reinterpret_cast<std::complex<double>*>(out)); }), fftw);
    ok &= report(label + " backward, complex interface", run_placements(offsets, coeff, n_field,
        [&](double* in, double* out)
        { fft.backward(reinterpret_cast<std::complex<double>*>(in), reinterpret_cast<T*>(out)); }), fftw);
    if (fftw)
    {
        ok &= report_in_place(label + " forward", field, coeff,
            [&](double* in, double* out) { fft.forward(reinterpret_cast<T*>(in), out); });
        ok &= report_in_place(label + " backward", coeff, back,
            [&](double* in, double* out) { fft.backward(in, reinterpret_cast<T*>(out)); });
    }
    return ok;
}

template <template <typename, int> class Library, typename T>
bool check_type(const std::string& library)
{
    bool ok = true;
    // The grid of TestBlockMultiplicity, then odd, larger and lower-dimensional grids
    ok &= check_grid<Library, T, 3>(library, {12, 10, 8});
    ok &= check_grid<Library, T, 3>(library, {15, 9, 7});
    ok &= check_grid<Library, T, 3>(library, {32, 32, 32});
    ok &= check_grid<Library, T, 2>(library, {24, 20});
    ok &= check_grid<Library, T, 1>(library, {64});
    if (library == "fftw")
    {
        // Odd grids, whose packed real fields alternate between the two alignments
        ok &= check_grid<Library, T, 3>(library, {5, 5, 5});
        ok &= check_grid<Library, T, 2>(library, {9, 7});
        ok &= check_grid<Library, T, 1>(library, {33});
    }
    return ok;
}

}  // namespace
#endif

int main()
{
#if defined(USE_CPU_MKL) || defined(USE_CPU_FFTW)
    try
    {
        bool ok = true;
#ifdef USE_CPU_MKL
        mkl_set_num_threads(1);  // As MklFactory does: no threading inside a transform
        std::cout << "Periodic MKL transforms from and into arrays 0, 16, 32 and 48 bytes after a 64-byte boundary"
                  << std::endl;
        ok &= check_type<MklFFT, double>("mkl");
        ok &= check_type<MklFFT, std::complex<double>>("mkl");
#endif
#ifdef USE_CPU_FFTW
        std::cout << "Periodic FFTW transforms from and into arrays 0, 8, ..., 56 bytes after a 64-byte boundary"
                  << std::endl;
        ok &= check_type<FftwFFT, double>("fftw");
        ok &= check_type<FftwFFT, std::complex<double>>("fftw");
#endif
        std::cout << (ok ? "All placements give the same result" : "Some placements give different results")
                  << std::endl;
        return ok ? 0 : 1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return 1;
    }
#else
    std::cout << "This build has neither MKL nor FFTW: skipped" << std::endl;
    return 77;
#endif
}
