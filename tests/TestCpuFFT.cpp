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

/**
 * @file TestCpuFFT.cpp
 * @brief The periodic FFT of each CPU library (FftwFFT, MklFFT) for real and complex fields.
 *
 * 1. Stored values: the forward transform of fixed fields on the 5, 5x4 and 5x4x3
 *    grids against stored coefficients (the half spectrum of the last axis for a
 *    real field), and the backward transform of the coefficients against the
 *    fields. The largest error of a value must be at most 1e-7.
 * 2. The O(N^2) DFT (TestMakhoulDefinitions.h) on even and odd grids of one to
 *    three axes, fields of standard deviation 5: the forward transform of a field;
 *    the backward transform, with the 1/N, of its DFT (Hermitian for a real
 *    field); and for complex fields the backward transform of arbitrary
 *    coefficients. The largest error relative to the largest value must be at
 *    most 1e-13.
 *
 * Each transform gets a copy of its input, which it may not change: backward()
 * keeps its coefficients as well (FftwFFT copies them, since FFTW's complex-to-
 * real transform overwrites its input). A NaN fails (TestNanSafe.h).
 */

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ComputationBox.h"
#include "Exception.h"
#include "FFT.h"
#include "TestMakhoulDefinitions.h"
#include "TestNanSafe.h"
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif
#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

namespace {

constexpr double STORED_TOLERANCE = 1e-7;
constexpr double DEFINITION_TOLERANCE = 1e-13;

// Stored fields and their coefficients
// Real field on the 5 grid
const std::vector<double> real_1d_field = {
    0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
};
const std::vector<Complex> real_1d_coefficients = {
    {3.497839818,0}, {-0.7248383037,0.4777381112}, {-0.5654030903,-0.05431399883},
};

// Complex field on the 5 grid
const std::vector<Complex> complex_1d_field = {
    {0.2733181541,-0.0187024081}, {-0.1507659497,0.6894673618},
    {-0.0350475991,0.2143790085}, {-0.0175411299,0.4013954178},
    {0.8336831753,0.4648317909},
};
const std::vector<Complex> complex_1d_coefficients = {
    {0.9036466507,1.7513711709}, {0.6306119942,0.7863804708},
    {0.0144763523,-0.2002702411}, {-0.6053249486,-1.3242603125},
    {0.4231807219,-1.1067331286},
};

// Real field on the 5x4 grid
const std::vector<double> real_2d_field = {
    0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
    0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
    0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
    0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
};
const std::vector<Complex> real_2d_coefficients = {
    {10.6881904,0},                {0.7954998885,0.143345017},
    {-0.6668551075,0},             {0.4954041066,1.798776899},
    {-0.5050260775,0.04850456904}, {-0.4504167737,-1.947311157},
    {0.5003159972,-1.738407521},   {-0.675930056,-0.09881542923},
    {-0.9823256426,-0.6471590658}, {0.5003159972,1.738407521},
    {-0.6941925918,0.7499083742},  {-0.9823256426,0.6471590658},
    {0.4954041066,-1.798776899},   {-1.659282438,1.023353594},
    {-0.4504167737,1.947311157},
};

// Complex field on the 5x4 grid
const std::vector<Complex> complex_2d_field = {
    {0.7479851438,0.5113923184}, {-0.611924137,0.6688134066},
    {-0.2562048442,0.7156670162}, {-0.698469607,0.3809731847},
    {0.5071785071,-0.8416524347}, {-0.1415995644,0.9279043295},
    {0.1582071603,-0.2578680913}, {-0.9819601031,0.4541845031},
    {0.2625178548,-0.7306389533}, {0.0197493655,0.6360389997},
    {-0.6824309284,-0.0573651711}, {-0.588476528,-0.4983797445},
    {0.6192052822,-0.3934574836}, {0.582460527,0.9074145996},
    {0.4822692317,-0.4687287157}, {-0.6937104783,-0.9072499747},
    {-0.7862444092,0.3346206151}, {-0.7524654201,0.0442255985},
    {0.420599632,-0.7957530045}, {0.9435906198,0.9249318616},
};
const std::vector<Complex> complex_2d_coefficients = {
    {-1.4497226956,1.5550728596}, {4.0581392311,-1.3709348392},
    {4.3958879555,-5.5226406686}, {-1.6017349765,0.8595588959},
    {-1.1054099382,5.1776018019}, {-6.4498665458,-0.692090622},
    {0.0443508843,-1.1905937191}, {2.0370359645,-2.3202722334},
    {-0.6401006748,-0.5453888601}, {1.1391904384,-3.7814794759},
    {0.6683255063,0.3989231452}, {2.334602,1.8269050208},
    {0.0275726665,2.8851798905}, {5.9827303722,-0.6033125243},
    {1.7837980751,5.1801546701}, {-1.3263453051,-0.7671836679},
    {-0.9254065801,2.3117639377}, {1.7299575537,4.9937166227},
    {2.1185077968,2.0205202883}, {2.1381911477,-0.1876541545},
};

// Real field on the 5x4x3 grid
const std::vector<double> real_3d_field = {
    0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
    0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
    0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
    0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
    0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
    0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
    0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
    0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
    0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
    0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
    0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
    0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
};
const std::vector<Complex> real_3d_coefficients = {
    {30.0601362322000,0.000000000000000e+0}, {0.353642310400000,-0.656637882635999},
    {1.84441281060000,-2.74233574840000}, {-2.46150775700102,4.133457522749440e-2},
    {0.817180262600000,0.000000000000000e+0}, {-0.825032729800000,0.726831889225593},
    {1.84441281060000,2.74233574840000}, {0.732077908401022,-1.00081656417251},
    {-0.458084677221565,0.153852624198044}, {-0.144717578762292,-2.28803500866730},
    {0.705009791908085,-2.01933903816772}, {2.964764771912920e-2,-2.25308567091476},
    {-0.973844201363205,-2.06176730216391}, {-0.997469264485567,-0.602564694796768},
    {1.40462480063246,1.01369298185703}, {1.75359066116106,0.326376682455054},
    {-1.44138430512843,-1.31255361296267}, {-0.744097894016966,-0.480832819344072},
    {-2.15508603929787,-1.93166708304686}, {-0.438393767508181,-1.10483993164215},
    {0.995576356313205,0.780345054198534}, {-3.86510390713435,4.55728094860600},
    {2.709703615732217e-2,-2.321947897161802e-2}, {0.208459929713295,2.48927791839840},
    {-1.44138430512843,1.31255361296267}, {1.68993207144876,-0.218131499149520},
    {2.709703615732217e-2,2.321947897161802e-2}, {-2.04216524355430,2.34581169104130},
    {0.995576356313205,-0.780345054198534}, {2.469425255987412e-2,1.01595227927158},
    {-2.15508603929787,1.93166708304686}, {-0.744464511100393,-5.438522180051475e-2},
    {-0.458084677221565,-0.153852624198044}, {-0.713266212819504,1.64663971056941},
    {1.40462480063246,-1.01369298185703}, {-2.32489174723348,-1.41241859006073},
    {-0.973844201363205,2.06176730216391}, {0.857829657610043,-1.36198877135770},
    {0.705009791908085,2.01933903816772}, {-0.231601465597141,0.142526551270715}
};

// Complex field on the 5x4x3 grid
const std::vector<Complex> complex_3d_field = {
    {-0.1635032683,0.811069098}, {-0.0405400687,-0.363793969},
    {-0.0089541085,-0.7288956966}, {0.961865777,0.4060443942},
    {0.6226145044,0.8452404454}, {0.5411463282,-0.062767949},
    {-0.9693732485,-0.9499292596}, {0.5224303106,-0.5839273136},
    {-0.3447755023,-0.4197040933}, {0.4940856956,-0.4183871742},
    {-0.7004420545,-0.6882638505}, {0.8146353239,0.1013541518},
    {0.9620226651,-0.5889567939}, {0.3434239009,0.1706384337},
    {0.0913065632,0.3684459488}, {-0.5342741345,-0.7536072852},
    {0.8617852131,-0.7329472429}, {0.9497634322,0.7397160288},
    {-0.1661953773,-0.1572153018}, {-0.7370987097,0.0374939326},
    {0.2351758146,-0.4826984599}, {0.088672368,0.2273836194},
    {-0.2984519416,-0.7871247207}, {-0.7900221425,0.6089911949},
    {-0.122475026,0.2386835201}, {0.7145724817,0.4987837298},
    {0.1659934706,-0.7761889359}, {-0.0440122022,-0.0980303281},
    {-0.6529576556,-0.570251554}, {-0.331532641,-0.1474882433},
    {-0.0698514944,-0.4268140909}, {-0.9324044806,-0.5513195217},
    {-0.9822079035,0.6878822424}, {0.3643357687,0.9003829004},
    {-0.0106640643,-0.3151330242}, {-0.0357722986,-0.9447605119},
    {0.9309176567,-0.995065976}, {-0.0016633233,-0.0901041639},
    {0.0022255336,-0.9070624083}, {-0.6161077971,0.5855078935},
    {-0.2609157228,-0.4955934519}, {0.3622729861,-0.34366892},
    {-0.2638397466,-0.424452688}, {-0.7895719819,0.4714340937},
    {0.1605312528,-0.5276342035}, {-0.6727767096,0.1929227653},
    {-0.8211839048,0.3575046233}, {0.0437074112,-0.1773034289},
    {-0.9007289997,-0.1719159753}, {-0.5065689519,-0.0810386444},
    {-0.0334737879,-0.607627272}, {-0.3003188629,-0.7740487229},
    {-0.7292275949,-0.3523647693}, {0.1555184893,0.6752701102},
    {0.1060268292,0.9655767429}, {-0.3232550322,-0.9330414915},
    {0.0519991505,-0.5462597072}, {-0.1468951143,0.2767532169},
    {-0.0529965958,0.9490296469}, {-0.0385422168,0.5787464737},
};
const std::vector<Complex> complex_3d_coefficients = {
    {-3.8465457376,-7.2805319371}, {0.0671209716,6.0678667687},
    {0.5921491036,-2.2496331674}, {3.9938793799,-2.1303505803},
    {0.9984423789,2.3497705907}, {-4.2234916919,2.6213330847},
    {-2.293165026,-6.8467445123}, {5.3876129616,-2.215959756},
    {-3.8491723315,2.3308782561}, {7.8760503334,3.365510609},
    {4.5694542911,-5.0046915783}, {-0.7975382994,0.5183186955},
    {3.5608506716,-2.8906238018}, {-7.0706416083,0.1040548927},
    {2.5884910331,-4.3714854865}, {-0.2821599507,-4.8071723303},
    {0.2719059946,1.7212767637}, {8.1814926627,3.5084691437},
    {-0.2969939241,-1.2524200662}, {-5.124283101,2.879697747},
    {-7.2011456506,6.5441289628}, {-1.6785608773,2.883732263},
    {0.6909818315,1.8674170957}, {0.7193916316,-5.7615633736},
    {0.3318243741,-4.3343780728}, {2.6062816943,1.8441623495},
    {-6.6103852856,3.7123812707}, {0.6490228927,-0.7320743633},
    {-0.1429825299,0.5940679779}, {6.7373370623,-5.0595941231},
    {-4.8849176929,-3.7391719107}, {-6.4015000483,-11.6596332975},
    {-3.4210504008,-0.3811734284}, {-0.2425524228,0.7840047881},
    {0.6470993913,9.7001499487}, {-5.8493488021,-1.5748315385},
    {3.5095032591,0.064225527}, {4.3018280799,1.4700662421},
    {1.2121815767,1.0242642919}, {10.709832195,2.8396157821},
    {1.6385734643,2.8564140072}, {0.4573314638,11.9374969229},
    {-1.6042265395,2.1635228649}, {-3.7223344043,0.3141137862},
    {2.8378292867,1.4327775111}, {-5.7107224937,1.5730798876},
    {5.3002846436,7.7000643364}, {-1.4673497269,6.0998828162},
    {5.0903158774,4.1815022018}, {-0.4220263037,-2.9008994341},
    {-1.0648233697,3.2909842309}, {-1.2079007263,5.6029437623},
    {-2.0912153734,-1.6679287909}, {-1.2355439938,-0.2359937051},
    {-9.6138041196,-2.417192635}, {0.1480323383,5.9528542443},
    {1.2066988104,4.996561508}, {-8.3196783842,7.3401111696},
    {1.8959891548,5.7224996013}, {2.0880759053,-1.7820081353},
};

/// The periodic FFT of the grid nx of a library
template <typename T, int DIM>
std::unique_ptr<FFT<T>> make_fft(const std::string& library, const std::array<int, DIM>& nx)
{
#ifdef USE_CPU_FFTW
    if (library == "fftw")
        return std::make_unique<FftwFFT<T>>(std::vector<int>(nx.begin(), nx.end()));
#endif
#ifdef USE_CPU_MKL
    if (library == "mkl")
        return std::make_unique<MklFFT<T>>(std::vector<int>(nx.begin(), nx.end()));
#endif
    throw_with_line_number("No FFT library " + library + ".");
}

/// Number of grid points and of coefficients (the half spectrum of the last axis for a real field)
template <typename T>
std::pair<int, int> sizes(const std::vector<int>& nx)
{
    int total = 1;
    for (int n : nx)
        total *= n;
    if constexpr (std::is_same_v<T, double>)
        return {total, total / nx.back() * (nx.back() / 2 + 1)};
    else
        return {total, total};
}

/// The coefficients of the half spectrum (real fields) or of every mode (complex fields) of a full spectrum
template <typename T>
std::vector<Complex> stored_part(const std::vector<Complex>& full, const std::vector<int>& nx)
{
    if constexpr (!std::is_same_v<T, double>)
        return full;
    const int n_last = nx.back();
    const int half = n_last / 2 + 1;
    std::vector<Complex> part;
    for (size_t i = 0; i < full.size(); ++i)
        if (static_cast<int>(i % n_last) < half)
            part.push_back(full[i]);
    return part;
}

template <typename T>
std::vector<Complex> as_complex(const std::vector<T>& v)
{
    return std::vector<Complex>(v.begin(), v.end());
}

bool report(const std::string& what, double error, double tolerance)
{
    const bool ok = std::isfinite(error) && error <= tolerance;
    std::cout << "  " << what << ": " << error << (ok ? "  ok" : "  FAIL") << std::endl;
    return ok;
}

/// forward(field) against the stored coefficients and backward(coefficients) against the field
template <typename T, int DIM>
bool check_stored(const std::string& library, const std::string& name, const std::array<int, DIM>& nx,
                  const std::vector<T>& field, const std::vector<Complex>& coefficients)
{
    const std::unique_ptr<FFT<T>> fft = make_fft<T, DIM>(library, nx);
    const auto [n_grid, n_coefficients] = sizes<T>(std::vector<int>(nx.begin(), nx.end()));
    if (static_cast<int>(field.size()) != n_grid || static_cast<int>(coefficients.size()) != n_coefficients)
        throw_with_line_number("The stored values of " + name + " do not fit the grid.");

    std::vector<T> in = field;
    std::vector<Complex> result(n_coefficients);
    fft->forward(in.data(), reinterpret_cast<double*>(result.data()));
    bool ok = report(library + " " + name + " forward against the stored coefficients",
                     max_difference(result, coefficients), STORED_TOLERANCE);
    ok &= report(library + " " + name + " forward keeps its input",
                 max_difference(as_complex(in), as_complex(field)), 0.0);

    std::vector<Complex> coefficients_in = coefficients;
    std::vector<T> back(n_grid);
    fft->backward(reinterpret_cast<double*>(coefficients_in.data()), back.data());
    ok &= report(library + " " + name + " backward against the stored field",
                 max_difference(as_complex(back), as_complex(field)), STORED_TOLERANCE);
    ok &= report(library + " " + name + " backward keeps its input",
                 max_difference(coefficients_in, coefficients), 0.0);
    return ok;
}

/// The transforms of a grid against the O(N^2) DFT
template <typename T, int DIM>
bool check_definition(const std::string& library, const std::array<int, DIM>& nx_array, std::mt19937& rng)
{
    const std::vector<int> nx(nx_array.begin(), nx_array.end());
    const std::vector<BoundaryCondition> periodic(DIM, BoundaryCondition::PERIODIC);
    const std::unique_ptr<FFT<T>> fft = make_fft<T, DIM>(library, nx_array);
    const auto [n_grid, n_coefficients] = sizes<T>(nx);
    constexpr bool real_field = std::is_same_v<T, double>;
    std::normal_distribution<double> normal(0.0, 5.0);

    std::vector<T> field(n_grid);
    for (T& x : field)
    {
        if constexpr (real_field)
            x = normal(rng);
        else
            x = Complex(normal(rng), normal(rng));
    }
    std::string grid = std::to_string(nx[0]);
    for (int d = 1; d < DIM; ++d)
        grid += "x" + std::to_string(nx[d]);
    const std::string label = library + (real_field ? " real " : " complex ") + grid;

    // Forward: the DFT of the field
    const std::vector<Complex> spectrum = mixed_definition(as_complex(field), nx, periodic, true);
    const std::vector<Complex> expected = stored_part<T>(spectrum, nx);
    std::vector<T> in = field;
    std::vector<Complex> result(n_coefficients);
    fft->forward(in.data(), reinterpret_cast<double*>(result.data()));
    bool ok = report(label + " forward against the DFT", max_difference(result, expected) / max_abs(expected),
                     DEFINITION_TOLERANCE);

    // Backward: the field back from the coefficients of the DFT, Hermitian for a real field
    std::vector<Complex> coefficients = expected;
    std::vector<T> back(n_grid);
    fft->backward(reinterpret_cast<double*>(coefficients.data()), back.data());
    const std::vector<Complex> field_back = mixed_definition(spectrum, nx, periodic, false);
    ok &= report(label + " backward of the DFT against the inverse DFT",
                 max_difference(as_complex(back), field_back) / max_abs(field_back), DEFINITION_TOLERANCE);

    // Complex fields: arbitrary coefficients
    if constexpr (!real_field)
    {
        std::vector<Complex> arbitrary(n_coefficients);
        for (Complex& x : arbitrary)
            x = Complex(normal(rng), normal(rng));
        const std::vector<Complex> inverse = mixed_definition(arbitrary, nx, periodic, false);
        fft->backward(reinterpret_cast<double*>(arbitrary.data()), back.data());
        ok &= report(label + " backward of arbitrary coefficients against the inverse DFT",
                     max_difference(as_complex(back), inverse) / max_abs(inverse), DEFINITION_TOLERANCE);
    }
    return ok;
}

template <typename T>
bool check_definition_grids(const std::string& library, std::mt19937& rng)
{
    bool ok = true;
    for (int n : {1, 2, 3, 4, 5, 7, 8, 16, 17})
        ok &= check_definition<T, 1>(library, {n}, rng);
    for (std::array<int, 2> nx : std::vector<std::array<int, 2>>{{2, 2}, {4, 6}, {5, 7}, {3, 8}, {7, 2}, {6, 1}, {1, 9}, {12, 10}})
        ok &= check_definition<T, 2>(library, nx, rng);
    for (std::array<int, 3> nx : std::vector<std::array<int, 3>>{{2, 2, 2}, {4, 6, 8}, {5, 7, 9}, {3, 4, 5}, {2, 3, 1}, {1, 5, 6}, {9, 8, 7}, {8, 6, 4}})
        ok &= check_definition<T, 3>(library, nx, rng);
    return ok;
}

bool check_library(const std::string& library)
{
    std::cout << library << ": stored values" << std::endl;
    bool ok = true;
    ok &= check_stored<double, 1>(library, "real 5", {5}, real_1d_field, real_1d_coefficients);
    ok &= check_stored<Complex, 1>(library, "complex 5", {5}, complex_1d_field, complex_1d_coefficients);
    ok &= check_stored<double, 2>(library, "real 5x4", {5, 4}, real_2d_field, real_2d_coefficients);
    ok &= check_stored<Complex, 2>(library, "complex 5x4", {5, 4}, complex_2d_field, complex_2d_coefficients);
    ok &= check_stored<double, 3>(library, "real 5x4x3", {5, 4, 3}, real_3d_field, real_3d_coefficients);
    ok &= check_stored<Complex, 3>(library, "complex 5x4x3", {5, 4, 3}, complex_3d_field, complex_3d_coefficients);

    std::cout << library << ": the DFT" << std::endl;
    std::mt19937 rng(2026);
    ok &= check_definition_grids<double>(library, rng);
    ok &= check_definition_grids<Complex>(library, rng);
    return ok;
}

}  // namespace

int main()
{
    try
    {
        bool ok = true;
#ifdef USE_CPU_FFTW
        ok &= check_library("fftw");
#endif
#ifdef USE_CPU_MKL
        ok &= check_library("mkl");
#endif
        std::cout << (ok ? "TestCpuFFT passed" : "TestCpuFFT FAILED") << std::endl;
        return ok ? 0 : -1;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
