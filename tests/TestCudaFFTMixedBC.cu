/**
 * @file TestCudaFFTMixedBC.cu
 * @brief Test the CUDA DCT-II/III and DST-II/III of boxes with reflecting and absorbing walls.
 *
 * The CUDA wall boxes transform through CudaMakhoulTransform. Taken with scale 1, its
 * forward transform is FFTW's unnormalized REDFT10 / RODFT10 per axis (DCT-II for
 * reflecting, DST-II for absorbing walls) and its backward transform REDFT01 / RODFT01,
 * which a backward scale of 1/(2N) per axis makes the inverse.
 *
 * Tests cover 1D, 2D, and 3D cases:
 * - odd lengths along reflecting and absorbing axes against direct sums and, with a CPU
 *   backend, against MakhoulTransform;
 * - round trips: backward(forward(x)) = x;
 * - the periodic transform of CudaFourierTransform (cuFFT): a round trip.
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>
#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <array>

#include "ComputationBox.h"
#include "CudaCommon.h"
#include "CudaFourierTransform.h"
#include "CudaMakhoulTransform.h"
#include "TestNanSafe.h"
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#include "FFTFactory.h"
#include "MakhoulTransform.h"
#endif

/// The wall transform of a real field on the device: FFTW's REDFT10/RODFT10 forward and
/// the normalized inverse backward
struct WallTransform
{
    CudaMakhoulTransform<double> transform;
    double backward_scale = 1.0;

    WallTransform(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc) : transform(nx, bc)
    {
        for (int n : nx)
            backward_scale /= 2.0 * n;
    }
    void forward(const double* d_in, double* d_out) { transform.forward(d_in, d_out, 1.0, 0); }
    void backward(const double* d_in, double* d_out) { transform.backward(d_in, nullptr, d_out, backward_scale, 0); }
};

#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
/// The forward transform (scale 1) of x on the CPU, MakhoulTransform over the real FFT of a backend
std::vector<double> cpu_forward(const std::vector<int>& nx, const std::vector<BoundaryCondition>& bc,
                                const std::vector<double>& x)
{
#ifdef USE_CPU_FFTW
    const FFTBackend backend = FFTBackend::FFTW;
#else
    const FFTBackend backend = FFTBackend::MKL;
#endif
    MakhoulTransform cpu(nx, bc, createRealFFT(nx, backend));
    std::vector<double> out(x.size());
    cpu.forward(x.data(), out.data(), 1.0);
    return out;
}
#endif

int main()
{
    try
    {
        const int N = 12;
        double error;
        std::vector<double> diff_sq(N);

        // Test input data
        std::vector<double> data_init(N);
        for (int i = 0; i < N; ++i)
            data_init[i] = i + 1;  // [1, 2, 3, ..., 12]

        std::vector<double> data_r(N);
        std::vector<double> data_k(N);

        // GPU buffers
        CudaDeviceBuffer<double> d_data_r(N);
        CudaDeviceBuffer<double> d_data_k(N);

        //=======================================================================
        // Test 0: Odd lengths along the walls. The forward transform is the
        // unnormalized DCT-II / DST-II per axis (FFTW REDFT10 / RODFT10), which
        // a direct sum checks; an odd absorbing axis is checked against the CPU.
        //=======================================================================
        {
            std::cout << "Test 0: odd wall axes (1D N=11, 3D 6x4x3 R/A/R, 2D 6x5 R/A) against direct sums" << std::endl;
            auto basis = [](BoundaryCondition bc, int k, int n, int len) {
                const double pi = std::numbers::pi;
                return (bc == BoundaryCondition::REFLECTING) ? 2.0 * std::cos(pi * k * (2 * n + 1) / (2.0 * len))
                                                             : 2.0 * std::sin(pi * (k + 1) * (2 * n + 1) / (2.0 * len));
            };
            std::mt19937 gen0(7);
            std::normal_distribution<double> normal(0.0, 5.0);

            // 1D, N = 11
            {
                const int N1 = 11;
                std::vector<double> x(N1), coeff(N1), back(N1), ref(N1, 0.0);
                for (double& v : x)
                    v = normal(gen0);
                for (int k = 0; k < N1; ++k)
                    for (int n = 0; n < N1; ++n)
                        ref[k] += x[n] * basis(BoundaryCondition::REFLECTING, k, n, N1);
                WallTransform fft_odd({N1}, {BoundaryCondition::REFLECTING});
                CudaDeviceBuffer<double> d_x(N1), d_c(N1);
                CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(double) * N1, cudaMemcpyHostToDevice));
                fft_odd.forward(d_x, d_c);
                CUDA_CHECK(cudaMemcpy(coeff.data(), d_c, sizeof(double) * N1, cudaMemcpyDeviceToHost));
                fft_odd.backward(d_c, d_x);
                CUDA_CHECK(cudaMemcpy(back.data(), d_x, sizeof(double) * N1, cudaMemcpyDeviceToHost));
                double err_fwd = 0.0, err_back = 0.0, scale = 0.0;
                for (int i = 0; i < N1; ++i)
                {
                    err_fwd = max_or_nan(err_fwd, std::abs(coeff[i] - ref[i]));
                    err_back = max_or_nan(err_back, std::abs(back[i] - x[i]));
                    scale = std::max(scale, std::abs(ref[i]));
                }
                std::cout << "  1D N=11: forward vs direct " << err_fwd / scale << ", round trip " << err_back << std::endl;
                if (!(err_fwd / scale < 1e-12) || !(err_back < 1e-12))
                {
                    std::cout << "  FAILED!" << std::endl;
                    return -1;
                }
            }

            // 3D, 6 (reflecting) x 4 (absorbing) x 3 (reflecting), and 2D, 6 (reflecting) x 5 (absorbing)
            for (const auto& [n3, bc3, label] : std::vector<std::tuple<std::vector<int>, std::vector<BoundaryCondition>, std::string>>{
                     {{6, 4, 3}, {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING, BoundaryCondition::REFLECTING},
                      "3D 6x4x3 R/A/R"},
                     {{1, 6, 5}, {BoundaryCondition::REFLECTING, BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING},
                      "2D 6x5 R/A"}})
            {
                // The 2D box is the 3D sum with one point along the first axis
                const std::vector<int> nx = (n3[0] == 1) ? std::vector<int>{n3[1], n3[2]} : n3;
                const std::vector<BoundaryCondition> bc = (n3[0] == 1) ? std::vector<BoundaryCondition>{bc3[1], bc3[2]} : bc3;
                const int M3 = n3[0] * n3[1] * n3[2];
                std::vector<double> x(M3), coeff(M3), back(M3), ref(M3, 0.0);
                for (double& v : x)
                    v = normal(gen0);
                for (int k0 = 0; k0 < n3[0]; ++k0)
                    for (int k1 = 0; k1 < n3[1]; ++k1)
                        for (int k2 = 0; k2 < n3[2]; ++k2)
                        {
                            double sum = 0.0;
                            for (int i0 = 0; i0 < n3[0]; ++i0)
                                for (int i1 = 0; i1 < n3[1]; ++i1)
                                    for (int i2 = 0; i2 < n3[2]; ++i2)
                                        sum += x[(i0 * n3[1] + i1) * n3[2] + i2]
                                             * (n3[0] == 1 ? 1.0 : basis(bc3[0], k0, i0, n3[0]))
                                             * basis(bc3[1], k1, i1, n3[1]) * basis(bc3[2], k2, i2, n3[2]);
                            ref[(k0 * n3[1] + k1) * n3[2] + k2] = sum;
                        }
                WallTransform fft_odd3(nx, bc);
                CudaDeviceBuffer<double> d_x(M3), d_c(M3);
                CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(double) * M3, cudaMemcpyHostToDevice));
                fft_odd3.forward(d_x, d_c);
                CUDA_CHECK(cudaMemcpy(coeff.data(), d_c, sizeof(double) * M3, cudaMemcpyDeviceToHost));
                fft_odd3.backward(d_c, d_x);
                CUDA_CHECK(cudaMemcpy(back.data(), d_x, sizeof(double) * M3, cudaMemcpyDeviceToHost));
                double err_fwd = 0.0, err_back = 0.0, scale = 0.0;
                for (int i = 0; i < M3; ++i)
                {
                    err_fwd = max_or_nan(err_fwd, std::abs(coeff[i] - ref[i]));
                    err_back = max_or_nan(err_back, std::abs(back[i] - x[i]));
                    scale = std::max(scale, std::abs(ref[i]));
                }
                std::cout << "  " << label << ": forward vs direct " << err_fwd / scale << ", round trip " << err_back;
                bool ok = err_fwd / scale < 1e-12 && err_back < 1e-12;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
                // The CPU MakhoulTransform of the same field (an odd absorbing axis was refused on CUDA before)
                const std::vector<double> cpu = cpu_forward(nx, bc, x);
                double err_cpu = 0.0;
                for (int i = 0; i < M3; ++i)
                    err_cpu = max_or_nan(err_cpu, std::abs(coeff[i] - cpu[i]));
                std::cout << ", vs CPU " << err_cpu / scale;
                ok = ok && err_cpu / scale < 1e-13;
#endif
                std::cout << std::endl;
                if (!ok)
                {
                    std::cout << "  FAILED!" << std::endl;
                    return -1;
                }
            }
            std::cout << "  PASSED!" << std::endl;
        }

        //=======================================================================
        // Test 1: DCT-II / DCT-III (Reflecting BC) Round-trip
        //=======================================================================
        std::cout << "\nTest 1: DCT-II/III (Reflecting BC) Round-trip" << std::endl;

        WallTransform fft_dct({N}, {BoundaryCondition::REFLECTING});

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DCT-II
        fft_dct.forward(d_data_r, d_data_k);

        // Backward DCT-III (inverse) - should recover original
        fft_dct.backward(d_data_k, d_data_r);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: DST-II / DST-III (Absorbing BC) Round-trip
        //=======================================================================
        std::cout << "\nTest 2: DST-II/III (Absorbing BC) Round-trip" << std::endl;

        WallTransform fft_dst({N}, {BoundaryCondition::ABSORBING});

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DST-II
        fft_dst.forward(d_data_r, d_data_k);

        // Backward DST-III (inverse) - should recover original
        fft_dst.backward(d_data_k, d_data_r);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 3: Random data round-trip test
        //=======================================================================
        std::cout << "\nTest 3: Random data round-trip" << std::endl;

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);

        std::vector<double> random_data(N);
        for (int i = 0; i < N; ++i)
            random_data[i] = dist(gen);

        // DCT round-trip
        CUDA_CHECK(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dct.forward(d_data_r, d_data_k);
        fft_dct.backward(d_data_k, d_data_r);

        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  DCT Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }

        // DST round-trip
        CUDA_CHECK(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dst.forward(d_data_r, d_data_k);
        fft_dst.backward(d_data_k, d_data_r);

        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  DST Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 4: 2D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 4: 2D Mixed BC Round-trip" << std::endl;

        const int NX = 8, NY = 6;
        const int M = NX * NY;
        std::vector<double> data2d_init(M), data2d_r(M), data2d_k(M);

        for (int i = 0; i < M; ++i)
            data2d_init[i] = dist(gen);

        CudaDeviceBuffer<double> d_data2d_r(M);
        CudaDeviceBuffer<double> d_data2d_k(M);

        WallTransform fft_2d({NX, NY}, {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING});

        CUDA_CHECK(cudaMemcpy(d_data2d_r, data2d_init.data(),
                                   sizeof(double) * M, cudaMemcpyHostToDevice));

        fft_2d.forward(d_data2d_r, d_data2d_k);
        fft_2d.backward(d_data2d_k, d_data2d_r);

        CUDA_CHECK(cudaMemcpy(data2d_r.data(), d_data2d_r,
                                   sizeof(double) * M, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_2d(M);
        for (int i = 0; i < M; ++i)
            diff_sq_2d[i] = std::pow(data2d_r[i] - data2d_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq_2d));
        std::cout << "  2D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 5: 3D Mixed BC test (the reflecting z axis has an odd length)
        //=======================================================================
        std::cout << "\nTest 5: 3D Mixed BC Round-trip" << std::endl;
        {
        const int NX3 = 6, NY3 = 4, NZ3 = 3;  // Only dim 2 is odd
        const int M3 = NX3 * NY3 * NZ3;
        std::vector<double> data3d_init(M3), data3d_r(M3), data3d_k(M3);

        for (int i = 0; i < M3; ++i)
            data3d_init[i] = dist(gen);

        CudaDeviceBuffer<double> d_data3d_r(M3);
        CudaDeviceBuffer<double> d_data3d_k(M3);

        WallTransform fft_3d({NX3, NY3, NZ3},
                             {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING, BoundaryCondition::REFLECTING});

        CUDA_CHECK(cudaMemcpy(d_data3d_r, data3d_init.data(),
                                   sizeof(double) * M3, cudaMemcpyHostToDevice));

        fft_3d.forward(d_data3d_r, d_data3d_k);
        fft_3d.backward(d_data3d_k, d_data3d_r);

        CUDA_CHECK(cudaMemcpy(data3d_r.data(), d_data3d_r,
                                   sizeof(double) * M3, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_3d(M3);
        for (int i = 0; i < M3; ++i)
            diff_sq_3d[i] = std::pow(data3d_r[i] - data3d_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq_3d));
        std::cout << "  3D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;
        }

        //=======================================================================
        // Periodic transform of CudaFourierTransform (cuFFT D2Z/Z2D): round trip
        //=======================================================================
        {
            std::cout << "\nPeriodic forward/backward round trip" << std::endl;
            const std::vector<int> nx_p = {8, 6, 10};
            const int MP = nx_p[0] * nx_p[1] * nx_p[2];
            ComputationBox<double> cb(nx_p, {1.0, 1.0, 1.0}, std::vector<std::string>(6, "periodic"));
            CudaFourierTransform<double> fft_p(&cb, std::vector<BoundaryCondition>(3, BoundaryCondition::PERIODIC));

            std::vector<double> data_p(MP), data_p_back(MP);
            for (int i = 0; i < MP; ++i)
                data_p[i] = std::sin(0.37 * i) + 0.1 * i / MP;
            std::vector<double> coefficients(2 * nx_p[0] * nx_p[1] * (nx_p[2] / 2 + 1));
            fft_p.forward(data_p.data(), 1, coefficients.data());
            fft_p.backward(coefficients.data(), 1, data_p_back.data());

            std::vector<double> diff_p(MP);
            for (int i = 0; i < MP; ++i)
                diff_p[i] = std::abs(data_p_back[i] - data_p[i]);
            error = max_or_nan(diff_p);
            std::cout << "  Round-trip Error: " << error << std::endl;
            if (!std::isfinite(error) || error > 1e-12)
            {
                std::cout << "  FAILED!" << std::endl;
                return -1;
            }
            std::cout << "  PASSED!" << std::endl;
        }

        std::cout << "\nAll tests passed!" << std::endl;

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
