/**
 * @file TestCudaDCT.cu
 * @brief Test CUDA DCT types 2 and 3 against FFTW reference.
 *
 * Compares CudaDCT output with FFTW REDFT10/01 for the DCT types of CudaRealTransform.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <fftw3.h>

#include "../src/platforms/cuda/CudaRealTransform.h"
#include "TestNanSafe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Test 1D DCT-2 (FFTW_REDFT10)
//==============================================================================
bool test_dct2_1d(int N)
{
    printf("Testing 1D DCT-2: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * cos(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDCT dct(N, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 1D DCT-3 (FFTW_REDFT01)
//==============================================================================
bool test_dct3_1d(int N)
{
    printf("Testing 1D DCT-3: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * i / N) + 0.5 * cos(2 * M_PI * i / N);
    }

    // === FFTW DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDCT dct(N, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-2
//==============================================================================
bool test_dct2_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-2: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5);
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-3
//==============================================================================
bool test_dct3_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-3: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double r2 = ix * ix + iy * iy;
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-2
//==============================================================================
bool test_dct2_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-2: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5) + (iz + 0.5) * (iz + 0.5);
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-3
//==============================================================================
bool test_dct3_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-3: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double r2 = ix * ix + iy * iy + iz * iz;
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test DCT-2/DCT-3 inverse pair
//==============================================================================
bool test_dct23_inverse(int N)
{
    printf("Testing DCT-2/DCT-3 inverse pair: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * cos(2 * M_PI * (i + 0.5) / N);
    }

    // FFTW: DCT-2 then DCT-3 should recover input (up to scale)
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan2 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);

    fftw_plan plan3 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan3);
    fftw_destroy_plan(plan3);

    // Normalization: 2N for round-trip
    double norm = 1.0 / (2.0 * N);
    for (int i = 0; i < N; ++i) {
        h_result[i] *= norm;
    }

    // Check recovery
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_input[i] - h_result[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_input);
    double rel_err = max_err / std::abs(max_val);

    printf("  FFTW DCT-2 + DCT-3 round-trip: rel_err = %.6e\n", rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D Mixed DCT (vs FFTW)
//==============================================================================
bool test_mixed_2d(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y)
{
    printf("Testing 2D Mixed (%s, %s): %d x %d\n",
           getTransformName(type_x), getTransformName(type_y), Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            h_input[idx] = sin(M_PI * (ix + 0.5) / Nx) * cos(M_PI * (iy + 0.5) / Ny)
                         + 0.5 * cos(2 * M_PI * (ix + 0.5) / Nx) * sin(2 * M_PI * (iy + 0.5) / Ny);
        }
    }

    // FFTW reference: separate 1D transforms per dimension
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Y dimension first (contiguous)
    fftw_r2r_kind kind_y = (type_y == CUDA_DCT_2) ? FFTW_REDFT10 : FFTW_REDFT01;
    for (int ix = 0; ix < Nx; ++ix) {
        fftw_plan plan = fftw_plan_r2r_1d(Ny, &h_fftw[ix * Ny], &h_fftw[ix * Ny], kind_y, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // X dimension (need transpose for FFTW or do row-by-row)
    std::vector<double> h_col(Nx);
    fftw_r2r_kind kind_x = (type_x == CUDA_DCT_2) ? FFTW_REDFT10 : FFTW_REDFT01;
    for (int iy = 0; iy < Ny; ++iy) {
        // Extract column
        for (int ix = 0; ix < Nx; ++ix) {
            h_col[ix] = h_fftw[ix * Ny + iy];
        }
        fftw_plan plan = fftw_plan_r2r_1d(Nx, h_col.data(), h_col.data(), kind_x, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        // Put back
        for (int ix = 0; ix < Nx; ++ix) {
            h_fftw[ix * Ny + iy] = h_col[ix];
        }
    }

    // CUDA
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, type_x, type_y);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D Mixed DCT (vs FFTW)
//==============================================================================
bool test_mixed_3d(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z)
{
    printf("Testing 3D Mixed (%s, %s, %s): %d x %d x %d\n",
           getTransformName(type_x), getTransformName(type_y), getTransformName(type_z), Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                h_input[idx] = sin(M_PI * (ix + 0.5) / Nx)
                             * cos(M_PI * (iy + 0.5) / Ny)
                             * sin(M_PI * (iz + 0.5) / Nz);
            }
        }
    }

    // FFTW reference
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Z dimension (contiguous)
    fftw_r2r_kind kind_z = (type_z == CUDA_DCT_2) ? FFTW_REDFT10 : FFTW_REDFT01;
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int offset = (ix * Ny + iy) * Nz;
            fftw_plan plan = fftw_plan_r2r_1d(Nz, &h_fftw[offset], &h_fftw[offset], kind_z, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }

    // Y dimension
    fftw_r2r_kind kind_y = (type_y == CUDA_DCT_2) ? FFTW_REDFT10 : FFTW_REDFT01;
    std::vector<double> h_tmp_y(Ny);
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iz = 0; iz < Nz; ++iz) {
            for (int iy = 0; iy < Ny; ++iy) {
                h_tmp_y[iy] = h_fftw[(ix * Ny + iy) * Nz + iz];
            }
            fftw_plan plan = fftw_plan_r2r_1d(Ny, h_tmp_y.data(), h_tmp_y.data(), kind_y, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
            for (int iy = 0; iy < Ny; ++iy) {
                h_fftw[(ix * Ny + iy) * Nz + iz] = h_tmp_y[iy];
            }
        }
    }

    // X dimension
    fftw_r2r_kind kind_x = (type_x == CUDA_DCT_2) ? FFTW_REDFT10 : FFTW_REDFT01;
    std::vector<double> h_tmp_x(Nx);
    for (int iy = 0; iy < Ny; ++iy) {
        for (int iz = 0; iz < Nz; ++iz) {
            for (int ix = 0; ix < Nx; ++ix) {
                h_tmp_x[ix] = h_fftw[(ix * Ny + iy) * Nz + iz];
            }
            fftw_plan plan = fftw_plan_r2r_1d(Nx, h_tmp_x.data(), h_tmp_x.data(), kind_x, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
            for (int ix = 0; ix < Nx; ++ix) {
                h_fftw[(ix * Ny + iy) * Nz + iz] = h_tmp_x[ix];
            }
        }
    }

    // CUDA
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, type_x, type_y, type_z);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = max_or_nan(max_err, err);
    }

    double max_val = max_abs_or_nan(h_fftw);
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Main
//==============================================================================
// Supported lengths: DCT-2/3 accept odd lengths (verified above against FFTW).
// Unsupported lengths must be refused in the constructor, not computed wrongly.
bool test_odd_length_rejected()
{
    printf("Testing length checks: supported lengths accepted, unsupported rejected\n");
    bool ok = true;
    try
    {
        CudaDCT a(15, CUDA_DCT_2); CudaDCT3D b(6, 5, 7, CUDA_DCT_3);
    }
    catch (const std::exception& e) { printf("  a supported length was rejected: %s\n", e.what()); ok = false; }
    // The 1D DCT-2/3 stage N and N + 2 values in the 48 KB of shared memory of a block: 6144 and
    // 6142 points fit, one more does not and must be refused
    try { CudaDCT f(6144, CUDA_DCT_2); CudaDCT g(6142, CUDA_DCT_3); }
    catch (const std::exception& e) { printf("  a 1D length within shared memory was rejected: %s\n", e.what()); ok = false; }
    for (auto [N, type] : {std::pair{6145, CUDA_DCT_2}, std::pair{6143, CUDA_DCT_3}})
    {
        bool threw = false;
        try { CudaDCT t(N, type); }
        catch (const std::exception&) { threw = true; }
        if (!threw) { printf("  1D DCT of %d points beyond shared memory was NOT rejected\n", N); ok = false; }
    }
    printf("  %s\n\n", ok ? "PASSED!" : "FAILED!");
    return ok;
}

int main()
{
    std::cout << "========== CUDA DCT vs FFTW Test ==========\n\n";

    bool all_passed = true;

    // Test 1D DCT-2
    std::cout << "--- 1D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_1d(16);
    all_passed &= test_dct2_1d(32);
    all_passed &= test_dct2_1d(64);
    all_passed &= test_dct2_1d(15);    // odd length: slot N/2 is an ordinary pair
    all_passed &= test_dct2_1d(31);
    all_passed &= test_dct2_1d(5001);  // beyond 4096: the shared memory holds N values, not 2^ceil(log2 N)
    all_passed &= test_dct2_1d(6144);

    // Test 1D DCT-3
    std::cout << "--- 1D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_1d(16);
    all_passed &= test_dct3_1d(32);
    all_passed &= test_dct3_1d(64);
    all_passed &= test_dct3_1d(15);
    all_passed &= test_dct3_1d(5001);
    all_passed &= test_dct3_1d(6142);

    // Test 2D DCT-2
    std::cout << "--- 2D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_2d(8, 8);
    all_passed &= test_dct2_2d(16, 16);
    all_passed &= test_dct2_2d(32, 32);
    all_passed &= test_dct2_2d(6, 5);     // odd length along one axis
    all_passed &= test_dct2_2d(5, 6);
    all_passed &= test_dct2_2d(7, 9);

    // Test 2D DCT-3
    std::cout << "--- 2D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_2d(8, 8);
    all_passed &= test_dct3_2d(16, 16);
    all_passed &= test_dct3_2d(32, 32);
    all_passed &= test_dct3_2d(7, 9);

    // Test 3D DCT-2
    std::cout << "--- 3D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_3d(8, 8, 8);
    all_passed &= test_dct2_3d(16, 16, 16);
    all_passed &= test_dct2_3d(32, 32, 32);
    all_passed &= test_dct2_3d(6, 6, 5);  // odd physical z (logical 12x12x10 CrysFFT grid)
    all_passed &= test_dct2_3d(6, 5, 6);
    all_passed &= test_dct2_3d(5, 6, 6);
    all_passed &= test_dct2_3d(5, 7, 9);

    // Test 3D DCT-3
    std::cout << "--- 3D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_3d(8, 8, 8);
    all_passed &= test_dct3_3d(16, 16, 16);
    all_passed &= test_dct3_3d(32, 32, 32);
    all_passed &= test_dct3_3d(5, 7, 9);

    all_passed &= test_odd_length_rejected();

    // Test round-trip
    std::cout << "--- Round-trip Tests ---\n\n";
    all_passed &= test_dct23_inverse(32);
    all_passed &= test_dct23_inverse(15);

    // Test 2D Mixed DCT
    std::cout << "--- 2D Mixed DCT Tests ---\n\n";
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_2, CUDA_DCT_3);
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_3, CUDA_DCT_2);

    // Test 3D Mixed DCT
    std::cout << "--- 3D Mixed DCT Tests ---\n\n";
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_2, CUDA_DCT_3, CUDA_DCT_2);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_3, CUDA_DCT_2, CUDA_DCT_3);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_3, CUDA_DCT_3, CUDA_DCT_2);

    if (all_passed)
        std::cout << "=== All tests PASSED! ===\n";
    else
        std::cout << "=== Some tests FAILED! ===\n";

    return all_passed ? 0 : 1;
}
