/**
 * @file TestCudaDST.cu
 * @brief Test CUDA DST types 2 and 3 against FFTW reference.
 *
 * Compares the 1D, 2D and 3D DSTs of CudaRealTransform with FFTW RODFT10/01.
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
// Test 1D DST-2 (FFTW_RODFT10)
//==============================================================================
bool test_dst2_1d(int N)
{
    printf("Testing 1D DST-2: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * sin(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaRealTransform1D dst(N, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 1D DST-3 (FFTW_RODFT01)
//==============================================================================
bool test_dst3_1d(int N)
{
    printf("Testing 1D DST-3: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 1) / (N + 1)) + 0.5 * sin(2 * M_PI * (i + 1) / (N + 1));
    }

    // === FFTW DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaRealTransform1D dst(N, CUDA_DST_3);
    dst.execute(d_data);

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
// Test 2D DST-2
//==============================================================================
bool test_dst2_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-2: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 0.5) / Nx;
            double y = (iy + 0.5) / Ny;
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT10, FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaRealTransform2D dst(Nx, Ny, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 2D DST-3
//==============================================================================
bool test_dst3_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-3: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 1.0) / (Nx + 1);
            double y = (iy + 1.0) / (Ny + 1);
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT01, FFTW_RODFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaRealTransform2D dst(Nx, Ny, CUDA_DST_3);
    dst.execute(d_data);

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
// Test 3D DST-2
//==============================================================================
bool test_dst2_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-2: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double x = (ix + 0.5) / Nx;
                double y = (iy + 0.5) / Ny;
                double z = (iz + 0.5) / Nz;
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT10, FFTW_RODFT10, FFTW_RODFT10,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaRealTransform3D dst(Nx, Ny, Nz, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 3D DST-3
//==============================================================================
bool test_dst3_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-3: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double x = (ix + 1.0) / (Nx + 1);
                double y = (iy + 1.0) / (Ny + 1);
                double z = (iz + 1.0) / (Nz + 1);
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT01, FFTW_RODFT01, FFTW_RODFT01,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaRealTransform3D dst(Nx, Ny, Nz, CUDA_DST_3);
    dst.execute(d_data);

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
// Test DST-2/DST-3 inverse pair
//==============================================================================
bool test_dst23_inverse(int N)
{
    printf("Testing DST-2/DST-3 inverse pair: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * sin(2 * M_PI * (i + 0.5) / N);
    }

    // FFTW: DST-2 then DST-3 should recover input (up to scale)
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan2 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);

    fftw_plan plan3 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_RODFT01, FFTW_ESTIMATE);
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

    printf("  FFTW DST-2 + DST-3 round-trip: rel_err = %.6e\n", rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D Mixed DST (vs FFTW)
//==============================================================================
bool test_mixed_2d(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y)
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
            h_input[idx] = sin(M_PI * (ix + 0.5) / Nx) * sin(M_PI * (iy + 0.5) / Ny)
                         + 0.5 * sin(2 * M_PI * (ix + 0.5) / Nx) * sin(2 * M_PI * (iy + 0.5) / Ny);
        }
    }

    // FFTW reference: separate 1D transforms per dimension
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Y dimension first (contiguous)
    fftw_r2r_kind kind_y = (type_y == CUDA_DST_2) ? FFTW_RODFT10 : FFTW_RODFT01;
    for (int ix = 0; ix < Nx; ++ix) {
        fftw_plan plan = fftw_plan_r2r_1d(Ny, &h_fftw[ix * Ny], &h_fftw[ix * Ny], kind_y, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // X dimension (need transpose for FFTW or do row-by-row)
    std::vector<double> h_col(Nx);
    fftw_r2r_kind kind_x = (type_x == CUDA_DST_2) ? FFTW_RODFT10 : FFTW_RODFT01;
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

    CudaRealTransform2D dst(Nx, Ny, type_x, type_y);
    dst.execute(d_data);

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
// Test 3D Mixed DST (vs FFTW)
//==============================================================================
bool test_mixed_3d(int Nx, int Ny, int Nz, CudaTransformType type_x, CudaTransformType type_y, CudaTransformType type_z)
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
                             * sin(M_PI * (iy + 0.5) / Ny)
                             * sin(M_PI * (iz + 0.5) / Nz);
            }
        }
    }

    // FFTW reference
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Z dimension (contiguous)
    fftw_r2r_kind kind_z = (type_z == CUDA_DST_2) ? FFTW_RODFT10 : FFTW_RODFT01;
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int offset = (ix * Ny + iy) * Nz;
            fftw_plan plan = fftw_plan_r2r_1d(Nz, &h_fftw[offset], &h_fftw[offset], kind_z, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }

    // Y dimension
    fftw_r2r_kind kind_y = (type_y == CUDA_DST_2) ? FFTW_RODFT10 : FFTW_RODFT01;
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
    fftw_r2r_kind kind_x = (type_x == CUDA_DST_2) ? FFTW_RODFT10 : FFTW_RODFT01;
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

    CudaRealTransform3D dst(Nx, Ny, Nz, type_x, type_y, type_z);
    dst.execute(d_data);

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
// Supported lengths: the 1D DST-3 accepts odd lengths (verified against FFTW);
// DST-2 and the 2D/3D DST-3 kernels assume even lengths, so odd lengths must be
// refused in the constructor, not computed wrongly.
bool test_odd_length_rejected()
{
    printf("Testing length checks: supported lengths accepted, unsupported rejected\n");
    bool ok = true;
    try { CudaRealTransform1D b(15, CUDA_DST_3); }
    catch (const std::exception& e) { printf("  a supported length was rejected: %s\n", e.what()); ok = false; }
    const struct { int nx, ny, nz; CudaTransformType type; } rejected[] = {
        {15, 0, 0, CUDA_DST_2}, {6, 5, 0, CUDA_DST_2}, {5, 6, 0, CUDA_DST_3},
        {6, 6, 5, CUDA_DST_2}, {6, 5, 6, CUDA_DST_3} };
    for (const auto& r : rejected)
    {
        bool threw = false;
        try
        {
            if (r.ny == 0) { CudaRealTransform1D t(r.nx, r.type); }
            else if (r.nz == 0) { CudaRealTransform2D t(r.nx, r.ny, r.type); }
            else { CudaRealTransform3D t(r.nx, r.ny, r.nz, r.type); }
        }
        catch (const std::exception&) { threw = true; }
        if (!threw) { printf("  type %d with odd length (%d,%d,%d) was NOT rejected\n", (int)r.type, r.nx, r.ny, r.nz); ok = false; }
    }
    printf("  %s\n\n", ok ? "PASSED!" : "FAILED!");
    return ok;
}

int main()
{
    std::cout << "========== CUDA DST vs FFTW Test ==========\n\n";

    bool all_passed = true;

    // odd lengths of the supported types (verified against FFTW)
    all_passed &= test_dst3_1d(15);
    all_passed &= test_odd_length_rejected();

    // Test 1D DST-2
    std::cout << "--- 1D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_1d(16);
    all_passed &= test_dst2_1d(32);
    all_passed &= test_dst2_1d(64);

    // Test 1D DST-3
    std::cout << "--- 1D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_1d(16);
    all_passed &= test_dst3_1d(32);
    all_passed &= test_dst3_1d(64);

    // Test 2D DST-2
    std::cout << "--- 2D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_2d(8, 8);
    all_passed &= test_dst2_2d(16, 16);
    all_passed &= test_dst2_2d(32, 32);

    // Test 2D DST-3
    std::cout << "--- 2D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_2d(8, 8);
    all_passed &= test_dst3_2d(16, 16);
    all_passed &= test_dst3_2d(32, 32);

    // Test 3D DST-2
    std::cout << "--- 3D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_3d(8, 8, 8);
    all_passed &= test_dst2_3d(16, 16, 16);
    all_passed &= test_dst2_3d(32, 32, 32);

    // Test 3D DST-3
    std::cout << "--- 3D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_3d(8, 8, 8);
    all_passed &= test_dst3_3d(16, 16, 16);
    all_passed &= test_dst3_3d(32, 32, 32);

    // Test round-trip
    std::cout << "--- Round-trip Tests ---\n\n";
    all_passed &= test_dst23_inverse(32);

    // Test 2D Mixed DST
    std::cout << "--- 2D Mixed DST Tests ---\n\n";
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_2, CUDA_DST_3);
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_3, CUDA_DST_2);

    // Test 3D Mixed DST
    std::cout << "--- 3D Mixed DST Tests ---\n\n";
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_2, CUDA_DST_3, CUDA_DST_2);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_3, CUDA_DST_2, CUDA_DST_3);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_3, CUDA_DST_3, CUDA_DST_2);

    if (all_passed)
        std::cout << "=== All tests PASSED! ===\n";
    else
        std::cout << "=== Some tests FAILED! ===\n";

    return all_passed ? 0 : 1;
}
