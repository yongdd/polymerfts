/**
 * @file CudaRealTransform.h
 * @brief Unified CUDA Real Transform (DCT/DST) implementation.
 *
 * Supports the DCT and DST types 2 and 3 for 1D, 2D, and 3D arrays (the library
 * runs reflecting and absorbing walls and the Pmmm CrysFFT on them).
 * Allows mixing different transform types across dimensions.
 *
 * Boundary condition mapping:
 *   - DCT (Discrete Cosine Transform): Neumann boundary conditions
 *   - DST (Discrete Sine Transform): Dirichlet boundary conditions
 *
 * FFTW correspondence:
 *   - DCT-2: FFTW_REDFT10   - DST-2: FFTW_RODFT10
 *   - DCT-3: FFTW_REDFT01   - DST-3: FFTW_RODFT01
 *
 * Reference: FCT/FST algorithm (Makhoul 1980, Martucci 1994)
 *            CUDA implementation: https://github.com/rmingming/cuHelmholtz
 */

#ifndef CUDA_REAL_TRANSFORM_H_
#define CUDA_REAL_TRANSFORM_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include "CudaCommon.h"

/**
 * Transform Type enumeration for DCT and DST.
 */
enum CudaTransformType {
    // DCT Types (Neumann boundary conditions)
    CUDA_DCT_2,   // FFTW_REDFT10: DCT-II
    CUDA_DCT_3,   // FFTW_REDFT01: DCT-III
    // DST Types (Dirichlet boundary conditions)
    CUDA_DST_2,   // FFTW_RODFT10: DST-II
    CUDA_DST_3    // FFTW_RODFT01: DST-III
};

/// Get the name string for a transform type
inline const char* getTransformName(CudaTransformType type) {
    switch (type) {
        case CUDA_DCT_2: return "DCT-2";
        case CUDA_DCT_3: return "DCT-3";
        case CUDA_DST_2: return "DST-2";
        case CUDA_DST_3: return "DST-3";
        default: return "Unknown";
    }
}

//==============================================================================
// CudaRealTransform1D
//==============================================================================

/**
 * @class CudaRealTransform1D
 * @brief CUDA 1D Real Transform (DCT/DST types 2 and 3).
 *
 * The constructor takes the number of points N; input and output have that size.
 *
 * Transform lengths (verified against FFTW): DCT-2/3 and the 1D DST-3 accept any N
 * (the CrysFFT physical grid, half the logical grid, is odd whenever a logical size
 * is 2 mod 4); DST-2 (all dimensions) and the 2D/3D DST-3 need even N. The constructors reject
 * unsupported lengths.
 */
class CudaRealTransform1D
{
private:
    int N_;                      ///< Transform size
    CudaTransformType type_;     ///< Transform type
    CufftPlan plan_;             ///< cuFFT plan
    CudaDeviceBuffer<double> d_work_;             ///< Work buffer
    CudaDeviceBuffer<double> d_x1_;               ///< Auxiliary buffer
    // Optional lookup tables for DCT2/DCT3 and DST2/3
    CudaDeviceBuffer<double> dct2_sin_;
    CudaDeviceBuffer<double> dct2_cos_;
    CudaDeviceBuffer<double> dct3_sin_;
    CudaDeviceBuffer<double> dct3_cos_;
    cudaStream_t stream_{0};     ///< CUDA stream for execution
    bool initialized_;

public:
    /**
     * @brief Construct 1D real transformer.
     * @param N Transform size
     * @param type Transform type (DCT-2/3 or DST-2/3)
     */
    CudaRealTransform1D(int N, CudaTransformType type);

    /**
     * @brief Execute transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /// Set CUDA stream for execution.
    void set_stream(cudaStream_t stream);

    /// Get the input/output size.
    int get_size() const;

    /// Get transform type.
    CudaTransformType get_type() const { return type_; }

    /// Get normalization factor for round-trip.
    double get_normalization() const;
};

//==============================================================================
// CudaRealTransform2D
//==============================================================================

/**
 * @class CudaRealTransform2D
 * @brief CUDA 2D Real Transform (DCT/DST).
 *
 * Supports any combination of DCT-2/3 and DST-2/3 across dimensions.
 *
 * Example use cases:
 *   - X: DCT-2 (Neumann), Y: DST-2 (Dirichlet)
 *   - X: DST-3 (Dirichlet), Y: DCT-3 (Neumann)
 */
class CudaRealTransform2D
{
private:
    int Nx_, Ny_;                ///< Transform sizes
    int M_;                      ///< Total size (Nx * Ny)
    CudaTransformType type_x_;   ///< Transform type for X dimension
    CudaTransformType type_y_;   ///< Transform type for Y dimension

    CufftPlan plan_x_;           ///< cuFFT plan for X dimension
    CufftPlan plan_y_;           ///< cuFFT plan for Y dimension

    CudaDeviceBuffer<double> d_work_;             ///< Work buffer
    CudaDeviceBuffer<double> d_temp_;             ///< Temporary buffer
    CudaDeviceBuffer<double> d_x1_;               ///< Auxiliary buffer

    // Optional lookup tables for DCT2/DCT3 and DST2/3
    CudaDeviceBuffer<double> dct2_sin_x_;
    CudaDeviceBuffer<double> dct2_cos_x_;
    CudaDeviceBuffer<double> dct2_sin_y_;
    CudaDeviceBuffer<double> dct2_cos_y_;
    CudaDeviceBuffer<double> dct3_sin_x_;
    CudaDeviceBuffer<double> dct3_cos_x_;
    CudaDeviceBuffer<double> dct3_sin_y_;
    CudaDeviceBuffer<double> dct3_cos_y_;

    bool initialized_;
    cudaStream_t stream_{0};     ///< CUDA stream for execution

    void init();

    // Y dimension processing
    void executeY_DCT2(double* d_data);
    void executeY_DCT3(double* d_data);
    void executeY_DST2(double* d_data);
    void executeY_DST3(double* d_data);

    // X dimension processing
    void executeX_DCT2(double* d_data);
    void executeX_DCT3(double* d_data);
    void executeX_DST2(double* d_data);
    void executeX_DST3(double* d_data);

public:
    /**
     * @brief Construct 2D real transformer with same type in both dimensions.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type Transform type for both dimensions
     */
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type);

    /**
     * @brief Construct 2D real transformer with different types per dimension.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type_x Transform type for X dimension
     * @param type_y Transform type for Y dimension
     */
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y);

    /**
     * @brief Execute 2D transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /// Set CUDA stream for execution.
    void set_stream(cudaStream_t stream);

    /// Get the input/output size.
    int get_size() const { return M_; }

    /// Get dimensions for this transform.
    void get_dims(int& nx, int& ny) const;

    /// Get transform types for each dimension.
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y) const;

    /// Get normalization factor for round-trip.
    double get_normalization() const;
};

//==============================================================================
// CudaRealTransform3D
//==============================================================================

/**
 * @class CudaRealTransform3D
 * @brief CUDA 3D Real Transform (DCT/DST).
 *
 * Supports any combination of DCT-2/3 and DST-2/3 across dimensions.
 *
 * Example use cases:
 *   - X: DCT-2, Y: DCT-2, Z: DST-2 (Neumann in XY, Dirichlet in Z)
 *   - X: DST-2, Y: DST-2, Z: DCT-2 (Dirichlet in XY, Neumann in Z)
 */
class CudaRealTransform3D
{
private:
    int Nx_, Ny_, Nz_;           ///< Transform sizes
    int M_;                      ///< Total size
    CudaTransformType type_x_;   ///< Transform type for X dimension
    CudaTransformType type_y_;   ///< Transform type for Y dimension
    CudaTransformType type_z_;   ///< Transform type for Z dimension

    CufftPlan plan_x_;           ///< cuFFT plan for X dimension
    CufftPlan plan_y_;           ///< cuFFT plan for Y dimension
    CufftPlan plan_z_;           ///< cuFFT plan for Z dimension

    CudaDeviceBuffer<double> d_work_;             ///< Work buffer
    CudaDeviceBuffer<double> d_temp_;             ///< Temporary buffer
    CudaDeviceBuffer<double> d_x1_;               ///< Auxiliary buffer

    // Optional lookup tables for DCT2/DCT3 (reduce sincos overhead)
    CudaDeviceBuffer<double> dct2_sin_x_;
    CudaDeviceBuffer<double> dct2_cos_x_;
    CudaDeviceBuffer<double> dct2_sin_y_;
    CudaDeviceBuffer<double> dct2_cos_y_;
    CudaDeviceBuffer<double> dct2_sin_z_;
    CudaDeviceBuffer<double> dct2_cos_z_;
    CudaDeviceBuffer<double> dct3_sin_x_;
    CudaDeviceBuffer<double> dct3_cos_x_;
    CudaDeviceBuffer<double> dct3_sin_y_;
    CudaDeviceBuffer<double> dct3_cos_y_;
    CudaDeviceBuffer<double> dct3_sin_z_;
    CudaDeviceBuffer<double> dct3_cos_z_;

    bool initialized_;
    cudaStream_t stream_{0};     ///< CUDA stream for execution

    void init();

    // Z dimension processing
    void executeZ_DCT2(double* d_data);
    void executeZ_DCT3(double* d_data);
    void executeZ_DST2(double* d_data);
    void executeZ_DST3(double* d_data);

    // Y dimension processing
    void executeY_DCT2(double* d_data);
    void executeY_DCT3(double* d_data);
    void executeY_DST2(double* d_data);
    void executeY_DST3(double* d_data);

    // X dimension processing
    void executeX_DCT2(double* d_data);
    void executeX_DCT3(double* d_data);
    void executeX_DST2(double* d_data);
    void executeX_DST3(double* d_data);

public:
    /**
     * @brief Construct 3D real transformer with same type in all dimensions.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type Transform type for all dimensions
     */
    CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type);

    /**
     * @brief Construct 3D real transformer with different types per dimension.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type_x Transform type for X dimension
     * @param type_y Transform type for Y dimension
     * @param type_z Transform type for Z dimension
     */
    CudaRealTransform3D(int Nx, int Ny, int Nz,
                        CudaTransformType type_x, CudaTransformType type_y, CudaTransformType type_z);

    /**
     * @brief Execute 3D transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /// Set CUDA stream for execution.
    void set_stream(cudaStream_t stream);

    /// Get the input/output size.
    int get_size() const { return M_; }

    /// Get dimensions for this transform.
    void get_dims(int& nx, int& ny, int& nz) const;

    /// Get transform types for each dimension.
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y, CudaTransformType& type_z) const;

    /// Get normalization factor for round-trip.
    double get_normalization() const;
};

//==============================================================================
// Backward compatibility aliases (deprecated)
//==============================================================================

// Old enum names for backward compatibility
typedef CudaTransformType CudaDCTType;
typedef CudaTransformType CudaDSTType;
typedef CudaTransformType CudaMixedTransformType;

// Old enum value aliases
#define CUDA_TRANSFORM_DCT_2 CUDA_DCT_2
#define CUDA_TRANSFORM_DCT_3 CUDA_DCT_3
#define CUDA_TRANSFORM_DST_2 CUDA_DST_2
#define CUDA_TRANSFORM_DST_3 CUDA_DST_3

// Old class name aliases
typedef CudaRealTransform1D CudaDCT;
typedef CudaRealTransform1D CudaDST;
typedef CudaRealTransform2D CudaDCT2D;
typedef CudaRealTransform2D CudaDST2D;
typedef CudaRealTransform2D CudaMixedTransform2D;
typedef CudaRealTransform3D CudaDCT3D;
typedef CudaRealTransform3D CudaDST3D;
typedef CudaRealTransform3D CudaMixedTransform3D;

#endif // CUDA_REAL_TRANSFORM_H_
