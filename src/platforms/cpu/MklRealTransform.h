/**
 * @file MklRealTransform.h
 * @brief MKL real-to-real DCT/DST (types 1-4) along the axes of a 1D-3D array.
 *
 * The solvers do not use it (they run MakhoulTransform); tests and the MKL/FFTW
 * benchmark use it for every DCT/DST type. The transforms are unnormalized, in
 * FFTW's conventions, and run one line at a time, the last axis first.
 *
 * Implementation of a line of N points:
 *   - DCT-I, DST-I: real FFT of the symmetric or antisymmetric extension
 *     (2(N-1) or 2(N+1) points), O(N log N)
 *   - DCT-II, DCT-III, DST-II, DST-III: MakhoulTransform over the N-point real
 *     FFT of MKL (MklRealFFT), O(N log N)
 *   - DCT-IV, DST-IV: MKL TT interface, O(N log N)
 *
 * MKL TT to FFTW mapping (empirically verified):
 *   - STAGGERED2_COSINE forward × N = DCT-IV
 *   - STAGGERED2_SINE forward × N   = DST-IV
 */

#ifndef MKL_REAL_TRANSFORM_H_
#define MKL_REAL_TRANSFORM_H_

#include <memory>
#include <vector>

/**
 * Transform Type enumeration for DCT and DST.
 *
 * FFTW correspondence:
 *   - DCT-1: FFTW_REDFT00   - DST-1: FFTW_RODFT00
 *   - DCT-2: FFTW_REDFT10   - DST-2: FFTW_RODFT10
 *   - DCT-3: FFTW_REDFT01   - DST-3: FFTW_RODFT01
 *   - DCT-4: FFTW_REDFT11   - DST-4: FFTW_RODFT11
 */
enum MklTransformType {
    MKL_DCT_1 = 0,
    MKL_DCT_2 = 1,
    MKL_DCT_3 = 2,
    MKL_DCT_4 = 3,
    MKL_DST_1 = 4,
    MKL_DST_2 = 5,
    MKL_DST_3 = 6,
    MKL_DST_4 = 7
};

inline const char* getTransformName(MklTransformType type) {
    switch (type) {
        case MKL_DCT_1: return "DCT-1";
        case MKL_DCT_2: return "DCT-2";
        case MKL_DCT_3: return "DCT-3";
        case MKL_DCT_4: return "DCT-4";
        case MKL_DST_1: return "DST-1";
        case MKL_DST_2: return "DST-2";
        case MKL_DST_3: return "DST-3";
        case MKL_DST_4: return "DST-4";
        default: return "Unknown";
    }
}

/**
 * @class MklRealTransform
 * @brief A DCT or DST of any type along each axis of a row-major array.
 */
class MklRealTransform
{
public:
    /**
     * @param nx    Grid dimensions (one to three axes, each of at least two points)
     * @param types The transform of each axis
     */
    MklRealTransform(const std::vector<int>& nx, const std::vector<MklTransformType>& types);
    ~MklRealTransform();

    // Non-copyable
    MklRealTransform(const MklRealTransform&) = delete;
    MklRealTransform& operator=(const MklRealTransform&) = delete;

    /// The transform of every axis of data, in place: along the last axis first, the first axis last.
    void execute(double* data);

private:
    class Line;   ///< The transform of one line of an axis

    std::vector<int> nx_;
    std::vector<std::unique_ptr<Line>> lines_;   ///< One per axis
};

#endif  // MKL_REAL_TRANSFORM_H_
