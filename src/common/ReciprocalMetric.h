/**
 * @file ReciprocalMetric.h
 * @brief Geometry of a cell given by its lengths and angles: lattice vectors,
 *        volume, reciprocal vectors and reciprocal metric tensor.
 *
 * The one computation of the cell geometry: ComputationBox takes it for its
 * cell, and the CPU and CUDA ObliqueZ CrysFFT engines, which build their
 * Boltzmann factors from cell parameters rather than from a ComputationBox,
 * take the reciprocal metric of the same cell from it.
 */

#ifndef RECIPROCAL_METRIC_H_
#define RECIPROCAL_METRIC_H_

#include <array>
#include <cmath>

/// Geometry of a cell {a, b, c, alpha, beta, gamma}
struct CellGeometry
{
    double volume;                       ///< V = a·(b×c)
    std::array<double, 9> lattice_vec;   ///< [a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z]
    std::array<double, 6> recip_metric;  ///< G* = (a*·a*, a*·b*, a*·c*, b*·b*, b*·c*, c*·c*) of the
                                         ///< reciprocal vectors (without 2π) a* = (b×c)/V, b* = (c×a)/V, c* = (a×b)/V
};

/**
 * @brief Geometry of the cell {a, b, c, alpha, beta, gamma}, angles in radians.
 *
 * Standard crystallographic parameterization:
 * - a along the x axis: a = (a, 0, 0)
 * - b in the xy plane:  b = (b cos γ, b sin γ, 0)
 * - c general:          c = (c cos β, c (cos α - cos β cos γ)/sin γ, V/(a b sin γ))
 *
 * with V = abc sqrt(1 - cos²α - cos²β - cos²γ + 2 cos α cos β cos γ). The
 * reciprocal vectors satisfy a*·a = b*·b = c*·c = 1, and |k|² = 4π² G*_ij m_i m_j
 * for the integer mode m.
 */
inline CellGeometry cell_geometry(const std::array<double, 6>& cell_para)
{
    /// Minimum sin(gamma) to avoid a division by zero in c_y
    constexpr double SIN_GAMMA_MIN = 1e-10;

    const double a = cell_para[0];
    const double b = cell_para[1];
    const double c = cell_para[2];
    const double alpha = cell_para[3];   // angle between b and c
    const double beta = cell_para[4];    // angle between a and c
    const double gamma = cell_para[5];   // angle between a and b

    const double cos_a = std::cos(alpha);
    const double cos_b = std::cos(beta);
    const double cos_g = std::cos(gamma);
    const double sin_g = std::sin(gamma);

    CellGeometry geometry{};

    // Cell volume V = abc sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ)
    const double vol_factor_sq =
        1.0 - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2.0 * cos_a * cos_b * cos_g;
    const double vol_factor = std::sqrt(vol_factor_sq);
    const double volume = a * b * c * vol_factor;
    geometry.volume = volume;

    // Direct lattice vectors
    const double ax = a, ay = 0.0, az = 0.0;
    const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
    const double cx = c * cos_b;
    const double cy = (sin_g > SIN_GAMMA_MIN) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    const double cz = volume / (a * b * sin_g);
    geometry.lattice_vec = {ax, ay, az, bx, by, bz, cx, cy, cz};

    // b × c, c × a and a × b
    const double bc_x = by * cz - bz * cy;
    const double bc_y = bz * cx - bx * cz;
    const double bc_z = bx * cy - by * cx;

    const double ca_x = cy * az - cz * ay;
    const double ca_y = cz * ax - cx * az;
    const double ca_z = cx * ay - cy * ax;

    const double ab_x = ay * bz - az * by;
    const double ab_y = az * bx - ax * bz;
    const double ab_z = ax * by - ay * bx;

    // Reciprocal lattice vectors (without 2π)
    const double a_star_x = bc_x / volume;
    const double a_star_y = bc_y / volume;
    const double a_star_z = bc_z / volume;
    const double b_star_x = ca_x / volume;
    const double b_star_y = ca_y / volume;
    const double b_star_z = ca_z / volume;
    const double c_star_x = ab_x / volume;
    const double c_star_y = ab_y / volume;
    const double c_star_z = ab_z / volume;

    // Reciprocal metric G*_ij = e*_i · e*_j
    geometry.recip_metric = {
        a_star_x * a_star_x + a_star_y * a_star_y + a_star_z * a_star_z,
        a_star_x * b_star_x + a_star_y * b_star_y + a_star_z * b_star_z,
        a_star_x * c_star_x + a_star_y * c_star_y + a_star_z * c_star_z,
        b_star_x * b_star_x + b_star_y * b_star_y + b_star_z * b_star_z,
        b_star_x * c_star_x + b_star_y * c_star_y + b_star_z * c_star_z,
        c_star_x * c_star_x + c_star_y * c_star_y + c_star_z * c_star_z};
    return geometry;
}

#endif  // RECIPROCAL_METRIC_H_
