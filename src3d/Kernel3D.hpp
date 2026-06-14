#pragma once

#include "Types3D.hpp"

namespace galaxy3d {

// Single source of truth for the 3D gravitational kernel: real Newtonian
// gravity (force ~ 1/r^2) with Plummer softening. NOTE the 3/2 exponent -- this
// is genuinely 3D, NOT the 2D log kernel.
//
//   Pairwise potential (per unordered pair i<j):
//       U_ij = -G * m_i * m_j / sqrt(|r_j - r_i|^2 + eps^2)
//
//   Acceleration on i (= -grad_{r_i} U / m_i):
//       a_i += G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//
// Verification that force = -grad U (Plummer-softened):
//   grad_{r_i} |r_j - r_i|^2 = -2 (r_j - r_i)
//   grad_{r_i} U_ij = -G m_i m_j * (-1/2)(|.|^2+eps^2)^(-3/2) * (-2)(r_j - r_i)
//                   = -G m_i m_j (r_j - r_i) / (|.|^2 + eps^2)^(3/2)
//   F_i = -grad_{r_i} U_ij = G m_i m_j (r_j - r_i)/(|.|^2+eps^2)^(3/2)
//   a_i = F_i / m_i = G m_j (r_j - r_i)/(|.|^2+eps^2)^(3/2)   (matches above)

struct KernelParams3D {
    double G = 1.0;     // gravitational constant
    double eps = 0.05;  // Plummer softening length
};

// Contribution to the acceleration of body i due to body j (mass mj).
// diff = r_j - r_i. Returns G * mj * diff / (|diff|^2 + eps^2)^(3/2).
inline Vec3 pairAcceleration(const Vec3& diff, double mj, const KernelParams3D& k) {
    const double r2 = norm2(diff) + k.eps * k.eps;
    const double inv = 1.0 / (r2 * std::sqrt(r2)); // (|diff|^2 + eps^2)^(-3/2)
    return (k.G * mj * inv) * diff;
}

// Pairwise potential energy for one unordered pair (i,j):
//   U_ij = -G * m_i * m_j / sqrt(|r_j - r_i|^2 + eps^2)
inline double pairPotential(const Vec3& diff, double mi, double mj, const KernelParams3D& k) {
    const double r2 = norm2(diff) + k.eps * k.eps;
    return -k.G * mi * mj / std::sqrt(r2);
}

} // namespace galaxy3d
