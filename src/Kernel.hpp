#pragma once

#include "Types.hpp"

namespace galaxy {

// The single source of truth for the 2D logarithmic gravity kernel.
//
// This is the 2D Laplace / log kernel (force ~ 1/r), chosen deliberately over
// the 3D 1/r^2 kernel so the FMM's complex multipole/local expansions line up
// later. Both the force and the potential below come from the SAME softened
// potential, so the energy diagnostic is self-consistent (force = -grad U).
//
//   Pairwise potential (per unordered pair i<j):
//       U_ij = 0.5 * G * m_i * m_j * ln(|z_j - z_i|^2 + eps^2)
//
//   Acceleration on i (negative gradient of U w.r.t. z_i, divided by m_i):
//       a_i += G * m_j * (z_j - z_i) / (|z_j - z_i|^2 + eps^2)
//
// Verification that force = -grad U:
//   grad_{z_i} ln(|z_j - z_i|^2 + eps^2) = -2 (z_j - z_i) / (|z_j - z_i|^2 + eps^2)
//   F_i = -grad_{z_i} U_ij = G m_i m_j (z_j - z_i)/(|.|^2+eps^2)
//   a_i = F_i / m_i = G m_j (z_j - z_i)/(|.|^2+eps^2)   (matches above)

struct KernelParams {
    double G = 1.0;      // gravitational constant
    double eps = 0.01;   // softening length
};

// Contribution to the acceleration of body i due to body j (mass mj).
// diff = z_j - z_i. Returns G * mj * diff / (|diff|^2 + eps^2).
inline Complex pairAcceleration(const Complex& diff, double mj, const KernelParams& k) {
    const double denom = std::norm(diff) + k.eps * k.eps; // std::norm = |diff|^2
    return (k.G * mj / denom) * diff;
}

// Pairwise potential energy for one unordered pair (i,j):
//   U_ij = 0.5 * G * m_i * m_j * ln(|z_j - z_i|^2 + eps^2)
inline double pairPotential(const Complex& diff, double mi, double mj, const KernelParams& k) {
    const double r2 = std::norm(diff) + k.eps * k.eps;
    return 0.5 * k.G * mi * mj * std::log(r2);
}

} // namespace galaxy
