#pragma once

#include "Kernel3D.hpp"
#include "Types3D.hpp"

namespace galaxy3d {

struct Diagnostics3D {
    double kinetic = 0.0;       // sum 0.5 m_i |v_i|^2
    double potential = 0.0;     // sum_{i<j} -G m_i m_j / sqrt(|r_ij|^2 + eps^2)
    double total = 0.0;         // kinetic + potential
    Vec3   momentum{};          // sum m_i v_i               (linear momentum)
    Vec3   angular{};           // sum m_i (r_i x v_i)       (angular momentum vector)
};

// Energy, linear momentum, and the angular-momentum VECTOR. The potential uses
// EXACTLY the softened form whose negative gradient is the coded force (see
// Kernel3D.hpp), so the energy check is self-consistent. Angular momentum is a
// 3-vector; for a central force all three components are conserved.
inline Diagnostics3D computeDiagnostics(const Bodies3D& bodies, const KernelParams3D& k) {
    Diagnostics3D d;
    const std::size_t n = bodies.size();

    for (std::size_t i = 0; i < n; ++i) {
        d.kinetic  += 0.5 * bodies.mass[i] * norm2(bodies.vel[i]);
        d.momentum += bodies.mass[i] * bodies.vel[i];
        d.angular  += bodies.mass[i] * cross(bodies.pos[i], bodies.vel[i]);
    }

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const Vec3 diff = bodies.pos[j] - bodies.pos[i];
            d.potential += pairPotential(diff, bodies.mass[i], bodies.mass[j], k);
        }

    d.total = d.kinetic + d.potential;
    return d;
}

} // namespace galaxy3d
