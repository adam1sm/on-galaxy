#pragma once

#include "Kernel.hpp"
#include "Types.hpp"

namespace galaxy {

struct Diagnostics {
    double kinetic = 0.0;     // sum 0.5 m_i |v_i|^2
    double potential = 0.0;   // sum_{i<j} 0.5 G m_i m_j ln(|z_j-z_i|^2 + eps^2)
    double total = 0.0;       // kinetic + potential
    Complex momentum{0.0, 0.0}; // sum m_i v_i  (Px + i Py)
};

// Compute energy and linear momentum. The potential uses EXACTLY the softened
// form whose negative gradient is the coded force (see Kernel.hpp), so the
// energy-conservation check is self-consistent with the dynamics.
inline Diagnostics computeDiagnostics(const Bodies& bodies, const KernelParams& k) {
    Diagnostics d;
    const std::size_t n = bodies.size();

    for (std::size_t i = 0; i < n; ++i) {
        d.kinetic += 0.5 * bodies.mass[i] * std::norm(bodies.vel[i]); // |v|^2
        d.momentum += bodies.mass[i] * bodies.vel[i];
    }

    // Sum potential over unique pairs (i<j) so each pair is counted once.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const Complex diff = bodies.pos[j] - bodies.pos[i];
            d.potential += pairPotential(diff, bodies.mass[i], bodies.mass[j], k);
        }
    }

    d.total = d.kinetic + d.potential;
    return d;
}

} // namespace galaxy
