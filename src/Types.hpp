#pragma once

#include <complex>
#include <vector>

namespace galaxy {

// Positions and velocities live in the complex plane so that the FMM's
// complex multipole / local expansions (Rung 2) drop in without a rewrite.
//   - position  z = x + i*y
//   - velocity  v = vx + i*vy
//   - separation diff = z_j - z_i      (complex subtraction)
//   - squared distance |diff|^2 = std::norm(diff)
using Complex = std::complex<double>;

// Structure-of-arrays layout for the particle set. SoA keeps the per-field
// arrays contiguous, which the later tree/FMM solvers will want for traversal
// and for building expansions over leaf boxes.
struct Bodies {
    std::vector<Complex> pos;   // positions z_i
    std::vector<Complex> vel;   // velocities v_i
    std::vector<double>  mass;  // masses m_i

    std::size_t size() const { return mass.size(); }

    void resize(std::size_t n) {
        pos.assign(n, Complex{0.0, 0.0});
        vel.assign(n, Complex{0.0, 0.0});
        mass.assign(n, 0.0);
    }
};

// Accelerations are complex too (ax + i*ay), one per body.
using Accelerations = std::vector<Complex>;

} // namespace galaxy
