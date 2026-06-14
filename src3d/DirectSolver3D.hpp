#pragma once

#include "ForceSolver3D.hpp"
#include "Kernel3D.hpp"

namespace galaxy3d {

// Rung 3: direct all-pairs O(N^2) 3D solver -- the project's 3D correctness
// ORACLE. Every later 3D solver (BarnesHut3D, FMM3D) is validated against the
// accelerations produced here, so its output is kept simple and deterministic.
class DirectSolver3D final : public ForceSolver3D {
public:
    explicit DirectSolver3D(KernelParams3D params) : params_(params) {}

    Accelerations3D computeAccelerations(const Bodies3D& bodies) const override {
        const std::size_t n = bodies.size();
        Accelerations3D acc(n, Vec3{});
        const double eps2 = params_.eps * params_.eps;
        const double G = params_.G;

        // Sum over unique pairs once, applying the equal-and-opposite
        // contribution to both bodies (Newton's third law).
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const Vec3 diff = bodies.pos[j] - bodies.pos[i]; // r_j - r_i
                const double r2 = norm2(diff) + eps2;
                const double inv = 1.0 / (r2 * std::sqrt(r2));   // (|diff|^2+eps^2)^(-3/2)
                const Vec3 base = (G * inv) * diff;              // G*diff/(...)^(3/2)
                acc[i] += bodies.mass[j] * base;
                acc[j] -= bodies.mass[i] * base;
            }
        }
        return acc;
    }

    const char* name() const override { return "DirectSolver3D(O(N^2))"; }

private:
    KernelParams3D params_;
};

} // namespace galaxy3d
