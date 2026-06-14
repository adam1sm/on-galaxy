#pragma once

#include "ForceSolver3D.hpp"

namespace galaxy3d {

// Velocity Verlet (leapfrog, kick-drift-kick), generalized to Vec3. Structurally
// identical to the 2D integrator -- the time-stepping is dimension-agnostic, it
// just operates on whatever vector type the bodies use. Symplectic, so total
// energy oscillates in a bounded band rather than drifting secularly. NEVER
// replace with forward Euler.
//
// Depends ONLY on the abstract ForceSolver3D interface, so swapping in
// BarnesHut3D or FMM3D later changes nothing here.
class VelocityVerlet3D {
public:
    explicit VelocityVerlet3D(const ForceSolver3D& solver) : solver_(solver) {}

    void initialize(const Bodies3D& bodies) {
        acc_ = solver_.computeAccelerations(bodies);
    }

    void step(Bodies3D& bodies, double dt) {
        const std::size_t n = bodies.size();
        const double half_dt2 = 0.5 * dt * dt;

        // Drift: x += v*dt + 0.5*a*dt^2
        for (std::size_t i = 0; i < n; ++i)
            bodies.pos[i] += bodies.vel[i] * dt + acc_[i] * half_dt2;

        // Recompute accelerations at the new positions.
        Accelerations3D acc_new = solver_.computeAccelerations(bodies);

        // Kick: v += 0.5*(a_old + a_new)*dt
        const double half_dt = 0.5 * dt;
        for (std::size_t i = 0; i < n; ++i)
            bodies.vel[i] += (acc_[i] + acc_new[i]) * half_dt;

        acc_ = std::move(acc_new);
    }

    const Accelerations3D& accelerations() const { return acc_; }

private:
    const ForceSolver3D& solver_;
    Accelerations3D acc_;
};

} // namespace galaxy3d
