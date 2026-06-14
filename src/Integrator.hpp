#pragma once

#include "ForceSolver.hpp"

namespace galaxy {

// Velocity Verlet (a.k.a. leapfrog in kick-drift-kick form). Symplectic, so the
// total energy oscillates within a bounded band rather than drifting secularly.
// NEVER replace this with forward Euler -- Euler injects energy and breaks the
// conservation checks that validate the whole simulation.
//
// One step (a = accelerations at current positions):
//   x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2
//   a_{n+1} = solver(x_{n+1})
//   v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt
//
// Crucially this depends ONLY on the abstract ForceSolver interface, never on a
// concrete solver, so swapping in Barnes-Hut or FMM later changes nothing here.
class VelocityVerlet {
public:
    explicit VelocityVerlet(const ForceSolver& solver) : solver_(solver) {}

    // Prime the cached accelerations from the current state. Call once before
    // the first step (or after any external change to positions).
    void initialize(const Bodies& bodies) {
        acc_ = solver_.computeAccelerations(bodies);
    }

    // Advance the system by one timestep dt, in place.
    void step(Bodies& bodies, double dt) {
        const std::size_t n = bodies.size();

        // Drift positions a half-and-half: x += v*dt + 0.5*a*dt^2
        const double half_dt2 = 0.5 * dt * dt;
        for (std::size_t i = 0; i < n; ++i) {
            bodies.pos[i] += bodies.vel[i] * dt + acc_[i] * half_dt2;
        }

        // Recompute accelerations at the new positions.
        Accelerations acc_new = solver_.computeAccelerations(bodies);

        // Kick velocities with the average of old and new accelerations.
        const double half_dt = 0.5 * dt;
        for (std::size_t i = 0; i < n; ++i) {
            bodies.vel[i] += (acc_[i] + acc_new[i]) * half_dt;
        }

        acc_ = std::move(acc_new);
    }

    const Accelerations& accelerations() const { return acc_; }

private:
    const ForceSolver& solver_;
    Accelerations acc_;
};

} // namespace galaxy
