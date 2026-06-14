#pragma once

#include "Types.hpp"

namespace galaxy {

// Abstract force-evaluation interface.
//
// This is the central seam of the whole project. The time integrator and every
// diagnostic depend ONLY on this interface, never on a concrete solver. Each
// rung of the roadmap supplies a different implementation of the SAME contract:
//
//   Rung 0  DirectSolver       O(N^2)      <-- implemented now (the ORACLE)
//   Rung 1  BarnesHutSolver    O(N log N)  <-- future, see BarnesHutSolver.hpp
//   Rung 2  FMMSolver          O(N)        <-- future, see FMMSolver.hpp
//
// All solvers compute the SAME physical quantity: the acceleration on every
// body from the 2D logarithmic (Laplace) gravity kernel with softening eps:
//
//   a_i = G * sum_{j != i} m_j * (z_j - z_i) / (|z_j - z_i|^2 + eps^2)
//
// Keeping the output identical in shape and meaning is what lets the
// force-comparison harness validate every later solver against DirectSolver.
class ForceSolver {
public:
    virtual ~ForceSolver() = default;

    // Compute accelerations for all bodies. Implementations must size the
    // returned vector to bodies.size().
    virtual Accelerations computeAccelerations(const Bodies& bodies) const = 0;

    // Human-readable name, used in logs and the comparison harness.
    virtual const char* name() const = 0;
};

} // namespace galaxy
