#pragma once

#include "Types3D.hpp"

namespace galaxy3d {

// Abstract 3D force-evaluation interface -- the 3D analog of the 2D ForceSolver
// and the central seam of the 3D track. The integrator and diagnostics depend
// ONLY on this interface; each rung supplies a different implementation of the
// SAME contract:
//
//   Rung 3  DirectSolver3D   O(N^2)      <-- implemented now (the ORACLE)
//   Rung 4+ BarnesHut3D      O(N log N)  <-- future seam, BarnesHut3D.hpp
//   Rung 4+ FMM3D            O(N)        <-- future seam, FMM3D.hpp
//
// All solvers compute the SAME physical quantity: the acceleration on every body
// from the 3D Plummer-softened Newtonian kernel (see Kernel3D.hpp). Keeping the
// output identical lets the force-comparison harness validate later solvers
// against DirectSolver3D.
class ForceSolver3D {
public:
    virtual ~ForceSolver3D() = default;

    // Compute accelerations for all bodies; sized to bodies.size().
    virtual Accelerations3D computeAccelerations(const Bodies3D& bodies) const = 0;

    virtual const char* name() const = 0;
};

} // namespace galaxy3d
