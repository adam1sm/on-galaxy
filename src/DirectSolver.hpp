#pragma once

#include "ForceSolver.hpp"
#include "Kernel.hpp"

namespace galaxy {

// Rung 0: the direct all-pairs O(N^2) solver.
//
// This is the project's correctness ORACLE. Every later solver
// (BarnesHutSolver, FMMSolver) is validated against the accelerations produced
// here via the force-comparison harness, so its output is kept simple and
// deterministic (no reordering of bodies, no approximation).
class DirectSolver final : public ForceSolver {
public:
    explicit DirectSolver(KernelParams params) : params_(params) {}

    Accelerations computeAccelerations(const Bodies& bodies) const override {
        const std::size_t n = bodies.size();
        Accelerations acc(n, Complex{0.0, 0.0});

        // Sum over all unique pairs once and apply the equal-and-opposite
        // contribution to both bodies (Newton's third law), halving the work.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const Complex diff = bodies.pos[j] - bodies.pos[i]; // z_j - z_i
                const double denom = std::norm(diff) + params_.eps * params_.eps;
                const Complex base = (params_.G / denom) * diff; // G*diff/(|diff|^2+eps^2)
                acc[i] += bodies.mass[j] * base;        // pull on i toward j
                acc[j] -= bodies.mass[i] * base;        // equal-and-opposite on j
            }
        }
        return acc;
    }

    const char* name() const override { return "DirectSolver(O(N^2))"; }

private:
    KernelParams params_;
};

} // namespace galaxy
