#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../Types3D.hpp" // galaxy3d::Bodies3D, Vec3, Accelerations3D

// ============================================================================
// RUNG 6: GPU direct O(N^2) solver -- public API (CUDA implementation in
// gpu_direct.cu). Same 3D Plummer-softened kernel as DirectSolver3D:
//     a_i = G * sum_{j!=i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//
// Templated on the arithmetic precision `real`:
//   - double : VALIDATION (must match CPU DirectSolver3D to ~roundoff).
//   - float  : PERFORMANCE / the eventual 1M run (matches the oracle only to
//              ~float precision, ~1e-5..1e-6 relative -- expected, characterized
//              in Gate B). Consumer Ampere runs fp64 at a small fraction of fp32
//              throughput, so fp64 is validation-only.
//
// STATUS: drafted on a machine WITHOUT a CUDA toolchain (no nvcc/WSL2) -- NOT
// yet compiled or run. Ready to build + validate once the toolchain is in place.
// ============================================================================

namespace galaxy3d {
namespace gpu {

struct DeviceInfo {
    bool ok = false;
    int id = 0;
    std::string name;
    int ccMajor = 0, ccMinor = 0;
    int smCount = 0;
    std::size_t globalMemBytes = 0;
    int clockKHz = 0;
    int warpSize = 0;
};

// Query device `id`. Returns ok=false if no CUDA device is available.
DeviceInfo queryDevice(int id = 0);

// All-pairs accelerations computed on the GPU in precision `real`. Inputs are
// the (double) Bodies3D; G/eps are doubles cast to `real` internally. Returns
// accelerations as double (cast up from `real`) so the comparison harness can
// diff directly against the CPU oracle.
template <typename real>
Accelerations3D directAccelerations(const Bodies3D& bodies, double G, double eps);

// Best-of-`reps` device kernel time in seconds (kernel only, excludes H2D/D2H),
// for the scaling baseline. Also returns the result if `out` is non-null.
template <typename real>
double timeDirect(const Bodies3D& bodies, double G, double eps, int reps,
                  Accelerations3D* out = nullptr);

} // namespace gpu
} // namespace galaxy3d
