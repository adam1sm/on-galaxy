#pragma once

#include "../Types3D.hpp"

// ============================================================================
// RUNG 7: GPU FMM public API. Tree is built on the CPU (Rung 5 logic, flat
// layout) and the passes run as CUDA kernels. Reports CPU tree-build time and
// GPU compute time SEPARATELY so the bottleneck is visible (Gate C).
// STATUS: built + validated against the CPU FMM / oracle via the harness.
// ============================================================================

namespace galaxy3d {
namespace gpu {

struct GpuTiming {
    double treeBuildSec = 0.0; // CPU octree build (+ host A-table)
    double uploadSec = 0.0;    // H2D transfer
    double computeSec = 0.0;   // GPU kernels (P2M..L2P+P2P), excludes tree/upload
    int depth = 0;
    int nBoxes = 0;
    int maxLeafOcc = 0; // adaptive only: peak leaf occupancy (eps-floor diagnostic)
};

// GPU FMM accelerations (precision `real`). p in {2,4,6,8}; depth<0 -> auto.
template <typename real>
Accelerations3D fmm3dGpu(const Bodies3D& bodies, double G, double eps, int p,
                         int depth = -1, int occTarget = 192, GpuTiming* timing = nullptr);

} // namespace gpu
} // namespace galaxy3d
