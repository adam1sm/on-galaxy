#pragma once

#include "../Types3D.hpp"
#include "FMM3DGPU.cuh" // GpuTiming

// ============================================================================
// RUNG 10b: GPU adaptive FMM (CGR U/V/W/X). The adaptive octree + lists are
// built on the CPU (Rung 10a logic, unchanged) and flattened to CSR; the passes
// run as CUDA kernels. fp32 (showcase) / fp64 (roundoff validation).
// ============================================================================

namespace galaxy3d {
namespace gpu {

// GPU adaptive FMM accelerations (precision `real`). p in {2,4,6,8}.
template <typename real>
Accelerations3D adaptiveFmm3dGpu(const Bodies3D& bodies, double G, double eps, int p,
                                 int threshold = 64, GpuTiming* timing = nullptr);

} // namespace gpu
} // namespace galaxy3d
