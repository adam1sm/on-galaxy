// ============================================================================
// RUNG 6: GPU direct-solver validation harness (the GPU-vs-CPU-oracle harness,
// reused by Rung 7). Device query + Gate A (GPU-double == CPU oracle), Gate B
// (GPU-float gap), Gate C (GPU-direct timing baseline).
// STATUS: drafted without a CUDA toolchain -- not yet compiled or run.
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "../DirectSolver3D.hpp" // CPU oracle
#include "../Types3D.hpp"
#include "gpu_direct.cuh"

using namespace galaxy3d;

namespace {

Bodies3D makeUniformCube(std::size_t N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0), um(0.5, 1.5);
    Bodies3D b;
    b.resize(N);
    for (std::size_t i = 0; i < N; ++i) { b.pos[i] = Vec3{u(rng), u(rng), u(rng)}; b.mass[i] = um(rng); }
    return b;
}

struct ErrStats { double max_rel, rms_rel; };

ErrStats relErr(const Accelerations3D& ref, const Accelerations3D& test) {
    const std::size_t n = ref.size();
    double amax = 0;
    for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, abs(ref[i]));
    const double floor = 1e-30 + 1e-6 * amax;
    double mx = 0, num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = abs(test[i] - ref[i]);
        mx = std::max(mx, e / std::max(abs(ref[i]), floor));
        num += e * e; den += norm2(ref[i]);
    }
    return {mx, std::sqrt(num / den)};
}

} // namespace

int main() {
    std::printf("=== Rung 6: GPU direct solver + validation harness ===\n\n");

    // ---- device query (the toolchain/passthrough confirmation) ------------
    const gpu::DeviceInfo dev = gpu::queryDevice(0);
    if (!dev.ok) {
        std::printf("No CUDA device found (cudaGetDeviceCount==0). "
                    "Check the driver / WSL2 passthrough.\n");
        return 1;
    }
    std::printf("--- CUDA device query ---\n");
    std::printf("  device %d        : %s\n", dev.id, dev.name.c_str());
    std::printf("  compute cap     : %d.%d  (expect 8.6 for RTX 30-series / sm_86)\n",
                dev.ccMajor, dev.ccMinor);
    std::printf("  SM count        : %d\n", dev.smCount);
    std::printf("  global memory   : %.2f GiB\n", dev.globalMemBytes / (1024.0 * 1024.0 * 1024.0));
    std::printf("  clock           : %.0f MHz\n", dev.clockKHz / 1000.0);
    std::printf("  warp size       : %d\n\n", dev.warpSize);

    const double G = 1.0, eps = 0.05;

    // ---- Gate A: GPU-double vs CPU DirectSolver3D (must match to roundoff) -
    std::printf("=== Gate A: GPU-double vs CPU DirectSolver3D (N=20000, eps=%.3f) ===\n", eps);
    {
        Bodies3D b = makeUniformCube(20000, 42);
        const Accelerations3D ref = DirectSolver3D(KernelParams3D{G, eps}).computeAccelerations(b);
        const Accelerations3D gpu = gpu::directAccelerations<double>(b, G, eps);
        const ErrStats e = relErr(ref, gpu);
        const bool pass = e.max_rel < 1e-12;
        std::printf("  max_rel=%.3e rms_rel=%.3e => %s\n", e.max_rel, e.rms_rel,
                    pass ? "GATE PASS (roundoff)" : "GATE FAIL");
    }

    // ---- Gate B: GPU-float vs CPU oracle (characterize the float gap) ------
    std::printf("\n=== Gate B: GPU-float vs CPU DirectSolver3D (the float-precision gap) ===\n");
    {
        Bodies3D b = makeUniformCube(20000, 42);
        const Accelerations3D ref = DirectSolver3D(KernelParams3D{G, eps}).computeAccelerations(b);
        const Accelerations3D gpu = gpu::directAccelerations<float>(b, G, eps);
        const ErrStats e = relErr(ref, gpu);
        // Expected ~1e-6 rms; a much larger gap would indicate a kernel bug.
        const bool atFloatPrecision = e.rms_rel < 1e-4 && e.rms_rel > 1e-8;
        std::printf("  max_rel=%.3e rms_rel=%.3e => %s\n", e.max_rel, e.rms_rel,
                    atFloatPrecision ? "at float precision (expected)" : "UNEXPECTED (investigate)");
        std::printf("  (this gap is comparable to FMM truncation at modest p; float is the\n");
        std::printf("   right choice for the 1M showcase, double is validation-only.)\n");
    }

    // ---- Gate C: GPU-direct timing baseline across N ----------------------
    std::printf("\n=== Gate C: GPU-direct timing baseline (kernel-only, best of reps) ===\n");
    std::printf("%10s  %14s  %14s  %12s\n", "N", "gpu_float_s", "gpu_double_s", "fp64/fp32");
    std::ofstream csv("outputs/gpu_scaling.csv");
    csv << "N,gpu_float_sec,gpu_double_sec\n";
    for (std::size_t N : {10000u, 30000u, 100000u, 300000u, 1000000u}) {
        Bodies3D b = makeUniformCube(N, 7);
        const int reps = N <= 100000 ? 5 : 2;
        const double tf = gpu::timeDirect<float>(b, G, eps, reps);
        // fp64 is far slower on consumer Ampere; cap the double timing at modest N.
        double td = -1.0;
        if (N <= 300000) td = gpu::timeDirect<double>(b, G, eps, reps);
        if (td > 0)
            std::printf("%10zu  %14.6e  %14.6e  %11.1fx\n", N, tf, td, td / tf);
        else
            std::printf("%10zu  %14.6e  %14s  %12s\n", N, tf, "(skip)", "-");
        csv << N << ',' << tf << ','; if (td > 0) csv << td; csv << '\n';
    }
    csv.close();
    std::printf("wrote outputs/gpu_scaling.csv\n");

    std::printf("\nNote: direct is O(N^2); 1M all-pairs is the slow baseline. FMM (Rung 7)\n"
                "is what makes 1M fast -- this rung only establishes the GPU baseline.\n");
    return 0;
}
