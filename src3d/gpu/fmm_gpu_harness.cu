// ============================================================================
// RUNG 7: GPU FMM validation + scaling + million-body harness.
//   Gate A: GPU-FMM fp64 vs CPU FMM3D (roundoff) and vs DirectSolver3D (err-vs-p)
//   Gate B: GPU-FMM fp32 vs oracle (float + truncation)
//   Gate C: scaling GPU-direct vs GPU-FMM (fp32), tree-build vs compute split
//   Gate D: million-body run (fp32) -- force-eval time + bounded-energy dynamics
// ============================================================================
#include <algorithm> // std::max (init-list) for FMM3D.hpp under nvcc host
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "../DirectSolver3D.hpp"
#include "../Diagnostics3D.hpp"
#include "../FMM3D.hpp"             // CPU FMM (Rung 5) for Gate A
#include "../Integrator3D.hpp"
#include "../InitialConditions3D.hpp"
#include "../Types3D.hpp"
#include "FMM3DGPU.cuh"
#include "gpu_direct.cuh"

using namespace galaxy3d;

namespace {

Bodies3D makeUniformCube(std::size_t N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0), um(0.5, 1.5);
    Bodies3D b; b.resize(N);
    for (std::size_t i = 0; i < N; ++i) { b.pos[i] = Vec3{u(rng), u(rng), u(rng)}; b.mass[i] = um(rng); }
    return b;
}

struct ErrStats { double max_rel, rms_rel; };
ErrStats relErr(const Accelerations3D& ref, const Accelerations3D& test) {
    const std::size_t n = ref.size();
    double amax = 0; for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, abs(ref[i]));
    const double floor = 1e-30 + 1e-6 * amax;
    double mx = 0, num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = abs(test[i] - ref[i]);
        mx = std::max(mx, e / std::max(abs(ref[i]), floor));
        num += e * e; den += norm2(ref[i]);
    }
    return {mx, std::sqrt(num / den)};
}

// GPU FMM wrapped as a ForceSolver3D (for the Gate D dynamics).
struct GpuFmmSolver : ForceSolver3D {
    KernelParams3D k; int p; int depth;
    GpuFmmSolver(KernelParams3D kk, int pp, int d) : k(kk), p(pp), depth(d) {}
    Accelerations3D computeAccelerations(const Bodies3D& b) const override {
        return gpu::fmm3dGpu<float>(b, k.G, k.eps, p, depth);
    }
    const char* name() const override { return "FMM3D-GPU(fp32)"; }
};

} // namespace

int main() {
    std::printf("=== Rung 7: GPU FMM (million-body) ===\n");
    const gpu::DeviceInfo dev = gpu::queryDevice(0);
    if (!dev.ok) { std::printf("no CUDA device\n"); return 1; }
    std::printf("device: %s (sm_%d%d, %d SM, %.1f GiB)\n\n", dev.name.c_str(),
                dev.ccMajor, dev.ccMinor, dev.smCount, dev.globalMemBytes / 1073741824.0);

    const double G = 1.0;

    // ---- Gate A: GPU-FMM fp64 vs CPU FMM3D (roundoff), same p + depth -------
    std::printf("=== Gate A: GPU-FMM(fp64) vs CPU FMM3D, same tree (N=8000, eps=0) ===\n");
    {
        const int depth = 2;
        Bodies3D b = makeUniformCube(8000, 1);
        const KernelParams3D k0{G, 0.0};
        for (int p : {4, 8}) {
            const Accelerations3D cpu = FMM3D(k0, p, depth).computeAccelerations(b);
            const Accelerations3D g = gpu::fmm3dGpu<double>(b, G, 0.0, p, depth);
            const ErrStats e = relErr(cpu, g);
            std::printf("  p=%d: max_rel=%.3e rms_rel=%.3e => %s\n", p, e.max_rel, e.rms_rel,
                        e.max_rel < 1e-9 ? "ROUNDOFF MATCH" : "MISMATCH");
        }
    }

    // ---- Gate A(2): GPU-FMM fp64 vs DirectSolver3D oracle, error-vs-p -------
    std::printf("\n=== Gate A(2)/B: GPU-FMM vs DirectSolver3D oracle (N=8000, eps=0) ===\n");
    std::printf("%4s  %14s  %14s  %14s  %14s\n", "p", "fp64_max", "fp64_rms", "fp32_max", "fp32_rms");
    {
        Bodies3D b = makeUniformCube(8000, 2);
        const Accelerations3D ref = DirectSolver3D(KernelParams3D{G, 0.0}).computeAccelerations(b);
        std::ofstream csv("outputs/fmm_gpu_error.csv");
        csv << "p,fp64_max,fp64_rms,fp32_max,fp32_rms\n";
        for (int p : {2, 4, 6, 8}) {
            const ErrStats ed = relErr(ref, gpu::fmm3dGpu<double>(b, G, 0.0, p, -1));
            const ErrStats ef = relErr(ref, gpu::fmm3dGpu<float>(b, G, 0.0, p, -1));
            std::printf("%4d  %14.6e  %14.6e  %14.6e  %14.6e\n", p, ed.max_rel, ed.rms_rel, ef.max_rel, ef.rms_rel);
            csv << p << ',' << ed.max_rel << ',' << ed.rms_rel << ',' << ef.max_rel << ',' << ef.rms_rel << '\n';
        }
        csv.close();
        std::printf("  (fp64 -> converges to roundoff; fp32 floors at ~float precision + truncation)\n");
    }

    // ---- Gate C: scaling GPU-direct vs GPU-FMM (fp32), tree vs compute ------
    std::printf("\n=== Gate C: GPU scaling, fp32, p=4 (uniform cube) ===\n");
    std::printf("%10s  %12s  %12s  %12s  %12s  %10s\n", "N", "gpu_direct", "fmm_compute", "fmm_tree",
                "fmm_total", "fmm/dir");
    {
        std::ofstream csv("outputs/fmm_gpu_scaling.csv");
        csv << "N,gpu_direct_sec,fmm_compute_sec,fmm_tree_sec,fmm_total_sec\n";
        long crossover = -1;
        for (std::size_t N : {30000u, 100000u, 300000u, 1000000u}) {
            Bodies3D b = makeUniformCube(N, 7);
            const int reps = N <= 100000 ? 3 : 1;
            const double tdir = gpu::timeDirect<float>(b, G, 0.05, reps);
            gpu::GpuTiming tm;
            // warm + timed FMM (best of reps on compute)
            double best = 1e300, tree = 0;
            for (int r = 0; r < reps; ++r) {
                gpu::fmm3dGpu<float>(b, G, 0.05, 4, -1, 192, &tm);
                best = std::min(best, tm.computeSec); tree = tm.treeBuildSec;
            }
            const double total = best + tree + tm.uploadSec;
            std::printf("%10zu  %12.4e  %12.4e  %12.4e  %12.4e  %9.2fx\n", N, tdir, best, tree, total, total / tdir);
            csv << N << ',' << tdir << ',' << best << ',' << tree << ',' << total << '\n';
            if (crossover < 0 && total < tdir) crossover = (long)N;
        }
        csv.close();
        if (crossover > 0) std::printf("  GPU-FMM(total) overtakes GPU-direct at N >= %ld\n", crossover);
    }

    // ---- Gate D: million-body run ------------------------------------------
    std::printf("\n=== Gate D: MILLION-BODY GPU FMM (fp32, p=4) ===\n");
    {
        Bodies3D b = makeUniformCube(1000000, 11);
        gpu::GpuTiming tm;
        const Accelerations3D a = gpu::fmm3dGpu<float>(b, G, 0.05, 4, -1, 192, &tm);
        double amax = 0, amean = 0;
        for (auto& v : a) { amax = std::max(amax, abs(v)); amean += abs(v); }
        amean /= a.size();
        std::printf("  N=1,000,000  depth=%d  boxes=%d\n", tm.depth, tm.nBoxes);
        std::printf("  tree-build (CPU): %.3f s | upload: %.3f s | GPU compute: %.3f s | total: %.3f s\n",
                    tm.treeBuildSec, tm.uploadSec, tm.computeSec, tm.treeBuildSec + tm.uploadSec + tm.computeSec);
        std::printf("  accel sanity: mean|a|=%.4e max|a|=%.4e (finite=%s)\n", amean, amax,
                    std::isfinite(amax) ? "yes" : "NO");
    }

    // ---- Gate D(2): bounded-energy dynamics with GPU FMM (feasible N) -------
    std::printf("\n=== Gate D(2): bounded energy under GPU-FMM dynamics (Plummer N=50000) ===\n");
    {
        PlummerConfig pc; pc.N = 50000; pc.a = 1.0; pc.total_mass = 1.0; pc.G = G; pc.seed = 7;
        Bodies3D b = makePlummerSphere(pc);
        const double eps = 0.01; // << leaf (auto depth ~3 -> leaf ~ box/8)
        const KernelParams3D k{G, eps};
        GpuFmmSolver solver(k, 4, -1);
        VelocityVerlet3D integ(solver);
        integ.initialize(b);
        const double dt = 0.004;
        const Diagnostics3D d0 = computeDiagnostics(b, k);
        double maxRel = 0;
        for (int step = 1; step <= 150; ++step) {
            integ.step(b, dt);
            if (step % 30 == 0 || step == 150) {
                const Diagnostics3D d = computeDiagnostics(b, k);
                maxRel = std::max(maxRel, std::abs(d.total - d0.total) / std::abs(d0.total));
            }
        }
        std::printf("  E0=%.6e  max|dE|/|E0| over t=%.2f : %.3e (bounded=%s)\n",
                    d0.total, 150 * dt, maxRel, maxRel < 1e-2 ? "yes" : "check");
    }

    std::printf("\n==========  RUNG 7 GATES COMPLETE  ==========\n");
    return 0;
}
