// ============================================================================
// RUNG 5 validation harness (3D assembled solvers).
//
//   --mode fmm      Gate A (depth-0 reduction), B (small, depth 2), C (global
//                   error-vs-p), all FMM3D vs DirectSolver3D; plus the softening
//                   floor: force-error RMS vs eps/leaf at fixed p.
//   --mode scaling  log-log wall-clock vs N for Direct3D, BarnesHut3D, FMM3D on
//                   uniform-random points in a cube (the 3D hero plot).
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "BarnesHut3D.hpp"
#include "DirectSolver3D.hpp"
#include "FMM3D.hpp"
#include "Kernel3D.hpp"
#include "Types3D.hpp"

using namespace galaxy3d;

namespace {

// Uniform-random points in [-1,1]^3, masses in [0.5,1.5].
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
    const double floor = 1e-6 * amax;
    double mx = 0, num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = abs(test[i] - ref[i]);
        mx = std::max(mx, e / std::max(abs(ref[i]), floor));
        num += e * e; den += norm2(ref[i]);
    }
    return {mx, std::sqrt(num / den)};
}

template <class F>
double timeBest(F&& fn, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int runFmm(unsigned seed) {
    int rc = 0;
    const KernelParams3D k0{1.0, 0.0};
    std::ofstream csv("outputs/fmm3d_solver_error.csv");
    csv << "test,p,max_rel,rms_rel\n";

    // Gate A: depth-0 reduction.
    std::printf("=== Gate A: depth-0 reduction (FMM3D == DirectSolver3D, eps=0.05) ===\n");
    {
        const KernelParams3D ke{1.0, 0.05};
        Bodies3D b = makeUniformCube(2000, seed);
        const Accelerations3D ref = DirectSolver3D(ke).computeAccelerations(b);
        const Accelerations3D got = FMM3D(ke, 8, /*depth=*/0).computeAccelerations(b);
        const ErrStats e = relErr(ref, got);
        const bool pass = e.max_rel < 1e-11;
        std::printf("  max_rel=%.3e rms_rel=%.3e => %s\n", e.max_rel, e.rms_rel,
                    pass ? "GATE PASS" : "GATE FAIL");
        if (!pass) rc = 1;
    }

    // Gate B: small uniform cube, depth 2, eps=0.
    std::printf("\n=== Gate B: small system, depth 2 (N=512, eps=0) ===\n");
    std::printf("%4s  %14s  %14s\n", "p", "max_rel", "rms_rel");
    {
        Bodies3D b = makeUniformCube(512, seed + 1);
        const Accelerations3D ref = DirectSolver3D(k0).computeAccelerations(b);
        double prev = 1e300; bool mono = true;
        for (int p : {2, 4, 6, 8}) {
            const ErrStats e = relErr(ref, FMM3D(k0, p, /*depth=*/2).computeAccelerations(b));
            std::printf("%4d  %14.6e  %14.6e\n", p, e.max_rel, e.rms_rel);
            csv << "small," << p << ',' << e.max_rel << ',' << e.rms_rel << '\n';
            if (!(e.rms_rel < prev)) mono = false;
            prev = e.rms_rel;
        }
        std::printf("  monotone(rms)=%s => %s\n", mono ? "yes" : "NO", mono ? "GATE PASS" : "GATE FAIL");
        if (!mono) rc = 1;
    }

    // Gate C: global error-vs-p, eps=0, auto depth.
    const std::size_t Nc = 8000;
    std::printf("\n=== Gate C: global error-vs-p (N=%zu, eps=0, depth auto=%d) ===\n",
                Nc, FMM3D::autoDepth(Nc));
    std::printf("%4s  %14s  %14s\n", "p", "max_rel(acc)", "rms_rel(acc)");
    {
        Bodies3D b = makeUniformCube(Nc, seed + 2);
        const Accelerations3D ref = DirectSolver3D(k0).computeAccelerations(b);
        double prevR = 1e300, prevM = 1e300, lastR = 1e300; bool mono = true;
        for (int p : {4, 6, 8, 10}) {
            const ErrStats e = relErr(ref, FMM3D(k0, p).computeAccelerations(b));
            std::printf("%4d  %14.6e  %14.6e\n", p, e.max_rel, e.rms_rel);
            csv << "global," << p << ',' << e.max_rel << ',' << e.rms_rel << '\n';
            if (!(e.rms_rel < prevR) || !(e.max_rel < prevM)) mono = false;
            prevR = e.rms_rel; prevM = e.max_rel; lastR = e.rms_rel;
        }
        const bool pass = mono && lastR < 1e-5;
        std::printf("  monotone=%s rms(p=10)<1e-5=%s => %s\n", mono ? "yes" : "NO",
                    lastR < 1e-5 ? "yes" : "NO", pass ? "GATE PASS" : "GATE FAIL");
        if (!pass) rc = 1;
    }

    // Softening floor: far field uses the pure kernel, so force error floors at
    // ~(eps/leaf)^2 for well-separated pairs. Report it explicitly vs eps/leaf.
    std::printf("\n=== Softening floor: FMM3D force error vs eps/leaf (N=8000, p=8) ===\n");
    std::printf("%10s  %10s  %14s  %14s\n", "eps", "eps/leaf", "max_rel", "rms_rel");
    {
        Bodies3D b = makeUniformCube(8000, seed + 2);
        const int L = FMM3D::autoDepth(8000);
        const double leaf = 2.0 / (1 << L); // cube side 2 (= [-1,1]) / 2^L
        for (double eps : {0.05, 0.02, 0.01, 0.005, 0.002}) {
            const KernelParams3D ke{1.0, eps};
            const Accelerations3D ref = DirectSolver3D(ke).computeAccelerations(b);
            const ErrStats e = relErr(ref, FMM3D(ke, 8).computeAccelerations(b));
            std::printf("%10.4f  %10.4f  %14.6e  %14.6e\n", eps, eps / leaf, e.max_rel, e.rms_rel);
            csv << "softening_eps" << eps << ",8," << e.max_rel << ',' << e.rms_rel << '\n';
        }
        std::printf("  (leaf size = %.4f at depth %d; error floors as eps/leaf grows)\n", leaf, L);
    }

    csv.close();
    std::printf("\nwrote outputs/fmm3d_solver_error.csv\n");
    std::printf("\n==========  %s  ==========\n", rc == 0 ? "FMM3D GATES A/B/C PASSED" : "FMM3D GATE FAILED");
    return rc;
}

int runScaling(unsigned seed, double theta, int pFMM) {
    // Occupancy-aligned sweep N = 64 * 8^L (L=0..4): the FMM octree holds mean
    // leaf occupancy ~constant (=64) at every point, so the per-point cost grows
    // linearly and the O(N) slope is not masked by depth-step occupancy swings
    // (integer octree depth changes occupancy by 8x in 3D).
    const std::vector<std::size_t> Ns = {64, 512, 4096, 32768, 262144};
    const std::size_t directCap = 32768;
    std::ofstream csv("outputs/scaling3d.csv");
    csv << "N,direct_sec,bh_sec,fmm_sec\n";
    std::printf("=== 3D force-eval wall-clock (uniform cube; BH theta=%.2f; FMM p=%d) ===\n", theta, pFMM);
    std::printf("%10s  %12s  %12s  %12s  %10s\n", "N", "direct_s", "bh_s", "fmm_s", "fmm/bh");

    long fmmVsBh = -1;
    for (std::size_t N : Ns) {
        Bodies3D b = makeUniformCube(N, seed);
        DirectSolver3D direct({1.0, 0.05});
        BarnesHut3D bh({1.0, 0.05}, theta);
        FMM3D fmm({1.0, 0.05}, pFMM);
        const int reps = N <= 4000 ? 5 : (N <= 32000 ? 3 : 1);
        const double tb = timeBest([&] { volatile auto a = bh.computeAccelerations(b); (void)a; }, reps);
        const double tf = timeBest([&] { volatile auto a = fmm.computeAccelerations(b); (void)a; }, reps);
        double td = -1;
        if (N <= directCap) td = timeBest([&] { volatile auto a = direct.computeAccelerations(b); (void)a; }, reps);

        if (td > 0) std::printf("%10zu  %12.4e  %12.4e  %12.4e  %9.2fx\n", N, td, tb, tf, tf / tb);
        else        std::printf("%10zu  %12s  %12.4e  %12.4e  %9.2fx\n", N, "(skip)", tb, tf, tf / tb);
        csv << N << ','; if (td > 0) csv << td; csv << ',' << tb << ',' << tf << '\n';
        if (fmmVsBh < 0 && tf < tb) fmmVsBh = (long)N;
    }
    std::printf("wrote outputs/scaling3d.csv\n");
    if (fmmVsBh > 0) std::printf("FMM3D overtakes BarnesHut3D at N >= %ld\n", fmmVsBh);
    else std::printf("FMM3D did not overtake BH within the tested N range\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "fmm";
    unsigned seed = 42;
    double theta = 0.5;
    int p = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) std::exit(2); return argv[++i]; };
        if (a == "--mode") mode = val();
        else if (a == "--seed") seed = (unsigned)std::stoul(val());
        else if (a == "--theta") theta = std::stod(val());
        else if (a == "--p") p = std::stoi(val());
        else { std::printf("unknown option: %s\n", a.c_str()); return 2; }
    }
    if (mode == "fmm") return runFmm(seed);
    if (mode == "scaling") return runScaling(seed, theta, p);
    std::printf("unknown mode: %s\n", mode.c_str());
    return 2;
}
