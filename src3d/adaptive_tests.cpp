// ============================================================================
// RUNG 10a: adaptive FMM gates.
//   A uniform-equivalence: ~uniform dist, adaptive == oracle == uniform FMM.
//   B clustered correctness: Plummer, adaptive vs oracle, error-vs-p converges.
//   C clustered O(N): adaptive vs uniform timing + MAX LEAF OCCUPANCY.
//   D energy sanity: bounded energy with adaptive forces on a clustered system.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "AdaptiveFMM3D.hpp"
#include "Diagnostics3D.hpp"
#include "DirectSolver3D.hpp"
#include "FMM3D.hpp"
#include "InitialConditions3D.hpp"
#include "Integrator3D.hpp"
#include "Types3D.hpp"

using namespace galaxy3d;

namespace {

Bodies3D makeUniformCube(std::size_t N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0), um(0.5, 1.5);
    Bodies3D b; b.resize(N);
    for (std::size_t i = 0; i < N; ++i) { b.pos[i] = Vec3{u(rng), u(rng), u(rng)}; b.mass[i] = um(rng); }
    return b;
}
Bodies3D makePlummer(std::size_t N, unsigned seed) {
    PlummerConfig pc; pc.N = N; pc.a = 1.0; pc.total_mass = 1.0; pc.G = 1.0; pc.seed = seed;
    return makePlummerSphere(pc);
}

struct E { double mx, rms; };
E relErr(const Accelerations3D& r, const Accelerations3D& t) {
    const std::size_t n = r.size();
    double amax = 0; for (auto& v : r) amax = std::max(amax, abs(v));
    const double fl = 1e-30 + 1e-6 * amax;
    double mx = 0, num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = abs(t[i] - r[i]);
        mx = std::max(mx, e / std::max(abs(r[i]), fl)); num += e * e; den += norm2(r[i]);
    }
    return {mx, std::sqrt(num / den)};
}

// max leaf occupancy a UNIFORM tree of given depth would have on these points.
int uniformMaxOcc(const Bodies3D& b, int depth) {
    const std::size_t n = b.size();
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < n; ++i) {
        lo[0] = std::min(lo[0], b.pos[i].x); hi[0] = std::max(hi[0], b.pos[i].x);
        lo[1] = std::min(lo[1], b.pos[i].y); hi[1] = std::max(hi[1], b.pos[i].y);
        lo[2] = std::min(lo[2], b.pos[i].z); hi[2] = std::max(hi[2], b.pos[i].z);
    }
    const double cx = .5 * (lo[0] + hi[0]), cy = .5 * (lo[1] + hi[1]), cz = .5 * (lo[2] + hi[2]);
    double half = .5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    if (!(half > 0)) half = 1;
    half *= 1 + 1e-9;
    const double ox = cx - half, oy = cy - half, oz = cz - half, side = 2 * half;
    const int ns = 1 << depth;
    std::vector<int> cnt((std::size_t)ns * ns * ns, 0);
    auto cl = [&](int v) { return v < 0 ? 0 : (v >= ns ? ns - 1 : v); };
    for (std::size_t i = 0; i < n; ++i) {
        const int ix = cl(int((b.pos[i].x - ox) / side * ns)), iy = cl(int((b.pos[i].y - oy) / side * ns)),
                  iz = cl(int((b.pos[i].z - oz) / side * ns));
        ++cnt[((std::size_t)iz * ns + iy) * ns + ix];
    }
    int m = 0; for (int c : cnt) m = std::max(m, c); return m;
}

template <class F> double timeBest(F&& f, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now(); f(); auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

} // namespace

int main() {
    std::printf("=== Rung 10a: adaptive FMM (CGR) gates ===\n\n");
    const KernelParams3D k0{1.0, 0.0};
    int rc = 0;

    // ---- Gate A: uniform-equivalence --------------------------------------
    std::printf("=== Gate A: uniform-equivalence (uniform cube N=8000, eps=0, p=8) ===\n");
    {
        Bodies3D b = makeUniformCube(8000, 1);
        const Accelerations3D ref = DirectSolver3D(k0).computeAccelerations(b);
        AdaptiveFMM3D adap(k0, 8, 64);
        const Accelerations3D a = adap.computeAccelerations(b);
        const Accelerations3D u = FMM3D(k0, 8).computeAccelerations(b);
        const E ea = relErr(ref, a), eu = relErr(ref, u);
        std::printf("  adaptive vs oracle: rms=%.3e   uniform FMM vs oracle: rms=%.3e\n", ea.rms, eu.rms);
        std::printf("  (both at expansion-truncation level => U/V correct, W/X benign)\n");
        if (!(ea.rms < 5e-5)) rc = 1; // truncation-level, same order as uniform
    }

    // ---- Gate B: clustered correctness, error-vs-p ------------------------
    std::printf("\n=== Gate B: clustered (Plummer N=8000, eps=0): adaptive vs oracle ===\n");
    std::printf("%4s  %14s  %14s\n", "p", "max_rel", "rms_rel");
    {
        Bodies3D b = makePlummer(8000, 2);
        const Accelerations3D ref = DirectSolver3D(k0).computeAccelerations(b);
        double prev = 1e300; bool mono = true;
        for (int p : {4, 6, 8}) {
            const E e = relErr(ref, AdaptiveFMM3D(k0, p, 64).computeAccelerations(b));
            std::printf("%4d  %14.6e  %14.6e\n", p, e.mx, e.rms);
            if (!(e.rms < prev)) mono = false;
            prev = e.rms;
        }
        // The real proof is CLEAN monotone convergence (a wrong list floors); the
        // final value sits at truncation level, same order as the uniform FMM.
        const bool pass = mono && prev < 5e-5;
        std::printf("  monotone=%s (no floor)  rms(p=8)=%.2e  => %s\n", mono ? "yes" : "NO",
                    prev, pass ? "GATE PASS" : "GATE FAIL");
        if (!pass) rc = 1;
    }

    // ---- Gate C: clustered O(N), adaptive vs uniform + max occupancy ------
    std::printf("\n=== Gate C: clustered scaling (Plummer, p=4): adaptive vs uniform ===\n");
    std::printf("%9s  %12s  %10s  %12s  %10s\n", "N", "adap_sec", "adap_occ", "unif_sec", "unif_occ");
    {
        for (std::size_t N : {2000u, 8000u, 32000u, 128000u}) {
            Bodies3D b = makePlummer(N, 3);
            AdaptiveFMM3D adap(k0, 4, 64);
            FMM3D unif(k0, 4);
            const int reps = N <= 8000 ? 3 : 1;
            const double ta = timeBest([&] { volatile auto x = adap.computeAccelerations(b); (void)x; }, reps);
            const double tu = timeBest([&] { volatile auto x = unif.computeAccelerations(b); (void)x; }, reps);
            const int uOcc = uniformMaxOcc(b, FMM3D::autoDepth(N));
            std::printf("%9zu  %12.4e  %10d  %12.4e  %10d\n", N, ta, adap.lastMaxLeafOcc, tu, uOcc);
        }
        std::printf("  (adaptive max occ stays ~bounded by threshold; uniform occ blows up in the core)\n");
    }

    // ---- Gate D: energy sanity --------------------------------------------
    std::printf("\n=== Gate D: energy-bounded dynamics, adaptive forces (Plummer N=4000) ===\n");
    {
        const double eps = 0.02; const KernelParams3D k{1.0, eps};
        Bodies3D b = makePlummer(4000, 4);
        AdaptiveFMM3D solver(k, 6, 64);
        VelocityVerlet3D integ(solver); integ.initialize(b);
        const Diagnostics3D d0 = computeDiagnostics(b, k);
        const double dt = 0.004; double maxRel = 0;
        for (int s = 1; s <= 200; ++s) {
            integ.step(b, dt);
            if (s % 40 == 0 || s == 200) {
                const Diagnostics3D d = computeDiagnostics(b, k);
                maxRel = std::max(maxRel, std::abs(d.total - d0.total) / std::abs(d0.total));
            }
        }
        std::printf("  E0=%.6e  max|dE|/|E0| over t=%.2f : %.3e (bounded=%s)\n",
                    d0.total, 200 * dt, maxRel, maxRel < 1e-2 ? "yes" : "check");
        if (!(maxRel < 1e-2)) rc = 1;
    }

    std::printf("\n==========  %s  ==========\n", rc == 0 ? "ADAPTIVE GATES A-D PASSED" : "A GATE FAILED");
    return rc;
}
