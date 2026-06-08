// ============================================================================
// RUNG 8: galaxy / merger simulation driver. Reuses the Rung 7 GPU FMM
// (UNCHANGED) for forces; leapfrog; dumps float32 position frames for the
// offline renderer. Reports energy drift (small N), eps/leaf, per-step time.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../Diagnostics3D.hpp"
#include "../ForceSolver3D.hpp"
#include "../Integrator3D.hpp"
#include "../Types3D.hpp"
#include "FMM3DGPU.cuh"
#include "GalaxyIC.hpp"
#include "AdaptiveFMM3DGPU.cuh" // Rung 11: adaptive GPU FMM solver option
#include "GpuOctree.hpp" // Rung 9: depth-sweep occupancy/box stats (measure mode)

using namespace galaxy3d;

namespace {

// Uniform GPU FMM (Rung 7) -- fixed-depth tree.
struct GpuFmmSolver : ForceSolver3D {
    KernelParams3D k; int p; int depth;
    GpuFmmSolver(KernelParams3D kk, int pp, int d) : k(kk), p(pp), depth(d) {}
    Accelerations3D computeAccelerations(const Bodies3D& b) const override {
        return gpu::fmm3dGpu<float>(b, k.G, k.eps, p, depth);
    }
    const char* name() const override { return "FMM3D-GPU(fp32)"; }
};

// Adaptive GPU FMM (Rung 10b) -- occupancy bounded by ncrit, no dense-core blowup.
struct GpuAdaptiveSolver : ForceSolver3D {
    KernelParams3D k; int p; int ncrit;
    mutable int lastMaxLeafOcc = 0; // eps-floor diagnostic (Rung 12)
    GpuAdaptiveSolver(KernelParams3D kk, int pp, int nc) : k(kk), p(pp), ncrit(nc) {}
    Accelerations3D computeAccelerations(const Bodies3D& b) const override {
        gpu::GpuTiming tm;
        Accelerations3D a = gpu::adaptiveFmm3dGpu<float>(b, k.G, k.eps, p, ncrit, &tm);
        lastMaxLeafOcc = tm.maxLeafOcc;
        return a;
    }
    const char* name() const override { return "AdaptiveFMM3D-GPU(fp32)"; }
};

double boxExtent(const Bodies3D& b) {
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < b.size(); ++i) {
        const Vec3& p = b.pos[i];
        lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
        lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
        lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
    }
    return std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
}

// Choose depth: keep occupancy moderate (GPU P2P is cheap, so a shallower tree
// that shifts work to P2P is the right trade) while ensuring leaf >> eps. Take
// the occupancy-based depth, capped by the eps constraint (leaf >= 16*eps).
int chooseDepth(double side, double eps, std::size_t N) {
    const int Locc = (int)std::lround(std::log(double(N) / 96.0) / std::log(8.0));
    const int Lmax = (int)std::floor(std::log2(side / (16.0 * eps)));
    int L = std::min(Locc, Lmax);
    if (L < 1) L = 1; if (L > 6) L = 6;
    return L;
}

} // namespace

int main(int argc, char** argv) {
    std::string scenario = "disk", out = "outputs/galaxy";
    std::size_t N = 200000;        // per-galaxy particle count
    int p = 4, steps = 600, frameStride = 4, depthOverride = -1;
    double eps = 0.05, dt = 0.01, G = 1.0;
    unsigned seed = 1;
    bool measure = false;
    std::string solverName = "fmm3d"; // "fmm3d" (uniform) | "fmm3d-adaptive"
    int ncrit = 64;                   // adaptive leaf threshold
    // merger choreography (CLI-tunable so trajectory tuning needs no rebuild)
    double mSep = 5.0, mImpact = 1.25, mVx = 0.6, mIncl2 = 1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto v = [&]() { return std::string(argv[++i]); };
        if (a == "--measure") measure = true;
        else if (a == "--solver") solverName = v();
        else if (a == "--ncrit") ncrit = std::stoi(v());
        else if (a == "--scenario") scenario = v();
        else if (a == "--N") N = std::stoul(v());
        else if (a == "--p") p = std::stoi(v());
        else if (a == "--steps") steps = std::stoi(v());
        else if (a == "--frame-stride") frameStride = std::stoi(v());
        else if (a == "--dt") dt = std::stod(v());
        else if (a == "--eps") eps = std::stod(v());
        else if (a == "--depth") depthOverride = std::stoi(v());
        else if (a == "--seed") seed = (unsigned)std::stoul(v());
        else if (a == "--out") out = v();
        else if (a == "--m-sep") mSep = std::stod(v());        // initial |x| of each galaxy
        else if (a == "--m-impact") mImpact = std::stod(v());  // |y| offset (impact param/2)
        else if (a == "--m-vx") mVx = std::stod(v());          // approach speed (each, toward center)
        else if (a == "--m-incl2") mIncl2 = std::stod(v());    // inclination of galaxy 2 (rad)
        else { std::printf("unknown arg %s\n", a.c_str()); return 2; }
    }

    // ---- build initial conditions -----------------------------------------
    Bodies3D bodies;
    std::vector<int> species;
    // Use the adaptive solver for the one-time IC force eval: robust at any N
    // (the uniform solver's auto-depth can be 2 for small per-galaxy N).
    auto icAccel = [&](const Bodies3D& b) { return gpu::adaptiveFmm3dGpu<float>(b, G, eps, p, ncrit); };

    if (scenario == "disk") {
        DiskParams dp; dp.N = N; dp.seed = seed;
        bodies = makeDisk(dp, &species, 0);
        setCircularVelocities(bodies, icAccel(bodies), dp);
    } else if (scenario == "merger") {
        DiskParams d1; d1.N = N; d1.seed = seed; d1.rotSign = 1.0;
        std::vector<int> s1;
        Bodies3D g1 = makeDisk(d1, &s1, 0);
        setCircularVelocities(g1, icAccel(g1), d1);
        placeGalaxy(g1, 0.0, Vec3{-mSep, -mImpact, 0.0}, Vec3{+mVx, 0.0, 0.0});

        DiskParams d2; d2.N = N; d2.seed = seed + 7; d2.rotSign = 1.0;
        std::vector<int> s2;
        Bodies3D g2 = makeDisk(d2, &s2, 1);
        setCircularVelocities(g2, icAccel(g2), d2);
        placeGalaxy(g2, mIncl2, Vec3{+mSep, +mImpact, 0.0}, Vec3{-mVx, 0.0, 0.0});

        bodies = concat(g1, g2, s1, s2, species);
    } else { std::printf("unknown scenario %s\n", scenario.c_str()); return 2; }

    const std::size_t n = bodies.size();
    const double side0 = boxExtent(bodies);
    const int depth = depthOverride >= 0 ? depthOverride : chooseDepth(side0, eps, n);
    const double leaf = side0 / double(1 << depth);
    const KernelParams3D k{G, eps};
    const bool adaptive = (solverName == "fmm3d-adaptive");
    std::unique_ptr<ForceSolver3D> solverPtr;
    if (adaptive) solverPtr = std::make_unique<GpuAdaptiveSolver>(k, p, ncrit);
    else          solverPtr = std::make_unique<GpuFmmSolver>(k, p, depth);
    ForceSolver3D& solver = *solverPtr;
    GpuAdaptiveSolver* adaptSolver = adaptive ? static_cast<GpuAdaptiveSolver*>(solverPtr.get()) : nullptr;

    std::printf("=== galaxy sim (%s, %s) ===\n", scenario.c_str(), solver.name());
    if (adaptive)
        std::printf("N=%zu  p=%d  eps=%.3f  ncrit=%d (adaptive, occupancy-bounded)  dt=%.4f  steps=%d\n",
                    n, p, eps, ncrit, dt, steps);
    else
        std::printf("N=%zu  p=%d  eps=%.3f  depth=%d  leaf=%.3f  eps/leaf=%.3f  dt=%.4f  steps=%d\n",
                    n, p, eps, depth, leaf, eps / leaf, dt, steps);

    // ---- Rung 9 MEASURE mode: depth-sweep stats, one timed force eval -------
    if (measure) {
        const gpu::GpuOctree t = gpu::buildGpuOctree(bodies, depth);
        int maxOcc = 0; double meanOcc = 0; int nonEmptyLeaves = 0;
        for (int l = 0; l < t.nLeaves; ++l) {
            maxOcc = std::max(maxOcc, t.leafCount[l]);
            if (t.leafCount[l] > 0) { meanOcc += t.leafCount[l]; ++nonEmptyLeaves; }
        }
        meanOcc /= std::max(1, nonEmptyLeaves);
        const double epsLeaf = eps / leaf, floor = epsLeaf * epsLeaf;
        gpu::GpuTiming tm;
        gpu::fmm3dGpu<float>(bodies, G, eps, p, depth, 192, &tm); // one timed eval
        const double total = tm.treeBuildSec + tm.uploadSec + tm.computeSec;
        std::printf("MEASURE depth=%d  boxes=%d  maxLeafOcc=%d  meanLeafOcc=%.0f  leaf=%.4f  "
                    "eps/leaf=%.3f  floor~%.2e  tree=%.3fs upload=%.3fs compute=%.3fs total=%.3fs\n",
                    depth, t.nBoxes, maxOcc, meanOcc, leaf, epsLeaf, floor,
                    tm.treeBuildSec, tm.uploadSec, tm.computeSec, total);
        return 0;
    }

    // ---- frame dump setup --------------------------------------------------
    const int nFrames = steps / frameStride + 1;
    std::ofstream fbin(out + "_frames.bin", std::ios::binary);
    { int32_t hN = (int32_t)n, hF = nFrames; fbin.write((char*)&hN, 4); fbin.write((char*)&hF, 4); }
    { std::ofstream sb(out + "_species.bin", std::ios::binary);
      std::vector<int8_t> s8(n); for (std::size_t i = 0; i < n; ++i) s8[i] = (int8_t)species[i];
      sb.write((char*)s8.data(), n); }
    std::vector<float> buf(n * 3);
    auto dumpFrame = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            buf[3 * i] = (float)bodies.pos[i].x; buf[3 * i + 1] = (float)bodies.pos[i].y;
            buf[3 * i + 2] = (float)bodies.pos[i].z;
        }
        fbin.write((char*)buf.data(), sizeof(float) * buf.size());
    };

    const bool doEnergy = (n <= 60000);
    Diagnostics3D d0{};
    std::ofstream encsv;
    // energy curve: O(N^2) diagnostics are only affordable for the tuning N, so
    // dump a dense curve (Rung 12 Gate: bounded AND not climbing) for n<=60000.
    const int energyStride = std::max(1, steps / 50);
    if (doEnergy) {
        d0 = computeDiagnostics(bodies, k);
        encsv.open(out + "_energy.csv");
        encsv << "step,time,total,relDrift\n";
        encsv << 0 << ',' << 0.0 << ',' << d0.total << ',' << 0.0 << '\n';
    }

    // OBJECTIVE INTERACTION GATE: COM separation of the two galaxies each step.
    const bool isMerger = (scenario == "merger");
    auto comSep = [&]() -> double {
        Vec3 c0{}, c1{}; double m0 = 0, m1 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (species[i] == 0) { c0 += bodies.mass[i] * bodies.pos[i]; m0 += bodies.mass[i]; }
            else { c1 += bodies.mass[i] * bodies.pos[i]; m1 += bodies.mass[i]; }
        }
        if (m0 <= 0 || m1 <= 0) return 0.0;
        return abs((1.0 / m0) * c0 - (1.0 / m1) * c1);
    };
    std::ofstream sepcsv;
    double minSep = 1e300; int periStep = 0; const double diskRadius = 4.0; // DiskParams.rMax
    if (isMerger) {
        sepcsv.open(out + "_sep.csv"); sepcsv << "step,time,sep\n";
        const double s = comSep();
        sepcsv << 0 << ',' << 0.0 << ',' << s << '\n';
        minSep = s;
    }

    // ---- leapfrog ----------------------------------------------------------
    VelocityVerlet3D integ(solver);
    integ.initialize(bodies);
    dumpFrame();
    double maxRelDrift = 0.0;
    int peakOcc = 0, peakOccStep = 0; // eps-floor diagnostic: max leaf occupancy
    auto t0 = std::chrono::steady_clock::now();
    auto tProg = t0;
    for (int step = 1; step <= steps; ++step) {
        integ.step(bodies, dt);
        if (adaptSolver && adaptSolver->lastMaxLeafOcc > peakOcc) {
            peakOcc = adaptSolver->lastMaxLeafOcc; peakOccStep = step;
        }
        if (step % frameStride == 0) dumpFrame();
        double curSep = 0.0;
        if (isMerger) {
            curSep = comSep();
            if (step % 5 == 0 || step == steps) sepcsv << step << ',' << step * dt << ',' << curSep << '\n';
            if (curSep < minSep) { minSep = curSep; periStep = step; }
        }
        // live progress (so a long 1M run can be monitored for pericenter spikes)
        if (step % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            const double recent = std::chrono::duration<double>(now - tProg).count() / 100.0;
            std::printf("  step %d/%d  t=%.2f  sep=%.2f  per-step(recent)=%.3f s  maxLeafOcc=%d\n",
                        step, steps, step * dt, curSep, recent,
                        adaptSolver ? adaptSolver->lastMaxLeafOcc : 0);
            std::fflush(stdout);
            tProg = now;
        }
        if (doEnergy && (step % energyStride == 0 || step == steps)) {
            const Diagnostics3D d = computeDiagnostics(bodies, k);
            const double rel = std::abs(d.total - d0.total) / std::abs(d0.total);
            maxRelDrift = std::max(maxRelDrift, rel);
            encsv << step << ',' << step * dt << ',' << d.total << ',' << rel << '\n';
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    fbin.close();
    if (isMerger) sepcsv.close();
    if (doEnergy) encsv.close();

    const double secPerStep = std::chrono::duration<double>(t1 - t0).count() / steps;
    std::printf("frames=%d  per-step=%.4f s  total-sim=%.1f s\n", nFrames, secPerStep, secPerStep * steps);
    if (adaptSolver)
        std::printf("PEAK LEAF OCCUPANCY: %d at step %d (t=%.2f)  [eps-floor diagnostic: "
                    "core leaves cap at edge~eps=%.3f, occupancy bounded => per-step tractable]\n",
                    peakOcc, peakOccStep, peakOccStep * dt, eps);
    if (isMerger)
        std::printf("SEPARATION GATE: minSep=%.3f at step=%d (t=%.2f, %.0f%% of run)  diskRadius=%.2f  "
                    "=> encounter=%s\n", minSep, periStep, periStep * dt, 100.0 * periStep / steps,
                    diskRadius, minSep <= diskRadius ? "YES" : "NO");
    if (doEnergy) std::printf("E0=%.6e  max|dE|/|E0|=%.3e (bounded=%s)\n",
                              d0.total, maxRelDrift, maxRelDrift < 0.1 ? "yes" : "check");
    std::printf("wrote %s_frames.bin (%d frames, N=%zu)\n", out.c_str(), nFrames, n);
    return 0;
}
