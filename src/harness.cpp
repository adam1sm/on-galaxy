// ============================================================================
// Rung 1 validation harness.
//
// Two jobs, both writing CSV artifacts under outputs/ and printing a summary:
//
//   --mode error    Force-comparison vs the DirectSolver oracle at fixed N.
//                   Sweeps theta and reports MAX and RMS relative acceleration
//                   error per body. Enforces the theta=0 correctness GATE
//                   (BH must reduce to direct summation, matching to roundoff).
//
//   --mode scaling  Wall-clock per single force evaluation for DirectSolver and
//                   BarnesHutSolver across a sweep of N. Writes a CSV that the
//                   Python script turns into the log-log time-vs-N hero plot.
//                   (A clearly-marked column seam is left for a future FMM line.)
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "BarnesHutSolver.hpp"
#include "DirectSolver.hpp"
#include "FMMSolver.hpp"
#include "InitialConditions.hpp"
#include "Kernel.hpp"
#include "Types.hpp"

using namespace galaxy;

namespace {

Bodies makeDisk(std::size_t N, unsigned seed) {
    DiskConfig dc;
    dc.N = N;
    dc.radius = 1.0;
    dc.total_mass = 1.0;
    dc.rotation = 0.6;
    dc.thermal = 0.05;
    dc.seed = seed;
    return makeRandomDisk(dc);
}

// Uniform-random points in [-1,1]^2 with masses in [0.5,1.5] -- the standard FMM
// benchmark distribution (uniform leaf occupancy, so depth ~ log4(N) is O(N)).
Bodies makeUniform(std::size_t N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> um(0.5, 1.5);
    Bodies b;
    b.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        b.pos[i] = Complex{u(rng), u(rng)};
        b.mass[i] = um(rng);
    }
    return b;
}

struct ErrStats { double max_rel; double rms_rel; };

// Per-body relative acceleration error of BH vs the direct oracle. The
// denominator is floored at a small fraction of the peak |a| so that a handful
// of near-zero-net-acceleration bodies (e.g. near the cluster center) can't
// dominate the metric with meaningless ratios.
ErrStats relativeError(const Accelerations& ref, const Accelerations& test) {
    const std::size_t n = ref.size();
    double amax = 0.0;
    for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::abs(ref[i]));
    const double floor = 1e-6 * amax;

    double max_rel = 0.0, sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double denom = std::max(std::abs(ref[i]), floor);
        const double e = std::abs(test[i] - ref[i]) / denom;
        max_rel = std::max(max_rel, e);
        sumsq += e * e;
    }
    return {max_rel, std::sqrt(sumsq / static_cast<double>(n))};
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

int runError(std::size_t N, unsigned seed, const KernelParams& k) {
    Bodies bodies = makeDisk(N, seed);
    DirectSolver direct(k);
    const Accelerations ref = direct.computeAccelerations(bodies);

    const std::vector<double> thetas = {0.0, 0.1, 0.3, 0.5, 0.8};

    std::ofstream csv("outputs/bh_error_vs_theta.csv");
    csv << "theta,max_rel_err,rms_rel_err\n";

    std::printf("=== force-comparison vs DirectSolver oracle (N=%zu, seed=%u) ===\n", N, seed);
    std::printf("%8s  %14s  %14s\n", "theta", "max_rel_err", "rms_rel_err");

    const double GATE_TOL = 1e-9; // theta=0 must match the oracle to roundoff
    int rc = 0;
    for (double th : thetas) {
        BarnesHutSolver bh(k, th);
        const Accelerations test = bh.computeAccelerations(bodies);
        const ErrStats e = relativeError(ref, test);
        std::printf("%8.2f  %14.6e  %14.6e%s\n", th, e.max_rel, e.rms_rel,
                    th == 0.0 ? "   <-- correctness gate" : "");
        csv << th << ',' << e.max_rel << ',' << e.rms_rel << '\n';

        if (th == 0.0) {
            if (e.max_rel > GATE_TOL) {
                std::printf("GATE FAIL: theta=0 max_rel_err %.3e exceeds tol %.1e\n",
                            e.max_rel, GATE_TOL);
                rc = 1;
            } else {
                std::printf("GATE PASS: theta=0 reduces to direct summation (max_rel_err %.3e <= %.1e)\n",
                            e.max_rel, GATE_TOL);
            }
        }
    }
    std::printf("wrote outputs/bh_error_vs_theta.csv\n");
    return rc;
}

// Hero scaling plot: wall-clock per single force evaluation for Direct, BH and
// FMM on UNIFORM-RANDOM points (the standard FMM benchmark), depth auto from N.
// Direct is skipped above directCap (too slow); its column is left blank.
int runScaling(const std::vector<std::size_t>& Ns, unsigned seed,
               const KernelParams& k, double theta, int pFMM, std::size_t directCap) {
    std::ofstream csv("outputs/scaling.csv");
    csv << "N,direct_sec,bh_sec,fmm_sec\n";

    std::printf("=== single force-evaluation wall-clock "
                "(uniform-random; BH theta=%.2f; FMM p=%d, depth auto) ===\n", theta, pFMM);
    std::printf("%10s  %12s  %12s  %12s  %10s\n",
                "N", "direct_s", "bh_s", "fmm_s", "fmm/bh");

    long bhVsDirect = -1, fmmVsBh = -1;
    for (std::size_t N : Ns) {
        Bodies bodies = makeUniform(N, seed);
        DirectSolver direct(k);
        BarnesHutSolver bh(k, theta);
        FMMSolver fmm(k, pFMM);

        const int reps = N <= 4000 ? 8 : (N <= 32000 ? 4 : (N <= 128000 ? 2 : 1));
        const double tb = timeBest([&] { volatile auto a = bh.computeAccelerations(bodies); (void)a; }, reps);
        const double tf = timeBest([&] { volatile auto a = fmm.computeAccelerations(bodies); (void)a; }, reps);
        double td = -1.0;
        if (N <= directCap)
            td = timeBest([&] { volatile auto a = direct.computeAccelerations(bodies); (void)a; }, reps);

        if (td > 0)
            std::printf("%10zu  %12.4e  %12.4e  %12.4e  %9.2fx\n", N, td, tb, tf, tf / tb);
        else
            std::printf("%10zu  %12s  %12.4e  %12.4e  %9.2fx\n", N, "(skip)", tb, tf, tf / tb);

        csv << N << ',';
        if (td > 0) csv << td;             // blank if skipped -> NaN in the plot
        csv << ',' << tb << ',' << tf << '\n';

        if (td > 0 && bhVsDirect < 0 && tb < td) bhVsDirect = static_cast<long>(N);
        if (fmmVsBh < 0 && tf < tb) fmmVsBh = static_cast<long>(N);
    }
    std::printf("wrote outputs/scaling.csv\n");
    if (bhVsDirect > 0) std::printf("BH overtakes Direct at N >= %ld\n", bhVsDirect);
    if (fmmVsBh > 0) std::printf("FMM overtakes BH at N >= %ld\n", fmmVsBh);
    else std::printf("FMM did not overtake BH within the tested N range\n");
    return 0;
}

// ---- Gates A (depth-0 reduction), B (small), C (global error-vs-p) ---------
int runFmm(unsigned seed) {
    int rc = 0;
    const KernelParams k0{1.0, 0.0}; // eps = 0 isolates truncation error

    std::ofstream csv("outputs/fmm_error_vs_p.csv");
    csv << "test,p,max_rel,rms_rel\n";

    // Gate A: depth 0 -> one leaf -> pure P2P -> must equal DirectSolver (same eps).
    std::printf("=== Gate A: depth-0 reduction (FMM == DirectSolver, eps=0.05) ===\n");
    {
        const KernelParams ke{1.0, 0.05};
        Bodies b = makeUniform(2000, seed);
        const Accelerations ref = DirectSolver(ke).computeAccelerations(b);
        const Accelerations got = FMMSolver(ke, 8, /*depth=*/0).computeAccelerations(b);
        const ErrStats e = relativeError(ref, got);
        const bool pass = e.max_rel < 1e-11;
        std::printf("  max_rel=%.3e rms_rel=%.3e  =>  %s\n",
                    e.max_rel, e.rms_rel, pass ? "GATE PASS" : "GATE FAIL");
        if (!pass) rc = 1;
    }

    // Gate B: small uniform set, shallow tree (depth 3), eps=0; error shrinks with p.
    std::printf("\n=== Gate B: small system, depth 3 (N=256, eps=0) ===\n");
    std::printf("%4s  %14s  %14s\n", "p", "max_rel", "rms_rel");
    {
        Bodies b = makeUniform(256, seed + 1);
        const Accelerations ref = DirectSolver(k0).computeAccelerations(b);
        double prev = 1e300;
        bool mono = true;
        for (int p : {2, 4, 6, 8}) {
            const Accelerations got = FMMSolver(k0, p, /*depth=*/3).computeAccelerations(b);
            const ErrStats e = relativeError(ref, got);
            std::printf("%4d  %14.6e  %14.6e\n", p, e.max_rel, e.rms_rel);
            csv << "small," << p << ',' << e.max_rel << ',' << e.rms_rel << '\n';
            if (!(e.rms_rel < prev)) mono = false;
            prev = e.rms_rel;
        }
        std::printf("  monotonic(rms)=%s  =>  %s\n", mono ? "yes" : "NO",
                    mono ? "GATE PASS" : "GATE FAIL");
        if (!mono) rc = 1;
    }

    // Gate C: global FMM vs DirectSolver, larger uniform set, eps=0, auto depth.
    std::printf("\n=== Gate C: global error-vs-p (N=8000, eps=0, depth auto=%d) ===\n",
                FMMSolver::autoDepth(8000));
    std::printf("%4s  %14s  %14s\n", "p", "max_rel", "rms_rel");
    {
        Bodies b = makeUniform(8000, seed + 2);
        const Accelerations ref = DirectSolver(k0).computeAccelerations(b);
        double prevRms = 1e300, prevMax = 1e300, lastRms = 1e300;
        bool mono = true;
        for (int p : {4, 8, 12}) {
            const Accelerations got = FMMSolver(k0, p).computeAccelerations(b);
            const ErrStats e = relativeError(ref, got);
            std::printf("%4d  %14.6e  %14.6e\n", p, e.max_rel, e.rms_rel);
            csv << "global," << p << ',' << e.max_rel << ',' << e.rms_rel << '\n';
            if (!(e.rms_rel < prevRms) || !(e.max_rel < prevMax)) mono = false;
            prevRms = e.rms_rel; prevMax = e.max_rel; lastRms = e.rms_rel;
        }
        const bool pass = mono && lastRms < 1e-6;
        std::printf("  monotonic=%s  rms(p=12)<1e-6=%s  =>  %s\n",
                    mono ? "yes" : "NO", lastRms < 1e-6 ? "yes" : "NO",
                    pass ? "GATE PASS" : "GATE FAIL");
        if (!pass) rc = 1;
    }

    // Diagnostic: FMM accuracy on the CLUSTERED disk vs softening eps. The
    // far-field uses the pure kernel, so "eps only in P2P" requires eps << leaf
    // size; otherwise the closest well-separated pairs carry a p-INDEPENDENT
    // ~(eps/r)^2 kernel mismatch. This sweep shows where the assumption holds
    // (error shrinks with p) and picks the eps for the Gate E dynamics.
    std::printf("\n=== Diagnostic: FMM vs Direct on disk IC (N=500, depth=%d) ===\n",
                FMMSolver::autoDepth(500));
    std::printf("%8s  %4s  %14s  %14s\n", "eps", "p", "max_rel", "rms_rel");
    {
        Bodies b = makeDisk(500, 42);
        for (double eps : {0.05, 0.01, 0.005}) {
            const KernelParams ke{1.0, eps};
            const Accelerations ref = DirectSolver(ke).computeAccelerations(b);
            for (int p : {4, 8, 12}) {
                const Accelerations got = FMMSolver(ke, p).computeAccelerations(b);
                const ErrStats e = relativeError(ref, got);
                std::printf("%8.3f  %4d  %14.6e  %14.6e\n", eps, p, e.max_rel, e.rms_rel);
            }
        }
    }

    csv.close();
    std::printf("\nwrote outputs/fmm_error_vs_p.csv\n");
    std::printf("\n==========  %s  ==========\n",
                rc == 0 ? "FMM GATES A/B/C PASSED" : "FMM GATE FAILED");
    return rc;
}

void usage() {
    std::printf(
        "Usage: harness --mode <error|scaling|fmm> [options]\n"
        "  --mode error      BH force-comparison + theta=0 gate (default N=4000)\n"
        "  --mode scaling    log-log timing sweep (Direct, BH, FMM)\n"
        "  --mode fmm        FMM gates A (depth-0), B (small), C (global err-vs-p)\n"
        "  --N <int>         N for error mode (default 4000)\n"
        "  --theta <float>   theta for scaling mode (default 0.5)\n"
        "  --p <int>         FMM order for scaling mode (default 4)\n"
        "  --eps <float>     softening (default 0.05)\n"
        "  --G <float>       grav constant (default 1)\n"
        "  --seed <int>      RNG seed (default 42)\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "error";
    std::size_t N = 4000;
    double theta = 0.5, eps = 0.05, G = 1.0;
    unsigned seed = 42;
    int p = 4; // FMM order for the scaling sweep

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mode") mode = val();
        else if (a == "--N") N = std::stoul(val());
        else if (a == "--theta") theta = std::stod(val());
        else if (a == "--p") p = std::stoi(val());
        else if (a == "--eps") eps = std::stod(val());
        else if (a == "--G") G = std::stod(val());
        else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(val()));
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::printf("unknown option: %s\n", a.c_str()); usage(); return 2; }
    }

    const KernelParams k{G, eps};
    if (mode == "error") {
        return runError(N, seed, k);
    } else if (mode == "fmm") {
        return runFmm(seed);
    } else if (mode == "scaling") {
        // Uniform-random sweep spanning Direct/BH/FMM. Direct is skipped above
        // 64k (too slow); BH and FMM continue so the FMM-vs-BH crossover shows.
        const std::vector<std::size_t> Ns =
            {100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000};
        return runScaling(Ns, seed, k, theta, p, /*directCap=*/64000);
    }
    std::printf("unknown mode: %s\n", mode.c_str());
    usage();
    return 2;
}
