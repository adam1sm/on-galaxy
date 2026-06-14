// ============================================================================
// RUNG 2a: isolated operator unit tests for the 2D FMM expansions.
//
// Each operator (P2M, M2L, M2M, L2L) is validated IN ISOLATION:
//   * potential      vs direct summation  phi(z) = sum_j q_j ln|z - z_j|
//   * acceleration   vs DirectSolver (the oracle), eps = 0
// For every operator we report a relative-error-vs-p table for BOTH quantities
// and GATE on (a) monotonic decrease as p grows and (b) a small final error.
//
// No tree, no passes, no interaction lists, no P2P, no ForceSolver wiring.
// ============================================================================
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "DirectSolver.hpp"   // the oracle (used for acceleration ground truth)
#include "FmmExpansions.hpp"  // the operators under test
#include "Kernel.hpp"
#include "Types.hpp"

using namespace galaxy;
using galaxy::fmm::Local;
using galaxy::fmm::Multipole;

namespace {

const double TAU = 2.0 * std::acos(-1.0);
const std::vector<int> PS = {2, 4, 6, 8, 10, 12};
const double FINAL_TOL = 1e-5; // required relative error at p = 12

std::ofstream g_csv;

// ---- ground truth ----------------------------------------------------------

// Direct potential phi(z) = sum_j q_j ln|z - z_j|  (Re of w(z); see FmmExpansions).
double directPotential(Complex z, const std::vector<Complex>& spos,
                       const std::vector<double>& sq) {
    double s = 0.0;
    for (std::size_t j = 0; j < spos.size(); ++j)
        s += sq[j] * std::log(std::abs(z - spos[j]));
    return s;
}

// Acceleration at each target from the sources, computed by DirectSolver with
// eps = 0. Targets are appended as zero-mass bodies so the oracle returns the
// pure source field on them (zero-mass bodies neither feel self nor pull others).
std::vector<Complex> directAccel(const std::vector<Complex>& spos,
                                 const std::vector<double>& sq,
                                 const std::vector<Complex>& tpos) {
    const std::size_t ns = spos.size(), nt = tpos.size();
    Bodies b;
    b.resize(ns + nt);
    for (std::size_t i = 0; i < ns; ++i) { b.pos[i] = spos[i]; b.mass[i] = sq[i]; }
    for (std::size_t j = 0; j < nt; ++j) { b.pos[ns + j] = tpos[j]; b.mass[ns + j] = 0.0; }
    DirectSolver ds(KernelParams{1.0, 0.0});
    const Accelerations acc = ds.computeAccelerations(b);
    return std::vector<Complex>(acc.begin() + ns, acc.end());
}

// ---- error metric -----------------------------------------------------------
struct Err { double pot; double acc; };

Err relErr(const std::vector<double>& potTrue, const std::vector<double>& potApprox,
           const std::vector<Complex>& accTrue, const std::vector<Complex>& accApprox) {
    const std::size_t n = potTrue.size();
    double potNum = 0.0, potDen = 0.0, accNum = 0.0, accDen = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        potNum = std::max(potNum, std::abs(potApprox[i] - potTrue[i]));
        potDen += potTrue[i] * potTrue[i];
        accNum = std::max(accNum, std::abs(accApprox[i] - accTrue[i]));
        accDen += std::norm(accTrue[i]);
    }
    const double potRMS = std::sqrt(potDen / n);
    const double accRMS = std::sqrt(accDen / n);
    return {potNum / potRMS, accNum / accRMS};
}

// ---- random geometry helpers ------------------------------------------------
void makeCluster(Complex center, double radius, int n, unsigned seed,
                 std::vector<Complex>& pos, std::vector<double>& q) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    pos.clear(); q.clear();
    for (int i = 0; i < n; ++i) {
        const double r = radius * std::sqrt(u(rng));
        const double th = TAU * u(rng);
        pos.push_back(center + Complex{r * std::cos(th), r * std::sin(th)});
        q.push_back(0.5 + u(rng)); // positive "masses" in [0.5, 1.5]
    }
}

// A tight cloud of points filling a disk of the given radius about a center
// (used for local-expansion targets, which sit in a small box near z_L).
std::vector<Complex> makePoints(Complex center, double radius, int n, unsigned seed) {
    std::vector<Complex> pos; std::vector<double> q;
    makeCluster(center, radius, n, seed, pos, q);
    return pos;
}

// A shell of points with radius in [rMin, rMax] about a center. Used for
// MULTIPOLE-evaluation targets, which must all lie OUTSIDE the source cluster
// (the multipole expansion legitimately diverges inside |z - z_c| < radius).
std::vector<Complex> makeShell(Complex center, double rMin, double rMax,
                               int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Complex> pos;
    for (int i = 0; i < n; ++i) {
        const double r = rMin + (rMax - rMin) * u(rng);
        const double th = TAU * u(rng);
        pos.push_back(center + Complex{r * std::cos(th), r * std::sin(th)});
    }
    return pos;
}

// Analytic local ("P2L") expansion about z_L from far sources -- a TEST helper
// used only to feed L2L a known-good input, isolating L2L from M2L.
//   ln(z - z_s) = ln(z_L - z_s) + sum_{l>=1} (-1)^{l+1}/l ((z-z_L)/(z_L-z_s))^l
Local localFromSources(Complex zL, const std::vector<Complex>& spos,
                       const std::vector<double>& sq, int p) {
    Local L(zL, p);
    for (std::size_t j = 0; j < spos.size(); ++j) {
        const Complex D = zL - spos[j];
        L.c[0] += sq[j] * std::log(D);
        Complex Dl = D; // D^l
        for (int l = 1; l <= p; ++l) {
            const double sign = (l % 2 == 0) ? -1.0 : 1.0; // (-1)^{l+1}
            L.c[l] += sq[j] * sign / (static_cast<double>(l) * Dl);
            Dl *= D;
        }
    }
    return L;
}

// ---- gate + reporting -------------------------------------------------------
bool gate(const char* op, const std::vector<int>& ps, const std::vector<Err>& e) {
    std::printf("\n=== %s : relative error vs p ===\n", op);
    std::printf("%4s  %16s  %16s\n", "p", "potential", "acceleration");
    for (std::size_t i = 0; i < ps.size(); ++i) {
        std::printf("%4d  %16.6e  %16.6e\n", ps[i], e[i].pot, e[i].acc);
        g_csv << op << ",potential," << ps[i] << ',' << e[i].pot << '\n';
        g_csv << op << ",acceleration," << ps[i] << ',' << e[i].acc << '\n';
    }
    bool monoPot = true, monoAcc = true;
    for (std::size_t i = 1; i < ps.size(); ++i) {
        if (!(e[i].pot < e[i - 1].pot)) monoPot = false;
        if (!(e[i].acc < e[i - 1].acc)) monoAcc = false;
    }
    const bool finalOk = e.back().pot < FINAL_TOL && e.back().acc < FINAL_TOL;
    const bool pass = monoPot && monoAcc && finalOk;
    std::printf("  monotonic(pot)=%s  monotonic(acc)=%s  final<%.0e=%s  =>  %s\n",
                monoPot ? "yes" : "NO", monoAcc ? "yes" : "NO", FINAL_TOL,
                finalOk ? "yes" : "NO", pass ? "GATE PASS" : "GATE FAIL");
    return pass;
}

// ---- the four operator tests ------------------------------------------------

// 1) P2M: source cluster near z_c; evaluate multipole at well-separated targets.
bool testP2M() {
    const Complex zc{0.0, 0.0};
    std::vector<Complex> spos; std::vector<double> sq;
    makeCluster(zc, 1.0, 60, 11, spos, sq);
    // Well-separated targets: a shell at radius [5,9], all outside the cluster.
    const std::vector<Complex> tpos = makeShell(zc, 5.0, 9.0, 48, 99);

    const std::vector<double> potTrue = [&] {
        std::vector<double> v; for (auto t : tpos) v.push_back(directPotential(t, spos, sq)); return v;
    }();
    const std::vector<Complex> accTrue = directAccel(spos, sq, tpos);

    std::vector<Err> errs;
    for (int p : PS) {
        Multipole M = fmm::P2M(zc, spos, sq, p);
        std::vector<double> potA; std::vector<Complex> accA;
        for (auto t : tpos) {
            potA.push_back(fmm::multipolePotential(M, t).real());
            accA.push_back(fmm::fieldToAccel(fmm::multipoleField(M, t)));
        }
        errs.push_back(relErr(potTrue, potA, accTrue, accA));
    }
    return gate("P2M", PS, errs);
}

// 2) M2L: build a multipole, M2L to a well-separated local center, evaluate the
//    local expansion at targets NEAR z_L. Tested over multiple geometries; the
//    reported error is the worst case across geometries (hardest test).
bool testM2L() {
    struct Geo { Complex zc, zL; const char* name; };
    const std::vector<Geo> geos = {
        {{0, 0}, {6, 0},     "real axis"},
        {{0, 0}, {0, 6},     "imag axis"},
        {{0, 0}, {5, 5},     "diagonal"},
        {{0, 0}, {-6, -2},   "third quadrant"},
    };

    std::vector<Err> worst(PS.size(), Err{0.0, 0.0});
    std::printf("\n--- M2L per-geometry error at p=12 (worst case gated) ---\n");

    for (const Geo& g : geos) {
        std::vector<Complex> spos; std::vector<double> sq;
        makeCluster(g.zc, 0.9, 60, 21, spos, sq);
        const std::vector<Complex> tpos = makePoints(g.zL, 0.9, 48, 77);

        std::vector<double> potTrue;
        for (auto t : tpos) potTrue.push_back(directPotential(t, spos, sq));
        const std::vector<Complex> accTrue = directAccel(spos, sq, tpos);

        Err last{};
        for (std::size_t i = 0; i < PS.size(); ++i) {
            const int p = PS[i];
            Multipole M = fmm::P2M(g.zc, spos, sq, p);
            Local L = fmm::M2L(M, g.zL);
            std::vector<double> potA; std::vector<Complex> accA;
            for (auto t : tpos) {
                potA.push_back(fmm::localPotential(L, t).real());
                accA.push_back(fmm::fieldToAccel(fmm::localField(L, t)));
            }
            const Err e = relErr(potTrue, potA, accTrue, accA);
            worst[i].pot = std::max(worst[i].pot, e.pot);
            worst[i].acc = std::max(worst[i].acc, e.acc);
            last = e;
        }
        std::printf("  %-16s zL=(%.1f,%.1f)  pot=%.3e  acc=%.3e\n",
                    g.name, g.zL.real(), g.zL.imag(), last.pot, last.acc);
    }
    return gate("M2L", PS, worst);
}

// 3) M2M: multipole about a child center, M2M to a parent center, evaluate from
//    the parent at well-separated targets.
bool testM2M() {
    const Complex zChild{0.8, 0.6}; // offset from parent
    const Complex zParent{0.0, 0.0};
    std::vector<Complex> spos; std::vector<double> sq;
    makeCluster(zChild, 0.5, 60, 31, spos, sq);
    // Targets in a far shell; the parent cluster radius is ~|zChild|+0.5 ~ 1.5.
    const std::vector<Complex> tpos = makeShell(zParent, 6.0, 10.0, 48, 55);

    std::vector<double> potTrue;
    for (auto t : tpos) potTrue.push_back(directPotential(t, spos, sq));
    const std::vector<Complex> accTrue = directAccel(spos, sq, tpos);

    std::vector<Err> errs;
    for (int p : PS) {
        Multipole Mc = fmm::P2M(zChild, spos, sq, p);
        Multipole Mp = fmm::M2M(Mc, zParent);
        std::vector<double> potA; std::vector<Complex> accA;
        for (auto t : tpos) {
            potA.push_back(fmm::multipolePotential(Mp, t).real());
            accA.push_back(fmm::fieldToAccel(fmm::multipoleField(Mp, t)));
        }
        errs.push_back(relErr(potTrue, potA, accTrue, accA));
    }
    return gate("M2M", PS, errs);
}

// 4) L2L: local expansion about a parent center from a FAR source, L2L to a
//    child center, evaluate near the child.
bool testL2L() {
    const Complex zParent{0.0, 0.0};
    const Complex zChild{0.6, 0.4};
    const Complex zFar{-9.0, 1.0};
    std::vector<Complex> spos; std::vector<double> sq;
    makeCluster(zFar, 1.0, 60, 41, spos, sq);
    const std::vector<Complex> tpos = makePoints(zChild, 0.5, 48, 66);

    std::vector<double> potTrue;
    for (auto t : tpos) potTrue.push_back(directPotential(t, spos, sq));
    const std::vector<Complex> accTrue = directAccel(spos, sq, tpos);

    std::vector<Err> errs;
    for (int p : PS) {
        Local Lp = localFromSources(zParent, spos, sq, p); // known-good parent local
        Local Lc = fmm::L2L(Lp, zChild);
        std::vector<double> potA; std::vector<Complex> accA;
        for (auto t : tpos) {
            potA.push_back(fmm::localPotential(Lc, t).real());
            accA.push_back(fmm::fieldToAccel(fmm::localField(Lc, t)));
        }
        errs.push_back(relErr(potTrue, potA, accTrue, accA));
    }
    return gate("L2L", PS, errs);
}

} // namespace

int main() {
    g_csv.open("outputs/fmm_operator_convergence.csv");
    g_csv << "operator,quantity,p,rel_err\n";

    std::printf("=== FMM operator unit tests (Rung 2a) ===\n");
    std::printf("Ground truth: potential via direct summation, acceleration via "
                "DirectSolver (eps=0).\n");

    bool ok = true;
    ok &= testP2M();
    ok &= testM2L();
    ok &= testM2M();
    ok &= testL2L();

    g_csv.close();
    std::printf("\nwrote outputs/fmm_operator_convergence.csv\n");
    std::printf("\n==================  %s  ==================\n",
                ok ? "ALL OPERATOR GATES PASSED" : "SOME OPERATOR GATE FAILED");
    return ok ? 0 : 1;
}
