// ============================================================================
// RUNG 4: isolated unit tests for the 3D Laplace FMM operators.
//
// Gate 0 : basis check (harmonics vs closed forms; gradient recurrences vs
//          finite differences) -- catch harmonic/gradient bugs before operators.
// Gates 1-4 : P2M, M2L, M2M, L2L, each validated IN ISOLATION against the
//          oracle for BOTH potential (vs direct sum sum q/|r-r'|) and
//          acceleration (vs DirectSolver3D, eps=0), error-vs-p, multiple
//          geometries. No octree, lists, passes, P2P, or FMM3D wiring.
// ============================================================================
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "DirectSolver3D.hpp" // the 3D oracle (acceleration ground truth)
#include "FmmExpansions3D.hpp"
#include "Kernel3D.hpp"
#include "Types3D.hpp"
#include "Vec3.hpp"

using namespace galaxy3d;
using fmm::Complex;
using fmm::Local;
using fmm::Multipole;

namespace {

const double TAU = 2.0 * std::acos(-1.0);
const std::vector<int> PS = {2, 4, 6, 8, 10};
std::ofstream g_csv;

// ---- ground truth ----------------------------------------------------------
double directPotential(Vec3 t, const std::vector<Vec3>& s, const std::vector<double>& q) {
    double phi = 0.0;
    for (std::size_t j = 0; j < s.size(); ++j) phi += q[j] / abs(t - s[j]);
    return phi;
}

std::vector<Vec3> directAccel(const std::vector<Vec3>& spos, const std::vector<double>& sq,
                              const std::vector<Vec3>& tpos) {
    const std::size_t ns = spos.size(), nt = tpos.size();
    Bodies3D b;
    b.resize(ns + nt);
    for (std::size_t i = 0; i < ns; ++i) { b.pos[i] = spos[i]; b.mass[i] = sq[i]; }
    for (std::size_t j = 0; j < nt; ++j) { b.pos[ns + j] = tpos[j]; b.mass[ns + j] = 0.0; }
    const Accelerations3D a = DirectSolver3D(KernelParams3D{1.0, 0.0}).computeAccelerations(b);
    return std::vector<Vec3>(a.begin() + ns, a.end());
}

// Analytic local expansion ("P2L") about z_L from far sources -- TEST helper to
// feed L2L a known-good input, isolating L2L.  L_n^m = sum_j q_j Y_n^{-m}/rho^{n+1}.
Local localFromSources(Vec3 zL, const std::vector<Vec3>& s,
                       const std::vector<double>& q, int p) {
    Local out(zL, p);
    std::vector<Complex> Y;
    for (std::size_t j = 0; j < s.size(); ++j) {
        const double rho = fmm::computeY(s[j] - zL, p, Y);
        double inv = 1.0 / rho; // rho^{-(n+1)}
        for (int n = 0; n <= p; ++n) {
            for (int m = -n; m <= n; ++m) out.L[fmm::idx(n, m)] += q[j] * Y[fmm::idx(n, -m)] * inv;
            inv /= rho;
        }
    }
    return out;
}

// ---- error metric -----------------------------------------------------------
struct Err { double potRms, accRms; };

Err relErr(const std::vector<double>& pT, const std::vector<double>& pA,
           const std::vector<Vec3>& aT, const std::vector<Vec3>& aA) {
    const std::size_t n = pT.size();
    double pNum = 0, pDen = 0, aNum = 0, aDen = 0;
    for (std::size_t i = 0; i < n; ++i) {
        pNum += (pA[i] - pT[i]) * (pA[i] - pT[i]); pDen += pT[i] * pT[i];
        aNum += norm2(aA[i] - aT[i]);              aDen += norm2(aT[i]);
    }
    return {std::sqrt(pNum / pDen), std::sqrt(aNum / aDen)};
}

// ---- random geometry --------------------------------------------------------
Vec3 randDir(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0, 1);
    const double cz = 2 * u(rng) - 1, sz = std::sqrt(std::max(0.0, 1 - cz * cz)), ph = TAU * u(rng);
    return Vec3{sz * std::cos(ph), sz * std::sin(ph), cz};
}
void ball(Vec3 c, double rad, int n, unsigned seed, std::vector<Vec3>& pos, std::vector<double>& q) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    pos.clear(); q.clear();
    for (int i = 0; i < n; ++i) {
        pos.push_back(c + (rad * std::cbrt(u(rng))) * randDir(rng));
        q.push_back(0.5 + u(rng));
    }
}
std::vector<Vec3> ballPts(Vec3 c, double rad, int n, unsigned seed) {
    std::vector<Vec3> p; std::vector<double> q; ball(c, rad, n, seed, p, q); return p;
}
std::vector<Vec3> shell(Vec3 c, double rmin, double rmax, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<Vec3> p;
    for (int i = 0; i < n; ++i) p.push_back(c + (rmin + (rmax - rmin) * u(rng)) * randDir(rng));
    return p;
}

bool gate(const char* op, const std::vector<Err>& e) {
    std::printf("\n=== %s : relative error vs p ===\n", op);
    std::printf("%4s  %16s  %16s\n", "p", "potential(rms)", "accel(rms)");
    bool mp = true, ma = true;
    for (std::size_t i = 0; i < PS.size(); ++i) {
        std::printf("%4d  %16.6e  %16.6e\n", PS[i], e[i].potRms, e[i].accRms);
        g_csv << op << ",potential," << PS[i] << ',' << e[i].potRms << '\n';
        g_csv << op << ",acceleration," << PS[i] << ',' << e[i].accRms << '\n';
        if (i && !(e[i].potRms < e[i - 1].potRms)) mp = false;
        if (i && !(e[i].accRms < e[i - 1].accRms)) ma = false;
    }
    const bool fin = e.back().potRms < 1e-4 && e.back().accRms < 1e-3;
    const bool pass = mp && ma && fin;
    std::printf("  monotone(pot)=%s monotone(acc)=%s final-small=%s => %s\n",
                mp ? "yes" : "NO", ma ? "yes" : "NO", fin ? "yes" : "NO",
                pass ? "GATE PASS" : "GATE FAIL");
    return pass;
}

// ---- Gate 0: basis check ----------------------------------------------------
bool gateBasis() {
    std::printf("=== Gate 0: basis check (harmonics + gradient recurrences) ===\n");
    bool ok = true;

    // (a) Y_n^m vs closed forms at theta=1.0, phi=0.7.
    const double th = 1.0, ph = 0.7, r = 1.0;
    const Vec3 v{r * std::sin(th) * std::cos(ph), r * std::sin(th) * std::sin(ph), r * std::cos(th)};
    std::vector<Complex> Y;
    fmm::computeY(v, 2, Y);
    const double c = std::cos(th), s = std::sin(th);
    auto e = [&](double m) { return Complex{std::cos(m * ph), std::sin(m * ph)}; };
    struct Ref { int n, m; Complex val; };
    const std::vector<Ref> refs = {
        {0, 0, {1, 0}},
        {1, 0, {c, 0}},
        {1, 1, std::sqrt(0.5) * s * e(1)},
        {2, 0, {(3 * c * c - 1) / 2, 0}},
        {2, 1, std::sqrt(1.0 / 6.0) * (3 * c * s) * e(1)},
        {2, 2, std::sqrt(1.0 / 24.0) * (3 * s * s) * e(2)},
    };
    double hmax = 0;
    for (const auto& rf : refs) {
        hmax = std::max(hmax, std::abs(Y[fmm::idx(rf.n, rf.m)] - rf.val));
        hmax = std::max(hmax, std::abs(Y[fmm::idx(rf.n, -rf.m)] - std::conj(rf.val)));
    }
    std::printf("  harmonics max abs error vs closed forms: %.3e  %s\n",
                hmax, hmax < 1e-13 ? "OK" : "FAIL");
    if (hmax >= 1e-13) ok = false;

    // (b) gradient recurrences vs central finite differences, for a random
    //     multipole and a random local expansion.
    std::mt19937 rng(123);
    std::normal_distribution<double> g(0, 1);
    const int p = 6;
    Multipole mp({0, 0, 0}, p);
    Local lp({0, 0, 0}, p);
    for (auto& z : mp.M) z = Complex{g(rng), g(rng)};
    for (auto& z : lp.L) z = Complex{g(rng), g(rng)};

    const double h = 1e-5;
    auto fdGrad = [&](auto pot, Vec3 t) {
        return Vec3{(pot(t + Vec3{h, 0, 0}).real() - pot(t - Vec3{h, 0, 0}).real()) / (2 * h),
                    (pot(t + Vec3{0, h, 0}).real() - pot(t - Vec3{0, h, 0}).real()) / (2 * h),
                    (pot(t + Vec3{0, 0, h}).real() - pot(t - Vec3{0, 0, h}).real()) / (2 * h)};
    };
    double gMax = 0;
    for (int i = 0; i < 20; ++i) {
        const Vec3 tFar = 5.0 * randDir(rng);   // multipole valid far
        const Vec3 tNear = 0.4 * randDir(rng);  // local valid near
        const Vec3 gMexact = fmm::multipoleField(mp, tFar);
        const Vec3 gMfd = fdGrad([&](Vec3 t) { return fmm::multipolePotential(mp, t); }, tFar);
        const Vec3 gLexact = fmm::localField(lp, tNear);
        const Vec3 gLfd = fdGrad([&](Vec3 t) { return fmm::localPotential(lp, t); }, tNear);
        gMax = std::max(gMax, abs(gMexact - gMfd) / abs(gMexact));
        gMax = std::max(gMax, abs(gLexact - gLfd) / abs(gLexact));
    }
    std::printf("  gradient recurrences vs finite-diff, max rel error: %.3e  %s\n",
                gMax, gMax < 1e-6 ? "OK" : "FAIL");
    if (gMax >= 1e-6) ok = false;

    std::printf("  => %s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok;
}

// Evaluate an expansion (multipole or local) at targets -> (pot, accel).
template <class Pot, class Field>
void evalAll(Pot pot, Field field, const std::vector<Vec3>& tpos,
             std::vector<double>& potA, std::vector<Vec3>& accA) {
    potA.clear(); accA.clear();
    for (auto t : tpos) { potA.push_back(pot(t).real()); accA.push_back(field(t)); }
}

bool testP2M() {
    const Vec3 zc{0, 0, 0};
    std::vector<Vec3> sp; std::vector<double> sq;
    ball(zc, 1.0, 80, 11, sp, sq);
    const std::vector<Vec3> tp = shell(zc, 4.0, 8.0, 64, 99);
    std::vector<double> pT; for (auto t : tp) pT.push_back(directPotential(t, sp, sq));
    const std::vector<Vec3> aT = directAccel(sp, sq, tp);
    std::vector<Err> errs;
    for (int p : PS) {
        Multipole M = fmm::P2M(zc, sp, sq, p);
        std::vector<double> pA; std::vector<Vec3> aA;
        evalAll([&](Vec3 t) { return fmm::multipolePotential(M, t); },
                [&](Vec3 t) { return fmm::multipoleField(M, t); }, tp, pA, aA);
        errs.push_back(relErr(pT, pA, aT, aA));
    }
    return gate("P2M", errs);
}

bool testM2L() {
    struct Geo { Vec3 zc, zL; const char* name; };
    const std::vector<Geo> geos = {
        {{0, 0, 0}, {6, 0, 0}, "x"}, {{0, 0, 0}, {0, 6, 0}, "y"},
        {{0, 0, 0}, {0, 0, 6}, "z"}, {{0, 0, 0}, {3.5, 3.5, 3.5}, "diagonal"},
    };
    std::vector<Err> worst(PS.size(), Err{0, 0});
    std::printf("\n--- M2L per-geometry error at p=10 (worst case gated) ---\n");
    for (const auto& gm : geos) {
        std::vector<Vec3> sp; std::vector<double> sq;
        ball(gm.zc, 0.9, 80, 21, sp, sq);
        const std::vector<Vec3> tp = ballPts(gm.zL, 0.9, 64, 77);
        std::vector<double> pT; for (auto t : tp) pT.push_back(directPotential(t, sp, sq));
        const std::vector<Vec3> aT = directAccel(sp, sq, tp);
        Err last{};
        for (std::size_t i = 0; i < PS.size(); ++i) {
            Multipole M = fmm::P2M(gm.zc, sp, sq, PS[i]);
            Local L = fmm::M2L(M, gm.zL);
            std::vector<double> pA; std::vector<Vec3> aA;
            evalAll([&](Vec3 t) { return fmm::localPotential(L, t); },
                    [&](Vec3 t) { return fmm::localField(L, t); }, tp, pA, aA);
            const Err e = relErr(pT, pA, aT, aA);
            worst[i].potRms = std::max(worst[i].potRms, e.potRms);
            worst[i].accRms = std::max(worst[i].accRms, e.accRms);
            last = e;
        }
        std::printf("  %-9s zL=(%.1f,%.1f,%.1f)  pot=%.3e acc=%.3e\n",
                    gm.name, gm.zL.x, gm.zL.y, gm.zL.z, last.potRms, last.accRms);
    }
    return gate("M2L", worst);
}

bool testM2M() {
    const Vec3 zChild{0.5, 0.4, 0.3}, zParent{0, 0, 0};
    std::vector<Vec3> sp; std::vector<double> sq;
    ball(zChild, 0.4, 80, 31, sp, sq);
    const std::vector<Vec3> tp = shell(zParent, 5.0, 9.0, 64, 55);
    std::vector<double> pT; for (auto t : tp) pT.push_back(directPotential(t, sp, sq));
    const std::vector<Vec3> aT = directAccel(sp, sq, tp);
    std::vector<Err> errs;
    for (int p : PS) {
        Multipole Mc = fmm::P2M(zChild, sp, sq, p);
        Multipole Mp = fmm::M2M(Mc, zParent);
        std::vector<double> pA; std::vector<Vec3> aA;
        evalAll([&](Vec3 t) { return fmm::multipolePotential(Mp, t); },
                [&](Vec3 t) { return fmm::multipoleField(Mp, t); }, tp, pA, aA);
        errs.push_back(relErr(pT, pA, aT, aA));
    }
    return gate("M2M", errs);
}

bool testL2L() {
    const Vec3 zParent{0, 0, 0}, zChild{0.5, 0.4, 0.3};
    std::vector<Vec3> sp; std::vector<double> sq;
    ball(Vec3{-8, 3, 2}, 1.0, 80, 41, sp, sq);
    const std::vector<Vec3> tp = ballPts(zChild, 0.4, 64, 66);
    std::vector<double> pT; for (auto t : tp) pT.push_back(directPotential(t, sp, sq));
    const std::vector<Vec3> aT = directAccel(sp, sq, tp);
    std::vector<Err> errs;
    for (int p : PS) {
        Local Lp = localFromSources(zParent, sp, sq, p);
        Local Lc = fmm::L2L(Lp, zChild);
        std::vector<double> pA; std::vector<Vec3> aA;
        evalAll([&](Vec3 t) { return fmm::localPotential(Lc, t); },
                [&](Vec3 t) { return fmm::localField(Lc, t); }, tp, pA, aA);
        errs.push_back(relErr(pT, pA, aT, aA));
    }
    return gate("L2L", errs);
}

} // namespace

int main() {
    g_csv.open("outputs/fmm3d_operator_convergence.csv");
    g_csv << "operator,quantity,p,rel_err\n";

    std::printf("=== 3D FMM operator unit tests (Rung 4) ===\n");
    std::printf("Convention: Greengard-Rokhlin solid harmonics; naive O(p^4) operators.\n");
    std::printf("Ground truth: potential via direct sum, acceleration via DirectSolver3D (eps=0).\n\n");

    bool ok = true;
    ok &= gateBasis();
    ok &= testP2M();
    ok &= testM2L();
    ok &= testM2M();
    ok &= testL2L();

    g_csv.close();
    std::printf("\nwrote outputs/fmm3d_operator_convergence.csv\n");
    std::printf("\n==========  %s  ==========\n",
                ok ? "ALL 3D OPERATOR GATES PASSED" : "SOME 3D OPERATOR GATE FAILED");
    return ok ? 0 : 1;
}
