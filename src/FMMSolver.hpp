#pragma once

#include <cmath>
#include <vector>

#include "FmmExpansions.hpp" // validated operators (Rung 2a) -- reused UNCHANGED
#include "ForceSolver.hpp"
#include "Kernel.hpp"

namespace galaxy {

// ============================================================================
// RUNG 2b: complete UNIFORM 2D Fast Multipole Method, O(N).
//
// Another ForceSolver behind the existing interface (kernel/integrator/Types
// untouched). It assembles the Rung 2a operators (P2M/M2M/M2L/L2L/eval, in
// FmmExpansions.hpp, reused unchanged) on a UNIFORM (non-adaptive) quadtree of
// depth L. Adaptive trees are a Rung 3 stretch.
//
// Depth L is chosen so 4^L ~ N (mean leaf occupancy ~ O(1)) -> O(N) total.
//
// Pipeline (rebuilt every call):
//   1. Uniform quadtree over the bounding square, depth L.
//   2. Upward:   P2M at every leaf, M2M up to a multipole at every box.
//   3. Downward: for each box, M2L from every box in its INTERACTION LIST
//      (children of the parent's neighbours that are not the box's own
//      neighbours -- the standard uniform-FMM list, <= 27 in 2D) into its local
//      expansion; then L2L the accumulated local down to children.
//   4. Leaves:   L2P (evaluate the local expansion's DERIVATIVE = far-field
//      acceleration) per particle, PLUS P2P direct summation over the particle's
//      own leaf and its nearest-neighbour leaves.
//
// Softening eps is applied ONLY in P2P (matching DirectSolver). Far-field
// expansions use the pure kernel -- well-separated pairs are never within eps.
//
// Reduction gates that this construction satisfies (see fmm_tests/harness):
//   * depth 0 -> the whole domain is one leaf -> pure P2P -> equals DirectSolver
//     to roundoff (validates P2P + plumbing, independent of the expansions).
//   * depth 1 -> all four boxes are mutual neighbours -> still pure P2P -> direct.
//   FMM only differs from direct at depth >= 2, where interaction lists appear.
// ============================================================================
class FMMSolver final : public ForceSolver {
public:
    // depthOverride < 0 selects the automatic depth (4^L ~ N).
    FMMSolver(KernelParams params, int p, int depthOverride = -1)
        : params_(params), p_(p), depthOverride_(depthOverride) {}

    // L ~ log4(N) so 4^L ~ N and mean leaf occupancy stays O(1) -> O(N) total.
    // We aim for occupancy ~4 (one level shallower than 4^L=N): still O(1)
    // occupancy, but it cuts the number of expensive M2L translations ~4x while
    // leaving the cheap P2P near-field small -- the constant-factor sweet spot.
    static int autoDepth(std::size_t n) {
        if (n <= 4) return 0;
        int L = static_cast<int>(std::lround(std::log(static_cast<double>(n)) / std::log(4.0))) - 1;
        return L < 0 ? 0 : L;
    }

    Accelerations computeAccelerations(const Bodies& bodies) const override {
        using fmm::Local;
        using fmm::Multipole;

        const std::size_t n = bodies.size();
        Accelerations acc(n, Complex{0.0, 0.0});
        if (n == 0) return acc;

        const int L = depthOverride_ >= 0 ? depthOverride_ : autoDepth(n);

        // ---- bounding square (recomputed each call: the system breathes) ----
        double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = bodies.pos[i].real(), y = bodies.pos[i].imag();
            minx = std::min(minx, x); maxx = std::max(maxx, x);
            miny = std::min(miny, y); maxy = std::max(maxy, y);
        }
        const double cx = 0.5 * (minx + maxx), cy = 0.5 * (miny + maxy);
        double side = std::max(maxx - minx, maxy - miny);
        if (!(side > 0.0)) side = 1.0;
        side *= 1.0 + 1e-9;
        const double ox = cx - 0.5 * side, oy = cy - 0.5 * side; // lower-left corner

        auto cellAt = [&](int d) { return side / static_cast<double>(1 << d); };
        auto centerAt = [&](int d, int ix, int iy) {
            const double c = cellAt(d);
            return Complex{ox + (ix + 0.5) * c, oy + (iy + 0.5) * c};
        };
        auto clampi = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };

        // ---- assign bodies to leaves at level L ----------------------------
        const int nsideL = 1 << L;
        std::vector<std::vector<int>> leafBodies(static_cast<std::size_t>(nsideL) * nsideL);
        for (std::size_t i = 0; i < n; ++i) {
            const double nx = (bodies.pos[i].real() - ox) / side;
            const double ny = (bodies.pos[i].imag() - oy) / side;
            const int ix = clampi(static_cast<int>(std::floor(nx * nsideL)), nsideL - 1);
            const int iy = clampi(static_cast<int>(std::floor(ny * nsideL)), nsideL - 1);
            leafBodies[static_cast<std::size_t>(iy) * nsideL + ix].push_back(static_cast<int>(i));
        }

        // ---- per-level storage ---------------------------------------------
        std::vector<std::vector<Multipole>> M(L + 1);
        std::vector<std::vector<Local>> Lex(L + 1);
        std::vector<std::vector<char>> nonempty(L + 1);
        for (int d = 0; d <= L; ++d) {
            const int ns = 1 << d;
            const int nb = ns * ns;
            M[d].assign(nb, Multipole(Complex{0, 0}, p_));
            Lex[d].assign(nb, Local(Complex{0, 0}, p_));
            nonempty[d].assign(nb, 0);
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int b = iy * ns + ix;
                    const Complex ctr = centerAt(d, ix, iy);
                    M[d][b].center = ctr;
                    Lex[d][b].center = ctr;
                }
        }

        // ---- upward pass: P2M at leaves ------------------------------------
        for (int iy = 0; iy < nsideL; ++iy)
            for (int ix = 0; ix < nsideL; ++ix) {
                const int b = iy * nsideL + ix;
                const auto& bl = leafBodies[b];
                if (bl.empty()) continue;
                nonempty[L][b] = 1;
                std::vector<Complex> pos;
                std::vector<double> q;
                pos.reserve(bl.size());
                q.reserve(bl.size());
                for (int j : bl) { pos.push_back(bodies.pos[j]); q.push_back(bodies.mass[j]); }
                M[L][b] = fmm::P2M(centerAt(L, ix, iy), pos, q, p_);
            }

        // ---- upward pass: M2M up the tree ----------------------------------
        for (int d = L - 1; d >= 0; --d) {
            const int ns = 1 << d;
            const int nsc = 1 << (d + 1);
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int b = iy * ns + ix;
                    const Complex ctr = M[d][b].center;
                    bool any = false;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const int cxi = 2 * ix + dx, cyi = 2 * iy + dy;
                            const int cb = cyi * nsc + cxi;
                            if (!nonempty[d + 1][cb]) continue;
                            any = true;
                            const Multipole sh = fmm::M2M(M[d + 1][cb], ctr);
                            M[d][b].Q += sh.Q;
                            for (int k = 1; k <= p_; ++k) M[d][b].a[k] += sh.a[k];
                        }
                    nonempty[d][b] = any ? 1 : 0;
                }
        }

        // ---- downward pass: M2L from interaction list, then L2L to children -
        for (int d = 2; d <= L; ++d) {
            const int ns = 1 << d;
            const int nsp = ns / 2;
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int b = iy * ns + ix;
                    const Complex ctr = Lex[d][b].center;

                    // Bring down the parent's fully-formed local expansion.
                    // (Levels 0,1 never receive M2L, so their locals are zero;
                    //  L2L only matters once the parent is at level >= 2.)
                    if (d >= 3) {
                        const int pb = (iy / 2) * nsp + (ix / 2);
                        const Local up = fmm::L2L(Lex[d - 1][pb], ctr);
                        for (int k = 0; k <= p_; ++k) Lex[d][b].c[k] += up.c[k];
                    }

                    // Interaction list: children of the parent's neighbours that
                    // are NOT this box's own nearest neighbours (those go to P2P).
                    const int px = ix / 2, py = iy / 2;
                    for (int qy = py - 1; qy <= py + 1; ++qy) {
                        if (qy < 0 || qy >= nsp) continue;
                        for (int qx = px - 1; qx <= px + 1; ++qx) {
                            if (qx < 0 || qx >= nsp) continue;
                            for (int dy = 0; dy < 2; ++dy)
                                for (int dx = 0; dx < 2; ++dx) {
                                    const int ax = 2 * qx + dx, ay = 2 * qy + dy;
                                    if (std::max(std::abs(ax - ix), std::abs(ay - iy)) <= 1)
                                        continue; // nearest neighbour -> handled by P2P
                                    const int ab = ay * ns + ax;
                                    if (!nonempty[d][ab]) continue;
                                    const Local m = fmm::M2L(M[d][ab], ctr);
                                    for (int k = 0; k <= p_; ++k) Lex[d][b].c[k] += m.c[k];
                                }
                        }
                    }
                }
        }

        // ---- leaf evaluation: far field (L2P) + near field (P2P) -----------
        const double eps2 = params_.eps * params_.eps;
        const double G = params_.G;
        for (int iy = 0; iy < nsideL; ++iy)
            for (int ix = 0; ix < nsideL; ++ix) {
                const int b = iy * nsideL + ix;
                const auto& bl = leafBodies[b];
                if (bl.empty()) continue;
                const Local& Lb = Lex[L][b];

                for (int ti : bl) {
                    const Complex zi = bodies.pos[ti];
                    // Far field: derivative of the local expansion -> acceleration.
                    Complex a = fmm::fieldToAccel(fmm::localField(Lb, zi), G);
                    // Near field: direct P2P over this leaf + 8 neighbour leaves,
                    // with softening eps (exactly DirectSolver for these pairs).
                    for (int ny = iy - 1; ny <= iy + 1; ++ny) {
                        if (ny < 0 || ny >= nsideL) continue;
                        for (int nx = ix - 1; nx <= ix + 1; ++nx) {
                            if (nx < 0 || nx >= nsideL) continue;
                            for (int tj : leafBodies[static_cast<std::size_t>(ny) * nsideL + nx]) {
                                if (tj == ti) continue; // skip self
                                const Complex diff = bodies.pos[tj] - zi;
                                const double denom = std::norm(diff) + eps2;
                                a += (G * bodies.mass[tj] / denom) * diff;
                            }
                        }
                    }
                    acc[ti] = a;
                }
            }
        return acc;
    }

    const char* name() const override { return "FMMSolver(O(N) uniform)"; }
    int order() const { return p_; }

private:
    KernelParams params_;
    int p_;
    int depthOverride_;
};

} // namespace galaxy
