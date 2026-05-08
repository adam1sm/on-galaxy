#pragma once

#include <vector>

#include "AdaptiveOctree.hpp"
#include "FmmExpansions3D.hpp" // validated operators, UNCHANGED
#include "ForceSolver3D.hpp"
#include "Kernel3D.hpp"

namespace galaxy3d {

// ============================================================================
// RUNG 10a: adaptive FMM (Carrier-Greengard-Rokhlin) on CPU. Another
// ForceSolver3D. Reuses the Rung 4 operators unchanged:
//   P2M/M2M (upward), M2L (V-list), L2L + L2P=localField (downward/leaf),
//   M2P=multipoleField (W-list), and P2L (X-list, built here from computeY),
//   P2P (U-list, softened). Two added operators, M2P and P2L, reuse the
//   harmonic machinery only. Softening eps applied ONLY in P2P.
// ============================================================================
class AdaptiveFMM3D final : public ForceSolver3D {
public:
    AdaptiveFMM3D(KernelParams3D params, int p, int threshold = 64)
        : params_(params), p_(p), threshold_(threshold) {}

    // exposed for the harness (Gate C occupancy report)
    mutable int lastMaxLeafOcc = 0;
    mutable int lastNumBoxes = 0;

    Accelerations3D computeAccelerations(const Bodies3D& bodies) const override {
        using fmm::Local;
        using fmm::Multipole;
        const std::size_t n = bodies.size();
        Accelerations3D acc(n, Vec3{});
        if (n == 0) return acc;

        AdaptiveOctree t;
        t.build(bodies, threshold_, params_.eps); // eps floor (Rung 12; inert at eps=0)
        t.buildLists();
        const int nb = (int)t.box.size();
        const int S = (p_ + 1) * (p_ + 1);
        lastMaxLeafOcc = t.maxLeafOcc();
        lastNumBoxes = nb;

        std::vector<Multipole> M(nb);
        std::vector<Local> L(nb);
        for (int b = 0; b < nb; ++b) {
            const Vec3 c{t.box[b].cx, t.box[b].cy, t.box[b].cz};
            M[b] = Multipole(c, p_);
            L[b] = Local(c, p_);
        }

        // ---- upward: P2M at leaves, M2M up ---------------------------------
        for (int b : t.leaves) {
            const ABox& bb = t.box[b];
            std::vector<Vec3> pos; std::vector<double> q;
            pos.reserve(bb.pcount); q.reserve(bb.pcount);
            for (int k = bb.pfirst; k < bb.pfirst + bb.pcount; ++k) {
                pos.push_back(Vec3{t.px[k], t.py[k], t.pz[k]}); q.push_back(t.mass[k]);
            }
            M[b] = fmm::P2M(Vec3{bb.cx, bb.cy, bb.cz}, pos, q, p_);
        }
        for (int b = nb - 1; b >= 0; --b) {
            if (t.box[b].leaf) continue;
            const Vec3 cb{t.box[b].cx, t.box[b].cy, t.box[b].cz};
            for (int o = 0; o < 8; ++o) {
                const int ch = t.box[b].child[o];
                if (ch < 0) continue;
                const Multipole sh = fmm::M2M(M[ch], cb);
                for (int s = 0; s < S; ++s) M[b].M[s] += sh.M[s];
            }
        }

        // ---- downward: L2L from parent, V->M2L, X->P2L ---------------------
        for (int b = 0; b < nb; ++b) {
            const Vec3 cb{t.box[b].cx, t.box[b].cy, t.box[b].cz};
            if (t.box[b].parent >= 0) {
                const Local up = fmm::L2L(L[t.box[b].parent], cb);
                for (int s = 0; s < S; ++s) L[b].L[s] += up.L[s];
            }
            for (int c : t.V[b]) {
                const Local m = fmm::M2L(M[c], cb);
                for (int s = 0; s < S; ++s) L[b].L[s] += m.L[s];
            }
            for (int c : t.X[b]) p2lAccum(L[b], t, t.box[c], cb); // c is a leaf
        }

        // ---- leaves: L2P (local) + W->M2P + U->P2P ------------------------
        const double eps2 = params_.eps * params_.eps, G = params_.G;
        for (int b : t.leaves) {
            const ABox& bb = t.box[b];
            const Vec3 cb{bb.cx, bb.cy, bb.cz};
            for (int ti = bb.pfirst; ti < bb.pfirst + bb.pcount; ++ti) {
                const Vec3 zi{t.px[ti], t.py[ti], t.pz[ti]};
                Vec3 field = fmm::localField(L[b], zi);          // far field (L2P)
                for (int c : t.W[b]) field += fmm::multipoleField(M[c], zi); // M2P
                Vec3 near{};
                for (int uc : t.U[b]) {                          // P2P (softened)
                    const ABox& cc = t.box[uc];
                    for (int j = cc.pfirst; j < cc.pfirst + cc.pcount; ++j) {
                        if (j == ti) continue;
                        const Vec3 d{t.px[j] - zi.x, t.py[j] - zi.y, t.pz[j] - zi.z};
                        const double r2 = norm2(d) + eps2;
                        near += (t.mass[j] / (r2 * std::sqrt(r2))) * d;
                    }
                }
                acc[t.origIndex[ti]] = G * (field + near);
            }
        }
        return acc;
    }

    const char* name() const override { return "AdaptiveFMM3D(CGR)"; }

private:
    // P2L: add a local expansion (about cb) from the source particles of leaf c.
    // L_n^m += sum_j q_j Y_n^{-m}(r_j - cb) / |r_j - cb|^{n+1}.  (reuses computeY)
    static void p2lAccum(fmm::Local& L, const AdaptiveOctree& t, const ABox& c, const Vec3& cb) {
        std::vector<fmm::Complex> Y;
        for (int k = c.pfirst; k < c.pfirst + c.pcount; ++k) {
            const Vec3 d{t.px[k] - cb.x, t.py[k] - cb.y, t.pz[k] - cb.z};
            const double rho = fmm::computeY(d, L.p, Y);
            double inv = 1.0 / rho; // rho^{-(n+1)}
            for (int n = 0; n <= L.p; ++n) {
                for (int m = -n; m <= n; ++m) L.L[fmm::idx(n, m)] += t.mass[k] * Y[fmm::idx(n, -m)] * inv;
                inv /= rho;
            }
        }
    }

    KernelParams3D params_;
    int p_;
    int threshold_;
};

} // namespace galaxy3d
