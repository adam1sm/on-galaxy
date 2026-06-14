#pragma once

#include <cmath>
#include <vector>

#include "FmmExpansions3D.hpp" // validated Rung 4 operators -- reused UNCHANGED
#include "ForceSolver3D.hpp"
#include "Kernel3D.hpp"

namespace galaxy3d {

// ============================================================================
// RUNG 5: complete UNIFORM 3D Fast Multipole Method, O(N).
//
// Assembles the Rung 4 spherical-harmonic operators (FmmExpansions3D, unchanged)
// on a UNIFORM octree of depth L. Another ForceSolver3D behind the interface.
//
// Depth L: 8^L ~ N. Occupancy is tuned ABOVE 1 because each M2L is O(p^4) and
// the 3D interaction list holds up to 189 boxes -- a deep tree (occupancy ~1)
// would drown in M2L work. We target occupancy ~OCC so M2L (~189*p^4 per box,
// boxes = N/OCC) balances P2P (~27*OCC per particle). Still O(N).
//
// Pipeline (rebuilt each call):
//   Upward:   P2M at leaves, M2M up to a multipole at every box.
//   Downward: M2L from each box's INTERACTION LIST (children of the parent's
//             neighbours that are not the box's own neighbours, up to 189) into
//             its local expansion; then L2L the accumulated local to children.
//   Leaves:   far field = G * grad(local expansion) [L2P], plus P2P over the
//             leaf and its <=26 neighbour leaves. Softening eps ONLY in P2P;
//             far field uses the pure kernel (valid when eps << leaf size).
//
// Reductions: depth 0 -> one leaf -> pure P2P -> equals DirectSolver3D. FMM only
// differs from direct at depth >= 2 (where interaction lists appear).
// ============================================================================
class FMM3D final : public ForceSolver3D {
public:
    FMM3D(KernelParams3D params, int p, int depthOverride = -1, int occTarget = 64)
        : params_(params), p_(p), depthOverride_(depthOverride), occTarget_(occTarget) {}

    static int autoDepth(std::size_t n, int occTarget = 64) {
        if (n <= (std::size_t)occTarget) return 0;
        int L = (int)std::lround(std::log(double(n) / occTarget) / std::log(8.0));
        return L < 0 ? 0 : L;
    }

    Accelerations3D computeAccelerations(const Bodies3D& bodies) const override {
        using fmm::Local;
        using fmm::Multipole;
        const std::size_t n = bodies.size();
        Accelerations3D acc(n, Vec3{});
        if (n == 0) return acc;

        const int L = depthOverride_ >= 0 ? depthOverride_ : autoDepth(n, occTarget_);
        const int nside = 1 << L;

        // ---- cubic bounding box -------------------------------------------
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < n; ++i) {
            const Vec3& p = bodies.pos[i];
            lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
            lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
            lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
        }
        const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]), cz = 0.5 * (lo[2] + hi[2]);
        double half = 0.5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
        if (!(half > 0.0)) half = 1.0;
        half *= 1.0 + 1e-9;
        const double side = 2.0 * half;
        const double ox = cx - half, oy = cy - half, oz = cz - half;

        auto cellAt = [&](int d) { return side / double(1 << d); };
        auto centerAt = [&](int d, int ix, int iy, int iz) {
            const double c = cellAt(d);
            return Vec3{ox + (ix + 0.5) * c, oy + (iy + 0.5) * c, oz + (iz + 0.5) * c};
        };
        auto clampi = [](int v, int hiv) { return v < 0 ? 0 : (v > hiv ? hiv : v); };
        auto bidx = [](int ix, int iy, int iz, int ns) { return (iz * ns + iy) * ns + ix; };

        // ---- assign bodies to leaves --------------------------------------
        std::vector<std::vector<int>> leafBodies((std::size_t)nside * nside * nside);
        for (std::size_t i = 0; i < n; ++i) {
            const int ix = clampi(int(std::floor((bodies.pos[i].x - ox) / side * nside)), nside - 1);
            const int iy = clampi(int(std::floor((bodies.pos[i].y - oy) / side * nside)), nside - 1);
            const int iz = clampi(int(std::floor((bodies.pos[i].z - oz) / side * nside)), nside - 1);
            leafBodies[bidx(ix, iy, iz, nside)].push_back((int)i);
        }

        // ---- per-level storage --------------------------------------------
        std::vector<std::vector<Multipole>> M(L + 1);
        std::vector<std::vector<Local>> Lex(L + 1);
        std::vector<std::vector<char>> nonempty(L + 1);
        for (int d = 0; d <= L; ++d) {
            const int ns = 1 << d;
            const std::size_t nb = (std::size_t)ns * ns * ns;
            M[d].assign(nb, Multipole(Vec3{}, p_));
            Lex[d].assign(nb, Local(Vec3{}, p_));
            nonempty[d].assign(nb, 0);
            for (int iz = 0; iz < ns; ++iz)
                for (int iy = 0; iy < ns; ++iy)
                    for (int ix = 0; ix < ns; ++ix) {
                        const Vec3 ctr = centerAt(d, ix, iy, iz);
                        const int b = bidx(ix, iy, iz, ns);
                        M[d][b].center = ctr;
                        Lex[d][b].center = ctr;
                    }
        }

        // ---- upward: P2M at leaves ----------------------------------------
        for (int iz = 0; iz < nside; ++iz)
            for (int iy = 0; iy < nside; ++iy)
                for (int ix = 0; ix < nside; ++ix) {
                    const int b = bidx(ix, iy, iz, nside);
                    const auto& bl = leafBodies[b];
                    if (bl.empty()) continue;
                    nonempty[L][b] = 1;
                    std::vector<Vec3> pos;
                    std::vector<double> q;
                    pos.reserve(bl.size()); q.reserve(bl.size());
                    for (int j : bl) { pos.push_back(bodies.pos[j]); q.push_back(bodies.mass[j]); }
                    M[L][b] = fmm::P2M(centerAt(L, ix, iy, iz), pos, q, p_);
                }

        // ---- upward: M2M up the tree --------------------------------------
        const int S = (p_ + 1) * (p_ + 1);
        for (int d = L - 1; d >= 0; --d) {
            const int ns = 1 << d, nsc = 1 << (d + 1);
            for (int iz = 0; iz < ns; ++iz)
                for (int iy = 0; iy < ns; ++iy)
                    for (int ix = 0; ix < ns; ++ix) {
                        const int b = bidx(ix, iy, iz, ns);
                        const Vec3 ctr = M[d][b].center;
                        bool any = false;
                        for (int o = 0; o < 8; ++o) {
                            const int cxi = 2 * ix + (o & 1), cyi = 2 * iy + ((o >> 1) & 1),
                                      czi = 2 * iz + ((o >> 2) & 1);
                            const int cb = bidx(cxi, cyi, czi, nsc);
                            if (!nonempty[d + 1][cb]) continue;
                            any = true;
                            const Multipole sh = fmm::M2M(M[d + 1][cb], ctr);
                            for (int t = 0; t < S; ++t) M[d][b].M[t] += sh.M[t];
                        }
                        nonempty[d][b] = any ? 1 : 0;
                    }
        }

        // ---- downward: L2L from parent + M2L from interaction list --------
        for (int d = 2; d <= L; ++d) {
            const int ns = 1 << d, nsp = ns / 2;
            for (int iz = 0; iz < ns; ++iz)
                for (int iy = 0; iy < ns; ++iy)
                    for (int ix = 0; ix < ns; ++ix) {
                        const int b = bidx(ix, iy, iz, ns);
                        const Vec3 ctr = Lex[d][b].center;

                        if (d >= 3) {
                            const int pb = bidx(ix / 2, iy / 2, iz / 2, nsp);
                            const Local up = fmm::L2L(Lex[d - 1][pb], ctr);
                            for (int t = 0; t < S; ++t) Lex[d][b].L[t] += up.L[t];
                        }

                        const int px = ix / 2, py = iy / 2, pz = iz / 2;
                        for (int qz = pz - 1; qz <= pz + 1; ++qz) {
                            if (qz < 0 || qz >= nsp) continue;
                            for (int qy = py - 1; qy <= py + 1; ++qy) {
                                if (qy < 0 || qy >= nsp) continue;
                                for (int qx = px - 1; qx <= px + 1; ++qx) {
                                    if (qx < 0 || qx >= nsp) continue;
                                    for (int o = 0; o < 8; ++o) {
                                        const int ax = 2 * qx + (o & 1), ay = 2 * qy + ((o >> 1) & 1),
                                                  az = 2 * qz + ((o >> 2) & 1);
                                        if (std::max({std::abs(ax - ix), std::abs(ay - iy),
                                                      std::abs(az - iz)}) <= 1)
                                            continue; // nearest neighbour -> P2P
                                        const int ab = bidx(ax, ay, az, ns);
                                        if (!nonempty[d][ab]) continue;
                                        const Local m = fmm::M2L(M[d][ab], ctr);
                                        for (int t = 0; t < S; ++t) Lex[d][b].L[t] += m.L[t];
                                    }
                                }
                            }
                        }
                    }
        }

        // ---- leaf evaluation: far field (L2P) + near field (P2P) ----------
        const double eps2 = params_.eps * params_.eps, G = params_.G;
        for (int iz = 0; iz < nside; ++iz)
            for (int iy = 0; iy < nside; ++iy)
                for (int ix = 0; ix < nside; ++ix) {
                    const int b = bidx(ix, iy, iz, nside);
                    const auto& bl = leafBodies[b];
                    if (bl.empty()) continue;
                    const Local& Lb = Lex[L][b];
                    for (int ti : bl) {
                        const Vec3 zi = bodies.pos[ti];
                        Vec3 a = G * fmm::localField(Lb, zi); // far field
                        for (int nz = iz - 1; nz <= iz + 1; ++nz) {
                            if (nz < 0 || nz >= nside) continue;
                            for (int ny = iy - 1; ny <= iy + 1; ++ny) {
                                if (ny < 0 || ny >= nside) continue;
                                for (int nx = ix - 1; nx <= ix + 1; ++nx) {
                                    if (nx < 0 || nx >= nside) continue;
                                    for (int tj : leafBodies[bidx(nx, ny, nz, nside)]) {
                                        if (tj == ti) continue;
                                        const Vec3 d = bodies.pos[tj] - zi;
                                        const double r2 = norm2(d) + eps2;
                                        a += (G * bodies.mass[tj] / (r2 * std::sqrt(r2))) * d;
                                    }
                                }
                            }
                        }
                        acc[ti] = a;
                    }
                }
        return acc;
    }

    const char* name() const override { return "FMM3D(O(N) uniform)"; }
    int order() const { return p_; }

private:
    KernelParams3D params_;
    int p_;
    int depthOverride_;
    int occTarget_;
};

} // namespace galaxy3d
