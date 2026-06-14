#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "ForceSolver3D.hpp"
#include "Kernel3D.hpp"

namespace galaxy3d {

// ============================================================================
// RUNG 5: Barnes-Hut O(N log N) baseline in 3D (the scaling reference).
//
// Adaptive OCTREE (8 children) over the cubic bounding box; each node stores
// total mass + center of mass; opened by the angle criterion node_size/dist <
// theta, approximating distant nodes by their monopole using the SAME 3D
// Plummer kernel (Kernel3D.hpp). Softening eps is applied consistently (the
// monopole is a softened-kernel approximation), so BH3D has no far-field
// softening mismatch. Implements ForceSolver3D, so the harness can diff it
// against DirectSolver3D.
// ============================================================================
class BarnesHut3D final : public ForceSolver3D {
public:
    BarnesHut3D(KernelParams3D params, double theta,
                int leaf_capacity = 1, int max_depth = 32)
        : params_(params), theta_(theta),
          leaf_capacity_(leaf_capacity < 1 ? 1 : leaf_capacity),
          max_depth_(max_depth) {}

    Accelerations3D computeAccelerations(const Bodies3D& bodies) const override {
        const std::size_t n = bodies.size();
        Accelerations3D acc(n, Vec3{});
        if (n == 0) return acc;

        Tree tree;
        tree.build(bodies, leaf_capacity_, max_depth_);
        const double theta2 = theta_ * theta_;
        const double eps2 = params_.eps * params_.eps;
        for (std::size_t i = 0; i < n; ++i)
            acc[i] = tree.walk(bodies, static_cast<int>(i), theta2, params_.G, eps2);
        return acc;
    }

    const char* name() const override { return "BarnesHut3D(O(N log N))"; }
    double theta() const { return theta_; }

private:
    struct Node {
        double cx = 0, cy = 0, cz = 0, half = 0; // cube center + half-width
        double mass = 0, comx = 0, comy = 0, comz = 0;
        int child[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int first = 0, count = 0;
        bool leaf = false;
    };

    struct Tree {
        std::vector<Node> nodes;
        std::vector<int> idx, tmp;
        const Bodies3D* b = nullptr;

        void build(const Bodies3D& bodies, int leaf_capacity, int max_depth) {
            b = &bodies;
            const std::size_t n = bodies.size();
            idx.resize(n);
            for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<int>(i);
            tmp.resize(n);
            nodes.clear();
            nodes.reserve(2 * n + 1);

            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (std::size_t i = 0; i < n; ++i) {
                const Vec3& p = bodies.pos[i];
                lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
                lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
                lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
            }
            const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]),
                         cz = 0.5 * (lo[2] + hi[2]);
            double half = 0.5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
            if (!(half > 0.0)) half = 1.0;
            half *= 1.0 + 1e-9;
            buildRange(0, static_cast<int>(n), cx, cy, cz, half, 0, leaf_capacity, max_depth);
        }

        int buildRange(int loi, int hii, double cx, double cy, double cz, double half,
                       int depth, int leaf_capacity, int max_depth) {
            const int self = static_cast<int>(nodes.size());
            nodes.push_back(Node{});
            nodes[self].cx = cx; nodes[self].cy = cy; nodes[self].cz = cz; nodes[self].half = half;

            double m = 0, sx = 0, sy = 0, sz = 0;
            for (int k = loi; k < hii; ++k) {
                const int j = idx[k];
                const double mj = b->mass[j];
                m += mj; sx += mj * b->pos[j].x; sy += mj * b->pos[j].y; sz += mj * b->pos[j].z;
            }
            nodes[self].mass = m;
            nodes[self].comx = m > 0 ? sx / m : cx;
            nodes[self].comy = m > 0 ? sy / m : cy;
            nodes[self].comz = m > 0 ? sz / m : cz;

            const int cnt = hii - loi;
            if (cnt <= leaf_capacity || depth >= max_depth) {
                nodes[self].leaf = true; nodes[self].first = loi; nodes[self].count = cnt;
                return self;
            }

            auto octant = [&](const Vec3& p) {
                return (p.x >= cx ? 1 : 0) | (p.y >= cy ? 2 : 0) | (p.z >= cz ? 4 : 0);
            };
            int oc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = loi; k < hii; ++k) ++oc[octant(b->pos[idx[k]])];
            int start[8], off[8], acc = loi;
            for (int o = 0; o < 8; ++o) { start[o] = acc; off[o] = acc; acc += oc[o]; }
            for (int k = loi; k < hii; ++k) { const int j = idx[k]; tmp[off[octant(b->pos[j])]++] = j; }
            for (int k = loi; k < hii; ++k) idx[k] = tmp[k];

            const double h2 = 0.5 * half;
            for (int o = 0; o < 8; ++o) {
                if (oc[o] == 0) continue;
                const double ccx = cx + ((o & 1) ? h2 : -h2);
                const double ccy = cy + ((o & 2) ? h2 : -h2);
                const double ccz = cz + ((o & 4) ? h2 : -h2);
                nodes[self].child[o] =
                    buildRange(start[o], start[o] + oc[o], ccx, ccy, ccz, h2, depth + 1,
                               leaf_capacity, max_depth);
            }
            return self;
        }

        Vec3 walk(const Bodies3D& bodies, int i, double theta2, double G, double eps2) const {
            const Vec3 pi = bodies.pos[i];
            Vec3 a{};
            int stack[512];
            int sp = 0;
            stack[sp++] = 0;
            while (sp > 0) {
                const Node& nd = nodes[stack[--sp]];
                if (nd.leaf) {
                    for (int k = nd.first; k < nd.first + nd.count; ++k) {
                        const int j = idx[k];
                        if (j == i) continue;
                        const Vec3 d = bodies.pos[j] - pi;
                        const double r2 = norm2(d) + eps2;
                        a += (G * bodies.mass[j] / (r2 * std::sqrt(r2))) * d;
                    }
                    continue;
                }
                const Vec3 d{nd.comx - pi.x, nd.comy - pi.y, nd.comz - pi.z};
                const double d2 = norm2(d), s = 2.0 * nd.half;
                if (s * s < theta2 * d2) {
                    const double r2 = d2 + eps2;
                    a += (G * nd.mass / (r2 * std::sqrt(r2))) * d;
                } else {
                    for (int o = 0; o < 8; ++o)
                        if (nd.child[o] >= 0) stack[sp++] = nd.child[o];
                }
            }
            return a;
        }
    };

    KernelParams3D params_;
    double theta_;
    int leaf_capacity_, max_depth_;
};

} // namespace galaxy3d
