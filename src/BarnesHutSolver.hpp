#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "ForceSolver.hpp"
#include "Kernel.hpp"

namespace galaxy {

// ============================================================================
// RUNG 1: Barnes-Hut O(N log N) solver.
// ============================================================================
//
// Just another ForceSolver behind the existing interface -- the kernel,
// integrator, and Types are untouched. It computes the SAME physical quantity
// as DirectSolver (2D log kernel, same eps), only approximately, by clustering
// distant bodies into their monopole (total mass at center of mass).
//
//   - Adaptive 2D quadtree, rebuilt from scratch every call. The root box is
//     recomputed each call to enclose all current bodies (the disk breathes).
//   - Each node stores total mass and center of mass (COM).
//   - MONOPOLE approximation only this rung (no quadrupole).
//   - Opening criterion: node_size / distance_to_COM < theta  ->  use monopole;
//     otherwise recurse into children.
//   - Robustness: max tree depth cap + a small per-leaf bucket so near- or
//     exactly-coincident bodies can't trigger infinite subdivision; closeness
//     is handled by the SAME softening eps as DirectSolver.
//   - Self-interaction is skipped: a body never pulls on itself.
//
// CORRECTNESS GATE: at theta = 0 the opening criterion (node_size^2 < 0) never
// fires, so the walk always recurses to the leaves and sums every other body
// directly -- i.e. BH reduces to exact direct summation and matches
// DirectSolver to roundoff. The harness asserts this.
class BarnesHutSolver final : public ForceSolver {
public:
    BarnesHutSolver(KernelParams params, double theta,
                    int leaf_capacity = 1, int max_depth = 64)
        : params_(params), theta_(theta),
          leaf_capacity_(leaf_capacity < 1 ? 1 : leaf_capacity),
          max_depth_(max_depth) {}

    Accelerations computeAccelerations(const Bodies& bodies) const override {
        const std::size_t n = bodies.size();
        Accelerations acc(n, Complex{0.0, 0.0});
        if (n == 0) return acc;

        Tree tree;
        tree.build(bodies, leaf_capacity_, max_depth_);

        const double theta2 = theta_ * theta_;
        const double eps2 = params_.eps * params_.eps;
        for (std::size_t i = 0; i < n; ++i) {
            acc[i] = tree.walk(bodies, static_cast<int>(i), theta2, params_.G, eps2);
        }
        return acc;
    }

    const char* name() const override { return "BarnesHutSolver(O(N log N))"; }

    double theta() const { return theta_; }

private:
    // ---- quadtree ---------------------------------------------------------
    struct Node {
        double cx = 0.0, cy = 0.0, half = 0.0; // box center and half-width
        double mass = 0.0;                     // total mass in subtree
        double comx = 0.0, comy = 0.0;         // center of mass
        int child[4] = {-1, -1, -1, -1};       // child node indices, -1 if absent
        int first = 0, count = 0;              // leaf body range into idx_
        bool leaf = false;
    };

    struct Tree {
        std::vector<Node> nodes;
        std::vector<int> idx;  // body indices, reordered in place during build
        std::vector<int> tmp;  // scratch for the per-range quadrant sort
        const Bodies* b = nullptr;

        void build(const Bodies& bodies, int leaf_capacity, int max_depth) {
            b = &bodies;
            const std::size_t n = bodies.size();
            idx.resize(n);
            for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<int>(i);
            tmp.resize(n);
            nodes.clear();
            nodes.reserve(2 * n + 1);

            // Root box: square enclosing all bodies, with a small margin so no
            // body sits exactly on a boundary.
            double minx = std::numeric_limits<double>::infinity();
            double miny = minx, maxx = -minx, maxy = -minx;
            for (std::size_t i = 0; i < n; ++i) {
                const double x = bodies.pos[i].real(), y = bodies.pos[i].imag();
                minx = std::min(minx, x); maxx = std::max(maxx, x);
                miny = std::min(miny, y); maxy = std::max(maxy, y);
            }
            const double cx = 0.5 * (minx + maxx);
            const double cy = 0.5 * (miny + maxy);
            double half = 0.5 * std::max(maxx - minx, maxy - miny);
            if (!(half > 0.0)) half = 1.0; // all coincident -> arbitrary nonzero box
            half *= 1.0 + 1e-9;

            buildRange(0, static_cast<int>(n), cx, cy, half, 0,
                       leaf_capacity, max_depth);
        }

        // Build the node spanning idx[lo,hi); returns its node index.
        int buildRange(int lo, int hi, double cx, double cy, double half,
                       int depth, int leaf_capacity, int max_depth) {
            const int self = static_cast<int>(nodes.size());
            nodes.push_back(Node{});
            {
                Node& nd = nodes[self];
                nd.cx = cx; nd.cy = cy; nd.half = half;
            }

            // Accumulate mass and COM over this range.
            double m = 0.0, sx = 0.0, sy = 0.0;
            for (int k = lo; k < hi; ++k) {
                const int j = idx[k];
                const double mj = b->mass[j];
                m += mj;
                sx += mj * b->pos[j].real();
                sy += mj * b->pos[j].imag();
            }
            {
                Node& nd = nodes[self];
                nd.mass = m;
                nd.comx = m > 0.0 ? sx / m : cx;
                nd.comy = m > 0.0 ? sy / m : cy;
            }

            const int cnt = hi - lo;
            if (cnt <= leaf_capacity || depth >= max_depth) {
                Node& nd = nodes[self];
                nd.leaf = true;
                nd.first = lo;
                nd.count = cnt;
                return self;
            }

            // Counting-sort idx[lo,hi) into 4 quadrant groups (in place via tmp).
            int qcount[4] = {0, 0, 0, 0};
            for (int k = lo; k < hi; ++k) {
                const int j = idx[k];
                const int q = (b->pos[j].real() >= cx ? 1 : 0) |
                              (b->pos[j].imag() >= cy ? 2 : 0);
                ++qcount[q];
            }
            int start[4], offset[4];
            int acc = lo;
            for (int q = 0; q < 4; ++q) { start[q] = acc; offset[q] = acc; acc += qcount[q]; }
            for (int k = lo; k < hi; ++k) {
                const int j = idx[k];
                const int q = (b->pos[j].real() >= cx ? 1 : 0) |
                              (b->pos[j].imag() >= cy ? 2 : 0);
                tmp[offset[q]++] = j;
            }
            for (int k = lo; k < hi; ++k) idx[k] = tmp[k];

            const double h2 = 0.5 * half;
            for (int q = 0; q < 4; ++q) {
                if (qcount[q] == 0) continue;
                const double ccx = cx + ((q & 1) ? h2 : -h2);
                const double ccy = cy + ((q & 2) ? h2 : -h2);
                const int childIdx = buildRange(start[q], start[q] + qcount[q],
                                                ccx, ccy, h2, depth + 1,
                                                leaf_capacity, max_depth);
                nodes[self].child[q] = childIdx;
            }
            return self;
        }

        // Acceleration on body i from the whole tree (iterative walk).
        Complex walk(const Bodies& bodies, int i, double theta2,
                     double G, double eps2) const {
            const double xi = bodies.pos[i].real();
            const double yi = bodies.pos[i].imag();
            double ax = 0.0, ay = 0.0;

            // Explicit stack avoids recursion overhead in the hot loop.
            int stack[256];
            int sp = 0;
            stack[sp++] = 0; // root
            while (sp > 0) {
                const Node& nd = nodes[stack[--sp]];
                if (nd.leaf) {
                    for (int k = nd.first; k < nd.first + nd.count; ++k) {
                        const int j = idx[k];
                        if (j == i) continue; // skip self-interaction
                        const double dx = bodies.pos[j].real() - xi;
                        const double dy = bodies.pos[j].imag() - yi;
                        const double denom = dx * dx + dy * dy + eps2;
                        const double f = G * bodies.mass[j] / denom;
                        ax += f * dx; ay += f * dy;
                    }
                    continue;
                }
                const double dx = nd.comx - xi;
                const double dy = nd.comy - yi;
                const double d2 = dx * dx + dy * dy;
                const double s = 2.0 * nd.half; // node_size (full box width)
                // node_size/dist < theta  <=>  s^2 < theta^2 * d2.
                // When d2 == 0 (or theta == 0) this is false, forcing recursion,
                // so a node containing the target body is always opened.
                if (s * s < theta2 * d2) {
                    const double f = G * nd.mass / (d2 + eps2);
                    ax += f * dx; ay += f * dy;
                } else {
                    for (int q = 0; q < 4; ++q) {
                        if (nd.child[q] >= 0) stack[sp++] = nd.child[q];
                    }
                }
            }
            return Complex{ax, ay};
        }
    };

    KernelParams params_;
    double theta_;
    int leaf_capacity_;
    int max_depth_;
};

} // namespace galaxy
