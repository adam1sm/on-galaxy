#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "Types3D.hpp"

// ============================================================================
// RUNG 10a: adaptive octree + Carrier-Greengard-Rokhlin U/V/W/X interaction
// lists (CPU). Subdivide a box only if it holds > threshold particles, so
// leaves sit at varying depths. Flat, index-based layout (GPU-ready for 10b).
//
// Adjacency (any levels): two boxes touch/overlap iff, per axis,
//   |center_a - center_b| <= half_a + half_b.  "Well-separated" = not adjacent.
//
// Lists (per box b), reusing the validated operators in the solver:
//   colleagues(b) : same-level boxes adjacent to b (incl. b).
//   V(b)  [any b] : separated children of colleagues(parent(b))  -> M2L.
//   U(b)  [leaf]  : all leaves adjacent to b (any level)         -> P2P.
//   W(b)  [leaf]  : separated boxes reached by descending b's non-leaf
//                   colleagues (finer, parent adjacent to b)     -> M2P.
//   X(b)  [any b] : exact dual of W -- c in X(b) iff b in W(c)   -> P2L.
//
// W/X are EMPTY for a uniform/balanced tree (no adjacent level mismatch), so
// adaptive generalizes the validated uniform behaviour (Gate A).
// ============================================================================

namespace galaxy3d {

struct ABox {
    int parent = -1;
    int child[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int level = 0;
    double cx = 0, cy = 0, cz = 0, half = 0;
    bool leaf = true;
    int pfirst = 0, pcount = 0; // particle range into reordered arrays
};

struct AdaptiveOctree {
    std::vector<ABox> box;
    // particles reordered so each leaf's are contiguous
    std::vector<double> px, py, pz, mass;
    std::vector<int> origIndex; // reordered -> original
    // per-box interaction lists
    std::vector<std::vector<int>> colleague, U, V, W, X;
    std::vector<int> leaves; // leaf box ids
    double tol = 1e-9;
    int maxDepth = 24; // safety cap: stop subdividing (>threshold leaf stays) so
                       // near-/exactly-coincident particle clumps cannot recurse
                       // forever (handled by P2P). No effect on non-degenerate data.
    double epsFloor = 0.0; // min-leaf floor: stop subdividing once a leaf edge
                           // (2*half) would fall to/below eps, so the pure-kernel
                           // far field never acts on pairs closer than the
                           // softening length (consistent softening, Rung 12).
                           // INERT at eps=0 -> 10a/10b convergence gates unchanged.

    bool adjacent(int a, int b) const {
        const ABox& A = box[a]; const ABox& B = box[b];
        const double s = A.half + B.half + tol;
        return std::abs(A.cx - B.cx) <= s && std::abs(A.cy - B.cy) <= s && std::abs(A.cz - B.cz) <= s;
    }

    // ---- build (recursive subdivide) --------------------------------------
    void build(const Bodies3D& b, int threshold, double epsFloor_ = 0.0) {
        epsFloor = epsFloor_;
        const std::size_t n = b.size();
        std::vector<int> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = (int)i;

        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (std::size_t i = 0; i < n; ++i) {
            lo[0] = std::min(lo[0], b.pos[i].x); hi[0] = std::max(hi[0], b.pos[i].x);
            lo[1] = std::min(lo[1], b.pos[i].y); hi[1] = std::max(hi[1], b.pos[i].y);
            lo[2] = std::min(lo[2], b.pos[i].z); hi[2] = std::max(hi[2], b.pos[i].z);
        }
        double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]), cz = 0.5 * (lo[2] + hi[2]);
        double half = 0.5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
        if (!(half > 0)) half = 1.0;
        half *= 1.0 + 1e-9;
        tol = 1e-9 * half;

        box.clear();
        ABox root; root.cx = cx; root.cy = cy; root.cz = cz; root.half = half;
        root.level = 0; root.pfirst = 0; root.pcount = (int)n;
        box.push_back(root);
        subdivide(0, b, idx, threshold);

        // gather reordered particle arrays in idx order
        px.resize(n); py.resize(n); pz.resize(n); mass.resize(n); origIndex.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const int j = idx[k];
            px[k] = b.pos[j].x; py[k] = b.pos[j].y; pz[k] = b.pos[j].z;
            mass[k] = b.mass[j]; origIndex[k] = j;
        }
        for (int i = 0; i < (int)box.size(); ++i) if (box[i].leaf && box[i].pcount > 0) leaves.push_back(i);
    }

    void subdivide(int bi, const Bodies3D& b, std::vector<int>& idx, int threshold) {
        // subdivide iff occupancy>threshold AND leaf-edge (2*half) > eps floor
        // (and below the depth cap). epsFloor=0 -> the edge test is always true.
        if (box[bi].pcount <= threshold || 2.0 * box[bi].half <= epsFloor ||
            box[bi].level >= maxDepth)
            return;
        const ABox cur = box[bi];
        const int lo = cur.pfirst, hi = cur.pfirst + cur.pcount;
        const double h2 = 0.5 * cur.half;
        // count octants
        auto oct = [&](int j) {
            return (b.pos[j].x >= cur.cx ? 1 : 0) | (b.pos[j].y >= cur.cy ? 2 : 0) | (b.pos[j].z >= cur.cz ? 4 : 0);
        };
        int cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = lo; k < hi; ++k) ++cnt[oct(idx[k])];
        // stable counting-sort into octant groups
        int start[8], off[8], acc = lo;
        for (int o = 0; o < 8; ++o) { start[o] = acc; off[o] = acc; acc += cnt[o]; }
        std::vector<int> tmp(idx.begin() + lo, idx.begin() + hi);
        for (int j : tmp) idx[off[oct(j)]++] = j;
        // create children, recurse
        box[bi].leaf = false;
        for (int o = 0; o < 8; ++o) {
            if (cnt[o] == 0) continue;
            ABox ch;
            ch.parent = bi; ch.level = cur.level + 1; ch.half = h2;
            ch.cx = cur.cx + ((o & 1) ? h2 : -h2);
            ch.cy = cur.cy + ((o & 2) ? h2 : -h2);
            ch.cz = cur.cz + ((o & 4) ? h2 : -h2);
            ch.pfirst = start[o]; ch.pcount = cnt[o];
            const int ci = (int)box.size();
            box.push_back(ch);
            box[bi].child[o] = ci;
        }
        for (int o = 0; o < 8; ++o) if (box[bi].child[o] >= 0) subdivide(box[bi].child[o], b, idx, threshold);
    }

    // ---- interaction lists ------------------------------------------------
    void buildLists() {
        const int nb = (int)box.size();
        colleague.assign(nb, {}); U.assign(nb, {}); V.assign(nb, {}); W.assign(nb, {}); X.assign(nb, {});

        // colleagues, level by level (BFS-ish: parents before children since boxes
        // are created parent-before-child, so index order is a valid topological order)
        colleague[0] = {0}; // root: only itself
        for (int b = 1; b < nb; ++b) {
            const int par = box[b].parent;
            for (int pc : colleague[par])
                for (int o = 0; o < 8; ++o) {
                    const int c = box[pc].child[o];
                    if (c >= 0 && adjacent(b, c)) colleague[b].push_back(c);
                }
        }

        // V(b): separated children of parent's colleagues
        for (int b = 1; b < nb; ++b) {
            const int par = box[b].parent;
            for (int pc : colleague[par])
                for (int o = 0; o < 8; ++o) {
                    const int c = box[pc].child[o];
                    if (c >= 0 && !adjacent(b, c)) V[b].push_back(c);
                }
        }

        // U(b) [leaf]: all leaves adjacent to b (descend from root through adjacent boxes)
        for (int b : leaves) collectAdjLeaves(0, b, U[b]);

        // W(b) [leaf]: descend non-leaf colleagues, collect separated boxes
        for (int b : leaves)
            for (int c : colleague[b])
                if (c != b && !box[c].leaf) descendW(c, b);

        // X = exact dual of W:  c in X(w) for every w in W(c)
        for (int c : leaves)
            for (int w : W[c]) X[w].push_back(c);
    }

    void collectAdjLeaves(int node, int b, std::vector<int>& out) const {
        if (!adjacent(node, b)) return;
        if (box[node].leaf) { if (box[node].pcount > 0) out.push_back(node); return; }
        for (int o = 0; o < 8; ++o) if (box[node].child[o] >= 0) collectAdjLeaves(box[node].child[o], b, out);
    }
    void descendW(int node, int b) {
        for (int o = 0; o < 8; ++o) {
            const int ch = box[node].child[o];
            if (ch < 0) continue;
            if (!adjacent(ch, b)) W[b].push_back(ch);       // well-sep finer box -> M2P
            else if (!box[ch].leaf) descendW(ch, b);        // adjacent non-leaf -> descend
            // adjacent leaf children are picked up by U via collectAdjLeaves
        }
    }

    int maxLeafOcc() const {
        int m = 0;
        for (int b : leaves) m = std::max(m, box[b].pcount);
        return m;
    }
};

} // namespace galaxy3d
