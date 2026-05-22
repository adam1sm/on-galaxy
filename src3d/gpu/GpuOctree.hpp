#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../Types3D.hpp"

// ============================================================================
// RUNG 7: flat, index-based UNIFORM octree layout, built CPU-side from the
// Rung 5 logic (FMM3D.hpp) but emitted as GPU-transferable arrays. GPU tree
// build is a Rung 8 optimization -- correctness first.
//
// Box global id = levelOffset[d] + bidx(ix,iy,iz; 2^d).  Leaves are level-L
// boxes. Particles are reordered into contiguous per-leaf ranges. Interaction
// lists (M2L) and neighbour lists (P2P) are stored CSR-style. Empty boxes are
// pruned from the lists (their multipoles are zero), exactly as FMM3D does.
// ============================================================================

namespace galaxy3d {
namespace gpu {

struct GpuOctree {
    int L = 0;
    double ox = 0, oy = 0, oz = 0, side = 0; // bounding cube origin corner + side
    std::vector<int> levelOffset;            // size L+2
    int nBoxes = 0;

    // Per box (global id):
    std::vector<double> cx, cy, cz;          // box centers
    std::vector<int> parent;                 // parent global id, -1 for root
    std::vector<int> child;                  // nBoxes*8, child global id or -1
    std::vector<char> nonempty;              // 1 if subtree holds any particle

    // M2L interaction lists, CSR over boxes (only nonempty target boxes get entries):
    std::vector<int> ilistStart;             // nBoxes+1
    std::vector<int> ilist;                  // source box global ids (nonempty)

    // Leaves (level L), indexed by leaf-local id = bidx at level L:
    int nLeaves = 0;
    std::vector<int> leafFirst, leafCount;   // particle range into reordered arrays
    std::vector<int> leafBox;                // leaf-local -> global box id
    // P2P neighbour lists, CSR over leaves (neighbour leaf-local ids, incl self):
    std::vector<int> nbrStart;               // nLeaves+1
    std::vector<int> nbr;                    // neighbour leaf-local ids

    // Particles reordered into leaf order:
    std::vector<double> px, py, pz, pmass;   // size N
    std::vector<int> origIndex;              // leaf-order -> original index

    double leafSize() const { return side / double(1 << L); }
    int boxId(int d, int ix, int iy, int iz) const {
        const int ns = 1 << d;
        return levelOffset[d] + (iz * ns + iy) * ns + ix;
    }
};

// Auto depth: 8^L ~ N / occTarget (occupancy tuned high for the GPU, where P2P
// is cheap and the O(p^4) per-box M2L is the cost to minimize).
inline int gpuAutoDepth(std::size_t n, int occTarget) {
    if (n <= (std::size_t)occTarget) return 0;
    int L = (int)std::lround(std::log(double(n) / occTarget) / std::log(8.0));
    return L < 0 ? 0 : L;
}

inline GpuOctree buildGpuOctree(const Bodies3D& b, int depth) {
    GpuOctree t;
    const int L = depth;
    t.L = L;
    const std::size_t n = b.size();
    const int nside = 1 << L;

    // ---- bounding cube (identical to FMM3D) -------------------------------
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3& p = b.pos[i];
        lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
        lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
        lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
    }
    const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]), cz = 0.5 * (lo[2] + hi[2]);
    double half = 0.5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    if (!(half > 0.0)) half = 1.0;
    half *= 1.0 + 1e-9;
    t.side = 2.0 * half;
    t.ox = cx - half; t.oy = cy - half; t.oz = cz - half;

    // ---- level offsets / box count ----------------------------------------
    t.levelOffset.assign(L + 2, 0);
    for (int d = 0; d <= L; ++d) t.levelOffset[d + 1] = t.levelOffset[d] + (1 << (3 * d));
    t.nBoxes = t.levelOffset[L + 1];

    auto clampi = [](int v, int hiv) { return v < 0 ? 0 : (v > hiv ? hiv : v); };
    auto bidx = [](int ix, int iy, int iz, int ns) { return (iz * ns + iy) * ns + ix; };

    // ---- assign particles to leaves, then reorder into contiguous ranges --
    t.nLeaves = 1 << (3 * L);
    std::vector<std::vector<int>> bucket(t.nLeaves);
    for (std::size_t i = 0; i < n; ++i) {
        const int ix = clampi(int(std::floor((b.pos[i].x - t.ox) / t.side * nside)), nside - 1);
        const int iy = clampi(int(std::floor((b.pos[i].y - t.oy) / t.side * nside)), nside - 1);
        const int iz = clampi(int(std::floor((b.pos[i].z - t.oz) / t.side * nside)), nside - 1);
        bucket[bidx(ix, iy, iz, nside)].push_back((int)i);
    }
    t.px.resize(n); t.py.resize(n); t.pz.resize(n); t.pmass.resize(n); t.origIndex.resize(n);
    t.leafFirst.assign(t.nLeaves, 0); t.leafCount.assign(t.nLeaves, 0);
    {
        int cursor = 0;
        for (int leaf = 0; leaf < t.nLeaves; ++leaf) {
            t.leafFirst[leaf] = cursor;
            for (int j : bucket[leaf]) {
                t.px[cursor] = b.pos[j].x; t.py[cursor] = b.pos[j].y; t.pz[cursor] = b.pos[j].z;
                t.pmass[cursor] = b.mass[j]; t.origIndex[cursor] = j;
                ++cursor;
            }
            t.leafCount[leaf] = cursor - t.leafFirst[leaf];
        }
    }

    // ---- centers, parent, child, nonempty ---------------------------------
    t.cx.assign(t.nBoxes, 0); t.cy.assign(t.nBoxes, 0); t.cz.assign(t.nBoxes, 0);
    t.parent.assign(t.nBoxes, -1);
    t.child.assign((std::size_t)t.nBoxes * 8, -1);
    t.nonempty.assign(t.nBoxes, 0);
    t.leafBox.assign(t.nLeaves, 0);

    for (int d = 0; d <= L; ++d) {
        const int ns = 1 << d;
        const double cell = t.side / double(ns);
        for (int iz = 0; iz < ns; ++iz)
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int g = t.boxId(d, ix, iy, iz);
                    t.cx[g] = t.ox + (ix + 0.5) * cell;
                    t.cy[g] = t.oy + (iy + 0.5) * cell;
                    t.cz[g] = t.oz + (iz + 0.5) * cell;
                    if (d > 0) t.parent[g] = t.boxId(d - 1, ix / 2, iy / 2, iz / 2);
                    if (d < L) {
                        const int nsc = ns * 2;
                        for (int o = 0; o < 8; ++o)
                            t.child[(std::size_t)g * 8 + o] =
                                t.boxId(d + 1, 2 * ix + (o & 1), 2 * iy + ((o >> 1) & 1), 2 * iz + ((o >> 2) & 1)),
                            (void)nsc;
                    }
                }
    }
    // leaf nonempty + leafBox
    for (int iz = 0; iz < nside; ++iz)
        for (int iy = 0; iy < nside; ++iy)
            for (int ix = 0; ix < nside; ++ix) {
                const int leaf = bidx(ix, iy, iz, nside);
                const int g = t.boxId(L, ix, iy, iz);
                t.leafBox[leaf] = g;
                if (t.leafCount[leaf] > 0) t.nonempty[g] = 1;
            }
    // propagate nonempty up
    for (int d = L - 1; d >= 0; --d) {
        const int ns = 1 << d;
        for (int iz = 0; iz < ns; ++iz)
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int g = t.boxId(d, ix, iy, iz);
                    bool any = false;
                    for (int o = 0; o < 8; ++o)
                        if (t.nonempty[t.child[(std::size_t)g * 8 + o]]) any = true;
                    t.nonempty[g] = any ? 1 : 0;
                }
    }

    // ---- M2L interaction lists (CSR), exactly FMM3D's rule ----------------
    t.ilistStart.assign(t.nBoxes + 1, 0);
    for (int d = 2; d <= L; ++d) {
        const int ns = 1 << d, nsp = ns / 2;
        for (int iz = 0; iz < ns; ++iz)
            for (int iy = 0; iy < ns; ++iy)
                for (int ix = 0; ix < ns; ++ix) {
                    const int g = t.boxId(d, ix, iy, iz);
                    if (!t.nonempty[g]) continue;
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
                                    if (std::max({std::abs(ax - ix), std::abs(ay - iy), std::abs(az - iz)}) <= 1)
                                        continue;
                                    const int ag = t.boxId(d, ax, ay, az);
                                    if (!t.nonempty[ag]) continue;
                                    t.ilist.push_back(ag);
                                    ++t.ilistStart[g + 1];
                                }
                            }
                        }
                    }
                }
    }
    // Build CSR offsets. ilist was appended in box-global order within each level;
    // since levels are processed in order and within a level boxes in bidx order,
    // and global ids are levelOffset[d]+bidx (monotone), the append order matches
    // ascending g. Convert per-box counts to prefix-sum offsets and reorder if
    // needed. To be safe, rebuild ilist grouped by g using the counts.
    {
        std::vector<int> cnt(t.nBoxes, 0);
        for (int g = 0; g < t.nBoxes; ++g) cnt[g] = t.ilistStart[g + 1];
        for (int g = 0; g < t.nBoxes; ++g) t.ilistStart[g + 1] = t.ilistStart[g] + cnt[g];
        // ilist is already in ascending-g append order, so it lines up with the
        // prefix offsets directly.
    }

    // ---- P2P neighbour lists (CSR over leaves) ----------------------------
    t.nbrStart.assign(t.nLeaves + 1, 0);
    for (int iz = 0; iz < nside; ++iz)
        for (int iy = 0; iy < nside; ++iy)
            for (int ix = 0; ix < nside; ++ix) {
                const int leaf = bidx(ix, iy, iz, nside);
                int c = 0;
                for (int nz = iz - 1; nz <= iz + 1; ++nz) {
                    if (nz < 0 || nz >= nside) continue;
                    for (int ny = iy - 1; ny <= iy + 1; ++ny) {
                        if (ny < 0 || ny >= nside) continue;
                        for (int nx = ix - 1; nx <= ix + 1; ++nx) {
                            if (nx < 0 || nx >= nside) continue;
                            t.nbr.push_back(bidx(nx, ny, nz, nside));
                            ++c;
                        }
                    }
                }
                t.nbrStart[leaf + 1] = c;
            }
    {
        std::vector<int> cnt(t.nLeaves, 0);
        for (int l = 0; l < t.nLeaves; ++l) cnt[l] = t.nbrStart[l + 1];
        for (int l = 0; l < t.nLeaves; ++l) t.nbrStart[l + 1] = t.nbrStart[l] + cnt[l];
    }
    return t;
}

} // namespace gpu
} // namespace galaxy3d
