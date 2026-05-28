// ============================================================================
// RUNG 7: GPU FMM kernels + host driver. Faithful GPU port of the validated CPU
// FMM3D (Rung 5) using the device operators in FmmDevice3D.cuh. Uniform octree
// built CPU-side (GpuOctree.hpp), passes run on GPU. fp32 (showcase) / fp64
// (roundoff validation) via the `real` template.
// ============================================================================
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "FMM3DGPU.cuh"
#include "FmmDevice3D.cuh"
#include "GpuOctree.hpp"

namespace galaxy3d {
namespace gpu {

#define CK(call)                                                                \
    do { cudaError_t e__ = (call); if (e__ != cudaSuccess) {                    \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                     __LINE__, cudaGetErrorString(e__)); std::exit(1); } } while (0)

template <typename real> using cplx = thrust::complex<real>;

// ---- kernels (templated on real and compile-time order P) -----------------

template <typename real, int P>
__global__ void p2mKernel(const real* px, const real* py, const real* pz, const real* pm,
                          const int* leafFirst, const int* leafCount, const int* leafBox,
                          const real* cx, const real* cy, const real* cz,
                          cplx<real>* M, int nLeaves, int p) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nLeaves) return;
    const int cnt = leafCount[l];
    if (cnt == 0) return;
    const int g = leafBox[l];
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(P + 1) * (P + 1)]; real Q[(P + 1) * (P + 1)];
    cplx<real>* Macc = M + (long)g * S;
    const int first = leafFirst[l];
    for (int t = 0; t < cnt; ++t) {
        const int i = first + t;
        dev::p2mAccum<real>(px[i], py[i], pz[i], pm[i], cx[g], cy[g], cz[g], p, Macc, Y, Q);
    }
}

template <typename real, int P>
__global__ void m2mKernel(const int* child, const real* cx, const real* cy, const real* cz,
                          const char* nonempty, const real* A, cplx<real>* M,
                          int levelOff, int count, int p) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int g = levelOff + t;
    if (!nonempty[g]) return;
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(P + 1) * (P + 1)]; real Q[(P + 1) * (P + 1)];
    cplx<real>* Macc = M + (long)g * S;
    for (int o = 0; o < 8; ++o) {
        const int cg = child[(long)g * 8 + o];
        if (cg < 0 || !nonempty[cg]) continue;
        dev::m2m<real>(M + (long)cg * S, cx[cg], cy[cg], cz[cg], cx[g], cy[g], cz[g], p, A, Macc, Y, Q);
    }
}

template <typename real, int P>
__global__ void downwardKernel(const int* parent, const int* ilistStart, const int* ilist,
                               const real* cx, const real* cy, const real* cz,
                               const char* nonempty, const real* A,
                               const cplx<real>* M, cplx<real>* L,
                               int levelOff, int count, int p, int applyL2L) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int g = levelOff + t;
    if (!nonempty[g]) return;
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(2 * P + 1) * (2 * P + 1)]; real Q[(2 * P + 1) * (2 * P + 1)];
    cplx<real>* Lacc = L + (long)g * S;
    if (applyL2L) {
        const int pg = parent[g];
        dev::l2l<real>(L + (long)pg * S, cx[pg], cy[pg], cz[pg], cx[g], cy[g], cz[g], p, A, Lacc, Y, Q);
    }
    for (int e = ilistStart[g]; e < ilistStart[g + 1]; ++e) {
        const int s = ilist[e];
        dev::m2l<real>(M + (long)s * S, cx[s], cy[s], cz[s], cx[g], cy[g], cz[g], p, A, Lacc, Y, Q);
    }
}

template <typename real, int P>
__global__ void l2pP2pKernel(const real* px, const real* py, const real* pz, const real* pm,
                             const int* partLeaf, const int* leafBox, const int* leafFirst,
                             const int* leafCount, const int* nbrStart, const int* nbr,
                             const real* cx, const real* cy, const real* cz,
                             const cplx<real>* L, real* ax, real* ay, real* az,
                             int n, int p, real eps2, real G) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int leaf = partLeaf[i];
    const int g = leafBox[leaf];
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(P + 1) * (P + 1)]; real Q[(P + 1) * (P + 1)];
    const real xi = px[i], yi = py[i], zi = pz[i];

    // far field (gradient of local expansion)
    real fx = 0, fy = 0, fz = 0;
    dev::localField<real>(L + (long)g * S, xi, yi, zi, cx[g], cy[g], cz[g], p, Y, Q, fx, fy, fz);

    // near field (P2P over neighbour leaves)
    real nx = 0, ny = 0, nz = 0;
    for (int e = nbrStart[leaf]; e < nbrStart[leaf + 1]; ++e) {
        const int nl = nbr[e];
        const int first = leafFirst[nl], cnt = leafCount[nl];
        for (int t = 0; t < cnt; ++t) {
            const int j = first + t;
            if (j == i) continue;
            const real dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            const real r2 = dx * dx + dy * dy + dz * dz + eps2;
            const real inv3 = rsqrt(r2) / r2;
            const real f = pm[j] * inv3;
            nx += f * dx; ny += f * dy; nz += f * dz;
        }
    }
    ax[i] = G * (fx + nx); ay[i] = G * (fy + ny); az[i] = G * (fz + nz);
}

// ---- host driver ----------------------------------------------------------

namespace {

// Precompute A_n^m = (-1)^n / sqrt((n-m)!(n+m)!) for n=0..maxDeg, flat idx(n,m).
template <typename real>
std::vector<real> buildAtable(int maxDeg) {
    std::vector<double> lf(2 * maxDeg + 4, 0.0);
    for (int i = 1; i < (int)lf.size(); ++i) lf[i] = lf[i - 1] + std::log((double)i);
    const int S = (maxDeg + 1) * (maxDeg + 1);
    std::vector<real> A(S, real(0));
    for (int n = 0; n <= maxDeg; ++n)
        for (int m = -n; m <= n; ++m) {
            const double sign = (n & 1) ? -1.0 : 1.0;
            A[n * n + n + m] = (real)(sign * std::exp(-0.5 * (lf[n - m] + lf[n + m])));
        }
    return A;
}

template <typename T> T* devUpload(const std::vector<T>& h) {
    T* d = nullptr;
    if (h.empty()) return d;
    CK(cudaMalloc(&d, sizeof(T) * h.size()));
    CK(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
    return d;
}
template <typename real>
std::vector<real> castVec(const std::vector<double>& v) {
    std::vector<real> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = (real)v[i];
    return o;
}

template <typename real, int P>
Accelerations3D runFMM(const GpuOctree& t, const Bodies3D& bodies, double G, double eps,
                       int p, GpuTiming* timing) {
    const std::size_t n = bodies.size();
    const int S = (P + 1) * (P + 1);

    auto t0 = std::chrono::steady_clock::now();
    // host A table (degree up to 2p), particle->leaf map
    std::vector<real> A = buildAtable<real>(2 * p);
    std::vector<int> partLeaf(n);
    for (int l = 0; l < t.nLeaves; ++l)
        for (int k = 0; k < t.leafCount[l]; ++k) partLeaf[t.leafFirst[l] + k] = l;

    // upload tree + particles
    real* dpx = devUpload(castVec<real>(t.px));
    real* dpy = devUpload(castVec<real>(t.py));
    real* dpz = devUpload(castVec<real>(t.pz));
    real* dpm = devUpload(castVec<real>(t.pmass));
    real* dcx = devUpload(castVec<real>(t.cx));
    real* dcy = devUpload(castVec<real>(t.cy));
    real* dcz = devUpload(castVec<real>(t.cz));
    real* dA = devUpload(A);
    int* dChild = devUpload(t.child);
    int* dParent = devUpload(t.parent);
    char* dNon = devUpload(t.nonempty);
    int* dIlS = devUpload(t.ilistStart);
    int* dIl = devUpload(t.ilist);
    int* dLeafFirst = devUpload(t.leafFirst);
    int* dLeafCount = devUpload(t.leafCount);
    int* dLeafBox = devUpload(t.leafBox);
    int* dNbrS = devUpload(t.nbrStart);
    int* dNbr = devUpload(t.nbr);
    int* dPartLeaf = devUpload(partLeaf);

    cplx<real>*dM = nullptr, *dL = nullptr;
    CK(cudaMalloc(&dM, sizeof(cplx<real>) * (long)t.nBoxes * S));
    CK(cudaMalloc(&dL, sizeof(cplx<real>) * (long)t.nBoxes * S));
    CK(cudaMemset(dM, 0, sizeof(cplx<real>) * (long)t.nBoxes * S));
    CK(cudaMemset(dL, 0, sizeof(cplx<real>) * (long)t.nBoxes * S));
    real *dax, *day, *daz;
    CK(cudaMalloc(&dax, sizeof(real) * n)); CK(cudaMalloc(&day, sizeof(real) * n));
    CK(cudaMalloc(&daz, sizeof(real) * n));
    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    const int blk = 128;
    auto grid = [&](int c) { return (c + blk - 1) / blk; };
    // --- upward: P2M then M2M up ---
    p2mKernel<real, P><<<grid(t.nLeaves), blk>>>(dpx, dpy, dpz, dpm, dLeafFirst, dLeafCount,
                                                 dLeafBox, dcx, dcy, dcz, dM, t.nLeaves, p);
    for (int d = t.L - 1; d >= 0; --d) {
        const int cnt = 1 << (3 * d);
        m2mKernel<real, P><<<grid(cnt), blk>>>(dChild, dcx, dcy, dcz, dNon, dA, dM,
                                               t.levelOffset[d], cnt, p);
    }
    // --- downward: M2L + L2L, level by level ---
    for (int d = 2; d <= t.L; ++d) {
        const int cnt = 1 << (3 * d);
        downwardKernel<real, P><<<grid(cnt), blk>>>(dParent, dIlS, dIl, dcx, dcy, dcz, dNon, dA,
                                                    dM, dL, t.levelOffset[d], cnt, p, d >= 3 ? 1 : 0);
    }
    // --- leaves: L2P + P2P ---
    l2pP2pKernel<real, P><<<grid((int)n), blk>>>(dpx, dpy, dpz, dpm, dPartLeaf, dLeafBox,
                                                 dLeafFirst, dLeafCount, dNbrS, dNbr, dcx, dcy, dcz,
                                                 dL, dax, day, daz, (int)n, p, (real)(eps * eps), (real)G);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();

    // download (leaf order) + scatter to original order
    std::vector<real> hax(n), hay(n), haz(n);
    CK(cudaMemcpy(hax.data(), dax, sizeof(real) * n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hay.data(), day, sizeof(real) * n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(haz.data(), daz, sizeof(real) * n, cudaMemcpyDeviceToHost));
    Accelerations3D acc(n);
    for (std::size_t i = 0; i < n; ++i)
        acc[t.origIndex[i]] = Vec3{(double)hax[i], (double)hay[i], (double)haz[i]};

    for (void* p2 : {(void*)dpx, (void*)dpy, (void*)dpz, (void*)dpm, (void*)dcx, (void*)dcy,
                     (void*)dcz, (void*)dA, (void*)dChild, (void*)dParent, (void*)dNon, (void*)dIlS,
                     (void*)dIl, (void*)dLeafFirst, (void*)dLeafCount, (void*)dLeafBox, (void*)dNbrS,
                     (void*)dNbr, (void*)dPartLeaf, (void*)dM, (void*)dL, (void*)dax, (void*)day,
                     (void*)daz})
        cudaFree(p2);

    if (timing) {
        timing->uploadSec = std::chrono::duration<double>(t1 - t0).count();
        timing->computeSec = std::chrono::duration<double>(t2 - t1).count();
        timing->depth = t.L;
        timing->nBoxes = t.nBoxes;
    }
    return acc;
}

} // namespace

template <typename real>
Accelerations3D fmm3dGpu(const Bodies3D& bodies, double G, double eps, int p, int depth,
                         int occTarget, GpuTiming* timing) {
    auto tb0 = std::chrono::steady_clock::now();
    const int L = depth >= 0 ? depth : gpuAutoDepth(bodies.size(), occTarget);
    GpuOctree tree = buildGpuOctree(bodies, L);
    auto tb1 = std::chrono::steady_clock::now();
    const double treeSec = std::chrono::duration<double>(tb1 - tb0).count();

    Accelerations3D acc;
    switch (p) {
        case 2: acc = runFMM<real, 2>(tree, bodies, G, eps, p, timing); break;
        case 4: acc = runFMM<real, 4>(tree, bodies, G, eps, p, timing); break;
        case 6: acc = runFMM<real, 6>(tree, bodies, G, eps, p, timing); break;
        case 8: acc = runFMM<real, 8>(tree, bodies, G, eps, p, timing); break;
        default:
            std::fprintf(stderr, "fmm3dGpu: unsupported p=%d (use 2,4,6,8)\n", p);
            std::exit(2);
    }
    if (timing) timing->treeBuildSec = treeSec;
    return acc;
}

// explicit instantiations
template Accelerations3D fmm3dGpu<float>(const Bodies3D&, double, double, int, int, int, GpuTiming*);
template Accelerations3D fmm3dGpu<double>(const Bodies3D&, double, double, int, int, int, GpuTiming*);

} // namespace galaxy3d
} // namespace galaxy3d
