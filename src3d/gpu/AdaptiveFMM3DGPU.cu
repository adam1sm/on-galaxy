// ============================================================================
// RUNG 10b: GPU adaptive FMM (CGR). CPU builds the adaptive octree + U/V/W/X
// lists (Rung 10a, unchanged) and flattens them to CSR; the passes run as
// kernels using the Rung 7 device operators (FmmDevice3D.cuh) plus device M2P
// and P2L added here (reusing dev::computeY). FmmDevice3D.cuh is untouched.
// ============================================================================
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../AdaptiveOctree.hpp" // CPU adaptive tree + lists (Rung 10a)
#include "AdaptiveFMM3DGPU.cuh"
#include "FmmDevice3D.cuh" // dev::computeY, didx, iabs, p2mAccum, m2m, m2l, l2l, localField

namespace galaxy3d {
namespace gpu {

#define CK(call)                                                                \
    do { cudaError_t e__ = (call); if (e__ != cudaSuccess) {                    \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                     __LINE__, cudaGetErrorString(e__)); std::exit(1); } } while (0)

template <typename real> using cplx = thrust::complex<real>;

// ---- device M2P (multipole gradient) + P2L, reusing dev:: machinery --------
namespace adev {

template <typename real> __device__ __forceinline__ real dzS(int n, int m) {
    return -sqrt((real)((n + 1) * (n + 1) - m * m));
}
template <typename real> __device__ __forceinline__ real dpS(int n, int m) {
    real s = (m >= 0) ? real(-1) : real(1);
    return s * sqrt((real)((n + m + 1) * (n + m + 2)));
}
template <typename real> __device__ __forceinline__ real dmS(int n, int m) {
    real s = (m > 0) ? real(1) : real(-1);
    return s * sqrt((real)((n - m + 1) * (n - m + 2)));
}

// M2P: gradient of box multipole M (about cx,cy,cz) at (px,py,pz). Adds to (ax,ay,az).
template <typename real>
__device__ void multipoleField(const cplx<real>* M, real px, real py, real pz,
                               real cx, real cy, real cz, int p, cplx<real>* Y, real* Q,
                               real& ax, real& ay, real& az) {
    const real r = dev::computeY<real>(px - cx, py - cy, pz - cz, p + 1, Y, Q);
    cplx<real> gx(0, 0), gy(0, 0), gz(0, 0);
    for (int n = 0; n <= p; ++n)
        for (int m = -n; m <= n; ++m) {
            const cplx<real> c = M[dev::didx(n, m)];
            real ip = real(1); // r^{-(n+2)}
            for (int t = 0; t < n + 2; ++t) ip /= r;
            auto S = [&](int N, int Mm) -> cplx<real> {
                if (N < 0 || dev::iabs(Mm) > N) return cplx<real>(0, 0);
                return Y[dev::didx(N, Mm)] * ip;
            };
            const cplx<real> sp = S(n + 1, m + 1), sm = S(n + 1, m - 1), s0 = S(n + 1, m);
            gx += c * (real(0.5) * (dpS<real>(n, m) * sp + dmS<real>(n, m) * sm));
            gy += c * (cplx<real>(0, real(-0.5)) * (dpS<real>(n, m) * sp - dmS<real>(n, m) * sm));
            gz += c * (dzS<real>(n, m) * s0);
        }
    ax += gx.real(); ay += gy.real(); az += gz.real();
}

// P2L: add a local (about cx,cy,cz) from source particles [first, first+count).
template <typename real>
__device__ void p2lAccum(cplx<real>* Lacc, const real* px, const real* py, const real* pz,
                         const real* mass, int first, int count, real cx, real cy, real cz,
                         int p, cplx<real>* Y, real* Q) {
    for (int k = first; k < first + count; ++k) {
        const real rho = dev::computeY<real>(px[k] - cx, py[k] - cy, pz[k] - cz, p, Y, Q);
        real inv = real(1) / rho;
        for (int n = 0; n <= p; ++n) {
            for (int m = -n; m <= n; ++m) Lacc[dev::didx(n, m)] += mass[k] * Y[dev::didx(n, -m)] * inv;
            inv /= rho;
        }
    }
}

} // namespace adev

// ---- kernels (templated on real and compile-time order P) -----------------

template <typename real, int P>
__global__ void adaptP2mKernel(const real* px, const real* py, const real* pz, const real* pm,
                          const int* leafBox, const int* pfirst, const int* pcount,
                          const real* cx, const real* cy, const real* cz, cplx<real>* M,
                          int nLeaves, int p) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nLeaves) return;
    const int g = leafBox[l];
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(P + 1) * (P + 1)]; real Q[(P + 1) * (P + 1)];
    cplx<real>* Macc = M + (long)g * S;
    const int first = pfirst[g], cnt = pcount[g];
    for (int t = 0; t < cnt; ++t)
        dev::p2mAccum<real>(px[first + t], py[first + t], pz[first + t], pm[first + t],
                            cx[g], cy[g], cz[g], p, Macc, Y, Q);
}

template <typename real, int P>
__global__ void adaptM2mKernel(const int* levelBox, int loIdx, int count, const int* child,
                          const char* leaf, const real* cx, const real* cy, const real* cz,
                          const real* A, cplx<real>* M, int p) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int g = levelBox[loIdx + t];
    if (leaf[g]) return;
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(P + 1) * (P + 1)]; real Q[(P + 1) * (P + 1)];
    cplx<real>* Macc = M + (long)g * S;
    for (int o = 0; o < 8; ++o) {
        const int ch = child[(long)g * 8 + o];
        if (ch < 0) continue;
        dev::m2m<real>(M + (long)ch * S, cx[ch], cy[ch], cz[ch], cx[g], cy[g], cz[g], p, A, Macc, Y, Q);
    }
}

template <typename real, int P>
__global__ void adaptDownwardKernel(const int* levelBox, int loIdx, int count, const int* parent,
                               const int* Voff, const int* Vidx, const int* Xoff, const int* Xidx,
                               const int* pfirst, const int* pcount, const real* px, const real* py,
                               const real* pz, const real* pm, const real* cx, const real* cy,
                               const real* cz, const real* A, const cplx<real>* M, cplx<real>* L, int p) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int g = levelBox[loIdx + t];
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(2 * P + 1) * (2 * P + 1)]; real Q[(2 * P + 1) * (2 * P + 1)];
    cplx<real>* Lacc = L + (long)g * S;
    if (parent[g] >= 0) {
        const int pg = parent[g];
        dev::l2l<real>(L + (long)pg * S, cx[pg], cy[pg], cz[pg], cx[g], cy[g], cz[g], p, A, Lacc, Y, Q);
    }
    for (int e = Voff[g]; e < Voff[g + 1]; ++e) {
        const int c = Vidx[e];
        dev::m2l<real>(M + (long)c * S, cx[c], cy[c], cz[c], cx[g], cy[g], cz[g], p, A, Lacc, Y, Q);
    }
    for (int e = Xoff[g]; e < Xoff[g + 1]; ++e) {
        const int c = Xidx[e]; // leaf source box
        adev::p2lAccum<real>(Lacc, px, py, pz, pm, pfirst[c], pcount[c], cx[g], cy[g], cz[g], p, Y, Q);
    }
}

template <typename real, int P>
__global__ void adaptLeafEvalKernel(const real* px, const real* py, const real* pz, const real* pm,
                               const int* partBox, const int* Woff, const int* Widx,
                               const int* Uoff, const int* Uidx, const int* pfirst, const int* pcount,
                               const real* cx, const real* cy, const real* cz, const cplx<real>* M,
                               const cplx<real>* L, real* ax, real* ay, real* az, int n, int p,
                               real eps2, real G) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = partBox[i];
    const int S = (P + 1) * (P + 1);
    cplx<real> Y[(2 * P + 1) * (2 * P + 1)]; real Q[(2 * P + 1) * (2 * P + 1)];
    const real xi = px[i], yi = py[i], zi = pz[i];

    // far: local expansion gradient (L2P) + W-list M2P
    real fx = 0, fy = 0, fz = 0;
    dev::localField<real>(L + (long)b * S, xi, yi, zi, cx[b], cy[b], cz[b], p, Y, Q, fx, fy, fz);
    for (int e = Woff[b]; e < Woff[b + 1]; ++e) {
        const int c = Widx[e];
        adev::multipoleField<real>(M + (long)c * S, xi, yi, zi, cx[c], cy[c], cz[c], p, Y, Q, fx, fy, fz);
    }
    // near: U-list P2P (softened)
    real nx = 0, ny = 0, nz = 0;
    for (int e = Uoff[b]; e < Uoff[b + 1]; ++e) {
        const int c = Uidx[e];
        const int first = pfirst[c], cnt = pcount[c];
        for (int t = 0; t < cnt; ++t) {
            const int j = first + t;
            if (j == i) continue;
            const real dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            const real r2 = dx * dx + dy * dy + dz * dz + eps2;
            const real f = pm[j] * rsqrt(r2) / r2;
            nx += f * dx; ny += f * dy; nz += f * dz;
        }
    }
    ax[i] = G * (fx + nx); ay[i] = G * (fy + ny); az[i] = G * (fz + nz);
}

// ---- host driver ----------------------------------------------------------
namespace {

template <typename real>
std::vector<real> buildAtable(int maxDeg) {
    std::vector<double> lf(2 * maxDeg + 4, 0.0);
    for (int i = 1; i < (int)lf.size(); ++i) lf[i] = lf[i - 1] + std::log((double)i);
    const int S = (maxDeg + 1) * (maxDeg + 1);
    std::vector<real> A(S, real(0));
    for (int n = 0; n <= maxDeg; ++n)
        for (int m = -n; m <= n; ++m)
            A[n * n + n + m] = (real)(((n & 1) ? -1.0 : 1.0) * std::exp(-0.5 * (lf[n - m] + lf[n + m])));
    return A;
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr; if (h.empty()) { CK(cudaMalloc(&d, sizeof(T))); return d; }
    CK(cudaMalloc(&d, sizeof(T) * h.size()));
    CK(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
    return d;
}
template <typename real> std::vector<real> castv(const std::vector<double>& v) {
    std::vector<real> o(v.size()); for (std::size_t i = 0; i < v.size(); ++i) o[i] = (real)v[i]; return o;
}
// flatten vector<vector<int>> -> CSR (offsets size nb+1, idx flat)
void toCSR(const std::vector<std::vector<int>>& lists, std::vector<int>& off, std::vector<int>& idx) {
    const int nb = (int)lists.size();
    off.assign(nb + 1, 0);
    for (int b = 0; b < nb; ++b) off[b + 1] = off[b] + (int)lists[b].size();
    idx.clear(); idx.reserve(off[nb]);
    for (int b = 0; b < nb; ++b) for (int c : lists[b]) idx.push_back(c);
}

template <typename real, int P>
Accelerations3D runAdaptive(const AdaptiveOctree& t, double G, double eps, int p, GpuTiming* timing) {
    const std::size_t n = t.px.size();
    const int nb = (int)t.box.size();
    const int S = (P + 1) * (P + 1);

    auto t0 = std::chrono::steady_clock::now();
    // per-box flat arrays
    std::vector<double> bcx(nb), bcy(nb), bcz(nb);
    std::vector<int> blevel(nb), bparent(nb), bpfirst(nb), bpcount(nb), bchild((std::size_t)nb * 8);
    std::vector<char> bleaf(nb);
    int Lmax = 0;
    for (int b = 0; b < nb; ++b) {
        bcx[b] = t.box[b].cx; bcy[b] = t.box[b].cy; bcz[b] = t.box[b].cz;
        blevel[b] = t.box[b].level; bparent[b] = t.box[b].parent;
        bpfirst[b] = t.box[b].pfirst; bpcount[b] = t.box[b].pcount; bleaf[b] = t.box[b].leaf ? 1 : 0;
        for (int o = 0; o < 8; ++o) bchild[(std::size_t)b * 8 + o] = t.box[b].child[o];
        Lmax = std::max(Lmax, blevel[b]);
    }
    // level CSR
    std::vector<int> levelOff(Lmax + 2, 0), levelBox(nb);
    for (int b = 0; b < nb; ++b) ++levelOff[blevel[b] + 1];
    for (int d = 0; d <= Lmax; ++d) levelOff[d + 1] += levelOff[d];
    { std::vector<int> cur(levelOff.begin(), levelOff.end() - 1);
      for (int b = 0; b < nb; ++b) levelBox[cur[blevel[b]]++] = b; }
    // leaves + partBox
    std::vector<int> leafBox = t.leaves;
    std::vector<int> partBox(n);
    for (int l : t.leaves) for (int k = t.box[l].pfirst; k < t.box[l].pfirst + t.box[l].pcount; ++k) partBox[k] = l;
    // CSR lists
    std::vector<int> Uoff, Uidx, Voff, Vidx, Woff, Widx, Xoff, Xidx;
    toCSR(t.U, Uoff, Uidx); toCSR(t.V, Voff, Vidx); toCSR(t.W, Woff, Widx); toCSR(t.X, Xoff, Xidx);

    // upload
    real* dpx = up(castv<real>(t.px)); real* dpy = up(castv<real>(t.py));
    real* dpz = up(castv<real>(t.pz)); real* dpm = up(castv<real>(t.mass));
    real* dcx = up(castv<real>(bcx)); real* dcy = up(castv<real>(bcy)); real* dcz = up(castv<real>(bcz));
    real* dA = up(buildAtable<real>(2 * p));
    int* dchild = up(bchild); int* dparent = up(bparent);
    int* dpfirst = up(bpfirst); int* dpcount = up(bpcount); char* dleaf = up(bleaf);
    int* dlevelBox = up(levelBox); int* dleafBox = up(leafBox); int* dpartBox = up(partBox);
    int* dUoff = up(Uoff); int* dUidx = up(Uidx); int* dVoff = up(Voff); int* dVidx = up(Vidx);
    int* dWoff = up(Woff); int* dWidx = up(Widx); int* dXoff = up(Xoff); int* dXidx = up(Xidx);
    cplx<real>*dM, *dL; CK(cudaMalloc(&dM, sizeof(cplx<real>) * (long)nb * S));
    CK(cudaMalloc(&dL, sizeof(cplx<real>) * (long)nb * S));
    CK(cudaMemset(dM, 0, sizeof(cplx<real>) * (long)nb * S));
    CK(cudaMemset(dL, 0, sizeof(cplx<real>) * (long)nb * S));
    real *dax, *day, *daz;
    CK(cudaMalloc(&dax, sizeof(real) * n)); CK(cudaMalloc(&day, sizeof(real) * n)); CK(cudaMalloc(&daz, sizeof(real) * n));
    CK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    const int blk = 128;
    auto grid = [&](int c) { return (c + blk - 1) / blk; };
    // upward
    adaptP2mKernel<real, P><<<grid((int)leafBox.size()), blk>>>(dpx, dpy, dpz, dpm, dleafBox, dpfirst, dpcount,
                                                           dcx, dcy, dcz, dM, (int)leafBox.size(), p);
    for (int d = Lmax - 1; d >= 0; --d) {
        const int cnt = levelOff[d + 1] - levelOff[d];
        if (cnt > 0) adaptM2mKernel<real, P><<<grid(cnt), blk>>>(dlevelBox, levelOff[d], cnt, dchild, dleaf,
                                                            dcx, dcy, dcz, dA, dM, p);
    }
    // downward (top-down)
    for (int d = 0; d <= Lmax; ++d) {
        const int cnt = levelOff[d + 1] - levelOff[d];
        if (cnt > 0) adaptDownwardKernel<real, P><<<grid(cnt), blk>>>(dlevelBox, levelOff[d], cnt, dparent,
            dVoff, dVidx, dXoff, dXidx, dpfirst, dpcount, dpx, dpy, dpz, dpm, dcx, dcy, dcz, dA, dM, dL, p);
    }
    // leaves
    adaptLeafEvalKernel<real, P><<<grid((int)n), blk>>>(dpx, dpy, dpz, dpm, dpartBox, dWoff, dWidx, dUoff, dUidx,
        dpfirst, dpcount, dcx, dcy, dcz, dM, dL, dax, day, daz, (int)n, p, (real)(eps * eps), (real)G);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();

    std::vector<real> hax(n), hay(n), haz(n);
    CK(cudaMemcpy(hax.data(), dax, sizeof(real) * n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hay.data(), day, sizeof(real) * n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(haz.data(), daz, sizeof(real) * n, cudaMemcpyDeviceToHost));
    Accelerations3D acc(n);
    for (std::size_t i = 0; i < n; ++i)
        acc[t.origIndex[i]] = Vec3{(double)hax[i], (double)hay[i], (double)haz[i]};

    for (void* p2 : {(void*)dpx, (void*)dpy, (void*)dpz, (void*)dpm, (void*)dcx, (void*)dcy, (void*)dcz,
                     (void*)dA, (void*)dchild, (void*)dparent, (void*)dpfirst, (void*)dpcount, (void*)dleaf,
                     (void*)dlevelBox, (void*)dleafBox, (void*)dpartBox, (void*)dUoff, (void*)dUidx,
                     (void*)dVoff, (void*)dVidx, (void*)dWoff, (void*)dWidx, (void*)dXoff, (void*)dXidx,
                     (void*)dM, (void*)dL, (void*)dax, (void*)day, (void*)daz})
        cudaFree(p2);
    if (timing) { timing->uploadSec = std::chrono::duration<double>(t1 - t0).count();
                  timing->computeSec = std::chrono::duration<double>(t2 - t1).count();
                  timing->nBoxes = nb; }
    return acc;
}

} // namespace

template <typename real>
Accelerations3D adaptiveFmm3dGpu(const Bodies3D& bodies, double G, double eps, int p,
                                 int threshold, GpuTiming* timing) {
    auto tb0 = std::chrono::steady_clock::now();
    AdaptiveOctree t; t.build(bodies, threshold, eps); t.buildLists(); // eps floor (Rung 12)
    auto tb1 = std::chrono::steady_clock::now();
    Accelerations3D acc;
    switch (p) {
        case 2: acc = runAdaptive<real, 2>(t, G, eps, p, timing); break;
        case 4: acc = runAdaptive<real, 4>(t, G, eps, p, timing); break;
        case 6: acc = runAdaptive<real, 6>(t, G, eps, p, timing); break;
        case 8: acc = runAdaptive<real, 8>(t, G, eps, p, timing); break;
        default: std::fprintf(stderr, "adaptiveFmm3dGpu: p must be 2,4,6,8\n"); std::exit(2);
    }
    if (timing) { timing->treeBuildSec = std::chrono::duration<double>(tb1 - tb0).count();
                  timing->depth = -1; timing->nBoxes = (int)t.box.size();
                  timing->maxLeafOcc = t.maxLeafOcc(); }
    return acc;
}

template Accelerations3D adaptiveFmm3dGpu<float>(const Bodies3D&, double, double, int, int, GpuTiming*);
template Accelerations3D adaptiveFmm3dGpu<double>(const Bodies3D&, double, double, int, int, GpuTiming*);

} // namespace gpu
} // namespace galaxy3d
