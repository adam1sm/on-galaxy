#pragma once

#include <thrust/complex.h>

// ============================================================================
// RUNG 7: device-side 3D FMM operators -- faithful transcriptions of the
// VALIDATED CPU operators in src3d/FmmExpansions3D.hpp (same GR convention,
// same formulas, same (-1)^j M2L fix), using thrust::complex and caller-
// provided scratch buffers (no std::vector / std::complex on device).
//
// Templated on `real` (float for the showcase, double for roundoff validation).
// The A_n^m coefficient table is precomputed on the host and passed in (device
// pointer), so no factorials/exp run on device. Gate A (fp64 vs CPU FMM to
// roundoff) is the proof these match the CPU operators.
// ============================================================================

namespace galaxy3d {
namespace gpu {
namespace dev {

template <typename real>
using cplx = thrust::complex<real>;

__device__ __forceinline__ int didx(int n, int m) { return n * n + n + m; }
// Integer abs: avoids resolving to galaxy3d::abs(const Vec3&), which is in scope.
__device__ __forceinline__ int iabs(int x) { return x < 0 ? -x : x; }

// i^e for even e -> +-1 (all GR exponents are even).
__device__ __forceinline__ double jpow(int e) {
    long h = e / 2;
    long hm = ((h % 2) + 2) % 2;
    return hm == 0 ? 1.0 : -1.0;
}

// gradient-of-regular-solid-harmonic coefficients (R_n^m -> degree n-1).
template <typename real> __device__ __forceinline__ real czR(int n, int m) {
    return sqrt((real)(n * n - m * m));
}
template <typename real> __device__ __forceinline__ real cpR(int n, int m) {
    real s = (m >= 0) ? real(-1) : real(1);
    return s * sqrt((real)((n - m) * (n - m - 1)));
}
template <typename real> __device__ __forceinline__ real cmR(int n, int m) {
    real s = (m > 0) ? real(1) : real(-1);
    return s * sqrt((real)((n + m) * (n + m - 1)));
}

// Spherical harmonics Y_n^m for n<=Lmax at vector (vx,vy,vz). Fills Y (size
// (Lmax+1)^2) and uses Q scratch (size (Lmax+1)^2). Returns |v|. Seminormalized
// Legendre recurrence (stable, no factorial overflow). Y_n^{-m}=conj(Y_n^m).
template <typename real>
__device__ real computeY(real vx, real vy, real vz, int Lmax, cplx<real>* Y, real* Q) {
    const real r = sqrt(vx * vx + vy * vy + vz * vz);
    const real cz = (r > 0) ? vz / r : real(1);
    real sxy2 = vx * vx + vy * vy; if (sxy2 < 0) sxy2 = 0;
    const real sxy = sqrt(sxy2);
    const real sth = (r > 0) ? sxy / r : real(0);
    const real phi = atan2(vy, vx);
    const int stride = Lmax + 1;
    const int S = stride * stride;
    for (int i = 0; i < S; ++i) { Y[i] = cplx<real>(0, 0); Q[i] = 0; }

    Q[0] = real(1); // Q[n*stride+m]
    for (int m = 1; m <= Lmax; ++m)
        Q[m * stride + m] = sth * sqrt((real)(2 * m - 1) / (real)(2 * m)) * Q[(m - 1) * stride + (m - 1)];
    for (int m = 0; m <= Lmax; ++m) {
        if (m + 1 <= Lmax) Q[(m + 1) * stride + m] = cz * sqrt((real)(2 * m + 1)) * Q[m * stride + m];
        for (int n = m + 2; n <= Lmax; ++n)
            Q[n * stride + m] = (cz * (real)(2 * n - 1) * Q[(n - 1) * stride + m] -
                                 sqrt((real)((n - 1) * (n - 1) - m * m)) * Q[(n - 2) * stride + m]) /
                                sqrt((real)(n * n - m * m));
    }
    for (int n = 0; n <= Lmax; ++n)
        for (int m = 0; m <= n; ++m) {
            const real cc = cos((real)m * phi), ss = sin((real)m * phi), qq = Q[n * stride + m];
            Y[didx(n, m)] = cplx<real>(qq * cc, qq * ss);
            if (m > 0) Y[didx(n, -m)] = cplx<real>(qq * cc, -qq * ss);
        }
    return r;
}

// ---- P2M: accumulate one particle (charge q at qx,qy,qz) into a leaf's
//      multipole Macc (size (p+1)^2) about (cx,cy,cz). Y,Q sized (p+1)^2. -----
template <typename real>
__device__ void p2mAccum(real qx, real qy, real qz, real q, real cx, real cy, real cz,
                         int p, cplx<real>* Macc, cplx<real>* Y, real* Q) {
    const real rho = computeY<real>(qx - cx, qy - cy, qz - cz, p, Y, Q);
    real rp = real(1);
    for (int n = 0; n <= p; ++n) {
        for (int m = -n; m <= n; ++m) Macc[didx(n, m)] += q * rp * Y[didx(n, -m)];
        rp *= rho;
    }
}

// ---- M2M: add child multipole O (about childC) into parent accumulator Macc
//      (about parentC). A table covers degree up to p. Y,Q sized (p+1)^2. ------
template <typename real>
__device__ void m2m(const cplx<real>* O, real chx, real chy, real chz,
                    real px, real py, real pz, int p, const real* A,
                    cplx<real>* Macc, cplx<real>* Y, real* Q) {
    const real rho = computeY<real>(chx - px, chy - py, chz - pz, p, Y, Q);
    // rho^n, n<=p
    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            cplx<real> acc(0, 0);
            real rp = real(1);
            for (int n = 0; n <= j; ++n) {
                for (int m = -n; m <= n; ++m) {
                    const int jj = j - n, kk = k - m;
                    if (jj < 0 || (kk < -jj) || (kk > jj)) continue;
                    const real J = (real)jpow(iabs(k) - iabs(m) - iabs(k - m));
                    acc += O[didx(jj, kk)] * (J * A[didx(n, m)] * A[didx(jj, kk)] * rp) * Y[didx(n, -m)];
                }
                rp *= rho;
            }
            Macc[didx(j, k)] += acc / A[didx(j, k)];
        }
}

// ---- M2L: add source multipole O (about srcC) into target local Lacc (about
//      tgtC). Needs Y up to degree 2p; Y,Q sized (2p+1)^2. A covers up to 2p. --
template <typename real>
__device__ void m2l(const cplx<real>* O, real sx, real sy, real sz,
                    real tx, real ty, real tz, int p, const real* A,
                    cplx<real>* Lacc, cplx<real>* Y, real* Q) {
    const real rho = computeY<real>(tx - sx, ty - sy, tz - sz, 2 * p, Y, Q);
    // inverse powers of rho up to (2p+1): build rp[deg+1] on the fly via division.
    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            cplx<real> acc(0, 0);
            for (int n = 0; n <= p; ++n)
                for (int m = -n; m <= n; ++m) {
                    const int deg = j + n, ord = m - k;
                    if (ord < -deg || ord > deg) continue;
                    const real J = (real)jpow(iabs(k - m) - iabs(k) - iabs(m));
                    // rho^{deg+1}
                    real rpow = real(1);
                    for (int t = 0; t < deg + 1; ++t) rpow *= rho;
                    acc += O[didx(n, m)] *
                           (J * A[didx(n, m)] * A[didx(j, k)] / (A[didx(deg, ord)] * rpow)) *
                           Y[didx(deg, ord)];
                }
            const real sgn = (j & 1) ? real(-1) : real(1);
            Lacc[didx(j, k)] += sgn * acc;
        }
}

// ---- L2L: add parent local O (about parentC) into child local Lacc (about
//      childC). Y up to degree p; Y,Q sized (p+1)^2. ------------------------
template <typename real>
__device__ void l2l(const cplx<real>* O, real px, real py, real pz,
                    real chx, real chy, real chz, int p, const real* A,
                    cplx<real>* Lacc, cplx<real>* Y, real* Q) {
    const real rho = computeY<real>(chx - px, chy - py, chz - pz, p, Y, Q);
    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            cplx<real> acc(0, 0);
            for (int n = j; n <= p; ++n)
                for (int m = -n; m <= n; ++m) {
                    const int nn = n - j, kk = m - k;
                    if (kk < -nn || kk > nn) continue;
                    const real J = (real)jpow(iabs(m) - iabs(m - k) - iabs(k));
                    real rpow = real(1);
                    for (int t = 0; t < nn; ++t) rpow *= rho;
                    acc += O[didx(n, m)] *
                           (J * A[didx(nn, kk)] * A[didx(j, k)] / A[didx(n, m)] * rpow) *
                           Y[didx(nn, kk)];
                }
            Lacc[didx(j, k)] += acc;
        }
}

// ---- L2P: gradient of the local expansion L (about cx,cy,cz) at (px,py,pz);
//      returns acceleration field (grad of Phi) in (ax,ay,az). Y,Q sized
//      (p+1)^2. (Caller multiplies by G.) ------------------------------------
template <typename real>
__device__ void localField(const cplx<real>* L, real px, real py, real pz,
                           real cx, real cy, real cz, int p, cplx<real>* Y, real* Q,
                           real& ax, real& ay, real& az) {
    const real r = computeY<real>(px - cx, py - cy, pz - cz, p, Y, Q);
    // r^n cache
    cplx<real> gx(0, 0), gy(0, 0), gz(0, 0);
    for (int n = 0; n <= p; ++n)
        for (int m = -n; m <= n; ++m) {
            const cplx<real> c = L[didx(n, m)];
            const int nm1 = n - 1;
            real rpow = real(1);
            for (int t = 0; t < nm1; ++t) rpow *= r; // r^{n-1}
            auto R = [&](int N, int M) -> cplx<real> {
                if (N < 0 || M < -N || M > N) return cplx<real>(0, 0);
                return Y[didx(N, M)] * rpow;
            };
            const cplx<real> rpp = R(nm1, m + 1), rmm = R(nm1, m - 1), r0 = R(nm1, m);
            gx += c * (real(0.5) * (cpR<real>(n, m) * rpp + cmR<real>(n, m) * rmm));
            gy += c * (cplx<real>(0, real(-0.5)) * (cpR<real>(n, m) * rpp - cmR<real>(n, m) * rmm));
            gz += c * (czR<real>(n, m) * r0);
        }
    ax = gx.real(); ay = gy.real(); az = gz.real();
}

} // namespace dev
} // namespace gpu
} // namespace galaxy3d
