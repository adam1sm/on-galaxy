#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include "Vec3.hpp" // galaxy3d::Vec3 (do not modify; only used here)

// ============================================================================
// RUNG 4: 3D Laplace FMM expansions and translation operators (MATH ONLY).
//
// SINGLE CONVENTION (Greengard-Rokhlin / Beatson-Greengard "short course").
// Everything below -- harmonics, normalization, and ALL operators -- comes from
// this one source. No mixing.
//
// Spherical harmonics (NO Condon-Shortley phase):
//     Y_n^m(theta,phi) = sqrt( (n-|m|)! / (n+|m|)! ) P_n^|m|(cos theta) e^{i m phi}
// so Y_n^{-m} = conj(Y_n^m). This "FMM normalization" omits the sqrt((2n+1)/4pi)
// of the physics convention, which is exactly what makes the Green's-function
// expansion coefficient-free:
//     1/|r - a| = sum_{n,m} (rho^n / r^{n+1}) Y_n^{-m}(a_dir) Y_n^m(r_dir),  r>rho.
// (Verified analytically for n=0,1: monopole 1/r and dipole (a.r_hat)/r^2.)
//
// Solid harmonics:  regular  R_n^m(r) = r^n     Y_n^m   (local expansions)
//                   singular  S_n^m(r) = Y_n^m / r^{n+1} (multipole expansions)
//
// Moments (P2M):    M_n^m = sum_i q_i rho_i^n Y_n^{-m}(dir_i)
// Multipole eval:   Phi(r) = sum_{n,m} M_n^m Y_n^m(dir)/r^{n+1} = sum M_n^m S_n^m
// Local eval:       Phi(r) = sum_{n,m} L_n^m r^n Y_n^m(dir)    = sum L_n^m R_n^m
//
// Acceleration: with q_i = m_i and Phi the field above, a(r) = G * grad Phi,
// because grad(1/|r-a|) = (a-r)/|r-a|^3, matching DirectSolver3D's
// a_i = G sum_j m_j (r_j - r_i)/|r_j-r_i|^3. Gradients use the solid-harmonic
// derivative recurrences below (a SEPARATE, FD-validated machinery).
//
// Translations are the naive O(p^4) GR1997 operators (A_n^m coefficients).
// NO Wigner/rotation acceleration -- correctness first.
// ============================================================================

namespace galaxy3d {
namespace fmm {

using Complex = std::complex<double>;

// ---- factorials / log-factorials (stable scalar constants) -----------------
inline double logFact(int k) {
    static std::vector<double> lf;
    if (lf.empty()) {
        lf.resize(256);
        lf[0] = 0.0;
        for (int i = 1; i < 256; ++i) lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
    }
    return (k < 0) ? 0.0 : lf[k];
}

// A_n^m = (-1)^n / sqrt((n-m)! (n+m)!), |m| <= n.
inline double Acoef(int n, int m) {
    const double sign = (n & 1) ? -1.0 : 1.0;
    return sign * std::exp(-0.5 * (logFact(n - m) + logFact(n + m)));
}

// i^e for EVEN e (all GR operator exponents are provably even) -> +-1.
inline double jpow(int e) {
    long h = e / 2;            // exact for even e
    long hm = ((h % 2) + 2) % 2;
    return hm == 0 ? 1.0 : -1.0;
}

// Flat (n,m) index: n in [0,L], m in [-n,n].
inline int idx(int n, int m) { return n * n + n + m; }

// ---- spherical harmonics Y_n^m for all (n,m), n<=L, at a vector v ----------
// Fills Y (size (L+1)^2) and returns |v|. Uses the seminormalized associated
// Legendre Q_n^m = sqrt((n-m)!/(n+m)!) P_n^m (stable to moderate L).
inline double computeY(const Vec3& v, int L, std::vector<Complex>& Y) {
    const double r = abs(v);
    Y.assign((L + 1) * (L + 1), Complex{0, 0});
    const double cz = (r > 0) ? v.z / r : 1.0;            // cos theta
    const double sxy = std::sqrt(std::max(0.0, v.x * v.x + v.y * v.y));
    const double sth = (r > 0) ? sxy / r : 0.0;           // sin theta
    const double phi = std::atan2(v.y, v.x);

    // Q[n][m], m = 0..n, via stable seminormalized recurrence.
    std::vector<double> Q((L + 1) * (L + 1), 0.0);
    auto q = [&](int n, int m) -> double& { return Q[n * (L + 1) + m]; };
    q(0, 0) = 1.0;
    for (int m = 1; m <= L; ++m)
        q(m, m) = sth * std::sqrt((2.0 * m - 1.0) / (2.0 * m)) * q(m - 1, m - 1);
    for (int m = 0; m <= L; ++m) {
        if (m + 1 <= L) q(m + 1, m) = cz * std::sqrt(2.0 * m + 1.0) * q(m, m);
        for (int n = m + 2; n <= L; ++n)
            q(n, m) = (cz * (2.0 * n - 1.0) * q(n - 1, m) -
                       std::sqrt(double((n - 1) * (n - 1) - m * m)) * q(n - 2, m)) /
                      std::sqrt(double(n * n - m * m));
    }

    // e^{i m phi} and assembly (Y_n^{-m} = conj(Y_n^m)).
    for (int n = 0; n <= L; ++n) {
        for (int m = 0; m <= n; ++m) {
            const Complex e{std::cos(m * phi), std::sin(m * phi)};
            const double qq = q(n, m);
            Y[idx(n, m)] = qq * e;
            if (m > 0) Y[idx(n, -m)] = qq * std::conj(e);
        }
    }
    return r;
}

// ---- gradient-of-solid-harmonic coefficients (derived for THIS convention) -
// Regular R_n^m -> degree n-1:  d/dz = cz_R, (dx+/-i dy) raise/lower m by +/-1.
inline double cz_R(int n, int m) { return std::sqrt(double(n * n - m * m)); }            // -> R_{n-1}^{m}
inline double cp_R(int n, int m) {                                                       // -> R_{n-1}^{m+1}
    const double s = (m >= 0) ? -1.0 : 1.0;
    return s * std::sqrt(double((n - m) * (n - m - 1)));
}
inline double cm_R(int n, int m) {                                                       // -> R_{n-1}^{m-1}
    const double s = (m > 0) ? 1.0 : -1.0;
    return s * std::sqrt(double((n + m) * (n + m - 1)));
}
// Singular S_n^m -> degree n+1.
inline double dz_S(int n, int m) { return -std::sqrt(double((n + 1) * (n + 1) - m * m)); } // -> S_{n+1}^{m}
inline double dp_S(int n, int m) {                                                         // -> S_{n+1}^{m+1}
    const double s = (m >= 0) ? -1.0 : 1.0;
    return s * std::sqrt(double((n + m + 1) * (n + m + 2)));
}
inline double dm_S(int n, int m) {                                                         // -> S_{n+1}^{m-1}
    const double s = (m > 0) ? 1.0 : -1.0;
    return s * std::sqrt(double((n - m + 1) * (n - m + 2)));
}

// ---- expansion containers --------------------------------------------------
struct Multipole {
    Vec3 center{};
    int p = 0;
    std::vector<Complex> M; // size (p+1)^2, index idx(n,m)
    Multipole() = default;
    Multipole(Vec3 c, int order) : center(c), p(order), M((order + 1) * (order + 1), Complex{0, 0}) {}
};
struct Local {
    Vec3 center{};
    int p = 0;
    std::vector<Complex> L; // size (p+1)^2
    Local() = default;
    Local(Vec3 c, int order) : center(c), p(order), L((order + 1) * (order + 1), Complex{0, 0}) {}
};

// ---- P2M -------------------------------------------------------------------
inline Multipole P2M(Vec3 center, const std::vector<Vec3>& pos,
                     const std::vector<double>& q, int p) {
    Multipole out(center, p);
    std::vector<Complex> Y;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        const Vec3 d = pos[i] - center;
        const double rho = computeY(d, p, Y);
        double rp = 1.0; // rho^n
        for (int n = 0; n <= p; ++n) {
            for (int m = -n; m <= n; ++m)
                out.M[idx(n, m)] += q[i] * rp * Y[idx(n, -m)]; // Y_n^{-m}
            rp *= rho;
        }
    }
    return out;
}

// ---- multipole evaluation: potential and gradient (acceleration field) -----
inline Complex multipolePotential(const Multipole& mp, Vec3 target) {
    std::vector<Complex> Y;
    const double r = computeY(target - mp.center, mp.p, Y);
    Complex phi{0, 0};
    double invr = 1.0 / r;
    double invrn1 = invr; // r^{-(n+1)}
    for (int n = 0; n <= mp.p; ++n) {
        for (int m = -n; m <= n; ++m) phi += mp.M[idx(n, m)] * Y[idx(n, m)] * invrn1;
        invrn1 *= invr;
    }
    return phi;
}

// grad Phi = sum M_n^m grad S_n^m, with grad S_n^m in terms of S_{n+1}^{m'}.
inline Vec3 multipoleField(const Multipole& mp, Vec3 target) {
    std::vector<Complex> Y;
    const double r = computeY(target - mp.center, mp.p + 1, Y); // need degree p+1
    std::vector<double> invr(mp.p + 3);
    invr[0] = 1.0;
    for (int k = 1; k < (int)invr.size(); ++k) invr[k] = invr[k - 1] / r;
    auto S = [&](int n, int m) -> Complex {
        if (n < 0 || std::abs(m) > n) return Complex{0, 0};
        return Y[idx(n, m)] * invr[n + 1];
    };
    Complex gx{0, 0}, gy{0, 0}, gz{0, 0};
    for (int n = 0; n <= mp.p; ++n)
        for (int m = -n; m <= n; ++m) {
            const Complex c = mp.M[idx(n, m)];
            const Complex sp = S(n + 1, m + 1), sm = S(n + 1, m - 1), s0 = S(n + 1, m);
            gx += c * 0.5 * (dp_S(n, m) * sp + dm_S(n, m) * sm);
            gy += c * Complex{0, -0.5} * (dp_S(n, m) * sp - dm_S(n, m) * sm);
            gz += c * dz_S(n, m) * s0;
        }
    return Vec3{gx.real(), gy.real(), gz.real()};
}

// ---- local evaluation: potential and gradient ------------------------------
inline Complex localPotential(const Local& lp, Vec3 target) {
    std::vector<Complex> Y;
    const double r = computeY(target - lp.center, lp.p, Y);
    Complex phi{0, 0};
    double rn = 1.0; // r^n
    for (int n = 0; n <= lp.p; ++n) {
        for (int m = -n; m <= n; ++m) phi += lp.L[idx(n, m)] * Y[idx(n, m)] * rn;
        rn *= r;
    }
    return phi;
}

inline Vec3 localField(const Local& lp, Vec3 target) {
    std::vector<Complex> Y;
    const double r = computeY(target - lp.center, lp.p, Y);
    std::vector<double> rp(lp.p + 2);
    rp[0] = 1.0;
    for (int k = 1; k < (int)rp.size(); ++k) rp[k] = rp[k - 1] * r;
    auto R = [&](int n, int m) -> Complex {
        if (n < 0 || std::abs(m) > n) return Complex{0, 0};
        return Y[idx(n, m)] * rp[n];
    };
    Complex gx{0, 0}, gy{0, 0}, gz{0, 0};
    for (int n = 0; n <= lp.p; ++n)
        for (int m = -n; m <= n; ++m) {
            const Complex c = lp.L[idx(n, m)];
            const Complex rpp = R(n - 1, m + 1), rmm = R(n - 1, m - 1), r0 = R(n - 1, m);
            gx += c * 0.5 * (cp_R(n, m) * rpp + cm_R(n, m) * rmm);
            gy += c * Complex{0, -0.5} * (cp_R(n, m) * rpp - cm_R(n, m) * rmm);
            gz += c * cz_R(n, m) * r0;
        }
    return Vec3{gx.real(), gy.real(), gz.real()};
}

// ---- M2M: shift a multipole from its center to newCenter -------------------
// t = oldCenter - newCenter = (rho,dir). GR1997 Theorem (translation of a
// multipole expansion):
//   M_j^k = sum_{n=0}^{j} sum_{m} O_{j-n}^{k-m} i^{|k|-|m|-|k-m|}
//           A_n^m A_{j-n}^{k-m} rho^n Y_n^{-m}(dir) / A_j^k.
inline Multipole M2M(const Multipole& in, Vec3 newCenter) {
    const int p = in.p;
    Multipole out(newCenter, p);
    std::vector<Complex> Y;
    const double rho = computeY(in.center - newCenter, p, Y);
    std::vector<double> rp(p + 1);
    rp[0] = 1.0;
    for (int n = 1; n <= p; ++n) rp[n] = rp[n - 1] * rho;

    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            Complex acc{0, 0};
            for (int n = 0; n <= j; ++n)
                for (int m = -n; m <= n; ++m) {
                    const int jj = j - n, kk = k - m;
                    if (std::abs(kk) > jj) continue;
                    const double J = jpow(std::abs(k) - std::abs(m) - std::abs(k - m));
                    acc += in.M[idx(jj, kk)] * J * Acoef(n, m) * Acoef(jj, kk) *
                           rp[n] * Y[idx(n, -m)];
                }
            out.M[idx(j, k)] = acc / Acoef(j, k);
        }
    return out;
}

// ---- M2L: multipole (about its center) -> local (about localCenter) --------
// t = localCenter - sourceCenter = (rho,dir). GR1997:
//   L_j^k = sum_{n,m} O_n^m i^{|k-m|-|k|-|m|} A_n^m A_j^k
//           Y_{j+n}^{m-k}(dir) / ( A_{j+n}^{m-k} rho^{j+n+1} ).
// (No explicit (-1)^n: our A_n^m already carries it, and the A powers here
//  cancel to (-1)^0 -- verified by the j=k=0 reduction L_0^0 = Phi(z_L).)
inline Local M2L(const Multipole& in, Vec3 localCenter) {
    const int p = in.p;
    Local out(localCenter, p);
    std::vector<Complex> Y;
    const double rho = computeY(localCenter - in.center, 2 * p, Y);
    std::vector<double> rp(2 * p + 2);
    rp[0] = 1.0;
    for (int n = 1; n < (int)rp.size(); ++n) rp[n] = rp[n - 1] * rho;

    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            Complex acc{0, 0};
            for (int n = 0; n <= p; ++n)
                for (int m = -n; m <= n; ++m) {
                    const int deg = j + n, ord = m - k;
                    if (std::abs(ord) > deg) continue;
                    const double J = jpow(std::abs(k - m) - std::abs(k) - std::abs(m));
                    acc += in.M[idx(n, m)] * J * Acoef(n, m) * Acoef(j, k) *
                           Y[idx(deg, ord)] / (Acoef(deg, ord) * rp[deg + 1]);
                }
            // The local-target A_j^k carries a (-1)^j that does not net out in
            // the multipole->local case (verified coefficient-by-coefficient
            // against the analytic local expansion: op/an = (-1)^j).
            out.L[idx(j, k)] = ((j & 1) ? -1.0 : 1.0) * acc;
        }
    return out;
}

// ---- L2L: shift a local expansion from its center to newCenter -------------
// t = newCenter - oldCenter = (rho,dir). GR1997:
//   L_j^k = sum_{n>=j,m} O_n^m i^{|m|-|m-k|-|k|} A_{n-j}^{m-k} A_j^k
//           Y_{n-j}^{m-k}(dir) rho^{n-j} / A_n^m.
// (Again no explicit (-1)^{n+j}: it is already in our A's, which cancel here;
//  verified by the j=k=0 reduction L_0^0 = Phi_local(z_new).)
inline Local L2L(const Local& in, Vec3 newCenter) {
    const int p = in.p;
    Local out(newCenter, p);
    std::vector<Complex> Y;
    const double rho = computeY(newCenter - in.center, p, Y);
    std::vector<double> rp(p + 1);
    rp[0] = 1.0;
    for (int n = 1; n <= p; ++n) rp[n] = rp[n - 1] * rho;

    for (int j = 0; j <= p; ++j)
        for (int k = -j; k <= j; ++k) {
            Complex acc{0, 0};
            for (int n = j; n <= p; ++n)
                for (int m = -n; m <= n; ++m) {
                    const int nn = n - j, kk = m - k;
                    if (std::abs(kk) > nn) continue;
                    const double J = jpow(std::abs(m) - std::abs(m - k) - std::abs(k));
                    acc += in.L[idx(n, m)] * J * Acoef(nn, kk) * Acoef(j, k) *
                           Y[idx(nn, kk)] * rp[nn] / Acoef(n, m);
                }
            out.L[idx(j, k)] = acc;
        }
    return out;
}

} // namespace fmm
} // namespace galaxy3d
