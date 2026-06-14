#pragma once

#include <cmath>
#include <random>
#include <stdexcept>

#include "Types3D.hpp"

namespace galaxy3d {

// ---------------------------------------------------------------------------
// (a) Two-body CIRCULAR orbit for the 3D 1/r^2 (Plummer-softened) kernel.
//
// This kernel IS Keplerian. Balancing centripetal and gravitational
// acceleration for body 1 (at COM radius r1 = D*m2/M):
//   omega^2 * r1 = G*m2*D / (D^2+eps^2)^(3/2)
// => omega = sqrt( G*M / (D^2 + eps^2)^(3/2) ),   M = m1 + m2.
// For eps -> 0 this is omega^2 = G*M/D^3, i.e. Kepler's third law
//   T^2 = 4*pi^2 * D^3 / (G*M).
//
// The orbit is built in the xy-plane and then TILTED by a fixed rotation so the
// angular-momentum vector has all three components nonzero -- a genuine 3D test
// (Lx, Ly, Lz must each be conserved). Velocities are perpendicular and opposite
// so total linear momentum is exactly zero.
// ---------------------------------------------------------------------------
struct TwoBodyConfig3D {
    double m1 = 1.0;
    double m2 = 1.0;
    double separation = 2.0; // D
    double G = 1.0;
    double eps = 0.0;        // default 0 so the orbit is exactly Keplerian
    double tilt_x = 0.6;     // tilt angles (radians) applied to the orbital plane
    double tilt_y = 0.9;
};

namespace detail {
inline Vec3 rotateX(const Vec3& v, double a) {
    const double c = std::cos(a), s = std::sin(a);
    return Vec3{v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}
inline Vec3 rotateY(const Vec3& v, double a) {
    const double c = std::cos(a), s = std::sin(a);
    return Vec3{c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}
inline Vec3 tilt(const Vec3& v, double ax, double ay) {
    return rotateY(rotateX(v, ax), ay);
}
} // namespace detail

inline double twoBodyOmega(const TwoBodyConfig3D& c) {
    const double M = c.m1 + c.m2;
    const double d2 = c.separation * c.separation + c.eps * c.eps;
    return std::sqrt(c.G * M / (d2 * std::sqrt(d2))); // (d2)^(3/2)
}

inline double twoBodyPeriod(const TwoBodyConfig3D& c) {
    return 2.0 * M_PI / twoBodyOmega(c);
}

inline Bodies3D makeTwoBodyCircular(const TwoBodyConfig3D& c) {
    Bodies3D b;
    b.resize(2);

    const double M = c.m1 + c.m2;
    const double r1 = c.separation * c.m2 / M;
    const double r2 = c.separation * c.m1 / M;
    const double omega = twoBodyOmega(c);
    const double v1 = omega * r1, v2 = omega * r2;

    // In-plane (xy) configuration, COM at origin, then tilted into 3D.
    b.mass[0] = c.m1;
    b.mass[1] = c.m2;
    b.pos[0] = detail::tilt(Vec3{-r1, 0.0, 0.0}, c.tilt_x, c.tilt_y);
    b.pos[1] = detail::tilt(Vec3{+r2, 0.0, 0.0}, c.tilt_x, c.tilt_y);
    b.vel[0] = detail::tilt(Vec3{0.0, -v1, 0.0}, c.tilt_x, c.tilt_y);
    b.vel[1] = detail::tilt(Vec3{0.0, +v2, 0.0}, c.tilt_x, c.tilt_y);
    return b;
}

// ---------------------------------------------------------------------------
// (b) 3D cluster: an equilibrium PLUMMER sphere (Aarseth/Henon sampling).
//
// Units G = total_mass = 1 are conventional; we keep them configurable. With
// Plummer scale a:
//   - enclosed-mass fraction u = r^3/(r^2+a^2)^(3/2)  =>  r = a / sqrt(u^(-2/3) - 1)
//   - escape speed v_esc(r) = sqrt(2 G M) (r^2 + a^2)^(-1/4)
//   - speed fraction q sampled by rejection from g(q) = q^2 (1-q^2)^(7/2); v = q*v_esc
// Net momentum and COM are removed so the momentum/angular-momentum diagnostics
// start clean. Being near virial equilibrium, the cluster stays compact and the
// energy oscillation is small and bounded under leapfrog.
// ---------------------------------------------------------------------------
struct PlummerConfig {
    std::size_t N = 1000;
    double a = 1.0;          // Plummer scale radius
    double total_mass = 1.0; // distributed equally over N bodies
    double G = 1.0;
    unsigned seed = 7;
};

inline Bodies3D makePlummerSphere(const PlummerConfig& c) {
    if (c.N == 0) throw std::invalid_argument("Plummer N must be > 0");
    Bodies3D b;
    b.resize(c.N);

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double m = c.total_mass / static_cast<double>(c.N);
    const double vfac = std::sqrt(2.0 * c.G * c.total_mass);

    auto randomDirection = [&]() {
        const double cz = 2.0 * uni(rng) - 1.0;       // cos(theta) in [-1,1]
        const double sz = std::sqrt(std::max(0.0, 1.0 - cz * cz));
        const double phi = 2.0 * M_PI * uni(rng);
        return Vec3{sz * std::cos(phi), sz * std::sin(phi), cz};
    };

    for (std::size_t i = 0; i < c.N; ++i) {
        // Radius from inverse-CDF of the enclosed Plummer mass.
        double u = uni(rng);
        if (u < 1e-12) u = 1e-12; // avoid r -> 0 degeneracy
        const double r = c.a / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);

        // Speed fraction q via rejection sampling of q^2 (1-q^2)^(7/2).
        double q, g;
        do {
            q = uni(rng);
            g = 0.1 * uni(rng); // 0.1 bounds the peak of q^2 (1-q^2)^3.5
        } while (g > q * q * std::pow(1.0 - q * q, 3.5));

        const double v_esc = vfac * std::pow(r * r + c.a * c.a, -0.25);

        b.mass[i] = m;
        b.pos[i] = r * randomDirection();
        b.vel[i] = (q * v_esc) * randomDirection();
    }

    // Remove COM position and net momentum.
    Vec3 pcom{}, pmom{};
    double mtot = 0.0;
    for (std::size_t i = 0; i < c.N; ++i) {
        pcom += b.mass[i] * b.pos[i];
        pmom += b.mass[i] * b.vel[i];
        mtot += b.mass[i];
    }
    const Vec3 xshift = (1.0 / mtot) * pcom;
    const Vec3 vshift = (1.0 / mtot) * pmom;
    for (std::size_t i = 0; i < c.N; ++i) {
        b.pos[i] -= xshift;
        b.vel[i] -= vshift;
    }
    return b;
}

} // namespace galaxy3d
