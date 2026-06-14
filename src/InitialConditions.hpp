#pragma once

#include <cmath>
#include <random>
#include <stdexcept>

#include "Types.hpp"

namespace galaxy {

// ---------------------------------------------------------------------------
// (a) Two-body CIRCULAR orbit for the 2D LOG kernel.
//
// Orbital speed is derived by balancing centripetal and gravitational
// acceleration FOR THIS KERNEL -- this is NOT the 3D Kepler relation.
//
// Two bodies (m1, m2) separated by distance D orbit their common center of
// mass at radii r1 = D*m2/(m1+m2), r2 = D*m1/(m1+m2). The acceleration of body
// 1 toward body 2 from the softened log kernel has magnitude
//       a1 = G * m2 * D / (D^2 + eps^2).
// Circular motion requires a1 = omega^2 * r1. Substituting r1 gives an angular
// velocity that is consistent for BOTH bodies:
//       omega = sqrt( G (m1 + m2) / (D^2 + eps^2) ).
// Speeds are v1 = omega*r1, v2 = omega*r2 (note: independent of the Kepler
// 1/r^(3/2) law -- here omega depends on (D^2+eps^2)^(-1/2)).
//
// Bodies are placed on the x-axis about the origin and given perpendicular,
// opposite velocities, so total momentum is exactly zero.
// ---------------------------------------------------------------------------
struct TwoBodyConfig {
    double m1 = 1.0;
    double m2 = 1.0;
    double separation = 2.0; // D
    double G = 1.0;
    double eps = 0.01;
};

inline Bodies makeTwoBodyCircular(const TwoBodyConfig& c) {
    Bodies b;
    b.resize(2);

    const double M = c.m1 + c.m2;
    const double r1 = c.separation * c.m2 / M; // distance of body1 from COM
    const double r2 = c.separation * c.m1 / M; // distance of body2 from COM

    const double omega = std::sqrt(c.G * M / (c.separation * c.separation + c.eps * c.eps));
    const double v1 = omega * r1;
    const double v2 = omega * r2;

    // Body 1 left of COM, body 2 right of COM (COM at origin).
    b.mass[0] = c.m1;
    b.mass[1] = c.m2;
    b.pos[0] = Complex{-r1, 0.0};
    b.pos[1] = Complex{+r2, 0.0};
    // Perpendicular velocities, opposite signs -> counter-rotation, zero net P.
    b.vel[0] = Complex{0.0, -v1};
    b.vel[1] = Complex{0.0, +v2};
    return b;
}

// Period of the two-body circular orbit, T = 2*pi/omega. Useful for choosing
// run length and for the validation report.
inline double twoBodyPeriod(const TwoBodyConfig& c) {
    const double M = c.m1 + c.m2;
    const double omega = std::sqrt(c.G * M / (c.separation * c.separation + c.eps * c.eps));
    return 2.0 * M_PI / omega;
}

// ---------------------------------------------------------------------------
// (b) Randomly sampled disk/cluster of N bodies.
//
// Positions are sampled uniformly over a disk of radius R (area-uniform via
// r = R*sqrt(u)). Each body gets a gentle rigid rotation plus a small random
// thermal velocity, then the net momentum is removed so total P starts at zero.
// The exact velocity recipe is unimportant for Rung 0 correctness; what matters
// is that energy/momentum are conserved by the integrator over a long run.
// ---------------------------------------------------------------------------
struct DiskConfig {
    std::size_t N = 500;
    double radius = 1.0;       // R
    double total_mass = 1.0;   // distributed equally over N bodies
    double rotation = 0.6;     // rigid-rotation angular speed scale
    double thermal = 0.05;     // stddev of random velocity component
    unsigned seed = 42;
};

inline Bodies makeRandomDisk(const DiskConfig& c) {
    if (c.N == 0) throw std::invalid_argument("disk N must be > 0");
    Bodies b;
    b.resize(c.N);

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, c.thermal);

    const double m = c.total_mass / static_cast<double>(c.N);

    for (std::size_t i = 0; i < c.N; ++i) {
        const double u = uni(rng);
        const double theta = 2.0 * M_PI * uni(rng);
        const double r = c.radius * std::sqrt(u); // area-uniform radius
        const Complex z{r * std::cos(theta), r * std::sin(theta)};

        b.mass[i] = m;
        b.pos[i] = z;
        // Rigid rotation: v = omega * r in the tangential (i*z) direction,
        // plus a small isotropic thermal kick.
        const Complex tangential = Complex{0.0, 1.0} * z; // rotate position +90 deg
        b.vel[i] = c.rotation * tangential + Complex{gauss(rng), gauss(rng)};
    }

    // Remove net momentum and net position drift so COM stays at origin with
    // zero total momentum (keeps the momentum diagnostic clean).
    Complex pmom{0.0, 0.0};
    Complex pcom{0.0, 0.0};
    double mtot = 0.0;
    for (std::size_t i = 0; i < c.N; ++i) {
        pmom += b.mass[i] * b.vel[i];
        pcom += b.mass[i] * b.pos[i];
        mtot += b.mass[i];
    }
    const Complex vshift = pmom / mtot;
    const Complex xshift = pcom / mtot;
    for (std::size_t i = 0; i < c.N; ++i) {
        b.vel[i] -= vshift;
        b.pos[i] -= xshift;
    }
    return b;
}

} // namespace galaxy
