#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "../Types3D.hpp"

// ============================================================================
// RUNG 8: galaxy / merger initial conditions (CPU). A rotating exponential disk
// + small Plummer bulge. Circular velocities are set SELF-CONSISTENTLY from the
// actual measured force field (the solver is called by the driver between
// makeDisk() and setCircularVelocities()), so the disk balances against the real
// softened gravity and holds together -- far more robust than an analytic v_c.
// ============================================================================

namespace galaxy3d {

struct DiskParams {
    std::size_t N = 200000;
    double Rd = 1.0;          // exponential disk scale radius
    double Mtot = 1.0;        // total mass
    double thickness = 0.06;  // vertical scale (Gaussian)
    double rMax = 4.0;        // truncate disk radius (bounds the box for the uniform tree)
    double bulgeFrac = 0.08;  // fraction of mass/particles in a central bulge
    double bulgeRadius = 0.5; // Plummer bulge scale (less concentrated -> cheaper P2P)
    double dispFrac = 0.22;   // velocity dispersion as a fraction of local v_circ
    double rotSign = 1.0;     // +1 prograde, -1 retrograde (about +z)
    unsigned seed = 1;
};

namespace detail {
inline Vec3 randUnit(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0, 1);
    const double cz = 2 * u(rng) - 1, sz = std::sqrt(std::max(0.0, 1 - cz * cz)), ph = 2 * M_PI * u(rng);
    return Vec3{sz * std::cos(ph), sz * std::sin(ph), cz};
}
} // namespace detail

// Positions + masses of one disk, centered at origin, in the z=0 plane.
// Velocities are left zero; fill them with setCircularVelocities() after a force
// evaluation. `species` (optional) is filled with the given tag per particle.
inline Bodies3D makeDisk(const DiskParams& p, std::vector<int>* species = nullptr, int tag = 0) {
    Bodies3D b;
    b.resize(p.N);
    if (species) species->assign(p.N, tag);
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> u(0, 1);
    std::normal_distribution<double> g(0, 1);
    const double m = p.Mtot / double(p.N);
    const std::size_t nBulge = (std::size_t)(p.bulgeFrac * p.N);

    for (std::size_t i = 0; i < p.N; ++i) {
        b.mass[i] = m;
        if (i < nBulge) {
            // central Plummer bulge (rounder, pressure-supported); cap radius.
            double r;
            do { const double uu = std::max(1e-9, u(rng));
                 r = p.bulgeRadius / std::sqrt(std::pow(uu, -2.0 / 3.0) - 1.0);
            } while (r > 3.0 * p.bulgeRadius);
            b.pos[i] = r * detail::randUnit(rng);
        } else {
            // exponential disk: pdf(r) ~ r exp(-r/Rd) = Gamma(2,Rd); truncate at rMax.
            double r;
            do { r = -p.Rd * (std::log(std::max(1e-12, u(rng))) + std::log(std::max(1e-12, u(rng))));
            } while (r > p.rMax);
            const double phi = 2 * M_PI * u(rng);
            b.pos[i] = Vec3{r * std::cos(phi), r * std::sin(phi), p.thickness * g(rng)};
        }
    }
    return b;
}

// Set velocities from measured accelerations: circular support in the disk plane
// (v_circ^2 = R * a_radial_inward) plus an isotropic velocity dispersion.
inline void setCircularVelocities(Bodies3D& b, const Accelerations3D& acc,
                                  const DiskParams& p) {
    std::mt19937 rng(p.seed + 12345);
    std::normal_distribution<double> g(0, 1);
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double x = b.pos[i].x, y = b.pos[i].y;
        const double R = std::sqrt(x * x + y * y);
        Vec3 v{};
        if (R > 1e-6) {
            const double a_rad = (acc[i].x * x + acc[i].y * y) / R; // inward => negative
            const double vc2 = -a_rad * R;
            const double vc = (vc2 > 0) ? std::sqrt(vc2) : 0.0;
            // tangential (prograde about +z)
            v = (p.rotSign * vc / R) * Vec3{-y, x, 0.0};
            const double sigma = p.dispFrac * vc;
            v += sigma * Vec3{g(rng), g(rng), 0.5 * g(rng)}; // thinner vertical heating
        } else {
            v = Vec3{0.3 * g(rng), 0.3 * g(rng), 0.3 * g(rng)};
        }
        b.vel[i] = v;
    }
}

// Rotate (incline about the x-axis by `incl`), translate to `center`, add bulk
// velocity `bulk`. Applied to both positions and velocities.
inline void placeGalaxy(Bodies3D& b, double incl, Vec3 center, Vec3 bulk) {
    const double c = std::cos(incl), s = std::sin(incl);
    auto rot = [&](Vec3 r) { return Vec3{r.x, c * r.y - s * r.z, s * r.y + c * r.z}; };
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.pos[i] = rot(b.pos[i]) + center;
        b.vel[i] = rot(b.vel[i]) + bulk;
    }
}

// Concatenate two body sets (and their species tags).
inline Bodies3D concat(const Bodies3D& a, const Bodies3D& b,
                       std::vector<int>& sa, std::vector<int>& sb, std::vector<int>& out) {
    Bodies3D r;
    const std::size_t n = a.size() + b.size();
    r.resize(n);
    out.resize(n);
    for (std::size_t i = 0; i < a.size(); ++i) {
        r.pos[i] = a.pos[i]; r.vel[i] = a.vel[i]; r.mass[i] = a.mass[i]; out[i] = sa[i];
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        r.pos[a.size() + i] = b.pos[i]; r.vel[a.size() + i] = b.vel[i];
        r.mass[a.size() + i] = b.mass[i]; out[a.size() + i] = sb[i];
    }
    return r;
}

} // namespace galaxy3d
