#pragma once

#include <cmath>

namespace galaxy3d {

// Minimal double-precision 3D vector. The 3D track uses plain Cartesian vectors
// (unlike the 2D track's std::complex, which existed to line up with the 2D
// FMM's complex expansions). The 3D FMM (Rung 4) uses spherical harmonics, for
// which a real Vec3 is the natural representation.
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s)      { x *= s;   y *= s;   z *= s;   return *this; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { a += b; return a; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { a -= b; return a; }
inline Vec3 operator*(Vec3 a, double s)      { a *= s; return a; }
inline Vec3 operator*(double s, Vec3 a)      { a *= s; return a; }
inline Vec3 operator-(const Vec3& a)         { return Vec3{-a.x, -a.y, -a.z}; }

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}

inline double norm2(const Vec3& a) { return dot(a, a); }     // |a|^2
inline double abs(const Vec3& a)   { return std::sqrt(dot(a, a)); }

} // namespace galaxy3d
