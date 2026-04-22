#pragma once

#include <cstddef>
#include <vector>

#include "Vec3.hpp"

namespace galaxy3d {

// Structure-of-arrays particle set (3D analog of the 2D Bodies). SoA keeps each
// field contiguous, which the later octree / 3D-FMM solvers will want.
struct Bodies3D {
    std::vector<Vec3>   pos;   // positions r_i
    std::vector<Vec3>   vel;   // velocities v_i
    std::vector<double> mass;  // masses m_i

    std::size_t size() const { return mass.size(); }

    void resize(std::size_t n) {
        pos.assign(n, Vec3{});
        vel.assign(n, Vec3{});
        mass.assign(n, 0.0);
    }
};

using Accelerations3D = std::vector<Vec3>;

} // namespace galaxy3d
