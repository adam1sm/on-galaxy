cmake_minimum_required(VERSION 3.16)
project(galaxy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable(galaxy
    src/main.cpp
)
target_include_directories(galaxy PRIVATE src)

# Rung 1 validation harness: force-comparison vs the oracle + timing/scaling.
add_executable(harness
    src/harness.cpp
)
target_include_directories(harness PRIVATE src)

# Rung 2a: isolated FMM operator unit tests (P2M, M2L, M2M, L2L).
add_executable(fmm_tests
    src/fmm_tests.cpp
)
target_include_directories(fmm_tests PRIVATE src)

# ---------------------------------------------------------------------------
# 3D TRACK (Rung 3+). Separate sources under src3d/; builds alongside the 2D
# track without touching it.
# ---------------------------------------------------------------------------
# Rung 3: 3D foundation + DirectSolver3D oracle.
add_executable(galaxy3d
    src3d/main3d.cpp
)
target_include_directories(galaxy3d PRIVATE src3d)

# Rung 4: isolated 3D FMM operator unit tests (spherical harmonics).
add_executable(fmm3d_tests
    src3d/fmm3d_tests.cpp
)
target_include_directories(fmm3d_tests PRIVATE src3d)

# Rung 5: 3D assembled-solver validation harness (gates A-E + scaling).
add_executable(harness3d
    src3d/harness3d.cpp
)
target_include_directories(harness3d PRIVATE src3d)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(galaxy PRIVATE -Wall -Wextra)
    target_compile_options(harness PRIVATE -Wall -Wextra)
    target_compile_options(fmm_tests PRIVATE -Wall -Wextra)
    target_compile_options(galaxy3d PRIVATE -Wall -Wextra)
    target_compile_options(fmm3d_tests PRIVATE -Wall -Wextra)
    target_compile_options(harness3d PRIVATE -Wall -Wextra)
endif()
