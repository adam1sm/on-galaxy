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

# Rung 10a: adaptive FMM (CGR U/V/W/X) gates -- CPU only.
add_executable(adaptive_tests
    src3d/adaptive_tests.cpp
)
target_include_directories(adaptive_tests PRIVATE src3d)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(galaxy PRIVATE -Wall -Wextra)
    target_compile_options(harness PRIVATE -Wall -Wextra)
    target_compile_options(fmm_tests PRIVATE -Wall -Wextra)
    target_compile_options(galaxy3d PRIVATE -Wall -Wextra)
    target_compile_options(fmm3d_tests PRIVATE -Wall -Wextra)
    target_compile_options(harness3d PRIVATE -Wall -Wextra)
    target_compile_options(adaptive_tests PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# RUNG 6: GPU track (CUDA). Built ONLY when a CUDA compiler is available, so the
# CPU tracks still build everywhere (e.g. the Mac, or this machine before the
# toolkit is installed). Target arch sm_86 (RTX 30-series / Ampere).
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    add_executable(gpu_harness
        src3d/gpu/gpu_harness.cu
        src3d/gpu/gpu_direct.cu
    )
    target_include_directories(gpu_harness PRIVATE src3d src3d/gpu)

    # Rung 7: GPU FMM (million-body). Larger per-thread local buffers -> bump.
    add_executable(fmm_gpu_harness
        src3d/gpu/fmm_gpu_harness.cu
        src3d/gpu/FMM3DGPU.cu
        src3d/gpu/gpu_direct.cu
    )
    target_include_directories(fmm_gpu_harness PRIVATE src3d src3d/gpu)
    set_target_properties(fmm_gpu_harness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Rung 8: galaxy / merger simulation driver (reuses the Rung 7 GPU FMM).
    add_executable(galaxy_sim
        src3d/gpu/galaxy_sim.cu
        src3d/gpu/FMM3DGPU.cu
        src3d/gpu/AdaptiveFMM3DGPU.cu
    )
    target_include_directories(galaxy_sim PRIVATE src3d src3d/gpu)
    set_target_properties(galaxy_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Rung 10b: GPU adaptive FMM (CGR) harness.
    add_executable(adaptive_gpu_harness
        src3d/gpu/adaptive_gpu_harness.cu
        src3d/gpu/AdaptiveFMM3DGPU.cu
        src3d/gpu/FMM3DGPU.cu
        src3d/gpu/gpu_direct.cu
    )
    target_include_directories(adaptive_gpu_harness PRIVATE src3d src3d/gpu)
    set_target_properties(adaptive_gpu_harness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER}) -- building gpu_harness + fmm_gpu_harness for sm_${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA not found -- building CPU tracks only (gpu_harness skipped)")
endif()
