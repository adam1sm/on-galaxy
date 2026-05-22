// ============================================================================
// RUNG 6: GPU direct O(N^2) N-body solver (CUDA). Classic tiled/shared-memory
// all-pairs kernel, templated on precision. Same 3D Plummer kernel as
// DirectSolver3D. STATUS: drafted without a CUDA toolchain -- not yet compiled.
// ============================================================================
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_direct.cuh"

namespace galaxy3d {
namespace gpu {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static constexpr int kBlock = 256;

// Tiled all-pairs kernel. Each thread owns one target body i, streams all source
// bodies j through shared memory in tiles of blockDim.x.
//   a_i = G * sum_{j!=i} m_j (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)
template <typename real>
__global__ void directKernel(const real* __restrict__ px, const real* __restrict__ py,
                             const real* __restrict__ pz, const real* __restrict__ mass,
                             real* __restrict__ ax, real* __restrict__ ay,
                             real* __restrict__ az, int n, real eps2, real G) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    real* sx = reinterpret_cast<real*>(smem_raw);
    real* sy = sx + kBlock;
    real* sz = sy + kBlock;
    real* sm = sz + kBlock;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const real xi = (i < n) ? px[i] : real(0);
    const real yi = (i < n) ? py[i] : real(0);
    const real zi = (i < n) ? pz[i] : real(0);

    real Ax = 0, Ay = 0, Az = 0;
    const int nTiles = (n + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < nTiles; ++tile) {
        const int j = tile * blockDim.x + threadIdx.x;
        sx[threadIdx.x] = (j < n) ? px[j] : real(0);
        sy[threadIdx.x] = (j < n) ? py[j] : real(0);
        sz[threadIdx.x] = (j < n) ? pz[j] : real(0);
        sm[threadIdx.x] = (j < n) ? mass[j] : real(0);
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < blockDim.x; ++k) {
            const int jj = tile * blockDim.x + k;
            const real dx = sx[k] - xi, dy = sy[k] - yi, dz = sz[k] - zi;
            const real r2 = dx * dx + dy * dy + dz * dz + eps2;
            // Short-circuit so self (r2 may be 0 at eps=0 -> inf) is never used.
            const real f = (jj < n && jj != i) ? sm[k] * rsqrt(r2) / r2 : real(0);
            Ax += f * dx; Ay += f * dy; Az += f * dz;
        }
        __syncthreads();
    }
    if (i < n) { ax[i] = G * Ax; ay[i] = G * Ay; az[i] = G * Az; }
}

// rsqrt(r2)/r2 = r2^{-3/2}; rsqrt is overloaded for float and double in CUDA.

namespace {
template <typename real>
struct DeviceArrays {
    real *px = nullptr, *py = nullptr, *pz = nullptr, *mass = nullptr;
    real *ax = nullptr, *ay = nullptr, *az = nullptr;
    int n = 0, nPad = 0;

    void alloc(const Bodies3D& b) {
        n = (int)b.size();
        nPad = ((n + kBlock - 1) / kBlock) * kBlock;
        const std::size_t bytes = sizeof(real) * nPad;
        for (real** p : {&px, &py, &pz, &mass, &ax, &ay, &az})
            CUDA_CHECK(cudaMalloc(p, bytes));
        std::vector<real> hx(nPad, 0), hy(nPad, 0), hz(nPad, 0), hm(nPad, 0);
        for (int i = 0; i < n; ++i) {
            hx[i] = (real)b.pos[i].x; hy[i] = (real)b.pos[i].y;
            hz[i] = (real)b.pos[i].z; hm[i] = (real)b.mass[i];
        }
        CUDA_CHECK(cudaMemcpy(px, hx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(py, hy.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pz, hz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mass, hm.data(), bytes, cudaMemcpyHostToDevice));
    }
    void launch(real eps2, real G) {
        const int grid = nPad / kBlock;
        const std::size_t shmem = 4 * kBlock * sizeof(real);
        directKernel<real><<<grid, kBlock, shmem>>>(px, py, pz, mass, ax, ay, az, n, eps2, G);
    }
    Accelerations3D fetch() {
        std::vector<real> hx(n), hy(n), hz(n);
        CUDA_CHECK(cudaMemcpy(hx.data(), ax, sizeof(real) * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), ay, sizeof(real) * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), az, sizeof(real) * n, cudaMemcpyDeviceToHost));
        Accelerations3D a(n);
        for (int i = 0; i < n; ++i) a[i] = Vec3{(double)hx[i], (double)hy[i], (double)hz[i]};
        return a;
    }
    void free() {
        for (real* p : {px, py, pz, mass, ax, ay, az}) cudaFree(p);
    }
};
} // namespace

template <typename real>
Accelerations3D directAccelerations(const Bodies3D& bodies, double G, double eps) {
    if (bodies.size() == 0) return {};
    DeviceArrays<real> d;
    d.alloc(bodies);
    d.launch((real)(eps * eps), (real)G);
    CUDA_CHECK(cudaDeviceSynchronize());
    Accelerations3D out = d.fetch();
    d.free();
    return out;
}

template <typename real>
double timeDirect(const Bodies3D& bodies, double G, double eps, int reps, Accelerations3D* out) {
    if (bodies.size() == 0) return 0.0;
    DeviceArrays<real> d;
    d.alloc(bodies);
    const real eps2 = (real)(eps * eps), Gr = (real)G;
    d.launch(eps2, Gr); // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(a));
        d.launch(eps2, Gr);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        best = (ms * 1e-3 < best) ? ms * 1e-3 : best;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    if (out) *out = d.fetch();
    d.free();
    return best;
}

DeviceInfo queryDevice(int id) {
    DeviceInfo info;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return info;
    if (id < 0 || id >= count) return info;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, id) != cudaSuccess) return info;
    info.ok = true;
    info.id = id;
    info.name = p.name;
    info.ccMajor = p.major; info.ccMinor = p.minor;
    info.smCount = p.multiProcessorCount;
    info.globalMemBytes = p.totalGlobalMem;
    // clockRate was removed from cudaDeviceProp in CUDA 13; query it directly.
    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, id);
    info.clockKHz = clk;
    info.warpSize = p.warpSize;
    return info;
}

// Explicit instantiations (float = perf, double = validation).
template Accelerations3D directAccelerations<float>(const Bodies3D&, double, double);
template Accelerations3D directAccelerations<double>(const Bodies3D&, double, double);
template double timeDirect<float>(const Bodies3D&, double, double, int, Accelerations3D*);
template double timeDirect<double>(const Bodies3D&, double, double, int, Accelerations3D*);

} // namespace gpu
} // namespace galaxy3d
