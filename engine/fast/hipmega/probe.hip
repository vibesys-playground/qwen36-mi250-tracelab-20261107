// Plan 39 step 0 (a): trivial HIP kernels to prove the run-time compile path and measure what a
// kernel held to 128 VGPRs (amdgpu_waves_per_eu(4)) can stream from HBM on one gfx90a GCD.
//
// bw<U>: every thread issues U 16-byte loads (global_load_dwordx4) before it consumes any, over a
// grid-stride loop; the XOR of what it read is stored only if it hits an impossible value, so
// the loads cannot be removed.
// p2p_pingpong: one thread on each GCD bounces a counter through the peer's fine-grained memory
// (system-scope atomic store to the peer, uncached spin on its own word): the one-way latency of
// an in-kernel flag across the in-package link, the cost floor of an in-kernel all-reduce.

typedef unsigned v4u __attribute__((ext_vector_type(4)));

#define CAP128 __attribute__((amdgpu_flat_work_group_size(1, 512), amdgpu_waves_per_eu(4)))

template <int U>
__device__ __forceinline__ void bw_body(const v4u* __restrict__ a, unsigned* out, long n) {
  unsigned s = 0;
  const long stride = (long)gridDim.x * blockDim.x;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n; i += U * stride) {
    v4u v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load(a + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n; i += stride) {
    v4u v = a[i];
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x9e3779b9u) out[0] = s;
}

extern "C" __global__ void CAP128 bw1(const v4u* __restrict__ a, unsigned* out, long n) { bw_body<1>(a, out, n); }
extern "C" __global__ void CAP128 bw4(const v4u* __restrict__ a, unsigned* out, long n) { bw_body<4>(a, out, n); }
extern "C" __global__ void CAP128 bw8(const v4u* __restrict__ a, unsigned* out, long n) { bw_body<8>(a, out, n); }
extern "C" __global__ void CAP128 bw16(const v4u* __restrict__ a, unsigned* out, long n) { bw_body<16>(a, out, n); }

// rank 0 starts: writes 1 to peer[0]; rank 1 waits for 1 on mine[0], writes 2 to peer[0]; ...
// iters round trips; mine/peer are fine-grained (hipExtMallocWithFlags(..., 1)) 64-bit words.
extern "C" __global__ void p2p_pingpong(unsigned long* mine, unsigned long* peer, int rank,
                                        long iters, long base, int* err) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (long it = 0; it < iters; ++it) {
    unsigned long v = (unsigned long)(base + 2 * it + 1);  // rank 0 sends odd, rank 1 even
    if (rank == 0) {
      __hip_atomic_store(peer, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      long spin = 0;
      while (__hip_atomic_load(mine, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != v + 1) {
        if (++spin > 50000000) { err[0] = 1; return; }
      }
    } else {
      long spin = 0;
      while (__hip_atomic_load(mine, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != v) {
        if (++spin > 50000000) { err[0] = 1; return; }
      }
      __hip_atomic_store(peer, v + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// Software grid barrier cost: every block arrives on a monotone agent-scope counter and spins
// until all blocks of this barrier arrived; ``iters`` barriers per launch.
extern "C" __global__ void CAP128 grid_barrier(unsigned* bar, unsigned base, int iters) {
  for (int it = 1; it <= iters; ++it) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      const unsigned target = base + (unsigned)it * gridDim.x;
      while ((int)(__hip_atomic_load(bar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) - target) < 0)
        __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
  }
}
