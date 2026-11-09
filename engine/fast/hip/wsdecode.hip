// Plan 65 stage 0: a wave-specialized persistent GEMV for gfx90a. Producer waves stream weight
// tiles HBM -> LDS with buffer_load_dword ... lds (DMA into LDS, no VGPR destination; in-flight
// bytes bounded by vmcnt, not by registers); consumer waves run the math from LDS with MFMA and
// handle the phase dependencies (a grid barrier and a latency-bound wait between phases), while
// the producers keep streaming the next phase's weights. Compiled with hipRTC
// (engine/fast/hipmega_rtc.py); no #include. Used only by tools/wsdecode_probe.py.
//
// out[r, n] = bf16(sum_k x[r, k] * W[n, k]) for r < R <= 16, W [N, K] bf16 row-major, K % 128 == 0,
// N % 16 == 0, x rows at stride xs, out rows at stride os (elements).
//
// Work: a "group" is 16 output rows (n0 .. n0 + 15) of one segment (weight matrix) over all K;
// a "tile" is that group's 16 rows x 128 columns (4 KB). The host gives each block (one per CU,
// persistent) a contiguous list of groups (``groups``: seg, n0, phase per group, block b's
// groups gstart[b] .. gstart[b + 1]); the block's tiles are numbered t = 0, 1, ... in group
// order, k chunk (kc = 0 .. K / 128 - 1) inner.
//
// Producers (waves 0 .. P - 1): wave p loads tiles t = p, p + P, ... into ring slot t % NS, row rho
// of the tile with one 64-lane dword DMA (256 B, lane i -> LDS dword i of the row). Lane i fetches
// 8-byte unit (i / 2) ^ rho of the row's 128 columns (an XOR swizzle by row), so the consumers'
// MFMA operand reads (lane (g, rho) reads unit s * 4 + g of row rho) hit 32 distinct banks. Up to
// PD tiles in flight per producer (16 DMAs each, vmcnt <= 63); the oldest is published
// (full[slot] = t + 1) once ``s_waitcnt vmcnt(16 * (PD - 1))`` says it landed. Before reusing a
// slot a producer publishes everything it has in flight, then waits for empty[slot] >= t - NS + 1.
// A producer's own LDS accesses are inline asm, so the compiler adds no vmcnt(0) for them.
//
// Consumers (waves P .. P + C - 1, index w): with KC = K / 128 chunks per group, wave w takes
// chunks kc = j * C + w (KC >= C) or the group's chunk w % KC of every group whose tiles it owns
// (KC < C, C % KC == 0), with x's MFMA A fragments for its chunks in registers (reloaded when the
// segment changes). Per tile: wait full[slot] == t + 1, 8 ds_read_b64 + 8
// v_mfma_f32_16x16x16bf16_1k (A = x rows, B = W rows), then empty[slot] = t + 1. A group's
// per-wave fp32 partials meet in LDS and the last wave to arrive sums them in a fixed order and
// stores BF16, so a run's bits never depend on timing, and rows never mix (MFMA rows are
// independent), so a row's bits do not depend on R.
//
// Phases: when a consumer reaches a group of a later phase, every consumer of the block
// arrives, consumer 0 runs a grid barrier (all blocks resident: grid = CUs, one block per CU)
// and then spins ``wait_ticks`` of s_memrealtime (100 MHz) as a stand-in for a latency-bound
// dependency phase, then releases the others. Producers ignore phases (PREFETCH != 0) and
// stream the next phase's tiles into the ring during the barrier and the wait; prefetch == 0
// holds them at the phase boundary (an A/B of the prefetch).
//
// Every spin is capped (SPIN_CAP iterations of s_sleep): on the cap the wave counts an error
// in err[0] and goes on, so a protocol bug shows as err > 0 and wrong numbers, never a hang.
//
// Macros: P (producer waves), C (consumer waves), PD (tiles in flight per producer), NS (ring
// slots of 4 KB), GR (partial buffers), HIGHREG (1: clobber v255 so the kernel is allocated 256
// VGPRs, the register budget of a fused GDN/MoE phase).

typedef unsigned int u32;
typedef unsigned long long u64;
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned int u2v __attribute__((ext_vector_type(2)));

#ifndef P
#define P 2
#endif
#ifndef C
#define C 4
#endif
#ifndef PD
#define PD 3
#endif
#ifndef NS
#define NS 12
#endif
#ifndef GR
#define GR 3
#endif
#ifndef HIGHREG
#define HIGHREG 1
#endif
#define KCW_MAX 16  // K <= 2048
#define KCW ((KCW_MAX + C - 1) / C)
#define TILE 4096
#define SPIN_CAP (1 << 22)
#define CAP (errs ? 256 : SPIN_CAP)  // after a first error, give up fast
#define WAVES (P + C)

#define LDS3 __attribute__((address_space(3)))

static __device__ __forceinline__ u32 lds_addr(const void* p) {
  return (u32)(size_t)(const LDS3 void*)p;
}

// s_waitcnt vmcnt(n) with expcnt / lgkmcnt left alone (gfx9 encoding).
#define WAIT_VM(n) __builtin_amdgcn_s_waitcnt(((n) & 15) | (7 << 4) | (15 << 8) | ((((n) >> 4) & 3) << 14))

static __device__ __forceinline__ u32 asm_lds_read(u32 addr) {
  u32 v;
  asm volatile("ds_read_b32 %0, %1\n\ts_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory");
  return v;
}

static __device__ __forceinline__ void asm_lds_write(u32 addr, u32 v) {
  asm volatile("ds_write_b32 %0, %1\n\ts_waitcnt lgkmcnt(0)" : : "v"(addr), "v"(v) : "memory");
}

static __device__ __forceinline__ unsigned short f2bf(float f) {
  u32 b = __float_as_uint(f);
  b += 0x7FFFu + ((b >> 16) & 1u);
  return (unsigned short)(b >> 16);
}

struct Seg {  // 8 x i64 per segment, as the host writes them
  long long w, x, out, N, K, xs, os, wbytes;
};

extern "C" __global__ __launch_bounds__(WAVES * 64) void ws_gemv(
    const long long* __restrict__ segs, const int* __restrict__ groups,
    const int* __restrict__ gstart, u32* gbar, int wait_ticks, int R, int prefetch,
    int* err) {
#if HIGHREG
  asm volatile("" ::: "v255");
#endif
  __shared__ __attribute__((aligned(16))) unsigned char ring[NS * TILE];
  __shared__ __attribute__((aligned(16))) f4 red[GR * C * 64];
  __shared__ u32 full[NS], empty[NS], arr[GR], done[GR], phase_arrive, phase_pass;

  const int tid = threadIdx.x;
  if (tid < NS) { full[tid] = 0; empty[tid] = 0; }
  if (tid < GR) { arr[tid] = 0; done[tid] = 0; }
  if (tid == 0) { phase_arrive = 0; phase_pass = 0; }
  __syncthreads();

  const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);
  const int lane = tid & 63;
  const int g0 = gstart[blockIdx.x], g1 = gstart[blockIdx.x + 1];
  const int nb = gridDim.x;
  int errs = 0;

  if (wave < P) {
    // ------------------------------------------------------------------ producer
    const int p = wave;
    // Lane i of a row DMA fetches unit (i / 2) ^ rho, dword i % 2 of it.
    const u32 lane_hi = (u32)(lane & 1) * 4u;
    const u32 lane_unit = (u32)(lane >> 1);
    int t = 0;          // block tile counter
    int oldest = p;     // oldest unpublished tile of this producer
    int npend = 0;
    int cur_phase = g0 < g1 ? groups[g0 * 4 + 2] : 0;
    for (int g = g0; g < g1; ++g) {
      const int seg = groups[g * 4 + 0];
      const int n0 = groups[g * 4 + 1];
      const int ph = groups[g * 4 + 2];
      const Seg* sd = (const Seg*)(segs + seg * 8);
      const long long K = sd->K;
      const int KC = (int)(K >> 7);
      if (ph != cur_phase) {
        if (!prefetch) {  // hold at the boundary: publish, then wait for the consumers
          WAIT_VM(0);
          for (; npend > 0; --npend, oldest += P) asm_lds_write(lds_addr(&full[oldest % NS]), (u32)(oldest + 1));
          int spins = 0;
          while ((int)asm_lds_read(lds_addr(&phase_pass)) < ph && spins < CAP) { __builtin_amdgcn_s_sleep(2); ++spins; }
          if (spins >= CAP) ++errs;
        }
        cur_phase = ph;
      }
      __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc(
          (void*)sd->w, (short)0, (int)(sd->wbytes > 0x7fffffffLL ? 0x7fffffff : sd->wbytes), 0x00027000);
      const u32 rowbytes = (u32)(K * 2);
      for (int kc = 0; kc < KC; ++kc, ++t) {
        if (t % P != p) continue;
        const int slot = t % NS;
        const int need = t - NS + 1;
        if (need > 0 && (int)asm_lds_read(lds_addr(&empty[slot])) < need) {
          WAIT_VM(0);
          for (; npend > 0; --npend, oldest += P) asm_lds_write(lds_addr(&full[oldest % NS]), (u32)(oldest + 1));
          int spins = 0;
          while ((int)asm_lds_read(lds_addr(&empty[slot])) < need && spins < CAP) { __builtin_amdgcn_s_sleep(1); ++spins; }
          if (spins >= CAP) ++errs;
        }
        unsigned char* dst = ring + slot * TILE;
        const u32 base = (u32)n0 * rowbytes + (u32)kc * 256u + lane_hi;
#pragma unroll
        for (int rho = 0; rho < 16; ++rho) {
          const u32 voff = base + (u32)rho * rowbytes + ((lane_unit ^ (u32)rho) << 3);
          __builtin_amdgcn_raw_ptr_buffer_load_lds(rs, (LDS3 void*)(dst + rho * 256), 4, voff, 0, 0, 0);
        }
        ++npend;
        if (npend == PD) {
          WAIT_VM(16 * (PD - 1));
          asm_lds_write(lds_addr(&full[oldest % NS]), (u32)(oldest + 1));
          oldest += P;
          --npend;
        }
      }
    }
    WAIT_VM(0);
    for (; npend > 0; --npend, oldest += P) asm_lds_write(lds_addr(&full[oldest % NS]), (u32)(oldest + 1));
  } else {
    // ------------------------------------------------------------------ consumer
    const int w = wave - P;
    const int rho = lane & 15;  // B column (W row) / A row (x row) of this lane
    const int gq = lane >> 4;   // k quad of this lane
    s4 xa[KCW][8];
#pragma unroll
    for (int j = 0; j < KCW; ++j)
#pragma unroll
      for (int s = 0; s < 8; ++s) xa[j][s] = (s4){0, 0, 0, 0};
    int cur_seg = -1;
    int cur_phase = g0 < g1 ? groups[g0 * 4 + 2] : 0;
    int t = 0;
    for (int g = g0, gi = 0; g < g1; ++g, ++gi) {
      const int seg = groups[g * 4 + 0];
      const int n0 = groups[g * 4 + 1];
      const int ph = groups[g * 4 + 2];
      const Seg* sd = (const Seg*)(segs + seg * 8);
      const int K = (int)sd->K;
      const int KC = K >> 7;
      if (ph != cur_phase) {
        // ---- phase boundary: block arrival, grid barrier, latency stand-in
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (lane == 0) __hip_atomic_fetch_add(&phase_arrive, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        if (w == 0) {
          int spins = 0;
          while ((int)__hip_atomic_load(&phase_arrive, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < C * ph && spins < CAP) {
            __builtin_amdgcn_s_sleep(1); ++spins;
          }
          if (spins >= CAP) ++errs;
          u32 old = 0;
          if (lane == 0) old = __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
          old = __builtin_amdgcn_readfirstlane(old);
          const u32 target = (old / (u32)nb + 1u) * (u32)nb;
          spins = 0;
          while (__hip_atomic_load(gbar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target && spins < CAP) {
            __builtin_amdgcn_s_sleep(1); ++spins;
          }
          if (spins >= CAP) ++errs;
          const long long t0 = (long long)__builtin_amdgcn_s_memrealtime();
          while ((long long)__builtin_amdgcn_s_memrealtime() - t0 < (long long)wait_ticks) __builtin_amdgcn_s_sleep(4);
          if (lane == 0) __hip_atomic_store(&phase_pass, (u32)ph, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        } else {
          int spins = 0;
          while ((int)__hip_atomic_load(&phase_pass, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < ph && spins < CAP) {
            __builtin_amdgcn_s_sleep(2); ++spins;
          }
          if (spins >= CAP) ++errs;
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        cur_phase = ph;
        cur_seg = -1;
      }
      // chunks of this group owned by this wave
      const bool wide = KC >= C;
      // (KC < C: chunk kc = w % KC of the group belongs to wave (t + kc) % C, t = the group's
      // first tile; the host keeps t % KC == 0 for such groups)
      if (seg != cur_seg) {
        const unsigned short* xp = (const unsigned short*)sd->x;
        const long long xs = sd->xs;
#pragma unroll
        for (int j = 0; j < KCW; ++j) {
          const int kc = wide ? j * C + w : (w % KC);
          if ((wide && kc < KC) || (!wide && j == 0)) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
              const int col = kc * 128 + s * 16 + gq * 4;
              u2v v = (u2v){0u, 0u};
              if (rho < R) v = *(const u2v*)(xp + (long long)rho * xs + col);
              xa[j][s] = __builtin_bit_cast(s4, v);
            }
          }
        }
        cur_seg = seg;
      }
      f4 acc = (f4){0.f, 0.f, 0.f, 0.f};
      int pslot = 0, nparts = 0;
      bool any = false;
      if (wide) {
        pslot = w;
        nparts = C;
        any = true;
#pragma unroll
        for (int j = 0; j < KCW; ++j) {
          const int kc = j * C + w;
          if (kc < KC) {
            const int tt = t + kc;
            const int slot = tt % NS;
            int spins = 0;
            while ((int)__hip_atomic_load(&full[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != tt + 1 && spins < CAP) {
              __builtin_amdgcn_s_sleep(1); ++spins;
            }
            if (spins >= CAP) ++errs;
            const unsigned char* src = ring + slot * TILE + rho * 256;
            s4 b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) b[s] = *(const s4*)(src + ((u32)((s * 4 + gq) ^ rho) << 3));
#pragma unroll
            for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(xa[j][s], b[s], acc, 0, 0, 0);
            __hip_atomic_store(&empty[slot], (u32)(tt + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
          }
        }
      } else {
        const int kc = w % KC;
        nparts = KC;
        pslot = kc;
        if (((t + kc) % C) == w) {
          any = true;
          const int tt = t + kc;
          const int slot = tt % NS;
          int spins = 0;
          while ((int)__hip_atomic_load(&full[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != tt + 1 && spins < CAP) {
            __builtin_amdgcn_s_sleep(1); ++spins;
          }
          if (spins >= CAP) ++errs;
          const unsigned char* src = ring + slot * TILE + rho * 256;
          s4 b[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) b[s] = *(const s4*)(src + ((u32)((s * 4 + gq) ^ rho) << 3));
#pragma unroll
          for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(xa[0][s], b[s], acc, 0, 0, 0);
          __hip_atomic_store(&empty[slot], (u32)(tt + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
      }
      if (any) {
        const int gb = gi % GR;
        int spins = 0;
        while ((int)__hip_atomic_load(&done[gb], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < gi - GR + 1 && spins < CAP) {
          __builtin_amdgcn_s_sleep(1); ++spins;
        }
        if (spins >= CAP) ++errs;
        red[(gb * C + pslot) * 64 + lane] = acc;
        u32 old = 0;
        if (lane == 0) old = __hip_atomic_fetch_add(&arr[gb], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_WORKGROUP);
        old = __builtin_amdgcn_readfirstlane(old);
        if ((int)old + 1 == nparts) {
          f4 sum = red[(gb * C + 0) * 64 + lane];
          for (int ps = 1; ps < nparts; ++ps) sum += red[(gb * C + ps) * 64 + lane];
          unsigned short* op = (unsigned short*)sd->out;
          const long long os = sd->os;
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const int r = gq * 4 + v;
            if (r < R) op[(long long)r * os + n0 + rho] = f2bf(sum[v]);
          }
          if (lane == 0) {
            __hip_atomic_store(&arr[gb], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
            __hip_atomic_store(&done[gb], (u32)(gi + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
          }
        }
      }
      t += KC;
    }
  }
  if (errs && lane == 0) atomicAdd(err, errs);
}

// The latency-bound stand-in of the graph baseline: every block spins ``ticks`` of
// s_memrealtime, then the grid copies x [R, n] (stride xs) to y [R, n] (contiguous) so the next
// GEMV reads a dense input, as a real dependency phase would write it.
extern "C" __global__ __launch_bounds__(64) void spin_copy(int ticks, const unsigned short* x, long long xs,
                                                           unsigned short* y, int R, int n) {
  const long long t0 = (long long)__builtin_amdgcn_s_memrealtime();
  while ((long long)__builtin_amdgcn_s_memrealtime() - t0 < (long long)ticks) __builtin_amdgcn_s_sleep(4);
  for (int i = blockIdx.x * 64 + threadIdx.x; i < R * n; i += gridDim.x * 64) y[i] = x[(long long)(i / n) * xs + (i % n)];
}
