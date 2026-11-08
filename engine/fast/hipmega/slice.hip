// Plan 39 step 0 (b, c): a persistent HIP kernel per GCD running consecutive Gated DeltaNet
// decoder layers of Qwen3.6-35B-A3B on one TP2 rank (its shard of the weights), for one chained
// verify step of N = 2 slots x G rows (row j * N + s is slot s's j-th token), with the two
// tensor-parallel all-reduces of each layer done inside the kernel over the GCD pair.
//
// One block of 512 threads (8 waves) per CU, all resident; phases are separated by a software
// grid barrier (a monotone agent-scope counter). The kernel is held to <= 128 VGPRs
// (amdgpu_waves_per_eu(4)). Weight-streaming phases are block-level GEMV tiles on the matrix
// cores (v_mfma_f32_16x16x16bf16_1k: rows padded to 16, 16 or 8 weight rows per tile), the K
// range split over the 8 waves and summed in a fixed order through LDS. Per layer (the
// engine's math, FastModel._fused_layer under TP2 with mfma.gdn_chain and FusedDecode._moe):
//   P_IN  every block: h = bf16(rms(x) * (1 + w_ln)) into LDS; in_proj GEMV (q|k|v|z|b|a rows)
//         with the short convolution (4 taps) of q|k|v as its epilogue: per slot its G rows in
//         turn, the window after row j stored to ring row states[j * N + s].
//   P_GDN 2 x 16 items (slot, v-head): the delta rule over the slot's G rows from state row
//         states[s] (fp32 128 x 128 tile in registers, 32 values per thread), the state after
//         row j stored to states[j * N + s], the gated RMSNorm into core.
//   P_OUT out_proj GEMV of core; its outputs go out as LL words (4 data bytes + 4 epoch bytes,
//         one 8-byte store) to this rank's and the peer's fine-grained receive buffers.
//   P_AR  every block reduces a slice of the words once both ranks' words carry this call's
//         epoch: x = bf16(x + bf16(p0 + p1)).
//   P_RT  every block: h2 = bf16(rms(x) * (1 + w_post)) into LDS; router|shared GEMV (769 rows).
//   P_UP  every block: fp32 softmax + top-8 + renormalize of each row (the same in every block),
//         the unique experts in ascending order with the (row, pick) pairs that chose each;
//         items (expert, 16 of its 512 gate|up rows): each weight row read once for all the
//         pairs that chose the expert -> bf16 gate/up outputs per pair.
//   P_DN  items of 8 output rows: act = bf16(bf16(silu(g)) * u) per pair into LDS; every unique
//         expert's 8 down rows read once (experts spread over the waves), each pair's part, the
//         shared expert (bf16(silu(g)) * u . w_sd) and its sigmoid gate; per row the parts
//         combined in pick order as the engine does; then the LL all-reduce as P_AR.
// Summation orders inside dot products differ from the Triton kernels (accepted class); the
// rounding points (every bf16 cast) are the engine's.

typedef unsigned short bf16_t;
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

#define KATTR __attribute__((amdgpu_flat_work_group_size(1, 512), amdgpu_waves_per_eu(4)))
#define INL __device__ __attribute__((always_inline))
// Phases are separate functions so each gets its own register allocation (a call per phase).
#define PHASE __device__ __attribute__((noinline))

#define H 2048
#define NIN 6176
#define CONVD 4096
#define KD 1024
#define Z_OFF 4096
#define B_OFF 6144
#define A_OFF 6160
#define VH 16
#define DK 128
#define DV 128
#define NE 256
#define TOPK 8
#define INTER 256
#define ISH 256
#define NMOE 769
#define NSLOT 2
#define MAXL 4
#define WORDS (8 * H / 2)
#define NWAVES 8
#define MAXSPIN (1 << 24)

struct LayerW {
  const bf16_t* ln_in;
  const bf16_t* ln_post;
  const bf16_t* w_in;     // [NIN, H]
  const bf16_t* conv_w;   // [CONVD, 4]
  const bf16_t* w_out;    // [H, VH * DV]
  const bf16_t* w_moe;    // [NMOE, H]
  const bf16_t* gate_up;  // [NE, 2 * INTER, H]
  const bf16_t* down;     // [NE, H, INTER]
  const bf16_t* sh_down;  // [H, ISH]
  bf16_t* conv_state;     // [rows, CONVD, 4]
  float* rec_state;       // [rows, VH, DK, DV]
  const float* A_log;     // [VH] fp32
  const float* dt_bias;   // [VH] fp32
  const float* norm_w;    // [DV] fp32
  long conv_stride;       // elements per conv state row
  long rec_stride;        // elements per recurrent state row
  float eps;              // decoder RMSNorm eps
  float gn_eps;           // gated norm eps
};

struct Args {
  LayerW L[MAXL];
  const bf16_t* x_in;     // [8, H]
  bf16_t* x_out;          // [8, H]
  const long* states;     // [8] state row of each row
  bf16_t* proj;           // [8, NIN]
  bf16_t* core;           // [8, VH * DV]
  bf16_t* rbuf;           // [MAXL, 8, NMOE]
  bf16_t* gu;             // [8 * TOPK, 2 * INTER] gate|up outputs per pair
  int* topk_idx;          // [MAXL, 8, TOPK]
  float* topk_w;          // [MAXL, 8, TOPK]
  bf16_t* part_out;       // [8, H] (phase probes without the all-reduce)
  unsigned long* recv;    // own fine-grained receive buffer [2 parity][2 src][WORDS]
  unsigned long* peer;    // the peer's
  unsigned* bar;          // grid barrier counter
  int* err;               // [0] LL timeouts, [1] barrier timeouts, [2] barriers passed
  int rank;
  int nl;
};

// ---------------------------------------------------------------- scalar helpers

__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float(((unsigned)x) << 16); }
__device__ __forceinline__ float lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf(f2bf(f)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ v4u ldg(const bf16_t* p) { return __builtin_nontemporal_load((const v4u*)p); }
__device__ __forceinline__ v4u ldc(const bf16_t* p) { return *(const v4u*)p; }
__device__ __forceinline__ int lane_id() { return threadIdx.x & 63; }
__device__ __forceinline__ int wave_id() { return threadIdx.x >> 6; }

__device__ __forceinline__ float wsum(float s) {
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) s += __shfl_xor(s, off);
  return s;
}
__device__ __forceinline__ float wmax(float s) {
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) s = fmaxf(s, __shfl_xor(s, off));
  return s;
}

__device__ __forceinline__ f4 mfma2(v4u a, v4u b, f4 c) {
  const v2u a0 = {a.x, a.y}, a1 = {a.z, a.w}, b0 = {b.x, b.y}, b1 = {b.z, b.w};
  c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(__builtin_bit_cast(s4, a0), __builtin_bit_cast(s4, b0), c, 0, 0, 0);
  c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(__builtin_bit_cast(s4, a1), __builtin_bit_cast(s4, b1), c, 0, 0, 0);
  return c;
}

// ---------------------------------------------------------------- grid barrier

// The barrier counter's value at this launch's start (a kernel argument; the host adds
// barriers x blocks after every launch, so the counter never needs a reset).
__shared__ unsigned bar_base0;

INL void grid_sync(const Args* a, unsigned& nbar) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  ++nbar;
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(a->bar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned target = bar_base0 + nbar * gridDim.x;
    int spin = 0;
    while ((int)(__hip_atomic_load(a->bar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) - target) < 0) {
      __builtin_amdgcn_s_sleep(1);
      if (++spin > MAXSPIN) { atomicAdd(a->err + 1, 1); break; }
    }
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// ---------------------------------------------------------------- shared memory

struct Smem {
  bf16_t xs[8 * H];               // GEMV A rows (8 x H) or, in P_DN, act (64 pairs x INTER)
  union {
    f4 red[NWAVES][64];           // K-split partials of a GEMV tile
    struct {
      float red[NWAVES][DV];
      float red2[NWAVES][DV];
    } g;
  } r;
  float ores[16][16];             // a GEMV tile's result [row][col]
  struct {                        // GDN row inputs
    float qn[DK], kn[DK], vv[DV];
    float beta, g, sq;
  } gdn;
  struct {                        // MoE routing (kept from P_UP to P_DN in each block)
    int idx[8 * TOPK];
    float wt[8 * TOPK];
    int cnt[NE];
    int ulist[8 * TOPK];
    int ustart[8 * TOPK + 1];
    int upair[8 * TOPK];
    int nu;
    bf16_t sa[8 * ISH];
    float sg[8];
    float parts[8 * TOPK][8];
    float shp[8][8];
  } moe;
};

// One static LDS block per workgroup (file scope, so every function addresses it as LDS).
__shared__ Smem sm;

// ---------------------------------------------------------------- pieces

// xs[r] = bf16(rms(x[r]) * (1 + w)) for r < R (wave r).
template <int R>
INL void rms_to_lds(const bf16_t* x, const bf16_t* w, float eps) {
  const int wave = wave_id(), lane = lane_id();
  for (int r = wave; r < R; r += NWAVES) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const v4u v = ldc(x + r * H + i * 512 + lane * 8);
      ss += lo(v.x) * lo(v.x) + hi(v.x) * hi(v.x) + lo(v.y) * lo(v.y) + hi(v.y) * hi(v.y)
          + lo(v.z) * lo(v.z) + hi(v.z) * hi(v.z) + lo(v.w) * lo(v.w) + hi(v.w) * hi(v.w);
    }
    ss = wsum(ss);
    const float rs = rsqrtf(ss / (float)H + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const v4u v = ldc(x + r * H + i * 512 + lane * 8);
      const v4u wv = ldc(w + i * 512 + lane * 8);
      v4u o;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float a0 = lo(v[k]) * rs * (1.0f + lo(wv[k]));
        const float a1 = hi(v[k]) * rs * (1.0f + hi(wv[k]));
        o[k] = (unsigned)f2bf(a0) | ((unsigned)f2bf(a1) << 16);
      }
      *(v4u*)(sm.xs + r * H + i * 512 + lane * 8) = o;
    }
  }
}

// One wave's MFMA partial of a tile: out[m][j] += sum over k in [kbeg, kbeg + 32 * STEPS) of
// A[m][k] * W[j][k], m = 4 * (lane >> 4) + reg, j = lane & 15. W rows j < nvalid (row stride
// ldw) from `w0`; A row m of this lane (m = lane & 15) is `arow` (a pointer to its k = 0) or null.
template <int STEPS>
INL f4 tile_part(const bf16_t* w0, long ldw, int nvalid, const bf16_t* arow, int kbeg, f4 c) {
  const int lane = lane_id(), g = lane >> 4, j = lane & 15;
  v4u b[STEPS];
  const bf16_t* wp = w0 + (long)j * ldw + kbeg + 8 * g;
#pragma unroll
  for (int s = 0; s < STEPS; ++s) b[s] = j < nvalid ? ldg(wp + s * 32) : (v4u){0u, 0u, 0u, 0u};
#pragma unroll
  for (int s = 0; s < STEPS; ++s) {
    const v4u av = arow ? *(const v4u*)(arow + kbeg + s * 32 + 8 * g) : (v4u){0u, 0u, 0u, 0u};
    c = mfma2(av, b[s], c);
  }
  return c;
}

// Block GEMV tile over K = 2048 (each wave 256 of it): sm.ores[m][j] = sum_k A[m][k] W[j][k]
// for 16 rows m (A row m = arow of the lane with lane & 15 == m) and j < nvalid; synced.
INL void tile_block(const bf16_t* w0, int nvalid, const bf16_t* arow) {
  const int wave = wave_id(), lane = lane_id();
  f4 c = {0.f, 0.f, 0.f, 0.f};
  c = tile_part<8>(w0, H, nvalid, arow, wave * 256, c);
  sm.r.red[wave][lane] = c;
  __syncthreads();
  if (wave == 0) {
    f4 s = sm.r.red[0][lane];
#pragma unroll
    for (int w = 1; w < NWAVES; ++w) s += sm.r.red[w][lane];
#pragma unroll
    for (int r = 0; r < 4; ++r) sm.ores[4 * (lane >> 4) + r][lane & 15] = s[r];
  }
  __syncthreads();
}

// ---------------------------------------------------------------- the LL all-reduce

__device__ __forceinline__ void ll_put(const Args* a, unsigned ep, int word, unsigned val) {
  const int par = ep & 1;
  const unsigned long v = ((unsigned long)ep << 32) | (unsigned long)val;
  const long off = ((long)par * 2 + a->rank) * WORDS + word;
  __hip_atomic_store(a->recv + off, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(a->peer + off, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// x_out[w] = bf16(res[w] + bf16(p0 + p1)) for this block's slice of the R * H / 2 words.
template <int R>
INL void ll_reduce(const Args* a, unsigned ep, const bf16_t* res) {
  const int nw = R * H / 2;
  const int per = (nw + gridDim.x - 1) / gridDim.x;
  const int w = blockIdx.x * per + threadIdx.x;
  if (threadIdx.x < per && w < nw) {
    const int par = ep & 1;
    const unsigned long* b0 = a->recv + ((long)par * 2 + 0) * WORDS + w;
    const unsigned long* b1 = a->recv + ((long)par * 2 + 1) * WORDS + w;
    unsigned long v0, v1;
    int spin = 0;
    while (true) {
      v0 = __hip_atomic_load(b0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if ((unsigned)(v0 >> 32) == ep) break;
      if (++spin > MAXSPIN) { atomicAdd(a->err, 1); break; }
    }
    spin = 0;
    while (true) {
      v1 = __hip_atomic_load(b1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if ((unsigned)(v1 >> 32) == ep) break;
      if (++spin > MAXSPIN) { atomicAdd(a->err, 1); break; }
    }
    const unsigned p0 = (unsigned)v0, p1 = (unsigned)v1;
    const unsigned rx = ((const unsigned*)res)[w];
    const float s0 = rbf(lo(p0) + lo(p1)), s1 = rbf(hi(p0) + hi(p1));
    ((unsigned*)a->x_out)[w] = (unsigned)f2bf(lo(rx) + s0) | ((unsigned)f2bf(hi(rx) + s1) << 16);
  }
}

// Store an 8-column tile result (rows < R) as bf16 words: to LL (out == nullptr) or `out`.
template <int R>
INL void put_tile8(const Args* a, int n0, unsigned ep, bf16_t* out) {
  const int t = threadIdx.x;
  if (t < R * 4) {
    const int r = t >> 2, np = (t & 3) * 2;
    const unsigned word = (unsigned)f2bf(sm.ores[r][np]) | ((unsigned)f2bf(sm.ores[r][np + 1]) << 16);
    if (out == nullptr) ll_put(a, ep, (r * H + n0 + np) / 2, word);
    else ((unsigned*)out)[(r * H + n0 + np) / 2] = word;
  }
}

// ---------------------------------------------------------------- phases

// P_IN: in_proj GEMV (+ conv epilogue when CONV) into proj.
template <int R, bool CONV>
PHASE void phase_in(const Args* a, const LayerW& L, const bf16_t* x) {
  rms_to_lds<R>(x, L.ln_in, L.eps);
  __syncthreads();
  const int lane = lane_id(), t = threadIdx.x;
  const bf16_t* arow = (lane & 15) < R ? sm.xs + (lane & 15) * H : nullptr;
  for (int tile = blockIdx.x; tile < NIN / 16; tile += gridDim.x) {
    const int n0 = tile * 16;
    tile_block(L.w_in + (long)n0 * H, 16, arow);
    if (!CONV || n0 >= CONVD) {
      if (t < R * 16) a->proj[(t >> 4) * NIN + n0 + (t & 15)] = f2bf(sm.ores[t >> 4][t & 15]);
      continue;
    }
    constexpr int G = R / NSLOT;
    if (t < NSLOT * 16) {
      const int s = t >> 4, c = n0 + (t & 15);
      const long first = a->states[s];
      const bf16_t* st = L.conv_state + first * L.conv_stride + c * 4;
      float s1 = bf(st[1]), s2 = bf(st[2]), s3 = bf(st[3]);
      const float w0 = bf(L.conv_w[c * 4 + 0]), w1 = bf(L.conv_w[c * 4 + 1]);
      const float w2 = bf(L.conv_w[c * 4 + 2]), w3 = bf(L.conv_w[c * 4 + 3]);
#pragma unroll
      for (int j = 0; j < G; ++j) {
        const int row = j * NSLOT + s;
        const float x = rbf(sm.ores[row][t & 15]);
        float acc = 0.f;
        acc += s1 * w0;
        acc += s2 * w1;
        acc += s3 * w2;
        acc += x * w3;
        const float y = rbf(acc);
        const long dst = a->states[row];
        bf16_t* d = L.conv_state + dst * L.conv_stride + c * 4;
        d[0] = f2bf(s1); d[1] = f2bf(s2); d[2] = f2bf(s3); d[3] = f2bf(x);
        a->proj[row * NIN + c] = f2bf(y * sigm(y));
        s1 = s2; s2 = s3; s3 = x;
      }
    }
  }
}

// P_GDN: the delta rule of (slot, head) items over the slot's G rows. Thread t holds the state
// rows k in [8 * (t >> 5), + 8) at columns v in [4 * (t & 31), + 4).
template <int R>
PHASE void phase_gdn(const Args* a, const LayerW& L) {
  constexpr int G = R / NSLOT;
  const int t = threadIdx.x, wave = wave_id(), lane = lane_id();
  const int vg = t & 31, kq = t >> 5;  // kq 0..15
  const float scale = rsqrtf((float)DK);
  for (int item = blockIdx.x; item < NSLOT * VH; item += gridDim.x) {
    const int s = item / VH, head = item % VH, khead = head / (VH / (KD / DK));
    const long first = a->states[s];
    const f4* src = (const f4*)(L.rec_state + first * L.rec_stride + (long)head * DK * DV);
    f4 S[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) S[i] = src[(kq * 8 + i) * (DV / 4) + vg];
#pragma unroll 1
    for (int j = 0; j < G; ++j) {
      const int row = j * NSLOT + s;
      const bf16_t* in = a->proj + row * NIN;
      if (wave == 0) {
        const float q0 = bf(in[khead * DK + 2 * lane]), q1 = bf(in[khead * DK + 2 * lane + 1]);
        const float rs = rsqrtf(wsum(q0 * q0 + q1 * q1) + 1e-6f) * scale;
        sm.gdn.qn[2 * lane] = q0 * rs;
        sm.gdn.qn[2 * lane + 1] = q1 * rs;
      } else if (wave == 1) {
        const float k0 = bf(in[KD + khead * DK + 2 * lane]), k1 = bf(in[KD + khead * DK + 2 * lane + 1]);
        const float rs = rsqrtf(wsum(k0 * k0 + k1 * k1) + 1e-6f);
        sm.gdn.kn[2 * lane] = k0 * rs;
        sm.gdn.kn[2 * lane + 1] = k1 * rs;
      } else if (wave == 2) {
        sm.gdn.vv[2 * lane] = bf(in[2 * KD + head * DV + 2 * lane]);
        sm.gdn.vv[2 * lane + 1] = bf(in[2 * KD + head * DV + 2 * lane + 1]);
      } else if (wave == 3 && lane == 0) {
        sm.gdn.beta = rbf(sigm(bf(in[B_OFF + head])));
        const float x = bf(in[A_OFF + head]) + L.dt_bias[head];
        const float sp = x > 20.f ? x : __logf(1.f + __expf(x));
        sm.gdn.g = -__expf(L.A_log[head]) * sp;
      }
      __syncthreads();
      const float decay = __expf(sm.gdn.g), beta = sm.gdn.beta;
      float kn[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) kn[i] = sm.gdn.kn[kq * 8 + i];
      f4 part = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        S[i] *= decay;
        part += S[i] * kn[i];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) part[r] += __shfl_xor(part[r], 32);
      if (lane < 32) *(f4*)&sm.r.g.red[wave][4 * vg] = part;
      __syncthreads();
      f4 kv = *(const f4*)&sm.r.g.red[0][4 * vg];
#pragma unroll
      for (int w = 1; w < NWAVES; ++w) kv += *(const f4*)&sm.r.g.red[w][4 * vg];
      const f4 vv = *(const f4*)&sm.gdn.vv[4 * vg];
      const f4 delta = (vv - kv) * beta;
      f4 op = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        S[i] += kn[i] * delta;
        op += S[i] * sm.gdn.qn[kq * 8 + i];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) op[r] += __shfl_xor(op[r], 32);
      if (lane < 32) *(f4*)&sm.r.g.red2[wave][4 * vg] = op;
      f4* dst = (f4*)(L.rec_state + a->states[row] * L.rec_stride + (long)head * DK * DV);
#pragma unroll
      for (int i = 0; i < 8; ++i) __builtin_nontemporal_store(S[i], dst + (kq * 8 + i) * (DV / 4) + vg);
      __syncthreads();
      if (t < 32) {
        f4 o = *(const f4*)&sm.r.g.red2[0][4 * vg];
#pragma unroll
        for (int w = 1; w < NWAVES; ++w) o += *(const f4*)&sm.r.g.red2[w][4 * vg];
#pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = rbf(o[r]);
        float ss = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) ss += __shfl_xor(ss, off);
        const float rs = rsqrtf(ss / (float)DV + L.gn_eps);
        unsigned short ob[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int v = 4 * vg + r;
          const float n = rbf(o[r] * rs);
          const float y = rbf(L.norm_w[v] * n);
          const float z = bf(in[Z_OFF + head * DV + v]);
          ob[r] = f2bf(y * z * sigm(z));
        }
        v2u w2 = {(unsigned)ob[0] | ((unsigned)ob[1] << 16), (unsigned)ob[2] | ((unsigned)ob[3] << 16)};
        *(v2u*)(a->core + row * (VH * DV) + head * DV + 4 * vg) = w2;
      }
      __syncthreads();
    }
  }
}

// Top-k of each row from rb (every block the same), unique experts and their pairs into LDS.
template <int R>
PHASE void topk_groups(const Args* a, const bf16_t* rb, int layer, bool write) {
  const int wave = wave_id(), lane = lane_id();
  for (int r = wave; r < R; r += NWAVES) {
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = bf(rb[r * NMOE + i * 64 + lane]);
    const float m = wmax(fmaxf(fmaxf(p[0], p[1]), fmaxf(p[2], p[3])));
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __expf(p[i] - m);
    const float tot = wsum(p[0] + p[1] + p[2] + p[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = p[i] / tot;
    float selw[TOPK];
    int seli[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      float bv = p[0];
      int bi = lane;
#pragma unroll
      for (int i = 1; i < 4; ++i)
        if (p[i] > bv) { bv = p[i]; bi = i * 64 + lane; }
#pragma unroll
      for (int off = 32; off >= 1; off >>= 1) {
        const float ov = __shfl_xor(bv, off);
        const int oi = __shfl_xor(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      selw[k] = bv;
      seli[k] = bi;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (bi == i * 64 + lane) p[i] = -1.0f;
    }
    float sw = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sw += selw[k];
    float wv = 0.f;
    int iv = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
      if (lane == k) { wv = selw[k]; iv = seli[k]; }
    if (lane < TOPK) {
      wv = rbf(wv / sw);
      sm.moe.idx[r * TOPK + lane] = iv;
      sm.moe.wt[r * TOPK + lane] = wv;
      if (write) {
        a->topk_idx[(layer * 8 + r) * TOPK + lane] = iv;
        a->topk_w[(layer * 8 + r) * TOPK + lane] = wv;
      }
    }
  }
  for (int e = threadIdx.x; e < NE; e += blockDim.x) sm.moe.cnt[e] = 0;
  __syncthreads();
  if (threadIdx.x < R * TOPK) atomicAdd(&sm.moe.cnt[sm.moe.idx[threadIdx.x]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int nu = 0, pos = 0;
    for (int e = 0; e < NE; ++e) {
      const int c = sm.moe.cnt[e];
      if (c) {
        sm.moe.ulist[nu] = e;
        sm.moe.ustart[nu] = pos;
        sm.moe.cnt[e] = pos;
        pos += c;
        ++nu;
      }
    }
    sm.moe.ustart[nu] = pos;
    sm.moe.nu = nu;
    for (int p = 0; p < R * TOPK; ++p) sm.moe.upair[sm.moe.cnt[sm.moe.idx[p]]++] = p;
  }
  __syncthreads();
}

// P_UP: items (unique expert u, tile of 16 of its 512 gate|up rows) -> gu[pair][row].
template <int R>
PHASE void phase_up(const Args* a, const LayerW& L) {
  const int lane = lane_id(), t = threadIdx.x;
  const int nu = sm.moe.nu;
  for (int item = blockIdx.x; item < nu * 32; item += gridDim.x) {
    const int u = item >> 5, tile = item & 31;
    const int e = sm.moe.ulist[u], s0 = sm.moe.ustart[u], c = sm.moe.ustart[u + 1] - s0;
    const int m = lane & 15;
    const bf16_t* arow = m < c ? sm.xs + (sm.moe.upair[s0 + m] / TOPK) * H : nullptr;
    tile_block(L.gate_up + ((long)e * 2 * INTER + tile * 16) * H, 16, arow);
    if (t < c * 16) a->gu[sm.moe.upair[s0 + (t >> 4)] * (2 * INTER) + tile * 16 + (t & 15)] = f2bf(sm.ores[t >> 4][t & 15]);
  }
}

// P_DN: items of 8 output rows; LL (all-reduce) or plain partial output.
template <int R, bool LL>
PHASE void phase_down(const Args* a, const LayerW& L, const bf16_t* rb, unsigned ep) {
  // act of every pair into xs (64 x INTER) and the shared expert's act into sa
  for (int i = threadIdx.x; i < R * TOPK * INTER; i += blockDim.x) {
    const int p = i / INTER, k = i % INTER;
    const float g = bf(a->gu[p * (2 * INTER) + k]), up = bf(a->gu[p * (2 * INTER) + INTER + k]);
    sm.xs[i] = f2bf(rbf(g * sigm(g)) * up);
  }
  for (int i = threadIdx.x; i < R * ISH; i += blockDim.x) {
    const int r = i / ISH, k = i % ISH;
    const float g = bf(rb[r * NMOE + NE + k]), up = bf(rb[r * NMOE + NE + ISH + k]);
    sm.moe.sa[i] = f2bf(rbf(g * sigm(g)) * up);
  }
  if (threadIdx.x < R) sm.moe.sg[threadIdx.x] = rbf(sigm(bf(rb[threadIdx.x * NMOE + NE + 2 * ISH])));
  __syncthreads();
  const int lane = lane_id(), wave = wave_id(), t = threadIdx.x;
  const int nu = sm.moe.nu;
  const int m = lane & 15, g = lane >> 4, j = lane & 15;
  for (int tile = blockIdx.x; tile < H / 8; tile += gridDim.x) {
    const int n0 = tile * 8;
    for (int u = wave; u <= nu; u += NWAVES) {
      f4 c = {0.f, 0.f, 0.f, 0.f};
      if (u < nu) {
        const int e = sm.moe.ulist[u], s0 = sm.moe.ustart[u], cnt = sm.moe.ustart[u + 1] - s0;
        const bf16_t* arow = m < cnt ? sm.xs + sm.moe.upair[s0 + m] * INTER : nullptr;
        c = tile_part<8>(L.down + ((long)e * H + n0) * INTER, INTER, 8, arow, 0, c);
        if (j < 8) {
#pragma unroll
          for (int r = 0; r < 4; ++r)
            if (4 * g + r < cnt) sm.moe.parts[sm.moe.upair[s0 + 4 * g + r]][j] = c[r];
        }
      } else {
        const bf16_t* arow = m < R ? sm.moe.sa + m * ISH : nullptr;
        c = tile_part<8>(L.sh_down + (long)n0 * ISH, ISH, 8, arow, 0, c);
        if (j < 8 && g < 2) {
#pragma unroll
          for (int r = 0; r < 4; ++r) sm.moe.shp[4 * g + r][j] = c[r];
        }
      }
    }
    __syncthreads();
    if (t < R * 8) {
      const int r = t >> 3, jj = t & 7;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const int p = r * TOPK + k;
        acc += rbf(rbf(sm.moe.parts[p][jj]) * sm.moe.wt[p]);
      }
      const float routed = rbf(acc);
      const float sh = rbf(sm.moe.shp[r][jj]);
      sm.ores[r][jj] = rbf(routed + rbf(sm.moe.sg[r] * sh));
    }
    __syncthreads();
    put_tile8<R>(a, n0, ep, LL ? nullptr : a->part_out);
  }
}

// ---------------------------------------------------------------- the slice kernel

template <int R>
INL void slice(const Args* a, unsigned bar_base, unsigned ep_base) {
  if (threadIdx.x == 0) bar_base0 = bar_base;
  __syncthreads();
  unsigned nbar = 0;
  unsigned ep = ep_base;
  const int lane = lane_id(), t = threadIdx.x;
#pragma unroll 1
  for (int l = 0; l < a->nl; ++l) {
    const LayerW& L = a->L[l];
    const bf16_t* x = l == 0 ? a->x_in : a->x_out;
    phase_in<R, true>(a, L, x);
    grid_sync(a, nbar);
    phase_gdn<R>(a, L);
    grid_sync(a, nbar);
    // P_OUT + P_AR
    for (int i = t; i < R * H / 8; i += blockDim.x) *(v4u*)(sm.xs + i * 8) = ldc(a->core + i * 8);
    __syncthreads();
    {
      const bf16_t* arow = (lane & 15) < R ? sm.xs + (lane & 15) * H : nullptr;
      for (int tile = blockIdx.x; tile < H / 8; tile += gridDim.x) {
        tile_block(L.w_out + (long)tile * 8 * H, 8, arow);
        put_tile8<R>(a, tile * 8, ep, nullptr);
      }
    }
    ll_reduce<R>(a, ep, x);
    ++ep;
    grid_sync(a, nbar);
    // P_RT
    rms_to_lds<R>(a->x_out, L.ln_post, L.eps);
    __syncthreads();
    bf16_t* rb = a->rbuf + (long)l * 8 * NMOE;
    {
      const bf16_t* arow = (lane & 15) < R ? sm.xs + (lane & 15) * H : nullptr;
      for (int tile = blockIdx.x; tile < (NMOE + 7) / 8; tile += gridDim.x) {
        const int n0 = tile * 8, nv = NMOE - n0 < 8 ? NMOE - n0 : 8;
        tile_block(L.w_moe + (long)n0 * H, nv, arow);
        if (t < R * 8 && (t & 7) < nv) rb[(t >> 3) * NMOE + n0 + (t & 7)] = f2bf(sm.ores[t >> 3][t & 7]);
      }
    }
    grid_sync(a, nbar);
    // P_UP
    topk_groups<R>(a, rb, l, blockIdx.x == 0);
    phase_up<R>(a, L);
    grid_sync(a, nbar);
    // P_DN + P_AR
    phase_down<R, true>(a, L, rb, ep);
    ll_reduce<R>(a, ep, a->x_out);
    ++ep;
    grid_sync(a, nbar);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) a->err[2] = (int)nbar;
}

extern "C" __global__ void KATTR hm_slice8(const Args* a, unsigned bar_base, unsigned ep_base) { slice<8>(a, bar_base, ep_base); }

// ---------------------------------------------------------------- phase probes
// One bandwidth phase alone over every layer of ``a``, ``reps`` times, a grid barrier after
// each: which = 0 in_proj (no conv), 1 out_proj, 2 router|shared, 3 MoE up, 4 MoE down (no
// all-reduce; routing from the top-k of rbuf[layer] as the slice left it, first R rows).

template <int R>
INL void probe(const Args* a, int which, int reps, unsigned bar_base) {
  if (threadIdx.x == 0) bar_base0 = bar_base;
  __syncthreads();
  unsigned nbar = 0;
  const int lane = lane_id(), t = threadIdx.x;
  for (int rep = 0; rep < reps; ++rep) {
    for (int l = 0; l < a->nl; ++l) {
      const LayerW& L = a->L[l];
      const bf16_t* rb = a->rbuf + (long)l * 8 * NMOE;
      if (which == 0) {
        phase_in<R, false>(a, L, a->x_in);
      } else if (which == 1 || which == 2) {
        rms_to_lds<R>(a->x_in, L.ln_post, L.eps);
        __syncthreads();
        const bf16_t* W = which == 1 ? L.w_out : L.w_moe;
        const int N = which == 1 ? H : NMOE;
        const bf16_t* arow = (lane & 15) < R ? sm.xs + (lane & 15) * H : nullptr;
        for (int tile = blockIdx.x; tile < (N + 7) / 8; tile += gridDim.x) {
          const int n0 = tile * 8, nv = N - n0 < 8 ? N - n0 : 8;
          tile_block(W + (long)n0 * H, nv, arow);
          if (t < R * 8 && (t & 7) < nv) a->part_out[(t >> 3) * H + n0 + (t & 7)] = f2bf(sm.ores[t >> 3][t & 7]);
        }
      } else if (which == 3) {
        rms_to_lds<R>(a->x_in, L.ln_post, L.eps);
        topk_groups<R>(a, rb, l, false);
        phase_up<R>(a, L);
      } else {
        topk_groups<R>(a, rb, l, false);
        phase_down<R, false>(a, L, rb, 0);
      }
      grid_sync(a, nbar);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) a->err[2] = (int)nbar;
}

extern "C" __global__ void KATTR hm_probe8(const Args* a, int which, int reps, unsigned bar_base) { probe<8>(a, which, reps, bar_base); }
extern "C" __global__ void KATTR hm_probe4(const Args* a, int which, int reps, unsigned bar_base) { probe<4>(a, which, reps, bar_base); }
