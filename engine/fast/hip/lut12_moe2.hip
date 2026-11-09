// Plan 44 kernel pass 2: expert-grouped routed-expert decode MoE kernels in HIP, for raw BF16
// and LUT-12 weights (engine/fast/lut12.py has the format), 16 weights per lane.
// Compiled with hipRTC (engine/fast/hipmega_rtc.py); no #include.
//
// Macros (host): H (2048), I (expert intermediate of the shard, 256), IS (shared, == I),
// E (experts), N3 (router row width E + 2 IS + 1), TOPK (8), MAXP (the step's rows: the most
// pairs one expert can have), NT (gate_up n per block, multiple of 16), SC (gate_up K steps
// per load chunk: 1, 2, 4 or 8), NTD (down rows per block, multiple of 16), INK (1: the down
// kernel's last block per output tile does the combine; 0: moe_combine does it).
//
// Both kernels: 256 threads = 16 groups of 16 lanes; a group owns one weight row at a time
// and lane l reads 16 consecutive weights per K step (raw: 32 B; LUT-12: 16 low bytes, 8
// nibble bytes, the tile's 8-byte lut), so the 16 lanes of a group read a contiguous 512 B
// (raw) per step. A row's dot is reduced over its 16 lanes with 4 DPP adds; no LDS
// reduction and no cross-wave sync.
//
// moe_up_{raw,pk,sraw,spk}: one block per (distinct expert u of the step, tile of NT rows n).
//   Every (row, pick) pair routed to the expert is done in the block, so each weight is read
//   and decoded once per step. h of the pairs is staged in LDS (bf16, MAXP x H). Group g does
//   n = tile*NT + g + 16 i (gate row n and up row I + n); K = H in 8 steps of 256 columns,
//   lane l at columns 256 s + 16 l. Loads are double-buffered in chunks of SC steps.
//   act[pair, n] = bf16(bf16(silu(bf16 g)) * bf16 u).
// moe_down_{raw,pk,sraw,spk}: one block per (item u, tile of NTD output rows). Items 0..U-1
//   are the step's distinct experts, item U the shared expert (all R rows, its activation
//   bf16(bf16(silu(g)) * u) from the router row). K = I = 256 is one step of 16 lanes x 16.
//   Writes bf16(part) per (pair, n) and bf16(shared) per (row, n) as fp32; then the combine
//   (Triton's order and roundings): out = bf16(res + bf16(routed + bf16(sg * shared))),
//   routed = bf16(sum_j bf16(part_j * w_j)). With INK the block that finishes a tile last
//   (atomic counter per tile, reset by that block) combines the tile; else moe_combine.
// s* variants (streaming ceiling): the same grid, loads and launch shape, no decode and no
//   math: the loaded words are xor-folded and written only if equal to an impossible value.
// lut12_decode: every tile back to BF16 through the in-register decoder (round trip).

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef float f2 __attribute__((ext_vector_type(2)));

#define TB 200  // bytes per LUT-12 tile of 128 weights

__device__ __forceinline__ float bf16r(float f) {
    u32 b = __float_as_uint(f);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xFFFF0000u);
}
__device__ __forceinline__ u16 bf16bits(float f) {
    u32 b = __float_as_uint(f);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (u16)(b >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bfv(u16 v) { return __uint_as_float((u32)v << 16); }

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}
// Sum over each row of 16 lanes; every lane gets its row's sum.
__device__ __forceinline__ float row16_sum(float v) {
    v += dpp<0xB1>(v);   // quad_perm [1,0,3,2]
    v += dpp<0x4E>(v);   // quad_perm [2,3,0,1]
    v += dpp<0x124>(v);  // row_ror:4
    v += dpp<0x128>(v);  // row_ror:8
    return v;
}

// ------------------------------------------------------------------ loads and decode
struct RawLd { uint4 a, b; };
struct PkLd { uint4 lo; uint2 nib; uint2 lut; };

template <typename T>
__device__ __forceinline__ T ldu(const u8* __restrict__ p) {  // any 4-byte alignment
    T v;
    __builtin_memcpy(&v, p, sizeof(T));
    return v;
}

// 16 weights starting at column c (multiple of 16) of a row.
__device__ __forceinline__ RawLd ld16(const u16* __restrict__ row, int c) {
    RawLd r;
    r.a = *reinterpret_cast<const uint4*>(row + c);
    r.b = *reinterpret_cast<const uint4*>(row + c + 8);
    return r;
}
__device__ __forceinline__ PkLd ld16(const u8* __restrict__ row, int c) {
    const u8* t = row + (c >> 7) * TB;
    const int s = (c & 127) >> 4;
    PkLd r;
    r.lo = ldu<uint4>(t + s * 16);
    r.nib = ldu<uint2>(t + 128 + s * 8);
    r.lut = ldu<uint2>(t + 192);
    return r;
}

__device__ __forceinline__ void bf2x(u32 d, f2& w) {
    w.x = __uint_as_float(d << 16);
    w.y = __uint_as_float(d & 0xFFFF0000u);
}
__device__ __forceinline__ void cvt16(const RawLd& r, f2 w[8], const u8* __restrict__, int) {
    bf2x(r.a.x, w[0]); bf2x(r.a.y, w[1]); bf2x(r.a.z, w[2]); bf2x(r.a.w, w[3]);
    bf2x(r.b.x, w[4]); bf2x(r.b.y, w[5]); bf2x(r.b.z, w[6]); bf2x(r.b.w, w[7]);
}
// High bytes (sign | 7 exponent bits) of 8 weights from their 4 nibble bytes and the lut.
__device__ __forceinline__ void hi8(u32 nib, uint2 lut, u32& HA, u32& HB) {
    const u32 A = nib & 0x0F0F0F0Fu, B = (nib >> 4) & 0x0F0F0F0Fu;
    HA = __builtin_amdgcn_perm(lut.y, lut.x, A & 0x07070707u) | ((A & 0x08080808u) << 4);
    HB = __builtin_amdgcn_perm(lut.y, lut.x, B & 0x07070707u) | ((B & 0x08080808u) << 4);
}
// fp32 words of 4 weights: high byte -> byte 3, low byte -> byte 2, zeros below.
__device__ __forceinline__ void as4(u32 Hx, u32 lo, f2& w0, f2& w1) {
    w0.x = __uint_as_float(__builtin_amdgcn_perm(Hx, lo, 0x04000c0cu));
    w0.y = __uint_as_float(__builtin_amdgcn_perm(Hx, lo, 0x05010c0cu));
    w1.x = __uint_as_float(__builtin_amdgcn_perm(Hx, lo, 0x06020c0cu));
    w1.y = __uint_as_float(__builtin_amdgcn_perm(Hx, lo, 0x07030c0cu));
}
__device__ __forceinline__ void cvt16(const PkLd& r, f2 w[8], const u8* __restrict__ side, int c) {
    u32 H0, H1, H2, H3;
    if (r.lut.y & 0x80000000u) {  // fallback tile: high bytes from the side table
        const uint4 h = ldu<uint4>(side + (size_t)r.lut.x * 128 + (c & 127));
        H0 = h.x; H1 = h.y; H2 = h.z; H3 = h.w;
    } else {
        hi8(r.nib.x, r.lut, H0, H1);
        hi8(r.nib.y, r.lut, H2, H3);
    }
    as4(H0, r.lo.x, w[0], w[1]);
    as4(H1, r.lo.y, w[2], w[3]);
    as4(H2, r.lo.z, w[4], w[5]);
    as4(H3, r.lo.w, w[6], w[7]);
}
__device__ __forceinline__ u32 fold(const RawLd& r) { return r.a.x ^ r.a.y ^ r.a.z ^ r.a.w ^ r.b.x ^ r.b.y ^ r.b.z ^ r.b.w; }
__device__ __forceinline__ u32 fold(const PkLd& r) { return r.lo.x ^ r.lo.y ^ r.lo.z ^ r.lo.w ^ r.nib.x ^ r.nib.y ^ r.lut.x ^ r.lut.y; }

__device__ __forceinline__ float dot16(const f2 w[8], const f2 x[8]) {
    f2 a = w[0] * x[0];
#pragma unroll
    for (int i = 1; i < 8; i++) a = __builtin_elementwise_fma(w[i], x[i], a);
    return a.x + a.y;
}

template <bool PK> struct Rows;
template <> struct Rows<false> {
    typedef RawLd Ld;
    static __device__ __forceinline__ const u16* row(const void* w, size_t r, int K) {
        return reinterpret_cast<const u16*>(w) + r * (size_t)K;
    }
};
template <> struct Rows<true> {
    typedef PkLd Ld;
    static __device__ __forceinline__ const u8* row(const void* w, size_t r, int K) {
        return reinterpret_cast<const u8*>(w) + r * (size_t)(K / 128 * TB);
    }
};

// ------------------------------------------------------------------ routing prologue
// s_misc: [0] the expert of item u (-1 if none), [1] its pair count, [2] distinct experts.
// s_pairs: its pairs, ascending. Every thread returns the same value.
__device__ __forceinline__ void expert_pairs(const int* __restrict__ idx, int P, int u,
                                             int* s_idx, int* s_first, int* s_misc, int* s_pairs) {
    const int tid = threadIdx.x;
    if (tid < P) s_idx[tid] = idx[tid];
    if (tid == 0) { s_misc[0] = -1; s_misc[1] = 0; s_misc[2] = 0; }
    __syncthreads();
    if (tid < P) {
        const int e = s_idx[tid];
        int f = 1;
        for (int t = 0; t < tid; t++) f &= (s_idx[t] != e);
        s_first[tid] = f;
    }
    __syncthreads();
    if (tid < P && s_first[tid]) {
        int rk = 0;
        for (int t = 0; t < tid; t++) rk += s_first[t];
        if (rk == u) s_misc[0] = s_idx[tid];
        atomicAdd(&s_misc[2], 1);
    }
    __syncthreads();
    const int e = s_misc[0];
    if (e >= 0 && tid < P && s_idx[tid] == e) {
        int slot = 0;
        for (int t = 0; t < tid; t++) slot += (s_idx[t] == e);
        s_pairs[slot] = tid;
        atomicAdd(&s_misc[1], 1);
    }
    __syncthreads();
}

// ------------------------------------------------------------------ gate_up
#define NTILE_UP (I / NT)
#define NPG (NT / 16)          // n per group
#define NCH (NPG * 8 / SC)     // load chunks per group

// MODE 0: compute; 1: streaming ceiling.
template <bool PK, int MODE>
__device__ __forceinline__ void moe_up(const u16* __restrict__ h, const int* __restrict__ idx,
                                       const void* __restrict__ w, const u8* __restrict__ side,
                                       u16* __restrict__ act, int R) {
    typedef Rows<PK> RW;
    typedef typename RW::Ld Ld;
    // LDS is exactly the staged h (MAXP x H bf16, 32 KB at 8 rows: two blocks per CU); the
    // routing prologue borrows its first bytes and the results move to registers first.
    __shared__ __attribute__((aligned(16))) u16 s_x[MAXP * H];
    int* s_idx = reinterpret_cast<int*>(s_x);
    int* s_first = s_idx + 64;
    int* s_misc = s_first + 64;
    int* s_pairs = s_misc + 4;
    const int P = R * TOPK;
    const int u = blockIdx.x / NTILE_UP, tile = blockIdx.x % NTILE_UP;
    expert_pairs(idx, P, u, s_idx, s_first, s_misc, s_pairs);
    const int e = s_misc[0];
    if (e < 0) return;
    const int np = s_misc[1];
    int pr[MAXP];
#pragma unroll
    for (int p = 0; p < MAXP; p++) pr[p] = p < np ? s_pairs[p] : 0;
    __syncthreads();
    const int tid = threadIdx.x, g = tid >> 4, l = tid & 15;
    if (MODE == 0) {
        // stage h of the pairs: np x H bf16, 16 B per thread per row (H / 8 == 256)
#pragma unroll
        for (int p = 0; p < MAXP; p++) {
            if (p < np) {
                for (int c = tid * 8; c < H; c += 256 * 8)
                    *reinterpret_cast<uint4*>(s_x + p * H + c) =
                        *reinterpret_cast<const uint4*>(h + (size_t)(pr[p] / TOPK) * H + c);
            }
        }
        __syncthreads();
    }
    const size_t rbase = (size_t)e * 2 * I;
    Ld buf[2][SC][2];
    f2 ag[MAXP], au[MAXP];
#pragma unroll
    for (int p = 0; p < MAXP; p++) { ag[p] = (f2){0.f, 0.f}; au[p] = (f2){0.f, 0.f}; }
    u32 fx = 0;
    auto issue = [&](int b, int ch) {
        const int q = ch * SC;  // linear step index over (n_i, s)
#pragma unroll
        for (int j = 0; j < SC; j++) {
            const int ni = (q + j) / 8, s = (q + j) % 8;
            const int n = tile * NT + g + 16 * ni;
            const int c = s * 256 + l * 16;
            buf[b][j][0] = ld16(RW::row(w, rbase + n, H), c);
            buf[b][j][1] = ld16(RW::row(w, rbase + I + n, H), c);
        }
    };
    issue(0, 0);
#pragma unroll
    for (int ch = 0; ch < NCH; ch++) {
        if (ch + 1 < NCH) issue((ch + 1) & 1, ch + 1);
#pragma unroll
        for (int j = 0; j < SC; j++) {
            const int q = ch * SC + j, ni = q / 8, s = q % 8;
            const int c = s * 256 + l * 16;
            if (MODE == 1) {
                fx ^= fold(buf[ch & 1][j][0]) ^ fold(buf[ch & 1][j][1]);
                continue;
            }
            f2 wg[8], wu[8];
            cvt16(buf[ch & 1][j][0], wg, side, c);
            cvt16(buf[ch & 1][j][1], wu, side, c);
#pragma unroll
            for (int p = 0; p < MAXP; p++) {
                if (p < np) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(s_x + p * H + c);
                    const uint4 xb = *reinterpret_cast<const uint4*>(s_x + p * H + c + 8);
                    f2 x[8];
                    bf2x(xa.x, x[0]); bf2x(xa.y, x[1]); bf2x(xa.z, x[2]); bf2x(xa.w, x[3]);
                    bf2x(xb.x, x[4]); bf2x(xb.y, x[5]); bf2x(xb.z, x[6]); bf2x(xb.w, x[7]);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        ag[p] = __builtin_elementwise_fma(wg[i], x[i], ag[p]);
                        au[p] = __builtin_elementwise_fma(wu[i], x[i], au[p]);
                    }
                }
            }
            if (s == 7) {  // row n done: reduce over the group's 16 lanes, write act
                const int n = tile * NT + g + 16 * ni;
#pragma unroll
                for (int p = 0; p < MAXP; p++) {
                    if (p < np) {
                        float gs = row16_sum(ag[p].x + ag[p].y);
                        float us = row16_sum(au[p].x + au[p].y);
                        if (l == 0) {
                            gs = bf16r(gs);
                            us = bf16r(us);
                            const float a = bf16r(gs * sigm(gs));
                            act[(size_t)pr[p] * I + n] = bf16bits(a * us);
                        }
                    }
                    ag[p] = (f2){0.f, 0.f};
                    au[p] = (f2){0.f, 0.f};
                }
            }
        }
    }
    if (MODE == 1 && fx == 0x9E3779B9u) act[tid] = (u16)fx;
}

#define UP_KERNEL(NAME, PK, MODE)                                                              \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                    \
        const u16* __restrict__ h, const int* __restrict__ idx, const void* __restrict__ w,    \
        const u8* __restrict__ side, u16* __restrict__ act, int R) {                          \
        moe_up<PK, MODE>(h, idx, w, side, act, R);                                            \
    }
UP_KERNEL(moe_up_raw, false, 0)
UP_KERNEL(moe_up_pk, true, 0)
UP_KERNEL(moe_up_sraw, false, 1)
UP_KERNEL(moe_up_spk, true, 1)

// ------------------------------------------------------------------ down + shared + combine
#define NTILE_DN (H / NTD)
#define NPGD (NTD / 16)

// out[row, n] for the tile's NTD rows n and all R rows (the combine of ``moe_down``).
__device__ __forceinline__ void combine_one(int row, int n, const float* __restrict__ parts,
                                            const float* __restrict__ shb, const float* __restrict__ wt,
                                            const u16* __restrict__ r, const u16* __restrict__ res,
                                            u16* __restrict__ out, int has_res, bool coherent) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < TOPK; j++) {
        const float* pp = parts + (size_t)(row * TOPK + j) * H + n;
        const float part = coherent ? __hip_atomic_load(pp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *pp;
        acc += bf16r(part * wt[row * TOPK + j]);
    }
    const float routed = bf16r(acc);
    const float* sp = shb + (size_t)row * H + n;
    const float shared = coherent ? __hip_atomic_load(sp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *sp;
    const float sg = bf16r(sigm(bfv(r[(size_t)row * N3 + E + 2 * IS])));
    const float moe = bf16r(routed + bf16r(sg * shared));
    const float rv = has_res ? bfv(res[(size_t)row * H + n]) : 0.f;
    out[(size_t)row * H + n] = bf16bits(rv + moe);
}

template <bool PK, int MODE>
__device__ __forceinline__ void moe_down(
    const u16* __restrict__ act, const u16* __restrict__ r, const int* __restrict__ idx,
    const float* __restrict__ wt, const void* __restrict__ w, const u8* __restrict__ side,
    const void* __restrict__ sd, const u8* __restrict__ sside, const u16* __restrict__ res,
    u16* __restrict__ out, float* __restrict__ parts, float* __restrict__ shb, int* __restrict__ cnt,
    int R, int has_res) {
    typedef Rows<PK> RW;
    typedef typename RW::Ld Ld;
    __shared__ int s_idx[64], s_first[64], s_misc[3], s_pairs[MAXP], s_last;
    __shared__ __attribute__((aligned(16))) float s_x[MAXP * I];
    const int P = R * TOPK;
    const int u = blockIdx.x / NTILE_DN, tile = blockIdx.x % NTILE_DN;
    expert_pairs(idx, P, u, s_idx, s_first, s_misc, s_pairs);
    const int U = s_misc[2];
    if (u > U) return;
    const bool sh = (u == U);
    const int e = sh ? 0 : s_misc[0];
    const int np = sh ? R : s_misc[1];
    const int tid = threadIdx.x, g = tid >> 4, l = tid & 15;
    if (MODE == 0) {
        for (int i = tid; i < np * I; i += 256) {
            const int p = i / I, k = i % I;
            float v;
            if (sh) {
                const u16* rr = r + (size_t)p * N3 + E;
                const float gg = bfv(rr[k]), uu = bfv(rr[IS + k]);
                v = bf16r(bf16r(gg * sigm(gg)) * uu);
            } else {
                v = bfv(act[(size_t)s_pairs[p] * I + k]);
            }
            s_x[p * I + k] = v;
        }
        __syncthreads();
    }
    const void* wb = sh ? sd : w;
    const u8* sb = sh ? sside : side;
    const size_t rb = sh ? 0 : (size_t)e * H;
    const int c = l * 16;
    Ld ld[NPGD];
#pragma unroll
    for (int i = 0; i < NPGD; i++) ld[i] = ld16(RW::row(wb, rb + tile * NTD + g + 16 * i, I), c);
    if (MODE == 1) {
        u32 fx = 0;
#pragma unroll
        for (int i = 0; i < NPGD; i++) fx ^= fold(ld[i]);
        if (fx == 0x9E3779B9u) out[tid] = (u16)fx;
        return;
    }
#pragma unroll
    for (int i = 0; i < NPGD; i++) {
        const int n = tile * NTD + g + 16 * i;
        f2 wv[8];
        cvt16(ld[i], wv, sb, c);
#pragma unroll
        for (int p = 0; p < MAXP; p++) {
            if (p < np) {
                const float4* xp = reinterpret_cast<const float4*>(s_x + p * I + c);
                f2 x[8];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float4 v = xp[q];
                    x[2 * q] = (f2){v.x, v.y};
                    x[2 * q + 1] = (f2){v.z, v.w};
                }
                const float s = row16_sum(dot16(wv, x));
                if (l == 0) {
                    if (sh) shb[(size_t)p * H + n] = bf16r(s);
                    else parts[(size_t)s_pairs[p] * H + n] = bf16r(s);
                }
            }
        }
    }
#if INK
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&cnt[tile], 1) == U);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int t = tid; t < R * NTD; t += 256) {
        const int row = t / NTD, n = tile * NTD + t % NTD;
        combine_one(row, n, parts, shb, wt, r, res, out, has_res, true);
    }
    if (tid == 0) cnt[tile] = 0;
#endif
}

#define DN_KERNEL(NAME, PK, MODE)                                                              \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                    \
        const u16* __restrict__ act, const u16* __restrict__ r, const int* __restrict__ idx,   \
        const float* __restrict__ wt, const void* __restrict__ w, const u8* __restrict__ side, \
        const void* __restrict__ sd, const u8* __restrict__ sside, const u16* __restrict__ res, \
        u16* __restrict__ out, float* __restrict__ parts, float* __restrict__ shb,              \
        int* __restrict__ cnt, int R, int has_res) {                                          \
        moe_down<PK, MODE>(act, r, idx, wt, w, side, sd, sside, res, out, parts, shb, cnt, R,  \
                           has_res);                                                           \
    }
DN_KERNEL(moe_down_raw, false, 0)
DN_KERNEL(moe_down_pk, true, 0)
DN_KERNEL(moe_down_sraw, false, 1)
DN_KERNEL(moe_down_spk, true, 1)

// The combine as its own pass (INK 0): one thread per (row, n).
extern "C" __global__ __launch_bounds__(256) void moe_combine(
    const float* __restrict__ parts, const float* __restrict__ shb, const float* __restrict__ wt,
    const u16* __restrict__ r, const u16* __restrict__ res, u16* __restrict__ out, int R, int has_res) {
    const int t = blockIdx.x * 256 + threadIdx.x;
    if (t >= R * H) return;
    combine_one(t / H, t % H, parts, shb, wt, r, res, out, has_res, false);
}

// ------------------------------------------------------------------ round trip
extern "C" __global__ __launch_bounds__(256) void lut12_decode(
    const u8* __restrict__ planes, const u8* __restrict__ side, u16* __restrict__ out, long ntiles) {
    const long t = (long)blockIdx.x * 32 + (threadIdx.x >> 3);
    if (t >= ntiles) return;
    const int c = (threadIdx.x & 7) * 16;
    const PkLd r = ld16(planes + t * TB, c);
    f2 w[8];
    cvt16(r, w, side, c);
    u32 o[8];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i] = (__float_as_uint(w[i].x) >> 16) | (__float_as_uint(w[i].y) & 0xFFFF0000u);
    *reinterpret_cast<uint4*>(out + t * 128 + c) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(out + t * 128 + c + 8) = make_uint4(o[4], o[5], o[6], o[7]);
}
