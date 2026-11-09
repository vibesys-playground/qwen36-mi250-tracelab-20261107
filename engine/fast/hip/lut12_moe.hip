// Plan 44 step 0: routed-expert decode MoE kernels in HIP, for raw BF16 and LUT-12 weights
// (engine/fast/lut12.py has the format). Compiled with hipRTC (engine/fast/hipmega_rtc.py);
// no #include: hipRTC provides the HIP device runtime.
//
// Macros (set by the host): H (hidden, 2048), I (expert intermediate of the shard, 256),
// IS (shared intermediate of the shard, == I), E (experts), N3 (router row width
// E + 2 IS + 1), TOPK (8), MAXP (rows that can share one expert: the step's row count),
// NT (gate_up output rows per block), ITD (down: n-iterations of 8 rows per block).
//
// moe_up_{raw,pk}: one block per (distinct expert u of the step, tile of NT rows n). Every
//   (row, pick) pair routed to that expert is computed in the same block, so each weight is
//   read from HBM and decoded once per step, whatever the number of rows sharing it.
//   256 threads; thread t holds K columns [8t, 8t+8) of every pair's h in registers (fp32),
//   each wave covers 512 columns; per n the 8-weight dot is reduced over the wave with DPP
//   and over the 4 waves through LDS. act[pair, n] = bf16(bf16(silu(bf16 g)) * bf16 u).
// moe_down_{raw,pk}: one block per (output tile of 8*ITD rows, row r), tile-outer like the
//   Triton kernel. 8 groups of 32 lanes, one output row n each; lane l holds columns
//   [8l, 8l+8) of the row's 8 act rows and of the shared expert's activation. Per n: the 8
//   routed experts and the shared expert, each a 32-lane DPP reduction, then the Triton
//   kernel's combine (bf16 roundings in its order) and the residual.
// lut12_decode: every tile back to BF16 through the same in-register decoder (round trip).

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

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}
// Sum over each row of 16 lanes, every lane gets its row's sum.
__device__ __forceinline__ float row16_sum(float v) {
    v += dpp<0xB1>(v);   // quad_perm [1,0,3,2]
    v += dpp<0x4E>(v);   // quad_perm [2,3,0,1]
    v += dpp<0x124>(v);  // row_ror:4
    v += dpp<0x128>(v);  // row_ror:8
    return v;
}
__device__ __forceinline__ float sum32(float v) {
    v = row16_sum(v);
    v += __shfl_xor(v, 16);
    return v;
}
__device__ __forceinline__ float sum64(float v) {
    v = sum32(v);
    v += __shfl_xor(v, 32);
    return v;
}

// ------------------------------------------------------------------ weight loads
// RAW: 8 consecutive BF16 of a row. PK: the lane's slot of a LUT-12 tile.
struct RawLd { uint4 v; };
struct PkLd { uint2 lo; u32 nib; uint2 lut; };

// k: first of the 8 columns (multiple of 8). row: element pointer of the row (raw) or byte
// pointer of its first tile (packed).
__device__ __forceinline__ RawLd ld_raw(const u16* __restrict__ row, int k) {
    RawLd r;
    r.v = *reinterpret_cast<const uint4*>(row + k);
    return r;
}
__device__ __forceinline__ PkLd ld_pk(const u8* __restrict__ row, int k) {
    const u8* t = row + (k >> 7) * TB;
    int s = (k & 127) >> 3;
    PkLd r;
    r.lo = *reinterpret_cast<const uint2*>(t + s * 8);
    r.nib = *reinterpret_cast<const u32*>(t + 128 + s * 4);
    r.lut = *reinterpret_cast<const uint2*>(t + 192);
    return r;
}

__device__ __forceinline__ void cvt(const RawLd& r, f2 w[4], const u8* __restrict__, int) {
    u32 d[4] = {r.v.x, r.v.y, r.v.z, r.v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        w[i].x = __uint_as_float(d[i] << 16);
        w[i].y = __uint_as_float(d[i] & 0xFFFF0000u);
    }
}
// The high bytes (sign | 7 exponent bits) of the lane's 8 weights: HA weights 0..3, HB 4..7.
__device__ __forceinline__ void pk_high(const PkLd& r, const u8* __restrict__ side, int k, u32& HA, u32& HB) {
    if (r.lut.y & 0x80000000u) {  // fallback tile: high bytes from the side table
        const uint2 h = *reinterpret_cast<const uint2*>(side + (size_t)r.lut.x * 128 + (k & 127));
        HA = h.x;
        HB = h.y;
    } else {
        u32 A = r.nib & 0x0F0F0F0Fu, B = (r.nib >> 4) & 0x0F0F0F0Fu;
        HA = __builtin_amdgcn_perm(r.lut.y, r.lut.x, A & 0x07070707u) | ((A & 0x08080808u) << 4);
        HB = __builtin_amdgcn_perm(r.lut.y, r.lut.x, B & 0x07070707u) | ((B & 0x08080808u) << 4);
    }
}
__device__ __forceinline__ void cvt(const PkLd& r, f2 w[4], const u8* __restrict__ side, int k) {
    u32 HA, HB;
    pk_high(r, side, k, HA, HB);
    // fp32 word of weight j: high byte -> byte 3, low byte -> byte 2, zeros below.
    w[0].x = __uint_as_float(__builtin_amdgcn_perm(HA, r.lo.x, 0x04000c0cu));
    w[0].y = __uint_as_float(__builtin_amdgcn_perm(HA, r.lo.x, 0x05010c0cu));
    w[1].x = __uint_as_float(__builtin_amdgcn_perm(HA, r.lo.x, 0x06020c0cu));
    w[1].y = __uint_as_float(__builtin_amdgcn_perm(HA, r.lo.x, 0x07030c0cu));
    w[2].x = __uint_as_float(__builtin_amdgcn_perm(HB, r.lo.y, 0x04000c0cu));
    w[2].y = __uint_as_float(__builtin_amdgcn_perm(HB, r.lo.y, 0x05010c0cu));
    w[3].x = __uint_as_float(__builtin_amdgcn_perm(HB, r.lo.y, 0x06020c0cu));
    w[3].y = __uint_as_float(__builtin_amdgcn_perm(HB, r.lo.y, 0x07030c0cu));
}

__device__ __forceinline__ void ld_x8(const u16* __restrict__ p, f2 x[4]) {
    RawLd r;
    r.v = *reinterpret_cast<const uint4*>(p);
    cvt(r, x, nullptr, 0);
}
// 8 BF16 at any 2-byte alignment (router rows are N3 = 769 elements apart).
__device__ __forceinline__ void ld_x8_u(const u16* __restrict__ p, f2 x[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        x[i].x = __uint_as_float((u32)p[2 * i] << 16);
        x[i].y = __uint_as_float((u32)p[2 * i + 1] << 16);
    }
}
__device__ __forceinline__ float dot8(const f2 w[4], const f2 x[4]) {
    f2 a = w[0] * x[0];
    a = __builtin_elementwise_fma(w[1], x[1], a);
    a = __builtin_elementwise_fma(w[2], x[2], a);
    a = __builtin_elementwise_fma(w[3], x[3], a);
    return a.x + a.y;
}

// Row pointers. Raw: element offsets. Packed: tiles of a row are contiguous (K/128 * TB bytes).
template <bool PK> struct Rows;
template <> struct Rows<false> {
    typedef RawLd Ld;
    typedef u16 T;
    static __device__ __forceinline__ const u16* row(const void* w, size_t r, int K) {
        return reinterpret_cast<const u16*>(w) + r * (size_t)K;
    }
    static __device__ __forceinline__ Ld ld(const u16* row, int k) { return ld_raw(row, k); }
};
template <> struct Rows<true> {
    typedef PkLd Ld;
    typedef u8 T;
    static __device__ __forceinline__ const u8* row(const void* w, size_t r, int K) {
        return reinterpret_cast<const u8*>(w) + r * (size_t)(K / 128 * TB);
    }
    static __device__ __forceinline__ Ld ld(const u8* row, int k) { return ld_pk(row, k); }
};

// ------------------------------------------------------------------ routing prologue
// The step's u-th distinct expert (in first-appearance order over pairs) and its pairs
// (ascending). Returns false when the step has fewer than u+1 distinct experts.
__device__ __forceinline__ bool expert_pairs(const int* __restrict__ idx, int P, int u,
                                             int* s_idx, int* s_first, int* s_misc, int* s_pairs) {
    const int tid = threadIdx.x;
    if (tid < P) s_idx[tid] = idx[tid];
    if (tid == 0) { s_misc[0] = -1; s_misc[1] = 0; }
    __syncthreads();
    if (tid < P) {
        int e = s_idx[tid], f = 1;
        for (int t = 0; t < tid; t++) f &= (s_idx[t] != e);
        s_first[tid] = f;
    }
    __syncthreads();
    if (tid < P && s_first[tid]) {
        int rk = 0;
        for (int t = 0; t < tid; t++) rk += s_first[t];
        if (rk == u) s_misc[0] = s_idx[tid];
    }
    __syncthreads();
    const int e = s_misc[0];
    if (e < 0) return false;
    if (tid < P && s_idx[tid] == e) {
        int slot = 0;
        for (int t = 0; t < tid; t++) slot += (s_idx[t] == e);
        s_pairs[slot] = tid;
        atomicAdd(&s_misc[1], 1);
    }
    __syncthreads();
    return true;
}

// ------------------------------------------------------------------ gate_up
#define NTILE_UP (I / NT)
#ifndef UN
#define UN 2  // n rows loaded ahead
#endif

template <bool PK>
__device__ __forceinline__ void moe_up(const u16* __restrict__ h, const int* __restrict__ idx,
                                       const void* __restrict__ w, const u8* __restrict__ side,
                                       u16* __restrict__ act, int R) {
    typedef Rows<PK> RW;
    __shared__ int s_idx[64], s_first[64], s_misc[2], s_pairs[MAXP];
    __shared__ float s_red[4][NT][MAXP][2];
    const int P = R * TOPK;
    const int u = blockIdx.x / NTILE_UP, tile = blockIdx.x % NTILE_UP;
    if (!expert_pairs(idx, P, u, s_idx, s_first, s_misc, s_pairs)) return;
    const int e = s_misc[0], np = s_misc[1];
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int k = tid * 8;
    f2 x[MAXP][4];
#pragma unroll
    for (int p = 0; p < MAXP; p++) {
        if (p < np) ld_x8(h + (size_t)(s_pairs[p] / TOPK) * H + k, x[p]);
        else for (int i = 0; i < 4; i++) x[p][i] = (f2){0.f, 0.f};
    }
    const size_t rbase = (size_t)e * 2 * I;
    for (int n0 = 0; n0 < NT; n0 += UN) {
        typename RW::Ld lg[UN], lu[UN];
#pragma unroll
        for (int j = 0; j < UN; j++) {
            const int n = tile * NT + n0 + j;
            lg[j] = RW::ld(RW::row(w, rbase + n, H), k);
            lu[j] = RW::ld(RW::row(w, rbase + I + n, H), k);
        }
#pragma unroll
        for (int j = 0; j < UN; j++) {
            f2 wg[4], wu[4];
            cvt(lg[j], wg, side, k);
            cvt(lu[j], wu, side, k);
#pragma unroll
            for (int p = 0; p < MAXP; p++) {
                if (p < np) {
                    float g = sum64(dot8(wg, x[p]));
                    float uu = sum64(dot8(wu, x[p]));
                    if (lane == 0) {
                        s_red[wave][n0 + j][p][0] = g;
                        s_red[wave][n0 + j][p][1] = uu;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int t = tid; t < NT * MAXP; t += 256) {
        const int nn = t / MAXP, p = t % MAXP;
        if (p < np) {
            float g = ((s_red[0][nn][p][0] + s_red[1][nn][p][0]) + s_red[2][nn][p][0]) + s_red[3][nn][p][0];
            float uu = ((s_red[0][nn][p][1] + s_red[1][nn][p][1]) + s_red[2][nn][p][1]) + s_red[3][nn][p][1];
            g = bf16r(g);
            uu = bf16r(uu);
            float a = bf16r(g * sigm(g));
            act[(size_t)s_pairs[p] * I + tile * NT + nn] = bf16bits(a * uu);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void moe_up_raw(
    const u16* __restrict__ h, const int* __restrict__ idx, const void* __restrict__ w,
    const u8* __restrict__ side, u16* __restrict__ act, int R) {
    moe_up<false>(h, idx, w, side, act, R);
}
extern "C" __global__ __launch_bounds__(256) void moe_up_pk(
    const u16* __restrict__ h, const int* __restrict__ idx, const void* __restrict__ w,
    const u8* __restrict__ side, u16* __restrict__ act, int R) {
    moe_up<true>(h, idx, w, side, act, R);
}

// ------------------------------------------------------------------ down + shared + combine
#define NTD (8 * ITD)
#define NTILE_DN (H / NTD)

template <bool PK>
__device__ __forceinline__ void moe_down(
    const u16* __restrict__ act, const u16* __restrict__ r, const int* __restrict__ idx,
    const float* __restrict__ wt, const void* __restrict__ w, const u8* __restrict__ side,
    const void* __restrict__ sd, const u8* __restrict__ sside, const u16* __restrict__ res,
    u16* __restrict__ out, int R, int has_res) {
    typedef Rows<PK> RW;
    const int tile = blockIdx.x / R, row = blockIdx.x % R;
    const int tid = threadIdx.x, grp = tid >> 5, l = tid & 31;
    const int k = l * 8;
    f2 x[TOPK][4], sa[4];
    int ex[TOPK];
    float wts[TOPK];
#pragma unroll
    for (int j = 0; j < TOPK; j++) {
        ex[j] = idx[row * TOPK + j];
        wts[j] = wt[row * TOPK + j];
        ld_x8(act + (size_t)(row * TOPK + j) * I + k, x[j]);
    }
    {
        f2 g[4], u[4];
        ld_x8_u(r + (size_t)row * N3 + E + k, g);
        ld_x8_u(r + (size_t)row * N3 + E + IS + k, u);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            sa[i].x = bf16r(bf16r(g[i].x * sigm(g[i].x)) * u[i].x);
            sa[i].y = bf16r(bf16r(g[i].y * sigm(g[i].y)) * u[i].y);
        }
    }
    const float sg = bf16r(sigm(__uint_as_float((u32)r[(size_t)row * N3 + E + 2 * IS] << 16)));
    for (int it = 0; it < ITD; it++) {
        const int n = tile * NTD + it * 8 + grp;
        typename RW::Ld lw[TOPK + 1];
#pragma unroll
        for (int j = 0; j < TOPK; j++) lw[j] = RW::ld(RW::row(w, (size_t)ex[j] * H + n, I), k);
        lw[TOPK] = RW::ld(RW::row(sd, (size_t)n, IS), k);
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < TOPK; j++) {
            f2 wv[4];
            cvt(lw[j], wv, side, k);
            const float part = sum32(dot8(wv, x[j]));
            acc += bf16r(bf16r(part) * wts[j]);
        }
        f2 wv[4];
        cvt(lw[TOPK], wv, sside, k);
        const float shared = bf16r(sum32(dot8(wv, sa)));
        const float routed = bf16r(acc);
        const float moe = bf16r(routed + bf16r(sg * shared));
        if (l == 0) {
            const float rv = has_res ? __uint_as_float((u32)res[(size_t)row * H + n] << 16) : 0.f;
            out[(size_t)row * H + n] = bf16bits(rv + moe);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void moe_down_raw(
    const u16* __restrict__ act, const u16* __restrict__ r, const int* __restrict__ idx,
    const float* __restrict__ wt, const void* __restrict__ w, const u8* __restrict__ side,
    const void* __restrict__ sd, const u8* __restrict__ sside, const u16* __restrict__ res,
    u16* __restrict__ out, int R, int has_res) {
    moe_down<false>(act, r, idx, wt, w, side, sd, sside, res, out, R, has_res);
}
extern "C" __global__ __launch_bounds__(256) void moe_down_pk(
    const u16* __restrict__ act, const u16* __restrict__ r, const int* __restrict__ idx,
    const float* __restrict__ wt, const void* __restrict__ w, const u8* __restrict__ side,
    const void* __restrict__ sd, const u8* __restrict__ sside, const u16* __restrict__ res,
    u16* __restrict__ out, int R, int has_res) {
    moe_down<true>(act, r, idx, wt, w, side, sd, sside, res, out, R, has_res);
}

// ------------------------------------------------------------------ round trip
// out[tile * 128 + c] = the BF16 bits of weight c of the tile, through the kernels' decoder.
extern "C" __global__ __launch_bounds__(256) void lut12_decode(
    const u8* __restrict__ planes, const u8* __restrict__ side, u16* __restrict__ out, long ntiles) {
    const long t = (long)blockIdx.x * 16 + (threadIdx.x >> 4);
    if (t >= ntiles) return;
    const int k = (threadIdx.x & 15) * 8;
    PkLd r = ld_pk(planes + t * TB, k);
    f2 w[4];
    cvt(r, w, side, k);
    u32 o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = (__float_as_uint(w[i].x) >> 16) | (__float_as_uint(w[i].y) & 0xFFFF0000u);
    *reinterpret_cast<uint4*>(out + t * 128 + k) = make_uint4(o[0], o[1], o[2], o[3]);
}
