// Plan 50 stage 0: the decode / chained-verify Gated DeltaNet step (short conv + delta rule +
// gated RMSNorm) of one slot's G rows, with each (slot, v-head)'s fp32 state held in the
// registers of a few wavefronts and every reduction over DK done across lanes with DPP
// (no LDS and no barrier inside the row chain). Compiled with hipRTC
// (engine/fast/hipmega_rtc.py); no #include. HOST_EMU builds it as plain C++ for a CPU test
// (tools/emu/gdn_hip_emu.cpp), one std::thread per lane.
//
// Macros (host): KEY_DIM (q / k channels of the shard, 1024), V_HEADS (16), DK, DV (128).
// RATIO = V_HEADS / (KEY_DIM / DK) v-heads share one k-head's q / k.
//
// Layout (template LPC, CPL): a group of LPC adjacent lanes (inside one DPP row of 16) owns CPL
// adjacent value columns; lane t of the group owns state rows t*R .. t*R+R-1 (R = DK / LPC) of
// those columns: R*CPL fp32 registers. kv = S^T k and o = S^T q per column are R in-lane FMAs
// then log2(LPC) DPP adds. A wave covers CW = 64 / LPC * CPL columns, a head WPH = DV / CW
// waves; a block holds HPW heads (HPW = RATIO: the whole k-head, so the q / k conv ring rows
// are written in the block with no other reader).
//
// CONV: 0 = the projection's q|k|v columns are already convolved (today's Triton conv kernel
// ran first; it also wrote the conv ring); 1 = the conv runs in the prologue with
// ``_gdn_conv_chain_kernel``'s per-channel math, the ring rows of the block's own channels are
// written here, and (HPW < RATIO) ``gdn_qk_ring`` writes the q / k ring rows afterwards (every
// block of the k-head reads them from conv state row ``slots[r]``, which ring entry 0 may be).
//
// VS (grid z): value-column splits per head (more blocks, so the per-row state stores spread
// over more CUs); with VS > 1 the pre-norm o goes to ``obuf`` and ``gdn_post`` (one more
// launch) runs the gated norm and the q / k conv ring.
//
// Per block (slot r, heads head0 .. head0+HPW-1, columns of split z):
//   state tile loads issued first; prologue: every load, then conv (or load) every row's q, k, v of the block into LDS, gates beta, exp(g);
//             l2-normalise q, k per row (one wave per vector, DPP + readlane).
//   chain:    for j < G: S *= exp(g_j); kv = S^T k_j; S += k_j (beta_j (v_j - kv)); store S to
//             ring row slots[j N + r]; o_j = bf16(S^T q_j) -> LDS.   (no barrier)
//   epilogue: gated RMSNorm of each o_j over DV (one wave per (head, row)), today's tail.
// Rounding points as ``mfma._gdn_row``: bf16 conv output and silu, bf16 beta, bf16 o, norm, y,
// out; fp32 state. Only the fp32 summation order of the DK / DV reductions differs.

#ifdef HOST_EMU
#include "gdn_hip_emu_shim.h"
#else
typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;
#define DPP_I(v, ctrl) __builtin_amdgcn_update_dpp(0, (v), (ctrl), 0xF, 0xF, false)
#define READLANE_I(v, l) __builtin_amdgcn_readlane((v), (l))
#define REALTIME() ((i64)__builtin_amdgcn_s_memrealtime())
#define CLOCK() ((i64)__builtin_amdgcn_s_memtime())
#define EXPF(x) __builtin_expf(x)
#define LOGF(x) __builtin_logf(x)
#define RSQRTF(x) __builtin_amdgcn_rsqf(x)
#define FMAF(a, b, c) __builtin_fmaf((a), (b), (c))
#endif

#define VALUE_DIM (V_HEADS * DV)
#define CONV_DIM (2 * KEY_DIM + VALUE_DIM)
#define Z_OFF CONV_DIM
#define B_OFF (Z_OFF + VALUE_DIM)
#define A_OFF (B_OFF + V_HEADS)
#define RATIO (V_HEADS / (KEY_DIM / DK))
#define PADX(x) ((x) + (((x) >> 5) << 2))  // LDS vector index with 4 floats of pad per 32
#define PADN(n) ((n) + ((n) >> 3))

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
__device__ __forceinline__ float bfv(u16 v) { return __uint_as_float((u32)v << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + EXPF(-x)); }

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
    return __int_as_float(DPP_I(__float_as_int(v), CTRL));
}

// Sum over each aligned group of LPC lanes (LPC in 1, 2, 4, 8, 16); every lane gets its sum.
// Pairs are symmetric, so every lane of a group holds the same bits.
template <int LPC>
__device__ __forceinline__ float group_sum(float v) {
    if (LPC >= 2) v += dpp<0xB1>(v);   // quad_perm [1,0,3,2]
    if (LPC >= 4) v += dpp<0x4E>(v);   // quad_perm [2,3,0,1]
    if (LPC == 8) v += dpp<0x141>(v);  // row_half_mirror: the other quad of the 8
    if (LPC >= 16) {
        v += dpp<0x124>(v);  // row_ror:4
        v += dpp<0x128>(v);  // row_ror:8
    }
    return v;
}

// Sum over the 64 lanes of a wave, uniform result.
__device__ __forceinline__ float wave_sum(float v) {
    v = group_sum<16>(v);
    float a = __int_as_float(READLANE_I(__float_as_int(v), 0));
    float b = __int_as_float(READLANE_I(__float_as_int(v), 16));
    float c = __int_as_float(READLANE_I(__float_as_int(v), 32));
    float d = __int_as_float(READLANE_I(__float_as_int(v), 48));
    return (a + b) + (c + d);
}

__device__ __forceinline__ void ring_store(u16* p, u16 s1, u16 s2, u16 s3, u16 x) {
    uint2 w;
    w.x = (u32)s1 | ((u32)s2 << 16);
    w.y = (u32)s3 | ((u32)x << 16);
    *(uint2*)p = w;
}

// Gated RMSNorm tail of one (row, head) vector o (bf16-rounded, lane owns columns lane and
// lane + 64), today's roundings; one wave.
__device__ __forceinline__ void gnorm_tail(float o0, float o1, const u16* __restrict__ zr,
                                           const float* __restrict__ nw, u16* __restrict__ outp,
                                           float eps, int lane) {
    const float ss = wave_sum(o0 * o0 + o1 * o1);
    const float inv = RSQRTF(ss / (float)DV + eps);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int c = lane + 64 * q;
        const float o = q ? o1 : o0;
        const float n = bf16r(o * inv);
        const float y = bf16r(nw[c] * n);
        const float z = bfv(zr[c]);
        outp[c] = bf16bits(y * z * sigm(z));
    }
}

// VS: value-column splits per head (grid z). VS == 1: the norm runs in the kernel. VS > 1
// (HPW == 1, CONV == 1): the pre-norm o_j (bf16-rounded) goes to ``obuf`` [rows, V_HEADS, DV]
// and ``gdn_post`` runs the norm and the q / k conv ring.
template <int LPC, int CPL, int HPW, int CONV, int VS, int G, int PROF>
__device__ __forceinline__ void gdn_chain_impl(
    u16* __restrict__ conv, float* __restrict__ rec, const i64* __restrict__ slots,
    const u16* __restrict__ proj, const float* __restrict__ cw, const float* __restrict__ alog,
    const float* __restrict__ dtb, const float* __restrict__ nw, u16* __restrict__ core,
    float* __restrict__ obuf, i64* __restrict__ prof, i64 N, i64 stride_in, i64 stride_cslot,
    i64 stride_slot, float eps, float scale) {
    constexpr int R = DK / LPC;
    constexpr int CW = (64 / LPC) * CPL;
    constexpr int VW = DV / VS;  // value columns of the block (per head)
    constexpr int WPH = VW / CW;
    constexpr int NW = WPH * HPW;
    constexpr int NT = NW * 64;
    constexpr int NACC = R < 4 ? R : 4;
    constexpr int CH = 2 * DK + HPW * VW;
    constexpr int ITER = (CH + NT - 1) / NT;
    static_assert(VW % CW == 0 && R % 4 == 0 && HPW <= RATIO, "layout");
    static_assert(VS == 1 || (HPW == 1 && CONV == 1), "split");

    __shared__ __attribute__((aligned(16))) float s_q[G][PADN(DK)];
    __shared__ __attribute__((aligned(16))) float s_k[G][PADN(DK)];
    __shared__ __attribute__((aligned(16))) float s_v[HPW][G][DV];
    __shared__ float s_o[VS == 1 ? HPW : 1][G][VS == 1 ? DV : 1];
    __shared__ float s_beta[HPW][G];
    __shared__ float s_dec[HPW][G];

    const int tid = threadIdx.x;
    const i64 r = blockIdx.x;
    const int head0 = blockIdx.y * HPW;
    const int k_head = head0 / RATIO;
    const int cb = blockIdx.z * VW;
    i64 t0 = 0, c0 = 0;
    if (PROF && tid == 0) { t0 = REALTIME(); c0 = CLOCK(); }
    const i64 first = slots[r];
    i64 dsts[G];
#pragma unroll
    for (int j = 0; j < G; ++j) dsts[j] = slots[j * N + r];

    // state tile first: its loads overlap the prologue
    const int wave = tid >> 6, lane = tid & 63;
    const int h = wave / WPH, slice = wave % WPH, head = head0 + h;
    const int grp = lane / LPC, t = lane % LPC;
    const int col0 = cb + slice * CW + grp * CPL;
    const int row0 = t * R;
    const i64 hoff = (i64)head * DK * DV;
    float S[R][CPL];
    {
        const float* src = rec + first * stride_slot + hoff + (i64)row0 * DV + col0;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if constexpr (CPL == 4) {
                const float4 v4 = *(const float4*)(src + (i64)i * DV);
                S[i][0] = v4.x; S[i][1] = v4.y; S[i][2] = v4.z; S[i][3] = v4.w;
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) S[i][c] = src[(i64)i * DV + c];
            }
        }
    }

    // ---------------------------------------------------------------- prologue: conv, gates
    // every load first (window, weights, the G inputs of each of the thread's channels), then
    // the math, the LDS stores and the ring stores.
    uint2 win[ITER];
    float4 wv[ITER];
    u16 xs[ITER][G];
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
        const int c = tid + it * NT;
        if (c < CH) {
            int gch;
            if (c < DK) gch = k_head * DK + c;
            else if (c < 2 * DK) gch = KEY_DIM + k_head * DK + (c - DK);
            else gch = 2 * KEY_DIM + (head0 + (c - 2 * DK) / VW) * DV + cb + (c - 2 * DK) % VW;
            if (CONV) {
                win[it] = *(const uint2*)(conv + first * stride_cslot + (i64)gch * 4);
                wv[it] = ((const float4*)cw)[gch];
            }
#pragma unroll
            for (int j = 0; j < G; ++j) xs[it][j] = proj[(j * N + r) * stride_in + gch];
        }
    }
    float gb = 0.0f, ga = 0.0f, galog = 0.0f;
    if (tid < HPW * G) {
        const int hh = tid / G, j = tid % G, hd = head0 + hh;
        const i64 row = j * N + r;
        gb = bfv(proj[row * stride_in + B_OFF + hd]);
        ga = bfv(proj[row * stride_in + A_OFF + hd]) + dtb[hd];
        galog = alog[hd];
    }
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
        const int c = tid + it * NT;
        if (c < CH) {
            int gch;
            float* dst;
            int dstride;
            bool own;  // this block writes the channel's ring rows
            if (c < DK) {
                gch = k_head * DK + c;
                dst = &s_q[0][PADX(c)];
                dstride = PADN(DK);
                own = HPW == RATIO;
            } else if (c < 2 * DK) {
                gch = KEY_DIM + k_head * DK + (c - DK);
                dst = &s_k[0][PADX(c - DK)];
                dstride = PADN(DK);
                own = HPW == RATIO;
            } else {
                const int hh = (c - 2 * DK) / VW, cc = cb + (c - 2 * DK) % VW;
                gch = 2 * KEY_DIM + (head0 + hh) * DV + cc;
                dst = &s_v[hh][0][cc];
                dstride = DV;
                own = true;
            }
            if (CONV) {
                u16 s1 = (u16)(win[it].x >> 16), s2 = (u16)(win[it].y & 0xFFFFu), s3 = (u16)(win[it].y >> 16);
                const float4 w = wv[it];
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    const u16 x = xs[it][j];
                    float acc = bfv(s1) * w.x;
                    acc = FMAF(bfv(s2), w.y, acc);
                    acc = FMAF(bfv(s3), w.z, acc);
                    acc = FMAF(bfv(x), w.w, acc);
                    float y = bf16r(acc);
                    y = bf16r(y * sigm(y));
                    dst[j * dstride] = y;
                    if (own) ring_store(conv + dsts[j] * stride_cslot + (i64)gch * 4, s1, s2, s3, x);
                    s1 = s2;
                    s2 = s3;
                    s3 = x;
                }
            } else {
#pragma unroll
                for (int j = 0; j < G; ++j) dst[j * dstride] = bfv(xs[it][j]);
            }
        }
    }
    if (tid < HPW * G) {
        const int hh = tid / G, j = tid % G;
        s_beta[hh][j] = bf16r(sigm(gb));
        const float sp = ga > 20.0f ? ga : LOGF(1.0f + EXPF(ga));
        const float g = -EXPF(galog) * sp;
        s_dec[hh][j] = EXPF(g);
    }
    __syncthreads();

    // l2 norm of q_j, k_j: one wave per vector, lane owns elements lane and lane + 64
    for (int e = wave; e < 2 * G; e += NW) {
        float* vec = e < G ? s_q[e] : s_k[e - G];
        const float x0 = vec[PADX(lane)], x1 = vec[PADX(lane + 64)];
        const float ss = wave_sum(x0 * x0 + x1 * x1);
        const float inv = RSQRTF(ss + 1e-6f);
        if (e < G) {
            vec[PADX(lane)] = x0 * inv * scale;
            vec[PADX(lane + 64)] = x1 * inv * scale;
        } else {
            vec[PADX(lane)] = x0 * inv;
            vec[PADX(lane + 64)] = x1 * inv;
        }
    }
    __syncthreads();
    i64 t1 = 0, c1 = 0;
    if (PROF && tid == 0) { t1 = REALTIME(); c1 = CLOCK(); }

    // ---------------------------------------------------------------- chain (no barrier)
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const float dec = s_dec[h][j], beta = s_beta[h][j];
        float kr[R], qr[R], vv[CPL];
#pragma unroll
        for (int i = 0; i < R; i += 4) {
            const float4 k4 = *(const float4*)&s_k[j][PADX(row0 + i)];
            const float4 q4 = *(const float4*)&s_q[j][PADX(row0 + i)];
            kr[i] = k4.x; kr[i + 1] = k4.y; kr[i + 2] = k4.z; kr[i + 3] = k4.w;
            qr[i] = q4.x; qr[i + 1] = q4.y; qr[i + 2] = q4.z; qr[i + 3] = q4.w;
        }
#pragma unroll
        for (int c = 0; c < CPL; ++c) vv[c] = s_v[h][j][col0 + c];

        float kv[CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            float acc[NACC];
#pragma unroll
            for (int a = 0; a < NACC; ++a) acc[a] = 0.0f;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                S[i][c] *= dec;
                acc[i % NACC] = FMAF(S[i][c], kr[i], acc[i % NACC]);
            }
            kv[c] = (acc[0] + acc[1]) + (acc[2] + acc[NACC - 1]);
        }
#pragma unroll
        for (int c = 0; c < CPL; ++c) kv[c] = group_sum<LPC>(kv[c]);
        float* out = rec + dsts[j] * stride_slot + hoff + (i64)row0 * DV + col0;
        float ov[CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            const float delta = (vv[c] - kv[c]) * beta;
            float acc[NACC];
#pragma unroll
            for (int a = 0; a < NACC; ++a) acc[a] = 0.0f;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                S[i][c] = FMAF(kr[i], delta, S[i][c]);
                acc[i % NACC] = FMAF(S[i][c], qr[i], acc[i % NACC]);
            }
            ov[c] = (acc[0] + acc[1]) + (acc[2] + acc[NACC - 1]);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if constexpr (CPL == 4) {
                float4 v4;
                v4.x = S[i][0]; v4.y = S[i][1]; v4.z = S[i][2]; v4.w = S[i][3];
                *(float4*)(out + (i64)i * DV) = v4;
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) out[(i64)i * DV + c] = S[i][c];
            }
        }
#pragma unroll
        for (int c = 0; c < CPL; ++c) ov[c] = group_sum<LPC>(ov[c]);
        if (t == 0) {
            if constexpr (VS == 1) {
#pragma unroll
                for (int c = 0; c < CPL; ++c) s_o[h][j][col0 + c] = bf16r(ov[c]);
            } else {
                float* ob = obuf + ((j * N + r) * V_HEADS + head) * DV + col0;
#pragma unroll
                for (int c = 0; c < CPL; ++c) ob[c] = bf16r(ov[c]);
            }
        }
    }
    i64 t2 = 0, c2 = 0;
    if (PROF && tid == 0) { t2 = REALTIME(); c2 = CLOCK(); }

    // ---------------------------------------------------------------- epilogue: gated norm
    if constexpr (VS == 1) {
        __syncthreads();
        for (int e = wave; e < HPW * G; e += NW) {
            const int hh = e / G, j = e % G, hd = head0 + hh;
            const i64 row = j * N + r;
            gnorm_tail(s_o[hh][j][lane], s_o[hh][j][lane + 64], proj + row * stride_in + Z_OFF + hd * DV,
                       nw, core + (row * V_HEADS + hd) * DV, eps, lane);
        }
    }
    if (PROF && tid == 0) {
        const i64 t3 = REALTIME(), c3 = CLOCK();
        i64* p = prof + ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 8;
        p[0] = t1 - t0; p[1] = t2 - t1; p[2] = t3 - t2; p[3] = t3 - t0;
        p[4] = c1 - c0; p[5] = c2 - c1; p[6] = c3 - c2; p[7] = c3 - c0;
    }
}

// q / k conv ring rows of slot r, channels ch (CONV = 1, HPW < RATIO).
template <int G>
__device__ __forceinline__ void qk_ring_one(u16* __restrict__ conv, const i64* __restrict__ slots,
                                            const u16* __restrict__ proj, i64 N, i64 stride_in,
                                            i64 stride_cslot, i64 r, int ch) {
    const uint2 w = *(const uint2*)(conv + slots[r] * stride_cslot + (i64)ch * 4);
    u16 s1 = (u16)(w.x >> 16), s2 = (u16)(w.y & 0xFFFFu), s3 = (u16)(w.y >> 16);
    u16 xs[G];
    i64 d[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        xs[j] = proj[(j * N + r) * stride_in + ch];
        d[j] = slots[j * N + r];
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        ring_store(conv + d[j] * stride_cslot + (i64)ch * 4, s1, s2, s3, xs[j]);
        s1 = s2;
        s2 = s3;
        s3 = xs[j];
    }
}

// grid (N, 2 KEY_DIM / 256), 256 threads.
template <int G>
__device__ __forceinline__ void gdn_qk_ring_impl(u16* __restrict__ conv, const i64* __restrict__ slots,
                                                 const u16* __restrict__ proj, i64 N, i64 stride_in,
                                                 i64 stride_cslot) {
    const int ch = blockIdx.y * 256 + threadIdx.x;
    if (ch < 2 * KEY_DIM) qk_ring_one<G>(conv, slots, proj, N, stride_in, stride_cslot, blockIdx.x, ch);
}

// After a VS > 1 chain: blocks 0 .. NBN-1 run the gated norm, one wave per (row, head) of
// ``obuf``; the next N * (2 KEY_DIM / 256) blocks write the q / k conv ring rows. 256 threads.
template <int G>
__device__ __forceinline__ void gdn_post_impl(const float* __restrict__ obuf, const u16* __restrict__ proj,
                                              const float* __restrict__ nw, u16* __restrict__ core,
                                              u16* __restrict__ conv, const i64* __restrict__ slots,
                                              i64 N, i64 stride_in, i64 stride_cslot, float eps) {
    const i64 vecs = N * G * V_HEADS;
    const i64 nbn = (vecs + 3) / 4;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    if ((i64)blockIdx.x < nbn) {
        const i64 e = (i64)blockIdx.x * 4 + wave;
        if (e < vecs) {
            const i64 row = e / V_HEADS;
            const int hd = (int)(e % V_HEADS);
            const float* o = obuf + e * DV;
            gnorm_tail(o[lane], o[lane + 64], proj + row * stride_in + Z_OFF + hd * DV, nw,
                       core + e * DV, eps, lane);
        }
        return;
    }
    constexpr int QKB = (2 * KEY_DIM + 255) / 256;
    const i64 b = blockIdx.x - nbn;
    const int ch = (int)(b % QKB) * 256 + tid;
    if (ch < 2 * KEY_DIM) qk_ring_one<G>(conv, slots, proj, N, stride_in, stride_cslot, b / QKB, ch);
}
