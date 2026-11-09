// Plan 50: CPU emulation of the few HIP features engine/fast/hip/gdn_chain.hip uses, for a
// logic test with g++ (tools/emu/gdn_hip_emu.cpp): one std::thread per lane, one block at a
// time, __shared__ as function statics, DPP / readlane through a per-wave exchange + barrier.
#pragma once
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;

struct Dim3 { unsigned x = 0, y = 0, z = 0; };
inline thread_local Dim3 threadIdx, blockIdx;
inline Dim3 gridDim, blockDim;
struct float4 { float x, y, z, w; };
struct uint2 { u32 x, y; };

#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(n)

inline u32 __float_as_uint(float f) { u32 u; std::memcpy(&u, &f, 4); return u; }
inline float __uint_as_float(u32 u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int __float_as_int(float f) { int u; std::memcpy(&u, &f, 4); return u; }
inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }

struct EmuBlock {
    std::barrier<>* blk = nullptr;
    std::vector<std::barrier<>*> wave;
    std::vector<int> xch;
};
inline EmuBlock g_emu;

inline void __syncthreads() { g_emu.blk->arrive_and_wait(); }

inline int emu_src_lane(int l, int ctrl) {
    const int row = l & ~15, i = l & 15;
    if (ctrl <= 0xFF) return (l & ~3) | ((ctrl >> (2 * (l & 3))) & 3);
    if (ctrl >= 0x121 && ctrl <= 0x12F) return row | ((i + (ctrl - 0x120)) & 15);
    if (ctrl == 0x140) return row | (15 - i);
    if (ctrl == 0x141) return row | (i & 8) | (7 - (i & 7));
    std::abort();
}
inline int emu_exchange(int v, int src_lane) {
    const int tid = threadIdx.x, w = tid / 64;
    g_emu.xch[tid] = v;
    g_emu.wave[w]->arrive_and_wait();
    const int r = g_emu.xch[w * 64 + src_lane];
    g_emu.wave[w]->arrive_and_wait();
    return r;
}
#define DPP_I(v, ctrl) emu_exchange((v), emu_src_lane(threadIdx.x & 63, (ctrl)))
#define READLANE_I(v, l) emu_exchange((v), (l))
#define REALTIME() ((i64)0)
#define CLOCK() ((i64)0)
#define EXPF(x) std::exp((float)(x))
#define LOGF(x) std::log((float)(x))
#define RSQRTF(x) (1.0f / std::sqrt((float)(x)))
#define FMAF(a, b, c) std::fma((float)(a), (float)(b), (float)(c))
