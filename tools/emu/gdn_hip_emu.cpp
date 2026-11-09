// Plan 50: CPU logic test of engine/fast/hip/gdn_chain.hip (every layout / conv mode) against a
// float64 reference of the step, small dims (KEY_DIM 256, V_HEADS 4, DK = DV = 128).
//   g++ -std=c++20 -O1 -pthread -DHOST_EMU -DKEY_DIM=256 -DV_HEADS=4 -DDK=128 -DDV=128 \
//       -Itools/emu tools/emu/gdn_hip_emu.cpp -o /tmp/gdn_emu && /tmp/gdn_emu
#include "gdn_hip_emu_shim.h"
#include "../../engine/fast/hip/gdn_chain.hip"

#include <cstdio>
#include <functional>
#include <random>
#include <thread>

using Kern = std::function<void()>;

static void launch(int gx, int gy, int nt, const Kern& k, int gz = 1) {
    gridDim.x = gx; gridDim.y = gy; gridDim.z = gz; blockDim.x = nt;
    for (int bz = 0; bz < gz; ++bz)
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx) {
            std::barrier<> blk(nt);
            std::vector<std::barrier<>*> waves;
            for (int w = 0; w < (nt + 63) / 64; ++w) waves.push_back(new std::barrier<>(64));
            g_emu.blk = &blk;
            g_emu.wave = waves;
            g_emu.xch.assign(nt, 0);
            std::vector<std::thread> th;
            for (int t = 0; t < nt; ++t)
                th.emplace_back([&, t] { threadIdx.x = t; blockIdx.x = bx; blockIdx.y = by; blockIdx.z = bz; k(); });
            for (auto& x : th) x.join();
            for (auto* w : waves) delete w;
        }
}

static u16 tobf(double d) { return bf16bits((float)d); }
static double bfd(double d) { return (double)bfv(tobf(d)); }
static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Case {
    int N, G;
    i64 width = CONV_DIM + VALUE_DIM + 2 * V_HEADS;
    int ringrows;
    std::vector<u16> conv, proj, raw;
    std::vector<float> rec, cw, alog, dtb, nw;
    std::vector<i64> slots;
    std::vector<u16> core;
    std::vector<i64> prof;
};

static Case make(int N, int G, unsigned seed) {
    Case c;
    c.N = N; c.G = G;
    c.ringrows = (N + 1) * G;
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(1, 16);
    c.conv.resize((size_t)c.ringrows * CONV_DIM * 4);
    for (auto& v : c.conv) v = tobf(nd(rng));
    c.rec.resize((size_t)c.ringrows * V_HEADS * DK * DV);
    for (auto& v : c.rec) v = (float)(0.05 * nd(rng));
    c.raw.resize((size_t)N * G * c.width);
    for (auto& v : c.raw) v = tobf(nd(rng));
    c.proj = c.raw;
    c.cw.resize((size_t)CONV_DIM * 4);
    for (auto& v : c.cw) v = bfv(tobf(0.3 * nd(rng)));
    for (int h = 0; h < V_HEADS; ++h) {
        c.alog.push_back((float)std::log(ud(rng)));
        c.dtb.push_back((float)(0.5 * nd(rng) + 1));
    }
    for (int i = 0; i < DV; ++i) c.nw.push_back(bfv(tobf(1 + 0.1 * nd(rng))));
    for (int j = 0; j < G; ++j)
        for (int r = 0; r < N; ++r) c.slots.push_back(r + (i64)j * (N + 1));  // entry 0 == first
    c.core.assign((size_t)N * G * VALUE_DIM, 0);
    c.prof.assign(4096, 0);
    return c;
}

// float64 reference of one step from the case's initial state; returns (states, out, rings)
struct Ref {
    std::vector<double> state;  // [G][N][H][DK][DV]
    std::vector<double> out;    // [G*N rows][H*DV]
    std::vector<u16> ring;      // [G][N][CONV_DIM][4]
    std::vector<u16> convout;   // [G*N][CONV_DIM] (bf16, the kernel-path roundings)
};

static Ref reference(const Case& c, double eps) {
    const int N = c.N, G = c.G;
    Ref R;
    R.state.assign((size_t)G * N * V_HEADS * DK * DV, 0);
    R.out.assign((size_t)G * N * VALUE_DIM, 0);
    R.ring.assign((size_t)G * N * CONV_DIM * 4, 0);
    R.convout.assign((size_t)G * N * CONV_DIM, 0);
    std::vector<double> y((size_t)G * N * CONV_DIM);
    for (int r = 0; r < N; ++r)
        for (int ch = 0; ch < CONV_DIM; ++ch) {
            const u16* st = &c.conv[((size_t)c.slots[r] * CONV_DIM + ch) * 4];
            u16 w[4] = {st[1], st[2], st[3], 0};
            for (int j = 0; j < G; ++j) {
                const i64 row = (i64)j * N + r;
                w[3] = c.raw[row * c.width + ch];
                double acc = 0;
                for (int q = 0; q < 4; ++q) acc += (double)bfv(w[q]) * c.cw[ch * 4 + q];
                double v = bfd(acc);
                v = bfd(v * sig(v));
                y[row * CONV_DIM + ch] = v;
                R.convout[row * CONV_DIM + ch] = tobf(v);
                for (int q = 0; q < 4; ++q) R.ring[(((size_t)j * N + r) * CONV_DIM + ch) * 4 + q] = w[q];
                w[0] = w[1]; w[1] = w[2]; w[2] = w[3];
            }
        }
    const double scale = 1.0 / std::sqrt((double)DK);
    for (int r = 0; r < N; ++r)
        for (int h = 0; h < V_HEADS; ++h) {
            const int kh = h / RATIO;
            std::vector<double> S((size_t)DK * DV);
            const float* s0 = &c.rec[((size_t)c.slots[r] * V_HEADS + h) * DK * DV];
            for (int i = 0; i < DK * DV; ++i) S[i] = s0[i];
            for (int j = 0; j < G; ++j) {
                const i64 row = (i64)j * N + r;
                const double* yr = &y[row * CONV_DIM];
                std::vector<double> q(DK), k(DK), v(DV), o(DV, 0), kv(DV, 0);
                double sq = 0, sk = 0;
                for (int i = 0; i < DK; ++i) {
                    q[i] = yr[kh * DK + i]; k[i] = yr[KEY_DIM + kh * DK + i];
                    sq += q[i] * q[i]; sk += k[i] * k[i];
                }
                for (int i = 0; i < DK; ++i) { q[i] *= scale / std::sqrt(sq + 1e-6); k[i] /= std::sqrt(sk + 1e-6); }
                for (int i = 0; i < DV; ++i) v[i] = yr[2 * KEY_DIM + h * DV + i];
                const u16* pr = &c.raw[row * c.width];
                const double beta = bfd(sig(bfv(pr[B_OFF + h])));
                const double a = (double)bfv(pr[A_OFF + h]) + c.dtb[h];
                const double sp = a > 20 ? a : std::log1p(std::exp(a));
                const double dec = std::exp(-std::exp((double)c.alog[h]) * sp);
                for (auto& s : S) s *= dec;
                for (int i = 0; i < DK; ++i)
                    for (int cc = 0; cc < DV; ++cc) kv[cc] += S[i * DV + cc] * k[i];
                for (int i = 0; i < DK; ++i)
                    for (int cc = 0; cc < DV; ++cc) S[i * DV + cc] += k[i] * (v[cc] - kv[cc]) * beta;
                for (int i = 0; i < DK; ++i)
                    for (int cc = 0; cc < DV; ++cc) o[cc] += S[i * DV + cc] * q[i];
                std::copy(S.begin(), S.end(), R.state.begin() + (((size_t)j * N + r) * V_HEADS + h) * DK * DV);
                double ss = 0;
                for (auto& x : o) { x = bfd(x); ss += x * x; }
                const double inv = 1.0 / std::sqrt(ss / DV + eps);
                for (int cc = 0; cc < DV; ++cc) {
                    const double n = bfd(o[cc] * inv);
                    const double yy = bfd(c.nw[cc] * n);
                    const double z = bfv(pr[Z_OFF + h * DV + cc]);
                    R.out[row * VALUE_DIM + h * DV + cc] = yy * z * sig(z);
                }
            }
        }
    return R;
}

template <int LPC, int CPL, int HPW, int CONV, int G, int VS = 1>
static int run(const char* name, int N, unsigned seed) {
    Case c = make(N, G, seed);
    const double eps = 1e-6;
    const Ref R = reference(c, eps);
    if (!CONV) {  // today's Triton conv kernel would have run: convolved q|k|v in place + ring
        for (int row = 0; row < N * G; ++row)
            for (int ch = 0; ch < CONV_DIM; ++ch) c.proj[(size_t)row * c.width + ch] = R.convout[(size_t)row * CONV_DIM + ch];
    }
    constexpr int NT = (DV / VS / ((64 / LPC) * CPL)) * HPW * 64;
    std::vector<float> obuf((size_t)N * G * VALUE_DIM, 0.0f);
    launch(N, V_HEADS / HPW, NT, [&] {
        gdn_chain_impl<LPC, CPL, HPW, CONV, VS, G, 1>(c.conv.data(), c.rec.data(), c.slots.data(), c.proj.data(),
                                                  c.cw.data(), c.alog.data(), c.dtb.data(), c.nw.data(),
                                                  c.core.data(), obuf.data(), c.prof.data(), N, c.width, (i64)CONV_DIM * 4,
                                                  (i64)V_HEADS * DK * DV, (float)eps, 1.0f / std::sqrt((float)DK));
    }, VS);
    if (VS > 1) {
        const int blocks = (N * G * V_HEADS + 3) / 4 + N * ((2 * KEY_DIM + 255) / 256);
        launch(blocks, 1, 256, [&] {
            gdn_post_impl<G>(obuf.data(), c.proj.data(), c.nw.data(), c.core.data(), c.conv.data(), c.slots.data(),
                             N, c.width, (i64)CONV_DIM * 4, (float)eps);
        });
    } else if (CONV && HPW < RATIO)
        launch(N, (2 * KEY_DIM + 255) / 256, 256, [&] {
            gdn_qk_ring_impl<G>(c.conv.data(), c.slots.data(), c.proj.data(), N, c.width, (i64)CONV_DIM * 4);
        });
    double smax = 0, serr = 0, oerr = 0, omax = 0;
    long omis = 0, rmis = 0;
    for (int j = 0; j < G; ++j)
        for (int r = 0; r < N; ++r) {
            const i64 dst = c.slots[(size_t)j * N + r];
            for (int i = 0; i < V_HEADS * DK * DV; ++i) {
                const double a = c.rec[(size_t)dst * V_HEADS * DK * DV + i];
                const double b = R.state[((size_t)j * N + r) * V_HEADS * DK * DV + i];
                smax = std::max(smax, std::fabs(b));
                serr = std::max(serr, std::fabs(a - b));
            }
            if (CONV)
                for (int i = 0; i < CONV_DIM * 4; ++i)
                    rmis += c.conv[(size_t)dst * CONV_DIM * 4 + i] != R.ring[((size_t)j * N + r) * CONV_DIM * 4 + i];
        }
    for (size_t i = 0; i < R.out.size(); ++i) {
        const double a = bfv(c.core[i]);
        omax = std::max(omax, std::fabs(R.out[i]));
        oerr = std::max(oerr, std::fabs(a - R.out[i]));
        omis += c.core[i] != tobf(R.out[i]);
    }
    const bool ok = serr / smax < 1e-5 && oerr / omax < 2e-2 && rmis == 0 && omis < (long)R.out.size() / 20;
    std::printf("%-14s N=%d G=%d NT=%d state_maxrel=%.2e out_maxrel=%.2e out_bf16_mismatch=%ld/%zu ring_mismatch=%ld %s\n",
                name, N, G, NT, serr / smax, oerr / omax, omis, R.out.size(), rmis, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int bad = 0;
    bad += run<4, 1, 1, 1, 4>("L4 f1", 2, 1);
    bad += run<4, 1, 2, 1, 4>("L4 f2", 2, 2);
    bad += run<4, 1, 1, 0, 4>("L4 sep", 2, 3);
    bad += run<16, 4, 2, 1, 4>("L16x4 f2", 2, 4);
    bad += run<16, 4, 1, 1, 2>("L16x4 f1", 1, 5);
    bad += run<2, 1, 2, 1, 4>("L2 f2", 1, 6);
    bad += run<8, 4, 1, 0, 2>("L8x4 sep", 2, 7);
    bad += run<8, 4, 2, 1, 4>("L8x4 f2", 2, 8);
    bad += run<4, 4, 2, 1, 4>("L4x4 f2", 1, 9);
    bad += run<4, 4, 1, 1, 4, 2>("L4x4 v2", 2, 10);
    bad += run<2, 1, 1, 1, 4, 4>("L2 v4", 2, 11);
    bad += run<16, 4, 1, 1, 2, 8>("L16x4 v8", 1, 12);
    bad += run<8, 4, 1, 1, 4, 2>("L8x4 v2", 2, 13);
    bad += run<4, 1, 1, 1, 2, 4>("L4 v4", 2, 14);
    std::printf("%s\n", bad ? "EMU FAIL" : "EMU OK");
    return bad;
}
