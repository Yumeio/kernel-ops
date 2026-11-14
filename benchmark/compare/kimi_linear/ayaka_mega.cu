// ayaka_mega.cu -- Kimi-Linear W4A8 hybrid decode step as ONE persistent CUDA kernel.
//
// Same job as kimi_mega_triton.py (3x(KDA+MoE) + 1x(MLA+MoE) per launch), written
// against what that kernel taught us and what a hand-written CUDA kernel can do
// that a Triton program cannot:
//
//  * The worker is the WARP, not the CTA.  One CTA per SM, 8 warps; every stage
//    hands each of the P*8 warps a contiguous range of work.  A GEMV unit is
//    128 output columns x one 128-element contraction group == 17 rows of
//    512 B ({bf16 scale, int8 zero} row + 16 int4 rows), 16 B per lane per load.
//    D = 2304 = 18 * 128, so nothing is padded and any grid size balances.
//  * Activations live in shared memory.  rmsnorm + int8 quantisation run once per
//    CTA with warp shuffles (a 128-group is one warp: 4 elements per lane), and
//    the GEMV reads its packed int8 words from smem instead of global scratch.
//  * Every GEMV accumulates with red.global.add.f32 into a pre-zeroed / residual
//    seeded stream, so there are no split-k partial slots and no reduce passes.
//  * 26 grid barriers instead of 33, each with half the arrivals: conv+delta-rule
//    are one stage, and activation+quantise+down_proj are one stage.
//  * MLA attention uses mma.sync (bf16) with the cache staged through smem by
//    cp.async; the P fragment is re-used directly as the A operand of P^T * C.
//
// Cross-CTA memory rule: scratch written by one CTA and read by another is always
// read with ld.global.cg (L2), every thread fences before a barrier arrival, and
// atomically-accumulated words are zeroed/seeded at least one barrier earlier.
//
// Build: nvcc -O3 -arch=sm_90a -shared -Xcompiler -fPIC -o libayaka_mega.so ayaka_mega.cu
#include <cuda_runtime.h>
#include <cstdint>

#define FI __device__ __forceinline__

namespace {
constexpr int D = 2304, H = 32, C4 = 4096, KVL = 512, QKN = 128, QKR = 64, QH = 192;
constexpr int MI = 1024, NEXP = 64;
constexpr int NW = 8, TPB = NW * 32;
constexpr int UR = 17 * 32;  // uint4 per unit (17 rows x 512 B)

// ---- weight arena, in units ----
constexpr int U_QKVG = 128 * 18, U_OP = 18 * 32, U_KDA = U_QKVG + U_OP;
constexpr int QKVA_NT = 53, U_QKVA = QKVA_NT * 18, U_KVBN = 128, U_KVBV = 128;
constexpr int U_MLA = U_QKVA + U_KVBN + U_KVBV + U_OP;
constexpr int U_GU = 16 * 18, U_DN = 18 * 8, U_EXP = U_GU + U_DN, U_MOE = (NEXP + 1) * U_EXP;
constexpr int A_KDA = 0, A_MLA = 3 * U_KDA, A_MOE = A_MLA + U_MLA;
constexpr int U_TOT = A_MOE + 4 * U_MOE;

// ---- bf16 side arena (same layout as the Triton kernel) ----
constexpr int WB_CW = 0, WB_BW = WB_CW + 3 * 3 * 4 * C4, WB_RW = WB_BW + 3 * H * D;
constexpr int WB_NRM = WB_RW + 4 * NEXP * D, WB_TOT = WB_NRM + 4 * 2 * D;

// ---- fp32 constants ----
constexpr int FC_INVF = 0, FC_SN8 = 64, FC_SNZ = FC_SN8 + H * 4 * 128, FC_TOT = FC_SNZ + H * 4 * 128;

// ---- fp32 scratch ----
constexpr int PMAX = 256;
constexpr int SC_ACCQ = 0;                          // [3][16384] q|k|v|g accumulators
constexpr int SC_HA = SC_ACCQ + 3 * 4 * C4;         // [4][2304]  hidden after attention
constexpr int SC_HM = SC_HA + 4 * D;                // [4][2304]  hidden after moe
constexpr int SC_O1 = SC_HM + 4 * D;                // [3][4096]  kda attention output
constexpr int SC_BETA = SC_O1 + 3 * C4;             // [3][64]
constexpr int SC_RLOG = SC_BETA + 3 * 64;           // [4][64]
constexpr int SC_ACCGU = SC_RLOG + 4 * 64;          // [4][9][2048] gate|up accumulators
constexpr int SC_ACCQA = SC_ACCGU + 4 * 9 * 2 * MI; // [53*128]   mla q_proj|kv_a accumulators
constexpr int SC_O1M = SC_ACCQA + QKVA_NT * 128;    // [4096]     mla attention output
constexpr int SC_AM = SC_O1M + C4;                  // [PMAX][32] per-chunk running max
constexpr int SC_AL = SC_AM + PMAX * H;             // [PMAX][32] per-chunk sumexp
constexpr int SC_TOT = SC_AL + PMAX * H;

// ---- bf16 scratch ----
constexpr int QROW = KVL + QKR;                     // 576: latent | rope
constexpr int CB_QAR = 0;                           // [32][576]  absorbed query | rope'd query
constexpr int CB_APC = CB_QAR + H * QROW;           // [PMAX][32][512] attention partials
constexpr int CB_TOT = CB_APC + PMAX * H * KVL;

constexpr int S_SLOT = H * 128 * 128, CV_SLOT = 3 * C4;
constexpr float MLA_SCALE = 0.07216878364870322f;   // 192^-0.5
constexpr float KDA_SCALE = 0.08838834764831845f;   // 128^-0.5
constexpr float ROUTED_SCALE = 2.446f;

// ---- shared memory layout (bytes) ----
constexpr int SM_XN = 0;                            // float[2304]   normed activation
constexpr int SM_XQ = SM_XN + D * 4;                // int[32][32]   packed int8 activation
constexpr int SM_XS = SM_XQ + 32 * 32 * 4;          // float2[32]    (scale, sum of int8)
constexpr int SM_RED = SM_XS + 32 * 8;              // float[16]
constexpr int SM_XW = SM_RED + 64;                  // int[8][32]    per-warp activation words
constexpr int SM_LG = SM_XW + NW * 32 * 4;          // float[64]
constexpr int SM_TOPE = SM_LG + 256;                // int[16]
constexpr int SM_TOPW = SM_TOPE + 64;               // float[16]
constexpr int SM_TOPP = SM_TOPW + 64;               // float[16]
constexpr int SM_WSC = SM_TOPP + 64;                // float[8][PMAX]
constexpr int SM_QS = SM_WSC + NW * PMAX * 4;       // bf16[32][576]
constexpr int NBUF = 4;
constexpr int SM_CKS = SM_QS + H * QROW * 2;        // bf16[NBUF][16][576]
constexpr int SM_PS = SM_CKS + NBUF * 16 * QROW * 2;  // float[8][32][16]
constexpr int SM_TOT = SM_PS + NW * 32 * 16 * 4;
}  // namespace

struct Params {
  const uint32_t* WA;   // weight arena [U_TOT][17][128]
  const uint16_t* WB;   // bf16 side weights
  const float* FC;      // fp32 constants
  float* SC;            // fp32 scratch
  uint16_t* CB;         // bf16 scratch
  int* BAR;             // barrier counter
  const uint16_t* HIN;  // [2304] bf16
  uint16_t* HOUT;       // [2304] bf16
  float* SB;            // [3][32][128][128] kda state
  const uint16_t* CVI;  // [9][3][4096] conv state in
  uint16_t* CVO;        // [9][3][4096] conv state out
  uint16_t* CACHE;      // [cap][576] mla cache (latent | rope)
  long long L, bar0, sstop, cw, nch;
};

// ------------------------------------------------------------------------------------------
// primitives
// ------------------------------------------------------------------------------------------
FI float bf(float x) {  // round to bf16 (nearest even), keep the fp32 container
  uint32_t u = __float_as_uint(x);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  return __uint_as_float(u);
}
FI uint32_t bfb(float x) { return __float_as_uint(bf(x)) >> 16; }
FI float b2f(uint32_t b) { return __uint_as_float(b << 16); }
FI uint32_t h16(uint2 r, int i) { return i == 0 ? (r.x & 0xFFFFu) : i == 1 ? (r.x >> 16) : i == 2 ? (r.y & 0xFFFFu) : (r.y >> 16); }

FI float ldcg(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
FI float2 ldcg2(const float* p) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}
FI float4 ldcg4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
FI uint2 ldcg2u(const void* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
FI uint4 ldcg4u(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
FI void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
FI int dp4a_us(unsigned a, int b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
FI float wmaxf(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
FI float wsumf(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
FI int wsumi(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
FI float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
FI float4 bf4(float4 v) { return make_float4(bf(v.x), bf(v.y), bf(v.z), bf(v.w)); }
FI float4 b2f4(uint2 r) { return make_float4(b2f(r.x & 0xFFFFu), b2f(r.x >> 16), b2f(r.y & 0xFFFFu), b2f(r.y >> 16)); }

// grid barrier on one monotonically increasing arrival counter (never reset)
FI void barrier(int* bar, int tgt) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(bar, 1);
    while (*(volatile int*)bar < tgt) {
    }
  }
  __syncthreads();
}

// int8-quantise one 128-group held by one warp (4 elements per lane).
// Returns the packed word of this lane; xs = scale, xsum = sum of the int8 values.
FI int quant_group(float4 x, float& xs, float& xsum) {
  float m = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
  m = wmaxf(m);
  const float sc = 127.0f / (m + 1.0e-30f);
  const int q0 = __float2int_rn(x.x * sc), q1 = __float2int_rn(x.y * sc);
  const int q2 = __float2int_rn(x.z * sc), q3 = __float2int_rn(x.w * sc);
  xs = m * (1.0f / 127.0f);
  xsum = (float)wsumi(q0 + q1 + q2 + q3);
  return (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
}

// one unit: 16 rows of int4 words against 32 packed int8 activation words
FI void unit_dot(const uint4* p, const int* xk, int& a0, int& a1, int& a2, int& a3, uint4& sz) {
  uint4 r[17];
#pragma unroll
  for (int m = 0; m < 17; ++m) r[m] = p[m * 32];
  sz = r[0];
  a0 = a1 = a2 = a3 = 0;
#pragma unroll
  for (int m = 0; m < 16; ++m) {
    const uint4 w = r[m + 1];
    const int xl = xk[2 * m], xh = xk[2 * m + 1];
    a0 = dp4a_us(w.x & 0x0F0F0F0Fu, xl, a0);
    a0 = dp4a_us((w.x >> 4) & 0x0F0F0F0Fu, xh, a0);
    a1 = dp4a_us(w.y & 0x0F0F0F0Fu, xl, a1);
    a1 = dp4a_us((w.y >> 4) & 0x0F0F0F0Fu, xh, a1);
    a2 = dp4a_us(w.z & 0x0F0F0F0Fu, xl, a2);
    a2 = dp4a_us((w.z >> 4) & 0x0F0F0F0Fu, xh, a2);
    a3 = dp4a_us(w.w & 0x0F0F0F0Fu, xl, a3);
    a3 = dp4a_us((w.w >> 4) & 0x0F0F0F0Fu, xh, a3);
  }
}
// sum_k (v-z)*s*x == s*xs*(sum_k v*xq - z*sum_k xq)
FI float epi(uint32_t szw, int a, float f, float sm) {
  return (b2f(szw & 0xFFFFu) * f) * ((float)a - (float)(int)(int8_t)(szw >> 16) * sm);
}
FI void flush4(float* o, float4 v) {
  red_add(o, v.x);
  red_add(o + 1, v.y);
  red_add(o + 2, v.z);
  red_add(o + 3, v.w);
}

// GEMV over units [u0,u1) of one [tile][kg] arena; partials red.add into out[tile*128 + col]
FI void gemv_run(const uint32_t* base, int u0, int u1, int nkg, float* out, const int* XQ,
                 const float2* XS, int lane) {
  if (u0 >= u1) return;
  float4 val = make_float4(0, 0, 0, 0);
  int tcur = u0 / nkg;
  for (int u = u0; u < u1; ++u) {
    const int tile = u / nkg, kg = u - tile * nkg;
    if (tile != tcur) {
      flush4(out + tcur * 128 + lane * 4, val);
      val = make_float4(0, 0, 0, 0);
      tcur = tile;
    }
    int a0, a1, a2, a3;
    uint4 sz;
    unit_dot((const uint4*)base + (size_t)u * UR + lane, XQ + kg * 32, a0, a1, a2, a3, sz);
    const float2 s = XS[kg];
    val.x += epi(sz.x, a0, s.x, s.y);
    val.y += epi(sz.y, a1, s.x, s.y);
    val.z += epi(sz.z, a2, s.x, s.y);
    val.w += epi(sz.w, a3, s.x, s.y);
  }
  flush4(out + tcur * 128 + lane * 4, val);
}

struct Ctx {
  int tid, w, l, pid, P, gw, nwt;
  float* XN;
  int* XQ;
  float2* XS;
  float* RED;
  int* XW;
};

// rmsnorm(h)*w -> smem XN (bf16-rounded), int8-quantised -> XQ/XS.  CTA 0 seeds `seed` with bf(h).
FI void prologue_norm(const Ctx& c, const float* hf, const uint16_t* hb, const uint16_t* wn, float* seed) {
  float4 hv[3];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int g = c.w + 8 * i;
    hv[i] = make_float4(0, 0, 0, 0);
    if (g < 18) {
      const int o = g * 128 + 4 * c.l;
      if (hb) hv[i] = b2f4(ldcg2u(hb + o));
      else hv[i] = bf4(ldcg4(hf + o));
      if (seed && c.pid == 0) *(float4*)(seed + o) = hv[i];
      ss += hv[i].x * hv[i].x + hv[i].y * hv[i].y + hv[i].z * hv[i].z + hv[i].w * hv[i].w;
    }
  }
  ss = wsumf(ss);
  if (c.l == 0) c.RED[c.w] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += c.RED[i];
  const float rs = rsqrtf(tot / 2304.0f + 1.0e-6f);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int g = c.w + 8 * i;
    if (g < 18) {
      const int o = g * 128 + 4 * c.l;
      const float4 wv = b2f4(*(const uint2*)(wn + o));
      float4 x;
      x.x = bf(hv[i].x * rs * wv.x);
      x.y = bf(hv[i].y * rs * wv.y);
      x.z = bf(hv[i].z * rs * wv.z);
      x.w = bf(hv[i].w * rs * wv.w);
      *(float4*)(c.XN + o) = x;
      float xs, xsum;
      c.XQ[g * 32 + c.l] = quant_group(x, xs, xsum);
      if (c.l == 0) c.XS[g] = make_float2(xs, xsum);
    }
  }
  __syncthreads();
}

// quantise 32 groups of a shared fp32 vector (the o_proj inputs) -> XQ/XS
FI void prologue_quant32(const Ctx& c, const float* src) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int g = c.w + 8 * i;
    const float4 x = ldcg4(src + g * 128 + 4 * c.l);
    float xs, xsum;
    c.XQ[g * 32 + c.l] = quant_group(x, xs, xsum);
    if (c.l == 0) c.XS[g] = make_float2(xs, xsum);
  }
  __syncthreads();
}

// <XN, w> over the hidden vector, w bf16; one warp, 72 elements per lane
FI float dot_xn(const Ctx& c, const uint16_t* wrow) {
  float acc = 0.f;
  const int o = c.l * 72;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    const uint4 r = *(const uint4*)(wrow + o + 8 * i);
    const float4 xa = *(const float4*)(c.XN + o + 8 * i), xb = *(const float4*)(c.XN + o + 8 * i + 4);
    acc += xa.x * b2f(r.x & 0xFFFFu) + xa.y * b2f(r.x >> 16) + xa.z * b2f(r.y & 0xFFFFu) + xa.w * b2f(r.y >> 16);
    acc += xb.x * b2f(r.z & 0xFFFFu) + xb.y * b2f(r.z >> 16) + xb.z * b2f(r.w & 0xFFFFu) + xb.w * b2f(r.w >> 16);
  }
  return wsumf(acc);
}

FI void zero128(float* p, int lane) { *(float4*)(p + 4 * lane) = make_float4(0, 0, 0, 0); }

// short causal conv + silu on the 4 channels of this lane (q or k); optionally writes the new taps
FI float4 conv4(const Params& p, int slot, int o, float4 acc, bool wr) {
  const float v[4] = {bf(acc.x), bf(acc.y), bf(acc.z), bf(acc.w)};
  const uint16_t* ci = p.CVI + slot * CV_SLOT + o;
  const uint2 t0 = ldcg2u(ci), t1 = ldcg2u(ci + C4), t2 = ldcg2u(ci + 2 * C4);
  const uint16_t* cw = p.WB + WB_CW + slot * 4 * C4 + o;
  const uint2 w0 = *(const uint2*)cw, w1 = *(const uint2*)(cw + C4), w2 = *(const uint2*)(cw + 2 * C4),
              w3 = *(const uint2*)(cw + 3 * C4);
  float r[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float x = b2f(h16(t0, i)) * b2f(h16(w0, i)) + b2f(h16(t1, i)) * b2f(h16(w1, i)) +
              b2f(h16(t2, i)) * b2f(h16(w2, i)) + v[i] * b2f(h16(w3, i));
    r[i] = bf(x * sigm(x));
  }
  if (wr) {
    uint16_t* co = p.CVO + slot * CV_SLOT + o;
    *(uint2*)co = t1;
    *(uint2*)(co + C4) = t2;
    *(uint2*)(co + 2 * C4) = make_uint2(bfb(v[0]) | (bfb(v[1]) << 16), bfb(v[2]) | (bfb(v[3]) << 16));
  }
  return make_float4(r[0], r[1], r[2], r[3]);
}

// conv + gate + gated delta rule for (head h, 8 value columns), one warp
FI void kda_task(const Params& p, const Ctx& c, int b, int t) {
  const int h = t >> 4, d0 = (t & 15) * 8, l = c.l;
  const float* acc = p.SC + SC_ACCQ + b * 4 * C4;
  const int o = h * 128 + 4 * l;
  float4 qa = ldcg4(acc + o), ka = ldcg4(acc + C4 + o);
  const float4 ga = ldcg4(acc + 3 * C4 + o);
  qa = conv4(p, b * 3 + 0, o, qa, d0 == 0);
  ka = conv4(p, b * 3 + 1, o, ka, d0 == 0);
  const float qq[4] = {qa.x * KDA_SCALE, qa.y * KDA_SCALE, qa.z * KDA_SCALE, qa.w * KDA_SCALE};
  const float kk[4] = {ka.x, ka.y, ka.z, ka.w};
  const float de[4] = {sigm(-bf(ga.x)), sigm(-bf(ga.y)), sigm(-bf(ga.z)), sigm(-bf(ga.w))};
  // v for the 8 columns of this task: lane j < 8 owns column j
  float vj = 0.f, bt = 0.f;
  if (l < 8) {
    const int ch = h * 128 + d0 + l;
    const float val = bf(ldcg(acc + 2 * C4 + ch));
    const uint16_t* ci = p.CVI + (b * 3 + 2) * CV_SLOT + ch;
    const uint2 tp = ldcg2u(ci - (ch & 3));  // 8 B aligned word holding this channel
    const uint2 tq = ldcg2u(ci + C4 - (ch & 3)), tr = ldcg2u(ci + 2 * C4 - (ch & 3));
    const int e = ch & 3;
    const uint16_t* cw = p.WB + WB_CW + (b * 3 + 2) * 4 * C4 + ch;
    const float x = b2f(h16(tp, e)) * b2f(cw[0]) + b2f(h16(tq, e)) * b2f(cw[C4]) +
                    b2f(h16(tr, e)) * b2f(cw[2 * C4]) + val * b2f(cw[3 * C4]);
    vj = bf(x * sigm(x));
    uint16_t* co = p.CVO + (b * 3 + 2) * CV_SLOT + ch;
    co[0] = (uint16_t)h16(tq, e);
    co[C4] = (uint16_t)h16(tr, e);
    co[2 * C4] = (uint16_t)bfb(val);
    if (l == 0) bt = ldcg(p.SC + SC_BETA + b * 64 + h);
  }
  bt = __shfl_sync(0xffffffffu, bt, 0);
  float* S = p.SB + (size_t)b * S_SLOT + h * 16384 + (4 * l) * 128 + d0;
  float sg[4][8], pred[8], aa[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) pred[j] = aa[j] = 0.f;
  float4 sa[4], sb[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    sa[r] = ldcg4(S + r * 128);
    sb[r] = ldcg4(S + r * 128 + 4);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    sg[r][0] = sa[r].x * de[r]; sg[r][1] = sa[r].y * de[r]; sg[r][2] = sa[r].z * de[r]; sg[r][3] = sa[r].w * de[r];
    sg[r][4] = sb[r].x * de[r]; sg[r][5] = sb[r].y * de[r]; sg[r][6] = sb[r].z * de[r]; sg[r][7] = sb[r].w * de[r];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      pred[j] += sg[r][j] * kk[r];
      aa[j] += sg[r][j] * qq[r];
    }
  }
  const float kq = wsumf(kk[0] * qq[0] + kk[1] * qq[1] + kk[2] * qq[2] + kk[3] * qq[3]);
  float dv[8], outv = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float pj = wsumf(pred[j]), aj = wsumf(aa[j]);
    dv[j] = bt * (__shfl_sync(0xffffffffu, vj, j) - pj);
    if (l == j) outv = bf(aj + dv[j] * kq);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    *(float4*)(S + r * 128) = make_float4(sg[r][0] + kk[r] * dv[0], sg[r][1] + kk[r] * dv[1],
                                          sg[r][2] + kk[r] * dv[2], sg[r][3] + kk[r] * dv[3]);
    *(float4*)(S + r * 128 + 4) = make_float4(sg[r][4] + kk[r] * dv[4], sg[r][5] + kk[r] * dv[5],
                                              sg[r][6] + kk[r] * dv[6], sg[r][7] + kk[r] * dv[7]);
  }
  if (l < 8) p.SC[SC_O1 + b * C4 + h * 128 + d0 + l] = outv;
}

// ------------------------------------------------------------------------------------------
// MLA attention pieces
// ------------------------------------------------------------------------------------------
FI void ldm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
FI void ldm4t(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
FI void mma_bf16(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}
FI void cp_async16(unsigned saddr, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(saddr), "l"(g) : "memory");
}

// flash-decode over one chunk of cache rows, all 32 heads; CTA-cooperative
FI void attention(const Params& p, const Ctx& c, unsigned char* smem) {
  if (c.pid >= (int)p.nch) return;
  uint16_t* QS = (uint16_t*)(smem + SM_QS);
  uint16_t* CKS = (uint16_t*)(smem + SM_CKS);
  float* PS = (float*)(smem + SM_PS);
  const unsigned qs_a = (unsigned)__cvta_generic_to_shared(QS);
  const unsigned ck_a = (unsigned)__cvta_generic_to_shared(CKS);
  for (int i = c.tid; i < H * QROW * 2 / 16; i += TPB)
    ((uint4*)QS)[i] = ldcg4u((const char*)(p.CB + CB_QAR) + i * 16);
  const long long l0 = (long long)c.pid * p.cw;
  const long long l1 = min(l0 + p.cw, p.L + 1);
  const int nstep = (int)((l1 - l0 + 15) / 16);
  auto issue = [&](int s) {
    const unsigned ba = ck_a + (s % NBUF) * (16 * QROW * 2);
    char* bp = (char*)CKS + (s % NBUF) * (16 * QROW * 2);
    const long long r0 = l0 + (long long)s * 16;
    for (int i = c.tid; i < 16 * 72; i += TPB) {
      const int r = i / 72;
      if (s < nstep && r0 + r < l1)
        cp_async16(ba + i * 16, (const char*)p.CACHE + ((r0 + r) * 72 + (i - r * 72)) * 16);
      else
        *(uint4*)(bp + i * 16) = make_uint4(0, 0, 0, 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
#pragma unroll
  for (int s = 0; s < NBUF - 1; ++s) issue(s);

  const int g = c.w >> 2, sl = c.w & 3, l = c.l, j = l & 3, q4 = l >> 2;
  const int h0 = g * 16;
  float mA = -1.0e30f, mB = -1.0e30f, lsA = 0.f, lsB = 0.f;
  float pc[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) pc[n][0] = pc[n][1] = pc[n][2] = pc[n][3] = 0.f;
  // ldmatrix row addresses (lane-dependent part)
  //   A (Q, 16 heads x 16 cols): lanes 0-7 rows 0-7 col+0 | 8-15 rows 8-15 col+0 | 16-23 rows 0-7 col+8 | 24-31 rows 8-15 col+8
  //   B (cache, non-trans, k = col, n = row): 0-7 rows 0-7 col+0 | 8-15 rows 0-7 col+8 | 16-23 rows 8-15 col+0 | 24-31 rows 8-15 col+8
  //   B (cache, trans, k = row, n = col): same pattern as A
  const int lr = l & 7, lm = l >> 3;
  const unsigned qa_off = qs_a + ((h0 + lr + ((lm & 1) ? 8 : 0)) * QROW + ((lm & 2) ? 8 : 0)) * 2;
  const unsigned bn_off = ((lr + ((lm & 2) ? 8 : 0)) * QROW + ((lm & 1) ? 8 : 0)) * 2;
  const unsigned bt_off = ((lr + ((lm & 1) ? 8 : 0)) * QROW + ((lm & 2) ? 8 : 0)) * 2;

  for (int s = 0; s < nstep; ++s) {
    asm volatile("cp.async.wait_group 2;" ::: "memory");  // NBUF - 2
    __syncthreads();
    issue(s + NBUF - 1);
    const unsigned buf = ck_a + (s % NBUF) * (16 * QROW * 2);
    // ---- partial scores over this warp's 144 contraction columns ----
    float sn[8], sr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) sn[i] = sr[i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 9; ++kk) {
      const int c0 = kk < 8 ? sl * 128 + kk * 16 : KVL + sl * 16;
      unsigned a0, a1, a2, a3, b0, b1, b2, b3;
      ldm4(qa_off + c0 * 2, a0, a1, a2, a3);
      ldm4(buf + bn_off + c0 * 2, b0, b1, b2, b3);
      if (kk < 8) {
        mma_bf16(sn, a0, a1, a2, a3, b0, b1);
        mma_bf16(sn + 4, a0, a1, a2, a3, b2, b3);
      } else {
        mma_bf16(sr, a0, a1, a2, a3, b0, b1);
        mma_bf16(sr + 4, a0, a1, a2, a3, b2, b3);
      }
    }
    float* ps = PS + (c.w * 32 + l) * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ps[i] = sn[i];
      ps[8 + i] = sr[i];
    }
    __syncthreads();
    // ---- full scores, online softmax ----
    float sc[8];
    const long long rb = l0 + (long long)s * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = 0.f, r = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float* pq = PS + ((g * 4 + q) * 32 + l) * 16;
        a += pq[i];
        r += pq[8 + i];
      }
      const int row = (i >> 2) * 8 + 2 * j + (i & 1);
      sc[i] = (rb + row < l1) ? a + r * MLA_SCALE : -1.0e30f;
    }
    float xA = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[4], sc[5]));
    float xB = fmaxf(fmaxf(sc[2], sc[3]), fmaxf(sc[6], sc[7]));
    xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
    xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
    xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
    xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
    const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
    const float alA = expf(mA - nA), alB = expf(mB - nB);
    mA = nA;
    mB = nB;
    float pe[8];
    pe[0] = expf(sc[0] - nA); pe[1] = expf(sc[1] - nA); pe[4] = expf(sc[4] - nA); pe[5] = expf(sc[5] - nA);
    pe[2] = expf(sc[2] - nB); pe[3] = expf(sc[3] - nB); pe[6] = expf(sc[6] - nB); pe[7] = expf(sc[7] - nB);
    lsA = lsA * alA + (pe[0] + pe[1] + pe[4] + pe[5]);
    lsB = lsB * alB + (pe[2] + pe[3] + pe[6] + pe[7]);
    const unsigned pa0 = bfb(pe[0]) | (bfb(pe[1]) << 16), pa1 = bfb(pe[2]) | (bfb(pe[3]) << 16);
    const unsigned pa2 = bfb(pe[4]) | (bfb(pe[5]) << 16), pa3 = bfb(pe[6]) | (bfb(pe[7]) << 16);
    // ---- pc[h, c] = pc * alpha + P^T C over this warp's 128 latent columns ----
#pragma unroll
    for (int n = 0; n < 16; n += 2) {
      unsigned b0, b1, b2, b3;
      ldm4t(buf + bt_off + (sl * 128 + n * 8) * 2, b0, b1, b2, b3);
      pc[n][0] *= alA; pc[n][1] *= alA; pc[n][2] *= alB; pc[n][3] *= alB;
      pc[n + 1][0] *= alA; pc[n + 1][1] *= alA; pc[n + 1][2] *= alB; pc[n + 1][3] *= alB;
      mma_bf16(pc[n], pa0, pa1, pa2, pa3, b0, b1);
      mma_bf16(pc[n + 1], pa0, pa1, pa2, pa3, b2, b3);
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  lsA += __shfl_xor_sync(0xffffffffu, lsA, 1);
  lsA += __shfl_xor_sync(0xffffffffu, lsA, 2);
  lsB += __shfl_xor_sync(0xffffffffu, lsB, 1);
  lsB += __shfl_xor_sync(0xffffffffu, lsB, 2);
  if (sl == 0 && j == 0) {
    p.SC[SC_AM + c.pid * H + h0 + q4] = mA;
    p.SC[SC_AM + c.pid * H + h0 + q4 + 8] = mB;
    p.SC[SC_AL + c.pid * H + h0 + q4] = lsA;
    p.SC[SC_AL + c.pid * H + h0 + q4 + 8] = lsB;
  }
  uint16_t* apc = p.CB + CB_APC + (size_t)c.pid * H * KVL;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int col = sl * 128 + n * 8 + 2 * j;
    *(uint32_t*)(apc + (h0 + q4) * KVL + col) = bfb(pc[n][0]) | (bfb(pc[n][1]) << 16);
    *(uint32_t*)(apc + (h0 + q4 + 8) * KVL + col) = bfb(pc[n][2]) | (bfb(pc[n][3]) << 16);
  }
}

// ------------------------------------------------------------------------------------------
// the megakernel
// ------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TPB, 1) ayk_mega(const Params p) {
  extern __shared__ __align__(16) unsigned char smem[];
  Ctx c;
  c.tid = threadIdx.x; c.w = c.tid >> 5; c.l = c.tid & 31;
  c.pid = blockIdx.x; c.P = gridDim.x; c.gw = c.pid * NW + c.w; c.nwt = c.P * NW;
  c.XN = (float*)(smem + SM_XN); c.XQ = (int*)(smem + SM_XQ); c.XS = (float2*)(smem + SM_XS);
  c.RED = (float*)(smem + SM_RED); c.XW = (int*)(smem + SM_XW);
  float* LG = (float*)(smem + SM_LG);
  int* TOPE = (int*)(smem + SM_TOPE);
  float* TOPW = (float*)(smem + SM_TOPW);
  float* TOPP = (float*)(smem + SM_TOPP);
  float* WSC = (float*)(smem + SM_WSC) + c.w * PMAX;
  int* xw = c.XW + c.w * 32;
  const int l = c.l, gw = c.gw, nwt = c.nwt, P = c.P;
  const int ss = (int)p.sstop;
  int nb = (int)p.bar0;
  float* SC = p.SC;
#define RANGE(U, lo, hi) const int lo = (int)((long long)(U) * gw / nwt), hi = (int)((long long)(U) * (gw + 1) / nwt)

  for (int b = 0; b < 4; ++b) {
    const float* hprev = SC + SC_HM + (b - 1) * D;  // only dereferenced for b > 0
    if (b < 3) {
      // ================= KDA: norm + q/k/v/g ====================================
      if (1 <= ss) {
        prologue_norm(c, b ? hprev : nullptr, b ? nullptr : p.HIN, p.WB + WB_NRM + b * 2 * D, SC + SC_HA + b * D);
        if (gw < 32) SC[SC_BETA + b * 64 + gw] = sigm(bf(dot_xn(c, p.WB + WB_BW + (b * H + gw) * D)));
        else if (gw < 32 + 144) zero128(SC + SC_ACCGU + ((b + 3) & 3) * 9 * 2 * MI + (gw - 32) * 128, l);
        RANGE(U_QKVG, u0, u1);
        gemv_run(p.WA + (size_t)(A_KDA + b * U_KDA) * UR * 4, u0, u1, 18, SC + SC_ACCQ + b * 4 * C4, c.XQ, c.XS, l);
      }
      nb += P; barrier(p.BAR, nb);
      // ================= KDA: conv + gate + delta rule ==========================
      if (2 <= ss)
        for (int t = gw; t < H * 16; t += nwt) kda_task(p, c, b, t);
      nb += P; barrier(p.BAR, nb);
      // ================= KDA: o_proj onto the residual ==========================
      if (3 <= ss) {
        prologue_quant32(c, SC + SC_O1 + b * C4);
        if (gw < 128) zero128(SC + SC_ACCQ + b * 4 * C4 + gw * 128, l);
        RANGE(U_OP, u0, u1);
        gemv_run(p.WA + (size_t)(A_KDA + b * U_KDA + U_QKVG) * UR * 4, u0, u1, 32, SC + SC_HA + b * D, c.XQ, c.XS, l);
      }
      nb += P; barrier(p.BAR, nb);
    } else {
      // ================= MLA: norm + q_proj | kv_a ==============================
      if (11 <= ss) {
        prologue_norm(c, hprev, nullptr, p.WB + WB_NRM + b * 2 * D, SC + SC_HA + b * D);
        if (gw >= 32 && gw < 32 + 144) zero128(SC + SC_ACCGU + ((b + 3) & 3) * 9 * 2 * MI + (gw - 32) * 128, l);
        RANGE(U_QKVA, u0, u1);
        gemv_run(p.WA + (size_t)A_MLA * UR * 4, u0, u1, 18, SC + SC_ACCQA, c.XQ, c.XS, l);
      }
      nb += P; barrier(p.BAR, nb);
      // ================= MLA: absorbed query, rope, cache append ================
      if (12 <= ss) {
        const float* acc = SC + SC_ACCQA;
        if (gw < 128) {  // qa[h, g*128 + c] = sum_d q_nope[h,d] Wb[c,h,d]
          const int h = gw >> 2, o = h * QH + 4 * l;
          float4 qn = bf4(ldcg4(acc + o));
          qn.x *= MLA_SCALE; qn.y *= MLA_SCALE; qn.z *= MLA_SCALE; qn.w *= MLA_SCALE;
          const float4 s8 = *(const float4*)(p.FC + FC_SN8 + gw * 128 + 4 * l);
          const float4 sz = *(const float4*)(p.FC + FC_SNZ + gw * 128 + 4 * l);
          const float cst = wsumf(qn.x * sz.x + qn.y * sz.y + qn.z * sz.z + qn.w * sz.w);
          float xs, xsum;
          xw[l] = quant_group(make_float4(qn.x * s8.x, qn.y * s8.y, qn.z * s8.z, qn.w * s8.w), xs, xsum);
          __syncwarp();
          int a0, a1, a2, a3; uint4 szr;
          unit_dot((const uint4*)p.WA + (size_t)(A_MLA + U_QKVA + gw) * UR + l, xw, a0, a1, a2, a3, szr);
          const uint32_t r0 = bfb(xs * (float)a0 - cst) | (bfb(xs * (float)a1 - cst) << 16);
          const uint32_t r1 = bfb(xs * (float)a2 - cst) | (bfb(xs * (float)a3 - cst) << 16);
          *(uint2*)(p.CB + CB_QAR + h * QROW + (gw & 3) * 128 + 4 * l) = make_uint2(r0, r1);
        } else if (gw < 128 + 32 + 5) {
          const float ang = (float)p.L * p.FC[FC_INVF + l];
          const float cs = cosf(ang), sn = sinf(ang);
          uint16_t* crow = p.CACHE + (size_t)p.L * QROW;
          if (gw < 160) {  // rope'd query of head gw-128
            const int hh = gw - 128;
            const float2 v = ldcg2(acc + hh * QH + QKN + 2 * l);
            const float e = bf(v.x), od = bf(v.y);
            *(uint32_t*)(p.CB + CB_QAR + hh * QROW + KVL + 2 * l) = bfb(e * cs - od * sn) | (bfb(od * cs + e * sn) << 16);
          } else if (gw < 164) {  // latent of the new token
            const int o = (gw - 160) * 128 + 4 * l;
            const float4 v = ldcg4(acc + H * QH + o);
            *(uint2*)(crow + o) = make_uint2(bfb(v.x) | (bfb(v.y) << 16), bfb(v.z) | (bfb(v.w) << 16));
          } else {  // rope'd key of the new token
            const float2 v = ldcg2(acc + H * QH + KVL + 2 * l);
            const float e = bf(v.x), od = bf(v.y);
            *(uint32_t*)(crow + KVL + 2 * l) = bfb(e * cs - od * sn) | (bfb(od * cs + e * sn) << 16);
          }
        }
      }
      nb += P; barrier(p.BAR, nb);
      // ================= MLA: latent attention ==================================
      if (13 <= ss) attention(p, c, smem);
      nb += P; barrier(p.BAR, nb);
      // ================= MLA: softmax combine + kv_b value half =================
      if (14 <= ss) {
        if (gw < 128) {
          const int h = gw >> 2, g = gw & 3, nch = (int)p.nch;
          float mx = -1.0e30f;
          for (int ch = l; ch < nch; ch += 32) {
            const float a = ldcg(SC + SC_AM + ch * H + h);
            WSC[ch] = a;
            mx = fmaxf(mx, a);
          }
          mx = wmaxf(mx);
          float dl = 0.f;
          for (int ch = l; ch < nch; ch += 32) {
            const float e = expf(WSC[ch] - mx);
            WSC[ch] = e;
            dl += e * ldcg(SC + SC_AL + ch * H + h);
          }
          const float den = wsumf(dl);
          __syncwarp();
          float4 pv = make_float4(0, 0, 0, 0);
          const uint16_t* ap = p.CB + CB_APC + (size_t)h * KVL + g * 128 + 4 * l;
          for (int ch = 0; ch < nch; ch += 8) {
            uint2 r[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) r[u] = ldcg2u(ap + (size_t)min(ch + u, nch - 1) * H * KVL);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const float e = (ch + u < nch) ? WSC[ch + u] : 0.f;
              const float4 v = b2f4(r[u]);
              pv.x += e * v.x; pv.y += e * v.y; pv.z += e * v.z; pv.w += e * v.w;
            }
          }
          pv.x /= den; pv.y /= den; pv.z /= den; pv.w /= den;
          float xs, xsum;
          xw[l] = quant_group(pv, xs, xsum);
          __syncwarp();
          int a0, a1, a2, a3; uint4 sz;
          unit_dot((const uint4*)p.WA + (size_t)(A_MLA + U_QKVA + U_KVBN + gw) * UR + l, xw, a0, a1, a2, a3, sz);
          flush4(SC + SC_O1M + h * 128 + 4 * l,
                 make_float4(epi(sz.x, a0, xs, xsum), epi(sz.y, a1, xs, xsum), epi(sz.z, a2, xs, xsum), epi(sz.w, a3, xs, xsum)));
        } else if (gw < 128 + QKVA_NT) {
          zero128(SC + SC_ACCQA + (gw - 128) * 128, l);
        }
      }
      nb += P; barrier(p.BAR, nb);
      // ================= MLA: o_proj onto the residual ==========================
      if (15 <= ss) {
        prologue_quant32(c, SC + SC_O1M);
        RANGE(U_OP, u0, u1);
        gemv_run(p.WA + (size_t)(A_MLA + U_QKVA + U_KVBN + U_KVBV) * UR * 4, u0, u1, 32, SC + SC_HA + b * D, c.XQ, c.XS, l);
      }
      nb += P; barrier(p.BAR, nb);
    }

    const int mb = b < 3 ? 3 : 15;  // stage-id base of this block's MoE
    const uint32_t* moe = p.WA + (size_t)(A_MOE + b * U_MOE) * UR * 4;
    // ================= MoE: norm + router =========================================
    if (mb + 1 <= ss) {
      prologue_norm(c, SC + SC_HA + b * D, nullptr, p.WB + WB_NRM + b * 2 * D + D, SC + SC_HM + b * D);
      if (gw < NEXP) SC[SC_RLOG + b * 64 + gw] = bf(dot_xn(c, p.WB + WB_RW + (b * NEXP + gw) * D));
      else if (b == 3 && gw < NEXP + 32) zero128(SC + SC_O1M + (gw - NEXP) * 128, l);
    }
    nb += P; barrier(p.BAR, nb);
    // ================= MoE: top-8 + gate/up ========================================
    if (mb + 2 <= ss) {
      if (c.tid < NEXP) LG[c.tid] = ldcg(SC + SC_RLOG + b * 64 + c.tid);
      if (c.tid < 9) TOPE[c.tid] = c.tid;
      __syncthreads();
      if (c.tid < NEXP) {
        const float me = LG[c.tid];
        float mx = -3.0e38f;
        int rk = 0;
        for (int i = 0; i < NEXP; ++i) {
          const float v = LG[i];
          mx = fmaxf(mx, v);
          rk += (v > me) | ((v == me) & (i < c.tid));
        }
        float den = 0.f;
        for (int i = 0; i < NEXP; ++i) den += expf(LG[i] - mx);
        if (rk < 8) {
          TOPE[rk] = c.tid;
          TOPP[rk] = expf(me - mx) / den;
        }
      }
      __syncthreads();
      if (c.tid == 0) {
        float ws = 0.f;
        for (int i = 0; i < 8; ++i) ws += TOPP[i];
        const float nrm = ROUTED_SCALE / (ws + 1.0e-9f);
        for (int i = 0; i < 8; ++i) TOPW[i] = TOPP[i] * nrm;
        TOPE[8] = NEXP;
        TOPW[8] = 1.0f;
      }
      __syncthreads();
      RANGE(9 * U_GU, u0, u1);
      for (int s = u0 / U_GU; s <= (u1 - 1) / U_GU && u0 < u1; ++s) {
        const int lo = max(u0, s * U_GU) - s * U_GU, hi = min(u1, (s + 1) * U_GU) - s * U_GU;
        gemv_run(moe + (size_t)TOPE[s] * U_EXP * UR * 4, lo, hi, 18, SC + SC_ACCGU + (b * 9 + s) * 2 * MI, c.XQ, c.XS, l);
      }
    }
    nb += P; barrier(p.BAR, nb);
    // ================= MoE: silu(gate)*up, quantise, down onto the residual ========
    if (mb + 3 <= ss) {
      RANGE(9 * U_DN, u0, u1);
      int key = -1;
      float xs = 0.f, xsum = 0.f;
      for (int u = u0; u < u1; ++u) {
        const int s = u / U_DN, r = u - s * U_DN, kg = r / 18, tile = r - kg * 18;
        if (s * 8 + kg != key) {
          key = s * 8 + kg;
          const float* ag = SC + SC_ACCGU + (b * 9 + s) * 2 * MI + kg * 128 + 4 * l;
          const float4 gv = ldcg4(ag), uv = ldcg4(ag + MI);
          xw[l] = quant_group(make_float4((gv.x * sigm(gv.x)) * uv.x, (gv.y * sigm(gv.y)) * uv.y,
                                          (gv.z * sigm(gv.z)) * uv.z, (gv.w * sigm(gv.w)) * uv.w), xs, xsum);
          __syncwarp();
        }
        int a0, a1, a2, a3; uint4 sz;
        unit_dot((const uint4*)moe + ((size_t)TOPE[s] * U_EXP + U_GU + tile * 8 + kg) * UR + l, xw, a0, a1, a2, a3, sz);
        const float wj = TOPW[s];
        flush4(SC + SC_HM + b * D + tile * 128 + 4 * l,
               make_float4(epi(sz.x, a0, xs, xsum) * wj, epi(sz.y, a1, xs, xsum) * wj,
                           epi(sz.z, a2, xs, xsum) * wj, epi(sz.w, a3, xs, xsum) * wj));
      }
    }
    nb += P; barrier(p.BAR, nb);
  }
  // ---- emit the bf16 next-token hidden ----
  if (c.pid == 0) {
    for (int i = c.tid; i < D / 4; i += TPB) {
      const float4 v = ldcg4(SC + SC_HM + 3 * D + 4 * i);
      *(uint2*)(p.HOUT + 4 * i) = make_uint2(bfb(v.x) | (bfb(v.y) << 16), bfb(v.z) | (bfb(v.w) << 16));
    }
  }
}

// ------------------------------------------------------------------------------------------
// host entry points (called through ctypes; launches on the default stream)
// ------------------------------------------------------------------------------------------
extern "C" {
int ayk_init() {
  cudaError_t e = cudaFuncSetAttribute(ayk_mega, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOT);
  if (e != cudaSuccess) return -(int)e;
  cudaDeviceProp prop;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  return prop.multiProcessorCount;
}
int ayk_step(const Params* p, int grid) {
  ayk_mega<<<grid, TPB, SM_TOT>>>(*p);
  return (int)cudaGetLastError();
}
void ayk_layout(long long* o) {
  const long long v[] = {U_TOT, UR * 4, WB_TOT, FC_TOT, SC_TOT, CB_TOT, PMAX, NW,
                         A_KDA, U_KDA, U_QKVG, A_MLA, U_QKVA, U_KVBN, U_KVBV, A_MOE, U_MOE, U_EXP, U_GU,
                         WB_CW, WB_BW, WB_RW, WB_NRM, FC_INVF, FC_SN8, FC_SNZ,
                         SC_ACCQ, SC_HA, SC_HM, SC_O1, SC_BETA, SC_RLOG, SC_ACCGU, SC_ACCQA, SC_O1M, SC_AM, SC_AL,
                         CB_QAR, CB_APC, QROW, 26 /* barriers per launch */};
  for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) o[i] = v[i];
}
}
