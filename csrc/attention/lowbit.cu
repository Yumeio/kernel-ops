// =============================================================================
// kernel-ops | csrc/attention/lowbit.cu
// B17 row 1 (EXPERIMENTAL, catalog A09): low-bit rotated KV and attention over it.
//   lowbit_kv_store_kernel   rotate (signed Hadamard) + quantize new K/V into bit-packed records
//   lowbit_decode_kernel     CUDA-core decode reading the records; dequant in registers
//   lowbit_prefill_kernel    paged FA2 prefill; K / V tiles dequantized into shared memory (fp16)
// Host entries (ops.h, namespace kops::experimental::lowbit): lowbit_codec, lowbit_entry_bytes,
// lowbit_resolve, lowbit_kv_store, lowbit_decode_plan / _run, lowbit_prefill_plan / _run.
// Split partials are combined by merge_split_kv_kernel (B1, merge.cu).
//
// Contract: csrc/attention/lowbit_kernels.cuh, CONTRACTS.md §1h.  Codecs: TurboQuant_mse /
// TurboQuant_prod (arXiv:2504.19874) with a signed-Hadamard rotation instead of a dense random
// orthogonal matrix, and the uniform absmax codes of attention.rs turbo4 / turbo3.
//
// Numerics.  Nothing is rotated back per key: q is rotated once per CTA (q_rot = Hn (s .* q);
// kTurboProd: q_sk = Hn (s' .* q_rot)), the score is scale * <q_rot, levels[code]>
// + gain * <q_sk, sgn>, V is accumulated in its rotated basis and the output is rotated back once
// per row.  Mask / window / chunk / softcap / ALiBi / sink rules, LSE and the fixed fp32 summation
// order follow decode.cu and prefill.cu (bit-reproducible; batch-invariant with
// ctx.batch_invariant).  The prefill kernel evaluates QK and PV on mma.sync fp16 whatever the q / o
// dtype: dequantized values must stay below 65504 in magnitude, as in an fp16 cache.
// =============================================================================
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/lowbit_kernels.cuh"
#include "csrc/attention/ops.h"

namespace kops::attn::internal {
Status merge_split_kv_run(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                          const Tensor& sink_logits, const Tensor& total_rows,
                          const LaunchContext& ctx);
}  // namespace kops::attn::internal

namespace kops::experimental::lowbit {
namespace {

using attn::AttentionModifiers;
using attn::AttentionPlan;
using attn::AttentionSpec;
using attn::DecodeParams;
using attn::MaskMode;
using attn::PagedPrefillParams;
using attn::SplitKVBuffers;
using attn::WorkPlan;

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kSqrtHalfPi = 1.2533141373155003f;  // sqrt(pi / 2): QJL gain of a sign sketch

__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }

__host__ __device__ __forceinline__ int64_t fdiv(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

__device__ __forceinline__ float tanh_fast(float x) {
  const float e = __expf(2.0f * fabsf(x));
  return copysignf(1.0f - __fdividef(2.0f, e + 1.0f), x);
}

/// 1 / sqrt(D): the normalization of the Hadamard transform.
template <int D>
KOPS_HOST_DEVICE constexpr float rsqrt_dim() {
  return D == 64 ? 0.125f : (D == 128 ? 0.08838834764831845f : 0.0625f);
}

/// v with its sign flipped when neg == 1.
__device__ __forceinline__ float flip(float v, uint32_t neg) {
  return __uint_as_float(__float_as_uint(v) ^ (neg << 31));
}

__device__ __forceinline__ float2 unpack2(uint32_t w, bool fp16) {
  return fp16 ? __half22float2(*reinterpret_cast<const __half2*>(&w))
              : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
}

__device__ __forceinline__ uint32_t pack2h(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi, bool fp16) {
  if (fp16) return pack2h(lo, hi);
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// ---- Hadamard ------------------------------------------------------------------------------------
// Element i of a vector lives in lane i / N, register i % N.  The Sylvester Hadamard matrix is a
// tensor power of H2, so its butterflies commute: the stages over the low index bits run inside the
// lane, the stages over the lane bits are shuffles.  Unnormalized (callers scale by rsqrt_dim).
template <int N>
__device__ __forceinline__ void wht_lane(float (&v)[N]) {
#pragma unroll
  for (int h = 1; h < N; h *= 2) {
#pragma unroll
    for (int i = 0; i < N; i += 2 * h) {
#pragma unroll
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

/// Stages over the lane index within an aligned group of `group` lanes (power of two <= 32).
template <int N>
__device__ __forceinline__ void wht_cross(float (&v)[N], int lane_in_group, int group) {
  for (int off = 1; off < group; off *= 2) {
    const bool upper = (lane_in_group & off) != 0;
#pragma unroll
    for (int e = 0; e < N; ++e) {
      const float o = __shfl_xor_sync(0xffffffffu, v[e], off);
      v[e] = upper ? o - v[e] : v[e] + o;
    }
  }
}

// ---- records -------------------------------------------------------------------------------------
/// Byte offsets inside one record (lowbit_kernels.cuh): [codes][signs][scale][gain].
struct RecLay {
  int bits;
  uint32_t mask;
  int sign_off, scale_off;
};

template <int D>
__device__ __forceinline__ RecLay rec_lay(const LowBitSide& s) {
  RecLay r;
  r.bits = s.code_bits;
  r.mask = (1u << s.code_bits) - 1u;
  r.sign_off = D * s.code_bits / 8;
  r.scale_off = r.sign_off + (s.qjl != 0 ? D / 8 : 0);
  return r;
}

/// 32 bits of a record starting at byte `off` (records are 4-byte aligned; the word after the
/// last code word is the sign plane or the scale, so the second read stays inside the record).
__device__ __forceinline__ uint32_t load_bits(const char* rec, int off) {
  const uint32_t* w = reinterpret_cast<const uint32_t*>(rec + (off & ~3));
  return __funnelshift_r(w[0], w[1], static_cast<uint32_t>(off & 3) * 8u);
}

// ---- store ---------------------------------------------------------------------------------------
constexpr int kStoreThreads = 128;
constexpr int kStoreWarps = 4;

/// One (token, head) vector of one side, by one warp: D / 32 elements per lane.
/// sm: [D] codes + [32] per-lane sign masks of this warp.
template <int D>
__device__ __forceinline__ void store_side(const char* row, bool fp16, const LowBitSide& sd,
                                           uint32_t seed, char* rec, uint8_t* sm) {
  constexpr int VEC = D / 32;
  constexpr float rn = rsqrt_dim<D>();
  const int lane = threadIdx.x & 31;
  const int e0 = lane * VEC;
  float v[VEC];
#pragma unroll
  for (int e = 0; e < VEC; e += 2) {
    const float2 f = unpack2(*reinterpret_cast<const uint32_t*>(row + (e0 + e) * 2), fp16);
    v[e] = f.x;
    v[e + 1] = f.y;
  }
  if (sd.rotated != 0) {
    const uint32_t sw = lowbit_sign_word(seed, sd.sign_stream, static_cast<uint32_t>(e0) >> 5) >> (e0 & 31);
#pragma unroll
    for (int e = 0; e < VEC; ++e) v[e] = flip(v[e], (sw >> e) & 1u);
    wht_lane(v);
    wht_cross(v, lane, 32);
#pragma unroll
    for (int e = 0; e < VEC; ++e) v[e] *= rn;
  }
  float ss = 0.0f, amax = 0.0f;
  uint32_t bad = 0;
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    ss = __fmaf_rn(v[e], v[e], ss);
    amax = fmaxf(amax, fabsf(v[e]));
    bad |= (__float_as_uint(v[e]) & 0x7F800000u) == 0x7F800000u ? 1u : 0u;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    bad |= __shfl_xor_sync(0xffffffffu, bad, off);
  }
  const int bits = sd.code_bits, top = (1 << bits) - 1;
  float a = sd.absmax != 0 ? amax / sd.levels[top] : sqrtf(ss) * rn;
  if (bad != 0) a = __uint_as_float(0x7FC00000u);  // non-finite input poisons the record's scale
  const bool live = a > 0.0f;
  const float inv = live ? 1.0f / a : 0.0f;
  float er[VEC];
  float ee = 0.0f;
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    const float z = live ? v[e] * inv : 0.0f;
    int code = 0;
#pragma unroll
    for (int t = 0; t < 15; ++t) code += sd.thresholds[t] < z ? 1 : 0;
    er[e] = z - sd.levels[code];
    ee = __fmaf_rn(er[e], er[e], ee);
    sm[e0 + e] = static_cast<uint8_t>(code);
  }
  const RecLay rl = rec_lay<D>(sd);
  if (sd.qjl != 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ee += __shfl_xor_sync(0xffffffffu, ee, off);
    const uint32_t sw = lowbit_sign_word(seed, kSignStreamSketch, static_cast<uint32_t>(e0) >> 5) >> (e0 & 31);
#pragma unroll
    for (int e = 0; e < VEC; ++e) er[e] = flip(er[e], (sw >> e) & 1u);
    wht_lane(er);
    wht_cross(er, lane, 32);
    uint32_t m = 0;
#pragma unroll
    for (int e = 0; e < VEC; ++e) m |= (er[e] < 0.0f ? 0u : 1u) << e;
    sm[D + lane] = static_cast<uint8_t>(m);
  }
  __syncwarp();
  if (lane < D * bits / 32) {  // code word `lane`: bits [32 lane, 32 lane + 32) of the code stream
    const int b0 = lane * 32;
    const int i0 = b0 / bits, i1 = min(D - 1, (b0 + 31) / bits);
    uint32_t w = 0;
    for (int i = i0; i <= i1; ++i) {
      const int sh = i * bits - b0;
      const uint32_t c = sm[i];
      w |= sh >= 0 ? c << sh : c >> (-sh);
    }
    *reinterpret_cast<uint32_t*>(rec + lane * 4) = w;
  }
  if (sd.qjl != 0 && lane < D / 32) {  // sign word `lane` from the masks of 32 / VEC lanes
    constexpr int LPW = 32 / VEC;
    uint32_t w = 0;
#pragma unroll
    for (int t = 0; t < LPW; ++t) w |= static_cast<uint32_t>(sm[D + lane * LPW + t]) << (t * VEC);
    *reinterpret_cast<uint32_t*>(rec + rl.sign_off + lane * 4) = w;
  }
  if (lane == 0) {
    *reinterpret_cast<float*>(rec + rl.scale_off) = a;
    if (sd.qjl != 0) *reinterpret_cast<float*>(rec + rl.scale_off + 4) = a * sqrtf(ee) * (kSqrtHalfPi * rn);
  }
  __syncwarp();
}

// ---- decode --------------------------------------------------------------------------------------
constexpr int kDecThreads = 128;

template <int D, int R8>
struct DecGeom {
  static_assert(D == 64 || D == 128 || D == 256, "Hadamard head dim");
  static constexpr int kBdx = D / 8;                  // lanes over the head dim (8 elements each)
  static constexpr int kGroups = kDecThreads / kBdx;  // token groups per CTA
  static constexpr int kSmem = (kGroups * R8 * (D + 2) + R8 * (D + 2)) * 4;
};

/// Per-request K/V addressing for one kv head (byte pointers / strides).
struct DecKv {
  const char* k;
  const char* v;
  int64_t k_sp, k_st, v_sp, v_st;
  const int32_t* pages;
  int ps;
  int kv_len;
};

__device__ __forceinline__ DecKv dec_kv(const kv::PagedKVDesc& d, int b, int hk) {
  DecKv r;
  r.k = static_cast<const char*>(d.k_data) + hk * d.k_stride_head;
  r.v = static_cast<const char*>(d.v_data) + hk * d.v_stride_head;
  r.k_sp = d.k_stride_page;
  r.k_st = d.k_stride_token;
  r.v_sp = d.v_stride_page;
  r.v_st = d.v_stride_token;
  r.ps = d.page_size;
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    r.pages = d.block_table + static_cast<int64_t>(b) * d.block_table_stride;
    r.kv_len = d.seq_lens[b];
  } else {
    const int p0 = d.kv_indptr[b], np = d.kv_indptr[b + 1] - p0;
    r.pages = d.kv_page_indices + p0;
    r.kv_len = np > 0 ? (np - 1) * d.page_size + d.kv_last_page_len[b] : 0;
  }
  return r;
}

/// Visible-key interval of one query row: [lo, hi] inclusive, keys below wlo hidden unless they
/// are sink tokens (j < num_sink).
struct RowMask {
  int lo, hi, wlo;
};

__device__ __forceinline__ RowMask row_mask(const AttentionModifiers& m, int qpos, int range_lo,
                                            int range_hi) {
  int64_t h = min(static_cast<int64_t>(range_hi) - 1, static_cast<int64_t>(qpos));
  int64_t l = range_lo;
  if (m.window_right >= 0) h = min(h, static_cast<int64_t>(qpos) + m.window_right);
  if (m.chunk_size > 0) {
    const int64_t c0 = fdiv(qpos, m.chunk_size) * m.chunk_size;
    l = max(l, c0);
    h = min(h, c0 + m.chunk_size - 1);
  }
  RowMask r;
  r.lo = static_cast<int>(l);
  r.hi = static_cast<int>(max(h, static_cast<int64_t>(-1)));
  r.wlo = m.window_left >= 0 ? static_cast<int>(max(static_cast<int64_t>(qpos) - m.window_left,
                                                    static_cast<int64_t>(INT_MIN)))
                             : INT_MIN;
  return r;
}

__device__ __forceinline__ bool row_visible(const RowMask& r, int j, int num_sink) {
  return j >= r.lo && j <= r.hi && (j >= r.wlo || j < num_sink);
}

/// Key range [lo, hi) of split c (of S) for a request with kv_len keys.  chunk > 0: fixed chunk
/// (batch-invariant); otherwise the request is cut into S near-equal 16-aligned pieces.
__device__ __forceinline__ void split_range(int kv_len, int c, int S, int chunk, int* lo, int* hi) {
  int64_t ck = chunk;
  if (ck <= 0) ck = S > 1 ? ((static_cast<int64_t>(kv_len) + S - 1) / S + 15) / 16 * 16 : INT_MAX;
  const int64_t l = static_cast<int64_t>(c) * ck;
  *lo = static_cast<int>(min(l, static_cast<int64_t>(kv_len)));
  *hi = static_cast<int>(min(l + ck, static_cast<int64_t>(kv_len)));
}

template <int D, int R8>
__device__ __forceinline__ void decode_body(const DecodeParams& p, const LowBitTables& tb) {
  using Geo = DecGeom<D, R8>;
  constexpr int BDX = Geo::kBdx, NG = Geo::kGroups;
  constexpr float rn = rsqrt_dim<D>();
  extern __shared__ __align__(16) unsigned char dec_smem[];
  float* sm_m = reinterpret_cast<float*>(dec_smem);   // [NG][R8]
  float* sm_l = sm_m + NG * R8;                        // [NG][R8]
  float* sm_o = sm_l + NG * R8;                        // [NG][R8][D]
  float* sm_f = sm_o + NG * R8 * D;                    // [R8][D] combined rows (rotated V basis)
  float* sm_r = sm_f + R8 * D;                         // [R8][2] row max / denominator

  const int tid = threadIdx.x, x = tid % BDX, g = tid / BDX;
  const int b = blockIdx.x, c = blockIdx.z, S = gridDim.z;
  const int G = p.num_qo_heads / p.num_kv_heads, q_len = p.q_len, R = G * q_len;
  const int nrb = (R + R8 - 1) / R8;
  const int hk = blockIdx.y / nrb, row0 = (blockIdx.y % nrb) * R8;
  const int rows = min(R8, R - row0);
  const AttentionModifiers& md = p.mod;
  const bool fp16 = tb.io_fp16 != 0;
  const bool qjl = tb.k.qjl != 0;
  const RecLay kl = rec_lay<D>(tb.k), vl = rec_lay<D>(tb.v);

  const DecKv kv = dec_kv(p.kv, b, hk);
  const int kv_len = kv.kv_len;
  int lo, hi;
  split_range(kv_len, c, S, p.plan.kv_chunk_size != nullptr ? *p.plan.kv_chunk_size : 0, &lo, &hi);
  const float sm = md.sm_scale != 0.0f ? md.sm_scale : rn;
  const float cap = md.logits_softcap;
  const float qs = sm * (cap > 0.0f ? 1.0f : kLog2e);

  // rows: packed r -> query token i = r / G, head hk * G + r % G.  q is rotated once here:
  // q[r] = qs * Hn (s .* q), qk[r] = Hn (s' .* q[r]) (kTurboProd sketch).
  const uint32_t sw_k = lowbit_sign_word(tb.seed, kSignStreamKey, static_cast<uint32_t>(x * 8) >> 5) >> ((x * 8) & 31);
  const uint32_t sw_s = lowbit_sign_word(tb.seed, kSignStreamSketch, static_cast<uint32_t>(x * 8) >> 5) >> ((x * 8) & 31);
  float q[R8][8], qk[R8][8];
  RowMask rm[R8];
  int qpos[R8], head[R8];
  float slope2[R8];
  int qmin = INT_MAX, qmax = INT_MIN;
#pragma unroll
  for (int r = 0; r < R8; ++r) {
    const int pr = row0 + min(r, rows - 1);
    const int i = pr / G;
    head[r] = hk * G + (pr - i * G);
    qpos[r] = kv_len - q_len + i;
    rm[r] = row_mask(md, qpos[r], lo, hi);
    slope2[r] = md.alibi_slopes != nullptr ? md.alibi_slopes[head[r]] * kLog2e : 0.0f;
    qmin = min(qmin, qpos[r]);
    qmax = max(qmax, qpos[r]);
    const char* qp = static_cast<const char*>(p.q) +
                     ((static_cast<int64_t>(b) * q_len + i) * p.q_stride_n +
                      static_cast<int64_t>(head[r]) * p.q_stride_h + x * 8) * 2;
    const uint4 w = *reinterpret_cast<const uint4*>(qp);
    const float2 a = unpack2(w.x, fp16), bb = unpack2(w.y, fp16), cc = unpack2(w.z, fp16), dd = unpack2(w.w, fp16);
    q[r][0] = a.x; q[r][1] = a.y; q[r][2] = bb.x; q[r][3] = bb.y;
    q[r][4] = cc.x; q[r][5] = cc.y; q[r][6] = dd.x; q[r][7] = dd.y;
    float qn = qs;
    if (tb.k.rotated != 0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) q[r][e] = flip(q[r][e], (sw_k >> e) & 1u);
      wht_lane(q[r]);
      wht_cross(q[r], x, BDX);
      qn = qs * rn;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      q[r][e] *= qn;
      qk[r][e] = flip(q[r][e], (sw_s >> e) & 1u);
    }
    if (qjl) {
      wht_lane(qk[r]);
      wht_cross(qk[r], x, BDX);
#pragma unroll
      for (int e = 0; e < 8; ++e) qk[r][e] *= rn;
    }
  }
  // Keys no row of this CTA can see are skipped (causal tail, window head, chunk bounds).
  int jlo = lo, jhi = min(hi, max(qmax + 1, 0));
  if (md.window_left >= 0 && md.num_sink_tokens == 0) jlo = max(jlo, qmin - md.window_left);
  if (md.chunk_size > 0) {
    jlo = static_cast<int>(max(static_cast<int64_t>(jlo), fdiv(qmin, md.chunk_size) * md.chunk_size));
    jhi = static_cast<int>(min(static_cast<int64_t>(jhi), (fdiv(qmax, md.chunk_size) + 1) * md.chunk_size));
  }
  const int gap_hi = qmin - md.window_left;  // window + sink tokens: [num_sink, gap_hi) is hidden
  const bool has_gap = md.window_left >= 0 && md.num_sink_tokens > 0 && gap_hi > md.num_sink_tokens;

  float m[R8], l[R8], o[R8][8];
#pragma unroll
  for (int r = 0; r < R8; ++r) {
    m[r] = ninf();
    l[r] = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) o[r][e] = 0.0f;
  }
  // Rounds of NG tokens (group g takes token j0 + g): every lane runs the same rounds, so the
  // full-warp shuffles below never see an exited lane.
  for (int j0 = jlo; j0 < jhi; j0 += NG) {
    if (has_gap && j0 >= md.num_sink_tokens && j0 + NG <= gap_hi) {
      j0 = gap_hi - NG;  // the next round starts at gap_hi
      continue;
    }
    const int j = j0 + g;
    const bool tok = j < jhi;
    float kx[8], vx[8];
    uint32_t sx = 0;   // bit e set: sketch sign of element e is -1
    float ga = 0.0f;
    if (tok) {
      const int pg = kv.pages[j / kv.ps], off = j % kv.ps;
      const char* kr = kv.k + static_cast<int64_t>(pg) * kv.k_sp + static_cast<int64_t>(off) * kv.k_st;
      const char* vr = kv.v + static_cast<int64_t>(pg) * kv.v_sp + static_cast<int64_t>(off) * kv.v_st;
      const uint32_t kw = load_bits(kr, x * kl.bits);
      const uint32_t vw = load_bits(vr, x * vl.bits);
      const float ka = *reinterpret_cast<const float*>(kr + kl.scale_off);
      const float va = *reinterpret_cast<const float*>(vr + vl.scale_off);
      if (qjl) {
        sx = ~static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(kr + kl.sign_off + x));
        ga = *reinterpret_cast<const float*>(kr + kl.scale_off + 4);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        kx[e] = ka * tb.k.levels[(kw >> (e * kl.bits)) & kl.mask];
        vx[e] = va * tb.v.levels[(vw >> (e * vl.bits)) & vl.mask];
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) kx[e] = vx[e] = 0.0f;
    }
#pragma unroll
    for (int r = 0; r < R8; ++r) {
      if (r >= rows) break;
      float s = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) s = __fmaf_rn(q[r][e], kx[e], s);
      if (qjl) {
        float t = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) t += flip(qk[r][e], (sx >> e) & 1u);
        s = __fmaf_rn(ga, t, s);
      }
#pragma unroll
      for (int off2 = BDX / 2; off2 > 0; off2 >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off2);
      if (cap > 0.0f) s = cap * tanh_fast(s / cap) * kLog2e;
      s += slope2[r] * static_cast<float>(j - qpos[r]);
      if (!tok || !row_visible(rm[r], j, md.num_sink_tokens)) continue;
      const float mn = fmaxf(m[r], s);
      const float a = exp2f(m[r] - mn), pj = exp2f(s - mn);
      l[r] = l[r] * a + pj;
#pragma unroll
      for (int e = 0; e < 8; ++e) o[r][e] = __fmaf_rn(pj, vx[e], o[r][e] * a);
      m[r] = mn;
    }
  }

  // combine the NG token groups (index order)
  if (x == 0) {
#pragma unroll
    for (int r = 0; r < R8; ++r) {
      sm_m[g * R8 + r] = m[r];
      sm_l[g * R8 + r] = l[r];
    }
  }
#pragma unroll
  for (int r = 0; r < R8; ++r) {
#pragma unroll
    for (int e = 0; e < 8; ++e) sm_o[(g * R8 + r) * D + x * 8 + e] = o[r][e];
  }
  __syncthreads();
  for (int idx = tid; idx < rows * D; idx += kDecThreads) {
    const int r = idx / D, d = idx - (idx / D) * D;
    float mx = ninf();
    for (int gg = 0; gg < NG; ++gg) mx = fmaxf(mx, sm_m[gg * R8 + r]);
    float den = 0.0f, acc = 0.0f;
    if (mx != ninf()) {
      for (int gg = 0; gg < NG; ++gg) {
        const float w = exp2f(sm_m[gg * R8 + r] - mx);
        den = __fmaf_rn(sm_l[gg * R8 + r], w, den);
        acc = __fmaf_rn(sm_o[(gg * R8 + r) * D + d], w, acc);
      }
    }
    sm_f[r * D + d] = acc;
    if (d == 0) {
      sm_r[2 * r] = mx;
      sm_r[2 * r + 1] = den;
    }
  }
  __syncthreads();
  // rotate the rows back: out = s_v .* Hn f  (in place, one Hadamard stage per barrier)
  const bool v_rot = tb.v.rotated != 0;
  if (v_rot) {
    for (int h = 1; h < D; h *= 2) {
      for (int pi = tid; pi < rows * (D / 2); pi += kDecThreads) {
        const int r = pi / (D / 2), kk = pi - r * (D / 2);
        const int i = (kk / h) * 2 * h + (kk % h);
        float* f = sm_f + r * D;
        const float a = f[i], bb = f[i + h];
        f[i] = a + bb;
        f[i + h] = a - bb;
      }
      __syncthreads();
    }
  }
  const bool split = p.plan.split_kv && p.split.o_partial != nullptr;
  const int Hq = p.num_qo_heads;
  for (int idx = tid; idx < rows * D; idx += kDecThreads) {
    const int r = idx / D, d = idx - (idx / D) * D;
    float acc = sm_f[r * D + d];
    if (v_rot) {
      const uint32_t sw = lowbit_sign_word(tb.seed, kSignStreamValue, static_cast<uint32_t>(d) >> 5);
      acc = flip(acc * rn, (sw >> (d & 31)) & 1u);
    }
    float mx = sm_r[2 * r], den = sm_r[2 * r + 1];
    const int pr = row0 + r, i = pr / G, h = hk * G + (pr - i * G);
    const int64_t row = static_cast<int64_t>(b) * q_len + i;
    if (split) {
      const int64_t prow = static_cast<int64_t>(p.split.merge_indptr[row]) + c;
      p.split.o_partial[(prow * Hq + h) * D + d] = den > 0.0f ? acc / den : 0.0f;
      if (d == 0) p.split.lse_partial[prow * Hq + h] = den > 0.0f ? (mx + log2f(den)) * kLn2 : ninf();
    } else {
      float num = 1.0f;
      if (md.sink_logits != nullptr) {
        const float sk = md.sink_logits[h] * kLog2e;
        const float mn = fmaxf(mx, sk);
        num = exp2f(mx - mn);
        den = den * num + exp2f(sk - mn);
        mx = mn;
      }
      const float val = den > 0.0f ? acc * num / den : 0.0f;
      char* dst = static_cast<char*>(p.o) + (row * p.o_stride_n + static_cast<int64_t>(h) * p.o_stride_h + d) * 2;
      if (fp16) *reinterpret_cast<__half*>(dst) = __float2half_rn(val);
      else *reinterpret_cast<__nv_bfloat16*>(dst) = __float2bfloat16_rn(val);
      if (d == 0 && p.lse != nullptr) p.lse[row * Hq + h] = den > 0.0f ? (mx + log2f(den)) * kLn2 : ninf();
    }
  }
}

// ---- prefill -------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

/// Rows of W fp16 elements, W % 64 == 0: XOR swizzle of the 16-byte chunk index by row & 7, so the
/// eight rows of an ldmatrix land on eight distinct 16-byte bank groups (prefill.cu Lay).
template <int W>
struct Lay {
  static_assert(W % 64 == 0, "swizzled rows");
  static __device__ __forceinline__ int off(int row, int chunk) { return row * W + ((chunk ^ (row & 7)) << 3); }
};

/// Packed query rows per CTA: 64, or 32 where the [q_rot | q_sk] tile of 64 rows does not fit.
KOPS_HOST_DEVICE constexpr int pf_bq(int d, bool qjl) { return (d == 256 && qjl) ? 32 : 64; }
/// Row width of the Q and K tiles: [levels | signs] for kTurboProd.
KOPS_HOST_DEVICE constexpr int pf_dx(int d, bool qjl) { return qjl ? 2 * d : d; }
/// K/V tile width: 64 keys while Q + K + V leave two CTAs resident in 100 KiB, else 32.
KOPS_HOST_DEVICE constexpr int pf_bkv(int d, bool qjl) {
  return (pf_bq(d, qjl) * pf_dx(d, qjl) + 64 * pf_dx(d, qjl) + 64 * d) * 2 <= 50 * 1024 ? 64 : 32;
}
KOPS_HOST_DEVICE constexpr int pf_smem(int d, bool qjl) {
  return (pf_bq(d, qjl) * pf_dx(d, qjl) + pf_bkv(d, qjl) * pf_dx(d, qjl) + pf_bkv(d, qjl) * d) * 2;
}

/// Dequantize records [t0, t0 + BKV) of one side into shared memory as fp16, rows of W elements:
/// chunk c < D / 8 holds scale * levels[code] of elements [8 c, 8 c + 8), chunk D / 8 + c (W = 2 D,
/// kTurboProd K) holds +-gain of sign bits [8 c, 8 c + 8).  Keys at or past `hi` are zero rows.
/// The global reads of a thread are issued in groups before the first convert / store.
template <int D, int W, int BKV, int THREADS>
__device__ __forceinline__ void load_tile(__half* s, const char* base, int64_t sp, int64_t st,
                                          const int32_t* pages, int ps, int t0, int hi,
                                          const LowBitSide& sd) {
  constexpr int CH = W / 8, CHM = D / 8, N = BKV * CH, IT = N / THREADS, GRP = 4;
  static_assert(N % THREADS == 0 && IT % GRP == 0, "whole groups of chunks per thread");
  const int tid = static_cast<int>(threadIdx.x);
  const RecLay rl = rec_lay<D>(sd);
#pragma unroll
  for (int g = 0; g < IT; g += GRP) {
    uint32_t w[GRP];
    float sc[GRP];
    int rr[GRP], cc[GRP];
#pragma unroll
    for (int j = 0; j < GRP; ++j) {
      const int idx = (g + j) * THREADS + tid;
      rr[j] = idx / CH;
      cc[j] = idx % CH;
      const int pos = t0 + rr[j];
      w[j] = 0u;
      sc[j] = 0.0f;
      if (pos < hi) {
        const char* rec = base + static_cast<int64_t>(pages[pos / ps]) * sp + static_cast<int64_t>(pos % ps) * st;
        if (cc[j] < CHM) {
          w[j] = load_bits(rec, cc[j] * rl.bits);
          sc[j] = *reinterpret_cast<const float*>(rec + rl.scale_off);
        } else {
          w[j] = *reinterpret_cast<const uint8_t*>(rec + rl.sign_off + (cc[j] - CHM));
          sc[j] = *reinterpret_cast<const float*>(rec + rl.scale_off + 4);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < GRP; ++j) {
      float x[8];
      if (cc[j] < CHM) {
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = sc[j] * sd.levels[(w[j] >> (e * rl.bits)) & rl.mask];
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = flip(sc[j], (~w[j] >> e) & 1u);
      }
      const uint4 v = make_uint4(pack2h(x[0], x[1]), pack2h(x[2], x[3]), pack2h(x[4], x[5]), pack2h(x[6], x[7]));
      *reinterpret_cast<uint4*>(s + Lay<W>::off(rr[j], cc[j])) = v;
    }
  }
}

/// FA2 body (prefill.cu fwd_body) over dequantized tiles.
template <int D, int BQ, bool QJL>
__device__ __forceinline__ void prefill_body(const PagedPrefillParams& p, const LowBitTables& tb) {
  constexpr int DX = pf_dx(D, QJL), BKV = pf_bkv(D, QJL), THREADS = BQ * 2;
  static_assert(BQ == pf_bq(D, QJL), "q tile of the plan");
  constexpr bool kQRegs = DX <= 128;
  constexpr int KQ = DX / 16, NS = BKV / 8, KP = BKV / 16, NO = D / 8;
  constexpr float rn = rsqrt_dim<D>();
  using LQ = Lay<DX>;
  using LV = Lay<D>;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);
  __half* sK = sQ + BQ * DX;
  __half* sV = sK + BKV * DX;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hk = blockIdx.y;
  const WorkPlan& wp = p.plan;
  // Heaviest item first (prefill.cu): the arithmetic of an item never depends on its CTA.
  const int item = wp.padded_num_items - 1 - static_cast<int>(blockIdx.x);
  if (wp.block_valid_mask != nullptr && wp.block_valid_mask[item] == 0) return;
  const int b = wp.request_indices[item];
  const int qtile = wp.qo_tile_indices[item];
  const int kvc = wp.kv_tile_indices != nullptr ? wp.kv_tile_indices[item] : 0;
  int chunk_sz = wp.kv_chunk_size != nullptr ? *wp.kv_chunk_size : INT_MAX;
  if (chunk_sz <= 0) chunk_sz = INT_MAX;

  const int Hq = p.num_qo_heads;
  const int G = Hq / p.num_kv_heads;
  const int qo_start = p.qo_indptr[b];
  const int qo_len = p.qo_indptr[b + 1] - qo_start;
  const int rows_total = qo_len * G;
  const int row0 = qtile * BQ;
  if (row0 >= rows_total) return;

  // ---- K/V records of (request b, kv head hk)
  const kv::PagedKVDesc& kd = p.kv;
  const char* kb = static_cast<const char*>(kd.k_data) + hk * kd.k_stride_head;
  const char* vb = static_cast<const char*>(kd.v_data) + hk * kd.v_stride_head;
  const int32_t* pages;
  int kv_len;
  if (kd.table_format == kv::PageTableFormat::kBlockTable) {
    pages = kd.block_table + static_cast<int64_t>(b) * kd.block_table_stride;
    kv_len = kd.seq_lens[b];
  } else {
    const int p0 = kd.kv_indptr[b], np = kd.kv_indptr[b + 1] - p0;
    pages = kd.kv_page_indices + p0;
    kv_len = np > 0 ? (np - 1) * kd.page_size + kd.kv_last_page_len[b] : 0;
  }
  const int ps = kd.page_size;
  const AttentionModifiers& md = p.mod;
  const int pos0 = p.q_pos_offset != nullptr ? p.q_pos_offset[b] : kv_len - qo_len;
  const float sm = md.sm_scale != 0.0f ? md.sm_scale : rn;
  const float scale_nat = sm, scale2 = scale_nat * kLog2e;
  const float cap = md.logits_softcap, inv_cap = cap > 0.0f ? 1.0f / cap : 0.0f;
  const bool causal = p.mask == MaskMode::kCausal;
  const bool custom = p.mask == MaskMode::kCustom;
  const int wl = md.window_left, wr = md.window_right, ck = md.chunk_size, nsink = md.num_sink_tokens;
  const float* alibi = md.alibi_slopes;

  // ---- key range of this item, tightened by the mask over the CTA's query positions
  const int qmin = pos0 + row0 / G;
  const int qmax = pos0 + (min(row0 + BQ, rows_total) - 1) / G;
  const int64_t lo64 = static_cast<int64_t>(kvc) * chunk_sz;
  int lo = lo64 >= kv_len ? kv_len : static_cast<int>(lo64);
  int hi = static_cast<int>(min(static_cast<int64_t>(kv_len), lo64 + chunk_sz));
  if (causal) hi = min(hi, max(qmax + 1, 0));
  if (wl >= 0 && nsink == 0) lo = max(lo, qmin - wl);
  if (wr >= 0) hi = static_cast<int>(min(static_cast<int64_t>(hi), max(static_cast<int64_t>(qmax) + wr + 1, int64_t{0})));
  if (ck > 0) {
    lo = static_cast<int>(max(static_cast<int64_t>(lo), fdiv(qmin, ck) * ck));
    hi = static_cast<int>(min(static_cast<int64_t>(hi), (fdiv(qmax, ck) + 1) * ck));
  }
  // Window with sink tokens: keys in [nsink, qmin - wl) are hidden from every row of the CTA.
  const bool has_gap = wl >= 0 && nsink > 0 && qmin - wl > nsink;
  const int gap_hi = qmin - wl;

  // ---- per-thread rows g and g + 8 of this warp
  int qpos[2], qi_r[2], head[2], rlo[2], rhi[2], rwlo[2];
  bool rvalid[2];
  float slope2[2] = {0.0f, 0.0f};
  int64_t mbase[2] = {0, 0};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int pr = row0 + warp * 16 + (lane >> 2) + 8 * r;
    rvalid[r] = pr < rows_total;
    const int prc = rvalid[r] ? pr : rows_total - 1;
    qi_r[r] = prc / G;
    head[r] = hk * G + (prc - qi_r[r] * G);
    qpos[r] = pos0 + qi_r[r];
    int64_t h64 = hi - 1, l64 = lo;
    if (causal) h64 = min(h64, static_cast<int64_t>(qpos[r]));
    if (wr >= 0) h64 = min(h64, static_cast<int64_t>(qpos[r]) + wr);
    if (ck > 0) {
      const int64_t c0 = fdiv(qpos[r], ck) * ck;
      l64 = max(l64, c0);
      h64 = min(h64, c0 + ck - 1);
    }
    rlo[r] = static_cast<int>(l64);
    rhi[r] = static_cast<int>(max(h64, static_cast<int64_t>(-1)));
    rwlo[r] = wl >= 0 ? static_cast<int>(max(static_cast<int64_t>(qpos[r]) - wl, static_cast<int64_t>(INT_MIN))) : INT_MIN;
    if (alibi != nullptr) slope2[r] = alibi[head[r]] * kLog2e;
    if (custom) mbase[r] = p.mask_indptr[b] + static_cast<int64_t>(qi_r[r]) * kv_len;
  }

  // ---- prologue: Q tile, one row per warp at a time (D / 32 elements per lane), rotated to
  // [q_rot | q_sk] and stored as fp16; then the first K tile.
  {
    constexpr int VEC = D / 32;
    const bool fp16 = tb.io_fp16 != 0;
    const int e0 = lane * VEC;
    const uint32_t sw_k = lowbit_sign_word(tb.seed, kSignStreamKey, static_cast<uint32_t>(e0) >> 5) >> (e0 & 31);
    [[maybe_unused]] const uint32_t sw_s =
        lowbit_sign_word(tb.seed, kSignStreamSketch, static_cast<uint32_t>(e0) >> 5) >> (e0 & 31);
    for (int i = 0; i < 16; ++i) {
      const int r = warp * 16 + i, pr = row0 + r;
      const bool ok = pr < rows_total;
      const int qi = ok ? pr / G : 0;
      const int h = hk * G + (ok ? pr - qi * G : 0);
      float v[VEC];
#pragma unroll
      for (int e = 0; e < VEC; ++e) v[e] = 0.0f;
      if (ok) {
        const char* src = static_cast<const char*>(p.q) +
                          ((static_cast<int64_t>(qo_start) + qi) * p.q_stride_n + static_cast<int64_t>(h) * p.q_stride_h + e0) * 2;
#pragma unroll
        for (int e = 0; e < VEC; e += 2) {
          const float2 f = unpack2(*reinterpret_cast<const uint32_t*>(src + e * 2), fp16);
          v[e] = f.x;
          v[e + 1] = f.y;
        }
      }
      if (tb.k.rotated != 0) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) v[e] = flip(v[e], (sw_k >> e) & 1u);
        wht_lane(v);
        wht_cross(v, lane, 32);
#pragma unroll
        for (int e = 0; e < VEC; ++e) v[e] *= rn;
      }
#pragma unroll
      for (int e = 0; e < VEC; e += 2) {
        const int col = e0 + e;
        *reinterpret_cast<uint32_t*>(sQ + LQ::off(r, col >> 3) + (col & 7)) = pack2h(v[e], v[e + 1]);
      }
      if constexpr (QJL) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) v[e] = flip(v[e], (sw_s >> e) & 1u);
        wht_lane(v);
        wht_cross(v, lane, 32);
#pragma unroll
        for (int e = 0; e < VEC; e += 2) {
          const int col = D + e0 + e;
          *reinterpret_cast<uint32_t*>(sQ + LQ::off(r, col >> 3) + (col & 7)) = pack2h(v[e] * rn, v[e + 1] * rn);
        }
      }
    }
  }
  int t0 = lo;
  if (has_gap && t0 >= nsink && t0 + BKV <= gap_hi) t0 = gap_hi;
  if (t0 < hi) load_tile<D, DX, BKV, THREADS>(sK, kb, kd.k_stride_page, kd.k_stride_token, pages, ps, t0, hi, tb.k);

  float m[2] = {ninf(), ninf()}, l[2] = {0.0f, 0.0f};
  float o[NO][4];
#pragma unroll
  for (int i = 0; i < NO; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  [[maybe_unused]] uint32_t qf[kQRegs ? KQ : 1][4];

  const uint32_t sQ_u = smem_u32(sQ), sK_u = smem_u32(sK), sV_u = smem_u32(sV);
  const int a_row = warp * 16 + (lane & 15), a_cs = lane >> 4;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_cs = (lane >> 3) & 1;
  const int v_row = (lane & 7) + (((lane >> 3) & 1) << 3), v_cs = lane >> 4;

  bool first = true;
  while (t0 < hi) {
    __syncthreads();  // Q (first pass) and K(t0) are in shared memory; every warp is done with sV
    if constexpr (kQRegs) {
      if (first) {
#pragma unroll
        for (int kk = 0; kk < KQ; ++kk) ldsm_x4(qf[kk], sQ_u + LQ::off(a_row, 2 * kk + a_cs) * 2);
      }
    }
    first = false;
    load_tile<D, D, BKV, THREADS>(sV, vb, kd.v_stride_page, kd.v_stride_token, pages, ps, t0, hi, tb.v);

    // S = [Q_rot | Q_sk] [K_levels | K_signs]^T
    float s[NS][4];
#pragma unroll
    for (int i = 0; i < NS; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < KQ; ++kk) {
      uint32_t af[4];
      if constexpr (kQRegs) {
#pragma unroll
        for (int i = 0; i < 4; ++i) af[i] = qf[kk][i];
      } else {
        ldsm_x4(af, sQ_u + LQ::off(a_row, 2 * kk + a_cs) * 2);
      }
#pragma unroll
      for (int np = 0; np < NS / 2; ++np) {
        uint32_t bb[4];
        ldsm_x4(bb, sK_u + LQ::off(np * 16 + b_row, 2 * kk + b_cs) * 2);
        mma16816(s[2 * np], af, bb[0], bb[1]);
        mma16816(s[2 * np + 1], af, bb[2], bb[3]);
      }
    }

    // scores -> base-2 logits: softcap, ALiBi, mask
    if (cap > 0.0f) {
#pragma unroll
      for (int nt = 0; nt < NS; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] = cap * tanh_fast(s[nt][e] * scale_nat * inv_cap) * kLog2e;
    } else {
#pragma unroll
      for (int nt = 0; nt < NS; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] *= scale2;
    }
    const int jb = t0 + 2 * (lane & 3);
    if (alibi != nullptr) {
#pragma unroll
      for (int nt = 0; nt < NS; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] += slope2[e >> 1] * static_cast<float>(jb + nt * 8 + (e & 1) - qpos[e >> 1]);
    }
    bool full = !custom && t0 + BKV <= hi;
    if (causal) full = full && t0 + BKV - 1 <= qmin;
    if (wl >= 0) full = full && t0 >= qmax - wl;
    if (wr >= 0) full = full && t0 + BKV - 1 <= qmin + wr;
    if (ck > 0) full = full && fdiv(qmax, ck) == fdiv(qmin, ck) && fdiv(t0, ck) == fdiv(qmin, ck) &&
                      fdiv(t0 + BKV - 1, ck) == fdiv(qmin, ck);
    if (!full) {
#pragma unroll
      for (int nt = 0; nt < NS; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = e >> 1, j = jb + nt * 8 + (e & 1);
          bool vis = j >= rlo[r] && j <= rhi[r] && (j >= rwlo[r] || j < nsink);
          if (custom && vis) {
            const int64_t bit = mbase[r] + j;
            vis = ((p.custom_mask[bit >> 3] >> (bit & 7)) & 1) != 0;
          }
          if (!vis) s[nt][e] = ninf();
        }
      }
    }

    // online softmax
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int nt = 0; nt < NS; ++nt) {
      mx[0] = fmaxf(mx[0], fmaxf(s[nt][0], s[nt][1]));
      mx[1] = fmaxf(mx[1], fmaxf(s[nt][2], s[nt][3]));
    }
    float mu[2], alpha[2], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      mu[r] = mx[r] == ninf() ? 0.0f : mx[r];
      alpha[r] = exp2f(m[r] - mu[r]);
      m[r] = mx[r];
    }
#pragma unroll
    for (int nt = 0; nt < NS; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s[nt][e] = exp2f(s[nt][e] - mu[e >> 1]);
        rs[e >> 1] += s[nt][e];
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) l[r] = l[r] * alpha[r] + rs[r];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (alpha[r] != 1.0f) {
#pragma unroll
        for (int i = 0; i < NO; ++i) {
          o[i][2 * r] *= alpha[r];
          o[i][2 * r + 1] *= alpha[r];
        }
      }
    }

    __syncthreads();  // V(t0) is in shared memory; every warp is done with sK
    int tn = t0 + BKV;
    if (has_gap && tn >= nsink && tn + BKV <= gap_hi) tn = gap_hi;
    if (tn < hi) load_tile<D, DX, BKV, THREADS>(sK, kb, kd.k_stride_page, kd.k_stride_token, pages, ps, tn, hi, tb.k);

    // O_rot += P V_rot
#pragma unroll
    for (int kk = 0; kk < KP; ++kk) {
      uint32_t af[4];
      af[0] = pack2h(s[2 * kk][0], s[2 * kk][1]);
      af[1] = pack2h(s[2 * kk][2], s[2 * kk][3]);
      af[2] = pack2h(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      af[3] = pack2h(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dp = 0; dp < NO / 2; ++dp) {
        uint32_t bb[4];
        ldsm_x4_t(bb, sV_u + LV::off(kk * 16 + v_row, 2 * dp + v_cs) * 2);
        mma16816(o[2 * dp], af, bb[0], bb[1]);
        mma16816(o[2 * dp + 1], af, bb[2], bb[3]);
      }
    }
    t0 = tn;
  }

  // ---- rotate the rows back: out = s_v .* Hn o.  o[i][2 r + c] is column 8 i + 2 (lane & 3) + c
  // of row r, so the stages over the column bits are: c inside the thread, lane & 3 by shuffles,
  // i inside the thread.
  if (tb.v.rotated != 0) {
#pragma unroll
    for (int i = 0; i < NO; ++i) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float a = o[i][2 * r], bb = o[i][2 * r + 1];
        o[i][2 * r] = a + bb;
        o[i][2 * r + 1] = a - bb;
      }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      const bool upper = (lane & off) != 0;
#pragma unroll
      for (int i = 0; i < NO; ++i) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t = __shfl_xor_sync(0xffffffffu, o[i][e], off);
          o[i][e] = upper ? t - o[i][e] : o[i][e] + t;
        }
      }
    }
#pragma unroll
    for (int h = 1; h < NO; h *= 2) {
#pragma unroll
      for (int i0 = 0; i0 < NO; i0 += 2 * h) {
#pragma unroll
        for (int i = i0; i < i0 + h; ++i) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float a = o[i][e], bb = o[i + h][e];
            o[i][e] = a + bb;
            o[i + h][e] = a - bb;
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NO; ++i) {
      const uint32_t sw = lowbit_sign_word(tb.seed, kSignStreamValue, static_cast<uint32_t>(8 * i) >> 5) >>
                          (((8 * i) & 31) + 2 * (lane & 3));
#pragma unroll
      for (int e = 0; e < 4; ++e) o[i][e] = flip(o[i][e] * rn, (sw >> (e & 1)) & 1u);
    }
  }

  // ---- finalize
  const bool split = wp.split_kv && p.split.o_partial != nullptr;
  const bool fp16 = tb.io_fp16 != 0;
  float scale[2], lse_n[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    float mf = m[r], lf = l[r], num = 1.0f;
    if (!split && md.sink_logits != nullptr) {
      const float sk = md.sink_logits[head[r]] * kLog2e;
      const float mn = fmaxf(mf, sk);
      num = exp2f(mf - mn);
      lf = lf * num + exp2f(sk - mn);
      mf = mn;
    }
    scale[r] = lf > 0.0f ? num / lf : 0.0f;
    lse_n[r] = lf > 0.0f ? (mf + log2f(lf)) * kLn2 : ninf();
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (!rvalid[r]) continue;
    const int64_t tok = static_cast<int64_t>(qo_start) + qi_r[r];
    if (split) {
      const int64_t prow = static_cast<int64_t>(p.split.merge_indptr[tok]) + kvc;
      float* dst = p.split.o_partial + (prow * Hq + head[r]) * D + 2 * (lane & 3);
#pragma unroll
      for (int i = 0; i < NO; ++i) {
        *reinterpret_cast<float2*>(dst + i * 8) = make_float2(o[i][2 * r] * scale[r], o[i][2 * r + 1] * scale[r]);
      }
      if ((lane & 3) == 0) p.split.lse_partial[prow * Hq + head[r]] = lse_n[r];
    } else {
      char* dst = static_cast<char*>(p.o) +
                  (tok * p.o_stride_n + static_cast<int64_t>(head[r]) * p.o_stride_h + 2 * (lane & 3)) * 2;
#pragma unroll
      for (int i = 0; i < NO; ++i) {
        *reinterpret_cast<uint32_t*>(dst + i * 16) = pack2(o[i][2 * r] * scale[r], o[i][2 * r + 1] * scale[r], fp16);
      }
      if (p.lse != nullptr && (lane & 3) == 0) p.lse[tok * Hq + head[r]] = lse_n[r];
    }
  }
}

}  // namespace

// ---- kernels (contract: lowbit_kernels.cuh) ------------------------------------------------------
template <int HEAD_DIM>
__global__ void __launch_bounds__(kStoreThreads)
    lowbit_kv_store_kernel(const __grid_constant__ LowBitStoreParams params) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256, "Hadamard head dim");
  __shared__ uint8_t sm[kStoreWarps][HEAD_DIM + 32];
  const int warp = threadIdx.x >> 5;
  const kv::PagedKVDesc& c = params.cache;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * kStoreWarps + warp;
  const int64_t token = pair / c.num_kv_heads;
  const int64_t head = pair - token * c.num_kv_heads;
  if (token >= params.num_tokens) return;
  const int64_t slot = params.slot_mapping[token];
  if (slot < 0) return;
  const int64_t page = slot / c.page_size, off = slot - page * c.page_size;
  char* kr = static_cast<char*>(c.k_data) + page * c.k_stride_page + off * c.k_stride_token + head * c.k_stride_head;
  char* vr = static_cast<char*>(c.v_data) + page * c.v_stride_page + off * c.v_stride_token + head * c.v_stride_head;
  const char* ks = static_cast<const char*>(params.key) + (token * params.k_stride_n + head * params.k_stride_h) * 2;
  const char* vs = static_cast<const char*>(params.value) + (token * params.v_stride_n + head * params.v_stride_h) * 2;
  const bool fp16 = params.tables.io_fp16 != 0;
  store_side<HEAD_DIM>(ks, fp16, params.tables.k, params.tables.seed, kr, sm[warp]);
  store_side<HEAD_DIM>(vs, fp16, params.tables.v, params.tables.seed, vr, sm[warp]);
}

template <int HEAD_DIM, int ROWS_PER_CTA>
__global__ void __launch_bounds__(kDecThreads)
    lowbit_decode_kernel(const __grid_constant__ attn::DecodeParams params,
                         const __grid_constant__ LowBitTables tables) {
  decode_body<HEAD_DIM, ROWS_PER_CTA>(params, tables);
}

template <int HEAD_DIM, int CTA_TILE_Q, bool kQjl>
__global__ void __launch_bounds__(CTA_TILE_Q * 2)
    lowbit_prefill_kernel(const __grid_constant__ attn::PagedPrefillParams params,
                          const __grid_constant__ LowBitTables tables) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  prefill_body<HEAD_DIM, CTA_TILE_Q, kQjl>(params, tables);
#else
  (void)params;
  (void)tables;
  if (threadIdx.x == 0) __trap();  // host refuses sm < 80
#endif
}

// =================================================================================================
// Host
// =================================================================================================
namespace {

constexpr uint64_t kLowBitKind = 0xB7;
constexpr int kDecRows = 4;               // packed query rows per decode CTA
constexpr int kDecodeInvariantChunk = 2048;
constexpr int kDecodeMaxSplits = 32;
constexpr int kPrefillInvariantChunk = 4096;
constexpr int kPrefillMaxSplits = 16;

#define KOPS_LOWBIT_DIMS(X) X(64) X(128) X(256)

struct FormatInfo {
  int bits = 0;        // code bits per element
  bool qjl = false, absmax = false, ok = false;
};

FormatInfo format_info(LowBitFormat f) {
  switch (f) {
    case LowBitFormat::kUniform4: return {4, false, true, true};
    case LowBitFormat::kUniform3: return {3, false, true, true};
    case LowBitFormat::kTurboMse2: return {2, false, false, true};
    case LowBitFormat::kTurboMse3: return {3, false, false, true};
    case LowBitFormat::kTurboMse4: return {4, false, false, true};
    case LowBitFormat::kTurboProd2: return {1, true, false, true};
    case LowBitFormat::kTurboProd3: return {2, true, false, true};
    case LowBitFormat::kTurboProd4: return {3, true, false, true};
    default: return {};
  }
}

const float* lloyd_max(int d, int bits) {
  switch (d * 8 + bits) {
    case 64 * 8 + 1: return kLloydMax64B1;
    case 64 * 8 + 2: return kLloydMax64B2;
    case 64 * 8 + 3: return kLloydMax64B3;
    case 64 * 8 + 4: return kLloydMax64B4;
    case 128 * 8 + 1: return kLloydMax128B1;
    case 128 * 8 + 2: return kLloydMax128B2;
    case 128 * 8 + 3: return kLloydMax128B3;
    case 128 * 8 + 4: return kLloydMax128B4;
    case 256 * 8 + 1: return kLloydMax256B1;
    case 256 * 8 + 2: return kLloydMax256B2;
    case 256 * 8 + 3: return kLloydMax256B3;
    case 256 * 8 + 4: return kLloydMax256B4;
    default: return nullptr;
  }
}

bool dim_ok(int d) { return d == 64 || d == 128 || d == 256; }

bool fill_side(LowBitFormat f, QuantTransform tr, int d, uint32_t stream, bool is_k, LowBitSide* s) {
  const FormatInfo fi = format_info(f);
  if (!fi.ok || (fi.qjl && !is_k)) return false;
  if (tr != QuantTransform::kNone && tr != QuantTransform::kSignedHadamard) return false;
  *s = LowBitSide{};
  s->code_bits = fi.bits;
  s->qjl = fi.qjl ? 1 : 0;
  s->rotated = tr == QuantTransform::kSignedHadamard ? 1 : 0;
  s->absmax = fi.absmax ? 1 : 0;
  s->entry_bytes = d * fi.bits / 8 + (fi.qjl ? d / 8 + 8 : 4);
  s->sign_stream = stream;
  const int n = 1 << fi.bits;
  if (fi.absmax) {
    for (int i = 0; i < n; ++i) s->levels[i] = static_cast<float>(2 * i - (n - 1));
  } else {
    const float* c = lloyd_max(d, fi.bits);
    for (int i = 0; i < n / 2; ++i) {
      s->levels[n / 2 + i] = c[i];
      s->levels[n / 2 - 1 - i] = -c[i];
    }
  }
  for (int t = 0; t < 16; ++t) {
    s->thresholds[t] = t < n - 1 ? (s->levels[t] + s->levels[t + 1]) * 0.5f : HUGE_VALF;
  }
  return true;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

Status check_ctx(const LaunchContext& ctx, const char* who) {
  if (ctx_is_amd(ctx)) return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA only", who);
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "%s: needs sm_80+", who);
  }
  return Status::kSuccess;
}

/// A low-bit cache descriptor the kernels can address (ops.h).  `read`: page table required.
Status check_cache(const kv::PagedKVDesc& d, const LowBitTables& tb, bool read, const char* who) {
  if (d.k_data == nullptr || d.v_data == nullptr || d.page_size <= 0 || d.num_kv_heads <= 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: k_data, v_data, page_size, num_kv_heads required", who);
  }
  if (d.k_codec != kv::KVCodec::kLowBitWHT || d.v_codec != kv::KVCodec::kLowBitWHT) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: cache codec must be kLowBitWHT", who);
  }
  if (d.head_dim_k != tb.head_dim || d.head_dim_v != tb.head_dim) {
    return detail::set_error(Status::kUnsupportedShape, "%s: cache head dims %d/%d, codec %d", who,
                             d.head_dim_k, d.head_dim_v, tb.head_dim);
  }
  if (d.layout != kv::KVLayout::kNHD && d.layout != kv::KVLayout::kHND) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: layout must be NHD or HND", who);
  }
  if (d.stride_unit != StrideUnit::kByte || (d.entry_bytes != 0 && d.entry_bytes != tb.k.entry_bytes)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: low-bit caches use byte strides and entry_bytes = %d (the K record)", who,
                             tb.k.entry_bytes);
  }
  if ((reinterpret_cast<uintptr_t>(d.k_data) & 15u) != 0 || (reinterpret_cast<uintptr_t>(d.v_data) & 15u) != 0 ||
      ((d.k_stride_page | d.k_stride_token | d.k_stride_head | d.v_stride_page | d.v_stride_token |
        d.v_stride_head) & 3) != 0) {
    return detail::set_error(Status::kMisaligned, "%s: pages 16-byte aligned, strides multiples of 4 bytes", who);
  }
  if (!read) return Status::kSuccess;
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    if (d.block_table == nullptr || d.seq_lens == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "%s: block_table and seq_lens required", who);
    }
  } else if (d.table_format != kv::PageTableFormat::kCSR || d.kv_indptr == nullptr ||
             d.kv_page_indices == nullptr || d.kv_last_page_len == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: CSR page table or block table required", who);
  }
  return Status::kSuccess;
}

/// The part of an AttentionSpec every low-bit attention entry shares.
Status check_spec(const AttentionSpec& spec, const LowBitTables& tb, const char* who) {
  if (spec.num_q_heads <= 0 || spec.num_kv_heads <= 0 || spec.num_q_heads % spec.num_kv_heads != 0) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_q_heads must be a multiple of num_kv_heads", who);
  }
  if (spec.head_dim_qk != tb.head_dim || spec.head_dim_vo != tb.head_dim) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head dims %d/%d, codec %d", who, spec.head_dim_qk,
                             spec.head_dim_vo, tb.head_dim);
  }
  const bool q16 = spec.q_dtype == DType::kBFloat16 || spec.q_dtype == DType::kFloat16;
  if (!q16 || spec.o_dtype != spec.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: q/o must be the same bf16 or fp16 dtype", who);
  }
  if (spec.k_codec != kv::KVCodec::kLowBitWHT || spec.v_codec != kv::KVCodec::kLowBitWHT ||
      spec.kv_dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: K/V must be kLowBitWHT records (kv_dtype kUInt8)", who);
  }
  if (spec.page_size <= 0 || (spec.kv_layout != kv::KVLayout::kNHD && spec.kv_layout != kv::KVLayout::kHND)) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: paged NHD or HND K/V required", who);
  }
  if (spec.fp8_tensor_core) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: no FP8 compute over low-bit K/V", who);
  }
  return Status::kSuccess;
}

int spec_mods(const AttentionSpec& s) {
  return (s.sliding_window ? 1 : 0) | (s.softcap ? 2 : 0) | (s.alibi ? 4 : 0) | (s.sinks ? 8 : 0) |
         (s.chunked ? 16 : 0);
}

/// Modifiers the spec did not enable are neutral, whatever the caller left in params.mod.
void neutralize(int mods, AttentionModifiers* m) {
  if ((mods & 1) == 0) {
    m->window_left = -1;
    m->window_right = -1;
    m->num_sink_tokens = 0;
  }
  if ((mods & 2) == 0) m->logits_softcap = 0.0f;
  if ((mods & 4) == 0) m->alibi_slopes = nullptr;
  if ((mods & 8) == 0) m->sink_logits = nullptr;
  if ((mods & 16) == 0) m->chunk_size = 0;
}

/// Everything run() needs, packed into AttentionPlan::kernel_id.
struct Key {
  int d = 0, splits = 1, q_len = 1, dt = 0, mods = 0, mask = 0, qjl = 0, split = 0, prefill = 0;
};
uint64_t encode(const Key& k) {
  return (kLowBitKind << 56) | (static_cast<uint64_t>(k.prefill) << 53) | (static_cast<uint64_t>(k.split) << 52) |
         (static_cast<uint64_t>(k.qjl) << 51) | (static_cast<uint64_t>(k.mask) << 49) |
         (static_cast<uint64_t>(k.mods) << 44) | (static_cast<uint64_t>(k.dt) << 42) |
         (static_cast<uint64_t>(k.q_len) << 32) | (static_cast<uint64_t>(k.splits) << 16) |
         static_cast<uint64_t>(k.d);
}
bool decode_key(uint64_t id, Key* k) {
  if ((id >> 56) != kLowBitKind) return false;
  k->prefill = static_cast<int>((id >> 53) & 1);
  k->split = static_cast<int>((id >> 52) & 1);
  k->qjl = static_cast<int>((id >> 51) & 1);
  k->mask = static_cast<int>((id >> 49) & 3);
  k->mods = static_cast<int>((id >> 44) & 31);
  k->dt = static_cast<int>((id >> 42) & 1);
  k->q_len = static_cast<int>((id >> 32) & 0x3FF);
  k->splits = static_cast<int>((id >> 16) & 0xFFFF);
  k->d = static_cast<int>(id & 0xFFFF);
  return true;
}

Tensor view_o(const void* o, DType dt, int64_t rows, int Hq, int dvo, int64_t sn, int64_t sh, int dev) {
  Tensor t{};
  t.data = const_cast<void*>(o);
  t.dtype = dt;
  t.ndim = 3;
  t.device = dev;
  t.shape[0] = rows;
  t.shape[1] = Hq;
  t.shape[2] = dvo;
  t.stride[0] = sn;
  t.stride[1] = sh;
  t.stride[2] = 1;
  return t;
}

Tensor view_1d(const void* p, DType dt, int64_t n, int dev) {
  Tensor t{};
  t.data = const_cast<void*>(p);
  t.dtype = dt;
  t.ndim = 1;
  t.device = dev;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

/// Scratch = [o_partial | lse_partial], 256-byte aligned regions sized for `rows` partial rows.
size_t scratch_bytes(int64_t rows, int Hq, int d) {
  return align_up(static_cast<size_t>(rows) * Hq * d * 4, 256) + align_up(static_cast<size_t>(rows) * Hq * 4, 256);
}

Status merge_partials(const Key& k, const void* o, int64_t o_sn, int64_t o_sh, float* lse_ptr, int64_t rows,
                      int Hq, const SplitKVBuffers& split, const float* sink_logits, const void* total_rows,
                      const LaunchContext& ctx) {
  const Tensor o_t = view_o(o, k.dt == 0 ? DType::kBFloat16 : DType::kFloat16, rows, Hq, k.d, o_sn, o_sh, ctx.device);
  Tensor lse{};
  if (lse_ptr != nullptr) {
    lse = view_o(lse_ptr, DType::kFloat32, rows, Hq, 1, Hq, 1, ctx.device);
    lse.ndim = 2;
  }
  const Tensor sinks = sink_logits != nullptr ? view_1d(sink_logits, DType::kFloat32, Hq, ctx.device) : Tensor{};
  const Tensor tr = total_rows != nullptr ? view_1d(total_rows, DType::kUInt32, 1, ctx.device) : Tensor{};
  return attn::internal::merge_split_kv_run(o_t, lse, split, sinks, tr, ctx);
}

// ---- store ---------------------------------------------------------------------------------------
template <int D>
Status launch_store(const LowBitStoreParams& p, cudaStream_t st) {
  const int64_t pairs = static_cast<int64_t>(p.num_tokens) * p.cache.num_kv_heads;
  const int64_t grid = (pairs + kStoreWarps - 1) / kStoreWarps;
  if (grid > INT32_MAX) return detail::set_error(Status::kCapacityExceeded, "lowbit_kv_store: %lld CTAs", static_cast<long long>(grid));
  lowbit_kv_store_kernel<D><<<static_cast<unsigned>(grid), kStoreThreads, 0, st>>>(p);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("lowbit_kv_store launch", err);
}

// ---- decode --------------------------------------------------------------------------------------
/// device_persistent of a decode plan: chunk | total rows | merge_indptr.
struct DCarve {
  size_t chunk, total_rows, merge, bytes;
};
DCarve dcarve(int64_t rows) {
  DCarve c{};
  c.chunk = 0;
  c.total_rows = 256;
  c.merge = 512;
  c.bytes = 512 + align_up(static_cast<size_t>(rows + 1) * 4, 256);
  return c;
}

template <int D>
Status launch_decode(const DecodeParams& p, const LowBitTables& tb, dim3 grid, cudaStream_t st) {
  auto* fn = lowbit_decode_kernel<D, kDecRows>;
  constexpr int smem = DecGeom<D, kDecRows>::kSmem;
  static const cudaError_t attr = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("lowbit_decode smem attribute", attr));
  fn<<<grid, kDecThreads, smem, st>>>(p, tb);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("lowbit_decode launch", err);
}

// ---- prefill -------------------------------------------------------------------------------------
/// device_persistent of a prefill plan: request | qo_tile | kv_tile | valid | chunk | total rows |
/// merge_indptr[total_q + 1].  Regions 256-byte aligned.
struct PCarve {
  size_t req, qtile, kvtile, valid, chunk, total_rows, merge, bytes;
};
PCarve pcarve(int64_t items, int64_t total_q) {
  PCarve c{};
  size_t o = 0;
  auto take = [&](size_t n) {
    const size_t at = o;
    o = align_up(o + n, 256);
    return at;
  };
  c.req = take(static_cast<size_t>(items) * 4);
  c.qtile = take(static_cast<size_t>(items) * 4);
  c.kvtile = take(static_cast<size_t>(items) * 4);
  c.valid = take(static_cast<size_t>(items));
  c.chunk = take(4);
  c.total_rows = take(4);
  c.merge = take(static_cast<size_t>(total_q + 1) * 4);
  c.bytes = o;
  return c;
}

template <int D, bool QJL>
Status launch_prefill(const PagedPrefillParams& p, const LowBitTables& tb, dim3 grid, cudaStream_t st) {
  constexpr int BQ = pf_bq(D, QJL), smem = pf_smem(D, QJL);
  auto* fn = lowbit_prefill_kernel<D, BQ, QJL>;
  static const cudaError_t attr = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("lowbit_prefill smem attribute", attr));
  fn<<<grid, BQ * 2, smem, st>>>(p, tb);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("lowbit_prefill launch", err);
}

Status read_host_i32(const Tensor& t, int64_t n, const char* name, std::vector<int32_t>* out) {
  if (t.data == nullptr || t.dtype != DType::kInt32 || t.ndim != 1 || t.shape[0] != n ||
      (t.stride[0] != 1 && n > 1) || t.device >= 0) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_plan: %s must be a host int32 [%lld]", name,
                             static_cast<long long>(n));
  }
  const int32_t* p = reinterpret_cast<const int32_t*>(static_cast<const char*>(t.data) + t.byte_offset);
  out->assign(p, p + n);
  return Status::kSuccess;
}

}  // namespace

// ---- ops.h ---------------------------------------------------------------------------------------
Status lowbit_codec(LowBitFormat k_format, LowBitFormat v_format, int32_t head_dim, uint32_t seed,
                    LowBitCodec* codec) {
  if (codec == nullptr) return detail::set_error(Status::kInvalidArgument, "lowbit_codec: null codec");
  if (!dim_ok(head_dim)) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit_codec: head_dim %d not in {64, 128, 256}", head_dim);
  }
  const FormatInfo fk = format_info(k_format), fv = format_info(v_format);
  if (!fk.ok || !fv.ok || fv.qjl) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "lowbit_codec: formats %d/%d (K: kUniform3/4, kTurboMse2-4, kTurboProd2-4; V: no kTurboProd)",
                             static_cast<int>(k_format), static_cast<int>(v_format));
  }
  codec->k_format = k_format;
  codec->v_format = v_format;
  codec->k_transform = QuantTransform::kSignedHadamard;
  // attention.rs keeps uniform V unrotated; the TurboQuant codebooks assume a rotated vector.
  codec->v_transform = fv.absmax ? QuantTransform::kNone : QuantTransform::kSignedHadamard;
  codec->transform_seed = seed;
  codec->packing_version = kLowBitPackingVersion;
  codec->head_dim = static_cast<int16_t>(head_dim);
  return Status::kSuccess;
}

Status lowbit_resolve(const LowBitCodec& codec, LowBitTables* tables) {
  if (tables == nullptr) return detail::set_error(Status::kInvalidArgument, "lowbit_resolve: null tables");
  if (codec.packing_version != kLowBitPackingVersion) {
    return detail::set_error(Status::kUnsupportedLayout, "lowbit: packing version %d, this build reads %d",
                             codec.packing_version, kLowBitPackingVersion);
  }
  if (!dim_ok(codec.head_dim)) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit: head_dim %d not in {64, 128, 256}", codec.head_dim);
  }
  *tables = LowBitTables{};
  if (!fill_side(codec.k_format, codec.k_transform, codec.head_dim, kSignStreamKey, true, &tables->k) ||
      !fill_side(codec.v_format, codec.v_transform, codec.head_dim, kSignStreamValue, false, &tables->v)) {
    return detail::set_error(Status::kUnsupportedDtype, "lowbit: formats %d/%d with transforms %d/%d are not compiled",
                             static_cast<int>(codec.k_format), static_cast<int>(codec.v_format),
                             static_cast<int>(codec.k_transform), static_cast<int>(codec.v_transform));
  }
  tables->seed = codec.transform_seed;
  tables->head_dim = codec.head_dim;
  tables->io_fp16 = 0;
  return Status::kSuccess;
}

int32_t lowbit_entry_bytes(const LowBitCodec& codec, bool is_k) {
  LowBitTables tb;
  if (lowbit_resolve(codec, &tb) != Status::kSuccess) return 0;
  return is_k ? tb.k.entry_bytes : tb.v.entry_bytes;
}

Status lowbit_kv_store(const Tensor& key, const Tensor& value, const kv::PagedKVDesc& cache,
                       const Tensor& slot_mapping, const LowBitCodec& codec, const LaunchContext& ctx) {
  Status s = check_ctx(ctx, "lowbit_kv_store");
  if (s != Status::kSuccess) return s;
  LowBitTables tb;
  s = lowbit_resolve(codec, &tb);
  if (s != Status::kSuccess) return s;
  if (key.dtype != DType::kBFloat16 && key.dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "lowbit_kv_store: key dtype %s (bf16 or fp16)", dtype_name(key.dtype));
  }
  s = check_tensor(key, key.dtype, 3, 16, "lowbit_kv_store.key");
  if (s != Status::kSuccess) return s;
  s = check_tensor(value, key.dtype, 3, 16, "lowbit_kv_store.value");
  if (s != Status::kSuccess) return s;
  s = check_cache(cache, tb, false, "lowbit_kv_store");
  if (s != Status::kSuccess) return s;
  const int64_t num_tokens = key.shape[0];
  if (key.shape[1] != cache.num_kv_heads || key.shape[2] != tb.head_dim || value.shape[0] != num_tokens ||
      value.shape[1] != key.shape[1] || value.shape[2] != key.shape[2]) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit_kv_store: key / value must be [T, %d, %d]",
                             cache.num_kv_heads, tb.head_dim);
  }
  if (key.stride_unit != StrideUnit::kElement || value.stride_unit != StrideUnit::kElement || key.stride[2] != 1 ||
      value.stride[2] != 1 || ((key.stride[0] | key.stride[1] | value.stride[0] | value.stride[1]) & 7) != 0 ||
      ((key.byte_offset | value.byte_offset) & 15) != 0) {
    return detail::set_error(Status::kMisaligned, "lowbit_kv_store: key / value rows must be dense and 16-byte aligned");
  }
  if (slot_mapping.data == nullptr || slot_mapping.dtype != DType::kInt64 || slot_mapping.ndim != 1 ||
      slot_mapping.shape[0] != num_tokens || (slot_mapping.stride[0] != 1 && num_tokens > 1)) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_kv_store: slot_mapping must be int64 [%lld]",
                             static_cast<long long>(num_tokens));
  }
  if (num_tokens > INT32_MAX) return detail::set_error(Status::kCapacityExceeded, "lowbit_kv_store: %lld tokens", static_cast<long long>(num_tokens));
  if (num_tokens == 0) return Status::kSuccess;
  LowBitStoreParams p{};
  p.key = static_cast<const char*>(key.data) + key.byte_offset;
  p.value = static_cast<const char*>(value.data) + value.byte_offset;
  p.k_stride_n = key.stride[0];
  p.k_stride_h = key.stride[1];
  p.v_stride_n = value.stride[0];
  p.v_stride_h = value.stride[1];
  p.cache = cache;
  p.slot_mapping = reinterpret_cast<const int64_t*>(static_cast<const char*>(slot_mapping.data) + slot_mapping.byte_offset);
  p.tables = tb;
  p.tables.io_fp16 = key.dtype == DType::kFloat16 ? 1 : 0;
  p.num_tokens = static_cast<int32_t>(num_tokens);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_D(a) if (tb.head_dim == (a)) return launch_store<a>(p, stream);
  KOPS_LOWBIT_DIMS(KOPS_D)
#undef KOPS_D
  return detail::set_error(Status::kUnsupportedShape, "lowbit_kv_store: head dim %d not compiled", tb.head_dim);
}

Status lowbit_decode_plan(const AttentionSpec& spec, const LowBitCodec& codec, int32_t max_batch,
                          int32_t max_kv_len, int32_t q_len, const LaunchContext& ctx,
                          const Workspace& ws, AttentionPlan* plan) {
  if (plan == nullptr) return detail::set_error(Status::kInvalidArgument, "lowbit_decode_plan: null plan");
  Status s = check_ctx(ctx, "lowbit_decode");
  if (s != Status::kSuccess) return s;
  LowBitTables tb;
  s = lowbit_resolve(codec, &tb);
  if (s != Status::kSuccess) return s;
  s = check_spec(spec, tb, "lowbit_decode");
  if (s != Status::kSuccess) return s;
  if (spec.mask == MaskMode::kCustom) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit_decode: custom masks go through lowbit_prefill");
  }
  if (q_len < 1 || q_len > 8 || max_batch < 1 || max_kv_len < 0) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_decode: q_len 1..8, max_batch >= 1, max_kv_len >= 0");
  }
  const int G = spec.num_q_heads / spec.num_kv_heads, R = G * q_len, D = tb.head_dim;
  Key k;
  k.d = D;
  k.dt = spec.q_dtype == DType::kFloat16 ? 1 : 0;
  k.mods = spec_mods(spec);
  k.q_len = q_len;
  k.qjl = tb.k.qjl;
  const int64_t rows = static_cast<int64_t>(max_batch) * q_len;
  const int64_t base = static_cast<int64_t>(max_batch) * spec.num_kv_heads * ((R + kDecRows - 1) / kDecRows);
  int chunk = 0, S = 1;
  if (ctx.batch_invariant) {
    chunk = kDecodeInvariantChunk;
    S = std::max(1, (max_kv_len + chunk - 1) / chunk);
    k.split = 1;
  } else {
    // Same rule as decode_plan: split while the grid is short of four CTAs per SM, and cut long
    // contexts into kDecodeInvariantChunk-key chunks while it is short of four waves.
    const int64_t target = static_cast<int64_t>(std::max(ctx.compute_units, 1)) * 4;
    if (base < target) {
      S = static_cast<int>(std::min<int64_t>({(target + base - 1) / base, kDecodeMaxSplits,
                                              std::max<int64_t>(1, max_kv_len / 256)}));
    }
    if (max_kv_len >= 4 * kDecodeInvariantChunk && base < 4 * target) {
      const int64_t desired = std::min<int64_t>(
          {static_cast<int64_t>(kDecodeMaxSplits),
           (static_cast<int64_t>(max_kv_len) + kDecodeInvariantChunk - 1) / kDecodeInvariantChunk,
           std::max<int64_t>(1, max_kv_len / 256)});
      if (desired > S) S = static_cast<int>(desired);
    }
    k.split = S > 1 ? 1 : 0;
  }
  if (S > 65535) return detail::set_error(Status::kCapacityExceeded, "lowbit_decode: %d splits", S);
  k.splits = S;
  const int64_t prow = k.split ? rows * S : 0;
  const DCarve cv = dcarve(rows);
  *plan = AttentionPlan{};
  plan->kernel_id = encode(k);
  plan->plan_generation = ctx.capture_generation;
  plan->strategy = ExecutionStrategy::kModular;
  plan->split_kv = k.split != 0;
  plan->max_batch = max_batch;
  plan->max_total_q = static_cast<int32_t>(rows);
  plan->max_kv_len = max_kv_len;
  plan->num_launches = 1 + (k.split ? 1 : 0);
  plan->workspace.device_persistent_bytes = cv.bytes;
  plan->workspace.device_scratch_bytes = k.split ? scratch_bytes(prow, spec.num_q_heads, D) : 0;
  plan->workspace.alignment = 256;
  if (ws.device_persistent == nullptr || ws.device_persistent_bytes < cv.bytes ||
      ws.device_scratch_bytes < plan->workspace.device_scratch_bytes ||
      (plan->workspace.device_scratch_bytes != 0 && ws.device_scratch == nullptr)) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "lowbit_decode_plan: needs %zu persistent + %zu scratch bytes (see plan->workspace)",
                             cv.bytes, plan->workspace.device_scratch_bytes);
  }
  if ((reinterpret_cast<uintptr_t>(ws.device_persistent) & 255u) != 0 ||
      (reinterpret_cast<uintptr_t>(ws.device_scratch) & 255u) != 0) {
    return detail::set_error(Status::kMisaligned, "lowbit_decode_plan: workspace must be 256-byte aligned");
  }
  std::vector<unsigned char> img(cv.bytes, 0);
  std::memcpy(img.data() + cv.chunk, &chunk, 4);
  const uint32_t total_rows = static_cast<uint32_t>(rows);
  std::memcpy(img.data() + cv.total_rows, &total_rows, 4);
  for (int64_t r = 0; r <= rows; ++r) {
    const int32_t v = static_cast<int32_t>(r * S);
    std::memcpy(img.data() + cv.merge + r * 4, &v, 4);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const cudaError_t err = cudaMemcpyAsync(ws.device_persistent, img.data(), img.size(), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return launch_error("lowbit_decode_plan upload", err);
  char* base_p = static_cast<char*>(ws.device_persistent);
  WorkPlan& wp = plan->device_plan;
  wp.kv_chunk_size = reinterpret_cast<const int32_t*>(base_p + cv.chunk);
  wp.total_num_rows = reinterpret_cast<const uint32_t*>(base_p + cv.total_rows);
  wp.o_indptr = reinterpret_cast<const int32_t*>(base_p + cv.merge);
  wp.num_items = S;
  wp.padded_num_items = S;
  wp.split_kv = k.split != 0;
  return Status::kSuccess;
}

Status lowbit_decode_run(const AttentionPlan& plan, const DecodeParams& params, const LowBitCodec& codec,
                         const Workspace& ws, const LaunchContext& ctx) {
  Key k;
  if (!decode_key(plan.kernel_id, &k) || k.prefill != 0) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_decode_run: plan was not made by lowbit_decode_plan");
  }
  LowBitTables tb;
  Status s = lowbit_resolve(codec, &tb);
  if (s != Status::kSuccess) return s;
  if (tb.head_dim != k.d || tb.k.qjl != k.qjl) {
    return detail::set_error(Status::kPlanInvalidated, "lowbit_decode_run: codec differs from the planned one");
  }
  if (params.q == nullptr || params.o == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_decode_run: q and o are required");
  }
  if (params.q_len != k.q_len) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_decode_run: q_len %d, planned %d", params.q_len, k.q_len);
  }
  if (params.batch_size < 0 || params.batch_size > plan.max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "lowbit_decode_run: batch %d > planned %d", params.batch_size,
                             plan.max_batch);
  }
  if (params.num_kv_heads <= 0 || params.num_qo_heads % params.num_kv_heads != 0 || params.num_kv_heads > 4096 ||
      params.num_kv_heads != params.kv.num_kv_heads) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit_decode_run: heads %d / %d (cache %d)",
                             params.num_qo_heads, params.num_kv_heads, params.kv.num_kv_heads);
  }
  s = check_cache(params.kv, tb, true, "lowbit_decode_run");
  if (s != Status::kSuccess) return s;
  if ((reinterpret_cast<uintptr_t>(params.q) & 15u) != 0 || ((params.q_stride_n | params.q_stride_h) & 7) != 0) {
    return detail::set_error(Status::kMisaligned, "lowbit_decode_run: q rows must be 16-byte aligned");
  }
  if (params.batch_size == 0) return Status::kSuccess;
  const int D = k.d, Hq = params.num_qo_heads, G = Hq / params.num_kv_heads, R = G * k.q_len;
  const int64_t rows_cap = static_cast<int64_t>(plan.max_batch) * k.q_len;
  const DCarve cv = dcarve(rows_cap);
  char* base = static_cast<char*>(ws.device_persistent);
  if (base == nullptr || reinterpret_cast<const char*>(plan.device_plan.kv_chunk_size) != base + cv.chunk) {
    return detail::set_error(Status::kPlanInvalidated, "lowbit_decode_run: workspace differs from the planned one");
  }
  DecodeParams p = params;
  neutralize(k.mods, &p.mod);
  p.plan = plan.device_plan;
  p.tasks = nullptr;
  p.num_tasks = nullptr;
  if (k.split) {
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < plan.workspace.device_scratch_bytes) {
      return detail::set_error(Status::kWorkspaceTooSmall, "lowbit_decode_run: scratch < %zu bytes",
                               plan.workspace.device_scratch_bytes);
    }
    const int64_t prow = rows_cap * k.splits;
    p.split.o_partial = static_cast<float*>(ws.device_scratch);
    p.split.lse_partial = reinterpret_cast<float*>(static_cast<char*>(ws.device_scratch) +
                                                   align_up(static_cast<size_t>(prow) * Hq * D * 4, 256));
    p.split.merge_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
    p.split.max_splits = k.splits;
  } else {
    p.split = SplitKVBuffers{};
  }
  tb.io_fp16 = k.dt;
  const int64_t gy = static_cast<int64_t>(params.num_kv_heads) * ((R + kDecRows - 1) / kDecRows);
  if (gy > 65535) return detail::set_error(Status::kCapacityExceeded, "lowbit_decode_run: grid.y %lld", static_cast<long long>(gy));
  const dim3 grid(static_cast<unsigned>(params.batch_size), static_cast<unsigned>(gy), static_cast<unsigned>(k.splits));
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  s = detail::set_error(Status::kUnsupportedShape, "lowbit_decode_run: head dim %d not compiled", D);
#define KOPS_D(a) if (D == (a)) s = launch_decode<a>(p, tb, grid, stream);
  KOPS_LOWBIT_DIMS(KOPS_D)
#undef KOPS_D
  if (s != Status::kSuccess || !k.split) return s;
  const int64_t rows = static_cast<int64_t>(params.batch_size) * k.q_len;
  return merge_partials(k, params.o, params.o_stride_n, params.o_stride_h, params.lse, rows, Hq, p.split,
                        p.mod.sink_logits, nullptr, ctx);
}

Status lowbit_prefill_plan(const AttentionSpec& spec, const LowBitCodec& codec, const Tensor& qo_indptr_host,
                           const Tensor& kv_lens_host, const LaunchContext& ctx, const Workspace& ws,
                           AttentionPlan* plan) {
  if (plan == nullptr) return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_plan: null plan");
  Status s = check_ctx(ctx, "lowbit_prefill");
  if (s != Status::kSuccess) return s;
  LowBitTables tb;
  s = lowbit_resolve(codec, &tb);
  if (s != Status::kSuccess) return s;
  s = check_spec(spec, tb, "lowbit_prefill");
  if (s != Status::kSuccess) return s;
  const int64_t B = qo_indptr_host.ndim == 1 ? qo_indptr_host.shape[0] - 1 : -1;
  if (B < 1) return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_plan: qo_indptr must be [B + 1], B >= 1");
  std::vector<int32_t> qo, kvl;
  s = read_host_i32(qo_indptr_host, B + 1, "qo_indptr_host", &qo);
  if (s != Status::kSuccess) return s;
  s = read_host_i32(kv_lens_host, B, "kv_lens_host", &kvl);
  if (s != Status::kSuccess) return s;
  if (qo[0] != 0) return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_plan: qo_indptr[0] != 0");
  const int D = tb.head_dim, G = spec.num_q_heads / spec.num_kv_heads;
  const bool qjl = tb.k.qjl != 0;
  const int bq = pf_bq(D, qjl), bkv = pf_bkv(D, qjl);
  int64_t base_ctas = 0;
  int max_kv = 0;
  for (int64_t b = 0; b < B; ++b) {
    const int64_t qn = static_cast<int64_t>(qo[b + 1]) - qo[b];
    if (qn < 0 || kvl[b] < 0) {
      return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_plan: negative length at request %lld",
                               static_cast<long long>(b));
    }
    base_ctas += (qn * G + bq - 1) / bq;
    max_kv = std::max(max_kv, kvl[b]);
  }
  const int64_t total_q = qo[B];
  base_ctas *= spec.num_kv_heads;
  // kv chunking, as in prefill_plan: a fixed chunk under batch_invariant, else split only when the
  // q tiles fill less than one wave.
  int chunk = 0;
  bool split = false;
  if (ctx.batch_invariant) {
    chunk = kPrefillInvariantChunk;
    split = true;
  } else {
    const int64_t sms = std::max(ctx.compute_units, 1);
    if (base_ctas > 0 && base_ctas < sms && max_kv >= 8 * bkv) {
      int64_t splits = std::min<int64_t>((2 * sms + base_ctas - 1) / base_ctas, kPrefillMaxSplits);
      splits = std::min<int64_t>(splits, max_kv / (4 * bkv));
      if (splits > 1) {
        const int64_t c = (max_kv + splits - 1) / splits;
        chunk = static_cast<int>((c + bkv - 1) / bkv * bkv);
        split = true;
      }
    }
  }
  auto chunks_of = [&](int64_t kv) -> int64_t { return split ? std::max<int64_t>(1, (kv + chunk - 1) / chunk) : 1; };
  std::vector<int32_t> req, qt, kt, merge(static_cast<size_t>(total_q + 1), 0);
  int64_t prow = 0;
  for (int64_t b = 0; b < B; ++b) {
    const int64_t qn = static_cast<int64_t>(qo[b + 1]) - qo[b];
    const int64_t tiles = (qn * G + bq - 1) / bq;
    const int64_t nc = chunks_of(kvl[b]);
    for (int64_t t = 0; t < tiles; ++t) {
      for (int64_t c = 0; c < nc; ++c) {
        req.push_back(static_cast<int32_t>(b));
        qt.push_back(static_cast<int32_t>(t));
        kt.push_back(static_cast<int32_t>(c));
      }
    }
    for (int64_t i = 0; i < qn; ++i) {
      merge[static_cast<size_t>(qo[b] + i)] = static_cast<int32_t>(prow);
      prow += nc;
    }
  }
  merge[static_cast<size_t>(total_q)] = static_cast<int32_t>(prow);
  const int64_t items = static_cast<int64_t>(req.size());
  if (!split) prow = 0;
  if (items > INT32_MAX || prow > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "lowbit_prefill_plan: %lld work items", static_cast<long long>(items));
  }
  const PCarve cv = pcarve(std::max<int64_t>(items, 1), total_q);
  Key k;
  k.d = D;
  k.dt = spec.q_dtype == DType::kFloat16 ? 1 : 0;
  k.mods = spec_mods(spec);
  k.mask = static_cast<int>(spec.mask);
  k.qjl = qjl ? 1 : 0;
  k.split = split ? 1 : 0;
  k.prefill = 1;
  *plan = AttentionPlan{};
  plan->kernel_id = encode(k);
  plan->plan_generation = ctx.capture_generation;
  plan->strategy = ExecutionStrategy::kModular;
  plan->split_kv = split;
  plan->max_batch = static_cast<int32_t>(B);
  plan->max_total_q = static_cast<int32_t>(total_q);
  plan->max_kv_len = max_kv;
  plan->num_launches = 1 + (split ? 1 : 0);
  plan->workspace.device_persistent_bytes = cv.bytes;
  plan->workspace.device_scratch_bytes = split ? scratch_bytes(prow, spec.num_q_heads, D) : 0;
  plan->workspace.alignment = 256;
  if (ws.device_persistent == nullptr || ws.device_persistent_bytes < cv.bytes ||
      ws.device_scratch_bytes < plan->workspace.device_scratch_bytes ||
      (plan->workspace.device_scratch_bytes != 0 && ws.device_scratch == nullptr)) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "lowbit_prefill_plan: needs %zu persistent + %zu scratch bytes (see plan->workspace)",
                             cv.bytes, plan->workspace.device_scratch_bytes);
  }
  if ((reinterpret_cast<uintptr_t>(ws.device_persistent) & 255u) != 0 ||
      (reinterpret_cast<uintptr_t>(ws.device_scratch) & 255u) != 0) {
    return detail::set_error(Status::kMisaligned, "lowbit_prefill_plan: workspace must be 256-byte aligned");
  }
  std::vector<unsigned char> img(cv.bytes, 0);
  const int32_t chunk32 = split ? chunk : 0;
  const uint32_t rows32 = static_cast<uint32_t>(total_q);
  if (items > 0) {
    std::memcpy(img.data() + cv.req, req.data(), req.size() * 4);
    std::memcpy(img.data() + cv.qtile, qt.data(), qt.size() * 4);
    std::memcpy(img.data() + cv.kvtile, kt.data(), kt.size() * 4);
    std::memset(img.data() + cv.valid, 1, static_cast<size_t>(items));
  }
  std::memcpy(img.data() + cv.merge, merge.data(), merge.size() * 4);
  std::memcpy(img.data() + cv.chunk, &chunk32, 4);
  std::memcpy(img.data() + cv.total_rows, &rows32, 4);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const cudaError_t err = cudaMemcpyAsync(ws.device_persistent, img.data(), cv.bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return launch_error("lowbit_prefill_plan upload", err);
  char* base = static_cast<char*>(ws.device_persistent);
  WorkPlan& wp = plan->device_plan;
  wp.request_indices = reinterpret_cast<const int32_t*>(base + cv.req);
  wp.qo_tile_indices = reinterpret_cast<const int32_t*>(base + cv.qtile);
  wp.kv_tile_indices = reinterpret_cast<const int32_t*>(base + cv.kvtile);
  wp.block_valid_mask = reinterpret_cast<const uint8_t*>(base + cv.valid);
  wp.kv_chunk_size = reinterpret_cast<const int32_t*>(base + cv.chunk);
  wp.o_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
  wp.total_num_rows = reinterpret_cast<const uint32_t*>(base + cv.total_rows);
  wp.num_items = static_cast<int32_t>(items);
  wp.padded_num_items = static_cast<int32_t>(items);
  wp.split_kv = split;
  return Status::kSuccess;
}

Status lowbit_prefill_run(const AttentionPlan& plan, const PagedPrefillParams& params, const LowBitCodec& codec,
                          const Workspace& ws, const LaunchContext& ctx) {
  Key k;
  if (!decode_key(plan.kernel_id, &k) || k.prefill != 1) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_run: plan was not made by lowbit_prefill_plan");
  }
  LowBitTables tb;
  Status s = lowbit_resolve(codec, &tb);
  if (s != Status::kSuccess) return s;
  if (tb.head_dim != k.d || tb.k.qjl != k.qjl) {
    return detail::set_error(Status::kPlanInvalidated, "lowbit_prefill_run: codec differs from the planned one");
  }
  if (params.q == nullptr || params.o == nullptr || params.qo_indptr == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_run: q, o and qo_indptr are required");
  }
  if (params.num_kv_heads <= 0 || params.num_qo_heads <= 0 || params.num_qo_heads % params.num_kv_heads != 0 ||
      params.num_kv_heads > 65535 || params.num_kv_heads != params.kv.num_kv_heads) {
    return detail::set_error(Status::kUnsupportedShape, "lowbit_prefill_run: heads %d / %d (cache %d)",
                             params.num_qo_heads, params.num_kv_heads, params.kv.num_kv_heads);
  }
  if (params.batch_size != plan.max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "lowbit_prefill_run: batch %d, planned %d (host-planned: replan)",
                             params.batch_size, plan.max_batch);
  }
  s = check_cache(params.kv, tb, true, "lowbit_prefill_run");
  if (s != Status::kSuccess) return s;
  if ((reinterpret_cast<uintptr_t>(params.q) & 15u) != 0 || ((params.q_stride_n | params.q_stride_h) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(params.o) & 3u) != 0 || ((params.o_stride_n | params.o_stride_h) & 1) != 0) {
    return detail::set_error(Status::kMisaligned, "lowbit_prefill_run: q needs 16-byte rows (strides %% 8), o 4-byte pairs");
  }
  const MaskMode mask = static_cast<MaskMode>(k.mask);
  if (mask == MaskMode::kCustom && (params.custom_mask == nullptr || params.mask_indptr == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "lowbit_prefill_run: kCustom needs custom_mask and mask_indptr");
  }
  const WorkPlan& dp = plan.device_plan;
  const PCarve cv = pcarve(std::max<int64_t>(dp.padded_num_items, 1), plan.max_total_q);
  char* base = static_cast<char*>(ws.device_persistent);
  if (base == nullptr || base + cv.req != reinterpret_cast<const char*>(dp.request_indices)) {
    return detail::set_error(Status::kPlanInvalidated, "lowbit_prefill_run: workspace differs from the planned one");
  }
  if (dp.padded_num_items == 0) return Status::kSuccess;
  PagedPrefillParams pp = params;
  pp.plan = dp;
  neutralize(k.mods, &pp.mod);
  pp.mask = mask;
  const int Hq = params.num_qo_heads;
  if (plan.split_kv) {
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < plan.workspace.device_scratch_bytes) {
      return detail::set_error(Status::kWorkspaceTooSmall, "lowbit_prefill_run: scratch < %zu bytes",
                               plan.workspace.device_scratch_bytes);
    }
    // the planned partial-row count is recovered from the scratch size (regions are 256-aligned)
    const size_t per = static_cast<size_t>(Hq) * (k.d + 1) * 4;
    int64_t prow = static_cast<int64_t>(plan.workspace.device_scratch_bytes / per);
    while (prow > 0 && scratch_bytes(prow, Hq, k.d) > plan.workspace.device_scratch_bytes) --prow;
    pp.split.merge_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
    pp.split.o_partial = static_cast<float*>(ws.device_scratch);
    pp.split.lse_partial = reinterpret_cast<float*>(static_cast<char*>(ws.device_scratch) +
                                                    align_up(static_cast<size_t>(prow) * Hq * k.d * 4, 256));
    pp.split.max_splits = kPrefillMaxSplits;
  } else {
    pp.split = SplitKVBuffers{};
  }
  tb.io_fp16 = k.dt;
  const dim3 grid(static_cast<unsigned>(dp.padded_num_items), static_cast<unsigned>(params.num_kv_heads));
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  s = detail::set_error(Status::kUnsupportedShape, "lowbit_prefill_run: head dim %d not compiled", k.d);
#define KOPS_D(a) \
  if (k.d == (a)) s = k.qjl ? launch_prefill<a, true>(pp, tb, grid, stream) : launch_prefill<a, false>(pp, tb, grid, stream);
  KOPS_LOWBIT_DIMS(KOPS_D)
#undef KOPS_D
  if (s != Status::kSuccess || !plan.split_kv) return s;
  return merge_partials(k, params.o, params.o_stride_n, params.o_stride_h, params.lse, plan.max_total_q, Hq,
                        pp.split, pp.mod.sink_logits, base + cv.total_rows, ctx);
}

}  // namespace kops::experimental::lowbit
