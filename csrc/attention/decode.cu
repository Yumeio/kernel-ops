// =============================================================================
// kernel-ops | csrc/attention/decode.cu
// B8 rows 3-5: paged decode on SM80-class GPUs (fallback on SM90a/SM100a).
//   paged_decode_kernel          A03 CUDA-core decode, any page size, FP8 K/V, split-KV
//   paged_decode_tc_kernel       A03/A06 tensor-core decode: the G * q_len rows of a kv head are
//                                the M of mma.sync (GQA packing, MTP / EAGLE verification); two
//                                warps and two cp.async stages per CTA at D = 128, so two CTAs
//                                (two independent KV streams) are resident per SM
//   assign_decode_tasks_kernel   A05 device planner: KV split into uniform tasks inside the graph
//   paged_decode_dynamic_kernel  A05 persistent decode over the DecodeTask list
// Host entries (ops.h): decode_plan, decode_run.  Split partials are combined by
// merge_split_kv_kernel (B1, merge.cu).
//
// Contract: csrc/attention/{attention_params,decode_kernels}.cuh, ops.h, CONTRACTS.md §1e.
//
// Numerics.  Query row i of request b (q_len rows) sits at position kv_len - q_len + i and sees
// keys [0, that position] (causal among drafts), further limited by the window / chunk rules of
// AttentionModifiers; softcap, ALiBi and sinks follow the prefill order (prefill.cu header).  All
// sums run in fp32 in a fixed order (per thread group in token order, then groups / warps /
// splits in index order): results are bit-reproducible, and batch-invariant with
// ctx.batch_invariant (fixed kv chunk, every row through merge_split_kv).
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
#include "csrc/attention/ops.h"

namespace kops::attn {
namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }

__host__ __device__ __forceinline__ int64_t fdiv(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

__device__ __forceinline__ float tanh_fast(float x) {
  const float e = __expf(2.0f * fabsf(x));
  return copysignf(1.0f - __fdividef(2.0f, e + 1.0f), x);
}

template <typename T>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  } else {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
}

template <typename T>
__device__ __forceinline__ float2 unpack2(uint32_t w) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
  } else {
    return __half22float2(*reinterpret_cast<const __half2*>(&w));
  }
}

/// Four FP8 codes -> four floats, exact.  e5m2 is the high byte of a half; e4m3 is rebiased into a
/// half, which yields value * 2^-8 (exact): dec_kv folds the 2^8 into k_scale / v_scale.
template <bool kE5M2>
__device__ __forceinline__ void fp8x4_to_f(uint32_t w, float* out) {
  uint32_t a = __byte_perm(w, 0u, 0x1404u);  // [0, c0, 0, c1]
  uint32_t b = __byte_perm(w, 0u, 0x3424u);  // [0, c2, 0, c3]
  if constexpr (!kE5M2) {
    a = (a & 0x80008000u) | ((a >> 1) & 0x3F803F80u);
    b = (b & 0x80008000u) | ((b >> 1) & 0x3F803F80u);
  }
  const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
  const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
  out[0] = fa.x;
  out[1] = fa.y;
  out[2] = fb.x;
  out[3] = fb.y;
}

/// Eight consecutive K/V elements of one row as floats.
template <typename T, bool FP8>
__device__ __forceinline__ void load8(const char* row, bool e5m2, float (&x)[8]) {
  if constexpr (FP8) {
    const uint2 w = *reinterpret_cast<const uint2*>(row);
    if (e5m2) {
      fp8x4_to_f<true>(w.x, x);
      fp8x4_to_f<true>(w.y, x + 4);
    } else {
      fp8x4_to_f<false>(w.x, x);
      fp8x4_to_f<false>(w.y, x + 4);
    }
  } else {
    const uint4 w = *reinterpret_cast<const uint4*>(row);
    const float2 a = unpack2<T>(w.x), b = unpack2<T>(w.y), c = unpack2<T>(w.z), d = unpack2<T>(w.w);
    x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
    x[4] = c.x; x[5] = c.y; x[6] = d.x; x[7] = d.y;
  }
}

/// Per-request K/V addressing for one kv head (byte pointers / strides).
struct DecKv {
  const char* k;
  const char* v;
  int64_t k_sp, k_st, v_sp, v_st;
  const int32_t* pages;
  int ps;
  int kv_len;
  float k_scale, v_scale;
  bool e5m2;
};

template <bool FP8>
__device__ __forceinline__ DecKv dec_kv(const kv::PagedKVDesc& d, int b, int hk, float mod_ks, float mod_vs) {
  DecKv r;
  const int64_t eb = d.stride_unit == StrideUnit::kByte ? 1 : (FP8 ? 1 : 2);
  r.k = static_cast<const char*>(d.k_data) + hk * d.k_stride_head * eb;
  r.v = static_cast<const char*>(d.v_data) + hk * d.v_stride_head * eb;
  r.k_sp = d.k_stride_page * eb;
  r.k_st = d.k_stride_token * eb;
  r.v_sp = d.v_stride_page * eb;
  r.v_st = d.v_stride_token * eb;
  r.ps = d.page_size;
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    r.pages = d.block_table + static_cast<int64_t>(b) * d.block_table_stride;
    r.kv_len = d.seq_lens[b];
  } else {
    const int p0 = d.kv_indptr[b], np = d.kv_indptr[b + 1] - p0;
    r.pages = d.kv_page_indices + p0;
    r.kv_len = np > 0 ? (np - 1) * d.page_size + d.kv_last_page_len[b] : 0;
  }
  r.k_scale = r.v_scale = 1.0f;
  if constexpr (FP8) {
    const int sh = d.scale_head_stride != 0 ? hk : 0;
    r.k_scale = d.k_scale != nullptr ? d.k_scale[sh] : mod_ks;
    r.v_scale = d.v_scale != nullptr ? d.v_scale[sh] : mod_vs;
  }
  r.e5m2 = d.k_dtype == DType::kFloat8E5M2;
  if (FP8 && !r.e5m2) {  // e4m3 codes are converted as value * 2^-8 (fp8x4_to_f)
    r.k_scale *= 256.0f;
    r.v_scale *= 256.0f;
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

// ---- CUDA-core decode ------------------------------------------------------------------------------
constexpr int kDecThreads = 128;
constexpr int kDecRows = 8;  // packed query rows (q_len x G) per CTA

/// R8 = packed rows per CTA: 8, or 1 for the one-row case (MHA, q_len 1), whose registers then
/// hold one q / o row instead of eight (about 4x the resident warps).
template <int D, int R8>
struct DecGeom {
  static_assert(D % 16 == 0 && D <= 256, "head dim");
  static constexpr int kBdx = D <= 64 ? 8 : (D <= 128 ? 16 : 32);  // lanes over the head dim (8 each)
  static constexpr int kGroups = kDecThreads / kBdx;                 // token groups per CTA
  static constexpr int kSmem = kGroups * R8 * (D + 2) * 4;
};

template <int D, typename T, bool FP8, int R8>
__device__ __forceinline__ void decode_simt_body(const DecodeParams& p) {
  using Geo = DecGeom<D, R8>;
  constexpr int BDX = Geo::kBdx, NG = Geo::kGroups;
  extern __shared__ __align__(16) unsigned char dec_smem[];
  float* sm_m = reinterpret_cast<float*>(dec_smem);   // [NG][R8]
  float* sm_l = sm_m + NG * R8;                        // [NG][R8]
  float* sm_o = sm_l + NG * R8;                        // [NG][R8][D]

  const int tid = threadIdx.x, x = tid % BDX, g = tid / BDX;
  const bool lane_on = x * 8 < D;
  const int b = blockIdx.x, c = blockIdx.z, S = gridDim.z;
  const int G = p.num_qo_heads / p.num_kv_heads, q_len = p.q_len, R = G * q_len;
  const int nrb = (R + R8 - 1) / R8;
  const int hk = blockIdx.y / nrb, row0 = (blockIdx.y % nrb) * R8;
  const int rows = min(R8, R - row0);
  const AttentionModifiers& md = p.mod;

  const DecKv kv = dec_kv<FP8>(p.kv, b, hk, md.k_scale, md.v_scale);
  const int kv_len = kv.kv_len;
  int lo, hi;
  split_range(kv_len, c, S, p.plan.kv_chunk_size != nullptr ? *p.plan.kv_chunk_size : 0, &lo, &hi);
  const float sm = md.sm_scale != 0.0f ? md.sm_scale : rsqrtf(static_cast<float>(D));
  const float cap = md.logits_softcap;
  const float qs = sm * kv.k_scale * (cap > 0.0f ? 1.0f : kLog2e);

  // rows: packed r -> query token i = r / G, head hk * G + r % G
  float q[R8][8];
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
#pragma unroll
    for (int e = 0; e < 8; ++e) q[r][e] = 0.0f;
    if (lane_on) {
      const T* qp = static_cast<const T*>(p.q) + (static_cast<int64_t>(b) * q_len + i) * p.q_stride_n +
                    static_cast<int64_t>(head[r]) * p.q_stride_h + x * 8;
      const uint4 w = *reinterpret_cast<const uint4*>(qp);
      const float2 a = unpack2<T>(w.x), bb = unpack2<T>(w.y), cc = unpack2<T>(w.z), dd = unpack2<T>(w.w);
      q[r][0] = a.x * qs; q[r][1] = a.y * qs; q[r][2] = bb.x * qs; q[r][3] = bb.y * qs;
      q[r][4] = cc.x * qs; q[r][5] = cc.y * qs; q[r][6] = dd.x * qs; q[r][7] = dd.y * qs;
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
    if (tok && lane_on) {
      constexpr int EB = FP8 ? 8 : 16;
      const int pg = kv.pages[j / kv.ps], off = j % kv.ps;
      load8<T, FP8>(kv.k + static_cast<int64_t>(pg) * kv.k_sp + static_cast<int64_t>(off) * kv.k_st + x * EB,
                    kv.e5m2, kx);
      load8<T, FP8>(kv.v + static_cast<int64_t>(pg) * kv.v_sp + static_cast<int64_t>(off) * kv.v_st + x * EB,
                    kv.e5m2, vx);
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

  // combine the NG token groups (index order), then normalize / write
  if (x == 0) {
#pragma unroll
    for (int r = 0; r < R8; ++r) {
      sm_m[g * R8 + r] = m[r];
      sm_l[g * R8 + r] = l[r];
    }
  }
  if (lane_on) {
#pragma unroll
    for (int r = 0; r < R8; ++r) {
#pragma unroll
      for (int e = 0; e < 8; ++e) sm_o[(g * R8 + r) * D + x * 8 + e] = o[r][e];
    }
  }
  __syncthreads();
  const bool split = p.plan.split_kv && p.split.o_partial != nullptr;
  const int Hq = p.num_qo_heads;
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
    const int pr = row0 + r, i = pr / G, h = hk * G + (pr - i * G);
    const int64_t row = static_cast<int64_t>(b) * q_len + i;
    if (split) {
      const int64_t prow = static_cast<int64_t>(p.split.merge_indptr[row]) + c;
      p.split.o_partial[(prow * Hq + h) * D + d] = den > 0.0f ? acc / den * kv.v_scale : 0.0f;
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
      const float val = den > 0.0f ? acc * num / den * kv.v_scale : 0.0f;
      T* dst = static_cast<T*>(p.o) + row * p.o_stride_n + static_cast<int64_t>(h) * p.o_stride_h + d;
      if constexpr (std::is_same_v<T, __nv_bfloat16>) *dst = __float2bfloat16_rn(val);
      else *dst = __float2half_rn(val);
      if (d == 0 && p.lse != nullptr) p.lse[row * Hq + h] = den > 0.0f ? (mx + log2f(den)) * kLn2 : ninf();
    }
  }
}

// ---- tensor-core decode ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
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
template <typename T>
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#endif
}

/// smem rows of D 16-bit elements: XOR swizzle when D % 64 == 0, one 16-byte pad otherwise.
template <int D>
struct Lay {
  static constexpr bool kSwz = (D % 64) == 0;
  static constexpr int kRow = kSwz ? D : D + 8;
  static __device__ __forceinline__ int off(int row, int chunk) {
    if constexpr (kSwz) return row * D + ((chunk ^ (row & 7)) << 3);
    else return row * kRow + (chunk << 3);
  }
};

constexpr int kTcRows = 16;  // packed query rows per CTA (the mma M)
constexpr int kTcKW = 16;    // keys per warp step

/// CTA shape of the tensor-core decode kernel: keys per CTA step (16 per warp) and prefetch stages.
/// The mainloop is a kStages-deep cp.async ring, so a CTA needs kStageBytes = keys_in_flight x D x
/// 2 bytes x 2 (K | V) of shared memory.  At D = 128 four warps and two stages need 68 KiB and only
/// one CTA fits per SM; two warps (32 keys per step) halve that, so two CTAs -- two independent KV
/// streams -- are resident per SM and a static grid of B x Hkv x splits no longer runs in lockstep
/// waves.  D = 256 keeps one stage: its 16-bit tiles are 8 KiB each.
struct TcCfg {
  int tile_kv;
  int stages;
};
constexpr TcCfg tc_cfg(int d) {
  return d == 256 ? TcCfg{64, 1} : (d == 128 ? TcCfg{32, 2} : TcCfg{64, 2});
}

/// Per-warp smem: kStages load buffers (K | V), plus for FP8 (D <= 128) one 16-bit K | V tile the
/// raw codes are converted into right before the MMAs (the raw tiles arrive by cp.async, one tile
/// ahead).  D = 256 uses one stage (16-bit tiles of 8 KiB each) and FP8 converts on load.
template <int D, bool FP8, int WARPS, int STAGES>
struct TcGeom {
  static constexpr int kRow = Lay<D>::kRow;
  static constexpr int kWarps = WARPS;
  static constexpr int kStages = STAGES;
  static constexpr bool kRaw8 = FP8 && D <= 128;
  static constexpr int kWarpTile = kTcKW * kRow;                         // elements of one K or V tile
  static constexpr int kTile16 = kWarpTile * 2;                          // bytes of one 16-bit tile
  static constexpr int kRawTile = kTcKW * D;                             // bytes of one raw FP8 tile
  static constexpr int kStageBuf = kRaw8 ? 2 * kRawTile : 2 * kTile16;
  static constexpr int kWarpBytes = kStages * kStageBuf + (kRaw8 ? 2 * kTile16 : 0);
  static constexpr int kQBytes = kTcRows * kRow * 2;
  static constexpr int kStageBytes = kWarps * kWarpBytes;
  static constexpr int kMergeBytes = kWarps * kTcRows * (D + 2) * 4;
  static constexpr int kSmem = (kQBytes + kStageBytes) > kMergeBytes ? (kQBytes + kStageBytes) : kMergeBytes;
  static constexpr int kStep = kWarps * kTcKW;                           // keys per CTA step
  static constexpr int kThreads = kWarps * 32;
};

/// Page base (bytes) of the page that holds key `t` for K or V.
template <bool kIsK>
__device__ __forceinline__ const char* tc_page_base(const DecKv& kv, int pg) {
  const int64_t sp = kIsK ? kv.k_sp : kv.v_sp;
  const char* base = kIsK ? kv.k : kv.v;
  return base + static_cast<int64_t>(kv.pages[pg]) * sp;
}

/// Byte address of key row `r` of the 16-key tile that starts at t, for K or V.  One page-table read
/// and one division per tile serve all 16-byte chunks while the tile stays inside one page (the
/// common case: page sizes are >= 16 and tiles are 16-aligned); crossing a page boundary falls back
/// to the per-row lookup.  `single` reports which case applies.
template <bool kIsK>
__device__ __forceinline__ const char* tc_row(const DecKv& kv, int pg0, const char* page0, int off0, int t,
                                             int r, bool single) {
  const int64_t st = kIsK ? kv.k_st : kv.v_st;
  if (single) return page0 + static_cast<int64_t>(off0 + r) * st;
  const int pos = t + r, pg = pos / kv.ps;
  if (pg == pg0) return page0 + static_cast<int64_t>(pos - pg * kv.ps) * st;
  return tc_page_base<kIsK>(kv, pg) + static_cast<int64_t>(pos - pg * kv.ps) * st;
}

/// Load the 16 keys at [t, t + 16) (valid below hi) of K or V into one warp tile.
template <int D, int KW, typename T, bool FP8, bool kIsK>
__device__ __forceinline__ void tc_load(T* s, const DecKv& kv, int t, int hi, int lane) {
  constexpr int CH = D / 8, N = KW * CH;
  const int pg0 = t / kv.ps, off0 = t - pg0 * kv.ps;
  const char* page0 = tc_page_base<kIsK>(kv, pg0);
  const bool single = off0 + KW <= kv.ps;
#pragma unroll
  for (int it = 0; it < (N + 31) / 32; ++it) {
    const int idx = it * 32 + lane;
    if (N % 32 != 0 && idx >= N) break;
    const int r = idx / CH, c = idx % CH, pos = t + r;
    const bool ok = pos < hi;
    const char* row = tc_row<kIsK>(kv, pg0, page0, off0, t, r, single);
    T* dst = s + Lay<D>::off(r, c);
    if constexpr (!FP8) {
      cp_async16(smem_u32(dst), ok ? row + c * 16 : row, ok);
    } else {
      uint4 w4 = make_uint4(0u, 0u, 0u, 0u);
      if (ok) {
        const uint2 w = *reinterpret_cast<const uint2*>(row + c * 8);
        float f[8];
        if (kv.e5m2) {
          fp8x4_to_f<true>(w.x, f);
          fp8x4_to_f<true>(w.y, f + 4);
        } else {
          fp8x4_to_f<false>(w.x, f);
          fp8x4_to_f<false>(w.y, f + 4);
        }
        w4.x = pack2<T>(f[0], f[1]);
        w4.y = pack2<T>(f[2], f[3]);
        w4.z = pack2<T>(f[4], f[5]);
        w4.w = pack2<T>(f[6], f[7]);
      }
      *reinterpret_cast<uint4*>(dst) = w4;
    }
  }
}

/// Raw FP8 codes of the 16 keys at [t, t + 16) into a [16][D]-byte tile (cp.async, zero past hi).
template <int D, int KW, bool kIsK>
__device__ __forceinline__ void tc_load_raw(char* dst, const DecKv& kv, int t, int hi, int lane) {
  constexpr int CH = D / 16, N = KW * CH;
  const int pg0 = t / kv.ps, off0 = t - pg0 * kv.ps;
  const char* page0 = tc_page_base<kIsK>(kv, pg0);
  const bool single = off0 + KW <= kv.ps;
#pragma unroll
  for (int it = 0; it < (N + 31) / 32; ++it) {
    const int idx = it * 32 + lane;
    if (N % 32 != 0 && idx >= N) break;
    const int r = idx / CH, c = idx % CH, pos = t + r;
    const bool ok = pos < hi;
    const char* row = tc_row<kIsK>(kv, pg0, page0, off0, t, r, single);
    cp_async16(smem_u32(dst + r * D + c * 16), ok ? row + c * 16 : row, ok);
  }
}

/// [16][D] raw FP8 codes -> the 16-bit ldmatrix tile (exact conversion).
template <int D, int KW, typename T>
__device__ __forceinline__ void tc_convert(const char* raw, T* dst, bool e5m2, int lane) {
  constexpr int CH = D / 8, N = KW * CH;
#pragma unroll
  for (int it = 0; it < N / 32; ++it) {
    const int idx = it * 32 + lane, r = idx / CH, c = idx % CH;
    const uint2 w = *reinterpret_cast<const uint2*>(raw + r * D + c * 8);
    float f[8];
    if (e5m2) {
      fp8x4_to_f<true>(w.x, f);
      fp8x4_to_f<true>(w.y, f + 4);
    } else {
      fp8x4_to_f<false>(w.x, f);
      fp8x4_to_f<false>(w.y, f + 4);
    }
    *reinterpret_cast<uint4*>(dst + Lay<D>::off(r, c)) =
        make_uint4(pack2<T>(f[0], f[1]), pack2<T>(f[2], f[3]), pack2<T>(f[4], f[5]), pack2<T>(f[6], f[7]));
  }
}

/// Issue the loads of one warp step (K then V) into a stage buffer.
template <int D, int KW, typename T, bool FP8>
__device__ __forceinline__ void tc_issue(char* buf, const DecKv& kv, int t, int hi, int lane) {
  constexpr bool kRaw8 = FP8 && D <= 128;
  constexpr int kTile16 = KW * Lay<D>::kRow * 2;  // bytes of one 16-bit K or V tile
  constexpr int kRawTile = KW * D;                // bytes of one raw FP8 tile
  if constexpr (kRaw8) {
    tc_load_raw<D, KW, true>(buf, kv, t, hi, lane);
    tc_load_raw<D, KW, false>(buf + kRawTile, kv, t, hi, lane);
  } else {
    tc_load<D, KW, T, FP8, true>(reinterpret_cast<T*>(buf), kv, t, hi, lane);
    tc_load<D, KW, T, FP8, false>(reinterpret_cast<T*>(buf + kTile16), kv, t, hi, lane);
  }
}

/// One tensor-core decode item: request b, kv head hk, packed rows [row0, row0 + 16), keys [lo, hi).
/// split_index >= 0 writes partial rows merge_indptr[row] + split_index (no sink); < 0 writes o.
template <int D, typename T, bool FP8, int WARPS, int STAGES>
__device__ __forceinline__ void decode_tc_item(const DecodeParams& p, unsigned char* smem, int b, int hk,
                                               int row0, int lo, int hi, int split_index) {
  constexpr bool kQRegs = D <= 128;
  constexpr int KQ = D / 16, NO = D / 8;
  using L = Lay<D>;
  using Geo = TcGeom<D, FP8, WARPS, STAGES>;
  T* sQ = reinterpret_cast<T*>(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int G = p.num_qo_heads / p.num_kv_heads, q_len = p.q_len, R = G * q_len;
  const int rows = min(kTcRows, R - row0);
  const AttentionModifiers& md = p.mod;
  const DecKv kv = dec_kv<FP8>(p.kv, b, hk, md.k_scale, md.v_scale);
  const int kv_len = kv.kv_len;
  hi = min(hi, kv_len);
  lo = min(lo, hi);
  const float sm = md.sm_scale != 0.0f ? md.sm_scale : rsqrtf(static_cast<float>(D));
  const float scale_nat = sm * kv.k_scale, scale2 = scale_nat * kLog2e, cap = md.logits_softcap;

  // Q tile (rows beyond R are zero)
  {
    constexpr int CH = D / 8;
    for (int idx = tid; idx < kTcRows * CH; idx += Geo::kThreads) {
      const int r = idx / CH, c = idx % CH;
      const bool ok = r < rows;
      const int pr = row0 + (ok ? r : 0), i = pr / G, h = hk * G + (pr - i * G);
      const char* src = static_cast<const char*>(p.q) +
                        ((static_cast<int64_t>(b) * q_len + i) * p.q_stride_n + static_cast<int64_t>(h) * p.q_stride_h) * 2 +
                        c * 16;
      cp_async16(smem_u32(sQ + L::off(r, c)), ok ? src : p.q, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
  }
  // per-thread rows g and g + 8
  int qpos[2], head[2];
  RowMask rm[2];
  float slope2[2];
  int qmin = INT_MAX, qmax = INT_MIN;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int pr = row0 + min((lane >> 2) + 8 * r, rows - 1), i = pr / G;
    head[r] = hk * G + (pr - i * G);
    qpos[r] = kv_len - q_len + i;
    rm[r] = row_mask(md, qpos[r], lo, hi);
    slope2[r] = md.alibi_slopes != nullptr ? md.alibi_slopes[head[r]] * kLog2e : 0.0f;
  }
  {
    const int i0 = row0 / G, i1 = (row0 + rows - 1) / G;
    qmin = kv_len - q_len + i0;
    qmax = kv_len - q_len + i1;
  }
  int jlo = lo, jhi = min(hi, max(qmax + 1, 0));
  if (md.window_left >= 0 && md.num_sink_tokens == 0) jlo = max(jlo, qmin - md.window_left);
  if (md.chunk_size > 0) {
    jlo = static_cast<int>(max(static_cast<int64_t>(jlo), fdiv(qmin, md.chunk_size) * md.chunk_size));
    jhi = static_cast<int>(min(static_cast<int64_t>(jhi), (fdiv(qmax, md.chunk_size) + 1) * md.chunk_size));
  }
  // Window + sink tokens: keys [num_sink, qmin - window_left) are hidden from every row.
  int r0hi = jhi, r1lo = jhi;
  if (md.window_left >= 0 && md.num_sink_tokens > 0 && qmin - md.window_left > md.num_sink_tokens) {
    r0hi = min(jhi, md.num_sink_tokens);
    r1lo = max(jlo, qmin - md.window_left);
  }

  [[maybe_unused]] uint32_t qf[kQRegs ? KQ : 1][4];
  const uint32_t sQ_u = smem_u32(sQ);
  if constexpr (kQRegs) {
#pragma unroll
    for (int kk = 0; kk < KQ; ++kk) ldsm_x4(qf[kk], sQ_u + L::off(lane & 15, 2 * kk + (lane >> 4)) * 2);
  }
  float m[2] = {ninf(), ninf()}, l[2] = {0.0f, 0.0f};
  float o[NO][4];
#pragma unroll
  for (int i = 0; i < NO; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

  char* wbuf = reinterpret_cast<char*>(smem) + Geo::kQBytes + warp * Geo::kWarpBytes;  // [stage][K|V]
  char* cbuf = wbuf + Geo::kStages * Geo::kStageBuf;  // FP8: converted 16-bit K | V
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_cs = (lane >> 3) & 1;
  const int v_row = (lane & 7) + (((lane >> 3) & 1) << 3), v_cs = lane >> 4;

  // Up to two key ranges: [jlo, r0hi) and [r1lo, jhi) (one when there is no sink-token gap).
  for (int rg = 0; rg < 2; ++rg) {
    const int rlo = rg == 0 ? jlo : r1lo, rhi = rg == 0 ? r0hi : jhi;
    if (rlo >= rhi) continue;
    int t = rlo + warp * kTcKW;
    // Fill kStages - 1 tiles ahead, then keep exactly kStages - 1 groups in flight: the tile being
    // computed and the ones behind it overlap with the loads of the tiles ahead.
#pragma unroll
    for (int pf = 0; pf < Geo::kStages - 1; ++pf) {
      const int tp = t + pf * Geo::kStep;
      if (tp < rhi) tc_issue<D, kTcKW, T, FP8>(wbuf + pf * Geo::kStageBuf, kv, tp, rhi, lane);
      cp_async_commit();
    }
    for (int st = 0; t < rhi; t += Geo::kStep, st = (st + 1) % Geo::kStages) {
      const int tn = t + (Geo::kStages - 1) * Geo::kStep;
      const int sn = (st + Geo::kStages - 1) % Geo::kStages;
      if (tn < rhi) tc_issue<D, kTcKW, T, FP8>(wbuf + sn * Geo::kStageBuf, kv, tn, rhi, lane);
      cp_async_commit();
      cp_async_wait<Geo::kStages - 1>();
      __syncwarp();
      const char* sb = wbuf + st * Geo::kStageBuf;
      if constexpr (Geo::kRaw8) {
        tc_convert<D, kTcKW, T>(sb, reinterpret_cast<T*>(cbuf), kv.e5m2, lane);
        tc_convert<D, kTcKW, T>(sb + Geo::kRawTile, reinterpret_cast<T*>(cbuf + Geo::kTile16), kv.e5m2, lane);
        __syncwarp();
        sb = cbuf;
      }
      const uint32_t sK_u = smem_u32(sb);
      const uint32_t sV_u = sK_u + Geo::kTile16;
      float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
      for (int kk = 0; kk < KQ; ++kk) {
        uint32_t af[4];
        if constexpr (kQRegs) {
#pragma unroll
          for (int i = 0; i < 4; ++i) af[i] = qf[kk][i];
        } else {
          ldsm_x4(af, sQ_u + L::off(lane & 15, 2 * kk + (lane >> 4)) * 2);
        }
        uint32_t bb[4];
        ldsm_x4(bb, sK_u + L::off(b_row, 2 * kk + b_cs) * 2);
        mma16816<T>(s[0], af, bb[0], bb[1]);
        mma16816<T>(s[1], af, bb[2], bb[3]);
      }
      float mx[2] = {m[0], m[1]};
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = e >> 1, j = t + nt * 8 + 2 * (lane & 3) + (e & 1);
          float x = cap > 0.0f ? cap * tanh_fast(s[nt][e] * scale_nat / cap) * kLog2e : s[nt][e] * scale2;
          x += slope2[r] * static_cast<float>(j - qpos[r]);
          if (j >= rhi || !row_visible(rm[r], j, md.num_sink_tokens)) x = ninf();
          s[nt][e] = x;
          mx[r] = fmaxf(mx[r], x);
        }
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
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s[nt][e] = exp2f(s[nt][e] - mu[e >> 1]);
          rs[e >> 1] += s[nt][e];
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) l[r] = l[r] * alpha[r] + rs[r];
      uint32_t af[4];
      af[0] = pack2<T>(s[0][0], s[0][1]);
      af[1] = pack2<T>(s[0][2], s[0][3]);
      af[2] = pack2<T>(s[1][0], s[1][1]);
      af[3] = pack2<T>(s[1][2], s[1][3]);
#pragma unroll
      for (int dp = 0; dp < NO / 2; ++dp) {
        o[2 * dp][0] *= alpha[0];
        o[2 * dp][1] *= alpha[0];
        o[2 * dp][2] *= alpha[1];
        o[2 * dp][3] *= alpha[1];
        o[2 * dp + 1][0] *= alpha[0];
        o[2 * dp + 1][1] *= alpha[0];
        o[2 * dp + 1][2] *= alpha[1];
        o[2 * dp + 1][3] *= alpha[1];
        uint32_t bb[4];
        ldsm_x4_t(bb, sV_u + L::off(v_row, 2 * dp + v_cs) * 2);
        mma16816<T>(o[2 * dp], af, bb[0], bb[1]);
        mma16816<T>(o[2 * dp + 1], af, bb[2], bb[3]);
      }
      __syncwarp();
    }
    cp_async_wait<0>();
    __syncwarp();
  }

  // combine the 4 warps in index order (smem reused after every warp is done)
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
  }
  __syncthreads();
  float* sm_m = reinterpret_cast<float*>(smem);           // [W][16]
  float* sm_l = sm_m + Geo::kWarps * kTcRows;             // [W][16]
  float* sm_o = sm_l + Geo::kWarps * kTcRows;             // [W][16][D]
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = (lane >> 2) + 8 * r;
    if ((lane & 3) == 0) {
      sm_m[warp * kTcRows + row] = m[r];
      sm_l[warp * kTcRows + row] = l[r];
    }
#pragma unroll
    for (int i = 0; i < NO; ++i) {
      *reinterpret_cast<float2*>(sm_o + (warp * kTcRows + row) * D + i * 8 + 2 * (lane & 3)) =
          make_float2(o[i][2 * r], o[i][2 * r + 1]);
    }
  }
  __syncthreads();
  const int Hq = p.num_qo_heads;
  for (int idx = tid; idx < rows * D; idx += Geo::kThreads) {
    const int r = idx / D, d = idx - (idx / D) * D;
    float mxx = ninf();
#pragma unroll
    for (int w = 0; w < Geo::kWarps; ++w) mxx = fmaxf(mxx, sm_m[w * kTcRows + r]);
    float den = 0.0f, acc = 0.0f;
    if (mxx != ninf()) {
#pragma unroll
      for (int w = 0; w < Geo::kWarps; ++w) {
        const float wt = exp2f(sm_m[w * kTcRows + r] - mxx);
        den = __fmaf_rn(sm_l[w * kTcRows + r], wt, den);
        acc = __fmaf_rn(sm_o[(w * kTcRows + r) * D + d], wt, acc);
      }
    }
    const int pr = row0 + r, i = pr / G, h = hk * G + (pr - i * G);
    const int64_t row = static_cast<int64_t>(b) * q_len + i;
    if (split_index >= 0) {
      const int64_t prow = static_cast<int64_t>(p.split.merge_indptr[row]) + split_index;
      p.split.o_partial[(prow * Hq + h) * D + d] = den > 0.0f ? acc / den * kv.v_scale : 0.0f;
      if (d == 0) p.split.lse_partial[prow * Hq + h] = den > 0.0f ? (mxx + log2f(den)) * kLn2 : ninf();
    } else {
      float num = 1.0f;
      if (md.sink_logits != nullptr) {
        const float sk = md.sink_logits[h] * kLog2e;
        const float mn = fmaxf(mxx, sk);
        num = exp2f(mxx - mn);
        den = den * num + exp2f(sk - mn);
        mxx = mn;
      }
      const float val = den > 0.0f ? acc * num / den * kv.v_scale : 0.0f;
      T* dst = static_cast<T*>(p.o) + row * p.o_stride_n + static_cast<int64_t>(h) * p.o_stride_h + d;
      if constexpr (std::is_same_v<T, __nv_bfloat16>) *dst = __float2bfloat16_rn(val);
      else *dst = __float2half_rn(val);
      if (d == 0 && p.lse != nullptr) p.lse[row * Hq + h] = den > 0.0f ? (mxx + log2f(den)) * kLn2 : ninf();
    }
  }
  __syncthreads();  // smem is reused by the next item of a persistent CTA
}

}  // namespace

/// GQA_GROUP = 0: runtime group size, 8 packed (q_len x G) rows per CTA; GQA_GROUP = 1: one row
/// per CTA (MHA decode, q_len 1).  PAGE_SIZE = 0: runtime page size; VEC_SIZE = 8 elements per lane;
/// NUM_STAGES = 1 (direct loads, one token per group in flight); kLayout kNHD = token-major
/// strided pages (NHD and HND).
template <int HEAD_DIM, int GQA_GROUP, int PAGE_SIZE, int VEC_SIZE, int NUM_STAGES, typename DTypeQ,
          typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout>
__global__ void __launch_bounds__(kDecThreads)
    paged_decode_kernel(const __grid_constant__ DecodeParams params) {
  static_assert((GQA_GROUP == 0 || GQA_GROUP == 1) && PAGE_SIZE == 0 && VEC_SIZE == 8 && NUM_STAGES == 1,
                "runtime geometry");
  static_assert(std::is_same_v<DTypeQ, DTypeO>, "o dtype = q dtype");
  static_assert(kLayout == kv::KVLayout::kNHD, "token-major strided pages");
  decode_simt_body<HEAD_DIM, DTypeQ, Variant::fp8_kv, GQA_GROUP == 1 ? 1 : kDecRows>(params);
}

/// Grid (B, Hkv * ceil(G * q_len / 16), splits).  ROWS_PER_CTA = 16 (the mma M), CTA_TILE_KV = the
/// keys of one CTA step (16 per warp, so 2..8 warps), NUM_STAGES = prefetch ring depth.  Each warp
/// keeps its own K | V ring, so the mainloop has no block barrier.
template <int HEAD_DIM_QK, int HEAD_DIM_VO, int ROWS_PER_CTA, int CTA_TILE_KV, int NUM_STAGES,
          typename DTypeQ, typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout,
          typename ArchTag>
__global__ void __launch_bounds__(CTA_TILE_KV / kTcKW * 32)
    paged_decode_tc_kernel(const __grid_constant__ DecodeParams params) {
  constexpr int kWarps = CTA_TILE_KV / kTcKW;
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO && ROWS_PER_CTA == kTcRows && CTA_TILE_KV == kWarps * kTcKW &&
                    CTA_TILE_KV % kTcKW == 0 && kWarps >= 1 && kWarps <= 8 && NUM_STAGES >= 1 &&
                    NUM_STAGES <= 3,
                "B8 tensor-core decode geometry");
  static_assert(std::is_same_v<DTypeQ, DTypeO> && kLayout == kv::KVLayout::kNHD, "o = q dtype, strided pages");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ __align__(16) unsigned char tc_smem[];
  const DecodeParams& p = params;
  const int G = p.num_qo_heads / p.num_kv_heads;
  const int nrb = (G * p.q_len + kTcRows - 1) / kTcRows;
  const int b = blockIdx.x, hk = blockIdx.y / nrb, row0 = (blockIdx.y % nrb) * kTcRows;
  const DecKv kv = dec_kv<Variant::fp8_kv>(p.kv, b, hk, 1.0f, 1.0f);
  int lo, hi;
  split_range(kv.kv_len, blockIdx.z, gridDim.z, p.plan.kv_chunk_size != nullptr ? *p.plan.kv_chunk_size : 0,
              &lo, &hi);
  const bool split = p.plan.split_kv && p.split.o_partial != nullptr;
  decode_tc_item<HEAD_DIM_QK, DTypeQ, Variant::fp8_kv, kWarps, NUM_STAGES>(
      p, tc_smem, b, hk, row0, lo, hi, split ? static_cast<int>(blockIdx.z) : -1);
#else
  (void)params;
  if (threadIdx.x == 0) __trap();  // host refuses sm < 80 (attention_supports / decode_plan)
#endif
}

namespace {
constexpr int kTaskThreads = 512;
constexpr int kTaskMaxBatch = 2048;

/// In-place inclusive scan of a[0, n) by one CTA (segment per thread, warp + block carry).
__device__ void task_inclusive_scan(int32_t* a, int n) {
  __shared__ int32_t warp_tot[kTaskThreads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = (n + kTaskThreads - 1) / kTaskThreads;
  const int lo = min(n, tid * per), hi = min(n, lo + per);
  int32_t sum = 0;
  for (int i = lo; i < hi; ++i) sum += a[i];
  int32_t inc = sum;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int32_t y = __shfl_up_sync(0xffffffffu, inc, d);
    if (lane >= d) inc += y;
  }
  if (lane == 31) warp_tot[warp] = inc;
  __syncthreads();
  if (warp == 0) {
    int32_t w = lane < kTaskThreads / 32 ? warp_tot[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += y;
    }
    if (lane < kTaskThreads / 32) warp_tot[lane] = w;
  }
  __syncthreads();
  int32_t run = inc - sum + (warp > 0 ? warp_tot[warp - 1] : 0);
  for (int i = lo; i < hi; ++i) {
    run += a[i];
    a[i] = run;
  }
  __syncthreads();
}

__device__ __forceinline__ int last_le_i(const int32_t* a, int n, int x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (a[mid] <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}
}  // namespace

/// One CTA.  Cost unit = min_kv_tile keys; a task gets about ceil(total / cta_budget) units, so
/// a 64K-token request and 4K-token requests are cut into tasks of similar length.  Tasks are
/// ordered (request, kv head, split), and every task writes partial rows.
__global__ void __launch_bounds__(kTaskThreads) assign_decode_tasks_kernel(
    DecodeTask* __restrict__ tasks, int32_t* __restrict__ num_tasks, int32_t* __restrict__ merge_indptr,
    int32_t* __restrict__ overflow_flag, const int32_t* __restrict__ kv_lens, const int batch_size,
    const int num_kv_heads, const int q_len, const int cta_budget, const int min_kv_tile,
    const int max_splits, const int max_tasks, const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_last_page_len, const int page_size, const int max_partial_rows,
    uint32_t* __restrict__ total_rows_out) {
  __shared__ int32_t kvl[kTaskMaxBatch], chunk[kTaskMaxBatch], nsp[kTaskMaxBatch];
  __shared__ int32_t task_off[kTaskMaxBatch + 1], prow_off[kTaskMaxBatch + 1];
  __shared__ int32_t units_total;
  __shared__ int ovf;
  const int B = batch_size, tid = threadIdx.x;
  if (tid == 0) units_total = 0;
  __syncthreads();
  int32_t my_units = 0;
  for (int b = tid; b < B; b += kTaskThreads) {
    int len;
    if (kv_lens != nullptr) {
      len = kv_lens[b];
    } else {
      const int np = kv_indptr[b + 1] - kv_indptr[b];
      len = np > 0 ? (np - 1) * page_size + kv_last_page_len[b] : 0;
    }
    len = max(len, 0);
    kvl[b] = len;
    my_units += max(1, (len + min_kv_tile - 1) / min_kv_tile);
  }
  atomicAdd(&units_total, my_units);  // integer sum: order-independent
  __syncthreads();
  const int64_t total = static_cast<int64_t>(units_total) * num_kv_heads;
  const int64_t budget = max(cta_budget, 1);
  const int per_task = static_cast<int>(max(static_cast<int64_t>(1), (total + budget - 1) / budget));
  for (int b = tid; b < B; b += kTaskThreads) {
    const int units = max(1, (kvl[b] + min_kv_tile - 1) / min_kv_tile);
    const int s = min(max_splits, max(1, (units + per_task - 1) / per_task));
    // chunk in whole cost units (16-aligned), then the split count it actually produces
    const int units_per = (units + s - 1) / s;
    const int ck = max(16, units_per * min_kv_tile);
    const int ns = max(1, (kvl[b] + ck - 1) / ck);
    chunk[b] = ck;
    nsp[b] = ns;
    task_off[b + 1] = ns * num_kv_heads;
    prow_off[b + 1] = ns * q_len;
  }
  if (tid == 0) {
    task_off[0] = 0;
    prow_off[0] = 0;
  }
  __syncthreads();
  task_inclusive_scan(task_off + 1, B);
  task_inclusive_scan(prow_off + 1, B);
  const int ntask = task_off[B], nprow = prow_off[B];
  if (tid == 0) {
    ovf = (ntask > max_tasks || nprow > max_partial_rows) ? 1 : 0;
    *overflow_flag = ovf;
    *num_tasks = ovf ? 0 : ntask;
    *total_rows_out = static_cast<uint32_t>(B * q_len);
  }
  __syncthreads();
  if (ovf) return;
  for (int t = tid; t < ntask; t += kTaskThreads) {
    const int b = last_le_i(task_off, B, t);
    const int local = t - task_off[b], ns = nsp[b];
    const int hk = local / ns, s = local - hk * ns;
    DecodeTask k;
    k.request = b;
    k.kv_head = hk;
    k.kv_begin = s * chunk[b];
    k.kv_end = min(kvl[b], (s + 1) * chunk[b]);
    k.partial_row = 0;
    k.split_index = s;
    k.num_splits = ns;
    k.flags = (q_len > 1 ? 1 : 0) | (s == ns - 1 ? 2 : 0);
    tasks[t] = k;
  }
  for (int r = tid; r <= B * q_len; r += kTaskThreads) {
    if (r == B * q_len) {
      merge_indptr[r] = nprow;
      continue;
    }
    const int b = r / q_len, i = r - b * q_len;
    merge_indptr[r] = prow_off[b] + i * nsp[b];
  }
}

/// Persistent CTAs pull tasks from task_counter[0]; each task covers every packed row block of
/// its (request, kv head).  GQA_GROUP = 0: runtime group size.
template <int HEAD_DIM_QK, int HEAD_DIM_VO, int GQA_GROUP, int CTA_TILE_KV, int NUM_STAGES,
          typename DTypeQ, typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout,
          typename ArchTag>
__global__ void __launch_bounds__(CTA_TILE_KV / kTcKW * 32)
    paged_decode_dynamic_kernel(const __grid_constant__ DecodeParams params,
                                uint32_t* __restrict__ task_counter, const uint32_t epoch) {
  constexpr int kWarps = CTA_TILE_KV / kTcKW;
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO && GQA_GROUP == 0 && CTA_TILE_KV == kWarps * kTcKW &&
                    CTA_TILE_KV % kTcKW == 0 && kWarps >= 1 && kWarps <= 8 && NUM_STAGES >= 1 &&
                    NUM_STAGES <= 3,
                "B8 dynamic decode geometry");
  static_assert(std::is_same_v<DTypeQ, DTypeO> && kLayout == kv::KVLayout::kNHD, "o = q dtype, strided pages");
  (void)epoch;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ __align__(16) unsigned char dyn_smem[];
  __shared__ int cur;
  const DecodeParams& p = params;
  const int n = *p.num_tasks;
  const int G = p.num_qo_heads / p.num_kv_heads;
  const int nrb = (G * p.q_len + kTcRows - 1) / kTcRows;
  for (;;) {
    if (threadIdx.x == 0) cur = static_cast<int>(atomicAdd(&task_counter[0], 1u));
    __syncthreads();
    const int t = cur;
    __syncthreads();
    if (t >= n) break;
    const DecodeTask k = p.tasks[t];
    for (int rb = 0; rb < nrb; ++rb) {
      decode_tc_item<HEAD_DIM_QK, DTypeQ, Variant::fp8_kv, kWarps, NUM_STAGES>(
          p, dyn_smem, k.request, k.kv_head, rb * kTcRows, k.kv_begin, k.kv_end, k.split_index);
    }
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&task_counter[1], 1u) == gridDim.x - 1) {
      task_counter[0] = 0u;  // every CTA has made its last grab: reset for the next launch / replay
      task_counter[1] = 0u;
    }
  }
#else
  (void)params;
  (void)task_counter;
  if (threadIdx.x == 0) __trap();  // host refuses sm < 80 (decode_plan)
#endif
}

// =================================================================================================
// Host
// =================================================================================================
namespace internal {
Status merge_split_kv_run(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                          const Tensor& sink_logits, const Tensor& total_rows,
                          const LaunchContext& ctx);
}  // namespace internal

namespace {

constexpr uint64_t kDecodeKind = 0xA3;
constexpr int kDecodeInvariantChunk = 2048;
constexpr int kDecodeMaxSplits = 32;
constexpr int kMinKvTile = 256;  // dynamic planner cost unit (keys)

enum DecodePath : int { kPathSimt = 0, kPathTc = 1, kPathDynamic = 2 };

#define KOPS_DECODE_SIMT_DIMS(X) X(64) X(80) X(96) X(128) X(192) X(256)
#define KOPS_DECODE_TC_DIMS(X) X(64) X(128) X(256)

bool simt_dim(int d) {
#define KOPS_D(a) if (d == (a)) return true;
  KOPS_DECODE_SIMT_DIMS(KOPS_D)
#undef KOPS_D
  return false;
}
bool tc_dim(int d) {
#define KOPS_D(a) if (d == (a)) return true;
  KOPS_DECODE_TC_DIMS(KOPS_D)
#undef KOPS_D
  return false;
}

struct DKey {
  int d = 0, dt = 0, fp8 = 0, mods = 0, path = 0, splits = 1, q_len = 1, split = 0;
};
uint64_t dencode(const DKey& k) {
  return (kDecodeKind << 56) | (static_cast<uint64_t>(k.split) << 55) | (static_cast<uint64_t>(k.path) << 52) |
         (static_cast<uint64_t>(k.mods) << 44) | (static_cast<uint64_t>(k.fp8) << 43) |
         (static_cast<uint64_t>(k.dt) << 42) | (static_cast<uint64_t>(k.q_len) << 32) |
         (static_cast<uint64_t>(k.splits) << 16) | static_cast<uint64_t>(k.d);
}
bool ddecode(uint64_t id, DKey* k) {
  if ((id >> 56) != kDecodeKind) return false;
  k->split = static_cast<int>((id >> 55) & 1);
  k->path = static_cast<int>((id >> 52) & 7);
  k->mods = static_cast<int>((id >> 44) & 31);
  k->fp8 = static_cast<int>((id >> 43) & 1);
  k->dt = static_cast<int>((id >> 42) & 1);
  k->q_len = static_cast<int>((id >> 32) & 0x3FF);
  k->splits = static_cast<int>((id >> 16) & 0xFFFF);
  k->d = static_cast<int>(id & 0xFFFF);
  return true;
}

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

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

/// device_persistent: chunk | num tasks | overflow | total rows | counter[2] | merge_indptr | tasks.
struct DCarve {
  size_t chunk, ntasks, overflow, total_rows, counter, merge, tasks, bytes;
};
DCarve dcarve(int64_t rows, int64_t max_tasks) {
  DCarve c{};
  size_t o = 0;
  auto take = [&](size_t n) {
    const size_t at = o;
    o = align_up(o + n, 256);
    return at;
  };
  c.chunk = take(4);
  c.ntasks = take(4);
  c.overflow = take(4);
  c.total_rows = take(4);
  c.counter = take(8);
  c.merge = take(static_cast<size_t>(rows + 1) * 4);
  c.tasks = take(static_cast<size_t>(std::max<int64_t>(max_tasks, 0)) * sizeof(DecodeTask));
  c.bytes = o;
  return c;
}

/// Static (non-dynamic) decode: the kernel for (D, dtype, fp8, path).
template <int D, typename T, bool FP8, int GROUP>
Status launch_simt(const DecodeParams& p, dim3 grid, cudaStream_t st) {
  using V = AttentionVariant<true, true, true, true, true, FP8>;
  using KVT = std::conditional_t<FP8, __nv_fp8_e4m3, T>;
  auto* fn = paged_decode_kernel<D, GROUP, 0, 8, 1, T, KVT, T, V, kv::KVLayout::kNHD>;
  constexpr int smem = DecGeom<D, GROUP == 1 ? 1 : kDecRows>::kSmem;
  static const cudaError_t attr = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("paged_decode smem attribute", attr));
  fn<<<grid, kDecThreads, smem, st>>>(p);
  return Status::kSuccess;
}

/// Shared memory of the tensor-core decode geometry selected for (D, FP8) and the number of CTAs of
/// that shape that fit on one SM (shared memory limited, at most four).
template <int D, bool FP8>
constexpr int tc_smem() {
  constexpr TcCfg c = tc_cfg(D);
  return TcGeom<D, FP8, c.tile_kv / kTcKW, c.stages>::kSmem;
}

int tc_smem_for(int d, bool fp8) {
  return fp8 ? (d == 64 ? tc_smem<64, true>() : (d == 128 ? tc_smem<128, true>() : tc_smem<256, true>()))
             : (d == 64 ? tc_smem<64, false>() : (d == 128 ? tc_smem<128, false>() : tc_smem<256, false>()));
}

int tc_ctas_per_sm(int smem, int device) {
  int per_sm_bytes = 0;
  if (cudaDeviceGetAttribute(&per_sm_bytes, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) != cudaSuccess ||
      per_sm_bytes <= 0) {
    per_sm_bytes = 100 * 1024;
  }
  return std::max(1, std::min(4, (per_sm_bytes - 1024) / (smem + 1024)));
}

template <int D, typename T, bool FP8, int WARPS, int STAGES>
Status launch_tc(const DecodeParams& p, dim3 grid, int ctas, uint32_t* counter, cudaStream_t st) {
  using V = AttentionVariant<true, true, true, true, true, FP8>;
  using KVT = std::conditional_t<FP8, __nv_fp8_e4m3, T>;
  constexpr int smem = TcGeom<D, FP8, WARPS, STAGES>::kSmem;
  if (counter == nullptr) {
    auto* fn = paged_decode_tc_kernel<D, D, kTcRows, WARPS * kTcKW, STAGES, T, KVT, T, V, kv::KVLayout::kNHD, Sm80>;
    static const cudaError_t attr = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("paged_decode_tc smem attribute", attr));
    fn<<<grid, WARPS * 32, smem, st>>>(p);
  } else {
    auto* fn =
        paged_decode_dynamic_kernel<D, D, 0, WARPS * kTcKW, STAGES, T, KVT, T, V, kv::KVLayout::kNHD, Sm80>;
    static const cudaError_t attr = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("paged_decode_dynamic smem attribute", attr));
    fn<<<ctas, WARPS * 32, smem, st>>>(p, counter, 0u);
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("decode launch", err);
}

/// Launch the tensor-core decode kernel with the CTA shape tc_cfg(D) picks; counter != 0 selects the
/// persistent (device planner) kernel.
template <int D, typename T, bool FP8>
Status launch_tc_pick(const DecodeParams& p, dim3 grid, int ctas, uint32_t* counter, cudaStream_t st) {
  if constexpr (D == 64 || D == 128 || D == 256) {
    constexpr TcCfg c = tc_cfg(D);
    return launch_tc<D, T, FP8, c.tile_kv / kTcKW, c.stages>(p, grid, ctas, counter, st);
  } else {
    return detail::set_error(Status::kUnsupportedShape, "paged_decode_tc: head dim %d not compiled", D);
  }
}

template <int D, typename T, bool FP8>
Status launch_static(int path, int rows, const DecodeParams& p, dim3 grid, cudaStream_t st) {
  if (path == kPathSimt) {
    const Status s = rows == 1 ? launch_simt<D, T, FP8, 1>(p, grid, st) : launch_simt<D, T, FP8, 0>(p, grid, st);
    if (s != Status::kSuccess) return s;
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? Status::kSuccess : launch_error("paged_decode launch", err);
  }
  return launch_tc_pick<D, T, FP8>(p, grid, 0, nullptr, st);
}

template <int D, typename T, bool FP8>
Status launch_dynamic(const DecodeParams& p, int ctas, uint32_t* counter, cudaStream_t st) {
  return launch_tc_pick<D, T, FP8>(p, dim3(1), ctas, counter, st);
}

template <typename T, bool FP8>
Status dispatch_d(const DKey& k, const DecodeParams& p, dim3 grid, int ctas, uint32_t* counter,
                  cudaStream_t st) {
  const int rows = p.num_qo_heads / p.num_kv_heads * p.q_len;
#define KOPS_D(a)                                                                       \
  if (k.d == (a)) {                                                                     \
    return k.path == kPathDynamic ? launch_dynamic<a, T, FP8>(p, ctas, counter, st)    \
                                  : launch_static<a, T, FP8>(k.path, rows, p, grid, st); \
  }
  KOPS_DECODE_SIMT_DIMS(KOPS_D)
#undef KOPS_D
  return detail::set_error(Status::kUnsupportedShape, "decode: head dim %d not compiled", k.d);
}

/// Persistent CTAs for the dynamic kernel: SMs x CTAs that fit the SM's shared memory (at most 4).
int dynamic_ctas(const DKey& k, const LaunchContext& ctx) {
  const int per_sm = tc_ctas_per_sm(tc_smem_for(k.d, k.fp8 != 0), ctx.device);
  return std::max(1, ctx.compute_units) * per_sm;
}

}  // namespace

Status decode_plan(const AttentionSpec& spec, int32_t max_batch, int32_t max_kv_len, int32_t q_len,
                   bool use_device_planner, const LaunchContext& ctx, const Workspace& ws,
                   AttentionPlan* plan) {
  if (plan == nullptr) return detail::set_error(Status::kInvalidArgument, "decode_plan: null plan");
  if (ctx_is_amd(ctx)) return detail::set_error(Status::kUnsupportedBackend, "decode: CUDA only");
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "decode: needs sm_80+");
  }
  if (spec.num_q_heads <= 0 || spec.num_kv_heads <= 0 || spec.num_q_heads % spec.num_kv_heads != 0) {
    return detail::set_error(Status::kUnsupportedShape, "decode: num_q_heads must be a multiple of num_kv_heads");
  }
  if (spec.head_dim_qk != spec.head_dim_vo || !simt_dim(spec.head_dim_qk)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "decode: head dims must be equal and in {64,80,96,128,192,256}");
  }
  const bool q16 = spec.q_dtype == DType::kBFloat16 || spec.q_dtype == DType::kFloat16;
  if (!q16 || spec.o_dtype != spec.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "decode: q/o must be the same bf16 or fp16 dtype");
  }
  const bool kv_native = spec.kv_dtype == spec.q_dtype && spec.k_codec == kv::KVCodec::kNative &&
                         spec.v_codec == kv::KVCodec::kNative;
  const bool kv_fp8 = (spec.kv_dtype == DType::kFloat8E4M3 || spec.kv_dtype == DType::kFloat8E5M2) &&
                      (spec.k_codec == kv::KVCodec::kFP8PerTensor || spec.k_codec == kv::KVCodec::kFP8PerHead) &&
                      spec.v_codec == spec.k_codec;
  if (!kv_native && !kv_fp8) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "decode: K/V must match q, or be FP8 e4m3/e5m2 with a per-tensor / per-head scale");
  }
  if (spec.page_size <= 0 || (spec.kv_layout != kv::KVLayout::kNHD && spec.kv_layout != kv::KVLayout::kHND)) {
    return detail::set_error(Status::kUnsupportedLayout, "decode: paged NHD or HND K/V required");
  }
  if (spec.mask == MaskMode::kCustom || spec.fp8_tensor_core) {
    return detail::set_error(Status::kUnsupportedShape,
                             "decode: custom masks / FP8 compute go through paged_prefill");
  }
  if (q_len < 1 || q_len > 8 || max_batch < 1 || max_kv_len < 0) {
    return detail::set_error(Status::kInvalidArgument, "decode: q_len 1..8, max_batch >= 1, max_kv_len >= 0");
  }
  const int G = spec.num_q_heads / spec.num_kv_heads, R = G * q_len, D = spec.head_dim_qk;
  DKey k;
  k.d = D;
  k.dt = spec.q_dtype == DType::kFloat16 ? 1 : 0;
  k.fp8 = kv_fp8 ? 1 : 0;
  k.mods = (spec.sliding_window ? 1 : 0) | (spec.softcap ? 2 : 0) | (spec.alibi ? 4 : 0) |
           (spec.sinks ? 8 : 0) | (spec.chunked ? 16 : 0);
  k.q_len = q_len;
  if (use_device_planner) {
    if (!tc_dim(D)) {
      return detail::set_error(Status::kUnsupportedShape, "decode: device planner needs head dim 64, 128 or 256");
    }
    if (max_batch > kTaskMaxBatch) {
      return detail::set_error(Status::kCapacityExceeded, "decode: device planner holds %d requests", kTaskMaxBatch);
    }
    k.path = kPathDynamic;
  } else {
    // Tensor cores when the packed rows of a kv head fill at least a quarter of the mma M.
    k.path = (R >= 4 && tc_dim(D)) ? kPathTc : kPathSimt;
  }
  const int64_t rows = static_cast<int64_t>(max_batch) * q_len;
  int chunk = 0, S = 1;
  int64_t max_tasks = 0, prow = 0;
  if (k.path == kPathDynamic) {
    k.split = 1;
    const int64_t units = static_cast<int64_t>(max_batch) * ((max_kv_len + kMinKvTile - 1) / kMinKvTile + 1);
    max_tasks = std::min<int64_t>(units, static_cast<int64_t>(max_batch) * kDecodeMaxSplits) * spec.num_kv_heads;
    prow = rows * kDecodeMaxSplits;
  } else {
    const int rb = k.path == kPathTc ? kTcRows : kDecRows;
    const int64_t base = static_cast<int64_t>(max_batch) * spec.num_kv_heads * ((R + rb - 1) / rb);
    if (ctx.batch_invariant) {
      chunk = kDecodeInvariantChunk;
      S = std::max(1, (max_kv_len + chunk - 1) / chunk);
      k.split = 1;
    } else {
      const int64_t target = static_cast<int64_t>(std::max(ctx.compute_units, 1)) * (k.path == kPathTc ? 2 : 4);
      if (base < target) {
        S = static_cast<int>(std::min<int64_t>({(target + base - 1) / base, kDecodeMaxSplits,
                                                std::max<int64_t>(1, max_kv_len / 256)}));
      }
      // A single long request can dominate a batch that otherwise has enough CTAs: the static grid
      // is (batch, kv-head, split), so with split=1 the 32k-token request runs on 8 CTAs while the
      // rest of the batch is already idle. Split long-context batches into kDecodeInvariantChunk-key
      // chunks whenever the grid is still short of four waves; short requests simply get a few
      // tiny splits and exit quickly.
      if (max_kv_len >= 4 * kDecodeInvariantChunk && base < 4 * target) {
        const int64_t desired = std::min<int64_t>(
            {static_cast<int64_t>(kDecodeMaxSplits),
             (static_cast<int64_t>(max_kv_len) + kDecodeInvariantChunk - 1) / kDecodeInvariantChunk,
             std::max<int64_t>(1, max_kv_len / 256)});
        if (desired > S) S = static_cast<int>(desired);
      }
      k.split = S > 1 ? 1 : 0;
    }
    if (S > 65535) return detail::set_error(Status::kCapacityExceeded, "decode: %d splits", S);
    prow = k.split ? rows * S : 0;
  }
  k.splits = S;
  const DCarve cv = dcarve(rows, max_tasks);
  *plan = AttentionPlan{};
  plan->kernel_id = dencode(k);
  plan->plan_generation = ctx.capture_generation;
  plan->strategy = use_device_planner ? ExecutionStrategy::kPersistent : ExecutionStrategy::kModular;
  plan->split_kv = k.split != 0;
  plan->max_batch = max_batch;
  plan->max_total_q = static_cast<int32_t>(rows);
  plan->max_kv_len = max_kv_len;
  plan->num_launches = (k.path == kPathDynamic ? 1 : 0) + 1 + (k.split ? 1 : 0);
  plan->workspace.device_persistent_bytes = cv.bytes;
  plan->workspace.device_scratch_bytes =
      k.split ? align_up(static_cast<size_t>(prow) * spec.num_q_heads * D * 4, 256) +
                    align_up(static_cast<size_t>(prow) * spec.num_q_heads * 4, 256)
              : 0;
  plan->workspace.alignment = 256;
  if (ws.device_persistent == nullptr || ws.device_persistent_bytes < cv.bytes ||
      ws.device_scratch_bytes < plan->workspace.device_scratch_bytes ||
      (plan->workspace.device_scratch_bytes != 0 && ws.device_scratch == nullptr)) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "decode_plan: needs %zu persistent + %zu scratch bytes (see plan->workspace)",
                             cv.bytes, plan->workspace.device_scratch_bytes);
  }
  if ((reinterpret_cast<uintptr_t>(ws.device_persistent) & 255u) != 0 ||
      (reinterpret_cast<uintptr_t>(ws.device_scratch) & 255u) != 0) {
    return detail::set_error(Status::kMisaligned, "decode_plan: workspace must be 256-byte aligned");
  }
  std::vector<unsigned char> img(cv.merge + static_cast<size_t>(rows + 1) * 4, 0);
  std::memcpy(img.data() + cv.chunk, &chunk, 4);
  const uint32_t total_rows = static_cast<uint32_t>(rows);
  std::memcpy(img.data() + cv.total_rows, &total_rows, 4);
  if (k.path != kPathDynamic) {
    for (int64_t r = 0; r <= rows; ++r) {
      const int32_t v = static_cast<int32_t>(r * S);
      std::memcpy(img.data() + cv.merge + r * 4, &v, 4);
    }
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const cudaError_t err = cudaMemcpyAsync(ws.device_persistent, img.data(), img.size(), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return launch_error("decode_plan upload", err);
  char* base = static_cast<char*>(ws.device_persistent);
  WorkPlan& wp = plan->device_plan;
  wp.kv_chunk_size = reinterpret_cast<const int32_t*>(base + cv.chunk);
  wp.total_num_rows = reinterpret_cast<const uint32_t*>(base + cv.total_rows);
  wp.o_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
  wp.num_items = static_cast<int32_t>(k.path == kPathDynamic ? max_tasks : S);
  wp.padded_num_items = wp.num_items;
  wp.split_kv = k.split != 0;
  return Status::kSuccess;
}

Status decode_run(const AttentionPlan& plan, const DecodeParams& params, const Workspace& ws,
                  const LaunchContext& ctx) {
  DKey k;
  if (!ddecode(plan.kernel_id, &k)) {
    return detail::set_error(Status::kInvalidArgument, "decode_run: plan was not made by decode_plan");
  }
  if (params.q == nullptr || params.o == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "decode_run: q and o are required");
  }
  if (params.q_len != k.q_len) {
    return detail::set_error(Status::kInvalidArgument, "decode_run: q_len %d, planned %d", params.q_len, k.q_len);
  }
  if (params.batch_size < 0 || params.batch_size > plan.max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "decode_run: batch %d > planned %d", params.batch_size,
                             plan.max_batch);
  }
  if (params.num_kv_heads <= 0 || params.num_qo_heads % params.num_kv_heads != 0 ||
      params.num_kv_heads > 4096) {
    return detail::set_error(Status::kUnsupportedShape, "decode_run: heads %d / %d", params.num_qo_heads,
                             params.num_kv_heads);
  }
  const kv::PagedKVDesc& d = params.kv;
  if (d.k_data == nullptr || d.v_data == nullptr || d.page_size <= 0) {
    return detail::set_error(Status::kInvalidArgument, "decode_run: k_data, v_data, page_size required");
  }
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    if (d.block_table == nullptr || d.seq_lens == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "decode_run: block_table and seq_lens required");
    }
  } else if (d.table_format != kv::PageTableFormat::kCSR || d.kv_indptr == nullptr ||
             d.kv_page_indices == nullptr || d.kv_last_page_len == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "decode_run: CSR page table or block table required");
  }
  const int64_t smask = (d.stride_unit == StrideUnit::kByte || k.fp8) ? 15 : 7;
  if ((reinterpret_cast<uintptr_t>(params.q) & 15u) != 0 || ((params.q_stride_n | params.q_stride_h) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(d.k_data) & 15u) != 0 || (reinterpret_cast<uintptr_t>(d.v_data) & 15u) != 0 ||
      ((d.k_stride_page | d.k_stride_token | d.k_stride_head | d.v_stride_page | d.v_stride_token |
        d.v_stride_head) & smask) != 0) {
    return detail::set_error(Status::kMisaligned, "decode_run: q / K / V rows must be 16-byte aligned");
  }
  if (params.batch_size == 0) return Status::kSuccess;
  const int D = k.d, Hq = params.num_qo_heads, G = Hq / params.num_kv_heads, R = G * k.q_len;
  const DCarve cv = dcarve(static_cast<int64_t>(plan.max_batch) * k.q_len,
                           k.path == kPathDynamic ? plan.device_plan.padded_num_items : 0);
  char* base = static_cast<char*>(ws.device_persistent);
  if (base == nullptr || reinterpret_cast<const char*>(plan.device_plan.kv_chunk_size) != base + cv.chunk) {
    return detail::set_error(Status::kPlanInvalidated, "decode_run: workspace differs from the planned one");
  }
  DecodeParams p = params;
  neutralize(k.mods, &p.mod);
  p.plan = plan.device_plan;
  p.tasks = nullptr;
  p.num_tasks = nullptr;
  const int64_t rows_cap = static_cast<int64_t>(plan.max_batch) * k.q_len;
  if (k.split) {
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < plan.workspace.device_scratch_bytes) {
      return detail::set_error(Status::kWorkspaceTooSmall, "decode_run: scratch < %zu bytes",
                               plan.workspace.device_scratch_bytes);
    }
    const int64_t prow = rows_cap * (k.path == kPathDynamic ? kDecodeMaxSplits : k.splits);
    p.split.o_partial = static_cast<float*>(ws.device_scratch);
    p.split.lse_partial = reinterpret_cast<float*>(static_cast<char*>(ws.device_scratch) +
                                                   align_up(static_cast<size_t>(prow) * Hq * D * 4, 256));
    p.split.merge_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
    p.split.max_splits = k.path == kPathDynamic ? kDecodeMaxSplits : k.splits;
  } else {
    p.split = SplitKVBuffers{};
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  Status s;
  if (k.path == kPathDynamic) {
    p.tasks = reinterpret_cast<const DecodeTask*>(base + cv.tasks);
    p.num_tasks = reinterpret_cast<const int32_t*>(base + cv.ntasks);
    const int ctas = dynamic_ctas(k, ctx);
    const bool table = d.table_format == kv::PageTableFormat::kBlockTable;
    assign_decode_tasks_kernel<<<1, kTaskThreads, 0, stream>>>(
        reinterpret_cast<DecodeTask*>(base + cv.tasks), reinterpret_cast<int32_t*>(base + cv.ntasks),
        reinterpret_cast<int32_t*>(base + cv.merge), reinterpret_cast<int32_t*>(base + cv.overflow),
        table ? d.seq_lens : nullptr, params.batch_size, params.num_kv_heads, k.q_len, ctas, kMinKvTile,
        kDecodeMaxSplits, plan.device_plan.padded_num_items, table ? nullptr : d.kv_indptr,
        table ? nullptr : d.kv_last_page_len, d.page_size, static_cast<int>(rows_cap * kDecodeMaxSplits),
        reinterpret_cast<uint32_t*>(base + cv.total_rows));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("assign_decode_tasks launch", err);
    uint32_t* counter = reinterpret_cast<uint32_t*>(base + cv.counter);
    s = k.dt == 0 ? (k.fp8 ? dispatch_d<__nv_bfloat16, true>(k, p, dim3(1), ctas, counter, stream)
                           : dispatch_d<__nv_bfloat16, false>(k, p, dim3(1), ctas, counter, stream))
                  : (k.fp8 ? dispatch_d<__half, true>(k, p, dim3(1), ctas, counter, stream)
                           : dispatch_d<__half, false>(k, p, dim3(1), ctas, counter, stream));
  } else {
    const int rb = k.path == kPathTc ? kTcRows : kDecRows;
    const int64_t gy = static_cast<int64_t>(params.num_kv_heads) * ((R + rb - 1) / rb);
    if (gy > 65535) return detail::set_error(Status::kCapacityExceeded, "decode_run: grid.y %lld", static_cast<long long>(gy));
    const dim3 grid(static_cast<unsigned>(params.batch_size), static_cast<unsigned>(gy),
                    static_cast<unsigned>(k.splits));
    s = k.dt == 0 ? (k.fp8 ? dispatch_d<__nv_bfloat16, true>(k, p, grid, 0, nullptr, stream)
                           : dispatch_d<__nv_bfloat16, false>(k, p, grid, 0, nullptr, stream))
                  : (k.fp8 ? dispatch_d<__half, true>(k, p, grid, 0, nullptr, stream)
                           : dispatch_d<__half, false>(k, p, grid, 0, nullptr, stream));
  }
  if (s != Status::kSuccess || !k.split) return s;

  const int64_t rows = static_cast<int64_t>(params.batch_size) * k.q_len;
  Tensor o{};
  o.data = params.o;
  o.dtype = k.dt == 0 ? DType::kBFloat16 : DType::kFloat16;
  o.ndim = 3;
  o.device = ctx.device;
  o.shape[0] = rows;
  o.shape[1] = Hq;
  o.shape[2] = D;
  o.stride[0] = params.o_stride_n;
  o.stride[1] = params.o_stride_h;
  o.stride[2] = 1;
  Tensor lse{};
  if (params.lse != nullptr) {
    lse.data = params.lse;
    lse.dtype = DType::kFloat32;
    lse.ndim = 2;
    lse.device = ctx.device;
    lse.shape[0] = rows;
    lse.shape[1] = Hq;
    lse.stride[0] = Hq;
    lse.stride[1] = 1;
  }
  Tensor sinks{};
  if (p.mod.sink_logits != nullptr) {
    sinks.data = const_cast<float*>(p.mod.sink_logits);
    sinks.dtype = DType::kFloat32;
    sinks.ndim = 1;
    sinks.device = ctx.device;
    sinks.shape[0] = Hq;
    sinks.stride[0] = 1;
  }
  return internal::merge_split_kv_run(o, lse, p.split, sinks, Tensor{}, ctx);
}

}  // namespace kops::attn
