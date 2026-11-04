// =============================================================================
// kernel-ops | csrc/attention/prefill.cu
// B8 rows 1-2: FA2-style prefill on mma.sync (SM80/86/89/120a/121a; fallback on SM90a/SM100a).
//   ragged_prefill_kernel   A01 dense / ragged K,V (no cache read)
//   paged_prefill_kernel    A02 chunked prefill over paged KV, A06 verification, A08 FP8 KV
//   prefill_plan_kernel     device planner: (request, q tile, kv chunk) items inside a graph
// Host entries (ops.h): attention_supports, prefill_plan, ragged_prefill_run, paged_prefill_run.
//
// Contract: csrc/attention/{attention_params,prefill_kernels}.cuh, ops.h, CONTRACTS.md §1e.
//
// Geometry.  One CTA = 4 warps = 64 packed query rows of ONE kv head: packed row r maps to query
// token r / G and q head hk * G + r % G (GQA packing, G = Hq / Hkv), so K/V tiles are read once for
// the G heads that share them.  Each warp owns 16 rows.  Q stays in registers when Dqk <= 128 (else
// it is re-read from smem with ldmatrix); K/V tiles of 64 (32 when a head dim exceeds 128) tokens go
// through cp.async into padded or XOR-swizzled smem.  QK^T and PV are mma.sync.m16n8k16 with fp32
// accumulators; the S accumulator is re-packed in registers as the A operand of PV (FA2).
//
// Numerics.  Scores are formed in fp32, s = sm_scale * k_scale * q.k, then softcap
// (cap * tanh(s / cap)), then ALiBi (slope * (k_pos - q_pos)), then the mask; the online softmax
// runs in base 2 (s * log2 e) with fp32 running max / sum.  P is rounded to the compute dtype for
// the PV product (FA2 convention); the row sum uses the fp32 P.  A sink adds exp(sink - m) to the
// denominator once, in the final normalization of an unsplit row, or in merge_split_kv for split
// rows.  The order of every sum depends only on the tile schedule of the request, never on other
// requests or on timing: results are bit-reproducible, and batch-invariant when the kv chunk size
// does not depend on the batch (ctx.batch_invariant, see prefill_plan).
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

// ---- PTX -----------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
/// 16-byte cp.async; pred == false zero-fills without reading `src`.
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
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;" ::: "memory");
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
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
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

/// Four FP8 codes (one u32) -> two u32 of packed 16-bit T, exact (every e4m3 / e5m2 value is a
/// half and a bf16).  e5m2 is the high byte of a half; e4m3 shifts its exponent / mantissa into the
/// half fields (exponent bias 7 -> 15), which yields value * 2^-8 (exact, subnormals included):
/// the 2^8 is folded into k_scale / v_scale by the caller.
template <typename T, bool kE5M2>
__device__ __forceinline__ void fp8x4_to(uint32_t w, uint32_t& lo, uint32_t& hi) {
  uint32_t a = __byte_perm(w, 0u, 0x1404u);  // [0, c0, 0, c1]
  uint32_t b = __byte_perm(w, 0u, 0x3424u);  // [0, c2, 0, c3]
  if constexpr (!kE5M2) {
    a = (a & 0x80008000u) | ((a >> 1) & 0x3F803F80u);
    b = (b & 0x80008000u) | ((b >> 1) & 0x3F803F80u);
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
    const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
    lo = pack2<T>(fa.x, fa.y);
    hi = pack2<T>(fb.x, fb.y);
  } else {
    lo = a;
    hi = b;
  }
}

__host__ __device__ __forceinline__ int64_t fdiv(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

// ---- smem row layout -----------------------------------------------------------------------------
/// Rows of D 16-bit elements.  D % 64 == 0: XOR swizzle of the 16-byte chunk index by row & 7;
/// otherwise (80, 96) one 16-byte pad per row (odd chunk stride).  Either way the eight rows of an
/// ldmatrix land on eight distinct 16-byte bank groups.
template <int D>
struct Lay {
  static constexpr bool kSwz = (D % 64) == 0;
  static constexpr int kRow = kSwz ? D : D + 8;
  static __device__ __forceinline__ int off(int row, int chunk) {
    if constexpr (kSwz) return row * D + ((chunk ^ (row & 7)) << 3);
    else return row * kRow + (chunk << 3);
  }
};

// ---- runtime arguments ---------------------------------------------------------------------------
/// What the shared body reads.  Each kernel wrapper fills it from its own params struct, with the
/// modifiers its Variant does not compile already neutral, so one body serves every mask mode,
/// modifier set and K/V source (ragged, CSR pages, block table) of a (head dims, dtype, FP8) cell.
struct FwdArgs {
  const char* q;
  int64_t q_sn, q_sh;  // elements
  const int32_t* qo_indptr;
  const int32_t* q_pos_offset;
  int mask;  // MaskMode
  const uint8_t* custom_mask;
  const int64_t* mask_indptr;
  float sm_scale, softcap;
  int window_left, window_right, chunk, num_sink;
  const float* alibi;
  const float* sinks;
  float k_scale, v_scale;  // FP8 per-tensor fallback when the descriptor has no device scale
  const float* k_scale_ptr;
  const float* v_scale_ptr;
  int scale_head_stride;
  int kv_format;  // 0 ragged, 1 CSR pages, 2 block table
  bool e5m2;
  const char* k;
  const char* v;
  int64_t k_sp, k_st, k_sh, v_sp, v_st, v_sh;  // bytes
  const int32_t* kv_indptr;                    // ragged: token indptr; CSR: page indptr
  const int32_t* page_ids;                     // CSR page indices | block table
  const int32_t* last_page_len;
  const int32_t* seq_lens;
  int64_t bt_stride;
  int page_size;
  void* o;
  int64_t o_sn, o_sh;
  float* lse;
  SplitKVBuffers split;
  WorkPlan plan;
  int Hq, Hkv;
};

template <typename Variant, typename P>
__device__ __forceinline__ void fill_common(FwdArgs& a, const P& p) {
  a.q = static_cast<const char*>(p.q);
  a.q_sn = p.q_stride_n;
  a.q_sh = p.q_stride_h;
  a.qo_indptr = p.qo_indptr;
  a.q_pos_offset = p.q_pos_offset;
  a.mask = static_cast<int>(p.mask);
  a.custom_mask = p.custom_mask;
  a.mask_indptr = p.mask_indptr;
  const AttentionModifiers& m = p.mod;
  a.sm_scale = m.sm_scale;
  a.softcap = Variant::use_softcap ? m.logits_softcap : 0.0f;
  a.window_left = Variant::use_sliding_window ? m.window_left : -1;
  a.window_right = Variant::use_sliding_window ? m.window_right : -1;
  a.num_sink = Variant::use_sliding_window ? m.num_sink_tokens : 0;
  a.chunk = Variant::use_chunked ? m.chunk_size : 0;
  a.alibi = Variant::use_alibi ? m.alibi_slopes : nullptr;
  a.sinks = Variant::use_sinks ? m.sink_logits : nullptr;
  a.k_scale = m.k_scale;
  a.v_scale = m.v_scale;
  a.k_scale_ptr = a.v_scale_ptr = nullptr;
  a.scale_head_stride = 0;
  a.o = p.o;
  a.o_sn = p.o_stride_n;
  a.o_sh = p.o_stride_h;
  a.lse = p.lse;
  a.split = p.split;
  a.plan = p.plan;
  a.Hq = p.num_qo_heads;
  a.Hkv = p.num_kv_heads;
}

__device__ __forceinline__ float tanh_fast(float x) {
  const float e = __expf(2.0f * fabsf(x));
  return copysignf(1.0f - __fdividef(2.0f, e + 1.0f), x);
}

/// Load BKV rows x D of K or V (positions [t0, t0 + BKV), valid below `hi`) into smem.
/// pages == nullptr: contiguous rows (ragged).
template <int D, int BKV, typename T, bool FP8>
__device__ __forceinline__ void load_kv(T* s, const char* base, int64_t sp, int64_t st,
                                        const int32_t* pages, int ps, int t0, int hi, bool e5m2) {
  constexpr int CH = D / 8, N = BKV * CH, IT = (N + 127) / 128;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int idx = it * 128 + static_cast<int>(threadIdx.x);
    if (N % 128 != 0 && idx >= N) break;
    const int r = idx / CH, c = idx % CH;
    const int pos = t0 + r;
    const bool ok = pos < hi;
    const char* row = base;
    if (ok) {
      row = pages != nullptr
                ? base + static_cast<int64_t>(pages[pos / ps]) * sp + static_cast<int64_t>(pos % ps) * st
                : base + static_cast<int64_t>(pos) * st;
    }
    T* dst = s + Lay<D>::off(r, c);
    if constexpr (!FP8) {
      cp_async16(smem_u32(dst), ok ? row + c * 16 : base, ok);
    } else {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (ok) {
        const uint2 w = *reinterpret_cast<const uint2*>(row + c * 8);
        if (e5m2) {
          fp8x4_to<T, true>(w.x, x.x, x.y);
          fp8x4_to<T, true>(w.y, x.z, x.w);
        } else {
          fp8x4_to<T, false>(w.x, x.x, x.y);
          fp8x4_to<T, false>(w.y, x.z, x.w);
        }
      }
      *reinterpret_cast<uint4*>(dst) = x;
    }
  }
}

// ---- FA2 body ------------------------------------------------------------------------------------
constexpr int kBQ = 64;      // packed query rows per CTA
constexpr int kWarps = 4;    // one warp per 16 rows
constexpr int kThreads = 128;

KOPS_HOST_DEVICE constexpr int bkv_of(int dqk, int dvo) { return (dqk > 128 || dvo > 128) ? 32 : 64; }

KOPS_HOST_DEVICE constexpr int fwd_smem_bytes(int dqk, int dvo, int rq, int rqk, int rv) {
  return (kBQ * rq + bkv_of(dqk, dvo) * rqk + bkv_of(dqk, dvo) * rv) * 2;
}

template <int DQK, int DVO>
constexpr int smem_bytes() {
  return fwd_smem_bytes(DQK, DVO, Lay<DQK>::kRow, Lay<DQK>::kRow, Lay<DVO>::kRow);
}

/// The shared body.  __noinline__: compiled once per (head dims, dtype, FP8) and called by every
/// mask / variant / source wrapper, which keeps the translation unit inside its compile budget.
template <int DQK, int DVO, typename T, bool FP8>
__device__ __noinline__ void fwd_body(const FwdArgs& a) {
  constexpr int BQ = kBQ, BKV = bkv_of(DQK, DVO);
  static_assert(DQK % 16 == 0 && DVO % 16 == 0, "mma.sync k16 tiles");
  constexpr bool kQRegs = DQK <= 128;
  constexpr int KQ = DQK / 16, NS = BKV / 8, KP = BKV / 16, NO = DVO / 8;
  using LQ = Lay<DQK>;
  using LV = Lay<DVO>;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* sQ = reinterpret_cast<T*>(smem_raw);
  T* sK = sQ + BQ * LQ::kRow;
  T* sV = sK + BKV * LQ::kRow;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int item = blockIdx.x, hk = blockIdx.y;
  const WorkPlan& wp = a.plan;
  if (wp.block_valid_mask != nullptr && wp.block_valid_mask[item] == 0) return;
  const int b = wp.request_indices[item];
  const int qtile = wp.qo_tile_indices[item];
  const int kvc = wp.kv_tile_indices != nullptr ? wp.kv_tile_indices[item] : 0;
  int chunk_sz = wp.kv_chunk_size != nullptr ? *wp.kv_chunk_size : INT_MAX;
  if (chunk_sz <= 0) chunk_sz = INT_MAX;

  const int G = a.Hq / a.Hkv;
  const int qo_start = a.qo_indptr[b];
  const int qo_len = a.qo_indptr[b + 1] - qo_start;
  const int rows_total = qo_len * G;
  const int row0 = qtile * BQ;
  if (row0 >= rows_total) return;

  // ---- K/V of (request b, kv head hk)
  const char* kb = a.k + hk * a.k_sh;
  const char* vb = a.v + hk * a.v_sh;
  const int32_t* pages = nullptr;
  int kv_len;
  if (a.kv_format == 0) {
    const int k0 = a.kv_indptr[b];
    kb += k0 * a.k_st;
    vb += k0 * a.v_st;
    kv_len = a.kv_indptr[b + 1] - k0;
  } else if (a.kv_format == 1) {
    const int p0 = a.kv_indptr[b], np = a.kv_indptr[b + 1] - p0;
    pages = a.page_ids + p0;
    kv_len = np > 0 ? (np - 1) * a.page_size + a.last_page_len[b] : 0;
  } else {
    pages = a.page_ids + b * a.bt_stride;
    kv_len = a.seq_lens[b];
  }
  const int ps = a.page_size;
  float k_scale = 1.0f;
  [[maybe_unused]] float v_scale = 1.0f;
  if constexpr (FP8) {
    const int sh = a.scale_head_stride != 0 ? hk : 0;
    k_scale = a.k_scale_ptr != nullptr ? a.k_scale_ptr[sh] : a.k_scale;
    v_scale = a.v_scale_ptr != nullptr ? a.v_scale_ptr[sh] : a.v_scale;
    if (!a.e5m2) {  // e4m3 codes are converted as value * 2^-8 (fp8x4_to)
      k_scale *= 256.0f;
      v_scale *= 256.0f;
    }
  }
  const int pos0 = a.q_pos_offset != nullptr ? a.q_pos_offset[b] : kv_len - qo_len;
  const float sm = a.sm_scale != 0.0f ? a.sm_scale : rsqrtf(static_cast<float>(DQK));
  const float scale_nat = sm * k_scale, scale2 = scale_nat * kLog2e;
  const float cap = a.softcap, inv_cap = cap > 0.0f ? 1.0f / cap : 0.0f;
  const bool causal = a.mask == static_cast<int>(MaskMode::kCausal);
  const bool custom = a.mask == static_cast<int>(MaskMode::kCustom);
  const int wl = a.window_left, wr = a.window_right, ck = a.chunk, nsink = a.num_sink;

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

  // ---- per-thread rows g and g + 8 of this warp: visible keys are [rlo, rhi] (inclusive),
  // further limited below by rwlo unless the key is a sink token, and by the custom mask.
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
    if (a.alibi != nullptr) slope2[r] = a.alibi[head[r]] * kLog2e;
    if (custom) mbase[r] = a.mask_indptr[b] + static_cast<int64_t>(qi_r[r]) * kv_len;
  }

  // ---- prologue: Q tile + first K tile
  {
    constexpr int CH = DQK / 8, N = BQ * CH, IT = (N + 127) / 128;
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int idx = it * 128 + tid;
      if (N % 128 != 0 && idx >= N) break;
      const int r = idx / CH, c = idx % CH;
      const int pr = row0 + r;
      const bool ok = pr < rows_total;
      const int qi = ok ? pr / G : 0;
      const int h = hk * G + (ok ? pr - qi * G : 0);
      const char* src = a.q + ((static_cast<int64_t>(qo_start) + qi) * a.q_sn + static_cast<int64_t>(h) * a.q_sh) * 2 + c * 16;
      cp_async16(smem_u32(sQ + LQ::off(r, c)), ok ? src : a.q, ok);
    }
  }
  int t0 = lo;
  if (has_gap && t0 >= nsink && t0 + BKV <= gap_hi) t0 = gap_hi;
  if (t0 < hi) load_kv<DQK, BKV, T, FP8>(sK, kb, a.k_sp, a.k_st, pages, ps, t0, hi, a.e5m2);
  cp_async_commit();

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
    cp_async_wait_all();
    __syncthreads();
    if constexpr (kQRegs) {
      if (first) {
#pragma unroll
        for (int kk = 0; kk < KQ; ++kk) ldsm_x4(qf[kk], sQ_u + LQ::off(a_row, 2 * kk + a_cs) * 2);
      }
    }
    first = false;
    load_kv<DVO, BKV, T, FP8>(sV, vb, a.v_sp, a.v_st, pages, ps, t0, hi, a.e5m2);
    cp_async_commit();

    // S = Q K^T
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
        mma16816<T>(s[2 * np], af, bb[0], bb[1]);
        mma16816<T>(s[2 * np + 1], af, bb[2], bb[3]);
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
    if (a.alibi != nullptr) {
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
            vis = ((a.custom_mask[bit >> 3] >> (bit & 7)) & 1) != 0;
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
    for (int i = 0; i < NO; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }

    cp_async_wait_all();
    __syncthreads();  // V(t0) landed; every warp is done with sK
    int tn = t0 + BKV;
    if (has_gap && tn >= nsink && tn + BKV <= gap_hi) tn = gap_hi;
    if (tn < hi) load_kv<DQK, BKV, T, FP8>(sK, kb, a.k_sp, a.k_st, pages, ps, tn, hi, a.e5m2);
    cp_async_commit();

    // O += P V
#pragma unroll
    for (int kk = 0; kk < KP; ++kk) {
      uint32_t af[4];
      af[0] = pack2<T>(s[2 * kk][0], s[2 * kk][1]);
      af[1] = pack2<T>(s[2 * kk][2], s[2 * kk][3]);
      af[2] = pack2<T>(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      af[3] = pack2<T>(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dp = 0; dp < NO / 2; ++dp) {
        uint32_t bb[4];
        ldsm_x4_t(bb, sV_u + LV::off(kk * 16 + v_row, 2 * dp + v_cs) * 2);
        mma16816<T>(o[2 * dp], af, bb[0], bb[1]);
        mma16816<T>(o[2 * dp + 1], af, bb[2], bb[3]);
      }
    }
    t0 = tn;
  }
  cp_async_wait_all();

  // ---- finalize
  const bool split = wp.split_kv && a.split.o_partial != nullptr;
  float scale[2], lse_n[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    float mf = m[r], lf = l[r], num = 1.0f;
    if (!split && a.sinks != nullptr) {
      const float sk = a.sinks[head[r]] * kLog2e;
      const float mn = fmaxf(mf, sk);
      num = exp2f(mf - mn);
      lf = lf * num + exp2f(sk - mn);
      mf = mn;
    }
    scale[r] = lf > 0.0f ? num / lf : 0.0f;
    lse_n[r] = lf > 0.0f ? (mf + log2f(lf)) * kLn2 : ninf();
    if constexpr (FP8) scale[r] *= v_scale;
  }
  const int Hq = a.Hq;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (!rvalid[r]) continue;
    const int64_t tok = static_cast<int64_t>(qo_start) + qi_r[r];
    if (split) {
      const int64_t prow = static_cast<int64_t>(a.split.merge_indptr[tok]) + kvc;
      float* dst = a.split.o_partial + (prow * Hq + head[r]) * DVO + 2 * (lane & 3);
#pragma unroll
      for (int i = 0; i < NO; ++i) {
        *reinterpret_cast<float2*>(dst + i * 8) = make_float2(o[i][2 * r] * scale[r], o[i][2 * r + 1] * scale[r]);
      }
      if ((lane & 3) == 0) a.split.lse_partial[prow * Hq + head[r]] = lse_n[r];
    } else {
      T* dst = static_cast<T*>(a.o) + tok * a.o_sn + static_cast<int64_t>(head[r]) * a.o_sh + 2 * (lane & 3);
#pragma unroll
      for (int i = 0; i < NO; ++i) {
        *reinterpret_cast<uint32_t*>(dst + i * 8) = pack2<T>(o[i][2 * r] * scale[r], o[i][2 * r + 1] * scale[r]);
      }
      if (a.lse != nullptr && (lane & 3) == 0) a.lse[tok * Hq + head[r]] = lse_n[r];
    }
  }
}

}  // namespace

// ---- kernels (contract: prefill_kernels.cuh) -----------------------------------------------------
// Thin wrappers around fwd_body.  Variant flags compile a modifier in (the runtime value in
// params.mod switches it on); the mask mode is params.mask.  kLayout kNHD names the token-major
// strided loader, which reads NHD and HND pages alike.
template <typename Traits, typename Variant>
__global__ void __launch_bounds__(kThreads)
    ragged_prefill_kernel(const __grid_constant__ RaggedPrefillParams params) {
  static_assert(!Variant::fp8_kv && sizeof(typename Traits::kv_type) == 2, "ragged K/V is 16-bit");
  static_assert(Traits::cta_tile_q == kBQ && Traits::num_warps_q == kWarps &&
                    Traits::cta_tile_kv == bkv_of(Traits::head_dim_qk, Traits::head_dim_vo),
                "fwd_body tile");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  FwdArgs a;
  fill_common<Variant>(a, params);
  const kv::RaggedKVDesc& d = params.kv;
  a.kv_format = 0;
  a.e5m2 = false;
  a.k = static_cast<const char*>(d.k);
  a.v = static_cast<const char*>(d.v);
  a.k_sp = a.v_sp = 0;
  a.k_st = d.k_stride_n * 2;
  a.k_sh = d.k_stride_h * 2;
  a.v_st = d.v_stride_n * 2;
  a.v_sh = d.v_stride_h * 2;
  a.kv_indptr = d.kv_indptr;
  a.page_ids = a.last_page_len = a.seq_lens = nullptr;
  a.bt_stride = 0;
  a.page_size = 1;
  fwd_body<Traits::head_dim_qk, Traits::head_dim_vo, typename Traits::q_type, false>(a);
#else
  (void)params;
  if (threadIdx.x == 0) __trap();  // host refuses sm < 80 (attention_supports / decode_plan)
#endif
}

template <typename Traits, typename Variant, kv::KVLayout kLayout>
__global__ void __launch_bounds__(kThreads)
    paged_prefill_kernel(const __grid_constant__ PagedPrefillParams params) {
  static_assert(kLayout == kv::KVLayout::kNHD || kLayout == kv::KVLayout::kHND,
                "token-major pages (strides carry NHD vs HND)");
  constexpr bool kFp8 = Variant::fp8_kv;
  static_assert(kFp8 == (sizeof(typename Traits::kv_type) == 1), "Variant::fp8_kv matches kv_type");
  static_assert(Traits::cta_tile_q == kBQ && Traits::num_warps_q == kWarps &&
                    Traits::cta_tile_kv == bkv_of(Traits::head_dim_qk, Traits::head_dim_vo),
                "fwd_body tile");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  FwdArgs a;
  fill_common<Variant>(a, params);
  const kv::PagedKVDesc& d = params.kv;
  const int64_t eb = d.stride_unit == StrideUnit::kByte ? 1 : (kFp8 ? 1 : 2);
  a.kv_format = d.table_format == kv::PageTableFormat::kBlockTable ? 2 : 1;
  a.e5m2 = d.k_dtype == DType::kFloat8E5M2;
  a.k = static_cast<const char*>(d.k_data);
  a.v = static_cast<const char*>(d.v_data);
  a.k_sp = d.k_stride_page * eb;
  a.k_st = d.k_stride_token * eb;
  a.k_sh = d.k_stride_head * eb;
  a.v_sp = d.v_stride_page * eb;
  a.v_st = d.v_stride_token * eb;
  a.v_sh = d.v_stride_head * eb;
  a.kv_indptr = d.kv_indptr;
  a.page_ids = a.kv_format == 2 ? d.block_table : d.kv_page_indices;
  a.last_page_len = d.kv_last_page_len;
  a.seq_lens = d.seq_lens;
  a.bt_stride = d.block_table_stride;
  a.page_size = d.page_size;
  a.k_scale_ptr = d.k_scale;
  a.v_scale_ptr = d.v_scale;
  a.scale_head_stride = d.scale_head_stride;
  fwd_body<Traits::head_dim_qk, Traits::head_dim_vo, typename Traits::q_type, kFp8>(a);
#else
  (void)params;
  if (threadIdx.x == 0) __trap();  // host refuses sm < 80 (attention_supports / decode_plan)
#endif
}

// ---- prefill_plan_kernel -------------------------------------------------------------------------
namespace {

constexpr int kPlanThreads = 512;
constexpr int kPlanMaxBatch = 2048;  // planner keeps four [B + 1] int32 arrays in smem (32 KiB)

/// In-place inclusive scan of a[0, n) by one CTA (segment per thread, warp + block carry).
__device__ void block_inclusive_scan(int32_t* a, int n) {
  __shared__ int32_t warp_tot[kPlanThreads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = (n + kPlanThreads - 1) / kPlanThreads;
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
    int32_t w = lane < kPlanThreads / 32 ? warp_tot[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += y;
    }
    if (lane < kPlanThreads / 32) warp_tot[lane] = w;
  }
  __syncthreads();
  int32_t run = inc - sum + (warp > 0 ? warp_tot[warp - 1] : 0);
  for (int i = lo; i < hi; ++i) {
    run += a[i];
    a[i] = run;
  }
  __syncthreads();
}

/// Largest index i in [0, n] with a[i] <= x (a non-decreasing, a[0] <= x).
template <typename P>
__device__ __forceinline__ int last_le(const P* a, int n, int64_t x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (a[mid] <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

}  // namespace

__global__ void __launch_bounds__(kPlanThreads) prefill_plan_kernel(
    int32_t* __restrict__ request_indices, int32_t* __restrict__ qo_tile_indices,
    int32_t* __restrict__ kv_tile_indices, int32_t* __restrict__ merge_indptr,
    int32_t* __restrict__ o_indptr, uint8_t* __restrict__ block_valid_mask,
    int32_t* __restrict__ num_items_out, int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_lens,
    const int batch_size, const int cta_tile_q, const int32_t* __restrict__ kv_chunk_size,
    const int gqa_group_size, const int padded_num_items,
    const int32_t* __restrict__ kv_indptr, const int32_t* __restrict__ kv_last_page_len,
    const int page_size, const int max_partial_rows, uint32_t* __restrict__ total_rows_out) {
  __shared__ int32_t item_off[kPlanMaxBatch + 1], prow_off[kPlanMaxBatch + 1];
  __shared__ int32_t nchunk[kPlanMaxBatch], qoff[kPlanMaxBatch + 1];
  __shared__ int overflow;
  const int B = batch_size, tid = threadIdx.x;
  const int chunk = kv_chunk_size != nullptr ? *kv_chunk_size : 0;
  for (int b = tid; b <= B; b += kPlanThreads) qoff[b] = qo_indptr[b];
  __syncthreads();
  for (int b = tid; b < B; b += kPlanThreads) {
    const int qn = max(qoff[b + 1] - qoff[b], 0);
    int kvl;
    if (kv_lens != nullptr) {
      kvl = kv_lens[b];
    } else if (kv_last_page_len == nullptr) {
      kvl = kv_indptr[b + 1] - kv_indptr[b];
    } else {
      const int np = kv_indptr[b + 1] - kv_indptr[b];
      kvl = np > 0 ? (np - 1) * page_size + kv_last_page_len[b] : 0;
    }
    const int nc = chunk > 0 ? max(1, static_cast<int>((static_cast<int64_t>(max(kvl, 0)) + chunk - 1) / chunk)) : 1;
    const int tiles = static_cast<int>((static_cast<int64_t>(qn) * gqa_group_size + cta_tile_q - 1) / cta_tile_q);
    nchunk[b] = nc;
    item_off[b + 1] = tiles * nc;
    prow_off[b + 1] = qn * nc;
  }
  if (tid == 0) {
    item_off[0] = 0;
    prow_off[0] = 0;
  }
  __syncthreads();
  block_inclusive_scan(item_off + 1, B);
  block_inclusive_scan(prow_off + 1, B);
  const int total_items = item_off[B], total_prow = prow_off[B], total_q = qoff[B];
  if (tid == 0) {
    overflow = total_items > padded_num_items || (chunk > 0 && total_prow > max_partial_rows) ? 1 : 0;
    *overflow_flag = overflow;
    *num_items_out = overflow ? 0 : total_items;
    *total_rows_out = static_cast<uint32_t>(max(total_q, 0));
  }
  __syncthreads();
  const bool ovf = overflow != 0;
  for (int i = tid; i < padded_num_items; i += kPlanThreads) {
    if (ovf || i >= total_items) {
      block_valid_mask[i] = 0;
      continue;
    }
    const int b = last_le(item_off, B, i);
    const int local = i - item_off[b], nc = nchunk[b];
    request_indices[i] = b;
    qo_tile_indices[i] = local / nc;
    kv_tile_indices[i] = local - (local / nc) * nc;
    block_valid_mask[i] = 1;
  }
  if (chunk > 0 && !ovf) {
    for (int r = tid; r <= total_q; r += kPlanThreads) {
      if (r == total_q) {
        merge_indptr[r] = total_prow;
        continue;
      }
      const int b = last_le(qoff, B, r);
      merge_indptr[r] = prow_off[b] + (r - qoff[b]) * nchunk[b];
    }
    for (int b = tid; b <= B; b += kPlanThreads) o_indptr[b] = prow_off[b];
  }
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

constexpr int kInvariantChunk = 4096;  // kv chunk under ctx.batch_invariant (never batch-dependent)
constexpr int kMaxSplits = 16;
constexpr uint64_t kPrefillKind = 0xA1;

// Compiled (Dqk, Dvo) pairs: each pair x {bf16, fp16} x {ragged, paged} is one kernel, and the
// FP8 pairs add one paged FP8 kernel each (e4m3 / e5m2 is a runtime choice).  ptxas takes ~2 s per
// FA2 kernel on the dev box, which is what bounds this list (TU budget 90 s, ROADMAP §1.5).
#define KOPS_PREFILL_DIMS(X) X(64, 64) X(80, 80) X(96, 96) X(128, 128) X(256, 256) X(192, 128)
#define KOPS_PREFILL_FP8_DIMS(X) X(64, 64) X(128, 128) X(256, 256)

bool dims_compiled(int dqk, int dvo) {
#define KOPS_DIM_CASE(a, b) if (dqk == (a) && dvo == (b)) return true;
  KOPS_PREFILL_DIMS(KOPS_DIM_CASE)
#undef KOPS_DIM_CASE
  return false;
}

bool fp8_dims_compiled(int dqk, int dvo) {
#define KOPS_DIM_CASE(a, b) if (dqk == (a) && dvo == (b)) return true;
  KOPS_PREFILL_FP8_DIMS(KOPS_DIM_CASE)
#undef KOPS_DIM_CASE
  return false;
}

template <int DQK, int DVO>
constexpr bool fp8_compiled() {
#define KOPS_DIM_CASE(a, b) || (DQK == (a) && DVO == (b))
  return false KOPS_PREFILL_FP8_DIMS(KOPS_DIM_CASE);
#undef KOPS_DIM_CASE
}

/// Everything run() needs to pick an instantiation, packed into AttentionPlan::kernel_id.
struct Key {
  int dqk = 0, dvo = 0;
  int dt = 0;    // 0 bf16, 1 fp16
  int fp8 = 0;   // FP8 K/V (e4m3 or e5m2, from PagedKVDesc::k_dtype)
  int mask = 0;  // MaskMode
  int mods = 0;  // bit0 window, bit1 softcap, bit2 alibi, bit3 sinks, bit4 chunked (AttentionSpec)
  int paged = 0;
};
uint64_t encode(const Key& k) {
  return (kPrefillKind << 56) | (static_cast<uint64_t>(k.paged) << 52) |
         (static_cast<uint64_t>(k.mods) << 44) | (static_cast<uint64_t>(k.mask) << 40) |
         (static_cast<uint64_t>(k.fp8) << 37) | (static_cast<uint64_t>(k.dt) << 36) |
         (static_cast<uint64_t>(k.dqk) << 16) | static_cast<uint64_t>(k.dvo);
}
bool decode_key(uint64_t id, Key* k) {
  if ((id >> 56) != kPrefillKind) return false;
  k->paged = static_cast<int>((id >> 52) & 1);
  k->mods = static_cast<int>((id >> 44) & 31);
  k->mask = static_cast<int>((id >> 40) & 3);
  k->fp8 = static_cast<int>((id >> 37) & 1);
  k->dt = static_cast<int>((id >> 36) & 1);
  k->dqk = static_cast<int>((id >> 16) & 0xFFFF);
  k->dvo = static_cast<int>(id & 0xFFFF);
  return true;
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

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

// ---- dispatch ------------------------------------------------------------------------------------
template <typename Traits, bool PAGED, typename P>
Status launch_one(const P& params, dim3 grid, cudaStream_t stream) {
  using V = AttentionVariant<true, true, true, true, true, sizeof(typename Traits::kv_type) == 1>;
  constexpr int smem = smem_bytes<Traits::head_dim_qk, Traits::head_dim_vo>();
  if constexpr (PAGED) {
    auto* fn = paged_prefill_kernel<Traits, V, kv::KVLayout::kNHD>;
    static const cudaError_t attr =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("paged_prefill smem attribute", attr));
    fn<<<grid, kThreads, smem, stream>>>(params);
  } else {
    auto* fn = ragged_prefill_kernel<Traits, V>;
    static const cudaError_t attr =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return (cudaGetLastError(), launch_error("ragged_prefill smem attribute", attr));
    fn<<<grid, kThreads, smem, stream>>>(params);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error(PAGED ? "paged_prefill launch" : "ragged_prefill launch", err);
  return Status::kSuccess;
}

template <int DQK, int DVO, typename T, bool PAGED, typename P>
Status launch_dtype_kv(const Key& k, const P& params, dim3 grid, cudaStream_t st) {
  constexpr int BKV = bkv_of(DQK, DVO);
  if (!k.fp8) {
    using Tr = PrefillTraits<DQK, DVO, kBQ, BKV, kWarps, 1, 1, T, T, T, Sm80>;
    return launch_one<Tr, PAGED>(params, grid, st);
  }
  if constexpr (PAGED && fp8_compiled<DQK, DVO>()) {
    using Tr = PrefillTraits<DQK, DVO, kBQ, BKV, kWarps, 1, 1, T, __nv_fp8_e4m3, T, Sm80>;
    return launch_one<Tr, true>(params, grid, st);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "prefill: FP8 K/V not compiled for %d/%d (paged only)",
                             DQK, DVO);
  }
}

template <bool PAGED, typename P>
Status dispatch(const Key& k, const P& params, dim3 grid, cudaStream_t st) {
#define KOPS_DIM_CASE(a, b)                                                                  \
  if (k.dqk == (a) && k.dvo == (b)) {                                                        \
    return k.dt == 0 ? launch_dtype_kv<a, b, __nv_bfloat16, PAGED>(k, params, grid, st)      \
                     : launch_dtype_kv<a, b, __half, PAGED>(k, params, grid, st);            \
  }
  KOPS_PREFILL_DIMS(KOPS_DIM_CASE)
#undef KOPS_DIM_CASE
  return detail::set_error(Status::kUnsupportedShape, "prefill: head dims %d/%d not compiled", k.dqk,
                           k.dvo);
}

// ---- plan layout ---------------------------------------------------------------------------------
size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

/// device_persistent layout: request | qo_tile | kv_tile | valid mask | chunk size | num items |
/// overflow | total rows | o_indptr[B + 1] | merge_indptr[total_q + 1].  Regions 256-byte aligned.
struct Carve {
  size_t req, qtile, kvtile, valid, chunk, nitems, overflow, total_rows, oind, merge, bytes;
};
Carve carve(int64_t items, int64_t total_q, int64_t batch) {
  Carve c{};
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
  c.nitems = take(4);
  c.overflow = take(4);
  c.total_rows = take(4);
  c.oind = take(static_cast<size_t>(batch + 1) * 4);
  c.merge = take(static_cast<size_t>(total_q + 1) * 4);
  c.bytes = o;
  return c;
}

/// Scratch = [o_partial | lse_partial], 256-byte aligned regions sized for `rows` partial rows.
size_t scratch_bytes(int64_t rows, int Hq, int dvo) {
  return align_up(static_cast<size_t>(rows) * Hq * dvo * 4, 256) +
         align_up(static_cast<size_t>(rows) * Hq * 4, 256);
}
/// Inverse of scratch_bytes: the largest row count whose layout fits in `bytes` (>= the planned
/// count, so the lse offset lies past every planned o row).
int64_t scratch_rows(size_t bytes, int Hq, int dvo) {
  const size_t per = static_cast<size_t>(Hq) * (dvo + 1) * 4;
  int64_t rows = static_cast<int64_t>(bytes / per);
  while (rows > 0 && scratch_bytes(rows, Hq, dvo) > bytes) --rows;
  return rows;
}

Status read_host_i32(const Tensor& t, int64_t n, const char* name, std::vector<int32_t>* out) {
  if (t.data == nullptr || t.dtype != DType::kInt32 || t.ndim != 1 || t.shape[0] != n ||
      (t.stride[0] != 1 && n > 1)) {
    return detail::set_error(Status::kInvalidArgument, "prefill_plan: %s must be int32 [%lld]", name,
                             static_cast<long long>(n));
  }
  if (t.device >= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "prefill_plan: %s must be host memory (device planning runs in run())",
                             name);
  }
  const int32_t* p = reinterpret_cast<const int32_t*>(static_cast<const char*>(t.data) + t.byte_offset);
  out->assign(p, p + n);
  return Status::kSuccess;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

// ---- ops.h ---------------------------------------------------------------------------------------
Status attention_supports(const AttentionSpec& spec, const LaunchContext& ctx,
                          RejectReason* reasons, int32_t max_reasons, int32_t* num_reasons) {
  int32_t n = 0;
  Status first = Status::kSuccess;
  const char* first_detail = "";
  auto reject = [&](Status code, const char* detail) {
    if (reasons != nullptr && n < max_reasons) reasons[n] = RejectReason{0, code, detail};
    ++n;
    if (first == Status::kSuccess) {
      first = code;
      first_detail = detail;
    }
  };
  if (ctx_is_amd(ctx)) reject(Status::kUnsupportedBackend, "attention: CUDA mma.sync only");
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    reject(Status::kUnsupportedArch, "attention: mma.sync bf16/fp16 needs sm_80+");
  }
  if (spec.num_q_heads <= 0 || spec.num_kv_heads <= 0 || spec.num_q_heads % spec.num_kv_heads != 0) {
    reject(Status::kUnsupportedShape, "attention: num_q_heads must be a multiple of num_kv_heads");
  }
  if (!dims_compiled(spec.head_dim_qk, spec.head_dim_vo)) {
    reject(Status::kUnsupportedShape,
           "attention: head dims not in {64,80,96,128,256} or 192/128");
  }
  const bool q16 = spec.q_dtype == DType::kBFloat16 || spec.q_dtype == DType::kFloat16;
  if (!q16 || spec.o_dtype != spec.q_dtype) {
    reject(Status::kUnsupportedDtype, "attention: q/o must be the same bf16 or fp16 dtype");
  }
  const bool kv_native = spec.kv_dtype == spec.q_dtype && spec.k_codec == kv::KVCodec::kNative &&
                         spec.v_codec == kv::KVCodec::kNative;
  const bool fp8_codec = (spec.k_codec == kv::KVCodec::kFP8PerTensor ||
                          spec.k_codec == kv::KVCodec::kFP8PerHead) &&
                         spec.v_codec == spec.k_codec;
  const bool kv_fp8 = (spec.kv_dtype == DType::kFloat8E4M3 || spec.kv_dtype == DType::kFloat8E5M2) &&
                      fp8_codec;
  if (!kv_native && !kv_fp8) {
    reject(Status::kUnsupportedDtype,
           "attention: K/V must match q, or be FP8 e4m3/e5m2 with a per-tensor / per-head scale");
  }
  if (kv_fp8 && spec.page_size == 0) {
    reject(Status::kUnsupportedDtype, "attention: FP8 K/V is supported on paged caches only");
  }
  if (kv_fp8 && !fp8_dims_compiled(spec.head_dim_qk, spec.head_dim_vo)) {
    reject(Status::kUnsupportedShape, "attention: FP8 K/V is compiled for head dims 64, 128, 256");
  }
  if (spec.page_size != 0 && spec.kv_layout != kv::KVLayout::kNHD &&
      spec.kv_layout != kv::KVLayout::kHND) {
    reject(Status::kUnsupportedLayout, "attention: paged K/V must be NHD or HND");
  }
  if (spec.fp8_tensor_core) {
    reject(Status::kUnsupportedDtype, "attention: FP8 tensor-core compute is not an SM80-class path");
  }
  if (num_reasons != nullptr) *num_reasons = n;
  if (first != Status::kSuccess) {
    return detail::set_error(first, "%s", first_detail);
  }
  return Status::kSuccess;
}

Status prefill_plan(const AttentionSpec& spec, const Tensor& qo_indptr_host,
                    const Tensor& kv_lens_host, int32_t max_batch, const LaunchContext& ctx,
                    const Workspace& ws, AttentionPlan* plan) {
  if (plan == nullptr) return detail::set_error(Status::kInvalidArgument, "prefill_plan: null plan");
  Status s = attention_supports(spec, ctx, nullptr, 0, nullptr);
  if (s != Status::kSuccess) return s;
  const int64_t B = qo_indptr_host.ndim == 1 ? qo_indptr_host.shape[0] - 1 : -1;
  if (B < 1) return detail::set_error(Status::kInvalidArgument, "prefill_plan: qo_indptr must be [B + 1], B >= 1");
  if (max_batch > 0 && B > max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "prefill_plan: batch %lld > max_batch %d",
                             static_cast<long long>(B), max_batch);
  }
  std::vector<int32_t> qo, kvl;
  s = read_host_i32(qo_indptr_host, B + 1, "qo_indptr_host", &qo);
  if (s != Status::kSuccess) return s;
  s = read_host_i32(kv_lens_host, B, "kv_lens_host", &kvl);
  if (s != Status::kSuccess) return s;
  if (qo[0] != 0) return detail::set_error(Status::kInvalidArgument, "prefill_plan: qo_indptr[0] != 0");
  // Graph strategy: the host lengths are the capacity; prefill_plan_kernel re-plans every replay
  // from the device lengths inside the graph.
  const bool dev_plan = ctx.strategy == ExecutionStrategy::kGraph;
  const int64_t B_cap = dev_plan ? std::max<int64_t>(max_batch, B) : B;
  if (dev_plan && B_cap > kPlanMaxBatch) {
    return detail::set_error(Status::kCapacityExceeded, "prefill_plan: device planner holds %d requests",
                             kPlanMaxBatch);
  }
  const int G = spec.num_q_heads / spec.num_kv_heads;
  int64_t base_ctas = 0;
  int max_kv = 0;
  for (int64_t b = 0; b < B; ++b) {
    const int64_t qn = static_cast<int64_t>(qo[b + 1]) - qo[b];
    if (qn < 0 || kvl[b] < 0) {
      return detail::set_error(Status::kInvalidArgument, "prefill_plan: negative length at request %lld",
                               static_cast<long long>(b));
    }
    base_ctas += (qn * G + kBQ - 1) / kBQ;
    max_kv = std::max(max_kv, kvl[b]);
  }
  const int64_t total_q = qo[B];
  base_ctas *= spec.num_kv_heads;

  // kv chunking.  batch_invariant: fixed chunk and always the split path, so a request's rounding
  // never depends on its neighbours.  Otherwise split only when the q tiles fill less than one
  // wave, towards two CTAs per SM.
  const int bkv = bkv_of(spec.head_dim_qk, spec.head_dim_vo);
  int chunk = 0;
  bool split = false;
  if (ctx.batch_invariant) {
    chunk = kInvariantChunk;
    split = true;
  } else {
    const int64_t sms = std::max(ctx.compute_units, 1);
    if (base_ctas < sms && max_kv >= 8 * bkv) {
      int64_t splits = std::min<int64_t>((2 * sms + base_ctas - 1) / base_ctas, kMaxSplits);
      splits = std::min<int64_t>(splits, max_kv / (4 * bkv));
      if (splits > 1) {
        const int64_t c = (max_kv + splits - 1) / splits;
        chunk = static_cast<int>((c + bkv - 1) / bkv * bkv);
        split = true;
      }
    }
  }
  auto chunks_of = [&](int64_t kv) -> int64_t {
    return split ? std::max<int64_t>(1, (kv + chunk - 1) / chunk) : 1;
  };

  // Work items in (request, q tile, kv chunk) order; prefill_plan_kernel reproduces this order.
  std::vector<int32_t> req, qt, kt, merge, oind;
  int64_t items = 0, prow = 0;
  if (dev_plan) {
    const int64_t mc = chunks_of(max_kv);
    items = ((total_q * G + kBQ - 1) / kBQ + B_cap) * mc;
    prow = split ? total_q * mc : 0;
  } else {
    merge.assign(static_cast<size_t>(total_q + 1), 0);
    oind.assign(static_cast<size_t>(B + 1), 0);
    for (int64_t b = 0; b < B; ++b) {
      const int64_t qn = static_cast<int64_t>(qo[b + 1]) - qo[b];
      const int64_t tiles = (qn * G + kBQ - 1) / kBQ;
      const int64_t nc = chunks_of(kvl[b]);
      for (int64_t t = 0; t < tiles; ++t) {
        for (int64_t c = 0; c < nc; ++c) {
          req.push_back(static_cast<int32_t>(b));
          qt.push_back(static_cast<int32_t>(t));
          kt.push_back(static_cast<int32_t>(c));
        }
      }
      oind[static_cast<size_t>(b)] = static_cast<int32_t>(prow);
      for (int64_t i = 0; i < qn; ++i) {
        merge[static_cast<size_t>(qo[b] + i)] = static_cast<int32_t>(prow);
        prow += nc;
      }
    }
    merge[static_cast<size_t>(total_q)] = static_cast<int32_t>(prow);
    oind[static_cast<size_t>(B)] = static_cast<int32_t>(prow);
    items = static_cast<int64_t>(req.size());
    if (!split) prow = 0;
  }
  if (items > INT32_MAX || prow > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "prefill_plan: %lld work items", static_cast<long long>(items));
  }

  const Carve cv = carve(std::max<int64_t>(items, 1), total_q, B_cap);
  *plan = AttentionPlan{};
  Key k;
  k.dqk = spec.head_dim_qk;
  k.dvo = spec.head_dim_vo;
  k.dt = spec.q_dtype == DType::kFloat16 ? 1 : 0;
  k.fp8 = (spec.kv_dtype == DType::kFloat8E4M3 || spec.kv_dtype == DType::kFloat8E5M2) ? 1 : 0;
  k.mask = static_cast<int>(spec.mask);
  k.mods = spec_mods(spec);
  k.paged = spec.page_size != 0 ? 1 : 0;
  plan->kernel_id = encode(k);
  plan->plan_generation = ctx.capture_generation;
  plan->strategy = dev_plan ? ExecutionStrategy::kGraph : ExecutionStrategy::kModular;
  plan->split_kv = split;
  plan->max_batch = static_cast<int32_t>(B_cap);
  plan->max_total_q = static_cast<int32_t>(total_q);
  plan->max_kv_len = max_kv;
  plan->num_launches = (dev_plan ? 1 : 0) + 1 + (split ? 1 : 0);
  plan->workspace.device_persistent_bytes = cv.bytes;
  plan->workspace.device_scratch_bytes = split ? scratch_bytes(prow, spec.num_q_heads, spec.head_dim_vo) : 0;
  plan->workspace.host_pinned_bytes = cv.bytes;  // optional staging of the upload
  plan->workspace.alignment = 256;
  if (ws.device_persistent == nullptr || ws.device_persistent_bytes < cv.bytes ||
      ws.device_scratch_bytes < plan->workspace.device_scratch_bytes ||
      (plan->workspace.device_scratch_bytes != 0 && ws.device_scratch == nullptr)) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "prefill_plan: needs %zu persistent + %zu scratch bytes (see plan->workspace)",
                             cv.bytes, plan->workspace.device_scratch_bytes);
  }
  if ((reinterpret_cast<uintptr_t>(ws.device_persistent) & 255u) != 0 ||
      (reinterpret_cast<uintptr_t>(ws.device_scratch) & 255u) != 0) {
    return detail::set_error(Status::kMisaligned, "prefill_plan: workspace must be 256-byte aligned");
  }

  // One host image of the persistent region, one copy (the device planner overwrites the item
  // arrays every replay; valid = 0 until it runs).
  std::vector<unsigned char> img(cv.bytes, 0);
  const int32_t chunk32 = split ? chunk : 0, n32 = static_cast<int32_t>(items), zero = 0;
  const uint32_t rows32 = static_cast<uint32_t>(total_q);
  if (!dev_plan) {
    std::memcpy(img.data() + cv.req, req.data(), req.size() * 4);
    std::memcpy(img.data() + cv.qtile, qt.data(), qt.size() * 4);
    std::memcpy(img.data() + cv.kvtile, kt.data(), kt.size() * 4);
    std::memset(img.data() + cv.valid, 1, static_cast<size_t>(items));
    std::memcpy(img.data() + cv.oind, oind.data(), oind.size() * 4);
    std::memcpy(img.data() + cv.merge, merge.data(), merge.size() * 4);
  }
  std::memcpy(img.data() + cv.chunk, &chunk32, 4);
  std::memcpy(img.data() + cv.nitems, dev_plan ? &zero : &n32, 4);
  std::memcpy(img.data() + cv.overflow, &zero, 4);
  std::memcpy(img.data() + cv.total_rows, &rows32, 4);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const void* src = img.data();
  if (ws.host_pinned != nullptr && ws.host_pinned_bytes >= cv.bytes) {
    std::memcpy(ws.host_pinned, img.data(), cv.bytes);
    src = ws.host_pinned;
  }
  cudaError_t err = cudaMemcpyAsync(ws.device_persistent, src, cv.bytes, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess && src == ws.host_pinned) err = cudaStreamSynchronize(stream);  // staging reuse
  if (err != cudaSuccess) return launch_error("prefill_plan upload", err);
  char* base = static_cast<char*>(ws.device_persistent);
  WorkPlan& wp = plan->device_plan;
  wp.request_indices = reinterpret_cast<const int32_t*>(base + cv.req);
  wp.qo_tile_indices = reinterpret_cast<const int32_t*>(base + cv.qtile);
  wp.kv_tile_indices = reinterpret_cast<const int32_t*>(base + cv.kvtile);
  wp.block_valid_mask = reinterpret_cast<const uint8_t*>(base + cv.valid);
  wp.kv_chunk_size = reinterpret_cast<const int32_t*>(base + cv.chunk);
  wp.o_indptr = reinterpret_cast<const int32_t*>(base + cv.oind);
  wp.total_num_rows = reinterpret_cast<const uint32_t*>(base + cv.total_rows);
  wp.num_items = n32;
  wp.padded_num_items = n32;
  wp.split_kv = split;
  return Status::kSuccess;
}

namespace {

Tensor merge_view_o(const void* o, DType dt, int64_t rows, int Hq, int dvo, int64_t sn, int64_t sh, int dev) {
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

Tensor merge_view_1d(const void* p, DType dt, int64_t n, int dev) {
  Tensor t{};
  t.data = const_cast<void*>(p);
  t.dtype = dt;
  t.ndim = 1;
  t.device = dev;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

/// Shared tail of ragged_prefill_run / paged_prefill_run.
template <bool PAGED, typename P>
Status prefill_run_impl(const AttentionPlan& plan, const P& params, const Workspace& ws,
                        const LaunchContext& ctx, const char* who) {
  Key k;
  if (!decode_key(plan.kernel_id, &k) || k.paged != (PAGED ? 1 : 0)) {
    return detail::set_error(Status::kInvalidArgument, "%s: plan was not made for this entry point", who);
  }
  const bool dev_plan = plan.strategy == ExecutionStrategy::kGraph;
  if (dev_plan && plan.plan_generation != ctx.capture_generation) {
    return detail::set_error(Status::kPlanInvalidated, "%s: plan generation %u != capture generation %u",
                             who, plan.plan_generation, ctx.capture_generation);
  }
  if (params.q == nullptr || params.o == nullptr || params.qo_indptr == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: q, o and qo_indptr are required", who);
  }
  if (params.num_kv_heads <= 0 || params.num_qo_heads <= 0 ||
      params.num_qo_heads % params.num_kv_heads != 0 || params.num_kv_heads > 65535) {
    return detail::set_error(Status::kUnsupportedShape, "%s: heads %d / %d", who, params.num_qo_heads,
                             params.num_kv_heads);
  }
  if (params.batch_size < 1 || params.batch_size > plan.max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "%s: batch %d outside the plan (1..%d)", who,
                             params.batch_size, plan.max_batch);
  }
  if (!aligned16(params.q) || (params.q_stride_n & 7) != 0 || (params.q_stride_h & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(params.o) & 3u) != 0 || (params.o_stride_n & 1) != 0 ||
      (params.o_stride_h & 1) != 0) {
    return detail::set_error(Status::kMisaligned,
                             "%s: q needs 16-byte rows (strides %% 8), o 4-byte pairs", who);
  }
  if (static_cast<MaskMode>(k.mask) == MaskMode::kCustom &&
      (params.custom_mask == nullptr || params.mask_indptr == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "%s: kCustom needs custom_mask and mask_indptr", who);
  }
  const WorkPlan& dp = plan.device_plan;
  const Carve cv = carve(std::max<int64_t>(dp.padded_num_items, 1), plan.max_total_q, plan.max_batch);
  char* base = static_cast<char*>(ws.device_persistent);
  if (base == nullptr || base + cv.req != reinterpret_cast<const char*>(dp.request_indices)) {
    return detail::set_error(Status::kPlanInvalidated, "%s: workspace differs from the planned one", who);
  }
  if (dp.padded_num_items == 0) return Status::kSuccess;
  P pp = params;
  pp.plan = dp;
  neutralize(k.mods, &pp.mod);
  pp.mask = static_cast<MaskMode>(k.mask);
  const int Hq = params.num_qo_heads;
  int64_t prow_cap = 0;
  if (plan.split_kv) {
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < plan.workspace.device_scratch_bytes) {
      return detail::set_error(Status::kWorkspaceTooSmall, "%s: scratch < %zu bytes", who,
                               plan.workspace.device_scratch_bytes);
    }
    prow_cap = scratch_rows(plan.workspace.device_scratch_bytes, Hq, k.dvo);
    pp.split.merge_indptr = reinterpret_cast<const int32_t*>(base + cv.merge);
    pp.split.o_partial = static_cast<float*>(ws.device_scratch);
    pp.split.lse_partial = reinterpret_cast<float*>(static_cast<char*>(ws.device_scratch) +
                                                    align_up(static_cast<size_t>(prow_cap) * Hq * k.dvo * 4, 256));
    pp.split.max_splits = kMaxSplits;
  } else {
    pp.split = SplitKVBuffers{};
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  if (dev_plan) {
    const int32_t* kv_lens = nullptr;
    const int32_t* kv_indptr = nullptr;
    const int32_t* last = nullptr;
    int ps = 0;
    if constexpr (PAGED) {
      if (params.kv.table_format == kv::PageTableFormat::kBlockTable) {
        kv_lens = params.kv.seq_lens;
      } else {
        kv_indptr = params.kv.kv_indptr;
        last = params.kv.kv_last_page_len;
        ps = params.kv.page_size;
      }
    } else {
      kv_indptr = params.kv.kv_indptr;
    }
    prefill_plan_kernel<<<1, kPlanThreads, 0, stream>>>(
        reinterpret_cast<int32_t*>(base + cv.req), reinterpret_cast<int32_t*>(base + cv.qtile),
        reinterpret_cast<int32_t*>(base + cv.kvtile), reinterpret_cast<int32_t*>(base + cv.merge),
        reinterpret_cast<int32_t*>(base + cv.oind), reinterpret_cast<uint8_t*>(base + cv.valid),
        reinterpret_cast<int32_t*>(base + cv.nitems), reinterpret_cast<int32_t*>(base + cv.overflow),
        params.qo_indptr, kv_lens, params.batch_size, kBQ, reinterpret_cast<const int32_t*>(base + cv.chunk),
        params.num_qo_heads / params.num_kv_heads, dp.padded_num_items, kv_indptr, last, ps,
        static_cast<int>(prow_cap), reinterpret_cast<uint32_t*>(base + cv.total_rows));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("prefill_plan_kernel launch", err);
  }
  const dim3 grid(static_cast<unsigned>(dp.padded_num_items), static_cast<unsigned>(params.num_kv_heads));
  Status s = dispatch<PAGED>(k, pp, grid, stream);
  if (s != Status::kSuccess || !plan.split_kv) return s;

  const DType odt = k.dt == 0 ? DType::kBFloat16 : DType::kFloat16;
  const Tensor o = merge_view_o(params.o, odt, plan.max_total_q, Hq, k.dvo, params.o_stride_n,
                                params.o_stride_h, ctx.device);
  Tensor lse{};
  if (params.lse != nullptr) {
    lse = merge_view_o(params.lse, DType::kFloat32, plan.max_total_q, Hq, 1, Hq, 1, ctx.device);
    lse.ndim = 2;
  }
  const Tensor sinks = pp.mod.sink_logits != nullptr
                           ? merge_view_1d(pp.mod.sink_logits, DType::kFloat32, Hq, ctx.device)
                           : Tensor{};
  const Tensor rows = merge_view_1d(base + cv.total_rows, DType::kUInt32, 1, ctx.device);
  return internal::merge_split_kv_run(o, lse, pp.split, sinks, rows, ctx);
}

}  // namespace

Status ragged_prefill_run(const AttentionPlan& plan, const RaggedPrefillParams& params,
                          const Workspace& ws, const LaunchContext& ctx) {
  if (params.kv.k == nullptr || params.kv.v == nullptr || params.kv.kv_indptr == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "ragged_prefill: kv.k, kv.v, kv.kv_indptr required");
  }
  if (!aligned16(params.kv.k) || !aligned16(params.kv.v) || ((params.kv.k_stride_n | params.kv.k_stride_h |
                                                               params.kv.v_stride_n | params.kv.v_stride_h) & 7) != 0) {
    return detail::set_error(Status::kMisaligned, "ragged_prefill: K/V rows must be 16-byte aligned");
  }
  return prefill_run_impl<false>(plan, params, ws, ctx, "ragged_prefill");
}

Status paged_prefill_run(const AttentionPlan& plan, const PagedPrefillParams& params,
                         const Workspace& ws, const LaunchContext& ctx) {
  const kv::PagedKVDesc& d = params.kv;
  if (d.k_data == nullptr || d.v_data == nullptr || d.page_size <= 0) {
    return detail::set_error(Status::kInvalidArgument, "paged_prefill: k_data, v_data, page_size required");
  }
  if (d.layout != kv::KVLayout::kNHD && d.layout != kv::KVLayout::kHND) {
    return detail::set_error(Status::kUnsupportedLayout, "paged_prefill: layout must be NHD or HND");
  }
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    if (d.block_table == nullptr || d.seq_lens == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "paged_prefill: block_table and seq_lens required");
    }
  } else if (d.table_format == kv::PageTableFormat::kCSR) {
    if (d.kv_indptr == nullptr || d.kv_page_indices == nullptr || d.kv_last_page_len == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "paged_prefill: kv_indptr, kv_page_indices, kv_last_page_len required");
    }
  } else {
    return detail::set_error(Status::kUnsupportedLayout, "paged_prefill: slot mapping is write-only");
  }
  const int64_t mask = (d.stride_unit == StrideUnit::kByte) ? 15 : 7;
  if (!aligned16(d.k_data) || !aligned16(d.v_data) ||
      ((d.k_stride_page | d.k_stride_token | d.k_stride_head | d.v_stride_page | d.v_stride_token |
        d.v_stride_head) & mask) != 0) {
    return detail::set_error(Status::kMisaligned, "paged_prefill: K/V rows must be 16-byte aligned");
  }
  return prefill_run_impl<true>(plan, params, ws, ctx, "paged_prefill");
}

Status block_sparse_prefill_run(const AttentionPlan&, const PagedPrefillParams&,
                                const BlockSparseMask&, const Workspace&, const LaunchContext&) {
  return detail::set_error(Status::kNotImplemented, "block_sparse_prefill: A10 is not part of B8");
}

}  // namespace kops::attn
