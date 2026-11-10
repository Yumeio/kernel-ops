// =============================================================================
// kernel-ops | csrc/moe/routing.cu
// B6 rows 1-4: router scoring + expert selection (M01).
//   moe_topk_kernel              flat top-k   (Mixtral, Qwen3-MoE, GPT-OSS, ...)
//   moe_grouped_topk_kernel      group-limited top-k (DeepSeek-V2/V3/V4, Kimi K2/K3)
//   moe_fused_router_topk_kernel decode fusion: router GEMM + scoring + top-k in one CTA
//   moe_hash_route_kernel        checkpoint hash table + uniform weights
//
// One warp handles one token. Each lane owns the experts congruent to its lane id,
// so a row of E <= 1024 experts costs at most 32 values per lane. Selection uses
// the documented tie-break: lower expert index wins. Weights use the unbiased
// score; correction_bias only participates in selection.
//
// The fused router copies the dot-product reduction order of gemm::router_gemm
// (eight warps split the hidden dimension, xor butterfly, then warp partials are
// summed in warp order), so route_fused is bit-equal to router_gemm + route.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <cuda.h>
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
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"

namespace kops::gemm {
Status router_gemm(const RouterGemmParams& params, bool two_term, DType x_dtype, DType w_dtype,
                   const LaunchContext& ctx);
}

namespace kops::moe {
namespace {

constexpr int kMaxExperts = 1024;
constexpr int kMaxValuesPerLane = kMaxExperts / 32;  // 32
constexpr int kMaxTopK = 16;
constexpr int kMaxGroups = 32;
constexpr int kRouterWarps = 8;
constexpr int kFusedM = 8;

__device__ __forceinline__ float load_bias(const RouterParams& p, int e) {
  return p.correction_bias != nullptr ? p.correction_bias[e] : 0.0f;
}

/// Score function selected at run time so one device body serves all five
/// RouterScore values (the template argument stays part of the kernel contract).
__device__ __forceinline__ float score_of(float x, int score) {
  switch (score) {
    case static_cast<int>(RouterScore::kSigmoid): return sigmoid(x);
    case static_cast<int>(RouterScore::kSqrtSoftplus): return sqrtf(softplus(x));
    default: return x;
  }
}

__device__ __forceinline__ void warp_max_idx(float& v, int& idx) {
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, off);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
}

/// Merge two sorted (top-1, top-2) pairs; both lanes keep the union's top two.
__device__ __forceinline__ void warp_top2(float& b1, float& b2) {
  for (int off = 16; off > 0; off >>= 1) {
    const float o1 = __shfl_xor_sync(0xFFFFFFFFu, b1, off);
    const float o2 = __shfl_xor_sync(0xFFFFFFFFu, b2, off);
    const float n1 = fmaxf(b1, o1);
    const float n2 = fmaxf(fminf(b1, o1), fmaxf(b2, o2));
    b1 = n1;
    b2 = n2;
  }
}

/// Order-preserving u32 image of an fp32 value: unsigned comparison of the image
/// is float comparison of the value (-inf < ... < +inf, NaN excluded by the
/// caller), so a warp-wide integer max is a warp-wide score max. kZerosEqual maps
/// -0.0 onto +0.0: the generic body compares scores with `==`, where the two zeros
/// are equal and the lower expert id wins, so the image must not separate them.
template <bool kZerosEqual>
__device__ __forceinline__ unsigned order_key(float v) {
  const unsigned u = __float_as_uint(v);
  unsigned k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  if constexpr (kZerosEqual) k = (k == 0x7FFFFFFFu) ? 0x80000000u : k;
  return k;
}

/// Warp max/min of one u32, one instruction on sm_80+ (redux.sync).
__device__ __forceinline__ unsigned warp_max_u32(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r;
  asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
#else
  for (int off = 16; off > 0; off >>= 1) {
    v = max(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  }
  return v;
#endif
}

__device__ __forceinline__ unsigned warp_min_u32(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r;
  asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
#else
  for (int off = 16; off > 0; off >>= 1) {
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  }
  return v;
#endif
}

/// Score recipe as a compile-time constant (kScore is a kernel template
/// parameter), identical formulas to score_of().
template <RouterScore kScore>
__device__ __forceinline__ float score_fixed(float x) {
  if constexpr (kScore == RouterScore::kSigmoid) {
    return sigmoid(x);
  } else if constexpr (kScore == RouterScore::kSqrtSoftplus) {
    return sqrtf(softplus(x));
  } else {
    return x;
  }
}

/// Preconditions of route_row_e256: E == 256 fp32 logits, k <= 8, and 16 B
/// alignment of the row vector loads. Warp-uniform, so every lane agrees.
__device__ __forceinline__ bool route_e256_ok(const RouterParams& p) {
  return p.num_experts == 256 && p.topk <= 8 && (p.logits_stride & 3) == 0 &&
         (reinterpret_cast<uintptr_t>(p.logits) & 15u) == 0u &&
         (p.correction_bias == nullptr ||
          (reinterpret_cast<uintptr_t>(p.correction_bias) & 15u) == 0u);
}

/// Specialized body for E == 256 fp32 logits and topk <= 8: two float4 vectors
/// per lane (experts 4*lane..+3 and 128+4*lane..+3, i.e. two 128 B transactions
/// per warp) instead of eight strided scalar loads. The eight candidates live in
/// registers as (score, 255 - expert) 64-bit keys sorted once by a bitonic
/// network, so each rank costs one warp max plus one warp min (both redux.sync)
/// and the removal is a register shift. Semantics match route_row exactly:
/// NaN -> -inf, softcap, correction bias for selection only, group-limited
/// routing, stable "lower expert id wins" tie-break, weight from the unbiased
/// score, softmax-after-top-k, renormalize, EPLB mapping and fused shared
/// experts. kSoftmax normalizes over all E with a warp reduction whose
/// association order is part of the fused-router bit-equality contract, so that
/// recipe stays on route_row.
template <RouterScore kScore, bool kGrouped>
__device__ __forceinline__ void route_row_e256(const RouterParams& p, const float* row, int token,
                                               int32_t* ids_out, float* weights_out) {
  const int lane = threadIdx.x & 31;
  const int K = p.topk;
  const int E = 256;
  const int e0 = 4 * lane;

  const float4 xa = *reinterpret_cast<const float4*>(row + e0);
  const float4 xb = *reinterpret_cast<const float4*>(row + 128 + e0);
  float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  const float cap = p.logit_softcap;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    xv[j] = (xv[j] == xv[j]) ? xv[j] : -INFINITY;  // NaN -> -inf
  }
  if (cap > 0.0f) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) xv[j] = softcap(xv[j], cap);
  }

  float wv[8];  // unbiased weight source
  float sv[8];  // selection score (bias applied)
  if (p.correction_bias != nullptr) {
    const float4 ba = *reinterpret_cast<const float4*>(p.correction_bias + e0);
    const float4 bb = *reinterpret_cast<const float4*>(p.correction_bias + 128 + e0);
    const float bv[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      wv[j] = score_fixed<kScore>(xv[j]);
      sv[j] = wv[j] + bv[j];
    }
  } else {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      wv[j] = score_fixed<kScore>(xv[j]);
      sv[j] = wv[j];
    }
  }

  // Group-limited routing: keep topk_groups groups by group score; the survivors
  // are applied once, as -inf masks, so the rank loop needs no group test.
  if constexpr (kGrouped) {
    const int G = p.num_expert_groups;
    const int keep = p.topk_groups < G ? p.topk_groups : G;
    if (keep < G) {
      float gscore[kMaxGroups];
      for (int g = 0; g < G; ++g) {
        float b1 = -INFINITY, b2 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int e = e0 + (j < 4 ? j : 124 + j);
          const int eg = (e * G) / E;
          if (eg != g) continue;
          const float v = sv[j];
          if (v > b1) {
            b2 = b1;
            b1 = v;
          } else if (v > b2) {
            b2 = v;
          }
        }
        warp_top2(b1, b2);
        gscore[g] = (p.group_score == GroupScore::kMax) ? b1 : (b1 + b2);
      }
      uint32_t kept = 0;
      for (int r = 0; r < keep; ++r) {
        float best = -INFINITY;
        int bg = INT32_MAX;
        for (int g = 0; g < G; ++g) {
          if (kept & (1u << g)) continue;
          if (gscore[g] > best || (gscore[g] == best && g < bg)) {
            best = gscore[g];
            bg = g;
          }
        }
        kept |= (1u << bg);
      }
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int e = e0 + (j < 4 ? j : 124 + j);
        if ((kept & (1u << ((e * G) / E))) == 0u) sv[j] = -INFINITY;
      }
    }
  }

  // (score, 255 - expert) descending: a larger key is a better expert, and equal
  // scores order by lower expert id, so the key carries the stable tie-break. Only
  // raw-logit recipes can produce -0.0, which the generic body treats as +0.0.
  constexpr bool kZerosEqual =
      kScore == RouterScore::kNone || kScore == RouterScore::kSoftmaxAfterTopK;
  unsigned long long key[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int e = e0 + (j < 4 ? j : 124 + j);
    key[j] = (static_cast<unsigned long long>(order_key<kZerosEqual>(sv[j])) << 32) |
             static_cast<unsigned>(255 - e);
  }
  // Batcher odd-even merge network for eight keys: 19 compare-exchanges, fully
  // in registers, no local memory.
  const auto ce = [&key](int a, int b) {
    const unsigned long long ka = key[a];
    const unsigned long long kb = key[b];
    const bool sw = ka < kb;
    key[a] = sw ? kb : ka;
    key[b] = sw ? ka : kb;
  };
  ce(0, 1); ce(2, 3); ce(4, 5); ce(6, 7);
  ce(0, 2); ce(1, 3); ce(4, 6); ce(5, 7);
  ce(0, 4); ce(1, 5); ce(2, 6); ce(3, 7);
  ce(2, 4); ce(3, 5);
  ce(1, 2); ce(3, 4); ce(5, 6);
  ce(2, 3); ce(4, 5);

  float sel[8] = {};
  int sel_id[8] = {};
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    if (r >= K) break;
    const unsigned head_key = static_cast<unsigned>(key[0] >> 32);
    const int head_id = 255 - static_cast<int>(key[0] & 0xFFu);
    const unsigned gkey = warp_max_u32(head_key);
    // Lowest expert id among the lanes holding the winning score.
    const int gid = static_cast<int>(warp_min_u32(
        head_key == gkey ? static_cast<unsigned>(head_id) : 0x7FFFFFFFu));
    const int rel = gid & 127;
    const int owner = rel >> 2;
    const int q = (gid & 128) ? 4 + (rel & 3) : (rel & 3);
    float wsel = wv[0];
    #pragma unroll
    for (int j = 1; j < 8; ++j) wsel = (j == q) ? wv[j] : wsel;
    sel[r] = __shfl_sync(0xFFFFFFFFu, wsel, owner);
    sel_id[r] = gid;
    if (head_id == gid) {  // this lane owns the winner: drop it
      #pragma unroll
      for (int j = 0; j < 7; ++j) key[j] = key[j + 1];
      key[7] = 0ull;
    }
  }

  if constexpr (kScore == RouterScore::kSoftmaxAfterTopK) {
    float m = -INFINITY;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r < K) m = fmaxf(m, sel[r]);
    }
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r < K) {
        sel[r] = m == -INFINITY ? 0.0f : __expf(sel[r] - m);
        sum += sel[r];
      }
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r < K) sel[r] *= inv;
    }
  }
  if (p.renormalize) {
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r < K) sum += sel[r];
    }
    if (sum > 0.0f) {
      const float inv = 1.0f / sum;
      #pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (r < K) sel[r] *= inv;
      }
    }
  }

  // One rank per lane: rank r is written by lane r.
  if (lane < K) {
    const int r = lane;
    const float w = (r == 0) ? sel[0] : (r == 1) ? sel[1] : (r == 2) ? sel[2] : (r == 3) ? sel[3]
                    : (r == 4) ? sel[4] : (r == 5) ? sel[5] : (r == 6) ? sel[6] : sel[7];
    int id = (r == 0) ? sel_id[0] : (r == 1) ? sel_id[1] : (r == 2) ? sel_id[2] : (r == 3) ? sel_id[3]
                  : (r == 4) ? sel_id[4] : (r == 5) ? sel_id[5] : (r == 6) ? sel_id[6] : sel_id[7];
    const int logical = id;
    if (p.logical_to_physical != nullptr && id >= 0 && id < E) {
      const int count = p.replica_count != nullptr ? p.replica_count[id] : 0;
      if (count > 0 && p.max_replicas > 0) {
        const int replica = token % count;
        const int mapped = p.logical_to_physical[id * p.max_replicas + replica];
        if (mapped >= 0) id = mapped;
      }
    }
    weights_out[r] = w * p.routed_scaling_factor;
    ids_out[r] = id;
    if (p.expert_load != nullptr && logical >= 0 && logical < E) {
      atomicAdd(&p.expert_load[logical], 1);
    }
  }
  if (lane == 0) {
    for (int s = 0; s < p.num_fused_shared; ++s) {
      weights_out[K + s] = p.shared_weight;
      ids_out[K + s] = E + s;
    }
  }
}

/// Full routing body for one token. `row` points at the fp32 or 16-bit logits of
/// the token (stride in elements), `score` is a RouterScore value.
template <typename logits_t, int MAXV>
__device__ __forceinline__ void route_row(const RouterParams& p, const logits_t* row,
                                          int64_t stride, int token, int score,
                                          int32_t* ids_out, float* weights_out) {
  const int lane = threadIdx.x & 31;
  const int E = p.num_experts;
  const int K = p.topk;
  const int G = p.num_expert_groups > 1 ? p.num_expert_groups : 1;
  const int out_k = K + p.num_fused_shared;
  const int n_per_lane = (E + 31) / 32;

  float wv[MAXV];   // unbiased weight source
  float sv[MAXV];   // selection score (bias applied)
  int idxs[MAXV];
  #pragma unroll
  for (int j = 0; j < MAXV; ++j) {
    const int e = lane + 32 * j;
    idxs[j] = e;
    if (e < E) {
      float x = static_cast<float>(row[static_cast<int64_t>(e) * stride]);
      if (!isfinite(x) && !isinf(x)) x = -INFINITY;  // NaN -> -inf
      x = softcap(x, p.logit_softcap);
      wv[j] = x;
      sv[j] = x;
    } else {
      wv[j] = -INFINITY;
      sv[j] = -INFINITY;
    }
  }

  // Softmax recipes normalize over all experts before selection.
  if (score == static_cast<int>(RouterScore::kSoftmax)) {
    float m = -INFINITY;
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int e = idxs[j];
      if (e < E) {
        float x = wv[j];
        m = fmaxf(m, x);
      }
    }
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 16));
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 8));
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
    float sum = 0.0f;
    int inf_count = 0;
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int e = idxs[j];
      if (e < E) {
        const float x = wv[j];
        if (isinf(x) && x > 0.0f) ++inf_count;
      }
    }
    inf_count += __shfl_xor_sync(0xFFFFFFFFu, inf_count, 16);
    inf_count += __shfl_xor_sync(0xFFFFFFFFu, inf_count, 8);
    inf_count += __shfl_xor_sync(0xFFFFFFFFu, inf_count, 4);
    inf_count += __shfl_xor_sync(0xFFFFFFFFu, inf_count, 2);
    inf_count += __shfl_xor_sync(0xFFFFFFFFu, inf_count, 1);
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int e = idxs[j];
      if (e >= E) continue;
      const float x = wv[j];
      float pv;
      if (inf_count > 0) {
        pv = (isinf(x) && x > 0.0f) ? 1.0f / static_cast<float>(inf_count) : 0.0f;
      } else if (m == -INFINITY) {
        pv = 0.0f;
      } else {
        pv = __expf(x - m);
        sum += pv;
      }
      wv[j] = pv;
    }
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      if (idxs[j] >= E) continue;
      if (inf_count > 0 || m == -INFINITY) {
        wv[j] = inf_count > 0 ? wv[j] : 0.0f;
      } else {
        wv[j] *= inv;
      }
      sv[j] = wv[j] + load_bias(p, idxs[j]);
    }
  } else {
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      if (idxs[j] >= E) continue;
      wv[j] = score_of(wv[j], score);
      sv[j] = wv[j] + load_bias(p, idxs[j]);
    }
  }

  // Group-limited routing: keep topk_groups by group score.
  uint32_t kept_groups = 0xFFFFFFFFu;
  if (G > 1) {
    float gscore[kMaxGroups];
    for (int g = 0; g < G; ++g) {
      float b1 = -INFINITY, b2 = -INFINITY;
      #pragma unroll
      for (int j = 0; j < MAXV; ++j) {
        const int e = idxs[j];
        if (e >= E) continue;
        const int eg = (e * G) / E;
        if (eg != g) continue;
        const float v = sv[j];
        if (v > b1) {
          b2 = b1;
          b1 = v;
        } else if (v > b2) {
          b2 = v;
        }
      }
      warp_top2(b1, b2);
      gscore[g] = (p.group_score == GroupScore::kMax) ? b1 : (b1 + b2);
    }
    int keep = p.topk_groups;
    if (keep > G) keep = G;
    if (keep < G) {
      kept_groups = 0;
      for (int r = 0; r < keep; ++r) {
        float best = -INFINITY;
        int bg = INT32_MAX;
        for (int g = 0; g < G; ++g) {
          if (kept_groups & (1u << g)) continue;
          if (gscore[g] > best || (gscore[g] == best && g < bg)) {
            best = gscore[g];
            bg = g;
          }
        }
        kept_groups |= (1u << bg);
      }
    }
  }

  // Stable selection: lower expert index wins on ties.
  float sel[kMaxTopK];
  int sel_ids[kMaxTopK];
  for (int r = 0; r < K; ++r) {
    float best = -INFINITY;
    int bidx = INT32_MAX;
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int e = idxs[j];
      if (e >= E) continue;
      if (G > 1 && (kept_groups & (1u << ((e * G) / E))) == 0u) continue;
      const float v = sv[j];
      if (v > best || (v == best && e < bidx)) {
        best = v;
        bidx = e;
      }
    }
    warp_max_idx(best, bidx);
    // The owner lane of bidx has idxs[j] == bidx; broadcast its raw weight.
    float mine = 0.0f;
    float mine_raw = 0.0f;
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      if (idxs[j] == bidx) {
        mine = sv[j];
        mine_raw = wv[j];
      }
    }
    const int owner = bidx & 31;
    float w = __shfl_sync(0xFFFFFFFFu, mine_raw, owner);
    (void)mine;
    if (lane == 0) {
      sel[r] = w;
      sel_ids[r] = bidx;
    }
    #pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      if (idxs[j] == bidx) {
        sv[j] = -INFINITY;
        wv[j] = -INFINITY;
      }
    }
  }

  if (lane != 0) return;

  if (score == static_cast<int>(RouterScore::kSoftmaxAfterTopK)) {
    float m = -INFINITY;
    for (int r = 0; r < K; ++r) m = fmaxf(m, sel[r]);
    float sum = 0.0f;
    for (int r = 0; r < K; ++r) {
      sel[r] = m == -INFINITY ? 0.0f : __expf(sel[r] - m);
      sum += sel[r];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int r = 0; r < K; ++r) sel[r] *= inv;
  }
  if (p.renormalize) {
    float sum = 0.0f;
    for (int r = 0; r < K; ++r) sum += sel[r];
    if (sum > 0.0f) {
      const float inv = 1.0f / sum;
      for (int r = 0; r < K; ++r) sel[r] *= inv;
    }
  }
  for (int r = 0; r < K; ++r) {
    float w = sel[r] * p.routed_scaling_factor;
    int id = sel_ids[r];
    const int logical = id;
    if (p.logical_to_physical != nullptr && id >= 0 && id < E) {
      const int count = p.replica_count != nullptr ? p.replica_count[id] : 0;
      if (count > 0 && p.max_replicas > 0) {
        const int replica = token % count;
        const int mapped = p.logical_to_physical[id * p.max_replicas + replica];
        if (mapped >= 0) id = mapped;
      }
    }
    weights_out[r] = w;
    ids_out[r] = id;
    if (p.expert_load != nullptr && logical >= 0 && logical < E) {
      atomicAdd(&p.expert_load[logical], 1);
    }
  }
  for (int s = 0; s < p.num_fused_shared; ++s) {
    weights_out[K + s] = p.shared_weight;
    ids_out[K + s] = E + s;
  }
  (void)out_k;
}

}  // namespace

// ---------------------------------------------------------------------------
// Declared kernels (kept outside the anonymous namespace so they match ops.h)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Flat / grouped top-k
// ---------------------------------------------------------------------------

template <typename logits_t, RouterScore kScore, int NUM_EXPERTS, int MAX_TOPK>
__global__ void moe_topk_kernel(const __grid_constant__ RouterParams params) {
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * (blockDim.x >> 5) + warp;
  if (token >= params.num_tokens) return;
  const auto* row = static_cast<const logits_t*>(params.logits) +
                    static_cast<int64_t>(token) * params.logits_stride;
  const int out_k = params.topk + params.num_fused_shared;
  constexpr int kMaxV = (NUM_EXPERTS > 0 && NUM_EXPERTS <= 256) ? 8 : 32;
  int32_t* ids = params.topk_ids + static_cast<int64_t>(token) * out_k;
  float* weights = params.topk_weights + static_cast<int64_t>(token) * out_k;
  if constexpr (std::is_same<logits_t, float>::value && NUM_EXPERTS == 256 &&
                kScore != RouterScore::kSoftmax) {
    if (route_e256_ok(params)) {
      route_row_e256<kScore, false>(params, reinterpret_cast<const float*>(row), token, ids,
                                    weights);
      return;
    }
  }
  route_row<logits_t, kMaxV>(params, row, 1, token, static_cast<int>(kScore), ids, weights);
}

template <typename logits_t, RouterScore kScore, int NUM_EXPERTS, int NUM_GROUPS, int MAX_TOPK>
__global__ void moe_grouped_topk_kernel(const __grid_constant__ RouterParams params) {
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * (blockDim.x >> 5) + warp;
  if (token >= params.num_tokens) return;
  const auto* row = static_cast<const logits_t*>(params.logits) +
                    static_cast<int64_t>(token) * params.logits_stride;
  const int out_k = params.topk + params.num_fused_shared;
  constexpr int kMaxV = (NUM_EXPERTS > 0 && NUM_EXPERTS <= 256) ? 8 : 32;
  int32_t* ids = params.topk_ids + static_cast<int64_t>(token) * out_k;
  float* weights = params.topk_weights + static_cast<int64_t>(token) * out_k;
  if constexpr (std::is_same<logits_t, float>::value && NUM_EXPERTS == 256 &&
                kScore != RouterScore::kSoftmax) {
    if (route_e256_ok(params)) {
      route_row_e256<kScore, true>(params, reinterpret_cast<const float*>(row), token, ids, weights);
      return;
    }
  }
  route_row<logits_t, kMaxV>(params, row, 1, token, static_cast<int>(kScore), ids, weights);
}

// ---------------------------------------------------------------------------
// Fused router GEMM + top-k (num_tokens <= 64)
// ---------------------------------------------------------------------------

template <typename w_t, int VEC>
__device__ __forceinline__ void load_w(const w_t* ptr, float* out) {
  if constexpr (std::is_same<w_t, float>::value) {
    const float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
  } else if constexpr (VEC == 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    float tmp[8];
    dev::unpack8<w_t>(raw, tmp);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = tmp[i];
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    const float2 a = dev::unpack2<w_t>(raw.x);
    const float2 b = dev::unpack2<w_t>(raw.y);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
  }
}

template <typename x_t, int VEC>
__device__ __forceinline__ void load_x(const x_t* ptr, float* out) {
  if constexpr (VEC == 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    float tmp[8];
    dev::unpack8<x_t>(raw, tmp);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = tmp[i];
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    const float2 a = dev::unpack2<x_t>(raw.x);
    const float2 b = dev::unpack2<x_t>(raw.y);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
  }
}

template <typename x_t, typename w_t, int MAXV>
__device__ __forceinline__ void fused_router_body(
    const RouterParams& params, const x_t* __restrict__ hidden, int64_t hidden_stride,
    const w_t* __restrict__ weight, const float* __restrict__ bias, int hidden_size, int score) {
  extern __shared__ float smem[];
  const int E = params.num_experts;
  float* slogits = smem;                       // [kFusedM][E]


  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int m0 = blockIdx.x * kFusedM;
  const int mt = min(kFusedM, params.num_tokens - m0);
  if (mt <= 0) return;
  constexpr int VEC = std::is_same<w_t, float>::value ? 4 : 8;
  const int hslice = (((hidden_size + kRouterWarps - 1) / kRouterWarps) + VEC - 1) / VEC * VEC;
  const int hbeg = warp * hslice;
  const int hend = min(hidden_size, hbeg + hslice);

  constexpr int CH = 4;
  float (*red)[kFusedM][CH] = reinterpret_cast<float (*)[kFusedM][CH]>(slogits + kFusedM * E);
  for (int e0 = 0; e0 < E; e0 += CH) {
    float acc[kFusedM][CH];
    #pragma unroll
    for (int m = 0; m < kFusedM; ++m) {
      #pragma unroll
      for (int c = 0; c < CH; ++c) acc[m][c] = 0.0f;
    }
    for (int kc = hbeg + lane * VEC; kc < hend; kc += 32 * VEC) {
      float xf[kFusedM][8];
      #pragma unroll
      for (int m = 0; m < kFusedM; ++m) {
        if (m < mt) {
          const x_t* xrow = hidden + static_cast<int64_t>(m0 + m) * hidden_stride;
          load_x<x_t, VEC>(xrow + kc, xf[m]);
        } else {
          #pragma unroll
          for (int i = 0; i < 8; ++i) xf[m][i] = 0.0f;
        }
      }
      #pragma unroll
      for (int c = 0; c < CH; ++c) {
        const int e = e0 + c;
        if (e >= E) continue;
        float wf[8];
        load_w<w_t, VEC>(weight + static_cast<int64_t>(e) * hidden_size + kc, wf);
        #pragma unroll
        for (int m = 0; m < kFusedM; ++m) {
          #pragma unroll
          for (int i = 0; i < VEC; ++i) acc[m][c] = fmaf(xf[m][i], wf[i], acc[m][c]);
        }
      }
    }
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      #pragma unroll
      for (int m = 0; m < kFusedM; ++m) {
        float v = acc[m][c];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) red[warp][m][c] = v;
      }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < mt * CH; idx += blockDim.x) {
      const int m = idx / CH;
      const int c = idx % CH;
      const int e = e0 + c;
      if (e >= E) continue;
      float v = 0.0f;
      #pragma unroll
      for (int w = 0; w < kRouterWarps; ++w) v += red[w][m][c];
      if (bias != nullptr) v += bias[e];
      slogits[m * E + e] = v;
    }
    __syncthreads();
  }

  // One warp per token performs scoring + top-k on the shared fp32 logits.
  if (warp < mt) {
    const int token = m0 + warp;
    const int out_k = params.topk + params.num_fused_shared;
    route_row<float, MAXV>(params, slogits + warp * E, 1, token, score,
                            params.topk_ids + static_cast<int64_t>(token) * out_k,
                            params.topk_weights + static_cast<int64_t>(token) * out_k);
  }
}

template <typename x_t, typename w_t, RouterScore kScore, int NUM_EXPERTS, int NUM_GROUPS,
          int MAX_TOPK>
__global__ void moe_fused_router_topk_kernel(
    const __grid_constant__ RouterParams params, const x_t* __restrict__ hidden,
    const int64_t hidden_stride, const w_t* __restrict__ router_weight,
    const float* __restrict__ router_bias, const int hidden_size) {
  constexpr int kMaxV = (NUM_EXPERTS > 0 && NUM_EXPERTS <= 256) ? 8 : 32;
  fused_router_body<x_t, w_t, kMaxV>(params, hidden, hidden_stride, router_weight, router_bias,
                                     hidden_size, static_cast<int>(kScore));
}

template <RouterScore kScore, int NUM_EXPERTS>
__global__ void moe_router_topk_from_logits_kernel(const __grid_constant__ RouterParams params,
                                                   const float* __restrict__ logits,
                                                   const int64_t logits_stride) {
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * (blockDim.x >> 5) + warp;
  if (token >= params.num_tokens) return;
  const int out_k = params.topk + params.num_fused_shared;
  constexpr int kMaxV = (NUM_EXPERTS > 0 && NUM_EXPERTS <= 256) ? 8 : 32;
  route_row<float, kMaxV>(params, logits + static_cast<int64_t>(token) * logits_stride, 1, token,
                          static_cast<int>(kScore),
                          params.topk_ids + static_cast<int64_t>(token) * out_k,
                          params.topk_weights + static_cast<int64_t>(token) * out_k);
}

// ---------------------------------------------------------------------------
// Hash route
// ---------------------------------------------------------------------------

__device__ __forceinline__ int32_t hash_fallback(int64_t token, int j, int num_experts) {
  uint64_t h = static_cast<uint64_t>(token) * 0x9E3779B97F4A7C15ull +
               static_cast<uint64_t>(j) * 0xD6E8FEB86659FD93ull;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  return static_cast<int32_t>(h % static_cast<uint64_t>(num_experts));
}

__global__ void moe_hash_route_kernel(
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_ids,
    const int64_t* __restrict__ token_ids, const int32_t* __restrict__ hash_table,
    const int32_t* __restrict__ expert_map, const float routed_scaling_factor,
    const int num_tokens, const int num_experts, const int topk, const int64_t vocab_size) {
  const int64_t total = static_cast<int64_t>(num_tokens) * topk;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(i / topk);
    const int j = static_cast<int>(i % topk);
    const int64_t tok = token_ids[t];
    int32_t id;
    if (hash_table != nullptr && tok >= 0 && tok < vocab_size) {
      id = hash_table[tok * topk + j];
    } else {
      id = hash_fallback(tok, j, num_experts);
    }
    if (expert_map != nullptr && id >= 0 && id < num_experts) {
      const int32_t local = expert_map[id];
      if (local >= 0) id = local;
    }
    topk_ids[i] = id;
    topk_weights[i] = routed_scaling_factor / static_cast<float>(topk);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

Status check_router(const RouterParams& p, DType dtype, const char* who, bool need_logits) {
  if (dtype != DType::kFloat32 && dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: logits must be fp32, fp16 or bf16", who);
  }
  if (p.num_tokens < 0) return detail::set_error(Status::kInvalidArgument, "%s: negative token count", who);
  if (p.num_experts < 1 || p.num_experts > kMaxExperts) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_experts %d not in [1, %d]", who,
                             p.num_experts, kMaxExperts);
  }
  if (p.topk < 1 || p.topk > kMaxTopK) {
    return detail::set_error(Status::kUnsupportedShape, "%s: topk %d not in [1, %d]", who, p.topk,
                             kMaxTopK);
  }
  if (p.topk > p.num_experts) {
    return detail::set_error(Status::kInvalidArgument, "%s: topk %d > num_experts %d", who, p.topk,
                             p.num_experts);
  }
  if (p.num_expert_groups < 1 || p.num_expert_groups > kMaxGroups) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_expert_groups %d not in [1, %d]", who,
                             p.num_expert_groups, kMaxGroups);
  }
  if (p.topk_groups < 1 || p.topk_groups > p.num_expert_groups) {
    return detail::set_error(Status::kInvalidArgument, "%s: topk_groups %d out of range", who,
                             p.topk_groups);
  }
  if (p.num_expert_groups > 1) {
    const int per_group = (p.num_experts + p.num_expert_groups - 1) / p.num_expert_groups;
    if (p.topk > p.topk_groups * per_group) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: topk %d exceeds the %d kept groups (%d experts each)", who, p.topk,
                               p.topk_groups, per_group);
    }
  }
  if (p.topk_ids == nullptr || p.topk_weights == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: null output", who);
  }
  if (need_logits && p.num_tokens > 0 && p.logits == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: null logits", who);
  }
  if (need_logits && p.logits_stride < p.num_experts) {
    return detail::set_error(Status::kInvalidArgument, "%s: logits stride %lld < E", who,
                             static_cast<long long>(p.logits_stride));
  }
  if (p.num_fused_shared < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: negative num_fused_shared", who);
  }
  return Status::kSuccess;
}

template <typename logits_t, RouterScore kScore>
Status launch_flat(const RouterParams& p, cudaStream_t s, bool grouped) {
  const int warps = 8;
  const dim3 grid(static_cast<unsigned>((p.num_tokens + warps - 1) / warps));
  if (grid.x == 0) return Status::kSuccess;
  if (p.num_experts <= 256) {
    if (grouped) {
      moe_grouped_topk_kernel<logits_t, kScore, 256, 0, kMaxTopK><<<grid, warps * 32, 0, s>>>(p);
    } else {
      moe_topk_kernel<logits_t, kScore, 256, kMaxTopK><<<grid, warps * 32, 0, s>>>(p);
    }
  } else {
    if (grouped) {
      moe_grouped_topk_kernel<logits_t, kScore, 0, 0, kMaxTopK><<<grid, warps * 32, 0, s>>>(p);
    } else {
      moe_topk_kernel<logits_t, kScore, 0, kMaxTopK><<<grid, warps * 32, 0, s>>>(p);
    }
  }
  return launch_error(grouped ? "moe_grouped_topk_kernel" : "moe_topk_kernel");
}

template <typename logits_t>
Status launch_flat_score(const RouterParams& p, RouterScore score, cudaStream_t s, bool grouped) {
  switch (score) {
    case RouterScore::kSoftmax: return launch_flat<logits_t, RouterScore::kSoftmax>(p, s, grouped);
    case RouterScore::kSoftmaxAfterTopK:
      return launch_flat<logits_t, RouterScore::kSoftmaxAfterTopK>(p, s, grouped);
    case RouterScore::kSigmoid: return launch_flat<logits_t, RouterScore::kSigmoid>(p, s, grouped);
    case RouterScore::kSqrtSoftplus:
      return launch_flat<logits_t, RouterScore::kSqrtSoftplus>(p, s, grouped);
    default: return launch_flat<logits_t, RouterScore::kNone>(p, s, grouped);
  }
}

template <RouterScore kScore>
Status launch_topk_from_logits(const RouterParams& p, const float* logits, int64_t stride,
                               cudaStream_t s) {
  const int warps = 8;
  const dim3 grid(static_cast<unsigned>((p.num_tokens + warps - 1) / warps));
  if (p.num_experts <= 256) {
    moe_router_topk_from_logits_kernel<kScore, 256><<<grid, warps * 32, 0, s>>>(p, logits, stride);
  } else {
    moe_router_topk_from_logits_kernel<kScore, 0><<<grid, warps * 32, 0, s>>>(p, logits, stride);
  }
  return launch_error("moe_router_topk_from_logits_kernel");
}

template <typename x_t, typename w_t, RouterScore kScore>
Status launch_fused_score(const RouterParams& p, const Tensor& hidden, const Tensor& weight,
                          const Tensor& bias, int hidden_size, cudaStream_t s) {
  const int smem = kFusedM * p.num_experts * static_cast<int>(sizeof(float)) +
                   kRouterWarps * kFusedM * 4 * static_cast<int>(sizeof(float));
  const dim3 grid(static_cast<unsigned>((p.num_tokens + kFusedM - 1) / kFusedM));
  const float* b = bias.data != nullptr ? static_cast<const float*>(bias.data) : nullptr;
  if (p.num_experts <= 256) {
    moe_fused_router_topk_kernel<x_t, w_t, kScore, 256, 0, kMaxTopK>
        <<<grid, kRouterWarps * 32, smem, s>>>(p, static_cast<const x_t*>(hidden.data),
                                               hidden.stride[0], static_cast<const w_t*>(weight.data),
                                               b, hidden_size);
  } else {
    moe_fused_router_topk_kernel<x_t, w_t, kScore, 0, 0, kMaxTopK>
        <<<grid, kRouterWarps * 32, smem, s>>>(p, static_cast<const x_t*>(hidden.data),
                                               hidden.stride[0], static_cast<const w_t*>(weight.data),
                                               b, hidden_size);
  }
  return launch_error("moe_fused_router_topk_kernel");
}

template <typename x_t, typename w_t>
Status launch_fused(const RouterParams& p, const Tensor& hidden, const Tensor& weight,
                    const Tensor& bias, int hidden_size, RouterScore score, cudaStream_t s) {
  switch (score) {
    case RouterScore::kSoftmax:
      return launch_fused_score<x_t, w_t, RouterScore::kSoftmax>(p, hidden, weight, bias, hidden_size, s);
    case RouterScore::kSoftmaxAfterTopK:
      return launch_fused_score<x_t, w_t, RouterScore::kSoftmaxAfterTopK>(p, hidden, weight, bias,
                                                                         hidden_size, s);
    case RouterScore::kSigmoid:
      return launch_fused_score<x_t, w_t, RouterScore::kSigmoid>(p, hidden, weight, bias, hidden_size, s);
    case RouterScore::kSqrtSoftplus:
      return launch_fused_score<x_t, w_t, RouterScore::kSqrtSoftplus>(p, hidden, weight, bias,
                                                                     hidden_size, s);
    default:
      return launch_fused_score<x_t, w_t, RouterScore::kNone>(p, hidden, weight, bias, hidden_size, s);
  }
}

}  // namespace

Status route(const RouterParams& params, RouterScore score, DType logits_dtype,
             const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "route: CUDA backend only");
  }
  Status st = check_router(params, logits_dtype, "route", true);
  if (st != Status::kSuccess) return st;
  if (params.num_tokens == 0) return Status::kSuccess;
  if (params.logits_stride < 0) {
    return detail::set_error(Status::kInvalidArgument, "route: negative logits stride");
  }
  const bool grouped = params.num_expert_groups > 1;
  const cudaStream_t s = stream_of(ctx);
  switch (logits_dtype) {
    case DType::kFloat32:
      return launch_flat_score<float>(params, score, s, grouped);
    case DType::kFloat16:
      return launch_flat_score<__half>(params, score, s, grouped);
    case DType::kBFloat16:
      return launch_flat_score<__nv_bfloat16>(params, score, s, grouped);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "route: %s", dtype_name(logits_dtype));
  }
}

Status route_fused(const RouterParams& params, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor& router_bias, RouterScore score, DType w_dtype,
                   const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "route_fused: CUDA backend only");
  }
  if (hidden.dtype != DType::kBFloat16 && hidden.dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "route_fused: hidden must be bf16 or fp16");
  }
  Status st = check_router(params, DType::kFloat32, "route_fused", false);
  if (st != Status::kSuccess) return st;
  if (params.num_tokens > 64) {
    return detail::set_error(Status::kUnsupportedShape, "route_fused: num_tokens %d > 64",
                             params.num_tokens);
  }
  if (hidden.ndim != 2 || hidden.shape[0] != params.num_tokens) {
    return detail::set_error(Status::kInvalidArgument, "route_fused: hidden must be [num_tokens, H]");
  }
  const int hidden_size = static_cast<int>(hidden.shape[1]);
  if (hidden_size < 8 || hidden_size % 8 != 0) {
    return detail::set_error(Status::kUnsupportedShape, "route_fused: hidden must be a multiple of 8");
  }
  if (hidden.stride[1] != 1 || hidden.stride[0] < hidden_size) {
    return detail::set_error(Status::kUnsupportedLayout, "route_fused: hidden must be row-contiguous");
  }
  if (router_weight.ndim != 2 || router_weight.shape[0] != params.num_experts ||
      router_weight.shape[1] != hidden_size) {
    return detail::set_error(Status::kInvalidArgument, "route_fused: router weight must be [E, H]");
  }
  if (router_weight.dtype != w_dtype) {
    return detail::set_error(Status::kInvalidArgument, "route_fused: w_dtype does not match the tensor");
  }
  if (!tensor_is_contiguous(router_weight)) {
    return detail::set_error(Status::kUnsupportedLayout, "route_fused: router weight must be contiguous");
  }
  if (tensor_present(router_bias)) {
    if (router_bias.dtype != DType::kFloat32 || router_bias.shape[0] != params.num_experts) {
      return detail::set_error(Status::kInvalidArgument, "route_fused: router bias must be fp32 [E]");
    }
    if (!aligned(router_bias.data, 4)) {
      return detail::set_error(Status::kMisaligned, "route_fused: router bias must be 4 B aligned");
    }
  }
  if (!aligned(hidden.data, 16) || (hidden.stride[0] * 2) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "route_fused: hidden must be 16 B aligned");
  }
  const int wb = w_dtype == DType::kFloat32 ? 4 : 2;
  if (!aligned(router_weight.data, 16) || (static_cast<int64_t>(hidden_size) * wb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "route_fused: router weights must be 16 B aligned");
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  const cudaStream_t s = stream_of(ctx);
  // Fast path: caller-provided fp32 logits [T, E] let the router use the tiled
  // router_gemm kernel (many CTAs) plus a small top-k kernel instead of the
  // single-CTA fused fallback (which exists to keep the no-scratch contract).
  if (params.logits != nullptr) {
    if (params.logits_stride < params.num_experts) {
      return detail::set_error(Status::kInvalidArgument,
                               "route_fused: logits stride %lld < num_experts %d",
                               static_cast<long long>(params.logits_stride), params.num_experts);
    }
    if (!aligned(params.logits, 4)) {
      return detail::set_error(Status::kMisaligned, "route_fused: logits scratch must be 4 B aligned");
    }
    gemm::RouterGemmParams gp{};
    gp.x = hidden.data;
    gp.ldx = hidden.stride[0];
    gp.w = router_weight.data;
    gp.bias = tensor_present(router_bias) ? static_cast<const float*>(router_bias.data) : nullptr;
    gp.logits = static_cast<float*>(const_cast<void*>(params.logits));
    gp.ld_logits = params.logits_stride;
    gp.m = params.num_tokens;
    gp.num_experts = params.num_experts;
    gp.hidden = hidden_size;
    gp.lo_scale = 1.0f / 256.0f;
    const Status st2 = gemm::router_gemm(gp, false, hidden.dtype, w_dtype, ctx);
    if (st2 != Status::kSuccess) return st2;
    auto* logits = static_cast<const float*>(params.logits);
    switch (score) {
      case RouterScore::kSoftmax:
        return launch_topk_from_logits<RouterScore::kSoftmax>(params, logits, params.logits_stride, s);
      case RouterScore::kSoftmaxAfterTopK:
        return launch_topk_from_logits<RouterScore::kSoftmaxAfterTopK>(params, logits,
                                                                      params.logits_stride, s);
      case RouterScore::kSigmoid:
        return launch_topk_from_logits<RouterScore::kSigmoid>(params, logits, params.logits_stride, s);
      case RouterScore::kSqrtSoftplus:
        return launch_topk_from_logits<RouterScore::kSqrtSoftplus>(params, logits, params.logits_stride, s);
      default:
        return launch_topk_from_logits<RouterScore::kNone>(params, logits, params.logits_stride, s);
    }
  }
  if (hidden.dtype == DType::kBFloat16 && w_dtype == DType::kBFloat16) {
    return launch_fused<__nv_bfloat16, __nv_bfloat16>(params, hidden, router_weight, router_bias,
                                                      hidden_size, score, s);
  }
  if (hidden.dtype == DType::kBFloat16 && w_dtype == DType::kFloat32) {
    return launch_fused<__nv_bfloat16, float>(params, hidden, router_weight, router_bias, hidden_size,
                                              score, s);
  }
  if (hidden.dtype == DType::kFloat16 && w_dtype == DType::kFloat16) {
    return launch_fused<__half, __half>(params, hidden, router_weight, router_bias, hidden_size, score,
                                        s);
  }
  return detail::set_error(Status::kUnsupportedDtype,
                           "route_fused: supported pairs are (bf16,bf16), (bf16,fp32), (fp16,fp16)");
}

Status hash_route(const Tensor& topk_weights, const Tensor& topk_ids, const Tensor& token_ids,
                  const Tensor& hash_table, const Tensor& expert_map, float routed_scaling_factor,
                  int32_t num_experts, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "hash_route: CUDA backend only");
  }
  Status s = check_tensor(topk_weights, DType::kFloat32, 2, 4, "hash_route.topk_weights");
  if (s != Status::kSuccess) return s;
  s = check_tensor(topk_ids, DType::kInt32, 2, 4, "hash_route.topk_ids");
  if (s != Status::kSuccess) return s;
  s = check_tensor(token_ids, DType::kInt64, 1, 8, "hash_route.token_ids");
  if (s != Status::kSuccess) return s;
  const int num_tokens = static_cast<int>(topk_ids.shape[0]);
  const int topk = static_cast<int>(topk_ids.shape[1]);
  if (topk_weights.shape[0] != num_tokens || topk_weights.shape[1] != topk) {
    return detail::set_error(Status::kInvalidArgument, "hash_route: topk output shapes differ");
  }
  if (token_ids.shape[0] != num_tokens) {
    return detail::set_error(Status::kInvalidArgument, "hash_route: token_ids length mismatch");
  }
  if (num_experts < 1 || topk < 1 || topk > num_experts) {
    return detail::set_error(Status::kInvalidArgument, "hash_route: bad num_experts/topk");
  }
  if (topk_ids.stride[1] != 1 || topk_weights.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "hash_route: rows must be contiguous");
  }
  int64_t vocab = 0;
  if (tensor_present(hash_table)) {
    s = check_tensor(hash_table, DType::kInt32, 2, 4, "hash_route.hash_table");
    if (s != Status::kSuccess) return s;
    if (hash_table.shape[1] != topk) {
      return detail::set_error(Status::kInvalidArgument, "hash_route: hash_table must be [V, topk]");
    }
    vocab = hash_table.shape[0];
  }
  if (tensor_present(expert_map)) {
    s = check_tensor(expert_map, DType::kInt32, 1, 4, "hash_route.expert_map");
    if (s != Status::kSuccess) return s;
    if (expert_map.shape[0] != num_experts) {
      return detail::set_error(Status::kInvalidArgument, "hash_route: expert_map must be [E]");
    }
  }
  if (num_tokens == 0) return Status::kSuccess;
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(num_tokens) * topk;
  const dim3 grid(static_cast<unsigned>((total + threads - 1) / threads));
  moe_hash_route_kernel<<<grid, threads, 0, stream_of(ctx)>>>(
      static_cast<float*>(topk_weights.data), static_cast<int32_t*>(topk_ids.data),
      static_cast<const int64_t*>(token_ids.data),
      tensor_present(hash_table) ? static_cast<const int32_t*>(hash_table.data) : nullptr,
      tensor_present(expert_map) ? static_cast<const int32_t*>(expert_map.data) : nullptr,
      routed_scaling_factor, num_tokens, num_experts, topk, vocab);
  return launch_error("moe_hash_route_kernel");
}

}  // namespace kops::moe
