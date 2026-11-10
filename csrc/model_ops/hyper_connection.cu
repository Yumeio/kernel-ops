// =============================================================================
// kernel-ops | csrc/model_ops/hyper_connection.cu                       X01 (B15)
// Hyper-Connections / manifold-constrained Hyper-Connections host launchers and
// the generic (T1, SM80-class) device bodies:
//   hc_expand, hc_head        stream initialization / collapse
//   hc_post                   new_residual = comb_res_mix^T residual + post_mix (x) y
//   hc_coefficients           RMSNorm(vec X) + skinny projection -> pre/post/comb
//   hc_pre                    layer input = pre^T new_residual (+ RMSNorm / FP8)
//   mega_mhc                  post + coefficients + pre in one launch
//
// Semantics are the official DeepSeek-V4 contract, transcribed from
//   3rd/DeepGEMM/tests/test_mega_mhc.py::mhc_reference and ::__sinkhorn_reference
// and 3rd/DeepGEMM/csrc/apis/mega_mhc.hpp.
//
// Numerics
//   * Every reduction and the projection accumulate in fp32. The reference runs
//     the projection with TF32 disabled; this body is more accurate than the
//     TF32 DeepGEMM kernel, so the row tolerance follows the fp32 reference.
//   * new_residual is rounded to the stream dtype ONCE (bf16/fp16/fp32) before it
//     is used by the projection and the pre norm, exactly like the reference's
//     `new_residual.bfloat16()`.
//   * `mix_scales` are replicated (n for pre, n for post, n*n for comb) and used
//     as multipliers; this matches `mix_scales[i].expand(...)` in the reference
//     and `mix_args.scales[idx]` in the DeepGEMM epilogue. The RMS scale divides
//     by n*H: rsqrt(sum_sq/(n*H) + hc_norm_eps).
//   * Sinkhorn = softmax(row) + sinkhorn_eps, one column normalization, then
//     num_sinkhorn_iters-1 x (row normalization, column normalization); every
//     denominator adds sinkhorn_eps.
//   * shifted mode replaces the pre-weighted mix that feeds the RMSNorm with
//     sum_i shifted_prev_mix[t,i] * new_residual[t,i,:]; the new pre coefficients
//     are still predicted and written.
//
// Launch shape / target
//   hc_post, hc_expand, hc_head are flat, grid-stride, 16 B pack kernels.
//   hc_coeff (small-T / no scratch), hc_pre and mega_mhc (T < kTileMinTokens) use one CTA
//   per token with 8 warps: the projection (K = n*H, N = n(n+2)), the Sinkhorn constraint and
//   the RMSNorm are token-local, so no cross-CTA barrier or caller workspace is needed.
//   The large-T coefficient body (T >= kTileMinTokens = 256) is one CTA per 16 tokens (m16):
//   the mma.sync tf32 projection of hc_tc_project reuses the [n(n+2), nH] predictor across
//   the 16 rows, writes the raw dots to `raw_logits` (+ `row_sumsq`) and applies the
//   constraint per row in the same launch. mega_mhc with the same shape fuses post + that
//   projection + constraint + pre for its own 16 rows, so the chain and mega bodies keep
//   bitwise parity (both pick the tile body for the same shape) and stay graph-safe: no host
//   sync, no allocation and no cross-CTA handoff. The threshold is measured: one tile CTA
//   streams the whole predictor, so with fewer than 16 tiles the one-CTA-per-token body has
//   more concurrent predictor readers and stays faster (T = 64: 0.165 ms vs 0.171 ms).
//   `raw_logits` alone selects the DeepGEMM contract (the caller ran the projection) and
//   never enables the tile body.
//   TODO(P7 split body): a cross-CTA split-K body with `exec::last_cta_arrive` needs a
//   caller-owned epoch-tagged barrier region; the public host API has no Workspace argument
//   (ops.h), so the tile body keeps K inside one CTA per 16 rows.
//   The bodies are arch-generic apart from the SM80 `mma.sync` instruction (guarded in
//   hc_mma_tf32, and the tile bodies are only selected on SM80+); exact-target selection
//   belongs to the registry.
//
// Limitations of this slice (documented, not silent):
//   * n_hc in {1, 2, 4}.
//   * hc_expand / hc_head need packed streams (stream_token_stride == n_hc*d) and
//     contiguous embed/out rows; hc_post accepts any token/route stride.
//   * hc_coefficients / hc_pre / mega_mhc require 16 B-aligned pointers and d and
//     token/route strides divisible by the 16 B pack of the stream dtype.
//   * ActQuantOut supports kNone and FP8 E4M3 per-group (kPerGroup / kMXFP8,
//     group 32 or 128, UE8M0 scales packed 4-per-int32 or one per group). Other
//     modes are rejected with kUnsupportedDtype. Quantized output requires
//     d % 128 == 0.
//   * the row stage for the pre norm is fp32 in dynamic shared memory when
//     d * 4 B <= 48 KB; wider rows take a second global read instead.
//
// Reference: semantics from DeepGEMM (Apache-2.0), see THIRD_PARTY_NOTICES.md.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>

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
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace kops::model_ops {
namespace {

using dev::from_f;
using dev::Pack;
using dev::to_f;

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
/// Dynamic fp32 row stage used by the norm/quant pass. 48 KB stays under the
/// no-opt-in shared-memory limit of every supported target; a wider row falls
/// back to a second global pass (same values, more traffic).
constexpr int64_t kStageCapBytes = 48 * 1024;

// ---- scalar helpers ----------------------------------------------------------
template <typename T>
__device__ __forceinline__ float round_stream(float v) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return v;
  } else {
    T t;
    from_f(t, v);
    return to_f(t);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void load_pack_f32(const T* __restrict__ p, float (&dst)[VEC]) {
  if constexpr (VEC == 1) {
    dst[0] = to_f(p[0]);
  } else {
    const Pack<T, VEC> pk = dev::load_pack<VEC>(p);
#pragma unroll
    for (int i = 0; i < VEC; ++i) dst[i] = to_f(pk.x[i]);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void store_pack_f32(T* __restrict__ p, const float (&src)[VEC]) {
  if constexpr (VEC == 1) {
    from_f(p[0], src[0]);
  } else {
    Pack<T, VEC> pk;
#pragma unroll
    for (int i = 0; i < VEC; ++i) from_f(pk.x[i], src[i]);
    dev::store_pack<VEC>(p, pk);
  }
}

/// fp32 ceil(log2(s)) exponent, the DeepGEMM `ceil_to_ue8m0` recipe: the biased
/// exponent, plus one when the mantissa is non-zero, clamped to [1, 254].
__device__ __forceinline__ uint8_t hc_ue8m0_ceil(float s) {
  const uint32_t bits = __float_as_uint(s);
  uint32_t e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
  e = e < 1u ? 1u : (e > 254u ? 254u : e);
  return static_cast<uint8_t>(e);
}

__device__ __forceinline__ float hc_ue8m0_value(uint8_t e) {
  return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

/// Round one fp32 value to the finite E4M3 range (rte), the shared payload rule.
__device__ __forceinline__ uint8_t hc_e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

/// Shared-expert FP8 scale layout of the DeepGEMM `extra` SF case: four 32-row
/// groups of a 128-row page interleaved. `num_rows` is the physical row count.
__device__ __forceinline__ int64_t hc_shared_sf_row(int64_t token, int32_t block_m) {
  const int64_t idx = token % block_m;
  return token / block_m * ((static_cast<int64_t>(block_m) + 127) & ~int64_t(127)) +
         (idx & ~int64_t(127)) + (idx & 31) * 4 + ((idx >> 5) & 3);
}

// ---- coefficient constraint --------------------------------------------------
/// mix[0..N) pre | mix[N..2N) post | mix[2N..N*N+2N) comb (row-major n x n).
/// kManifold applies sigmoid / hc_post_scale / Sinkhorn; kNone copies the raw
/// values through (HC / iHC).
template <int N_HC>
__device__ __forceinline__ void hc_apply_constraint(const float (&mix)[N_HC * (N_HC + 2)],
                                                    const HCParams& p, float (&pre)[N_HC],
                                                    float (&post)[N_HC],
                                                    float (&comb)[N_HC * N_HC]) {
  constexpr int N = N_HC;
  if (p.constraint == HCConstraint::kManifold) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pre[i] = 1.0f / (1.0f + __expf(-mix[i])) + p.hc_pre_eps;
      post[i] = (1.0f / (1.0f + __expf(-mix[N + i]))) * p.hc_post_scale;
    }
    float v[N][N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float m = mix[2 * N + i * N];
#pragma unroll
      for (int j = 1; j < N; ++j) m = fmaxf(m, mix[2 * N + i * N + j]);
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < N; ++j) {
        const float e = __expf(mix[2 * N + i * N + j] - m);
        v[i][j] = e;
        s += e;
      }
      const float inv = 1.0f / s;
#pragma unroll
      for (int j = 0; j < N; ++j) v[i][j] = v[i][j] * inv + p.sinkhorn_eps;
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
      float c = 0.0f;
#pragma unroll
      for (int i = 0; i < N; ++i) c += v[i][j];
      const float inv = 1.0f / (c + p.sinkhorn_eps);
#pragma unroll
      for (int i = 0; i < N; ++i) v[i][j] *= inv;
    }
    for (int64_t it = 1; it < static_cast<int64_t>(p.num_sinkhorn_iters); ++it) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        float r = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) r += v[i][j];
        const float inv = 1.0f / (r + p.sinkhorn_eps);
#pragma unroll
        for (int j = 0; j < N; ++j) v[i][j] *= inv;
      }
#pragma unroll
      for (int j = 0; j < N; ++j) {
        float c = 0.0f;
#pragma unroll
        for (int i = 0; i < N; ++i) c += v[i][j];
        const float inv = 1.0f / (c + p.sinkhorn_eps);
#pragma unroll
        for (int i = 0; i < N; ++i) v[i][j] *= inv;
      }
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
      for (int j = 0; j < N; ++j) comb[i * N + j] = v[i][j];
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pre[i] = mix[i];
      post[i] = mix[N + i];
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
      for (int j = 0; j < N; ++j) comb[i * N + j] = mix[2 * N + i * N + j];
  }
}

/// Scale one projection output: mix = dot * rms_scale * scales[group] + bases.
template <int N_HC>
__device__ __forceinline__ void hc_affine_mix(const float* __restrict__ dot, float rms_scale,
                                              const float* __restrict__ mix_scales,
                                              const float* __restrict__ mix_bases,
                                              float* __restrict__ mix) {
  constexpr int N = N_HC;
#pragma unroll
  for (int i = 0; i < N; ++i) mix[i] = dot[i] * rms_scale * mix_scales[0] + mix_bases[i];
#pragma unroll
  for (int i = 0; i < N; ++i)
    mix[N + i] = dot[N + i] * rms_scale * mix_scales[1] + mix_bases[N + i];
#pragma unroll
  for (int i = 0; i < N * N; ++i)
    mix[2 * N + i] = dot[2 * N + i] * rms_scale * mix_scales[2] + mix_bases[2 * N + i];
}

// ---- tensor-core projection tile ---------------------------------------------
/// m16 of one tile CTA: 16 tokens per block for the projection / coefficient body.
constexpr int kTileRows = 16;
/// Tile body threshold: below this the per-token body has more CTAs (and therefore more
/// predictor-read bandwidth) than the 16-row tile body, so it stays faster.
constexpr int kTileMinTokens = 16 * kTileRows;

/// Split `x` into a tf32-exact high part and its fp32 remainder. The mask keeps the top 11
/// significant bits (10 explicit mantissa bits + the implicit one), which is exactly what
/// tf32 stores, so `hi` survives cvt.rna unchanged and `lo = x - hi` is exact in fp32.
__device__ __forceinline__ void hc_tf32_split(float x, float& hi, float& lo) {
  hi = __uint_as_float(__float_as_uint(x) & 0xFFFFE000u);
  lo = x - hi;
}

__device__ __forceinline__ uint32_t hc_tf32_bits(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
#else
  return __float_as_uint(x);
#endif
}

/// m16n8k8 tf32 x tf32 -> fp32. Lane (g = lane/4, c = lane%4) owns A rows g / g+8 at k = c
/// and c+4, B columns g at k = c and c+4, and C rows g / g+8 at columns 2c and 2c+1.
__device__ __forceinline__ void hc_mma_tf32(float (&c)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/// Shape eligibility shared by both tile bodies: a full 16-token row, whole n8 output tiles
/// (n_hc 2 and 4; n_hc 1 is 3 wide), a `d` that is a multiple of the mma k step (so one k8
/// step never crosses a stream route), and enough rows to fill the machine: one tile CTA
/// streams the whole [n(n+2), n*d] predictor, so below `kTileMinTokens` tiles the per-token
/// body wins (64 rows = 4 CTAs cannot saturate the predictor read).
__host__ __device__ __forceinline__ bool hc_tile_shape_ok(const HCParams& p) {
  const int32_t out = p.n_hc * (p.n_hc + 2);
  return p.fn != nullptr && p.num_tokens >= kTileMinTokens && p.d >= 8 && (p.d % 8) == 0 &&
         out > 0 && (out % 8) == 0;
}

/// hc_coefficients opt-in for the tensor-core body: the caller handed in the projection
/// scratch pair, so `raw_logits` is written (and `row_sumsq`), then the constraint runs.
__host__ __device__ __forceinline__ bool hc_coeff_tile_mode(const HCParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  return false;
#else
  return p.raw_logits != nullptr && p.row_sumsq != nullptr && hc_tile_shape_ok(p);
#endif
}

/// mega_mhc owns its projection, so only the shape decides.
__host__ __device__ __forceinline__ bool hc_mega_tile_mode(const HCParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  return false;
#else
  return hc_tile_shape_ok(p);
#endif
}

/// dot[t][o] = sum_k X[t][k] * fn[o][k] over one 16-token tile, plus the row sums of squares,
/// on tf32 tensor cores. Every warp accumulates all `kOut` outputs of the tile in m16n8k8
/// fragments while splitting K; the CTA then reduces the per-warp partials in a fixed warp
/// order into `logits` ([kTileRows][kOut]) and `sq_out` ([kTileRows]) in shared memory.
///
/// Accuracy: `fn` is fp32, so it is split into hi/lo tf32 halves and every product is
/// accumulated twice (`a*hi + a*lo`). The stream operand is exact in tf32 for bf16/fp16 and
/// is split the same way for fp32. The dropped `lo*lo` term is O(2^-22) of each product, so
/// the result is fp32-class accuracy, not the 2^-11 of a single-pass TF32 GEMM.
///
/// Requires: kWarps warps in the CTA, `warp_acc` = kWarps*kTileRows*kOut floats, `warp_sq` =
/// kWarps*kTileRows floats. Rows past `num_tokens` read the last valid row (their results are
/// never stored). Ends with a CTA barrier.
template <typename stream_t, int N_HC>
__device__ __forceinline__ void hc_tc_project(const stream_t* __restrict__ residual,
                                              int64_t token_stride, int64_t route_stride,
                                              int64_t token_base, int64_t num_tokens,
                                              const float* __restrict__ fn, int64_t d,
                                              float* __restrict__ logits,
                                              float* __restrict__ warp_acc,
                                              float* __restrict__ sq_out,
                                              float* __restrict__ warp_sq) {
  constexpr int N = N_HC;
  constexpr int kOut = N * (N + 2);
  constexpr int kNTiles = kOut >= 8 ? kOut / 8 : 1;
  constexpr bool kSplitA = sizeof(stream_t) == sizeof(float);
  if constexpr (kOut % 8 != 0) return;  // n_hc = 1 (3 outputs): never selected
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int grp = lane >> 2;  // groupID: A row, B column
  const int c = lane & 3;     // threadID_in_group: A/B k, C column pair
  const int64_t K = static_cast<int64_t>(N) * d;
  const int64_t steps = K / 8;
  const int64_t per = (steps + kWarps - 1) / kWarps;
  const int64_t s_begin = static_cast<int64_t>(warp) * per;
  const int64_t s_end = s_begin + per < steps ? s_begin + per : steps;

  float acc[kNTiles][4];
  float sum[kNTiles][4];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[t][i] = 0.0f;
      sum[t][i] = 0.0f;
    }
  float sq0 = 0.0f;
  float sq1 = 0.0f;

  if (s_begin < s_end) {
    const int64_t row_a = token_base + grp < num_tokens ? token_base + grp : num_tokens - 1;
    const int64_t row_b = token_base + grp + 8 < num_tokens ? token_base + grp + 8 : num_tokens - 1;
    int64_t k = s_begin * 8;
    int64_t route = k / d;
    int64_t h = k - route * d;
    const stream_t* pa = residual + row_a * token_stride + route * route_stride + h;
    const stream_t* pb = residual + row_b * token_stride + route * route_stride + h;
    const float* fp = fn + static_cast<int64_t>(grp) * K + k;
    const int64_t tile_stride = 8 * K;

    // One mma k8 step: A rows grp / grp+8 at k = c and c+4, B column grp at k = c and c+4.
    // The tf32 fragment is zeroed every step and folded into the fp32 register sum `sum`:
    // the tensor-core C add is biased (not round-to-nearest), so a long K chain would drift
    // by ~1 ulp of the partial sum per step. One mma's k8 dot is a single fp32 rounding, and
    // the fold is a round-to-nearest fp32 add, which keeps the row at fp32 accuracy.
    for (int64_t s = s_begin; s < s_end; ++s) {
      const float a0 = to_f(pa[c]);
      const float a1 = to_f(pb[c]);
      const float a2 = to_f(pa[c + 4]);
      const float a3 = to_f(pb[c + 4]);
      sq0 = __fmaf_rn(a0, a0, sq0);
      sq0 = __fmaf_rn(a2, a2, sq0);
      sq1 = __fmaf_rn(a1, a1, sq1);
      sq1 = __fmaf_rn(a3, a3, sq1);
      uint32_t ah[4], al[4];
      if constexpr (kSplitA) {
        float hi, lo;
        hc_tf32_split(a0, hi, lo);
        ah[0] = hc_tf32_bits(hi);
        al[0] = hc_tf32_bits(lo);
        hc_tf32_split(a1, hi, lo);
        ah[1] = hc_tf32_bits(hi);
        al[1] = hc_tf32_bits(lo);
        hc_tf32_split(a2, hi, lo);
        ah[2] = hc_tf32_bits(hi);
        al[2] = hc_tf32_bits(lo);
        hc_tf32_split(a3, hi, lo);
        ah[3] = hc_tf32_bits(hi);
        al[3] = hc_tf32_bits(lo);
      } else {
        ah[0] = hc_tf32_bits(a0);
        ah[1] = hc_tf32_bits(a1);
        ah[2] = hc_tf32_bits(a2);
        ah[3] = hc_tf32_bits(a3);
        al[0] = al[1] = al[2] = al[3] = 0u;
      }
      const float* b0p = fp + c;
      const float* b1p = fp + c + 4;
#pragma unroll
      for (int t = 0; t < kNTiles; ++t) {
        float h0, l0, h1, l1;
        hc_tf32_split(b0p[t * tile_stride], h0, l0);
        hc_tf32_split(b1p[t * tile_stride], h1, l1);
        const uint32_t bh[2] = {hc_tf32_bits(h0), hc_tf32_bits(h1)};
        const uint32_t bl[2] = {hc_tf32_bits(l0), hc_tf32_bits(l1)};
        hc_mma_tf32(acc[t], ah, bh);
        hc_mma_tf32(acc[t], ah, bl);
        if constexpr (kSplitA) hc_mma_tf32(acc[t], al, bh);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sum[t][i] += acc[t][i];
          acc[t][i] = 0.0f;
        }
      }
      pa += 8;
      pb += 8;
      fp += 8;
      h += 8;
      if (h == d) {
        h = 0;
        ++route;
        pa += route_stride - d;
        pb += route_stride - d;
      }
    }
  }

  // Per-warp partials -> fixed-order CTA reduction.
  float* wrow = warp_acc + warp * (kTileRows * kOut);
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) {
    const int col = t * 8 + 2 * c;
    wrow[grp * kOut + col] = sum[t][0];
    wrow[grp * kOut + col + 1] = sum[t][1];
    wrow[(grp + 8) * kOut + col] = sum[t][2];
    wrow[(grp + 8) * kOut + col + 1] = sum[t][3];
  }
  sq0 += __shfl_xor_sync(0xffffffffu, sq0, 1);
  sq0 += __shfl_xor_sync(0xffffffffu, sq0, 2);
  sq1 += __shfl_xor_sync(0xffffffffu, sq1, 1);
  sq1 += __shfl_xor_sync(0xffffffffu, sq1, 2);
  if (c == 0) {
    warp_sq[warp * kTileRows + grp] = sq0;
    warp_sq[warp * kTileRows + grp + 8] = sq1;
  }
  __syncthreads();
  constexpr int kTotal = kTileRows * kOut;
  for (int i = static_cast<int>(threadIdx.x); i < kTotal; i += static_cast<int>(blockDim.x)) {
    float s = warp_acc[i];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) s += warp_acc[w * kTotal + i];
    logits[i] = s;
  }
  if (threadIdx.x < kTileRows) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += warp_sq[w * kTileRows + static_cast<int>(threadIdx.x)];
    sq_out[threadIdx.x] = s;
  }
  __syncthreads();
}

// ---- block reductions --------------------------------------------------------
/// Deterministic reduction over the block: warp tree then a fixed warp order.
/// On return every thread sees every total (smem holds (nwarp + 1) * N floats).
template <int N>
__device__ __forceinline__ void hc_block_reduce_all(float (&v)[N], float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarp = (static_cast<int>(blockDim.x) + 31) >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = warp_reduce_sum(v[i]);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) smem[warp * N + i] = v[i];
  }
  __syncthreads();
  if (threadIdx.x < N) {
    float s = 0.0f;
    for (int w = 0; w < nwarp; ++w) s += smem[w * N + static_cast<int>(threadIdx.x)];
    smem[nwarp * N + threadIdx.x] = s;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = smem[nwarp * N + i];
}

// ---- post mixing (shared by hc_post and mega_mhc) ----------------------------
/// One 16 B pack of one token: out[t, j, col..col+VEC) =
///   post[j] * y[t, col..] + sum_i comb[i, j] * residual[t, i, col..]
/// Each thread reads every route before writing any, so the exact in-place view
/// (out == residual) is safe. Returns the sum of squares of the stored values.
template <typename stream_t, typename y_t, int N_HC, int VEC_SIZE>
__device__ __forceinline__ float hc_mix_pack(const stream_t* __restrict__ residual,
                                             int64_t residual_token_stride,
                                             int64_t residual_route_stride,
                                             const y_t* __restrict__ y, int64_t y_stride,
                                             const float* __restrict__ post,
                                             const float* __restrict__ comb,
                                             stream_t* __restrict__ out, int64_t out_token_stride,
                                             int64_t out_route_stride, int64_t token, int64_t col) {
  float yv[VEC_SIZE];
  load_pack_f32<y_t, VEC_SIZE>(y + token * y_stride + col, yv);
  float v[N_HC][VEC_SIZE];
#pragma unroll
  for (int j = 0; j < N_HC; ++j) {
#pragma unroll
    for (int c = 0; c < VEC_SIZE; ++c) v[j][c] = yv[c] * post[j];
  }
  const stream_t* rp = residual + token * residual_token_stride + col;
#pragma unroll
  for (int i = 0; i < N_HC; ++i) {
    float rv[VEC_SIZE];
    load_pack_f32<stream_t, VEC_SIZE>(rp + static_cast<int64_t>(i) * residual_route_stride, rv);
#pragma unroll
    for (int j = 0; j < N_HC; ++j) {
      const float b = comb[i * N_HC + j];
#pragma unroll
      for (int c = 0; c < VEC_SIZE; ++c) v[j][c] += b * rv[c];
    }
  }
  float sq = 0.0f;
  stream_t* op = out + token * out_token_stride + col;
#pragma unroll
  for (int j = 0; j < N_HC; ++j) {
    float rounded[VEC_SIZE];
#pragma unroll
    for (int c = 0; c < VEC_SIZE; ++c) {
      rounded[c] = round_stream<stream_t>(v[j][c]);
      sq += rounded[c] * rounded[c];
    }
    store_pack_f32<stream_t, VEC_SIZE>(op + static_cast<int64_t>(j) * out_route_stride, rounded);
  }
  return sq;
}

/// pre^T residual for one 16 B pack, rounded to the stream dtype.
template <typename stream_t, int N_HC, int VEC_SIZE>
__device__ __forceinline__ void hc_mix_streams_pack(const stream_t* __restrict__ residual,
                                                    int64_t token_stride, int64_t route_stride,
                                                    int64_t token, int64_t col,
                                                    const float* __restrict__ pre,
                                                    float (&vals)[VEC_SIZE]) {
#pragma unroll
  for (int c = 0; c < VEC_SIZE; ++c) vals[c] = 0.0f;
  const stream_t* rp = residual + token * token_stride + col;
#pragma unroll
  for (int i = 0; i < N_HC; ++i) {
    float rv[VEC_SIZE];
    load_pack_f32<stream_t, VEC_SIZE>(rp + static_cast<int64_t>(i) * route_stride, rv);
#pragma unroll
    for (int c = 0; c < VEC_SIZE; ++c) vals[c] = __fmaf_rn(pre[i], rv[c], vals[c]);
  }
#pragma unroll
  for (int c = 0; c < VEC_SIZE; ++c) vals[c] = round_stream<stream_t>(vals[c]);
}

/// Scalar form over one element (exact same fp32 order), used when the row did
/// not fit the shared stage.
template <typename stream_t, int N_HC>
__device__ __forceinline__ float hc_mix_streams_one(const stream_t* __restrict__ residual,
                                                    int64_t token_stride, int64_t route_stride,
                                                    int64_t token, int64_t idx,
                                                    const float* __restrict__ pre) {
  float acc = 0.0f;
  const stream_t* rp = residual + token * token_stride + idx;
#pragma unroll
  for (int i = 0; i < N_HC; ++i)
    acc = __fmaf_rn(pre[i], to_f(rp[static_cast<int64_t>(i) * route_stride]), acc);
  return round_stream<stream_t>(acc);
}

// ---- per-token projection ----------------------------------------------------
/// acc[o] += sum_k new_residual_flat[k] * fn[o, k], with the token's residual
/// read as 16 B packs. Returns the fp32 sum of squares of the stored values.
template <typename stream_t, int N_HC>
__device__ __forceinline__ float hc_project_row(const stream_t* __restrict__ residual,
                                                int64_t token_stride, int64_t route_stride,
                                                int64_t token, const float* __restrict__ fn,
                                                int64_t d, float* __restrict__ acc) {
  constexpr int N = N_HC;
  constexpr int kOut = N * (N + 2);
  constexpr int VEC = 16 / static_cast<int>(sizeof(stream_t));
  const int64_t total = static_cast<int64_t>(N) * d;
  const int64_t packs = total / VEC;
  const int64_t base = token * token_stride;
  float sq = 0.0f;
  for (int64_t p = threadIdx.x; p < packs; p += blockDim.x) {
    const int64_t k = p * VEC;
    const int64_t route = k / d;
    const int64_t h = k - route * d;
    float rv[VEC];
    load_pack_f32<stream_t, VEC>(residual + base + route * route_stride + h, rv);
#pragma unroll
    for (int c = 0; c < VEC; ++c) sq += rv[c] * rv[c];
    const float* fp = fn + k;
#pragma unroll
    for (int o = 0; o < kOut; ++o) {
      const float* fo = fp + static_cast<int64_t>(o) * total;
#pragma unroll
      for (int c = 0; c < VEC; ++c) acc[o] = __fmaf_rn(rv[c], fo[c], acc[o]);
    }
  }
  return sq;
}

// ---- coefficient emit --------------------------------------------------------
/// Constraint + output stage shared by every coefficient body: RMS scale, affine mixing,
/// sigmoid / Sinkhorn, then every requested coefficient output. `pre_out` (optional) receives
/// the n pre coefficients, which mega_mhc feeds into its layer-input phase.
template <int N_HC>
__device__ __forceinline__ void hc_coeff_emit(const HCParams& p, int64_t t,
                                              const float* __restrict__ dot, float sq_sum,
                                              bool write_sumsq, bool write_logits,
                                              float* __restrict__ pre_out) {
  constexpr int N = N_HC;
  constexpr int kOut = N * (N + 2);
  const float rms_scale = rsqrtf(sq_sum * (1.0f / static_cast<float>(N * p.d)) + p.hc_norm_eps);
  if (write_sumsq && p.row_sumsq != nullptr) p.row_sumsq[t] = sq_sum;
  if (write_logits && p.raw_logits != nullptr) {
    float* dst = const_cast<float*>(p.raw_logits) + t * kOut;
#pragma unroll
    for (int o = 0; o < kOut; ++o) dst[o] = dot[o];
  }
  float mix[kOut];
  hc_affine_mix<N>(dot, rms_scale, p.mix_scales, p.mix_bases, mix);

  float pre[N], post[N], comb[N * N];
  hc_apply_constraint<N>(mix, p, pre, post, comb);

  float* ca = p.coeff_a;
  float* cb = p.coeff_b;
  float* cc = p.coeff_c;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    if (ca != nullptr) ca[t * N + i] = pre[i];
    if (cc != nullptr) cc[t * N + i] = post[i];
    if (pre_out != nullptr) pre_out[i] = pre[i];
  }
  if (cb != nullptr) {
#pragma unroll
    for (int i = 0; i < N * N; ++i) cb[t * N * N + i] = comb[i];
  }
  if (p.new_prev_mix != nullptr) {
    for (int i = 0; i < N; ++i) static_cast<float*>(p.new_prev_mix)[t * N + i] = pre[i];
  }
  if (p.new_post_mix != nullptr) {
    for (int i = 0; i < N; ++i) static_cast<float*>(p.new_post_mix)[t * N + i] = post[i];
  }
  if (p.new_comb_res_mix != nullptr) {
    for (int i = 0; i < N * N; ++i)
      static_cast<float*>(p.new_comb_res_mix)[t * N * N + i] = comb[i];
  }
}

// ---- norm weight access ------------------------------------------------------
template <typename W>
__device__ __forceinline__ float hc_weight_at(const void* w, int64_t idx) {
  return w != nullptr ? to_f(static_cast<const W*>(w)[idx]) : 1.0f;
}

/// Second pass of the pre/norm: y = round(mix) * rms_scale * weight, written as
/// out_t and/or quantized to FP8 with the requested scale layout.
template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant, typename W>
__device__ __forceinline__ void hc_pre_write(const stream_t* __restrict__ residual,
                                             int64_t token_stride, int64_t route_stride,
                                             const float* __restrict__ pre, int64_t token,
                                             int64_t d, const void* __restrict__ w, float rms_scale,
                                             const float* __restrict__ stage,
                                             out_t* __restrict__ layer_in, const ActQuantOut& q,
                                             const HCParams& p) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = static_cast<int>(blockDim.x) >> 5;

  if constexpr (kQuant == ActQuantMode::kNone) {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float v = stage != nullptr
                          ? stage[idx]
                          : hc_mix_streams_one<stream_t, N_HC>(residual, token_stride, route_stride,
                                                               token, idx, pre);
      const float y = v * rms_scale * hc_weight_at<W>(w, idx);
      if (layer_in != nullptr) from_f(layer_in[token * d + idx], y);
    }
    return;
  } else {
    // FP8 per-group path: one warp per 128-element chunk, lane l owns element
    // g*32 + l of each of the four 32-element groups in the chunk.
    const int32_t group =
        q.group_size > 0 ? q.group_size : (q.mode == ActQuantMode::kMXFP8 ? 32 : 128);
    const int64_t chunks = d / 128;
    for (int64_t chunk = warp; chunk < chunks; chunk += num_warps) {
      const int64_t base = chunk * 128;
      float y[4];
      float amax[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int64_t idx = base + g * 32 + lane;
        const float v = stage != nullptr
                            ? stage[idx]
                            : hc_mix_streams_one<stream_t, N_HC>(residual, token_stride, route_stride,
                                                                 token, idx, pre);
        // The official FP8 payload is derived from the stream-dtype output
        // (DeepGEMM rounds to BF16 before casting), so round before the amax.
        y[g] = round_stream<stream_t>(v * rms_scale * hc_weight_at<W>(w, idx));
        amax[g] = fabsf(y[g]);
        if (layer_in != nullptr) from_f(layer_in[token * d + idx], y[g]);
      }
      uint8_t exp[4] = {0, 0, 0, 0};
      float scale[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float inv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (group == 32) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const float a = warp_reduce_max(amax[g]);
          scale[g] = fmaxf(a, 1e-4f) / kFp8E4M3Max;
          if (q.scale_ue8m0) {
            exp[g] = hc_ue8m0_ceil(scale[g]);
            inv[g] = 1.0f / hc_ue8m0_value(exp[g]);
          } else {
            inv[g] = 1.0f / scale[g];
          }
        }
      } else {
        const float a =
            warp_reduce_max(fmaxf(fmaxf(amax[0], amax[1]), fmaxf(amax[2], amax[3])));
        const float s = fmaxf(a, 1e-4f) / kFp8E4M3Max;
        float inv_all;
        if (q.scale_ue8m0) {
          exp[0] = hc_ue8m0_ceil(s);
          inv_all = 1.0f / hc_ue8m0_value(exp[0]);
        } else {
          inv_all = 1.0f / s;
        }
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          scale[g] = s;
          inv[g] = inv_all;
        }
      }
      uint8_t* qp = static_cast<uint8_t*>(q.q) + token * q.q_row_stride + base;
#pragma unroll
      for (int g = 0; g < 4; ++g) qp[g * 32 + lane] = hc_e4m3(y[g] * inv[g]);

      // Scale factors (lane 0 owns the row/word, the values are block-uniform).
      if (lane == 0) {
        if (group == 32) {
          if (q.scale_ue8m0 && q.scale_layout == ScaleLayout::kPackedUE8M0x4) {
            const uint32_t packed = static_cast<uint32_t>(exp[0]) |
                                    (static_cast<uint32_t>(exp[1]) << 8) |
                                    (static_cast<uint32_t>(exp[2]) << 16) |
                                    (static_cast<uint32_t>(exp[3]) << 24);
            if (q.scales != nullptr)
              static_cast<uint32_t*>(q.scales)[token * q.scale_row_stride + chunk] = packed;
            if (p.y_shared_sf != nullptr) {
              const int64_t num_rows =
                  (p.num_tokens + p.shared_sf_block_m - 1) / p.shared_sf_block_m *
                  ((static_cast<int64_t>(p.shared_sf_block_m) + 127) & ~int64_t(127));
              const int64_t row = hc_shared_sf_row(token, p.shared_sf_block_m);
              static_cast<uint32_t*>(p.y_shared_sf)[row + chunk * num_rows] = packed;
            }
          } else if (q.scales != nullptr) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
              if (q.scale_ue8m0) {
                static_cast<uint8_t*>(q.scales)[token * q.scale_row_stride + chunk * 4 + g] =
                    exp[g];
              } else {
                static_cast<float*>(q.scales)[token * q.scale_row_stride + chunk * 4 + g] =
                    scale[g];
              }
            }
          }
        } else if (q.scales != nullptr) {
          if (q.scale_ue8m0) {
            static_cast<uint8_t*>(q.scales)[token * q.scale_row_stride + chunk] = exp[0];
          } else {
            static_cast<float*>(q.scales)[token * q.scale_row_stride + chunk] = scale[0];
          }
        }
      }
    }
  }
}

template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant>
__device__ __forceinline__ void hc_pre_write_dispatch(
    const stream_t* __restrict__ residual, int64_t token_stride, int64_t route_stride,
    const float* __restrict__ pre, int64_t token, int64_t d, const void* __restrict__ w,
    DType w_dtype, float rms_scale, const float* __restrict__ stage,
    out_t* __restrict__ layer_in, const ActQuantOut& q, const HCParams& p) {
  if (w_dtype == DType::kBFloat16) {
    hc_pre_write<stream_t, out_t, N_HC, kQuant, __nv_bfloat16>(
        residual, token_stride, route_stride, pre, token, d, w, rms_scale, stage, layer_in, q, p);
  } else if (w_dtype == DType::kFloat16) {
    hc_pre_write<stream_t, out_t, N_HC, kQuant, __half>(
        residual, token_stride, route_stride, pre, token, d, w, rms_scale, stage, layer_in, q, p);
  } else {
    hc_pre_write<stream_t, out_t, N_HC, kQuant, float>(
        residual, token_stride, route_stride, pre, token, d, w, rms_scale, stage, layer_in, q, p);
  }
}

// ---- first pass: stage + squared sum ----------------------------------------
/// Rounds the pre-weighted mix to the stream dtype, optionally stages it in
/// shared memory, and returns this thread's contribution to sum(y^2).
template <typename stream_t, int N_HC, int VEC_SIZE>
__device__ __forceinline__ float hc_stage_row(const stream_t* __restrict__ residual,
                                              int64_t token_stride, int64_t route_stride,
                                              const float* __restrict__ pre, int64_t token, int64_t d,
                                              float* __restrict__ stage) {
  const int64_t packs = d / VEC_SIZE;
  float sq = 0.0f;
  for (int64_t p = threadIdx.x; p < packs; p += blockDim.x) {
    const int64_t col = p * VEC_SIZE;
    float vals[VEC_SIZE];
    hc_mix_streams_pack<stream_t, N_HC, VEC_SIZE>(residual, token_stride, route_stride, token, col,
                                                  pre, vals);
    if (stage != nullptr) {
#pragma unroll
      for (int c = 0; c < VEC_SIZE; ++c) stage[col + c] = vals[c];
    }
#pragma unroll
    for (int c = 0; c < VEC_SIZE; ++c) sq += vals[c] * vals[c];
  }
  return sq;
}

/// Stage the pre mix, reduce its squared sum, then run the weighted output pass.
/// The norm weight is `w` (rmsnorm_weight when present, else sub_norm_weight).
template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant>
__device__ __forceinline__ void hc_pre_row(const stream_t* __restrict__ residual,
                                           int64_t token_stride, int64_t route_stride,
                                           const float* __restrict__ pre, int64_t token, int64_t d,
                                           const void* __restrict__ w, DType w_dtype, float norm_eps,
                                           float norm_scale, float* __restrict__ stage,
                                           float* __restrict__ red, out_t* __restrict__ layer_in,
                                           const ActQuantOut& q, const HCParams& p) {
  constexpr int VEC = 16 / static_cast<int>(sizeof(stream_t));
  float vals[N_HC + 1];
  vals[N_HC] = hc_stage_row<stream_t, N_HC, VEC>(residual, token_stride, route_stride, pre, token,
                                                 d, stage);
  hc_block_reduce_all<N_HC + 1>(vals, red);
  const float rms_scale = rsqrtf(vals[N_HC] * (1.0f / static_cast<float>(d)) + norm_eps) * norm_scale;
  hc_pre_write_dispatch<stream_t, out_t, N_HC, kQuant>(
      residual, token_stride, route_stride, pre, token, d, w, w_dtype, rms_scale, stage, layer_in, q,
      p);
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernels (declared in hyper_connection_kernels.cuh, so they live in
// kops::model_ops rather than the file-local namespace above).
// ---------------------------------------------------------------------------

/// X_0[t, i, :] = embed[t, :] for every stream i (packed streams).
template <typename in_t, typename stream_t, int N_HC, int VEC_SIZE>
__global__ void hc_expand_kernel(stream_t* __restrict__ streams, const in_t* __restrict__ embed,
                                 const int num_tokens, const int d) {
  const int64_t packs = d / VEC_SIZE;
  const int64_t total = static_cast<int64_t>(num_tokens) * packs;
  for (int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = p / packs;
    const int64_t col = (p - t * packs) * VEC_SIZE;
    float v[VEC_SIZE];
    load_pack_f32<in_t, VEC_SIZE>(embed + t * d + col, v);
    stream_t* dst = streams + (t * N_HC) * d + col;
#pragma unroll
    for (int i = 0; i < N_HC; ++i) store_pack_f32<stream_t, VEC_SIZE>(dst + i * d, v);
  }
}

/// out[t, :] = sum_i w_i streams[t, i, :] (packed streams).
template <typename stream_t, typename out_t, int N_HC, int VEC_SIZE>
__global__ void hc_head_kernel(out_t* __restrict__ out, const stream_t* __restrict__ streams,
                               const float* __restrict__ head_weights,
                               const bool per_token_weights, const int num_tokens, const int d) {
  const int64_t packs = d / VEC_SIZE;
  const int64_t total = static_cast<int64_t>(num_tokens) * packs;
  for (int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = p / packs;
    const int64_t col = (p - t * packs) * VEC_SIZE;
    float w[N_HC];
    if (head_weights == nullptr) {
#pragma unroll
      for (int i = 0; i < N_HC; ++i) w[i] = 1.0f / static_cast<float>(N_HC);
    } else if (per_token_weights) {
#pragma unroll
      for (int i = 0; i < N_HC; ++i) w[i] = head_weights[t * N_HC + i];
    } else {
#pragma unroll
      for (int i = 0; i < N_HC; ++i) w[i] = head_weights[i];
    }
    const stream_t* src = streams + (t * N_HC) * d + col;
    float acc[VEC_SIZE];
#pragma unroll
    for (int c = 0; c < VEC_SIZE; ++c) acc[c] = 0.0f;
#pragma unroll
    for (int i = 0; i < N_HC; ++i) {
      float rv[VEC_SIZE];
      load_pack_f32<stream_t, VEC_SIZE>(src + i * d, rv);
#pragma unroll
      for (int c = 0; c < VEC_SIZE; ++c) acc[c] = __fmaf_rn(w[i], rv[c], acc[c]);
    }
    store_pack_f32<out_t, VEC_SIZE>(out + t * d + col, acc);
  }
}

template <typename stream_t, typename y_t, int N_HC, int VEC_SIZE>
__global__ void hc_post_kernel(const __grid_constant__ HCParams params,
                               const y_t* __restrict__ y, const int64_t y_stride) {
  const int64_t d = params.d;
  const int64_t packs = d / VEC_SIZE;
  const int64_t total = static_cast<int64_t>(params.num_tokens) * packs;
  const auto* residual = static_cast<const stream_t*>(params.residual);
  stream_t* out = static_cast<stream_t*>(params.new_residual != nullptr ? params.new_residual
                                                                       : params.residual);
  const int64_t out_token_stride =
      params.new_residual != nullptr && params.new_residual_token_stride > 0
          ? params.new_residual_token_stride
          : params.residual_token_stride;
  const int64_t out_route_stride =
      params.new_residual != nullptr && params.new_residual_route_stride > 0
          ? params.new_residual_route_stride
          : params.residual_route_stride;
  for (int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = p / packs;
    const int64_t col = (p - t * packs) * VEC_SIZE;
    hc_mix_pack<stream_t, y_t, N_HC, VEC_SIZE>(
        residual, params.residual_token_stride, params.residual_route_stride, y, y_stride,
        params.post_mix + t * N_HC, params.comb_res_mix + t * N_HC * N_HC, out, out_token_stride,
        out_route_stride, t, col);
  }
}

template <typename stream_t, typename w_t, int N_HC, bool kFromLogits>
__global__ void hc_coeff_kernel(const __grid_constant__ HCParams params) {
  constexpr int N = N_HC;
  constexpr int kOut = N * (N + 2);
  constexpr int VEC = 16 / static_cast<int>(sizeof(stream_t));
  __shared__ float red[(kWarps + 1) * (kOut + 1)];
  __shared__ float tile_logits[kTileRows * kOut];
  __shared__ float tile_sq[kTileRows];
  __shared__ float tile_acc[kWarps * kTileRows * kOut];
  __shared__ float tile_wsq[kWarps * kTileRows];
  const int64_t d = params.d;
  const auto* residual = static_cast<const stream_t*>(
      params.new_residual != nullptr ? params.new_residual : params.residual);
  const int64_t token_stride = params.new_residual != nullptr && params.new_residual_token_stride > 0
                                   ? params.new_residual_token_stride
                                   : params.residual_token_stride;
  const int64_t route_stride = params.new_residual != nullptr &&
                                       params.new_residual_route_stride > 0
                                   ? params.new_residual_route_stride
                                   : params.residual_route_stride;

  if constexpr (!kFromLogits) {
    if (hc_coeff_tile_mode(params)) {
      // One CTA per 16 tokens: tensor-core projection -> raw_logits / row_sumsq ->
      // constraint per row. No cross-CTA dependency, so this stays a single launch.
      const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTileRows;
      hc_tc_project<stream_t, N>(residual, token_stride, route_stride, t0, params.num_tokens,
                                 params.fn, d, tile_logits, tile_acc, tile_sq, tile_wsq);
      const int64_t t = t0 + threadIdx.x;
      if (threadIdx.x < kTileRows && t < params.num_tokens) {
        float dot[kOut];
#pragma unroll
        for (int o = 0; o < kOut; ++o) dot[o] = tile_logits[threadIdx.x * kOut + o];
        hc_coeff_emit<N>(params, t, dot, tile_sq[threadIdx.x], true, true, nullptr);
      }
      return;
    }
  }

  const int64_t t = blockIdx.x;

  float vals[kOut + 1];
#pragma unroll
  for (int o = 0; o < kOut; ++o) vals[o] = 0.0f;
  vals[kOut] = 0.0f;

  if constexpr (!kFromLogits) {
    vals[kOut] = hc_project_row<stream_t, N>(residual, token_stride, route_stride, t, params.fn, d,
                                             vals);
  } else if (params.row_sumsq == nullptr) {
    // No GEMM-provided row sum of squares: recompute it from the stored stream.
    const int64_t packs = (static_cast<int64_t>(N) * d) / VEC;
    for (int64_t p = threadIdx.x; p < packs; p += blockDim.x) {
      const int64_t k = p * VEC;
      const int64_t route = k / d;
      const int64_t h = k - route * d;
      float rv[VEC];
      load_pack_f32<stream_t, VEC>(residual + t * token_stride + route * route_stride + h, rv);
#pragma unroll
      for (int c = 0; c < VEC; ++c) vals[kOut] += rv[c] * rv[c];
    }
  }
  hc_block_reduce_all<kOut + 1>(vals, red);

  if (threadIdx.x != 0) return;
  const float sq_sum =
      (kFromLogits && params.row_sumsq != nullptr) ? params.row_sumsq[t] : vals[kOut];
  float dot[kOut];
  if constexpr (kFromLogits) {
#pragma unroll
    for (int o = 0; o < kOut; ++o) dot[o] = params.raw_logits[t * kOut + o];
  } else {
#pragma unroll
    for (int o = 0; o < kOut; ++o) dot[o] = vals[o];
  }
  hc_coeff_emit<N>(params, t, dot, sq_sum, !kFromLogits, !kFromLogits, nullptr);
}

template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant, int VEC_SIZE>
__global__ void hc_pre_kernel(const __grid_constant__ HCParams params,
                              out_t* __restrict__ layer_in, const ActQuantOut quant,
                              const void* __restrict__ sub_norm_weight, const float sub_norm_eps) {
  constexpr int N = N_HC;
  extern __shared__ float stage_ext[];
  __shared__ float red[(kWarps + 1) * (N + 1)];
  float* stage = (params.d * static_cast<int64_t>(sizeof(float)) <= kStageCapBytes) ? stage_ext
                                                                                    : nullptr;
  const int64_t t = blockIdx.x;
  const int64_t d = params.d;
  float pre[N];
  const float* shifted = params.shifted_prev_mix;
#pragma unroll
  for (int i = 0; i < N; ++i)
    pre[i] = shifted != nullptr ? shifted[t * N + i]
                                : (params.coeff_a != nullptr ? params.coeff_a[t * N + i] : 0.0f);

  const bool use_rms = params.rmsnorm_weight != nullptr;
  const void* w = use_rms ? params.rmsnorm_weight : sub_norm_weight;
  const float eps = use_rms ? params.rmsnorm_eps : sub_norm_eps;
  const float scale = use_rms ? params.rmsnorm_scale : 1.0f;

  const auto* residual = static_cast<const stream_t*>(
      params.new_residual != nullptr ? params.new_residual : params.residual);
  const int64_t token_stride = params.new_residual != nullptr && params.new_residual_token_stride > 0
                                   ? params.new_residual_token_stride
                                   : params.residual_token_stride;
  const int64_t route_stride = params.new_residual != nullptr && params.new_residual_route_stride > 0
                                   ? params.new_residual_route_stride
                                   : params.residual_route_stride;

  hc_pre_row<stream_t, out_t, N, kQuant>(residual, token_stride, route_stride, pre, t, d, w,
                                         params.rmsnorm_dtype, eps, scale, stage, red, layer_in,
                                         quant, params);
}

// ---- mega_mhc ---------------------------------------------------------------
template <typename stream_t, typename y_t, typename out_t, int N_HC, int D_TILE,
          ActQuantMode kQuant, typename ArchTag>
__global__ void mega_mhc_kernel(const __grid_constant__ HCParams next_layer,
                                const float* __restrict__ prev_coeff_b,
                                const float* __restrict__ prev_coeff_c,
                                const y_t* __restrict__ prev_y, out_t* __restrict__ layer_in,
                                const ActQuantOut quant, const void* __restrict__ sub_norm_weight,
                                const float sub_norm_eps) {
  constexpr int N = N_HC;
  constexpr int kOut = N * (N + 2);
  constexpr int VEC = 16 / static_cast<int>(sizeof(stream_t));
  extern __shared__ float stage_ext[];
  __shared__ float red[(kWarps + 1) * (kOut + 1)];
  __shared__ float pre_sh[kTileRows * N];
  __shared__ float tile_logits[kTileRows * kOut];
  __shared__ float tile_sq[kTileRows];
  __shared__ float tile_acc[kWarps * kTileRows * kOut];
  __shared__ float tile_wsq[kWarps * kTileRows];
  const int64_t d = next_layer.d;
  (void)ArchTag::kArch;
  float* stage =
      (d * static_cast<int64_t>(sizeof(float)) <= kStageCapBytes) ? stage_ext : nullptr;

  const auto* residual = static_cast<const stream_t*>(next_layer.residual);
  stream_t* out = static_cast<stream_t*>(next_layer.new_residual != nullptr
                                            ? next_layer.new_residual
                                            : next_layer.residual);
  const int64_t out_token_stride =
      next_layer.new_residual != nullptr && next_layer.new_residual_token_stride > 0
          ? next_layer.new_residual_token_stride
          : next_layer.residual_token_stride;
  const int64_t out_route_stride =
      next_layer.new_residual != nullptr && next_layer.new_residual_route_stride > 0
          ? next_layer.new_residual_route_stride
          : next_layer.residual_route_stride;

  if constexpr (D_TILE == kTileRows) {
    (void)VEC;
    // ---- fused 16-token tile body: post -> tensor-core projection -> constraint -> pre ----
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTileRows;

    // Phase A: X_l = B_{l-1} X_{l-1} + C_{l-1} y_{l-1} for the tile's rows (hc_mix_pack is
    // the hc_post_kernel body, so the chain and the mega body store identical rows).
    {
      const int64_t packs = d / VEC;
      const int64_t total = static_cast<int64_t>(kTileRows) * packs;
      for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int64_t row = idx / packs;
        const int64_t t = t0 + row;
        if (t >= next_layer.num_tokens) continue;
        hc_mix_pack<stream_t, y_t, N, VEC>(
            residual, next_layer.residual_token_stride, next_layer.residual_route_stride, prev_y,
            next_layer.d, prev_coeff_c + t * N, prev_coeff_b + t * N * N, out, out_token_stride,
            out_route_stride, t, (idx - row * packs) * VEC);
      }
    }
    __syncthreads();

    // Phase B: projection of the rows just written (visible after the barrier).
    hc_tc_project<stream_t, N>(out, out_token_stride, out_route_stride, t0,
                               next_layer.num_tokens, next_layer.fn, d, tile_logits, tile_acc,
                               tile_sq, tile_wsq);

    // Phase C: coefficients for the same rows.
    if (threadIdx.x < kTileRows && t0 + threadIdx.x < next_layer.num_tokens) {
      float dot[kOut];
#pragma unroll
      for (int o = 0; o < kOut; ++o) dot[o] = tile_logits[threadIdx.x * kOut + o];
      hc_coeff_emit<N>(next_layer, t0 + threadIdx.x, dot, tile_sq[threadIdx.x], true, false,
                       pre_sh + threadIdx.x * N);
    }
    __syncthreads();

    // Phase D: x_in for the tile's rows (shifted state replaces the fresh pre).
    const bool use_rms = next_layer.rmsnorm_weight != nullptr;
    const void* w = use_rms ? next_layer.rmsnorm_weight : sub_norm_weight;
    const float eps = use_rms ? next_layer.rmsnorm_eps : sub_norm_eps;
    const float scale = use_rms ? next_layer.rmsnorm_scale : 1.0f;
    const float* shifted = next_layer.shifted_prev_mix;
    for (int row = 0; row < kTileRows; ++row) {
      const int64_t t = t0 + row;
      if (t < next_layer.num_tokens) {
        float pre[N];
#pragma unroll
        for (int i = 0; i < N; ++i)
          pre[i] = shifted != nullptr ? shifted[t * N + i] : pre_sh[row * N + i];
        hc_pre_row<stream_t, out_t, N, kQuant>(out, out_token_stride, out_route_stride, pre, t, d,
                                               w, next_layer.rmsnorm_dtype, eps, scale, stage, red,
                                               layer_in, quant, next_layer);
      }
      __syncthreads();
    }
    return;
  }

  const int64_t t = blockIdx.x;
  (void)D_TILE;  // 0: the generic body owns one full token.

  // Phase A: X_l = B_{l-1} X_{l-1} + C_{l-1} y_{l-1}; rounded once, then read back.
  float sq = 0.0f;
  {
    const int64_t packs = d / VEC;
    for (int64_t p = threadIdx.x; p < packs; p += blockDim.x) {
      const int64_t col = p * VEC;
      sq += hc_mix_pack<stream_t, y_t, N, VEC>(
          residual, next_layer.residual_token_stride, next_layer.residual_route_stride, prev_y,
          next_layer.d, prev_coeff_c + t * N, prev_coeff_b + t * N * N, out, out_token_stride,
          out_route_stride, t, col);
    }
  }
  __syncthreads();

  // Phase B: coefficients from X_l (identical order to hc_coeff_kernel).
  float vals[kOut + 1];
#pragma unroll
  for (int o = 0; o < kOut; ++o) vals[o] = 0.0f;
  vals[kOut] = 0.0f;
  vals[kOut] =
      hc_project_row<stream_t, N>(out, out_token_stride, out_route_stride, t, next_layer.fn, d, vals);
  hc_block_reduce_all<kOut + 1>(vals, red);
  if (threadIdx.x == 0) {
    hc_coeff_emit<N>(next_layer, t, vals, vals[kOut], true, false, pre_sh);
  }
  __syncthreads();

  // Phase C: x_in for layer l. Shifted state replaces the freshly predicted pre.
  float pre[N];
  const float* shifted = next_layer.shifted_prev_mix;
#pragma unroll
  for (int i = 0; i < N; ++i) pre[i] = shifted != nullptr ? shifted[t * N + i] : pre_sh[i];

  const bool use_rms = next_layer.rmsnorm_weight != nullptr;
  const void* w = use_rms ? next_layer.rmsnorm_weight : sub_norm_weight;
  const float eps = use_rms ? next_layer.rmsnorm_eps : sub_norm_eps;
  const float scale = use_rms ? next_layer.rmsnorm_scale : 1.0f;

  hc_pre_row<stream_t, out_t, N, kQuant>(out, out_token_stride, out_route_stride, pre, t, d, w,
                                         next_layer.rmsnorm_dtype, eps, scale, stage, red, layer_in,
                                         quant, next_layer);
}

namespace {

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

bool is_stream_dtype(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

bool is_n_hc(int32_t n) { return n == 1 || n == 2 || n == 4; }

bool aligned16(const void* p) {
  return p == nullptr || (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

bool aligned4(const void* p) {
  return p == nullptr || (reinterpret_cast<uintptr_t>(p) & 3u) == 0;
}

template <typename T>
constexpr int pack16() {
  return 16 / static_cast<int>(sizeof(T));
}

int grid_for(int64_t work, int32_t units) {
  int64_t g = (work + kBlock - 1) / kBlock;
  if (g < 1) g = 1;
  if (units > 0) {
    const int64_t cap = static_cast<int64_t>(units) * 32;
    if (g > cap) g = cap;
  }
  return static_cast<int>(g > INT32_MAX ? INT32_MAX : g);
}

Status launch_check(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s launch: %s",
                                     what, cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

/// Element count of the new_residual view (falls back to residual in place).
struct HCOutView {
  void* data = nullptr;
  int64_t token_stride = 0;
  int64_t route_stride = 0;
};

HCOutView hc_out_view(const HCParams& p) {
  HCOutView v{};
  if (p.new_residual != nullptr) {
    v.data = p.new_residual;
    v.token_stride =
        p.new_residual_token_stride > 0 ? p.new_residual_token_stride : p.residual_token_stride;
    v.route_stride =
        p.new_residual_route_stride > 0 ? p.new_residual_route_stride : p.residual_route_stride;
  } else {
    v.data = p.residual;
    v.token_stride = p.residual_token_stride;
    v.route_stride = p.residual_route_stride;
  }
  return v;
}

Status hc_check_common(const HCParams& p, const char* who) {
  if (p.num_tokens <= 0 || p.n_hc <= 0 || p.d <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: num_tokens %d, n_hc %d, d %d must all be > 0", who, p.num_tokens,
                             p.n_hc, p.d);
  }
  if (p.num_tokens > INT32_MAX || p.d > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_tokens %d, d %d exceed int32", who,
                             p.num_tokens, p.d);
  }
  if (!is_n_hc(p.n_hc)) {
    return detail::set_error(Status::kUnsupportedShape, "%s: n_hc %d not in {1, 2, 4}", who,
                             p.n_hc);
  }
  if (!is_stream_dtype(p.residual_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: residual dtype %s", who,
                             dtype_name(p.residual_dtype));
  }
  if (p.residual == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: residual is null", who);
  }
  if (p.residual_token_stride <= 0 || p.residual_route_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: residual strides must be positive (token %lld, route %lld)", who,
                             static_cast<long long>(p.residual_token_stride),
                             static_cast<long long>(p.residual_route_stride));
  }
  if (p.residual_route_stride < p.d) {
    return detail::set_error(Status::kInvalidArgument, "%s: residual_route_stride %lld < d %d", who,
                             static_cast<long long>(p.residual_route_stride), p.d);
  }
  if (p.new_residual != nullptr) {
    if (p.new_residual_token_stride < 0 || p.new_residual_route_stride < 0) {
      return detail::set_error(Status::kInvalidArgument, "%s: new_residual strides are negative",
                               who);
    }
  }
  return Status::kSuccess;
}

/// All fp32 mix/projection pointers are read as scalars; 4 B alignment is enough.
Status hc_check_mix_pointers(const HCParams& p, const char* who, bool need_fn,
                             bool need_scales) {
  if (p.post_mix != nullptr && !aligned4(p.post_mix)) {
    return detail::set_error(Status::kMisaligned, "%s: post_mix is not 4 B aligned", who);
  }
  if (p.comb_res_mix != nullptr && !aligned4(p.comb_res_mix)) {
    return detail::set_error(Status::kMisaligned, "%s: comb_res_mix is not 4 B aligned", who);
  }
  if (p.shifted_prev_mix != nullptr && !aligned4(p.shifted_prev_mix)) {
    return detail::set_error(Status::kMisaligned, "%s: shifted_prev_mix is not 4 B aligned", who);
  }
  if (need_fn && p.fn == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: fn is required", who);
  }
  if (need_scales && (p.mix_scales == nullptr || p.mix_bases == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "%s: mix_scales / mix_bases are required",
                             who);
  }
  if ((p.mix_scales != nullptr && !aligned4(p.mix_scales)) ||
      (p.mix_bases != nullptr && !aligned4(p.mix_bases))) {
    return detail::set_error(Status::kMisaligned, "%s: mix_scales / mix_bases are not 4 B aligned",
                             who);
  }
  return Status::kSuccess;
}

/// Validate an optional layer output / FP8 quant descriptor.
Status hc_check_quant(const ActQuantOut& q, const Tensor& layer_in, const HCParams& p,
                      const char* who) {
  if (q.mode == ActQuantMode::kNone) {
    if (!tensor_present(layer_in)) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: layer_in is absent and quant.mode is kNone", who);
    }
    return Status::kSuccess;
  }
  if (q.mode != ActQuantMode::kPerGroup && q.mode != ActQuantMode::kMXFP8) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: ActQuantMode %d is not implemented",
                             who, static_cast<int>(q.mode));
  }
  if (q.q == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: quant.q is null", who);
  }
  if (q.q_dtype != DType::kFloat8E4M3) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: quant payload %s (only e4m3)", who,
                             dtype_name(q.q_dtype));
  }
  if (q.q_row_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: quant.q_row_stride must be > 0", who);
  }
  const int32_t group = q.group_size > 0 ? q.group_size : (q.mode == ActQuantMode::kMXFP8 ? 32 : 128);
  if (group != 32 && group != 128) {
    return detail::set_error(Status::kUnsupportedShape, "%s: quant group %d not in {32, 128}", who,
                             group);
  }
  if (q.mode == ActQuantMode::kMXFP8 && group != 32) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: kMXFP8 is the 1x32 UE8M0 recipe (group 32), got %d", who, group);
  }
  if (p.d % 128 != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: quantized output needs d %% 128 == 0 (d=%d)", who, p.d);
  }
  if (q.scale_ue8m0) {
    if (q.scale_layout != ScaleLayout::kPackedUE8M0x4 &&
        q.scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: UE8M0 scale layout %d", who,
                               static_cast<int>(q.scale_layout));
    }
    if (q.scale_layout == ScaleLayout::kPackedUE8M0x4 && group != 32) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "%s: packed UE8M0 x4 needs group 32", who);
    }
  } else if (q.scale_layout != ScaleLayout::kRowMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: fp32 scales must be row-major", who);
  }
  if (q.scales == nullptr && p.y_shared_sf == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: quant scales and y_shared_sf are both absent", who);
  }
  if (q.scales != nullptr && q.scale_row_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: quant.scale_row_stride must be > 0",
                             who);
  }
  if (q.zero_points != nullptr || q.static_scale != nullptr || q.global_scale != nullptr ||
      q.scale_upper_bound != nullptr || q.amax_out != nullptr) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "%s: only per-group FP8 output is implemented", who);
  }
  if (p.y_shared_sf != nullptr) {
    if (p.shared_sf_block_m <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: y_shared_sf needs shared_sf_block_m > 0", who);
    }
    if (!q.scale_ue8m0 || group != 32) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "%s: y_shared_sf needs UE8M0 group-32 scales", who);
    }
    if (q.scales == nullptr && q.scale_layout != ScaleLayout::kPackedUE8M0x4) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "%s: y_shared_sf without quant.scales needs packed UE8M0 x4", who);
    }
  }
  return Status::kSuccess;
}

Status hc_check_layer_in(const Tensor& t, const HCParams& p, DType* out_dtype, const char* who) {
  Status s = check_tensor(t, DType::kInvalid, 2, 1, who);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != p.num_tokens || t.shape[1] != p.d) {
    return detail::set_error(Status::kInvalidArgument, "%s: shape [%lld,%lld] != [%d,%d]", who,
                             static_cast<long long>(t.shape[0]),
                             static_cast<long long>(t.shape[1]), p.num_tokens, p.d);
  }
  if (t.stride[1] != 1 || t.stride[0] < p.d) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: row must be contiguous", who);
  }
  if (!is_stream_dtype(t.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s", who, dtype_name(t.dtype));
  }
  *out_dtype = t.dtype;
  return Status::kSuccess;
}

/// 16 B pack viability of the per-token kernels (coeff / pre / mega).
Status hc_check_pack(const HCParams& p, const HCOutView& out, const void* y, int64_t y_stride,
                     const char* who) {
  const int64_t elem = (p.residual_dtype == DType::kFloat32) ? 4 : 2;
  const int64_t vec = 16 / elem;
  if (p.d % vec != 0 || p.residual_token_stride % vec != 0 ||
      p.residual_route_stride % vec != 0 || out.token_stride % vec != 0 ||
      out.route_stride % vec != 0 || !aligned16(p.residual) || !aligned16(out.data)) {
    return detail::set_error(Status::kMisaligned,
                             "%s: stream pointers/strides must be 16 B aligned and d=%d a multiple "
                             "of %lld",
                             who, p.d, static_cast<long long>(vec));
  }
  if (y != nullptr && (y_stride % vec != 0 || !aligned16(y))) {
    return detail::set_error(Status::kMisaligned, "%s: y must be 16 B aligned with a %lld-aligned "
                                                  "row stride",
                             who, static_cast<long long>(vec));
  }
  return Status::kSuccess;
}

// ---- per-op launchers --------------------------------------------------------

template <typename stream_t, int N_HC>
Status launch_post_pack(const HCParams& p, const void* y, int64_t y_stride, bool vec,
                        const LaunchContext& ctx) {
  constexpr int V = pack16<stream_t>();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int64_t packs = p.d / (vec ? V : 1);
  const int grid = grid_for(static_cast<int64_t>(p.num_tokens) * packs, ctx.compute_units);
  const auto* yp = static_cast<const stream_t*>(y);
  if (vec) {
    hc_post_kernel<stream_t, stream_t, N_HC, V><<<grid, kBlock, 0, stream>>>(p, yp, y_stride);
  } else {
    hc_post_kernel<stream_t, stream_t, N_HC, 1><<<grid, kBlock, 0, stream>>>(p, yp, y_stride);
  }
  return launch_check("hc_post");
}

template <typename stream_t>
Status launch_post(const HCParams& p, const void* y, int64_t y_stride, bool vec,
                   const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1: return launch_post_pack<stream_t, 1>(p, y, y_stride, vec, ctx);
    case 2: return launch_post_pack<stream_t, 2>(p, y, y_stride, vec, ctx);
    case 4: return launch_post_pack<stream_t, 4>(p, y, y_stride, vec, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "hc_post: n_hc %d", p.n_hc);
  }
}

/// Tensor-core bodies need SM80+; a hand-built context with arch_id 0 is left to the
/// generic body only when the device cannot run them (the device-side predicates carry
/// the same arch guard, so host and device always agree on the body).
bool hc_mma_arch_ok(const LaunchContext& ctx) {
  return !ctx_is_amd(ctx) && (ctx.arch_id == 0 || ctx.arch_id >= KOPS_TARGET_SM80);
}

template <typename stream_t, int N_HC>
Status launch_coeff_pack(const HCParams& p, bool from_logits, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const bool tile = !from_logits && hc_mma_arch_ok(ctx) && hc_coeff_tile_mode(p);
  const int64_t tiles = (static_cast<int64_t>(p.num_tokens) + kTileRows - 1) / kTileRows;
  const int grid = tile ? static_cast<int>(tiles) : p.num_tokens;
  if (from_logits) {
    hc_coeff_kernel<stream_t, float, N_HC, true><<<grid, kBlock, 0, stream>>>(p);
  } else {
    hc_coeff_kernel<stream_t, float, N_HC, false><<<grid, kBlock, 0, stream>>>(p);
  }
  return launch_check("hc_coefficients");
}

template <typename stream_t>
Status launch_coeff(const HCParams& p, bool from_logits, const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1: return launch_coeff_pack<stream_t, 1>(p, from_logits, ctx);
    case 2: return launch_coeff_pack<stream_t, 2>(p, from_logits, ctx);
    case 4: return launch_coeff_pack<stream_t, 4>(p, from_logits, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "hc_coefficients: n_hc %d",
                                      p.n_hc);
  }
}

template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant>
Status launch_pre_pack(const HCParams& p, out_t* layer_in, const ActQuantOut& q,
                       const void* sub_w, const float sub_eps, bool staged,
                       const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int grid = p.num_tokens;
  const size_t smem = staged ? static_cast<size_t>(p.d) * sizeof(float) : 0;
  hc_pre_kernel<stream_t, out_t, N_HC, kQuant, pack16<stream_t>()>
      <<<grid, kBlock, smem, stream>>>(p, layer_in, q, sub_w, sub_eps);
  return launch_check("hc_pre");
}

template <typename stream_t, typename out_t, int N_HC>
Status launch_pre_mode(const HCParams& p, out_t* layer_in, const ActQuantOut& q, const void* sub_w,
                       const float sub_eps, bool staged, const LaunchContext& ctx) {
  if (q.mode == ActQuantMode::kNone) {
    return launch_pre_pack<stream_t, out_t, N_HC, ActQuantMode::kNone>(p, layer_in, q, sub_w,
                                                                       sub_eps, staged, ctx);
  }
  return launch_pre_pack<stream_t, out_t, N_HC, ActQuantMode::kPerGroup>(p, layer_in, q, sub_w,
                                                                         sub_eps, staged, ctx);
}

template <typename stream_t, typename out_t>
Status launch_pre_n(const HCParams& p, out_t* layer_in, const ActQuantOut& q, const void* sub_w,
                    const float sub_eps, bool staged, const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1: return launch_pre_mode<stream_t, out_t, 1>(p, layer_in, q, sub_w, sub_eps, staged, ctx);
    case 2: return launch_pre_mode<stream_t, out_t, 2>(p, layer_in, q, sub_w, sub_eps, staged, ctx);
    case 4: return launch_pre_mode<stream_t, out_t, 4>(p, layer_in, q, sub_w, sub_eps, staged, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "hc_pre: n_hc %d", p.n_hc);
  }
}

template <typename stream_t>
Status launch_pre_out(const HCParams& p, const Tensor& layer_in, const ActQuantOut& q,
                      const void* sub_w, const float sub_eps, bool staged,
                      const LaunchContext& ctx) {
  if (tensor_present(layer_in) && layer_in.dtype == DType::kFloat32) {
    return launch_pre_n<stream_t, float>(p, static_cast<float*>(layer_in.data), q, sub_w, sub_eps,
                                         staged, ctx);
  }
  if (tensor_present(layer_in) && layer_in.dtype == DType::kFloat16) {
    return launch_pre_n<stream_t, __half>(p, static_cast<__half*>(layer_in.data), q, sub_w,
                                          sub_eps, staged, ctx);
  }
  if (tensor_present(layer_in) && layer_in.dtype == DType::kBFloat16) {
    return launch_pre_n<stream_t, __nv_bfloat16>(p, static_cast<__nv_bfloat16*>(layer_in.data), q,
                                                 sub_w, sub_eps, staged, ctx);
  }
  // FP8-only call: the template type never writes layer_in.
  return launch_pre_n<stream_t, float>(p, nullptr, q, sub_w, sub_eps, staged, ctx);
}

template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant>
Status launch_mega_pack(const HCParams& p, const float* prev_b, const float* prev_c,
                        const void* prev_y, out_t* layer_in, const ActQuantOut& q,
                        const void* sub_w, float sub_eps, bool staged, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const bool tile = hc_mma_arch_ok(ctx) && hc_mega_tile_mode(p);
  const int64_t tiles = (static_cast<int64_t>(p.num_tokens) + kTileRows - 1) / kTileRows;
  const int grid = tile ? static_cast<int>(tiles) : p.num_tokens;
  const size_t smem = staged ? static_cast<size_t>(p.d) * sizeof(float) : 0;
  if (tile) {
    mega_mhc_kernel<stream_t, stream_t, out_t, N_HC, kTileRows, kQuant, Sm80>
        <<<grid, kBlock, smem, stream>>>(p, prev_b, prev_c, static_cast<const stream_t*>(prev_y),
                                         layer_in, q, sub_w, sub_eps);
  } else {
    mega_mhc_kernel<stream_t, stream_t, out_t, N_HC, 0, kQuant, Sm80>
        <<<grid, kBlock, smem, stream>>>(p, prev_b, prev_c, static_cast<const stream_t*>(prev_y),
                                         layer_in, q, sub_w, sub_eps);
  }
  return launch_check("mega_mhc");
}

template <typename stream_t, typename out_t, int N_HC>
Status launch_mega_mode(const HCParams& p, const float* prev_b, const float* prev_c,
                        const void* prev_y, out_t* layer_in, const ActQuantOut& q,
                        const void* sub_w, float sub_eps, bool staged, const LaunchContext& ctx) {
  if (q.mode == ActQuantMode::kNone) {
    return launch_mega_pack<stream_t, out_t, N_HC, ActQuantMode::kNone>(
        p, prev_b, prev_c, prev_y, layer_in, q, sub_w, sub_eps, staged, ctx);
  }
  return launch_mega_pack<stream_t, out_t, N_HC, ActQuantMode::kPerGroup>(
      p, prev_b, prev_c, prev_y, layer_in, q, sub_w, sub_eps, staged, ctx);
}

template <typename stream_t, typename out_t>
Status launch_mega_n(const HCParams& p, const float* prev_b, const float* prev_c,
                     const void* prev_y, out_t* layer_in, const ActQuantOut& q, const void* sub_w,
                     float sub_eps, bool staged, const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1:
      return launch_mega_mode<stream_t, out_t, 1>(p, prev_b, prev_c, prev_y, layer_in, q, sub_w,
                                                  sub_eps, staged, ctx);
    case 2:
      return launch_mega_mode<stream_t, out_t, 2>(p, prev_b, prev_c, prev_y, layer_in, q, sub_w,
                                                  sub_eps, staged, ctx);
    case 4:
      return launch_mega_mode<stream_t, out_t, 4>(p, prev_b, prev_c, prev_y, layer_in, q, sub_w,
                                                  sub_eps, staged, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "mega_mhc: n_hc %d", p.n_hc);
  }
}

template <typename stream_t>
Status launch_mega_out(const HCParams& p, const float* prev_b, const float* prev_c,
                       const void* prev_y, const Tensor& layer_in, const ActQuantOut& q,
                       const void* sub_w, float sub_eps, bool staged, const LaunchContext& ctx) {
  if (tensor_present(layer_in) && layer_in.dtype == DType::kFloat32) {
    return launch_mega_n<stream_t, float>(p, prev_b, prev_c, prev_y,
                                          static_cast<float*>(layer_in.data), q, sub_w, sub_eps,
                                          staged, ctx);
  }
  if (tensor_present(layer_in) && layer_in.dtype == DType::kFloat16) {
    return launch_mega_n<stream_t, __half>(p, prev_b, prev_c, prev_y,
                                           static_cast<__half*>(layer_in.data), q, sub_w, sub_eps,
                                           staged, ctx);
  }
  if (tensor_present(layer_in) && layer_in.dtype == DType::kBFloat16) {
    return launch_mega_n<stream_t, __nv_bfloat16>(
        p, prev_b, prev_c, prev_y, static_cast<__nv_bfloat16*>(layer_in.data), q, sub_w, sub_eps,
        staged, ctx);
  }
  return launch_mega_n<stream_t, float>(p, prev_b, prev_c, prev_y, nullptr, q, sub_w, sub_eps,
                                        staged, ctx);
}

template <typename stream_t, int N_HC>
Status launch_expand_pack(const HCStreamParams& p, const void* embed, int vec,
                          const LaunchContext& ctx) {
  constexpr int V = pack16<stream_t>();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int64_t packs = p.d / vec;
  const int grid = grid_for(static_cast<int64_t>(p.num_tokens) * packs, ctx.compute_units);
  if (vec == V) {
    hc_expand_kernel<stream_t, stream_t, N_HC, V><<<grid, kBlock, 0, stream>>>(
        static_cast<stream_t*>(p.streams), static_cast<const stream_t*>(embed), p.num_tokens, p.d);
  } else {
    hc_expand_kernel<stream_t, stream_t, N_HC, 1><<<grid, kBlock, 0, stream>>>(
        static_cast<stream_t*>(p.streams), static_cast<const stream_t*>(embed), p.num_tokens, p.d);
  }
  return launch_check("hc_expand");
}

template <typename stream_t>
Status launch_expand(const HCStreamParams& p, const void* embed, int vec, const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1: return launch_expand_pack<stream_t, 1>(p, embed, vec, ctx);
    case 2: return launch_expand_pack<stream_t, 2>(p, embed, vec, ctx);
    case 4: return launch_expand_pack<stream_t, 4>(p, embed, vec, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "hc_expand: n_hc %d", p.n_hc);
  }
}

template <typename stream_t, typename out_t, int N_HC>
Status launch_head_pack(const HCStreamParams& p, out_t* out, const float* w, bool per_token,
                        int vec, const LaunchContext& ctx) {
  constexpr int V = pack16<stream_t>();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int64_t packs = p.d / vec;
  const int grid = grid_for(static_cast<int64_t>(p.num_tokens) * packs, ctx.compute_units);
  if (vec == V) {
    hc_head_kernel<stream_t, out_t, N_HC, V><<<grid, kBlock, 0, stream>>>(
        out, static_cast<const stream_t*>(p.streams), w, per_token, p.num_tokens, p.d);
  } else {
    hc_head_kernel<stream_t, out_t, N_HC, 1><<<grid, kBlock, 0, stream>>>(
        out, static_cast<const stream_t*>(p.streams), w, per_token, p.num_tokens, p.d);
  }
  return launch_check("hc_head");
}

template <typename stream_t, typename out_t>
Status launch_head_n(const HCStreamParams& p, out_t* out, const float* w, bool per_token, int vec,
                     const LaunchContext& ctx) {
  switch (p.n_hc) {
    case 1: return launch_head_pack<stream_t, out_t, 1>(p, out, w, per_token, vec, ctx);
    case 2: return launch_head_pack<stream_t, out_t, 2>(p, out, w, per_token, vec, ctx);
    case 4: return launch_head_pack<stream_t, out_t, 4>(p, out, w, per_token, vec, ctx);
    default: return detail::set_error(Status::kUnsupportedShape, "hc_head: n_hc %d", p.n_hc);
  }
}

template <typename stream_t>
Status launch_head_out(const HCStreamParams& p, const Tensor& out, const float* w, bool per_token,
                       int vec, const LaunchContext& ctx) {
  if (out.dtype == DType::kFloat32) {
    return launch_head_n<stream_t, float>(p, static_cast<float*>(out.data), w, per_token, vec, ctx);
  }
  if (out.dtype == DType::kFloat16) {
    return launch_head_n<stream_t, __half>(p, static_cast<__half*>(out.data), w, per_token, vec,
                                           ctx);
  }
  if (out.dtype == DType::kBFloat16) {
    return launch_head_n<stream_t, __nv_bfloat16>(p, static_cast<__nv_bfloat16*>(out.data), w,
                                                  per_token, vec, ctx);
  }
  return detail::set_error(Status::kUnsupportedDtype, "hc_head: out dtype %s",
                           dtype_name(out.dtype));
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host API
// ---------------------------------------------------------------------------

Status hc_post(const HCParams& params, const Tensor& y, const LaunchContext& ctx) {
  const char* who = "hc_post";
  Status s = hc_check_common(params, who);
  if (s != Status::kSuccess) return s;
  s = hc_check_mix_pointers(params, who, false, false);
  if (s != Status::kSuccess) return s;
  if (params.post_mix == nullptr || params.comb_res_mix == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: post_mix / comb_res_mix are required",
                             who);
  }
  const void* yp = nullptr;
  int64_t y_stride = 0;
  DType y_dtype = DType::kInvalid;
  if (tensor_present(y)) {
    s = check_tensor(y, DType::kInvalid, 2, 1, "hc_post.y");
    if (s != Status::kSuccess) return s;
    if (y.shape[0] != params.num_tokens || y.shape[1] != params.d) {
      return detail::set_error(Status::kInvalidArgument, "%s: y shape [%lld,%lld] != [%d,%d]", who,
                               static_cast<long long>(y.shape[0]),
                               static_cast<long long>(y.shape[1]), params.num_tokens, params.d);
    }
    if (y.stride[1] != 1 || y.stride[0] < params.d) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: y rows must be contiguous", who);
    }
    yp = y.data;
    y_stride = y.stride[0];
    y_dtype = y.dtype;
  } else if (params.x != nullptr) {
    yp = params.x;
    y_stride = params.x_stride > 0 ? params.x_stride : params.d;
    y_dtype = params.x_dtype;
  } else {
    return detail::set_error(Status::kInvalidArgument, "%s: neither y nor params.x is present", who);
  }
  if (y_dtype != params.residual_dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "%s: y dtype %s must equal residual dtype %s", who,
                             dtype_name(y_dtype), dtype_name(params.residual_dtype));
  }

  const int64_t elem = params.residual_dtype == DType::kFloat32 ? 4 : 2;
  const int64_t vec = 16 / elem;
  const HCOutView out = hc_out_view(params);
  const bool fast = params.d % vec == 0 && params.residual_token_stride % vec == 0 &&
                    params.residual_route_stride % vec == 0 && out.token_stride % vec == 0 &&
                    out.route_stride % vec == 0 && y_stride % vec == 0 && aligned16(params.residual) &&
                    aligned16(out.data) && aligned16(yp);

  switch (params.residual_dtype) {
    case DType::kFloat32: return launch_post<float>(params, yp, y_stride, fast, ctx);
    case DType::kFloat16: return launch_post<__half>(params, yp, y_stride, fast, ctx);
    case DType::kBFloat16: return launch_post<__nv_bfloat16>(params, yp, y_stride, fast, ctx);
    default: return detail::set_error(Status::kUnsupportedDtype, "%s: residual dtype %s", who,
                                      dtype_name(params.residual_dtype));
  }
}

Status hc_coefficients(const HCParams& params, const LaunchContext& ctx) {
  const char* who = "hc_coefficients";
  Status s = hc_check_common(params, who);
  if (s != Status::kSuccess) return s;
  // raw_logits alone is the DeepGEMM contract (the caller already ran the projection);
  // raw_logits + row_sumsq is the scratch pair, so this call projects and writes both.
  const bool from_logits = params.raw_logits != nullptr && params.row_sumsq == nullptr;
  s = hc_check_mix_pointers(params, who, !from_logits, true);
  if (s != Status::kSuccess) return s;
  if (params.raw_logits != nullptr && !aligned4(params.raw_logits)) {
    return detail::set_error(Status::kMisaligned, "%s: raw_logits is not 4 B aligned", who);
  }
  if (params.constraint == HCConstraint::kManifold && params.num_sinkhorn_iters < 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: kManifold needs num_sinkhorn_iters >= 1", who);
  }
  if (params.coeff_a == nullptr && params.coeff_b == nullptr && params.coeff_c == nullptr &&
      params.new_prev_mix == nullptr && params.new_post_mix == nullptr &&
      params.new_comb_res_mix == nullptr && params.row_sumsq == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: no coefficient output", who);
  }
  for (const float* p : {params.coeff_a, params.coeff_b, params.coeff_c, params.row_sumsq}) {
    if (p != nullptr && !aligned4(p)) {
      return detail::set_error(Status::kMisaligned, "%s: coefficient output is not 4 B aligned", who);
    }
  }
  const HCOutView out = hc_out_view(params);
  s = hc_check_pack(params, out, nullptr, 0, who);
  if (s != Status::kSuccess) return s;
  if (params.fn != nullptr && (!aligned16(params.fn) || params.d % 4 != 0)) {
    return detail::set_error(Status::kMisaligned,
                             "%s: fn must be 16 B aligned and d a multiple of 4", who);
  }

  switch (params.residual_dtype) {
    case DType::kFloat32: return launch_coeff<float>(params, from_logits, ctx);
    case DType::kFloat16: return launch_coeff<__half>(params, from_logits, ctx);
    case DType::kBFloat16: return launch_coeff<__nv_bfloat16>(params, from_logits, ctx);
    default: return detail::set_error(Status::kUnsupportedDtype, "%s: residual dtype %s", who,
                                      dtype_name(params.residual_dtype));
  }
}

Status hc_pre(const HCParams& params, const Tensor& layer_in, const ActQuantOut& quant,
              const Tensor& sub_norm_weight, float sub_norm_eps, const LaunchContext& ctx) {
  const char* who = "hc_pre";
  Status s = hc_check_common(params, who);
  if (s != Status::kSuccess) return s;
  if (params.coeff_a == nullptr && params.shifted_prev_mix == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: coeff_a and shifted_prev_mix are both absent", who);
  }
  s = hc_check_mix_pointers(params, who, false, false);
  if (s != Status::kSuccess) return s;
  if (params.shifted_prev_mix != nullptr && !aligned4(params.shifted_prev_mix)) {
    return detail::set_error(Status::kMisaligned, "%s: shifted_prev_mix is not 4 B aligned", who);
  }
  if (params.coeff_a != nullptr && !aligned4(params.coeff_a)) {
    return detail::set_error(Status::kMisaligned, "%s: coeff_a is not 4 B aligned", who);
  }
  s = hc_check_quant(quant, layer_in, params, who);
  if (s != Status::kSuccess) return s;
  DType out_dtype = DType::kFloat32;
  if (tensor_present(layer_in)) {
    s = hc_check_layer_in(layer_in, params, &out_dtype, "hc_pre.layer_in");
    if (s != Status::kSuccess) return s;
  }
  if (params.rmsnorm_weight != nullptr && !is_stream_dtype(params.rmsnorm_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: rmsnorm dtype %s", who,
                             dtype_name(params.rmsnorm_dtype));
  }
  if (tensor_present(sub_norm_weight)) {
    s = check_tensor(sub_norm_weight, DType::kInvalid, 1, 1, "hc_pre.sub_norm_weight");
    if (s != Status::kSuccess) return s;
    if (sub_norm_weight.shape[0] != params.d) {
      return detail::set_error(Status::kInvalidArgument, "%s: sub_norm_weight len %lld != d %d",
                               who, static_cast<long long>(sub_norm_weight.shape[0]), params.d);
    }
    if (sub_norm_weight.dtype != params.rmsnorm_dtype) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "%s: sub_norm_weight dtype %s must equal rmsnorm_dtype %s", who,
                               dtype_name(sub_norm_weight.dtype),
                               dtype_name(params.rmsnorm_dtype));
    }
  }
  const HCOutView out = hc_out_view(params);
  s = hc_check_pack(params, out, nullptr, 0, who);
  if (s != Status::kSuccess) return s;
  const bool staged = static_cast<int64_t>(params.d) * static_cast<int64_t>(sizeof(float)) <=
                      kStageCapBytes;
  const void* sub_w = tensor_present(sub_norm_weight) ? sub_norm_weight.data : nullptr;

  switch (params.residual_dtype) {
    case DType::kFloat32:
      return launch_pre_out<float>(params, layer_in, quant, sub_w, sub_norm_eps, staged, ctx);
    case DType::kFloat16:
      return launch_pre_out<__half>(params, layer_in, quant, sub_w, sub_norm_eps, staged, ctx);
    case DType::kBFloat16:
      return launch_pre_out<__nv_bfloat16>(params, layer_in, quant, sub_w, sub_norm_eps, staged, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "%s: residual dtype %s", who,
                               dtype_name(params.residual_dtype));
  }
}

/// Single-pass post + coefficients + pre. The previous sublayer's coefficients
/// are read from prev_coeff_b / prev_coeff_c and its output from prev_y;
/// next_layer.x, next_layer.post_mix and next_layer.comb_res_mix are therefore
/// not read (they belong to the unfused hc_post call).
Status mega_mhc(const HCParams& next_layer, const Tensor& prev_coeff_b, const Tensor& prev_coeff_c,
                const Tensor& prev_y, const Tensor& layer_in, const ActQuantOut& quant,
                const Tensor& sub_norm_weight, float sub_norm_eps, const LaunchContext& ctx) {
  const char* who = "mega_mhc";
  Status s = hc_check_common(next_layer, who);
  if (s != Status::kSuccess) return s;
  if (next_layer.raw_logits != nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: mega_mhc computes the projection itself; raw_logits must be null",
                             who);
  }
  s = hc_check_mix_pointers(next_layer, who, true, true);
  if (s != Status::kSuccess) return s;
  if (next_layer.constraint == HCConstraint::kManifold && next_layer.num_sinkhorn_iters < 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: kManifold needs num_sinkhorn_iters >= 1", who);
  }
  if (next_layer.coeff_a == nullptr && next_layer.coeff_b == nullptr &&
      next_layer.coeff_c == nullptr && next_layer.new_prev_mix == nullptr &&
      next_layer.new_post_mix == nullptr && next_layer.new_comb_res_mix == nullptr &&
      next_layer.row_sumsq == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: no coefficient output", who);
  }
  if (prev_coeff_b.data == nullptr || prev_coeff_c.data == nullptr || prev_y.data == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: prev_coeff_b / prev_coeff_c / prev_y are required", who);
  }
  s = check_tensor(prev_coeff_b, DType::kFloat32, 3, 4, "mega_mhc.prev_coeff_b");
  if (s != Status::kSuccess) return s;
  s = check_tensor(prev_coeff_c, DType::kFloat32, 3, 4, "mega_mhc.prev_coeff_c");
  if (s != Status::kSuccess) return s;
  const int64_t n = next_layer.n_hc;
  if (prev_coeff_b.shape[0] != next_layer.num_tokens || prev_coeff_b.shape[1] != n ||
      prev_coeff_b.shape[2] != n || prev_coeff_c.shape[0] != next_layer.num_tokens ||
      prev_coeff_c.shape[1] != n || prev_coeff_c.shape[2] != 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: prev coefficient shapes must be [T,n,n] and [T,n,1]", who);
  }
  s = check_tensor(prev_y, next_layer.residual_dtype, 2, 1, "mega_mhc.prev_y");
  if (s != Status::kSuccess) return s;
  if (prev_y.shape[0] != next_layer.num_tokens || prev_y.shape[1] != next_layer.d ||
      prev_y.stride[1] != 1 || prev_y.stride[0] < next_layer.d) {
    return detail::set_error(Status::kInvalidArgument, "%s: prev_y must be [T,d] contiguous", who);
  }
  s = hc_check_quant(quant, layer_in, next_layer, who);
  if (s != Status::kSuccess) return s;
  DType out_dtype = DType::kFloat32;
  if (tensor_present(layer_in)) {
    s = hc_check_layer_in(layer_in, next_layer, &out_dtype, "mega_mhc.layer_in");
    if (s != Status::kSuccess) return s;
  }
  if (next_layer.rmsnorm_weight != nullptr && !is_stream_dtype(next_layer.rmsnorm_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: rmsnorm dtype %s", who,
                             dtype_name(next_layer.rmsnorm_dtype));
  }
  if (tensor_present(sub_norm_weight)) {
    s = check_tensor(sub_norm_weight, DType::kInvalid, 1, 1, "mega_mhc.sub_norm_weight");
    if (s != Status::kSuccess) return s;
    if (sub_norm_weight.shape[0] != next_layer.d) {
      return detail::set_error(Status::kInvalidArgument, "%s: sub_norm_weight len %lld != d %d",
                               who, static_cast<long long>(sub_norm_weight.shape[0]),
                               next_layer.d);
    }
    if (sub_norm_weight.dtype != next_layer.rmsnorm_dtype) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "%s: sub_norm_weight dtype %s must equal rmsnorm_dtype %s", who,
                               dtype_name(sub_norm_weight.dtype),
                               dtype_name(next_layer.rmsnorm_dtype));
    }
  }
  const HCOutView out = hc_out_view(next_layer);
  s = hc_check_pack(next_layer, out, prev_y.data, prev_y.stride[0], who);
  if (s != Status::kSuccess) return s;
  const bool staged = static_cast<int64_t>(next_layer.d) * static_cast<int64_t>(sizeof(float)) <=
                      kStageCapBytes;
  const void* sub_w = tensor_present(sub_norm_weight) ? sub_norm_weight.data : nullptr;

  switch (next_layer.residual_dtype) {
    case DType::kFloat32:
      return launch_mega_out<float>(next_layer, static_cast<const float*>(prev_coeff_b.data),
                                    static_cast<const float*>(prev_coeff_c.data), prev_y.data,
                                    layer_in, quant, sub_w, sub_norm_eps, staged, ctx);
    case DType::kFloat16:
      return launch_mega_out<__half>(next_layer, static_cast<const float*>(prev_coeff_b.data),
                                     static_cast<const float*>(prev_coeff_c.data), prev_y.data,
                                     layer_in, quant, sub_w, sub_norm_eps, staged, ctx);
    case DType::kBFloat16:
      return launch_mega_out<__nv_bfloat16>(
          next_layer, static_cast<const float*>(prev_coeff_b.data),
          static_cast<const float*>(prev_coeff_c.data), prev_y.data, layer_in, quant, sub_w,
          sub_norm_eps, staged, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "%s: residual dtype %s", who,
                               dtype_name(next_layer.residual_dtype));
  }
}

Status hc_expand(const HCStreamParams& params, const Tensor& embed, const LaunchContext& ctx) {
  const char* who = "hc_expand";
  if (params.streams == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: streams is null", who);
  }
  if (params.num_tokens <= 0 || params.n_hc <= 0 || params.d <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: num_tokens %d, n_hc %d, d %d must all be > 0", who,
                             params.num_tokens, params.n_hc, params.d);
  }
  if (!is_n_hc(params.n_hc)) {
    return detail::set_error(Status::kUnsupportedShape, "%s: n_hc %d not in {1, 2, 4}", who,
                             params.n_hc);
  }
  if (!is_stream_dtype(params.stream_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: stream dtype %s", who,
                             dtype_name(params.stream_dtype));
  }
  const int64_t packed = static_cast<int64_t>(params.n_hc) * params.d;
  if (params.stream_token_stride != 0 && params.stream_token_stride != packed) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: streams must be packed ([T,n_hc,d]); token stride %lld != %lld",
                             who, static_cast<long long>(params.stream_token_stride),
                             static_cast<long long>(packed));
  }
  Status s = check_tensor(embed, params.stream_dtype, 2, 1, "hc_expand.embed");
  if (s != Status::kSuccess) return s;
  if (embed.shape[0] != params.num_tokens || embed.shape[1] != params.d ||
      embed.stride[1] != 1 || embed.stride[0] < params.d) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: embed must be [T,d] contiguous rows", who);
  }
  const int64_t elem = params.stream_dtype == DType::kFloat32 ? 4 : 2;
  const int64_t v = 16 / elem;
  const int vec = (params.d % v == 0 && embed.stride[0] % v == 0 && aligned16(params.streams) &&
                   aligned16(embed.data))
                      ? static_cast<int>(v)
                      : 1;
  switch (params.stream_dtype) {
    case DType::kFloat32: return launch_expand<float>(params, embed.data, vec, ctx);
    case DType::kFloat16: return launch_expand<__half>(params, embed.data, vec, ctx);
    case DType::kBFloat16: return launch_expand<__nv_bfloat16>(params, embed.data, vec, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "%s: stream dtype %s", who,
                               dtype_name(params.stream_dtype));
  }
}

Status hc_head(const Tensor& out, const HCStreamParams& params, const Tensor& head_weights,
               const LaunchContext& ctx) {
  const char* who = "hc_head";
  if (params.streams == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: streams is null", who);
  }
  if (params.num_tokens <= 0 || params.n_hc <= 0 || params.d <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: num_tokens %d, n_hc %d, d %d must all be > 0", who,
                             params.num_tokens, params.n_hc, params.d);
  }
  if (!is_n_hc(params.n_hc)) {
    return detail::set_error(Status::kUnsupportedShape, "%s: n_hc %d not in {1, 2, 4}", who,
                             params.n_hc);
  }
  if (!is_stream_dtype(params.stream_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: stream dtype %s", who,
                             dtype_name(params.stream_dtype));
  }
  const int64_t packed = static_cast<int64_t>(params.n_hc) * params.d;
  if (params.stream_token_stride != 0 && params.stream_token_stride != packed) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: streams must be packed ([T,n_hc,d]); token stride %lld != %lld",
                             who, static_cast<long long>(params.stream_token_stride),
                             static_cast<long long>(packed));
  }
  Status s = check_tensor(out, DType::kInvalid, 2, 1, "hc_head.out");
  if (s != Status::kSuccess) return s;
  if (!is_stream_dtype(out.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: out dtype %s", who,
                             dtype_name(out.dtype));
  }
  if (out.shape[0] != params.num_tokens || out.shape[1] != params.d || out.stride[1] != 1 ||
      out.stride[0] < params.d) {
    return detail::set_error(Status::kInvalidArgument, "%s: out must be [T,d] contiguous rows", who);
  }
  const float* w = nullptr;
  bool per_token = false;
  if (tensor_present(head_weights)) {
    s = check_tensor(head_weights, DType::kFloat32, -1, 4, "hc_head.head_weights");
    if (s != Status::kSuccess) return s;
    if (head_weights.ndim == 2) {
      if (head_weights.shape[0] != params.num_tokens || head_weights.shape[1] != params.n_hc ||
          head_weights.stride[1] != 1 || head_weights.stride[0] < params.n_hc) {
        return detail::set_error(Status::kInvalidArgument,
                                 "%s: per-token weights must be [T,n] contiguous rows", who);
      }
      per_token = true;
    } else if (head_weights.ndim == 1) {
      if (head_weights.shape[0] != params.n_hc || head_weights.stride[0] != 1) {
        return detail::set_error(Status::kInvalidArgument, "%s: static weights must be [n]", who);
      }
    } else {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: head_weights must be [T,n], [n] or absent", who);
    }
    w = static_cast<const float*>(head_weights.data);
  }
  const int64_t elem = params.stream_dtype == DType::kFloat32 ? 4 : 2;
  const int64_t v = 16 / elem;
  const int vec = (params.d % v == 0 && out.stride[0] % v == 0 && aligned16(params.streams) &&
                   aligned16(out.data))
                      ? static_cast<int>(v)
                      : 1;
  switch (params.stream_dtype) {
    case DType::kFloat32: return launch_head_out<float>(params, out, w, per_token, vec, ctx);
    case DType::kFloat16: return launch_head_out<__half>(params, out, w, per_token, vec, ctx);
    case DType::kBFloat16:
      return launch_head_out<__nv_bfloat16>(params, out, w, per_token, vec, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "%s: stream dtype %s", who,
                               dtype_name(params.stream_dtype));
  }
}

}  // namespace kops::model_ops
