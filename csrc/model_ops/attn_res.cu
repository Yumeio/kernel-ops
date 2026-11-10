// =============================================================================
// kernel-ops | csrc/model_ops/attn_res.cu
// X01 / B15: attention residuals (AttnRes; Kimi K3 uses it as "Block AttnRes").
//
// One CTA per token row. A row is mixed over `depth` vectors in this order:
//   completed block residuals 0 .. num_sources-1   (the `sources` stack)
//   the running partial residual                   (when include_partial)
// with
//   partial  <- partial + delta            fused residual update (fp32), if delta
//   logit_i   = (q . v_i) / rms(v_i)       q_j = query_j * key_norm_weight_j
//   h         = sum_i softmax_i(logit) * v_i
//   out       = h                          (raw mix)
//   normed    = RMSNorm(h) * out_norm_weight
//   payload   = MXFP8(normed)              e4m3 + one UE8M0 scale per 32 channels
//
// The inverse RMS of a completed source is either read from the fp32 cache that
// attn_res_commit_kernel writes ([max_sources, num_tokens]) or recomputed from
// the row's sum of squares. The partial row is always computed on the fly: it
// changes with every delta, so a cached value would be stale. Every reduction is
// fp32; the element type only decides how the operands are stored.
//
// Two passes over the sources (stats, then the weighted sum) keep the working
// set in registers instead of caching every source row: caching would need
// depth * d * 4 B, 258 KB at Kimi K3's H=7168 / 8 blocks + partial, while the
// two-pass form needs d * 4 B only when the fused output norm/quant is requested
// and nothing when only `out` is asked for. Issued loads per row are
// 2 * depth * d * sizeof(T) (the second pass normally hits L2, because a CTA
// walks one row's sources back to back) + 2 * d * sizeof(T) for the delta update
// + d * sizeof(T) per emitted output (+ d bytes for the MX payload).
//
// Determinism / batch invariance: one CTA owns a whole row, every reduction is
// an intra-warp shuffle plus a fixed-order sum over the 8 warps, sources are
// visited in increasing index, there is no atomic and no allocation on the
// launch path. A row therefore produces bit-identical out / normed_out /
// alpha_out alone or inside any batch, and a graph replay reproduces the bits.
//
// Alias: one thread reads and writes the same column, so an exact in-place view
// is safe for every output (out / normed_out may be the row they mix) and for
// the partial (partial += delta, and commit's restart_with). Two partially
// overlapping rows are not supported: the host has only element strides and
// cannot detect that case, so it is the caller's responsibility.
//
// Degenerate rows: an empty source set (num_sources == 0 and include_partial
// false) or a set whose logits are all non-finite (a NaN cached inverse RMS,
// say) yields alpha = 0 and an all-zero `h` / `normed_out` (and zero payload
// codes) instead of NaN. Sources with weight exactly zero are not even read in
// the mix pass, so a NaN in a masked-out source cannot leak into the output.
//
// Layout: `sources` is [sources, tokens, d] with independent source and token
// strides, so both `[blocks, T, H]` (source_stride = T*d) and K3's token-major
// `[T, blocks, H]` (source_stride = d, source_token_stride = blocks*d) describe
// the same stack. Rows are contiguous along d. 16 B vector loads run when d and
// every host-visible stride are multiples of the pack and every base pointer is
// 16 B aligned; otherwise the scalar path runs with the same fp32 expression.
// The fused output norm caches h in dynamic shared memory (d * 4 B, plus
// d / 32 * 4 B for the MX scales); a row must fit the 48 KB default dynamic
// limit, so d <= 12288 without the scale cache, which covers every K3 shape.
//
// Reference semantics: 3rd/vllm/csrc/kimi_k3/attn_res_kernel.cu (commit 07aea9d,
// SM100 tcgen05 body) and vllm/_custom_ops.py kimi_k3_attn_res. This is the
// generic CUDA-core body; H=7168 is a supported and tested point, not a
// requirement, and no performance parity with the tcgen05 body is claimed.
// =============================================================================
#include <cstdint>

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
#include "csrc/model_ops/residual_kernels.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace kops::model_ops {
namespace {

using dev::Pack;
using dev::from_f;
using dev::to_f;

/// One CTA per token row; every reduction below is sized from this.
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
/// Completed block residuals; the partial row is one more (kMaxDepth).
constexpr int kMaxSources = 8;
constexpr int kMaxDepth = kMaxSources + 1;
/// exp2-based softmax, as in the reference body.
constexpr float kLog2e = 1.4426950408889634f;
/// e4m3 finite max (common/dtype.cuh kFp8E4M3Max) and the smallest amax that
/// may drive a scale, so a zero row still produces a finite (zero) payload.
constexpr float kQuantFloor = 1e-10f;
/// Dynamic shared memory the fused output norm may request without opting into
/// a larger carve-out (the CUDA default per-block dynamic limit).
constexpr size_t kMaxFusedSmem = 48u * 1024u;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

/// fp32 -> e4m3fn payload, round-to-nearest-even, saturating at +-448.
/// sm_86 has no FP8 convert instruction, so the bit-level form in
/// common/device_impl.cuh is used on every target (it is the same code the
/// toolkit intrinsic expands to, and test_gemm.cc pins it against the toolkit).
__device__ __forceinline__ uint8_t mx_e4m3(float x) { return dev::e4m3_from_float_sat(x); }

/// Positive fp32 -> UE8M0 exponent byte, rounded UP to the next power of two so
/// that amax / scale <= 448 (the OCP MX recipe Kimi K3 uses). 0, negative and
/// non-finite inputs collapse to the smallest representable exponent.
__device__ __forceinline__ uint8_t mx_ue8m0(float s) {
  int e = -127;
  if (s > 0.0f && isfinite(s)) {
    int ex = 0;
    frexpf(s, &ex);   // s = m * 2^ex with 0.5 <= m < 1
    e = ex - 1;       // 2^e <= s < 2^(e+1)
    if (ldexpf(1.0f, e) < s) ++e;
    if (e > 127) e = 127;
  }
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float mx_ue8m0_to_float(uint8_t e) {
  return ldexpf(1.0f, static_cast<int>(e) - 127);
}

/// RMSNorm(h) * weight for one chunk of the row: `s_h` holds h in fp32 and is
/// overwritten with the normed value, `dst` receives the storage-dtype copy.
template <typename scalar_t, int VEC>
__device__ __forceinline__ void norm_chunk(float* __restrict__ s_h, const scalar_t* __restrict__ w,
                                           scalar_t* __restrict__ dst, const int64_t col,
                                           const float rstd) {
  const Pack<scalar_t, VEC> wv = dev::load_pack<VEC>(w + col);
  Pack<scalar_t, VEC> nv;
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    const float y = s_h[col + k] * rstd * to_f(wv.x[k]);
    s_h[col + k] = y;
    from_f(nv.x[k], y);
  }
  if (dst != nullptr) dev::store_pack<VEC>(dst + col, nv);
}

/// Pass B: h = sum_i alpha_i * v_i for one chunk of the row.
template <typename scalar_t, int DEPTH, int VEC>
__device__ __forceinline__ void mix_chunk(const scalar_t* const (&rows)[DEPTH], const int depth,
                                          const float (&alpha)[DEPTH], const int64_t col,
                                          float (&acc)[VEC]) {
#pragma unroll
  for (int k = 0; k < VEC; ++k) acc[k] = 0.0f;
#pragma unroll
  for (int i = 0; i < DEPTH; ++i) {
    // alpha is block-uniform, so the skip is uniform too: a masked source is
    // never read and its contents (even NaN) cannot reach the output.
    if (i < depth && alpha[i] != 0.0f) {
      const Pack<scalar_t, VEC> v = dev::load_pack<VEC>(rows[i] + col);
#pragma unroll
      for (int k = 0; k < VEC; ++k) acc[k] = fmaf(alpha[i], to_f(v.x[k]), acc[k]);
    }
  }
}

/// Per-source (sum of squares, dot with the folded pseudo-query) for one chunk.
template <typename scalar_t, int DEPTH, int VEC>
__device__ __forceinline__ void stats_chunk(const scalar_t* const (&rows)[DEPTH], const int depth,
                                            const float (&q)[VEC], const int64_t col,
                                            float (&sq)[DEPTH], float (&dt)[DEPTH]) {
#pragma unroll
  for (int i = 0; i < DEPTH; ++i) {
    if (i < depth) {
      const Pack<scalar_t, VEC> v = dev::load_pack<VEC>(rows[i] + col);
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const float x = to_f(v.x[k]);
        sq[i] = fmaf(x, x, sq[i]);
        dt[i] = fmaf(x, q[k], dt[i]);
      }
    }
  }
}

}  // namespace

/**
 * attn_res_kernel — one CTA per token row (see the file comment for the math and
 * the layout). `MAX_SOURCES` is the compiled cap on completed sources; the
 * partial row adds one more slot. `kQuant` selects the fused MXFP8 epilogue.
 */
template <typename scalar_t, int MAX_SOURCES, int VEC_SIZE, ActQuantMode kQuant>
__global__ __launch_bounds__(kBlock) void attn_res_kernel(
    const __grid_constant__ AttnResParams params) {
  constexpr int kVec = VEC_SIZE;
  constexpr int kD = MAX_SOURCES + 1;
  static_assert(kD <= kMaxDepth, "AttnRes: at most 8 completed sources plus the partial row");
  static_assert(kVec == 1 || kVec == static_cast<int>(16 / sizeof(scalar_t)),
                "AttnRes: the vector path moves one 16 B pack");

  __shared__ float s_red[2 * kD][kWarps];
  __shared__ float s_sum[2 * kD];
  __shared__ float s_bred[kWarps];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t token = blockIdx.x;
  const int64_t d = params.d;
  const int64_t chunks = d / kVec;
  const int num_sources = params.num_sources;
  const int depth = num_sources + (params.include_partial ? 1 : 0);
  const float inv_d = 1.0f / static_cast<float>(d);

  auto* partial = static_cast<scalar_t*>(params.partial);
  const auto* sources = static_cast<const scalar_t*>(params.sources);

  // Row pointers of the two source families. Constant indices (the loops are
  // fully unrolled) keep the table in registers.
  const scalar_t* rows[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    if (i < num_sources) {
      rows[i] = sources + static_cast<int64_t>(i) * params.source_stride +
                token * params.source_token_stride;
    } else if (i == num_sources && params.include_partial) {
      rows[i] = partial + token * params.partial_stride;
    } else {
      rows[i] = nullptr;
    }
  }

  // ---- fused residual update: partial += delta ---------------------------------
  if (params.delta != nullptr) {
    const auto* delta = static_cast<const scalar_t*>(params.delta) + token * params.delta_stride;
    scalar_t* prow = partial + token * params.partial_stride;
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      const int64_t col = c * kVec;
      Pack<scalar_t, kVec> pv = dev::load_pack<kVec>(prow + col);
      const Pack<scalar_t, kVec> dv = dev::load_pack<kVec>(delta + col);
#pragma unroll
      for (int k = 0; k < kVec; ++k) from_f(pv.x[k], to_f(pv.x[k]) + to_f(dv.x[k]));
      dev::store_pack<kVec>(prow + col, pv);
    }
  }
  __syncthreads();

  // ---- pass A: per-source sum of squares and dot with the folded query ---------
  float sq[kD], dt[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    sq[i] = 0.0f;
    dt[i] = 0.0f;
  }
  if (depth > 0) {
    const auto* query = static_cast<const scalar_t*>(params.query);
    const auto* knw = static_cast<const scalar_t*>(params.key_norm_weight);
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      const int64_t col = c * kVec;
      float q[kVec];
      const Pack<scalar_t, kVec> qp = dev::load_pack<kVec>(query + col);
      if (knw != nullptr) {
        const Pack<scalar_t, kVec> wp = dev::load_pack<kVec>(knw + col);
#pragma unroll
        for (int k = 0; k < kVec; ++k) q[k] = to_f(qp.x[k]) * to_f(wp.x[k]);
      } else {
#pragma unroll
        for (int k = 0; k < kVec; ++k) q[k] = to_f(qp.x[k]);
      }
      stats_chunk<scalar_t, kD, kVec>(rows, depth, q, col, sq, dt);
    }
  }

  // Fixed-shape reduction: warp shuffles, then a sum over the 8 warp slots in
  // increasing warp order. Every thread ends up with the same totals.
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    float a = (i < depth) ? sq[i] : 0.0f;
    float b = (i < depth) ? dt[i] : 0.0f;
    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
      s_red[2 * i][warp] = a;
      s_red[2 * i + 1][warp] = b;
    }
  }
  __syncthreads();
  if (tid < 2 * depth) {
    float t = s_red[tid][0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) t += s_red[tid][w];
    s_sum[tid] = t;
  }
  __syncthreads();

  // ---- softmax over depth ------------------------------------------------------
  float alpha[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i) alpha[i] = 0.0f;
  if (depth > 0) {
    const float* rinv = params.source_rinv;   // [max_sources, num_tokens], may be null
    float logit[kD];
    float m = neg_inf();
#pragma unroll
    for (int i = 0; i < kD; ++i) {
      if (i < depth) {
        const float r = (i < num_sources && rinv != nullptr)
                            ? rinv[static_cast<int64_t>(i) * params.num_tokens + token]
                            : rsqrtf(s_sum[2 * i] * inv_d + params.key_eps);
        const float l = s_sum[2 * i + 1] * r;
        logit[i] = isfinite(l) ? l : neg_inf();
      } else {
        logit[i] = neg_inf();
      }
      m = fmaxf(m, logit[i]);
    }
    const bool any = isfinite(m);
    float denom = 0.0f;
#pragma unroll
    for (int i = 0; i < kD; ++i) {
      const float e = (any && logit[i] != neg_inf()) ? exp2f((logit[i] - m) * kLog2e) : 0.0f;
      alpha[i] = isfinite(e) ? e : 0.0f;
      denom += alpha[i];
    }
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
#pragma unroll
    for (int i = 0; i < kD; ++i) alpha[i] *= inv_denom;
  }

  // Debug / analysis coefficients, laid out [num_tokens, num_sources + 1] with
  // the partial weight in the last column (0 when it is not a source here).
  if (params.alpha_out != nullptr && tid == 0) {
    float* arow = params.alpha_out + token * (static_cast<int64_t>(num_sources) + 1);
#pragma unroll
    for (int i = 0; i < kD; ++i) {
      if (i <= num_sources) arow[i] = (i < depth) ? alpha[i] : 0.0f;
    }
  }

  // ---- pass B: weighted sum, raw output, optional fp32 cache -------------------
  scalar_t* out_row = params.out != nullptr
                          ? static_cast<scalar_t*>(params.out) + token * params.out_stride
                          : nullptr;
  // The header carries one output row stride; normed_out reuses it.
  scalar_t* normed_row = params.normed_out != nullptr
                             ? static_cast<scalar_t*>(params.normed_out) + token * params.out_stride
                             : nullptr;
  constexpr bool kQuantize = (kQuant != ActQuantMode::kNone);
  const bool need_norm = (normed_row != nullptr) || kQuantize;
  extern __shared__ float s_h[];   // d floats when need_norm, empty otherwise
  float* s_scale = nullptr;
  if constexpr (kQuantize) s_scale = s_h + d;   // d / 32 fp32 group scales

  float sumsq = 0.0f;
  if (out_row != nullptr || need_norm) {
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      const int64_t col = c * kVec;
      float acc[kVec];
      mix_chunk<scalar_t, kD, kVec>(rows, depth, alpha, col, acc);
      if (out_row != nullptr) {
        Pack<scalar_t, kVec> ov;
#pragma unroll
        for (int k = 0; k < kVec; ++k) from_f(ov.x[k], acc[k]);
        dev::store_pack<kVec>(out_row + col, ov);
      }
      if (need_norm) {
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          s_h[col + k] = acc[k];
          sumsq = fmaf(acc[k], acc[k], sumsq);
        }
      }
    }
  }
  if (!need_norm) return;

  const auto* nw = static_cast<const scalar_t*>(params.out_norm_weight);
  const float rstd =
      rsqrtf(block_reduce_sum<float, kWarps>(sumsq, s_bred) * inv_d + params.out_norm_eps);

  if constexpr (kQuantize) {
    constexpr int kGroup = kMxBlock;   // 32 channels per MX scale
    // Stage 1: normed value -> normed_out and the fp32 cache in s_h. One chunk
    // per thread per pass, so no lane-group shuffle is needed (a short row can
    // leave part of a warp idle, and a partial-warp reduction would deadlock).
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      const int64_t col = c * kVec;
      const Pack<scalar_t, kVec> wv = dev::load_pack<kVec>(nw + col);
      Pack<scalar_t, kVec> nv;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float y = s_h[col + k] * rstd * to_f(wv.x[k]);
        s_h[col + k] = y;
        from_f(nv.x[k], y);
      }
      if (normed_row != nullptr) dev::store_pack<kVec>(normed_row + col, nv);
    }
    __syncthreads();
    // Stage 2: one MX group per thread: amax -> UE8M0 scale (round up to a power
    // of two, so amax / scale <= 448 by construction). The scale is written both
    // to the caller's row-major scale array and to the shared cache stage 3 reads.
    const int64_t groups = d / kGroup;
    auto* scales = static_cast<float*>(params.normed_quant.scales) +
                   token * params.normed_quant.scale_row_stride;
    for (int64_t g = tid; g < groups; g += blockDim.x) {
      float amax = 0.0f;
      for (int j = 0; j < kGroup; ++j) amax = fmaxf(amax, fabsf(s_h[g * kGroup + j]));
      const float s = mx_ue8m0_to_float(mx_ue8m0(fmaxf(amax, kQuantFloor) / kFp8E4M3Max));
      s_scale[g] = s;
      scales[g] = s;
    }
    __syncthreads();
    // Stage 3: payload = round(y / scale); the scale is a power of two, so the
    // division is exact and independent of the value's magnitude.
    auto* payload = static_cast<uint8_t*>(params.normed_quant.q) +
                    token * params.normed_quant.q_row_stride;
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      const int64_t col = c * kVec;
      const float s = s_scale[col / kGroup];
#pragma unroll
      for (int k = 0; k < kVec; ++k) payload[col + k] = mx_e4m3(__fdiv_rn(s_h[col + k], s));
    }
  } else {
    // No quantization: one pass writes normed_out (s_h itself is not reused).
    for (int64_t c = tid; c < chunks; c += blockDim.x) {
      norm_chunk<scalar_t, kVec>(s_h, nw, normed_row, c * kVec, rstd);
    }
  }
}

/**
 * attn_res_commit_kernel — one CTA per token row. Copies partial into slot
 * `slot` of the source stack, caches 1 / sqrt(mean(partial^2) + eps) for that
 * slot in fp32, then resets partial to `restart_with` (or zero).
 *
 * Numerics: the copy is a pure move in the storage dtype, the inverse RMS is
 * fp32 over the same values, and the reset happens after the copy in the same
 * thread, so a row's cached rinv always describes the vector it stored.
 */
template <typename scalar_t, int VEC_SIZE>
__global__ __launch_bounds__(kBlock) void attn_res_commit_kernel(
    void* __restrict__ sources, const int64_t source_stride,
    float* __restrict__ source_rinv,
    void* __restrict__ partial,
    const void* __restrict__ restart_with,
    const int slot, const float eps, const int num_tokens, const int d) {
  constexpr int kVec = VEC_SIZE;
  static_assert(kVec == 1 || kVec == static_cast<int>(16 / sizeof(scalar_t)),
                "AttnRes commit: the vector path moves one 16 B pack");

  __shared__ float s_bred[kWarps];

  const int64_t token = blockIdx.x;
  const int64_t chunks = d / kVec;
  auto* dst = static_cast<scalar_t*>(sources) + static_cast<int64_t>(slot) * source_stride +
              token * d;
  scalar_t* prow = static_cast<scalar_t*>(partial) + token * d;
  const auto* restart =
      restart_with != nullptr ? static_cast<const scalar_t*>(restart_with) + token * d : nullptr;
  const float inv_d = 1.0f / static_cast<float>(d);

  float sumsq = 0.0f;
  for (int64_t c = threadIdx.x; c < chunks; c += blockDim.x) {
    const int64_t col = c * kVec;
    const Pack<scalar_t, kVec> v = dev::load_pack<kVec>(prow + col);
    dev::store_pack<kVec>(dst + col, v);
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const float x = to_f(v.x[k]);
      sumsq = fmaf(x, x, sumsq);
    }
  }
  const float total = block_reduce_sum<float, kWarps>(sumsq, s_bred);
  if (threadIdx.x == 0) {
    source_rinv[static_cast<int64_t>(slot) * num_tokens + token] =
        rsqrtf(total * inv_d + eps);
  }
  // Reset after the copy: the cached rinv and the stored vector stay paired.
  for (int64_t c = threadIdx.x; c < chunks; c += blockDim.x) {
    const int64_t col = c * kVec;
    if (restart != nullptr) {
      dev::store_pack<kVec>(prow + col, dev::load_pack<kVec>(restart + col));
    } else {
      Pack<scalar_t, kVec> z;
#pragma unroll
      for (int k = 0; k < kVec; ++k) from_f(z.x[k], 0.0f);
      dev::store_pack<kVec>(prow + col, z);
    }
  }
}

// -----------------------------------------------------------------------------
// Host side
// -----------------------------------------------------------------------------

namespace {

int64_t element_bytes(DType t) { return (dtype_bits(t) + 7) / 8; }

bool is_supported_float(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

/// Elements of `dtype` in one 16 B pack (the widest vector access).
int pack_elements(DType dtype) {
  const int bytes = static_cast<int>(element_bytes(dtype));
  return bytes > 0 ? 16 / bytes : 1;
}

/// Smallest compiled source cap that covers `n` (0..kMaxSources).
int source_cap(int n) { return n <= 1 ? 1 : (n <= 2 ? 2 : (n <= 4 ? 4 : 8)); }

bool aligned16(const void* p) { return p == nullptr || is_aligned(p, 16); }

/// True when every accessed row start and base pointer is 16 B aligned, so the
/// 16 B pack path is legal. The scalar path accepts anything.
bool vector_path_ok(const AttnResParams& p, DType dtype) {
  const int pack = pack_elements(dtype);
  if (pack <= 1 || p.d % pack != 0) return false;
  if (!aligned16(p.sources) || !aligned16(p.partial) || !aligned16(p.delta) ||
      !aligned16(p.out) || !aligned16(p.normed_out) || !aligned16(p.query) ||
      !aligned16(p.key_norm_weight) || !aligned16(p.out_norm_weight)) {
    return false;
  }
  if (p.num_sources > 0 &&
      (p.source_stride % pack != 0 || p.source_token_stride % pack != 0)) {
    return false;
  }
  if (p.partial != nullptr && p.partial_stride % pack != 0) return false;
  if (p.delta != nullptr && p.delta_stride % pack != 0) return false;
  if ((p.out != nullptr || p.normed_out != nullptr) && p.out_stride % pack != 0) return false;
  return true;
}

template <typename scalar_t, int MAX_SOURCES, int VEC_SIZE>
Status launch_attn_res(const AttnResParams& p, bool quant, size_t smem, cudaStream_t stream) {
  if (quant) {
    attn_res_kernel<scalar_t, MAX_SOURCES, VEC_SIZE, ActQuantMode::kMXFP8>
        <<<p.num_tokens, kBlock, smem, stream>>>(p);
  } else {
    attn_res_kernel<scalar_t, MAX_SOURCES, VEC_SIZE, ActQuantMode::kNone>
        <<<p.num_tokens, kBlock, smem, stream>>>(p);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "attn_res launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename scalar_t>
Status dispatch_attn_res(const AttnResParams& p, bool quant, int vector, size_t smem,
                         cudaStream_t stream) {
  const int cap = source_cap(p.num_sources);
  const int pack = static_cast<int>(16 / sizeof(scalar_t));
#define KOPS_ATTN_RES_VEC(V)                                                                        \
  do {                                                                                              \
    if (cap == 1) return launch_attn_res<scalar_t, 1, V>(p, quant, smem, stream);                    \
    if (cap == 2) return launch_attn_res<scalar_t, 2, V>(p, quant, smem, stream);                    \
    if (cap == 4) return launch_attn_res<scalar_t, 4, V>(p, quant, smem, stream);                    \
    return launch_attn_res<scalar_t, 8, V>(p, quant, smem, stream);                                  \
  } while (0)
  if (vector == pack) {
    KOPS_ATTN_RES_VEC(pack);
  } else {
    KOPS_ATTN_RES_VEC(1);
  }
#undef KOPS_ATTN_RES_VEC
}

template <typename scalar_t, int VEC_SIZE>
Status launch_attn_res_commit(void* sources, int64_t source_stride, float* source_rinv,
                              void* partial, const void* restart_with, int32_t slot, float eps,
                              int32_t num_tokens, int32_t d, cudaStream_t stream) {
  attn_res_commit_kernel<scalar_t, VEC_SIZE><<<num_tokens, kBlock, 0, stream>>>(
      sources, source_stride, source_rinv, partial, restart_with, slot, eps, num_tokens, d);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "attn_res_commit launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename scalar_t>
Status dispatch_attn_res_commit(void* sources, int64_t source_stride, float* source_rinv,
                                void* partial, const void* restart_with, int32_t slot, float eps,
                                int32_t num_tokens, int32_t d, bool vector, cudaStream_t stream) {
  const int pack = static_cast<int>(16 / sizeof(scalar_t));
  if (vector) {
    return launch_attn_res_commit<scalar_t, pack>(sources, source_stride, source_rinv, partial,
                                                  restart_with, slot, eps, num_tokens, d, stream);
  }
  return launch_attn_res_commit<scalar_t, 1>(sources, source_stride, source_rinv, partial,
                                             restart_with, slot, eps, num_tokens, d, stream);
}

}  // namespace

Status attn_res(const AttnResParams& params, DType dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "attn_res: CUDA backend only");
  }
  if (!is_supported_float(dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "attn_res: dtype %s", dtype_name(dtype));
  }
  if (params.num_sources < 0 || params.num_sources > kMaxSources) {
    return detail::set_error(Status::kUnsupportedShape,
                             "attn_res: num_sources %d outside 0..%d", params.num_sources,
                             kMaxSources);
  }
  if (params.d <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "attn_res: d %d", params.d);
  }
  if (params.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument, "attn_res: num_tokens %d",
                             params.num_tokens);
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  if (params.num_tokens > INT32_MAX || params.d > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape,
                             "attn_res: num_tokens %d, d %d exceed int32", params.num_tokens,
                             params.d);
  }
  const int num_sources = params.num_sources;
  const int depth = num_sources + (params.include_partial ? 1 : 0);

  // ---- source stack ------------------------------------------------------------
  if (num_sources > 0) {
    if (params.sources == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "attn_res: sources is null with %d sources",
                               num_sources);
    }
    if (params.source_stride < params.d || params.source_token_stride < params.d) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "attn_res: source strides %lld / %lld smaller than d %d",
                               static_cast<long long>(params.source_stride),
                               static_cast<long long>(params.source_token_stride), params.d);
    }
  }
  // ---- running partial ---------------------------------------------------------
  if (params.delta != nullptr && params.partial == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "attn_res: delta needs partial (partial += delta)");
  }
  if (params.include_partial && params.partial == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "attn_res: include_partial needs partial");
  }
  if (params.partial != nullptr && params.partial_stride < params.d) {
    return detail::set_error(Status::kUnsupportedLayout, "attn_res: partial_stride %lld < d %d",
                             static_cast<long long>(params.partial_stride), params.d);
  }
  if (params.delta != nullptr && params.delta_stride < params.d) {
    return detail::set_error(Status::kUnsupportedLayout, "attn_res: delta_stride %lld < d %d",
                             static_cast<long long>(params.delta_stride), params.d);
  }
  // ---- pseudo-query ------------------------------------------------------------
  if (depth > 0 && params.query == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "attn_res: query is null with depth %d",
                             depth);
  }
  if (!(params.key_eps >= 0.0f) || !isfinite(params.key_eps)) {
    return detail::set_error(Status::kInvalidArgument, "attn_res: key_eps %g", params.key_eps);
  }
  // ---- outputs -----------------------------------------------------------------
  const bool quant = params.normed_quant.mode != ActQuantMode::kNone;
  const bool need_norm = params.normed_out != nullptr || quant;
  if (params.out == nullptr && params.normed_out == nullptr && !quant &&
      params.alpha_out == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "attn_res: no output (out, normed_out, quant, alpha_out)");
  }
  if (params.out != nullptr || params.normed_out != nullptr) {
    const int64_t stride = params.out_stride;
    if (params.num_tokens > 1 && stride < params.d) {
      return detail::set_error(Status::kUnsupportedLayout, "attn_res: out_stride %lld < d %d",
                               static_cast<long long>(stride), params.d);
    }
    if (stride < 0) {
      return detail::set_error(Status::kUnsupportedLayout, "attn_res: out_stride %lld < 0",
                               static_cast<long long>(stride));
    }
  }
  if (need_norm) {
    if (params.out_norm_weight == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "attn_res: normed output needs out_norm_weight");
    }
    if (!(params.out_norm_eps >= 0.0f) || !isfinite(params.out_norm_eps)) {
      return detail::set_error(Status::kInvalidArgument, "attn_res: out_norm_eps %g",
                               params.out_norm_eps);
    }
  }
  // ---- optional MXFP8 epilogue -------------------------------------------------
  int64_t groups = 0;
  if (quant) {
    const ActQuantOut& q = params.normed_quant;
    if (q.mode != ActQuantMode::kMXFP8) {
      return detail::set_error(Status::kUnsupportedShape,
                               "attn_res: quant mode %d (only MXFP8 is implemented)",
                               static_cast<int>(q.mode));
    }
    if (q.q == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "attn_res: quant.q is null");
    }
    if (q.q_dtype != DType::kFloat8E4M3 || q.fp8 != Fp8Encoding::kOCP) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "attn_res: MXFP8 payload %s / encoding %d (needs e4m3 OCP)",
                               dtype_name(q.q_dtype), static_cast<int>(q.fp8));
    }
    if (params.d % kMxBlock != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "attn_res: MXFP8 needs d %% %d, d %d", kMxBlock, params.d);
    }
    if (q.scales == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "attn_res: quant.scales is null");
    }
    if (q.scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "attn_res: MXFP8 scale layout %d (row-major only)",
                               static_cast<int>(q.scale_layout));
    }
    groups = params.d / kMxBlock;
    if (q.q_row_stride < params.d) {
      return detail::set_error(Status::kUnsupportedLayout, "attn_res: q_row_stride %lld < d %d",
                               static_cast<long long>(q.q_row_stride), params.d);
    }
    if (q.scale_row_stride < groups) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "attn_res: scale_row_stride %lld < %lld groups",
                               static_cast<long long>(q.scale_row_stride),
                               static_cast<long long>(groups));
    }
  }

  // The fused output norm caches h in dynamic shared memory; stay inside the
  // default per-block limit instead of opting into a larger carve-out (that call
  // would have to happen outside graph capture).
  size_t smem = 0;
  if (need_norm) {
    smem = static_cast<size_t>(params.d) * sizeof(float) +
           static_cast<size_t>(groups) * sizeof(float);
    if (smem > kMaxFusedSmem) {
      return detail::set_error(Status::kUnsupportedShape,
                               "attn_res: fused output norm needs %zu B of shared memory "
                               "(d %d), limit %zu", smem, params.d, kMaxFusedSmem);
    }
  }

  const int vector = vector_path_ok(params, dtype) ? pack_elements(dtype) : 1;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  if (dtype == DType::kFloat32) {
    return dispatch_attn_res<float>(params, quant, vector, smem, stream);
  }
  if (dtype == DType::kFloat16) {
    return dispatch_attn_res<__half>(params, quant, vector, smem, stream);
  }
  return dispatch_attn_res<__nv_bfloat16>(params, quant, vector, smem, stream);
}

Status attn_res_commit(const Tensor& sources, const Tensor& source_rinv, const Tensor& partial,
                       const Tensor& restart_with, int32_t slot, float eps,
                       const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "attn_res_commit: CUDA backend only");
  }
  Status s = check_tensor(sources, DType::kInvalid, 3, 1, "attn_res_commit.sources");
  if (s != Status::kSuccess) return s;
  if (!is_supported_float(sources.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "attn_res_commit: sources dtype %s",
                             dtype_name(sources.dtype));
  }
  s = check_tensor(source_rinv, DType::kFloat32, 2, 1, "attn_res_commit.source_rinv");
  if (s != Status::kSuccess) return s;
  s = check_tensor(partial, sources.dtype, 2, 1, "attn_res_commit.partial");
  if (s != Status::kSuccess) return s;

  const int64_t num_slots = sources.shape[0];
  const int64_t num_tokens = sources.shape[1];
  const int64_t d = sources.shape[2];
  if (num_slots <= 0 || d <= 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "attn_res_commit: sources [%lld, %lld, %lld]",
                             static_cast<long long>(num_slots),
                             static_cast<long long>(num_tokens), static_cast<long long>(d));
  }
  if (slot < 0 || static_cast<int64_t>(slot) >= num_slots) {
    return detail::set_error(Status::kInvalidArgument, "attn_res_commit: slot %d outside 0..%lld",
                             slot, static_cast<long long>(num_slots - 1));
  }
  // The commit signature carries one slot stride and dense rows: the copy target
  // is sources[slot, token, :] with the token stride equal to d.
  if (sources.stride[2] != 1 || sources.stride[1] != d) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "attn_res_commit: sources must be [slots, T, d] with token stride d "
                             "(got %lld, d %lld)",
                             static_cast<long long>(sources.stride[1]),
                             static_cast<long long>(d));
  }
  if (sources.stride[0] < num_tokens * d) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "attn_res_commit: source slot stride %lld < T * d = %lld",
                             static_cast<long long>(sources.stride[0]),
                             static_cast<long long>(num_tokens * d));
  }
  if (partial.shape[0] != num_tokens || partial.shape[1] != d || partial.stride[1] != 1 ||
      partial.stride[0] != d) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "attn_res_commit: partial must be [T, d] dense (got [%lld, %lld], "
                             "stride %lld)",
                             static_cast<long long>(partial.shape[0]),
                             static_cast<long long>(partial.shape[1]),
                             static_cast<long long>(partial.stride[0]));
  }
  if (source_rinv.shape[0] <= slot || source_rinv.shape[1] != num_tokens ||
      source_rinv.stride[1] != 1 || source_rinv.stride[0] != num_tokens) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "attn_res_commit: source_rinv must be [slots, T] dense fp32 "
                             "(got [%lld, %lld], stride %lld)",
                             static_cast<long long>(source_rinv.shape[0]),
                             static_cast<long long>(source_rinv.shape[1]),
                             static_cast<long long>(source_rinv.stride[0]));
  }
  const void* restart = nullptr;
  if (tensor_present(restart_with)) {
    s = check_tensor(restart_with, sources.dtype, 2, 1, "attn_res_commit.restart_with");
    if (s != Status::kSuccess) return s;
    if (restart_with.shape[0] != num_tokens || restart_with.shape[1] != d ||
        restart_with.stride[1] != 1 || restart_with.stride[0] != d) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "attn_res_commit: restart_with must be [T, d] dense (got [%lld, "
                               "%lld], stride %lld)",
                               static_cast<long long>(restart_with.shape[0]),
                               static_cast<long long>(restart_with.shape[1]),
                               static_cast<long long>(restart_with.stride[0]));
    }
    restart = restart_with.data;
  }
  if (!(eps >= 0.0f) || !isfinite(eps)) {
    return detail::set_error(Status::kInvalidArgument, "attn_res_commit: eps %g", eps);
  }
  if (num_tokens == 0) return Status::kSuccess;
  if (num_tokens > INT32_MAX || d > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape,
                             "attn_res_commit: T %lld, d %lld exceed int32",
                             static_cast<long long>(num_tokens), static_cast<long long>(d));
  }

  auto* src = static_cast<char*>(sources.data) + sources.byte_offset;
  auto* dst = static_cast<char*>(partial.data) + partial.byte_offset;
  const void* reset =
      restart != nullptr
          ? static_cast<const void*>(static_cast<const char*>(restart) + restart_with.byte_offset)
          : nullptr;
  auto* rinv =
      reinterpret_cast<float*>(static_cast<char*>(source_rinv.data) + source_rinv.byte_offset);
  const int pack = pack_elements(sources.dtype);
  const bool vector = pack > 1 && d % pack == 0 && sources.stride[0] % pack == 0 &&
                      aligned16(src) && aligned16(dst) && aligned16(reset);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int32_t tokens = static_cast<int32_t>(num_tokens);
  const int32_t width = static_cast<int32_t>(d);
  if (sources.dtype == DType::kFloat32) {
    return dispatch_attn_res_commit<float>(src, sources.stride[0], rinv, dst, reset, slot, eps,
                                           tokens, width, vector, stream);
  }
  if (sources.dtype == DType::kFloat16) {
    return dispatch_attn_res_commit<__half>(src, sources.stride[0], rinv, dst, reset, slot, eps,
                                            tokens, width, vector, stream);
  }
  return dispatch_attn_res_commit<__nv_bfloat16>(src, sources.stride[0], rinv, dst, reset, slot, eps,
                                                 tokens, width, vector, stream);
}

}  // namespace kops::model_ops
