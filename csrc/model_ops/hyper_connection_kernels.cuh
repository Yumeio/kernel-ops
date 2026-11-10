#pragma once
// =============================================================================
// kernel-ops | csrc/model_ops/hyper_connection_kernels.cuh            Catalog: X01
// Hyper-Connections (HC, Zhu et al.), HPC-Ops iHC, and Manifold-Constrained
// Hyper-Connections (mHC, DeepSeek-V4/V4.1) incl. the single-pass fused variant.
//
// The residual stream is widened to n_hc copies: residual in R^{T x n x d}.
// The DeepSeek-V4 contract (DeepGEMM tests/test_mega_mhc.py, mhc_reference) is:
//   new_residual = x * post_mix + comb_res_mix^T * residual
//   mixes        = (flatten(new_residual) @ fn^T)
//                  * rsqrt(mean(new_residual^2) + hc_norm_eps)
//                  * mix_scales + mix_bases
//   pre          = sigmoid(mixes[:, :n]) + hc_pre_eps
//   post         = sigmoid(mixes[:, n:2n]) * hc_post_scale
//   comb         = Sinkhorn(softmax(mixes[:, 2n:]) + sinkhorn_eps)
//   layer input  = sum_i pre_i * new_residual_i, then RMSNorm * rmsnorm_scale
//   output       = optional bf16 / FP8 1x32 UE8M0 in three SF layouts
// HC / iHC use the same kernels with HCConstraint::kNone: no sigmoid/Sinkhorn.
// EXPERIMENTAL namespace until a model target passes its conformance fixture.
// Requires: common/*.
// =============================================================================

namespace kops::model_ops {

enum class HCConstraint : uint8_t {
  kNone = 0,      // HC / iHC: raw dynamic + static coefficients
  kManifold = 1,  // mHC: sigmoid / 2*sigmoid / Sinkhorn-Knopp (DeepSeek-V4: 20 iterations)
};

/// Parameters of hc_expand_kernel / hc_head_kernel. The stream tensor is [T, n_hc, d].
struct HCStreamParams {
  void* streams = nullptr;          // [num_tokens, n_hc, d]
  int64_t stream_token_stride = 0;  // elements between token rows
  DType stream_dtype = DType::kBFloat16;
  int32_t num_tokens = 0;
  int32_t n_hc = 0;
  int32_t d = 0;
};

/// Arguments shared by hc_coeff / hc_pre / hc_post / mega_mhc.
///
/// `x` is the previous sublayer output y_{l-1}; `residual` is X_{l-1} (in/out).
/// `new_residual` may alias `residual`; null means write in place. The coefficients
/// coeff_a/coeff_b/coeff_c are fp32 and use the official names pre/comb/post.
struct HCParams {
  // ---- inputs / residual state -------------------------------------------------
  const void* x = nullptr;              // [T, H] previous sublayer output, bf16/fp16/fp32
  int64_t x_stride = 0;                 // elements between token rows
  DType x_dtype = DType::kBFloat16;
  void* residual = nullptr;             // [T, n_hc, H] bf16/fp16/fp32, in/out
  int64_t residual_token_stride = 0;
  int64_t residual_route_stride = 0;    // elements between the n_hc streams of one token
  DType residual_dtype = DType::kBFloat16;
  void* new_residual = nullptr;         // [T, n_hc, H] or null (write residual in place)
  int64_t new_residual_token_stride = 0;
  int64_t new_residual_route_stride = 0;
  const float* post_mix = nullptr;      // [T, n_hc, 1]
  const float* comb_res_mix = nullptr;  // [T, n_hc, n_hc]
  const float* shifted_prev_mix = nullptr;  // [T, n_hc, 1] shifted mode, or null
  // ---- mHC projection ----------------------------------------------------------
  const float* fn = nullptr;            // [n_hc*(n_hc+2), n_hc*H] fp32
  const float* mix_scales = nullptr;    // [3] fp32 (pre, post, comb)
  const float* mix_bases = nullptr;     // [n_hc*(n_hc+2)] fp32
  float hc_norm_eps = 0.0f;
  float hc_pre_eps = 0.0f;
  float hc_post_scale = 1.0f;
  float sinkhorn_eps = 0.0f;
  int32_t num_sinkhorn_iters = 0;
  HCConstraint constraint = HCConstraint::kManifold;
  // ---- fused sub-layer RMSNorm / quant -----------------------------------------
  const void* rmsnorm_weight = nullptr; // [H] bf16/fp16/fp32, or null
  DType rmsnorm_dtype = DType::kBFloat16;
  float rmsnorm_eps = 0.0f;
  float rmsnorm_scale = 1.0f;
  // ---- coefficients and optional projection scratch ----------------------------
  float* coeff_a = nullptr;             // [T, n_hc] pre
  float* coeff_b = nullptr;             // [T, n_hc, n_hc] comb
  float* coeff_c = nullptr;             // [T, n_hc] post
  // raw_logits is the projection buffer. With `row_sumsq` also set it is writable scratch:
  // hc_coefficients runs the projection (m16n8 tf32 tensor cores for T >= 16, the per-token
  // fp32 body below that), stores the raw dots here and the row sums of squares in
  // `row_sumsq`, then applies the constraint. `raw_logits` alone keeps the DeepGEMM contract
  // (the caller ran the GEMM, the projection is skipped); see hc_coeff_kernel.
  const float* raw_logits = nullptr;    // [T, n_hc*(n_hc+2)] projection output / input
  float* row_sumsq = nullptr;           // [T] sum(new_residual^2): scratch or output
  // ---- optional coefficient outputs (official DeepGEMM names) -----------------
  void* new_prev_mix = nullptr;         // [T, n_hc, 1] or null
  void* new_post_mix = nullptr;         // [T, n_hc, 1] or null
  void* new_comb_res_mix = nullptr;     // [T, n_hc, n_hc] or null
  // ---- optional FP8 output in the extra shared-expert SF layout -----------------
  void* y_shared_sf = nullptr;          // [ceil(T/block_m)*align(block_m,128), H/128] int32
  int32_t shared_sf_block_m = 0;        // > 0 when y_shared_sf is present
  int32_t num_tokens = 0;
  int32_t n_hc = 0;
  int32_t d = 0;
};

/**
 * hc_coeff_kernel — new_residual is already formed (by hc_post or the caller). Computes
 * RMSNorm(vec X) + the skinny projection (K = n*d, N = 2n + n^2), then applies the
 * constraints. Three bodies, selected from HCParams so host and device agree:
 *   kCoeffTile     raw_logits + row_sumsq set, T >= 16, whole m16 rows / n8 outputs and
 *                  d % 8 == 0: one CTA per 16 tokens runs the projection on tf32
 *                  m16n8k8 tensor cores (hi/lo split -> fp32-class accuracy), writes
 *                  raw_logits / row_sumsq, and applies the constraint per row.
 *   kFromLogits    raw_logits set, row_sumsq null: the caller ran the projection as a GEMM;
 *                  only the constraint runs (the row sums are recomputed from the stream).
 *   generic        one CTA per token, fp32 CUDA-core projection (the T1 / fallback body).
 * All three share hc_coeff_emit, so every coefficient output is produced identically.
 */
template <typename stream_t, typename w_t, int N_HC, bool kFromLogits>
__global__ void hc_coeff_kernel(const __grid_constant__ HCParams params);

/**
 * hc_pre_kernel — x_in[t] = sum_i A[t,i] X[t,i,:], optionally fused with the sublayer
 * pre-norm (RMSNorm, weight sub_norm_weight) and activation quantization.
 */
template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant, int VEC_SIZE>
__global__ void hc_pre_kernel(
    const __grid_constant__ HCParams params,
    out_t* __restrict__ layer_in,                 // [num_tokens, d] (or quantized payload)
    const ActQuantOut quant,
    const void* __restrict__ sub_norm_weight,     // [d] or null (no fused pre-norm)
    const float sub_norm_eps);

/**
 * hc_post_kernel — X_new[t] = B[t] X_old[t] + C[t] (x) y[t], y = sublayer output.
 * Writes new_residual (or residual in place). fp32 mixing; each token row is independent.
 */
template <typename stream_t, typename y_t, int N_HC, int VEC_SIZE>
__global__ void hc_post_kernel(
    const __grid_constant__ HCParams params,
    const y_t* __restrict__ y,                    // [num_tokens, d]
    const int64_t y_stride);

/**
 * mega_mhc_kernel — single-pass fusion (DeepSeek-V4 "Mega-mHC"):
 *   X_l = B_{l-1} X_{l-1} + C_{l-1} y_{l-1}  -> write X_l
 *   predict (A_l, B_l, C_l) from X_l, emit x_in = A_l X_l (+ fused RMSNorm / quant)
 * Two bodies. D_TILE == kTileRows is the fused 16-token tile body: phase A forms the tile's
 * rows (identical arithmetic to hc_post_kernel), the tensor-core projection of hc_coeff_kernel
 * reads them back after a CTA barrier, the constraint runs per row, and the shared pre row
 * stage emits the layer input for the same 16 rows — no cross-CTA dependency, one launch.
 * D_TILE == 0 is the generic one-CTA-per-token body (the T1 / small-T path). The two bodies
 * differ only in the projection arithmetic (tf32 hi/lo split vs fp32 CUDA cores), so a chain
 * and a mega launch pick the same projection whenever they see the same shape.
 */
template <typename stream_t, typename y_t, typename out_t, int N_HC, int D_TILE, ActQuantMode kQuant,
          typename ArchTag>
__global__ void mega_mhc_kernel(
    const __grid_constant__ HCParams next_layer,
    const float* __restrict__ prev_coeff_b,       // [num_tokens, n, n] B_{l-1}
    const float* __restrict__ prev_coeff_c,       // [num_tokens, n]    C_{l-1}
    const y_t* __restrict__ prev_y,               // [num_tokens, d]    y_{l-1}
    out_t* __restrict__ layer_in,                 // [num_tokens, d]    x_in for layer l
    const ActQuantOut quant,
    const void* __restrict__ sub_norm_weight, const float sub_norm_eps);

/// hc_expand_kernel — X_0[t, i, :] = embed[t, :] for all i (stream initialization after embedding).
template <typename in_t, typename stream_t, int N_HC, int VEC_SIZE>
__global__ void hc_expand_kernel(
    stream_t* __restrict__ streams, const in_t* __restrict__ embed,
    const int num_tokens, const int d);

/// hc_head_kernel — collapse streams before the final norm / LM head: out = sum_i w_i X[t, i, :]
/// (w from a learned head mapping or uniform 1/n; HPC-Ops fuse_ihc_head).
template <typename stream_t, typename out_t, int N_HC, int VEC_SIZE>
__global__ void hc_head_kernel(
    out_t* __restrict__ out, const stream_t* __restrict__ streams,
    const float* __restrict__ head_weights,       // [T, n] dynamic, [n] static, or null (uniform)
    const bool per_token_weights, const int num_tokens, const int d);

}  // namespace kops::model_ops
