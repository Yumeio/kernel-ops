#pragma once
// =============================================================================
// kernel-ops | csrc/model_ops/hyper_connection_kernels.cuh            Catalog: X01
// Hyper-Connections (HC, Zhu et al.), HPC-Ops iHC, and Manifold-Constrained
// Hyper-Connections (mHC, DeepSeek-V4) incl. the single-pass fused variant.
// Residual stream is widened to n_hc copies: X_l in R^{n x d} per token.
//   X^ = RMSNorm(vec(X_l))                                     (1 x n*d)
//   A~ = a_pre  * (X^ W_pre)  + S_pre      -> A = sigmoid(A~)       (1 x n)   input mixing
//   B~ = a_res  * Mat(X^ W_res) + S_res    -> B = Sinkhorn(exp(B~)) (n x n)   doubly stochastic
//   C~ = a_post * (X^ W_post)^T + S_post   -> C = 2 sigmoid(C~)     (n x 1)   output mixing
//   layer input  x_in = A X_l ;   X_{l+1} = B X_l + C F(x_in)
// Plain HC / iHC use the same kernels with Constraint::kNone (no sigmoid/Sinkhorn).
// EXPERIMENTAL namespace until a model target passes its conformance fixture.
// Requires: common/*.
// =============================================================================

namespace kops::model_ops {

enum class HCConstraint : uint8_t {
  kNone = 0,      // HC / iHC: raw dynamic + static coefficients
  kManifold = 1,  // mHC: sigmoid / 2*sigmoid / Sinkhorn-Knopp (t_max iterations, DeepSeek-V4: 20)
};

/// Arguments shared by the HC kernels.
struct HCParams {
  void* streams;                  // X_l [num_tokens, n_hc, d] (bf16 or fp32 residual streams)
  int64_t token_stride;
  DType stream_dtype;
  const void* norm_weight;        // [n_hc * d] RMSNorm weight of vec(X_l)
  float norm_eps;
  const void* w_all;              // [n_hc * d, 2*n_hc + n_hc^2] packed [W_pre | W_post | W_res]
  float alpha_pre, alpha_res, alpha_post;
  const float* s_pre;             // [n_hc]
  const float* s_post;            // [n_hc]
  const float* s_res;             // [n_hc, n_hc]
  int32_t sinkhorn_iters;         // 20
  HCConstraint constraint;
  // Coefficients (fp32), produced by hc_coeff_kernel, consumed by pre/post
  float* coeff_a;                 // [num_tokens, n_hc]
  float* coeff_b;                 // [num_tokens, n_hc, n_hc]
  float* coeff_c;                 // [num_tokens, n_hc]
  const float* raw_logits;        // [num_tokens, 2n + n^2] when the projection ran as a separate GEMM
  int32_t num_tokens, n_hc, d;
};

/**
 * hc_coeff_kernel — RMSNorm(vec X) + skinny projection (K = n*d, N = 2n + n^2 = 24 for n = 4) +
 * constraints, one CTA per token tile. When num_tokens is large the projection runs as a GEMM
 * (raw_logits) and this kernel only applies constraints (kFromLogits = true).
 */
template <typename stream_t, typename w_t, int N_HC, bool kFromLogits>
__global__ void hc_coeff_kernel(const __grid_constant__ HCParams params);

/**
 * hc_pre_kernel — x_in[t] = sum_i A[t,i] X[t,i,:], optionally fused with the sublayer pre-norm
 * (RMSNorm, weight sub_norm_weight) and activation quantization for the first projection.
 */
template <typename stream_t, typename out_t, int N_HC, ActQuantMode kQuant, int VEC_SIZE>
__global__ void hc_pre_kernel(
    const __grid_constant__ HCParams params,
    out_t* __restrict__ layer_in,                 // [num_tokens, d] (or quantized payload)
    const ActQuantOut quant,
    const void* __restrict__ sub_norm_weight,     // [d] or null (no fused pre-norm)
    const float sub_norm_eps);

/**
 * hc_post_kernel — X_{l+1}[t] = B[t] X_l[t] + C[t] (x) y[t], y = sublayer output [num_tokens, d].
 * In place on streams (each token row is independent). fp32 mixing.
 */
template <typename stream_t, typename y_t, int N_HC, int VEC_SIZE>
__global__ void hc_post_kernel(
    const __grid_constant__ HCParams params,
    const y_t* __restrict__ y,                    // [num_tokens, d]
    const int64_t y_stride);

/**
 * mega_mhc_kernel — single-pass fusion (DeepSeek-V4 deployment "Mega-mHC" / Single-Pass mHC):
 *   X_l = B_{l-1} X_{l-1} + C_{l-1} y_{l-1}  -> write X_l
 *   predict (A_l, B_l, C_l) from X_l, emit x_in = A_l X_l (+ fused sub-layer RMSNorm / quant)
 * Processes the hidden dim in tiles; norm and projection partials are reduced across tiles inside
 * the CTA cluster, so one token row is owned by one cluster (no grid-wide sync).
 */
template <typename stream_t, typename y_t, typename out_t, int N_HC, int D_TILE, ActQuantMode kQuant,
          typename ArchTag>
__global__ void mega_mhc_kernel(
    const __grid_constant__ HCParams next_layer,  // coefficients/weights of layer l (outputs A_l, B_l, C_l)
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
    const float* __restrict__ head_weights,       // [num_tokens, n] dynamic, [n] static, or null (uniform)
    const bool per_token_weights, const int num_tokens, const int d);

}  // namespace kops::model_ops
