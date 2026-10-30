#pragma once
// =============================================================================
// kernel-ops | csrc/model_ops/attn_res_kernels.cuh               Catalog: X01 (depth mixing)
// Attention Residuals (AttnRes, Kimi team; used by Kimi K3 as Block AttnRes):
// each sublayer input is a softmax-weighted mix over depth sources instead of a
// fixed residual sum.
//   sources v_0 = token embedding, v_1..v_{N} = completed block representations,
//           v_partial = running sum of sublayer outputs inside the current block
//   logit_i = w_l . RMSNorm(v_i)    (w_l: learned pseudo-query of this sublayer site)
//   h_l     = sum_i softmax_i(logit) * v_i
// Kimi K3: 8 blocks of ~12 layers; fused "residual update + AttnRes mixing + output RMSNorm".
// Requires: common/*.
// =============================================================================

namespace kops::model_ops {

/// Arguments of attn_res_kernel.
struct AttnResParams {
  // Completed sources: stacked [max_sources, num_tokens, d] (block reps, embedding at index 0).
  const void* sources;
  int64_t source_stride;              // elements between sources
  int64_t source_token_stride;
  int32_t num_sources;                // completed sources visible at this site
  const float* source_rinv;           // [max_sources, num_tokens] cached 1/RMS(v_i) (attn_res_commit writes it)
  // Running partial block sum (in/out)
  void* partial;                      // [num_tokens, d]
  int64_t partial_stride;
  // Fused residual update: partial += delta before mixing (delta = previous sublayer output)
  const void* delta;                  // [num_tokens, d] or null
  int64_t delta_stride;
  bool include_partial;               // partial participates as the last source (false right after a commit)
  // Pseudo-query and key normalization
  const void* query;                  // [d] w_l
  const void* key_norm_weight;        // [d] RMSNorm weight folded into the query, or null
  float key_eps;
  // Output: h_l and (fused) sublayer pre-norm of h_l
  void* out;                          // [num_tokens, d] h_l (or null if only normed output needed)
  int64_t out_stride;
  void* normed_out;                   // [num_tokens, d] RMSNorm(h_l) * out_norm_weight, or null
  const void* out_norm_weight;        // [d]
  float out_norm_eps;
  ActQuantOut normed_quant;           // optional quantized normed output (MXFP8 for Kimi-K3 projections)
  float* alpha_out;                   // [num_tokens, num_sources + 1] debugging / analysis, or null
  int32_t num_tokens, d;
};

/**
 * attn_res_kernel — one CTA (or warp group) per token row: optional partial += delta, logits over
 * sources using cached inverse RMS (fp32), softmax over depth, weighted sum, optional output
 * RMSNorm + quantization. Bitwise independent of batch composition.
 */
template <typename scalar_t, int MAX_SOURCES, int VEC_SIZE, ActQuantMode kQuant>
__global__ void attn_res_kernel(const KOPS_GRID_CONST AttnResParams params);

/**
 * attn_res_commit_kernel — at a block boundary: copy partial into source slot `slot`, compute and
 * cache its inverse RMS (source_rinv[slot]), then reset partial to zero (or to `delta`).
 */
template <typename scalar_t, int VEC_SIZE>
__global__ void attn_res_commit_kernel(
    void* __restrict__ sources,                  // [max_sources, num_tokens, d]
    const int64_t source_stride,
    float* __restrict__ source_rinv,             // [max_sources, num_tokens]
    void* __restrict__ partial,                  // [num_tokens, d] in/out
    const void* __restrict__ restart_with,       // [num_tokens, d] new partial value, or null (zeros)
    const int slot, const float eps, const int num_tokens, const int d);

}  // namespace kops::model_ops
