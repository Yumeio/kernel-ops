#pragma once
// =============================================================================
// kernel-ops | csrc/norm/norm_kernels.cuh                      Catalog: N01, N02
// RMSNorm / LayerNorm / QK-norm / L2-norm / gated-norm and residual fusions.
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh, common/quant.cuh.
//
// Numerics (all kernels): reductions in fp32, one rounding to the output dtype
// at the end. weight_offset folds "(1 + w)" conventions:
//   0.0 -> LLaMA/Mistral/Qwen/DeepSeek/Kimi       1.0 -> Gemma 1/2/3, Qwen3-Next/3.5 (zero-centred),
//   LayerNorm1P (Nemotron) uses weight_offset = 1.0 with LayerNorm.
// Launch (row kernels): grid = num_rows (or ceil(num_rows / ROWS_PER_CTA)),
//   block = min(1024, round_up(hidden / VEC_SIZE, 32)); one warp per row when hidden <= 1024.
// =============================================================================

namespace kops::norm {

/**
 * rms_norm_kernel
 *   out[r, :] = in[r, :] * rsqrt(mean(in[r, :]^2) + eps) * (w[g(r), :] + weight_offset)
 *
 * Models : LLaMA 1-4, Mistral/Mixtral, Qwen 1.5-3.5, DeepSeek V2-V4, Kimi K2/K3, GLM-4.5/5,
 *          MiniMax, GPT-OSS, Gemma (offset 1), Phi-3/4, InternLM, Yi, Baichuan, Granite, OLMo 2.
 * Input  : up to 4-D, last dim contiguous; strides for dims -2..-4 allow normalizing
 *          non-contiguous views (e.g. q/k slices of a fused QKV, per-head views) without copies.
 * Weight : shared [hidden] (num_weight_groups == 1) or per-group [num_weight_groups, hidden]
 *          selected by g(r) = r % num_weight_groups (Cohere Command-R+ per-head QK LayerNorm/RMS weights).
 * Alias  : out may alias input only if their row layouts are identical.
 * Align  : hidden_size % VEC_SIZE == 0 and 16-byte aligned rows when VEC_SIZE*sizeof(T) == 16.
 */
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, hidden_size], row stride out_stride
    const int64_t out_stride,             // out.stride(-2)
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3); unused if NUM_DIMS < 3
    const int64_t input_stride_d4,        // input.stride(-4); unused if NUM_DIMS < 4
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3); unused if NUM_DIMS < 4
    const scalar_t* __restrict__ weight,  // [hidden_size] or [num_weight_groups, hidden_size];
                                          // null if !HasWeight (Llama-4 QK L2Norm-as-RMS, OLMoE)
    const int64_t weight_stride,          // 0 (shared) or weight.stride(0)
    const int num_weight_groups,          // 1, or number of per-row weight groups
    const float weight_offset,            // 0.0 | 1.0 (Gemma / zero-centred)
    const float epsilon,                  // 1e-5 / 1e-6 typical
    const int num_tokens,                 // flattened rows = prod(input.shape[:-1])
    const int hidden_size);

/**
 * fused_add_rms_norm_kernel (pre-norm residual pattern, in-place)
 *   residual[r] = residual[r] + input_scale * input[r]            (written back, residual_t)
 *   input[r]    = RMSNorm(residual[r]) * (w + weight_offset)      (written back, scalar_t)
 *
 * Models : every pre-norm decoder (attention/MLP boundaries). input_scale = Granite
 *          residual_multiplier, MiniCPM scale_depth/sqrt(L); 1.0 otherwise.
 * Residual dtype may differ (fp32 residual stream for precision-sensitive deployments).
 * Sandwich norm (Gemma 2/3 post_attention_layernorm + pre_feedforward_layernorm): call
 * rms_norm_kernel on the sublayer output first, then this kernel.
 */
template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [num_tokens, hidden] in: sublayer output, out: normed
    const int64_t input_stride,           // input.stride(0)
    residual_t* __restrict__ residual,    // [num_tokens, hidden] contiguous, in/out
    const scalar_t* __restrict__ weight,  // [hidden]
    const float weight_offset,
    const float input_scale,              // multiplier applied to input before the add
    const float epsilon, const int num_tokens, const int hidden_size);

/**
 * fused_add_rms_norm_out_kernel (out-of-place variant, keeps the sublayer output intact)
 *   residual_out = residual_in + input_scale * input;  out = RMSNorm(residual_out)
 * Needed when the sublayer output is consumed twice (AttnRes partial sums, mHC post).
 */
template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__global__ void fused_add_rms_norm_out_kernel(
    scalar_t* __restrict__ out,                 // [num_tokens, hidden]
    residual_t* __restrict__ residual_out,      // [num_tokens, hidden]; may alias residual_in
    const scalar_t* __restrict__ input,         // [num_tokens, hidden]
    const int64_t input_stride,
    const residual_t* __restrict__ residual_in, // [num_tokens, hidden]
    const scalar_t* __restrict__ weight,        // [hidden]
    const float weight_offset, const float input_scale, const float epsilon,
    const int num_tokens, const int hidden_size);

/**
 * layer_norm_kernel
 *   out = (x - mean) * rsqrt(var + eps) * (w + weight_offset) + b      (biased variance, like torch)
 *
 * Models : GPT-2, GPT-J, GPT-NeoX/Pythia, OPT, BLOOM (+ embedding LayerNorm), Falcon, MPT
 *          (no bias), StarCoder/StarCoder2, Phi-1/2, Cohere Command-R (no bias), Nemotron
 *          (LayerNorm1P: weight_offset=1), vision towers (CLIP/SigLIP/MoonViT uses RMS).
 */
template <typename scalar_t, int VEC_SIZE, bool HasWeight, bool HasBias>
__global__ void layer_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, hidden]
    const int64_t out_stride,
    const scalar_t* __restrict__ input,   // [num_tokens, hidden], row stride input_stride
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden] or null
    const scalar_t* __restrict__ bias,    // [hidden] or null
    const float weight_offset, const float epsilon, const int num_tokens, const int hidden_size);

/**
 * fused_add_layer_norm_kernel (in-place, same contract as fused_add_rms_norm_kernel).
 * Models : GPT-2 / OPT / BLOOM / StarCoder residual + ln_2; parallel-residual models
 *          (GPT-J, NeoX, Falcon, Cohere) use model_ops::residual_add3 instead.
 */
template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight, bool HasBias>
__global__ void fused_add_layer_norm_kernel(
    scalar_t* __restrict__ input,         // [num_tokens, hidden] in: sublayer out, out: normed
    const int64_t input_stride,
    residual_t* __restrict__ residual,    // [num_tokens, hidden] in/out
    const scalar_t* __restrict__ weight,  // [hidden]
    const scalar_t* __restrict__ bias,    // [hidden] or null
    const float weight_offset, const float input_scale, const float epsilon,
    const int num_tokens, const int hidden_size);

/**
 * qk_norm_kernel — in-place per-head (or full-width) norm of Q and K inside a packed QKV.
 *   q[t, h, :] = Norm(q[t, h, :]) * (q_w + off);  k[t, h, :] = Norm(k[t, h, :]) * (k_w + off)
 *
 * Models : per-head RMS  -> Qwen3/Qwen3-MoE/Qwen3-Next, Gemma 3 (off=1), GLM-4.5 (use_qk_norm),
 *                           Llama-4 (L2Norm without weight), DeepSeek-V4 (query/KV RMSNorm), Kimi MLA.
 *          full-width RMS -> OLMo 2 / OLMoE (norm over Hq*D; with TP the sum of squares
 *                           must be all-reduced: use kFullWidth with cross-rank partials).
 *          per-head LN    -> Cohere Command-R+ (weights [H, D]).
 * V is untouched. If RoPE follows, prefer position::fused_qk_norm_rope_kernel.
 */
template <typename scalar_t, int HEAD_DIM, NormType kNorm, bool kFullWidth>
__global__ void qk_norm_kernel(
    scalar_t* __restrict__ qkv,             // [num_tokens, (Hq + 2*Hkv) * HEAD_DIM] or separate q/k views
    const int64_t token_stride,             // qkv.stride(0)
    const int64_t q_offset,                 // element offset of q inside a token row (usually 0)
    const int64_t k_offset,                 // element offset of k (usually Hq * HEAD_DIM)
    const scalar_t* __restrict__ q_weight,  // [HEAD_DIM] | [Hq, HEAD_DIM] | [Hq*HEAD_DIM] (kFullWidth); null = no affine
    const scalar_t* __restrict__ k_weight,  // [HEAD_DIM] | [Hkv, HEAD_DIM] | [Hkv*HEAD_DIM]; null = no affine
    const int64_t q_weight_head_stride,     // 0 if shared across heads
    const int64_t k_weight_head_stride,
    const float* __restrict__ full_width_sq_partials,  // kFullWidth && TP>1: [num_tokens, 2] summed sq
                                            // over all ranks (q, k); null = compute locally
    const float weight_offset, const float epsilon,
    const int num_tokens, const int num_q_heads, const int num_kv_heads);

/**
 * l2_norm_kernel — out = x * rsqrt(sum(x^2) + eps) * scale   (true L2, not mean)
 * Models : Gated DeltaNet (Qwen3-Next/3.5) and KDA (Kimi Linear/K3) q,k normalization when
 *          not fused into the delta-rule kernel; Llama-4 uses rms_norm without weight instead.
 */
template <typename scalar_t, int VEC_SIZE>
__global__ void l2_norm_kernel(
    scalar_t* __restrict__ out,          // [num_rows, dim]
    const int64_t out_stride,
    const scalar_t* __restrict__ input,  // [num_rows, dim] (e.g. [T*H, Dk] view)
    const int64_t input_stride,
    const float scale,                   // extra multiplier (1.0, or Dk^-0.5 folded for q)
    const float epsilon, const int num_rows, const int dim);

/**
 * gated_rms_norm_kernel — grouped RMSNorm combined with a gate branch.
 *   kNormBeforeGate = true : y = RMSNorm_g(x) * (w + off) * act(z)          (Qwen3-Next GDN, KDA: act=sigmoid)
 *   kNormBeforeGate = false: y = RMSNorm_g(x * act(z)) * (w + off)          (Mamba-2: act=SiLU)
 * Normalization runs independently over contiguous groups of `group_size` channels
 * (Mamba-2 ngroups; GDN/KDA per value head: group_size = Dv).
 */
template <typename scalar_t, int VEC_SIZE, ActFn kGateAct, bool kNormBeforeGate>
__global__ void gated_rms_norm_kernel(
    scalar_t* __restrict__ out,          // [num_rows, hidden]
    const int64_t out_stride,
    const scalar_t* __restrict__ x,      // [num_rows, hidden]
    const int64_t x_stride,
    const scalar_t* __restrict__ z,      // [num_rows, hidden] gate pre-activation
    const int64_t z_stride,
    const scalar_t* __restrict__ weight, // [hidden] (or [group_size] when shared across groups)
    const int64_t weight_group_stride,   // 0 if weight is [group_size], else group_size
    const float weight_offset, const float epsilon,
    const int num_rows, const int hidden_size, const int group_size);

// -----------------------------------------------------------------------------
// N02: norm + residual + quantization (one read of x, one write of q + scales)
// -----------------------------------------------------------------------------

/// Arguments of norm_quant_kernel.
struct NormQuantParams {
  const void* input;          // [num_tokens, hidden] sublayer output (scalar_t)
  int64_t input_stride;
  void* residual;             // [num_tokens, hidden] residual_t, in/out; null => no fused add
  const void* residual_in;    // optional separate residual source (out-of-place); null => use residual
  float input_scale;          // residual multiplier (see fused_add_rms_norm_kernel)
  const void* weight;         // [hidden]
  const void* bias;           // [hidden] (LayerNorm only) or null
  float weight_offset;
  float epsilon;
  void* norm_out;             // optional [num_tokens, hidden] scalar_t copy of the normed value
                              // (needed when a non-quantized consumer also reads it); null = skip
  int64_t norm_out_stride;
  ActQuantOut quant;          // quantized output (FP8 / INT8 / MXFP8 / MXFP4 / NVFP4)
  int32_t num_tokens;
  int32_t hidden_size;
};

/**
 * norm_quant_kernel — (optional residual add) -> RMSNorm/LayerNorm -> quantize.
 *   kMode = kStaticPerTensor : q = sat(y / s)                         (vLLM FP8 static, INT8 static)
 *   kMode = kDynamicPerToken : s_r = max(|y_r|)/qmax (clamped by scale_upper_bound)  (FP8/INT8 dynamic)
 *   kMode = kPerGroup        : s_{r,g} per 128 channels, optional UE8M0 round-up, row-major or
 *                              col-major TMA scales (DeepSeek-V3/V3.2/V4 FP8 linear inputs)
 *   kMode = kMXFP8 / kMXFP4  : UE8M0 per 32 channels (Kimi-K3 MXFP8 activations, GPT-OSS MXFP4)
 *   kMode = kNVFP4           : E4M3 per 16 + caller global scale, swizzled 128x4 scale layout
 * Rounding point: y is computed in fp32 and quantized directly (no intermediate bf16 rounding)
 * unless norm_out is requested, in which case norm_out is the bf16 rounding of the same y.
 */
template <typename scalar_t, typename residual_t, NormType kNorm, ActQuantMode kMode, int VEC_SIZE>
__global__ void norm_quant_kernel(const __grid_constant__ NormQuantParams params);

}  // namespace kops::norm
