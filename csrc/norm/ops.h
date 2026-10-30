#pragma once
// =============================================================================
// kernel-ops | csrc/norm/ops.h                                  Catalog: N01, N02
// Host launchers. Validate -> pick specialization -> launch on ctx.stream.
// No allocation, no host sync, graph-capturable.
// Requires: common/*, norm/norm_kernels.cuh.
// =============================================================================

namespace kops::norm {

/// out = RMSNorm(input) * (weight + weight_offset). weight may be absent.
Status rms_norm(const Tensor& out, const Tensor& input, const Tensor& weight, float epsilon,
                float weight_offset, const LaunchContext& ctx);

/// In-place: residual += input_scale * input; input = RMSNorm(residual).
Status fused_add_rms_norm(const Tensor& input_inout, const Tensor& residual_inout,
                          const Tensor& weight, float epsilon, float weight_offset,
                          float input_scale, const LaunchContext& ctx);

/// Out-of-place variant (residual_out may alias residual_in).
Status fused_add_rms_norm_out(const Tensor& out, const Tensor& residual_out, const Tensor& input,
                              const Tensor& residual_in, const Tensor& weight, float epsilon,
                              float weight_offset, float input_scale, const LaunchContext& ctx);

Status layer_norm(const Tensor& out, const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float epsilon, float weight_offset, const LaunchContext& ctx);

Status fused_add_layer_norm(const Tensor& input_inout, const Tensor& residual_inout,
                            const Tensor& weight, const Tensor& bias, float epsilon,
                            float weight_offset, float input_scale, const LaunchContext& ctx);

/// In-place QK norm on a packed qkv tensor (or separate q/k tensors when qkv_or_q is q
/// and k is present). full_width = OLMo-2 style; full_width_sq_partials for TP > 1.
Status qk_norm(const Tensor& qkv_or_q, const Tensor& k, const Tensor& q_weight,
               const Tensor& k_weight, int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
               NormType norm, bool full_width, const Tensor& full_width_sq_partials, float epsilon,
               float weight_offset, const LaunchContext& ctx);

Status l2_norm(const Tensor& out, const Tensor& input, float scale, float epsilon,
               const LaunchContext& ctx);

Status gated_rms_norm(const Tensor& out, const Tensor& x, const Tensor& z, const Tensor& weight,
                      int32_t group_size, ActFn gate_act, bool norm_before_gate, float epsilon,
                      float weight_offset, const LaunchContext& ctx);

/// Residual add (optional) + RMSNorm/LayerNorm + activation quantization in one pass.
Status norm_quant(const NormQuantParams& params, NormType norm, DType input_dtype,
                  DType residual_dtype, const LaunchContext& ctx);

}  // namespace kops::norm
