#pragma once
// =============================================================================
// kernel-ops | csrc/quantization/ops.h                    Catalog: Q01, Q02, Q03, Q04
// Requires: common/tensor.h, common/status.h, common/quant.cuh,
//           quantization/quant_kernels.cuh, quantization/weight_repack_kernels.cuh.
// =============================================================================

namespace kops::quant {

// ---- Q01 activation / tensor quantization ----------------------------------------
Status absmax(const Tensor& amax_out, const Tensor& input, const LaunchContext& ctx);
/// Quantize `input` according to out.mode (static/dynamic per-tensor, per-token, per-group,
/// MXFP8/MXFP4/NVFP4). Writes payload + scales described by `out`.
/// Per-group (`kPerGroup`) supports group 32/64/128, a row-major or kColMajorTma scale layout,
/// the DeepEP masked layout ([E, max_m, cols] with input.stride[0] as the padded expert stride)
/// and `scale_ue8m0` (scales rounded up to powers of two, stored as fp32). Other scale layouts
/// return kUnsupportedLayout.
/// `out.math` is the quotient arithmetic: kExact (default) divides by the scale with __fdiv_rn;
/// kReciprocal uses one RN reciprocal plus a Markstein correction, which produces the same
/// payload bytes but is only implemented for kPerGroup (other modes return kUnsupportedLayout)
/// and only on the vectorized path (an unaligned/unsupported shape falls back to exact maths).
Status quantize_activation(const ActQuantOut& out, const Tensor& input,
                           const Tensor& masked_m, int32_t max_m, const LaunchContext& ctx);
Status dequantize(const Tensor& out, const Tensor& q, const Tensor& scales,
                  const Tensor& zero_points, const QuantScheme& scheme, const LaunchContext& ctx);
Status pack_int4(const Tensor& out, const Tensor& input, bool is_unsigned, const LaunchContext& ctx);
Status unpack_int4(const Tensor& out, const Tensor& input, bool is_unsigned, const LaunchContext& ctx);
Status hadamard_transform(const Tensor& out, const Tensor& input, float scale, bool signed_diag,
                          uint32_t seed, const LaunchContext& ctx);

// ---- Q02 AWQ / Q03 GPTQ import ----------------------------------------------------
/// Validate checkpoint metadata (bits, group_size, zero_point, desc_act/g_idx, sym) against
/// what the selected backend supports; returns kUnsupportedLayout with a reason otherwise.
Status validate_wna16_checkpoint(const QuantScheme& scheme, int64_t size_k, int64_t size_n,
                                 const Tensor& g_idx);
Status awq_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& out_zeros,
                     const Tensor& qweight, const Tensor& scales, const Tensor& qzeros,
                     int32_t num_bits, int32_t group_size, const LaunchContext& ctx);
/// perm_out receives argsort(g_idx) (act-order); the same perm must be passed to marlin_gemm.
Status gptq_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& perm_out,
                      const Tensor& qweight, const Tensor& scales, const Tensor& g_idx,
                      int32_t num_bits, int32_t group_size, bool is_sym, const LaunchContext& ctx);
Status awq_dequantize(const Tensor& out, const Tensor& qweight, const Tensor& scales,
                      const Tensor& qzeros, int32_t group_size, const LaunchContext& ctx);

// ---- Q04 SmoothQuant ---------------------------------------------------------------
Status smoothquant_compute_scales(const Tensor& smooth, const Tensor& act_absmax,
                                  const Tensor& weight_absmax, float alpha, const LaunchContext& ctx);
Status smoothquant_fold(const Tensor& norm_weight, const Tensor& norm_bias,
                        const Tensor& linear_weight, const Tensor& smooth, const LaunchContext& ctx);

// ---- FP4 / FP8 / GGUF weight preparation ------------------------------------------
Status fp4_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& weight,
                     const Tensor& scales, int32_t block_size, bool scale_is_ue8m0,
                     const LaunchContext& ctx);
Status fp4_backend_shuffle(const Tensor& out_weight, const Tensor& out_scales, const Tensor& weight,
                           const Tensor& scales, int32_t block_size, WeightPacking target,
                           const LaunchContext& ctx);
Status fp8_block_scale_to_ue8m0(const Tensor& weight_inout, const Tensor& scales_inout,
                                const Tensor& ue8m0_out, const LaunchContext& ctx);
Status gguf_dequantize(const Tensor& out, const Tensor& qweight, GGUFType type,
                       const LaunchContext& ctx);

}  // namespace kops::quant
