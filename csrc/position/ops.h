#pragma once
// =============================================================================
// kernel-ops | csrc/position/ops.h                                    Catalog: R01
// Requires: common/tensor.h, common/status.h, common/op_types.cuh, position/rope_kernels.cuh.
// =============================================================================

namespace kops::rope {

/// Host: fill inv_freq[rot/2] and return the cos/sin magnitude multiplier (YaRN mscale).
Status compute_inv_freq(const RopeConfig& cfg, float* inv_freq_host, float* cos_sin_mscale);

/// Host: ALiBi slopes for num_heads (Press et al.; non power-of-two heads use the
/// interleaved closest-power-of-two rule as in BLOOM/MPT). out_host: [num_heads].
Status compute_alibi_slopes(int32_t num_heads, float max_bias, float* out_host);

/// Host launchers.
Status build_cos_sin_cache(const Tensor& cache, const RopeConfig& cfg, int32_t max_position,
                           const LaunchContext& ctx);
Status apply_rotary(const Tensor& positions, const Tensor& query, const Tensor& key,
                    const Tensor& cos_sin_cache, const Tensor& cache_offsets, int32_t rot_dim,
                    int32_t rot_offset, RopeStyle style, const LaunchContext& ctx);
Status apply_rotary_on_the_fly(const Tensor& positions, const Tensor& query, const Tensor& key,
                               const Tensor& inv_freq, float mscale, int32_t rot_dim,
                               int32_t rot_offset, RopeStyle style, const LaunchContext& ctx);
Status apply_mrope(const Tensor& positions_thw, const Tensor& query, const Tensor& key,
                   const Tensor& cos_sin_cache, const int32_t mrope_section[3], bool interleaved,
                   RopeStyle style, const LaunchContext& ctx);
Status fused_qk_norm_rope(const Tensor& qkv, const Tensor& q_weight, const Tensor& k_weight,
                          const Tensor& positions, const Tensor& cos_sin_cache,
                          int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
                          int32_t rot_dim, int32_t rot_offset, RopeStyle style,
                          float weight_offset, float epsilon, const LaunchContext& ctx);
Status attn_temperature_tuning(const Tensor& query, const Tensor& positions, float floor_scale,
                               float attn_scale, const LaunchContext& ctx);

}  // namespace kops::rope
