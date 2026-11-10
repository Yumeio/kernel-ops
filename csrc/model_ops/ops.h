#pragma once
// =============================================================================
// kernel-ops | csrc/model_ops/ops.h                                Catalog: X01, X02
// Requires: common/*, model_ops/*.cuh.
// =============================================================================

namespace kops::model_ops {

Status hc_coefficients(const HCParams& params, const LaunchContext& ctx);
Status hc_pre(const HCParams& params, const Tensor& layer_in, const ActQuantOut& quant,
              const Tensor& sub_norm_weight, float sub_norm_eps, const LaunchContext& ctx);
Status hc_post(const HCParams& params, const Tensor& y, const LaunchContext& ctx);
Status mega_mhc(const HCParams& next_layer, const Tensor& prev_coeff_b, const Tensor& prev_coeff_c,
                const Tensor& prev_y, const Tensor& layer_in, const ActQuantOut& quant,
                const Tensor& sub_norm_weight, float sub_norm_eps, const LaunchContext& ctx);
Status hc_expand(const HCStreamParams& params, const Tensor& embed, const LaunchContext& ctx);
Status hc_head(const Tensor& out, const HCStreamParams& params, const Tensor& head_weights,
               const LaunchContext& ctx);

Status attn_res(const AttnResParams& params, DType dtype, const LaunchContext& ctx);
Status attn_res_commit(const Tensor& sources, const Tensor& source_rinv, const Tensor& partial,
                       const Tensor& restart_with, int32_t slot, float eps, const LaunchContext& ctx);

Status residual_add(const Tensor& out, const Tensor& a, float a_scale, const Tensor& b,
                    float b_scale, const Tensor& c, float c_scale, const LaunchContext& ctx);
Status cast(const Tensor& out, const Tensor& in, float scale, const LaunchContext& ctx);

}  // namespace kops::model_ops
