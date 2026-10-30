#pragma once
// =============================================================================
// kernel-ops | csrc/activation/ops.h                                  Catalog: F01
// Requires: common/tensor.h, common/status.h, common/op_types.cuh, activation/activation_kernels.cuh.
// =============================================================================

namespace kops::act {

Status activation(const Tensor& out, const Tensor& input, ActFn act, const GluParams& params,
                  const LaunchContext& ctx);

/// out[:, :d] = GLU(input[:, :2d]); row_weight optional [num_tokens] fp32.
Status act_and_mul(const Tensor& out, const Tensor& input, const Tensor& row_weight, ActFn act,
                   GateLayout layout, const GluParams& params, const LaunchContext& ctx);

Status act_and_mul_quant(const ActMulQuantParams& params, ActFn act, GateLayout layout,
                         DType input_dtype, const LaunchContext& ctx);

Status gate_mul(const Tensor& out, const Tensor& x, const Tensor& gate,
                const Tensor& accumulate_into, ActFn gate_act, const LaunchContext& ctx);

}  // namespace kops::act
