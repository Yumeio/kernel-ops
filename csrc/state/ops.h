#pragma once
// =============================================================================
// kernel-ops | csrc/state/ops.h                                  Catalog: L01-L03
// Requires: common/*, state/*.cuh.
// =============================================================================

namespace kops::state {

Status causal_conv1d_fwd(const Conv1dFwdParams& params, DType dtype, const LaunchContext& ctx);
Status causal_conv1d_update(const Conv1dUpdateParams& params, DType dtype, const LaunchContext& ctx);

Status selective_scan_fwd(const SelectiveScanParams& params, DType dtype, const LaunchContext& ctx);
Status selective_state_update(const SSMStateUpdateParams& params, DType dtype,
                              const LaunchContext& ctx);
/// Workspace for the SSD pipeline (dt_processed, dA_cumsum, CB, chunk/passed states).
Status ssd_workspace_requirements(int32_t max_tokens, int32_t max_batch, int32_t nheads,
                                  int32_t headdim, int32_t dstate, int32_t ngroups,
                                  int32_t chunk_size, WorkspaceRequirements* out);
Status ssd_chunked_scan(const SSDParams& params, DType dtype, const Workspace& ws,
                        const LaunchContext& ctx);

Status delta_rule_gating(const Tensor& g, const Tensor& beta, const Tensor& a, const Tensor& b,
                         const Tensor& A_log, const Tensor& dt_bias, DecayGranularity decay,
                         float lower_bound, const LaunchContext& ctx);
Status delta_rule_workspace_requirements(int32_t max_tokens, int32_t max_batch,
                                         int32_t num_v_heads, int32_t head_dim_k,
                                         int32_t head_dim_v, int32_t chunk_size,
                                         DecayGranularity decay, WorkspaceRequirements* out);
/// Chunked prefill (6 launches) or fused recurrent path chosen by the plan.
Status delta_rule_prefill(const DeltaRuleParams& params, DType dtype, const Workspace& ws,
                          const LaunchContext& ctx);
Status delta_rule_decode(const DeltaRuleParams& params, DType dtype, const LaunchContext& ctx);
Status fused_delta_decode(const FusedDeltaDecodeParams& params, DType dtype,
                          const LaunchContext& ctx);

Status lightning_attention_prefill(const LinearAttentionParams& params, DType dtype,
                                   const LaunchContext& ctx);
Status linear_attention_decode(const LinearAttentionParams& params, DType dtype,
                               const LaunchContext& ctx);

Status state_copy(const Tensor& pool_ptrs, const Tensor& slot_bytes, const Tensor& slot_pairs,
                  const LaunchContext& ctx);
Status state_gather(const Tensor& out, const Tensor& pool_ptrs, const Tensor& slot_bytes,
                    const Tensor& slots, const LaunchContext& ctx);
Status state_scatter(const Tensor& pool_ptrs, const Tensor& src, const Tensor& slot_bytes,
                     const Tensor& slots, const LaunchContext& ctx);
Status state_zero(const Tensor& pool_ptrs, const Tensor& slot_bytes, const Tensor& slots,
                  const LaunchContext& ctx);
Status state_commit_accepted(const Tensor& pool_ptrs, const Tensor& slot_bytes,
                             const Tensor& live_slots, const Tensor& intermediate_slots,
                             const Tensor& num_accepted, const LaunchContext& ctx);

}  // namespace kops::state
