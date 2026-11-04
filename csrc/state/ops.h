#pragma once
// =============================================================================
// kernel-ops | csrc/state/ops.h                                  Catalog: L01-L03
// Host entries of the state family. The B9 entries take the kernel parameter structs
// (device pointers + element strides, filled by the engine) and validate every host-visible
// invariant before any launch; device data (indices, lengths) is never read on the host.
// No allocation, no synchronization: every entry is graph-capturable. CONTRACTS.md §1f.
// Requires: common/*, state/*.cuh.
// =============================================================================

namespace kops::state {

/// L01. dtype = dtype of x / out / conv_states (fp32, fp16, bf16); params.weight_dtype = dtype or fp32.
Status causal_conv1d_fwd(const Conv1dFwdParams& params, DType dtype, const LaunchContext& ctx);
Status causal_conv1d_update(const Conv1dUpdateParams& params, DType dtype, const LaunchContext& ctx);

/// L02 SSM. dtype = dtype of the activation operands (fp32, fp16, bf16); states / A / D are fp32.
/// selective_scan_fwd: dstate in {8, 16, 32}; selective_state_update: dstate in {8,16,32,64,128,256}.
Status selective_scan_fwd(const SelectiveScanParams& params, DType dtype, const LaunchContext& ctx);
Status selective_state_update(const SSMStateUpdateParams& params, DType dtype,
                              const LaunchContext& ctx);
/// Workspace for the SSD pipeline (dt', dA_cumsum, CB, chunk/passed states, each 256-byte aligned)
/// for at most ceil(max_tokens / chunk_size) + max_batch chunks.
Status ssd_workspace_requirements(int32_t max_tokens, int32_t max_batch, int32_t nheads,
                                  int32_t headdim, int32_t dstate, int32_t ngroups,
                                  int32_t chunk_size, WorkspaceRequirements* out);
/// Mamba-2 chunked prefill: the five SSD kernels; chunk_size in {64, 128, 256}.
Status ssd_chunked_scan(const SSDParams& params, DType dtype, const Workspace& ws,
                        const LaunchContext& ctx);

/// L02 delta rule. dtype = dtype of a and b (fp32, fp16, bf16).
Status delta_rule_gating(const DeltaGatingParams& params, DType dtype, const LaunchContext& ctx);
/// chunk_size must be 64. Rows of h: ceil(max_tokens / 64) + max_batch + 1.
Status delta_rule_workspace_requirements(int32_t max_tokens, int32_t max_batch,
                                         int32_t num_v_heads, int32_t head_dim_k,
                                         int32_t head_dim_v, int32_t chunk_size,
                                         DecayGranularity decay, WorkspaceRequirements* out);
/// Chunked prefill (6 launches: cumsum, kkt, solve_tril, w/u, state, output); equal to
/// delta_rule_decode over every token within the numerical contract. params.math selects the
/// fp32 or tensor-core bodies (kTensorCore: bf16 inputs, SM80+).
Status delta_rule_prefill(const DeltaRuleParams& params, DType dtype, const Workspace& ws,
                          const LaunchContext& ctx);
/// Fused recurrent kernel over every token of every sequence (decode, spec verification).
Status delta_rule_decode(const DeltaRuleParams& params, DType dtype, const LaunchContext& ctx);
Status fused_delta_decode(const FusedDeltaDecodeParams& params, DType dtype,
                          const LaunchContext& ctx);

/// L02 decayed linear attention (lightning). Dk, Dv in 1..256.
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
