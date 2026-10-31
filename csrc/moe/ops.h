#pragma once
// =============================================================================
// kernel-ops | csrc/moe/ops.h                                    Catalog: M01-M04
// Requires: common/*, gemm/*, moe/*.cuh.
// =============================================================================

namespace kops::moe {

/// M01: routing. Chooses moe_topk / moe_grouped_topk / hash / fused-router kernels.
Status route(const RouterParams& params, RouterScore score, DType logits_dtype,
             const LaunchContext& ctx);
Status hash_route(const Tensor& topk_weights, const Tensor& topk_ids, const Tensor& token_ids,
                  const Tensor& hash_table, const Tensor& expert_map, float routed_scaling_factor,
                  int32_t num_experts, const LaunchContext& ctx);

/// M02: routing metadata + permutation.
Status align_block_size(const Tensor& topk_ids, const Tensor& sorted_token_ids,
                        const Tensor& expert_ids, const Tensor& num_tokens_post_pad,
                        const Tensor& expert_map, int32_t num_experts, int32_t block_size,
                        const Workspace& ws, const LaunchContext& ctx);
Status count_and_offsets(const Tensor& topk_ids, const Tensor& expert_counts,
                         const Tensor& expert_offsets, const Tensor& expert_map,
                         int32_t num_local_experts, int32_t align_m, const LaunchContext& ctx);
Status permute(const PermuteParams& params, DType hidden_dtype, const LaunchContext& ctx);
Status unpermute_reduce(const Tensor& out, const Tensor& expert_out,
                        const Tensor& slot_to_permuted_row, const Tensor& topk_weights,
                        const Tensor& shared_out, const Tensor& residual, float routed_scale,
                        bool weights_already_applied, const LaunchContext& ctx);

/// M03: whole MoE layer (route? -> permute -> W13 -> act/quant -> W2 -> reduce, latent
/// projections when latent_dim > 0). *num_launches (optional) receives the exact number of
/// device launches issued, so "fused" never hides multi-launch behaviour in benchmarks.
/// M03 decode GEMV: one expert GEMM stage over block-sorted slots on CUDA cores.  The sorted ids come
/// from align_block_size with block_size == slots_per_cta (1, 2, 4 or 8); weights are 16-bit (same
/// dtype as the activations) or OCP e4m3 with fp32 scales (w_quant.granularity per channel, per
/// expert or 128x128 block).  num_local_experts bounds the number of padded slot blocks.
Status moe_gemv(const FusedMoEGemmParams& params, int32_t num_local_experts, int32_t slots_per_cta,
                DType a_dtype, DType w_dtype, const LaunchContext& ctx);

Status moe_workspace_requirements(const MoELayerParams& params, const LaunchContext& ctx,
                                  WorkspaceRequirements* out);
Status moe_forward(const MoELayerParams& params, const Workspace& ws, const LaunchContext& ctx,
                   int32_t* num_launches);

}  // namespace kops::moe
