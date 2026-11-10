#pragma once
// =============================================================================
// kernel-ops | csrc/moe/ops.h                                    Catalog: M01-M04
// Requires: common/*, gemm/*, moe/*.cuh.
// =============================================================================

namespace kops::moe {

/// M01: routing. Chooses moe_topk / moe_grouped_topk / hash / fused-router kernels.
Status route(const RouterParams& params, RouterScore score, DType logits_dtype,
             const LaunchContext& ctx);
/// M01: decode-time fused router GEMM + scoring + (grouped) top-k, num_tokens <= 64.
/// Same RouterParams semantics as route(). When `params.logits` points at a caller-owned
/// fp32 [num_tokens, num_experts] scratch (logits_stride >= num_experts), the host uses the
/// tiled router_gemm kernel plus a small top-k kernel; when it is null the single-CTA fused
/// fallback computes the logits in shared memory and never touches HBM. Both paths are
/// bit-equal to gemm::router_gemm followed by route() (identical dot-product reduction order).
Status route_fused(const RouterParams& params, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor& router_bias, RouterScore score, DType w_dtype,
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
/// M02: out[t] = sum_k input[t, k] (vLLM fused_moe "moe_sum"); input is [T, topk, H].
Status moe_sum(const Tensor& out, const Tensor& input, int32_t topk, const LaunchContext& ctx);
/// M03 tail: Kimi-K3 Stable LatentMoE combine. z[t] = sum_k w[t,k] * expert_out[row(t,k)],
/// out[t] = RMSNorm(z[t]) * norm_weight, optionally quantized per `quant`.
Status latent_combine_norm(const Tensor& out, const Tensor& expert_out,
                           const Tensor& slot_to_permuted_row, const Tensor& topk_weights,
                           const Tensor& norm_weight, float routed_scale, float epsilon,
                           int32_t topk, const ActQuantOut& quant, const LaunchContext& ctx);

/// M03: whole MoE layer (route? -> permute -> W13 -> act/quant -> W2 -> reduce, latent
/// projections when latent_dim > 0). *num_launches (optional) receives the exact number of
/// device launches issued, so "fused" never hides multi-launch behaviour in benchmarks.
/// M03 decode GEMV: one expert GEMM stage over block-sorted slots on CUDA cores.  The sorted ids come
/// from align_block_size with block_size == slots_per_cta (1, 2, 4 or 8); weights are 16-bit (same
/// dtype as the activations) or OCP e4m3 with fp32 scales (w_quant.granularity per channel, per
/// expert or 128x128 block).  num_local_experts bounds the number of padded slot blocks.
Status moe_gemv(const FusedMoEGemmParams& params, int32_t num_local_experts, int32_t slots_per_cta,
                DType a_dtype, DType w_dtype, const LaunchContext& ctx);
/// M03 expert GEMM on tensor cores (ABI 6, B7): fused_moe_gemm_kernel = gemm::gemm_mma_sync_kernel over the
/// block-sorted slots of align_block_size(block_size) (16, 32, 64, 128 or 256). Same row semantics as
/// moe_gemv (A row = slot or slot / top_k, C row = slot, padding slots and experts < 0 untouched), with the
/// full gemm epilogue: GLU on interleaved W13 (C has N / 2 columns), routed weight, bias [E, N] in c_dtype,
/// quantized output. Recipes by dtype: bf16 / fp16 weights = activations; e4m3 x e4m3 (SM89+) with
/// w_quant per expert, per channel or 128x128 blocks (a_scale per token or 1x128); int8 x int8 per channel
/// with per-token a_scale. c_dtype bf16 / fp16 / fp32.
Status moe_gemm(const FusedMoEGemmParams& params, int32_t num_local_experts, int32_t block_size,
                DType a_dtype, DType w_dtype, DType c_dtype, const LaunchContext& ctx);

Status moe_workspace_requirements(const MoELayerParams& params, const LaunchContext& ctx,
                                  WorkspaceRequirements* out);
Status moe_forward(const MoELayerParams& params, const Workspace& ws, const LaunchContext& ctx,
                   int32_t* num_launches);

}  // namespace kops::moe

// ---- M04 experimental: 2-bit experts (B17, opt-in only) -----------------------------------------
// Weight format: one 2-bit code per element (four per byte, little-endian) into the four levels
// {-4, -1, +1, +4}, with one fp32 dequant multiplier per `group_size` consecutive K elements.
// `pack` accepts bf16 weights (WeightPacking::kNone) or packed e2m1 pairs
// (WeightPacking::kFP4E2M1, element 2i in the low nibble).  `gemm` consumes block-sorted slots
// from moe::align_block_size with block_size == 8 (A row = slot when a_rows_are_slots, else
// slot / top_k; C row = slot; padding slots and expert_ids < 0 untouched).  Opt-in: no routing
// or model path selects these entries.  Quality/PPL is a model-level gate (ROADMAP B17).
namespace kops::experimental::moe_w2 {

/// Pack expert weights into [E, N, K/4] u8 codes + [E, N, K/group_size] fp32 scales.
/// `weight` is bf16 [E, N, K] for kNone or packed e2m1 [E, N, K/2] u8 for kFP4E2M1.
Status pack(const Tensor& packed, const Tensor& scales, const Tensor& weight,
            WeightPacking source, int32_t group_size, const LaunchContext& ctx);

/// W2/W13 expert GEMM over block-sorted slots.  Weight params in `FusedMoEGemmParams`:
/// `w` = packed codes [E, N, K/4] (byte stride), `w_scale` = fp32 [E, N, K/group],
/// `w_expert_stride` = N*K/4 bytes, `w_scale_expert_stride` = N*(K/group) floats,
/// `w_quant.packing = kW2`, `w_quant.granularity = kPerGroup`, `w_quant.group_k = group`.
/// a_dtype / c_dtype are bf16, fp16 or fp32.  `bias` (if set) is [E, N] in c_dtype.
/// block_size is 16 (small-M WMMA, K % 64 == 0) or 64 (one expert tile, K % 64 == 0);
/// fp32 activations or K % 64 != 0 take the scalar path at block_size 16.
Status gemm(const moe::FusedMoEGemmParams& params, int32_t num_local_experts, int32_t block_size,
            DType a_dtype, DType c_dtype, const LaunchContext& ctx);

}  // namespace kops::experimental::moe_w2
