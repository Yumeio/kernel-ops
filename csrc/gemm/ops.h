#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/ops.h                                  Catalog: G01-G07
// Host entry points for every GEMM family. Each call:
//   validate (recipe x dtypes x layouts x alignment) -> resolve tactic (explicit, cached,
//   or heuristic) -> launch on ctx.stream with caller workspace. No allocation, no sync.
// Backend choice (native / cuBLASLt / CUTLASS / DeepGEMM / TRTLLM-gen / Marlin / Machete)
// is a registry decision; kAuto only picks release-supported rows for the exact SM.
// Requires: common/*, quantization/weight_repack_kernels.cuh, gemm/*.cuh.
// =============================================================================

namespace kops::gemm {

// ---- Tactics / autotuning (I03) ----------------------------------------------------
/// List compiled tactics valid for (problem, exact SM), best-first by the static heuristic.
Status get_tactics(const GemmProblem& problem, GemmBackend backend, const LaunchContext& ctx,
                   GemmTactic* out, int32_t max_tactics, int32_t* num_tactics);
/// Offline autotune over M buckets (e.g. 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192);
/// results stored in the tactic cache keyed by (recipe, N, K, M bucket, SM, kernel-ops commit).
Status autotune(const GemmProblem& problem, const int64_t* m_buckets, int32_t num_buckets,
                GemmBackend backend, const Workspace& ws, const LaunchContext& ctx);
Status tactic_cache_lookup(const GemmProblem& problem, const LaunchContext& ctx, GemmTactic* out);
Status tactic_cache_save(const char* path);
Status tactic_cache_load(const char* path);

/// Workspace for any GEMM selection (split-K partials, stream-K fix-up, locks, task maps,
/// grouped tensor maps).
Status gemm_workspace_requirements(const GemmProblem& problem, const GemmTactic* tactic,
                                   GemmBackend backend, const LaunchContext& ctx,
                                   WorkspaceRequirements* out);

// ---- Tensor-core mainloop (gemm_mma.cu) -------------------------------------------------------
/// bf16 / fp16 GEMM on mma.sync (SM80+). gemm() / scaled_gemm() dispatch here for M > 16, batched or
/// swap_ab problems, when a tactic names kMultistageMmaSync, and when the tactic cache has an entry.
/// tactic == nullptr: cache, then the static heuristic. Needs a Workspace only for split-K.
Status mma_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, const Workspace& ws, const LaunchContext& ctx);
/// True when mma_gemm would take this problem on a cache hit / explicit mma tactic (the GEMV owns the rest).
bool mma_selects(const ScaledGemmArgs& args, const GemmTactic* tactic, const LaunchContext& ctx);
Status mma_workspace(const GemmProblem& problem, const GemmTactic* tactic, const LaunchContext& ctx,
                     WorkspaceRequirements* out);
/// Shared launcher of gemm_mma_sync_kernel (ABI 6) for the B7 rows without a body of their own: the
/// FP8 (Sm89+), INT8 and TF32 recipes and the grouped / masked / sorted-block row maps (grouped GEMM,
/// fused MoE, LoRA). Validates operands, picks the tactic (explicit, cached, heuristic) and launches.
/// Split-K needs Workspace::device_scratch of mma_workspace() size (grouped schedules: rows.max_tasks
/// counters instead of tiles).
Status launch_gemm_mma_sync(const MmaGemmParams& params, const GemmTactic* tactic, const Workspace& ws,
                            const LaunchContext& ctx);

// ---- G01 dense 16/32-bit -------------------------------------------------------------
/// Workspace is passed explicitly (not through GemmArgs) so every family has one
/// convention: query gemm_workspace_requirements(), then pass the same lease here.
Status gemm(const GemmProblem& problem, const GemmArgs& args, GemmBackend backend,
            const Workspace& ws, const LaunchContext& ctx);
Status gemm_with_tactic(const ScaledGemmArgs& args, const GemmTactic& tactic, const Workspace& ws,
                        const LaunchContext& ctx);
Status dual_gemm(const DualGemmParams& params, DType dtype, const LaunchContext& ctx);
/// Greedy LM head.  The caller sizes partial_max / partial_idx / partial_sumexp as
/// [M, gemv_argmax_num_ctas(v_shard)] and keeps done_counter zero between launches (the last CTA
/// restores it).  Ids are global (vocab_start + row).  After the launch slot 0 of each row of
/// partial_max / partial_sumexp holds the shard-wide max logit and sum-exp(logit - max), which a
/// vocab-parallel caller allgathers together with out_ids.
Status gemv_argmax(const GemvArgmaxParams& params, DType hidden_dtype, DType weight_dtype,
                   const LaunchContext& ctx);
int32_t gemv_argmax_num_ctas(int32_t v_shard);
Status bmm(const BmmParams& params, DType a_dtype, DType b_dtype, DType d_dtype,
           const LaunchContext& ctx);

// ---- G02 router -------------------------------------------------------------------------
Status router_gemm(const RouterGemmParams& params, bool two_term, DType x_dtype, DType w_dtype,
                   const LaunchContext& ctx);
Status router_weight_split_bf16x2(const Tensor& w_hi, const Tensor& w_lo, const Tensor& w_fp32,
                                  const LaunchContext& ctx);

// ---- G03 / G04 / G05 scaled & quantized -------------------------------------------------
/// One entry for every recipe in GemmRecipe; dispatches to the FP8 / INT8 / FP4 / mixed-input /
/// GGUF families. tactic may be null (cache, then heuristic).
Status scaled_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                   const Workspace& ws, const LaunchContext& ctx);
/// Convenience wrappers that additionally check the recipe family.
Status fp8_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                const Workspace& ws, const LaunchContext& ctx);
Status fp8_bmm(const ScaledGemmArgs& args, const LaunchContext& ctx);
Status int8_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                 const Workspace& ws, const LaunchContext& ctx);
Status fp4_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                const Workspace& ws, const LaunchContext& ctx);
Status mixed_input_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                        const Workspace& ws, const LaunchContext& ctx);
Status marlin_gemm(const MarlinParams& params, quant::MarlinWeightType wtype, DType act_dtype,
                   const LaunchContext& ctx);
/// MoE mode of marlin_gemm (ABI 6, gemm/marlin_moe.cu): rows by sorted_token_ids in blocks of moe_block_size
/// (16, 32 or 64) with expert_ids per block; marlin_gemm forwards here when sorted_token_ids is set.
Status marlin_moe_gemm(const MarlinParams& params, quant::MarlinWeightType wtype, DType act_dtype,
                       const LaunchContext& ctx);
/// GGUF decode GEMV (M <= 16): quantizes `a` into `workspace_q8_1` (uint8, M * K/32 * 36 bytes) and
/// multiplies by ggml blocks.  qweight is a uint8 vector of N * K/block * block_bytes; Q4_0/1, Q5_0/1,
/// Q8_0 and the K-quants are native, the IQ types return kNotImplemented.
Status gguf_gemm(const Tensor& out, const Tensor& a, const Tensor& qweight, GGUFType type,
                 const Tensor& workspace_q8_1, const LaunchContext& ctx);
/// block_q8_1 activation quantization on its own (so MoE callers quantize once per token).
Status gguf_quantize_q8_1(const Tensor& out_q8_1, const Tensor& a, const LaunchContext& ctx);
/// Expert-indexed GGUF GEMV: out[token * topk + j] = W[topk_ids[token, j]] . a_q8_1[token]; negative
/// ids are skipped (the row is left untouched).
Status gguf_moe_gemm(const Tensor& out, const Tensor& a_q8_1, const Tensor& qweight, int64_t expert_stride_bytes,
                     const Tensor& topk_ids, GGUFType type, const LaunchContext& ctx);
Status lut_gemv(const Tensor& out, const Tensor& a, const Tensor& qweight, const Tensor& codebook,
                const Tensor& absmax, const Tensor& absmax_scale, const Tensor& absmax_offset,
                int32_t block_size, const LaunchContext& ctx);
/// Legacy flat adapter entry (BlockScaledGemmParams). It is a thin alias: the call is
/// validated against the same recipe rules and dispatched to fp4_gemm_tcgen05_kernel
/// (SM100a/103a) or fp4_gemm_sm120_kernel (SM120a/121a). There is no separate kernel
/// and no separate header; NVFP4 != MXFP4 (scale dtype, block size, global scale).
Status block_scaled_gemm(const BlockScaledGemmParams& params, GemmBackend backend,
                         const LaunchContext& ctx);

// ---- G06 grouped ------------------------------------------------------------------------
Status grouped_gemm_workspace_requirements(const GroupedGemmParams& params, const GemmTactic* tactic,
                                           const LaunchContext& ctx, WorkspaceRequirements* out);
/// Device-side preparation (problems, tensor maps, task list). Call inside the same graph as the
/// grouped GEMM when expert counts change every step.
Status grouped_gemm_prepare(GroupedGemmParams* params, const GemmTactic& tactic, const Workspace& ws,
                            const LaunchContext& ctx);
Status grouped_gemm(const GroupedGemmParams& params, const GemmTactic* tactic, GemmBackend backend,
                    const Workspace& ws, const LaunchContext& ctx);
Status lora_sgmv(const LoraSgmvParams& shrink, const LoraSgmvParams& expand, DType dtype,
                 const Tensor& shrink_buffer, const LaunchContext& ctx);

// ---- Operand preparation ------------------------------------------------------------------
Status weight_transpose(const Tensor& out, const Tensor& in, const LaunchContext& ctx);
Status glu_weight_interleave(const Tensor& out_weight, const Tensor& out_scales,
                             const Tensor& out_bias, const Tensor& weight, const Tensor& scales,
                             const Tensor& bias, int32_t src_interleave, int32_t dst_interleave,
                             const LaunchContext& ctx);
Status machete_prepack(const Tensor& out, const Tensor& weight, int32_t weight_bits,
                       int32_t atom_layout_id, const LaunchContext& ctx);
Status int8_weight_colsum(const Tensor& colsum, const Tensor& weight, const LaunchContext& ctx);
Status block_scale_swizzle(const Tensor& out, const Tensor& in, bool unswizzle,
                           const LaunchContext& ctx);
Status fp8_scale_to_tma_layout(const Tensor& out, const Tensor& in, const Tensor& expert_offsets,
                               bool pack_ue8m0, const LaunchContext& ctx);
Status fp8_block2d_quant_weight(const Tensor& out, const Tensor& scales, const Tensor& weight,
                                int32_t block_n, int32_t block_k, bool ue8m0,
                                const LaunchContext& ctx);

}  // namespace kops::gemm
