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

// ---- G01 dense 16/32-bit -------------------------------------------------------------
Status gemm(const GemmProblem& problem, const GemmArgs& args, GemmBackend backend,
            const LaunchContext& ctx);
Status gemm_with_tactic(const ScaledGemmArgs& args, const GemmTactic& tactic, const Workspace& ws,
                        const LaunchContext& ctx);
Status dual_gemm(const DualGemmParams& params, DType dtype, const LaunchContext& ctx);
Status gemv_argmax(const GemvArgmaxParams& params, DType hidden_dtype, DType weight_dtype,
                   const LaunchContext& ctx);
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
Status gguf_gemm(const Tensor& out, const Tensor& a, const Tensor& qweight, GGUFType type,
                 const Tensor& workspace_q8_1, const LaunchContext& ctx);
Status lut_gemv(const Tensor& out, const Tensor& a, const Tensor& qweight, const Tensor& codebook,
                const Tensor& absmax, const Tensor& absmax_scale, const Tensor& absmax_offset,
                int32_t block_size, const LaunchContext& ctx);
/// Legacy flat block-scaled entry (adapters).
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
