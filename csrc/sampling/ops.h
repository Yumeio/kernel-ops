#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/ops.h                               Catalog: S01-S03
// Requires: common/*, sampling/*.cuh (including fused_sampler_kernels.cuh).
// =============================================================================

namespace kops::sample {

/// Capture-stable plan for the fused sampler. Resolved by sampling_plan() before
/// graph capture; run() must not re-resolve it. `splits == 1` is the one-CTA-per-
/// row path; larger values split a row across CTAs for small batches.
struct SamplingPlan {
  int32_t splits = 1;
  int32_t candidate_cap = 0;
  int32_t hist_bins = 0;
  int32_t block_threads = 256;
  uint32_t flags = 0;
  uint64_t capture_generation = 0;
};

/// Workspace for one sample() call: the CTAS_PER_ROW softmax state when the vocabulary
/// is split, plus the per-row rejection-sampling scratch. Callers must query this
/// before reserving (the sampler may switch split counts by vocab size).
Status sampling_workspace_requirements(int32_t max_batch, int32_t max_vocab, int32_t ctas_per_row,
                                       WorkspaceRequirements* out);

/// One-call sampler: (optional processors already applied) -> greedy rows via argmax, others via
/// top-k/top-p/min-p rejection sampling or Gumbel-max, chosen per capability. Device outputs only.
/// The workspace holds the fp32 probability row of the call (`num_tokens * vocab * 4` bytes,
/// plus the optional split-softmax state); sampling_workspace_requirements() reports it. A row
/// with no admissible token writes `out_ids = -1`.
Status sample(const Tensor& out_ids, const Tensor& logits, const SamplingParams& params,
              const Workspace& ws, const LaunchContext& ctx);

/// Resolve the fused-sampler plan before capture. The fused path implements greedy,
/// unfiltered categorical, top-k, min-p and top-p (scalar or per-row, sequential or joint).
/// The plan is a pure function of the logits/params metadata, the device and
/// ctx.capture_generation, so it is stable across replay; sample_fused() rejects a plan
/// resolved for other filter settings (kPlanInvalidated). `splits` (a power of two <= 16)
/// splits each row over that many CTAs launched cooperatively, with tokens bit-identical to
/// splits == 1; callers may lower it, not raise it past what stays co-resident. A split row must
/// be 16-byte vectorizable and keep at least 4096 tokens per CTA, so the auto plan stops before
/// the per-CTA shard gets too small to pay for the row barrier.
Status sampling_plan(const Tensor& logits, const SamplingParams& params, const LaunchContext& ctx,
                     SamplingPlan* out);

/// Workspace for sample_fused(). One-CTA-per-row plans (splits == 1) keep their state in shared
/// memory and need nothing. Split-row plans (splits > 1, chosen for small batches) need
/// max_batch per-row exchange areas in device_scratch and one 8-byte barrier counter per row in
/// device_persistent, zero-filled once before the first call (any split count may reuse it).
/// One workspace serves one call at a time; concurrent streams need their own.
Status sampling_fused_workspace_requirements(int32_t max_batch, int32_t max_vocab,
                                             const SamplingPlan& plan,
                                             WorkspaceRequirements* out);

/// Fused sampler: greedy or unfiltered full-vocab categorical sampling in one
/// launch, with optional penalty processors folded into the logits load. NaN/-inf
/// policy and truncation order match sample(); top-k keeps the whole tie group and
/// draws match sample() in distribution only (CONTRACTS.md §1c). Rows with only
/// top-p active use a fixed-point mass histogram to find the boundary bin, then race
/// the bins above it and refine the boundary candidates; the nucleus is unchanged.
Status sample_fused(const Tensor& out_ids, const Tensor& logits,
                    const FusedSamplingParams& params, const SamplingPlan& plan,
                    const Workspace& ws, const LaunchContext& ctx);

/// Gumbel-max sampler: out_ids[b] = argmax(logits[b] / T_b + Gumbel(u_id)) over the top-k/min-p
/// admissible set. Same counter RNG and tie-break as the other samplers.
Status gumbel_max_sample(const Tensor& out_ids, const Tensor& logits, const SamplingParams& params,
                         const LaunchContext& ctx);
Status softmax(const Tensor& probs, const Tensor& logits, const Tensor& temperature,
               const Workspace& ws, const LaunchContext& ctx);
Status argmax(const Tensor& out_ids, const Tensor& out_logprob, const Tensor& logits,
              const LaunchContext& ctx);
Status top_k_renorm_probs(const Tensor& out, const Tensor& probs, const Tensor& top_k,
                          int32_t default_top_k, const LaunchContext& ctx);
Status top_p_renorm_probs(const Tensor& out, const Tensor& probs, const Tensor& top_p,
                          float default_top_p, const LaunchContext& ctx);
Status top_k_mask_logits(const Tensor& out, const Tensor& logits, const Tensor& top_k,
                         int32_t default_top_k, const LaunchContext& ctx);
Status compute_logprobs(const Tensor& chosen_logprob, const Tensor& chosen_rank,
                        const Tensor& top_ids, const Tensor& top_logprobs, const Tensor& logits,
                        const Tensor& chosen_ids, int32_t top_n, const LaunchContext& ctx);

Status token_bincount(const Tensor& counts, const Tensor& prompt_mask, const Tensor& token_ids,
                      const Tensor& token_indptr, const Tensor& is_prompt, const LaunchContext& ctx);
Status apply_penalties(const Tensor& logits, const Tensor& output_counts, const Tensor& prompt_mask,
                       const Tensor& repetition, const Tensor& frequency, const Tensor& presence,
                       const LaunchContext& ctx);
Status apply_token_bitmask(const Tensor& logits, const Tensor& bitmask, const Tensor& rows,
                           const LaunchContext& ctx);
Status apply_logit_bias(const Tensor& logits, const Tensor& bias_indptr, const Tensor& token_ids,
                        const Tensor& values, const LaunchContext& ctx);

Status rejection_sample(const SpecVerifyParams& params, const LaunchContext& ctx);
Status tree_verify(const TreeVerifyParams& params, const LaunchContext& ctx);
Status build_tree_attention_mask(const Tensor& packed_mask, const Tensor& mask_indptr,
                                 const Tensor& position_offsets, const Tensor& parent,
                                 const Tensor& prefix_lens, const LaunchContext& ctx);

}  // namespace kops::sample
