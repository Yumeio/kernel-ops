#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/ops.h                               Catalog: S01-S03
// Requires: common/*, sampling/*.cuh.
// =============================================================================

namespace kops::sample {

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
