#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/logits_processor_kernels.cuh            Catalog: S02
// In-place logit processors. Grammar compilation (xgrammar / llguidance / outlines)
// stays in the engine; kernels only APPLY the resulting bitmask.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::sample {

/**
 * token_bincount_kernel — counts[b, v] = occurrences of v in the request's output tokens
 * (and prompt_mask bit set for tokens seen in the prompt). Incremental mode adds only the
 * newly generated tokens of this step.
 */
__global__ void token_bincount_kernel(
    int32_t* __restrict__ counts,              // [B, V] in/out
    uint32_t* __restrict__ prompt_mask,        // [B, ceil(V/32)] in/out or null
    const int32_t* __restrict__ token_ids,     // ragged ids
    const int32_t* __restrict__ token_indptr,  // [B + 1]
    const uint8_t* __restrict__ is_prompt,     // [num_ids] or null (all output tokens)
    const int batch_size, const int vocab_size);

/**
 * apply_penalties_kernel — HF/OpenAI semantics:
 *   repetition (tokens in prompt or output): l = l > 0 ? l / r : l * r
 *   frequency: l -= f * count;   presence: l -= p * (count > 0)
 */
template <typename logits_t, int VEC_SIZE>
__global__ void apply_penalties_kernel(
    logits_t* __restrict__ logits,             // [B, V] in place
    const int64_t logits_stride,
    const int32_t* __restrict__ output_counts, // [B, V]
    const uint32_t* __restrict__ prompt_mask,  // [B, ceil(V/32)] or null
    const float* __restrict__ repetition,      // [B] (1 = off)
    const float* __restrict__ frequency,       // [B] (0 = off)
    const float* __restrict__ presence,        // [B] (0 = off)
    const int batch_size, const int vocab_size);

/**
 * apply_token_bitmask_kernel — structured output / allowed tokens: bit v of row b set = allowed;
 * disallowed logits -> -inf. rows (optional) maps bitmask rows to logits rows (only requests with a
 * grammar carry a mask; spec-decode verification has one mask per draft position).
 */
template <typename logits_t, int VEC_SIZE>
__global__ void apply_token_bitmask_kernel(
    logits_t* __restrict__ logits,             // [num_logit_rows, V]
    const int64_t logits_stride,
    const int32_t* __restrict__ bitmask,       // [num_mask_rows, ceil(V/32)]
    const int32_t* __restrict__ rows,          // [num_mask_rows] target logits row, or null (identity)
    const int num_mask_rows, const int vocab_size);

/**
 * apply_logit_bias_kernel — sparse additive bias (OpenAI logit_bias, bad_words / stop-token bans
 * with -inf, min_tokens: EOS banned until length reached).
 */
template <typename logits_t>
__global__ void apply_logit_bias_kernel(
    logits_t* __restrict__ logits,             // [B, V]
    const int64_t logits_stride,
    const int32_t* __restrict__ bias_indptr,   // [B + 1]
    const int32_t* __restrict__ bias_token_ids,// [nnz]
    const float* __restrict__ bias_values,     // [nnz]
    const int batch_size);

}  // namespace kops::sample
