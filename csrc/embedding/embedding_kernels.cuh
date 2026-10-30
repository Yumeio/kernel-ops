#pragma once
// =============================================================================
// kernel-ops | csrc/embedding/embedding_kernels.cuh                   Catalog: X02
// Token embedding (vocab-parallel), learned absolute positions, multimodal
// embedding merge, row gather (last-token selection), logits post-processing.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::embed {

/**
 * embedding_lookup_kernel — out[t] = scale * W[ids[t] - vocab_start] if id in [vocab_start, vocab_end)
 *                           else 0     (vocab-parallel shard; caller all-reduces across TP)
 * scale : 1.0 | sqrt(hidden) (Gemma 1-3) | embedding_multiplier (Granite, MiniCPM scale_emb).
 * Ids outside [0, vocab_size_global) are a caller error; in debug builds they raise kNumericalGuard.
 */
template <typename scalar_t, typename index_t, int VEC_SIZE>
__global__ void embedding_lookup_kernel(
    scalar_t* __restrict__ out,             // [num_tokens, hidden]
    const int64_t out_stride,
    const index_t* __restrict__ token_ids,  // [num_tokens] int32 | int64
    const scalar_t* __restrict__ weight,    // [vocab_end - vocab_start (+pad), hidden]
    const int64_t weight_stride,
    const int64_t vocab_start, const int64_t vocab_end,
    const float scale, const int num_tokens, const int hidden);

/**
 * embedding_with_position_kernel — token embedding + learned absolute position embedding.
 *   out[t] = W_tok[id[t]] + W_pos[pos[t] + position_offset]
 * Models : GPT-2, GPT-Neo, OPT (position_offset = 2), BLOOM uses ALiBi (no pos table),
 *          StarCoder (GPTBigCode), Whisper-decoder style. Same TP masking rule for tokens.
 */
template <typename scalar_t, typename index_t, int VEC_SIZE>
__global__ void embedding_with_position_kernel(
    scalar_t* __restrict__ out,                 // [num_tokens, hidden]
    const index_t* __restrict__ token_ids,      // [num_tokens]
    const int64_t* __restrict__ positions,      // [num_tokens]
    const scalar_t* __restrict__ token_weight,  // [V_shard, hidden]
    const scalar_t* __restrict__ pos_weight,    // [max_position + offset, hidden] (replicated)
    const int64_t vocab_start, const int64_t vocab_end, const int position_offset,
    const int is_first_tp_rank,                 // position rows added on one rank only before all-reduce
    const int num_tokens, const int hidden);

/**
 * scatter_rows_kernel — out[dst_rows[i]] = src[i] (optionally blended).
 * Use   : merge vision/audio encoder outputs into the text embedding stream at placeholder
 *         positions (Qwen-VL, Kimi-K3 MoonViT-V2, Llama-4, Gemma 3, GLM-4.5V); MTP hidden injection.
 */
template <typename scalar_t, int VEC_SIZE>
__global__ void scatter_rows_kernel(
    scalar_t* __restrict__ out,              // [num_out_rows, hidden]
    const int64_t out_stride,
    const scalar_t* __restrict__ src,        // [num_src_rows, hidden]
    const int64_t src_stride,
    const int32_t* __restrict__ dst_rows,    // [num_src_rows]
    const int num_src_rows, const int hidden);

/**
 * gather_rows_kernel — out[i] = in[rows[i]]. Selects the hidden states whose logits are needed
 * (last token of each request, all tokens of spec-decode verification, prompt logprobs).
 */
template <typename scalar_t, int VEC_SIZE>
__global__ void gather_rows_kernel(
    scalar_t* __restrict__ out,              // [num_rows, hidden]
    const int64_t out_stride,
    const scalar_t* __restrict__ input,      // [num_tokens, hidden]
    const int64_t input_stride,
    const int32_t* __restrict__ rows,        // [num_rows]
    const int num_rows, const int hidden);

/**
 * logits_postprocess_kernel — LM-head output fix-ups in one pass, written as fp32.
 *   x = logits * scale                  (Cohere logit_scale, Granite 1/logits_scaling, MiniCPM 1/(H/dim_base))
 *   x = cap * tanh(x / cap)             (Gemma 2 final_logit_softcapping 30, Grok-1/2; cap = 0 off)
 *   x = -inf for v >= vocab_size        (TP padded vocabulary / padded embedding rows)
 * Input may be a vocab shard [num_rows, V_shard] with column offset vocab_start; the
 * all-gather across ranks is done by communication/ before or after this kernel.
 */
template <typename in_t>
__global__ void logits_postprocess_kernel(
    float* __restrict__ out,                 // [num_rows, V_shard]
    const int64_t out_stride,
    const in_t* __restrict__ logits,         // [num_rows, V_shard]; may alias out if in_t == float
    const int64_t logits_stride,
    const float scale, const float final_softcap,
    const int64_t vocab_start, const int64_t vocab_size,  // global real vocab size
    const int num_rows, const int v_shard);

}  // namespace kops::embed
