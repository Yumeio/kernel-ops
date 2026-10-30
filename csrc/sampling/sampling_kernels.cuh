#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/sampling_kernels.cuh                    Catalog: S01
// Token selection on device. All outputs stay on device (no host sync); every
// random draw uses the counter RNG (common/rng.cuh), so results do not depend
// on batch composition, splits or replay.
// Semantics are explicit: greedy / full-vocab categorical / top-k / top-p / min-p,
// and the ORDER of truncations (top-k then top-p by default, joint optional).
// Requires: common/macros.cuh, common/dtype.cuh, common/rng.cuh.
// =============================================================================

namespace kops::sample {

/// Per-request sampling parameters (device arrays, [B]; null = use the scalar default).
struct SamplingParams {
  const float* temperature;     // <= 0 means greedy for that row
  const int32_t* top_k;         // <= 0 or >= V means disabled
  const float* top_p;           // >= 1 disabled
  const float* min_p;           // <= 0 disabled
  float default_temperature;
  int32_t default_top_k;
  float default_top_p;
  float default_min_p;
  bool joint_topk_topp;         // false: top-k first then top-p on the renormalized remainder
  RngParams rng;
};

/**
 * softmax_kernel — probs[b] = softmax(logits[b] / T_b) in fp32 with online max/sum. For V up to
 * 256K (Gemma 262K, Kimi 160K, GLM 151K, Qwen 152K) rows are split across CTAS_PER_ROW CTAs with a
 * second merge phase (SoftmaxState algebra), deterministic.
 */
template <typename in_t, int VEC_SIZE, int CTAS_PER_ROW>
__global__ void softmax_kernel(
    float* __restrict__ probs,                 // [B, V]
    const in_t* __restrict__ logits,           // [B, V]
    const int64_t logits_stride,
    const float* __restrict__ temperature,     // [B] or null (1.0)
    float* __restrict__ row_state_workspace,   // [B, CTAS_PER_ROW, 2] when CTAS_PER_ROW > 1
    const int batch_size, const int vocab_size);

/**
 * argmax_kernel — greedy: out_ids[b] = argmax logits[b], ties -> lowest id; optional logprob of
 * the winner (needs the row LSE, computed in the same pass).
 */
template <typename in_t, int VEC_SIZE>
__global__ void argmax_kernel(
    int32_t* __restrict__ out_ids,             // [B]
    float* __restrict__ out_logprob,           // [B] or null
    const in_t* __restrict__ logits,           // [B, V]
    const int64_t logits_stride,
    const int batch_size, const int vocab_size);

/**
 * top_k_top_p_sampling_kernel — sorting-free rejection sampling over probabilities (FlashInfer
 * dual-pivot style): draw u, binary-search pivots until the sampled token satisfies top-k / top-p /
 * min-p, bounded by max_rounds; rows exceeding it fall back to an exact radix select.
 * Exactly matches sampling from the renormalized truncated distribution.
 */
template <int VEC_SIZE, int MAX_ROUNDS>
__global__ void top_k_top_p_sampling_kernel(
    int32_t* __restrict__ out_ids,             // [B]
    uint8_t* __restrict__ out_valid,           // [B] 0 if the row had no admissible token
    const float* __restrict__ probs,           // [B, V]
    const SamplingParams params,
    const int batch_size, const int vocab_size);

/**
 * gumbel_max_sampling_kernel — sample directly from logits: argmax(logits/T + Gumbel(u)) over the
 * admissible set (optional top-k candidates computed in the same pass). No softmax materialized;
 * equivalent in distribution to categorical sampling (HPC-Ops fused sampler semantics, generalized).
 */
template <typename in_t, int VEC_SIZE, int MAX_TOPK_CANDIDATES>
__global__ void gumbel_max_sampling_kernel(
    int32_t* __restrict__ out_ids,             // [B]
    const in_t* __restrict__ logits,           // [B, V]
    const int64_t logits_stride,
    const SamplingParams params,
    const int batch_size, const int vocab_size);

/// Renormalization primitives (also used by speculative decoding to build target/draft distributions).
template <int VEC_SIZE>
__global__ void top_k_renorm_probs_kernel(
    float* __restrict__ out, const float* __restrict__ probs, const int32_t* __restrict__ top_k,
    const int32_t default_top_k, const int batch_size, const int vocab_size);

template <int VEC_SIZE>
__global__ void top_p_renorm_probs_kernel(
    float* __restrict__ out, const float* __restrict__ probs, const float* __restrict__ top_p,
    const float default_top_p, const int batch_size, const int vocab_size);

/// top_k_mask_logits_kernel — logits outside the top-k set -> -inf (keeps logits, no softmax).
template <typename in_t, int VEC_SIZE>
__global__ void top_k_mask_logits_kernel(
    in_t* __restrict__ out, const in_t* __restrict__ logits, const int32_t* __restrict__ top_k,
    const int32_t default_top_k, const int batch_size, const int vocab_size);

/**
 * compute_logprobs_kernel — log_softmax row statistics + requested outputs:
 *   chosen_logprob[b] (token out_ids[b]), chosen_rank[b], top-n (ids, logprobs) with n <= 20.
 * Prompt logprobs use the same kernel on gathered rows. Raw (pre-processing) or processed
 * logits are chosen by the caller (vLLM logprobs_mode semantics).
 */
template <typename in_t, int VEC_SIZE, int MAX_TOP_N>
__global__ void compute_logprobs_kernel(
    float* __restrict__ chosen_logprob,        // [B]
    int32_t* __restrict__ chosen_rank,         // [B] or null
    int32_t* __restrict__ top_ids,             // [B, top_n] or null
    float* __restrict__ top_logprobs,          // [B, top_n] or null
    const in_t* __restrict__ logits,           // [B, V]
    const int64_t logits_stride,
    const int32_t* __restrict__ chosen_ids,    // [B]
    const int top_n, const int batch_size, const int vocab_size);

}  // namespace kops::sample
