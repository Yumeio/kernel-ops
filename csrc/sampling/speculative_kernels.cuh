#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/speculative_kernels.cuh                 Catalog: S03
// Speculative decoding verification: exact rejection sampling (output distribution
// equals the target model), greedy verification, tree (EAGLE-2/3) acceptance, and
// helpers to build verification masks. Drafts may come from MTP heads (DeepSeek
// V3/V4, GLM-4.5, Qwen3-Next, MiMo), EAGLE, n-gram / suffix decoding, or DSpark-style drafters.
// Requires: common/macros.cuh, common/rng.cuh.
// =============================================================================

namespace kops::sample {

// Row layout, acceptance rules, output padding and the counter-RNG draw map are
// part of the B10 contract in CONTRACTS.md §1g.  In particular:
//   * target_probs[j] predicts draft j; the bonus row is total_drafts + b;
//   * draft_probs == null selects the deterministic-drafter (target-only) rule;
//   * output_token_ids is [B, max_draft + 1] with -1 padding.

/// Arguments of chain (linear draft) verification.
struct SpecVerifyParams {
  const int32_t* draft_token_ids;     // [num_draft_tokens] ragged
  const int32_t* cu_num_draft;        // [B + 1]
  const float* draft_probs;           // [num_draft_tokens, V] or null (deterministic drafter: q = one-hot)
  const float* target_probs;          // [num_draft_tokens + B, V] (k+1 rows per request incl. bonus)
  const int32_t* target_argmax;       // [num_draft_tokens + B] (greedy mode) or null
  const uint8_t* is_greedy;           // [B] per-request greedy flag
  RngParams rng;                      // uniforms for acceptance and recovery
  int32_t* output_token_ids;          // [B, max_draft + 1], -1 padding
  int32_t* num_accepted;              // [B] accepted drafts (0..k); emitted tokens = num_accepted + 1
  int32_t batch_size, max_draft, vocab_size;
};

/**
 * rejection_sample_kernel — accept draft x_i with prob min(1, p(x_i)/q(x_i)); at the first rejection
 * sample from norm(max(0, p - q)); if all accepted sample the bonus token from p_{k+1}.
 * Greedy rows: accept iff target_argmax == draft. Deterministic given RngParams.
 */
template <int VEC_SIZE>
__global__ void rejection_sample_kernel(const __grid_constant__ SpecVerifyParams params);

/// Tree verification (EAGLE-2/3, Medusa): tree encoded as first-child / next-sibling
/// arrays.  Node 0 is the current/root token (not emitted); retrieve_index[node]
/// selects the target row *after* that node; a child is verified with its parent's
/// row.  CONTRACTS.md §1g fixes the modes, accept_index layout and RNG draws.
struct TreeVerifyParams {
  const int32_t* candidates;          // [B, num_nodes] draft token per node (node 0 = root/current)
  const int32_t* retrieve_index;      // [B, num_nodes] row of target logits for each node
  const int32_t* first_child;         // [B, num_nodes] (-1 none)
  const int32_t* next_sibling;        // [B, num_nodes] (-1 none)
  const float* target_probs;          // [B * num_nodes, V] or null (greedy)
  const int32_t* target_argmax;       // [B * num_nodes] (greedy) or null
  const float* draft_probs;           // [B * num_nodes, V] or null
  RngParams rng;
  int32_t* accept_index;              // [B, max_depth + 1] accepted node indices along the path
  int32_t* accept_length;             // [B]
  int32_t* output_token_ids;          // [B, max_depth + 1]
  int32_t batch_size, num_nodes, max_depth, vocab_size;
};

template <int VEC_SIZE>
__global__ void tree_verify_kernel(const __grid_constant__ TreeVerifyParams params);

/**
 * build_tree_attention_mask_kernel — packed custom mask (attention MaskMode::kCustom)
 * where node i sees the committed prefix plus its ancestor-or-self nodes; also
 * writes per-node depth offsets.  parent names draft nodes only (-1 = prefix).
 * Rows are [prefix_lens[b] + num_nodes] bits, little-endian, with no padding
 * between query rows; mask_indptr[b] is the bit base of request b.  The engine
 * owns the packed_mask capacity and alignment (CONTRACTS.md §1g).
 */
__global__ void build_tree_attention_mask_kernel(
    uint8_t* __restrict__ packed_mask,         // bit-packed [B, num_nodes, prefix_len + num_nodes]
    int64_t* __restrict__ mask_indptr,         // [B + 1] bit offsets
    int32_t* __restrict__ position_offsets,    // [B, num_nodes] depth of each node
    const int32_t* __restrict__ parent,        // [B, num_nodes] parent node (-1 root)
    const int32_t* __restrict__ prefix_lens,   // [B]
    const int batch_size, const int num_nodes);

}  // namespace kops::sample
