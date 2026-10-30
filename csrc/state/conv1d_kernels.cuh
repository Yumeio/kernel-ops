#pragma once
// =============================================================================
// kernel-ops | csrc/state/conv1d_kernels.cuh                         Catalog: L01
// Causal depthwise conv1d for hybrid models (Mamba-1/2, GDN, KDA short conv,
// LFM2 short-conv blocks). Conv history lives in caller-owned state slots.
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh.
// =============================================================================

namespace kops::state {

/// Arguments of causal_conv1d_fwd_kernel (prefill / chunked prefill, varlen).
struct Conv1dFwdParams {
  const void* x;                     // channel-last [total_tokens, dim] (or channel-first [dim, total_tokens])
  int64_t x_token_stride, x_dim_stride;
  const void* weight;                // [dim, width] (width 2..4)
  const void* bias;                  // [dim] or null
  void* out;                         // same layout as x (may alias x)
  int64_t out_token_stride, out_dim_stride;
  const int32_t* query_start_loc;    // [B + 1] token ranges per sequence
  const int32_t* cache_indices;      // [B] state slot per sequence; pad_slot_id = skip
  const uint8_t* has_initial_state;  // [B] 1 = continue from conv_states, 0 = zero history
  void* conv_states;                 // [num_slots, dim, width - 1] in/out
  int64_t conv_state_slot_stride;
  // Prefix-cache snapshots: write the conv state after every `snapshot_interval` tokens into
  // snapshot_slots[b * max_snapshots + j] (KDA/GDN fine-grained prefix caching). null = off.
  const int32_t* snapshot_slots;
  int32_t snapshot_interval, max_snapshots;
  int32_t batch_size, dim, width;
  int32_t pad_slot_id;
  ActFn act;                         // kSiLU (Mamba, GDN, KDA) | kIdentity (LFM2)
};

/**
 * causal_conv1d_fwd_kernel — y[t, c] = act(bias[c] + sum_{j<W} w[c, j] * x[t - W + 1 + j, c]),
 * using the slot's conv_states for t < 0 and updating them with the last W-1 inputs.
 * Launch : grid = (B, ceil(dim / CHANNELS_PER_CTA), num_token_chunks), block = kNThreads.
 */
template <typename input_t, typename weight_t, int kWidth, int kNThreads, bool kChannelLast>
__global__ void causal_conv1d_fwd_kernel(const KOPS_GRID_CONST Conv1dFwdParams params);

/// Arguments of causal_conv1d_update_kernel (decode, 1..k tokens per sequence).
struct Conv1dUpdateParams {
  void* x;                           // [B, seqlen, dim] in; overwritten with output when out == x
  int64_t x_batch_stride, x_token_stride;
  const void* weight;                // [dim, width]
  const void* bias;                  // [dim] or null
  void* out;
  void* conv_states;                 // [num_slots, dim, state_len] (state_len >= width - 1)
  int64_t conv_state_slot_stride;
  const int32_t* cache_indices;      // [B]
  const int32_t* cache_seqlens;      // [B] circular-buffer write position, or null (shift buffer)
  // Speculative decoding: tokens per sequence = seqlen (drafts); after verification only
  // num_accepted_tokens[b] are committed. intermediate_state_slots keep per-draft states.
  const int32_t* num_accepted_tokens;// [B] or null
  const int32_t* intermediate_state_slots;  // [B, seqlen] or null
  int32_t batch_size, seqlen, dim, width, state_len;
  int32_t pad_slot_id;
  ActFn act;
};

/// causal_conv1d_update_kernel — decode-time conv with in-place state update.
template <typename input_t, typename weight_t, int kWidth, int kNThreads>
__global__ void causal_conv1d_update_kernel(const KOPS_GRID_CONST Conv1dUpdateParams params);

}  // namespace kops::state
