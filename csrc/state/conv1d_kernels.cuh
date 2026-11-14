#pragma once
// =============================================================================
// kernel-ops | csrc/state/conv1d_kernels.cuh                         Catalog: L01
// Causal depthwise conv1d for hybrid models (Mamba-1/2, GDN, KDA short conv,
// LFM2 short-conv blocks). Conv history lives in caller-owned state slots.
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh.
// =============================================================================

namespace kops::state {

/// Arguments of causal_conv1d_fwd_kernel (prefill / chunked prefill, varlen). Pointers are device
/// pointers, strides count elements. x / out are [total_tokens, dim] (dim stride 1 and 16-byte rows
/// take the vectorized path, any non-negative strides work); weight is [dim, width] (width 2..4);
/// conv_states is [num_slots, dim, width - 1] with any strides, in the dtype of x.
struct Conv1dFwdParams {
  const void* x;
  int64_t x_token_stride, x_dim_stride;
  const void* weight;                // dtype `weight_dtype` (x's dtype or fp32)
  int64_t weight_dim_stride, weight_width_stride;
  const void* bias;                  // [dim] contiguous, weight dtype, or null
  void* out;                         // same shape as x, disjoint from x (token chunks run in parallel)
  int64_t out_token_stride, out_dim_stride;
  const int32_t* query_start_loc;    // [B + 1] token ranges per sequence (monotonic, ends at total_tokens)
  const int32_t* cache_indices;      // [B] slot per sequence, null = slot b
  const uint8_t* has_initial_state;  // [B] 1 = history from the slot, 0 / null = zero history
  void* conv_states;                 // null = stateless (zero history, no final state)
  int64_t conv_state_slot_stride, conv_state_dim_stride, conv_state_len_stride;
  int32_t num_slots;
  // Prefix-cache snapshots: the state after (j + 1) * snapshot_interval tokens goes to
  // snapshot_slots[b * max_snapshots + j]; destinations outside [0, num_slots) are dropped. null = off.
  const int32_t* snapshot_slots;
  int32_t snapshot_interval, max_snapshots;
  int32_t batch_size, dim, width, total_tokens;
  int32_t max_seqlen;                // bound of every sequence (0 = total_tokens); sizes the grid only
  int32_t pad_slot_id;               // a slot < 0, >= num_slots or == pad_slot_id skips the sequence
  ActFn act;                         // kSiLU (Mamba, GDN, KDA) | kIdentity (LFM2)
  DType weight_dtype;                // kInvalid = dtype of x
};

/**
 * causal_conv1d_fwd_kernel — y[t, c] = act(bias[c] + sum_{j<W} w[c, j] * x[t - W + 1 + j, c]),
 * using the slot's conv_states for t < 0 and replacing them with the last W-1 inputs (raw bits).
 * Taps accumulate in fp32 as bias, w[0] x[t-W+1], ..., w[W-1] x[t] (fused multiply-adds), rounded once.
 * Launch : grid = (ceil(dim / (128 * kVec)), ceil(max_seqlen / 64), B), block = 128. Chunk 0 of a
 * sequence owns its slot (reads the history, writes snapshots and the final state).
 */
template <typename input_t, typename weight_t, int kWidth, int kVec>
__global__ void causal_conv1d_fwd_kernel(const __grid_constant__ Conv1dFwdParams params);

/// Arguments of causal_conv1d_update_kernel (decode, 1..S tokens per sequence). x / out are
/// [B, seqlen, dim] with dim stride 1; out is x itself (in place) or disjoint from it.
struct Conv1dUpdateParams {
  const void* x;
  int64_t x_batch_stride, x_token_stride;
  const void* weight;
  int64_t weight_dim_stride, weight_width_stride;
  const void* bias;                  // [dim] or null
  void* out;
  int64_t out_batch_stride, out_token_stride;
  void* conv_states;                 // [num_slots, dim, width - 1]
  int64_t conv_state_slot_stride, conv_state_dim_stride, conv_state_len_stride;
  int32_t num_slots;
  const int32_t* cache_indices;      // [B] live slot, null = slot b
  // Speculative decoding: slot receiving the state after token i, [B, seqlen] (invalid slots dropped);
  // state_commit_accepted (B10) rolls back from them. null = off.
  const int32_t* intermediate_state_slots;
  int32_t batch_size, seqlen, dim, width;
  int32_t pad_slot_id;
  ActFn act;
  DType weight_dtype;                // kInvalid = dtype of x
};

/// causal_conv1d_update_kernel — decode-time conv with in-place state update. A thread runs all
/// S tokens of its channels and reads each x before writing the matching out, so out may be x.
/// Launch : grid = (ceil(dim / (128 * kVec)), B), block = 128.
template <typename input_t, typename weight_t, int kWidth, int kVec>
__global__ void causal_conv1d_update_kernel(const __grid_constant__ Conv1dUpdateParams params);

}  // namespace kops::state
