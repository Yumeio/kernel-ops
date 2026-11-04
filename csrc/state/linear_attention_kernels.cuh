#pragma once
// =============================================================================
// kernel-ops | csrc/state/linear_attention_kernels.cuh       Catalog: L02 (decay linear attn)
// Lightning attention / decayed linear attention (MiniMax-01, MiniMax-M1 hybrid layers;
// generic RetNet-style fixed decay):
//   S_t = exp(-s_h) S_{t-1} + k_t^T v_t,   o_t = scale * q_t S_t     (S: Dk x Dv per head, fp32)
// Prefill: intra-block quadratic part with decay mask + inter-block state passing.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::state {

/// q, k are [T, H, Dk] and v, o [T, H, Dv]: the Dk / Dv axes are contiguous, token and head
/// strides are given in elements. Decode reads one token per sequence (token index = b).
struct LinearAttentionParams {
  const void* q;
  const void* k;
  const void* v;
  int64_t q_token_stride, q_head_stride, k_token_stride, k_head_stride, v_token_stride, v_head_stride;
  const float* slope;                // [H] decay rate s_h (ALiBi-like geometric slopes in MiniMax), >= 0
  float scale;
  float* state_pool;                 // [num_slots, H, Dk, Dv], contiguous within a slot
  int64_t state_slot_stride;
  const int32_t* state_indices;      // [B] slot per sequence, null = b
  void* o;                           // [T, H, Dv], disjoint from q, k, v
  int64_t o_token_stride, o_head_stride;
  const int32_t* cu_seqlens;         // [B + 1] (prefill)
  const uint8_t* has_initial_state;  // [B], null = every sequence starts from its slot (prefill)
  const int32_t* snapshot_slots;     // [B, max_snapshots] (prefill) or null
  int32_t snapshot_interval, max_snapshots;  // interval: positive multiple of 64
  int32_t num_slots, total_tokens, batch_size, num_heads, head_dim_k, head_dim_v;
  int32_t pad_slot_id;
};

/// Chunked prefill, blocks of 64 tokens: the CTA keeps S[:, Dv tile] of one (sequence, head) in
/// shared memory. Launch : grid = (ceil(Dv / kDvTile), H, B), block = 256, dynamic smem.
template <typename input_t, int kDvTile>
__global__ void lightning_attention_prefill_kernel(const __grid_constant__ LinearAttentionParams params);

/// Decode: per (sequence, head) state update and readout, one thread per value column.
/// Launch : grid = (ceil(Dv / 128), H, B), block = 128.
template <typename input_t>
__global__ void linear_attention_decode_kernel(const __grid_constant__ LinearAttentionParams params);

}  // namespace kops::state
