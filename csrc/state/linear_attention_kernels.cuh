#pragma once
// =============================================================================
// kernel-ops | csrc/state/linear_attention_kernels.cuh       Catalog: L02 (decay linear attn)
// Lightning attention / decayed linear attention (MiniMax-01, MiniMax-M1 hybrid layers;
// generic RetNet-style fixed decay):
//   S_t = exp(-s_h) S_{t-1} + k_t^T v_t,   o_t = q_t S_t     (S: Dk x Dv per head, fp32)
// Prefill: intra-block quadratic part with decay mask + inter-block state passing.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::state {

struct LinearAttentionParams {
  const void* q;                     // [T, H, Dk]
  const void* k;                     // [T, H, Dk]
  const void* v;                     // [T, H, Dv]
  int64_t token_stride_q, token_stride_k, token_stride_v;
  const float* slope;                // [H] decay rate s_h (ALiBi-like geometric slopes in MiniMax)
  const int32_t* cu_seqlens;         // [B + 1]
  const int32_t* state_indices;      // [B]
  const uint8_t* has_initial_state;  // [B]
  float* state_pool;                 // [num_slots, H, Dk, Dv]
  const int32_t* snapshot_slots;     // prefix-cache snapshots or null
  int32_t snapshot_interval, max_snapshots;
  void* o;                           // [T, H, Dv]
  int32_t total_tokens, batch_size, num_heads, head_dim_k, head_dim_v, block_size;
  int32_t pad_slot_id;
};

/// Chunked prefill (block_size 64..256): intra-block with decay mask + state passing.
template <typename input_t, int HEAD_DIM, int BLOCK>
__global__ void lightning_attention_prefill_kernel(const __grid_constant__ LinearAttentionParams params);

/// Decode: per (sequence, head) state update and readout.
template <typename input_t, int HEAD_DIM, int BLOCK_V>
__global__ void linear_attention_decode_kernel(const __grid_constant__ LinearAttentionParams params);

}  // namespace kops::state
