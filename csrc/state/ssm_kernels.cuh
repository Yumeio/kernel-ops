#pragma once
// =============================================================================
// kernel-ops | csrc/state/ssm_kernels.cuh                         Catalog: L02 (SSM)
// Selective state-space models.
//   Mamba-1 : Falcon-Mamba, Jamba / Jamba-1.5 (Mamba layers)        selective scan, per-channel A [D, N]
//   Mamba-2 : Codestral-Mamba, Nemotron-H, Falcon-H1, Granite-4.0-H, Zamba2, Bamba
//             SSD chunked scan, scalar A per head, state [nheads, headdim, dstate]
// States are fp32 in caller-owned slots; slot mapping per sequence. Strides count elements; the
// innermost axis of every activation operand (headdim / dstate) is contiguous unless noted.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::state {

/// Mamba-1 selective scan (prefill, varlen). Operands are [dim, T] / [G, N, T] views with arbitrary
/// element strides (token-major memory, i.e. dim or N contiguous, is staged with coalesced loads).
struct SelectiveScanParams {
  const void* u;            // [dim, T] input after conv
  int64_t u_dim_stride, u_token_stride;
  const void* delta;        // [dim, T]
  int64_t delta_dim_stride, delta_token_stride;
  const float* A;           // [dim, dstate] (negative, = -exp(A_log))
  int64_t A_dim_stride;
  const void* B;            // [ngroups, dstate, T]
  int64_t B_group_stride, B_state_stride, B_token_stride;
  const void* C;            // [ngroups, dstate, T]
  int64_t C_group_stride, C_state_stride, C_token_stride;
  const float* D;           // [dim] skip or null
  const void* z;            // [dim, T] gate or null
  int64_t z_dim_stride, z_token_stride;
  const float* delta_bias;  // [dim] or null
  bool delta_softplus;
  void* out;                // [dim, T], disjoint from every input
  int64_t out_dim_stride, out_token_stride;
  float* ssm_states;        // [num_slots, dim, dstate] in/out (dstate contiguous)
  int64_t ssm_state_slot_stride, ssm_state_dim_stride;
  const int32_t* query_start_loc;    // [B + 1]
  const int32_t* cache_indices;      // [B] slot per sequence, null = b
  const uint8_t* has_initial_state;  // [B], null = every sequence starts from its slot
  int32_t num_slots, batch_size, dim, dstate, ngroups, total_tokens;
  int32_t pad_slot_id;
};

/**
 * selective_scan_fwd_kernel — h_t = exp(delta_t * A) h_{t-1} + delta_t * B_t u_t;
 * y_t = C_t h_t + D u_t; y *= silu(z). Parallel affine scan over tiles of 128 tokens counted from
 * the sequence start (state carried in fp32), dstate reduced in increasing n.
 * Launch : grid = (ceil(dim / kCh), B), block = 256, kCh = 16 (8 for dstate 32).
 */
template <typename input_t, int kNState>
__global__ void selective_scan_fwd_kernel(const __grid_constant__ SelectiveScanParams params);

/// Decode / short-sequence state update (Mamba-1 and Mamba-2), S = num_steps tokens per sequence.
/// Rows are the nheads * headdim state rows [dstate]; Mamba-1 is nheads = dim, headdim = 1 with
/// a_per_state = true (A [dim, dstate]).
struct SSMStateUpdateParams {
  float* state;             // [num_slots, nheads, headdim, dstate]
  int64_t state_slot_stride, state_head_stride, state_p_stride;
  const void* x;            // [B, S, nheads, headdim]
  int64_t x_batch_stride, x_token_stride, x_head_stride;
  const void* dt;           // [B, S, nheads]
  int64_t dt_batch_stride, dt_token_stride, dt_head_stride;
  const float* dt_bias;     // [nheads] or null
  const float* A;           // [nheads] (Mamba-2) or [nheads, dstate] (a_per_state)
  int64_t A_head_stride;
  bool a_per_state;
  const void* B;            // [B, S, ngroups, dstate]
  int64_t B_batch_stride, B_token_stride, B_group_stride;
  const void* C;
  int64_t C_batch_stride, C_token_stride, C_group_stride;
  const float* D;           // [nheads] (D_p_stride 0) or [nheads, headdim], or null
  int64_t D_head_stride, D_p_stride;
  const void* z;            // like x, or null
  int64_t z_batch_stride, z_token_stride, z_head_stride;
  void* out;                // like x; x itself or disjoint
  int64_t out_batch_stride, out_token_stride, out_head_stride;
  const int32_t* state_batch_indices;       // [B] slot per sequence, null = b
  const int32_t* intermediate_state_slots;  // [B, S]: slot receiving the state after token i, or null
  bool dt_softplus;
  bool has_dt_limit;
  float dt_min, dt_max;     // clamp after softplus (Mamba-2 dt_limit) when has_dt_limit
  int32_t num_slots, num_steps, batch_size, nheads, headdim, dstate, ngroups;
  int32_t pad_slot_id;
};

/// selective_state_update_kernel — per token and row (h, p), dt' = clamp(softplus(dt + dt_bias)):
///   state[n] = state[n] * exp(dt' A) + dt' B[g(h), n] x[p];  y = sum_n state[n] C[g(h), n] + D x[p];
///   y *= silu(z). kLanesN lanes hold kNPerLane state values each (dstate = kLanesN * kNPerLane).
/// Launch : grid = (ceil(nheads * headdim / 32), B), block = 128.
template <typename input_t, int kLanesN, int kNPerLane>
__global__ void selective_state_update_kernel(const __grid_constant__ SSMStateUpdateParams params);

/// Mamba-2 SSD (state space duality) chunked prefill: shared arguments of the 5-kernel pipeline.
/// x, z, out are [T, H, P], B and C [T, G, N], dt [T, H] (unit stride on P and N); A, dt_bias fp32
/// [H]; D [H] (D_p_stride 0) or [H, P]. Chunk metadata is device data resolved by the caller:
/// cu_chunks [B + 1] (chunks [cu_chunks[b], cu_chunks[b + 1]) belong to sequence b) and
/// cu_chunk_seqlens [num_chunks + 1] (chunk c = tokens [cu_chunk_seqlens[c], cu_chunk_seqlens[c+1]),
/// at most chunk_size of them, every chunk of a sequence full except the last).
struct SSDParams {
  const void* x;
  int64_t x_token_stride, x_head_stride;
  const void* dt;
  int64_t dt_token_stride, dt_head_stride;
  const float* dt_bias;     // [H] or null
  const float* A;           // [H]
  const void* B;
  int64_t B_token_stride, B_group_stride;
  const void* C;
  int64_t C_token_stride, C_group_stride;
  const float* D;           // or null
  int64_t D_head_stride, D_p_stride;
  const void* z;            // or null
  int64_t z_token_stride, z_head_stride;
  void* out;
  int64_t out_token_stride, out_head_stride;
  bool dt_softplus, has_dt_limit;
  float dt_min, dt_max;
  const int32_t* cu_seqlens;        // [B + 1]
  const int32_t* cu_chunks;         // [B + 1]
  const int32_t* cu_chunk_seqlens;  // [num_chunks + 1]
  float* ssm_states;                // [num_slots, H, P, N], contiguous within a slot
  int64_t ssm_state_slot_stride;
  const int32_t* state_indices;     // [B] slot per sequence, null = b
  const uint8_t* has_initial_state; // [B], null = every sequence starts from its slot
  const int32_t* snapshot_slots;    // [B, max_snapshots] or null
  int32_t snapshot_interval, max_snapshots;  // interval: positive multiple of chunk_size
  // fp32 workspace (carved from Workspace::device_scratch by ssd_chunked_scan)
  float* dt_proc;           // [H, num_chunks, chunk_size]
  float* dA_cumsum;         // [H, num_chunks, chunk_size]
  float* CB;                // [num_chunks, G, chunk_size, chunk_size]
  float* chunk_states;      // [num_chunks, H, P, N]
  float* passed_states;     // [num_chunks, H, P, N]
  int32_t num_slots, batch_size, num_chunks, chunk_size, nheads, headdim, dstate, ngroups;
  int32_t pad_slot_id;
};

/// 1) dt' = clamp(softplus(dt + bias)); dA_cumsum = inclusive cumsum(dt' * A) within chunks.
/// Launch : grid = num_chunks * H, block = 256.
template <typename input_t>
__global__ void ssd_chunk_cumsum_kernel(const __grid_constant__ SSDParams params);
/// 2) CB[c, g, l, s] = C_l . B_s within each chunk (s <= l). Grid = chunks * G * 64-tile pairs.
template <typename input_t>
__global__ void ssd_bmm_chunk_kernel(const __grid_constant__ SSDParams params);
/// 3) chunk_states[c] = sum_t exp(dA_cs[last] - dA_cs[t]) dt'_t x_t^T B_t.
template <typename input_t>
__global__ void ssd_chunk_state_kernel(const __grid_constant__ SSDParams params);
/// 4) inter-chunk recurrence from the slot state -> passed states, final state, snapshots.
__global__ void ssd_state_passing_kernel(const __grid_constant__ SSDParams params);
/// 5) y = intra-chunk (CB masked by decay) x + C . passed_state (decayed) + D x, gated by silu(z).
template <typename input_t>
__global__ void ssd_chunk_scan_kernel(const __grid_constant__ SSDParams params);

}  // namespace kops::state
