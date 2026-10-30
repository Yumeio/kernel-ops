#pragma once
// =============================================================================
// kernel-ops | csrc/state/ssm_kernels.cuh                         Catalog: L02 (SSM)
// Selective state-space models.
//   Mamba-1 : Falcon-Mamba, Jamba / Jamba-1.5 (Mamba layers)        selective scan, per-channel A [D, N]
//   Mamba-2 : Codestral-Mamba, Nemotron-H, Falcon-H1, Granite-4.0-H, Zamba2, Bamba
//             SSD chunked scan, scalar A per head, state [nheads, headdim, dstate]
// States are fp32 in caller-owned slots; slot mapping per sequence.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::state {

/// Mamba-1 selective scan (prefill, varlen).
struct SelectiveScanParams {
  const void* u;            // [dim, total_tokens] input after conv
  const void* delta;        // [dim, total_tokens]
  const float* A;           // [dim, dstate] (negative, = -exp(A_log))
  const void* B;            // [ngroups, dstate, total_tokens]
  const void* C;            // [ngroups, dstate, total_tokens]
  const float* D;           // [dim] skip or null
  const void* z;            // [dim, total_tokens] gate or null
  const float* delta_bias;  // [dim] or null
  bool delta_softplus;
  void* out;                // [dim, total_tokens]
  float* ssm_states;        // [num_slots, dim, dstate] in/out
  const int32_t* query_start_loc;    // [B + 1]
  const int32_t* cache_indices;      // [B]
  const uint8_t* has_initial_state;  // [B]
  int32_t batch_size, dim, dstate, ngroups, pad_slot_id;
  int64_t u_dim_stride, delta_dim_stride, B_group_stride, C_group_stride, out_dim_stride;
};

/**
 * selective_scan_fwd_kernel — h_t = exp(delta_t * A) h_{t-1} + delta_t * B_t u_t;
 * y_t = C_t h_t + D u_t; y *= silu(z). Chunked over time with fp32 state in registers.
 */
template <typename input_t, typename weight_t, int kNThreads, int kNItems, int kNRows, bool kIsVarlen>
__global__ void selective_scan_fwd_kernel(const KOPS_GRID_CONST SelectiveScanParams params);

/// Decode / short-sequence state update (Mamba-1 and Mamba-2).
struct SSMStateUpdateParams {
  float* state;             // Mamba-1: [num_slots, dim, dstate]; Mamba-2: [num_slots, nheads, headdim, dstate]
  const void* x;            // [B, nheads, headdim] (Mamba-2) or [B, dim]
  const void* dt;           // [B, nheads] (Mamba-2, tied per head) or [B, dim]
  const float* dt_bias;     // [nheads] / [dim] or null
  const float* A;           // [nheads] scalar per head (Mamba-2) or [dim, dstate] (Mamba-1)
  const void* B;            // [B, ngroups, dstate]
  const void* C;            // [B, ngroups, dstate]
  const float* D;           // [nheads] / [dim] or null
  const void* z;            // gate or null
  void* out;                // like x
  const int32_t* state_batch_indices;  // [B] slot per sequence
  bool dt_softplus;
  float dt_min, dt_max;     // clamp after softplus (Mamba-2 dt_limit), 0/0 = off
  // speculative decoding: per-step states written to intermediate slots
  const int32_t* intermediate_state_slots;  // [B, num_steps] or null
  int32_t num_steps;        // tokens per sequence in this call (1 = plain decode)
  int32_t batch_size, nheads, headdim, dstate, ngroups, pad_slot_id;
  bool is_mamba2;
};

template <typename input_t, typename weight_t, int DSTATE, int HEADDIM_TILE>
__global__ void selective_state_update_kernel(const KOPS_GRID_CONST SSMStateUpdateParams params);

/// Mamba-2 SSD (state space duality) chunked prefill: shared arguments of the 5-kernel pipeline.
struct SSDParams {
  const void* x;            // [total_tokens, nheads, headdim]
  const void* dt;           // [total_tokens, nheads]
  const float* dt_bias;     // [nheads]
  const float* A;           // [nheads]
  const void* B;            // [total_tokens, ngroups, dstate]
  const void* C;            // [total_tokens, ngroups, dstate]
  const float* D;           // [nheads] or [nheads, headdim]
  const void* z;            // [total_tokens, nheads, headdim] or null
  float dt_min, dt_max;
  bool dt_softplus;
  int32_t chunk_size;       // 256 typical
  const int32_t* cu_seqlens;      // [B + 1]
  const int32_t* seq_idx;         // [total_tokens] sequence id (chunk boundaries never mix sequences)
  const int32_t* cu_chunk_seqlens;// [num_chunks + 1] (chunks aligned to sequence starts)
  const int32_t* last_chunk_indices; // [B]
  float* initial_states;    // gathered [B, nheads, headdim, dstate] or null
  const uint8_t* has_initial_state;
  // workspaces / intermediates (fp32)
  float* dt_processed;      // [nheads, num_chunks, chunk_size]
  float* dA_cumsum;         // [nheads, num_chunks, chunk_size]
  float* CB;                // [num_chunks, ngroups, chunk_size, chunk_size]
  float* chunk_states;      // [num_chunks, nheads, headdim, dstate]
  float* passed_states;     // [num_chunks, nheads, headdim, dstate]
  float* final_states;      // [B, nheads, headdim, dstate] (scattered into slots by state_cache kernels)
  void* out;                // [total_tokens, nheads, headdim]
  int32_t total_tokens, num_chunks, nheads, headdim, dstate, ngroups;
};

/// 1) dt = clamp(softplus(dt + bias)); dA_cumsum = cumsum(dt * A) within chunks.
template <typename input_t, int BLOCK_H, int CHUNK>
__global__ void ssd_chunk_cumsum_kernel(const KOPS_GRID_CONST SSDParams params);
/// 2) CB[c] = C_c B_c^T within each chunk (per group).
template <typename input_t, int CHUNK, int BLOCK_N>
__global__ void ssd_bmm_chunk_kernel(const KOPS_GRID_CONST SSDParams params);
/// 3) chunk_states[c] = sum_t exp(dA_cs[last] - dA_cs[t]) dt_t B_t x_t^T.
template <typename input_t, int CHUNK, int BLOCK_P, int BLOCK_N>
__global__ void ssd_chunk_state_kernel(const KOPS_GRID_CONST SSDParams params);
/// 4) inter-chunk recurrence with initial states, reset at sequence starts -> passed / final states.
template <int BLOCK_P, int BLOCK_N>
__global__ void ssd_state_passing_kernel(const KOPS_GRID_CONST SSDParams params);
/// 5) y = intra-chunk (CB masked by decay) x + C . passed_state (decayed) + D x, gated by silu(z).
template <typename input_t, int CHUNK, int BLOCK_P>
__global__ void ssd_chunk_scan_kernel(const KOPS_GRID_CONST SSDParams params);

}  // namespace kops::state
