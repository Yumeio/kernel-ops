#pragma once
// =============================================================================
// kernel-ops | csrc/state/delta_rule_kernels.cuh                 Catalog: L02 (delta rule)
// Gated delta rule linear attention:
//   GDN : Gated DeltaNet — Qwen3-Next, Qwen3.5 (scalar decay per value head, GVA: Hv >= Hk)
//   KDA : Kimi Delta Attention — Kimi Linear, Kimi K3 (channel-wise decay per key dim)
// Recurrence per head (S is Dk x Dv, fp32):
//   S_t = Diag(exp(g_t)) S_{t-1}                       (GDN: exp(g_t) scalar; KDA: vector over Dk)
//   S_t = S_t + beta_t k_t (v_t - S_t^T k_t)^T         (delta rule correction)
//   o_t = scale * S_t^T q_t
// Prefill uses the chunked WY formulation (FLA): cumsum -> KK^T -> solve_tril -> w,u ->
// chunk states -> outputs. Decode uses the fused recurrent kernel. Both must agree
// (serial vs chunked parity test, fp32 state).
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh.
// =============================================================================

namespace kops::state {

enum class DecayGranularity : uint8_t {
  kPerHead = 0,     // g: [T, Hv]       (GDN)
  kPerChannel = 1,  // g: [T, Hv, Dk]   (KDA)
};

/// Shared arguments of the delta-rule kernels.
struct DeltaRuleParams {
  const void* q;                 // [T, Hk, Dk] (raw if use_qk_l2norm, else already normalized)
  const void* k;                 // [T, Hk, Dk]
  const void* v;                 // [T, Hv, Dv]
  int64_t q_token_stride, k_token_stride, v_token_stride;
  const float* g;                // log-decay (<= 0): [T, Hv] or [T, Hv, Dk]
  const float* beta;             // [T, Hv] in (0, 1) (or (0, 2) when allow_negative_eigval)
  float scale;                   // Dk^-0.5 typically
  bool use_qk_l2norm;            // fuse L2 normalization of q and k
  // Sequences and state slots
  const int32_t* cu_seqlens;     // [B + 1]
  const int32_t* state_indices;  // [B] slot of each sequence in state_pool
  const uint8_t* has_initial_state;  // [B]
  float* state_pool;             // [num_slots, Hv, Dk, Dv] (or [.., Dv, Dk] if state_transposed)
  bool state_transposed;
  // Prefix caching (fine-grained KDA/GDN state snapshots inside larger physical state blocks):
  // write S after every snapshot_interval tokens to snapshot_slots[b * max_snapshots + j].
  const int32_t* snapshot_slots;
  int32_t snapshot_interval, max_snapshots;
  // Speculative decoding: keep S after every draft token to roll back to num_accepted.
  const int32_t* intermediate_state_slots;  // [B, max_draft + 1] or null
  // Chunked-prefill workspaces (fp32 unless noted)
  float* g_cumsum;               // like g
  float* A;                      // [T, Hv, chunk] beta * K K^T (decay-weighted), then (I + A)^-1
  void* w;                       // [T, Hv, Dk]
  void* u;                       // [T, Hv, Dv]
  float* h;                      // [num_chunks, Hv, Dk, Dv] chunk-start states
  const int32_t* chunk_indices;  // [num_chunks, 2] (sequence, chunk within sequence)
  int32_t num_chunks, chunk_size;  // chunk_size = 64
  // Output
  void* o;                       // [T, Hv, Dv]
  int64_t o_token_stride;
  int32_t total_tokens, batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v;
  DecayGranularity decay;
  int32_t pad_slot_id;
};

/**
 * delta_rule_gating_kernel — fused gate preparation:
 *   GDN : g = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]);           beta = sigmoid(b[t,h])
 *   KDA : g = -exp(A_log[h]) * softplus(f[t,h,k] + dt_bias[h,k]) (per channel); beta = sigmoid(b)
 * Optional lower bound on g for numerical stability (lower_bound < 0, 0 = off).
 */
template <typename input_t, DecayGranularity kDecay>
__global__ void delta_rule_gating_kernel(
    float* __restrict__ g,                     // [T, Hv] or [T, Hv, Dk]
    float* __restrict__ beta,                  // [T, Hv]
    const input_t* __restrict__ a,             // [T, Hv] or [T, Hv, Dk] decay pre-activation
    const input_t* __restrict__ b,             // [T, Hv] beta pre-activation
    const float* __restrict__ A_log,           // [Hv]
    const float* __restrict__ dt_bias,         // [Hv] or [Hv, Dk]
    const float lower_bound,
    const int total_tokens, const int num_v_heads, const int head_dim_k);

/// 1) chunk-local cumulative sum of g (reset at chunk and sequence boundaries).
template <DecayGranularity kDecay, int CHUNK>
__global__ void chunk_local_cumsum_kernel(const KOPS_GRID_CONST DeltaRuleParams params);
/// 2) A = strictly-lower( beta_i * (k_i * exp(G_i - G_j)) . k_j ) per chunk.
template <typename input_t, DecayGranularity kDecay, int CHUNK, int BLOCK_K>
__global__ void chunk_scaled_dot_kkt_kernel(const KOPS_GRID_CONST DeltaRuleParams params);
/// 3) A <- (I + A)^{-1} for 64x64 unit-lower-triangular blocks (16x16 hierarchical, fp32).
template <int CHUNK>
__global__ void solve_tril_kernel(const KOPS_GRID_CONST DeltaRuleParams params);
/// 4) w = A (beta k exp(G)), u = A (beta v).
template <typename input_t, DecayGranularity kDecay, int CHUNK, int BLOCK_K, int BLOCK_V>
__global__ void recompute_w_u_kernel(const KOPS_GRID_CONST DeltaRuleParams params);
/// 5) chunk recurrence: h_{c+1} = decay(c) h_c + K_c^T (U_c - W_c h_c); writes final state and
///    prefix-cache snapshots; handles initial states and sequence boundaries.
template <typename input_t, DecayGranularity kDecay, int CHUNK, int BLOCK_V>
__global__ void chunk_state_fwd_kernel(const KOPS_GRID_CONST DeltaRuleParams params);
/// 6) o = scale * ( (Q exp(G)) h_c + causal(Q K^T decay) (U - W h_c) ).
template <typename input_t, DecayGranularity kDecay, int CHUNK, int BLOCK_K, int BLOCK_V>
__global__ void chunk_output_fwd_kernel(const KOPS_GRID_CONST DeltaRuleParams params);

/**
 * fused_recurrent_delta_rule_kernel — token-by-token recurrence (decode, spec-decode verification,
 * very short prefill chunks). One CTA per (sequence, v head, Dv tile); S tile in registers.
 */
template <typename input_t, DecayGranularity kDecay, int HEAD_DIM_K, int BLOCK_V, bool kUseL2Norm>
__global__ void fused_recurrent_delta_rule_kernel(const KOPS_GRID_CONST DeltaRuleParams params);

/// Arguments of the fully fused decode layer core (vLLM/NVIDIA fused KDA decode).
struct FusedDeltaDecodeParams {
  DeltaRuleParams rule;             // q/k/v pointers refer to the conv OUTPUT scratch when conv is fused
  const void* qkv_pre_conv;         // [B, (2*Hk*Dk + Hv*Dv)] projection output before short conv
  const void* conv_weight;          // [conv_dim, width]
  const void* conv_bias;            // [conv_dim] or null
  void* conv_states;                // [num_slots, conv_dim, width - 1]
  int32_t conv_width;
  const void* gate_a;               // decay pre-activation (see delta_rule_gating_kernel)
  const void* gate_b;               // beta pre-activation
  const float* A_log;
  const float* dt_bias;
  const void* out_gate;             // [B, Hv * Dv] output gate pre-activation (KDA sigmoid / GDN silu)
  ActFn out_gate_act;
  const void* out_norm_weight;      // [Dv] gated RMSNorm weight
  float out_norm_eps;
  void* out;                        // [B, Hv * Dv] gated + normalized output (input of o_proj)
};

/**
 * fused_short_conv_delta_rule_decode_kernel — one launch per layer for decode: short conv ->
 * gating -> L2 norm -> recurrent update -> output gate -> RMSNorm. Removes ~6 launches per
 * KDA/GDN layer (important for Kimi K3: three of every four layers are KDA).
 */
template <typename input_t, DecayGranularity kDecay, int HEAD_DIM_K, int HEAD_DIM_V, int CONV_WIDTH>
__global__ void fused_short_conv_delta_rule_decode_kernel(
    const KOPS_GRID_CONST FusedDeltaDecodeParams params);

}  // namespace kops::state
