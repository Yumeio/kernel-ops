#pragma once
// =============================================================================
// kernel-ops | csrc/state/delta_rule_kernels.cuh                 Catalog: L02 (delta rule)
// Gated delta rule linear attention:
//   GDN : Gated DeltaNet — Qwen3-Next, Qwen3.5 (scalar decay per value head, GVA: Hv >= Hk)
//   KDA : Kimi Delta Attention — Kimi Linear, Kimi K3 (channel-wise decay per key dim)
// Recurrence per value head h (S is Dk x Dv, fp32; key head hk = h / (Hv / Hk)):
//   S_t = Diag(exp(g_t)) S_{t-1}                       (GDN: exp(g_t) scalar; KDA: vector over Dk)
//   S_t = S_t + k_t (beta_t (v_t - S_t^T k_t))^T       (delta rule correction)
//   o_t = S_t^T (scale * q_t)                          (q, k optionally L2-normalized first)
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

/// Arithmetic of the chunked prefill (steps kkt, w/u, state and output). Workspace layout,
/// inputs and output dtypes are the same in both modes.
enum class ChunkMath : uint8_t {
  kFp32 = 0,        // fp32 FMA in a fixed order (~1e-6 relative to the fp64 oracle for fp32 inputs)
  kTensorCore = 1,  // mma.sync m16n8k16 on bf16 operands, fp32 accumulation, fp32 state/workspace (SM80+, bf16 inputs)
};

/// Shared arguments of the delta-rule kernels. q, k are [T, Hk, Dk]; v and o [T, Hv, Dv] with Hv a
/// multiple of Hk; the last axis of each is contiguous, token and head strides count elements.
/// g (log-decay <= 0) and beta are contiguous fp32. The state pool is fp32 [num_slots, Hv, Dk, Dv]
/// (or [.., Dv, Dk] with state_transposed), contiguous within a slot.
struct DeltaRuleParams {
  const void* q;
  const void* k;
  const void* v;
  int64_t q_token_stride, q_head_stride, k_token_stride, k_head_stride, v_token_stride, v_head_stride;
  const float* g;                // [T, Hv] or [T, Hv, Dk]
  const float* beta;             // [T, Hv]
  float scale;                   // applied to q after its normalization (Dk^-0.5 typically)
  bool use_qk_l2norm;            // x / sqrt(sum x^2 + 1e-6) on q and k
  const int32_t* cu_seqlens;     // [B + 1]
  const int32_t* state_indices;  // [B] slot per sequence, null = b
  const uint8_t* has_initial_state;  // [B], null = every sequence starts from its slot
  float* state_pool;
  int64_t state_slot_stride;
  bool state_transposed;
  // Prefix caching: the state after every snapshot_interval tokens goes to snapshot_slots[b, j]
  // (chunked prefill: interval a positive multiple of 64). Invalid destinations are dropped.
  const int32_t* snapshot_slots;
  int32_t snapshot_interval, max_snapshots;
  // Speculative decoding (recurrent kernel): state after token i -> intermediate_state_slots[b, i].
  const int32_t* intermediate_state_slots;  // [B, max_intermediate] or null
  int32_t max_intermediate;
  void* o;
  int64_t o_token_stride, o_head_stride;
  int32_t num_slots, total_tokens, batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v;
  DecayGranularity decay;
  int32_t pad_slot_id;
  // Chunked prefill: caller-owned fp32 workspace (carved by delta_rule_prefill from the Workspace)
  float* gc;                     // [T, Hv] or [T, Hv, Dk]: cumulative log-decay inside the chunk
  float* A;                      // [T, Hv, 64]: strictly lower beta (k decayed k^T), then (I + A)^-1
  float* w;                      // [T, Hv, Dk]
  float* u;                      // [T, Hv, Dv]
  float* h;                      // [rows, Hv, Dk, Dv]: state entering each chunk, row = (first token >> 6) + b
  float* v_new;                  // [T, Hv, Dv]
  int32_t max_seqlen;            // bounds every sequence (0 = total_tokens); only the grids depend on it
  ChunkMath math;
};

/// Arguments of delta_rule_gating_kernel.
struct DeltaGatingParams {
  float* g;                      // [T, Hv] or [T, Hv, Dk], contiguous
  float* beta;                   // [T, Hv], contiguous
  const void* a;                 // decay pre-activation [T, Hv] or [T, Hv, Dk] (unit last stride)
  int64_t a_token_stride, a_head_stride;
  const void* b;                 // beta pre-activation [T, Hv] (unit head stride)
  int64_t b_token_stride;
  const float* A_log;            // [Hv]
  const float* dt_bias;          // [Hv] or [Hv, Dk]
  float lower_bound;             // < 0: g = lower_bound * sigmoid(exp(A_log) (a + dt_bias)); 0: softplus form
  DecayGranularity decay;
  int32_t total_tokens, num_v_heads, head_dim_k;
};

/**
 * delta_rule_gating_kernel — fused gate preparation:
 *   GDN : g = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]);           beta = sigmoid(b[t,h])
 *   KDA : g = -exp(A_log[h]) * softplus(a[t,h,k] + dt_bias[h,k]) (per channel); beta = sigmoid(b)
 * lower_bound < 0 selects the bounded FLA/GDN variant (Kimi Linear uses softplus, lower_bound = 0).
 * Launch : one thread per element of g, block = 256.
 */
template <typename input_t, DecayGranularity kDecay>
__global__ void delta_rule_gating_kernel(const __grid_constant__ DeltaGatingParams params);

/// 1) chunk-local inclusive cumsum of g (64-token chunks from each sequence start). Grid (chunks, Hv, B).
template <DecayGranularity kDecay>
__global__ void chunk_local_cumsum_kernel(const __grid_constant__ DeltaRuleParams params);
/// 2) A = strictly-lower( beta_i * (k_i * exp(G_i - G_j)) . k_j ) per chunk.
template <typename input_t, bool kPerChannel>
__global__ void chunk_scaled_dot_kkt_kernel(const __grid_constant__ DeltaRuleParams params);
/// 3) A <- (I + A)^{-1} for 64x64 unit-lower-triangular blocks (16x16 block forward substitution, fp32).
__global__ void solve_tril_kernel(const __grid_constant__ DeltaRuleParams params);
/// 4) w = A (beta k exp(G)), u = A (beta v).
template <typename input_t, bool kPerChannel>
__global__ void recompute_w_u_kernel(const __grid_constant__ DeltaRuleParams params);
/// 5) chunk recurrence: h_{c+1} = decay(c) h_c + K_c^T (U_c - W_c h_c); writes v_new, the final state
///    and prefix-cache snapshots; handles initial states and sequence boundaries.
template <typename input_t, bool kPerChannel, int kDvTile>
__global__ void chunk_state_fwd_kernel(const __grid_constant__ DeltaRuleParams params);
/// 6) o = scale * ( (Q exp(G)) h_c + causal(Q K^T decay) v_new ).
template <typename input_t, bool kPerChannel, int kDvTile>
__global__ void chunk_output_fwd_kernel(const __grid_constant__ DeltaRuleParams params);

/**
 * fused_recurrent_delta_rule_kernel — token-by-token recurrence (decode, spec-decode verification,
 * very short prefill chunks). A warp owns 32 / kSplit value columns of one (sequence, v head) with
 * the S column in registers (kSplit = 2 for Dk = 128). Launch : grid = (ceil(Dv / cols), Hv, B), block = 128.
 */
template <typename input_t, DecayGranularity kDecay, int kHeadDimK, bool kTransposed>
__global__ void fused_recurrent_delta_rule_kernel(const __grid_constant__ DeltaRuleParams params);

/// Arguments of the fully fused decode layer core (one token per sequence). `rule` supplies scale,
/// use_qk_l2norm, the state pool and slot mapping, head counts/dims, decay and pad_slot_id; its
/// q/k/v/g/beta/o pointers are unused. The conv runs over [q (Hk Dk) | k (Hk Dk) | v (Hv Dv)].
struct FusedDeltaDecodeParams {
  DeltaRuleParams rule;
  const void* qkv;                  // [B, 2 Hk Dk + Hv Dv] projection output before the short conv
  int64_t qkv_batch_stride;
  const void* conv_weight;          // [conv_dim, width]
  int64_t conv_weight_dim_stride, conv_weight_width_stride;
  const void* conv_bias;            // [conv_dim] or null
  void* conv_states;                // [num_slots, conv_dim, width - 1], same slot as rule.state_pool
  int64_t conv_state_slot_stride, conv_state_dim_stride, conv_state_len_stride;
  int32_t conv_width;
  const void* gate_a;               // [B, Hv] (GDN) or [B, Hv, Dk] (KDA): decay pre-activation
  int64_t gate_a_batch_stride;
  const void* gate_b;               // [B, Hv]: beta pre-activation
  int64_t gate_b_batch_stride;
  const float* A_log;               // [Hv]
  const float* dt_bias;             // [Hv] or [Hv, Dk]
  float lower_bound;                // see DeltaGatingParams
  const void* out_gate;             // [B, Hv * Dv] output gate pre-activation
  int64_t out_gate_batch_stride;
  ActFn out_gate_act;               // kSigmoid (KDA) | kSiLU (GDN)
  const float* out_norm_weight;     // [Dv] gated RMSNorm weight
  float out_norm_eps;               // Kimi Linear: 1e-5
  void* out;                        // [B, Hv * Dv] gated + normalized output (input of o_proj)
  int64_t out_batch_stride;
};

/**
 * fused_short_conv_delta_rule_decode_kernel — one launch per layer core for decode: short conv
 * (SiLU, state update) -> gating -> L2 norm -> recurrent update -> RMSNorm(o) * w * act(out_gate).
 * q/k/v, f_a/f_b, b_proj, g_a/g_b and o_proj projections stay outside (GEMV). One CTA owns key head
 * hk with its Hv / Hk value heads. Launch : grid = (Hk, B), block <= 256.
 */
template <typename input_t, DecayGranularity kDecay, int kHeadDimK, int kConvWidth>
__global__ void fused_short_conv_delta_rule_decode_kernel(
    const __grid_constant__ FusedDeltaDecodeParams params);

}  // namespace kops::state
