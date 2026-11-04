#pragma once
// =============================================================================
// kernel-ops | csrc/reference/ops.h
// Host-only fp64 reference semantics used by tests/reference/. This family is
// NEVER linked into the release library: it exists so every kernel has exactly
// one executable definition of "correct" that is independent of GPU code.
//
// Contract:
//   * All math is double precision, scalar loops, fixed evaluation order.
//     Do not vectorize, do not reorder, do not fuse: a reference that changes
//     between compilers is not a reference.
//   * Every function takes plain host pointers and explicit row strides in
//     ELEMENTS. No Tensor, no descriptors, no backend state, no workspace.
//   * Validation is limited to what a test would get wrong: null pointers,
//     non-positive sizes, unsupported enum values -> Status. No alignment or
//     device checks (there is no device).
//   * Compile the .cc with -ffp-contract=off and without -ffast-math so FMA
//     contraction cannot change the golden values (see docs/CONTRACTS.md).
//
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh, common/quant.cuh,
//           common/op_types.cuh.
// =============================================================================

namespace kops::ref {

// ---- Normalization -----------------------------------------------------------
/// out[r,:] = in[r,:] * rsqrt(mean(in[r,:]^2) + eps) * (w + weight_offset)
Status rms_norm_f64(const double* in, int64_t in_row_stride, const double* weight,
                    double weight_offset, double eps, int64_t rows, int64_t hidden,
                    double* out, int64_t out_row_stride);
/// out = (x - mean) * rsqrt(var + eps) * (w + weight_offset) + b   (biased variance)
Status layer_norm_f64(const double* in, int64_t in_row_stride, const double* weight,
                      const double* bias, double weight_offset, double eps, int64_t rows,
                      int64_t hidden, double* out, int64_t out_row_stride);
/// Grouped RMSNorm with a gate branch; mirrors gated_rms_norm_kernel's two orders.
Status gated_rms_norm_f64(const double* x, const double* z, const double* weight,
                          int64_t group_size, ActFn gate_act, bool norm_before_gate,
                          double weight_offset, double eps, int64_t rows, int64_t hidden,
                          double* out);

// ---- Activations -------------------------------------------------------------
/// Elementwise apply_act (GluParams constants included).
Status act_f64(const double* in, int64_t n, ActFn act, const GluParams& p, double* out);
/// out = f(gate) * g(up) with GateLayout / GluParams semantics (row_weight optional).
Status act_and_mul_f64(const double* in, int64_t in_row_stride, const double* row_weight,
                       GluParams p, ActFn act, GateLayout layout, int64_t rows, int64_t d,
                       double* out);

// ---- Position -----------------------------------------------------------------
/// NeoX/GPTJ rotary over dims [rot_offset, rot_offset+rot_dim) using a cos/sin table
/// laid out as cache[pos, 0:rot/2]=cos, cache[pos, rot/2:rot]=sin.
Status rope_f64(const int64_t* positions, const double* cos_sin_cache, int64_t cache_stride,
                int64_t rot_dim, int64_t rot_offset, RopeStyle style, int64_t rows,
                int64_t heads, int64_t head_dim, double* q);
/// mRoPE: positions[3, rows] with contiguous/interleaved sections.
Status mrope_f64(const int64_t* positions_thw, const double* cos_sin_cache, int64_t cache_stride,
                 int32_t section_t, int32_t section_h, int32_t section_w, int64_t rot_dim,
                 bool interleaved, RopeStyle style, int64_t rows, int64_t heads,
                 int64_t head_dim, double* q);

// ---- GEMM ---------------------------------------------------------------------
/// D[M,N] = epilogue(A[M,K] . B[N,K]^T) with the subset of EpilogueOp that the
/// golden tests need: alpha, row/col scale, bias, clamp, residual add, fp32 out.
/// split_k is emulated exactly (fp32 partials in the declared order), so a kernel
/// claiming deterministic split-K can be compared bit-for-bit.
Status gemm_f64(const double* a, int64_t lda, const double* b, int64_t ldb, double alpha,
                const double* row_scale, const double* col_scale, const double* bias,
                const double* residual, int64_t m, int64_t n, int64_t k, int32_t split_k,
                double* d, int64_t ldd);
/// Strided batched GEMM for MLA absorption: D[b] = A[b] . B[b]^T.
Status bmm_f64(const double* a, int64_t lda, int64_t a_batch_stride, const double* b, int64_t ldb,
               int64_t b_batch_stride, int64_t batch, int64_t m, int64_t n, int64_t k,
               double* d, int64_t ldd, int64_t d_batch_stride);
/// Router / LM-head logits kept in fp64 (G02, G01 gemv_argmax).
Status gemv_f64(const double* a, int64_t lda, const double* b, int64_t ldb, int64_t m, int64_t n,
                int64_t k, double* d);

// ---- Quantization -------------------------------------------------------------
/// Quantize `rows x cols` with the scheme's mode; writes payload as fp64 dequant-
/// equivalent plus the scale/zero tensors the caller provides. Used to prove that a
/// kernel's rounding point matches the documented one.
Status quantize_f64(const double* in, int64_t in_row_stride, const QuantScheme& scheme,
                    int64_t rows, int64_t cols, double* q_out, double* scales,
                    double* zero_points);
/// Inverse of quantize_f64 (same scheme), so `dequant(quant(x))` can be compared.
Status dequantize_f64(const double* q, const double* scales, const double* zero_points,
                      const QuantScheme& scheme, int64_t rows, int64_t cols, double* out);

// ---- Attention ----------------------------------------------------------------
/// Arguments of attention_f64. Packed layouts: q [total_q, Hq, Dqk], k [total_kv, Hkv, Dqk],
/// v [total_kv, Hkv, Dvo], o [total_q, Hq, Dvo], lse [total_q, Hq]. Optional pointers may be null.
struct AttentionRefArgs {
  const double* q = nullptr;
  const double* k = nullptr;               // already dequantized (FP8 KV: k8 * k_scale)
  const double* v = nullptr;
  const int32_t* qo_indptr = nullptr;      // [B + 1]
  const int32_t* kv_indptr = nullptr;      // [B + 1]
  const int32_t* q_pos_offset = nullptr;   // [B] position of query row 0; null = kv_len - qo_len
  int32_t batch = 0, q_heads = 0, kv_heads = 0, head_dim_qk = 0, head_dim_vo = 0;
  double sm_scale = 0.0;
  double logits_softcap = 0.0;             // > 0: s = cap * tanh(s / cap)
  int32_t window_left = -1, window_right = -1, chunk_size = 0, num_sink_tokens = 0;
  const double* sink_logits = nullptr;     // [Hq], natural-log units of s
  const double* alibi_slopes = nullptr;    // [Hq]; s += slope * (k_pos - q_pos)
  bool causal = true;
  const uint8_t* custom_mask = nullptr;    // packed LSB-first bits over [qo_len, kv_len] per request
  const int64_t* mask_indptr = nullptr;    // [B + 1] bit offsets
  double* o = nullptr;
  double* lse = nullptr;                   // may be null
};
/// Dense attention with every AttentionModifiers rule; the golden reference for
/// A01/A02/A03/A06/A07/A08 (CONTRACTS §1e). Order per score: s = sm_scale * q.k, softcap,
/// + ALiBi, mask; softmax denominator += exp(sink) once. Fully masked row without sink:
/// o = 0, lse = -inf.
Status attention_f64(const AttentionRefArgs& args);
/// Merge (o, lse) states; mirrors merge_states_kernel's deterministic order.
Status merge_states_f64(const double* o_states, const double* lse_states, int64_t rows, int64_t n,
                        int64_t heads, int64_t head_dim, double* o, double* lse);

// ---- KV cache -----------------------------------------------------------------
/// Reference page write: slot = page * page_size + offset, slot < 0 skipped.
/// `entry_bytes`-style packed codecs are NOT modelled here; tests use this for
/// native dtypes and compare quantized codecs after dequantize_f64.
Status reshape_and_cache_f64(const double* key, const double* value, int64_t tokens, int64_t heads,
                             int64_t head_dim, int32_t page_size, int32_t num_pages,
                             const int64_t* slot_mapping, double* k_cache, double* v_cache);

// ---- MoE ----------------------------------------------------------------------
/// Routing (softmax/sigmoid/grouped) + top-k with the documented tie-break
/// (lower expert index wins). `scores` is [tokens, experts] fp64.
Status route_f64(const double* scores, int32_t tokens, int32_t experts, int32_t topk,
                 bool renormalize, double routed_scale, int32_t* topk_ids, double* topk_weights);
/// Permute + unpermute round trip: proves the slot mapping is lossless.
Status permute_unpermute_f64(const double* in, const int32_t* topk_ids, int32_t tokens,
                             int32_t hidden, int32_t topk, double* out);

// ---- State / recurrent --------------------------------------------------------
// One sequence per call; every state is updated in place, so a varlen batch, initial states,
// snapshots (call on a prefix) and resumed calls are composed by the test.
/// Causal depthwise conv1d (Mamba / GDN / KDA / LFM2 short conv): x, y [len, dim] row-major,
/// w [dim, width], bias [dim] or null, hist [dim, width - 1] oldest first (updated to the last
/// width - 1 inputs). y = act(bias + sum_j w[j] * window[j]), act kSiLU or kIdentity.
Status causal_conv1d_f64(const double* x, int64_t len, int64_t dim, const double* w,
                         int32_t width, const double* bias, ActFn act, double* hist, double* y);

/// Selective SSM recurrence (Mamba-1 selective scan / state update, Mamba-2 SSD) on state
/// [H, P, N]: per token, dt' = clamp(softplus(dt + dt_bias)) (each step optional),
/// state = state * exp(dt' A) + dt' B[g(h)] x[h, p];  y = state . C[g(h)] + D x;  y *= silu(z).
struct SsmRefArgs {
  const double* x = nullptr;        // [len, H, P]
  const double* dt = nullptr;       // [len, H]
  const double* dt_bias = nullptr;  // [H] or null
  const double* A = nullptr;        // [H] or [H, N] (a_per_state)
  const double* B = nullptr;        // [len, G, N]
  const double* C = nullptr;        // [len, G, N]
  const double* D = nullptr;        // [H] or [H, P] (d_per_p), or null
  const double* z = nullptr;        // [len, H, P] or null
  int64_t len = 0, nheads = 0, headdim = 0, dstate = 0, ngroups = 1;
  bool a_per_state = false, d_per_p = false, dt_softplus = true, has_dt_limit = false;
  double dt_min = 0.0, dt_max = 0.0;
  double* state = nullptr;          // [H, P, N] in/out
  double* y = nullptr;              // [len, H, P]
};
Status ssm_recurrence_f64(const SsmRefArgs& args);

/// Decayed linear attention: S = exp(-slope[h]) S + k^T v, o = scale * q S. q, k [len, H, Dk],
/// v, o [len, H, Dv], state [H, Dk, Dv] in/out.
Status linear_attention_f64(const double* q, const double* k, const double* v,
                            const double* slope, double scale, int64_t len, int64_t heads,
                            int64_t dk, int64_t dv, double* state, double* o);

/// Gated delta rule (GDN per-head / KDA per-channel decay) with optional L2 norm of q and k
/// (x / sqrt(sum x^2 + 1e-6)), q scaled after the norm. Value head h uses key head h / (Hv / Hk).
struct DeltaRuleRefArgs {
  const double* q = nullptr;     // [len, Hk, Dk]
  const double* k = nullptr;     // [len, Hk, Dk]
  const double* v = nullptr;     // [len, Hv, Dv]
  const double* g = nullptr;     // [len, Hv] or [len, Hv, Dk] (per_channel), log-decay
  const double* beta = nullptr;  // [len, Hv]
  int64_t len = 0, num_k_heads = 0, num_v_heads = 0, head_dim_k = 0, head_dim_v = 0;
  bool per_channel = false, use_qk_l2norm = true;
  double scale = 1.0;
  double* state = nullptr;       // [Hv, Dk, Dv] in/out
  double* o = nullptr;           // [len, Hv, Dv]
};
Status delta_rule_f64(const DeltaRuleRefArgs& args);
/// Gate preparation: g = -exp(A_log) softplus(a + dt_bias) (or lower_bound * sigmoid(exp(A_log)
/// (a + dt_bias)) when lower_bound < 0), beta = sigmoid(b). a, g are [rows, Hv, per] with per = Dk
/// (KDA) or 1 (GDN); dt_bias [Hv, per]; b, beta [rows, Hv].
Status delta_rule_gating_f64(const double* a, const double* b, const double* A_log,
                             const double* dt_bias, double lower_bound, int64_t rows,
                             int64_t num_v_heads, int64_t per, double* g, double* beta);

// ---- Sampling -----------------------------------------------------------------
/// softmax(logits / temperature) in fp64 with max subtraction.
Status softmax_f64(const double* logits, int64_t rows, int64_t cols, double temperature,
                   double* probs);
/// Exact top-k/top-p truncation + categorical draw from a caller-provided uniform in
/// (0,1]; returns the selected id. Used to validate the rejection sampler.
Status sample_f64(const double* probs, int64_t vocab, int32_t top_k, double top_p, double min_p,
                  double uniform, int32_t* out_id);

// ---- Communication ------------------------------------------------------------
/// out = sum over ranks of in[r]; element-wise, in rank order (deterministic).
Status allreduce_sum_f64(const double* const* in_per_rank, int32_t ranks, int64_t numel,
                         double* out);

}  // namespace kops::ref
