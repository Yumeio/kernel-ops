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
/// Dense causal/window/softcap attention with optional sinks and ALiBi; the golden
/// reference for A01/A02/A03/A06/A07/A08. `qkv` is [rows, heads, dim]; `k/v` are
/// ragged [kv_rows, kv_heads, dim] with `kv_indptr[B+1]`. Writes o and lse (-inf row
/// -> o = 0, lse = -inf).
Status attention_f64(const double* q, const double* k, const double* v, const int32_t* qo_indptr,
                     const int32_t* kv_indptr, int32_t batch, int32_t q_heads, int32_t kv_heads,
                     int32_t head_dim_qk, int32_t head_dim_vo, double sm_scale,
                     double logits_softcap, int32_t window_left, int32_t num_sinks,
                     const double* sink_logits, const double* alibi_slopes, double* o,
                     double* lse, bool causal);
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
/// Causal depthwise conv1d (Mamba / short-conv), silu/identity activation optional.
Status causal_conv1d_f64(const double* x, const double* w, const double* bias, int64_t batch,
                         int64_t channels, int64_t length, int32_t width, ActFn act,
                         double* y);
/// Selective scan (Mamba-1) over a batch of sequences.
Status selective_scan_f64(const double* u, const double* delta, const double* a, const double* b,
                          const double* c, const double* d, int64_t batch, int64_t dim,
                          int64_t dstate, int64_t length, double* y);

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
