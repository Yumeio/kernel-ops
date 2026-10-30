#pragma once
// =============================================================================
// kernel-ops | csrc/attention/attention_params.cuh                Catalog: A01-A08, A14
// Shared attention descriptors: masks, modifiers (window/softcap/ALiBi/sinks/
// chunked), work plans, split-KV partial outputs and LSE convention.
// Requires: common/*, kvcache/kv_layout.cuh.
//
// LSE convention: natural-log log-sum-exp of the SCALED logits (s = sm_scale * q.k
// after softcap/bias), fp32, shape [total_q, Hq]. Fully masked rows: lse = -inf, o = 0.
// Causal alignment: bottom-right. Query row i of request b (qo_len_b rows, kv_len_b keys)
// sits at absolute position kv_len_b - qo_len_b + i and sees keys [0, that position].
// =============================================================================

namespace kops::attn {

enum class MaskMode : uint8_t {
  kNone = 0,      // bidirectional (encoders, cross-attn, prefix part of prefix-LM, vision towers)
  kCausal = 1,    // bottom-right causal
  kCustom = 2,    // bit-packed mask per request (tree/EAGLE verification, Gemma-3 image spans,
                  // prefix-LM), bits laid out row-major over [qo_len, kv_len], LSB first
};

/// Position-dependent logit modifiers. Compile-time enabled via AttentionVariant flags.
struct AttentionModifiers {
  float sm_scale = 0.0f;           // 1/sqrt(Dqk) default; Gemma query_pre_attn_scalar^-0.5;
                                   // DeepSeek YaRN: mscale^2 / sqrt(Dqk); Granite attention_multiplier
  float logits_softcap = 0.0f;     // cap * tanh(s / cap) before masking (Gemma 2: 50, Grok-1: 30)
  int32_t window_left = -1;        // keys with pos < q_pos - window_left are masked (-1 = off)
                                   // Mistral 4096, Gemma 2 4096, Gemma 3 1024, GPT-OSS 128,
                                   // Phi-3 2047, Qwen2 (per layer), DeepSeek-V4 SWA branch
  int32_t window_right = -1;       // bidirectional local windows (encoders, ModernBERT); -1 = off
  int32_t chunk_size = 0;          // chunked local attention: q and k must share floor(pos / chunk)
                                   // (Llama-4 attention_chunk_size = 8192); 0 = off
  int32_t num_sink_tokens = 0;     // StreamingLLM: first N positions always visible under SWA
  const float* alibi_slopes = nullptr;  // [Hq]; bias = slope_h * (k_pos - q_pos) (BLOOM, MPT, Baichuan-13B, Falcon-RW)
  const float* sink_logits = nullptr;   // [Hq] learned sinks: softmax denominator += exp(sink_h - m)
                                        // (GPT-OSS, DeepSeek-V4). Added ONCE (first split / at merge).
  float q_scale = 1.0f;            // FP8 Q dequant (per tensor)
  const float* q_scale_ptr = nullptr;
  float k_scale = 1.0f, v_scale = 1.0f;     // FP8 KV per-tensor (device pointers in PagedKVDesc win)
  float rope_theta = 0.0f;         // only for in-kernel RoPE variants (PosEncoding::kRoPEInKernel)
  float rope_scale = 1.0f;
};

/// Compile-time variant flags (template parameter of every attention kernel). Each flag
/// removes dead code when off; the registry records which combinations are compiled.
template <bool kSlidingWindow, bool kSoftcap, bool kAlibi, bool kSinks, bool kChunked, bool kFp8Kv>
struct AttentionVariant {
  static constexpr bool use_sliding_window = kSlidingWindow;
  static constexpr bool use_softcap = kSoftcap;
  static constexpr bool use_alibi = kAlibi;
  static constexpr bool use_sinks = kSinks;
  static constexpr bool use_chunked = kChunked;
  static constexpr bool fp8_kv = kFp8Kv;
};

/// Split-KV partial outputs (A04). Row r of the merged output has merge_indptr[r+1]-merge_indptr[r]
/// partial rows in o_partial / lse_partial.
struct SplitKVBuffers {
  float* o_partial = nullptr;        // [num_partial_rows, Hq, Dvo] fp32
  float* lse_partial = nullptr;      // [num_partial_rows, Hq] fp32 natural log
  const int32_t* merge_indptr = nullptr;  // [total_q + 1]
  int32_t max_splits = 1;
};

/// Device work plan produced by plan functions (host or device planner). One entry per CTA work item.
struct WorkPlan {
  const int32_t* request_indices = nullptr;   // [num_items] request b
  const int32_t* qo_tile_indices = nullptr;   // [num_items] q tile within request
  const int32_t* kv_tile_indices = nullptr;   // [num_items] kv chunk index (split-KV)
  const int32_t* o_indptr = nullptr;          // [B + 1] row offsets into partial buffers
  const int32_t* kv_chunk_size = nullptr;     // [1] tokens per kv chunk
  const uint32_t* total_num_rows = nullptr;   // [1] for padded (graph) batches
  const uint8_t* block_valid_mask = nullptr;  // [num_items] padded items are skipped
  int32_t num_items = 0;
  int32_t padded_num_items = 0;               // fixed grid size under CUDA graphs
  bool split_kv = false;
};

/// Attention problem used for capability lookup and planning.
struct AttentionSpec {
  int32_t num_q_heads = 0, num_kv_heads = 0;
  int32_t head_dim_qk = 0, head_dim_vo = 0;   // independent (MLA prefill 192/128, DSV4, Gemma 256)
  DType q_dtype = DType::kBFloat16;
  DType kv_dtype = DType::kBFloat16;
  DType o_dtype = DType::kBFloat16;
  kv::KVCodec k_codec = kv::KVCodec::kNative;
  kv::KVCodec v_codec = kv::KVCodec::kNative;
  kv::KVLayout kv_layout = kv::KVLayout::kNHD;
  int32_t page_size = 0;                      // 0 = ragged (non-paged)
  MaskMode mask = MaskMode::kCausal;
  bool sliding_window = false, softcap = false, alibi = false, sinks = false, chunked = false;
  bool return_lse = false;
  bool fp8_tensor_core = false;               // compute QK/PV in FP8 (distinct from FP8 KV storage)
};

}  // namespace kops::attn
