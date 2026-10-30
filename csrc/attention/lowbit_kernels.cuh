#pragma once
// =============================================================================
// kernel-ops | csrc/attention/lowbit_kernels.cuh           Catalog: A09 (EXPERIMENTAL, P2/R)
// Low-bit / rotated KV codecs and attention over them (attention.rs TurboQuant-
// derived K8V4 / K4V4 / K3V4). Names describe the implementation, not a paper:
// codec = (transform, bits, scale granularity, packing version).
// Opt-in only (namespace kops::experimental). Physical bytes come from
// kv::kv_physical_bytes(), never from nominal bit width.
// Requires: common/*, kvcache/kv_layout.cuh, attention/attention_params.cuh.
// =============================================================================

namespace kops::experimental::lowbit {

/// Element codec for one of K or V.
enum class LowBitFormat : uint8_t {
  kFP8 = 0,          // e4m3, per (token, head) absmax scale, no rotation      (K of K8V4)
  kUniform4 = 1,     // signed 4-bit uniform, per (token, head) fp32 absmax
  kUniform3 = 2,     // signed 3-bit uniform, bit-packed 8 values / 3 bytes
  kNVFP4 = 3,        // e2m1 + E4M3 per 16 (tracked separately from uniform int4)
};

/// Codec descriptor (versioned; stored next to the cache so readers never guess).
struct LowBitCodec {
  LowBitFormat k_format;
  LowBitFormat v_format;
  QuantTransform k_transform;    // kSignedHadamard over head_dim for rotated codecs, or kNone
  QuantTransform v_transform;
  uint32_t transform_seed;       // sign convention; must be identical for Q and K
  uint16_t packing_version;
  int16_t head_dim;
};

/**
 * lowbit_kv_store_kernel — quantize + (optionally) rotate new K/V and write them into low-bit pages
 * with per-(token, head) scales. Q must be rotated with the same transform at attention time.
 */
template <typename scalar_t, LowBitFormat kKFormat, LowBitFormat kVFormat, int HEAD_DIM>
__global__ void lowbit_kv_store_kernel(
    const scalar_t* __restrict__ key,           // [num_tokens, Hkv, HEAD_DIM]
    const scalar_t* __restrict__ value,         // [num_tokens, Hkv, HEAD_DIM]
    const kv::PagedKVDesc cache,                // codec kLowBitWHT, side scales in k/v_token_scales
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const LowBitCodec codec, const int num_tokens);

/**
 * lowbit_decode_kernel — decode attention reading low-bit K/V, dequantizing in registers;
 * rotated codecs compute scores in the rotated basis (q rotated once per step) and apply the
 * inverse transform to the output when V is rotated.
 */
template <int HEAD_DIM, int GQA_GROUP, int PAGE_SIZE, LowBitFormat kKFormat, LowBitFormat kVFormat,
          typename DTypeQ, typename DTypeO>
__global__ void lowbit_decode_kernel(
    const void* __restrict__ q,                 // [B, Hq, HEAD_DIM]
    const kv::PagedKVDesc cache,
    const LowBitCodec codec,
    const attn::AttentionModifiers mod,
    void* __restrict__ o,                       // [B, Hq, HEAD_DIM]
    const attn::SplitKVBuffers split,
    const int batch_size, const int num_q_heads, const int num_kv_heads);

/// lowbit_prefill_kernel — paged prefill over low-bit KV (dequantize tiles into smem).
template <typename Traits, attn::MaskMode kMask, LowBitFormat kKFormat, LowBitFormat kVFormat>
__global__ void lowbit_prefill_kernel(
    const void* __restrict__ q,                 // [total_q, Hq, D]
    const int32_t* __restrict__ qo_indptr,
    const kv::PagedKVDesc cache,
    const LowBitCodec codec,
    const attn::AttentionModifiers mod,
    void* __restrict__ o, float* __restrict__ lse,
    const attn::WorkPlan plan, const attn::SplitKVBuffers split,
    const int batch_size, const int num_q_heads, const int num_kv_heads);

}  // namespace kops::experimental::lowbit
