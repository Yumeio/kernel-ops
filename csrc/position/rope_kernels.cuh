#pragma once
// =============================================================================
// kernel-ops | csrc/position/rope_kernels.cuh                         Catalog: R01
// RoPE (all scaling families), partial/offset rotary, M-RoPE, fused QK-norm+RoPE,
// Llama-4 attention temperature tuning, ALiBi slopes.
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh.
//
// cos/sin cache layout (library-wide): cache[pos, 0:rot/2] = cos, cache[pos, rot/2:rot] = sin,
// fp32 (or scalar_t for bandwidth-bound decode). Any scaling (linear, NTK, YaRN, Llama-3,
// LongRoPE) and YaRN mscale/attention_factor are baked in by build_cos_sin_cache_kernel.
// =============================================================================

namespace kops::rope {

/// Frequency scaling families.
enum class RopeScaling : uint8_t {
  kNone = 0,
  kLinear = 1,        // positions / factor (Code Llama 2-era, Vicuna-LongChat)
  kDynamicNTK = 2,    // base * (factor*L/L0 - (factor-1))^(d/(d-2)) when L > L0 (Qwen1, Phi)
  kYaRN = 3,          // ramp-blended interpolation + mscale (DeepSeek V2-V4, Qwen2.5/3 1M, Kimi, GPT-OSS)
  kLlama3 = 4,        // low/high frequency factor band (Llama 3.1/3.2/3.3)
  kLongRoPE = 5,      // per-dim short/long factors (Phi-3/3.5/4-mini, MiniCPM)
  kNTKAlpha = 6,      // base * alpha^(d/(d-2)) (Hunyuan, InternLM2 dynamic)
};

/// Host-side description; compute_inv_freq() turns it into inv_freq[rot/2].
struct RopeConfig {
  RopeScaling scaling = RopeScaling::kNone;
  float theta = 10000.0f;              // base (LLaMA 1/2: 1e4, Llama 3: 5e5, Qwen3: 1e6, Gemma3 global: 1e6)
  int32_t rotary_dim = 0;              // <= head_dim; partial rotary (NeoX 0.25, Phi-2 0.4, GLM 0.5, DeepSeek 64/192)
  float factor = 1.0f;
  int32_t original_max_position = 0;
  float beta_fast = 32.0f, beta_slow = 1.0f;       // YaRN
  float mscale = 1.0f, mscale_all_dim = 0.0f;      // YaRN / DeepSeek (attention factor)
  float attention_factor = 0.0f;                   // explicit override (0 = derive)
  float low_freq_factor = 1.0f, high_freq_factor = 4.0f;  // Llama 3
  const float* short_factor = nullptr;   // LongRoPE [rot/2] host array
  const float* long_factor = nullptr;    // LongRoPE [rot/2] host array
  bool truncate = true;                  // YaRN low/high bound truncation (GPT-OSS uses false)
};

/**
 * build_cos_sin_cache_kernel — cache[p, i] = mscale * cos(p * inv_freq[i]), sin likewise.
 * Launch : grid = ceil(max_position / 4), block = rot/2 (x4 positions per CTA). Run once at load.
 * Note   : positions are computed as float(p) * inv_freq in fp32 using exact integer p; for
 *          1M-token contexts use kOnTheFly in rotary_embedding_kernel instead of a 1M-row table.
 */
template <typename cache_t>
__global__ void build_cos_sin_cache_kernel(
    cache_t* __restrict__ cos_sin_cache,    // [max_position, rot_dim]
    const float* __restrict__ inv_freq,     // [rot_dim / 2]
    const float mscale,
    const int max_position, const int rot_dim);

/**
 * rotary_embedding_kernel — in-place RoPE on Q and (optionally) K.
 *   Rotates dims [rot_offset, rot_offset + rot_dim) of every head; the rest is untouched.
 *   rot_offset != 0: DeepSeek MLA / V4 (rope part stored after the nope part), Kimi MLA.
 * Models : every RoPE model. kGPTJ for GPT-J, ChatGLM/GLM-4, DeepSeek MLA rope dims; kNeoX otherwise.
 * cos_sin_cache_offsets: per-token row offset into a concatenated cache (Phi-3 LongRoPE short/long
 *          switch, multi-LoRA long-context); null = 0.
 * kOnTheFly: compute sin/cos from inv_freq with sincosf in fp32 (no table; 1M-context models).
 * Launch : grid = num_tokens, block = min(512, max(num_heads, num_kv_heads) * rot_dim / 2).
 */
template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kOnTheFly>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,     // [num_tokens] absolute positions
    scalar_t* __restrict__ query,              // [num_tokens, num_heads, head_size] (strided)
    scalar_t* __restrict__ key,                // [num_tokens, num_kv_heads, head_size] or null
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]; null if kOnTheFly
    const float* __restrict__ inv_freq,        // [rot_dim/2]; used if kOnTheFly
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [num_tokens] or null
    const float on_the_fly_mscale,             // YaRN magnitude when kOnTheFly
    const int rot_dim,                         // rotated width (even)
    const int rot_offset,                      // first rotated dim inside a head
    const int64_t query_token_stride,          // query.stride(0)
    const int64_t query_head_stride,           // query.stride(1)
    const int64_t key_token_stride,
    const int64_t key_head_stride,
    const int num_heads, const int num_kv_heads, const int head_size, const int num_tokens);

/**
 * mrope_kernel — multimodal RoPE: positions are (t, h, w) triples; rotary frequency band i
 * takes its position from section s(i).
 *   kInterleaved = false: contiguous sections mrope_section = [16, 24, 24] (Qwen2-VL / Qwen2.5-VL)
 *   kInterleaved = true : t/h/w interleaved across frequencies (Qwen3-VL, GLM-4.5V)
 * Text tokens carry t == h == w, reducing to standard RoPE.
 */
template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kInterleaved>
__global__ void mrope_kernel(
    const int64_t* __restrict__ positions,     // [3, num_tokens]
    scalar_t* __restrict__ query,              // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // [num_tokens, num_kv_heads, head_size] or null
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]
    const int mrope_section_t, const int mrope_section_h, const int mrope_section_w,
    const int rot_dim,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int num_heads, const int num_kv_heads, const int head_size, const int num_tokens);

/**
 * fused_qk_norm_rope_kernel — in-place on packed QKV: per-head RMSNorm(q), RMSNorm(k)
 * (weights q_w/k_w, offset), then RoPE; V untouched. One read + one write of Q and K.
 * Models : Qwen3 / Qwen3-MoE / Qwen3-Next / Qwen3.5, Gemma 3 (offset 1, local/global caches),
 *          GLM-4.5 (qk norm on), DeepSeek-V4 (q / KV-entry RMSNorm + partial RoPE).
 * For K-cache store in the same pass use kvcache::fused_qkv_norm_rope_append_kernel.
 */
template <typename scalar_t, typename cache_t, int HEAD_DIM, RopeStyle kStyle>
__global__ void fused_qk_norm_rope_kernel(
    scalar_t* __restrict__ qkv,                // [num_tokens, (Hq + 2*Hkv) * HEAD_DIM]
    const int64_t token_stride,
    const scalar_t* __restrict__ q_weight,     // [HEAD_DIM] or null (no q norm)
    const scalar_t* __restrict__ k_weight,     // [HEAD_DIM] or null (no k norm)
    const int64_t* __restrict__ positions,     // [num_tokens]
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]
    const float weight_offset, const float epsilon,
    const int rot_dim, const int rot_offset,
    const int num_tokens, const int num_q_heads, const int num_kv_heads);

/**
 * attn_temperature_tuning_kernel — Llama-4 NoPE layers (iRoPE): scale queries by
 *   s(p) = log(floor((p + 1) / floor_scale) + 1) * attn_scale + 1
 * (floor_scale = 8192, attn_scale = 0.1 in Llama-4 configs). Applied in place.
 */
template <typename scalar_t>
__global__ void attn_temperature_tuning_kernel(
    scalar_t* __restrict__ query,              // [num_tokens, num_heads, head_size]
    const int64_t query_token_stride,
    const int64_t* __restrict__ positions,     // [num_tokens]
    const float floor_scale, const float attn_scale,
    const int num_tokens, const int num_heads, const int head_size);

}  // namespace kops::rope
