#pragma once
// =============================================================================
// kernel-ops | csrc/kvcache/kvcache_kernels.cuh                  Catalog: K01, K02, K03
// KV-cache writes, gathers, movement and codec conversion. Feature parity target:
// vLLM csrc/cache_kernels.cu + cache_kernels_fused.cu, FlashInfer page.py,
// HPC-Ops rope_norm_store_kv, attention.rs cache/scale-update paths.
//
// Contract for every slot-mapping writer:
//   * num_tokens = number of slot_mapping entries (real tokens). Source tensors (key, value,
//     q_pe, kv_c, ...) may have MORE rows (CUDA-graph padding); extra rows are never read.
//   * slot = page * page_size + offset; slot < 0 means "skip" (padding / dummy rows).
//   * Only the addressed cache records are written. Scales are dequant multipliers
//     (stored = convert(x / scale)), read from device memory so graphs replay with new scales.
//   * Launch: one CTA per token (grid.x = num_tokens) unless stated otherwise.
// Requires: common/*, kvcache/kv_layout.cuh.
// =============================================================================

namespace kops::kv {

// =============================================================================
// K01  standard K/V writes
// =============================================================================

/**
 * reshape_and_cache_flash_kernel — write K and V of each token into NHD or HND pages.
 *   NHD: cache [num_blocks, block_size, num_heads, head_size] (head_stride == head_size)
 *   HND: cache [num_blocks, num_heads, block_size, head_size]
 * Fast path (NHD and per-tensor scale): the CTA copies the token's num_heads*head_size row with
 * VEC_SIZE-wide vector stores. Otherwise one warp per head (strided heads or per-head scales).
 * kCodec: kNative | kFP8PerTensor | kFP8PerHead (cache_t = uint8 storage of e4m3/e5m2).
 * Launch : grid = num_tokens, block = min(num_heads * head_size / VEC_SIZE, 512) (>= 32).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens(+pad), num_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens(+pad), num_heads, head_size_v]
    cache_t* __restrict__ key_cache,           // NHD or HND, see above
    cache_t* __restrict__ value_cache,         // same layout as key_cache
    const int64_t* __restrict__ slot_mapping,  // [num_tokens], -1 = skip
    const int64_t block_stride,                // key_cache.stride(0)
    const int64_t page_stride,                 // stride between tokens of a page (NHD: num_heads*head_size; HND: head_size)
    const int64_t head_stride,                 // stride between heads (NHD: head_size; HND: block_size*head_size)
    const int64_t key_stride,                  // key.stride(0)
    const int64_t value_stride,                // value.stride(0)
    const int num_heads, const int head_size,
    const int head_size_v,                     // == head_size except models with Dk != Dv
    const int block_size,
    const float* __restrict__ k_scale,         // [1] or [num_heads]; unused for kNative
    const float* __restrict__ v_scale,         // [1] or [num_heads]
    const int kv_scale_stride);                // 0 = per-tensor scale, 1 = per-head scale

/**
 * reshape_and_cache_legacy_kernel — vLLM paged-attention v1/v2 layout (KVLayout::kVllmLegacy):
 *   key_cache   [num_blocks, num_heads, head_size / x, block_size, x]   (x = 16 bytes / sizeof(cache_t))
 *   value_cache [num_blocks, num_heads, head_size, block_size]          (V transposed within a page)
 * Required only by legacy paged-attention kernels; new backends use NHD/HND.
 * Launch : grid = num_tokens, block = num_heads * head_size / x.
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void reshape_and_cache_legacy_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens(+pad), num_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens(+pad), num_heads, head_size]
    cache_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
    cache_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size,
    const int x,                               // elements per 16-byte vector of cache_t
    const float* __restrict__ k_scale,         // [1]
    const float* __restrict__ v_scale);        // [1]

/**
 * append_paged_kv_kernel — FlashInfer CSR append: token i of the append batch belongs to request
 * batch_indices[i] at position positions[i] inside that request's page list (no slot mapping).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec, KVLayout kLayout>
__global__ void append_paged_kv_kernel(
    const PagedKVDesc desc,                     // CSR page table of the destination
    const scalar_t* __restrict__ append_key,    // [nnz, Hkv, Dk]
    const scalar_t* __restrict__ append_value,  // [nnz, Hkv, Dv]
    const int32_t* __restrict__ batch_indices,  // [nnz]
    const int32_t* __restrict__ positions,      // [nnz] position within the request
    const int64_t key_stride_n, const int64_t key_stride_h,
    const int64_t value_stride_n, const int64_t value_stride_h,
    const int nnz);

/**
 * reshape_and_cache_per_token_quant_kernel — K01 + K03: quantize each (token, head) with its own
 * absmax scale (FP8 / INT8 / INT4 per-token-head codecs) and write payload + side scales.
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec, KVLayout kLayout>
__global__ void reshape_and_cache_per_token_quant_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens(+pad), Hkv, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens(+pad), Hkv, head_size]
    const PagedKVDesc cache,                   // k/v_token_scales receive per-(token, head) scales (+ zeros)
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t key_stride, const int64_t value_stride,
    const int num_tokens);

/// Arguments of fused_qkv_norm_rope_append_kernel (K01 fused path).
struct FusedQKVAppendParams {
  void* qkv;                        // [num_tokens, (Hq + 2*Hkv) * D] packed projection output (in place for q)
  int64_t qkv_token_stride;
  const void* q_norm_weight;        // [D] or null
  const void* k_norm_weight;        // [D] or null
  float norm_eps;
  float norm_weight_offset;
  const int64_t* positions;         // [num_tokens]
  const void* cos_sin_cache;        // [max_pos, rot_dim] or null (no RoPE: NoPE layers, Llama-4 iRoPE)
  int32_t rot_dim, rot_offset;
  RopeStyle rope_style;
  void* q_out;                      // optional separate Q output (e.g. FP8 Q for FP8 attention); null = in place
  int64_t q_out_token_stride;
  const float* q_out_scale;         // FP8 Q quant scale or null
  PagedKVDesc cache;                // destination pages
  const int64_t* slot_mapping;      // [num_tokens], -1 = skip
  int32_t num_tokens, num_q_heads, num_kv_heads, head_dim;
};

/**
 * fused_qkv_norm_rope_append_kernel — one pass over the QKV projection:
 *   q <- RoPE(Norm(q)) [-> FP8];  k <- codec(RoPE(Norm(k))) -> cache;  v <- codec(v) -> cache
 * Models : Qwen3/3.5 family, Gemma 3, GLM-4.5, Llama 2/3 (no norm), Mistral, Phi, GPT-OSS
 *          (no norm, sinks handled in attention), HPC-Ops rope_norm_store_kv equivalent.
 * Numerics must match the unfused chain bit-for-bit in fp32 order (norm -> rope -> round once).
 */
template <typename scalar_t, typename cache_t, int HEAD_DIM, KVCodec kCodec, KVLayout kLayout, bool kHasNorm>
__global__ void fused_qkv_norm_rope_append_kernel(const __grid_constant__ FusedQKVAppendParams params);

/**
 * ring_kv_append_kernel — per-sequence ring buffer for sliding-window layers: position p is written
 * at ring slot p % window. Saves pages for SWA-only layers (Mistral, Gemma 2/3 local layers,
 * GPT-OSS 128-window layers, Llama-4 chunked layers, DeepSeek-V4 SWA branch and uncompressed tail).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void ring_kv_append_kernel(
    const scalar_t* __restrict__ key,           // [num_tokens, Hkv, Dk]
    const scalar_t* __restrict__ value,         // [num_tokens, Hkv, Dv] or null (shared KV)
    cache_t* __restrict__ ring_k,               // [num_slots, window, Hkv, Dk]
    cache_t* __restrict__ ring_v,               // [num_slots, window, Hkv, Dv] or null
    const int32_t* __restrict__ token_slot,     // [num_tokens] state slot of the owning sequence (-1 skip)
    const int64_t* __restrict__ positions,      // [num_tokens] absolute positions
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int window, const int num_kv_heads, const int head_dim_k, const int head_dim_v,
    const int num_tokens);

// =============================================================================
// K01  MLA latent cache writes (DeepSeek V2/V3/V3.2, Kimi K2/K2.5/Linear/K3, MiniCPM3)
// =============================================================================

/**
 * concat_and_cache_mla_kernel — cache[slot] = codec([ kv_c[t, 0:kv_lora_rank] | k_pe[t, 0:pe_dim] ]).
 * kCodec: kNative (bf16/fp16, 576 elements) | kFP8PerTensor (e4m3/e5m2, one scale).
 * Launch : grid = num_tokens, block = min(kv_lora_rank, 512).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,         // [num_tokens(+pad), kv_lora_rank]   (512)
    const scalar_t* __restrict__ k_pe,         // [num_tokens(+pad), pe_dim]         (64)
    cache_t* __restrict__ kv_cache,            // [num_blocks, block_size, kv_lora_rank + pe_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    // kv_cache.stride(0)
    const int entry_stride,                    // kv_cache.stride(1)
    const int kv_c_stride,                     // kv_c.stride(0)
    const int k_pe_stride,                     // k_pe.stride(0)
    const int kv_lora_rank, const int pe_dim, const int block_size,
    const float* __restrict__ scale);          // [1] per-tensor FP8 scale; unused for kNative

/**
 * concat_and_cache_mla_grouped_kernel — the same write for SEVERAL layers in one launch (MTP /
 * EAGLE multi-layer drafters inserting context K/V for every draft layer). grid = (num_tokens,
 * num_layers); each layer reads its own cache base pointer and scale (copy_pages pointer-array pattern).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void concat_and_cache_mla_grouped_kernel(
    const scalar_t* __restrict__ kv_c,          // [num_layers, num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,          // [num_layers, num_tokens, pe_dim]
    const int64_t* __restrict__ kv_cache_ptrs,  // [num_layers] base addresses of each layer's cache
    const float* __restrict__ kv_scales,        // [num_layers] per-layer FP8 scale, or null for kNative
    const int64_t* __restrict__ slot_mapping,   // [num_layers, num_tokens]
    const int64_t kv_c_layer_stride, const int64_t kv_c_token_stride,
    const int64_t k_pe_layer_stride, const int64_t k_pe_token_stride,
    const int64_t slot_layer_stride,
    const int64_t block_stride, const int64_t entry_stride,
    const int kv_lora_rank, const int pe_dim, const int block_size);

/**
 * concat_and_cache_ds_mla_kernel — KVCodec::kMLAFp8DS 656-byte records (DeepSeek-V3.2 / FlashMLA
 * sparse FP8 format). Per token: 4 NoPE tiles of 128 values, each tile quantized with its own
 * power-of-two fp32 scale 2^ceil(log2(max(amax/448, 1e-4))); RoPE kept in bf16. pe_dim == 0
 * (NoPE rows) zeroes the RoPE tail.
 * Launch : grid = num_tokens, block = 96 (2 warps: NoPE, half-warp per tile with lanes 0/16
 *          writing scales; 1 warp: RoPE, 2 values per thread).
 */
template <typename scalar_t>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,         // [num_tokens(+pad), 512]
    const scalar_t* __restrict__ k_pe,         // [num_tokens(+pad), 64] (or ignored when pe_dim == 0)
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size, 656] bytes
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    // bytes
    const int entry_stride,                    // bytes (656, or padded)
    const int kv_c_stride, const int k_pe_stride,
    const int kv_lora_rank,                    // 512
    const int pe_dim,                          // 64 or 0
    const int block_size);

/**
 * concat_and_cache_ds_mla_nvfp4_kernel — KVCodec::kMLANvfp4DS 352-byte records (NVFP4 latent cache).
 * Sm100a only; record layout is owned by this kernel and its readers
 * (cp_gather_and_upconvert_ds_mla_nvfp4_kernel, NVFP4 sparse-MLA kernels).
 */
template <typename scalar_t>
__global__ void concat_and_cache_ds_mla_nvfp4_kernel(
    const scalar_t* __restrict__ kv_c,         // [num_tokens(+pad), 512]
    const scalar_t* __restrict__ k_pe,         // [num_tokens(+pad), 64]
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size, 352] bytes
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t entry_stride,
    const int kv_c_stride, const int k_pe_stride, const int block_size);

/**
 * concat_and_cache_mla_rope_fused_kernel — RoPE on q_pe (all heads, in place) and k_pe (single
 * latent head, in place), then write [kv_c | rotated k_pe] into the MLA cache. Replaces
 * "DeepseekScalingRotaryEmbedding + concat_and_cache_mla" (2 launches -> 1).
 * cos_sin_cache: [max_position, rot_dim] = [cos(0..rot/2) | sin(0..rot/2)], YaRN mscale baked in.
 * kNeoX selects rotate-half pairing, otherwise GPT-J interleaved (DeepSeek uses interleaved).
 * raw_kv_t is the bit-container type used for the cache store (uint16_t for fp16/bf16) so the
 * stored bits equal the rotated values exactly.
 * Launch : grid = num_tokens, block = min(max(num_q_heads * rot_dim / 2, kv_lora_rank), 512).
 */
template <typename qk_t, typename cos_sin_t, bool kNeoX, typename raw_kv_t, typename cache_t, KVCodec kCodec>
__global__ void concat_and_cache_mla_rope_fused_kernel(
    const int64_t* __restrict__ positions,          // [num_tokens(+pad)]
    qk_t* __restrict__ q_pe,                        // [num_tokens(+pad), num_q_heads, rot_dim] in/out
    qk_t* __restrict__ k_pe,                        // [num_tokens(+pad), rot_dim] in/out
    const qk_t* __restrict__ kv_c,                  // [num_tokens(+pad), kv_lora_rank]
    const cos_sin_t* __restrict__ rope_cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim,
    const int64_t q_pe_stride_token, const int64_t q_pe_stride_head,
    const int64_t k_pe_stride, const int64_t kv_c_stride,
    const int num_q_heads,
    cache_t* __restrict__ kv_cache,                 // [num_blocks, block_size, kv_lora_rank + rot_dim]
    const int64_t* __restrict__ slot_mapping,       // [num_tokens] (num_tokens <= padded rows)
    const int block_stride, const int entry_stride, const int kv_lora_rank, const int block_size,
    const float* __restrict__ kv_cache_quant_scale);  // [1]; unused for kNative

// =============================================================================
// K01  sparse-attention indexer cache (DSA / CSA lightning indexer)
// =============================================================================

/**
 * indexer_k_quant_and_cache_kernel — quantize indexer keys to e4m3 with one scale per
 * quant_block_size (128) values and store them in the SPLIT page layout of KVCodec::kIndexerFP8:
 *   page p = [ block_size x head_dim e4m3 keys | block_size x head_dim/quant_block fp32 scales ]
 * scale = max(amax, 1e-4) / 448, rounded up to a power of two when use_ue8m0.
 * Launch : grid = (num_tokens, ceil(head_dim / (quant_block_size * 4))), block = (32, 4); each warp
 *          covers one quant block (4 values per lane, warp-level amax).
 */
template <typename scalar_t>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,            // [num_tokens, head_dim]  (head_dim = 128)
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size, cache_stride] bytes
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,
    const int quant_block_size,                // 128
    const int cache_block_size,                // tokens per page
    const int64_t cache_block_stride,          // bytes per page
    const bool use_ue8m0);

/**
 * cp_gather_indexer_k_quant_cache_kernel — gather indexer keys + scales of all cached tokens of the
 * requests in a ragged batch (prefill-time indexer over context) into contiguous buffers.
 * Each CTA row (threadIdx.y) owns one output token and finds its request with a shared-memory scan
 * over cu_seq_lens; rows beyond cu_seq_lens[num_reqs] (upper-bound num_tokens) are skipped.
 * Launch : grid = (ceil(num_tokens / BLOCK_Y), ceil(head_dim / (8 * 16))), block = (8, BLOCK_Y),
 *          BLOCK_Y in {1, 2, 4, 8, 16, 32} chosen by num_tokens.
 */
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const uint8_t* __restrict__ kv_cache,      // [num_blocks, block_size, cache_stride]
    uint8_t* __restrict__ dst_k,               // [num_tokens, head_dim] e4m3
    uint8_t* __restrict__ dst_scale,           // [num_tokens, head_dim / quant_block_size * 4] fp32 bytes
    const int32_t* __restrict__ block_table,   // [batch_size, num_blocks_per_seq]
    const int32_t* __restrict__ cu_seq_lens,   // [batch_size + 1]
    const int batch_size,
    const int64_t token_stride,                // dst_k.stride(0)
    const int64_t head_dim,
    const int64_t block_stride,                // kv_cache.stride(0) bytes
    const int64_t cache_token_stride,          // kv_cache.stride(1) bytes
    const int64_t cache_block_size,            // tokens per page
    const int num_blocks_per_seq,              // block_table.size(1)
    const int num_tokens,                      // upper bound allowed
    const int quant_block_size);

// =============================================================================
// K02  gathers out of the paged cache (chunked prefill context, MLA up-projection,
//      KV transfer / offload staging). All use GatherSpec + GatherPageTask mapping.
// =============================================================================

/**
 * gather_and_maybe_dequant_cache_kernel — copy the cached tokens of each request (positions
 * [seq_starts[r], seq_starts[r] + len_r)) into dst rows [cu_seq_lens[r], cu_seq_lens[r+1]),
 * dequantizing FP8 per-tensor caches to scalar_t. ENTRY_SIZE is the record width in elements
 * (576 MLA latent, 512 latent without RoPE, 320 indexer/other); compiled per size for vectorization.
 * Logical blocks beyond block_table_stride are skipped (partial tables).
 * Launch : grid = ceil(num_tokens / block_size) + 2 * num_reqs, block = 256 (128 when >= 2^20 tokens).
 */
template <typename scalar_t, typename cache_t, KVCodec kCodec, int ENTRY_SIZE>
__global__ void gather_and_maybe_dequant_cache_kernel(
    const cache_t* __restrict__ src_cache,     // [num_blocks, block_size, ENTRY_SIZE...]
    scalar_t* __restrict__ dst,                // [num_tokens, ENTRY_SIZE]
    const GatherSpec spec,
    const int32_t block_size,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride,
    const float* __restrict__ scale);          // [1] FP8 dequant scale; unused for kNative

/**
 * cp_gather_cache_kernel — byte-exact gather of records of any type/width (no conversion).
 *   kContiguousEntries: source records of a page and destination rows are both dense, so a whole
 *                       page segment is one contiguous memcpy-like range
 *   kVectorized       : all pointers/strides 16-byte aligned -> int4 copies (2 in flight per thread)
 * Launch : grid = ceil(num_tokens / block_size) + 2 * num_reqs, block = 256 (32 threads per token
 *          in the non-contiguous form).
 */
template <bool kContiguousEntries, bool kVectorized>
__global__ void cp_gather_cache_kernel(
    const uint8_t* __restrict__ src_cache,     // [num_blocks, block_size, entry_bytes]
    uint8_t* __restrict__ dst,                 // [num_tokens, entry_bytes]
    const GatherSpec spec,
    const int32_t block_size, const int32_t entry_size_bytes,
    const int64_t cache_block_stride,          // bytes
    const int64_t cache_entry_stride,          // bytes
    const int64_t dst_entry_stride);           // bytes

/**
 * cp_gather_and_upconvert_ds_mla_fp8_kernel — kMLAFp8DS 656-byte records -> bf16 [num_tokens, 576]
 * (512 NoPE dequantized with their tile scales + 64 RoPE copied). Supports partially offloaded
 * caches: when device_row_ids is given, source row r is read from the device cache if
 * device_row_ids[r] >= 0, otherwise from pinned host memory host_cache[host_row_ids[r]] through UVA
 * (no staging copy). One warp per token.
 * Launch : grid = ceil(num_tokens / block_size) + 2 * num_reqs, block = 512 (16 warps).
 */
__global__ void cp_gather_and_upconvert_ds_mla_fp8_kernel(
    const uint8_t* __restrict__ src_cache,     // [num_blocks, block_size, 656]
    bf16_t* __restrict__ dst,                  // [num_tokens, 576]
    const GatherSpec spec,                     // cu_seq_lens = workspace_starts (per-request output offsets)
    const int32_t block_size,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride,
    const uint8_t* __restrict__ host_cache,    // [num_host_rows, 656] pinned, or null
    const int32_t* __restrict__ host_row_ids,  // [num_blocks * block_size] row in host_cache, or null
    const int32_t* __restrict__ device_row_ids,// [num_blocks * block_size] row in src_cache or -1, or null
    const int64_t host_entry_stride);

/// cp_gather_and_upconvert_ds_mla_nvfp4_kernel — kMLANvfp4DS 352-byte records -> bf16 [num_tokens, 576].
/// Sm100a only (provider format).
__global__ void cp_gather_and_upconvert_ds_mla_nvfp4_kernel(
    const uint8_t* __restrict__ src_cache,     // [num_blocks, block_size, 352]
    bf16_t* __restrict__ dst,                  // [num_tokens, 576]
    const GatherSpec spec,
    const int32_t block_size,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride);

/**
 * gather_pages_kernel — index-driven gather (arbitrary token slots or whole pages) with optional
 * dequant; scatter_pages_kernel is its inverse (load offloaded KV, PD-disaggregation receive).
 */
template <typename cache_t, typename out_t, KVCodec kCodec>
__global__ void gather_pages_kernel(
    out_t* __restrict__ out,                    // [num_items, entry_elems] contiguous
    const cache_t* __restrict__ cache,          // [num_pages, page_size, entry]
    const int64_t* __restrict__ indices,        // [num_items] token slots or page ids
    const float* __restrict__ scale,            // dequant scale for fp8 codecs or null
    const PagedKVDesc desc, const bool gather_by_token, const int num_items);

template <typename in_t, typename cache_t, KVCodec kCodec>
__global__ void scatter_pages_kernel(
    cache_t* __restrict__ cache,
    const in_t* __restrict__ src,               // [num_items, entry_elems]
    const int64_t* __restrict__ indices,        // [num_items]
    const float* __restrict__ scale,
    const PagedKVDesc desc, const bool scatter_by_token, const int num_items);

// =============================================================================
// K02  page movement on device (byte-preserving, codec-agnostic)
// =============================================================================

/**
 * copy_pages_kernel — dst_page <- src_page for every (src, dst) pair across all layers' caches
 * (vLLM copy_blocks): COW on prefix-cache hits, beam / n>1 forks, defragmentation.
 * Launch : grid = (num_pairs, num_layers), block = 256; 16-byte vectorized when aligned.
 */
__global__ void copy_pages_kernel(
    void* const* __restrict__ layer_caches,     // [num_layers * caches_per_layer] base pointers
    const int64_t* __restrict__ page_pairs,     // [num_pairs, 2] (src, dst)
    const int64_t page_bytes,                   // bytes of one page of one cache tensor
    const int caches_per_layer,                 // 2 (K,V) | 1 (MLA, shared KV) | 3 (K, V, scales)
    const int num_pairs, const int num_layers);

/// zero_pages_kernel — memset selected pages (fresh pages of quantized caches, debugging, indexer pages).
__global__ void zero_pages_kernel(
    void* const* __restrict__ layer_caches,
    const int64_t* __restrict__ pages,          // [num_pages]
    const int64_t page_bytes, const int caches_per_layer, const int num_pages, const int num_layers);

// =============================================================================
// K03  codec conversion (scale metadata always travels with converted pages)
// =============================================================================

/**
 * convert_cache_kernel — elementwise conversion of whole cache tensors (vLLM convert_fp8): bf16/fp16/
 * fp32 <-> e4m3/e5m2 with a per-tensor scale. Testing, checkpoint migration, offline cache import.
 * Launch : grid = num_blocks, block = min(block_stride, 512).
 */
template <typename out_t, typename in_t, KVCodec kCodec>
__global__ void convert_cache_kernel(
    const in_t* __restrict__ src_cache,         // [num_blocks, ...]
    out_t* __restrict__ dst_cache,              // [num_blocks, ...]
    const float scale,
    const int64_t block_stride);                // elements per block

/**
 * kv_requantize_pages_kernel — convert selected pages between codecs (bf16 -> fp8 / int4 / nvfp4,
 * fp8 per-tensor -> per-token-head, low-bit WHT pack). In-place forbidden; dst pages are new pages
 * handed out by the engine, which swaps page-table entries after the completion event.
 */
template <typename src_t, typename dst_t, KVCodec kSrcCodec, KVCodec kDstCodec>
__global__ void kv_requantize_pages_kernel(
    const PagedKVDesc src_desc,
    const PagedKVDesc dst_desc,
    const int64_t* __restrict__ page_pairs,     // [num_pairs, 2] (src page, dst page)
    const int num_pairs);

/**
 * kv_scale_update_kernel — track running per-head absmax of newly written K/V (attention.rs
 * KvScaleUpdate). New scales apply ONLY to pages written after the update epoch; pages written
 * under an older scale keep their own scale (stored per page or requantized) — never silently
 * reinterpreted.
 */
template <typename scalar_t>
__global__ void kv_scale_update_kernel(
    float* __restrict__ k_absmax,               // [Hkv] running max (in/out)
    float* __restrict__ v_absmax,               // [Hkv]
    const scalar_t* __restrict__ key,           // [num_tokens, Hkv, Dk]
    const scalar_t* __restrict__ value,         // [num_tokens, Hkv, Dv]
    const int num_tokens, const int num_kv_heads, const int head_dim_k, const int head_dim_v);

/// kv_relayout_pages_kernel — NHD <-> HND <-> legacy, or page-size change (e.g. 16 -> 64 for TMA).
template <typename cache_t>
__global__ void kv_relayout_pages_kernel(
    const PagedKVDesc src_desc,
    const PagedKVDesc dst_desc,
    const int32_t* __restrict__ src_seq_page_indptr,  // [B + 1] pages per sequence in src
    const int32_t* __restrict__ dst_seq_page_indptr,  // [B + 1]
    const int batch_size);

}  // namespace kops::kv
