#pragma once
// =============================================================================
// kernel-ops | csrc/attention/mla_kernels.cuh                        Catalog: A12
// Multi-head Latent Attention (DeepSeek V2/V3/V3.1, Kimi K2/K2.5, Kimi Linear
// and K3 gated-MLA layers, MiniCPM3). Latent cache entry = [c_kv (512) | k_rope (64)].
//   decode (absorbed): q_abs = q_nope . W_UK   -> scores = q_abs.c_kv + q_pe.k_pe
//                      o_lat = softmax . c_kv  -> o = o_lat . W_UV  (W_UK/W_UV via gemm::bmm)
//   prefill (expanded): k = [c_kv . W_UK | k_pe], v = c_kv . W_UV -> ragged prefill (Dqk 192, Dvo 128)
// Requires: common/*, kvcache/kv_layout.cuh, attention/attention_params.cuh.
// =============================================================================

namespace kops::attn {

/// Arguments of absorbed MLA decode / verification.
struct MLADecodeParams {
  const void* q_nope;              // [B * q_len, H, kv_lora_rank]  (absorbed: 512)
  int64_t q_nope_stride_n, q_nope_stride_h;
  const void* q_pe;                // [B * q_len, H, rope_dim]      (64)
  int64_t q_pe_stride_n, q_pe_stride_h;
  int32_t q_len;                   // 1 | 2 (DeepSeek MTP) | up to 8
  kv::PagedKVDesc kv;              // single latent cache (v_data null); codec kNative | kMLAFp8DS | kFP8PerTensor
  AttentionModifiers mod;          // sm_scale = mscale^2 / sqrt(qk_nope + qk_rope) (192 for DeepSeek)
  void* o;                         // [B * q_len, H, kv_lora_rank]
  int64_t o_stride_n, o_stride_h;
  float* lse;                      // [B * q_len, H] or null
  // Kimi K3 gated MLA: o *= sigmoid(gate) when gate is non-null. Gate has the same shape as o.
  const void* output_gate = nullptr;
  int64_t gate_stride_n = 0, gate_stride_h = 0;
  DType gate_dtype = DType::kBFloat16;
  SplitKVBuffers split;
  const int32_t* tile_scheduler_metadata;  // [num_sm_parts, 8] from mla_get_metadata_kernel (FlashMLA style)
  const int32_t* num_splits;       // [B + 1]
  int32_t batch_size, num_heads, kv_lora_rank, rope_dim;
};

/**
 * mla_decode_kernel — absorbed MLA over the paged latent cache. Heads of one request share the
 * latent KV, so H query heads form the M dimension (MQA-like, 128 heads for DeepSeek-V3 / 64 for
 * Kimi K2 / 16-32 per rank under TP). Variants:
 *   Sm90a: FlashMLA-style WGMMA, 64-head tiles, BF16 or FP8 (kMLAFp8DS) cache
 *   Sm100a: tcgen05 2-CTA (CUTLASS MLA / TRT-LLM gen style)
 *   Sm80: mma.sync fallback
 */
template <int KV_LORA_RANK, int ROPE_DIM, int HEADS_PER_CTA, int PAGE_SIZE, typename DTypeQ,
          typename DTypeKV, typename DTypeO, typename ArchTag>
__global__ void mla_decode_kernel(const __grid_constant__ MLADecodeParams params);

/**
 * mla_get_metadata_kernel — split the (request, kv) work of all requests into num_sm_parts balanced
 * partitions (FlashMLA get_mla_metadata), writing tile_scheduler_metadata and num_splits. Device-side
 * so it can live inside the decode graph.
 */
__global__ void mla_get_metadata_kernel(
    int32_t* __restrict__ tile_scheduler_metadata,  // [num_sm_parts, 8]
    int32_t* __restrict__ num_splits,               // [B + 1]
    const int32_t* __restrict__ seq_lens,           // [B]
    const int batch_size, const int q_len_times_heads_per_group, const int num_sm_parts,
    const int block_size_n, const int fixed_overhead_num_blocks);

/**
 * mla_context_gather_kernel — chunked prefill with a cached prefix: gather (and dequantize) the
 * latent entries of context tokens into contiguous [n, 512] c_kv and [n, 64] k_pe so the caller can
 * up-project with W_UK/W_UV (gemm) and run ragged_prefill; workspace bounded by chunk size.
 */
template <typename cache_t, typename out_t, kv::KVCodec kCodec>
__global__ void mla_context_gather_kernel(
    out_t* __restrict__ c_kv_out,                // [num_ctx_tokens, kv_lora_rank]
    out_t* __restrict__ k_pe_out,                // [num_ctx_tokens, rope_dim]
    const cache_t* __restrict__ kv_cache,        // paged latent cache
    const int32_t* __restrict__ block_table,     // [B, max_pages]
    const int32_t* __restrict__ ctx_token_indptr,// [B + 1] context tokens gathered per request
    const int32_t* __restrict__ ctx_start,       // [B] first context position of this chunk
    const float* __restrict__ scale,
    const kv::PagedKVDesc desc, const int batch_size);

/**
 * concat_mla_q_kernel — q_out[t, h] = [ ql_nope[t, h, 0:NOPE] | q_pe[t, h, 0:64] ] for MLA / DSA
 * kernels that need one contiguous 576-wide (or 512k + 64) query (replaces torch.cat on the hot
 * path). NOPE is a multiple of 512 (absorbed latent); last dims must be contiguous.
 * Launch : one warp per (token, head); block = 8 warps, grid = ceil(T * H / 8).
 */
template <typename scalar_t, int NOPE_CHUNK>
__global__ void concat_mla_q_kernel(
    scalar_t* __restrict__ q_out,                // [num_tokens, num_heads, nope_dim + 64]
    const scalar_t* __restrict__ ql_nope,        // [num_tokens, num_heads, nope_dim]
    const scalar_t* __restrict__ q_pe,           // [num_tokens, num_heads, 64]
    const int num_tokens, const int num_heads, const int nope_dim,
    const int64_t out_stride_token, const int64_t out_stride_head,
    const int64_t nope_stride_token, const int64_t nope_stride_head,
    const int64_t pe_stride_token, const int64_t pe_stride_head);

}  // namespace kops::attn
