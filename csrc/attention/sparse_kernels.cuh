#pragma once
// =============================================================================
// kernel-ops | csrc/attention/sparse_kernels.cuh                Catalog: A10, A11, A13
// Sparse attention pipelines, with scoring, selection and attention as three
// separate APIs (mask generation cost is always visible):
//   * DSA (DeepSeek-V3.2, GLM-5-style): lightning indexer -> top-k tokens -> sparse MLA
//   * CSA / HCA (DeepSeek-V4): KV compressor -> (indexer over compressed keys -> top-k) ->
//     shared-KV MQA over selected compressed entries + sliding-window branch + sink
//   * Block selection (Quest / STEM OAM+TPD / MInference): block summaries -> scores -> top blocks
// Requires: common/*, kvcache/kv_layout.cuh, attention/attention_params.cuh, attention/prefill_kernels.cuh.
// =============================================================================

namespace kops::attn {

// -----------------------------------------------------------------------------
// A11 scoring + selection
// -----------------------------------------------------------------------------

/// Lightning indexer arguments: I[t, s] = sum_h w[t, h] * ReLU(q_I[t, h] . k_I[s]) (* k scale).
struct IndexerParams {
  const void* q;                  // [T, H_I, D_I] (H_I = 64, D_I = 128 for DSA) fp8 / fp4 / bf16
  int64_t q_stride_n, q_stride_h;
  const float* q_scale;           // per-token-head scales for quantized q, or null
  const float* head_weights;      // [T, H_I] w_t,h (already multiplied by softmax scale / H_I^-0.5)
  kv::PagedKVDesc k_cache;        // indexer key cache (kIndexerFP8 / kIndexerFP4), one "head"
  const int32_t* row_starts;      // [T] first visible key index for query t (prefill ragged)
  const int32_t* row_ends;        // [T] one past last visible key (causal: own position + 1)
  float* logits;                  // [T, max_kv] fp32 output (entries outside [start, end) = -inf)
  int64_t logits_stride;
  int32_t num_tokens, num_index_heads, index_dim, max_kv;
  int32_t compress_ratio;         // 1 (DSA: token level) | m (DeepSeek-V4 CSA: compressed blocks)
};

/**
 * lightning_indexer_kernel — DSA/CSA scoring. FP8 (DeepSeek-V3.2) or FP4 (DeepSeek-V4 QAT) QK MMA,
 * fp32 ReLU-weighted reduction across indexer heads. kPaged: decode against the paged index cache;
 * otherwise ragged prefill keys.
 */
template <typename DTypeQ, typename DTypeK, int INDEX_HEADS, int INDEX_DIM, bool kPaged, typename ArchTag>
__global__ void lightning_indexer_kernel(const KOPS_GRID_CONST IndexerParams params);

/**
 * sparse_topk_kernel — exact top-K indices per row of a variable-length fp32 score matrix
 * (K = 512 / 1024 / 2048; HPC-Ops filtered radix top-k). Output unordered; ties broken by lower
 * index; rows shorter than K are padded with -1. Indices are relative to row_starts or absolute
 * per `absolute_indices`; when to_page_slots is set they are converted to physical cache slots
 * through the block table (ready for sparse MLA).
 */
template <int TOPK>
__global__ void sparse_topk_kernel(
    int32_t* __restrict__ indices,              // [T, TOPK]
    float* __restrict__ values,                 // [T, TOPK] or null
    const float* __restrict__ scores,           // [T, max_kv]
    const int64_t scores_stride,
    const int32_t* __restrict__ row_starts,     // [T] or null
    const int32_t* __restrict__ row_ends,       // [T]
    const int32_t* __restrict__ block_table,    // [B, max_pages] for slot conversion or null
    const int32_t* __restrict__ token_to_request,// [T] or null
    const int block_table_stride, const int page_size,
    const bool absolute_indices, const bool to_page_slots, const int num_rows);

/**
 * kv_block_summary_kernel — per KV block (block_size tokens) summary for block selection:
 * mean / min / max of K (Quest), or STEM OAM "outlier-aware mean". Written once per filled block
 * and cached alongside the page.
 */
template <typename DTypeKV, int HEAD_DIM, int BLOCK_SIZE>
__global__ void kv_block_summary_kernel(
    void* __restrict__ summaries,               // [num_blocks_total, Hkv, num_stats, HEAD_DIM]
    const kv::PagedKVDesc kv,
    const int32_t* __restrict__ block_ids,      // [num_blocks_to_update] global block ids
    const int num_stats,                        // 1 (mean) | 2 (min, max)
    const int num_blocks_to_update);

/**
 * block_topk_select_kernel — score query blocks against KV block summaries and keep the top
 * blocks per (request, head) plus mandatory sink / local blocks; emits a BlockSparseMask (BSR).
 */
template <typename DTypeQ, int HEAD_DIM>
__global__ void block_topk_select_kernel(
    int32_t* __restrict__ block_row_indptr,     // BSR output
    int32_t* __restrict__ block_col_indices,
    const void* __restrict__ q,                 // [total_q, Hq, HEAD_DIM]
    const void* __restrict__ summaries,
    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_lens,
    const int q_block, const int kv_block, const int topk_blocks,
    const int num_sink_blocks, const int num_local_blocks,
    const int batch_size, const int num_q_heads, const int num_kv_heads);

// -----------------------------------------------------------------------------
// A13 sparse MLA (DSA)
// -----------------------------------------------------------------------------

struct SparseMLAParams {
  const void* q;                  // [T, H, 576] (512 absorbed nope | 64 rope)
  int64_t q_stride_n, q_stride_h;
  kv::PagedKVDesc kv;             // latent cache (kNative 576 bf16 | kMLAFp8DS 656 B)
  const int32_t* indices;         // [T, topk] physical token slots (-1 = invalid)
  int32_t topk;                   // 2048 (DeepSeek-V3.2)
  AttentionModifiers mod;         // includes sink_logits when the model has a sink
  void* o;                        // [T, H, 512]
  int64_t o_stride_n, o_stride_h;
  float* lse;                     // [T, H] or null
  int32_t num_tokens, num_heads;
};

/**
 * sparse_mla_kernel — each query token attends only to its `topk` selected latent entries.
 * Same kernel for decode and prefill (index-driven; no causal logic inside, the indexer applies it).
 */
template <int HEADS_PER_CTA, int TOPK_TILE, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename ArchTag>
__global__ void sparse_mla_kernel(const KOPS_GRID_CONST SparseMLAParams params);

// -----------------------------------------------------------------------------
// DeepSeek-V4 CSA / HCA
// -----------------------------------------------------------------------------

/// KV compressor (DeepSeek-V4 eq. 9-12): every m tokens -> one entry, built from 2m overlapping
/// candidates with softmax weights over (Z + positional bias) per channel.
struct KVCompressorParams {
  const void* ca;                 // [T, c]  C^a = H W^aKV
  const void* cb;                 // [T, c]  C^b = H W^bKV (previous window, overlapped)
  const void* za;                 // [T, c]  Z^a compression logits
  const void* zb;                 // [T, c]
  int64_t token_stride;
  const float* bias_a;            // [m, c] learnable positional bias B^a
  const float* bias_b;            // [m, c]
  int32_t ratio_m;                // compression ratio (CSA: m; HCA: m' >> m, e.g. 128)
  int32_t entry_dim;              // c (head dim of the compressed entry)
  // Post-processing fused before the store (model contract)
  const void* norm_weight;        // [c] RMSNorm on compressed KV entries (V4 KV-entry normalization) or null
  float norm_eps;
  const int64_t* positions;       // [T] absolute positions (rope position of an entry = its last token)
  const void* cos_sin_cache;      // partial RoPE on the rope slice of the entry or null
  int32_t rot_dim, rot_offset;
  // Ragged sequence bookkeeping: tokens of request b are [cu_seqlens[b], cu_seqlens[b+1]),
  // the first token sits at absolute position seq_start_pos[b].
  const int32_t* cu_seqlens;      // [B + 1]
  const int32_t* seq_start_pos;   // [B]
  // Tail state: tokens not yet forming a full window are kept per sequence (engine slot).
  void* tail_state;               // [num_slots, 2*m, 2c (+ 2c logits)] uncompressed carry
  const int32_t* state_slots;     // [B]
  // Output: compressed entries appended to a paged cache (one entry per completed window).
  kv::PagedKVDesc out_cache;
  const int64_t* out_slot_mapping;// [max_entries] slot of each produced entry (prepared by engine)
  int32_t* num_entries_out;       // [B] entries produced per sequence in this call
  int32_t batch_size;
};

/**
 * kv_compressor_kernel — produces compressed KV entries (also used for indexer keys K^IComp with
 * entry_dim = c^I). Windows straddling the chunk boundary are completed from tail_state, so
 * prefill chunks and decode steps produce identical entries (serial/chunk parity test).
 */
template <typename DTypeIn, typename DTypeCache, int RATIO_M, int ENTRY_DIM, bool kOverlap>
__global__ void kv_compressor_kernel(const KOPS_GRID_CONST KVCompressorParams params);

/// Shared-KV MQA over (a) indexed compressed entries and/or (b) all compressed entries and
/// (c) a sliding-window branch of raw recent tokens, with an attention sink, in ONE softmax.
struct CompressedAttentionParams {
  const void* q;                  // [T, Hq, c] (RMS-normalized queries, partial RoPE applied)
  int64_t q_stride_n, q_stride_h;
  kv::PagedKVDesc compressed;     // compressed entries (shared K = V, v_data null)
  const int32_t* selected;        // [T, topk] CSA selected compressed indices (-1 pad); null = HCA dense
  int32_t topk;
  const int32_t* num_compressed;  // [B] visible compressed entries per request (HCA dense range)
  kv::PagedKVDesc window;         // sliding-window raw KV (ring buffer or pages)
  int32_t window_size;
  const int32_t* token_to_request;// [T]
  const int64_t* positions;       // [T]
  AttentionModifiers mod;         // sm_scale, sink_logits
  void* o;                        // [T, Hq, c]
  int64_t o_stride_n, o_stride_h;
  float* lse;                     // [T, Hq] or null
  int32_t num_tokens, num_q_heads, entry_dim, value_dim;  // value_dim <= entry_dim (shared KV slice)
};

/**
 * compressed_attention_kernel — DeepSeek-V4 CSA (selected != null) and HCA (selected == null)
 * attention including the SWA branch and sink in a single online softmax. Grouped output
 * projection is a separate gemm::bmm over head groups.
 */
template <int HEADS_PER_CTA, int ENTRY_DIM, int VALUE_DIM, typename DTypeQ, typename DTypeKV,
          typename DTypeO, typename ArchTag>
__global__ void compressed_attention_kernel(const KOPS_GRID_CONST CompressedAttentionParams params);

}  // namespace kops::attn
