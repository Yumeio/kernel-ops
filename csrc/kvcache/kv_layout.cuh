#pragma once
// =============================================================================
// kernel-ops | csrc/kvcache/kv_layout.cuh                    Catalog: K01-K03, A02/A03
// Paged KV descriptors shared by kvcache/ and attention/. The engine owns page
// allocation, ref-counts, COW and prefix identity; kernels only read/write the
// pages a descriptor points to.
// Requires: common/macros.cuh, common/dtype.cuh, common/quant.cuh.
// =============================================================================

namespace kops::kv {

/// Physical order inside one page ("block" in vLLM terminology).
enum class KVLayout : uint8_t {
  kNHD = 0,          // [page][token][head][dim]   (FlashInfer default, vLLM flash backend)
  kHND = 1,          // [page][head][token][dim]   (TRT-LLM, FlashInfer HND, per-head TMA tiles)
  kVllmLegacy = 2,   // vLLM paged-attention v1/v2 layout:
                     //   K: [num_blocks, num_heads, head_size / x, block_size, x],  x = 16 / sizeof(cache_t)
                     //   V: [num_blocks, num_heads, head_size, block_size]
  kLatent = 3,       // single latent cache, one record per token: [page][token][entry] (MLA, DSA indexer,
                     //   DeepSeek-V4 compressed entries); `entry_bytes` gives the record size
};

/// How a request maps to its pages.
enum class PageTableFormat : uint8_t {
  kCSR = 0,         // kv_indptr[B+1], kv_page_indices[nnz], kv_last_page_len[B]  (FlashInfer)
  kBlockTable = 1,  // block_table[B, max_pages], seq_lens[B]                     (vLLM / FlashMLA)
  kSlotMapping = 2, // per-token physical slot = page * page_size + offset; -1 = padding (writes only)
};

/// Storage codec of a KV entry (K and V carry independent codecs).
enum class KVCodec : uint8_t {
  kNative = 0,            // same dtype as compute (bf16 / fp16 / fp32)
  kFP8PerTensor = 1,      // e4m3 or e5m2 (PagedKVDesc::k_dtype) + one scale (k_scale / v_scale)
  kFP8PerHead = 2,        // e4m3 + scale per kv head (k_scale[Hkv], scale_head_stride = 1)
  kFP8PerTokenHead = 3,   // e4m3 + scale per (token, head) stored in a side array
  kInt8PerTokenHead = 4,
  kInt4PerTokenHead = 5,  // packed int4 + (scale, zero) per (token, head)
  kNVFP4 = 6,             // e2m1 + E4M3 per 16 + global (Blackwell KV FP4)
  kMLAFp8DS = 7,          // DeepSeek "ds_mla" FP8 latent record, 656 bytes per token:
                          //   [0, 512)   512 x e4m3 NoPE values in 4 tiles of 128
                          //   [512, 528) 4 x fp32 tile scales, scale = 2^ceil(log2(max(amax / 448, 1e-4)))
                          //              (power of two: SM90 and SM100 readers rescale exactly)
                          //   [528, 656) 64 x bf16 RoPE values (zeroed for NoPE rows, pe_dim = 0)
  kMLANvfp4DS = 8,        // NVFP4 "ds_mla" latent record, 352 bytes per token (e2m1 NoPE + block scales +
                          //   RoPE); Sm100a-only provider format, read/written only by its own kernels
  kLowBitWHT = 9,         // EXPERIMENTAL A09: WHT-rotated uniform 3/4-bit (+ fp32 absmax), K8V4/K4V4/K3V4
  kIndexerFP8 = 10,       // DSA (DeepSeek-V3.2) / CSA (DeepSeek-V4) indexer keys, SPLIT page layout:
                          //   page = [page_size x head_dim e4m3 | page_size x (head_dim / quant_block) fp32 scales]
                          //   (scales follow all keys of the page, they are NOT interleaved per token)
  kIndexerFP4 = 11,       // DeepSeek-V4 FP4 indexer QK path (same split-page idea, e2m1 + block scales)
};

/// Paged KV cache descriptor (one attention layer).
struct PagedKVDesc {
  void* k_data = nullptr;          // K pages (or the single latent/shared-KV cache)
  void* v_data = nullptr;          // V pages; null for latent caches and shared-KV MQA (DeepSeek-V4 CSA/HCA)
  DType k_dtype = DType::kBFloat16;
  DType v_dtype = DType::kBFloat16;
  KVCodec k_codec = KVCodec::kNative;
  KVCodec v_codec = KVCodec::kNative;
  KVLayout layout = KVLayout::kNHD;
  int32_t page_size = 16;          // tokens per page (1, 16, 32, 64, 128, 256)
  int32_t num_kv_heads = 0;        // local kv heads (1 for MLA / MQA)
  int32_t head_dim_k = 0;          // e.g. 128; MLA: 576 (512 latent + 64 rope)
  int32_t head_dim_v = 0;          // e.g. 128; MLA: 512 (reads first 512 of the same entry)
  int32_t entry_bytes = 0;         // bytes per (token, head) record incl. inline scales; 0 = dense dtype
  int32_t legacy_x = 0;            // kVllmLegacy: elements per 16-byte K vector (8 for bf16, 16 for fp8)
  int64_t k_stride_page = 0;       // elements (or bytes when entry_bytes != 0) between pages ("block_stride")
  int64_t k_stride_token = 0;      // "page_stride" in vLLM flash terms (between tokens of a page)
  int64_t k_stride_head = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_token = 0;
  int64_t v_stride_head = 0;
  PageTableFormat table_format = PageTableFormat::kCSR;
  const int32_t* kv_indptr = nullptr;         // [B + 1]
  const int32_t* kv_page_indices = nullptr;   // [nnz_pages]
  const int32_t* kv_last_page_len = nullptr;  // [B], in [1, page_size]
  const int32_t* block_table = nullptr;       // [B, block_table_stride]
  int32_t block_table_stride = 0;
  const int32_t* seq_lens = nullptr;          // [B] total KV length (incl. tokens appended this step)
  // Scales for non-native codecs (null when codec is native or scales are inline).
  const float* k_scale = nullptr;  // [1] | [Hkv] (device scalars keep graphs replayable)
  const float* v_scale = nullptr;
  int32_t scale_head_stride = 0;   // 0: k_scale/v_scale are per-tensor [1]; 1: per-head [Hkv]
  void* k_token_scales = nullptr;  // [P, PS, Hkv] (+ zeros) for per-token-head codecs
  void* v_token_scales = nullptr;
  int64_t token_scale_stride_page = 0;
};

/// Non-paged (ragged, contiguous) KV for prefill without cache / cross-attention.
struct RaggedKVDesc {
  const void* k = nullptr;         // [total_kv, Hkv, Dk]
  const void* v = nullptr;         // [total_kv, Hkv, Dv]
  int64_t k_stride_n = 0, k_stride_h = 0;
  int64_t v_stride_n = 0, v_stride_h = 0;
  const int32_t* kv_indptr = nullptr;  // [B + 1]
};

/// Ragged per-request gather of cached tokens (chunked-prefill context, MLA up-projection,
/// DSA indexer prefill, KV export). Output token o of request r is written at
/// cu_seq_lens[r] + o and read from cache position seq_starts[r] + o of that request.
struct GatherSpec {
  const int32_t* block_table = nullptr;   // [num_reqs, block_table_stride] physical page ids
  int64_t block_table_stride = 0;
  const int32_t* cu_seq_lens = nullptr;   // [num_reqs + 1] output offsets (device-exact)
  const int32_t* seq_starts = nullptr;    // [num_reqs] first cached position to gather per request, or null (0)
  int32_t num_reqs = 0;
  int32_t num_tokens = 0;                 // output rows; may be an allocation upper bound (no D2H sync):
                                          //   only rows covered by cu_seq_lens are written
};

/**
 * One page-sized unit of gather work (vLLM GatherPageTask). Gather kernels launch
 * grid = ceil(num_tokens / page_size) + 2 * num_reqs CTAs — an upper bound on touched pages since
 * each request may start and end inside a page — and map blockIdx.x to a task by walking the
 * requests; CTAs without a task exit immediately.
 */
struct GatherPageTask {
  int32_t req_id;
  int32_t logical_block;        // page index inside the request's block table row
  int32_t page_token_begin;     // first token inside the page to copy
  int32_t page_token_end;       // one past last token inside the page
  int32_t output_token_begin;   // destination row of page_token_begin
};

/// Physical bytes for `num_pages` pages under this descriptor, including inline / side scales
/// and alignment padding. Source of truth for KV memory ledgers (not dtype bits).
KOPS_HOST_DEVICE int64_t kv_physical_bytes(const PagedKVDesc& desc, int64_t num_pages);

}  // namespace kops::kv
