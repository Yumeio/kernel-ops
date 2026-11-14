#pragma once
// =============================================================================
// kernel-ops | csrc/attention/prefill_kernels.cuh       Catalog: A01, A02, A06, A07, A08, A10
// Tiled QK/PV + online-softmax attention for prefill, chunked prefill with
// prefix reuse, multi-token verification and block-sparse prefill.
// Requires: common/*, kvcache/kv_layout.cuh, attention/attention_params.cuh.
// =============================================================================

namespace kops::attn {

/// Compile-time tile configuration. Each ArchTag maps to a different mainloop:
///   Sm80/Sm86/Sm89 : cp.async multi-stage + mma.sync (FA2 style)
///   Sm90a          : TMA + WGMMA, producer/consumer warp specialization, ping-pong (FA3 style)
///   Sm100a/Sm103a  : TMA + tcgen05 MMA into TMEM, 2-CTA (FA4 / cuDNN style)
///   Sm120a         : mma.sync (+ FP8 block-scaled) without TMEM; separate tuning from Sm100a
template <int HEAD_DIM_QK, int HEAD_DIM_VO, int CTA_TILE_Q, int CTA_TILE_KV, int NUM_WARPS_Q,
          int NUM_WARPS_KV, int NUM_STAGES, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename ArchTag>
struct PrefillTraits {
  static constexpr int head_dim_qk = HEAD_DIM_QK;   // 64, 80, 96, 128, 192, 256 (+ 512 for MLA-style)
  static constexpr int head_dim_vo = HEAD_DIM_VO;
  static constexpr int cta_tile_q = CTA_TILE_Q;     // 16 (decode-like), 64, 128
  static constexpr int cta_tile_kv = CTA_TILE_KV;   // 32..176
  static constexpr int num_warps_q = NUM_WARPS_Q;
  static constexpr int num_warps_kv = NUM_WARPS_KV;
  static constexpr int num_stages = NUM_STAGES;
  using q_type = DTypeQ;
  using kv_type = DTypeKV;
  using o_type = DTypeO;
  using arch = ArchTag;
};

/// Arguments of ragged (non-paged) prefill.
struct RaggedPrefillParams {
  const void* q;                     // [total_q, Hq, Dqk]
  int64_t q_stride_n, q_stride_h;
  const int32_t* qo_indptr;          // [B + 1]
  kv::RaggedKVDesc kv;               // K/V [total_kv, Hkv, D*] with kv_indptr
  const int32_t* q_pos_offset;       // [B] optional explicit position of query row 0 (else bottom-right)
  const uint8_t* custom_mask;        // kCustom: packed bits
  const int64_t* mask_indptr;        // [B + 1] bit offsets per request
  AttentionModifiers mod;
  void* o;                           // [total_q, Hq, Dvo]
  int64_t o_stride_n, o_stride_h;
  float* lse;                        // [total_q, Hq] or null
  SplitKVBuffers split;              // used when plan.split_kv (filled by ragged_prefill_run)
  WorkPlan plan;                     // filled by ragged_prefill_run from the AttentionPlan
  int32_t batch_size, num_qo_heads, num_kv_heads;
  int32_t total_q;
  MaskMode mask;                     // ABI 6: runtime mask mode (ragged_prefill_run sets it from the plan)
};

/**
 * ragged_prefill_kernel — A01 dense/ragged prefill (no cache read): QK^T tiles, online softmax
 * in fp32 (base-2 exp internally), PV accumulation, optional LSE / split partials.
 * Supports Dqk != Dvo (MLA prefill: q/k 192 = 128 nope + 64 rope, v 128), GQA/MQA via head mapping.
 * Models : every model's prompt phase without prefix cache; vision/audio encoders (kNone mask).
 * kMask names the compile-time mask specialization; the sentinel kMaskRuntime (prefill.cu) reads the
 * runtime field params.mask instead.  Variant flags compile a modifier in, its runtime value in
 * params.mod switches it on (CONTRACTS §1e).
 * Tile: Traits::cta_tile_q rows per CTA (128 on the narrow head dims, 64 on 192/256), one warp per
 * 16 rows, two K/V stages of Traits::cta_tile_kv keys (64, or 128 on SM90+ where a CTA may opt into
 * 227 KiB of dynamic shared memory).  The tile policy is part of the plan (plan.kernel_id), so the
 * grid a plan was built with always matches the kernel run().
 */
template <typename Traits, typename Variant, MaskMode kMask>
__global__ void ragged_prefill_kernel(const __grid_constant__ RaggedPrefillParams params);

/// Arguments of paged prefill (chunked prefill, prefix reuse, verification).
struct PagedPrefillParams {
  const void* q;                     // [total_q, Hq, Dqk]
  int64_t q_stride_n, q_stride_h;
  const int32_t* qo_indptr;          // [B + 1]
  kv::PagedKVDesc kv;                // cached + newly appended KV (append happens before this kernel)
  const int32_t* q_pos_offset;       // [B] or null (bottom-right default)
  const uint8_t* custom_mask;        // tree / speculative verification masks, packed
  const int64_t* mask_indptr;        // [B + 1]
  AttentionModifiers mod;
  void* o;
  int64_t o_stride_n, o_stride_h;
  float* lse;
  SplitKVBuffers split;              // filled by paged_prefill_run when plan.split_kv
  WorkPlan plan;                     // filled by paged_prefill_run from the AttentionPlan
  int32_t batch_size, num_qo_heads, num_kv_heads;
  int32_t total_q;
  MaskMode mask;                     // ABI 6: runtime mask mode (paged_prefill_run sets it from the plan)
};

/**
 * paged_prefill_kernel — A02 chunked prefill over paged KV with prefix reuse (cached prefix pages +
 * this chunk), A06 multi-token verification (q_len = draft_len + 1, causal or tree custom mask,
 * MTP / EAGLE / DSpark-style drafts), A08 FP8 KV dequant on load (kFp8Kv) or FP8 tensor-core
 * compute (separate Traits with DTypeQ = f8e4m3_t).
 * Page tables: CSR or block table (PagedKVDesc::table_format), NHD or HND.
 * kLayout = kNHD names the token-major strided loader, which also reads HND pages (the strides
 * carry the difference); kMask is the mask specialization (see ragged_prefill_kernel).
 */
template <typename Traits, typename Variant, kv::KVLayout kLayout, MaskMode kMask>
__global__ void paged_prefill_kernel(const __grid_constant__ PagedPrefillParams params);

/// Block-sparse mask in BSR form over (R x C) blocks of the [qo_len, kv_len] score matrix.
struct BlockSparseMask {
  const int32_t* block_row_indptr;   // [B * num_block_rows + 1] (per request, per head if per_head)
  const int32_t* block_col_indices;  // [nnz_blocks] kv block index
  int32_t block_r, block_c;          // e.g. 64 x 64 or HPC-Ops 128 x 128
  bool per_head;                     // mask differs per q head (MInference / STEM style)
  int32_t num_heads;
};

/**
 * block_sparse_prefill_kernel — A10: only (q-block, kv-block) pairs present in the BSR mask are
 * computed; inside a block the usual causal/window rules still apply. Full mask must equal the
 * dense result (golden test). Mask generation cost (A11) is accounted separately.
 */
template <typename Traits, typename Variant, kv::KVLayout kLayout>
__global__ void block_sparse_prefill_kernel(const __grid_constant__ PagedPrefillParams params,
                                            const BlockSparseMask mask);

/**
 * prefill_plan_kernel — device-side planner (graph friendly): from qo_indptr / kv lengths builds
 * WorkPlan items (request, q tile, kv chunk) and merge_indptr without host sync, bounded by
 * padded_num_items (overflow flag set if exceeded -> caller replans outside the graph).
 * One CTA; item i keeps the host planner's order (request, q tile, kv chunk), so a graph replay
 * plans exactly what prefill_plan would plan for the same lengths and chunk size.
 * ABI 6: kv lengths come from kv_lens[b] when non-null (block-table seq_lens), else from
 * kv_indptr (ragged token indptr when kv_last_page_len is null, CSR page indptr otherwise, with
 * page_size); kv_chunk_size is a device scalar (the plan's slot; null or <= 0 means one
 * chunk per request, so the value is replay-stable); padded items beyond the plan get
 * block_valid_mask = 0; total_rows_out[0] = qo_indptr[B] (row bound for merge_split_kv);
 * max_partial_rows bounds merge_indptr[total_q] (overflow otherwise).
 */
__global__ void prefill_plan_kernel(
    int32_t* __restrict__ request_indices, int32_t* __restrict__ qo_tile_indices,
    int32_t* __restrict__ kv_tile_indices, int32_t* __restrict__ merge_indptr,
    int32_t* __restrict__ o_indptr, uint8_t* __restrict__ block_valid_mask,
    int32_t* __restrict__ num_items_out, int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ qo_indptr, const int32_t* __restrict__ kv_lens,
    const int batch_size, const int cta_tile_q, const int32_t* __restrict__ kv_chunk_size,
    const int gqa_group_size, const int padded_num_items,
    const int32_t* __restrict__ kv_indptr, const int32_t* __restrict__ kv_last_page_len,
    const int page_size, const int max_partial_rows, uint32_t* __restrict__ total_rows_out);

}  // namespace kops::attn
