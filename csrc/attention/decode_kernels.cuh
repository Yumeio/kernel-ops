#pragma once
// =============================================================================
// kernel-ops | csrc/attention/decode_kernels.cuh            Catalog: A03, A05, A06, A07, A08
// Paged decode (MHA/GQA/MQA), GQA-packed tensor-core decode for small q_len
// (MTP/speculative verification), split-KV and device task scheduling
// (HPC-Ops dynamic decode, generalized).
// Requires: common/*, kvcache/kv_layout.cuh, attention/attention_params.cuh.
// =============================================================================

namespace kops::attn {

/// One unit of decode work (A05). Mirrors HPC-Ops TaskScheduleInfo, extended for q_len > 1.
struct DecodeTask {
  int32_t request;        // b
  int32_t kv_head;        // kv head (all G query heads of the group are processed together)
  int32_t kv_begin;       // first token
  int32_t kv_end;         // one past last token
  int32_t partial_row;    // destination row in SplitKVBuffers (or -1 when the task covers all KV)
  int32_t split_index;    // index of this split within the request (sinks added only by split 0)
  int32_t num_splits;     // total splits of this (request, kv_head)
  int32_t flags;          // bit0: causal tail crosses the task (q_len > 1), bit1: last split
};

/// Arguments of decode kernels. One query token per request unless q_len > 1 (A06).
struct DecodeParams {
  const void* q;                     // [B * q_len, Hq, Dqk]
  int64_t q_stride_n, q_stride_h;
  int32_t q_len;                     // 1 (decode) | 2..8 (MTP / EAGLE chain verification, causal among drafts)
  kv::PagedKVDesc kv;
  AttentionModifiers mod;
  void* o;                           // [B * q_len, Hq, Dvo]
  int64_t o_stride_n, o_stride_h;
  float* lse;                        // [B * q_len, Hq] or null
  SplitKVBuffers split;
  WorkPlan plan;                     // static grid (request, kv chunk) items
  const DecodeTask* tasks;           // dynamic task list (A05) or null
  const int32_t* num_tasks;          // [1]
  int32_t batch_size, num_qo_heads, num_kv_heads;
};

/**
 * paged_decode_kernel — CUDA-core (SIMT) decode: each CTA owns (request, kv head[, kv chunk]),
 * loads K/V pages once and serves all G = Hq/Hkv query heads; online softmax in fp32.
 * Best for MHA / small G, FP8 KV with per-head scales, odd head dims (80, 96, 112).
 * Launch : grid = (B, Hkv, num_kv_chunks) or (plan.padded_num_items), block = 128..256.
 */
template <int HEAD_DIM, int GQA_GROUP, int PAGE_SIZE, int VEC_SIZE, int NUM_STAGES, typename DTypeQ,
          typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout>
__global__ void paged_decode_kernel(const __grid_constant__ DecodeParams params);

/**
 * paged_decode_tc_kernel — tensor-core decode: the G * q_len query rows of a kv head form the M
 * dimension of the QK/PV MMAs (GQA packing). Preferred for G >= 4 (Llama-3 8B/70B: 4/8, Qwen3: 4-16,
 * GPT-OSS: 8, Gemma 3), MQA, and multi-token verification (A06). Sm90a variant uses WGMMA with
 * M = 64 tiles; Sm100a uses tcgen05 + TMEM.
 */
template <int HEAD_DIM_QK, int HEAD_DIM_VO, int ROWS_PER_CTA, int CTA_TILE_KV, int NUM_STAGES,
          typename DTypeQ, typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout,
          typename ArchTag>
__global__ void paged_decode_tc_kernel(const __grid_constant__ DecodeParams params);

/**
 * paged_decode_dynamic_kernel — A05 persistent decode: CTAs pull DecodeTask items (balanced by KV
 * tile count, not by request), so 1 x 64K + 31 x 4K style skew does not idle SMs. Writes partials
 * for split tasks; merge_split_kv_kernel (or an in-kernel last-arriver combine guarded by an epoch
 * counter) produces the final output.
 * B8 (SM80 class): task_counter is uint32[2] = {next task, finished CTAs}, zero at plan time; the
 * last CTA to finish resets both, so the counter is replay-safe without epochs (`epoch` unused).
 * A task's math does not depend on which CTA runs it: the output is bit-reproducible.
 */
template <int HEAD_DIM_QK, int HEAD_DIM_VO, int GQA_GROUP, int CTA_TILE_KV, int NUM_STAGES,
          typename DTypeQ, typename DTypeKV, typename DTypeO, typename Variant, kv::KVLayout kLayout,
          typename ArchTag>
__global__ void paged_decode_dynamic_kernel(const __grid_constant__ DecodeParams params,
                                            uint32_t* __restrict__ task_counter,  // epoch-tagged
                                            const uint32_t epoch);

/**
 * assign_decode_tasks_kernel — A05 device planner (HPC-Ops assign_attention_decode_task, generalized):
 * from kv lengths, q_len, Hkv and a CTA budget (num_sms * ctas_per_sm) builds uniform-cost
 * DecodeTask buckets and merge_indptr. Runs inside the graph; capacity-bounded with overflow flag.
 * Cost of planning + combine is always reported together with attention time in benchmarks.
 * ABI 6: kv lengths come from kv_lens[b] when non-null, else from the CSR page table
 * (kv_indptr, kv_last_page_len, page_size); merge_indptr is per query row [B * q_len + 1] (the
 * merge_split_kv convention) and every task writes partial rows (partial_row = 0, split_index);
 * total_rows_out[0] = B * q_len; max_partial_rows bounds merge_indptr[B * q_len] (overflow).
 */
__global__ void assign_decode_tasks_kernel(
    DecodeTask* __restrict__ tasks,             // [max_tasks]
    int32_t* __restrict__ num_tasks,            // [1]
    int32_t* __restrict__ merge_indptr,         // [B * q_len + 1]
    int32_t* __restrict__ overflow_flag,        // [1]
    const int32_t* __restrict__ kv_lens,        // [B] or null
    const int batch_size, const int num_kv_heads, const int q_len,
    const int cta_budget, const int min_kv_tile, const int max_splits, const int max_tasks,
    const int32_t* __restrict__ kv_indptr, const int32_t* __restrict__ kv_last_page_len,
    const int page_size, const int max_partial_rows, uint32_t* __restrict__ total_rows_out);

}  // namespace kops::attn
