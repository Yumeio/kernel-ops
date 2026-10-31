#pragma once
// =============================================================================
// kernel-ops | csrc/attention/ops.h                              Catalog: A01-A14
// Plan / run API. plan() is host work done outside graph capture (or on device via
// the *_plan_kernel planners); run() only enqueues kernels on ctx.stream with the
// caller's workspace. A plan never owns requests, page allocation or prefix trees.
// Requires: common/*, kvcache/kv_layout.cuh, attention/*.cuh.
// =============================================================================

namespace kops::attn {

/// Opaque prepared execution for one AttentionSpec at a given capacity.
struct AttentionPlan {
  uint64_t kernel_id = 0;            // registry kernel identity
  uint32_t plan_generation = 0;      // must equal ctx.capture_generation at run() under graphs
  ExecutionStrategy strategy = ExecutionStrategy::kModular;
  bool split_kv = false;
  int32_t max_batch = 0, max_total_q = 0, max_kv_len = 0;  // capacity bounds
  int32_t num_launches = 0;          // exact device launches per run (incl. merge / planning)
  WorkspaceRequirements workspace;
  WorkPlan device_plan;              // pointers into the caller's persistent workspace
};

/// Why a candidate was rejected (explain()).
struct RejectReason {
  uint64_t kernel_id;
  Status code;
  const char* detail;                // static string, e.g. "head_dim_qk=96 not compiled for sm_100a"
};

/// Capability check without side effects; fills up to max_reasons rejected candidates.
Status attention_supports(const AttentionSpec& spec, const LaunchContext& ctx,
                          RejectReason* reasons, int32_t max_reasons, int32_t* num_reasons);

// ---- A01/A02/A06/A10 prefill ------------------------------------------------------
Status prefill_plan(const AttentionSpec& spec, const Tensor& qo_indptr_host,
                    const Tensor& kv_lens_host, int32_t max_batch, const LaunchContext& ctx,
                    const Workspace& ws, AttentionPlan* plan);
Status ragged_prefill_run(const AttentionPlan& plan, const RaggedPrefillParams& params,
                          const Workspace& ws, const LaunchContext& ctx);
Status paged_prefill_run(const AttentionPlan& plan, const PagedPrefillParams& params,
                         const Workspace& ws, const LaunchContext& ctx);
Status block_sparse_prefill_run(const AttentionPlan& plan, const PagedPrefillParams& params,
                                const BlockSparseMask& mask, const Workspace& ws,
                                const LaunchContext& ctx);

// ---- A03/A05/A06 decode -------------------------------------------------------------
/// Host planning (split counts from kv-length histogram) or device planning when
/// use_device_planner = true (assign_decode_tasks_kernel inside the graph).
Status decode_plan(const AttentionSpec& spec, int32_t max_batch, int32_t max_kv_len, int32_t q_len,
                   bool use_device_planner, const LaunchContext& ctx, const Workspace& ws,
                   AttentionPlan* plan);
Status decode_run(const AttentionPlan& plan, const DecodeParams& params, const Workspace& ws,
                  const LaunchContext& ctx);

// ---- A04/A14 merges --------------------------------------------------------------------
Status merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                      const Tensor& sink_logits, const LaunchContext& ctx);
Status merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                   const Tensor& o_b, const Tensor& lse_b, const LaunchContext& ctx);
Status merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                    const Tensor& lse_states, const LaunchContext& ctx);

/// Host arrays of device pointers for branches that scored different key sets
/// under one softmax (DeepSeek-V4: compressed + sliding-window + sink). The
/// launcher copies the pointers into the kernel parameter (HybridBranchTable),
/// so these arrays do not need to be pinned; the branch buffers themselves must
/// stay at stable device addresses for CUDA-graph capture/replay.
struct HybridBranches {
  const float* const* o = nullptr;    // [num_branches] device pointers to [rows, Hq, D] fp32
  const float* const* lse = nullptr;  // [num_branches] device pointers to [rows, Hq] fp32
  int32_t num_branches = 0;
};
Status merge_hybrid_branches(const Tensor& o, const Tensor& lse, const HybridBranches& branches,
                             const Tensor& sink_logits, const LaunchContext& ctx);

// ---- A12/A13 MLA ---------------------------------------------------------------------
Status mla_decode_plan(const AttentionSpec& spec, int32_t max_batch, int32_t q_len,
                       const LaunchContext& ctx, const Workspace& ws, AttentionPlan* plan);
Status mla_decode_run(const AttentionPlan& plan, const MLADecodeParams& params,
                      const Workspace& ws, const LaunchContext& ctx);
Status mla_context_gather(const Tensor& c_kv_out, const Tensor& k_pe_out,
                          const kv::PagedKVDesc& cache, const Tensor& ctx_token_indptr,
                          const Tensor& ctx_start, const LaunchContext& ctx);
Status sparse_mla_run(const SparseMLAParams& params, const LaunchContext& ctx);
Status concat_mla_q(const Tensor& q_out, const Tensor& ql_nope, const Tensor& q_pe,
                    const LaunchContext& ctx);

// ---- A11 sparse scoring / selection, DeepSeek-V4 CSA/HCA ----------------------------------
Status lightning_indexer(const IndexerParams& params, DType q_dtype, const LaunchContext& ctx);
Status sparse_topk(const Tensor& indices, const Tensor& values, const Tensor& scores,
                   const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                   const Tensor& block_table, const Tensor& token_to_request, int32_t page_size,
                   bool absolute_indices, bool to_page_slots, const LaunchContext& ctx);
Status block_select(const Tensor& block_row_indptr, const Tensor& block_col_indices,
                    const Tensor& q, const Tensor& summaries, const Tensor& qo_indptr,
                    const Tensor& kv_lens, int32_t q_block, int32_t kv_block, int32_t topk_blocks,
                    int32_t num_sink_blocks, int32_t num_local_blocks, const LaunchContext& ctx);
Status kv_compress(const KVCompressorParams& params, DType in_dtype, const LaunchContext& ctx);
Status compressed_attention(const CompressedAttentionParams& params, const LaunchContext& ctx);

}  // namespace kops::attn
