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

/// Host-only split-KV plan for one dense MLA decode shape: a uniform split count over every
/// request plus the sizes of the three partial buffers mla_decode_run() then consumes.
///
/// The count scales the natural grid — q_len * ceil(H / 16) CTAs per request, one per 16-head
/// tile — to the `waves * ctx.compute_units` CTA budget (occupancy.h). It is the largest count
/// whose grid still fits that budget, not the closest rounding up: this body runs two CTAs per SM
/// (128 threads, 92 KiB smem), so waves = 2 is exactly one resident grid, and a count one step
/// above it adds a partial second residency whose tail runs at a fraction of the bandwidth
/// (measured 1.3-1.4x slower for one extra split at 1M tokens). max_splits and at least 512
/// visible keys per split bound it from above.
///
/// `kv_len` is a host-known length bound (a capacity, the benchmark's fixed context length, a max
/// over the batch the caller already computed); the function never reads seq_lens, num_splits or
/// any other device array, so it stays valid across CUDA-graph capture and replay. splits == 1
/// means the plan needs no partial buffers at all.
///
/// The caller owns the buffers and fills num_splits[b] = b * splits and
/// merge_indptr[row] = row * splits (FlashMLA cumulative convention) before mla_decode_run;
/// merge_split_kv then folds them (and the sink, exactly once) into o/lse.
struct MLASplitHint {
  int32_t splits = 1;             // uniform split count (1 = no split)
  int64_t total_q = 0;            // batch_size * q_len partial rows
  size_t o_partial_bytes = 0;     // [total_q * splits, H, kv_lora_rank] fp32
  size_t lse_partial_bytes = 0;   // [total_q * splits, H] fp32
  size_t merge_indptr_bytes = 0;  // [total_q + 1] int32
  size_t total_bytes = 0;         // 256-aligned sum of the three buffers
};
Status mla_decode_split_hint(int32_t batch_size, int32_t q_len, int32_t num_heads,
                             int32_t kv_len, int32_t max_splits, int32_t waves,
                             const LaunchContext& ctx, MLASplitHint* out);
/// Device-side split planner (FlashMLA get_decoding_sched_meta semantics). seq_lens is [B];
/// tile_scheduler_metadata is [num_sm_parts, 8], num_splits is [B + 1]. Graph-capturable.
Status mla_get_metadata(const Tensor& tile_scheduler_metadata, const Tensor& num_splits,
                        const Tensor& seq_lens, int32_t q_len_times_heads_per_group,
                        int32_t num_sm_parts, int32_t block_size_n,
                        int32_t fixed_overhead_num_blocks, const LaunchContext& ctx);
Status mla_context_gather(const Tensor& c_kv_out, const Tensor& k_pe_out,
                          const kv::PagedKVDesc& cache, const Tensor& ctx_token_indptr,
                          const Tensor& ctx_start, const LaunchContext& ctx);
Status sparse_mla_run(const SparseMLAParams& params, const LaunchContext& ctx);
/// Split-KV body of the same op for one short, wide query row: grid.z splits the visible key list
/// (primary + extra scope, `topk_length`-truncated) across CTAs, each writes an fp32 partial to
/// ws.device_scratch and merge_states folds them (and the sink, exactly once) into params.o/lse.
///
/// The count scales `num_tokens * ceil(num_heads / 16)` to two waves of ctx's compute units
/// (occupancy.h), capped at 64 and at one split per 256 visible keys; a shape that already fills
/// the machine, a shape outside the tensor-core body, a padded params.o or `splits == 1` runs the
/// plain sparse_mla_run body and needs no workspace. An explicit lease smaller than
/// sparse_mla_workspace_bytes() is kWorkspaceTooSmall. All three functions are pure w.r.t. device
/// state, so a plan can size the lease before capture and replay it, provided the same shard runs
/// on the same device.
int32_t sparse_mla_split_count(const SparseMLAParams& params, const LaunchContext& ctx);
/// Device scratch bytes sparse_mla_run_ws uses for these params (0 = the plain path).
size_t sparse_mla_workspace_bytes(const SparseMLAParams& params, const LaunchContext& ctx);
Status sparse_mla_run_ws(const SparseMLAParams& params, const Workspace& ws,
                         const LaunchContext& ctx);
Status concat_mla_q(const Tensor& q_out, const Tensor& ql_nope, const Tensor& q_pe,
                    const LaunchContext& ctx);

// ---- A11 sparse scoring / selection, DeepSeek-V4 CSA/HCA ----------------------------------
Status lightning_indexer(const IndexerParams& params, DType q_dtype, const LaunchContext& ctx);
Status sparse_topk(const Tensor& indices, const Tensor& values, const Tensor& scores,
                   const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                   const Tensor& block_table, const Tensor& token_to_request, int32_t page_size,
                   bool absolute_indices, bool to_page_slots, const LaunchContext& ctx);
/// Same selection with a caller-owned device scratch lease. Wide rows (max_kv >= 256 Ki) switch to
/// a cooperative multi-CTA path that chunks every row across (rows * splits) CTAs (three passes
/// over each row instead of one CTA scanning it); every other shape - and a lease that is absent,
/// smaller than sparse_topk_rows_workspace_bytes(rows) or not 256-byte aligned - runs the
/// workspace-free path unchanged, so the result never depends on the workspace. No allocation, no
/// device query, no host synchronization: safe under CUDA-graph capture and replay.
Status sparse_topk_ws(const Tensor& indices, const Tensor& values, const Tensor& scores,
                      const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                      const Tensor& block_table, const Tensor& token_to_request,
                      int32_t page_size, bool absolute_indices, bool to_page_slots,
                      const Workspace& ws, const LaunchContext& ctx);
/// Device scratch bytes one fast-path row uses; sparse_topk_rows_workspace_bytes(rows) covers a
/// batch of `rows` wide rows. Reserving it unconditionally covers every shape (the other shapes
/// ignore it).
size_t sparse_topk_workspace_bytes() noexcept;
size_t sparse_topk_rows_workspace_bytes(int32_t rows) noexcept;
Status block_select(const Tensor& block_row_indptr, const Tensor& block_col_indices,
                    const Tensor& q, const Tensor& summaries, const Tensor& qo_indptr,
                    const Tensor& kv_lens, int32_t q_block, int32_t kv_block, int32_t topk_blocks,
                    int32_t num_sink_blocks, int32_t num_local_blocks, const LaunchContext& ctx);
Status kv_compress(const KVCompressorParams& params, DType in_dtype, const LaunchContext& ctx);
Status compressed_attention(const CompressedAttentionParams& params, const LaunchContext& ctx);
/// Split-KV body for small grids: while `num_tokens * ceil(Hq / 16)` stays below two waves of
/// ctx's compute units the one-CTA-per-16-heads launch leaves most of the partition idle and every
/// CTA pays the full compressed + window scan. grid.z then splits the visible key list (compressed
/// entries and the SWA branch together, one softmax) and each CTA writes an fp32 partial that
/// merge_states folds back, with the sink counted exactly once (split 0). The split count caps at
/// 64 and at one split per 256 visible keys of `topk + window_size`.
///
/// Same contract as the sparse_mla pair: the count is a pure host function of the shape and the
/// context; splits == 1, a shape outside the tensor-core body or a padded params.o falls back to
/// the plain launch; a lease smaller than compressed_attention_workspace_bytes() is
/// kWorkspaceTooSmall; the workspace is only read during run.
int32_t compressed_attention_split_count(const CompressedAttentionParams& params,
                                         const LaunchContext& ctx);
size_t compressed_attention_workspace_bytes(const CompressedAttentionParams& params,
                                            const LaunchContext& ctx);
Status compressed_attention_ws(const CompressedAttentionParams& params, const Workspace& ws,
                               const LaunchContext& ctx);

}  // namespace kops::attn

// ---- A09 low-bit rotated KV (EXPERIMENTAL, B17) -------------------------------------------------
// Opt-in entry points; nothing in kops::attn selects them. A low-bit cache is a PagedKVDesc with
// k_codec = v_codec = KVCodec::kLowBitWHT, k_dtype = v_dtype = kUInt8, stride_unit = kByte (every
// stride a multiple of 4 bytes, pages 16-byte aligned), entry_bytes = the K record size and the
// scales inline in the records (k_token_scales / v_token_scales unused). The record layout and the
// codec semantics are in attention/lowbit_kernels.cuh and CONTRACTS §1h.
namespace kops::experimental::lowbit {

struct LowBitCodec;                 // attention/lowbit_kernels.cuh
struct LowBitTables;
enum class LowBitFormat : uint8_t;

/// Fill the descriptor of (k_format, v_format) at head_dim: transforms, packing version, seed.
/// kUnsupportedShape unless head_dim is 64, 128 or 256; kUnsupportedDtype for a format that is
/// not compiled (kFP8, kNVFP4) or a kTurboProd V.
Status lowbit_codec(LowBitFormat k_format, LowBitFormat v_format, int32_t head_dim, uint32_t seed,
                    LowBitCodec* codec);
/// Bytes of one (token, head) record of K (is_k) or V; 0 for an unsupported codec.
int32_t lowbit_entry_bytes(const LowBitCodec& codec, bool is_k);
/// Level tables, thresholds and record sizes the kernels read (also what a host reader needs).
Status lowbit_resolve(const LowBitCodec& codec, LowBitTables* tables);

/// Quantize key / value [num_tokens, Hkv, head_dim] (bf16 or fp16) into the slots of slot_mapping
/// (int64 [num_tokens], < 0 skipped). Mutates only the records of the named slots.
Status lowbit_kv_store(const Tensor& key, const Tensor& value, const kv::PagedKVDesc& cache,
                       const Tensor& slot_mapping, const LowBitCodec& codec,
                       const LaunchContext& ctx);

/// Decode over a low-bit cache. Same plan / run split, workspace and split-KV rules as
/// decode_plan / decode_run (static grid only: no device planner, no tensor-core path).
Status lowbit_decode_plan(const attn::AttentionSpec& spec, const LowBitCodec& codec,
                          int32_t max_batch, int32_t max_kv_len, int32_t q_len,
                          const LaunchContext& ctx, const Workspace& ws, attn::AttentionPlan* plan);
Status lowbit_decode_run(const attn::AttentionPlan& plan, const attn::DecodeParams& params,
                         const LowBitCodec& codec, const Workspace& ws, const LaunchContext& ctx);

/// Paged prefill over a low-bit cache (the chunk's K/V are stored with lowbit_kv_store first).
/// The plan is built on the host from qo_indptr_host / kv_lens_host (int32 host tensors) and is
/// valid for those lengths: replan when they change (no device planner).
Status lowbit_prefill_plan(const attn::AttentionSpec& spec, const LowBitCodec& codec,
                           const Tensor& qo_indptr_host, const Tensor& kv_lens_host,
                           const LaunchContext& ctx, const Workspace& ws,
                           attn::AttentionPlan* plan);
Status lowbit_prefill_run(const attn::AttentionPlan& plan, const attn::PagedPrefillParams& params,
                          const LowBitCodec& codec, const Workspace& ws, const LaunchContext& ctx);

}  // namespace kops::experimental::lowbit
