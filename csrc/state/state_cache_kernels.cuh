#pragma once
// =============================================================================
// kernel-ops | csrc/state/state_cache_kernels.cuh                    Catalog: L03
// Byte-preserving primitives for recurrent / conv states (Mamba, GDN, KDA,
// lightning attention, DeepSeek-V4 compressor tails). Policy (which slot, when
// to snapshot, prefix identity, eviction) belongs to the engine.
// Requires: common/macros.cuh.
// =============================================================================

namespace kops::state {

/**
 * state_copy_kernel — dst_slot <- src_slot for every pair, across all state pools of all layers
 * (conv state, SSM / delta-rule state, compressor tail). Used for copy-on-write when a request
 * hits a cached prefix state (vLLM Kimi-K3 fine-grained prefix hits copy into a private slot
 * before extending), forks (n > 1 sampling, beam), and defragmentation.
 * Launch : grid = (num_pairs, num_pools), block = 256, 16-byte vectorized.
 */
__global__ void state_copy_kernel(
    void* const* __restrict__ pools,           // [num_pools] base pointers
    const int64_t* __restrict__ slot_bytes,    // [num_pools] bytes per slot of each pool
    const int32_t* __restrict__ slot_pairs,    // [num_pairs, 2] (src, dst)
    const int num_pairs, const int num_pools);

/**
 * state_gather_kernel / state_scatter_kernel — move states between pool slots and a contiguous
 * staging buffer (offload to CPU / disk tiers, PD-disaggregation transfer, batched init).
 */
__global__ void state_gather_kernel(
    void* __restrict__ out,                    // [num_items, total_bytes_per_item]
    void* const* __restrict__ pools,
    const int64_t* __restrict__ slot_bytes,
    const int32_t* __restrict__ slots,         // [num_items]
    const int num_items, const int num_pools);

__global__ void state_scatter_kernel(
    void* const* __restrict__ pools,
    const void* __restrict__ src,              // [num_items, total_bytes_per_item]
    const int64_t* __restrict__ slot_bytes,
    const int32_t* __restrict__ slots,         // [num_items]
    const int num_items, const int num_pools);

/// state_zero_kernel — reset slots for new sequences (when has_initial_state is not used).
__global__ void state_zero_kernel(
    void* const* __restrict__ pools,
    const int64_t* __restrict__ slot_bytes,
    const int32_t* __restrict__ slots,         // [num_items]
    const int num_items, const int num_pools);

/**
 * state_commit_accepted_kernel — speculative decoding rollback/commit: intermediate[b, j] is the
 * state to adopt when j drafts were accepted (the state after prefix + j draft tokens).  The live
 * slot already holds the state after the full draft forward, so the all-accepted row needs no copy.
 *   live[b] <- intermediate[b, num_accepted[b]]  (no copy when num_accepted == num_drafts)
 */
__global__ void state_commit_accepted_kernel(
    void* const* __restrict__ pools,
    const int64_t* __restrict__ slot_bytes,
    const int32_t* __restrict__ live_slots,           // [B]
    const int32_t* __restrict__ intermediate_slots,   // [B, max_draft + 1]
    const int32_t* __restrict__ num_accepted,         // [B]
    const int max_draft_plus_one, const int batch_size, const int num_pools);

}  // namespace kops::state
