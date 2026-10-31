#pragma once
// =============================================================================
// kernel-ops | csrc/attention/merge_kernels.cuh                    Catalog: A04, A14
// Attention-state algebra on (o, lse) pairs: split-KV combine, two-state merge,
// N-state merge (cascade / shared-prefix attention), sink-aware finalize.
//   merge((o_a, l_a), (o_b, l_b)) = ((e^{l_a-l} o_a + e^{l_b-l} o_b), l),  l = logaddexp(l_a, l_b)
// Fully masked inputs (lse = -inf) contribute nothing; all -inf -> o = 0, lse = -inf.
// Requires: common/*, attention/attention_params.cuh.
// =============================================================================

namespace kops::attn {

/// Maximum branches in one hybrid merge. Kept small so the pointer table fits
/// in the kernel parameter space and is graph-safe without pinned host arrays.
inline constexpr int kMaxHybridBranches = 8;

/// Branch pointers carried by value in the hybrid-merge kernel parameter.
struct HybridBranchTable {
  const float* o[kMaxHybridBranches];    // device pointers to [rows, Hq, D] fp32
  const float* lse[kMaxHybridBranches];  // device pointers to [rows, Hq] fp32
  int32_t count;
};

/**
 * merge_split_kv_kernel — A04: combine split partials into the final output.
 *   For row r: l = logsumexp_s lse_partial[s]; o = sum_s exp(lse_partial[s] - l) * o_partial[s]
 * sinks (optional, [Hq]): l <- logaddexp(l, sink_h); o <- o * exp(l_old - l) — applied here when
 * the split kernels did not include it (exactly once).
 * Launch : grid = (total_rows, Hq / HEADS_PER_CTA), block = HEAD_DIM / VEC * HEADS_PER_CTA.
 */
template <typename DTypeO, int HEAD_DIM, int VEC_SIZE, int HEADS_PER_CTA>
__global__ void merge_split_kv_kernel(
    DTypeO* __restrict__ o,                     // [total_rows, Hq, HEAD_DIM]
    const int64_t o_stride_n, const int64_t o_stride_h,
    float* __restrict__ lse,                    // [total_rows, Hq] or null
    const float* __restrict__ o_partial,        // [num_partial_rows, Hq, HEAD_DIM]
    const float* __restrict__ lse_partial,      // [num_partial_rows, Hq]
    const int32_t* __restrict__ merge_indptr,   // [total_rows + 1]
    const float* __restrict__ sink_logits,      // [Hq] or null
    const uint32_t* __restrict__ total_rows_ptr,// [1] device row count for padded graph batches, or null
    const int total_rows, const int num_heads);

/**
 * merge_state_kernel — A14: (o_a, l_a) + (o_b, l_b) -> (o, l). In-place variant writes into a.
 * Typical use: cascade / shared-prefix attention (prefix attended once for many requests, then
 * merged with each request's suffix attention).
 */
template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_state_kernel(
    DTypeO* __restrict__ o,                     // [num_rows, H, HEAD_DIM]
    float* __restrict__ lse,                    // [num_rows, H]
    const DTypeIn* __restrict__ o_a, const float* __restrict__ lse_a,
    const DTypeIn* __restrict__ o_b, const float* __restrict__ lse_b,
    const uint8_t* __restrict__ row_mask,       // [num_rows] 0 = take a unchanged (b absent), or null
    const int num_rows, const int num_heads);

/**
 * merge_states_kernel — A14: merge N states per row: o = sum_i e^{l_i - l} o_i, l = logsumexp l_i.
 * Deterministic summation order i = 0..N-1.
 */
template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_states_kernel(
    DTypeO* __restrict__ o,                     // [num_rows, H, HEAD_DIM]
    float* __restrict__ lse,                    // [num_rows, H] or null
    const DTypeIn* __restrict__ o_states,       // [num_rows, N, H, HEAD_DIM]
    const float* __restrict__ lse_states,       // [num_rows, N, H]
    const int num_rows, const int num_states, const int num_heads);

/**
 * merge_hybrid_branches_kernel — combine attention branches that score DIFFERENT key sets with a
 * shared softmax (DeepSeek-V4: compressed sparse / heavily compressed entries + sliding-window
 * branch + sink; Llama-4/Gemma local+global never share a softmax and do not use this).
 * Identical to merge_states but tolerates per-branch head counts mapped by branch_head_map.
 *
 * Branch pointers are packed into HybridBranchTable by value, so the kernel never dereferences
 * host memory and the caller does not need pinned arrays. The branch buffers themselves must
 * stay at stable device addresses between graph capture and replay (as with any other pointer
 * argument); a null o pointer marks an absent branch.
 */
template <typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_hybrid_branches_kernel(
    DTypeO* __restrict__ o,                     // [num_rows, Hq, HEAD_DIM]
    float* __restrict__ lse,                    // [num_rows, Hq] or null
    const HybridBranchTable branches,           // by value; null o = absent
    const float* __restrict__ sink_logits,      // [Hq] or null
    const int num_rows, const int num_heads);

}  // namespace kops::attn
