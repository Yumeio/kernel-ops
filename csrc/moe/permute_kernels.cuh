#pragma once
// =============================================================================
// kernel-ops | csrc/moe/permute_kernels.cuh                          Catalog: M02
// Routing metadata (count / prefix / block alignment), token permutation with
// optional fused quantization, weighted unpermute + reduce.
// Metadata produced here is valid for exactly one routing decision; its lifetime
// is that of the caller's buffers (reusable across layers only if routes are shared).
// Requires: common/macros.cuh, common/dtype.cuh, common/quant.cuh.
// =============================================================================

namespace kops::moe {

/**
 * moe_align_block_size_kernel — Triton/Marlin-style metadata: sort (token, k) slots by expert and
 * pad each expert segment to a multiple of block_size.
 *   sorted_token_ids[i]   : flat slot id t*topk + k, or numel (padding sentinel)
 *   expert_ids[b]         : local expert of block b (-1 for experts not on this rank)
 *   num_tokens_post_pad[0]: total padded length
 * Empty experts produce no blocks. Deterministic order inside an expert (stable by slot id).
 */
__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,       // [num_tokens * topk]
    int32_t* __restrict__ sorted_token_ids,     // [max_num_tokens_padded]
    int32_t* __restrict__ expert_ids,           // [max_num_blocks]
    int32_t* __restrict__ num_tokens_post_pad,  // [1]
    int32_t* __restrict__ cumsum_workspace,     // [num_experts + 1]
    const int32_t* __restrict__ expert_map,     // [num_experts] or null
    const int num_experts, const int block_size, const int64_t numel,
    const int max_num_tokens_padded);

/**
 * moe_count_and_offsets_kernel — expert_counts[e] and exclusive prefix expert_offsets[e] (size E+1),
 * optionally padded to align_m (DeepGEMM contiguous layout requires 128-row alignment).
 */
__global__ void moe_count_and_offsets_kernel(
    const int32_t* __restrict__ topk_ids,       // [num_tokens * topk]
    int32_t* __restrict__ expert_counts,        // [num_local_experts]
    int64_t* __restrict__ expert_offsets,       // [num_local_experts + 1]
    const int32_t* __restrict__ expert_map,     // [num_experts] or null
    const int64_t numel, const int num_experts, const int num_local_experts, const int align_m);

/// Arguments of moe_permute_kernel.
struct PermuteParams {
  const void* hidden;                 // [num_tokens, hidden]
  int64_t hidden_stride;
  const int32_t* topk_ids;            // [num_tokens, topk]
  const float* topk_weights;          // [num_tokens, topk]; applied to input only if apply_weight_on_input
  bool apply_weight_on_input;         // Llama-4: router weight multiplies the expert INPUT (top-1)
  const int64_t* expert_offsets;      // [E_local + 1] from moe_count_and_offsets_kernel
  const int32_t* expert_map;          // [E] or null
  int32_t* write_cursor;              // unused since the parallel map; kept so existing callers still compile
  int32_t num_local_experts;          // E_local, required (sizes the map kernel's shared memory)
  // outputs
  void* permuted;                     // [num_rows_padded, hidden] (or quantized payload)
  int64_t permuted_stride;
  int32_t* permuted_row_to_slot;      // [num_rows_padded] -> t*topk + k (or -1 padding)
  int32_t* slot_to_permuted_row;      // [num_tokens * topk] -> row (or -1 if expert not local)
  ActQuantOut quant;                  // optional fused quantization of permuted rows (FP8 128-group, MXFP8, NVFP4)
  int32_t num_tokens, topk, hidden_size;
};

/**
 * moe_permute_kernel — scatter each token to topk expert-contiguous rows (deterministic: rows of a
 * given expert ordered by slot id), with optional fused activation quantization so the expert
 * GEMM reads FP8/FP4 directly.
 */
template <typename scalar_t, ActQuantMode kQuant, int VEC_SIZE>
__global__ void moe_permute_kernel(const __grid_constant__ PermuteParams params);

/**
 * moe_unpermute_reduce_kernel — combine expert outputs back to tokens:
 *   out[t] = routed_scale * sum_k w[t,k] * expert_out[row(t,k)]  (+ shared_out[t]) (+ residual[t])
 * Fixed k order -> bitwise deterministic. Rows of non-local experts contribute 0 (EP combine
 * happens in communication/ep_kernels.cuh). weights_already_applied = true when the GEMM epilogue
 * or act_and_mul already multiplied routing weights.
 */
template <typename scalar_t, typename acc_t, int VEC_SIZE>
__global__ void moe_unpermute_reduce_kernel(
    scalar_t* __restrict__ out,                   // [num_tokens, hidden]
    const int64_t out_stride,
    const scalar_t* __restrict__ expert_out,      // [num_rows_padded, hidden]
    const int64_t expert_out_stride,
    const int32_t* __restrict__ slot_to_permuted_row,  // [num_tokens * topk]
    const float* __restrict__ topk_weights,       // [num_tokens, topk]
    const scalar_t* __restrict__ shared_out,      // [num_tokens, hidden] shared expert output or null
    const scalar_t* __restrict__ residual,        // [num_tokens, hidden] or null
    const float routed_scale,
    const bool weights_already_applied,
    const int num_tokens, const int topk, const int hidden_size);

/**
 * moe_sum_kernel — out[t] = sum_k in[t, k] (vLLM fused_moe "moe_sum" for the [T, K, H] layout).
 */
template <typename scalar_t, int TOPK, int VEC_SIZE>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,                   // [num_tokens, hidden]
    const scalar_t* __restrict__ input,           // [num_tokens, TOPK, hidden]
    const int num_tokens, const int hidden_size);

}  // namespace kops::moe
