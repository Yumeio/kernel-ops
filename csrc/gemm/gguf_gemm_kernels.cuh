#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/gguf_gemm_kernels.cuh                        Catalog: G04
// GGUF (ggml block formats) GEMM / GEMV, dense and MoE (expert-indexed).
// Activations are quantized to block_q8_1 on the fly; dot products use dp4a /
// int8 mma on the dequantized ggml blocks. Supported block types: GGUFType.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/// q8_1 activation quantization for GGUF kernels (32-element blocks with fp16 d and sum).
template <typename scalar_t>
__global__ void gguf_quantize_q8_1_kernel(
    void* __restrict__ out_q8_1,            // [M, K/32] block_q8_1
    const scalar_t* __restrict__ a,         // [M, K]
    const int m, const int k);

/// gguf_mmvq_kernel — matrix-vector product (M <= 8) against ggml blocks of [N, K].
template <typename scalar_t, GGUFType kType, int M_TILE>
__global__ void gguf_mmvq_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ qweight,    // ggml blocks of [N, K]
    const void* __restrict__ a_q8_1,        // activations pre-quantized to block_q8_1 [M, K/32]
    const int m, const int n, const int k);

/// gguf_mmq_kernel — tiled matrix-matrix variant (prefill), int8 mma on Sm80+ / dp4a fallback.
template <typename scalar_t, GGUFType kType, int TILE_M, int TILE_N>
__global__ void gguf_mmq_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ qweight,
    const void* __restrict__ a_q8_1,
    const int m, const int n, const int k);

/**
 * gguf_moe_mmvq_kernel / gguf_moe_mmq_kernel — expert-indexed GGUF GEMMs: rows of A grouped by
 * sorted_token_ids / expert_ids (moe_align_block_size_kernel), weights [E, N, K] ggml blocks.
 * Models : GGUF MoE checkpoints (Mixtral, Qwen3-MoE, DeepSeek, GPT-OSS GGUF, GLM-4.5 GGUF).
 */
template <typename scalar_t, GGUFType kType, int M_TILE>
__global__ void gguf_moe_mmvq_kernel(
    scalar_t* __restrict__ out,                   // [num_tokens * topk, N]
    const uint8_t* __restrict__ qweight,          // [E, N, K] blocks
    const int64_t expert_stride_bytes,
    const void* __restrict__ a_q8_1,              // [num_tokens, K/32]
    const int32_t* __restrict__ topk_ids,         // [num_tokens, topk]
    const int num_tokens, const int topk, const int n, const int k);

template <typename scalar_t, GGUFType kType, int TILE_M, int TILE_N>
__global__ void gguf_moe_mmq_kernel(
    scalar_t* __restrict__ out,                   // [num_tokens_post_pad, N]
    const uint8_t* __restrict__ qweight,
    const int64_t expert_stride_bytes,
    const void* __restrict__ a_q8_1,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_pad,
    const int top_k, const int n, const int k);

}  // namespace kops::gemm
