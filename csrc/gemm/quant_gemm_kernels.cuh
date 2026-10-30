#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/quant_gemm_kernels.cuh                  Catalog: G03, G04, G05
// Weight-only and block-scaled low-bit GEMMs: Marlin (W4A16/W8A16/FP8/FP4 weight-only),
// small-M W4A16 GEMV, GGUF MMVQ/MMQ, MX/NVFP4 block-scaled FP4 GEMM, W4A8.
// Requires: common/*, gemm/gemm_params.cuh, quantization/weight_repack_kernels.cuh (MarlinWeightType).
// =============================================================================

namespace kops::gemm {

/// Arguments of marlin_gemm_kernel (also used by the MoE Marlin path with expert routing).
struct MarlinParams {
  const void* a;                    // [M, K] fp16 / bf16 activations (permuted if act-order)
  int64_t lda;
  const uint32_t* b;                // Marlin-packed weights [K/16, N*16/pack_factor]
  void* c;                          // [M, N] output
  int64_t ldc;
  float* c_tmp;                     // [M, N] fp32 reduce buffer when use_fp32_reduce
  const void* bias;                 // [N] or null
  const void* scales;               // [K/group, N] (Marlin permuted) or [1, N] channelwise
  const float* global_scale;        // NVFP4 weight-only global scale or null
  const void* zeros;                // AWQ / GPTQ-asym zero points (Marlin layout) or null
  const int32_t* g_idx;             // act-order group index [K] or null
  const int32_t* perm;              // act-order permutation of A columns [K] or null
  // MoE extension (null for dense): rows are gathered by sorted_token_ids per expert block.
  const int32_t* sorted_token_ids;  // [num_tokens_post_pad]
  const int32_t* expert_ids;        // [num_blocks]
  const int32_t* num_tokens_post_pad;
  const float* topk_weights;        // [M_tokens * topk] applied in epilogue when mul_topk_weights
  int32_t moe_block_size;
  int32_t top_k;
  bool mul_topk_weights;
  int32_t* locks;                   // [num_sms * 16] zeroed workspace (epoch-tagged)
  int32_t num_groups;               // K / group_size (1 for channelwise)
  int32_t group_size;
  int32_t prob_m, prob_n, prob_k;
  bool is_k_full;                   // false when K is sharded with act-order (TP row-parallel)
  bool use_atomic_add;              // small-M: atomic fp16/bf16 adds instead of global reduce
  bool use_fp32_reduce;
  int32_t max_shared_mem;
};

/**
 * marlin_gemm_kernel — mixed-input tensor-core GEMM with in-register dequantization.
 * Weight types: see quant::MarlinWeightType (U4 + zeros = AWQ, U4B8 = GPTQ sym, U8B128,
 * FE4M3 = FP8 weight-only, FE2M1 = MXFP4/NVFP4 weight-only).
 * Models : any AWQ/GPTQ/compressed-tensors W4A16 checkpoint (Llama, Qwen, Mistral, DeepSeek
 *          distills), Kimi K2-Thinking INT4 QAT (group 32), FP4 weight-only fallback on SM80-SM90.
 * Template knobs mirror Marlin: threads, m/n/k thread blocks, pipeline stages, group_blocks
 * (-1 = channelwise, 0 = act-order, else group_size/16).
 */
template <typename scalar_t, quant::MarlinWeightType kWType, int THREADS, int THREAD_M_BLOCKS,
          int THREAD_N_BLOCKS, int THREAD_K_BLOCKS, bool M_BLOCK_SIZE_8, int STAGES,
          int GROUP_BLOCKS, bool IS_ZP_FLOAT>
__global__ void marlin_gemm_kernel(const __grid_constant__ MarlinParams params);

/**
 * w4a16_gemv_kernel — M <= 4 weight-only int4 GEMV with group scales/zeros (AWQ or GPTQ layout
 * read directly, no Marlin repack). Decode fallback for consumer GPUs and odd shapes.
 */
template <typename scalar_t, WeightPacking kPacking, int GROUP_SIZE, bool kHasZeros, int M_TILE>
__global__ void w4a16_gemv_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const scalar_t* __restrict__ a,         // [M, K]
    const uint32_t* __restrict__ qweight,   // AWQ [K, N/8] or GPTQ [K/8, N]
    const scalar_t* __restrict__ scales,    // [K/GROUP_SIZE, N]
    const uint32_t* __restrict__ qzeros,    // [K/GROUP_SIZE, N/8] or null
    const int32_t* __restrict__ g_idx,      // GPTQ act-order [K] or null
    const scalar_t* __restrict__ bias,      // [N] or null
    const int m, const int n, const int k);

/**
 * gguf_mmvq_kernel — GGUF matrix-vector product (M <= 8): quantize activations to q8_1 on the fly,
 * dot products with ggml block formats (dp4a). gguf_mmq_kernel — tiled matrix-matrix variant.
 */
template <typename scalar_t, GGUFType kType, int M_TILE>
__global__ void gguf_mmvq_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ qweight,    // ggml blocks of [N, K]
    const void* __restrict__ a_q8_1,        // activations pre-quantized to block_q8_1 [M, K/32]
    const int m, const int n, const int k);

template <typename scalar_t, GGUFType kType, int TILE_M, int TILE_N>
__global__ void gguf_mmq_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const uint8_t* __restrict__ qweight,
    const void* __restrict__ a_q8_1,
    const int m, const int n, const int k);

/// q8_1 activation quantization for GGUF kernels (32-element blocks with fp16 d and sum).
template <typename scalar_t>
__global__ void gguf_quantize_q8_1_kernel(
    void* __restrict__ out_q8_1,            // [M, K/32] block_q8_1
    const scalar_t* __restrict__ a,         // [M, K]
    const int m, const int k);

/// Arguments for block-scaled FP4/FP8 GEMMs (MXFP4, MXFP8, NVFP4, W4A8 MX).
struct BlockScaledGemmParams {
  const void* a;            int64_t lda;   // [M, K] e2m1 packed or e4m3
  const void* a_scales;                    // swizzled UE8M0 (MX) or E4M3 (NVFP4)
  const void* b;            int64_t ldb;   // [N, K] e2m1 packed or e4m3
  const void* b_scales;
  const float* a_global_scale;             // NVFP4 only
  const float* b_global_scale;             // NVFP4 only
  void* d;                  int64_t ldd;   // [M, N] bf16 / fp16 / fp32
  DType a_dtype, b_dtype;
  ActQuantMode a_format, b_format;         // kMXFP8 | kMXFP4 | kNVFP4
  EpilogueParams epi;
  int32_t m, n, k;
};

/**
 * block_scaled_gemm_kernel — native FP4/FP8 block-scaled MMA.
 *   Sm100a/Sm103a: tcgen05.mma.kind::mxf4nvf4 / mxf8f6f4 with TMEM accumulators, 2-CTA clusters.
 *   Sm120a/Sm121a: mma.sync block-scaled (different tile shapes, no TMEM) -> separate instantiation.
 * W4A4 = (kNVFP4, kNVFP4) or (kMXFP4, kMXFP4); W4A8 = (kMXFP8 activations, kMXFP4 weights) as
 * in the Kimi-K3 release recipe; NVFP4 != MXFP4 (scale dtype, block size, global scale).
 */
template <typename d_t, int TILE_M, int TILE_N, int TILE_K, int CLUSTER_M, typename ArchTag>
__global__ void block_scaled_gemm_kernel(const __grid_constant__ BlockScaledGemmParams params);

}  // namespace kops::gemm
