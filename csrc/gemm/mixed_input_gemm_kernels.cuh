#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/mixed_input_gemm_kernels.cuh                Catalog: G04
// Weight-only and mixed-precision GEMMs: low-bit weights are converted in-kernel
// to the MMA type of the activations.
//   Marlin   (Sm80+)  int4/uint4b8/uint8b128/fp8/fp4 weights x fp16/bf16, group / channel / act-order
//   Machete  (Sm90a)  CUTLASS mixed-input WGMMA: (u)int4, (u)int8, int4b8 x fp16/bf16 (+ fp8 act for W4A8)
//   tcgen05  (Sm100a) int4 / FP4 / FP8 weights -> bf16 (kind::f16) or e4m3 (kind::f8f6f4) in smem
//   GEMV     (any)    M <= 4 direct AWQ / GPTQ layouts, LUT codebooks (NF4)
// Recipes: kW4A16Int, kW8A16Int, kW8A16FP8, kW4A16MXFP4, kW4A16NVFP4, kW4A8IntFP8, kLUT4.
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
 * marlin_gemm_kernel — mixed-input tensor-core GEMM with in-register dequantization (lop3 tricks).
 * Weight types: quant::MarlinWeightType (kU4 + zeros = AWQ, kU4B8 = GPTQ sym, kU8B128, kFE4M3 =
 * FP8 weight-only, kFE2M1 = MXFP4/NVFP4 weight-only). Template knobs mirror Marlin: threads,
 * m/n/k thread blocks, pipeline stages, group_blocks (-1 channelwise, 0 act-order, else group/16).
 * Models : AWQ/GPTQ/compressed-tensors W4A16 (Llama, Qwen, Mistral, DeepSeek distills),
 *          Kimi K2-Thinking INT4 QAT (group 32), FP8/FP4 checkpoints on Sm80/Sm86/Sm89.
 */
template <typename scalar_t, quant::MarlinWeightType kWType, int THREADS, int THREAD_M_BLOCKS,
          int THREAD_N_BLOCKS, int THREAD_K_BLOCKS, bool M_BLOCK_SIZE_8, int STAGES,
          int GROUP_BLOCKS, bool IS_ZP_FLOAT>
__global__ void marlin_gemm_kernel(const KOPS_GRID_CONST MarlinParams params);

/**
 * machete_gemm_kernel — Hopper mixed-input GEMM (TMA + WGMMA): weights pre-packed by
 * gemm_prep_kernels.cuh::machete_prepack_kernel into the layout the converter warps expect,
 * converted to the activation type in registers before wgmma. Group scales (+zeros) applied in
 * the mainloop, channel/token scales in the epilogue. kRecipe: kW4A16Int, kW8A16Int, kW4A8IntFP8
 * (int4 -> e4m3 with scale folding, fp8 activations; Hopper W4A8-AWQ).
 */
template <typename Traits, GemmRecipe kRecipe, int GROUP_SIZE, bool kHasZeros>
__global__ void machete_gemm_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/**
 * mixed_input_gemm_tcgen05_kernel — Blackwell mixed-input: a converter warpgroup dequantizes weight
 * tiles (int4 groupwise, MXFP4, NVFP4, FP8 weight-only) from smem into bf16 (kind::f16) or e4m3
 * (kind::f8f6f4) staging buffers consumed by tcgen05.mma. kRecipe: kW4A16Int, kW4A16MXFP4 (GPT-OSS
 * MXFP4 x BF16 on B200), kW4A16NVFP4, kW8A16FP8, kW4A8IntFP8, kW4A8FP4xFP8.
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void mixed_input_gemm_tcgen05_kernel(const KOPS_GRID_CONST GemmTmaParams params);

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
 * lut_gemv_kernel — 4-bit codebook weights (bitsandbytes NF4 / FP4, custom 16-entry LUTs) with
 * blockwise absmax that may itself be double-quantized (int8 absmax + fp32 second-level scale).
 * M <= 8; larger M dequantizes to bf16 tiles (quant::dequantize_kernel) + dense GEMM.
 */
template <typename scalar_t, int BLOCK_SIZE, bool kDoubleQuant, int M_TILE>
__global__ void lut_gemv_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const scalar_t* __restrict__ a,         // [M, K]
    const uint8_t* __restrict__ qweight,    // [N, K/2] packed 4-bit codes
    const float* __restrict__ codebook,     // [16]
    const void* __restrict__ absmax,        // [N*K/BLOCK_SIZE] fp32, or uint8 when kDoubleQuant
    const float* __restrict__ absmax_scale, // second-level scales [N*K/(BLOCK_SIZE*256)] or null
    const float* __restrict__ absmax_offset,// double-quant offset [1] or null
    const int m, const int n, const int k);

}  // namespace kops::gemm
