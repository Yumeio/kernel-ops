#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/fp4_gemm_kernels.cuh                        Catalog: G05
// FP4 / FP6 block-scaled GEMMs. NVFP4 != MXFP4: different block size (16 vs 32),
// scale type (E4M3 vs UE8M0) and NVFP4's extra fp32 per-tensor global scale.
//   kNVFP4         A, B e2m1; sA, sB E4M3 per 16; alpha = 1 / (gA * gB)       W4A4
//   kMXFP4         A, B e2m1; sA, sB UE8M0 per 32                             W4A4
//   kMXFP4xMXFP8   B e2m1 (UE8M0/32), A e4m3 (UE8M0/32)                        W4A8-MX
//   kMXFP6xMXFP8   B e2m3 or e3m2 (UE8M0/32), A e4m3                           spec-complete
//   kW4A8FP4xFP8   B MXFP4/NVFP4 converted to e4m3 in-kernel, A fp8 per-token/per-group (no native FP4 MMA)
// Arch mapping:
//   Sm100a/103a  tcgen05.mma kind::mxf4nvf4 (scale_vec 4X = NVFP4, 2X = MXFP4), kind::mxf4,
//                kind::mxf8f6f4 for mixed FP4/FP6 x FP8 (FP4/FP6 operand in the padded smem format,
//                loaded by TMA in its 4/6-bit -> byte-aligned mode). Scale factors staged in TMEM.
//   Sm120a/121a  mma.sync kind::mxf4nvf4.block_scale (k64) / kind::mxf8f6f4.block_scale (k32)
//   Sm90a        no FP4 MMA: kW4A8FP4xFP8 (FP4 -> e4m3 + wgmma) or kW4A16* (mixed_input)
//   Sm80/89      kW4A16MXFP4 / kW4A16NVFP4 via Marlin FE2M1 or fp4_gemv_kernel
// Scale layout: kSwizzled128x4 (both MX and NV) for tensor-core paths; produced by
// quantization/quant_kernels.cuh or gemm/gemm_prep_kernels.cuh::block_scale_swizzle_kernel.
// Models: GPT-OSS (MXFP4 experts), Kimi K3 (MXFP4 weights / MXFP8 acts), DeepSeek-V4 (FP4 experts),
//         NVFP4 ModelOpt checkpoints (Llama 3.x/4, Qwen3, DeepSeek-R1/V3 FP4, Nemotron), MXFP4 QAT models.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/// Flat arguments for block-scaled GEMMs (kept for adapters; native kernels take GemmTmaParams).
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
 * fp4_gemm_tcgen05_kernel — Blackwell datacenter native FP4/FP6 GEMM. kRecipe in
 * {kNVFP4, kMXFP4, kMXFP4xMXFP8, kMXFP6xMXFP8}. 1-SM / 2-SM, persistent, TMA multicast of the
 * weight tile in 2-SM mode. Epilogue may emit NVFP4/MXFP8 output for the next GEMM (kEpiQuantOut,
 * block amax computed within the N tile, which must be a multiple of the block size).
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void fp4_gemm_tcgen05_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * fp4_gemm_sm120_kernel — RTX Blackwell / GB10 native FP4: block-scaled mma.sync
 * (kind::mxf4nvf4 for NVFP4/MXFP4, kind::mxf8f6f4 for MXFP4xMXFP8), TMA + warp specialization.
 * Separate tile tables and smem budget from Sm100a; never selected by an sm_100 capability row.
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void fp4_gemm_sm120_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * fp4_to_fp8_gemm_wgmma_kernel — Hopper path for FP4 weights (kW4A8FP4xFP8): each FP4 weight tile
 * is converted to e4m3 in shared memory with its block scale folded into a per-(N, 32|16 K)
 * fp32 factor applied at promotion time; activations are FP8 (per-token or 1x128 group);
 * wgmma e4m3 with Fp8Accum::kPromoted. Halves weight bytes vs FP8 at FP8 MMA rate.
 */
template <typename Traits, ScaleGranularity kWeightBlock>
__global__ void fp4_to_fp8_gemm_wgmma_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * fp4_gemv_kernel — M <= 8 decode GEMV for any FP4 weight (MXFP4 / NVFP4) on any arch: e2m1 is
 * decoded arithmetically in registers, with no table (three magnitude bits placed under the fp32
 * exponent, times 2^-126; exact for all 16 codes); block scales applied per 16/32; activations bf16
 * (W4A16) or e4m3 / MXFP8 (W4A8). Weight-bandwidth bound; split-K for small N.
 * A table lookup and an arch-specific e2m1 cvt were both considered: the NF4 measurement in
 * mixed_input_gemm.cu (LUT refill slower than the ALU it removes) and the fact that SM100a/SM120a
 * consume e2m1 natively (B13: fp4_gemm_sm120_kernel / fp4_gemm_tcgen05_kernel) keep this body
 * arch-generic.
 * ROWS_PER_WARP == 0 selects the tensor-core body instead: fp4_to_marlin tiles
 * (WeightPacking::kMarlin) on mma.sync m16n8k16, SM80+, M <= 16 (M_TILE 8 or 16), N % 64 == 0,
 * WARPS == 8; one CTA per 64-column n-tile.
 */
template <typename a_t, GemmRecipe kRecipe, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void fp4_gemv_kernel(const __grid_constant__ ScaledGemmArgs args);

}  // namespace kops::gemm
