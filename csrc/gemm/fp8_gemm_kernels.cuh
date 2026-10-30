#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/fp8_gemm_kernels.cuh                    Catalog: G03, G07
// FP8 GEMM family. Recipes (GemmRecipe):
//   kFP8Tensorwise      sA[1]       x sB[1]          epilogue alpha
//   kFP8Rowwise         sA[M]       x sB[N]          epilogue row/col scale
//   kFP8Blockwise       sA[M, K/128] x sB[N/128, K/128] fp32, applied per 128-K block in the mainloop
//   kFP8BlockwiseUE8M0  same blocks, UE8M0 (power-of-two) scales, packed 4 per int32 when required
//   kMXFP8              sA[M, K/32] x sB[N, K/32] UE8M0 (OCP MX, hardware block scaling on Blackwell)
// Arch mapping:
//   Sm89          mma.sync e4m3/e5m2 m16n8k32 (Ada: L4, L40S); blockwise via per-K-block fp32 rescale
//   Sm90a         wgmma e4m3 k32; Fp8Accum::kPromoted promotes to fp32 every fp8_promotion_k (128)
//   Sm100a/103a   tcgen05 kind::f8f6f4 (tensor/row) | kind::mxf8f6f4 block_scale (MXFP8, UE8M0 blockwise)
//   Sm120a/121a   mma.sync kind::f8f6f4 / kind::mxf8f6f4.block_scale
//   Sm80/Sm86     no FP8 MMA: FP8 checkpoints run as kW8A16FP8 (mixed_input_gemm_kernels.cuh)
// All FP8 tensor-core paths require K-major A and B.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/**
 * fp8_gemm_mma_sync_kernel — Ada / RTX-Blackwell (non-block-scaled) FP8 GEMM with cp.async (Sm89)
 * or TMA (Sm120a) pipelines. kRecipe: kFP8Tensorwise | kFP8Rowwise | kFP8Blockwise (per-128-K
 * block: acc_block * sA[m, kb] * sB[n/128, kb] added into fp32 accumulators).
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void fp8_gemm_mma_sync_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

/**
 * fp8_gemm_wgmma_kernel — Hopper FP8 GEMM (TMA + WGMMA, warp-specialized cooperative / ping-pong,
 * persistent). kRecipe:
 *   kFP8Tensorwise / kFP8Rowwise : scales in epilogue; kAccum = kFastAccum allowed
 *   kFP8Blockwise                : DeepSeek recipe; two-level accumulation (WGMMA partial -> fp32 regs
 *                                  * sA * sB every 128 K); scales of A read with TMA from the
 *                                  col-major TMA-aligned layout produced by quant/norm/act kernels
 *   kFP8BlockwiseUE8M0           : same, power-of-two scales (exact rescale by exponent add)
 * Swap-AB instantiations serve M <= 64 decode shapes (N becomes the 64-row WGMMA dimension).
 */
template <typename Traits, GemmRecipe kRecipe, Fp8Accum kAccum>
__global__ void fp8_gemm_wgmma_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/**
 * fp8_gemm_tcgen05_kernel — Blackwell datacenter FP8 GEMM.
 *   kFP8Tensorwise / kFP8Rowwise : tcgen05.mma kind::f8f6f4, fp32 TMEM accumulation, scales in epilogue
 *   kMXFP8                       : tcgen05.mma kind::mxf8f6f4.block_scale, scale_vec 1X (32), UE8M0
 *                                  scale tiles copied to TMEM (tcgen05.cp)
 *   kFP8BlockwiseUE8M0           : realized on the MX block-scale hardware path with each 1x128 / 128x128
 *                                  UE8M0 scale broadcast to its 32-element sub-blocks
 *   kFP8Blockwise (fp32 scales)  : per-128-K partial accumulators read from TMEM and rescaled in registers
 *                                  (slower; prefer converting checkpoints to UE8M0 when numerics allow)
 * 1-SM and 2-SM (CTA pair) instantiations; persistent.
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void fp8_gemm_tcgen05_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/**
 * fp8_gemm_sm120_kernel — RTX Blackwell FP8: kFP8Tensorwise / kFP8Rowwise via mma.sync kind::f8f6f4,
 * kMXFP8 / kFP8BlockwiseUE8M0 via mma.sync kind::mxf8f6f4.block_scale; TMA + warp-specialized.
 */
template <typename Traits, GemmRecipe kRecipe>
__global__ void fp8_gemm_sm120_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/// Parameters of fp8_block_gemm_small_m_kernel.
struct Fp8BlockGemmParams {
  const f8e4m3_t* a;          // [M, K] e4m3
  int64_t lda;
  const void* a_scale;        // [K/128, round_up(M,4)] col-major fp32 (or packed UE8M0)
  const f8e4m3_t* b;          // [N, K] e4m3
  int64_t ldb;
  const void* b_scale;        // [N/128, K/128] fp32 (or UE8M0)
  bool scales_ue8m0;
  void* d;                    // [M, N] bf16
  int64_t ldd;
  EpilogueParams epi;
  int32_t m, n, k;
};

/**
 * fp8_block_gemm_small_m_kernel — DeepSeek-style 1x128 (A) x 128x128 (B) block-scaled FP8 GEMM for
 * M <= 64 (decode) using mma.sync tiles with per-block promotion; avoids the 64-row minimum of
 * WGMMA without swap-AB. Larger M goes to fp8_gemm_wgmma_kernel / fp8_gemm_tcgen05_kernel.
 * Models : DeepSeek V3/V3.1/V3.2/V4 dense layers & attention projections, Kimi K2 FP8, GLM-4.5 FP8,
 *          Qwen3 FP8, MiniMax-M2 FP8.
 */
template <int M_TILE, int N_TILE, typename ArchTag>
__global__ void fp8_block_gemm_small_m_kernel(const KOPS_GRID_CONST Fp8BlockGemmParams params);

/**
 * fp8_gemv_kernel — M <= 8 FP8-weight GEMV on CUDA cores: e4m3 weights widened in registers
 * (cvt.rn.f16x2.e4m3x2 on Sm89+, LUT on Sm80), activations bf16 or e4m3, any FP8 recipe's scales.
 * The weight-bandwidth-optimal decode path on every arch including those without FP8 MMA.
 */
template <typename a_t, GemmRecipe kRecipe, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void fp8_gemv_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

/**
 * fp8_bmm_kernel — strided-batched FP8 GEMM (per-batch tensorwise or rowwise scales): MLA
 * absorbed projections W_UK / W_UV with FP8 weights (DeepSeek / Kimi FP8 checkpoints),
 * FlashInfer bmm_fp8 equivalent. A may be bf16 (quantized in the prologue) or e4m3.
 */
template <typename a_t, typename d_t, GemmRecipe kRecipe, typename ArchTag>
__global__ void fp8_bmm_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

}  // namespace kops::gemm
