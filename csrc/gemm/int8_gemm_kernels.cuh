#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/int8_gemm_kernels.cuh                        Catalog: G03
// INT8 W8A8 and INT4xINT8 (W4A8) GEMMs. INT8 and FP8 are separate capability
// rows: different rounding, saturation, zero-point and accumulation (s32) rules.
//   kINT8Tensorwise  s8 x s8 -> s32, alpha = sA * sB
//   kINT8Rowwise     per-token sA[M] (symmetric, or asymmetric with azp[M]) x per-channel sB[N]
//                    D = (acc - azp[m] * colsum_b[n]) * sA[m] * sB[n] + bias      (kEpiAzpCorrection)
//   kW4A8IntInt8     QServe-style: int4 weights with per-channel int8 scale + per-group uint4 zero/scale
//                    (two-level), int8 activations, s32 accumulation, progressive dequant in registers
// Arch mapping: Sm80/86/89 mma.sync s8 m16n8k32; Sm90a wgmma s8; Sm100a tcgen05 kind::i8;
//               Sm120a mma.sync s8. (Throughput of INT8 differs per SKU; the registry records it.)
// Models: SmoothQuant W8A8 checkpoints (OPT, BLOOM, Llama, Mistral), compressed-tensors W8A8-INT8,
//         QServe W4A8 checkpoints.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/// int8_gemm_mma_sync_kernel — Ampere/Ada/RTX-Blackwell W8A8. kAsymA adds the azp correction.
template <typename Traits, GemmRecipe kRecipe, bool kAsymA>
__global__ void int8_gemm_mma_sync_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

/// int8_gemm_wgmma_kernel — Hopper W8A8 (TMA + wgmma s8, warp-specialized, persistent).
template <typename Traits, GemmRecipe kRecipe, bool kAsymA>
__global__ void int8_gemm_wgmma_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/// int8_gemm_tcgen05_kernel — Blackwell datacenter W8A8 (tcgen05.mma kind::i8, s32 TMEM accumulator).
template <typename Traits, GemmRecipe kRecipe, bool kAsymA>
__global__ void int8_gemm_tcgen05_kernel(const KOPS_GRID_CONST GemmTmaParams params);

/**
 * w4a8_int_gemm_kernel — QServe W4A8: int4 weights unpacked to s8 in registers using the
 * two-level scheme (s8 channel scale, u4 group zero/scale; subtraction after multiplication to stay
 * in s8 range), s8 activations, s32 accumulate. Sm80 (mma.sync) and Sm90a (wgmma) instantiations.
 */
template <typename Traits, int GROUP_SIZE>
__global__ void w4a8_int_gemm_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

/// int8_gemv_kernel — M <= 8 W8A8 / W8A16-int8 decode GEMV (dp4a on CUDA cores).
template <typename a_t, bool kAsymA, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void int8_gemv_kernel(const KOPS_GRID_CONST ScaledGemmArgs args);

}  // namespace kops::gemm
