#pragma once
// =============================================================================
// kernel-ops | csrc/quantization/weight_repack_kernels.cuh   Catalog: Q02, Q03, Q04, G04/G05 prep
// Offline / load-time weight conversion: AWQ & GPTQ import, Marlin repack,
// SmoothQuant folding, GGUF dequant, FP8 block scale re-encoding, FP4 swizzles.
// Never on the serving hot path; run once after weight load, outside graph capture.
// Requires: common/macros.cuh, common/dtype.cuh, common/quant.cuh.
// =============================================================================

namespace kops::quant {

/// Marlin weight element types (mirrors vLLM ScalarType ids used by Marlin).
enum class MarlinWeightType : uint8_t {
  kU4 = 0,       // uint4 with explicit zero points (AWQ)
  kU4B8 = 1,     // uint4 with bias 8 (GPTQ symmetric)
  kU8B128 = 2,   // uint8 with bias 128 (GPTQ 8-bit)
  kFE4M3 = 3,    // fp8 weight-only (W8A16 FP8)
  kFE2M1 = 4,    // fp4 weight-only (MXFP4 / NVFP4 on GPUs without native FP4 MMA)
};

/**
 * awq_dequantize_kernel — w[k, n] = (q[k, n] - z[k/g, n]) * s[k/g, n]
 *   qweight [K, N/8] int32 with AWQ nibble order {0,2,4,6,1,3,5,7}; qzeros [K/g, N/8] same order.
 * Reference / fallback path and checkpoint conversion.
 */
template <typename scalar_t>
__global__ void awq_dequantize_kernel(
    scalar_t* __restrict__ out,             // [K, N]
    const int32_t* __restrict__ qweight,    // [K, N/8]
    const scalar_t* __restrict__ scales,    // [K/group_size, N]
    const int32_t* __restrict__ qzeros,     // [K/group_size, N/8]
    const int size_k, const int size_n, const int group_size);

/**
 * awq_marlin_repack_kernel — AWQ qweight -> Marlin tile layout (num_bits 4 or 8).
 * Launch : grid = num_sms (persistent over 16x64 tiles), block = 128..256; workspace none.
 */
__global__ void awq_marlin_repack_kernel(
    uint32_t* __restrict__ out,             // [K/16, N*16/pack_factor] Marlin layout
    const uint32_t* __restrict__ qweight,   // [K, N/pack_factor]
    const int size_k, const int size_n, const int num_bits);

/**
 * gptq_marlin_repack_kernel — GPTQ qweight [K/pack, N] -> Marlin, applying the act-order
 * permutation `perm` (argsort of g_idx) so the GEMM reads K in group order. Must not drop the
 * permutation: the activation side is permuted with the same `perm` (see gemm marlin_gemm).
 */
__global__ void gptq_marlin_repack_kernel(
    uint32_t* __restrict__ out,             // Marlin layout
    const uint32_t* __restrict__ qweight,   // [K/pack_factor, N]
    const uint32_t* __restrict__ perm,      // [K] or null (no act-order)
    const int size_k, const int size_n, const int num_bits);

/**
 * marlin_permute_scales_kernel — reorder per-group (or per-channel) scales into Marlin order.
 * marlin_awq_zero_points_kernel — AWQ packed zeros -> Marlin zero layout (undo AWQ interleave,
 *                                 apply Marlin permutation, repack).
 */
template <typename scale_t>
__global__ void marlin_permute_scales_kernel(
    scale_t* __restrict__ out,              // [K/group or 1, N]
    const scale_t* __restrict__ scales,     // [K/group or 1, N]
    const int size_k, const int size_n, const int group_size);

__global__ void marlin_awq_zero_points_kernel(
    uint32_t* __restrict__ out,             // [K/group, N/pack_factor] Marlin zero layout
    const uint32_t* __restrict__ qzeros,    // [K/group, N/pack_factor] AWQ layout
    const int num_groups, const int size_n, const int num_bits);

/**
 * fp4_marlin_repack_kernel — packed e2m1 weights + MX(UE8M0)/NVFP4(E4M3) scales -> Marlin FP4
 * (weight-only FP4 on SM80/SM89/SM90 where native block-scaled FP4 MMA is absent).
 * Models : GPT-OSS MXFP4, Kimi-K3 MXFP4, DeepSeek-V4 FP4 experts, NVFP4 ModelOpt checkpoints.
 */
__global__ void fp4_marlin_repack_kernel(
    uint32_t* __restrict__ out_weight,      // Marlin layout
    void* __restrict__ out_scales,          // Marlin-permuted scales (bf16/fp16 or e8m0 bits)
    const uint8_t* __restrict__ weight,     // [N, K/2] packed e2m1 (row = output channel)
    const void* __restrict__ scales,        // [N, K/block] UE8M0 (block 32) or E4M3 (block 16)
    const int size_k, const int size_n, const int block_size, const bool scale_is_ue8m0);

/**
 * fp4_backend_shuffle_kernel — reorder FP4/FP8 expert weights + scales into a backend
 * layout (TRTLLM-gen shuffled, CUTLASS SM100 interleaved, SM120 layout). Per expert.
 */
__global__ void fp4_backend_shuffle_kernel(
    uint8_t* __restrict__ out_weight,       // [E, N, K/2] target layout
    uint8_t* __restrict__ out_scales,       // [E, ...] swizzled scales
    const uint8_t* __restrict__ weight,     // [E, N, K/2]
    const uint8_t* __restrict__ scales,     // [E, N, K/block]
    const int num_experts, const int size_n, const int size_k,
    const int block_size, const int target_layout);  // WeightPacking value of the target

// NVFP4 / MX scale swizzling: see gemm/gemm_prep_kernels.cuh::block_scale_swizzle_kernel.

/**
 * fp8_block_scale_to_ue8m0_kernel — re-encode DeepSeek-style 128x128 fp32 weight scales as
 * UE8M0 (rounding scale up and requantizing the FP8 payload so q*s is preserved within one
 * FP8 ulp). Required by DeepGEMM sm_100 recipes (DeepSeek-V3.1+, Kimi K2 on Blackwell).
 */
__global__ void fp8_block_scale_to_ue8m0_kernel(
    f8e4m3_t* __restrict__ weight,          // [N, K] in/out (requantized in place)
    float* __restrict__ scales_inout,       // [N/128, K/128] in fp32; out: power-of-two values
    uint8_t* __restrict__ ue8m0_out,        // [N/128, K/128] UE8M0 bits (optional, may be null)
    const int size_n, const int size_k);

/**
 * smoothquant_fold_kernel — Q04: fold smoothing factors s[k] into the preceding norm and the
 * linear: norm_w[k] /= s[k]; W[n, k] *= s[k]. Exact up to fp rounding; afterwards the GEMM sees
 * smoothed activations without a runtime division. s is produced by offline calibration.
 */
template <typename scalar_t>
__global__ void smoothquant_fold_kernel(
    scalar_t* __restrict__ norm_weight,     // [K] in/out (RMSNorm/LayerNorm gamma preceding the linear)
    scalar_t* __restrict__ norm_bias,       // [K] in/out (LayerNorm beta) or null
    scalar_t* __restrict__ linear_weight,   // [N, K] in/out (all linears sharing this input: q,k,v / gate,up)
    const int64_t linear_weight_stride,
    const float* __restrict__ smooth,       // [K] s_k = max|X_k|^a / max|W_k|^(1-a)
    const int size_n, const int size_k);

/**
 * smoothquant_compute_scales_kernel — s_k = act_absmax_k^alpha / w_absmax_k^(1-alpha), clamped >= 1e-5.
 * Inputs are calibration statistics collected outside the library.
 */
__global__ void smoothquant_compute_scales_kernel(
    float* __restrict__ smooth,             // [K]
    const float* __restrict__ act_absmax,   // [K]
    const float* __restrict__ weight_absmax,// [K] (max over all fused linears)
    const float alpha, const int size_k);

/**
 * gguf_dequantize_kernel — ggml block formats -> dense out_t. One CTA per 256-element super-block
 * group. All GGUFType values except IQ1_M are supported in one kernel family.
 * Models : any GGUF checkpoint (llama.cpp ecosystem), Q4_K_M / Q5_K_M / Q6_K / Q8_0 / IQ4_XS common.
 */
template <typename out_t, GGUFType kType>
__global__ void gguf_dequantize_kernel(
    out_t* __restrict__ out,                // [N, K]
    const uint8_t* __restrict__ qweight,    // raw ggml blocks for an [N, K] matrix
    const int64_t size_n, const int64_t size_k);

}  // namespace kops::quant
