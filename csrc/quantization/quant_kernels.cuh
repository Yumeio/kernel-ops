#pragma once
// =============================================================================
// kernel-ops | csrc/quantization/quant_kernels.cuh                    Catalog: Q01
// Activation / tensor quantization primitives: absmax, per-tensor, per-token,
// per-group (DeepSeek 1x128, UE8M0), MXFP8/MXFP4, NVFP4, dequantize, int4
// pack/unpack and online Hadamard rotation.
// Requires: common/macros.cuh, common/dtype.cuh, common/quant.cuh.
// Scale convention: dequant multiplier (x ~= q * s). See common/quant.cuh.
// =============================================================================

namespace kops::quant {

/**
 * absmax_reduce_kernel — amax[0] = max(amax[0], max|x|) via atomicMax on the float bit pattern.
 * Caller zeroes amax beforehand (or uses epoch-tagged buffers). Used for dynamic per-tensor FP8
 * and for NVFP4 global scale (global = 448 * 6 / amax).
 * Launch : grid-stride, grid = min(num_sms * 4, ceil(n / (VEC * 256))), block = 256.
 */
template <typename scalar_t, int VEC_SIZE>
__global__ void absmax_reduce_kernel(
    float* __restrict__ amax,             // [1] in/out
    const scalar_t* __restrict__ input,   // [num_rows, cols]
    const int64_t input_row_stride,
    const int64_t num_rows, const int64_t cols);

/**
 * quantize_per_tensor_kernel — q = sat(x / s) with s = *scale (dequant multiplier).
 * out_t: f8e4m3_t | f8e5m2_t | int8_t.  Pre-quantization used for static FP8 W8A8 and FP8 KV.
 */
template <typename scalar_t, typename out_t, int VEC_SIZE>
__global__ void quantize_per_tensor_kernel(
    out_t* __restrict__ out,               // [num_rows, cols]
    const int64_t out_row_stride,
    const scalar_t* __restrict__ input,    // [num_rows, cols]
    const int64_t input_row_stride,
    const float* __restrict__ scale,       // [1]
    const int64_t num_rows, const int64_t cols);

/**
 * quantize_per_token_kernel — one scale per row.
 *   symmetric : s_r = max(|x_r|) / qmax (FP8: clamped by scale_ub, min 1/(qmax*512)); q = sat(x/s)
 *   asymmetric (int8 only): s_r = (max - min)/255, azp_r = round(-128 - min/s_r); q = sat(x/s + azp)
 * Models : FP8-dynamic checkpoints (Llama 3.x FP8, Qwen FP8 dynamic), SmoothQuant W8A8 INT8.
 */
template <typename scalar_t, typename out_t, int VEC_SIZE, bool kAsymmetric>
__global__ void quantize_per_token_kernel(
    out_t* __restrict__ out,               // [num_rows, cols]
    const int64_t out_row_stride,
    float* __restrict__ scales,            // [num_rows]
    int32_t* __restrict__ azp,             // [num_rows] if kAsymmetric else null
    const scalar_t* __restrict__ input,    // [num_rows, cols]
    const int64_t input_row_stride,
    const float* __restrict__ scale_ub,    // [1] or null
    const int num_rows, const int cols);

/**
 * quantize_per_group_kernel — 1 x GROUP_SIZE groups along the last dim.
 *   s_{r,g} = max|x| / qmax; if kUE8M0: s rounded UP to a power of two (stored as fp32 or packed UE8M0)
 *   kColMajorScale: scales laid out [cols/GROUP, round_up(rows, 4)] for DeepGEMM / TMA.
 * Masked layout support: when masked_m != null input is [E, max_m, cols] and rows
 *   >= masked_m[e] are skipped (DeepEP low-latency receive buffers).
 * Models : DeepSeek V3/V3.1/V3.2/V4, Kimi K2 FP8, Qwen3 FP8 block, GLM-4.5 FP8, MiniMax-M2 FP8.
 */
template <typename scalar_t, typename out_t, int GROUP_SIZE, bool kColMajorScale, bool kUE8M0>
__global__ void quantize_per_group_kernel(
    out_t* __restrict__ out,               // [num_rows, cols]
    const int64_t out_row_stride,
    void* __restrict__ scales,             // fp32 or packed UE8M0 (see ScaleLayout)
    const int64_t scale_stride,            // row stride (row-major) or padded rows (col-major)
    const scalar_t* __restrict__ input,    // [num_rows, cols] or [E, max_m, cols]
    const int64_t input_row_stride,
    const int32_t* __restrict__ masked_m,  // [E] or null
    const int max_m,
    const int num_rows, const int cols,
    const float eps);                      // lower bound on amax (avoid 0 scales), e.g. 1e-10

/**
 * quantize_mx_kernel — OCP MX block quantization, 32 elements per UE8M0 scale.
 *   e = ceil(log2(amax / elem_max)) (or floor with saturation when kRoundDown), q = sat(x * 2^-e)
 * out_t: f8e4m3_t (MXFP8) | f4e2m1x2_t (MXFP4, packed). Scale layout: row-major or 128x4 swizzled.
 * Models : Kimi-K3 (MXFP4 weights / MXFP8 activations), GPT-OSS MXFP4 experts, DeepSeek-V4 FP4 paths
 *          on Blackwell, any MX-format QAT checkpoint.
 */
template <typename scalar_t, typename out_t, ScaleLayout kScaleLayout, bool kRoundDown>
__global__ void quantize_mx_kernel(
    out_t* __restrict__ out,               // [num_rows, cols] (cols/2 bytes for fp4)
    const int64_t out_row_stride,
    uint8_t* __restrict__ scales,          // UE8M0, [num_rows, cols/32] or swizzled
    const scalar_t* __restrict__ input,
    const int64_t input_row_stride,
    const int num_rows, const int cols);

/**
 * quantize_nvfp4_kernel — NVFP4: 16-element blocks, E4M3 block scale, fp32 global scale.
 *   s_b = e4m3(amax_b / 6 * global_scale); q = sat(x * global_scale / s_b) in e2m1
 *   (global_scale = 448 * 6 / tensor_amax, supplied by caller or computed by absmax_reduce_kernel)
 * Scale layout: kSwizzled128x4 for CUTLASS/tcgen05 sm_100a and sm_120a block-scaled GEMM.
 */
template <typename scalar_t, ScaleLayout kScaleLayout>
__global__ void quantize_nvfp4_kernel(
    uint8_t* __restrict__ out,             // [num_rows, cols/2] packed e2m1
    const int64_t out_row_stride,
    f8e4m3_t* __restrict__ block_scales,   // [round_up(rows,128), round_up(cols/16,4)] swizzled
    const scalar_t* __restrict__ input,
    const int64_t input_row_stride,
    const float* __restrict__ global_scale,// [1]
    const int num_rows, const int cols);

/**
 * dequantize_kernel — out = (q - z) * s for any supported QuantScheme (reference / fallback /
 * offline conversion). Not intended for hot paths; GEMMs dequantize in registers.
 */
template <typename out_t, ScaleGranularity kGranularity>
__global__ void dequantize_kernel(
    out_t* __restrict__ out,               // [num_rows, cols]
    const int64_t out_row_stride,
    const void* __restrict__ q,            // payload per scheme.storage_dtype / packing
    const int64_t q_row_stride,
    const void* __restrict__ scales,
    const void* __restrict__ zero_points,
    const QuantScheme scheme,
    const int64_t num_rows, const int64_t cols);

/**
 * pack_int4_kernel / unpack_int4_kernel — int8 values in [-8, 7] (or [0, 15]) <-> packed nibbles,
 * element 2i in the low nibble. Used by RTN tooling, KV int4 codecs and checkpoint conversion.
 */
__global__ void pack_int4_kernel(
    uint8_t* __restrict__ out,             // [num_rows, cols/2]
    const int8_t* __restrict__ input,      // [num_rows, cols]
    const int64_t num_rows, const int64_t cols, const bool is_unsigned);

__global__ void unpack_int4_kernel(
    int8_t* __restrict__ out,              // [num_rows, cols]
    const uint8_t* __restrict__ input,     // [num_rows, cols/2]
    const int64_t num_rows, const int64_t cols, const bool is_unsigned);

/**
 * hadamard_transform_kernel — y = scale * H_n x along the last dim (n = 2^LOG_N, up to 2^15),
 * optionally with a signed diagonal from transform_seed. Online rotation for QuaRot/SpinQuant
 * style W4A4/W4A8, Hadamard-rotated FP4 activations and rotated KV codecs (catalog A09).
 * Non power-of-two sizes use the 12/20/28/40 x 2^k Sylvester-Paley factorization.
 */
template <typename scalar_t, int LOG_N, bool kSigned>
__global__ void hadamard_transform_kernel(
    scalar_t* __restrict__ out,            // [num_rows, n] (may alias input)
    const int64_t out_row_stride,
    const scalar_t* __restrict__ input,
    const int64_t input_row_stride,
    const float scale,                     // 1/sqrt(n) for orthonormal
    const uint32_t transform_seed,
    const int num_rows);

}  // namespace kops::quant
