#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/gemm_prep_kernels.cuh              Catalog: G03-G06 (operand prep)
// Operand / scale preparation consumed by the GEMM families. Two classes:
//   load-time (once, outside capture): weight transpose to K-major, GLU interleave,
//     Machete prepack, router weight split, colsum for INT8 azp, block-scale swizzles
//     of weights, FP8 2-D block quantization of weights.
//   per-step (inside graphs, only when the producer did not already emit the
//     required layout): activation scale layout transforms, UE8M0 packing.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/**
 * weight_transpose_kernel — [K, N] (MN-major, e.g. GPT-2 Conv1D, some exported checkpoints) ->
 * [N, K] K-major, any element size incl. packed fp4/int4 (nibble-aware). Required by FP8/INT8/FP4
 * tensor-core paths, which accept K-major operands only.
 */
__global__ void weight_transpose_kernel(
    void* __restrict__ out,                 // [N, K]
    const void* __restrict__ in,            // [K, N]
    const int64_t rows_k, const int64_t cols_n, const int element_bits);

/**
 * glu_weight_interleave_kernel — reorder a fused [2I, K] gate|up weight (and its scales / bias) so
 * rows alternate gate/up in blocks of `interleave` rows. With interleave == tile_n / 2 a single
 * N tile holds matching gate and up columns and the GLU (+ quant) epilogue writes [M, I] directly.
 * Also converts between GPT-OSS row-interleaved (interleave = 1) and halves layouts.
 */
__global__ void glu_weight_interleave_kernel(
    void* __restrict__ out_weight,          // [2I, K] (or [E, 2I, K])
    void* __restrict__ out_scales,          // matching scale rows or null
    void* __restrict__ out_bias,            // [2I] (or [E, 2I]) or null
    const void* __restrict__ weight,
    const void* __restrict__ scales,
    const void* __restrict__ bias,
    const int64_t row_bytes, const int64_t scale_row_bytes, const int bias_elem_bytes,
    const int intermediate, const int num_experts,
    const int src_interleave,               // 0 = [gate | up] halves, 1 = GPT-OSS alternating rows
    const int dst_interleave);

/**
 * Layout atom of a Machete prepacked weight (machete_prepack_kernel / gemm::machete_prepack).
 * The prepacked block layout is derived from the MMA atom the Machete mainloop issues
 * (machete_prepacked_layout.cuh: PrepackedLayoutBTemplate::PPBlockShape_NK = 128 x 64, thrfrg_A of
 * TiledMma, plus the FrgV interleave that W4A16 uses for its prmt upconvert), so the atom must be
 * named by the caller. Values are ABI: append only.
 */
enum class MacheteAtomLayout : int32_t {
  kInt4Mma16 = 0,   // W4A16: 16-bit MMA atom (wgmma k16), packed int4, interleaved 32-element value blocks
  kInt8Mma16 = 1,   // W8A16: 16-bit MMA atom, int8 weights, plain value order
  kInt4Mma8 = 2,    // W4A8: 8-bit MMA atom (k32), packed int4, plain value order
  kInt8Mma8 = 3,    // W8A8: 8-bit MMA atom (k32), int8 weights, plain value order
};

/**
 * machete_prepack_kernel — pack (u)int4 / (u)int8 weights [N, K] into the Machete (CUTLASS mixed-input
 * SM90) layout for a given atom shape. `out_weight` holds the same N * K elements as `weight`, permuted:
 * blocks of 128 x 64 elements, block (n_tile, k_tile) in column-major order over (n_tile, k_tile), each
 * block permuted by the atom's thrfrg/value-order layout (see gemm_prep.cu for the exact bit fields).
 * N must be a multiple of 128 and K a multiple of 64 (PPBlockShape_NK).
 * Requires atom_layout_id to agree with weight_bits: kInt4Mma16 / kInt4Mma8 -> 4, kInt8Mma16 /
 * kInt8Mma8 -> 8.
 */
__global__ void machete_prepack_kernel(
    void* __restrict__ out_weight,
    const void* __restrict__ weight,        // [N, K] packed low-bit, K-major
    const int size_n, const int size_k, const int weight_bits, const int atom_layout_id);

/// router_weight_split_bf16x2_kernel — fp32 W -> (W_hi = bf16(W), W_lo = bf16((W - W_hi) * 256)).
__global__ void router_weight_split_bf16x2_kernel(
    bf16_t* __restrict__ w_hi, bf16_t* __restrict__ w_lo,
    const float* __restrict__ w_fp32, const int64_t numel);

/// int8_weight_colsum_kernel — colsum_b[n] = sum_k qB[n, k] for the asymmetric-activation correction.
__global__ void int8_weight_colsum_kernel(
    int32_t* __restrict__ colsum, const int8_t* __restrict__ weight, const int size_n, const int size_k);

/**
 * block_scale_swizzle_kernel — row-major block scales [rows, cols/block] (UE8M0 for MX, E4M3 for
 * NVFP4) -> 128x4 swizzled tile layout used by tcgen05 / SM120 block-scaled MMA (and CUTLASS
 * SfAtom), zero-padded to (128, 4). Works per batch / per expert. Inverse when `unswizzle`.
 */
__global__ void block_scale_swizzle_kernel(
    uint8_t* __restrict__ out,              // [batch, round_up(rows,128), round_up(cols_blocks,4)]
    const uint8_t* __restrict__ in,         // [batch, rows, cols_blocks]
    const int batch, const int rows, const int cols_blocks, const bool unswizzle);

/**
 * fp8_scale_to_tma_layout_kernel — activation 1x128 scales fp32 [M, K/128] row-major ->
 * [K/128, round_up(M, 4)] column-major (TMA-aligned) as required by blockwise FP8 WGMMA/tcgen05
 * kernels; optional UE8M0 packing (4 exponents per int32 along K). Grouped variant honours
 * expert_offsets / masked_m so padded rows get zero scales.
 */
__global__ void fp8_scale_to_tma_layout_kernel(
    void* __restrict__ out,                 // fp32 [K/128, M_pad] or int32 packed UE8M0 [K/512, M_pad]
    const float* __restrict__ in,           // [M, K/128]
    const int64_t* __restrict__ expert_offsets,  // [G + 1] or null
    const int m, const int k_blocks, const int m_pad, const bool pack_ue8m0);

/**
 * fp8_block2d_quant_weight_kernel — offline FP8 quantization of a bf16/fp32 weight with 128x128
 * (or block_n x block_k) blocks: s = amax / 448 (optionally rounded up to a power of two), q = sat(w / s).
 * Produces DeepSeek-format FP8 weights from bf16 checkpoints (e.g. self-quantized MTP heads, LoRA merges).
 */
template <typename in_t>
__global__ void fp8_block2d_quant_weight_kernel(
    f8e4m3_t* __restrict__ out,             // [N, K]
    float* __restrict__ scales,             // [N/block_n, K/block_k]
    const in_t* __restrict__ weight,        // [N, K]
    const int size_n, const int size_k, const int block_n, const int block_k, const bool ue8m0);

}  // namespace kops::gemm
