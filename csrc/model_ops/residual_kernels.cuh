#pragma once
// =============================================================================
// kernel-ops | csrc/model_ops/residual_kernels.cuh                     Catalog: X02
// Small elementwise glue that otherwise becomes un-fused framework ops.
// Requires: common/*.
// =============================================================================

namespace kops::model_ops {

/**
 * residual_add_kernel — out = a_scale * a + b_scale * b (+ c_scale * c)
 * Models : parallel residual h = x + attn(ln(x)) + mlp(ln(x)) (GPT-J, GPT-NeoX/Pythia, Falcon,
 *          Phi-1/2, Cohere Command-R, StableLM); Granite / MiniCPM residual multipliers;
 *          MoE shared + routed expert sum; Arctic dense + MoE residual.
 */
template <typename out_t, typename in_t, int VEC_SIZE, bool kHasC>
__global__ void residual_add_kernel(
    out_t* __restrict__ out,                     // [num_tokens, d] (may alias a)
    const int64_t out_stride,
    const in_t* __restrict__ a, const int64_t a_stride, const float a_scale,
    const in_t* __restrict__ b, const int64_t b_stride, const float b_scale,
    const in_t* __restrict__ c, const int64_t c_stride, const float c_scale,
    const int num_tokens, const int d);

/// cast_kernel — dtype conversion with optional scale (fp32 residual <-> bf16, fp8 dequant copies).
template <typename out_t, typename in_t, int VEC_SIZE>
__global__ void cast_kernel(
    out_t* __restrict__ out, const int64_t out_stride,
    const in_t* __restrict__ in, const int64_t in_stride,
    const float scale, const int num_rows, const int cols);

}  // namespace kops::model_ops
