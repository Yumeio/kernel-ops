#pragma once
// =============================================================================
// kernel-ops | csrc/activation/activation_kernels.cuh                Catalog: F01
// Plain activations, gated (GLU) activations, gate-multiply and fused
// activation + quantization (incl. masked MoE layout).
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh, common/quant.cuh.
// Launch: grid = (ceil(d / (VEC_SIZE * 1024)), num_tokens), block = 256, 4 packs per
// thread; the input is read with ld.global.cs and the output written with
// st.global.cs (elementwise results are never re-read). act_and_mul_quant uses
// grid = ceil(rows / kRows) with a (256, kRows) block, kRows = 2 for the dynamic
// per-token mode.
// =============================================================================

namespace kops::act {

/**
 * act_kernel — out = act(x) elementwise (non-gated MLPs).
 * Models : GPT-2 / GPT-J / NeoX / StarCoder / Phi-2 (kGeLUTanh), Falcon / BLOOM (kGeLUErf / kGeLUTanh),
 *          OPT (kReLU), Nemotron-4/H, Nemotron-3 MoE experts (kReLU2), vision towers (kQuickGeLU).
 * Bias + activation inside the up-projection GEMM epilogue (G07) is preferred when available.
 */
template <typename scalar_t, ActFn kAct, int VEC_SIZE>
__global__ void act_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const int64_t out_stride,
    const scalar_t* __restrict__ input,  // [num_tokens, d]
    const int64_t input_stride,
    const GluParams params,              // only fatrelu_threshold / situ_beta used
    const int num_tokens, const int d);

/**
 * act_and_mul_kernel — gated linear unit on a fused gate/up projection.
 *   out[t, i] = row_weight[t] * GLU(gate[t, i], up[t, i])   (see GluParams for exact formulas)
 *
 * Models : SwiGLU (LLaMA, Mistral, Qwen, DeepSeek, Kimi K2, GLM, MiniMax, Phi-3+),
 *          GeGLU tanh (Gemma 1-3), GeGLU erf, ReGLU, GPT-OSS clamped SwiGLU (kInterleaved),
 *          DeepSeek-V4 clamped SwiGLU, Kimi-K3 SiTU-GLU (kSiTU + situ_linear_beta).
 * row_weight: optional per-row multiplier (MoE routing weight applied in the activation —
 *          Kimi-K3 "weighted SiTU"; Llama-4 applies router weights on the input instead).
 * Precision: fp32 math; Kimi-K3 SiTU must be fp32 (bf16 rounding near tanh saturation is visible).
 */
template <typename scalar_t, ActFn kAct, GateLayout kLayout, int VEC_SIZE>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const int64_t out_stride,
    const scalar_t* __restrict__ input,  // [num_tokens, 2 * d] (layout per kLayout)
    const int64_t input_stride,
    const float* __restrict__ row_weight,// [num_tokens] or null
    const GluParams params,
    const int num_tokens, const int d);

/// Arguments of act_and_mul_quant_kernel.
struct ActMulQuantParams {
  const void* input;            // [rows, 2*d] contiguous rows (or masked [E, max_m, 2*d])
  int64_t input_stride;         // row stride in elements
  const float* row_weight;      // [rows] optional routing weight
  GluParams glu;
  ActQuantOut quant;            // FP8 per-token / per-group(128) / MXFP8 / MXFP4 / NVFP4 / INT8
  // Masked (DeepEP low-latency) layout: rows of expert e are [e*max_m, e*max_m + masked_m[e]).
  const int32_t* masked_m;      // [num_experts] valid rows per expert, null = dense
  int32_t max_m;                // rows reserved per expert in the masked layout
  int32_t num_experts;
  // Contiguous-grouped layout: expert_offsets[e]..expert_offsets[e+1]; rows beyond
  // expert_offsets[E] (padding) are skipped and their scales zeroed.
  const int64_t* expert_offsets;// [num_experts + 1] or null
  int32_t num_rows;
  int32_t d;
};

/**
 * act_and_mul_quant_kernel — GLU + quantization of the result for the down projection.
 * Models : DeepSeek-V3/V3.2/V4 FP8 block (kPerGroup, UE8M0 on Blackwell), Kimi-K2 FP8,
 *          Kimi-K3 MXFP8 activations, GPT-OSS / Qwen3 NVFP4/MXFP4 MoE, Llama-4 FP8.
 * Skips padded / masked rows (no NaN from garbage), matching HPC-Ops masked act_mul_and_quant.
 */
template <typename scalar_t, ActFn kAct, GateLayout kLayout, ActQuantMode kMode, int VEC_SIZE,
          int kRows = 1>
__global__ void act_and_mul_quant_kernel(const __grid_constant__ ActMulQuantParams params);

/**
 * gate_mul_kernel — out = x * act(gate) with broadcast of the gate.
 *   gate_width == d             : elementwise (Qwen3-Next / Qwen3.5 gated attention output,
 *                                 Kimi-K3 gated MLA output gate, KDA output gate)
 *   gate_width == num_heads     : one gate per head, broadcast over head_dim (head-wise gating)
 *   gate_width == 1             : one gate per token (Qwen2-MoE / Qwen3-Next shared_expert_gate)
 * Optional accumulate: out = accumulate_into + x * act(gate) (shared expert added to routed sum).
 */
template <typename scalar_t, ActFn kGateAct, int VEC_SIZE>
__global__ void gate_mul_kernel(
    scalar_t* __restrict__ out,                   // [num_tokens, d]
    const int64_t out_stride,
    const scalar_t* __restrict__ x,               // [num_tokens, d]
    const int64_t x_stride,
    const scalar_t* __restrict__ gate,            // [num_tokens, gate_width]
    const int64_t gate_stride,
    const scalar_t* __restrict__ accumulate_into, // [num_tokens, d] or null
    const int64_t accumulate_stride,
    const int num_tokens, const int d, const int gate_width);

}  // namespace kops::act
