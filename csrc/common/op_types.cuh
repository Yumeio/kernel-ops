#pragma once
// =============================================================================
// kernel-ops | csrc/common/op_types.cuh
// Enums/structs shared across families (activation fns, GLU params, norm type,
// RoPE style). Kept here so gemm/moe/mega epilogues do not depend on family headers.
// Requires: common/macros.cuh.
// =============================================================================

namespace kops {

/// Elementwise activation functions. Exact formulas (fp32 math):
///   kSiLU        x * sigmoid(alpha * x)                 (alpha = GluParams::gate_alpha, 1 by default)
///   kGeLUErf     0.5 x (1 + erf(x / sqrt2))             (BERT, Gemma "gelu", Falcon)
///   kGeLUTanh    0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))  (GPT-2 gelu_new, Gemma gelu_pytorch_tanh)
///   kQuickGeLU   x * sigmoid(1.702 x)                   (CLIP / vision towers)
///   kReLU        max(x, 0)                              (OPT)
///   kReLU2       max(x, 0)^2                            (Nemotron-4/H, Nemotron-3 MoE, Primer)
///   kSigmoid     1 / (1 + exp(-x))                      (gates: Qwen3-Next, Kimi gated MLA/KDA)
///   kTanh        tanh(x)
///   kSoftplus    log(1 + exp(x)), linear above threshold 20
///   kSiTU        beta * tanh(x / beta) * sigmoid(x)     (Kimi-K3 SiTU gate branch, beta = situ_beta)
///   kFatReLU     x if x > threshold else 0
///   kIdentity    x
enum class ActFn : uint8_t {
  kIdentity = 0, kSiLU = 1, kGeLUErf = 2, kGeLUTanh = 3, kQuickGeLU = 4, kReLU = 5,
  kReLU2 = 6, kSigmoid = 7, kTanh = 8, kSoftplus = 9, kSiTU = 10, kFatReLU = 11,
};

/// How gate and up halves are laid out in a fused [*, 2*I] projection output.
enum class GateLayout : uint8_t {
  kGateUp = 0,       // [gate(0..I) | up(I..2I)]  (vLLM/HF merged gate_up_proj)
  kUpGate = 1,       // [up | gate]
  kInterleaved = 2,  // gate = x[2i], up = x[2i+1]  (GPT-OSS checkpoint layout)
};

/// Parameters of a gated linear unit out = f(gate) * g(up).
///   generic        : out = act(gate) * up
///   GPT-OSS        : gate = min(gate, limit); up = clamp(up, -limit, limit);
///                    out = gate * sigmoid(1.702 gate) * (up + 1)         (gate_alpha=1.702, up_offset=1, limit=7)
///   DeepSeek-V4    : SwiGLU with clamping (gate_clamp_max / up_clamp from config)
///   Kimi-K3 SiTU-GLU: out = b1 tanh(g/b1) sigmoid(g) * b2 tanh(u/b2)     (b1=4, b2=25, fp32 math)
struct GluParams {
  float gate_alpha = 1.0f;        // sigmoid sharpness inside SiLU
  float gate_clamp_max = 0.0f;    // >0: gate = min(gate, gate_clamp_max) before activation
  float up_clamp = 0.0f;          // >0: up = clamp(up, -up_clamp, up_clamp)
  float up_offset = 0.0f;         // added to up after clamp (GPT-OSS: 1.0)
  float situ_beta = 4.0f;         // kSiTU gate soft-cap
  float situ_linear_beta = 0.0f;  // >0: up = situ_linear_beta * tanh(up / situ_linear_beta) (Kimi-K3: 25)
  float fatrelu_threshold = 0.0f;
};

/// Normalization flavour for fused kernels.
enum class NormType : uint8_t {
  kRMSNorm = 0,     // x * rsqrt(mean(x^2) + eps) * (w + weight_offset)
  kLayerNorm = 1,   // (x - mean) * rsqrt(var + eps) * (w + weight_offset) + b
  kL2Norm = 2,      // x * rsqrt(sum(x^2) + eps)   (GDN/KDA q,k; no affine)
  kNone = 3,
};

/// RoPE pairing convention.
enum class RopeStyle : uint8_t {
  kNeoX = 0,  // rotate_half: pairs (i, i + rot_dim/2)  (LLaMA, Qwen, Mistral, GPT-NeoX)
  kGPTJ = 1,  // interleaved: pairs (2i, 2i+1)           (GPT-J, ChatGLM, DeepSeek MLA rope part, GLM-4)
};

}  // namespace kops
