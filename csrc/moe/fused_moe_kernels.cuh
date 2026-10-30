#pragma once
// =============================================================================
// kernel-ops | csrc/moe/fused_moe_kernels.cuh                    Catalog: M03, M04
// Expert FFN execution: block-sorted fused expert GEMM (Triton fused_moe
// semantics in CUDA), small-M MoE GEMV, latent-MoE projections (Kimi-K3 /
// NVIDIA LatentMoE) and experimental W2 experts.
// A "fused MoE" API may still be several launches (route -> permute -> W13 ->
// act/quant -> W2 -> reduce); the plan reports the exact launch count.
// Requires: common/*, gemm/gemm_params.cuh, gemm/grouped_gemm_kernels.cuh, moe/routing_kernels.cuh,
//           moe/permute_kernels.cuh.
// =============================================================================

namespace kops::moe {

enum class ExpertFFN : uint8_t {
  kGated = 0,     // down( act(gate) * up )           (SwiGLU / GeGLU / SiTU-GLU / clamped SwiGLU)
  kNonGated = 1,  // down( act(up) )                  (Nemotron-H / Nemotron-3 MoE relu^2, Switch-style GELU)
};

/// Arguments of one expert GEMM stage over block-sorted tokens.
struct FusedMoEGemmParams {
  const void* a;                       // stage W13: hidden [num_tokens, H]; stage W2: intermediate [rows, I]
  int64_t lda;
  bool a_rows_are_slots;               // true: A row = sorted_token_ids[i] (stage W2), false: slot / topk (stage W13)
  const void* a_scale;                 // activation scales (per-token / per-group / MX / NVFP4)
  const void* w;                       // [E_local, N, K] expert weights (bf16 / fp8 / int4 / fp4 packed)
  int64_t w_expert_stride;
  const void* w_scale;                 // per expert scales (per-channel / 128x128 block / group / MX / NVFP4)
  int64_t w_scale_expert_stride;
  const void* w_zeros;                 // int4 asymmetric zeros or null
  const void* bias;                    // [E_local, N] (GPT-OSS expert biases) or null
  void* c;                             // stage W13: [rows, N]; stage W2: [rows, H] (or scattered to [T, K, H])
  int64_t ldc;
  const int32_t* sorted_token_ids;     // from moe_align_block_size_kernel
  const int32_t* expert_ids;           // [num_blocks]
  const int32_t* num_tokens_post_pad;  // [1]
  const float* topk_weights;           // [num_tokens * topk] (applied when mul_routed_weight)
  bool mul_routed_weight;
  int32_t top_k;
  int32_t n, k;                        // N = 2I (W13, gated) | I (W13, non-gated) | H (W2)
  int64_t num_valid_slots;             // num_tokens * topk
  QuantScheme a_quant, w_quant;
  gemm::EpilogueParams epi;            // act/GLU fused into W13 epilogue when profitable
};

/**
 * fused_moe_gemm_kernel — one expert GEMM stage: each CTA takes a BLOCK_M block of sorted slots
 * belonging to one expert, gathers A rows by slot id, multiplies by that expert's weight tile.
 * Precision variants: BF16, FP8 per-tensor, FP8 1x128/128x128 block, INT8 W8A8, W4A16 (int4 /
 * Marlin path), MXFP4 / NVFP4 (sm_100a / sm_120a native, else Marlin FP4).
 * Models : all MoE families; default path when no faster grouped backend applies.
 */
template <typename a_t, typename w_t, typename c_t, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES,
          typename ArchTag>
__global__ void fused_moe_gemm_kernel(const KOPS_GRID_CONST FusedMoEGemmParams params);

/**
 * moe_gemv_kernel — decode MoE (num_tokens * topk small): one CTA per (slot, N-tile) streams the
 * selected expert's rows once. Weight bandwidth bound; supports bf16 / fp8 block / int4 / fp4.
 */
template <typename a_t, typename w_t, typename c_t, int SLOTS_PER_CTA, int ROWS_PER_WARP, int WARPS>
__global__ void moe_gemv_kernel(const KOPS_GRID_CONST FusedMoEGemmParams params);

/// Whole-layer description consumed by the host orchestrator moe_forward() in ops.h.
struct MoELayerParams {
  // Inputs
  const void* hidden;                  // [num_tokens, H]
  int64_t hidden_stride;
  int32_t num_tokens, hidden_size;
  // Routing (precomputed by the caller or produced in-layer)
  RouterParams router;                 // if router.topk_ids already filled and routes_precomputed
  bool routes_precomputed;
  const void* router_weight;           // [E, H] when routing in-layer
  const void* router_bias;             // [E] or null
  // Experts
  int32_t num_local_experts;
  int32_t intermediate_size;           // per expert I (Kimi-K3: 3072; DSv3: 2048; Qwen3-235B: 1536)
  ExpertFFN ffn;
  ActFn act;
  GateLayout gate_layout;
  GluParams glu;
  const void* w13;                     // [E_local, 2I (or I), K_in]
  const void* w13_scale;
  const void* w13_bias;
  const void* w2;                      // [E_local, K_in, I]
  const void* w2_scale;
  const void* w2_bias;
  QuantScheme act_quant, w13_quant, w2_quant;
  gemm::GemmRecipe recipe;             // expert GEMM recipe (kBF16, kFP8Blockwise, kMXFP4xMXFP8, kNVFP4, kW4A16Int ...)
  gemm::GroupedLayout grouped_layout;  // kContiguous (prefill) | kMasked (EP low-latency) | kIndexed (no permute copy)
  bool fuse_glu_into_w13;              // GLU (+quant) in the W13 epilogue (requires interleaved W13)
  bool fuse_finalize_into_w2;          // weighted scatter in the W2 epilogue (non-deterministic atomics)
  // Latent MoE (Kimi-K3 Stable LatentMoE, NVIDIA LatentMoE):
  //   u = x . W_down^T (H -> L); experts operate on u (K_in = L); y = RMSNorm(sum_k w_k E_k(u)) . W_up^T (L -> H)
  int32_t latent_dim;                  // 0 = classic MoE (K_in = H); Kimi-K3: 3584 (H = 7168)
  const void* latent_down;             // [L, H]
  const void* latent_up;               // [H, L]
  const void* latent_norm_weight;      // [L] RMSNorm before up-projection (latent_moe_use_norm)
  float latent_norm_eps;
  // Shared experts (computed by the caller as a dense MLP, or fused as pseudo-experts via router)
  const void* shared_out;              // [num_tokens, H] added in the final reduce, or null
  // Output
  void* out;                           // [num_tokens, H]
  int64_t out_stride;
  float routed_scaling_factor;
};

/**
 * latent_moe_combine_norm_kernel — Kimi-K3 Stable LatentMoE tail: weighted sum of expert outputs
 * in latent space, then RMSNorm (weight [L]) and optional quantization for the up-projection.
 *   z[t] = RMSNorm( sum_k w[t,k] * expert_out[row(t,k)] )   (fp32 accumulation)
 */
template <typename scalar_t, ActQuantMode kQuant, int VEC_SIZE>
__global__ void latent_moe_combine_norm_kernel(
    void* __restrict__ out,                       // [num_tokens, L] (scalar_t, or quantized payload)
    const ActQuantOut quant,
    const scalar_t* __restrict__ expert_out,      // [num_rows_padded, L]
    const int32_t* __restrict__ slot_to_permuted_row,  // [num_tokens * topk]
    const float* __restrict__ topk_weights,       // [num_tokens, topk]
    const scalar_t* __restrict__ norm_weight,     // [L]
    const float routed_scale, const float epsilon,
    const int num_tokens, const int topk, const int latent_dim);

// ---- M04 experimental: 2-bit experts (attention.rs moe_w2) -----------------------
/**
 * moe_w2_gemm_kernel — EXPERIMENTAL (catalog M04, status R). 2-bit packed expert weights with
 * per-group scales, unpacked to FP8/BF16 in registers. Opt-in only; quality gate first.
 */
template <typename a_t, typename c_t, int BLOCK_M, int BLOCK_N, int GROUP_SIZE>
__global__ void moe_w2_gemm_kernel(const KOPS_GRID_CONST FusedMoEGemmParams params);

/// Pack an MXFP4 / bf16 expert tensor into the experimental W2 layout (offline).
__global__ void moe_w2_pack_kernel(
    uint8_t* __restrict__ out,                    // [E, N, K/4] 2-bit packed
    void* __restrict__ out_scales,                // [E, N, K/GROUP] scales
    const void* __restrict__ weight,              // [E, N, K] source
    const int source_format,                      // WeightPacking value of the source
    const int num_experts, const int size_n, const int size_k, const int group_size);

}  // namespace kops::moe
