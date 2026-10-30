#pragma once
// =============================================================================
// kernel-ops | csrc/mega/mega_kernels.cuh                        Catalog: E04-E08
// Megakernel entry points. Each is ONE device kernel; zero/reset/copy work that
// remains outside is listed in MegaPlanInfo::num_device_launches.
// Requires: common/*, execution/persistent.cuh, gemm/gemm_params.cuh,
//           kvcache/kv_layout.cuh, attention/attention_params.cuh,
//           moe/routing_kernels.cuh, communication/ep_kernels.cuh, mega/mega_plan.cuh.
// =============================================================================

namespace kops::mega {

// ---- E04 dense MLP -----------------------------------------------------------------
struct MegaMLPParams {
  MegaPlanDevice plan;
  const void* x;                   // [M, H] (normed input; or raw when fuse_pre_norm)
  int64_t ldx;
  const void* pre_norm_weight;     // [H] fused RMSNorm before gate_up, or null
  float pre_norm_eps;
  const void* w13;                 // [2I, H] gate/up
  const void* w13_scale;
  const void* w2;                  // [H, I]
  const void* w2_scale;
  QuantScheme w_quant;
  ActFn act;
  GateLayout gate_layout;
  GluParams glu;
  void* intermediate;              // [M, I] slot (global workspace)
  void* out;                       // [M, H]
  int64_t ldo;
  const void* residual;            // fused residual add or null
  int32_t m, hidden_size, intermediate_size;
};

/**
 * mega_mlp_kernel — E04: gate_up tiles -> GLU -> down tiles in one persistent kernel. Down tiles
 * wait (epoch counters) for the I-range tiles they consume; final output published once all K
 * contributors finished. Target domain: small M decode (M <= 64), BF16 first, FP8 next.
 */
template <typename T, typename W, int BM, int BN1, int BN2, int BK, int STAGES, typename ArchTag>
__global__ void mega_mlp_kernel(const KOPS_GRID_CONST MegaMLPParams params);

// ---- E05 attention block (decode) ---------------------------------------------------
struct MegaAttentionBlockParams {
  MegaPlanDevice plan;
  const void* hidden;              // [M, H] residual stream input
  void* residual;                  // [M, H] in/out
  const void* in_norm_weight;      // [H]
  float norm_eps;
  const void* w_qkv;               // [(Hq + 2Hkv) * D, H]
  const void* qkv_bias;            // [...] or null (Qwen2)
  const void* q_norm_weight;       // per-head QK norm or null
  const void* k_norm_weight;
  const int64_t* positions;        // [M]
  const void* cos_sin_cache;
  int32_t rot_dim, rot_offset;
  RopeStyle rope_style;
  kv::PagedKVDesc kv;              // KV append + attention read (mutation declared: appended slots only)
  const int64_t* slot_mapping;     // [M]
  attn::AttentionModifiers mod;
  const void* w_o;                 // [H, Hq * D]
  const void* o_gate;              // gated attention output (Qwen3-Next / Kimi gated MLA style) or null
  void* out;                       // [M, H] (residual added when residual != null)
  int32_t m, hidden_size, num_q_heads, num_kv_heads, head_dim;
};

/**
 * mega_attention_block_kernel — E05 (P2/R): norm -> QKV -> QK-norm/RoPE -> KV append -> paged
 * decode attention -> O-proj (+residual). Plans may choose a sub-range of stages (e.g. QKV..append
 * only). KV pages written here must not be read by another concurrently running plan.
 */
template <typename T, int HEAD_DIM, int GQA_GROUP, int PAGE_SIZE, typename ArchTag>
__global__ void mega_attention_block_kernel(const KOPS_GRID_CONST MegaAttentionBlockParams params);

// ---- E06 local MoE -------------------------------------------------------------------
struct MegaMoELocalParams {
  MegaPlanDevice plan;
  const void* hidden;              // [M, H]
  // routes: precomputed (compute-only mode) or fused router (separate capability)
  const int32_t* topk_ids;         // [M, K] or null when fused routing
  const float* topk_weights;
  moe::RouterParams router;        // used when fused routing
  const void* router_weight;
  const void* w13;                 // [E_local, 2I, H]
  const void* w13_scale;
  const void* w2;                  // [E_local, H, I]
  const void* w2_scale;
  QuantScheme w_quant;
  ActFn act;
  GateLayout gate_layout;
  GluParams glu;
  int32_t latent_dim;              // Kimi-K3 LatentMoE (0 = off); down/up projections stay outside
  void* out;                       // [M, H]
  const void* shared_out;          // shared experts output to add, or null
  float routed_scaling_factor;
  int32_t m, hidden_size, intermediate_size, num_local_experts, topk;
};

/**
 * mega_moe_local_kernel — E06: (route) -> gather -> W13 -> act/quant -> W2 -> weighted reduce,
 * single persistent kernel over expert tiles, skew-aware task ordering. Empty experts skipped.
 */
template <typename T, typename W, int BM, int BN, int BK, int STAGES, bool kFusedRouter, typename ArchTag>
__global__ void mega_moe_local_kernel(const KOPS_GRID_CONST MegaMoELocalParams params);

// ---- E07 distributed MegaMoE -----------------------------------------------------------
struct MegaMoEEPParams {
  MegaMoELocalParams local;        // expert compute part (local experts of this rank)
  comm::EPBuffers ep;              // symmetric buffers, signals
  ActQuantMode payload_quant;      // FP8 / MXFP8 / NVFP4 token payloads
  int32_t max_tokens_per_rank;
  uint32_t epoch;
  bool output_needs_zero;          // in-kernel FC2 reduction requires zeroed output: counted as a launch
};

/**
 * mega_moe_ep_kernel — E07: token dispatch <-> expert compute <-> combine overlapped inside one
 * kernel (FlashInfer MoE-EP mega mode / DeepSeek-V4 fused MoE). Liveness requires every rank to
 * launch with the same epoch; slowest-rank latency is the reported metric.
 */
template <typename T, typename W, int BM, int BN, int BK, int NUM_RANKS_PER_NODE, typename ArchTag>
__global__ void mega_moe_ep_kernel(const KOPS_GRID_CONST MegaMoEEPParams params);

// ---- E08 decoder layer (research) -------------------------------------------------------
/**
 * mega_decoder_layer_kernel — E08 (R, experimental): a full decoder layer (attention block + MLP or
 * MoE + residual / HC / AttnRes mixing) as one DAG; multi-layer chaining bounded by plan capacity.
 * Weights stream from HBM; no request scheduling or KV allocation inside the kernel.
 */
template <typename T, typename ArchTag>
__global__ void mega_decoder_layer_kernel(const KOPS_GRID_CONST MegaPlanDevice plan,
                                          const KOPS_GRID_CONST MegaAttentionBlockParams attn,
                                          const KOPS_GRID_CONST MegaMLPParams mlp,
                                          const KOPS_GRID_CONST MegaMoELocalParams moe);

}  // namespace kops::mega
