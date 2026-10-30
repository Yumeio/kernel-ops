#pragma once
// =============================================================================
// kernel-ops | csrc/moe/routing_kernels.cuh                          Catalog: M01
// Router scoring + expert selection for every production MoE family.
// Requires: common/macros.cuh, common/dtype.cuh.
//
// Semantics are explicit (no "default" router):
//   score function  : softmax | sigmoid | sqrt(softplus) | softmax-after-topk | none
//   selection bias  : correction_bias added for SELECTION only, weights use raw scores
//   grouping        : n_group / topk_group with group score = max or sum of top-2
//   renormalize     : divide selected weights by their sum (before routed_scaling_factor)
//   tie-break       : lower expert index wins (deterministic across GPUs)
// =============================================================================

namespace kops::moe {

enum class RouterScore : uint8_t {
  kSoftmax = 0,            // softmax over all E, then top-k        (Mixtral*, Qwen1.5/2/3-MoE, OLMoE, DBRX, Jamba,
                           //                                        Grok, ERNIE-4.5, Hunyuan, MiniMax-01, DeepSeek-V2)
  kSoftmaxAfterTopK = 1,   // top-k on logits, softmax over the k   (GPT-OSS; == Mixtral with renormalize)
  kSigmoid = 2,            // sigmoid, top-k (with correction bias)  (DeepSeek-V3/V3.1/V3.2, Kimi K2/K2.5/K3,
                           //                                        GLM-4.5/5, MiniMax-M2, Llama-4 top-1, Nemotron-3)
  kSqrtSoftplus = 3,       // sqrt(softplus(x))                      (DeepSeek-V4)
  kNone = 4,               // raw logits as weights (custom routers)
};

enum class GroupScore : uint8_t {
  kMax = 0,                // group score = max expert score in group (DeepSeek-V2 group_limited_greedy)
  kTop2Sum = 1,            // group score = sum of top-2 (biased) scores in group (DeepSeek-V3 noaux_tc)
};

/// Arguments of the routing kernels. All outputs are written for every token row.
struct RouterParams {
  const void* logits;              // [num_tokens, num_experts] fp32 (or bf16) router logits
  int64_t logits_stride;
  const float* correction_bias;    // [num_experts] e_score_correction_bias / quantile-balancing bias, or null
  int32_t num_tokens;
  int32_t num_experts;             // global routed experts (excludes shared experts)
  int32_t topk;                    // 1 (Llama-4, Switch), 2 (Mixtral, Grok), 4 (GPT-OSS), 8 (DSv3, Qwen3, K2), 16 (K3)
  int32_t num_expert_groups;       // 1 = flat (DeepSeek-V3: 8, Kimi K2: 1)
  int32_t topk_groups;             // groups kept (DeepSeek-V3: 4)
  GroupScore group_score;
  bool renormalize;                // Mixtral/Qwen3 norm_topk_prob/DSv3/K2/K3: true; Qwen2-MoE: false
  float routed_scaling_factor;     // DeepSeek-V3 2.5, Kimi K2 2.827, 1.0 otherwise (applied after renorm)
  float logit_softcap;             // router logit soft-cap before scoring (0 = off)
  // Expert placement (EP / EPLB)
  const int32_t* expert_map;       // [num_experts] global -> local id or -1 (not on this rank); null = identity
  const int32_t* logical_to_physical;  // [num_experts, max_replicas] EPLB redundant replicas or null
  const int32_t* replica_count;    // [num_experts]
  int32_t max_replicas;
  // Shared-expert fusion: append `num_fused_shared` pseudo-experts with ids
  // [num_experts, num_experts + num_fused_shared) and weight shared_weight to every token.
  int32_t num_fused_shared;
  float shared_weight;             // usually 1 / routed_scaling_factor when weights are scaled later
  // Outputs
  float* topk_weights;             // [num_tokens, topk + num_fused_shared]
  int32_t* topk_ids;               // [num_tokens, topk + num_fused_shared] (physical ids after EPLB map)
  int32_t* expert_load;            // [num_experts] optional atomically accumulated histogram (EPLB/balancing stats)
};

/**
 * moe_topk_kernel — flat routing (num_expert_groups == 1).
 * Launch : one warp per token for E <= 256 (each lane holds E/32 scores in registers),
 *          one CTA per token for E in (256, 1024] (Kimi-K3 896 experts), grid-stride over tokens.
 * NUM_EXPERTS = 0 selects the generic runtime-E path (slower; for unusual E).
 */
template <typename logits_t, RouterScore kScore, int NUM_EXPERTS, int MAX_TOPK>
__global__ void moe_topk_kernel(const KOPS_GRID_CONST RouterParams params);

/**
 * moe_grouped_topk_kernel — group-limited routing:
 *   s = score(logits); b = s + correction_bias; group_score per GroupScore over b;
 *   keep topk_groups groups; top-k over b inside kept groups; weights = s[selected] (unbiased)
 *   -> renormalize -> * routed_scaling_factor.
 * Models : DeepSeek-V2/V2.5 (softmax, kMax), DeepSeek-V3/V3.1/V3.2/V4 (sigmoid / sqrt-softplus, kTop2Sum),
 *          GLM-4.5 (n_group = 1 reduces to moe_topk_kernel with bias), MiniMax-M2.
 */
template <typename logits_t, RouterScore kScore, int NUM_EXPERTS, int NUM_GROUPS, int MAX_TOPK>
__global__ void moe_grouped_topk_kernel(const KOPS_GRID_CONST RouterParams params);

/**
 * moe_hash_route_kernel — hash routing (Roller et al. 2021): experts chosen from the token id,
 * weights uniform (1/topk, then routed_scaling_factor). DeepSeek-V4 uses it in the first MoE
 * layers instead of dense FFNs. hash_table [vocab, topk] precomputed by the checkpoint
 * (preferred, exact); when null a fixed multiplicative hash of (token_id, j) is used.
 */
__global__ void moe_hash_route_kernel(
    float* __restrict__ topk_weights,         // [num_tokens, topk]
    int32_t* __restrict__ topk_ids,           // [num_tokens, topk]
    const int64_t* __restrict__ token_ids,    // [num_tokens]
    const int32_t* __restrict__ hash_table,   // [vocab_size, topk] or null
    const int32_t* __restrict__ expert_map,   // [num_experts] or null
    const float routed_scaling_factor,
    const int num_tokens, const int num_experts, const int topk, const int64_t vocab_size);

/**
 * moe_fused_router_topk_kernel — decode-time fusion: router GEMM (x . W^T, fp32 accumulate,
 * optional bias) + scoring + (grouped) top-k in one kernel for num_tokens <= 64. Avoids writing
 * logits to HBM. Same RouterParams semantics; logits pointer is unused.
 */
template <typename x_t, typename w_t, RouterScore kScore, int NUM_EXPERTS, int NUM_GROUPS, int MAX_TOPK>
__global__ void moe_fused_router_topk_kernel(
    const KOPS_GRID_CONST RouterParams params,
    const x_t* __restrict__ hidden,           // [num_tokens, hidden]
    const int64_t hidden_stride,
    const w_t* __restrict__ router_weight,    // [num_experts, hidden]
    const float* __restrict__ router_bias,    // [num_experts] (GPT-OSS) or null
    const int hidden_size);

}  // namespace kops::moe
