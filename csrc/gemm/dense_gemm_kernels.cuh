#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/dense_gemm_kernels.cuh               Catalog: G01, G02, G07
// 16/32-bit dense GEMM family, one mainloop per architecture, plus the shapes
// that dominate LLM serving and do not fit a generic tile GEMM well:
//   * gemm_mma_sync_kernel     SM80/86/89/120  cp.async + mma.sync (bf16/fp16/tf32)
//   * gemm_wgmma_kernel        SM90a           TMA + WGMMA, warp-specialized (coop / ping-pong), persistent
//   * gemm_tcgen05_kernel      SM100a/103a     TMA + tcgen05.mma (kind::f16 / kind::tf32), TMEM, 1-SM / 2-SM
//   * gemm_sm120_kernel        SM120a/121a     TMA + warp-specialized mma.sync (no TMEM)
//   * gemm_simt_kernel         any             exact fp32 / odd dtypes / reference
//   * gemv_kernel              any             decode M <= 16 (weight-bandwidth bound)
//   * gemv_argmax_kernel       any             greedy LM head: logits never written to HBM
//   * dual_gemm_kernel         SM80+/SM90a     gate & up from two weights, one read of A, GLU epilogue
//   * splitk_reduce_kernel                     parallel split-K fix-up
//   * router_gemm_kernel / router_gemm_bf16x2_kernel   (G02)
//   * bmm_kernel                               strided-batched (MLA absorption, grouped out-proj)
// FP8 / INT8 / FP4 / mixed-input live in their own headers.
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

// -----------------------------------------------------------------------------
// Tile GEMMs (one per architecture family)
// -----------------------------------------------------------------------------

/**
 * gemm_mma_sync_kernel — multistage cp.async + ldmatrix + mma.sync, the shared tensor-core body of B7.
 * One k-step reads 32 bytes of every operand row: m16n8k16 bf16/fp16, m16n8k32 e4m3 (Sm89+) and s8
 * (s32 accumulators), m16n8k8 tf32. Traits::mainloop = kMultistageMmaSync. Split-K serial/parallel,
 * swap-AB (Traits instantiated with M/N roles exchanged), FP8 1x128 x 128x128 promotion per 128 K,
 * every EpilogueOp. MmaGemmParams::rows (ABI 6) maps logical rows to groups / experts / adapters, so
 * grouped GEMM, the fused-MoE expert GEMM and LoRA run this body (persistent task loop).
 * Arch   : Sm80 (A100/A30), Sm86 (A10/A40/RTX 30), Sm89 (L4/L40S/RTX 40), Sm120a (no WGMMA/TMEM).
 * Launch : grid = (tiles or persistent CTAs, 1, batch x split_k); block = 128; dynamic smem =
 *          STAGES * (TILE_M + TILE_N) * 64 bytes.
 */
template <typename Traits>
__global__ void gemm_mma_sync_kernel(const __grid_constant__ MmaGemmParams params);

/**
 * gemm_wgmma_kernel — Hopper: TMA loads into a STAGES-deep smem ring guarded by mbarriers;
 * producer warp issues TMA (optionally multicast across the cluster), consumer warpgroup(s) issue
 * wgmma.mma_async m64nNk16; register reallocation (setmaxnreg) between roles.
 *   kTmaWarpSpecializedCooperative : 2 consumer WGs on one 128xN tile (large M prefill)
 *   kTmaWarpSpecializedPingpong    : 2 consumer WGs on alternating tiles, epilogue overlapped (mid M)
 * Persistent with a linear-tile scheduler (+ stream-K when Traits enable it).
 */
template <typename Traits>
__global__ void gemm_wgmma_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * gemm_tcgen05_kernel — Blackwell datacenter: TMA -> smem, tcgen05.mma (kind::f16 / kind::tf32)
 * accumulating in Tensor Memory, tcgen05.ld for the epilogue, tcgen05.commit + mbarriers.
 *   kTcgen05OneSm : cta_group::1, tiles 64/128 x N
 *   kTcgen05TwoSm : cta_group::2 on a CTA pair (256 x N), B tile multicast within the pair
 * Persistent; tile assignment via cluster launch control when available, else tile_counter.
 * Sm100a (B200/GB200) and Sm103a (B300/GB300) are separate instantiations.
 */
template <typename Traits>
__global__ void gemm_tcgen05_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * gemm_sm120_kernel — Blackwell GeForce / RTX PRO / GB10: TMA + warp-specialized pipeline feeding
 * mma.sync (no WGMMA, no TMEM, smaller smem than SM100). Separate tile tables from Sm100a.
 */
template <typename Traits>
__global__ void gemm_sm120_kernel(const __grid_constant__ GemmTmaParams params);

/**
 * gemm_simt_kernel — CUDA-core FMA GEMM: exact fp32 (fp32 routers / LM heads kept in fp32,
 * numerical references), fp64 for tests, unsupported dtype combinations. Never auto-selected
 * when a tensor-core candidate with the same semantics exists.
 */
template <typename a_t, typename b_t, typename d_t, typename acc_t, int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_simt_kernel(const __grid_constant__ ScaledGemmArgs args);

// -----------------------------------------------------------------------------
// Small-M / decode shapes
// -----------------------------------------------------------------------------

/// Parameters of gemv_kernel (M <= 16, decode / small batch).
struct GemvParams {
  const void* a;            // [M, K] activations (bf16 / fp16)
  int64_t lda;
  const void* b;            // [N, K] weights (bf16 / fp16; quantized weights use fp8/fp4/mixed GEMV kernels)
  int64_t ldb;
  void* d;                  // [M, N]
  int64_t ldd;
  const float* a_scale;     // unused for 16-bit; kept for layout compatibility with quantized GEMVs
  const float* b_scale;
  ScaleGranularity a_scale_granularity;
  ScaleGranularity b_scale_granularity;
  EpilogueParams epi;
  int32_t m, n, k;
  int32_t split_k;          // K partitions accumulated with atomics (deterministic=false) or workspace
  float* split_workspace;   // [split_k, M, N] fp32 when deterministic split-K
};

/**
 * gemv_kernel — memory-bound small-M GEMM: each warp owns ROWS_PER_WARP output channels n and
 * streams B[n, :] once for all M_TILE rows of A (A tile kept in registers/smem).
 * Use    : decode-phase linears where M = batch * q_len <= 16 and weights dominate bytes.
 * Launch : grid = (ceil(N / (WARPS * ROWS_PER_WARP)), split_k), block = 32 * WARPS.
 * Numerics: fp32 accumulation; deterministic when split_k == 1 or split_workspace is used.
 */
template <typename a_t, typename b_t, typename d_t, int M_TILE, int ROWS_PER_WARP, int WARPS, int VEC_SIZE>
__global__ void gemv_kernel(const __grid_constant__ GemvParams params);

/// Arguments of gemv_argmax_kernel.
struct GemvArgmaxParams {
  const void* hidden;           // [M, H] final-normed hidden states (bf16)
  int64_t ld_hidden;
  const void* lm_head;          // [V_shard, H] (bf16 / fp8 with b_scale)
  int64_t ld_lm_head;
  const float* b_scale;         // per-channel fp8 scales or null
  float logit_scale;            // Cohere / Granite logits scaling
  float final_softcap;          // Gemma-2 / Grok (0 = off)
  int64_t vocab_start, vocab_size;   // shard offset and real (unpadded) global vocab
  float* partial_max;           // [M, num_ctas] per-CTA max logit
  int32_t* partial_idx;         // [M, num_ctas] argmax within the CTA's vocab slice
  float* partial_sumexp;        // [M, num_ctas] sum exp(logit - partial_max) for the chosen logprob
  int32_t* out_ids;             // [M] final ids (written by the last CTA via an epoch counter)
  float* out_logprob;           // [M] logprob of the chosen id or null
  uint32_t* done_counter;       // epoch-tagged
  uint32_t epoch;
  int32_t m, hidden_size, v_shard;
};

/**
 * gemv_argmax_kernel — greedy decode without materializing [M, V] logits: LM-head GEMV fused with
 * scale/softcap/padded-vocab masking and a (max, argmax, sum-exp) reduction. Valid ONLY when no
 * logit processor (penalties, bitmask, bias) or logprobs-of-processed-logits are requested; the
 * sampler path is used otherwise. Ties -> lowest id. TP: cross-rank argmax via communication/.
 */
template <typename h_t, typename w_t, int M_TILE, int WARPS>
__global__ void gemv_argmax_kernel(const __grid_constant__ GemvArgmaxParams params);

/// Arguments of dual_gemm_kernel.
struct DualGemmParams {
  const void* a;      int64_t lda;   // [M, K]
  const void* b0;     int64_t ldb0;  // [N, K] gate weight
  const void* b1;     int64_t ldb1;  // [N, K] up weight
  const float* b0_scale;             // per-channel / block scales when FP8 (null for bf16)
  const float* b1_scale;
  const float* a_scale;
  void* d;            int64_t ldd;   // [M, N] = GLU(A.B0^T, A.B1^T)
  ActFn act;
  GluParams glu;
  EpilogueParams epi;                // bias0/bias1 unsupported: fold into weights or use GLU on a fused GEMM
  int32_t m, n, k;
};

/**
 * dual_gemm_kernel — gate and up projections from two separate weight tensors sharing one pass
 * over A, GLU applied in registers (CUTLASS DualGemm idea). Use when checkpoints keep gate_proj /
 * up_proj separate and repacking into a fused [2I, H] weight is not desired.
 */
template <typename Traits>
__global__ void dual_gemm_kernel(const __grid_constant__ DualGemmParams params);

/**
 * splitk_reduce_kernel — D = epilogue(sum_s partial[s]) for parallel split-K workspaces.
 * Deterministic: fixed summation order over s.
 */
template <typename d_t, int VEC_SIZE>
__global__ void splitk_reduce_kernel(
    d_t* __restrict__ d,                    // [M, N]
    const int64_t ldd,
    const float* __restrict__ partials,     // [num_splits, M, N]
    const int num_splits, const int m, const int n,
    const EpilogueParams epi);

// -----------------------------------------------------------------------------
// G02 router GEMMs (logits always fp32)
// -----------------------------------------------------------------------------

struct RouterGemmParams {
  const void* x;              // [M, H] hidden (bf16)
  int64_t ldx;
  const void* w;              // [E, H] router weight (bf16 or fp32)
  const void* w_hi;           // [E, H] bf16 high part (two-term mode)
  const void* w_lo;           // [E, H] bf16 low part scaled by 256 (two-term mode)
  const float* bias;          // [E] router bias (GPT-OSS) or null
  float* logits;              // [M, E] fp32
  int64_t ld_logits;
  int32_t m, num_experts, hidden;
  float lo_scale;             // 1/256 for the HPC-Ops decomposition W ~= W_hi + W_lo / 256
};

/**
 * router_gemm_kernel — logits[M, E] = x . W^T in fp32 accumulation, specialized for the narrow
 * expert dimension (E = 8..1024) and small/medium M. Optionally fused with router bias.
 * Models : Mixtral (8), Qwen3-MoE (128), DeepSeek-V3/V4 & GLM-4.5 (256, H=7168/5120),
 *          Kimi K2 (384), Kimi K3 (896), Llama-4 (16/128), GPT-OSS (32/128 + bias), MiniMax (32/256).
 * Launch : grid = (ceil(E / EXPERTS_PER_CTA), ceil(M / M_TILE)), block = 128..256.
 */
template <typename x_t, typename w_t, int M_TILE, int EXPERTS_PER_CTA>
__global__ void router_gemm_kernel(const __grid_constant__ RouterGemmParams params);

/**
 * router_gemm_bf16x2_kernel — G02 two-term decomposition (HPC-Ops gemm_bf16xfp32):
 *   logits = x . W_hi^T + lo_scale * (x . W_lo^T), bf16 MMA, fp32 accumulation, one pass over x.
 * Approximates an fp32 router weight with ~16 mantissa bits; it is NOT exact FP32 — the
 * registry records max-abs error vs an fp32 reference per (H, E) shape.
 */
template <int M_TILE, int EXPERTS_PER_CTA, typename ArchTag>
__global__ void router_gemm_bf16x2_kernel(const __grid_constant__ RouterGemmParams params);

// -----------------------------------------------------------------------------
// Strided-batched GEMM
// -----------------------------------------------------------------------------

/// Strided-batched GEMM: D[b] = A[b] . B[b]^T for b in [0, batch).
struct BmmParams {
  const void* a;  int64_t lda;  int64_t stride_a_batch;   // A[b]: [M, K]
  const void* b;  int64_t ldb;  int64_t stride_b_batch;   // B[b]: [N, K]  (or [K, N] if b_k_major=false)
  void* d;        int64_t ldd;  int64_t stride_d_batch;   // D[b]: [M, N]
  const float* a_scale;                                   // fp8 per-tensor/per-batch, null for bf16
  const float* b_scale;
  bool b_k_major;
  int32_t batch, m, n, k;
};

/**
 * bmm_kernel — MLA weight absorption and similar per-head projections:
 *   decode: q_nope[T, H, 128] x W_UK[H, 128, 512] -> q_absorbed[T, H, 512]   (batch = H, M = T)
 *           o_latent[T, H, 512] x W_UV[H, 512, 128] -> o[T, H, 128]
 * Also DeepSeek-V4 grouped output projection. A/D are head-strided views of [T, H, *].
 * FP8 batched variant: fp8_gemm_kernels.cuh::fp8_bmm_kernel.
 * Models : DeepSeek V2/V3/V3.2/V4, Kimi K2/K2.5/K3 (gated MLA), Kimi Linear MLA layers, MiniCPM3.
 */
template <typename a_t, typename b_t, typename d_t, int TILE_M, int TILE_N, int TILE_K, typename ArchTag>
__global__ void bmm_kernel(const __grid_constant__ BmmParams params);

}  // namespace kops::gemm
