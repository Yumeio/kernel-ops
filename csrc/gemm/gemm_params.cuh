#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/gemm_params.cuh                        Catalog: G01-G07
// Shared GEMM vocabulary: precision recipes, operand descriptors (data + scales),
// epilogue, tactics (tile / cluster / schedule / split-K / swap-AB), TMA argument
// block and arch traits. Used by dense, FP8, INT8, FP4, mixed-input, grouped,
// comm-overlapped GEMMs, MoE expert GEMMs and Megakernel tiles.
//
// Convention: D[M, N] = epilogue( A[M, K] . B[N, K]^T )
//   A = activations (row-major, K-major), B = nn.Linear weight [out, in] (row-major, K-major).
//   Every scale is a DEQUANT multiplier: A ~= qA * sA, B ~= qB * sB (common/quant.cuh).
// Requires: common/macros.cuh, common/dtype.cuh, common/quant.cuh, common/op_types.cuh.
//           Toolkit: <cuda.h> (CUtensorMap) in addition to the common prerequisites.
// =============================================================================

namespace kops::gemm {

// -----------------------------------------------------------------------------
// Precision recipes. The name states exactly what is multiplied and how it is scaled;
// a capability row is (recipe x arch x tactic), never "FP8 supported" in general.
// -----------------------------------------------------------------------------
enum class GemmRecipe : uint16_t {
  // ---- 16/32-bit -------------------------------------------------------------
  kBF16 = 0,                 // bf16 x bf16 -> fp32 acc
  kFP16 = 1,                 // fp16 x fp16 -> fp32 acc (fp16 acc only on explicit request)
  kTF32 = 2,                 // fp32 inputs rounded to tf32 (LaunchContext::allow_tf32)
  kFP32 = 3,                 // exact fp32 SIMT (router / LM head in fp32, reference)
  // ---- FP8 (e4m3 x e4m3 unless the operand dtype says e5m2) ------------------------
  kFP8Tensorwise = 10,       // one scale per tensor for A and B (Llama-3.x FP8 static, Mistral FP8)
  kFP8Rowwise = 11,          // per-token sA[M] x per-channel sB[N] (FP8-dynamic checkpoints, torchao rowwise)
  kFP8Blockwise = 12,        // sA 1x128 (per token, per 128 K) x sB 128x128, fp32 scales
                             //   DeepSeek-V3/V3.1/V3.2/V4 non-expert linears, Kimi K2/K2.5 FP8, Qwen3-FP8,
                             //   GLM-4.5-FP8, MiniMax-M2 FP8
  kFP8BlockwiseUE8M0 = 13,   // same blocks, power-of-two UE8M0 scales (Blackwell recipes, DeepSeek-V3.1+ UE8M0)
  kMXFP8 = 14,               // OCP MXFP8: e4m3, UE8M0 per 32 along K on both operands (Kimi-K3 activations)
  // ---- INT8 ---------------------------------------------------------------------------
  kINT8Tensorwise = 20,
  kINT8Rowwise = 21,         // per-token (sym, or asym with azp) x per-channel: SmoothQuant W8A8
  // ---- FP4 / FP6 (native block-scaled MMA) ----------------------------------------------
  kNVFP4 = 30,               // W4A4: e2m1 x e2m1, E4M3 scale per 16 + fp32 global per tensor
  kMXFP4 = 31,               // W4A4: e2m1 x e2m1, UE8M0 per 32
  kMXFP4xMXFP8 = 32,         // W4A8-MX: B = MXFP4 weights, A = MXFP8 activations (GPT-OSS / Kimi-K3 on Blackwell)
  kMXFP6xMXFP8 = 33,         // e2m3 / e3m2 weights x MXFP8 (spec-complete; rare in checkpoints)
  // ---- Mixed input: weights converted in-kernel to the MMA type of A --------------------
  kW4A16Int = 40,            // int4 groupwise (+ zeros) x bf16/fp16: AWQ / GPTQ / compressed-tensors / Kimi K2-Thinking
  kW8A16Int = 41,            // int8 weight-only
  kW8A16FP8 = 42,            // fp8 weight-only x bf16 (FP8 checkpoints on SM80/SM86)
  kW4A16MXFP4 = 43,          // MXFP4 weight x bf16 (GPT-OSS on Hopper/Ampere, Kimi-K3 fallback)
  kW4A16NVFP4 = 44,          // NVFP4 weight x bf16 (NVFP4 checkpoints on non-FP4 GPUs)
  kW4A8IntFP8 = 45,          // int4 groupwise weight x fp8 activation (Hopper W4A8-AWQ)
  kW4A8IntInt8 = 46,         // int4 (two-level scales) x int8 activation (QServe W4A8)
  kW4A8FP4xFP8 = 47,         // FP4 (MX/NV) weight -> e4m3 in-kernel x fp8 activation (FP4 experts on Hopper,
                             //   DeepSeek-V4 FP4 experts on FP8-only hardware; NVFP4 weights x FP8 acts)
  // ---- Other formats ---------------------------------------------------------------------
  kGGUF = 50,                // ggml block formats (see GGUFType)
  kLUT4 = 51,                // 4-bit codebook (NF4 / FP4 LUT / custom), double-quantized absmax
};

/// Operand storage order inside a GEMM. FP8/INT8/FP4 tensor-core paths on SM90/SM100
/// require K-major A and B; MN-major operands are transposed once at load time
/// (gemm_prep_kernels.cuh::weight_transpose_kernel).
enum class MajorOrder : uint8_t {
  kKMajor = 0,   // contiguous along K: A[M,K] row-major, B[N,K] row-major (nn.Linear)
  kMNMajor = 1,  // contiguous along M or N: e.g. GPT-2 Conv1D weights [K, N]
};

/// Hopper FP8 WGMMA accumulates with reduced internal precision. kPromoted moves partial
/// sums to fp32 registers every `fp8_promotion_k` K elements (128 = 4 WGMMA k32 steps per
/// the DeepSeek-V3 report) and applies block scales there. kFastAccum keeps everything in
/// the MMA accumulator (faster, only for per-tensor/rowwise recipes with a validated tolerance).
/// SM100 tcgen05 accumulates FP8 in full fp32 TMEM; the flag is ignored there.
enum class Fp8Accum : uint8_t { kPromoted = 0, kFastAccum = 1 };

/// One GEMM operand with its quantization metadata.
struct ScaledOperand {
  const void* data = nullptr;           // payload; packed two-per-byte for fp4/int4
  int64_t ld = 0;                       // leading dimension (elements; storage bytes for packed)
  int64_t batch_stride = 0;             // between batches / experts (elements or bytes as above)
  DType dtype = DType::kBFloat16;
  MajorOrder major = MajorOrder::kKMajor;
  WeightPacking packing = WeightPacking::kNone;   // Marlin / Machete / TRTLLM-gen / GGUF layouts
  // Scales
  const void* scale = nullptr;          // see granularity / layout
  DType scale_dtype = DType::kFloat32;  // f32 | bf16 | e4m3 (NVFP4) | e8m0 (MX / UE8M0 recipes)
  ScaleGranularity granularity = ScaleGranularity::kNone;
  ScaleLayout scale_layout = ScaleLayout::kRowMajor;
  int16_t group_mn = 1;                 // scale block along M (A) or N (B): 1 or 128
  int16_t group_k = 0;                  // scale block along K: 16 (NVFP4), 32 (MX), 64/128 (int4/fp8 groups)
  int64_t scale_ld = 0;
  int64_t scale_batch_stride = 0;
  const float* global_scale = nullptr;  // NVFP4 per-tensor scale ([1], or [E] for grouped)
  // Asymmetric / act-order metadata
  const void* zero_points = nullptr;    // int8 azp per row (A) | int4 group zeros (B) | float zeros (HQQ)
  DType zero_dtype = DType::kInvalid;
  const int32_t* g_idx = nullptr;       // GPTQ act-order group index [K]
  const int32_t* perm = nullptr;        // K permutation (GPTQ desc_act) applied consistently to A and B
  const void* second_level = nullptr;   // two-level scales (QServe int8 channel scales, NF4 double-quant)
};

// -----------------------------------------------------------------------------
// Epilogue
// -----------------------------------------------------------------------------

/// Epilogue operations, applied in this fixed order (bit flags):
///   acc(fp32) -> *alpha -> *row_scale[m] -> *col_scale[n] -> int8 azp correction -> +bias[n]
///   -> clamp -> softcap -> act / GLU (paired columns) -> *routing_weight[m]
///   -> sigmoid gate multiply -> +beta*residual[m,n] -> quantize or round to D dtype
///   -> store (direct / scattered / atomically accumulated).
enum EpilogueOp : uint32_t {
  kEpiNone = 0u,
  kEpiRowScale = 1u << 0,       // per-token activation dequant scale (rowwise FP8/INT8)
  kEpiColScale = 1u << 1,       // per-channel weight dequant scale
  kEpiBias = 1u << 2,           // GPT-2/OPT/BLOOM/Falcon/Qwen2 QKV bias, GPT-OSS expert bias
  kEpiSoftcap = 1u << 3,        // cap * tanh(x / cap)
  kEpiAct = 1u << 4,            // elementwise ActFn on every column
  kEpiGLU = 1u << 5,            // N columns form (gate, up) pairs -> N/2 outputs (see glu_interleave)
  kEpiRoutingWeight = 1u << 6,  // MoE: scale row m by routing weight
  kEpiSigmoidGateMul = 1u << 7, // out = gate_src * sigmoid(acc)  (Kimi-K3 gated-MLA gate projection epilogue,
                                //  HPC-Ops gated_mla_gemm); gate_is_acc=false: out = acc * sigmoid(gate_src)
  kEpiResidualAdd = 1u << 8,    // + beta * residual
  kEpiQuantOut = 1u << 9,       // write quantized D (FP8 per-token / per-group, MXFP8, NVFP4, INT8)
  kEpiScatterRows = 1u << 10,   // write row m to scatter_rows[m] (MoE finalize, index_add)
  kEpiAtomicAdd = 1u << 11,     // atomically accumulate into D (non-deterministic; disabled by ctx.deterministic)
  kEpiClamp = 1u << 12,         // clamp(x, -clamp_limit, clamp_limit) (MPT / DBRX clip_qkv)
  kEpiAzpCorrection = 1u << 13, // INT8 asymmetric: acc -= azp[m] * colsum_b[n]
  kEpiOutFp32 = 1u << 14,       // keep D in fp32 (router logits, LM head logits for sampling)
};

struct EpilogueParams {
  uint32_t ops = kEpiNone;                 // OR of EpilogueOp
  float alpha = 1.0f;                      // per-tensor dequant product (sA*sB), NVFP4 1/(gA*gB)
  const float* alpha_ptr = nullptr;        // device scalar / per-group array (grouped: [G]); overrides alpha
  float beta = 1.0f;                       // residual multiplier
  float softcap = 0.0f;
  float clamp_limit = 0.0f;                // kEpiClamp (MPT clip_qkv, DBRX clip_qkv = 8)
  ActFn act = ActFn::kIdentity;
  GateLayout glu_layout = GateLayout::kGateUp;
  int32_t glu_interleave = 0;              // kEpiGLU with GateLayout::kInterleaved: rows of B alternate
                                           // gate/up in blocks of glu_interleave (1 = GPT-OSS checkpoint
                                           // order; 8/16/64 = prep-interleaved so one N tile holds both halves)
  GluParams glu;
  const float* row_scale = nullptr;        // [M]
  const float* col_scale = nullptr;        // [N] (grouped: [G, N] with col_scale_group_stride)
  int64_t col_scale_group_stride = 0;
  const int32_t* azp = nullptr;            // [M] int8 activation zero points
  const int32_t* colsum_b = nullptr;       // [N] sum_k qB[n, k] (precomputed at load)
  const void* bias = nullptr;              // [N] (or [G, N] for grouped, stride bias_group_stride)
  DType bias_dtype = DType::kBFloat16;
  int64_t bias_group_stride = 0;
  const float* routing_weight = nullptr;   // [M]
  const void* gate_src = nullptr;          // [M, N] tensor combined with sigmoid (kEpiSigmoidGateMul)
  int64_t gate_src_stride = 0;
  bool gate_is_acc = true;
  const void* residual = nullptr;          // [M, N]
  int64_t residual_stride = 0;
  const int32_t* scatter_rows = nullptr;   // [M] destination row in D (-1 = drop)
  ActQuantOut quant_out;                   // kEpiQuantOut (block amax computed inside the N tile)
};

// -----------------------------------------------------------------------------
// Problem / arguments
// -----------------------------------------------------------------------------

/// Static description used for capability lookup / tactic selection.
struct GemmProblem {
  int64_t m = 0, n = 0, k = 0;
  int32_t batch = 1;                       // strided batched (MLA absorption bmm)
  GemmRecipe recipe = GemmRecipe::kBF16;
  DType a_dtype = DType::kBFloat16;
  DType b_dtype = DType::kBFloat16;
  DType d_dtype = DType::kBFloat16;
  DType accum_dtype = DType::kFloat32;
  QuantScheme a_quant;                     // activation scheme (granularity kNone for bf16)
  QuantScheme b_quant;                     // weight scheme
  MajorOrder a_major = MajorOrder::kKMajor;
  MajorOrder b_major = MajorOrder::kKMajor;
  uint32_t epilogue_ops = kEpiNone;
};

/// Legacy flat argument block (BF16 / simple paths). New code uses ScaledGemmArgs.
struct GemmArgs {
  const void* a = nullptr;      int64_t lda = 0;   int64_t batch_stride_a = 0;
  const void* b = nullptr;      int64_t ldb = 0;   int64_t batch_stride_b = 0;
  void* d = nullptr;            int64_t ldd = 0;   int64_t batch_stride_d = 0;
  const void* a_scale = nullptr;                   // layout per GemmProblem::a_quant
  const void* b_scale = nullptr;                   // layout per GemmProblem::b_quant
  const void* b_zeros = nullptr;                   // asymmetric weights (AWQ, GPTQ asym)
  const int32_t* b_g_idx = nullptr;                // GPTQ act-order group index [K]
  const int32_t* a_perm = nullptr;                 // GPTQ act-order permutation applied to A columns [K]
  const float* b_global_scale = nullptr;           // NVFP4 weight global scale
  const int32_t* a_azp = nullptr;                  // INT8 asymmetric activation zero points [M]
  const int32_t* b_azp_adj = nullptr;              // precomputed sum_k(B) * azp correction [N]
  EpilogueParams epi;
  int32_t split_k = 1;                             // >1 uses fp32 workspace + splitk_reduce_kernel
  void* workspace = nullptr;                       // device scratch from WorkspaceRequirements
  size_t workspace_bytes = 0;
  int32_t* locks = nullptr;                        // Marlin / stream-K inter-CTA locks (zeroed, epoch-tagged)
};

/// Full argument block for every scaled / quantized / mixed-input GEMM.
struct ScaledGemmArgs {
  ScaledOperand a;                       // activations [M, K] (or [batch, M, K])
  ScaledOperand b;                       // weights [N, K]     (or [batch, N, K])
  void* d = nullptr;                     // [M, N]
  int64_t ldd = 0;
  int64_t batch_stride_d = 0;
  DType d_dtype = DType::kBFloat16;
  EpilogueParams epi;
  int64_t m = 0, n = 0, k = 0;
  int32_t batch = 1;
  GemmRecipe recipe = GemmRecipe::kBF16;
  Fp8Accum fp8_accum = Fp8Accum::kPromoted;
  int32_t fp8_promotion_k = 128;         // K elements between fp32 promotions (kPromoted)
  void* workspace = nullptr;             // split-K partials, stream-K fix-up, locks
  size_t workspace_bytes = 0;
  uint32_t epoch = 0;                    // epoch for semaphores / locks (graph replay safe)
};

// -----------------------------------------------------------------------------
// Tactics (what the autotuner chooses; recorded per capability row)
// -----------------------------------------------------------------------------

enum class MainloopSchedule : uint8_t {
  kSimtFma = 0,                       // CUDA cores (fp32, GEMV, LUT, fallback)
  kMultistageMmaSync = 1,             // cp.async + ldmatrix + mma.sync (SM80/86/89, SM120 non-scaled)
  kTmaWarpSpecialized = 2,            // SM90a: 1 producer warp(group) + 1 consumer warpgroup
  kTmaWarpSpecializedCooperative = 3, // SM90a: 2 consumer warpgroups share one 128..256-row tile
  kTmaWarpSpecializedPingpong = 4,    // SM90a: 2 consumer warpgroups alternate tiles (epilogue overlap)
  kTcgen05OneSm = 5,                  // SM100a/103a: tcgen05.mma cta_group::1, TMEM accumulator
  kTcgen05TwoSm = 6,                  // SM100a/103a: cta_group::2 (CTA pair, 256-row tiles, B multicast)
  kSm120BlockScaledWS = 7,            // SM120a/121a: TMA + warp-specialized block-scaled mma.sync
  kSm120WarpSpecialized = 8,          // SM120a/121a: TMA + warp-specialized mma.sync (bf16 / fp8)
};

enum class EpilogueSchedule : uint8_t {
  kDirectStore = 0,          // registers -> global (small tiles, scatter, atomics)
  kSmemStore = 1,            // registers -> smem -> coalesced global
  kTmaStore = 2,             // registers/TMEM -> smem -> TMA store
  kTmaStoreWarpSpecialized = 3,  // dedicated epilogue warps overlapping the next mainloop (SM90/SM100)
};

enum class SplitKMode : uint8_t {
  kNone = 0,
  kSerial = 1,               // in-kernel semaphore reduction into D (deterministic order by slice id)
  kParallel = 2,             // fp32 partials in workspace + splitk_reduce_kernel
  kStreamK = 3,              // stream-K with fix-up (persistent)
  kStreamKHybrid = 4,        // data-parallel full waves + stream-K tail wave
};

enum class RasterOrder : uint8_t { kAlongM = 0, kAlongN = 1, kHeuristic = 2 };

/// One compiled + tunable configuration. tile/cluster/stages are compile-time in the kernel
/// instantiation; the rest is runtime. kernel_id ties the tactic to the registry row.
struct GemmTactic {
  int16_t tile_m = 128, tile_n = 128, tile_k = 64;
  int8_t cluster_m = 1, cluster_n = 1, cluster_k = 1;
  int8_t stages = 0;                         // 0 = max that fits shared memory
  MainloopSchedule mainloop = MainloopSchedule::kMultistageMmaSync;
  EpilogueSchedule epilogue = EpilogueSchedule::kSmemStore;
  SplitKMode split_k_mode = SplitKMode::kNone;
  int16_t split_k_slices = 1;
  RasterOrder raster = RasterOrder::kHeuristic;
  int8_t swizzle_log = 0;                    // CTA rasterization swizzle (L2 reuse)
  bool swap_ab = false;                      // compute D^T = B . A^T: tiny M becomes the MMA N dim (decode)
  bool persistent = false;                   // persistent CTAs with tile scheduler
  bool use_pdl = false;                      // griddepcontrol in prologue / epilogue
  bool tma_multicast = false;                // cluster multicast of A/B tiles
  uint64_t kernel_id = 0;
};

// -----------------------------------------------------------------------------
// TMA-based kernel arguments (SM90a / SM100a / SM103a / SM120a)
// -----------------------------------------------------------------------------

/// Tensor maps are encoded on the host at plan time (cuTensorMapEncodeTiled), or patched on
/// device per group for grouped GEMM (grouped_gemm_prepare_kernel). Passed as __grid_constant__.
struct GemmTmaParams {
  CUtensorMap tma_a;           // A tiles (FP4 operands for kind::mxf8f6f4 use the padded 4->8-bit TMA mode)
  CUtensorMap tma_b;
  CUtensorMap tma_sfa;         // block scale factors of A (block-scaled recipes)
  CUtensorMap tma_sfb;
  CUtensorMap tma_c;           // source C / residual (TMA load in epilogue)
  CUtensorMap tma_d;           // TMA store of D
  ScaledGemmArgs args;         // raw pointers + scalars (epilogue inputs, fallbacks)
  int32_t num_tiles_m, num_tiles_n, num_tiles_k;
  uint32_t* tile_counter;      // dynamic persistent scheduler / CLC fallback (epoch-tagged)
  float* streamk_partials;     // stream-K fix-up workspace
  uint32_t* streamk_flags;
};

/// Compile-time configuration of a tensor-core GEMM instantiation.
template <typename ElementA, typename ElementB, typename ElementD, typename ElementAccum, int TILE_M,
          int TILE_N, int TILE_K, int CLUSTER_M, int CLUSTER_N, int STAGES, MainloopSchedule kMainloop,
          EpilogueSchedule kEpilogue, typename ArchTag>
struct GemmTraits {
  using a_type = ElementA;
  using b_type = ElementB;
  using d_type = ElementD;
  using accum_type = ElementAccum;
  using arch = ArchTag;
  static constexpr int tile_m = TILE_M;
  static constexpr int tile_n = TILE_N;
  static constexpr int tile_k = TILE_K;
  static constexpr int cluster_m = CLUSTER_M;
  static constexpr int cluster_n = CLUSTER_N;
  static constexpr int stages = STAGES;
  static constexpr MainloopSchedule mainloop = kMainloop;
  static constexpr EpilogueSchedule epilogue = kEpilogue;
};

/// Backend implementing a capability (recorded in the registry, never inferred).
enum class GemmBackend : uint8_t {
  kNative = 0,       // kernels declared in csrc/gemm/*.cuh
  kCublasLt = 1,     // dense BF16/FP16/TF32; FP8 scale modes as supported by the installed cuBLASLt
  kCutlass = 2,      // CUTLASS 3.x/4.x CollectiveBuilder instantiations (sm90a / sm100a / sm120a)
  kDeepGemm = 3,     // FP8/FP4 dense + grouped (contiguous / masked), JIT
  kTrtllmGen = 4,    // TRT-LLM generated cubins (dense + MoE, FP4/FP8)
  kMarlin = 5,
  kMachete = 6,
  kTriton = 7,
  kAuto = 255,       // registry picks among release-supported candidates
};

}  // namespace kops::gemm
