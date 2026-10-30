#pragma once
// =============================================================================
// kernel-ops | csrc/load/ops.h                                   Catalog: X05
// Checkpoint import: route a checkpoint tensor to the physical layout a kernel
// recipe expects, and run the repack/quantize kernels that get it there.
//
// This family exists because "which packing does this GEMM want" is a contract,
// not glue code: Marlin needs its tile interleave, Machete a different one, TRT-LLM
// gen a shuffled layout, GGUF its block structs, and a Conv1D checkpoint needs a
// transpose before any of them. Doing that in the engine means every consumer
// re-derives it (vLLM, SGLang and llm-awq each have their own copy).
//
// Execution model: load_plan_* are host-only (no device work). load_plan_run()
// enqueues the repack/quantize kernels on ctx.stream with caller workspace, so a
// weight load can be captured like any other phase. Allocation is the caller's job.
//
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh,
//           common/quant.cuh, common/tensor.h, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::load {

/// Where the tensors come from (provenance only; never changes layout math).
enum class SourceFormat : uint8_t {
  kRaw = 0,          // caller already parsed the container
  kSafetensors = 1,
  kGGUF = 2,
  kPyTorch = 3,
};

/// Physical transform applied while copying. Each value maps to exactly one kernel
/// so the plan can report an exact launch count.
enum class RepackKind : uint8_t {
  kCopy = 0,               // contiguous copy, dtype cast only
  kTransposeKMajor = 1,    // [K, N] -> [N, K] (GPT-2 Conv1D, some GGUF)
  kAwqToMarlin = 2,        // quantization::awq_to_marlin
  kGptqToMarlin = 3,       // quantization::gptq_to_marlin (writes the act-order perm)
  kFp4ToMarlin = 4,        // quantization::fp4_to_marlin
  kFp4BackendShuffle = 5,  // quantization::fp4_backend_shuffle (NVFP4/MXFP4 target packing)
  kMachetePrepack = 6,     // gemm::machete_prepack
  kTrtllmShuffle = 7,      // gemm_prep: TRT-LLM gen weight layout
  kGluInterleave = 8,      // gemm::glu_weight_interleave (GPT-OSS gate/up order)
  kBlockScaleSwizzle = 9,  // gemm::block_scale_swizzle (128x4)
  kFp8ScaleTmaLayout = 10, // gemm::fp8_scale_to_tma_layout (+ optional UE8M0)
  kUe8m0Round = 11,        // quantization::fp8_block_scale_to_ue8m0
  kInt8ColSum = 12,        // gemm::int8_weight_colsum (asymmetric azp precompute)
};

/// Checkpoint metadata as read from `quantization_config` / GGUF header. This is
/// the only place a checkpoint's claims are interpreted; kernels never guess.
struct CheckpointMeta {
  SourceFormat format = SourceFormat::kRaw;
  DType src_dtype = DType::kBFloat16;
  WeightPacking packing = WeightPacking::kNone;
  int32_t bits = 16;             // 4 / 8 / 16
  int32_t group_size = 0;        // 32 / 64 / 128 / 0 (per-channel)
  bool symmetric = true;
  bool desc_act = false;         // GPTQ act-order
  bool has_g_idx = false;
  bool zero_point_is_float = false;   // HQQ / AWQ-float
  bool pre_quant_scale = false;       // AWQ activation scale tensor present
  DType scale_dtype = DType::kFloat32;
  QuantProvenance provenance = QuantProvenance::kUnknown;
};

/// One checkpoint tensor -> one destination buffer.
struct TensorRoute {
  const char* name = nullptr;    // e.g. "model.layers.3.mlp.down_proj.qweight"
  const void* src = nullptr;
  int64_t src_offset = 0;        // bytes into src (packed containers)
  DType src_dtype = DType::kInvalid;
  int32_t src_ndim = 0;
  int64_t src_shape[4] = {};
  int64_t src_stride[4] = {};    // elements (kops::StrideUnit::kElement)
  void* dst = nullptr;
  DType dst_dtype = DType::kInvalid;
  int32_t dst_ndim = 0;
  int64_t dst_shape[4] = {};
  int64_t dst_stride[4] = {};
  WeightPacking dst_packing = WeightPacking::kNone;
  RepackKind repack = RepackKind::kCopy;
  // Tensor-parallel / expert-parallel sharding: which slice of the source this
  // route owns. shard_count == 1 means "no sharding".
  int32_t shard_index = 0;
  int32_t shard_count = 1;
  int32_t shard_dim = 0;
};

/// Caller-owned plan storage (same philosophy as every other plan here: no
/// allocation inside the library).
struct LoadPlan {
  TensorRoute* routes = nullptr;   // [capacity]
  int32_t num_routes = 0;
  int32_t capacity = 0;
  int32_t num_launches = 0;        // filled by finalize(): exact device launches
  uint32_t plan_generation = 0;
};

Status load_plan_begin(LoadPlan* plan, TensorRoute* storage, int32_t capacity);
Status load_plan_add(LoadPlan* plan, const TensorRoute& route);
/// Resolve every route to a kernel + validate shapes/packing/alignment. Host-only;
/// rejects when stream_is_capturing(). Fills plan->num_launches.
Status load_plan_finalize(LoadPlan* plan, const CheckpointMeta& meta, const LaunchContext& ctx);
Status load_plan_workspace_requirements(const LoadPlan& plan, WorkspaceRequirements* out);
/// Enqueue the resolved repack/quantize kernels. Capture-safe: no allocation, no sync.
Status load_plan_run(const LoadPlan& plan, const Workspace& ws, const LaunchContext& ctx);
/// Human-readable route table (why each repack was chosen / rejected).
Status load_plan_explain(const LoadPlan& plan, char* buffer, size_t buffer_bytes);

/// Check that a checkpoint's metadata is representable by `target` on this backend
/// (e.g. W4A8-AWQ needs SM90a; MXFP4 needs a block-scaled target). Fills `reason`
/// with the first mismatch. Never silently substitutes a recipe: the caller decides
/// whether to accept a lowering (see docs/ARCH_MATRIX.md §5).
Status validate_checkpoint(const CheckpointMeta& meta, gemm::GemmRecipe target,
                           const LaunchContext& ctx, char* reason, size_t reason_bytes);

/// Offline weight quantization (the direction llm-awq / compressed-tensors provide).
/// `provenance` selects the algorithm; extra pointers are required per algorithm:
///   kRTN                     -> none
///   kAWQ                     -> awq_scale ([cols]) required
///   kGPTQ                    -> hessian ([cols, cols]) required, desc_act honoured
///   kHQQ                     -> block_size (>= 8) required, zero_point_is_float
///   kBitsAndBytesNF4         -> block_size (64) required
/// Not capture-safe (offline); runs on ctx.stream and may use workspace internally.
struct QuantizeRequest {
  const void* weight = nullptr;
  int64_t rows = 0, cols = 0;
  DType in_dtype = DType::kBFloat16;
  int64_t in_row_stride = 0;
  QuantScheme out_scheme;             // storage/scale/group/packing of the result
  void* q_out = nullptr;              // payload
  int64_t q_row_stride = 0;
  void* scales = nullptr;
  int64_t scale_row_stride = 0;
  void* zero_points = nullptr;        // int or float per out_scheme
  const float* awq_scale = nullptr;   // kAWQ
  const void* hessian = nullptr;      // kGPTQ (fp32, [cols, cols])
  int32_t block_size = 0;             // kHQQ / NF4 / NVFP4
  uint64_t seed = 0;                  // any randomized initialization (HQQ)
};

Status quantize_weights(const QuantizeRequest& req, QuantProvenance provenance,
                        const LaunchContext& ctx);

}  // namespace kops::load
