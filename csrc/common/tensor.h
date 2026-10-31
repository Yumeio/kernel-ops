#pragma once
// =============================================================================
// kernel-ops | csrc/common/tensor.h
// Framework-independent host ABI: tensor views, launch context, workspace.
// Every host launcher in csrc/<family>/ops.h takes only these types, so the
// library can be wrapped by PyTorch, JAX/XLA FFI, TVM-FFI, Rust or C, without
// naming a backend type.
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh.
// =============================================================================

namespace kops {

inline constexpr int kMaxTensorDims = 8;

/// Non-owning DLPack-like view. A Tensor with data == nullptr means "absent"
/// for optional arguments. `stride_unit` (kops::StrideUnit, common/macros.cuh)
/// says whether the stride array counts elements or bytes; never mix units
/// inside one view. Launchers validate it against the dtype.
struct Tensor {
  void* data = nullptr;
  DType dtype = DType::kInvalid;
  int32_t ndim = 0;
  int32_t device = -1;                  // backend ordinal; -1 = host memory
  int64_t shape[kMaxTensorDims] = {};
  int64_t stride[kMaxTensorDims] = {};
  int64_t byte_offset = 0;              // added to data (DLPack compatibility)
  StrideUnit stride_unit = StrideUnit::kElement;
};

/// Execution strategy (catalog E01-E08). Orthogonal to op_id and precision.
enum class ExecutionStrategy : uint8_t {
  kModular = 0,     // one or more ordinary launches
  kGraph = 1,       // launches recorded into a caller-owned CUDA graph
  kPDL = 2,         // programmatic dependent launch between consecutive kernels (SM90+)
  kPersistent = 3,  // persistent CTAs pulling tiles of ONE operator
  kMegakernel = 4,  // several operators of a DAG inside one kernel
};

/// Per-call launch context. The library never picks a device, stream or
/// allocator on its own: everything here is caller-owned or caller-probed.
/// v2: backend-neutral stream/handles, arch_id + wave_size instead of sm_version,
/// compute_units / max_ctas_per_cu / lds_bytes instead of num_sms / max_ctas_per_sm.
struct LaunchContext {
  kops_stream_t stream = nullptr;
  BackendKind backend = BackendKind::kCuda;
  int32_t device = 0;               // backend ordinal (MIG / green-context aware)
  int32_t arch_id = 0;              // KOPS_TARGET_*: 800, 900, 1000, 9420, 11000 ...
  int32_t wave_size = 32;           // 32 on CUDA; 0 = not probed
  int32_t compute_units = 0;        // SMs in this partition
  int32_t max_ctas_per_cu = 0;      // 0 = query occupancy
  int32_t lds_bytes = 0;            // shared memory per block, 0 = not probed
  bool enable_pdl = false;          // programmatic stream serialization (SM90+)
  bool deterministic = false;       // forbid reductions whose order depends on scheduling/atomics
  bool batch_invariant = false;     // per-row results independent of batch composition
  bool allow_tf32 = false;          // fp32 GEMM inputs may be rounded to tf32
  uint32_t capture_generation = 0;  // incremented by the engine on every graph (re)capture
  ExecutionStrategy strategy = ExecutionStrategy::kModular;
};

/// True when the context was probed on AMD hardware (reserved; always false
/// until the HIP path is built).
KOPS_HOST_DEVICE_INLINE bool ctx_is_amd(const LaunchContext& ctx) {
  return ctx.backend == BackendKind::kHip;
}
/// PDL is a CUDA SM90+ feature; false everywhere else.
KOPS_HOST_DEVICE_INLINE bool ctx_supports_pdl(const LaunchContext& ctx) {
  return !ctx_is_amd(ctx) && ctx.arch_id >= 900 && ctx.arch_id < 9000;
}
/// Exact wave/sub-group width for reductions and "one wave per row" layouts.
KOPS_HOST_DEVICE_INLINE int32_t ctx_wave_size(const LaunchContext& ctx) {
  return ctx.wave_size > 0 ? ctx.wave_size : (ctx_is_amd(ctx) ? 64 : 32);
}

/// What a plan/op needs; queried before reservation, outside graph capture.
struct WorkspaceRequirements {
  size_t device_scratch_bytes = 0;     // valid only during one run; may be reused after completion event
  size_t device_persistent_bytes = 0;  // must live as long as the plan (task maps, counters, epochs)
  size_t host_pinned_bytes = 0;        // pinned staging for plan metadata (never touched in replay)
  size_t alignment = 256;              // required alignment of every region
};

/// Caller-owned memory leases. The library never allocates in run().
struct Workspace {
  void* device_scratch = nullptr;
  size_t device_scratch_bytes = 0;
  void* device_persistent = nullptr;
  size_t device_persistent_bytes = 0;
  void* host_pinned = nullptr;
  size_t host_pinned_bytes = 0;
};

// ---- Helpers ------------------------------------------------------------------
bool tensor_present(const Tensor& t) noexcept;
bool tensor_is_contiguous(const Tensor& t) noexcept;
int64_t tensor_numel(const Tensor& t) noexcept;
/// Validate device, dtype, rank, stride unit and alignment in one call (used by every launcher).
Status check_tensor(const Tensor& t, DType expected, int32_t expected_ndim, size_t alignment_bytes,
                    const char* arg_name) noexcept;
/// Fill backend / arch_id / wave_size / compute_units / max_ctas_per_cu / lds_bytes
/// for `device` (host query; never call during capture). Queries the CUDA runtime.
Status make_launch_context(int32_t device, kops_stream_t stream, LaunchContext* out) noexcept;

}  // namespace kops
