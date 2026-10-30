#pragma once
// =============================================================================
// kernel-ops | csrc/common/tensor.h
// Framework-independent host ABI: tensor views, launch context, workspace.
// Every host launcher in csrc/<family>/ops.h takes only these types, so the
// library can be wrapped by PyTorch, JAX/XLA FFI, TVM-FFI, Rust or C.
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh.
// =============================================================================

namespace kops {

inline constexpr int kMaxTensorDims = 8;

/// Non-owning DLPack-like view. A Tensor with data == nullptr means "absent"
/// for optional arguments. Strides are in elements; for packed sub-byte dtypes
/// (fp4/int4) strides count storage bytes along the packed (last) dimension.
struct Tensor {
  void* data = nullptr;
  DType dtype = DType::kInvalid;
  int32_t ndim = 0;
  int32_t device = -1;                  // CUDA ordinal; -1 = host memory
  int64_t shape[kMaxTensorDims] = {};
  int64_t stride[kMaxTensorDims] = {};
  int64_t byte_offset = 0;              // added to data (DLPack compatibility)
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
/// allocator on its own.
struct LaunchContext {
  cudaStream_t stream = nullptr;
  int32_t device = 0;
  int32_t sm_version = 0;           // exact: 80, 86, 89, 90, 100, 103, 120, 121
  int32_t num_sms = 0;              // multiprocessor count (or MIG/green-context slice)
  int32_t max_ctas_per_sm = 0;      // 0 = query occupancy
  bool enable_pdl = false;          // set programmatic stream serialization attribute (SM90+)
  bool deterministic = false;       // forbid reductions whose order depends on scheduling/atomics
  bool batch_invariant = false;     // per-row results independent of batch composition
  bool allow_tf32 = false;          // fp32 GEMM inputs may be rounded to tf32
  uint32_t capture_generation = 0;  // incremented by the engine on every graph (re)capture
  ExecutionStrategy strategy = ExecutionStrategy::kModular;
};

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
/// Validate device, dtype, rank and alignment in one call (used by every launcher).
Status check_tensor(const Tensor& t, DType expected, int32_t expected_ndim, size_t alignment_bytes,
                    const char* arg_name) noexcept;
/// Fill sm_version / num_sms for `device` (host query; never call during capture).
Status make_launch_context(int32_t device, cudaStream_t stream, LaunchContext* out) noexcept;

}  // namespace kops
