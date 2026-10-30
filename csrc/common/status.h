#pragma once
// =============================================================================
// kernel-ops | csrc/common/status.h
// Error model for every host launcher. Kernels never printf-and-return; host
// launchers validate, return a Status, and keep a thread-local detail string.
// Requires (TU): <cstdint>, <cuda_runtime.h>.
// =============================================================================

namespace kops {

/// Return code of every host-side launcher / plan function.
/// Values are ABI: append only, never renumber.
enum class Status : int32_t {
  kSuccess = 0,
  kInvalidArgument = 1,     // null required pointer, negative size, inconsistent shapes
  kUnsupportedDtype = 2,    // dtype combination has no compiled kernel
  kUnsupportedShape = 3,    // head_dim / group size / page size / hidden size outside compiled set
  kUnsupportedLayout = 4,   // stride, NHD/HND, scale layout or packing not supported
  kUnsupportedArch = 5,     // no kernel compiled for this exact SM (e.g. sm_120 vs sm_100)
  kMisaligned = 6,          // pointer / stride alignment requirement violated (usually 16 B)
  kWorkspaceTooSmall = 7,   // caller workspace < WorkspaceRequirements
  kCapacityExceeded = 8,    // runtime size beyond the capacity the plan was prepared for
  kPlanInvalidated = 9,     // plan generation / capture generation / buffer epoch mismatch
  kCudaError = 10,          // CUDA runtime/driver error; see last_cuda_error()
  kNotImplemented = 11,     // declared in the catalog but still `planned`
  kCommError = 12,          // peer access / symmetric memory / transport failure
  kNumericalGuard = 13,     // optional debug guard hit (NaN/Inf in a checked input)
  kInternal = 14,
};

/// Stable lowercase name, e.g. "unsupported_shape".
const char* status_name(Status s) noexcept;
/// Human readable detail of the last non-success Status produced on the calling
/// host thread (e.g. "attention.decode: head_dim_qk=96 not in {64,128,256}").
/// Valid until the next kops call on the same thread.
const char* last_error_message() noexcept;
/// cudaError_t captured together with the last kCudaError.
cudaError_t last_cuda_error() noexcept;
/// Reset the thread-local error slot.
void clear_last_error() noexcept;

}  // namespace kops

// ---- Plain C ABI for FFI consumers (Rust, Go, Zig, C) -----------------------
extern "C" {
typedef int32_t kops_status_t;  // same values as kops::Status
const char* kops_status_name(kops_status_t s);
const char* kops_last_error_message(void);
int32_t kops_abi_version(void);  // == KOPS_ABI_VERSION of the loaded binary
}
