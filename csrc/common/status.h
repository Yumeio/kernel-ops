#pragma once
// =============================================================================
// kernel-ops | csrc/common/status.h
// Error model for every host launcher. Kernels never printf-and-return; host
// launchers validate, return a Status, and keep a thread-local detail string.
// Requires: common/macros.cuh (kops_error_t, KOPS_ABI_VERSION).
//   TU also needs <cstdint> and the backend runtime headers (<cuda_runtime.h> or
//   <hip/hip_runtime.h>); the backend error type is widened to int32 and never
//   appears in this header.
//
// ABI: Status values are append-only. The plain C surface is
// bindings/c_api/kops_c.h; this header is the C++ view of the same contract.
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
  kUnsupportedArch = 5,     // no kernel compiled for this target (e.g. sm_120 vs sm_100, gfx942 vs gfx950)
  kMisaligned = 6,          // pointer / stride alignment requirement violated (usually 16 B)
  kWorkspaceTooSmall = 7,   // caller workspace < WorkspaceRequirements
  kCapacityExceeded = 8,    // runtime size beyond the capacity the plan was prepared for
  kPlanInvalidated = 9,     // plan generation / capture generation / buffer epoch mismatch
  kCudaError = 10,          // backend runtime/driver error; see last_backend_error() (name kept for ABI)
  kNotImplemented = 11,     // declared in the catalog but still `planned`
  kCommError = 12,          // peer access / symmetric memory / transport failure
  kNumericalGuard = 13,     // optional debug guard hit (NaN/Inf in a checked input)
  kInternal = 14,
  kUnsupportedBackend = 15, // feature exists only on the other backend (e.g. PDL, tcgen05, multimem)
};

/// Stable lowercase name, e.g. "unsupported_shape".
const char* status_name(Status s) noexcept;
/// Human readable detail of the last non-success Status produced on the calling
/// host thread (e.g. "attention.decode: head_dim_qk=96 not in {64,128,256}").
/// Valid until the next kops call on the same thread.
const char* last_error_message() noexcept;
/// Backend error code captured together with the last kCudaError (cudaError_t or
/// hipError_t, widened to int32). 0 means "no backend error recorded".
kops_error_t last_backend_error() noexcept;
/// "cuda" | "hip" for the backend that produced the last error, or "none".
const char* last_error_backend() noexcept;
/// Reset the thread-local error slot.
void clear_last_error() noexcept;

namespace detail {
/// Record the last error on the calling thread. Returns `s` so launchers can
/// `return detail::set_error(kInvalidArgument, "x: %s", why);`. printf-style,
/// truncated to 480 bytes; never allocates, never throws.
Status set_error(Status s, const char* fmt, ...) noexcept;
/// Same, plus the backend error code and backend name ("cuda"/"hip").
Status set_backend_error(Status s, int32_t backend_error, const char* fmt, ...) noexcept;
}  // namespace detail

}  // namespace kops

// ---- Plain C ABI for FFI consumers (Rust, Go, Zig, C) -----------------------
// Mirrors bindings/c_api/kops_c.h; keep the two in sync (checked by
// tools/header_check/abi_layout_test.cc and the C-ABI compile check).
extern "C" {
typedef int32_t kops_status_t;  // same values as kops::Status
const char* kops_status_name(kops_status_t s);
const char* kops_last_error_message(void);
const char* kops_last_error_backend(void);
int32_t kops_last_backend_error(void);
uint32_t kops_abi_version(void);  // == KOPS_ABI_VERSION of the loaded binary
void kops_clear_last_error(void);
}
