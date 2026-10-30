// =============================================================================
// kernel-ops | csrc/common/status.cc
// Error model implementation: thread-local detail, backend error, C ABI.
// Host-only, no CUDA/HIP calls, no allocation, no exceptions.
// =============================================================================
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"

namespace {

struct ErrorSlot {
  kops::Status status = kops::Status::kSuccess;
  kops::kops_error_t backend_error = 0;
  const char* backend = "none";
  char message[480] = {0};
};

thread_local ErrorSlot g_error;

}  // namespace

namespace kops {

const char* status_name(Status s) noexcept {
  switch (s) {
    case Status::kSuccess: return "success";
    case Status::kInvalidArgument: return "invalid_argument";
    case Status::kUnsupportedDtype: return "unsupported_dtype";
    case Status::kUnsupportedShape: return "unsupported_shape";
    case Status::kUnsupportedLayout: return "unsupported_layout";
    case Status::kUnsupportedArch: return "unsupported_arch";
    case Status::kMisaligned: return "misaligned";
    case Status::kWorkspaceTooSmall: return "workspace_too_small";
    case Status::kCapacityExceeded: return "capacity_exceeded";
    case Status::kPlanInvalidated: return "plan_invalidated";
    case Status::kCudaError: return "backend_error";
    case Status::kNotImplemented: return "not_implemented";
    case Status::kCommError: return "comm_error";
    case Status::kNumericalGuard: return "numerical_guard";
    case Status::kInternal: return "internal";
    case Status::kUnsupportedBackend: return "unsupported_backend";
  }
  return "unknown";
}

const char* last_error_message() noexcept { return g_error.message; }

kops_error_t last_backend_error() noexcept { return g_error.backend_error; }

const char* last_error_backend() noexcept { return g_error.backend; }

void clear_last_error() noexcept {
  g_error.status = Status::kSuccess;
  g_error.backend_error = 0;
  g_error.backend = "none";
  g_error.message[0] = '\0';
}

namespace detail {

Status set_error(Status s, const char* fmt, ...) noexcept {
  g_error.status = s;
  if (fmt != nullptr) {
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(g_error.message, sizeof(g_error.message), fmt, args);
    va_end(args);
  } else {
    g_error.message[0] = '\0';
  }
  return s;
}

Status set_backend_error(Status s, int32_t backend_error, const char* fmt, ...) noexcept {
  g_error.backend_error = backend_error;
  g_error.backend = KOPS_BACKEND_HIP ? "hip" : "cuda";
  g_error.status = s;
  if (fmt != nullptr) {
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(g_error.message, sizeof(g_error.message), fmt, args);
    va_end(args);
  } else {
    g_error.message[0] = '\0';
  }
  return s;
}

}  // namespace detail
}  // namespace kops

extern "C" {

const char* kops_status_name(kops_status_t s) {
  return kops::status_name(static_cast<kops::Status>(s));
}
const char* kops_last_error_message(void) { return kops::last_error_message(); }
const char* kops_last_error_backend(void) { return kops::last_error_backend(); }
int32_t kops_last_backend_error(void) { return kops::last_backend_error(); }
void kops_clear_last_error(void) { kops::clear_last_error(); }
uint32_t kops_abi_version(void) { return KOPS_ABI_VERSION; }

}  // extern "C"
