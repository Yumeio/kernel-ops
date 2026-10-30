// =============================================================================
// kernel-ops | csrc/runtime/registry.cc
// Capability registry host logic. The table itself is generated from
// registry/capabilities/*.yaml at build time; this file only provides identity
// and lookup over whatever table the generator emits.
// =============================================================================
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/runtime/registry.h"

namespace kops::registry {
namespace {

// Weak default table so a build without a generated registry still links.
constexpr Capability kNoCapabilities[1] = {};

}  // namespace

uint64_t kernel_id_from_string(const char* canonical) noexcept {
  if (canonical == nullptr) return kInvalidKernelId;
  uint64_t hash = kKernelIdHashFnv1a64;
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(canonical); *p != '\0'; ++p) {
    hash ^= static_cast<uint64_t>(*p);
    hash *= 0x100000001b3ull;
  }
  // 0 is reserved for "no kernel id".
  return hash == 0 ? 1u : hash;
}

Status find_capabilities(const char* op_id, int32_t arch_id, const Capability** rows,
                         int32_t max_rows, int32_t* num_rows) {
  if (rows == nullptr || num_rows == nullptr || op_id == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "find_capabilities: null argument");
  }
  (void)arch_id;
  (void)max_rows;
  (void)kNoCapabilities;
  *num_rows = 0;
  return detail::set_error(Status::kNotImplemented,
                           "find_capabilities: no generated table in this build (M1)");
}

Status select(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
              Provider force_provider, bool allow_non_release, const Capability** chosen) {
  (void)op_id;
  (void)spec_kind;
  (void)spec;
  (void)ctx;
  (void)force_provider;
  (void)allow_non_release;
  if (chosen == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "select: null out pointer");
  }
  *chosen = nullptr;
  return detail::set_error(Status::kNotImplemented, "select: registry not generated yet");
}

Status explain(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
               char* buffer, size_t buffer_bytes) {
  (void)op_id;
  (void)spec_kind;
  (void)spec;
  (void)ctx;
  if (buffer == nullptr || buffer_bytes == 0) {
    return detail::set_error(Status::kInvalidArgument, "explain: null buffer");
  }
  std::snprintf(buffer, buffer_bytes, "registry: no generated table in this build");
  return Status::kSuccess;
}

}  // namespace kops::registry
