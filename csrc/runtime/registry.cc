// =============================================================================
// kernel-ops | csrc/runtime/registry.cc
// Capability registry host logic. The table itself is generated from
// registry/capabilities/*.yaml at build time; this file only provides identity
// and lookup over whatever table the generator emits.
// =============================================================================
#include <cstddef>
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

#if defined(KOPS_REGISTRY_HAS_TABLE)
#include "registry_table.inc"
#else
constexpr Capability kGeneratedCapabilities[1] = {};
constexpr size_t kGeneratedCapabilityCount = 0;
#endif

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
  int32_t matched = 0;
  for (size_t i = 0; i < kGeneratedCapabilityCount; ++i) {
    const Capability& c = kGeneratedCapabilities[i];
    if (std::strcmp(c.op_id, op_id) != 0) continue;
    if (arch_id != 0 && c.arch_id != 0 && c.arch_id != arch_id) continue;
    if (matched < max_rows) rows[matched] = &c;
    ++matched;
  }
  *num_rows = matched;
  return Status::kSuccess;
}

Status select(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
              Provider force_provider, bool allow_non_release, const Capability** chosen) {
  (void)spec_kind;
  (void)spec;
  if (op_id == nullptr || chosen == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "select: null argument");
  }
  *chosen = nullptr;
  const Capability* best = nullptr;
  for (size_t i = 0; i < kGeneratedCapabilityCount; ++i) {
    const Capability& c = kGeneratedCapabilities[i];
    if (std::strcmp(c.op_id, op_id) != 0) continue;
    if (c.arch_id != 0 && ctx.arch_id != 0 && c.arch_id != ctx.arch_id) continue;
    if (c.backend != ctx.backend) continue;
    if (force_provider != Provider::kNative && c.provider != force_provider) continue;
    if (!allow_non_release && c.status != CapabilityStatus::kReleaseSupported) continue;
    if (best == nullptr || static_cast<int>(c.status) > static_cast<int>(best->status)) best = &c;
  }
  if (best == nullptr) {
    return detail::set_error(Status::kUnsupportedArch,
                             "select: no candidate for %s on arch %d", op_id, ctx.arch_id);
  }
  *chosen = best;
  return Status::kSuccess;
}

Status explain(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
               char* buffer, size_t buffer_bytes) {
  (void)spec_kind;
  (void)spec;
  if (buffer == nullptr || buffer_bytes == 0) {
    return detail::set_error(Status::kInvalidArgument, "explain: null buffer");
  }
  size_t used = 0;
  used += static_cast<size_t>(std::snprintf(buffer + used, buffer_bytes - used,
                                            "registry: %zu rows, arch %d\n",
                                            kGeneratedCapabilityCount, ctx.arch_id));
  for (size_t i = 0; i < kGeneratedCapabilityCount && used + 1 < buffer_bytes; ++i) {
    const Capability& c = kGeneratedCapabilities[i];
    if (op_id != nullptr && std::strcmp(c.op_id, op_id) != 0) continue;
    used += static_cast<size_t>(std::snprintf(buffer + used, buffer_bytes - used,
                                              "  %s %s arch=%d status=%d\n", c.op_id,
                                              c.kernel_symbol, c.arch_id,
                                              static_cast<int>(c.status)));
  }
  return Status::kSuccess;
}

}  // namespace kops::registry
