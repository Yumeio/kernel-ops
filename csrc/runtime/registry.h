#pragma once
// =============================================================================
// kernel-ops | csrc/runtime/registry.h                             Catalog: I01
// Capability registry: every (op, kernel, provider, exact SM, shape/dtype/layout
// predicates) row with its evidence and status. `auto` dispatch only considers
// rows at kReleaseSupported; explain() returns reject reasons for the rest.
// Requires: common/*.
// =============================================================================

namespace kops::registry {

enum class CapabilityStatus : uint8_t {
  kPlanned = 0, kImplemented = 1, kCompiled = 2, kNumericallyValidated = 3, kBenchmarked = 4,
  kReleaseSupported = 5,
};

enum class Provider : uint8_t {
  kNative = 0, kFlashInfer = 1, kCutlass = 2, kCublasLt = 3, kDeepGemm = 4, kTrtllmGen = 5,
  kFlashMLA = 6, kFLA = 7, kTriton = 8, kNccl = 9, kDeepEP = 10,
};

/// One registry row (generated from registry/*.yaml at build time).
struct Capability {
  const char* op_id;             // catalog id + op, e.g. "A03.attention.decode"
  uint64_t kernel_id;            // stable hash of (symbol, template args, arch)
  const char* kernel_symbol;     // mangled-free readable name of the instantiation
  Provider provider;
  const char* provider_commit;   // pinned upstream commit for non-native providers
  int32_t sm;                    // exact SM (80, 86, 89, 90, 100, 103, 120, 121)
  int32_t min_cuda_version;      // e.g. 12080
  ExecutionStrategy strategy;
  CapabilityStatus status;
  const char* predicates;        // canonical predicate string: "dtype_q=bf16;head_dim_qk in {64,128,256};..."
  const char* mutations;         // "kv_pages(slot_mapping)", "residual", "" ...
  const char* evidence;          // links: test ids, benchmark case ids, compile log hash
};

/// Query rows matching an op id and device.
Status find_capabilities(const char* op_id, int32_t sm, const Capability** rows, int32_t max_rows,
                         int32_t* num_rows);
/// Pick the best release-supported candidate for a spec (spec is op-specific, e.g. AttentionSpec*),
/// or an explicitly requested provider; never falls back to a semantics-changing candidate.
Status select(const char* op_id, const void* spec, const LaunchContext& ctx, Provider force_provider,
              bool allow_non_release, const Capability** chosen);
/// Human-readable explanation of why each candidate was accepted / rejected.
Status explain(const char* op_id, const void* spec, const LaunchContext& ctx, char* buffer,
               size_t buffer_bytes);

}  // namespace kops::registry
