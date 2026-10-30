#pragma once
// =============================================================================
// kernel-ops | csrc/backends/ops.h                               Catalog: X03
// Contract for EXTERNAL providers (FlashInfer, FlashAttention, AITER, CK,
// cuBLASLt/hipBLASLt, DeepGEMM, TRT-LLM, NCCL/RCCL, mscclpp).
//
// Why this exists: a consumer (vLLM / SGLang) chooses a backend per op family at
// init time and must be able to fall back without changing its own code. An
// adapter therefore:
//   1. registers what it can serve (op_id x target x feature bits) *before* any
//      plan is built;
//   2. reports workspace requirements with the same contract as a native plan;
//   3. runs through the same Status/error/epoch rules, so a graph captured with a
//      native kernel can be replaced by an adapter kernel (and vice versa).
// An adapter never bypasses validation: kernel-ops validates the spec, then hands
// the identical argument block to the provider.
//
// Requires: common/*, runtime/registry.h, gemm/gemm_params.cuh,
//           attention/attention_params.cuh, kvcache/kv_layout.cuh.
// =============================================================================

namespace kops::backend {

/// Capabilities an adapter may claim. A registry row is only allowed to be
/// kReleaseSupported if every claimed bit is covered by a passing conformance case.
enum ProviderFeature : uint64_t {
  kFeatNone = 0,
  kFeatGraphSafe = 1ull << 0,        // run() is capture-safe (no alloc/sync/host read)
  kFeatPdl = 1ull << 1,              // honours ctx.enable_pdl
  kFeatDeterministic = 1ull << 2,    // supports ctx.deterministic
  kFeatBatchInvariant = 1ull << 3,   // supports ctx.batch_invariant
  kFeatFp8FNUZ = 1ull << 4,          // understands Fp8Encoding::kFNUZ payloads
  kFeatMultiNode = 1ull << 5,        // communicator spans > 1 node
  kFeatQuantizedReduce = 1ull << 6,  // quickreduce-style quantized allreduce
};

/// What the provider is and where it runs.
struct ProviderInfo {
  registry::Provider provider;
  BackendKind backend;
  const int32_t* arch_ids;      // [num_archs] KOPS_TARGET_* this build actually supports
  int32_t num_archs;
  const char* version;          // provider version string ("0.4.2", "vllm-0.11")
  const char* commit;           // pinned commit / cubin hash; required for non-native providers
  const char* license;          // SPDX id, e.g. "Apache-2.0"
  uint64_t features;            // OR of ProviderFeature
};

/// One op binding. `spec_kind` must match what kernel-ops passes as `spec`
/// (registry::SpecKind); a mismatch is kInvalidArgument, never a reinterpret_cast.
struct AdapterEntry {
  const char* op_id;            // catalog id, e.g. "A03.attention.decode"
  registry::SpecKind spec_kind;
  /// Same contract as a native plan: fills workspace, may reject the spec.
  Status (*workspace)(void* self, const void* spec, const LaunchContext& ctx,
                      WorkspaceRequirements* out);
  /// Enqueue on ctx.stream. `args` is the family argument block (e.g. DecodeParams*).
  Status (*run)(void* self, const void* spec, const void* args, const Workspace& ws,
                const LaunchContext& ctx);
};

/// A provider registers one table. Calls are host-only and must happen before any
/// plan is built (registering during capture is kInvalidArgument).
struct AdapterTable {
  ProviderInfo info;
  const AdapterEntry* entries;
  int32_t num_entries;
  void* self;                   // provider-private state (allocator, handles)
};

Status adapter_register(const AdapterTable* table);
Status adapter_unregister(registry::Provider provider);
/// True when the provider is registered, built for ctx.arch_id, and serves op_id.
bool adapter_available(registry::Provider provider, const char* op_id,
                       const LaunchContext& ctx) noexcept;
/// Load an optional provider module from a shared library (dlopen/LoadLibrary).
/// Returns kNotImplemented when the build has no dynamic-loading support.
Status adapter_load(const char* library_path, const char* symbol_name);
/// Unload every module loaded through adapter_load(). Host-only, never in capture.
Status adapter_unload_all();

/// One conformance case: a spec + input generator seed + the tolerance row of
/// docs/CONTRACTS.md that applies. Generated from registry/*.yaml at build time.
struct ConformanceCase {
  const char* op_id;
  registry::SpecKind spec_kind;
  const void* spec;             // concrete spec (AttentionSpec*, GemmProblem*, ...)
  uint64_t input_seed;          // feeds the fp64 reference and the tensor generator
  uint32_t tolerance_row;       // index into the generated tolerance table
  uint32_t flags;               // bit0: run under graph capture, bit1: run with deterministic
};

/// Result of one case; a registry row cannot become release-supported without
/// at least one passing result per predicate class it claims.
struct ConformanceResult {
  const char* op_id;
  Status status;
  double max_abs_err;
  double max_rel_err;
  int32_t num_launches;         // adapters must not hide multi-launch behaviour
  bool bit_exact;               // true when the tolerance row is exact
  char detail[256];             // first failure message, empty on success
};

Status adapter_conformance_run(const AdapterTable* table, const ConformanceCase* cases,
                               int32_t num_cases, ConformanceResult* results);
/// Aggregate: true only if every case passed and every claimed feature is covered.
bool adapter_conformance_ok(const ConformanceResult* results, int32_t num_cases) noexcept;

}  // namespace kops::backend
