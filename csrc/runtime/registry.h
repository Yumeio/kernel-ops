#pragma once
// =============================================================================
// kernel-ops | csrc/runtime/registry.h                             Catalog: I01
// Capability registry: every (op, kernel, provider, target, shape/dtype/layout
// predicates) row with its evidence and status. `auto` dispatch only considers
// rows at kReleaseSupported; explain() returns reject reasons for the rest.
// Requires: common/*.
//
// Kernel identity (ABI v2): kernel_id = FNV-1a 64 over the canonical string
//   "<kernel_symbol>|<template args>|<arch_id>|<backend>"
// computed by kernel_id_from_string(). The hash function and the string format
// are part of the contract: a generated table must be reproducible from the
// same sources on both backends and from a different compiler.
// =============================================================================

namespace kops::registry {

enum class CapabilityStatus : uint8_t {
  kPlanned = 0, kImplemented = 1, kCompiled = 2, kNumericallyValidated = 3, kBenchmarked = 4,
  kReleaseSupported = 5,
};

/// Who implements a capability. Append-only. Values 0..10 were fixed by ABI v1;
/// v2 appends the AMD and attention providers so one enum describes both
/// backends (the old split with gemm::GemmBackend could not represent Marlin,
/// Machete, FlashAttention or AITER at all).
enum class Provider : uint8_t {
  kNative = 0,
  kFlashInfer = 1,
  kCutlass = 2,
  kCublasLt = 3,
  kDeepGemm = 4,
  kTrtllmGen = 5,
  kFlashMLA = 6,
  kFla = 7,
  kTriton = 8,
  kNccl = 9,
  kDeepEP = 10,
  kFlashAttention = 11,  // FA2 / FA3 / FA4 (Dao-AILab); the default vLLM attention backend
  kMarlin = 12,
  kMachete = 13,
  kRccl = 14,            // AMD equivalent of NCCL
  kMscclpp = 15,         // symmetric-memory collectives (multi-node)
  kCk = 16,              // AMD Composable Kernel
  kAiter = 17,           // AMD AITER
  kCudnn = 18,           // cuDNN frontend attention / convolution (vision towers)
};

/// Discriminant for select()/explain(): the spec pointer must match this kind.
/// Without it the `const void* spec` signature is an unchecked cast.
enum class SpecKind : uint8_t {
  kNone = 0,
  kAttention = 1,     // const attn::AttentionSpec*
  kGemm = 2,          // const gemm::GemmProblem*
  kGroupedGemm = 3,   // const gemm::GroupedGemmParams*
  kMoeLayer = 4,      // const moe::MoELayerParams*
  kKvDesc = 5,        // const kv::PagedKVDesc*
  kState = 6,         // const state::*Params* (family-tagged by op_id)
  kComm = 7,          // const comm::*Params*
};

/// One registry row (generated from registry/*.yaml at build time).
struct Capability {
  const char* op_id;             // catalog id + op, e.g. "A03.attention.decode"
  uint64_t kernel_id;            // stable hash of (symbol, template args, arch, backend)
  const char* kernel_symbol;     // mangled-free readable name of the instantiation
  Provider provider;
  const char* provider_commit;   // pinned upstream commit for non-native providers
  BackendKind backend;           // kCuda | kHip
  int32_t arch_id;               // KOPS_TARGET_* (800, 900, 1000, 9420, 11000, ...)
  int32_t min_toolkit_version;   // e.g. 12080 for CUDA 12.8, 60400 for ROCm 6.4
  ExecutionStrategy strategy;
  CapabilityStatus status;
  const char* predicates;        // canonical predicate string: "dtype_q=bf16;head_dim_qk in {64,128,256};..."
  const char* mutations;         // "kv_pages(slot_mapping)", "residual", "" ...
  const char* evidence;          // links: test ids, benchmark case ids, compile log hash
};

/// FNV-1a 64 of the canonical identity string (see the file header). Exposed so
/// generators and the Python/Rust bindings produce identical ids.
uint64_t kernel_id_from_string(const char* canonical) noexcept;

/// Query rows matching an op id and target.
Status find_capabilities(const char* op_id, int32_t arch_id, const Capability** rows, int32_t max_rows,
                         int32_t* num_rows);
/// Pick the best release-supported candidate for a spec, or an explicitly requested
/// provider; never falls back to a semantics-changing candidate. spec_kind must match
/// the pointer passed in `spec`.
Status select(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
              Provider force_provider, bool allow_non_release, const Capability** chosen);
/// Human-readable explanation of why each candidate was accepted / rejected.
Status explain(const char* op_id, SpecKind spec_kind, const void* spec, const LaunchContext& ctx,
               char* buffer, size_t buffer_bytes);

}  // namespace kops::registry
