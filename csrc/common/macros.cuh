#pragma once
// =============================================================================
// kernel-ops | csrc/common/macros.cuh
// Qualifiers, backend, target ids, version tags.
//
// HEADER POLICY (whole repo)
//   * Headers declare; they never #include anything. The translation unit
//     (.cu / .cc) includes the toolkit headers and kernel-ops headers in the
//     dependency order documented in docs/TREE.md ("Include order").
//   * Toolkit prerequisites for every CUDA TU: <cstdint>, <cstddef>, <cuda.h>
//     (CUtensorMap), <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>, <cuda_fp8.h>,
//     <cuda_fp4.h> (CUDA >= 12.8 for __nv_fp8_e8m0 and __nv_fp4x2_e2m1).
//   * Every __global__ kernel declared here is a contract: template params,
//     argument order, shapes, dtypes, aliasing and launch geometry are part of
//     KOPS_ABI_VERSION.
//
// BACKENDS
//   CUDA is the only built backend for now. HIP/AMD target ids are reserved
//   (see "Target ids") and must not be reported as supported. Nothing in the
//   public surface may name a backend type: use kops_stream_t / kops_error_t /
//   kops_graph_t / KOPS_TMA_DESC.
//
// SHAPE NOTATION (used in all docstrings)
//   T   = total tokens in a ragged batch        B   = number of requests/sequences
//   H   = hidden size                           I   = MLP / expert intermediate size
//   Hq  = query heads (local to this TP rank)   Hkv = key/value heads (local)
//   D   = head dim (Dqk / Dvo when they differ) G   = Hq / Hkv (GQA group size)
//   E   = experts (global unless noted)         K   = top-k experts per token
//   V   = vocabulary (local shard unless noted) P   = KV pages, PS = page size
//   S   = number of KV splits                   N/M = generic GEMM dims (D = A[M,K] * B[N,K]^T)
//   W   = wave/sub-group width (32 on CUDA; 64 on CDNA when HIP is enabled);
//         "one wave per row" means W lanes, never a hard-coded 32.
// =============================================================================

#define KOPS_VERSION_MAJOR 0
#define KOPS_VERSION_MINOR 2
#define KOPS_VERSION_PATCH 0
// Bump on any change to a public struct layout, enum value or kernel/launcher signature.
// v2: backend-neutral stream/error/graph/tma types, arch_id + wave_size, Fp8Encoding.
// v3: HybridBranches + merge_hybrid_branches host entry.
// v4: by-value InvFreqTable and HybridBranchTable kernel parameters.
// v6: gemm::EpilogueOp kEpiRowSumSq / kEpiKvAppend + their EpilogueParams fields; ScaledGemmArgs carries
//     the runtime tactic (split_k_mode, split_k_slices, raster, split_k_counters); SplitKMode/RasterOrder
//     are declared before ScaledGemmArgs.
//     sample::SamplingPlan, sample::FusedSamplingParams and the sample_fused / sampling_plan /
//     sampling_fused_workspace_requirements entry points (C++ only, no C mirror).
//     B7: gemm::MmaRowMap / MmaGemmParams, MarlinParams::a_tmp, and the gemm_mma_sync_kernel /
//     gguf_moe_mmq_kernel parameter contracts; B8: runtime mask mode in the prefill parameter blocks.
//     ABI 6 is not released yet, so these share the version instead of opening a v7 line.
#define KOPS_ABI_VERSION 6

// =============================================================================
// Backend
//   CUDA only for now. The HIP value stays reserved so the C ABI keeps its
//   meaning when the AMD path is brought back.
// =============================================================================
#define KOPS_BACKEND_CUDA 1
#define KOPS_BACKEND_HIP 0

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define KOPS_HOST_DEVICE __host__ __device__
#define KOPS_DEVICE __device__ __forceinline__
#define KOPS_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
// Plain host translation unit (bindings, reference, tests): device qualifiers do not exist.
#define KOPS_HOST_DEVICE
#define KOPS_DEVICE inline
#define KOPS_HOST_DEVICE_INLINE inline
#endif

namespace kops {

// ---- Backend-neutral handle types (ABI v2) -----------------------------------
/// Stream/handle owned by the caller (cudaStream_t). Never dereferenced by the
/// library; cast to the backend type inside a launcher.
using kops_stream_t = void*;
/// Backend error code (cudaError_t widened to int32).
using kops_error_t = int32_t;
/// Graph / graph-exec handles (cudaGraph_t).
using kops_graph_t = void*;
/// Event handle (cudaEvent_t).
using kops_event_t = void*;

inline constexpr int32_t kInvalidDevice = -1;
/// Kernel parameter space: 32764 B (CUDA, sm_70+ with CUDA >= 12.1).
inline constexpr int32_t kKernargLimitBytes = 32764;
/// Reserved kernel id: a row without an assigned instantiation.
inline constexpr uint64_t kInvalidKernelId = 0;
/// kernel_id = FNV-1a 64 of the canonical string
/// "<symbol>|<template args>|<arch_id>|<backend>" (see runtime/registry.h).
inline constexpr uint64_t kKernelIdHashFnv1a64 = 0x100000001b3ull;

/// kHip = 1 is reserved for the deferred AMD path; only kCuda is produced.
enum class BackendKind : uint8_t { kCuda = 0, kHip = 1 };

/// Unit of a stride field. Used by kops::Tensor and kv::PagedKVDesc alike: one
/// enum, one meaning, so a view and a page table can never disagree silently.
/// A packed sub-byte dtype (fp4/int4) with kByte counts storage bytes along the
/// packed last dim; with kElement it counts logical elements.
enum class StrideUnit : uint8_t { kElement = 0, kByte = 1 };

}  // namespace kops

// =============================================================================
// Target ids
//   NVIDIA: the SM number    (700, 750, 800, 860, 890, 900, 1000, 1030, 1200, 1210)
//   AMD   : 9000 + gfx id    (gfx90a = 9010, gfx942 = 9420, gfx950 = 9500,
//                             gfx1100 = 11000, gfx1200 = 12000) -- reserved,
//                             no AMD target is built or reported yet.
// Ids are compile-time (device pass) and runtime (LaunchContext::arch_id).
// =============================================================================
#define KOPS_TARGET_SM70 700
#define KOPS_TARGET_SM75 750
#define KOPS_TARGET_SM80 800
#define KOPS_TARGET_SM86 860
#define KOPS_TARGET_SM89 890
#define KOPS_TARGET_SM90A 900
#define KOPS_TARGET_SM100A 1000
#define KOPS_TARGET_SM103A 1030
#define KOPS_TARGET_SM120A 1200
#define KOPS_TARGET_SM121A 1210
#define KOPS_TARGET_GFX90A 9010
#define KOPS_TARGET_GFX942 9420
#define KOPS_TARGET_GFX950 9500
#define KOPS_TARGET_GFX1100 11000
#define KOPS_TARGET_GFX1200 12000

namespace kops {

/// True for every AMD target id (>= 9000; no NVIDIA SM id reaches 9000).
KOPS_HOST_DEVICE_INLINE bool target_is_amd(int32_t arch_id) { return arch_id >= 9000; }
KOPS_HOST_DEVICE_INLINE bool target_is_nvidia(int32_t arch_id) {
  return arch_id > 0 && arch_id < 9000;
}

}  // namespace kops

// =============================================================================
// Capabilities
//   There are deliberately no KOPS_HAS_* feature macros. A kernel that needs
//   wgmma, tcgen05/TMEM, TMA or SM120 block-scaled MMA is compiled for the exact
//   target that provides it (optionally guarded by the compiler's own
//   __CUDA_ARCH_FEAT_* macros); runtime choice goes through
//   kops::runtime::Registry. Never infer a feature from a larger numeric SM or
//   from a neighbouring gfx target: sm_100 is not sm_120, gfx942 is not gfx950.
// =============================================================================

// =============================================================================
// Backend-defined TMA descriptor
//   CUtensorMap (128 B, toolkit prerequisite <cuda.h>). The name stays a macro
//   because it is part of the public operand descriptor contract.
// =============================================================================
#define KOPS_TMA_DESC CUtensorMap

// =============================================================================
// Compile-time architecture tags
// =============================================================================
namespace kops {

/// Compile-time architecture tags. Kernels templated on an ArchTag must only be
/// instantiated in the matching target. SM100 and SM120 are NOT interchangeable
/// even though both report compute capability >= 10.0, and gfx942 is not
/// forward-compatible with gfx950. Gfx tags below are reserved for the deferred
/// HIP path and have no build.
struct Sm70 { static constexpr int kArch = KOPS_TARGET_SM70; };     // V100 (no bf16, no cp.async)
struct Sm75 { static constexpr int kArch = KOPS_TARGET_SM75; };     // T4, RTX 20xx (ldmatrix, no bf16)
struct Sm80 { static constexpr int kArch = KOPS_TARGET_SM80; };     // A100, A30
struct Sm86 { static constexpr int kArch = KOPS_TARGET_SM86; };     // A10, RTX 30xx, A40
struct Sm89 { static constexpr int kArch = KOPS_TARGET_SM89; };     // L4, L40S, RTX 40xx (FP8 mma.sync)
struct Sm90a { static constexpr int kArch = KOPS_TARGET_SM90A; };   // H100, H200, H20, GH200
struct Sm100a { static constexpr int kArch = KOPS_TARGET_SM100A; }; // B200, GB200
struct Sm103a { static constexpr int kArch = KOPS_TARGET_SM103A; }; // B300, GB300
struct Sm120a { static constexpr int kArch = KOPS_TARGET_SM120A; }; // RTX PRO 6000 Blackwell, RTX 50xx
struct Sm121a { static constexpr int kArch = KOPS_TARGET_SM121A; }; // DGX Spark (GB10)
struct Gfx90a { static constexpr int kArch = KOPS_TARGET_GFX90A; }; // MI200 (CDNA2)
struct Gfx942 { static constexpr int kArch = KOPS_TARGET_GFX942; }; // MI300X/MI325X (CDNA3)
struct Gfx950 { static constexpr int kArch = KOPS_TARGET_GFX950; }; // MI350 (CDNA4)
struct Gfx1100 { static constexpr int kArch = KOPS_TARGET_GFX1100; };  // RDNA3
struct Gfx1200 { static constexpr int kArch = KOPS_TARGET_GFX1200; };  // RDNA4

}  // namespace kops
