#pragma once
// =============================================================================
// kernel-ops | csrc/common/macros.cuh
// Qualifiers, backend detection, target ids, capability gates, version tags.
//
// HEADER POLICY (whole repo)
//   * Headers declare; they never #include anything. The translation unit
//     (.cu / .cc) includes the toolkit headers and kernel-ops headers in the
//     dependency order documented in docs/TREE.md ("Include order").
//   * Toolkit prerequisites for every CUDA TU: <cstdint>, <cstddef>, <cuda.h>
//     (CUtensorMap), <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>, <cuda_fp8.h>,
//     <cuda_fp4.h> (CUDA >= 12.8 for __nv_fp8_e8m0 and __nv_fp4x2_e2m1).
//     For every HIP TU: <cstdint>, <cstddef>, <hip/hip_runtime.h>,
//     <hip/hip_fp16.h>, <hip/hip_bf16.h>, <hip/hip_fp8.h>.
//   * Every __global__ kernel declared here is a contract: template params,
//     argument order, shapes, dtypes, aliasing and launch geometry are part of
//     KOPS_ABI_VERSION.
//
// BACKENDS (ABI v2)
//   One contract, two backends. Nothing in the public surface may name a
//   backend type: use kops_stream_t / kops_error_t / kops_graph_t / KOPS_TMA_DESC.
//   Target ids are integers (see "Target ids"); NVIDIA and AMD ids never alias.
//
// SHAPE NOTATION (used in all docstrings)
//   T   = total tokens in a ragged batch        B   = number of requests/sequences
//   H   = hidden size                           I   = MLP / expert intermediate size
//   Hq  = query heads (local to this TP rank)   Hkv = key/value heads (local)
//   D   = head dim (Dqk / Dvo when they differ) G   = Hq / Hkv (GQA group size)
//   E   = experts (global unless noted)         K   = top-k experts per token
//   V   = vocabulary (local shard unless noted) P   = KV pages, PS = page size
//   S   = number of KV splits                   N/M = generic GEMM dims (D = A[M,K] * B[N,K]^T)
//   W   = wave/sub-group width (32 on CUDA, 64 on CDNA); "one wave per row" means
//         KOPS_WAVE lanes, never a hard-coded 32.
// =============================================================================

#define KOPS_VERSION_MAJOR 0
#define KOPS_VERSION_MINOR 2
#define KOPS_VERSION_PATCH 0
// Bump on any change to a public struct layout, enum value or kernel/launcher signature.
// v2: backend-neutral stream/error/graph/tma types, arch_id + wave_size, Fp8Encoding.
#define KOPS_ABI_VERSION 2

// =============================================================================
// Backend detection
// =============================================================================
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define KOPS_BACKEND_HIP 1
#define KOPS_BACKEND_CUDA 0
#else
#define KOPS_BACKEND_HIP 0
#define KOPS_BACKEND_CUDA 1
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define KOPS_DEVICE_PASS 1
#else
#define KOPS_DEVICE_PASS 0
#endif

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__CUDA_ARCH__)
#define KOPS_HOST_DEVICE __host__ __device__
#define KOPS_DEVICE __device__ __forceinline__
#define KOPS_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
// Plain host translation unit (bindings, reference, tests): device qualifiers do not exist.
#define KOPS_HOST_DEVICE
#define KOPS_DEVICE inline
#define KOPS_HOST_DEVICE_INLINE inline
#endif
#define KOPS_RESTRICT __restrict__

// KOPS_GRID_CONST needs CUDA >= 11.7 and has no HIP equivalent: on HIP the
// parameter is a plain const value and oversized plans must live in memory
// (see KOPS_KERNARG_LIMIT_BYTES).
#if KOPS_BACKEND_HIP || !defined(__CUDACC__)
#define KOPS_GRID_CONST
#else
#define KOPS_GRID_CONST __grid_constant__
#endif

namespace kops {

// ---- Backend-neutral handle types (ABI v2) -----------------------------------
/// Stream/handle owned by the caller (cudaStream_t or hipStream_t). Never
/// dereferenced by the library; cast to the backend type inside a launcher.
using kops_stream_t = void*;
/// Backend error code (cudaError_t / hipError_t widened to int32).
using kops_error_t = int32_t;
/// Graph / graph-exec handles (cudaGraph_t / hipGraph_t).
using kops_graph_t = void*;
/// Event handle (cudaEvent_t / hipEvent_t).
using kops_event_t = void*;

inline constexpr int32_t kInvalidDevice = -1;
/// Kernel parameter space: 32764 B (CUDA, sm_70+ with CUDA >= 12.1) vs ~4 KB (AMD).
inline constexpr int32_t kKernargLimitBytes = KOPS_BACKEND_HIP ? 4096 : 32764;
/// Reserved kernel id: a row without an assigned instantiation.
inline constexpr uint64_t kInvalidKernelId = 0;
/// kernel_id = FNV-1a 64 of the canonical string
/// "<symbol>|<template args>|<arch_id>|<backend>" (see runtime/registry.h).
inline constexpr uint64_t kKernelIdHashFnv1a64 = 0x100000001b3ull;

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
//                             gfx1100 = 11000, gfx1200 = 12000)
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
// Capability gates
//   NVIDIA gates are monotonic *within* the 80..121 ladder but not below it:
//   sm70/sm75 have no bf16 MMA and no async copy, sm75 has ldmatrix.
//   AMD gates are per-family; never infer one gfx target from another.
// =============================================================================
#if KOPS_BACKEND_HIP && KOPS_DEVICE_PASS
#if defined(__gfx90a__)
#define KOPS_ARCH KOPS_TARGET_GFX90A
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define KOPS_ARCH KOPS_TARGET_GFX942
#elif defined(__gfx950__)
#define KOPS_ARCH KOPS_TARGET_GFX950
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
#define KOPS_ARCH KOPS_TARGET_GFX1100
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define KOPS_ARCH KOPS_TARGET_GFX1200
#else
#define KOPS_ARCH 0
#endif
#define KOPS_WAVE 64
#define KOPS_HAS_BF16_MMA 1
#define KOPS_HAS_ASYNC_COPY 1
#define KOPS_HAS_LDMATRIX 0      // no ldmatrix: use ds_read_b128 / global_load_lds
#define KOPS_HAS_DS_READ 1
#define KOPS_HAS_TMA 0
#define KOPS_HAS_CLUSTERS 0
#define KOPS_HAS_PDL 0
#define KOPS_HAS_MULTIMEM 0
#define KOPS_HAS_WGMMA 0
#define KOPS_HAS_TCGEN05 0
#define KOPS_HAS_SM120_BLOCKSCALED 0
#define KOPS_HAS_MFMA 1
#define KOPS_HAS_WMMA 0
#if defined(__gfx90a__)
#define KOPS_HAS_FP8_MMA 0
#define KOPS_HAS_SCALED_FP4 0
#else
#define KOPS_HAS_FP8_MMA 1
#define KOPS_HAS_SCALED_FP4 0
#endif
#if defined(__gfx950__)
#undef KOPS_HAS_SCALED_FP4
#define KOPS_HAS_SCALED_FP4 1
#endif
#elif KOPS_DEVICE_PASS
#define KOPS_ARCH (__CUDA_ARCH__)
#define KOPS_WAVE 32
#define KOPS_HAS_BF16_MMA (KOPS_ARCH >= 800)
#define KOPS_HAS_ASYNC_COPY (KOPS_ARCH >= 800)     // cp.async
#define KOPS_HAS_LDMATRIX (KOPS_ARCH >= 750)
#define KOPS_HAS_DS_READ 0
#define KOPS_HAS_TMA (KOPS_ARCH >= 900)            // cp.async.bulk.tensor
#define KOPS_HAS_CLUSTERS (KOPS_ARCH >= 900)       // thread block clusters, DSMEM
#define KOPS_HAS_PDL (KOPS_ARCH >= 900)            // griddepcontrol.{wait,launch_dependents}
#define KOPS_HAS_MULTIMEM (KOPS_ARCH >= 900)       // multimem.ld_reduce / st (NVLS)
#define KOPS_HAS_MFMA 0
#define KOPS_HAS_WMMA 0
// ---- Arch-specific ("a") features: only when compiled for that exact target --
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define KOPS_HAS_WGMMA 1                           // sm_90a: wgmma.mma_async, setmaxnreg
#else
#define KOPS_HAS_WGMMA 0
#endif
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
#define KOPS_HAS_TCGEN05 1                         // sm_100a/103a: tcgen05.mma, TMEM, 2-CTA MMA
#else
#define KOPS_HAS_TCGEN05 0
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define KOPS_HAS_SM120_BLOCKSCALED 1               // sm_120a/121a: mma.sync block-scaled FP4/FP6/FP8
#else
#define KOPS_HAS_SM120_BLOCKSCALED 0
#endif
#define KOPS_HAS_FP8_MMA (KOPS_ARCH >= 890)        // mma.sync e4m3/e5m2 (Ada+)
#define KOPS_HAS_SCALED_FP4 (KOPS_HAS_TCGEN05 || KOPS_HAS_SM120_BLOCKSCALED)
#else
// Host pass: no device feature is usable; launchers must use the runtime
// LaunchContext::arch_id and the registry, never these macros. KOPS_WAVE still
// reports the backend's wave width because host code formats launch geometry.
#define KOPS_ARCH 0
#define KOPS_WAVE (KOPS_BACKEND_HIP ? 64 : 32)
#define KOPS_HAS_BF16_MMA 0
#define KOPS_HAS_ASYNC_COPY 0
#define KOPS_HAS_LDMATRIX 0
#define KOPS_HAS_DS_READ 0
#define KOPS_HAS_TMA 0
#define KOPS_HAS_CLUSTERS 0
#define KOPS_HAS_PDL 0
#define KOPS_HAS_MULTIMEM 0
#define KOPS_HAS_WGMMA 0
#define KOPS_HAS_TCGEN05 0
#define KOPS_HAS_SM120_BLOCKSCALED 0
#define KOPS_HAS_MFMA 0
#define KOPS_HAS_WMMA 0
#define KOPS_HAS_FP8_MMA 0
#define KOPS_HAS_SCALED_FP4 0
#endif

// =============================================================================
// Backend-defined TMA descriptor
//   CUDA: CUtensorMap (128 B, toolkit prerequisite <cuda.h>).
//   HIP : no TMA; the field is an opaque 128 B blob so the struct layout, and
//         therefore the rest of the ABI, stays identical across backends.
// =============================================================================
#if KOPS_BACKEND_HIP
namespace kops {
struct alignas(64) KopsTmaDesc {
  uint64_t opaque[16];
};
}  // namespace kops
#define KOPS_TMA_DESC ::kops::KopsTmaDesc
#else
#define KOPS_TMA_DESC CUtensorMap
#endif

// =============================================================================
// Compile-time architecture tags
// =============================================================================
namespace kops {

/// Compile-time architecture tags. Kernels templated on an ArchTag must only be
/// instantiated in the matching target. SM100 and SM120 are NOT interchangeable
/// even though both report compute capability >= 10.0, and gfx942 is not
/// forward-compatible with gfx950.
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
