#pragma once
// =============================================================================
// kernel-ops | csrc/common/macros.cuh
// Qualifiers, architecture feature gates, version and arch tags.
//
// HEADER POLICY (whole repo)
//   * Headers declare; they never #include anything. The translation unit
//     (.cu / .cc) includes the CUDA toolkit headers and kernel-ops headers in
//     the dependency order documented in docs/TREE.md ("Include order").
//   * Toolkit prerequisites for every TU: <cstdint>, <cstddef>, <cuda.h>
//     (CUtensorMap), <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>, <cuda_fp8.h>,
//     <cuda_fp4.h> (CUDA >= 12.8 for __nv_fp8_e8m0 and __nv_fp4x2_e2m1).
//   * Every __global__ kernel declared here is a contract: template params,
//     argument order, shapes, dtypes, aliasing and launch geometry are part of
//     KOPS_ABI_VERSION.
//
// SHAPE NOTATION (used in all docstrings)
//   T   = total tokens in a ragged batch        B   = number of requests/sequences
//   H   = hidden size                           I   = MLP / expert intermediate size
//   Hq  = query heads (local to this TP rank)   Hkv = key/value heads (local)
//   D   = head dim (Dqk / Dvo when they differ) G   = Hq / Hkv (GQA group size)
//   E   = experts (global unless noted)         K   = top-k experts per token
//   V   = vocabulary (local shard unless noted) P   = KV pages, PS = page size
//   S   = number of KV splits                   N/M = generic GEMM dims (D = A[M,K] * B[N,K]^T)
// =============================================================================

#define KOPS_VERSION_MAJOR 0
#define KOPS_VERSION_MINOR 1
#define KOPS_VERSION_PATCH 0
// Bump on any change to a public struct layout, enum value or kernel/launcher signature.
#define KOPS_ABI_VERSION 1

#define KOPS_HOST_DEVICE __host__ __device__
#define KOPS_DEVICE __device__ __forceinline__
#define KOPS_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define KOPS_RESTRICT __restrict__

#if defined(__CUDA_ARCH__)
#define KOPS_DEVICE_PASS 1
#define KOPS_ARCH __CUDA_ARCH__
#else
#define KOPS_DEVICE_PASS 0
#define KOPS_ARCH 0
#endif

// ---- Forward-compatible features (true on every arch >= threshold) ----------
#define KOPS_HAS_CP_ASYNC (KOPS_ARCH >= 800)      // cp.async, mma.sync bf16
#define KOPS_HAS_FP8_MMA (KOPS_ARCH >= 890)       // mma.sync e4m3/e5m2 (Ada+)
#define KOPS_HAS_TMA (KOPS_ARCH >= 900)           // cp.async.bulk.tensor
#define KOPS_HAS_CLUSTERS (KOPS_ARCH >= 900)      // thread block clusters, DSMEM
#define KOPS_HAS_PDL (KOPS_ARCH >= 900)           // griddepcontrol.{wait,launch_dependents}
#define KOPS_HAS_MULTIMEM (KOPS_ARCH >= 900)      // multimem.ld_reduce / st (NVLS)

// ---- Arch-specific ("a") features: only when compiled for that exact target --
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define KOPS_HAS_WGMMA 1                          // sm_90a: wgmma.mma_async, setmaxnreg
#else
#define KOPS_HAS_WGMMA 0
#endif
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
#define KOPS_HAS_TCGEN05 1                        // sm_100a/103a: tcgen05.mma, TMEM, 2-CTA MMA
#else
#define KOPS_HAS_TCGEN05 0
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define KOPS_HAS_SM120_BLOCKSCALED 1              // sm_120a/121a: mma.sync block-scaled FP4/FP6/FP8
#else
#define KOPS_HAS_SM120_BLOCKSCALED 0
#endif

namespace kops {

/// Compile-time architecture tags. Kernels templated on an ArchTag must only be
/// instantiated in the matching -gencode target; SM100 and SM120 are NOT
/// interchangeable even though both report compute capability >= 10.0.
struct Sm80 { static constexpr int kSm = 80; };    // A100, A30 (also runs on 86/87/89 when not Ada-specific)
struct Sm86 { static constexpr int kSm = 86; };    // A10, RTX 30xx, A40
struct Sm89 { static constexpr int kSm = 89; };    // L4, L40S, RTX 40xx (FP8 mma.sync)
struct Sm90a { static constexpr int kSm = 90; };   // H100, H200, H20, GH200
struct Sm100a { static constexpr int kSm = 100; }; // B200, GB200
struct Sm103a { static constexpr int kSm = 103; }; // B300, GB300
struct Sm120a { static constexpr int kSm = 120; }; // RTX PRO 6000 Blackwell, RTX 50xx
struct Sm121a { static constexpr int kSm = 121; }; // DGX Spark (GB10)

}  // namespace kops
