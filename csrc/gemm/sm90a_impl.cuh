#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/sm90a_impl.cuh                     Catalog: G01, G03, G06
// B11 SM90a arch cell: the TMA + wgmma bodies behind the *_wgmma_kernel declarations.
//
// Hopper has wgmma.mma_async (one warpgroup = 4 warps issues an m64nNkK MMA that reads both
// operands straight from swizzled shared memory) and TMA (cp.async.bulk.tensor: one thread
// copies a whole box global <-> shared and signals an mbarrier). The pipeline is therefore
// not the B7 cp.async ring: one producer thread feeds a STAGES-deep ring guarded by mbarriers
// and one or two consumer warpgroups run the MMA, with the register file re-split between
// the two roles (setmaxnreg).
//
// Source read before writing (ROADMAP 2.3; provenance in THIRD_PARTY_NOTICES.md):
//   DeepGEMM 5081f85  deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh   pipeline, epilogue
//                     deep_gemm/include/deep_gemm/scheduler/gemm.cuh         persistent raster
//                     deep_gemm/include/deep_gemm/mma/sm90.cuh               descriptor arithmetic
//   CUTLASS (hpc-ops/3rd/cutlass) cute/arch/{mma_sm90_gmma,copy_sm90_tma,mma_sm90_desc}.hpp,
//                     cutlass/arch/{barrier,reg_reconfig}.h                  PTX spellings
// No CuTe / CUTLASS type is used: every instruction below is written as PTX.
//
// Contract per CTA (persistent: CTA b computes tiles b, b + grid, b + 2 grid, ...):
//   * threads: 128 producer (one elected thread issues TMA; the group only exists so that
//     setmaxnreg can hand its registers to the math groups) + 128 math threads for a 64-row
//     tile or 256 for a 128 / 256-row tile (two warpgroups, rows [64 wg, 64 wg + 64) of each
//     128-row wave: kTmaWarpSpecializedCooperative).
//   * smem: [D staging (16-bit outputs only)] [STAGES x A tile] [STAGES x B tile] [2 STAGES mbarriers].
//     One stage holds 128 bytes of every operand row (TILE_K = 64 elements of a 16-bit
//     operand), loaded by TMA with CU_TENSOR_MAP_SWIZZLE_128B, so the wgmma descriptor is
//     the K-major 128B-swizzle one: SBO = 8 rows x 128 B, LBO unused. Every region is
//     1024-byte aligned (the swizzle pattern is a function of the absolute address bits).
//   * out-of-range rows / columns / K tail are zero-filled by the TMA load, so they add exact
//     zeros; the TMA store clips them. No NaN can reach a valid output element.
//   * accumulation: fp32, one k16 atom per 16 K elements in increasing K order, so an output
//     element is the same fp32 sequence whatever the tile shape or the grid is (batch and
//     tile invariant; not bit-identical to the B7 mma.sync body, whose atom is a different
//     instruction).
//   * epilogue: `ops == 0` (the dense inference GEMM): alpha in registers, then
//       - 16-bit D with a 16-byte aligned pointer and row pitch: stmatrix into the swizzled
//         D staging tile + TMA store (BN / 64 boxes);
//       - otherwise (fp32 D, odd pitch): direct global stores from the fragment.
//     `ops != 0`: the documented dense subset (CONTRACTS.md order) element-wise through
//     dev::gemm_epilogue and direct stores. Fragment map of one warp (wgmma m64nNk16 is the
//     m16n8 fragment repeated): element (i, h, c) of warp w, lane l sits at
//       row 16 w + 8 h + l / 4,  column 8 i + 2 (l % 4) + c,  register 4 i + 2 h + c.
//
// Launch contract (host side in gemm/gemm_wgmma.cu):
//   grid.x = min(SMs, tiles), one CTA per SM (__launch_bounds__(threads, 1)), dynamic smem =
//   dense_smem_bytes(). Tensor maps of A / B / D are encoded by the host per launch and
//   passed in GemmTmaParams (`__grid_constant__`). No split-K, no swap-AB, batch == 1.
//
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh, common/tensor.h,
//           common/op_types.cuh, common/quant.cuh, common/device.cuh,
//           common/device_impl.cuh, gemm/gemm_params.cuh, gemm/dense_gemm_kernels.cuh,
//           gemm/grouped_gemm_kernels.cuh (GroupedTmaParams).
//           Device-only: include from a CUDA device pass (a .cu translation unit),
//           after the toolkit headers and common/device_impl.cuh.
// =============================================================================

// The bodies only exist in an exact sm_90a device pass. Every other pass (host, sm_80 ...
// sm_120a) compiles the trap stub; the host launcher refuses those targets first.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define KOPS_SM90A_BODY 1
#else
#define KOPS_SM90A_BODY 0
#endif

namespace kops::gemm {
namespace sm90a {

// -----------------------------------------------------------------------------
// Geometry shared by the host launcher and the device body
// -----------------------------------------------------------------------------

constexpr int kTmaThreads = 128;      // producer warpgroup
constexpr int kTileK16 = 64;          // K elements of one 128-byte stage row, 16-bit operands
constexpr int kTileK8 = 128;          // the same row holds 128 e4m3 elements

__host__ __device__ constexpr int math_threads(int tile_m) { return tile_m <= 64 ? 128 : 256; }
__host__ __device__ constexpr uint32_t align1024(uint32_t x) { return (x + 1023u) & ~1023u; }

/// Dynamic shared memory of one dense 16-bit tile; `d_bytes` is sizeof(output element)
/// (only a 16-bit output has a staging tile for the TMA store).
__host__ __device__ constexpr uint32_t dense_smem_bytes(int tile_m, int tile_n, int stages, int d_bytes) {
  const uint32_t d = d_bytes == 2 ? align1024(static_cast<uint32_t>(tile_m) * tile_n * 2u) : 0u;
  return d + static_cast<uint32_t>(stages) * static_cast<uint32_t>(tile_m + tile_n) * kTileK16 * 2u +
         static_cast<uint32_t>(stages) * 16u;
}

/// Threads of one CTA. Named so the kernel definition can write __launch_bounds__ with a
/// single argument token (tools/check_roadmap.py parses kernel names out of the header text).
template <typename Traits>
struct Threads {
  static constexpr int value = kTmaThreads + (Traits::tile_m <= 64 ? 128 : 256);
};

// -----------------------------------------------------------------------------
// Host: tensor maps. Encoded per launch from the caller's pointers and passed by value in
// GemmTmaParams, so a captured graph replays the same maps (recapture when a pointer changes).
// -----------------------------------------------------------------------------

/// 2-D tiled tensor map over a row-major [outer, inner] tensor whose rows are `outer_stride_bytes`
/// apart. TMA needs a 16-byte aligned base and row pitch; callers check both and report
/// kMisaligned themselves, so a failure here is a driver error.
inline Status encode_tma_2d(KOPS_TMA_DESC* out, CUtensorMapDataType dtype, const void* base, uint64_t inner,
                            uint64_t outer, uint64_t outer_stride_bytes, uint32_t box_inner, uint32_t box_outer,
                            CUtensorMapSwizzle swizzle, const char* who) {
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {outer_stride_bytes};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(out, dtype, 2, const_cast<void*>(base), dims, strides, box, elem_strides,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(r),
                                     "%s: cuTensorMapEncodeTiled failed (CUresult %d; dims %llu x %llu, pitch %llu B, box %u x %u)",
                                     who, static_cast<int>(r), static_cast<unsigned long long>(inner),
                                     static_cast<unsigned long long>(outer),
                                     static_cast<unsigned long long>(outer_stride_bytes), box_inner, box_outer);
  }
  return Status::kSuccess;
}

// Tile schedulers (defined with the device bodies; named here so that the kernel definitions can spell
// them in every compilation pass).
struct TileSched;    // dense: GemmTmaParams
struct GroupSched;   // grouped: GroupedTmaParams

#if KOPS_SM90A_BODY

// -----------------------------------------------------------------------------
// PTX helpers (sm_90a)
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t l;
  asm("mov.u32 %0, %%laneid;" : "=r"(l));
  return l;
}

/// True on exactly one lane of the calling warp.
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0, lane = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(lane), "+r"(pred)
               : "r"(0xFFFFFFFFu));
  return pred != 0u;
}

// ---- mbarrier ------------------------------------------------------------------

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" : : "r"(count), "r"(smem_u32(bar)) : "memory");
}

/// Makes the initialized barriers visible to the async proxy (TMA completes transactions on them).
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

/// Spins until the phase of parity `phase` has completed.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
               "@P1 bra DONE;\n"
               "bra LAB_WAIT;\n"
               "DONE:\n"
               "}\n"
               :
               : "r"(smem_u32(bar)), "r"(phase), "r"(0x989680)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_u32(bar)) : "memory");
}

/// One arrival plus `bytes` expected TMA bytes (atomic: the phase cannot complete in between).
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
               :
               : "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}

/// Arrival on the barrier at the same smem offset of CTA `cta` of this cluster (also valid for the own rank).
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t* bar, uint32_t cta) {
  asm volatile("{\n"
               ".reg .b32 remAddr32;\n"
               "mapa.shared::cluster.u32 remAddr32, %0, %1;\n"
               "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\n"
               "}\n"
               :
               : "r"(smem_u32(bar)), "r"(cta)
               : "memory");
}

// ---- cluster ----------------------------------------------------------------------

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r) :);
  return r;
}

/// Cluster-wide barrier with the relaxed arrive (DeepGEMM cluster_sync_with_relaxed_arrive).
__device__ __forceinline__ void cluster_sync_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

// ---- TMA ------------------------------------------------------------------------

__device__ __forceinline__ void tma_prefetch_desc(const KOPS_TMA_DESC* desc) {
  asm volatile("prefetch.tensormap [%0];" : : "l"(reinterpret_cast<uint64_t>(desc)) : "memory");
}

/// Box at (c0 = inner coordinate, c1 = outer coordinate) -> shared memory; completes
/// box-bytes on `bar` (out-of-range elements are zero-filled and still counted).
__device__ __forceinline__ void tma_load_2d(const KOPS_TMA_DESC* desc, uint64_t* bar, void* smem, int32_t c0,
                                            int32_t c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
               : "memory");
}

/// Same box into the same smem offset of every CTA in `mask` (bit = cluster rank); each of them gets
/// the bytes completed on its own barrier at the offset of `bar`.
__device__ __forceinline__ void tma_load_2d_multicast(const KOPS_TMA_DESC* desc, uint64_t* bar, void* smem, int32_t c0,
                                                      int32_t c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%4, %5}], [%2], %3;"
               :
               : "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "h"(mask), "r"(c0),
                 "r"(c1)
               : "memory");
}

__device__ __forceinline__ void tma_load_3d(const KOPS_TMA_DESC* desc, uint64_t* bar, void* smem, int32_t c0,
                                            int32_t c1, int32_t c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];"
               :
               : "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1),
                 "r"(c2)
               : "memory");
}

/// Shared memory box -> global at (c0, c1); elements outside the tensor are dropped.
__device__ __forceinline__ void tma_store_2d(const KOPS_TMA_DESC* desc, const void* smem, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :
               : "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem)), "r"(c0), "r"(c1)
               : "memory");
}

/// Generic-proxy smem writes -> visible to the async proxy (before a TMA store reads them).
__device__ __forceinline__ void tma_store_fence() { asm volatile("fence.proxy.async.shared::cta;" : : : "memory"); }
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" : : : "memory"); }
template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
}

// ---- wgmma / register file ---------------------------------------------------------

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" : : : "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
}
/// Keeps an accumulator in a register across the async MMA (no reordering past the fence).
__device__ __forceinline__ void fence_operand(float& reg) { asm volatile("" : "+f"(reg) : : "memory"); }

template <uint32_t N>
__device__ __forceinline__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(N));
}
template <uint32_t N>
__device__ __forceinline__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(N));
}

/// Named barrier over `threads` threads (barrier 0 is __syncthreads).
__device__ __forceinline__ void bar_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(id), "r"(threads) : "memory");
}

/// Two 8x8 16-bit matrices from the m16n8 fragment registers; lanes 0-7 / 8-15 give the row addresses.
__device__ __forceinline__ void stsm_x2(void* smem, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
               :
               : "r"(smem_u32(smem)), "r"(r0), "r"(r1)
               : "memory");
}

/// griddepcontrol.wait: with PDL the previous kernel's writes are visible after this; a no-op otherwise.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" : : : "memory"); }

// -----------------------------------------------------------------------------
// wgmma atoms. Operand lists are spelled out because inline asm has no variadic form:
// an m64nNkK instruction takes N / 2 fp32 accumulator registers per thread.
//   wgmma.mma_async.sync.aligned.<shape>.f32.<a>.<b> {d...}, a-desc, b-desc, scale-d, imm-scale-a,
//   imm-scale-b[, imm-trans-a, imm-trans-b];      (the transpose immediates exist for 16-bit types only)
// scale-d = 1: D += A . B^T (the accumulators are zeroed in software per tile).
// -----------------------------------------------------------------------------

#define KOPS_WG_P8(a, b, c, d, e, f, g, h) "%" #a ",%" #b ",%" #c ",%" #d ",%" #e ",%" #f ",%" #g ",%" #h
#define KOPS_WG_P32A                                                                                       \
  KOPS_WG_P8(0, 1, 2, 3, 4, 5, 6, 7) "," KOPS_WG_P8(8, 9, 10, 11, 12, 13, 14, 15) ","                      \
  KOPS_WG_P8(16, 17, 18, 19, 20, 21, 22, 23) "," KOPS_WG_P8(24, 25, 26, 27, 28, 29, 30, 31)
#define KOPS_WG_P32B                                                                                       \
  KOPS_WG_P8(32, 33, 34, 35, 36, 37, 38, 39) "," KOPS_WG_P8(40, 41, 42, 43, 44, 45, 46, 47) ","            \
  KOPS_WG_P8(48, 49, 50, 51, 52, 53, 54, 55) "," KOPS_WG_P8(56, 57, 58, 59, 60, 61, 62, 63)
#define KOPS_WG_P32C                                                                                       \
  KOPS_WG_P8(64, 65, 66, 67, 68, 69, 70, 71) "," KOPS_WG_P8(72, 73, 74, 75, 76, 77, 78, 79) ","            \
  KOPS_WG_P8(80, 81, 82, 83, 84, 85, 86, 87) "," KOPS_WG_P8(88, 89, 90, 91, 92, 93, 94, 95)
#define KOPS_WG_P32D                                                                                       \
  KOPS_WG_P8(96, 97, 98, 99, 100, 101, 102, 103) "," KOPS_WG_P8(104, 105, 106, 107, 108, 109, 110, 111) "," \
  KOPS_WG_P8(112, 113, 114, 115, 116, 117, 118, 119) "," KOPS_WG_P8(120, 121, 122, 123, 124, 125, 126, 127)
#define KOPS_WG_P64 KOPS_WG_P32A "," KOPS_WG_P32B
#define KOPS_WG_P96 KOPS_WG_P64 "," KOPS_WG_P32C
#define KOPS_WG_P128 KOPS_WG_P96 "," KOPS_WG_P32D

#define KOPS_WG_R8(d, o)                                                                                   \
  "+f"(d[(o) + 0]), "+f"(d[(o) + 1]), "+f"(d[(o) + 2]), "+f"(d[(o) + 3]), "+f"(d[(o) + 4]), "+f"(d[(o) + 5]), \
      "+f"(d[(o) + 6]), "+f"(d[(o) + 7])
#define KOPS_WG_R32(d, o) KOPS_WG_R8(d, o), KOPS_WG_R8(d, (o) + 8), KOPS_WG_R8(d, (o) + 16), KOPS_WG_R8(d, (o) + 24)
#define KOPS_WG_R64(d) KOPS_WG_R32(d, 0), KOPS_WG_R32(d, 32)
#define KOPS_WG_R96(d) KOPS_WG_R64(d), KOPS_WG_R32(d, 64)
#define KOPS_WG_R128(d) KOPS_WG_R96(d), KOPS_WG_R32(d, 96)

/// One instruction: NAME(a-desc, b-desc, accumulators, scale_d). IA / IB / IP are the operand numbers
/// of the two descriptors and of the scale-d source, i.e. N / 2, N / 2 + 1, N / 2 + 2. scale_d != 0:
/// D += A . B^T; scale_d == 0: D = A . B^T (the first atom of a promoted 128-K block).
#define KOPS_WGMMA_FN(NAME, SHAPE_TYPES, PLIST, RLIST, IA, IB, IP, TAIL)                                     \
  __device__ __forceinline__ void NAME(uint64_t da, uint64_t db, float* d, int scale_d) {                    \
    asm volatile("{\n"                                                                                       \
                 ".reg .pred p;\n"                                                                           \
                 "setp.ne.b32 p, %" #IP ", 0;\n"                                                             \
                 "wgmma.mma_async.sync.aligned." SHAPE_TYPES " {" PLIST "}, %" #IA ", %" #IB ", p, " TAIL ";\n" \
                 "}\n"                                                                                       \
                 : RLIST                                                                                     \
                 : "l"(da), "l"(db), "r"(scale_d));                                                          \
  }

// 16-bit operands: k16, scale immediates 1 / 1, both operands K-major (transpose immediates 0 / 0).
KOPS_WGMMA_FN(wgmma_bf16_n64, "m64n64k16.f32.bf16.bf16", KOPS_WG_P32A, KOPS_WG_R32(d, 0), 32, 33, 34, "1, 1, 0, 0")
KOPS_WGMMA_FN(wgmma_bf16_n128, "m64n128k16.f32.bf16.bf16", KOPS_WG_P64, KOPS_WG_R64(d), 64, 65, 66, "1, 1, 0, 0")
KOPS_WGMMA_FN(wgmma_bf16_n256, "m64n256k16.f32.bf16.bf16", KOPS_WG_P128, KOPS_WG_R128(d), 128, 129, 130, "1, 1, 0, 0")
KOPS_WGMMA_FN(wgmma_f16_n64, "m64n64k16.f32.f16.f16", KOPS_WG_P32A, KOPS_WG_R32(d, 0), 32, 33, 34, "1, 1, 0, 0")
KOPS_WGMMA_FN(wgmma_f16_n128, "m64n128k16.f32.f16.f16", KOPS_WG_P64, KOPS_WG_R64(d), 64, 65, 66, "1, 1, 0, 0")
KOPS_WGMMA_FN(wgmma_f16_n256, "m64n256k16.f32.f16.f16", KOPS_WG_P128, KOPS_WG_R128(d), 128, 129, 130, "1, 1, 0, 0")
// e4m3 x e4m3: k32, K-major only (no transpose immediates).
KOPS_WGMMA_FN(wgmma_e4m3_n64, "m64n64k32.f32.e4m3.e4m3", KOPS_WG_P32A, KOPS_WG_R32(d, 0), 32, 33, 34, "1, 1")
KOPS_WGMMA_FN(wgmma_e4m3_n128, "m64n128k32.f32.e4m3.e4m3", KOPS_WG_P64, KOPS_WG_R64(d), 64, 65, 66, "1, 1")
KOPS_WGMMA_FN(wgmma_e4m3_n256, "m64n256k32.f32.e4m3.e4m3", KOPS_WG_P128, KOPS_WG_R128(d), 128, 129, 130, "1, 1")

/// m64nNkK atom of an operand element type. One atom consumes 32 bytes of every operand row
/// (k16 for the 16-bit types, k32 for e4m3), so a 128-byte stage row is four atoms whatever the type is.
template <typename E, int N>
struct Wgmma;
#define KOPS_WGMMA_ATOM(E, N, FN)                                                                            \
  template <>                                                                                                \
  struct Wgmma<E, N> {                                                                                       \
    static constexpr int kAcc = N / 2;                                                                       \
    __device__ static __forceinline__ void mma(uint64_t da, uint64_t db, float* d, int scale_d) {            \
      FN(da, db, d, scale_d);                                                                                \
    }                                                                                                        \
  };
KOPS_WGMMA_ATOM(bf16_t, 64, wgmma_bf16_n64)
KOPS_WGMMA_ATOM(bf16_t, 128, wgmma_bf16_n128)
KOPS_WGMMA_ATOM(bf16_t, 256, wgmma_bf16_n256)
KOPS_WGMMA_ATOM(f16_t, 64, wgmma_f16_n64)
KOPS_WGMMA_ATOM(f16_t, 128, wgmma_f16_n128)
KOPS_WGMMA_ATOM(f16_t, 256, wgmma_f16_n256)
KOPS_WGMMA_ATOM(f8e4m3_t, 64, wgmma_e4m3_n64)
KOPS_WGMMA_ATOM(f8e4m3_t, 128, wgmma_e4m3_n128)
KOPS_WGMMA_ATOM(f8e4m3_t, 256, wgmma_e4m3_n256)
#undef KOPS_WGMMA_ATOM

/// Upper word of the smem matrix descriptor of a K-major operand tile whose rows are 128 bytes
/// under the 128B swizzle: stride-byte-offset = 8 rows x 128 B (bits [32, 46), in 16-byte
/// units), leading-byte-offset unused, layout type 1 = SWIZZLE_128B (bits [62, 64)). The lower word is
/// the 16-byte unit address (bits [0, 14)), so a row / K offset is a plain add.
constexpr uint64_t kDescK128Hi = (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);

__device__ __forceinline__ uint64_t desc_k128(uint32_t lo) { return kDescK128Hi | static_cast<uint64_t>(lo); }

/// Two fp32 -> two packed 16-bit outputs (round to nearest even), as one stmatrix register.
template <typename Dt>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  if constexpr (sizeof(Dt) == 2 && std::is_same<Dt, f16_t>::value) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
  } else {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
  }
}

// -----------------------------------------------------------------------------
// Persistent tile scheduler (dense): CTA b takes linear tile indices b, b + grid, ... and
// maps them through the L2-reuse raster of DeepGEMM (groups of 8 or 16 tiles of the primary
// dimension swept along the other one). Every thread of the CTA runs its own copy; the
// sequence only depends on (blockIdx.x, gridDim.x, tile counts), never on timing.
//
// Multicast (cluster of two CTAs, MC != 0): the pair 2c / 2c + 1 always gets two tiles that are
// neighbours along the primary dimension and share the other operand tile, which rank 0 loads
// once into both CTAs. `multicast` is false for the tiles where that is impossible (an odd tail
// of a group): both CTAs then load on their own. `peer_alive` is false when the other CTA of
// the pair has no tile left in this iteration.
// -----------------------------------------------------------------------------

/// What a scheduler hands to the tile body for the current tile (global coordinates):
///   a_row0 / col0 : first row of A and D / first column of D
///   b_row0        : first row of the B tensor map
///   row_end       : rows >= row_end are not written (dense: M; grouped: the end of the group)
///   slot          : operand slot of the per-slot B scales (dense: 0; grouped: the group)
struct TileSched {
  uint32_t num_m, num_n, num_blocks, grid, group;
  uint32_t mc;          // CTAs per cluster sharing a tile: 1 or 2
  bool on_a;            // multicast operand: A (neighbours along N) or B (neighbours along M)
  bool multicast, peer_alive;
  int32_t iter;
  uint32_t tm, tn;
  int64_t a_row0, col0, b_row0, row_end;
  uint32_t slot;

  __device__ __forceinline__ TileSched(const GemmTmaParams& P, uint32_t tile_m, uint32_t tile_n, uint32_t mc_,
                                       bool on_a_) {
    const uint32_t nm = static_cast<uint32_t>(P.num_tiles_m), nn = static_cast<uint32_t>(P.num_tiles_n);
    tm = tile_m;
    tn = tile_n;
    a_row0 = col0 = b_row0 = 0;
    row_end = P.args.m;
    slot = 0;
    num_m = nm;
    num_n = nn;
    num_blocks = nm * nn;
    grid = gridDim.x;
    mc = mc_;
    on_a = on_a_;
    multicast = false;
    peer_alive = true;
    iter = -1;
    const uint32_t prim = on_a ? tile_n : tile_m, sec = on_a ? tile_m : tile_n;
    const uint32_t use8 = 8u * prim + ((grid + 7u) / 8u) * sec;
    const uint32_t use16 = 16u * prim + ((grid + 15u) / 16u) * sec;
    group = use16 < use8 ? 16u : 8u;
  }

  __device__ __forceinline__ bool next() {
    uint32_t mb, nb;
    const uint32_t idx = static_cast<uint32_t>(++iter) * grid + blockIdx.x;
    if (idx >= num_blocks) return false;
    peer_alive = num_n % mc == 0u || num_m % mc == 0u || (idx ^ 1u) < num_blocks;
    const uint32_t primary = on_a ? num_n : num_m;
    const uint32_t secondary = on_a ? num_m : num_n;
    const uint32_t per_group = secondary * group;
    uint32_t first = (idx / per_group) * group;
    uint32_t in_group = idx % per_group;
    uint32_t rows = min(group, primary - first);
    if (mc > 1u && (rows & 1u) != 0u) {
      // odd group tail: the even part keeps its pairs, the last line of tiles runs unpaired
      if (in_group < (rows ^ 1u) * secondary) {
        rows ^= 1u;
      } else {
        in_group -= (rows ^ 1u) * secondary;
        first += rows ^ 1u;
        rows = 1u;
      }
    }
    multicast = mc > 1u && rows != 1u;
    if (on_a) {
      mb = in_group / rows;
      nb = first + in_group % rows;
    } else {
      mb = first + in_group % rows;
      nb = in_group / rows;
    }
    a_row0 = static_cast<int64_t>(mb) * tm;
    col0 = static_cast<int64_t>(nb) * tn;
    b_row0 = col0;
    return true;
  }
};

// -----------------------------------------------------------------------------
// Persistent scheduler of the grouped GEMM: the linear tile index walks the groups in order
// (group g owns ceil(rows_g / tile_m) x num_n tiles, M fastest so that one expert weight tile is
// reused by neighbouring CTAs). The row counts are read from device memory on every launch, so a
// captured graph follows them; empty groups own no tile. Cluster 1x1 only.
// -----------------------------------------------------------------------------

struct GroupSched {
  uint32_t num_n, grid, g, cum, num_groups;
  bool multicast, peer_alive;
  int32_t iter;
  uint32_t tm, tn;
  const int64_t* group_rows;
  const int32_t* masked_rows;
  int64_t max_rows, n;
  int64_t a_row0, col0, b_row0, row_end;
  uint32_t slot;

  __device__ __forceinline__ GroupSched(const GroupedTmaParams& P, uint32_t tile_m, uint32_t tile_n, uint32_t, bool) {
    num_n = static_cast<uint32_t>(P.num_tiles_n);
    grid = gridDim.x;
    g = 0;
    cum = 0;
    num_groups = static_cast<uint32_t>(P.num_groups);
    multicast = false;
    peer_alive = true;
    iter = -1;
    tm = tile_m;
    tn = tile_n;
    group_rows = P.group_rows;
    masked_rows = P.masked_rows;
    max_rows = P.max_rows;
    n = P.args.n;
    a_row0 = col0 = b_row0 = row_end = 0;
    slot = 0;
  }

  __device__ __forceinline__ bool next() {
    const uint32_t idx = static_cast<uint32_t>(++iter) * grid + blockIdx.x;
    int64_t begin = 0, rows = 0;
    uint32_t mt = 0;
    for (;;) {
      if (g >= num_groups) return false;
      if (masked_rows != nullptr) {
        const int64_t r = masked_rows[g];
        rows = r < 0 ? 0 : (r > max_rows ? max_rows : r);
        begin = static_cast<int64_t>(g) * max_rows;
      } else {
        begin = group_rows[g];
        rows = group_rows[g + 1] - begin;
        if (rows < 0) rows = 0;
      }
      mt = static_cast<uint32_t>((rows + tm - 1) / tm);
      if (idx < (cum + mt) * num_n) break;
      cum += mt;
      ++g;
    }
    const uint32_t local = idx - cum * num_n;
    a_row0 = begin + static_cast<int64_t>(local % mt) * tm;
    row_end = begin + rows;
    col0 = static_cast<int64_t>(local / mt) * tn;
    b_row0 = static_cast<int64_t>(g) * n + col0;
    slot = g;
    return true;
  }
};

/// FP8 block scale: an fp32 value or a UE8M0 byte (2^(b - 127)).
__device__ __forceinline__ float load_block_scale(const void* p, int64_t i, bool ue8m0) {
  if (ue8m0) return __uint_as_float(static_cast<uint32_t>(static_cast<const uint8_t*>(p)[i]) << 23);
  return static_cast<const float*>(p)[i];
}

// -----------------------------------------------------------------------------
// Epilogue shared by every tile body
// -----------------------------------------------------------------------------

/// Finishes and stores the fp32 accumulators of one [BM, BN] tile (`acc`: WAVES x NACC per thread, the
/// fragment map of the header comment).
///   xs_row / ws_col : per-token (A) / per-channel (B) operand scales of the FP8 rowwise recipe, or null
///   scale           : per-tensor operand scales folded into one factor (1 when there are none)
///   use_tma         : the 16-bit TMA-store path may be used (aligned D, see the caller)
///   row0 / col0     : global coordinates of the tile; rows >= row_end are never written
template <typename Dt, int BM, int BN, int WAVES, int NACC, int kMath>
__device__ __forceinline__ void tile_epilogue(float* acc, const ScaledGemmArgs& args, const KOPS_TMA_DESC* tma_d,
                                              int64_t row0, int64_t col0, int64_t row_end, uint32_t warp_idx,
                                              uint32_t lane, uint8_t* smem_d, const float* xs_row, const float* ws_col,
                                              float scale, bool use_tma) {
  constexpr int WAVE_M = BM / WAVES;
  constexpr int kStoreBoxes = BN / 64;                 // TMA store boxes of one tile (128-byte rows)
  const EpilogueParams& e = args.epi;
  const int64_t M = row_end, N = args.n, ldd = args.ldd;
  Dt* const dbase = static_cast<Dt*>(args.d);
  const int rl = static_cast<int>(lane >> 2);          // fragment row inside the 8-row half
  const int cl = static_cast<int>(lane & 3u) * 2;      // fragment column pair
  const int64_t r0 = row0 + static_cast<int64_t>(warp_idx) * 16 + rl;
  const int64_t c0 = col0 + cl;
  // The TMA store clips at the end of the tensor only: a tile that ends inside it (the last tile of a
  // group) must not write the rows of the next group and takes the direct stores.
  use_tma = use_tma && (row0 + BM <= row_end || row_end >= args.m);

  // FP8 rowwise with no other epilogue op (the FP8-dynamic checkpoint recipe): the per-token and
  // per-channel operand scales are applied in registers and the tile keeps the store paths below. Each
  // thread owns two rows per wave and two columns per 8-column group of the fragment.
  if (e.ops == 0u && (xs_row != nullptr || ws_col != nullptr)) {
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      float sr[2] = {1.0f, 1.0f};
      if (xs_row != nullptr) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int64_t row = r0 + w * WAVE_M + h * 8;
          sr[h] = xs_row[row < M ? row : M - 1];          // rows past M hold exact zeros: any finite scale
        }
      }
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        float sc[2] = {1.0f, 1.0f};
        if (ws_col != nullptr) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int64_t col = c0 + i * 8 + c;
            sc[c] = ws_col[col < N ? col : N - 1];
          }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[w * NACC + i * 4 + j] *= sr[j >> 1] * sc[j & 1];
      }
    }
    xs_row = nullptr;
    ws_col = nullptr;
  }

  if (e.ops != 0u || xs_row != nullptr || ws_col != nullptr) {
    // General dense subset: element-wise in the documented order (operand scales, then alpha, row / col
    // scale, bias, clamp, softcap, act, residual), direct stores. The values go through a local copy
    // so that the runtime loop does not force `acc` out of its registers in the mainloop.
    float tmp[NACC * WAVES];
#pragma unroll
    for (int i = 0; i < NACC * WAVES; ++i) tmp[i] = acc[i];
    for (int w = 0; w < WAVES; ++w) {
      for (int i = 0; i < NACC / 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int64_t row = r0 + w * WAVE_M + (j >> 1) * 8;
          const int64_t col = c0 + i * 8 + (j & 1);
          if (row >= M || col >= N) continue;
          float v = tmp[w * NACC + i * 4 + j] * scale;
          if (xs_row != nullptr) v *= xs_row[row];
          if (ws_col != nullptr) v *= ws_col[col];
          dev::from_f(dbase[row * ldd + col], dev::gemm_epilogue<Dt>(v, e, row, col));
        }
      }
    }
    return;
  }

  const float alpha = (e.alpha_ptr != nullptr ? e.alpha_ptr[0] : e.alpha) * scale;
  if (alpha != 1.0f) {
#pragma unroll
    for (int i = 0; i < NACC * WAVES; ++i) acc[i] *= alpha;
  }

  if (use_tma) {
    if constexpr (sizeof(Dt) == 2) {
      // The previous tile's TMA store must have read the staging tile before it is overwritten.
      if (threadIdx.x < kStoreBoxes) tma_store_wait<0>();
      bar_sync(1, kMath);
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
#pragma unroll
        for (int i = 0; i < NACC / 4; ++i) {
          // 64-column box `i / 8`, 16-byte chunk `i % 8` of the row, XOR-swizzled by the row (128B
          // pattern). Only lanes 0-15 supply an address: row `lane` of the warp's 16 rows.
          const uint32_t chunk = static_cast<uint32_t>(i % 8) ^ (lane & 7u);
          uint8_t* const p = smem_d + static_cast<uint32_t>(i / 8) * (BM * 128u) +
                             (static_cast<uint32_t>(w * WAVE_M) + warp_idx * 16u + lane) * 128u + chunk * 16u;
          stsm_x2(p, pack2<Dt>(acc[w * NACC + i * 4 + 0], acc[w * NACC + i * 4 + 1]),
                  pack2<Dt>(acc[w * NACC + i * 4 + 2], acc[w * NACC + i * 4 + 3]));
        }
      }
      tma_store_fence();
      bar_sync(1, kMath);
      if (threadIdx.x < kStoreBoxes) {
        tma_store_2d(tma_d, smem_d + threadIdx.x * (BM * 128u), static_cast<int32_t>(col0 + threadIdx.x * 64u),
                     static_cast<int32_t>(row0));
        tma_store_commit();
      }
      __syncwarp();
    }
    return;
  }
#pragma unroll
  for (int w = 0; w < WAVES; ++w) {
#pragma unroll
    for (int i = 0; i < NACC / 4; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = r0 + w * WAVE_M + h * 8;
        const int64_t col = c0 + i * 8;
        if (row >= M || col >= N) continue;
        Dt* const p = dbase + row * ldd + col;
        dev::from_f(p[0], acc[w * NACC + i * 4 + h * 2]);
        if (col + 1 < N) dev::from_f(p[1], acc[w * NACC + i * 4 + h * 2 + 1]);
      }
    }
  }
}

#endif  // KOPS_SM90A_BODY

// -----------------------------------------------------------------------------
// Tile body
// -----------------------------------------------------------------------------

/// Accumulation of one tile body.
///   kDirect   : every atom adds into the output accumulators (16-bit operands; FP8 kFastAccum)
///   kPromoted : FP8, Fp8Accum::kPromoted. The four atoms of a 128-K block accumulate in the wgmma
///               registers, the block sum is multiplied by the block scale sA[m, kb] * sB[n / 128, kb]
///               (1 for the tensorwise / rowwise recipes) and added to fp32 accumulators on the CUDA
///               cores (DeepSeek-V3 "increasing accumulation precision"; DeepGEMM sm90_fp8_gemm_1d2d)
enum class Accum : int { kDirect = 0, kPromoted = 1 };

/// D = epilogue(A . B^T), A [M, K] and B [N, K] K-major. At: operand element type (bf16 / fp16 / e4m3).
/// MC: 0 = one CTA per cluster; 1 = clusters of two CTAs sharing the A tile (neighbours along N,
/// GemmTactic::cluster_n == 2); 2 = sharing the B tile (neighbours along M, cluster_m == 2).
/// PT / Sched: GemmTmaParams + TileSched (dense) or GroupedTmaParams + GroupSched (grouped).
template <typename PT, typename Sched, typename At, typename Dt, int BM, int BN, int STAGES, int MC, Accum kAccum>
__device__ __forceinline__ void gemm_tile(const PT& P) {
#if KOPS_SM90A_BODY
  static_assert(BM == 64 || BM == 128 || BM == 256, "SM90a tile M is 64, 128 or 256");
  static_assert(!(BM > 128 && BN > 128), "one tile dimension must be <= 128 (accumulator registers)");
  static_assert(kAccum == Accum::kDirect || (sizeof(At) == 1 && BN <= 128 && MC == 0),
                "the promoted body is FP8, one 128-column B scale block per tile, cluster 1x1");
  using Mma = Wgmma<At, BN>;
  constexpr bool kPromote = kAccum == Accum::kPromoted;
  constexpr int BK = 128 / static_cast<int>(sizeof(At));   // K elements of one 128-byte stage row
  constexpr int kMath = BM <= 64 ? 128 : 256;
  constexpr int kMathWarps = kMath / 32;
  constexpr int WAVE_M = BM <= 64 ? 64 : 128;          // rows the math warpgroups cover in one pass
  constexpr int WAVES = BM / WAVE_M;
  constexpr int NACC = Mma::kAcc;                      // accumulators per thread and wave
  constexpr bool kD16 = sizeof(Dt) == 2;
  constexpr uint32_t D_BYTES = kD16 ? align1024(static_cast<uint32_t>(BM) * BN * 2u) : 0u;
  constexpr uint32_t A_STAGE = static_cast<uint32_t>(BM) * 128u;
  constexpr uint32_t B_STAGE = static_cast<uint32_t>(BN) * 128u;
  constexpr int kStoreBoxes = BN / 64;                 // TMA store boxes of one tile (128-byte rows)
  constexpr uint32_t kCluster = MC == 0 ? 1u : 2u;     // CTAs that share one operand tile

  // `__shfl_sync` keeps the warp index in one register for the whole warp (DeepGEMM).
  const uint32_t warp_idx = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
  const uint32_t lane = lane_id();

  if (warp_idx == kMathWarps && elect_one()) {
    tma_prefetch_desc(&P.tma_a);
    tma_prefetch_desc(&P.tma_b);
    tma_prefetch_desc(&P.tma_d);
  }
  __syncwarp();

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* const smem_d = smem;
  uint8_t* const smem_a = smem + D_BYTES;
  uint8_t* const smem_b = smem_a + STAGES * A_STAGE;
  uint64_t* const full = reinterpret_cast<uint64_t*>(smem_b + STAGES * B_STAGE);
  uint64_t* const empty = full + STAGES;

  if (warp_idx == kMathWarps + 1 && elect_one()) {
#pragma unroll
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full + i, 1);                           // the producer thread
      mbar_init(empty + i, kCluster * kMathWarps);      // one lane of every math warp of the cluster
    }
    fence_mbar_init();
  }
  // The barriers must be visible to the whole cluster before a peer arrives on them.
  if constexpr (MC != 0) {
    cluster_sync_relaxed();
  } else {
    __syncthreads();
  }
  pdl_wait();

  Sched sched(P, BM, BN, kCluster, MC == 1);
  const uint32_t num_k = static_cast<uint32_t>(P.num_tiles_k);
  uint32_t stage = 0, phase = 0;

  if (warp_idx >= kMathWarps) {
    // ---- producer warpgroup: gives its registers away, one thread issues the TMA loads ----
    // Register split (per SM: 65536). 16-bit / fast FP8: 48 here + 248 x 128 or 224 x 256 math. Promoted
    // FP8 keeps a second accumulator set, so its math groups take 232 x 256 and this group 40 (the
    // proven DeepGEMM 1d2d split; 48 would ask for exactly 65536 and setmaxnreg.inc never returns).
    reg_dealloc<kPromote ? 40 : 48>();
    if (warp_idx == kMathWarps + 2 && elect_one()) {
      while (sched.next()) {
        const int32_t ar = static_cast<int32_t>(sched.a_row0), br = static_cast<int32_t>(sched.b_row0);
        for (uint32_t kb = 0; kb < num_k; ++kb) {
          mbar_wait(empty + stage, phase ^ 1u);
          const int32_t kc = static_cast<int32_t>(kb * BK);
          // The shared operand tile is loaded once, by rank 0, into both CTAs of the cluster; the
          // bytes still complete on each CTA's own barrier, so both expect a full stage.
          if (MC == 1 && sched.multicast) {
            if (cluster_rank() == 0u) {
              tma_load_2d_multicast(&P.tma_a, full + stage, smem_a + stage * A_STAGE, kc, ar, 3);
            }
          } else {
            tma_load_2d(&P.tma_a, full + stage, smem_a + stage * A_STAGE, kc, ar);
          }
          if (MC == 2 && sched.multicast) {
            if (cluster_rank() == 0u) {
              tma_load_2d_multicast(&P.tma_b, full + stage, smem_b + stage * B_STAGE, kc, br, 3);
            }
          } else {
            tma_load_2d(&P.tma_b, full + stage, smem_b + stage * B_STAGE, kc, br);
          }
          mbar_arrive_expect_tx(full + stage, A_STAGE + B_STAGE);
          stage = stage == STAGES - 1 ? 0u : stage + 1u;
          phase ^= static_cast<uint32_t>(stage == 0u);
        }
      }
      // A peer may still arrive on these barriers (and rank 0 may still write into this CTA): one more
      // round of empty waits before the shared memory goes away.
      if constexpr (MC != 0) {
        for (int i = 0; i < STAGES; ++i) {
          mbar_wait(empty + stage, phase ^ 1u);
          stage = stage == STAGES - 1 ? 0u : stage + 1u;
          phase ^= static_cast<uint32_t>(stage == 0u);
        }
      }
    }
  } else {
    // ---- math warpgroups ----
    reg_alloc<kMath == 128 ? 248 : (kPromote ? 232 : 224)>();
    const uint32_t wg = __shfl_sync(0xffffffffu, threadIdx.x / 128, 0);
    const uint32_t a_lo0 = __shfl_sync(0xffffffffu, (smem_u32(smem_a) + wg * (64u * 128u)) >> 4, 0);
    const uint32_t b_lo0 = __shfl_sync(0xffffffffu, smem_u32(smem_b) >> 4, 0);

    const ScaledGemmArgs& args = P.args;
    const int64_t M = args.m;
    // TMA store: 16-bit output, 16-byte aligned base and row pitch (the tensor map needs both).
    const bool use_tma = kD16 && args.epi.ops == 0u && (reinterpret_cast<uintptr_t>(args.d) & 15u) == 0u &&
                         ((args.ldd * static_cast<int64_t>(sizeof(Dt))) & 15) == 0;

    // Operand scales outside the mainloop (FP8 tensorwise / rowwise): per tensor folds into one factor,
    // per token (A) / per channel (B) stay vectors for the epilogue. Block scales are the mainloop's.
    float op_scale = 1.0f;
    const float* xs_row = nullptr;
    const float* ws_col = nullptr;
    bool block_scales = false;
    if constexpr (sizeof(At) == 1) {
      if (args.a.scale != nullptr) {
        if (args.a.granularity == ScaleGranularity::kPerTensor) op_scale *= static_cast<const float*>(args.a.scale)[0];
        else if (args.a.granularity == ScaleGranularity::kPerToken) xs_row = static_cast<const float*>(args.a.scale);
      }
      if (args.b.scale != nullptr) {
        if (args.b.granularity == ScaleGranularity::kPerTensor) op_scale *= static_cast<const float*>(args.b.scale)[0];
        else if (args.b.granularity == ScaleGranularity::kPerChannel) ws_col = static_cast<const float*>(args.b.scale);
      }
      block_scales = kPromote && (args.recipe == GemmRecipe::kFP8Blockwise || args.recipe == GemmRecipe::kFP8BlockwiseUE8M0);
      if (block_scales) xs_row = ws_col = nullptr;
    }
    // Block scale addressing (kPromoted + blockwise): A 1x128 as [M, K/128] row-major or the TMA
    // column-major [K/128, round_up(M, 4)] layout; B 128x128 as [ceil(N/128), K/128] row-major.
    [[maybe_unused]] const bool sa_ue = args.a.scale_dtype == DType::kFloat8E8M0;
    [[maybe_unused]] const bool sb_ue = args.b.scale_dtype == DType::kFloat8E8M0;
    [[maybe_unused]] const bool sa_col = args.a.scale_layout == ScaleLayout::kColMajorTma;
    [[maybe_unused]] const int64_t sa_rstride = sa_col ? 1 : args.a.scale_ld;
    [[maybe_unused]] const int64_t sa_kstride = sa_col ? args.a.scale_ld : 1;

    while (sched.next()) {
      float acc[NACC * WAVES];
#pragma unroll
      for (int i = 0; i < NACC * WAVES; ++i) acc[i] = 0.0f;

      if constexpr (!kPromote) {
        for (uint32_t kb = 0; kb < num_k; ++kb) {
          const uint32_t a_base = a_lo0 + stage * (A_STAGE / 16u);
          const uint32_t b_base = b_lo0 + stage * (B_STAGE / 16u);
          mbar_wait(full + stage, phase);
#pragma unroll
          for (int i = 0; i < NACC * WAVES; ++i) fence_operand(acc[i]);
          wgmma_fence();
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              // rows w * WAVE_M (+ 64 wg, in a_lo0) and byte offset 32 k of the stage row, in 16-byte units
              const uint32_t a_lo = a_base + static_cast<uint32_t>((w * WAVE_M * 128 + k * 32) / 16);
              const uint32_t b_lo = b_base + static_cast<uint32_t>(k * 32 / 16);
              Mma::mma(desc_k128(a_lo), desc_k128(b_lo), acc + NACC * w, 1);
            }
          }
          wgmma_commit();
#pragma unroll
          for (int i = 0; i < NACC * WAVES; ++i) fence_operand(acc[i]);
          wgmma_wait<0>();
          if constexpr (MC == 0) {
            if (lane == 0u) mbar_arrive(empty + stage);
          } else {
            // lanes 0 / 1 release the stage on CTA 0 / 1 of the cluster (both on the own CTA once the peer
            // has run out of tiles), so a producer never overwrites a tile the peer still reads
            if (lane < kCluster) mbar_arrive_cluster(empty + stage, sched.peer_alive ? lane : cluster_rank());
          }
          stage = stage == STAGES - 1 ? 0u : stage + 1u;
          phase ^= static_cast<uint32_t>(stage == 0u);
        }
      } else {
        // ---- promoted FP8: one 128-K block per stage ----
        float part[NACC];
        // This thread's two activation rows of every wave (clamped: rows past M are zero-filled operand
        // rows, their partial sums are exact zeros and only a finite scale is needed).
        int64_t sa_row[WAVES][2];
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int64_t row = sched.a_row0 + w * WAVE_M + static_cast<int64_t>(warp_idx) * 16 + h * 8 +
                                static_cast<int64_t>(lane >> 2);
            sa_row[w][h] = (row < M ? row : M - 1) * sa_rstride;
          }
        }
        const int64_t sb_off = static_cast<int64_t>(sched.slot) * args.b.scale_batch_stride +
                               (sched.col0 / 128) * args.b.scale_ld;
        for (uint32_t kb = 0; kb < num_k; ++kb) {
          const uint32_t a_base = a_lo0 + stage * (A_STAGE / 16u);
          const uint32_t b_base = b_lo0 + stage * (B_STAGE / 16u);
          // Scales first: nothing but registers may be touched between the wgmma fence and the wait.
          float s[WAVES][2];
          if (block_scales) {
            const float sb = load_block_scale(args.b.scale, sb_off + kb, sb_ue);
#pragma unroll
            for (int w = 0; w < WAVES; ++w) {
              s[w][0] = load_block_scale(args.a.scale, sa_row[w][0] + static_cast<int64_t>(kb) * sa_kstride, sa_ue) * sb;
              s[w][1] = load_block_scale(args.a.scale, sa_row[w][1] + static_cast<int64_t>(kb) * sa_kstride, sa_ue) * sb;
            }
          } else {
#pragma unroll
            for (int w = 0; w < WAVES; ++w) s[w][0] = s[w][1] = 1.0f;
          }
          mbar_wait(full + stage, phase);
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
#pragma unroll
            for (int i = 0; i < NACC; ++i) fence_operand(part[i]);
            wgmma_fence();
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const uint32_t a_lo = a_base + static_cast<uint32_t>((w * WAVE_M * 128 + k * 32) / 16);
              const uint32_t b_lo = b_base + static_cast<uint32_t>(k * 32 / 16);
              Mma::mma(desc_k128(a_lo), desc_k128(b_lo), part, k);   // k == 0 overwrites: a fresh block sum
            }
            wgmma_commit();
#pragma unroll
            for (int i = 0; i < NACC; ++i) fence_operand(part[i]);
            wgmma_wait<0>();
            if (w == WAVES - 1 && lane == 0u) mbar_arrive(empty + stage);
#pragma unroll
            for (int i = 0; i < NACC / 4; ++i) {
              acc[w * NACC + i * 4 + 0] += s[w][0] * part[i * 4 + 0];
              acc[w * NACC + i * 4 + 1] += s[w][0] * part[i * 4 + 1];
              acc[w * NACC + i * 4 + 2] += s[w][1] * part[i * 4 + 2];
              acc[w * NACC + i * 4 + 3] += s[w][1] * part[i * 4 + 3];
            }
          }
          stage = stage == STAGES - 1 ? 0u : stage + 1u;
          phase ^= static_cast<uint32_t>(stage == 0u);
        }
      }

      tile_epilogue<Dt, BM, BN, WAVES, NACC, kMath>(acc, args, &P.tma_d, sched.a_row0, sched.col0, sched.row_end, warp_idx,
                                                     lane, smem_d, xs_row, ws_col, op_scale, use_tma);
    }
    // The last tile's TMA store reads the staging tile asynchronously: hold the CTA until it is done.
    if (use_tma && threadIdx.x < kStoreBoxes) tma_store_wait<0>();
  }
#else
  (void)P;
  if (threadIdx.x == 0) __trap();   // never launched: the host refuses every target but sm_90a
#endif
}

}  // namespace sm90a
}  // namespace kops::gemm
