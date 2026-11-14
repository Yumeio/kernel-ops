// =============================================================================
// kernel-ops | csrc/gemm/gemm_wgmma.cu
// B11 row 1: host side of the SM90a (Hopper) dense GEMM cell.
//   gemm_wgmma_kernel      bf16 / fp16 dense GEMM on TMA + wgmma, warp-specialized, persistent
// B11 row 2 (same pipeline, e4m3 atoms; kept here so that the Hopper tile table, validation and
// tensor-map code exist once; fp8_gemm() in fp8_gemm.cu forwards an SM90a tactic to wgmma_gemm):
//   grouped_gemm_wgmma_kernel (B11 row 7)  the same tile body with the grouped scheduler: kContiguous /
//                          kMasked, bf16 / fp16 and FP8 blockwise (host entry wgmma_grouped_gemm)
//   fp8_gemm_wgmma_kernel  <kFP8Rowwise, kFastAccum>   tensorwise / rowwise, every atom adds into the
//                                                      output accumulators, scales in the epilogue
//                          <kFP8Blockwise, kPromoted>  Fp8Accum::kPromoted of every FP8 recipe: the block
//                                                      sum of each 128 K is moved to fp32 accumulators on
//                                                      the CUDA cores, multiplied by sA[m, kb] * sB[n/128, kb]
//                                                      for kFP8Blockwise / kFP8BlockwiseUE8M0 (DeepSeek-V3)
// The device body is the arch cell csrc/gemm/sm90a_impl.cuh; this file holds the kernel
// definition, the explicit instantiation table (compiled tiles), tensor-map encoding,
// validation and the launch.
// Host entry (ops.h): wgmma_gemm. scaled_gemm() / gemm_with_tactic() dispatch here when the
// tactic names an SM90a mainloop (kTmaWarpSpecialized / kTmaWarpSpecializedCooperative); a call
// without such a tactic keeps the B7 mma.sync path, exactly like the SM120a cell.
//
// Contract: csrc/gemm/dense_gemm_kernels.cuh (gemm_wgmma_kernel), csrc/gemm/gemm_params.cuh
// (GemmTmaParams), csrc/gemm/ops.h.
//
// Numerics. Products and accumulation happen inside wgmma (fp32 accumulators), one m64nNk16
// atom per 16 K elements in increasing K order, so one output element is the same fp32 sequence
// in every compiled tile and for every grid size: tiles agree bitwise with each other and a row
// alone equals the same row inside a batch. The epilogue is fp32 and rounds once to the output
// dtype. No atomics, no split-K.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/fp8_gemm_kernels.cuh"
#include "csrc/gemm/int8_gemm_kernels.cuh"
#include "csrc/gemm/fp4_gemm_kernels.cuh"
#include "csrc/gemm/mixed_input_gemm_kernels.cuh"
#include "csrc/gemm/gguf_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/gemm/gemm_prep_kernels.cuh"
#include "csrc/gemm/ops.h"
#include "csrc/gemm/sm90a_impl.cuh"

namespace kops::gemm {

// =============================================================================
// gemm_wgmma_kernel
// =============================================================================

/// One CTA per SM (min blocks 1): the math warpgroups raise their register budget with setmaxnreg.
template <typename Traits>
__global__ void __launch_bounds__(sm90a::Threads<Traits>::value, 1)
    gemm_wgmma_kernel(const __grid_constant__ GemmTmaParams params) {
  static_assert(Traits::mainloop == MainloopSchedule::kTmaWarpSpecialized ||
                    Traits::mainloop == MainloopSchedule::kTmaWarpSpecializedCooperative,
                "gemm_wgmma_kernel serves the SM90a warp-specialized mainloops");
  static_assert(Traits::tile_k == sm90a::kTileK16, "a 16-bit stage row is 128 bytes = 64 K elements");
  static_assert(Traits::cluster_m * Traits::cluster_n <= 2, "SM90a clusters are 1x1, 1x2 (A multicast) or 2x1 (B multicast)");
  sm90a::gemm_tile<GemmTmaParams, sm90a::TileSched, typename Traits::a_type, typename Traits::d_type, Traits::tile_m,
                   Traits::tile_n, Traits::stages, Traits::cluster_n == 2 ? 1 : (Traits::cluster_m == 2 ? 2 : 0),
                   sm90a::Accum::kDirect>(params);
}

// =============================================================================
// fp8_gemm_wgmma_kernel
// =============================================================================

/// The recipe's scales are resolved at run time from ScaledGemmArgs (per tensor / per token / per
/// channel in the epilogue, 1x128 x 128x128 blocks in the promoted mainloop), so two instantiations per
/// tile cover the four FP8 recipes: <kFP8Rowwise, kFastAccum> and <kFP8Blockwise, kPromoted>.
template <typename Traits, GemmRecipe kRecipe, Fp8Accum kAccum>
__global__ void __launch_bounds__(sm90a::Threads<Traits>::value, 1)
    fp8_gemm_wgmma_kernel(const __grid_constant__ GemmTmaParams params) {
  static_assert((kRecipe == GemmRecipe::kFP8Rowwise && kAccum == Fp8Accum::kFastAccum) ||
                    (kRecipe == GemmRecipe::kFP8Blockwise && kAccum == Fp8Accum::kPromoted),
                "fp8_gemm_wgmma_kernel is instantiated as <kFP8Rowwise, kFastAccum> and <kFP8Blockwise, kPromoted>");
  static_assert(Traits::tile_k == sm90a::kTileK8, "an e4m3 stage row is 128 bytes = 128 K elements");
  sm90a::gemm_tile<GemmTmaParams, sm90a::TileSched, f8e4m3_t, typename Traits::d_type, Traits::tile_m, Traits::tile_n,
                   Traits::stages, 0,
                   kAccum == Fp8Accum::kPromoted ? sm90a::Accum::kPromoted : sm90a::Accum::kDirect>(params);
}

// =============================================================================
// grouped_gemm_wgmma_kernel
// =============================================================================

/// Same tile body with the grouped scheduler. The layout (kContiguous / kMasked) is resolved at run time
/// from GroupedTmaParams, so kLayout = kContiguous serves both; kRecipe / kAccum name the accumulation:
/// <kBF16, kFastAccum> for the 16-bit operand types, <kFP8Blockwise, kPromoted> for the FP8 block recipes.
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout, Fp8Accum kAccum>
__global__ void __launch_bounds__(sm90a::Threads<Traits>::value, 1)
    grouped_gemm_wgmma_kernel(const __grid_constant__ GroupedTmaParams params) {
  static_assert(kLayout == GroupedLayout::kContiguous, "the layout is a run-time property of GroupedTmaParams");
  static_assert((kRecipe == GemmRecipe::kBF16 && kAccum == Fp8Accum::kFastAccum) ||
                    (kRecipe == GemmRecipe::kFP8Blockwise && kAccum == Fp8Accum::kPromoted),
                "grouped_gemm_wgmma_kernel is instantiated as <kBF16, kFastAccum> and <kFP8Blockwise, kPromoted>");
  sm90a::gemm_tile<GroupedTmaParams, sm90a::GroupSched, typename Traits::a_type, typename Traits::d_type, Traits::tile_m,
                   Traits::tile_n, Traits::stages, 0,
                   kAccum == Fp8Accum::kPromoted ? sm90a::Accum::kPromoted : sm90a::Accum::kDirect>(params);
}

// =============================================================================
// Host
// =============================================================================
namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

/// Compiled tiles. `stages` is the deepest ring that fits the 227 KB opt-in next to the D staging tile
/// (the rule of DeepGEMM's SM90 pipeline config); a 64-row tile runs one math warpgroup
/// (kTmaWarpSpecialized), a 128 / 256-row tile two on each 128-row wave (kTmaWarpSpecializedCooperative).
struct Sm90Tile {
  int16_t tm, tn;
  int8_t stages;
};
constexpr Sm90Tile kSm90DenseTiles[] = {{64, 128, 8}, {64, 256, 4}, {128, 128, 6}, {128, 256, 3}, {256, 128, 3}};
constexpr int kSm90NumDenseTiles = static_cast<int>(sizeof(kSm90DenseTiles) / sizeof(kSm90DenseTiles[0]));

MainloopSchedule sm90_tile_mainloop(const Sm90Tile& t) {
  return t.tm <= 64 ? MainloopSchedule::kTmaWarpSpecialized : MainloopSchedule::kTmaWarpSpecializedCooperative;
}

/// Largest dynamic shared memory one block may opt into on this device (cached).
int smem_limit(int device) {
  static std::mutex mu;
  static std::map<int, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;
  int v = 48 * 1024;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cache[device] = v;
  return v;
}

int sm_count(const LaunchContext& ctx) {
  if (ctx.compute_units > 0) return ctx.compute_units;
  int v = 1;
  cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, ctx.device);
  return v;
}

/// Epilogue ops of the dense SM90a cell: the dense subset of CONTRACTS.md. The grouped / fused-MoE /
/// quantizing / KV-append epilogues belong to the B7 body and are refused instead of being dropped.
constexpr uint32_t kSm90EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                     kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

Status sm90_check_epilogue(const EpilogueParams& e, DType d_dtype, const char* who) {
  if ((e.ops & ~kSm90EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "%s: epilogue ops 0x%x have no SM90a body (B7 mma.sync serves them)",
                             who, e.ops & ~kSm90EpiAllowed);
  }
  if (((e.ops & kEpiRowScale) != 0u && e.row_scale == nullptr) || ((e.ops & kEpiColScale) != 0u && e.col_scale == nullptr) ||
      ((e.ops & kEpiBias) != 0u && e.bias == nullptr) || ((e.ops & kEpiResidualAdd) != 0u && e.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "%s: epilogue operand missing (residual is read in the output dtype)", who);
  }
  if ((e.ops & kEpiBias) != 0u && e.bias_dtype != DType::kBFloat16 && e.bias_dtype != DType::kFloat16 &&
      e.bias_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: bias must be bf16, fp16 or fp32", who);
  }
  if ((e.ops & kEpiOutFp32) != 0u && d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "%s: kEpiOutFp32 needs an fp32 output", who);
  }
  return Status::kSuccess;
}

/// Tile of a tactic: the compiled (tile_m, tile_n) pairs, with the mainloop name that goes with the
/// tile height. tile_k is fixed by the atom (128 bytes of a row) and ignored, as on the B7 path.
Status sm90_tile_of(const GemmTactic& t, const char* who, int* out_id) {
  if (t.mainloop == MainloopSchedule::kTmaWarpSpecializedPingpong) {
    return detail::set_error(Status::kNotImplemented,
                             "%s: the SM90a ping-pong mainloop is not implemented (kTmaWarpSpecialized for 64-row "
                             "tiles, kTmaWarpSpecializedCooperative for 128 / 256-row tiles)", who);
  }
  int id = -1;
  for (int i = 0; i < kSm90NumDenseTiles; ++i) {
    if (kSm90DenseTiles[i].tm == t.tile_m && kSm90DenseTiles[i].tn == t.tile_n) id = i;
  }
  if (id < 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: no SM90a tile %dx%d (compiled: 64x128, 64x256, 128x128, 128x256, 256x128)", who,
                             t.tile_m, t.tile_n);
  }
  if (sm90_tile_mainloop(kSm90DenseTiles[id]) != t.mainloop) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: tile %dx%d runs %s", who, t.tile_m, t.tile_n,
                             t.tile_m <= 64 ? "one math warpgroup (kTmaWarpSpecialized)"
                                            : "two cooperative math warpgroups (kTmaWarpSpecializedCooperative)");
  }
  if (t.stages != 0 && t.stages != kSm90DenseTiles[id].stages) {
    return detail::set_error(Status::kUnsupportedShape, "%s: tile %dx%d is compiled with %d stages, not %d", who,
                             t.tile_m, t.tile_n, kSm90DenseTiles[id].stages, t.stages);
  }
  *out_id = id;
  return Status::kSuccess;
}

/// Operand / problem validation of the 16-bit cell, then the tile choice. Argument errors first (they
/// are meaningful on any GPU); the exact-SM check is the caller's last step.
Status sm90_check_args(const ScaledGemmArgs& a, const GemmTactic& t, int eb, const char* who, int* tile_id) {
  if (a.m < 1 || a.n < 1 || a.k < 1 || a.batch < 1) {
    return detail::set_error(Status::kInvalidArgument, "%s: empty problem", who);
  }
  if (a.m > 0x7FFFFFFF || a.n > 0x7FFFFFFF || a.k > 0x7FFFFFFF) {
    return detail::set_error(Status::kUnsupportedShape, "%s: extents beyond int32", who);
  }
  if (a.a.data == nullptr || a.b.data == nullptr || a.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: null operand", who);
  }
  if (a.a.major != MajorOrder::kKMajor || a.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kNotImplemented, "%s: MN-major operands (weight_transpose first)", who);
  }
  if (a.a.ld < a.k || a.b.ld < a.k || a.ldd < a.n) {
    return detail::set_error(Status::kInvalidArgument, "%s: leading dimensions smaller than the extents", who);
  }
  if (a.workspace != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: pass workspace through the Workspace argument", who);
  }
  if (a.batch != 1) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM90a cell has no batched tensor maps yet (batch == 1)", who);
  }
  // TMA reads the operands: 16-byte aligned base and row pitch. The output may be anything (an
  // unaligned D takes the direct-store epilogue).
  if (!aligned(a.a.data, 16) || !aligned(a.b.data, 16) || (a.a.ld * eb) % 16 != 0 || (a.b.ld * eb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned,
                             "%s: SM90a operands are read by TMA and need a 16 B aligned pointer and row pitch", who);
  }
  if (t.split_k_mode != SplitKMode::kNone || t.split_k_slices > 1) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM90a cell has no split-K", who);
  }
  if (t.swap_ab) return detail::set_error(Status::kNotImplemented, "%s: swap_ab has no SM90a body yet", who);
  // Cluster: 1x1, or two CTAs sharing one operand tile by TMA multicast (cluster_n == 2: A, the pair is
  // neighbours along N; cluster_m == 2: B). cluster_k is unused. Only the 128 / 256-row tiles and a
  // 16-bit output are compiled with a cluster.
  const int cm = t.cluster_m < 1 ? 1 : t.cluster_m, cn = t.cluster_n < 1 ? 1 : t.cluster_n;
  if (cm * cn > 2 || t.cluster_k > 1) {
    return detail::set_error(Status::kUnsupportedShape, "%s: SM90a clusters are 1x1, 1x2 or 2x1 (found %dx%dx%d)", who, cm,
                             cn, t.cluster_k);
  }
  if (t.tma_multicast != (cm * cn > 1)) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: GemmTactic::tma_multicast must be set exactly when the cluster has two CTAs", who);
  }
  const Status st = sm90_tile_of(t, who, tile_id);
  if (st != Status::kSuccess) return st;
  if (cm * cn > 1 && (t.tile_m < 128 || a.d_dtype == DType::kFloat32)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: SM90a multicast is compiled for the 128 / 256-row tiles with a 16-bit output", who);
  }
  return Status::kSuccess;
}

template <typename At, typename Dt, int TM, int TN, int ST, int CM, int CN>
Status launch_dense_tile(const GemmTmaParams& p, unsigned grid, cudaStream_t s) {
  using Tr = GemmTraits<At, At, Dt, float, TM, TN, sm90a::kTileK16, CM, CN, ST,
                        TM <= 64 ? MainloopSchedule::kTmaWarpSpecialized
                                 : MainloopSchedule::kTmaWarpSpecializedCooperative,
                        EpilogueSchedule::kTmaStore, Sm90a>;
  constexpr size_t kSmem = sm90a::dense_smem_bytes(TM, TN, ST, static_cast<int>(sizeof(Dt)));
  static const cudaError_t attr = cudaFuncSetAttribute(gemm_wgmma_kernel<Tr>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "gemm_wgmma_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  if constexpr (CM * CN > 1) {
    // Clusters of two CTAs (the grid is even: see sm90_dense_gemm).
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(static_cast<unsigned>(sm90a::Threads<Tr>::value));
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = s;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, gemm_wgmma_kernel<Tr>, p);
    if (err != cudaSuccess) {
      return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "gemm_wgmma_kernel: %s",
                                       cudaGetErrorString(err));
    }
  } else {
    gemm_wgmma_kernel<Tr><<<grid, static_cast<unsigned>(sm90a::Threads<Tr>::value), kSmem, s>>>(p);
  }
  return launch_error("gemm_wgmma_kernel");
}

/// Explicit instantiation table: (tile, stages) x cluster. `mc` is 0 (1x1), 1 (1x2: A multicast) or
/// 2 (2x1: B multicast); clusters exist for the 128 / 256-row tiles with a 16-bit output only.
template <typename At, typename Dt>
Status dense_tile_id(int id, int mc, const GemmTmaParams& p, unsigned grid, cudaStream_t s) {
  if (mc == 0) {
    switch (id) {
      case 0: return launch_dense_tile<At, Dt, 64, 128, 8, 1, 1>(p, grid, s);
      case 1: return launch_dense_tile<At, Dt, 64, 256, 4, 1, 1>(p, grid, s);
      case 2: return launch_dense_tile<At, Dt, 128, 128, 6, 1, 1>(p, grid, s);
      case 3: return launch_dense_tile<At, Dt, 128, 256, 3, 1, 1>(p, grid, s);
      case 4: return launch_dense_tile<At, Dt, 256, 128, 3, 1, 1>(p, grid, s);
      default: break;
    }
  } else if constexpr (sizeof(Dt) == 2) {
    if (mc == 1) {
      switch (id) {
        case 2: return launch_dense_tile<At, Dt, 128, 128, 6, 1, 2>(p, grid, s);
        case 3: return launch_dense_tile<At, Dt, 128, 256, 3, 1, 2>(p, grid, s);
        case 4: return launch_dense_tile<At, Dt, 256, 128, 3, 1, 2>(p, grid, s);
        default: break;
      }
    } else {
      switch (id) {
        case 2: return launch_dense_tile<At, Dt, 128, 128, 6, 2, 1>(p, grid, s);
        case 3: return launch_dense_tile<At, Dt, 128, 256, 3, 2, 1>(p, grid, s);
        case 4: return launch_dense_tile<At, Dt, 256, 128, 3, 2, 1>(p, grid, s);
        default: break;
      }
    }
  }
  return detail::set_error(Status::kInternal, "gemm: SM90a dense tile id %d cluster mode %d", id, mc);
}

template <typename At>
Status dense_out(int id, int mc, const GemmTmaParams& p, unsigned grid, cudaStream_t s) {
  if (p.args.d_dtype == DType::kFloat32) return dense_tile_id<At, float>(id, mc, p, grid, s);
  return dense_tile_id<At, At>(id, mc, p, grid, s);
}

Status sm90_dense_gemm(const ScaledGemmArgs& args, const GemmTactic& t, const LaunchContext& ctx) {
  const DType want = args.recipe == GemmRecipe::kFP16 ? DType::kFloat16 : DType::kBFloat16;
  if (args.a.dtype != want || args.b.dtype != want) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "gemm: the SM90a 16-bit tile takes bf16 x bf16 (kBF16) or fp16 x fp16 (kFP16)");
  }
  if (args.d_dtype != want && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: SM90a output is the operand dtype or fp32");
  }
  if (args.a.scale != nullptr || args.b.scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemm: 16-bit operands carry no operand scales");
  }
  int id = -1;
  Status st = sm90_check_args(args, t, 2, "gemm", &id);
  if (st != Status::kSuccess) return st;
  st = sm90_check_epilogue(args.epi, args.d_dtype, "gemm");
  if (st != Status::kSuccess) return st;
  const Sm90Tile& tile = kSm90DenseTiles[id];
  const int d_bytes = args.d_dtype == DType::kFloat32 ? 4 : 2;
  const size_t smem = sm90a::dense_smem_bytes(tile.tm, tile.tn, tile.stages, d_bytes);
  if (ctx.arch_id != KOPS_TARGET_SM90A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "gemm: the mainloop %d is the SM90a cell and needs KOPS_TARGET_SM90A, found arch %d",
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  if (smem > static_cast<size_t>(smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: the SM90a tile needs %zu B of shared memory", smem);
  }

  GemmTmaParams p{};
  p.args = args;
  p.args.split_k_mode = SplitKMode::kNone;
  p.args.split_k_slices = 1;
  p.num_tiles_m = static_cast<int32_t>((args.m + tile.tm - 1) / tile.tm);
  p.num_tiles_n = static_cast<int32_t>((args.n + tile.tn - 1) / tile.tn);
  p.num_tiles_k = static_cast<int32_t>((args.k + sm90a::kTileK16 - 1) / sm90a::kTileK16);

  const CUtensorMapDataType tm_dt = want == DType::kFloat16 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                                            : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  // A [M, K] and B [N, K]: boxes of 64 K elements (128 bytes, 128B swizzle) x tile rows.
  st = sm90a::encode_tma_2d(&p.tma_a, tm_dt, args.a.data, static_cast<uint64_t>(args.k), static_cast<uint64_t>(args.m),
                            static_cast<uint64_t>(args.a.ld) * 2u, sm90a::kTileK16, static_cast<uint32_t>(tile.tm),
                            CU_TENSOR_MAP_SWIZZLE_128B, "gemm (A)");
  if (st != Status::kSuccess) return st;
  st = sm90a::encode_tma_2d(&p.tma_b, tm_dt, args.b.data, static_cast<uint64_t>(args.k), static_cast<uint64_t>(args.n),
                            static_cast<uint64_t>(args.b.ld) * 2u, sm90a::kTileK16, static_cast<uint32_t>(tile.tn),
                            CU_TENSOR_MAP_SWIZZLE_128B, "gemm (B)");
  if (st != Status::kSuccess) return st;
  // D [M, N]: only the plain 16-bit epilogue stores through TMA (64-column boxes); the kernel takes the
  // direct-store path under the same predicate, so the map is simply left zero otherwise.
  const bool d_tma = d_bytes == 2 && args.epi.ops == 0u && aligned(args.d, 16) && (args.ldd * 2) % 16 == 0;
  if (d_tma) {
    st = sm90a::encode_tma_2d(&p.tma_d, tm_dt, args.d, static_cast<uint64_t>(args.n), static_cast<uint64_t>(args.m),
                              static_cast<uint64_t>(args.ldd) * 2u, 64u, static_cast<uint32_t>(tile.tm),
                              CU_TENSOR_MAP_SWIZZLE_128B, "gemm (D)");
    if (st != Status::kSuccess) return st;
  }

  const unsigned tiles = static_cast<unsigned>(p.num_tiles_m) * static_cast<unsigned>(p.num_tiles_n);
  unsigned grid = std::min<unsigned>(tiles, static_cast<unsigned>(std::max(1, sm_count(ctx))));
  const int mc = t.cluster_n == 2 ? 1 : (t.cluster_m == 2 ? 2 : 0);
  if (mc != 0) {
    // A cluster launch needs an even grid; one CTA per SM still, so an odd SM count loses one. A CTA
    // without a tile exits after the barrier setup.
    const unsigned sms = static_cast<unsigned>(std::max(2, sm_count(ctx))) & ~1u;
    grid = std::min<unsigned>((tiles + 1u) & ~1u, sms);
  }
  const cudaStream_t s = stream_of(ctx);
  if (want == DType::kBFloat16) return dense_out<__nv_bfloat16>(id, mc, p, grid, s);
  return dense_out<__half>(id, mc, p, grid, s);
}

// ---- FP8 (B11 row 2) ---------------------------------------------------------------------

template <typename Dt, int TM, int TN, int ST, bool kPromoted>
Status launch_fp8_tile(const GemmTmaParams& p, unsigned grid, cudaStream_t s) {
  using Tr = GemmTraits<f8e4m3_t, f8e4m3_t, Dt, float, TM, TN, sm90a::kTileK8, 1, 1, ST,
                        TM <= 64 ? MainloopSchedule::kTmaWarpSpecialized
                                 : MainloopSchedule::kTmaWarpSpecializedCooperative,
                        EpilogueSchedule::kTmaStore, Sm90a>;
  constexpr size_t kSmem = sm90a::dense_smem_bytes(TM, TN, ST, static_cast<int>(sizeof(Dt)));
  constexpr GemmRecipe kR = kPromoted ? GemmRecipe::kFP8Blockwise : GemmRecipe::kFP8Rowwise;
  constexpr Fp8Accum kA = kPromoted ? Fp8Accum::kPromoted : Fp8Accum::kFastAccum;
  static const cudaError_t attr = cudaFuncSetAttribute(fp8_gemm_wgmma_kernel<Tr, kR, kA>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "fp8_gemm_wgmma_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  fp8_gemm_wgmma_kernel<Tr, kR, kA><<<grid, static_cast<unsigned>(sm90a::Threads<Tr>::value), kSmem, s>>>(p);
  return launch_error("fp8_gemm_wgmma_kernel");
}

/// Explicit instantiation table. The promoted body keeps one 128-column B scale block per tile, so
/// it exists for the 128-wide tiles only.
template <typename Dt>
Status fp8_tile_id(int id, bool promoted, const GemmTmaParams& p, unsigned grid, cudaStream_t s) {
  if (!promoted) {
    switch (id) {
      case 0: return launch_fp8_tile<Dt, 64, 128, 8, false>(p, grid, s);
      case 1: return launch_fp8_tile<Dt, 64, 256, 4, false>(p, grid, s);
      case 2: return launch_fp8_tile<Dt, 128, 128, 6, false>(p, grid, s);
      case 3: return launch_fp8_tile<Dt, 128, 256, 3, false>(p, grid, s);
      case 4: return launch_fp8_tile<Dt, 256, 128, 3, false>(p, grid, s);
      default: break;
    }
  } else {
    switch (id) {
      case 0: return launch_fp8_tile<Dt, 64, 128, 8, true>(p, grid, s);
      case 2: return launch_fp8_tile<Dt, 128, 128, 6, true>(p, grid, s);
      case 4: return launch_fp8_tile<Dt, 256, 128, 3, true>(p, grid, s);
      default: break;
    }
  }
  return detail::set_error(Status::kInternal, "fp8_gemm: SM90a tile id %d promoted %d", id, static_cast<int>(promoted));
}

Status sm90_fp8_gemm(const ScaledGemmArgs& args_in, const GemmTactic& t, const LaunchContext& ctx) {
  ScaledGemmArgs args = args_in;
  const bool block = args.recipe == GemmRecipe::kFP8Blockwise || args.recipe == GemmRecipe::kFP8BlockwiseUE8M0;
  if (args.a.dtype != DType::kFloat8E4M3 || args.b.dtype != DType::kFloat8E4M3 || args.a.fp8 != Fp8Encoding::kOCP ||
      args.b.fp8 != Fp8Encoding::kOCP) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: the SM90a FP8 tile takes OCP e4m3 x e4m3");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: SM90a output is bf16, fp16 or fp32");
  }
  const bool promoted = args.fp8_accum == Fp8Accum::kPromoted;
  if (block) {
    // DeepSeek recipe: 1x128 activation scales x 128x128 weight scales, applied to the fp32 partial sum
    // of every 128 K. The validation mirrors the B7 mma.sync body (gemm_mma.cu) so the two bodies take
    // the same arguments.
    if (!promoted) {
      return detail::set_error(Status::kInvalidArgument,
                               "fp8_gemm: the blockwise recipes apply their scales at the fp32 promotion (Fp8Accum::kPromoted)");
    }
    if (args.fp8_promotion_k != 128) {
      return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: SM90a promotes every 128 K (fp8_promotion_k = %d)",
                               args.fp8_promotion_k);
    }
    if (args.k % 128 != 0) return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: FP8 blockwise needs K %% 128 == 0");
    if (args.a.scale == nullptr || args.b.scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "fp8_gemm: FP8 blockwise needs 1x128 A scales and 128x128 B scales");
    }
    const bool ue = args.recipe == GemmRecipe::kFP8BlockwiseUE8M0;
    const DType want = ue ? DType::kFloat8E8M0 : DType::kFloat32;
    if ((args.a.scale_dtype != want && !(args.a.scale_dtype == DType::kFloat8E8M0 && !ue)) ||
        (args.b.scale_dtype != want && !(args.b.scale_dtype == DType::kFloat8E8M0 && !ue))) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "fp8_gemm: FP8 block scales are fp32 (kFP8Blockwise; UE8M0 bytes also accepted) or UE8M0 "
                               "bytes (kFP8BlockwiseUE8M0)");
    }
    if ((args.a.scale_layout != ScaleLayout::kRowMajor && args.a.scale_layout != ScaleLayout::kColMajorTma) ||
        args.b.scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "fp8_gemm: A scales row-major or TMA column-major, B scales row-major (packed UE8M0 is not "
                               "implemented)");
    }
    if (args.a.scale_ld == 0) {
      args.a.scale_ld = args.a.scale_layout == ScaleLayout::kColMajorTma ? ((args.m + 3) / 4) * 4 : args.k / 128;
    }
    if (args.b.scale_ld == 0) args.b.scale_ld = args.k / 128;
  } else {
    // Tensorwise / rowwise: per tensor folds into alpha, per token (A) and per channel (B) stay vectors.
    for (const ScaledOperand* op : {&args.a, &args.b}) {
      if (op->scale == nullptr) continue;
      if (op->scale_dtype != DType::kFloat32) {
        return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: SM90a tensorwise / rowwise scales are fp32");
      }
      const bool per_tensor = op->granularity == ScaleGranularity::kPerTensor;
      const bool vec = op == &args.a ? op->granularity == ScaleGranularity::kPerToken
                                     : op->granularity == ScaleGranularity::kPerChannel;
      if (!per_tensor && !vec) {
        return detail::set_error(Status::kUnsupportedLayout,
                                 "fp8_gemm: operand scales are fp32, per tensor or per token (A) / per channel (B)");
      }
      if (op->scale_layout != ScaleLayout::kRowMajor) {
        return detail::set_error(Status::kUnsupportedLayout, "fp8_gemm: SM90a operand scales are the contiguous row vector");
      }
    }
  }
  int id = -1;
  Status st = sm90_check_args(args, t, 1, "fp8_gemm", &id);
  if (st != Status::kSuccess) return st;
  if (t.cluster_m > 1 || t.cluster_n > 1) {
    return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: the SM90a FP8 tiles are compiled with cluster 1x1");
  }
  const Sm90Tile& tile = kSm90DenseTiles[id];
  if (promoted && tile.tn > 128) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fp8_gemm: Fp8Accum::kPromoted is compiled for the 128-wide tiles (64x128, 128x128, 256x128), "
                             "not %dx%d", tile.tm, tile.tn);
  }
  st = sm90_check_epilogue(args.epi, args.d_dtype, "fp8_gemm");
  if (st != Status::kSuccess) return st;
  const int d_bytes = args.d_dtype == DType::kFloat32 ? 4 : 2;
  const size_t smem = sm90a::dense_smem_bytes(tile.tm, tile.tn, tile.stages, d_bytes);
  if (ctx.arch_id != KOPS_TARGET_SM90A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "fp8_gemm: the mainloop %d is the SM90a cell and needs KOPS_TARGET_SM90A, found arch %d",
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  if (smem > static_cast<size_t>(smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: the SM90a tile needs %zu B of shared memory", smem);
  }

  GemmTmaParams p{};
  p.args = args;
  p.args.split_k_mode = SplitKMode::kNone;
  p.args.split_k_slices = 1;
  p.num_tiles_m = static_cast<int32_t>((args.m + tile.tm - 1) / tile.tm);
  p.num_tiles_n = static_cast<int32_t>((args.n + tile.tn - 1) / tile.tn);
  p.num_tiles_k = static_cast<int32_t>((args.k + sm90a::kTileK8 - 1) / sm90a::kTileK8);
  // A [M, K] and B [N, K] e4m3: boxes of 128 K elements (128 bytes, 128B swizzle) x tile rows. The zero
  // fill of the K tail and of the rows past M / N is the e4m3 code of 0.0.
  st = sm90a::encode_tma_2d(&p.tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, args.a.data, static_cast<uint64_t>(args.k),
                            static_cast<uint64_t>(args.m), static_cast<uint64_t>(args.a.ld), sm90a::kTileK8,
                            static_cast<uint32_t>(tile.tm), CU_TENSOR_MAP_SWIZZLE_128B, "fp8_gemm (A)");
  if (st != Status::kSuccess) return st;
  st = sm90a::encode_tma_2d(&p.tma_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, args.b.data, static_cast<uint64_t>(args.k),
                            static_cast<uint64_t>(args.n), static_cast<uint64_t>(args.b.ld), sm90a::kTileK8,
                            static_cast<uint32_t>(tile.tn), CU_TENSOR_MAP_SWIZZLE_128B, "fp8_gemm (B)");
  if (st != Status::kSuccess) return st;
  // D: the kernel stores through TMA under exactly this predicate (and only when no operand scale
  // vector sends it to the element-wise epilogue).
  const bool d_tma = d_bytes == 2 && args.epi.ops == 0u && aligned(args.d, 16) && (args.ldd * 2) % 16 == 0;
  if (d_tma) {
    st = sm90a::encode_tma_2d(&p.tma_d,
                              args.d_dtype == DType::kFloat16 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                                              : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                              args.d, static_cast<uint64_t>(args.n), static_cast<uint64_t>(args.m),
                              static_cast<uint64_t>(args.ldd) * 2u, 64u, static_cast<uint32_t>(tile.tm),
                              CU_TENSOR_MAP_SWIZZLE_128B, "fp8_gemm (D)");
    if (st != Status::kSuccess) return st;
  }
  const unsigned tiles = static_cast<unsigned>(p.num_tiles_m) * static_cast<unsigned>(p.num_tiles_n);
  const unsigned grid = std::min<unsigned>(tiles, static_cast<unsigned>(std::max(1, sm_count(ctx))));
  const cudaStream_t s = stream_of(ctx);
  switch (args.d_dtype) {
    case DType::kFloat32: return fp8_tile_id<float>(id, promoted, p, grid, s);
    case DType::kFloat16: return fp8_tile_id<__half>(id, promoted, p, grid, s);
    default: return fp8_tile_id<__nv_bfloat16>(id, promoted, p, grid, s);
  }
}

// ---- grouped (B11 row 7) -------------------------------------------------------------------

template <typename At, typename Dt, int TM, int TN, int ST, bool kPromoted>
Status launch_grouped_tile(const GroupedTmaParams& p, unsigned grid, cudaStream_t s) {
  using Tr = GemmTraits<At, At, Dt, float, TM, TN, 128 / static_cast<int>(sizeof(At)), 1, 1, ST,
                        TM <= 64 ? MainloopSchedule::kTmaWarpSpecialized
                                 : MainloopSchedule::kTmaWarpSpecializedCooperative,
                        EpilogueSchedule::kTmaStore, Sm90a>;
  constexpr size_t kSmem = sm90a::dense_smem_bytes(TM, TN, ST, static_cast<int>(sizeof(Dt)));
  constexpr GemmRecipe kR = kPromoted ? GemmRecipe::kFP8Blockwise : GemmRecipe::kBF16;
  constexpr Fp8Accum kA = kPromoted ? Fp8Accum::kPromoted : Fp8Accum::kFastAccum;
  static const cudaError_t attr = cudaFuncSetAttribute(grouped_gemm_wgmma_kernel<Tr, kR, GroupedLayout::kContiguous, kA>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "grouped_gemm_wgmma_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  grouped_gemm_wgmma_kernel<Tr, kR, GroupedLayout::kContiguous, kA>
      <<<grid, static_cast<unsigned>(sm90a::Threads<Tr>::value), kSmem, s>>>(p);
  return launch_error("grouped_gemm_wgmma_kernel");
}

/// Instantiation table of the grouped cell: 16-bit 64x128 / 128x128 / 128x256, FP8 blockwise 64x128 / 128x128.
template <typename At>
Status grouped16_tile_id(int id, const GroupedTmaParams& p, unsigned grid, cudaStream_t s) {
  switch (id) {
    case 0: return launch_grouped_tile<At, At, 64, 128, 8, false>(p, grid, s);
    case 2: return launch_grouped_tile<At, At, 128, 128, 6, false>(p, grid, s);
    case 3: return launch_grouped_tile<At, At, 128, 256, 3, false>(p, grid, s);
    default: return detail::set_error(Status::kInternal, "grouped_gemm: SM90a tile id %d", id);
  }
}

Status grouped_fp8_tile_id(int id, const GroupedTmaParams& p, unsigned grid, cudaStream_t s) {
  switch (id) {
    case 0: return launch_grouped_tile<f8e4m3_t, __nv_bfloat16, 64, 128, 8, true>(p, grid, s);
    case 2: return launch_grouped_tile<f8e4m3_t, __nv_bfloat16, 128, 128, 6, true>(p, grid, s);
    default: return detail::set_error(Status::kInternal, "grouped_gemm: SM90a FP8 tile id %d", id);
  }
}

}  // namespace

Status wgmma_grouped_gemm(const GroupedGemmParams& p, const GemmTactic& t, const Workspace& ws, const LaunchContext& ctx) {
  (void)ws;   // the tile sequence is derived in-kernel from the device row counts: no workspace
  const char* who = "grouped_gemm";
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", who);
  const bool block = p.recipe == GemmRecipe::kFP8Blockwise || p.recipe == GemmRecipe::kFP8BlockwiseUE8M0;
  const bool f16 = p.recipe == GemmRecipe::kFP16;
  if (!block && !f16 && p.recipe != GemmRecipe::kBF16) {
    return detail::set_error(Status::kNotImplemented,
                             "%s: the SM90a grouped cell has kBF16, kFP16, kFP8Blockwise and kFP8BlockwiseUE8M0 (recipe %d runs "
                             "on the mma.sync body)", who, static_cast<int>(p.recipe));
  }
  if (p.layout != GroupedLayout::kContiguous && p.layout != GroupedLayout::kMasked) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM90a grouped cell has the kContiguous and kMasked layouts", who);
  }
  if (p.num_groups < 1 || p.n < 1 || p.k < 1) return detail::set_error(Status::kInvalidArgument, "%s: empty problem", who);
  if (p.a == nullptr || p.b == nullptr || p.d == nullptr) return detail::set_error(Status::kInvalidArgument, "%s: null operand", who);
  const bool masked = p.layout == GroupedLayout::kMasked;
  if (masked ? (p.masked_m == nullptr || p.max_m < 1) : p.expert_offsets == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: row metadata missing for the selected layout", who);
  }
  if (!masked && p.max_m < 1) {
    // The tensor maps of A and D need the extent of the routed row space; the counts themselves stay on the device.
    return detail::set_error(Status::kInvalidArgument,
                             "%s: SM90a kContiguous needs max_m = rows of A / D (the capacity expert_offsets[G] never exceeds)", who);
  }
  if (p.b_packing != WeightPacking::kNone || p.b_zeros != nullptr || p.b_global_scale != nullptr ||
      p.a_global_scale != nullptr || p.swap_ab || p.a_ptrs != nullptr || p.b_ptrs != nullptr || p.d_ptrs != nullptr) {
    return detail::set_error(Status::kNotImplemented, "%s: packed weights, zeros, global scales, swap_ab and pointer arrays have no SM90a body", who);
  }
  if (p.epi.ops != 0u) {
    return detail::set_error(Status::kNotImplemented, "%s: epilogue ops 0x%x have no SM90a grouped body (the mma.sync body serves them)",
                             who, p.epi.ops);
  }
  if (p.lda < p.k || p.ldb < p.k || p.ldd < p.n) {
    return detail::set_error(Status::kInvalidArgument, "%s: leading dimensions smaller than the extents", who);
  }
  if (p.b_group_stride != static_cast<int64_t>(p.n) * p.ldb) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: SM90a reads the experts through one tensor map: b_group_stride must be N * ldb", who);
  }
  const int eb = block ? 1 : 2;
  if (!aligned(p.a, 16) || !aligned(p.b, 16) || (p.lda * eb) % 16 != 0 || (p.ldb * eb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "%s: SM90a operands are read by TMA and need a 16 B aligned pointer and row pitch", who);
  }
  const int64_t rows = masked ? static_cast<int64_t>(p.max_m) * p.num_groups : static_cast<int64_t>(p.max_m);
  if (rows > 0x7FFFFFFF || static_cast<int64_t>(p.num_groups) * p.n > 0x7FFFFFFF) {
    return detail::set_error(Status::kUnsupportedShape, "%s: extents beyond int32", who);
  }
  GroupedTmaParams q{};
  ScaledGemmArgs& a = q.args;
  a.recipe = p.recipe;
  const DType act = block ? DType::kFloat8E4M3 : (f16 ? DType::kFloat16 : DType::kBFloat16);
  a.a.data = p.a; a.a.ld = p.lda; a.a.dtype = act;
  a.b.data = p.b; a.b.ld = p.ldb; a.b.dtype = act; a.b.batch_stride = p.b_group_stride;
  a.d = p.d; a.ldd = p.ldd;
  a.d_dtype = block ? DType::kBFloat16 : act;
  a.epi = p.epi;
  a.m = rows; a.n = p.n; a.k = p.k;
  a.fp8_accum = Fp8Accum::kPromoted;
  if (block) {
    if (p.k % 128 != 0) return detail::set_error(Status::kUnsupportedShape, "%s: FP8 blockwise needs K %% 128 == 0", who);
    if (p.a_scale == nullptr || p.b_scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "%s: FP8 blockwise needs 1x128 A scales and per-expert 128x128 B scales", who);
    }
    if (p.a_scale_layout != ScaleLayout::kRowMajor && p.a_scale_layout != ScaleLayout::kColMajorTma) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: A scales row-major or TMA column-major", who);
    }
    const DType sdt = p.recipe == GemmRecipe::kFP8BlockwiseUE8M0 ? DType::kFloat8E8M0 : DType::kFloat32;
    a.a.scale = p.a_scale; a.a.scale_dtype = sdt; a.a.scale_layout = p.a_scale_layout;
    a.a.granularity = ScaleGranularity::kPerGroup;
    a.a.scale_ld = p.a_scale_layout == ScaleLayout::kColMajorTma ? (rows + 3) / 4 * 4 : p.k / 128;
    a.b.scale = p.b_scale; a.b.scale_dtype = sdt; a.b.granularity = ScaleGranularity::kPerBlock2D;
    a.b.scale_ld = p.k / 128;
    a.b.scale_batch_stride = p.b_scale_group_stride != 0 ? p.b_scale_group_stride
                                                         : ((static_cast<int64_t>(p.n) + 127) / 128) * (p.k / 128);
  } else if (p.a_scale != nullptr || p.b_scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: 16-bit operands carry no operand scales", who);
  }
  if (t.split_k_mode != SplitKMode::kNone || t.split_k_slices > 1 || t.swap_ab) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM90a grouped cell has no split-K / swap_ab", who);
  }
  if (t.cluster_m > 1 || t.cluster_n > 1 || t.tma_multicast) {
    return detail::set_error(Status::kUnsupportedShape, "%s: the SM90a grouped tiles are compiled with cluster 1x1", who);
  }
  int id = -1;
  Status st = sm90_tile_of(t, who, &id);
  if (st != Status::kSuccess) return st;
  if (block ? (id != 0 && id != 2) : (id != 0 && id != 2 && id != 3)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: SM90a grouped tiles are 64x128, 128x128%s (found %dx%d)", who,
                             block ? " (FP8 blockwise)" : " and 128x256", t.tile_m, t.tile_n);
  }
  const Sm90Tile& tile = kSm90DenseTiles[id];
  const size_t smem = sm90a::dense_smem_bytes(tile.tm, tile.tn, tile.stages, 2);
  if (ctx.arch_id != KOPS_TARGET_SM90A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "%s: the mainloop %d is the SM90a cell and needs KOPS_TARGET_SM90A, found arch %d", who,
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  if (smem > static_cast<size_t>(smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "%s: the SM90a tile needs %zu B of shared memory", who, smem);
  }
  q.num_tiles_n = static_cast<int32_t>((p.n + tile.tn - 1) / tile.tn);
  const int bk = 128 / eb;
  q.num_tiles_k = static_cast<int32_t>((p.k + bk - 1) / bk);
  q.num_groups = p.num_groups;
  q.max_rows = masked ? p.max_m : 0;
  q.group_rows = masked ? nullptr : p.expert_offsets;
  q.masked_rows = masked ? p.masked_m : nullptr;
  const CUtensorMapDataType in_dt = block ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                                          : (f16 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);
  st = sm90a::encode_tma_2d(&q.tma_a, in_dt, p.a, static_cast<uint64_t>(p.k), static_cast<uint64_t>(rows),
                            static_cast<uint64_t>(p.lda) * eb, static_cast<uint32_t>(bk), static_cast<uint32_t>(tile.tm),
                            CU_TENSOR_MAP_SWIZZLE_128B, "grouped_gemm (A)");
  if (st != Status::kSuccess) return st;
  st = sm90a::encode_tma_2d(&q.tma_b, in_dt, p.b, static_cast<uint64_t>(p.k),
                            static_cast<uint64_t>(p.num_groups) * static_cast<uint64_t>(p.n),
                            static_cast<uint64_t>(p.ldb) * eb, static_cast<uint32_t>(bk), static_cast<uint32_t>(tile.tn),
                            CU_TENSOR_MAP_SWIZZLE_128B, "grouped_gemm (B)");
  if (st != Status::kSuccess) return st;
  if (aligned(p.d, 16) && (p.ldd * 2) % 16 == 0) {
    st = sm90a::encode_tma_2d(&q.tma_d, f16 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.d,
                              static_cast<uint64_t>(p.n), static_cast<uint64_t>(rows), static_cast<uint64_t>(p.ldd) * 2u, 64u,
                              static_cast<uint32_t>(tile.tm), CU_TENSOR_MAP_SWIZZLE_128B, "grouped_gemm (D)");
    if (st != Status::kSuccess) return st;
  }
  // The tile count lives on the device; this is its upper bound (every group may end with a partial tile).
  const int64_t bound = (rows / tile.tm + p.num_groups) * static_cast<int64_t>(q.num_tiles_n);
  const unsigned grid = static_cast<unsigned>(std::min<int64_t>(bound, std::max(1, sm_count(ctx))));
  const cudaStream_t s = stream_of(ctx);
  if (block) return grouped_fp8_tile_id(id, q, grid, s);
  return f16 ? grouped16_tile_id<__half>(id, q, grid, s) : grouped16_tile_id<__nv_bfloat16>(id, q, grid, s);
}

Status wgmma_gemm(const ScaledGemmArgs& args, const GemmTactic& tactic, const Workspace& ws, const LaunchContext& ctx) {
  (void)ws;   // one CTA per SM, no split-K: the cell needs no workspace
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "gemm: CUDA backend only");
  }
  switch (args.recipe) {
    case GemmRecipe::kBF16:
    case GemmRecipe::kFP16:
      return sm90_dense_gemm(args, tactic, ctx);
    case GemmRecipe::kFP8Tensorwise:
    case GemmRecipe::kFP8Rowwise:
    case GemmRecipe::kFP8Blockwise:
    case GemmRecipe::kFP8BlockwiseUE8M0:
      return sm90_fp8_gemm(args, tactic, ctx);
    default:
      return detail::set_error(Status::kNotImplemented, "gemm: recipe %d has no SM90a (wgmma) body yet",
                               static_cast<int>(args.recipe));
  }
}

}  // namespace kops::gemm
