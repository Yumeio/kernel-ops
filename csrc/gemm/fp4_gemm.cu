// =============================================================================
// kernel-ops | csrc/gemm/fp4_gemm.cu
// B5 row 7: FP4-weight GEMV.
//   fp4_gemv_kernel   MXFP4 (UE8M0 / 32) and NVFP4 (E4M3 / 16 + fp32 global) weights x bf16/fp16
//                     activations: CUDA cores, M <= 8 (ROWS_PER_WARP > 0);
//                     WeightPacking::kMarlin = fp4_to_marlin tiles on mma.sync m16n8k16, SM80+,
//                     M <= 16, N % 64 == 0 (ROWS_PER_WARP == 0)
// Host entry: fp4_gemm (kW4A16MXFP4 / kW4A16NVFP4); the W4A4 / W4A8 recipes need FP4 activations
// or tensor cores (B11 / B13) and return kNotImplemented.
//
// Contract: csrc/gemm/fp4_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// Decode.  e2m1 is decoded without a table: the three magnitude bits (E1 E0 M) placed under the fp32
// exponent give value * 2^-126 for normals and the single subnormal alike; the restoring multiply by
// 2^126 is exact.  Sm100a / sm120a could use cvt.rn.f16x2.e2m1x2; this body is the arch-generic one
// and gives the same float for all 16 codes (tests/unit/test_gemm.cc).
//
// Scales.  b.scale is [N, K/16] (E4M3) or [N, K/32] (UE8M0), row-major or the 128x4 swizzle.  The
// partial sum of one block is scaled once and added to the fp32 accumulator.  The NVFP4 global
// scale follows the repo convention (quant side: the activation/weight were multiplied by it), so
// the dequant multiplier applied to the result is 1 / *b.global_scale.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

namespace kops::gemm {
namespace {

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

__device__ __forceinline__ float ue8m0_to_f(uint32_t e) {
  return e == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(e << 23);
}

/// Offset of scale (row, blk) in the 128x4 swizzled layout (zero padded to 128 rows x 4 columns).
__device__ __forceinline__ int64_t swizzled_offset(int64_t row, int64_t blk, int64_t blocks_per_row) {
  const int64_t tile = (row >> 7) * ((blocks_per_row + 3) >> 2) + (blk >> 2);
  return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (blk & 3);
}

__device__ __forceinline__ void store_result(const ScaledGemmArgs& a, int64_t m, int64_t n, float acc) {
  const int64_t idx = m * a.ldd + n;
  switch (a.d_dtype) {
    case DType::kFloat32:
      static_cast<float*>(a.d)[idx] = dev::gemm_epilogue<float>(acc, a.epi, m, n);
      break;
    case DType::kFloat16:
      static_cast<__half*>(a.d)[idx] = __float2half_rn(dev::gemm_epilogue<__half>(acc, a.epi, m, n));
      break;
    default:
      static_cast<__nv_bfloat16*>(a.d)[idx] = __float2bfloat16_rn(dev::gemm_epilogue<__nv_bfloat16>(acc, a.epi, m, n));
      break;
  }
}

// -----------------------------------------------------------------------------
// Marlin tiles on tensor cores (SM80+): fp4_to_marlin output, weight = MMA operand A.
// Geometry, fragment mapping and numerics: common/device_impl.cuh (dev::tc_gemv_body).  The weight
// fragment is the bare e2m1 code (exact in bf16 and fp16); the block scale (E4M3 per 16 / UE8M0 per 32,
// any exponent) is applied in fp32 to the block's accumulator, so nothing depends on the fp16 range.
// -----------------------------------------------------------------------------

template <typename a_t, bool kNV>
struct Fp4TilePolicy {
  using Tc = dev::TcFrag<a_t>;
  static constexpr int kBlockK = kNV ? 16 : 32;

  const uint8_t* scales;   // [K/blk, N] bytes, Marlin-permuted: lane r reads its columns {r + 8 j} as 8 bytes
  int n;

  struct State {
    uint2 s;               // scale bytes j = 2 w + h: row r (h = 0) or r + 8 (h = 1) of sub-tile w
  };

  __device__ __forceinline__ State load(int blk, int nt, int r) const {
    State st;
    st.s = *reinterpret_cast<const uint2*>(scales + static_cast<int64_t>(blk) * n + nt * dev::kTcTileN + 8 * r);
    return st;
  }

  __device__ __forceinline__ void unpack_scales(const State& st, float (&sf)[8]) const {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t byte = ((j < 4 ? st.s.x : st.s.y) >> (8 * (j & 3))) & 0xFFu;
      sf[j] = kNV ? dev::e4m3_decode(byte) : ue8m0_to_f(byte);
    }
  }

  __device__ __forceinline__ void dequant(const State&, int, uint32_t q, uint32_t (&fa)[4]) const {
    fa[0] = Tc::e2m1x2(q & 0x000F000Fu);           // row r,     k 2c, 2c+1
    fa[2] = Tc::e2m1x2((q >> 4) & 0x000F000Fu);    // row r,     k 2c+8, 2c+9
    fa[1] = Tc::e2m1x2((q >> 8) & 0x000F000Fu);    // row r + 8, k 2c, 2c+1
    fa[3] = Tc::e2m1x2((q >> 12) & 0x000F000Fu);   // row r + 8, k 2c+8, 2c+9
  }
};

/// 1 / global scale, then the repo epilogue, then the round to the output dtype (== activation dtype).
template <typename a_t>
struct Fp4TcStore {
  const ScaledGemmArgs* args;
  float inv_g;
  __device__ __forceinline__ void operator()(float v, int tok, int col) const {
    const ScaledGemmArgs& a = *args;
    dev::from_f(static_cast<a_t*>(a.d)[static_cast<int64_t>(tok) * a.ldd + col],
                dev::gemm_epilogue<a_t>(v * inv_g, a.epi, tok, col));
  }
};

template <typename a_t, GemmRecipe kRecipe, int M_TILE>
__device__ __forceinline__ void fp4_tc_body(const ScaledGemmArgs& args) {
  constexpr bool kNV = kRecipe == GemmRecipe::kW4A16NVFP4 || kRecipe == GemmRecipe::kNVFP4;
  const Fp4TilePolicy<a_t, kNV> pol{static_cast<const uint8_t*>(args.b.scale), static_cast<int>(args.n)};
  const Fp4TcStore<a_t> epi{&args, args.b.global_scale != nullptr ? 1.0f / *args.b.global_scale : 1.0f};
  dev::tc_gemv_body<Fp4TilePolicy<a_t, kNV>, Fp4TcStore<a_t>, a_t, M_TILE>(
      static_cast<const a_t*>(args.a.data), args.a.ld, static_cast<const uint32_t*>(args.b.data), pol, epi,
      static_cast<int>(args.m), static_cast<int>(args.n), static_cast<int>(args.k));
}

}  // namespace

template <typename a_t, GemmRecipe kRecipe, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void fp4_gemv_kernel(const __grid_constant__ ScaledGemmArgs args) {
  if constexpr (ROWS_PER_WARP == 0) {
    // Marlin tiles on mma.sync (M_TILE 8 or 16, WARPS == dev::kTcWarps); the CUDA-core body below is
    // dead code for this instantiation and the compiler drops it.
    fp4_tc_body<a_t, kRecipe, M_TILE>(args);
    return;
  }
  constexpr bool kNV = kRecipe == GemmRecipe::kW4A16NVFP4 || kRecipe == GemmRecipe::kNVFP4;
  constexpr int BLK = kNV ? 16 : 32;
  constexpr int R = ROWS_PER_WARP > 0 ? ROWS_PER_WARP : 1;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;
  if (n0 >= args.n) return;
  const int k = static_cast<int>(args.k);
  const int64_t m_total = args.m;
  const int64_t blocks_per_row = k / BLK;
  const bool swizzled = args.b.scale_layout == ScaleLayout::kSwizzled128x4;
  const int64_t sld = args.b.scale_ld != 0 ? args.b.scale_ld : blocks_per_row;

  const a_t* arow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    arow[m] = static_cast<const a_t*>(args.a.data) + imin(m, m_total - 1) * args.a.ld;
  }
  const uint8_t* brow[R];
  int64_t nrow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    nrow[r] = imin(n0 + r, args.n - 1);
    brow[r] = static_cast<const uint8_t*>(args.b.data) + nrow[r] * args.b.ld;
  }
  const uint8_t* sc = static_cast<const uint8_t*>(args.b.scale);

  float acc[M_TILE][R];
  float part[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) { acc[m][r] = 0.0f; part[m][r] = 0.0f; }
  }

  for (int kc = lane * 32; kc < k; kc += 1024) {
    uint4 wv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = dev::ld_stream16(brow[r] + (kc >> 1));
    float s[R][2];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      const int64_t b0 = kc / BLK;
      const int64_t o0 = swizzled ? swizzled_offset(nrow[r], b0, blocks_per_row) : nrow[r] * sld + b0;
      if constexpr (kNV) {
        const int64_t o1 = swizzled ? swizzled_offset(nrow[r], b0 + 1, blocks_per_row) : nrow[r] * sld + b0 + 1;
        s[r][0] = dev::e4m3_decode(sc[o0]);
        s[r][1] = dev::e4m3_decode(sc[o1]);
      } else {
        s[r][0] = ue8m0_to_f(sc[o0]);
        s[r][1] = 0.0f;
      }
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      float af[M_TILE][8];
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[m] + kc + 8 * q), af[m]);
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t w = q == 0 ? wv[r].x : (q == 1 ? wv[r].y : (q == 2 ? wv[r].z : wv[r].w));
        float wf[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) wf[i] = dev::e2m1_decode(w >> (4 * i));
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) part[m][r] = fmaf(af[m][i], wf[i], part[m][r]);
        }
      }
      if (kNV ? (q & 1) == 1 : q == 3) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          #pragma unroll
          for (int r = 0; r < R; ++r) {
            acc[m][r] = fmaf(s[r][kNV ? q >> 1 : 0], part[m][r], acc[m][r]);
            part[m][r] = 0.0f;
          }
        }
      }
    }
  }

  const float inv_g = args.b.global_scale != nullptr ? 1.0f / *args.b.global_scale : 1.0f;
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == m * R + r && m < m_total && n0 + r < args.n) store_result(args, m, n0 + r, v * inv_g);
    }
  }
}

namespace {

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

template <typename a_t, GemmRecipe kR, int MT, int R>
Status launch_fp4_tile(const ScaledGemmArgs& args, cudaStream_t s) {
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((args.n + kWarps * R - 1) / (kWarps * R)));
  fp4_gemv_kernel<a_t, kR, MT, R, kWarps><<<grid, kWarps * 32, 0, s>>>(args);
  return launch_error("fp4_gemv_kernel");
}

/// Marlin tiles: one CTA of dev::kTcWarps warps per 64-column n-tile, M <= 16 in one launch.
template <typename a_t, GemmRecipe kR>
Status launch_fp4_tc(const ScaledGemmArgs& args, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(args.n / dev::kTcTileN));
  if (args.m <= 8) {
    fp4_gemv_kernel<a_t, kR, 8, 0, dev::kTcWarps><<<grid, dev::kTcWarps * 32, 0, s>>>(args);
  } else {
    fp4_gemv_kernel<a_t, kR, 16, 0, dev::kTcWarps><<<grid, dev::kTcWarps * 32, 0, s>>>(args);
  }
  return launch_error("fp4_gemv_kernel");
}

template <typename a_t, GemmRecipe kR>
Status launch_fp4(const ScaledGemmArgs& args, cudaStream_t s) {
  if (args.b.packing == WeightPacking::kMarlin) return launch_fp4_tc<a_t, kR>(args, s);
  if (args.m <= 1) return launch_fp4_tile<a_t, kR, 1, 4>(args, s);
  if (args.m <= 2) return launch_fp4_tile<a_t, kR, 2, 4>(args, s);
  if (args.m <= 4) return launch_fp4_tile<a_t, kR, 4, 2>(args, s);
  return launch_fp4_tile<a_t, kR, 8, 2>(args, s);
}

constexpr uint32_t kFp4EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                    kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

}  // namespace

Status fp4_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                const Workspace& ws, const LaunchContext& ctx) {
  (void)tactic;
  (void)ws;
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "fp4_gemm: CUDA backend only");
  }
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "fp4_gemm: only the native provider exists");
  }
  const bool nv = args.recipe == GemmRecipe::kW4A16NVFP4;
  if (args.recipe != GemmRecipe::kW4A16MXFP4 && !nv) {
    return detail::set_error(Status::kNotImplemented,
                             "fp4_gemm: recipe %d needs FP4 activations or tensor cores (B11 / B13)", static_cast<int>(args.recipe));
  }
  const DType adt = args.a.dtype;
  if (adt != DType::kBFloat16 && adt != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: activations must be bf16 or fp16");
  }
  // WeightPacking::kMarlin = fp4_to_marlin tiles (int32 words) on mma.sync, SM80+, M <= 16.
  const bool marlin = args.b.packing == WeightPacking::kMarlin;
  if (marlin && ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch,
                             "fp4_gemm: the Marlin-tile body is mma.sync bf16/fp16 and needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  if (marlin ? (args.b.dtype != DType::kUInt32 && args.b.dtype != DType::kInt32) : args.b.dtype != DType::kFloat4E2M1x2) {
    return detail::set_error(Status::kUnsupportedDtype, marlin ? "fp4_gemm: Marlin tiles are int32 words" : "fp4_gemm: weights must be packed e2m1");
  }
  if (args.b.scale_dtype != (nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0)) {
    return detail::set_error(Status::kUnsupportedDtype, nv ? "fp4_gemm: NVFP4 scales must be E4M3" : "fp4_gemm: MXFP4 scales must be UE8M0");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: output must be bf16, fp16 or fp32");
  }
  if (marlin && args.d_dtype != adt) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: the Marlin-tile body writes the activation dtype (fp32 output needs the CUDA-core path)");
  }
  const int blk = nv ? 16 : 32;
  if (args.b.group_k != 0 && args.b.group_k != blk) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: block size must be %d", blk);
  }
  if (args.b.major != MajorOrder::kKMajor || args.a.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "fp4_gemm: K-major operands only");
  }
  if (marlin) {
    if (args.b.scale_layout != ScaleLayout::kMarlinPermuted) {
      return detail::set_error(Status::kUnsupportedLayout, "fp4_gemm: Marlin tiles need scale_layout kMarlinPermuted (fp4_to_marlin output)");
    }
  } else if (args.b.scale_layout != ScaleLayout::kRowMajor && args.b.scale_layout != ScaleLayout::kSwizzled128x4) {
    return detail::set_error(Status::kUnsupportedLayout, "fp4_gemm: scales must be row-major or 128x4 swizzled");
  }
  if (args.batch != 1) return detail::set_error(Status::kNotImplemented, "fp4_gemm: batch > 1");
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "fp4_gemm: empty problem");
  if (args.m > (marlin ? 16 : 8)) {
    return detail::set_error(Status::kNotImplemented, marlin ? "fp4_gemm: M > 16 needs the tensor-core path (B7)"
                                                             : "fp4_gemm: M > 8 needs the tensor-core path (kMarlin tiles take M <= 16)");
  }
  if (args.k % 32 != 0) return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: K must be a multiple of 32");
  if (marlin && args.n % 64 != 0) return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: Marlin tiles need N to be a multiple of 64");
  if (args.a.data == nullptr || args.b.data == nullptr || args.b.scale == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: null operand");
  }
  // b.ld is the byte pitch of a packed [N, K/2] row; Marlin tiles have none.
  if (args.a.ld < args.k || (!marlin && args.b.ld < args.k / 2) || args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: leading dimensions smaller than the extents (b.ld is in bytes)");
  }
  if (!aligned(args.a.data, 16) || (args.a.ld * 2) % 16 != 0 || !aligned(args.b.data, 16) || (!marlin && args.b.ld % 16 != 0)) {
    return detail::set_error(Status::kMisaligned, "fp4_gemm: operand base and row pitch must be 16 B aligned");
  }
  if (marlin && !aligned(args.b.scale, 8)) {
    return detail::set_error(Status::kMisaligned, "fp4_gemm: Marlin-tile scales must be 8 B aligned");
  }
  if ((args.epi.ops & ~kFp4EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "fp4_gemm: epilogue ops 0x%x are not implemented", args.epi.ops & ~kFp4EpiAllowed);
  }
  if (((args.epi.ops & kEpiRowScale) != 0u && args.epi.row_scale == nullptr) ||
      ((args.epi.ops & kEpiColScale) != 0u && args.epi.col_scale == nullptr) ||
      ((args.epi.ops & kEpiBias) != 0u && args.epi.bias == nullptr) ||
      ((args.epi.ops & kEpiResidualAdd) != 0u && args.epi.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: epilogue operand missing (residual is read in the output dtype)");
  }
  if ((args.epi.ops & kEpiOutFp32) != 0u && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: kEpiOutFp32 needs an fp32 output");
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  if (adt == DType::kBFloat16) {
    return nv ? launch_fp4<__nv_bfloat16, GemmRecipe::kW4A16NVFP4>(args, s) : launch_fp4<__nv_bfloat16, GemmRecipe::kW4A16MXFP4>(args, s);
  }
  return nv ? launch_fp4<__half, GemmRecipe::kW4A16NVFP4>(args, s) : launch_fp4<__half, GemmRecipe::kW4A16MXFP4>(args, s);
}

}  // namespace kops::gemm
