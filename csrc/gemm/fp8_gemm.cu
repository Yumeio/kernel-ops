// =============================================================================
// kernel-ops | csrc/gemm/fp8_gemm.cu
// B5 row 7: FP8-weight GEMV on CUDA cores.
//   fp8_gemv_kernel   e4m3 weights [N, K] widened in registers, M <= 16
// Host entry: fp8_gemm (kFP8Tensorwise / kFP8Rowwise / kFP8Blockwise, M <= 16).
//
// Contract: csrc/gemm/fp8_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// Decode.  e4m3 -> fp32 is exact.  Sm89+ converts with cvt.rn.f16x2.e4m3x2; everything else uses
// the integer decoder in common/device_impl.cuh (the seven magnitude bits placed under the fp32
// exponent give value * 2^-120, restored by an exact multiply).  Both give the same float for
// every one of the 256 codes (test_fp8_decode in tests/unit/test_gemm.cc; the sm89 handoff kit
// repeats it for the cvt branch).
//
// Scales.  kFP8Tensorwise / kFP8Rowwise: the scales live in the epilogue (alpha, row_scale[m],
// col_scale[n]) so the mainloop stays a plain FMA chain.  kFP8Blockwise: B scale [ceil(N/128),
// K/128] and, for FP8 activations, A scale 1x128 ([M, K/128] row-major or [K/128, round_up(M,4)]
// TMA column-major), fp32 or UE8M0 bytes; each lane's 16 K elements lie in one 128 block, so the
// block partial sum is scaled once and added to the fp32 accumulator.
//
// Determinism.  A given (m, n) is the same lane-strided FMA chain followed by a xor-butterfly
// reduction whatever M_TILE, ROWS_PER_WARP or grid is used (no atomics, no split-K), so one row
// alone and the same row inside a batch are bit-identical.
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

__device__ __forceinline__ float scale_at(const void* p, DType t, int64_t i) {
  if (t == DType::kFloat8E8M0) return __uint_as_float(static_cast<uint32_t>(static_cast<const uint8_t*>(p)[i]) << 23);
  return static_cast<const float*>(p)[i];
}

/// 16 consecutive activations -> floats.
template <typename a_t>
__device__ __forceinline__ void load_a16(const a_t* p, float (&f)[16]) {
  if constexpr (std::is_same<a_t, f8e4m3_t>::value) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    float t[4];
    dev::e4m3x4_decode(v.x, t); f[0] = t[0]; f[1] = t[1]; f[2] = t[2]; f[3] = t[3];
    dev::e4m3x4_decode(v.y, t); f[4] = t[0]; f[5] = t[1]; f[6] = t[2]; f[7] = t[3];
    dev::e4m3x4_decode(v.z, t); f[8] = t[0]; f[9] = t[1]; f[10] = t[2]; f[11] = t[3];
    dev::e4m3x4_decode(v.w, t); f[12] = t[0]; f[13] = t[1]; f[14] = t[2]; f[15] = t[3];
  } else {
    float lo[8], hi[8];
    dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(p), lo);
    dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(p + 8), hi);
    #pragma unroll
    for (int i = 0; i < 8; ++i) { f[i] = lo[i]; f[8 + i] = hi[i]; }
  }
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

}  // namespace

template <typename a_t, GemmRecipe kRecipe, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void fp8_gemv_kernel(const __grid_constant__ ScaledGemmArgs args) {
  constexpr bool kBlock = kRecipe == GemmRecipe::kFP8Blockwise;
  constexpr int R = ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;
  if (n0 >= args.n) return;
  const int k = static_cast<int>(args.k);
  const int64_t m_total = args.m;

  const a_t* arow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    arow[m] = static_cast<const a_t*>(args.a.data) + imin(m, m_total - 1) * args.a.ld;
  }
  const uint8_t* brow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    brow[r] = static_cast<const uint8_t*>(args.b.data) + imin(n0 + r, args.n - 1) * args.b.ld;
  }
  [[maybe_unused]] const bool a_scaled = args.a.scale != nullptr;
  [[maybe_unused]] const int64_t kblocks = k >> 7;
  [[maybe_unused]] const int64_t sb_ld = args.b.scale_ld != 0 ? args.b.scale_ld : kblocks;
  [[maybe_unused]] const bool a_colmajor = args.a.scale_layout == ScaleLayout::kColMajorTma;
  [[maybe_unused]] const int64_t sa_ld = args.a.scale_ld != 0 ? args.a.scale_ld
                                                             : (a_colmajor ? ((m_total + 3) / 4) * 4 : kblocks);

  float acc[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;
  }

  #pragma unroll 2
  for (int kc = lane * 16; kc < k; kc += 512) {
    uint4 wv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = dev::ld_stream16(brow[r] + kc);
    float wf[R][16];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float t[4];
      dev::e4m3x4_decode(wv[r].x, t); wf[r][0] = t[0]; wf[r][1] = t[1]; wf[r][2] = t[2]; wf[r][3] = t[3];
      dev::e4m3x4_decode(wv[r].y, t); wf[r][4] = t[0]; wf[r][5] = t[1]; wf[r][6] = t[2]; wf[r][7] = t[3];
      dev::e4m3x4_decode(wv[r].z, t); wf[r][8] = t[0]; wf[r][9] = t[1]; wf[r][10] = t[2]; wf[r][11] = t[3];
      dev::e4m3x4_decode(wv[r].w, t); wf[r][12] = t[0]; wf[r][13] = t[1]; wf[r][14] = t[2]; wf[r][15] = t[3];
    }
    [[maybe_unused]] float sb[R];
    if constexpr (kBlock) {
      const int kb = kc >> 7;
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const int64_t nrow = imin(n0 + r, args.n - 1);
        sb[r] = scale_at(args.b.scale, args.b.scale_dtype, (nrow >> 7) * sb_ld + kb);
      }
    }
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      float af[16];
      load_a16<a_t>(arow[m] + kc, af);
      [[maybe_unused]] float sa = 1.0f;
      if constexpr (kBlock) {
        if (a_scaled) {
          const int64_t mrow = imin(m, m_total - 1);
          sa = scale_at(args.a.scale, args.a.scale_dtype, a_colmajor ? (kc >> 7) * sa_ld + mrow : mrow * sa_ld + (kc >> 7));
        }
      }
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        if constexpr (kBlock) {
          float part = 0.0f;
          #pragma unroll
          for (int i = 0; i < 16; ++i) part = fmaf(af[i], wf[r][i], part);
          acc[m][r] = fmaf(sa * sb[r], part, acc[m][r]);
        } else {
          #pragma unroll
          for (int i = 0; i < 16; ++i) acc[m][r] = fmaf(af[i], wf[r][i], acc[m][r]);
        }
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == m * R + r && m < m_total && n0 + r < args.n) store_result(args, m, n0 + r, v);
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
Status launch_fp8_tile(const ScaledGemmArgs& args, cudaStream_t s) {
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((args.n + kWarps * R - 1) / (kWarps * R)));
  fp8_gemv_kernel<a_t, kR, MT, R, kWarps><<<grid, kWarps * 32, 0, s>>>(args);
  return launch_error("fp8_gemv_kernel");
}

template <typename a_t, GemmRecipe kR>
Status launch_fp8(const ScaledGemmArgs& args, cudaStream_t s) {
  if (args.m <= 1) return launch_fp8_tile<a_t, kR, 1, 4>(args, s);
  if (args.m <= 2) return launch_fp8_tile<a_t, kR, 2, 4>(args, s);
  if (args.m <= 4) return launch_fp8_tile<a_t, kR, 4, 2>(args, s);
  if (args.m <= 8) return launch_fp8_tile<a_t, kR, 8, 2>(args, s);
  return launch_fp8_tile<a_t, kR, 16, 1>(args, s);
}

template <typename a_t>
Status launch_fp8_recipe(const ScaledGemmArgs& args, cudaStream_t s) {
  switch (args.recipe) {
    case GemmRecipe::kFP8Tensorwise: return launch_fp8<a_t, GemmRecipe::kFP8Tensorwise>(args, s);
    case GemmRecipe::kFP8Rowwise: return launch_fp8<a_t, GemmRecipe::kFP8Rowwise>(args, s);
    default: return launch_fp8<a_t, GemmRecipe::kFP8Blockwise>(args, s);
  }
}

constexpr uint32_t kFp8EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                    kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

}  // namespace

Status fp8_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                const Workspace& ws, const LaunchContext& ctx) {
  (void)tactic;
  (void)ws;
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "fp8_gemm: CUDA backend only");
  }
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "fp8_gemm: only the native provider exists");
  }
  const GemmRecipe rc = args.recipe;
  if (rc != GemmRecipe::kFP8Tensorwise && rc != GemmRecipe::kFP8Rowwise && rc != GemmRecipe::kFP8Blockwise) {
    return detail::set_error(Status::kNotImplemented, "fp8_gemm: recipe %d has no native body yet", static_cast<int>(rc));
  }
  if (args.b.dtype != DType::kFloat8E4M3 || args.b.fp8 != Fp8Encoding::kOCP) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: weights must be OCP e4m3");
  }
  const DType adt = args.a.dtype;
  if (adt != DType::kBFloat16 && adt != DType::kFloat16 && adt != DType::kFloat8E4M3) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: activations must be bf16, fp16 or e4m3");
  }
  if (adt == DType::kFloat8E4M3 && args.a.fp8 != Fp8Encoding::kOCP) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: FNUZ activations are not supported");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: output must be bf16, fp16 or fp32");
  }
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "fp8_gemm: K-major operands only (weight_transpose first)");
  }
  if (args.batch != 1) return detail::set_error(Status::kNotImplemented, "fp8_gemm: batch > 1");
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "fp8_gemm: empty problem");
  if (args.m > 16) return detail::set_error(Status::kNotImplemented, "fp8_gemm: M > 16 needs the tensor-core path (B7)");
  if (args.k % 16 != 0) return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: K must be a multiple of 16");
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "fp8_gemm: null operand");
  }
  const int a_bytes = adt == DType::kFloat8E4M3 ? 1 : 2;
  if (args.a.ld < args.k || args.b.ld < args.k || args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "fp8_gemm: leading dimensions smaller than the extents");
  }
  if (!aligned(args.a.data, 16) || (args.a.ld * a_bytes) % 16 != 0 || !aligned(args.b.data, 16) || args.b.ld % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "fp8_gemm: operand base and row pitch must be 16 B aligned");
  }
  if ((args.epi.ops & ~kFp8EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "fp8_gemm: epilogue ops 0x%x are not implemented", args.epi.ops & ~kFp8EpiAllowed);
  }
  if (((args.epi.ops & kEpiRowScale) != 0u && args.epi.row_scale == nullptr) ||
      ((args.epi.ops & kEpiColScale) != 0u && args.epi.col_scale == nullptr) ||
      ((args.epi.ops & kEpiBias) != 0u && args.epi.bias == nullptr) ||
      ((args.epi.ops & kEpiResidualAdd) != 0u && args.epi.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "fp8_gemm: epilogue operand missing (residual is read in the output dtype)");
  }
  if ((args.epi.ops & kEpiOutFp32) != 0u && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "fp8_gemm: kEpiOutFp32 needs an fp32 output");
  }
  if (rc == GemmRecipe::kFP8Blockwise) {
    if (args.k % 128 != 0) return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: blockwise K must be a multiple of 128");
    if (args.b.scale == nullptr) return detail::set_error(Status::kInvalidArgument, "fp8_gemm: blockwise needs B scales");
    if (adt == DType::kFloat8E4M3 && args.a.scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "fp8_gemm: blockwise e4m3 activations need 1x128 scales");
    }
    const bool ok_b = args.b.scale_dtype == DType::kFloat32 || args.b.scale_dtype == DType::kFloat8E8M0;
    const bool ok_a = args.a.scale == nullptr || args.a.scale_dtype == DType::kFloat32 || args.a.scale_dtype == DType::kFloat8E8M0;
    if (!ok_a || !ok_b) return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: block scales must be fp32 or UE8M0 bytes");
    if (args.a.scale_layout == ScaleLayout::kPackedUE8M0x4 || args.b.scale_layout == ScaleLayout::kPackedUE8M0x4 ||
        (args.a.scale != nullptr && args.a.scale_layout != ScaleLayout::kRowMajor && args.a.scale_layout != ScaleLayout::kColMajorTma) ||
        (args.b.scale_layout != ScaleLayout::kRowMajor)) {
      return detail::set_error(Status::kUnsupportedLayout, "fp8_gemm: A scales row-major or TMA column-major, B scales row-major");
    }
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (adt) {
    case DType::kBFloat16: return launch_fp8_recipe<__nv_bfloat16>(args, s);
    case DType::kFloat16: return launch_fp8_recipe<__half>(args, s);
    default: return launch_fp8_recipe<f8e4m3_t>(args, s);
  }
}

}  // namespace kops::gemm
