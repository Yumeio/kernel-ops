// =============================================================================
// kernel-ops | csrc/gemm/int8_gemm.cu
// B5 row 7: INT8 GEMV on CUDA cores.
//   int8_gemv_kernel   s8 weights [N, K]; s8 activations (dp4a, s32 accumulate) or bf16/fp16
//                      activations (W8A16, fp32 accumulate), M <= 16
// Host entry: int8_gemm (kINT8Tensorwise / kINT8Rowwise, M <= 16).
//
// Contract: csrc/gemm/int8_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// W8A8 is exact integer arithmetic up to the s32 -> fp32 conversion, so it is bit-exact against
// an integer oracle.  Asymmetric activations subtract azp[m] * colsum_b[n] from the s32
// accumulator (D = (acc - azp * colsum) * sA * sB + bias, the INT8 header's formula); the scales
// live in the epilogue (alpha, row_scale[m], col_scale[n]).  Reduction over the lanes of a warp is
// an integer xor-butterfly, so the result does not depend on M_TILE, the grid or the other rows.
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

/// Exact float of the signed byte b (low 8 bits): 2^23 + (b + 128) - (2^23 + 128).
__device__ __forceinline__ float s8_to_f(uint32_t b) {
  return __uint_as_float(0x4B000000u | ((b & 0xFFu) ^ 0x80u)) - 8388736.0f;
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

template <typename a_t, bool kAsymA, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void int8_gemv_kernel(const __grid_constant__ ScaledGemmArgs args) {
  constexpr bool kS8 = std::is_same<a_t, int8_t>::value;
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

  using acc_t = typename std::conditional<kS8, int32_t, float>::type;
  acc_t acc[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = acc_t(0);
  }

  #pragma unroll 2
  for (int kc = lane * 16; kc < k; kc += 512) {
    uint4 wv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = dev::ld_stream16(brow[r] + kc);
    if constexpr (kS8) {
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) {
        const uint4 av = *reinterpret_cast<const uint4*>(arow[m] + kc);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          int32_t t = acc[m][r];
          t = __dp4a(static_cast<int>(av.x), static_cast<int>(wv[r].x), t);
          t = __dp4a(static_cast<int>(av.y), static_cast<int>(wv[r].y), t);
          t = __dp4a(static_cast<int>(av.z), static_cast<int>(wv[r].z), t);
          t = __dp4a(static_cast<int>(av.w), static_cast<int>(wv[r].w), t);
          acc[m][r] = t;
        }
      }
    } else {
      float wf[R][16];
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t w4[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) wf[r][4 * q + i] = s8_to_f(w4[q] >> (8 * i));
        }
      }
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) {
        float lo[8], hi[8];
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[m] + kc), lo);
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[m] + kc + 8), hi);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[m][r] = fmaf(lo[i], wf[r][i], acc[m][r]);
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[m][r] = fmaf(hi[i], wf[r][8 + i], acc[m][r]);
        }
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      acc_t v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == m * R + r && m < m_total && n0 + r < args.n) {
        float f;
        if constexpr (kS8) {
          if constexpr (kAsymA) v -= args.epi.azp[m] * args.epi.colsum_b[n0 + r];
          f = static_cast<float>(v);
        } else {
          f = v;
        }
        store_result(args, m, n0 + r, f);
      }
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

template <typename a_t, bool kAsym, int MT, int R>
Status launch_int8_tile(const ScaledGemmArgs& args, cudaStream_t s) {
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((args.n + kWarps * R - 1) / (kWarps * R)));
  int8_gemv_kernel<a_t, kAsym, MT, R, kWarps><<<grid, kWarps * 32, 0, s>>>(args);
  return launch_error("int8_gemv_kernel");
}

template <typename a_t, bool kAsym>
Status launch_int8(const ScaledGemmArgs& args, cudaStream_t s) {
  if (args.m <= 1) return launch_int8_tile<a_t, kAsym, 1, 4>(args, s);
  if (args.m <= 2) return launch_int8_tile<a_t, kAsym, 2, 4>(args, s);
  if (args.m <= 4) return launch_int8_tile<a_t, kAsym, 4, 2>(args, s);
  if (args.m <= 8) return launch_int8_tile<a_t, kAsym, 8, 2>(args, s);
  return launch_int8_tile<a_t, kAsym, 16, 1>(args, s);
}

constexpr uint32_t kInt8EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                     kEpiResidualAdd | kEpiClamp | kEpiOutFp32 | kEpiAzpCorrection;

}  // namespace

Status int8_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                 const Workspace& ws, const LaunchContext& ctx) {
  (void)tactic;
  (void)ws;
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "int8_gemm: CUDA backend only");
  }
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "int8_gemm: only the native provider exists");
  }
  if (args.recipe != GemmRecipe::kINT8Tensorwise && args.recipe != GemmRecipe::kINT8Rowwise) {
    return detail::set_error(Status::kNotImplemented, "int8_gemm: recipe %d has no native body yet", static_cast<int>(args.recipe));
  }
  if (args.b.dtype != DType::kInt8) return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: weights must be int8");
  const DType adt = args.a.dtype;
  if (adt != DType::kInt8 && adt != DType::kBFloat16 && adt != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: activations must be int8, bf16 or fp16");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: output must be bf16, fp16 or fp32");
  }
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "int8_gemm: K-major operands only (weight_transpose first)");
  }
  if (args.batch != 1) return detail::set_error(Status::kNotImplemented, "int8_gemm: batch > 1");
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "int8_gemm: empty problem");
  if (args.m > 16) return detail::set_error(Status::kNotImplemented, "int8_gemm: M > 16 needs the tensor-core path (B7)");
  if (args.k % 16 != 0) return detail::set_error(Status::kUnsupportedShape, "int8_gemm: K must be a multiple of 16");
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: null operand");
  }
  const int a_bytes = adt == DType::kInt8 ? 1 : 2;
  if (args.a.ld < args.k || args.b.ld < args.k || args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: leading dimensions smaller than the extents");
  }
  if (!aligned(args.a.data, 16) || (args.a.ld * a_bytes) % 16 != 0 || !aligned(args.b.data, 16) || args.b.ld % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "int8_gemm: operand base and row pitch must be 16 B aligned");
  }
  if ((args.epi.ops & ~kInt8EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "int8_gemm: epilogue ops 0x%x are not implemented", args.epi.ops & ~kInt8EpiAllowed);
  }
  const bool asym = (args.epi.ops & kEpiAzpCorrection) != 0u;
  if (asym && (adt != DType::kInt8 || args.epi.azp == nullptr || args.epi.colsum_b == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: azp correction needs int8 activations, azp[M] and colsum_b[N]");
  }
  if (((args.epi.ops & kEpiRowScale) != 0u && args.epi.row_scale == nullptr) ||
      ((args.epi.ops & kEpiColScale) != 0u && args.epi.col_scale == nullptr) ||
      ((args.epi.ops & kEpiBias) != 0u && args.epi.bias == nullptr) ||
      ((args.epi.ops & kEpiResidualAdd) != 0u && args.epi.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: epilogue operand missing (residual is read in the output dtype)");
  }
  if ((args.epi.ops & kEpiOutFp32) != 0u && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: kEpiOutFp32 needs an fp32 output");
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (adt) {
    case DType::kInt8: return asym ? launch_int8<int8_t, true>(args, s) : launch_int8<int8_t, false>(args, s);
    case DType::kBFloat16: return launch_int8<__nv_bfloat16, false>(args, s);
    default: return launch_int8<__half, false>(args, s);
  }
}

}  // namespace kops::gemm
