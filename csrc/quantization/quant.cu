// =============================================================================
// kernel-ops | csrc/quantization/quant.cu
// B3 rows 1-4 native implementation:
//   Q01  absmax / per-tensor / per-token / per-group / MX / NVFP4 activations,
//        dequantize, int4 pack/unpack and the Hadamard rotation.
//
// Contract: csrc/quantization/ops.h + csrc/quantization/quant_kernels.cuh.
// Scales are dequant multipliers (x ~= q * s); every quotient is a correctly
// rounded fp32 division and every conversion is round-to-nearest-even with
// saturation, so the bytes are a function of the data alone (not of the fork,
// grid or vector width).  The kernels are architecture-generic, graph
// capturable and never allocate or synchronize on the host.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

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
#include "csrc/quantization/ops.h"

namespace kops::quant {
namespace {

using dev::from_f;
using dev::to_f;

// -----------------------------------------------------------------------------
// Device conversion helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

__device__ __forceinline__ uint8_t e2m1_pair(float a, float b) {
  const __nv_fp4x2_storage_t p =
      __nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest);
  return static_cast<uint8_t>(p);
}

/// e2m1 code -> fp32 (nibble: sign | exp:2 | mantissa:1).
__device__ __forceinline__ float e2m1_to_float(uint8_t code) {
  constexpr float kTable[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kTable[code & 0x7u];
  return (code & 0x8u) ? -v : v;
}

__device__ __forceinline__ int8_t int8_from_float(float x) {
  int q = __float2int_rn(x);
  q = q > 127 ? 127 : (q < -127 ? -127 : q);
  return static_cast<int8_t>(q);
}

__host__ __device__ __forceinline__ bool is_subbyte_dtype(DType t) {
  return t == DType::kFloat4E2M1x2 || t == DType::kInt4x2 || t == DType::kUInt4x2;
}

__device__ __forceinline__ float group_reduce_max_16(float v) {
  for (int off = 8; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off, 16));
  return __shfl_sync(0xffffffffu, v, 0, 16);
}

__global__ void quantize_scale_kernel(const float* __restrict__ amax, float* __restrict__ scale,
                                      float qmax) {
  const float a = amax[0];
  scale[0] = fmaxf(a, 1e-10f) / qmax;
}

/// Output storage tag -> one encoded byte. `out_t` is only a tag; storage is 1 B.
template <typename out_t>
__device__ __forceinline__ uint8_t encode_byte(float x);
template <>
__device__ __forceinline__ uint8_t encode_byte<__nv_fp8_e4m3>(float x) {
  return e4m3_from_float(x);
}
template <>
__device__ __forceinline__ uint8_t encode_byte<int8_t>(float x) {
  return static_cast<uint8_t>(int8_from_float(x));
}



/// Scale of one (row, group) for a given layout.
KOPS_DEVICE int64_t scale_offset(ScaleLayout layout, int64_t row, int64_t group, int64_t rows,
                                 int64_t groups) {
  switch (layout) {
    case ScaleLayout::kRowMajor:
    case ScaleLayout::kPackedUE8M0x4:
      return row * groups + group;
    case ScaleLayout::kColMajorTma: {
      const int64_t padded = (rows + 3) & ~int64_t(3);
      return group * padded + row;
    }
    case ScaleLayout::kSwizzled128x4: {
      // [rows/128, ceil(groups/4), 32, 4, 4]
      const int64_t m_tile = row >> 7;
      const int64_t outer_m = row & 31;
      const int64_t inner_m = (row >> 5) & 3;
      const int64_t k_tile = group >> 2;
      const int64_t inner_k = group & 3;
      const int64_t k_tiles = (groups + 3) >> 2;
      return ((m_tile * k_tiles + k_tile) << 9) + (outer_m << 4) + (inner_m << 2) + inner_k;
    }
    default:
      return row * groups + group;
  }
}

__device__ __forceinline__ uint8_t ue8m0_ceil_byte(float s) {
  int e = 0;
  if (s > 0.0f && isfinite(s)) {
    frexpf(s, &e);  // s = m * 2^e, m in [0.5, 1)
    e = e - 1;
    if (ldexpf(1.0f, e) < s) ++e;
  } else {
    e = -127;
  }
  if (e < -127) e = -127;
  if (e > 127) e = 127;
  return static_cast<uint8_t>(e + 127);
}



__device__ __forceinline__ float ue8m0_byte_to_float(uint8_t e) {
  if (e == 0xFFu) return NAN;
  return ldexpf(1.0f, static_cast<int>(e) - 127);
}

__device__ __forceinline__ float quant_qmax(DType t) {
  switch (t) {
    case DType::kFloat8E4M3: return kFp8E4M3Max;
    case DType::kInt8: return kInt8Max;
    case DType::kFloat4E2M1x2: return kFp4E2M1Max;
    default: return kFp8E4M3Max;
  }
}

__device__ __forceinline__ float quant_scale_from_amax(float amax, float eps, float qmax,
                                                       bool ue8m0) {
  float s = fmaxf(amax, eps) / qmax;
  if (ue8m0) s = ue8m0_byte_to_float(ue8m0_ceil_byte(s));
  return s;
}

template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_floats(const scalar_t* __restrict__ src, int64_t n, int64_t c,
                                            float (&x)[VEC]) {
  if (c + VEC <= n) {
    const dev::Pack<scalar_t, VEC> p = dev::load_pack<VEC>(src + c);
    #pragma unroll
    for (int j = 0; j < VEC; ++j) x[j] = to_f<scalar_t>(p.x[j]);
  } else {
    #pragma unroll
    for (int j = 0; j < VEC; ++j) x[j] = (c + j < n) ? to_f<scalar_t>(src[c + j]) : 0.0f;
  }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

template <typename T>
T* mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

Status backend_ok(const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "quantization: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool supported_scalar(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

// =============================================================================
// Absmax
// =============================================================================

template <typename scalar_t, int VEC_SIZE>
__global__ void absmax_reduce_kernel(float* __restrict__ amax, const scalar_t* __restrict__ input,
                                     const int64_t input_row_stride, const int64_t num_rows,
                                     const int64_t cols) {
  __shared__ float smem[32];
  const int64_t vpr = (cols + VEC_SIZE - 1) / VEC_SIZE;
  const int64_t total = num_rows * vpr;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  float vmax = 0.0f;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += step) {
    const int64_t row = i / vpr;
    const int64_t c = (i - row * vpr) * VEC_SIZE;
    float x[VEC_SIZE];
    load_floats<scalar_t, VEC_SIZE>(input + row * input_row_stride, cols, c, x);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) vmax = fmaxf(vmax, fabsf(x[j]));
  }
  vmax = kops::block_reduce_max<float, 8>(vmax, smem);
  if (threadIdx.x == 0 && vmax > 0.0f) {
    atomicMax(reinterpret_cast<unsigned*>(amax), __float_as_uint(vmax));
  }
}

Status absmax(const Tensor& amax_out, const Tensor& input, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(amax_out, DType::kFloat32, 1, 4, "absmax.amax_out");
  if (s != Status::kSuccess) return s;
  if (amax_out.shape[0] < 1) {
    return detail::set_error(Status::kInvalidArgument, "absmax.amax_out: empty");
  }
  if (!supported_scalar(input.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "absmax: input dtype %s",
                             dtype_name(input.dtype));
  }
  s = check_tensor(input, input.dtype, 2, 1, "absmax.input");
  if (s != Status::kSuccess) return s;
  if (input.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "absmax: last dim must be contiguous");
  }
  const int64_t rows = input.shape[0];
  const int64_t cols = input.shape[1];
  if (rows == 0 || cols == 0) return Status::kSuccess;
  const int64_t work = rows * ceil_div(cols, 8);
  const int sms = ctx.compute_units > 0 ? ctx.compute_units : 32;
  int grid = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(sms) * 4,
                                                ceil_div(work, 256)));
  if (grid < 1) grid = 1;
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_ABSMAX(T, V)                                                                          \
  do {                                                                                             \
    absmax_reduce_kernel<T, V><<<grid, 256, 0, stream>>>(                                          \
        mut_ptr<float>(amax_out), const_ptr<T>(input), input.stride[0], rows, cols);               \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("absmax", err);                                    \
    return Status::kSuccess;                                                                       \
  } while (0)
  if (input.dtype == DType::kFloat32) KOPS_ABSMAX(float, 4);
  if (input.dtype == DType::kFloat16) KOPS_ABSMAX(__half, 8);
  if (input.dtype == DType::kBFloat16) KOPS_ABSMAX(__nv_bfloat16, 8);
#undef KOPS_ABSMAX
  return Status::kUnsupportedDtype;
}

// =============================================================================
// Per-tensor / per-token / per-group
// =============================================================================

template <typename scalar_t, typename out_t, int VEC_SIZE>
__global__ void quantize_per_tensor_kernel(out_t* __restrict__ out, const int64_t out_row_stride,
                                           const scalar_t* __restrict__ input,
                                           const int64_t input_row_stride,
                                           const float* __restrict__ scale, const int64_t num_rows,
                                           const int64_t cols) {
  const float s = scale[0];
  uint8_t* dst = reinterpret_cast<uint8_t*>(out);
  const int64_t vpr = (cols + VEC_SIZE - 1) / VEC_SIZE;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_rows * vpr; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / vpr;
    const int64_t c = (i - row * vpr) * VEC_SIZE;
    float x[VEC_SIZE];
    load_floats<scalar_t, VEC_SIZE>(input + row * input_row_stride, cols, c, x);
    uint8_t* q = dst + row * out_row_stride + c;
    const int64_t valid = (cols - c < VEC_SIZE) ? (cols - c) : VEC_SIZE;
    for (int64_t j = 0; j < valid; ++j) q[j] = encode_byte<out_t>(__fdiv_rn(x[j], s));
  }
}

template <typename scalar_t, typename out_t, int VEC_SIZE, bool kAsymmetric>
__global__ void quantize_per_token_kernel(out_t* __restrict__ out, const int64_t out_row_stride,
                                          float* __restrict__ scales, int32_t* __restrict__ azp,
                                          const scalar_t* __restrict__ input,
                                          const int64_t input_row_stride,
                                          const float* __restrict__ scale_ub, const int num_rows,
                                          const int cols) {
  __shared__ float smem[32];
  const bool warp_row = cols <= 1024;
  const int tpr = warp_row ? 32 : static_cast<int>(blockDim.x);
  const int lane = threadIdx.x & 31;
  const int64_t row = warp_row ? static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y
                               : blockIdx.x;
  const bool active = row < num_rows;
  const int64_t r = active ? row : 0;
  const scalar_t* src = input + r * input_row_stride;
  const int64_t vpr = (cols + VEC_SIZE - 1) / VEC_SIZE;
  float row_max = -__int_as_float(0x7f800000);
  float row_min = __int_as_float(0x7f800000);
  for (int64_t i = warp_row ? lane : threadIdx.x; i < vpr; i += tpr) {
    float x[VEC_SIZE];
    load_floats<scalar_t, VEC_SIZE>(src, cols, i * VEC_SIZE, x);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      row_max = fmaxf(row_max, x[j]);
      row_min = fminf(row_min, x[j]);
    }
  }
  float scale = 1.0f;
  float zero = -128.0f;
  (void)zero;
  if constexpr (kAsymmetric) {
    if (warp_row) {
      row_max = kops::warp_reduce_max(row_max);
      row_min = __shfl_sync(0xffffffffu, kops::warp_reduce_min(row_min), 0);
    } else {
      row_max = kops::block_reduce_max<float, 8>(row_max, smem);
      __syncthreads();
      row_min = -kops::block_reduce_max<float, 8>(-row_min, smem);
    }
    scale = __fdiv_rn(row_max - row_min, 255.0f);
    scale = (scale == scale) ? fmaxf(scale, 1.0f / (127.0f * 512.0f)) : scale;
    const int z = __float2int_rn(-128.0f - __fdiv_rn(row_min, scale));
    zero = static_cast<float>(z);
    if ((warp_row ? lane : threadIdx.x) == 0) {
      if (active) scales[r] = scale;
      if (active && azp != nullptr) azp[r] = z;
    }
  } else {
    float amax = 0.0f;
    if (warp_row) {
      amax = kops::warp_reduce_max(fabsf(row_max) > fabsf(row_min) ? fabsf(row_max) : fabsf(row_min));
    } else {
      const float m = fmaxf(fabsf(row_max), fabsf(row_min));
      amax = kops::block_reduce_max<float, 8>(m, smem);
    }
    const float qmax = std::is_same<out_t, int8_t>::value ? kInt8Max : kFp8E4M3Max;
    scale = fmaxf(__fdiv_rn(amax, qmax), 1.0f / (qmax * 512.0f));
    if (scale_ub != nullptr) scale = fminf(scale, scale_ub[0]);
    if ((warp_row ? lane : threadIdx.x) == 0) {
      if (active) scales[r] = scale;
    }
  }
  (void)zero;
  __syncthreads();
  uint8_t* dst = reinterpret_cast<uint8_t*>(out) + r * out_row_stride;
  if (!active) return;
  for (int64_t i = warp_row ? lane : threadIdx.x; i < vpr; i += tpr) {
    float x[VEC_SIZE];
    load_floats<scalar_t, VEC_SIZE>(src, cols, i * VEC_SIZE, x);
    const int64_t c = i * VEC_SIZE;
    const int64_t valid = (cols - c < VEC_SIZE) ? (cols - c) : VEC_SIZE;
    for (int64_t j = 0; j < valid; ++j) {
      if constexpr (kAsymmetric) {
        const float y = __fadd_rn(__fdiv_rn(x[j], scale), zero);
        int q = __float2int_rn(y);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        dst[c + j] = static_cast<uint8_t>(static_cast<int8_t>(q));
      } else {
        dst[c + j] = encode_byte<out_t>(__fdiv_rn(x[j], scale));
      }
    }
  }
}

/// One warp owns one group of 32/64/128 values; lane 0 writes the scale.
template <typename scalar_t, typename out_t, int GROUP_SIZE, bool kColMajorScale, bool kUE8M0>
__global__ void quantize_per_group_kernel(out_t* __restrict__ out, const int64_t out_row_stride,
                                          void* __restrict__ scales, const int64_t scale_stride,
                                          const scalar_t* __restrict__ input,
                                          const int64_t input_row_stride,
                                          const int32_t* __restrict__ masked_m,
                                          const int max_m, const int num_rows, const int cols,
                                          const float eps) {
  constexpr int kWarps = 8;
  constexpr int kPerLane = GROUP_SIZE / 32;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t groups = cols / GROUP_SIZE;
  uint8_t* dst = reinterpret_cast<uint8_t*>(out);
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * kWarps + warp; group < num_rows * groups;
       group += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = group / groups;
    const int64_t g = group - row * groups;
    if (masked_m != nullptr) {
      const int e = static_cast<int>(row / max_m);
      const int m = static_cast<int>(row - static_cast<int64_t>(e) * max_m);
      if (m >= masked_m[e]) continue;
    }
    const scalar_t* src = input + row * input_row_stride + g * GROUP_SIZE;
    float x[kPerLane];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int j = lane + 32 * i;
      x[i] = to_f<scalar_t>(src[j]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    amax = kops::warp_reduce_max(amax);
    float s = quant_scale_from_amax(amax, eps, quant_qmax(DType::kFloat8E4M3), kUE8M0);
    if (lane == 0) {
      float* sc = static_cast<float*>(scales);
      const int64_t off = kColMajorScale
                              ? scale_offset(ScaleLayout::kColMajorTma, row, g, num_rows, groups)
                              : row * scale_stride + g;
      sc[off] = s;
    }
    s = __shfl_sync(0xffffffffu, s, 0);
    uint8_t* q = dst + row * out_row_stride + g * GROUP_SIZE;
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int j = lane + 32 * i;
      q[j] = encode_byte<out_t>(__fdiv_rn(x[i], s));
    }
  }
}

template <typename scalar_t, typename out_t, int GROUP_SIZE>
void launch_group_impl(const ActQuantOut& out, const Tensor& input, const int32_t* masked_m,
                       int max_m, int64_t rows, int64_t cols, int64_t input_row_stride,
                       int64_t input_expert_stride, cudaStream_t stream) {
  constexpr int kWarps = 8;
  const int64_t groups = cols / GROUP_SIZE;
  const int64_t tasks = rows * groups;
  const int grid = static_cast<int>(std::min<int64_t>(ceil_div(tasks, kWarps), INT32_MAX));
  const int64_t q_stride = out.q_row_stride > 0 ? out.q_row_stride : cols;
  const int64_t scale_stride =
      out.scale_row_stride > 0 ? out.scale_row_stride : groups;
  const float eps = 1e-10f;
  if (out.scale_layout == ScaleLayout::kColMajorTma) {
    quantize_per_group_kernel<scalar_t, out_t, GROUP_SIZE, true, false>
        <<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<out_t*>(out.q), q_stride, out.scales, scale_stride,
            const_ptr<scalar_t>(input), input_row_stride, masked_m, max_m, static_cast<int>(rows),
            static_cast<int>(cols), eps);
  } else {
    quantize_per_group_kernel<scalar_t, out_t, GROUP_SIZE, false, false>
        <<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<out_t*>(out.q), q_stride, out.scales, scale_stride,
            const_ptr<scalar_t>(input), input_row_stride, masked_m, max_m, static_cast<int>(rows),
            static_cast<int>(cols), eps);
  }
  (void)input_expert_stride;
}

// =============================================================================
// MX / NVFP4
// =============================================================================

template <typename scalar_t, typename out_t, ScaleLayout kScaleLayout, bool kRoundDown>
__global__ void quantize_mx_kernel(out_t* __restrict__ out, const int64_t out_row_stride,
                                   uint8_t* __restrict__ scales, const scalar_t* __restrict__ input,
                                   const int64_t input_row_stride, const int num_rows,
                                   const int cols) {
  constexpr bool kFp4 = std::is_same<out_t, __nv_fp4x2_e2m1>::value;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = 8;
  const int64_t groups = cols / kMxBlock;
  uint8_t* dst = reinterpret_cast<uint8_t*>(out);
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * kWarps + warp; group < num_rows * groups;
       group += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = group / groups;
    const int64_t g = group - row * groups;
    const scalar_t* src = input + row * input_row_stride + g * kMxBlock;
    const float x = to_f<scalar_t>(src[lane]);
    const float amax = kops::warp_reduce_max(fabsf(x));
    const float qmax = kFp4 ? kFp4E2M1Max : kFp8E4M3Max;
    float s = 0.0f;
    uint8_t scale_byte = 0;
    if (lane == 0) {
      s = fmaxf(amax, 1e-10f) / qmax;
      int e = 0;
      if (s > 0.0f && isfinite(s)) {
        frexpf(s, &e);
        e = e - 1;
        if (kRoundDown) {
          if (ldexpf(1.0f, e) > s) --e;
        } else {
          if (ldexpf(1.0f, e) < s) ++e;
        }
      } else {
        e = -127;
      }
      e = e < -127 ? -127 : (e > 127 ? 127 : e);
      scale_byte = static_cast<uint8_t>(e + 127);
      s = ldexpf(1.0f, e);
      scales[scale_offset(kScaleLayout, row, g, num_rows, groups)] = scale_byte;
    }
    s = __shfl_sync(0xffffffffu, s, 0);
    const float q = __fdiv_rn(x, s);
    if constexpr (kFp4) {
      const float q_other = __shfl_down_sync(0xffffffffu, q, 1);
      if ((lane & 1) == 0) {
        dst[row * out_row_stride + g * (kMxBlock / 2) + lane / 2] = e2m1_pair(q, q_other);
      }
    } else {
      dst[row * out_row_stride + g * kMxBlock + lane] = e4m3_from_float(q);
    }
  }
}

template <typename scalar_t, ScaleLayout kScaleLayout>
__global__ void quantize_nvfp4_kernel(uint8_t* __restrict__ out, const int64_t out_row_stride,
                                      f8e4m3_t* __restrict__ block_scales,
                                      const scalar_t* __restrict__ input,
                                      const int64_t input_row_stride,
                                      const float* __restrict__ global_scale, const int num_rows,
                                      const int cols) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = 8;
  const int64_t groups = cols / kNvFp4Block;
  const float gscale = global_scale != nullptr ? global_scale[0] : 1.0f;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * kWarps + warp; group < num_rows * groups;
       group += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = group / groups;
    const int64_t g = group - row * groups;
    const int half = lane & 15;
    const int64_t base = row * input_row_stride + g * kNvFp4Block;
    const float x = (lane < 16) ? to_f<scalar_t>(input[base + half]) : 0.0f;
    const float amax = group_reduce_max_16(fabsf(x));
    float sf = 0.0f;
    if (lane == 0) {
      sf = __fdiv_rn(__fmul_rn(amax, gscale), kFp4E2M1Max);
      const uint8_t sf_byte = e4m3_from_float(sf);
      sf = e4m3_to_float(sf_byte);
      reinterpret_cast<uint8_t*>(block_scales)[scale_offset(kScaleLayout, row, g, num_rows, groups)] =
          sf_byte;
    }
    sf = __shfl_sync(0xffffffffu, sf, 0);
    const float q = (lane < 16 && sf > 0.0f) ? __fdiv_rn(__fmul_rn(x, gscale), sf) : 0.0f;
    const float q_other = __shfl_down_sync(0xffffffffu, q, 1);
    if (lane < 16 && (lane & 1) == 0) {
      const int64_t off = row * out_row_stride + g * (kNvFp4Block / 2) + lane / 2;
      out[off] = e2m1_pair(q, q_other);
    }
  }
}

template <typename scalar_t, ScaleLayout kScaleLayout>
__global__ void quantize_nvfp4_pair_kernel(uint8_t* __restrict__ out,
                                           const int64_t out_row_stride,
                                           f8e4m3_t* __restrict__ block_scales,
                                           const scalar_t* __restrict__ input,
                                           const int64_t input_row_stride,
                                           const float* __restrict__ global_scale,
                                           const int num_rows, const int cols) {
  const int lane = threadIdx.x & 31;
  const int half = lane & 15;
  const bool second = lane >= 16;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = 8;
  const int64_t pairs = static_cast<int64_t>(num_rows) * (cols / kNvFp4Block / 2);
  const float gscale = global_scale != nullptr ? global_scale[0] : 1.0f;
  for (int64_t p = static_cast<int64_t>(blockIdx.x) * kWarps + warp; p < pairs;
       p += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = p / (cols / kNvFp4Block / 2);
    const int64_t gp = p - row * (cols / kNvFp4Block / 2);
    const int64_t g = 2 * gp + (second ? 1 : 0);
    const int64_t base = row * input_row_stride + g * kNvFp4Block;
    const float x = to_f<scalar_t>(input[base + half]);
    const float amax = group_reduce_max_16(fabsf(x));
    float sf = 0.0f;
    if (half == 0) {
      sf = __fdiv_rn(__fmul_rn(amax, gscale), kFp4E2M1Max);
      const uint8_t sf_byte = e4m3_from_float(sf);
      sf = e4m3_to_float(sf_byte);
      reinterpret_cast<uint8_t*>(block_scales)[
          scale_offset(kScaleLayout, row, g, num_rows, cols / kNvFp4Block)] = sf_byte;
    }
    sf = __shfl_sync(0xffffffffu, sf, 0, 16);
    const float q = (sf > 0.0f) ? __fdiv_rn(__fmul_rn(x, gscale), sf) : 0.0f;
    const float q_other = __shfl_down_sync(0xffffffffu, q, 1, 16);
    if ((half & 1) == 0) {
      out[row * out_row_stride + g * (kNvFp4Block / 2) + half / 2] = e2m1_pair(q, q_other);
    }
  }
}

// =============================================================================
// Dequantize
// =============================================================================

template <typename out_t, ScaleGranularity kGranularity>
__global__ void dequantize_kernel(out_t* __restrict__ out, const int64_t out_row_stride,
                                  const void* __restrict__ q, const int64_t q_row_stride,
                                  const void* __restrict__ scales,
                                  const void* __restrict__ zero_points,
                                  const QuantScheme scheme, const int64_t num_rows,
                                  const int64_t cols) {
  const int64_t total = num_rows * cols;
  const int group = scheme.group_k > 0 ? scheme.group_k : 128;
  (void)group;
  const uint8_t* payload = static_cast<const uint8_t*>(q);
  const bool packed = is_subbyte_dtype(scheme.storage_dtype);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = idx / cols;
    const int64_t col = idx - row * cols;
    float x = 0.0f;
    if (scheme.storage_dtype == DType::kFloat8E4M3 || scheme.storage_dtype == DType::kUInt8) {
      const uint8_t b = payload[row * q_row_stride + col];
      x = e4m3_to_float(b);
    } else if (scheme.storage_dtype == DType::kInt8) {
      x = static_cast<float>(reinterpret_cast<const int8_t*>(payload)[row * q_row_stride + col]);
    } else if (scheme.storage_dtype == DType::kInt4x2 || scheme.storage_dtype == DType::kUInt4x2) {
      const uint8_t b = payload[row * q_row_stride + col / 2];
      int v = (col & 1) ? (b >> 4) : (b & 15);
      if (scheme.storage_dtype == DType::kInt4x2 && (v & 8)) v -= 16;
      x = static_cast<float>(v);
    } else if (scheme.storage_dtype == DType::kFloat4E2M1x2) {
      const uint8_t b = payload[row * q_row_stride + col / 2];
      x = e2m1_to_float((col & 1) ? (b >> 4) : (b & 15));
    } else {
      continue;
    }
    float s = 1.0f;
    if constexpr (kGranularity == ScaleGranularity::kPerTensor) {
      s = static_cast<const float*>(scales)[0];
    } else if constexpr (kGranularity == ScaleGranularity::kPerChannel) {
      s = static_cast<const float*>(scales)[col];
    } else if constexpr (kGranularity == ScaleGranularity::kPerToken) {
      s = static_cast<const float*>(scales)[row];
    } else if constexpr (kGranularity == ScaleGranularity::kPerGroup) {
      s = static_cast<const float*>(scales)[row * (cols / group) + col / group];
    } else if constexpr (kGranularity == ScaleGranularity::kPerBlock2D) {
      const int gm = scheme.group_m > 0 ? scheme.group_m : 1;
      s = static_cast<const float*>(scales)[(row / gm) * (cols / group) + col / group];
    } else if constexpr (kGranularity == ScaleGranularity::kMX) {
      if (scheme.scale_dtype == DType::kFloat8E8M0) {
        const uint8_t e = static_cast<const uint8_t*>(scales)[row * (cols / 32) + col / 32];
        s = ue8m0_byte_to_float(e);
      } else {
        s = static_cast<const float*>(scales)[row * (cols / 32) + col / 32];
      }
    } else {
      s = 1.0f;
    }
    float z = 0.0f;
    if (zero_points != nullptr && scheme.has_zero_point) {
      if (scheme.zero_point_is_float) {
        z = static_cast<const float*>(zero_points)[0];
      } else {
        z = static_cast<float>(static_cast<const int8_t*>(zero_points)[0]);
      }
    }
    out_t y;
    from_f<out_t>(y, __fmul_rn(x - z, s));
    out[row * out_row_stride + col] = y;
  }
}

// =============================================================================
// INT4 pack / unpack and Hadamard
// =============================================================================

template <typename out_t>
__global__ void dequant_fp8_per_token_vec_kernel(out_t* __restrict__ out,
                                                 const int64_t out_row_stride,
                                                 const uint8_t* __restrict__ q,
                                                 const int64_t q_row_stride,
                                                 const float* __restrict__ scales,
                                                 const int64_t num_rows, const int64_t cols) {
  const int64_t vpr = cols / 4;
  const int64_t total = num_rows * vpr;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / vpr;
    const int64_t c = (i - row * vpr) * 4;
    const uint32_t w = *reinterpret_cast<const uint32_t*>(q + row * q_row_stride + c);
    const float s = scales[row];
    dev::Pack<out_t, 4> p;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = e4m3_to_float(static_cast<uint8_t>(w >> (8 * j))) * s;
      from_f<out_t>(p.x[j], x);
    }
    *reinterpret_cast<dev::Pack<out_t, 4>*>(out + row * out_row_stride + c) = p;
  }
}

__global__ void pack_int4_kernel(uint8_t* __restrict__ out, const int8_t* __restrict__ input,
                                 const int64_t num_rows, const int64_t cols,
                                 const bool is_unsigned) {
  const int64_t pairs = num_rows * (cols / 2);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pairs;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / (cols / 2);
    const int64_t c = (i - row * (cols / 2)) * 2;
    uint8_t lo = static_cast<uint8_t>(input[row * cols + c]) & 0x0Fu;
    uint8_t hi = static_cast<uint8_t>(input[row * cols + c + 1]) & 0x0Fu;
    if (is_unsigned) {
      lo &= 0x0Fu;
      hi &= 0x0Fu;
    }
    out[i] = static_cast<uint8_t>(lo | (hi << 4));
  }
}

__global__ void unpack_int4_vec_kernel(int8_t* __restrict__ out, const uint8_t* __restrict__ input,
                                       const int64_t num_rows, const int64_t cols,
                                       const bool is_unsigned) {
  const int64_t bytes = num_rows * (cols / 2);
  constexpr int kBytes = 4;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes / kBytes;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / (cols / 8);
    const int64_t c = (i - row * (cols / 8)) * 8;
    const uint32_t w =
        *reinterpret_cast<const uint32_t*>(input + row * (cols / 2) + c / 2);
    uint8_t codes[8];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint8_t byte = static_cast<uint8_t>(w >> (8 * b));
      int lo = byte & 15;
      int hi = (byte >> 4) & 15;
      if (!is_unsigned) {
        if (lo & 8) lo -= 16;
        if (hi & 8) hi -= 16;
      }
      codes[2 * b] = static_cast<uint8_t>(lo);
      codes[2 * b + 1] = static_cast<uint8_t>(hi);
    }
    *reinterpret_cast<uint32_t*>(out + row * cols + c) =
        static_cast<uint32_t>(codes[0]) | (static_cast<uint32_t>(codes[1]) << 8) |
        (static_cast<uint32_t>(codes[2]) << 16) | (static_cast<uint32_t>(codes[3]) << 24);
    *reinterpret_cast<uint32_t*>(out + row * cols + c + 4) =
        static_cast<uint32_t>(codes[4]) | (static_cast<uint32_t>(codes[5]) << 8) |
        (static_cast<uint32_t>(codes[6]) << 16) | (static_cast<uint32_t>(codes[7]) << 24);
  }
}

__global__ void unpack_int4_kernel(int8_t* __restrict__ out, const uint8_t* __restrict__ input,
                                   const int64_t num_rows, const int64_t cols,
                                   const bool is_unsigned) {
  const int64_t total = num_rows * cols;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / cols;
    const int64_t c = i - row * cols;
    const uint8_t b = input[row * (cols / 2) + c / 2];
    int v = (c & 1) ? (b >> 4) : (b & 15);
    if (!is_unsigned && (v & 8)) v -= 16;
    out[i] = static_cast<int8_t>(v);
  }
}

template <typename scalar_t, int LOG_N, bool kSigned>
__global__ void hadamard_transform_kernel(scalar_t* __restrict__ out, const int64_t out_row_stride,
                                          const scalar_t* __restrict__ input,
                                          const int64_t input_row_stride, const float scale,
                                          const uint32_t transform_seed, const int num_rows) {
  extern __shared__ float smem[];
  const int n = 1 << LOG_N;
  const int64_t row = blockIdx.x;
  if (row >= num_rows) return;
  const scalar_t* src = input + row * input_row_stride;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = to_f<scalar_t>(src[i]);
    if constexpr (kSigned) {
      // Deterministic +-1 diagonal from the seed and the column index.
      uint32_t h = transform_seed ^ (static_cast<uint32_t>(i) * 0x9E3779B9u);
      h ^= h >> 16;
      h *= 0x7FEB352Du;
      h ^= h >> 15;
      v = (h & 1u) ? -v : v;
    }
    smem[i] = v;
  }
  __syncthreads();
  for (int len = 1; len < n; len <<= 1) {
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
      const int block = (i / len) * (len << 1);
      const int j = i % len;
      const int a = block + j;
      const int b = a + len;
      const float u = smem[a];
      const float v = smem[b];
      smem[a] = u + v;
      smem[b] = u - v;
    }
    __syncthreads();
  }
  scalar_t* dst = out + row * out_row_stride;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    scalar_t y;
    from_f<scalar_t>(y, smem[i] * scale);
    dst[i] = y;
  }
}

// =============================================================================
// Host entry points
// =============================================================================

Status quantize_activation(const ActQuantOut& out, const Tensor& input, const Tensor& masked_m,
                           int32_t max_m, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(input.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "quantize_activation: input dtype %s",
                             dtype_name(input.dtype));
  }
  s = check_tensor(input, input.dtype, -1, 1, "quantize_activation.input");
  if (s != Status::kSuccess) return s;
  if (input.ndim != 2 && input.ndim != 3) {
    return detail::set_error(Status::kInvalidArgument, "quantize_activation: input rank %d",
                             input.ndim);
  }
  if (input.stride[input.ndim - 1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "quantize_activation: last dim must be contiguous");
  }
  const bool masked = masked_m.data != nullptr;
  if (masked) {
    s = check_tensor(masked_m, DType::kInt32, 1, 4, "quantize_activation.masked_m");
    if (s != Status::kSuccess) return s;
    if (input.ndim != 3 || max_m <= 0 ||
        input.shape[0] != masked_m.shape[0] || input.shape[1] != max_m) {
      return detail::set_error(Status::kInvalidArgument,
                               "quantize_activation: masked input must be [E, max_m, cols]");
    }
  } else if (input.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument,
                             "quantize_activation: dense input must be [rows, cols]");
  }
  if (masked && out.mode != ActQuantMode::kPerGroup) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "quantize_activation: masked layout supports per-group only");
  }
  const int64_t rows = masked ? input.shape[0] * input.shape[1] : input.shape[0];
  const int64_t cols = input.shape[input.ndim - 1];
  if (rows == 0 || cols == 0) return Status::kSuccess;
  if (out.q == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "quantize_activation: out.q is null");
  }
  if (out.q_dtype == DType::kInvalid) {
    return detail::set_error(Status::kUnsupportedDtype, "quantize_activation: q_dtype invalid");
  }
  const bool fp8 = out.q_dtype == DType::kFloat8E4M3;
  const bool int8 = out.q_dtype == DType::kInt8;
  const int64_t q_row_stride = out.q_row_stride > 0 ? out.q_row_stride
                                                    : (is_subbyte_dtype(out.q_dtype) ? cols / 2 : cols);
  const int64_t input_row_stride = masked ? input.stride[1] : input.stride[0];
  const int64_t input_expert_stride = masked ? input.stride[0] : 0;
  const cudaStream_t stream = stream_of(ctx);
  const int sms = ctx.compute_units > 0 ? ctx.compute_units : 32;

  if (out.mode == ActQuantMode::kStaticPerTensor) {
    if (out.static_scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "quantize_activation: static mode needs static_scale");
    }
    const int grid = static_cast<int>(std::min<int64_t>(sms * 4, ceil_div(rows * cols, 256)));
    if (fp8) {
      if (input.dtype == DType::kBFloat16) {
        quantize_per_tensor_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<__nv_bfloat16>(input),
            input_row_stride, out.static_scale, rows, cols);
      } else if (input.dtype == DType::kFloat16) {
        quantize_per_tensor_kernel<__half, __nv_fp8_e4m3, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<__half>(input),
            input_row_stride, out.static_scale, rows, cols);
      } else {
        quantize_per_tensor_kernel<float, __nv_fp8_e4m3, 4><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<float>(input),
            input_row_stride, out.static_scale, rows, cols);
      }
    } else if (int8) {
      if (input.dtype == DType::kBFloat16) {
        quantize_per_tensor_kernel<__nv_bfloat16, int8_t, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<__nv_bfloat16>(input),
            input_row_stride, out.static_scale, rows, cols);
      } else if (input.dtype == DType::kFloat16) {
        quantize_per_tensor_kernel<__half, int8_t, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<__half>(input),
            input_row_stride, out.static_scale, rows, cols);
      } else {
        quantize_per_tensor_kernel<float, int8_t, 4><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<float>(input),
            input_row_stride, out.static_scale, rows, cols);
      }
    } else {
      return detail::set_error(Status::kUnsupportedDtype,
                               "quantize_activation: static mode supports FP8 / INT8");
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("quantize_activation static", err);
    return Status::kSuccess;
  }

  if (out.mode == ActQuantMode::kDynamicPerTensor) {
    if (out.scales == nullptr || out.amax_out == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "quantize_activation: dynamic per-tensor needs scales and amax_out");
    }
    cudaMemsetAsync(out.amax_out, 0, sizeof(float), stream);
    Tensor amax_t{};
    amax_t.data = out.amax_out;
    amax_t.dtype = DType::kFloat32;
    amax_t.ndim = 1;
    amax_t.device = input.device;
    amax_t.shape[0] = 1;
    amax_t.stride[0] = 1;
    Tensor in_view = input;
    if (masked) {
      // Dense upper-bound view: skipped rows only affect amax if they contain
      // garbage; callers zero them (DeepEP contract).
      in_view.ndim = 2;
      in_view.shape[0] = rows;
      in_view.shape[1] = cols;
      in_view.stride[0] = input_expert_stride != 0 ? input_expert_stride : cols;
    }
    s = absmax(amax_t, in_view, ctx);
    if (s != Status::kSuccess) return s;
    // scale = max(amax, eps) / qmax
    const float qmax = fp8 ? kFp8E4M3Max : kInt8Max;
    if (input.dtype == DType::kBFloat16) {
      quantize_scale_kernel<<<1, 1, 0, stream>>>(out.amax_out,
                                                 reinterpret_cast<float*>(out.scales), qmax);
    } else {
      quantize_scale_kernel<<<1, 1, 0, stream>>>(out.amax_out,
                                                 reinterpret_cast<float*>(out.scales), qmax);
    }
    const int grid = static_cast<int>(std::min<int64_t>(sms * 4, ceil_div(rows * cols, 256)));
    if (fp8) {
      if (input.dtype == DType::kBFloat16) {
        quantize_per_tensor_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<__nv_bfloat16>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      } else if (input.dtype == DType::kFloat16) {
        quantize_per_tensor_kernel<__half, __nv_fp8_e4m3, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<__half>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      } else {
        quantize_per_tensor_kernel<float, __nv_fp8_e4m3, 4><<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride, const_ptr<float>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      }
    } else if (int8) {
      if (input.dtype == DType::kBFloat16) {
        quantize_per_tensor_kernel<__nv_bfloat16, int8_t, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<__nv_bfloat16>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      } else if (input.dtype == DType::kFloat16) {
        quantize_per_tensor_kernel<__half, int8_t, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<__half>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      } else {
        quantize_per_tensor_kernel<float, int8_t, 4><<<grid, 256, 0, stream>>>(
            reinterpret_cast<int8_t*>(out.q), q_row_stride, const_ptr<float>(input),
            input_row_stride, static_cast<float*>(out.scales), rows, cols);
      }
    } else {
      return detail::set_error(Status::kUnsupportedDtype,
                               "quantize_activation: per-tensor supports FP8 / INT8");
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("quantize_activation dynamic", err);
    return Status::kSuccess;
  }

  if (out.mode == ActQuantMode::kDynamicPerToken) {
    if (out.scales == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "quantize_activation: per-token needs scales");
    }
    const bool asymmetric = !out.symmetric && int8;
      const dim3 block = cols <= 1024 ? dim3(32, 4) : dim3(256);
    if (cols <= 1024) {
      const dim3 grid(static_cast<unsigned>(ceil_div(rows, 4)));
      if (fp8) {
        if (input.dtype == DType::kBFloat16) {
          quantize_per_token_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8, false>
              <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<__nv_bfloat16>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        } else if (input.dtype == DType::kFloat16) {
          quantize_per_token_kernel<__half, __nv_fp8_e4m3, 8, false>
              <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<__half>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        } else {
          quantize_per_token_kernel<float, __nv_fp8_e4m3, 4, false>
              <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<float>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        }
      } else if (int8 && asymmetric) {
        if (input.dtype == DType::kBFloat16) {
          quantize_per_token_kernel<__nv_bfloat16, int8_t, 8, true>
              <<<grid, block, 0, stream>>>(reinterpret_cast<int8_t*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<__nv_bfloat16>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        } else if (input.dtype == DType::kFloat16) {
          quantize_per_token_kernel<__half, int8_t, 8, true>
              <<<grid, block, 0, stream>>>(reinterpret_cast<int8_t*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<__half>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        } else {
          quantize_per_token_kernel<float, int8_t, 4, true>
              <<<grid, block, 0, stream>>>(reinterpret_cast<int8_t*>(out.q), q_row_stride,
                                            static_cast<float*>(out.scales), out.zero_points,
                                            const_ptr<float>(input), input_row_stride,
                                            out.scale_upper_bound, static_cast<int>(rows),
                                            static_cast<int>(cols));
        }
      } else {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "quantize_activation: per-token supports FP8 / symmetric INT8");
      }
    } else {
      if (fp8) {
        if (input.dtype == DType::kBFloat16) {
          quantize_per_token_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8, false>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                  static_cast<float*>(out.scales), out.zero_points, const_ptr<__nv_bfloat16>(input),
                  input_row_stride, out.scale_upper_bound, static_cast<int>(rows),
                  static_cast<int>(cols));
        } else if (input.dtype == DType::kFloat16) {
          quantize_per_token_kernel<__half, __nv_fp8_e4m3, 8, false>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                  static_cast<float*>(out.scales), out.zero_points, const_ptr<__half>(input),
                  input_row_stride, out.scale_upper_bound, static_cast<int>(rows),
                  static_cast<int>(cols));
        } else {
          quantize_per_token_kernel<float, __nv_fp8_e4m3, 4, false>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,
                  static_cast<float*>(out.scales), out.zero_points, const_ptr<float>(input),
                  input_row_stride, out.scale_upper_bound, static_cast<int>(rows),
                  static_cast<int>(cols));
        }
      } else if (int8 && asymmetric) {
        if (input.dtype == DType::kBFloat16) {
          quantize_per_token_kernel<__nv_bfloat16, int8_t, 8, true>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<int8_t*>(out.q), q_row_stride, static_cast<float*>(out.scales),
                  out.zero_points, const_ptr<__nv_bfloat16>(input), input_row_stride,
                  out.scale_upper_bound, static_cast<int>(rows), static_cast<int>(cols));
        } else if (input.dtype == DType::kFloat16) {
          quantize_per_token_kernel<__half, int8_t, 8, true>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<int8_t*>(out.q), q_row_stride, static_cast<float*>(out.scales),
                  out.zero_points, const_ptr<__half>(input), input_row_stride,
                  out.scale_upper_bound, static_cast<int>(rows), static_cast<int>(cols));
        } else {
          quantize_per_token_kernel<float, int8_t, 4, true>
              <<<static_cast<int>(rows), block, 0, stream>>>(
                  reinterpret_cast<int8_t*>(out.q), q_row_stride, static_cast<float*>(out.scales),
                  out.zero_points, const_ptr<float>(input), input_row_stride,
                  out.scale_upper_bound, static_cast<int>(rows), static_cast<int>(cols));
        }
      } else {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "quantize_activation: per-token supports FP8 / symmetric INT8");
      }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("quantize_activation per-token", err);
    return Status::kSuccess;
  }

  if (out.mode == ActQuantMode::kPerGroup) {
    if (out.scales == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "quantize_activation: per-group needs scales");
    }
    if (out.group_size != 32 && out.group_size != 64 && out.group_size != 128) {
      return detail::set_error(Status::kUnsupportedShape,
                               "quantize_activation: group_size %d (32/64/128)",
                               out.group_size);
    }
    if (cols % out.group_size != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "quantize_activation: cols %lld not divisible by group %d",
                               static_cast<long long>(cols), out.group_size);
    }
    const int32_t* masked_ptr = masked ? const_ptr<int32_t>(masked_m) : nullptr;
    if (out.scale_row_stride == 0) {
      // keep zero: kernels fall back to groups for row-major
    }
    const int64_t scale_stride = out.scale_row_stride > 0 ? out.scale_row_stride
                                                          : cols / out.group_size;
    #define KOPS_GQ(G, T)                                                                          \
      do {                                                                                         \
        if (out.q_dtype == DType::kFloat8E4M3) {                                                   \
          if (input.dtype == DType::kBFloat16) {                                                   \
            launch_group_impl<__nv_bfloat16, __nv_fp8_e4m3, G>(out, input, masked_ptr, max_m, rows, \
                                                               cols, input_row_stride,             \
                                                               input_expert_stride, stream);       \
          } else if (input.dtype == DType::kFloat16) {                                             \
            launch_group_impl<__half, __nv_fp8_e4m3, G>(out, input, masked_ptr, max_m, rows, cols, \
                                                        input_row_stride, input_expert_stride,     \
                                                        stream);                                   \
          } else {                                                                                 \
            launch_group_impl<float, __nv_fp8_e4m3, G>(out, input, masked_ptr, max_m, rows, cols,  \
                                                       input_row_stride, input_expert_stride,      \
                                                       stream);                                    \
          }                                                                                        \
        } else if (out.q_dtype == DType::kInt8) {                                                  \
          return detail::set_error(Status::kUnsupportedDtype,                                      \
                                   "quantize_activation: per-group INT8 is not compiled");        \
        } else {                                                                                   \
          return detail::set_error(Status::kUnsupportedDtype,                                      \
                                   "quantize_activation: per-group payload %s",                    \
                                   dtype_name(out.q_dtype));                                       \
        }                                                                                          \
      } while (0)
    if (out.group_size == 32) { KOPS_GQ(32, T); }
    else if (out.group_size == 64) { KOPS_GQ(64, T); }
    else { KOPS_GQ(128, T); }
    #undef KOPS_GQ
    (void)scale_stride;
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("quantize_activation per-group", err);
    return Status::kSuccess;
  }

  if (out.mode == ActQuantMode::kMXFP8 || out.mode == ActQuantMode::kMXFP4) {
    if (out.scales == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "quantize_activation: MX needs scales");
    }
    if (cols % kMxBlock != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "quantize_activation: MX needs cols %% 32");
    }
    const int64_t groups = cols / kMxBlock;
    const int64_t tasks = rows * groups;
    const int grid = static_cast<int>(
        std::min<int64_t>(ceil_div(tasks, 8), std::max<int64_t>(rows, 1)));
    const bool sw = out.scale_layout == ScaleLayout::kSwizzled128x4;
    #define KOPS_MX(T)                                                                             \
      do {                                                                                         \
        if (out.mode == ActQuantMode::kMXFP8) {                                                    \
          if (sw) {                                                                                \
            quantize_mx_kernel<T, __nv_fp8_e4m3, ScaleLayout::kSwizzled128x4, false>               \
                <<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,  \
                                            static_cast<uint8_t*>(out.scales),                      \
                                            const_ptr<T>(input), input_row_stride,                  \
                                            static_cast<int>(rows), static_cast<int>(cols));       \
          } else {                                                                                 \
            quantize_mx_kernel<T, __nv_fp8_e4m3, ScaleLayout::kRowMajor, false>                    \
                <<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3*>(out.q), q_row_stride,  \
                                            static_cast<uint8_t*>(out.scales),                      \
                                            const_ptr<T>(input), input_row_stride,                  \
                                            static_cast<int>(rows), static_cast<int>(cols));       \
          }                                                                                        \
        } else {                                                                                   \
          if (sw) {                                                                                \
            quantize_mx_kernel<T, __nv_fp4x2_e2m1, ScaleLayout::kSwizzled128x4, false>             \
                <<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_fp4x2_e2m1*>(out.q), q_row_stride,\
                                            static_cast<uint8_t*>(out.scales),                      \
                                            const_ptr<T>(input), input_row_stride,                  \
                                            static_cast<int>(rows), static_cast<int>(cols));       \
          } else {                                                                                 \
            quantize_mx_kernel<T, __nv_fp4x2_e2m1, ScaleLayout::kRowMajor, false>                  \
                <<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_fp4x2_e2m1*>(out.q), q_row_stride,\
                                            static_cast<uint8_t*>(out.scales),                      \
                                            const_ptr<T>(input), input_row_stride,                  \
                                            static_cast<int>(rows), static_cast<int>(cols));       \
          }                                                                                        \
        }                                                                                          \
        return Status::kSuccess;                                                                   \
      } while (0)
    if (input.dtype == DType::kBFloat16) { KOPS_MX(__nv_bfloat16); }
    if (input.dtype == DType::kFloat16) { KOPS_MX(__half); }
    if (input.dtype == DType::kFloat32) { KOPS_MX(float); }
    #undef KOPS_MX
    return detail::set_error(Status::kUnsupportedDtype, "quantize_activation: MX input dtype");
  }

  if (out.mode == ActQuantMode::kNVFP4) {
    if (out.scales == nullptr || out.global_scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "quantize_activation: NVFP4 needs scales and global_scale");
    }
    if (cols % kNvFp4Block != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "quantize_activation: NVFP4 needs cols %% 16");
    }
    const int64_t groups = cols / kNvFp4Block;
    const int64_t tasks = rows * groups;
    const int grid = static_cast<int>(
        std::min<int64_t>(ceil_div(tasks, 8), std::max<int64_t>(rows, 1)));
    #define KOPS_NV(T)                                                                             \
      do {                                                                                         \
        quantize_nvfp4_kernel<T, ScaleLayout::kSwizzled128x4><<<grid, 256, 0, stream>>>(           \
            reinterpret_cast<uint8_t*>(out.q), q_row_stride,                                      \
            reinterpret_cast<f8e4m3_t*>(out.scales), const_ptr<T>(input), input_row_stride,        \
            out.global_scale, static_cast<int>(rows), static_cast<int>(cols));                     \
        return Status::kSuccess;                                                                   \
      } while (0)
    if (input.dtype == DType::kBFloat16) { KOPS_NV(__nv_bfloat16); }
    if (input.dtype == DType::kFloat16) { KOPS_NV(__half); }
    if (input.dtype == DType::kFloat32) { KOPS_NV(float); }
    #undef KOPS_NV
    return detail::set_error(Status::kUnsupportedDtype, "quantize_activation: NVFP4 input dtype");
  }

  return detail::set_error(Status::kUnsupportedShape, "quantize_activation: mode %d",
                           static_cast<int>(out.mode));
}

Status dequantize(const Tensor& out, const Tensor& q, const Tensor& scales,
                  const Tensor& zero_points, const QuantScheme& scheme, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(out.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "dequantize: out dtype %s",
                             dtype_name(out.dtype));
  }
  s = check_tensor(out, out.dtype, -1, 1, "dequantize.out");
  if (s != Status::kSuccess) return s;
  if (out.ndim != 2 || out.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "dequantize: out must be contiguous rows");
  }
  s = check_tensor(q, q.dtype, -1, 1, "dequantize.q");
  if (s != Status::kSuccess) return s;
  const int64_t rows = out.shape[0];
  const int64_t cols = out.shape[1];
  if (rows == 0 || cols == 0) return Status::kSuccess;
  if (scales.data == nullptr && scheme.granularity != ScaleGranularity::kNone) {
    return detail::set_error(Status::kInvalidArgument, "dequantize: scales are null");
  }
  const int64_t q_row_stride =
      is_subbyte_dtype(q.dtype) ? q.stride[0] / 2 : q.stride[0];
  const int64_t total = rows * cols;
  const int grid = static_cast<int>(std::min<int64_t>(ceil_div(total, 256), INT32_MAX));
  const cudaStream_t stream = stream_of(ctx);
  if (scheme.storage_dtype == DType::kFloat8E4M3 &&
      scheme.granularity == ScaleGranularity::kPerToken && zero_points.data == nullptr &&
      (cols % 4) == 0 && (q_row_stride % 4) == 0) {
    const int gridv = static_cast<int>(
        std::min<int64_t>(ceil_div(rows * (cols / 4), 256), INT32_MAX));
    if (out.dtype == DType::kFloat32) {
      dequant_fp8_per_token_vec_kernel<float><<<gridv, 256, 0, stream>>>(
          mut_ptr<float>(out), out.stride[0], const_ptr<uint8_t>(q), q_row_stride,
          const_ptr<float>(scales), rows, cols);
    } else if (out.dtype == DType::kFloat16) {
      dequant_fp8_per_token_vec_kernel<__half><<<gridv, 256, 0, stream>>>(
          mut_ptr<__half>(out), out.stride[0], const_ptr<uint8_t>(q), q_row_stride,
          const_ptr<float>(scales), rows, cols);
    } else {
      dequant_fp8_per_token_vec_kernel<__nv_bfloat16><<<gridv, 256, 0, stream>>>(
          mut_ptr<__nv_bfloat16>(out), out.stride[0], const_ptr<uint8_t>(q), q_row_stride,
          const_ptr<float>(scales), rows, cols);
    }
    const cudaError_t verr = cudaGetLastError();
    if (verr != cudaSuccess) return launch_error("dequantize per-token vector", verr);
    return Status::kSuccess;
  }
#define KOPS_DEQ(OUT, G)                                                                            \
  do {                                                                                              \
    dequantize_kernel<OUT, G><<<grid, 256, 0, stream>>>(                                            \
        mut_ptr<OUT>(out), out.stride[0], const_ptr<uint8_t>(q), q_row_stride,                       \
        scales.data != nullptr ? static_cast<const void*>(const_ptr<uint8_t>(scales)) : nullptr,      \
        zero_points.data != nullptr ? static_cast<const void*>(const_ptr<int8_t>(zero_points))       \
                                    : nullptr,                                                       \
        scheme, rows, cols);                                                                        \
    const cudaError_t err = cudaGetLastError();                                                     \
    if (err != cudaSuccess) return launch_error("dequantize", err);                                 \
    return Status::kSuccess;                                                                        \
  } while (0)
#define KOPS_DEQ_G(OUT)                                                                             \
  do {                                                                                              \
    switch (scheme.granularity) {                                                                   \
      case ScaleGranularity::kPerTensor: KOPS_DEQ(OUT, ScaleGranularity::kPerTensor);               \
      case ScaleGranularity::kPerChannel: KOPS_DEQ(OUT, ScaleGranularity::kPerChannel);             \
      case ScaleGranularity::kPerToken: KOPS_DEQ(OUT, ScaleGranularity::kPerToken);                 \
      case ScaleGranularity::kPerGroup: KOPS_DEQ(OUT, ScaleGranularity::kPerGroup);                 \
      case ScaleGranularity::kPerBlock2D: KOPS_DEQ(OUT, ScaleGranularity::kPerBlock2D);             \
      case ScaleGranularity::kMX: KOPS_DEQ(OUT, ScaleGranularity::kMX);                             \
      default: return detail::set_error(Status::kUnsupportedShape,                                   \
                                        "dequantize: granularity %d",                               \
                                        static_cast<int>(scheme.granularity));                      \
    }                                                                                              \
  } while (0)
  if (out.dtype == DType::kFloat32) KOPS_DEQ_G(float);
  if (out.dtype == DType::kFloat16) KOPS_DEQ_G(__half);
  if (out.dtype == DType::kBFloat16) KOPS_DEQ_G(__nv_bfloat16);
#undef KOPS_DEQ_G
#undef KOPS_DEQ
  return detail::set_error(Status::kUnsupportedDtype, "dequantize: out dtype %s",
                           dtype_name(out.dtype));
}

Status pack_int4(const Tensor& out, const Tensor& input, bool is_unsigned,
                 const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (input.dtype != DType::kInt8 && input.dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "pack_int4: input must be int8/uint8");
  }
  if (out.dtype != DType::kUInt8 && out.dtype != DType::kInt4x2) {
    return detail::set_error(Status::kUnsupportedDtype, "pack_int4: out must be uint8/int4x2");
  }
  s = check_tensor(input, input.dtype, 2, 1, "pack_int4.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, out.dtype, 2, 1, "pack_int4.out");
  if (s != Status::kSuccess) return s;
  if (input.stride[1] != 1 || out.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "pack_int4: last dim must be contiguous");
  }
  if (input.shape[1] % 2 != 0 || out.shape[1] != input.shape[1] / 2 ||
      out.shape[0] != input.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "pack_int4: shapes [rows, cols] -> [rows, cols/2]");
  }
  const int64_t total = out.shape[0] * out.shape[1];
  if (total == 0) return Status::kSuccess;
  const int grid = static_cast<int>(std::min<int64_t>(ceil_div(total, 256), INT32_MAX));
  pack_int4_kernel<<<grid, 256, 0, stream_of(ctx)>>>(
      reinterpret_cast<uint8_t*>(mut_ptr<uint8_t>(out)), const_ptr<int8_t>(input), input.shape[0],
      input.shape[1], is_unsigned);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("pack_int4", err);
  return Status::kSuccess;
}

Status unpack_int4(const Tensor& out, const Tensor& input, bool is_unsigned,
                   const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (out.dtype != DType::kInt8 && out.dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "unpack_int4: out must be int8/uint8");
  }
  if (input.dtype != DType::kUInt8 && input.dtype != DType::kInt4x2) {
    return detail::set_error(Status::kUnsupportedDtype, "unpack_int4: input must be uint8/int4x2");
  }
  s = check_tensor(input, input.dtype, 2, 1, "unpack_int4.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, out.dtype, 2, 1, "unpack_int4.out");
  if (s != Status::kSuccess) return s;
  if (input.stride[1] != 1 || out.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "unpack_int4: last dim must be contiguous");
  }
  if (out.shape[1] != input.shape[1] * 2 || out.shape[0] != input.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "unpack_int4: shapes [rows, cols] -> [rows, 2*cols]");
  }
  const int64_t total = out.shape[0] * out.shape[1];
  if (total == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  if (out.shape[1] % 8 == 0 && input.shape[1] % 4 == 0) {
    const int64_t bytes = out.shape[0] * (out.shape[1] / 2);
    const int grid = static_cast<int>(std::min<int64_t>(ceil_div(bytes / 4, 256), INT32_MAX));
    unpack_int4_vec_kernel<<<grid, 256, 0, stream>>>(
        mut_ptr<int8_t>(out), const_ptr<uint8_t>(input), out.shape[0], out.shape[1], is_unsigned);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return launch_error("unpack_int4 vector", err);
    return Status::kSuccess;
  }
  const int grid = static_cast<int>(std::min<int64_t>(ceil_div(total, 256), INT32_MAX));
  unpack_int4_kernel<<<grid, 256, 0, stream>>>(
      mut_ptr<int8_t>(out), const_ptr<uint8_t>(input), out.shape[0], out.shape[1], is_unsigned);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("unpack_int4", err);
  return Status::kSuccess;
}

Status hadamard_transform(const Tensor& out, const Tensor& input, float scale, bool signed_diag,
                          uint32_t seed, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(input.dtype) || out.dtype != input.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "hadamard_transform: dtype");
  }
  s = check_tensor(input, input.dtype, 2, 1, "hadamard_transform.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, out.dtype, 2, 1, "hadamard_transform.out");
  if (s != Status::kSuccess) return s;
  if (input.stride[1] != 1 || out.stride[1] != 1 || input.shape[0] != out.shape[0] ||
      input.shape[1] != out.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "hadamard_transform: contiguous equal shapes");
  }
  const int64_t n = input.shape[1];
  int log_n = -1;
  for (int b = 1; b <= 15; ++b) {
    if ((int64_t(1) << b) == n) {
      log_n = b;
      break;
    }
  }
  if (log_n < 1) {
    return detail::set_error(Status::kUnsupportedShape,
                             "hadamard_transform: n %lld is not 2^k (k <= 15)",
                             static_cast<long long>(n));
  }
  const int64_t rows = input.shape[0];
  if (rows == 0) return Status::kSuccess;
  const size_t smem = static_cast<size_t>(n) * sizeof(float);
  const cudaStream_t stream = stream_of(ctx);
  const unsigned grid = static_cast<unsigned>(rows);
#define KOPS_HAD_ONE(T, L, S)                                                                       \
  do {                                                                                              \
    if (smem > 48 * 1024) {                                                                         \
      const cudaError_t aerr = cudaFuncSetAttribute(                                                 \
          reinterpret_cast<const void*>(&hadamard_transform_kernel<T, L, S>),                        \
          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));                     \
      if (aerr != cudaSuccess) return launch_error("hadamard_transform attr", aerr);                 \
    }                                                                                               \
    hadamard_transform_kernel<T, L, S><<<grid, 256, smem, stream>>>(                                 \
        mut_ptr<T>(out), out.stride[0], const_ptr<T>(input), input.stride[0], scale, seed,           \
        static_cast<int>(rows));                                                                    \
    const cudaError_t err = cudaGetLastError();                                                      \
    if (err != cudaSuccess) return launch_error("hadamard_transform", err);                          \
    return Status::kSuccess;                                                                         \
  } while (0)
#define KOPS_HAD_LOG(T, L)                                                                          \
  do {                                                                                              \
    if (signed_diag) { KOPS_HAD_ONE(T, L, true); } else { KOPS_HAD_ONE(T, L, false); }               \
  } while (0)
#define KOPS_HAD(T)                                                                                 \
  do {                                                                                              \
    switch (log_n) {                                                                                \
      case 1: KOPS_HAD_LOG(T, 1);                                                                   \
      case 2: KOPS_HAD_LOG(T, 2);                                                                   \
      case 3: KOPS_HAD_LOG(T, 3);                                                                   \
      case 4: KOPS_HAD_LOG(T, 4);                                                                   \
      case 5: KOPS_HAD_LOG(T, 5);                                                                   \
      case 6: KOPS_HAD_LOG(T, 6);                                                                   \
      case 7: KOPS_HAD_LOG(T, 7);                                                                   \
      case 8: KOPS_HAD_LOG(T, 8);                                                                   \
      case 9: KOPS_HAD_LOG(T, 9);                                                                   \
      case 10: KOPS_HAD_LOG(T, 10);                                                                 \
      case 11: KOPS_HAD_LOG(T, 11);                                                                 \
      case 12: KOPS_HAD_LOG(T, 12);                                                                 \
      case 13: KOPS_HAD_LOG(T, 13);                                                                 \
      case 14: KOPS_HAD_LOG(T, 14);                                                                 \
      case 15: KOPS_HAD_LOG(T, 15);                                                                 \
    }                                                                                               \
  } while (0)
  if (input.dtype == DType::kFloat32) KOPS_HAD(float);
  if (input.dtype == DType::kFloat16) KOPS_HAD(__half);
  if (input.dtype == DType::kBFloat16) KOPS_HAD(__nv_bfloat16);
#undef KOPS_HAD
#undef KOPS_HAD_LOG
#undef KOPS_HAD_ONE
  return detail::set_error(Status::kUnsupportedDtype, "hadamard_transform: dtype");
}

}  // namespace kops::quant
