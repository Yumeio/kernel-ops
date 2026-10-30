// =============================================================================
// kernel-ops | csrc/norm/norm.cu
// N01/N02 first implementations: rms_norm, fused_add_rms_norm(_out).
// One CTA per row, fp32 reductions, one rounding at the store. Vector width is
// 16 B (8 halves / 4 floats); a scalar tail handles hidden % VEC.
//
// Target support: this body is arch-generic (no cp.async, no TMA) and is meant
// for Tier A/B/C alike; exact-target selection happens in the registry.
// =============================================================================
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"

namespace kops::norm {
namespace {

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T x[N];
};

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ void from_f(T& d, float s);
template <>
__device__ __forceinline__ void from_f<float>(float& d, float s) { d = s; }
template <>
__device__ __forceinline__ void from_f<__half>(__half& d, float s) { d = __float2half_rn(s); }
template <>
__device__ __forceinline__ void from_f<__nv_bfloat16>(__nv_bfloat16& d, float s) {
  d = __float2bfloat16_rn(s);
}

/// Sum across the whole block; every thread receives the result.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

/// Row offset for a view whose only non-contiguous dims are -2, -3, -4.
template <int NUM_DIMS>
__device__ __forceinline__ int64_t row_offset(int64_t row, int64_t s2, int64_t s3, int64_t s4,
                                              int64_t n2, int64_t n3) {
  if (NUM_DIMS <= 2) return row * s2;
  const int64_t i2 = row % n2;
  const int64_t t = row / n2;
  if (NUM_DIMS == 3) return t * s3 + i2 * s2;
  const int64_t i3 = t % n3;
  const int64_t i4 = t / n3;
  return i4 * s4 + i3 * s3 + i2 * s2;
}

template <typename T, int VEC, int NUM_DIMS, bool HasWeight>
__device__ __forceinline__ void rms_row(const T* __restrict__ x, T* __restrict__ y,
                                        const T* __restrict__ w, int num_weight_groups,
                                        float weight_offset, float eps, int hidden,
                                        int64_t weight_stride, int64_t row, float* smem) {
  float ss = 0.0f;
  const int nvec = hidden / VEC;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    Pack<T, VEC> p;
    p = *reinterpret_cast<const Pack<T, VEC>*>(x + i * VEC);
    for (int k = 0; k < VEC; ++k) {
      const float v = to_f<T>(p.x[k]);
      ss += v * v;
    }
  }
  for (int i = nvec * VEC + threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = to_f<T>(x[i]);
    ss += v * v;
  }
  ss = block_sum(ss, smem);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  const T* wp = w;
  if (HasWeight && num_weight_groups > 1) {
    wp = w + static_cast<int64_t>(row % num_weight_groups) * weight_stride;
  }
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    Pack<T, VEC> p;
    p = *reinterpret_cast<const Pack<T, VEC>*>(x + i * VEC);
    Pack<T, VEC> q;
    for (int k = 0; k < VEC; ++k) {
      const float s = to_f<T>(p.x[k]) * rstd;
      const float wv = HasWeight ? (to_f<T>(wp[i * VEC + k]) + weight_offset) : 1.0f;
      from_f<T>(q.x[k], s * wv);
    }
    *reinterpret_cast<Pack<T, VEC>*>(y + i * VEC) = q;
  }
  for (int i = nvec * VEC + threadIdx.x; i < hidden; i += blockDim.x) {
    const float s = to_f<T>(x[i]) * rstd;
    const float wv = HasWeight ? (to_f<T>(wp[i]) + weight_offset) : 1.0f;
    from_f<T>(y[i], s * wv);
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// N01 kernels
// -----------------------------------------------------------------------------

template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                                const scalar_t* __restrict__ input,
                                const int64_t input_stride_d2, const int64_t input_stride_d3,
                                const int64_t input_stride_d4, const int64_t input_shape_d2,
                                const int64_t input_shape_d3, const scalar_t* __restrict__ weight,
                                const int64_t weight_stride, const int num_weight_groups,
                                const float weight_offset, const float epsilon,
                                const int num_tokens, const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  const int64_t off = row_offset<NUM_DIMS>(row, input_stride_d2, input_stride_d3, input_stride_d4,
                                           input_shape_d2, input_shape_d3);
  rms_row<scalar_t, VEC_SIZE, NUM_DIMS, HasWeight>(
      input + off, out + row * out_stride, weight, num_weight_groups, weight_offset, epsilon,
      hidden_size, weight_stride, row, smem);
}

template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__global__ void fused_add_rms_norm_kernel(scalar_t* __restrict__ input, const int64_t input_stride,
                                          residual_t* __restrict__ residual,
                                          const scalar_t* __restrict__ weight,
                                          const float weight_offset, const float input_scale,
                                          const float epsilon, const int num_tokens,
                                          const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  scalar_t* x = input + row * input_stride;
  residual_t* r = residual + row * input_stride;
  float ss = 0.0f;
  const int nvec = hidden_size / VEC_SIZE;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    Pack<scalar_t, VEC_SIZE> p;
    Pack<residual_t, VEC_SIZE> q;
    p = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + i * VEC_SIZE);
    q = *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE);
    for (int k = 0; k < VEC_SIZE; ++k) {
      float rv = to_f<residual_t>(q.x[k]);
      rv += input_scale * to_f<scalar_t>(p.x[k]);
      from_f<residual_t>(q.x[k], rv);
      ss += rv * rv;
    }
    *reinterpret_cast<Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE) = q;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
    float rv = to_f<residual_t>(r[i]);
    rv += input_scale * to_f<scalar_t>(x[i]);
    from_f<residual_t>(r[i], rv);
    ss += rv * rv;
  }
  ss = block_sum(ss, smem);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden_size) + epsilon);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    Pack<scalar_t, VEC_SIZE> p;
    Pack<residual_t, VEC_SIZE> q;
    q = *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE);
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float rv = to_f<residual_t>(q.x[k]);
      const float wv = HasWeight ? (to_f<scalar_t>(weight[i * VEC_SIZE + k]) + weight_offset) : 1.0f;
      from_f<scalar_t>(p.x[k], rv * rstd * wv);
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(x + i * VEC_SIZE) = p;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float rv = to_f<residual_t>(r[i]);
    const float wv = HasWeight ? (to_f<scalar_t>(weight[i]) + weight_offset) : 1.0f;
    from_f<scalar_t>(x[i], rv * rstd * wv);
  }
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------
namespace {

template <typename T>
constexpr int vec_width() {
  return 8;  // 16 B for 2-byte types
}
template <>
constexpr int vec_width<float>() {
  return 4;
}

inline int block_for(int hidden, int vec) {
  int n = (hidden + vec - 1) / vec;
  n = ((n + 31) / 32) * 32;
  if (n < 64) n = 64;
  if (n > 256) n = 256;
  return n;
}

struct RowArgs {
  int num_tokens;
  int hidden;
  int64_t s2, s3, s4, n2, n3;
};

template <typename T, int VEC, bool HasWeight>
Status launch_rms(const Tensor& out, const Tensor& input, const Tensor& weight, int num_groups,
                  int64_t weight_stride, float eps, float w_off, const RowArgs& a,
                  const LaunchContext& ctx) {
  const T* x = reinterpret_cast<const T*>(static_cast<const char*>(input.data) + input.byte_offset);
  T* y = reinterpret_cast<T*>(static_cast<char*>(out.data) + out.byte_offset);
  const T* w = HasWeight ? static_cast<const T*>(weight.data) : nullptr;
  const int64_t out_stride = out.stride[out.ndim - 2];
  const int block = block_for(a.hidden, VEC);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_LAUNCH_RMS(ND)                                                                       \
  rms_norm_kernel<T, VEC, ND, HasWeight><<<a.num_tokens, block, 0, stream>>>(                     \
      y, out_stride, x, a.s2, a.s3, a.s4, a.n2, a.n3, w, weight_stride, num_groups, w_off, eps,   \
      a.num_tokens, a.hidden)
  switch (input.ndim) {
    case 2: KOPS_LAUNCH_RMS(2); break;
    case 3: KOPS_LAUNCH_RMS(3); break;
    case 4: KOPS_LAUNCH_RMS(4); break;
    default: return detail::set_error(Status::kUnsupportedShape, "rms_norm: ndim %d", input.ndim);
  }
#undef KOPS_LAUNCH_RMS
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "rms_norm launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T>
Status dispatch_rms(const Tensor& out, const Tensor& input, const Tensor& weight, int num_groups,
                    int64_t weight_stride, float eps, float w_off, const RowArgs& a,
                    const LaunchContext& ctx) {
  if (tensor_present(weight)) return launch_rms<T, vec_width<T>(), true>(out, input, weight, num_groups, weight_stride, eps, w_off, a, ctx);
  return launch_rms<T, vec_width<T>(), false>(out, input, weight, num_groups, weight_stride, eps, w_off, a, ctx);
}

}  // namespace

Status rms_norm(const Tensor& out, const Tensor& input, const Tensor& weight, float epsilon,
                float weight_offset, const LaunchContext& ctx) {
  Status s = check_tensor(input, input.dtype, -1, 16, "rms_norm.input");
  if (s != Status::kSuccess) return s;
  if (input.ndim < 2 || input.ndim > 4) {
    return detail::set_error(Status::kUnsupportedShape, "rms_norm: ndim %d not in {2,3,4}", input.ndim);
  }
  s = check_tensor(out, input.dtype, input.ndim, 16, "rms_norm.out");
  if (s != Status::kSuccess) return s;
  for (int d = 0; d + 1 < input.ndim; ++d) {
    if (out.shape[d] != input.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "rms_norm: shape mismatch at dim %d", d);
    }
  }
  if (out.shape[input.ndim - 1] != input.shape[input.ndim - 1]) {
    return detail::set_error(Status::kInvalidArgument, "rms_norm: hidden size mismatch");
  }
  const int hidden = static_cast<int>(input.shape[input.ndim - 1]);
  if (hidden <= 0) return detail::set_error(Status::kUnsupportedShape, "rms_norm: hidden <= 0");
  const int64_t numel = tensor_numel(input);
  const int num_tokens = static_cast<int>(numel / hidden);
  int num_groups = 1;
  int64_t weight_stride = 0;
  if (tensor_present(weight)) {
    s = check_tensor(weight, input.dtype, -1, 16, "rms_norm.weight");
    if (s != Status::kSuccess) return s;
    if (weight.ndim == 2) {
      num_groups = static_cast<int>(weight.shape[0]);
      weight_stride = weight.stride[0];
    } else if (weight.ndim != 1) {
      return detail::set_error(Status::kUnsupportedShape, "rms_norm: weight ndim %d", weight.ndim);
    }
    if (weight.shape[weight.ndim - 1] != hidden) {
      return detail::set_error(Status::kInvalidArgument, "rms_norm: weight width mismatch");
    }
  }
  RowArgs a{num_tokens, hidden, input.stride[input.ndim - 2], 0, 0, 0, 0};
  if (input.ndim >= 3) a.s3 = input.stride[input.ndim - 3];
  if (input.ndim >= 4) a.s4 = input.stride[input.ndim - 4];
  if (input.ndim >= 3) a.n2 = input.shape[input.ndim - 2];
  if (input.ndim >= 4) a.n3 = input.shape[input.ndim - 3];

  switch (input.dtype) {
    case DType::kBFloat16:
      return dispatch_rms<__nv_bfloat16>(out, input, weight, num_groups, weight_stride, epsilon,
                                         weight_offset, a, ctx);
    case DType::kFloat16:
      return dispatch_rms<__half>(out, input, weight, num_groups, weight_stride, epsilon,
                                  weight_offset, a, ctx);
    case DType::kFloat32:
      return dispatch_rms<float>(out, input, weight, num_groups, weight_stride, epsilon,
                                 weight_offset, a, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "rms_norm: dtype %s",
                               dtype_name(input.dtype));
  }
}

Status fused_add_rms_norm(const Tensor& input_inout, const Tensor& residual_inout,
                          const Tensor& weight, float epsilon, float weight_offset,
                          float input_scale, const LaunchContext& ctx) {
  Status s = check_tensor(input_inout, input_inout.dtype, 2, 16, "fused_add_rms_norm.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(residual_inout, residual_inout.dtype, 2, 16, "fused_add_rms_norm.residual");
  if (s != Status::kSuccess) return s;
  if (input_inout.shape[0] != residual_inout.shape[0] ||
      input_inout.shape[1] != residual_inout.shape[1]) {
    return detail::set_error(Status::kInvalidArgument, "fused_add_rms_norm: shape mismatch");
  }
  if (residual_inout.stride[0] != input_inout.stride[0]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "fused_add_rms_norm: residual row stride %lld != input stride %lld",
                             static_cast<long long>(residual_inout.stride[0]),
                             static_cast<long long>(input_inout.stride[0]));
  }
  const int hidden = static_cast<int>(input_inout.shape[1]);
  const int tokens = static_cast<int>(input_inout.shape[0]);
  const bool has_w = tensor_present(weight);
  if (has_w) {
    s = check_tensor(weight, input_inout.dtype, 1, 16, "fused_add_rms_norm.weight");
    if (s != Status::kSuccess) return s;
  }
  const int block = block_for(hidden, 8);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);

#define KOPS_LAUNCH_ADD(T, R, HW)                                                                 \
  fused_add_rms_norm_kernel<T, R, 8, HW><<<tokens, block, 0, stream>>>(                           \
      static_cast<T*>(input_inout.data), input_inout.stride[0],                                   \
      static_cast<R*>(residual_inout.data), has_w ? static_cast<const T*>(weight.data) : nullptr,  \
      weight_offset, input_scale, epsilon, tokens, hidden)

  if (input_inout.dtype == DType::kBFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) KOPS_LAUNCH_ADD(__nv_bfloat16, float, true);
      else KOPS_LAUNCH_ADD(__nv_bfloat16, float, false);
    } else if (residual_inout.dtype == DType::kBFloat16) {
      if (has_w) KOPS_LAUNCH_ADD(__nv_bfloat16, __nv_bfloat16, true);
      else KOPS_LAUNCH_ADD(__nv_bfloat16, __nv_bfloat16, false);
    } else {
      return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: residual %s",
                               dtype_name(residual_inout.dtype));
    }
  } else if (input_inout.dtype == DType::kFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) KOPS_LAUNCH_ADD(__half, float, true);
      else KOPS_LAUNCH_ADD(__half, float, false);
    } else if (residual_inout.dtype == DType::kFloat16) {
      if (has_w) KOPS_LAUNCH_ADD(__half, __half, true);
      else KOPS_LAUNCH_ADD(__half, __half, false);
    } else {
      return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: residual %s",
                               dtype_name(residual_inout.dtype));
    }
  } else if (input_inout.dtype == DType::kFloat32) {
    if (has_w) KOPS_LAUNCH_ADD(float, float, true);
    else KOPS_LAUNCH_ADD(float, float, false);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: input %s",
                             dtype_name(input_inout.dtype));
  }
#undef KOPS_LAUNCH_ADD

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "fused_add_rms_norm launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

Status fused_add_rms_norm_out(const Tensor& out, const Tensor& residual_out, const Tensor& input,
                              const Tensor& residual_in, const Tensor& weight, float epsilon,
                              float weight_offset, float input_scale, const LaunchContext& ctx) {
  (void)out;
  (void)residual_out;
  (void)input;
  (void)residual_in;
  (void)weight;
  (void)epsilon;
  (void)weight_offset;
  (void)input_scale;
  (void)ctx;
  return detail::set_error(Status::kNotImplemented, "fused_add_rms_norm_out: M1.1");
}

Status layer_norm(const Tensor& out, const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float epsilon, float weight_offset, const LaunchContext& ctx) {
  (void)out; (void)input; (void)weight; (void)bias; (void)epsilon; (void)weight_offset; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "layer_norm: M1.2");
}

Status fused_add_layer_norm(const Tensor& input_inout, const Tensor& residual_inout,
                            const Tensor& weight, const Tensor& bias, float epsilon,
                            float weight_offset, float input_scale, const LaunchContext& ctx) {
  (void)input_inout; (void)residual_inout; (void)weight; (void)bias; (void)epsilon;
  (void)weight_offset; (void)input_scale; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "fused_add_layer_norm: M1.2");
}

Status qk_norm(const Tensor& qkv_or_q, const Tensor& k, const Tensor& q_weight,
               const Tensor& k_weight, int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
               NormType norm, bool full_width, const Tensor& full_width_sq_partials, float epsilon,
               float weight_offset, const LaunchContext& ctx) {
  (void)qkv_or_q; (void)k; (void)q_weight; (void)k_weight; (void)num_q_heads; (void)num_kv_heads;
  (void)head_dim; (void)norm; (void)full_width; (void)full_width_sq_partials; (void)epsilon;
  (void)weight_offset; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "qk_norm: M1.2");
}

Status l2_norm(const Tensor& out, const Tensor& input, float scale, float epsilon,
               const LaunchContext& ctx) {
  (void)out; (void)input; (void)scale; (void)epsilon; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "l2_norm: M1.2");
}

Status gated_rms_norm(const Tensor& out, const Tensor& x, const Tensor& z, const Tensor& weight,
                      int32_t group_size, ActFn gate_act, bool norm_before_gate, float epsilon,
                      float weight_offset, const LaunchContext& ctx) {
  (void)out; (void)x; (void)z; (void)weight; (void)group_size; (void)gate_act;
  (void)norm_before_gate; (void)epsilon; (void)weight_offset; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "gated_rms_norm: M1.2");
}

Status norm_quant(const NormQuantParams& params, NormType norm, DType input_dtype,
                  DType residual_dtype, const LaunchContext& ctx) {
  (void)params; (void)norm; (void)input_dtype; (void)residual_dtype; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "norm_quant: M1.2");
}

}  // namespace kops::norm
