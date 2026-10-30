// =============================================================================
// kernel-ops | csrc/activation/activation.cu
// F01 first implementations: act_kernel (elementwise) and act_and_mul_kernel
// (gated linear unit). fp32 math, one rounding at the store.
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
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"

namespace kops::act {
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

template <ActFn kAct>
__device__ __forceinline__ float actf(float x, const GluParams& p) {
  switch (kAct) {
    case ActFn::kIdentity: return x;
    case ActFn::kSiLU: return x / (1.0f + __expf(-p.gate_alpha * x));
    case ActFn::kGeLUErf: return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    case ActFn::kGeLUTanh:
      return 0.5f * x *
             (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case ActFn::kQuickGeLU: return x / (1.0f + __expf(-1.702f * x));
    case ActFn::kReLU: return x > 0.0f ? x : 0.0f;
    case ActFn::kReLU2: return x > 0.0f ? x * x : 0.0f;
    case ActFn::kSigmoid: return 1.0f / (1.0f + __expf(-x));
    case ActFn::kTanh: return tanhf(x);
    case ActFn::kSoftplus: return x > 20.0f ? x : log1pf(__expf(x));
    case ActFn::kSiTU:
      return p.situ_beta * tanhf(x / p.situ_beta) * (1.0f / (1.0f + __expf(-x)));
    case ActFn::kFatReLU: return x > p.fatrelu_threshold ? x : 0.0f;
  }
  return x;
}

/// out = act(gate) * up with the documented clamps / offsets / SiTU soft-cap.
template <ActFn kAct>
__device__ __forceinline__ float glu(float gate, float up, const GluParams& p, float row_w) {
  if (p.gate_clamp_max > 0.0f && gate > p.gate_clamp_max) gate = p.gate_clamp_max;
  if (p.up_clamp > 0.0f) {
    up = up > p.up_clamp ? p.up_clamp : up;
    up = up < -p.up_clamp ? -p.up_clamp : up;
  }
  up += p.up_offset;
  if (p.situ_linear_beta > 0.0f) up = p.situ_linear_beta * tanhf(up / p.situ_linear_beta);
  return row_w * actf<kAct>(gate, p) * up;
}

}  // namespace

template <typename scalar_t, ActFn kAct, int VEC_SIZE>
__global__ void act_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                           const scalar_t* __restrict__ input, const int64_t input_stride,
                           const GluParams params, const int num_tokens, const int d) {
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  const scalar_t* x = input + row * input_stride;
  scalar_t* y = out + row * out_stride;
  const int nvec = d / VEC_SIZE;
  const int base = blockIdx.y * blockDim.x * VEC_SIZE;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int idx = base + i * VEC_SIZE;
    if (idx >= nvec * VEC_SIZE) break;
    Pack<scalar_t, VEC_SIZE> p;
    p = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + idx);
    Pack<scalar_t, VEC_SIZE> q;
    for (int k = 0; k < VEC_SIZE; ++k) from_f<scalar_t>(q.x[k], actf<kAct>(to_f<scalar_t>(p.x[k]), params));
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(y + idx) = q;
  }
  if (blockIdx.y == 0) {
    for (int i = nvec * VEC_SIZE + threadIdx.x; i < d; i += blockDim.x) {
      from_f<scalar_t>(y[i], actf<kAct>(to_f<scalar_t>(x[i]), params));
    }
  }
}

template <typename scalar_t, ActFn kAct, GateLayout kLayout, int VEC_SIZE>
__global__ void act_and_mul_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                                   const scalar_t* __restrict__ input, const int64_t input_stride,
                                   const float* __restrict__ row_weight, const GluParams params,
                                   const int num_tokens, const int d) {
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  const scalar_t* x = input + row * input_stride;
  scalar_t* y = out + row * out_stride;
  const float rw = (row_weight != nullptr) ? row_weight[row] : 1.0f;
  const int base = blockIdx.y * blockDim.x * VEC_SIZE;
  if (kLayout == GateLayout::kInterleaved) {
    for (int i = base + threadIdx.x; i < d; i += blockDim.x) {
      if (i >= base + static_cast<int>(blockDim.x)) break;
      const float gate = to_f<scalar_t>(x[2 * i]);
      const float up = to_f<scalar_t>(x[2 * i + 1]);
      from_f<scalar_t>(y[i], glu<kAct>(gate, up, params, rw));
    }
    return;
  }
  const int nvec = d / VEC_SIZE;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int idx = base + i * VEC_SIZE;
    if (idx >= nvec * VEC_SIZE) break;
    Pack<scalar_t, VEC_SIZE> g;
    Pack<scalar_t, VEC_SIZE> u;
    g = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + idx);
    const int other = (kLayout == GateLayout::kGateUp) ? idx + d : idx;
    u = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + other);
    Pack<scalar_t, VEC_SIZE> q;
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float gv = (kLayout == GateLayout::kGateUp) ? to_f<scalar_t>(g.x[k]) : to_f<scalar_t>(u.x[k]);
      const float uv = (kLayout == GateLayout::kGateUp) ? to_f<scalar_t>(u.x[k]) : to_f<scalar_t>(g.x[k]);
      from_f<scalar_t>(q.x[k], glu<kAct>(gv, uv, params, rw));
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(y + idx) = q;
  }
  if (blockIdx.y == 0) {
    for (int i = nvec * VEC_SIZE + threadIdx.x; i < d; i += blockDim.x) {
      const float gate = to_f<scalar_t>(x[i]);
      const float up = to_f<scalar_t>(x[i + d]);
      from_f<scalar_t>(y[i], glu<kAct>(gate, up, params, rw));
    }
  }
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------
namespace {

template <typename T>
constexpr int vec_width() { return 8; }
template <>
constexpr int vec_width<float>() { return 4; }

template <typename T, ActFn A>
Status run_act(const Tensor& out, const Tensor& input, const GluParams& p, int tokens, int d,
               cudaStream_t stream) {
  const int vec = vec_width<T>();
  const int nvec = (d + vec - 1) / vec;
  const int block = nvec < 128 ? ((nvec + 31) / 32) * 32 : 128;
  const int ny = (nvec + block - 1) / block;
  const T* x = reinterpret_cast<const T*>(static_cast<const char*>(input.data) + input.byte_offset);
  T* y = reinterpret_cast<T*>(static_cast<char*>(out.data) + out.byte_offset);
  act_kernel<T, A, vec_width<T>()><<<dim3(tokens, ny), block, 0, stream>>>(
      y, out.stride[0], x, input.stride[0], p, tokens, d);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "activation launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T, ActFn A, GateLayout L>
Status run_glu(const Tensor& out, const Tensor& input, const Tensor& row_weight, const GluParams& p,
               int tokens, int d, cudaStream_t stream) {
  const int vec = vec_width<T>();
  const int nvec = (d + vec - 1) / vec;
  const int block = nvec < 128 ? ((nvec + 31) / 32) * 32 : 128;
  const int ny = (nvec + block - 1) / block;
  const T* x = reinterpret_cast<const T*>(static_cast<const char*>(input.data) + input.byte_offset);
  T* y = reinterpret_cast<T*>(static_cast<char*>(out.data) + out.byte_offset);
  const float* rw = tensor_present(row_weight) ? static_cast<const float*>(row_weight.data) : nullptr;
  act_and_mul_kernel<T, A, L, vec_width<T>()><<<dim3(tokens, ny), block, 0, stream>>>(
      y, out.stride[0], x, input.stride[0], rw, p, tokens, d);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "act_and_mul launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T, ActFn A>
Status run_glu_layout(const Tensor& out, const Tensor& input, const Tensor& row_weight,
                      const GluParams& p, GateLayout layout, int tokens, int d,
                      cudaStream_t stream) {
  switch (layout) {
    case GateLayout::kGateUp: return run_glu<T, A, GateLayout::kGateUp>(out, input, row_weight, p, tokens, d, stream);
    case GateLayout::kUpGate: return run_glu<T, A, GateLayout::kUpGate>(out, input, row_weight, p, tokens, d, stream);
    case GateLayout::kInterleaved:
      return run_glu<T, A, GateLayout::kInterleaved>(out, input, row_weight, p, tokens, d, stream);
  }
  return detail::set_error(Status::kInvalidArgument, "act_and_mul: bad layout");
}

template <typename T>
Status dispatch_glu(const Tensor& out, const Tensor& input, const Tensor& row_weight,
                    const GluParams& p, ActFn act, GateLayout layout, int tokens, int d,
                    cudaStream_t stream) {
#define KOPS_CASE_ACT(NAME) case ActFn::NAME: return run_glu_layout<T, ActFn::NAME>(out, input, row_weight, p, layout, tokens, d, stream)
  switch (act) {
    KOPS_CASE_ACT(kSiLU);
    KOPS_CASE_ACT(kGeLUErf);
    KOPS_CASE_ACT(kGeLUTanh);
    KOPS_CASE_ACT(kReLU2);
    KOPS_CASE_ACT(kIdentity);
    KOPS_CASE_ACT(kSigmoid);
    KOPS_CASE_ACT(kQuickGeLU);
    KOPS_CASE_ACT(kReLU);
    default:
      return detail::set_error(Status::kNotImplemented, "act_and_mul: ActFn %d not compiled yet",
                               static_cast<int>(act));
  }
#undef KOPS_CASE_ACT
}

}  // namespace

Status activation(const Tensor& out, const Tensor& input, ActFn act, const GluParams& params,
                  const LaunchContext& ctx) {
  Status s = check_tensor(input, input.dtype, 2, 16, "activation.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, input.dtype, 2, 16, "activation.out");
  if (s != Status::kSuccess) return s;
  if (out.shape[0] != input.shape[0] || out.shape[1] != input.shape[1]) {
    return detail::set_error(Status::kInvalidArgument, "activation: shape mismatch");
  }
  const int tokens = static_cast<int>(input.shape[0]);
  const int d = static_cast<int>(input.shape[1]);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (input.dtype) {
    case DType::kBFloat16:
      switch (act) {
        case ActFn::kSiLU: return run_act<__nv_bfloat16, ActFn::kSiLU>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUTanh: return run_act<__nv_bfloat16, ActFn::kGeLUTanh>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUErf: return run_act<__nv_bfloat16, ActFn::kGeLUErf>(out, input, params, tokens, d, stream);
        case ActFn::kReLU2: return run_act<__nv_bfloat16, ActFn::kReLU2>(out, input, params, tokens, d, stream);
        case ActFn::kIdentity: return run_act<__nv_bfloat16, ActFn::kIdentity>(out, input, params, tokens, d, stream);
        default: break;
      }
      break;
    case DType::kFloat16:
      switch (act) {
        case ActFn::kSiLU: return run_act<__half, ActFn::kSiLU>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUTanh: return run_act<__half, ActFn::kGeLUTanh>(out, input, params, tokens, d, stream);
        default: break;
      }
      break;
    case DType::kFloat32:
      switch (act) {
        case ActFn::kSiLU: return run_act<float, ActFn::kSiLU>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUTanh: return run_act<float, ActFn::kGeLUTanh>(out, input, params, tokens, d, stream);
        default: break;
      }
      break;
    default: break;
  }
  return detail::set_error(Status::kNotImplemented, "activation: %s x ActFn %d not compiled",
                           dtype_name(input.dtype), static_cast<int>(act));
}

Status act_and_mul(const Tensor& out, const Tensor& input, const Tensor& row_weight, ActFn act,
                   GateLayout layout, const GluParams& params, const LaunchContext& ctx) {
  Status s = check_tensor(input, input.dtype, 2, 16, "act_and_mul.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, input.dtype, 2, 16, "act_and_mul.out");
  if (s != Status::kSuccess) return s;
  const int tokens = static_cast<int>(out.shape[0]);
  const int d = static_cast<int>(out.shape[1]);   // output width; input is 2*d
  if (input.shape[0] != tokens || input.shape[1] < 2 * d) {
    return detail::set_error(Status::kInvalidArgument,
                             "act_and_mul: input [%d, %d] does not match out [%d, %d]",
                             static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                             tokens, d);
  }
  if (tensor_present(row_weight) && tensor_numel(row_weight) < tokens) {
    return detail::set_error(Status::kInvalidArgument, "act_and_mul: row_weight too short");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (input.dtype) {
    case DType::kBFloat16:
      return dispatch_glu<__nv_bfloat16>(out, input, row_weight, params, act, layout, tokens, d, stream);
    case DType::kFloat16:
      return dispatch_glu<__half>(out, input, row_weight, params, act, layout, tokens, d, stream);
    case DType::kFloat32:
      return dispatch_glu<float>(out, input, row_weight, params, act, layout, tokens, d, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "act_and_mul: dtype %s",
                               dtype_name(input.dtype));
  }
}

Status act_and_mul_quant(const ActMulQuantParams& params, ActFn act, GateLayout layout,
                         DType input_dtype, const LaunchContext& ctx) {
  (void)params; (void)act; (void)layout; (void)input_dtype; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "act_and_mul_quant: M2");
}

Status gate_mul(const Tensor& out, const Tensor& x, const Tensor& gate,
                const Tensor& accumulate_into, ActFn gate_act, const LaunchContext& ctx) {
  (void)out; (void)x; (void)gate; (void)accumulate_into; (void)gate_act; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "gate_mul: M2");
}

}  // namespace kops::act
