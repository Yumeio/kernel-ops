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
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < nvec) {
    const int idx = i * VEC_SIZE;
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
  if (kLayout == GateLayout::kInterleaved) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < d) {
      const float gate = to_f<scalar_t>(x[2 * i]);
      const float up = to_f<scalar_t>(x[2 * i + 1]);
      from_f<scalar_t>(y[i], glu<kAct>(gate, up, params, rw));
    }
    return;
  }
  const int nvec = d / VEC_SIZE;
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < nvec) {
    const int idx = i * VEC_SIZE;
    Pack<scalar_t, VEC_SIZE> g;
    Pack<scalar_t, VEC_SIZE> u;
    g = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + idx);
    u = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + idx + d);
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
      const float a = to_f<scalar_t>(x[i]);
      const float b = to_f<scalar_t>(x[i + d]);
      const float gate = (kLayout == GateLayout::kGateUp) ? a : b;
      const float up = (kLayout == GateLayout::kGateUp) ? b : a;
      from_f<scalar_t>(y[i], glu<kAct>(gate, up, params, rw));
    }
  }
}

template <typename scalar_t, ActFn kAct, int VEC_SIZE>
__global__ void gate_mul_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                                const scalar_t* __restrict__ x, const int64_t x_stride,
                                const scalar_t* __restrict__ gate, const int64_t gate_stride,
                                const scalar_t* __restrict__ accumulate_into,
                                const int64_t accumulate_stride, const int num_tokens, const int d,
                                const int gate_width) {
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  scalar_t* dst = out + row * out_stride;
  const scalar_t* src = x + row * x_stride;
  const scalar_t* g = gate + row * gate_stride;
  const scalar_t* acc = accumulate_into ? (accumulate_into + row * accumulate_stride) : nullptr;
  const int group = d / gate_width;
  const int nvec = d / VEC_SIZE;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col < nvec) {
    const int idx = col * VEC_SIZE;
    Pack<scalar_t, VEC_SIZE> px;
    Pack<scalar_t, VEC_SIZE> py;
    px = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(src + idx);
    if (acc != nullptr) py = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(acc + idx);
    Pack<scalar_t, VEC_SIZE> pg;
    if (gate_width == d) pg = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(g + idx);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const int e = idx + j;
      const float gv = (gate_width == d) ? to_f<scalar_t>(pg.x[j])
                                          : to_f<scalar_t>(g[e / group]);
      float v = to_f<scalar_t>(px.x[j]) * actf<kAct>(gv, GluParams{});
      if (acc != nullptr) v += to_f<scalar_t>(py.x[j]);
      from_f<scalar_t>(py.x[j], v);
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(dst + idx) = py;
    return;
  }
  // Scalar tail; only the first y tile performs it.
  if (blockIdx.y == 0) {
    for (int e = nvec * VEC_SIZE + threadIdx.x; e < d; e += blockDim.x) {
      const float gv = to_f<scalar_t>(g[e / group]);
      float v = to_f<scalar_t>(src[e]) * actf<kAct>(gv, GluParams{});
      if (acc != nullptr) v += to_f<scalar_t>(acc[e]);
      from_f<scalar_t>(dst[e], v);
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
    KOPS_CASE_ACT(kTanh);
    KOPS_CASE_ACT(kSoftplus);
    KOPS_CASE_ACT(kSiTU);
    KOPS_CASE_ACT(kFatReLU);
    default:
      return detail::set_error(Status::kNotImplemented, "act_and_mul: ActFn %d not compiled yet",
                               static_cast<int>(act));
  }
#undef KOPS_CASE_ACT
}

template <typename T, ActFn A>
Status run_gate_mul(const Tensor& out, const Tensor& x, const Tensor& gate, const Tensor& accumulate,
                    int tokens, int d, int gate_width, cudaStream_t stream) {
  const int vec = vec_width<T>();
  const int nvec = (d + vec - 1) / vec;
  const int block = nvec < 128 ? ((nvec + 31) / 32) * 32 : 128;
  const int ny = (nvec + block - 1) / block;
  const T* xp = reinterpret_cast<const T*>(static_cast<const char*>(x.data) + x.byte_offset);
  const T* gp = reinterpret_cast<const T*>(static_cast<const char*>(gate.data) + gate.byte_offset);
  const T* ap = tensor_present(accumulate)
                    ? reinterpret_cast<const T*>(static_cast<const char*>(accumulate.data) +
                                                 accumulate.byte_offset)
                    : nullptr;
  T* op = reinterpret_cast<T*>(static_cast<char*>(out.data) + out.byte_offset);
  gate_mul_kernel<T, A, vec_width<T>()><<<dim3(tokens, ny), block, 0, stream>>>(
      op, out.stride[0], xp, x.stride[0], gp, gate.stride[0], ap,
      tensor_present(accumulate) ? accumulate.stride[0] : 0, tokens, d, gate_width);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "gate_mul launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T>
Status dispatch_gate_mul(const Tensor& out, const Tensor& x, const Tensor& gate,
                         const Tensor& accumulate, ActFn act, int tokens, int d, int gate_width,
                         cudaStream_t stream) {
#define KOPS_GATE_CASE(NAME) \
  case ActFn::NAME: return run_gate_mul<T, ActFn::NAME>(out, x, gate, accumulate, tokens, d, gate_width, stream)
  switch (act) {
    KOPS_GATE_CASE(kIdentity);
    KOPS_GATE_CASE(kSiLU);
    KOPS_GATE_CASE(kGeLUErf);
    KOPS_GATE_CASE(kGeLUTanh);
    KOPS_GATE_CASE(kQuickGeLU);
    KOPS_GATE_CASE(kReLU);
    KOPS_GATE_CASE(kReLU2);
    KOPS_GATE_CASE(kSigmoid);
    KOPS_GATE_CASE(kTanh);
    KOPS_GATE_CASE(kSoftplus);
    KOPS_GATE_CASE(kSiTU);
    KOPS_GATE_CASE(kFatReLU);
    default:
      return detail::set_error(Status::kNotImplemented, "gate_mul: ActFn %d", static_cast<int>(act));
  }
#undef KOPS_GATE_CASE
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
        case ActFn::kQuickGeLU: return run_act<__nv_bfloat16, ActFn::kQuickGeLU>(out, input, params, tokens, d, stream);
        case ActFn::kReLU: return run_act<__nv_bfloat16, ActFn::kReLU>(out, input, params, tokens, d, stream);
        case ActFn::kSigmoid: return run_act<__nv_bfloat16, ActFn::kSigmoid>(out, input, params, tokens, d, stream);
        case ActFn::kTanh: return run_act<__nv_bfloat16, ActFn::kTanh>(out, input, params, tokens, d, stream);
        case ActFn::kSoftplus: return run_act<__nv_bfloat16, ActFn::kSoftplus>(out, input, params, tokens, d, stream);
        case ActFn::kSiTU: return run_act<__nv_bfloat16, ActFn::kSiTU>(out, input, params, tokens, d, stream);
        case ActFn::kFatReLU: return run_act<__nv_bfloat16, ActFn::kFatReLU>(out, input, params, tokens, d, stream);
        default: break;
      }
      break;
    case DType::kFloat16:
      switch (act) {
        case ActFn::kSiLU: return run_act<__half, ActFn::kSiLU>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUTanh: return run_act<__half, ActFn::kGeLUTanh>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUErf: return run_act<__half, ActFn::kGeLUErf>(out, input, params, tokens, d, stream);
        case ActFn::kReLU2: return run_act<__half, ActFn::kReLU2>(out, input, params, tokens, d, stream);
        case ActFn::kIdentity: return run_act<__half, ActFn::kIdentity>(out, input, params, tokens, d, stream);
        case ActFn::kQuickGeLU: return run_act<__half, ActFn::kQuickGeLU>(out, input, params, tokens, d, stream);
        case ActFn::kReLU: return run_act<__half, ActFn::kReLU>(out, input, params, tokens, d, stream);
        case ActFn::kSigmoid: return run_act<__half, ActFn::kSigmoid>(out, input, params, tokens, d, stream);
        case ActFn::kTanh: return run_act<__half, ActFn::kTanh>(out, input, params, tokens, d, stream);
        case ActFn::kSoftplus: return run_act<__half, ActFn::kSoftplus>(out, input, params, tokens, d, stream);
        case ActFn::kSiTU: return run_act<__half, ActFn::kSiTU>(out, input, params, tokens, d, stream);
        case ActFn::kFatReLU: return run_act<__half, ActFn::kFatReLU>(out, input, params, tokens, d, stream);
        default: break;
      }
      break;
    case DType::kFloat32:
      switch (act) {
        case ActFn::kSiLU: return run_act<float, ActFn::kSiLU>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUTanh: return run_act<float, ActFn::kGeLUTanh>(out, input, params, tokens, d, stream);
        case ActFn::kGeLUErf: return run_act<float, ActFn::kGeLUErf>(out, input, params, tokens, d, stream);
        case ActFn::kReLU2: return run_act<float, ActFn::kReLU2>(out, input, params, tokens, d, stream);
        case ActFn::kIdentity: return run_act<float, ActFn::kIdentity>(out, input, params, tokens, d, stream);
        case ActFn::kQuickGeLU: return run_act<float, ActFn::kQuickGeLU>(out, input, params, tokens, d, stream);
        case ActFn::kReLU: return run_act<float, ActFn::kReLU>(out, input, params, tokens, d, stream);
        case ActFn::kSigmoid: return run_act<float, ActFn::kSigmoid>(out, input, params, tokens, d, stream);
        case ActFn::kTanh: return run_act<float, ActFn::kTanh>(out, input, params, tokens, d, stream);
        case ActFn::kSoftplus: return run_act<float, ActFn::kSoftplus>(out, input, params, tokens, d, stream);
        case ActFn::kSiTU: return run_act<float, ActFn::kSiTU>(out, input, params, tokens, d, stream);
        case ActFn::kFatReLU: return run_act<float, ActFn::kFatReLU>(out, input, params, tokens, d, stream);
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

// -----------------------------------------------------------------------------
// F01 + quantization (B3 row 5)
// -----------------------------------------------------------------------------

namespace {

__device__ __forceinline__ uint8_t aq_e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t aq_e2m1_pair(float a, float b) {
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ float aq_warp_max(float v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float aq_block_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = aq_warp_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < warps) ? smem[threadIdx.x] : -__int_as_float(0x7f800000);
    t = aq_warp_max(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float aq_block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < warps) ? smem[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ int64_t aq_scale_offset(ScaleLayout layout, int64_t row, int64_t group,
                                                   int64_t rows, int64_t groups) {
  switch (layout) {
    case ScaleLayout::kRowMajor:
    case ScaleLayout::kPackedUE8M0x4:
      return row * groups + group;
    case ScaleLayout::kColMajorTma: {
      const int64_t padded = (rows + 3) & ~int64_t(3);
      return group * padded + row;
    }
    case ScaleLayout::kSwizzled128x4: {
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

__device__ __forceinline__ uint8_t aq_ue8m0(float s) {
  int e = 0;
  if (s > 0.0f && isfinite(s)) {
    frexpf(s, &e);
    e -= 1;
    if (ldexpf(1.0f, e) < s) ++e;
  } else {
    e = -127;
  }
  e = e < -127 ? -127 : (e > 127 ? 127 : e);
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float aq_ue8m0_to_float(uint8_t e) {
  return ldexpf(1.0f, static_cast<int>(e) - 127);
}

}  // namespace

template <typename scalar_t, ActFn kAct, GateLayout kLayout, ActQuantMode kMode, int VEC_SIZE>
__global__ void act_and_mul_quant_kernel(const __grid_constant__ ActMulQuantParams params) {
  __shared__ float smem[32];
  (void)smem;
  const bool masked = params.masked_m != nullptr;
  const int64_t rows = masked ? static_cast<int64_t>(params.num_experts) * params.max_m
                              : params.num_rows;
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  if (masked) {
    const int e = static_cast<int>(row / params.max_m);
    const int m = static_cast<int>(row - static_cast<int64_t>(e) * params.max_m);
    if (m >= params.masked_m[e]) return;
  }
  if (params.expert_offsets != nullptr && params.num_experts > 0 &&
      row >= params.expert_offsets[params.num_experts]) {
    return;
  }
  const int64_t d = params.d;
  const scalar_t* src = static_cast<const scalar_t*>(params.input) + row * params.input_stride;
  const float rw = params.row_weight != nullptr ? params.row_weight[row] : 1.0f;
  ActQuantOut q = params.quant;
  const int64_t q_stride = q.q_row_stride > 0 ? q.q_row_stride : d;
  const float qmax = q.q_dtype == DType::kInt8 ? kInt8Max : kFp8E4M3Max;
  uint8_t* qptr = static_cast<uint8_t*>(q.q);

  auto gated = [&](int64_t c) -> float {
    float gate, up;
    if constexpr (kLayout == GateLayout::kInterleaved) {
      gate = to_f<scalar_t>(src[2 * c]);
      up = to_f<scalar_t>(src[2 * c + 1]);
    } else if constexpr (kLayout == GateLayout::kUpGate) {
      gate = to_f<scalar_t>(src[d + c]);
      up = to_f<scalar_t>(src[c]);
    } else {
      gate = to_f<scalar_t>(src[c]);
      up = to_f<scalar_t>(src[d + c]);
    }
    return glu<kAct>(gate, up, params.glu, rw);
  };

  if constexpr (kMode == ActQuantMode::kStaticPerTensor) {
    const float s = *q.static_scale;
    for (int64_t c = threadIdx.x; c < d; c += blockDim.x) {
      const float y = gated(c);
      if (q.q_dtype == DType::kInt8) {
        int qv = __float2int_rn(y / s);
        qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
        qptr[row * q_stride + c] = static_cast<uint8_t>(static_cast<int8_t>(qv));
      } else {
        qptr[row * q_stride + c] = aq_e4m3(__fdiv_rn(y, s));
      }
    }
    return;
  }

  if constexpr (kMode == ActQuantMode::kDynamicPerToken) {
    __shared__ __align__(16) float row_cache[4096];
    const bool cache_ok = d <= 4096;
    const bool asym = !q.symmetric && q.q_dtype == DType::kInt8;
    float amax = 0.0f, amin = __int_as_float(0x7f800000);
    const int64_t nv = d / 4;
    if (cache_ok && d % 4 == 0 && (q_stride % 4 == 0) && !asym) {
      for (int64_t v = threadIdx.x; v < nv; v += blockDim.x) {
        float y4[4];
        if constexpr (kLayout == GateLayout::kInterleaved) {
          const Pack<scalar_t, 8> p =
              *reinterpret_cast<const Pack<scalar_t, 8>*>(src + 2 * (v * 4));
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            y4[j] = glu<kAct>(to_f<scalar_t>(p.x[2 * j]), to_f<scalar_t>(p.x[2 * j + 1]),
                              params.glu, rw);
          }
        } else {
          const scalar_t* gp = src + v * 4 + (kLayout == GateLayout::kUpGate ? d : 0);
          const scalar_t* up = src + v * 4 + (kLayout == GateLayout::kUpGate ? 0 : d);
          const Pack<scalar_t, 4> pg = *reinterpret_cast<const Pack<scalar_t, 4>*>(gp);
          const Pack<scalar_t, 4> pu = *reinterpret_cast<const Pack<scalar_t, 4>*>(up);
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            y4[j] = glu<kAct>(to_f<scalar_t>(pg.x[j]), to_f<scalar_t>(pu.x[j]), params.glu, rw);
          }
        }
        *reinterpret_cast<float4*>(&row_cache[v * 4]) =
            make_float4(y4[0], y4[1], y4[2], y4[3]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          amax = fmaxf(amax, fabsf(y4[j]));
          amin = fminf(amin, y4[j]);
        }
      }
      amax = aq_block_max(amax, smem);
      __syncthreads();
      amin = aq_block_sum(amin, smem);
      float s;
      if (threadIdx.x == 0) {
        s = fmaxf(amax, 1e-10f) / qmax;
        if (q.scale_upper_bound != nullptr) s = fminf(s, q.scale_upper_bound[0]);
        static_cast<float*>(q.scales)[row] = s;
      }
      __syncthreads();
      s = static_cast<float*>(q.scales)[row];
      for (int64_t v = threadIdx.x; v < nv; v += blockDim.x) {
        const float4 y4 = *reinterpret_cast<const float4*>(&row_cache[v * 4]);
        uint8_t codes[4];
        const float yy[4] = {y4.x, y4.y, y4.z, y4.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (q.q_dtype == DType::kInt8) {
            int qv = __float2int_rn(yy[j] / s);
            qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
            codes[j] = static_cast<uint8_t>(static_cast<int8_t>(qv));
          } else {
            codes[j] = aq_e4m3(__fdiv_rn(yy[j], s));
          }
        }
        *reinterpret_cast<uint32_t*>(qptr + row * q_stride + v * 4) =
            static_cast<uint32_t>(codes[0]) | (static_cast<uint32_t>(codes[1]) << 8) |
            (static_cast<uint32_t>(codes[2]) << 16) | (static_cast<uint32_t>(codes[3]) << 24);
      }
      return;
    }
    for (int64_t c = threadIdx.x; c < d; c += blockDim.x) {
      const float y = gated(c);
      if (cache_ok) row_cache[c] = y;
      amax = fmaxf(amax, fabsf(y));
      amin = fminf(amin, y);
    }
    amax = aq_block_max(amax, smem);
    __syncthreads();
    amin = aq_block_sum(amin, smem);
    __syncthreads();
    float s = 0.0f, zero = -128.0f;
    if (asym) {
      s = __fdiv_rn(fmaxf(amax, 1e-10f) - amin, 255.0f);
    } else {
      s = fmaxf(amax, 1e-10f) / qmax;
      if (q.scale_upper_bound != nullptr) s = fminf(s, q.scale_upper_bound[0]);
    }
    if (threadIdx.x == 0) {
      static_cast<float*>(q.scales)[row] = s;
      if (asym) {
        const int z = __float2int_rn(-128.0f - __fdiv_rn(amin, s));
        q.zero_points[row] = z;
      }
    }
    __syncthreads();
    s = static_cast<float*>(q.scales)[row];
    if (asym) zero = static_cast<float>(q.zero_points[row]);
    for (int64_t c = threadIdx.x; c < d; c += blockDim.x) {
      const float y = cache_ok ? row_cache[c] : gated(c);
      if (asym) {
        int v = __float2int_rn(y / s + zero);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        qptr[row * q_stride + c] = static_cast<uint8_t>(static_cast<int8_t>(v));
      } else {
        qptr[row * q_stride + c] = q.q_dtype == DType::kInt8
                                       ? static_cast<uint8_t>(static_cast<int8_t>(
                                             __float2int_rn(y / s)))
                                       : aq_e4m3(__fdiv_rn(y, s));
      }
    }
    return;
  }

  const int group = kMode == ActQuantMode::kNVFP4
                        ? kNvFp4Block
                        : (kMode == ActQuantMode::kMXFP8 || kMode == ActQuantMode::kMXFP4
                               ? kMxBlock
                               : (q.group_size > 0 ? q.group_size : 128));
  const int64_t group_count = d / group;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = 8;
  const int per_lane = group / 32;
  for (int64_t g = warp; g < group_count; g += kWarps) {
    float y[8];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j = lane + 32 * i;
      y[i] = (i < per_lane) ? gated(g * group + j) : 0.0f;
      if (i < per_lane) amax = fmaxf(amax, fabsf(y[i]));
    }
    amax = aq_warp_max(amax);
    float s = fmaxf(amax, 1e-10f) / qmax;
    if constexpr (kMode == ActQuantMode::kPerGroup) {
      if (lane == 0) {
        static_cast<float*>(q.scales)[aq_scale_offset(q.scale_layout, row, g, rows,
                                                      group_count)] = s;
      }
      s = __shfl_sync(0xffffffffu, s, 0);
    } else if constexpr (kMode == ActQuantMode::kMXFP8 || kMode == ActQuantMode::kMXFP4) {
      const uint8_t e = aq_ue8m0(s);
      if (lane == 0) {
        static_cast<uint8_t*>(q.scales)[aq_scale_offset(q.scale_layout, row, g, rows,
                                                        group_count)] = e;
      }
      s = __shfl_sync(0xffffffffu, aq_ue8m0_to_float(e), 0);
    } else if constexpr (kMode == ActQuantMode::kNVFP4) {
      const float gscale = q.global_scale != nullptr ? q.global_scale[0] : 1.0f;
      const uint8_t e = aq_e4m3(__fdiv_rn(__fmul_rn(amax, gscale), kFp4E2M1Max));
      if (lane == 0) {
        static_cast<uint8_t*>(q.scales)[aq_scale_offset(q.scale_layout, row, g, rows,
                                                        group_count)] = e;
      }
      s = __shfl_sync(0xffffffffu, __half2float(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)), 0) /
          gscale;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i >= per_lane) break;
      const int j = lane + 32 * i;
      const int64_t c = g * group + j;
      if constexpr (kMode == ActQuantMode::kMXFP4 || kMode == ActQuantMode::kNVFP4) {
        const float gscale = (kMode == ActQuantMode::kNVFP4 && q.global_scale != nullptr)
                                 ? q.global_scale[0]
                                 : 1.0f;
        if ((lane & 1) == 0) {
          const float y2 = __shfl_down_sync(0xffffffffu, y[i], 1);
          qptr[row * q_stride + (c / 2)] =
              aq_e2m1_pair(__fdiv_rn(__fmul_rn(y[i], gscale), s),
                           __fdiv_rn(__fmul_rn(y2, gscale), s));
        }
      } else {
        qptr[row * q_stride + c] = q.q_dtype == DType::kInt8
                                       ? static_cast<uint8_t>(static_cast<int8_t>(
                                             __float2int_rn(y[i] / s)))
                                       : aq_e4m3(__fdiv_rn(y[i], s));
      }
    }
  }
}

namespace {

template <typename scalar_t, ActFn kAct, GateLayout kLayout, ActQuantMode kMode>
Status launch_act_quant(const ActMulQuantParams& p, cudaStream_t stream) {
  const int64_t rows = p.masked_m != nullptr
                           ? static_cast<int64_t>(p.num_experts) * p.max_m
                           : p.num_rows;
  if (rows <= 0) return Status::kSuccess;
  if (p.d % 8 == 0) {
    act_and_mul_quant_kernel<scalar_t, kAct, kLayout, kMode, 8>
        <<<static_cast<unsigned>(rows), 256, 0, stream>>>(p);
  } else if (p.d % 4 == 0) {
    act_and_mul_quant_kernel<scalar_t, kAct, kLayout, kMode, 4>
        <<<static_cast<unsigned>(rows), 256, 0, stream>>>(p);
  } else {
    act_and_mul_quant_kernel<scalar_t, kAct, kLayout, kMode, 1>
        <<<static_cast<unsigned>(rows), 256, 0, stream>>>(p);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "act_and_mul_quant: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

}  // namespace

Status act_and_mul_quant(const ActMulQuantParams& params, ActFn act, GateLayout layout,
                         DType input_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "act_and_mul_quant: CUDA backend only");
  }
  if (params.input == nullptr || params.d <= 0 ||
      (params.masked_m == nullptr && params.num_rows < 0)) {
    return detail::set_error(Status::kInvalidArgument, "act_and_mul_quant: input/extents");
  }
  if (params.quant.mode != ActQuantMode::kNone && params.quant.q == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "act_and_mul_quant: quant.q is null");
  }
  if (params.quant.mode == ActQuantMode::kNone) {
    return detail::set_error(Status::kUnsupportedShape,
                             "act_and_mul_quant: a quantization mode is required");
  }
  if (params.quant.mode != ActQuantMode::kStaticPerTensor && params.quant.scales == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "act_and_mul_quant: quant.scales is null");
  }
  if (params.quant.mode == ActQuantMode::kStaticPerTensor &&
      params.quant.static_scale == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "act_and_mul_quant: static_scale is null");
  }
  if (params.quant.mode == ActQuantMode::kPerGroup) {
    const int group = params.quant.group_size > 0 ? params.quant.group_size : 128;
    if ((group != 32 && group != 64 && group != 128) || params.d % group != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "act_and_mul_quant: group %d does not divide d", group);
    }
  }
  if ((params.quant.mode == ActQuantMode::kMXFP8 ||
       params.quant.mode == ActQuantMode::kMXFP4) &&
      params.d % kMxBlock != 0) {
    return detail::set_error(Status::kUnsupportedShape, "act_and_mul_quant: MX needs d %% 32");
  }
  if (params.quant.mode == ActQuantMode::kNVFP4 &&
      (params.d % kNvFp4Block != 0 || params.quant.global_scale == nullptr)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "act_and_mul_quant: NVFP4 needs d %% 16 and global_scale");
  }
  if (params.expert_offsets != nullptr &&
      params.quant.scale_layout == ScaleLayout::kColMajorTma) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "act_and_mul_quant: grouped layout needs row-major scales");
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_AQ_LAYOUT(S, A, L, M)                                                                  \
  do {                                                                                             \
    if (layout == L) return launch_act_quant<S, A, L, M>(params, stream);                           \
  } while (0)
#define KOPS_AQ_MODE(S, A, L)                                                                       \
  do {                                                                                             \
    switch (params.quant.mode) {                                                                   \
      case ActQuantMode::kStaticPerTensor:                                                         \
        return launch_act_quant<S, A, L, ActQuantMode::kStaticPerTensor>(params, stream);          \
      case ActQuantMode::kDynamicPerToken:                                                         \
        return launch_act_quant<S, A, L, ActQuantMode::kDynamicPerToken>(params, stream);          \
      case ActQuantMode::kPerGroup:                                                                \
        return launch_act_quant<S, A, L, ActQuantMode::kPerGroup>(params, stream);                 \
      case ActQuantMode::kMXFP8:                                                                   \
        return launch_act_quant<S, A, L, ActQuantMode::kMXFP8>(params, stream);                    \
      case ActQuantMode::kMXFP4:                                                                   \
        return launch_act_quant<S, A, L, ActQuantMode::kMXFP4>(params, stream);                    \
      case ActQuantMode::kNVFP4:                                                                   \
        return launch_act_quant<S, A, L, ActQuantMode::kNVFP4>(params, stream);                    \
      default:                                                                                     \
        return detail::set_error(Status::kUnsupportedShape, "act_and_mul_quant: mode %d",           \
                                 static_cast<int>(params.quant.mode));                             \
    }                                                                                              \
  } while (0)
#define KOPS_AQ_ACT(S, L)                                                                           \
  do {                                                                                             \
    switch (act) {                                                                                 \
      case ActFn::kSiLU: KOPS_AQ_MODE(S, ActFn::kSiLU, L);                                          \
      case ActFn::kGeLUErf: KOPS_AQ_MODE(S, ActFn::kGeLUErf, L);                                    \
      case ActFn::kGeLUTanh: KOPS_AQ_MODE(S, ActFn::kGeLUTanh, L);                                  \
      case ActFn::kQuickGeLU: KOPS_AQ_MODE(S, ActFn::kQuickGeLU, L);                                \
      case ActFn::kReLU: KOPS_AQ_MODE(S, ActFn::kReLU, L);                                          \
      case ActFn::kReLU2: KOPS_AQ_MODE(S, ActFn::kReLU2, L);                                        \
      case ActFn::kSigmoid: KOPS_AQ_MODE(S, ActFn::kSigmoid, L);                                    \
      case ActFn::kTanh: KOPS_AQ_MODE(S, ActFn::kTanh, L);                                          \
      case ActFn::kSoftplus: KOPS_AQ_MODE(S, ActFn::kSoftplus, L);                                  \
      case ActFn::kSiTU: KOPS_AQ_MODE(S, ActFn::kSiTU, L);                                          \
      case ActFn::kFatReLU: KOPS_AQ_MODE(S, ActFn::kFatReLU, L);                                    \
      case ActFn::kIdentity: KOPS_AQ_MODE(S, ActFn::kIdentity, L);                                  \
    }                                                                                              \
    return detail::set_error(Status::kUnsupportedShape, "act_and_mul_quant: act %d",                 \
                             static_cast<int>(act));                                                \
  } while (0)
#define KOPS_AQ_DTYPE(S)                                                                            \
  do {                                                                                             \
    switch (layout) {                                                                              \
      case GateLayout::kGateUp: KOPS_AQ_ACT(S, GateLayout::kGateUp);                                \
      case GateLayout::kUpGate: KOPS_AQ_ACT(S, GateLayout::kUpGate);                                \
      case GateLayout::kInterleaved: KOPS_AQ_ACT(S, GateLayout::kInterleaved);                      \
    }                                                                                              \
    return detail::set_error(Status::kUnsupportedShape, "act_and_mul_quant: layout %d",              \
                             static_cast<int>(layout));                                            \
  } while (0)
  if (input_dtype == DType::kBFloat16) KOPS_AQ_DTYPE(__nv_bfloat16);
  if (input_dtype == DType::kFloat16) KOPS_AQ_DTYPE(__half);
  if (input_dtype == DType::kFloat32) KOPS_AQ_DTYPE(float);
#undef KOPS_AQ_DTYPE
#undef KOPS_AQ_ACT
#undef KOPS_AQ_MODE
#undef KOPS_AQ_LAYOUT
  return detail::set_error(Status::kUnsupportedDtype, "act_and_mul_quant: dtype %s",
                           dtype_name(input_dtype));
}

Status gate_mul(const Tensor& out, const Tensor& x, const Tensor& gate,
                const Tensor& accumulate_into, ActFn gate_act, const LaunchContext& ctx) {
  Status s = check_tensor(x, x.dtype, 2, 16, "gate_mul.x");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, x.dtype, 2, 16, "gate_mul.out");
  if (s != Status::kSuccess) return s;
  // A gate of width one or a few heads has a row stride smaller than 16 B;
  // only the full-width path needs 16-byte packets.
  s = check_tensor(gate, x.dtype, 2, 1, "gate_mul.gate");
  if (s != Status::kSuccess) return s;
  const int64_t tokens = x.shape[0];
  const int64_t d = x.shape[1];
  if (out.shape[0] != tokens || out.shape[1] != d) {
    return detail::set_error(Status::kInvalidArgument, "gate_mul: out shape mismatch");
  }
  if (gate.shape[0] != tokens) {
    return detail::set_error(Status::kInvalidArgument, "gate_mul: gate rows mismatch");
  }
  const int64_t gate_width = gate.shape[1];
  if (gate_width <= 0 || d % gate_width != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "gate_mul: d=%lld not divisible by gate_width=%lld",
                             static_cast<long long>(d), static_cast<long long>(gate_width));
  }
  if (tensor_present(accumulate_into)) {
    s = check_tensor(accumulate_into, x.dtype, 2, 16, "gate_mul.accumulate");
    if (s != Status::kSuccess) return s;
    if (accumulate_into.shape[0] != tokens || accumulate_into.shape[1] != d) {
      return detail::set_error(Status::kInvalidArgument, "gate_mul: accumulate shape mismatch");
    }
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (x.dtype) {
    case DType::kBFloat16:
      return dispatch_gate_mul<__nv_bfloat16>(out, x, gate, accumulate_into, gate_act,
                                              static_cast<int>(tokens), static_cast<int>(d),
                                              static_cast<int>(gate_width), stream);
    case DType::kFloat16:
      return dispatch_gate_mul<__half>(out, x, gate, accumulate_into, gate_act,
                                       static_cast<int>(tokens), static_cast<int>(d),
                                       static_cast<int>(gate_width), stream);
    case DType::kFloat32:
      return dispatch_gate_mul<float>(out, x, gate, accumulate_into, gate_act,
                                      static_cast<int>(tokens), static_cast<int>(d),
                                      static_cast<int>(gate_width), stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "gate_mul: %s", dtype_name(x.dtype));
  }
}

}  // namespace kops::act
