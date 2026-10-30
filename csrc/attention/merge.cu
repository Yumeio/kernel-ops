// =============================================================================
// kernel-ops | csrc/attention/merge.cu
// A14 first implementations: merge_state and merge_states (attention-state
// algebra on (o, lse) pairs). Deterministic order: state i is consumed in
// increasing i, so the result does not depend on scheduling.
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
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace kops::attn {
namespace {

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

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

/// Merge two states; both -inf yields 0 output and -inf lse.
__device__ __forceinline__ void combine(float la, float lb, float& out_l, float& wa, float& wb) {
  const float m = fmaxf(la, lb);
  if (m == neg_inf()) {
    out_l = neg_inf();
    wa = 0.0f;
    wb = 0.0f;
    return;
  }
  const float ea = expf(la - m);
  const float eb = expf(lb - m);
  const float denom = ea + eb;
  out_l = m + logf(denom);
  wa = ea / denom;
  wb = eb / denom;
}

}  // namespace

template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_state_kernel(DTypeO* __restrict__ o, float* __restrict__ lse,
                                   const DTypeIn* __restrict__ o_a, const float* __restrict__ lse_a,
                                   const DTypeIn* __restrict__ o_b, const float* __restrict__ lse_b,
                                   const uint8_t* __restrict__ row_mask, const int num_rows,
                                   const int num_heads) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const bool take_a_only = (row_mask != nullptr) && (row_mask[row] == 0);
  for (int h = 0; h < num_heads; ++h) {
    const int64_t idx = static_cast<int64_t>(row) * num_heads + h;
    const int64_t base = idx * HEAD_DIM;
    if (take_a_only) {
      for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
        from_f<DTypeO>(o[base + d], to_f<DTypeIn>(o_a[base + d]));
      if (lse != nullptr) lse[idx] = lse_a[idx];
      continue;
    }
    float l, wa, wb;
    combine(lse_a[idx], lse_b[idx], l, wa, wb);
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
      const float v = wa * to_f<DTypeIn>(o_a[base + d]) + wb * to_f<DTypeIn>(o_b[base + d]);
      from_f<DTypeO>(o[base + d], v);
    }
    if (lse != nullptr) lse[idx] = l;
  }
}

template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_states_kernel(DTypeO* __restrict__ o, float* __restrict__ lse,
                                    const DTypeIn* __restrict__ o_states,
                                    const float* __restrict__ lse_states, const int num_rows,
                                    const int num_states, const int num_heads) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  for (int h = 0; h < num_heads; ++h) {
    const int64_t idx = static_cast<int64_t>(row) * num_heads + h;
    float m = neg_inf();
    for (int s = 0; s < num_states; ++s) {
      const float l = lse_states[static_cast<int64_t>(row) * num_states * num_heads +
                                 static_cast<int64_t>(s) * num_heads + h];
      m = fmaxf(m, l);
    }
    float weight_sum = 0.0f;
    if (m != neg_inf()) {
      for (int s = 0; s < num_states; ++s) {
        weight_sum += expf(lse_states[static_cast<int64_t>(row) * num_states * num_heads +
                                      static_cast<int64_t>(s) * num_heads + h] -
                           m);
      }
    }
    const float l_out = (m == neg_inf()) ? neg_inf() : m + logf(weight_sum);
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
      float acc = 0.0f;
      if (m != neg_inf()) {
        for (int s = 0; s < num_states; ++s) {
          const float l = lse_states[static_cast<int64_t>(row) * num_states * num_heads +
                                     static_cast<int64_t>(s) * num_heads + h];
          const float w = expf(l - l_out);
          const int64_t src = ((static_cast<int64_t>(row) * num_states + s) * num_heads + h) * HEAD_DIM;
          acc += w * to_f<DTypeIn>(o_states[src + d]);
        }
      }
      from_f<DTypeO>(o[idx * HEAD_DIM + d], acc);
    }
    if (lse != nullptr) lse[idx] = l_out;
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

template <typename In, typename Out, int HEAD>
Status run_merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                       const Tensor& o_b, const Tensor& lse_b, int rows, int heads,
                       cudaStream_t stream) {
  merge_state_kernel<In, Out, HEAD, vec_width<Out>()><<<rows, 128, 0, stream>>>(
      reinterpret_cast<Out*>(static_cast<char*>(o.data) + o.byte_offset),
      lse.data != nullptr ? reinterpret_cast<float*>(static_cast<char*>(lse.data) + lse.byte_offset) : nullptr,
      reinterpret_cast<const In*>(static_cast<const char*>(o_a.data) + o_a.byte_offset),
      reinterpret_cast<const float*>(static_cast<const char*>(lse_a.data) + lse_a.byte_offset),
      reinterpret_cast<const In*>(static_cast<const char*>(o_b.data) + o_b.byte_offset),
      reinterpret_cast<const float*>(static_cast<const char*>(lse_b.data) + lse_b.byte_offset),
      nullptr, rows, heads);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "merge_state launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename In, typename Out>
Status dispatch_merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a,
                            const Tensor& lse_a, const Tensor& o_b, const Tensor& lse_b,
                            cudaStream_t stream) {
  const int head = static_cast<int>(o.shape[2]);
  const int rows = static_cast<int>(o.shape[0]);
  const int heads = static_cast<int>(o.shape[1]);
  switch (head) {
    case 64: return run_merge_state<In, Out, 64>(o, lse, o_a, lse_a, o_b, lse_b, rows, heads, stream);
    case 128: return run_merge_state<In, Out, 128>(o, lse, o_a, lse_a, o_b, lse_b, rows, heads, stream);
    case 256: return run_merge_state<In, Out, 256>(o, lse, o_a, lse_a, o_b, lse_b, rows, heads, stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "merge_state: head_dim %d", head);
  }
}

}  // namespace

Status merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                   const Tensor& o_b, const Tensor& lse_b, const LaunchContext& ctx) {
  Status s = check_tensor(o, o.dtype, 3, 16, "merge_state.o");
  if (s != Status::kSuccess) return s;
  s = check_tensor(o_a, o_a.dtype, 3, 16, "merge_state.o_a");
  if (s != Status::kSuccess) return s;
  s = check_tensor(o_b, o_b.dtype, 3, 16, "merge_state.o_b");
  if (s != Status::kSuccess) return s;
  if (o_a.dtype != o_b.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_state: o_a %s vs o_b %s",
                             dtype_name(o_a.dtype), dtype_name(o_b.dtype));
  }
  for (int d = 0; d < 3; ++d) {
    if (o.shape[d] != o_a.shape[d] || o.shape[d] != o_b.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "merge_state: shape mismatch at dim %d", d);
    }
  }
  if (o_a.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "merge_state: partial states must be fp32, got %s",
                             dtype_name(o_a.dtype));
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (o.dtype) {
    case DType::kBFloat16: return dispatch_merge_state<float, __nv_bfloat16>(o, lse, o_a, lse_a, o_b, lse_b, stream);
    case DType::kFloat16: return dispatch_merge_state<float, __half>(o, lse, o_a, lse_a, o_b, lse_b, stream);
    case DType::kFloat32: return dispatch_merge_state<float, float>(o, lse, o_a, lse_a, o_b, lse_b, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "merge_state: out dtype %s",
                               dtype_name(o.dtype));
  }
}

Status merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                    const Tensor& lse_states, const LaunchContext& ctx) {
  (void)o_states; (void)lse_states;
  (void)o; (void)lse; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "merge_states: M1.1");
}

Status merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                      const Tensor& sink_logits, const LaunchContext& ctx) {
  (void)o; (void)lse; (void)split; (void)sink_logits; (void)ctx;
  return detail::set_error(Status::kNotImplemented, "merge_split_kv: M1.1");
}

}  // namespace kops::attn
