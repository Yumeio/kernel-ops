#pragma once
// =============================================================================
// kernel-ops | csrc/common/device_impl.cuh
// Device bodies for the primitives declared in common/device.cuh, plus the small
// scalar/vector helpers shared by the B0/B1 kernel groups.
//
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh,
//           <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>.
// =============================================================================

namespace kops {
namespace dev {

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

template <int N, typename T>
__device__ __forceinline__ Pack<T, N> load_pack(const T* __restrict__ p) {
  return *reinterpret_cast<const Pack<T, N>*>(p);
}

template <int N, typename T>
__device__ __forceinline__ void store_pack(T* __restrict__ p, const Pack<T, N>& v) {
  *reinterpret_cast<Pack<T, N>*>(p) = v;
}

}  // namespace dev

// ---- Vector primitives (declared in device.cuh) ----------------------------
template <typename T, int N>
KOPS_DEVICE Vec<T, N> load_vec(const T* __restrict__ ptr) {
  return *reinterpret_cast<const Vec<T, N>*>(ptr);
}

template <typename T, int N>
KOPS_DEVICE void store_vec(T* __restrict__ ptr, const Vec<T, N>& v) {
  *reinterpret_cast<Vec<T, N>*>(ptr) = v;
}

template <typename T, int N>
KOPS_DEVICE Vec<T, N> load_vec_streaming(const T* __restrict__ ptr) {
  return *reinterpret_cast<const Vec<T, N>*>(ptr);
}

// ---- Fast math -------------------------------------------------------------
KOPS_DEVICE float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
KOPS_DEVICE float fast_log2(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
KOPS_DEVICE float fast_rcp(float x) { return __frcp_rn(x); }
KOPS_DEVICE float fast_rsqrt(float x) { return rsqrtf(x); }
KOPS_DEVICE float fast_tanh(float x) { return tanhf(x); }

KOPS_DEVICE float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
KOPS_DEVICE float softplus(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

template <ActFn kAct>
KOPS_DEVICE float apply_act(float x, const GluParams& p) {
  switch (kAct) {
    case ActFn::kIdentity: return x;
    case ActFn::kSiLU: return x * sigmoid(p.gate_alpha * x);
    case ActFn::kGeLUErf: return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    case ActFn::kGeLUTanh:
      return 0.5f * x *
             (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case ActFn::kQuickGeLU: return x * sigmoid(1.702f * x);
    case ActFn::kReLU: return x > 0.0f ? x : 0.0f;
    case ActFn::kReLU2: return x > 0.0f ? x * x : 0.0f;
    case ActFn::kSigmoid: return sigmoid(x);
    case ActFn::kTanh: return tanhf(x);
    case ActFn::kSoftplus: return softplus(x);
    case ActFn::kSiTU:
      return p.situ_beta * tanhf(x / p.situ_beta) * sigmoid(x);
    case ActFn::kFatReLU: return x > p.fatrelu_threshold ? x : 0.0f;
  }
  return x;
}

template <ActFn kAct>
KOPS_DEVICE float apply_glu(float gate, float up, const GluParams& p) {
  if (p.gate_clamp_max > 0.0f && gate > p.gate_clamp_max) gate = p.gate_clamp_max;
  if (p.up_clamp > 0.0f) {
    up = up > p.up_clamp ? p.up_clamp : up;
    up = up < -p.up_clamp ? -p.up_clamp : up;
  }
  up += p.up_offset;
  if (p.situ_linear_beta > 0.0f) up = p.situ_linear_beta * tanhf(up / p.situ_linear_beta);
  return apply_act<kAct>(gate, p) * up;
}

KOPS_DEVICE float softcap(float x, float cap) {
  return cap > 0.0f ? cap * tanhf(x / cap) : x;
}

// ---- Reductions ------------------------------------------------------------
template <typename T>
KOPS_DEVICE T warp_reduce_sum(T v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T>
KOPS_DEVICE T warp_reduce_max(T v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T>
KOPS_DEVICE T warp_reduce_min(T v) {
  for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T, int kWidth>
KOPS_DEVICE T group_reduce_sum(T v) {
  for (int off = kWidth / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, kWidth);
  return __shfl_sync(0xffffffffu, v, 0, kWidth);
}

template <typename T, int NUM_WARPS>
KOPS_DEVICE T block_reduce_sum(T v, T* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T t = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : T{};
    t = warp_reduce_sum(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

template <typename T, int NUM_WARPS>
KOPS_DEVICE T block_reduce_max(T v, T* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T t = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : __int_as_float(0xff800000);
    t = warp_reduce_max(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

/// (value, index) argmax with deterministic tie-break to the lowest index.
template <typename T>
KOPS_DEVICE void warp_argmax(T& v, int32_t& idx) {
  for (int off = 16; off > 0; off >>= 1) {
    const T ov = __shfl_down_sync(0xffffffffu, v, off);
    const int32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
}

// ---- Online softmax --------------------------------------------------------
KOPS_DEVICE void softmax_state_init(SoftmaxState& s) {
  s.m = __int_as_float(0xff800000);
  s.d = 0.0f;
}

KOPS_DEVICE void softmax_state_update(SoftmaxState& s, float logit) {
  const float m = fmaxf(s.m, logit);
  s.d = s.d * __expf(s.m - m) + __expf(logit - m);
  s.m = m;
}

KOPS_DEVICE void softmax_state_merge(SoftmaxState& a, const SoftmaxState& b) {
  const float m = fmaxf(a.m, b.m);
  a.d = a.d * __expf(a.m - m) + b.d * __expf(b.m - m);
  a.m = m;
}

}  // namespace kops
