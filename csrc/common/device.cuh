#pragma once
// =============================================================================
// kernel-ops | csrc/common/device.cuh
// Device building blocks: vector types, fast math, warp/block reductions,
// online-softmax state. Declarations only; bodies live in device_impl.cuh.
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh.
// =============================================================================

namespace kops {

/// Register vector of N elements of T with natural alignment (N*sizeof(T) <= 16
/// enables 128-bit LDG/STG). Used for VEC_SIZE template parameters.
template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
};

/// 128-bit vectorized load/store; pointer must be 16-byte aligned.
template <typename T, int N> KOPS_DEVICE Vec<T, N> load_vec(const T* KOPS_RESTRICT ptr);
template <typename T, int N> KOPS_DEVICE void store_vec(T* KOPS_RESTRICT ptr, const Vec<T, N>& v);
/// Load with L1 no-allocate / evict-first hint (streamed weights, KV pages).
template <typename T, int N> KOPS_DEVICE Vec<T, N> load_vec_streaming(const T* KOPS_RESTRICT ptr);

// ---- Fast math (fp32) ------------------------------------------------------
KOPS_DEVICE float fast_exp2(float x);        // ex2.approx.ftz
KOPS_DEVICE float fast_log2(float x);        // lg2.approx
KOPS_DEVICE float fast_rcp(float x);         // rcp.approx
KOPS_DEVICE float fast_rsqrt(float x);       // rsqrt.approx
KOPS_DEVICE float fast_tanh(float x);        // tanh.approx (sm_75+)
KOPS_DEVICE float sigmoid(float x);
KOPS_DEVICE float softplus(float x);
/// Apply ActFn with GluParams-dependent constants (alpha for SiLU, beta for SiTU).
template <ActFn kAct> KOPS_DEVICE float apply_act(float x, const GluParams& p);
/// out = f(gate) * g(up) exactly as documented on GluParams.
template <ActFn kAct> KOPS_DEVICE float apply_glu(float gate, float up, const GluParams& p);
/// Logit soft-capping: cap * tanh(x / cap).
KOPS_DEVICE float softcap(float x, float cap);

// ---- Reductions ----------------------------------------------------------------
template <typename T> KOPS_DEVICE T warp_reduce_sum(T v);
template <typename T> KOPS_DEVICE T warp_reduce_max(T v);
template <typename T> KOPS_DEVICE T warp_reduce_min(T v);
/// Butterfly reduction across `kWidth` lanes (kWidth power of two <= 32).
template <typename T, int kWidth> KOPS_DEVICE T group_reduce_sum(T v);
/// Block reduction; result broadcast to all threads. smem must hold NUM_WARPS T.
template <typename T, int NUM_WARPS> KOPS_DEVICE T block_reduce_sum(T v, T* smem);
template <typename T, int NUM_WARPS> KOPS_DEVICE T block_reduce_max(T v, T* smem);
/// (value, index) argmax with deterministic tie-break to the lowest index.
template <typename T> KOPS_DEVICE void warp_argmax(T& v, int32_t& idx);

/// Online-softmax running state (max m, denominator d) used by attention,
/// sampling softmax and split-KV merges. LSE convention of the whole library:
/// lse = m + log(d) in NATURAL log, fp32. Kernels that work in base 2 convert
/// at their boundary (lse_e = lse_2 * ln 2).
struct SoftmaxState {
  float m;  // running max of logits (already scaled by sm_scale)
  float d;  // running sum of exp(logit - m)
};
KOPS_DEVICE void softmax_state_init(SoftmaxState& s);
KOPS_DEVICE void softmax_state_update(SoftmaxState& s, float logit);
/// Merge b into a: m = max, d = da*exp(ma-m) + db*exp(mb-m).
KOPS_DEVICE void softmax_state_merge(SoftmaxState& a, const SoftmaxState& b);

}  // namespace kops
