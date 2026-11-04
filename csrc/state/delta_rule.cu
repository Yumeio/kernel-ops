// =============================================================================
// kernel-ops | csrc/state/delta_rule.cu
// B9 gated delta rule, recurrent half (L02): GDN (Qwen3-Next / Qwen3.5, scalar decay per value
// head) and KDA (Kimi Linear / Kimi K3, decay per key channel).
//   row 3  fused_recurrent_delta_rule_kernel          decode, spec verification, short prefill
//          fused_short_conv_delta_rule_decode_kernel  short conv + gating + L2 + rule + gated RMSNorm
//   row 7  delta_rule_gating_kernel                   gate preparation (step 1 of the chunked prefill)
// The chunked WY prefill is delta_rule_chunk.cu. Per token, with q, k optionally L2-normalized
// (x / sqrt(sum x^2 + 1e-6)) and q scaled by `scale`:
//   S = exp(g) S;  S += k (beta (v - S^T k))^T;  o = S^T q      (S [Dk, Dv] fp32 per value head)
// Contract: delta_rule_kernels.cuh, ops.h, CONTRACTS.md §1f.
// Ported from the ayaka implementation (3rd/csrc/state/delta_rule.cu).
// =============================================================================
#include <algorithm>
#include <cstdint>
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
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

namespace kops::state {
namespace {

using dev::group_sum;
using dev::sigmoid_f;
using dev::silu_f;
using dev::softplus_f;

constexpr float kL2Eps = 1e-6f;

template <typename T>
__device__ __forceinline__ float to_float(T v) { return dev::to_f(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v) {
  T r;
  dev::from_f(r, v);
  return r;
}

}  // namespace

namespace {
// ---- ported from ayaka state/delta_rule.cu:29-327 ----
constexpr int kMaxHeadDimV = 256;

// ---- fused_recurrent_delta_rule_kernel -------------------------------------------------------------------

constexpr int kRecWarps = 4;  // warps per CTA, each with its own Dv tile

/**
 * Grid (ceil(Dv / (kCols kRecWarps)), Hv, B). A warp owns kCols = 32 / kSplit value columns of one (sequence, value head).
 * Each column lives in kSplit lanes (kSplit = 2 for Dk = 128, to stay under the register file) that hold kRows = Dk / kSplit
 * rows of it in registers; the contractions over Dk (S^T k, S^T q) are per-lane sums plus, for kSplit = 2, one shuffle.
 * k, q (normalized and scaled) and the KDA decays of the token are staged once per token in a warp-private shared buffer
 * and read as broadcasts. Warps never synchronize with each other.
 */
}  // namespace
template <typename T, DecayGranularity kDecay, int kDk, bool kTransposed>
__global__ __launch_bounds__(kRecWarps * 32, 1) void fused_recurrent_delta_rule_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kSplit = kDk >= 128 ? 2 : 1, kCols = 32 / kSplit, kRows = kDk / kSplit;
  __shared__ float stage[kRecWarps][(kDecay == DecayGranularity::kPerChannel ? 3 : 2) * kDk];
  const int b = blockIdx.z, hv = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int dv_base = (blockIdx.x * kRecWarps + warp) * kCols;
  if (dv_base >= p.head_dim_v) return;  // uniform per warp; no CTA-wide barrier is used
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;
  const int col = lane % kCols, part = lane / kCols, row0 = part * kRows;
  const int dv = dv_base + col;
  const bool dv_ok = dv < p.head_dim_v;
  const int seq_start = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - seq_start;
  const int hk = hv / (p.num_v_heads / p.num_k_heads);
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  float* ks = stage[warp];
  float* qs = ks + kDk;
  float* ds = qs + kDk;

  const T* q = static_cast<const T*>(p.q) + int64_t(hk) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + int64_t(hv) * p.v_head_stride + dv;
  T* o = static_cast<T*>(p.o) + int64_t(hv) * p.o_head_stride + dv;

  // Element (row0 + j, dv) of the slot's state.
  const auto state_ptr = [&](int64_t s) {
    return p.state_pool + s * p.state_slot_stride + int64_t(hv) * p.head_dim_k * p.head_dim_v +
           (kTransposed ? int64_t(dv) * kDk + row0 : int64_t(row0) * p.head_dim_v + dv);
  };
  float S[kRows];
  {
    const float* sp = state_ptr(slot);
#pragma unroll
    for (int j = 0; j < kRows; ++j) S[j] = (has_init && dv_ok) ? sp[kTransposed ? j : int64_t(j) * p.head_dim_v] : 0.0f;
  }
  const auto store_state = [&](int64_t dst) {
    if (!dv_ok) return;
    float* sp = state_ptr(dst);
#pragma unroll
    for (int j = 0; j < kRows; ++j) sp[kTransposed ? j : int64_t(j) * p.head_dim_v] = S[j];
  };
  const float* ksr = ks + row0;
  const float* qsr = qs + row0;
  [[maybe_unused]] const float* dsr = ds + row0;

  for (int t = 0; t < n; ++t) {
    const int64_t tok = int64_t(seq_start) + t;
    float ssk = 0.0f, ssq = 0.0f;
    for (int i = lane; i < kDk; i += 32) {
      const float kf = to_float(k[tok * p.k_token_stride + i]), qf = to_float(q[tok * p.q_token_stride + i]);
      ks[i] = kf;
      qs[i] = qf;
      ssk = __fmaf_rn(kf, kf, ssk);
      ssq = __fmaf_rn(qf, qf, ssq);
      if constexpr (kDecay == DecayGranularity::kPerChannel) ds[i] = __expf(p.g[(tok * p.num_v_heads + hv) * kDk + i]);
    }
    float inv_k = 1.0f, inv_q = p.scale;
    if (p.use_qk_l2norm) {
      inv_k = rsqrtf(group_sum<32>(ssk) + kL2Eps);
      inv_q = rsqrtf(group_sum<32>(ssq) + kL2Eps) * p.scale;
    }
    for (int i = lane; i < kDk; i += 32) {
      ks[i] *= inv_k;
      qs[i] *= inv_q;
    }
    __syncwarp();
    const float beta = p.beta[tok * p.num_v_heads + hv];
    const float vv = dv_ok ? to_float(v[tok * p.v_token_stride]) : 0.0f;
    float kv = 0.0f;
    if constexpr (kDecay == DecayGranularity::kPerHead) {
      const float decay = __expf(p.g[tok * p.num_v_heads + hv]);
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
        S[j] *= decay;
        kv = __fmaf_rn(S[j], ksr[j], kv);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
        S[j] *= dsr[j];
        kv = __fmaf_rn(S[j], ksr[j], kv);
      }
    }
    if constexpr (kSplit == 2) kv += __shfl_xor_sync(0xffffffffu, kv, kCols);
    const float delta = beta * (vv - kv);
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      S[j] = __fmaf_rn(ksr[j], delta, S[j]);
      acc = __fmaf_rn(S[j], qsr[j], acc);
    }
    if constexpr (kSplit == 2) acc += __shfl_xor_sync(0xffffffffu, acc, kCols);
    if (dv_ok && part == 0) o[tok * p.o_token_stride] = from_float<T>(acc);
    const int count = t + 1;
    if (p.snapshot_slots && p.snapshot_interval > 0 && count % p.snapshot_interval == 0) {
      const int j = count / p.snapshot_interval - 1;
      if (j < p.max_snapshots) {
        const int64_t dst = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
        if (dst >= 0 && dst < p.num_slots) store_state(dst);
      }
    }
    if (p.intermediate_state_slots && t < p.max_intermediate) {
      const int64_t dst = p.intermediate_state_slots[int64_t(b) * p.max_intermediate + t];
      if (dst >= 0 && dst < p.num_slots) store_state(dst);
    }
    __syncwarp();  // the staging buffer is rewritten by the next token
  }
  store_state(slot);
}
namespace {

// ---- fused_short_conv_delta_rule_decode_kernel -----------------------------------------------------------

constexpr int kFusedMaxThreads = 256;
constexpr int kFusedMaxCols = 1024;  // (Hv / Hk) * Dv, the shared staging of v and o
constexpr int kFusedMaxRatio = 8;

/** Gate pre-activations -> log decay: -exp(A_log) softplus(x), or the bounded lower_bound * sigmoid(exp(A_log) x). */
__device__ __forceinline__ float gate_log_decay(float x, float a_log, float lower_bound) {
  const float rate = __expf(a_log);
  return lower_bound < 0.0f ? lower_bound * sigmoid_f(rate * x) : -rate * softplus_f(x);
}

/**
 * Grid (Hk, B). One CTA owns key head hk of one sequence with its R = Hv / Hk value heads, so each conv channel it touches
 * (q and k of hk, v of the R heads) and each state column has a single writer and nothing needs a grid-wide barrier. Phases,
 * separated by __syncthreads: (1) short conv with SiLU and state update into shared q/k/v; (2) L2 norms of q, k; (3) gating of
 * the R heads; (4) delta-rule update of S, using the recurrent kernel's reduction order, in passes if R * Dv columns exceed
 * the block; (5) per-head RMSNorm of o times the weight and act(out_gate).
 */
}  // namespace
template <typename T, DecayGranularity kDecay, int kDk, int kWidth>
__global__ __launch_bounds__(kFusedMaxThreads, 1) void fused_short_conv_delta_rule_decode_kernel(
    const __grid_constant__ FusedDeltaDecodeParams fp) {
  constexpr int kSplit = kDk >= 128 ? 2 : 1, kCols = 32 / kSplit, kRows = kDk / kSplit, kHist = kWidth - 1;
  constexpr bool kKda = kDecay == DecayGranularity::kPerChannel;
  const DeltaRuleParams& p = fp.rule;
  // Keep broadcasts at their point of use: hoisting all 64 keys/queries/decays beside S spills the Dk=64 cell.
  __shared__ volatile float qs[kDk], ks[kDk], ds[kKda ? kFusedMaxCols : 1];
  __shared__ float vs[kFusedMaxCols], os[kFusedMaxCols];
  __shared__ float gate_s[2 * kFusedMaxRatio], inv_rms[kFusedMaxRatio], norms[2];
  float* bs = gate_s;
  [[maybe_unused]] float* gs = gate_s + kFusedMaxRatio;  // GDN only

  const int hk = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, nthreads = blockDim.x;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int dk = kDk, dv = p.head_dim_v, ratio = p.num_v_heads / p.num_k_heads, hv0 = hk * ratio, cols = ratio * dv;
  const int hk_total = p.num_k_heads;
  const T* x = static_cast<const T*>(fp.qkv) + int64_t(b) * fp.qkv_batch_stride;
  const T* cw = static_cast<const T*>(fp.conv_weight);
  T* cstate = static_cast<T*>(fp.conv_states);

  // (1) conv over the 2 Dk + R Dv channels of this CTA
  for (int idx = tid; idx < 2 * dk + cols; idx += nthreads) {
    int ch;
    volatile float* dst;
    if (idx < dk) {
      ch = hk * dk + idx;
      dst = &qs[idx];
    } else if (idx < 2 * dk) {
      ch = hk_total * dk + hk * dk + (idx - dk);
      dst = &ks[idx - dk];
    } else {
      ch = 2 * hk_total * dk + hv0 * dv + (idx - 2 * dk);
      dst = &vs[idx - 2 * dk];
    }
    T* st = cstate + slot * fp.conv_state_slot_stride + int64_t(ch) * fp.conv_state_dim_stride;
    float win[kHist];
#pragma unroll
    for (int j = 0; j < kHist; ++j) win[j] = to_float(st[int64_t(j) * fp.conv_state_len_stride]);
    const float xin = to_float(x[ch]);
    float acc = fp.conv_bias ? to_float(static_cast<const T*>(fp.conv_bias)[ch]) : 0.0f;
#pragma unroll
    for (int j = 0; j < kHist; ++j) acc = __fmaf_rn(to_float(cw[int64_t(ch) * fp.conv_weight_dim_stride + j * fp.conv_weight_width_stride]), win[j], acc);
    acc = __fmaf_rn(to_float(cw[int64_t(ch) * fp.conv_weight_dim_stride + kHist * fp.conv_weight_width_stride]), xin, acc);
    *dst = to_float(from_float<T>(silu_f(acc)));  // the same storage boundary as standalone causal_conv1d_update
#pragma unroll
    for (int j = 0; j < kHist - 1; ++j)
      st[int64_t(j) * fp.conv_state_len_stride] = st[int64_t(j + 1) * fp.conv_state_len_stride];
    st[int64_t(kHist - 1) * fp.conv_state_len_stride] = x[ch];
  }
  __syncthreads();

  // (2) L2 norms of q and k (one warp), (3) gating of the R value heads
  if (tid < 32) {
    float ssk = 0.0f, ssq = 0.0f;
    for (int i = tid; i < dk; i += 32) {
      ssk = __fmaf_rn(ks[i], ks[i], ssk);
      ssq = __fmaf_rn(qs[i], qs[i], ssq);
    }
    if (tid == 0) {
      norms[0] = 1.0f;
      norms[1] = p.scale;
    }
    ssk = group_sum<32>(ssk);
    ssq = group_sum<32>(ssq);
    if (p.use_qk_l2norm && tid == 0) {
      norms[0] = rsqrtf(ssk + kL2Eps);
      norms[1] = rsqrtf(ssq + kL2Eps) * p.scale;
    }
  }
  const float* a_log = fp.A_log;
  if constexpr (kKda) {
    for (int idx = tid; idx < ratio * dk; idx += nthreads) {
      const int j = idx / dk, hv = hv0 + j;
      const float xg = to_float(static_cast<const T*>(fp.gate_a)[int64_t(b) * fp.gate_a_batch_stride + int64_t(hv) * dk + (idx - j * dk)]) +
                       fp.dt_bias[int64_t(hv) * dk + (idx - j * dk)];
      ds[idx] = __expf(gate_log_decay(xg, a_log[hv], fp.lower_bound));
    }
  }
  if (tid < ratio) {
    const int hv = hv0 + tid;
    if constexpr (!kKda) {
      const float xg = to_float(static_cast<const T*>(fp.gate_a)[int64_t(b) * fp.gate_a_batch_stride + hv]) + fp.dt_bias[hv];
      gs[tid] = __expf(gate_log_decay(xg, a_log[hv], fp.lower_bound));
    }
    bs[tid] = sigmoid_f(to_float(static_cast<const T*>(fp.gate_b)[int64_t(b) * fp.gate_b_batch_stride + hv]));
  }
  __syncthreads();
  for (int i = tid; i < dk; i += nthreads) {
    ks[i] *= norms[0];
    qs[i] *= norms[1];
  }
  __syncthreads();

  // (4) delta-rule update; thread -> (column in pass, row half)
  const int warp = tid >> 5, lane = tid & 31, col_in_pass = warp * kCols + (lane % kCols), part = lane / kCols, row0 = part * kRows;
  const int cols_per_pass = (nthreads >> 5) * kCols;
  for (int pass_base = 0; pass_base < cols; pass_base += cols_per_pass) {
    const int cidx = pass_base + col_in_pass;
    const bool live = cidx < cols;
    const int j = live ? cidx / dv : 0, dvi = live ? cidx - j * dv : 0, hv = hv0 + j;
    const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
    const int64_t row_stride = p.state_transposed ? 1 : dv, col_stride = p.state_transposed ? dk : 1;
    float* sp = p.state_pool + slot * p.state_slot_stride + int64_t(hv) * dk * dv + int64_t(row0) * row_stride + int64_t(dvi) * col_stride;
    float S[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) S[r] = (live && has_init) ? sp[int64_t(r) * row_stride] : 0.0f;
    float kv = 0.0f;
    if constexpr (!kKda) {
      const float decay = gs[j];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        S[r] *= decay;
        kv = __fmaf_rn(S[r], ks[row0 + r], kv);
      }
    } else {
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        S[r] *= ds[j * dk + row0 + r];
        kv = __fmaf_rn(S[r], ks[row0 + r], kv);
      }
    }
    if constexpr (kSplit >= 2) kv += __shfl_xor_sync(0xffffffffu, kv, kCols);
    if constexpr (kSplit == 4) kv += __shfl_xor_sync(0xffffffffu, kv, 2 * kCols);
    const float delta = bs[j] * (vs[cidx < cols ? cidx : 0] - kv);
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      S[r] = __fmaf_rn(ks[row0 + r], delta, S[r]);
      acc = __fmaf_rn(S[r], qs[row0 + r], acc);
    }
    if constexpr (kSplit >= 2) acc += __shfl_xor_sync(0xffffffffu, acc, kCols);
    if constexpr (kSplit == 4) acc += __shfl_xor_sync(0xffffffffu, acc, 2 * kCols);
    if (live) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) sp[int64_t(r) * row_stride] = S[r];
      if (part == 0) os[cidx] = to_float(from_float<T>(acc));  // recurrent output rounds before gated RMSNorm
    }
  }
  __syncthreads();

  // (5) gated RMSNorm per value head
  for (int j = warp; j < ratio; j += (nthreads >> 5)) {
    float ssq = 0.0f;
    for (int i = lane; i < dv; i += 32) ssq = __fmaf_rn(os[j * dv + i], os[j * dv + i], ssq);
    ssq = group_sum<32>(ssq);
    if (lane == 0) inv_rms[j] = rsqrtf(ssq / float(dv) + fp.out_norm_eps);
  }
  __syncthreads();
  for (int e = tid; e < cols; e += nthreads) {
    const int j = e / dv, dvi = e - j * dv, hv = hv0 + j;
    const float gate = to_float(static_cast<const T*>(fp.out_gate)[int64_t(b) * fp.out_gate_batch_stride + int64_t(hv) * dv + dvi]);
    const float ag = fp.out_gate_act == ActFn::kSiLU ? silu_f(gate) : sigmoid_f(gate);
    static_cast<T*>(fp.out)[int64_t(b) * fp.out_batch_stride + int64_t(hv) * dv + dvi] =
        from_float<T>(os[e] * inv_rms[j] * fp.out_norm_weight[dvi] * ag);
  }
}
namespace {


// ---- ported from ayaka state/delta_rule.cu:449-469 ----
// ---- delta_rule_gating_kernel (1) ------

/** One thread per element of g: g = -exp(A_log) softplus(a + dt_bias), or lower_bound sigmoid(exp(A_log)(a + dt_bias)) when
 * lower_bound < 0; beta = sigmoid(b) from the element with d = 0. */
}  // namespace
template <typename T, DecayGranularity kDecay>
__global__ __launch_bounds__(256) void delta_rule_gating_kernel(const __grid_constant__ DeltaGatingParams p) {
  constexpr bool kPer = kDecay == DecayGranularity::kPerChannel;
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = int64_t(p.total_tokens) * p.num_v_heads * (kPer ? p.head_dim_k : 1);
  if (idx >= total) return;
  const int d = kPer ? int(idx % p.head_dim_k) : 0;
  const int64_t th = kPer ? idx / p.head_dim_k : idx;
  const int h = int(th % p.num_v_heads);
  const int64_t t = th / p.num_v_heads;
  const float x = to_float(static_cast<const T*>(p.a)[t * p.a_token_stride + int64_t(h) * p.a_head_stride + d]) +
                  (kPer ? p.dt_bias[int64_t(h) * p.head_dim_k + d] : p.dt_bias[h]);
  const float rate = __expf(p.A_log[h]);
  p.g[idx] = p.lower_bound < 0.0f ? p.lower_bound * sigmoid_f(rate * x) : -rate * softplus_f(x);
  if (d == 0) p.beta[th] = sigmoid_f(to_float(static_cast<const T*>(p.b)[t * p.b_token_stride + h]));
}
namespace {


}  // namespace

// ---- host launchers ----------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

Status launch_status(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

template <typename F>
void by_dtype(DType dtype, F&& f) {
  if (dtype == DType::kFloat32) f(float{});
  else if (dtype == DType::kFloat16) f(__half{});
  else f(__nv_bfloat16{});
}

Status check_backend_dtype(const char* fn, const LaunchContext& ctx, DType dtype) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", fn);
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: activations must be fp32, fp16 or bf16", fn);
  }
  return Status::kSuccess;
}

/// Head geometry, state pool and slot tables of a DeltaRuleParams (shared with delta_rule_chunk.cu's rules).
Status check_rule(const char* fn, const DeltaRuleParams& p) {
  if (p.batch_size < 0 || p.batch_size > 65535 || p.num_k_heads < 1 || p.num_v_heads < 1 ||
      p.num_v_heads > 65535 || p.num_v_heads % p.num_k_heads != 0 || p.num_slots < 0 || p.total_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B, Hv <= 65535, Hk | Hv)", fn);
  }
  if (p.decay != DecayGranularity::kPerHead && p.decay != DecayGranularity::kPerChannel) {
    return detail::set_error(Status::kInvalidArgument, "%s: unknown decay granularity", fn);
  }
  if (!std::isfinite(p.scale)) return detail::set_error(Status::kInvalidArgument, "%s: scale must be finite", fn);
  if (p.state_slot_stride < int64_t(p.num_v_heads) * p.head_dim_k * p.head_dim_v) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: state slots overlap (slot stride)", fn);
  }
  if (p.snapshot_slots ? (p.snapshot_interval <= 0 || p.max_snapshots < 0) : p.snapshot_interval != 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: snapshot_slots needs snapshot_interval > 0", fn);
  }
  if (p.intermediate_state_slots && p.max_intermediate < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: max_intermediate < 0", fn);
  }
  return Status::kSuccess;
}

template <typename T, DecayGranularity kDecay, int kDk, bool kTransposed>
void launch_recurrent_impl(const DeltaRuleParams& p, cudaStream_t stream) {
  constexpr int kCols = 32 / (kDk >= 128 ? 2 : 1);
  const dim3 grid(static_cast<unsigned>((p.head_dim_v + kCols * kRecWarps - 1) / (kCols * kRecWarps)),
                  static_cast<unsigned>(p.num_v_heads), static_cast<unsigned>(p.batch_size));
  fused_recurrent_delta_rule_kernel<T, kDecay, kDk, kTransposed><<<grid, kRecWarps * 32, 0, stream>>>(p);
}

template <typename T, DecayGranularity kDecay, bool kTransposed>
void dispatch_recurrent_dk(const DeltaRuleParams& p, cudaStream_t stream) {
  switch (p.head_dim_k) {
    case 16: launch_recurrent_impl<T, kDecay, 16, kTransposed>(p, stream); break;
    case 32: launch_recurrent_impl<T, kDecay, 32, kTransposed>(p, stream); break;
    case 64: launch_recurrent_impl<T, kDecay, 64, kTransposed>(p, stream); break;
    default: launch_recurrent_impl<T, kDecay, 128, kTransposed>(p, stream); break;  // validated
  }
}

template <typename T>
void dispatch_recurrent(const DeltaRuleParams& p, cudaStream_t stream) {
  if (p.decay == DecayGranularity::kPerHead) {
    if (p.state_transposed) dispatch_recurrent_dk<T, DecayGranularity::kPerHead, true>(p, stream);
    else dispatch_recurrent_dk<T, DecayGranularity::kPerHead, false>(p, stream);
  } else {
    if (p.state_transposed) dispatch_recurrent_dk<T, DecayGranularity::kPerChannel, true>(p, stream);
    else dispatch_recurrent_dk<T, DecayGranularity::kPerChannel, false>(p, stream);
  }
}

template <typename T, DecayGranularity kDecay, int kDk, int kWidth>
void launch_fused_impl(const FusedDeltaDecodeParams& fp, cudaStream_t stream) {
  constexpr int kSplit = kDk >= 128 ? 2 : 1;
  const DeltaRuleParams& p = fp.rule;
  const int ratio = p.num_v_heads / p.num_k_heads;
  const int cols_wanted = ratio * p.head_dim_v;
  constexpr int kCols = 32 / kSplit;
  const int cols_cap = kFusedMaxThreads / kSplit;
  const int cols_block = std::min(cols_wanted, cols_cap);
  const int threads = ((cols_block + kCols - 1) / kCols) * 32;
  const dim3 grid(static_cast<unsigned>(p.num_k_heads), static_cast<unsigned>(p.batch_size));
  fused_short_conv_delta_rule_decode_kernel<T, kDecay, kDk, kWidth><<<grid, threads, 0, stream>>>(fp);
}

template <typename T, DecayGranularity kDecay, int kDk>
void dispatch_fused_width(const FusedDeltaDecodeParams& fp, cudaStream_t stream) {
  switch (fp.conv_width) {
    case 2: launch_fused_impl<T, kDecay, kDk, 2>(fp, stream); break;
    case 3: launch_fused_impl<T, kDecay, kDk, 3>(fp, stream); break;
    default: launch_fused_impl<T, kDecay, kDk, 4>(fp, stream); break;  // validated
  }
}

template <typename T, DecayGranularity kDecay>
void dispatch_fused_dk(const FusedDeltaDecodeParams& fp, cudaStream_t stream) {
  if (fp.rule.head_dim_k == 64) dispatch_fused_width<T, kDecay, 64>(fp, stream);
  else dispatch_fused_width<T, kDecay, 128>(fp, stream);
}

}  // namespace

Status delta_rule_decode(const DeltaRuleParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "delta_rule_decode";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  const DeltaRuleParams& p = params;
  s = check_rule(fn, p);
  if (s != Status::kSuccess) return s;
  if (p.head_dim_k != 16 && p.head_dim_k != 32 && p.head_dim_k != 64 && p.head_dim_k != 128) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head_dim_k must be 16, 32, 64 or 128, got %d", fn,
                             p.head_dim_k);
  }
  if (p.head_dim_v < 1 || p.head_dim_v > kMaxHeadDimV) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head_dim_v must be in 1..%d", fn, kMaxHeadDimV);
  }
  if (p.batch_size == 0) return Status::kSuccess;
  if (!p.q || !p.k || !p.v || !p.g || !p.beta || !p.o || !p.cu_seqlens || !p.state_pool) {
    return detail::set_error(Status::kInvalidArgument, "%s: q, k, v, g, beta, o, cu_seqlens, state_pool required", fn);
  }
  if (p.o == p.q || p.o == p.k || p.o == p.v) {
    return detail::set_error(Status::kInvalidArgument, "%s: o must be disjoint from q, k and v", fn);
  }
  cudaStream_t stream = stream_of(ctx);
  by_dtype(dtype, [&](auto tag) { dispatch_recurrent<decltype(tag)>(p, stream); });
  return launch_status(fn);
}

Status fused_delta_decode(const FusedDeltaDecodeParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "fused_delta_decode";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  const FusedDeltaDecodeParams& fp = params;
  const DeltaRuleParams& p = fp.rule;
  s = check_rule(fn, p);
  if (s != Status::kSuccess) return s;
  const int ratio = p.num_v_heads / p.num_k_heads;
  if (p.head_dim_k != 64 && p.head_dim_k != 128) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head_dim_k must be 64 or 128, got %d", fn, p.head_dim_k);
  }
  if (p.head_dim_v < 16 || p.head_dim_v > 128 || ratio > kFusedMaxRatio || ratio * p.head_dim_v > kFusedMaxCols ||
      p.num_k_heads > 65535) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: Dv in 16..128, Hv / Hk <= %d, (Hv / Hk) Dv <= %d", fn, kFusedMaxRatio, kFusedMaxCols);
  }
  if (fp.conv_width < 2 || fp.conv_width > 4) {
    return detail::set_error(Status::kUnsupportedShape, "%s: conv width must be 2, 3 or 4", fn);
  }
  if (fp.out_gate_act != ActFn::kSigmoid && fp.out_gate_act != ActFn::kSiLU) {
    return detail::set_error(Status::kInvalidArgument, "%s: out_gate_act must be Sigmoid or SiLU", fn);
  }
  if (!(fp.out_norm_eps > 0.0f) || !std::isfinite(fp.out_norm_eps) || fp.lower_bound > 0.0f) {
    return detail::set_error(Status::kInvalidArgument, "%s: out_norm_eps > 0, lower_bound <= 0", fn);
  }
  if (p.batch_size == 0) return Status::kSuccess;
  if (!fp.qkv || !fp.conv_weight || !fp.conv_states || !fp.gate_a || !fp.gate_b || !fp.A_log || !fp.dt_bias ||
      !fp.out_gate || !fp.out_norm_weight || !fp.out || !p.state_pool) {
    return detail::set_error(Status::kInvalidArgument, "%s: missing operand", fn);
  }
  if (fp.qkv_batch_stride < 0 || fp.conv_weight_dim_stride < 0 || fp.conv_weight_width_stride < 0 ||
      fp.conv_state_slot_stride < 0 || fp.conv_state_dim_stride < 0 || fp.conv_state_len_stride < 0 ||
      fp.gate_a_batch_stride < 0 || fp.gate_b_batch_stride < 0 || fp.out_gate_batch_stride < 0 ||
      fp.out_batch_stride < 0) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides", fn);
  }
  cudaStream_t stream = stream_of(ctx);
  by_dtype(dtype, [&](auto tag) {
    using T = decltype(tag);
    if (p.decay == DecayGranularity::kPerHead) dispatch_fused_dk<T, DecayGranularity::kPerHead>(fp, stream);
    else dispatch_fused_dk<T, DecayGranularity::kPerChannel>(fp, stream);
  });
  return launch_status(fn);
}

Status delta_rule_gating(const DeltaGatingParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "delta_rule_gating";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  const DeltaGatingParams& p = params;
  const bool per = p.decay == DecayGranularity::kPerChannel;
  if (p.total_tokens < 0 || p.num_v_heads < 0 || (per && p.head_dim_k < 1) || p.lower_bound > 0.0f ||
      p.a_token_stride < 0 || p.a_head_stride < 0 || p.b_token_stride < 0 ||
      (p.decay != DecayGranularity::kPerHead && !per)) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents, strides or lower_bound > 0", fn);
  }
  const int64_t total = int64_t(p.total_tokens) * p.num_v_heads * (per ? p.head_dim_k : 1);
  if (total == 0) return Status::kSuccess;
  if (!p.g || !p.beta || !p.a || !p.b || !p.A_log || !p.dt_bias) {
    return detail::set_error(Status::kInvalidArgument, "%s: g, beta, a, b, A_log, dt_bias required", fn);
  }
  if ((total + 255) / 256 > INT32_MAX) return detail::set_error(Status::kUnsupportedShape, "%s: too many elements", fn);
  const unsigned grid = static_cast<unsigned>((total + 255) / 256);
  cudaStream_t stream = stream_of(ctx);
  by_dtype(dtype, [&](auto tag) {
    using T = decltype(tag);
    if (per) delta_rule_gating_kernel<T, DecayGranularity::kPerChannel><<<grid, 256, 0, stream>>>(p);
    else delta_rule_gating_kernel<T, DecayGranularity::kPerHead><<<grid, 256, 0, stream>>>(p);
  });
  return launch_status(fn);
}

}  // namespace kops::state
