// =============================================================================
// kernel-ops | csrc/state/conv1d.cu
// B9 row 1 (L01): causal depthwise conv1d with per-slot history, as varlen prefill
// (causal_conv1d_fwd_kernel) and as decode / multi-token update (causal_conv1d_update_kernel).
//
// A thread owns kVec consecutive channels (8 half or 4 fp32 values per 16-byte access when x is
// channel-last and aligned, one channel otherwise) and slides a window of the last W inputs down
// a run of tokens, so x is read once and out written once. The taps accumulate in fp32 as bias,
// w[0] x[t-W+1], ..., w[W-1] x[t] with fused multiply-adds, which fixes the rounding independently
// of the batch composition; the output is rounded once.
//
// State semantics: the state of a slot is the last W-1 inputs of its sequence, oldest first, as
// raw storage bits (NaN payloads and signed zeros survive). Prefill chunks a sequence over CTAs;
// only the CTA of chunk 0 touches the slot (it reads the history before it writes the final state
// and the snapshots from the original history and immutable x), so chunks have no hazard.
// Contract: csrc/state/conv1d_kernels.cuh, ops.h, CONTRACTS.md §1f.
// Ported from the ayaka implementation (3rd/csrc/state/conv1d.cu).
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

using dev::silu_f;

template <typename T>
__device__ __forceinline__ float to_float(T v) { return dev::to_f(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v) {
  T r;
  dev::from_f(r, v);
  return r;
}

constexpr int kThreads = 128;
constexpr int kChunkTokens = 64;  // tokens of one sequence handled by a CTA
constexpr int kUnroll = 8;        // tokens loaded ahead of their arithmetic

template <typename T, int N>
__device__ __forceinline__ void load_channels(const T* ptr, float (&out)[N], T* raw = nullptr) {
  if constexpr (N == 1) {
    const T v = *ptr;
    out[0] = to_float(v);
    if (raw) raw[0] = v;
  } else {
    const dev::Pack<T, N> v = dev::load_pack<N>(ptr);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out[i] = to_float(v.x[i]);
      if (raw) raw[i] = v.x[i];
    }
  }
}

template <typename T, int N>
__device__ __forceinline__ void store_channels(T* ptr, const float (&in)[N]) {
  if constexpr (N == 1) {
    *ptr = from_float<T>(in[0]);
  } else {
    dev::Pack<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) v.x[i] = from_float<T>(in[i]);
    dev::store_pack<N>(ptr, v);
  }
}

}  // namespace

namespace {
// ---- ported from ayaka state/conv1d.cu:63-244 ----
template <int kWidth>
__device__ __forceinline__ float conv_taps(const float (&w)[kWidth], const float (&win)[kWidth - 1], float x, float bias,
                                           bool silu) {
  float acc = bias;
#pragma unroll
  for (int j = 0; j < kWidth - 1; ++j) acc = __fmaf_rn(w[j], win[j], acc);
  acc = __fmaf_rn(w[kWidth - 1], x, acc);
  return silu ? silu_f(acc) : acc;
}

// ---- causal_conv1d_fwd_kernel ----------------------------------------------------------------------------

/** Grid (ceil(dim / (kThreads * kVec)), ceil(max_seqlen / kChunkTokens), B). Chunk 0 of a sequence owns the slot. */
}  // namespace
template <typename T, typename W, int kWidth, int kVec>
__global__ __launch_bounds__(kThreads, 1) void causal_conv1d_fwd_kernel(const __grid_constant__ Conv1dFwdParams p) {
  constexpr int kHist = kWidth - 1;
  const int b = blockIdx.z;
  const int64_t slot = p.cache_indices ? int64_t(p.cache_indices[b]) : int64_t(b);
  const bool stateful = p.conv_states != nullptr;
  if (stateful && (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id)) return;
  const int seq_start = p.query_start_loc[b], n = p.query_start_loc[b + 1] - seq_start;
  const int t0 = blockIdx.y * kChunkTokens;
  const bool owner = blockIdx.y == 0 && stateful;  // the CTA that reads the history and writes the new states
  if (t0 >= n && !owner) return;
  const int c0 = (blockIdx.x * kThreads + threadIdx.x) * kVec;
  if (c0 >= p.dim) return;

  const T* __restrict__ x = static_cast<const T*>(p.x);
  T* __restrict__ out = static_cast<T*>(p.out);
  T* states = static_cast<T*>(p.conv_states);
  const bool has_init = stateful && p.has_initial_state && p.has_initial_state[b];
  const bool silu = p.act == ActFn::kSiLU;

  float w[kVec][kWidth], bias[kVec], win[kVec][kHist];
#pragma unroll
  for (int c = 0; c < kVec; ++c) {
    const W* wp = static_cast<const W*>(p.weight) + int64_t(c0 + c) * p.weight_dim_stride;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) w[c][j] = to_float(wp[j * p.weight_width_stride]);
    bias[c] = p.bias ? to_float(static_cast<const W*>(p.bias)[c0 + c]) : 0.0f;
  }
  // The window before token t0: inputs t0-W+1 .. t0-1, taken from x inside the sequence and from the history before it.
#pragma unroll
  for (int j = 0; j < kHist; ++j) {
    const int pos = t0 - kHist + j;
    if (pos >= 0) {
      float v[kVec];
      load_channels<T, kVec>(x + int64_t(seq_start + pos) * p.x_token_stride + int64_t(c0) * p.x_dim_stride, v);
#pragma unroll
      for (int c = 0; c < kVec; ++c) win[c][j] = v[c];
    } else {
#pragma unroll
      for (int c = 0; c < kVec; ++c)
        win[c][j] = has_init ? to_float(states[slot * p.conv_state_slot_stride + int64_t(c0 + c) * p.conv_state_dim_stride +
                                               int64_t(kHist + pos) * p.conv_state_len_stride])
                             : 0.0f;
    }
  }
  const int t_end = min(t0 + kChunkTokens, n);
  for (int t = t0; t < t_end; t += kUnroll) {
    float xin[kUnroll][kVec];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      if (t + u < t_end)
        load_channels<T, kVec>(x + int64_t(seq_start + t + u) * p.x_token_stride + int64_t(c0) * p.x_dim_stride, xin[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (t + u >= t_end) break;
      float y[kVec];
#pragma unroll
      for (int c = 0; c < kVec; ++c) {
        y[c] = conv_taps<kWidth>(w[c], win[c], xin[u][c], bias[c], silu);
#pragma unroll
        for (int j = 0; j < kHist - 1; ++j) win[c][j] = win[c][j + 1];
        win[c][kHist - 1] = xin[u][c];
      }
      store_channels<T, kVec>(out + int64_t(seq_start + t + u) * p.out_token_stride + int64_t(c0) * p.out_dim_stride, y);
    }
  }

  if (!owner) return;
  // State after `count` tokens = the last W-1 entries of [history, x[0:count]], i.e. entries count .. count+W-2 of it.
  const auto write_state = [&](int64_t dst, int count) {
#pragma unroll
    for (int l = 0; l < kHist; ++l) {
      const int64_t idx = int64_t(count) + l;
      T v[kVec];
      if (idx < kHist) {
        // Snapshots precede the final live write. Remaining initial taps move left, so the forward copy cannot
        // overwrite an unread tap. Preserve raw bits rather than round through fp32.
#pragma unroll
        for (int c = 0; c < kVec; ++c)
          v[c] = has_init ? states[slot * p.conv_state_slot_stride + int64_t(c0 + c) * p.conv_state_dim_stride +
                                  idx * p.conv_state_len_stride]
                          : from_float<T>(0.0f);
      } else {
        const T* src = x + (seq_start + idx - kHist) * p.x_token_stride + int64_t(c0) * p.x_dim_stride;
#pragma unroll
        for (int c = 0; c < kVec; ++c) v[c] = src[int64_t(c) * p.x_dim_stride];
      }
#pragma unroll
      for (int c = 0; c < kVec; ++c)
        states[dst * p.conv_state_slot_stride + int64_t(c0 + c) * p.conv_state_dim_stride + int64_t(l) * p.conv_state_len_stride] =
            v[c];
    }
  };
  if (p.snapshot_slots && p.snapshot_interval > 0) {
    for (int j = 0; j < p.max_snapshots; ++j) {
      const int64_t count = (int64_t(j) + 1) * p.snapshot_interval;
      if (count > n) break;
      const int64_t dst = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
      if (dst >= 0 && dst < p.num_slots) write_state(dst, int(count));
    }
  }
  write_state(slot, n);
}
namespace {

// ---- causal_conv1d_update_kernel -------------------------------------------------------------------------

/** Grid (ceil(dim / (kThreads * kVec)), B). A thread runs all S tokens of its channels, reading each x before it stores
 * the matching out, so out may be x. */
}  // namespace
template <typename T, typename W, int kWidth, int kVec>
__global__ __launch_bounds__(kThreads, 1) void causal_conv1d_update_kernel(const __grid_constant__ Conv1dUpdateParams p) {
  constexpr int kHist = kWidth - 1;
  const int b = blockIdx.y;
  const int64_t slot = p.cache_indices ? int64_t(p.cache_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;
  const int c0 = (blockIdx.x * kThreads + threadIdx.x) * kVec;
  if (c0 >= p.dim) return;

  const T* x = static_cast<const T*>(p.x) + int64_t(b) * p.x_batch_stride + c0;
  T* out = static_cast<T*>(p.out) + int64_t(b) * p.out_batch_stride + c0;
  T* states = static_cast<T*>(p.conv_states);
  const bool silu = p.act == ActFn::kSiLU;

  float w[kVec][kWidth], bias[kVec], win[kVec][kHist];
  T raw_win[kVec][kHist];
#pragma unroll
  for (int c = 0; c < kVec; ++c) {
    const W* wp = static_cast<const W*>(p.weight) + int64_t(c0 + c) * p.weight_dim_stride;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) w[c][j] = to_float(wp[j * p.weight_width_stride]);
    bias[c] = p.bias ? to_float(static_cast<const W*>(p.bias)[c0 + c]) : 0.0f;
#pragma unroll
    for (int j = 0; j < kHist; ++j) {
      raw_win[c][j] = states[slot * p.conv_state_slot_stride + int64_t(c0 + c) * p.conv_state_dim_stride +
                            int64_t(j) * p.conv_state_len_stride];
      win[c][j] = to_float(raw_win[c][j]);
    }
  }
  const auto write_state = [&](int64_t dst) {
#pragma unroll
    for (int c = 0; c < kVec; ++c)
#pragma unroll
      for (int j = 0; j < kHist; ++j)
        states[dst * p.conv_state_slot_stride + int64_t(c0 + c) * p.conv_state_dim_stride + int64_t(j) * p.conv_state_len_stride] =
            raw_win[c][j];
  };

  for (int i = 0; i < p.seqlen; ++i) {
    float xin[kVec], y[kVec];
    T xin_raw[kVec];
    load_channels<T, kVec>(x + int64_t(i) * p.x_token_stride, xin, xin_raw);
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
      y[c] = conv_taps<kWidth>(w[c], win[c], xin[c], bias[c], silu);
#pragma unroll
      for (int j = 0; j < kHist - 1; ++j) {
        win[c][j] = win[c][j + 1];
        raw_win[c][j] = raw_win[c][j + 1];
      }
      win[c][kHist - 1] = xin[c];
      raw_win[c][kHist - 1] = xin_raw[c];
    }
    store_channels<T, kVec>(out + int64_t(i) * p.out_token_stride, y);
    if (p.intermediate_state_slots) {
      const int64_t dst = p.intermediate_state_slots[int64_t(b) * p.seqlen + i];
      if (dst >= 0 && dst < p.num_slots && dst != p.pad_slot_id) write_state(dst);
    }
  }
  write_state(slot);
}


// ---- host launchers ----------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

Status launch_status(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool float_dtype(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

int dtype_bytes(DType t) { return t == DType::kFloat32 ? 4 : 2; }

bool aligned16(const void* ptr, int64_t elem_stride, int elem_bytes) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0 && (elem_stride * elem_bytes) % 16 == 0;
}

template <typename T>
constexpr int vec_width() { return 16 / static_cast<int>(sizeof(T)); }

template <typename T, typename W, int kWidth>
void launch_fwd(const Conv1dFwdParams& p, int64_t chunks, cudaStream_t stream, bool vectorize) {
  const auto go = [&](auto vec) {
    constexpr int kVec = decltype(vec)::value;
    const dim3 grid(static_cast<unsigned>((int64_t(p.dim) + kThreads * kVec - 1) / (kThreads * kVec)),
                    static_cast<unsigned>(chunks), static_cast<unsigned>(p.batch_size));
    causal_conv1d_fwd_kernel<T, W, kWidth, kVec><<<grid, kThreads, 0, stream>>>(p);
  };
  if (vectorize) go(std::integral_constant<int, vec_width<T>()>());
  else go(std::integral_constant<int, 1>());
}

template <typename T, typename W, int kWidth>
void launch_update(const Conv1dUpdateParams& p, cudaStream_t stream, bool vectorize) {
  const auto go = [&](auto vec) {
    constexpr int kVec = decltype(vec)::value;
    const dim3 grid(static_cast<unsigned>((int64_t(p.dim) + kThreads * kVec - 1) / (kThreads * kVec)),
                    static_cast<unsigned>(p.batch_size));
    causal_conv1d_update_kernel<T, W, kWidth, kVec><<<grid, kThreads, 0, stream>>>(p);
  };
  if (vectorize) go(std::integral_constant<int, vec_width<T>()>());
  else go(std::integral_constant<int, 1>());
}

// (activation dtype, weight dtype): weights are in the activation dtype or fp32.
#define KOPS_CONV1D_TYPES(X)                                         \
  X(DType::kFloat32, DType::kFloat32, float, float)                  \
  X(DType::kFloat16, DType::kFloat16, __half, __half)                \
  X(DType::kFloat16, DType::kFloat32, __half, float)                 \
  X(DType::kBFloat16, DType::kBFloat16, __nv_bfloat16, __nv_bfloat16) \
  X(DType::kBFloat16, DType::kFloat32, __nv_bfloat16, float)

template <typename F>
void by_width(int width, F&& f) {
  switch (width) {
    case 2: f(std::integral_constant<int, 2>()); break;
    case 3: f(std::integral_constant<int, 3>()); break;
    default: f(std::integral_constant<int, 4>()); break;
  }
}

Status check_common(const char* fn, const LaunchContext& ctx, DType dtype, DType wdtype, int width,
                    ActFn act, int64_t batch, int64_t dim) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", fn);
  }
  if (!float_dtype(dtype) || (wdtype != dtype && wdtype != DType::kFloat32)) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "%s: x must be fp32/fp16/bf16 and the weight x's dtype or fp32", fn);
  }
  if (width < 2 || width > 4) {
    return detail::set_error(Status::kUnsupportedShape, "%s: width must be 2, 3 or 4, got %d", fn,
                             width);
  }
  if (act != ActFn::kSiLU && act != ActFn::kIdentity) {
    return detail::set_error(Status::kInvalidArgument, "%s: activation must be SiLU or Identity",
                             fn);
  }
  if (batch < 0 || batch > 65535 || dim < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: batch %lld (<= 65535), dim %lld", fn,
                             static_cast<long long>(batch), static_cast<long long>(dim));
  }
  return Status::kSuccess;
}

}  // namespace

Status causal_conv1d_fwd(const Conv1dFwdParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "causal_conv1d_fwd";
  Conv1dFwdParams p = params;
  if (p.weight_dtype == DType::kInvalid) p.weight_dtype = dtype;
  Status s = check_common(fn, ctx, dtype, p.weight_dtype, p.width, p.act, p.batch_size, p.dim);
  if (s != Status::kSuccess) return s;
  if (p.total_tokens < 0 || p.max_seqlen < 0 || p.num_slots < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: negative total_tokens/max_seqlen/num_slots", fn);
  }
  if (p.batch_size == 0 || p.dim == 0) return Status::kSuccess;
  if (!p.x || !p.out || !p.weight || !p.query_start_loc) {
    return detail::set_error(Status::kInvalidArgument, "%s: x, out, weight, query_start_loc required", fn);
  }
  if (p.x == p.out) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: out must not alias x (token chunks of a sequence run in parallel)", fn);
  }
  if (p.x_token_stride < 0 || p.x_dim_stride < 0 || p.out_token_stride < 0 ||
      p.out_dim_stride < 0 || p.weight_dim_stride < 0 || p.weight_width_stride < 0 ||
      p.conv_state_slot_stride < 0 || p.conv_state_dim_stride < 0 || p.conv_state_len_stride < 0) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides", fn);
  }
  if (p.conv_states == nullptr && (p.has_initial_state || p.cache_indices || p.snapshot_slots)) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: has_initial_state/cache_indices/snapshots need conv_states", fn);
  }
  if (p.snapshot_slots ? (p.snapshot_interval <= 0 || p.max_snapshots < 0) : p.snapshot_interval != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: snapshot_slots needs snapshot_interval > 0 (and vice versa)", fn);
  }
  if (p.max_seqlen == 0) p.max_seqlen = p.total_tokens;
  const int64_t chunks = std::max<int64_t>(1, (int64_t(p.max_seqlen) + kChunkTokens - 1) / kChunkTokens);
  if (chunks > 65535) {
    return detail::set_error(Status::kUnsupportedShape, "%s: max_seqlen %d too large", fn, p.max_seqlen);
  }
  const int eb = dtype_bytes(dtype);
  // Small channel counts need active lanes more than wide accesses (128 channels would use only 16 half lanes).
  const bool vectorize = p.dim >= kThreads * (16 / eb) && p.x_dim_stride == 1 && p.out_dim_stride == 1 &&
                         p.dim % (16 / eb) == 0 && aligned16(p.x, p.x_token_stride, eb) &&
                         aligned16(p.out, p.out_token_stride, eb);
  cudaStream_t stream = stream_of(ctx);
#define X(XD, WD, XT, WT)                                                                    \
  if (dtype == XD && p.weight_dtype == WD) {                                                 \
    by_width(p.width, [&](auto w) { launch_fwd<XT, WT, decltype(w)::value>(p, chunks, stream, vectorize); }); \
  } else
  KOPS_CONV1D_TYPES(X)
#undef X
  return detail::set_error(Status::kUnsupportedDtype, "%s: unsupported dtype pair", fn);
  return launch_status(fn);
}

Status causal_conv1d_update(const Conv1dUpdateParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "causal_conv1d_update";
  Conv1dUpdateParams p = params;
  if (p.weight_dtype == DType::kInvalid) p.weight_dtype = dtype;
  Status s = check_common(fn, ctx, dtype, p.weight_dtype, p.width, p.act, p.batch_size, p.dim);
  if (s != Status::kSuccess) return s;
  if (p.seqlen < 0 || p.num_slots < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: negative seqlen/num_slots", fn);
  }
  if (p.batch_size == 0 || p.dim == 0 || p.seqlen == 0) return Status::kSuccess;
  if (!p.x || !p.out || !p.weight || !p.conv_states) {
    return detail::set_error(Status::kInvalidArgument, "%s: x, out, weight, conv_states required", fn);
  }
  if (p.x == p.out && (p.x_batch_stride != p.out_batch_stride || p.x_token_stride != p.out_token_stride)) {
    return detail::set_error(Status::kInvalidArgument, "%s: in-place out must be the same view as x", fn);
  }
  if (p.x_batch_stride < 0 || p.x_token_stride < 0 || p.out_batch_stride < 0 ||
      p.out_token_stride < 0 || p.weight_dim_stride < 0 || p.weight_width_stride < 0 ||
      p.conv_state_slot_stride < 0 || p.conv_state_dim_stride < 0 || p.conv_state_len_stride < 0) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides", fn);
  }
  const int eb = dtype_bytes(dtype);
  const bool vectorize = p.dim % (16 / eb) == 0 && aligned16(p.x, p.x_batch_stride, eb) &&
                         aligned16(p.x, p.x_token_stride, eb) && aligned16(p.out, p.out_batch_stride, eb) &&
                         aligned16(p.out, p.out_token_stride, eb);
  cudaStream_t stream = stream_of(ctx);
#define X(XD, WD, XT, WT)                                                                    \
  if (dtype == XD && p.weight_dtype == WD) {                                                 \
    by_width(p.width, [&](auto w) { launch_update<XT, WT, decltype(w)::value>(p, stream, vectorize); }); \
  } else
  KOPS_CONV1D_TYPES(X)
#undef X
  return detail::set_error(Status::kUnsupportedDtype, "%s: unsupported dtype pair", fn);
  return launch_status(fn);
}

}  // namespace kops::state
