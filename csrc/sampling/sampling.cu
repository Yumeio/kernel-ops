// =============================================================================
// kernel-ops | csrc/sampling/sampling.cu
// B1 first implementation: row statistics of logits (S01).
//   softmax          probs = softmax(logits / T)          fp32 probabilities
//   argmax           greedy id (ties -> lowest id) + optional logprob
//   compute_logprobs chosen logprob / rank / top-n (n <= 20)
//
// Every kernel runs one CTA per row with 256 threads and reduces with a fixed
// tree (warp butterfly, then warp states merged in warp order), so a row's
// bits do not depend on the batch, the grid or neighbouring rows. The
// CTAS_PER_ROW template parameter of softmax_kernel is part of the declared
// ABI; this body implements the CTAS_PER_ROW == 1 case only, so
// sampling_workspace_requirements() reports zero device scratch. A split-row
// body would store [B, CTAS_PER_ROW, 2] fp32 (max, sum) in
// row_state_workspace and merge in a second launch; that path is not built yet.
//
// Numerical policy (shared with the rest of the family):
//   * NaN logits act as -inf.
//   * A row with +inf logits has no finite normalization: probabilities are
//     uniform over the +inf entries (matching the fp64 reference path).
//   * A row with no finite/+inf logit (all -inf/NaN) has all-zero
//     probabilities and -inf logprobs.
//   * temperature <= 0 or non-finite means greedy: a one-hot row at the
//     argmax (ties -> lowest id), the same rule SamplingParams documents.
//   * argmax ties resolve to the lowest token id; logprob is log p of the
//     winner under the row LSE (natural log).
//   * chosen_rank is 1-based: 1 + number of tokens with a strictly larger
//     logit (ties share a rank, the vLLM convention). An out-of-range chosen
//     id writes logprob -inf and rank 0. Top-n slots with no candidate
//     (top_n > V) are id INT32_MAX and logprob -inf.
// =============================================================================
#include <cmath>
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
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"

namespace kops::sample {
namespace {

constexpr int kThreads = 512;   // one CTA per row, 16 warps (softmax/argmax)
constexpr int kLogprobThreads = 256;  // top-n selection is faster at 8 warps
constexpr int kMaxWarps = 32;   // shared-state slots for blocks up to 1024 threads
constexpr int kMaxTopN = 20;    // compute_logprobs top-n cap (kernel contract)
constexpr int kMaxSoftmaxCtas = 16;  // split-row softmax cap

/// Softmax split count is a function of vocabulary only, so a row's result does
/// not depend on the batch composition (batch-invariant).
inline int softmax_ctas_for_vocab(int vocab) {
  if (vocab <= 32768) return 1;
  if (vocab <= 65536) return 2;
  if (vocab <= 131072) return 4;
  if (vocab <= 196608) return 8;
  return 16;
}

__device__ __forceinline__ float inf_pos() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float inf_neg() { return __int_as_float(0xff800000); }

/// Row statistics: the max finite logit, the sum of exp(x - m) over the finite
/// logits (online, one pass), the count of +inf logits, and the argmax.
struct RowState {
  float m;           // running max of the finite logits (NaN counted as -inf)
  float d;           // sum of exp(x - m) over the finite logits
  int32_t plus_inf;  // number of +inf logits
  float best;        // argmax value (NaN counted as -inf)
  int32_t best_index;
};

__device__ __forceinline__ RowState row_state_empty() {
  RowState s;
  s.m = inf_neg();
  s.d = 0.0f;
  s.plus_inf = 0;
  s.best = inf_neg();
  s.best_index = INT32_MAX;
  return s;
}

__device__ __forceinline__ float clean_logit(float x) { return isnan(x) ? inf_neg() : x; }

/// Argmax update: value descending, then lowest index.
__device__ __forceinline__ void row_state_add_best(RowState& s, float x, int32_t index) {
  x = clean_logit(x);
  if (x > s.best || (x == s.best && x > inf_neg() && index < s.best_index)) {
    s.best = x;
    s.best_index = index;
  }
}

/// Online max/sum update. -inf and NaN contribute nothing; +inf is counted and
/// kept out of the finite state (its distribution is uniform, see the header).
__device__ __forceinline__ void row_state_add_sum(RowState& s, float x) {
  x = clean_logit(x);
  if (x == inf_pos()) {
    ++s.plus_inf;
    return;
  }
  if (x == inf_neg()) return;
  if (x > s.m) {
    s.d = s.d * __expf(s.m - x) + 1.0f;
    s.m = x;
  } else {
    s.d += __expf(x - s.m);
  }
}

__device__ __forceinline__ RowState row_state_merge(RowState a, const RowState& b) {
  a.plus_inf += b.plus_inf;
  if (b.best > a.best || (b.best == a.best && b.best > inf_neg() && b.best_index < a.best_index)) {
    a.best = b.best;
    a.best_index = b.best_index;
  }
  if (a.m == inf_neg()) {
    a.m = b.m;
    a.d = b.d;
    return a;
  }
  if (b.m == inf_neg()) return a;
  const float m = fmaxf(a.m, b.m);
  a.d = a.d * __expf(a.m - m) + b.d * __expf(b.m - m);
  a.m = m;
  return a;
}

/// Butterfly merge over a warp; every lane holds the merged state.
__device__ __forceinline__ RowState row_state_warp_merge(RowState s) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    RowState o;
    o.m = __shfl_xor_sync(0xffffffffu, s.m, offset);
    o.d = __shfl_xor_sync(0xffffffffu, s.d, offset);
    o.plus_inf = __shfl_xor_sync(0xffffffffu, s.plus_inf, offset);
    o.best = __shfl_xor_sync(0xffffffffu, s.best, offset);
    o.best_index = __shfl_xor_sync(0xffffffffu, s.best_index, offset);
    s = row_state_merge(s, o);
  }
  return s;
}

/// Merge the states of the whole block in fixed warp order; every thread gets
/// the result. `shared` must hold kMaxWarps + 1 states.
__device__ __forceinline__ RowState row_state_block_merge(RowState s, RowState* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  s = row_state_warp_merge(s);
  if (lane == 0) shared[warp] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    RowState t = shared[0];
    for (int w = 1; w < warps; ++w) t = row_state_merge(t, shared[w]);
    shared[kMaxWarps] = t;
  }
  __syncthreads();
  return shared[kMaxWarps];
}

/// Integer sum across the block; every thread gets the result.
__device__ __forceinline__ int block_sum_i32(int v, int* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  if (warp == 0) {
    int t = threadIdx.x < warps ? shared[threadIdx.x] : 0;
    t = warp_reduce_sum(t);
    if (threadIdx.x == 0) shared[0] = t;
  }
  __syncthreads();
  return shared[0];
}

/// (value, lowest index) max across the block; every thread gets the winner.
/// `shared` must hold kMaxWarps + 1 entries of each array.
__device__ __forceinline__ void block_argmax(float& value, int32_t& index, float* shared_value,
                                             int32_t* shared_index) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  warp_argmax(value, index);
  if (lane == 0) {
    shared_value[warp] = value;
    shared_index[warp] = index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = shared_value[0];
    int32_t best_index = shared_index[0];
    for (int w = 1; w < warps; ++w) {
      const float ov = shared_value[w];
      const int32_t oi = shared_index[w];
      if (ov > best || (ov == best && oi < best_index)) {
        best = ov;
        best_index = oi;
      }
    }
    shared_value[kMaxWarps] = best;
    shared_index[kMaxWarps] = best_index;
  }
  __syncthreads();
  value = shared_value[kMaxWarps];
  index = shared_index[kMaxWarps];
}

/// log p(value) under the row state, natural log. +inf rows: uniform over the
/// +inf entries. Empty rows and -inf values: -inf.
__device__ __forceinline__ float logprob_of(float value, const RowState& s) {
  value = clean_logit(value);
  if (value == inf_neg()) return inf_neg();
  if (s.plus_inf > 0) {
    return value == inf_pos() ? static_cast<float>(-::log(static_cast<double>(s.plus_inf)))
                              : inf_neg();
  }
  if (s.m == inf_neg() || s.d <= 0.0f) return inf_neg();
  return static_cast<float>((static_cast<double>(value) - static_cast<double>(s.m)) -
                            ::log(static_cast<double>(s.d)));
}

/// A 16-byte pack load is legal only when the row base and the row stride are
/// pack aligned and the vocabulary splits into whole packs.
template <typename in_t, int VEC_SIZE>
__device__ __forceinline__ bool row_vec_ok(const in_t* __restrict__ row, const int64_t stride,
                                           const int vocab) {
  if (vocab % VEC_SIZE != 0) return false;
  const std::uintptr_t mask = static_cast<std::uintptr_t>(sizeof(in_t) * VEC_SIZE) - 1u;
  if ((reinterpret_cast<std::uintptr_t>(row) & mask) != 0u) return false;
  return (stride % VEC_SIZE) == 0;
}

template <typename in_t, int VEC_SIZE>
__device__ __forceinline__ void load_chunk(const in_t* __restrict__ row, const int64_t col,
                                           const bool wide, const int vocab,
                                           float (&v)[VEC_SIZE]) {
  if (wide) {
    const dev::Pack<in_t, VEC_SIZE> pack = dev::load_pack<VEC_SIZE>(row + col);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) v[j] = dev::to_f<in_t>(pack.x[j]);
    return;
  }
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) {
    v[j] = (col + j) < vocab ? dev::to_f<in_t>(row[col + j]) : inf_neg();
  }
}

/// Store VEC_SIZE fp32 probabilities (VEC_SIZE is 4 or 8; 8 = two float4).
template <int VEC_SIZE>
__device__ __forceinline__ void store_prob_chunk(float* __restrict__ out, const int64_t col,
                                                 const float (&p)[VEC_SIZE], const bool vectorized) {
  static_assert(VEC_SIZE == 4 || VEC_SIZE == 8, "probability store expects 4 or 8 lanes");
  if (vectorized) {
#pragma unroll
    for (int q = 0; q < VEC_SIZE / 4; ++q) {
      const float4 v = make_float4(p[4 * q + 0], p[4 * q + 1], p[4 * q + 2], p[4 * q + 3]);
      *reinterpret_cast<float4*>(out + col + 4 * q) = v;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) out[col + j] = p[j];
}

template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 4 ? 4 : 8;
}

}  // namespace

// -----------------------------------------------------------------------------
// S01 kernels
// -----------------------------------------------------------------------------

template <typename in_t, int VEC_SIZE, int CTAS_PER_ROW>
__global__ void softmax_kernel(float* __restrict__ probs, const in_t* __restrict__ logits,
                               const int64_t logits_stride, const float* __restrict__ temperature,
                               float* __restrict__ row_state_workspace, const int batch_size,
                               const int vocab_size) {
  static_assert(CTAS_PER_ROW == 1,
                "sampling: only the one-CTA-per-row softmax body is implemented");
  (void)row_state_workspace;  // unused while rows are not split
  const int row = blockIdx.x;
  if (row >= batch_size) return;

  __shared__ RowState s_state[kMaxWarps + 1];
  const in_t* __restrict__ lr = logits + static_cast<int64_t>(row) * logits_stride;
  float* __restrict__ pr = probs + static_cast<int64_t>(row) * vocab_size;

  const float t = temperature != nullptr ? temperature[row] : 1.0f;
  const bool greedy = !(t > 0.0f) || !isfinite(t);  // <= 0, NaN or +-inf
  const float scale = greedy ? 1.0f : 1.0f / t;
  const bool wide = row_vec_ok<in_t, VEC_SIZE>(lr, logits_stride, vocab_size);
  const bool vec_out =
      (vocab_size % VEC_SIZE == 0) && ((reinterpret_cast<std::uintptr_t>(pr) & 15u) == 0u);
  const int64_t nvec = vocab_size / VEC_SIZE;

  RowState s = row_state_empty();
  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(lr, c * VEC_SIZE, wide, vocab_size, v);
    const int32_t col = static_cast<int32_t>(c * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = v[j] * scale;
      // Greedy rows need the argmax only; the sum (+inf count) is used by the
      // probability pass of the other rows.
      if (greedy) {
        row_state_add_best(s, x, col + j);
      } else {
        row_state_add_sum(s, x);
      }
    }
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < vocab_size; i += blockDim.x) {
    const float x = dev::to_f<in_t>(lr[i]) * scale;
    if (greedy) {
      row_state_add_best(s, x, static_cast<int32_t>(i));
    } else {
      row_state_add_sum(s, x);
    }
  }
  s = row_state_block_merge(s, s_state);

  const bool has_mass = s.plus_inf == 0 && s.m > inf_neg() && s.d > 0.0f;
  const float inv_d = has_mass ? 1.0f / s.d : 0.0f;
  const float inv_inf = s.plus_inf > 0 ? 1.0f / static_cast<float>(s.plus_inf) : 0.0f;
  const float best = s.best;
  const int32_t best_index = s.best_index;

  auto probability = [&](float x, int32_t index) -> float {
    x = clean_logit(x);
    if (greedy) return (best > inf_neg() && index == best_index) ? 1.0f : 0.0f;
    if (s.plus_inf > 0) return x == inf_pos() ? inv_inf : 0.0f;
    if (!has_mass || x == inf_neg()) return 0.0f;
    return __expf(x - s.m) * inv_d;
  };

  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(lr, c * VEC_SIZE, wide, vocab_size, v);
    float p[VEC_SIZE];
    const int32_t col = static_cast<int32_t>(c * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) p[j] = probability(v[j] * scale, col + j);
    store_prob_chunk<VEC_SIZE>(pr, c * VEC_SIZE, p, vec_out);
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < vocab_size; i += blockDim.x) {
    pr[i] = probability(dev::to_f<in_t>(lr[i]) * scale, static_cast<int32_t>(i));
  }
}

/// Split-row phase 1: one CTA per (row, slice) writes a neutral pair
/// (max, sum) or, for greedy rows, (best_value, best_index + 1).
template <typename in_t, int VEC_SIZE, int CTAS_PER_ROW>
__global__ void softmax_stats_kernel(float* __restrict__ row_state_workspace,
                                     const in_t* __restrict__ logits, const int64_t logits_stride,
                                     const float* __restrict__ temperature, const int batch_size,
                                     const int vocab_size) {
  static_assert(CTAS_PER_ROW >= 2 && CTAS_PER_ROW <= kMaxSoftmaxCtas,
                "softmax_stats_kernel: invalid split count");
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const int cta = blockIdx.y;
  float* state = row_state_workspace + (static_cast<int64_t>(row) * CTAS_PER_ROW + cta) * 2;
  const int chunk = ((vocab_size + CTAS_PER_ROW - 1) / CTAS_PER_ROW + VEC_SIZE - 1) / VEC_SIZE *
                    VEC_SIZE;
  const int64_t begin = static_cast<int64_t>(cta) * chunk;
  const int64_t end = begin < vocab_size ? (begin + chunk < vocab_size ? begin + chunk : vocab_size)
                                         : begin;
  if (begin >= vocab_size || begin >= end) {
    if (threadIdx.x == 0) {
      state[0] = inf_neg();
      state[1] = 0.0f;
    }
    return;
  }

  const in_t* row_ptr = logits + static_cast<int64_t>(row) * logits_stride + begin;
  const int len = static_cast<int>(end - begin);
  const float t = temperature != nullptr ? temperature[row] : 1.0f;
  const bool greedy = !(t > 0.0f) || !isfinite(t);
  const float scale = greedy ? 1.0f : 1.0f / t;
  const bool wide = row_vec_ok<in_t, VEC_SIZE>(row_ptr, logits_stride, len);
  const int64_t nvec = len / VEC_SIZE;

  __shared__ RowState s_state[kMaxWarps + 1];
  RowState s = row_state_empty();
  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(row_ptr, c * VEC_SIZE, wide, len, v);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = v[j] * scale;
      if (greedy) row_state_add_best(s, x, static_cast<int32_t>(begin + c * VEC_SIZE + j));
      else row_state_add_sum(s, x);
    }
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < len; i += blockDim.x) {
    const float x = dev::to_f<in_t>(row_ptr[i]) * scale;
    if (greedy) row_state_add_best(s, x, static_cast<int32_t>(begin + i));
    else row_state_add_sum(s, x);
  }
  s = row_state_block_merge(s, s_state);
  if (threadIdx.x == 0) {
    if (greedy) {
      state[0] = s.best;
      state[1] = s.best_index == INT32_MAX ? 0.0f : static_cast<float>(s.best_index + 1);
    } else if (s.plus_inf > 0) {
      state[0] = inf_pos();
      state[1] = static_cast<float>(s.plus_inf);
    } else {
      state[0] = s.m;
      state[1] = s.d;
    }
  }
}

/// Split-row phase 2: every CTA merges the small state array itself (no third
/// launch) and writes its slice of the probability row.
template <typename in_t, int VEC_SIZE, int CTAS_PER_ROW>
__global__ void softmax_norm_kernel(float* __restrict__ probs, const in_t* __restrict__ logits,
                                    const int64_t logits_stride,
                                    const float* __restrict__ temperature,
                                    const float* __restrict__ row_state_workspace,
                                    const int batch_size, const int vocab_size) {
  static_assert(CTAS_PER_ROW >= 2 && CTAS_PER_ROW <= kMaxSoftmaxCtas,
                "softmax_norm_kernel: invalid split count");
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const int cta = blockIdx.y;
  const int chunk = ((vocab_size + CTAS_PER_ROW - 1) / CTAS_PER_ROW + VEC_SIZE - 1) / VEC_SIZE *
                    VEC_SIZE;
  const int64_t begin = static_cast<int64_t>(cta) * chunk;
  const int64_t end = begin < vocab_size ? (begin + chunk < vocab_size ? begin + chunk : vocab_size)
                                         : begin;
  if (begin >= vocab_size || begin >= end) return;

  const float t = temperature != nullptr ? temperature[row] : 1.0f;
  const bool greedy = !(t > 0.0f) || !isfinite(t);
  const float scale = greedy ? 1.0f : 1.0f / t;

  float m = inf_neg();
  float d = 0.0f;
  int32_t plus_inf = 0;
  float best = inf_neg();
  int32_t best_index = INT32_MAX;
  for (int c = 0; c < CTAS_PER_ROW; ++c) {
    const float* state = row_state_workspace + (static_cast<int64_t>(row) * CTAS_PER_ROW + c) * 2;
    if (greedy) {
      if (state[1] > 0.0f) {
        const int32_t idx = static_cast<int32_t>(state[1]) - 1;
        if (state[0] > best || (state[0] == best && idx < best_index)) {
          best = state[0];
          best_index = idx;
        }
      }
    } else if (state[0] == inf_pos()) {
      plus_inf += static_cast<int32_t>(state[1]);
    } else if (state[0] > inf_neg()) {
      if (m == inf_neg()) {
        m = state[0];
        d = state[1];
      } else if (state[0] > m) {
        d = d * __expf(m - state[0]) + state[1];
        m = state[0];
      } else {
        d += state[1] * __expf(state[0] - m);
      }
    }
  }

  const bool has_mass = !greedy && plus_inf == 0 && m > inf_neg() && d > 0.0f;
  const float inv_d = has_mass ? 1.0f / d : 0.0f;
  const float inv_inf = plus_inf > 0 ? 1.0f / static_cast<float>(plus_inf) : 0.0f;
  auto probability = [&](float x, int32_t index) -> float {
    x = clean_logit(x);
    if (greedy) return (best > inf_neg() && index == best_index) ? 1.0f : 0.0f;
    if (plus_inf > 0) return x == inf_pos() ? inv_inf : 0.0f;
    if (!has_mass || x == inf_neg()) return 0.0f;
    return __expf(x - m) * inv_d;
  };

  float* row_probs = probs + static_cast<int64_t>(row) * vocab_size + begin;
  const in_t* row_logits = logits + static_cast<int64_t>(row) * logits_stride + begin;
  const int len = static_cast<int>(end - begin);
  const bool wide = row_vec_ok<in_t, VEC_SIZE>(row_logits, logits_stride, len);
  const int64_t nvec = len / VEC_SIZE;
  const bool vec_out =
      (len % VEC_SIZE == 0) && ((reinterpret_cast<std::uintptr_t>(row_probs) & 15u) == 0u);
  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(row_logits, c * VEC_SIZE, wide, len, v);
    float p[VEC_SIZE];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      p[j] = probability(v[j] * scale, static_cast<int32_t>(begin + c * VEC_SIZE + j));
    }
    store_prob_chunk<VEC_SIZE>(row_probs, c * VEC_SIZE, p, vec_out);
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < len; i += blockDim.x) {
    row_probs[i] = probability(dev::to_f<in_t>(row_logits[i]) * scale,
                               static_cast<int32_t>(begin + i));
  }
}

template <typename in_t, int VEC_SIZE>
__global__ void argmax_kernel(int32_t* __restrict__ out_ids, float* __restrict__ out_logprob,
                              const in_t* __restrict__ logits, const int64_t logits_stride,
                              const int batch_size, const int vocab_size) {
  const int row = blockIdx.x;
  if (row >= batch_size) return;

  __shared__ RowState s_state[kMaxWarps + 1];
  const in_t* __restrict__ lr = logits + static_cast<int64_t>(row) * logits_stride;
  const bool want_logprob = out_logprob != nullptr;
  const bool wide = row_vec_ok<in_t, VEC_SIZE>(lr, logits_stride, vocab_size);
  const int64_t nvec = vocab_size / VEC_SIZE;

  RowState s = row_state_empty();
  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(lr, c * VEC_SIZE, wide, vocab_size, v);
    const int32_t col = static_cast<int32_t>(c * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      row_state_add_best(s, v[j], col + j);
      if (want_logprob) row_state_add_sum(s, v[j]);
    }
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < vocab_size; i += blockDim.x) {
    const float x = dev::to_f<in_t>(lr[i]);
    row_state_add_best(s, x, static_cast<int32_t>(i));
    if (want_logprob) row_state_add_sum(s, x);
  }
  s = row_state_block_merge(s, s_state);

  if (threadIdx.x == 0) {
    out_ids[row] = s.best_index == INT32_MAX ? 0 : s.best_index;
    if (want_logprob) out_logprob[row] = logprob_of(s.best, s);
  }
}

/// One candidate of the top-n selection: logit descending, then lowest id.
struct Candidate {
  float value;
  int32_t index;
};

template <typename in_t, int VEC_SIZE, int MAX_TOP_N>
__global__ void compute_logprobs_kernel(float* __restrict__ chosen_logprob,
                                        int32_t* __restrict__ chosen_rank,
                                        int32_t* __restrict__ top_ids,
                                        float* __restrict__ top_logprobs,
                                        const in_t* __restrict__ logits,
                                        const int64_t logits_stride,
                                        const int32_t* __restrict__ chosen_ids, const int top_n,
                                        const int batch_size, const int vocab_size) {
  static_assert(MAX_TOP_N >= 1 && MAX_TOP_N <= kMaxTopN,
                "compute_logprobs top-n must be within the S01 contract");
  const int row = blockIdx.x;
  if (row >= batch_size) return;

  __shared__ RowState s_state[kMaxWarps + 1];
  __shared__ float s_value[kMaxWarps + 1];
  __shared__ int32_t s_index[kMaxWarps + 1];
  __shared__ int s_rank[kMaxWarps + 1];

  const in_t* __restrict__ lr = logits + static_cast<int64_t>(row) * logits_stride;
  const int32_t chosen = chosen_ids[row];
  const bool id_ok = chosen >= 0 && chosen < vocab_size;
  const float chosen_value = id_ok ? clean_logit(dev::to_f<in_t>(lr[chosen])) : inf_neg();
  const int n = top_n < MAX_TOP_N ? top_n : MAX_TOP_N;
  const bool track_top = top_n > 0;
  const bool wide = row_vec_ok<in_t, VEC_SIZE>(lr, logits_stride, vocab_size);
  const int64_t nvec = vocab_size / VEC_SIZE;

  Candidate local[MAX_TOP_N];
#pragma unroll
  for (int k = 0; k < MAX_TOP_N; ++k) local[k] = Candidate{inf_neg(), INT32_MAX};
  int greater = 0;
  RowState s = row_state_empty();

  // One per thread: sorted local top-n, rank count, online row statistics.
  auto consume = [&](float x, int32_t index) {
    const float value = clean_logit(x);
    row_state_add_sum(s, value);
    if (value > chosen_value) ++greater;
    if (track_top && value > inf_neg()) {
      float carried = value;
      int32_t idx = index;
#pragma unroll
      for (int k = 0; k < MAX_TOP_N; ++k) {
        if (k >= n) break;
        const bool take =
            carried > local[k].value || (carried == local[k].value && idx < local[k].index);
        if (!take) break;
        const Candidate displaced = local[k];
        local[k] = Candidate{carried, idx};
        carried = displaced.value;
        idx = displaced.index;
      }
    }
  };

  for (int64_t c = threadIdx.x; c < nvec; c += blockDim.x) {
    float v[VEC_SIZE];
    load_chunk<in_t, VEC_SIZE>(lr, c * VEC_SIZE, wide, vocab_size, v);
    const int32_t col = static_cast<int32_t>(c * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) consume(v[j], col + j);
  }
  for (int64_t i = nvec * VEC_SIZE + threadIdx.x; i < vocab_size; i += blockDim.x) {
    consume(dev::to_f<in_t>(lr[i]), static_cast<int32_t>(i));
  }

  s = row_state_block_merge(s, s_state);
  const int rank = block_sum_i32(greater, s_rank);

  if (threadIdx.x == 0) {
    chosen_logprob[row] = id_ok ? logprob_of(chosen_value, s) : inf_neg();
    if (chosen_rank != nullptr) chosen_rank[row] = id_ok ? rank + 1 : 0;
  }

  for (int t = 0; t < n; ++t) {
    float winner_value = local[0].value;
    int32_t winner_index = local[0].index;
    block_argmax(winner_value, winner_index, s_value, s_index);
    if (threadIdx.x == 0) {
      top_ids[static_cast<int64_t>(row) * n + t] = winner_index;
      top_logprobs[static_cast<int64_t>(row) * n + t] = logprob_of(winner_value, s);
    }
    // Only the thread that owns the winner removes it; an empty slot matches
    // every thread, which is harmless because every list is already empty.
    if (local[0].value == winner_value && local[0].index == winner_index) {
#pragma unroll
      for (int k = 0; k + 1 < MAX_TOP_N; ++k) local[k] = local[k + 1];
      local[MAX_TOP_N - 1] = Candidate{inf_neg(), INT32_MAX};
    }
  }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

namespace {

template <typename T>
const T* as_const(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

template <typename T>
T* as_mutable(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

size_t element_bytes(DType dtype) {
  const int bits = dtype_bits(dtype);
  return bits > 0 ? static_cast<size_t>((bits + 7) / 8) : 1u;
}

Status check_logits_tensor(const Tensor& logits, const char* name) {
  const Status s = check_tensor(logits, logits.dtype, 2, element_bytes(logits.dtype), name);
  if (s != Status::kSuccess) return s;
  if (logits.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last dim must be contiguous", name);
  }
  return Status::kSuccess;
}

Status check_batch_vocab(const Tensor& t, const char* name, int* batch, int* vocab) {
  if (t.shape[0] > INT32_MAX || t.shape[1] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: shape %lld x %lld exceeds int32", name,
                             static_cast<long long>(t.shape[0]), static_cast<long long>(t.shape[1]));
  }
  *batch = static_cast<int>(t.shape[0]);
  *vocab = static_cast<int>(t.shape[1]);
  return Status::kSuccess;
}

Status check_temperature(const Tensor& temperature, int batch) {
  if (!tensor_present(temperature)) return Status::kSuccess;
  Status s = check_tensor(temperature, DType::kFloat32, 1, 4, "temperature");
  if (s != Status::kSuccess) return s;
  if (temperature.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument, "temperature: length %lld, expected %d",
                             static_cast<long long>(temperature.shape[0]), batch);
  }
  if (batch > 1 && temperature.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "temperature: stride[0] %lld",
                             static_cast<long long>(temperature.stride[0]));
  }
  return Status::kSuccess;
}

template <typename in_t>
Status launch_softmax_split(const Tensor& probs, const Tensor& logits, const Tensor& temperature,
                            float* row_state, int batch, int vocab, int ctas,
                            const LaunchContext& ctx) {
  constexpr int VEC = vec_width<in_t>();
  const in_t* l = as_const<in_t>(logits);
  float* p = as_mutable<float>(probs);
  const float* t = tensor_present(temperature) ? as_const<float>(temperature) : nullptr;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_SOFTMAX_SPLIT(N)                                                                      \
  do {                                                                                             \
    softmax_stats_kernel<in_t, VEC, N><<<dim3(batch, N), kThreads, 0, stream>>>(                  \
        row_state, l, logits.stride[0], t, batch, vocab);                                         \
    softmax_norm_kernel<in_t, VEC, N><<<dim3(batch, N), kThreads, 0, stream>>>(                   \
        p, l, logits.stride[0], t, row_state, batch, vocab);                                      \
  } while (0)
  switch (ctas) {
    case 2: KOPS_SOFTMAX_SPLIT(2); break;
    case 4: KOPS_SOFTMAX_SPLIT(4); break;
    case 8: KOPS_SOFTMAX_SPLIT(8); break;
    case 16: KOPS_SOFTMAX_SPLIT(16); break;
    default:
      return detail::set_error(Status::kInternal, "softmax: unsupported split count %d", ctas);
  }
#undef KOPS_SOFTMAX_SPLIT
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "softmax split launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename in_t>
Status launch_softmax(const Tensor& probs, const Tensor& logits, const Tensor& temperature,
                      const Workspace& ws, int batch, int vocab, const LaunchContext& ctx) {
  constexpr int VEC = vec_width<in_t>();
  const in_t* l = as_const<in_t>(logits);
  float* p = as_mutable<float>(probs);
  const float* t = tensor_present(temperature) ? as_const<float>(temperature) : nullptr;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int ctas = softmax_ctas_for_vocab(vocab);
  if (ctas > 1) {
    const size_t need = static_cast<size_t>(batch) * ctas * 2 * sizeof(float);
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < need) {
      return detail::set_error(Status::kWorkspaceTooSmall,
                               "softmax: workspace %zu B, need %zu B for %d CTAs/row",
                               ws.device_scratch_bytes, need, ctas);
    }
    return launch_softmax_split<in_t>(probs, logits, temperature,
                                      static_cast<float*>(ws.device_scratch), batch, vocab, ctas,
                                      ctx);
  }
  softmax_kernel<in_t, VEC, 1><<<batch, kThreads, 0, stream>>>(p, l, logits.stride[0], t, nullptr,
                                                               batch, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "softmax launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename in_t>
Status launch_argmax(const Tensor& out_ids, const Tensor& out_logprob, const Tensor& logits,
                     int batch, int vocab, const LaunchContext& ctx) {
  constexpr int VEC = vec_width<in_t>();
  const bool has_logprob = tensor_present(out_logprob);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  argmax_kernel<in_t, VEC><<<batch, kThreads, 0, stream>>>(
      as_mutable<int32_t>(out_ids), has_logprob ? as_mutable<float>(out_logprob) : nullptr,
      as_const<in_t>(logits), logits.stride[0], batch, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "argmax launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename in_t, int MAX_TOP_N>
Status launch_compute_logprobs(const Tensor& chosen_logprob, const Tensor& chosen_rank,
                               const Tensor& top_ids, const Tensor& top_logprobs,
                               const Tensor& logits, const Tensor& chosen_ids, int top_n,
                               int batch, int vocab, const LaunchContext& ctx) {
  constexpr int VEC = vec_width<in_t>();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  compute_logprobs_kernel<in_t, VEC, MAX_TOP_N><<<batch, kLogprobThreads, 0, stream>>>(
      as_mutable<float>(chosen_logprob),
      tensor_present(chosen_rank) ? as_mutable<int32_t>(chosen_rank) : nullptr,
      tensor_present(top_ids) ? as_mutable<int32_t>(top_ids) : nullptr,
      tensor_present(top_logprobs) ? as_mutable<float>(top_logprobs) : nullptr,
      as_const<in_t>(logits), logits.stride[0], as_const<int32_t>(chosen_ids), top_n, batch, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "compute_logprobs launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename in_t>
Status dispatch_compute_logprobs(const Tensor& chosen_logprob, const Tensor& chosen_rank,
                                 const Tensor& top_ids, const Tensor& top_logprobs,
                                 const Tensor& logits, const Tensor& chosen_ids, int top_n,
                                 int batch, int vocab, const LaunchContext& ctx) {
  if (top_n > 0) {
    return launch_compute_logprobs<in_t, kMaxTopN>(chosen_logprob, chosen_rank, top_ids,
                                                   top_logprobs, logits, chosen_ids, top_n, batch,
                                                   vocab, ctx);
  }
  return launch_compute_logprobs<in_t, 1>(chosen_logprob, chosen_rank, top_ids, top_logprobs,
                                          logits, chosen_ids, 0, batch, vocab, ctx);
}

}  // namespace

// -----------------------------------------------------------------------------
// S01 public API
// -----------------------------------------------------------------------------

Status sampling_workspace_requirements(int32_t max_batch, int32_t max_vocab, int32_t ctas_per_row,
                                       WorkspaceRequirements* out) {
  if (out == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "sampling_workspace_requirements: null out");
  }
  if (max_batch < 0 || max_vocab < 0 || ctas_per_row < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "sampling_workspace_requirements: negative size (%d, %d, %d)",
                             max_batch, max_vocab, ctas_per_row);
  }
  // Split-row softmax state: [max_batch, ctas_per_row, 2] fp32. The softmax
  // entry point may choose more CTAs than the caller requested for a large
  // vocabulary, so reserve the maximum of the request and the policy.
  const int requested = ctas_per_row > 0 ? ctas_per_row : 0;
  const int derived = softmax_ctas_for_vocab(max_vocab);
  const int ctas = requested > derived ? requested : derived;
  if (ctas > kMaxSoftmaxCtas) {
    return detail::set_error(Status::kInvalidArgument,
                             "sampling_workspace_requirements: ctas_per_row %d > %d", ctas,
                             kMaxSoftmaxCtas);
  }
  *out = WorkspaceRequirements{};
  out->device_scratch_bytes =
      static_cast<size_t>(max_batch) * static_cast<size_t>(ctas) * 2 * sizeof(float);
  out->alignment = 256;
  return Status::kSuccess;
}

Status softmax(const Tensor& probs, const Tensor& logits, const Tensor& temperature,
               const Workspace& ws, const LaunchContext& ctx) {
  Status s = check_logits_tensor(logits, "softmax.logits");
  if (s != Status::kSuccess) return s;
  s = check_tensor(probs, DType::kFloat32, 2, 4, "softmax.probs");
  if (s != Status::kSuccess) return s;
  int batch = 0;
  int vocab = 0;
  s = check_batch_vocab(logits, "softmax.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  if (probs.shape[0] != batch || probs.shape[1] != vocab) {
    return detail::set_error(Status::kInvalidArgument, "softmax: probs shape mismatch");
  }
  if (probs.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "softmax.probs: last dim must be contiguous");
  }
  if (probs.stride[0] != vocab && batch > 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "softmax.probs: row stride %lld != vocab %d (no probs_stride in the ABI)",
                             static_cast<long long>(probs.stride[0]), vocab);
  }
  s = check_temperature(temperature, batch);
  if (s != Status::kSuccess) return s;
  if (logits.dtype != DType::kFloat32 && logits.dtype != DType::kFloat16 &&
      logits.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "softmax.logits: %s",
                             dtype_name(logits.dtype));
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_softmax<float>(probs, logits, temperature, ws, batch, vocab, ctx);
    case DType::kFloat16:
      return launch_softmax<__half>(probs, logits, temperature, ws, batch, vocab, ctx);
    case DType::kBFloat16:
      return launch_softmax<__nv_bfloat16>(probs, logits, temperature, ws, batch, vocab, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "softmax.logits: %s",
                               dtype_name(logits.dtype));
  }
}

Status argmax(const Tensor& out_ids, const Tensor& out_logprob, const Tensor& logits,
              const LaunchContext& ctx) {
  Status s = check_logits_tensor(logits, "argmax.logits");
  if (s != Status::kSuccess) return s;
  int batch = 0;
  int vocab = 0;
  s = check_batch_vocab(logits, "argmax.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  s = check_tensor(out_ids, DType::kInt32, 1, 4, "argmax.out_ids");
  if (s != Status::kSuccess) return s;
  if (out_ids.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument, "argmax: out_ids length %lld, expected %d",
                             static_cast<long long>(out_ids.shape[0]), batch);
  }
  if (batch > 1 && out_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "argmax.out_ids: stride[0] %lld",
                             static_cast<long long>(out_ids.stride[0]));
  }
  if (tensor_present(out_logprob)) {
    s = check_tensor(out_logprob, DType::kFloat32, 1, 4, "argmax.out_logprob");
    if (s != Status::kSuccess) return s;
    if (out_logprob.shape[0] != batch) {
      return detail::set_error(Status::kInvalidArgument,
                               "argmax: out_logprob length %lld, expected %d",
                               static_cast<long long>(out_logprob.shape[0]), batch);
    }
    if (batch > 1 && out_logprob.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout, "argmax.out_logprob: stride[0] %lld",
                               static_cast<long long>(out_logprob.stride[0]));
    }
  }
  if (logits.dtype != DType::kFloat32 && logits.dtype != DType::kFloat16 &&
      logits.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "argmax.logits: %s",
                             dtype_name(logits.dtype));
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_argmax<float>(out_ids, out_logprob, logits, batch, vocab, ctx);
    case DType::kFloat16:
      return launch_argmax<__half>(out_ids, out_logprob, logits, batch, vocab, ctx);
    case DType::kBFloat16:
      return launch_argmax<__nv_bfloat16>(out_ids, out_logprob, logits, batch, vocab, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "argmax.logits: %s",
                               dtype_name(logits.dtype));
  }
}

Status compute_logprobs(const Tensor& chosen_logprob, const Tensor& chosen_rank,
                        const Tensor& top_ids, const Tensor& top_logprobs, const Tensor& logits,
                        const Tensor& chosen_ids, int32_t top_n, const LaunchContext& ctx) {
  if (top_n < 0 || top_n > kMaxTopN) {
    return detail::set_error(Status::kUnsupportedShape, "compute_logprobs: top_n %d not in [0, %d]",
                             top_n, kMaxTopN);
  }
  Status s = check_logits_tensor(logits, "compute_logprobs.logits");
  if (s != Status::kSuccess) return s;
  int batch = 0;
  int vocab = 0;
  s = check_batch_vocab(logits, "compute_logprobs.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  s = check_tensor(chosen_logprob, DType::kFloat32, 1, 4, "compute_logprobs.chosen_logprob");
  if (s != Status::kSuccess) return s;
  if (chosen_logprob.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_logprobs: chosen_logprob length %lld, expected %d",
                             static_cast<long long>(chosen_logprob.shape[0]), batch);
  }
  if (batch > 1 && chosen_logprob.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "compute_logprobs.chosen_logprob: stride[0] %lld",
                             static_cast<long long>(chosen_logprob.stride[0]));
  }
  s = check_tensor(chosen_ids, DType::kInt32, 1, 4, "compute_logprobs.chosen_ids");
  if (s != Status::kSuccess) return s;
  if (chosen_ids.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_logprobs: chosen_ids length %lld, expected %d",
                             static_cast<long long>(chosen_ids.shape[0]), batch);
  }
  if (batch > 1 && chosen_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "compute_logprobs.chosen_ids: stride[0] %lld",
                             static_cast<long long>(chosen_ids.stride[0]));
  }
  if (tensor_present(chosen_rank)) {
    s = check_tensor(chosen_rank, DType::kInt32, 1, 4, "compute_logprobs.chosen_rank");
    if (s != Status::kSuccess) return s;
    if (chosen_rank.shape[0] != batch) {
      return detail::set_error(Status::kInvalidArgument,
                               "compute_logprobs: chosen_rank length %lld, expected %d",
                               static_cast<long long>(chosen_rank.shape[0]), batch);
    }
    if (batch > 1 && chosen_rank.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "compute_logprobs.chosen_rank: stride[0] %lld",
                               static_cast<long long>(chosen_rank.stride[0]));
    }
  }
  const bool has_ids = tensor_present(top_ids);
  const bool has_logprobs = tensor_present(top_logprobs);
  if (top_n > 0 && (!has_ids || !has_logprobs)) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_logprobs: top_n %d needs both top_ids and top_logprobs",
                             top_n);
  }
  if (top_n == 0 && (has_ids || has_logprobs)) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_logprobs: top_n 0 with a top output tensor");
  }
  if (top_n > 0) {
    s = check_tensor(top_ids, DType::kInt32, 2, 4, "compute_logprobs.top_ids");
    if (s != Status::kSuccess) return s;
    s = check_tensor(top_logprobs, DType::kFloat32, 2, 4, "compute_logprobs.top_logprobs");
    if (s != Status::kSuccess) return s;
    if (top_ids.shape[0] != batch || top_ids.shape[1] != top_n) {
      return detail::set_error(Status::kInvalidArgument,
                               "compute_logprobs: top_ids must be [%d, %d]", batch, top_n);
    }
    if (top_logprobs.shape[0] != batch || top_logprobs.shape[1] != top_n) {
      return detail::set_error(Status::kInvalidArgument,
                               "compute_logprobs: top_logprobs must be [%d, %d]", batch, top_n);
    }
    if (top_ids.stride[1] != 1 || top_logprobs.stride[1] != 1 ||
        (batch > 1 && (top_ids.stride[0] != top_n || top_logprobs.stride[0] != top_n))) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "compute_logprobs: top outputs must be row-contiguous");
    }
  }
  if (logits.dtype != DType::kFloat32 && logits.dtype != DType::kFloat16 &&
      logits.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "compute_logprobs.logits: %s",
                             dtype_name(logits.dtype));
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return dispatch_compute_logprobs<float>(chosen_logprob, chosen_rank, top_ids, top_logprobs,
                                              logits, chosen_ids, top_n, batch, vocab, ctx);
    case DType::kFloat16:
      return dispatch_compute_logprobs<__half>(chosen_logprob, chosen_rank, top_ids, top_logprobs,
                                               logits, chosen_ids, top_n, batch, vocab, ctx);
    case DType::kBFloat16:
      return dispatch_compute_logprobs<__nv_bfloat16>(chosen_logprob, chosen_rank, top_ids,
                                                      top_logprobs, logits, chosen_ids, top_n,
                                                      batch, vocab, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "compute_logprobs.logits: %s",
                               dtype_name(logits.dtype));
  }
}

}  // namespace kops::sample
