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
//   * sample() runs the fused one-launch body for greedy, unfiltered
//     categorical, top-p-only and min-p rows; top-k keeps the probability path
//     below because the fused top-k keeps the whole tie group.
//   * argmax ties resolve to the lowest token id; logprob is log p of the
//     winner under the row LSE (natural log).
//   * chosen_rank is 1-based: 1 + number of tokens with a strictly larger
//     logit (ties share a rank, the vLLM convention). An out-of-range chosen
//     id writes logprob -inf and rank 0. Top-n slots with no candidate
//     (top_n > V) are id INT32_MAX and logprob -inf.
// =============================================================================
#include <cmath>
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
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/fused_sampler_kernels.cuh"
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
  // A 128k+ row has enough vectors to keep 32 warps busy and 1024-thread blocks
  // fill the SM when the batch is small; short rows stay at 16 warps.
  const int threads = vocab >= 65536 ? 1024 : kThreads;
  argmax_kernel<in_t, VEC><<<batch, threads, 0, stream>>>(
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
  // sample() materializes one fp32 probability row per request and, when the
  // caller leaves SamplingParams::temperature null, a per-row temperature array.
  // A softmax-only caller may ignore the extra region.
  const size_t probs = static_cast<size_t>(max_batch) * static_cast<size_t>(max_vocab) * sizeof(float);
  const size_t temp = static_cast<size_t>(max_batch) * sizeof(float);
  const size_t state =
      static_cast<size_t>(max_batch) * static_cast<size_t>(ctas) * 2 * sizeof(float);
  out->device_scratch_bytes = probs + temp + state;
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

// =============================================================================
// B6: truncation and sampling (S01/S03)
// The truncation thresholds are found with a radix search over the monotonic
// fp32 bit pattern (4 x 8 bits); every reduction is a fixed tree, so a row's
// result never depends on the batch or the grid. Probabilities are sampled with
// an exact ordered prefix scan (renormalized truncated distribution), which is
// the exact fallback the sampling header documents. The public dynamics are
// identical to rejection sampling; only the number of passes differs.
// =============================================================================
namespace {

constexpr int kRowThreads = 256;
constexpr int kRowWarps = kRowThreads / 32;

__device__ __forceinline__ uint32_t prob_key(float x) {
  return x > 0.0f ? __float_as_uint(x) : 0u;
}

__device__ __forceinline__ uint32_t logit_key(float x) {
  if (isnan(x)) x = -INFINITY;
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ uint64_t fp_mass(float p) {
  const double d = static_cast<double>(p) * 1099511627776.0;  // 2^40
  return d > 0.0 ? static_cast<uint64_t>(__double2ull_rn(d)) : 0ull;
}

template <int T>
__device__ __forceinline__ uint64_t block_exscan_u64(uint64_t v, uint64_t* sh) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  #pragma unroll
  for (int off = 1; off < T; off <<= 1) {
    const uint64_t x = (t >= off) ? sh[t - off] : 0ull;
    __syncthreads();
    sh[t] += x;
    __syncthreads();
  }
  const uint64_t r = t > 0 ? sh[t - 1] : 0ull;
  __syncthreads();
  return r;
}

/// 16-byte aligned rows with V % 4 == 0 take the float4 paths below; any other row stays scalar.
__device__ __forceinline__ bool row_vec4(const float* x, int V) {
  return (V & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0;
}

/// k-th largest key (k is 1-based), deterministic tie handling by index is done
/// by the boundary helper below. hist[256], sh[4]. sh[2] returns how many elements share the
/// returned key, so boundary_equals can skip its scan when every one of them is kept.
template <int T>
__device__ void select_kth_key(const float* x, int V, int k, bool is_logit, int* hist, int* sh,
                               uint32_t* out_key, int* out_need) {
  const int t = threadIdx.x;
  const bool vec4 = row_vec4(x, V);
  uint32_t prefix = 0;
  int rank = k;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int b = t; b < 256; b += T) hist[b] = 0;
    __syncthreads();
    // Equal digits of one thread's vector (probabilities of a row share an exponent byte) merge
    // into a single shared atomic.
    int prev = -1, run = 0;
    const auto add = [&](float v) {
      const uint32_t key = is_logit ? logit_key(v) : prob_key(v);
      int bin = static_cast<int>((key >> shift) & 255);
      if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) bin = -1;
      if (bin == prev) {
        ++run;
      } else {
        if (prev >= 0) atomicAdd(&hist[prev], run);
        prev = bin;
        run = 1;
      }
    };
    if (vec4) {
      for (int i = t; i < V / 4; i += T) {
        const float4 q = reinterpret_cast<const float4*>(x)[i];
        add(q.x);
        add(q.y);
        add(q.z);
        add(q.w);
      }
    } else {
      for (int i = t; i < V; i += T) add(x[i]);
    }
    if (prev >= 0) atomicAdd(&hist[prev], run);
    __syncthreads();
    if (t == 0) {
      int run = 0;
      int chosen = 0;
      int rem = rank;
      for (int b = 255; b >= 0; --b) {
        if (run + hist[b] >= rank) {
          chosen = b;
          rem = rank - run;
          break;
        }
        run += hist[b];
      }
      sh[0] = chosen;
      sh[1] = rem;
      sh[2] = hist[chosen];
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(sh[0]) << shift;
    rank = sh[1];
    __syncthreads();
  }
  *out_key = prefix;
  *out_need = rank;
}

/// Index of the need-th element (1-based) with key == tkey, in ascending id order.
template <int T>
__device__ int32_t boundary_equals(const float* x, int V, bool is_logit, uint32_t tkey, int need,
                                   uint64_t* scan, int* sh) {
  const int t = threadIdx.x;
  // select_kth_key left the number of elements sharing tkey in sh[2]: when `need` of them is all of
  // them, the whole tie group is kept and no index scan is needed.
  if (sh[2] == need) return INT32_MAX;
  if (t == 0) sh[0] = INT32_MAX;
  __syncthreads();
  const int chunk = (V + T - 1) / T;
  const int beg = t * chunk;
  const int end = min(V, beg + chunk);
  uint64_t cnt = 0;
  for (int i = beg; i < end; ++i) {
    const uint32_t key = is_logit ? logit_key(x[i]) : prob_key(x[i]);
    if (key == tkey) ++cnt;
  }
  const uint64_t before = block_exscan_u64<T>(cnt, scan);
  if (before < static_cast<uint64_t>(need) && static_cast<uint64_t>(need) <= before + cnt) {
    uint64_t c = before;
    for (int i = beg; i < end; ++i) {
      const uint32_t key = is_logit ? logit_key(x[i]) : prob_key(x[i]);
      if (key != tkey) continue;
      ++c;
      if (c == static_cast<uint64_t>(need)) {
        sh[0] = i;
        break;
      }
    }
  }
  __syncthreads();
  return sh[0];
}

struct RowMask {
  int use_k;
  uint32_t kkey;
  int kbound;
  int use_mp;
  float mp_thresh;
  __device__ __forceinline__ bool ok(uint32_t key, float v, int i) const {
    if (use_k && !(key > kkey || (key == kkey && i <= kbound))) return false;
    if (use_mp && !(v >= mp_thresh)) return false;
    return true;
  }
};

/// Nucleus threshold over the set selected by `mask`; frac is the target mass
/// fraction (0, 1). Deterministic fixed-point (2^40) integer accumulation.
template <int T>
__device__ void select_nucleus(const float* p, int V, const RowMask& mask, double frac,
                               uint64_t* wh, uint64_t* rh, uint64_t* scan, uint64_t* tgt,
                               int* sh, uint32_t* out_key, uint64_t* out_rem) {
  const int t = threadIdx.x;
  uint32_t prefix = 0;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int i = t; i < kRowWarps * 256; i += T) wh[i] = 0ull;
    __syncthreads();
    for (int i = t; i < V; i += T) {
      const float v = p[i];
      const uint32_t key = prob_key(v);
      if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) continue;
      if (!mask.ok(key, v, i)) continue;
      const uint64_t mass = fp_mass(v);
      if (mass == 0) continue;
      const int w = t >> 5;
      atomicAdd(reinterpret_cast<unsigned long long*>(&wh[w * 256 + ((key >> shift) & 255)]),
                static_cast<unsigned long long>(mass));
    }
    __syncthreads();
    for (int b = t; b < 256; b += T) {
      uint64_t s = 0;
      for (int w = 0; w < kRowWarps; ++w) s += wh[w * 256 + b];
      rh[b] = s;
    }
    __syncthreads();
    if (pass == 0) {
      uint64_t part = 0;
      for (int b = t; b < 256; b += T) part += rh[b];
      const uint64_t total = block_reduce_sum<uint64_t, kRowWarps>(part, scan);
      if (t == 0) tgt[0] = static_cast<uint64_t>(frac * static_cast<double>(total) + 0.5);
      __syncthreads();
    }
    if (t == 0) {
      uint64_t run = 0;
      int chosen = 0;
      uint64_t rem = tgt[0];
      for (int b = 255; b >= 0; --b) {
        if (run + rh[b] >= tgt[0]) {
          chosen = b;
          rem = tgt[0] - run;
          break;
        }
        run += rh[b];
      }
      sh[0] = chosen;
      tgt[1] = rem;
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(sh[0]) << shift;
    tgt[0] = tgt[1];
    __syncthreads();
  }
  *out_key = prefix;
  *out_rem = tgt[0];
}

/// Index of the token whose equal-key mass makes the cumulative mass reach rem.
template <int T>
__device__ int32_t boundary_equals_mass(const float* p, int V, uint32_t tkey, uint64_t rem,
                                        uint64_t* scan, int* sh) {
  const int t = threadIdx.x;
  if (t == 0) sh[0] = INT32_MAX;
  __syncthreads();
  const int chunk = (V + T - 1) / T;
  const int beg = t * chunk;
  const int end = min(V, beg + chunk);
  uint64_t mass = 0;
  for (int i = beg; i < end; ++i) {
    if (prob_key(p[i]) == tkey) mass += fp_mass(p[i]);
  }
  const uint64_t before = block_exscan_u64<T>(mass, scan);
  if (before < rem && rem <= before + mass) {
    uint64_t c = before;
    for (int i = beg; i < end; ++i) {
      if (prob_key(p[i]) != tkey) continue;
      c += fp_mass(p[i]);
      if (c >= rem) {
        sh[0] = i;
        break;
      }
    }
  }
  __syncthreads();
  return sh[0];
}

__device__ __forceinline__ bool in_nucleus(uint32_t key, int i, uint32_t tkey, int32_t bound) {
  return key > tkey || (key == tkey && i <= bound);
}

/// Candidate budget of the fast top-k path (value + index = 16 KiB of shared memory) and its
/// first-pass histogram (top 11 key bits = exponent + 2 mantissa bits).
constexpr int kTopkCand = 2048;
constexpr int kTopkBins = 2048;

/// Top-k threshold of a non-negative fp32 row in three reads instead of six: (1) an 11-bit
/// histogram finds the bin that holds the k-th largest value, (2) everything above the bin is
/// summed and the bin's elements (at most kTopkCand) are collected in shared memory, (3) the
/// caller writes the row. The pivot is picked among the candidates by rank in the total order
/// (key descending, index ascending), so the result does not depend on the collection order and
/// matches select_kth_key + boundary_equals. Returns false (uniformly) when the bin is too large,
/// e.g. a row dominated by zeros; the caller then takes the radix path.
///   out_key / out_bound : the k-th element's key and index (in_nucleus semantics)
///   out_sum             : sum of the kept values (block-reduced, deterministic)
template <int T>
__device__ bool topk_by_candidates(const float* x, int V, int k, int* hist, int* csum, float* cval,
                                   int* cidx, int* sh, float* red, uint32_t* out_key,
                                   int32_t* out_bound, float* out_sum) {
  const int t = threadIdx.x;
  const bool vec4 = row_vec4(x, V);
  for (int b = t; b < kTopkBins; b += T) hist[b] = 0;
  if (t == 0) sh[3] = 0;
  __syncthreads();
  int prev = -1, run = 0;
  const auto count = [&](float v) {
    const int bin = static_cast<int>(prob_key(v) >> 21);
    if (bin == prev) {
      ++run;
    } else {
      if (prev >= 0) atomicAdd(&hist[prev], run);
      prev = bin;
      run = 1;
    }
  };
  if (vec4) {
    for (int i = t; i < V / 4; i += T) {
      const float4 q = reinterpret_cast<const float4*>(x)[i];
      count(q.x);
      count(q.y);
      count(q.z);
      count(q.w);
    }
  } else {
    for (int i = t; i < V; i += T) count(x[i]);
  }
  if (prev >= 0) atomicAdd(&hist[prev], run);
  __syncthreads();
  if (t < kTopkBins / 32) {
    int s = 0;
    for (int j = 0; j < 32; ++j) s += hist[t * 32 + j];
    csum[t] = s;
  }
  __syncthreads();
  if (t == 0) {
    int above = 0, c = kTopkBins / 32 - 1;
    for (; c > 0; --c) {
      if (above + csum[c] >= k) break;
      above += csum[c];
    }
    int b = c * 32 + 31;
    for (; b > c * 32; --b) {
      if (above + hist[b] >= k) break;
      above += hist[b];
    }
    sh[0] = b;
    sh[1] = above;
    sh[2] = hist[b];
  }
  __syncthreads();
  const int bin = sh[0], above = sh[1], m = sh[2];
  if (m > kTopkCand) return false;
  float sa = 0.0f;
  const auto gather = [&](float v, int i) {
    const int d = static_cast<int>(prob_key(v) >> 21);
    if (d > bin) {
      sa += v;
    } else if (d == bin) {
      const int pos = atomicAdd(&sh[3], 1);
      cval[pos] = v;
      cidx[pos] = i;
    }
  };
  if (vec4) {
    for (int i = t; i < V / 4; i += T) {
      const float4 q = reinterpret_cast<const float4*>(x)[i];
      gather(q.x, 4 * i);
      gather(q.y, 4 * i + 1);
      gather(q.z, 4 * i + 2);
      gather(q.w, 4 * i + 3);
    }
  } else {
    for (int i = t; i < V; i += T) gather(x[i], i);
  }
  __syncthreads();
  const int need = k - above;  // 1 <= need <= m
  float sc = 0.0f;
  for (int c = t; c < m; c += T) {
    const float v = cval[c];
    const int id = cidx[c];
    const uint32_t kv = prob_key(v);
    int rank = 0;
    for (int j = 0; j < m; ++j) {
      const uint32_t kj = prob_key(cval[j]);
      rank += (kj > kv) || (kj == kv && cidx[j] < id);
    }
    if (rank < need) sc += v;
    if (rank == need - 1) {
      sh[0] = static_cast<int>(kv);
      sh[1] = id;
    }
  }
  const float total = block_reduce_sum<float, kRowWarps>(sa + sc, red);
  __syncthreads();
  *out_key = static_cast<uint32_t>(sh[0]);
  *out_bound = sh[1];
  *out_sum = total;
  __syncthreads();
  return true;
}

/// Candidate top-k threshold over logits for gumbel_max_sample(). Same 11-bit
/// histogram scheme as topk_by_candidates(), but reads in_t and orders by logit_key.
/// Returns false when the boundary bin exceeds the candidate buffer.
template <typename in_t, int VEC, int T>
__device__ bool topk_logits_by_candidates(const in_t* x, int V, int k, int* hist, int* csum,
                                          float* cval, int* cidx, int* sh, uint32_t* out_key,
                                          int32_t* out_bound) {
  const int t = threadIdx.x;
  for (int b = t; b < kTopkBins; b += T) hist[b] = 0;
  if (t == 0) sh[3] = 0;
  __syncthreads();
  int prev = -1, run = 0;
  const auto count = [&](float v) {
    const int bin = static_cast<int>(logit_key(v) >> 21);
    if (bin == prev) {
      ++run;
    } else {
      if (prev >= 0) atomicAdd(&hist[prev], run);
      prev = bin;
      run = 1;
    }
  };
  (void)VEC;
  for (int i = t; i < V; i += T) count(dev::to_f(x[i]));
  if (prev >= 0) atomicAdd(&hist[prev], run);
  __syncthreads();
  if (t < kTopkBins / 32) {
    int s = 0;
    for (int j = 0; j < 32; ++j) s += hist[t * 32 + j];
    csum[t] = s;
  }
  __syncthreads();
  if (t == 0) {
    int above = 0, c = kTopkBins / 32 - 1;
    for (; c > 0; --c) {
      if (above + csum[c] >= k) break;
      above += csum[c];
    }
    int b = c * 32 + 31;
    for (; b > c * 32; --b) {
      if (above + hist[b] >= k) break;
      above += hist[b];
    }
    sh[0] = b;
    sh[1] = above;
    sh[2] = hist[b];
  }
  __syncthreads();
  const int bin = sh[0], above = sh[1], m = sh[2];
  if (m > kTopkCand) return false;
  for (int i = t; i < V; i += T) {
    const float v = dev::to_f(x[i]);
    const int d = static_cast<int>(logit_key(v) >> 21);
    if (d == bin) {
      const int pos = atomicAdd(&sh[3], 1);
      cval[pos] = v;
      cidx[pos] = i;
    }
  }
  __syncthreads();
  const int need = k - above;
  for (int c = t; c < m; c += T) {
    const float v = cval[c];
    const int id = cidx[c];
    const uint32_t kv = logit_key(v);
    int rank = 0;
    for (int j = 0; j < m; ++j) {
      rank += (logit_key(cval[j]) > kv) || (logit_key(cval[j]) == kv && cidx[j] < id);
    }
    if (rank == need - 1) {
      sh[0] = static_cast<int>(kv);
      sh[1] = id;
    }
  }
  __syncthreads();
  *out_key = static_cast<uint32_t>(sh[0]);
  *out_bound = sh[1];
  __syncthreads();
  return true;
}

__device__ __forceinline__ float row_max(const float* p, int V, float* red) {
  float m = -INFINITY;
  for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, p[i]);
  return block_reduce_max<float, kRowWarps>(m, red);
}

__device__ __forceinline__ float row_sum(const float* p, int V, float* red) {
  float s = 0.0f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) s += p[i];
  return block_reduce_sum<float, kRowWarps>(s, red);
}

__device__ __forceinline__ int row_topk(const SamplingParams& params, int row, int vocab,
                                        int default_top_k) {
  int k = params.top_k != nullptr ? params.top_k[row] : default_top_k;
  if (k < 0) k = 0;
  if (k > vocab) k = vocab;
  return k;
}

__device__ __forceinline__ float row_top_p(const SamplingParams& params, int row,
                                           float default_top_p) {
  float p = params.top_p != nullptr ? params.top_p[row] : default_top_p;
  return p;
}

__device__ __forceinline__ float row_min_p(const SamplingParams& params, int row,
                                           float default_min_p) {
  return params.min_p != nullptr ? params.min_p[row] : default_min_p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Declared kernels
// ---------------------------------------------------------------------------

/// Top-k mask by radix select on the logit key. Three costs of the old kernel are gone:
///  * a 16-bit logit has only 16 significant key bits (bf16) or 24 (fp16, low 13 bits constant per sign),
///    so 2 / 3 digit passes replace 4;
///  * every pass reads 16-byte vectors; per-thread runs of equal digits (logits of one row share
///    their exponent byte) become one shared atomic instead of eight;
///  * the "k-th tie in index order" scan only runs when the last digit holds more rows than needed.
template <typename in_t, int VEC_SIZE>
__global__ void top_k_mask_logits_kernel(in_t* __restrict__ out, const in_t* __restrict__ logits,
                                         const int32_t* __restrict__ top_k, const int32_t default_top_k,
                                         const int batch_size, const int vocab_size) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  __shared__ uint64_t scan[256];
  constexpr int kKeyShift = sizeof(in_t) == 4 ? 0 : (std::is_same<in_t, __nv_bfloat16>::value ? 16 : 8);
  constexpr int kPasses = (32 - kKeyShift) / 8;
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const int k = top_k != nullptr ? top_k[row] : default_top_k;
  const in_t* src = logits + static_cast<int64_t>(row) * vocab_size;
  in_t* dst = out + static_cast<int64_t>(row) * vocab_size;
  const int nvec = vocab_size / VEC_SIZE;  // the host picks VEC_SIZE > 1 only when it divides vocab_size
  if (k <= 0 || k >= vocab_size) {
    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
      store_vec<in_t, VEC_SIZE>(dst + static_cast<int64_t>(v) * VEC_SIZE,
                                load_vec<in_t, VEC_SIZE>(src + static_cast<int64_t>(v) * VEC_SIZE));
    }
    return;
  }
  uint32_t tkey = 0;
  int need = 0;
  int eq_count = 0;
  {
    uint32_t prefix = 0;
    int rank = k;
    for (int pass = 0; pass < kPasses; ++pass) {
      const int shift = 8 * (kPasses - 1 - pass);
      for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
      __syncthreads();
      for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        const Vec<in_t, VEC_SIZE> x = load_vec<in_t, VEC_SIZE>(src + static_cast<int64_t>(v) * VEC_SIZE);
        int prev = -1, run = 0;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const uint32_t key = logit_key(dev::to_f(x[i])) >> kKeyShift;
          int bin = static_cast<int>((key >> shift) & 255);
          if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) bin = -1;
          if (bin == prev) {
            ++run;
          } else {
            if (prev >= 0) atomicAdd(&hist[prev], run);
            prev = bin;
            run = 1;
          }
        }
        if (prev >= 0) atomicAdd(&hist[prev], run);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int run = 0, chosen = 0, rem = rank;
        for (int b = 255; b >= 0; --b) {
          if (run + hist[b] >= rank) {
            chosen = b;
            rem = rank - run;
            break;
          }
          run += hist[b];
        }
        sh[0] = chosen;
        sh[1] = rem;
        sh[2] = hist[chosen];
      }
      __syncthreads();
      prefix |= static_cast<uint32_t>(sh[0]) << shift;
      rank = sh[1];
      eq_count = sh[2];
      __syncthreads();
    }
    tkey = prefix;
    need = rank;
  }
  int32_t boundary = INT32_MAX;
  if (eq_count > need) {  // ties straddle the k-th place: keep the first `need` of them by index
    if (threadIdx.x == 0) sh[0] = INT32_MAX;
    __syncthreads();
    const int chunk = (nvec + kRowThreads - 1) / kRowThreads;
    const int vbeg = threadIdx.x * chunk;
    const int vend = min(nvec, vbeg + chunk);
    uint64_t cnt = 0;
    for (int v = vbeg; v < vend; ++v) {
      const Vec<in_t, VEC_SIZE> x = load_vec<in_t, VEC_SIZE>(src + static_cast<int64_t>(v) * VEC_SIZE);
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) cnt += (logit_key(dev::to_f(x[i])) >> kKeyShift) == tkey;
    }
    const uint64_t before = block_exscan_u64<kRowThreads>(cnt, scan);
    if (before < static_cast<uint64_t>(need) && static_cast<uint64_t>(need) <= before + cnt) {
      uint64_t c = before;
      for (int v = vbeg; v < vend && sh[0] == INT32_MAX; ++v) {
        const Vec<in_t, VEC_SIZE> x = load_vec<in_t, VEC_SIZE>(src + static_cast<int64_t>(v) * VEC_SIZE);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          if ((logit_key(dev::to_f(x[i])) >> kKeyShift) != tkey) continue;
          if (++c == static_cast<uint64_t>(need)) {
            sh[0] = v * VEC_SIZE + i;
            break;
          }
        }
      }
    }
    __syncthreads();
    boundary = sh[0];
  }
  for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
    const int64_t base = static_cast<int64_t>(v) * VEC_SIZE;
    Vec<in_t, VEC_SIZE> x = load_vec<in_t, VEC_SIZE>(src + base);
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const uint32_t key = logit_key(dev::to_f(x[i])) >> kKeyShift;
      if (!(key > tkey || (key == tkey && base + i <= boundary))) dev::from_f(x[i], -INFINITY);
    }
    store_vec<in_t, VEC_SIZE>(dst + base, x);
  }
}

template <int VEC_SIZE>
__global__ void top_k_renorm_probs_kernel(float* __restrict__ out, const float* __restrict__ probs,
                                          const int32_t* __restrict__ top_k,
                                          const int32_t default_top_k, const int batch_size,
                                          const int vocab_size) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  __shared__ uint64_t scan[256];
  __shared__ float red[8];
  __shared__ int cbins[kTopkBins];
  __shared__ int csum[kTopkBins / 32];
  __shared__ float cval[kTopkCand];
  __shared__ int cidx[kTopkCand];
  (void)VEC_SIZE;
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const int k = top_k != nullptr ? top_k[row] : default_top_k;
  const float* src = probs + static_cast<int64_t>(row) * vocab_size;
  float* dst = out + static_cast<int64_t>(row) * vocab_size;
  if (k <= 0 || k >= vocab_size) {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) dst[i] = src[i];
    return;
  }
  const bool vec4 = row_vec4(src, vocab_size) && row_vec4(dst, vocab_size);
  uint32_t tkey = 0;
  int32_t bound = 0;
  float sum = 0.0f;
  if (!topk_by_candidates<kRowThreads>(src, vocab_size, k, cbins, csum, cval, cidx, sh, red,
                                       &tkey, &bound, &sum)) {
    int need = 0;
    select_kth_key<kRowThreads>(src, vocab_size, k, false, hist, sh, &tkey, &need);
    bound = boundary_equals<kRowThreads>(src, vocab_size, false, tkey, need, scan, sh);
    float s = 0.0f;
    if (vec4) {
      for (int v = threadIdx.x; v < vocab_size / 4; v += blockDim.x) {
        const float4 q = reinterpret_cast<const float4*>(src)[v];
        const float e[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (in_nucleus(prob_key(e[j]), 4 * v + j, tkey, bound)) s += e[j];
        }
      }
    } else {
      for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if (in_nucleus(prob_key(src[i]), i, tkey, bound)) s += src[i];
      }
    }
    sum = block_reduce_sum<float, kRowWarps>(s, red);
  }
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  if (vec4) {
    for (int v = threadIdx.x; v < vocab_size / 4; v += blockDim.x) {
      const float4 q = reinterpret_cast<const float4*>(src)[v];
      const float e[4] = {q.x, q.y, q.z, q.w};
      float4 o;
      float* po = &o.x;
      #pragma unroll
      for (int j = 0; j < 4; ++j) po[j] = in_nucleus(prob_key(e[j]), 4 * v + j, tkey, bound) ? e[j] * inv : 0.0f;
      reinterpret_cast<float4*>(dst)[v] = o;
    }
  } else {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      dst[i] = in_nucleus(prob_key(src[i]), i, tkey, bound) ? src[i] * inv : 0.0f;
    }
  }
}

template <int VEC_SIZE>
__global__ void top_p_renorm_probs_kernel(float* __restrict__ out, const float* __restrict__ probs,
                                          const float* __restrict__ top_p, const float default_top_p,
                                          const int batch_size, const int vocab_size) {
  __shared__ int hist[256];
  __shared__ uint64_t wh[kRowWarps * 256];
  __shared__ uint64_t rh[256];
  __shared__ uint64_t scan[256];
  __shared__ uint64_t tgt[2];
  __shared__ int sh[4];
  __shared__ float red[8];
  (void)VEC_SIZE;
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const float p = top_p != nullptr ? top_p[row] : default_top_p;
  const float* src = probs + static_cast<int64_t>(row) * vocab_size;
  float* dst = out + static_cast<int64_t>(row) * vocab_size;
  if (!(p > 0.0f)) {
    // p <= 0 keeps the single largest token (ties -> lowest id).
    uint32_t tkey = 0;
    int need = 0;
    select_kth_key<kRowThreads>(src, vocab_size, 1, false, hist, sh, &tkey, &need);
    const int32_t bound = boundary_equals<kRowThreads>(src, vocab_size, false, tkey, need, scan, sh);
    float s = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      if (in_nucleus(prob_key(src[i]), i, tkey, bound)) s += src[i];
    }
    const float sum = block_reduce_sum<float, kRowWarps>(s, red);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      dst[i] = in_nucleus(prob_key(src[i]), i, tkey, bound) ? src[i] * inv : 0.0f;
    }
    return;
  }
  if (p >= 1.0f) {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) dst[i] = src[i];
    return;
  }
  RowMask none{0, 0, 0, 0, 0.0f};
  uint32_t tkey = 0;
  uint64_t rem = 0;
  select_nucleus<kRowThreads>(src, vocab_size, none, static_cast<double>(p), wh, rh, scan, tgt, sh,
                              &tkey, &rem);
  const int32_t bound =
      boundary_equals_mass<kRowThreads>(src, vocab_size, tkey, rem, scan, sh);
  const bool vec4 = row_vec4(src, vocab_size) && row_vec4(dst, vocab_size);
  float s = 0.0f;
  if (vec4) {
    for (int v = threadIdx.x; v < vocab_size / 4; v += blockDim.x) {
      const float4 q = reinterpret_cast<const float4*>(src)[v];
      const float e[4] = {q.x, q.y, q.z, q.w};
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (in_nucleus(prob_key(e[j]), 4 * v + j, tkey, bound)) s += e[j];
      }
    }
  } else {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      if (in_nucleus(prob_key(src[i]), i, tkey, bound)) s += src[i];
    }
  }
  const float sum = block_reduce_sum<float, kRowWarps>(s, red);
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  if (vec4) {
    for (int v = threadIdx.x; v < vocab_size / 4; v += blockDim.x) {
      const float4 q = reinterpret_cast<const float4*>(src)[v];
      const float e[4] = {q.x, q.y, q.z, q.w};
      float4 o;
      float* po = &o.x;
      #pragma unroll
      for (int j = 0; j < 4; ++j) po[j] = in_nucleus(prob_key(e[j]), 4 * v + j, tkey, bound) ? e[j] * inv : 0.0f;
      reinterpret_cast<float4*>(dst)[v] = o;
    }
  } else {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      dst[i] = in_nucleus(prob_key(src[i]), i, tkey, bound) ? src[i] * inv : 0.0f;
    }
  }
}

template <int VEC_SIZE, int MAX_ROUNDS>
__global__ void top_k_top_p_sampling_kernel(int32_t* __restrict__ out_ids,
                                            uint8_t* __restrict__ out_valid,
                                            const float* __restrict__ probs,
                                            const SamplingParams params, const int batch_size,
                                            const int vocab_size) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  __shared__ uint64_t scan[256];
  __shared__ uint64_t wh[kRowWarps * 256];
  __shared__ uint64_t rh[256];
  __shared__ uint64_t tgt[2];
  __shared__ float red[8];
  // Candidate top-k path: one 2048-bin count histogram plus the boundary-bin
  // candidates, instead of four full-row radix passes.
  __shared__ int cbins[kTopkBins];
  __shared__ int csum[kTopkBins / 32];
  __shared__ float cval[kTopkCand];
  __shared__ int cidx[kTopkCand];
  (void)VEC_SIZE;
  (void)MAX_ROUNDS;
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const float* src = probs + static_cast<int64_t>(row) * vocab_size;
  const int k = row_topk(params, row, vocab_size, params.default_top_k);
  float tp = row_top_p(params, row, params.default_top_p);
  const float mp = row_min_p(params, row, params.default_min_p);
  const bool use_k = k > 0 && k < vocab_size;
  bool use_tp = tp > 0.0f && tp < 1.0f;
  if (!(tp > 0.0f)) {
    // top_p <= 0 keeps only the largest token; model it as top-k == 1.
    tp = 0.0f;
  }
  const bool use_mp = mp > 0.0f;
  float maxp = use_mp ? row_max(src, vocab_size, red) : 0.0f;
  uint32_t kkey = 0;
  int kbound = INT32_MAX;
  int need = 0;
  bool have_ksum = false;
  float ksum = 0.0f;
  if (use_k) {
    have_ksum = topk_by_candidates<kRowThreads>(src, vocab_size, k, cbins, csum, cval, cidx, sh,
                                                red, &kkey, &kbound, &ksum);
    if (!have_ksum) {
      select_kth_key<kRowThreads>(src, vocab_size, k, false, hist, sh, &kkey, &need);
      kbound = boundary_equals<kRowThreads>(src, vocab_size, false, kkey, need, scan, sh);
    }
  } else if (tp == 0.0f) {
    select_kth_key<kRowThreads>(src, vocab_size, 1, false, hist, sh, &kkey, &need);
    kbound = boundary_equals<kRowThreads>(src, vocab_size, false, kkey, need, scan, sh);
  }
  const float mp_thresh = use_mp ? mp * maxp : 0.0f;
  RowMask base{use_k || tp == 0.0f, kkey, kbound, use_mp, mp_thresh};
  // Fast path: with a small top-k set and non-joint truncation the nucleus is a
  // subset of the top-k candidates. Compact them (<= 64) and sample in shared
  // memory; this avoids four full-row nucleus passes and the final row scan.
  if (use_k && use_tp && !params.joint_topk_topp && k <= 64) {
    __shared__ float cp[64];
    __shared__ int32_t ci[64];
    __shared__ int cn;
    if (threadIdx.x == 0) cn = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      const uint32_t key = prob_key(src[i]);
      if (!base.ok(key, src[i], i)) continue;
      const int pos = atomicAdd(&cn, 1);
      if (pos < 64) {
        cp[pos] = src[i];
        ci[pos] = i;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int n = min(cn, 64);
      for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
          if (cp[b] > cp[a] || (cp[b] == cp[a] && ci[b] < ci[a])) {
            const float tv = cp[a];
            cp[a] = cp[b];
            cp[b] = tv;
            const int32_t ti = ci[a];
            ci[a] = ci[b];
            ci[b] = ti;
          }
        }
      }
      float mass = 0.0f;
      for (int i = 0; i < n; ++i) mass += cp[i];
      const float target = tp * mass;
      float acc = 0.0f;
      int take = 0;
      for (int i = 0; i < n; ++i) {
        acc += cp[i];
        take = i + 1;
        if (acc >= target) break;
      }
      float z = 0.0f;
      for (int i = 0; i < take; ++i) z += cp[i];
      if (n == 0 || z <= 0.0f) {
        out_ids[row] = -1;
        out_valid[row] = 0;
        return;
      }
      const int64_t offset =
          params.rng.per_request_offset != nullptr ? params.rng.per_request_offset[row] : 0;
      const uint64_t seed = params.rng.per_request_seed != nullptr
                                ? params.rng.per_request_seed[row]
                                : params.rng.global_seed;
      const float u = uniform_from_counter(seed, static_cast<uint64_t>(row),
                                           static_cast<uint64_t>(offset), 0,
                                           params.rng.stream_id);
      const float t = u * z;
      float c = 0.0f;
      int32_t id = -1;
      for (int i = 0; i < take; ++i) {
        c += cp[i];
        if (c >= t) {
          id = ci[i];
          break;
        }
      }
      out_ids[row] = id;
      out_valid[row] = id >= 0 ? 1 : 0;
    }
    return;
  }
  uint32_t pkey = 0;
  int pbound = INT32_MAX;
  if (use_tp) {
    RowMask nuc_mask = params.joint_topk_topp ? RowMask{0, 0, 0, 0, 0.0f} : base;
    uint64_t rem = 0;
    select_nucleus<kRowThreads>(src, vocab_size, nuc_mask, static_cast<double>(tp), wh, rh, scan,
                                tgt, sh, &pkey, &rem);
    pbound = boundary_equals_mass<kRowThreads>(src, vocab_size, pkey, rem, scan, sh);
  }
  const bool use_p = use_tp;
  float mass;
  if (!use_p && !use_mp && have_ksum) {
    // topk_by_candidates already summed exactly the top-k set, so the z scan is redundant.
    mass = ksum;
  } else {
    float z = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      const uint32_t key = prob_key(src[i]);
      const bool ok = base.ok(key, src[i], i) && (!use_p || in_nucleus(key, i, pkey, pbound));
      if (ok) z += src[i];
    }
    mass = block_reduce_sum<float, kRowWarps>(z, red);
  }
  if (mass <= 0.0f) {
    if (threadIdx.x == 0) {
      out_ids[row] = -1;
      out_valid[row] = 0;
    }
    return;
  }
  const int64_t offset = params.rng.per_request_offset != nullptr ? params.rng.per_request_offset[row]
                                                                 : 0;
  const uint64_t seed = params.rng.per_request_seed != nullptr ? params.rng.per_request_seed[row]
                                                               : params.rng.global_seed;
  const float u = uniform_from_counter(seed, static_cast<uint64_t>(row),
                                       static_cast<uint64_t>(offset), 0, params.rng.stream_id);
  const uint64_t target = static_cast<uint64_t>(
      static_cast<double>(u) * static_cast<double>(mass) * 1099511627776.0 + 0.5);
  const int chunk = (vocab_size + kRowThreads - 1) / kRowThreads;
  const int beg = threadIdx.x * chunk;
  const int end = min(vocab_size, beg + chunk);
  uint64_t local = 0;
  for (int i = beg; i < end; ++i) {
    const uint32_t key = prob_key(src[i]);
    const bool ok = base.ok(key, src[i], i) && (!use_p || in_nucleus(key, i, pkey, pbound));
    if (ok) local += fp_mass(src[i]);
  }
  const uint64_t before = block_exscan_u64<kRowThreads>(local, scan);
  if (threadIdx.x == 0) sh[0] = -1;
  __syncthreads();
  if (before < target && target <= before + local) {
    uint64_t c = before;
    for (int i = beg; i < end; ++i) {
      const uint32_t key = prob_key(src[i]);
      const bool ok = base.ok(key, src[i], i) && (!use_p || in_nucleus(key, i, pkey, pbound));
      if (!ok) continue;
      c += fp_mass(src[i]);
      if (c >= target) {
        sh[0] = i;
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    out_ids[row] = sh[0];
    out_valid[row] = sh[0] >= 0 ? 1 : 0;
  }
}

template <typename in_t, int VEC_SIZE, int MAX_TOPK_CANDIDATES>
__global__ void gumbel_max_sampling_kernel(int32_t* __restrict__ out_ids,
                                           const in_t* __restrict__ logits,
                                           const int64_t logits_stride, const SamplingParams params,
                                           const int batch_size, const int vocab_size) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  __shared__ uint64_t scan[256];
  __shared__ float red[8];
  __shared__ float wval[8];
  __shared__ int widx[8];
  __shared__ int cbins[kTopkBins];
  __shared__ int csum[kTopkBins / 32];
  __shared__ float cval[kTopkCand];
  __shared__ int cidx[kTopkCand];
  (void)VEC_SIZE;
  (void)MAX_TOPK_CANDIDATES;
  const int row = blockIdx.x;
  if (row >= batch_size) return;
  const in_t* src = logits + static_cast<int64_t>(row) * logits_stride;
  const float t = params.temperature != nullptr ? params.temperature[row] : params.default_temperature;
  const int k = row_topk(params, row, vocab_size, params.default_top_k);
  const float mp = row_min_p(params, row, params.default_min_p);
  const bool use_k = k > 0 && k < vocab_size;
  const bool greedy = !(t > 0.0f);
  float mx = 0.0f;
  const bool use_mp = mp > 0.0f;
  if (use_mp || greedy) {
    mx = -INFINITY;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      float v = dev::to_f(src[i]);
      if (isnan(v)) v = -INFINITY;
      mx = fmaxf(mx, v);
    }
    mx = block_reduce_max<float, kRowWarps>(mx, red);
  }
  uint32_t kkey = 0;
  int kbound = INT32_MAX;
  if (use_k) {
    if (!topk_logits_by_candidates<in_t, VEC_SIZE, kRowThreads>(
            src, vocab_size, k, cbins, csum, cval, cidx, sh, &kkey, &kbound)) {
        // Fallback: boundary bin larger than the candidate buffer, so use the exact
        // count-only radix. It reads the input one element at a time, no staging.
        uint32_t prefix = 0;
        int rank = k;
        for (int pass = 0; pass < 4; ++pass) {
          const int shift = 24 - 8 * pass;
          for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
          __syncthreads();
          for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            const uint32_t key = logit_key(dev::to_f(src[i]));
            if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) continue;
            atomicAdd(&hist[(key >> shift) & 255], 1);
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            int run = 0, chosen = 0, rem = rank;
            for (int b = 255; b >= 0; --b) {
              if (run + hist[b] >= rank) {
                chosen = b;
                rem = rank - run;
                break;
              }
              run += hist[b];
            }
            sh[0] = chosen;
            sh[1] = rem;
          }
          __syncthreads();
          prefix |= static_cast<uint32_t>(sh[0]) << shift;
          rank = sh[1];
          __syncthreads();
        }
        kkey = prefix;
        // boundary among equal logits
        if (threadIdx.x == 0) sh[0] = INT32_MAX;
        __syncthreads();
        const int chunk = (vocab_size + kRowThreads - 1) / kRowThreads;
        const int beg = threadIdx.x * chunk;
        const int end = min(vocab_size, beg + chunk);
        uint64_t cnt = 0;
        for (int i = beg; i < end; ++i) {
          if (logit_key(dev::to_f(src[i])) == kkey) ++cnt;
        }
        const uint64_t before = block_exscan_u64<kRowThreads>(cnt, scan);
        if (before < static_cast<uint64_t>(rank) && static_cast<uint64_t>(rank) <= before + cnt) {
          uint64_t c = before;
          for (int i = beg; i < end; ++i) {
            if (logit_key(dev::to_f(src[i])) != kkey) continue;
            ++c;
            if (c == static_cast<uint64_t>(rank)) {
              sh[0] = i;
              break;
            }
          }
        }
        __syncthreads();
        kbound = sh[0];
    }
  }
  const float mp_thresh = (mp > 0.0f && mx > -INFINITY) ? mx + t * logf(mp) : -INFINITY;
  float best = -INFINITY;
  int bidx = INT32_MAX;
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    float v = dev::to_f(src[i]);
    if (isnan(v)) v = -INFINITY;
    const uint32_t key = logit_key(v);
    if (use_k && !(key > kkey || (key == kkey && i <= kbound))) continue;
    if (use_mp && !(v >= mp_thresh)) continue;
    float score;
    if (greedy || mx == -INFINITY) {
      score = v;
    } else {
      const int64_t offset = params.rng.per_request_offset != nullptr
                                 ? params.rng.per_request_offset[row]
                                 : 0;
      const uint64_t seed = params.rng.per_request_seed != nullptr
                                ? params.rng.per_request_seed[row]
                                : params.rng.global_seed;
      const float g = gumbel_from_counter(seed, static_cast<uint64_t>(row),
                                          static_cast<uint64_t>(offset),
                                          static_cast<uint32_t>(i), params.rng.stream_id);
      score = v / t + g;
    }
    if (score > best || (score == best && i < bidx)) {
      best = score;
      bidx = i;
    }
  }
  warp_argmax(best, bidx);
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    wval[warp] = best;
    widx[warp] = bidx;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float b = -INFINITY;
    int bi = INT32_MAX;
    for (int w = 0; w < kRowWarps; ++w) {
      if (wval[w] > b || (wval[w] == b && widx[w] < bi)) {
        b = wval[w];
        bi = widx[w];
      }
    }
    out_ids[row] = bi == INT32_MAX ? 0 : bi;
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

__global__ void fill_temperature_kernel(float* dst, float value, int batch) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    dst[i] = value;
  }
}

Status check_row_tensor(const Tensor& t, DType dtype, const char* name, int batch, int vocab) {
  Status s = check_tensor(t, dtype, 2, dtype == DType::kFloat32 ? 4 : 2, name);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != batch || t.shape[1] != vocab) {
    return detail::set_error(Status::kInvalidArgument, "%s: shape mismatch", name);
  }
  if (t.stride[1] != 1 || (batch > 1 && t.stride[0] != vocab)) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: rows must be contiguous", name);
  }
  return Status::kSuccess;
}

}  // namespace

Status top_k_renorm_probs(const Tensor& out, const Tensor& probs, const Tensor& top_k,
                          int32_t default_top_k, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "top_k_renorm_probs: CUDA backend only");
  }
  if (!tensor_present(probs) || probs.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "top_k_renorm_probs: probs must be [B, V]");
  }
  const int batch = static_cast<int>(probs.shape[0]);
  const int vocab = static_cast<int>(probs.shape[1]);
  Status s = check_row_tensor(probs, DType::kFloat32, "top_k_renorm_probs.probs", batch, vocab);
  if (s != Status::kSuccess) return s;
  s = check_row_tensor(out, DType::kFloat32, "top_k_renorm_probs.out", batch, vocab);
  if (s != Status::kSuccess) return s;
  if (tensor_present(top_k)) {
    s = check_tensor(top_k, DType::kInt32, 1, 4, "top_k_renorm_probs.top_k");
    if (s != Status::kSuccess) return s;
    if (top_k.shape[0] != batch) {
      return detail::set_error(Status::kInvalidArgument, "top_k_renorm_probs: top_k length mismatch");
    }
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;
  top_k_renorm_probs_kernel<1><<<batch, kRowThreads, 0, stream_of(ctx)>>>(
      static_cast<float*>(out.data), static_cast<const float*>(probs.data),
      tensor_present(top_k) ? static_cast<const int32_t*>(top_k.data) : nullptr, default_top_k,
      batch, vocab);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess
                            : detail::set_backend_error(Status::kCudaError,
                                                        static_cast<int32_t>(err),
                                                        "top_k_renorm_probs: %s",
                                                        cudaGetErrorString(err));
}

Status top_p_renorm_probs(const Tensor& out, const Tensor& probs, const Tensor& top_p,
                          float default_top_p, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "top_p_renorm_probs: CUDA backend only");
  }
  if (!tensor_present(probs) || probs.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "top_p_renorm_probs: probs must be [B, V]");
  }
  const int batch = static_cast<int>(probs.shape[0]);
  const int vocab = static_cast<int>(probs.shape[1]);
  Status s = check_row_tensor(probs, DType::kFloat32, "top_p_renorm_probs.probs", batch, vocab);
  if (s != Status::kSuccess) return s;
  s = check_row_tensor(out, DType::kFloat32, "top_p_renorm_probs.out", batch, vocab);
  if (s != Status::kSuccess) return s;
  if (tensor_present(top_p)) {
    s = check_tensor(top_p, DType::kFloat32, 1, 4, "top_p_renorm_probs.top_p");
    if (s != Status::kSuccess) return s;
    if (top_p.shape[0] != batch) {
      return detail::set_error(Status::kInvalidArgument, "top_p_renorm_probs: top_p length mismatch");
    }
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;
  top_p_renorm_probs_kernel<1><<<batch, kRowThreads, 0, stream_of(ctx)>>>(
      static_cast<float*>(out.data), static_cast<const float*>(probs.data),
      tensor_present(top_p) ? static_cast<const float*>(top_p.data) : nullptr, default_top_p, batch,
      vocab);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess
                            : detail::set_backend_error(Status::kCudaError,
                                                        static_cast<int32_t>(err),
                                                        "top_p_renorm_probs: %s",
                                                        cudaGetErrorString(err));
}

Status top_k_mask_logits(const Tensor& out, const Tensor& logits, const Tensor& top_k,
                         int32_t default_top_k, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "top_k_mask_logits: CUDA backend only");
  }
  if (!tensor_present(logits) || logits.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "top_k_mask_logits: logits must be [B, V]");
  }
  const int batch = static_cast<int>(logits.shape[0]);
  const int vocab = static_cast<int>(logits.shape[1]);
  if (logits.dtype != DType::kFloat32 && logits.dtype != DType::kFloat16 &&
      logits.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "top_k_mask_logits: %s",
                             dtype_name(logits.dtype));
  }
  Status s = check_row_tensor(logits, logits.dtype, "top_k_mask_logits.logits", batch, vocab);
  if (s != Status::kSuccess) return s;
  s = check_row_tensor(out, logits.dtype, "top_k_mask_logits.out", batch, vocab);
  if (s != Status::kSuccess) return s;
  if (tensor_present(top_k)) {
    s = check_tensor(top_k, DType::kInt32, 1, 4, "top_k_mask_logits.top_k");
    if (s != Status::kSuccess) return s;
    if (top_k.shape[0] != batch) {
      return detail::set_error(Status::kInvalidArgument, "top_k_mask_logits: top_k length mismatch");
    }
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;
  const int32_t* kp = tensor_present(top_k) ? static_cast<const int32_t*>(top_k.data) : nullptr;
  cudaStream_t st = stream_of(ctx);
  const auto launch_mask = [&](auto tag) {
    using T = decltype(tag);
    constexpr int kWide = max_vec_size_v<T>;
    T* o = static_cast<T*>(out.data);
    const T* l = static_cast<const T*>(logits.data);
    if (pick_vec_size<T>(vocab, o, l) == kWide) {
      top_k_mask_logits_kernel<T, kWide><<<batch, kRowThreads, 0, st>>>(o, l, kp, default_top_k, batch, vocab);
    } else {
      top_k_mask_logits_kernel<T, 1><<<batch, kRowThreads, 0, st>>>(o, l, kp, default_top_k, batch, vocab);
    }
  };
  if (logits.dtype == DType::kFloat32) {
    launch_mask(float{});
  } else if (logits.dtype == DType::kFloat16) {
    launch_mask(__half{});
  } else {
    launch_mask(__nv_bfloat16{});
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess
                            : detail::set_backend_error(Status::kCudaError,
                                                        static_cast<int32_t>(err),
                                                        "top_k_mask_logits: %s",
                                                        cudaGetErrorString(err));
}

Status sample(const Tensor& out_ids, const Tensor& logits, const SamplingParams& params,
              const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sample: CUDA backend only");
  }
  Status s = check_logits_tensor(logits, "sample.logits");
  if (s != Status::kSuccess) return s;
  int batch = 0;
  int vocab = 0;
  s = check_batch_vocab(logits, "sample.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  s = check_tensor(out_ids, DType::kInt32, 1, 4, "sample.out_ids");
  if (s != Status::kSuccess) return s;
  if (out_ids.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument, "sample: out_ids length mismatch");
  }
  if (batch > 1 && out_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "sample: out_ids must be contiguous");
  }
  if (params.temperature == nullptr && !std::isfinite(params.default_temperature)) {
    return detail::set_error(Status::kInvalidArgument, "sample: non-finite default temperature");
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;
  const int ctas = softmax_ctas_for_vocab(vocab);
  const size_t probs_bytes = static_cast<size_t>(batch) * vocab * sizeof(float);
  const size_t temp_bytes = static_cast<size_t>(batch) * sizeof(float);
  const size_t state_bytes = static_cast<size_t>(batch) * ctas * 2 * sizeof(float);
  const size_t need = probs_bytes + temp_bytes + state_bytes;
  if (ws.device_scratch == nullptr || ws.device_scratch_bytes < need) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "sample: workspace %zu B, need %zu B", ws.device_scratch_bytes, need);
  }
  auto* base = static_cast<uint8_t*>(ws.device_scratch);
  cudaStream_t st = stream_of(ctx);
  // Fast path: the modes whose distribution and tie-breaking agree with
  // sample_fused() run its one-launch body straight from the logits, so no fp32
  // probability row is materialized. Top-k stays on the legacy path below
  // because the fused top-k keeps the whole tie group at the k-th value (the
  // declared semantic difference), and a per-row temperature tensor is not
  // equivalent for non-finite entries.
  const bool k_off = params.top_k == nullptr &&
                     !(params.default_top_k > 0 && params.default_top_k < vocab);
  const bool mp_off = params.min_p == nullptr && !(params.default_min_p > 0.0f);
  const bool all_greedy = !(params.default_temperature > 0.0f);
  // Top-k disabled: greedy, unfiltered, top-p only and min-p all share the fused
  // body. Greedy ignores min-p, so it keeps the legacy path when min-p is active
  // (a min_p > 1 row is empty there, while the greedy kernel would return the
  // argmax).
  if (params.temperature == nullptr && k_off && (mp_off || !all_greedy) &&
      !std::isnan(params.default_top_p) && !std::isnan(params.default_min_p)) {
    FusedSamplingParams fused{};
    fused.sampling = params;
    fused.default_repetition_penalty = 1.0f;
    SamplingPlan plan{};
    if (sampling_plan(logits, params, ctx, &plan) == Status::kSuccess) {
      Workspace fws = ws;
      WorkspaceRequirements freq{};
      if (plan.splits > 1 &&
          sampling_fused_workspace_requirements(batch, vocab, plan, &freq) == Status::kSuccess) {
        // sample() reserves no barrier region, but the fp32 row it would have
        // written is larger than the split exchange area in every case that
        // splits; carve both from that unused region and zero the counters.
        const size_t align = freq.alignment == 0 ? 256 : freq.alignment;
        const size_t scratch = (freq.device_scratch_bytes + align - 1) / align * align;
        if (scratch + freq.device_persistent_bytes <= probs_bytes) {
          fws.device_scratch = base;
          fws.device_scratch_bytes = freq.device_scratch_bytes;
          fws.device_persistent = base + scratch;
          fws.device_persistent_bytes = freq.device_persistent_bytes;
          const cudaError_t zero =
              cudaMemsetAsync(fws.device_persistent, 0, freq.device_persistent_bytes, st);
          if (zero != cudaSuccess) {
            return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(zero),
                                             "sample: %s", cudaGetErrorString(zero));
          }
        } else {
          plan.splits = 1;
        }
      } else if (plan.splits > 1) {
        plan.splits = 1;
      }
      const Status fused_status = sample_fused(out_ids, logits, fused, plan, fws, ctx);
      if (fused_status != Status::kPlanInvalidated) return fused_status;
    }
  }
  float* probs_ptr = reinterpret_cast<float*>(base);
  float* temp_ptr = probs_ptr + static_cast<size_t>(batch) * vocab;
  float* state_ptr = temp_ptr + batch;
  const Tensor temperature = [&] {
    if (params.temperature != nullptr) {
      Tensor t{};
      t.data = const_cast<float*>(params.temperature);
      t.dtype = DType::kFloat32;
      t.ndim = 1;
      t.device = logits.device;
      t.shape[0] = batch;
      t.stride[0] = 1;
      return t;
    }
    fill_temperature_kernel<<<(batch + 255) / 256, 256, 0, st>>>(temp_ptr,
                                                                 params.default_temperature, batch);
    Tensor t{};
    t.data = temp_ptr;
    t.dtype = DType::kFloat32;
    t.ndim = 1;
    t.device = logits.device;
    t.shape[0] = batch;
    t.stride[0] = 1;
    return t;
  }();
  Tensor probs{};
  probs.data = probs_ptr;
  probs.dtype = DType::kFloat32;
  probs.ndim = 2;
  probs.device = logits.device;
  probs.shape[0] = batch;
  probs.shape[1] = vocab;
  probs.stride[0] = vocab;
  probs.stride[1] = 1;
  Workspace sws{};
  sws.device_scratch = state_ptr;
  sws.device_scratch_bytes = state_bytes;
  s = softmax(probs, logits, temperature, sws, ctx);
  if (s != Status::kSuccess) return s;
  // out_valid is a transient row flag in the workspace; reuse the temperature
  // buffer (batch bytes) after softmax has consumed it.
  auto* valid_ptr = reinterpret_cast<uint8_t*>(temp_ptr);
  top_k_top_p_sampling_kernel<1, 8><<<batch, kRowThreads, 0, st>>>(
      static_cast<int32_t*>(out_ids.data), valid_ptr, probs_ptr, params, batch, vocab);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess
                            : detail::set_backend_error(Status::kCudaError,
                                                        static_cast<int32_t>(err),
                                                        "sample: %s", cudaGetErrorString(err));
}

Status gumbel_max_sample(const Tensor& out_ids, const Tensor& logits, const SamplingParams& params,
                         const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "gumbel_max_sample: CUDA backend only");
  }
  if (params.top_p != nullptr || params.default_top_p < 1.0f) {
    return detail::set_error(Status::kNotImplemented,
                             "gumbel_max_sample: top-p truncation needs sample() (probability path)");
  }
  Status s = check_logits_tensor(logits, "gumbel_max_sample.logits");
  if (s != Status::kSuccess) return s;
  int batch = 0;
  int vocab = 0;
  s = check_batch_vocab(logits, "gumbel_max_sample.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  s = check_tensor(out_ids, DType::kInt32, 1, 4, "gumbel_max_sample.out_ids");
  if (s != Status::kSuccess) return s;
  if (out_ids.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument, "gumbel_max_sample: out_ids length mismatch");
  }
  if (batch == 0 || vocab == 0) return Status::kSuccess;
  cudaStream_t st = stream_of(ctx);
  const int64_t stride = logits.stride[0];
  if (logits.dtype == DType::kFloat32) {
    gumbel_max_sampling_kernel<float, 1, 64><<<batch, kRowThreads, 0, st>>>(
        static_cast<int32_t*>(out_ids.data), static_cast<const float*>(logits.data), stride, params,
        batch, vocab);
  } else if (logits.dtype == DType::kFloat16) {
    gumbel_max_sampling_kernel<__half, 1, 64><<<batch, kRowThreads, 0, st>>>(
        static_cast<int32_t*>(out_ids.data), static_cast<const __half*>(logits.data), stride, params,
        batch, vocab);
  } else if (logits.dtype == DType::kBFloat16) {
    gumbel_max_sampling_kernel<__nv_bfloat16, 1, 64><<<batch, kRowThreads, 0, st>>>(
        static_cast<int32_t*>(out_ids.data), static_cast<const __nv_bfloat16*>(logits.data), stride,
        params, batch, vocab);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "gumbel_max_sample: %s",
                             dtype_name(logits.dtype));
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess
                            : detail::set_backend_error(Status::kCudaError,
                                                        static_cast<int32_t>(err),
                                                        "gumbel_max_sample: %s",
                                                        cudaGetErrorString(err));
}

}  // namespace kops::sample
