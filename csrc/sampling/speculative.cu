// =============================================================================
// kernel-ops | csrc/sampling/speculative.cu                         Catalog: S03
// B10 speculative decoding:
//   rejection_sample_kernel          chain (MTP / EAGLE / n-gram) verification
//   tree_verify_kernel               EAGLE-2/3 tree verification
//   build_tree_attention_mask_kernel packed custom-mask builder
//
// The exact row layouts, acceptance rules and RNG draw map are documented in
// CONTRACTS.md §1g.  In short:
//   * chain: target row j predicts draft j; bonus row = total_drafts + b;
//     draft_probs == null means the deterministic-drafter (target-only) rule.
//   * tree: node 0 is the current token; retrieve_index[node] is the target row
//     *after* that node; children are verified with their parent's row.
//   * every random draw is a pure function of (seed, request, position,
//     draw_index, stream_id); no host state is read or advanced.
//
// One CTA owns one request and walks its chain/tree.  The full-vocabulary draw
// uses a fixed-order block CDF: thread t owns vectors t, t + T, ... (coalesced),
// a block inclusive scan gives every thread's interval, and the owner re-reads
// its vectors to find the token.  The order is independent of the batch, the
// grid and graph replay.  This TU adds no workspace and allocates nothing.
// =============================================================================
#include <cfloat>
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
#include "csrc/sampling/fused_sampler_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"

namespace kops::sample {
namespace {

constexpr int kSpecThreads = 1024;

__device__ __forceinline__ float clean_prob(float x) {
  // A probability outside the representable distribution contributes nothing.
  // NaN, negative and +inf entries are not admissible; callers provide finite
  // non-negative rows (the host contract documents this).
  return (x > 0.0f && isfinite(x)) ? x : 0.0f;
}

__device__ __forceinline__ bool valid_token(int32_t token, int vocab) {
  return token >= 0 && token < vocab;
}

/// Inclusive scan in thread order.  `s` must hold `nthreads` floats and is
/// overwritten with the inclusive prefix sums.  Two barriers per step keep the
/// scan race-free on every architecture.
__device__ __forceinline__ float block_scan_thread_order(float v, float* s, int nthreads) {
  const int t = static_cast<int>(threadIdx.x);
  s[t] = v;
  __syncthreads();
  for (int off = 1; off < nthreads; off <<= 1) {
    const float x = (t >= off) ? s[t - off] : 0.0f;
    __syncthreads();
    s[t] += x;
    __syncthreads();
  }
  return s[t];
}

/// Weight of token `idx` for one of the three draw modes.
///   q != nullptr        : max(p - q, 0)              (chain, probabilistic)
///   sparse != nullptr   : max(p - sum(q on token), 0) (tree residual)
///   else                : max(p - [idx == exclude], 0) (target-only chain)
/// For the pure target draw pass exclude = -1, sparse = nullptr and q = nullptr,
/// which reduces to p itself.
template <int VEC>
__device__ __forceinline__ void load_row_vec(const float* p, int64_t base, int vocab, float* v) {
  if constexpr (VEC == 4) {
    if (base + 4 <= vocab) {
      const float4 x = *reinterpret_cast<const float4*>(p + base);
      v[0] = x.x;
      v[1] = x.y;
      v[2] = x.z;
      v[3] = x.w;
      return;
    }
  }
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    const int64_t idx = base + e;
    v[e] = idx < vocab ? p[idx] : 0.0f;
  }
}

template <int VEC>
__device__ __forceinline__ float row_weight(const float* p, const float* q, int vocab,
                                            const int32_t* sparse_tokens, const float* sparse_q,
                                            int nsparse, int exclude, int64_t base) {
  float pv[VEC];
  float qv[VEC];
  load_row_vec<VEC>(p, base, vocab, pv);
  if (q != nullptr) load_row_vec<VEC>(q, base, vocab, qv);
  float w = 0.0f;
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    const int64_t idx = base + e;
    if (idx >= vocab) break;
    const float pe = clean_prob(pv[e]);
    float sub = 0.0f;
    if (q != nullptr) {
      sub = clean_prob(qv[e]);
    } else if (sparse_tokens != nullptr) {
      for (int c = 0; c < nsparse; ++c) {
        if (sparse_tokens[c] == idx) sub += sparse_q[c];
      }
    } else if (exclude >= 0 && idx == exclude) {
      sub = 1.0f;
    }
    const float r = pe - sub;
    if (r > 0.0f) w += r;
  }
  return w;
}

/// Draw one token from the row weight.  All threads must call it.  Returns -1
/// when the row has no positive mass (or the arithmetic cannot select a token).
template <int VEC>
__device__ __forceinline__ int32_t sample_weighted(const float* p, const float* q, int vocab,
                                                   const int32_t* sparse_tokens,
                                                   const float* sparse_q, int nsparse, int exclude,
                                                   float u, float* s_scan, int32_t* s_ctl) {
  const int t = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  float local = 0.0f;
  for (int64_t v = t; v * VEC < vocab; v += nthreads) {
    local += row_weight<VEC>(p, q, vocab, sparse_tokens, sparse_q, nsparse, exclude, v * VEC);
  }

  const float inclusive = block_scan_thread_order(local, s_scan, nthreads);
  const float total = s_scan[nthreads - 1];
  if (!(total > 0.0f) || !isfinite(total)) return -1;

  float target = u * total;
  if (target >= total) target = nextafterf(total, 0.0f);
  if (!(target >= 0.0f)) target = 0.0f;
  const float exclusive = inclusive - local;

  if (t == 0) s_ctl[0] = -1;
  __syncthreads();
  if (local > 0.0f && target >= exclusive && target < inclusive) s_ctl[0] = t;
  __syncthreads();
  const int owner = s_ctl[0];

  int32_t chosen = -1;
  if (owner >= 0 && t == owner) {
    float running = exclusive;
    for (int64_t v = t; v * VEC < vocab && chosen < 0; v += nthreads) {
      const int64_t base = v * VEC;
      float pv[VEC];
      float qv[VEC];
      load_row_vec<VEC>(p, base, vocab, pv);
      if (q != nullptr) load_row_vec<VEC>(q, base, vocab, qv);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const int64_t idx = base + e;
        if (idx >= vocab) break;
        const float pe = clean_prob(pv[e]);
        float sub = 0.0f;
        if (q != nullptr) {
          sub = clean_prob(qv[e]);
        } else if (sparse_tokens != nullptr) {
          for (int c = 0; c < nsparse; ++c) {
            if (sparse_tokens[c] == idx) sub += sparse_q[c];
          }
        } else if (exclude >= 0 && idx == exclude) {
          sub = 1.0f;
        }
        const float w = pe > sub ? pe - sub : 0.0f;
        if (w > 0.0f && target < running + w) {
          chosen = static_cast<int32_t>(idx);
          break;
        }
        running += w;
      }
    }
  }

  if (t == 0) s_ctl[1] = -1;
  __syncthreads();
  if (t == owner && chosen >= 0) s_ctl[1] = chosen;
  __syncthreads();
  return s_ctl[1];
}

/// RNG value for a request/step.  The draw map is part of the contract.
__device__ __forceinline__ float draw(const SpecVerifyParams& p, int b, int64_t /*unused*/,
                                      uint32_t index) {
  const uint64_t seed = p.rng.per_request_seed != nullptr ? p.rng.per_request_seed[b]
                                                          : p.rng.global_seed;
  const uint64_t position =
      p.rng.per_request_offset != nullptr ? static_cast<uint64_t>(p.rng.per_request_offset[b]) : 0;
  return uniform_from_counter(seed, static_cast<uint64_t>(b), position, index, p.rng.stream_id);
}

__device__ __forceinline__ float draw_tree(const TreeVerifyParams& p, int b, uint32_t index) {
  const uint64_t seed = p.rng.per_request_seed != nullptr ? p.rng.per_request_seed[b]
                                                          : p.rng.global_seed;
  const uint64_t position =
      p.rng.per_request_offset != nullptr ? static_cast<uint64_t>(p.rng.per_request_offset[b]) : 0;
  return uniform_from_counter(seed, static_cast<uint64_t>(b), position, index, p.rng.stream_id);
}

}  // namespace

// -----------------------------------------------------------------------------
// Chain verification
// -----------------------------------------------------------------------------
template <int VEC_SIZE>
__global__ void rejection_sample_kernel(const __grid_constant__ SpecVerifyParams params) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= params.batch_size) return;

  __shared__ float s_scan[kSpecThreads];
  __shared__ int32_t s_ctl[2];
  __shared__ int32_t s_int;
  __shared__ int32_t s_recovered;

  if (threadIdx.x == 0) {
    s_int = 0;
    s_recovered = -1;
  }
  __syncthreads();

  const int64_t total_drafts = params.cu_num_draft[params.batch_size];
  int64_t start = params.cu_num_draft[b];
  int64_t end = params.cu_num_draft[b + 1];
  if (start < 0) start = 0;
  if (start > total_drafts) start = total_drafts;
  if (end < start) end = start;
  if (end > total_drafts) end = total_drafts;
  int64_t k = end - start;
  if (k > params.max_draft) k = params.max_draft;

  const bool greedy = params.is_greedy != nullptr && params.is_greedy[b] != 0;
  const int64_t bonus_row = total_drafts + b;

  int accepted = 0;
  bool rejected = false;
  for (int64_t i = 0; i < k; ++i) {
    const int32_t token = params.draft_token_ids[start + i];
    const bool token_ok = valid_token(token, params.vocab_size);
    if (greedy) {
      if (threadIdx.x == 0) {
        const int32_t am = params.target_argmax != nullptr ? params.target_argmax[start + i]
                                                           : INT32_MIN;
        if (token_ok && am == token) {
          s_int = 1;
        } else {
          s_int = 0;
          s_recovered = valid_token(am, params.vocab_size) ? am : -1;
        }
      }
      __syncthreads();
      if (s_int) {
        ++accepted;
        continue;
      }
      rejected = true;
      break;
    }

    if (threadIdx.x == 0) {
      float pv = 0.0f;
      float qv = 1.0f;
      if (token_ok) {
        pv = clean_prob(params.target_probs[(start + i) * params.vocab_size + token]);
        if (params.draft_probs != nullptr) {
          qv = clean_prob(params.draft_probs[(start + i) * params.vocab_size + token]);
        }
      }
      float ratio = 1.0f;
      if (params.draft_probs != nullptr) {
        ratio = (qv > 0.0f) ? pv / qv : (pv > 0.0f ? 1.0f : 0.0f);
      } else {
        ratio = token_ok ? pv : 0.0f;
      }
      const float u = draw(params, b, i, static_cast<uint32_t>(2 * i));
      const float r = fmaxf(0.0f, ratio);
      const bool keep = token_ok && (r >= 1.0f || u < r);
      s_int = keep ? 1 : 0;
    }
    __syncthreads();
    if (s_int) {
      ++accepted;
      continue;
    }

    // First rejection: draw the corrected token from norm(max(0, p - q)).  For
    // a deterministic drafter q is one-hot at the draft token.
    const int32_t recovered = sample_weighted<VEC_SIZE>(
        params.target_probs + (start + i) * params.vocab_size,
        params.draft_probs != nullptr ? params.draft_probs + (start + i) * params.vocab_size
                                      : nullptr,
        params.vocab_size, nullptr, nullptr, 0, params.draft_probs == nullptr ? token : -1,
        draw(params, b, i, static_cast<uint32_t>(2 * i + 1)), s_scan, s_ctl);
    if (threadIdx.x == 0) s_recovered = recovered;
    __syncthreads();
    rejected = true;
    break;
  }

  if (!rejected) {
    if (greedy) {
      if (threadIdx.x == 0) {
        const int32_t am = params.target_argmax != nullptr ? params.target_argmax[bonus_row]
                                                           : INT32_MIN;
        s_recovered = valid_token(am, params.vocab_size) ? am : -1;
      }
      __syncthreads();
    } else {
      const int32_t recovered = sample_weighted<VEC_SIZE>(
          params.target_probs + bonus_row * params.vocab_size, nullptr, params.vocab_size, nullptr,
          nullptr, 0, -1, draw(params, b, 0, static_cast<uint32_t>(2 * k + 1)), s_scan, s_ctl);
      if (threadIdx.x == 0) s_recovered = recovered;
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    const int64_t stride = params.max_draft + 1;
    int64_t j = 0;
    for (; j < accepted; ++j) {
      params.output_token_ids[b * stride + j] = params.draft_token_ids[start + j];
    }
    params.output_token_ids[b * stride + j] = s_recovered;
    ++j;
    for (; j <= params.max_draft; ++j) params.output_token_ids[b * stride + j] = -1;
    params.num_accepted[b] = accepted;
  }
}

// -----------------------------------------------------------------------------
// Tree verification
// -----------------------------------------------------------------------------
template <int VEC_SIZE>
__global__ void tree_verify_kernel(const __grid_constant__ TreeVerifyParams params) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= params.batch_size) return;

  __shared__ float s_scan[kSpecThreads];
  __shared__ int32_t s_ctl[2];
  __shared__ int32_t s_meta[8];
  __shared__ int32_t s_recovered;
  __shared__ int32_t s_recovered_node;
  extern __shared__ __align__(16) unsigned char s_dyn[];
  int32_t* s_tok = reinterpret_cast<int32_t*>(s_dyn);
  float* s_q = reinterpret_cast<float*>(s_dyn + static_cast<int64_t>(params.num_nodes) *
                                                    static_cast<int64_t>(sizeof(int32_t)));
  int32_t* s_path_tok = reinterpret_cast<int32_t*>(
      s_dyn + static_cast<int64_t>(params.num_nodes) * (sizeof(int32_t) + sizeof(float)));
  int32_t* s_path_node = s_path_tok + params.max_depth + 1;

  const int N = params.num_nodes;
  const int V = params.vocab_size;
  const bool greedy = params.target_probs == nullptr;
  if (threadIdx.x == 0) {
    s_meta[0] = 0;  // number of children
    s_meta[1] = -1; // selected child
    s_meta[2] = 0;  // stop after this step
    s_meta[3] = 0;  // force residual draw
    s_meta[4] = 0;  // node of the residual token, or -1
    s_meta[5] = 0;  // a recovery/bonus token was already produced
    s_recovered = -1;
    s_recovered_node = -1;
  }
  __syncthreads();

  int cur = 0;
  int accepted = 0;
  while (accepted < params.max_depth) {
    if (threadIdx.x == 0) {
      s_meta[0] = 0;
      s_meta[1] = -1;
      s_meta[2] = 0;
      s_meta[3] = 0;
      s_meta[4] = -1;
      int n = 0;
      int32_t c = params.first_child[b * N + cur];
      int guard = 0;
      while (c >= 0 && c < N && guard++ < N) {
        s_tok[n] = c;
        s_q[n] = 0.0f;
        ++n;
        c = params.next_sibling[b * N + c];
      }
      s_meta[0] = n;
    }
    __syncthreads();

    const int nchild = s_meta[0];
    if (nchild == 0) break;

    int32_t ridx = params.retrieve_index != nullptr ? params.retrieve_index[b * N + cur] : cur;
    if (ridx < 0 || ridx >= N) ridx = cur;
    const float* target_row = params.target_probs != nullptr
                                  ? params.target_probs +
                                        (static_cast<int64_t>(b) * N + ridx) * V
                                  : nullptr;
    const float* draft_row = params.draft_probs != nullptr
                                 ? params.draft_probs +
                                       (static_cast<int64_t>(b) * N + cur) * V
                                 : nullptr;

    if (greedy) {
      if (threadIdx.x == 0) {
        const int32_t am = params.target_argmax != nullptr ? params.target_argmax[b * N + cur]
                                                           : INT32_MIN;
        int chosen = -1;
        for (int j = 0; j < nchild; ++j) {
          const int c = s_tok[j];
          const int32_t cand = params.candidates[b * N + c];
          if (cand == am) {
            chosen = j;
            break;
          }
        }
        if (chosen >= 0) {
          s_meta[1] = s_tok[chosen];
          s_path_tok[accepted] = params.candidates[b * N + s_tok[chosen]];
          s_path_node[accepted] = s_tok[chosen];
        } else {
          s_meta[2] = 1;
          s_recovered = valid_token(am, V) ? am : -1;
          s_recovered_node = -1;
          s_meta[5] = 1;
        }
      }
      __syncthreads();
      if (s_meta[2] != 0) break;
      cur = s_meta[1];
      ++accepted;
      continue;
    }

    if (draft_row == nullptr) {
      // Deterministic drafter: the first child is the proposal, q = one-hot.
      if (threadIdx.x == 0) {
        const int c = s_tok[0];
        const int32_t cand = params.candidates[b * N + c];
        const float pv = valid_token(cand, V) ? clean_prob(target_row[cand]) : 0.0f;
        const float u = draw_tree(params, b, static_cast<uint32_t>(4 * accepted));
        if (pv >= 1.0f || u < pv) {
          s_meta[1] = c;
          s_path_tok[accepted] = cand;
          s_path_node[accepted] = c;
        } else {
          s_meta[2] = 1;
          s_meta[3] = 1;
          s_meta[4] = c;
          s_meta[5] = 1;
        }
      }
      __syncthreads();
      if (s_meta[2] != 0) {
        const int32_t recovered = sample_weighted<VEC_SIZE>(
            target_row, nullptr, V, nullptr, nullptr, 0,
            params.candidates[b * N + s_meta[4]],
            draw_tree(params, b, static_cast<uint32_t>(4 * accepted + 2)), s_scan, s_ctl);
        if (threadIdx.x == 0) s_recovered = recovered;
        __syncthreads();
        break;
      }
      cur = s_meta[1];
      ++accepted;
      continue;
    }

    if (threadIdx.x == 0) {
      float sum_q = 0.0f;
      for (int j = 0; j < nchild; ++j) {
        const int c = s_tok[j];
        const int32_t cand = params.candidates[b * N + c];
        const float qv = valid_token(cand, V) ? clean_prob(draft_row[cand]) : 0.0f;
        s_q[j] = qv;
        sum_q += qv;
      }
      if (!(sum_q > 0.0f)) {
        s_meta[2] = 1;
        s_meta[3] = 2;  // residual is the plain target row
      } else {
        const float r = draw_tree(params, b, static_cast<uint32_t>(4 * accepted)) * sum_q;
        float acc = 0.0f;
        int cq = -1;
        for (int j = 0; j < nchild; ++j) {
          s_q[j] /= sum_q;
          acc += s_q[j];
          if (cq < 0 && r < acc) cq = j;
        }
        if (cq < 0) cq = nchild - 1;
        const int c = s_tok[cq];
        const int32_t cand = params.candidates[b * N + c];
        const float pv = valid_token(cand, V) ? clean_prob(target_row[cand]) : 0.0f;
        const float qn = s_q[cq];
        const float ratio = (qn > 0.0f) ? pv / qn : (pv > 0.0f ? 1.0f : 0.0f);
        const float u = draw_tree(params, b, static_cast<uint32_t>(4 * accepted + 1));
        const float rr = fmaxf(0.0f, ratio);
        if (rr >= 1.0f || u < rr) {
          s_meta[1] = c;
          s_path_tok[accepted] = cand;
          s_path_node[accepted] = c;
        } else {
          s_meta[2] = 1;
          s_meta[3] = 1;
          s_meta[5] = 1;
        }
      }
    }
    __syncthreads();
    if (s_meta[2] != 0) {
      int32_t recovered = -1;
      if (s_meta[3] == 1) {
        recovered = sample_weighted<VEC_SIZE>(
            target_row, nullptr, V, s_tok, s_q, nchild, -1,
            draw_tree(params, b, static_cast<uint32_t>(4 * accepted + 2)), s_scan, s_ctl);
      }
      if (recovered < 0) {
        recovered = sample_weighted<VEC_SIZE>(
            target_row, nullptr, V, nullptr, nullptr, 0, -1,
            draw_tree(params, b, static_cast<uint32_t>(4 * accepted + 2)), s_scan, s_ctl);
      }
      if (threadIdx.x == 0) {
        s_recovered = recovered;
        s_recovered_node = -1;
        s_meta[5] = 1;
        if (recovered >= 0) {
          for (int j = 0; j < nchild; ++j) {
            if (params.candidates[b * N + s_tok[j]] == recovered) {
              s_recovered_node = s_tok[j];
              break;
            }
          }
        }
      }
      __syncthreads();
      break;
    }
    cur = s_meta[1];
    ++accepted;
  }

  // A leaf, max_depth, or a stopped tree gets one bonus token from the target
  // row of the current node (the same row used to verify its children).  A
  // recovery token from the residual path already fills this slot.
  if (threadIdx.x == 0) s_meta[2] = 0;
  if (s_meta[5] == 0) {
    int32_t ridx = params.retrieve_index != nullptr ? params.retrieve_index[b * N + cur] : cur;
    if (ridx < 0 || ridx >= N) ridx = cur;
    const float* target_row = params.target_probs != nullptr
                                  ? params.target_probs +
                                        (static_cast<int64_t>(b) * N + ridx) * V
                                  : nullptr;
    if (greedy) {
      if (threadIdx.x == 0) {
        const int32_t am = params.target_argmax != nullptr ? params.target_argmax[b * N + cur]
                                                           : INT32_MIN;
        s_recovered = valid_token(am, V) ? am : -1;
      }
      __syncthreads();
    } else {
      const int32_t token = sample_weighted<VEC_SIZE>(
          target_row, nullptr, V, nullptr, nullptr, 0, -1,
          draw_tree(params, b, static_cast<uint32_t>(4 * accepted + 3)), s_scan, s_ctl);
      if (threadIdx.x == 0) {
        s_recovered = token;
        s_recovered_node = -1;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) s_meta[5] = 1;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int64_t stride = params.max_depth + 1;
    for (int j = 0; j < accepted; ++j) {
      params.output_token_ids[b * stride + j] = s_path_tok[j];
      params.accept_index[b * stride + j] = s_path_node[j];
    }
    params.output_token_ids[b * stride + accepted] = s_recovered;
    params.accept_index[b * stride + accepted] = s_recovered_node;
    for (int j = accepted + 1; j < stride; ++j) {
      params.output_token_ids[b * stride + j] = -1;
      params.accept_index[b * stride + j] = -1;
    }
    params.accept_length[b] = accepted;
  }
}

// -----------------------------------------------------------------------------
// Packed tree attention mask
// -----------------------------------------------------------------------------
__global__ void build_tree_attention_mask_kernel(
    uint8_t* __restrict__ packed_mask, int64_t* __restrict__ mask_indptr,
    int32_t* __restrict__ position_offsets, const int32_t* __restrict__ parent,
    const int32_t* __restrict__ prefix_lens, const int batch_size, const int num_nodes) {
  const int b = static_cast<int>(blockIdx.y);
  const int node = static_cast<int>(blockIdx.x);
  if (b >= batch_size || node >= num_nodes) return;

  extern __shared__ __align__(16) unsigned char s_dyn[];
  uint32_t* anc = reinterpret_cast<uint32_t*>(s_dyn);
  const int words = (num_nodes + 31) / 32;

  int64_t base = 0;
  for (int r = 0; r < b; ++r) {
    const int pl = prefix_lens[r] > 0 ? prefix_lens[r] : 0;
    base += static_cast<int64_t>(num_nodes) * (static_cast<int64_t>(pl) + num_nodes);
  }
  const int prefix = prefix_lens[b] > 0 ? prefix_lens[b] : 0;
  const int64_t kv_len = static_cast<int64_t>(prefix) + num_nodes;

  if (threadIdx.x == 0) {
    for (int w = 0; w < words; ++w) anc[w] = 0u;
    int cur = node;
    int guard = 0;
    while (cur >= 0 && cur < num_nodes && guard++ <= num_nodes) {
      anc[cur >> 5] |= 1u << (cur & 31);
      cur = parent[b * num_nodes + cur];
    }
    int depth = 0;
    cur = parent[b * num_nodes + node];
    guard = 0;
    while (cur >= 0 && cur < num_nodes && guard++ <= num_nodes) {
      ++depth;
      cur = parent[b * num_nodes + cur];
    }
    position_offsets[b * num_nodes + node] = depth;

    if (b == 0 && node == 0) {
      int64_t offset = 0;
      mask_indptr[0] = 0;
      for (int r = 0; r < batch_size; ++r) {
        const int pl = prefix_lens[r] > 0 ? prefix_lens[r] : 0;
        offset += static_cast<int64_t>(num_nodes) * (static_cast<int64_t>(pl) + num_nodes);
        mask_indptr[r + 1] = offset;
      }
    }
  }
  __syncthreads();

  const int64_t row_bits = static_cast<int64_t>(node) * kv_len;
  const int64_t work = (kv_len + 31) / 32;
  for (int64_t wi = threadIdx.x; wi < work; wi += blockDim.x) {
    const int64_t bit0 = wi * 32;
    uint32_t bits = 0u;
    for (int e = 0; e < 32; ++e) {
      const int64_t key = bit0 + e;
      if (key >= kv_len) break;
      bool visible;
      if (key < prefix) {
        visible = true;
      } else {
        const int t = static_cast<int>(key - prefix);
        visible = ((anc[t >> 5] >> (t & 31)) & 1u) != 0u;
      }
      if (visible) bits |= 1u << e;
    }
    if (bits == 0u) continue;
    const int64_t abs = base + row_bits + bit0;
    const int64_t word = abs >> 5;
    const int shift = static_cast<int>(abs & 31);
    unsigned int* out = reinterpret_cast<unsigned int*>(packed_mask);
    atomicOr(out + word, bits << shift);
    if (shift != 0 && (bits >> (32 - shift)) != 0u) {
      atomicOr(out + word + 1, bits >> (32 - shift));
    }
  }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------
namespace {

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

Status backend_ok(const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sampling: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

int pick_spec_vec(float* p, float* q, int vocab) {
  if (vocab % 4 != 0) return 1;
  const int v = pick_vec_size<float>(vocab, p, q);
  return v >= 4 ? 4 : 1;
}

}  // namespace

Status rejection_sample(const SpecVerifyParams& params, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (params.batch_size < 0) {
    return detail::set_error(Status::kInvalidArgument, "rejection_sample: batch_size %d",
                             params.batch_size);
  }
  if (params.max_draft < 0) {
    return detail::set_error(Status::kInvalidArgument, "rejection_sample: max_draft %d",
                             params.max_draft);
  }
  if (params.vocab_size <= 0) {
    return detail::set_error(Status::kInvalidArgument, "rejection_sample: vocab_size %d",
                             params.vocab_size);
  }
  if (params.is_greedy != nullptr && params.target_argmax == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "rejection_sample: is_greedy needs target_argmax");
  }
  if (params.batch_size == 0) return Status::kSuccess;
  if (params.draft_token_ids == nullptr || params.cu_num_draft == nullptr ||
      params.target_probs == nullptr || params.output_token_ids == nullptr ||
      params.num_accepted == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "rejection_sample: null required pointer");
  }
  const int vec = pick_spec_vec(const_cast<float*>(params.target_probs),
                                const_cast<float*>(params.draft_probs), params.vocab_size);
  const dim3 grid(static_cast<unsigned>(params.batch_size));
  switch (vec) {
    case 4:
      rejection_sample_kernel<4><<<grid, kSpecThreads, 0, stream_of(ctx)>>>(params);
      break;
    default:
      rejection_sample_kernel<1><<<grid, kSpecThreads, 0, stream_of(ctx)>>>(params);
      break;
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("rejection_sample", err);
  return Status::kSuccess;
}

Status tree_verify(const TreeVerifyParams& params, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (params.batch_size < 0) {
    return detail::set_error(Status::kInvalidArgument, "tree_verify: batch_size %d",
                             params.batch_size);
  }
  if (params.num_nodes <= 0) {
    return detail::set_error(Status::kInvalidArgument, "tree_verify: num_nodes %d",
                             params.num_nodes);
  }
  if (params.max_depth < 0 || params.max_depth > params.num_nodes - 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "tree_verify: max_depth %d, num_nodes %d", params.max_depth,
                             params.num_nodes);
  }
  if (params.vocab_size <= 0) {
    return detail::set_error(Status::kInvalidArgument, "tree_verify: vocab_size %d",
                             params.vocab_size);
  }
  if (params.target_probs == nullptr && params.target_argmax == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "tree_verify: greedy needs target_argmax");
  }
  if (params.target_probs != nullptr && params.retrieve_index == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "tree_verify: stochastic needs retrieve_index");
  }
  if (params.batch_size == 0) return Status::kSuccess;
  if (params.candidates == nullptr || params.first_child == nullptr ||
      params.next_sibling == nullptr || params.accept_index == nullptr ||
      params.accept_length == nullptr || params.output_token_ids == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "tree_verify: null required pointer");
  }
  const int vec = pick_spec_vec(const_cast<float*>(params.target_probs),
                                const_cast<float*>(params.draft_probs), params.vocab_size);
  const size_t smem = static_cast<size_t>(params.num_nodes) * (sizeof(int32_t) + sizeof(float)) +
                      static_cast<size_t>(params.max_depth + 1) * 2u * sizeof(int32_t);
  if (smem > 48u * 1024u) {
    const cudaError_t attr = cudaFuncSetAttribute(
        vec >= 4 ? reinterpret_cast<const void*>(tree_verify_kernel<4>)
                 : reinterpret_cast<const void*>(tree_verify_kernel<1>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (attr != cudaSuccess) return launch_error("tree_verify: shared attribute", attr);
  }
  const dim3 grid(static_cast<unsigned>(params.batch_size));
  if (vec >= 4) {
    tree_verify_kernel<4><<<grid, kSpecThreads, smem, stream_of(ctx)>>>(params);
  } else {
    tree_verify_kernel<1><<<grid, kSpecThreads, smem, stream_of(ctx)>>>(params);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("tree_verify", err);
  return Status::kSuccess;
}

Status build_tree_attention_mask(const Tensor& packed_mask, const Tensor& mask_indptr,
                                 const Tensor& position_offsets, const Tensor& parent,
                                 const Tensor& prefix_lens, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(packed_mask, DType::kUInt8, 1, 4, "build_tree_attention_mask.packed_mask");
  if (s != Status::kSuccess) return s;
  s = check_tensor(mask_indptr, DType::kInt64, 1, 8, "build_tree_attention_mask.mask_indptr");
  if (s != Status::kSuccess) return s;
  s = check_tensor(position_offsets, DType::kInt32, 2, 4,
                   "build_tree_attention_mask.position_offsets");
  if (s != Status::kSuccess) return s;
  s = check_tensor(parent, DType::kInt32, 2, 4, "build_tree_attention_mask.parent");
  if (s != Status::kSuccess) return s;
  s = check_tensor(prefix_lens, DType::kInt32, 1, 4, "build_tree_attention_mask.prefix_lens");
  if (s != Status::kSuccess) return s;
  if (packed_mask.shape[0] <= 0 || packed_mask.stride[0] != 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "build_tree_attention_mask: packed_mask must be a non-empty byte vector");
  }
  if (mask_indptr.stride[0] != 1 || parent.stride[1] != 1 || position_offsets.stride[1] != 1 ||
      parent.stride[0] != parent.shape[1] || position_offsets.stride[0] != position_offsets.shape[1] ||
      prefix_lens.stride[0] != 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "build_tree_attention_mask: tensors must be packed");
  }
  const int batch = static_cast<int>(parent.shape[0]);
  const int nodes = static_cast<int>(parent.shape[1]);
  if (batch < 0 || nodes <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "build_tree_attention_mask: parent is [%d, %d]", batch, nodes);
  }
  if (position_offsets.shape[0] != batch || position_offsets.shape[1] != nodes ||
      prefix_lens.shape[0] != batch || mask_indptr.shape[0] != batch + 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "build_tree_attention_mask: shape mismatch");
  }
  if (batch == 0) return Status::kSuccess;

  unsigned char* base = reinterpret_cast<unsigned char*>(static_cast<char*>(packed_mask.data) +
                                                         packed_mask.byte_offset);
  const size_t bytes = static_cast<size_t>(packed_mask.shape[0]);
  cudaError_t err = cudaMemsetAsync(base, 0, bytes, stream_of(ctx));
  if (err != cudaSuccess) return launch_error("build_tree_attention_mask: memset", err);

  const size_t smem = static_cast<size_t>((nodes + 31) / 32) * sizeof(uint32_t);
  if (smem > 48u * 1024u) {
    err = cudaFuncSetAttribute(reinterpret_cast<const void*>(build_tree_attention_mask_kernel),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem));
    if (err != cudaSuccess) return launch_error("build_tree_attention_mask: shared attribute", err);
  }
  const dim3 grid(static_cast<unsigned>(nodes), static_cast<unsigned>(batch));
  build_tree_attention_mask_kernel<<<grid, 256, smem, stream_of(ctx)>>>(
      base, reinterpret_cast<int64_t*>(static_cast<char*>(mask_indptr.data) + mask_indptr.byte_offset),
      reinterpret_cast<int32_t*>(static_cast<char*>(position_offsets.data) +
                                 position_offsets.byte_offset),
      reinterpret_cast<const int32_t*>(static_cast<const char*>(parent.data) + parent.byte_offset),
      reinterpret_cast<const int32_t*>(static_cast<const char*>(prefix_lens.data) +
                                       prefix_lens.byte_offset),
      batch, nodes);
  err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("build_tree_attention_mask", err);
  return Status::kSuccess;
}

}  // namespace kops::sample
