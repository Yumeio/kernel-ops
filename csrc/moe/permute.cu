// =============================================================================
// kernel-ops | csrc/moe/permute.cu
// B6 rows 5-6: routing metadata, token permutation with optional fused FP8
// quantization, weighted unpermute + reduce, and the [T, K, H] sum (M02).
//   moe_count_and_offsets_kernel  expert counts + aligned exclusive offsets
//   moe_align_block_size_kernel   Triton/Marlin metadata, stable by slot id
//   moe_permute_kernel            scatter rows into expert-contiguous order
//   moe_unpermute_reduce_kernel   combine expert outputs, fixed k order
//   moe_sum_kernel                out[t] = sum_k input[t, k]
//
// Determinism: placement walks the slot array in ascending order inside one warp
// (chunks of 32 slots), so each expert's rows are ordered by slot id. The copy
// phase is order-free (each destination row is written exactly once).
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <cuda.h>
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
#include "csrc/common/quant.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"

namespace kops::moe {
namespace {

constexpr int kMaxExperts = 1024;
constexpr int kAlignThreads = 1024;

__device__ __forceinline__ int local_expert(int e, const int32_t* expert_map) {
  if (e < 0) return -1;
  if (expert_map == nullptr) return e;
  return expert_map[e];
}

// Stable parallel counting sort (the vLLM/sglang align kernels give every thread a contiguous chunk
// and prefix the per-thread histograms; here each *warp* owns one contiguous segment of the slot
// array so the histogram is [warps][experts] and fits shared memory for E <= 1024).
//   1. count : warp w histograms its segment into cnt[w][e] (warp-private row, no atomics);
//   2. scan  : cnt[w][e] becomes the exclusive prefix over warps, tot[e] the column sum;
//   3. place : warp w walks its segment again; slot -> base[e] + cnt[w][e] + rank-in-step.
// Rank inside a 32-slot step comes from __match_any_sync, so the order is ascending slot id inside
// every expert, exactly as the old single-warp walk, with W warps working at once.
constexpr int kSortSmemBytes = 48 * 1024;

/// Warps for a block that sorts into `num_experts` bins: cnt[W][E] + one E-sized vector in 48 KiB.
inline int sort_warps(int num_experts) {
  const int w = (kSortSmemBytes / 4 - num_experts) / num_experts;
  return w < 1 ? 1 : (w > 32 ? 32 : w);
}
inline int sort_smem_bytes(int num_experts, int warps) {
  return (warps * num_experts + num_experts) * static_cast<int>(sizeof(int));
}

/// Slot range [lo, hi) of warp w: contiguous, 32-aligned so the loads coalesce.
__device__ __forceinline__ void sort_segment(int64_t numel, int w, int W, int64_t* lo, int64_t* hi) {
  const int64_t seg = (((numel + W - 1) / W) + 31) & ~static_cast<int64_t>(31);
  const int64_t l = static_cast<int64_t>(w) * seg;
  *lo = l < numel ? l : numel;
  *hi = *lo + seg < numel ? *lo + seg : numel;
}

/// Pass 1/3 of the sort. `local(e)` maps a raw topk id to a bin in [0, E) or -1.
template <typename BinFn>
__device__ __forceinline__ void sort_count(int* cnt, const int32_t* __restrict__ topk_ids, int64_t numel,
                                           int E, BinFn bin) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, W = blockDim.x >> 5;
  int64_t lo, hi;
  sort_segment(numel, w, W, &lo, &hi);
  int* mine = cnt + w * E;
  for (int64_t s = lo; s < hi; s += 32) {
    const int64_t pos = s + lane;
    const int b = pos < hi ? bin(topk_ids[pos]) : -1;
    const unsigned mask = __match_any_sync(0xFFFFFFFFu, b);
    if (b >= 0 && lane == __ffs(mask) - 1) mine[b] += __popc(mask);
    __syncwarp();
  }
}

/// Pass 3/3: write `put(slot, bin, dest)` for every slot; base[bin] is the bin's first destination.
template <typename BinFn, typename PutFn>
__device__ __forceinline__ void sort_place(int* cnt, const int* base, const int32_t* __restrict__ topk_ids,
                                           int64_t numel, int E, BinFn bin, PutFn put) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, W = blockDim.x >> 5;
  int64_t lo, hi;
  sort_segment(numel, w, W, &lo, &hi);
  int* mine = cnt + w * E;
  for (int64_t s = lo; s < hi; s += 32) {
    const int64_t pos = s + lane;
    const int b = pos < hi ? bin(topk_ids[pos]) : -1;
    const unsigned mask = __match_any_sync(0xFFFFFFFFu, b);
    const int rank = __popc(mask & ((1u << lane) - 1u));
    if (b >= 0) put(pos, b, base[b] + mine[b] + rank);
    __syncwarp();
    if (b >= 0 && lane == __ffs(mask) - 1) mine[b] += __popc(mask);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// align_block_size: multi-CTA stable counting sort
//
// One CTA per contiguous slot chunk. Every CTA builds the *whole* histogram (a chunk boundary is
// independent of the expert layout, so an expert's rows come from every chunk) and keeps, on top of
// it, the histogram of the slots that precede its own chunk: that is exactly the chunk's starting
// rank per expert, so no cross-CTA prefix scan and no global scratch are needed. Per-chunk ordering
// is then the usual warp-segment scheme, and the rank inside a 32-slot step comes from a bit-sliced
// ballot compare instead of __match_any_sync (match.any is serialized in the SM's vote path; on
// SM89 it costs ~60 cycles per instruction, i.e. 6x the 8-ballot sequence that replaces it).
// ---------------------------------------------------------------------------
constexpr int kAlignSortSmemBytes = 48 * 1024;
constexpr int kAlignMaxCtas = 32;
constexpr int kAlignSlotsPerCta = 1024;

/// Warps for an align block: rows[W][E+1] + pre[E] + tot[E] + base[E] inside 48 KiB.
inline int align_warps(int num_experts) {
  const int w = (kAlignSortSmemBytes / 4 - 3 * num_experts) / (num_experts + 1);
  return w < 1 ? 1 : (w > 32 ? 32 : w);
}
__host__ __device__ inline int align_row_stride(int num_experts) { return num_experts + 1; }
inline int align_smem_bytes(int num_experts, int warps) {
  return (warps * align_row_stride(num_experts) + 3 * num_experts) * static_cast<int>(sizeof(int));
}
/// CTAs for `numel` slots: every CTA redundantly histograms the whole array (that is what keeps the
/// chunk prefix local), so the grid is capped instead of scaled with the slot count.
inline int align_ctas(int64_t numel) {
  if (numel <= 0) return 1;
  int64_t g = (numel + kAlignSlotsPerCta - 1) / kAlignSlotsPerCta;
  if (g > kAlignMaxCtas) g = kAlignMaxCtas;
  return g < 1 ? 1 : static_cast<int>(g);
}

/// Lanes of a warp whose key equals this lane's key, over the low `bits` bits of the expert id.
/// `bits` must be >= 1 and cover num_experts - 1; invalid keys are passed as ~0u.
__device__ __forceinline__ unsigned align_equal_mask(unsigned key, int bits) {
  unsigned same = 0xFFFFFFFFu;
  for (int t = 0; t < bits; ++t) {
    const unsigned bit = (key >> t) & 1u;
    const unsigned ball = __ballot_sync(0xFFFFFFFFu, bit);
    same &= bit ? ball : ~ball;
  }
  return same;
}

/// Histogram [0, numel) in one pass: slots below `split` count into `pre`, the rest into `tot`.
/// Counting is order-free, so the whole block cooperates (vector loads, warp rows are not needed).
template <typename BinFn>
__device__ __forceinline__ void align_hist_split(int* pre, int* tot, const int32_t* __restrict__ ids,
                                                 int64_t numel, int64_t split, BinFn bin) {
  const int64_t n4 = numel & ~static_cast<int64_t>(3);
  if ((reinterpret_cast<uintptr_t>(ids) & 15u) == 0u) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * 4;
    for (int64_t s = static_cast<int64_t>(threadIdx.x) * 4; s < n4; s += stride) {
      const int4 v = *reinterpret_cast<const int4*>(ids + s);
      int* dst = s + 3 < split ? pre : (s >= split ? tot : nullptr);
      const int b0 = bin(v.x), b1 = bin(v.y), b2 = bin(v.z), b3 = bin(v.w);
      if (dst != nullptr) {
        if (b0 >= 0) atomicAdd(dst + b0, 1);
        if (b1 >= 0) atomicAdd(dst + b1, 1);
        if (b2 >= 0) atomicAdd(dst + b2, 1);
        if (b3 >= 0) atomicAdd(dst + b3, 1);
      } else {  // the group straddles the chunk boundary
        if (b0 >= 0) atomicAdd((s < split ? pre : tot) + b0, 1);
        if (b1 >= 0) atomicAdd((s + 1 < split ? pre : tot) + b1, 1);
        if (b2 >= 0) atomicAdd((s + 2 < split ? pre : tot) + b2, 1);
        if (b3 >= 0) atomicAdd((s + 3 < split ? pre : tot) + b3, 1);
      }
    }
  } else {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * 4;
    for (int64_t s = static_cast<int64_t>(threadIdx.x) * 4; s < n4; s += stride) {
      for (int64_t i = s; i < s + 4; ++i) {
        const int b = bin(ids[i]);
        if (b >= 0) atomicAdd((i < split ? pre : tot) + b, 1);
      }
    }
  }
  for (int64_t i = n4 + threadIdx.x; i < numel; i += blockDim.x) {
    const int b = bin(ids[i]);
    if (b >= 0) atomicAdd((i < split ? pre : tot) + b, 1);
  }
}

/// Per-warp histogram of [lo, hi) in the placement's own sub-segment layout, so rows[w][e] is the
/// count of expert e among the slots warp w will place. Counting is order-free; the ordering is
/// established by pass 2 (`align_place_chunk`), which walks the same sub-segments.
template <typename BinFn>
__device__ __forceinline__ void align_hist_rows(int* rows, int row_stride, const int32_t* __restrict__ ids,
                                                int64_t lo, int64_t hi, BinFn bin) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, W = blockDim.x >> 5;
  const int64_t len = hi - lo;
  if (len <= 0) return;
  const int64_t seg = ((len + W - 1) / W + 31) & ~static_cast<int64_t>(31);
  const int64_t a = lo + static_cast<int64_t>(w) * seg;
  if (a >= hi) return;
  const int64_t b_end = a + seg < hi ? a + seg : hi;
  int* mine = rows + static_cast<int64_t>(w) * row_stride;
  for (int64_t s = a + lane; s < b_end; s += 32) {
    const int b = bin(ids[s]);
    if (b >= 0) atomicAdd(mine + b, 1);
  }
}

/// Pass over [lo, hi) in 32-slot warp steps: rows[w][e] holds warp w's exclusive prefix for expert e.
/// `put(slot, expert, dest)` runs for every valid slot; the order inside an expert is ascending slot.
template <typename BinFn, typename PutFn>
__device__ __forceinline__ void align_place_chunk(int* rows, int row_stride, const int* start,
                                                  const int32_t* __restrict__ ids, int64_t lo, int64_t hi,
                                                  int bits, BinFn bin, PutFn put) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, W = blockDim.x >> 5;
  const int64_t len = hi - lo;
  if (len <= 0) return;
  const int64_t seg = ((len + W - 1) / W + 31) & ~static_cast<int64_t>(31);
  const int64_t a = lo + static_cast<int64_t>(w) * seg;
  if (a >= hi) return;
  const int64_t b_end = a + seg < hi ? a + seg : hi;
  int* mine = rows + static_cast<int64_t>(w) * row_stride;
  const unsigned below = (1u << lane) - 1u;
  int b = a + lane < b_end ? bin(ids[a + lane]) : -1;
  for (int64_t s = a; s < b_end; s += 32) {
    const int64_t ns = s + 32;
    int nb = -1;
    if (ns < b_end) nb = ns + lane < b_end ? bin(ids[ns + lane]) : -1;
    const unsigned mask = align_equal_mask(b < 0 ? 0xFFFFFFFFu : static_cast<unsigned>(b), bits);
    const int rank = __popc(mask & below);
    if (b >= 0) {
      const int64_t pos = s + lane;
      if (pos < b_end) put(pos, b, start[b] + mine[b] + rank);
    }
    if (b >= 0 && rank == 0) mine[b] += __popc(mask);
    b = nb;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

__global__ void moe_count_and_offsets_kernel(const int32_t* __restrict__ topk_ids,
                                             int32_t* __restrict__ expert_counts,
                                             int64_t* __restrict__ expert_offsets,
                                             const int32_t* __restrict__ expert_map,
                                             const int64_t numel, const int num_experts,
                                             const int num_local_experts, const int align_m) {
  extern __shared__ int counts[];
  for (int i = threadIdx.x; i < kMaxExperts; i += blockDim.x) counts[i] = 0;
  __syncthreads();
  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int e = topk_ids[i];
    if (e < 0 || e >= num_experts) continue;
    const int l = local_expert(e, expert_map);
    if (l < 0 || l >= num_local_experts) continue;
    atomicAdd(&counts[l], 1);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  const int am = align_m > 0 ? align_m : 1;
  int64_t run = 0;
  for (int l = 0; l < num_local_experts; ++l) {
    const int n = counts[l];
    expert_counts[l] = n;
    expert_offsets[l] = run;
    const int64_t padded = static_cast<int64_t>((n + am - 1) / am) * am;
    run += padded;
  }
  expert_offsets[num_local_experts] = run;
}

__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ num_tokens_post_pad,
    int32_t* __restrict__ cumsum_workspace, const int32_t* __restrict__ expert_map,
    const int num_experts, const int block_size, const int64_t numel,
    const int max_num_tokens_padded) {
  extern __shared__ int smem[];
  const int E = num_experts;
  const int W = blockDim.x >> 5;
  const int row_stride = align_row_stride(E);
  int* rows = smem;                       // [W][E + 1] per-warp placement counters
  int* pre = rows + W * row_stride;       // [E] count of [0, chunk_lo), then base + that count
  int* tot = pre + E;                     // [E] count of [chunk_lo, numel), then first padding slot
  int* base = tot + E;                    // [E] first destination of every expert
  __shared__ int s_total;
  const int bs = block_size > 0 ? block_size : 1;
  const auto bin = [&](int e) {
    return (e >= 0 && e < E && (expert_map == nullptr || expert_map[e] >= 0)) ? e : -1;
  };
  // Slot chunk owned by this CTA. Placement is ordered by slot id, so (chunk, in-chunk rank) is
  // exactly the global order and the chunk's per-expert start rank is a plain prefix count.
  const int64_t chunk = gridDim.x > 0 ? (numel + gridDim.x - 1) / gridDim.x : numel;
  const int64_t lo = static_cast<int64_t>(blockIdx.x) * chunk;
  const int64_t hi = lo < numel ? (lo + chunk < numel ? lo + chunk : numel) : numel;
  for (int i = threadIdx.x; i < W * row_stride; i += blockDim.x) rows[i] = 0;
  for (int i = threadIdx.x; i < E; i += blockDim.x) pre[i] = 0;
  for (int i = threadIdx.x; i < E; i += blockDim.x) tot[i] = 0;
  __syncthreads();
  align_hist_split(pre, tot, topk_ids, numel, lo, bin);
  align_hist_rows(rows, row_stride, topk_ids, lo, hi, bin);
  __syncthreads();
  for (int i = threadIdx.x; i < E; i += blockDim.x) tot[i] += pre[i];
  __syncthreads();
  // Padded layout: base[e] is expert e's first destination (padding included, so `total` stays a
  // multiple of block_size), tot[e] becomes its first padding slot.
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int per = (E + 31) / 32;
    const int e0 = min(lane * per, E), e1 = min(e0 + per, E);
    int sum = 0;
    for (int e = e0; e < e1; ++e) sum += (tot[e] + bs - 1) / bs * bs;
    int incl = sum;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int v = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) incl += v;
    }
    int run = incl - sum;
    for (int e = e0; e < e1; ++e) {
      const int n = tot[e];
      const int padded = (n + bs - 1) / bs * bs;
      base[e] = run;
      tot[e] = run + n;
      run += padded;
    }
    if (lane == 31) s_total = incl;
  }
  // Per-warp exclusive prefix of this chunk's counts: rows[w][e] <- sum of the counts of warps < w.
  {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < E; e += W) {
      int c = lane < W ? rows[lane * row_stride + e] : 0;
      int incl = c;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_up_sync(0xFFFFFFFFu, incl, off);
        if (lane >= off) incl += v;
      }
      if (lane < W) rows[lane * row_stride + e] = incl - c;
    }
  }
  __syncthreads();
  const int total = s_total;
  for (int i = threadIdx.x; i < E; i += blockDim.x) pre[i] += base[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    num_tokens_post_pad[0] = total;
    cumsum_workspace[0] = total;
  }
  __syncthreads();
  // Block table and padding sentinels, one warp per (CTA, expert) pair. A padded block sits at the
  // end of every expert, so the sentinels are [first padding slot, next expert's first slot); those
  // positions are disjoint from the placement's writes, so both halves need no further sync.
  {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * W + (threadIdx.x >> 5);
    const int nw = gridDim.x * W;
    const int32_t sentinel = static_cast<int32_t>(numel);
    for (int e = gw; e < E; e += nw) {
      const int first = base[e] / bs;
      const int end = e + 1 < E ? base[e + 1] : total;
      const int last = end / bs;
      const int id = expert_map != nullptr ? expert_map[e] : e;
      for (int b = first + lane; b < last; b += 32) {
        if (b < max_num_tokens_padded / bs + 1) expert_ids[b] = id;
      }
      const int stop = end < max_num_tokens_padded ? end : max_num_tokens_padded;
      for (int i = tot[e] + lane; i < stop; i += 32) sorted_token_ids[i] = sentinel;
    }
  }
  int bits = 1;
  while ((1 << bits) < E) ++bits;
  align_place_chunk(rows, row_stride, pre, topk_ids, lo, hi, bits, bin,
                    [&](int64_t pos, int, int dest) {
                      if (dest < max_num_tokens_padded) sorted_token_ids[dest] = static_cast<int32_t>(pos);
                    });
}

// ---------------------------------------------------------------------------
// Permute
// ---------------------------------------------------------------------------

/// Deterministic slot -> row map: row = expert_offsets[l] + (rows of l from earlier slots). One block
/// of sort_warps(E_local) warps runs the stable counting sort; write_cursor is no longer touched.
__global__ void moe_permute_map_kernel(const int32_t* __restrict__ topk_ids,
                                       const int64_t* __restrict__ expert_offsets,
                                       const int32_t* __restrict__ expert_map,
                                       int32_t* __restrict__ permuted_row_to_slot,
                                       int32_t* __restrict__ slot_to_permuted_row,
                                       const int num_tokens, const int topk,
                                       const int num_local_experts) {
  extern __shared__ int smem[];
  const int E = num_local_experts;
  const int W = blockDim.x >> 5;
  int* cnt = smem;              // [W][E]
  int* base = smem + W * E;     // [E] first row of each local expert
  const int64_t numel = static_cast<int64_t>(num_tokens) * topk;
  for (int i = threadIdx.x; i < W * E; i += blockDim.x) cnt[i] = 0;
  for (int e = threadIdx.x; e < E; e += blockDim.x) base[e] = static_cast<int>(expert_offsets[e]);
  __syncthreads();
  const auto bin = [&](int e) {
    const int l = e >= 0 ? local_expert(e, expert_map) : -1;
    return (l >= 0 && l < E) ? l : -1;
  };
  // Rows come from expert_offsets, so only the within-expert rank is needed: count, scan over warps,
  // then place with base = expert_offsets.
  sort_count(cnt, topk_ids, numel, E, bin);
  __syncthreads();
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    int run = 0;
    for (int w = 0; w < W; ++w) {
      const int c = cnt[w * E + e];
      cnt[w * E + e] = run;
      run += c;
    }
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) slot_to_permuted_row[i] = -1;
  __syncthreads();
  sort_place(cnt, base, topk_ids, numel, E, bin, [&](int64_t pos, int, int row) {
    slot_to_permuted_row[pos] = row;
    permuted_row_to_slot[row] = static_cast<int32_t>(pos);
  });
}

template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_pack(const scalar_t* p, float* out) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
  } else if constexpr (VEC == 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    float tmp[8];
    dev::unpack8<scalar_t>(raw, tmp);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = tmp[i];
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const float2 a = dev::unpack2<scalar_t>(raw.x);
    const float2 b = dev::unpack2<scalar_t>(raw.y);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
  }
}

template <typename scalar_t, int VEC>
__device__ __forceinline__ void store_pack(scalar_t* p, const float* v) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    float4 o;
    o.x = v[0];
    o.y = v[1];
    o.z = v[2];
    o.w = v[3];
    *reinterpret_cast<float4*>(p) = o;
  } else if constexpr (VEC == 8) {
    uint4 raw;
    scalar_t tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) dev::from_f(tmp[i], v[i]);
    raw.x = *reinterpret_cast<uint32_t*>(&tmp[0]);
    raw.y = *reinterpret_cast<uint32_t*>(&tmp[2]);
    raw.z = *reinterpret_cast<uint32_t*>(&tmp[4]);
    raw.w = *reinterpret_cast<uint32_t*>(&tmp[6]);
    *reinterpret_cast<uint4*>(p) = raw;
  } else {
    scalar_t tmp[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) dev::from_f(tmp[i], v[i]);
    uint2 raw;
    raw.x = *reinterpret_cast<uint32_t*>(&tmp[0]);
    raw.y = *reinterpret_cast<uint32_t*>(&tmp[2]);
    *reinterpret_cast<uint2*>(p) = raw;
  }
}

template <typename scalar_t, ActQuantMode kQuant, int VEC_SIZE>
__global__ void moe_permute_kernel(const __grid_constant__ PermuteParams params) {
  const int64_t slot = blockIdx.x;
  if (slot >= params.num_tokens * params.topk) return;
  const int row = params.slot_to_permuted_row[slot];
  if (row < 0) return;
  const bool weight_input = params.apply_weight_on_input && params.topk_weights != nullptr;
  const float w = weight_input ? params.topk_weights[slot] : 1.0f;
  const int64_t src = slot / params.topk;
  const int hidden = params.hidden_size;
  if constexpr (kQuant == ActQuantMode::kNone) {
    for (int col = threadIdx.x * VEC_SIZE; col < hidden; col += blockDim.x * VEC_SIZE) {
      float v[VEC_SIZE];
      load_pack<scalar_t, VEC_SIZE>(static_cast<const scalar_t*>(params.hidden) +
                                        src * params.hidden_stride + col, v);
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) v[i] *= w;
      store_pack<scalar_t, VEC_SIZE>(static_cast<scalar_t*>(params.permuted) +
                                         row * params.permuted_stride + col, v);
    }
  } else {
    // One warp (32 lanes) quantizes one 1x128 group: 4 values per lane.
    const int group = blockIdx.y;
    const int group_size = params.quant.group_size > 0 ? params.quant.group_size : 128;
    const int lane = threadIdx.x & 31;
    const int base = group * group_size;
    if (base >= hidden) return;
    const int n = min(group_size, hidden - base);
    float v[4];
    float local_absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = base + lane * 4 + i;
      v[i] = col < base + n
                 ? dev::to_f(static_cast<const scalar_t*>(params.hidden)[src * params.hidden_stride + col]) * w
                 : 0.0f;
      local_absmax = fmaxf(local_absmax, fabsf(v[i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
      local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xFFFFFFFFu, local_absmax, off));
    }
    const float scale = fmaxf(local_absmax, 1e-4f) / 448.0f;
    uint8_t* q = static_cast<uint8_t*>(params.quant.q) +
                 static_cast<int64_t>(row) * params.quant.q_row_stride + base;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = base + lane * 4 + i;
      if (col < base + n) q[lane * 4 + i] = dev::e4m3_from_float_sat(v[i] / scale);
    }
    if (lane == 0) {
      float* scales = static_cast<float*>(params.quant.scales);
      scales[static_cast<int64_t>(row) * params.quant.scale_row_stride + group] = scale;
    }
  }
}

// ---------------------------------------------------------------------------
// Unpermute / sum
// ---------------------------------------------------------------------------

template <typename scalar_t, typename acc_t, int VEC_SIZE>
__global__ void moe_unpermute_reduce_kernel(
    scalar_t* __restrict__ out, const int64_t out_stride,
    const scalar_t* __restrict__ expert_out, const int64_t expert_out_stride,
    const int32_t* __restrict__ slot_to_permuted_row, const float* __restrict__ topk_weights,
    const scalar_t* __restrict__ shared_out, const scalar_t* __restrict__ residual,
    const float routed_scale, const bool weights_already_applied, const int num_tokens,
    const int topk, const int hidden_size) {
  const int64_t token = blockIdx.y;
  if (token >= num_tokens) return;
  const int col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (col >= hidden_size) return;
  const int n = min(VEC_SIZE, hidden_size - col);
  acc_t acc[VEC_SIZE];
  #pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) acc[i] = acc_t{};
  for (int k = 0; k < topk; ++k) {
    const int32_t row = slot_to_permuted_row[token * topk + k];
    if (row < 0) continue;
    const float w = weights_already_applied
                        ? 1.0f
                        : (topk_weights != nullptr ? topk_weights[token * topk + k] : 1.0f);
    const scalar_t* p = expert_out + static_cast<int64_t>(row) * expert_out_stride + col;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      if (i < n) acc[i] = fmaf(static_cast<float>(dev::to_f(p[i])), w, static_cast<float>(acc[i]));
    }
  }
  #pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    if (i >= n) break;
    float v = static_cast<float>(acc[i]) * routed_scale;
    if (shared_out != nullptr) v += dev::to_f(shared_out[token * out_stride + col + i]);
    if (residual != nullptr) v += dev::to_f(residual[token * out_stride + col + i]);
    dev::from_f(out[token * out_stride + col + i], v);
  }
}

/// out[t] = sum_k input[t, k], fp32 accumulate in ascending k. One thread owns VEC_SIZE columns of
/// one token: all TOPK 16-byte loads are issued before the first add (memory-level parallelism),
/// then a single vector store. hidden % VEC_SIZE == 0 and 16-byte alignment are checked on the host.
template <typename scalar_t, int TOPK, int VEC_SIZE>
__global__ void moe_sum_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
                               const int num_tokens, const int hidden_size) {
  const int64_t token = blockIdx.y;
  if (token >= num_tokens) return;
  const int col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (col >= hidden_size) return;
  const scalar_t* base = input + token * TOPK * static_cast<int64_t>(hidden_size) + col;
  Vec<scalar_t, VEC_SIZE> in[TOPK];
  #pragma unroll
  for (int k = 0; k < TOPK; ++k) in[k] = load_vec<scalar_t, VEC_SIZE>(base + static_cast<int64_t>(k) * hidden_size);
  Vec<scalar_t, VEC_SIZE> r;
  #pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) acc += dev::to_f(in[k][i]);
    dev::from_f(r[i], acc);
  }
  store_vec<scalar_t, VEC_SIZE>(out + token * hidden_size + col, r);
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

int dtype_bytes(DType t) {
  switch (t) {
    case DType::kFloat32: return 4;
    case DType::kFloat16:
    case DType::kBFloat16: return 2;
    case DType::kFloat8E4M3: return 1;
    default: return 0;
  }
}

constexpr int kAlignSmem = 3 * 1024 * static_cast<int>(sizeof(int));

}  // namespace

Status count_and_offsets(const Tensor& topk_ids, const Tensor& expert_counts,
                         const Tensor& expert_offsets, const Tensor& expert_map,
                         int32_t num_local_experts, int32_t align_m, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "count_and_offsets: CUDA backend only");
  }
  Status s = check_tensor(topk_ids, DType::kInt32, 1, 4, "count_and_offsets.topk_ids");
  if (s != Status::kSuccess) return s;
  s = check_tensor(expert_counts, DType::kInt32, 1, 4, "count_and_offsets.expert_counts");
  if (s != Status::kSuccess) return s;
  s = check_tensor(expert_offsets, DType::kInt64, 1, 8, "count_and_offsets.expert_offsets");
  if (s != Status::kSuccess) return s;
  const int E = static_cast<int>(expert_map.shape[0]);
  if (expert_counts.shape[0] != num_local_experts ||
      expert_offsets.shape[0] != num_local_experts + 1) {
    return detail::set_error(Status::kInvalidArgument, "count_and_offsets: output length mismatch");
  }
  if (num_local_experts < 0 || num_local_experts > kMaxExperts) {
    return detail::set_error(Status::kUnsupportedShape, "count_and_offsets: num_local_experts %d",
                             num_local_experts);
  }
  if (topk_ids.shape[0] == 0) return Status::kSuccess;
  const int num_experts = E > 0 ? E : num_local_experts;
  if (tensor_present(expert_map)) {
    s = check_tensor(expert_map, DType::kInt32, 1, 4, "count_and_offsets.expert_map");
    if (s != Status::kSuccess) return s;
  }
  moe_count_and_offsets_kernel<<<1, kAlignThreads, kAlignSmem, stream_of(ctx)>>>(
      static_cast<const int32_t*>(topk_ids.data), static_cast<int32_t*>(expert_counts.data),
      static_cast<int64_t*>(expert_offsets.data),
      tensor_present(expert_map) ? static_cast<const int32_t*>(expert_map.data) : nullptr,
      topk_ids.shape[0], num_experts, num_local_experts, align_m);
  return launch_error("moe_count_and_offsets_kernel");
}

Status align_block_size(const Tensor& topk_ids, const Tensor& sorted_token_ids,
                        const Tensor& expert_ids, const Tensor& num_tokens_post_pad,
                        const Tensor& expert_map, int32_t num_experts, int32_t block_size,
                        const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "align_block_size: CUDA backend only");
  }
  Status s = check_tensor(topk_ids, DType::kInt32, 1, 4, "align_block_size.topk_ids");
  if (s != Status::kSuccess) return s;
  s = check_tensor(sorted_token_ids, DType::kInt32, 1, 4, "align_block_size.sorted_token_ids");
  if (s != Status::kSuccess) return s;
  s = check_tensor(expert_ids, DType::kInt32, 1, 4, "align_block_size.expert_ids");
  if (s != Status::kSuccess) return s;
  s = check_tensor(num_tokens_post_pad, DType::kInt32, 1, 4, "align_block_size.num_tokens_post_pad");
  if (s != Status::kSuccess) return s;
  if (num_experts < 1 || num_experts > kMaxExperts || block_size < 1) {
    return detail::set_error(Status::kUnsupportedShape, "align_block_size: bad num_experts/block_size");
  }
  if (tensor_present(expert_map)) {
    s = check_tensor(expert_map, DType::kInt32, 1, 4, "align_block_size.expert_map");
    if (s != Status::kSuccess) return s;
    if (expert_map.shape[0] != num_experts) {
      return detail::set_error(Status::kInvalidArgument, "align_block_size: expert_map must be [E]");
    }
  }
  const size_t need = (static_cast<size_t>(num_experts) + 1) * sizeof(int32_t);
  if (ws.device_scratch == nullptr || ws.device_scratch_bytes < need) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "align_block_size: workspace needs %zu bytes (num_experts + 1)", need);
  }
  const int64_t numel = topk_ids.shape[0];
  const int64_t max_padded = sorted_token_ids.shape[0];
  if (max_padded <= 0 || expert_ids.shape[0] <= 0) {
    return detail::set_error(Status::kInvalidArgument, "align_block_size: empty output buffers");
  }
  const int64_t upper = numel + static_cast<int64_t>(num_experts) * (block_size - 1);
  if (upper > max_padded) {
    return detail::set_error(Status::kCapacityExceeded,
                             "align_block_size: sorted_token_ids holds %lld < upper bound %lld",
                             static_cast<long long>(max_padded), static_cast<long long>(upper));
  }
  const int64_t max_blocks = (max_padded + block_size - 1) / block_size + 1;
  if (expert_ids.shape[0] < max_blocks) {
    return detail::set_error(Status::kCapacityExceeded,
                             "align_block_size: expert_ids holds %lld < %lld blocks",
                             static_cast<long long>(expert_ids.shape[0]),
                             static_cast<long long>(max_blocks));
  }
  const int sort_threads = 32 * align_warps(num_experts);
  const int sort_smem = align_smem_bytes(num_experts, sort_threads / 32);
  const int ctas = align_ctas(numel);
  moe_align_block_size_kernel<<<ctas, sort_threads, sort_smem, stream_of(ctx)>>>(
      static_cast<const int32_t*>(topk_ids.data), static_cast<int32_t*>(sorted_token_ids.data),
      static_cast<int32_t*>(expert_ids.data), static_cast<int32_t*>(num_tokens_post_pad.data),
      static_cast<int32_t*>(ws.device_scratch),
      tensor_present(expert_map) ? static_cast<const int32_t*>(expert_map.data) : nullptr,
      num_experts, block_size, numel, static_cast<int>(max_padded));
  return launch_error("moe_align_block_size_kernel");
}

template <typename scalar_t, int VEC>
Status launch_permute_copy(const PermuteParams& p, int num_tokens, int topk, int hidden,
                           cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(num_tokens) * topk);
  moe_permute_kernel<scalar_t, ActQuantMode::kNone, VEC><<<grid, 256, 0, s>>>(p);
  return launch_error("moe_permute_kernel");
}

template <typename scalar_t, int VEC>
Status launch_permute_quant(const PermuteParams& p, int num_tokens, int topk, int hidden,
                            int group_size, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(num_tokens) * topk,
                  static_cast<unsigned>((hidden + group_size - 1) / group_size));
  moe_permute_kernel<scalar_t, ActQuantMode::kPerGroup, VEC><<<grid, 32, 0, s>>>(p);
  return launch_error("moe_permute_kernel");
}

Status permute(const PermuteParams& p, DType hidden_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "permute: CUDA backend only");
  }
  if (hidden_dtype != DType::kFloat32 && hidden_dtype != DType::kFloat16 &&
      hidden_dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "permute: hidden must be fp32, fp16 or bf16");
  }
  if (p.num_tokens < 0 || p.topk < 1 || p.hidden_size < 1) {
    return detail::set_error(Status::kInvalidArgument, "permute: bad shape");
  }
  if (p.hidden == nullptr || p.topk_ids == nullptr || p.expert_offsets == nullptr ||
      p.permuted == nullptr || p.permuted_row_to_slot == nullptr ||
      p.slot_to_permuted_row == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "permute: null operand");
  }
  if (p.num_local_experts < 1 || p.num_local_experts > kMaxExperts) {
    return detail::set_error(Status::kInvalidArgument, "permute: num_local_experts %d not in [1, %d]",
                             p.num_local_experts, kMaxExperts);
  }
  if (p.apply_weight_on_input && p.topk_weights == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "permute: weight_on_input needs topk_weights");
  }
  const int eb = dtype_bytes(hidden_dtype);
  const int vec = eb == 4 ? 4 : 8;
  if (p.hidden_size % vec != 0) {
    return detail::set_error(Status::kUnsupportedShape, "permute: hidden must be a multiple of %d", vec);
  }
  if (p.hidden_stride < p.hidden_size || p.permuted_stride < p.hidden_size) {
    return detail::set_error(Status::kInvalidArgument, "permute: leading dimension too small");
  }
  if (!aligned(p.hidden, 16) || !aligned(p.permuted, 16) ||
      (p.hidden_stride * eb) % 16 != 0 || (p.permuted_stride * eb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "permute: rows must be 16 B aligned");
  }
  cudaStream_t s = stream_of(ctx);
  const int map_warps = sort_warps(p.num_local_experts);
  moe_permute_map_kernel<<<1, 32 * map_warps, sort_smem_bytes(p.num_local_experts, map_warps), s>>>(
      p.topk_ids, p.expert_offsets, p.expert_map, p.permuted_row_to_slot, p.slot_to_permuted_row,
      p.num_tokens, p.topk, p.num_local_experts);
  Status st = launch_error("moe_permute_map_kernel");
  if (st != Status::kSuccess) return st;

  const bool quant = p.quant.mode != ActQuantMode::kNone;
  if (quant) {
    if (p.quant.mode != ActQuantMode::kPerGroup) {
      return detail::set_error(Status::kNotImplemented, "permute: only kPerGroup quantization");
    }
    if (p.quant.q == nullptr || p.quant.scales == nullptr || p.quant.q_dtype != DType::kFloat8E4M3) {
      return detail::set_error(Status::kInvalidArgument, "permute: quantized output needs e4m3 payload and scales");
    }
    const int group = p.quant.group_size > 0 ? p.quant.group_size : 128;
    if (group != 128 || p.hidden_size % 128 != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "permute: kPerGroup requires group_size 128 and hidden % 128 == 0");
    }
    if (p.quant.scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kNotImplemented, "permute: only row-major scales");
    }
    if (p.quant.q_row_stride < p.hidden_size ||
        p.quant.scale_row_stride < (p.hidden_size / group)) {
      return detail::set_error(Status::kInvalidArgument, "permute: quantized leading dimensions too small");
    }
    if (!aligned(p.quant.q, 16) || !aligned(p.quant.scales, 4)) {
      return detail::set_error(Status::kMisaligned, "permute: quantized output must be aligned");
    }
    if (hidden_dtype == DType::kBFloat16) {
      return launch_permute_quant<__nv_bfloat16, 8>(p, p.num_tokens, p.topk, p.hidden_size, group, s);
    }
    if (hidden_dtype == DType::kFloat16) {
      return launch_permute_quant<__half, 8>(p, p.num_tokens, p.topk, p.hidden_size, group, s);
    }
    return launch_permute_quant<float, 4>(p, p.num_tokens, p.topk, p.hidden_size, group, s);
  }
  if (hidden_dtype == DType::kBFloat16) return launch_permute_copy<__nv_bfloat16, 8>(p, p.num_tokens, p.topk, p.hidden_size, s);
  if (hidden_dtype == DType::kFloat16) return launch_permute_copy<__half, 8>(p, p.num_tokens, p.topk, p.hidden_size, s);
  return launch_permute_copy<float, 4>(p, p.num_tokens, p.topk, p.hidden_size, s);
}

template <typename scalar_t, int VEC>
Status launch_unpermute(const Tensor& out, const Tensor& expert_out,
                        const Tensor& slot_to_permuted_row, const Tensor& topk_weights,
                        const Tensor& shared_out, const Tensor& residual, float routed_scale,
                        bool weights_already_applied, int num_tokens, int topk, int hidden,
                        cudaStream_t s) {
  const int threads = 128;  // hidden 7168 / 8 = 896 vectors = 7 blocks, no idle lanes
  const int64_t cols = static_cast<int64_t>(threads) * VEC;
  const dim3 grid(static_cast<unsigned>((hidden + cols - 1) / cols),
                  static_cast<unsigned>(num_tokens));
  moe_unpermute_reduce_kernel<scalar_t, float, VEC><<<grid, threads, 0, s>>>(
      static_cast<scalar_t*>(out.data), out.stride[0],
      static_cast<const scalar_t*>(expert_out.data), expert_out.stride[0],
      static_cast<const int32_t*>(slot_to_permuted_row.data),
      tensor_present(topk_weights) ? static_cast<const float*>(topk_weights.data) : nullptr,
      tensor_present(shared_out) ? static_cast<const scalar_t*>(shared_out.data) : nullptr,
      tensor_present(residual) ? static_cast<const scalar_t*>(residual.data) : nullptr,
      routed_scale, weights_already_applied, num_tokens, topk, hidden);
  return launch_error("moe_unpermute_reduce_kernel");
}

Status unpermute_reduce(const Tensor& out, const Tensor& expert_out,
                        const Tensor& slot_to_permuted_row, const Tensor& topk_weights,
                        const Tensor& shared_out, const Tensor& residual, float routed_scale,
                        bool weights_already_applied, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "unpermute_reduce: CUDA backend only");
  }
  Status s = check_tensor(out, out.dtype, 2, 2, "unpermute_reduce.out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(expert_out, out.dtype, 2, 2, "unpermute_reduce.expert_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(slot_to_permuted_row, DType::kInt32, 1, 4, "unpermute_reduce.slot_to_permuted_row");
  if (s != Status::kSuccess) return s;
  const int num_tokens = static_cast<int>(out.shape[0]);
  const int hidden = static_cast<int>(out.shape[1]);
  if (num_tokens <= 0) return Status::kSuccess;
  const int topk = static_cast<int>(slot_to_permuted_row.shape[0] / num_tokens);
  if (topk < 1 || static_cast<int64_t>(num_tokens) * topk != slot_to_permuted_row.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "unpermute_reduce: slot array must be [T*topk]");
  }
  if (tensor_present(topk_weights)) {
    s = check_tensor(topk_weights, DType::kFloat32, 1, 4, "unpermute_reduce.topk_weights");
    if (s != Status::kSuccess) return s;
    if (topk_weights.shape[0] != static_cast<int64_t>(num_tokens) * topk) {
      return detail::set_error(Status::kInvalidArgument, "unpermute_reduce: weight length mismatch");
    }
  }
  if (tensor_present(shared_out)) {
    s = check_tensor(shared_out, out.dtype, 2, 2, "unpermute_reduce.shared_out");
    if (s != Status::kSuccess) return s;
  }
  if (tensor_present(residual)) {
    s = check_tensor(residual, out.dtype, 2, 2, "unpermute_reduce.residual");
    if (s != Status::kSuccess) return s;
  }
  if (expert_out.shape[1] != hidden) {
    return detail::set_error(Status::kInvalidArgument, "unpermute_reduce: hidden mismatch");
  }
  const int vec = dtype_bytes(out.dtype) == 4 ? 4 : 8;
  if (hidden % vec != 0) {
    return detail::set_error(Status::kUnsupportedShape, "unpermute_reduce: hidden must be a multiple of %d", vec);
  }
  const cudaStream_t s2 = stream_of(ctx);
  if (out.dtype == DType::kBFloat16) {
    if (!aligned(out.data, 16) || (out.stride[0] * 2) % 16 != 0) {
      return detail::set_error(Status::kMisaligned, "unpermute_reduce: out rows must be 16 B aligned");
    }
    return launch_unpermute<__nv_bfloat16, 8>(out, expert_out, slot_to_permuted_row, topk_weights,
                                              shared_out, residual, routed_scale,
                                              weights_already_applied, num_tokens, topk, hidden, s2);
  }
  if (out.dtype == DType::kFloat16) {
    if (!aligned(out.data, 16) || (out.stride[0] * 2) % 16 != 0) {
      return detail::set_error(Status::kMisaligned, "unpermute_reduce: out rows must be 16 B aligned");
    }
    return launch_unpermute<__half, 8>(out, expert_out, slot_to_permuted_row, topk_weights,
                                       shared_out, residual, routed_scale, weights_already_applied,
                                       num_tokens, topk, hidden, s2);
  }
  if (out.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "unpermute_reduce: %s", dtype_name(out.dtype));
  }
  return launch_unpermute<float, 4>(out, expert_out, slot_to_permuted_row, topk_weights, shared_out,
                                    residual, routed_scale, weights_already_applied, num_tokens,
                                    topk, hidden, s2);
}

template <typename scalar_t, int TOPK, int VEC>
Status launch_sum(const Tensor& out, const Tensor& input, int num_tokens, int hidden,
                  cudaStream_t s) {
  const int threads = 128;  // hidden 7168 / 8 = 896 vectors = 7 blocks, no idle lanes
  const int64_t cols = static_cast<int64_t>(threads) * VEC;
  const dim3 grid(static_cast<unsigned>((hidden + cols - 1) / cols),
                  static_cast<unsigned>(num_tokens));
  moe_sum_kernel<scalar_t, TOPK, VEC><<<grid, threads, 0, s>>>(
      static_cast<scalar_t*>(out.data), static_cast<const scalar_t*>(input.data), num_tokens, hidden);
  return launch_error("moe_sum_kernel");
}

template <typename scalar_t, int VEC>
Status dispatch_sum(const Tensor& out, const Tensor& input, int topk, int num_tokens, int hidden,
                    cudaStream_t s) {
  switch (topk) {
    case 1: return launch_sum<scalar_t, 1, VEC>(out, input, num_tokens, hidden, s);
    case 2: return launch_sum<scalar_t, 2, VEC>(out, input, num_tokens, hidden, s);
    case 4: return launch_sum<scalar_t, 4, VEC>(out, input, num_tokens, hidden, s);
    case 8: return launch_sum<scalar_t, 8, VEC>(out, input, num_tokens, hidden, s);
    case 16: return launch_sum<scalar_t, 16, VEC>(out, input, num_tokens, hidden, s);
    default:
      return detail::set_error(Status::kUnsupportedShape, "moe_sum: topk %d not in {1,2,4,8,16}", topk);
  }
}

Status moe_sum(const Tensor& out, const Tensor& input, int32_t topk, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "moe_sum: CUDA backend only");
  }
  Status s = check_tensor(out, out.dtype, 2, 2, "moe_sum.out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(input, out.dtype, 3, 2, "moe_sum.input");
  if (s != Status::kSuccess) return s;
  if (input.shape[0] != out.shape[0] || input.shape[1] != topk || input.shape[2] != out.shape[1]) {
    return detail::set_error(Status::kInvalidArgument, "moe_sum: input must be [T, topk, H]");
  }
  const int num_tokens = static_cast<int>(out.shape[0]);
  const int hidden = static_cast<int>(out.shape[1]);
  const int vec = dtype_bytes(out.dtype) == 4 ? 4 : 8;
  if (hidden % vec != 0) {
    return detail::set_error(Status::kUnsupportedShape, "moe_sum: hidden must be a multiple of %d", vec);
  }
  if (!tensor_is_contiguous(input)) {
    return detail::set_error(Status::kUnsupportedLayout, "moe_sum: input must be contiguous");
  }
  if (!is_aligned(out.data, 16) || !is_aligned(input.data, 16) || (out.stride[0] * dtype_bytes(out.dtype)) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "moe_sum: out and input rows must be 16 B aligned");
  }
  const cudaStream_t s2 = stream_of(ctx);
  if (out.dtype == DType::kBFloat16) return dispatch_sum<__nv_bfloat16, 8>(out, input, topk, num_tokens, hidden, s2);
  if (out.dtype == DType::kFloat16) return dispatch_sum<__half, 8>(out, input, topk, num_tokens, hidden, s2);
  if (out.dtype == DType::kFloat32) return dispatch_sum<float, 4>(out, input, topk, num_tokens, hidden, s2);
  return detail::set_error(Status::kUnsupportedDtype, "moe_sum: %s", dtype_name(out.dtype));
}

}  // namespace kops::moe
