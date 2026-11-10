// =============================================================================
// kernel-ops | csrc/attention/sparse.cu
// B14 A11: sparse_topk — exact top-K selection over the fp32 score row of every
// query (row) of a variable-length score matrix.
//
// This file carries the A11 scoring/selection group. The sparse_topk group is
// the radix select (`sparse_topk_kernel<TOPK>` and the chunked cooperative fast
// path below); the lightning indexer and sparse MLA bodies follow it, and the V4
// compressor / compressed attention land later.
//
// Algorithm (one CTA per row, no workspace, no host round trip):
//   1. Fold every score into a monotone unsigned key (IEEE-754 bits with the
//      sign folded in), so unsigned comparison is numeric comparison. NaN ranks
//      with -inf (a NaN never wins a slot) and -0.0 with +0.0 (the index tie
//      rule decides), so the key order matches the documented
//      (value desc, index asc) contract exactly.
//   2. Three histogram passes over the 32 key bits (11 + 11 + 10). A pass
//      histograms only the entries that still match the prefix resolved so far
//      and keeps the digit bucket that holds the K-th largest entry. 2048 shared
//      bins cover the widest digit, so the row is never sorted and no pass needs
//      O(max_kv * TOPK) storage.
//   3. One collect pass: every entry above the exact threshold is emitted with
//      a shared atomic slot counter (the output order is unspecified); exact
//      ties at the threshold are buffered up to TOPK columns and the K-above
//      lowest column indices win, which is the deterministic tie rule. A tie set
//      larger than TOPK falls back to three more column-index passes, so an
//      all-equal row still selects the lowest indices (lowest TOPK of them).
//
// Fast path (rows == 1, max_kv >= 256 Ki, caller-supplied workspace):
//   One cooperative launch, three chunked passes over the row, so the whole
//   memory system participates instead of one SM:
//     1. every CTA histograms the top 11 key bits of its chunk in shared memory
//        and merges the non-zero bins into a global 2048-bin histogram;
//     2. after a grid sync the grid resolves the bucket that holds the K-th
//        score and re-scans: entries above that bucket are emitted into the
//        output (one global slot reservation per CTA, not per entry) and the
//        entries inside it fill a second 2048-bin histogram (next 11 key bits);
//     3. after a second grid sync it emits the entries above the resulting 22-bit
//        bucket and buffers the rest - the candidates, ~1 entry for uniform
//        scores - so the last stage resolves the exact 32-bit threshold and the
//        column tie rule inside one CTA.
//   A row whose 22-bit bucket holds more than kTopkFastCand entries (a constant
//   or otherwise tie-heavy row) reselects itself with the exact single-CTA body
//   inside the same launch, so the fast path never changes the result. The
//   workspace is caller-owned (sparse_topk_workspace_bytes); the path adds no
//   allocation, no device query and no host synchronization, and stays
//   CUDA-graph capturable.
//
// Rows with fewer than TOPK visible columns write -1 for the unused index slots
// (and -inf into `values` when it is present); such a row is copied in one pass
// because every visible column fits. Indices are columns of the score matrix
// (absolute_indices = true) or offsets from row_starts (false); with
// to_page_slots they become physical cache slots through the per-request block
// table (indexed by the absolute column) and an unmapped page yields -1.
// row_starts / row_ends / token_to_request are device data: the caller guarantees
// 0 <= start <= end <= max_kv and 0 <= token_to_request[row] < block_table rows.
//
// Launch: one CTA of 512 threads per row, so sparse_topk_kernel<TOPK> requires
// blockDim.x == 512 (the shared histogram bucket search is a 32-lane chunk
// scan). No allocation, no device query, no host synchronization: the entry is
// workspace-free and CUDA-graph capturable. The host validates every shape,
// dtype, stride, alignment and page-table argument before the single launch.
// sparse_topk_ws takes the same arguments plus a caller-owned workspace and only
// differs where the fast path applies (rows == 1 and max_kv >= 256 Ki): a
// missing, too small or misaligned workspace is not an error, it selects the
// workspace-free path above.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
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
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace kops::attn {
namespace {

/// Threads per row (CTA). The histogram bucket search is one 32-lane chunk
/// scan, so the kernel requires exactly this block size.
constexpr int kTopkThreads = 512;
/// Widest radix digit: 11 bits -> three passes over the 32 key bits and 8 KiB
/// of shared histogram.
constexpr int kHistBins = 2048;
constexpr int kRadixPasses = 3;
__device__ constexpr int kRadixShift[kRadixPasses] = {21, 10, 0};
__device__ constexpr int kRadixWidth[kRadixPasses] = {11, 11, 10};

__device__ __forceinline__ float topk_neg_inf() { return __int_as_float(0xff800000); }

/// Total order over fp32 as an unsigned key: unsigned(a) < unsigned(b) exactly
/// when score a ranks below score b. NaN maps to the -inf key and -0.0 to the
/// +0.0 key, matching the documented tie rule.
__device__ __forceinline__ uint32_t topk_key(float v) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0xff800000u;  // NaN -> -inf
  if (u == 0x80000000u) u = 0u;                          // -0.0 -> +0.0
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Visit every column of [lo, hi) of one score row with 16-byte loads: a
/// thread reads four consecutive floats per step, with scalar edges until the
/// row is 16-byte aligned. The body never uses warp intrinsics, so the head and
/// tail may diverge.
template <typename Fn>
__device__ __forceinline__ void topk_each(const float* __restrict__ row, int64_t lo, int64_t hi,
                                          Fn fn) {
  const int64_t block = static_cast<int64_t>(blockDim.x);
  const float* p = row + lo;
  const int64_t n = hi - lo;
  if (n <= 0) return;
  int64_t head = static_cast<int64_t>((16u - (reinterpret_cast<uintptr_t>(p) & 15u)) & 15u) >> 2;
  if (head > n) head = n;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < head; i += block) fn(lo + i, p[i]);
  const float4* p4 = reinterpret_cast<const float4*>(p + head);
  const int64_t nv = (n - head) >> 2;
  for (int64_t v = static_cast<int64_t>(threadIdx.x); v < nv; v += block) {
    const float4 q = __ldg(p4 + v);
    const int64_t col = lo + head + v * 4;
    fn(col, q.x);
    fn(col + 1, q.y);
    fn(col + 2, q.z);
    fn(col + 3, q.w);
  }
  for (int64_t i = head + (nv << 2) + static_cast<int64_t>(threadIdx.x); i < n; i += block) {
    fn(lo + i, p[i]);
  }
}

/// One histogram pass of the value-key radix: count the `width`-bit digit of all
/// columns whose key matches the resolved prefix (mask, sel).
__device__ __forceinline__ void topk_hist_pass(const float* __restrict__ row, int64_t lo,
                                               int64_t hi, uint32_t mask, uint32_t sel, int shift,
                                               int width, uint32_t* __restrict__ hist) {
  const uint32_t digit_mask = (1u << width) - 1u;
  topk_each(row, lo, hi, [&](int64_t, float v) {
    const uint32_t key = topk_key(v);
    if ((key & mask) != sel) return;
    atomicAdd(&hist[(key >> shift) & digit_mask], 1u);
  });
}

/// Bucket holding the `nth` largest entry (1-based) of a histogram, plus the
/// number of entries in higher buckets. Warp 0 covers the bins with 32
/// descending chunks of `nbins / 32` bins; requires 32 <= nbins and nbins % 32
/// == 0, which holds for 1024 and 2048.
__device__ __forceinline__ void topk_pick_desc(const uint32_t* __restrict__ hist, int nbins,
                                               int32_t nth, int32_t* bin_out, int32_t* above_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if ((threadIdx.x >> 5) == 0) {
    const int per = nbins / 32;
    const int hi_bin = nbins - 1 - lane * per;
    uint32_t sum = 0;
    for (int k = 0; k < per; ++k) sum += hist[hi_bin - k];
    uint32_t inclusive = sum;
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, inclusive, off);
      if (lane >= off) inclusive += up;
    }
    const uint32_t above = inclusive - sum;
    const uint32_t n = static_cast<uint32_t>(nth);
    if (n > above && n <= inclusive) {
      uint32_t acc = above;
      for (int k = 0; k < per; ++k) {
        const int bin = hi_bin - k;
        const uint32_t c = hist[bin];
        if (acc + c >= n) {
          *bin_out = bin;
          *above_out = static_cast<int32_t>(acc);
          break;
        }
        acc += c;
      }
    }
  }
  __syncthreads();
}

/// Bucket holding the `nth` smallest entry (1-based) of a histogram, plus the
/// number of entries in lower buckets. Mirror of topk_pick_desc.
__device__ __forceinline__ void topk_pick_asc(const uint32_t* __restrict__ hist, int nbins,
                                              int32_t nth, int32_t* bin_out, int32_t* below_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if ((threadIdx.x >> 5) == 0) {
    const int per = nbins / 32;
    const int lo_bin = lane * per;
    uint32_t sum = 0;
    for (int k = 0; k < per; ++k) sum += hist[lo_bin + k];
    uint32_t inclusive = sum;
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, inclusive, off);
      if (lane >= off) inclusive += up;
    }
    const uint32_t below = inclusive - sum;
    const uint32_t n = static_cast<uint32_t>(nth);
    if (n > below && n <= inclusive) {
      uint32_t acc = below;
      for (int k = 0; k < per; ++k) {
        const int bin = lo_bin + k;
        const uint32_t c = hist[bin];
        if (acc + c >= n) {
          *bin_out = bin;
          *below_out = static_cast<int32_t>(acc);
          break;
        }
        acc += c;
      }
    }
  }
  __syncthreads();
}

/// Column of the selected entry: relative to row_starts, absolute, or the
/// physical KV slot through the per-request block table. An unmapped page (or a
/// request outside the table) yields -1.
__device__ __forceinline__ int32_t topk_map_column(
    int row, int64_t col, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ token_to_request, int block_table_stride, int page_size,
    bool absolute_indices, bool to_page_slots, int64_t row_start) {
  if (!to_page_slots) {
    return static_cast<int32_t>(absolute_indices ? col : col - row_start);
  }
  const int request = (token_to_request != nullptr) ? token_to_request[row] : row;
  const int64_t page_index = col / page_size;
  if (request < 0 || page_index >= block_table_stride) return -1;
  const int32_t page = block_table[static_cast<int64_t>(request) * block_table_stride + page_index];
  if (page < 0) return -1;
  const int64_t slot = static_cast<int64_t>(page) * page_size + (col % page_size);
  return static_cast<int32_t>(slot);
}

// -----------------------------------------------------------------------------
// Multi-CTA fast path: workspace layout, chunk partition, candidate record.
// -----------------------------------------------------------------------------

/// One buffered candidate of the fast path: an entry of the 22-bit key bucket
/// that holds the K-th score. `col` is the absolute column in the score row.
struct TopkFastCand {
  float value;
  int32_t col;
};

/// Candidates buffered before the last stage falls back to the exact single-CTA
/// body. Uniform scores leave ~1 entry (the bucket that holds the K-th score), so
/// this is ~3 orders of magnitude of headroom; a constant row overflows on
/// purpose and takes the fallback.
constexpr int kTopkFastCand = 4096;
/// Threads per CTA of the fast path.
constexpr int kTopkFastThreads = 512;
/// Rows at least this wide switch to the chunked path when a workspace exists:
/// one CTA only reaches a single SM's slice of the memory system.
constexpr int64_t kTopkFastMinKv = 262144;
/// CTAs the fast path asks for per SM, capped by cooperative-launch residency.
constexpr int kTopkFastCtasPerSm = 2;
/// Visible columns per CTA below which extra CTAs stop paying for their syncs.
constexpr int64_t kTopkFastMinChunk = 4096;

/// Workspace layout of a 256-byte aligned device_scratch:
///   [0, 8192)        hist1  uint32[2048]  top-11-bit key digits
///   [8192, 16384)    hist2  uint32[2048]  next-11-bit digits inside hist1's bucket
///   [16384, 18432)   ctrl   uint32[512]   [0] candidates, [1] emitted slots
///   [18432, 51200)   cand   TopkFastCand[4096] value + absolute column
constexpr size_t kTopkFastHistBytes = kHistBins * sizeof(uint32_t);
constexpr size_t kTopkFastCtrlOffset = 2 * kTopkFastHistBytes;
constexpr size_t kTopkFastCtrlBytes = 512 * sizeof(uint32_t);
constexpr size_t kTopkFastCandOffset = kTopkFastCtrlOffset + kTopkFastCtrlBytes;
constexpr size_t kTopkFastWorkspaceBytes =
    kTopkFastCandOffset + kTopkFastCand * sizeof(TopkFastCand);
static_assert(kTopkFastWorkspaceBytes == 51200, "fast-path workspace layout changed");
/// ctrl[] words.
constexpr int kTopkFastCtrlCand = 0;   // candidates buffered (may exceed kTopkFastCand)
constexpr int kTopkFastCtrlAbove = 1;  // output slots emitted by the pass flushes

/// Chunk of [start, end) owned by this CTA. Every CTA of the launch reads the same
/// row data, so the partition is uniform and no CTA skips a grid sync.
__device__ __forceinline__ void topk_fast_chunk(int64_t start, int64_t end, int64_t* lo,
                                               int64_t* hi) {
  const int64_t blocks = static_cast<int64_t>(gridDim.x);
  const int64_t chunk = (end - start + blocks - 1) / blocks;
  int64_t l = start + static_cast<int64_t>(blockIdx.x) * chunk;
  if (l > end) l = end;
  int64_t h = l + chunk;
  if (h > end) h = end;
  *lo = l;
  *hi = h;
}

}  // namespace

// -----------------------------------------------------------------------------
// sparse_topk_kernel — one CTA per row (blockDim.x == 512)
// -----------------------------------------------------------------------------

namespace {

/// Exact top-K of one score row over [start, end). `out` / `val` are the row's
/// output slots, `srow` the score row. `hist` (kHistBins bins) and `tie_col`
/// (TOPK columns) are the calling block's scratch: the single-CTA kernel owns
/// them per row, and the last stage of the multi-CTA fast path reuses its own
/// scratch, so both paths run through the same exact body. `row` only feeds the
/// page-slot mapping.
template <int TOPK>
__device__ void sparse_topk_row_body(int32_t* __restrict__ out, float* __restrict__ val,
                                     const float* __restrict__ srow, int64_t start, int64_t end,
                                     const int32_t* __restrict__ block_table,
                                     const int32_t* __restrict__ token_to_request,
                                     int block_table_stride, int page_size, bool absolute_indices,
                                     bool to_page_slots, int row, uint32_t* __restrict__ hist,
                                     int32_t* __restrict__ tie_col) {
  __shared__ int32_t s_bin, s_above, s_gt, s_tie, s_write;

  if (start < 0) start = 0;
  if (end < start) end = start;
  const int64_t visible = end - start;
  const int m = static_cast<int>(visible < TOPK ? visible : static_cast<int64_t>(TOPK));
  const int block = static_cast<int>(blockDim.x);

  if (m == 0) {
    for (int i = static_cast<int>(threadIdx.x); i < TOPK; i += block) {
      out[i] = -1;
      if (val != nullptr) val[i] = topk_neg_inf();
    }
    return;
  }

  // A row no longer than K selects every visible column: one pass, no radix.
  if (visible <= TOPK) {
    for (int64_t col = start + static_cast<int64_t>(threadIdx.x); col < end;
         col += static_cast<int64_t>(block)) {
      const int32_t slot = static_cast<int32_t>(col - start);
      out[slot] = topk_map_column(row, col, block_table, token_to_request, block_table_stride,
                                  page_size, absolute_indices, to_page_slots, start);
      if (val != nullptr) val[slot] = srow[col];
    }
    for (int i = m + static_cast<int>(threadIdx.x); i < TOPK; i += block) {
      out[i] = -1;
      if (val != nullptr) val[i] = topk_neg_inf();
    }
    return;
  }

  // ---- 1. exact value threshold: 11 + 11 + 10 bits, MSB first ----------------
  int32_t remaining = m;  // entries still to pick inside the current prefix
  uint32_t sel = 0u;      // resolved key prefix (low bits zero)
  uint32_t mask = 0u;     // bits resolved so far
#pragma unroll
  for (int p = 0; p < kRadixPasses; ++p) {
    const int nbins = 1 << kRadixWidth[p];
    for (int i = static_cast<int>(threadIdx.x); i < nbins; i += block) hist[i] = 0u;
    __syncthreads();
    topk_hist_pass(srow, start, end, mask, sel, kRadixShift[p], kRadixWidth[p], hist);
    __syncthreads();
    topk_pick_desc(hist, nbins, remaining, &s_bin, &s_above);
    remaining -= s_above;
    sel |= static_cast<uint32_t>(s_bin) << kRadixShift[p];
    mask |= static_cast<uint32_t>(nbins - 1) << kRadixShift[p];
    __syncthreads();
  }
  const uint32_t threshold = sel;  // exact 32-bit key of the K-th largest entry

  // ---- 2. collect everything above the threshold, buffer the exact ties ------
  if (threadIdx.x == 0) {
    s_gt = 0;
    s_tie = 0;
  }
  __syncthreads();
  topk_each(srow, start, end, [&](int64_t col, float v) {
    const uint32_t key = topk_key(v);
    if (key > threshold) {
      const int32_t slot = atomicAdd(&s_gt, 1);
      if (slot < TOPK) {
        out[slot] = topk_map_column(row, col, block_table, token_to_request, block_table_stride,
                                    page_size, absolute_indices, to_page_slots, start);
        if (val != nullptr) val[slot] = v;
      }
    } else if (key == threshold) {
      const int32_t slot = atomicAdd(&s_tie, 1);
      if (slot < TOPK) tie_col[slot] = static_cast<int32_t>(col);
    }
  });
  __syncthreads();
  const int32_t above = s_gt;      // count above the threshold, always < m
  const int32_t num_ties = s_tie;  // count at the threshold (may exceed TOPK)
  const int32_t need = m - above;  // tie entries that must be taken (>= 1)

  // ---- 3. deterministic ties: the `need` lowest column indices ---------------
  int32_t cutoff = 0x7fffffff;  // take every buffered tie when need == num_ties
  if (num_ties <= TOPK) {
    if (need < num_ties) {
      // Three index-bit passes over the buffered columns in shared memory.
      int32_t rem = need;
      uint32_t isel = 0u, imask = 0u;
#pragma unroll
      for (int p = 0; p < kRadixPasses; ++p) {
        const int nbins = 1 << kRadixWidth[p];
        for (int i = static_cast<int>(threadIdx.x); i < nbins; i += block) hist[i] = 0u;
        __syncthreads();
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < num_ties; i += block) {
          const uint32_t u = static_cast<uint32_t>(tie_col[i]);
          if ((u & imask) == isel) atomicAdd(&hist[(u >> kRadixShift[p]) & (nbins - 1)], 1u);
        }
        __syncthreads();
        topk_pick_asc(hist, nbins, rem, &s_bin, &s_above);
        rem -= s_above;
        isel |= static_cast<uint32_t>(s_bin) << kRadixShift[p];
        imask |= static_cast<uint32_t>(nbins - 1) << kRadixShift[p];
        __syncthreads();
      }
      cutoff = static_cast<int32_t>(isel);
    }
    if (threadIdx.x == 0) s_write = 0;
    __syncthreads();
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < num_ties; i += block) {
      const int32_t col = tie_col[i];
      if (col > cutoff) continue;
      const int32_t slot = above + atomicAdd(&s_write, 1);
      out[slot] = topk_map_column(row, col, block_table, token_to_request, block_table_stride,
                                  page_size, absolute_indices, to_page_slots, start);
      if (val != nullptr) val[slot] = srow[col];
    }
  } else {
    // More exact ties than the shared buffer holds (e.g. a constant row): select
    // the `need` lowest columns with the same three index-bit passes over the
    // row. Columns are unique, so `col <= cutoff` picks exactly `need` of them.
    int32_t rem = need;
    uint32_t isel = 0u, imask = 0u;
#pragma unroll
    for (int p = 0; p < kRadixPasses; ++p) {
      const int nbins = 1 << kRadixWidth[p];
      for (int i = static_cast<int>(threadIdx.x); i < nbins; i += block) hist[i] = 0u;
      __syncthreads();
      topk_each(srow, start, end, [&](int64_t col, float v) {
        if (topk_key(v) != threshold) return;
        const uint32_t u = static_cast<uint32_t>(col);
        if ((u & imask) == isel) atomicAdd(&hist[(u >> kRadixShift[p]) & (nbins - 1)], 1u);
      });
      __syncthreads();
      topk_pick_asc(hist, nbins, rem, &s_bin, &s_above);
      rem -= s_above;
      isel |= static_cast<uint32_t>(s_bin) << kRadixShift[p];
      imask |= static_cast<uint32_t>(nbins - 1) << kRadixShift[p];
      __syncthreads();
    }
    cutoff = static_cast<int32_t>(isel);
    if (threadIdx.x == 0) s_write = 0;
    __syncthreads();
    topk_each(srow, start, end, [&](int64_t col, float v) {
      if (topk_key(v) != threshold) return;
      if (col > static_cast<int64_t>(cutoff)) return;
      const int32_t slot = above + atomicAdd(&s_write, 1);
      out[slot] = topk_map_column(row, col, block_table, token_to_request, block_table_stride,
                                  page_size, absolute_indices, to_page_slots, start);
      if (val != nullptr) val[slot] = v;
    });
  }

  // ---- 4. pad a short row (or a row with fewer visible columns than TOPK) ----
  for (int i = m + static_cast<int>(threadIdx.x); i < TOPK; i += block) {
    out[i] = -1;
    if (val != nullptr) val[i] = topk_neg_inf();
  }
}

}  // namespace

template <int TOPK>
__global__ void sparse_topk_kernel(int32_t* __restrict__ indices, float* __restrict__ values,
                                   const float* __restrict__ scores, const int64_t scores_stride,
                                   const int32_t* __restrict__ row_starts,
                                   const int32_t* __restrict__ row_ends,
                                   const int32_t* __restrict__ block_table,
                                   const int32_t* __restrict__ token_to_request,
                                   const int block_table_stride, const int page_size,
                                   const bool absolute_indices, const bool to_page_slots,
                                   const int num_rows) {
  __shared__ uint32_t hist[kHistBins];
  __shared__ int32_t tie_col[TOPK];

  const int row = static_cast<int>(blockIdx.x);
  if (row >= num_rows) return;
  const int64_t start = (row_starts != nullptr) ? static_cast<int64_t>(row_starts[row]) : 0;
  const int64_t end = static_cast<int64_t>(row_ends[row]);
  sparse_topk_row_body<TOPK>(
      indices + static_cast<int64_t>(row) * TOPK,
      (values != nullptr) ? values + static_cast<int64_t>(row) * TOPK : nullptr,
      scores + static_cast<int64_t>(row) * scores_stride, start, end, block_table,
      token_to_request, block_table_stride, page_size, absolute_indices, to_page_slots, row, hist,
      tie_col);
}

// -----------------------------------------------------------------------------
// sparse_topk_fast_kernel — the multi-CTA fast path of rows == 1 and a wide row.
//
// One cooperative launch, three chunked passes over the single visible row:
//   pass 1  hist1[top 11 key bits], merged from every CTA's shared histogram;
//   pass 2  emit the entries above hist1's bucket (the K-th score is inside it)
//           and fill hist2[next 11 key bits] from the entries inside the bucket;
//   pass 3  emit the entries above hist2's bucket and buffer the rest (the
//           candidates: entries of the 22-bit bucket that holds the K-th score);
//   stage 4 (CTA 0) turns the candidates into the exact 32-bit threshold, applies
//           the (value desc, column asc) tie rule and pads the row.
// Every CTA reads the same row_starts / row_ends, so the early return and the
// chunk partition are uniform: a grid sync is never skipped by part of the grid.
// A candidate overflow (constant / tie-heavy row) reselects the row with the
// exact single-CTA body inside stage 4, which rewrites every output slot.
// -----------------------------------------------------------------------------
namespace {

/// One reserved output run for a buffered pass flush: a pass costs one global
/// atomic per CTA instead of one per selected entry. Thread 0 only.
__device__ __forceinline__ int32_t topk_fast_reserve(uint32_t* __restrict__ ctrl, int32_t count) {
  return (count > 0) ? static_cast<int32_t>(atomicAdd(&ctrl[kTopkFastCtrlAbove],
                                                    static_cast<uint32_t>(count)))
                     : 0;
}

template <int TOPK>
__global__ void sparse_topk_fast_kernel(
    int32_t* __restrict__ out, float* __restrict__ val, const float* __restrict__ srow,
    const int32_t* __restrict__ row_starts, const int32_t* __restrict__ row_ends,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ token_to_request,
    const int block_table_stride, const int page_size, const bool absolute_indices,
    const bool to_page_slots, void* __restrict__ ws) {
  __shared__ uint32_t lhist[kHistBins];
  __shared__ int32_t emit_idx[TOPK];
  __shared__ float emit_val[TOPK];
  __shared__ int32_t s_emit, s_base, s_bin, s_above, s_gt, s_tie, s_write;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  int64_t start = (row_starts != nullptr) ? static_cast<int64_t>(row_starts[0]) : 0;
  int64_t end = static_cast<int64_t>(row_ends[0]);
  if (start < 0) start = 0;
  if (end < start) end = start;
  const int64_t visible = end - start;
  const int m = static_cast<int>(visible < TOPK ? visible : static_cast<int64_t>(TOPK));
  const int block = static_cast<int>(blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);

  uint32_t* const hist1 = static_cast<uint32_t*>(ws);
  uint32_t* const hist2 = hist1 + kHistBins;
  uint32_t* const ctrl = reinterpret_cast<uint32_t*>(static_cast<char*>(ws) + kTopkFastCtrlOffset);
  TopkFastCand* const cand =
      reinterpret_cast<TopkFastCand*>(static_cast<char*>(ws) + kTopkFastCandOffset);

  if (m == 0) {
    for (int i = tid; i < TOPK; i += block) {
      out[i] = -1;
      if (val != nullptr) val[i] = topk_neg_inf();
    }
    return;
  }

  // ---- 0. clear the workspace (one cooperative sweep, no memset launch) ------
  const int gtid = static_cast<int>(blockIdx.x) * block + tid;
  const int gstep = static_cast<int>(gridDim.x) * block;
  for (int i = gtid; i < kHistBins; i += gstep) {
    hist1[i] = 0u;
    hist2[i] = 0u;
  }
  if (gtid < 2) ctrl[gtid] = 0u;
  grid.sync();

  int64_t lo = 0, hi = 0;
  topk_fast_chunk(start, end, &lo, &hi);

  // ---- 1. chunked top-11-bit histogram --------------------------------------
  for (int i = tid; i < kHistBins; i += block) lhist[i] = 0u;
  __syncthreads();
  topk_hist_pass(srow, lo, hi, 0u, 0u, kRadixShift[0], kRadixWidth[0], lhist);
  __syncthreads();
  for (int i = tid; i < kHistBins; i += block) {
    const uint32_t c = lhist[i];
    if (c != 0u) atomicAdd(&hist1[i], c);
  }
  grid.sync();

  // ---- 2. resolve the bucket holding the K-th score, emit what is above it ---
  topk_pick_desc(hist1, kHistBins, m, &s_bin, &s_above);
  const int32_t b0 = s_bin;
  const int32_t above0 = s_above;
  if (tid == 0) s_emit = 0;
  __syncthreads();
  topk_each(srow, lo, hi, [&](int64_t col, float v) {
    const uint32_t key = topk_key(v);
    const uint32_t d1 = key >> kRadixShift[0];
    if (d1 > static_cast<uint32_t>(b0)) {
      const int32_t slot = atomicAdd(&s_emit, 1);
      if (slot < TOPK) {  // provably < above0 < m, guarded for safety
        emit_idx[slot] = topk_map_column(0, col, block_table, token_to_request, block_table_stride,
                                        page_size, absolute_indices, to_page_slots, start);
        if (val != nullptr) emit_val[slot] = v;
      }
    } else if (d1 == static_cast<uint32_t>(b0)) {
      atomicAdd(&hist2[(key >> kRadixShift[1]) & (kHistBins - 1)], 1u);
    }
  });
  __syncthreads();
  if (tid == 0) s_base = topk_fast_reserve(ctrl, s_emit);
  __syncthreads();
  for (int i = tid; i < s_emit; i += block) {
    out[s_base + i] = emit_idx[i];
    if (val != nullptr) val[s_base + i] = emit_val[i];
  }
  grid.sync();

  // ---- 3. emit above the 22-bit bucket, buffer the bucket itself ------------
  topk_pick_desc(hist2, kHistBins, m - above0, &s_bin, &s_above);
  const int32_t b1 = s_bin;
  const int32_t above1 = s_above;
  if (tid == 0) s_emit = 0;
  __syncthreads();
  topk_each(srow, lo, hi, [&](int64_t col, float v) {
    const uint32_t key = topk_key(v);
    if ((key >> kRadixShift[0]) != static_cast<uint32_t>(b0)) return;
    const uint32_t d2 = (key >> kRadixShift[1]) & (kHistBins - 1);
    if (d2 > static_cast<uint32_t>(b1)) {
      const int32_t slot = atomicAdd(&s_emit, 1);
      if (slot < TOPK) {
        emit_idx[slot] = topk_map_column(0, col, block_table, token_to_request, block_table_stride,
                                        page_size, absolute_indices, to_page_slots, start);
        if (val != nullptr) emit_val[slot] = v;
      }
    } else if (d2 == static_cast<uint32_t>(b1)) {
      const int32_t slot = static_cast<int32_t>(atomicAdd(&ctrl[kTopkFastCtrlCand], 1u));
      if (slot < kTopkFastCand) {
        cand[slot].value = v;
        cand[slot].col = static_cast<int32_t>(col);
      }
    }
  });
  __syncthreads();
  // The reservation counter already carries the slots pass 2 emitted, so the run
  // continues right after them: above0 must not be added a second time.
  if (tid == 0) s_base = topk_fast_reserve(ctrl, s_emit);
  __syncthreads();
  for (int i = tid; i < s_emit; i += block) {
    out[s_base + i] = emit_idx[i];
    if (val != nullptr) val[s_base + i] = emit_val[i];
  }
  grid.sync();

  // ---- 4. exact threshold and tie rule over the candidates (one CTA) --------
  if (blockIdx.x != 0) return;
  const int32_t above = above0 + above1;
  const int32_t need = m - above;
  const int32_t n_cand = static_cast<int32_t>(ctrl[kTopkFastCtrlCand]);
  if (need <= 0 || need > n_cand || n_cand > kTopkFastCand) {
    // Tie-heavy or degenerate row: the single-CTA body reselects it exactly.
    sparse_topk_row_body<TOPK>(out, val, srow, start, end, block_table, token_to_request,
                               block_table_stride, page_size, absolute_indices, to_page_slots, 0,
                               lhist, emit_idx);
    return;
  }

  if (need == n_cand) {
    // Every candidate is selected; the output order is unspecified.
    for (int i = tid; i < n_cand; i += block) {
      const TopkFastCand c = cand[i];
      out[above + i] = topk_map_column(0, c.col, block_table, token_to_request, block_table_stride,
                                      page_size, absolute_indices, to_page_slots, start);
      if (val != nullptr) val[above + i] = c.value;
    }
  } else {
    // Candidates all share the 22-bit prefix (b0, b1): the low 10 key bits pick
    // the exact threshold, then equal keys fall back to the column tie rule.
    constexpr int kLowBins = 1 << kRadixWidth[2];
    const uint32_t prefix = (static_cast<uint32_t>(b0) << kRadixShift[0]) |
                            (static_cast<uint32_t>(b1) << kRadixShift[1]);
    const uint32_t low_mask = static_cast<uint32_t>(kLowBins - 1);
    for (int i = tid; i < kLowBins; i += block) lhist[i] = 0u;
    __syncthreads();
    for (int32_t i = tid; i < n_cand; i += block) {
      atomicAdd(&lhist[topk_key(cand[i].value) & low_mask], 1u);
    }
    __syncthreads();
    topk_pick_desc(lhist, kLowBins, need, &s_bin, &s_above);
    const int32_t d2 = s_bin;
    const int32_t above2 = s_above;
    const uint32_t threshold = prefix | static_cast<uint32_t>(d2);

    // Entries strictly above the threshold (above2 of them, all distinct keys).
    if (tid == 0) s_gt = 0;
    __syncthreads();
    for (int32_t i = tid; i < n_cand; i += block) {
      const TopkFastCand c = cand[i];
      if (topk_key(c.value) <= threshold) continue;
      const int32_t slot = above + atomicAdd(&s_gt, 1);
      if (slot < TOPK) {
        out[slot] = topk_map_column(0, c.col, block_table, token_to_request, block_table_stride,
                                    page_size, absolute_indices, to_page_slots, start);
        if (val != nullptr) val[slot] = c.value;
      }
    }

    // Exact ties at the threshold: take the lowest columns (<= tie_need of them).
    const int32_t tie_need = need - above2;
    if (tid == 0) s_tie = 0;
    __syncthreads();
    for (int32_t i = tid; i < n_cand; i += block) {
      if (topk_key(cand[i].value) == threshold) atomicAdd(&s_tie, 1);
    }
    __syncthreads();
    const int32_t n_tie = s_tie;
    int32_t cutoff = 0x7fffffff;  // take every tie when all of them are needed
    if (tie_need < n_tie) {
      int32_t rem = tie_need;
      uint32_t isel = 0u, imask = 0u;
#pragma unroll
      for (int p = 0; p < kRadixPasses; ++p) {
        const int nbins = 1 << kRadixWidth[p];
        for (int i = tid; i < nbins; i += block) lhist[i] = 0u;
        __syncthreads();
        for (int32_t i = tid; i < n_cand; i += block) {
          if (topk_key(cand[i].value) != threshold) continue;
          const uint32_t u = static_cast<uint32_t>(cand[i].col);
          if ((u & imask) == isel) atomicAdd(&lhist[(u >> kRadixShift[p]) & (nbins - 1)], 1u);
        }
        __syncthreads();
        topk_pick_asc(lhist, nbins, rem, &s_bin, &s_above);
        rem -= s_above;
        isel |= static_cast<uint32_t>(s_bin) << kRadixShift[p];
        imask |= static_cast<uint32_t>(nbins - 1) << kRadixShift[p];
        __syncthreads();
      }
      cutoff = static_cast<int32_t>(isel);
    }
    if (tid == 0) s_write = 0;
    __syncthreads();
    for (int32_t i = tid; i < n_cand; i += block) {
      const TopkFastCand c = cand[i];
      if (topk_key(c.value) != threshold) continue;
      if (c.col > cutoff) continue;
      const int32_t slot = above + above2 + atomicAdd(&s_write, 1);
      if (slot < TOPK) {
        out[slot] = topk_map_column(0, c.col, block_table, token_to_request, block_table_stride,
                                    page_size, absolute_indices, to_page_slots, start);
        if (val != nullptr) val[slot] = c.value;
      }
    }
  }

  for (int i = m + tid; i < TOPK; i += block) {
    out[i] = -1;
    if (val != nullptr) val[i] = topk_neg_inf();
  }
}

/// CTAs per SM a cooperative launch of the fast path can keep resident.
template <int TOPK>
int topk_fast_ctas_per_sm() {
  static const int cached = [] {
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, &sparse_topk_fast_kernel<TOPK>,
                                                      kTopkFastThreads, 0) != cudaSuccess) {
      (void)cudaGetLastError();
      n = 0;
    }
    return (n > 0) ? n : 1;
  }();
  return cached;
}

}  // namespace

// -----------------------------------------------------------------------------
// Host entry
// -----------------------------------------------------------------------------
namespace {

template <typename T>
T* mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

}  // namespace

Status sparse_topk_impl(const Tensor& indices, const Tensor& values, const Tensor& scores,
                        const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                        const Tensor& block_table, const Tensor& token_to_request,
                        int32_t page_size, bool absolute_indices, bool to_page_slots,
                        const Workspace* ws_opt, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sparse_topk: CUDA backend only");
  }
  if (topk != 512 && topk != 1024 && topk != 2048) {
    return detail::set_error(Status::kUnsupportedShape,
                             "sparse_topk: topk %d not in {512, 1024, 2048}", topk);
  }

  Status s = check_tensor(indices, DType::kInt32, 2, 16, "sparse_topk.indices");
  if (s != Status::kSuccess) return s;
  const int64_t rows = indices.shape[0];
  if (rows > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "sparse_topk: %lld rows exceed int32",
                             static_cast<long long>(rows));
  }
  if (indices.shape[1] != topk) {
    return detail::set_error(Status::kInvalidArgument,
                             "sparse_topk: indices [%lld, %lld], expected [rows, %d]",
                             static_cast<long long>(rows),
                             static_cast<long long>(indices.shape[1]), topk);
  }
  if (indices.stride[1] != 1 || indices.stride[0] != topk) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sparse_topk: indices must be packed [rows, topk]");
  }

  const bool has_values = tensor_present(values);
  if (has_values) {
    s = check_tensor(values, DType::kFloat32, 2, 16, "sparse_topk.values");
    if (s != Status::kSuccess) return s;
    if (values.shape[0] != rows || values.shape[1] != topk) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: values [%lld, %lld], expected [%lld, %d]",
                               static_cast<long long>(values.shape[0]),
                               static_cast<long long>(values.shape[1]),
                               static_cast<long long>(rows), topk);
    }
    if (values.stride[1] != 1 || values.stride[0] != topk) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "sparse_topk: values must be packed [rows, topk]");
    }
  }

  s = check_tensor(scores, DType::kFloat32, 2, 16, "sparse_topk.scores");
  if (s != Status::kSuccess) return s;
  if (scores.shape[0] < rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "sparse_topk: scores has %lld rows, indices %lld",
                             static_cast<long long>(scores.shape[0]),
                             static_cast<long long>(rows));
  }
  if (scores.shape[1] < 1) {
    return detail::set_error(Status::kUnsupportedShape, "sparse_topk: max_kv %lld",
                             static_cast<long long>(scores.shape[1]));
  }
  if (scores.stride[1] != 1 || scores.stride[0] < scores.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sparse_topk: scores rows must be contiguous, stride[0] >= max_kv");
  }

  s = check_tensor(row_ends, DType::kInt32, 1, 4, "sparse_topk.row_ends");
  if (s != Status::kSuccess) return s;
  if (row_ends.shape[0] < rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "sparse_topk: row_ends has %lld entries, indices %lld",
                             static_cast<long long>(row_ends.shape[0]),
                             static_cast<long long>(rows));
  }
  if (row_ends.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "sparse_topk: row_ends stride");
  }

  const bool has_starts = tensor_present(row_starts);
  if (has_starts) {
    s = check_tensor(row_starts, DType::kInt32, 1, 4, "sparse_topk.row_starts");
    if (s != Status::kSuccess) return s;
    if (row_starts.shape[0] < rows) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: row_starts has %lld entries, indices %lld",
                               static_cast<long long>(row_starts.shape[0]),
                               static_cast<long long>(rows));
    }
    if (row_starts.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "sparse_topk: row_starts must be unit-stride");
    }
  }

  const bool has_table = tensor_present(block_table);
  if (has_table) {
    s = check_tensor(block_table, DType::kInt32, 2, 4, "sparse_topk.block_table");
    if (s != Status::kSuccess) return s;
    if (block_table.shape[1] < 1) {
      return detail::set_error(Status::kInvalidArgument, "sparse_topk: block_table max_pages %lld",
                               static_cast<long long>(block_table.shape[1]));
    }
    if (block_table.stride[1] != 1 || block_table.stride[0] < block_table.shape[1]) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "sparse_topk: block_table rows must be contiguous");
    }
    if (block_table.stride[0] > INT32_MAX) {
      return detail::set_error(Status::kCapacityExceeded,
                               "sparse_topk: block_table stride exceeds int32");
    }
  }

  const bool has_request = tensor_present(token_to_request);
  if (has_request) {
    s = check_tensor(token_to_request, DType::kInt32, 1, 4, "sparse_topk.token_to_request");
    if (s != Status::kSuccess) return s;
    if (token_to_request.shape[0] < rows) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: token_to_request has %lld entries, indices %lld",
                               static_cast<long long>(token_to_request.shape[0]),
                               static_cast<long long>(rows));
    }
    if (token_to_request.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "sparse_topk: token_to_request must be unit-stride");
    }
  }

  if (page_size < 0) {
    return detail::set_error(Status::kInvalidArgument, "sparse_topk: page_size %d", page_size);
  }
  if (to_page_slots) {
    if (!has_table) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: to_page_slots requires a block_table");
    }
    if (page_size <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: to_page_slots requires page_size > 0");
    }
    if (static_cast<int64_t>(block_table.shape[1]) * page_size < scores.shape[1]) {
      return detail::set_error(Status::kUnsupportedShape,
                               "sparse_topk: block_table covers %lld tokens, max_kv is %lld",
                               static_cast<long long>(block_table.shape[1]) * page_size,
                               static_cast<long long>(scores.shape[1]));
    }
    if (!has_request && block_table.shape[0] < rows) {
      return detail::set_error(Status::kInvalidArgument,
                               "sparse_topk: block_table has %lld requests, indices %lld "
                               "(token_to_request is absent, so rows index requests)",
                               static_cast<long long>(block_table.shape[0]),
                               static_cast<long long>(rows));
    }
  }

  if (rows == 0) return Status::kSuccess;

  const int grid = static_cast<int>(rows);
  const int table_stride = has_table ? static_cast<int>(block_table.stride[0]) : 0;
  const int32_t* starts_ptr = has_starts ? const_ptr<int32_t>(row_starts) : nullptr;
  const int32_t* table_ptr = has_table ? const_ptr<int32_t>(block_table) : nullptr;
  const int32_t* request_ptr = has_request ? const_ptr<int32_t>(token_to_request) : nullptr;
  float* values_ptr = has_values ? mut_ptr<float>(values) : nullptr;
  const int64_t scores_stride = scores.stride[0];
  int32_t* indices_ptr = mut_ptr<int32_t>(indices);
  const float* scores_ptr = const_ptr<float>(scores);
  const int32_t* ends_ptr = const_ptr<int32_t>(row_ends);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);

  // ---- multi-CTA fast path: one long row, caller-supplied workspace ----------
  // A single CTA reaches ~1/36 of this memory system, so a 1M-column row is
  // chunked across a cooperative grid. The workspace is optional: a missing,
  // small or misaligned lease silently keeps the workspace-free path below.
  const bool ws_ready =
      ws_opt != nullptr && ws_opt->device_scratch != nullptr &&
      ws_opt->device_scratch_bytes >= kTopkFastWorkspaceBytes &&
      (reinterpret_cast<uintptr_t>(ws_opt->device_scratch) & 255u) == 0u;
  if (ws_ready && rows == 1 && scores.shape[1] >= kTopkFastMinKv &&
      scores.shape[1] <= INT32_MAX) {
    void* ws_ptr = ws_opt->device_scratch;
    int table_stride_arg = table_stride;
    const int sms = ctx.compute_units > 0 ? ctx.compute_units : 1;
    int64_t by_work = (scores.shape[1] + kTopkFastMinChunk - 1) / kTopkFastMinChunk;
    if (by_work < 1) by_work = 1;
    void* args[] = {&indices_ptr, &values_ptr, &scores_ptr, &starts_ptr, &ends_ptr,
                    &table_ptr,   &request_ptr, &table_stride_arg, &page_size, &absolute_indices,
                    &to_page_slots, &ws_ptr};
    int desired = sms * kTopkFastCtasPerSm;
    if (by_work < desired) desired = static_cast<int>(by_work);
    if (desired < 1) desired = 1;
#define KOPS_LAUNCH_SPARSE_TOPK_FAST(TOPKV)                                                        \
  do {                                                                                             \
    const int cap = topk_fast_ctas_per_sm<TOPKV>() * sms;                                           \
    const int blocks = desired < cap ? desired : cap;                                               \
    const cudaError_t ferr = cudaLaunchCooperativeKernel(                                           \
        reinterpret_cast<void*>(&sparse_topk_fast_kernel<TOPKV>), dim3(blocks),                     \
        dim3(kTopkFastThreads), args, 0, stream);                                                   \
    if (ferr == cudaSuccess) return Status::kSuccess;                                               \
    (void)cudaGetLastError(); /* fall through to the workspace-free path */                          \
  } while (0)
    switch (topk) {
      case 512:
        KOPS_LAUNCH_SPARSE_TOPK_FAST(512);
        break;
      case 1024:
        KOPS_LAUNCH_SPARSE_TOPK_FAST(1024);
        break;
      default:
        KOPS_LAUNCH_SPARSE_TOPK_FAST(2048);
        break;
    }
#undef KOPS_LAUNCH_SPARSE_TOPK_FAST
  }

#define KOPS_LAUNCH_SPARSE_TOPK(TOPKV)                                                          \
  sparse_topk_kernel<TOPKV><<<grid, kTopkThreads, 0, stream>>>(                                 \
      indices_ptr, values_ptr, scores_ptr, scores_stride, starts_ptr, ends_ptr, table_ptr,       \
      request_ptr, table_stride, page_size, absolute_indices, to_page_slots, grid)
  switch (topk) {
    case 512:
      KOPS_LAUNCH_SPARSE_TOPK(512);
      break;
    case 1024:
      KOPS_LAUNCH_SPARSE_TOPK(1024);
      break;
    default:
      KOPS_LAUNCH_SPARSE_TOPK(2048);
      break;
  }
#undef KOPS_LAUNCH_SPARSE_TOPK

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("sparse_topk launch", err);
  return Status::kSuccess;
}

Status sparse_topk(const Tensor& indices, const Tensor& values, const Tensor& scores,
                   const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                   const Tensor& block_table, const Tensor& token_to_request, int32_t page_size,
                   bool absolute_indices, bool to_page_slots, const LaunchContext& ctx) {
  return sparse_topk_impl(indices, values, scores, row_starts, row_ends, topk, block_table,
                          token_to_request, page_size, absolute_indices, to_page_slots, nullptr,
                          ctx);
}

Status sparse_topk_ws(const Tensor& indices, const Tensor& values, const Tensor& scores,
                      const Tensor& row_starts, const Tensor& row_ends, int32_t topk,
                      const Tensor& block_table, const Tensor& token_to_request, int32_t page_size,
                      bool absolute_indices, bool to_page_slots, const Workspace& ws,
                      const LaunchContext& ctx) {
  return sparse_topk_impl(indices, values, scores, row_starts, row_ends, topk, block_table,
                          token_to_request, page_size, absolute_indices, to_page_slots, &ws, ctx);
}

size_t sparse_topk_workspace_bytes() noexcept { return kTopkFastWorkspaceBytes; }

// -----------------------------------------------------------------------------
// Shared mma.sync m16n8k16 helpers for the DSA/V4 sparse-attention bodies below.
//
// The indexer, sparse MLA and compressed-attention bodies all score 16 rows
// (indexer heads or query heads) against 16-key latent tiles on tensor cores, so
// they share one tile convention:
//   * a 64-element-swizzled smem row (TkLay) keeps ldmatrix conflict-free and
//     the cp.async tile writes coalesced - the same swizzle as attention/mla.cu,
//   * m16n8k16 with fp32 accumulate in the query dtype (bf16 / fp16); fp8
//     indexer keys are dequantized into the bf16 tile on the way into smem,
//   * four warps per CTA, one warp-level value slice each (128 of 512 latent
//     dims) so the output tile needs no cross-warp reduction.
// Only this block touches ldmatrix / mma / cp.async PTX.
// -----------------------------------------------------------------------------
namespace {

constexpr int tkWarps = 4;
constexpr int tkThreads = tkWarps * 32;
constexpr int tkKeys = 16;  // keys per mma tile

constexpr float kTkLog2e = 1.4426950408889634f;
constexpr float kTkLn2 = 0.6931471805599453f;

__device__ __forceinline__ float tk_ninf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ uint32_t tk_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// 16-byte global -> shared copy; `pred == false` writes 16 zero bytes and reads
/// nothing (the documented padding value for a masked key tile).
__device__ __forceinline__ void tk_cp16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#else
  (void)dst;
  (void)src;
  (void)pred;
#endif
}

__device__ __forceinline__ void tk_cp_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}

template <int N>
__device__ __forceinline__ void tk_cp_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ void tk_ldsm4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#else
  (void)r;
  (void)addr;
#endif
}

__device__ __forceinline__ void tk_ldsm4t(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#else
  (void)r;
  (void)addr;
#endif
}

/// D += A * B for one m16n8k16 tile (A row-major, B col-major, fp32 accumulate).
template <typename T>
__device__ __forceinline__ void tk_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                       uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#else
  (void)d;
  (void)a;
  (void)b0;
  (void)b1;
#endif
}

template <typename T>
__device__ __forceinline__ uint32_t tk_pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  } else {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
}

/// Four consecutive e4m3 codes -> four floats (exact: OCP e4m3 is a subset of fp16).
__device__ __forceinline__ void tk_fp8x4(uint32_t w, float* out) {
  uint32_t a = __byte_perm(w, 0u, 0x1404u);
  uint32_t b = __byte_perm(w, 0u, 0x3424u);
  a = (a & 0x80008000u) | ((a >> 1) & 0x3F803F80u);
  b = (b & 0x80008000u) | ((b >> 1) & 0x3F803F80u);
  const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
  const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
  out[0] = fa.x;
  out[1] = fa.y;
  out[2] = fb.x;
  out[3] = fb.y;
}

/// Shared rows of D 16-bit elements with the decode.cu/mla.cu swizzle (D % 64 == 0).
template <int D>
struct TkLay {
  static_assert(D % 64 == 0, "swizzled tile");
  static __device__ __forceinline__ int off(int row, int chunk) {
    return row * D + ((chunk ^ (row & 7)) << 3);
  }
};

/// Byte base of one latent record. Mirrors the record addressing used by the
/// scalar readers (sp_latent / sp_row_*) so codecs and page tables agree.
template <typename DTypeKV>
__device__ __forceinline__ const char* sp_row_base(const kv::PagedKVDesc& d, int64_t slot) {
  const int32_t page_size = d.page_size > 0 ? d.page_size : 1;
  // Physical slots index a token cache, so 32-bit page arithmetic is exact and
  // avoids a 64-bit integer division per key.
  const int32_t slot32 = static_cast<int32_t>(slot);
  const int32_t page = slot32 / page_size;
  const int off = slot32 - page * page_size;
  const bool byte_strides = d.stride_unit == StrideUnit::kByte || d.entry_bytes != 0;
  const int64_t unit = byte_strides ? 1 : static_cast<int64_t>(sizeof(DTypeKV));
  const int64_t page_stride =
      (d.k_stride_page > 0 ? d.k_stride_page : static_cast<int64_t>(page_size) * d.entry_bytes) *
      unit;
  const int64_t token_stride =
      (d.k_stride_token > 0 ? d.k_stride_token : d.entry_bytes) * unit;
  return static_cast<const char*>(d.k_data) + page * page_stride + off * token_stride;
}

__device__ __forceinline__ float sp_neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float sp_e4m3(uint8_t v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}
__device__ __forceinline__ float sp_e2m1(uint8_t code) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = mag[code & 7u];
  return (code & 8u) ? -m : m;
}
__device__ __forceinline__ float sp_ue8m0(uint8_t v) {
  return ldexpf(1.0f, static_cast<int>(v) - 127);
}
template <typename T>
__device__ __forceinline__ float sp_to_f(T v) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
  else if constexpr (std::is_same_v<T, __half>) return __half2float(v);
  else return static_cast<float>(v);
}
template <typename T>
__device__ __forceinline__ T sp_from_f(float v) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16_rn(v);
  else if constexpr (std::is_same_v<T, __half>) return __float2half_rn(v);
  else return static_cast<T>(v);
}

/// Element d of the latent record at physical token slot `slot`. The record is
/// read through the scope descriptor so 656/584/528/288 share one path.
template <typename DTypeKV>
__device__ __forceinline__ float sp_latent(const kv::PagedKVDesc& d, int64_t slot, int d_idx,
                                           int qk_dim) {
  (void)qk_dim;
  const char* base = sp_row_base<DTypeKV>(d, slot);
  switch (d.k_codec) {
    case kv::KVCodec::kNative: {
      const DTypeKV* row = reinterpret_cast<const DTypeKV*>(base);
      return sp_to_f<DTypeKV>(row[d_idx]);
    }
    case kv::KVCodec::kMLAFp8DS: {
      if (d_idx < 512) {
        const float* scales = reinterpret_cast<const float*>(base + 512);
        return sp_e4m3(static_cast<uint8_t>(base[d_idx])) * scales[d_idx / 128];
      }
      const __nv_bfloat16* rope = reinterpret_cast<const __nv_bfloat16*>(base + 528);
      return __bfloat162float(rope[d_idx - 512]);
    }
    case kv::KVCodec::kMLAFp8DSV40: {
      const int f = kv::mla_kv_format(d.k_codec).nope_dim;
      if (d_idx < f) {
        const uint8_t* scales = static_cast<const uint8_t*>(d.k_scale_pages) +
                                (slot / (d.page_size > 0 ? d.page_size : 1)) *
                                    (d.k_scale_stride_page > 0
                                         ? d.k_scale_stride_page
                                         : static_cast<int64_t>(d.page_size > 0 ? d.page_size : 1) * 8) +
                                (slot % (d.page_size > 0 ? d.page_size : 1)) * 8;
        return sp_e4m3(static_cast<uint8_t>(base[d_idx])) * sp_ue8m0(scales[d_idx / 64]);
      }
      const __nv_bfloat16* rope = reinterpret_cast<const __nv_bfloat16*>(base + f);
      return __bfloat162float(rope[d_idx - f]);
    }
    case kv::KVCodec::kMLAFp8DSV41: {
      const uint8_t* scales = reinterpret_cast<const uint8_t*>(base + 512);
      return sp_e4m3(static_cast<uint8_t>(base[d_idx])) * sp_ue8m0(scales[d_idx / 32]);
    }
    case kv::KVCodec::kMLAFp8DSV41Fp4: {
      const uint8_t packed = static_cast<uint8_t>(base[d_idx >> 1]);
      const uint8_t code = (d_idx & 1) ? static_cast<uint8_t>(packed >> 4) : (packed & 0x0f);
      const uint8_t* scales = reinterpret_cast<const uint8_t*>(base + 256);
      return sp_e2m1(code) * sp_e4m3(scales[d_idx / 16]);
    }
    default:
      return 0.0f;
  }
}

/// Eight consecutive latent dims (dim0 % 8 == 0) converted into the compute dtype.
/// Used by the tensor-core tile loaders for every non-native codec; the native
/// path is a direct 16-byte copy and does not come through here.
template <typename T, typename DTypeKV>
__device__ __forceinline__ void sp_read8_base(const kv::PagedKVDesc& d, const char* base,
                                              int64_t slot, int dim0, T* out) {
  switch (d.k_codec) {
    case kv::KVCodec::kNative: {
      const DTypeKV* row = reinterpret_cast<const DTypeKV*>(base);
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i] = sp_from_f<T>(sp_to_f<DTypeKV>(row[dim0 + i]));
      return;
    }
    case kv::KVCodec::kMLAFp8DS: {
      if (dim0 < 512) {
        const float s = reinterpret_cast<const float*>(base + 512)[dim0 >> 7];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          out[i] = sp_from_f<T>(sp_e4m3(static_cast<uint8_t>(base[dim0 + i])) * s);
        }
      } else {
        const __nv_bfloat16* rope = reinterpret_cast<const __nv_bfloat16*>(base + 528);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          out[i] = sp_from_f<T>(__bfloat162float(rope[dim0 - 512 + i]));
        }
      }
      return;
    }
    case kv::KVCodec::kMLAFp8DSV40: {
      const int f = kv::mla_kv_format(d.k_codec).nope_dim;
      if (dim0 < f) {
        const int32_t ps = d.page_size > 0 ? d.page_size : 1;
        const int32_t s32 = static_cast<int32_t>(slot);
        const uint8_t* scales =
            static_cast<const uint8_t*>(d.k_scale_pages) +
            (s32 / ps) * (d.k_scale_stride_page > 0 ? d.k_scale_stride_page
                                                    : static_cast<int64_t>(ps) * 8) +
            (s32 - (s32 / ps) * ps) * 8;
        const float s = sp_ue8m0(scales[dim0 / 64]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          out[i] = sp_from_f<T>(sp_e4m3(static_cast<uint8_t>(base[dim0 + i])) * s);
        }
      } else {
        const __nv_bfloat16* rope = reinterpret_cast<const __nv_bfloat16*>(base + f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          out[i] = sp_from_f<T>(__bfloat162float(rope[dim0 - f + i]));
        }
      }
      return;
    }
    case kv::KVCodec::kMLAFp8DSV41: {
      const uint8_t* scales = reinterpret_cast<const uint8_t*>(base + 512);
      const float s = sp_ue8m0(scales[dim0 / 32]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        out[i] = sp_from_f<T>(sp_e4m3(static_cast<uint8_t>(base[dim0 + i])) * s);
      }
      return;
    }
    case kv::KVCodec::kMLAFp8DSV41Fp4: {
      const uint8_t* scales = reinterpret_cast<const uint8_t*>(base + 256);
      const float s = sp_e4m3(scales[dim0 / 16]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int di = dim0 + i;
        const uint8_t packed = static_cast<uint8_t>(base[di >> 1]);
        const uint8_t code = (di & 1) ? static_cast<uint8_t>(packed >> 4) : (packed & 0x0f);
        out[i] = sp_from_f<T>(sp_e2m1(code) * s);
      }
      return;
    }
    default:
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i] = sp_from_f<T>(0.0f);
      return;
  }
}

template <typename T, typename DTypeKV>
__device__ __forceinline__ void sp_read8(const kv::PagedKVDesc& d, int64_t slot, int dim0,
                                         T* out) {
  sp_read8_base<T, DTypeKV>(d, sp_row_base<DTypeKV>(d, slot), slot, dim0, out);
}

__device__ __forceinline__ float sp_block_max(float v, float* red, int block) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = red[lane];
  if (warp == 0) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  (void)block;
  return red[0];
}

__device__ __forceinline__ float sp_block_sum(float v, float* red, int block) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = (warp == 0 && lane < 32) ? red[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  (void)block;
  return red[0];
}

}  // namespace

// -----------------------------------------------------------------------------
// A11 lightning indexer (T1, tensor cores): I[t,s] = sum_h w[t,h] * ReLU(q.k) * scales
//
// One CTA = (query row, a contiguous run of 64-key tiles), 4 warps. Warp w owns
// index heads [16w, 16w+16) as the mma M dimension and walks the tile as four
// 16-key n-tiles, so the Q fragments stay in registers (one ldmatrix.x4 per 16
// dims, before the tile loop) and the K tile is shared by all four warps through
// smem. Each warp applies the per-head weight and ReLU to its 16 mma rows and
// folds them into one partial per key with a shuffle butterfly over the eight
// row groups; the four warp partials are summed once in smem into the fp32
// logits row, with keys outside [row_starts, row_ends) written as -inf.
// Q/K are loaded with cp.async as bf16/fp16; FP8 K (kIndexerFP8 split page) and
// FP8 Q are dequantized into the tile on load (per-group / per-head scales).
// -----------------------------------------------------------------------------
namespace {

/// Data and scale pointers of logical key `s` of one request. For kPaged the
/// indexer cache is a split page [page_size x head_dim data | page_size x groups
/// fp32 scales]; for ragged prefill the row is k_data + s * k_stride_n and the
/// scales follow the head_dim data elements.
template <typename DTypeK, bool kPaged>
__device__ __forceinline__ void idx_row_of(const IndexerParams& p, int req, int64_t s, int groups,
                                           const char** data, const char** scales) {
  if (scales != nullptr) *scales = nullptr;
  if constexpr (!kPaged) {
    const int64_t stride = p.k_stride_n > 0 ? p.k_stride_n : p.index_dim;
    const DTypeK* row = reinterpret_cast<const DTypeK*>(p.k_cache.k_data) + s * stride;
    *data = reinterpret_cast<const char*>(row);
    if constexpr (std::is_same_v<DTypeK, uint8_t>) {
      if (groups > 0) *scales = *data + p.index_dim;
    }
  } else {
    const int32_t page_size = p.k_cache.page_size;
    const int64_t logical_page = s / page_size;
    const int off = static_cast<int>(s - logical_page * page_size);
    int32_t page = -1;
    if (p.k_cache.table_format == kv::PageTableFormat::kBlockTable) {
      page = p.k_cache.block_table[static_cast<int64_t>(req) * p.k_cache.block_table_stride +
                                   logical_page];
    } else if (p.k_cache.table_format == kv::PageTableFormat::kCSR) {
      const int32_t begin = p.k_cache.kv_indptr[req];
      page = p.k_cache.kv_page_indices[begin + logical_page];
    } else {
      page = static_cast<int32_t>(logical_page);
    }
    if (page < 0) {
      *data = nullptr;
      return;
    }
    const int64_t page_bytes =
        p.k_cache.k_stride_page > 0
            ? p.k_cache.k_stride_page
            : static_cast<int64_t>(page_size) * (p.index_dim + groups * 4);
    const char* base = static_cast<const char*>(p.k_cache.k_data) + page * page_bytes;
    *data = base + static_cast<int64_t>(off) * p.index_dim;
    if constexpr (std::is_same_v<DTypeK, uint8_t>) {
      if (groups > 0) {
        *scales = base + static_cast<int64_t>(page_size) * p.index_dim +
                  static_cast<int64_t>(off) * groups * 4;
      }
    }
  }
}

/// Scale group of element `d` (index_dim / groups elements per group).
__device__ __forceinline__ int idx_group(int d, int index_dim, int groups) {
  const int per = groups > 0 ? index_dim / groups : index_dim;
  return per > 0 ? d / per : 0;
}

}  // namespace

template <typename DTypeQ, typename DTypeK, int INDEX_HEADS, int INDEX_DIM, bool kPaged,
          typename ArchTag>
__global__ void __launch_bounds__(tkThreads)
    lightning_indexer_kernel(const __grid_constant__ IndexerParams params) {
  static_assert(INDEX_HEADS % tkWarps == 0, "index heads split across four warps");
  static_assert(INDEX_DIM % 64 == 0, "64-element swizzled tile");
  constexpr int kTile = 64;  // keys per tile
  constexpr int kHeadsPerWarp = INDEX_HEADS / tkWarps;
  constexpr int kChunks = INDEX_DIM / 8;  // 16-byte chunks per key row
  constexpr int kTileChunks = kTile * kChunks;
  constexpr int kPerThread = kTileChunks / tkThreads;
  constexpr int kSteps = INDEX_DIM / 16;
  constexpr int kNTiles = kTile / tkKeys;
  static_assert(kTileChunks % tkThreads == 0, "tile load");
  static_assert(kPerThread * tkThreads == kTileChunks, "tile load");
  using T = std::conditional_t<std::is_same_v<DTypeQ, __half>, __half, __nv_bfloat16>;
  using L = TkLay<INDEX_DIM>;
  (void)sizeof(ArchTag);

  const int row = static_cast<int>(blockIdx.x);
  if (row >= params.num_tokens) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int h0 = warp * kHeadsPerWarp;
  const int start = params.row_starts != nullptr ? params.row_starts[row] : 0;
  const int end = params.row_ends[row];
  const int tiles = (params.max_kv + kTile - 1) / kTile;

  extern __shared__ __align__(16) unsigned char idx_smem[];
  T* sQ = reinterpret_cast<T*>(idx_smem);                    // [INDEX_HEADS][INDEX_DIM]
  T* sK = sQ + INDEX_HEADS * INDEX_DIM;                      // [kTile][INDEX_DIM]
  float* sPart = reinterpret_cast<float*>(sK + kTile * INDEX_DIM);  // [tkWarps][kTile]

  // ---- Q tile: all index heads of this row, shared by the whole CTA.
  const DTypeQ* qbase = reinterpret_cast<const DTypeQ*>(params.q) +
                        static_cast<int64_t>(row) * params.q_stride_n;
  if constexpr (std::is_same_v<DTypeQ, uint8_t>) {
    constexpr int kQPairs = INDEX_DIM / 16;
    for (int i = tid; i < INDEX_HEADS * kQPairs; i += tkThreads) {
      const int h = i / kQPairs, c = i - h * kQPairs;
      const float s = params.q_scale != nullptr ? params.q_scale[row * INDEX_HEADS + h] : 1.0f;
      const uint4 w = *reinterpret_cast<const uint4*>(qbase + static_cast<int64_t>(h) * params.q_stride_h +
                                                      c * 16);
      float f[16];
      tk_fp8x4(w.x, f + 0);
      tk_fp8x4(w.y, f + 4);
      tk_fp8x4(w.z, f + 8);
      tk_fp8x4(w.w, f + 12);
#pragma unroll
      for (int j = 0; j < 16; ++j) f[j] *= s;
      const uint4 lo = make_uint4(tk_pack2<T>(f[0], f[1]), tk_pack2<T>(f[2], f[3]),
                                  tk_pack2<T>(f[4], f[5]), tk_pack2<T>(f[6], f[7]));
      const uint4 hi = make_uint4(tk_pack2<T>(f[8], f[9]), tk_pack2<T>(f[10], f[11]),
                                  tk_pack2<T>(f[12], f[13]), tk_pack2<T>(f[14], f[15]));
      *reinterpret_cast<uint4*>(sQ + L::off(h, 2 * c)) = lo;
      *reinterpret_cast<uint4*>(sQ + L::off(h, 2 * c + 1)) = hi;
    }
  } else {
    constexpr int kQPerThread = INDEX_HEADS * kChunks / tkThreads;
    static_assert(INDEX_HEADS * kChunks % tkThreads == 0, "Q load");
#pragma unroll
    for (int it = 0; it < kQPerThread; ++it) {
      const int i = it * tkThreads + tid;
      const int h = i / kChunks, c = i - h * kChunks;
      tk_cp16(tk_u32(sQ + L::off(h, c)),
              qbase + static_cast<int64_t>(h) * params.q_stride_h + c * 8, true);
    }
    tk_cp_commit();
    tk_cp_wait<0>();
  }
  __syncthreads();

  // A fragments of this warp's 16 index heads, kept in registers for every tile.
  const uint32_t sQ_u = tk_u32(sQ);
  uint32_t af[kSteps][4];
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    tk_ldsm4(af[s], sQ_u + L::off(h0 + (lane & 15), 2 * s + (lane >> 4)) * 2);
  }

  const float w_lo = params.head_weights != nullptr
                         ? params.head_weights[static_cast<int64_t>(row) * INDEX_HEADS + h0 +
                                               (lane >> 2)]
                         : 1.0f;
  const float w_hi = params.head_weights != nullptr
                         ? params.head_weights[static_cast<int64_t>(row) * INDEX_HEADS + h0 +
                                               (lane >> 2) + 8]
                         : 1.0f;
  const int groups = params.k_cache.k_codec == kv::KVCodec::kIndexerFP8
                         ? static_cast<int>(params.k_cache.entry_bytes > 0
                                                ? (params.k_cache.entry_bytes - params.index_dim) / 4
                                                : 0)
                         : 0;
  const int req = params.token_to_request != nullptr ? params.token_to_request[row] : row;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_cs = (lane >> 3) & 1;

  const int64_t per_cta = (static_cast<int64_t>(tiles) + gridDim.y - 1) / gridDim.y;
  const int64_t t_begin = static_cast<int64_t>(blockIdx.y) * per_cta;
  const int64_t t_end = min(static_cast<int64_t>(tiles), t_begin + per_cta);

  for (int64_t t = t_begin; t < t_end; ++t) {
    const int64_t key0 = t * kTile;
    // ---- K tile. Keys outside [start, end) or beyond max_kv become zero rows and
    // are overwritten with the documented -inf padding in the epilogue.
    if constexpr (std::is_same_v<DTypeK, uint8_t>) {
#pragma unroll
      for (int it = 0; it < kPerThread; ++it) {
        const int i = it * tkThreads + tid;
        const int r = i / kChunks, c = i - r * kChunks;
        const int64_t key = key0 + r;
        const bool ok = key < params.max_kv && key >= start && key < end;
        uint4 packed = make_uint4(0u, 0u, 0u, 0u);
        float f[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) f[j] = 0.0f;
        if (ok) {
          const char* data = nullptr;
          const char* scales = nullptr;
          idx_row_of<DTypeK, kPaged>(params, req, key, groups, &data, &scales);
          if (data != nullptr) {
            packed = *reinterpret_cast<const uint4*>(data + c * 16);
            tk_fp8x4(packed.x, f + 0);
            tk_fp8x4(packed.y, f + 4);
            tk_fp8x4(packed.z, f + 8);
            tk_fp8x4(packed.w, f + 12);
            if (scales != nullptr) {
              const float* sc = reinterpret_cast<const float*>(scales);
#pragma unroll
              for (int j = 0; j < 16; ++j) {
                f[j] *= sc[idx_group(c * 16 + j, INDEX_DIM, groups)];
              }
            }
          }
        }
        const uint4 lo = make_uint4(tk_pack2<T>(f[0], f[1]), tk_pack2<T>(f[2], f[3]),
                                    tk_pack2<T>(f[4], f[5]), tk_pack2<T>(f[6], f[7]));
        const uint4 hi = make_uint4(tk_pack2<T>(f[8], f[9]), tk_pack2<T>(f[10], f[11]),
                                    tk_pack2<T>(f[12], f[13]), tk_pack2<T>(f[14], f[15]));
        *reinterpret_cast<uint4*>(sK + L::off(r, 2 * c)) = lo;
        *reinterpret_cast<uint4*>(sK + L::off(r, 2 * c + 1)) = hi;
      }
    } else {
#pragma unroll
      for (int it = 0; it < kPerThread; ++it) {
        const int i = it * tkThreads + tid;
        const int r = i / kChunks, c = i - r * kChunks;
        const int64_t key = key0 + r;
        const bool ok = key < params.max_kv && key >= start && key < end;
        const char* data = reinterpret_cast<const char*>(params.k_cache.k_data);
        if (ok) {
          idx_row_of<DTypeK, kPaged>(params, req, key, groups, &data, nullptr);
        }
        tk_cp16(tk_u32(sK + L::off(r, c)), data + c * 16, ok && data != nullptr);
      }
      tk_cp_commit();
      tk_cp_wait<0>();
    }
    __syncthreads();

    // ---- 16 heads x 64 keys, four 16-key n-tiles.
#pragma unroll
    for (int np = 0; np < kNTiles; ++np) {
      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] = 0.0f;
      }
      const uint32_t sK_u = tk_u32(sK);
#pragma unroll
      for (int step = 0; step < kSteps; ++step) {
        uint32_t bb[4];
        tk_ldsm4(bb, sK_u + L::off(np * tkKeys + b_row, 2 * step + b_cs) * 2);
        tk_mma<T>(s[0], af[step], bb[0], bb[1]);
        tk_mma<T>(s[1], af[step], bb[2], bb[3]);
      }
      // ReLU + per-head weight, then fold the sixteen mma rows (heads) into one
      // partial per key: rows are (lane>>2) and (lane>>2)+8, sixteen heads are
      // eight lane groups apart by four, so a xor butterfly over 4/8/16 sums them.
      float v0 = s[0][0], v1 = s[0][1], v2 = s[0][2], v3 = s[0][3];
      float u0 = s[1][0], u1 = s[1][1], u2 = s[1][2], u3 = s[1][3];
      if (params.apply_relu) {
        v0 = fmaxf(v0, 0.0f);
        v1 = fmaxf(v1, 0.0f);
        v2 = fmaxf(v2, 0.0f);
        v3 = fmaxf(v3, 0.0f);
        u0 = fmaxf(u0, 0.0f);
        u1 = fmaxf(u1, 0.0f);
        u2 = fmaxf(u2, 0.0f);
        u3 = fmaxf(u3, 0.0f);
      }
      float lo0 = __fmaf_rn(w_lo, v0, w_hi * v2);
      float lo1 = __fmaf_rn(w_lo, v1, w_hi * v3);
      float hi0 = __fmaf_rn(w_lo, u0, w_hi * u2);
      float hi1 = __fmaf_rn(w_lo, u1, w_hi * u3);
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) {
        lo0 += __shfl_xor_sync(0xffffffffu, lo0, o);
        lo1 += __shfl_xor_sync(0xffffffffu, lo1, o);
        hi0 += __shfl_xor_sync(0xffffffffu, hi0, o);
        hi1 += __shfl_xor_sync(0xffffffffu, hi1, o);
      }
      if (lane < 4) {
        float* dst = sPart + warp * kTile + np * tkKeys + 2 * lane;
        dst[0] = lo0;
        dst[1] = lo1;
        dst[8] = hi0;
        dst[9] = hi1;
      }
    }
    __syncthreads();

    // ---- Four warp partials -> one fp32 logit per key, -inf outside the row.
    if (tid < kTile) {
      const int64_t key = key0 + tid;
      if (key < params.max_kv) {
        float acc = tk_ninf();
        if (key >= start && key < end) {
          acc = (sPart[tid] + sPart[kTile + tid]) + (sPart[2 * kTile + tid] + sPart[3 * kTile + tid]);
        }
        params.logits[static_cast<int64_t>(row) * params.logits_stride + key] = acc;
      }
    }
    __syncthreads();
  }
}

namespace {

Status launch_indexer(const IndexerParams& p, DType q_dtype, DType k_dtype, bool paged,
                      cudaStream_t stream) {
  const int64_t tiles = (p.max_kv + 63) / 64;
  if (p.num_tokens > 65535) {
    return detail::set_error(Status::kCapacityExceeded, "lightning_indexer: grid too large");
  }
  if (p.num_index_heads != 64 || p.index_dim != 128) {
    return detail::set_error(Status::kUnsupportedShape,
                             "lightning_indexer: expects 64 heads x 128 dims");
  }
  const size_t smem = (64 * 128 + 64 * 128) * sizeof(__nv_bfloat16) + (4 * 64) * sizeof(float);
  const int gy = static_cast<int>(tiles < 1 ? 1 : (tiles > 1024 ? 1024 : tiles));
  const dim3 grid(static_cast<unsigned>(p.num_tokens), static_cast<unsigned>(gy));
  const dim3 block(tkThreads);
  if (smem > 48u * 1024u) {
    return detail::set_error(Status::kInternal, "lightning_indexer: smem budget");
  }
#define KOPS_LAUNCH_IDX(QT, KT, PAGED)                                                       \
  do {                                                                                       \
    lightning_indexer_kernel<QT, KT, 64, 128, PAGED, Sm80><<<grid, block, smem, stream>>>(p); \
    const cudaError_t err = cudaGetLastError();                                              \
    if (err != cudaSuccess) return launch_error("lightning_indexer launch", err);            \
    return Status::kSuccess;                                                                 \
  } while (0)
  if (paged) {
    if (q_dtype == DType::kBFloat16 && k_dtype == DType::kBFloat16) KOPS_LAUNCH_IDX(__nv_bfloat16, __nv_bfloat16, true);
    if (q_dtype == DType::kFloat16 && k_dtype == DType::kFloat16) KOPS_LAUNCH_IDX(__half, __half, true);
    if (q_dtype == DType::kBFloat16 && k_dtype == DType::kFloat8E4M3) KOPS_LAUNCH_IDX(__nv_bfloat16, uint8_t, true);
    if (q_dtype == DType::kFloat8E4M3 && k_dtype == DType::kFloat8E4M3) KOPS_LAUNCH_IDX(uint8_t, uint8_t, true);
  } else {
    if (q_dtype == DType::kBFloat16 && k_dtype == DType::kBFloat16) KOPS_LAUNCH_IDX(__nv_bfloat16, __nv_bfloat16, false);
    if (q_dtype == DType::kFloat16 && k_dtype == DType::kFloat16) KOPS_LAUNCH_IDX(__half, __half, false);
    if (q_dtype == DType::kBFloat16 && k_dtype == DType::kFloat8E4M3) KOPS_LAUNCH_IDX(__nv_bfloat16, uint8_t, false);
    if (q_dtype == DType::kFloat8E4M3 && k_dtype == DType::kFloat8E4M3) KOPS_LAUNCH_IDX(uint8_t, uint8_t, false);
  }
#undef KOPS_LAUNCH_IDX
  return detail::set_error(Status::kUnsupportedDtype, "lightning_indexer: dtype pair");
}

}  // namespace

Status lightning_indexer(const IndexerParams& params, DType q_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "lightning_indexer: CUDA backend only");
  }
  if (params.q == nullptr || params.logits == nullptr || params.head_weights == nullptr ||
      params.row_ends == nullptr || params.num_tokens < 0 || params.max_kv < 0 ||
      params.num_index_heads <= 0 || params.index_dim <= 0) {
    return detail::set_error(Status::kInvalidArgument, "lightning_indexer: bad argument");
  }
  if (params.q_stride_n < params.index_dim || params.q_stride_h < params.index_dim ||
      params.logits_stride < params.max_kv) {
    return detail::set_error(Status::kUnsupportedLayout, "lightning_indexer: strides");
  }
  const bool paged = params.k_cache.table_format != kv::PageTableFormat::kSlotMapping &&
                     params.k_cache.k_data != nullptr && params.k_cache.page_size > 0 &&
                     params.k_cache.head_dim_k == params.index_dim;
  const DType k_dtype = params.k_cache.k_dtype;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  return launch_indexer(params, q_dtype, k_dtype, paged, stream);
}

// -----------------------------------------------------------------------------
// A13 sparse MLA (T1, tensor cores): one CTA per (query row, 16 query heads).
//
// The 576-element latent record (512 NoPE + 64 RoPE, or the 512-element V4.1
// record) is shared by all 16 heads of the CTA, so the kernel loads each key
// tile once and reuses it for the QK scores of every head and for the latent
// output slice:
//   * QK  = Q[16 heads][qk_dim] x K^T: the 16 keys are two n8 tiles and the
//     qk_dim steps are split four ways across the warps (warp w owns steps
//     w, w+4, ...), so all four SMSPs stay busy. The four partials are summed
//     in fp32 through smem, which gives the softmax its 16x16 score tile once
//     (not once per warp).
//   * PV  = P[16 heads][16 keys] x V[16 keys][512]: warp w owns dims
//     [128w, 128w+128) as 16 n8 mma groups, so the latent output tile needs no
//     cross-warp reduction. P is repacked from the summed scores as the mma A
//     fragment.
//   * K tiles are double-buffered with cp.async (one key tile in flight
//     while the previous one is consumed); native bf16/fp16 records take the
//     cp.async path, the 656/584/528/288 byte codecs are dequantized into the
//     bf16 tile on load. The 16 keys of the next tile are resolved (page table
//     + record addressing, once per key rather than once per chunk) and
//     published to smem one iteration ahead, so a tile costs one barrier for
//     the pipeline plus one for the cross-warp score reduction.
// Online softmax runs in the log2 domain (fp32), covers both scopes plus the
// attention sink, and writes natural-log LSE. qk_dim / value_dim outside the
// tensor-core shapes (and fp32 q) fall back to the scalar reader.
// -----------------------------------------------------------------------------
namespace {

constexpr int spTile = 16;                   // keys per softmax tile (one m16n8k16 PV step)
constexpr int spRedWords = (spTile / 8) * 4 * tkWarps * 32;  // [pair][warp][lane]

/// Query/output/lse/sink view shared by the sparse-MLA and compressed-attention
/// tensor-core bodies.
struct SpAttnIo {
  const void* q;
  int64_t q_stride_n, q_stride_h;
  void* o;
  int64_t o_stride_n, o_stride_h;
  float* lse;
  const float* sink;
  int num_heads;
  float scale;
  float q_scale;
  int v_dim;
};

/// One physical key of the visible list: which latent cache and which token slot.
struct SpKeyRef {
  const kv::PagedKVDesc* kv;
  int64_t slot;
};

/// One visible key resolved to its latent record: the page table and record
/// addressing are evaluated once per key, not once per 16-byte chunk.
struct SpRow {
  const kv::PagedKVDesc* kv;  // null when the key is masked
  const char* base;           // byte base of the record, null when masked
  int64_t slot;               // physical token slot (codec scale pages)
  int32_t code;               // encoded key, < 0 when masked
  int32_t native;             // record is a plain dtype array
};

/// Tensor-core attention over the visible key list `keys`. `smem` must hold
///   2 * TILE * QK_DIM * sizeof(T)  double-buffered latent tile (Q staging aliases it)
/// + (TILE / 8) * 4 * tkWarps * 32 * 4  QK warp partials
/// + 2 * TILE * sizeof(SpRow)       per-tile resolved key rows
/// + 8                                small scalars
/// and blockDim.x must be tkThreads. `keys.prepare(base, count, rows)` resolves
/// up to TILE keys; sparse MLA uses TILE = spTile (16), compressed attention a
/// wider tile so one shared K/V staging pass feeds all 16 query heads.
template <int QK_DIM, int VALUE_DIM, int TILE, typename T, typename DTypeKV, typename DTypeO,
          typename Keys>
__device__ __forceinline__ void sp_tc_attend(const SpAttnIo& io, const Keys& keys, int row,
                                             unsigned char* smem) {
  constexpr int kNT = TILE / 8;    // n8 score tiles of one key tile
  constexpr int kPairs = kNT * 4;    // score values per lane
  constexpr int kHalves = TILE / 16;  // 16-key steps of the PV accumulation
  constexpr int kChunks = QK_DIM / 8;  // 16-byte chunks per latent row
  constexpr int kPerThread = TILE * kChunks / tkThreads;
  constexpr int kSteps = QK_DIM / 16;  // mma k steps of one QK tile
  constexpr int kStepsPerWarp = kSteps / tkWarps;
  constexpr int kTileElems = TILE * QK_DIM;
  constexpr int kRedWords = kPairs * tkWarps * 32;
  static_assert(TILE % 16 == 0, "two 16-key PV steps");
  static_assert(TILE * kChunks % tkThreads == 0, "latent tile load");
  static_assert(kSteps % tkWarps == 0, "QK k split");
  using L = TkLay<QK_DIM>;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int h0 = static_cast<int>(blockIdx.y) * tkKeys;
  const int H = io.num_heads;

  T* sK = reinterpret_cast<T*>(smem);                             // [2][TILE][QK_DIM]
  float* sRed = reinterpret_cast<float*>(sK + 2 * kTileElems);     // [kPairs][tkWarps][32]
  SpRow* sRow = reinterpret_cast<SpRow*>(sRed + kRedWords);       // [2][TILE]
  T* sQ = sK;                                                     // Q staging (aliases buffer 0)

  // ---- Q tile of this 16-head block, staged where K buffer 0 will live.
  const bool q_scale_on = io.q_scale != 0.0f;
  {
    const T* qbase = static_cast<const T*>(io.q) + static_cast<int64_t>(row) * io.q_stride_n;
    constexpr int kQPerThread = tkKeys * kChunks / tkThreads;
    static_assert(tkKeys * kChunks % tkThreads == 0, "Q load");
    if (!q_scale_on) {
#pragma unroll
      for (int it = 0; it < kQPerThread; ++it) {
        const int i = it * tkThreads + tid;
        const int r = i / kChunks, c = i - r * kChunks;
        const int h = h0 + r;
        tk_cp16(tk_u32(sQ + L::off(r, c)),
                qbase + static_cast<int64_t>(h) * io.q_stride_h + c * 8, h < H);
      }
      tk_cp_commit();
      tk_cp_wait<0>();
    } else {
#pragma unroll
      for (int it = 0; it < kQPerThread; ++it) {
        const int i = it * tkThreads + tid;
        const int r = i / kChunks, c = i - r * kChunks;
        const int h = h0 + r;
        T v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = sp_from_f<T>(0.0f);
        if (h < H) {
          const T* src = qbase + static_cast<int64_t>(h) * io.q_stride_h + c * 8;
#pragma unroll
          for (int j = 0; j < 8; ++j) v[j] = sp_from_f<T>(sp_to_f<T>(src[j]) * io.q_scale);
        }
        *reinterpret_cast<uint4*>(sQ + L::off(r, c)) = *reinterpret_cast<const uint4*>(v);
      }
    }
  }
  __syncthreads();

  // ---- A fragments: warp w owns QK steps (w, w + 4, ...) for all 16 heads.
  const uint32_t sQ_u = tk_u32(sQ);
  uint32_t af[kStepsPerWarp][4];
#pragma unroll
  for (int s = 0; s < kStepsPerWarp; ++s) {
    tk_ldsm4(af[s], sQ_u + L::off(lane & 15, 2 * (warp + tkWarps * s) + (lane >> 4)) * 2);
  }
  __syncthreads();  // staging area is now free for the first K tile

  const int n_flat = keys.count();
  const int ntiles = (n_flat + TILE - 1) / TILE;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_cs = (lane >> 3) & 1;
  const int v_row = (lane & 7) + (((lane >> 3) & 1) << 3), v_cs = lane >> 4;
  const float score_scale = io.scale * kTkLog2e;

  // Fill `dst` with the next tile's latent rows: native records are a direct
  // 16-byte cp.async copy, every other codec is dequantized into the tile.
  auto load_tile = [&](T* dst, const SpRow* rows) {
    const char* dummy = static_cast<const char*>(io.q);
#pragma unroll
    for (int it = 0; it < kPerThread; ++it) {
      const int i = it * tkThreads + tid;
      const int r = i / kChunks, c = i - r * kChunks;
      const SpRow& rk = rows[r];
      bool done = false;
      if constexpr (std::is_same_v<DTypeKV, T>) {
        if (rk.native != 0) {
          tk_cp16(tk_u32(dst + L::off(r, c)),
                  rk.base != nullptr ? rk.base + c * 16 : dummy, rk.base != nullptr);
          done = true;
        }
      }
      if (done) continue;
      T v[8];
      if (rk.base != nullptr) {
        sp_read8_base<T, DTypeKV>(*rk.kv, rk.base, rk.slot, c * 8, v);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = sp_from_f<T>(0.0f);
      }
      *reinterpret_cast<uint4*>(dst + L::off(r, c)) = *reinterpret_cast<const uint4*>(v);
    }
  };

  float m_run[2] = {tk_ninf(), tk_ninf()};
  float l_run[2] = {0.0f, 0.0f};
  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  }

  if (ntiles > 0) {
    keys.prepare(0, TILE, sRow);
    __syncthreads();
    load_tile(sK, sRow);
    tk_cp_commit();
  }

  for (int t = 0; t < ntiles; ++t) {
    const int cur = t & 1, nxt = cur ^ 1;
    if (t + 1 < ntiles) {
      keys.prepare((t + 1) * TILE, TILE, sRow + nxt * TILE);
    }
    tk_cp_wait<0>();
    __syncthreads();  // tile t visible to every warp; the next key rows are ready
    if (t + 1 < ntiles) {
      load_tile(sK + nxt * kTileElems, sRow + nxt * TILE);
      tk_cp_commit();
    }
    T* sKc_p = sK + cur * kTileElems;
    const uint32_t sKc = tk_u32(sKc_p);

    // ---- QK partial of this warp's k steps (16 heads x TILE keys), fp32.
    float qk[kNT][4];
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) qk[nt][e] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < kStepsPerWarp; ++s) {
      const int kk = warp + tkWarps * s;
#pragma unroll
      for (int g = 0; g < kNT / 2; ++g) {
        uint32_t bb[4];
        tk_ldsm4(bb, sKc + L::off(g * tkKeys + b_row, 2 * kk + b_cs) * 2);
        tk_mma<T>(qk[2 * g + 0], af[s], bb[0], bb[1]);
        tk_mma<T>(qk[2 * g + 1], af[s], bb[2], bb[3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) sRed[((nt * 4 + e) * tkWarps + warp) * 32 + lane] = qk[nt][e];
    }
    __syncthreads();

    // ---- Sum the four warp partials once (P-once), scale, mask, softmax.
    const SpRow* cur_rows = sRow + cur * TILE;
    float p[kPairs];
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const float a0 = sRed[((pair * tkWarps + 0) * 32) + lane];
      const float a1 = sRed[((pair * tkWarps + 1) * 32) + lane];
      const float a2 = sRed[((pair * tkWarps + 2) * 32) + lane];
      const float a3 = sRed[((pair * tkWarps + 3) * 32) + lane];
      const int nt = pair >> 2, e = pair & 3;
      const int kcol = nt * 8 + 2 * (lane & 3) + (e & 1);
      p[pair] = cur_rows[kcol].code >= 0 ? ((a0 + a1) + (a2 + a3)) * score_scale : tk_ninf();
    }
    float mx0 = m_run[0], mx1 = m_run[1];
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
      if ((i & 2) == 0) mx0 = fmaxf(mx0, p[i]); else mx1 = fmaxf(mx1, p[i]);
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mu0 = mx0 == tk_ninf() ? 0.0f : mx0;
    const float mu1 = mx1 == tk_ninf() ? 0.0f : mx1;
    const float alpha0 = exp2f(m_run[0] - mu0);
    const float alpha1 = exp2f(m_run[1] - mu1);
    m_run[0] = mx0;
    m_run[1] = mx1;
    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
      const bool row0 = (i & 2) == 0;
      const float pi = exp2f(p[i] - (row0 ? mu0 : mu1));
      p[i] = pi;
      if (row0) sum0 += pi; else sum1 += pi;
    }
    l_run[0] = l_run[0] * alpha0 + sum0;
    l_run[1] = l_run[1] * alpha1 + sum1;

    // ---- PV: P (from the summed scores) x the 512-dim latent slice of the tile.
    uint32_t pa[kHalves][4];
#pragma unroll
    for (int h = 0; h < kHalves; ++h) {
      pa[h][0] = tk_pack2<T>(p[h * 8 + 0], p[h * 8 + 1]);
      pa[h][1] = tk_pack2<T>(p[h * 8 + 2], p[h * 8 + 3]);
      pa[h][2] = tk_pack2<T>(p[h * 8 + 4], p[h * 8 + 5]);
      pa[h][3] = tk_pack2<T>(p[h * 8 + 6], p[h * 8 + 7]);
    }
#pragma unroll
    for (int dp = 0; dp < 8; ++dp) {
      uint32_t bbh[kHalves][4];
#pragma unroll
      for (int h = 0; h < kHalves; ++h) {
        tk_ldsm4t(bbh[h], sKc + L::off(h * tkKeys + v_row, 2 * (8 * warp + dp) + v_cs) * 2);
      }
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        float* a = acc[2 * dp + g];
        a[0] *= alpha0;
        a[1] *= alpha0;
        a[2] *= alpha1;
        a[3] *= alpha1;
#pragma unroll
        for (int h = 0; h < kHalves; ++h) {
          tk_mma<T>(acc[2 * dp + g], pa[h], bbh[h][2 * g + 0], bbh[h][2 * g + 1]);
        }
      }
    }
  }

  // ---- Sink, normalisation, latent output slice and LSE.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
  }
  float oscale[2], lse[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int h = h0 + (lane >> 2) + 8 * r;
    const bool has_sink = io.sink != nullptr && h < H;
    const float sk = has_sink ? io.sink[h] * kTkLog2e : tk_ninf();
    const float mn = fmaxf(m_run[r], sk);
    const float al = mn == tk_ninf() ? 0.0f : exp2f(m_run[r] - mn);
    const float den = mn == tk_ninf()
                          ? 0.0f
                          : l_run[r] * al + (sk == tk_ninf() ? 0.0f : exp2f(sk - mn));
    oscale[r] = den > 0.0f ? al / den : 0.0f;
    lse[r] = den > 0.0f ? (mn + log2f(den)) * kTkLn2 : tk_ninf();
  }
  const int v_dim = io.v_dim;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = 128 * warp + 8 * i + 2 * (lane & 3);
    if (d >= v_dim) continue;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int h = h0 + (lane >> 2) + 8 * r;
      if (h >= H) continue;
      const float v0 = acc[i][2 * r] * oscale[r];
      const float v1 = acc[i][2 * r + 1] * oscale[r];
      DTypeO* dst = reinterpret_cast<DTypeO*>(io.o) + static_cast<int64_t>(row) * io.o_stride_n +
                    static_cast<int64_t>(h) * io.o_stride_h + d;
      if constexpr (std::is_same_v<DTypeO, float>) {
        dst[0] = v0;
        dst[1] = v1;
      } else {
        *reinterpret_cast<uint32_t*>(dst) = tk_pack2<DTypeO>(v0, v1);
      }
    }
  }
  if ((lane & 3) == 0 && io.lse != nullptr) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int h = h0 + (lane >> 2) + 8 * r;
      if (h < H) io.lse[static_cast<int64_t>(row) * H + h] = lse[r];
    }
  }
}

/// Visible-key list of one sparse-MLA query: the primary scope followed by the
/// optional extra scope, `topk_length`-truncated, `-1` holes kept as masked keys.
template <typename DTypeKV>
struct SpMlaKeys {
  const SparseMLAParams* p;
  int row, n0, n1, topk0, topk1;

  __device__ __forceinline__ int count() const { return n0 + n1; }

  __device__ __forceinline__ SpKeyRef resolve(int32_t code) const {
    const bool extra = (code & (1 << 30)) != 0;
    return SpKeyRef{extra ? &p->extra.kv : &p->primary.kv,
                    static_cast<int64_t>(code & ~(1 << 30))};
  }

  /// Resolve the keys `[base, base + count)`: the page table and record
  /// addressing are evaluated once per key (thread i handles key base + i),
  /// never once per 16-byte chunk.
  __device__ __forceinline__ void prepare(int base, int count, SpRow* dst) const {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= count) return;
    const int flat = base + i;
    int32_t code = -1;
    if (flat < n0) {
      code = p->primary.indices[static_cast<int64_t>(row) * topk0 + flat];
    } else if (flat < n0 + n1) {
      const int32_t x = p->extra.indices[static_cast<int64_t>(row) * topk1 + (flat - n0)];
      code = x < 0 ? -1 : ((1 << 30) | x);
    }
    SpRow& r = dst[i];
    r.code = code;
    r.kv = nullptr;
    r.base = nullptr;
    r.slot = 0;
    r.native = 0;
    if (code >= 0) {
      const SpKeyRef ref = resolve(code);
      r.kv = ref.kv;
      r.slot = ref.slot;
      r.base = sp_row_base<DTypeKV>(*ref.kv, ref.slot);
      r.native = ref.kv->k_codec == kv::KVCodec::kNative ? 1 : 0;
    }
  }
};

/// Scalar fallback of sparse MLA (fp32 q/o and shapes outside the tensor-core
/// tiles): exact fp32 softmax over the compacted valid entries, one CTA per
/// (row, head). Kept numerically identical to the previous body.
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__device__ void sp_mla_scalar(const SparseMLAParams& params, float* smem) {
  const int row = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  if (row >= params.num_tokens || head >= params.num_heads) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int qk_dim = params.qk_dim > 0 ? params.qk_dim : 576;
  const int v_dim = params.v_dim > 0 ? params.v_dim : 512;
  const int topk0 = params.primary.topk;
  const int topk1 = params.extra.indices != nullptr ? params.extra.topk : 0;
  const int cap = topk0 + topk1;
  float* sQ = smem;                                // [qk_dim]
  float* sLogits = sQ + qk_dim;                    // [cap]
  float* sRed = sLogits + (cap > 0 ? cap : 1);     // [32]
  int32_t* sMap = reinterpret_cast<int32_t*>(sRed + 32);  // [cap]

  for (int d = tid; d < qk_dim; d += blockDim.x) {
    const DTypeQ* qbase = reinterpret_cast<const DTypeQ*>(params.q);
    const DTypeQ* q = qbase + static_cast<int64_t>(row) * params.q_stride_n +
                      static_cast<int64_t>(head) * params.q_stride_h;
    float v = sp_to_f<DTypeQ>(q[d]);
    if (params.mod.q_scale != 0.0f) v *= params.mod.q_scale;
    sQ[d] = v;
  }
  if (tid < 32) sRed[tid] = 0.0f;
  __syncthreads();

  int total = 0;
  if (tid == 0) {
    const int n0 = params.primary.indices != nullptr
                       ? (params.primary.topk_length != nullptr
                              ? (params.primary.topk_length[row] < topk0
                                     ? params.primary.topk_length[row]
                                     : topk0)
                              : topk0)
                       : 0;
    for (int i = 0; i < n0; ++i) {
      const int32_t idx = params.primary.indices[static_cast<int64_t>(row) * topk0 + i];
      if (idx >= 0) sMap[total++] = idx;
    }
    if (topk1 > 0) {
      const int n1 = params.extra.topk_length != nullptr
                         ? (params.extra.topk_length[row] < topk1 ? params.extra.topk_length[row]
                                                                  : topk1)
                         : topk1;
      for (int i = 0; i < n1; ++i) {
        const int32_t idx = params.extra.indices[static_cast<int64_t>(row) * topk1 + i];
        if (idx >= 0) sMap[total++] = (1 << 30) | idx;
      }
    }
    sRed[31] = static_cast<float>(total);
  }
  __syncthreads();
  total = static_cast<int>(sRed[31]);
  if (tid < 32) sRed[tid] = 0.0f;
  __syncthreads();
  const float scale = params.mod.sm_scale != 0.0f ? params.mod.sm_scale
                                                   : rsqrtf(static_cast<float>(qk_dim));
  for (int i = tid; i < total; i += blockDim.x) {
    const int32_t code = sMap[i];
    const bool extra = (code & (1 << 30)) != 0;
    const int64_t slot = static_cast<int64_t>(code & ~(1 << 30));
    const kv::PagedKVDesc& d = extra ? params.extra.kv : params.primary.kv;
    float dot = 0.0f;
    for (int k = 0; k < qk_dim; ++k) {
      dot = __fmaf_rn(sQ[k], sp_latent<DTypeKV>(d, slot, k, qk_dim), dot);
    }
    sLogits[i] = dot * scale;
  }
  __syncthreads();
  float vmax = sp_neg_inf();
  for (int i = tid; i < total; i += blockDim.x) vmax = fmaxf(vmax, sLogits[i]);
  vmax = sp_block_max(vmax, sRed, blockDim.x);
  const float sink = params.mod.sink_logits != nullptr ? params.mod.sink_logits[head] : sp_neg_inf();
  const float m = fmaxf(vmax, sink);
  float vsum = 0.0f;
  for (int i = tid; i < total; i += blockDim.x) {
    if (sLogits[i] > sp_neg_inf()) vsum += __expf(sLogits[i] - m);
  }
  if (tid == 0 && sink > sp_neg_inf()) vsum += __expf(sink - m);
  const float denom = sp_block_sum(vsum, sRed, blockDim.x);
  const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
  DTypeO* dbase = reinterpret_cast<DTypeO*>(params.o);
  DTypeO* dst = dbase + static_cast<int64_t>(row) * params.o_stride_n +
                static_cast<int64_t>(head) * params.o_stride_h;
  for (int d = tid; d < v_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < total; ++i) {
      const float p = sLogits[i] > sp_neg_inf() ? __expf(sLogits[i] - m) : 0.0f;
      if (p == 0.0f) continue;
      const int32_t code = sMap[i];
      const bool extra = (code & (1 << 30)) != 0;
      const int64_t slot = static_cast<int64_t>(code & ~(1 << 30));
      const kv::PagedKVDesc& cache = extra ? params.extra.kv : params.primary.kv;
      acc = __fmaf_rn(p * inv, sp_latent<DTypeKV>(cache, slot, d, qk_dim), acc);
    }
    dst[d] = sp_from_f<DTypeO>(acc);
  }
  if (tid == 0 && params.lse != nullptr) {
    params.lse[static_cast<int64_t>(row) * params.num_heads + head] =
        denom > 0.0f ? (m + __logf(denom)) : sp_neg_inf();
  }
}

}  // namespace

template <int HEADS_PER_CTA, int TOPK_TILE, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename ArchTag>
__global__ void __launch_bounds__(tkThreads)
    sparse_mla_kernel(const __grid_constant__ SparseMLAParams params) {
  static_assert(HEADS_PER_CTA == tkKeys, "16 query heads per CTA");
  (void)sizeof(ArchTag);
  (void)sizeof(TOPK_TILE);
  const int row = static_cast<int>(blockIdx.x);
  const int h0 = static_cast<int>(blockIdx.y) * tkKeys;
  if (row >= params.num_tokens || h0 >= params.num_heads) return;
  const int qk_dim = params.qk_dim > 0 ? params.qk_dim : 576;
  const int v_dim = params.v_dim > 0 ? params.v_dim : 512;
  extern __shared__ __align__(16) unsigned char sp_tc_smem[];
  const bool tc_dtype = std::is_same_v<DTypeQ, __nv_bfloat16> || std::is_same_v<DTypeQ, __half>;
  if (!tc_dtype || v_dim > 512 || (qk_dim != 576 && qk_dim != 512)) {
    sp_mla_scalar<DTypeQ, DTypeKV, DTypeO>(params, reinterpret_cast<float*>(sp_tc_smem));
    return;
  }
  const int topk0 = params.primary.topk;
  const int topk1 = params.extra.indices != nullptr ? params.extra.topk : 0;
  SpMlaKeys<DTypeKV> keys{};
  keys.p = &params;
  keys.row = row;
  keys.topk0 = topk0;
  keys.topk1 = topk1;
  keys.n0 = params.primary.indices != nullptr
                ? (params.primary.topk_length != nullptr
                       ? min(max(params.primary.topk_length[row], 0), topk0)
                       : topk0)
                : 0;
  keys.n1 = params.extra.indices != nullptr
                ? (params.extra.topk_length != nullptr
                       ? min(max(params.extra.topk_length[row], 0), topk1)
                       : topk1)
                : 0;
  SpAttnIo io{};
  io.q = params.q;
  io.q_stride_n = params.q_stride_n;
  io.q_stride_h = params.q_stride_h;
  io.o = params.o;
  io.o_stride_n = params.o_stride_n;
  io.o_stride_h = params.o_stride_h;
  io.lse = params.lse;
  io.sink = params.mod.sink_logits;
  io.num_heads = params.num_heads;
  io.scale = params.mod.sm_scale != 0.0f ? params.mod.sm_scale
                                         : rsqrtf(static_cast<float>(qk_dim));
  io.q_scale = params.mod.q_scale;
  io.v_dim = v_dim;
  if (qk_dim == 576) {
    sp_tc_attend<576, 512, spTile, DTypeQ, DTypeKV, DTypeO>(io, keys, row, sp_tc_smem);
  } else {
    sp_tc_attend<512, 512, spTile, DTypeQ, DTypeKV, DTypeO>(io, keys, row, sp_tc_smem);
  }
}

namespace {

/// Shared-memory bytes of one sp_tc_attend<QK_DIM, ..., TILE> staging pass.
template <int QK_DIM, int TILE>
inline size_t sp_tc_smem_bytes_t() {
  constexpr int kPairs = (TILE / 8) * 4;
  return static_cast<size_t>(2 * TILE * QK_DIM) * sizeof(__nv_bfloat16) +
         static_cast<size_t>(kPairs * tkWarps * 32) * sizeof(float) +
         static_cast<size_t>(2 * TILE) * sizeof(SpRow) + 8 * sizeof(float);
}

inline size_t sp_smem_bytes(int qk_dim) {
  return static_cast<size_t>(2 * spTile * qk_dim) * sizeof(__nv_bfloat16) +
         static_cast<size_t>(spRedWords) * sizeof(float) +
         static_cast<size_t>(2 * spTile) * sizeof(SpRow) + 8 * sizeof(float);
}

Status sparse_mla_dispatch(const SparseMLAParams& params, const LaunchContext& ctx) {
  const int qk_dim = params.qk_dim > 0 ? params.qk_dim : 576;
  const int v_dim = params.v_dim > 0 ? params.v_dim : 512;
  const int topk = params.primary.topk + (params.extra.indices != nullptr ? params.extra.topk : 0);
  const bool tc_dtype = params.q_dtype == DType::kBFloat16 || params.q_dtype == DType::kFloat16;
  const bool tc = tc_dtype && v_dim <= 512 && (qk_dim == 576 || qk_dim == 512) &&
                  params.o_dtype == params.q_dtype;
  const size_t smem = tc ? sp_smem_bytes(qk_dim)
                         : static_cast<size_t>(qk_dim + topk + 32) * sizeof(float) +
                               static_cast<size_t>(topk) * sizeof(int32_t);
  const dim3 grid(static_cast<unsigned>(params.num_tokens),
                  static_cast<unsigned>((params.num_heads + tkKeys - 1) / tkKeys));
  const dim3 block(tkThreads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_SPARSE_MLA_TC(QT, KT, OT)                                                    \
  do {                                                                                    \
    if (smem > 48u * 1024u) {                                                             \
      (void)cudaFuncSetAttribute(                                                         \
          reinterpret_cast<const void*>(&sparse_mla_kernel<16, 64, QT, KT, OT, Sm80>),     \
          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));           \
    }                                                                                     \
    sparse_mla_kernel<16, 64, QT, KT, OT, Sm80><<<grid, block, smem, stream>>>(params);    \
    return Status::kSuccess;                                                              \
  } while (0)
  if (tc) {
    if (params.q_dtype == DType::kBFloat16) {
      KOPS_SPARSE_MLA_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
    } else {
      KOPS_SPARSE_MLA_TC(__half, __half, __half);
    }
  }
  if (params.q_dtype == DType::kFloat32 && params.o_dtype == DType::kFloat32) {
    KOPS_SPARSE_MLA_TC(float, float, float);
  }
  if (params.q_dtype == DType::kFloat16 && params.o_dtype == DType::kFloat16) {
    KOPS_SPARSE_MLA_TC(__half, __half, __half);
  }
  if (params.q_dtype == DType::kBFloat16 && params.o_dtype == DType::kBFloat16) {
    KOPS_SPARSE_MLA_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
  }
#undef KOPS_SPARSE_MLA_TC
  return detail::set_error(Status::kUnsupportedDtype, "sparse_mla: q/o dtype pair");
}

}  // namespace

Status sparse_mla_run(const SparseMLAParams& params, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sparse_mla: CUDA backend only");
  }
  if (params.q == nullptr || params.o == nullptr || params.num_tokens < 0 || params.num_heads <= 0 ||
      params.qk_dim <= 0 || params.v_dim <= 0) {
    return detail::set_error(Status::kInvalidArgument, "sparse_mla: bad argument");
  }
  if (params.variant != SparseMLAVariant::kPlain) {
    return detail::set_error(Status::kNotImplemented, "sparse_mla: fused V4.1 variant lands later");
  }
  if (params.primary.kv.k_codec == kv::KVCodec::kIndexerFP4 ||
      params.extra.kv.k_codec == kv::KVCodec::kIndexerFP4 ||
      params.primary.kv.k_codec == kv::KVCodec::kIndexerFP8) {
    return detail::set_error(Status::kUnsupportedDtype, "sparse_mla: indexer caches are not latent");
  }
  if (params.primary.indices == nullptr && params.extra.indices == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "sparse_mla: no index scope");
  }
  Status s = kv::kv_desc_validate(params.primary.kv);
  if (s != Status::kSuccess) return s;
  if (params.extra.indices != nullptr) {
    s = kv::kv_desc_validate(params.extra.kv);
    if (s != Status::kSuccess) return s;
  }
  if (params.primary.kv.k_codec == kv::KVCodec::kNative &&
      params.primary.kv.k_dtype != params.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "sparse_mla: native KV/q dtype mismatch");
  }
  if (params.extra.indices != nullptr && params.extra.kv.k_codec == kv::KVCodec::kNative &&
      params.extra.kv.k_dtype != params.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "sparse_mla: native extra KV/q mismatch");
  }
  if (params.primary.kv.k_codec == kv::KVCodec::kMLAFp8DSV40 &&
      params.primary.kv.k_scale_pages == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "sparse_mla: V4.0 primary cache needs k_scale_pages");
  }
  if (params.extra.indices != nullptr && params.extra.kv.k_codec == kv::KVCodec::kMLAFp8DSV40 &&
      params.extra.kv.k_scale_pages == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "sparse_mla: V4.0 extra cache needs k_scale_pages");
  }
  if ((reinterpret_cast<uintptr_t>(params.q) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.o) & 15u) != 0 ||
      (params.q_stride_n < static_cast<int64_t>(params.num_heads) * params.qk_dim) ||
      (params.o_stride_n < static_cast<int64_t>(params.num_heads) * params.v_dim)) {
    return detail::set_error(Status::kMisaligned, "sparse_mla: q/o alignment or strides");
  }
  const int topk = params.primary.topk + (params.extra.indices != nullptr ? params.extra.topk : 0);
  if (topk <= 0 || topk > 8192) {
    return detail::set_error(Status::kUnsupportedShape, "sparse_mla: topk %d", topk);
  }
  return sparse_mla_dispatch(params, ctx);
}

// -----------------------------------------------------------------------------
// DeepSeek-V4 KV compressor (generic/T1). One CTA per (sequence, entry); the
// entry is built channel-by-channel from ratio (HCA) or 2*ratio (CSA) tokens.
// -----------------------------------------------------------------------------
namespace {

template <typename T>
__device__ __forceinline__ float comp_f(T v) { return sp_to_f<T>(v); }

template <typename DTypeIn>
__device__ __forceinline__ float comp_load(const void* p, int64_t i) {
  return comp_f<DTypeIn>(reinterpret_cast<const DTypeIn*>(p)[i]);
}

template <typename DTypeCache>
__device__ __forceinline__ DTypeCache comp_store(float v) {
  if constexpr (std::is_same_v<DTypeCache, __nv_bfloat16>) return __float2bfloat16_rn(v);
  else if constexpr (std::is_same_v<DTypeCache, __half>) return __float2half_rn(v);
  else return v;
}

}  // namespace

template <typename DTypeIn, typename DTypeCache, int RATIO_M, int ENTRY_DIM, bool kOverlap>
__global__ void kv_compressor_kernel(const __grid_constant__ KVCompressorParams params) {
  const int entry = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  if (b >= params.batch_size || params.cu_seqlens == nullptr) return;
  const int start = params.cu_seqlens[b];
  const int end = params.cu_seqlens[b + 1];
  const int ratio = params.ratio_m > 0 ? params.ratio_m : (RATIO_M > 0 ? RATIO_M : 1);
  const int n = end > start ? end - start : 0;
  const int num_entries = n / ratio;
  if (entry >= num_entries) return;
  const int dim = params.entry_dim > 0 ? params.entry_dim : ENTRY_DIM;
  const int cand = kOverlap ? 2 * ratio : ratio;
  const int base = start + entry * ratio;
  extern __shared__ float sEntry[];
  for (int c = static_cast<int>(threadIdx.x); c < dim; c += blockDim.x) {
    float mx = -INFINITY;
    for (int k = 0; k < cand; ++k) {
      const int t = base + k;
      if (t < start || t >= end) continue;
      const int64_t row = static_cast<int64_t>(t) * (params.token_stride > 0 ? params.token_stride : dim) + c;
      const float z = (kOverlap && k < ratio) ? comp_load<DTypeIn>(params.zb, row)
                                               : comp_load<DTypeIn>(params.za, row);
      const int bias_k = k % ratio;
      const float bias = (kOverlap && k < ratio)
                             ? (params.bias_b != nullptr ? params.bias_b[bias_k * dim + c] : 0.0f)
                             : (params.bias_a != nullptr ? params.bias_a[bias_k * dim + c] : 0.0f);
      mx = fmaxf(mx, z + bias);
    }
    float den = 0.0f;
    float acc = 0.0f;
    for (int k = 0; k < cand; ++k) {
      const int t = base + k;
      if (t < start || t >= end) continue;
      const int64_t row = static_cast<int64_t>(t) * (params.token_stride > 0 ? params.token_stride : dim) + c;
      const float z = (kOverlap && k < ratio) ? comp_load<DTypeIn>(params.zb, row)
                                               : comp_load<DTypeIn>(params.za, row);
      const float bias = (kOverlap && k < ratio)
                             ? (params.bias_b != nullptr ? params.bias_b[(k % ratio) * dim + c] : 0.0f)
                             : (params.bias_a != nullptr ? params.bias_a[(k % ratio) * dim + c] : 0.0f);
      const float w = __expf(z + bias - mx);
      den += w;
      const float v = (kOverlap && k < ratio) ? comp_load<DTypeIn>(params.cb, row)
                                               : comp_load<DTypeIn>(params.ca, row);
      acc = __fmaf_rn(w, v, acc);
    }
    sEntry[c] = den > 0.0f ? acc / den : 0.0f;
  }
  __syncthreads();
  if (params.norm_weight != nullptr) {
    float ss = 0.0f;
    for (int c = static_cast<int>(threadIdx.x); c < dim; c += blockDim.x) ss += sEntry[c] * sEntry[c];
    ss = sp_block_sum(ss, sEntry + dim, blockDim.x);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + params.norm_eps);
    for (int c = static_cast<int>(threadIdx.x); c < dim; c += blockDim.x) {
      sEntry[c] *= inv * params.norm_scale * comp_load<DTypeIn>(params.norm_weight, c);
    }
    __syncthreads();
  }
  const int64_t slot = params.out_slot_mapping != nullptr
                           ? params.out_slot_mapping[static_cast<int64_t>(b) * params.max_entries + entry]
                           : -1;
  if (slot >= 0) {
    const int32_t page_size = params.out_cache.page_size > 0 ? params.out_cache.page_size : 1;
    const int64_t page = slot / page_size;
    const int off = static_cast<int>(slot - page * page_size);
    const bool bytes = params.out_cache.stride_unit == StrideUnit::kByte ||
                       params.out_cache.entry_bytes != 0;
    const int64_t unit = bytes ? 1 : static_cast<int64_t>(sizeof(DTypeCache));
    const int64_t page_stride = (params.out_cache.k_stride_page > 0
                                     ? params.out_cache.k_stride_page
                                     : static_cast<int64_t>(page_size) * dim) *
                                unit;
    const int64_t token_stride = (params.out_cache.k_stride_token > 0
                                      ? params.out_cache.k_stride_token
                                      : dim) *
                                 unit;
    char* base_ptr = static_cast<char*>(params.out_cache.k_data) + page * page_stride + off * token_stride;
    DTypeCache* out = reinterpret_cast<DTypeCache*>(base_ptr);
    for (int c = static_cast<int>(threadIdx.x); c < dim; c += blockDim.x) out[c] = comp_store<DTypeCache>(sEntry[c]);
  }
  if (entry == 0 && threadIdx.x == 0 && params.num_entries_out != nullptr) {
    params.num_entries_out[b] = num_entries;
  }
}

namespace {

Status launch_compressor(const KVCompressorParams& p, DType in_dtype, cudaStream_t stream) {
  if (p.ratio_m <= 0 || p.entry_dim <= 0 || p.batch_size < 0 || p.max_entries <= 0 ||
      p.cu_seqlens == nullptr || p.out_slot_mapping == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "kv_compressor: bad argument");
  }
  if (p.tail_state != nullptr) {
    return detail::set_error(Status::kNotImplemented,
                             "kv_compressor: tail_state parity lands with the decode path");
  }
  if (p.entry_dim > 1024) {
    return detail::set_error(Status::kUnsupportedShape, "kv_compressor: entry_dim %d", p.entry_dim);
  }
  const size_t smem = static_cast<size_t>(p.entry_dim + 32) * sizeof(float);
  const dim3 grid(static_cast<unsigned>(p.max_entries), static_cast<unsigned>(p.batch_size));
  const dim3 block(128);
#define KOPS_LAUNCH_COMP(T, C, OV)                                                              \
  do {                                                                                         \
    kv_compressor_kernel<T, C, 0, 0, OV><<<grid, block, smem, stream>>>(p);                     \
    return Status::kSuccess;                                                                    \
  } while (0)
  if (p.overlap) {
    if (in_dtype == DType::kBFloat16) KOPS_LAUNCH_COMP(__nv_bfloat16, __nv_bfloat16, true);
    if (in_dtype == DType::kFloat16) KOPS_LAUNCH_COMP(__half, __half, true);
  } else {
    if (in_dtype == DType::kBFloat16) KOPS_LAUNCH_COMP(__nv_bfloat16, __nv_bfloat16, false);
    if (in_dtype == DType::kFloat16) KOPS_LAUNCH_COMP(__half, __half, false);
  }
#undef KOPS_LAUNCH_COMP
  return detail::set_error(Status::kUnsupportedDtype, "kv_compressor: in dtype");
}

}  // namespace

Status kv_compress(const KVCompressorParams& params, DType in_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "kv_compress: CUDA backend only");
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  return launch_compressor(params, in_dtype, stream);
}

// -----------------------------------------------------------------------------
// A13 DeepSeek-V4 compressed attention: selected/dense compressed entries + SWA
// + sink in one online softmax (generic/T1 body).
//
// K and V are the same record, so one record is read once and used for both the
// QK dot and the P*V accumulation. A warp owns one key end to end: its 32 lanes
// hold the whole record (8-element vector reads), keep their own running
// max/denominator/accumulators in registers, and merge once at the end. That
// leaves two block-wide barriers per CTA instead of one per key tile, and one
// softmax weight per key instead of one per (key, output dim).
// -----------------------------------------------------------------------------
namespace {

__device__ __forceinline__ int64_t ca_slot(const kv::PagedKVDesc& d, int req, int64_t logical) {
  if (d.page_size <= 0) return logical;
  const int64_t lp = logical / d.page_size;
  int32_t page = -1;
  if (d.table_format == kv::PageTableFormat::kBlockTable && d.block_table != nullptr) {
    page = d.block_table[static_cast<int64_t>(req) * d.block_table_stride + lp];
  } else if (d.table_format == kv::PageTableFormat::kCSR && d.kv_indptr != nullptr &&
             d.kv_page_indices != nullptr) {
    page = d.kv_page_indices[d.kv_indptr[req] + lp];
  } else {
    page = static_cast<int32_t>(lp);
  }
  if (page < 0) return -1;
  return static_cast<int64_t>(page) * d.page_size + (logical % d.page_size);
}

/// CTA size of the T1 body: 8 warps, one key per warp at a time.
constexpr int kCAttnThreads = 256;
constexpr int kCAttnWarps = kCAttnThreads / 32;
/// Elements per vector read, and vector reads per lane (a record of up to
/// 32 * kCAttnVec * kCAttnSteps elements is held in one warp's registers).
constexpr int kCAttnVec = 8;
constexpr int kCAttnSteps = 2;
constexpr int kCAttnLaneVals = kCAttnVec * kCAttnSteps;
constexpr int kCAttnMaxDim = 32 * kCAttnLaneVals;

/// Eight consecutive latent values of a record as fp32. A native record with a
/// 16-byte-aligned base takes one vector load; every other codec goes through
/// the shared sp_read8_base reader, so the codec switch runs once per 8 dims
/// instead of once per dim.
template <typename DTypeKV>
__device__ __forceinline__ void ca_read8(const kv::PagedKVDesc& d, const char* rec, int64_t slot,
                                         int dim0, float* out) {
  if (d.k_codec == kv::KVCodec::kNative && (reinterpret_cast<uintptr_t>(rec) & 15u) == 0) {
    if constexpr (std::is_same_v<DTypeKV, __nv_bfloat16> || std::is_same_v<DTypeKV, __half>) {
      const uint4 v = *reinterpret_cast<const uint4*>(reinterpret_cast<const DTypeKV*>(rec) + dim0);
      const DTypeKV* h = reinterpret_cast<const DTypeKV*>(&v);
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i] = sp_to_f<DTypeKV>(h[i]);
      return;
    }
  }
  sp_read8_base<float, DTypeKV>(d, rec, slot, dim0, out);
}

/// Shared-memory bytes of one CTA: the key map, the per-warp staging area for
/// the final merge, and the per-warp softmax scalars (max, denominator, merged
/// weight, weighted denominator).
inline size_t ca_smem_bytes(int32_t cap, int32_t value_dim) {
  return static_cast<size_t>(cap) * sizeof(int32_t) +
         static_cast<size_t>(kCAttnWarps) * static_cast<size_t>(value_dim) * sizeof(float) +
         static_cast<size_t>(kCAttnWarps) * 4 * sizeof(float);
}

/// Tensor-core tiles of the head-grouped compressed-attention body: 16 query
/// heads x TILE keys per staging pass. Both are multiples of the 16-key
/// m16n8k16 PV step. The 16-key tile is ~38 KiB (two CTAs per 100 KiB SM), the
/// 32-key tile ~76 KiB (one CTA) but halves the barrier count, so it wins when
/// the launch has too few CTAs to fill the SMs and loses when it has enough to
/// occupy every SM with two resident CTAs.
constexpr int kCaTcTile = 16;
constexpr int kCaTcTileWide = 32;
constexpr int kCaTcDim = 512;

/// Tile for this launch: a grid of at most 16 (row, 16-head group) CTAs cannot
/// fill the SMs, so it takes the wider tile; a larger grid keeps the two-CTA
/// tile for throughput. Host launch geometry and the device branch both call
/// this, so the shared-memory size they choose cannot disagree.
__host__ __device__ __forceinline__ int ca_tc_tile(const CompressedAttentionParams& p) {
  const int groups = (p.num_q_heads + 15) / 16;
  const int64_t ctas = static_cast<int64_t>(p.num_tokens) * groups;
  return ctas <= 16 ? kCaTcTileWide : kCaTcTile;
}

/// Shapes and alignments the m16n8k16 body supports. The scalar reader keeps
/// every other contract (entry_dim < 512, odd value_dim, unaligned q/o, fp32
/// paths) unchanged; host launch geometry and device routing both call this, so
/// they cannot disagree about the shared-memory size or block size.
__host__ __device__ __forceinline__ bool ca_tc_eligible(const CompressedAttentionParams& p) {
  if (p.entry_dim != kCaTcDim || p.value_dim <= 0 || p.value_dim > kCaTcDim) return false;
  if ((p.value_dim & 1) != 0) return false;
  if (p.q_stride_h < p.entry_dim || p.o_stride_h < p.value_dim) return false;
  if ((p.q_stride_h & 7) != 0 || (p.o_stride_h & 1) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(p.q) & 15u) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(p.o) & 3u) != 0) return false;
  return true;
}

/// Visible-key list of one compressed-attention query row: the selected (CSA)
/// or dense (HCA) compressed entries followed by the SWA window. `-1` selected
/// holes and unmapped pages stay masked keys; the compressed/window bit picks
/// the cache the record comes from. Address resolution happens once per key
/// (`prepare`), never once per 16-byte chunk.
template <typename DTypeKV>
struct CaKeys {
  const CompressedAttentionParams* p;
  int row, req, n_sel, n_win, sel_stride;
  int64_t win_end;

  __device__ __forceinline__ int count() const { return n_sel + n_win; }

  __device__ __forceinline__ SpKeyRef resolve(int32_t code) const {
    const bool win = (code & (1 << 30)) != 0;
    return SpKeyRef{win ? &p->window : &p->compressed,
                    static_cast<int64_t>(code & ~(1 << 30))};
  }

  __device__ __forceinline__ void prepare(int base, int n, SpRow* dst) const {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= n) return;
    const int flat = base + i;
    int32_t code = -1;
    if (flat < n_sel) {
      if (p->selected != nullptr) {
        code = p->selected[static_cast<int64_t>(row) * sel_stride + flat];
      } else {
        const int64_t slot = ca_slot(p->compressed, req, flat);
        if (slot >= 0) code = static_cast<int32_t>(slot);
      }
    } else if (flat < n_sel + n_win) {
      const int64_t pos = win_end - p->window_size + 1 + (flat - n_sel);
      if (pos >= 0) {
        const int64_t slot = ca_slot(p->window, req, pos);
        if (slot >= 0) code = (1 << 30) | static_cast<int32_t>(slot);
      }
    }
    SpRow& r = dst[i];
    r.code = code;
    r.kv = nullptr;
    r.base = nullptr;
    r.slot = 0;
    r.native = 0;
    if (code >= 0) {
      const SpKeyRef ref = resolve(code);
      r.kv = ref.kv;
      r.slot = ref.slot;
      r.base = sp_row_base<DTypeKV>(*ref.kv, ref.slot);
      r.native = ref.kv->k_codec == kv::KVCodec::kNative ? 1 : 0;
    }
  }
};

/// One CTA: 16 query heads of one query row over its compressed + window keys,
/// with the sink folded into the same online softmax.
template <int ENTRY_DIM, int VALUE_DIM, int TILE, typename DTypeQ, typename DTypeKV, typename DTypeO>
__device__ __forceinline__ void ca_tc_attend(const CompressedAttentionParams& params,
                                             unsigned char* smem) {
  const int row = static_cast<int>(blockIdx.x);
  const int h0 = static_cast<int>(blockIdx.y) * tkKeys;
  if (row >= params.num_tokens || h0 >= params.num_q_heads) return;
  const int topk = params.topk;
  const int sel_stride = params.selected_stride > 0 ? params.selected_stride : topk;
  const int req = params.token_to_request != nullptr ? params.token_to_request[row] : row;
  const int sel_len =
      params.selected != nullptr
          ? (params.selected_topk_length != nullptr
                 ? (params.selected_topk_length[row] < topk ? params.selected_topk_length[row]
                                                           : topk)
                 : topk)
          : (params.num_compressed != nullptr
                 ? (params.num_compressed[req] < topk ? params.num_compressed[req] : topk)
                 : 0);

  CaKeys<DTypeKV> keys{};
  keys.p = &params;
  keys.row = row;
  keys.req = req;
  keys.sel_stride = sel_stride;
  keys.n_sel = sel_len > 0 ? sel_len : 0;
  keys.n_win = (params.window_size > 0 && params.positions != nullptr) ? params.window_size : 0;
  keys.win_end = params.positions != nullptr ? params.positions[row] : 0;

  SpAttnIo io{};
  io.q = params.q;
  io.q_stride_n = params.q_stride_n;
  io.q_stride_h = params.q_stride_h;
  io.o = params.o;
  io.o_stride_n = params.o_stride_n;
  io.o_stride_h = params.o_stride_h;
  io.lse = params.lse;
  io.sink = params.mod.sink_logits;
  io.num_heads = params.num_q_heads;
  io.scale = params.mod.sm_scale != 0.0f ? params.mod.sm_scale
                                         : rsqrtf(static_cast<float>(ENTRY_DIM));
  io.q_scale = 0.0f;
  io.v_dim = params.value_dim > 0 ? params.value_dim : VALUE_DIM;

  sp_tc_attend<ENTRY_DIM, VALUE_DIM, TILE, DTypeQ, DTypeKV, DTypeO>(io, keys, row, smem);
}

}  // namespace

template <int HEADS_PER_CTA, int ENTRY_DIM, int VALUE_DIM, typename DTypeQ, typename DTypeKV,
          typename DTypeO, typename ArchTag>
__global__ void compressed_attention_kernel(const __grid_constant__ CompressedAttentionParams params) {
  static_assert(HEADS_PER_CTA == tkKeys, "16 query heads per CTA");
  extern __shared__ __align__(16) unsigned char ca_smem[];
  // Head-grouped tensor-core body: one CTA stages each K/V tile once for 16
  // query heads (grid.y = Hq / 16, block = tkThreads). The scalar reader below
  // keeps the ungrouped geometry (grid.y = Hq, one key per warp) for every
  // shape the m16n8k16 tiles cannot express.
  if (ca_tc_eligible(params)) {
    if (ca_tc_tile(params) == kCaTcTileWide) {
      ca_tc_attend<ENTRY_DIM, VALUE_DIM, kCaTcTileWide, DTypeQ, DTypeKV, DTypeO>(params, ca_smem);
    } else {
      ca_tc_attend<ENTRY_DIM, VALUE_DIM, kCaTcTile, DTypeQ, DTypeKV, DTypeO>(params, ca_smem);
    }
    return;
  }
  constexpr int kThreads = kCAttnThreads;
  constexpr int kWarps = kCAttnWarps;
  constexpr int kVec = kCAttnVec;
  constexpr int kLaneVals = kCAttnLaneVals;
  const int row = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  if (row >= params.num_tokens || head >= params.num_q_heads) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int dim = params.entry_dim > 0 ? params.entry_dim : ENTRY_DIM;
  const int vdim = params.value_dim > 0 ? params.value_dim : VALUE_DIM;
  const int topk = params.topk;   // selected count (CSA) or dense-range cap (HCA)
  const int cap = topk + (params.window_size > 0 ? params.window_size : 0);
  if (cap <= 0) return;
  float* smem = reinterpret_cast<float*>(ca_smem);
  int32_t* sMap = reinterpret_cast<int32_t*>(smem);            // [cap]
  float* sMerge = reinterpret_cast<float*>(sMap + cap);        // [kWarps][vdim]
  float* sState = sMerge + kWarps * vdim;                      // [kWarps][2] max, den
  float* sWeight = sState + kWarps * 2;                        // [kWarps] merged weight
  float* sPart = sWeight + kWarps;                             // [kWarps] weighted den

  // Each lane holds `kLaneVals` dims of every record it scores, and the same
  // dims of q, in registers.
  const DTypeQ* qrow = reinterpret_cast<const DTypeQ*>(params.q) +
                       static_cast<int64_t>(row) * params.q_stride_n +
                       static_cast<int64_t>(head) * params.q_stride_h;
  float qreg[kLaneVals];
  for (int s = 0; s < kCAttnSteps; ++s) {
    const int d0 = (s * 32 + lane) * kVec;
    for (int u = 0; u < kVec; ++u) {
      const int d = d0 + u;
      qreg[s * kVec + u] = d < dim ? sp_to_f<DTypeQ>(qrow[d]) : 0.0f;
    }
  }

  // Key map, filled in parallel: every thread writes its strided share of the
  // selected/dense range and of the SWA range. An index that is -1 (or a page
  // the table does not map) stays a hole (-1) and becomes a zero-weight key.
  const int req = params.token_to_request != nullptr ? params.token_to_request[row] : row;
  const int n_sel = params.selected != nullptr
                        ? (params.selected_topk_length != nullptr
                               ? (params.selected_topk_length[row] < topk
                                      ? params.selected_topk_length[row]
                                      : topk)
                               : topk)
                        : (params.num_compressed != nullptr
                               ? (params.num_compressed[req] < topk ? params.num_compressed[req]
                                                                    : topk)
                               : 0);
  const int n_win =
      (params.window_size > 0 && params.positions != nullptr) ? params.window_size : 0;
  const int sel_stride = params.selected_stride > 0 ? params.selected_stride : params.topk;
  for (int i = tid; i < n_sel; i += kThreads) {
    int32_t code;
    if (params.selected != nullptr) {
      code = params.selected[static_cast<int64_t>(row) * sel_stride + i];
    } else {
      const int64_t slot = ca_slot(params.compressed, req, i);
      code = slot >= 0 ? static_cast<int32_t>(slot) : -1;
    }
    sMap[i] = code;
  }
  if (n_win > 0) {
    const int end = static_cast<int>(params.positions[row]);
    const int begin = end - params.window_size + 1;
    for (int j = tid; j < n_win; j += kThreads) {
      const int p = begin + j;
      int32_t code = -1;
      if (p >= 0) {
        const int64_t slot = ca_slot(params.window, req, p);
        if (slot >= 0) code = (1 << 30) | static_cast<int32_t>(slot);
      }
      sMap[n_sel + j] = code;
    }
  }

  const float scale = params.mod.sm_scale != 0.0f ? params.mod.sm_scale
                                                   : rsqrtf(static_cast<float>(dim));
  const int total = n_sel + n_win;
  float acc[kLaneVals];
  float run_m = sp_neg_inf();
  float run_d = 0.0f;
#pragma unroll
  for (int u = 0; u < kLaneVals; ++u) acc[u] = 0.0f;
  __syncthreads();

  // One key per warp per iteration: the whole record is read once, the QK dot
  // and the P*V accumulation use the same registers, and no block-wide barrier
  // sits inside the loop.
  for (int i = warp; i < total; i += kWarps) {
    const int32_t code = sMap[i];
    float v[kLaneVals];
    float dot = 0.0f;
#pragma unroll
    for (int u = 0; u < kLaneVals; ++u) v[u] = 0.0f;
    if (code < 0) {
      dot = sp_neg_inf();
    } else {
      const bool win = (code & (1 << 30)) != 0;
      const int64_t slot = static_cast<int64_t>(code & ~(1 << 30));
      const kv::PagedKVDesc& cache = win ? params.window : params.compressed;
      const char* rec = sp_row_base<DTypeKV>(cache, slot);
      for (int s = 0; s < kCAttnSteps; ++s) {
        const int d0 = (s * 32 + lane) * kVec;
        if (d0 + kVec <= dim) {
          ca_read8<DTypeKV>(cache, rec, slot, d0, v + s * kVec);
#pragma unroll
          for (int u = 0; u < kVec; ++u) {
            dot = __fmaf_rn(qreg[s * kVec + u], v[s * kVec + u], dot);
          }
        } else if (d0 < dim) {
          const int end = (d0 + kVec) < dim ? (d0 + kVec) : dim;
          for (int d = d0; d < end; ++d) {
            const float x = sp_latent<DTypeKV>(cache, slot, d, dim);
            v[s * kVec + (d - d0)] = x;
            dot = __fmaf_rn(qreg[s * kVec + (d - d0)], x, dot);
          }
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      dot *= scale;
    }
    // Online softmax: one weight per key, one rescale of this lane's dims.
    const float m_new = fmaxf(run_m, dot);
    const float corr = run_m > sp_neg_inf() ? __expf(run_m - m_new) : 0.0f;
    const float p = dot > sp_neg_inf() ? __expf(dot - m_new) : 0.0f;
    run_d = run_d * corr + p;
    run_m = m_new;
    if (corr < 1.0f) {   // the running max only moves when a key beats it
#pragma unroll
      for (int u = 0; u < kLaneVals; ++u) acc[u] *= corr;
    }
#pragma unroll
    for (int u = 0; u < kLaneVals; ++u) acc[u] = __fmaf_rn(p, v[u], acc[u]);
  }

  // Stage every warp's partial state, then merge the 8 partial softmaxes (plus
  // the sink) under one max/denominator.
#pragma unroll
  for (int s = 0; s < kCAttnSteps; ++s) {
    const int d0 = (s * 32 + lane) * kVec;
    for (int u = 0; u < kVec; ++u) {
      const int d = d0 + u;
      if (d < vdim) sMerge[warp * vdim + d] = acc[s * kVec + u];
    }
  }
  if (lane == 0) {
    sState[warp * 2 + 0] = run_m;
    sState[warp * 2 + 1] = run_d;
  }
  __syncthreads();

  const float sink = params.mod.sink_logits != nullptr ? params.mod.sink_logits[head] : sp_neg_inf();
  float gmax = sp_neg_inf();
  for (int w = 0; w < kWarps; ++w) gmax = fmaxf(gmax, sState[w * 2 + 0]);
  const float m_final = fmaxf(gmax, sink);
  if (tid < kWarps) {
    const float wm = sState[tid * 2 + 0];
    const float wc = wm > sp_neg_inf() ? __expf(wm - m_final) : 0.0f;
    sWeight[tid] = wc;
    sPart[tid] = sState[tid * 2 + 1] * wc;
  }
  __syncthreads();
  float den = sink > sp_neg_inf() ? __expf(sink - m_final) : 0.0f;
  for (int w = 0; w < kWarps; ++w) den += sPart[w];
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  DTypeO* dbase = reinterpret_cast<DTypeO*>(params.o);
  DTypeO* dst = dbase + static_cast<int64_t>(row) * params.o_stride_n +
                static_cast<int64_t>(head) * params.o_stride_h;
  for (int d = tid; d < vdim; d += kThreads) {
    float a = 0.0f;
    for (int w = 0; w < kWarps; ++w) a += sMerge[w * vdim + d] * sWeight[w];
    a *= inv;
    if constexpr (std::is_same_v<DTypeO, __nv_bfloat16>) dst[d] = __float2bfloat16_rn(a);
    else if constexpr (std::is_same_v<DTypeO, __half>) dst[d] = __float2half_rn(a);
    else dst[d] = a;
  }
  if (tid == 0 && params.lse != nullptr) {
    params.lse[static_cast<int64_t>(row) * params.num_q_heads + head] =
        den > 0.0f ? (m_final + __logf(den)) : sp_neg_inf();
  }
}

Status compressed_attention(const CompressedAttentionParams& params, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "compressed_attention: CUDA only");
  }
  if (params.q == nullptr || params.o == nullptr || params.num_tokens < 0 || params.num_q_heads <= 0 ||
      params.entry_dim <= 0 || params.value_dim <= 0) {
    return detail::set_error(Status::kInvalidArgument, "compressed_attention: bad argument");
  }
  // `topk` is the capacity of the compressed branch in both modes: the number of
  // selected entries (CSA) or the upper bound of the dense range (HCA, where the
  // per-request count is the device array num_compressed). It sizes the shared
  // key map, so an HCA caller passes its max num_compressed here.
  if (params.selected == nullptr && params.num_compressed != nullptr && params.topk <= 0) {
    return detail::set_error(
        Status::kInvalidArgument,
        "compressed_attention: HCA needs topk >= max num_compressed");
  }
  const int cap = params.topk + (params.window_size > 0 ? params.window_size : 0);
  if (cap <= 0 || cap > 8192) {
    return detail::set_error(Status::kUnsupportedShape, "compressed_attention: cap %d", cap);
  }
  if (params.entry_dim > kCAttnMaxDim) {
    return detail::set_error(Status::kUnsupportedShape, "compressed_attention: entry_dim %d",
                             params.entry_dim);
  }
  if (params.value_dim > params.entry_dim) {
    return detail::set_error(Status::kInvalidArgument, "compressed_attention: value_dim %d",
                             params.value_dim);
  }
  const bool tc = ca_tc_eligible(params);
  const size_t smem = tc ? (ca_tc_tile(params) == kCaTcTileWide
                                ? sp_tc_smem_bytes_t<kCaTcDim, kCaTcTileWide>()
                                : sp_tc_smem_bytes_t<kCaTcDim, kCaTcTile>())
                         : ca_smem_bytes(cap, params.value_dim);
  const dim3 grid(static_cast<unsigned>(params.num_tokens),
                  tc ? static_cast<unsigned>((params.num_q_heads + 15) / 16)
                     : static_cast<unsigned>(params.num_q_heads));
  const dim3 block(tc ? tkThreads : kCAttnThreads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  cudaError_t err = cudaSuccess;
#define KOPS_CA_LAUNCH(QT, KT, OT)                                                              \
  do {                                                                                          \
    if (smem > 48u * 1024u) {                                                                   \
      (void)cudaFuncSetAttribute(                                                               \
          reinterpret_cast<const void*>(&compressed_attention_kernel<16, 512, 512, QT, KT, OT,   \
                                                                     Sm80>),                    \
          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));                 \
    }                                                                                           \
    compressed_attention_kernel<16, 512, 512, QT, KT, OT, Sm80><<<grid, block, smem, stream>>>( \
        params);                                                                                \
    err = cudaGetLastError();                                                                   \
  } while (0)
  if (params.q_dtype == DType::kBFloat16 && params.o_dtype == DType::kBFloat16) {
    KOPS_CA_LAUNCH(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
  } else if (params.q_dtype == DType::kFloat16 && params.o_dtype == DType::kFloat16) {
    KOPS_CA_LAUNCH(__half, __half, __half);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "compressed_attention: q/o dtype");
  }
#undef KOPS_CA_LAUNCH
  if (err != cudaSuccess) return launch_error("compressed_attention launch", err);
  return Status::kSuccess;
}

}  // namespace kops::attn
