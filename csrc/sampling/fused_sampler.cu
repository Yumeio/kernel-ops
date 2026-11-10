// =============================================================================
// kernel-ops | csrc/sampling/fused_sampler.cu
// Fused sampler: greedy, unfiltered full-vocab categorical and top-k / min-p / top-p draws
// from logits, one launch, no [B, V] fp32 probability row.
//
//   greedy          one pass, one CTA per row
//   categorical     pass 1 row max (and +inf count); pass 2 fixed-point masses
//                   in token order + per-segment totals in shared memory; one
//                   warp then re-reads the single segment that holds the target
//
// Row layout: the row is cut into chunks of VEC tokens (one 16-byte load); warp
// w owns a contiguous chunk segment, so "warp, then iteration, then lane, then
// element" is token-index order and the inverse CDF is a function of the row
// alone (not of the block size or a later split count).
//   filtered        top-k (vLLM tie group) / min-p / top-p: count histogram + row
//                   max, then a Gumbel race over the bins kept for sure with an exact
//                   refine of the boundary bins (sorted in shared memory); boundary
//                   bins beyond the buffer take an exact multi-pass radix path
// Processors: repetition (packed bitmask),
// frequency/presence (token counts), optional bitmask write-back of the draw.
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

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
/// Per-warp segment records of the categorical pass (8 KiB of shared memory). A warp with more
/// iterations than records groups G consecutive iterations per record; the final re-read covers
/// one record, i.e. G iterations of 32 chunks.
constexpr int kMaxRecords = 128;

constexpr int kBins = 4096;  // linear bins of width 1/32 on [-64, 64), clamped at both ends
constexpr int kMassBins = kBins / 2;  // top-p-only mass bins of width 1/16; overlays hist[]
constexpr int kCand = 2048;  // boundary-bin candidates in shared memory: exactly the histogram's
                             // 16 KiB, which keeps 5 CTAs per SM (4096 halved the throughput)
constexpr uint32_t kPlanFiltered = 1u;
constexpr uint32_t kPlanToppOnly = 2u;  // p-only rows take the mass-histogram path

// ---- split-row (Path B) layout -------------------------------------------------------------------
constexpr int kMaxSplits = 16;
constexpr int kExWords = 4;                          // words each CTA contributes per exchange
constexpr size_t kExArrayBytes = size_t{kBins} * 4;  // per-CTA array slot (u32[4096] / u64[2048])
constexpr int kMinShard = 4096;                      // tokens per CTA below which a split stops paying

/// Bytes of one row's exchange area for S CTAs: two alternating sides of S x 4 words and of S
/// array slots, then the candidate buffer and its count. A function of S only, so the offsets are
/// stable across graph replays.
__host__ __device__ constexpr size_t split_words_bytes(int S) { return size_t{2} * S * kExWords * 8; }
__host__ __device__ constexpr size_t split_arrays_bytes(int S) { return size_t{2} * S * kExArrayBytes; }
__host__ __device__ constexpr size_t split_row_bytes(int S) {
  return (split_words_bytes(S) + split_arrays_bytes(S) + size_t{kCand} * 8 + 16 + 255) / 256 * 256;
}

__device__ __forceinline__ float inf_pos() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float inf_neg() { return __int_as_float(0xff800000); }

/// round(p * 2^40) for p in [0, 1], bit-identical to __double2ull_rn(double(p) * 2^40) but with
/// integer instructions only (FP64 runs at 1/64 rate on consumer GPUs).
__device__ __forceinline__ uint64_t fixed_mass40(float p) {
  const uint32_t bits = __float_as_uint(p);
  const int e = static_cast<int>(bits >> 23);  // p >= 0: no sign bit
  if (e == 0) return 0ull;                     // zero or subnormal (< 2^-126): rounds to 0
  const uint32_t mant = (bits & 0x7fffffu) | 0x800000u;
  const int sh = e - 110;                      // value = mant * 2^(e - 127 - 23 + 40)
  if (sh >= 0) return static_cast<uint64_t>(mant) << sh;
  const int s = -sh;
  if (s > 24) return 0ull;                     // value < 0.5
  const uint32_t q = mant >> s;
  const uint32_t rem = mant & ((1u << s) - 1u);
  const uint32_t half = 1u << (s - 1);
  return static_cast<uint64_t>(q + ((rem > half || (rem == half && (q & 1u))) ? 1u : 0u));
}

__device__ __forceinline__ uint64_t warp_sum_u64(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ uint64_t warp_incl_scan_u64(uint64_t v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint64_t y = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += y;
  }
  return v;
}

/// 64 random bits from the same Philox counter as uniform_from_counter(draw_index = 0).
__device__ __forceinline__ uint64_t row_random_u64(const RngParams& rng, int row) {
  const uint64_t seed = rng.per_request_seed != nullptr ? rng.per_request_seed[row] : rng.global_seed;
  const uint64_t position =
      rng.per_request_offset != nullptr ? static_cast<uint64_t>(rng.per_request_offset[row]) : 0ull;
  const uint64_t request = static_cast<uint64_t>(row);
  const uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const uint4 ctr = make_uint4(static_cast<uint32_t>(request),
                               static_cast<uint32_t>(request >> 32) ^ rng.stream_id,
                               static_cast<uint32_t>(position), static_cast<uint32_t>(position >> 32));
  const uint4 r = philox4x32_10(key, ctr);
  return (static_cast<uint64_t>(r.x) << 32) | r.y;
}

/// Processor state of one row, resolved once per CTA.
struct RowProc {
  uint8_t* mask;          // penalty bitmask row or null
  const int32_t* counts;  // token-count row or null
  float rep;              // repetition penalty (applied when the mask bit is set and rep > 0)
  float freq;
  float pres;

  __device__ __forceinline__ float apply(float x, int token) const {
    if (isnan(x)) x = inf_neg();
    if (mask != nullptr && rep > 0.0f && ((mask[token >> 3] >> (token & 7)) & 1u)) {
      x = x > 0.0f ? x / rep : x * rep;
    }
    if (counts != nullptr) {
      const int32_t c = counts[token];
      x -= freq * static_cast<float>(c) + (c > 0 ? pres : 0.0f);
    }
    return x;
  }
};

__device__ __forceinline__ RowProc row_proc(const FusedSamplingParams& p, int row, int vocab,
                                            int penalty_row_bytes) {
  const int64_t slot = p.slot_id != nullptr ? p.slot_id[row] : row;
  RowProc r;
  r.mask = p.penalty_mask != nullptr ? p.penalty_mask + slot * penalty_row_bytes : nullptr;
  r.counts = p.token_counts != nullptr ? p.token_counts + slot * vocab : nullptr;
  r.rep = p.repetition_penalty != nullptr ? p.repetition_penalty[row] : p.default_repetition_penalty;
  r.freq = p.frequency_penalty != nullptr ? p.frequency_penalty[row] : 0.0f;
  r.pres = p.presence_penalty != nullptr ? p.presence_penalty[row] : 0.0f;
  return r;
}

/// Set the drawn token's bit (several rows may share a mask byte, hence the word atomic).
__device__ __forceinline__ void mark_token(uint8_t* mask, int token) {
  uint8_t* byte = mask + (token >> 3);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(byte);
  auto* word = reinterpret_cast<unsigned int*>(addr & ~uintptr_t{3});
  atomicOr(word, 1u << (8u * static_cast<unsigned>(addr & 3u) + static_cast<unsigned>(token & 7)));
}

template <typename in_t, int VEC>
__device__ __forceinline__ void load_chunk(const in_t* __restrict__ row, int chunk, float (&x)[VEC]) {
  if constexpr (VEC == 1) {
    x[0] = dev::to_f<in_t>(row[chunk]);
  } else {
    const Vec<in_t, VEC> v = load_vec<in_t, VEC>(row + static_cast<int64_t>(chunk) * VEC);
#pragma unroll
    for (int i = 0; i < VEC; ++i) x[i] = dev::to_f<in_t>(v[i]);
  }
}

/// Chunk segment of a warp: [beg, end) with a 32-multiple length so every iteration is warp-uniform.
struct Segment {
  int beg;
  int end;
  int iters;
};

__device__ __forceinline__ Segment warp_segment(int beg, int end, int warp) {
  const int chunks = end - beg;
  const int per = ((chunks + kWarps - 1) / kWarps + 31) / 32 * 32;
  Segment s;
  s.beg = min(end, beg + warp * per);
  s.end = min(end, s.beg + per);
  s.iters = (s.end - s.beg + 31) / 32;
  return s;
}

/// (value, lowest index) max across the block; thread 0 gets the winner.
__device__ __forceinline__ void block_argmax0(float& value, int32_t& index, float* sv, int32_t* si) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  kops::warp_argmax(value, index);
  if (lane == 0) {
    sv[warp] = value;
    si[warp] = index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kWarps; ++w) {
      if (sv[w] > value || (sv[w] == value && si[w] < index)) {
        value = sv[w];
        index = si[w];
      }
    }
  }
}


/// One row as seen by one CTA. Path A (SPLIT = false): the CTA owns the whole row and every
/// exchange below is a no-op. Path B: CTA s of S owns the s-th contiguous shard of the row's
/// chunks (so "CTA, warp, iteration, lane, element" is still token order); row-level values are
/// combined through `area` after a barrier on `bar`, in CTA order, so they equal Path A exactly
/// (integer sums, max, merged histograms). Exchanges alternate between two sides: a side is
/// rewritten two exchanges later, after every CTA has passed the barrier in between.
template <bool SPLIT>
struct RowCtx {
  int row;
  int s;
  int S;
  int chunk_beg;
  int chunk_end;
  uint8_t* area;
  unsigned long long* bar;
  int32_t* out;
  uint8_t* mark_mask;  // non-null: set the drawn token's bit
  int side_w;
  int side_a;

  __device__ __forceinline__ bool leader() const { return !SPLIT || s == 0; }
  __device__ __forceinline__ Segment seg() const {
    return warp_segment(chunk_beg, chunk_end, static_cast<int>(threadIdx.x >> 5));
  }
  /// Write the row's token (one thread of the deciding CTA).
  __device__ __forceinline__ void emit(int token) const {
    out[row] = token;
    if (mark_mask != nullptr && token >= 0) mark_token(mark_mask, token);
  }
  __device__ __forceinline__ unsigned long long* cand() const {
    return reinterpret_cast<unsigned long long*>(area + split_words_bytes(S) + split_arrays_bytes(S));
  }
  __device__ __forceinline__ unsigned int* cand_count() const {
    return reinterpret_cast<unsigned int*>(area + split_words_bytes(S) + split_arrays_bytes(S) +
                                           size_t{kCand} * 8);
  }
  /// Barrier of the row's CTAs: every arrival adds 16 / S, so the counter stays a multiple of 16
  /// between launches whatever S is and never needs a reset.
  __device__ __forceinline__ void barrier() const {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned long long old = atomicAdd(bar, 16ull / static_cast<unsigned long long>(S));
      const unsigned long long target = (old / 16ull + 1ull) * 16ull;
      while (*reinterpret_cast<volatile unsigned long long*>(bar) < target) __nanosleep(32);
      __threadfence();
    }
    __syncthreads();
  }
  /// Thread 0's four words of every CTA of the row, CTA-major, into tab (shared memory).
  __device__ __forceinline__ void exch(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3,
                                       uint64_t (*tab)[kExWords]) {
    __syncthreads();  // the previous table may still be read
    if constexpr (!SPLIT) {
      if (threadIdx.x == 0) {
        tab[0][0] = w0;
        tab[0][1] = w1;
        tab[0][2] = w2;
        tab[0][3] = w3;
      }
      __syncthreads();
    } else {
      unsigned long long* words = reinterpret_cast<unsigned long long*>(area) + side_w * S * kExWords;
      if (threadIdx.x == 0) {
        __stcg(words + s * kExWords + 0, static_cast<unsigned long long>(w0));
        __stcg(words + s * kExWords + 1, static_cast<unsigned long long>(w1));
        __stcg(words + s * kExWords + 2, static_cast<unsigned long long>(w2));
        __stcg(words + s * kExWords + 3, static_cast<unsigned long long>(w3));
      }
      barrier();
      for (int i = threadIdx.x; i < S * kExWords; i += kThreads) tab[i / kExWords][i % kExWords] = __ldcg(words + i);
      __syncthreads();
      side_w ^= 1;
    }
  }
  /// Element-wise sum of a shared array over the row's CTAs, in place, in CTA order. @pre the
  /// array is complete (a __syncthreads after its last update).
  template <typename T>
  __device__ __forceinline__ void merge(T* a, int n) {
    if constexpr (SPLIT) {
      using U = typename std::conditional<sizeof(T) == 8, unsigned long long, unsigned int>::type;
      constexpr size_t kSlot = kExArrayBytes / sizeof(U);
      U* slots = reinterpret_cast<U*>(area + split_words_bytes(S) + size_t(side_a) * S * kExArrayBytes);
      for (int i = threadIdx.x; i < n; i += kThreads) __stcg(slots + size_t(s) * kSlot + i, static_cast<U>(a[i]));
      barrier();
      for (int i = threadIdx.x; i < n; i += kThreads) {
        U acc = 0;
        for (int j = 0; j < S; ++j) acc += __ldcg(slots + size_t(j) * kSlot + i);
        a[i] = static_cast<T>(acc);
      }
      __syncthreads();
      side_a ^= 1;
    }
  }
};

template <bool SPLIT>
__device__ __forceinline__ RowCtx<SPLIT> make_row_ctx(int row, int chunks, int32_t* out,
                                                     uint8_t* mark_mask, const FusedSplitDevice& sp) {
  RowCtx<SPLIT> c;
  c.row = row;
  c.out = out;
  c.mark_mask = mark_mask;
  c.side_w = 0;
  c.side_a = 0;
  if constexpr (SPLIT) {
    c.S = sp.splits;
    c.s = static_cast<int>(blockIdx.x);
    const int per = (chunks + c.S - 1) / c.S;
    c.chunk_beg = min(chunks, c.s * per);
    c.chunk_end = min(chunks, c.chunk_beg + per);
    c.area = sp.scratch + static_cast<size_t>(row) * split_row_bytes(c.S);
    c.bar = sp.barriers + row;
  } else {
    c.S = 1;
    c.s = 0;
    c.chunk_beg = 0;
    c.chunk_end = chunks;
    c.area = nullptr;
    c.bar = nullptr;
  }
  return c;
}

/// Combine the (score, lowest token) winners of every thread and CTA of the row; the leader writes
/// the token (-1 when nothing was admissible).
template <bool SPLIT>
__device__ __forceinline__ void race_finish(RowCtx<SPLIT>& c, float best, int32_t best_token,
                                            float* sv, int32_t* si, uint64_t (*tab)[kExWords]) {
  block_argmax0(best, best_token, sv, si);
  c.exch(__float_as_uint(best), static_cast<uint32_t>(best_token), 0, 0, tab);
  if (c.leader() && threadIdx.x == 0) {
    float b = inf_neg();
    int32_t t = INT32_MAX;
    for (int j = 0; j < c.S; ++j) {
      const float v = __uint_as_float(static_cast<uint32_t>(tab[j][0]));
      const int32_t k = static_cast<int32_t>(static_cast<uint32_t>(tab[j][1]));
      if (v > b || (v == b && k < t)) {
        b = v;
        t = k;
      }
    }
    c.emit(b == inf_neg() ? -1 : t);
  }
}

template <typename in_t, int VEC, bool SPLIT>
__device__ __forceinline__ void greedy_row(RowCtx<SPLIT>& c, const in_t* __restrict__ row,
                                           const RowProc& proc, float* sv, int32_t* si,
                                           uint64_t (*tab)[kExWords]) {
  const int lane = threadIdx.x & 31;
  const Segment seg = c.seg();
  float best = inf_neg();
  int32_t best_index = INT32_MAX;
  for (int ch = seg.beg + lane; ch < seg.end; ch += 32) {
    float x[VEC];
    load_chunk<in_t, VEC>(row, ch, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int token = ch * VEC + i;
      const float y = proc.apply(x[i], token);
      // Tokens of one thread rise, so a strict > keeps the lowest index of a tie.
      if (y > best) {
        best = y;
        best_index = token;
      }
    }
  }
  race_finish(c, best, best_index, sv, si, tab);
}

/// Weight of one processed, temperature-scaled logit in the categorical draw: fixed-point
/// exp(y - m), or a unit weight per +inf entry when the row has any (uniform over them).
__device__ __forceinline__ uint64_t token_weight(float y, float m, bool inf_row) {
  if (inf_row) return y == inf_pos() ? 1ull : 0ull;
  return y == inf_neg() ? 0ull : fixed_mass40(__expf(y - m));
}

template <typename in_t, int VEC>
__device__ __forceinline__ uint64_t chunk_weight(const in_t* __restrict__ row, int c, const RowProc& proc,
                                                 float inv_t, float m, bool inf_row) {
  float x[VEC];
  load_chunk<in_t, VEC>(row, c, x);
  uint64_t w = 0;
#pragma unroll
  for (int i = 0; i < VEC; ++i) w += token_weight(proc.apply(x[i], c * VEC + i) * inv_t, m, inf_row);
  return w;
}

template <typename in_t, int VEC, bool SPLIT>
__device__ void categorical_row(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, const RowProc& proc,
                                float inv_t, const RngParams& rng, float* sv, int32_t* si, int* sinf,
                                uint64_t (*rec)[kMaxRecords], uint64_t* sh64, int* shi,
                                uint64_t (*tab)[kExWords]) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const Segment seg = rc.seg();

  // ---- pass 1: max of the finite values and count of +inf ----
  float mx = inf_neg();
  int infs = 0;
  for (int c = seg.beg + lane; c < seg.end; c += 32) {
    float x[VEC];
    load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float y = proc.apply(x[i], c * VEC + i) * inv_t;
      if (y == inf_pos()) {
        ++infs;
      } else {
        mx = fmaxf(mx, y);
      }
    }
  }
  const float m_cta = kops::block_reduce_max<float, kWarps>(mx, sv);
  const int inf_cta = kops::block_reduce_sum<int, kWarps>(infs, sinf);
  rc.exch(__float_as_uint(m_cta), static_cast<uint32_t>(inf_cta), 0, 0, tab);
  float m = inf_neg();
  int inf_count = 0;
  for (int j = 0; j < rc.S; ++j) {
    m = fmaxf(m, __uint_as_float(static_cast<uint32_t>(tab[j][0])));
    inf_count += static_cast<int>(tab[j][1]);
  }
  const bool inf_row = inf_count > 0;
  if (!inf_row && m == inf_neg()) {  // no admissible token (all -inf / NaN)
    if (rc.leader() && tid == 0) rc.emit(-1);
    return;
  }

  // ---- pass 2: per-record fixed-point mass in token order ----
  const int max_iters = warp_segment(rc.chunk_beg, rc.chunk_end, 0).iters;  // warp 0 is longest
  const int group = max(1, (max_iters + kMaxRecords - 1) / kMaxRecords);
  const int nrec = (max_iters + group - 1) / group;
  for (int i = tid; i < kWarps * kMaxRecords; i += kThreads) rec[i / kMaxRecords][i % kMaxRecords] = 0ull;
  __syncthreads();
  for (int it = 0; it < seg.iters; ++it) {
    const int c = seg.beg + it * 32 + lane;
    uint64_t w = c < seg.end ? chunk_weight<in_t, VEC>(row, c, proc, inv_t, m, inf_row) : 0ull;
    w = warp_sum_u64(w);
    if (lane == 0) rec[warp][it / group] += w;
  }
  __syncthreads();

  // ---- the CTA owning the target: CTA totals in token order ----
  if (warp == 0) {
    uint64_t wt = 0;
    if (lane < kWarps) {
      for (int r = 0; r < nrec; ++r) wt += rec[lane][r];
    }
    wt = warp_sum_u64(wt);
    if (lane == 0) sh64[0] = wt;
  }
  __syncthreads();
  const uint64_t cta_total = sh64[0];
  rc.exch(cta_total, 0, 0, 0, tab);
  uint64_t before = 0, total = 0;
  for (int j = 0; j < rc.S; ++j) {
    if (j < rc.s) before += tab[j][0];
    total += tab[j][0];
  }
  if (total == 0) {
    if (rc.leader() && tid == 0) rc.emit(-1);
    return;
  }
  const uint64_t row_target = __umul64hi(total, row_random_u64(rng, rc.row)) + 1ull;
  if (!(before < row_target && row_target <= before + cta_total)) return;  // another CTA's token
  const uint64_t target = row_target - before;

  // ---- locate the record that holds the target (warp 0) ----
  if (warp == 0) {
    uint64_t wt = 0;
    if (lane < kWarps) {
      for (int r = 0; r < nrec; ++r) wt += rec[lane][r];
    }
    const uint64_t incl = warp_incl_scan_u64(wt);
    // The owning warp: excl < target <= incl (empty warps never satisfy it).
    const unsigned own = __ballot_sync(0xffffffffu, lane < kWarps && incl - wt < target && target <= incl);
    const int w_sel = own != 0u ? __ffs(own) - 1 : -1;
    const uint64_t wb = w_sel >= 0 ? __shfl_sync(0xffffffffu, incl - wt, w_sel) : 0ull;
    // Records of that warp, four per lane in order.
    uint64_t part = 0;
    if (w_sel >= 0) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int r = lane * 4 + j;
        if (r < nrec) part += rec[w_sel][r];
      }
    }
    const uint64_t pincl = warp_incl_scan_u64(part) + wb;
    const bool hit = w_sel >= 0 && pincl - part < target && target <= pincl;
    const unsigned hits = __ballot_sync(0xffffffffu, hit);
    if (hit) {
      uint64_t run = pincl - part;
      int r = lane * 4;
      for (int j = 0; j < 4; ++j, ++r) {
        if (r >= nrec) break;
        const uint64_t v = rec[w_sel][r];
        if (target <= run + v) break;
        run += v;
      }
      sh64[0] = run;     // mass before record r
      shi[0] = r;        // record
    }
    if (lane == 0) {
      sh64[1] = target;
      shi[1] = hits != 0 ? w_sel : -1;
      shi[2] = -1;
    }
  }
  __syncthreads();

  // ---- re-read one record of the owning warp and pick the token ----
  if (warp == shi[1]) {
    const uint64_t tgt = sh64[1];
    uint64_t wb = sh64[0];
    const int it_end = min(seg.iters, (shi[0] + 1) * group);
    for (int it = shi[0] * group; it < it_end; ++it) {
      const int c = seg.beg + it * 32 + lane;
      const uint64_t w = c < seg.end ? chunk_weight<in_t, VEC>(row, c, proc, inv_t, m, inf_row) : 0ull;
      const uint64_t incl = warp_incl_scan_u64(w) + wb;
      const uint64_t tot = __shfl_sync(0xffffffffu, incl, 31);
      if (incl - w < tgt && tgt <= incl) {
        float x[VEC];
        load_chunk<in_t, VEC>(row, c, x);
        uint64_t run = incl - w;
        int token = c * VEC + VEC - 1;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          run += token_weight(proc.apply(x[i], c * VEC + i) * inv_t, m, inf_row);
          if (run >= tgt) {
            token = c * VEC + i;
            break;
          }
        }
        shi[2] = token;
      }
      if (tgt <= tot) break;
      wb = tot;
    }
  }
  __syncthreads();
  if (tid == 0) rc.emit(shi[1] >= 0 ? shi[2] : -1);
}

// ---- filtered path (top-k / min-p / top-p) ------------------------------------------------------


/// Bin of a processed logit (NaN already mapped to -inf). Non-decreasing in x.
__device__ __forceinline__ int logit_bin(float x) {
  const float f = (x + 64.0f) * 32.0f;
  return f <= 0.0f ? 0 : (f >= static_cast<float>(kBins - 1) ? kBins - 1 : static_cast<int>(f));
}

/// Order-preserving uint32 key of a non-NaN float, and its inverse.
__device__ __forceinline__ uint32_t order_key(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_value(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

/// Gumbel noise of one token for the race: Philox counter of uniform_from_counter with
/// draw_index = token, 64 random bits, and the lower half of the unit interval resolved through
/// log1p so the tail reaches -log(2^-64) instead of the 2^-24 resolution of a float uniform.
__device__ __forceinline__ float race_gumbel(const RngParams& rng, int row, int token) {
  const uint64_t seed = rng.per_request_seed != nullptr ? rng.per_request_seed[row] : rng.global_seed;
  const uint64_t position =
      rng.per_request_offset != nullptr ? static_cast<uint64_t>(rng.per_request_offset[row]) : 0ull;
  const uint64_t request = static_cast<uint64_t>(row);
  const uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const uint4 ctr = make_uint4(static_cast<uint32_t>(request),
                               static_cast<uint32_t>(request >> 32) ^ rng.stream_id,
                               static_cast<uint32_t>(position),
                               static_cast<uint32_t>(position >> 32) ^ static_cast<uint32_t>(token));
  const uint4 r = philox4x32_10(key, ctr);
  float e;  // Exponential(1) = -log(u), u = (r.x:r.y) / 2^64
  if ((r.x & 0x80000000u) == 0u) {
    const float u = fmaf(__uint2float_rn(r.y), 0x1p-64f, __uint2float_rn(r.x) * 0x1p-32f) + 0x1p-65f;
    e = -logf(u);  // u in (0, 0.5]
  } else {
    const float s = fmaf(__uint2float_rn(~r.y), 0x1p-64f, __uint2float_rn(~r.x) * 0x1p-32f) + 0x1p-65f;
    e = -log1pf(-s);  // s = 1 - u in (0, 0.5]
  }
  return -logf(e);
}

__device__ __forceinline__ void race_take(float score, int token, float& best, int32_t& best_token) {
  if (score > best || (score == best && token < best_token)) {
    best = score;
    best_token = token;
  }
}

/// Inclusive scan of one value per thread in thread order; sw holds kWarps slots and may be
/// reused right after the call. The association order is fixed, so float results are reproducible.
template <typename T>
__device__ __forceinline__ T block_incl_scan(T v, T* sw) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const T y = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += y;
  }
  if (lane == 31) sw[warp] = v;
  __syncthreads();
  T off = T{};
  for (int w = 0; w < warp; ++w) off += sw[w];
  __syncthreads();
  return v + off;
}

struct OpSum {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};
struct OpMin {
  __device__ __forceinline__ int operator()(int a, int b) const { return min(a, b); }
};
struct OpMax {
  __device__ __forceinline__ int operator()(int a, int b) const { return max(a, b); }
};

/// Block reduction in a fixed order (lane 0 of each warp, then warps 0..7); every thread gets the
/// result and sw may be reused right after the call.
template <typename T, typename Op>
__device__ __forceinline__ T block_all(T v, T* sw, Op op) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
  if (lane == 0) sw[warp] = v;
  __syncthreads();
  T r = sw[0];
  for (int w = 1; w < kWarps; ++w) r = op(r, sw[w]);
  __syncthreads();
  return r;
}

/// Fixed-point mass of a processed, scaled logit relative to the row max (same as the categorical
/// path, so both paths weigh a token identically).
__device__ __forceinline__ uint64_t mass40(float y, float m) { return fixed_mass40(__expf(y - m)); }

/// exp(t) on t in [-1/16, 0]: the mass histogram's per-element weight relative to its bin's
/// upper edge. The cubic is accurate to ~2e-7 there, so the fixed-point bin masses keep the
/// boundary decision exact enough to be refined on the candidates.
__device__ __forceinline__ float exp_small(float t) {
  return fmaf(t, fmaf(t, fmaf(t, 1.0f / 6.0f, 0.5f), 1.0f), 1.0f);
}

/// Candidate slot: key in the high word, ~token in the low word, so a descending sort orders by
/// value descending, then token ascending. 0 is an empty slot (every real key is > 0).
__device__ __forceinline__ uint64_t pack_cand(uint32_t key, int token) {
  return (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(~static_cast<uint32_t>(token));
}
__device__ __forceinline__ uint32_t cand_key(uint64_t c) { return static_cast<uint32_t>(c >> 32); }
__device__ __forceinline__ int cand_token(uint64_t c) {
  return static_cast<int>(~static_cast<uint32_t>(c & 0xffffffffull));
}

/// need-th largest key (1-based, with multiplicity) among the candidates of one bin: 4 x 8-bit
/// radix passes over shared memory.
__device__ uint32_t cand_kth_key(const uint64_t* cand, int m, int bin, int need, int* hist, int* sh) {
  uint32_t prefix = 0;
  int rank = need;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int b = threadIdx.x; b < 256; b += kThreads) hist[b] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += kThreads) {
      const uint32_t k = cand_key(cand[i]);
      if (logit_bin(key_value(k)) != bin) continue;
      if (pass > 0 && (k >> (shift + 8)) != (prefix >> (shift + 8))) continue;
      atomicAdd(&hist[(k >> shift) & 255u], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int run = 0, b = 255;
      for (; b > 0; --b) {
        if (run + hist[b] >= rank) break;
        run += hist[b];
      }
      sh[0] = b;
      sh[1] = rank - run;
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(sh[0]) << shift;
    rank = sh[1];
    __syncthreads();
  }
  return prefix;
}

/// Same selection over the processed row itself (fallback when the boundary bins overflow).
template <typename in_t, int VEC, bool SPLIT>
__device__ uint32_t row_kth_key(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, const RowProc& proc,
                                int k, int* hist, int* sh) {
  const int lane = threadIdx.x & 31;
  const Segment seg = rc.seg();
  uint32_t prefix = 0;
  int rank = k;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int b = threadIdx.x; b < 256; b += kThreads) hist[b] = 0;
    __syncthreads();
    int prev = 0, run = 0;
    for (int c = seg.beg + lane; c < seg.end; c += 32) {
      float x[VEC];
      load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const uint32_t key = order_key(proc.apply(x[i], c * VEC + i));
        if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) continue;
        const int d = static_cast<int>((key >> shift) & 255u);
        if (d != prev) {
          if (run > 0) atomicAdd(&hist[prev], run);
          prev = d;
          run = 0;
        }
        ++run;
      }
    }
    if (run > 0) atomicAdd(&hist[prev], run);
    __syncthreads();
    rc.merge(hist, 256);
    if (threadIdx.x == 0) {
      int acc = 0, b = 255;
      for (; b > 0; --b) {
        if (acc + hist[b] >= rank) break;
        acc += hist[b];
      }
      sh[0] = b;
      sh[1] = rank - acc;
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(sh[0]) << shift;
    rank = sh[1];
    __syncthreads();
  }
  return prefix;
}

/// Descending bitonic sort of a[0, n), n a power of two.
__device__ void bitonic_desc(uint64_t* a, int n) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < n; i += kThreads) {
        const int l = i ^ j;
        if (l > i) {
          const uint64_t x = a[i], y = a[l];
          if ((i & k) == 0 ? x < y : x > y) {
            a[i] = y;
            a[l] = x;
          }
        }
      }
      __syncthreads();
    }
  }
}

constexpr int kQueue = 64;  // per-warp race queue slots (two warps' worth of entries)

/// Per-warp queue of Gumbel-race entries. Only a few tokens of each load survive a filter, so racing
/// in place would run the Philox + log code once per load slot with most lanes idle; entries are
/// compacted here and raced 32 at a time on a full warp. push()/drain() are warp-collective.
struct RaceQueue {
  int* token;
  float* y;
  int n;  // warp-uniform fill
  __device__ __forceinline__ void push(bool take, int tok, float val, const RngParams& rng, int row,
                                       float& best, int32_t& best_token) {
    const int lane = threadIdx.x & 31;
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (m == 0u) return;
    if (take) {
      const int pos = n + __popc(m & ((1u << lane) - 1u));
      token[pos] = tok;
      y[pos] = val;
    }
    n += __popc(m);
    if (n >= 32) {
      __syncwarp();
      race_take(y[lane] + race_gumbel(rng, row, token[lane]), token[lane], best, best_token);
      const int rest = n - 32;
      int t2 = 0;
      float y2 = 0.0f;
      if (lane < rest) {
        t2 = token[32 + lane];
        y2 = y[32 + lane];
      }
      __syncwarp();
      if (lane < rest) {
        token[lane] = t2;
        y[lane] = y2;
      }
      __syncwarp();
      n = rest;
    }
  }
  /// Push the kept elements of one chunk (bit i of `take` = element i); one vote when none is kept.
  template <int VEC>
  __device__ __forceinline__ void push_chunk(unsigned take, int first_token, const float (&val)[VEC],
                                             const RngParams& rng, int row, float& best,
                                             int32_t& best_token) {
    if (!__any_sync(0xffffffffu, take != 0u)) return;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      push(((take >> i) & 1u) != 0u, first_token + i, val[i], rng, row, best, best_token);
    }
  }
  __device__ __forceinline__ void drain(const RngParams& rng, int row, float& best,
                                        int32_t& best_token) {
    const int lane = threadIdx.x & 31;
    __syncwarp();
    if (lane < n) race_take(y[lane] + race_gumbel(rng, row, token[lane]), token[lane], best, best_token);
    __syncwarp();
    n = 0;
  }
};

/// Shared memory of the filtered kernel. The pass-1 histogram, the pass-2 candidates, the
/// categorical records and the fallback mass digits are never live at the same time.
struct FilteredSmem {
  union {
    uint32_t hist[kBins];
    uint64_t cand[kCand];
    uint64_t rec[kWarps][kMaxRecords];
    uint64_t mh[256];
  } u;
  int qtok[kWarps][kQueue];
  float qy[kWarps][kQueue];
  int radix[256];
  uint32_t scan[kWarps];
  uint64_t w64[kWarps];
  double wd[kWarps];
  float wf[kWarps];
  int wi[kWarps];
  float sv[kWarps];
  int32_t si[kWarps];
  int sinf[kWarps];
  int sfin[kWarps];
  uint64_t sh64[2];
  int shi[3];
  int sh[4];
  int cand_count;
  uint64_t tab[kMaxSplits][kExWords];
};

/// Filters of one row (T > 0). The base set is min-p and top-k (both value thresholds over the
/// whole row); top-p keeps the prefix of its domain, in (value desc, token asc) order, whose mass
/// before a token is below target = max(1, round(p * Z)). Domain = base (sequential, default) or
/// every finite token (joint); joint rows keep the intersection of the nucleus and the base.
struct RowFilter {
  float inv_t;
  float thr;  // min-p threshold on y = x * inv_t (-inf when off)
  float p;    // top-p in [0, 1) when use_p
  int k;
  bool use_k, use_mp, use_p, joint;
  __device__ __forceinline__ bool in_base(uint32_t key, float y, uint32_t kth) const {
    return y >= thr && (!use_k || key >= kth);
  }
};


/// Append one candidate: shared memory on Path A, the row's buffer in the workspace on Path B
/// (ordered before the leader's read by the next row barrier). Slots beyond kCand are dropped;
/// the callers' bounds keep them unused except where the count is checked afterwards.
template <bool SPLIT>
__device__ __forceinline__ void cand_push(const RowCtx<SPLIT>& c, FilteredSmem& sm, uint64_t cd) {
  if constexpr (SPLIT) {
    const unsigned pos = atomicAdd(c.cand_count(), 1u);
    if (pos < static_cast<unsigned>(kCand)) __stcg(c.cand() + pos, static_cast<unsigned long long>(cd));
  } else {
    const int pos = atomicAdd(&sm.cand_count, 1);
    if (pos < kCand) sm.u.cand[pos] = cd;
  }
}

/// Candidate count of the row after the barrier that follows the appends.
template <bool SPLIT>
__device__ __forceinline__ int cand_total(const RowCtx<SPLIT>& c, const FilteredSmem& sm) {
  if constexpr (SPLIT) {
    return static_cast<int>(__ldcg(c.cand_count()));
  } else {
    return sm.cand_count;
  }
}

/// Leader: bring the row's candidates into shared memory (Path B), returns their count.
template <bool SPLIT>
__device__ __forceinline__ int cand_gather(const RowCtx<SPLIT>& c, FilteredSmem& sm) {
  const int n = cand_total(c, sm);
  if constexpr (SPLIT) {
    const int m = min(n, kCand);
    for (int i = threadIdx.x; i < m; i += kThreads) sm.u.cand[i] = __ldcg(c.cand() + i);
  }
  __syncthreads();
  return n;
}

/// Token index of the r-th token (1-based, token order) whose processed key equals `key`.
template <typename in_t, int VEC, bool SPLIT>
__device__ int rth_equal_token(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, const RowProc& proc,
                               uint32_t key, uint32_t r, FilteredSmem& sm) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const Segment seg = rc.seg();
  uint32_t cnt = 0;
  for (int c = seg.beg + lane; c < seg.end; c += 32) {
    float x[VEC];
    load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) cnt += order_key(proc.apply(x[i], c * VEC + i)) == key;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  if (lane == 0) sm.scan[warp] = cnt;
  __syncthreads();
  uint32_t cta = 0;
  for (int w = 0; w < kWarps; ++w) cta += sm.scan[w];
  rc.exch(cta, 0, 0, 0, sm.tab);
  uint32_t before_cta = 0;
  for (int j = 0; j < rc.s; ++j) before_cta += static_cast<uint32_t>(sm.tab[j][0]);
  int token = INT32_MAX;
  if (before_cta < r && r <= before_cta + cta) {
    const uint32_t rl = r - before_cta;
    if (threadIdx.x == 0) {
      uint32_t run = 0;
      int w = 0;
      for (; w < kWarps - 1; ++w) {
        if (run + sm.scan[w] >= rl) break;
        run += sm.scan[w];
      }
      sm.shi[0] = w;
      sm.sh64[1] = run;
      sm.shi[1] = INT32_MAX;
    }
    __syncthreads();
    if (warp == sm.shi[0]) {
      uint32_t before = static_cast<uint32_t>(sm.sh64[1]);
      for (int it = 0; it < seg.iters; ++it) {
        const int c = seg.beg + it * 32 + lane;
        float x[VEC];
        uint32_t e = 0;
        if (c < seg.end) {
          load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
          for (int i = 0; i < VEC; ++i) e += order_key(proc.apply(x[i], c * VEC + i)) == key;
        }
        uint32_t incl = e;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lane >= o) incl += y;
        }
        const uint32_t tot = __shfl_sync(0xffffffffu, incl, 31);
        if (before + incl - e < rl && rl <= before + incl) {
          uint32_t run = before + incl - e;
          for (int i = 0; i < VEC; ++i) {
            if (order_key(proc.apply(x[i], c * VEC + i)) != key) continue;
            if (++run == rl) {
              sm.shi[1] = c * VEC + i;
              break;
            }
          }
        }
        if (rl <= before + tot) break;
        before += tot;
      }
    }
    __syncthreads();
    token = sm.shi[1];
  }
  rc.exch(static_cast<uint32_t>(token), 0, 0, 0, sm.tab);
  int res = INT32_MAX;
  for (int j = 0; j < rc.S; ++j) res = min(res, static_cast<int>(static_cast<uint32_t>(sm.tab[j][0])));
  return res;
}

/// Exact multi-pass path for rows whose boundary bins exceed the candidate buffer: radix top-k
/// threshold (4 passes), exact Z (1), 4 x 8-bit radix over fixed-point mass for the nucleus key,
/// the boundary token among equal keys by token order (1-2), then the race (1).
template <typename in_t, int VEC, bool SPLIT>
__device__ void filtered_fallback(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, const RowProc& proc,
                                  const RowFilter& f, float m, const RngParams& rng, FilteredSmem& sm,
                                  float& best, int32_t& best_token) {
  const int lane = threadIdx.x & 31;
  const Segment seg = rc.seg();
  const uint32_t kth = f.use_k ? row_kth_key<in_t, VEC, SPLIT>(rc, row, proc, f.k, sm.radix, sm.sh) : 0u;
  uint32_t pkey = 0;
  int pbound = INT32_MAX;  // nucleus: key > pkey, or key == pkey and token <= pbound
  bool empty = false;
  if (f.use_p) {
    uint64_t z = 0;
    for (int c = seg.beg + lane; c < seg.end; c += 32) {
      float x[VEC];
      load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float v = proc.apply(x[i], c * VEC + i);
        if (v == inf_neg()) continue;
        const float y = v * f.inv_t;
        if (f.joint || f.in_base(order_key(v), y, kth)) z += mass40(y, m);
      }
    }
    rc.exch(block_all(z, sm.w64, OpSum{}), 0, 0, 0, sm.tab);
    uint64_t total = 0;
    for (int j = 0; j < rc.S; ++j) total += sm.tab[j][0];
    empty = total == 0;  // min_p > 1: nothing is admissible
    if (!empty) {
      uint64_t rem = static_cast<uint64_t>(static_cast<double>(f.p) * static_cast<double>(total) + 0.5);
      if (rem < 1) rem = 1;
      uint32_t prefix = 0;
      for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = threadIdx.x; b < 256; b += kThreads) sm.u.mh[b] = 0ull;
        __syncthreads();
        int prev = 0;
        uint64_t acc = 0;
        for (int c = seg.beg + lane; c < seg.end; c += 32) {
          float x[VEC];
          load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const float v = proc.apply(x[i], c * VEC + i);
            if (v == inf_neg()) continue;
            const uint32_t key = order_key(v);
            if (pass > 0 && (key >> (shift + 8)) != (prefix >> (shift + 8))) continue;
            const float y = v * f.inv_t;
            if (!(f.joint || f.in_base(key, y, kth))) continue;
            const uint64_t w = mass40(y, m);
            const int d = static_cast<int>((key >> shift) & 255u);
            if (d != prev) {
              if (acc != 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&sm.u.mh[prev]),
                          static_cast<unsigned long long>(acc));
              }
              prev = d;
              acc = 0;
            }
            acc += w;
          }
        }
        if (acc != 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&sm.u.mh[prev]), static_cast<unsigned long long>(acc));
        }
        __syncthreads();
        rc.merge(sm.u.mh, 256);
        if (threadIdx.x == 0) {
          uint64_t run = 0;
          int b = 255;
          for (; b > 0; --b) {
            if (run + sm.u.mh[b] >= rem) break;
            run += sm.u.mh[b];
          }
          sm.sh[0] = b;
          sm.sh64[0] = rem - run;
        }
        __syncthreads();
        prefix |= static_cast<uint32_t>(sm.sh[0]) << shift;
        rem = sm.sh64[0];
        __syncthreads();
      }
      pkey = prefix;
      // Every token of key pkey has the same mass, so the boundary is the r-th of them.
      const uint64_t wk = mass40(key_value(pkey) * f.inv_t, m);
      const uint64_t r = wk > 0 ? (rem + wk - 1) / wk : 1;
      pbound = rth_equal_token<in_t, VEC, SPLIT>(rc, row, proc, pkey, static_cast<uint32_t>(r), sm);
    }
  }
  if (empty) return;
  RaceQueue q{sm.qtok[threadIdx.x >> 5], sm.qy[threadIdx.x >> 5], 0};
  for (int it = 0; it < seg.iters; ++it) {
    const int c = seg.beg + it * 32 + lane;
    const bool valid = c < seg.end;
    float x[VEC];
    if (valid) load_chunk<in_t, VEC>(row, c, x);
    unsigned take = 0u;
    float ys[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int token = c * VEC + i;
      const float v = valid ? proc.apply(x[i], token) : inf_neg();
      const uint32_t key = order_key(v);
      ys[i] = v * f.inv_t;
      if (v != inf_neg() && f.in_base(key, ys[i], kth) &&
          (!f.use_p || key > pkey || (key == pkey && token <= pbound))) {
        take |= 1u << i;
      }
    }
    q.push_chunk<VEC>(take, c * VEC, ys, rng, rc.row, best, best_token);
  }
  q.drain(rng, rc.row, best, best_token);
}

/// Top-p bin range from mass bounds after pass 1 (units: 1.0 = exp(0) = 2^40 fixed point). A token
/// of bin j has mass in [exp(y_lo - m), exp(y_hi - m)]; with the domain restricted per bin (the
/// top-k bin contributes at least `need` tokens, a bin cut by min-p at least 0) the mass before any
/// token of bin j lies in [S_L(j + 1), S_U(j)]. Bins whose upper sum stays below p * Z_L are in the
/// nucleus for sure, bins whose lower sum above them exceeds p * Z_U are out; the 1e-3 relative and
/// 1e-6 absolute margins cover fp32 summation, __expf and fixed-point rounding.
__device__ void topp_range(FilteredSmem& sm, const RowFilter& f, float m, int bin_k, int need,
                           int* lo, int* hi, int* n_range) {
  constexpr int kPer = kBins / kThreads;
  const int c = kThreads - 1 - static_cast<int>(threadIdx.x);  // chunks from the top
  float l[kPer], u[kPer];
  int dom[kPer];  // tokens of the bin that may belong to the domain
  float sl = 0.0f, su = 0.0f;
#pragma unroll
  for (int jj = 0; jj < kPer; ++jj) {
    const int j = c * kPer + jj;
    const float cn = static_cast<float>(sm.u.hist[j]);
    float cl = cn, cu = cn;
    const float y_lo = j == 0 ? inf_neg() : static_cast<float>(j - kBins / 2) * (1.0f / 32.0f) * f.inv_t;
    const float y_hi =
        j == kBins - 1 ? m : fminf(static_cast<float>(j + 1 - kBins / 2) * (1.0f / 32.0f) * f.inv_t, m);
    if (!f.joint) {
      if (f.use_k) {
        if (j < bin_k) cl = cu = 0.0f;
        if (j == bin_k) cl = static_cast<float>(need);
      }
      if (f.use_mp) {
        if (y_hi < f.thr) cl = cu = 0.0f;
        if (!(y_lo >= f.thr)) cl = 0.0f;
      }
    }
    l[jj] = cl > 0.0f ? cl * expf(y_lo - m) : 0.0f;
    u[jj] = cu > 0.0f ? cu * expf(y_hi - m) : 0.0f;
    dom[jj] = static_cast<int>(cu);
    sl += l[jj];
    su += u[jj];
  }
  const float zl = block_all(sl, sm.wf, OpSum{});
  const float zu = block_all(su, sm.wf, OpSum{});
  float run_l = block_incl_scan(sl, sm.wf) - sl;  // mass of the chunks above this one
  float run_u = block_incl_scan(su, sm.wf) - su;
  const float lim_keep = f.p * zl * (1.0f - 1e-3f) - 1e-6f;
  const float lim_drop = f.p * zu * (1.0f + 1e-3f) + 2e-6f;
  int keep = kBins, drop = -1;
#pragma unroll
  for (int jj = kPer - 1; jj >= 0; --jj) {
    const int j = c * kPer + jj;
    const float above_l = run_l;
    run_u += u[jj];
    run_l += l[jj];
    if (run_u * (1.0f + 1e-3f) < lim_keep) keep = j;
    if (above_l * (1.0f - 1e-3f) > lim_drop) drop = max(drop, j);
  }
  keep = block_all(keep, sm.wi, OpMin{});
  drop = block_all(drop, sm.wi, OpMax{});
  int n = 0;
#pragma unroll
  for (int jj = 0; jj < kPer; ++jj) {
    const int j = c * kPer + jj;
    if (j > drop && j < keep) n += dom[jj];
  }
  *lo = drop + 1;
  *hi = keep - 1;
  *n_range = block_all(n, sm.wi, OpSum{});
}

/// Sort the first nc candidate slots (rn of them non-empty, all in the nucleus domain) into nucleus
/// order and race the ones whose mass before them, counted from mass_above, is below target.
__device__ void nucleus_resolve(FilteredSmem& sm, int nc, int rn, uint64_t mass_above,
                                uint64_t target, const RowFilter& f, uint32_t kth, float m,
                                const RngParams& rng, int row_index, float& best,
                                int32_t& best_token) {
  const int tid = threadIdx.x;
  int n = 1;
  while (n < nc) n <<= 1;
  for (int i = nc + tid; i < n; i += kThreads) sm.u.cand[i] = 0ull;
  __syncthreads();
  bitonic_desc(sm.u.cand, n);  // the rn domain candidates come first, in nucleus order
  const int per = (rn + kThreads - 1) / kThreads;
  const int beg = min(rn, tid * per);
  const int end = min(rn, beg + per);
  uint64_t s = 0;
  for (int i = beg; i < end; ++i) s += mass40(key_value(cand_key(sm.u.cand[i])) * f.inv_t, m);
  uint64_t cum = mass_above + block_incl_scan(s, sm.w64) - s;
  for (int i = beg; i < end && cum < target; ++i) {
    const uint64_t cd = sm.u.cand[i];
    const uint32_t key = cand_key(cd);
    const float y = key_value(key) * f.inv_t;
    const int token = cand_token(cd);
    if (f.in_base(key, y, kth)) race_take(y + race_gumbel(rng, row_index, token), token, best, best_token);
    cum += mass40(y, m);
  }
}

/// Top-p rows without top-k whose bound range holds more tokens than the buffer: pass 2 sums the
/// exact mass of every range bin, which fixes the boundary bin, and pass 3 collects that bin alone.
/// Returns false when the boundary bin itself exceeds the buffer (the caller runs the fallback).
template <typename in_t, int VEC, bool SPLIT>
__device__ bool topp_three_pass(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, const RowProc& proc,
                                const RowFilter& f, float m, const RngParams& rng, int p_lo, int p_hi,
                                FilteredSmem& sm, float& best, int32_t& best_token) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const Segment seg = rc.seg();
  const int nb = p_hi - p_lo + 1;
  uint64_t* bm = sm.u.cand;  // exact domain mass of range bin p_lo + i
  for (int i = tid; i < nb; i += kThreads) bm[i] = 0ull;
  if (tid == 0) sm.sh[0] = -1;
  __syncthreads();
  uint64_t z = 0, above = 0;
  RaceQueue q{sm.qtok[tid >> 5], sm.qy[tid >> 5], 0};
  for (int it = 0; it < seg.iters; ++it) {
    const int c = seg.beg + it * 32 + lane;
    const bool valid = c < seg.end;
    float x[VEC];
    if (valid) load_chunk<in_t, VEC>(row, c, x);
    unsigned take = 0u;
    float ys[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float v = valid ? proc.apply(x[i], c * VEC + i) : inf_neg();
      const float y = v * f.inv_t;
      ys[i] = y;
      const bool in_mp = v != inf_neg() && y >= f.thr;
      if (v != inf_neg() && (f.joint || in_mp)) {
        const uint64_t w = mass40(y, m);
        z += w;
        const int b = logit_bin(v);
        if (b > p_hi) {
          above += w;
          if (in_mp) take |= 1u << i;
        } else if (b >= p_lo && w != 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&bm[b - p_lo]), static_cast<unsigned long long>(w));
        }
      }
    }
    q.push_chunk<VEC>(take, c * VEC, ys, rng, rc.row, best, best_token);
  }
  q.drain(rng, rc.row, best, best_token);
  const uint64_t z_cta = block_all(z, sm.w64, OpSum{});
  const uint64_t a_cta = block_all(above, sm.w64, OpSum{});
  rc.merge(bm, nb);
  rc.exch(z_cta, a_cta, 0, 0, sm.tab);
  uint64_t total = 0, mass_above = 0;
  for (int j = 0; j < rc.S; ++j) {
    total += sm.tab[j][0];
    mass_above += sm.tab[j][1];
  }
  uint64_t target = static_cast<uint64_t>(static_cast<double>(f.p) * static_cast<double>(total) + 0.5);
  if (target < 1) target = 1;
  // Boundary bin: walk the range from the top, thread t owning `per` consecutive bins.
  const int per = (nb + kThreads - 1) / kThreads;
  const int beg = min(nb, tid * per);
  const int end = min(nb, beg + per);
  uint64_t s = 0;
  for (int i = beg; i < end; ++i) s += bm[p_hi - p_lo - i];
  const uint64_t before = mass_above + block_incl_scan(s, sm.w64) - s;
  if (before < target && target <= before + s) {
    uint64_t run = before;
    for (int i = beg; i < end; ++i) {
      const uint64_t w = bm[p_hi - p_lo - i];
      if (target <= run + w) {
        sm.sh[0] = p_hi - i;
        sm.sh64[0] = run;
        break;
      }
      run += w;
    }
  }
  __syncthreads();
  const int bstar = sm.sh[0];
  const uint64_t mass_before = sm.sh64[0];
  if (bstar < 0) return true;  // the nucleus ends above the range
  if (!SPLIT && tid == 0) sm.cand_count = 0;
  __syncthreads();  // the bin masses are dead from here on
  for (int it = 0; it < seg.iters; ++it) {
    const int c = seg.beg + it * 32 + lane;
    const bool valid = c < seg.end;
    float x[VEC];
    if (valid) load_chunk<in_t, VEC>(row, c, x);
    unsigned take = 0u;
    float ys[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int token = c * VEC + i;
      const float v = valid ? proc.apply(x[i], token) : inf_neg();
      const int b = logit_bin(v);
      ys[i] = v * f.inv_t;
      const bool live = v != inf_neg() && b >= bstar && b <= p_hi;
      const bool in_mp = live && ys[i] >= f.thr;
      if (live && b == bstar && (f.joint || in_mp)) cand_push(rc, sm, pack_cand(order_key(v), token));
      if (in_mp && b > bstar) take |= 1u << i;
    }
    q.push_chunk<VEC>(take, c * VEC, ys, rng, rc.row, best, best_token);
  }
  q.drain(rng, rc.row, best, best_token);
  if constexpr (SPLIT) {
    rc.barrier();  // orders every CTA's appends before the reads below
  } else {
    __syncthreads();
  }
  if (cand_total(rc, sm) > kCand) return false;
  if (rc.leader()) {
    const int nc = cand_gather(rc, sm);
    nucleus_resolve(sm, nc, nc, mass_before, target, f, 0u, m, rng, rc.row, best, best_token);
  }
  return true;
}

/// Top-p-only row (no top-k, no min-p, 0 < p < 1). Pass 1 builds a fixed-point mass histogram
/// with one polynomial exp per element instead of count-bin bounds; the exact boundary mass bin
/// follows from it, so pass 2 only re-reads the row above that bin and collects the boundary
/// tokens. The above-boundary tokens race inline; the boundary prefix is refined in shared
/// memory. 2048 mass bins overlay the 4096 count bins, so this path adds no static shared memory.
/// A clamped top bin, a zero-mass row or a boundary bin beyond kCand takes the exact fallback.
/// For p >= 0.95 a pass after the row max builds the exact mass40 histogram instead: the cut
/// then lands in one bin, so the candidate range needs no slack and stays inside kCand.
template <typename in_t, int VEC, bool SPLIT>
__device__ void topp_only_row(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, int vocab,
                              const RowProc& proc, float inv_t, float top_p,
                              const RngParams& rng, FilteredSmem& sm) {
  (void)vocab;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const Segment seg = rc.seg();
  // A high top-p cut keeps almost the whole row above the boundary. The polynomial
  // mass histogram then needs the two-bin slack, and the three boundary bins can
  // exceed the candidate buffer. For p >= 0.97 build an exact mass40 histogram
  // after the row max is known instead: the cut lands in one bin, so the candidate
  // range is a single bin and the fast path stays available.
  const bool exact_hist = top_p >= 0.95f;
  uint64_t* hm = reinterpret_cast<uint64_t*>(sm.u.hist);
  for (int b = tid; b < kMassBins; b += kThreads) hm[b] = 0ull;
  __syncthreads();

  // ---- pass 1: max/argmax and the mass histogram of y = v * (1/T) ----
  float mx = inf_neg();
  int infs = 0, finite = 0;
  float best = inf_neg();
  int32_t best_token = INT32_MAX;
  bool clamped_hi = false;
  for (int c = seg.beg + lane; c < seg.end; c += 32) {
    float x[VEC];
    load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int token = c * VEC + i;
      const float v = proc.apply(x[i], token);
      if (v == inf_pos()) {
        ++infs;
        continue;
      }
      if (v == inf_neg()) continue;
      ++finite;
      const float y = v * inv_t;
      mx = fmaxf(mx, y);
      if (y > best) {
        best = y;
        best_token = token;
      }
      if (!exact_hist) {
        if (y > 64.0f) clamped_hi = true;
        const int mb = logit_bin(y) >> 1;
        const float hi = static_cast<float>(mb - (kMassBins / 2 - 1)) * (1.0f / 16.0f);
        const float t = fminf(y, hi) - hi;
        const uint64_t mq = t >= -1.0f / 16.0f ? fixed_mass40(exp_small(t)) : 0ull;
        atomicAdd(reinterpret_cast<unsigned long long*>(&hm[mb]),
                  static_cast<unsigned long long>(mq));
      }
    }
  }
  const float m_cta = kops::block_reduce_max<float, kWarps>(mx, sm.sv);
  const int inf_cta = kops::block_reduce_sum<int, kWarps>(infs, sm.sinf);
  const int fin_cta = kops::block_reduce_sum<int, kWarps>(finite, sm.sfin);
  const int clamp_cta = block_all(clamped_hi ? 1 : 0, sm.wi, OpMax{});
  rc.exch(__float_as_uint(m_cta), static_cast<uint32_t>(inf_cta), static_cast<uint32_t>(fin_cta),
          static_cast<uint32_t>(clamp_cta), sm.tab);
  float m = inf_neg();
  int inf_count = 0, finite_count = 0, range_hi = 0;
  for (int j = 0; j < rc.S; ++j) {
    m = fmaxf(m, __uint_as_float(static_cast<uint32_t>(sm.tab[j][0])));
    inf_count += static_cast<int>(sm.tab[j][1]);
    finite_count += static_cast<int>(sm.tab[j][2]);
    range_hi |= static_cast<int>(sm.tab[j][3]);
  }
  if (inf_count > 0 || top_p >= 1.0f) {
    // +inf entries win uniformly; top_p >= 1 is the unfiltered categorical row.
    __syncthreads();
    categorical_row<in_t, VEC, SPLIT>(rc, row, proc, inv_t, rng, sm.sv, sm.si, sm.sinf, sm.u.rec,
                                      sm.sh64, sm.shi, sm.tab);
    return;
  }
  if (m == inf_neg() || finite_count == 0) {
    if (rc.leader() && tid == 0) rc.emit(-1);
    return;
  }
  // p <= 0 (or NaN) keeps only the largest token.
  if (!(top_p > 0.0f)) {
    race_finish(rc, best, best_token, sm.sv, sm.si, sm.tab);
    return;
  }

  RowFilter f;
  f.inv_t = inv_t;
  f.thr = inf_neg();
  f.p = top_p;
  f.k = 0;
  f.use_k = false;
  f.use_mp = false;
  f.use_p = true;
  f.joint = false;
  if (range_hi != 0) {  // the top bin's clamped mass is not a usable Z
    __syncthreads();
    best = inf_neg();
    best_token = INT32_MAX;
    filtered_fallback<in_t, VEC, SPLIT>(rc, row, proc, f, m, rng, sm, best, best_token);
    race_finish(rc, best, best_token, sm.sv, sm.si, sm.tab);
    return;
  }

  if (exact_hist) {
    // Exact per-bin mass40, built after m is known. This makes the cut land in
    // exactly one bin, so the candidate range needs no slack.
    for (int b = tid; b < kMassBins; b += kThreads) hm[b] = 0ull;
    __syncthreads();
    for (int c2 = seg.beg + lane; c2 < seg.end; c2 += 32) {
      float x[VEC];
      load_chunk<in_t, VEC>(row, c2, x);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float v = proc.apply(x[i], c2 * VEC + i);
        if (v == inf_neg()) continue;
        const float y = v * inv_t;
        atomicAdd(reinterpret_cast<unsigned long long*>(&hm[logit_bin(y) >> 1]),
                  static_cast<unsigned long long>(mass40(y, m)));
      }
    }
    __syncthreads();
  }
  rc.merge(hm, kMassBins);
  // Bin masses in place: each hm[] slot is dead once scaled by exp(hi_bin - m).
  double* bm = reinterpret_cast<double*>(hm);
  for (int j = tid; j < kMassBins; j += kThreads) {
    const double hi = (static_cast<double>(j) - (kMassBins / 2 - 1)) * (1.0 / 16.0);
    bm[j] = exact_hist
                ? static_cast<double>(hm[j])
                : static_cast<double>(hm[j]) * exp2((hi - static_cast<double>(m)) * 1.4426950408889634);
  }
  __syncthreads();

  // Boundary bin: the lowest bin whose mass above it is still below p * Z. Chunks run from the
  // bottom so the first crossing is the boundary; a small margin makes the range a superset.
  constexpr int kPer = kMassBins / kThreads;  // 8 bins per thread
  const int c = tid;
  double s = 0.0;
#pragma unroll
  for (int jj = 0; jj < kPer; ++jj) s += bm[c * kPer + jj];
  const double z = block_all(s, sm.wd, OpSum{});
  const double target = static_cast<double>(top_p) * z;
  const double eps = exact_hist ? 0.0 : 1e-4 * z + 1e-9;
  const double above_chunk = z - (block_incl_scan(s, sm.wd));
  double run = above_chunk + (s - bm[c * kPer]);
  int hi_bin = kMassBins;
#pragma unroll
  for (int jj = 0; jj < kPer; ++jj) {
    const int j = c * kPer + jj;
    if (run < target + eps) {
      hi_bin = j;
      break;
    }
    if (jj + 1 < kPer) run -= bm[j + 1];
  }
  const int boundary = block_all(hi_bin, sm.wi, OpMin{});
  if (z <= 0.0 || boundary >= kMassBins) {
    if (rc.leader() && tid == 0) rc.emit(-1);
    return;
  }
  // Keep one bin of slack on each side of the histogram boundary: the exact mass above the
  // candidate range is then safely below the cut, and a per-bin histogram error can only move the
  // true boundary inside the collected range. The exact histogram lands in one bin already.
  const int hi = exact_hist ? boundary : min(kMassBins - 1, boundary + 1);
  const int lo = exact_hist ? boundary : (boundary > 0 ? boundary - 1 : 0);
  // bm[] is already in fixed-point 2^40 units; the histogram total is accurate enough for the
  // boundary cut, and the exact masses of the collected/raced tokens refine the draw.
  const uint64_t target40 = static_cast<uint64_t>(target + 0.5);
  if (!SPLIT && tid == 0) sm.cand_count = 0;
  __syncthreads();  // bm is dead from here on (candidates reuse the same storage)

  // Mass above the candidate range from the histogram (fixed 2^40). All tokens above the range
  // are inside the nucleus when this is below the cut, so pass 2 does not need an exact
  // fixed-point mass for them: only the boundary candidates are refined exactly.
  double above_hi = 0.0;
  for (int j = tid; j < kMassBins; j += kThreads) {
    if (j > hi) above_hi += bm[j];
  }
  const uint64_t mass_above_hist = static_cast<uint64_t>(block_all(above_hi, sm.wd, OpSum{}) + 0.5);
  if (!exact_hist && mass_above_hist >= target40) {
    __syncthreads();
    float best = inf_neg();
    int32_t best_token = INT32_MAX;
    filtered_fallback<in_t, VEC, SPLIT>(rc, row, proc, f, m, rng, sm, best, best_token);
    race_finish(rc, best, best_token, sm.sv, sm.si, sm.tab);
    return;
  }

  // ---- pass 2: race the tokens above the boundary bin, collect the boundary bins ----
  RaceQueue q{sm.qtok[warp], sm.qy[warp], 0};
  best = inf_neg();
  best_token = INT32_MAX;
  for (int it = 0; it < seg.iters; ++it) {
    const int c2 = seg.beg + it * 32 + lane;
    const bool valid = c2 < seg.end;
    float x[VEC];
    if (valid) load_chunk<in_t, VEC>(row, c2, x);
    unsigned take = 0u;
    float ys[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int token = c2 * VEC + i;
      const float v = valid ? proc.apply(x[i], token) : inf_neg();
      const float y = v * inv_t;
      ys[i] = y;
      if (v == inf_neg()) continue;
      const int mb = logit_bin(y) >> 1;
      if (mb > hi) {
        take |= 1u << i;
      } else if (mb >= lo) {
        cand_push(rc, sm, pack_cand(order_key(v), token));
      }
    }
    q.push_chunk<VEC>(take, c2 * VEC, ys, rng, rc.row, best, best_token);
  }
  q.drain(rng, rc.row, best, best_token);
  block_argmax0(best, best_token, sm.sv, sm.si);
  rc.exch(__float_as_uint(best), static_cast<uint32_t>(best_token), 0, 0, sm.tab);
  float cta_best = inf_neg();
  int32_t cta_token = INT32_MAX;
  for (int j = 0; j < rc.S; ++j) {
    const float v = __uint_as_float(static_cast<uint32_t>(sm.tab[j][0]));
    const int32_t k = static_cast<int32_t>(static_cast<uint32_t>(sm.tab[j][1]));
    if (v > cta_best || (v == cta_best && k < cta_token)) {
      cta_best = v;
      cta_token = k;
    }
  }
  if constexpr (!SPLIT) {
    __syncthreads();
  }
  const int nc_total = cand_total(rc, sm);
  if (nc_total > kCand) {
    filtered_fallback<in_t, VEC, SPLIT>(rc, row, proc, f, m, rng, sm, cta_best, cta_token);
    race_finish(rc, cta_best, cta_token, sm.sv, sm.si, sm.tab);
    return;
  }
  if (rc.leader()) {
    const int nc = cand_gather(rc, sm);
    nucleus_resolve(sm, nc, nc, mass_above_hist, target40, f, 0u, m, rng, rc.row, cta_best,
                    cta_token);
  }
  race_finish(rc, cta_best, cta_token, sm.sv, sm.si, sm.tab);
}

/// TOPP is a compile-time copy of "top-p active" so rows without top-p (top-k / min-p only) carry
/// none of the top-p code and keep their registers in the hot loops.
template <typename in_t, int VEC, bool TOPP, bool SPLIT>
__device__ void filtered_row(RowCtx<SPLIT>& rc, const in_t* __restrict__ row, int vocab,
                             const RowProc& proc, float inv_t, int k, float min_p, float top_p,
                             bool joint, const RngParams& rng, FilteredSmem& sm) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const Segment seg = rc.seg();
  const float p = top_p > 0.0f ? top_p : 0.0f;  // <= 0 or NaN: only the largest token
  constexpr bool use_p = TOPP;  // == (p < 1), decided by the kernel
  const bool want_hist = (k > 0 && k < vocab) || use_p;
  if (want_hist) {
    for (int b = tid; b < kBins; b += kThreads) sm.u.hist[b] = 0u;
    __syncthreads();
  }

  // ---- pass 1: max of y = x / T, +inf / finite counts, histogram of the processed logits ----
  float mx = inf_neg();
  int infs = 0, finite = 0;
  for (int c = seg.beg + lane; c < seg.end; c += 32) {
    float x[VEC];
    load_chunk<in_t, VEC>(row, c, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float v = proc.apply(x[i], c * VEC + i);
      if (v == inf_pos()) {
        ++infs;
      } else if (v != inf_neg()) {
        ++finite;
        mx = fmaxf(mx, v * inv_t);
        if (want_hist) atomicAdd(&sm.u.hist[logit_bin(v)], 1u);
      }
    }
  }
  const float m_cta = kops::block_reduce_max<float, kWarps>(mx, sm.sv);
  const int inf_cta = kops::block_reduce_sum<int, kWarps>(infs, sm.sinf);
  const int fin_cta = kops::block_reduce_sum<int, kWarps>(finite, sm.sfin);
  rc.exch(__float_as_uint(m_cta), static_cast<uint32_t>(inf_cta), static_cast<uint32_t>(fin_cta), 0, sm.tab);
  float m = inf_neg();
  int inf_count = 0, finite_count = 0;
  for (int j = 0; j < rc.S; ++j) {
    m = fmaxf(m, __uint_as_float(static_cast<uint32_t>(sm.tab[j][0])));
    inf_count += static_cast<int>(sm.tab[j][1]);
    finite_count += static_cast<int>(sm.tab[j][2]);
  }
  if (want_hist) rc.merge(sm.u.hist, kBins);
  const bool use_k = k > 0 && k < vocab && k < finite_count;
  const bool use_mp = min_p > 0.0f;
  if (inf_count > 0 || (!use_k && !use_mp && !use_p)) {
    // +inf entries win outright (uniform over them); a row without an active filter is plain
    // categorical. Both re-read the row; neither is a decode-loop case.
    __syncthreads();
    categorical_row<in_t, VEC, SPLIT>(rc, row, proc, inv_t, rng, sm.sv, sm.si, sm.sinf, sm.u.rec,
                                      sm.sh64, sm.shi, sm.tab);
    return;
  }
  if (m == inf_neg()) {
    if (rc.leader() && tid == 0) rc.emit(-1);
    return;
  }
  RowFilter f;
  f.inv_t = inv_t;
  f.thr = use_mp ? m + logf(min_p) : inf_neg();
  f.p = p;
  f.k = k;
  f.use_k = use_k;
  f.use_mp = use_mp;
  f.use_p = use_p;
  f.joint = joint && use_p;

  // ---- top-k boundary bin: thread t owns the 16 bins of chunk 255 - t (scan from the top) ----
  int bin_k = -1, k_cand = 0, need = 0;
  if (use_k) {
    constexpr int kPer = kBins / kThreads;
    const int c = kThreads - 1 - tid;
    uint32_t s = 0;
#pragma unroll
    for (int j = 0; j < kPer; ++j) s += sm.u.hist[c * kPer + j];
    const uint32_t incl = block_incl_scan(s, sm.scan);
    const uint32_t kk = static_cast<uint32_t>(k);
    if (incl - s < kk && kk <= incl) {
      uint32_t run = incl - s;
      int b = c * kPer + kPer - 1;
      for (; b > c * kPer; --b) {
        if (run + sm.u.hist[b] >= kk) break;
        run += sm.u.hist[b];
      }
      sm.sh[0] = b;
      sm.sh[1] = static_cast<int>(sm.u.hist[b]);
      sm.sh[2] = k - static_cast<int>(run);
    }
    __syncthreads();
    bin_k = sm.sh[0];
    k_cand = sm.sh[1];
    need = sm.sh[2];
    __syncthreads();
  }
  // ---- top-p boundary bins [p_lo, p_hi] from the mass bounds ----
  int p_lo = kBins, p_hi = -1, p_cand = 0;
  if (use_p) topp_range(sm, f, m, bin_k, need, &p_lo, &p_hi, &p_cand);
  if (!SPLIT && tid == 0) sm.cand_count = 0;
  __syncthreads();  // the histogram is dead from here on (candidates reuse it)

  float best = inf_neg();
  int32_t best_token = INT32_MAX;
  if (k_cand + p_cand > kCand) {
    const bool done = !use_k && use_p && p_hi - p_lo + 1 <= kCand &&
                      topp_three_pass<in_t, VEC, SPLIT>(rc, row, proc, f, m, rng, p_lo, p_hi, sm, best,
                                                        best_token);
    if (!done) filtered_fallback<in_t, VEC, SPLIT>(rc, row, proc, f, m, rng, sm, best, best_token);
  } else {
    // ---- pass 2: race over the tokens kept for sure, exact domain mass, collect the rest ----
    uint64_t z = 0, above = 0;
    RaceQueue q{sm.qtok[tid >> 5], sm.qy[tid >> 5], 0};
    for (int it = 0; it < seg.iters; ++it) {
      const int c = seg.beg + it * 32 + lane;
      const bool valid = c < seg.end;
      float x[VEC];
      if (valid) load_chunk<in_t, VEC>(row, c, x);
      unsigned take = 0u;
      float ys[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const int token = c * VEC + i;
        const float v = valid ? proc.apply(x[i], token) : inf_neg();
        const float y = v * inv_t;
        ys[i] = y;
        if (v != inf_neg()) {
          const bool in_mp = y >= f.thr;
          const int b = logit_bin(v);
          const int ks = !use_k ? 1 : (b > bin_k ? 1 : (b < bin_k ? -1 : 0));
          const int ps = !use_p ? 1 : (b > p_hi ? 1 : (b < p_lo ? -1 : 0));
          const bool base = in_mp && ks == 1;
          if (ks == 0 || (ps == 0 && (f.joint || base))) {
            cand_push(rc, sm, pack_cand(order_key(v), token));
          } else {
            if (use_p && (f.joint || base)) {
              const uint64_t w = mass40(y, m);
              z += w;
              if (ps == 1) above += w;
            }
            if (base && ps == 1) take |= 1u << i;
          }
        }
      }
      q.push_chunk<VEC>(take, c * VEC, ys, rng, rc.row, best, best_token);
    }
    q.drain(rng, rc.row, best, best_token);
    const uint64_t z_cta = use_p ? block_all(z, sm.w64, OpSum{}) : 0ull;
    const uint64_t a_cta = use_p ? block_all(above, sm.w64, OpSum{}) : 0ull;
    rc.exch(z_cta, a_cta, 0, 0, sm.tab);  // on Path B also orders the candidate appends
    uint64_t z_rest = 0, a_rest = 0;
    for (int j = 0; j < rc.S; ++j) {
      z_rest += sm.tab[j][0];
      a_rest += sm.tab[j][1];
    }

    // ---- leader: exact top-k threshold, then domain mass, then the nucleus of the candidates ----
    if (rc.leader()) {
      const int nc = cand_gather(rc, sm);
      const uint32_t kth = use_k ? cand_kth_key(sm.u.cand, nc, bin_k, need, sm.radix, sm.sh) : 0u;
      uint64_t zc = 0, ac = 0;
      int n_sort = 0;
      for (int i = tid; i < nc; i += kThreads) {
        const uint64_t cd = sm.u.cand[i];
        const uint32_t key = cand_key(cd);
        const int token = cand_token(cd);
        const float v = key_value(key);
        const float y = v * inv_t;
        const bool base = f.in_base(key, y, kth);
        const bool dom = f.joint || base;
        const int b = logit_bin(v);
        const int ps = !use_p ? 1 : (b > p_hi ? 1 : (b < p_lo ? -1 : 0));
        if (use_p && dom) {
          const uint64_t w = mass40(y, m);
          zc += w;
          if (ps == 1) ac += w;
        }
        if (ps == 1 && base) race_take(y + race_gumbel(rng, rc.row, token), token, best, best_token);
        const bool sort_me = use_p && ps == 0 && dom;
        n_sort += sort_me;
        if (!sort_me) sm.u.cand[i] = 0ull;
      }
      if (use_p) {
        const uint64_t total = z_rest + block_all(zc, sm.w64, OpSum{});
        const uint64_t mass_above = a_rest + block_all(ac, sm.w64, OpSum{});
        const int rn = block_all(n_sort, sm.wi, OpSum{});
        uint64_t target = static_cast<uint64_t>(static_cast<double>(p) * static_cast<double>(total) + 0.5);
        if (target < 1) target = 1;
        nucleus_resolve(sm, nc, rn, mass_above, target, f, kth, m, rng, rc.row, best, best_token);
      }
    }
  }
  race_finish(rc, best, best_token, sm.sv, sm.si, sm.tab);
}

/// 16-byte loads need every row 16-byte aligned and V a multiple of the vector width.
template <typename in_t>
__device__ __forceinline__ bool rows_vectorizable(const in_t* logits, int64_t stride, int vocab) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(in_t));
  return vocab % kVec == 0 && (stride * static_cast<int64_t>(sizeof(in_t))) % 16 == 0 &&
         (reinterpret_cast<uintptr_t>(logits) & 15u) == 0;
}

}  // namespace

template <typename in_t, bool SPLIT>
__global__ void __launch_bounds__(kThreads)
fused_greedy_kernel(int32_t* __restrict__ out_ids, const in_t* __restrict__ logits,
                    int64_t logits_stride, FusedSamplingParams params, int batch_size,
                    int vocab_size, int penalty_row_bytes, FusedSplitDevice split) {
  const int row_index = SPLIT ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
  if (row_index >= batch_size) return;
  __shared__ float s_value[kWarps];
  __shared__ int32_t s_index[kWarps];
  __shared__ uint64_t s_tab[kMaxSplits][kExWords];
  const in_t* row = logits + static_cast<int64_t>(row_index) * logits_stride;
  const RowProc proc = row_proc(params, row_index, vocab_size, penalty_row_bytes);
  uint8_t* mark = params.update_penalty_mask != 0 ? proc.mask : nullptr;
  constexpr int kVec = 16 / static_cast<int>(sizeof(in_t));
  // Split rows are always vectorizable (the host keeps other layouts on one CTA per row), so the
  // split kernels compile only the 16-byte path.
  if (SPLIT || rows_vectorizable(logits, logits_stride, vocab_size)) {
    RowCtx<SPLIT> c = make_row_ctx<SPLIT>(row_index, vocab_size / kVec, out_ids, mark, split);
    greedy_row<in_t, kVec, SPLIT>(c, row, proc, s_value, s_index, s_tab);
  } else if constexpr (!SPLIT) {
    RowCtx<SPLIT> c = make_row_ctx<SPLIT>(row_index, vocab_size, out_ids, mark, split);
    greedy_row<in_t, 1, SPLIT>(c, row, proc, s_value, s_index, s_tab);
  }
}

template <typename in_t, bool SPLIT>
__global__ void __launch_bounds__(kThreads)
fused_categorical_kernel(int32_t* __restrict__ out_ids, const in_t* __restrict__ logits,
                         int64_t logits_stride, FusedSamplingParams params, int batch_size,
                         int vocab_size, int penalty_row_bytes, FusedSplitDevice split) {
  const int row_index = SPLIT ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
  if (row_index >= batch_size) return;
  __shared__ float s_value[kWarps];
  __shared__ int32_t s_index[kWarps];
  __shared__ int s_int[kWarps];
  __shared__ uint64_t s_rec[kWarps][kMaxRecords];
  __shared__ uint64_t s_u64[2];
  __shared__ int s_i[3];
  __shared__ uint64_t s_tab[kMaxSplits][kExWords];
  const in_t* row = logits + static_cast<int64_t>(row_index) * logits_stride;
  const RowProc proc = row_proc(params, row_index, vocab_size, penalty_row_bytes);
  uint8_t* mark = params.update_penalty_mask != 0 ? proc.mask : nullptr;
  const float temperature = params.sampling.temperature != nullptr
                                ? params.sampling.temperature[row_index]
                                : params.sampling.default_temperature;
  constexpr int kVec = 16 / static_cast<int>(sizeof(in_t));
  const bool vec = SPLIT || rows_vectorizable(logits, logits_stride, vocab_size);
  RowCtx<SPLIT> c = make_row_ctx<SPLIT>(row_index, vec ? vocab_size / kVec : vocab_size, out_ids, mark, split);
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    if (vec) {
      greedy_row<in_t, kVec, SPLIT>(c, row, proc, s_value, s_index, s_tab);
    } else if constexpr (!SPLIT) {
      greedy_row<in_t, 1, SPLIT>(c, row, proc, s_value, s_index, s_tab);
    }
  } else {
    const float inv_t = 1.0f / temperature;
    if (vec) {
      categorical_row<in_t, kVec, SPLIT>(c, row, proc, inv_t, params.sampling.rng, s_value, s_index, s_int,
                                         s_rec, s_u64, s_i, s_tab);
    } else if constexpr (!SPLIT) {
      categorical_row<in_t, 1, SPLIT>(c, row, proc, inv_t, params.sampling.rng, s_value, s_index, s_int,
                                      s_rec, s_u64, s_i, s_tab);
    }
  }
}

template <typename in_t, bool SPLIT, bool P_ONLY>
__global__ void __launch_bounds__(kThreads, 4)
fused_filtered_kernel(int32_t* __restrict__ out_ids, const in_t* __restrict__ logits,
                      int64_t logits_stride, FusedSamplingParams params, int batch_size,
                      int vocab_size, int penalty_row_bytes, FusedSplitDevice split) {
  const int row_index = SPLIT ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
  if (row_index >= batch_size) return;
  __shared__ FilteredSmem sm;
  const in_t* row = logits + static_cast<int64_t>(row_index) * logits_stride;
  const RowProc proc = row_proc(params, row_index, vocab_size, penalty_row_bytes);
  uint8_t* mark = params.update_penalty_mask != 0 ? proc.mask : nullptr;
  const SamplingParams& sp = params.sampling;
  const float temperature = sp.temperature != nullptr ? sp.temperature[row_index] : sp.default_temperature;
  const int k = sp.top_k != nullptr ? sp.top_k[row_index] : sp.default_top_k;
  const float min_p = sp.min_p != nullptr ? sp.min_p[row_index] : sp.default_min_p;
  const float top_p = sp.top_p != nullptr ? sp.top_p[row_index] : sp.default_top_p;
  (void)k;
  (void)min_p;
  constexpr int kVec = 16 / static_cast<int>(sizeof(in_t));
  const bool vec = SPLIT || rows_vectorizable(logits, logits_stride, vocab_size);
  RowCtx<SPLIT> c = make_row_ctx<SPLIT>(row_index, vec ? vocab_size / kVec : vocab_size, out_ids, mark, split);
  if (SPLIT && c.leader() && threadIdx.x == 0) *c.cand_count() = 0u;  // before the first row barrier
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    if (vec) {
      greedy_row<in_t, kVec, SPLIT>(c, row, proc, sm.sv, sm.si, sm.tab);
    } else if constexpr (!SPLIT) {
      greedy_row<in_t, 1, SPLIT>(c, row, proc, sm.sv, sm.si, sm.tab);
    }
  } else if constexpr (P_ONLY) {
    if (vec) {
      topp_only_row<in_t, kVec, SPLIT>(c, row, vocab_size, proc, 1.0f / temperature, top_p, sp.rng, sm);
    } else if constexpr (!SPLIT) {
      topp_only_row<in_t, 1, SPLIT>(c, row, vocab_size, proc, 1.0f / temperature, top_p, sp.rng, sm);
    }
  } else if constexpr (SPLIT) {
    // One filtered body for split rows: with top_p >= 1 the nucleus target is >= Z, i.e. the
    // whole domain, which is the top-k / min-p result (one copy keeps the TU compile time down).
    filtered_row<in_t, kVec, true, true>(c, row, vocab_size, proc, 1.0f / temperature, k, min_p, top_p,
                                         sp.joint_topk_topp, sp.rng, sm);
  } else {
    const float inv_t = 1.0f / temperature;
    const bool topp = !(top_p >= 1.0f);  // NaN / <= 0 keep only the largest token
    const bool joint = sp.joint_topk_topp;
    if (vec && topp) {
      filtered_row<in_t, kVec, true, SPLIT>(c, row, vocab_size, proc, inv_t, k, min_p, top_p, joint, sp.rng, sm);
    } else if (vec) {
      filtered_row<in_t, kVec, false, SPLIT>(c, row, vocab_size, proc, inv_t, k, min_p, top_p, joint, sp.rng, sm);
    } else if (topp) {
      filtered_row<in_t, 1, true, SPLIT>(c, row, vocab_size, proc, inv_t, k, min_p, top_p, joint, sp.rng, sm);
    } else {
      filtered_row<in_t, 1, false, SPLIT>(c, row, vocab_size, proc, inv_t, k, min_p, top_p, joint, sp.rng, sm);
    }
  }
}

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

bool supported_dtype(DType dtype) {
  return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16;
}

Status check_logits(const Tensor& logits) {
  if (logits.data == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "sample_fused.logits: null data");
  }
  if (!supported_dtype(logits.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "sample_fused.logits: dtype %s",
                             dtype_name(logits.dtype));
  }
  if (logits.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "sample_fused.logits: ndim %d, expected 2",
                             logits.ndim);
  }
  if (logits.device < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "sample_fused.logits: host tensor is not supported");
  }
  if (logits.stride_unit != StrideUnit::kElement) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sample_fused.logits: only element strides are supported");
  }
  if (logits.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sample_fused.logits: inner stride must be 1");
  }
  if (logits.shape[0] < 0 || logits.shape[1] < 0) {
    return detail::set_error(Status::kInvalidArgument, "sample_fused.logits: negative shape");
  }
  if (logits.shape[1] > 0 && logits.stride[0] < logits.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sample_fused.logits: stride[0] must be >= vocab_size");
  }
  return Status::kSuccess;
}

/// Top-k, min-p or top-p may be active on some row: the filtered kernel and its plan are needed.
bool filters_active(const SamplingParams& params, int vocab) {
  return params.top_k != nullptr || (params.default_top_k > 0 && params.default_top_k < vocab) ||
         params.min_p != nullptr || params.default_min_p > 0.0f || params.top_p != nullptr ||
         params.default_top_p < 1.0f;
}

/// No top-k and no min-p on any row, with top-p active: the mass-histogram path applies. Per-row
/// top-k / min-p tensors keep the general path (their values are not known at plan time).
bool p_only_of(const SamplingParams& params, int vocab) {
  const bool k_off = params.top_k == nullptr &&
                     !(params.default_top_k > 0 && params.default_top_k < vocab);
  const bool mp_off = params.min_p == nullptr && !(params.default_min_p > 0.0f);
  const bool p_on = params.top_p != nullptr || params.default_top_p < 1.0f;
  return k_off && mp_off && p_on;
}

/// The only plan this build resolves for (params, vocab); sample_fused() requires an equal one.
SamplingPlan expected_plan(const SamplingParams& params, int vocab, uint64_t capture_generation) {
  SamplingPlan p{};
  p.splits = 1;
  p.block_threads = kThreads;
  if (filters_active(params, vocab)) {
    p.candidate_cap = kCand;
    p.hist_bins = kBins;
    p.flags = kPlanFiltered | (p_only_of(params, vocab) ? kPlanToppOnly : 0u);
  }
  p.capture_generation = capture_generation;
  return p;
}

/// Same filter configuration (the split count is validated on its own: callers may lower it).
bool same_plan(const SamplingPlan& a, const SamplingPlan& b) {
  return a.candidate_cap == b.candidate_cap && a.hist_bins == b.hist_bins &&
         a.block_threads == b.block_threads && a.flags == b.flags;
}

bool all_greedy_of(const SamplingParams& p) {
  return p.temperature == nullptr && !(p.default_temperature > 0.0f);
}

bool valid_splits(int s) { return s >= 1 && s <= kMaxSplits && (s & (s - 1)) == 0; }

/// The split kernels only compile the 16-byte path (same test as rows_vectorizable()).
bool host_rows_vectorizable(const Tensor& logits) {
  const int64_t es = logits.dtype == DType::kFloat32 ? 4 : 2;
  const int64_t vec = 16 / es;
  return logits.shape[1] % vec == 0 && (logits.stride[0] * es) % 16 == 0 &&
         (reinterpret_cast<uintptr_t>(logits.data) & 15u) == 0;
}

template <typename in_t>
const void* split_kernel(bool all_greedy, bool filtered, bool p_only) {
  if (all_greedy) return reinterpret_cast<const void*>(&fused_greedy_kernel<in_t, true>);
  if (filtered) {
    if (p_only) return reinterpret_cast<const void*>(&fused_filtered_kernel<in_t, true, true>);
    return reinterpret_cast<const void*>(&fused_filtered_kernel<in_t, true, false>);
  }
  return reinterpret_cast<const void*>(&fused_categorical_kernel<in_t, true>);
}

const void* split_kernel_for(DType dtype, bool all_greedy, bool filtered, bool p_only) {
  switch (dtype) {
    case DType::kFloat32:
      return split_kernel<float>(all_greedy, filtered, p_only);
    case DType::kFloat16:
      return split_kernel<__half>(all_greedy, filtered, p_only);
    default:
      return split_kernel<__nv_bfloat16>(all_greedy, filtered, p_only);
  }
}

/// CTAs of a split kernel that fit on the current device at once (0 without cooperative launch).
int split_capacity(const void* fn, int smem_bytes) {
  int dev = 0, coop = 0, sms = 0, occ = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess || coop == 0) return 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, kThreads, smem_bytes) != cudaSuccess) return 0;
  return sms * occ;
}

/// Largest power-of-two split <= 16 whose grid stays resident and keeps >= kMinShard tokens per CTA.
int choose_splits(int batch, int vocab, int capacity) {
  int s = 1;
  while (s < kMaxSplits && static_cast<int64_t>(batch) * s * 2 <= capacity && vocab / (s * 2) >= kMinShard) {
    s *= 2;
  }
  return s;
}

Status check_filters(const SamplingParams& params, int vocab) {
  (void)vocab;
  if (isnan(params.default_min_p)) {
    return detail::set_error(Status::kInvalidArgument, "sample_fused: min_p is NaN");
  }
  if (isnan(params.default_top_p)) {
    return detail::set_error(Status::kInvalidArgument, "sample_fused: top_p is NaN");
  }
  return Status::kSuccess;
}

template <typename in_t>
Status launch_fused(const Tensor& out_ids, const Tensor& logits, const FusedSamplingParams& params,
                    const LaunchContext& ctx, int batch, int vocab, bool all_greedy, bool filtered,
                    bool p_only, const SamplingPlan& plan, const Workspace& ws) {
  int64_t stride = logits.stride[0];
  int penalty_row_bytes = (vocab + 7) / 8;
  cudaStream_t st = stream_of(ctx);
  auto* out = static_cast<int32_t*>(out_ids.data);
  const auto* in = static_cast<const in_t*>(logits.data);
  FusedSplitDevice split{};
  split.splits = plan.splits;
  cudaError_t err = cudaSuccess;
  if (plan.splits == 1) {
    if (all_greedy) {
      fused_greedy_kernel<in_t, false><<<batch, kThreads, 0, st>>>(out, in, stride, params, batch, vocab,
                                                                  penalty_row_bytes, split);
    } else if (filtered) {
      if (p_only) {
        fused_filtered_kernel<in_t, false, true><<<batch, kThreads, 0, st>>>(
            out, in, stride, params, batch, vocab, penalty_row_bytes, split);
      } else {
        fused_filtered_kernel<in_t, false, false><<<batch, kThreads, 0, st>>>(
            out, in, stride, params, batch, vocab, penalty_row_bytes, split);
      }
    } else {
      fused_categorical_kernel<in_t, false><<<batch, kThreads, 0, st>>>(out, in, stride, params, batch,
                                                                       vocab, penalty_row_bytes, split);
    }
    err = cudaGetLastError();
  } else {
    // Cooperative launch: every CTA of the (splits, batch) grid is resident, so the row barriers
    // cannot deadlock (the launch fails instead when the grid does not fit).
    split.scratch = static_cast<uint8_t*>(ws.device_scratch);
    split.barriers = static_cast<unsigned long long*>(ws.device_persistent);
    FusedSamplingParams p = params;
    void* args[] = {&out, &in, &stride, &p, &batch, &vocab, &penalty_row_bytes, &split};
    err = cudaLaunchCooperativeKernel(split_kernel<in_t>(all_greedy, filtered, p_only),
                                      dim3(static_cast<unsigned>(plan.splits), static_cast<unsigned>(batch)),
                                      dim3(kThreads), args, 0, st);
  }
  return err == cudaSuccess
             ? Status::kSuccess
             : detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                         "sample_fused: %s", cudaGetErrorString(err));
}

}  // namespace

Status sampling_plan(const Tensor& logits, const SamplingParams& params, const LaunchContext& ctx,
                     SamplingPlan* out) {
  if (out == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "sampling_plan: out is null");
  }
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sampling_plan: CUDA backend only");
  }
  const Status s = check_logits(logits);
  if (s != Status::kSuccess) return s;
  if (logits.shape[1] <= 0 || logits.shape[0] <= 0) {
    *out = SamplingPlan{};
    out->capture_generation = ctx.capture_generation;
    return Status::kSuccess;
  }
  const int vocab = static_cast<int>(logits.shape[1]);
  const Status f = check_filters(params, vocab);
  if (f != Status::kSuccess) return f;
  *out = expected_plan(params, vocab, ctx.capture_generation);
  const int batch = static_cast<int>(logits.shape[0]);
  const bool p_only = p_only_of(params, vocab);
  out->splits = host_rows_vectorizable(logits)
                    ? choose_splits(batch, vocab,
                                    split_capacity(split_kernel_for(logits.dtype, all_greedy_of(params),
                                                                    filters_active(params, vocab), p_only),
                                                   0))
                    : 1;
  return Status::kSuccess;
}

Status sampling_fused_workspace_requirements(int32_t max_batch, int32_t max_vocab,
                                             const SamplingPlan& plan,
                                             WorkspaceRequirements* out) {
  if (out == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "sampling_fused_workspace_requirements: out is null");
  }
  if (max_batch < 0 || max_vocab < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "sampling_fused_workspace_requirements: negative extent");
  }
  const bool filtered = (plan.flags == kPlanFiltered || plan.flags == (kPlanFiltered | kPlanToppOnly)) &&
                        plan.candidate_cap == kCand && plan.hist_bins == kBins;
  const bool plain = plan.flags == 0 && plan.candidate_cap == 0 && plan.hist_bins == 0;
  if (!valid_splits(plan.splits) || plan.block_threads != kThreads || !(filtered || plain)) {
    return detail::set_error(Status::kNotImplemented,
                             "sampling_fused_workspace_requirements: unsupported plan");
  }
  *out = WorkspaceRequirements{};
  // Split rows: per-row exchange areas (scratch) and one barrier counter per row (persistent,
  // zero-filled once before the first call).
  out->device_scratch_bytes = plan.splits > 1 ? static_cast<size_t>(max_batch) * split_row_bytes(plan.splits) : 0;
  out->device_persistent_bytes = plan.splits > 1 ? static_cast<size_t>(max_batch) * 8 : 0;
  out->host_pinned_bytes = 0;
  out->alignment = 256;
  return Status::kSuccess;
}

Status sample_fused(const Tensor& out_ids, const Tensor& logits,
                    const FusedSamplingParams& params, const SamplingPlan& plan,
                    const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "sample_fused: CUDA backend only");
  }
  const Status s = check_logits(logits);
  if (s != Status::kSuccess) return s;
  const Status o = check_tensor(out_ids, DType::kInt32, 1, 4, "sample_fused.out_ids");
  if (o != Status::kSuccess) return o;
  const int batch = static_cast<int>(logits.shape[0]);
  const int vocab = static_cast<int>(logits.shape[1]);
  if (out_ids.shape[0] != batch) {
    return detail::set_error(Status::kInvalidArgument,
                             "sample_fused: out_ids length mismatch");
  }
  if (batch > 1 && out_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "sample_fused: out_ids must be contiguous");
  }
  if (!same_plan(plan, expected_plan(params.sampling, vocab, ctx.capture_generation))) {
    return detail::set_error(Status::kPlanInvalidated,
                             "sample_fused: plan was resolved for other parameters");
  }
  if (!valid_splits(plan.splits)) {
    return detail::set_error(Status::kPlanInvalidated,
                             "sample_fused: splits %d is not a power of two <= 16", plan.splits);
  }
  if (plan.capture_generation != ctx.capture_generation) {
    return detail::set_error(Status::kPlanInvalidated,
                             "sample_fused: plan capture generation is stale");
  }
  if (params.update_penalty_mask != 0 && params.penalty_mask == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "sample_fused: update_penalty_mask requires penalty_mask");
  }
  if (params.sampling.temperature == nullptr &&
      !std::isfinite(params.sampling.default_temperature)) {
    return detail::set_error(Status::kInvalidArgument,
                             "sample_fused: non-finite default temperature");
  }
  const Status f = check_filters(params.sampling, vocab);
  if (f != Status::kSuccess) return f;
  if (batch == 0 || vocab == 0) return Status::kSuccess;

  const bool all_greedy = all_greedy_of(params.sampling);
  const bool filtered = filters_active(params.sampling, vocab);
  const bool p_only = p_only_of(params.sampling, vocab);
  if (plan.splits > 1) {
    if (!host_rows_vectorizable(logits)) {
      return detail::set_error(Status::kPlanInvalidated,
                               "sample_fused: split rows need 16-byte aligned rows and V a multiple of 16 B");
    }
    const int capacity =
        split_capacity(split_kernel_for(logits.dtype, all_greedy, filtered, p_only), 0);
    if (static_cast<int64_t>(batch) * plan.splits > capacity) {
      return detail::set_error(Status::kPlanInvalidated,
                               "sample_fused: %d x %d split CTAs exceed the %d co-resident CTAs", batch,
                               plan.splits, capacity);
    }
    const size_t scratch = static_cast<size_t>(batch) * split_row_bytes(plan.splits);
    const size_t persistent = static_cast<size_t>(batch) * 8;
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < scratch ||
        ws.device_persistent == nullptr || ws.device_persistent_bytes < persistent) {
      return detail::set_error(Status::kWorkspaceTooSmall,
                               "sample_fused: split plan needs %zu B scratch and %zu B persistent",
                               scratch, persistent);
    }
  }
  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_fused<float>(out_ids, logits, params, ctx, batch, vocab, all_greedy, filtered,
                                 p_only, plan, ws);
    case DType::kFloat16:
      return launch_fused<__half>(out_ids, logits, params, ctx, batch, vocab, all_greedy, filtered,
                                  p_only, plan, ws);
    case DType::kBFloat16:
      return launch_fused<__nv_bfloat16>(out_ids, logits, params, ctx, batch, vocab, all_greedy,
                                         filtered, p_only, plan, ws);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "sample_fused: dtype %s",
                               dtype_name(logits.dtype));
  }
}

}  // namespace kops::sample
