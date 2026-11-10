// =============================================================================
// kernel-ops | csrc/communication/allreduce.cu
// B12 rows 1-3: intra-node collectives over peer-mapped staging buffers.
//   peer_barrier_kernel     -> barrier_peer_kernel (+ epoch protocol shared by all collectives)
//   allreduce_kernel        -> allreduce_peer_kernel (one-shot), allreduce_two_shot_kernel (W = 1: a D2D copy)
//   allgather_kernel        -> allgather_peer_kernel
//   reduce_scatter_kernel   -> reduce_scatter_peer_kernel
//   allreduce_fusion_kernel -> fusion_peer_kernel (see the second half of this file)
//
// Transport model. Every rank owns one single-writer staging buffer (PeerBuffers::data[r]) and one
// signal area (PeerBuffers::flags[r]); both are only *pointers* to the kernel, so the same body runs
//   * on virtual ranks (N regions of one allocation on one GPU; tests, Mega MoE EP),
//   * over P2P / NVLink peer mappings, and
//   * over pinned host memory when the GPUs have no P2P (this repo's dev box: 2 x RTX 5060 Ti).
// Pull model: a rank copies its input into its own staging slot, publishes a flag, and reads the
// peers' slots after their flags arrive. No rank ever writes another rank's memory. A message above
// the single-phase threshold splits the two directions over separate CTAs of the same launch: one
// producer stages chunk by chunk and publishes a flag per chunk, the consumers read the peers' chunks
// (see run_pipeline). That is what keeps a read-only PCIe link busy in both directions at once.
//
// Epoch / graph replay. The call epoch is E = generation + calls + 1, read from the rank's private
// scratch at kernel start; the last CTA to finish advances `calls`, so a captured graph replays any
// number of times. The staging slot of a call is E & 1. One flag per phase is posted by the last CTA
// to finish staging (device-scope arrival counter), because a system-scope release costs ~1.5 us and
// the GPU serialises them across CTAs; a chunked call instead posts one release per chunk from its
// producer (at most kPipeMaxChunks per call), which is what buys the read/write overlap. All CTAs of a
// rank must be co-resident (the host never launches more than one wave); across ranks only the usual
// rule applies: ranks run concurrently. Slot reuse is
// safe without an end-of-call barrier: rank p posts its flag for call E only after its call E - 1
// finished, so a rank that observed that flag in the previous call (it needs it to read the data)
// knows the slot of call E - 2 is no longer being read.
//
// Numerics. Every algorithm accumulates in fp32 in ascending rank order and rounds once, so all
// ranks produce bitwise identical results independent of arrival order, grid and algorithm
// (kOneShot == kTwoShot == kDeterministic bit for bit). For two ranks the result equals NCCL's
// (a single fp32 add, one rounding).
//
// Waits are bounded: after the per-wait timeout (kScrTimeoutUs, default 20 s) the kernel writes
// CommError into kScrError, skips the data phase and returns; it never traps or hangs.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/communication/comm_gemm_kernels.cuh"
#include "csrc/communication/ops.h"

namespace kops::comm {
namespace {

constexpr uint32_t kDefaultTimeoutUs = 20u * 1000u * 1000u;

// ---- PTX helpers (system-scope so the same code serves peer GPUs and pinned host memory) -------
__device__ __forceinline__ uint32_t ld_acq_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t global_timer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
/// Peer / host data is read around the cache hierarchy: it changes under us between calls.
__device__ __forceinline__ uint4 ld_peer16(const void* p) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_local16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void st16(void* p, const uint4& v) { *reinterpret_cast<uint4*>(p) = v; }

// ---- vector <-> fp32 lanes ---------------------------------------------------------------------
template <typename T>
struct Lanes;
template <>
struct Lanes<__nv_bfloat16> {
  static constexpr int kN = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = __uint_as_float(w[i] << 16);
      f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162 h = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
};
template <>
struct Lanes<__half> {
  static constexpr int kN = 8;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __half2 h = *reinterpret_cast<const __half2*>(&w[i]);
      const float2 g = __half22float2(h);
      f[2 * i] = g.x;
      f[2 * i + 1] = g.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __half2 h = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
};
template <>
struct Lanes<float> {
  static constexpr int kN = 4;
  __device__ __forceinline__ static void unpack(const uint4& v, float* f) {
    f[0] = __uint_as_float(v.x);
    f[1] = __uint_as_float(v.y);
    f[2] = __uint_as_float(v.z);
    f[3] = __uint_as_float(v.w);
  }
  __device__ __forceinline__ static uint4 pack(const float* f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]),
                      __float_as_uint(f[3]));
  }
};

// ---- CTA-level protocol --------------------------------------------------------------------------
struct CtaState {
  uint32_t epoch;      // E of this call
  uint32_t counter;    // calls completed before this one (E = generation + counter + 1)
  uint32_t ok;         // 1 while no wait timed out
  uint32_t workers;    // CTAs that move data (gridDim.x, minus the poster CTA of a pipelined call)
  uint32_t seen[2];    // lowest peer flag value observed per flag word (skips redundant polls)
  uint64_t timeout_ns;
};

/// Flag value that says "chunk c (1-based) of call E is staged": monotone within and across calls.
__host__ __device__ constexpr uint32_t flag_value(uint32_t epoch, uint32_t chunk) { return (epoch << 8) | chunk; }

__device__ __forceinline__ uint32_t atom_add_acqrel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ uint32_t ld_vol(const uint32_t* p) { return *reinterpret_cast<const volatile uint32_t*>(p); }

/// Thread 0 reads the call counter and the timeout from the rank's private scratch.
__device__ __forceinline__ void cta_begin(const PeerBuffers& peers, uint32_t generation, CtaState* s) {
  if (threadIdx.x == 0) {
    const uint32_t cnt = ld_vol(peers.scratch + kScrCalls);
    const uint32_t tmo = ld_vol(peers.scratch + kScrTimeoutUs);
    s->counter = cnt;
    s->epoch = generation + cnt + 1u;
    s->workers = gridDim.x;
    s->seen[0] = 0u;
    s->seen[1] = 0u;
    s->timeout_ns = static_cast<uint64_t>(tmo != 0u ? tmo : kDefaultTimeoutUs) * 1000ull;
    s->ok = 1u;
  }
  __syncthreads();
}

/// All threads of the CTA have finished their staging stores; the last CTA to arrive publishes the
/// flag with one system-scope release (cumulative over the other CTAs' stores through the acq_rel
/// arrival counter) and resets the counter for the next call.
template <int kArrive, int kFlag>
__device__ __forceinline__ void cta_arrive_post(const PeerBuffers& peers, const CtaState& s) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t prev = atom_add_acqrel_gpu(peers.scratch + kArrive, 1u);
    if (prev + 1u == s.workers) {
      peers.scratch[kArrive] = 0u;
      st_rel_sys(peers.flags[peers.rank] + kFlag, flag_value(s.epoch, 1u));
    }
  }
}

/// Warp 0 waits until every peer's flag >= E (one lane per peer). On timeout it records CommError and
/// clears s->ok. Followed by a __syncthreads() so the whole CTA sees the result.
template <int kFlag>
__device__ __forceinline__ void cta_wait_peers(const PeerBuffers& peers, CtaState* s, int world, uint32_t chunk = 1u) {
  const uint32_t want = flag_value(s->epoch, chunk);
  const int widx = kFlag == kSigReady1 ? 0 : 1;
  // Always poll every chunk. The `seen` shortcut (skip a poll when the previous chunk's minimum
  // flag already covered this one) is not needed: the poster publishes chunks in order, but a
  // virtual-rank call with 8 ranks and several CTAs per rank can observe a mixed set of poster
  // states, and the shortcut then hid a peer that had not staged this chunk yet.
  if (s->ok != 0u && threadIdx.x < 32) {
    const int lane = threadIdx.x;
    bool ready = lane >= world || lane == peers.rank;
    const uint32_t* flag = lane < world ? peers.flags[lane] + kFlag : nullptr;
    const uint64_t t0 = global_timer_ns();
    uint32_t low = 0xFFFFFFFFu;  // flag value this lane's peer showed when it became ready
    for (;;) {
      if (!ready) {
        const uint32_t v = ld_acq_sys(flag);
        ready = static_cast<int32_t>(v - want) >= 0;
        low = ready ? v : low;
      }
      if (__all_sync(0xFFFFFFFFu, ready)) {
        // every peer has reached `want`; remember the lowest value seen so later chunks skip the poll
        const uint32_t m = __reduce_min_sync(0xFFFFFFFFu, low);
        if (lane == 0) s->seen[widx] = m == 0xFFFFFFFFu ? want : m;
        break;
      }
      const bool late = (global_timer_ns() - t0) > s->timeout_ns;
      if (__any_sync(0xFFFFFFFFu, late)) {
        if (lane == 0) {
          peers.scratch[kScrError] = static_cast<uint32_t>(CommError::kPeerTimeout);
          s->ok = 0u;
        }
        break;
      }
    }
  }
  __syncthreads();
}

/// The last CTA to finish advances the call counter (also after a timeout: the epoch of a failed
/// call is burned) and resets the finish counter.
__device__ __forceinline__ void cta_end(const PeerBuffers& peers, const CtaState& s) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t prev = atom_add_acqrel_gpu(peers.scratch + kScrDone, 1u);
    if (prev + 1u == gridDim.x) {
      peers.scratch[kScrDone] = 0u;
      peers.scratch[kScrCalls] = s.counter + 1u;
    }
  }
}

__device__ __forceinline__ char* slot_base(const PeerBuffers& peers, int rank, uint32_t epoch,
                                           size_t slot_bytes) {
  return static_cast<char*>(peers.data[rank]) + static_cast<size_t>(epoch & 1u) * slot_bytes;
}

/// Contiguous stripe [lo, hi) of `n` 16-byte vectors owned by this CTA.
__device__ __forceinline__ void cta_stripe(int64_t n, int64_t* lo, int64_t* hi, uint32_t workers) {
  const int64_t per = (n + workers - 1) / workers;
  const int64_t l = static_cast<int64_t>(blockIdx.x) * per;
  *lo = l < n ? l : n;
  *hi = (*lo + per) < n ? (*lo + per) : n;
}

/// Sub-range of [a, b) handled by this CTA.
__device__ __forceinline__ void cta_sub_stripe(int64_t a, int64_t b, int64_t* lo, int64_t* hi, uint32_t workers) {
  const int64_t n = b > a ? b - a : 0;
  const int64_t per = (n + workers - 1) / workers;
  const int64_t l = a + static_cast<int64_t>(blockIdx.x) * per;
  *lo = l < b ? l : b;
  *hi = (*lo + per) < b ? (*lo + per) : b;
}

/// Independent 16 B loads in flight per thread: the staging (write) and consume (read) loops of a
/// message above the single-phase threshold are PCIe-bound, so each thread keeps kStreamVectors
/// vectors in flight; a CTA of kCommThreads threads then holds ~32 KiB of requests.
constexpr int kStreamVectors = 8;
__host__ __device__ constexpr int unroll_for(int w) { return w <= 2 ? 8 : (w <= 4 ? 4 : 2); }

/// Streaming copy over the 16 B vectors of [lo, hi): kStreamVectors vectors in flight per thread, so
/// the loop keeps a full PCIe burst outstanding instead of one 16 B request per thread step.
template <typename Load, typename Emit>
__device__ __forceinline__ void stream_vectors(int64_t lo, int64_t hi, Load&& load, Emit&& emit) {
  for (int64_t i0 = lo + threadIdx.x; i0 < hi; i0 += static_cast<int64_t>(kCommThreads) * kStreamVectors) {
    uint4 v[kStreamVectors];
#pragma unroll
    for (int u = 0; u < kStreamVectors; ++u) {
      const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
      if (i < hi) v[u] = load(i);
    }
#pragma unroll
    for (int u = 0; u < kStreamVectors; ++u) {
      const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
      if (i < hi) emit(i, v[u]);
    }
  }
}

// ---- barrier -------------------------------------------------------------------------------------
template <int W>
__global__ void __launch_bounds__(kCommThreads)
    barrier_peer_kernel(const __grid_constant__ PeerBuffers peers, const uint32_t generation) {
  __shared__ CtaState st;
  cta_begin(peers, generation, &st);
  cta_arrive_post<kScrArrive1, kSigReady1>(peers, st);
  cta_wait_peers<kSigReady1>(peers, &st, W);
  cta_end(peers, st);
}

// ---- one-phase collectives, pipelined over chunks ---------------------------------------------------------
// allreduce (one-shot), allgather and reduce_scatter share one skeleton over the iteration space of
// `space16` 16-byte vectors.
//
// Small messages run in a single phase: every worker CTA stages its stripe into its own staging slot,
// the last arriving CTA posts one flag with a system-scope release, every CTA consumes its stripe.
//
// Above the chunk threshold the call is split into two roles that run in the same launch:
//   * blockIdx.x == 0 is the producer: it stages chunk c into this rank's slot and publishes chunk c
//     with one system-scope release per chunk;
//   * blockIdx.x > 0 are the consumers: each waits for every peer's chunk-c flag and then reduces /
//     gathers its stripe of chunk c.
// The reason is the host-staged transport (no P2P): a PCIe Gen4 x8 slot does ~12.5 GB/s in one
// direction, and a CTA that stages and then consumes alternates the two, so the read of chunk c only
// starts once the write of chunk c - 1 finished. A dedicated producer and consumer keep both
// directions busy; measured on 2 x RTX 5060 Ti this is ~18 GB/s of combined traffic instead of ~7.
// The protocol is also progress-safe by construction: a rank's producer never waits for a peer, so a
// rank can always publish, and only its consumers wait (bounded by kScrTimeoutUs like every other wait).
/// Producer: publish "chunk c of call E is staged". Preceded by __syncthreads() so the release covers
/// the stores of every thread of the CTA.
__device__ __forceinline__ void cta_post_chunk(const PeerBuffers& peers, const CtaState& s, uint32_t chunk) {
  st_rel_sys(peers.flags[peers.rank] + kSigReady1, flag_value(s.epoch, chunk));
}

template <int W, typename Stage, typename Consume>
__device__ __forceinline__ void run_pipeline(const PeerBuffers& peers, CtaState* st, int64_t space16, int chunks,
                                             int64_t chunk16, Stage&& stage, Consume&& consume) {
  auto chunk_range = [&](int c, int64_t* a, int64_t* b) {
    *a = static_cast<int64_t>(c) * chunk16;
    *b = (*a + chunk16) < space16 ? (*a + chunk16) : space16;
  };
  if (chunks > 1 && gridDim.x > 1) {
    if (blockIdx.x == 0) {
      for (int c = 0; c < chunks; ++c) {
        int64_t a, b;
        chunk_range(c, &a, &b);
        if (st->ok != 0u) stage(a, b);
        __syncthreads();
        if (threadIdx.x == 0) cta_post_chunk(peers, *st, static_cast<uint32_t>(c + 1));
      }
      return;
    }
    // Consumer stripe over the whole message; the per-chunk range is intersected with it. A consumer
    // then streams one long contiguous range (a quarter of the message at four consumers), which
    // measured faster here than giving every consumer a slice of every chunk.
    const int64_t consumers = gridDim.x - 1;
    const int64_t per = (space16 + consumers - 1) / consumers;
    const int64_t s_lo = static_cast<int64_t>(blockIdx.x - 1) * per;
    const int64_t s_hi = (s_lo + per) < space16 ? (s_lo + per) : space16;
    for (int c = 0; c < chunks; ++c) {
      cta_wait_peers<kSigReady1>(peers, st, W, static_cast<uint32_t>(c + 1));
      if (st->ok == 0u) continue;
      int64_t a, b;
      chunk_range(c, &a, &b);
      const int64_t lo = a > s_lo ? a : s_lo;
      const int64_t hi = b < s_hi ? b : s_hi;
      if (lo < hi) consume(lo, hi);
    }
    return;
  }
  int64_t lo, hi;
  cta_stripe(space16, &lo, &hi, st->workers);
  if (st->ok != 0u) stage(lo, hi);
  cta_arrive_post<kScrArrive1, kSigReady1>(peers, *st);
  cta_wait_peers<kSigReady1>(peers, st, W);
  if (st->ok != 0u) consume(lo, hi);
}

// ---- one-shot allreduce ---------------------------------------------------------------------------
// out = sum_p in_p. Every rank stages its input, then reduces the vectors of all peers locally.
template <typename T, int W>
__global__ void __launch_bounds__(kCommThreads)
    allreduce_peer_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                          const __grid_constant__ PeerBuffers peers, const int64_t n16, const int chunks,
                          const int64_t chunk16, const uint32_t generation, const size_t slot_bytes) {
  constexpr int kN = Lanes<T>::kN;
  __shared__ CtaState st;
  cta_begin(peers, generation, &st);
  uint4* mine = reinterpret_cast<uint4*>(slot_base(peers, peers.rank, st.epoch, slot_bytes));
  const uint4* src[W];
#pragma unroll
  for (int p = 0; p < W; ++p) src[p] = reinterpret_cast<const uint4*>(slot_base(peers, p, st.epoch, slot_bytes));
  run_pipeline<W>(
      peers, &st, n16, chunks, chunk16,
      [&](int64_t lo, int64_t hi) {
        stream_vectors(
            lo, hi, [&](int64_t i) { return ld_local16(in + i); },
            [&](int64_t i, const uint4& v) { st16(mine + i, v); });
      },
      [&](int64_t lo, int64_t hi) {
        constexpr int kUnroll = unroll_for(W);
        for (int64_t i0 = lo + threadIdx.x; i0 < hi; i0 += kCommThreads * kUnroll) {
          uint4 v[kUnroll][W];
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
            if (i < hi) {
#pragma unroll
              for (int p = 0; p < W; ++p)
                v[u][p] = (p == peers.rank) ? ld_local16(in + i) : ld_peer16(src[p] + i);
            }
          }
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
            if (i < hi) {
              float acc[kN], x[kN];
              Lanes<T>::unpack(v[u][0], acc);
#pragma unroll
              for (int p = 1; p < W; ++p) {
                Lanes<T>::unpack(v[u][p], x);
#pragma unroll
                for (int e = 0; e < kN; ++e) acc[e] += x[e];
              }
              st16(out + i, Lanes<T>::pack(acc));
            }
          }
        }
      });
  cta_end(peers, st);
}

// ---- two-shot allreduce (reduce-scatter + all-gather) ----------------------------------------------
// Rank r owns slice r of the message (global split, independent of the CTA count): it reduces that slice,
// publishes the result in place of the slice it never staged, and gathers the other slices from their
// owners.
template <typename T, int W>
__global__ void __launch_bounds__(kCommThreads)
    allreduce_two_shot_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                              const __grid_constant__ PeerBuffers peers, const int64_t n16,
                              const uint32_t generation, const size_t slot_bytes) {
  constexpr int kN = Lanes<T>::kN;
  __shared__ CtaState st;
  cta_begin(peers, generation, &st);
  const int rank = peers.rank;
  const int64_t S = (n16 + W - 1) / W;
  auto slice_lo = [&](int k) { const int64_t a = k * S; return a < n16 ? a : n16; };
  auto slice_hi = [&](int k) { const int64_t a = (k + 1) * S; return a < n16 ? a : n16; };
  const int64_t own_lo = slice_lo(rank), own_hi = slice_hi(rank);
  uint4* mine = reinterpret_cast<uint4*>(slot_base(peers, rank, st.epoch, slot_bytes));
  // phase 1: stage everything except the slice this rank reduces
  int64_t lo, hi;
  cta_stripe(n16, &lo, &hi, st.workers);
  for (int64_t i = lo + threadIdx.x; i < hi; i += kCommThreads) {
    if (i < own_lo || i >= own_hi) st16(mine + i, ld_local16(in + i));
  }
  cta_arrive_post<kScrArrive1, kSigReady1>(peers, st);
  cta_wait_peers<kSigReady1>(peers, &st, W);
  if (st.ok != 0u) {
    // reduce own slice in rank order; keep the result in the staging slice for the gatherers
    cta_sub_stripe(own_lo, own_hi, &lo, &hi, st.workers);
    for (int64_t i = lo + threadIdx.x; i < hi; i += kCommThreads) {
      float acc[kN], x[kN];
#pragma unroll
      for (int p = 0; p < W; ++p) {
        const uint4 v = (p == rank) ? ld_local16(in + i)
                                    : ld_peer16(reinterpret_cast<const uint4*>(
                                                    slot_base(peers, p, st.epoch, slot_bytes)) + i);
        if (p == 0) {
          Lanes<T>::unpack(v, acc);
        } else {
          Lanes<T>::unpack(v, x);
#pragma unroll
          for (int e = 0; e < kN; ++e) acc[e] += x[e];
        }
      }
      const uint4 r = Lanes<T>::pack(acc);
      st16(mine + i, r);
      st16(out + i, r);
    }
  }
  cta_arrive_post<kScrArrive2, kSigReady2>(peers, st);
  cta_wait_peers<kSigReady2>(peers, &st, W);
  if (st.ok != 0u) {
    for (int k = 0; k < W; ++k) {
      if (k == rank) continue;
      const uint4* src = reinterpret_cast<const uint4*>(slot_base(peers, k, st.epoch, slot_bytes));
      cta_sub_stripe(slice_lo(k), slice_hi(k), &lo, &hi, st.workers);
      for (int64_t i = lo + threadIdx.x; i < hi; i += kCommThreads) st16(out + i, ld_peer16(src + i));
    }
  }
  cta_end(peers, st);
}

// ---- all-gather -------------------------------------------------------------------------------------
template <typename T, int W>
__global__ void __launch_bounds__(kCommThreads)
    allgather_peer_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                          const __grid_constant__ PeerBuffers peers, const int64_t n16, const int chunks,
                          const int64_t chunk16, const uint32_t generation, const size_t slot_bytes) {
  __shared__ CtaState st;
  cta_begin(peers, generation, &st);
  const int rank = peers.rank;
  uint4* mine = reinterpret_cast<uint4*>(slot_base(peers, rank, st.epoch, slot_bytes));
  run_pipeline<W>(
      peers, &st, n16, chunks, chunk16,
      [&](int64_t lo, int64_t hi) {
        stream_vectors(
            lo, hi, [&](int64_t i) { return ld_local16(in + i); },
            [&](int64_t i, const uint4& v) {
              st16(mine + i, v);
              st16(out + static_cast<int64_t>(rank) * n16 + i, v);
            });
      },
      [&](int64_t lo, int64_t hi) {
        constexpr int kUnroll = unroll_for(W) * 2;
#pragma unroll
        for (int p = 0; p < W; ++p) {
          if (p == rank) continue;
          const uint4* src = reinterpret_cast<const uint4*>(slot_base(peers, p, st.epoch, slot_bytes));
          for (int64_t i0 = lo + threadIdx.x; i0 < hi; i0 += kCommThreads * kUnroll) {
            uint4 v[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
              const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
              if (i < hi) v[u] = ld_peer16(src + i);
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
              const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
              if (i < hi) st16(out + static_cast<int64_t>(p) * n16 + i, v[u]);
            }
          }
        }
      });
  cta_end(peers, st);
}

// ---- reduce-scatter -----------------------------------------------------------------------------------
// in: [W * m] vectors; out: [m] vectors = slice `rank` summed over all ranks.
template <typename T, int W>
__global__ void __launch_bounds__(kCommThreads)
    reduce_scatter_peer_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                               const __grid_constant__ PeerBuffers peers, const int64_t m16, const int chunks,
                               const int64_t chunk16, const uint32_t generation, const size_t slot_bytes) {
  constexpr int kN = Lanes<T>::kN;
  __shared__ CtaState st;
  cta_begin(peers, generation, &st);
  const int rank = peers.rank;
  uint4* mine = reinterpret_cast<uint4*>(slot_base(peers, rank, st.epoch, slot_bytes));
  const uint4* src[W];
#pragma unroll
  for (int p = 0; p < W; ++p) src[p] = reinterpret_cast<const uint4*>(slot_base(peers, p, st.epoch, slot_bytes));
  run_pipeline<W>(
      peers, &st, m16, chunks, chunk16,
      [&](int64_t lo, int64_t hi) {
        for (int k = 0; k < W; ++k) {
          if (k == rank) continue;
          const int64_t off = static_cast<int64_t>(k) * m16;
          stream_vectors(
              lo, hi, [&](int64_t i) { return ld_local16(in + off + i); },
              [&](int64_t i, const uint4& v) { st16(mine + off + i, v); });
        }
      },
      [&](int64_t lo, int64_t hi) {
        constexpr int kUnroll = unroll_for(W);
        for (int64_t i0 = lo + threadIdx.x; i0 < hi; i0 += kCommThreads * kUnroll) {
          uint4 v[kUnroll][W];
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
            if (i < hi) {
#pragma unroll
              for (int p = 0; p < W; ++p)
                v[u][p] = (p == rank) ? ld_local16(in + static_cast<int64_t>(rank) * m16 + i)
                                      : ld_peer16(src[p] + static_cast<int64_t>(rank) * m16 + i);
            }
          }
#pragma unroll
          for (int u = 0; u < kUnroll; ++u) {
            const int64_t i = i0 + static_cast<int64_t>(u) * kCommThreads;
            if (i < hi) {
              float acc[kN], x[kN];
              Lanes<T>::unpack(v[u][0], acc);
#pragma unroll
              for (int p = 1; p < W; ++p) {
                Lanes<T>::unpack(v[u][p], x);
#pragma unroll
                for (int e = 0; e < kN; ++e) acc[e] += x[e];
              }
              st16(out + i, Lanes<T>::pack(acc));
            }
          }
        }
      });
  cta_end(peers, st);
}

// ---- fused allreduce + residual add + RMSNorm (+ per-token FP8) ----------------------------------------------
// One kernel per TP layer boundary. The arithmetic is the chain `allreduce -> norm::fused_add_rms_norm_out
// -> quantize_activation(kDynamicPerToken)` expression for expression, and the sum of squares is
// accumulated in the same order (CH = 2 vectors per thread step, block size of the norm launcher, the
// same warp-shuffle + shared-memory tree), so the result matches the separate chain bit for bit:
//   x   = round_T(sum_ranks x_p)                    (fp32, ascending rank order)
//   rv  = r + x ; r' = round_T(rv) ; ss = sum rv^2  (rv unrounded, like the norm kernel)
//   n   = round_T(r' * rstd * (w + offset))         rstd = rsqrtf(ss / hidden + eps)
//   q   = e4m3(n / scale), scale = max(amax(n) / 448, 1 / (448 * 512))   (kDynamicPerToken)
// Rows are the pipeline iteration space: the CTAs stage their rows of the partial sums, count their
// arrival and normalise the rows of chunk c - 1 while chunk c drains.
__device__ __forceinline__ float fusion_block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();  // smem is reused by the next reduction
  return r;
}
__device__ __forceinline__ float fusion_block_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

/// Normalise one row. `reduce(vector index) -> uint4` returns the cross-rank sum of that 16-byte vector
/// of the row (rounded to T) -- or, for world_size 1, the vector itself.
template <typename T, int kQuant, typename ReduceVec>
__device__ __forceinline__ void fusion_row(const AllReduceFusionParams& p, int64_t row, T* srow,
                                           float* red, ReduceVec&& reduce_vec) {
  constexpr int VEC = Lanes<T>::kN;
  constexpr int CH = 2 * VEC;
  const int hidden = p.hidden;
  const int vph = hidden / VEC;                       // 16 B vectors per row
  uint4* res_row = reinterpret_cast<uint4*>(static_cast<T*>(p.residual_inout) + row * hidden);
  const T* w = static_cast<const T*>(p.norm_weight);
  float ss = 0.0f;
  const int nchunk = hidden / CH;
  for (int i = threadIdx.x; i < nchunk; i += blockDim.x) {
    const uint4 xv[2] = {reduce_vec(2 * i), reduce_vec(2 * i + 1)};
    uint4 rv[2] = {ld_local16(res_row + 2 * i), ld_local16(res_row + 2 * i + 1)};
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      T* rp = reinterpret_cast<T*>(&rv[u]);
      const T* xp = reinterpret_cast<const T*>(&xv[u]);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float v = to_f32<T>(rp[e]) + 1.0f * to_f32<T>(xp[e]);
        rp[e] = from_f32<T>(v);
        ss += v * v;
      }
    }
    st16(res_row + 2 * i, rv[0]);
    st16(res_row + 2 * i + 1, rv[1]);
    *reinterpret_cast<uint4*>(srow + static_cast<int64_t>(2 * i) * VEC) = rv[0];
    *reinterpret_cast<uint4*>(srow + static_cast<int64_t>(2 * i + 1) * VEC) = rv[1];
  }
  ss = fusion_block_sum(ss, red);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + p.norm_eps);
  float amax = 0.0f;
  for (int i = threadIdx.x; i < vph; i += blockDim.x) {
    const uint4 rq = *reinterpret_cast<const uint4*>(srow + static_cast<int64_t>(i) * VEC);
    const T* rp = reinterpret_cast<const T*>(&rq);
    uint4 ov;
    T* op = reinterpret_cast<T*>(&ov);
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
      const float wv = w != nullptr ? (to_f32<T>(w[i * VEC + e]) + p.weight_offset) : 1.0f;
      const float y = to_f32<T>(rp[e]) * rstd * wv;
      op[e] = from_f32<T>(y);
      if (kQuant != 0) amax = fmaxf(amax, fabsf(y));
    }
    if (p.norm_out != nullptr) st16(static_cast<T*>(p.norm_out) + (row * hidden + static_cast<int64_t>(i) * VEC), ov);
  }
  if (kQuant != 0) {
    // Per-token dynamic FP8: one absmax reduction over the normalized row, then a direct scalar
    // pass from the shared normalized row. There is no second staging buffer, so the dynamic
    // shared memory is the same for the plain and quantized epilogues.
    amax = fusion_block_max(amax, red);
    const float scale = fmaxf(__fdiv_rn(amax, kFp8E4M3Max), 1.0f / (kFp8E4M3Max * 512.0f));
    const float upper = p.quant.scale_upper_bound != nullptr ? p.quant.scale_upper_bound[0] : scale;
    const float applied = fminf(scale, upper);
    if (threadIdx.x == 0) static_cast<float*>(p.quant.scales)[row] = applied;
    uint8_t* q = static_cast<uint8_t*>(p.quant.q) + row * p.quant.q_row_stride;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float wv = w != nullptr ? (to_f32<T>(w[i]) + p.weight_offset) : 1.0f;
      const float y = to_f32<T>(srow[i]) * rstd * wv;
      q[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(__fdiv_rn(y, applied), __NV_SATFINITE, __NV_E4M3));
    }
    __syncthreads();
  }
}

template <typename T, int W, int kQuant>
__global__ void __launch_bounds__(512)
    fusion_peer_kernel(const __grid_constant__ AllReduceFusionParams p) {
  constexpr int kN = Lanes<T>::kN;
  extern __shared__ __align__(16) unsigned char fusion_smem[];
  __shared__ CtaState st;
  __shared__ float red[32];
  const PeerBuffers& peers = p.peers;
  // All ranks stage the whole message; the extra kernel arguments of the earlier chunked version
  // were not part of the public ABI and a struct-size mismatch between the host and device passes
  // shifted them. Deriving the two values from the descriptor keeps the kernel parameter a single
  // by-value struct (the shape the header declares) and is still graph-safe.
  const size_t slot_bytes = (peers.buffer_bytes / 2) & ~static_cast<size_t>(127);
  const int chunks = 1;
  const int64_t chunk_rows = p.num_tokens;
  cta_begin(peers, p.epoch, &st);
  const int hidden = p.hidden;
  const int vph = hidden / kN;
  T* srow = reinterpret_cast<T*>(fusion_smem);
  const uint4* in = static_cast<const uint4*>(p.input);
  uint4* mine = reinterpret_cast<uint4*>(slot_base(peers, peers.rank, st.epoch, slot_bytes));
  const uint4* src[W];
#pragma unroll
  for (int q = 0; q < W; ++q) src[q] = reinterpret_cast<const uint4*>(slot_base(peers, q, st.epoch, slot_bytes));
  run_pipeline<W>(
      peers, &st, p.num_tokens, chunks, chunk_rows,
      [&](int64_t lo, int64_t hi) {
        const int64_t v0 = lo * vph, v1 = hi * vph;
        for (int64_t i = v0 + threadIdx.x; i < v1; i += blockDim.x) st16(mine + i, ld_local16(in + i));
      },
      [&](int64_t lo, int64_t hi) {
        for (int64_t row = lo; row < hi; ++row) {
          const int64_t base = row * vph;
          fusion_row<T, kQuant>(p, row, srow, red, [&](int vi) -> uint4 {
            float acc[kN], x[kN];
#pragma unroll
            for (int q = 0; q < W; ++q) {
              const uint4 v = (q == peers.rank) ? ld_local16(in + base + vi) : ld_peer16(src[q] + base + vi);
              if (q == 0) {
                Lanes<T>::unpack(v, acc);
              } else {
                Lanes<T>::unpack(v, x);
#pragma unroll
                for (int e = 0; e < kN; ++e) acc[e] += x[e];
              }
            }
            return Lanes<T>::pack(acc);
          });
        }
      });
  cta_end(peers, st);
}

/// world_size == 1: no communication, one CTA per row (the roadmap's fusion_w1_kernel).
template <typename T, int kQuant>
__global__ void __launch_bounds__(512) fusion_w1_kernel(const __grid_constant__ AllReduceFusionParams p) {
  constexpr int kN = Lanes<T>::kN;
  extern __shared__ __align__(16) unsigned char fusion_smem[];
  __shared__ float red[32];
  const int vph = p.hidden / kN;
  T* srow = reinterpret_cast<T*>(fusion_smem);
  const uint4* in = static_cast<const uint4*>(p.input);
  const int64_t row = blockIdx.x;
  fusion_row<T, kQuant>(p, row, srow, red, [&](int vi) -> uint4 { return ld_local16(in + row * vph + vi); });
}

// ---- host side ------------------------------------------------------------------------------------------
Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: launch failed: %s",
                                     what, cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

constexpr bool supported_world(int w) { return w == 1 || w == 2 || w == 4 || w == 8; }

int elem_bytes(DType t) {
  switch (t) {
    case DType::kBFloat16:
    case DType::kFloat16:
      return 2;
    case DType::kFloat32:
      return 4;
    default:
      return 0;
  }
}

size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/// Structural checks shared by every collective. `slot_payload` is the bytes one slot must hold.
Status check_peers(const PeerBuffers& p, size_t slot_payload, const char* who) {
  if (p.world_size < 1 || p.world_size > kMaxRanks || p.rank < 0 || p.rank >= p.world_size) {
    return detail::set_error(Status::kInvalidArgument, "%s: rank %d / world_size %d", who, p.rank,
                             p.world_size);
  }
  if (!supported_world(p.world_size)) {
    return detail::set_error(Status::kUnsupportedShape, "%s: world_size %d (compiled: 1, 2, 4, 8)", who,
                             p.world_size);
  }
  if (p.num_nodes != 1) {
    return detail::set_error(Status::kUnsupportedBackend,
                             "%s: num_nodes %d; only the intra-node transport is implemented", who,
                             p.num_nodes);
  }
  if (p.world_size == 1) return Status::kSuccess;
  if (p.scratch == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: PeerBuffers::scratch is null", who);
  }
  for (int r = 0; r < p.world_size; ++r) {
    if (p.data[r] == nullptr || p.flags[r] == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "%s: data[%d] / flags[%d] is null", who, r, r);
    }
    if ((reinterpret_cast<uintptr_t>(p.data[r]) & (kStagingAlign - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(p.flags[r]) & 15u) != 0) {
      return detail::set_error(Status::kMisaligned, "%s: staging of rank %d must be 128 B aligned", who,
                               r);
    }
  }
  const size_t half = (p.buffer_bytes / 2) & ~(kStagingAlign - 1);
  if (half < slot_payload) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "%s: buffer_bytes %zu < %zu needed (two slots of %zu)", who, p.buffer_bytes,
                             2 * round_up(slot_payload, kStagingAlign), round_up(slot_payload, kStagingAlign));
  }
  return Status::kSuccess;
}

size_t slot_bytes_of(const PeerBuffers& p) { return (p.buffer_bytes / 2) & ~(kStagingAlign - 1); }

/// CTAs of one call: a message of a few KiB is latency bound and one CTA keeps the number of system
/// fences at one; large messages need many CTAs to keep enough PCIe / NVLink requests in flight.
/// All CTAs of a rank must be co-resident (they wait for each other through the arrival counters).
int pick_ctas(size_t bytes, const LaunchContext& ctx) {
  // Measured on 2 x PCIe Gen4 x8 (host-staged): every extra CTA adds arrival traffic and the poster's
  // fence waits for the writes of all of them, so 4 workers saturate the link; 1-2 serve small messages.
  int n = bytes <= (32u << 10) ? 1 : bytes <= (128u << 10) ? 2 : 4;
  const int sms = ctx.compute_units > 0 ? ctx.compute_units : 1;
  if (n > sms) n = sms;
  return n < 1 ? 1 : n;
}

/// Chunking of a one-phase collective. A message above 128 KiB is cut into at most kPipeMaxChunks
/// chunks of at least kPipeMinChunk bytes and runs as a producer/consumer pipeline (see run_pipeline);
/// `workers` stays the phase-oriented CTA count the two-shot kernel uses.
constexpr int kPipeConsumers = 4;          // consumer CTAs of a chunked call (one producer + these)
constexpr size_t kPipeMinChunk = 128u << 10;
constexpr size_t kPipeMaxChunks = 32;
struct Pipe {
  int workers;
  int chunks;
  int64_t chunk16;
  int grid;
};
Pipe plan_pipe(size_t bytes, const LaunchContext& ctx) {
  Pipe p{};
  p.workers = pick_ctas(bytes, ctx);
  if (const char* e = std::getenv("KOPS_COMM_WORKERS")) p.workers = std::atoi(e);  // TUNING-ONLY
  p.chunks = 1;
  const int64_t n16 = static_cast<int64_t>(bytes / 16);
  p.chunk16 = n16;
  size_t thresh = 128u << 10;  // TUNING-ONLY env overrides below
  size_t minchunk = kPipeMinChunk;
  size_t maxchunks = kPipeMaxChunks;
  if (const char* e = std::getenv("KOPS_COMM_THRESH_KB")) thresh = size_t(std::atoi(e)) << 10;
  if (const char* e = std::getenv("KOPS_COMM_CHUNK_KB")) minchunk = size_t(std::atoi(e)) << 10;
  if (const char* e = std::getenv("KOPS_COMM_MAXCHUNKS")) maxchunks = size_t(std::atoi(e));
  if (maxchunks > 255) maxchunks = 255;  // the chunk number is the low byte of a flag value
  if (bytes > thresh) {
    size_t chunk_bytes = (bytes + maxchunks - 1) / maxchunks;
    if (chunk_bytes < minchunk) chunk_bytes = minchunk;
    chunk_bytes = (chunk_bytes + 15) & ~size_t(15);
    p.chunks = static_cast<int>((bytes + chunk_bytes - 1) / chunk_bytes);
    p.chunk16 = static_cast<int64_t>(chunk_bytes / 16);
  }
  p.grid = p.workers;
  if (p.chunks > 1) {
    int consumers = kPipeConsumers;
    if (const char* e = std::getenv("KOPS_COMM_CONSUMERS")) consumers = std::atoi(e);  // TUNING-ONLY
    if (consumers < 1) consumers = 1;
    const int sms = ctx.compute_units > 0 ? ctx.compute_units : 1;
    if (consumers + 1 > sms) consumers = sms > 1 ? sms - 1 : 1;
    p.grid = consumers + 1;
  }
  return p;
}

/// A chunked call needs its producer and its consumers co-resident; one producer plus a couple of
/// consumers per rank is small enough for every device the rank model supports (the producer never
/// waits for a peer, so a late consumer delays only its own rank). Unchunked messages keep the
/// bandwidth-oriented worker count. The two-shot kernel is phase-oriented (each phase moves data in a
/// single direction) and does not pipeline, so it keeps one CTA per phase above the threshold.
Pipe plan_pipe_world(size_t bytes, const LaunchContext& ctx, int world) {
  Pipe p = plan_pipe(bytes, ctx);
  if (p.chunks > 1) p.workers = 1;
  (void)world;
  return p;
}

template <typename T, int W>
Status launch_allreduce(AllReduceAlgo algo, void* out, const void* in, const PeerBuffers& peers,
                        int64_t n16, uint32_t gen, const Pipe& pp, cudaStream_t s) {
  const size_t slot = slot_bytes_of(peers);
  if (algo == AllReduceAlgo::kOneShot || algo == AllReduceAlgo::kDeterministic) {
    // kDeterministic is the fixed rank-order reduction every algorithm here already performs; the
    // one-shot body is the lowest-latency instance of it.
    allreduce_peer_kernel<T, W><<<pp.grid, kCommThreads, 0, s>>>(
        static_cast<uint4*>(out), static_cast<const uint4*>(in), peers, n16, pp.chunks, pp.chunk16, gen, slot);
  } else {
    allreduce_two_shot_kernel<T, W><<<pp.workers, kCommThreads, 0, s>>>(
        static_cast<uint4*>(out), static_cast<const uint4*>(in), peers, n16, gen, slot);
  }
  return launch_error("allreduce");
}

template <typename T>
Status dispatch_allreduce_world(int world, AllReduceAlgo algo, void* out, const void* in,
                                const PeerBuffers& peers, int64_t n16, uint32_t gen, const Pipe& pp, cudaStream_t s) {
  switch (world) {
    case 2: return launch_allreduce<T, 2>(algo, out, in, peers, n16, gen, pp, s);
    case 4: return launch_allreduce<T, 4>(algo, out, in, peers, n16, gen, pp, s);
    case 8: return launch_allreduce<T, 8>(algo, out, in, peers, n16, gen, pp, s);
    default: return detail::set_error(Status::kUnsupportedShape, "allreduce: world_size %d", world);
  }
}

Status check_flat(const Tensor& t, DType dtype, const char* name, int64_t* numel) {
  Status s = check_tensor(t, t.dtype, t.ndim, 16, name);
  if (s != Status::kSuccess) return s;
  if (t.dtype != dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s != %s", name, dtype_name(t.dtype),
                             dtype_name(dtype));
  }
  if (!tensor_is_contiguous(t)) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: collectives need a contiguous tensor", name);
  }
  *numel = tensor_numel(t);
  return Status::kSuccess;
}

// ---- fusion host side ---------------------------------------------------------------------------------------
int fusion_block(int hidden, int vec) {
  int n = (hidden + vec - 1) / vec;
  n = ((n + 31) / 32) * 32;
  if (n < 64) n = 64;
  if (n > 512) n = 512;
  return n;  // the launcher of csrc/norm picks the same block, which fixes the order of the sum of squares
}

template <typename T, int W, int kQuant>
Status launch_fusion(const AllReduceFusionParams& p, const LaunchContext& ctx, cudaStream_t s) {
  constexpr int kN = Lanes<T>::kN;
  const size_t row_bytes = static_cast<size_t>(p.hidden) * sizeof(T);
  const size_t smem = static_cast<size_t>(p.hidden) * sizeof(T);
  const int block = fusion_block(p.hidden, kN);
  if (smem > 48u * 1024u) {
    return detail::set_error(Status::kUnsupportedShape, "allreduce_fusion: hidden %d needs %zu B of shared memory", p.hidden, smem);
  }
  if (W == 1) {
    fusion_w1_kernel<T, kQuant><<<p.num_tokens, block, smem, s>>>(p);
    return launch_error("allreduce_fusion(world=1)");
  }
  const size_t bytes = row_bytes * static_cast<size_t>(p.num_tokens);
  Pipe pp = plan_pipe_world(bytes, ctx, p.peers.world_size);
  // One pipeline stage: the body stages the whole message with every worker CTA and then reduces
  // row stripes. The kernel parameter is the public AllReduceFusionParams by value; the chunk
  // counters stay in the body (see fusion_peer_kernel).
  int workers = std::min<int>(pp.workers, p.num_tokens);
  if (workers < 1) workers = 1;
  fusion_peer_kernel<T, W, kQuant><<<workers, block, smem, s>>>(p);
  return launch_error("allreduce_fusion");
}

template <typename T, int kQuant>
Status dispatch_fusion_world(const AllReduceFusionParams& p, const LaunchContext& ctx, cudaStream_t s) {
  switch (p.peers.world_size) {
    case 1: return launch_fusion<T, 1, kQuant>(p, ctx, s);
    case 2: return launch_fusion<T, 2, kQuant>(p, ctx, s);
    case 4: return launch_fusion<T, 4, kQuant>(p, ctx, s);
    default: return launch_fusion<T, 8, kQuant>(p, ctx, s);
  }
}

template <typename T>
Status dispatch_fusion_quant(const AllReduceFusionParams& p, const LaunchContext& ctx, cudaStream_t s) {
  return p.quant.mode == ActQuantMode::kDynamicPerToken ? dispatch_fusion_world<T, 1>(p, ctx, s)
                                                       : dispatch_fusion_world<T, 0>(p, ctx, s);
}

}  // namespace

// ---- public: sizes / configuration ----------------------------------------------------------------------
size_t allreduce_staging_bytes(AllReduceAlgo algo, DType dtype, int64_t numel, int32_t world_size) {
  const int eb = elem_bytes(dtype);
  if (eb == 0 || numel < 0 || !supported_world(world_size)) return 0;
  if (algo == AllReduceAlgo::kMultimem || algo == AllReduceAlgo::kQuickReduce ||
      algo == AllReduceAlgo::kLamportOneShot) {
    return 0;
  }
  if (world_size == 1) return 0;
  return 2 * round_up(static_cast<size_t>(numel) * eb, kStagingAlign);
}

size_t allgather_staging_bytes(DType dtype, int64_t numel_per_rank, int32_t world_size) {
  const int eb = elem_bytes(dtype);
  if (eb == 0 || numel_per_rank < 0 || !supported_world(world_size)) return 0;
  if (world_size == 1) return 0;
  return 2 * round_up(static_cast<size_t>(numel_per_rank) * eb, kStagingAlign);
}

size_t allreduce_fusion_staging_bytes(DType dtype, int32_t num_tokens, int32_t hidden, int32_t world_size) {
  const int eb = elem_bytes(dtype);
  if (eb == 0 || num_tokens < 0 || hidden <= 0 || !supported_world(world_size)) return 0;
  if (world_size == 1) return 0;
  return 2 * round_up(static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden) * eb, kStagingAlign);
}

size_t reduce_scatter_staging_bytes(DType dtype, int64_t numel, int32_t world_size) {
  const int eb = elem_bytes(dtype);
  if (eb == 0 || numel < 0 || !supported_world(world_size)) return 0;
  if (world_size == 1) return 0;
  return 2 * round_up(static_cast<size_t>(numel) * eb, kStagingAlign);
}

Status validate_peer_buffers(const PeerBuffers& peers, size_t required_bytes) {
  Status s = check_peers(peers, required_bytes / 2, "validate_peer_buffers");
  if (s != Status::kSuccess || peers.world_size == 1) return s;
  int cur = 0;
  cudaGetDevice(&cur);
  for (int r = 0; r < peers.world_size; ++r) {
    cudaPointerAttributes attr{};
    for (int k = 0; k < 2; ++k) {
      const void* ptr = k == 0 ? static_cast<const void*>(peers.data[r]) : static_cast<const void*>(peers.flags[r]);
      if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError();
        return detail::set_error(Status::kCommError, "validate_peer_buffers: rank %d %s is not CUDA-visible",
                                 r, k == 0 ? "staging" : "signal area");
      }
      if (attr.type == cudaMemoryTypeHost) {
        if (attr.devicePointer == nullptr) {
          return detail::set_error(Status::kCommError,
                                   "validate_peer_buffers: rank %d %s is host memory that is not mapped "
                                   "(cudaHostAllocMapped | cudaHostAllocPortable)",
                                   r, k == 0 ? "staging" : "signal area");
        }
      } else if (attr.type == cudaMemoryTypeDevice && attr.device != cur) {
        int can = 0;
        cudaDeviceCanAccessPeer(&can, cur, attr.device);
        if (can == 0) {
          return detail::set_error(Status::kCommError,
                                   "validate_peer_buffers: device %d cannot access rank %d memory on "
                                   "device %d (no P2P); stage through pinned host memory instead",
                                   cur, r, attr.device);
        }
      }
    }
  }
  cudaPointerAttributes sa{};
  if (cudaPointerGetAttributes(&sa, peers.scratch) != cudaSuccess) {
    cudaGetLastError();
    return detail::set_error(Status::kCommError, "validate_peer_buffers: scratch is not CUDA-visible");
  }
  if (sa.type != cudaMemoryTypeDevice || sa.device != cur) {
    return detail::set_error(Status::kCommError,
                             "validate_peer_buffers: scratch must be device memory of the current device");
  }
  if (peers.multicast_ptr != nullptr) {
    return detail::set_error(Status::kUnsupportedBackend,
                             "validate_peer_buffers: multicast (NVLS) objects are not supported");
  }
  return Status::kSuccess;
}

namespace {
bool is_host_ptr(const void* p) {
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
    cudaGetLastError();
    return true;
  }
  return a.type == cudaMemoryTypeHost || a.type == cudaMemoryTypeUnregistered;
}
}  // namespace

Status comm_configure(const PeerBuffers& peers, uint32_t timeout_us) {
  if (peers.world_size < 1 || peers.rank < 0 || peers.rank >= peers.world_size) {
    return detail::set_error(Status::kInvalidArgument, "comm_configure: rank %d / world_size %d", peers.rank,
                             peers.world_size);
  }
  if (peers.world_size == 1) return Status::kSuccess;
  uint32_t* sig = peers.flags[peers.rank];
  if (sig == nullptr || peers.scratch == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "comm_configure: flags[rank] / scratch is null");
  }
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(sync),
                                     "comm_configure: %s", cudaGetErrorString(sync));
  }
  uint32_t zeros[kSignalWords] = {};
  uint32_t scr[kScratchWords] = {};
  scr[kScrTimeoutUs] = timeout_us;
  cudaError_t e = cudaMemcpy(peers.scratch, scr, sizeof(scr), cudaMemcpyHostToDevice);
  if (e == cudaSuccess) {
    if (is_host_ptr(sig)) {
      std::memcpy(sig, zeros, sizeof(zeros));
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    } else {
      e = cudaMemcpy(sig, zeros, sizeof(zeros), cudaMemcpyHostToDevice);
    }
  }
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e), "comm_configure: %s",
                                     cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

Status comm_error(const PeerBuffers& peers, int32_t* error_code) {
  if (error_code == nullptr) return detail::set_error(Status::kInvalidArgument, "comm_error: null out");
  *error_code = 0;
  if (peers.world_size <= 1) return Status::kSuccess;
  if (peers.scratch == nullptr) return detail::set_error(Status::kInvalidArgument, "comm_error: scratch is null");
  uint32_t v = 0;
  const cudaError_t e = cudaMemcpy(&v, peers.scratch + kScrError, sizeof(v), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e), "comm_error: %s",
                                     cudaGetErrorString(e));
  }
  *error_code = static_cast<int32_t>(v);
  return Status::kSuccess;
}

// ---- public: collectives ------------------------------------------------------------------------------------
Status peer_barrier(const PeerBuffers& peers, uint32_t epoch, const LaunchContext& ctx) {
  Status s = check_peers(peers, 0, "peer_barrier");
  if (s != Status::kSuccess || peers.world_size == 1) return s;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (peers.world_size) {
    case 2: barrier_peer_kernel<2><<<1, kCommThreads, 0, st>>>(peers, epoch); break;
    case 4: barrier_peer_kernel<4><<<1, kCommThreads, 0, st>>>(peers, epoch); break;
    default: barrier_peer_kernel<8><<<1, kCommThreads, 0, st>>>(peers, epoch); break;
  }
  return launch_error("peer_barrier");
}

Status allreduce(const Tensor& out, const Tensor& in, const PeerBuffers& peers, AllReduceAlgo algo,
                 uint32_t epoch, const LaunchContext& ctx) {
  int64_t n_in = 0, n_out = 0;
  Status s = check_flat(in, in.dtype, "allreduce.in", &n_in);
  if (s != Status::kSuccess) return s;
  s = check_flat(out, in.dtype, "allreduce.out", &n_out);
  if (s != Status::kSuccess) return s;
  const int eb = elem_bytes(in.dtype);
  if (eb == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "allreduce: dtype %s (bf16, fp16, fp32)",
                             dtype_name(in.dtype));
  }
  if (n_in != n_out) {
    return detail::set_error(Status::kInvalidArgument, "allreduce: out has %lld elements, in has %lld",
                             static_cast<long long>(n_out), static_cast<long long>(n_in));
  }
  if ((static_cast<size_t>(n_in) * eb) % 16 != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "allreduce: numel %lld is not a multiple of %d elements (16 B vectors)",
                             static_cast<long long>(n_in), 16 / eb);
  }
  switch (algo) {
    case AllReduceAlgo::kOneShot:
    case AllReduceAlgo::kTwoShot:
    case AllReduceAlgo::kDeterministic:
      break;
    case AllReduceAlgo::kMultimem:
      return detail::set_error(Status::kUnsupportedBackend,
                               "allreduce: kMultimem needs an NVSwitch multicast object (none bound)");
    case AllReduceAlgo::kLamportOneShot:
    case AllReduceAlgo::kQuickReduce:
      return detail::set_error(Status::kNotImplemented, "allreduce: algo %d is not implemented",
                               static_cast<int>(algo));
  }
  const size_t bytes = static_cast<size_t>(n_in) * eb;
  s = check_peers(peers, bytes, "allreduce");
  if (s != Status::kSuccess) return s;
  if (n_in == 0) return Status::kSuccess;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const uintptr_t pin = reinterpret_cast<uintptr_t>(in.data) + static_cast<uintptr_t>(in.byte_offset);
  const uintptr_t pout = reinterpret_cast<uintptr_t>(out.data) + static_cast<uintptr_t>(out.byte_offset);
  if (peers.world_size == 1) {
    if (pin != pout) {
      const cudaError_t e = cudaMemcpyAsync(reinterpret_cast<void*>(pout), reinterpret_cast<void*>(pin), bytes,
                                            cudaMemcpyDeviceToDevice, st);
      if (e != cudaSuccess) {
        return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                         "allreduce(world=1): %s", cudaGetErrorString(e));
      }
    }
    return Status::kSuccess;
  }
  if (pin == pout) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "allreduce: in-place is not supported (peers still read the input stripe)");
  }
  void* o = reinterpret_cast<void*>(pout);
  const void* i = reinterpret_cast<const void*>(pin);
  const int64_t n16 = static_cast<int64_t>(bytes / 16);
  switch (in.dtype) {
    case DType::kBFloat16:
      return dispatch_allreduce_world<__nv_bfloat16>(peers.world_size, algo, o, i, peers, n16, epoch, plan_pipe_world(bytes, ctx, peers.world_size), st);
    case DType::kFloat16:
      return dispatch_allreduce_world<__half>(peers.world_size, algo, o, i, peers, n16, epoch, plan_pipe_world(bytes, ctx, peers.world_size), st);
    default:
      return dispatch_allreduce_world<float>(peers.world_size, algo, o, i, peers, n16, epoch, plan_pipe_world(bytes, ctx, peers.world_size), st);
  }
}

Status allgather(const Tensor& out, const Tensor& in, const PeerBuffers& peers, uint32_t epoch,
                 const LaunchContext& ctx) {
  int64_t n_in = 0, n_out = 0;
  Status s = check_flat(in, in.dtype, "allgather.in", &n_in);
  if (s != Status::kSuccess) return s;
  s = check_flat(out, in.dtype, "allgather.out", &n_out);
  if (s != Status::kSuccess) return s;
  const int eb = elem_bytes(in.dtype);
  if (eb == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "allgather: dtype %s (bf16, fp16, fp32)",
                             dtype_name(in.dtype));
  }
  if (n_out != n_in * peers.world_size) {
    return detail::set_error(Status::kInvalidArgument,
                             "allgather: out has %lld elements, want world_size * %lld",
                             static_cast<long long>(n_out), static_cast<long long>(n_in));
  }
  if ((static_cast<size_t>(n_in) * eb) % 16 != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "allgather: per-rank numel %lld is not a multiple of %d elements",
                             static_cast<long long>(n_in), 16 / eb);
  }
  const size_t bytes = static_cast<size_t>(n_in) * eb;
  s = check_peers(peers, bytes, "allgather");
  if (s != Status::kSuccess) return s;
  if (n_in == 0) return Status::kSuccess;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const uintptr_t pin = reinterpret_cast<uintptr_t>(in.data) + static_cast<uintptr_t>(in.byte_offset);
  const uintptr_t pout = reinterpret_cast<uintptr_t>(out.data) + static_cast<uintptr_t>(out.byte_offset);
  if (peers.world_size == 1) {
    if (pin != pout) {
      cudaMemcpyAsync(reinterpret_cast<void*>(pout), reinterpret_cast<void*>(pin), bytes, cudaMemcpyDeviceToDevice, st);
    }
    return launch_error("allgather(world=1)");
  }
  const int64_t n16 = static_cast<int64_t>(bytes / 16);
  const size_t slot = slot_bytes_of(peers);
  const Pipe pp = plan_pipe_world(bytes, ctx, peers.world_size);
  uint4* o = reinterpret_cast<uint4*>(pout);
  const uint4* i = reinterpret_cast<const uint4*>(pin);
#define KOPS_AG_LAUNCH(T)                                                                          \
  switch (peers.world_size) {                                                                      \
    case 2: allgather_peer_kernel<T, 2><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, n16, pp.chunks, pp.chunk16, epoch, slot); break; \
    case 4: allgather_peer_kernel<T, 4><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, n16, pp.chunks, pp.chunk16, epoch, slot); break; \
    default: allgather_peer_kernel<T, 8><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, n16, pp.chunks, pp.chunk16, epoch, slot); break; \
  }
  switch (in.dtype) {
    case DType::kBFloat16: KOPS_AG_LAUNCH(__nv_bfloat16) break;
    case DType::kFloat16: KOPS_AG_LAUNCH(__half) break;
    default: KOPS_AG_LAUNCH(float) break;
  }
#undef KOPS_AG_LAUNCH
  return launch_error("allgather");
}

Status reduce_scatter(const Tensor& out, const Tensor& in, const PeerBuffers& peers, uint32_t epoch,
                      const LaunchContext& ctx) {
  int64_t n_in = 0, n_out = 0;
  Status s = check_flat(in, in.dtype, "reduce_scatter.in", &n_in);
  if (s != Status::kSuccess) return s;
  s = check_flat(out, in.dtype, "reduce_scatter.out", &n_out);
  if (s != Status::kSuccess) return s;
  const int eb = elem_bytes(in.dtype);
  if (eb == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "reduce_scatter: dtype %s (bf16, fp16, fp32)",
                             dtype_name(in.dtype));
  }
  const int world = peers.world_size;
  if (world < 1 || n_in % world != 0 || n_out * world != n_in) {
    return detail::set_error(Status::kInvalidArgument,
                             "reduce_scatter: in has %lld elements, out %lld; need in = world_size * out",
                             static_cast<long long>(n_in), static_cast<long long>(n_out));
  }
  if ((static_cast<size_t>(n_out) * eb) % 16 != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "reduce_scatter: per-rank numel %lld is not a multiple of %d elements",
                             static_cast<long long>(n_out), 16 / eb);
  }
  const size_t bytes = static_cast<size_t>(n_in) * eb;
  s = check_peers(peers, bytes, "reduce_scatter");
  if (s != Status::kSuccess) return s;
  if (n_in == 0) return Status::kSuccess;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const uintptr_t pin = reinterpret_cast<uintptr_t>(in.data) + static_cast<uintptr_t>(in.byte_offset);
  const uintptr_t pout = reinterpret_cast<uintptr_t>(out.data) + static_cast<uintptr_t>(out.byte_offset);
  if (world == 1) {
    if (pin != pout) {
      cudaMemcpyAsync(reinterpret_cast<void*>(pout), reinterpret_cast<void*>(pin), bytes, cudaMemcpyDeviceToDevice, st);
    }
    return launch_error("reduce_scatter(world=1)");
  }
  const int64_t m16 = static_cast<int64_t>(n_out) * eb / 16;
  const size_t slot = slot_bytes_of(peers);
  const Pipe pp = plan_pipe_world(static_cast<size_t>(n_out) * eb, ctx, world);
  uint4* o = reinterpret_cast<uint4*>(pout);
  const uint4* i = reinterpret_cast<const uint4*>(pin);
#define KOPS_RS_LAUNCH(T)                                                                          \
  switch (world) {                                                                                 \
    case 2: reduce_scatter_peer_kernel<T, 2><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, m16, pp.chunks, pp.chunk16, epoch, slot); break; \
    case 4: reduce_scatter_peer_kernel<T, 4><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, m16, pp.chunks, pp.chunk16, epoch, slot); break; \
    default: reduce_scatter_peer_kernel<T, 8><<<pp.grid, kCommThreads, 0, st>>>(o, i, peers, m16, pp.chunks, pp.chunk16, epoch, slot); break; \
  }
  switch (in.dtype) {
    case DType::kBFloat16: KOPS_RS_LAUNCH(__nv_bfloat16) break;
    case DType::kFloat16: KOPS_RS_LAUNCH(__half) break;
    default: KOPS_RS_LAUNCH(float) break;
  }
#undef KOPS_RS_LAUNCH
  return launch_error("reduce_scatter");
}

Status allreduce_fusion(const AllReduceFusionParams& params, AllReduceAlgo algo, DType dtype, const LaunchContext& ctx) {
  const int eb = elem_bytes(dtype);
  if (eb == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "allreduce_fusion: dtype %s (bf16, fp16, fp32)", dtype_name(dtype));
  }
  if (params.num_tokens < 0 || params.hidden <= 0) {
    return detail::set_error(Status::kInvalidArgument, "allreduce_fusion: num_tokens %d, hidden %d", params.num_tokens,
                             params.hidden);
  }
  const int vec = 16 / eb;
  if (params.hidden % (2 * vec) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "allreduce_fusion: hidden %d must be a multiple of %d (the norm kernel's 2-vector step)",
                             params.hidden, 2 * vec);
  }
  if (params.input == nullptr || params.residual_inout == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "allreduce_fusion: input / residual_inout is null");
  }
  if (params.norm_out == nullptr && params.quant.mode == ActQuantMode::kNone) {
    return detail::set_error(Status::kInvalidArgument, "allreduce_fusion: need norm_out or a quantized output");
  }
  const void* const checked[] = {params.input, params.residual_inout, params.norm_out, params.norm_weight};
  for (const void* ptr : checked) {
    if (ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & 15u) != 0) {
      return detail::set_error(Status::kMisaligned, "allreduce_fusion: pointers must be 16 B aligned");
    }
  }
  switch (algo) {
    case AllReduceAlgo::kOneShot:
    case AllReduceAlgo::kTwoShot:  // the fused body is the one-shot pipeline for every message size
    case AllReduceAlgo::kDeterministic:
      break;
    case AllReduceAlgo::kMultimem:
      return detail::set_error(Status::kUnsupportedBackend, "allreduce_fusion: kMultimem needs a multicast object");
    default:
      return detail::set_error(Status::kNotImplemented, "allreduce_fusion: algo %d is not implemented", static_cast<int>(algo));
  }
  if (params.quant.mode != ActQuantMode::kNone) {
    if (params.quant.mode != ActQuantMode::kDynamicPerToken) {
      return detail::set_error(Status::kNotImplemented,
                               "allreduce_fusion: quant mode %d (only kDynamicPerToken)",
                               static_cast<int>(params.quant.mode));
    }
    if (params.quant.q_dtype != DType::kFloat8E4M3 || params.quant.fp8 != Fp8Encoding::kOCP ||
        !params.quant.symmetric) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "allreduce_fusion: per-token quantization is OCP e4m3, symmetric");
    }
    if (params.quant.q == nullptr || params.quant.scales == nullptr ||
        params.quant.q_row_stride < params.hidden) {
      return detail::set_error(Status::kInvalidArgument,
                               "allreduce_fusion: quant.q / scales null or q_row_stride < hidden");
    }
    if ((reinterpret_cast<uintptr_t>(params.quant.scales) & 3u) != 0) {
      return detail::set_error(Status::kMisaligned, "allreduce_fusion: quant.scales must be 4 B aligned");
    }
  }
  const size_t bytes = static_cast<size_t>(params.num_tokens) * static_cast<size_t>(params.hidden) * eb;
  Status s = check_peers(params.peers, bytes, "allreduce_fusion");
  if (s != Status::kSuccess) return s;
  if (params.num_tokens == 0) return Status::kSuccess;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (dtype) {
    case DType::kBFloat16: return dispatch_fusion_quant<__nv_bfloat16>(params, ctx, st);
    case DType::kFloat16: return dispatch_fusion_quant<__half>(params, ctx, st);
    default: return dispatch_fusion_quant<float>(params, ctx, st);
  }
}

}  // namespace kops::comm
