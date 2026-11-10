// =============================================================================
// kernel-ops | csrc/communication/comm_gemm.cu
// B12 row 6: tensor-parallel GEMMs overlapped with the collectives.
//   allgather_gemm_kernel      -> comm_gemm_allgather_kernel
//   gemm_reduce_scatter_kernel -> comm_gemm_reduce_scatter_kernel
//   gemm_allreduce_kernel      -> comm_gemm_allreduce_kernel
//
// The GEMM body is the mma.sync m16n8k16 mainloop used by B7 (bf16/fp16 atoms,
// fp32 accumulation), specialised here to a 64x32 CTA tile with eight warps.
// The communication is chunked along M. A chunk is owned by a small group of
// CTAs (up to 8 for world 2); the last local CTA of the group publishes
// `chunk_ready[rank * num_chunks + chunk] = epoch + 1` as soon as all of the
// group's staging/partial writes are visible. A consumer polls the
// same flag for every rank before touching the peer slot, so the first tiles of
// chunk c can run while chunk c + 1 is still being produced.
//
//   allgather_gemm      A is the local M shard [m_local, K].  Each rank stages
//                       the local rows of every global chunk, then computes the
//                       full D [m_local * world, N] from its own + peer shards.
//   gemm_reduce_scatter Each rank computes its partial D [M, N] chunk by chunk;
//                       the owner rank sums the peer partials into its M slice
//                       out [M / world, N].
//   gemm_allreduce      Same partial GEMM, every rank reduces each chunk into
//                       its own D [M, N] (optionally + residual + RMSNorm).
//
// Staging bytes per rank (PeerBuffers::buffer_bytes / 2, 128 B aligned):
//   allgather_gemm      m_local * K * sizeof(T)
//   reduce_scatter /    M * N * sizeof(float)          (fp32 partials)
//   allreduce
// Flags live in CommGemmParams::chunk_ready [world * num_chunks] and are
// epoch-tagged, so a captured graph replays without a host reset. Waits are
// bounded by PeerBuffers::scratch (CommError::kPeerTimeout) and never trap.
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

constexpr int kTileM = 64;   // CTA tile (4 warps along M * 16 rows)
constexpr int kTileN = 32;   // CTA tile (2 warps along N * 16 cols)
constexpr int kThreads = 256;
constexpr size_t kStageAlign = 128;
constexpr int kScrComBase = 8;  // per-chunk arrival counters live above the collective scratch words
constexpr uint64_t kDefaultTimeoutNs = 20ull * 1000ull * 1000ull * 1000ull;

enum class CommGemmMode : int { kAllGather = 0, kReduceScatter = 1, kAllReduce = 2 };

__device__ __forceinline__ uint32_t cg_ld_acq_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void cg_st_rel_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t cg_timer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t cg_atom_add_acqrel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
/// The last CTA of the call advances the on-device call counter, so a captured graph replays with a
/// fresh epoch tag without a host update (same rule as the collectives of allreduce.cu).
__device__ __forceinline__ void cg_end(const PeerBuffers& p, uint32_t counter) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t prev = cg_atom_add_acqrel_gpu(p.scratch + kScrDone, 1u);
    if (prev + 1u == gridDim.x) {
      p.scratch[kScrDone] = 0u;
      p.scratch[kScrCalls] = counter + 1u;
    }
  }
}

/// One flag per chunk is posted by the last CTA of that chunk's local group. The cumulative
/// acq_rel arrival counter carries every CTA's writes to the poster; the poster then makes them
/// visible with a system-scope release.
__device__ __forceinline__ void cg_post_chunk(uint32_t* flag, uint32_t* counter, int group,
                                              uint32_t tag) {
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) {
    const uint32_t prev = cg_atom_add_acqrel_gpu(counter, 1u);
    if (prev + 1u == static_cast<uint32_t>(group)) {
      *counter = 0u;
      __threadfence_system();
      cg_st_rel_sys(flag, tag);
    }
  }
}

// ---- mma.sync atom ----------------------------------------------------------------------------
template <typename T>
struct CgMma;
template <>
struct CgMma<__nv_bfloat16> {
  __device__ __forceinline__ static void run(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)c;(void)a;(void)b;
    __trap();
#endif
  }
};
template <>
struct CgMma<__half> {
  __device__ __forceinline__ static void run(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)c;(void)a;(void)b;
    __trap();
#endif
  }
};

__device__ __forceinline__ uint32_t cg_ld2(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

template <typename T>
__device__ __forceinline__ T cg_cast(float v);
template <> __device__ __forceinline__ __nv_bfloat16 cg_cast<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <> __device__ __forceinline__ __half cg_cast<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float cg_cast<float>(float v) { return v; }

template <typename T>
__device__ __forceinline__ float cg_f32(T v);
template <> __device__ __forceinline__ float cg_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ __forceinline__ float cg_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float cg_f32<float>(float v) { return v; }

/// One 64x32 CTA tile. `a_row(r)` returns the row pointer of the (possibly remote) A shard and
/// `b_row(n)` the row pointer of this rank's B. Only the (row, col) pairs inside [m, n) are stored.
template <typename T, typename D, typename ARow, typename BRow>
__device__ __forceinline__ void cg_tile(ARow&& a_row, BRow&& b_row, const gemm::EpilogueParams& epi,
                                        D* d, int64_t ldd, int64_t row0, int64_t col0, int64_t m,
                                        int64_t n, int64_t k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp >= 8) return;
  const int wm = warp >> 1;   // 4 warps along M
  const int wn = warp & 1;    // 2 warps along N
  const int64_t r = row0 + static_cast<int64_t>(wm) * 16;
  const int64_t c = col0 + static_cast<int64_t>(wn) * 16;
  const int g = lane >> 2;
  const int t = lane & 3;

  float acc[2][4] = {};
  const int64_t ra0 = (r + g) < m ? (r + g) : (m - 1);
  const int64_t ra1 = (r + g + 8) < m ? (r + g + 8) : (m - 1);
  for (int64_t k0 = 0; k0 < k; k0 += 16) {
    const T* pa0 = a_row(ra0) + k0 + t * 2;
    const T* pa1 = a_row(ra1) + k0 + t * 2;
    uint32_t a[4] = {cg_ld2(pa0), cg_ld2(pa1), cg_ld2(pa0 + 8), cg_ld2(pa1 + 8)};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int64_t nc = c + static_cast<int64_t>(half) * 8 + g;
      const int64_t ncl = nc < n ? nc : (n - 1);
      const T* pb = b_row(ncl) + k0 + t * 2;
      uint32_t b[2] = {cg_ld2(pb), cg_ld2(pb + 8)};
      CgMma<T>::run(acc[half], a, b);
    }
  }

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int64_t cc = c + static_cast<int64_t>(half) * 8 + t * 2;
    const int64_t r0 = r + g;
    const int64_t r1 = r + g + 8;
    if (r0 < m && cc < n) {
      float v0 = acc[half][0], v1 = acc[half][1];
      if ((epi.ops & gemm::kEpiBias) != 0u && epi.bias != nullptr) {
        v0 += cg_f32(*reinterpret_cast<const T*>(static_cast<const char*>(epi.bias) + cc * sizeof(T)));
        if (cc + 1 < n)
          v1 += cg_f32(*reinterpret_cast<const T*>(static_cast<const char*>(epi.bias) + (cc + 1) * sizeof(T)));
      }
      d[r0 * ldd + cc] = cg_cast<D>(v0);
      if (cc + 1 < n) d[r0 * ldd + cc + 1] = cg_cast<D>(v1);
    }
    if (r1 < m && cc < n) {
      float v0 = acc[half][2], v1 = acc[half][3];
      if ((epi.ops & gemm::kEpiBias) != 0u && epi.bias != nullptr) {
        v0 += cg_f32(*reinterpret_cast<const T*>(static_cast<const char*>(epi.bias) + cc * sizeof(T)));
        if (cc + 1 < n)
          v1 += cg_f32(*reinterpret_cast<const T*>(static_cast<const char*>(epi.bias) + (cc + 1) * sizeof(T)));
      }
      d[r1 * ldd + cc] = cg_cast<D>(v0);
      if (cc + 1 < n) d[r1 * ldd + cc + 1] = cg_cast<D>(v1);
    }
  }
}

__device__ __forceinline__ size_t cg_slot_bytes(const PeerBuffers& p) {
  return (p.buffer_bytes / 2) & ~(kStageAlign - 1);
}
__device__ __forceinline__ char* cg_slot(const PeerBuffers& p, int rank, uint32_t epoch) {
  return static_cast<char*>(p.data[rank]) + static_cast<size_t>(epoch & 1u) * cg_slot_bytes(p);
}

/// Main chunk body. MODE is a compile-time CommGemmMode. One chunk is handled by `group` CTAs;
/// each CTA owns a subset of the chunk's (m tile, n tile) tasks. The last CTA of the group posts
/// the chunk flag after its local writes, so the peer stride and one-flag-per-chunk contract stay
/// unchanged while the GEMM uses the whole device.
template <typename T, typename D, int W, CommGemmMode kMode>
__global__ void __launch_bounds__(kThreads)
    cg_kernel(const __grid_constant__ CommGemmParams p) {
  using namespace gemm;
  const ScaledGemmArgs& a = p.gemm.args;
  const int64_t k = a.k, n = a.n;
  const int64_t m_local = a.m;
  const int64_t m_total = (kMode == CommGemmMode::kAllGather) ? m_local * W : a.m;
  const int num_chunks = p.num_chunks;
  const int group = static_cast<int>(gridDim.x) / num_chunks;
  const int chunk = static_cast<int>(blockIdx.x) / group;
  const int cta = static_cast<int>(blockIdx.x) % group;
  const int64_t chunk_rows = m_total / num_chunks;
  const int64_t g0 = static_cast<int64_t>(chunk) * chunk_rows;
  const int64_t g1 = g0 + chunk_rows;
  const int rank = p.peers.rank;
  const uint32_t call_base =
      (W > 1) ? *reinterpret_cast<volatile const uint32_t*>(p.peers.scratch + kScrCalls) : 0u;
  const uint32_t epoch_tag = p.epoch + call_base + 1u;
  char* const staging = cg_slot(p.peers, rank, epoch_tag);
  const T* A = static_cast<const T*>(a.a.data);
  const T* B = static_cast<const T*>(a.b.data);
  D* Dp = static_cast<D*>(a.d);
  const int64_t lda = a.a.ld, ldb = a.b.ld, ldd = a.ldd;
  uint32_t* flag = nullptr;
  uint32_t* counter = nullptr;
  if constexpr (W > 1) {
    flag = p.chunk_ready + static_cast<int64_t>(rank) * num_chunks + chunk;
    counter = p.peers.scratch + kScrComBase + chunk;
  }
  const int tiles_n = static_cast<int>((n + kTileN - 1) / kTileN);
  const int tiles = static_cast<int>((chunk_rows + kTileM - 1) / kTileM) * tiles_n;

  if constexpr (W == 1) {
    auto a_row = [&](int64_t gr) -> const T* { return A + gr * lda; };
    auto b_row = [&](int64_t nc) -> const T* { return B + nc * ldb; };
    for (int tile = cta; tile < tiles; tile += group) {
      const int64_t mi = static_cast<int64_t>(tile / tiles_n) * kTileM;
      const int64_t ni = static_cast<int64_t>(tile % tiles_n) * kTileN;
      cg_tile<T, D>(a_row, b_row, a.epi, Dp, ldd, g0 + mi, ni, g1, n, k);
    }
    return;
  }

  // ---------------------------------------------------------------------------------------------
  // allgather: stage the local shard rows of this chunk, wait for every peer, compute the chunk.
  // ---------------------------------------------------------------------------------------------
  if (kMode == CommGemmMode::kAllGather) {
    const int64_t own_lo = g0 > static_cast<int64_t>(rank) * m_local ? g0
                                                                     : static_cast<int64_t>(rank) * m_local;
    const int64_t own_hi = g1 < static_cast<int64_t>(rank + 1) * m_local
                               ? g1
                               : static_cast<int64_t>(rank + 1) * m_local;
    T* sd = reinterpret_cast<T*>(staging);
    const int64_t rows = own_hi > own_lo ? own_hi - own_lo : 0;
    for (int64_t i = cta; i < rows; i += group) {
      const int64_t gr = own_lo + i;
      const int64_t local = gr - static_cast<int64_t>(rank) * m_local;
      for (int64_t c = threadIdx.x; c < k; c += blockDim.x) sd[local * k + c] = A[local * lda + c];
    }
    cg_post_chunk(flag, counter, group, epoch_tag);

    __shared__ uint32_t ok;
    if (threadIdx.x == 0) {
      const uint64_t t0 = cg_timer_ns();
      ok = 1u;
      for (int s = 0; s < W && ok != 0u; ++s) {
        const uint32_t* f = p.chunk_ready + static_cast<int64_t>(s) * num_chunks + chunk;
        for (;;) {
          if (static_cast<int32_t>(cg_ld_acq_sys(f) - epoch_tag) >= 0) break;
          if (cg_timer_ns() - t0 > kDefaultTimeoutNs) {
            p.peers.scratch[kScrError] = static_cast<uint32_t>(CommError::kPeerTimeout);
            ok = 0u;
            break;
          }
        }
      }
    }
    __syncthreads();
    if (ok == 0u) { cg_end(p.peers, call_base); return; }
    auto a_row = [&](int64_t gr) -> const T* {
      const int owner = static_cast<int>(gr / m_local);
      const int64_t local = gr - static_cast<int64_t>(owner) * m_local;
      if (owner == rank) return A + local * lda;
      const char* peer = cg_slot(p.peers, owner, epoch_tag);
      return reinterpret_cast<const T*>(peer) + local * k;
    };
    auto b_row = [&](int64_t nc) -> const T* { return B + nc * ldb; };
    for (int tile = cta; tile < tiles; tile += group) {
      const int64_t mi = static_cast<int64_t>(tile / tiles_n) * kTileM;
      const int64_t ni = static_cast<int64_t>(tile % tiles_n) * kTileN;
      cg_tile<T, D>(a_row, b_row, a.epi, Dp, ldd, g0 + mi, ni, g1, n, k);
    }
    cg_end(p.peers, call_base);
    return;
  }

  // ---------------------------------------------------------------------------------------------
  // reduce_scatter / allreduce: local partial GEMM, one flag per chunk, then reduce the chunk.
  // ---------------------------------------------------------------------------------------------
  float* part = reinterpret_cast<float*>(staging) + g0 * n;
  auto a_row_local = [&](int64_t gr) -> const T* { return A + (g0 + gr) * lda; };
  auto b_row = [&](int64_t nc) -> const T* { return B + nc * ldb; };
  for (int tile = cta; tile < tiles; tile += group) {
    const int64_t mi = static_cast<int64_t>(tile / tiles_n) * kTileM;
    const int64_t ni = static_cast<int64_t>(tile % tiles_n) * kTileN;
    cg_tile<T, float>(a_row_local, b_row, gemm::EpilogueParams{}, part, n, mi, ni, chunk_rows, n, k);
  }
  cg_post_chunk(flag, counter, group, epoch_tag);
  if (kMode == CommGemmMode::kReduceScatter) {
    const int64_t slice = m_total / W;
    const int owner = static_cast<int>(g0 / slice);
    if (owner != rank) { cg_end(p.peers, call_base); return; }
  }

  __shared__ uint32_t ok2;
  if (threadIdx.x == 0) {
    const uint64_t t0 = cg_timer_ns();
    ok2 = 1u;
    for (int s = 0; s < W && ok2 != 0u; ++s) {
      const uint32_t* f = p.chunk_ready + static_cast<int64_t>(s) * num_chunks + chunk;
      for (;;) {
        if (static_cast<int32_t>(cg_ld_acq_sys(f) - epoch_tag) >= 0) break;
        if (cg_timer_ns() - t0 > kDefaultTimeoutNs) {
          p.peers.scratch[kScrError] = static_cast<uint32_t>(CommError::kPeerTimeout);
          ok2 = 0u;
          break;
        }
      }
    }
  }
  __syncthreads();
  if (ok2 == 0u) { cg_end(p.peers, call_base); return; }

  const int64_t out_base = (kMode == CommGemmMode::kReduceScatter)
                               ? (g0 - static_cast<int64_t>(rank) * (m_total / W))
                               : g0;
  for (int tile = cta; tile < tiles; tile += group) {
    const int64_t mi = static_cast<int64_t>(tile / tiles_n) * kTileM;
    const int64_t ni = static_cast<int64_t>(tile % tiles_n) * kTileN;
    const int64_t rend = (mi + kTileM) < chunk_rows ? (mi + kTileM) : chunk_rows;
    const int64_t cend = (ni + kTileN) < n ? (ni + kTileN) : n;
    for (int64_t r = mi; r < rend; ++r) {
      const int64_t gr = g0 + r;
      const int64_t orow = out_base + r;
      for (int64_t c = ni + threadIdx.x; c < cend; c += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < W; ++s) {
          const float* peer = reinterpret_cast<const float*>(cg_slot(p.peers, s, epoch_tag));
          v += peer[gr * n + c];
        }
        if ((a.epi.ops & gemm::kEpiBias) != 0u && a.epi.bias != nullptr) {
          v += cg_f32(*reinterpret_cast<const T*>(static_cast<const char*>(a.epi.bias) + c * sizeof(T)));
        }
        if (kMode == CommGemmMode::kAllReduce && p.fuse_residual_rmsnorm && p.residual != nullptr) {
          const T* res = static_cast<const T*>(p.residual);
          v += cg_f32(res[gr * n + c]);
        }
        Dp[orow * ldd + c] = cg_cast<D>(v);
      }
    }
  }
  cg_end(p.peers, call_base);
}

// ---- fused residual norm (helper launched by gemm_allreduce after the reduction) ---------------
template <typename T>
__global__ void __launch_bounds__(256)
    cg_norm_kernel(T* __restrict__ d, int64_t rows, int64_t cols, const T* __restrict__ w,
                   float eps, int64_t ldd) {
  __shared__ float red[32];
  const int64_t row = blockIdx.x;
  float ss = 0.0f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = cg_f32(d[row * ldd + c]);
    ss += v * v;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float t = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rstd = rsqrtf(red[0] / static_cast<float>(cols) + eps);
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float wv = w != nullptr ? cg_f32(w[c]) : 1.0f;
    d[row * ldd + c] = cg_cast<T>(cg_f32(d[row * ldd + c]) * rstd * wv);
  }
}

// ---- host -------------------------------------------------------------------------------------
template <typename T>
Status cg_launch(const CommGemmParams& p, int mode, cudaStream_t s) {
  using gemm::GemmRecipe;
  const int w = p.peers.world_size;
  const int chunks = p.num_chunks;
  const gemm::ScaledGemmArgs& a = p.gemm.args;
  const int64_t m_total = (mode == 0) ? a.m * w : a.m;
  const int64_t chunk_rows = m_total / chunks;
  const int tiles_m = static_cast<int>((chunk_rows + kTileM - 1) / kTileM);
  const int tiles_n = static_cast<int>((a.n + kTileN - 1) / kTileN);
  const int target_ctas = (w <= 2) ? 64 : 32;
  int group = std::max(1, target_ctas / chunks);
  group = std::min(group, w <= 2 ? 8 : 4);
  group = std::min(group, std::max(1, tiles_m * tiles_n));
  if (w == 1) group = 1;
  dim3 grid(static_cast<unsigned>(chunks * group), 1, 1);
  switch (mode) {
    case 0:
      switch (w) {
        case 1: cg_kernel<T, T, 1, CommGemmMode::kAllGather><<<grid, kThreads, 0, s>>>(p); break;
        case 2: cg_kernel<T, T, 2, CommGemmMode::kAllGather><<<grid, kThreads, 0, s>>>(p); break;
        case 4: cg_kernel<T, T, 4, CommGemmMode::kAllGather><<<grid, kThreads, 0, s>>>(p); break;
        default: cg_kernel<T, T, 8, CommGemmMode::kAllGather><<<grid, kThreads, 0, s>>>(p); break;
      }
      break;
    case 1:
      switch (w) {
        case 1: cg_kernel<T, T, 1, CommGemmMode::kReduceScatter><<<grid, kThreads, 0, s>>>(p); break;
        case 2: cg_kernel<T, T, 2, CommGemmMode::kReduceScatter><<<grid, kThreads, 0, s>>>(p); break;
        case 4: cg_kernel<T, T, 4, CommGemmMode::kReduceScatter><<<grid, kThreads, 0, s>>>(p); break;
        default: cg_kernel<T, T, 8, CommGemmMode::kReduceScatter><<<grid, kThreads, 0, s>>>(p); break;
      }
      break;
    default:
      switch (w) {
        case 1: cg_kernel<T, T, 1, CommGemmMode::kAllReduce><<<grid, kThreads, 0, s>>>(p); break;
        case 2: cg_kernel<T, T, 2, CommGemmMode::kAllReduce><<<grid, kThreads, 0, s>>>(p); break;
        case 4: cg_kernel<T, T, 4, CommGemmMode::kAllReduce><<<grid, kThreads, 0, s>>>(p); break;
        default: cg_kernel<T, T, 8, CommGemmMode::kAllReduce><<<grid, kThreads, 0, s>>>(p); break;
      }
      break;
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                     "comm_gemm: launch failed: %s", cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

/// Structural validation of the common GEMM parameters.
Status cg_check(const CommGemmParams& p, gemm::GemmRecipe recipe, int mode, const LaunchContext& ctx) {
  const gemm::ScaledGemmArgs& a = p.gemm.args;
  if (recipe != gemm::GemmRecipe::kBF16 && recipe != gemm::GemmRecipe::kFP16) {
    return detail::set_error(Status::kNotImplemented, "comm_gemm: recipe %d (bf16, fp16)",
                             static_cast<int>(recipe));
  }
  if (a.m <= 0 || a.n <= 0 || a.k <= 0 || a.batch != 1) {
    return detail::set_error(Status::kInvalidArgument, "comm_gemm: m %lld n %lld k %lld batch %d",
                             static_cast<long long>(a.m), static_cast<long long>(a.n),
                             static_cast<long long>(a.k), a.batch);
  }
  if (a.a.data == nullptr || a.b.data == nullptr || a.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "comm_gemm: null A/B/D");
  }
  if (a.a.major != gemm::MajorOrder::kKMajor || a.b.major != gemm::MajorOrder::kKMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "comm_gemm: operands must be K-major");
  }
  if (a.a.scale != nullptr || a.b.scale != nullptr || a.a.packing != WeightPacking::kNone ||
      a.b.packing != WeightPacking::kNone) {
    return detail::set_error(Status::kNotImplemented, "comm_gemm: quantized/packed operands are not implemented");
  }
  if ((a.k % 16) != 0) {
    return detail::set_error(Status::kUnsupportedShape, "comm_gemm: K %lld must be a multiple of 16",
                             static_cast<long long>(a.k));
  }
  if (a.a.ld < a.k || a.b.ld < a.k || a.ldd < a.n) {
    return detail::set_error(Status::kInvalidArgument, "comm_gemm: leading dimensions smaller than K/N");
  }
  const DType want = recipe == gemm::GemmRecipe::kBF16 ? DType::kBFloat16 : DType::kFloat16;
  if (a.a.dtype != want || a.b.dtype != want || a.d_dtype != want) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "comm_gemm: recipe %d needs A/B/D = %s", static_cast<int>(recipe),
                             dtype_name(want));
  }
  if ((a.epi.ops & gemm::kEpiBias) != 0u && a.epi.bias_dtype != want) {
    return detail::set_error(Status::kUnsupportedDtype, "comm_gemm: bias dtype must match the output dtype");
  }
  const uint32_t allowed = gemm::kEpiNone | gemm::kEpiBias;
  if ((a.epi.ops & ~allowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "comm_gemm: epilogue ops 0x%x",
                             static_cast<unsigned>(a.epi.ops));
  }
  if ((a.epi.ops & gemm::kEpiBias) != 0u && a.epi.bias == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "comm_gemm: kEpiBias without bias");
  }
  if (p.num_chunks < 1 || p.num_chunks % p.peers.world_size != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "comm_gemm: num_chunks %d must be a multiple of world_size %d", p.num_chunks,
                             p.peers.world_size);
  }
  if (p.num_chunks + kScrComBase > kScratchWords) {
    return detail::set_error(Status::kUnsupportedShape,
                             "comm_gemm: num_chunks %d exceeds the %d per-chunk counters in scratch",
                             p.num_chunks, kScratchWords - kScrComBase);
  }
  const int64_t m_total = mode == 0 ? a.m * p.peers.world_size : a.m;
  if (m_total % p.num_chunks != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "comm_gemm: M %lld is not divisible by num_chunks %d",
                             static_cast<long long>(m_total), p.num_chunks);
  }
  if (mode == 1 && (a.m % p.peers.world_size) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "comm_gemm: reduce_scatter needs M divisible by world_size");
  }
  const int w = p.peers.world_size;
  if (w < 1 || w > kMaxRanks || !(w == 1 || w == 2 || w == 4 || w == 8)) {
    return detail::set_error(Status::kUnsupportedShape, "comm_gemm: world_size %d", w);
  }
  if (p.peers.rank < 0 || p.peers.rank >= w) {
    return detail::set_error(Status::kInvalidArgument, "comm_gemm: rank %d / world %d", p.peers.rank, w);
  }
  if (p.peers.num_nodes != 1) {
    return detail::set_error(Status::kUnsupportedBackend, "comm_gemm: only the intra-node transport is implemented");
  }
  if (ctx.arch_id > 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "comm_gemm: mma.sync needs SM80 or newer (arch %d)",
                             ctx.arch_id);
  }
  if (w > 1) {
    if (p.chunk_ready == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "comm_gemm: chunk_ready is null");
    }
    if (p.peers.scratch == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "comm_gemm: PeerBuffers::scratch is null");
    }
    for (int r = 0; r < w; ++r) {
      if (p.peers.data[r] == nullptr) {
        return detail::set_error(Status::kInvalidArgument, "comm_gemm: peers.data[%d] is null", r);
      }
      if ((reinterpret_cast<uintptr_t>(p.peers.data[r]) & (kStageAlign - 1)) != 0) {
        return detail::set_error(Status::kMisaligned, "comm_gemm: peers.data[%d] must be 128 B aligned", r);
      }
    }
    const size_t slot = (p.peers.buffer_bytes / 2) & ~(kStageAlign - 1);
    const size_t need = mode == 0 ? static_cast<size_t>(a.m) * static_cast<size_t>(a.k) * sizeof(uint16_t)
                                  : static_cast<size_t>(m_total) * static_cast<size_t>(a.n) * sizeof(float);
    if (slot < need) {
      return detail::set_error(Status::kWorkspaceTooSmall,
                               "comm_gemm: staging slot %zu B < %zu B needed", slot, need);
    }
  }
  if (p.fuse_residual_rmsnorm) {
    if (mode != 2) return detail::set_error(Status::kInvalidArgument, "comm_gemm: fused RMSNorm is GEMM+AR only");
    if (p.residual == nullptr || p.norm_weight == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "comm_gemm: fused RMSNorm needs residual and weight");
    }
    if (p.norm_quant.mode != ActQuantMode::kNone) {
      return detail::set_error(Status::kNotImplemented, "comm_gemm: quantized normalized output");
    }
  }
  return Status::kSuccess;
}

Status cg_dispatch(const CommGemmParams& p, gemm::GemmRecipe recipe, int mode, const LaunchContext& ctx) {
  Status s = cg_check(p, recipe, mode, ctx);
  if (s != Status::kSuccess) return s;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (recipe) {
    case gemm::GemmRecipe::kBF16: return cg_launch<__nv_bfloat16>(p, mode, st);
    case gemm::GemmRecipe::kFP16: return cg_launch<__half>(p, mode, st);
    default: return detail::set_error(Status::kNotImplemented, "comm_gemm: recipe");
  }
}

}  // namespace

Status allgather_gemm(const CommGemmParams& params, gemm::GemmRecipe recipe, const LaunchContext& ctx) {
  Status s = cg_dispatch(params, recipe, 0, ctx);
  if (s != Status::kSuccess) return s;
  return Status::kSuccess;
}

Status gemm_reduce_scatter(const CommGemmParams& params, gemm::GemmRecipe recipe,
                           const LaunchContext& ctx) {
  return cg_dispatch(params, recipe, 1, ctx);
}

Status gemm_allreduce(const CommGemmParams& params, gemm::GemmRecipe recipe, AllReduceAlgo algo,
                      const LaunchContext& ctx) {
  if (algo != AllReduceAlgo::kOneShot && algo != AllReduceAlgo::kTwoShot &&
      algo != AllReduceAlgo::kDeterministic) {
    return detail::set_error(Status::kUnsupportedBackend, "gemm_allreduce: algo %d", static_cast<int>(algo));
  }
  Status s = cg_dispatch(params, recipe, 2, ctx);
  if (s != Status::kSuccess) return s;
  if (!params.fuse_residual_rmsnorm) return Status::kSuccess;
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int64_t rows = params.gemm.args.m;
  const int64_t cols = params.gemm.args.n;
  switch (recipe) {
    case gemm::GemmRecipe::kBF16:
      cg_norm_kernel<__nv_bfloat16><<<static_cast<unsigned>(rows), 256, 0, st>>>(
          static_cast<__nv_bfloat16*>(params.gemm.args.d), rows, cols,
          static_cast<const __nv_bfloat16*>(params.norm_weight), params.norm_eps, params.gemm.args.ldd);
      break;
    case gemm::GemmRecipe::kFP16:
      cg_norm_kernel<__half><<<static_cast<unsigned>(rows), 256, 0, st>>>(
          static_cast<__half*>(params.gemm.args.d), rows, cols,
          static_cast<const __half*>(params.norm_weight), params.norm_eps, params.gemm.args.ldd);
      break;
    default: break;
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                     "gemm_allreduce: norm launch failed: %s", cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

}  // namespace kops::comm
