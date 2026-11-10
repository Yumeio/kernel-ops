// =============================================================================
// kernel-ops | csrc/attention/mla.cu
// B2 A12 copy-side helpers: concat_mla_q and mla_context_gather.
//
//   concat_mla_q        q_out[t,h] = [ ql_nope[t,h,:] | q_pe[t,h,:] ]
//   mla_context_gather  gather cached latent entries of a cached prefix back
//                       into contiguous c_kv / k_pe buffers for up-projection.
//
// Both are plain index copies (no MMA); they accept strided query views and a
// paged latent cache. FP8 latent caches (per-tensor e4m3) are dequantized on the
// fly; the B3 ds_mla record codecs are rejected here.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstring>
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

using dev::from_f;
using dev::to_f;

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

template <typename cache_t, kv::KVCodec kCodec>
__device__ __forceinline__ float cache_load(cache_t v, float scale) {
  if constexpr (kCodec == kv::KVCodec::kNative) {
    return to_f<cache_t>(v);
  } else {
    return e4m3_to_float(static_cast<uint8_t>(v)) * scale;
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// mla_get_metadata: FlashMLA get_decoding_sched_meta (dense decode), 1 warp.
// DecodingSchedMeta is 8 int32 words: begin_req, end_req, begin_block,
// end_block, begin_split, is_first_req_splitted, is_last_req_splitted, pad.
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(32, 1, 1)
mla_get_metadata_kernel(int32_t* __restrict__ tile_scheduler_metadata,
                        int32_t* __restrict__ num_splits,
                        const int32_t* __restrict__ seq_lens, const int batch_size,
                        const int q_len_times_heads_per_group, const int num_sm_parts,
                        const int block_size_n, const int fixed_overhead_num_blocks) {
  (void)q_len_times_heads_per_group;
  extern __shared__ int32_t smem[];
  int32_t* num_blocks = smem;                       // [B]
  int32_t* num_splits_shared = smem + batch_size;   // [B + 1]
  int32_t* seqlens_k = smem + batch_size * 2 + 1;   // [B]
  int32_t* first_block = smem + batch_size * 3 + 1; // [B]
  int32_t* last_block = smem + batch_size * 4 + 1;  // [B]

  for (int i = static_cast<int>(threadIdx.x); i <= batch_size; i += 32) {
    num_splits_shared[i] = 0;
  }

  int total_blocks = 0;
  for (int i = static_cast<int>(threadIdx.x); i < batch_size; i += 32) {
    const int cur_s_k = seq_lens[i] < 0 ? 0 : seq_lens[i];
    seqlens_k[i] = cur_s_k;
    const int first_token = 0;
    const int last_token = cur_s_k > 0 ? cur_s_k - 1 : 0;
    const int fb = first_token / block_size_n;
    const int lb = last_token / block_size_n;
    const int blocks = lb - fb + 1;
    total_blocks += blocks + fixed_overhead_num_blocks;
    num_blocks[i] = blocks;
    first_block[i] = fb;
    last_block[i] = lb;
  }
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    total_blocks += __shfl_xor_sync(0xffffffffu, total_blocks, offset);
  }
  __syncwarp();

  if (threadIdx.x == 0 && batch_size > 0) {
    const int payload = (total_blocks + num_sm_parts - 1) / num_sm_parts +
                        fixed_overhead_num_blocks;
    int now_req = 0, now_block = 0, now_split = 0, cum_splits = 0;
    num_splits_shared[0] = 0;
    for (int i = 0; i < num_sm_parts; ++i) {
      // Trailing partitions (nothing left to schedule): begin > end, zero splits,
      // and an explicit empty block range so every field is deterministic.
      if (now_req >= batch_size) {
        const int empty_block = last_block[batch_size - 1] + 1;
        int32_t* meta = tile_scheduler_metadata + static_cast<int64_t>(i) * 8;
        meta[0] = batch_size;
        meta[1] = batch_size - 1;
        meta[2] = empty_block;
        meta[3] = empty_block;
        meta[4] = now_split;
        meta[5] = 0;
        meta[6] = 0;
        meta[7] = 0;
        continue;
      }
      const int begin_req = now_req;
      const int begin_block = now_block + first_block[now_req];
      const int begin_split = now_split;
      const bool first_splitted = now_block != 0;
      int remain = payload;
      while (now_req < batch_size) {
        const int blocks = num_blocks[now_req];
        const int remain_blocks = blocks - now_block;
        if (remain >= remain_blocks + fixed_overhead_num_blocks) {
          cum_splits += now_split + 1;
          num_splits_shared[now_req + 1] = cum_splits;
          remain -= remain_blocks + fixed_overhead_num_blocks;
          ++now_req;
          now_block = 0;
          now_split = 0;
        } else {
          if (remain - fixed_overhead_num_blocks > 0) {
            now_block += remain - fixed_overhead_num_blocks;
            ++now_split;
          }
          break;
        }
      }
      const int end_req = now_block > 0 ? now_req : now_req - 1;
      const int end_block =
          now_block > 0
              ? now_block + first_block[now_req]
              : (seqlens_k[now_req - 1] == 0 ? 0 : last_block[now_req - 1] + 1);
      bool last_splitted =
          end_block != last_block[end_req] + 1 && seqlens_k[end_req] != 0;
      bool first = first_splitted;
      if (begin_req == end_req) first = last_splitted = first || last_splitted;
      int32_t* meta = tile_scheduler_metadata + static_cast<int64_t>(i) * 8;
      meta[0] = begin_req;
      meta[1] = end_req;
      meta[2] = begin_block;
      meta[3] = end_block;
      meta[4] = begin_split;
      meta[5] = first ? 1 : 0;
      meta[6] = last_splitted ? 1 : 0;
      meta[7] = 0;
    }
  }
  __syncwarp();
  for (int i = static_cast<int>(threadIdx.x); i <= batch_size; i += 32) {
    num_splits[i] = num_splits_shared[i];
  }
}

using dev::from_f;
using dev::to_f;

// -----------------------------------------------------------------------------
// concat_mla_q
// -----------------------------------------------------------------------------

template <typename scalar_t, int NOPE_CHUNK>
__global__ void concat_mla_q_kernel(scalar_t* __restrict__ q_out,
                                    const scalar_t* __restrict__ ql_nope,
                                    const scalar_t* __restrict__ q_pe, const int num_tokens,
                                    const int num_heads, const int nope_dim,
                                    const int64_t out_stride_token,
                                    const int64_t out_stride_head,
                                    const int64_t nope_stride_token,
                                    const int64_t nope_stride_head,
                                    const int64_t pe_stride_token, const int64_t pe_stride_head) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t task = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  const int64_t total = static_cast<int64_t>(num_tokens) * num_heads;
  if (task >= total) return;
  const int t = static_cast<int>(task / num_heads);
  const int h = static_cast<int>(task % num_heads);
  const scalar_t* nsrc =
      ql_nope + static_cast<int64_t>(t) * nope_stride_token + static_cast<int64_t>(h) * nope_stride_head;
  const scalar_t* psrc =
      q_pe + static_cast<int64_t>(t) * pe_stride_token + static_cast<int64_t>(h) * pe_stride_head;
  scalar_t* dst =
      q_out + static_cast<int64_t>(t) * out_stride_token + static_cast<int64_t>(h) * out_stride_head;
  (void)NOPE_CHUNK;
  for (int i = lane; i < nope_dim; i += 32) dst[i] = nsrc[i];
  for (int i = lane; i < 64; i += 32) dst[nope_dim + i] = psrc[i];
}

// -----------------------------------------------------------------------------
// mla_context_gather
// -----------------------------------------------------------------------------

__device__ __forceinline__ int find_ctx_request(const int32_t* indptr, int batch_size,
                                                int token) {
  int lo = 0;
  int hi = batch_size;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indptr[mid + 1] <= token) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < batch_size ? lo : -1;
}

template <typename cache_t, typename out_t, kv::KVCodec kCodec>
__global__ void mla_context_gather_kernel(
    out_t* __restrict__ c_kv_out, out_t* __restrict__ k_pe_out, const cache_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ ctx_token_indptr,
    const int32_t* __restrict__ ctx_start, const float* __restrict__ scale,
    const kv::PagedKVDesc desc, const int batch_size) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
  out_t* c_base = c_kv_out;
  if (c_base == nullptr) return;
  if (token < 0) return;
  const int r = find_ctx_request(ctx_token_indptr, batch_size, token);
  if (r < 0) return;
  const int64_t pos = static_cast<int64_t>(ctx_start[r]) + (token - ctx_token_indptr[r]);
  const int64_t logical = pos / desc.page_size;
  if (logical < 0 || logical >= desc.block_table_stride) return;
  const int32_t page =
      block_table[static_cast<int64_t>(r) * desc.block_table_stride + logical];
  if (page < 0) return;
  const int64_t off = pos % desc.page_size;
  const cache_t* src =
      kv_cache + static_cast<int64_t>(page) * desc.k_stride_page + off * desc.k_stride_token;
  const int rank = desc.head_dim_v;
  const int pe = desc.head_dim_k - desc.head_dim_v;
  const float s = scale != nullptr ? scale[0] : 1.0f;
  out_t* c_dst = c_kv_out + static_cast<int64_t>(token) * rank;
  out_t* p_dst = k_pe_out + static_cast<int64_t>(token) * pe;
  for (int i = lane; i < rank; i += 32) {
    const float v = cache_load<cache_t, kCodec>(src[i], s);
    out_t y;
    from_f<out_t>(y, v);
    c_dst[i] = y;
  }
  for (int i = lane; i < pe; i += 32) {
    const float v = cache_load<cache_t, kCodec>(src[rank + i], s);
    out_t y;
    from_f<out_t>(y, v);
    p_dst[i] = y;
  }
}

namespace {

// -----------------------------------------------------------------------------
// Dense absorbed MLA decode (A12, T1) — device helpers.
//
// CTA geometry: one query token (of q_len) of one request, HEADS_PER_CTA = 16
// query heads = the mma M dimension, 4 warps. A latent tile holds 32 keys:
//   * warp w computes the QK columns [8w, 8w + 8) of that tile (one n8 slice
//     each), so the 16 x 32 score tile is produced once per CTA instead of once
//     per warp, and the QK phase keeps all four mma pipes busy;
//   * the same warp then owns one 128-dim slice of the 512-dim latent output,
//     so the PV accumulation needs no cross-warp reduction either.
// The softmax row maximum is combined across the four warps through smem
// (sMax), the exponentiated tile is published as two 16 x 16 bf16 A-operands
// (sP) and consumed with ldmatrix, and the online rescale alpha is applied to
// the per-warp O accumulators exactly as before. Two cp.async stages of latent
// tiles are in flight while Q (18 KiB) stays resident: 92 KiB of dynamic smem.
// -----------------------------------------------------------------------------

constexpr int kMlaRowDim = 512 + 64;   // latent | rope
constexpr int kMlaNope = 512;
constexpr int kMlaRope = 64;
constexpr int kMlaWarps = 4;
constexpr int kMlaThreads = kMlaWarps * 32;
constexpr int kMlaKeys = 32;                     // keys per latent tile (8 per warp)
constexpr int kMlaStages = 2;
constexpr int kMlaKeyGroup = kMlaKeys / kMlaWarps;  // 8 keys per warp = one n8 mma
constexpr int kMlaPStride = 24;                  // 16 bf16 + pad: 48 B rows, bank clean
constexpr int kMlaPSub = 16 * kMlaPStride;       // one 16 (heads) x 16 (keys) A-operand
constexpr float kMlaLog2e = 1.4426950408889634f;
constexpr float kMlaLn2 = 0.6931471805599453f;

// Dynamic smem: Q | K stages | P sub-tiles | per-warp row extrema.
constexpr int kMlaQElems = 16 * kMlaRowDim;                       // 9216 elements
constexpr int kMlaKElems = kMlaKeys * kMlaRowDim;                 // 18432 elements
constexpr int kMlaPElems = 2 * kMlaPSub;                          // 768 elements
constexpr int kMlaSmemBytes =
    (kMlaQElems + kMlaStages * kMlaKElems + kMlaPElems) * 2 + 4 * kMlaWarps * 16;

__device__ __forceinline__ float mla_ninf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ uint32_t mla_smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mla_cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#else
  (void)dst;
  (void)src;
  (void)pred;
#endif
}
__device__ __forceinline__ void mla_cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void mla_cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}
__device__ __forceinline__ void mla_ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
#else
  (void)r;
  (void)addr;
#endif
}
__device__ __forceinline__ void mla_ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#else
  (void)r;
  (void)addr;
#endif
}
__device__ __forceinline__ void mla_ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#else
  (void)r;
  (void)addr;
#endif
}
template <typename T>
__device__ __forceinline__ void mla_mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
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
__device__ __forceinline__ uint32_t mla_pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  } else {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
}
/// Four e4m3 codes -> four floats carrying value * 2^-8 (exact), as decode.cu.
template <bool kE5M2>
__device__ __forceinline__ void mla_fp8x4_to_f(uint32_t w, float* out) {
  uint32_t a = __byte_perm(w, 0u, 0x1404u);
  uint32_t b = __byte_perm(w, 0u, 0x3424u);
  if constexpr (!kE5M2) {
    a = (a & 0x80008000u) | ((a >> 1) & 0x3F803F80u);
    b = (b & 0x80008000u) | ((b >> 1) & 0x3F803F80u);
  }
  const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
  const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
  out[0] = fa.x;
  out[1] = fa.y;
  out[2] = fb.x;
  out[3] = fb.y;
}

/// smem rows of D 16-bit elements with the decode.cu swizzle (D % 64 == 0).
template <int D>
struct MlaLay {
  static_assert(D % 64 == 0, "swizzled MLA tile");
  static __device__ __forceinline__ int off(int row, int chunk) {
    return row * D + ((chunk ^ (row & 7)) << 3);
  }
};

/// The kMlaKeys keys of one tile touch at most two pages: page sizes are 16/32/64
/// and tiles start 32-aligned, so one base pair replaces a page-table lookup and
/// a 64-bit multiply per 16-byte row chunk.
struct MlaKeySpan {
  const char* lo;
  const char* hi;
  int split;       // first row that lives in the high page
  int off;         // row 0 offset inside its page, in tokens
  int64_t stride;  // bytes per token

  __device__ __forceinline__ const char* row(int r) const {
    return r < split ? lo + static_cast<int64_t>(off + r) * stride
                     : hi + static_cast<int64_t>(r - split) * stride;
  }
};

template <int PAGE_SIZE>
__device__ __forceinline__ MlaKeySpan mla_key_span(const char* kbase, int64_t ksp, int64_t kst,
                                                   const int32_t* pages, int t, int hi) {
  MlaKeySpan s;
  const int pg0 = t / PAGE_SIZE;
  const int off0 = t - pg0 * PAGE_SIZE;
  s.stride = kst;
  s.off = off0;
  s.split = PAGE_SIZE - off0;
  s.lo = kbase + static_cast<int64_t>(pages[pg0]) * ksp;
  s.hi = s.lo;
  // The high page is only read when the tile actually extends past the boundary.
  if (kMlaKeys > s.split && t + s.split < hi) {
    s.hi = kbase + static_cast<int64_t>(pages[pg0 + 1]) * ksp;
  }
  return s;
}

/// Native bf16/fp16 latent tile: kMlaKeys keys x 576 values, 16-byte cp.async chunks.
template <typename T>
__device__ __forceinline__ void mla_load_tile_native(T* sK, const MlaKeySpan& span, int t, int hi,
                                                     int tid) {
  using L = MlaLay<kMlaRowDim>;
  constexpr int kChunks = kMlaRowDim / 8;
  constexpr int kIter = kMlaKeys * kChunks / kMlaThreads;
  static_assert(kMlaKeys * kChunks % kMlaThreads == 0, "tile load");
#pragma unroll
  for (int it = 0; it < kIter; ++it) {
    const int idx = it * kMlaThreads + tid;
    const int r = idx / kChunks, c = idx - r * kChunks;
    const bool ok = t + r < hi;
    mla_cp_async16(mla_smem_u32(sK + L::off(r, c)), span.row(r) + c * 16, ok);
  }
}

/// Raw kMLAFp8DS chunks of one tile, prefetched into registers: the fp32-scaled
/// e4m3 -> 16-bit conversion of a tile then overlaps the MMAs of the tile before
/// it, which the single-buffered version could not do (the 656 B record is
/// converted on the way into smem either way).
struct MlaFp8Raw {
  uint4 nope[kMlaKeys * 32 / kMlaThreads];   // 8 fixed 16-byte NoPE columns per thread
  float scale[kMlaKeys * 32 / kMlaThreads];
  uint4 rope[kMlaKeys * 8 / kMlaThreads];    // 2 rope chunks per thread
};

__device__ __forceinline__ void mla_fp8_prefetch(MlaFp8Raw& raw, const MlaKeySpan& span, int t,
                                                 int hi, int tid) {
  constexpr int kNope = kMlaKeys * 32 / kMlaThreads;
  constexpr int kRope = kMlaKeys * 8 / kMlaThreads;
  static_assert(kMlaKeys * 8 % kMlaThreads == 0, "rope load");
  const int c16 = tid & 31;   // NoPE 16-byte column, fixed per thread
  const int rb = tid >> 5;    // 4 row lanes
  const int g = c16 >> 3;     // fp32 tile scale index
#pragma unroll
  for (int it = 0; it < kNope; ++it) {
    const int r = it * 4 + rb;
    if (t + r < hi) {
      const char* row = span.row(r);
      raw.nope[it] = *reinterpret_cast<const uint4*>(row + c16 * 16);
      raw.scale[it] = reinterpret_cast<const float*>(row + kMlaNope)[g] * 256.0f;
    } else {
      raw.nope[it] = make_uint4(0u, 0u, 0u, 0u);
      raw.scale[it] = 0.0f;
    }
  }
  const int rr = tid >> 3;                       // 16 row lanes
  const int rc = kMlaNope + 16 + (tid & 7) * 16;  // rope half of the record
#pragma unroll
  for (int it = 0; it < kRope; ++it) {
    const int r = it * 16 + rr;
    raw.rope[it] = t + r < hi ? *reinterpret_cast<const uint4*>(span.row(r) + rc)
                              : make_uint4(0u, 0u, 0u, 0u);
  }
}

template <typename T>
__device__ __forceinline__ void mla_fp8_convert(const MlaFp8Raw& raw, T* sK, int tid) {
  using L = MlaLay<kMlaRowDim>;
  constexpr int kNope = kMlaKeys * 32 / kMlaThreads;
  constexpr int kRope = kMlaKeys * 8 / kMlaThreads;
  const int c16 = tid & 31;
  const int rb = tid >> 5;
#pragma unroll
  for (int it = 0; it < kNope; ++it) {
    float f[16];
    mla_fp8x4_to_f<false>(raw.nope[it].x, f + 0);
    mla_fp8x4_to_f<false>(raw.nope[it].y, f + 4);
    mla_fp8x4_to_f<false>(raw.nope[it].z, f + 8);
    mla_fp8x4_to_f<false>(raw.nope[it].w, f + 12);
    const float sc = raw.scale[it];
#pragma unroll
    for (int j = 0; j < 16; ++j) f[j] *= sc;
    const int r = it * 4 + rb;
    *reinterpret_cast<uint4*>(sK + L::off(r, 2 * c16)) =
        make_uint4(mla_pack2<T>(f[0], f[1]), mla_pack2<T>(f[2], f[3]),
                   mla_pack2<T>(f[4], f[5]), mla_pack2<T>(f[6], f[7]));
    *reinterpret_cast<uint4*>(sK + L::off(r, 2 * c16 + 1)) =
        make_uint4(mla_pack2<T>(f[8], f[9]), mla_pack2<T>(f[10], f[11]),
                   mla_pack2<T>(f[12], f[13]), mla_pack2<T>(f[14], f[15]));
  }
  // RoPE: the record stores bf16, so fp16 compute may need a conversion.
  const int rr = tid >> 3;
  const int c = kMlaNope / 8 + (tid & 7);
#pragma unroll
  for (int it = 0; it < kRope; ++it) {
    const int r = it * 16 + rr;
    const uint4 w = raw.rope[it];
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      *reinterpret_cast<uint4*>(sK + L::off(r, c)) = w;
    } else {
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
      __half h[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(__bfloat162float(b[i]));
      *reinterpret_cast<uint4*>(sK + L::off(r, c)) = *reinterpret_cast<const uint4*>(h);
    }
  }
}

/// Key range [lo, hi) of split c (of S), same 16-aligned split as decode.cu.
__device__ __forceinline__ void mla_split_range(int kv_len, int c, int S, int* lo, int* hi) {
  int64_t ck = S > 1 ? ((static_cast<int64_t>(kv_len) + S - 1) / S + 15) / 16 * 16 : INT64_MAX;
  const int64_t l = static_cast<int64_t>(c) * ck;
  *lo = static_cast<int>(min(l, static_cast<int64_t>(kv_len)));
  *hi = static_cast<int>(min(l + ck, static_cast<int64_t>(kv_len)));
}

/// Two gate values of one output row (dtype is a runtime property of the param).
__device__ __forceinline__ float2 mla_gate2(const void* base, int64_t idx, DType dt) {
  if (dt == DType::kFloat32) {
    return *reinterpret_cast<const float2*>(static_cast<const float*>(base) + idx);
  }
  if (dt == DType::kFloat16) {
    const uint32_t w =
        *reinterpret_cast<const uint32_t*>(static_cast<const __half*>(base) + idx);
    return __half22float2(*reinterpret_cast<const __half2*>(&w));
  }
  const uint32_t w =
      *reinterpret_cast<const uint32_t*>(static_cast<const __nv_bfloat16*>(base) + idx);
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
}

}  // namespace

// -----------------------------------------------------------------------------
// mla_decode_kernel — dense absorbed MLA decode (ArchTag = Sm80 instantiation).
//
// One CTA = (request, query token, 16-head tile), split c of the request.
//   sQ [16][576] = [ q_nope[512] | q_pe[64] ]     (strided q views)
//   sK [2][32][576] = two latent tiles in flight   (native or 656 B record)
//   score (16 heads x 32 keys) computed once per CTA, 4-way split over the warps
//   o [16 heads][512] accumulated as 16 n8 mma groups per warp (128 dims/warp)
// Split-KV: num_splits[B+1] cumulative (FlashMLA). A request with one split
// writes o/lse and folds the sink; otherwise split c writes the normalized fp32
// partial row merge_indptr[row] + c (no sink) for merge_split_kv.
// -----------------------------------------------------------------------------
template <int KV_LORA_RANK, int ROPE_DIM, int HEADS_PER_CTA, int PAGE_SIZE, typename DTypeQ,
          typename DTypeKV, typename DTypeO, typename ArchTag>
__global__ void __launch_bounds__(kMlaThreads, 1)
    mla_decode_kernel(const __grid_constant__ MLADecodeParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  static_assert(KV_LORA_RANK == kMlaNope && ROPE_DIM == kMlaRope, "dense MLA 512 + 64");
  static_assert(HEADS_PER_CTA == 16, "16 mma rows");
  static_assert(PAGE_SIZE == 16 || PAGE_SIZE == 32 || PAGE_SIZE == 64, "page size");
  static_assert(std::is_same_v<DTypeQ, DTypeO>, "o dtype = q dtype");
  static_assert(std::is_same_v<DTypeQ, __nv_bfloat16> || std::is_same_v<DTypeQ, __half>,
                "q bf16 / fp16 only");
  (void)sizeof(ArchTag);
  using T = DTypeQ;
  constexpr bool kFp8 = std::is_same_v<DTypeKV, uint8_t>;
  constexpr int kQkSteps = kMlaRowDim / 16;
  constexpr int kDGroups = kMlaNope / (8 * kMlaWarps);  // 16 n8 groups per warp
  using L = MlaLay<kMlaRowDim>;

  extern __shared__ __align__(16) unsigned char mla_smem[];
  T* sQ = reinterpret_cast<T*>(mla_smem);
  T* sK = sQ + kMlaQElems;
  T* sP = sK + kMlaStages * kMlaKElems;
  float* sMax = reinterpret_cast<float*>(sP + kMlaPElems);  // [kMlaWarps][16]

  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int H = params.num_heads;
  const int q_len = params.q_len;
  const int ht = (H + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
  const int qi = static_cast<int>(blockIdx.x) / ht;               // query token of the request
  const int h0 = (static_cast<int>(blockIdx.x) % ht) * HEADS_PER_CTA;
  const int b = static_cast<int>(blockIdx.y);
  const int ns = params.num_splits != nullptr
                     ? params.num_splits[b + 1] - params.num_splits[b]
                     : 1;
  const bool split_mode = params.num_splits != nullptr && params.split.max_splits > 1;
  if (static_cast<int>(blockIdx.z) >= (split_mode ? ns : 1)) return;
  if (qi >= q_len) return;

  const kv::PagedKVDesc& kd = params.kv;
  const int kv_len = max(kd.seq_lens[b], 0);
  const int qpos = kv_len - q_len + qi;
  int lo = 0, hi = kv_len;
  if (split_mode) mla_split_range(kv_len, static_cast<int>(blockIdx.z), ns, &lo, &hi);
  const int hi_eff = min(hi, max(qpos + 1, 0));

  const int64_t esz = kFp8 ? 1 : static_cast<int64_t>(sizeof(T));
  const int64_t ksp = kd.k_stride_page * esz;
  const int64_t kst = kd.k_stride_token * esz;
  const char* kbase = static_cast<const char*>(kd.k_data);
  const int32_t* pages = kd.block_table + static_cast<int64_t>(b) * kd.block_table_stride;

  // Q tile: q_nope 512 + q_pe 64, strided per (token, head).
  {
    const int64_t qrow = static_cast<int64_t>(b) * q_len + qi;
    const char* qn = static_cast<const char*>(params.q_nope) +
                     qrow * params.q_nope_stride_n * static_cast<int64_t>(sizeof(T));
    const char* qp = static_cast<const char*>(params.q_pe) +
                     qrow * params.q_pe_stride_n * static_cast<int64_t>(sizeof(T));
    constexpr int kNopeChunks = kMlaNope / 8;
#pragma unroll
    for (int it = 0; it < HEADS_PER_CTA * kNopeChunks / kMlaThreads; ++it) {
      const int idx = it * kMlaThreads + tid;
      const int r = idx / kNopeChunks, c = idx - r * kNopeChunks;
      const int h = h0 + r;
      const bool ok = h < H;
      const char* src = qn + static_cast<int64_t>(h) * params.q_nope_stride_h * sizeof(T) + c * 16;
      mla_cp_async16(mla_smem_u32(sQ + L::off(r, c)), src, ok);
    }
#pragma unroll
    for (int it = 0; it < HEADS_PER_CTA * 8 / kMlaThreads; ++it) {
      const int idx = it * kMlaThreads + tid;
      const int r = idx >> 3, c = idx & 7;
      const int h = h0 + r;
      const bool ok = h < H;
      const char* src = qp + static_cast<int64_t>(h) * params.q_pe_stride_h * sizeof(T) + c * 16;
      mla_cp_async16(mla_smem_u32(sQ + L::off(r, kNopeChunks + c)), src, ok);
    }
  }

  // Prologue: Q plus the first latent tile in flight.
  MlaFp8Raw raw;
  if constexpr (!kFp8) {
    if (lo < hi_eff) {
      const MlaKeySpan span = mla_key_span<PAGE_SIZE>(kbase, ksp, kst, pages, lo, hi_eff);
      mla_load_tile_native<T>(sK, span, lo, hi_eff, tid);
    }
    mla_cp_async_commit();
    mla_cp_async_wait<0>();
    __syncthreads();
  } else {
    mla_cp_async_commit();
    mla_cp_async_wait<0>();
    __syncthreads();
    if (lo < hi_eff) {
      const MlaKeySpan span = mla_key_span<PAGE_SIZE>(kbase, ksp, kst, pages, lo, hi_eff);
      mla_fp8_prefetch(raw, span, lo, hi_eff, tid);
      mla_fp8_convert<T>(raw, sK, tid);                       // stage 0
      if (lo + kMlaKeys < hi_eff) {
        const int tp = lo + kMlaKeys;
        const MlaKeySpan sp = mla_key_span<PAGE_SIZE>(kbase, ksp, kst, pages, tp, hi_eff);
        mla_fp8_prefetch(raw, sp, tp, hi_eff, tid);            // one tile ahead
      }
      __syncthreads();
    }
  }

  const uint32_t sQ_u = mla_smem_u32(sQ);
  const uint32_t sP_u = mla_smem_u32(sP);
  const int a_row = lane & 15, a_col = (lane >> 4) << 3;              // A operand (Q, P)
  const int k_row = kMlaKeyGroup * warp + (lane & 7);                // B operand (K): 8 keys
  const int k_cs = (lane >> 3) & 1;
  const int v_row = (lane & 7) + (((lane >> 3) & 1) << 3);           // B operand (V): 16 keys
  const int v_cs = lane >> 4;
  const int v_chunk = warp * (kMlaNope / 8) / kMlaWarps;
  const float sm_scale = params.mod.sm_scale != 0.0f
                             ? params.mod.sm_scale
                             : rsqrtf(static_cast<float>(kMlaRowDim));
  const float score_scale = sm_scale * kMlaLog2e;

  float m_run[2] = {mla_ninf(), mla_ninf()};
  float l_run[2] = {0.0f, 0.0f};
  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

  int stage = 0;
  for (int t = lo; t < hi_eff; t += kMlaKeys, stage ^= 1) {
    T* const sKn = sK + (stage ^ 1) * kMlaKElems;   // buffer the previous tile released
    const int tn = t + kMlaKeys;
    if (tn < hi_eff) {
      if constexpr (kFp8) {
        mla_fp8_convert<T>(raw, sKn, tid);
        const int tp = tn + kMlaKeys;
        if (tp < hi_eff) {
          const MlaKeySpan sp = mla_key_span<PAGE_SIZE>(kbase, ksp, kst, pages, tp, hi_eff);
          mla_fp8_prefetch(raw, sp, tp, hi_eff, tid);
        }
      } else {
        const MlaKeySpan span = mla_key_span<PAGE_SIZE>(kbase, ksp, kst, pages, tn, hi_eff);
        mla_load_tile_native<T>(sKn, span, tn, hi_eff, tid);
      }
    }
    if constexpr (!kFp8) {
      mla_cp_async_commit();
      mla_cp_async_wait<1>();
    }
    __syncthreads();

    const uint32_t sK_u = mla_smem_u32(sK + stage * kMlaKElems);

    // QK: warp w produces the 16 heads x 8 keys slice [t + 8w, t + 8w + 8).
    float s[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) s[e] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < kQkSteps; ++kk) {
      uint32_t af[4], bb[2];
      mla_ldsm_x4(af, sQ_u + L::off(a_row, 2 * kk + (lane >> 4)) * 2);
      mla_ldsm_x2(bb, sK_u + L::off(k_row, 2 * kk + k_cs) * 2);
      mla_mma16816<T>(s, af, bb[0], bb[1]);
    }

    // Causal mask (bottom-right) + the tile row maximum, across all four warps.
    const int j0 = t + kMlaKeyGroup * warp + 2 * (lane & 3);
    const bool ok0 = j0 < hi_eff, ok1 = j0 + 1 < hi_eff;
    float x0 = ok0 ? s[0] * score_scale : mla_ninf();
    float x1 = ok1 ? s[1] * score_scale : mla_ninf();
    float x2 = ok0 ? s[2] * score_scale : mla_ninf();
    float x3 = ok1 ? s[3] * score_scale : mla_ninf();
    float mx0 = fmaxf(x0, x1), mx1 = fmaxf(x2, x3);
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const int row0 = lane >> 2;
    if ((lane & 3) == 0) {
      sMax[warp * 16 + row0] = mx0;
      sMax[warp * 16 + row0 + 8] = mx1;
    }
    __syncthreads();
    float m0 = sMax[row0], m1 = sMax[row0 + 8];
#pragma unroll
    for (int w = 1; w < kMlaWarps; ++w) {
      m0 = fmaxf(m0, sMax[w * 16 + row0]);
      m1 = fmaxf(m1, sMax[w * 16 + row0 + 8]);
    }
    const float mu0 = m0 == mla_ninf() ? 0.0f : m0;
    const float mu1 = m1 == mla_ninf() ? 0.0f : m1;
    const float alpha0 = exp2f(m_run[0] - mu0);
    const float alpha1 = exp2f(m_run[1] - mu1);
    m_run[0] = m0;
    m_run[1] = m1;
    const float p0 = exp2f(x0 - mu0), p1 = exp2f(x1 - mu0);
    const float p2 = exp2f(x2 - mu1), p3 = exp2f(x3 - mu1);
    l_run[0] = l_run[0] * alpha0 + (p0 + p1);
    l_run[1] = l_run[1] * alpha1 + (p2 + p3);

    // Publish P (16 heads x 32 keys) as two 16x16 bf16 A-operands; warp w owns
    // keys [8w, 8w + 8), i.e. sub-tile w >> 1 at column 8 * (w & 1).
    {
      T* const dst = sP + (warp >> 1) * kMlaPSub + (8 * (warp & 1) + 2 * (lane & 3));
      *reinterpret_cast<uint32_t*>(dst + row0 * kMlaPStride) = mla_pack2<T>(p0, p1);
      *reinterpret_cast<uint32_t*>(dst + (row0 + 8) * kMlaPStride) = mla_pack2<T>(p2, p3);
    }
    __syncthreads();

    // PV: A = the shared P tile, B = the latent tile's first 512 dims.
    uint32_t pa[2][4];
    mla_ldsm_x4(pa[0], sP_u + (a_row * kMlaPStride + a_col) * 2);
    mla_ldsm_x4(pa[1], sP_u + (kMlaPSub + a_row * kMlaPStride + a_col) * 2);
#pragma unroll
    for (int i = 0; i < kDGroups; ++i) {
      acc[i][0] *= alpha0;
      acc[i][1] *= alpha0;
      acc[i][2] *= alpha1;
      acc[i][3] *= alpha1;
    }
#pragma unroll
    for (int dg = 0; dg < kDGroups / 2; ++dg) {
      uint32_t b0[4], b1[4];
      mla_ldsm_x4_t(b0, sK_u + L::off(v_row, v_chunk + 2 * dg + v_cs) * 2);
      mla_ldsm_x4_t(b1, sK_u + L::off(kMlaKeys / 2 + v_row, v_chunk + 2 * dg + v_cs) * 2);
      mla_mma16816<T>(acc[2 * dg], pa[0], b0[0], b0[1]);
      mla_mma16816<T>(acc[2 * dg + 1], pa[0], b0[2], b0[3]);
      mla_mma16816<T>(acc[2 * dg], pa[1], b1[0], b1[1]);
      mla_mma16816<T>(acc[2 * dg + 1], pa[1], b1[2], b1[3]);
    }
    __syncthreads();
  }

  // Row totals: each warp holds its own 8-key share (quad-reduced), so the four
  // warps add up to the tile total; m is already identical in every warp.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
  }
  const int row0 = lane >> 2;
  if ((lane & 3) == 0) {
    sMax[warp * 16 + row0] = l_run[0];
    sMax[warp * 16 + row0 + 8] = l_run[1];
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kMlaWarps; ++w) sum += sMax[w * 16 + row0 + 8 * r];
    l_run[r] = sum;
  }

  // Sink (once) + normalization. Row r of the mma holds head (lane>>2) + 8r.
  const int64_t row_out = static_cast<int64_t>(b) * q_len + qi;
  int64_t prow = 0;
  if (split_mode) {
    prow = params.split.merge_indptr[row_out] + static_cast<int32_t>(blockIdx.z);
  }
  const float* sinks = params.mod.sink_logits;
  float lse2[2], oscale[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int h = h0 + (lane >> 2) + 8 * r;
    const bool has_sink = sinks != nullptr && h < H;
    const float sk = has_sink ? sinks[h] * kMlaLog2e : mla_ninf();
    if (split_mode) {
      // Partials never include the sink; merge_split_kv adds it exactly once.
      oscale[r] = l_run[r] > 0.0f ? 1.0f / l_run[r] : 0.0f;
      lse2[r] = l_run[r] > 0.0f ? (m_run[r] + log2f(l_run[r])) * kMlaLn2 : mla_ninf();
    } else {
      const float mn = fmaxf(m_run[r], sk);
      const float al = mn == mla_ninf() ? 0.0f : exp2f(m_run[r] - mn);
      const float den =
          mn == mla_ninf() ? 0.0f : l_run[r] * al + (sk == mla_ninf() ? 0.0f : exp2f(sk - mn));
      oscale[r] = den > 0.0f ? al / den : 0.0f;
      lse2[r] = den > 0.0f ? (mn + log2f(den)) * kMlaLn2 : mla_ninf();
    }
  }

  const void* gate = params.output_gate;
  T* obase = static_cast<T*>(params.o);
#pragma unroll
  for (int i = 0; i < kDGroups; ++i) {
    const int d = 128 * warp + 8 * i + 2 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int h = h0 + (lane >> 2) + 8 * r;
      if (h >= H) continue;
      float v0 = acc[i][2 * r] * oscale[r];
      float v1 = acc[i][2 * r + 1] * oscale[r];
      if (gate != nullptr) {
        const int64_t gi = row_out * params.gate_stride_n +
                           static_cast<int64_t>(h) * params.gate_stride_h + d;
        const float2 g = mla_gate2(gate, gi, params.gate_dtype);
        v0 *= 1.0f / (1.0f + __expf(-g.x));
        v1 *= 1.0f / (1.0f + __expf(-g.y));
      }
      if (split_mode) {
        float* dst = params.split.o_partial +
                     (static_cast<int64_t>(prow) * H + h) * kMlaNope + d;
        *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);
      } else {
        T* dst = obase + row_out * params.o_stride_n + static_cast<int64_t>(h) * params.o_stride_h + d;
        *reinterpret_cast<uint32_t*>(dst) = mla_pack2<T>(v0, v1);
      }
    }
  }
  if ((lane & 3) == 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int h = h0 + (lane >> 2) + 8 * r;
      if (h >= H) continue;
      if (split_mode) {
        params.split.lse_partial[static_cast<int64_t>(prow) * H + h] = lse2[r];
      } else if (params.lse != nullptr) {
        params.lse[row_out * H + h] = lse2[r];
      }
    }
  }
#else
  (void)params;
  if (threadIdx.x == 0) __trap();
#endif
}

namespace {

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

template <typename T>
T* mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

Status backend_ok(const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "attention.mla: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool scalar_dtype(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Build an element-strided copy of an MLA descriptor. Native byte strides are
/// divided by sizeof(cache_t); FP8 (uint8) strides are already elements.
template <typename cache_t>
Status element_desc(const kv::PagedKVDesc& in, kv::PagedKVDesc* out) {
  *out = in;
  const int64_t es = static_cast<int64_t>(sizeof(cache_t));
  if (in.stride_unit == StrideUnit::kByte) {
    if ((in.k_stride_page % es) != 0 || (in.k_stride_token % es) != 0 ||
        (in.k_stride_head % es) != 0) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "mla_context_gather: byte strides not divisible by element size");
    }
    out->k_stride_page /= es;
    out->k_stride_token /= es;
    out->k_stride_head /= es;
    out->stride_unit = StrideUnit::kElement;
  }
  if (out->k_stride_page == 0) {
    out->k_stride_page = static_cast<int64_t>(out->page_size) * (in.head_dim_k > 0 ? in.head_dim_k : 1);
  }
  if (out->k_stride_token == 0) out->k_stride_token = in.head_dim_k > 0 ? in.head_dim_k : 1;
  return Status::kSuccess;
}

// ---- dense MLA decode (A12) plan id + launcher ------------------------------------
constexpr int kMlaDecodeHeads = 16;                                            // HEADS_PER_CTA
constexpr int kMlaDecodeSmem = kMlaSmemBytes;  // Q | two K stages | P | row extrema
constexpr uint64_t kMlaDecodeTag = 0x4D4C4144ull << 32;                        // "MLAD"
constexpr uint64_t kMlaDecodeTagMask = 0xFFFFFFFFull << 32;

/// Plan identity of one dense MLA decode kernel configuration.
struct MlaDecodeKey {
  int32_t heads = 0;
  int32_t page_size = 0;
  int32_t q_len = 0;
  DType dt = DType::kBFloat16;
  bool fp8 = false;
};

uint64_t mla_decode_id(const MlaDecodeKey& k) {
  const uint32_t pc = k.page_size == 16 ? 0u : (k.page_size == 32 ? 1u : 2u);
  uint64_t id = kMlaDecodeTag | static_cast<uint64_t>(k.heads & 0xFFFF);
  id |= static_cast<uint64_t>(pc & 3u) << 16;
  id |= static_cast<uint64_t>(k.dt == DType::kFloat16 ? 1u : 0u) << 18;
  id |= static_cast<uint64_t>(k.fp8 ? 1u : 0u) << 19;
  id |= static_cast<uint64_t>(k.q_len & 0xF) << 20;
  return id;
}

bool mla_decode_key(uint64_t id, MlaDecodeKey* k) {
  if ((id & kMlaDecodeTagMask) != kMlaDecodeTag) return false;
  k->heads = static_cast<int32_t>(id & 0xFFFFu);
  const uint32_t pc = static_cast<uint32_t>((id >> 16) & 3u);
  k->page_size = pc == 0 ? 16 : (pc == 1 ? 32 : 64);
  k->dt = ((id >> 18) & 1u) != 0 ? DType::kFloat16 : DType::kBFloat16;
  k->fp8 = ((id >> 19) & 1u) != 0;
  k->q_len = static_cast<int32_t>((id >> 20) & 0xFu);
  return k->heads > 0 && k->q_len >= 1 && k->q_len <= 8;
}

template <int PAGE_SIZE, typename T, typename KVT>
Status launch_mla_decode_typed(const MLADecodeParams& p, dim3 grid, cudaStream_t st) {
  auto* fn = mla_decode_kernel<512, 64, kMlaDecodeHeads, PAGE_SIZE, T, KVT, T, Sm80>;
  static const cudaError_t attr =
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kMlaDecodeSmem);
  if (attr != cudaSuccess) return launch_error("mla_decode smem attribute", attr);
  fn<<<grid, kMlaThreads, kMlaDecodeSmem, st>>>(p);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("mla_decode launch", err);
  return Status::kSuccess;
}

template <typename T, typename KVT>
Status launch_mla_decode(const MLADecodeParams& p, dim3 grid, cudaStream_t st) {
  switch (p.kv.page_size) {
    case 16: return launch_mla_decode_typed<16, T, KVT>(p, grid, st);
    case 32: return launch_mla_decode_typed<32, T, KVT>(p, grid, st);
    case 64: return launch_mla_decode_typed<64, T, KVT>(p, grid, st);
    default:
      return detail::set_error(Status::kUnsupportedShape, "mla_decode: page size %d not compiled",
                               p.kv.page_size);
  }
}

template <typename T>
Status launch_mla_decode_dt(const MLADecodeParams& p, bool fp8, dim3 grid, cudaStream_t st) {
  if (fp8) return launch_mla_decode<T, uint8_t>(p, grid, st);
  return launch_mla_decode<T, T>(p, grid, st);
}

bool mla_mod_supported(const AttentionModifiers& m) {
  return m.logits_softcap == 0.0f && m.window_left < 0 && m.window_right < 0 &&
         m.chunk_size == 0 && m.alibi_slopes == nullptr;
}

}  // namespace

Status concat_mla_q(const Tensor& q_out, const Tensor& ql_nope, const Tensor& q_pe,
                    const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!scalar_dtype(q_out.dtype) || q_out.dtype != ql_nope.dtype || q_out.dtype != q_pe.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "concat_mla_q: dtype mismatch");
  }
  s = check_tensor(q_out, q_out.dtype, 3, 16, "concat_mla_q.q_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ql_nope, ql_nope.dtype, 3, 16, "concat_mla_q.ql_nope");
  if (s != Status::kSuccess) return s;
  s = check_tensor(q_pe, q_pe.dtype, 3, 16, "concat_mla_q.q_pe");
  if (s != Status::kSuccess) return s;
  if (q_out.stride[2] != 1 || ql_nope.stride[2] != 1 || q_pe.stride[2] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "concat_mla_q: last dim must be contiguous");
  }
  const int64_t tokens = q_out.shape[0];
  const int64_t heads = q_out.shape[1];
  const int64_t nope = ql_nope.shape[2];
  const int64_t pe = q_pe.shape[2];
  if (ql_nope.shape[0] != tokens || ql_nope.shape[1] != heads || q_pe.shape[0] != tokens ||
      q_pe.shape[1] != heads || q_out.shape[2] != nope + pe || pe != 64) {
    return detail::set_error(Status::kInvalidArgument,
                             "concat_mla_q: expected q_out [T,H,nope+64] and q_pe [T,H,64]");
  }
  if (tokens == 0 || heads == 0) return Status::kSuccess;
  if (tokens * heads > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "concat_mla_q: too many tasks");
  }
  const int64_t grid = ceil_div(tokens * heads, 8);
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_MLA_CONCAT(T)                                                                     \
  do {                                                                                         \
    concat_mla_q_kernel<T, 1><<<static_cast<int>(grid), 256, 0, stream>>>(                     \
        mut_ptr<T>(q_out), const_ptr<T>(ql_nope), const_ptr<T>(q_pe), static_cast<int>(tokens), \
        static_cast<int>(heads), static_cast<int>(nope), q_out.stride[0], q_out.stride[1],      \
        ql_nope.stride[0], ql_nope.stride[1], q_pe.stride[0], q_pe.stride[1]);                 \
    const cudaError_t err = cudaGetLastError();                                                \
    if (err != cudaSuccess) return launch_error("concat_mla_q", err);                          \
    return Status::kSuccess;                                                                   \
  } while (0)
  if (q_out.dtype == DType::kBFloat16) KOPS_MLA_CONCAT(__nv_bfloat16);
  if (q_out.dtype == DType::kFloat16) KOPS_MLA_CONCAT(__half);
  if (q_out.dtype == DType::kFloat32) KOPS_MLA_CONCAT(float);
#undef KOPS_MLA_CONCAT
  return detail::set_error(Status::kUnsupportedDtype, "concat_mla_q dtype");
}

Status mla_context_gather(const Tensor& c_kv_out, const Tensor& k_pe_out,
                          const kv::PagedKVDesc& cache, const Tensor& ctx_token_indptr,
                          const Tensor& ctx_start, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv::kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (!scalar_dtype(c_kv_out.dtype) || k_pe_out.dtype != c_kv_out.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather: dtype mismatch");
  }
  s = check_tensor(c_kv_out, c_kv_out.dtype, 2, 16, "mla_context_gather.c_kv_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(k_pe_out, k_pe_out.dtype, 2, 16, "mla_context_gather.k_pe_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ctx_token_indptr, DType::kInt32, 1, 4, "mla_context_gather.ctx_token_indptr");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ctx_start, DType::kInt32, 1, 4, "mla_context_gather.ctx_start");
  if (s != Status::kSuccess) return s;
  if (c_kv_out.stride[1] != 1 || k_pe_out.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mla_context_gather: outputs must be contiguous rows");
  }
  const int64_t tokens = c_kv_out.shape[0];
  const int64_t rank = c_kv_out.shape[1];
  const int64_t pe = k_pe_out.shape[1];
  if (k_pe_out.shape[0] < tokens || rank <= 0 || pe <= 0) {
    return detail::set_error(Status::kInvalidArgument, "mla_context_gather: output shapes");
  }
  if (cache.head_dim_v != rank || cache.head_dim_k != rank + pe) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_context_gather: descriptor entry %d+%d, outputs %lld+%lld",
                             cache.head_dim_v, cache.head_dim_k - cache.head_dim_v,
                             static_cast<long long>(rank), static_cast<long long>(pe));
  }
  if (cache.block_table == nullptr || cache.block_table_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_context_gather: a block table is required");
  }
  const int batch = static_cast<int>(ctx_token_indptr.shape[0] - 1);
  if (batch <= 0 || ctx_start.shape[0] < batch) {
    return detail::set_error(Status::kInvalidArgument, "mla_context_gather: batch mismatch");
  }
  if (cache.k_codec != kv::KVCodec::kNative && cache.k_codec != kv::KVCodec::kFP8PerTensor &&
      cache.k_codec != kv::KVCodec::kFP8PerHead) {
    return detail::set_error(Status::kNotImplemented,
                             "mla_context_gather: codec %d lands in B3",
                             static_cast<int>(cache.k_codec));
  }
  const bool fp8 = cache.k_codec != kv::KVCodec::kNative;
  if (fp8 && (cache.fp8 != Fp8Encoding::kOCP || cache.k_dtype == DType::kFloat8E5M2 ||
              cache.k_dtype == DType::kFloat8E5M2FNUZ)) {
    return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather: e4m3 OCP only");
  }
  if (!fp8 && cache.k_dtype != c_kv_out.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_context_gather: native cache/output dtype mismatch");
  }
  if (tokens == 0) return Status::kSuccess;
  const int64_t grid = ceil_div(tokens, 8);
  const cudaStream_t stream = stream_of(ctx);

#define KOPS_MLA_GATHER(CACHE_T, OUT_T, CODEC)                                                    \
  do {                                                                                             \
    kv::PagedKVDesc e{};                                                                           \
    Status es = element_desc<CACHE_T>(cache, &e);                                                   \
    if (es != Status::kSuccess) return es;                                                          \
    mla_context_gather_kernel<CACHE_T, OUT_T, CODEC><<<static_cast<int>(grid), 256, 0, stream>>>(   \
        mut_ptr<OUT_T>(c_kv_out), mut_ptr<OUT_T>(k_pe_out), static_cast<const CACHE_T*>(cache.k_data), \
        cache.block_table, const_ptr<int32_t>(ctx_token_indptr),                                      \
        const_ptr<int32_t>(ctx_start), cache.k_scale, e, batch);                                    \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("mla_context_gather", err);                         \
    return Status::kSuccess;                                                                        \
  } while (0)

#define KOPS_MLA_GATHER_OUT(CACHE_T, CODEC)                                                        \
  do {                                                                                              \
    if (c_kv_out.dtype == DType::kBFloat16) KOPS_MLA_GATHER(CACHE_T, __nv_bfloat16, CODEC);          \
    if (c_kv_out.dtype == DType::kFloat16) KOPS_MLA_GATHER(CACHE_T, __half, CODEC);                  \
    if (c_kv_out.dtype == DType::kFloat32) KOPS_MLA_GATHER(CACHE_T, float, CODEC);                   \
  } while (0)

  if (fp8) KOPS_MLA_GATHER_OUT(uint8_t, kv::KVCodec::kFP8PerTensor);
  if (cache.k_dtype == DType::kBFloat16) KOPS_MLA_GATHER_OUT(__nv_bfloat16, kv::KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat16) KOPS_MLA_GATHER_OUT(__half, kv::KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat32) KOPS_MLA_GATHER_OUT(float, kv::KVCodec::kNative);
#undef KOPS_MLA_GATHER_OUT
#undef KOPS_MLA_GATHER
  return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather cache dtype");
}

Status mla_get_metadata(const Tensor& tile_scheduler_metadata, const Tensor& num_splits,
                        const Tensor& seq_lens, int32_t q_len_times_heads_per_group,
                        int32_t num_sm_parts, int32_t block_size_n,
                        int32_t fixed_overhead_num_blocks, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(tile_scheduler_metadata, DType::kInt32, 2, 16,
                   "mla_get_metadata.tile_scheduler_metadata");
  if (s != Status::kSuccess) return s;
  s = check_tensor(num_splits, DType::kInt32, 1, 4, "mla_get_metadata.num_splits");
  if (s != Status::kSuccess) return s;
  s = check_tensor(seq_lens, DType::kInt32, 1, 4, "mla_get_metadata.seq_lens");
  if (s != Status::kSuccess) return s;
  if (num_sm_parts <= 0 || block_size_n <= 0 || fixed_overhead_num_blocks < 0 ||
      q_len_times_heads_per_group < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_get_metadata: num_sm_parts/block_size_n must be positive");
  }
  const int64_t batch64 = seq_lens.shape[0];
  if (batch64 < 0 || batch64 > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "mla_get_metadata: batch too large");
  }
  const int batch = static_cast<int>(batch64);
  if (tile_scheduler_metadata.shape[0] < num_sm_parts ||
      tile_scheduler_metadata.shape[1] < 8) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_get_metadata: metadata must be [num_sm_parts, 8]");
  }
  if (num_splits.shape[0] < batch + 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_get_metadata: num_splits must hold B + 1 entries");
  }
  if (batch == 0) return Status::kSuccess;
  const size_t smem = static_cast<size_t>(batch) * 5u * sizeof(int32_t) + sizeof(int32_t);
  const cudaStream_t stream = stream_of(ctx);
  const cudaError_t attr = cudaFuncSetAttribute(
      mla_get_metadata_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem));
  if (attr != cudaSuccess) return launch_error("mla_get_metadata attribute", attr);
  mla_get_metadata_kernel<<<1, 32, smem, stream>>>(
      mut_ptr<int32_t>(tile_scheduler_metadata), mut_ptr<int32_t>(num_splits),
      const_ptr<int32_t>(seq_lens), batch, q_len_times_heads_per_group, num_sm_parts,
      block_size_n, fixed_overhead_num_blocks);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("mla_get_metadata", err);
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// mla_decode_plan / mla_decode_run — dense absorbed MLA decode (A12, T1)
//
// Contract: one launch of mla_decode_kernel<..., Sm80> over
//   grid = (ceil(H/16) * q_len, B, S)
// where S = 1 without split-KV and S = params.split.max_splits with split-KV.
// The kernel is memory-bound: every query head of a request shares the latent
// cache, and the 512-dim latent output is summed in fp32 registers.
//
// Split-KV is a run-time property because the per-request split counts live in
// the device-side `num_splits[B+1]` produced by mla_get_metadata (FlashMLA
// convention). The static plan cannot know them, so it reserves no workspace
// and reports split_kv = false; the caller owns the partial buffers.
// -----------------------------------------------------------------------------

Status mla_decode_plan(const AttentionSpec& spec, int32_t max_batch, int32_t q_len,
                       const LaunchContext& ctx, const Workspace& ws, AttentionPlan* plan) {
  if (plan == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "mla_decode_plan: null plan");
  }
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "mla_decode: needs sm_80+");
  }
  if (spec.num_kv_heads != 1) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode: the latent MLA cache has one kv head, got %d",
                             spec.num_kv_heads);
  }
  if (spec.num_q_heads <= 0 || spec.num_q_heads > 65535) {
    return detail::set_error(Status::kUnsupportedShape, "mla_decode: num_q_heads %d",
                             spec.num_q_heads);
  }
  if (spec.head_dim_qk != kMlaNope + kMlaRope || spec.head_dim_vo != kMlaNope) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode: absorbed dims must be 576/512, got %d/%d",
                             spec.head_dim_qk, spec.head_dim_vo);
  }
  const bool dt16 = spec.q_dtype == DType::kBFloat16 || spec.q_dtype == DType::kFloat16;
  if (!dt16 || spec.o_dtype != spec.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_decode: q/o must be the same bf16 or fp16 dtype");
  }
  const bool fp8 = spec.k_codec == kv::KVCodec::kMLAFp8DS;
  if (!fp8 && spec.k_codec != kv::KVCodec::kNative) {
    return detail::set_error(Status::kNotImplemented,
                             "mla_decode: codec %d is not implemented for the dense path",
                             static_cast<int>(spec.k_codec));
  }
  if (spec.v_codec != kv::KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_decode: the latent cache has no separate V codec");
  }
  if (fp8) {
    if (spec.kv_dtype != DType::kFloat8E4M3) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "mla_decode: the kMLAFp8DS payload is e4m3, got %s",
                               dtype_name(spec.kv_dtype));
    }
  } else if (spec.kv_dtype != spec.q_dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_decode: native latent dtype %s must match q %s",
                             dtype_name(spec.kv_dtype), dtype_name(spec.q_dtype));
  }
  if (spec.kv_layout != kv::KVLayout::kLatent) {
    return detail::set_error(Status::kUnsupportedLayout, "mla_decode: a kLatent cache is required");
  }
  if (spec.page_size != 16 && spec.page_size != 32 && spec.page_size != 64) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode: page_size %d not compiled (16/32/64)", spec.page_size);
  }
  if (spec.mask != MaskMode::kCausal) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode: bottom-right causal masking only");
  }
  if (spec.sliding_window || spec.softcap || spec.alibi || spec.chunked || spec.fp8_tensor_core) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode: window / softcap / ALiBi / chunk / FP8-compute are not "
                             "part of the absorbed MLA decode contract");
  }
  if (q_len < 1 || q_len > 8 || max_batch < 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_decode_plan: q_len 1..8 and max_batch >= 1");
  }
  (void)ws;  // one launch over caller-owned tensors; no workspace
  MlaDecodeKey key;
  key.heads = spec.num_q_heads;
  key.page_size = spec.page_size;
  key.q_len = q_len;
  key.dt = spec.q_dtype;
  key.fp8 = fp8;
  *plan = AttentionPlan{};
  plan->kernel_id = mla_decode_id(key);
  plan->plan_generation = ctx.capture_generation;
  plan->strategy = ExecutionStrategy::kModular;
  plan->split_kv = false;
  plan->max_batch = max_batch;
  plan->max_total_q = max_batch * q_len;
  plan->max_kv_len = 0;  // the KV length is a run-time property (block table + seq_lens)
  plan->num_launches = 1;
  plan->workspace = WorkspaceRequirements{};
  plan->device_plan = WorkPlan{};
  return Status::kSuccess;
}

Status mla_decode_run(const AttentionPlan& plan, const MLADecodeParams& params,
                      const Workspace& ws, const LaunchContext& ctx) {
  MlaDecodeKey key;
  if (!mla_decode_key(plan.kernel_id, &key)) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_decode_run: plan was not made by mla_decode_plan");
  }
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "mla_decode: needs sm_80+");
  }
  (void)ws;
  const int B = params.batch_size;
  if (B < 0 || B > plan.max_batch) {
    return detail::set_error(Status::kCapacityExceeded, "mla_decode_run: batch %d > planned %d", B,
                             plan.max_batch);
  }
  if (params.q_len != key.q_len) {
    return detail::set_error(Status::kInvalidArgument, "mla_decode_run: q_len %d, planned %d",
                             params.q_len, key.q_len);
  }
  if (params.num_heads != key.heads) {
    return detail::set_error(Status::kInvalidArgument, "mla_decode_run: num_heads %d, planned %d",
                             params.num_heads, key.heads);
  }
  if (params.kv_lora_rank != kMlaNope || params.rope_dim != kMlaRope) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode_run: absorbed dims must be %d/%d, got %d/%d", kMlaNope,
                             kMlaRope, params.kv_lora_rank, params.rope_dim);
  }
  if (params.q_nope == nullptr || params.q_pe == nullptr || params.o == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_decode_run: q_nope, q_pe and o are required");
  }
  if (!mla_mod_supported(params.mod)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode_run: window / softcap / ALiBi / chunk modifiers are "
                             "unsupported");
  }
  const kv::PagedKVDesc& d = params.kv;
  s = kv::kv_desc_validate(d);
  if (s != Status::kSuccess) return s;
  if (d.layout != kv::KVLayout::kLatent) {
    return detail::set_error(Status::kUnsupportedLayout, "mla_decode_run: kLatent cache required");
  }
  if (d.table_format != kv::PageTableFormat::kBlockTable) {
    return detail::set_error(Status::kNotImplemented,
                             "mla_decode_run: the dense path reads a block table");
  }
  if (d.block_table == nullptr || d.seq_lens == nullptr || d.block_table_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_decode_run: block_table, seq_lens and block_table_stride are "
                             "required");
  }
  if (d.num_kv_heads > 1 || d.head_dim_k != kMlaNope + kMlaRope || d.head_dim_v != kMlaNope) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_decode_run: one latent head with head dims 576/512");
  }
  if (d.v_data != nullptr || d.v_codec != kv::KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mla_decode_run: the latent cache is shared (v_data null, V native)");
  }
  if (d.page_size != key.page_size) {
    return detail::set_error(Status::kUnsupportedShape, "mla_decode_run: page_size %d, planned %d",
                             d.page_size, key.page_size);
  }
  const bool fp8 = d.k_codec == kv::KVCodec::kMLAFp8DS;
  if (fp8 != key.fp8) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_decode_run: kv codec differs from the plan");
  }
  if (fp8) {
    if (d.entry_bytes != 656 || d.stride_unit != StrideUnit::kByte) {
      return detail::set_error(Status::kUnsupportedShape,
                               "mla_decode_run: kMLAFp8DS needs a 656 B record and byte strides");
    }
    if (d.k_dtype != DType::kFloat8E4M3 && d.k_dtype != DType::kUInt8) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "mla_decode_run: kMLAFp8DS record view must be e4m3/uint8, got %s",
                               dtype_name(d.k_dtype));
    }
  } else {
    if (d.entry_bytes != 0 || d.stride_unit != StrideUnit::kElement) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "mla_decode_run: native latent uses element strides");
    }
    if (d.k_dtype != key.dt) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "mla_decode_run: native KV dtype %s != q dtype %s",
                               dtype_name(d.k_dtype), dtype_name(key.dt));
    }
  }
  // 16-byte rows: q/o base + strides, and the latent page/record strides.
  if ((reinterpret_cast<uintptr_t>(params.q_nope) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.q_pe) & 15u) != 0 ||
      (reinterpret_cast<uintptr_t>(params.o) & 15u) != 0 ||
      ((params.q_nope_stride_n | params.q_nope_stride_h | params.q_pe_stride_n |
        params.q_pe_stride_h | params.o_stride_n | params.o_stride_h) & 7) != 0) {
    return detail::set_error(Status::kMisaligned,
                             "mla_decode_run: q/o base and strides must be 16-byte aligned");
  }
  if (params.q_nope_stride_h < kMlaNope || params.q_pe_stride_h < kMlaRope ||
      params.o_stride_h < kMlaNope ||
      params.q_nope_stride_n < static_cast<int64_t>(params.num_heads) * params.q_nope_stride_h ||
      params.q_pe_stride_n < static_cast<int64_t>(params.num_heads) * params.q_pe_stride_h ||
      params.o_stride_n < static_cast<int64_t>(params.num_heads) * params.o_stride_h) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mla_decode_run: q/o strides overlap rows");
  }
  if (params.lse != nullptr && (reinterpret_cast<uintptr_t>(params.lse) & 3u) != 0) {
    return detail::set_error(Status::kMisaligned, "mla_decode_run: lse must be 4-byte aligned");
  }
  const int64_t stride_mask = fp8 ? 15 : 7;
  if ((reinterpret_cast<uintptr_t>(d.k_data) & 15u) != 0 ||
      ((d.k_stride_page | d.k_stride_token | d.k_stride_head) & stride_mask) != 0) {
    return detail::set_error(Status::kMisaligned,
                             "mla_decode_run: latent cache base/strides must be 16-byte aligned");
  }
  const int64_t record = fp8 ? 656 : (kMlaNope + kMlaRope);
  if (d.k_stride_token < record ||
      d.k_stride_page < d.k_stride_token * static_cast<int64_t>(d.page_size)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mla_decode_run: latent strides must hold a token record and a page");
  }

  const bool split_mode = params.num_splits != nullptr && params.split.max_splits > 1;
  if (params.num_splits == nullptr && params.split.max_splits > 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_decode_run: num_splits is required when max_splits > 1");
  }
  if (params.split.max_splits < 1 || params.split.max_splits > 65535) {
    return detail::set_error(Status::kCapacityExceeded, "mla_decode_run: max_splits %d",
                             params.split.max_splits);
  }
  if (split_mode) {
    if (params.split.o_partial == nullptr || params.split.lse_partial == nullptr ||
        params.split.merge_indptr == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "mla_decode_run: split-KV needs o_partial, lse_partial and "
                               "merge_indptr");
    }
    if ((reinterpret_cast<uintptr_t>(params.split.o_partial) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(params.split.lse_partial) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(params.split.merge_indptr) & 3u) != 0) {
      return detail::set_error(Status::kMisaligned,
                               "mla_decode_run: split buffers violate 16/4/4-byte alignment");
    }
  }
  if (params.output_gate != nullptr) {
    if (params.gate_dtype != DType::kBFloat16 && params.gate_dtype != DType::kFloat16 &&
        params.gate_dtype != DType::kFloat32) {
      return detail::set_error(Status::kUnsupportedDtype, "mla_decode_run: gate dtype %s",
                               dtype_name(params.gate_dtype));
    }
    if (((params.gate_stride_n | params.gate_stride_h) & 1) != 0 ||
        params.gate_stride_h < kMlaNope ||
        params.gate_stride_n < static_cast<int64_t>(params.num_heads) * params.gate_stride_h) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "mla_decode_run: gate strides must be even and non-overlapping");
    }
    const uintptr_t align = params.gate_dtype == DType::kFloat32 ? 7u : 3u;
    if ((reinterpret_cast<uintptr_t>(params.output_gate) & align) != 0) {
      return detail::set_error(Status::kMisaligned, "mla_decode_run: gate base alignment");
    }
  }
  if (B == 0) return Status::kSuccess;

  const int ht = (params.num_heads + kMlaDecodeHeads - 1) / kMlaDecodeHeads;
  const int64_t gx = static_cast<int64_t>(ht) * params.q_len;
  if (gx <= 0 || gx > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "mla_decode_run: grid.x %lld",
                             static_cast<long long>(gx));
  }
  const unsigned gz = static_cast<unsigned>(split_mode ? params.split.max_splits : 1);
  const dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(B), gz);
  const cudaStream_t stream = stream_of(ctx);
  if (key.dt == DType::kFloat16) {
    return launch_mla_decode_dt<__half>(params, fp8, grid, stream);
  }
  return launch_mla_decode_dt<__nv_bfloat16>(params, fp8, grid, stream);
}

}  // namespace kops::attn
