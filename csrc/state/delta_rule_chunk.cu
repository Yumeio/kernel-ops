// =============================================================================
// kernel-ops | csrc/state/delta_rule_chunk.cu
// B9 row 7 (L02): chunked WY prefill of the gated delta rule (GDN, KDA), FLA's
// chunk_gated_delta_rule / chunk_kda in fp32 or on mma.sync:
//   chunk_local_cumsum_kernel     G = inclusive cumsum of g inside each 64-token chunk
//   chunk_scaled_dot_kkt_kernel   A_jm = beta_j (k_j Gamma_j/Gamma_m . k_m), m < j
//   solve_tril_kernel             T = (I + A)^-1, 16 x 16 block forward substitution (fp32)
//   recompute_w_u_kernel          W = T (beta Gamma k), U = T (beta v)
//   chunk_state_fwd_kernel        v_new = U - W h, h' = Gamma_last h + (k Gamma_last/Gamma)^T v_new
//   chunk_output_fwd_kernel       o = scale ((q Gamma) h + causal(q k^T Gamma/Gamma) v_new)
// ChunkMath::kTensorCore runs kkt / w,u / state / output on bf16 mma.sync (the *_tc_* bodies,
// ROADMAP 1.6 variant table); cumsum and solve_tril stay fp32. Every exponent is a difference of
// cumulative sums with the later position first, hence <= 0: nothing overflows (KDA forms the
// pairwise factors per channel). A sequence's result does not depend on the other sequences and a
// call cut at a chunk boundary and resumed from the saved state is bitwise the one-call result.
// Contract: delta_rule_kernels.cuh, ops.h, CONTRACTS.md §1f.
// Ported from the ayaka implementation (3rd/csrc/state/delta_rule.cu, delta_rule_part2.cu).
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
#include "csrc/state/state_mma.cuh"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

// The GDN and KDA instantiations of one body each leave operands of the other granularity unread.
#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 550

namespace kops::state {
namespace {

using dev::group_sum;
using dev::smem_gemm_kmajor;
using bf16 = __nv_bfloat16;

template <typename T>
__device__ __forceinline__ float to_float(T v) { return dev::to_f(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v) {
  T r;
  dev::from_f(r, v);
  return r;
}

constexpr float kL2Eps = 1e-6f;

constexpr int kDrChunk = 64;
constexpr int kDrThreads = 256;  // a 16 x 16 grid of threads with a 4 x 4 register tile each, as the other chunked kernels
constexpr int kDrSlice = 32;     // features per slice of a staged product
constexpr int kDrLd = kDrChunk + 4;

/** The chunk `chunk` of sequence `b`: tokens [t0, t0 + m). False when the chunk lies past the end of the sequence. */
__device__ __forceinline__ bool dr_chunk_of(const DeltaRuleParams& p, int chunk, int b, int& t0, int& m) {
  const int start = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - start;
  if (int64_t(chunk) * kDrChunk >= n) return false;
  t0 = start + chunk * kDrChunk;
  m = min(kDrChunk, n - chunk * kDrChunk);
  return true;
}

__device__ __forceinline__ bool dr_slot_ok(const DeltaRuleParams& p, int b) {
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  return slot >= 0 && slot < p.num_slots && slot != p.pad_slot_id;
}

/** rinv[i] = rsqrt(sum_d x[t0 + i, d]^2 + eps) (1 without the L2 norm) for the 64 token rows of a chunk; 256 threads, 4 per row. The
 * caller synchronizes before reading rinv. */
template <typename T>
__device__ __forceinline__ void dr_row_rinv(float* rinv, const T* x, int64_t token_stride, int t0, int m, int dk, bool l2norm) {
  const int i = threadIdx.x >> 2, sub = threadIdx.x & 3;
  float ss = 0.0f;
  if (i < m) {
    const T* row = x + int64_t(t0 + i) * token_stride;
#pragma unroll 8
    for (int d = sub; d < dk; d += 4) {
      const float v = to_float(row[d]);
      ss = __fmaf_rn(v, v, ss);
    }
  }
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  if (sub == 0) rinv[i] = (l2norm && i < m) ? rsqrtf(ss + kL2Eps) : 1.0f;
}

/** Grid x of a chunk-parallel kernel: the chunks of the longest sequence the caller allows. */
__host__ __device__ inline int64_t dr_grid_chunks(const DeltaRuleParams& p) {
  const int64_t bound = p.max_seqlen > 0 ? p.max_seqlen : p.total_tokens;
  return (bound + kDrChunk - 1) / kDrChunk;
}

// ---- tensor-core fragments (mma.sync m16n8k16, bf16 operands, fp32 accumulation) ----------------
// The primitives live in state_mma.cuh (shared with SSD and lightning attention).
using mma::frag_a;
using mma::frag_a_trans;
using mma::frag_b2;
using mma::frag_b2_trans;
using mma::ldsm_x4;
using mma::ldsm_x4_trans;
using mma::mma_bf16_16x8x16;
using mma::mma_f16_16x8x16;
using mma::pack_bf16x2;

}  // namespace

namespace {
// ---- ported from ayaka state/delta_rule.cu:396-448 ----
// ---- chunked prefill: the seven kernels of row 7 -----------------------------------------------------------
//
// A sequence is cut into chunks of kDrChunk = 64 tokens from its first token; the WY form of the recurrence inside a chunk (FLA's
// chunk_gated_delta_rule / chunk_kda) is evaluated with 64 x 64 matmuls in fp32, and a short sequential pass over the chunks of a
// sequence moves the state. With G the cumulative log-decay inside the chunk and Gamma_i = exp(G_i):
//   A_jm = beta_j (k_j . Gamma_j/Gamma_m k_m)  (m < j),   U = T (beta v),   W = T (beta Gamma k),   T = (I + A)^-1,
//   v_new = U - W h,   o = scale ((q Gamma) h + (q k^T Gamma/Gamma, causal) v_new),   h' = Gamma_last h + (k Gamma_last/Gamma)^T v_new.
// Every exponent is a difference of cumulative sums with the later position first, hence <= 0: nothing overflows. For KDA the
// decay is a vector over Dk, so the pairwise dot products are formed channel by channel (exp inside the sum); for GDN one
// scalar factor multiplies a plain matmul. q and k are L2-normalised where they are staged (the row norms come from a pre-pass
// with a fixed order), so no normalised copy is stored.

/**
 * acc[r][c] += sum over the 32 staged channels of a[i][d] b[j][d] exp(G_i[d] - G_j[d]) for the 4 x 4 pairs (i = 4 ty + r, j = 4 tx + c),
 * the rows `a`, columns `b` feature-major tiles and the cumulative log-decay `ga`, `gb` staged beside them (KDA).
 * Below the diagonal (tx < ty, every j < every i) the factor splits through G of the last token of the column group,
 * exp(G_i - G_ref) exp(G_ref - G_j), and both parts are <= 1 (G falls with the token), so a pair costs one multiply-add and a tile
 * 8 exponentials per channel instead of 16. On the diagonal tile each pair keeps its own exponent, taken only where j < i
 * (or j <= i with kInclusive) so that no exponent is ever positive.
 */
template <bool kDiagonal, bool kInclusive = false>
__device__ __forceinline__ void kda_pair_tile(float (&acc)[4][4], const float* a, const float* ga, const float* b, const float* gb, int ty, int tx) {
  for (int kk = 0; kk < kDrSlice; ++kk) {
    const float4 a4 = *reinterpret_cast<const float4*>(&a[kk * kDrLd + ty * 4]), gi4 = *reinterpret_cast<const float4*>(&ga[kk * kDrLd + ty * 4]);
    const float4 b4 = *reinterpret_cast<const float4*>(&b[kk * kDrLd + tx * 4]), gj4 = *reinterpret_cast<const float4*>(&gb[kk * kDrLd + tx * 4]);
    const float av[4] = {a4.x, a4.y, a4.z, a4.w}, gi[4] = {gi4.x, gi4.y, gi4.z, gi4.w};
    const float bv[4] = {b4.x, b4.y, b4.z, b4.w}, gj[4] = {gj4.x, gj4.y, gj4.z, gj4.w};
    if constexpr (kDiagonal) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool in = kInclusive ? (c <= r) : (c < r);  // tx == ty here, so j <= i is c <= r
          acc[r][c] = __fmaf_rn(av[r] * bv[c], in ? __expf(gi[r] - gj[c]) : 0.0f, acc[r][c]);
        }
      }
    } else {
      const float gref = gj[3];
      float ar[4], bc[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        ar[x] = av[x] * __expf(gi[x] - gref);
        bc[x] = bv[x] * __expf(gref - gj[x]);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = __fmaf_rn(ar[r], bc[c], acc[r][c]);
      }
    }
  }
}


// ---- ported from ayaka state/delta_rule.cu:470-948 ----
// ---- chunk_local_cumsum_kernel (2) ------

/** Grid (chunks, Hv, B), 128 threads. KDA: thread d walks the (at most 64) tokens of its channel; GDN: a two-warp inclusive scan. */
}  // namespace
template <DecayGranularity kDecay>
__global__ __launch_bounds__(128) void chunk_local_cumsum_kernel(const __grid_constant__ DeltaRuleParams p) {
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;  // uniform per CTA
  const int h = blockIdx.y, hv = p.num_v_heads;
  if constexpr (kDecay == DecayGranularity::kPerChannel) {
    const int d = threadIdx.x;
    if (d >= p.head_dim_k) return;
    float run = 0.0f;
    for (int i = 0; i < m; ++i) {
      const int64_t at = (int64_t(t0 + i) * hv + h) * p.head_dim_k + d;
      run += p.g[at];
      p.gc[at] = run;
    }
  } else {
    const int i = threadIdx.x, lane = i & 31, warp = i >> 5;
    __shared__ float warp_total[2];
    float v = (i < m) ? p.g[int64_t(t0 + i) * hv + h] : 0.0f;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float u = __shfl_up_sync(0xffffffffu, v, off);
      if (lane >= off) v += u;
    }
    if (lane == 31 && warp < 2) warp_total[warp] = v;
    __syncthreads();
    if (warp == 1) v += warp_total[0];
    if (i < m) p.gc[int64_t(t0 + i) * hv + h] = v;
  }
}
namespace {

// ---- chunk_scaled_dot_kkt_kernel (3) ------

/** Grid (chunks, Hv, B). A_jm = beta_j (k_j . k_m decayed) for m < j: GDN is the 64 x 64 matmul k k^T scaled by exp(G_j - G_m), KDA
 * sums k_j[d] k_m[d] exp(G_j[d] - G_m[d]) over the channels directly (only the lower triangle of the tile grid is computed). Rows
 * are written whole, with zeros on and above the diagonal, so the buffer needs no initialisation. */
}  // namespace
template <typename T, bool kPerChannel>
__global__ __launch_bounds__(kDrThreads, 1) void chunk_scaled_dot_kkt_kernel(const __grid_constant__ DeltaRuleParams p) {
  __shared__ __align__(16) float ks[kDrSlice * kDrLd];
  __shared__ __align__(16) float gs[kPerChannel ? kDrSlice * kDrLd : 4];
  __shared__ float rinv[kDrChunk], gt[kDrChunk];
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;
  const int h = blockIdx.y, hv = p.num_v_heads, dk = p.head_dim_k, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  dr_row_rinv(rinv, k, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
  if (!kPerChannel && tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
  float acc[4][4] = {};
  for (int d0 = 0; d0 < dk; d0 += kDrSlice) {
    __syncthreads();  // rinv and gt are written; the previous slice is consumed
    for (int idx = tid; idx < kDrChunk * kDrSlice; idx += kDrThreads) {
      const int kk = idx % kDrSlice, i = idx / kDrSlice;
      float kv = 0.0f, gv = 0.0f;
      if (i < m && d0 + kk < dk) {
        kv = to_float(k[int64_t(t0 + i) * p.k_token_stride + d0 + kk]) * rinv[i];
        if constexpr (kPerChannel) gv = p.gc[(int64_t(t0 + i) * hv + h) * dk + d0 + kk];
      }
      ks[kk * kDrLd + i] = kv;
      if constexpr (kPerChannel) gs[kk * kDrLd + i] = gv;
    }
    __syncthreads();
    if constexpr (!kPerChannel) {
      smem_gemm_kmajor<4, 4>(acc, ks, kDrLd, ty * 4, ks, kDrLd, tx * 4, kDrSlice);
    } else if (tx < ty) {
      kda_pair_tile<false>(acc, ks, gs, ks, gs, ty, tx);
    } else if (tx == ty) {
      kda_pair_tile<true, false>(acc, ks, gs, ks, gs, ty, tx);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = ty * 4 + r;
    if (i >= m) continue;
    const float beta = p.beta[int64_t(t0 + i) * hv + h];
    float row[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = tx * 4 + c;
      row[c] = j < i ? beta * acc[r][c] * (kPerChannel ? 1.0f : __expf(gt[i] - gt[j])) : 0.0f;
    }
    *reinterpret_cast<float4*>(&p.A[(int64_t(t0 + i) * hv + h) * kDrChunk + tx * 4]) = make_float4(row[0], row[1], row[2], row[3]);
  }
}
namespace {

// ---- solve_tril_kernel (4) ------

/**
 * Grid (chunks, Hv, B), 64 threads: (I + A)^-1 of the unit lower triangular 64 x 64 matrix of a chunk by block forward substitution on its
 * sixteen 16 x 16 blocks, all in fp32 with a fixed order. First the four diagonal blocks D_b = (I + A_bb)^-1 (thread 16 b + c forward-substitutes
 * column c), then block row i, level by level l = i - j = 1..3: T_ij = -D_i (sum over k = j .. i-1 of A_ik T_kj), the blocks of a level in
 * parallel (a 16 x 16 product is 4 outputs per thread). The chain per thread is a few hundred dependent FMAs instead of the 2,000 of a
 * column-by-column substitution of the whole matrix. Rows past the chunk are not written.
 */
}  // namespace
__global__ __launch_bounds__(kDrChunk) void solve_tril_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kLd = kDrChunk + 1;
  __shared__ float L[kDrChunk * kLd], T[kDrChunk * kLd], S[3 * 16 * 17];
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;
  const int h = blockIdx.y, tid = threadIdx.x, hv = p.num_v_heads;
  float* rows = p.A + (int64_t(t0) * hv + h) * kDrChunk;
  const int64_t row_stride = int64_t(hv) * kDrChunk;
  for (int i = 0; i < kDrChunk; ++i) L[i * kLd + tid] = (i < m && tid < i) ? rows[i * row_stride + tid] : 0.0f;
  __syncthreads();
  // the diagonal blocks: thread (b, c) is column c of D_b
  {
    const int b = tid >> 4, c = tid & 15, base = 16 * b;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = (i == c) ? 1.0f : 0.0f;
      if (i >= c) {
#pragma unroll
        for (int j = 0; j < i; ++j)
          if (j >= c) acc = __fmaf_rn(-L[(base + i) * kLd + base + j], x[j], acc);
      }
      x[i] = (i >= c) ? acc : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) T[(base + i) * kLd + base + c] = x[i];
  }
  __syncthreads();
  // block row i, level l: S = sum_{k = j}^{i-1} A_ik T_kj (16 x 16), then T_ij = -D_i S. The 4 outputs of a thread: row r, columns 4 cq .. 4 cq + 3.
  const int r = tid >> 2, cq = tid & 3;
#pragma unroll
  for (int l = 1; l <= 3; ++l) {
    // the blocks of this level: (i, j = i - l) for i = l .. 3; every thread works on all of them in turn (their number is 4 - l)
    float s_acc[3][4];
#pragma unroll
    for (int bi = 0; bi < 3; ++bi) {
      const int i = l + bi;
#pragma unroll
      for (int q = 0; q < 4; ++q) s_acc[bi][q] = 0.0f;
      if (i <= 3) {
        const int j = i - l;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          if (j + k <= i - 1) {  // A_{i, j+k} T_{j+k, j}
            const int kb = j + k;
            for (int kk = 0; kk < 16; ++kk) {
              const float a = L[(16 * i + r) * kLd + 16 * kb + kk];
#pragma unroll
              for (int q = 0; q < 4; ++q) s_acc[bi][q] = __fmaf_rn(a, T[(16 * kb + kk) * kLd + 16 * j + 4 * cq + q], s_acc[bi][q]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int bi = 0; bi < 3; ++bi) {
      if (l + bi <= 3) {
#pragma unroll
        for (int q = 0; q < 4; ++q) S[(bi * 16 + r) * 17 + 4 * cq + q] = s_acc[bi][q];
      }
    }
    __syncthreads();  // S is complete (the loop ends with a barrier, so the previous level is done with it)
#pragma unroll
    for (int bi = 0; bi < 3; ++bi) {
      const int i = l + bi;
      if (i <= 3) {
        const int j = i - l;
        float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int kk = 0; kk < 16; ++kk) {
          const float d = T[(16 * i + r) * kLd + 16 * i + kk];
#pragma unroll
          for (int q = 0; q < 4; ++q) t[q] = __fmaf_rn(d, S[(bi * 16 + kk) * 17 + 4 * cq + q], t[q]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) T[(16 * i + r) * kLd + 16 * j + 4 * cq + q] = -t[q];
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < kDrChunk; ++i) {
    if (i < m) rows[i * row_stride + tid] = tid <= i ? T[i * kLd + tid] : 0.0f;
  }
}
namespace {

// ---- recompute_w_u_kernel (5) ------

/** Grid (chunks * column tiles, Hv, B): a 64 x 64 tile of W = T (beta Gamma k) (the first ceil(Dk / 64) tiles) or of U = T (beta v), T^T
 * staged k-major from the buffer that solve_tril left in A. */
}  // namespace
template <typename T, bool kPerChannel>
__global__ __launch_bounds__(kDrThreads, 1) void recompute_w_u_kernel(const __grid_constant__ DeltaRuleParams p) {
  __shared__ __align__(16) float ts[kDrChunk * kDrLd], bs[kDrChunk * kDrLd];
  __shared__ float rinv[kDrChunk];
  const int dk = p.head_dim_k, dv = p.head_dim_v, hv = p.num_v_heads;
  const int dkt = (dk + kDrChunk - 1) / kDrChunk, dvt = (dv + kDrChunk - 1) / kDrChunk, coltiles = dkt + dvt;
  const int tile = blockIdx.x % coltiles, chunk = blockIdx.x / coltiles;
  int t0, m;
  if (!dr_chunk_of(p, chunk, blockIdx.z, t0, m)) return;
  const int h = blockIdx.y, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const bool is_w = tile < dkt;
  const int c0 = (is_w ? tile : tile - dkt) * kDrChunk;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + int64_t(h) * p.v_head_stride;
  if (is_w) dr_row_rinv(rinv, k, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
  for (int idx = tid; idx < kDrChunk * kDrChunk; idx += kDrThreads) {
    const int j = idx / kDrChunk, mm = idx % kDrChunk;
    ts[mm * kDrLd + j] = (j < m && mm <= j) ? p.A[(int64_t(t0 + j) * hv + h) * kDrChunk + mm] : 0.0f;
  }
  __syncthreads();
  for (int idx = tid; idx < kDrChunk * kDrChunk; idx += kDrThreads) {
    const int cc = idx % kDrChunk, mm = idx / kDrChunk;
    float val = 0.0f;
    if (mm < m) {
      const int64_t tok = int64_t(t0 + mm);
      const float beta = p.beta[tok * hv + h];
      if (is_w) {
        const int d = c0 + cc;
        if (d < dk) val = beta * __expf(kPerChannel ? p.gc[(tok * hv + h) * dk + d] : p.gc[tok * hv + h]) * to_float(k[tok * p.k_token_stride + d]) * rinv[mm];
      } else if (c0 + cc < dv) {
        val = beta * to_float(v[tok * p.v_token_stride + c0 + cc]);
      }
    }
    bs[mm * kDrLd + cc] = val;
  }
  __syncthreads();
  float acc[4][4] = {};
  smem_gemm_kmajor<4, 4>(acc, ts, kDrLd, ty * 4, bs, kDrLd, tx * 4, kDrChunk);
  float* out = is_w ? p.w : p.u;
  const int width = is_w ? dk : dv;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int j = ty * 4 + r;
    if (j >= m) continue;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      if (c0 + tx * 4 + c < width) out[(int64_t(t0 + j) * hv + h) * width + c0 + tx * 4 + c] = acc[r][c];
    }
  }
}
namespace {

// ---- chunk_state_fwd_kernel (6) ------

__host__ __device__ constexpr size_t dr_state_smem_bytes(int dk_pad, int dv_tile) {
  return (size_t(dk_pad) * dv_tile + size_t(kDrChunk) * kDrLd + size_t(kDrChunk) * (dv_tile + 4) + size_t(dk_pad) + 192) * sizeof(float);
}

/**
 * Grid (ceil(Dv / kDv), Hv, B): a CTA owns the S[:, dv tile] columns of one (sequence, value head) in shared memory and walks the
 * chunks of its sequence in order. Per chunk: the state is stored to h (for the output kernel and prefix caching), v_new = U - W S is
 * formed 64 x kDv at a time (the W slice staged k-major), and S <- Gamma_last S + (k Gamma_last/Gamma)^T v_new, 64 rows of S at a time.
 */
}  // namespace
template <typename T, bool kPerChannel, int kDv>
__global__ __launch_bounds__(kDrThreads, 1) void chunk_state_fwd_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int RN = kDv / 16;
  extern __shared__ __align__(16) float smem[];
  const int dk = p.head_dim_k, dv = p.head_dim_v, hv = p.num_v_heads;
  const int dk_pad = (dk + kDrChunk - 1) / kDrChunk * kDrChunk;
  float* S = smem;                                // [dk_pad][kDv]
  float* wk = S + dk_pad * kDv;                   // W^T slice [32][kDrLd], then k^T-scaled [64][kDrLd] (token-major)
  float* vs = wk + kDrChunk * kDrLd;              // [64][kDv + 4]
  float* dec = vs + kDrChunk * (kDv + 4);         // [dk_pad]: exp(G_last[d]) (KDA)
  float* gt = dec + dk_pad;                       // [64]: G_j (GDN), then rinv
  float* rinv = gt + 64;                          // [64]
  const int b = blockIdx.z, h = blockIdx.y, dv0 = blockIdx.x * kDv;
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int start = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - start;
  const int hk = h / (hv / p.num_k_heads);
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  float* pool = p.state_pool + slot * p.state_slot_stride + int64_t(h) * dk * dv;
  const auto pool_at = [&](int d, int c) -> int64_t { return p.state_transposed ? int64_t(c) * dk + d : int64_t(d) * dv + c; };

  for (int idx = tid; idx < dk_pad * kDv; idx += kDrThreads) {
    const int d = idx / kDv, c = idx % kDv;
    S[idx] = (has_init && d < dk && dv0 + c < dv) ? pool[pool_at(d, dv0 + c)] : 0.0f;
  }
  for (int chunk = 0; int64_t(chunk) * kDrChunk < n; ++chunk) {
    const int t0 = start + chunk * kDrChunk, m = min(kDrChunk, n - chunk * kDrChunk);
    __syncthreads();  // the previous chunk is done with S, dec, gt and rinv
    dr_row_rinv(rinv, k, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
    float g_last_scalar = 0.0f;
    if constexpr (kPerChannel) {
      for (int d = tid; d < dk_pad; d += kDrThreads) dec[d] = d < dk ? __expf(p.gc[(int64_t(t0 + m - 1) * hv + h) * dk + d]) : 0.0f;
    } else {
      if (tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
    }
    __syncthreads();
    if constexpr (!kPerChannel) g_last_scalar = gt[m - 1];
    // the state entering the chunk
    float* hbuf = p.h + (int64_t((t0 >> 6) + b) * hv + h) * dk * dv;
    for (int idx = tid; idx < dk * kDv; idx += kDrThreads) {
      const int d = idx / kDv, c = idx % kDv;
      if (dv0 + c < dv) hbuf[int64_t(d) * dv + dv0 + c] = S[idx];
    }
    // v_new = U - W S
    float acc[4][RN] = {};
    for (int d0 = 0; d0 < dk_pad; d0 += kDrSlice) {
      __syncthreads();
      for (int idx = tid; idx < kDrChunk * kDrSlice; idx += kDrThreads) {
        const int kk = idx % kDrSlice, j = idx / kDrSlice;
        wk[kk * kDrLd + j] = (j < m && d0 + kk < dk) ? p.w[(int64_t(t0 + j) * hv + h) * dk + d0 + kk] : 0.0f;
      }
      __syncthreads();
      smem_gemm_kmajor<4, RN>(acc, wk, kDrLd, ty * 4, S + d0 * kDv, kDv, tx * RN, kDrSlice);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = ty * 4 + r;
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        const int col = dv0 + tx * RN + c;
        float val = 0.0f;
        if (i < m && col < dv) {
          const int64_t at = (int64_t(t0 + i) * hv + h) * dv + col;
          val = p.u[at] - acc[r][c];
          p.v_new[at] = val;
        }
        vs[i * (kDv + 4) + tx * RN + c] = val;
      }
    }
    // the state after the chunk
    const int done = t0 + m - start;
    int64_t snapshot_slot = -1;
    if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
      const int j = done / p.snapshot_interval - 1;
      if (j < p.max_snapshots) snapshot_slot = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
    }
    const bool snapshot = snapshot_slot >= 0 && snapshot_slot < p.num_slots;
    for (int d0 = 0; d0 < dk_pad; d0 += kDrChunk) {
      __syncthreads();  // vs is complete; the W slice (which wk also holds) is consumed
      for (int idx = tid; idx < kDrChunk * kDrChunk; idx += kDrThreads) {
        const int kk = idx % kDrChunk, j = idx / kDrChunk;
        float val = 0.0f;
        if (j < m && d0 + kk < dk) {
          const int64_t tok = int64_t(t0 + j);
          const float to_end = kPerChannel ? __expf(p.gc[(int64_t(t0 + m - 1) * hv + h) * dk + d0 + kk] - p.gc[(tok * hv + h) * dk + d0 + kk])
                                           : __expf(g_last_scalar - gt[j]);
          val = to_float(k[tok * p.k_token_stride + d0 + kk]) * rinv[j] * to_end;
        }
        wk[j * kDrLd + kk] = val;
      }
      __syncthreads();
      float sacc[4][RN];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int d = d0 + ty * 4 + r;
        const float decay = kPerChannel ? dec[d] : __expf(g_last_scalar);
#pragma unroll
        for (int c = 0; c < RN; ++c) sacc[r][c] = decay * S[d * kDv + tx * RN + c];
      }
      smem_gemm_kmajor<4, RN>(sacc, wk, kDrLd, ty * 4, vs, kDv + 4, tx * RN, kDrChunk);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int d = d0 + ty * 4 + r;
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          S[d * kDv + tx * RN + c] = sacc[r][c];
          if (snapshot && d < dk && dv0 + tx * RN + c < dv)
            p.state_pool[snapshot_slot * p.state_slot_stride + int64_t(h) * dk * dv + pool_at(d, dv0 + tx * RN + c)] = sacc[r][c];
        }
      }
    }
  }
  __syncthreads();
  for (int idx = tid; idx < dk * kDv; idx += kDrThreads) {
    const int d = idx / kDv, c = idx % kDv;
    if (dv0 + c < dv) pool[pool_at(d, dv0 + c)] = S[idx];
  }
}
namespace {

// ---- chunk_output_fwd_kernel (7) ------

__host__ __device__ constexpr size_t dr_output_smem_bytes(bool per_channel, int dv_tile) {
  return ((per_channel ? 4 : 2) * size_t(kDrSlice) * kDrLd + size_t(kDrSlice) * (dv_tile + 4) + size_t(kDrChunk) * kDrLd +
          size_t(kDrChunk) * (dv_tile + 4) + 3 * kDrChunk) * sizeof(float);
}

/**
 * Grid (chunks * ceil(Dv / kDv), Hv, B): the kDv output columns of one chunk. The features are walked in slices of 32 that stage q and k
 * (normalised), for KDA also G and q Gamma, and the slice of the chunk's state h: Q h accumulates with the plain matmul (GDN: q h, scaled
 * by exp(G_i) afterwards; KDA: (q Gamma) h) while the 64 x 64 attention matrix is built beside it (GDN: q k^T, masked and scaled by
 * exp(G_i - G_m); KDA: the channel-wise sum, causal). Then o = scale (O + P v_new) with P^T staged k-major.
 */
}  // namespace
template <typename T, bool kPerChannel, int kDv>
__global__ __launch_bounds__(kDrThreads, 1) void chunk_output_fwd_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int RN = kDv / 16;
  extern __shared__ __align__(16) float smem[];
  const int dk = p.head_dim_k, dv = p.head_dim_v, hv = p.num_v_heads;
  const int dvt = (dv + kDv - 1) / kDv, chunk = blockIdx.x / dvt, dv0 = (blockIdx.x % dvt) * kDv;
  int t0, m;
  if (!dr_chunk_of(p, chunk, blockIdx.z, t0, m)) return;
  if (!dr_slot_ok(p, blockIdx.z)) return;
  float* qs = smem;                               // [32][kDrLd] feature-major
  float* ks = qs + kDrSlice * kDrLd;
  float* gs = ks + kDrSlice * kDrLd;              // KDA: G, q Gamma
  float* qd = kPerChannel ? gs + kDrSlice * kDrLd : gs;
  float* hs = (kPerChannel ? qd + kDrSlice * kDrLd : gs);  // [32][kDv + 4]
  float* ps = hs + kDrSlice * (kDv + 4);          // ps[j][i]: P transposed
  float* vs = ps + kDrChunk * kDrLd;              // [64][kDv + 4]
  float* rq = vs + kDrChunk * (kDv + 4);
  float* rk = rq + kDrChunk;
  float* gt = rk + kDrChunk;
  const int b = blockIdx.z, h = blockIdx.y, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const T* q = static_cast<const T*>(p.q) + int64_t(hk) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const float* hbuf = p.h + (int64_t((t0 >> 6) + b) * hv + h) * dk * dv;
  dr_row_rinv(rq, q, p.q_token_stride, t0, m, dk, p.use_qk_l2norm);
  __syncthreads();
  dr_row_rinv(rk, k, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
  if (!kPerChannel && tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
  float pacc[4][4] = {}, oacc[4][RN] = {};
  for (int d0 = 0; d0 < dk; d0 += kDrSlice) {
    __syncthreads();  // rk and gt are written; the previous slice is consumed
    for (int idx = tid; idx < kDrChunk * kDrSlice; idx += kDrThreads) {
      const int kk = idx % kDrSlice, i = idx / kDrSlice;
      float qv = 0.0f, kv = 0.0f, gv = 0.0f;
      if (i < m && d0 + kk < dk) {
        const int64_t tok = int64_t(t0 + i);
        qv = to_float(q[tok * p.q_token_stride + d0 + kk]) * rq[i];
        kv = to_float(k[tok * p.k_token_stride + d0 + kk]) * rk[i];
        if constexpr (kPerChannel) gv = p.gc[(tok * hv + h) * dk + d0 + kk];
      }
      qs[kk * kDrLd + i] = qv;
      ks[kk * kDrLd + i] = kv;
      if constexpr (kPerChannel) {
        gs[kk * kDrLd + i] = gv;
        qd[kk * kDrLd + i] = qv * __expf(gv);
      }
    }
    for (int idx = tid; idx < kDrSlice * kDv; idx += kDrThreads) {
      const int c = idx % kDv, kk = idx / kDv;
      hs[kk * (kDv + 4) + c] = (d0 + kk < dk && dv0 + c < dv) ? hbuf[int64_t(d0 + kk) * dv + dv0 + c] : 0.0f;
    }
    __syncthreads();
    if constexpr (!kPerChannel) {
      smem_gemm_kmajor<4, 4>(pacc, qs, kDrLd, ty * 4, ks, kDrLd, tx * 4, kDrSlice);
      smem_gemm_kmajor<4, RN>(oacc, qs, kDrLd, ty * 4, hs, kDv + 4, tx * RN, kDrSlice);
    } else {
      smem_gemm_kmajor<4, RN>(oacc, qd, kDrLd, ty * 4, hs, kDv + 4, tx * RN, kDrSlice);
      if (tx < ty) kda_pair_tile<false>(pacc, qs, gs, ks, gs, ty, tx);
      else if (tx == ty) kda_pair_tile<true, true>(pacc, qs, gs, ks, gs, ty, tx);
    }
  }
  // P (causal, decayed) transposed for the P v_new product; the GDN part from before the chunk is scaled by exp(G_i)
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    float col[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = ty * 4 + r, j = tx * 4 + c;
      col[r] = j <= i ? (kPerChannel ? pacc[r][c] : pacc[r][c] * __expf(gt[i] - gt[j])) : 0.0f;
    }
    *reinterpret_cast<float4*>(&ps[(tx * 4 + c) * kDrLd + ty * 4]) = make_float4(col[0], col[1], col[2], col[3]);
  }
  if constexpr (!kPerChannel) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float e = __expf(gt[ty * 4 + r]);
#pragma unroll
      for (int c = 0; c < RN; ++c) oacc[r][c] *= e;
    }
  }
  for (int idx = tid; idx < kDrChunk * kDv; idx += kDrThreads) {
    const int c = idx % kDv, j = idx / kDv;
    vs[j * (kDv + 4) + c] = (j < m && dv0 + c < dv) ? p.v_new[(int64_t(t0 + j) * hv + h) * dv + dv0 + c] : 0.0f;
  }
  __syncthreads();
  smem_gemm_kmajor<4, RN>(oacc, ps, kDrLd, ty * 4, vs, kDv + 4, tx * RN, kDrChunk);
  T* o = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = ty * 4 + r;
    if (i >= m) continue;
#pragma unroll
    for (int c = 0; c < RN; ++c) {
      if (dv0 + tx * RN + c < dv) o[int64_t(t0 + i) * p.o_token_stride + dv0 + tx * RN + c] = from_float<T>(oacc[r][c] * p.scale);
    }
  }
}
namespace {


// ---- ported from ayaka state/delta_rule_part2.cu:27-1070 ----

using bf16 = __nv_bfloat16;

constexpr int kTcWarps = 8;
constexpr int kTcLd64 = kDrChunk + 8;  // bf16 row stride of a 64-column tile: 144 bytes, ldmatrix is conflict-free

// shared device primitives (state_mma.cuh); the staging helpers take the CTA size
using mma::tc_ld;
using mma::tc_load8;
using mma::tc_load8_f32;
using mma::tc_stage_f32;
using mma::tc_stage_input;


// ---- chunk_scaled_dot_kkt, tensor cores (3) ------------------------------------------------------------------

/**
 * Grid (chunks, Hv, B), 256 threads. A_jm = beta_j rinv_j rinv_m (k_j . k_m) exp(G_j - G_m) for m < j, 0 on and above the diagonal (the
 * rows of the chunk are written whole). The raw k is the bf16 operand of both sides of the product (exact for bf16 inputs); the row norms
 * and the decay are applied to the fp32 accumulator. Warp w owns the 16 rows I = w & 3 and the column pairs 2 (w >> 2) + {0, 1} of 16.
 */
template <typename T, int kDk>
__global__ __launch_bounds__(kDrThreads, 2) void chunk_scaled_dot_kkt_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kLd = tc_ld(kDk);
  __shared__ __align__(16) bf16 ks[kDrChunk * kLd];
  __shared__ float rinv[kDrChunk], gt[kDrChunk];
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;  // uniform per CTA
  const int h = blockIdx.y, hv = p.num_v_heads, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const bool vec = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  dr_row_rinv(rinv, k, p.k_token_stride, t0, m, kDk, p.use_qk_l2norm);
  if (tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
  tc_stage_input<T, kDrThreads>(ks, kLd, k, p.k_token_stride, t0, m, kDk, vec);
  __syncthreads();
  const int I = wid & 3, half = wid >> 2;
  float acc[4][4] = {};
#pragma unroll
  for (int np = 0; np < 2; ++np) {
    if (2 * half + np > I) continue;  // wholly above the diagonal
#pragma unroll
    for (int kk = 0; kk < kDk / 16; ++kk) {
      uint32_t a[4], b[4];
      frag_a(a, ks, kLd, 16 * I, 16 * kk);
      frag_b2(b, ks, kLd, 32 * half + 16 * np, 16 * kk);
      mma_bf16_16x8x16(acc[2 * np], a, b[0], b[1]);
      mma_bf16_16x8x16(acc[2 * np + 1], a, b[2], b[3]);
    }
  }
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = 16 * I + g + 8 * rr;
    if (i >= m) continue;
    const float scale_i = p.beta[int64_t(t0 + i) * hv + h] * rinv[i], gi = gt[i];
    float* row = p.A + (int64_t(t0 + i) * hv + h) * kDrChunk;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int np = nt >> 1, j0 = 32 * half + 16 * np + 8 * (nt & 1) + 2 * t;
      float v0 = 0.0f, v1 = 0.0f;
      if (2 * half + np <= I) {
        v0 = j0 < i ? acc[nt][2 * rr] * scale_i * rinv[j0] * __expf(gi - gt[j0]) : 0.0f;
        v1 = j0 + 1 < i ? acc[nt][2 * rr + 1] * scale_i * rinv[j0 + 1] * __expf(gi - gt[j0 + 1]) : 0.0f;
      }
      *reinterpret_cast<float2*>(row + j0) = make_float2(v0, v1);
    }
  }
}

// ---- recompute_w_u, tensor cores (5) -------------------------------------------------------------------------

/**
 * Grid (chunks * column tiles, Hv, B): a 64 x 64 tile of W = T (beta exp(G) rinv k) (the first ceil(Dk / 64) tiles) or of U = T (beta v).
 * The per-row scalar s_m of the right operand is folded into the columns of T (T_jm s_m, one rounding), so that k and v enter the product
 * as exact bf16 values. T is lower triangular: warp (I = w & 3, half = w >> 2) multiplies the k-steps 0..I only.
 */
template <typename T, bool kPer>
__global__ __launch_bounds__(kDrThreads, 2) void recompute_w_u_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  __shared__ __align__(16) bf16 ts[kDrChunk * kTcLd64], bs[kDrChunk * kTcLd64];
  __shared__ float rinv[kDrChunk], sm[kDrChunk];
  const int dk = p.head_dim_k, dv = p.head_dim_v, hv = p.num_v_heads;
  const int dkt = (dk + kDrChunk - 1) / kDrChunk, dvt = (dv + kDrChunk - 1) / kDrChunk, coltiles = dkt + dvt;
  const int tile = blockIdx.x % coltiles, chunk = blockIdx.x / coltiles;
  int t0, m;
  if (!dr_chunk_of(p, chunk, blockIdx.z, t0, m)) return;
  const int h = blockIdx.y, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool is_w = tile < dkt;
  const int c0 = (is_w ? tile : tile - dkt) * kDrChunk;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + int64_t(h) * p.v_head_stride;
  if (is_w) dr_row_rinv(rinv, k, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
  __syncthreads();
  if (tid < kDrChunk) {
    float s = 0.0f;
    if (tid < m) {
      const int64_t tok = int64_t(t0 + tid);
      s = p.beta[tok * hv + h];
      if (is_w) s = kPer ? 1.0f : s * __expf(p.gc[tok * hv + h]) * rinv[tid];  // KDA: the decay is per channel, it goes into the right operand
    }
    sm[tid] = s;
  }
  // the right operand: raw k or v rows (bf16), zero past the sequence and the head dimension
  const T* src = is_w ? k : v;
  const int64_t stride = is_w ? p.k_token_stride : p.v_token_stride;
  const int width = is_w ? dk : dv;
  for (int idx = tid; idx < kDrChunk * (kDrChunk / 2); idx += kDrThreads) {
    const int mm = idx / (kDrChunk / 2), cc = (idx % (kDrChunk / 2)) * 2;
    float x0 = 0.0f, x1 = 0.0f;
    if (mm < m) {
      const T* row = src + int64_t(t0 + mm) * stride + c0 + cc;
      if (c0 + cc < width) x0 = to_float(row[0]);
      if (c0 + cc + 1 < width) x1 = to_float(row[1]);
      if constexpr (kPer) {
        if (is_w) {
          const int64_t tk = int64_t(t0 + mm);
          const float f = p.beta[tk * hv + h] * rinv[mm];
          const float* gcp = p.gc + (tk * hv + h) * dk + c0 + cc;
          x0 = c0 + cc < width ? x0 * f * __expf(gcp[0]) : 0.0f;
          x1 = c0 + cc + 1 < width ? x1 * f * __expf(gcp[1]) : 0.0f;
        }
      }
    }
    *reinterpret_cast<uint32_t*>(&bs[mm * kTcLd64 + cc]) = pack_bf16x2(x0, x1);
  }
  __syncthreads();
  // T s: the unit lower triangle of the inverse (fp32 in A), columns scaled by s_m, rows past the sequence zero
  for (int idx = tid; idx < kDrChunk * (kDrChunk / 4); idx += kDrThreads) {
    const int j = idx / (kDrChunk / 4), mm = (idx % (kDrChunk / 4)) * 4;
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (j < m && mm <= j) a = *reinterpret_cast<const float4*>(&p.A[(int64_t(t0 + j) * hv + h) * kDrChunk + mm]);
    const float x0 = mm <= j ? a.x * sm[mm] : 0.0f, x1 = mm + 1 <= j ? a.y * sm[mm + 1] : 0.0f;
    const float x2 = mm + 2 <= j ? a.z * sm[mm + 2] : 0.0f, x3 = mm + 3 <= j ? a.w * sm[mm + 3] : 0.0f;
    *reinterpret_cast<uint2*>(&ts[j * kTcLd64 + mm]) = make_uint2(pack_bf16x2(x0, x1), pack_bf16x2(x2, x3));
  }
  __syncthreads();
  const int I = wid & 3, half = wid >> 2;
  float acc[4][4] = {};
  for (int kk = 0; kk <= I; ++kk) {
    uint32_t a[4];
    frag_a(a, ts, kTcLd64, 16 * I, 16 * kk);
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t b[4];
      frag_b2_trans(b, bs, kTcLd64, 16 * kk, 32 * half + 16 * np);
      mma_bf16_16x8x16(acc[2 * np], a, b[0], b[1]);
      mma_bf16_16x8x16(acc[2 * np + 1], a, b[2], b[3]);
    }
  }
  float* out = is_w ? p.w : p.u;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int j = 16 * I + g + 8 * rr;
    if (j >= m) continue;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = c0 + 32 * half + 8 * nt + 2 * t;
      float* o = out + (int64_t(t0 + j) * hv + h) * width + col;
      if (col < width) o[0] = acc[nt][2 * rr];
      if (col + 1 < width) o[1] = acc[nt][2 * rr + 1];
    }
  }
}

// ---- chunk_state_fwd, tensor cores (6) -----------------------------------------------------------------------

/** Dynamic shared memory of the state kernel: Sb [Dk][kDv + 8], Vb [64][kDv + 8], Wb [64][Dk + 8], Kt [2][64][Dk + 8] (double buffered) in bf16, then
 * rinv and gt [2][64] and gl [2][Dk] (KDA) in fp32. */
template <int kDk, int kDv>
__host__ __device__ constexpr size_t tc_state_smem_bytes() {
  return (size_t(kDk) * tc_ld(kDv) + size_t(kDrChunk) * tc_ld(kDv) + 3 * size_t(kDrChunk) * tc_ld(kDk)) * sizeof(bf16) +
         (4 * kDrChunk + 2 * kDk) * sizeof(float);
}

/**
 * Grid (ceil(Dv / kDv), Hv, B): a CTA owns the S[:, dv tile] columns of one (sequence, value head) and walks its chunks in order, exactly as
 * the fp32 kernel. The state lives in the accumulator registers of the warps that own its row tiles (16 rows of Dk each; with Dk < 128 the
 * extra warps split the columns); a bf16 copy in shared memory is the right operand of v_new = U - W S. Per chunk: h <- S (fp32), v_new = U - W S
 * (fp32 to v_new, bf16 to the update), S <- exp(G_last) S + (k exp(G_last - G) rinv)^T v_new.
 * GDN: the token scalar rinv exp(G_last - G_j) is folded into the v_new operand, so k enters the update as an exact bf16 value. KDA: the
 * decay is per channel (exp(G_last[d] - G_j[d]) rinv_j goes into the k tile, exp(G_last[d]) scales the rows of S).
 *
 * The loop is software pipelined: after the first product of chunk c the fp32 W, the k rows (and for KDA their decays) of chunk c + 1 are
 * loaded into registers, they land while the second product runs, and are then converted into the W tile (which no one reads any more) and the
 * other half of the double-buffered k tile; two barriers per chunk.
 */
template <typename T, bool kPer, int kDk, int kDv, bool kVec>
__global__ __launch_bounds__(kDrThreads, 1) void chunk_state_fwd_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kNmt = kDk / 16, kNts = kDv / 8;
  constexpr int kWpm = (8 / kNmt) < kNts ? (8 / kNmt) : kNts;  // warps per row tile
  constexpr int kNtw = kNts / kWpm;                              // n8 tiles of the state per owner warp
  constexpr int kNtv = kNts / 2;                                 // n8 tiles of v_new per warp (two column halves)
  constexpr int kLdS = tc_ld(kDv), kLdK = tc_ld(kDk);
  constexpr int kPieces = kDk / 8;                               // 8-element pieces of a row of W / k
  constexpr int kPerThread = (kDrChunk * kPieces + kDrThreads - 1) / kDrThreads;
  static_assert(kNtv >= 2 && kNtv % 2 == 0, "v_new is formed two n8 tiles at a time");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  bf16* Sb = reinterpret_cast<bf16*>(tc_smem);   // [kDk][kLdS]: the state, k-major for the B fragments
  bf16* Vb = Sb + kDk * kLdS;                    // [64][kLdS]
  bf16* Wb = Vb + kDrChunk * kLdS;               // [64][kLdK]
  bf16* Kt = Wb + kDrChunk * kLdK;               // [2][64][kLdK]
  float* rinv = reinterpret_cast<float*>(Kt + 2 * kDrChunk * kLdK);  // [2][64]
  float* gt = rinv + 2 * kDrChunk;               // [2][64]
  float* gl = gt + 2 * kDrChunk;                 // [2][kDk]: G_last per channel (KDA)
  const int dk = kDk, dv = p.head_dim_v, hv = p.num_v_heads;
  const int b = blockIdx.z, h = blockIdx.y, dv0 = blockIdx.x * kDv;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int start = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - start;
  const int hk = h / (hv / p.num_k_heads);
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  float* pool = p.state_pool + slot * p.state_slot_stride + int64_t(h) * dk * dv;
  const auto pool_at = [&](int d, int c) -> int64_t { return p.state_transposed ? int64_t(c) * dk + d : int64_t(d) * dv + c; };
  const int nchunks = (n + kDrChunk - 1) / kDrChunk;

  // the owner of state tile (row tile mt, columns [8 ntb, 8 (ntb + kNtw))) of this warp
  const bool owner = wid < kNmt * kWpm;
  const int mt = wid % kNmt, ntb = (wid / kNmt) * kNtw;
  float S[(kNtw + 1) / 2 * 2][4] = {};  // an even number of tiles: the update forms two n8 tiles per fragment load
#pragma unroll
  for (int nt = 0; nt < kNtw; ++nt) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
      S[nt][e] = (owner && has_init && dv0 + c < dv) ? pool[pool_at(d, dv0 + c)] : 0.0f;
    }
  }
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2)
        *reinterpret_cast<uint32_t*>(&Sb[(16 * mt + g + 8 * e2) * kLdS + 8 * (ntb + nt) + 2 * t]) = pack_bf16x2(S[nt][2 * e2], S[nt][2 * e2 + 1]);
  }

  // ---- the prefetch of one chunk: the pieces of W (and, for GDN, k) this thread stages, and the cumulative decay of the chunk. KDA loads its k rows and
  // their decays inside stage() instead (their 64 registers do not fit next to the state), which exposes one load latency per chunk.
  constexpr bool kRawK = std::is_same_v<T, bf16>;  // bfloat16 k: keep the loaded 16-byte pieces (4 registers per piece) and widen them in stage()
  struct Next {
    float w[kPerThread][8];
    float k[(!kPer && !kRawK) ? kPerThread : 1][8];
    uint4 kraw[(!kPer && kRawK) ? kPerThread : 1];
    float gl8[kPer ? 8 : 1];
    float gt;
    int t0, m;
  } nx;
  const auto fetch = [&](int chunk) {
    nx.t0 = start + chunk * kDrChunk;
    nx.m = min(kDrChunk, n - chunk * kDrChunk);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
        (void)r, (void)c8;  // unused on the raw-k path
      const bool in = piece < kDrChunk * kPieces && r < nx.m;
      if (in) {
        tc_load8_f32(nx.w[i], p.w + (int64_t(nx.t0 + r) * hv + h) * dk + c8, kVec);
        if constexpr (!kPer) {
          if constexpr (kRawK) {
            nx.kraw[i] = kVec ? *reinterpret_cast<const uint4*>(k + int64_t(nx.t0 + r) * p.k_token_stride + c8) : [&] {
              float tmp[8];
              tc_load8<T>(tmp, k + int64_t(nx.t0 + r) * p.k_token_stride + c8, false);
              return make_uint4(pack_bf16x2(tmp[0], tmp[1]), pack_bf16x2(tmp[2], tmp[3]), pack_bf16x2(tmp[4], tmp[5]), pack_bf16x2(tmp[6], tmp[7]));
            }();
          } else {
            tc_load8<T>(nx.k[i], k + int64_t(nx.t0 + r) * p.k_token_stride + c8, kVec);
          }
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          nx.w[i][e] = 0.0f;
          if constexpr (!kPer && !kRawK) nx.k[i][e] = 0.0f;
        }
        if constexpr (!kPer && kRawK) nx.kraw[i] = make_uint4(0, 0, 0, 0);
      }
    }
    if constexpr (kPer) {
      tc_load8_f32(nx.gl8, p.gc + (int64_t(nx.t0 + nx.m - 1) * hv + h) * dk + ((tid % kPieces) * 8), kVec);
    } else {
      nx.gt = tid < kDrChunk && tid < nx.m ? p.gc[int64_t(nx.t0 + tid) * hv + h] : 0.0f;
    }
  };
  // convert the prefetched chunk into buffer `bf`: W tile (bf16), k tile (raw bf16 for GDN, scaled for KDA), rinv, gt (GDN) / gl (KDA).
  // KDA works on two pieces at a time (it loads them here), so that the loaded k and decays take 32 registers, not 64.
  const auto stage = [&](int bf) {
    bf16* wb = Wb;
    bf16* kt = Kt + bf * kDrChunk * kLdK;
    constexpr int kBatch = kPer ? (kPerThread < 2 ? kPerThread : 2) : kPerThread;
#pragma unroll
    for (int i0 = 0; i0 < kPerThread; i0 += kBatch) {
      float kv[kBatch][8];  // the 8 elements of each piece of k as floats, and (KDA) their decays
      float gcv[kPer ? kBatch : 1][8];
      (void)gcv;
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int i = i0 + j, piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
        (void)r, (void)c8;  // unused on the raw-k path
        if constexpr (kPer) {
          const bool in = piece < kDrChunk * kPieces && r < nx.m;
          if (in) {
            tc_load8<T>(kv[j], k + int64_t(nx.t0 + r) * p.k_token_stride + c8, kVec);
            tc_load8_f32(gcv[j], p.gc + (int64_t(nx.t0 + r) * hv + h) * dk + c8, kVec);
          } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) kv[j][e] = 0.0f, gcv[j][e] = 0.0f;
          }
        } else if constexpr (kRawK) {
          const uint32_t wd[4] = {nx.kraw[i].x, nx.kraw[i].y, nx.kraw[i].z, nx.kraw[i].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            kv[j][2 * e] = __uint_as_float(wd[e] << 16);
            kv[j][2 * e + 1] = __uint_as_float(wd[e] & 0xffff0000u);
          }
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) kv[j][e] = nx.k[i][e];
        }
      }
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int i = i0 + j, piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
        (void)r, (void)c8;  // unused on the raw-k path
        float sq = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) sq = __fmaf_rn(kv[j][e], kv[j][e], sq);
        // the pieces of one row are consecutive lanes of the same warp (kPieces divides 32): add them with the xor butterfly over the row's lanes
#pragma unroll
        for (int off = kPieces / 2; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
        if (piece >= kDrChunk * kPieces) continue;
        *reinterpret_cast<uint4*>(wb + r * kLdK + c8) = make_uint4(pack_bf16x2(nx.w[i][0], nx.w[i][1]), pack_bf16x2(nx.w[i][2], nx.w[i][3]),
                                                                     pack_bf16x2(nx.w[i][4], nx.w[i][5]), pack_bf16x2(nx.w[i][6], nx.w[i][7]));
        const float ri = (r < nx.m && p.use_qk_l2norm) ? rsqrtf(sq + kL2Eps) : 1.0f;
        if ((tid % kPieces) == 0) rinv[bf * kDrChunk + r] = ri;
        float x[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if constexpr (kPer) x[e] = kv[j][e] * ri * __expf(nx.gl8[e] - gcv[j][e]);
          else x[e] = kv[j][e];
        }
        *reinterpret_cast<uint4*>(kt + r * kLdK + c8) = make_uint4(pack_bf16x2(x[0], x[1]), pack_bf16x2(x[2], x[3]), pack_bf16x2(x[4], x[5]), pack_bf16x2(x[6], x[7]));
      }
    }
    if constexpr (kPer) {
      if (tid < kPieces) {
#pragma unroll
        for (int e = 0; e < 8; ++e) gl[bf * kDk + tid * 8 + e] = nx.gl8[e];
      }
    } else {
      if (tid < kDrChunk) gt[bf * kDrChunk + tid] = nx.gt;
    }
  };

  if (nchunks > 0) {
    fetch(0);
    stage(0);
  }
  __syncthreads();
  // One chunk. The prefetch of the next chunk is a compile-time choice: the last chunk has none, and keeping `fetch` and `stage` in one
  // straight-line region (instead of under two runtime conditions) lets the compiler end the life of the ~100 prefetch registers at `stage`.
  const auto iteration = [&](int chunk, auto prefetch) {
    constexpr bool kPrefetch = decltype(prefetch)::value;
    const int bf = chunk & 1;
    const int t0 = start + chunk * kDrChunk, m = min(kDrChunk, n - chunk * kDrChunk);
    // the loads of the next chunk and of this chunk's u go out before the products
    float ur[kNtv][4] = {};
    {
      const int I = wid & 3, half = wid >> 2;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int j = 16 * I + g + 8 * rr;
#pragma unroll
        for (int nt = 0; nt < kNtv; ++nt) {
          const int c = 8 * (half * kNtv + nt) + 2 * t;
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (j < m && dv0 + c + e < dv) ur[nt][2 * rr + e] = p.u[(int64_t(t0 + j) * hv + h) * dv + dv0 + c + e];
        }
      }
    }
    // the state entering the chunk
    float* hbuf = p.h + (int64_t((t0 >> 6) + b) * hv + h) * dk * dv;
    if (owner) {
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
          if (dv0 + c < dv) hbuf[int64_t(d) * dv + dv0 + c] = S[nt][e];
        }
      }
    }
    const float g_last = kPer ? 0.0f : gt[bf * kDrChunk + m - 1];
    // v_new = U - W S: rows I = wid & 3 of the chunk, columns of half wid >> 2
    {
      const int I = wid & 3, half = wid >> 2;
      float acc[kNtv][4] = {};
      const bf16* wb = Wb;
#pragma unroll 2
      for (int kk = 0; kk < kDk / 16; ++kk) {
        uint32_t a[4];
        frag_a(a, wb, kLdK, 16 * I, 16 * kk);
#pragma unroll
        for (int np = 0; np < kNtv / 2; ++np) {
          uint32_t bb[4];
          frag_b2_trans(bb, Sb, kLdS, 16 * kk, 8 * (half * kNtv) + 16 * np);
          mma_bf16_16x8x16(acc[2 * np], a, bb[0], bb[1]);
          mma_bf16_16x8x16(acc[2 * np + 1], a, bb[2], bb[3]);
        }
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int j = 16 * I + g + 8 * rr;
        const float fold = kPer ? 1.0f : (j < m ? rinv[bf * kDrChunk + j] * __expf(g_last - gt[bf * kDrChunk + j]) : 0.0f);
#pragma unroll
        for (int nt = 0; nt < kNtv; ++nt) {
          const int c = 8 * (half * kNtv + nt) + 2 * t;
          float val[2] = {0.0f, 0.0f};
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            if (j < m && dv0 + c + e < dv) {
              val[e] = ur[nt][2 * rr + e] - acc[nt][2 * rr + e];
              p.v_new[(int64_t(t0 + j) * hv + h) * dv + dv0 + c + e] = val[e];
            }
          }
          *reinterpret_cast<uint32_t*>(&Vb[j * kLdS + c]) = pack_bf16x2(val[0] * fold, val[1] * fold);
        }
      }
    }
    __syncthreads();  // Vb is complete; every warp is done with Sb and with this chunk's W tile
    if constexpr (kPrefetch) fetch(chunk + 1);  // the loads land during the second product
    // S <- exp(G_last) S + k^T (rinv exp(G_last - G) v_new)
    const int done = t0 + m - start;
    int64_t snapshot_slot = -1;
    if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
      const int j = done / p.snapshot_interval - 1;
      if (j < p.max_snapshots) snapshot_slot = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
    }
    const bool snapshot = snapshot_slot >= 0 && snapshot_slot < p.num_slots;
    if (owner) {
      const float decay_rows[2] = {kPer ? __expf(gl[bf * kDk + 16 * mt + g]) : __expf(g_last), kPer ? __expf(gl[bf * kDk + 16 * mt + g + 8]) : __expf(g_last)};
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) S[nt][e] *= decay_rows[e >> 1];
      const bf16* kt = Kt + bf * kDrChunk * kLdK;
#pragma unroll 1
      for (int kk = 0; kk < kDrChunk / 16; ++kk) {
        uint32_t a[4];
        frag_a_trans(a, kt, kLdK, 16 * mt, 16 * kk);
#pragma unroll
        for (int np = 0; np < (kNtw + 1) / 2; ++np) {
          uint32_t bb[4];
          frag_b2_trans(bb, Vb, kLdS, 16 * kk, 8 * ntb + 16 * np);
          mma_bf16_16x8x16(S[2 * np], a, bb[0], bb[1]);
          mma_bf16_16x8x16(S[2 * np + 1], a, bb[2], bb[3]);  // for an odd kNtw this tile is padding and never read
        }
      }
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt) {
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          *reinterpret_cast<uint32_t*>(&Sb[(16 * mt + g + 8 * e2) * kLdS + 8 * (ntb + nt) + 2 * t]) = pack_bf16x2(S[nt][2 * e2], S[nt][2 * e2 + 1]);
        }
        if (snapshot) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
            if (dv0 + c < dv) p.state_pool[snapshot_slot * p.state_slot_stride + int64_t(h) * dk * dv + pool_at(d, dv0 + c)] = S[nt][e];
          }
        }
      }
    }
    if constexpr (kPrefetch) stage(bf ^ 1);  // the W tile (free), the other half of the k tile, rinv, gt / gl: nobody reads them until the next iteration
    __syncthreads();  // Sb and the staged tiles: the next chunk may read them, and this chunk's Kt / Vb are consumed
  };
  for (int chunk = 0; chunk + 1 < nchunks; ++chunk) iteration(chunk, std::true_type{});
  if (nchunks > 0) iteration(nchunks - 1, std::false_type{});
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
        if (dv0 + c < dv) pool[pool_at(d, dv0 + c)] = S[nt][e];
      }
    }
  }
}

// ---- chunk_output_fwd, tensor cores (7) ----------------------------------------------------------------------

/** Dynamic shared memory of the output kernel: bf16 slice tiles Qs, Ks [64][kSl + 8], Hs [kSl][kDv + 8], Pb [64][72], Vb [64][kDv + 8]; fp32 rq, rk, gt. */
template <int kDk, int kDv>
__host__ __device__ constexpr size_t tc_output_smem_bytes() {
  constexpr int kSl = kDk >= 32 ? 32 : 16;
  return (2 * size_t(kDrChunk) * tc_ld(kSl) + size_t(kSl) * tc_ld(kDv) + size_t(kDrChunk) * kTcLd64 + size_t(kDrChunk) * tc_ld(kDv)) * sizeof(bf16) +
         3 * kDrChunk * sizeof(float);
}

/**
 * Grid (chunks * ceil(Dv / kDv), Hv, B): the kDv output columns of one chunk. o = scale (exp(G_i) rq_i (q h) + P v_new) with
 * P_ij = rq_i rk_j (q_i . k_j) exp(G_i - G_j) for j <= i: q and k are the raw bf16 operands of q k^T and the row norms and decays scale the
 * fp32 accumulators; P (bf16) goes through shared memory because the two warps of a row tile hold different column pairs of it. The channels of
 * Dk are walked in slices of 32 (q, k as bf16 tiles, the matching rows of h), which keeps the CTA small enough for two per SM; q k^T and q h
 * accumulate across the slices in registers.
 */
template <typename T, int kDk, int kDv>
__global__ __launch_bounds__(kDrThreads, 2) void chunk_output_fwd_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kSl = kDk >= 32 ? 32 : 16, kNts = kDv / 8, kNt1 = kNts / 2, kLdQ = tc_ld(kSl), kLdV = tc_ld(kDv);
  static_assert(kNt1 >= 2 && kNt1 % 2 == 0, "the output tile is formed two n8 tiles at a time");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  bf16* Qs = reinterpret_cast<bf16*>(tc_smem);  // [64][kLdQ]
  bf16* Ks = Qs + kDrChunk * kLdQ;              // [64][kLdQ]
  bf16* Hs = Ks + kDrChunk * kLdQ;              // [kSl][kLdV]
  bf16* Pb = Hs + kSl * kLdV;                   // [64][kTcLd64]
  bf16* Vb = Pb + kDrChunk * kTcLd64;           // [64][kLdV]
  float* rq = reinterpret_cast<float*>(Vb + kDrChunk * kLdV);
  float* rk = rq + kDrChunk;
  float* gt = rk + kDrChunk;
  const int dv = p.head_dim_v, hv = p.num_v_heads;
  const int dvt = (dv + kDv - 1) / kDv, chunk = blockIdx.x / dvt, dv0 = (blockIdx.x % dvt) * kDv;
  int t0, m;
  if (!dr_chunk_of(p, chunk, blockIdx.z, t0, m)) return;
  if (!dr_slot_ok(p, blockIdx.z)) return;
  const int b = blockIdx.z, h = blockIdx.y, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const T* q = static_cast<const T*>(p.q) + int64_t(hk) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const bool vec_q = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(q) % 16 == 0 && p.q_token_stride % 8 == 0;
  const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  const bool vec_f = dv % 4 == 0 && reinterpret_cast<uintptr_t>(p.h) % 16 == 0 && reinterpret_cast<uintptr_t>(p.v_new) % 16 == 0 && (dv0 % 4 == 0);
  const float* hbuf = p.h + (int64_t((t0 >> 6) + b) * hv + h) * kDk * dv;
  dr_row_rinv(rq, q, p.q_token_stride, t0, m, kDk, p.use_qk_l2norm);
  __syncthreads();
  dr_row_rinv(rk, k, p.k_token_stride, t0, m, kDk, p.use_qk_l2norm);
  if (tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
  tc_stage_f32<bf16, kDrThreads>(Vb, kLdV, p.v_new + (int64_t(t0) * hv + h) * dv + dv0, int64_t(hv) * dv, m, kDrChunk, kDv, min(kDv, dv - dv0), vec_f);
  const int I = wid & 3, half = wid >> 2;
  float sacc[4][4] = {}, o[kNt1][4] = {};
  for (int d0 = 0; d0 < kDk; d0 += kSl) {
    __syncthreads();  // rk, gt and Vb on the first slice; the previous slice is consumed afterwards
    tc_stage_input<T, kDrThreads>(Qs, kLdQ, q + d0, p.q_token_stride, t0, m, kSl, vec_q);
    tc_stage_input<T, kDrThreads>(Ks, kLdQ, k + d0, p.k_token_stride, t0, m, kSl, vec_k);
    tc_stage_f32<bf16, kDrThreads>(Hs, kLdV, hbuf + int64_t(d0) * dv + dv0, dv, kSl, kSl, kDv, min(kDv, dv - dv0), vec_f);
    __syncthreads();
    // S += q k^T over the slice: column pairs 2 half + {0, 1} of 16, only on and below the diagonal
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      if (2 * half + np > I) continue;
#pragma unroll
      for (int kk = 0; kk < kSl / 16; ++kk) {
        uint32_t a[4], bb[4];
        frag_a(a, Qs, kLdQ, 16 * I, 16 * kk);
        frag_b2(bb, Ks, kLdQ, 32 * half + 16 * np, 16 * kk);
        mma_bf16_16x8x16(sacc[2 * np], a, bb[0], bb[1]);
        mma_bf16_16x8x16(sacc[2 * np + 1], a, bb[2], bb[3]);
      }
    }
    // O += q h over the slice
#pragma unroll
    for (int kk = 0; kk < kSl / 16; ++kk) {
      uint32_t a[4];
      frag_a(a, Qs, kLdQ, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < kNt1 / 2; ++np) {
        uint32_t bb[4];
        frag_b2_trans(bb, Hs, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
        mma_bf16_16x8x16(o[2 * np], a, bb[0], bb[1]);
        mma_bf16_16x8x16(o[2 * np + 1], a, bb[2], bb[3]);
      }
    }
  }
  // P = causal, decayed q k^T (bf16 into Pb)
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = 16 * I + g + 8 * rr;
    const float ri = rq[i], gi = gt[i];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int np = nt >> 1, j0 = 32 * half + 16 * np + 8 * (nt & 1) + 2 * t;
      if (2 * half + np > I) continue;
      const float v0 = j0 <= i ? sacc[nt][2 * rr] * ri * rk[j0] * __expf(gi - gt[j0]) : 0.0f;
      const float v1 = j0 + 1 <= i ? sacc[nt][2 * rr + 1] * ri * rk[j0 + 1] * __expf(gi - gt[j0 + 1]) : 0.0f;
      *reinterpret_cast<uint32_t*>(&Pb[i * kTcLd64 + j0]) = pack_bf16x2(v0, v1);
    }
  }
  // O scaled by rq exp(G)
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = 16 * I + g + 8 * rr;
    const float f = rq[i] * __expf(gt[i]);
#pragma unroll
    for (int nt = 0; nt < kNt1; ++nt) {
      o[nt][2 * rr] *= f;
      o[nt][2 * rr + 1] *= f;
    }
  }
  __syncthreads();  // Pb is complete
  for (int kk = 0; kk <= I; ++kk) {
    uint32_t a[4];
    frag_a(a, Pb, kTcLd64, 16 * I, 16 * kk);
#pragma unroll
    for (int np = 0; np < kNt1 / 2; ++np) {
      uint32_t bb[4];
      frag_b2_trans(bb, Vb, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
      mma_bf16_16x8x16(o[2 * np], a, bb[0], bb[1]);
      mma_bf16_16x8x16(o[2 * np + 1], a, bb[2], bb[3]);
    }
  }
  T* out = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = 16 * I + g + 8 * rr;
    if (i >= m) continue;
#pragma unroll
    for (int nt = 0; nt < kNt1; ++nt) {
      const int c = dv0 + 8 * (half * kNt1 + nt) + 2 * t;
      if (c < dv) out[int64_t(t0 + i) * p.o_token_stride + c] = from_float<T>(o[nt][2 * rr] * p.scale);
      if (c + 1 < dv) out[int64_t(t0 + i) * p.o_token_stride + c + 1] = from_float<T>(o[nt][2 * rr + 1] * p.scale);
    }
  }
}

// ---- KDA: the decayed pairs of a chunk (steps 3 and 7) --------------------------------------------------------
//
// With one log-decay per key channel, P_ij = sum_d x_i[d] y_j[d] exp(G_i[d] - G_j[d]) (j < i, or j <= i) cannot be a plain matmul. The 64
// tokens form four tiles of 16. For a tile pair (I > J) both factors exp(G_i[d] - Gref[d]) and exp(Gref[d] - G_j[d]), with Gref the
// cumulative decay of the last token of tile J, are <= 1 (G falls with the token): the pair is an mma on the scaled operands and no
// exponent can overflow. Inside a tile (I == J) every pair keeps its own exponent and is summed in fp32 by the threads, as in the fp32
// kernels. The channels are walked in slices; the slice tiles are fp32 and feature-major ([channel][token], stride 68) in shared memory.

constexpr int kKdaLd = kDrLd;

/**
 * acc[nt] += sum over the slice of Q'[i][d] K'[j][d] for the 16 x 16 tile (I, J), rows 16 I.., columns 16 J.. (n8 tiles nt = 0, 1), with
 * Q'[i][d] = q_i[d] exp(G_i[d] - G_ref[d]) and K'[j][d] = k_j[d] exp(G_ref[d] - G_j[d]) where `jref` is the token whose cumulative decay is the
 * reference: the last token of tile J for an off-diagonal tile (both exponents <= 0), the first token of the tile for a diagonal one (the second
 * exponent is then >= 0 and bounded by the decay inside the tile, which kda_tile_safe has checked).
 */
template <int kSl>
__device__ __forceinline__ void kda_tile_slice(float (&acc)[2][4], const float* qs, const float* ks, const float* gs, int I, int J, int jref) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int kk = 0; kk < kSl / 16; ++kk) {
    uint32_t a[4], b[2][2];
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
      const int d = 16 * kk + 2 * t + 8 * hi;
      const float gr0 = gs[d * kKdaLd + jref], gr1 = gs[(d + 1) * kKdaLd + jref];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int i = 16 * I + g + 8 * r;
        a[r + 2 * hi] = pack_bf16x2(qs[d * kKdaLd + i] * __expf(gs[d * kKdaLd + i] - gr0), qs[(d + 1) * kKdaLd + i] * __expf(gs[(d + 1) * kKdaLd + i] - gr1));
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int j = 16 * J + 8 * nt + g;
        b[nt][hi] = pack_bf16x2(ks[d * kKdaLd + j] * __expf(gr0 - gs[d * kKdaLd + j]), ks[(d + 1) * kKdaLd + j] * __expf(gr1 - gs[(d + 1) * kKdaLd + j]));
      }
    }
    mma_bf16_16x8x16(acc[0], a, b[0][0], b[0][1]);
    mma_bf16_16x8x16(acc[1], a, b[1][0], b[1][1]);
  }
}

template <int kSl>
__device__ __forceinline__ void kda_offdiag_slice(float (&acc)[2][4], const float* qs, const float* ks, const float* gs, int I, int J) {
  kda_tile_slice<kSl>(acc, qs, ks, gs, I, J, 16 * J + 15);
}

/** The decay of a slice across tile J stays within a bound for which the factored (tensor-core) form of the diagonal tile cannot overflow: for all
 * channels G(first token) - G(last token) < 60. Warp-uniform (every lane of the warp calls it); NaN is "not safe". */
template <int kSl>
__device__ __forceinline__ bool kda_tile_safe(const float* gs, int J) {
  const int lane = threadIdx.x & 31;
  float dmax = 0.0f;
  if (lane < kSl) dmax = gs[lane * kKdaLd + 16 * J] - gs[lane * kKdaLd + 16 * J + 15];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) dmax = fmaxf(dmax, __shfl_xor_sync(0xffffffffu, dmax, off));
  return dmax < 60.0f;
}

/** The exact form of a diagonal tile for one warp: lane l owns the pairs (row 16 J + l / 2, columns 16 J + 8 (l % 2) + 0..7) with j <= i (kInclusive) or
 * j < i, every pair with its own exponent, as the fp32 kernels do. */
template <bool kInclusive, int kSl>
__device__ __forceinline__ void kda_diag_exact_warp(float (&acc)[8], const float* qs, const float* ks, const float* gs, int J) {
  const int lane = threadIdx.x & 31, i = 16 * J + (lane >> 1), jb = 16 * J + 8 * (lane & 1);
#pragma unroll 4
  for (int kk = 0; kk < kSl; ++kk) {
    const float qv = qs[kk * kKdaLd + i], gi = gs[kk * kKdaLd + i];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int j = jb + c;
      if (kInclusive ? (j <= i) : (j < i)) acc[c] = __fmaf_rn(qv * ks[kk * kKdaLd + j], __expf(gi - gs[kk * kKdaLd + j]), acc[c]);
    }
  }
}

/** One diagonal tile of a warp for one slice: the tensor-core form when the decay allows it, the exact one otherwise (the two add into separate
 * accumulators, combined by kda_diag_store). */
template <bool kInclusive, int kSl>
__device__ __forceinline__ void kda_diag_tile_slice(float (&fast)[2][4], float (&slow)[8], const float* qs, const float* ks, const float* gs, int J) {
  if (kda_tile_safe<kSl>(gs, J)) kda_tile_slice<kSl>(fast, qs, ks, gs, J, J, 16 * J);
  else kda_diag_exact_warp<kInclusive, kSl>(slow, qs, ks, gs, J);
}

/** The 16 x 16 tile (fast + slow) as fp32 in `tile` (row stride `ld`): the fast accumulator in its fragment layout, then the exact pairs added by their owners. */
__device__ __forceinline__ void kda_diag_store(float* tile, int ld, const float (&fast)[2][4], const float (&slow)[8]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) tile[(g + 8 * (e >> 1)) * ld + 8 * nt + 2 * t + (e & 1)] = fast[nt][e];
  __syncwarp();
#pragma unroll
  for (int c = 0; c < 8; ++c) tile[(lane >> 1) * ld + 8 * (lane & 1) + c] += slow[c];
  __syncwarp();
}

/** The off-diagonal tile (I, J) a warp owns in the KDA kernels: warps 0..5 take (1,0) (2,0) (2,1) (3,0) (3,1) (3,2). */
__device__ __forceinline__ void kda_pair_of_warp(int wid, int& I, int& J) {
  I = wid == 0 ? 1 : (wid < 3 ? 2 : 3);
  J = wid - (I == 1 ? 0 : (I == 2 ? 1 : 3));
}

// ---- chunk_scaled_dot_kkt, tensor cores, KDA (3) -------------------------------------------------------------

/**
 * Grid (chunks, Hv, B), 256 threads. A_jm = beta_j sum_d k_j[d] k_m[d] exp(G_j[d] - G_m[d]) (rinv_j rinv_m folded into the staged k) for m < j,
 * 0 elsewhere: the off-diagonal tiles on mma (warps 0..5), the diagonal ones (all threads, 2 x 2 pairs each) in fp32; the 64 x 64 result goes
 * through shared memory so that whole rows are written.
 */
template <typename T, int kDk>
__global__ __launch_bounds__(kDrThreads, 2) void chunk_scaled_dot_kkt_tc_kda_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kSl = kDk >= 32 ? 32 : 16, kLdP = kDrChunk + 4;
  __shared__ __align__(16) float ks[kSl * kKdaLd], gs[kSl * kKdaLd];
  __shared__ __align__(16) float Ps[kDrChunk * kLdP];
  __shared__ float rinv[kDrChunk];
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;
  const int h = blockIdx.y, hv = p.num_v_heads, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  const bool vec_g = reinterpret_cast<uintptr_t>(p.gc) % 16 == 0;
  dr_row_rinv(rinv, k, p.k_token_stride, t0, m, kDk, p.use_qk_l2norm);
  int I = 0, J = 0;
  kda_pair_of_warp(wid, I, J);
  float oacc[2][4] = {}, dfast[2][2][4] = {}, dslow[2][8] = {};  // warps 6 and 7 own the diagonal tiles 2 (wid - 6) and 2 (wid - 6) + 1
  for (int d0 = 0; d0 < kDk; d0 += kSl) {
    __syncthreads();  // rinv; the previous slice is consumed
    for (int idx = tid; idx < kDrChunk * (kSl / 8); idx += kDrThreads) {
      const int i = idx / (kSl / 8), c8 = (idx % (kSl / 8)) * 8;
      float kv[8] = {}, gv[8];
      if (i < m) {
        tc_load8<T>(kv, k + int64_t(t0 + i) * p.k_token_stride + d0 + c8, vec_k);
#pragma unroll
        for (int e = 0; e < 8; ++e) kv[e] *= rinv[i];
      }
      // past the sequence the decay of the last real token (zero would give a positive exponent against a real reference)
      tc_load8_f32(gv, p.gc + (int64_t(t0 + min(i, m - 1)) * hv + h) * kDk + d0 + c8, vec_g);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        ks[(c8 + e) * kKdaLd + i] = kv[e];
        gs[(c8 + e) * kKdaLd + i] = gv[e];
      }
    }
    __syncthreads();
    if (wid < 6) kda_offdiag_slice<kSl>(oacc, ks, ks, gs, I, J);
    if (wid >= 6) {
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) kda_diag_tile_slice<false, kSl>(dfast[tt], dslow[tt], ks, ks, gs, 2 * (wid - 6) + tt);
    }
  }
  if (wid < 6) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) Ps[(16 * I + g + 8 * (e >> 1)) * kLdP + 16 * J + 8 * nt + 2 * t + (e & 1)] = oacc[nt][e];
  }
  if (wid >= 6) {
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
      const int J = 2 * (wid - 6) + tt;
      kda_diag_store(Ps + (16 * J) * kLdP + 16 * J, kLdP, dfast[tt], dslow[tt]);
    }
  }
  __syncthreads();
  for (int idx = tid; idx < kDrChunk * (kDrChunk / 4); idx += kDrThreads) {
    const int i = idx / (kDrChunk / 4), c4 = (idx % (kDrChunk / 4)) * 4;
    if (i >= m) continue;
    const float beta = p.beta[int64_t(t0 + i) * hv + h];
    float4 row;
    row.x = c4 + 0 < i ? beta * Ps[i * kLdP + c4 + 0] : 0.0f;
    row.y = c4 + 1 < i ? beta * Ps[i * kLdP + c4 + 1] : 0.0f;
    row.z = c4 + 2 < i ? beta * Ps[i * kLdP + c4 + 2] : 0.0f;
    row.w = c4 + 3 < i ? beta * Ps[i * kLdP + c4 + 3] : 0.0f;
    *reinterpret_cast<float4*>(&p.A[(int64_t(t0 + i) * hv + h) * kDrChunk + c4]) = row;
  }
}

// ---- chunk_output_fwd, tensor cores, KDA (7) -----------------------------------------------------------------

template <int kDk, int kDv>
__host__ __device__ constexpr size_t tc_output_kda_smem_bytes() {
  constexpr int kSl = kDk >= 32 ? 32 : 16;
  // the slice tiles qs, ks, gs and, after the slice loop, the bf16 P tile and the fp32 diagonal tiles (4 x 16 x 17) in their place
  constexpr size_t kTiles = 3 * size_t(kSl) * kKdaLd * sizeof(float), kP = size_t(kDrChunk) * kTcLd64 * sizeof(bf16) + 4 * 16 * 17 * sizeof(float);
  return (kTiles > kP ? kTiles : kP) + 2 * kDrChunk * sizeof(float) +
         (size_t(kDrChunk) * tc_ld(kSl) + size_t(kSl) * tc_ld(kDv) + size_t(kDrChunk) * tc_ld(kDv)) * sizeof(bf16);
}

/**
 * Grid (chunks * ceil(Dv / kDv), Hv, B). o = scale ((q rq exp(G)) h + P v_new) with P_ij = sum_d q_i[d] k_j[d] rq_i rk_j exp(G_i[d] - G_j[d]) for
 * j <= i. The channels are walked in slices of 32: per slice the q exp(G) tile (bf16) times the slice of h accumulates the first product, and
 * the decayed pairs accumulate P (see the KDA block above); P (bf16) then multiplies v_new as in the GDN kernel.
 */
template <typename T, int kDk, int kDv>
__global__ __launch_bounds__(kDrThreads, 2) void chunk_output_fwd_tc_kda_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kSl = kDk >= 32 ? 32 : 16, kNts = kDv / 8, kNt1 = kNts / 2, kLdV = tc_ld(kDv), kLdQ = tc_ld(kSl);
  static_assert(kNt1 >= 2 && kNt1 % 2 == 0, "the output tile is formed two n8 tiles at a time");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  float* qs = reinterpret_cast<float*>(tc_smem);  // [kSl][68] feature-major
  float* ks = qs + kSl * kKdaLd;
  float* gs = ks + kSl * kKdaLd;
  bf16* Pb = reinterpret_cast<bf16*>(tc_smem);    // [64][kTcLd64], in place of qs, ks, gs once the slices are done
  constexpr size_t kTiles = 3 * size_t(kSl) * kKdaLd * sizeof(float), kP = size_t(kDrChunk) * kTcLd64 * sizeof(bf16) + 4 * 16 * 17 * sizeof(float);
  float* rq = reinterpret_cast<float*>(tc_smem + (kTiles > kP ? kTiles : kP));
  float* rk = rq + kDrChunk;
  bf16* Qd = reinterpret_cast<bf16*>(rk + kDrChunk);  // [64][kLdQ]: q rq exp(G)
  bf16* Hs = Qd + kDrChunk * kLdQ;                    // [kSl][kLdV]
  bf16* Vb = Hs + kSl * kLdV;                         // [64][kLdV]
  const int dv = p.head_dim_v, hv = p.num_v_heads;
  const int dvt = (dv + kDv - 1) / kDv, chunk = blockIdx.x / dvt, dv0 = (blockIdx.x % dvt) * kDv;
  int t0, m;
  if (!dr_chunk_of(p, chunk, blockIdx.z, t0, m)) return;
  if (!dr_slot_ok(p, blockIdx.z)) return;
  const int b = blockIdx.z, h = blockIdx.y, hk = h / (hv / p.num_k_heads);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const T* q = static_cast<const T*>(p.q) + int64_t(hk) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const bool vec_f = dv % 4 == 0 && reinterpret_cast<uintptr_t>(p.h) % 16 == 0 && reinterpret_cast<uintptr_t>(p.v_new) % 16 == 0;
  const bool vec_q = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(q) % 16 == 0 && p.q_token_stride % 8 == 0;
  const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  const bool vec_g = reinterpret_cast<uintptr_t>(p.gc) % 16 == 0;
  const float* hbuf = p.h + (int64_t((t0 >> 6) + b) * hv + h) * kDk * dv;
  dr_row_rinv(rq, q, p.q_token_stride, t0, m, kDk, p.use_qk_l2norm);
  __syncthreads();
  dr_row_rinv(rk, k, p.k_token_stride, t0, m, kDk, p.use_qk_l2norm);
  tc_stage_f32<bf16, kDrThreads>(Vb, kLdV, p.v_new + (int64_t(t0) * hv + h) * dv + dv0, int64_t(hv) * dv, m, kDrChunk, kDv, min(kDv, dv - dv0), vec_f && (dv0 % 4 == 0));
  const int I = wid & 3, half = wid >> 2;
  int PI = 0, PJ = 0;
  kda_pair_of_warp(wid, PI, PJ);
  float o[kNt1][4] = {}, oacc[2][4] = {}, dfast[2][2][4] = {}, dslow[2][8] = {};  // warps 6 and 7 own the diagonal tiles 2 (wid - 6) and 2 (wid - 6) + 1
  for (int d0 = 0; d0 < kDk; d0 += kSl) {
    __syncthreads();  // rk and Vb; the previous slice is consumed
    for (int idx = tid; idx < kDrChunk * (kSl / 8); idx += kDrThreads) {
      const int i = idx / (kSl / 8), c8 = (idx % (kSl / 8)) * 8;
      float qv[8] = {}, kv[8] = {}, gv[8];
      if (i < m) {
        tc_load8<T>(qv, q + int64_t(t0 + i) * p.q_token_stride + d0 + c8, vec_q);
        tc_load8<T>(kv, k + int64_t(t0 + i) * p.k_token_stride + d0 + c8, vec_k);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          qv[e] *= rq[i];
          kv[e] *= rk[i];
        }
      }
      // past the sequence the decay of the last real token
      tc_load8_f32(gv, p.gc + (int64_t(t0 + min(i, m - 1)) * hv + h) * kDk + d0 + c8, vec_g);
      float qd[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        qs[(c8 + e) * kKdaLd + i] = qv[e];
        ks[(c8 + e) * kKdaLd + i] = kv[e];
        gs[(c8 + e) * kKdaLd + i] = gv[e];
        qd[e] = qv[e] * __expf(gv[e]);
      }
      *reinterpret_cast<uint4*>(&Qd[i * kLdQ + c8]) = make_uint4(pack_bf16x2(qd[0], qd[1]), pack_bf16x2(qd[2], qd[3]), pack_bf16x2(qd[4], qd[5]), pack_bf16x2(qd[6], qd[7]));
    }
    tc_stage_f32<bf16, kDrThreads>(Hs, kLdV, hbuf + int64_t(d0) * dv + dv0, dv, kSl, kSl, kDv, min(kDv, dv - dv0), vec_f && (dv0 % 4 == 0));
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kSl / 16; ++kk) {
      uint32_t a[4];
      frag_a(a, Qd, kLdQ, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < kNt1 / 2; ++np) {
        uint32_t bb[4];
        frag_b2_trans(bb, Hs, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
        mma_bf16_16x8x16(o[2 * np], a, bb[0], bb[1]);
        mma_bf16_16x8x16(o[2 * np + 1], a, bb[2], bb[3]);
      }
    }
    if (wid < 6) kda_offdiag_slice<kSl>(oacc, qs, ks, gs, PI, PJ);
    if (wid >= 6) {
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) kda_diag_tile_slice<true, kSl>(dfast[tt], dslow[tt], qs, ks, gs, 2 * (wid - 6) + tt);
    }
  }
  __syncthreads();  // the slice tiles are consumed: Pb takes their place
  // P (bf16): the off-diagonal tiles of warps 0..5 and the diagonal tiles; the part of a diagonal tile above its diagonal is zero
  if (wid < 6) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        *reinterpret_cast<uint32_t*>(&Pb[(16 * PI + g + 8 * rr) * kTcLd64 + 16 * PJ + 8 * nt + 2 * t]) = pack_bf16x2(oacc[nt][2 * rr], oacc[nt][2 * rr + 1]);
  }
  if (wid >= 6) {  // the diagonal tiles: fp32 in a scratch behind Pb (inside the slice tiles' memory), then the causal part as bf16 into Pb
    float* td = reinterpret_cast<float*>(tc_smem + size_t(kDrChunk) * kTcLd64 * sizeof(bf16)) + (wid - 6) * 2 * 16 * 17;
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
      const int J = 2 * (wid - 6) + tt;
      kda_diag_store(td + tt * 16 * 17, 17, dfast[tt], dslow[tt]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = lane >> 1, c = 8 * (lane & 1) + e;
        Pb[(16 * J + r) * kTcLd64 + 16 * J + c] = __float2bfloat16_rn(c <= r ? td[tt * 16 * 17 + r * 17 + c] : 0.0f);
      }
    }
  }
  __syncthreads();  // Pb is complete
  for (int kk = 0; kk <= I; ++kk) {
    uint32_t a[4];
    frag_a(a, Pb, kTcLd64, 16 * I, 16 * kk);
#pragma unroll
    for (int np = 0; np < kNt1 / 2; ++np) {
      uint32_t bb[4];
      frag_b2_trans(bb, Vb, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
      mma_bf16_16x8x16(o[2 * np], a, bb[0], bb[1]);
      mma_bf16_16x8x16(o[2 * np + 1], a, bb[2], bb[3]);
    }
  }
  T* out = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = 16 * I + g + 8 * rr;
    if (i >= m) continue;
#pragma unroll
    for (int nt = 0; nt < kNt1; ++nt) {
      const int c = dv0 + 8 * (half * kNt1 + nt) + 2 * t;
      if (c < dv) out[int64_t(t0 + i) * p.o_token_stride + c] = from_float<T>(o[nt][2 * rr] * p.scale);
      if (c + 1 < dv) out[int64_t(t0 + i) * p.o_token_stride + c + 1] = from_float<T>(o[nt][2 * rr + 1] * p.scale);
    }
  }
}


// ---- fused kkt + solve + w/u, tensor cores -------------------------------------------------------

/**
 * The shared-memory budget of chunk_kkt_solve_wu_tc_kernel: a staging region (the 16-bit k tile of the
 * kkt step, or the KDA fp32 slices, or the W/U tiles ts + bs), the strictly lower A tile, the inverse,
 * the 64-thread substitution scratch and the per-row scalars.
 */
template <typename T, int kDk, bool kPer>
__host__ __device__ constexpr size_t kkt_solve_wu_smem_bytes() {
  constexpr int kSl = kDk >= 32 ? 32 : 16;
  constexpr size_t kStg = 2 * size_t(kDrChunk) * kTcLd64 * sizeof(T);
  const size_t kkt = kPer ? 2 * size_t(kSl) * kDrLd * sizeof(float) : size_t(kDrChunk) * tc_ld(kDk) * sizeof(T);
  return (kStg > kkt ? kStg : kkt) + 2 * size_t(kDrChunk) * kDrLd * sizeof(float) +
         (3 * 16 * 17 + 3 * kDrChunk) * sizeof(float);
}

/**
 * (I + A)^-1 of the unit lower triangular 64 x 64 inverse in shared memory: 16 x 16 block forward
 * substitution, the first 64 threads own a column of a diagonal block and then a 4-column strip of a
 * block row; every thread of the CTA joins the two barriers of a level. L and T are distinct tiles
 * (row stride ld); rows past the chunk are zero in L.
 */
__device__ __forceinline__ void dr_solve_tril_smem(float* L, float* T, int ld, float* S, int m) {
  const int tid = threadIdx.x;
  if (tid < kDrChunk) {
    const int b = tid >> 4, c = tid & 15, base = 16 * b;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = (i == c) ? 1.0f : 0.0f;
      if (i >= c) {
#pragma unroll
        for (int j = 0; j < i; ++j)
          if (j >= c) acc = __fmaf_rn(-L[(base + i) * ld + base + j], x[j], acc);
      }
      x[i] = (i >= c) ? acc : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) T[(base + i) * ld + base + c] = x[i];
  }
  __syncthreads();
  const int r = tid >> 2, cq = tid & 3;
#pragma unroll
  for (int l = 1; l <= 3; ++l) {
    if (tid < kDrChunk) {
      float s_acc[3][4];
#pragma unroll
      for (int bi = 0; bi < 3; ++bi) {
        const int i = l + bi;
#pragma unroll
        for (int q = 0; q < 4; ++q) s_acc[bi][q] = 0.0f;
        if (i <= 3) {
          const int j = i - l;
#pragma unroll
          for (int k = 0; k < 3; ++k) {
            if (j + k <= i - 1) {
              const int kb = j + k;
              for (int kk = 0; kk < 16; ++kk) {
                const float a = L[(16 * i + r) * ld + 16 * kb + kk];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                  s_acc[bi][q] = __fmaf_rn(a, T[(16 * kb + kk) * ld + 16 * j + 4 * cq + q], s_acc[bi][q]);
              }
            }
          }
        }
      }
#pragma unroll
      for (int bi = 0; bi < 3; ++bi)
        if (l + bi <= 3)
#pragma unroll
          for (int q = 0; q < 4; ++q) S[(bi * 16 + r) * 17 + 4 * cq + q] = s_acc[bi][q];
    }
    __syncthreads();
    if (tid < kDrChunk) {
#pragma unroll
      for (int bi = 0; bi < 3; ++bi) {
        const int i = l + bi;
        if (i <= 3) {
          const int j = i - l;
          float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int kk = 0; kk < 16; ++kk) {
            const float d = T[(16 * i + r) * ld + 16 * i + kk];
#pragma unroll
            for (int q = 0; q < 4; ++q) t[q] = __fmaf_rn(d, S[(bi * 16 + kk) * 17 + 4 * cq + q], t[q]);
          }
#pragma unroll
          for (int q = 0; q < 4; ++q) T[(16 * i + r) * ld + 16 * j + 4 * cq + q] = -t[q];
        }
      }
    }
    __syncthreads();
  }
}

/**
 * Grid (chunks, Hv, B), 256 threads. FLA's fwd_kkt_solve: the strictly lower A_jm of a chunk is built on
 * mma.sync (GDN: k k^T scaled by exp(G_j - G_m); KDA: the decayed pairs channel by channel), inverted in
 * shared memory by dr_solve_tril_smem, and the inverse immediately multiplies the beta-scaled k rows
 * (W) and v rows (U) of the same chunk, so A and its inverse never leave the CTA. The inverse is also
 * written to p.A (the workspace contract) and W/U to p.w / p.u, which the state kernel reads.
 */
template <typename T, int kDk, bool kPer>
__global__ __launch_bounds__(kDrThreads, 2) void chunk_kkt_solve_wu_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kLdK = tc_ld(kDk);
  constexpr int kSl = kDk >= 32 ? 32 : 16;
  constexpr size_t kStg = 2 * size_t(kDrChunk) * kTcLd64 * sizeof(T);
  constexpr size_t kKkt = kPer ? 2 * size_t(kSl) * kDrLd * sizeof(float) : size_t(kDrChunk) * kLdK * sizeof(T);
  constexpr size_t kStgBytes = kStg > kKkt ? kStg : kKkt;
  extern __shared__ __align__(16) unsigned char fus_smem[];
  bf16* stg = reinterpret_cast<bf16*>(fus_smem);
  float* As = reinterpret_cast<float*>(fus_smem + kStgBytes);  // [64][kDrLd]
  float* Tinv = As + kDrChunk * kDrLd;                         // [64][kDrLd]
  float* scr = Tinv + kDrChunk * kDrLd;                        // [3][16][17]
  float* rinv = scr + 3 * 16 * 17;                             // [64]
  float* gt = rinv + kDrChunk;                                 // [64]
  float* sm = gt + kDrChunk;                                   // [64]
  int t0, m;
  if (!dr_chunk_of(p, blockIdx.x, blockIdx.z, t0, m)) return;  // uniform per CTA
  const int h = blockIdx.y, hv = p.num_v_heads, hk = h / (hv / p.num_k_heads), dk = p.head_dim_k;
  const int dv = p.head_dim_v;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const T* kp = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const T* vp = static_cast<const T*>(p.v) + int64_t(h) * p.v_head_stride;
  const int I = wid & 3, half = wid >> 2;
  dr_row_rinv(rinv, kp, p.k_token_stride, t0, m, dk, p.use_qk_l2norm);
  if constexpr (!kPer) {
    if (tid < kDrChunk) gt[tid] = tid < m ? p.gc[int64_t(t0 + tid) * hv + h] : 0.0f;
  }
  if constexpr (!kPer) {
    // ---- A = beta_j rinv_j rinv_m (k_j . k_m) exp(G_j - G_m), m < j ----
    const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(kp) % 16 == 0 && p.k_token_stride % 8 == 0;
    tc_stage_input<T, kDrThreads>(stg, kLdK, kp, p.k_token_stride, t0, m, kDk, vec_k);
    __syncthreads();
    float acc[4][4] = {};
#pragma unroll
    for (int kk = 0; kk < kDk / 16; ++kk) {
      uint32_t a[4];
      frag_a(a, stg, kLdK, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        if (2 * half + np > I) continue;
        uint32_t b[4];
        frag_b2(b, stg, kLdK, 32 * half + 16 * np, 16 * kk);
        mma_bf16_16x8x16(acc[2 * np], a, b[0], b[1]);
        mma_bf16_16x8x16(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = 16 * I + g + 8 * rr;
      if (i >= m) continue;
      const float scale_i = p.beta[int64_t(t0 + i) * hv + h] * rinv[i], gi = gt[i];
      float* row = As + i * kDrLd;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int np = nt >> 1, j0 = 32 * half + 16 * np + 8 * (nt & 1) + 2 * t;
        float v0 = 0.0f, v1 = 0.0f;
        if (2 * half + np <= I) {
          v0 = j0 < i ? acc[nt][2 * rr] * scale_i * rinv[j0] * __expf(gi - gt[j0]) : 0.0f;
          v1 = j0 + 1 < i ? acc[nt][2 * rr + 1] * scale_i * rinv[j0 + 1] * __expf(gi - gt[j0 + 1]) : 0.0f;
        }
        row[j0] = v0;
        row[j0 + 1] = v1;
      }
    }
  } else {
    // ---- A_jm = beta_j sum_d k_j[d] k_m[d] exp(G_j[d] - G_m[d]), m < j (KDA) ----
    float* ks = reinterpret_cast<float*>(stg);  // [kSl][kDrLd], feature-major
    float* gs = ks + kSl * kDrLd;
    const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(kp) % 16 == 0 && p.k_token_stride % 8 == 0;
    const bool vec_g = reinterpret_cast<uintptr_t>(p.gc) % 16 == 0;
    int PI = 0, PJ = 0;
    kda_pair_of_warp(wid, PI, PJ);
    float oacc[2][4] = {}, dfast[2][2][4] = {}, dslow[2][8] = {};
    for (int d0 = 0; d0 < kDk; d0 += kSl) {
      __syncthreads();  // the previous slice is consumed
      for (int idx = tid; idx < kDrChunk * (kSl / 8); idx += kDrThreads) {
        const int i = idx / (kSl / 8), c8 = (idx % (kSl / 8)) * 8;
        float kv[8] = {}, gv[8];
        if (i < m) {
          tc_load8<T>(kv, kp + int64_t(t0 + i) * p.k_token_stride + d0 + c8, vec_k);
#pragma unroll
          for (int e = 0; e < 8; ++e) kv[e] *= rinv[i];
        }
        tc_load8_f32(gv, p.gc + (int64_t(t0 + min(i, m - 1)) * hv + h) * dk + d0 + c8, vec_g);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          ks[(c8 + e) * kDrLd + i] = kv[e];
          gs[(c8 + e) * kDrLd + i] = gv[e];
        }
      }
      __syncthreads();
      if (wid < 6) kda_offdiag_slice<kSl>(oacc, ks, ks, gs, PI, PJ);
      if (wid >= 6) {
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) kda_diag_tile_slice<false, kSl>(dfast[tt], dslow[tt], ks, ks, gs, 2 * (wid - 6) + tt);
      }
    }
    if (wid < 6) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          As[(16 * PI + g + 8 * (e >> 1)) * kDrLd + 16 * PJ + 8 * nt + 2 * t + (e & 1)] = oacc[nt][e];
    }
    if (wid >= 6) {
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int J = 2 * (wid - 6) + tt;
        kda_diag_store(As + (16 * J) * kDrLd + 16 * J, kDrLd, dfast[tt], dslow[tt]);
      }
    }
    __syncthreads();
    // beta scaling and the causal mask, in place
    for (int idx = tid; idx < kDrChunk * (kDrChunk / 4); idx += kDrThreads) {
      const int i = idx / (kDrChunk / 4), c4 = (idx % (kDrChunk / 4)) * 4;
      if (i >= m) continue;
      const float beta = p.beta[int64_t(t0 + i) * hv + h];
      float* q4 = As + i * kDrLd + c4;
      q4[0] = c4 + 0 < i ? beta * q4[0] : 0.0f;
      q4[1] = c4 + 1 < i ? beta * q4[1] : 0.0f;
      q4[2] = c4 + 2 < i ? beta * q4[2] : 0.0f;
      q4[3] = c4 + 3 < i ? beta * q4[3] : 0.0f;
    }
  }
  for (int idx = tid; idx < kDrChunk * kDrLd; idx += kDrThreads) {
    if (idx / kDrLd >= m) As[idx] = 0.0f;
  }
  __syncthreads();
  // ---- the inverse ----
  dr_solve_tril_smem(As, Tinv, kDrLd, scr, m);
  if (tid < kDrChunk) {
    for (int i = 0; i < kDrChunk; ++i)
      if (i < m) p.A[(int64_t(t0 + i) * hv + h) * kDrChunk + tid] = tid <= i ? Tinv[i * kDrLd + tid] : 0.0f;
  }
  __syncthreads();
  // ---- W = T (beta exp(G) rinv k), U = T (beta v); a column tile of 64 at a time ----
  bf16* ts = stg;                              // [64][kTcLd64]
  bf16* bs = stg + kDrChunk * kTcLd64;         // [64][kTcLd64]
  const int dkt = (dk + kDrChunk - 1) / kDrChunk, dvt = (dv + kDrChunk - 1) / kDrChunk, coltiles = dkt + dvt;
  for (int ct = 0; ct < coltiles; ++ct) {
    const bool is_w = ct < dkt;
    const int c0 = (is_w ? ct : ct - dkt) * kDrChunk;
    const int width = is_w ? dk : dv;
    if (tid < kDrChunk) {
      float s = 0.0f;
      if (tid < m) {
        const int64_t tok = int64_t(t0 + tid);
        s = p.beta[tok * hv + h];
        if (is_w) s = kPer ? 1.0f : s * __expf(p.gc[tok * hv + h]) * rinv[tid];
      }
      sm[tid] = s;
    }
    const T* src = is_w ? kp : vp;
    const int64_t stride = is_w ? p.k_token_stride : p.v_token_stride;
    for (int idx = tid; idx < kDrChunk * (kDrChunk / 2); idx += kDrThreads) {
      const int mm = idx / (kDrChunk / 2), cc = (idx % (kDrChunk / 2)) * 2;
      float x0 = 0.0f, x1 = 0.0f;
      if (mm < m) {
        const T* row = src + int64_t(t0 + mm) * stride + c0 + cc;
        if (c0 + cc < width) x0 = to_float(row[0]);
        if (c0 + cc + 1 < width) x1 = to_float(row[1]);
        if constexpr (kPer) {
          if (is_w) {
            const int64_t tk = int64_t(t0 + mm);
            const float f = p.beta[tk * hv + h] * rinv[mm];
            const float* gcp = p.gc + (tk * hv + h) * dk + c0 + cc;
            x0 = c0 + cc < width ? x0 * f * __expf(gcp[0]) : 0.0f;
            x1 = c0 + cc + 1 < width ? x1 * f * __expf(gcp[1]) : 0.0f;
          }
        }
      }
      *reinterpret_cast<uint32_t*>(&bs[mm * kTcLd64 + cc]) = pack_bf16x2(x0, x1);
    }
    __syncthreads();  // sm and bs are visible
    for (int idx = tid; idx < kDrChunk * (kDrChunk / 4); idx += kDrThreads) {
      const int j = idx / (kDrChunk / 4), mm = (idx % (kDrChunk / 4)) * 4;
      float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
      if (j < m && mm <= j) {
        x0 = mm <= j ? Tinv[j * kDrLd + mm] * sm[mm] : 0.0f;
        x1 = mm + 1 <= j ? Tinv[j * kDrLd + mm + 1] * sm[mm + 1] : 0.0f;
        x2 = mm + 2 <= j ? Tinv[j * kDrLd + mm + 2] * sm[mm + 2] : 0.0f;
        x3 = mm + 3 <= j ? Tinv[j * kDrLd + mm + 3] * sm[mm + 3] : 0.0f;
      }
      *reinterpret_cast<uint2*>(&ts[j * kTcLd64 + mm]) =
          make_uint2(pack_bf16x2(x0, x1), pack_bf16x2(x2, x3));
    }
    __syncthreads();
    float acc[4][4] = {};
    for (int kk = 0; kk <= I; ++kk) {
      uint32_t a[4];
      frag_a(a, ts, kTcLd64, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        frag_b2_trans(b, bs, kTcLd64, 16 * kk, 32 * half + 16 * np);
        mma_bf16_16x8x16(acc[2 * np], a, b[0], b[1]);
        mma_bf16_16x8x16(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
    float* out = is_w ? p.w : p.u;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int j = 16 * I + g + 8 * rr;
      if (j >= m) continue;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = c0 + 32 * half + 8 * nt + 2 * t;
        float* o = out + (int64_t(t0 + j) * hv + h) * width + col;
        if (col < width) o[0] = acc[nt][2 * rr];
        if (col + 1 < width) o[1] = acc[nt][2 * rr + 1];
      }
    }
  }
}

// ---- fused chunk state + output, tensor cores ----------------------------------------------------

/** Dynamic shared memory of chunk_state_output_tc_kernel: Sb [Dk][kLdS], Vb [64][kLdS], Wb [64][kLdK]
 * (also the raw v_new tile of the GDN output), Kt [2][64][kLdK], Pb [64][kTcLd64], the GDN q tile
 * Qs [64][kLdK] (the KDA slice tiles qs/ks/gs [kSl][kDrLd], their Qd [64][kLdQ] and the diagonal
 * scratch take its place), and the fp32 rinv [2][64], gt [2][64], gl [2][Dk], rq [64]. */
template <typename T, bool kPer, int kDk, int kDv>
__host__ __device__ constexpr size_t tc_state_output_smem_bytes() {
  constexpr int kSl = kDk >= 32 ? 32 : 16;
  constexpr size_t k16 = (size_t(kDk) * tc_ld(kDv) + 2 * size_t(kDrChunk) * tc_ld(kDv) + 3 * size_t(kDrChunk) * tc_ld(kDk) +
                          size_t(kDrChunk) * kTcLd64) *
                         sizeof(__nv_bfloat16);
  constexpr size_t kOut = kPer ? (3 * size_t(kSl) * kDrLd * sizeof(float) + size_t(kDrChunk) * tc_ld(kSl) * sizeof(__nv_bfloat16))
                               : size_t(kDrChunk) * tc_ld(kDk) * sizeof(__nv_bfloat16);
  return k16 + kOut + (5 * kDrChunk + 2 * kDk) * sizeof(float);
}

/**
 * Grid (ceil(Dv / kDv), Hv, B): one persistent CTA per (sequence, value head, V tile) walks the chunks
 * in order with the state S[:, dv tile] in the fp32 accumulator registers of its owner warps. The
 * previous pipeline wrote the state entering every chunk to the h workspace and read it back in a second
 * chunk-parallel kernel; here each chunk's output is formed immediately after its v_new, from the same
 * registers that the recurrence carries, so h never goes through memory.
 *
 * Per chunk, after v_new = U - W S (fp32 to the v_new workspace, 16-bit to the update operand):
 *   GDN: o = scale (rq exp(G) (q S) + P v_new), P_ij = rq_i rk_j (q_i . k_j) exp(G_i - G_j), j <= i
 *   KDA: o = scale ((q rq exp(G)) S + P v_new), P_ij = sum_d q_i[d] k_j[d] rq_i rk_j exp(G_i[d] - G_j[d])
 * The q k^T and P v_new products are mma.sync on 16-bit tiles; the KDA pairs keep the per-channel decay
 * with the per-16-token reference of the chunked output kernel. The state update is unchanged.
 */
template <typename T, bool kPer, int kDk, int kDv>
__global__ __launch_bounds__(kDrThreads, 1) void chunk_state_output_tc_kernel(const __grid_constant__ DeltaRuleParams p) {
  constexpr int kNmt = kDk / 16, kNts = kDv / 8;
  constexpr int kWpm = (8 / kNmt) < kNts ? (8 / kNmt) : kNts;
  constexpr int kNtw = kNts / kWpm;
  constexpr int kNtv = kNts / 2;
  constexpr int kLdS = tc_ld(kDv), kLdK = tc_ld(kDk);
  constexpr int kPieces = kDk / 8;
  constexpr int kPerThread = (kDrChunk * kPieces + kDrThreads - 1) / kDrThreads;
  constexpr int kSl = kDk >= 32 ? 32 : 16, kLdQ = tc_ld(kSl);
  static_assert(kNtv >= 2 && kNtv % 2 == 0, "v_new is formed two n8 tiles at a time");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  bf16* Sb = reinterpret_cast<bf16*>(tc_smem);   // [kDk][kLdS]: the state, k-major for the B fragments
  bf16* Vb = Sb + kDk * kLdS;                    // [64][kLdS]: v_new scaled for the state update
  bf16* Wb = Vb + kDrChunk * kLdS;               // [64][kLdK]: the W tile, then the raw v_new tile
  bf16* Kt = Wb + kDrChunk * kLdK;               // [2][64][kLdK]: k scaled for the state update
  bf16* Pb = Kt + 2 * kDrChunk * kLdK;           // [64][kTcLd64]: the causal attention tile
  bf16* Vn = Pb + kDrChunk * kTcLd64;            // [64][kLdS]: raw v_new for the P v_new product
  bf16* Qs = Vn + kDrChunk * kLdS;               // [64][kLdK]: GDN q tile
  unsigned char* tail = reinterpret_cast<unsigned char*>(Qs) + (kPer ? 0 : size_t(kDrChunk) * kLdK * sizeof(bf16));
  float* qs_s = reinterpret_cast<float*>(tail);  // KDA: [kSl][kDrLd] q slice, feature-major
  float* ks_s = qs_s + kSl * kDrLd;
  float* gs_s = ks_s + kSl * kDrLd;
  bf16* Qd = reinterpret_cast<bf16*>(gs_s + kSl * kDrLd);  // KDA: [64][kLdQ] q rq exp(G)
  float* rinv = kPer ? reinterpret_cast<float*>(Qd + kDrChunk * kLdQ) : reinterpret_cast<float*>(tail);  // [2][64]
  float* gt = rinv + 2 * kDrChunk;               // [2][64]
  float* gl = gt + 2 * kDrChunk;                 // [2][kDk]
  float* rq = gl + 2 * kDk;                      // [64]
  const int dk = kDk, dv = p.head_dim_v, hv = p.num_v_heads;
  const int b = blockIdx.z, h = blockIdx.y, dv0 = blockIdx.x * kDv;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int start = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - start;
  const int hk = h / (hv / p.num_k_heads);
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  const T* k = static_cast<const T*>(p.k) + int64_t(hk) * p.k_head_stride;
  const T* q = static_cast<const T*>(p.q) + int64_t(hk) * p.q_head_stride;
  T* out = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
  float* pool = p.state_pool + slot * p.state_slot_stride + int64_t(h) * dk * dv;
  const auto pool_at = [&](int d, int c) -> int64_t { return p.state_transposed ? int64_t(c) * dk + d : int64_t(d) * dv + c; };
  const int nchunks = (n + kDrChunk - 1) / kDrChunk;
  const bool vec_q = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(q) % 16 == 0 && p.q_token_stride % 8 == 0;
  const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  const bool vec_g = reinterpret_cast<uintptr_t>(p.gc) % 16 == 0;
  // 16-byte loads of W, the decays and k (aligned bases, rows a whole number of 16-byte pieces
  // apart) or the scalar fallback
  const bool vec = reinterpret_cast<uintptr_t>(p.w) % 16 == 0 && reinterpret_cast<uintptr_t>(p.gc) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(k) % 16 == 0 && (p.k_token_stride * int64_t(sizeof(T))) % 16 == 0 &&
                   (p.k_head_stride * int64_t(sizeof(T))) % 16 == 0;
  // the owner of state tile (row tile mt, columns [8 ntb, 8 (ntb + kNtw))) of this warp
  const bool owner = wid < kNmt * kWpm;
  const int mt = wid % kNmt, ntb = (wid / kNmt) * kNtw;
  const int OI = wid & 3, half = wid >> 2;
  float S[(kNtw + 1) / 2 * 2][4] = {};
#pragma unroll
  for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
      S[nt][e] = (owner && has_init && dv0 + c < dv) ? pool[pool_at(d, dv0 + c)] : 0.0f;
    }
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2)
        *reinterpret_cast<uint32_t*>(&Sb[(16 * mt + g + 8 * e2) * kLdS + 8 * (ntb + nt) + 2 * t]) =
            pack_bf16x2(S[nt][2 * e2], S[nt][2 * e2 + 1]);
  }

  // ---- the prefetch of one chunk: the pieces of W (and, for GDN, k) this thread stages, the cumulative decay
  constexpr bool kRawK = std::is_same_v<T, bf16>;
  struct Next {
    float w[kPerThread][8];
    float k[(!kPer && !kRawK) ? kPerThread : 1][8];
    uint4 kraw[(!kPer && kRawK) ? kPerThread : 1];
    float gl8[kPer ? 8 : 1];
    float gt;
    int t0, m;
  } nx;
  const auto fetch = [&](int chunk) {
    nx.t0 = start + chunk * kDrChunk;
    nx.m = min(kDrChunk, n - chunk * kDrChunk);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
      (void)r, (void)c8;
      const bool in = piece < kDrChunk * kPieces && r < nx.m;
      if (in) {
        tc_load8_f32(nx.w[i], p.w + (int64_t(nx.t0 + r) * hv + h) * dk + c8, vec);
        if constexpr (!kPer) {
          if constexpr (kRawK) {
            if (vec) {
              nx.kraw[i] = *reinterpret_cast<const uint4*>(k + int64_t(nx.t0 + r) * p.k_token_stride + c8);
            } else {
              float tmp[8];
              tc_load8<T>(tmp, k + int64_t(nx.t0 + r) * p.k_token_stride + c8, false);
              nx.kraw[i] = make_uint4(pack_bf16x2(tmp[0], tmp[1]), pack_bf16x2(tmp[2], tmp[3]),
                                      pack_bf16x2(tmp[4], tmp[5]), pack_bf16x2(tmp[6], tmp[7]));
            }
          } else {
            tc_load8<T>(nx.k[i], k + int64_t(nx.t0 + r) * p.k_token_stride + c8, vec);
          }
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          nx.w[i][e] = 0.0f;
          if constexpr (!kPer && !kRawK) nx.k[i][e] = 0.0f;
        }
        if constexpr (!kPer && kRawK) nx.kraw[i] = make_uint4(0, 0, 0, 0);
      }
    }
    if constexpr (kPer) {
      tc_load8_f32(nx.gl8, p.gc + (int64_t(nx.t0 + nx.m - 1) * hv + h) * dk + ((tid % kPieces) * 8), vec);
    } else {
      nx.gt = tid < kDrChunk && tid < nx.m ? p.gc[int64_t(nx.t0 + tid) * hv + h] : 0.0f;
    }
  };
  const auto stage = [&](int bf) {
    bf16* wb = Wb;
    bf16* kt = Kt + bf * kDrChunk * kLdK;
    constexpr int kBatch = kPer ? (kPerThread < 2 ? kPerThread : 2) : kPerThread;
#pragma unroll
    for (int i0 = 0; i0 < kPerThread; i0 += kBatch) {
      float kv[kBatch][8];
      float gcv[kPer ? kBatch : 1][8];
      (void)gcv;
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int i = i0 + j, piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
        (void)r, (void)c8;
        if constexpr (kPer) {
          const bool in = piece < kDrChunk * kPieces && r < nx.m;
          if (in) {
            tc_load8<T>(kv[j], k + int64_t(nx.t0 + r) * p.k_token_stride + c8, vec);
            tc_load8_f32(gcv[j], p.gc + (int64_t(nx.t0 + r) * hv + h) * dk + c8, vec);
          } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) kv[j][e] = 0.0f, gcv[j][e] = 0.0f;
          }
        } else if constexpr (kRawK) {
          const uint32_t wd[4] = {nx.kraw[i].x, nx.kraw[i].y, nx.kraw[i].z, nx.kraw[i].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            kv[j][2 * e] = __uint_as_float(wd[e] << 16);
            kv[j][2 * e + 1] = __uint_as_float(wd[e] & 0xffff0000u);
          }
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) kv[j][e] = nx.k[i][e];
        }
      }
#pragma unroll
      for (int j = 0; j < kBatch; ++j) {
        const int i = i0 + j, piece = tid + kDrThreads * i, r = piece / kPieces, c8 = (piece % kPieces) * 8;
        (void)r, (void)c8;
        float sq = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) sq = __fmaf_rn(kv[j][e], kv[j][e], sq);
#pragma unroll
        for (int off = kPieces / 2; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
        if (piece >= kDrChunk * kPieces) continue;
        *reinterpret_cast<uint4*>(wb + r * kLdK + c8) = make_uint4(pack_bf16x2(nx.w[i][0], nx.w[i][1]), pack_bf16x2(nx.w[i][2], nx.w[i][3]),
                                                                     pack_bf16x2(nx.w[i][4], nx.w[i][5]), pack_bf16x2(nx.w[i][6], nx.w[i][7]));
        const float ri = (r < nx.m && p.use_qk_l2norm) ? rsqrtf(sq + kL2Eps) : 1.0f;
        if ((tid % kPieces) == 0) rinv[bf * kDrChunk + r] = ri;
        float x[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if constexpr (kPer) x[e] = kv[j][e] * ri * __expf(nx.gl8[e] - gcv[j][e]);
          else x[e] = kv[j][e];
        }
        *reinterpret_cast<uint4*>(kt + r * kLdK + c8) = make_uint4(pack_bf16x2(x[0], x[1]), pack_bf16x2(x[2], x[3]), pack_bf16x2(x[4], x[5]), pack_bf16x2(x[6], x[7]));
      }
    }
    if constexpr (kPer) {
      if (tid < kPieces) {
#pragma unroll
        for (int e = 0; e < 8; ++e) gl[bf * kDk + tid * 8 + e] = nx.gl8[e];
      }
    } else {
      if (tid < kDrChunk) gt[bf * kDrChunk + tid] = nx.gt;
    }
  };

  if (nchunks > 0) {
    fetch(0);
    stage(0);
  }
  __syncthreads();
  const auto iteration = [&](int chunk, auto prefetch) {
    constexpr bool kPrefetch = decltype(prefetch)::value;
    const int bf = chunk & 1;
    const int t0 = start + chunk * kDrChunk, m = min(kDrChunk, n - chunk * kDrChunk);
    // the loads of the next chunk and of this chunk's u go out before the products
    float ur[kNtv][4] = {};
    {
      const int I = OI;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int j = 16 * I + g + 8 * rr;
#pragma unroll
        for (int nt = 0; nt < kNtv; ++nt) {
          const int c = 8 * (half * kNtv + nt) + 2 * t;
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (j < m && dv0 + c + e < dv) ur[nt][2 * rr + e] = p.u[(int64_t(t0 + j) * hv + h) * dv + dv0 + c + e];
        }
      }
    }
    dr_row_rinv(rq, q, p.q_token_stride, t0, m, dk, p.use_qk_l2norm);
    if constexpr (!kPer) {
      tc_stage_input<T, kDrThreads>(Qs, kLdK, q, p.q_token_stride, t0, m, dk, vec_q);
    }
    // v_new = U - W S: rows OI of the chunk, columns of half
    float vraw[kNtv][4] = {};
    {
      const int I = OI;
      float acc[kNtv][4] = {};
      const bf16* wb = Wb;
#pragma unroll 2
      for (int kk = 0; kk < kDk / 16; ++kk) {
        uint32_t a[4];
        frag_a(a, wb, kLdK, 16 * I, 16 * kk);
#pragma unroll
        for (int np = 0; np < kNtv / 2; ++np) {
          uint32_t bb[4];
          frag_b2_trans(bb, Sb, kLdS, 16 * kk, 8 * (half * kNtv) + 16 * np);
          mma_bf16_16x8x16(acc[2 * np], a, bb[0], bb[1]);
          mma_bf16_16x8x16(acc[2 * np + 1], a, bb[2], bb[3]);
        }
      }
      const float g_last = kPer ? 0.0f : gt[bf * kDrChunk + m - 1];
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int j = 16 * I + g + 8 * rr;
        const float fold = kPer ? 1.0f : (j < m ? rinv[bf * kDrChunk + j] * __expf(g_last - gt[bf * kDrChunk + j]) : 0.0f);
#pragma unroll
        for (int nt = 0; nt < kNtv; ++nt) {
          const int c = 8 * (half * kNtv + nt) + 2 * t;
          float val[2] = {0.0f, 0.0f};
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            if (j < m && dv0 + c + e < dv) {
              val[e] = ur[nt][2 * rr + e] - acc[nt][2 * rr + e];
              p.v_new[(int64_t(t0 + j) * hv + h) * dv + dv0 + c + e] = val[e];
            }
            vraw[nt][2 * rr + e] = val[e];
          }
          *reinterpret_cast<uint32_t*>(&Vb[j * kLdS + c]) = pack_bf16x2(val[0] * fold, val[1] * fold);
          *reinterpret_cast<uint32_t*>(&Vn[j * kLdS + c]) = pack_bf16x2(val[0], val[1]);
        }
      }
    }
    __syncthreads();  // Vb, Vn are complete; every warp is done with Wb
    if constexpr (kPrefetch) fetch(chunk + 1);
    // ---- the output of this chunk ----
    float o[kNtv][4] = {};
    if constexpr (!kPer) {
      const bf16* kraw = Kt + bf * kDrChunk * kLdK;  // GDN: the k tile is the raw k of the chunk
      float sacc[4][4] = {};
      for (int kk = 0; kk < kDk / 16; ++kk) {
        uint32_t a[4];
        frag_a(a, Qs, kLdK, 16 * OI, 16 * kk);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          if (2 * half + np > OI) continue;
          uint32_t b[4];
          frag_b2(b, kraw, kLdK, 32 * half + 16 * np, 16 * kk);
          mma_bf16_16x8x16(sacc[2 * np], a, b[0], b[1]);
          mma_bf16_16x8x16(sacc[2 * np + 1], a, b[2], b[3]);
        }
#pragma unroll
        for (int np = 0; np < kNtv / 2; ++np) {
          uint32_t b[4];
          frag_b2_trans(b, Sb, kLdS, 16 * kk, 8 * (half * kNtv) + 16 * np);
          mma_bf16_16x8x16(o[2 * np], a, b[0], b[1]);
          mma_bf16_16x8x16(o[2 * np + 1], a, b[2], b[3]);
        }
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int i = 16 * OI + g + 8 * rr;
        const float ri = rq[i], gi = gt[bf * kDrChunk + i], f = ri * __expf(gi);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int np = nt >> 1, j0 = 32 * half + 16 * np + 8 * (nt & 1) + 2 * t;
          if (2 * half + np > OI) continue;
          const float v0 = j0 <= i ? sacc[nt][2 * rr] * ri * rinv[bf * kDrChunk + j0] * __expf(gi - gt[bf * kDrChunk + j0]) : 0.0f;
          const float v1 = j0 + 1 <= i ? sacc[nt][2 * rr + 1] * ri * rinv[bf * kDrChunk + j0 + 1] * __expf(gi - gt[bf * kDrChunk + j0 + 1]) : 0.0f;
          *reinterpret_cast<uint32_t*>(&Pb[i * kTcLd64 + j0]) = pack_bf16x2(v0, v1);
        }
#pragma unroll
        for (int nt = 0; nt < kNtv; ++nt) {
          o[nt][2 * rr] *= f;
          o[nt][2 * rr + 1] *= f;
        }
      }
    } else {
      int PI = 0, PJ = 0;
      kda_pair_of_warp(wid, PI, PJ);
      float oacc[2][4] = {}, dfast[2][2][4] = {}, dslow[2][8] = {};
      for (int d0 = 0; d0 < kDk; d0 += kSl) {
        __syncthreads();  // the previous slice is consumed
        for (int idx = tid; idx < kDrChunk * (kSl / 8); idx += kDrThreads) {
          const int i = idx / (kSl / 8), c8 = (idx % (kSl / 8)) * 8;
          float qv[8] = {}, kv[8] = {}, gv[8];
          if (i < m) {
            tc_load8<T>(qv, q + int64_t(t0 + i) * p.q_token_stride + d0 + c8, vec_q);
            tc_load8<T>(kv, k + int64_t(t0 + i) * p.k_token_stride + d0 + c8, vec_k);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              qv[e] *= rq[i];
              kv[e] *= rinv[bf * kDrChunk + i];
            }
          }
          tc_load8_f32(gv, p.gc + (int64_t(t0 + min(i, m - 1)) * hv + h) * dk + d0 + c8, vec_g);
          float qd[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            qs_s[(c8 + e) * kDrLd + i] = qv[e];
            ks_s[(c8 + e) * kDrLd + i] = kv[e];
            gs_s[(c8 + e) * kDrLd + i] = gv[e];
            qd[e] = qv[e] * __expf(gv[e]);
          }
          *reinterpret_cast<uint4*>(&Qd[i * kLdQ + c8]) = make_uint4(pack_bf16x2(qd[0], qd[1]), pack_bf16x2(qd[2], qd[3]),
                                                                    pack_bf16x2(qd[4], qd[5]), pack_bf16x2(qd[6], qd[7]));
        }
        __syncthreads();
        for (int kk = 0; kk < kSl / 16; ++kk) {
          uint32_t a[4];
          frag_a(a, Qd, kLdQ, 16 * OI, 16 * kk);
#pragma unroll
          for (int np = 0; np < kNtv / 2; ++np) {
            uint32_t b[4];
            frag_b2_trans(b, Sb + d0 * kLdS, kLdS, 16 * kk, 8 * (half * kNtv) + 16 * np);
            mma_bf16_16x8x16(o[2 * np], a, b[0], b[1]);
            mma_bf16_16x8x16(o[2 * np + 1], a, b[2], b[3]);
          }
        }
        if (wid < 6) kda_offdiag_slice<kSl>(oacc, qs_s, ks_s, gs_s, PI, PJ);
        if (wid >= 6) {
#pragma unroll
          for (int tt = 0; tt < 2; ++tt)
            kda_diag_tile_slice<true, kSl>(dfast[tt], dslow[tt], qs_s, ks_s, gs_s, 2 * (wid - 6) + tt);
        }
      }
      __syncthreads();  // the slice tiles are consumed: the diagonal scratch and Pb take their place
      float* td = reinterpret_cast<float*>(tail) + (wid >= 6 ? (wid - 6) * 2 * 16 * 17 : 0);
      if (wid < 6) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int rr = 0; rr < 2; ++rr)
            *reinterpret_cast<uint32_t*>(&Pb[(16 * PI + g + 8 * rr) * kTcLd64 + 16 * PJ + 8 * nt + 2 * t]) =
                pack_bf16x2(oacc[nt][2 * rr], oacc[nt][2 * rr + 1]);
      }
      if (wid >= 6) {
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int J = 2 * (wid - 6) + tt;
          kda_diag_store(td + tt * 16 * 17, 17, dfast[tt], dslow[tt]);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = lane >> 1, c = 8 * (lane & 1) + e;
            Pb[(16 * J + r) * kTcLd64 + 16 * J + c] = __float2bfloat16_rn(c <= r ? td[tt * 16 * 17 + r * 17 + c] : 0.0f);
          }
        }
      }
    }
    __syncthreads();  // Pb (and the raw v_new tile) are complete
    for (int kk = 0; kk <= OI; ++kk) {
      uint32_t a[4];
      frag_a(a, Pb, kTcLd64, 16 * OI, 16 * kk);
#pragma unroll
      for (int np = 0; np < kNtv / 2; ++np) {
        uint32_t b[4];
        frag_b2_trans(b, Vn, kLdS, 16 * kk, 8 * (half * kNtv) + 16 * np);
        mma_bf16_16x8x16(o[2 * np], a, b[0], b[1]);
        mma_bf16_16x8x16(o[2 * np + 1], a, b[2], b[3]);
      }
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = 16 * OI + g + 8 * rr;
      if (i >= m) continue;
#pragma unroll
      for (int nt = 0; nt < kNtv; ++nt) {
        const int c = dv0 + 8 * (half * kNtv + nt) + 2 * t;
        if (c < dv) out[int64_t(t0 + i) * p.o_token_stride + c] = from_float<T>(o[nt][2 * rr] * p.scale);
        if (c + 1 < dv) out[int64_t(t0 + i) * p.o_token_stride + c + 1] = from_float<T>(o[nt][2 * rr + 1] * p.scale);
      }
    }
    // ---- the state after the chunk: S <- exp(G_last) S + k^T (scaled v_new) ----
    const int done = t0 + m - start;
    int64_t snapshot_slot = -1;
    if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
      const int j = done / p.snapshot_interval - 1;
      if (j < p.max_snapshots) snapshot_slot = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
    }
    const bool snapshot = snapshot_slot >= 0 && snapshot_slot < p.num_slots;
    if (owner) {
      const float decay_rows[2] = {kPer ? __expf(gl[bf * kDk + 16 * mt + g]) : __expf(gt[bf * kDrChunk + m - 1]),
                                   kPer ? __expf(gl[bf * kDk + 16 * mt + g + 8]) : __expf(gt[bf * kDrChunk + m - 1])};
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) S[nt][e] *= decay_rows[e >> 1];
      const bf16* kt = Kt + bf * kDrChunk * kLdK;
#pragma unroll 1
      for (int kk = 0; kk < kDrChunk / 16; ++kk) {
        uint32_t a[4];
        frag_a_trans(a, kt, kLdK, 16 * mt, 16 * kk);
#pragma unroll
        for (int np = 0; np < (kNtw + 1) / 2; ++np) {
          uint32_t bb[4];
          frag_b2_trans(bb, Vb, kLdS, 16 * kk, 8 * ntb + 16 * np);
          mma_bf16_16x8x16(S[2 * np], a, bb[0], bb[1]);
          mma_bf16_16x8x16(S[2 * np + 1], a, bb[2], bb[3]);
        }
      }
      if (snapshot) {
#pragma unroll
        for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
            if (dv0 + c < dv) p.state_pool[snapshot_slot * p.state_slot_stride + int64_t(h) * dk * dv + pool_at(d, dv0 + c)] = S[nt][e];
          }
      }
    }
    __syncthreads();  // every read of Sb, Vb and Kt (output and update) is done
    if constexpr (kPrefetch) stage(bf ^ 1);
    if (owner) {  // publish the state for the next chunk's Q S
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2)
          *reinterpret_cast<uint32_t*>(&Sb[(16 * mt + g + 8 * e2) * kLdS + 8 * (ntb + nt) + 2 * t]) =
              pack_bf16x2(S[nt][2 * e2], S[nt][2 * e2 + 1]);
    }
    __syncthreads();
  };
  for (int chunk = 0; chunk + 1 < nchunks; ++chunk) iteration(chunk, std::true_type{});
  if (nchunks > 0) iteration(nchunks - 1, std::false_type{});
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int d = 16 * mt + g + 8 * (e >> 1), c = 8 * (ntb + nt) + 2 * t + (e & 1);
        if (dv0 + c < dv) pool[pool_at(d, dv0 + c)] = S[nt][e];
      }
  }
}

}  // namespace

// ---- host launchers ----------------------------------------------------------------------------

namespace {

constexpr int kMaxHeadDimV = 256;

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

template <typename F>
void by_dtype(DType dtype, F&& f) {
  if (dtype == DType::kFloat32) f(float{});
  else if (dtype == DType::kFloat16) f(__half{});
  else f(__nv_bfloat16{});
}

template <typename F>
void by_dk(int dk, F&& f) {
  switch (dk) {
    case 16: f(std::integral_constant<int, 16>{}); break;
    case 32: f(std::integral_constant<int, 32>{}); break;
    case 64: f(std::integral_constant<int, 64>{}); break;
    default: f(std::integral_constant<int, 128>{}); break;  // validated
  }
}

constexpr int64_t dr_align256(int64_t bytes) { return (bytes + 255) / 256 * 256; }

struct DrLayout {
  int64_t gc, A, w, u, h, v_new, total;  // byte offsets into the workspace
};

DrLayout dr_layout(int64_t tokens, int64_t batch, int64_t hv, int64_t dk, int64_t dv, bool per_channel) {
  DrLayout o{};
  int64_t at = 0;
  o.gc = at, at += dr_align256(4 * tokens * hv * (per_channel ? dk : 1));
  o.A = at, at += dr_align256(4 * tokens * hv * kDrChunk);
  o.w = at, at += dr_align256(4 * tokens * hv * dk);
  o.u = at, at += dr_align256(4 * tokens * hv * dv);
  o.h = at, at += dr_align256(4 * ((tokens + kDrChunk - 1) / kDrChunk + batch + 1) * hv * dk * dv);
  o.v_new = at, at += dr_align256(4 * tokens * hv * dv);
  o.total = at;
  return o;
}

// fp32 bodies ------------------------------------------------------------------------------------

template <typename T, bool kPer>
void launch_kkt(const DeltaRuleParams& p, cudaStream_t stream) {
  chunk_scaled_dot_kkt_kernel<T, kPer><<<dim3(unsigned(dr_grid_chunks(p)), p.num_v_heads, p.batch_size), kDrThreads, 0, stream>>>(p);
}

template <typename T, bool kPer>
void launch_w_u(const DeltaRuleParams& p, cudaStream_t stream) {
  const int64_t tiles = (p.head_dim_k + kDrChunk - 1) / kDrChunk + (p.head_dim_v + kDrChunk - 1) / kDrChunk;
  recompute_w_u_kernel<T, kPer><<<dim3(unsigned(dr_grid_chunks(p) * tiles), p.num_v_heads, p.batch_size), kDrThreads, 0, stream>>>(p);
}

template <typename T, bool kPer, int kDv>
cudaError_t launch_state(const DeltaRuleParams& p, cudaStream_t stream) {
  const int dk_pad = (p.head_dim_k + kDrChunk - 1) / kDrChunk * kDrChunk;
  const size_t bytes = dr_state_smem_bytes(dk_pad, kDv);
  const cudaError_t e = cudaFuncSetAttribute(chunk_state_fwd_kernel<T, kPer, kDv>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (e != cudaSuccess) return e;
  chunk_state_fwd_kernel<T, kPer, kDv><<<dim3((p.head_dim_v + kDv - 1) / kDv, p.num_v_heads, p.batch_size), kDrThreads, bytes, stream>>>(p);
  return cudaSuccess;
}

template <typename T, bool kPer, int kDv>
cudaError_t launch_output(const DeltaRuleParams& p, cudaStream_t stream) {
  const size_t bytes = dr_output_smem_bytes(kPer, kDv);
  const cudaError_t e = cudaFuncSetAttribute(chunk_output_fwd_kernel<T, kPer, kDv>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (e != cudaSuccess) return e;
  const int64_t tiles = (p.head_dim_v + kDv - 1) / kDv;
  chunk_output_fwd_kernel<T, kPer, kDv><<<dim3(unsigned(dr_grid_chunks(p) * tiles), p.num_v_heads, p.batch_size), kDrThreads, bytes, stream>>>(p);
  return cudaSuccess;
}

// tensor-core bodies (bf16 inputs only) ---------------------------------------------------------

template <bool kPer, int kDk>
cudaError_t launch_kkt_solve_wu_tc(const DeltaRuleParams& p, cudaStream_t stream) {
  using T = __nv_bfloat16;
  constexpr size_t bytes = kkt_solve_wu_smem_bytes<T, kDk, kPer>();
  cudaError_t e = cudaFuncSetAttribute(chunk_kkt_solve_wu_tc_kernel<T, kDk, kPer>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (e != cudaSuccess) return e;
  const dim3 grid(unsigned(dr_grid_chunks(p)), p.num_v_heads, p.batch_size);
  chunk_kkt_solve_wu_tc_kernel<T, kDk, kPer><<<grid, kDrThreads, bytes, stream>>>(p);
  return cudaSuccess;
}

template <bool kPer, int kDk, int kDv>
cudaError_t launch_state_output_tc(const DeltaRuleParams& p, cudaStream_t stream) {
  using T = __nv_bfloat16;
  constexpr size_t bytes = tc_state_output_smem_bytes<T, kPer, kDk, kDv>();
  cudaError_t e = cudaFuncSetAttribute(chunk_state_output_tc_kernel<T, kPer, kDk, kDv>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (e != cudaSuccess) return e;
  const dim3 grid((p.head_dim_v + kDv - 1) / kDv, p.num_v_heads, p.batch_size);
  chunk_state_output_tc_kernel<T, kPer, kDk, kDv><<<grid, kDrThreads, bytes, stream>>>(p);
  return cudaSuccess;
}

template <bool kPer, int kDk, int kDv>
cudaError_t launch_state_tc(const DeltaRuleParams& p, cudaStream_t stream) {
  using T = __nv_bfloat16;
  constexpr size_t bytes = tc_state_smem_bytes<kDk, kDv>();
  // 16-byte loads of k, W and the decays (aligned bases, k rows a whole number of 16-byte pieces apart) or the scalar fallback
  const bool vec = reinterpret_cast<uintptr_t>(p.w) % 16 == 0 && reinterpret_cast<uintptr_t>(p.gc) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(p.k) % 16 == 0 && (p.k_token_stride * int64_t(sizeof(T))) % 16 == 0 &&
                   (p.k_head_stride * int64_t(sizeof(T))) % 16 == 0;
  cudaError_t e = cudaSuccess;
  const auto run = [&](auto vec_tag) {
    constexpr bool kVec = decltype(vec_tag)::value;
    e = cudaFuncSetAttribute(chunk_state_fwd_tc_kernel<T, kPer, kDk, kDv, kVec>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    if (e != cudaSuccess) return;
    chunk_state_fwd_tc_kernel<T, kPer, kDk, kDv, kVec><<<dim3((p.head_dim_v + kDv - 1) / kDv, p.num_v_heads, p.batch_size), kDrThreads, bytes, stream>>>(p);
  };
  if (vec) run(std::true_type{});
  else run(std::false_type{});
  return e;
}

template <bool kPer, int kDk, int kDv>
cudaError_t launch_output_tc(const DeltaRuleParams& p, cudaStream_t stream) {
  using T = __nv_bfloat16;
  const int64_t tiles = (p.head_dim_v + kDv - 1) / kDv;
  const dim3 grid(unsigned(dr_grid_chunks(p) * tiles), p.num_v_heads, p.batch_size);
  if constexpr (kPer) {
    constexpr size_t bytes = tc_output_kda_smem_bytes<kDk, kDv>();
    const cudaError_t e = cudaFuncSetAttribute(chunk_output_fwd_tc_kda_kernel<T, kDk, kDv>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    if (e != cudaSuccess) return e;
    chunk_output_fwd_tc_kda_kernel<T, kDk, kDv><<<grid, kDrThreads, bytes, stream>>>(p);
  } else {
    constexpr size_t bytes = tc_output_smem_bytes<kDk, kDv>();
    const cudaError_t e = cudaFuncSetAttribute(chunk_output_fwd_tc_kernel<T, kDk, kDv>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    if (e != cudaSuccess) return e;
    chunk_output_fwd_tc_kernel<T, kDk, kDv><<<grid, kDrThreads, bytes, stream>>>(p);
  }
  return cudaSuccess;
}

/// Steps 2..6 of the pipeline (cumsum, kkt, solve_tril, w/u, state, output).
cudaError_t run_pipeline(const DeltaRuleParams& p, DType dtype, cudaStream_t stream) {
  const bool per = p.decay == DecayGranularity::kPerChannel, wide = p.head_dim_v > 32;
  const dim3 cgrid(unsigned(dr_grid_chunks(p)), p.num_v_heads, p.batch_size);
  if (per) chunk_local_cumsum_kernel<DecayGranularity::kPerChannel><<<cgrid, 128, 0, stream>>>(p);
  else chunk_local_cumsum_kernel<DecayGranularity::kPerHead><<<cgrid, 128, 0, stream>>>(p);
  cudaError_t e = cudaSuccess;
  if (p.math == ChunkMath::kTensorCore) {
    by_dk(p.head_dim_k, [&](auto dk) {
      constexpr int kDk = decltype(dk)::value;
      using T = __nv_bfloat16;
      const int64_t tiles = (p.head_dim_k + kDrChunk - 1) / kDrChunk + (p.head_dim_v + kDrChunk - 1) / kDrChunk;
      const dim3 wgrid(unsigned(dr_grid_chunks(p) * tiles), p.num_v_heads, p.batch_size);
      // fused kkt + solve + w/u (one launch, the 64 x 64 matrix stays in shared memory); a shape whose
      // shared-memory budget does not fit falls back to the three-kernel sequence.
      const cudaError_t fused = per ? launch_kkt_solve_wu_tc<true, kDk>(p, stream) : launch_kkt_solve_wu_tc<false, kDk>(p, stream);
      if (fused != cudaSuccess) {
        (void)cudaGetLastError();
        if (per) chunk_scaled_dot_kkt_tc_kda_kernel<T, kDk><<<cgrid, kDrThreads, 0, stream>>>(p);
        else chunk_scaled_dot_kkt_tc_kernel<T, kDk><<<cgrid, kDrThreads, 0, stream>>>(p);
        solve_tril_kernel<<<cgrid, kDrChunk, 0, stream>>>(p);
        if (per) recompute_w_u_tc_kernel<T, true><<<wgrid, kDrThreads, 0, stream>>>(p);
        else recompute_w_u_tc_kernel<T, false><<<wgrid, kDrThreads, 0, stream>>>(p);
      }
      // fused state + output: the state entering each chunk stays in registers/shared memory. The
      // 64-column state tile does not fit the registers with the prefetch at Dk = 128. A shape whose
      // budget does not fit falls back to the two chunk-parallel kernels.
      cudaError_t fused_out = cudaSuccess;
      if constexpr (kDk == 128) {  // the 64-column fused tile does not fit the registers at Dk = 128
        fused_out = per ? launch_state_output_tc<true, kDk, 32>(p, stream) : launch_state_output_tc<false, kDk, 32>(p, stream);
      } else {
        fused_out = per ? (wide ? launch_state_output_tc<true, kDk, 64>(p, stream) : launch_state_output_tc<true, kDk, 32>(p, stream))
                        : (wide ? launch_state_output_tc<false, kDk, 64>(p, stream) : launch_state_output_tc<false, kDk, 32>(p, stream));
      }
      if (fused_out == cudaSuccess) return;
      (void)cudaGetLastError();
      if constexpr (kDk == 128) {
        e = per ? launch_state_tc<true, kDk, 32>(p, stream) : launch_state_tc<false, kDk, 32>(p, stream);
      } else {
        e = per ? (wide ? launch_state_tc<true, kDk, 64>(p, stream) : launch_state_tc<true, kDk, 32>(p, stream))
                : (wide ? launch_state_tc<false, kDk, 64>(p, stream) : launch_state_tc<false, kDk, 32>(p, stream));
      }
      if (e != cudaSuccess) return;
      e = per ? (wide ? launch_output_tc<true, kDk, 64>(p, stream) : launch_output_tc<true, kDk, 32>(p, stream))
              : (wide ? launch_output_tc<false, kDk, 64>(p, stream) : launch_output_tc<false, kDk, 32>(p, stream));
    });
    return e;
  }
  by_dtype(dtype, [&](auto tag) {
    using T = decltype(tag);
    if (per) launch_kkt<T, true>(p, stream);
    else launch_kkt<T, false>(p, stream);
    solve_tril_kernel<<<cgrid, kDrChunk, 0, stream>>>(p);
    if (per) launch_w_u<T, true>(p, stream);
    else launch_w_u<T, false>(p, stream);
    e = per ? (wide ? launch_state<T, true, 64>(p, stream) : launch_state<T, true, 32>(p, stream))
            : (wide ? launch_state<T, false, 64>(p, stream) : launch_state<T, false, 32>(p, stream));
    if (e != cudaSuccess) return;
    e = per ? (wide ? launch_output<T, true, 64>(p, stream) : launch_output<T, true, 32>(p, stream))
            : (wide ? launch_output<T, false, 64>(p, stream) : launch_output<T, false, 32>(p, stream));
  });
  return e;
}

}  // namespace

Status delta_rule_workspace_requirements(int32_t max_tokens, int32_t max_batch, int32_t num_v_heads,
                                         int32_t head_dim_k, int32_t head_dim_v, int32_t chunk_size,
                                         DecayGranularity decay, WorkspaceRequirements* out) {
  if (out == nullptr || max_tokens < 0 || max_batch < 0 || num_v_heads < 1 || head_dim_k < 1 || head_dim_v < 1) {
    return detail::set_error(Status::kInvalidArgument, "delta_rule_workspace_requirements: invalid extents");
  }
  if (chunk_size != kDrChunk) {
    return detail::set_error(Status::kUnsupportedShape, "delta_rule_workspace_requirements: chunk_size must be 64");
  }
  *out = WorkspaceRequirements{};
  out->device_scratch_bytes = static_cast<size_t>(
      dr_layout(max_tokens, max_batch, num_v_heads, head_dim_k, head_dim_v, decay == DecayGranularity::kPerChannel).total);
  out->alignment = 256;
  return Status::kSuccess;
}

Status delta_rule_prefill(const DeltaRuleParams& params, DType dtype, const Workspace& ws, const LaunchContext& ctx) {
  const char* fn = "delta_rule_prefill";
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", fn);
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: q/k/v must be fp32, fp16 or bf16", fn);
  }
  DeltaRuleParams p = params;
  if (p.math != ChunkMath::kFp32 && p.math != ChunkMath::kTensorCore) {
    return detail::set_error(Status::kInvalidArgument, "%s: unknown ChunkMath", fn);
  }
  if (p.math == ChunkMath::kTensorCore) {
    if (dtype != DType::kBFloat16) {
      return detail::set_error(Status::kUnsupportedDtype, "%s: ChunkMath::kTensorCore needs bf16 q/k/v", fn);
    }
    if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
      return detail::set_error(Status::kUnsupportedArch, "%s: ChunkMath::kTensorCore needs SM80+", fn);
    }
  }
  if (p.batch_size < 0 || p.batch_size > 65535 || p.num_k_heads < 1 || p.num_v_heads < 1 || p.num_v_heads > 65535 ||
      p.num_v_heads % p.num_k_heads != 0 || p.num_slots < 0 || p.total_tokens < 0 ||
      p.total_tokens > INT32_MAX / 2 || p.max_seqlen < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B, Hv <= 65535, Hk | Hv)", fn);
  }
  if (p.decay != DecayGranularity::kPerHead && p.decay != DecayGranularity::kPerChannel) {
    return detail::set_error(Status::kInvalidArgument, "%s: unknown decay granularity", fn);
  }
  if (p.head_dim_k != 16 && p.head_dim_k != 32 && p.head_dim_k != 64 && p.head_dim_k != 128) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head_dim_k must be 16, 32, 64 or 128", fn);
  }
  if (p.head_dim_v < 1 || p.head_dim_v > kMaxHeadDimV) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head_dim_v must be in 1..%d", fn, kMaxHeadDimV);
  }
  if (!std::isfinite(p.scale)) return detail::set_error(Status::kInvalidArgument, "%s: scale must be finite", fn);
  if (p.state_slot_stride < int64_t(p.num_v_heads) * p.head_dim_k * p.head_dim_v) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: state slots overlap (slot stride)", fn);
  }
  if (p.snapshot_slots ? (p.snapshot_interval <= 0 || p.snapshot_interval % kDrChunk != 0 || p.max_snapshots < 0)
                       : p.snapshot_interval != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: snapshot_interval must be a positive multiple of 64 with snapshot_slots", fn);
  }
  const int64_t chunks = ((p.max_seqlen > 0 ? p.max_seqlen : p.total_tokens) + kDrChunk - 1) / kDrChunk;
  if (chunks * ((p.head_dim_v + kDrChunk - 1) / kDrChunk + (p.head_dim_k + kDrChunk - 1) / kDrChunk) > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: too many chunks", fn);
  }
  if (p.batch_size == 0 || chunks == 0) return Status::kSuccess;
  if (!p.q || !p.k || !p.v || !p.g || !p.beta || !p.o || !p.cu_seqlens || !p.state_pool) {
    return detail::set_error(Status::kInvalidArgument, "%s: q, k, v, g, beta, o, cu_seqlens, state_pool required", fn);
  }
  if (p.o == p.q || p.o == p.k || p.o == p.v) {
    return detail::set_error(Status::kInvalidArgument, "%s: o must be disjoint from q, k and v", fn);
  }
  if (p.q_token_stride < 0 || p.q_head_stride < 0 || p.k_token_stride < 0 || p.k_head_stride < 0 ||
      p.v_token_stride < 0 || p.v_head_stride < 0 || p.o_token_stride < 0 || p.o_head_stride < 0) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides", fn);
  }
  const DrLayout layout = dr_layout(p.total_tokens, p.batch_size, p.num_v_heads, p.head_dim_k, p.head_dim_v,
                                    p.decay == DecayGranularity::kPerChannel);
  if (ws.device_scratch == nullptr || ws.device_scratch_bytes < static_cast<size_t>(layout.total) ||
      reinterpret_cast<uintptr_t>(ws.device_scratch) % 256 != 0) {
    return detail::set_error(Status::kWorkspaceTooSmall, "%s: need %lld scratch bytes, 256-aligned", fn,
                             static_cast<long long>(layout.total));
  }
  uint8_t* base = static_cast<uint8_t*>(ws.device_scratch);
  p.gc = reinterpret_cast<float*>(base + layout.gc);
  p.A = reinterpret_cast<float*>(base + layout.A);
  p.w = reinterpret_cast<float*>(base + layout.w);
  p.u = reinterpret_cast<float*>(base + layout.u);
  p.h = reinterpret_cast<float*>(base + layout.h);
  p.v_new = reinterpret_cast<float*>(base + layout.v_new);
  const cudaError_t e = run_pipeline(p, dtype, stream_of(ctx));
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e), "%s: %s", fn, cudaGetErrorString(e));
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", fn, cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

}  // namespace kops::state
