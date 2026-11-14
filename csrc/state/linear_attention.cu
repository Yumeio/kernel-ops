// =============================================================================
// kernel-ops | csrc/state/linear_attention.cu
// B9 decayed linear attention (L02, MiniMax-01 / MiniMax-M1 lightning attention, RetNet-style):
//   row 2  linear_attention_decode_kernel      one token per sequence, one thread per value column
//   row 5  lightning_attention_prefill_kernel  64-token blocks: intra-block masked attention with
//                                              decay + inter-block state recurrence
// S_t = exp(-slope_h) S_{t-1} + k_t^T v_t, o_t = scale * q_t S_t (current token included). Only
// decaying factors (<= 1) appear, so nothing overflows for any slope >= 0; all arithmetic is fp32
// FMA in a fixed order, so a call cut at a block boundary and resumed from the saved state is
// bitwise the one-call result. Contract: linear_attention_kernels.cuh, ops.h, CONTRACTS.md §1f.
// Ported from the ayaka implementation (3rd/csrc/state/linear_attention.cu).
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cmath>
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

namespace kops::state {
namespace {

using dev::smem_gemm_kmajor;

template <typename T>
__device__ __forceinline__ float to_float(T v) { return dev::to_f(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v) {
  T r;
  dev::from_f(r, v);
  return r;
}

}  // namespace

namespace {
// ---- ported from ayaka state/linear_attention.cu:25-219 ----
constexpr int kDecodeThreads = 128;
constexpr int kMaxHeadDim = 256;

// ---- linear_attention_decode_kernel ----------------------------------------------------------------------

/**
 * Grid (ceil(Dv / kDecodeThreads), H, B). A thread owns one value column dv of the S tile and walks down the Dk rows of its
 * (sequence, head): S[dk, dv] = decay * S[dk, dv] + k[dk] * v[dv]; o[dv] += q[dk] * S[dk, dv]. Neighbouring threads touch
 * neighbouring columns of a row, so every access is coalesced and no cross-lane reduction is needed.
 */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kDecodeThreads) void linear_attention_decode_kernel(const __grid_constant__ LinearAttentionParams p) {
  __shared__ float q_s[kMaxHeadDim], k_s[kMaxHeadDim];
  const int b = blockIdx.z, h = blockIdx.y;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;
  const T* q = static_cast<const T*>(p.q) + int64_t(b) * p.q_token_stride + int64_t(h) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(b) * p.k_token_stride + int64_t(h) * p.k_head_stride;
  for (int i = threadIdx.x; i < p.head_dim_k; i += kDecodeThreads) {
    q_s[i] = to_float(q[i]);
    k_s[i] = to_float(k[i]);
  }
  __syncthreads();
  const int dv = blockIdx.x * kDecodeThreads + threadIdx.x;
  if (dv >= p.head_dim_v) return;
  const float vv = to_float(static_cast<const T*>(p.v)[int64_t(b) * p.v_token_stride + int64_t(h) * p.v_head_stride + dv]);
  const float decay = __expf(-p.slope[h]);
  float* s = p.state_pool + slot * p.state_slot_stride + int64_t(h) * p.head_dim_k * p.head_dim_v + dv;
  float acc = 0.0f;
#pragma unroll 8
  for (int dk = 0; dk < p.head_dim_k; ++dk) {
    const float v_new = __fmaf_rn(k_s[dk], vv, decay * s[int64_t(dk) * p.head_dim_v]);
    s[int64_t(dk) * p.head_dim_v] = v_new;
    acc = __fmaf_rn(q_s[dk], v_new, acc);
  }
  static_cast<T*>(p.o)[int64_t(b) * p.o_token_stride + int64_t(h) * p.o_head_stride + dv] = from_float<T>(acc * p.scale);
}
namespace {

// ---- lightning_attention_prefill_kernel -------------------------------------------------------------------

constexpr int kLightThreads = 256;  // a 16 x 16 grid of threads, each with a 4-row register tile
constexpr int kLightBlock = 64;     // tokens per block
constexpr int kLightSlice = 32;     // features per slice of the q k^T and q S products
constexpr int kLightLd = kLightBlock + 4;      // row stride of the staged [feature][token] tiles: 16-byte rows, spread banks
constexpr int kLightKdChunk = 64;              // rows of the state updated at once
constexpr int kLightUnionFloats = 2 * kLightSlice * kLightLd + kLightBlock * kLightLd;  // q, k slices and P; aliased by k^T V's operand

__host__ __device__ constexpr size_t lightning_smem_bytes(int dk_pad, int dv_tile) {
  return (size_t(dk_pad) * dv_tile + kLightUnionFloats + size_t(kLightBlock) * (dv_tile + 4) + 68) * sizeof(float);
}

/**
 * Grid (ceil(Dv / kDv), H, B): a CTA owns the S[:, dv tile] columns of one (sequence, head) in shared memory and walks the
 * sequence in blocks of kLightBlock tokens. Per block, with decay powers w^i = exp(-slope i) in a table:
 *   P = (Q K^T) * w^(i - j) for j <= i             64 x 64, accumulated over feature slices of 32 together with Q S,
 *   O = w^(i + 1) (Q S) + P V                       64 x kDv, V staged once per block,
 *   S <- w^m S + (w^(m - 1 - j) K)^T V             Dk x kDv, 64 rows at a time (m = tokens in the block).
 * Every product is smem_gemm_kmajor on operands staged zero-padded (tokens beyond the block, features beyond Dk), so a partial
 * block needs no special case. The tiles of one product are all read before any barrier that lets the staging of the next
 * overwrite them; the q/k slices, P and the operand of the state update share one region.
 */
}  // namespace
template <typename T, int kDv>
__global__ __launch_bounds__(kLightThreads, 1) void lightning_attention_prefill_kernel(const __grid_constant__ LinearAttentionParams p) {
  constexpr int RN = kDv / 16;
  extern __shared__ __align__(16) float smem[];
  const int dk = p.head_dim_k, dv = p.head_dim_v;
  const int dk_pad = (dk + kLightKdChunk - 1) / kLightKdChunk * kLightKdChunk;
  float* S = smem;                                // [dk_pad][kDv]
  float* qs = S + dk_pad * kDv;                   // [kLightSlice][kLightLd], feature-major
  float* ks = qs + kLightSlice * kLightLd;
  float* ps = ks + kLightSlice * kLightLd;        // ps[j][i]: P transposed
  float* kd = qs;                                 // [kLightBlock][kLightKdChunk + 4], token-major (aliases qs, ks, ps)
  float* vs = qs + kLightUnionFloats;             // [kLightBlock][kDv + 4]
  float* pw = vs + kLightBlock * (kDv + 4);       // exp(-slope i), i = 0..64

  const int b = blockIdx.z, h = blockIdx.y, dv0 = blockIdx.x * kDv;
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int row0 = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - row0;
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  const T* q = static_cast<const T*>(p.q) + int64_t(h) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(h) * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + int64_t(h) * p.v_head_stride;
  T* o = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
  float* pool = p.state_pool + slot * p.state_slot_stride + int64_t(h) * dk * dv;  // [Dk][Dv]

  if (tid <= kLightBlock) pw[tid] = __expf(-p.slope[h] * float(tid));
  for (int idx = tid; idx < dk_pad * kDv; idx += kLightThreads) {
    const int kk = idx / kDv, c = idx % kDv;
    S[idx] = (has_init && kk < dk && dv0 + c < dv) ? pool[int64_t(kk) * dv + dv0 + c] : 0.0f;
  }

  for (int t0 = 0; t0 < n; t0 += kLightBlock) {
    const int m = min(kLightBlock, n - t0);
    float pacc[4][4], oacc[4][RN];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c) pacc[r][c] = 0.0f;
#pragma unroll
      for (int c = 0; c < RN; ++c) oacc[r][c] = 0.0f;
    }
    for (int d0 = 0; d0 < dk_pad; d0 += kLightSlice) {
      __syncthreads();  // the previous slice (or block) is done with qs/ks/kd, and S is written
      for (int idx = tid; idx < kLightBlock * kLightSlice; idx += kLightThreads) {
        const int kk = idx % kLightSlice, i = idx / kLightSlice;
        float qv = 0.0f, kv = 0.0f;
        if (i < m && d0 + kk < dk) {
          const int64_t row = int64_t(row0 + t0 + i);
          qv = to_float(q[row * p.q_token_stride + d0 + kk]);
          kv = to_float(k[row * p.k_token_stride + d0 + kk]);
        }
        qs[kk * kLightLd + i] = qv;
        ks[kk * kLightLd + i] = kv;
      }
      __syncthreads();
      smem_gemm_kmajor<4, 4>(pacc, qs, kLightLd, ty * 4, ks, kLightLd, tx * 4, kLightSlice);
      smem_gemm_kmajor<4, RN>(oacc, qs, kLightLd, ty * 4, S + d0 * kDv, kDv, tx * RN, kLightSlice);
    }
    // P: the causal decay mask, stored transposed for the P V product; O: the part from before the block
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float col[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int i = ty * 4 + r, j = tx * 4 + c;
        col[r] = j <= i ? pacc[r][c] * pw[i - j] : 0.0f;
      }
      *reinterpret_cast<float4*>(&ps[(tx * 4 + c) * kLightLd + ty * 4]) = make_float4(col[0], col[1], col[2], col[3]);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < RN; ++c) oacc[r][c] *= pw[ty * 4 + r + 1];
    }
    for (int idx = tid; idx < kLightBlock * kDv; idx += kLightThreads) {
      const int c = idx % kDv, j = idx / kDv;
      vs[j * (kDv + 4) + c] = (j < m && dv0 + c < dv) ? to_float(v[int64_t(row0 + t0 + j) * p.v_token_stride + dv0 + c]) : 0.0f;
    }
    __syncthreads();
    smem_gemm_kmajor<4, RN>(oacc, ps, kLightLd, ty * 4, vs, kDv + 4, tx * RN, kLightBlock);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = ty * 4 + r;
      if (i < m) {
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          if (dv0 + tx * RN + c < dv) o[int64_t(row0 + t0 + i) * p.o_token_stride + dv0 + tx * RN + c] = from_float<T>(oacc[r][c] * p.scale);
        }
      }
    }

    // state: S <- w^m S + (w^(m - 1 - j) K)^T V, 64 rows at a time
    const int done = t0 + m;
    int64_t snapshot_slot = -1;
    if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
      const int j = done / p.snapshot_interval - 1;
      if (j < p.max_snapshots) snapshot_slot = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
    }
    const bool snapshot = snapshot_slot >= 0 && snapshot_slot < p.num_slots;
    for (int d0 = 0; d0 < dk_pad; d0 += kLightKdChunk) {
      __syncthreads();  // ps / qs / ks, which kd aliases, are no longer read
      for (int idx = tid; idx < kLightBlock * kLightKdChunk; idx += kLightThreads) {
        const int kk = idx % kLightKdChunk, j = idx / kLightKdChunk;
        kd[j * (kLightKdChunk + 4) + kk] =
            (j < m && d0 + kk < dk) ? pw[m - 1 - j] * to_float(k[int64_t(row0 + t0 + j) * p.k_token_stride + d0 + kk]) : 0.0f;
      }
      __syncthreads();
      float sacc[4][RN];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < RN; ++c) sacc[r][c] = pw[m] * S[(d0 + ty * 4 + r) * kDv + tx * RN + c];
      }
      smem_gemm_kmajor<4, RN>(sacc, kd, kLightKdChunk + 4, ty * 4, vs, kDv + 4, tx * RN, kLightBlock);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int kk = d0 + ty * 4 + r;
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          S[kk * kDv + tx * RN + c] = sacc[r][c];
          if (snapshot && kk < dk && dv0 + tx * RN + c < dv)
            p.state_pool[snapshot_slot * p.state_slot_stride + int64_t(h) * dk * dv + int64_t(kk) * dv + dv0 + tx * RN + c] = sacc[r][c];
        }
      }
    }
  }
  __syncthreads();
  for (int idx = tid; idx < dk * kDv; idx += kLightThreads) {
    const int kk = idx / kDv, c = idx % kDv;
    if (dv0 + c < dv) pool[int64_t(kk) * dv + dv0 + c] = S[kk * kDv + c];
  }
}
namespace {

// ---- lightning_attention_prefill_tc_kernel ----------------------------------------------------------------

constexpr int kLightTcThreads = kLightThreads;

/// Dynamic shared memory of the tensor-core lightning body: q, k [64][tc_ld(Dk)], the raw / decayed v
/// tiles [64][tc_ld(kDv)] (two copies of the decayed operand, hi + lo), the P tile [64][tc_ld(64)], the
/// state [Dk][tc_ld(kDv)] and the 65 decay powers.
__host__ __device__ constexpr size_t lightning_tc_smem_bytes(int dk, int dv_tile) {
  return (size_t(2) * kLightBlock * mma::tc_ld(dk) + size_t(3) * kLightBlock * mma::tc_ld(dv_tile) +
          size_t(kLightBlock) * mma::tc_ld(kLightBlock) + size_t(dk) * mma::tc_ld(dv_tile)) *
             sizeof(__nv_bfloat16) +
         66 * sizeof(float);
}

/**
 * One persistent CTA per (sequence, head, Dv tile): the state S[:, dv tile] lives in the fp32 accumulator
 * registers of the warps that own its 16-row tiles and is published as a 16-bit shared-memory copy once per
 * block. Per 64-token block, with the decay powers w^i = exp(-slope i) of the block:
 *   phase 1  P = (Q K^T) w^(i - j), j <= i                      64 x 64, mma over the Dk features
 *   phase 2  O = w^(i + 1) (Q S)                                64 x kDv
 *   phase 3  O += P V                                           64 x kDv (P and V are 16-bit tiles)
 *   phase 4  S <- w^m S + (w^(m - 1 - j) K)^T V                 Dk x kDv, fp32 accumulator
 * The state-update operand (the decayed V) is split into a 16-bit hi + lo pair and multiplied twice, so the
 * fp32 state keeps ~fp32 accuracy across the block walk; the other phases only feed the 16-bit output.
 * Every tile is staged zero-padded, so a partial block and a partial Dv tile need no special case.
 */
}  // namespace
template <typename T, int kDv>
__global__ __launch_bounds__(kLightTcThreads, 1) void lightning_attention_prefill_tc_kernel(const __grid_constant__ LinearAttentionParams p) {
  constexpr int kLdV = mma::tc_ld(kDv), kLdP = mma::tc_ld(kLightBlock);
  constexpr int kNtw = kDv / 8, kNt1 = kNtw / 2;
  static_assert(kNt1 >= 2 && kNt1 % 2 == 0, "the output tile is formed two n8 tiles at a time");
  extern __shared__ __align__(16) unsigned char light_smem[];
  const int dk = p.head_dim_k, dv = p.head_dim_v, kLd = mma::tc_ld(dk);
  T* qs = reinterpret_cast<T*>(light_smem);        // [64][kLd]
  T* ks = qs + kLightBlock * kLd;                  // [64][kLd]
  T* vs = ks + kLightBlock * kLd;                  // [64][kLdV]: raw V
  T* vh = vs + kLightBlock * kLdV;                 // [64][kLdV]: w^(m - 1 - j) V, hi half
  T* vl = vh + kLightBlock * kLdV;                 // [64][kLdV]: the lo half of the same operand
  T* ps = vl + kLightBlock * kLdV;                 // [64][kLdP]: P (causal, decayed)
  T* sb = ps + kLightBlock * kLdP;                 // [dk][kLdV]: the state at the start of the block
  float* pw = reinterpret_cast<float*>(sb + dk * kLdV);  // [65]
  const int b = blockIdx.z, h = blockIdx.y, dv0 = blockIdx.x * kDv;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int row0 = p.cu_seqlens[b], n = p.cu_seqlens[b + 1] - row0;
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];
  const T* q = static_cast<const T*>(p.q) + int64_t(h) * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + int64_t(h) * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + int64_t(h) * p.v_head_stride + dv0;
  T* o = static_cast<T*>(p.o) + int64_t(h) * p.o_head_stride;
  float* pool = p.state_pool + slot * p.state_slot_stride + int64_t(h) * dk * dv;
  const bool vec_q = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(q) % 16 == 0 && p.q_token_stride % 8 == 0;
  const bool vec_k = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(k) % 16 == 0 && p.k_token_stride % 8 == 0;
  const int dvcols = min(kDv, dv - dv0);
  const int I = wid & 3, half = wid >> 2;
  const bool owner = wid * 16 < dk;

  float S[kNtw][4] = {};
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int d = 16 * wid + gq + 8 * (e >> 1), c = 8 * nt + 2 * t + (e & 1);
        S[nt][e] = (has_init && dv0 + c < dv) ? pool[int64_t(d) * dv + dv0 + c] : 0.0f;
      }
  }
  if (tid <= kLightBlock) pw[tid] = __expf(-p.slope[h] * float(tid));

  for (int t0 = 0; t0 < n; t0 += kLightBlock) {
    const int m = min(kLightBlock, n - t0);
    __syncthreads();  // the previous block is done with every tile and with sb
    mma::tc_stage_input<T, kLightTcThreads>(qs, kLd, q, p.q_token_stride, row0 + t0, m, dk, vec_q);
    mma::tc_stage_input<T, kLightTcThreads>(ks, kLd, k, p.k_token_stride, row0 + t0, m, dk, vec_k);
    for (int idx = tid; idx < kLightBlock * (kDv / 8); idx += kLightTcThreads) {
      const int j = idx / (kDv / 8), c = (idx % (kDv / 8)) * 8;
      const bool row_ok = j < m;
      const float w = row_ok ? pw[m - 1 - j] : 0.0f;
      float vv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        vv[i] = (row_ok && c + i < dvcols) ? to_float(v[int64_t(row0 + t0 + j) * p.v_token_stride + c + i]) : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        vs[j * kLdV + c + i] = from_float<T>(vv[i]);
        const T hi = from_float<T>(w * vv[i]);
        vh[j * kLdV + c + i] = hi;
        vl[j * kLdV + c + i] = from_float<T>(w * vv[i] - to_float(hi));
      }
    }
    if (owner) {  // the state entering the block, for the Q S product
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2)
          *reinterpret_cast<uint32_t*>(&sb[(16 * wid + gq + 8 * e2) * kLdV + 8 * nt + 2 * t]) =
              mma::Bits16<T>::pack(S[nt][2 * e2], S[nt][2 * e2 + 1]);
    }
    __syncthreads();
    float qk[4][4] = {}, out[kNt1][4] = {};
    for (int kk = 0; kk < dk / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, qs, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        mma::frag_b2(b, ks, kLd, 32 * half + 16 * np, 16 * kk);
        mma::Mma16x8x16<T>::run(qk[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(qk[2 * np + 1], a, b[2], b[3]);
      }
    }
    // phase 2: O = w^(i + 1) (Q S) over the staged state copy
    for (int kk = 0; kk < dk / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, qs, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < kNt1 / 2; ++np) {
        uint32_t b[4];
        mma::frag_b2_trans(b, sb, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
        mma::Mma16x8x16<T>::run(out[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(out[2 * np + 1], a, b[2], b[3]);
      }
    }
    // P: the causal decay mask, stored for the P V product; O: the part from before the block
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = 16 * I + gq + 8 * rr;
      const float wi = pw[i + 1];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int j = 32 * half + 8 * nt + 2 * t;
        const float p0 = j <= i ? qk[nt][2 * rr] * pw[i - j] : 0.0f;
        const float p1 = j + 1 <= i ? qk[nt][2 * rr + 1] * pw[i - j - 1] : 0.0f;
        *reinterpret_cast<uint32_t*>(&ps[i * kLdP + j]) = mma::Bits16<T>::pack(p0, p1);
      }
#pragma unroll
      for (int nt = 0; nt < kNt1; ++nt) {
        out[nt][2 * rr] *= wi;
        out[nt][2 * rr + 1] *= wi;
      }
    }
    __syncthreads();  // P is complete
    for (int kk = 0; kk < kLightBlock / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, ps, kLdP, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < kNt1 / 2; ++np) {
        uint32_t b[4];
        mma::frag_b2_trans(b, vs, kLdV, 16 * kk, 8 * (half * kNt1) + 16 * np);
        mma::Mma16x8x16<T>::run(out[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(out[2 * np + 1], a, b[2], b[3]);
      }
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = 16 * I + gq + 8 * rr;
      if (i >= m) continue;
#pragma unroll
      for (int nt = 0; nt < kNt1; ++nt) {
        const int c = dv0 + 8 * (half * kNt1 + nt) + 2 * t;
        if (c < dv) o[int64_t(row0 + t0 + i) * p.o_token_stride + c] = from_float<T>(out[nt][2 * rr] * p.scale);
        if (c + 1 < dv)
          o[int64_t(row0 + t0 + i) * p.o_token_stride + c + 1] = from_float<T>(out[nt][2 * rr + 1] * p.scale);
      }
    }
    // phase 4: the state after the block, S <- w^m S + (w^(m - 1 - j) K)^T V (the decayed operand in hi + lo)
    if (owner) {
#pragma unroll
      for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) S[nt][e] *= pw[m];
      for (int kk = 0; kk < kLightBlock / 16; ++kk) {
        uint32_t a[4];
        mma::frag_a_trans(a, ks, kLd, 16 * wid, 16 * kk);
#pragma unroll
        for (int np = 0; np < kNtw / 2; ++np) {
          uint32_t bh[4], bl[4];
          mma::frag_b2_trans(bh, vh, kLdV, 16 * kk, 16 * np);
          mma::frag_b2_trans(bl, vl, kLdV, 16 * kk, 16 * np);
          mma::Mma16x8x16<T>::run(S[2 * np], a, bh[0], bh[1]);
          mma::Mma16x8x16<T>::run(S[2 * np + 1], a, bh[2], bh[3]);
          mma::Mma16x8x16<T>::run(S[2 * np], a, bl[0], bl[1]);
          mma::Mma16x8x16<T>::run(S[2 * np + 1], a, bl[2], bl[3]);
        }
      }
      const int done = t0 + m;
      int64_t snapshot_slot = -1;
      if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
        const int j = done / p.snapshot_interval - 1;
        if (j < p.max_snapshots) snapshot_slot = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
      }
      if (snapshot_slot >= 0 && snapshot_slot < p.num_slots) {
        float* snap = p.state_pool + snapshot_slot * p.state_slot_stride + int64_t(h) * dk * dv;
#pragma unroll
        for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int d = 16 * wid + gq + 8 * (e >> 1), c = 8 * nt + 2 * t + (e & 1);
            if (dv0 + c < dv) snap[int64_t(d) * dv + dv0 + c] = S[nt][e];
          }
      }
    }
  }
  __syncthreads();
  if (owner) {
#pragma unroll
    for (int nt = 0; nt < kNtw; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int d = 16 * wid + gq + 8 * (e >> 1), c = 8 * nt + 2 * t + (e & 1);
        if (dv0 + c < dv) pool[int64_t(d) * dv + dv0 + c] = S[nt][e];
      }
  }
}
namespace {


}  // namespace

// ---- host launchers ----------------------------------------------------------------------------

namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

Status launch_status(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

template <typename F>
void by_dtype(DType dtype, F&& f) {
  if (dtype == DType::kFloat32) f(float{});
  else if (dtype == DType::kFloat16) f(__half{});
  else f(__nv_bfloat16{});
}

/// Checks shared by decode and prefill; `tokens` is the token count the q/k/v rows span.
Status check_linear(const char* fn, const LinearAttentionParams& p, DType dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", fn);
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: q/k/v must be fp32, fp16 or bf16", fn);
  }
  if (!std::isfinite(p.scale)) {
    return detail::set_error(Status::kInvalidArgument, "%s: scale must be finite", fn);
  }
  if (p.batch_size < 0 || p.batch_size > 65535 || p.num_heads < 0 || p.num_heads > 65535 ||
      p.num_slots < 0 || p.total_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B, H <= 65535)", fn);
  }
  if (p.head_dim_k < 1 || p.head_dim_k > kMaxHeadDim || p.head_dim_v < 1 || p.head_dim_v > kMaxHeadDim) {
    return detail::set_error(Status::kUnsupportedShape, "%s: Dk and Dv must be in 1..%d", fn, kMaxHeadDim);
  }
  if (p.q_token_stride < 0 || p.q_head_stride < 0 || p.k_token_stride < 0 || p.k_head_stride < 0 ||
      p.v_token_stride < 0 || p.v_head_stride < 0 || p.o_token_stride < 0 || p.o_head_stride < 0 ||
      p.state_slot_stride < int64_t(p.num_heads) * p.head_dim_k * p.head_dim_v) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides or overlapping state slots", fn);
  }
  if (p.batch_size > 0 && p.num_heads > 0 &&
      (!p.q || !p.k || !p.v || !p.o || !p.slope || !p.state_pool)) {
    return detail::set_error(Status::kInvalidArgument, "%s: q, k, v, o, slope, state_pool required", fn);
  }
  if (p.o != nullptr && (p.o == p.q || p.o == p.k || p.o == p.v)) {
    return detail::set_error(Status::kInvalidArgument, "%s: o must be disjoint from q, k and v", fn);
  }
  return Status::kSuccess;
}

}  // namespace

Status linear_attention_decode(const LinearAttentionParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "linear_attention_decode";
  Status s = check_linear(fn, params, dtype, ctx);
  if (s != Status::kSuccess) return s;
  const LinearAttentionParams& p = params;
  if (p.batch_size == 0 || p.num_heads == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>((p.head_dim_v + kDecodeThreads - 1) / kDecodeThreads),
                  static_cast<unsigned>(p.num_heads), static_cast<unsigned>(p.batch_size));
  cudaStream_t stream = stream_of(ctx);
  by_dtype(dtype, [&](auto tag) {
    linear_attention_decode_kernel<decltype(tag)><<<grid, kDecodeThreads, 0, stream>>>(p);
  });
  return launch_status(fn);
}

Status lightning_attention_prefill(const LinearAttentionParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "lightning_attention_prefill";
  Status s = check_linear(fn, params, dtype, ctx);
  if (s != Status::kSuccess) return s;
  const LinearAttentionParams& p = params;
  if (p.snapshot_slots ? (p.snapshot_interval <= 0 || p.snapshot_interval % kLightBlock != 0 || p.max_snapshots < 0)
                       : p.snapshot_interval != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: snapshot_interval must be a positive multiple of %d with snapshot_slots", fn,
                             kLightBlock);
  }
  if (p.batch_size == 0 || p.num_heads == 0) return Status::kSuccess;
  if (!p.cu_seqlens) return detail::set_error(Status::kInvalidArgument, "%s: cu_seqlens required", fn);
  cudaStream_t stream = stream_of(ctx);
  const int dk_pad = (p.head_dim_k + kLightKdChunk - 1) / kLightKdChunk * kLightKdChunk;
  cudaError_t attr = cudaSuccess;
  // 16-bit q/k/v on SM80+ walk 64-token blocks on mma.sync; the fp32 body stays the reference path
  // (and the fallback when a shape or the shared-memory budget does not fit the tensor-core body).
  const bool tc = dtype != DType::kFloat32 && (ctx.arch_id == 0 || ctx.arch_id >= KOPS_TARGET_SM80) &&
                  p.head_dim_k % 16 == 0 && p.head_dim_k <= 128;
  if (tc) {
    const auto go_tc = [&](auto tag, auto tile) {
      using T = decltype(tag);
      constexpr int kDv = decltype(tile)::value;
      const size_t bytes = lightning_tc_smem_bytes(p.head_dim_k, kDv);
      attr = cudaFuncSetAttribute(lightning_attention_prefill_tc_kernel<T, kDv>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
      if (attr != cudaSuccess) return;
      const dim3 grid(static_cast<unsigned>((p.head_dim_v + kDv - 1) / kDv), static_cast<unsigned>(p.num_heads),
                      static_cast<unsigned>(p.batch_size));
      lightning_attention_prefill_tc_kernel<T, kDv><<<grid, kLightTcThreads, bytes, stream>>>(p);
    };
    if (dtype == DType::kBFloat16) {
      if (p.head_dim_v > 32) go_tc(__nv_bfloat16{}, std::integral_constant<int, 64>{});
      else go_tc(__nv_bfloat16{}, std::integral_constant<int, 32>{});
    } else {
      if (p.head_dim_v > 32) go_tc(__half{}, std::integral_constant<int, 64>{});
      else go_tc(__half{}, std::integral_constant<int, 32>{});
    }
    if (attr == cudaSuccess) return launch_status(fn);
    (void)cudaGetLastError();  // the tensor-core tile does not fit: run the fp32 body instead
  }
  by_dtype(dtype, [&](auto tag) {
    using T = decltype(tag);
    const auto go = [&](auto tile) {
      constexpr int kDv = decltype(tile)::value;
      const size_t bytes = lightning_smem_bytes(dk_pad, kDv);
      attr = cudaFuncSetAttribute(lightning_attention_prefill_kernel<T, kDv>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
      if (attr != cudaSuccess) return;
      const dim3 grid(static_cast<unsigned>((p.head_dim_v + kDv - 1) / kDv), static_cast<unsigned>(p.num_heads),
                      static_cast<unsigned>(p.batch_size));
      lightning_attention_prefill_kernel<T, kDv><<<grid, kLightThreads, bytes, stream>>>(p);
    };
    if (p.head_dim_k <= 128 && p.head_dim_v > 32) go(std::integral_constant<int, 64>());
    else go(std::integral_constant<int, 32>());
  });
  if (attr != cudaSuccess) {
    (void)cudaGetLastError();
    return detail::set_backend_error(Status::kUnsupportedShape, static_cast<int32_t>(attr),
                                     "%s: shared memory for Dk %d exceeds the device limit", fn, p.head_dim_k);
  }
  return launch_status(fn);
}

}  // namespace kops::state
