// =============================================================================
// kernel-ops | csrc/state/ssm.cu
// B9 selective state-space models (L02 SSM):
//   row 2  selective_state_update_kernel   Mamba-1 / Mamba-2 decode, 1..S tokens per sequence
//   row 4  selective_scan_fwd_kernel        Mamba-1 varlen prefill, parallel affine scan
//   row 6  ssd_chunk_cumsum / ssd_bmm_chunk / ssd_chunk_state / ssd_state_passing /
//          ssd_chunk_scan                   Mamba-2 SSD chunked prefill (5 steps, mamba_ssm order)
// States are fp32 in caller-owned slots. A state row is one (head, p) pair of dstate values;
// Mamba-1 is nheads = dim, headdim = 1 with a per-state A. Reductions over dstate are fp32 in a
// fixed order that does not depend on the batch, so a sequence's result is independent of the
// other sequences of the call. Contract: csrc/state/ssm_kernels.cuh, ops.h, CONTRACTS.md §1f.
// Ported from the ayaka implementation (3rd/csrc/state/ssm.cu).
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

namespace kops::state {
namespace {

using dev::group_sum;
using dev::silu_f;
using dev::smem_gemm_kmajor;
using dev::softplus_f;

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
// ---- ported from ayaka state/ssm.cu:27-107 ----
// ---- selective_state_update_kernel -----------------------------------------------------------------------

constexpr int kUpdateWarps = 4;
constexpr int kUpdateRowsPerWarp = 8;  // state rows a warp walks through, sequentially

/**
 * Grid (ceil(nheads * headdim / (kUpdateWarps * kUpdateRowsPerWarp)), B). The kLanesN lanes of a row group each hold
 * kNPerLane state values (n = lane + kLanesN * j, so every access of the warp is a coalesced run); a warp serves
 * 32 / kLanesN rows at once. A row loops over the S tokens with its state in registers and reduces y over the lane group.
 */
}  // namespace
template <typename T, int kLanesN, int kNPerLane>
__global__ __launch_bounds__(kUpdateWarps * 32) void selective_state_update_kernel(const __grid_constant__ SSMStateUpdateParams p) {
  constexpr int kRowGroups = 32 / kLanesN;
  const int b = blockIdx.y;
  const int64_t slot = p.state_batch_indices ? int64_t(p.state_batch_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int lane_n = lane % kLanesN, group = lane / kLanesN;
  const int rows = p.nheads * p.headdim;
  const int heads_per_group = p.nheads / p.ngroups;
  const int row_base = (blockIdx.x * kUpdateWarps + warp) * kUpdateRowsPerWarp;

  const T* x = static_cast<const T*>(p.x);
  const T* dt = static_cast<const T*>(p.dt);
  const T* Bm = static_cast<const T*>(p.B);
  const T* Cm = static_cast<const T*>(p.C);
  const T* z = static_cast<const T*>(p.z);
  T* out = static_cast<T*>(p.out);

  for (int it = 0; it < kUpdateRowsPerWarp; it += kRowGroups) {
    const int row = row_base + it + group;
    const bool valid = row < rows;
    const int rr = valid ? row : 0;  // every lane of the warp joins the shuffles, so invalid rows run on row 0 and do not store
    const int h = rr / p.headdim, pi = rr - h * p.headdim, g = h / heads_per_group;
    float* srow = p.state + slot * p.state_slot_stride + int64_t(h) * p.state_head_stride + int64_t(pi) * p.state_p_stride;
    float s[kNPerLane];
#pragma unroll
    for (int j = 0; j < kNPerLane; ++j) s[j] = srow[lane_n + kLanesN * j];
    const float a_head = p.a_per_state ? 0.0f : p.A[int64_t(h) * p.A_head_stride];
    const float bias = p.dt_bias ? p.dt_bias[h] : 0.0f;
    const float d_skip = p.D ? p.D[int64_t(h) * p.D_head_stride + int64_t(pi) * p.D_p_stride] : 0.0f;

    for (int t = 0; t < p.num_steps; ++t) {
      float dtv = to_float(dt[int64_t(b) * p.dt_batch_stride + int64_t(t) * p.dt_token_stride + int64_t(h) * p.dt_head_stride]) + bias;
      if (p.dt_softplus) dtv = softplus_f(dtv);
      if (p.has_dt_limit) dtv = fminf(fmaxf(dtv, p.dt_min), p.dt_max);
      const float xv = to_float(x[int64_t(b) * p.x_batch_stride + int64_t(t) * p.x_token_stride + int64_t(h) * p.x_head_stride + pi]);
      const T* brow = Bm + int64_t(b) * p.B_batch_stride + int64_t(t) * p.B_token_stride + int64_t(g) * p.B_group_stride;
      const T* crow = Cm + int64_t(b) * p.C_batch_stride + int64_t(t) * p.C_token_stride + int64_t(g) * p.C_group_stride;
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kNPerLane; ++j) {
        const int n = lane_n + kLanesN * j;
        const float an = p.a_per_state ? p.A[int64_t(h) * p.A_head_stride + n] : a_head;
        const float da = __expf(dtv * an);
        const float db = dtv * to_float(brow[n]);
        s[j] = __fmaf_rn(db, xv, s[j] * da);
        acc = __fmaf_rn(s[j], to_float(crow[n]), acc);
      }
      acc = group_sum<kLanesN>(acc);
      if (valid && lane_n == 0) {
        float y = __fmaf_rn(d_skip, xv, acc);
        if (z) y *= silu_f(to_float(z[int64_t(b) * p.z_batch_stride + int64_t(t) * p.z_token_stride + int64_t(h) * p.z_head_stride + pi]));
        out[int64_t(b) * p.out_batch_stride + int64_t(t) * p.out_token_stride + int64_t(h) * p.out_head_stride + pi] = from_float<T>(y);
      }
      if (valid && p.intermediate_state_slots) {
        const int64_t dst = p.intermediate_state_slots[int64_t(b) * p.num_steps + t];
        if (dst >= 0 && dst < p.num_slots && dst != p.pad_slot_id) {
          float* drow = p.state + dst * p.state_slot_stride + int64_t(h) * p.state_head_stride + int64_t(pi) * p.state_p_stride;
#pragma unroll
          for (int j = 0; j < kNPerLane; ++j) drow[lane_n + kLanesN * j] = s[j];
        }
      }
    }
    if (valid) {
#pragma unroll
      for (int j = 0; j < kNPerLane; ++j) srow[lane_n + kLanesN * j] = s[j];
    }
  }
}
namespace {


// ---- ported from ayaka state/ssm.cu:129-313 ----
// ---- selective_scan_fwd_kernel ---------------------------------------------------------------------------

constexpr int kScanWarps = 8;
constexpr int kScanPerLane = 4;                         // consecutive tokens one lane runs serially
constexpr int kScanTile = 32 * kScanPerLane;            // tokens scanned in parallel before the state moves to the next tile
constexpr int kScanRow = kScanTile + 4;                 // staged rows: 16-byte aligned, and a stride that does not map onto one bank

/** Split a staging index over a [tokens, width] tile so that the fast axis follows the unit-stride axis of the operand. */
__device__ __forceinline__ void decode_tile(int idx, int tn, int width, bool token_fast, int& tt, int& col) {
  if (token_fast) {
    tt = idx % tn;
    col = idx / tn;
  } else {
    col = idx % width;
    tt = idx / width;
  }
}

/** Channels a CTA owns: the staged tiles of u, delta, z and of B, C must fit the 48 KB of static shared memory. */
__host__ __device__ constexpr int scan_channels(int dstate) { return dstate == 32 ? 8 : 16; }

/** Left inclusive step of the affine scan over (P, Q): the state after a segment is h_out = P h_in + Q, and the segment
 * `prev` (offset lanes to the left) is applied first. */
__device__ __forceinline__ void affine_shfl_up(float& P, float& Q, int off, int lane) {
  const float Pp = __shfl_up_sync(0xffffffffu, P, off), Qp = __shfl_up_sync(0xffffffffu, Q, off);
  if (lane >= off) {
    Q = __fmaf_rn(P, Qp, Q);
    P *= Pp;
  }
}

/**
 * Parallel affine scan over the tokens. Grid (ceil(dim / scan_channels), B). The recurrence h_t = a_t h_{t-1} + b_t of one state
 * element (channel, n) composes associatively as (P, Q): h_out = P h_in + Q, so a CTA scans a tile of kScanTile tokens at once.
 * The tile of u, delta (bias and softplus applied once), z, B and C is staged in shared memory with accesses coalesced along the
 * operand's unit-stride axis. A warp owns one channel at a time and each lane kScanPerLane consecutive tokens: the lane folds
 * its tokens into one (P, Q), a warp scan over the 32 lane segments gives every lane the state at its first token (carrying the
 * state of the previous tile through shared memory), and the lane walks its tokens once more to produce h_t. y_t = sum_n C h
 * accumulates over n in registers in increasing n, so no cross-lane reduction is needed and the order is fixed. Steps beyond
 * the end of the sequence (and padded channels) have delta = 0, i.e. a = 1 and b = 0: the identity, so a tail tile needs no
 * special case and the carried state after the last tile is the final state. The association of the recurrence depends on
 * the position of a token inside its sequence only, so a sequence's result does not depend on the other sequences.
 */
}  // namespace
template <typename T, int kN>
__global__ __launch_bounds__(kScanWarps * 32, 2) void selective_scan_fwd_kernel(const __grid_constant__ SelectiveScanParams p) {
  constexpr int kCh = scan_channels(kN);
  __shared__ __align__(16) float u_s[kCh][kScanRow];  // u, then y in place (a lane reads and writes only its own tokens)
  __shared__ __align__(16) float d_s[kCh][kScanRow];
  __shared__ __align__(16) float z_s[kCh][kScanRow];
  __shared__ __align__(16) float b_s[kN][kScanRow];
  __shared__ __align__(16) float c_s[kN][kScanRow];
  __shared__ float a_s[kCh][kN], h_s[kCh][kN];
  const int b = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int64_t slot = p.cache_indices ? int64_t(p.cache_indices[b]) : int64_t(b);
  if (slot < 0 || slot >= p.num_slots || slot == p.pad_slot_id) return;  // uniform per CTA
  const int c0 = blockIdx.x * kCh;
  const int group = p.ngroups == 1 ? 0 : c0 / (p.dim / p.ngroups);
  const int seq_start = p.query_start_loc[b], n = p.query_start_loc[b + 1] - seq_start;
  const bool has_init = p.has_initial_state == nullptr || p.has_initial_state[b];

  const T* u = static_cast<const T*>(p.u);
  const T* delta = static_cast<const T*>(p.delta);
  const T* Bm = static_cast<const T*>(p.B) + int64_t(group) * p.B_group_stride;
  const T* Cm = static_cast<const T*>(p.C) + int64_t(group) * p.C_group_stride;
  const T* z = static_cast<const T*>(p.z);
  T* out = static_cast<T*>(p.out);
  float* state = p.ssm_states + slot * p.ssm_state_slot_stride;
  for (int i = tid; i < kCh * kN; i += kScanWarps * 32) {
    const int cl = i / kN, ln = i % kN, ch = c0 + cl;
    a_s[cl][ln] = ch < p.dim ? p.A[int64_t(ch) * p.A_dim_stride + ln] : 0.0f;
    h_s[cl][ln] = (ch < p.dim && has_init) ? state[int64_t(ch) * p.ssm_state_dim_stride + ln] : 0.0f;
  }

  for (int t0 = 0; t0 < n; t0 += kScanTile) {
    const int tn = min(kScanTile, n - t0);
    __syncthreads();  // the previous tile's y and its B/C are done with (and a_s/h_s are written)
    for (int idx = tid; idx < kCh * kScanTile; idx += kScanWarps * 32) {
      int tt, l;
      decode_tile(idx, kScanTile, kCh, p.u_token_stride == 1, tt, l);
      const int c = c0 + l;
      float uv = 0.0f;
      if (c < p.dim && tt < tn) uv = to_float(u[int64_t(c) * p.u_dim_stride + (int64_t(seq_start) + t0 + tt) * p.u_token_stride]);
      u_s[l][tt] = uv;
    }
    for (int idx = tid; idx < kCh * kScanTile; idx += kScanWarps * 32) {
      int tt, l;
      decode_tile(idx, kScanTile, kCh, p.delta_token_stride == 1, tt, l);
      const int c = c0 + l;
      float dv = 0.0f;  // 0 is the identity step
      if (c < p.dim && tt < tn) {
        dv = to_float(delta[int64_t(c) * p.delta_dim_stride + (int64_t(seq_start) + t0 + tt) * p.delta_token_stride]);
        if (p.delta_bias) dv += p.delta_bias[c];
        if (p.delta_softplus) dv = softplus_f(dv);
      }
      d_s[l][tt] = dv;
    }
    if (z) {
      for (int idx = tid; idx < kCh * kScanTile; idx += kScanWarps * 32) {
        int tt, l;
        decode_tile(idx, kScanTile, kCh, p.z_token_stride == 1, tt, l);
        const int c = c0 + l;
        z_s[l][tt] = (c < p.dim && tt < tn) ? to_float(z[int64_t(c) * p.z_dim_stride + (int64_t(seq_start) + t0 + tt) * p.z_token_stride]) : 0.0f;
      }
    }
    for (int idx = tid; idx < kN * kScanTile; idx += kScanWarps * 32) {
      int tt, l;
      decode_tile(idx, kScanTile, kN, p.B_token_stride == 1, tt, l);
      b_s[l][tt] = tt < tn ? to_float(Bm[int64_t(l) * p.B_state_stride + (int64_t(seq_start) + t0 + tt) * p.B_token_stride]) : 0.0f;
    }
    for (int idx = tid; idx < kN * kScanTile; idx += kScanWarps * 32) {
      int tt, l;
      decode_tile(idx, kScanTile, kN, p.C_token_stride == 1, tt, l);
      c_s[l][tt] = tt < tn ? to_float(Cm[int64_t(l) * p.C_state_stride + (int64_t(seq_start) + t0 + tt) * p.C_token_stride]) : 0.0f;
    }
    __syncthreads();

    for (int cl = warp; cl < kCh; cl += kScanWarps) {
      const int ch = c0 + cl;
      if (ch >= p.dim) continue;  // uniform per warp
      const float4 d4 = *reinterpret_cast<const float4*>(&d_s[cl][lane * kScanPerLane]);
      const float4 u4 = *reinterpret_cast<const float4*>(&u_s[cl][lane * kScanPerLane]);
      const float dd[4] = {d4.x, d4.y, d4.z, d4.w}, uu[4] = {u4.x, u4.y, u4.z, u4.w};
      float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int ln = 0; ln < kN; ++ln) {
        const float a_n = a_s[cl][ln], carry = h_s[cl][ln];
        const float4 b4 = *reinterpret_cast<const float4*>(&b_s[ln][lane * kScanPerLane]);
        const float4 c4 = *reinterpret_cast<const float4*>(&c_s[ln][lane * kScanPerLane]);
        const float bv[4] = {b4.x, b4.y, b4.z, b4.w}, cv[4] = {c4.x, c4.y, c4.z, c4.w};
        float a[4], bb[4];
#pragma unroll
        for (int i = 0; i < kScanPerLane; ++i) {
          a[i] = __expf(dd[i] * a_n);
          bb[i] = dd[i] * uu[i] * bv[i];
        }
        float P = a[0], Q = bb[0];  // the lane's tokens folded into one segment
#pragma unroll
        for (int i = 1; i < kScanPerLane; ++i) {
          Q = __fmaf_rn(a[i], Q, bb[i]);
          P *= a[i];
        }
        affine_shfl_up(P, Q, 1, lane);
        affine_shfl_up(P, Q, 2, lane);
        affine_shfl_up(P, Q, 4, lane);
        affine_shfl_up(P, Q, 8, lane);
        affine_shfl_up(P, Q, 16, lane);
        // (P, Q) is now the inclusive segment up to this lane; the state entering the lane is the exclusive one.
        float Pe = __shfl_up_sync(0xffffffffu, P, 1), Qe = __shfl_up_sync(0xffffffffu, Q, 1);
        const float P_tile = __shfl_sync(0xffffffffu, P, 31), Q_tile = __shfl_sync(0xffffffffu, Q, 31);
        if (lane == 0) {
          Pe = 1.0f;
          Qe = 0.0f;
        }
        float h = __fmaf_rn(Pe, carry, Qe);
#pragma unroll
        for (int i = 0; i < kScanPerLane; ++i) {
          h = __fmaf_rn(a[i], h, bb[i]);
          y[i] = __fmaf_rn(h, cv[i], y[i]);
        }
        __syncwarp();  // every lane has read the carry of this (channel, n) before lane 0 replaces it
        if (lane == 0) h_s[cl][ln] = __fmaf_rn(P_tile, carry, Q_tile);
      }
      const float d_skip = p.D ? p.D[ch] : 0.0f;
#pragma unroll
      for (int i = 0; i < kScanPerLane; ++i) {
        y[i] = __fmaf_rn(d_skip, uu[i], y[i]);
        if (z) y[i] *= silu_f(z_s[cl][lane * kScanPerLane + i]);
      }
      *reinterpret_cast<float4*>(&u_s[cl][lane * kScanPerLane]) = make_float4(y[0], y[1], y[2], y[3]);
    }
    __syncthreads();
    for (int idx = tid; idx < kCh * kScanTile; idx += kScanWarps * 32) {
      int tt, l;
      decode_tile(idx, kScanTile, kCh, p.out_token_stride == 1, tt, l);
      const int c = c0 + l;
      if (c < p.dim && tt < tn) out[int64_t(c) * p.out_dim_stride + (int64_t(seq_start) + t0 + tt) * p.out_token_stride] = from_float<T>(u_s[l][tt]);
    }
  }
  __syncthreads();
  for (int i = tid; i < kCh * kN; i += kScanWarps * 32) {
    const int cl = i / kN, ln = i % kN, ch = c0 + cl;
    if (ch < p.dim) state[int64_t(ch) * p.ssm_state_dim_stride + ln] = h_s[cl][ln];
  }
}
namespace {


// ---- ported from ayaka state/ssm.cu:331-574 ----
// ---- Mamba-2 SSD: the five steps of the chunked scan ------------------------------------------------------
//
// The matmul steps use the same 64 x 64 output tile as lightning_attention_prefill: 256 threads in a 16 x 16 grid with a 4 x 4
// register tile each, operands staged k-major in 32-wide slices of the reduction and multiplied by smem_gemm_kmajor in increasing
// k, so every result is an fp32 FMA chain of fixed order. Every exponent is a difference of two cumulative sums with the later
// one first, hence <= 0 and nothing overflows; entries a causal mask removes are zeroed before the exp.

constexpr int kSsdThreads = 256;
constexpr int kSsdTile = 64;
constexpr int kSsdSlice = 32;
constexpr int kSsdLd = kSsdTile + 4;  // row stride of the staged tiles: 16-byte rows, spread banks

/** Step 1. Grid NC * H, one block of 256 threads per (chunk, head): thread l holds dt'[l], an inclusive warp scan plus the sequential sum of
 * the (at most 7) lower warp totals gives dA_cumsum[l]. Threads beyond the chunk's length hold dt' = 0, so the sum stays at its last value. */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads) void ssd_chunk_cumsum_kernel(const __grid_constant__ SSDParams p) {
  __shared__ float warp_total[kSsdThreads / 32];
  const int c = blockIdx.x / p.nheads, h = blockIdx.x % p.nheads;
  const int l = threadIdx.x, lane = l & 31, warp = l >> 5;
  const int s0 = p.cu_chunk_seqlens[c], len = p.cu_chunk_seqlens[c + 1] - s0;
  float d = 0.0f;
  if (l < len) {
    d = to_float(static_cast<const T*>(p.dt)[int64_t(s0 + l) * p.dt_token_stride + int64_t(h) * p.dt_head_stride]);
    if (p.dt_bias) d += p.dt_bias[h];
    if (p.dt_softplus) d = softplus_f(d);
    if (p.has_dt_limit) d = fminf(fmaxf(d, p.dt_min), p.dt_max);
  }
  float v = d * p.A[h];
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float u = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += u;
  }
  if (lane == 31) warp_total[warp] = v;
  __syncthreads();
  for (int w = 0; w < warp; ++w) v += warp_total[w];
  if (l < p.chunk_size) {
    const int64_t at = (int64_t(h) * p.num_chunks + c) * p.chunk_size + l;
    p.dt_proc[at] = d;
    p.dA_cumsum[at] = v;
  }
}
namespace {

/** Step 2. Grid NC * G * tiles, tiles = nt (nt + 1) / 2 lower-triangular 64 x 64 tiles of the L x L square (nt = L / 64). A tile is
 * C_tile . B_tile^T over the N features in slices of 32; a tile past the chunk's length is skipped. */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads) void ssd_bmm_chunk_kernel(const __grid_constant__ SSDParams p) {
  __shared__ __align__(16) float cs[kSsdSlice * kSsdLd], bs[kSsdSlice * kSsdLd];
  const int nt = p.chunk_size / kSsdTile, tiles = nt * (nt + 1) / 2;
  const int tile = blockIdx.x % tiles, cg = blockIdx.x / tiles;
  const int chunk = cg / p.ngroups, g = cg % p.ngroups;
  int ti = 0;
  while ((ti + 1) * (ti + 2) / 2 <= tile) ++ti;
  const int tj = tile - ti * (ti + 1) / 2;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  const int i0 = ti * kSsdTile, j0 = tj * kSsdTile;
  if (i0 >= len) return;  // uniform per CTA; j0 <= i0
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const T* C = static_cast<const T*>(p.C) + int64_t(g) * p.C_group_stride;
  const T* B = static_cast<const T*>(p.B) + int64_t(g) * p.B_group_stride;
  float acc[4][4] = {};
  for (int n0 = 0; n0 < p.dstate; n0 += kSsdSlice) {
    __syncthreads();
    for (int idx = tid; idx < kSsdTile * kSsdSlice; idx += kSsdThreads) {
      const int kk = idx % kSsdSlice, i = idx / kSsdSlice;
      float cv = 0.0f, bv = 0.0f;
      if (n0 + kk < p.dstate) {
        if (i0 + i < len) cv = to_float(C[int64_t(s0 + i0 + i) * p.C_token_stride + n0 + kk]);
        if (j0 + i < len) bv = to_float(B[int64_t(s0 + j0 + i) * p.B_token_stride + n0 + kk]);
      }
      cs[kk * kSsdLd + i] = cv;
      bs[kk * kSsdLd + i] = bv;
    }
    __syncthreads();
    smem_gemm_kmajor<4, 4>(acc, cs, kSsdLd, ty * 4, bs, kSsdLd, tx * 4, kSsdSlice);
  }
  float* out = p.CB + (int64_t(cg) * p.chunk_size) * p.chunk_size;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = i0 + ty * 4 + r;
    if (i >= len) continue;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int j = j0 + tx * 4 + cc;
      if (j < len) out[int64_t(i) * p.chunk_size + j] = acc[r][cc];
    }
  }
}
namespace {

/** Step 2, tensor cores (bf16 / fp16 activations, SM80+). Same tile pair and grid as the fp32 body: C and B are staged
 * as 16-bit tiles over slices of 64 N features and multiplied by mma.sync m16n8k16. Warp w owns the 16 rows 16 (w & 3)
 * and the 32 columns 32 (w >> 2); the fp32 accumulator is written to CB. */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads, 2) void ssd_bmm_chunk_tc_kernel(const __grid_constant__ SSDParams p) {
  constexpr int kLd = mma::tc_ld(kSsdTile);
  __shared__ __align__(16) T cs[kSsdTile * kLd], bs[kSsdTile * kLd];
  const int nt = p.chunk_size / kSsdTile, tiles = nt * (nt + 1) / 2;
  const int tile = blockIdx.x % tiles, cg = blockIdx.x / tiles;
  const int chunk = cg / p.ngroups, g = cg % p.ngroups;
  int ti = 0;
  while ((ti + 1) * (ti + 2) / 2 <= tile) ++ti;
  const int tj = tile - ti * (ti + 1) / 2;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  const int i0 = ti * kSsdTile, j0 = tj * kSsdTile;
  if (i0 >= len) return;  // uniform per CTA; j0 <= i0
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const T* C = static_cast<const T*>(p.C) + int64_t(g) * p.C_group_stride;
  const T* B = static_cast<const T*>(p.B) + int64_t(g) * p.B_group_stride;
  const bool vec_c = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(C) % 16 == 0 && p.C_token_stride % 8 == 0;
  const bool vec_b = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(B) % 16 == 0 && p.B_token_stride % 8 == 0;
  const int I = wid & 3, half = wid >> 2;
  float acc[4][4] = {};
  for (int n0 = 0; n0 < p.dstate; n0 += kSsdTile) {
    __syncthreads();
    const int cols = min(kSsdTile, p.dstate - n0);
    mma::tc_stage_input<T, kSsdThreads>(cs, kLd, C + n0, p.C_token_stride, s0 + i0, len - i0, kSsdTile, vec_c, cols);
    mma::tc_stage_input<T, kSsdThreads>(bs, kLd, B + n0, p.B_token_stride, s0 + j0, len - j0, kSsdTile, vec_b, cols);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kSsdTile / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, cs, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        mma::frag_b2(b, bs, kLd, 32 * half + 16 * np, 16 * kk);
        mma::Mma16x8x16<T>::run(acc[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
  }
  float* out = p.CB + (int64_t(cg) * p.chunk_size) * p.chunk_size;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int i = i0 + 16 * I + gq + 8 * rr;
    if (i >= len) continue;
#pragma unroll
    for (int ntl = 0; ntl < 4; ++ntl) {
      const int j = j0 + 32 * half + 8 * ntl + 2 * t;
      const float v0 = j < len ? acc[ntl][2 * rr] : 0.0f;
      const float v1 = j + 1 < len ? acc[ntl][2 * rr + 1] : 0.0f;
      *reinterpret_cast<float2*>(out + int64_t(i) * p.chunk_size + j) = make_float2(v0, v1);
    }
  }
}

namespace {

/** Step 3. Grid NC * H * (P tiles) * (N tiles): a 64 x 64 tile of the P x N chunk state, accumulated over the chunk's tokens in slices
 * of 32 from x'[l, p] = dt'[l] exp(dA_last - dA[l]) x[l, p] and B[l, n]. */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads) void ssd_chunk_state_kernel(const __grid_constant__ SSDParams p) {
  __shared__ __align__(16) float xs[kSsdSlice * kSsdLd], bs[kSsdSlice * kSsdLd];
  const int pt = (p.headdim + kSsdTile - 1) / kSsdTile, nt = (p.dstate + kSsdTile - 1) / kSsdTile, tiles = pt * nt;
  const int tile = blockIdx.x % tiles, ch = blockIdx.x / tiles;
  const int chunk = ch / p.nheads, h = ch % p.nheads, g = h / (p.nheads / p.ngroups);
  const int p0 = (tile / nt) * kSsdTile, n0 = (tile % nt) * kSsdTile;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const T* x = static_cast<const T*>(p.x) + int64_t(h) * p.x_head_stride;
  const T* B = static_cast<const T*>(p.B) + int64_t(g) * p.B_group_stride;
  const float* dA = p.dA_cumsum + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* dtv = p.dt_proc + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float dA_last = len > 0 ? dA[len - 1] : 0.0f;
  float acc[4][4] = {};
  for (int l0 = 0; l0 < len; l0 += kSsdSlice) {
    __syncthreads();
    for (int idx = tid; idx < kSsdSlice * kSsdTile; idx += kSsdThreads) {
      const int col = idx % kSsdTile, l = idx / kSsdTile;
      float xv = 0.0f, bv = 0.0f;
      if (l0 + l < len) {
        const int64_t row = int64_t(s0 + l0 + l);
        if (p0 + col < p.headdim) xv = dtv[l0 + l] * __expf(dA_last - dA[l0 + l]) * to_float(x[row * p.x_token_stride + p0 + col]);
        if (n0 + col < p.dstate) bv = to_float(B[row * p.B_token_stride + n0 + col]);
      }
      xs[l * kSsdLd + col] = xv;
      bs[l * kSsdLd + col] = bv;
    }
    __syncthreads();
    smem_gemm_kmajor<4, 4>(acc, xs, kSsdLd, ty * 4, bs, kSsdLd, tx * 4, kSsdSlice);
  }
  float* out = p.chunk_states + ((int64_t(chunk) * p.nheads + h) * p.headdim) * p.dstate;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int pp = p0 + ty * 4 + r;
    if (pp >= p.headdim) continue;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int n = n0 + tx * 4 + cc;
      if (n < p.dstate) out[int64_t(pp) * p.dstate + n] = acc[r][cc];
    }
  }
}
namespace {

/** Step 3, tensor cores (bf16 / fp16 activations, SM80+). Grid and tiles as the fp32 body. x' = dt' exp(dA_last - dA) x
 * is scaled in fp32 and split into a 16-bit hi/lo pair, so the P x N chunk state keeps fp32-level accuracy through two
 * mma per product (the state is the input of the fp32 state passing and is compared at fp32 tolerance). Warp w owns the
 * 16 P rows 16 (w & 3) and the 32 N columns 32 (w >> 2). */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads, 1) void ssd_chunk_state_tc_kernel(const __grid_constant__ SSDParams p) {
  constexpr int kLd = mma::tc_ld(kSsdTile);
  __shared__ __align__(16) T xs_hi[kSsdTile * kLd], xs_lo[kSsdTile * kLd], bs[kSsdTile * kLd];
  const int pt = (p.headdim + kSsdTile - 1) / kSsdTile, nt = (p.dstate + kSsdTile - 1) / kSsdTile, tiles = pt * nt;
  const int tile = blockIdx.x % tiles, ch = blockIdx.x / tiles;
  const int chunk = ch / p.nheads, h = ch % p.nheads, g = h / (p.nheads / p.ngroups);
  const int p0 = (tile / nt) * kSsdTile, n0 = (tile % nt) * kSsdTile;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const T* x = static_cast<const T*>(p.x) + int64_t(h) * p.x_head_stride;
  const T* B = static_cast<const T*>(p.B) + int64_t(g) * p.B_group_stride;
  const float* dA = p.dA_cumsum + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* dtv = p.dt_proc + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float dA_last = len > 0 ? dA[len - 1] : 0.0f;
  const int pcols = p.headdim - p0, ncols = p.dstate - n0;
  const bool vec_b = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(B) % 16 == 0 && p.B_token_stride % 8 == 0;
  const int I = wid & 3, half = wid >> 2;
  float acc[4][4] = {};
  for (int l0 = 0; l0 < len; l0 += kSsdTile) {
    __syncthreads();
    const int rows = min(kSsdTile, len - l0);
    for (int idx = tid; idx < kSsdTile * (kSsdTile / 8); idx += kSsdThreads) {
      const int r = idx / (kSsdTile / 8), c = (idx % (kSsdTile / 8)) * 8;
      const bool row_ok = l0 + r < len;
      const float w = row_ok ? dtv[l0 + r] * __expf(dA_last - dA[l0 + r]) : 0.0f;
      float xv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        xv[i] = (row_ok && c + i < pcols) ? w * to_float(x[int64_t(s0 + l0 + r) * p.x_token_stride + p0 + c + i]) : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        T hi, lo;
        mma::st16(hi, xv[i]);
        mma::st16(lo, xv[i] - mma::ld16(hi));
        xs_hi[r * kLd + c + i] = hi;
        xs_lo[r * kLd + c + i] = lo;
      }
    }
    mma::tc_stage_input<T, kSsdThreads>(bs, kLd, B + n0, p.B_token_stride, s0 + l0, rows, kSsdTile, vec_b, ncols);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kSsdTile / 16; ++kk) {
      uint32_t ahi[4], alo[4];
      mma::frag_a_trans(ahi, xs_hi, kLd, 16 * I, 16 * kk);
      mma::frag_a_trans(alo, xs_lo, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        mma::frag_b2_trans(b, bs, kLd, 16 * kk, 32 * half + 16 * np);
        mma::Mma16x8x16<T>::run(acc[2 * np], ahi, b[0], b[1]);
        mma::Mma16x8x16<T>::run(acc[2 * np + 1], ahi, b[2], b[3]);
        mma::Mma16x8x16<T>::run(acc[2 * np], alo, b[0], b[1]);
        mma::Mma16x8x16<T>::run(acc[2 * np + 1], alo, b[2], b[3]);
      }
    }
  }
  float* out = p.chunk_states + ((int64_t(chunk) * p.nheads + h) * p.headdim) * p.dstate;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int pp = p0 + 16 * I + gq + 8 * rr;
    if (pp >= p.headdim) continue;
    float* row = out + int64_t(pp) * p.dstate;
#pragma unroll
    for (int ntl = 0; ntl < 4; ++ntl) {
      const int n = n0 + 32 * half + 8 * ntl + 2 * t;
      if (n < p.dstate) row[n] = acc[ntl][2 * rr];
      if (n + 1 < p.dstate) row[n + 1] = acc[ntl][2 * rr + 1];
    }
  }
}

namespace {

/** Step 4. Grid (ceil(H P N / 256), B): a thread owns one element (h, p, n) of the state of one sequence and walks the sequence's
 * chunks in order, s <- exp(dA_last) s + states[c], writing the state entering each chunk. The state starts from the slot (zero without
 * has_initial_state or with an invalid slot), goes back to the slot after the last chunk and, at multiples of the snapshot
 * interval, to the snapshot slot. */
}  // namespace
__global__ __launch_bounds__(kSsdThreads) void ssd_state_passing_kernel(const __grid_constant__ SSDParams p) {
  const int64_t per_seq = int64_t(p.nheads) * p.headdim * p.dstate;
  const int64_t e = int64_t(blockIdx.x) * kSsdThreads + threadIdx.x;
  if (e >= per_seq) return;
  const int b = blockIdx.y;
  const int h = int(e / (int64_t(p.headdim) * p.dstate));
  const int64_t slot = p.state_indices ? int64_t(p.state_indices[b]) : int64_t(b);
  const bool live = slot >= 0 && slot < p.num_slots && slot != p.pad_slot_id;
  const bool init = p.has_initial_state == nullptr || p.has_initial_state[b];
  float s = (live && init) ? p.ssm_states[slot * p.ssm_state_slot_stride + e] : 0.0f;
  const int c0 = p.cu_chunks[b], c1 = p.cu_chunks[b + 1], seq0 = p.cu_seqlens[b];
  for (int c = c0; c < c1; ++c) {
    p.passed_states[int64_t(c) * per_seq + e] = s;
    const int len = p.cu_chunk_seqlens[c + 1] - p.cu_chunk_seqlens[c];
    const float decay = len > 0 ? __expf(p.dA_cumsum[(int64_t(h) * p.num_chunks + c) * p.chunk_size + len - 1]) : 1.0f;
    s = __fmaf_rn(decay, s, p.chunk_states[int64_t(c) * per_seq + e]);
    const int done = p.cu_chunk_seqlens[c + 1] - seq0;
    if (p.snapshot_interval > 0 && done % p.snapshot_interval == 0) {
      const int j = done / p.snapshot_interval - 1;
      if (j < p.max_snapshots) {
        const int64_t dst = p.snapshot_slots[int64_t(b) * p.max_snapshots + j];
        if (dst >= 0 && dst < p.num_slots) p.ssm_states[dst * p.ssm_state_slot_stride + e] = s;
      }
    }
  }
  if (live) p.ssm_states[slot * p.ssm_state_slot_stride + e] = s;
}
namespace {

/** Step 5, tensor cores (bf16 / fp16 activations, SM80+). Grid and tiles as the fp32 body. The part from before the chunk
 * is C . passed^T over the N features in slices of 64 (passed is fp32 workspace, staged as 16-bit); the part inside the
 * chunk is M x with M[l, s] = CB[l, s] exp(dA[l] - dA[s]) dt'[s] (s <= l) staged as a 16-bit tile and x staged 16-bit
 * [token][P]. Both accumulate the same fp32 registers; warp w owns the 16 rows 16 (w & 3) and 32 columns 32 (w >> 2). */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads, 1) void ssd_chunk_scan_tc_kernel(const __grid_constant__ SSDParams p) {
  constexpr int kLd = mma::tc_ld(kSsdTile);
  __shared__ __align__(16) T as[kSsdTile * kLd], bs[kSsdTile * kLd];
  const int lt = p.chunk_size / kSsdTile, pt = (p.headdim + kSsdTile - 1) / kSsdTile, tiles = lt * pt;
  const int tile = blockIdx.x % tiles, ch = blockIdx.x / tiles;
  const int chunk = ch / p.nheads, h = ch % p.nheads, g = h / (p.nheads / p.ngroups);
  const int l0 = (tile / pt) * kSsdTile, p0 = (tile % pt) * kSsdTile;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  if (l0 >= len) return;  // uniform per CTA
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const T* x = static_cast<const T*>(p.x) + int64_t(h) * p.x_head_stride;
  const T* C = static_cast<const T*>(p.C) + int64_t(g) * p.C_group_stride;
  const float* dA = p.dA_cumsum + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* dtv = p.dt_proc + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* passed = p.passed_states + (int64_t(chunk) * p.nheads + h) * p.headdim * p.dstate;
  const float* CB = p.CB + (int64_t(chunk) * p.ngroups + g) * p.chunk_size * p.chunk_size;
  const int pcols = min(kSsdTile, p.headdim - p0), rows_l = len - l0;
  const bool vec_c = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(C) % 16 == 0 && p.C_token_stride % 8 == 0;
  const bool vec_x = sizeof(T) == 2 && reinterpret_cast<uintptr_t>(x) % 16 == 0 && p.x_token_stride % 8 == 0;
  const bool vec_p = p.dstate % 4 == 0 && reinterpret_cast<uintptr_t>(passed) % 16 == 0;
  const int I = wid & 3, half = wid >> 2;
  float acc[4][4] = {};
  for (int n0 = 0; n0 < p.dstate; n0 += kSsdTile) {  // from before the chunk
    __syncthreads();
    const int cols = min(kSsdTile, p.dstate - n0);
    mma::tc_stage_input<T, kSsdThreads>(as, kLd, C + n0, p.C_token_stride, s0 + l0, rows_l, kSsdTile, vec_c, cols);
    mma::tc_stage_f32<T, kSsdThreads>(bs, kLd, passed + int64_t(p0) * p.dstate + n0, p.dstate, pcols, kSsdTile,
                                      kSsdTile, cols, vec_p);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kSsdTile / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, as, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        mma::frag_b2(b, bs, kLd, 32 * half + 16 * np, 16 * kk);
        mma::Mma16x8x16<T>::run(acc[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
  }
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int l = l0 + 16 * I + gq + 8 * rr;
    const float w = l < len ? __expf(dA[l]) : 0.0f;
#pragma unroll
    for (int ntl = 0; ntl < 4; ++ntl) {
      acc[ntl][2 * rr] *= w;
      acc[ntl][2 * rr + 1] *= w;
    }
  }
  const int s_end = min(len, l0 + kSsdTile);
  for (int sb = 0; sb < s_end; sb += kSsdTile) {  // inside the chunk
    __syncthreads();
    for (int idx = tid; idx < kSsdTile * (kSsdTile / 8); idx += kSsdThreads) {
      const int r = idx / (kSsdTile / 8), c = (idx % (kSsdTile / 8)) * 8;
      const int l = l0 + r;
      float mv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int s = sb + c + i;
        mv[i] = (l < len && s <= l) ? CB[int64_t(l) * p.chunk_size + s] * __expf(dA[l] - dA[s]) * dtv[s] : 0.0f;
      }
      uint4 out;
      out.x = mma::Bits16<T>::pack(mv[0], mv[1]);
      out.y = mma::Bits16<T>::pack(mv[2], mv[3]);
      out.z = mma::Bits16<T>::pack(mv[4], mv[5]);
      out.w = mma::Bits16<T>::pack(mv[6], mv[7]);
      *reinterpret_cast<uint4*>(as + r * kLd + c) = out;
    }
    mma::tc_stage_input<T, kSsdThreads>(bs, kLd, x + p0, p.x_token_stride, s0 + sb, min(kSsdTile, len - sb), kSsdTile,
                                        vec_x, pcols);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kSsdTile / 16; ++kk) {
      uint32_t a[4];
      mma::frag_a(a, as, kLd, 16 * I, 16 * kk);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b[4];
        mma::frag_b2_trans(b, bs, kLd, 16 * kk, 32 * half + 16 * np);
        mma::Mma16x8x16<T>::run(acc[2 * np], a, b[0], b[1]);
        mma::Mma16x8x16<T>::run(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
  }
  const T* z = static_cast<const T*>(p.z);
  T* out = static_cast<T*>(p.out);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int l = l0 + 16 * I + gq + 8 * rr;
    if (l >= len) continue;
    const int64_t tok = int64_t(s0 + l);
#pragma unroll
    for (int ntl = 0; ntl < 4; ++ntl) {
      const int pp = p0 + 32 * half + 8 * ntl + 2 * t;
      if (pp >= p.headdim) continue;
      float y0 = acc[ntl][2 * rr], y1 = acc[ntl][2 * rr + 1];
      if (p.D) {
        y0 = __fmaf_rn(p.D[int64_t(h) * p.D_head_stride + int64_t(pp) * p.D_p_stride],
                       to_float(x[tok * p.x_token_stride + pp]), y0);
        if (pp + 1 < p.headdim)
          y1 = __fmaf_rn(p.D[int64_t(h) * p.D_head_stride + int64_t(pp + 1) * p.D_p_stride],
                         to_float(x[tok * p.x_token_stride + pp + 1]), y1);
      }
      if (z) {
        y0 *= silu_f(to_float(z[tok * p.z_token_stride + int64_t(h) * p.z_head_stride + pp]));
        if (pp + 1 < p.headdim)
          y1 *= silu_f(to_float(z[tok * p.z_token_stride + int64_t(h) * p.z_head_stride + pp + 1]));
      }
      out[tok * p.out_token_stride + int64_t(h) * p.out_head_stride + pp] = from_float<T>(y0);
      if (pp + 1 < p.headdim) out[tok * p.out_token_stride + int64_t(h) * p.out_head_stride + pp + 1] = from_float<T>(y1);
    }
  }
}

namespace {

/** Step 5. Grid NC * H * (L / 64) * (P tiles): a 64 x 64 tile of y for rows l0.. of a chunk. First the part from before the chunk,
 * C . passed^T over the N features, scaled by exp(dA[l]); then the part inside the chunk, M x over the tokens s <= l, with
 * M[l, s] = CB[l, s] exp(dA[l] - dA[s]) dt'[s] staged transposed and the tokens beyond the tile's last row skipped; then D x and the
 * silu(z) gate. */
}  // namespace
template <typename T>
__global__ __launch_bounds__(kSsdThreads, 1) void ssd_chunk_scan_kernel(const __grid_constant__ SSDParams p) {
  __shared__ __align__(16) float as[kSsdSlice * kSsdLd], bs[kSsdSlice * kSsdLd];
  const int lt = p.chunk_size / kSsdTile, pt = (p.headdim + kSsdTile - 1) / kSsdTile, tiles = lt * pt;
  const int tile = blockIdx.x % tiles, ch = blockIdx.x / tiles;
  const int chunk = ch / p.nheads, h = ch % p.nheads, g = h / (p.nheads / p.ngroups);
  const int l0 = (tile / pt) * kSsdTile, p0 = (tile % pt) * kSsdTile;
  const int s0 = p.cu_chunk_seqlens[chunk], len = p.cu_chunk_seqlens[chunk + 1] - s0;
  if (l0 >= len) return;  // uniform per CTA
  const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
  const T* x = static_cast<const T*>(p.x) + int64_t(h) * p.x_head_stride;
  const T* C = static_cast<const T*>(p.C) + int64_t(g) * p.C_group_stride;
  const float* dA = p.dA_cumsum + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* dtv = p.dt_proc + (int64_t(h) * p.num_chunks + chunk) * p.chunk_size;
  const float* passed = p.passed_states + (int64_t(chunk) * p.nheads + h) * p.headdim * p.dstate;
  const float* CB = p.CB + (int64_t(chunk) * p.ngroups + g) * p.chunk_size * p.chunk_size;
  float acc[4][4] = {};
  for (int n0 = 0; n0 < p.dstate; n0 += kSsdSlice) {  // from before the chunk
    __syncthreads();
    for (int idx = tid; idx < kSsdTile * kSsdSlice; idx += kSsdThreads) {
      const int kk = idx % kSsdSlice, i = idx / kSsdSlice;
      float cv = 0.0f, pv = 0.0f;
      if (n0 + kk < p.dstate) {
        if (l0 + i < len) cv = to_float(C[int64_t(s0 + l0 + i) * p.C_token_stride + n0 + kk]);
        if (p0 + i < p.headdim) pv = passed[int64_t(p0 + i) * p.dstate + n0 + kk];
      }
      as[kk * kSsdLd + i] = cv;
      bs[kk * kSsdLd + i] = pv;
    }
    __syncthreads();
    smem_gemm_kmajor<4, 4>(acc, as, kSsdLd, ty * 4, bs, kSsdLd, tx * 4, kSsdSlice);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int l = l0 + ty * 4 + r;
    const float w = l < len ? __expf(dA[l]) : 0.0f;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) acc[r][cc] *= w;
  }
  const int s_end = min(len, l0 + kSsdTile);
  for (int sb = 0; sb < s_end; sb += kSsdSlice) {  // inside the chunk
    __syncthreads();
    for (int idx = tid; idx < kSsdTile * kSsdSlice; idx += kSsdThreads) {
      const int ss = idx % kSsdSlice, ll = idx / kSsdSlice;
      const int s = sb + ss, l = l0 + ll;
      as[ss * kSsdLd + ll] = (s <= l && l < len) ? CB[int64_t(l) * p.chunk_size + s] * __expf(dA[l] - dA[s]) * dtv[s] : 0.0f;
    }
    for (int idx = tid; idx < kSsdSlice * kSsdTile; idx += kSsdThreads) {
      const int col = idx % kSsdTile, ss = idx / kSsdTile;
      bs[ss * kSsdLd + col] = (sb + ss < len && p0 + col < p.headdim) ? to_float(x[int64_t(s0 + sb + ss) * p.x_token_stride + p0 + col]) : 0.0f;
    }
    __syncthreads();
    smem_gemm_kmajor<4, 4>(acc, as, kSsdLd, ty * 4, bs, kSsdLd, tx * 4, kSsdSlice);
  }
  const T* z = static_cast<const T*>(p.z);
  T* out = static_cast<T*>(p.out);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int l = l0 + ty * 4 + r;
    if (l >= len) continue;
    const int64_t tok = int64_t(s0 + l);
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int pp = p0 + tx * 4 + cc;
      if (pp >= p.headdim) continue;
      float y = acc[r][cc];
      if (p.D) y = __fmaf_rn(p.D[int64_t(h) * p.D_head_stride + int64_t(pp) * p.D_p_stride], to_float(x[tok * p.x_token_stride + pp]), y);
      if (z) y *= silu_f(to_float(z[tok * p.z_token_stride + int64_t(h) * p.z_head_stride + pp]));
      out[tok * p.out_token_stride + int64_t(h) * p.out_head_stride + pp] = from_float<T>(y);
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

Status check_backend_dtype(const char* fn, const LaunchContext& ctx, DType dtype) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", fn);
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: activations must be fp32, fp16 or bf16", fn);
  }
  return Status::kSuccess;
}

/// f(T{}) with T the element type of dtype (validated before).
template <typename F>
void by_dtype(DType dtype, F&& f) {
  if (dtype == DType::kFloat32) f(float{});
  else if (dtype == DType::kFloat16) f(__half{});
  else f(__nv_bfloat16{});
}

template <typename T, int kLanesN, int kNPerLane>
void launch_update_impl(const SSMStateUpdateParams& p, cudaStream_t stream) {
  const int rows = p.nheads * p.headdim;
  const int per_cta = kUpdateWarps * kUpdateRowsPerWarp;
  const dim3 grid(static_cast<unsigned>((rows + per_cta - 1) / per_cta), static_cast<unsigned>(p.batch_size));
  selective_state_update_kernel<T, kLanesN, kNPerLane><<<grid, kUpdateWarps * 32, 0, stream>>>(p);
}

template <typename T>
void dispatch_update(const SSMStateUpdateParams& p, cudaStream_t stream) {
  switch (p.dstate) {
    case 8: launch_update_impl<T, 8, 1>(p, stream); break;
    case 16: launch_update_impl<T, 16, 1>(p, stream); break;
    case 32: launch_update_impl<T, 32, 1>(p, stream); break;
    case 64: launch_update_impl<T, 32, 2>(p, stream); break;
    case 128: launch_update_impl<T, 32, 4>(p, stream); break;
    default: launch_update_impl<T, 32, 8>(p, stream); break;  // 256 (validated)
  }
}

template <typename T, int kN>
void launch_scan_impl(const SelectiveScanParams& p, cudaStream_t stream) {
  constexpr int kCh = scan_channels(kN);
  const dim3 grid(static_cast<unsigned>((p.dim + kCh - 1) / kCh), static_cast<unsigned>(p.batch_size));
  selective_scan_fwd_kernel<T, kN><<<grid, kScanWarps * 32, 0, stream>>>(p);
}

constexpr int64_t align256(int64_t bytes) { return (bytes + 255) / 256 * 256; }

/// The five SSD steps. kTc selects the mma.sync bodies of the three matmul-heavy steps (bf16 / fp16
/// activations on SM80+); cumsum and state passing are the same fp32 kernels in both paths.
template <typename T, bool kTc>
void launch_ssd(const SSDParams& p, cudaStream_t stream, unsigned g_cumsum, unsigned g_bmm, unsigned g_state,
                dim3 g_pass, unsigned g_scan) {
  ssd_chunk_cumsum_kernel<T><<<g_cumsum, kSsdThreads, 0, stream>>>(p);
  if constexpr (kTc) {
    ssd_bmm_chunk_tc_kernel<T><<<g_bmm, kSsdThreads, 0, stream>>>(p);
    ssd_chunk_state_tc_kernel<T><<<g_state, kSsdThreads, 0, stream>>>(p);
  } else {
    ssd_bmm_chunk_kernel<T><<<g_bmm, kSsdThreads, 0, stream>>>(p);
    ssd_chunk_state_kernel<T><<<g_state, kSsdThreads, 0, stream>>>(p);
  }
  ssd_state_passing_kernel<<<g_pass, kSsdThreads, 0, stream>>>(p);
  if constexpr (kTc) {
    ssd_chunk_scan_tc_kernel<T><<<g_scan, kSsdThreads, 0, stream>>>(p);
  } else {
    ssd_chunk_scan_kernel<T><<<g_scan, kSsdThreads, 0, stream>>>(p);
  }
}

struct SsdLayout {
  int64_t dt, dA, CB, states, passed, total;  // byte offsets into the workspace
};

SsdLayout ssd_layout(int64_t chunks, int64_t heads, int64_t headdim, int64_t dstate, int64_t groups,
                     int64_t chunk_size) {
  SsdLayout o{};
  int64_t at = 0;
  o.dt = at, at += align256(4 * heads * chunks * chunk_size);
  o.dA = at, at += align256(4 * heads * chunks * chunk_size);
  o.CB = at, at += align256(4 * chunks * groups * chunk_size * chunk_size);
  o.states = at, at += align256(4 * chunks * heads * headdim * dstate);
  o.passed = at, at += align256(4 * chunks * heads * headdim * dstate);
  o.total = at;
  return o;
}

}  // namespace

Status selective_state_update(const SSMStateUpdateParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "selective_state_update";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  const SSMStateUpdateParams& p = params;
  if (p.batch_size < 0 || p.batch_size > 65535 || p.nheads < 0 || p.headdim < 0 || p.num_steps < 0 ||
      p.num_slots < 0 || p.ngroups < 1 || (p.nheads > 0 && p.nheads % p.ngroups != 0) ||
      int64_t(p.nheads) * p.headdim > INT32_MAX) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B <= 65535, ngroups | nheads)", fn);
  }
  if (p.dstate != 8 && p.dstate != 16 && p.dstate != 32 && p.dstate != 64 && p.dstate != 128 && p.dstate != 256) {
    return detail::set_error(Status::kUnsupportedShape, "%s: dstate must be 8..256 (power of two), got %d", fn, p.dstate);
  }
  if (p.has_dt_limit && !(p.dt_min <= p.dt_max)) {
    return detail::set_error(Status::kInvalidArgument, "%s: dt_limit must be (min, max) with min <= max", fn);
  }
  if (p.batch_size == 0 || p.nheads == 0 || p.headdim == 0 || p.num_steps == 0) return Status::kSuccess;
  if (!p.state || !p.x || !p.dt || !p.A || !p.B || !p.C || !p.out) {
    return detail::set_error(Status::kInvalidArgument, "%s: state, x, dt, A, B, C, out required", fn);
  }
  if (p.out == p.x && (p.out_batch_stride != p.x_batch_stride || p.out_token_stride != p.x_token_stride ||
                       p.out_head_stride != p.x_head_stride)) {
    return detail::set_error(Status::kInvalidArgument, "%s: in-place out must be the same view as x", fn);
  }
  by_dtype(dtype, [&](auto tag) { dispatch_update<decltype(tag)>(p, stream_of(ctx)); });
  return launch_status(fn);
}

Status selective_scan_fwd(const SelectiveScanParams& params, DType dtype, const LaunchContext& ctx) {
  const char* fn = "selective_scan_fwd";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  const SelectiveScanParams& p = params;
  if (p.batch_size < 0 || p.batch_size > 65535 || p.dim < 0 || p.num_slots < 0 || p.total_tokens < 0 ||
      p.ngroups < 1 || (p.dim > 0 && p.dim % p.ngroups != 0)) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B <= 65535, ngroups | dim)", fn);
  }
  if (p.dstate != 8 && p.dstate != 16 && p.dstate != 32) {
    return detail::set_error(Status::kUnsupportedShape, "%s: dstate must be 8, 16 or 32, got %d", fn, p.dstate);
  }
  const int channels = scan_channels(p.dstate);
  if (p.ngroups > 1 && (p.dim / p.ngroups) % channels != 0) {
    return detail::set_error(Status::kUnsupportedShape, "%s: dim / ngroups must be a multiple of %d", fn, channels);
  }
  if (p.batch_size == 0 || p.dim == 0) return Status::kSuccess;
  if (!p.u || !p.delta || !p.A || !p.B || !p.C || !p.out || !p.ssm_states || !p.query_start_loc) {
    return detail::set_error(Status::kInvalidArgument, "%s: u, delta, A, B, C, out, ssm_states, query_start_loc required", fn);
  }
  if (p.out == p.u || p.out == p.delta || p.out == p.z) {
    return detail::set_error(Status::kInvalidArgument, "%s: out must be disjoint from the inputs", fn);
  }
  if (p.u_dim_stride < 0 || p.u_token_stride < 0 || p.delta_dim_stride < 0 || p.delta_token_stride < 0 ||
      p.B_group_stride < 0 || p.B_state_stride < 0 || p.B_token_stride < 0 || p.C_group_stride < 0 ||
      p.C_state_stride < 0 || p.C_token_stride < 0 || p.out_dim_stride < 0 || p.out_token_stride < 0 ||
      p.z_dim_stride < 0 || p.z_token_stride < 0) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides", fn);
  }
  cudaStream_t stream = stream_of(ctx);
  by_dtype(dtype, [&](auto tag) {
    using T = decltype(tag);
    if (p.dstate == 8) launch_scan_impl<T, 8>(p, stream);
    else if (p.dstate == 16) launch_scan_impl<T, 16>(p, stream);
    else launch_scan_impl<T, 32>(p, stream);
  });
  return launch_status(fn);
}

Status ssd_workspace_requirements(int32_t max_tokens, int32_t max_batch, int32_t nheads, int32_t headdim,
                                  int32_t dstate, int32_t ngroups, int32_t chunk_size,
                                  WorkspaceRequirements* out) {
  if (out == nullptr || max_tokens < 0 || max_batch < 0 || nheads < 1 || headdim < 1 || dstate < 1 || ngroups < 1) {
    return detail::set_error(Status::kInvalidArgument, "ssd_workspace_requirements: invalid extents");
  }
  if (chunk_size != 64 && chunk_size != 128 && chunk_size != 256) {
    return detail::set_error(Status::kUnsupportedShape, "ssd_workspace_requirements: chunk_size must be 64, 128 or 256");
  }
  const int64_t chunks = (int64_t(max_tokens) + chunk_size - 1) / chunk_size + max_batch;
  *out = WorkspaceRequirements{};
  out->device_scratch_bytes = static_cast<size_t>(ssd_layout(chunks, nheads, headdim, dstate, ngroups, chunk_size).total);
  out->alignment = 256;
  return Status::kSuccess;
}

Status ssd_chunked_scan(const SSDParams& params, DType dtype, const Workspace& ws, const LaunchContext& ctx) {
  const char* fn = "ssd_chunked_scan";
  Status s = check_backend_dtype(fn, ctx, dtype);
  if (s != Status::kSuccess) return s;
  SSDParams p = params;
  if (p.chunk_size != 64 && p.chunk_size != 128 && p.chunk_size != 256) {
    return detail::set_error(Status::kUnsupportedShape, "%s: chunk_size must be 64, 128 or 256, got %d", fn, p.chunk_size);
  }
  if (p.batch_size < 0 || p.batch_size > 65535 || p.num_chunks < 0 || p.nheads < 1 || p.ngroups < 1 ||
      p.nheads % p.ngroups != 0 || p.num_slots < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: bad extents (B <= 65535, ngroups | nheads)", fn);
  }
  if (p.headdim < 1 || p.headdim > 256 || p.dstate < 1 || p.dstate > 256) {
    return detail::set_error(Status::kUnsupportedShape, "%s: headdim and dstate must be in 1..256", fn);
  }
  if (int64_t(p.num_chunks) * p.nheads * (p.chunk_size / 64) * ((p.headdim + 63) / 64) > INT32_MAX ||
      int64_t(p.num_chunks) * p.nheads * ((p.headdim + 63) / 64) * ((p.dstate + 63) / 64) > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: too many chunks", fn);
  }
  if (p.has_dt_limit && !(p.dt_min <= p.dt_max)) {
    return detail::set_error(Status::kInvalidArgument, "%s: dt_limit must be (min, max) with min <= max", fn);
  }
  if (p.snapshot_slots ? (p.snapshot_interval <= 0 || p.snapshot_interval % p.chunk_size != 0 || p.max_snapshots < 0)
                       : p.snapshot_interval != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: snapshot_interval must be a positive multiple of chunk_size with snapshot_slots", fn);
  }
  if (p.batch_size == 0 || p.num_chunks == 0) return Status::kSuccess;
  if (!p.x || !p.dt || !p.A || !p.B || !p.C || !p.out || !p.cu_seqlens || !p.cu_chunks || !p.cu_chunk_seqlens ||
      !p.ssm_states) {
    return detail::set_error(Status::kInvalidArgument, "%s: x, dt, A, B, C, out, chunk metadata, ssm_states required", fn);
  }
  if (p.x_token_stride < 0 || p.x_head_stride < 0 || p.dt_token_stride < 0 || p.dt_head_stride < 0 ||
      p.B_token_stride < 0 || p.B_group_stride < 0 || p.C_token_stride < 0 || p.C_group_stride < 0 ||
      p.out_token_stride < 0 || p.out_head_stride < 0 || p.z_token_stride < 0 || p.z_head_stride < 0 ||
      p.ssm_state_slot_stride < int64_t(p.nheads) * p.headdim * p.dstate) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: negative strides or overlapping state slots", fn);
  }
  if (p.out == p.x || p.out == p.z) {
    return detail::set_error(Status::kInvalidArgument, "%s: out must be disjoint from x and z", fn);
  }
  const SsdLayout layout = ssd_layout(p.num_chunks, p.nheads, p.headdim, p.dstate, p.ngroups, p.chunk_size);
  if (ws.device_scratch == nullptr || ws.device_scratch_bytes < static_cast<size_t>(layout.total) ||
      reinterpret_cast<uintptr_t>(ws.device_scratch) % 256 != 0) {
    return detail::set_error(Status::kWorkspaceTooSmall, "%s: need %lld scratch bytes, 256-aligned", fn,
                             static_cast<long long>(layout.total));
  }
  uint8_t* base = static_cast<uint8_t*>(ws.device_scratch);
  p.dt_proc = reinterpret_cast<float*>(base + layout.dt);
  p.dA_cumsum = reinterpret_cast<float*>(base + layout.dA);
  p.CB = reinterpret_cast<float*>(base + layout.CB);
  p.chunk_states = reinterpret_cast<float*>(base + layout.states);
  p.passed_states = reinterpret_cast<float*>(base + layout.passed);

  cudaStream_t stream = stream_of(ctx);
  const int nt = p.chunk_size / kSsdTile;
  const unsigned g_cumsum = static_cast<unsigned>(int64_t(p.num_chunks) * p.nheads);
  const unsigned g_bmm = static_cast<unsigned>(int64_t(p.num_chunks) * p.ngroups * (nt * (nt + 1) / 2));
  const unsigned g_state = static_cast<unsigned>(int64_t(p.num_chunks) * p.nheads *
                                                 ((p.headdim + kSsdTile - 1) / kSsdTile) *
                                                 ((p.dstate + kSsdTile - 1) / kSsdTile));
  const int64_t per_seq = int64_t(p.nheads) * p.headdim * p.dstate;
  const dim3 g_pass(static_cast<unsigned>((per_seq + kSsdThreads - 1) / kSsdThreads), static_cast<unsigned>(p.batch_size));
  const unsigned g_scan = static_cast<unsigned>(int64_t(p.num_chunks) * p.nheads * nt * ((p.headdim + kSsdTile - 1) / kSsdTile));
  // 16-bit activations on SM80+: the three matmul steps run on mma.sync (fp32 state and workspace
  // unchanged); fp32 keeps the scalar bodies.
  const bool tc = dtype != DType::kFloat32 && (ctx.arch_id == 0 || ctx.arch_id >= KOPS_TARGET_SM80);
  if (tc && dtype == DType::kBFloat16) {
    launch_ssd<__nv_bfloat16, true>(p, stream, g_cumsum, g_bmm, g_state, g_pass, g_scan);
  } else if (tc) {
    launch_ssd<__half, true>(p, stream, g_cumsum, g_bmm, g_state, g_pass, g_scan);
  } else {
    launch_ssd<float, false>(p, stream, g_cumsum, g_bmm, g_state, g_pass, g_scan);
  }
  return launch_status(fn);
}

}  // namespace kops::state
