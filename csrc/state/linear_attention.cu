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
