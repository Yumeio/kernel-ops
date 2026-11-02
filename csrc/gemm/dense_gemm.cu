// =============================================================================
// kernel-ops | csrc/gemm/dense_gemm.cu
// B5 rows 1-4: 16/32-bit dense GEMM on CUDA cores and the scaled_gemm dispatcher.
//   gemm_simt_kernel         exact fp32 (or fp64) tiled GEMM: router/LM-head fp32 and the golden
//                            reference for large-M tests (never auto-picked over a tensor-core row)
//   gemv_kernel              M <= 16 decode GEMV, deterministic split-K for small N
//   splitk_reduce_kernel     fixed-order reduce of the split-K partials + epilogue
//   gemv_argmax_kernel       greedy LM head without writing logits
//   router_gemm_kernel       fp32 router logits (E = 8..1024), optional two-term bf16x2 weights
// Host entries: gemm, gemm_with_tactic, scaled_gemm, gemv_argmax, router_gemm,
//               gemm_workspace_requirements.
//
// Contract: csrc/gemm/dense_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// Numerics.  Every product is an fp32 FMA; accumulation order is fixed by the shape alone:
// lane-strided FMA chain, xor-butterfly over the 32 lanes, then (split-K) partials summed in
// split order.  No atomics, no dependence on M, the grid or the other rows, so one row alone and
// the same row inside a batch are bit-identical.  `krot` (rotating the K start by blockIdx) is not
// used: it changes the addition order and breaks batch invariance.
// =============================================================================
#include <algorithm>
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
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/fp8_gemm_kernels.cuh"
#include "csrc/gemm/int8_gemm_kernels.cuh"
#include "csrc/gemm/fp4_gemm_kernels.cuh"
#include "csrc/gemm/mixed_input_gemm_kernels.cuh"
#include "csrc/gemm/gguf_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/gemm/gemm_prep_kernels.cuh"
#include "csrc/gemm/ops.h"

namespace kops::gemm {
namespace {

using dev::Pack;

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

/// -inf without the host INFINITY macro (MSVC expands it to a constant nvcc warns about).
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xFF800000); }

/// `VEC` consecutive elements -> floats (the pointer is VEC * sizeof(T) aligned).
template <typename T, int VEC>
__device__ __forceinline__ void load_f(const T* p, float (&f)[VEC]) {
  const Pack<T, VEC> v = *reinterpret_cast<const Pack<T, VEC>*>(p);
  #pragma unroll
  for (int i = 0; i < VEC; ++i) f[i] = dev::to_f(v.x[i]);
}

/// Same for weights: 16-byte packs stream through ld.global.nc.L1::no_allocate.
template <typename T, int VEC>
__device__ __forceinline__ void load_w_f(const T* p, float (&f)[VEC]) {
  static_assert(sizeof(T) * VEC == 16, "weights are loaded as one 16-byte pack");
  const uint4 raw = dev::ld_stream16(p);
  Pack<T, VEC> v;
  memcpy(&v, &raw, 16);
  #pragma unroll
  for (int i = 0; i < VEC; ++i) f[i] = dev::to_f(v.x[i]);
}

template <typename D>
__device__ __forceinline__ void store_out(D* d, int64_t idx, float v) {
  dev::from_f(d[idx], v);
}

}  // namespace

// =============================================================================
// gemv_kernel + splitk_reduce_kernel
// =============================================================================

template <typename a_t, typename b_t, typename d_t, int M_TILE, int ROWS_PER_WARP, int WARPS, int VEC_SIZE>
__global__ void gemv_kernel(const __grid_constant__ GemvParams params) {
  constexpr int R = ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;
  if (n0 >= params.n) return;
  const int k = params.k;
  const int64_t m_total = params.m;
  const int splits = params.split_k < 1 ? 1 : params.split_k;
  const int split = blockIdx.y;
  const int kchunk = splits == 1 ? k : ((((k + splits - 1) / splits) + 255) / 256) * 256;
  const int kbeg = split * kchunk;
  const int kend = min(k, kbeg + kchunk);

  const a_t* arow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    arow[m] = static_cast<const a_t*>(params.a) + imin(m, m_total - 1) * params.lda;
  }
  const b_t* brow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    brow[r] = static_cast<const b_t*>(params.b) + imin(n0 + r, params.n - 1) * params.ldb;
  }

  float acc[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;
  }

  #pragma unroll 2
  for (int kc = kbeg + lane * VEC_SIZE; kc < kend; kc += 32 * VEC_SIZE) {
    float wf[R][VEC_SIZE];
    #pragma unroll
    for (int r = 0; r < R; ++r) load_w_f<b_t, VEC_SIZE>(brow[r] + kc, wf[r]);
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      float af[VEC_SIZE];
      load_f<a_t, VEC_SIZE>(arow[m] + kc, af);
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) acc[m][r] = fmaf(af[i], wf[r][i], acc[m][r]);
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == m * R + r && m < m_total && n0 + r < params.n) {
        if (splits == 1) {
          store_out(static_cast<d_t*>(params.d), m * params.ldd + n0 + r,
                    dev::gemm_epilogue<d_t>(v, params.epi, m, n0 + r));
        } else {
          params.split_workspace[(static_cast<int64_t>(split) * m_total + m) * params.n + n0 + r] = v;
        }
      }
    }
  }
}

template <typename d_t, int VEC_SIZE>
__global__ void splitk_reduce_kernel(d_t* __restrict__ d, const int64_t ldd,
                                     const float* __restrict__ partials, const int num_splits,
                                     const int m, const int n, const EpilogueParams epi) {
  const int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC_SIZE;
  const int64_t total = static_cast<int64_t>(m) * n;
  if (base >= total) return;
  #pragma unroll
  for (int v = 0; v < VEC_SIZE; ++v) {
    const int64_t e = base + v;
    if (e >= total) break;
    float s = 0.0f;
    for (int i = 0; i < num_splits; ++i) s += partials[static_cast<int64_t>(i) * total + e];
    const int64_t row = e / n;
    const int64_t col = e - row * n;
    store_out(d, row * ldd + col, dev::gemm_epilogue<d_t>(s, epi, row, col));
  }
}

// =============================================================================
// gemm_simt_kernel
// =============================================================================

namespace {

template <typename acc_t, typename T>
__device__ __forceinline__ acc_t cvt_in(T v) {
  if constexpr (std::is_same<T, double>::value) {
    return static_cast<acc_t>(v);
  } else {
    return static_cast<acc_t>(dev::to_f(v));
  }
}

}  // namespace

template <typename a_t, typename b_t, typename d_t, typename acc_t, int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_simt_kernel(const __grid_constant__ ScaledGemmArgs args) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  __shared__ acc_t As[TILE_M][TILE_K + 1];
  __shared__ acc_t Bs[TILE_N][TILE_K + 1];
  const int tx = threadIdx.x & 15;
  const int ty = threadIdx.x >> 4;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * TILE_M;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * TILE_N;
  const int64_t z = blockIdx.z;
  const a_t* A = static_cast<const a_t*>(args.a.data) + z * args.a.batch_stride;
  const b_t* B = static_cast<const b_t*>(args.b.data) + z * args.b.batch_stride;
  const bool a_kmajor = args.a.major == MajorOrder::kKMajor;
  const bool b_kmajor = args.b.major == MajorOrder::kKMajor;

  acc_t acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = acc_t(0);
  }
  for (int64_t k0 = 0; k0 < args.k; k0 += TILE_K) {
    for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += 256) {
      const int r = idx / TILE_K;
      const int c = idx % TILE_K;
      const int64_t gm = m0 + r;
      const int64_t gk = k0 + c;
      acc_t v = acc_t(0);
      if (gm < args.m && gk < args.k) {
        v = cvt_in<acc_t>(a_kmajor ? A[gm * args.a.ld + gk] : A[gk * args.a.ld + gm]);
      }
      As[r][c] = v;
    }
    for (int idx = threadIdx.x; idx < TILE_N * TILE_K; idx += 256) {
      const int r = idx / TILE_K;
      const int c = idx % TILE_K;
      const int64_t gn = n0 + r;
      const int64_t gk = k0 + c;
      acc_t v = acc_t(0);
      if (gn < args.n && gk < args.k) {
        v = cvt_in<acc_t>(b_kmajor ? B[gn * args.b.ld + gk] : B[gk * args.b.ld + gn]);
      }
      Bs[r][c] = v;
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      acc_t av[TM], bv[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = As[ty + 16 * i][kk];
      #pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = Bs[tx + 16 * j][kk];
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          if constexpr (std::is_same<acc_t, double>::value) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
          else acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int64_t gm = m0 + ty + 16 * i;
      const int64_t gn = n0 + tx + 16 * j;
      if (gm >= args.m || gn >= args.n) continue;
      d_t* D = static_cast<d_t*>(args.d) + z * args.batch_stride_d;
      if constexpr (std::is_same<acc_t, double>::value) {
        double v = acc[i][j] * static_cast<double>(args.epi.alpha_ptr != nullptr ? *args.epi.alpha_ptr : args.epi.alpha);
        if (args.epi.ops & kEpiBias) v += static_cast<const double*>(args.epi.bias)[gn];
        D[gm * args.ldd + gn] = static_cast<d_t>(v);
      } else {
        store_out(D, gm * args.ldd + gn, dev::gemm_epilogue<d_t>(acc[i][j], args.epi, gm, gn));
      }
    }
  }
}

// =============================================================================
// gemv_argmax_kernel
// =============================================================================

namespace {

struct ArgState {
  float mx;     // max logit
  int32_t idx;  // global vocab id of the max, -1 when no valid row
  float sm;     // sum exp(logit - mx)
};

__device__ __forceinline__ ArgState arg_merge(ArgState a, const ArgState& b) {
  const bool take_b = b.mx > a.mx || (b.mx == a.mx && b.idx >= 0 && (a.idx < 0 || b.idx < a.idx));
  const float M = fmaxf(a.mx, b.mx);
  const float ea = a.mx == neg_inf() ? 0.0f : a.sm * expf(a.mx - M);
  const float eb = b.mx == neg_inf() ? 0.0f : b.sm * expf(b.mx - M);
  ArgState r;
  r.mx = M;
  r.idx = take_b ? b.idx : a.idx;
  r.sm = ea + eb;
  return r;
}

constexpr int kArgRowsPerWarp = 4;

}  // namespace

template <typename h_t, typename w_t, int M_TILE, int WARPS>
__global__ void gemv_argmax_kernel(const __grid_constant__ GemvArgmaxParams p) {
  constexpr int R = kArgRowsPerWarp;
  constexpr int RC = WARPS * R;
  constexpr int VEC = 16 / sizeof(w_t);
  __shared__ float s_logit[M_TILE][RC];
  __shared__ ArgState s_state[WARPS];
  __shared__ int s_last;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int cta = blockIdx.x;
  const int num_ctas = gridDim.x;
  const int hidden = p.hidden_size;

  const h_t* hrow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) hrow[m] = static_cast<const h_t*>(p.hidden) + imin(m, p.m - 1) * p.ld_hidden;
  const w_t* wrow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    wrow[r] = static_cast<const w_t*>(p.lm_head) + imin(static_cast<int64_t>(cta) * RC + warp * R + r, p.v_shard - 1) * p.ld_lm_head;
  }
  float acc[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;
  }
  #pragma unroll 2
  for (int kc = lane * VEC; kc < hidden; kc += 32 * VEC) {
    float wf[R][VEC];
    #pragma unroll
    for (int r = 0; r < R; ++r) load_w_f<w_t, VEC>(wrow[r] + kc, wf[r]);
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      float af[VEC];
      load_f<h_t, VEC>(hrow[m] + kc, af);
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc[m][r] = fmaf(af[i], wf[r][i], acc[m][r]);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == 0) {
        const int64_t row = static_cast<int64_t>(cta) * RC + warp * R + r;
        const bool valid = row < p.v_shard && p.vocab_start + row < p.vocab_size;
        if (p.b_scale != nullptr) v *= p.b_scale[row < p.v_shard ? row : 0];
        v *= p.logit_scale;
        if (p.final_softcap > 0.0f) v = p.final_softcap * tanhf(v / p.final_softcap);
        s_logit[m][warp * R + r] = valid ? v : neg_inf();
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < M_TILE && threadIdx.x < p.m) {
    const int m = threadIdx.x;
    float best = neg_inf();
    int bi = -1;
    for (int j = 0; j < RC; ++j) {
      const float l = s_logit[m][j];
      if (l > best) { best = l; bi = j; }
    }
    float sum = 0.0f;
    if (bi >= 0) {
      for (int j = 0; j < RC; ++j) {
        const float l = s_logit[m][j];
        if (l != neg_inf()) sum += expf(l - best);
      }
    }
    const int64_t o = static_cast<int64_t>(m) * num_ctas + cta;
    p.partial_max[o] = best;
    p.partial_idx[o] = bi >= 0 ? static_cast<int32_t>(p.vocab_start + static_cast<int64_t>(cta) * RC + bi) : -1;
    p.partial_sumexp[o] = sum;
  }
  // The last CTA to finish folds the per-CTA partials (fixed order, so the result is reproducible).
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(p.done_counter, 1u) == static_cast<uint32_t>(num_ctas - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int m = 0; m < p.m; ++m) {
    ArgState st{neg_inf(), -1, 0.0f};
    for (int c = threadIdx.x; c < num_ctas; c += blockDim.x) {
      const int64_t o = static_cast<int64_t>(m) * num_ctas + c;
      st = arg_merge(st, ArgState{__ldcg(p.partial_max + o), __ldcg(p.partial_idx + o), __ldcg(p.partial_sumexp + o)});
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      ArgState other;
      other.mx = __shfl_xor_sync(0xFFFFFFFFu, st.mx, off);
      other.idx = __shfl_xor_sync(0xFFFFFFFFu, st.idx, off);
      other.sm = __shfl_xor_sync(0xFFFFFFFFu, st.sm, off);
      st = arg_merge(st, other);
    }
    if (lane == 0) s_state[warp] = st;
    __syncthreads();
    if (threadIdx.x == 0) {
      ArgState tot = s_state[0];
      for (int w = 1; w < WARPS; ++w) tot = arg_merge(tot, s_state[w]);
      p.out_ids[m] = tot.idx;
      if (p.out_logprob != nullptr) p.out_logprob[m] = -logf(tot.sm);
      // Slot 0 of each row carries the shard-wide max and sum-exp so a tensor-parallel caller can
      // allgather (max, id, sumexp) per rank.
      p.partial_max[static_cast<int64_t>(m) * num_ctas] = tot.mx;
      p.partial_sumexp[static_cast<int64_t>(m) * num_ctas] = tot.sm;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *p.done_counter = 0u;
}

// =============================================================================
// router_gemm_kernel / router_gemm_bf16x2_kernel
// =============================================================================

namespace {

constexpr int kRouterWarps = 8;

/// Eight warps split H; each warp computes EXPERTS_PER_CTA experts for M_TILE rows.  kTwoTerm
/// adds lo_scale * (x . W_lo^T) from the second weight.
template <typename x_t, typename w_t, int M_TILE, int EPC, bool kTwoTerm>
__device__ __forceinline__ void router_body(const RouterGemmParams& p) {
  constexpr int VEC = 16 / sizeof(w_t);
  __shared__ float red[kRouterWarps][EPC][M_TILE][kTwoTerm ? 2 : 1];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t e0 = static_cast<int64_t>(blockIdx.x) * EPC;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * M_TILE;
  const int h = p.hidden;
  const int hslice = (((h + kRouterWarps - 1) / kRouterWarps) + VEC - 1) / VEC * VEC;
  const int hbeg = warp * hslice;
  const int hend = min(h, hbeg + hslice);

  const x_t* xrow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) xrow[m] = static_cast<const x_t*>(p.x) + imin(m0 + m, p.m - 1) * p.ldx;
  const w_t* wrow[EPC];
  [[maybe_unused]] const w_t* lrow[EPC];
  #pragma unroll
  for (int e = 0; e < EPC; ++e) {
    const int64_t er = imin(e0 + e, p.num_experts - 1) * h;
    wrow[e] = static_cast<const w_t*>(kTwoTerm ? p.w_hi : p.w) + er;
    lrow[e] = static_cast<const w_t*>(kTwoTerm ? p.w_lo : p.w) + er;
  }
  float acc[EPC][M_TILE], acc2[EPC][M_TILE];
  #pragma unroll
  for (int e = 0; e < EPC; ++e) {
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) { acc[e][m] = 0.0f; acc2[e][m] = 0.0f; }
  }
  for (int kc = hbeg + lane * VEC; kc < hend; kc += 32 * VEC) {
    float xf[M_TILE][VEC];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) load_f<x_t, VEC>(xrow[m] + kc, xf[m]);
    #pragma unroll
    for (int e = 0; e < EPC; ++e) {
      float wf[VEC];
      load_w_f<w_t, VEC>(wrow[e] + kc, wf);
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc[e][m] = fmaf(xf[m][i], wf[i], acc[e][m]);
      }
      if constexpr (kTwoTerm) {
        float lf[VEC];
        load_w_f<w_t, VEC>(lrow[e] + kc, lf);
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          #pragma unroll
          for (int i = 0; i < VEC; ++i) acc2[e][m] = fmaf(xf[m][i], lf[i], acc2[e][m]);
        }
      }
    }
  }
  #pragma unroll
  for (int e = 0; e < EPC; ++e) {
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      float v = acc[e][m];
      [[maybe_unused]] float v2 = acc2[e][m];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if constexpr (kTwoTerm) v2 += __shfl_xor_sync(0xFFFFFFFFu, v2, o);
      }
      if (lane == 0) {
        red[warp][e][m][0] = v;
        if constexpr (kTwoTerm) red[warp][e][m][1] = v2;
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < EPC * M_TILE; idx += blockDim.x) {
    const int e = idx / M_TILE;
    const int m = idx % M_TILE;
    if (e0 + e >= p.num_experts || m0 + m >= p.m) continue;
    float s = 0.0f;
    [[maybe_unused]] float s2 = 0.0f;
    #pragma unroll
    for (int w = 0; w < kRouterWarps; ++w) {
      s += red[w][e][m][0];
      if constexpr (kTwoTerm) s2 += red[w][e][m][kTwoTerm ? 1 : 0];
    }
    float out = s;
    if constexpr (kTwoTerm) out = fmaf(p.lo_scale, s2, s);
    if (p.bias != nullptr) out += p.bias[e0 + e];
    p.logits[(m0 + m) * p.ld_logits + e0 + e] = out;
  }
}

}  // namespace

template <typename x_t, typename w_t, int M_TILE, int EXPERTS_PER_CTA>
__global__ void router_gemm_kernel(const __grid_constant__ RouterGemmParams params) {
  router_body<x_t, w_t, M_TILE, EXPERTS_PER_CTA, false>(params);
}

template <int M_TILE, int EXPERTS_PER_CTA, typename ArchTag>
__global__ void router_gemm_bf16x2_kernel(const __grid_constant__ RouterGemmParams params) {
  static_assert(sizeof(ArchTag) > 0, "the CUDA-core body is architecture generic");
  router_body<__nv_bfloat16, __nv_bfloat16, M_TILE, EXPERTS_PER_CTA, true>(params);
}

// =============================================================================
// Host
// =============================================================================
namespace {

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

Status backend_ok(const LaunchContext& ctx, GemmBackend backend, const char* who) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", who);
  }
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: only the native provider exists", who);
  }
  return Status::kSuccess;
}

int dtype_bytes(DType t) {
  const int bits = dtype_bits(t);
  return bits >= 8 ? bits / 8 : 0;
}

bool is_float16(DType t) { return t == DType::kBFloat16 || t == DType::kFloat16; }

constexpr uint32_t kDenseEpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                      kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

/// Split-K is a function of the shape alone (never of the device or of M), so results do not
/// change between GPUs or batch sizes.
int auto_split_k(int64_t n, int64_t k) {
  if (n >= 2048) return 1;
  int s = static_cast<int>(std::min<int64_t>(8, std::max<int64_t>(1, 4096 / std::max<int64_t>(n, 1))));
  while (s > 1 && k / s < 1024) s >>= 1;
  return s;
}

Status check_epilogue(const EpilogueParams& e, DType d_dtype, const char* who) {
  if ((e.ops & ~kDenseEpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "%s: epilogue ops 0x%x are not implemented", who, e.ops & ~kDenseEpiAllowed);
  }
  if (((e.ops & kEpiRowScale) != 0u && e.row_scale == nullptr) || ((e.ops & kEpiColScale) != 0u && e.col_scale == nullptr) ||
      ((e.ops & kEpiBias) != 0u && e.bias == nullptr) ||
      ((e.ops & kEpiResidualAdd) != 0u && e.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "%s: epilogue operand missing (residual is read in the output dtype)", who);
  }
  if ((e.ops & kEpiOutFp32) != 0u && d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "%s: kEpiOutFp32 needs an fp32 output", who);
  }
  return Status::kSuccess;
}

template <typename a_t, typename b_t, typename d_t, int MT, int R>
Status launch_gemv_tile(const GemvParams& gp, cudaStream_t s) {
  constexpr int kWarps = 8;
  constexpr int VEC = 16 / sizeof(b_t);
  const dim3 grid(static_cast<unsigned>((gp.n + kWarps * R - 1) / (kWarps * R)), static_cast<unsigned>(gp.split_k));
  gemv_kernel<a_t, b_t, d_t, MT, R, kWarps, VEC><<<grid, kWarps * 32, 0, s>>>(gp);
  return launch_error("gemv_kernel");
}

template <typename a_t, typename b_t, typename d_t>
Status launch_gemv(const GemvParams& gp, cudaStream_t s) {
  if (gp.m <= 1) return launch_gemv_tile<a_t, b_t, d_t, 1, 4>(gp, s);
  if (gp.m <= 2) return launch_gemv_tile<a_t, b_t, d_t, 2, 4>(gp, s);
  if (gp.m <= 4) return launch_gemv_tile<a_t, b_t, d_t, 4, 2>(gp, s);
  if (gp.m <= 8) return launch_gemv_tile<a_t, b_t, d_t, 8, 2>(gp, s);
  return launch_gemv_tile<a_t, b_t, d_t, 16, 1>(gp, s);
}

template <typename d_t>
Status launch_splitk_reduce(const ScaledGemmArgs& args, const GemvParams& gp, cudaStream_t s) {
  constexpr int kVec = 4;
  const int64_t total = static_cast<int64_t>(gp.m) * gp.n;
  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((total + static_cast<int64_t>(threads) * kVec - 1) / (static_cast<int64_t>(threads) * kVec));
  splitk_reduce_kernel<d_t, kVec><<<blocks, threads, 0, s>>>(static_cast<d_t*>(args.d), args.ldd, gp.split_workspace,
                                                              gp.split_k, gp.m, gp.n, args.epi);
  return launch_error("splitk_reduce_kernel");
}

template <typename a_t, typename b_t, typename d_t>
Status run_gemv_typed(const ScaledGemmArgs& args, GemvParams gp, cudaStream_t s) {
  Status st = launch_gemv<a_t, b_t, d_t>(gp, s);
  if (st != Status::kSuccess || gp.split_k == 1) return st;
  return launch_splitk_reduce<d_t>(args, gp, s);
}

template <typename a_t, typename b_t>
Status run_gemv_out(const ScaledGemmArgs& args, const GemvParams& gp, cudaStream_t s) {
  switch (args.d_dtype) {
    case DType::kFloat32: return run_gemv_typed<a_t, b_t, float>(args, gp, s);
    case DType::kFloat16: return run_gemv_typed<a_t, b_t, __half>(args, gp, s);
    default: return run_gemv_typed<a_t, b_t, __nv_bfloat16>(args, gp, s);
  }
}

template <typename a_t, typename b_t, typename d_t, typename acc_t>
Status launch_simt_typed(const ScaledGemmArgs& args, cudaStream_t s) {
  constexpr int kTile = 64;
  const dim3 grid(static_cast<unsigned>((args.n + kTile - 1) / kTile), static_cast<unsigned>((args.m + kTile - 1) / kTile),
                  static_cast<unsigned>(args.batch));
  gemm_simt_kernel<a_t, b_t, d_t, acc_t, kTile, kTile, 16><<<grid, 256, 0, s>>>(args);
  return launch_error("gemm_simt_kernel");
}

template <typename a_t, typename b_t>
Status launch_simt_out(const ScaledGemmArgs& args, cudaStream_t s) {
  switch (args.d_dtype) {
    case DType::kFloat32: return launch_simt_typed<a_t, b_t, float, float>(args, s);
    case DType::kFloat16: return launch_simt_typed<a_t, b_t, __half, float>(args, s);
    default: return launch_simt_typed<a_t, b_t, __nv_bfloat16, float>(args, s);
  }
}

template <typename a_t>
Status launch_simt_b(const ScaledGemmArgs& args, cudaStream_t s) {
  switch (args.b.dtype) {
    case DType::kFloat32: return launch_simt_out<a_t, float>(args, s);
    case DType::kFloat16: return launch_simt_out<a_t, __half>(args, s);
    default: return launch_simt_out<a_t, __nv_bfloat16>(args, s);
  }
}

Status run_simt(const ScaledGemmArgs& args, cudaStream_t s) {
  if (args.a.dtype == DType::kFloat64) {
    return launch_simt_typed<double, double, double, double>(args, s);
  }
  switch (args.a.dtype) {
    case DType::kFloat32: return launch_simt_b<float>(args, s);
    case DType::kFloat16: return launch_simt_b<__half>(args, s);
    default: return launch_simt_b<__nv_bfloat16>(args, s);
  }
}

/// Dense 16/32-bit entry: GEMV for M <= 16 (K-major B), exact SIMT for recipe kFP32.
Status run_dense(const ScaledGemmArgs& args, const GemmTactic* tactic, const Workspace& ws,
                 const LaunchContext& ctx) {
  const GemmRecipe rc = args.recipe;
  const bool fp32 = rc == GemmRecipe::kFP32;
  if (rc != GemmRecipe::kBF16 && rc != GemmRecipe::kFP16 && !fp32) {
    return detail::set_error(Status::kNotImplemented, "gemm: recipe %d has no native body yet", static_cast<int>(rc));
  }
  const DType adt = args.a.dtype;
  const DType bdt = args.b.dtype;
  if (rc == GemmRecipe::kBF16 && (adt != DType::kBFloat16 || bdt != DType::kBFloat16)) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: recipe kBF16 needs bf16 operands");
  }
  if (rc == GemmRecipe::kFP16 && (adt != DType::kFloat16 || bdt != DType::kFloat16)) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: recipe kFP16 needs fp16 operands");
  }
  const bool f64 = adt == DType::kFloat64;
  if (fp32 && !(is_float16(adt) || adt == DType::kFloat16 || adt == DType::kFloat32 || f64) ) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: recipe kFP32 operands must be fp32, bf16, fp16 or fp64");
  }
  if (fp32 && (!(is_float16(bdt) || bdt == DType::kFloat32 || bdt == DType::kFloat64) || (f64 != (bdt == DType::kFloat64)))) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: recipe kFP32 operands must be fp32, bf16, fp16 or fp64");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32 &&
      !(f64 && args.d_dtype == DType::kFloat64)) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: output must be bf16, fp16 or fp32 (fp64 for fp64 operands)");
  }
  if (args.m < 1 || args.n < 1 || args.k < 1 || args.batch < 1) {
    return detail::set_error(Status::kInvalidArgument, "gemm: empty problem");
  }
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemm: null operand");
  }
  const bool a_k = args.a.major == MajorOrder::kKMajor;
  const bool b_k = args.b.major == MajorOrder::kKMajor;
  if (args.a.ld < (a_k ? args.k : args.m) || args.b.ld < (b_k ? args.k : args.n) || args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "gemm: leading dimensions smaller than the extents");
  }
  Status st = Status::kSuccess;
  if (!f64) {
    st = check_epilogue(args.epi, args.d_dtype, "gemm");
    if (st != Status::kSuccess) return st;
  } else if ((args.epi.ops & ~static_cast<uint32_t>(kEpiBias)) != 0u) {
    return detail::set_error(Status::kNotImplemented, "gemm: fp64 supports alpha and Bias only");
  }
  if (args.workspace != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemm: pass workspace through the Workspace argument");
  }
  const cudaStream_t s = stream_of(ctx);

  const bool homogeneous = adt == bdt && (adt == DType::kBFloat16 || adt == DType::kFloat16 || adt == DType::kFloat32);
  const bool gemv_ok = args.m <= 16 && args.batch == 1 && a_k && b_k && homogeneous && !f64;
  if (!gemv_ok) {
    if (!fp32) {
      return detail::set_error(Status::kNotImplemented,
                               "gemm: bf16/fp16 with M > 16 needs the tensor-core path (B7); recipe kFP32 runs the SIMT golden");
    }
    if (args.batch > 1 && (args.a.batch_stride == 0 || args.b.batch_stride == 0 || args.batch_stride_d == 0)) {
      return detail::set_error(Status::kInvalidArgument, "gemm: batched problems need batch strides");
    }
    return run_simt(args, s);
  }

  const int es = dtype_bytes(adt);
  if (!aligned(args.a.data, 16) || (args.a.ld * es) % 16 != 0 || !aligned(args.b.data, 16) || (args.b.ld * es) % 16 != 0 ||
      args.k % (16 / es) != 0) {
    return detail::set_error(Status::kMisaligned, "gemm: GEMV operands need 16 B alignment and K a multiple of 16 B");
  }
  int split = tactic != nullptr && tactic->split_k_mode == SplitKMode::kParallel ? tactic->split_k_slices
                                                                                  : auto_split_k(args.n, args.k);
  if (tactic != nullptr && tactic->split_k_mode != SplitKMode::kNone && tactic->split_k_mode != SplitKMode::kParallel) {
    return detail::set_error(Status::kNotImplemented, "gemm: only parallel split-K exists on the CUDA-core path");
  }
  if (split < 1 || split > 64) return detail::set_error(Status::kInvalidArgument, "gemm: split_k must be 1..64");
  GemvParams gp{};
  gp.a = args.a.data; gp.lda = args.a.ld;
  gp.b = args.b.data; gp.ldb = args.b.ld;
  gp.d = args.d; gp.ldd = args.ldd;
  gp.epi = args.epi;
  gp.m = static_cast<int32_t>(args.m); gp.n = static_cast<int32_t>(args.n); gp.k = static_cast<int32_t>(args.k);
  gp.split_k = split;
  if (split > 1) {
    const size_t bytes = static_cast<size_t>(split) * args.m * args.n * sizeof(float);
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < bytes) {
      return detail::set_error(Status::kWorkspaceTooSmall, "gemm: split-K needs %zu B of device scratch", bytes);
    }
    gp.split_workspace = static_cast<float*>(ws.device_scratch);
  }
  switch (adt) {
    case DType::kBFloat16: return run_gemv_out<__nv_bfloat16, __nv_bfloat16>(args, gp, s);
    case DType::kFloat16: return run_gemv_out<__half, __half>(args, gp, s);
    default: return run_gemv_out<float, float>(args, gp, s);
  }
}

ScaledGemmArgs to_scaled(const GemmProblem& p, const GemmArgs& a) {
  ScaledGemmArgs s{};
  s.a.data = a.a; s.a.ld = a.lda; s.a.batch_stride = a.batch_stride_a; s.a.dtype = p.a_dtype; s.a.major = p.a_major;
  s.b.data = a.b; s.b.ld = a.ldb; s.b.batch_stride = a.batch_stride_b; s.b.dtype = p.b_dtype; s.b.major = p.b_major;
  s.d = a.d; s.ldd = a.ldd; s.batch_stride_d = a.batch_stride_d; s.d_dtype = p.d_dtype;
  s.epi = a.epi;
  s.m = p.m; s.n = p.n; s.k = p.k; s.batch = p.batch;
  s.recipe = p.recipe;
  s.workspace = a.workspace; s.workspace_bytes = a.workspace_bytes;
  return s;
}

}  // namespace

Status gemm(const GemmProblem& problem, const GemmArgs& args, GemmBackend backend, const Workspace& ws,
            const LaunchContext& ctx) {
  Status st = backend_ok(ctx, backend, "gemm");
  if (st != Status::kSuccess) return st;
  if (problem.epilogue_ops != args.epi.ops) {
    return detail::set_error(Status::kInvalidArgument, "gemm: GemmProblem::epilogue_ops disagrees with GemmArgs::epi.ops");
  }
  if (args.split_k > 1 || args.a_perm != nullptr || args.b_g_idx != nullptr) {
    if (args.split_k > 1) {
      GemmTactic t{};
      t.split_k_mode = SplitKMode::kParallel;
      t.split_k_slices = static_cast<int16_t>(args.split_k);
      return run_dense(to_scaled(problem, args), &t, ws, ctx);
    }
    return detail::set_error(Status::kNotImplemented, "gemm: act-order permutations belong to the quantized recipes");
  }
  return run_dense(to_scaled(problem, args), nullptr, ws, ctx);
}

Status gemm_with_tactic(const ScaledGemmArgs& args, const GemmTactic& tactic, const Workspace& ws,
                        const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "gemm_with_tactic");
  if (st != Status::kSuccess) return st;
  return scaled_gemm(args, &tactic, GemmBackend::kNative, ws, ctx);
}

Status scaled_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                   const Workspace& ws, const LaunchContext& ctx) {
  Status st = backend_ok(ctx, backend, "scaled_gemm");
  if (st != Status::kSuccess) return st;
  switch (args.recipe) {
    case GemmRecipe::kBF16:
    case GemmRecipe::kFP16:
    case GemmRecipe::kTF32:
    case GemmRecipe::kFP32:
      if (args.recipe == GemmRecipe::kTF32) return detail::set_error(Status::kNotImplemented, "scaled_gemm: kTF32 needs the tensor-core path (B7)");
      return run_dense(args, tactic, ws, ctx);
    case GemmRecipe::kFP8Tensorwise:
    case GemmRecipe::kFP8Rowwise:
    case GemmRecipe::kFP8Blockwise:
    case GemmRecipe::kFP8BlockwiseUE8M0:
    case GemmRecipe::kMXFP8:
      return fp8_gemm(args, tactic, backend, ws, ctx);
    case GemmRecipe::kINT8Tensorwise:
    case GemmRecipe::kINT8Rowwise:
      return int8_gemm(args, tactic, backend, ws, ctx);
    case GemmRecipe::kNVFP4:
    case GemmRecipe::kMXFP4:
    case GemmRecipe::kMXFP4xMXFP8:
    case GemmRecipe::kMXFP6xMXFP8:
    case GemmRecipe::kW4A8FP4xFP8:
      return fp4_gemm(args, tactic, backend, ws, ctx);
    case GemmRecipe::kW4A16Int:
    case GemmRecipe::kW8A16Int:
    case GemmRecipe::kW8A16FP8:
    case GemmRecipe::kW4A16MXFP4:
    case GemmRecipe::kW4A16NVFP4:
    case GemmRecipe::kW4A8IntFP8:
    case GemmRecipe::kW4A8IntInt8:
      if (args.recipe == GemmRecipe::kW4A16MXFP4 || args.recipe == GemmRecipe::kW4A16NVFP4) return fp4_gemm(args, tactic, backend, ws, ctx);
      return mixed_input_gemm(args, tactic, backend, ws, ctx);
    case GemmRecipe::kGGUF:
    case GemmRecipe::kLUT4:
      return detail::set_error(Status::kNotImplemented, "scaled_gemm: GGUF / LUT4 use gguf_gemm / lut_gemv");
  }
  return detail::set_error(Status::kInvalidArgument, "scaled_gemm: unknown recipe %d", static_cast<int>(args.recipe));
}

Status gemm_workspace_requirements(const GemmProblem& problem, const GemmTactic* tactic,
                                   GemmBackend backend, const LaunchContext& ctx,
                                   WorkspaceRequirements* out) {
  Status st = backend_ok(ctx, backend, "gemm_workspace_requirements");
  if (st != Status::kSuccess) return st;
  if (out == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm_workspace_requirements: null output");
  *out = WorkspaceRequirements{};
  const bool dense = problem.recipe == GemmRecipe::kBF16 || problem.recipe == GemmRecipe::kFP16 || problem.recipe == GemmRecipe::kFP32;
  if (dense && problem.m <= 16 && problem.batch == 1) {
    const int split = tactic != nullptr && tactic->split_k_mode == SplitKMode::kParallel ? tactic->split_k_slices
                                                                                          : auto_split_k(problem.n, problem.k);
    if (split > 1) out->device_scratch_bytes = static_cast<size_t>(split) * problem.m * problem.n * sizeof(float);
  }
  return Status::kSuccess;
}

// ---- gemv_argmax -------------------------------------------------------------------------

int32_t gemv_argmax_num_ctas(int32_t v_shard) {
  constexpr int kRows = 8 * kArgRowsPerWarp;
  return (v_shard + kRows - 1) / kRows;
}

namespace {

template <typename h_t, typename w_t, int MT>
Status launch_argmax_tile(const GemvArgmaxParams& p, cudaStream_t s) {
  gemv_argmax_kernel<h_t, w_t, MT, 8><<<gemv_argmax_num_ctas(p.v_shard), 256, 0, s>>>(p);
  return launch_error("gemv_argmax_kernel");
}

template <typename h_t, typename w_t>
Status launch_argmax(const GemvArgmaxParams& p, cudaStream_t s) {
  if (p.m <= 1) return launch_argmax_tile<h_t, w_t, 1>(p, s);
  if (p.m <= 4) return launch_argmax_tile<h_t, w_t, 4>(p, s);
  if (p.m <= 8) return launch_argmax_tile<h_t, w_t, 8>(p, s);
  return launch_argmax_tile<h_t, w_t, 16>(p, s);
}

}  // namespace

Status gemv_argmax(const GemvArgmaxParams& params, DType hidden_dtype, DType weight_dtype,
                   const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "gemv_argmax");
  if (st != Status::kSuccess) return st;
  if (!is_float16(hidden_dtype) || !is_float16(weight_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "gemv_argmax: hidden and lm_head must be bf16 or fp16");
  }
  if (params.m < 1 || params.m > 16 || params.hidden_size < 8 || params.v_shard < 1) {
    return detail::set_error(Status::kUnsupportedShape, "gemv_argmax: M must be 1..16, hidden >= 8 and v_shard >= 1");
  }
  if (params.hidden_size % 8 != 0) return detail::set_error(Status::kUnsupportedShape, "gemv_argmax: hidden must be a multiple of 8");
  if (params.hidden == nullptr || params.lm_head == nullptr || params.partial_max == nullptr || params.partial_idx == nullptr ||
      params.partial_sumexp == nullptr || params.out_ids == nullptr || params.done_counter == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemv_argmax: null operand or workspace");
  }
  if (params.ld_hidden < params.hidden_size || params.ld_lm_head < params.hidden_size || params.vocab_size <= params.vocab_start) {
    return detail::set_error(Status::kInvalidArgument, "gemv_argmax: leading dimension or vocabulary range inconsistent");
  }
  if (!aligned(params.hidden, 16) || !aligned(params.lm_head, 16) || (params.ld_hidden * 2) % 16 != 0 || (params.ld_lm_head * 2) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "gemv_argmax: operands must be 16 B aligned");
  }
  const cudaStream_t s = stream_of(ctx);
  if (hidden_dtype == DType::kBFloat16) {
    return weight_dtype == DType::kBFloat16 ? launch_argmax<__nv_bfloat16, __nv_bfloat16>(params, s)
                                            : launch_argmax<__nv_bfloat16, __half>(params, s);
  }
  return weight_dtype == DType::kBFloat16 ? launch_argmax<__half, __nv_bfloat16>(params, s)
                                          : launch_argmax<__half, __half>(params, s);
}

// ---- router_gemm -------------------------------------------------------------------------

namespace {

template <typename x_t, typename w_t, int MT, int EPC>
Status launch_router_tile(const RouterGemmParams& p, bool two_term, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>((p.num_experts + EPC - 1) / EPC), static_cast<unsigned>((p.m + MT - 1) / MT));
  if (two_term) router_gemm_bf16x2_kernel<MT, EPC, Sm80><<<grid, kRouterWarps * 32, 0, s>>>(p);
  else router_gemm_kernel<x_t, w_t, MT, EPC><<<grid, kRouterWarps * 32, 0, s>>>(p);
  return launch_error("router_gemm_kernel");
}

template <typename x_t, typename w_t>
Status launch_router(const RouterGemmParams& p, bool two_term, cudaStream_t s) {
  // L2 traffic is x: (E / EPC) * M * H and W: (M / M_TILE) * E * H; M_TILE * EPC = 32 accumulators is
  // the register budget, so (8, 4) halves the traffic of the old (8, 2) and (16, 1) tiles. The k order
  // of every output element is independent of the tile, so the logits stay bit-identical.
  if (p.m <= 4) return launch_router_tile<x_t, w_t, 4, 4>(p, two_term, s);
  return launch_router_tile<x_t, w_t, 8, 4>(p, two_term, s);
}

}  // namespace

Status router_gemm(const RouterGemmParams& params, bool two_term, DType x_dtype, DType w_dtype,
                   const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "router_gemm");
  if (st != Status::kSuccess) return st;
  if (x_dtype != DType::kBFloat16 && x_dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "router_gemm: hidden must be bf16 or fp16");
  }
  if (two_term ? (x_dtype != DType::kBFloat16 || w_dtype != DType::kBFloat16) : (w_dtype != DType::kBFloat16 && w_dtype != DType::kFloat32 && w_dtype != DType::kFloat16)) {
    return detail::set_error(Status::kUnsupportedDtype, two_term ? "router_gemm: two-term mode needs bf16 hidden and bf16 hi/lo weights"
                                                                  : "router_gemm: router weight must be bf16, fp16 or fp32");
  }
  if (!two_term && w_dtype == DType::kFloat16 && x_dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "router_gemm: fp16 weights pair with fp16 hidden");
  }
  if (!two_term && w_dtype == DType::kBFloat16 && x_dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "router_gemm: bf16 weights pair with bf16 hidden");
  }
  if (params.m < 1 || params.num_experts < 1 || params.hidden < 8 || params.hidden % 8 != 0) {
    return detail::set_error(Status::kUnsupportedShape, "router_gemm: M >= 1, E >= 1 and hidden a multiple of 8 required");
  }
  if (params.x == nullptr || params.logits == nullptr || (two_term ? (params.w_hi == nullptr || params.w_lo == nullptr) : params.w == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "router_gemm: null operand");
  }
  if (params.ldx < params.hidden || params.ld_logits < params.num_experts) {
    return detail::set_error(Status::kInvalidArgument, "router_gemm: leading dimensions smaller than the extents");
  }
  const int ws = w_dtype == DType::kFloat32 ? 4 : 2;
  const void* w0 = two_term ? params.w_hi : params.w;
  if (!aligned(params.x, 16) || (params.ldx * 2) % 16 != 0 || !aligned(w0, 16) || (static_cast<int64_t>(params.hidden) * ws) % 16 != 0 ||
      (two_term && !aligned(params.w_lo, 16))) {
    return detail::set_error(Status::kMisaligned, "router_gemm: operands must be 16 B aligned and rows 16 B multiples");
  }
  const cudaStream_t s = stream_of(ctx);
  if (two_term) return launch_router<__nv_bfloat16, __nv_bfloat16>(params, true, s);
  if (w_dtype == DType::kFloat32) {
    return x_dtype == DType::kBFloat16 ? launch_router<__nv_bfloat16, float>(params, false, s) : launch_router<__half, float>(params, false, s);
  }
  return w_dtype == DType::kBFloat16 ? launch_router<__nv_bfloat16, __nv_bfloat16>(params, false, s)
                                     : launch_router<__half, __half>(params, false, s);
}

}  // namespace kops::gemm
