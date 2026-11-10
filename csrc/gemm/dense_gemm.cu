// =============================================================================
// kernel-ops | csrc/gemm/dense_gemm.cu
// B5 rows 1-4: 16/32-bit dense GEMM on CUDA cores and the scaled_gemm dispatcher.
//   gemm_simt_kernel         exact fp32 (or fp64) tiled GEMM: router/LM-head fp32 and the golden
//                            reference for large-M tests (never auto-picked over a tensor-core row)
//   gemv_kernel              M <= 16 decode GEMV, deterministic split-K for small N
//   splitk_reduce_kernel     fixed-order reduce of the split-K partials + epilogue
//   gemv_argmax_kernel       greedy LM head without writing logits
//   router_gemm_kernel       fp32 router logits (E = 8..1024), optional two-term bf16x2 weights
//   dual_gemm_kernel         (B7 row 2, end of file) D = GLU(A.B0^T, A.B1^T) on mma.sync, one pass over A
//   bmm_kernel               (B7 row 2, end of file) strided batched bf16/fp16 GEMM on mma.sync, B as [N,K] or [K,N]
// Host entries: gemm, gemm_with_tactic, scaled_gemm, gemv_argmax, router_gemm,
//               gemm_workspace_requirements.
// B13: scaled_gemm routes an explicit SM120a tactic (kSm120WarpSpecialized /
//      kSm120BlockScaledWS) to the sm120a cells of sm120a_impl.cuh for bf16 / fp16 and
//      tensorwise / rowwise FP8, and reports kNotImplemented for the recipes with no
//      SM120a body. gemm() / gemm_with_tactic() without such a tactic keep the B7 path.
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
#include <map>
#include <mutex>
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
#include "csrc/gemm/sm120a_impl.cuh"

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
  const bool quant_out = !fp32 && !f64 && (args.epi.ops & kEpiQuantOut) != 0u;   // D is replaced by quant_out.q
  if (args.a.data == nullptr || args.b.data == nullptr || (args.d == nullptr && !quant_out)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: null operand");
  }
  const bool a_k = args.a.major == MajorOrder::kKMajor;
  const bool b_k = args.b.major == MajorOrder::kKMajor;
  const int64_t d_cols = !fp32 && !f64 && (args.epi.ops & kEpiGLU) != 0u ? args.n / 2 : args.n;   // GLU halves N
  if (args.a.ld < (a_k ? args.k : args.m) || args.b.ld < (b_k ? args.k : args.n) || (args.ldd < d_cols && !quant_out)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: leading dimensions smaller than the extents");
  }
  if (!fp32 && !f64) {
    const bool small_gemv = args.m <= 16 && args.batch == 1 && a_k && b_k && adt == bdt && !quant_out;
    if (!small_gemv || mma_selects(args, tactic, ctx)) return mma_gemm(args, tactic, ws, ctx);
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

// =============================================================================
// SM120a (B13) host: dense bf16/fp16 and FP8 tiles of sm120a_impl.cuh
// =============================================================================
// Reached only through an explicit tactic whose mainloop is kSm120WarpSpecialized or
// kSm120BlockScaledWS on a KOPS_TARGET_SM120A context; every other path keeps the B7
// mma.sync body. One CTA per output tile (no persistent scheduler, no split-K), so the
// launch needs no workspace.

bool sm120_mainloop(MainloopSchedule m) {
  return m == MainloopSchedule::kSm120WarpSpecialized || m == MainloopSchedule::kSm120BlockScaledWS;
}

/// Shapes instantiated in this build. tile_k is in elements (64 / element size), stages the ring depth.
struct Sm120Tile {
  int16_t tm, tn, tk;
  int8_t stages;
};
constexpr Sm120Tile kSm120DenseTiles[] = {{64, 128, 32, 4}, {128, 128, 32, 3}};
constexpr Sm120Tile kSm120Fp8Tiles[] = {{64, 128, 64, 4}, {128, 128, 64, 3}};
constexpr Sm120Tile kSm120Fp4Tiles[] = {{64, 128, 32, 4}, {128, 128, 32, 3}};
constexpr int kSm120NumDenseTiles = static_cast<int>(sizeof(kSm120DenseTiles) / sizeof(kSm120DenseTiles[0]));
constexpr int kSm120NumFp8Tiles = static_cast<int>(sizeof(kSm120Fp8Tiles) / sizeof(kSm120Fp8Tiles[0]));
constexpr int kSm120NumFp4Tiles = static_cast<int>(sizeof(kSm120Fp4Tiles) / sizeof(kSm120Fp4Tiles[0]));

constexpr size_t sm120_tile_smem(const Sm120Tile& t) {
  return static_cast<size_t>(t.stages) * (static_cast<size_t>(t.tm) + t.tn) * 64;
}

/// Largest dynamic shared memory one block may opt into on this device (cached).
int sm120_smem_limit(int device) {
  static std::mutex mu;
  static std::map<int, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;
  int v = 48 * 1024;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cache[device] = v;
  return v;
}

/// Tile of a tactic. Only the compiled shapes exist; a caller asking for another one gets the list.
/// tile_k is not part of the geometry (it is 64 / element size, fixed by the atom) and is ignored, exactly
/// as on the B7 path, so a tactic that only names tile_m / tile_n keeps GemmTactic's default tile_k.
Status sm120_tile_of(const Sm120Tile* table, int n, const GemmTactic& t, const char* who, int* out_id) {
  int id = -1;
  for (int i = 0; i < n; ++i) {
    if (table[i].tm == t.tile_m && table[i].tn == t.tile_n) id = i;
  }
  if (id < 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: no SM120a tile %dx%d (compiled: 64x128, 128x128)", who, t.tile_m, t.tile_n);
  }
  if (t.stages != 0 && t.stages != table[id].stages) {
    return detail::set_error(Status::kUnsupportedShape, "%s: tile %dx%d is compiled with %d stages, not %d", who,
                             t.tile_m, t.tile_n, table[id].stages, t.stages);
  }
  *out_id = id;
  return Status::kSuccess;
}

/// Epilogue ops of the SM120a cells: the dense subset plus the operand scales of the FP8 recipes. The
/// grouped / fused-MoE / quantizing / KV-append epilogues belong to the B7 body and are refused here
/// instead of being dropped.
constexpr uint32_t kSm120EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                      kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

Status sm120_check_epilogue(const EpilogueParams& e, DType d_dtype, const char* who) {
  if ((e.ops & ~kSm120EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "%s: epilogue ops 0x%x have no SM120a body (B7 mma.sync serves them)",
                             who, e.ops & ~kSm120EpiAllowed);
  }
  if (((e.ops & kEpiRowScale) != 0u && e.row_scale == nullptr) || ((e.ops & kEpiColScale) != 0u && e.col_scale == nullptr) ||
      ((e.ops & kEpiBias) != 0u && e.bias == nullptr) || ((e.ops & kEpiResidualAdd) != 0u && e.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "%s: epilogue operand missing (residual is read in the output dtype)", who);
  }
  if ((e.ops & kEpiBias) != 0u && e.bias_dtype != DType::kBFloat16 && e.bias_dtype != DType::kFloat16 &&
      e.bias_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: bias must be bf16, fp16 or fp32", who);
  }
  if ((e.ops & kEpiOutFp32) != 0u && d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "%s: kEpiOutFp32 needs an fp32 output", who);
  }
  return Status::kSuccess;
}

/// Operand / problem validation shared by the SM120a tiles, then the tile choice. The order is the B7
/// one: argument errors first (they are meaningful on any GPU), the exact-SM check last.
Status sm120_check_args(const ScaledGemmArgs& a, const GemmTactic& t, int eb, const Sm120Tile* table, int ntiles,
                        const char* who, int* tile_id) {
  if (a.m < 1 || a.n < 1 || a.k < 1 || a.batch < 1) {
    return detail::set_error(Status::kInvalidArgument, "%s: empty problem", who);
  }
  if (a.m > 0x7FFFFFFF || a.n > 0x7FFFFFFF || a.k > 0x7FFFFFFF) {
    return detail::set_error(Status::kUnsupportedShape, "%s: extents beyond int32", who);
  }
  if (a.a.data == nullptr || a.b.data == nullptr || a.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: null operand", who);
  }
  if (a.a.major != MajorOrder::kKMajor || a.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kNotImplemented, "%s: MN-major operands (weight_transpose first)", who);
  }
  if (a.a.ld < a.k || a.b.ld < a.k || a.ldd < a.n) {
    return detail::set_error(Status::kInvalidArgument, "%s: leading dimensions smaller than the extents", who);
  }
  if (a.workspace != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: pass workspace through the Workspace argument", who);
  }
  if (a.batch > 1 && (a.a.batch_stride == 0 || a.b.batch_stride == 0 || a.batch_stride_d == 0)) {
    return detail::set_error(Status::kInvalidArgument, "%s: batched problems need batch strides", who);
  }
  if (a.batch > 65535) return detail::set_error(Status::kUnsupportedShape, "%s: batch > 65535", who);
  const int kmul = 16 / eb;   // K elements of one 16-byte chunk (the loader's granularity)
  const bool a16 = aligned(a.a.data, 16) && aligned(a.b.data, 16) && aligned(a.d, 16);
  if (!a16 || (a.a.ld * eb) % 16 != 0 || (a.b.ld * eb) % 16 != 0 || (a.a.batch_stride * eb) % 16 != 0 ||
      (a.b.batch_stride * eb) % 16 != 0 || a.k % kmul != 0) {
    return detail::set_error(Status::kMisaligned,
                             "%s: SM120a operands need 16 B alignment (pointers, ld, batch stride) and K %% %d == 0",
                             who, kmul);
  }
  if (t.split_k_mode != SplitKMode::kNone || t.split_k_slices > 1) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM120a cell has no split-K yet", who);
  }
  if (t.swap_ab) return detail::set_error(Status::kNotImplemented, "%s: swap_ab has no SM120a body yet", who);
  if (t.persistent) {
    return detail::set_error(Status::kNotImplemented, "%s: the SM120a persistent tile scheduler is not implemented (one CTA per tile)",
                             who);
  }
  return sm120_tile_of(table, ntiles, t, who, tile_id);
}

GemmTmaParams sm120_params(const ScaledGemmArgs& args, const Sm120Tile& cfg) {
  GemmTmaParams p{};
  p.args = args;
  p.args.split_k_mode = SplitKMode::kNone;
  p.args.split_k_slices = 1;
  p.num_tiles_m = static_cast<int32_t>((args.m + cfg.tm - 1) / cfg.tm);
  p.num_tiles_n = static_cast<int32_t>((args.n + cfg.tn - 1) / cfg.tn);
  p.num_tiles_k = static_cast<int32_t>((args.k + cfg.tk - 1) / cfg.tk);
  return p;
}

template <typename T, typename Dt, int TM, int TN, int TK, int ST>
Status launch_sm120_tile(const GemmTmaParams& p, cudaStream_t s) {
  using Tr = GemmTraits<T, T, Dt, float, TM, TN, TK, 1, 1, ST, MainloopSchedule::kSm120WarpSpecialized,
                        EpilogueSchedule::kDirectStore, Sm120a>;
  constexpr size_t kSmem = static_cast<size_t>(ST) * (TM + TN) * 64;
  static const cudaError_t attr = cudaFuncSetAttribute(gemm_sm120_kernel<Tr>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "gemm_sm120_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  const unsigned tiles = static_cast<unsigned>(p.num_tiles_m) * static_cast<unsigned>(p.num_tiles_n);
  // Block size is the body's warp layout (sm120a::Threads), not a copy of it: a tile row is one warp
  // row per 32 output channels, so a 128-wide tile runs eight warps.
  gemm_sm120_kernel<Tr><<<dim3(tiles, 1u, static_cast<unsigned>(p.args.batch)),
                          static_cast<unsigned>(sm120a::Threads<Tr>::value), kSmem, s>>>(p);
  return launch_error("gemm_sm120_kernel");
}

template <typename T, typename Dt, int TM, int TN, int TK, int ST>
Status launch_fp8_sm120_tile(const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  using Tr = GemmTraits<T, T, Dt, float, TM, TN, TK, 1, 1, ST, MainloopSchedule::kSm120WarpSpecialized,
                        EpilogueSchedule::kDirectStore, Sm120a>;
  constexpr size_t kSmem = static_cast<size_t>(ST) * (TM + TN) * 64;
  const cudaError_t attr = cudaFuncSetAttribute(
      recipe == GemmRecipe::kFP8Tensorwise ? fp8_gemm_sm120_kernel<Tr, GemmRecipe::kFP8Tensorwise>
                                           : fp8_gemm_sm120_kernel<Tr, GemmRecipe::kFP8Rowwise>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "fp8_gemm_sm120_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  const unsigned tiles = static_cast<unsigned>(p.num_tiles_m) * static_cast<unsigned>(p.num_tiles_n);
  const unsigned threads = static_cast<unsigned>(sm120a::Threads<Tr>::value);
  if (recipe == GemmRecipe::kFP8Tensorwise) {
    fp8_gemm_sm120_kernel<Tr, GemmRecipe::kFP8Tensorwise>
        <<<dim3(tiles, 1u, static_cast<unsigned>(p.args.batch)), threads, kSmem, s>>>(p);
  } else {
    fp8_gemm_sm120_kernel<Tr, GemmRecipe::kFP8Rowwise>
        <<<dim3(tiles, 1u, static_cast<unsigned>(p.args.batch)), threads, kSmem, s>>>(p);
  }
  return launch_error("fp8_gemm_sm120_kernel");
}

template <typename T, typename Dt>
Status sm120_dense_tile_id(int id, const GemmTmaParams& p, cudaStream_t s) {
  switch (id) {
    case 0: return launch_sm120_tile<T, Dt, 64, 128, 32, 4>(p, s);
    case 1: return launch_sm120_tile<T, Dt, 128, 128, 32, 3>(p, s);
    default: return detail::set_error(Status::kInternal, "gemm: SM120a dense tile id %d", id);
  }
}

// ---- SM120a FP4 (B13): software-decode body ---------------------------------
// Packed e2m1 operands are decoded to bf16 while staging (sm120a::fp4_tile), so tile_k stays 32 K
// elements per stage (64 decoded bytes per row) and the atom is m16n8k16 bf16.

template <typename Dt, int TM, int TN, int ST>
Status launch_fp4_sm120_tile(const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  using Tr = GemmTraits<f4e2m1x2_t, f4e2m1x2_t, Dt, float, TM, TN, 32, 1, 1, ST,
                        MainloopSchedule::kSm120BlockScaledWS, EpilogueSchedule::kDirectStore, Sm120a>;
  constexpr size_t kSmem = static_cast<size_t>(ST) * (TM + TN) * 64;
  const cudaError_t attr = cudaFuncSetAttribute(
      recipe == GemmRecipe::kNVFP4 ? fp4_gemm_sm120_kernel<Tr, GemmRecipe::kNVFP4>
                                   : fp4_gemm_sm120_kernel<Tr, GemmRecipe::kMXFP4>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "fp4_gemm_sm120_kernel: %s",
                                     cudaGetErrorString(attr));
  }
  const unsigned tiles = static_cast<unsigned>(p.num_tiles_m) * static_cast<unsigned>(p.num_tiles_n);
  const unsigned threads = static_cast<unsigned>(sm120a::Threads<Tr>::value);
  if (recipe == GemmRecipe::kNVFP4) {
    fp4_gemm_sm120_kernel<Tr, GemmRecipe::kNVFP4>
        <<<dim3(tiles, 1u, static_cast<unsigned>(p.args.batch)), threads, kSmem, s>>>(p);
  } else {
    fp4_gemm_sm120_kernel<Tr, GemmRecipe::kMXFP4>
        <<<dim3(tiles, 1u, static_cast<unsigned>(p.args.batch)), threads, kSmem, s>>>(p);
  }
  return launch_error("fp4_gemm_sm120_kernel");
}

template <typename Dt>
Status fp4_sm120_tile_id(int id, const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  switch (id) {
    case 0: return launch_fp4_sm120_tile<Dt, 64, 128, 4>(p, s, recipe);
    case 1: return launch_fp4_sm120_tile<Dt, 128, 128, 3>(p, s, recipe);
    default: return detail::set_error(Status::kInternal, "gemm: SM120a FP4 tile id %d", id);
  }
}

Status fp4_sm120_out(int id, const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  switch (p.args.d_dtype) {
    case DType::kFloat32: return fp4_sm120_tile_id<float>(id, p, s, recipe);
    case DType::kFloat16: return fp4_sm120_tile_id<__half>(id, p, s, recipe);
    default: return fp4_sm120_tile_id<__nv_bfloat16>(id, p, s, recipe);
  }
}

template <typename T, typename Dt>
Status fp8_sm120_tile_id(int id, const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  switch (id) {
    case 0: return launch_fp8_sm120_tile<T, Dt, 64, 128, 64, 4>(p, s, recipe);
    case 1: return launch_fp8_sm120_tile<T, Dt, 128, 128, 64, 3>(p, s, recipe);
    default: return detail::set_error(Status::kInternal, "gemm: SM120a FP8 tile id %d", id);
  }
}

/// Output dtype dispatch (bf16 / fp16 / fp32); the operand type is fixed by the recipe.
template <typename T>
Status sm120_dense_out(int id, const GemmTmaParams& p, cudaStream_t s) {
  switch (p.args.d_dtype) {
    case DType::kFloat32: return sm120_dense_tile_id<T, float>(id, p, s);
    case DType::kFloat16: return sm120_dense_tile_id<T, __half>(id, p, s);
    default: return sm120_dense_tile_id<T, __nv_bfloat16>(id, p, s);
  }
}

template <typename T>
Status fp8_sm120_out(int id, const GemmTmaParams& p, cudaStream_t s, GemmRecipe recipe) {
  switch (p.args.d_dtype) {
    case DType::kFloat32: return fp8_sm120_tile_id<T, float>(id, p, s, recipe);
    case DType::kFloat16: return fp8_sm120_tile_id<T, __half>(id, p, s, recipe);
    default: return fp8_sm120_tile_id<T, __nv_bfloat16>(id, p, s, recipe);
  }
}

Status sm120_dense_gemm(const ScaledGemmArgs& args, const GemmTactic& t, const LaunchContext& ctx) {
  const DType want = args.recipe == GemmRecipe::kFP16 ? DType::kFloat16 : DType::kBFloat16;
  if (args.a.dtype != want || args.b.dtype != want) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "gemm: the SM120a 16-bit tile takes bf16 x bf16 (kBF16) or fp16 x fp16 (kFP16)");
  }
  if (args.d_dtype != want && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: SM120a output is the operand dtype or fp32");
  }
  if (args.a.scale != nullptr || args.b.scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemm: 16-bit operands carry no operand scales");
  }
  int id = -1;
  Status st = sm120_check_args(args, t, 2, kSm120DenseTiles, kSm120NumDenseTiles, "gemm", &id);
  if (st != Status::kSuccess) return st;
  st = sm120_check_epilogue(args.epi, args.d_dtype, "gemm");
  if (st != Status::kSuccess) return st;
  if (sm120_tile_smem(kSm120DenseTiles[id]) > static_cast<size_t>(sm120_smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: the SM120a tile needs %zu B of shared memory",
                             sm120_tile_smem(kSm120DenseTiles[id]));
  }
  if (ctx.arch_id != KOPS_TARGET_SM120A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "gemm: the mainloop %d is the SM120a cell and needs KOPS_TARGET_SM120A, found arch %d",
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  const GemmTmaParams p = sm120_params(args, kSm120DenseTiles[id]);
  const cudaStream_t s = stream_of(ctx);
  if (want == DType::kBFloat16) return sm120_dense_out<__nv_bfloat16>(id, p, s);
  return sm120_dense_out<__half>(id, p, s);
}

/// SM120a FP4 (B13): kNVFP4 (block 16, E4M3 block scales + per-tensor global scales) and kMXFP4
/// (block 32, UE8M0 block scales). Packed e2m1 operands: `ld` and the batch stride are byte pitches,
/// as in gemm/fp4_gemm.cu. Block scales are read per block boundary in the mainloop; only the
/// row-major and 128x4 swizzled layouts exist (the swizzled one is the output of
/// block_scale_swizzle_kernel with cols_blocks = K / block).
Status sm120_fp4_gemm(const ScaledGemmArgs& args, const GemmTactic& t, const LaunchContext& ctx) {
  const bool nv = args.recipe == GemmRecipe::kNVFP4;
  const int block = nv ? 16 : 32;
  if (args.a.dtype != DType::kFloat4E2M1x2 || args.b.dtype != DType::kFloat4E2M1x2) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: the SM120a FP4 tile takes packed e2m1 (kFloat4E2M1x2) operands");
  }
  for (const ScaledOperand* op : {&args.a, &args.b}) {
    if (op->packing != WeightPacking::kNone && op->packing != WeightPacking::kFP4E2M1) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "fp4_gemm: SM120a reads plain packed e2m1 rows (packing kNone / kFP4E2M1); repack Marlin / "
                               "TRT-LLM layouts first");
    }
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: SM120a output is bf16, fp16 or fp32");
  }
  // Block scales: one byte per (row, block) on both operands, E4M3 for NVFP4 and UE8M0 for MXFP4.
  const DType want_scale = nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0;
  const ScaleGranularity want_gran = nv ? ScaleGranularity::kNVFP4 : ScaleGranularity::kMX;
  for (const ScaledOperand* op : {&args.a, &args.b}) {
    if (op->scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "fp4_gemm: %s block scales are required (per %d K elements)",
                               op == &args.a ? "activation" : "weight", block);
    }
    if (op->scale_dtype != want_scale) {
      return detail::set_error(Status::kUnsupportedDtype, "fp4_gemm: %s scales must be %s for this recipe",
                               op == &args.a ? "activation" : "weight",
                               nv ? "E4M3 (kFloat8E4M3)" : "UE8M0 (kFloat8E8M0)");
    }
    if (op->granularity != want_gran && op->granularity != ScaleGranularity::kNone) {
      return detail::set_error(Status::kInvalidArgument, "fp4_gemm: %s scale granularity must be %s",
                               op == &args.a ? "activation" : "weight", nv ? "kNVFP4 (1x16)" : "kMX (1x32)");
    }
    if (op->group_k != 0 && op->group_k != block) {
      return detail::set_error(Status::kInvalidArgument, "fp4_gemm: %s group_k must be %d for this recipe",
                               op == &args.a ? "activation" : "weight", block);
    }
    // The swizzled tile stride is derived from K / block (the layout block_scale_swizzle_kernel writes),
    // so ScaledOperand::scale_ld is only read for the row-major form.
    if (op->scale_layout != ScaleLayout::kRowMajor && op->scale_layout != ScaleLayout::kSwizzled128x4) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "fp4_gemm: SM120a block scales are row-major or the 128x4 swizzled layout; the TMA "
                               "column-major / packed-UE8M0 / Marlin layouts are not decoded");
    }
  }
  if (!nv && (args.a.global_scale != nullptr || args.b.global_scale != nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: MXFP4 has no per-tensor global scale (E8M0 blocks only)");
  }
  if (args.m < 1 || args.n < 1 || args.k < 1 || args.batch < 1) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: empty problem");
  }
  if (args.m > 0x7FFFFFFF || args.n > 0x7FFFFFFF || args.k > 0x7FFFFFFF) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: extents beyond int32");
  }
  if (args.batch > 65535) return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: batch > 65535");
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kNotImplemented, "fp4_gemm: MN-major operands (weight_transpose first)");
  }
  if (args.k % block != 0) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: K must be a multiple of the %d-element scale block", block);
  }
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: null operand");
  }
  const int64_t k_bytes = args.k / 2;   // packed: 2 elements per byte
  if (args.a.ld < k_bytes || args.b.ld < k_bytes) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: leading dimensions smaller than K / 2 bytes");
  }
  if (args.ldd < args.n) return detail::set_error(Status::kInvalidArgument, "fp4_gemm: ldd smaller than N");
  if (args.batch > 1 && (args.a.batch_stride == 0 || args.b.batch_stride == 0 || args.batch_stride_d == 0)) {
    return detail::set_error(Status::kInvalidArgument, "fp4_gemm: batched problems need batch strides");
  }
  // The packed rows are read in 4-byte groups (8 e2m1), so the byte pitch only has to be a multiple of
  // 4; the base pointers keep the 16 B requirement of every GEMM entry.
  const bool a16 = aligned(args.a.data, 16) && aligned(args.b.data, 16) && aligned(args.d, 16);
  if (!a16 || args.a.ld % 4 != 0 || args.b.ld % 4 != 0 || args.a.batch_stride % 4 != 0 ||
      args.b.batch_stride % 4 != 0) {
    return detail::set_error(Status::kMisaligned,
                             "fp4_gemm: SM120a operands need 16 B aligned pointers and a 4-byte row pitch (packed bytes)");
  }
  if (t.split_k_mode != SplitKMode::kNone || t.split_k_slices > 1) {
    return detail::set_error(Status::kNotImplemented, "fp4_gemm: the SM120a cell has no split-K yet");
  }
  if (t.swap_ab) return detail::set_error(Status::kNotImplemented, "fp4_gemm: swap_ab has no SM120a body yet");
  if (t.persistent) {
    return detail::set_error(Status::kNotImplemented, "fp4_gemm: the SM120a persistent tile scheduler is not implemented");
  }
  int id = -1;
  Status st = sm120_tile_of(kSm120Fp4Tiles, kSm120NumFp4Tiles, t, "fp4_gemm", &id);
  if (st != Status::kSuccess) return st;
  st = sm120_check_epilogue(args.epi, args.d_dtype, "fp4_gemm");
  if (st != Status::kSuccess) return st;
  if (sm120_tile_smem(kSm120Fp4Tiles[id]) > static_cast<size_t>(sm120_smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_gemm: the SM120a tile needs %zu B of shared memory",
                             sm120_tile_smem(kSm120Fp4Tiles[id]));
  }
  if (ctx.arch_id != KOPS_TARGET_SM120A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "fp4_gemm: the mainloop %d is the SM120a cell and needs KOPS_TARGET_SM120A, found arch %d",
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  const GemmTmaParams p = sm120_params(args, kSm120Fp4Tiles[id]);
  return fp4_sm120_out(id, p, stream_of(ctx), args.recipe);
}

Status sm120_fp8_gemm(const ScaledGemmArgs& args, const GemmTactic& t, const LaunchContext& ctx) {
  if (args.a.dtype != DType::kFloat8E4M3 || args.b.dtype != DType::kFloat8E4M3 ||
      args.a.fp8 != Fp8Encoding::kOCP || args.b.fp8 != Fp8Encoding::kOCP) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: the SM120a FP8 tile takes OCP e4m3 x e4m3");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: SM120a output is bf16, fp16 or fp32");
  }
  // Operand scales of the tensorwise / rowwise recipes: per tensor folds into alpha, per token (A) and
  // per channel (B) stay vectors. Block / MX scales need the block-scaled path, refused here.
  for (const ScaledOperand* op : {&args.a, &args.b}) {
    if (op->scale == nullptr) continue;
    if (op->scale_dtype != DType::kFloat32) {
      return detail::set_error(Status::kUnsupportedDtype, "fp8_gemm: SM120a tensorwise / rowwise scales are fp32");
    }
    const bool per_tensor = op->granularity == ScaleGranularity::kPerTensor;
    const bool vec = op == &args.a ? op->granularity == ScaleGranularity::kPerToken
                                   : op->granularity == ScaleGranularity::kPerChannel;
    if (!per_tensor && !vec) {
      return detail::set_error(Status::kNotImplemented,
                               "fp8_gemm: SM120a covers per-tensor, per-token (A) and per-channel (B) FP8 scales; the 1x128 "
                               "/ 128x128 blockwise recipes keep the B7 mma.sync promotion path and the MX recipes have "
                               "no SM120a body");
    }
    if (op->scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kUnsupportedLayout, "fp8_gemm: SM120a operand scales are the contiguous row vector");
    }
  }
  int id = -1;
  Status st = sm120_check_args(args, t, 1, kSm120Fp8Tiles, kSm120NumFp8Tiles, "fp8_gemm", &id);
  if (st != Status::kSuccess) return st;
  st = sm120_check_epilogue(args.epi, args.d_dtype, "fp8_gemm");
  if (st != Status::kSuccess) return st;
  if (sm120_tile_smem(kSm120Fp8Tiles[id]) > static_cast<size_t>(sm120_smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "fp8_gemm: the SM120a tile needs %zu B of shared memory",
                             sm120_tile_smem(kSm120Fp8Tiles[id]));
  }
  if (ctx.arch_id != KOPS_TARGET_SM120A) {
    return detail::set_error(Status::kUnsupportedArch,
                             "fp8_gemm: the mainloop %d is the SM120a cell and needs KOPS_TARGET_SM120A, found arch %d",
                             static_cast<int>(t.mainloop), ctx.arch_id);
  }
  const GemmTmaParams p = sm120_params(args, kSm120Fp8Tiles[id]);
  return fp8_sm120_out<f8e4m3_t>(id, p, stream_of(ctx), args.recipe);
}

/// One entry for every recipe whose tactic names an SM120a mainloop (called by scaled_gemm).
Status launch_gemm_sm120(const ScaledGemmArgs& args, const GemmTactic& t, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "gemm: CUDA backend only");
  }
  switch (args.recipe) {
    case GemmRecipe::kBF16:
    case GemmRecipe::kFP16:
      return sm120_dense_gemm(args, t, ctx);
    case GemmRecipe::kFP8Tensorwise:
    case GemmRecipe::kFP8Rowwise:
      return sm120_fp8_gemm(args, t, ctx);
    case GemmRecipe::kFP8Blockwise:
    case GemmRecipe::kFP8BlockwiseUE8M0:
      return detail::set_error(Status::kNotImplemented,
                               "gemm: SM120a has no block-scaled FP8 body yet (1x128 x 128x128 promotion); kFP8Blockwise "
                               "and kFP8BlockwiseUE8M0 keep the B7 mma.sync path");
    case GemmRecipe::kMXFP8:
      return detail::set_error(Status::kNotImplemented,
                               "gemm: SM120 block-scaled MXFP8 (mma.sync kind::mxf8f6f4.block_scale) is not implemented");
    case GemmRecipe::kNVFP4:
    case GemmRecipe::kMXFP4:
      return sm120_fp4_gemm(args, t, ctx);
    case GemmRecipe::kMXFP4xMXFP8:
    case GemmRecipe::kMXFP6xMXFP8:
      return detail::set_error(Status::kNotImplemented,
                               "gemm: mixed MX (e4m3 / e2m3 activations with e2m1 / e2m3 weights) needs a second operand "
                               "loader in sm120a_impl.cuh; only kNVFP4 and kMXFP4 are instantiated");
    case GemmRecipe::kINT8Tensorwise:
    case GemmRecipe::kINT8Rowwise:
      return detail::set_error(Status::kNotImplemented,
                               "gemm: the SM120a INT8 tile is not implemented (B7 mma.sync s8 serves kINT8Rowwise)");
    default:
      return detail::set_error(Status::kNotImplemented, "gemm: recipe %d has no SM120a body", static_cast<int>(args.recipe));
  }
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
  // SM120a / SM121a cells (B13): only an explicit tactic whose mainloop names the SM120a pipeline selects
  // them, and only on the exact target. Everything else keeps the B7 mma.sync path below.
  if (tactic != nullptr && sm120_mainloop(tactic->mainloop)) return launch_gemm_sm120(args, *tactic, ctx);
  switch (args.recipe) {
    case GemmRecipe::kBF16:
    case GemmRecipe::kFP16:
    case GemmRecipe::kTF32:
    case GemmRecipe::kFP32:
      if (args.recipe == GemmRecipe::kTF32) return mma_gemm(args, tactic, ws, ctx);   // tensor core only (B7)
      return run_dense(args, tactic, ws, ctx);
    case GemmRecipe::kFP8Tensorwise:
    case GemmRecipe::kFP8Rowwise:
    case GemmRecipe::kFP8Blockwise:
    case GemmRecipe::kFP8BlockwiseUE8M0:
    case GemmRecipe::kMXFP8:
      return fp8_gemm(args, tactic, backend, ws, ctx);
    case GemmRecipe::kINT8Tensorwise:
    case GemmRecipe::kINT8Rowwise:
    case GemmRecipe::kW4A8IntInt8:   // QServe W4A8: w4a8_int_gemm_kernel in int8_gemm.cu
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
  if (tactic != nullptr && sm120_mainloop(tactic->mainloop)) {
    return Status::kSuccess;   // the SM120a cell runs one CTA per tile: no split-K / scheduler workspace
  }
  const bool dense = problem.recipe == GemmRecipe::kBF16 || problem.recipe == GemmRecipe::kFP16 || problem.recipe == GemmRecipe::kFP32;
  const bool mma16 = problem.recipe == GemmRecipe::kBF16 || problem.recipe == GemmRecipe::kFP16;
  if (mma16 && (problem.m > 16 || problem.batch > 1 || (tactic != nullptr && tactic->mainloop == MainloopSchedule::kMultistageMmaSync))) {
    return mma_workspace(problem, tactic, ctx, out);
  }
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


// =============================================================================
// B7 row 2: dual_gemm_kernel and bmm_kernel on mma.sync.m16n8k16 (SM80+)
//   dual_gemm_kernel  D = GLU(A . B0^T, A . B1^T): one pass over the A tile feeds two accumulator sets.
//   bmm_kernel        D[b] = A[b] . B[b] (B K-major [N, K] or [K, N] through ldmatrix.trans), head-strided
//                     views of [T, H, *] through lda / stride_a_batch.
// The tensor-core helpers below (tc_*) duplicate the small helpers of gemm_mma.cu (cp.async, ldmatrix,
// the 64 B-row XOR swizzle) on purpose: this file does not depend on gemm_mma.cu internals.
// Numerics: fp32 accumulation inside mma.sync, the K chain of one element is the same sequence of k16
// steps in every tile, one rounding at the end.
// =============================================================================
namespace {

__device__ __forceinline__ uint32_t tc_smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void tc_cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#endif
}
__device__ __forceinline__ void tc_cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void tc_cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}
__device__ __forceinline__ void tc_ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}
__device__ __forceinline__ void tc_ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

/// K-major tile [NROWS][32] (64 B rows): 16 B chunk c of row r lives at chunk c ^ ((r >> 1) & 3).
template <typename T, int NROWS, int THREADS>
__device__ __forceinline__ void tc_load_kmajor(T* sdst, const T* g, int64_t ld, int rows_limit, int row0, int k0, int K, int tid) {
  for (int c = tid; c < NROWS * 4; c += THREADS) {
    const int row = c >> 2;
    const int ch = c & 3;
    const int gr = row0 + row;
    const int gk = k0 + ch * 8;
    const bool pred = gr < rows_limit && gk < K;
    const T* src = g + static_cast<int64_t>(pred ? gr : 0) * ld + (pred ? gk : 0);
    tc_cp_async16(tc_smem_u32(sdst + row * 32 + ((ch ^ ((row >> 1) & 3)) << 3)), src, pred);
  }
}

/// N-major tile [32 k][BN n]: chunk c of row r lives at chunk c ^ (r & 7) (BN * 2 bytes per row).
template <typename T, int BN, int THREADS>
__device__ __forceinline__ void tc_load_nmajor(T* sdst, const T* g, int64_t ld, int N, int n0, int k0, int K, int tid) {
  constexpr int CH = BN / 8;
  for (int c = tid; c < 32 * CH; c += THREADS) {
    const int row = c / CH;
    const int ch = c % CH;
    const int gk = k0 + row;
    const int gn = n0 + ch * 8;
    const bool pred = gk < K && gn < N;
    const T* src = g + static_cast<int64_t>(pred ? gk : 0) * ld + (pred ? gn : 0);
    tc_cp_async16(tc_smem_u32(sdst + row * BN + ((ch ^ (row & 7)) << 3)), src, pred);
  }
}

/// Grouped tile order (8 row tiles per group) so neighbouring CTAs share B tiles in L2.
__device__ __forceinline__ void tc_tile_map(int pid, int tiles_m, int tiles_n, int* tm, int* tn) {
  constexpr int G = 8;
  const int width = G * tiles_n;
  const int first_m = (pid / width) * G;
  const int gsz = min(tiles_m - first_m, G);
  *tm = first_m + (pid % width) % gsz;
  *tn = (pid % width) / gsz;
}

__device__ __forceinline__ float tc_glu_runtime(ActFn fn, float gate, float up, const GluParams& p) {
  switch (fn) {
    case ActFn::kSiLU: return apply_glu<ActFn::kSiLU>(gate, up, p);
    case ActFn::kGeLUErf: return apply_glu<ActFn::kGeLUErf>(gate, up, p);
    case ActFn::kGeLUTanh: return apply_glu<ActFn::kGeLUTanh>(gate, up, p);
    case ActFn::kQuickGeLU: return apply_glu<ActFn::kQuickGeLU>(gate, up, p);
    case ActFn::kReLU: return apply_glu<ActFn::kReLU>(gate, up, p);
    case ActFn::kReLU2: return apply_glu<ActFn::kReLU2>(gate, up, p);
    case ActFn::kSigmoid: return apply_glu<ActFn::kSigmoid>(gate, up, p);
    case ActFn::kTanh: return apply_glu<ActFn::kTanh>(gate, up, p);
    case ActFn::kSoftplus: return apply_glu<ActFn::kSoftplus>(gate, up, p);
    case ActFn::kSiTU: return apply_glu<ActFn::kSiTU>(gate, up, p);
    case ActFn::kFatReLU: return apply_glu<ActFn::kFatReLU>(gate, up, p);
    default: return apply_glu<ActFn::kIdentity>(gate, up, p);
  }
}

/// Stores one or two adjacent outputs (m, n), (m, n + 1): packed when the address allows.
template <typename D>
__device__ __forceinline__ void tc_store_pair(D* dp, float v0, float v1, bool has1) {
  D s0, s1{};
  dev::from_f(s0, v0);
  if (has1) dev::from_f(s1, v1);
  if (has1 && (reinterpret_cast<uintptr_t>(dp) & (2 * sizeof(D) - 1)) == 0) {
    struct alignas(2 * sizeof(D)) Two { D a, b; } two{s0, s1};
    *reinterpret_cast<Two*>(dp) = two;
  } else {
    dp[0] = s0;
    if (has1) dp[1] = s1;
  }
}

}  // namespace

/// Compile-time shape of one dual_gemm_kernel instantiation: BM x BN output tile, BK = 32, ST stages,
/// 2 x WN warps. Per stage: A [BM, 32], B0 [BN, 32], B1 [BN, 32]. WN = 4 gives eight warps per CTA
/// (16 warps per SM at two CTAs), which is what the large-M shapes need to reach the tensor pipe.
template <typename T, int BM, int BN, int ST, int WN = 2>
struct DualTileTraits {
  using a_type = T;
  static constexpr int tile_m = BM, tile_n = BN, tile_k = 32, stages = ST;
  static constexpr int warps_m = 2, warps_n = WN, threads = 64 * WN;
  static constexpr size_t smem_bytes = static_cast<size_t>(ST) * (BM + 2 * BN) * 32 * sizeof(T);
  static_assert(BM % 32 == 0 && BN % (WN * 16) == 0, "warp tile: m16 x (two n8 per ldmatrix)");
};

template <typename Traits>
__global__ void __launch_bounds__(Traits::threads) dual_gemm_kernel(const __grid_constant__ DualGemmParams p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using T = typename Traits::a_type;
  using Tc = dev::TcFrag<T>;
  constexpr int BM = Traits::tile_m, BN = Traits::tile_n, ST = Traits::stages;
  constexpr int WN = Traits::warps_n;
  constexpr int MT = BM / 32;        // m16 tiles per warp (two warp rows)
  constexpr int NT = BN / (WN * 8);  // n8 tiles per warp
  constexpr int kThreads = Traits::threads;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* const sA = reinterpret_cast<T*>(smem_raw);
  T* const sB0 = sA + ST * BM * 32;
  T* const sB1 = sB0 + ST * BN * 32;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int M = p.m, N = p.n, K = p.k;
  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n = (N + BN - 1) / BN;
  int tm, tn;
  tc_tile_map(blockIdx.x, tiles_m, tiles_n, &tm, &tn);
  const int row0 = tm * BM;
  const int col0 = tn * BN;
  const T* const gA = static_cast<const T*>(p.a);
  const T* const gB0 = static_cast<const T*>(p.b0);
  const T* const gB1 = static_cast<const T*>(p.b1);

  const int nk = (K + 31) / 32;
  auto load_stage = [&](int stage, int kt) {
    tc_load_kmajor<T, BM, kThreads>(sA + stage * BM * 32, gA, p.lda, M, row0, kt * 32, K, tid);
    tc_load_kmajor<T, BN, kThreads>(sB0 + stage * BN * 32, gB0, p.ldb0, N, col0, kt * 32, K, tid);
    tc_load_kmajor<T, BN, kThreads>(sB1 + stage * BN * 32, gB1, p.ldb1, N, col0, kt * 32, K, tid);
  };

  float acc0[MT][NT][4];
  float acc1[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i) {
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc0[i][j][q] = 0.0f;
        acc1[i][j][q] = 0.0f;
      }
    }
  }
  #pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nk) load_stage(s, s);
    tc_cp_async_commit();
  }

  const int warp_r0 = (warp / WN) * (BM / 2);
  const int warp_c0 = (warp % WN) * (BN / WN);

  for (int it = 0; it < nk; ++it) {
    tc_cp_async_wait<ST - 2>();
    __syncthreads();
    const int nxt = it + ST - 1;
    if (nxt < nk) load_stage(nxt % ST, nxt);
    tc_cp_async_commit();

    const T* const a_st = sA + (it % ST) * BM * 32;
    const T* const b0_st = sB0 + (it % ST) * BN * 32;
    const T* const b1_st = sB1 + (it % ST) * BN * 32;
    #pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[MT][4];
      uint32_t b0f[NT][2];
      uint32_t b1f[NT][2];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = warp_r0 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kk * 2 + (lane >> 4);
        tc_ldmatrix_x4(af[mt], tc_smem_u32(a_st + row * 32 + ((ch ^ ((row >> 1) & 3)) << 3)));
      }
      #pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        const int row = warp_c0 + np * 16 + (lane & 7) + (lane >> 4) * 8;
        const int ch = kk * 2 + ((lane >> 3) & 1);
        const int off = row * 32 + ((ch ^ ((row >> 1) & 3)) << 3);
        uint32_t r[4];
        tc_ldmatrix_x4(r, tc_smem_u32(b0_st + off));
        b0f[2 * np][0] = r[0]; b0f[2 * np][1] = r[1]; b0f[2 * np + 1][0] = r[2]; b0f[2 * np + 1][1] = r[3];
        tc_ldmatrix_x4(r, tc_smem_u32(b1_st + off));
        b1f[2 * np][0] = r[0]; b1f[2 * np][1] = r[1]; b1f[2 * np + 1][0] = r[2]; b1f[2 * np + 1][1] = r[3];
      }
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          Tc::mma(acc0[mt][nt], af[mt], b0f[nt][0], b0f[nt][1]);
          Tc::mma(acc1[mt][nt], af[mt], b1f[nt][0], b1f[nt][1]);
        }
      }
    }
  }
  tc_cp_async_wait<0>();

  // ---- epilogue: gate and up of one output stay together; stage both through shared memory ----
  constexpr int PASS_ROWS = BM / 2;
  constexpr int LDC = BN + 4;
  static_assert(2u * PASS_ROWS * LDC * sizeof(float) <= Traits::smem_bytes, "epilogue staging fits the pipeline buffers");
  float* const sG = reinterpret_cast<float*>(smem_raw);
  float* const sU = sG + PASS_ROWS * LDC;
  const int g = lane >> 2;
  const int t = lane & 3;
  const EpilogueParams& e = p.epi;
  const float alpha = e.alpha_ptr != nullptr ? *e.alpha_ptr : e.alpha;
  T* const dbase = static_cast<T*>(p.d);
  __syncthreads();
  for (int wi = 0; wi < 2; ++wi) {
    if ((warp / WN) == wi) {
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int r = mt * 16 + g;
          const int c = warp_c0 + nt * 8 + 2 * t;
          *reinterpret_cast<float2*>(&sG[r * LDC + c]) = make_float2(acc0[mt][nt][0], acc0[mt][nt][1]);
          *reinterpret_cast<float2*>(&sG[(r + 8) * LDC + c]) = make_float2(acc0[mt][nt][2], acc0[mt][nt][3]);
          *reinterpret_cast<float2*>(&sU[r * LDC + c]) = make_float2(acc1[mt][nt][0], acc1[mt][nt][1]);
          *reinterpret_cast<float2*>(&sU[(r + 8) * LDC + c]) = make_float2(acc1[mt][nt][2], acc1[mt][nt][3]);
        }
      }
    }
    __syncthreads();
    const int per_row = BN / 2;
    for (int idx = tid; idx < PASS_ROWS * per_row; idx += kThreads) {
      const int r = idx / per_row;
      const int c = (idx - r * per_row) * 2;
      const int64_t m = row0 + wi * PASS_ROWS + r;
      const int64_t n = col0 + c;
      if (m >= M || n >= N) continue;
      const bool has1 = n + 1 < N;
      float out[2];
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        float gate = sG[r * LDC + c + j] * alpha;
        float up = sU[r * LDC + c + j] * alpha;
        if (e.ops & kEpiRowScale) {
          const float rs = e.row_scale[m];
          gate *= rs;
          up *= rs;
        }
        if (e.ops & kEpiClamp) {
          gate = fminf(fmaxf(gate, -e.clamp_limit), e.clamp_limit);
          up = fminf(fmaxf(up, -e.clamp_limit), e.clamp_limit);
        }
        if (e.ops & kEpiSoftcap) {
          gate = softcap(gate, e.softcap);
          up = softcap(up, e.softcap);
        }
        float v = tc_glu_runtime(p.act, gate, up, p.glu);
        if (e.ops & kEpiRoutingWeight) v *= e.routing_weight[m];
        if (e.ops & kEpiResidualAdd) v = fmaf(e.beta, dev::to_f(static_cast<const T*>(e.residual)[m * e.residual_stride + n + j]), v);
        out[j] = v;
        if (!has1) break;
      }
      tc_store_pair<T>(dbase + m * p.ldd + n, out[0], has1 ? out[1] : 0.0f, has1);
    }
    __syncthreads();
  }
#else
  (void)p;
  if (threadIdx.x == 0) __trap();
#endif
}

/// bmm_kernel: BM x BN output tile of one batch (blockIdx.z), BK = 32, 2 x 2 warps. B is read as
/// [N, K] (b_k_major) or as [K, N] through ldmatrix.trans; the choice is a uniform runtime branch.
template <typename a_t, typename b_t, typename d_t, int TILE_M, int TILE_N, int TILE_K, typename ArchTag>
__global__ void __launch_bounds__(128) bmm_kernel(const __grid_constant__ BmmParams p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  static_assert(TILE_K == 32 && TILE_M % 32 == 0 && TILE_N % 32 == 0, "bmm tiles: BK 32, 2 x 2 warps");
  static_assert(sizeof(a_t) == sizeof(b_t) && sizeof(a_t) == 2, "bmm: 16-bit operands");
  using Tc = dev::TcFrag<a_t>;
  constexpr int BM = TILE_M, BN = TILE_N;
  constexpr int ST = BM * BN >= 128 * 128 ? 3 : 4;
  constexpr int MT = BM / 32;
  constexpr int NT = BN / 16;
  constexpr int kThreads = 128;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  a_t* const sA = reinterpret_cast<a_t*>(smem_raw);
  b_t* const sB = reinterpret_cast<b_t*>(sA + ST * BM * 32);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int M = p.m, N = p.n, K = p.k;
  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n = (N + BN - 1) / BN;
  int tm, tn;
  tc_tile_map(blockIdx.x, tiles_m, tiles_n, &tm, &tn);
  const int row0 = tm * BM;
  const int col0 = tn * BN;
  const int64_t z = blockIdx.z;
  const a_t* const gA = static_cast<const a_t*>(p.a) + z * p.stride_a_batch;
  const b_t* const gB = static_cast<const b_t*>(p.b) + z * p.stride_b_batch;
  const bool kmajor = p.b_k_major;

  const int nk = (K + 31) / 32;
  auto load_stage = [&](int stage, int kt) {
    tc_load_kmajor<a_t, BM, kThreads>(sA + stage * BM * 32, gA, p.lda, M, row0, kt * 32, K, tid);
    if (kmajor) tc_load_kmajor<b_t, BN, kThreads>(sB + stage * BN * 32, gB, p.ldb, N, col0, kt * 32, K, tid);
    else tc_load_nmajor<b_t, BN, kThreads>(sB + stage * BN * 32, gB, p.ldb, N, col0, kt * 32, K, tid);
  };

  float acc[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i) {
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
      #pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.0f;
    }
  }
  #pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nk) load_stage(s, s);
    tc_cp_async_commit();
  }

  const int warp_r0 = (warp >> 1) * (BM / 2);
  const int warp_c0 = (warp & 1) * (BN / 2);

  for (int it = 0; it < nk; ++it) {
    tc_cp_async_wait<ST - 2>();
    __syncthreads();
    const int nxt = it + ST - 1;
    if (nxt < nk) load_stage(nxt % ST, nxt);
    tc_cp_async_commit();

    const a_t* const a_st = sA + (it % ST) * BM * 32;
    const b_t* const b_st = sB + (it % ST) * BN * 32;
    #pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[MT][4];
      uint32_t bf[NT][2];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = warp_r0 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kk * 2 + (lane >> 4);
        tc_ldmatrix_x4(af[mt], tc_smem_u32(a_st + row * 32 + ((ch ^ ((row >> 1) & 3)) << 3)));
      }
      #pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        uint32_t r[4];
        if (kmajor) {
          const int row = warp_c0 + np * 16 + (lane & 7) + (lane >> 4) * 8;
          const int ch = kk * 2 + ((lane >> 3) & 1);
          tc_ldmatrix_x4(r, tc_smem_u32(b_st + row * 32 + ((ch ^ ((row >> 1) & 3)) << 3)));
        } else {
          // matrices: (k 0-7, n 0-7) (k 8-15, n 0-7) (k 0-7, n 8-15) (k 8-15, n 8-15), transposed on load
          const int j = lane >> 3;
          const int krow = kk * 16 + (j & 1) * 8 + (lane & 7);
          const int ch = (warp_c0 + np * 16 + (j >> 1) * 8) >> 3;
          tc_ldmatrix_x4_trans(r, tc_smem_u32(b_st + krow * BN + ((ch ^ (krow & 7)) << 3)));
        }
        bf[2 * np][0] = r[0]; bf[2 * np][1] = r[1]; bf[2 * np + 1][0] = r[2]; bf[2 * np + 1][1] = r[3];
      }
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) Tc::mma(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
      }
    }
  }
  tc_cp_async_wait<0>();

  const int g = lane >> 2;
  const int t = lane & 3;
  d_t* const dbase = static_cast<d_t*>(p.d) + z * p.stride_d_batch;
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t m = row0 + warp_r0 + mt * 16 + g + h * 8;
        const int64_t n = col0 + warp_c0 + nt * 8 + 2 * t;
        if (m < M && n < N) tc_store_pair<d_t>(dbase + m * p.ldd + n, acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1], n + 1 < N);
      }
    }
  }
#else
  (void)p;
  if (threadIdx.x == 0) __trap();
#endif
}

// ---- host: dual_gemm / bmm --------------------------------------------------------------------------
namespace {

template <typename T, int BM, int BN, int ST, int WN = 2>
Status launch_dual_tile(const DualGemmParams& p, cudaStream_t s) {
  using Tr = DualTileTraits<T, BM, BN, ST, WN>;
  static const cudaError_t attr =
      cudaFuncSetAttribute(dual_gemm_kernel<Tr>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Tr::smem_bytes));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "dual_gemm_kernel: %s", cudaGetErrorString(attr));
  }
  const int64_t tiles = static_cast<int64_t>((p.m + BM - 1) / BM) * ((p.n + BN - 1) / BN);
  if (tiles > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "dual_gemm: too many tiles");
  dual_gemm_kernel<Tr><<<static_cast<unsigned>(tiles), Tr::threads, Tr::smem_bytes, s>>>(p);
  return launch_error("dual_gemm_kernel");
}

template <typename T>
Status launch_dual(const DualGemmParams& p, cudaStream_t s) {
  // The 64-column N tile leaves only n / 64 CTAs: at M <= 128 that is half the part, and the run is a
  // single pass over the two weight matrices (64 MB at 4096 x 4096) with no compute to hide it. The
  // 32-column tile doubles the grid and halves the weight bytes per CTA, so every SM gets a CTA and
  // the weight traffic per pass stays the same. From M = 65 the 32 x 64 tile runs eight warps per CTA
  // (two CTAs per SM), and from M = 129 the 128 x 64 tile keeps the same 16 warps per SM.
  if (p.m <= 32) return launch_dual_tile<T, 32, 32, 8>(p, s);
  if (p.m <= 64) return launch_dual_tile<T, 32, 64, 4, 4>(p, s);
  if (p.m <= 128) return launch_dual_tile<T, 128, 32, 8>(p, s);
  // The eight-warp variant only pays off while the two weight matrices stay L2-resident; a wide N
  // streams them from DRAM, where the extra A-fragment traffic of the narrower warp tile costs more
  // than the extra warps win (measured 1024 x 14336: 2.4 ms at WN = 4 vs 1.7 ms at WN = 2).
  if (p.n <= 8192) return launch_dual_tile<T, 128, 64, 3, 4>(p, s);
  return launch_dual_tile<T, 128, 64, 3>(p, s);
}

template <typename T, typename D, int TM, int TN>
Status launch_bmm_tile(const BmmParams& p, cudaStream_t s) {
  constexpr int ST = TM * TN >= 128 * 128 ? 3 : 4;
  constexpr size_t smem = static_cast<size_t>(ST) * (TM + TN) * 32 * sizeof(T);
  static const cudaError_t attr = cudaFuncSetAttribute(bmm_kernel<T, T, D, TM, TN, 32, Sm80>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "bmm_kernel: %s", cudaGetErrorString(attr));
  }
  const int64_t tiles = static_cast<int64_t>((p.m + TM - 1) / TM) * ((p.n + TN - 1) / TN);
  if (tiles > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "bmm: too many tiles");
  // grid.z holds the batch (at most 65535 per launch)
  constexpr int32_t kMaxZ = 65535;
  for (int32_t b0 = 0; b0 < p.batch; b0 += kMaxZ) {
    BmmParams q = p;
    const int32_t nb = std::min(kMaxZ, p.batch - b0);
    q.a = static_cast<const T*>(p.a) + static_cast<int64_t>(b0) * p.stride_a_batch;
    q.b = static_cast<const T*>(p.b) + static_cast<int64_t>(b0) * p.stride_b_batch;
    q.d = static_cast<D*>(p.d) + static_cast<int64_t>(b0) * p.stride_d_batch;
    q.batch = nb;
    bmm_kernel<T, T, D, TM, TN, 32, Sm80><<<dim3(static_cast<unsigned>(tiles), 1u, static_cast<unsigned>(nb)), 128, smem, s>>>(q);
    Status st = launch_error("bmm_kernel");
    if (st != Status::kSuccess) return st;
  }
  return Status::kSuccess;
}

template <typename T, typename D>
Status launch_bmm(const BmmParams& p, cudaStream_t s) {
  if (p.m <= 32) return launch_bmm_tile<T, D, 32, 64>(p, s);
  if (p.m <= 64 || p.n < 128) return launch_bmm_tile<T, D, 64, 64>(p, s);
  if (p.m >= 128) return launch_bmm_tile<T, D, 128, 128>(p, s);
  return launch_bmm_tile<T, D, 64, 128>(p, s);
}

}  // namespace

Status dual_gemm(const DualGemmParams& params, DType dtype, const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "dual_gemm");
  if (st != Status::kSuccess) return st;
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "dual_gemm: the tensor-core path needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  if (dtype == DType::kFloat8E4M3 || dtype == DType::kFloat8E5M2) {
    return detail::set_error(Status::kNotImplemented, "dual_gemm: FP8 operands (scaled dual GEMM) are not implemented");
  }
  if (dtype != DType::kBFloat16 && dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "dual_gemm: bf16 or fp16 only");
  }
  if (params.a_scale != nullptr || params.b0_scale != nullptr || params.b1_scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "dual_gemm: scale pointers are for FP8 operands; pass null for bf16 / fp16");
  }
  const EpilogueParams& e = params.epi;
  constexpr uint32_t kAllowed = kEpiRowScale | kEpiSoftcap | kEpiAct | kEpiGLU | kEpiRoutingWeight | kEpiResidualAdd | kEpiClamp;
  if ((e.ops & kEpiBias) != 0u) {
    return detail::set_error(Status::kNotImplemented, "dual_gemm: bias0 / bias1 are unsupported; fold them into the weights");
  }
  if ((e.ops & ~kAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "dual_gemm: epilogue ops 0x%x are not implemented", e.ops & ~kAllowed);
  }
  if (((e.ops & kEpiRowScale) != 0u && e.row_scale == nullptr) || ((e.ops & kEpiRoutingWeight) != 0u && e.routing_weight == nullptr) ||
      ((e.ops & kEpiResidualAdd) != 0u && e.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "dual_gemm: epilogue operand missing (residual is read in the output dtype)");
  }
  if (params.m < 1 || params.n < 1 || params.k < 1) return detail::set_error(Status::kInvalidArgument, "dual_gemm: empty problem");
  if (params.a == nullptr || params.b0 == nullptr || params.b1 == nullptr || params.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "dual_gemm: null operand");
  }
  if (params.lda < params.k || params.ldb0 < params.k || params.ldb1 < params.k || params.ldd < params.n ||
      ((e.ops & kEpiResidualAdd) != 0u && e.residual_stride < params.n)) {
    return detail::set_error(Status::kInvalidArgument, "dual_gemm: leading dimensions smaller than the extents");
  }
  if (!aligned(params.a, 16) || !aligned(params.b0, 16) || !aligned(params.b1, 16) || (params.lda * 2) % 16 != 0 ||
      (params.ldb0 * 2) % 16 != 0 || (params.ldb1 * 2) % 16 != 0 || params.k % 8 != 0) {
    return detail::set_error(Status::kMisaligned, "dual_gemm: operands need 16 B alignment (pointers, ld) and K %% 8 == 0");
  }
  const cudaStream_t s = stream_of(ctx);
  return dtype == DType::kBFloat16 ? launch_dual<__nv_bfloat16>(params, s) : launch_dual<__half>(params, s);
}

Status bmm(const BmmParams& params, DType a_dtype, DType b_dtype, DType d_dtype, const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "bmm");
  if (st != Status::kSuccess) return st;
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "bmm: the tensor-core path needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  if (a_dtype == DType::kFloat8E4M3 || a_dtype == DType::kFloat8E5M2 || b_dtype == DType::kFloat8E4M3 || b_dtype == DType::kFloat8E5M2) {
    return detail::set_error(Status::kNotImplemented, "bmm: FP8 operands belong to fp8_bmm");
  }
  if ((a_dtype != DType::kBFloat16 && a_dtype != DType::kFloat16) || a_dtype != b_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "bmm: A and B must both be bf16 or both fp16");
  }
  if (d_dtype != a_dtype && d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "bmm: the output is the operand dtype or fp32");
  }
  if (params.a_scale != nullptr || params.b_scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "bmm: scales are for FP8 operands; pass null for bf16 / fp16");
  }
  if (params.m < 1 || params.n < 1 || params.k < 1 || params.batch < 1) return detail::set_error(Status::kInvalidArgument, "bmm: empty problem");
  if (params.a == nullptr || params.b == nullptr || params.d == nullptr) return detail::set_error(Status::kInvalidArgument, "bmm: null operand");
  if (params.lda < params.k || params.ldb < (params.b_k_major ? params.k : params.n) || params.ldd < params.n ||
      params.stride_a_batch < 0 || params.stride_b_batch < 0 || params.stride_d_batch < 0) {
    return detail::set_error(Status::kInvalidArgument, "bmm: leading dimensions smaller than the extents or negative batch strides");
  }
  if (params.batch > 1 && params.stride_d_batch == 0) return detail::set_error(Status::kInvalidArgument, "bmm: batched output needs stride_d_batch");
  if (!aligned(params.a, 16) || !aligned(params.b, 16) || (params.lda * 2) % 16 != 0 || (params.ldb * 2) % 16 != 0 ||
      (params.stride_a_batch * 2) % 16 != 0 || (params.stride_b_batch * 2) % 16 != 0 || params.k % 8 != 0 ||
      (!params.b_k_major && params.n % 8 != 0)) {
    return detail::set_error(Status::kMisaligned,
                             "bmm: A / B need 16 B alignment (pointers, ld, batch strides), K %% 8 == 0 (and N %% 8 == 0 when B is [K, N])");
  }
  const cudaStream_t s = stream_of(ctx);
  const bool bf = a_dtype == DType::kBFloat16;
  if (d_dtype == DType::kFloat32) return bf ? launch_bmm<__nv_bfloat16, float>(params, s) : launch_bmm<__half, float>(params, s);
  return bf ? launch_bmm<__nv_bfloat16, __nv_bfloat16>(params, s) : launch_bmm<__half, __half>(params, s);
}

}  // namespace kops::gemm
