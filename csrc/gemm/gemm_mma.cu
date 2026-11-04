// =============================================================================
// kernel-ops | csrc/gemm/gemm_mma.cu
// B7 row 1: bf16 / fp16 dense GEMM on mma.sync (SM80/86/89/120), the shared GEMM frame.
//   gemm_mma_sync_kernel   cp.async multistage + ldmatrix + mma.sync.m16n8k16, XOR-swizzled smem,
//                          swap_ab, split-K (serial = last CTA of a tile sums the slices in slice order),
//                          every EpilogueOp except kEpiQuantOut and kEpiAzpCorrection
//   tactics                get_tactics / autotune / tactic_cache_{lookup,save,load}
// Host entries used by dense_gemm.cu: mma_gemm, mma_selects, mma_workspace (declared in ops.h).
//
// Contract: csrc/gemm/dense_gemm_kernels.cuh (gemm_mma_sync_kernel), csrc/gemm/ops.h, CONTRACTS.md.
//
// Numerics.  A and B are bf16 or fp16, products and accumulation are fp32 inside mma.sync.  For one
// output element the K chain is the same sequence of k16 steps in every tile shape, so a tactic with
// the same split-K slices gives the same rounding of the accumulator in any tile; the epilogue is fp32
// and rounds once to the output dtype.  Split-K sums the slice partials in slice order (0..S-1), never
// by arrival order, and there are no atomics unless the caller asks for kEpiAtomicAdd.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

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

// ---- PTX ---------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// 16-byte cp.async (L2 only); pred == false zero-fills the 16 bytes without reading `src`.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

/// 16-byte chunk index XOR: 64 B rows (BK = 32) swap on row/2, 128 B rows (BK = 64) on row; both keep
/// the eight rows of an ldmatrix on eight distinct 16-byte bank groups.
template <int BK>
__device__ __forceinline__ int swz(int row) {
  return BK == 32 ? ((row >> 1) & 3) : (row & 7);
}

// ---- Epilogue ----------------------------------------------------------------------------------
__device__ __forceinline__ float glu_runtime(ActFn fn, float gate, float up, const GluParams& p) {
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

/// acc -> *alpha -> *row_scale -> *col_scale -> +bias -> clamp -> softcap (everything before act / GLU).
__device__ __forceinline__ float epi_linear(float v, const EpilogueParams& e, int64_t m, int64_t n) {
  const uint32_t ops = e.ops;
  v *= e.alpha_ptr != nullptr ? *e.alpha_ptr : e.alpha;
  if (ops & kEpiRowScale) v *= e.row_scale[m];
  if (ops & kEpiColScale) v *= e.col_scale[n];
  if (ops & kEpiBias) v += dev::load_as_float(e.bias, e.bias_dtype, n);
  if (ops & kEpiClamp) v = fminf(fmaxf(v, -e.clamp_limit), e.clamp_limit);
  if (ops & kEpiSoftcap) v = softcap(v, e.softcap);
  return v;
}

/// act (no GLU) -> *routing_weight -> sigmoid gate multiply -> + beta * residual, in output coordinates.
template <typename Dt>
__device__ __forceinline__ float epi_tail(float v, const EpilogueParams& e, int64_t m, int64_t oc) {
  const uint32_t ops = e.ops;
  if (ops & kEpiRoutingWeight) v *= e.routing_weight[m];
  if (ops & kEpiSigmoidGateMul) {
    const float gs = dev::to_f(static_cast<const Dt*>(e.gate_src)[m * e.gate_src_stride + oc]);
    v = e.gate_is_acc ? gs * sigmoid(v) : v * sigmoid(gs);
  }
  if (ops & kEpiResidualAdd) v = fmaf(e.beta, dev::to_f(static_cast<const Dt*>(e.residual)[m * e.residual_stride + oc]), v);
  return v;
}

template <typename Dt>
__device__ __forceinline__ void atomic_add_elem(Dt* p, float v);
template <>
__device__ __forceinline__ void atomic_add_elem<float>(float* p, float v) { atomicAdd(p, v); }
template <>
__device__ __forceinline__ void atomic_add_elem<__nv_bfloat16>(__nv_bfloat16* p, float v) { atomicAdd(p, __float2bfloat16(v)); }
template <>
__device__ __forceinline__ void atomic_add_elem<__half>(__half* p, float v) { atomicAdd(p, __float2half(v)); }

/// Finishes and stores one or two adjacent output elements (m, oc) / (m, oc + 1): the post-GLU ops, the
/// store (direct / scattered / atomic), then the KV append of the stored value.
template <typename Dt>
__device__ __forceinline__ void store_pair(const EpilogueParams& e, Dt* dbase, int64_t ldd, int64_t m, int64_t oc,
                                           float v0, float v1, bool has1) {
  const uint32_t ops = e.ops;
  v0 = epi_tail<Dt>(v0, e, m, oc);
  if (has1) v1 = epi_tail<Dt>(v1, e, m, oc + 1);
  int64_t drow = m;
  if (ops & kEpiScatterRows) drow = e.scatter_rows[m];
  Dt s0, s1{};
  dev::from_f(s0, v0);
  if (has1) dev::from_f(s1, v1);
  if (drow >= 0) {
    Dt* p = dbase + drow * ldd + oc;
    if (ops & kEpiAtomicAdd) {
      atomic_add_elem<Dt>(p, v0);
      if (has1) atomic_add_elem<Dt>(p + 1, v1);
    } else if (has1 && (reinterpret_cast<uintptr_t>(p) & (2 * sizeof(Dt) - 1)) == 0) {
      struct alignas(2 * sizeof(Dt)) Two { Dt a, b; } two{s0, s1};
      *reinterpret_cast<Two*>(p) = two;
    } else {
      p[0] = s0;
      if (has1) p[1] = s1;
    }
  }
  if (ops & kEpiKvAppend) {
    const int32_t slot = e.kv_slots[m];
    if (slot >= 0) {
      Dt* kv = static_cast<Dt*>(e.kv_dst) + static_cast<int64_t>(slot) * e.kv_stride;
      if (oc >= e.kv_col_begin && oc < e.kv_col_begin + e.kv_cols) kv[oc - e.kv_col_begin] = s0;
      if (has1 && oc + 1 >= e.kv_col_begin && oc + 1 < e.kv_col_begin + e.kv_cols) kv[oc + 1 - e.kv_col_begin] = s1;
    }
  }
}

/// row_sumsq[m] = sum_k A[m, k]^2 for the rows [m0, m1): one warp per row, 8 elements per lane per step,
/// then the xor butterfly. The order depends on K only.
template <typename T>
__device__ __forceinline__ void row_sumsq_rows(const T* a, int64_t lda, int64_t k, int64_t m0, int64_t m1, float* out,
                                               int warp, int nwarps, int lane) {
  for (int64_t m = m0 + warp; m < m1; m += nwarps) {
    const T* row = a + m * lda;
    float s = 0.0f;
    for (int64_t kk = static_cast<int64_t>(lane) * 8; kk < k; kk += 256) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row + kk);
      const T* h = reinterpret_cast<const T*>(&raw);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float f = dev::to_f(h[i]);
        s = fmaf(f, f, s);
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) out[m] = s;
  }
}

}  // namespace

// =============================================================================
// Traits and kernel
// =============================================================================

/// BR / BC are the row counts of the mma-A / mma-B operand tiles. Without swap_ab the mma-A operand is
/// the activations (BR = tile_m); with swap_ab it is the weights (BR = tile_n) and the accumulator is
/// D^T, which the epilogue transposes back on the way out.
template <typename T, typename Dt, int BR, int BC, int BK, int ST, int WM, int WN, bool SWAP>
struct MmaTraits {
  using a_type = T;
  using b_type = T;
  using d_type = Dt;
  using accum_type = float;
  static constexpr int tile_m = SWAP ? BC : BR;
  static constexpr int tile_n = SWAP ? BR : BC;
  static constexpr int tile_k = BK;
  static constexpr int cluster_m = 1;
  static constexpr int cluster_n = 1;
  static constexpr int stages = ST;
  static constexpr MainloopSchedule mainloop = MainloopSchedule::kMultistageMmaSync;
  static constexpr EpilogueSchedule epilogue = EpilogueSchedule::kDirectStore;
  static constexpr int br = BR;
  static constexpr int bc = BC;
  static constexpr int warps_m = WM;
  static constexpr int warps_n = WN;
  static constexpr bool swap_ab = SWAP;
  static constexpr int threads = 32 * WM * WN;
  static constexpr size_t smem_bytes = static_cast<size_t>(ST) * (BR + BC) * BK * sizeof(T);
  static_assert(BR % (WM * 16) == 0 && BC % (WN * 16) == 0, "warp tile: m16 x (two n8 per ldmatrix)");
  static_assert(BK == 32 || BK == 64, "swizzle is defined for 64 B and 128 B rows");
};

template <typename Traits>
__global__ void __launch_bounds__(Traits::threads) gemm_mma_sync_kernel(const __grid_constant__ ScaledGemmArgs args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using T = typename Traits::a_type;
  using Dt = typename Traits::d_type;
  using Tc = dev::TcFrag<T>;
  constexpr int BR = Traits::br, BC = Traits::bc, BK = Traits::tile_k, ST = Traits::stages;
  constexpr int WM = Traits::warps_m, WN = Traits::warps_n;
  constexpr bool SWAP = Traits::swap_ab;
  constexpr int MT = BR / WM / 16;
  constexpr int NT = BC / WN / 8;
  constexpr int kThreads = Traits::threads;
  constexpr int CH = BK / 8;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* const sA = reinterpret_cast<T*>(smem_raw);
  T* const sB = sA + ST * BR * BK;
  __shared__ int s_last;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t M = args.m, N = args.n, K = args.k;
  const int rows_total = static_cast<int>(SWAP ? N : M);
  const int cols_total = static_cast<int>(SWAP ? M : N);
  const int tiles_r = (rows_total + BR - 1) / BR;
  const int tiles_c = (cols_total + BC - 1) / BC;
  const int tiles_m = SWAP ? tiles_c : tiles_r;
  const int tiles_n = SWAP ? tiles_r : tiles_c;

  // ---- tile id -> (tm, tn) ----
  const int pid = blockIdx.x;
  int tm, tn;
  if (args.raster == RasterOrder::kAlongN) {
    tn = pid % tiles_n;
    tm = pid / tiles_n;
  } else if (args.raster == RasterOrder::kAlongM) {
    tm = pid % tiles_m;
    tn = pid / tiles_m;
  } else {
    constexpr int G = 8;
    const int width = G * tiles_n;
    const int first_m = (pid / width) * G;
    const int gsz = min(tiles_m - first_m, G);
    tm = first_m + (pid % width) % gsz;
    tn = (pid % width) / gsz;
  }
  const int tile_r = SWAP ? tn : tm;
  const int tile_c = SWAP ? tm : tn;

  const int slices = max(args.split_k_slices, 1);
  const int zb = blockIdx.z / slices;
  const int slice = blockIdx.z % slices;
  const ScaledOperand& opA = SWAP ? args.b : args.a;
  const ScaledOperand& opB = SWAP ? args.a : args.b;
  const T* const gA = static_cast<const T*>(opA.data) + zb * opA.batch_stride;
  const T* const gB = static_cast<const T*>(opB.data) + zb * opB.batch_stride;
  const int64_t ldA = opA.ld, ldB = opB.ld;

  const int kt_total = static_cast<int>((K + BK - 1) / BK);
  const int kt_per = (kt_total + slices - 1) / slices;
  const int kt_begin = min(slice * kt_per, kt_total);
  const int kt_end = min(kt_begin + kt_per, kt_total);
  const int nk = kt_end - kt_begin;

  constexpr int KS = BK / 16;
  constexpr int R = kThreads / CH;   // rows one cp.async pass of the whole CTA covers
  constexpr int A_IT = BR / R;
  constexpr int B_IT = BC / R;
  static_assert(kThreads % CH == 0 && BR % R == 0 && BC % R == 0 && R % 8 == 0, "cp.async passes tile the operand and keep the swizzle phase");
  static_assert(KS % 2 == 0 && A_IT <= 32 && B_IT <= 32, "fragment double buffering / row masks");
  constexpr uint32_t A_STAGE = BR * BK * sizeof(T);
  constexpr uint32_t B_STAGE = BC * BK * sizeof(T);

  const int row0 = tile_r * BR;
  const int col0 = tile_c * BC;
  const int warp_r0 = (warp / WN) * (BR / WM);
  const int warp_c0 = (warp % WN) * (BC / WN);

  // ---- loader state hoisted out of the K loop: one smem offset, one global pointer per operand, row masks ----
  const uint32_t sA_u = smem_u32(sA);
  const uint32_t sB_u = smem_u32(sB);
  const int ld_ch = tid % CH;
  const int ld_row = tid / CH;
  const uint32_t ld_soff = static_cast<uint32_t>((ld_row * BK + ((ld_ch ^ swz<BK>(ld_row)) << 3)) * sizeof(T));
  const T* pa = gA + static_cast<int64_t>(row0 + ld_row) * ldA + ld_ch * 8 + static_cast<int64_t>(kt_begin) * BK;
  const T* pb = gB + static_cast<int64_t>(col0 + ld_row) * ldB + ld_ch * 8 + static_cast<int64_t>(kt_begin) * BK;
  const int64_t stepA = static_cast<int64_t>(R) * ldA;
  const int64_t stepB = static_cast<int64_t>(R) * ldB;
  uint32_t vmaskA = 0, vmaskB = 0;
  #pragma unroll
  for (int i = 0; i < A_IT; ++i) vmaskA |= static_cast<uint32_t>(row0 + ld_row + i * R < rows_total) << i;
  #pragma unroll
  for (int i = 0; i < B_IT; ++i) vmaskB |= static_cast<uint32_t>(col0 + ld_row + i * R < cols_total) << i;
  int64_t kld = static_cast<int64_t>(kt_begin) * BK + ld_ch * 8;   // k of this thread's chunk in the next tile to load

  auto issue = [&](uint32_t wsA, uint32_t wsB) {
    const bool kok = kld < K;
    const T* qa = pa;
    #pragma unroll
    for (int i = 0; i < A_IT; ++i) {
      cp_async16(sA_u + wsA + ld_soff + i * R * BK * sizeof(T), qa, kok && ((vmaskA >> i) & 1u));
      qa += stepA;
    }
    const T* qb = pb;
    #pragma unroll
    for (int i = 0; i < B_IT; ++i) {
      cp_async16(sB_u + wsB + ld_soff + i * R * BK * sizeof(T), qb, kok && ((vmaskB >> i) & 1u));
      qb += stepB;
    }
    pa += BK;
    pb += BK;
    kld += BK;
  };

  // ---- fragment addressing: lane-constant parts computed once; rows of one ldmatrix share a swizzle phase ----
  const int sw_l = swz<BK>(lane & 7);
  const uint32_t a_lane = static_cast<uint32_t>((warp_r0 + (lane & 7) + ((lane >> 3) & 1) * 8) * BK * sizeof(T));
  const uint32_t b_lane = static_cast<uint32_t>((warp_c0 + (lane & 7) + (lane >> 4) * 8) * BK * sizeof(T));
  uint32_t a_koff[KS], b_koff[KS];
  #pragma unroll
  for (int kk = 0; kk < KS; ++kk) {
    a_koff[kk] = static_cast<uint32_t>((((kk * 2 + (lane >> 4)) ^ sw_l) << 3) * sizeof(T));
    b_koff[kk] = static_cast<uint32_t>((((kk * 2 + ((lane >> 3) & 1)) ^ sw_l) << 3) * sizeof(T));
  }
  auto load_a = [&](int kk, uint32_t stA, uint32_t (&af)[MT][4]) {
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) ldmatrix_x4(af[mt], sA_u + stA + a_lane + mt * 16 * BK * sizeof(T) + a_koff[kk]);
  };
  auto load_b = [&](int kk, uint32_t stB, uint32_t (&bf)[NT][2]) {
    #pragma unroll
    for (int np = 0; np < NT / 2; ++np) {
      uint32_t r[4];
      ldmatrix_x4(r, sB_u + stB + b_lane + np * 16 * BK * sizeof(T) + b_koff[kk]);
      bf[2 * np][0] = r[0];
      bf[2 * np][1] = r[1];
      bf[2 * np + 1][0] = r[2];
      bf[2 * np + 1][1] = r[3];
    }
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
    if (s < nk) issue(s * A_STAGE, s * B_STAGE);
    cp_async_commit();
  }

  uint32_t af[2][MT][4];
  uint32_t bf[NT][2];
  int rs = 0;
  int ws = ST - 1;
  cp_async_wait<ST - 2>();
  __syncthreads();
  load_a(0, 0, af[0]);

  for (int it = 0; it < nk; ++it) {
    #pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      load_b(kk, rs * B_STAGE, bf);
      if (kk == KS - 1) {
        cp_async_wait<ST - 2>();   // tile it + 1 has landed
        __syncthreads();           // and every warp has its fragments of tile it in registers
        rs = rs + 1 == ST ? 0 : rs + 1;
      }
      load_a((kk + 1) % KS, rs * A_STAGE, af[(kk + 1) & 1]);
      if (kk == 0) {
        if (it + ST - 1 < nk) issue(ws * A_STAGE, ws * B_STAGE);
        cp_async_commit();
        ws = ws + 1 == ST ? 0 : ws + 1;
      }
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) Tc::mma(acc[mt][nt], af[kk & 1][mt], bf[nt][0], bf[nt][1]);
      }
    }
  }
  cp_async_wait<0>();

  // ---- accumulator coordinates: element (mt, nt, i) sits at mma-row r_base + 16 mt + 8 (i >> 1),
  //      mma-col c_base + 8 nt + (i & 1) ----
  const int g = lane >> 2;
  const int t = lane & 3;
  const int r_base = row0 + warp_r0 + g;
  const int c_base = col0 + warp_c0 + 2 * t;

  // ---- split-K ----
  if (slices > 1) {
    float* const ws = static_cast<float*>(args.workspace) + static_cast<int64_t>(slice) * M * N;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = r_base + mt * 16 + (i >> 1) * 8;
          const int c = c_base + nt * 8 + (i & 1);
          const int64_t m = SWAP ? c : r;
          const int64_t n = SWAP ? r : c;
          if (m < M && n < N) ws[m * N + n] = acc[mt][nt][i];
        }
      }
    }
    if (args.split_k_mode != SplitKMode::kSerial) return;   // kParallel: splitk_reduce_kernel finishes
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int prev = atomicAdd(&args.split_k_counters[tm * tiles_n + tn], 1);
      s_last = prev == slices - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const float* const ws0 = static_cast<const float*>(args.workspace);
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = r_base + mt * 16 + (i >> 1) * 8;
          const int c = c_base + nt * 8 + (i & 1);
          const int64_t m = SWAP ? c : r;
          const int64_t n = SWAP ? r : c;
          float s = 0.0f;
          if (m < M && n < N) {
            for (int sl = 0; sl < slices; ++sl) s += __ldcg(ws0 + static_cast<int64_t>(sl) * M * N + m * N + n);
          }
          acc[mt][nt][i] = s;
        }
      }
    }
  }

  // ---- epilogue: the fp32 accumulators go through shared memory one warp-row group at a time, then one
  //      (non-unrolled) loop finishes and stores them with coalesced access. One copy of the epilogue
  //      code per kernel instead of one per accumulator register. ----
  const EpilogueParams& e = args.epi;
  const bool glu = (e.ops & kEpiGLU) != 0u;
  const int db = e.glu_interleave > 0 ? e.glu_interleave : 1;
  Dt* const dbase = static_cast<Dt*>(args.d) + static_cast<int64_t>(zb) * args.batch_stride_d;
  constexpr int PASS_ROWS = BR / WM;                // mma rows of one pass
  constexpr int LDC = BC + 4;
  constexpr int LR = SWAP ? BC : PASS_ROWS;         // logical rows (m) of one pass
  constexpr int LC = SWAP ? PASS_ROWS : BC;         // logical columns (n) of one pass
  static_assert(static_cast<size_t>(PASS_ROWS) * LDC * sizeof(float) <= Traits::smem_bytes, "epilogue staging fits the pipeline buffers");
  float* const sC = reinterpret_cast<float*>(smem_raw);
  __syncthreads();   // the pipeline buffers are dead

  for (int wi = 0; wi < WM; ++wi) {
    if (warp / WN == wi) {
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int r = mt * 16 + g;
          const int c = warp_c0 + nt * 8 + 2 * t;
          *reinterpret_cast<float2*>(&sC[r * LDC + c]) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
          *reinterpret_cast<float2*>(&sC[(r + 8) * LDC + c]) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
      }
    }
    __syncthreads();
    const int64_t m0 = SWAP ? col0 : row0 + wi * PASS_ROWS;
    const int64_t n0 = SWAP ? row0 + wi * PASS_ROWS : col0;
    const int lco = glu ? LC / 2 : LC;              // output columns of the pass
    const int vec = glu ? 1 : 2;
    const int per_row = lco / vec;
    const int items = LR * per_row;
    for (int idx = tid; idx < items; idx += kThreads) {
      const int r = idx / per_row;
      const int cv = (idx - r * per_row) * vec;     // first output column (local)
      const int64_t m = m0 + r;
      if (m >= M) continue;
      auto at = [&](int rr, int cc) { return SWAP ? sC[cc * LDC + rr] : sC[rr * LDC + cc]; };
      if (glu) {
        // blocks of db gate columns, then db up columns; n0 is a multiple of 2 db, so both sit in this pass
        const int lc = (cv / db) * 2 * db + cv % db;
        const int64_t gcol = n0 + lc;
        if (gcol >= N) continue;
        const float gate = epi_linear(at(r, lc), e, m, gcol);
        const float up = epi_linear(at(r, lc + db), e, m, gcol + db);
        store_pair<Dt>(e, dbase, args.ldd, m, n0 / 2 + cv, glu_runtime(e.act, gate, up, e.glu), 0.0f, false);
      } else {
        const int64_t n = n0 + cv;
        if (n >= N) continue;
        const bool has1 = n + 1 < N;
        float v0 = epi_linear(at(r, cv), e, m, n);
        float v1 = has1 ? epi_linear(at(r, cv + 1), e, m, n + 1) : 0.0f;
        if (e.ops & kEpiAct) {
          v0 = dev::act_runtime(e.act, v0, e.glu);
          v1 = dev::act_runtime(e.act, v1, e.glu);
        }
        store_pair<Dt>(e, dbase, args.ldd, m, n, v0, v1, has1);
      }
    }
    __syncthreads();   // sC is overwritten by the next pass
  }

  // ---- kEpiRowSumSq: the CTAs of the first N tile own the rows of their M tile ----
  if ((e.ops & kEpiRowSumSq) != 0u && (SWAP ? tile_r : tile_c) == 0) {
    const int m_tile = SWAP ? BC : BR;
    const int64_t m0 = static_cast<int64_t>(SWAP ? tile_c : tile_r) * m_tile;
    row_sumsq_rows<T>(static_cast<const T*>(args.a.data), args.a.ld, K, m0, (M < m0 + m_tile ? M : m0 + m_tile), e.row_sumsq,
                      warp, kThreads / 32, lane);
  }
#else
  (void)args;
  if (threadIdx.x == 0) __trap();
#endif
}

namespace {

// =============================================================================
// Host
// =============================================================================

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }

/// Parallel split-K finish: D = dense_epilogue(sum_s partial[s]) in slice order (the dense epilogue subset).
template <typename Dt>
__global__ void mma_splitk_reduce_kernel(Dt* __restrict__ d, const int64_t ldd, const float* __restrict__ partials,
                                         const int num_splits, const int64_t m, const int64_t n, const EpilogueParams epi) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m * n;
  if (e >= total) return;
  float sum = 0.0f;
  for (int i = 0; i < num_splits; ++i) sum += partials[static_cast<int64_t>(i) * total + e];
  const int64_t row = e / n;
  const int64_t col = e - row * n;
  dev::from_f(d[row * ldd + col], dev::gemm_epilogue<Dt>(sum, epi, row, col));
}

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what, cudaGetErrorString(err));
}

template <typename T, typename Dt, int BR, int BC, int BK, int ST, int WM, int WN, bool SWAP>
Status launch_mma(const ScaledGemmArgs& a, cudaStream_t s) {
  using Tr = MmaTraits<T, Dt, BR, BC, BK, ST, WM, WN, SWAP>;
  static const cudaError_t attr =
      cudaFuncSetAttribute(gemm_mma_sync_kernel<Tr>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Tr::smem_bytes));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "gemm_mma_sync_kernel: %s", cudaGetErrorString(attr));
  }
  const int64_t rows = SWAP ? a.n : a.m;
  const int64_t cols = SWAP ? a.m : a.n;
  const int64_t tiles = ((rows + BR - 1) / BR) * ((cols + BC - 1) / BC);
  if (tiles > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "gemm: too many tiles");
  const int slices = std::max(a.split_k_slices, 1);
  const dim3 grid(static_cast<unsigned>(tiles), 1u, static_cast<unsigned>(a.batch * slices));
  gemm_mma_sync_kernel<Tr><<<grid, Tr::threads, Tr::smem_bytes, s>>>(a);
  return launch_error("gemm_mma_sync_kernel");
}

/// Tile table. tm / tn are logical (tm along M, tn along N); a swap_ab tile loads the weights as the
/// mma-A operand.
struct TileCfg {
  int16_t tm, tn;
  int8_t stages;
  bool swap;
  size_t smem;
  int16_t glu_span;   // output columns one epilogue pass covers: kEpiGLU needs span % (2 * glu_interleave) == 0
};
constexpr TileCfg kTiles[] = {
    {128, 128, 3, false, 3 * 256 * 32 * 2, 128},
    {128, 64, 4, false, 4 * 192 * 32 * 2, 64},
    {64, 128, 4, false, 4 * 192 * 32 * 2, 128},
    {64, 64, 4, false, 4 * 128 * 32 * 2, 64},
    {64, 128, 4, true, 4 * 192 * 32 * 2, 64},    // swap: 64 tokens x 128 channels, two passes of 64 channels
    {32, 128, 4, true, 4 * 160 * 32 * 2, 32},    // swap: 32 tokens x 128 channels, four passes of 32 channels
};
constexpr int kNumTiles = static_cast<int>(sizeof(kTiles) / sizeof(kTiles[0]));
constexpr uint64_t kKernelIdBase = 0x4D4D4100u;   // "MMA" + tile id

template <typename T, typename Dt>
Status launch_tile(int id, const ScaledGemmArgs& a, cudaStream_t s) {
  switch (id) {
    case 0: return launch_mma<T, Dt, 128, 128, 32, 3, 2, 2, false>(a, s);
    case 1: return launch_mma<T, Dt, 128, 64, 32, 4, 2, 2, false>(a, s);
    case 2: return launch_mma<T, Dt, 64, 128, 32, 4, 2, 2, false>(a, s);
    case 3: return launch_mma<T, Dt, 64, 64, 32, 4, 2, 2, false>(a, s);
    case 4: return launch_mma<T, Dt, 128, 64, 32, 4, 2, 2, true>(a, s);
    case 5: return launch_mma<T, Dt, 128, 32, 32, 4, 4, 1, true>(a, s);
    default: return detail::set_error(Status::kInternal, "gemm_mma: tile id %d", id);
  }
}

Status launch_by_type(bool bf16, DType d, int id, const ScaledGemmArgs& a, cudaStream_t s) {
  if (bf16) {
    return d == DType::kFloat32 ? launch_tile<__nv_bfloat16, float>(id, a, s) : launch_tile<__nv_bfloat16, __nv_bfloat16>(id, a, s);
  }
  return d == DType::kFloat32 ? launch_tile<__half, float>(id, a, s) : launch_tile<__half, __half>(id, a, s);
}

int tile_of(const GemmTactic& t) {
  for (int i = 0; i < kNumTiles; ++i) {
    if (kTiles[i].tm == t.tile_m && kTiles[i].tn == t.tile_n && kTiles[i].swap == t.swap_ab) return i;
  }
  return -1;
}

GemmTactic make_tactic(int id, SplitKMode mode, int slices) {
  GemmTactic t{};
  t.tile_m = kTiles[id].tm;
  t.tile_n = kTiles[id].tn;
  t.tile_k = 32;
  t.stages = kTiles[id].stages;
  t.mainloop = MainloopSchedule::kMultistageMmaSync;
  t.epilogue = EpilogueSchedule::kDirectStore;
  t.split_k_mode = mode;
  t.split_k_slices = static_cast<int16_t>(slices);
  t.raster = RasterOrder::kHeuristic;
  t.swap_ab = kTiles[id].swap;
  t.kernel_id = kKernelIdBase + static_cast<uint64_t>(id);
  return t;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Largest dynamic shared memory per block the device offers (cached per device).
int smem_limit(int device) {
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

/// Static heuristic: swap_ab for tiny M, otherwise the largest tile that still fills two waves;
/// split-K only when the tile grid cannot fill the GPU and K is long. batch_invariant keeps one slice.
GemmTactic heuristic_tactic(int64_t m, int64_t n, int64_t k, int64_t batch, bool glu, const LaunchContext& ctx) {
  const int64_t sms = ctx.compute_units > 0 ? ctx.compute_units : 16;
  int id = 3;
  if (glu) {
    id = m >= 128 ? 0 : 2;   // GLU wants the 128-column passes (any glu_interleave up to 64)
  } else if (m <= 64 && !ctx.batch_invariant) {
    id = m <= 32 ? 5 : 4;
  } else {
    static constexpr int order[] = {0, 1, 2, 3};
    id = 3;
    for (int o : order) {
      if (ceil_div(m, kTiles[o].tm) * ceil_div(n, kTiles[o].tn) * batch >= 2 * sms) {
        id = o;
        break;
      }
    }
  }
  const int64_t ctas = ceil_div(m, kTiles[id].tm) * ceil_div(n, kTiles[id].tn) * batch;
  int slices = 1;
  if (batch == 1 && !ctx.batch_invariant && ctas < sms && k >= 2048) {
    slices = static_cast<int>(std::min<int64_t>(8, ceil_div(2 * sms, ctas)));
    while (slices > 1 && k / slices < 512) slices >>= 1;
  }
  return make_tactic(id, slices > 1 ? SplitKMode::kSerial : SplitKMode::kNone, slices);
}

// ---- tactic cache --------------------------------------------------------------------------------
struct CacheKey {
  int recipe, arch, cuda;
  int64_t n, k, mbucket;
  bool operator<(const CacheKey& o) const {
    if (recipe != o.recipe) return recipe < o.recipe;
    if (arch != o.arch) return arch < o.arch;
    if (cuda != o.cuda) return cuda < o.cuda;
    if (n != o.n) return n < o.n;
    if (k != o.k) return k < o.k;
    return mbucket < o.mbucket;
  }
};

int64_t m_bucket(int64_t m) {
  int64_t b = 1;
  while (b < m) b <<= 1;
  return b;
}

CacheKey make_key(GemmRecipe recipe, int64_t m, int64_t n, int64_t k, const LaunchContext& ctx) {
  return CacheKey{static_cast<int>(recipe), ctx.arch_id, CUDART_VERSION, n, k, m_bucket(m)};
}

std::mutex& cache_mutex() {
  static std::mutex mu;
  return mu;
}
std::map<CacheKey, GemmTactic>& cache_map() {
  static std::map<CacheKey, GemmTactic> c;
  return c;
}

bool cache_find(const CacheKey& key, GemmTactic* out) {
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto it = cache_map().find(key);
  if (it == cache_map().end()) return false;
  *out = it->second;
  return true;
}

// ---- validation -------------------------------------------------------------------------------------
constexpr uint32_t kMmaEpiAllowed =
    kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct | kEpiGLU | kEpiRoutingWeight | kEpiSigmoidGateMul |
    kEpiResidualAdd | kEpiScatterRows | kEpiAtomicAdd | kEpiClamp | kEpiOutFp32 | kEpiRowSumSq | kEpiKvAppend;
constexpr uint32_t kReducerEpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct | kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status check_mma_epilogue(const ScaledGemmArgs& args, const LaunchContext& ctx, SplitKMode mode, int slices) {
  const EpilogueParams& e = args.epi;
  const uint32_t ops = e.ops;
  if ((ops & kEpiQuantOut) != 0u) {
    return detail::set_error(Status::kNotImplemented, "gemm: kEpiQuantOut is not implemented on the mma.sync path yet");
  }
  if ((ops & kEpiAzpCorrection) != 0u) {
    return detail::set_error(Status::kNotImplemented, "gemm: kEpiAzpCorrection belongs to the INT8 recipe");
  }
  if ((ops & ~kMmaEpiAllowed) != 0u) {
    return detail::set_error(Status::kInvalidArgument, "gemm: unknown epilogue bits 0x%x", ops & ~kMmaEpiAllowed);
  }
  if (((ops & kEpiRowScale) != 0u && e.row_scale == nullptr) || ((ops & kEpiColScale) != 0u && e.col_scale == nullptr) ||
      ((ops & kEpiBias) != 0u && e.bias == nullptr) || ((ops & kEpiResidualAdd) != 0u && e.residual == nullptr) ||
      ((ops & kEpiRoutingWeight) != 0u && e.routing_weight == nullptr) ||
      ((ops & kEpiSigmoidGateMul) != 0u && e.gate_src == nullptr) || ((ops & kEpiScatterRows) != 0u && e.scatter_rows == nullptr) ||
      ((ops & kEpiRowSumSq) != 0u && e.row_sumsq == nullptr) ||
      ((ops & kEpiKvAppend) != 0u && (e.kv_dst == nullptr || e.kv_slots == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "gemm: epilogue operand missing (residual, gate_src and kv_dst are in the output dtype)");
  }
  if ((ops & kEpiBias) != 0u && e.bias_dtype != DType::kBFloat16 && e.bias_dtype != DType::kFloat16 && e.bias_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: bias must be bf16, fp16 or fp32");
  }
  if (((ops & kEpiOutFp32) != 0u) != (args.d_dtype == DType::kFloat32)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: kEpiOutFp32 and an fp32 output go together");
  }
  if ((ops & kEpiAtomicAdd) != 0u) {
    if (ctx.deterministic) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiAtomicAdd is disabled by ctx.deterministic");
    if ((ops & kEpiKvAppend) != 0u) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiKvAppend reads the stored value; it cannot follow an atomic accumulate");
    if (slices > 1) return detail::set_error(Status::kNotImplemented, "gemm: kEpiAtomicAdd with split-K");
  }
  if ((ops & kEpiKvAppend) != 0u && (e.kv_cols < 0 || e.kv_col_begin < 0 || e.kv_stride < e.kv_cols)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: kv_stride must cover kv_cols");
  }
  if ((ops & (kEpiRowSumSq | kEpiKvAppend)) != 0u && args.batch != 1) {
    return detail::set_error(Status::kNotImplemented, "gemm: kEpiRowSumSq / kEpiKvAppend with batch > 1");
  }
  if ((ops & kEpiGLU) != 0u) {
    if (e.glu_layout != GateLayout::kInterleaved) {
      return detail::set_error(Status::kNotImplemented,
                               "gemm: kEpiGLU on the tile GEMM needs GateLayout::kInterleaved (use glu_weight_interleave, or dual_gemm for separate weights)");
    }
    const int db = e.glu_interleave > 0 ? e.glu_interleave : 1;
    if (args.n % (2 * db) != 0) return detail::set_error(Status::kUnsupportedShape, "gemm: kEpiGLU needs N %% (2 * glu_interleave) == 0");
    if (db > 64) return detail::set_error(Status::kUnsupportedShape, "gemm: glu_interleave must be 1..64");
    if (args.ldd < args.n / 2) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiGLU output has N / 2 columns; ldd is smaller");
  } else if (args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "gemm: ldd smaller than N");
  }
  (void)mode;
  return Status::kSuccess;
}

/// Output columns of the epilogue after GLU.
int64_t out_cols(const ScaledGemmArgs& a) { return (a.epi.ops & kEpiGLU) != 0u ? a.n / 2 : a.n; }

/// Tactic -> tile id, with the constraints GLU and the device impose. Returns <0 and sets an error.
Status resolve_tile(const GemmTactic& t, const ScaledGemmArgs& args, const LaunchContext& ctx, int* out_id) {
  const int id = tile_of(t);
  *out_id = id;
  if (id < 0) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: no mma.sync tile %dx%d (swap_ab %d)", t.tile_m, t.tile_n, t.swap_ab ? 1 : 0);
  }
  if (t.stages != 0 && t.stages != kTiles[id].stages) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: tile %dx%d is compiled with %d stages", t.tile_m, t.tile_n, kTiles[id].stages);
  }
  if (kTiles[id].smem > static_cast<size_t>(smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: tile %dx%d needs %zu B of shared memory", t.tile_m, t.tile_n, kTiles[id].smem);
  }
  if ((args.epi.ops & kEpiGLU) != 0u) {
    const int db = args.epi.glu_interleave > 0 ? args.epi.glu_interleave : 1;
    if (kTiles[id].glu_span % (2 * db) != 0) {
      return detail::set_error(Status::kUnsupportedShape, "gemm: glu_interleave %d needs a tile whose epilogue pass covers a multiple of %d columns (tile %dx%d covers %d)",
                               db, 2 * db, t.tile_m, t.tile_n, kTiles[id].glu_span);
    }
  }
  return Status::kSuccess;
}

Status split_workspace_bytes(int64_t m, int64_t n, int64_t tiles, SplitKMode mode, int slices, size_t* partial_bytes, size_t* counter_bytes) {
  *partial_bytes = 0;
  *counter_bytes = 0;
  if (slices <= 1 || mode == SplitKMode::kNone) return Status::kSuccess;
  if (mode != SplitKMode::kSerial && mode != SplitKMode::kParallel) {
    return detail::set_error(Status::kNotImplemented, "gemm: stream-K is not implemented");
  }
  *partial_bytes = static_cast<size_t>(slices) * m * n * sizeof(float);
  if (mode == SplitKMode::kSerial) *counter_bytes = static_cast<size_t>(tiles) * sizeof(int32_t);
  return Status::kSuccess;
}

size_t align256(size_t x) { return (x + 255) & ~static_cast<size_t>(255); }

bool is_dense_recipe(GemmRecipe r) { return r == GemmRecipe::kBF16 || r == GemmRecipe::kFP16; }

}  // namespace

// =============================================================================
// Entry points shared with dense_gemm.cu
// =============================================================================
bool mma_selects(const ScaledGemmArgs& args, const GemmTactic* tactic, const LaunchContext& ctx) {
  if (tactic != nullptr) return tactic->mainloop == MainloopSchedule::kMultistageMmaSync;
  GemmTactic t;
  return cache_find(make_key(args.recipe, args.m, args.n, args.k, ctx), &t);
}

Status mma_workspace(const GemmProblem& p, const GemmTactic* tactic, const LaunchContext& ctx, WorkspaceRequirements* out) {
  GemmTactic t;
  if (tactic != nullptr) {
    t = *tactic;
  } else if (!cache_find(make_key(p.recipe, p.m, p.n, p.k, ctx), &t)) {
    t = heuristic_tactic(p.m, p.n, p.k, p.batch, (p.epilogue_ops & kEpiGLU) != 0u, ctx);
  }
  const int id = tile_of(t);
  if (id < 0) return detail::set_error(Status::kUnsupportedShape, "gemm: no mma.sync tile %dx%d", t.tile_m, t.tile_n);
  const int64_t tiles = ceil_div(p.m, kTiles[id].tm) * ceil_div(p.n, kTiles[id].tn);
  size_t partial = 0, counters = 0;
  Status st = split_workspace_bytes(p.m, p.n, tiles, t.split_k_mode, t.split_k_slices, &partial, &counters);
  if (st != Status::kSuccess) return st;
  out->device_scratch_bytes = align256(counters) + partial;
  return Status::kSuccess;
}

Status mma_gemm(const ScaledGemmArgs& args_in, const GemmTactic* tactic, const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "gemm: the tensor-core path is mma.sync bf16/fp16 and needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  ScaledGemmArgs args = args_in;
  const bool bf16 = args.a.dtype == DType::kBFloat16;
  if (!is_dense_recipe(args.recipe)) return detail::set_error(Status::kNotImplemented, "gemm: recipe %d has no mma.sync body (kTF32 is not implemented)", static_cast<int>(args.recipe));
  if (args.a.dtype != args.b.dtype || (!bf16 && args.a.dtype != DType::kFloat16) ||
      (args.recipe == GemmRecipe::kBF16) != bf16) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync needs bf16 x bf16 (kBF16) or fp16 x fp16 (kFP16)");
  }
  if (args.d_dtype != args.a.dtype && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync output is the operand dtype or fp32");
  }
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kNotImplemented, "gemm: MN-major operands are not implemented on the mma.sync path (weight_transpose first)");
  }
  if (args.m < 1 || args.n < 1 || args.k < 1 || args.batch < 1) return detail::set_error(Status::kInvalidArgument, "gemm: empty problem");
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: null operand");
  if (args.a.ld < args.k || args.b.ld < args.k) return detail::set_error(Status::kInvalidArgument, "gemm: leading dimensions smaller than K");
  if (args.m > 0x7FFFFFFF || args.n > 0x7FFFFFFF || args.k > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "gemm: extents beyond int32");
  if (args.workspace != nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: pass workspace through the Workspace argument");
  if (args.batch > 1 && (args.a.batch_stride == 0 || args.b.batch_stride == 0 || args.batch_stride_d == 0)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: batched problems need batch strides");
  }
  if (args.batch > 65535) return detail::set_error(Status::kUnsupportedShape, "gemm: batch > 65535");
  const int es = 2;
  if (!aligned(args.a.data, 16) || !aligned(args.b.data, 16) || (args.a.ld * es) % 16 != 0 || (args.b.ld * es) % 16 != 0 ||
      (args.a.batch_stride * es) % 16 != 0 || (args.b.batch_stride * es) % 16 != 0 || args.k % 8 != 0) {
    return detail::set_error(Status::kMisaligned, "gemm: mma.sync operands need 16 B alignment (pointers, ld, batch stride) and K %% 8 == 0");
  }

  GemmTactic t;
  if (tactic != nullptr) {
    t = *tactic;
  } else if (!cache_find(make_key(args.recipe, args.m, args.n, args.k, ctx), &t)) {
    t = heuristic_tactic(args.m, args.n, args.k, args.batch, (args.epi.ops & kEpiGLU) != 0u, ctx);
  }
  if (t.mainloop != MainloopSchedule::kMultistageMmaSync) {
    return detail::set_error(Status::kUnsupportedArch, "gemm: tactic mainloop %d needs a newer architecture", static_cast<int>(t.mainloop));
  }
  int id = -1;
  {
    const Status rs = resolve_tile(t, args, ctx, &id);
    if (rs != Status::kSuccess) return rs;
  }

  SplitKMode mode = t.split_k_mode;
  int slices = std::max<int>(t.split_k_slices, 1);
  if (mode == SplitKMode::kNone) slices = 1;
  if (slices == 1) mode = SplitKMode::kNone;
  if (mode != SplitKMode::kNone && mode != SplitKMode::kSerial && mode != SplitKMode::kParallel) {
    return detail::set_error(Status::kNotImplemented, "gemm: stream-K is not implemented");
  }
  if (slices > 1 && args.batch != 1) return detail::set_error(Status::kNotImplemented, "gemm: split-K with batch > 1");
  if (slices > 64) return detail::set_error(Status::kInvalidArgument, "gemm: split_k_slices must be 1..64");
  if (mode == SplitKMode::kParallel && (args.epi.ops & ~kReducerEpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "gemm: parallel split-K supports the dense epilogue subset; use kSerial for the rest");
  }
  Status st = check_mma_epilogue(args, ctx, mode, slices);
  if (st != Status::kSuccess) return st;

  const int64_t tiles = ceil_div(args.m, kTiles[id].tm) * ceil_div(args.n, kTiles[id].tn);
  size_t partial = 0, counters = 0;
  st = split_workspace_bytes(args.m, args.n, tiles, mode, slices, &partial, &counters);
  if (st != Status::kSuccess) return st;
  const cudaStream_t s = stream_of(ctx);
  if (slices > 1) {
    if (ws.device_scratch == nullptr || ws.device_scratch_bytes < align256(counters) + partial) {
      return detail::set_error(Status::kWorkspaceTooSmall, "gemm: split-K needs %zu B of device scratch", align256(counters) + partial);
    }
    unsigned char* base = static_cast<unsigned char*>(ws.device_scratch);
    args.split_k_counters = counters != 0 ? reinterpret_cast<int32_t*>(base) : nullptr;
    args.workspace = base + align256(counters);
    args.workspace_bytes = partial;
    if (counters != 0) {
      const cudaError_t err = cudaMemsetAsync(base, 0, counters, s);
      if (err != cudaSuccess) return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "gemm: cudaMemsetAsync: %s", cudaGetErrorString(err));
    }
  }
  args.split_k_mode = mode;
  args.split_k_slices = slices;
  args.raster = t.raster;

  if (mode == SplitKMode::kParallel) {
    // Partials land in args.workspace ([slice][M][N] fp32); the reducer applies the dense epilogue.
    ScaledGemmArgs part = args;
    part.epi = EpilogueParams{};   // the mainloop kernel only writes partials
    st = launch_by_type(bf16, args.d_dtype, id, part, s);
    if (st != Status::kSuccess) return st;
    const int64_t total = args.m * args.n;
    const unsigned blocks = static_cast<unsigned>(ceil_div(total, 256));
    const float* partials = static_cast<const float*>(args.workspace);
    switch (args.d_dtype) {
      case DType::kFloat32:
        mma_splitk_reduce_kernel<float><<<blocks, 256, 0, s>>>(static_cast<float*>(args.d), args.ldd, partials, slices, args.m, args.n, args.epi);
        break;
      case DType::kFloat16:
        mma_splitk_reduce_kernel<__half><<<blocks, 256, 0, s>>>(static_cast<__half*>(args.d), args.ldd, partials, slices, args.m, args.n, args.epi);
        break;
      default:
        mma_splitk_reduce_kernel<__nv_bfloat16><<<blocks, 256, 0, s>>>(static_cast<__nv_bfloat16*>(args.d), args.ldd, partials, slices, args.m, args.n, args.epi);
        break;
    }
    return launch_error("mma_splitk_reduce_kernel");
  }
  return launch_by_type(bf16, args.d_dtype, id, args, s);
}

// =============================================================================
// Tactics, autotune, cache
// =============================================================================

namespace {

Status native_only(GemmBackend backend, const LaunchContext& ctx, const char* who) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "%s: CUDA backend only", who);
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: only the native provider exists", who);
  }
  return Status::kSuccess;
}

std::vector<GemmTactic> candidates(const GemmProblem& p, const LaunchContext& ctx) {
  std::vector<GemmTactic> out;
  const bool glu = (p.epilogue_ops & kEpiGLU) != 0u;
  const int lim = smem_limit(ctx.device);
  const GemmTactic first = heuristic_tactic(p.m, p.n, p.k, p.batch, glu, ctx);
  out.push_back(first);
  for (int id = 0; id < kNumTiles; ++id) {
    if (kTiles[id].smem > static_cast<size_t>(lim)) continue;
    for (int slices : {1, 2, 4, 8}) {
      if (slices > 1 && (p.batch != 1 || p.k / slices < 512)) continue;
      GemmTactic t = make_tactic(id, slices > 1 ? SplitKMode::kSerial : SplitKMode::kNone, slices);
      if (t.tile_m == first.tile_m && t.tile_n == first.tile_n && t.swap_ab == first.swap_ab && t.split_k_slices == first.split_k_slices) continue;
      out.push_back(t);
    }
  }
  return out;
}

}  // namespace

Status get_tactics(const GemmProblem& problem, GemmBackend backend, const LaunchContext& ctx, GemmTactic* out,
                   int32_t max_tactics, int32_t* num_tactics) {
  Status st = native_only(backend, ctx, "get_tactics");
  if (st != Status::kSuccess) return st;
  if (out == nullptr || num_tactics == nullptr || max_tactics < 0) return detail::set_error(Status::kInvalidArgument, "get_tactics: null output");
  *num_tactics = 0;
  if (!is_dense_recipe(problem.recipe) || problem.m < 1 || problem.n < 1 || problem.k < 1) return Status::kSuccess;
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) return Status::kSuccess;
  const std::vector<GemmTactic> all = candidates(problem, ctx);
  const int32_t n = std::min<int32_t>(max_tactics, static_cast<int32_t>(all.size()));
  for (int32_t i = 0; i < n; ++i) out[i] = all[i];
  *num_tactics = n;
  return Status::kSuccess;
}

Status autotune(const GemmProblem& problem, const int64_t* m_buckets, int32_t num_buckets, GemmBackend backend,
                const Workspace& ws, const LaunchContext& ctx) {
  Status st = native_only(backend, ctx, "autotune");
  if (st != Status::kSuccess) return st;
  if (m_buckets == nullptr || num_buckets < 1) return detail::set_error(Status::kInvalidArgument, "autotune: no M buckets");
  if (!is_dense_recipe(problem.recipe)) return detail::set_error(Status::kNotImplemented, "autotune: only the bf16 / fp16 dense recipes have tactics");
  const cudaStream_t s = stream_of(ctx);
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
    cudaGetLastError();
    return detail::set_error(Status::kInvalidArgument, "autotune: must not run while a CUDA graph is being captured");
  }
  const size_t es = 2;
  const DType dt = problem.recipe == GemmRecipe::kBF16 ? DType::kBFloat16 : DType::kFloat16;
  for (int32_t bi = 0; bi < num_buckets; ++bi) {
    const int64_t m = std::max<int64_t>(m_buckets[bi], 1);
    if (m <= 16) continue;   // the CUDA-core GEMV owns decode shapes
    GemmProblem p = problem;
    p.m = m;
    p.epilogue_ops = problem.epilogue_ops & kEpiGLU;
    void *da = nullptr, *db = nullptr, *dd = nullptr;
    const size_t a_bytes = static_cast<size_t>(m * p.k) * es, b_bytes = static_cast<size_t>(p.n * p.k) * es;
    const size_t d_bytes = static_cast<size_t>(m * p.n) * 4;
    if (cudaMalloc(&da, a_bytes) != cudaSuccess || cudaMalloc(&db, b_bytes) != cudaSuccess || cudaMalloc(&dd, d_bytes) != cudaSuccess) {
      cudaFree(da); cudaFree(db); cudaFree(dd);
      cudaGetLastError();
      return detail::set_error(Status::kCudaError, "autotune: cannot allocate the timing buffers");
    }
    cudaMemset(da, 0x3C, a_bytes);
    cudaMemset(db, 0x3C, b_bytes);
    ScaledGemmArgs args{};
    args.recipe = problem.recipe;
    args.a.data = da; args.a.ld = p.k; args.a.dtype = dt;
    args.b.data = db; args.b.ld = p.k; args.b.dtype = dt;
    args.d = dd; args.d_dtype = dt; args.ldd = p.n;
    args.m = m; args.n = p.n; args.k = p.k;
    if ((p.epilogue_ops & kEpiGLU) != 0u) {   // timing GLU needs a plausible N / 2 output
      args.epi.ops = kEpiGLU; args.epi.glu_layout = GateLayout::kInterleaved; args.epi.glu_interleave = 1; args.epi.act = ActFn::kSiLU; args.ldd = p.n / 2;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best_ms = 1e30f;
    GemmTactic best{};
    bool have = false;
    for (const GemmTactic& t : candidates(p, ctx)) {
      Status r = mma_gemm(args, &t, ws, ctx);
      if (r != Status::kSuccess) continue;   // e.g. split-K without enough workspace
      constexpr int kReps = 24;
      cudaEventRecord(e0, s);
      for (int i = 0; i < kReps; ++i) mma_gemm(args, &t, ws, ctx);
      cudaEventRecord(e1, s);
      cudaEventSynchronize(e1);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best_ms) { best_ms = ms; best = t; have = true; }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(da); cudaFree(db); cudaFree(dd);
    if (cudaGetLastError() != cudaSuccess) return detail::set_error(Status::kCudaError, "autotune: a candidate failed");
    if (!have) return detail::set_error(Status::kInternal, "autotune: no tactic ran for M=%lld", static_cast<long long>(m));
    std::lock_guard<std::mutex> lock(cache_mutex());
    cache_map()[make_key(problem.recipe, m, p.n, p.k, ctx)] = best;
  }
  return Status::kSuccess;
}

Status tactic_cache_lookup(const GemmProblem& problem, const LaunchContext& ctx, GemmTactic* out) {
  if (out == nullptr) return detail::set_error(Status::kInvalidArgument, "tactic_cache_lookup: null output");
  if (!cache_find(make_key(problem.recipe, problem.m, problem.n, problem.k, ctx), out)) {
    return detail::set_error(Status::kInvalidArgument, "tactic_cache_lookup: no entry for this (recipe, M bucket, N, K, SM, CUDA)");
  }
  return Status::kSuccess;
}

Status tactic_cache_save(const char* path) {
  if (path == nullptr) return detail::set_error(Status::kInvalidArgument, "tactic_cache_save: null path");
  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) return detail::set_error(Status::kInvalidArgument, "tactic_cache_save: cannot open %s", path);
  std::lock_guard<std::mutex> lock(cache_mutex());
  std::fprintf(f, "kops-tactic-cache 1 abi %d\n", KOPS_ABI_VERSION);
  for (const auto& kv : cache_map()) {
    const CacheKey& k = kv.first;
    const GemmTactic& t = kv.second;
    std::fprintf(f, "%d %d %d %lld %lld %lld %d %d %d %d %d %d %d %d %llu\n", k.recipe, k.arch, k.cuda, static_cast<long long>(k.n),
                 static_cast<long long>(k.k), static_cast<long long>(k.mbucket), t.tile_m, t.tile_n, t.tile_k, t.stages,
                 static_cast<int>(t.split_k_mode), t.split_k_slices, static_cast<int>(t.raster), t.swap_ab ? 1 : 0,
                 static_cast<unsigned long long>(t.kernel_id));
  }
  const bool ok = std::fclose(f) == 0;
  return ok ? Status::kSuccess : detail::set_error(Status::kInvalidArgument, "tactic_cache_save: write failed");
}

Status tactic_cache_load(const char* path) {
  if (path == nullptr) return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: null path");
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: cannot open %s", path);
  int version = 0, abi = 0;
  if (std::fscanf(f, "kops-tactic-cache %d abi %d\n", &version, &abi) != 2 || version != 1 || abi != KOPS_ABI_VERSION) {
    std::fclose(f);
    return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: %s is not a cache written by ABI %d", path, KOPS_ABI_VERSION);
  }
  std::map<CacheKey, GemmTactic> loaded;
  for (;;) {
    CacheKey k{};
    long long n, kk, mb;
    int tm, tn, tk, stages, mode, slices, raster, swap;
    unsigned long long kid;
    const int got = std::fscanf(f, "%d %d %d %lld %lld %lld %d %d %d %d %d %d %d %d %llu\n", &k.recipe, &k.arch, &k.cuda, &n, &kk, &mb,
                                &tm, &tn, &tk, &stages, &mode, &slices, &raster, &swap, &kid);
    if (got == EOF) break;
    if (got != 15) {
      std::fclose(f);
      return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: malformed line");
    }
    k.n = n; k.k = kk; k.mbucket = mb;
    GemmTactic t{};
    t.tile_m = static_cast<int16_t>(tm); t.tile_n = static_cast<int16_t>(tn); t.tile_k = static_cast<int16_t>(tk);
    t.stages = static_cast<int8_t>(stages);
    t.mainloop = MainloopSchedule::kMultistageMmaSync;
    t.epilogue = EpilogueSchedule::kDirectStore;
    t.split_k_mode = static_cast<SplitKMode>(mode);
    t.split_k_slices = static_cast<int16_t>(slices);
    t.raster = static_cast<RasterOrder>(raster);
    t.swap_ab = swap != 0;
    t.kernel_id = kid;
    if (tile_of(t) < 0) {
      std::fclose(f);
      return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: tile %dx%d does not exist in this build", tm, tn);
    }
    loaded[k] = t;
  }
  std::fclose(f);
  std::lock_guard<std::mutex> lock(cache_mutex());
  for (const auto& kv : loaded) cache_map()[kv.first] = kv.second;
  return Status::kSuccess;
}

}  // namespace kops::gemm
