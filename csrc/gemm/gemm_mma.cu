// =============================================================================
// kernel-ops | csrc/gemm/gemm_mma.cu
// B7: the shared tensor-core GEMM frame on mma.sync (SM80/86/89/120).
//   gemm_mma_sync_kernel   cp.async multistage + ldmatrix + mma.sync, XOR-swizzled smem, swap_ab, split-K
//                          (serial = last CTA of a tile sums the slices in slice order), every EpilogueOp.
//                          Atoms (one k-step = 32 bytes of each operand row):
//                            bf16 / fp16  m16n8k16, fp32 accumulators              (row 1, SM80+)
//                            e4m3         m16n8k32, fp32 accumulators              (rows 3, 4, 7, 8; SM89+)
//                            s8           m16n8k32, s32 accumulators               (rows 5, 7; SM80+)
//                            tf32         m16n8k8 on fp32 operands (cvt.rna)       (row 1; ctx.allow_tf32)
//                          A staging buffer holds 64 bytes of each operand row. The small-M swap tiles
//                          (wide_ld, 6+ stages) fetch two buffers per request, i.e. 128 bytes of a row, so
//                          the DRAM sees full cache lines instead of half ones: the bandwidth-bound
//                          shapes go from ~0.5 to ~0.55-0.6 TB/s on the 100 KB parts.
//                          Small M is a per-CTA mainloop-rate problem: one CTA reaches ~25% of the tensor
//                          pipe, two CTAs of eight warps ~50%. The band therefore runs tiles that fit two
//                          CTAs per SM: 16 x 128 (M <= 64), 32 x 64 (M <= 128), 64 x 128 with eight warps
//                          from M = 129 on; 64 x 128 with eight warps + two CTAs (16 warps) is also the
//                          best tile at M = 4096 (measured 168 TF, ~1.0x cuBLAS on SM89).
//                          FP8 kFP8Blockwise: 1x128 (A) x 128x128 (B) scales applied to a per-128-K
//                          partial accumulator (DeepSeek promotion) in registers.
//                          MmaRowMap (ABI 6): grouped rows, masked groups and block-sorted rows with
//                          row ids, so grouped GEMM, the fused-MoE expert GEMM and LoRA run this body
//                          through a persistent task loop.
//   tactics                get_tactics / autotune / tactic_cache_{lookup,save,load}
// Host entries (ops.h): mma_gemm, mma_selects, mma_workspace, launch_gemm_mma_sync.
//
// Contract: csrc/gemm/dense_gemm_kernels.cuh (gemm_mma_sync_kernel), csrc/gemm/gemm_params.cuh
// (MmaGemmParams), csrc/gemm/ops.h, CONTRACTS.md §1d.
//
// Numerics.  Products and accumulation happen inside mma.sync (fp32, or exact s32 for s8).  For one
// output element the K chain is the same sequence of k-steps in every tile shape and with or without
// swap_ab or a row map, so tactics with the same split-K slices agree bitwise and a row alone equals the
// same row inside a batch / group.  Split-K sums the slice partials in slice order (0..S-1, integers for
// s8), never by arrival order; there are no atomics unless the caller asks for kEpiAtomicAdd.  The
// epilogue is fp32 and rounds once to the output dtype (or quantizes once for kEpiQuantOut).
// Stream-K (SplitKMode::kStreamK / kStreamKHybrid) is intentionally absent: it only rebalances K slices
// across SMs, and the measured serial split-K -- which uses the same partial traffic and the same fixed
// slice-order reduction -- is slower than one CTA per tile at every M <= 512 on SM89. The gap there is
// the per-CTA mainloop rate, not wave quantization, so stream-K would add workspace and reduction risk
// without a payoff. split_workspace_bytes() keeps rejecting it.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
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

/// 16-byte chunk index XOR for 64-byte stage rows: swap on row / 2, so the eight rows of an ldmatrix sit on
/// eight distinct 16-byte bank groups.
__device__ __forceinline__ int swz(int row) { return (row >> 1) & 3; }

// ---- MMA atoms ---------------------------------------------------------------------------------
// A k-step reads 32 bytes of each operand row through the same ldmatrix.x4 (b16 view), which yields
// the A {a0..a3} and B {b0, b1} fragments of m16n8k16 (16-bit), m16n8k32 (8-bit) and m16n8k8 (tf32).

struct AtomBf16 {
  using elem_t = __nv_bfloat16;
  using acc_t = float;
  static constexpr int kElemBytes = 2;
  static constexpr bool kInt = false;
  static constexpr bool kTf32 = false;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    dev::TcFrag<__nv_bfloat16>::mma(c, a, b0, b1);
  }
};
struct AtomF16 {
  using elem_t = __half;
  using acc_t = float;
  static constexpr int kElemBytes = 2;
  static constexpr bool kInt = false;
  static constexpr bool kTf32 = false;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    dev::TcFrag<__half>::mma(c, a, b0, b1);
  }
};
struct AtomE4M3 {
  using elem_t = uint8_t;
  using acc_t = float;
  static constexpr int kElemBytes = 1;
  static constexpr bool kInt = false;
  static constexpr bool kTf32 = false;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;
    __trap();   // the hosts refuse FP8 mma.sync below SM89 (kUnsupportedArch)
#endif
  }
};
struct AtomS8 {
  using elem_t = int8_t;
  using acc_t = int32_t;
  static constexpr int kElemBytes = 1;
  static constexpr bool kInt = true;
  static constexpr bool kTf32 = false;
  __device__ static __forceinline__ void mma(int32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
  }
};
struct AtomTf32 {
  using elem_t = float;
  using acc_t = float;
  static constexpr int kElemBytes = 4;
  static constexpr bool kInt = false;
  static constexpr bool kTf32 = true;
  __device__ static __forceinline__ uint32_t round(uint32_t x) {
    uint32_t r = x;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(x)));
#endif
    return r;
  }
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
  }
};

// ---- Epilogue ----------------------------------------------------------------------------------
// The activation switches are out of line: one copy for every instantiation instead of three inlined
// copies of eleven activation bodies per kernel (compile time), and they run once per output element.
__device__ __noinline__ float glu_runtime(ActFn fn, float gate, float up, const GluParams& p) {
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

__device__ __noinline__ float act_outlined(ActFn fn, float x, const GluParams& p) { return dev::act_runtime(fn, x, p); }

/// Per-task constants of the epilogue.
struct EpiTask {
  float alpha;            // alpha (or alpha_ptr[slot]) x per-tensor operand scales of this batch / slot
  int colslot;            // slot of the column vectors (0 for the dense schedule)
  const float* xs_row;    // per-token operand scale of A (indexed by the A source row), or null
  const float* ws_col;    // per-channel operand scale of B (indexed by n), or null
};

/// acc -> *alpha -> *sA[row] -> *sB[n] -> *row_scale -> *col_scale -> +bias -> clamp -> softcap (everything
/// before act / GLU). sA / sB are the operand scales of ScaledOperand (FP8 / INT8 tensorwise and rowwise).
__device__ __forceinline__ float epi_linear(float v, const EpilogueParams& e, const EpiTask& tk, int64_t vid, int64_t xrow, int64_t n) {
  const uint32_t ops = e.ops;
  v *= tk.alpha;
  if (tk.xs_row != nullptr) v *= tk.xs_row[xrow];
  if (tk.ws_col != nullptr) v *= tk.ws_col[n];
  if (ops & kEpiRowScale) v *= e.row_scale[vid];
  if (ops & kEpiColScale) v *= e.col_scale[tk.colslot * e.col_scale_group_stride + n];
  if (ops & kEpiBias) v += dev::load_as_float(e.bias, e.bias_dtype, tk.colslot * e.bias_group_stride + n);
  if (ops & kEpiClamp) v = fminf(fmaxf(v, -e.clamp_limit), e.clamp_limit);
  if (ops & kEpiSoftcap) v = softcap(v, e.softcap);
  return v;
}

/// *routing_weight -> sigmoid gate multiply -> + beta * residual, in output coordinates (operands in D's dtype).
__device__ __forceinline__ float epi_tail(float v, const EpilogueParams& e, DType dt, int64_t vid, int64_t oc) {
  const uint32_t ops = e.ops;
  if (ops & kEpiRoutingWeight) v *= e.routing_weight[vid];
  if (ops & kEpiSigmoidGateMul) {
    const float gs = dev::load_as_float(e.gate_src, dt, vid * e.gate_src_stride + oc);
    v = e.gate_is_acc ? gs * sigmoid(v) : v * sigmoid(gs);
  }
  if (ops & kEpiResidualAdd) v = fmaf(e.beta, dev::load_as_float(e.residual, dt, vid * e.residual_stride + oc), v);
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

/// Stores one or two adjacent finished output elements (vid, oc) / (vid, oc + 1) (direct / scattered /
/// atomic), then appends the stored value to the KV destination.
template <typename Dt>
__device__ __forceinline__ void store_pair(const EpilogueParams& e, void* dvoid, int64_t ldd, int64_t vid, int64_t oc,
                                           float v0, float v1, bool has1) {
  Dt* const dbase = static_cast<Dt*>(dvoid);
  const uint32_t ops = e.ops;
  int64_t drow = vid;
  if (ops & kEpiScatterRows) drow = e.scatter_rows[vid];
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
    const int32_t slot = e.kv_slots[vid];
    if (slot >= 0) {
      Dt* kv = static_cast<Dt*>(e.kv_dst) + static_cast<int64_t>(slot) * e.kv_stride;
      if (oc >= e.kv_col_begin && oc < e.kv_col_begin + e.kv_cols) kv[oc - e.kv_col_begin] = s0;
      if (has1 && oc + 1 >= e.kv_col_begin && oc + 1 < e.kv_col_begin + e.kv_cols) kv[oc + 1 - e.kv_col_begin] = s1;
    }
  }
}

__device__ __forceinline__ void store_pair_rt(DType dt, const EpilogueParams& e, void* d, int64_t ldd, int64_t vid, int64_t oc,
                                              float v0, float v1, bool has1) {
  switch (dt) {
    case DType::kFloat32: store_pair<float>(e, d, ldd, vid, oc, v0, v1, has1); break;
    case DType::kFloat16: store_pair<__half>(e, d, ldd, vid, oc, v0, v1, has1); break;
    default: store_pair<__nv_bfloat16>(e, d, ldd, vid, oc, v0, v1, has1); break;
  }
}

// ---- kEpiQuantOut ------------------------------------------------------------------------------
// Static per-tensor (q = x / static_scale) and dynamic per-group (one scale per row and `group_size`
// output columns of the pass, s = max(amax, 1e-10) / qmax, optional UE8M0 ceiling), FP8 e4m3 or INT8
// symmetric: the rules of quantization/quant.cu applied to the fp32 epilogue value.
__device__ __forceinline__ uint8_t quant_byte(float x, bool int8) {
  if (int8) {
    const int q = __float2int_rn(x);
    return static_cast<uint8_t>(static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q)));
  }
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float ue8m0_ceil(float s) {
  int e = 0;
  if (s > 0.0f && isfinite(s)) {
    frexpf(s, &e);
    e = e - 1;
    if (ldexpf(1.0f, e) < s) ++e;
  } else {
    e = -127;
  }
  e = e < -127 ? -127 : (e > 127 ? 127 : e);
  return ldexpf(1.0f, e);
}

/// FP8 block scale (fp32 or a UE8M0 byte).
__device__ __forceinline__ float block_scale_at(const void* p, DType t, int64_t i) {
  if (t == DType::kFloat8E8M0) return __uint_as_float(static_cast<uint32_t>(static_cast<const uint8_t*>(p)[i]) << 23);
  return static_cast<const float*>(p)[i];
}

// ---- kEpiRowSumSq ------------------------------------------------------------------------------
/// row_sumsq = sum_k A[k]^2 of one row: one warp, 16 bytes per lane per step (8 x 16-bit or 4 x fp32), then
/// the xor butterfly. The order depends on K only.
template <typename E>
__device__ __forceinline__ float row_sumsq_warp(const E* row, int64_t k, int lane) {
  constexpr int V = 16 / static_cast<int>(sizeof(E));
  float s = 0.0f;
  for (int64_t kk = static_cast<int64_t>(lane) * V; kk < k; kk += 32 * V) {
    const uint4 raw = *reinterpret_cast<const uint4*>(row + kk);
    const E* h = reinterpret_cast<const E*>(&raw);
    #pragma unroll
    for (int i = 0; i < V; ++i) {
      const float f = dev::to_f(h[i]);
      s = fmaf(f, f, s);
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
  return s;
}

/// Group extent of the grouped schedules.
__device__ __forceinline__ void group_extent(const MmaRowMap& rm, int g, int64_t* rb, int64_t* re) {
  if (rm.masked_rows != nullptr) {
    *rb = static_cast<int64_t>(g) * rm.max_rows;
    const int mr = rm.masked_rows[g];
    *re = *rb + (mr < 0 ? 0 : (mr > rm.max_rows ? rm.max_rows : mr));
  } else if (rm.group_rows != nullptr) {
    *rb = rm.group_rows[g];
    *re = rm.group_rows[g + 1];
  } else {
    *rb = rm.group_rows32[g];
    *re = rm.group_rows32[g + 1];
  }
}

}  // namespace

// =============================================================================
// Traits and kernel
// =============================================================================

/// BR / BC are the row counts of the mma-A / mma-B operand tiles. Without swap_ab the mma-A operand is
/// the activations (BR = tile_m); with swap_ab it is the weights (BR = tile_n) and the accumulator is
/// D^T, which the epilogue transposes back on the way out. BLOCK: FP8 1x128 x 128x128 promotion. AF32:
/// the activations are fp32 in memory and rounded to the atom's 16-bit type on the way to shared memory
/// (LoRA expand reads the fp32 shrink buffer).
template <typename Atom, int BR, int BC, int ST, int WM, int WN, bool SWAP, bool BLOCK, bool AF32>
struct MmaTraits {
  using atom = Atom;
  using a_type = typename Atom::elem_t;
  using b_type = typename Atom::elem_t;
  using accum_type = typename Atom::acc_t;
  static constexpr int tile_m = SWAP ? BC : BR;
  static constexpr int tile_n = SWAP ? BR : BC;
  static constexpr int tile_k = 64 / Atom::kElemBytes;   // K elements per stage (64 bytes per row)
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
  static constexpr bool block_scaled = BLOCK;
  static constexpr bool a_fp32 = AF32;
  /// 8 stages = 4 pairs: the loader fetches 128 bytes of each operand row at a time (two staging buffers per
  /// issue), see the prefetch-group logic in the kernel. Only the 16-bit atoms compile the small-M swap
  /// tiles that deep; the fp32-activation tile keeps the 64-byte fetch (its loader is synchronous and short).
  static constexpr bool wide_ld = ST >= 6 && !AF32;
  static_assert(ST == 3 || ST == 4 || (ST >= 6 && ST % 2 == 0), "stages are 3, 4 or a whole number of pairs (>= 6)");
  static constexpr int threads = 32 * WM * WN;
  static constexpr size_t smem_bytes = static_cast<size_t>(ST) * (BR + BC) * 64;
  // CTAs per SM that shared memory allows on the 100 KB parts (SM86/89/120): telling ptxas keeps it from
  // trading registers (spills) for an occupancy the pipeline buffers cannot reach anyway.
  static constexpr int min_blocks = static_cast<int>((101376 / (smem_bytes + static_cast<size_t>(tile_m) * 8 + 1024)) < 1 ? 1
                                    : ((101376 / (smem_bytes + static_cast<size_t>(tile_m) * 8 + 1024)) > 4 ? 4
                                    : (101376 / (smem_bytes + static_cast<size_t>(tile_m) * 8 + 1024))));
  static_assert(BR % (WM * 16) == 0 && BC % (WN * 16) == 0, "warp tile: m16 x (two n8 per ldmatrix)");
  static_assert(!AF32 || Atom::kElemBytes == 2, "fp32 activations are rounded to a 16-bit atom");
  static_assert(!BLOCK || (!Atom::kInt && Atom::kElemBytes == 1), "block promotion is the FP8 recipe");
};

template <typename Traits>
__global__ void __launch_bounds__(Traits::threads, Traits::min_blocks) gemm_mma_sync_kernel(const __grid_constant__ MmaGemmParams P) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Atom = typename Traits::atom;
  using Acc = typename Atom::acc_t;
  constexpr int BR = Traits::br, BC = Traits::bc, ST = Traits::stages;
  constexpr int WM = Traits::warps_m, WN = Traits::warps_n;
  constexpr bool SWAP = Traits::swap_ab;
  constexpr bool BLOCK = Traits::block_scaled;
  constexpr bool AF32 = Traits::a_fp32;
  constexpr int MT = BR / WM / 16;
  constexpr int NT = BC / WN / 8;
  constexpr int kThreads = Traits::threads;
  // WIDE: one global fetch covers 128 bytes of each operand row (two 64-byte staging rows), so the DRAM
  // sees full cache lines instead of half ones. The staging layout, the swizzle and the fragments do not
  // change; the extra stages give the pair of buffers the fetch lands in room to be in flight.
  constexpr bool WIDE = Traits::wide_ld;
  constexpr int CH = WIDE ? 8 : 4;         // 16-byte chunks per fetch (64-byte stage row / 128-byte pair)
  constexpr int KS = 2;                    // 32-byte k-steps per stage
  constexpr int RW = kThreads / CH;        // rows one cp.async pass of the whole CTA covers
  constexpr int A_IT = (BR + RW - 1) / RW;
  constexpr int B_IT = (BC + RW - 1) / RW;
  constexpr uint32_t A_STAGE = BR * 64;
  constexpr uint32_t B_STAGE = BC * 64;
  constexpr int TM = SWAP ? BC : BR;       // logical rows per tile
  constexpr int TN = SWAP ? BR : BC;       // output columns per tile
  constexpr int EB = Atom::kElemBytes;
  static_assert(RW % 8 == 0 && (BR % RW == 0 || BR < RW) && (BC % RW == 0 || BC < RW), "loader passes keep the swizzle phase");
  static_assert(A_IT <= 32 && B_IT <= 32, "row masks");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* const sA = smem_raw;
  unsigned char* const sB = smem_raw + ST * A_STAGE;
  // Row tables of the grouped schedules, after the pipeline buffers (the host adds 8 * TM bytes of dynamic
  // shared memory for them; the dense schedule computes rows arithmetically and keeps its occupancy).
  int32_t* const s_xrow = reinterpret_cast<int32_t*>(smem_raw + Traits::smem_bytes);   // A source row, -1 = padding
  int32_t* const s_vid = s_xrow + TM;                                                  // id for D / row vectors
  __shared__ int64_t s_task64[3];    // r0, r_end, cursor base
  __shared__ int s_task[4];          // status (0 run, 1 skip, 2 done), tn, slot, cursor group
  __shared__ int s_last;

  const ScaledGemmArgs& args = P.args;
  const MmaRowMap& rm = P.rows;
  const bool dense = rm.schedule == MmaSchedule::kDense;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t M = args.m, N = args.n, K = args.k;
  const int KB = static_cast<int>(K) * EB;                // bytes of one operand row (< 2^31, checked by the host)
  const int tiles_n = static_cast<int>((N + TN - 1) / TN);
  const int slices = max(args.split_k_slices, 1);
  // Serial split-K packs (tile, slice) into blockIdx.x for the dense schedule (see launch_mma), so the
  // slices of one tile land on neighbouring CTAs and can share an SM; the reduce below sums them in
  // slice order whatever the arrival order is. kParallel keeps the slices in blockIdx.z for its reducer.
  const bool xslice = dense && slices > 1 && args.split_k_mode == SplitKMode::kSerial;
  const int tiles_m = static_cast<int>((M + TM - 1) / TM);
  const int tiles_xy = tiles_m * tiles_n;
  const int zb = dense ? (xslice ? static_cast<int>(blockIdx.z) : static_cast<int>(blockIdx.z) / slices) : 0;
  const int slice =
      dense ? (xslice ? static_cast<int>(blockIdx.x) / tiles_xy : static_cast<int>(blockIdx.z) % slices)
            : static_cast<int>(blockIdx.z) % slices;
  const ScaledOperand& opW = args.b;                       // weights (rows = n)
  const ScaledOperand& opX = args.a;                       // activations (rows = logical rows)
  const int64_t ldw_b = opW.ld * EB;
  const int64_t ldx_b = opX.ld * (AF32 ? 4 : EB);
  const EpilogueParams& e = args.epi;

  const int kt_total = static_cast<int>((KB + 63) / 64);
  int kt_per = (kt_total + slices - 1) / slices;
  if (BLOCK) kt_per = (kt_per + 1) & ~1;                   // slices start on a 128-K block
  const int kt_begin = min(slice * kt_per, kt_total);
  const int kt_end = min(kt_begin + kt_per, kt_total);
  const int nk = kt_end - kt_begin;

  if (tid == 0) {
    s_task[3] = 0;
    s_task64[2] = 0;
  }
  __syncthreads();

  for (int64_t task = blockIdx.x;; task += gridDim.x) {
    // ---------------- resolve the task -> (rows [r0, r_end), n tile, slot) ----------------
    if (dense) {
      if (task != blockIdx.x) break;
      if (tid == 0) {
        const int pid = xslice ? static_cast<int>(task) % tiles_xy : static_cast<int>(task);
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
        s_task[0] = 0;
        s_task[1] = tn;
        s_task[2] = 0;
        s_task64[0] = static_cast<int64_t>(tm) * TM;
        s_task64[1] = M;
      }
    } else if (rm.schedule == MmaSchedule::kSortedBlocks) {
      if (tid == 0) {
        const int64_t rows = rm.num_rows != nullptr ? static_cast<int64_t>(*rm.num_rows) : M;
        const int64_t tiles_m = (rows + TM - 1) / TM;
        if (task >= tiles_m * tiles_n) {
          s_task[0] = 2;
        } else {
          const int64_t tm = task % tiles_m;
          const int tn = static_cast<int>(task / tiles_m);
          const int64_t r0 = tm * TM;
          const int slot = rm.block_group[r0 / rm.block_rows];
          s_task[0] = slot < 0 ? 1 : 0;
          s_task[1] = tn;
          s_task[2] = slot;
          s_task64[0] = r0;
          s_task64[1] = r0 + TM < rows ? r0 + TM : rows;
        }
      }
    } else if (rm.schedule == MmaSchedule::kTaskMap) {
      if (tid == 0) {
        if (task >= *rm.num_tasks) {
          s_task[0] = 2;
        } else {
          const int g = rm.task_map[3 * task];
          int64_t rb, re;
          group_extent(rm, g, &rb, &re);
          const int slot = rm.group_slot != nullptr ? rm.group_slot[g] : g;
          s_task[0] = slot < 0 ? 1 : 0;
          s_task[1] = rm.task_map[3 * task + 2];
          s_task[2] = slot;
          s_task64[0] = rb + static_cast<int64_t>(rm.task_map[3 * task + 1]) * TM;
          s_task64[1] = re;
        }
      }
    } else if (warp == 0) {
      // Grouped rows: tasks of group g are ceil(rows_g / TM) * tiles_n consecutive ids (M tiles fastest, so
      // the CTAs of one weight N tile run together). Tasks of a CTA increase, so the cursor only moves forward.
      const int G = rm.num_groups;
      int g0 = s_task[3];
      int64_t base = s_task64[2];
      int status = 2;
      int64_t local = 0;
      while (g0 < G) {
        const int g = g0 + lane;
        int64_t nt_g = 0;
        if (g < G) {
          int64_t rb, re;
          group_extent(rm, g, &rb, &re);
          const int sl = rm.group_slot != nullptr ? rm.group_slot[g] : g;
          if (sl >= 0 && re > rb) nt_g = ((re - rb + TM - 1) / TM) * tiles_n;
        }
        int64_t incl = nt_g;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int64_t v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
          if (lane >= o) incl += v;
        }
        const int64_t total = __shfl_sync(0xFFFFFFFFu, incl, 31);
        if (task < base + total) {
          const unsigned hit = __ballot_sync(0xFFFFFFFFu, task < base + incl);
          const int L = __ffs(hit) - 1;
          const int64_t excl = __shfl_sync(0xFFFFFFFFu, incl - nt_g, L);
          g0 += L;
          base += excl;
          local = task - base;
          status = 0;
          break;
        }
        base += total;
        g0 += 32;
      }
      if (lane == 0) {
        s_task[0] = status;
        s_task[3] = g0 < G ? g0 : G;
        s_task64[2] = base;
        if (status == 0) {
          int64_t rb, re;
          group_extent(rm, g0, &rb, &re);
          const int64_t mtiles = (re - rb + TM - 1) / TM;
          s_task[1] = static_cast<int>(local / mtiles);
          s_task[2] = rm.group_slot != nullptr ? rm.group_slot[g0] : g0;
          s_task64[0] = rb + (local % mtiles) * TM;
          s_task64[1] = re;
        }
      }
    }
    __syncthreads();
    const int status = s_task[0];
    if (status == 2) break;
    if (status == 1) {
      __syncthreads();   // s_task is rewritten by the next resolution
      continue;
    }
    const int tn = s_task[1];
    const int slot = s_task[2];
    const int r0 = static_cast<int>(s_task64[0]);
    const int r_end = static_cast<int>(s_task64[1]);

    // ---------------- row table of the tile (grouped schedules) ----------------
    if (!dense) {
      for (int i = tid; i < TM; i += kThreads) {
        const int m = r0 + i;
        int32_t xrow = -1, vid = -1;
        if (m < r_end && (rm.schedule != MmaSchedule::kSortedBlocks || rm.block_rows != 1 || rm.block_group[m] == slot)) {
          const int64_t id = rm.row_ids != nullptr ? static_cast<int64_t>(rm.row_ids[m]) : m;
          if (id >= 0 && (rm.id_limit <= 0 || id < rm.id_limit)) {
            xrow = static_cast<int32_t>(rm.a_rows_by_id ? id / rm.a_row_div : m);
            vid = static_cast<int32_t>(rm.out_rows_by_id ? id : m);
          }
        }
        s_xrow[i] = xrow;
        s_vid[i] = vid;
      }
      __syncthreads();
    }
    // logical row r0 + i of the tile: A source row and id (-1 = padding)
    auto x_row = [&](int i) -> int { return dense ? (r0 + i < M ? r0 + i : -1) : s_xrow[i]; };
    auto v_id = [&](int i) -> int { return dense ? (r0 + i < M ? r0 + i : -1) : s_vid[i]; };

    const int n0 = tn * TN;
    const unsigned char* const xbase = static_cast<const unsigned char*>(opX.data) + static_cast<int64_t>(zb) * opX.batch_stride * (AF32 ? 4 : EB);
    const int wslot = dense ? zb : slot;
    const unsigned char* const wbase =
        rm.b_ptrs != nullptr ? static_cast<const unsigned char*>(rm.b_ptrs[wslot])
                             : static_cast<const unsigned char*>(opW.data) + static_cast<int64_t>(wslot) * opW.batch_stride * EB;

    // ---------------- loaders ----------------
    // A fetch covers CH 16-byte chunks: 64 B of one staging buffer (narrow) or 128 B split over two
    // consecutive staging buffers of the same ring slot pair (WIDE).
    const int ld_ch = tid % CH;
    const int ld_row = tid / CH;
    const int ld_half = WIDE ? (ld_ch >> 2) : 0;    // second 64-byte stage row of a wide fetch
    const int ld_c4 = WIDE ? (ld_ch & 3) : ld_ch;   // 16-byte chunk inside the 64-byte stage row
    const uint32_t ld_soff = static_cast<uint32_t>(ld_row * 64 + ((ld_c4 ^ swz(ld_row)) << 4));
    const uint32_t sA_u = smem_u32(sA);
    const uint32_t sB_u = smem_u32(sB);
    const uint32_t ld_offA = sA_u + static_cast<uint32_t>(ld_half) * A_STAGE;
    const uint32_t ld_offB = sB_u + static_cast<uint32_t>(ld_half) * B_STAGE;
    // weights: rows n0 + ld_row + i * RW, one pointer stepped by RW rows
    constexpr int W_ROWS = SWAP ? BR : BC;
    constexpr int W_IT = SWAP ? A_IT : B_IT;
    constexpr int X_ROWS = SWAP ? BC : BR;
    constexpr int X_IT = SWAP ? B_IT : A_IT;
    const unsigned char* pw = wbase + static_cast<int64_t>(n0 + ld_row) * ldw_b + ld_ch * 16 + kt_begin * 64;
    uint32_t wmask = 0;
    #pragma unroll
    for (int i = 0; i < W_IT; ++i) {
      const int rr = ld_row + i * RW;
      wmask |= static_cast<uint32_t>(rr < W_ROWS && n0 + rr < N) << i;
    }
    int kld = kt_begin * 64 + ld_ch * 16;   // byte k of this thread's chunk (smem side)
    const int64_t stepW = static_cast<int64_t>(RW) * ldw_b;
    // activations of the dense schedule: one pointer stepped by RW rows, like the weights
    const unsigned char* px = xbase + static_cast<int64_t>(r0 + ld_row) * ldx_b + (AF32 ? 2 : 1) * (ld_ch * 16 + kt_begin * 64);
    uint32_t xmask = 0;
    if (dense) {
      #pragma unroll
      for (int i = 0; i < X_IT; ++i) {
        const int rr = ld_row + i * RW;
        xmask |= static_cast<uint32_t>(rr < X_ROWS && r0 + rr < M) << i;
      }
    }
    const int64_t stepX = static_cast<int64_t>(RW) * ldx_b;

    auto issue = [&](uint32_t wsA, uint32_t wsB) {
      const bool kok = kld < KB;
      const uint32_t sW = (SWAP ? ld_offA + wsA : ld_offB + wsB);
      const uint32_t sX = (SWAP ? ld_offB + wsB : ld_offA + wsA);
      {
        const unsigned char* q = pw;
        #pragma unroll
        for (int i = 0; i < W_IT; ++i) {
          if (W_ROWS % RW == 0 || ld_row + i * RW < W_ROWS) cp_async16(sW + ld_soff + i * RW * 64, q, kok && ((wmask >> i) & 1u));
          q += stepW;
        }
      }
      #pragma unroll
      for (int i = 0; i < X_IT; ++i) {
        const int rr = ld_row + i * RW;
        if (X_ROWS % RW != 0 && rr >= X_ROWS) continue;
        const unsigned char* src_row;
        bool ok;
        if (dense) {
          src_row = px + i * stepX;
          ok = kok && ((xmask >> i) & 1u);
        } else {
          const int xr = s_xrow[rr];
          ok = kok && xr >= 0;
          src_row = xbase + static_cast<int64_t>(ok ? xr : 0) * ldx_b + (AF32 ? 2 : 1) * kld;
        }
        if constexpr (AF32) {
          // fp32 source -> 8 elements of the 16-bit atom type, synchronous (K is the LoRA rank, short)
          using E = typename Atom::elem_t;
          uint4 packed = make_uint4(0u, 0u, 0u, 0u);
          if (ok) {
            const float4* src = reinterpret_cast<const float4*>(src_row);
            const float4 f0 = __ldg(src);
            const float4 f1 = __ldg(src + 1);
            E h[8];
            dev::from_f(h[0], f0.x); dev::from_f(h[1], f0.y); dev::from_f(h[2], f0.z); dev::from_f(h[3], f0.w);
            dev::from_f(h[4], f1.x); dev::from_f(h[5], f1.y); dev::from_f(h[6], f1.z); dev::from_f(h[7], f1.w);
            packed = *reinterpret_cast<const uint4*>(h);
          }
          *reinterpret_cast<uint4*>(smem_raw + (sX - smem_u32(smem_raw)) + ld_soff + i * RW * 64) = packed;
        } else {
          cp_async16(sX + ld_soff + i * RW * 64, ok ? static_cast<const void*>(src_row) : static_cast<const void*>(xbase), ok);
        }
      }
      pw += WIDE ? 128 : 64;
      px += (AF32 ? 2 : 1) * (WIDE ? 128 : 64);
      kld += WIDE ? 128 : 64;
    };

    // ---------------- fragments ----------------
    const int warp_r0 = (warp / WN) * (BR / WM);
    const int warp_c0 = (warp % WN) * (BC / WN);
    const int sw_l = swz(lane & 7);
    const uint32_t a_lane = static_cast<uint32_t>((warp_r0 + (lane & 7) + ((lane >> 3) & 1) * 8) * 64);
    const uint32_t b_lane = static_cast<uint32_t>((warp_c0 + (lane & 7) + (lane >> 4) * 8) * 64);
    uint32_t a_koff[KS], b_koff[KS];
    #pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      a_koff[kk] = static_cast<uint32_t>(((kk * 2 + (lane >> 4)) ^ sw_l) << 4);
      b_koff[kk] = static_cast<uint32_t>(((kk * 2 + ((lane >> 3) & 1)) ^ sw_l) << 4);
    }
    auto load_a = [&](int kk, uint32_t stA, uint32_t (&af)[MT][4]) {
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        ldmatrix_x4(af[mt], sA_u + stA + a_lane + mt * 16 * 64 + a_koff[kk]);
        if constexpr (Atom::kTf32) {
          #pragma unroll
          for (int q = 0; q < 4; ++q) af[mt][q] = AtomTf32::round(af[mt][q]);
        }
      }
    };
    auto load_b = [&](int kk, uint32_t stB, uint32_t (&bf)[NT][2]) {
      #pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        uint32_t r[4];
        ldmatrix_x4(r, sB_u + stB + b_lane + np * 16 * 64 + b_koff[kk]);
        if constexpr (Atom::kTf32) {
          #pragma unroll
          for (int q = 0; q < 4; ++q) r[q] = AtomTf32::round(r[q]);
        }
        bf[2 * np][0] = r[0];
        bf[2 * np][1] = r[1];
        bf[2 * np + 1][0] = r[2];
        bf[2 * np + 1][1] = r[3];
      }
    };

    Acc acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
      #pragma unroll
      for (int j = 0; j < NT; ++j) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = Acc(0);
      }
    }
    // FP8 block promotion: partial accumulator of the current 128-K block and its scales.
    constexpr int PR = BLOCK ? (SWAP ? NT * 2 : MT * 2) : 1;   // activation rows of this thread
    [[maybe_unused]] float pacc[BLOCK ? MT : 1][BLOCK ? NT : 1][4];
    [[maybe_unused]] float sa_blk[PR];
    [[maybe_unused]] float sb_blk = 0.0f;
    [[maybe_unused]] int64_t sa_idx[PR];
    [[maybe_unused]] int64_t sa_kstride = 0, sa_rstride = 0, sb_off = 0, sb_ld = 0;
    if constexpr (BLOCK) {
      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          #pragma unroll
          for (int q = 0; q < 4; ++q) pacc[i][j][q] = 0.0f;
        }
      }
      const bool colmajor = opX.scale_layout == ScaleLayout::kColMajorTma;
      const int64_t kblocks = K / 128;
      const int64_t ld = opX.scale_ld;
      sa_kstride = colmajor ? ld : 1;
      sa_rstride = colmajor ? 1 : ld;
      (void)kblocks;
      #pragma unroll
      for (int j = 0; j < PR; ++j) {
        // thread's activation rows: mma rows (no swap) or mma columns (swap)
        const int local = SWAP ? warp_c0 + (j >> 1) * 8 + 2 * (lane & 3) + (j & 1) : warp_r0 + (j >> 1) * 16 + (lane >> 2) + (j & 1) * 8;
        const int xr = x_row(local);
        sa_idx[j] = static_cast<int64_t>(xr < 0 ? 0 : xr) * sa_rstride;
      }
      sb_ld = opW.scale_ld;
      sb_off = static_cast<int64_t>(dense ? 0 : slot) * opW.scale_batch_stride + (n0 / 128) * sb_ld;
    }

    // Prefetch groups: one staging buffer per group, or two (a pair) when the loader fetches 128 bytes
    // of each operand row at a time. One group is consumed while NGRP - 2 are in flight.
    constexpr int GRP = WIDE ? 2 : 1;        // staging buffers per prefetch group
    constexpr int NGRP = ST / GRP;           // groups the ring holds
    static_assert(ST % GRP == 0 && NGRP >= 3, "the ring needs at least three prefetch groups");

    #pragma unroll
    for (int g = 0; g < NGRP - 1; ++g) {
      if (g * GRP < nk) issue(g * GRP * A_STAGE, g * GRP * B_STAGE);
      cp_async_commit();
    }

    uint32_t af[2][MT][4];
    uint32_t bf[NT][2];
    int rs = 0;
    int ws = (NGRP - 1) * GRP;
    cp_async_wait<NGRP - 2>();
    __syncthreads();
    load_a(0, 0, af[0]);

    for (int it = 0; it < nk; ++it) {
      if constexpr (BLOCK) {
        const int kt = kt_begin + it;
        if ((kt & 1) == 0) {   // first stage of a 128-K block: fetch its scales
          const int kb = kt >> 1;
          sb_blk = block_scale_at(opW.scale, opW.scale_dtype, sb_off + kb);
          #pragma unroll
          for (int j = 0; j < PR; ++j) sa_blk[j] = block_scale_at(opX.scale, opX.scale_dtype, sa_idx[j] + kb * sa_kstride);
        }
      }
      #pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        load_b(kk, rs * B_STAGE, bf);
        if (kk == KS - 1) {
          cp_async_wait<NGRP - 2>();   // the next group (stage, or stage pair) has landed
          __syncthreads();             // and every warp has its fragments of tile it in registers
          rs = rs + 1 == ST ? 0 : rs + 1;
        }
        load_a((kk + 1) % KS, rs * A_STAGE, af[(kk + 1) & 1]);
        if (kk == 0 && it % GRP == 0) {
          if (it + ST - GRP < nk) issue(ws * A_STAGE, ws * B_STAGE);
          cp_async_commit();
          ws = ws + GRP == ST ? 0 : ws + GRP;
        }
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if constexpr (BLOCK) {
              Atom::mma(pacc[mt][nt], af[kk & 1][mt], bf[nt][0], bf[nt][1]);
            } else {
              Atom::mma(acc[mt][nt], af[kk & 1][mt], bf[nt][0], bf[nt][1]);
            }
          }
        }
      }
      if constexpr (BLOCK) {
        if (((kt_begin + it) & 1) == 1) {   // end of the 128-K block: acc += partial * sA * sB
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              #pragma unroll
              for (int q = 0; q < 4; ++q) {
                const int j = SWAP ? nt * 2 + (q & 1) : mt * 2 + (q >> 1);
                acc[mt][nt][q] = fmaf(sa_blk[j] * sb_blk, pacc[mt][nt][q], acc[mt][nt][q]);
                pacc[mt][nt][q] = 0.0f;
              }
            }
          }
        }
      }
    }
    cp_async_wait<0>();

    // ---- accumulator coordinates: element (mt, nt, i) sits at mma-row warp_r0 + 16 mt + 8 (i >> 1) + g,
    //      mma-col warp_c0 + 8 nt + 2 t + (i & 1) ----
    const int g = lane >> 2;
    const int t = lane & 3;
    const int64_t row0_mma = SWAP ? n0 : r0;    // first mma row of the tile (n or logical row)
    const int64_t col0_mma = SWAP ? r0 : n0;

    // ---------------- split-K ----------------
    if (slices > 1) {
      const int64_t tile_elems = static_cast<int64_t>(TM) * TN;
      const int64_t tile_id = dense ? static_cast<int64_t>(r0 / TM) * tiles_n + tn : task;
      uint32_t* const wbase = static_cast<uint32_t*>(args.workspace);
      // One warp owns 32 lanes * 4 values = 512 contiguous bytes per (mt, nt). Writing the partials in
      // that register order (kSerial) keeps the store and the fixed-order sum coalesced; the compact
      // [slice][M][N] layout of kParallel has to stay for its elementwise reducer.
      const int64_t lane4 = ((static_cast<int64_t>(warp) * MT * NT) * 32 + lane) * 4;
      if (args.split_k_mode == SplitKMode::kParallel) {
        uint32_t* const wsp = wbase + static_cast<int64_t>(slice) * M * N;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int lr = warp_r0 + mt * 16 + (i >> 1) * 8 + g;
              const int lc = warp_c0 + nt * 8 + 2 * t + (i & 1);
              const int64_t m = SWAP ? col0_mma + lc : row0_mma + lr;
              const int64_t n = SWAP ? row0_mma + lr : col0_mma + lc;
              const int loc = SWAP ? lc : lr;
              uint32_t bits;
              if constexpr (Atom::kInt) bits = static_cast<uint32_t>(acc[mt][nt][i]);
              else bits = __float_as_uint(acc[mt][nt][i]);
              if (v_id(loc) >= 0 && m < M && n < N) wsp[m * N + n] = bits;
            }
          }
        }
        __syncthreads();
        continue;   // the reducer finishes (dense only)
      }
      uint32_t* const dst = wbase + (tile_id * slices + slice) * tile_elems;
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          uint32_t* const d4 = dst + lane4 + static_cast<int64_t>(mt * NT + nt) * 128;
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            if constexpr (Atom::kInt) d4[i] = static_cast<uint32_t>(acc[mt][nt][i]);
            else d4[i] = __float_as_uint(acc[mt][nt][i]);
          }
        }
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const int prev = atomicAdd(&args.split_k_counters[tile_id], 1);
        s_last = prev == slices - 1;
      }
      __syncthreads();
      if (!s_last) continue;
      __threadfence();
      const uint32_t* const src0 = wbase + tile_id * slices * tile_elems;
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][nt][i] = Acc(0);
        }
      }
      for (int sl = 0; sl < slices; ++sl) {   // slice order, never arrival order
        const uint32_t* const src = src0 + static_cast<int64_t>(sl) * tile_elems;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const uint32_t* const s4 = src + lane4 + static_cast<int64_t>(mt * NT + nt) * 128;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t bits = __ldcg(s4 + i);
              if constexpr (Atom::kInt) acc[mt][nt][i] += static_cast<int32_t>(bits);
              else acc[mt][nt][i] += __uint_as_float(bits);
            }
          }
        }
      }
    }

    // ---------------- epilogue ----------------
    // The accumulators go through shared memory one warp-row group at a time; one (non-unrolled) loop
    // finishes and stores them with coalesced access.
    const bool glu = (e.ops & kEpiGLU) != 0u;
    const bool quant = (e.ops & kEpiQuantOut) != 0u;
    const int db = e.glu_interleave > 0 ? e.glu_interleave : 1;
    const DType dt = args.d_dtype;
    const int64_t d_es = dt == DType::kFloat32 ? 4 : 2;
    void* const dbase = static_cast<unsigned char*>(args.d) + static_cast<int64_t>(zb) * args.batch_stride_d * d_es;
    EpiTask tk;
    tk.colslot = dense ? 0 : slot;
    tk.alpha = e.alpha_ptr != nullptr ? e.alpha_ptr[tk.colslot] : e.alpha;
    tk.xs_row = nullptr;
    tk.ws_col = nullptr;
    if (!BLOCK) {
      if (opX.scale != nullptr) {
        const float* b = static_cast<const float*>(opX.scale) + static_cast<int64_t>(zb) * opX.scale_batch_stride;
        if (opX.granularity == ScaleGranularity::kPerTensor) tk.alpha *= b[0];
        else tk.xs_row = b;
      }
      if (opW.scale != nullptr) {
        const float* b = static_cast<const float*>(opW.scale) + static_cast<int64_t>(dense ? zb : slot) * opW.scale_batch_stride;
        if (opW.granularity == ScaleGranularity::kPerTensor) tk.alpha *= b[0];
        else tk.ws_col = b;
      }
    }
    const bool azp = Atom::kInt && (e.ops & kEpiAzpCorrection) != 0u;
    constexpr int LDC = BC + 4;
    // Warp rows staged per pass: the whole tile when it fits the pipeline buffers (the accumulators are then
    // dead during the stores), else one warp row per pass.
    constexpr int WPASS = static_cast<size_t>(BR) * LDC * sizeof(float) <= Traits::smem_bytes ? WM : 1;
    constexpr int PASS_ROWS = BR / WM * WPASS;        // mma rows of one pass
    constexpr int LR = SWAP ? BC : PASS_ROWS;         // logical rows of one pass
    constexpr int LC = SWAP ? PASS_ROWS : BC;         // logical columns (n) of one pass
    static_assert(static_cast<size_t>(PASS_ROWS) * LDC * sizeof(float) <= Traits::smem_bytes, "epilogue staging fits the pipeline buffers");
    float* const sC = reinterpret_cast<float*>(smem_raw);
    __syncthreads();   // the pipeline buffers are dead

    for (int wi = 0; wi < WM; wi += WPASS) {
      if (warp / WN >= wi && warp / WN < wi + WPASS) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const int r = (warp / WN - wi) * (BR / WM) + mt * 16 + g;
            const int c = warp_c0 + nt * 8 + 2 * t;
            float2 lo, hi;
            if constexpr (Atom::kInt) {
              lo = make_float2(__int_as_float(acc[mt][nt][0]), __int_as_float(acc[mt][nt][1]));
              hi = make_float2(__int_as_float(acc[mt][nt][2]), __int_as_float(acc[mt][nt][3]));
            } else {
              lo = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
              hi = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
            }
            *reinterpret_cast<float2*>(&sC[r * LDC + c]) = lo;
            *reinterpret_cast<float2*>(&sC[(r + 8) * LDC + c]) = hi;
          }
        }
      }
      __syncthreads();
      const int lrow0 = SWAP ? 0 : wi * (BR / WM);      // first local logical row of the pass
      const int64_t n0p = SWAP ? n0 + wi * (BR / WM) : n0;
      const int lco = glu ? LC / 2 : LC;                // output columns of the pass
      // raw accumulator at (local row rr, local column cc) of the pass -> fp32, azp applied in the integers
      auto accv = [&](int rr, int cc, int64_t vid, int64_t n) -> float {
        const float raw = SWAP ? sC[cc * LDC + rr] : sC[rr * LDC + cc];
        if constexpr (Atom::kInt) {
          int32_t iv = __float_as_int(raw);
          if (azp) iv -= e.azp[vid] * e.colsum_b[static_cast<int64_t>(tk.colslot) * N + n];
          return static_cast<float>(iv);
        } else {
          (void)vid; (void)n;
          return raw;
        }
      };
      // finished value of output column oc_l (local, after GLU) of local row rr (tile row lr)
      auto final_value = [&](int rr, int oc_l, int64_t vid, int lr) -> float {
        const int64_t xr = tk.xs_row != nullptr ? x_row(lr) : 0;
        float v;
        if (glu) {
          // blocks of db gate columns, then db up columns; n0p is a multiple of 2 db, so both sit in this pass
          const int lc = (oc_l / db) * 2 * db + oc_l % db;
          const int64_t gcol = n0p + lc;
          const float gate = epi_linear(accv(rr, lc, vid, gcol), e, tk, vid, xr, gcol);
          const float up = epi_linear(accv(rr, lc + db, vid, gcol + db), e, tk, vid, xr, gcol + db);
          v = glu_runtime(e.act, gate, up, e.glu);
        } else {
          const int64_t n = n0p + oc_l;
          v = epi_linear(accv(rr, oc_l, vid, n), e, tk, vid, xr, n);
          if (e.ops & kEpiAct) v = act_outlined(e.act, v, e.glu);
        }
        return epi_tail(v, e, dt, vid, (glu ? n0p / 2 : n0p) + oc_l);
      };
      const int64_t n_out = glu ? N / 2 : N;
      const int64_t oc0 = glu ? n0p / 2 : n0p;
      if (quant && e.quant_out.mode == ActQuantMode::kPerGroup) {
        // one warp per (row, group): amax over the group, s = max(amax, 1e-10) / qmax, q = x / s
        const ActQuantOut& qo = e.quant_out;
        const int gs = qo.group_size;
        const int gpr = lco / gs;
        const bool i8 = qo.q_dtype == DType::kInt8;
        const float qmax = i8 ? 127.0f : 448.0f;
        for (int idx = warp; idx < LR * gpr; idx += kThreads / 32) {
          const int r = idx / gpr;
          const int gi = idx - r * gpr;
          const int lr = lrow0 + r;
          const int64_t vid = v_id(lr);
          if (vid < 0) continue;
          const int64_t oc_first = oc0 + static_cast<int64_t>(gi) * gs;
          if (oc_first >= n_out) continue;
          float x[4];
          float amax = 0.0f;
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            x[j] = 0.0f;
            if (j * 32 < gs) {
              const int ocl = gi * gs + j * 32 + lane;
              x[j] = final_value(r, ocl, vid, lr);
              amax = fmaxf(amax, fabsf(x[j]));
            }
          }
          #pragma unroll
          for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
          float s = fmaxf(amax, 1e-10f) / qmax;
          if (qo.scale_ue8m0) s = ue8m0_ceil(s);
          int64_t drow = vid;
          if (e.ops & kEpiScatterRows) drow = e.scatter_rows[vid];
          if (drow < 0) continue;
          const int64_t grp = oc_first / gs;
          if (lane == 0) {
            const int64_t soff = qo.scale_layout == ScaleLayout::kColMajorTma ? grp * qo.scale_row_stride + drow
                                                                              : drow * qo.scale_row_stride + grp;
            static_cast<float*>(qo.scales)[soff] = s;
          }
          uint8_t* qrow = static_cast<uint8_t*>(qo.q) + drow * qo.q_row_stride + oc_first;
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            if (j * 32 < gs) qrow[j * 32 + lane] = quant_byte(__fdiv_rn(x[j], s), i8);
          }
        }
      } else {
        const int per_row = (lco + 1) / 2;
        const int items = LR * per_row;
        for (int idx = tid; idx < items; idx += kThreads) {
          const int r = idx / per_row;
          const int cv = (idx - r * per_row) * 2;     // first output column (local)
          const int lr = lrow0 + r;
          const int64_t vid = v_id(lr);
          if (vid < 0) continue;
          const int64_t oc = oc0 + cv;
          if (oc >= n_out || cv >= lco) continue;
          const bool has1 = oc + 1 < n_out && cv + 1 < lco;
          const float v0 = final_value(r, cv, vid, lr);
          const float v1 = has1 ? final_value(r, cv + 1, vid, lr) : 0.0f;
          if (quant) {   // kStaticPerTensor
            const ActQuantOut& qo = e.quant_out;
            const float s = *qo.static_scale;
            const bool i8 = qo.q_dtype == DType::kInt8;
            int64_t drow = vid;
            if (e.ops & kEpiScatterRows) drow = e.scatter_rows[vid];
            if (drow >= 0) {
              uint8_t* q = static_cast<uint8_t*>(qo.q) + drow * qo.q_row_stride + oc;
              q[0] = quant_byte(__fdiv_rn(v0, s), i8);
              if (has1) q[1] = quant_byte(__fdiv_rn(v1, s), i8);
            }
          } else {
            store_pair_rt(dt, e, dbase, args.ldd, vid, oc, v0, v1, has1);
          }
        }
      }
      __syncthreads();   // sC is overwritten by the next pass
    }

    // ---- kEpiRowSumSq: the CTAs of the first N tile own the rows of their M tile ----
    if ((e.ops & kEpiRowSumSq) != 0u && tn == 0) {
      for (int r = warp; r < TM; r += kThreads / 32) {
        const int64_t vid = v_id(r);
        if (vid < 0) continue;
        const unsigned char* row = xbase + static_cast<int64_t>(x_row(r)) * ldx_b;
        float s = 0.0f;
        if constexpr (Atom::kElemBytes == 4) s = row_sumsq_warp<float>(reinterpret_cast<const float*>(row), K, lane);
        else if constexpr (Atom::kElemBytes == 2) s = row_sumsq_warp<typename Atom::elem_t>(reinterpret_cast<const typename Atom::elem_t*>(row), K, lane);
        else (void)row;   // refused by the host for 8-bit operands
        if (lane == 0) e.row_sumsq[vid] = s;
      }
    }
    __syncthreads();   // the row tables and the task slots are rewritten by the next task
  }
#else
  (void)P;
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

/// Operand atom of a launch.
enum class AtomKind : int { kBf16 = 0, kF16 = 1, kE4M3 = 2, kS8 = 3, kTf32 = 4 };

constexpr int atom_bytes(AtomKind a) {
  switch (a) {
    case AtomKind::kE4M3:
    case AtomKind::kS8: return 1;
    case AtomKind::kTf32: return 4;
    default: return 2;
  }
}

int sm_count(const LaunchContext& ctx) {
  if (ctx.compute_units > 0) return ctx.compute_units;
  int v = 16;
  cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, ctx.device);
  return v;
}

template <typename Atom, int BR, int BC, int ST, int WM, int WN, bool SWAP, bool BLOCK, bool AF32>
Status launch_mma(const MmaGemmParams& p, int64_t persistent_ctas, cudaStream_t s) {
  using Tr = MmaTraits<Atom, BR, BC, ST, WM, WN, SWAP, BLOCK, AF32>;
  constexpr size_t kTables = static_cast<size_t>(Tr::tile_m) * 8;   // row tables of the grouped schedules
  static const cudaError_t attr =
      cudaFuncSetAttribute(gemm_mma_sync_kernel<Tr>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Tr::smem_bytes + kTables));
  if (attr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "gemm_mma_sync_kernel: %s", cudaGetErrorString(attr));
  }
  const ScaledGemmArgs& a = p.args;
  const size_t smem = Tr::smem_bytes + (p.rows.schedule == MmaSchedule::kDense ? 0 : kTables);
  const int slices = std::max(a.split_k_slices, 1);
  int64_t ctas;
  unsigned gz;
  if (p.rows.schedule == MmaSchedule::kDense) {
    const int64_t rows = SWAP ? a.n : a.m;
    const int64_t cols = SWAP ? a.m : a.n;
    const int64_t tiles = ((rows + BR - 1) / BR) * ((cols + BC - 1) / BC);
    ctas = tiles;
    gz = static_cast<unsigned>(a.batch * slices);
    if (slices > 1 && a.split_k_mode == SplitKMode::kSerial) {
      // Slices interleaved in x: block x = slice * tiles + tile. A tile that only fills part of the
      // part then still runs its slices side by side (two CTAs per SM in the small-M band): block i
      // and block i + tiles land on the same SM, so the slices reduce once both have written.
      ctas = tiles * slices;
      gz = static_cast<unsigned>(a.batch);
    }
  } else {
    static const int occ = [] {
      int v = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, gemm_mma_sync_kernel<Tr>, Tr::threads, Tr::smem_bytes + kTables);
      return std::max(v, 1);
    }();
    ctas = std::max<int64_t>(1, persistent_ctas) * occ;   // persistent: every CTA walks the task list
    gz = static_cast<unsigned>(slices);
  }
  if (ctas > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "gemm: too many tiles");
  gemm_mma_sync_kernel<Tr><<<dim3(static_cast<unsigned>(ctas), 1u, gz), Tr::threads, smem, s>>>(p);
  return launch_error("gemm_mma_sync_kernel");
}

/// Tile table. tm / tn are logical (tm along M, tn along N); a swap_ab tile loads the weights as the
/// mma-A operand. glu_span: output columns one epilogue pass covers before GLU (kEpiGLU needs
/// span % (2 * glu_interleave) == 0; kEpiQuantOut per-group needs (span / 2 or span) % group == 0).
struct TileCfg {
  int16_t tm, tn;
  int8_t stages;
  bool swap;
  size_t smem;
  int16_t glu_span;
  bool block_ok;      // FP8 block promotion fits the registers
};
constexpr TileCfg kTiles[] = {
    {128, 128, 3, false, 3 * 256 * 64, 128, false},
    {128, 64, 4, false, 4 * 192 * 64, 64, true},
    {64, 128, 4, false, 4 * 192 * 64, 128, true},
    {64, 64, 4, false, 4 * 128 * 64, 64, true},
    {64, 128, 4, true, 4 * 192 * 64, 128, true},   // swap: 64 tokens x 128 channels (one pass of 128 channels)
    {32, 128, 8, true, 8 * 160 * 64, 128, true},   // swap: 32 tokens x 128 channels, 128-byte row fetch
    {16, 128, 8, true, 8 * 144 * 64, 128, true},   // swap: 16 tokens x 128 channels (MoE blocks of 16, LoRA decode)
    {64, 256, 3, false, 3 * 320 * 64, 256, false}, // 256-wide pass: GLU + 1x128 quantized output
    // 16-bit-only small-M tiles: 32 x 64 and 16 x 64 with 8 stages (2 CTAs per SM, 4 prefetch groups in
    // flight); the 16-row tile doubles the N grid for the M <= 64 band so all SMs get work.
    {32, 64, 8, true, 8 * 96 * 64, 64, true},
    {16, 64, 8, true, 8 * 80 * 64, 64, true},
};
// The swap tiles of the bandwidth-bound small-M band (m <= 256) carry 8 stages for the 16-bit atoms: the
// loader then issues one 128-byte fetch per operand row into two staging buffers, which needs a whole
// number of buffer pairs in the ring (see MmaTraits::wide_ld and the prefetch-group logic in
// gemm_mma_sync_kernel). The 8- and 32-bit atoms compile those tiles with 4 stages (tile_stages()).
constexpr int kNumTiles = static_cast<int>(sizeof(kTiles) / sizeof(kTiles[0]));
constexpr int kNumTiles16 = 8;   // tiles 8.. compile only for bf16 / fp16
constexpr uint64_t kKernelIdBase = 0x4D4D4100u;   // "MMA" + tile id

/// Tiles compiled for an atom (the explicit instantiation list; compile time of this TU is the budget):
/// bf16 / e4m3 all nine, fp16 and s8 all but the 64 x 256 GLU + 1x128 tile, tf32 the three non-swap mid tiles.
/// The tiles from kNumTiles16 on are 16-bit only (their 6- and 8-stage rings are tuned for the small-M band).
constexpr bool tile_compiled(AtomKind atom, int id) {
  if (id >= kNumTiles16) return atom == AtomKind::kBf16 || atom == AtomKind::kF16;
  switch (atom) {
    case AtomKind::kTf32: return id == 1 || id == 2 || id == 3;
    case AtomKind::kF16:
    case AtomKind::kS8: return id != 7;
    default: return true;
  }
}

template <typename Atom, bool BLOCK, bool AF32>
Status launch_tile(int id, const MmaGemmParams& p, int64_t pctas, cudaStream_t s) {
  // Staging depth of the small-M swap tiles for this atom: 8 stages (128-byte row fetch) for 16-bit
  // operands, 4 stages (64-byte fetch, two CTAs per SM) for the 8- and 32-bit ones. Must match tile_stages().
  constexpr int S5 = Atom::kElemBytes == 2 ? 8 : 4;
  if constexpr (BLOCK) {
    switch (id) {
      case 1: return launch_mma<Atom, 128, 64, 4, 2, 2, false, true, false>(p, pctas, s);
      case 2: return launch_mma<Atom, 64, 128, 4, 2, 2, false, true, false>(p, pctas, s);
      case 3: return launch_mma<Atom, 64, 64, 4, 2, 2, false, true, false>(p, pctas, s);
      case 4: return launch_mma<Atom, 128, 64, 4, 2, 2, true, true, false>(p, pctas, s);
      case 5: return launch_mma<Atom, 128, 32, S5, 4, 1, true, true, false>(p, pctas, s);
      case 6: return launch_mma<Atom, 128, 16, S5, 4, 1, true, true, false>(p, pctas, s);
      default: return detail::set_error(Status::kInternal, "gemm_mma: tile id %d has no block-scaled instantiation", id);
    }
  } else if constexpr (AF32) {
    switch (id) {
      case 3: return launch_mma<Atom, 64, 64, 4, 2, 2, false, false, true>(p, pctas, s);
      case 6: return launch_mma<Atom, 128, 16, 8, 4, 1, true, false, true>(p, pctas, s);
      default: return detail::set_error(Status::kInternal, "gemm_mma: tile id %d has no fp32-activation instantiation", id);
    }
  } else if constexpr (std::is_same<Atom, AtomTf32>::value) {
    switch (id) {
      case 1: return launch_mma<Atom, 128, 64, 4, 2, 2, false, false, false>(p, pctas, s);
      case 2: return launch_mma<Atom, 64, 128, 4, 2, 2, false, false, false>(p, pctas, s);
      case 3: return launch_mma<Atom, 64, 64, 4, 2, 2, false, false, false>(p, pctas, s);
      default: return detail::set_error(Status::kInternal, "gemm_mma: tf32 tile id %d", id);
    }
  } else {
    constexpr bool kWide = std::is_same<Atom, AtomBf16>::value || std::is_same<Atom, AtomE4M3>::value;
    constexpr bool kWide16 = std::is_same<Atom, AtomBf16>::value || std::is_same<Atom, AtomF16>::value;
    switch (id) {
      case 0: return launch_mma<Atom, 128, 128, 3, 2, 2, false, false, false>(p, pctas, s);
      case 1: return launch_mma<Atom, 128, 64, 4, 4, 2, false, false, false>(p, pctas, s);
      case 2: return launch_mma<Atom, 64, 128, 4, 2, 4, false, false, false>(p, pctas, s);
      case 3: return launch_mma<Atom, 64, 64, 4, 2, 2, false, false, false>(p, pctas, s);
      case 4: return launch_mma<Atom, 128, 64, 4, 4, 2, true, false, false>(p, pctas, s);
      case 5: return launch_mma<Atom, 128, 32, S5, 4, 1, true, false, false>(p, pctas, s);
      case 6: return launch_mma<Atom, 128, 16, S5, 4, 1, true, false, false>(p, pctas, s);
      case 7:
        if constexpr (kWide) return launch_mma<Atom, 64, 256, 3, 2, 2, false, false, false>(p, pctas, s);
        else return detail::set_error(Status::kInternal, "gemm_mma: tile 7 is not compiled for this atom");
      case 8:
        if constexpr (kWide16) return launch_mma<Atom, 64, 32, 8, 2, 2, true, false, false>(p, pctas, s);
        else return detail::set_error(Status::kInternal, "gemm_mma: tile 8 is not compiled for this atom");
      case 9:
        if constexpr (kWide16) return launch_mma<Atom, 64, 16, 8, 4, 1, true, false, false>(p, pctas, s);
        else return detail::set_error(Status::kInternal, "gemm_mma: tile 9 is not compiled for this atom");
      default: return detail::set_error(Status::kInternal, "gemm_mma: tile id %d", id);
    }
  }
}

/// The explicit instantiation list: (atom, block promotion, fp32 activations).
Status launch_by_atom(AtomKind atom, bool block, bool af32, int id, const MmaGemmParams& p, int64_t pctas, cudaStream_t s) {
  switch (atom) {
    case AtomKind::kBf16:
      return af32 ? launch_tile<AtomBf16, false, true>(id, p, pctas, s) : launch_tile<AtomBf16, false, false>(id, p, pctas, s);
    case AtomKind::kF16:
      return af32 ? launch_tile<AtomF16, false, true>(id, p, pctas, s) : launch_tile<AtomF16, false, false>(id, p, pctas, s);
    case AtomKind::kE4M3:
      return block ? launch_tile<AtomE4M3, true, false>(id, p, pctas, s) : launch_tile<AtomE4M3, false, false>(id, p, pctas, s);
    case AtomKind::kS8: return launch_tile<AtomS8, false, false>(id, p, pctas, s);
    default: return launch_tile<AtomTf32, false, false>(id, p, pctas, s);
  }
}

/// Staging depth a tile compiles with for an atom. The 16-bit atoms fetch 128 bytes of each operand row per
/// issue (two staging buffers), which needs a whole number of buffer pairs: 8 stages = 4 pairs. The 8- and
/// 32-bit atoms keep the 64-byte fetch and their 4 (or 3) stages, so the small-M tiles keep two CTAs per SM.
constexpr int tile_stages(AtomKind atom, int id) {
  return atom_bytes(atom) == 2 || kTiles[id].stages < 8 ? kTiles[id].stages : 4;
}

/// Tile id of a tactic. The geometry and the staging depth are both part of the kernel: a 32 x 128 swap
/// tile exists with 4 and with 8 stages, and they are different bodies. t.stages == 0 means "the depth the
/// atom compiles for this geometry".
int tile_of(AtomKind atom, const GemmTactic& t) {
  for (int i = 0; i < kNumTiles; ++i) {
    if (kTiles[i].tm != t.tile_m || kTiles[i].tn != t.tile_n || kTiles[i].swap != t.swap_ab) continue;
    if (t.stages != 0 && tile_stages(atom, i) != t.stages) continue;
    return i;
  }
  return -1;
}

/// Dynamic shared memory of a tile for an atom: one staging buffer holds 64 bytes of each operand row.
constexpr size_t tile_smem(AtomKind atom, int id) {
  return static_cast<size_t>(tile_stages(atom, id)) * (kTiles[id].tm + kTiles[id].tn) * 64;
}

GemmTactic make_tactic(int id, SplitKMode mode, int slices, AtomKind atom = AtomKind::kBf16) {
  const int elem_bytes = atom_bytes(atom);
  GemmTactic t{};
  t.tile_m = kTiles[id].tm;
  t.tile_n = kTiles[id].tn;
  t.tile_k = static_cast<int16_t>(64 / elem_bytes);
  t.stages = static_cast<int8_t>(tile_stages(atom, id));
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

/// Static heuristic: a two-CTA-per-SM tile for small and mid M (the per-CTA mainloop rate is the limit
/// there), the eight-warp 64 x 128 tile for large M; split-K only when the tile grid cannot fill half the
/// GPU and K is long. batch_invariant keeps one slice. FP8 block promotion avoids the eight-warp and
/// 64 x 256 tiles (two accumulator sets); min_span is the epilogue pass width the GLU pairs or a
/// per-group quantized output need (the 64 x 256 tile serves GLU + 1x128).
GemmTactic heuristic_tactic(int64_t m, int64_t n, int64_t k, int64_t batch, bool glu, bool block, int min_span,
                            int elem_bytes, const LaunchContext& ctx, AtomKind atom = AtomKind::kBf16) {
  const int64_t sms = ctx.compute_units > 0 ? ctx.compute_units : 16;
  auto fits = [&](int id) { return tile_compiled(atom, id) && kTiles[id].glu_span >= min_span && (!block || kTiles[id].block_ok); };
  int id = -1;
  if (min_span > 128) {
    id = 7;
  } else if (glu && fits(2)) {
    id = 2;   // GLU wants the 128-column passes (any glu_interleave up to 64)
  } else if (m <= 512 && !ctx.batch_invariant) {
    // Small and mid M: the band is limited by the per-CTA mainloop rate (one CTA of four warps reaches
    // ~25% of the tensor pipe), so the tiles are chosen to fit two CTAs per SM and to put enough warps
    // in each. 16 x 64 with the 128-byte row fetch for M <= 64 (64 N tiles put a CTA on every SM),
    // the 32 x 64 tile (8 stages, 2 CTAs) for M = 65..128, and the 64 x 128 swap tile (eight warps,
    // 2 CTAs) from M = 129 on. Measured on SM89, 4096 x 4096, cold L2 (ms): M 32 .059 / 64 .064 /
    // 128 .069 / 256 .104 / 512 .123.
    // FP8 block promotion keeps the 32 x 128 tile through M = 256 (its per-128-K partial accumulator
    // costs more registers in the eight-warp tile: measured .075 vs .065).
    const int pick = m <= 64 ? (fits(9) ? 9 : 6) : (m <= 128 ? (fits(8) ? 8 : 5) : (block && m <= 256 ? 5 : 4));
    id = fits(pick) ? pick : -1;
  }
  if (id < 0) {
    // Mid / large M: the 64 x 128 tile with eight warps per CTA keeps two CTAs (16 warps) per SM and
    // reaches 169 TF on SM89, ~9% above the 128 x 128 tile it replaced. Narrow N keeps the square
    // tiles; the loop then shrinks the tile until the grid fills two waves.
    if (m > 512 && fits(2)) id = 2;
    if (id < 0) {
      static constexpr int order[] = {0, 1, 2, 3};
      for (int o : order) {
        if (!fits(o)) continue;
        id = o;   // fallback: the largest tile that fits
        if (ceil_div(m, kTiles[o].tm) * ceil_div(n, kTiles[o].tn) * batch >= 2 * sms) break;
        // keep shrinking while the grid is short of two waves: more CTAs hide the tail / split-K
      }
    }
    if (id < 0) id = 2;
  }
  const int64_t ctas = ceil_div(m, kTiles[id].tm) * ceil_div(n, kTiles[id].tn) * batch;
  int slices = 1;
  const int64_t kmin = 512 * 2 / elem_bytes;   // >= 1 KB of each row per slice
  // Split-K only when the tile grid cannot fill half the part: on a bandwidth-bound problem the extra
  // CTAs add no bandwidth, and the fp32 partials (slices * tiles * tile) plus the serial reduce cost
  // more than the wave they buy. The small-M band is not split even when its grid is short: measured
  // cold on SM89, every serial split of the 64 x 128 tile at M = 128..512 was slower than one CTA per
  // tile (the second CTA per SM buys less than the partial traffic + fixed-order sum cost).
  if (batch == 1 && !ctx.batch_invariant && 2 * ctas < sms && k >= 4 * kmin) {
    slices = static_cast<int>(std::min<int64_t>(8, ceil_div(2 * sms, ctas)));
    while (slices > 1 && k / slices < kmin) slices >>= 1;
  }
  return make_tactic(id, slices > 1 ? SplitKMode::kSerial : SplitKMode::kNone, slices, atom);
}

/// Epilogue pass width (output columns before GLU) the epilogue needs.
int required_span(const EpilogueParams& e) {
  const bool glu = (e.ops & kEpiGLU) != 0u;
  const int db = e.glu_interleave > 0 ? e.glu_interleave : 1;
  int span = glu ? 2 * db : 1;
  if ((e.ops & kEpiQuantOut) != 0u && e.quant_out.mode == ActQuantMode::kPerGroup) {
    span = std::max(span, (glu ? 2 : 1) * e.quant_out.group_size);
  }
  return span;
}

/// Tile of a block-sorted schedule: tile M divides the block, the pass covers min_span.
int sorted_block_tile(int block_rows, bool block, bool af32, int min_span) {
  if (block_rows == 1) return min_span > 128 ? 7 : 2;   // per-row groups padded to 64 rows
  if (af32) return block_rows % 64 == 0 ? 3 : 6;
  if (min_span > 128 && block_rows % 64 == 0) return 7;
  if (block_rows % 128 == 0 && !block && min_span <= 128) return 0;
  if (block_rows % 64 == 0 && min_span <= 128) return 2;
  if (block_rows % 32 == 0 && min_span <= 32) return 5;
  if (min_span <= 32) return 6;
  return block_rows % 64 == 0 ? 2 : 6;   // resolve_tile reports the conflict
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
    kEpiResidualAdd | kEpiQuantOut | kEpiScatterRows | kEpiAtomicAdd | kEpiClamp | kEpiAzpCorrection | kEpiOutFp32 |
    kEpiRowSumSq | kEpiKvAppend;
constexpr uint32_t kReducerEpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct | kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

Status check_mma_epilogue(const ScaledGemmArgs& args, const LaunchContext& ctx, AtomKind atom, bool af32, int slices) {
  const EpilogueParams& e = args.epi;
  const uint32_t ops = e.ops;
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
  if ((ops & kEpiAzpCorrection) != 0u) {
    if (atom != AtomKind::kS8) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiAzpCorrection belongs to the INT8 recipes");
    if (e.azp == nullptr || e.colsum_b == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiAzpCorrection needs azp [M] and colsum_b [N]");
  }
  if ((ops & kEpiBias) != 0u && e.bias_dtype != DType::kBFloat16 && e.bias_dtype != DType::kFloat16 && e.bias_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: bias must be bf16, fp16 or fp32");
  }
  if (((ops & kEpiOutFp32) != 0u) != (args.d_dtype == DType::kFloat32) && (ops & kEpiQuantOut) == 0u) {
    return detail::set_error(Status::kInvalidArgument, "gemm: kEpiOutFp32 and an fp32 output go together");
  }
  if ((ops & kEpiAtomicAdd) != 0u) {
    if (ctx.deterministic) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiAtomicAdd is disabled by ctx.deterministic");
    if ((ops & kEpiKvAppend) != 0u) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiKvAppend reads the stored value; it cannot follow an atomic accumulate");
    if ((ops & kEpiQuantOut) != 0u) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiQuantOut cannot accumulate atomically");
    if (slices > 1) return detail::set_error(Status::kNotImplemented, "gemm: kEpiAtomicAdd with split-K");
  }
  if ((ops & kEpiKvAppend) != 0u && (e.kv_cols < 0 || e.kv_col_begin < 0 || e.kv_stride < e.kv_cols)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: kv_stride must cover kv_cols");
  }
  if ((ops & (kEpiRowSumSq | kEpiKvAppend)) != 0u && args.batch != 1) {
    return detail::set_error(Status::kNotImplemented, "gemm: kEpiRowSumSq / kEpiKvAppend with batch > 1");
  }
  if ((ops & kEpiRowSumSq) != 0u && (atom == AtomKind::kE4M3 || atom == AtomKind::kS8 || af32)) {
    return detail::set_error(Status::kNotImplemented, "gemm: kEpiRowSumSq needs bf16 / fp16 / fp32 activations in memory");
  }
  const int64_t n_out = (ops & kEpiGLU) != 0u ? args.n / 2 : args.n;
  if ((ops & kEpiGLU) != 0u) {
    if (e.glu_layout != GateLayout::kInterleaved) {
      return detail::set_error(Status::kNotImplemented,
                               "gemm: kEpiGLU on the tile GEMM needs GateLayout::kInterleaved (use glu_weight_interleave, or dual_gemm for separate weights)");
    }
    const int db = e.glu_interleave > 0 ? e.glu_interleave : 1;
    if (args.n % (2 * db) != 0) return detail::set_error(Status::kUnsupportedShape, "gemm: kEpiGLU needs N %% (2 * glu_interleave) == 0");
    if (db > 128) return detail::set_error(Status::kUnsupportedShape, "gemm: glu_interleave must be 1..128");
  }
  if ((ops & kEpiQuantOut) != 0u) {
    const ActQuantOut& q = e.quant_out;
    if ((ops & kEpiKvAppend) != 0u) return detail::set_error(Status::kInvalidArgument, "gemm: kEpiKvAppend copies the stored D; it cannot follow kEpiQuantOut");
    if (q.q == nullptr || (q.q_dtype != DType::kFloat8E4M3 && q.q_dtype != DType::kInt8) || q.fp8 != Fp8Encoding::kOCP) {
      return detail::set_error(Status::kUnsupportedDtype, "gemm: kEpiQuantOut writes OCP e4m3 or int8");
    }
    if (q.q_row_stride < n_out) return detail::set_error(Status::kInvalidArgument, "gemm: quant_out.q_row_stride smaller than the output columns");
    if (q.mode == ActQuantMode::kStaticPerTensor) {
      if (q.static_scale == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: static quantized output needs static_scale");
    } else if (q.mode == ActQuantMode::kPerGroup) {
      if (q.group_size != 32 && q.group_size != 64 && q.group_size != 128) {
        return detail::set_error(Status::kUnsupportedShape, "gemm: per-group quantized output needs group 32, 64 or 128");
      }
      if (q.scales == nullptr || q.scale_row_stride < 1 || (q.scale_layout != ScaleLayout::kRowMajor && q.scale_layout != ScaleLayout::kColMajorTma)) {
        return detail::set_error(Status::kInvalidArgument, "gemm: per-group quantized output needs fp32 scales, row-major or TMA column-major");
      }
      if (q.zero_points != nullptr || !q.symmetric) return detail::set_error(Status::kNotImplemented, "gemm: asymmetric quantized output");
      if (n_out % q.group_size != 0) return detail::set_error(Status::kUnsupportedShape, "gemm: output columns must be a multiple of the quant group");
    } else {
      return detail::set_error(Status::kNotImplemented, "gemm: kEpiQuantOut supports kStaticPerTensor and kPerGroup (the per-token amax spans CTAs)");
    }
  } else if (args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "gemm: null output");
  }
  if ((ops & kEpiQuantOut) == 0u && args.ldd < n_out) {
    return detail::set_error(Status::kInvalidArgument, (ops & kEpiGLU) != 0u ? "gemm: kEpiGLU output has N / 2 columns; ldd is smaller" : "gemm: ldd smaller than N");
  }
  return Status::kSuccess;
}

/// Tactic -> tile id, with the constraints GLU, quantized output, block promotion and the device impose.
Status resolve_tile(const GemmTactic& t, const MmaGemmParams& p, AtomKind atom, bool block, bool af32, const LaunchContext& ctx, int* out_id) {
  const ScaledGemmArgs& args = p.args;
  const int id = tile_of(atom, t);
  *out_id = id;
  if (id < 0 || (!af32 && !tile_compiled(atom, id))) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: no mma.sync tile %dx%d (swap_ab %d)", t.tile_m, t.tile_n, t.swap_ab ? 1 : 0);
  }
  const int stages = tile_stages(atom, id);
  if (t.stages != 0 && t.stages != stages) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: tile %dx%d is compiled with %d stages for this atom", t.tile_m, t.tile_n, stages);
  }
  if (tile_smem(atom, id) > static_cast<size_t>(smem_limit(ctx.device))) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: tile %dx%d needs %zu B of shared memory", t.tile_m, t.tile_n, tile_smem(atom, id));
  }
  if (block && !kTiles[id].block_ok) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: FP8 block promotion has no %dx%d tile (two accumulator sets)", t.tile_m, t.tile_n);
  }
  if (af32 && id != 3 && id != 6) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: fp32 activations run on the 64x64 and swap 16x128 tiles");
  }
  const uint32_t ops = args.epi.ops;
  const bool glu = (ops & kEpiGLU) != 0u;
  if (glu) {
    const int db = args.epi.glu_interleave > 0 ? args.epi.glu_interleave : 1;
    if (kTiles[id].glu_span % (2 * db) != 0) {
      return detail::set_error(Status::kUnsupportedShape, "gemm: glu_interleave %d needs a tile whose epilogue pass covers a multiple of %d columns (tile %dx%d covers %d)",
                               db, 2 * db, t.tile_m, t.tile_n, kTiles[id].glu_span);
    }
  }
  if ((ops & kEpiQuantOut) != 0u && args.epi.quant_out.mode == ActQuantMode::kPerGroup) {
    const int span = glu ? kTiles[id].glu_span / 2 : kTiles[id].glu_span;
    if (span % args.epi.quant_out.group_size != 0) {
      return detail::set_error(Status::kUnsupportedShape, "gemm: quant group %d needs a tile whose pass covers a multiple of it (tile %dx%d covers %d output columns)",
                               args.epi.quant_out.group_size, t.tile_m, t.tile_n, span);
    }
  }
  if (p.rows.schedule == MmaSchedule::kSortedBlocks && p.rows.block_rows != 1 && (p.rows.block_rows % kTiles[id].tm) != 0) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: tile M %d must divide the sorted block of %d rows", kTiles[id].tm, p.rows.block_rows);
  }
  if (block && kTiles[id].tn > 128) {
    return detail::set_error(Status::kUnsupportedShape, "gemm: FP8 block promotion needs a tile N within one 128 block");
  }
  return Status::kSuccess;
}

Status split_workspace_bytes(int64_t m, int64_t n, int64_t tiles, int tile_m, int tile_n, SplitKMode mode, int slices,
                             size_t* partial_bytes, size_t* counter_bytes) {
  *partial_bytes = 0;
  *counter_bytes = 0;
  if (slices <= 1 || mode == SplitKMode::kNone) return Status::kSuccess;
  if (mode != SplitKMode::kSerial && mode != SplitKMode::kParallel) {
    return detail::set_error(Status::kNotImplemented, "gemm: stream-K is not implemented");
  }
  // kSerial stores one [tile_m][tile_n] block per (tile, slice) in register order (see the kernel);
  // kParallel keeps the compact [slice][M][N] planes its elementwise reducer reads.
  *partial_bytes = mode == SplitKMode::kSerial
                       ? static_cast<size_t>(slices) * tiles * tile_m * tile_n * sizeof(float)
                       : static_cast<size_t>(slices) * m * n * sizeof(float);
  if (mode == SplitKMode::kSerial) *counter_bytes = static_cast<size_t>(tiles) * sizeof(int32_t);
  return Status::kSuccess;
}

size_t align256(size_t x) { return (x + 255) & ~static_cast<size_t>(255); }

bool is_dense_recipe(GemmRecipe r) { return r == GemmRecipe::kBF16 || r == GemmRecipe::kFP16; }
bool is_fp8_recipe(GemmRecipe r) {
  return r == GemmRecipe::kFP8Tensorwise || r == GemmRecipe::kFP8Rowwise || r == GemmRecipe::kFP8Blockwise ||
         r == GemmRecipe::kFP8BlockwiseUE8M0;
}
bool is_int8_recipe(GemmRecipe r) { return r == GemmRecipe::kINT8Tensorwise || r == GemmRecipe::kINT8Rowwise; }
bool is_block_recipe(GemmRecipe r) { return r == GemmRecipe::kFP8Blockwise || r == GemmRecipe::kFP8BlockwiseUE8M0; }
bool has_mma_recipe(GemmRecipe r) { return is_dense_recipe(r) || r == GemmRecipe::kTF32 || is_fp8_recipe(r) || is_int8_recipe(r); }
AtomKind recipe_atom(GemmRecipe r) {
  if (is_fp8_recipe(r)) return AtomKind::kE4M3;
  if (is_int8_recipe(r)) return AtomKind::kS8;
  if (r == GemmRecipe::kTF32) return AtomKind::kTf32;
  return r == GemmRecipe::kFP16 ? AtomKind::kF16 : AtomKind::kBf16;
}

/// Recipe + operand dtypes -> atom. af32: fp32 activations with 16-bit weights (LoRA expand).
Status atom_of(const ScaledGemmArgs& a, const LaunchContext& ctx, AtomKind* atom, bool* af32) {
  *af32 = false;
  const GemmRecipe r = a.recipe;
  if (is_dense_recipe(r)) {
    const DType w = r == GemmRecipe::kBF16 ? DType::kBFloat16 : DType::kFloat16;
    if (a.b.dtype != w || (a.a.dtype != w && a.a.dtype != DType::kFloat32)) {
      return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync needs bf16 x bf16 (kBF16) or fp16 x fp16 (kFP16)");
    }
    *af32 = a.a.dtype == DType::kFloat32;
    *atom = r == GemmRecipe::kBF16 ? AtomKind::kBf16 : AtomKind::kF16;
    return Status::kSuccess;
  }
  if (r == GemmRecipe::kTF32) {
    if (!ctx.allow_tf32) return detail::set_error(Status::kInvalidArgument, "gemm: kTF32 needs ctx.allow_tf32");
    if (a.a.dtype != DType::kFloat32 || a.b.dtype != DType::kFloat32) return detail::set_error(Status::kUnsupportedDtype, "gemm: kTF32 takes fp32 operands");
    *atom = AtomKind::kTf32;
    return Status::kSuccess;
  }
  if (is_fp8_recipe(r)) {
    if (a.a.dtype != DType::kFloat8E4M3 || a.b.dtype != DType::kFloat8E4M3 || a.a.fp8 != Fp8Encoding::kOCP || a.b.fp8 != Fp8Encoding::kOCP) {
      return detail::set_error(Status::kUnsupportedDtype, "gemm: the FP8 mma.sync recipes take OCP e4m3 activations and weights");
    }
    *atom = AtomKind::kE4M3;
    return Status::kSuccess;
  }
  if (is_int8_recipe(r)) {
    if (a.a.dtype != DType::kInt8 || a.b.dtype != DType::kInt8) return detail::set_error(Status::kUnsupportedDtype, "gemm: the INT8 recipes take s8 activations and weights");
    *atom = AtomKind::kS8;
    return Status::kSuccess;
  }
  return detail::set_error(Status::kNotImplemented, "gemm: recipe %d has no mma.sync body", static_cast<int>(r));
}

Status check_rows(const MmaGemmParams& p, int64_t a_rows_hint) {
  const MmaRowMap& rm = p.rows;
  (void)a_rows_hint;
  switch (rm.schedule) {
    case MmaSchedule::kDense:
      if (rm.row_ids != nullptr || rm.a_rows_by_id || rm.out_rows_by_id || rm.b_ptrs != nullptr) {
        return detail::set_error(Status::kInvalidArgument, "gemm: the dense schedule takes no row ids or pointer array");
      }
      return Status::kSuccess;
    case MmaSchedule::kRowGroups:
      if (rm.num_groups < 1 || (rm.group_rows == nullptr && rm.group_rows32 == nullptr)) {
        return detail::set_error(Status::kInvalidArgument, "gemm: kRowGroups needs num_groups and group_rows");
      }
      break;
    case MmaSchedule::kMaskedGroups:
      if (rm.num_groups < 1 || rm.masked_rows == nullptr || rm.max_rows < 1) {
        return detail::set_error(Status::kInvalidArgument, "gemm: kMaskedGroups needs num_groups, masked_rows and max_rows");
      }
      if (static_cast<int64_t>(rm.num_groups) * rm.max_rows > p.args.m) {
        return detail::set_error(Status::kInvalidArgument, "gemm: kMaskedGroups needs M >= num_groups * max_rows");
      }
      break;
    case MmaSchedule::kSortedBlocks:
      if (rm.block_group == nullptr || rm.block_rows < 1 || (rm.num_rows == nullptr && rm.block_rows != 1)) {
        return detail::set_error(Status::kInvalidArgument, "gemm: kSortedBlocks needs block_group, block_rows and num_rows");
      }
      break;
    case MmaSchedule::kTaskMap:
      if (rm.task_map == nullptr || rm.num_tasks == nullptr || rm.num_groups < 1 ||
          (rm.masked_rows == nullptr ? (rm.group_rows == nullptr && rm.group_rows32 == nullptr) : rm.max_rows < 1)) {
        return detail::set_error(Status::kInvalidArgument, "gemm: kTaskMap needs task_map, num_tasks and the group rows (offsets or masked_rows)");
      }
      break;
    default:
      return detail::set_error(Status::kInvalidArgument, "gemm: unknown schedule");
  }
  if (p.args.batch != 1) return detail::set_error(Status::kInvalidArgument, "gemm: grouped schedules take batch == 1 (groups replace batches)");
  if (rm.a_rows_by_id && (rm.row_ids == nullptr || rm.a_row_div < 1)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: a_rows_by_id needs row_ids and a_row_div >= 1");
  }
  if (rm.out_rows_by_id && rm.row_ids == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: out_rows_by_id needs row_ids");
  return Status::kSuccess;
}

}  // namespace

// =============================================================================
// Entry points
// =============================================================================
bool mma_selects(const ScaledGemmArgs& args, const GemmTactic* tactic, const LaunchContext& ctx) {
  if (tactic != nullptr) return tactic->mainloop == MainloopSchedule::kMultistageMmaSync;
  GemmTactic t;
  return cache_find(make_key(args.recipe, args.m, args.n, args.k, ctx), &t);
}

Status mma_workspace(const GemmProblem& p, const GemmTactic* tactic, const LaunchContext& ctx, WorkspaceRequirements* out) {
  GemmTactic t;
  const bool block = is_block_recipe(p.recipe);
  const int eb = is_fp8_recipe(p.recipe) || is_int8_recipe(p.recipe) ? 1 : (p.recipe == GemmRecipe::kTF32 ? 4 : 2);
  if (tactic != nullptr) {
    t = *tactic;
  } else if (!cache_find(make_key(p.recipe, p.m, p.n, p.k, ctx), &t)) {
    t = heuristic_tactic(p.m, p.n, p.k, p.batch, (p.epilogue_ops & kEpiGLU) != 0u, block, (p.epilogue_ops & kEpiGLU) != 0u ? 2 : 1, eb, ctx,
                         recipe_atom(p.recipe));
  }
  const int id = tile_of(recipe_atom(p.recipe), t);
  if (id < 0) return detail::set_error(Status::kUnsupportedShape, "gemm: no mma.sync tile %dx%d", t.tile_m, t.tile_n);
  const int64_t tiles = ceil_div(p.m, kTiles[id].tm) * ceil_div(p.n, kTiles[id].tn);
  size_t partial = 0, counters = 0;
  Status st = split_workspace_bytes(p.m, p.n, tiles, kTiles[id].tm, kTiles[id].tn, t.split_k_mode, t.split_k_slices, &partial, &counters);
  if (st != Status::kSuccess) return st;
  out->device_scratch_bytes = align256(counters) + partial;
  return Status::kSuccess;
}

Status launch_gemm_mma_sync(const MmaGemmParams& params_in, const GemmTactic* tactic, const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "gemm: CUDA backend only");
  MmaGemmParams p = params_in;
  ScaledGemmArgs& args = p.args;
  AtomKind atom = AtomKind::kBf16;
  bool af32 = false;
  Status st = atom_of(args, ctx, &atom, &af32);
  if (st != Status::kSuccess) return st;
  const bool block = is_block_recipe(args.recipe);
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync output is bf16, fp16 or fp32 (kEpiQuantOut for 8-bit outputs)");
  }
  if ((atom == AtomKind::kBf16 || atom == AtomKind::kF16) && args.d_dtype != args.b.dtype && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync output is the operand dtype or fp32");
  }
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kNotImplemented, "gemm: MN-major operands are not implemented on the mma.sync path (weight_transpose first)");
  }
  if (args.m < 1 || args.n < 1 || args.k < 1 || args.batch < 1) return detail::set_error(Status::kInvalidArgument, "gemm: empty problem");
  if (args.a.data == nullptr || (args.b.data == nullptr && p.rows.b_ptrs == nullptr)) return detail::set_error(Status::kInvalidArgument, "gemm: null operand");
  if (args.a.ld < args.k || args.b.ld < args.k) return detail::set_error(Status::kInvalidArgument, "gemm: leading dimensions smaller than K");
  if (args.m > 0x7FFFFFFF || args.n > 0x7FFFFFFF || args.k > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "gemm: extents beyond int32");
  if (args.workspace != nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: pass workspace through the Workspace argument");
  if (args.batch > 1 && (args.a.batch_stride == 0 || args.b.batch_stride == 0 || args.batch_stride_d == 0)) {
    return detail::set_error(Status::kInvalidArgument, "gemm: batched problems need batch strides");
  }
  if (args.batch > 65535) return detail::set_error(Status::kUnsupportedShape, "gemm: batch > 65535");
  st = check_rows(p, args.m);
  if (st != Status::kSuccess) return st;
  const int eb = atom_bytes(atom);
  const int xb = af32 ? 4 : eb;
  const int kmul = 16 / eb;   // K elements per 16-byte chunk
  bool ptrs_ok = aligned(args.a.data, 16) && (args.b.data == nullptr || aligned(args.b.data, 16));
  if (p.rows.b_ptrs != nullptr) ptrs_ok = ptrs_ok && aligned(p.rows.b_ptrs, 8);
  if (!ptrs_ok || (args.a.ld * xb) % 16 != 0 || (args.b.ld * eb) % 16 != 0 || (args.a.batch_stride * xb) % 16 != 0 ||
      (args.b.batch_stride * eb) % 16 != 0 || args.k % kmul != 0) {
    return detail::set_error(Status::kMisaligned, "gemm: mma.sync operands need 16 B alignment (pointers, ld, batch / group stride) and K %% %d == 0", kmul);
  }
  if (block) {
    if (args.k % 128 != 0) return detail::set_error(Status::kUnsupportedShape, "gemm: FP8 blockwise needs K %% 128 == 0");
    if (args.a.scale == nullptr || args.b.scale == nullptr) return detail::set_error(Status::kInvalidArgument, "gemm: FP8 blockwise needs 1x128 A scales and 128x128 B scales");
    const bool ue = args.recipe == GemmRecipe::kFP8BlockwiseUE8M0;
    const DType want = ue ? DType::kFloat8E8M0 : DType::kFloat32;
    if ((args.a.scale_dtype != want && !(args.a.scale_dtype == DType::kFloat8E8M0 && !ue)) ||
        (args.b.scale_dtype != want && !(args.b.scale_dtype == DType::kFloat8E8M0 && !ue))) {
      return detail::set_error(Status::kUnsupportedDtype, "gemm: FP8 block scales are fp32 (kFP8Blockwise; UE8M0 bytes also accepted) or UE8M0 bytes (kFP8BlockwiseUE8M0)");
    }
    if ((args.a.scale_layout != ScaleLayout::kRowMajor && args.a.scale_layout != ScaleLayout::kColMajorTma) ||
        args.b.scale_layout != ScaleLayout::kRowMajor) {
      return detail::set_error(Status::kUnsupportedLayout, "gemm: A scales row-major or TMA column-major, B scales row-major (packed UE8M0 is not implemented)");
    }
    if (args.a.scale_ld == 0) {
      const int64_t rows = p.rows.schedule == MmaSchedule::kDense ? args.m : args.m;
      args.a.scale_ld = args.a.scale_layout == ScaleLayout::kColMajorTma ? ((rows + 3) / 4) * 4 : args.k / 128;
    }
    if (args.b.scale_ld == 0) args.b.scale_ld = args.k / 128;
    if (args.batch > 1) return detail::set_error(Status::kNotImplemented, "gemm: FP8 blockwise with batch > 1 (use fp8_bmm with tensorwise / rowwise scales)");
  } else if (args.a.scale != nullptr || args.b.scale != nullptr) {
    // operand scales of the tensorwise / rowwise recipes: per tensor (per batch / slot) or per token / channel
    if (atom != AtomKind::kE4M3 && atom != AtomKind::kS8) {
      return detail::set_error(Status::kInvalidArgument, "gemm: operand scales belong to the FP8 / INT8 recipes");
    }
    if ((args.a.scale != nullptr && (args.a.scale_dtype != DType::kFloat32 ||
                                     (args.a.granularity != ScaleGranularity::kPerTensor && args.a.granularity != ScaleGranularity::kPerToken))) ||
        (args.b.scale != nullptr && (args.b.scale_dtype != DType::kFloat32 ||
                                     (args.b.granularity != ScaleGranularity::kPerTensor && args.b.granularity != ScaleGranularity::kPerChannel)))) {
      return detail::set_error(Status::kUnsupportedLayout, "gemm: operand scales are fp32, per tensor or per token (A) / per channel (B)");
    }
  }

  const bool grouped = p.rows.schedule != MmaSchedule::kDense;
  const bool glu = (args.epi.ops & kEpiGLU) != 0u;
  const int min_span = required_span(args.epi);
  GemmTactic t;
  if (tactic != nullptr) {
    t = *tactic;
  } else if (p.rows.schedule == MmaSchedule::kSortedBlocks) {
    t = make_tactic(sorted_block_tile(p.rows.block_rows, block, af32, min_span), SplitKMode::kNone, 1, atom);
  } else if (grouped || af32 || !cache_find(make_key(args.recipe, args.m, args.n, args.k, ctx), &t)) {
    // grouped schedules: tiles sized for the average rows per group, no split-K
    const int64_t m_hint = grouped && p.rows.num_groups > 0 ? std::max<int64_t>(1, args.m / p.rows.num_groups) : args.m;
    t = heuristic_tactic(m_hint, args.n, args.k, args.batch, glu, block, min_span, eb, ctx, atom);
    if (af32) t = make_tactic(m_hint <= 16 ? 6 : 3, SplitKMode::kNone, 1, atom);
    if (grouped) {
      t.split_k_mode = SplitKMode::kNone;
      t.split_k_slices = 1;
    }
  }
  if (t.mainloop != MainloopSchedule::kMultistageMmaSync) {
    return detail::set_error(Status::kUnsupportedArch, "gemm: tactic mainloop %d needs a newer architecture", static_cast<int>(t.mainloop));
  }
  int id = -1;
  st = resolve_tile(t, p, atom, block, af32, ctx, &id);
  if (st != Status::kSuccess) return st;

  SplitKMode mode = t.split_k_mode;
  int slices = std::max<int>(t.split_k_slices, 1);
  if (mode == SplitKMode::kNone) slices = 1;
  if (slices == 1) mode = SplitKMode::kNone;
  if (mode != SplitKMode::kNone && mode != SplitKMode::kSerial && mode != SplitKMode::kParallel) {
    return detail::set_error(Status::kNotImplemented, "gemm: stream-K is not implemented");
  }
  if (slices > 1 && args.batch != 1) return detail::set_error(Status::kNotImplemented, "gemm: split-K with batch > 1");
  if (slices > 64) return detail::set_error(Status::kInvalidArgument, "gemm: split_k_slices must be 1..64");
  if (mode == SplitKMode::kParallel &&
      ((args.epi.ops & ~kReducerEpiAllowed) != 0u || atom == AtomKind::kS8 || grouped || args.d_dtype == DType::kFloat8E4M3)) {
    return detail::set_error(Status::kNotImplemented, "gemm: parallel split-K supports the dense schedule, float accumulators and the dense epilogue subset; use kSerial");
  }
  if (slices > 1 && grouped && p.rows.max_tasks < 1) {
    return detail::set_error(Status::kInvalidArgument, "gemm: split-K on a grouped schedule needs rows.max_tasks");
  }
  st = check_mma_epilogue(args, ctx, atom, af32, slices);
  if (st != Status::kSuccess) return st;
  // Arguments are valid; now the exact SM (a machine without it still gets the contract errors above).
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "gemm: the tensor-core path is mma.sync and needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  if (atom == AtomKind::kE4M3 && ctx.arch_id != 0 && (ctx.arch_id < KOPS_TARGET_SM89 || ctx.arch_id >= 9000)) {
    return detail::set_error(Status::kUnsupportedArch, "gemm: FP8 mma.sync (e4m3 m16n8k32) needs SM89 or newer, found arch %d", ctx.arch_id);
  }

  const int64_t tiles = grouped ? p.rows.max_tasks : ceil_div(args.m, kTiles[id].tm) * ceil_div(args.n, kTiles[id].tn);
  size_t partial = 0, counters = 0;
  st = split_workspace_bytes(args.m, args.n, tiles, kTiles[id].tm, kTiles[id].tn, mode, slices, &partial, &counters);
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
  const int64_t pctas = grouped ? static_cast<int64_t>(sm_count(ctx)) : 0;

  if (mode == SplitKMode::kParallel) {
    // Partials land in args.workspace ([slice][M][N] fp32); the reducer applies the dense epilogue.
    MmaGemmParams part = p;
    part.args.epi = EpilogueParams{};   // the mainloop kernel only writes partials
    st = launch_by_atom(atom, block, af32, id, part, pctas, s);
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
  return launch_by_atom(atom, block, af32, id, p, pctas, s);
}

Status mma_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, const Workspace& ws, const LaunchContext& ctx) {
  if (!is_dense_recipe(args.recipe) && args.recipe != GemmRecipe::kTF32) {
    return detail::set_error(Status::kNotImplemented, "gemm: recipe %d goes through fp8_gemm / int8_gemm", static_cast<int>(args.recipe));
  }
  if (is_dense_recipe(args.recipe) && args.a.dtype == DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gemm: mma.sync needs bf16 x bf16 (kBF16) or fp16 x fp16 (kFP16)");
  }
  MmaGemmParams p{};
  p.args = args;
  return launch_gemm_mma_sync(p, tactic, ws, ctx);
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

int recipe_elem_bytes(GemmRecipe r) {
  if (is_fp8_recipe(r) || is_int8_recipe(r)) return 1;
  return r == GemmRecipe::kTF32 ? 4 : 2;
}

std::vector<GemmTactic> candidates(const GemmProblem& p, const LaunchContext& ctx) {
  std::vector<GemmTactic> out;
  const bool glu = (p.epilogue_ops & kEpiGLU) != 0u;
  const bool block = is_block_recipe(p.recipe);
  const int eb = recipe_elem_bytes(p.recipe);
  const int lim = smem_limit(ctx.device);
  const AtomKind atom = recipe_atom(p.recipe);
  const GemmTactic first = heuristic_tactic(p.m, p.n, p.k, p.batch, glu, block, glu ? 2 : 1, eb, ctx, atom);
  out.push_back(first);
  const int64_t kmin = 512 * 2 / eb;
  for (int id = 0; id < kNumTiles; ++id) {
    if (tile_smem(atom, id) > static_cast<size_t>(lim)) continue;
    if (!tile_compiled(atom, id)) continue;
    if (block && !kTiles[id].block_ok) continue;
    if (id == 7 && !glu) continue;   // the 256-wide tile is for GLU + 1x128 quantized outputs
    for (int slices : {1, 2, 4, 8}) {
      if (slices > 1 && (p.batch != 1 || p.k / slices < kmin)) continue;
      GemmTactic t = make_tactic(id, slices > 1 ? SplitKMode::kSerial : SplitKMode::kNone, slices, atom);
      if (t.tile_m == first.tile_m && t.tile_n == first.tile_n && t.swap_ab == first.swap_ab && t.stages == first.stages &&
          t.split_k_slices == first.split_k_slices) {
        continue;
      }
      out.push_back(t);
    }
  }
  return out;
}

DType recipe_operand_dtype(GemmRecipe r) {
  if (is_fp8_recipe(r)) return DType::kFloat8E4M3;
  if (is_int8_recipe(r)) return DType::kInt8;
  if (r == GemmRecipe::kTF32) return DType::kFloat32;
  return r == GemmRecipe::kBF16 ? DType::kBFloat16 : DType::kFloat16;
}

}  // namespace

Status get_tactics(const GemmProblem& problem, GemmBackend backend, const LaunchContext& ctx, GemmTactic* out,
                   int32_t max_tactics, int32_t* num_tactics) {
  Status st = native_only(backend, ctx, "get_tactics");
  if (st != Status::kSuccess) return st;
  if (out == nullptr || num_tactics == nullptr || max_tactics < 0) return detail::set_error(Status::kInvalidArgument, "get_tactics: null output");
  *num_tactics = 0;
  if (!has_mma_recipe(problem.recipe) || problem.m < 1 || problem.n < 1 || problem.k < 1) return Status::kSuccess;
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) return Status::kSuccess;
  if (is_fp8_recipe(problem.recipe) && ctx.arch_id != 0 && (ctx.arch_id < KOPS_TARGET_SM89 || ctx.arch_id >= 9000)) return Status::kSuccess;
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
  if (!has_mma_recipe(problem.recipe)) return detail::set_error(Status::kNotImplemented, "autotune: recipe %d has no mma.sync tactics", static_cast<int>(problem.recipe));
  if (is_fp8_recipe(problem.recipe) && ctx.arch_id != 0 && (ctx.arch_id < KOPS_TARGET_SM89 || ctx.arch_id >= 9000)) {
    return detail::set_error(Status::kUnsupportedArch, "autotune: FP8 mma.sync needs SM89 or newer");
  }
  const cudaStream_t s = stream_of(ctx);
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
    cudaGetLastError();
    return detail::set_error(Status::kInvalidArgument, "autotune: must not run while a CUDA graph is being captured");
  }
  const DType dt = recipe_operand_dtype(problem.recipe);
  const size_t es = static_cast<size_t>(recipe_elem_bytes(problem.recipe));
  const bool block = is_block_recipe(problem.recipe);
  const DType out_dt = (dt == DType::kBFloat16 || dt == DType::kFloat16) ? dt : DType::kBFloat16;
  for (int32_t bi = 0; bi < num_buckets; ++bi) {
    const int64_t m = std::max<int64_t>(m_buckets[bi], 1);
    if (m <= 16) continue;   // the CUDA-core GEMV owns decode shapes
    GemmProblem p = problem;
    p.m = m;
    p.epilogue_ops = problem.epilogue_ops & kEpiGLU;
    void *da = nullptr, *db = nullptr, *dd = nullptr, *dsa = nullptr, *dsb = nullptr;
    const size_t a_bytes = static_cast<size_t>(m * p.k) * es, b_bytes = static_cast<size_t>(p.n * p.k) * es;
    const size_t d_bytes = static_cast<size_t>(m * p.n) * 4;
    const size_t sa_bytes = block ? static_cast<size_t>(m * (p.k / 128)) * 4 : 0;
    const size_t sb_bytes = block ? static_cast<size_t>(ceil_div(p.n, 128) * (p.k / 128)) * 4 : 0;
    bool ok = cudaMalloc(&da, a_bytes) == cudaSuccess && cudaMalloc(&db, b_bytes) == cudaSuccess && cudaMalloc(&dd, d_bytes) == cudaSuccess;
    if (ok && block) ok = cudaMalloc(&dsa, sa_bytes) == cudaSuccess && cudaMalloc(&dsb, sb_bytes) == cudaSuccess;
    if (!ok) {
      cudaFree(da); cudaFree(db); cudaFree(dd); cudaFree(dsa); cudaFree(dsb);
      cudaGetLastError();
      return detail::set_error(Status::kCudaError, "autotune: cannot allocate the timing buffers");
    }
    cudaMemset(da, 0x3C, a_bytes);
    cudaMemset(db, 0x3C, b_bytes);
    if (block) {
      cudaMemset(dsa, 0, sa_bytes);   // zero scales: values do not matter for timing
      cudaMemset(dsb, 0, sb_bytes);
    }
    ScaledGemmArgs args{};
    args.recipe = problem.recipe;
    args.a.data = da; args.a.ld = p.k; args.a.dtype = dt;
    args.b.data = db; args.b.ld = p.k; args.b.dtype = dt;
    if (block) {
      args.a.scale = dsa; args.a.scale_dtype = DType::kFloat32;
      args.b.scale = dsb; args.b.scale_dtype = DType::kFloat32;
      if (problem.recipe == GemmRecipe::kFP8BlockwiseUE8M0) { args.a.scale_dtype = DType::kFloat8E8M0; args.b.scale_dtype = DType::kFloat8E8M0; }
    }
    args.d = dd; args.d_dtype = out_dt; args.ldd = p.n;
    args.m = m; args.n = p.n; args.k = p.k;
    if ((p.epilogue_ops & kEpiGLU) != 0u) {   // timing GLU needs a plausible N / 2 output
      args.epi.ops = kEpiGLU; args.epi.glu_layout = GateLayout::kInterleaved; args.epi.glu_interleave = 1; args.epi.act = ActFn::kSiLU; args.ldd = p.n / 2;
    }
    MmaGemmParams mp{};
    mp.args = args;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best_ms = 1e30f;
    GemmTactic best{};
    bool have = false;
    for (const GemmTactic& t : candidates(p, ctx)) {
      Status r = launch_gemm_mma_sync(mp, &t, ws, ctx);
      if (r != Status::kSuccess) continue;   // e.g. split-K without enough workspace
      constexpr int kReps = 24;
      cudaEventRecord(e0, s);
      for (int i = 0; i < kReps; ++i) launch_gemm_mma_sync(mp, &t, ws, ctx);
      cudaEventRecord(e1, s);
      cudaEventSynchronize(e1);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, e0, e1);
      if (ms < best_ms) { best_ms = ms; best = t; have = true; }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(da); cudaFree(db); cudaFree(dd); cudaFree(dsa); cudaFree(dsb);
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
    if (tile_of(recipe_atom(static_cast<GemmRecipe>(k.recipe)), t) < 0) {
      std::fclose(f);
      return detail::set_error(Status::kInvalidArgument, "tactic_cache_load: tile %dx%d (stages %d) does not exist in this build", tm, tn, stages);
    }
    loaded[k] = t;
  }
  std::fclose(f);
  std::lock_guard<std::mutex> lock(cache_mutex());
  for (const auto& kv : loaded) cache_map()[kv.first] = kv.second;
  return Status::kSuccess;
}

}  // namespace kops::gemm
