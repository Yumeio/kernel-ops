#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/sm120a_impl.cuh                    Catalog: G01-G03, G07
// B13 SM120a arch cell: the mma.sync bodies behind the *_sm120_kernel declarations.
//
// SM120/121 has no wgmma and no tcgen05/TMEM, and less shared memory than SM100, so
// the dense tensor-core path is the B7 frame (cp.async multistage ring, XOR-swizzled
// 64-byte stage rows, ldmatrix.x4, m16n8k16 / m16n8k32) with the SM120 tile table.
// Nothing here is shared with the SM90a/SM100a cells: those need their own pipelines.
//
// Implemented in this pass (correctness first):
//   gemm_sm120_kernel<Traits>               bf16 / fp16 dense, batch in blockIdx.z
//   fp8_gemm_sm120_kernel<Traits, kRecipe>  e4m3 x e4m3, kFP8Tensorwise / kFP8Rowwise
//   fp4_gemm_sm120_kernel<Traits, kRecipe>  kNVFP4 / kMXFP4 (W4A4) on the software-decode body:
//                                           packed e2m1 -> bf16 while staging, block scales applied
//                                           per block boundary (16 K NVFP4 / 32 K MXFP4). NOT the
//                                           hardware block-scaled mma.sync kind::mxf4nvf4; the
//                                           effective recipe is documented on fp4_tile below.
// Not implemented here; the host launcher returns kNotImplemented and names the gap:
//   fp4_gemm_sm120_kernel with kMXFP4xMXFP8 / kMXFP6xMXFP8
//                                           needs a second operand loader (e4m3 / e2m3 A next to an
//                                           e2m1 / e2m3 B) and a mixed-atom decode
//   grouped_gemm_sm120_kernel               B13 grouped rows are owned elsewhere
//
// Contract per CTA (one [TILE_M, TILE_N] output tile of D = epilogue(A . B^T)):
//   * STAGES-deep smem ring; one stage holds 64 bytes of every operand row, i.e.
//     TILE_K = 64 / sizeof(elem) K elements per stage.
//   * loader (16-bit / FP8): cp.async.cg, 16 B per thread chunk, XOR swizzle on row / 2
//     so the eight rows of an ldmatrix span eight different 16-byte bank groups.
//     Out-of-range chunks use cp.async with sz = 0 (zero fill), so the K tail and
//     the rows past M / N contribute exact zeros and never a NaN to a valid
//     output element. The FP4 body decodes while staging instead (see fp4_tile):
//     a 4-byte packed load becomes 16 decoded bytes, with the same ring, swizzle and
//     zero-fill rule.
//   * mma.sync atoms: m16n8k16 (bf16 / fp16), m16n8k32 (e4m3, SM89+ encoding on
//     the SM120 pipe). The K chain is increasing k-steps only, so one output
//     element is the same fp32 sequence whatever the tile shape is; for 16-bit
//     operands it is bit-identical to the B7 gemm_mma_sync_kernel body, which uses
//     the same atom and the same k-step order.
//   * warp layout: 2 warp rows x (TILE_N / 32) warp columns, i.e. 8 warps = 256 threads on a
//     128-wide tile. The mma.sync pipe of one SM sustains one m16n8k16 every ~8 cycles and a warp
//     can only issue one every ~29, so four warps per SMSP are needed to keep the pipe busy while
//     the same warps run ldmatrix + the cp.async issue. The tiles are sized so two CTAs fit both
//     the 100 KB smem opt-in and the register file, which is what puts 16 warps on an SM (measured:
//     a 4-warp CTA reaches 39 TF on 2048^3 bf16, 8 warps with two CTAs per SM 47 TF).
//   * epilogue in registers, fp32, one rounding to the output dtype. The no-op epilogue (alpha only,
//     no operand scale vector) is a separate path that writes two adjacent columns per store; the
//     general path below it is element-wise and keeps the documented op order. Op subset:
//     alpha / row_scale / col_scale / bias / clamp / softcap / act / residual, plus
//     the per-token (A) and per-channel (B) operand scales of the FP8 recipes, in the
//     shared order of GemmParams (kEpiGLU, kEpiQuantOut, kEpiAtomicAdd, kEpiScatterRows,
//     kEpiRoutingWeight, kEpiSigmoidGateMul, kEpiRowSumSq, kEpiKvAppend are rejected by
//     the host, never silently dropped). FP4 block scales are applied by the mainloop.
//
// Launch contract (host side in gemm/dense_gemm.cu: launch_gemm_sm120 and the
// per-recipe bodies sm120_dense_gemm / sm120_fp8_gemm / sm120_fp4_gemm):
//   grid.x = num_tiles_m * num_tiles_n with tile M fastest, grid.z = batch, one CTA per
//   tile, dynamic smem = STAGES * (TILE_M + TILE_N) * 64. No split-K, no swap-AB, no
//   persistent scheduler (all three return kNotImplemented), so a launch needs no
//   workspace. GemmTactic::raster is a hint only: the tile order is fixed to M-fastest
//   and results do not depend on it.
//
// TODO(TMA/WS): a TMA + mbarrier ring (descriptors from GemmTmaParams, one producer thread or warp)
// was implemented and measured on this cell: on the SM120a part it was 11-15% *slower* than the
// cp.async ring at 2048^3 bf16 (0.42 vs 0.36 ms), with the mma removed the TMA ring alone costs
// 0.27 ms of the 0.36 ms budget, i.e. the strided 64-byte box fetch is close to the L2 bandwidth of
// the part and does not overlap the mma as well as the cp.async chunks do. The cp.async ring stays
// until a box shape (128-byte stage rows, 2-D rather than 3-D, multicast within a cluster) or a real
// producer warp changes that. Both tactic names are accepted by the host so callers can address this
// cell today; the mainloop body below is the cp.async one. Swapping it is a pure mainloop change: the
// epilogue and the launch contract stay as they are.
//
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh, common/tensor.h,
//           common/op_types.cuh, common/quant.cuh, common/device.cuh,
//           common/device_impl.cuh, gemm/gemm_params.cuh,
//           gemm/dense_gemm_kernels.cuh, gemm/fp8_gemm_kernels.cuh,
//           gemm/fp4_gemm_kernels.cuh.
//           Device-only: include from a CUDA device pass (a .cu translation unit),
//           after the toolkit headers and common/device_impl.cuh.
// =============================================================================

namespace kops::gemm {
namespace sm120a {

// -----------------------------------------------------------------------------
// PTX helpers (arch-generic text; every asm is inside the SM120a target anyway)
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// 16-byte chunk index XOR for 64-byte stage rows: swap on row / 2, so the eight rows of an ldmatrix sit
/// on eight distinct 16-byte bank groups.
__device__ __forceinline__ int swz(int row) { return (row >> 1) & 3; }

/// 16-byte cp.async (L2 only); pred == false zero-fills the 16 bytes without reading `src`.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// -----------------------------------------------------------------------------
// MMA atoms. One k-step consumes 32 bytes of every operand row, which is exactly the
// ldmatrix.x4 view the loader stages: m16n8k16 for the 16-bit payloads, m16n8k32 for
// e4m3. A = row-major [M, K] activations, B = row-major [N, K] weights, D = A . B^T.
// -----------------------------------------------------------------------------

struct AtomBf16 {
  using elem_t = bf16_t;
  using acc_t = float;
  static constexpr int kElemBytes = 2;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

struct AtomF16 {
  using elem_t = f16_t;
  using acc_t = float;
  static constexpr int kElemBytes = 2;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

struct AtomE4M3 {
  using elem_t = f8e4m3_t;
  using acc_t = float;
  static constexpr int kElemBytes = 1;
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

/// Operand element type -> atom (the Traits element types of the instantiation).
template <typename E>
struct AtomOf;template <>
struct AtomOf<bf16_t> {
  using type = AtomBf16;
};
template <>
struct AtomOf<f16_t> {
  using type = AtomF16;
};
template <>
struct AtomOf<f8e4m3_t> {
  using type = AtomE4M3;
};

/// Threads of one CTA of a tile: 2 warp rows x (tile_n / 64) warp columns. Named so the kernel
/// definition can write __launch_bounds__ with a single argument token (tools/check_roadmap.py parses
/// kernel names out of the header text).
template <typename Traits>
struct Threads {
  static constexpr int value = 32 * 2 * (Traits::tile_n / 32);
};

// -----------------------------------------------------------------------------
// Epilogue
// -----------------------------------------------------------------------------

/// Finishes one fp32 accumulator in the shared GemmParams order (see CONTRACTS.md):
///   *alpha -> *sA[row] -> *sB[col] -> *row_scale -> *col_scale -> +bias -> clamp
///   -> softcap -> act -> +beta * residual[row, col] (residual in the output dtype)
/// The operand scales of the FP8 recipes are passed in already resolved to the row /
/// column vector or folded into `alpha` (per tensor) by the kernel prologue.
template <typename Dt>
__device__ __forceinline__ float epi_finish(float v, const EpilogueParams& e, int64_t row, int64_t col) {
  const uint32_t ops = e.ops;
  if ((ops & kEpiRowScale) != 0u) v *= e.row_scale[row];
  if ((ops & kEpiColScale) != 0u) v *= e.col_scale[col];
  if ((ops & kEpiBias) != 0u) v += dev::load_as_float(e.bias, e.bias_dtype, col);
  if ((ops & kEpiClamp) != 0u) v = fminf(fmaxf(v, -e.clamp_limit), e.clamp_limit);
  if ((ops & kEpiSoftcap) != 0u) v = softcap(v, e.softcap);
  if ((ops & kEpiAct) != 0u) v = dev::act_runtime(e.act, v, e.glu);
  if ((ops & kEpiResidualAdd) != 0u) {
    v = fmaf(e.beta, dev::to_f(static_cast<const Dt*>(e.residual)[row * e.residual_stride + col]), v);
  }
  return v;
}

// -----------------------------------------------------------------------------
// Dense tile body
// -----------------------------------------------------------------------------

/// Thread -> output element map of every tile body is the m16n8k16 fragment layout:
///   element (mt, nt, i) sits at row warp_r0 + 16 mt + 8 (i >> 1) + lane/4 and
///   column warp_c0 + 8 nt + 2 (lane % 4) + (i & 1).

/// Two adjacent output columns of one lane as one store. The mma fragment map puts elements 2h and 2h + 1
/// on the same row, at columns 2*gc and 2*gc + 1, so the pair is contiguous; for a 16-bit output dtype it
/// becomes one 4-byte store as soon as the element index is even (the column always is, the row is not
/// when ldd is odd -- see the pair_ok test at the call site). An fp32 output keeps two 4-byte stores:
/// pairing it would need an 8-byte alignment that ldd and the batch stride do not guarantee.
template <typename Dt>
__device__ __forceinline__ void store_pair(Dt* p, float a, float b) {
  if constexpr (sizeof(Dt) == 4) {
    dev::from_f(p[0], a);
    dev::from_f(p[1], b);
  } else {
    Dt v[2];
    dev::from_f(v[0], a);
    dev::from_f(v[1], b);
    *reinterpret_cast<uint32_t*>(p) = *reinterpret_cast<const uint32_t*>(v);
  }
}

/// Finishes and stores one [MT, NT] accumulator tile. `extra_alpha` multiplies the epilogue alpha
/// (the NVFP4 global-scale product is folded in there).
template <typename Dt, int MT, int NT>
__device__ __forceinline__ void epilogue_store(const float (&acc)[MT][NT][4], const ScaledGemmArgs& args,
                                               int r0, int n0, int zb, int warp_r0, int warp_c0,
                                               float extra_alpha) {
  const EpilogueParams& e = args.epi;
  const int lane = threadIdx.x & 31;
  const int gl = lane >> 2;    // mma row inside the 8-row half
  const int gc = lane & 3;     // mma column pair
  const int64_t M = args.m, N = args.n;
  const DType dt = args.d_dtype;
  const int64_t d_es = dt == DType::kFloat32 ? 4 : 2;
  Dt* const dbase = reinterpret_cast<Dt*>(static_cast<unsigned char*>(args.d) +
                                           static_cast<int64_t>(zb) * args.batch_stride_d * d_es);
  // Operand scales: per tensor folds into alpha, per token (A) / per channel (B) stay vectors and are
  // applied before row_scale / col_scale. Block scales (kMX / kNVFP4 / kPerGroup / kPerBlock2D) are
  // consumed by the mainloop of the recipe that owns them, never here.
  float alpha = (e.alpha_ptr != nullptr ? e.alpha_ptr[0] : e.alpha) * extra_alpha;
  const float* xs_row = nullptr;
  const float* ws_col = nullptr;
  if (args.a.scale != nullptr) {
    const float* s = static_cast<const float*>(args.a.scale) + static_cast<int64_t>(zb) * args.a.scale_batch_stride;
    if (args.a.granularity == ScaleGranularity::kPerTensor) {
      alpha *= s[0];
    } else if (args.a.granularity == ScaleGranularity::kPerToken) {
      xs_row = s;   // [M]
    }
  }
  if (args.b.scale != nullptr) {
    const float* s = static_cast<const float*>(args.b.scale) + static_cast<int64_t>(zb) * args.b.scale_batch_stride;
    if (args.b.granularity == ScaleGranularity::kPerTensor) {
      alpha *= s[0];
    } else if (args.b.granularity == ScaleGranularity::kPerChannel) {
      ws_col = s;   // [N]
    }
  }
  if (xs_row == nullptr && ws_col == nullptr && e.ops == 0u) {
    // Plain product (no epilogue op, no operand scale vector): one multiply and one paired store per
    // two output elements. This is the hot path of a dense inference GEMM -- the per-element flag
    // checks of the general loop below cost ~4% of a 2048^3 bf16 launch, more than the mma they follow.
    // A 4-byte pair store needs an even element index of the global address. The column is always even,
    // so what decides it is the parity of the batch offset, of ldd * row and of row -- three loop
    // invariants, folded into one per-thread predicate that costs nothing in the loop.
    const int row_par = (r0 + warp_r0 + gl) & 1;
    const int bsd_par = static_cast<int>((static_cast<int64_t>(zb) * args.batch_stride_d) & 1);
    const int ldd_par = static_cast<int>(args.ldd & 1);
    const bool pair_ok = sizeof(Dt) == 4 || (bsd_par ^ (ldd_par & row_par)) == 0;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int64_t row = r0 + warp_r0 + mt * 16 + h * 8 + gl;
          const int64_t col = n0 + warp_c0 + nt * 8 + 2 * gc;
          if (row >= M || col >= N) continue;
          Dt* const p = dbase + row * args.ldd + col;
          const float v0 = acc[mt][nt][2 * h] * alpha;
          if (col + 1 >= N) {                              // last column of an odd N
            dev::from_f(p[0], v0);
          } else if (pair_ok) {
            store_pair<Dt>(p, v0, acc[mt][nt][2 * h + 1] * alpha);
          } else {                                         // odd ldd: two 2-byte stores of the pair
            dev::from_f(p[0], v0);
            dev::from_f(p[1], acc[mt][nt][2 * h + 1] * alpha);
          }
        }
      }
    }
    return;
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int lr = warp_r0 + mt * 16 + (i >> 1) * 8 + gl;
        const int lc = warp_c0 + nt * 8 + 2 * gc + (i & 1);
        const int64_t row = r0 + lr;
        const int64_t col = n0 + lc;
        if (row >= M || col >= N) continue;
        float v = acc[mt][nt][i] * alpha;
        if (xs_row != nullptr) v *= xs_row[row];
        if (ws_col != nullptr) v *= ws_col[col];
        v = epi_finish<Dt>(v, e, row, col);
        dev::from_f(dbase[row * args.ldd + col], v);
      }
    }
  }
}

/// One CTA computes one [TILE_M, TILE_N] tile. Grid: x = num_tiles_m * num_tiles_n (tile M fastest),
/// z = batch; the K loop runs num_tiles_k stages. Warp layout: 2 warp rows x (TILE_N / 64) warp
/// columns, i.e. 128 threads for a 128-wide tile and 256 for a 256/4 = 64-wide warp column.
template <typename Atom, typename Dt, int TILE_M, int TILE_N, int STAGES>
__device__ __forceinline__ void dense_tile(const GemmTmaParams& P) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  using E = typename Atom::elem_t;
  constexpr int EB = Atom::kElemBytes;
  constexpr int TILE_K = 64 / EB;              // K elements of one 64-byte stage row
  constexpr int KS = TILE_K * EB / 32;         // 32-byte k-steps per stage (always 2 here)
  constexpr int WM = 2;                        // warp rows
  constexpr int WN = TILE_N / 32;              // warp columns (8 warps on a 128-wide tile)
  constexpr int kThreads = 32 * WM * WN;
  constexpr int MT = TILE_M / WM / 16;         // m16 tiles per warp
  constexpr int NT = TILE_N / WN / 8;          // n8 tiles per warp
  constexpr int CH = 4;                        // 16-byte chunks per 64-byte stage row
  constexpr int RW = kThreads / CH;            // rows one loader pass covers
  constexpr int A_IT = (TILE_M + RW - 1) / RW;
  constexpr int B_IT = (TILE_N + RW - 1) / RW;
  constexpr uint32_t A_STAGE = TILE_M * 64;    // bytes of one A stage
  constexpr uint32_t B_STAGE = TILE_N * 64;
  static_assert(KS == 2, "a stage is two 32-byte k-steps");
  static_assert(STAGES >= 3, "the ring keeps at least two stages in flight");
  static_assert(TILE_M % (WM * 16) == 0 && TILE_N % (WN * 8) == 0 && NT % 2 == 0, "warp tile shape");
  static_assert(TILE_N % 64 == 0, "the warp column is 64 output channels");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* const sA = smem_raw;
  unsigned char* const sB = smem_raw + STAGES * A_STAGE;
  const uint32_t sA_u = smem_u32(sA);
  const uint32_t sB_u = smem_u32(sB);

  const ScaledGemmArgs& args = P.args;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t M = args.m, N = args.n, K = args.k;
  const int zb = static_cast<int>(blockIdx.z);
  const int tile = static_cast<int>(blockIdx.x);
  const int tn = P.num_tiles_n > 0 ? tile / P.num_tiles_m : 0;
  const int tm = P.num_tiles_n > 0 ? tile - tn * P.num_tiles_m : 0;
  const int r0 = tm * TILE_M;
  const int n0 = tn * TILE_N;
  const int64_t KB = K * EB;                   // bytes of one operand row
  const int64_t ldx_b = args.a.ld * EB;
  const int64_t ldw_b = args.b.ld * EB;
  const unsigned char* const xbase =
      static_cast<const unsigned char*>(args.a.data) + static_cast<int64_t>(zb) * args.a.batch_stride * EB;
  const unsigned char* const wbase =
      static_cast<const unsigned char*>(args.b.data) + static_cast<int64_t>(zb) * args.b.batch_stride * EB;

  // ---------------- loader ----------------
  const int ld_ch = tid % CH;
  const int ld_row = tid / CH;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * 64 + ((ld_ch ^ swz(ld_row)) << 4));
  const unsigned char* pw = wbase + static_cast<int64_t>(n0 + ld_row) * ldw_b + ld_ch * 16;
  const unsigned char* px = xbase + static_cast<int64_t>(r0 + ld_row) * ldx_b + ld_ch * 16;
  const int64_t stepW = static_cast<int64_t>(RW) * ldw_b;
  const int64_t stepX = static_cast<int64_t>(RW) * ldx_b;
  uint32_t wmask = 0, xmask = 0;
#pragma unroll
  for (int i = 0; i < B_IT; ++i) {
    const int rr = ld_row + i * RW;
    wmask |= static_cast<uint32_t>(rr < TILE_N && n0 + rr < N) << i;
  }
#pragma unroll
  for (int i = 0; i < A_IT; ++i) {
    const int rr = ld_row + i * RW;
    xmask |= static_cast<uint32_t>(rr < TILE_M && r0 + rr < M) << i;
  }
  int kld = ld_ch * 16;                        // byte offset of this thread's chunk in the current stage
  auto issue = [&](uint32_t dstA, uint32_t dstB) {
    const bool kok = kld < KB;
    const unsigned char* qw = pw;
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
      if (B_IT * RW == TILE_N || ld_row + i * RW < TILE_N) {
        cp_async16(dstB + ld_soff + static_cast<uint32_t>(i * RW * 64), qw, kok && ((wmask >> i) & 1u));
      }
      qw += stepW;
    }
    const unsigned char* qx = px;
#pragma unroll
    for (int i = 0; i < A_IT; ++i) {
      if (A_IT * RW == TILE_M || ld_row + i * RW < TILE_M) {
        cp_async16(dstA + ld_soff + static_cast<uint32_t>(i * RW * 64), qx, kok && ((xmask >> i) & 1u));
      }
      qx += stepX;
    }
    pw += 64;
    px += 64;
    kld += 64;
  };

  // ---------------- fragments ----------------
  const int warp_r0 = (warp / WN) * (TILE_M / WM);
  const int warp_c0 = (warp % WN) * (TILE_N / WN);
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
    for (int mt = 0; mt < MT; ++mt) ldmatrix_x4(af[mt], sA_u + stA + a_lane + mt * 16 * 64 + a_koff[kk]);
  };
  auto load_b = [&](int kk, uint32_t stB, uint32_t (&bf)[NT][2]) {
#pragma unroll
    for (int np = 0; np < NT / 2; ++np) {
      uint32_t r[4];
      ldmatrix_x4(r, sB_u + stB + b_lane + np * 16 * 64 + b_koff[kk]);
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

  // ---------------- mainloop ----------------
  const int nk = P.num_tiles_k;
#pragma unroll
  for (int g = 0; g < STAGES - 1; ++g) {
    if (g < nk) issue(sA_u + static_cast<uint32_t>(g) * A_STAGE, sB_u + static_cast<uint32_t>(g) * B_STAGE);
    cp_commit();
  }
  uint32_t af[2][MT][4];
  uint32_t bf[NT][2];
  int rs = 0;                 // stage being read
  int ws = STAGES - 1;        // stage being filled next
  cp_wait<STAGES - 2>();
  __syncthreads();
  load_a(0, 0, af[0]);
  for (int it = 0; it < nk; ++it) {
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      load_b(kk, static_cast<uint32_t>(rs) * B_STAGE, bf);
      if (kk == KS - 1) {
        cp_wait<STAGES - 2>();      // the next stage has landed
        __syncthreads();            // and every warp has its fragments of this one in registers
        rs = rs + 1 == STAGES ? 0 : rs + 1;
      }
      load_a((kk + 1) % KS, static_cast<uint32_t>(rs) * A_STAGE, af[(kk + 1) & 1]);
      if (kk == 0) {
        if (it + STAGES - 1 < nk) issue(sA_u + static_cast<uint32_t>(ws) * A_STAGE, sB_u + static_cast<uint32_t>(ws) * B_STAGE);
        cp_commit();
        ws = ws + 1 == STAGES ? 0 : ws + 1;
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) Atom::mma(acc[mt][nt], af[kk & 1][mt], bf[nt][0], bf[nt][1]);
      }
    }
  }
  cp_wait<0>();

  // ---------------- epilogue ----------------
  epilogue_store<Dt, MT, NT>(acc, args, r0, n0, zb, warp_r0, warp_c0, 1.0f);
#else
  (void)P;
  if (threadIdx.x == 0) __trap();   // the hosts select this cell only for SM120a/SM121a
#endif
}

// -----------------------------------------------------------------------------
// Block scales (FP4 recipes)
// -----------------------------------------------------------------------------

/// Offset of scale (row, blk) in the 128x4 swizzled layout (zero padded to 128 rows x 4 columns),
/// identical to `block_scale_swizzle_kernel` / torchao `to_blocked` /
/// gemm/fp4_gemm.cu::swizzled_offset: the array is [ceil(rows,128)][ceil(blocks_per_row,4)] 512-byte
/// tiles, and inside a tile byte (r, c) sits at (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4).
__device__ __forceinline__ int64_t sf_swizzled_offset(int64_t row, int64_t blk, int64_t blocks_per_row) {
  const int64_t tile = (row >> 7) * ((blocks_per_row + 3) >> 2) + (blk >> 2);
  return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (blk & 3);
}

/// One block-scale byte -> fp32 dequant multiplier (E4M3 for NVFP4, UE8M0 for MX).
__device__ __forceinline__ float sf_decode(uint32_t b, DType t) {
  if (t == DType::kFloat8E8M0) return (b & 0xFFu) == 0xFFu ? __uint_as_float(0x7FC00000u) : __uint_as_float((b & 0xFFu) << 23);
  return dev::e4m3_decode(b & 0xFFu);
}

/// scale(row, kb) of one operand in the declared layout. Row-major uses `scale_ld` (default: the
/// number of K blocks per row); the swizzled form derives its tile stride from that same block count.
__device__ __forceinline__ float sf_at(const uint8_t* p, DType dtype, ScaleLayout layout, int64_t scale_ld,
                                       int64_t row, int64_t kb, int64_t blocks_per_row) {
  const int64_t idx = layout == ScaleLayout::kSwizzled128x4 ? sf_swizzled_offset(row, kb, blocks_per_row)
                                                           : row * scale_ld + kb;
  return sf_decode(p[idx], dtype);
}

/// Eight packed e2m1 codes (one 32-bit word, element 2i in the low nibble of byte i) -> 16 bytes of
/// bf16. Exact: every e2m1 magnitude is representable in bf16, so the decode only changes the encoding.
__device__ __forceinline__ uint4 e2m1x8_to_bf16x8(uint32_t w) {
  bf16_t h[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = __float2bfloat16_rn(dev::e2m1_decode(w >> (4 * i)));
  return *reinterpret_cast<const uint4*>(h);
}

// -----------------------------------------------------------------------------
// FP4 tile body (software decode + block-scaled promotion)
// -----------------------------------------------------------------------------

/**
 * fp4_tile — the SM120a FP4 body of this pass. Effective recipe (documented, not the hardware one):
 *   1. the packed e2m1 operands are decoded to bf16 while they are staged into shared memory
 *      (4 packed bytes -> 16 decoded bytes per thread chunk, so a stage row is 64 bytes of bf16 =
 *      TILE_K 32 K elements, exactly the dense_tile layout);
 *   2. the mma is the plain m16n8k16 bf16 atom, accumulating one scale block into a temporary fragment;
 *   3. at every block boundary (16 K for NVFP4, 32 K for MXFP4) the block product sA[row, kb] *
 *      sB[col, kb] is applied to the temporary fragment and added to the fp32 accumulator, which is
 *      the order the block-scaled hardware uses (sum the block, then scale once).
 *   4. NVFP4 per-tensor global scales (ScaledOperand::global_scale) divide the result (1 / (gA * gB)),
 *      the repo convention of gemm/fp4_gemm.cu ("dequant multiplier is 1 / global_scale").
 * Not the SM120 block-scaled mma.sync (kind::mxf4nvf4): that needs the scale-factor register mapping
 * and is the follow-up. This body has the same numerics up to fp32 rounding, at bf16 mma rate and with
 * the decode in the load path.
 *
 * Loader: the copies are synchronous (LDG -> decode -> STS), so the ring is filled one stage ahead with
 * the global loads issued before the mma of the current stage. A 16 B chunk is either fully inside K or
 * zero-filled (K is a multiple of the block size, so it is a multiple of 8).
 */
template <typename Dt, int TILE_M, int TILE_N, int STAGES, int kBlockK>
__device__ __forceinline__ void fp4_tile(const GemmTmaParams& P) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  constexpr int WM = 2;                          // warp rows
  constexpr int WN = TILE_N / 32;                // warp columns
  constexpr int kThreads = 32 * WM * WN;
  constexpr int MT = TILE_M / WM / 16;
  constexpr int NT = TILE_N / WN / 8;
  constexpr int TILE_K = 32;                     // decoded K elements of one stage (16 packed bytes)
  constexpr int KS = TILE_K / 16;                // m16n8k16 atoms per stage
  constexpr int ATOMS_PER_BLOCK = kBlockK / 16;  // NVFP4: 1, MXFP4: 2
  constexpr int CH = 4;                          // 16-byte decoded chunks per stage row
  constexpr int RW = kThreads / CH;              // rows one loader pass covers
  constexpr int A_IT = (TILE_M + RW - 1) / RW;
  constexpr int B_IT = (TILE_N + RW - 1) / RW;
  constexpr uint32_t A_STAGE = TILE_M * 64;      // decoded bytes of one A stage
  constexpr uint32_t B_STAGE = TILE_N * 64;
  static_assert(kBlockK == 16 || kBlockK == 32, "FP4 scale blocks are 16 (NVFP4) or 32 (MX)");
  static_assert(KS % ATOMS_PER_BLOCK == 0, "a stage holds whole scale blocks");
  static_assert(TILE_K % kBlockK == 0, "a stage starts on a block boundary");
  static_assert(STAGES >= 3, "the ring keeps at least two stages in flight");
  static_assert(TILE_M % (WM * 16) == 0 && TILE_N % (WN * 8) == 0 && NT % 2 == 0, "warp tile shape");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  unsigned char* const sA = smem_raw;
  unsigned char* const sB = smem_raw + STAGES * A_STAGE;
  const uint32_t sA_u = smem_u32(sA);
  const uint32_t sB_u = smem_u32(sB);

  const ScaledGemmArgs& args = P.args;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t M = args.m, N = args.n, K = args.k;
  const int zb = static_cast<int>(blockIdx.z);
  const int tile = static_cast<int>(blockIdx.x);
  const int tn = P.num_tiles_n > 0 ? tile / P.num_tiles_m : 0;
  const int tm = P.num_tiles_n > 0 ? tile - tn * P.num_tiles_m : 0;
  const int r0 = tm * TILE_M;
  const int n0 = tn * TILE_N;
  // Packed operands: `ld` and the batch stride are byte pitches (gemm/fp4_gemm.cu convention).
  const int64_t ldx_b = args.a.ld;
  const int64_t ldw_b = args.b.ld;
  const unsigned char* const xbase =
      static_cast<const unsigned char*>(args.a.data) + static_cast<int64_t>(zb) * args.a.batch_stride;
  const unsigned char* const wbase =
      static_cast<const unsigned char*>(args.b.data) + static_cast<int64_t>(zb) * args.b.batch_stride;

  // ---------------- loader ----------------
  const int ld_ch = tid % CH;
  const int ld_row = tid / CH;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * 64 + ((ld_ch ^ swz(ld_row)) << 4));
  // This thread's chunk is 4 packed bytes (8 e2m1) -> 16 decoded bytes.
  const unsigned char* pw = wbase + static_cast<int64_t>(n0 + ld_row) * ldw_b + ld_ch * 4;
  const unsigned char* px = xbase + static_cast<int64_t>(r0 + ld_row) * ldx_b + ld_ch * 4;
  const int64_t stepW = static_cast<int64_t>(RW) * ldw_b;
  const int64_t stepX = static_cast<int64_t>(RW) * ldx_b;
  uint32_t wmask = 0, xmask = 0;
#pragma unroll
  for (int i = 0; i < B_IT; ++i) {
    const int rr = ld_row + i * RW;
    wmask |= static_cast<uint32_t>(rr < TILE_N && n0 + rr < N) << i;
  }
#pragma unroll
  for (int i = 0; i < A_IT; ++i) {
    const int rr = ld_row + i * RW;
    xmask |= static_cast<uint32_t>(rr < TILE_M && r0 + rr < M) << i;
  }
  int koff = ld_ch * 8;                        // element offset of this thread's chunk in the stage
  auto fetch = [&](uint32_t (&ra)[A_IT], uint32_t (&rb)[B_IT]) {
    const bool kok = koff < K;
    const unsigned char* qw = pw;
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
      const bool ok = kok && ((wmask >> i) & 1u) != 0u;
      rb[i] = ok ? *reinterpret_cast<const uint32_t*>(qw) : 0u;
      qw += stepW;
    }
    const unsigned char* qx = px;
#pragma unroll
    for (int i = 0; i < A_IT; ++i) {
      const bool ok = kok && ((xmask >> i) & 1u) != 0u;
      ra[i] = ok ? *reinterpret_cast<const uint32_t*>(qx) : 0u;
      qx += stepX;
    }
    pw += 16;                                  // one stage = 16 packed bytes per row
    px += 16;
    koff += TILE_K;
  };
  auto store_stage = [&](int stage, const uint32_t (&ra)[A_IT], const uint32_t (&rb)[B_IT]) {
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
      if (B_IT * RW == TILE_N || ld_row + i * RW < TILE_N) {
        *reinterpret_cast<uint4*>(sB + static_cast<size_t>(stage) * B_STAGE + ld_soff + i * RW * 64) =
            e2m1x8_to_bf16x8(rb[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < A_IT; ++i) {
      if (A_IT * RW == TILE_M || ld_row + i * RW < TILE_M) {
        *reinterpret_cast<uint4*>(sA + static_cast<size_t>(stage) * A_STAGE + ld_soff + i * RW * 64) =
            e2m1x8_to_bf16x8(ra[i]);
      }
    }
  };

  // ---------------- fragments ----------------
  const int warp_r0 = (warp / WN) * (TILE_M / WM);
  const int warp_c0 = (warp % WN) * (TILE_N / WN);
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
    for (int mt = 0; mt < MT; ++mt) ldmatrix_x4(af[mt], sA_u + stA + a_lane + mt * 16 * 64 + a_koff[kk]);
  };
  auto load_b = [&](int kk, uint32_t stB, uint32_t (&bf)[NT][2]) {
#pragma unroll
    for (int np = 0; np < NT / 2; ++np) {
      uint32_t r[4];
      ldmatrix_x4(r, sB_u + stB + b_lane + np * 16 * 64 + b_koff[kk]);
      bf[2 * np][0] = r[0];
      bf[2 * np][1] = r[1];
      bf[2 * np + 1][0] = r[2];
      bf[2 * np + 1][1] = r[3];
    }
  };

  // ---------------- block scales ----------------
  const int64_t blocks_per_row = K / kBlockK;
  const int64_t sa_ld = args.a.scale_ld != 0 ? args.a.scale_ld : blocks_per_row;
  const int64_t sb_ld = args.b.scale_ld != 0 ? args.b.scale_ld : blocks_per_row;
  const uint8_t* const sfa = static_cast<const uint8_t*>(args.a.scale);
  const uint8_t* const sfb = static_cast<const uint8_t*>(args.b.scale);
  const DType sdt = args.a.scale_dtype;

  // ---------------- mainloop ----------------
  float acc[MT][NT][4];
  float tmp[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc[i][j][q] = 0.0f;
        tmp[i][j][q] = 0.0f;
      }
    }
  }

  const int nk = P.num_tiles_k;
  uint32_t ra[A_IT], rb[B_IT];
#pragma unroll
  for (int g = 0; g < STAGES - 1; ++g) {
    if (g < nk) {
      fetch(ra, rb);
      store_stage(g, ra, rb);
    }
  }
  __syncthreads();

  // Scale product of one block for this thread's rows and columns.
  const int gl = lane >> 2;
  const int gc = lane & 3;
  auto promote = [&](int64_t kb) {
    float sa[MT][2], sb[NT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = r0 + warp_r0 + mt * 16 + gl + h * 8;
        const int64_t rc = row < M ? row : (M > 0 ? M - 1 : 0);
        sa[mt][h] = sf_at(sfa, sdt, args.a.scale_layout, sa_ld, rc, kb, blocks_per_row);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int64_t col = n0 + warp_c0 + nt * 8 + 2 * gc + c;
        const int64_t cc = col < N ? col : (N > 0 ? N - 1 : 0);
        sb[nt][c] = sf_at(sfb, sdt, args.b.scale_layout, sb_ld, cc, kb, blocks_per_row);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[mt][nt][i] = fmaf(tmp[mt][nt][i], sa[mt][i >> 1] * sb[nt][i & 1], acc[mt][nt][i]);
          tmp[mt][nt][i] = 0.0f;
        }
      }
    }
  };

  int rs = 0;
  for (int it = 0; it < nk; ++it) {
    const int ws = (it + STAGES - 1) % STAGES;
    const bool have_next = it + STAGES - 1 < nk;
    if (have_next) fetch(ra, rb);              // global loads of the next stage, overlapped with the mma
    const int64_t kbase = static_cast<int64_t>(it) * TILE_K;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      uint32_t af[MT][4];
      uint32_t bf[NT][2];
      load_b(kk, static_cast<uint32_t>(rs) * B_STAGE, bf);
      load_a(kk, static_cast<uint32_t>(rs) * A_STAGE, af);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) AtomBf16::mma(tmp[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
      }
      if ((kk + 1) % ATOMS_PER_BLOCK == 0) promote((kbase + kk * 16) / kBlockK);
    }
    __syncthreads();                           // every warp is done with stage rs
    if (have_next) store_stage(ws, ra, rb);
    __syncthreads();                           // stage ws is visible before it is read
    rs = rs + 1 == STAGES ? 0 : rs + 1;
  }

  // ---------------- epilogue ----------------
  // NVFP4 per-tensor global scales are dequant multipliers, so the result is divided by gA * gB
  // (gemm/fp4_gemm.cu convention). A caller that prefers to fold them into epi.alpha leaves both null.
  float inv_g = 1.0f;
  if (args.a.global_scale != nullptr) inv_g /= *args.a.global_scale;
  if (args.b.global_scale != nullptr) inv_g /= *args.b.global_scale;
  epilogue_store<Dt, MT, NT>(acc, args, r0, n0, zb, warp_r0, warp_c0, inv_g);
#else
  (void)P;
  if (threadIdx.x == 0) __trap();   // the hosts select this cell only for SM120a/SM121a
#endif
}

}  // namespace sm120a

// =============================================================================
// Kernel definitions (the declarations live in the family *_kernels.cuh headers)
// =============================================================================

/// gemm_sm120_kernel — bf16 / fp16 dense GEMM, dense schedule, batch in blockIdx.z.
template <typename Traits>
__global__ void __launch_bounds__(sm120a::Threads<Traits>::value)
    gemm_sm120_kernel(const __grid_constant__ GemmTmaParams params) {
  static_assert(Traits::mainloop == MainloopSchedule::kSm120WarpSpecialized ||
                    Traits::mainloop == MainloopSchedule::kSm120BlockScaledWS,
                "gemm_sm120_kernel serves the SM120a mainloop tactics only");
  using Atom = typename sm120a::AtomOf<typename Traits::a_type>::type;
  sm120a::dense_tile<Atom, typename Traits::d_type, Traits::tile_m, Traits::tile_n, Traits::stages>(params);
}

/// fp8_gemm_sm120_kernel — e4m3 x e4m3, kFP8Tensorwise / kFP8Rowwise. The scales are epilogue
/// inputs (alpha / per-token A / per-channel B); the mainloop is a plain e4m3 mma.sync chain.
/// kFP8Blockwise / kFP8BlockwiseUE8M0 / kMXFP8 need per-128-K promotion or the block-scaled
/// hardware path and are rejected by the host launcher.
template <typename Traits, GemmRecipe kRecipe>
__global__ void __launch_bounds__(sm120a::Threads<Traits>::value)
    fp8_gemm_sm120_kernel(const __grid_constant__ GemmTmaParams params) {
  static_assert(kRecipe == GemmRecipe::kFP8Tensorwise || kRecipe == GemmRecipe::kFP8Rowwise,
                "fp8_gemm_sm120_kernel covers the tensorwise and rowwise FP8 recipes");
  using Atom = typename sm120a::AtomOf<typename Traits::a_type>::type;
  sm120a::dense_tile<Atom, typename Traits::d_type, Traits::tile_m, Traits::tile_n, Traits::stages>(params);
}

/// fp4_gemm_sm120_kernel — kNVFP4 (block 16, E4M3 scales) and kMXFP4 (block 32, UE8M0 scales) on the
/// software-decode body of this cell: packed e2m1 -> bf16 in the load path, block scales applied at
/// every block boundary (see sm120a::fp4_tile). kMXFP4xMXFP8 / kMXFP6xMXFP8 need a second operand
/// loader (an e4m3 / e2m3 A next to an e2m1 B) and are not instantiated.
template <typename Traits, GemmRecipe kRecipe>
__global__ void __launch_bounds__(sm120a::Threads<Traits>::value)
    fp4_gemm_sm120_kernel(const __grid_constant__ GemmTmaParams params) {
  static_assert(kRecipe == GemmRecipe::kNVFP4 || kRecipe == GemmRecipe::kMXFP4,
                "fp4_gemm_sm120_kernel covers NVFP4 and MXFP4 (W4A4)");
  constexpr int kBlockK = kRecipe == GemmRecipe::kNVFP4 ? 16 : 32;
  sm120a::fp4_tile<typename Traits::d_type, Traits::tile_m, Traits::tile_n, Traits::stages, kBlockK>(params);
}

}  // namespace kops::gemm
