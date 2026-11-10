// =============================================================================
// kernel-ops | csrc/gemm/int8_gemm.cu
// B5 row 7: INT8 GEMV on CUDA cores.
//   int8_gemv_kernel   s8 weights [N, K]; s8 activations (dp4a, s32 accumulate) or bf16/fp16
//                      activations (W8A16, fp32 accumulate), M <= 16
// Host entry: int8_gemm (kINT8Tensorwise / kINT8Rowwise / kW4A8IntInt8). M > 16, batch > 1 (or an mma tactic / cache entry) runs
// int8_gemm_mma_sync_kernel = gemm_mma_sync_kernel<AtomS8> (B7 row 5, gemm_mma.cu): s8 mma.sync.m16n8k32,
// s32 accumulators, azp correction in the integers before the s32 -> fp32 conversion.
//
// B7 row 5, second kernel: w4a8_int_gemm_kernel (QServe-style W4A8, kW4A8IntInt8), its own body here:
//   weights u4 [N, K/2] (kUInt4x2, low nibble = even k), two-level dequant to s8 in shared memory
//     GROUP_SIZE 128 (per group): w8[n, k] = q * s2[k/128, n] - sz[k/128, n]   (s2, sz uint8 [K/128, N];
//                                 sz = z * s2, "subtraction after multiplication"; w8 must lie in [-128, 127],
//                                 QServe's protective range — the kernel does not saturate)
//     GROUP_SIZE 0   (per channel): w8[n, k] = q - z[n]  (z uint8 [N], 0..15)
//   activations s8 [M, K] (per-token scale in epi.row_scale), mma.sync.m16n8k32 s8, s32 accumulate, then
//   D = epilogue(float(acc)) with the level-1 channel scale s1 in epi.col_scale.
//   Operands: b.second_level = s2, b.zero_points = sz (group) or z (channel), b.group_k = 128 or 0.
//
// Contract: csrc/gemm/int8_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// W8A8 is exact integer arithmetic up to the s32 -> fp32 conversion, so it is bit-exact against
// an integer oracle.  Asymmetric activations subtract azp[m] * colsum_b[n] from the s32
// accumulator (D = (acc - azp * colsum) * sA * sB + bias, the INT8 header's formula); the scales
// live in the epilogue (alpha, row_scale[m], col_scale[n]).  Reduction over the lanes of a warp is
// an integer xor-butterfly, so the result does not depend on M_TILE, the grid or the other rows.
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

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

/// Exact float of the signed byte b (low 8 bits): 2^23 + (b + 128) - (2^23 + 128).
__device__ __forceinline__ float s8_to_f(uint32_t b) {
  return __uint_as_float(0x4B000000u | ((b & 0xFFu) ^ 0x80u)) - 8388736.0f;
}

__device__ __forceinline__ void store_result(const ScaledGemmArgs& a, int64_t m, int64_t n, float acc) {
  const int64_t idx = m * a.ldd + n;
  switch (a.d_dtype) {
    case DType::kFloat32:
      static_cast<float*>(a.d)[idx] = dev::gemm_epilogue<float>(acc, a.epi, m, n);
      break;
    case DType::kFloat16:
      static_cast<__half*>(a.d)[idx] = __float2half_rn(dev::gemm_epilogue<__half>(acc, a.epi, m, n));
      break;
    default:
      static_cast<__nv_bfloat16*>(a.d)[idx] = __float2bfloat16_rn(dev::gemm_epilogue<__nv_bfloat16>(acc, a.epi, m, n));
      break;
  }
}

}  // namespace

template <typename a_t, bool kAsymA, int M_TILE, int ROWS_PER_WARP, int WARPS>
__global__ void int8_gemv_kernel(const __grid_constant__ ScaledGemmArgs args) {
  constexpr bool kS8 = std::is_same<a_t, int8_t>::value;
  constexpr int R = ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;
  if (n0 >= args.n) return;
  const int k = static_cast<int>(args.k);
  const int64_t m_total = args.m;

  const a_t* arow[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    arow[m] = static_cast<const a_t*>(args.a.data) + imin(m, m_total - 1) * args.a.ld;
  }
  const uint8_t* brow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    brow[r] = static_cast<const uint8_t*>(args.b.data) + imin(n0 + r, args.n - 1) * args.b.ld;
  }

  using acc_t = typename std::conditional<kS8, int32_t, float>::type;
  acc_t acc[M_TILE][R];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[m][r] = acc_t(0);
  }

  #pragma unroll 2
  for (int kc = lane * 16; kc < k; kc += 512) {
    uint4 wv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = dev::ld_stream16(brow[r] + kc);
    if constexpr (kS8) {
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) {
        const uint4 av = *reinterpret_cast<const uint4*>(arow[m] + kc);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          int32_t t = acc[m][r];
          t = __dp4a(static_cast<int>(av.x), static_cast<int>(wv[r].x), t);
          t = __dp4a(static_cast<int>(av.y), static_cast<int>(wv[r].y), t);
          t = __dp4a(static_cast<int>(av.z), static_cast<int>(wv[r].z), t);
          t = __dp4a(static_cast<int>(av.w), static_cast<int>(wv[r].w), t);
          acc[m][r] = t;
        }
      }
    } else {
      float wf[R][16];
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t w4[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) wf[r][4 * q + i] = s8_to_f(w4[q] >> (8 * i));
        }
      }
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) {
        float lo[8], hi[8];
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[m] + kc), lo);
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[m] + kc + 8), hi);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[m][r] = fmaf(lo[i], wf[r][i], acc[m][r]);
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[m][r] = fmaf(hi[i], wf[r][8 + i], acc[m][r]);
        }
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      acc_t v = acc[m][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == m * R + r && m < m_total && n0 + r < args.n) {
        float f;
        if constexpr (kS8) {
          if constexpr (kAsymA) v -= args.epi.azp[m] * args.epi.colsum_b[n0 + r];
          f = static_cast<float>(v);
        } else {
          f = v;
        }
        store_result(args, m, n0 + r, f);
      }
    }
  }
}

// =============================================================================
// B7 row 5: w4a8_int_gemm_kernel
// -----------------------------------------------------------------------------
// Semantics (recipe kW4A8IntInt8, entry int8_gemm; QServe W4A8, sglang qserve_w4a8_per_{chn,group}_gemm):
//   a   s8 [M, K], a.ld in elements (= bytes), K-major.                             a.dtype   kInt8
//   b   u4 [N, K] packed [N, K/2] bytes, b.ld in BYTES (>= K/2); byte j of row n holds
//       q[n, 2j] in the low nibble and q[n, 2j+1] in the high nibble, q in 0..15.   b.dtype   kUInt4x2
//   per group (b.group_k == 128): g = k / 128,
//       s2 = b.second_level  uint8 [K/128, N] row-major (element g * N + n)        level-2 scale
//       sz = b.zero_points   uint8 [K/128, N] row-major                            = z * s2 (QServe "scaled zeros")
//       w8[n, k] = q[n, k] * s2[g, n] - sz[g, n]                                   (multiply, then subtract)
//   per channel (b.group_k == 0):
//       z  = b.zero_points   uint8 [N]
//       w8[n, k] = q[n, k] - z[n]
//   Precondition (QServe's protective range): every w8 lies in [-128, 127].  The kernel does not
//   saturate; an out-of-range value keeps its low byte (two's complement wrap), so the result is then
//   that of the wrapped weight.  The per-channel form is in range for every z <= 128.
//   acc[m, n] = sum_k a[m, k] * w8[n, k] in s32, exact (|acc| <= K * 2^14, no overflow for K < 2^17).
//   D[m, n] = epilogue(float(acc[m, n])) with dev::gemm_epilogue:
//       * alpha (or *alpha_ptr) * row_scale[m] (per-token activation scale) * col_scale[n] (level-1
//       per-channel weight scale s1, fp32) + bias[n] -> clamp -> softcap -> act -> + beta * residual,
//       rounded once to bf16 / fp16 (or kept fp32 with kEpiOutFp32).  ScaledOperand::scale is not used
//       (non-null is kInvalidArgument).
//   Shapes: K % 128 == 0 per group, K % 64 == 0 per channel; any M, N; batch 1; 16 B aligned bases and
//   row pitches of a and b.  float(acc) is exact for |acc| <= 2^24, so D does not depend on the tile:
//   each acc element is an exact integer sum, there is no split-K, and the run is deterministic.
// Body: 64 x 64 tile, 4 warps (2 x 2, 32 x 32 each), 64 K per stage, 2-stage cp.async of A (s8) and of
// the packed nibbles; each stage is expanded to s8 in shared memory (sW, same swizzle as sA) and fed
// to ldmatrix + mma.sync.m16n8k32 s8.
// =============================================================================
namespace {

__device__ __forceinline__ uint32_t w4_smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void w4_cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
#endif
}
__device__ __forceinline__ void w4_ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#endif
}
__device__ __forceinline__ void w4_mma_s8(int32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
/// 64-byte rows: 16-byte chunk c of row r lives at chunk c ^ ((r >> 1) & 3).
__device__ __forceinline__ int w4_swz(int row) { return (row >> 1) & 3; }

}  // namespace

/// Tile of w4a8_int_gemm_kernel: BM x BN outputs, 64 K per stage, 2 x 2 warps.
template <int BM_, int BN_>
struct W4A8Tile {
  static constexpr int bm = BM_, bn = BN_, bk = 64, threads = 128;
};

template <typename Traits, int GROUP_SIZE>
__global__ void __launch_bounds__(128) w4a8_int_gemm_kernel(const __grid_constant__ ScaledGemmArgs args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int BM = Traits::bm, BN = Traits::bn;
  constexpr int MT = BM / 32, NT = BN / 16;   // per warp: (BM / 2) rows, (BN / 2) columns
  __shared__ __align__(128) uint8_t sA[2][BM * 64];
  __shared__ __align__(128) uint8_t sP[2][BN * 32];   // packed u4: 32 bytes per row and stage
  __shared__ __align__(128) uint8_t sW[BN * 64];      // expanded s8
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = static_cast<int>(args.m), N = static_cast<int>(args.n), K = static_cast<int>(args.k);
  const int tiles_n = (N + BN - 1) / BN;
  const int row0 = (blockIdx.x / tiles_n) * BM;
  const int col0 = (blockIdx.x % tiles_n) * BN;
  const int8_t* gA = static_cast<const int8_t*>(args.a.data);
  const uint8_t* gB = static_cast<const uint8_t*>(args.b.data);
  const uint8_t* s2 = static_cast<const uint8_t*>(args.b.second_level);
  const uint8_t* zz = static_cast<const uint8_t*>(args.b.zero_points);
  const int nk = (K + 63) / 64;

  auto load = [&](int st, int kt) {
    const int k0 = kt * 64;
    for (int c = tid; c < BM * 4; c += 128) {   // A: 4 chunks of 16 B per row
      const int r = c >> 2, ch = c & 3;
      const bool ok = row0 + r < M && k0 + ch * 16 < K;
      w4_cp_async16(w4_smem_u32(&sA[st][r * 64 + ((ch ^ w4_swz(r)) << 4)]), ok ? gA + static_cast<int64_t>(row0 + r) * args.a.ld + k0 + ch * 16 : gA, ok);
    }
    for (int c = tid; c < BN * 2; c += 128) {   // B: 2 chunks of 16 B (32 nibbles) per row
      const int r = c >> 1, ch = c & 1;
      const bool ok = col0 + r < N && k0 + ch * 32 < K;
      w4_cp_async16(w4_smem_u32(&sP[st][r * 32 + ch * 16]), ok ? gB + static_cast<int64_t>(col0 + r) * args.b.ld + k0 / 2 + ch * 16 : gB, ok);
    }
    asm volatile("cp.async.commit_group;");
  };
  // packed stage -> s8: thread takes 16 output bytes (one swizzled chunk) of one row
  auto expand = [&](int st, int kt) {
    const int k0 = kt * 64;
    for (int c = tid; c < BN * 4; c += 128) {
      const int r = c >> 2, ch = c & 3;
      const int n = col0 + r;
      int scale = 1, zero = 0;
      if (n < N) {
        if constexpr (GROUP_SIZE > 0) {
          const int64_t gi = static_cast<int64_t>((k0 + ch * 16) / GROUP_SIZE) * N + n;
          scale = s2[gi];
          zero = zz[gi];
        } else {
          zero = zz[n];
        }
      }
      const uint2 pk = *reinterpret_cast<const uint2*>(&sP[st][r * 32 + ch * 8]);
      uint32_t out[4];
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t word = w < 2 ? (w == 0 ? pk.x : pk.x >> 16) : (w == 2 ? pk.y : pk.y >> 16);
        uint32_t o = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int q = static_cast<int>((word >> (4 * j)) & 0xFu);
          const int v = q * scale - zero;
          o |= (static_cast<uint32_t>(v) & 0xFFu) << (8 * j);
        }
        out[w] = o;
      }
      *reinterpret_cast<uint4*>(&sW[r * 64 + ((ch ^ w4_swz(r)) << 4)]) = make_uint4(out[0], out[1], out[2], out[3]);
    }
  };

  int32_t acc[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      #pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0;
  const int wr0 = (warp >> 1) * (BM / 2);
  const int wc0 = (warp & 1) * (BN / 2);
  if (nk > 0) load(0, 0);
  for (int it = 0; it < nk; ++it) {
    const int st = it & 1;
    if (it + 1 < nk) {
      load(st ^ 1, it + 1);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();   // stage it landed; sW is free (previous mma done)
    expand(st, it);
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[MT][4];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r = wr0 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kk * 2 + (lane >> 4);
        w4_ldmatrix_x4(af[mt], w4_smem_u32(&sA[st][r * 64 + ((ch ^ w4_swz(r)) << 4)]));
      }
      #pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        const int r = wc0 + np * 16 + (lane & 7) + (lane >> 4) * 8;
        const int ch = kk * 2 + ((lane >> 3) & 1);
        uint32_t b[4];
        w4_ldmatrix_x4(b, w4_smem_u32(&sW[r * 64 + ((ch ^ w4_swz(r)) << 4)]));
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          w4_mma_s8(acc[mt][2 * np], af[mt], b[0], b[1]);
          w4_mma_s8(acc[mt][2 * np + 1], af[mt], b[2], b[3]);
        }
      }
    }
    __syncthreads();   // sA[st] and sW are reused
  }
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = row0 + wr0 + mt * 16 + g + (i >> 1) * 8;
        const int n = col0 + wc0 + nt * 8 + 2 * t + (i & 1);
        if (m >= M || n >= N) continue;
        const int64_t idx = static_cast<int64_t>(m) * args.ldd + n;
        const float v = static_cast<float>(acc[mt][nt][i]);
        switch (args.d_dtype) {
          case DType::kFloat32: static_cast<float*>(args.d)[idx] = dev::gemm_epilogue<float>(v, args.epi, m, n); break;
          case DType::kFloat16: static_cast<__half*>(args.d)[idx] = __float2half_rn(dev::gemm_epilogue<__half>(v, args.epi, m, n)); break;
          default: static_cast<__nv_bfloat16*>(args.d)[idx] = __float2bfloat16_rn(dev::gemm_epilogue<__nv_bfloat16>(v, args.epi, m, n)); break;
        }
      }
    }
  }
#else
  (void)args;
  if (threadIdx.x == 0) __trap();
#endif
}

namespace {

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

template <typename a_t, bool kAsym, int MT, int R>
Status launch_int8_tile(const ScaledGemmArgs& args, cudaStream_t s) {
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((args.n + kWarps * R - 1) / (kWarps * R)));
  int8_gemv_kernel<a_t, kAsym, MT, R, kWarps><<<grid, kWarps * 32, 0, s>>>(args);
  return launch_error("int8_gemv_kernel");
}

template <typename a_t, bool kAsym>
Status launch_int8(const ScaledGemmArgs& args, cudaStream_t s) {
  if (args.m <= 1) return launch_int8_tile<a_t, kAsym, 1, 4>(args, s);
  if (args.m <= 2) return launch_int8_tile<a_t, kAsym, 2, 4>(args, s);
  if (args.m <= 4) return launch_int8_tile<a_t, kAsym, 4, 2>(args, s);
  if (args.m <= 8) return launch_int8_tile<a_t, kAsym, 8, 2>(args, s);
  return launch_int8_tile<a_t, kAsym, 16, 1>(args, s);
}

constexpr uint32_t kInt8EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                     kEpiResidualAdd | kEpiClamp | kEpiOutFp32 | kEpiAzpCorrection;
constexpr uint32_t kW4A8EpiAllowed = kEpiRowScale | kEpiColScale | kEpiBias | kEpiSoftcap | kEpiAct |
                                     kEpiResidualAdd | kEpiClamp | kEpiOutFp32;

template <int G>
Status launch_w4a8(const ScaledGemmArgs& a, cudaStream_t s) {
  using Tr = W4A8Tile<64, 64>;
  const int64_t tiles = ((a.m + 63) / 64) * ((a.n + 63) / 64);
  if (tiles > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "w4a8: too many tiles");
  w4a8_int_gemm_kernel<Tr, G><<<static_cast<unsigned>(tiles), Tr::threads, 0, s>>>(a);
  return launch_error("w4a8_int_gemm_kernel");
}

Status w4a8_gemm(const ScaledGemmArgs& args, const LaunchContext& ctx) {
  if (args.a.dtype != DType::kInt8) return detail::set_error(Status::kUnsupportedDtype, "w4a8: activations must be int8");
  if (args.b.dtype != DType::kUInt4x2) return detail::set_error(Status::kUnsupportedDtype, "w4a8: weights must be packed uint4 (kUInt4x2)");
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "w4a8: output must be bf16, fp16 or fp32");
  }
  if (args.batch != 1) return detail::set_error(Status::kNotImplemented, "w4a8: batch > 1");
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) return detail::set_error(Status::kUnsupportedLayout, "w4a8: K-major operands only");
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "w4a8: empty problem");
  if (args.m > 0x7FFFFFFF || args.n > 0x7FFFFFFF || args.k > 0x7FFFFFFF) return detail::set_error(Status::kUnsupportedShape, "w4a8: extents beyond int32");
  const int gk = args.b.group_k;
  if (gk != 0 && gk != 128) return detail::set_error(Status::kUnsupportedShape, "w4a8: group_k must be 128 (per group) or 0 (per channel)");
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr || args.b.zero_points == nullptr || (gk != 0 && args.b.second_level == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "w4a8: null operand (zero_points, and second_level for per-group scales, are required)");
  }
  if (args.b.zero_dtype != DType::kInvalid && args.b.zero_dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "w4a8: zero points (and level-2 scales) are uint8");
  }
  if (args.a.scale != nullptr || args.b.scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "w4a8: the per-token / level-1 scales go in epi.row_scale / epi.col_scale, not ScaledOperand::scale");
  }
  if (args.workspace != nullptr) return detail::set_error(Status::kInvalidArgument, "w4a8: pass workspace through the Workspace argument");
  if (args.k % 64 != 0 || (gk != 0 && args.k % gk != 0)) return detail::set_error(Status::kUnsupportedShape, "w4a8: K must be a multiple of 64 (and of the group)");
  if (args.a.ld < args.k || args.b.ld < args.k / 2 || args.ldd < args.n) return detail::set_error(Status::kInvalidArgument, "w4a8: leading dimensions smaller than the extents");
  if (!aligned(args.a.data, 16) || args.a.ld % 16 != 0 || !aligned(args.b.data, 16) || args.b.ld % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "w4a8: operand base and row pitch must be 16 B aligned");
  }
  if ((args.epi.ops & ~kW4A8EpiAllowed) != 0u) return detail::set_error(Status::kNotImplemented, "w4a8: epilogue ops 0x%x are not implemented", args.epi.ops & ~kW4A8EpiAllowed);
  if (((args.epi.ops & kEpiRowScale) != 0u && args.epi.row_scale == nullptr) || ((args.epi.ops & kEpiColScale) != 0u && args.epi.col_scale == nullptr) ||
      ((args.epi.ops & kEpiBias) != 0u && args.epi.bias == nullptr) || ((args.epi.ops & kEpiResidualAdd) != 0u && args.epi.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "w4a8: epilogue operand missing");
  }
  if (((args.epi.ops & kEpiOutFp32) != 0u) != (args.d_dtype == DType::kFloat32)) {
    return detail::set_error(Status::kInvalidArgument, "w4a8: kEpiOutFp32 and an fp32 output go together");
  }
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "w4a8: mma.sync s8 needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  return gk == 0 ? launch_w4a8<0>(args, s) : launch_w4a8<128>(args, s);
}

}  // namespace

Status int8_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                 const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "int8_gemm: CUDA backend only");
  }
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "int8_gemm: only the native provider exists");
  }
  if (args.recipe == GemmRecipe::kW4A8IntInt8) return w4a8_gemm(args, ctx);
  if (args.recipe != GemmRecipe::kINT8Tensorwise && args.recipe != GemmRecipe::kINT8Rowwise) {
    return detail::set_error(Status::kNotImplemented, "int8_gemm: recipe %d has no native body yet", static_cast<int>(args.recipe));
  }
  if (args.b.dtype != DType::kInt8) return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: weights must be int8");
  const DType adt = args.a.dtype;
  if (adt != DType::kInt8 && adt != DType::kBFloat16 && adt != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: activations must be int8, bf16 or fp16");
  }
  if (args.d_dtype != DType::kBFloat16 && args.d_dtype != DType::kFloat16 && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: output must be bf16, fp16 or fp32");
  }
  if (args.a.major != MajorOrder::kKMajor || args.b.major != MajorOrder::kKMajor) {
    return detail::set_error(Status::kUnsupportedLayout, "int8_gemm: K-major operands only (weight_transpose first)");
  }
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "int8_gemm: empty problem");
  // one K rule (and one status) for both bodies: the GEMV reads 16-element chunks, mma.sync 16-byte cp.async chunks
  if (args.k % 16 != 0) return detail::set_error(Status::kUnsupportedShape, "int8_gemm: K must be a multiple of 16");
  if (args.m > 16 || args.batch != 1 || mma_selects(args, tactic, ctx)) {
    if (adt != DType::kInt8) {
      return detail::set_error(Status::kUnsupportedDtype, "int8_gemm: the tensor-core path takes int8 activations (W8A16 runs M <= 16 only)");
    }
    MmaGemmParams p{};
    p.args = args;
    return launch_gemm_mma_sync(p, tactic, ws, ctx);
  }
  if (args.a.data == nullptr || args.b.data == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: null operand");
  }
  const int a_bytes = adt == DType::kInt8 ? 1 : 2;
  if (args.a.ld < args.k || args.b.ld < args.k || args.ldd < args.n) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: leading dimensions smaller than the extents");
  }
  if (!aligned(args.a.data, 16) || (args.a.ld * a_bytes) % 16 != 0 || !aligned(args.b.data, 16) || args.b.ld % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "int8_gemm: operand base and row pitch must be 16 B aligned");
  }
  if ((args.epi.ops & ~kInt8EpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "int8_gemm: epilogue ops 0x%x are not implemented", args.epi.ops & ~kInt8EpiAllowed);
  }
  const bool asym = (args.epi.ops & kEpiAzpCorrection) != 0u;
  if (asym && (adt != DType::kInt8 || args.epi.azp == nullptr || args.epi.colsum_b == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: azp correction needs int8 activations, azp[M] and colsum_b[N]");
  }
  if (((args.epi.ops & kEpiRowScale) != 0u && args.epi.row_scale == nullptr) ||
      ((args.epi.ops & kEpiColScale) != 0u && args.epi.col_scale == nullptr) ||
      ((args.epi.ops & kEpiBias) != 0u && args.epi.bias == nullptr) ||
      ((args.epi.ops & kEpiResidualAdd) != 0u && args.epi.residual == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: epilogue operand missing (residual is read in the output dtype)");
  }
  if ((args.epi.ops & kEpiOutFp32) != 0u && args.d_dtype != DType::kFloat32) {
    return detail::set_error(Status::kInvalidArgument, "int8_gemm: kEpiOutFp32 needs an fp32 output");
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (adt) {
    case DType::kInt8: return asym ? launch_int8<int8_t, true>(args, s) : launch_int8<int8_t, false>(args, s);
    case DType::kBFloat16: return launch_int8<__nv_bfloat16, false>(args, s);
    default: return launch_int8<__half, false>(args, s);
  }
}

}  // namespace kops::gemm
