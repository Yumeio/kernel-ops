// =============================================================================
// kernel-ops | csrc/gemm/mixed_input_gemm.cu
// B5 rows 7: weight-only 4-bit GEMV on CUDA cores.
//   w4a16_gemv_kernel   AWQ [K, N/8] and GPTQ [K/8, N] int32 layouts read directly
//   lut_gemv_kernel     4-bit codebook weights (NF4 / FP4 LUT / custom) with block absmax
// Host entries: mixed_input_gemm (kW4A16Int, M <= 16) and lut_gemv (M <= 16).
//
// Contract: csrc/gemm/mixed_input_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// Numerics.  fp32 accumulation everywhere.  A weight is (q - z) * s with q, z small integers, so
// q - z is exact in fp32; the scale is applied once per group to the group's fp32 partial sum
// (acc += s * sum_k a_k (q_k - z)).  Each output element is a function of (row of A, column of W)
// only: the order of additions does not depend on M_TILE, the grid or the other rows, so one row
// alone and the same row inside a batch are bit-identical (tests/unit/test_gemm.cc).
//
// Conventions.
//   * AWQ column n = 8j + c lives at nibble awq_position(c) of word j (same table as
//     quantization/weight_repack.cu), zeros share the layout; GPTQ k = 8kb + i lives at nibble i
//     of word kb, zeros are [K/g, N/8] in natural column order.
//   * qzeros store the zero point itself (GPTQ v2 / compressed-tensors).  Legacy GPTQ v1
//     checkpoints store z - 1 and must be fixed up at load.  No zeros -> symmetric uint4b8, z = 8.
//   * lut codes: low nibble is the even element (repo-wide int4 order); bitsandbytes stores the
//     even element in the high nibble, so a loader that imports NF4 swaps nibbles once.
//   * Double-quantized absmax is linear: absmax = q * absmax_scale[block / 256] + absmax_offset.
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

constexpr int kKS = 8;       // GPTQ: K slices per block (threadIdx.y)
constexpr int kLanes = 32;   // GPTQ: columns per block (threadIdx.x)
// AWQ packs 8 columns per word, so N/8 words is a short axis (512 for N = 4096): a block takes only
// 8 words (32 B of every weight row, one full DRAM sector) and 32 K slices, giving 4x the blocks.
constexpr int kAwqWords = 8;
constexpr int kAwqKS = 32;

/// Position of column c (0..7) inside an AWQ word: {0,4,1,5,2,6,3,7}.
__host__ __device__ constexpr int awq_pos(int c) { return ((c & 1) << 2) | (c >> 1); }

__device__ __forceinline__ uint32_t nib(uint32_t w, int pos) { return (w >> (4 * pos)) & 0xFu; }

/// 2^23 + q as a float: q - z is then `qm - zm` with zm = 2^23 + z, exact for q, z < 2^23.
__device__ __forceinline__ float qmagic(uint32_t q) { return __uint_as_float(0x4B000000u | q); }

constexpr float kSymZeroMagic = 8388608.0f + 8.0f;   // 2^23 + 8

// -----------------------------------------------------------------------------
// AWQ: thread = one packed word (8 adjacent columns), threadIdx.y = K slice (whole groups)
// -----------------------------------------------------------------------------

template <typename scalar_t, int GROUP_SIZE, bool kHasZeros, int M_TILE>
__device__ __forceinline__ void w4a16_awq_body(scalar_t* __restrict__ out,
                                               const scalar_t* __restrict__ a,
                                               const uint32_t* __restrict__ qweight,
                                               const scalar_t* __restrict__ scales,
                                               const uint32_t* __restrict__ qzeros,
                                               const scalar_t* __restrict__ bias, const int n,
                                               const int k) {
  __shared__ float red[kAwqKS][8][kAwqWords];
  const int nw = n >> 3;
  const int j = blockIdx.x * kAwqWords + threadIdx.x;
  const int ky = threadIdx.y;
  const int gsz = GROUP_SIZE > 0 ? GROUP_SIZE : k;
  const int ngroups = k / gsz;

  float acc[8][M_TILE];
  #pragma unroll
  for (int c = 0; c < 8; ++c) {
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[c][m] = 0.0f;
  }

  if (j < nw) {
    for (int g = ky; g < ngroups; g += kAwqKS) {
      float sf[8];
      dev::unpack8<scalar_t>(*reinterpret_cast<const uint4*>(scales + static_cast<int64_t>(g) * n + 8 * j), sf);
      float zm[8];
      if constexpr (kHasZeros) {
        const uint32_t zw = qzeros[static_cast<int64_t>(g) * nw + j];
        #pragma unroll
        for (int c = 0; c < 8; ++c) zm[c] = qmagic(nib(zw, awq_pos(c)));
      } else {
        #pragma unroll
        for (int c = 0; c < 8; ++c) zm[c] = kSymZeroMagic;
      }
      float accg[8][M_TILE];
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) accg[c][m] = 0.0f;
      }
      const int k0 = g * gsz;
      for (int kk = k0; kk < k0 + gsz; kk += 8) {
        uint32_t w[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = dev::ld_stream4(qweight + static_cast<int64_t>(kk + i) * nw + j);
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          float af[8];
          dev::unpack8<scalar_t>(*reinterpret_cast<const uint4*>(a + static_cast<int64_t>(m) * k + kk), af);
          #pragma unroll
          for (int c = 0; c < 8; ++c) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              accg[c][m] = fmaf(af[i], qmagic(nib(w[i], awq_pos(c))) - zm[c], accg[c][m]);
            }
          }
        }
      }
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) acc[c][m] = fmaf(sf[c], accg[c][m], acc[c][m]);
      }
    }
  }

  // Fixed-order reduction over the K slices: thread (x, ky < 8) finishes column c = ky of word x.
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int c = 0; c < 8; ++c) red[ky][c][threadIdx.x] = acc[c][m];
    __syncthreads();
    if (ky < 8 && j < nw) {
      float sum = 0.0f;
      #pragma unroll
      for (int t = 0; t < kAwqKS; ++t) sum += red[t][ky][threadIdx.x];
      const int col = 8 * j + ky;
      if (bias != nullptr) sum += dev::to_f(bias[col]);
      dev::from_f(out[static_cast<int64_t>(m) * n + col], sum);
    }
    __syncthreads();
  }
}

// -----------------------------------------------------------------------------
// GPTQ: thread = one column, threadIdx.y = K slice (whole groups); act-order slow path
// -----------------------------------------------------------------------------

template <typename scalar_t, int GROUP_SIZE, bool kHasZeros, int M_TILE>
__device__ __forceinline__ void w4a16_gptq_body(scalar_t* __restrict__ out,
                                                const scalar_t* __restrict__ a,
                                                const uint32_t* __restrict__ qweight,
                                                const scalar_t* __restrict__ scales,
                                                const uint32_t* __restrict__ qzeros,
                                                const int32_t* __restrict__ g_idx,
                                                const scalar_t* __restrict__ bias, const int n,
                                                const int k) {
  __shared__ float red[kKS][M_TILE][kLanes];
  const int col = blockIdx.x * kLanes + threadIdx.x;
  const int ky = threadIdx.y;
  const int gsz = GROUP_SIZE > 0 ? GROUP_SIZE : k;
  const int ngroups = k / gsz;
  [[maybe_unused]] const int nzw = n >> 3;

  float acc[M_TILE];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

  if (col < n && g_idx == nullptr) {
    for (int g = ky; g < ngroups; g += kKS) {
      const float s = dev::to_f(scales[static_cast<int64_t>(g) * n + col]);
      float zm = kSymZeroMagic;
      if constexpr (kHasZeros) {
        zm = qmagic((qzeros[static_cast<int64_t>(g) * nzw + (col >> 3)] >> (4 * (col & 7))) & 0xFu);
      }
      float accg[M_TILE];
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) accg[m] = 0.0f;
      const int kb0 = g * gsz / 8;
      const int kb1 = kb0 + gsz / 8;
      for (int kb = kb0; kb < kb1; kb += 4) {
        uint32_t w[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) w[u] = dev::ld_stream4(qweight + static_cast<int64_t>(kb + u) * n + col);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          #pragma unroll
          for (int m = 0; m < M_TILE; ++m) {
            float af[8];
            dev::unpack8<scalar_t>(
                *reinterpret_cast<const uint4*>(a + static_cast<int64_t>(m) * k + 8 * (kb + u)), af);
            #pragma unroll
            for (int i = 0; i < 8; ++i) accg[m] = fmaf(af[i], qmagic(nib(w[u], i)) - zm, accg[m]);
          }
        }
      }
      #pragma unroll
      for (int m = 0; m < M_TILE; ++m) acc[m] = fmaf(s, accg[m], acc[m]);
    }
  } else if (col < n) {
    // Act-order: the group of every k comes from g_idx, so scale and zero are fetched per weight.
    for (int kb = ky; kb < k / 8; kb += kKS) {
      const uint32_t w = qweight[static_cast<int64_t>(kb) * n + col];
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int kk = 8 * kb + i;
        const int g = g_idx[kk];
        const float s = dev::to_f(scales[static_cast<int64_t>(g) * n + col]);
        float zm = kSymZeroMagic;
        if constexpr (kHasZeros) {
          zm = qmagic((qzeros[static_cast<int64_t>(g) * nzw + (col >> 3)] >> (4 * (col & 7))) & 0xFu);
        }
        const float wf = (qmagic(nib(w, i)) - zm) * s;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          acc[m] = fmaf(dev::to_f(a[static_cast<int64_t>(m) * k + kk]), wf, acc[m]);
        }
      }
    }
  }

  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) red[ky][m][threadIdx.x] = acc[m];
  __syncthreads();
  if (col < n) {
    for (int m = ky; m < M_TILE; m += kKS) {
      float sum = 0.0f;
      #pragma unroll
      for (int t = 0; t < kKS; ++t) sum += red[t][m][threadIdx.x];
      if (bias != nullptr) sum += dev::to_f(bias[col]);
      dev::from_f(out[static_cast<int64_t>(m) * n + col], sum);
    }
  }
}

}  // namespace

template <typename scalar_t, WeightPacking kPacking, int GROUP_SIZE, bool kHasZeros, int M_TILE>
__global__ void w4a16_gemv_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ a,
                                  const uint32_t* __restrict__ qweight,
                                  const scalar_t* __restrict__ scales,
                                  const uint32_t* __restrict__ qzeros,
                                  const int32_t* __restrict__ g_idx,
                                  const scalar_t* __restrict__ bias, const int m, const int n,
                                  const int k) {
  (void)m;
  static_assert(kPacking == WeightPacking::kAWQ || kPacking == WeightPacking::kGPTQ,
                "w4a16_gemv_kernel reads AWQ or GPTQ layouts");
  if constexpr (kPacking == WeightPacking::kAWQ) {
    (void)g_idx;
    w4a16_awq_body<scalar_t, GROUP_SIZE, kHasZeros, M_TILE>(out, a, qweight, scales, qzeros, bias, n, k);
  } else {
    w4a16_gptq_body<scalar_t, GROUP_SIZE, kHasZeros, M_TILE>(out, a, qweight, scales, qzeros, g_idx, bias, n, k);
  }
}

// =============================================================================
// lut_gemv_kernel: warp per row, lane = 32 k (16 B of codes) per iteration
// =============================================================================

template <typename scalar_t, int BLOCK_SIZE, bool kDoubleQuant, int M_TILE>
__global__ void lut_gemv_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ a,
                                const uint8_t* __restrict__ qweight,
                                const float* __restrict__ codebook, const void* __restrict__ absmax,
                                const float* __restrict__ absmax_scale,
                                const float* __restrict__ absmax_offset, const int m, const int n,
                                const int k) {
  (void)m;
  constexpr int kWarps = 8;
  // A bank-replicated table (cb[e][lane]) was tried and is slower: refilling 2 KB per block costs more
  // than the lookup conflicts it removes (NF4 4096x4096: 0.085 ms -> 0.134 ms).
  __shared__ float cb[16];
  if (threadIdx.x < 16) cb[threadIdx.x] = codebook[threadIdx.x];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= n) return;

  const float off = kDoubleQuant ? absmax_offset[0] : 0.0f;
  float acc[M_TILE];
  #pragma unroll
  for (int mm = 0; mm < M_TILE; ++mm) acc[mm] = 0.0f;

  const uint8_t* wrow = qweight + static_cast<int64_t>(row) * (k / 2);
  const int64_t blocks_per_row = k / BLOCK_SIZE;
  for (int kc = lane * 32; kc < k; kc += 32 * 32) {
    const uint4 wv = dev::ld_stream16(wrow + kc / 2);
    const int64_t blk = static_cast<int64_t>(row) * blocks_per_row + kc / BLOCK_SIZE;
    float amax;
    if constexpr (kDoubleQuant) {
      amax = fmaf(static_cast<float>(static_cast<const uint8_t*>(absmax)[blk]), absmax_scale[blk >> 8], off);
    } else {
      amax = static_cast<const float*>(absmax)[blk];
    }
    float wf[32];
    const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) wf[wi * 8 + i] = cb[(words[wi] >> (4 * i)) & 0xFu];
    }
    float part[M_TILE];
    #pragma unroll
    for (int mm = 0; mm < M_TILE; ++mm) {
      part[mm] = 0.0f;
      #pragma unroll
      for (int v = 0; v < 4; ++v) {
        float af[8];
        dev::unpack8<scalar_t>(*reinterpret_cast<const uint4*>(a + static_cast<int64_t>(mm) * k + kc + 8 * v), af);
        #pragma unroll
        for (int i = 0; i < 8; ++i) part[mm] = fmaf(af[i], wf[v * 8 + i], part[mm]);
      }
      acc[mm] = fmaf(amax, part[mm], acc[mm]);
    }
  }
  #pragma unroll
  for (int mm = 0; mm < M_TILE; ++mm) {
    float v = acc[mm];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if (lane == 0) dev::from_f(out[static_cast<int64_t>(mm) * n + row], v);
  }
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

const char* byte_ptr(const Tensor& t) { return static_cast<const char*>(t.data) + t.byte_offset; }

template <typename scalar_t, WeightPacking kPacking, int G, bool kZ, int MT>
Status launch_w4a16(scalar_t* out, const scalar_t* a, const uint32_t* qw, const scalar_t* sc,
                    const uint32_t* qz, const int32_t* gi, const scalar_t* bias, int m, int n, int k,
                    cudaStream_t stream) {
  constexpr bool kAwq = kPacking == WeightPacking::kAWQ;
  const dim3 block(kAwq ? kAwqWords : kLanes, kAwq ? kAwqKS : kKS);
  const int units = kAwq ? n / 8 : n;
  const dim3 grid((units + block.x - 1) / block.x);
  w4a16_gemv_kernel<scalar_t, kPacking, G, kZ, MT><<<grid, block, 0, stream>>>(out, a, qw, sc, qz, gi, bias, m, n, k);
  return launch_error("w4a16_gemv_kernel");
}

template <typename scalar_t, WeightPacking kPacking, int G, bool kZ>
Status dispatch_w4a16_m(int mt, scalar_t* out, const scalar_t* a, const uint32_t* qw,
                        const scalar_t* sc, const uint32_t* qz, const int32_t* gi,
                        const scalar_t* bias, int m, int n, int k, cudaStream_t s) {
  switch (mt) {
    case 1: return launch_w4a16<scalar_t, kPacking, G, kZ, 1>(out, a, qw, sc, qz, gi, bias, m, n, k, s);
    case 2: return launch_w4a16<scalar_t, kPacking, G, kZ, 2>(out, a, qw, sc, qz, gi, bias, m, n, k, s);
    case 4: return launch_w4a16<scalar_t, kPacking, G, kZ, 4>(out, a, qw, sc, qz, gi, bias, m, n, k, s);
    default: return launch_w4a16<scalar_t, kPacking, G, kZ, 8>(out, a, qw, sc, qz, gi, bias, m, n, k, s);
  }
}

template <typename scalar_t, WeightPacking kPacking, int G>
Status dispatch_w4a16_z(bool zeros, int mt, scalar_t* out, const scalar_t* a, const uint32_t* qw,
                        const scalar_t* sc, const uint32_t* qz, const int32_t* gi,
                        const scalar_t* bias, int m, int n, int k, cudaStream_t s) {
  return zeros ? dispatch_w4a16_m<scalar_t, kPacking, G, true>(mt, out, a, qw, sc, qz, gi, bias, m, n, k, s)
               : dispatch_w4a16_m<scalar_t, kPacking, G, false>(mt, out, a, qw, sc, qz, gi, bias, m, n, k, s);
}

template <typename scalar_t, WeightPacking kPacking>
Status dispatch_w4a16_g(int group, bool zeros, int mt, scalar_t* out, const scalar_t* a,
                        const uint32_t* qw, const scalar_t* sc, const uint32_t* qz,
                        const int32_t* gi, const scalar_t* bias, int m, int n, int k,
                        cudaStream_t s) {
  switch (group) {
    case 32: return dispatch_w4a16_z<scalar_t, kPacking, 32>(zeros, mt, out, a, qw, sc, qz, gi, bias, m, n, k, s);
    case 64: return dispatch_w4a16_z<scalar_t, kPacking, 64>(zeros, mt, out, a, qw, sc, qz, gi, bias, m, n, k, s);
    case 128: return dispatch_w4a16_z<scalar_t, kPacking, 128>(zeros, mt, out, a, qw, sc, qz, gi, bias, m, n, k, s);
    default: return dispatch_w4a16_z<scalar_t, kPacking, 0>(zeros, mt, out, a, qw, sc, qz, gi, bias, m, n, k, s);
  }
}

template <typename scalar_t>
Status run_w4a16(const ScaledGemmArgs& args, const LaunchContext& ctx) {
  const int n = static_cast<int>(args.n);
  const int k = static_cast<int>(args.k);
  const int group = args.b.group_k;
  const bool zeros = args.b.zero_points != nullptr;
  const auto* a = static_cast<const scalar_t*>(args.a.data);
  const auto* qw = static_cast<const uint32_t*>(args.b.data);
  const auto* sc = static_cast<const scalar_t*>(args.b.scale);
  const auto* qz = static_cast<const uint32_t*>(args.b.zero_points);
  const scalar_t* bias = (args.epi.ops & kEpiBias) != 0u ? static_cast<const scalar_t*>(args.epi.bias) : nullptr;
  auto* out = static_cast<scalar_t*>(args.d);
  Status st = Status::kSuccess;
  // Rows go out in tiles of 8/4/2/1; a tile never reads or writes past M.
  for (int64_t m0 = 0; m0 < args.m && st == Status::kSuccess;) {
    const int64_t rem = args.m - m0;
    const int mt = rem >= 8 ? 8 : (rem >= 4 ? 4 : (rem >= 2 ? 2 : 1));
    if (args.b.packing == WeightPacking::kAWQ) {
      st = dispatch_w4a16_g<scalar_t, WeightPacking::kAWQ>(group, zeros, mt, out + m0 * n, a + m0 * k, qw, sc, qz,
                                                          nullptr, bias, mt, n, k, stream_of(ctx));
    } else {
      st = dispatch_w4a16_g<scalar_t, WeightPacking::kGPTQ>(group, zeros, mt, out + m0 * n, a + m0 * k, qw, sc, qz,
                                                           args.b.g_idx, bias, mt, n, k, stream_of(ctx));
    }
    m0 += mt;
  }
  return st;
}

Status check_w4a16(const ScaledGemmArgs& args) {
  const DType adt = args.a.dtype;
  if (adt != DType::kBFloat16 && adt != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "w4a16_gemv: activations must be bf16 or fp16");
  }
  if (args.d_dtype != adt) {
    return detail::set_error(Status::kUnsupportedDtype, "w4a16_gemv: output dtype must equal the activation dtype");
  }
  if (args.b.packing != WeightPacking::kAWQ && args.b.packing != WeightPacking::kGPTQ) {
    return detail::set_error(Status::kUnsupportedLayout, "w4a16_gemv: weight packing must be kAWQ or kGPTQ");
  }
  if (args.batch != 1) return detail::set_error(Status::kNotImplemented, "w4a16_gemv: batch > 1");
  if (args.m < 1 || args.n < 1 || args.k < 1) return detail::set_error(Status::kInvalidArgument, "w4a16_gemv: empty problem");
  if (args.m > 16) {
    return detail::set_error(Status::kNotImplemented, "w4a16_gemv: M > 16 needs the tensor-core path (B7)");
  }
  const int g = args.b.group_k;
  if (g != 0 && g != 32 && g != 64 && g != 128) {
    return detail::set_error(Status::kUnsupportedShape, "w4a16_gemv: group_k must be 0 (per channel), 32, 64 or 128");
  }
  if (args.k % 32 != 0 || (g != 0 && args.k % g != 0)) {
    return detail::set_error(Status::kUnsupportedShape, "w4a16_gemv: K must be a multiple of 32 and of the group size");
  }
  if (args.n % 8 != 0) return detail::set_error(Status::kUnsupportedShape, "w4a16_gemv: N must be a multiple of 8");
  if (args.a.ld != args.k || args.ldd != args.n) {
    return detail::set_error(Status::kUnsupportedLayout, "w4a16_gemv: activations and output must be contiguous");
  }
  if (args.a.data == nullptr || args.b.data == nullptr || args.b.scale == nullptr || args.d == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "w4a16_gemv: null operand");
  }
  if (args.b.scale_dtype != adt) {
    return detail::set_error(Status::kUnsupportedDtype, "w4a16_gemv: scales must have the activation dtype");
  }
  if (args.b.packing == WeightPacking::kGPTQ && args.b.g_idx != nullptr && g == 0) {
    return detail::set_error(Status::kInvalidArgument, "w4a16_gemv: act-order needs a grouped scheme");
  }
  if (args.b.packing == WeightPacking::kAWQ && args.b.g_idx != nullptr) {
    return detail::set_error(Status::kUnsupportedLayout, "w4a16_gemv: AWQ has no act-order");
  }
  if ((args.epi.ops & ~static_cast<uint32_t>(kEpiBias)) != 0u) {
    return detail::set_error(Status::kNotImplemented, "w4a16_gemv: only the Bias epilogue is implemented");
  }
  if ((args.epi.ops & kEpiBias) != 0u && (args.epi.bias == nullptr || args.epi.bias_dtype != adt)) {
    return detail::set_error(Status::kInvalidArgument, "w4a16_gemv: bias must be [N] with the activation dtype");
  }
  if (!aligned(args.a.data, 16) || !aligned(args.b.scale, 16) || !aligned(args.b.data, 4)) {
    return detail::set_error(Status::kMisaligned, "w4a16_gemv: operands must be 16 B (scales, activations) / 4 B (weights) aligned");
  }
  return Status::kSuccess;
}

}  // namespace

Status mixed_input_gemm(const ScaledGemmArgs& args, const GemmTactic* tactic, GemmBackend backend,
                        const Workspace& ws, const LaunchContext& ctx) {
  (void)tactic;
  (void)ws;
  Status st = backend_ok(ctx, backend, "mixed_input_gemm");
  if (st != Status::kSuccess) return st;
  if (args.recipe != GemmRecipe::kW4A16Int) {
    return detail::set_error(Status::kNotImplemented, "mixed_input_gemm: recipe %d has no native body yet",
                             static_cast<int>(args.recipe));
  }
  st = check_w4a16(args);
  if (st != Status::kSuccess) return st;
  return args.a.dtype == DType::kBFloat16 ? run_w4a16<__nv_bfloat16>(args, ctx) : run_w4a16<__half>(args, ctx);
}

namespace {

template <typename scalar_t, int BS, bool kDQ, int MT>
Status launch_lut(void* out, const void* a, const uint8_t* qw, const float* cb, const void* amax,
                  const float* ascale, const float* aoff, int m, int n, int k, cudaStream_t s) {
  constexpr int kWarps = 8;
  const dim3 grid((n + kWarps - 1) / kWarps);
  lut_gemv_kernel<scalar_t, BS, kDQ, MT><<<grid, kWarps * 32, 0, s>>>(
      static_cast<scalar_t*>(out), static_cast<const scalar_t*>(a), qw, cb, amax, ascale, aoff, m, n, k);
  return launch_error("lut_gemv_kernel");
}

template <typename scalar_t, int BS, bool kDQ>
Status dispatch_lut_m(int mt, void* out, const void* a, const uint8_t* qw, const float* cb,
                      const void* amax, const float* ascale, const float* aoff, int m, int n, int k,
                      cudaStream_t s) {
  switch (mt) {
    case 1: return launch_lut<scalar_t, BS, kDQ, 1>(out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    case 2: return launch_lut<scalar_t, BS, kDQ, 2>(out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    case 4: return launch_lut<scalar_t, BS, kDQ, 4>(out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    default: return launch_lut<scalar_t, BS, kDQ, 8>(out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
  }
}

template <typename scalar_t, bool kDQ>
Status dispatch_lut_bs(int bs, int mt, void* out, const void* a, const uint8_t* qw, const float* cb,
                       const void* amax, const float* ascale, const float* aoff, int m, int n, int k,
                       cudaStream_t s) {
  switch (bs) {
    case 32: return dispatch_lut_m<scalar_t, 32, kDQ>(mt, out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    case 64: return dispatch_lut_m<scalar_t, 64, kDQ>(mt, out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    case 128: return dispatch_lut_m<scalar_t, 128, kDQ>(mt, out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
    default: return dispatch_lut_m<scalar_t, 256, kDQ>(mt, out, a, qw, cb, amax, ascale, aoff, m, n, k, s);
  }
}

}  // namespace

Status lut_gemv(const Tensor& out, const Tensor& a, const Tensor& qweight, const Tensor& codebook,
                const Tensor& absmax, const Tensor& absmax_scale, const Tensor& absmax_offset,
                int32_t block_size, const LaunchContext& ctx) {
  Status st = backend_ok(ctx, GemmBackend::kNative, "lut_gemv");
  if (st != Status::kSuccess) return st;
  if ((st = check_tensor(out, DType::kInvalid, 2, 1, "lut_gemv.out")) != Status::kSuccess) return st;
  if ((st = check_tensor(a, DType::kInvalid, 2, 1, "lut_gemv.a")) != Status::kSuccess) return st;
  if ((st = check_tensor(qweight, DType::kUInt8, 2, 1, "lut_gemv.qweight")) != Status::kSuccess) return st;
  if ((st = check_tensor(codebook, DType::kFloat32, 1, 1, "lut_gemv.codebook")) != Status::kSuccess) return st;
  const DType adt = a.dtype;
  if (adt != DType::kBFloat16 && adt != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "lut_gemv: activations must be bf16 or fp16");
  }
  if (out.dtype != adt) return detail::set_error(Status::kUnsupportedDtype, "lut_gemv: output dtype must equal the activation dtype");
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = out.shape[1];
  if (out.shape[0] != m || qweight.shape[0] != n || qweight.shape[1] * 2 != k || codebook.shape[0] != 16) {
    return detail::set_error(Status::kInvalidArgument, "lut_gemv: shapes disagree (out [M,N], a [M,K], qweight [N,K/2], codebook [16])");
  }
  if (m < 1 || m > 16) return detail::set_error(Status::kNotImplemented, "lut_gemv: M must be 1..16");
  if (block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256) {
    return detail::set_error(Status::kUnsupportedShape, "lut_gemv: block_size must be 32, 64, 128 or 256");
  }
  if (k % 32 != 0) return detail::set_error(Status::kUnsupportedShape, "lut_gemv: K must be a multiple of 32");
  if (k % block_size != 0) return detail::set_error(Status::kUnsupportedShape, "lut_gemv: K must be a multiple of block_size");
  if (a.stride[1] != 1 || a.stride[0] != k || out.stride[1] != 1 || out.stride[0] != n ||
      qweight.stride[1] != 1 || qweight.stride[0] != k / 2) {
    return detail::set_error(Status::kUnsupportedLayout, "lut_gemv: operands must be contiguous");
  }
  const bool dq = absmax.dtype == DType::kUInt8;
  if (!dq && absmax.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "lut_gemv: absmax must be fp32 or uint8 (double-quantized)");
  }
  if (absmax.ndim != 1 || absmax.shape[0] != n * (k / block_size)) {
    return detail::set_error(Status::kInvalidArgument, "lut_gemv: absmax must be [N*K/block_size]");
  }
  if (dq && (absmax_scale.data == nullptr || absmax_offset.data == nullptr || absmax_scale.dtype != DType::kFloat32 ||
             absmax_offset.dtype != DType::kFloat32 || absmax_scale.shape[0] < (absmax.shape[0] + 255) / 256)) {
    return detail::set_error(Status::kInvalidArgument, "lut_gemv: double quantization needs fp32 absmax_scale [ceil(blocks/256)] and absmax_offset [1]");
  }
  if (!aligned(byte_ptr(a), 16) || !aligned(byte_ptr(qweight), 16)) {
    return detail::set_error(Status::kMisaligned, "lut_gemv: activations and codes must be 16 B aligned");
  }
  const auto* qw = reinterpret_cast<const uint8_t*>(byte_ptr(qweight));
  const auto* cb = reinterpret_cast<const float*>(byte_ptr(codebook));
  const auto* sc = dq ? reinterpret_cast<const float*>(byte_ptr(absmax_scale)) : nullptr;
  const auto* of = dq ? reinterpret_cast<const float*>(byte_ptr(absmax_offset)) : nullptr;
  const auto* ab = byte_ptr(absmax);
  const size_t es = sizeof(__nv_bfloat16);
  for (int64_t m0 = 0; m0 < m && st == Status::kSuccess;) {
    const int64_t rem = m - m0;
    const int mt = rem >= 8 ? 8 : (rem >= 4 ? 4 : (rem >= 2 ? 2 : 1));
    void* o = const_cast<char*>(byte_ptr(out)) + m0 * n * es;
    const void* ap = byte_ptr(a) + m0 * k * es;
    const int bs = block_size;
    if (adt == DType::kBFloat16) {
      st = dq ? dispatch_lut_bs<__nv_bfloat16, true>(bs, mt, o, ap, qw, cb, ab, sc, of, mt, static_cast<int>(n), static_cast<int>(k), stream_of(ctx))
              : dispatch_lut_bs<__nv_bfloat16, false>(bs, mt, o, ap, qw, cb, ab, sc, of, mt, static_cast<int>(n), static_cast<int>(k), stream_of(ctx));
    } else {
      st = dq ? dispatch_lut_bs<__half, true>(bs, mt, o, ap, qw, cb, ab, sc, of, mt, static_cast<int>(n), static_cast<int>(k), stream_of(ctx))
              : dispatch_lut_bs<__half, false>(bs, mt, o, ap, qw, cb, ab, sc, of, mt, static_cast<int>(n), static_cast<int>(k), stream_of(ctx));
    }
    m0 += mt;
  }
  return st;
}

}  // namespace kops::gemm
