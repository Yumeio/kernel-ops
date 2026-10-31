// =============================================================================
// kernel-ops | csrc/gemm/gguf_gemm.cu
// B5 row 8: GGUF decode GEMV on CUDA cores (llama.cpp mmvq semantics).
//   gguf_quantize_q8_1_kernel   activations -> block_q8_1 (36 B / 32 elements)
//   gguf_mmvq_kernel            dense: ggml blocks of [N, K] x q8_1 activations, M <= 8
//   gguf_moe_mmvq_kernel        expert-indexed: weights [E, N, K] blocks, rows chosen by topk_ids
// Host entries: gguf_gemm (dense), gguf_quantize_q8_1, gguf_moe_gemm.
// Types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.  The IQ (codebook) types
// return kNotImplemented: their grid tables live inside quantization/weight_repack.cu and the
// closed file budget (ROADMAP 1.2) leaves no header to share them through.
//
// Contract: csrc/gemm/gguf_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// One unit = 32 weights = one block_q8_1 of activations.  Every format is decoded to
//   x_w = alpha * q' + beta           (q' a small integer, one (alpha, beta) per 16 weights)
// so the dot product with the q8_1 activations is two integer dp4a chains per half unit,
//   sum_k x_w a_k = d_a * (alpha * sum(q' qa) + beta * sum(qa)),
// exact up to the final fp32 scaling, which is what llama.cpp's vec_dot_*_q8_1 computes.  The
// per-format unpacking follows ggml-common.h / dequantize_row_* (validated bit-for-bit by B4's
// gguf_dequantize, which the test uses as the weight oracle).
//
// Activation quantization follows ggml-cuda quantize_q8_1: d = amax / 127, q = roundf(x / d),
// the scale is stored as fp16 (so d_a carries one fp16 rounding, as in llama.cpp).
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

constexpr int kQ81Bytes = 36;   // half d, half s, int8 qs[32]
constexpr int kRowsPerWarp = 2;
constexpr int kWarps = 8;

KOPS_HOST_DEVICE constexpr int block_elems(GGUFType t) {
  switch (t) {
    case GGUFType::kQ2_K: case GGUFType::kQ3_K: case GGUFType::kQ4_K: case GGUFType::kQ5_K: case GGUFType::kQ6_K:
      return 256;
    default: return 32;
  }
}

KOPS_HOST_DEVICE constexpr int block_bytes(GGUFType t) {
  switch (t) {
    case GGUFType::kQ4_0: return 18;
    case GGUFType::kQ4_1: return 20;
    case GGUFType::kQ5_0: return 22;
    case GGUFType::kQ5_1: return 24;
    case GGUFType::kQ8_0: return 34;
    case GGUFType::kQ2_K: return 84;
    case GGUFType::kQ3_K: return 110;
    case GGUFType::kQ4_K: return 144;
    case GGUFType::kQ5_K: return 176;
    case GGUFType::kQ6_K: return 210;
    default: return 0;
  }
}

bool supported_type(GGUFType t) { return block_bytes(t) != 0; }

__device__ __forceinline__ uint32_t ld8(const uint8_t* p) { return __ldg(p); }
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned short*>(p)); }
/// 32-bit word at a 2-byte aligned address.
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float ldh(const uint8_t* p) { return __half2float(__ushort_as_half(static_cast<unsigned short>(ld16(p)))); }

/// get_scale_min_k4 of ggml (Q4_K / Q5_K): 6-bit scale and min of sub-block j from 12 bytes.
__device__ __forceinline__ void scale_min_k4(const uint8_t* s, int j, int& sc, int& mn) {
  if (j < 4) {
    sc = ld8(s + j) & 63;
    mn = ld8(s + j + 4) & 63;
  } else {
    sc = (ld8(s + j + 4) & 0xF) | ((ld8(s + j - 4) >> 6) << 4);
    mn = (ld8(s + j + 4) >> 4) | ((ld8(s + j) >> 6) << 4);
  }
}

/// Q3_K scale of index idx (0..15), signed (stored minus 32).
__device__ __forceinline__ int q3k_scale(const uint8_t* s, int idx) {
  const int lo = idx < 8 ? (ld8(s + idx) & 0xF) : (ld8(s + idx - 8) >> 4);
  const int hi = (ld8(s + 8 + (idx & 3)) >> (2 * (idx >> 2))) & 3;
  return (lo | (hi << 4)) - 32;
}

/// Spread bit b of each nibble position: bits 0..3 of `t` -> bit 4 of bytes 0..3 (Q5_0 / Q5_1).
__device__ __forceinline__ uint32_t q5_high_bits(uint32_t t) {
  return ((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) | ((t & 8u) << 25);
}

/// Decodes unit `j` of the block at `blk` into packed integer weights and the affine map.
template <GGUFType T>
__device__ __forceinline__ void decode_unit(const uint8_t* blk, int j, uint32_t (&q)[8], float (&alpha)[2],
                                            float (&beta)[2]) {
  if constexpr (T == GGUFType::kQ4_0 || T == GGUFType::kQ4_1) {
    constexpr int off = T == GGUFType::kQ4_0 ? 2 : 4;
    const float d = ldh(blk);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t r = ld32(blk + off + 4 * i);
      q[i] = r & 0x0F0F0F0Fu;
      q[4 + i] = (r >> 4) & 0x0F0F0F0Fu;
    }
    const float b = T == GGUFType::kQ4_0 ? -8.0f * d : ldh(blk + 2);
    alpha[0] = alpha[1] = d;
    beta[0] = beta[1] = b;
  } else if constexpr (T == GGUFType::kQ5_0 || T == GGUFType::kQ5_1) {
    constexpr int qh_off = T == GGUFType::kQ5_0 ? 2 : 4;
    const float d = ldh(blk);
    const uint32_t qh = ld32(blk + qh_off);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t r = ld32(blk + qh_off + 4 + 4 * i);
      q[i] = (r & 0x0F0F0F0Fu) | q5_high_bits(qh >> (4 * i));
      q[4 + i] = ((r >> 4) & 0x0F0F0F0Fu) | q5_high_bits(qh >> (16 + 4 * i));
    }
    const float b = T == GGUFType::kQ5_0 ? -16.0f * d : ldh(blk + 2);
    alpha[0] = alpha[1] = d;
    beta[0] = beta[1] = b;
  } else if constexpr (T == GGUFType::kQ8_0) {
    const float d = ldh(blk);
    #pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = ld32(blk + 2 + 4 * i);
    alpha[0] = alpha[1] = d;
    beta[0] = beta[1] = 0.0f;
  } else if constexpr (T == GGUFType::kQ4_K) {
    const float d = ldh(blk);
    const float dmin = ldh(blk + 2);
    int sc, mn;
    scale_min_k4(blk + 4, j, sc, mn);
    const uint8_t* qs = blk + 16 + 32 * (j >> 1);
    const int sh = 4 * (j & 1);
    #pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = (ld32(qs + 4 * i) >> sh) & 0x0F0F0F0Fu;
    alpha[0] = alpha[1] = d * static_cast<float>(sc);
    beta[0] = beta[1] = -dmin * static_cast<float>(mn);
  } else if constexpr (T == GGUFType::kQ5_K) {
    const float d = ldh(blk);
    const float dmin = ldh(blk + 2);
    int sc, mn;
    scale_min_k4(blk + 4, j, sc, mn);
    const uint8_t* qh = blk + 16;
    const uint8_t* qs = blk + 48 + 32 * (j >> 1);
    const int sh = 4 * (j & 1);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = ((ld32(qs + 4 * i) >> sh) & 0x0F0F0F0Fu) | (((ld32(qh + 4 * i) >> j) & 0x01010101u) << 4);
    }
    alpha[0] = alpha[1] = d * static_cast<float>(sc);
    beta[0] = beta[1] = -dmin * static_cast<float>(mn);
  } else if constexpr (T == GGUFType::kQ6_K) {
    const int n = j >> 2;
    const int p = j & 3;
    const uint8_t* ql = blk + 64 * n + (p & 1) * 32;
    const uint8_t* qh = blk + 128 + 32 * n;
    const int sh = 4 * (p >> 1);
    const float d = ldh(blk + 208);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = ((ld32(ql + 4 * i) >> sh) & 0x0F0F0F0Fu) | (((ld32(qh + 4 * i) >> (2 * p)) & 0x03030303u) << 4);
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sc = static_cast<float>(static_cast<int8_t>(ld8(blk + 192 + 8 * n + 2 * p + h)));
      alpha[h] = d * sc;
      beta[h] = -32.0f * alpha[h];
    }
  } else if constexpr (T == GGUFType::kQ2_K) {
    const int n = j >> 2;
    const int jj = j & 3;
    const float d = ldh(blk + 80);
    const float dmin = ldh(blk + 82);
    const uint8_t* qs = blk + 16 + 32 * n;
    #pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = (ld32(qs + 4 * i) >> (2 * jj)) & 0x03030303u;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int sc = ld8(blk + 8 * n + 2 * jj + h);
      alpha[h] = d * static_cast<float>(sc & 0xF);
      beta[h] = -dmin * static_cast<float>(sc >> 4);
    }
  } else {   // kQ3_K
    const int n = j >> 2;
    const int jj = j & 3;
    const float d = ldh(blk + 108);
    const uint8_t* qs = blk + 32 + 32 * n;
    const int hbit = 4 * n + jj;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = ((ld32(qs + 4 * i) >> (2 * jj)) & 0x03030303u) | (((ld32(blk + 4 * i) >> hbit) & 0x01010101u) << 2);
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      alpha[h] = d * static_cast<float>(q3k_scale(blk + 96, 8 * n + 2 * jj + h));
      beta[h] = -4.0f * alpha[h];
    }
  }
}

/// One q8_1 block -> 8 packed int8 words and d_a.
struct QUnit {
  uint32_t w[8];
  float d;
};

__device__ __forceinline__ QUnit load_q81(const uint8_t* p) {
  QUnit u;
  u.d = ldh(p);
  #pragma unroll
  for (int i = 0; i < 8; ++i) u.w[i] = ld32(p + 4 + 4 * i);
  return u;
}

/// sum over one unit: d_a * (alpha_h * sum(q' qa) + beta_h * sum(qa)) for both halves.
__device__ __forceinline__ float unit_dot(const uint32_t (&q)[8], const float (&alpha)[2], const float (&beta)[2],
                                          const QUnit& a) {
  float r = 0.0f;
  #pragma unroll
  for (int h = 0; h < 2; ++h) {
    int isum = 0, asum = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      isum = __dp4a(static_cast<int>(q[4 * h + i]), static_cast<int>(a.w[4 * h + i]), isum);
      asum = __dp4a(0x01010101, static_cast<int>(a.w[4 * h + i]), asum);
    }
    r = fmaf(alpha[h], static_cast<float>(isum), fmaf(beta[h], static_cast<float>(asum), r));
  }
  return a.d * r;
}

template <GGUFType T, int M_TILE, typename scalar_t>
__device__ __forceinline__ void mmvq_body(scalar_t* __restrict__ out, const uint8_t* __restrict__ wbase,
                                          const uint8_t* __restrict__ a_q81, const int64_t a_row_stride,
                                          const int m, const int n, const int k, const int64_t out_ld) {
  constexpr int R = kRowsPerWarp;
  constexpr int QK = block_elems(T);
  constexpr int UPB = QK / 32;
  constexpr int BYTES = block_bytes(T);
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5)) * R;
  if (n0 >= n) return;
  const int units = k / 32;
  const int64_t row_bytes = static_cast<int64_t>(k / QK) * BYTES;
  const uint8_t* wrow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) wrow[r] = wbase + (n0 + r < n ? n0 + r : n - 1) * row_bytes;

  float acc[M_TILE][R];
  #pragma unroll
  for (int mm = 0; mm < M_TILE; ++mm) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[mm][r] = 0.0f;
  }
  for (int u = lane; u < units; u += 32) {
    QUnit au[M_TILE];
    #pragma unroll
    for (int mm = 0; mm < M_TILE; ++mm) au[mm] = load_q81(a_q81 + (mm < m ? mm : m - 1) * a_row_stride + static_cast<int64_t>(u) * kQ81Bytes);
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      uint32_t q[8];
      float alpha[2], beta[2];
      decode_unit<T>(wrow[r] + static_cast<int64_t>(u / UPB) * BYTES, u % UPB, q, alpha, beta);
      #pragma unroll
      for (int mm = 0; mm < M_TILE; ++mm) acc[mm][r] += unit_dot(q, alpha, beta, au[mm]);
    }
  }
  #pragma unroll
  for (int mm = 0; mm < M_TILE; ++mm) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[mm][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      if (lane == mm * R + r && mm < m && n0 + r < n) dev::from_f(out[mm * out_ld + n0 + r], v);
    }
  }
}

}  // namespace

template <typename scalar_t>
__global__ void gguf_quantize_q8_1_kernel(void* __restrict__ out_q8_1, const scalar_t* __restrict__ a,
                                          const int m, const int k) {
  const int lane = threadIdx.x & 31;
  const int64_t block = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int64_t blocks_per_row = k / 32;
  if (block >= static_cast<int64_t>(m) * blocks_per_row) return;
  const float x = dev::to_f(a[block * 32 + lane]);
  float amax = warp_reduce_max(fabsf(x));
  const float d = amax / 127.0f;
  const int q = amax == 0.0f ? 0 : static_cast<int>(roundf(x / d));
  int sum = q;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  uint8_t* dst = static_cast<uint8_t*>(out_q8_1) + block * kQ81Bytes;
  if (lane == 0) {
    *reinterpret_cast<__half*>(dst) = __float2half_rn(d);
    *reinterpret_cast<__half*>(dst + 2) = __float2half_rn(d * static_cast<float>(sum));
  }
  dst[4 + lane] = static_cast<uint8_t>(static_cast<int8_t>(q));
}

template <typename scalar_t, GGUFType kType, int M_TILE>
__global__ void gguf_mmvq_kernel(scalar_t* __restrict__ out, const uint8_t* __restrict__ qweight,
                                 const void* __restrict__ a_q8_1, const int m, const int n, const int k) {
  mmvq_body<kType, M_TILE, scalar_t>(out, qweight, static_cast<const uint8_t*>(a_q8_1),
                                     static_cast<int64_t>(k / 32) * kQ81Bytes, m, n, k, n);
}

template <typename scalar_t, GGUFType kType, int M_TILE>
__global__ void gguf_moe_mmvq_kernel(scalar_t* __restrict__ out, const uint8_t* __restrict__ qweight,
                                     const int64_t expert_stride_bytes, const void* __restrict__ a_q8_1,
                                     const int32_t* __restrict__ topk_ids, const int num_tokens,
                                     const int topk, const int n, const int k) {
  static_assert(M_TILE == 1, "one (token, slot) row per CTA column: experts differ between slots");
  const int slot = blockIdx.y;              // token * topk + j
  const int token = slot / topk;
  if (token >= num_tokens) return;
  const int expert = topk_ids[slot];
  if (expert < 0) return;                   // dropped slot (expert not on this rank)
  mmvq_body<kType, 1, scalar_t>(out + static_cast<int64_t>(slot) * n, qweight + expert * expert_stride_bytes,
                                static_cast<const uint8_t*>(a_q8_1) + static_cast<int64_t>(token) * (k / 32) * kQ81Bytes,
                                0, 1, n, k, n);
}

namespace {

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

cudaStream_t stream_of(const LaunchContext& ctx) { return reinterpret_cast<cudaStream_t>(ctx.stream); }
bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }
const char* byte_ptr(const Tensor& t) { return static_cast<const char*>(t.data) + t.byte_offset; }

template <typename scalar_t, GGUFType T, int MT>
Status launch_mmvq(scalar_t* out, const uint8_t* qw, const void* aq, int m, int n, int k, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>((n + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp)));
  gguf_mmvq_kernel<scalar_t, T, MT><<<grid, kWarps * 32, 0, s>>>(out, qw, aq, m, n, k);
  return launch_error("gguf_mmvq_kernel");
}

template <typename scalar_t, GGUFType T>
Status launch_mmvq_tiles(scalar_t* out, const uint8_t* qw, const uint8_t* aq, int m, int n, int k, cudaStream_t s) {
  Status st = Status::kSuccess;
  const int64_t a_stride = static_cast<int64_t>(k / 32) * kQ81Bytes;
  for (int m0 = 0; m0 < m && st == Status::kSuccess;) {
    const int rem = m - m0;
    const int mt = rem >= 8 ? 8 : (rem >= 4 ? 4 : (rem >= 2 ? 2 : 1));
    scalar_t* o = out + static_cast<int64_t>(m0) * n;
    const uint8_t* a = aq + m0 * a_stride;
    switch (mt) {
      case 1: st = launch_mmvq<scalar_t, T, 1>(o, qw, a, mt, n, k, s); break;
      case 2: st = launch_mmvq<scalar_t, T, 2>(o, qw, a, mt, n, k, s); break;
      case 4: st = launch_mmvq<scalar_t, T, 4>(o, qw, a, mt, n, k, s); break;
      default: st = launch_mmvq<scalar_t, T, 8>(o, qw, a, mt, n, k, s); break;
    }
    m0 += mt;
  }
  return st;
}

template <typename scalar_t>
Status dispatch_mmvq(GGUFType t, scalar_t* out, const uint8_t* qw, const uint8_t* aq, int m, int n, int k, cudaStream_t s) {
  switch (t) {
    case GGUFType::kQ4_0: return launch_mmvq_tiles<scalar_t, GGUFType::kQ4_0>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ4_1: return launch_mmvq_tiles<scalar_t, GGUFType::kQ4_1>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ5_0: return launch_mmvq_tiles<scalar_t, GGUFType::kQ5_0>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ5_1: return launch_mmvq_tiles<scalar_t, GGUFType::kQ5_1>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ8_0: return launch_mmvq_tiles<scalar_t, GGUFType::kQ8_0>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ2_K: return launch_mmvq_tiles<scalar_t, GGUFType::kQ2_K>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ3_K: return launch_mmvq_tiles<scalar_t, GGUFType::kQ3_K>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ4_K: return launch_mmvq_tiles<scalar_t, GGUFType::kQ4_K>(out, qw, aq, m, n, k, s);
    case GGUFType::kQ5_K: return launch_mmvq_tiles<scalar_t, GGUFType::kQ5_K>(out, qw, aq, m, n, k, s);
    default: return launch_mmvq_tiles<scalar_t, GGUFType::kQ6_K>(out, qw, aq, m, n, k, s);
  }
}

template <typename scalar_t, GGUFType T>
Status launch_moe(scalar_t* out, const uint8_t* qw, int64_t stride, const void* aq, const int32_t* ids, int tokens,
                  int topk, int n, int k, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>((n + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp)),
                  static_cast<unsigned>(tokens * topk));
  gguf_moe_mmvq_kernel<scalar_t, T, 1><<<grid, kWarps * 32, 0, s>>>(out, qw, stride, aq, ids, tokens, topk, n, k);
  return launch_error("gguf_moe_mmvq_kernel");
}

template <typename scalar_t>
Status dispatch_moe(GGUFType t, scalar_t* out, const uint8_t* qw, int64_t stride, const void* aq, const int32_t* ids,
                    int tokens, int topk, int n, int k, cudaStream_t s) {
  switch (t) {
    case GGUFType::kQ4_0: return launch_moe<scalar_t, GGUFType::kQ4_0>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ4_1: return launch_moe<scalar_t, GGUFType::kQ4_1>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ5_0: return launch_moe<scalar_t, GGUFType::kQ5_0>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ5_1: return launch_moe<scalar_t, GGUFType::kQ5_1>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ8_0: return launch_moe<scalar_t, GGUFType::kQ8_0>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ2_K: return launch_moe<scalar_t, GGUFType::kQ2_K>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ3_K: return launch_moe<scalar_t, GGUFType::kQ3_K>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ4_K: return launch_moe<scalar_t, GGUFType::kQ4_K>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    case GGUFType::kQ5_K: return launch_moe<scalar_t, GGUFType::kQ5_K>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
    default: return launch_moe<scalar_t, GGUFType::kQ6_K>(out, qw, stride, aq, ids, tokens, topk, n, k, s);
  }
}

Status check_type(GGUFType t, int64_t k, const char* who) {
  if (!supported_type(t)) {
    return detail::set_error(Status::kNotImplemented, "%s: GGUF type %d has no native decode GEMV (IQ and float types)", who, static_cast<int>(t));
  }
  if (k % block_elems(t) != 0) {
    return detail::set_error(Status::kUnsupportedShape, "%s: K must be a multiple of the %d-element block", who, block_elems(t));
  }
  return Status::kSuccess;
}

}  // namespace

Status gguf_quantize_q8_1(const Tensor& out_q8_1, const Tensor& a, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "gguf_quantize_q8_1: CUDA backend only");
  Status st = check_tensor(a, DType::kInvalid, 2, 1, "gguf_quantize_q8_1.a");
  if (st != Status::kSuccess) return st;
  if (a.dtype != DType::kBFloat16 && a.dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "gguf_quantize_q8_1: activations must be bf16 or fp16");
  }
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  if (m < 1 || k < 32 || k % 32 != 0 || a.stride[1] != 1 || a.stride[0] != k) {
    return detail::set_error(Status::kUnsupportedShape, "gguf_quantize_q8_1: contiguous [M, K] with K a multiple of 32 required");
  }
  if (out_q8_1.data == nullptr || out_q8_1.dtype != DType::kUInt8 || out_q8_1.ndim != 1 ||
      out_q8_1.shape[0] < m * (k / 32) * kQ81Bytes) {
    return detail::set_error(Status::kWorkspaceTooSmall, "gguf_quantize_q8_1: output must be a uint8 vector of at least %lld bytes",
                             static_cast<long long>(m * (k / 32) * kQ81Bytes));
  }
  if (!aligned(byte_ptr(a), 4) || !aligned(byte_ptr(out_q8_1), 4)) {
    return detail::set_error(Status::kMisaligned, "gguf_quantize_q8_1: operands must be 4 B aligned");
  }
  const int64_t blocks = m * (k / 32);
  const int warps = 8;
  const unsigned grid = static_cast<unsigned>((blocks + warps - 1) / warps);
  void* dst = const_cast<char*>(byte_ptr(out_q8_1));
  if (a.dtype == DType::kBFloat16) {
    gguf_quantize_q8_1_kernel<__nv_bfloat16><<<grid, warps * 32, 0, stream_of(ctx)>>>(dst, reinterpret_cast<const __nv_bfloat16*>(byte_ptr(a)), static_cast<int>(m), static_cast<int>(k));
  } else {
    gguf_quantize_q8_1_kernel<__half><<<grid, warps * 32, 0, stream_of(ctx)>>>(dst, reinterpret_cast<const __half*>(byte_ptr(a)), static_cast<int>(m), static_cast<int>(k));
  }
  return launch_error("gguf_quantize_q8_1_kernel");
}

Status gguf_gemm(const Tensor& out, const Tensor& a, const Tensor& qweight, GGUFType type,
                 const Tensor& workspace_q8_1, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "gguf_gemm: CUDA backend only");
  Status st = check_tensor(out, DType::kInvalid, 2, 1, "gguf_gemm.out");
  if (st != Status::kSuccess) return st;
  if ((st = check_tensor(a, DType::kInvalid, 2, 1, "gguf_gemm.a")) != Status::kSuccess) return st;
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = out.shape[1];
  if ((st = check_type(type, k, "gguf_gemm")) != Status::kSuccess) return st;
  if (out.dtype != a.dtype) return detail::set_error(Status::kUnsupportedDtype, "gguf_gemm: output dtype must equal the activation dtype");
  if (out.shape[0] != m) return detail::set_error(Status::kInvalidArgument, "gguf_gemm: out must be [M, N]");
  if (m > 16) return detail::set_error(Status::kNotImplemented, "gguf_gemm: M > 16 needs the mmq path (B7)");
  const int64_t row_bytes = k / block_elems(type) * block_bytes(type);
  if (qweight.data == nullptr || qweight.dtype != DType::kUInt8 || qweight.ndim != 1 || qweight.shape[0] != n * row_bytes) {
    return detail::set_error(Status::kInvalidArgument, "gguf_gemm: qweight must be a uint8 vector of N * K/block * block_bytes (%lld)",
                             static_cast<long long>(n * row_bytes));
  }
  if (out.stride[1] != 1 || out.stride[0] != n || !aligned(byte_ptr(out), 2) || !aligned(byte_ptr(qweight), 2)) {
    return detail::set_error(Status::kMisaligned, "gguf_gemm: contiguous output and 2-byte aligned weights required");
  }
  st = gguf_quantize_q8_1(workspace_q8_1, a, ctx);
  if (st != Status::kSuccess) return st;
  const auto* qw = reinterpret_cast<const uint8_t*>(byte_ptr(qweight));
  const auto* aq = reinterpret_cast<const uint8_t*>(byte_ptr(workspace_q8_1));
  return a.dtype == DType::kBFloat16
             ? dispatch_mmvq<__nv_bfloat16>(type, reinterpret_cast<__nv_bfloat16*>(const_cast<char*>(byte_ptr(out))), qw, aq, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream_of(ctx))
             : dispatch_mmvq<__half>(type, reinterpret_cast<__half*>(const_cast<char*>(byte_ptr(out))), qw, aq, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream_of(ctx));
}

Status gguf_moe_gemm(const Tensor& out, const Tensor& a_q8_1, const Tensor& qweight, int64_t expert_stride_bytes,
                     const Tensor& topk_ids, GGUFType type, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "gguf_moe_gemm: CUDA backend only");
  Status st = check_tensor(out, DType::kInvalid, 2, 1, "gguf_moe_gemm.out");
  if (st != Status::kSuccess) return st;
  if ((st = check_tensor(topk_ids, DType::kInt32, 2, 1, "gguf_moe_gemm.topk_ids")) != Status::kSuccess) return st;
  const int64_t tokens = topk_ids.shape[0];
  const int64_t topk = topk_ids.shape[1];
  const int64_t n = out.shape[1];
  if (out.dtype != DType::kBFloat16 && out.dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "gguf_moe_gemm: output must be bf16 or fp16");
  }
  if (out.shape[0] != tokens * topk) return detail::set_error(Status::kInvalidArgument, "gguf_moe_gemm: out must be [tokens * topk, N]");
  if (a_q8_1.data == nullptr || a_q8_1.dtype != DType::kUInt8 || a_q8_1.ndim != 1 || a_q8_1.shape[0] % (tokens * kQ81Bytes) != 0) {
    return detail::set_error(Status::kInvalidArgument, "gguf_moe_gemm: a_q8_1 must be the uint8 q8_1 buffer of [tokens, K/32] blocks");
  }
  const int64_t k = a_q8_1.shape[0] / tokens / kQ81Bytes * 32;
  if ((st = check_type(type, k, "gguf_moe_gemm")) != Status::kSuccess) return st;
  const int64_t row_bytes = k / block_elems(type) * block_bytes(type);
  if (qweight.data == nullptr || qweight.dtype != DType::kUInt8 || expert_stride_bytes < n * row_bytes) {
    return detail::set_error(Status::kInvalidArgument, "gguf_moe_gemm: expert stride smaller than one expert matrix");
  }
  if (out.stride[1] != 1 || out.stride[0] != n || topk_ids.stride[1] != 1 || topk_ids.stride[0] != topk ||
      !aligned(byte_ptr(qweight), 2) || expert_stride_bytes % 2 != 0 || !aligned(byte_ptr(a_q8_1), 4)) {
    return detail::set_error(Status::kMisaligned, "gguf_moe_gemm: contiguous operands, 2-byte aligned weights and 4-byte aligned activations required");
  }
  const auto* qw = reinterpret_cast<const uint8_t*>(byte_ptr(qweight));
  const void* aq = byte_ptr(a_q8_1);
  const auto* ids = reinterpret_cast<const int32_t*>(byte_ptr(topk_ids));
  void* o = const_cast<char*>(byte_ptr(out));
  return out.dtype == DType::kBFloat16
             ? dispatch_moe<__nv_bfloat16>(type, static_cast<__nv_bfloat16*>(o), qw, expert_stride_bytes, aq, ids, static_cast<int>(tokens), static_cast<int>(topk), static_cast<int>(n), static_cast<int>(k), stream_of(ctx))
             : dispatch_moe<__half>(type, static_cast<__half*>(o), qw, expert_stride_bytes, aq, ids, static_cast<int>(tokens), static_cast<int>(topk), static_cast<int>(n), static_cast<int>(k), stream_of(ctx));
}

}  // namespace kops::gemm
