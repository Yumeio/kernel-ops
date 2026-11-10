// =============================================================================
// kernel-ops | csrc/gemm/gemm_prep.cu
// B4 rows 5-6 native implementation (operand preparation for the GEMM families):
//   weight_transpose, glu_weight_interleave, router_weight_split_bf16x2,
//   int8_weight_colsum, block_scale_swizzle, fp8_scale_to_tma_layout,
//   fp8_block2d_quant_weight.
// B11 operand prep: machete_prepack (Machete SM90 prepacked weight layout).
//
// Contract: csrc/gemm/ops.h + csrc/gemm/gemm_prep_kernels.cuh.
// Load-time operations: run once after the checkpoint is loaded, outside graph
// capture is not required (nothing allocates or synchronizes), and no workspace
// is kept.  Every kernel is architecture-generic.  Layout kernels move bytes
// without arithmetic; the two arithmetic ones (router split, FP8 block quant)
// use correctly rounded fp32 operations so the result is a function of the data
// alone.
//
// Provenance (THIRD_PARTY_NOTICES.md, "Derived implementations"): the 128x4 swizzle is the CUTLASS
// SfAtom / torchao `to_blocked` layout, packed UE8M0 and the FP8 128x128 block quantization follow
// DeepGEMM deep_gemm/utils/math.py @ 5081f85 (MIT).  The Machete prepacked layout follows
// vLLM csrc/quantization/machete (machete_prepacked_layout.cuh / machete_prepack_kernel.cuh,
// Apache-2.0); the block and within-block formulas are re-derived in this file and re-implemented
// without CUTE.  No upstream source text is copied.
//
// Scale-layout conventions (shared with B3 quantize_per_group):
//   128x4 swizzle   offset = tile*512 + (row&31)*16 + ((row>>5)&3)*4 + (col&3),
//                   tile = (row>>7)*ceil(cols/4) + (col>>2), zero padded.
//   TMA col-major   [K/128, round_up(M, 4)] fp32; packed UE8M0 = 4 exponent
//                   bytes along K per int32, byte j in bits [8j, 8j+8).
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

using dev::to_f;

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

template <typename T>
T* mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

Status backend_ok(const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "gemm_prep: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

/// Bytes per logical element; 0 for sub-byte dtypes (handled as nibbles).
int elem_bytes(DType t) {
  const int bits = dtype_bits(t);
  return (bits > 0 && bits % 8 == 0) ? bits / 8 : 0;
}

bool is_nibble(DType t) { return dtype_bits(t) == 4; }

bool byte_dtype(DType t) { return dtype_bits(t) == 8; }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

const void* data_of(const Tensor& t) {
  return static_cast<const char*>(t.data) + t.byte_offset;
}

/// Row-major 2-D view with a unit inner stride (the row stride may be larger).
Status check_rows(const Tensor& t, int expected_ndim, const char* name) {
  Status s = check_tensor(t, DType::kInvalid, expected_ndim, 1, name);
  if (s != Status::kSuccess) return s;
  if (t.stride_unit != StrideUnit::kElement) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: element strides only", name);
  }
  if (t.stride[t.ndim - 1] != 1 && t.shape[t.ndim - 1] > 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last dim must be contiguous", name);
  }
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint8_t e4m3_from_float(float x) { return dev::e4m3_from_float_sat(x); }

/// Smallest power of two >= s as a UE8M0 biased exponent (matches DeepGEMM
/// ceil_to_ue8m0: exponent field + 1 if the mantissa is non-zero, clamped to
/// [1, 254]).
__device__ __forceinline__ uint32_t ue8m0_exponent_ceil(float s) {
  const uint32_t bits = __float_as_uint(s) & 0x7FFFFFFFu;
  uint32_t e = (bits >> 23) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
  e = e < 1u ? 1u : (e > 254u ? 254u : e);
  return e;
}

// =============================================================================
// weight_transpose:  [K, N] -> [N, K]
// =============================================================================

constexpr int kTile = 32;
constexpr int kTileRows = 8;

/// 2/4/8-byte elements: 32x32 shared-memory tile, padded against bank conflicts.
template <typename T>
__global__ void weight_transpose_tile_kernel(T* __restrict__ out, const int64_t out_ld,
                                             const T* __restrict__ in, const int64_t in_ld,
                                             const int64_t rows_k, const int64_t cols_n) {
  __shared__ T tile[kTile][kTile + 1];
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t k0 = static_cast<int64_t>(blockIdx.y) * kTile;
  #pragma unroll
  for (int i = 0; i < kTile; i += kTileRows) {
    const int64_t k = k0 + threadIdx.y + i;
    const int64_t n = n0 + threadIdx.x;
    if (k < rows_k && n < cols_n) tile[threadIdx.y + i][threadIdx.x] = in[k * in_ld + n];
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < kTile; i += kTileRows) {
    const int64_t n = n0 + threadIdx.y + i;
    const int64_t k = k0 + threadIdx.x;
    if (n < cols_n && k < rows_k) out[n * out_ld + k] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/// 1-byte elements, fast path: 64x64 tile moved as 4-byte words in both directions.
/// Requires rows_k % 4 == 0, cols_n % 4 == 0 and 4-byte aligned base/leading dims.
__global__ void weight_transpose_bytes_kernel(uint8_t* __restrict__ out, const int64_t out_ld,
                                              const uint8_t* __restrict__ in, const int64_t in_ld,
                                              const int64_t rows_k, const int64_t cols_n) {
  __shared__ uint8_t tile[64][65];
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * 64;
  const int64_t k0 = static_cast<int64_t>(blockIdx.y) * 64;
  const int c = threadIdx.x & 15;   // word column
  const int r = threadIdx.x >> 4;   // 0..15
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = r + 16 * i;
    const int64_t k = k0 + row;
    const int64_t n = n0 + 4 * c;
    uint32_t w = 0;
    if (k < rows_k && n < cols_n) w = *reinterpret_cast<const uint32_t*>(in + k * in_ld + n);
    tile[row][4 * c + 0] = static_cast<uint8_t>(w);
    tile[row][4 * c + 1] = static_cast<uint8_t>(w >> 8);
    tile[row][4 * c + 2] = static_cast<uint8_t>(w >> 16);
    tile[row][4 * c + 3] = static_cast<uint8_t>(w >> 24);
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int nr = r + 16 * i;
    const int64_t n = n0 + nr;
    const int64_t k = k0 + 4 * c;
    if (n < cols_n && k < rows_k) {
      const uint32_t w = static_cast<uint32_t>(tile[4 * c + 0][nr]) |
                         (static_cast<uint32_t>(tile[4 * c + 1][nr]) << 8) |
                         (static_cast<uint32_t>(tile[4 * c + 2][nr]) << 16) |
                         (static_cast<uint32_t>(tile[4 * c + 3][nr]) << 24);
      *reinterpret_cast<uint32_t*>(out + n * out_ld + k) = w;
    }
  }
}

/// Packed 4-bit elements, low nibble = even element.  ld values are in bytes;
/// rows_k and cols_n are logical element counts and both must be even.
__global__ void weight_transpose_nibble_kernel(uint8_t* __restrict__ out, const int64_t out_ld,
                                               const uint8_t* __restrict__ in, const int64_t in_ld,
                                               const int64_t rows_k, const int64_t cols_n) {
  __shared__ uint8_t tile[64][65];  // [k][n], one nibble per byte
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * 64;
  const int64_t k0 = static_cast<int64_t>(blockIdx.y) * 64;
  const int c = threadIdx.x & 31;   // byte column inside the tile
  const int r = threadIdx.x >> 5;   // 0..7
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = r + 8 * i;
    const int64_t k = k0 + row;
    const int64_t n = n0 + 2 * c;
    uint8_t b = 0;
    if (k < rows_k && n < cols_n) b = in[k * in_ld + (n >> 1)];
    tile[row][2 * c + 0] = b & 0xFu;
    tile[row][2 * c + 1] = b >> 4;
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int nr = r + 8 * i;
    const int64_t n = n0 + nr;
    const int64_t k = k0 + 2 * c;
    if (n < cols_n && k < rows_k) {
      out[n * out_ld + (k >> 1)] =
          static_cast<uint8_t>(tile[2 * c + 0][nr] | (tile[2 * c + 1][nr] << 4));
    }
  }
}

// =============================================================================
// glu_weight_interleave
// =============================================================================

/// Source row (inside one expert) of destination row `r`.  `src`/`dst` follow the
/// header convention: 0 = [gate | up] halves, d >= 1 = blocks of d rows that
/// alternate gate, up.
__device__ __forceinline__ int glu_source_row(int r, int inter, int src, int dst) {
  int is_up;
  int j;
  if (dst == 0) {
    is_up = r >= inter;
    j = is_up ? r - inter : r;
  } else {
    const int block = r / (2 * dst);
    const int within = r - block * 2 * dst;
    is_up = within >= dst;
    j = block * dst + (within - (is_up ? dst : 0));
  }
  if (src == 0) return j + (is_up ? inter : 0);
  const int block = j / src;
  return block * 2 * src + (is_up ? src : 0) + (j - block * src);
}

template <typename V>
__device__ __forceinline__ void copy_row(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                                         int64_t bytes) {
  const int64_t n = bytes / static_cast<int64_t>(sizeof(V));
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    reinterpret_cast<V*>(dst)[i] = reinterpret_cast<const V*>(src)[i];
  }
}

__global__ void glu_weight_interleave_rows_kernel(
    uint8_t* __restrict__ out_weight, uint8_t* __restrict__ out_scales,
    uint8_t* __restrict__ out_bias, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales, const uint8_t* __restrict__ bias, const int64_t row_bytes,
    const int64_t scale_row_bytes, const int bias_elem_bytes, const int intermediate,
    const int src_interleave, const int dst_interleave) {
  const int rows = 2 * intermediate;
  const int64_t row_global = blockIdx.x;  // expert * rows + dst_row
  const int64_t expert = row_global / rows;
  const int dst_row = static_cast<int>(row_global - expert * rows);
  const int src_row = glu_source_row(dst_row, intermediate, src_interleave, dst_interleave);
  const int64_t src_global = expert * rows + src_row;
  const int which = blockIdx.y;  // 0 weight, 1 scales, 2 bias
  if (which == 0) {
    uint8_t* d = out_weight + row_global * row_bytes;
    const uint8_t* s = weight + src_global * row_bytes;
    if ((row_bytes & 15) == 0) copy_row<uint4>(d, s, row_bytes);
    else if ((row_bytes & 3) == 0) copy_row<uint32_t>(d, s, row_bytes);
    else copy_row<uint8_t>(d, s, row_bytes);
  } else if (which == 1 && out_scales != nullptr) {
    uint8_t* d = out_scales + row_global * scale_row_bytes;
    const uint8_t* s = scales + src_global * scale_row_bytes;
    if ((scale_row_bytes & 15) == 0) copy_row<uint4>(d, s, scale_row_bytes);
    else if ((scale_row_bytes & 3) == 0) copy_row<uint32_t>(d, s, scale_row_bytes);
    else copy_row<uint8_t>(d, s, scale_row_bytes);
  } else if (which == 2 && out_bias != nullptr && threadIdx.x == 0) {
    for (int b = 0; b < bias_elem_bytes; ++b) {
      out_bias[row_global * bias_elem_bytes + b] = bias[src_global * bias_elem_bytes + b];
    }
  }
}

// =============================================================================
// router_weight_split_bf16x2
// =============================================================================

__global__ void router_weight_split_vec_kernel(__nv_bfloat16* __restrict__ w_hi,
                                               __nv_bfloat16* __restrict__ w_lo,
                                               const float* __restrict__ w, const int64_t numel) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t vecs = numel / 4;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < vecs;
       i += stride) {
    const float4 v = reinterpret_cast<const float4*>(w)[i];
    float x[4] = {v.x, v.y, v.z, v.w};
    __nv_bfloat16 hi[4];
    __nv_bfloat16 lo[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      hi[j] = __float2bfloat16_rn(x[j]);
      lo[j] = __float2bfloat16_rn((x[j] - __bfloat162float(hi[j])) * 256.0f);
    }
    reinterpret_cast<uint2*>(w_hi)[i] = *reinterpret_cast<const uint2*>(hi);
    reinterpret_cast<uint2*>(w_lo)[i] = *reinterpret_cast<const uint2*>(lo);
  }
  // Tail (numel % 4), one element per thread of the first block.
  const int64_t tail = numel - vecs * 4;
  if (blockIdx.x == 0 && threadIdx.x < tail) {
    const int64_t i = vecs * 4 + threadIdx.x;
    const __nv_bfloat16 hi = __float2bfloat16_rn(w[i]);
    w_hi[i] = hi;
    w_lo[i] = __float2bfloat16_rn((w[i] - __bfloat162float(hi)) * 256.0f);
  }
}

// =============================================================================
// int8_weight_colsum:  colsum[n] = sum_k qB[n, k]
// =============================================================================

template <bool kVec>
__global__ void int8_weight_colsum_rows_kernel(int32_t* __restrict__ colsum,
                                               const int8_t* __restrict__ weight,
                                               const int64_t weight_ld, const int size_n,
                                               const int size_k) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= size_n) return;
  const int8_t* row = weight + static_cast<int64_t>(n) * weight_ld;
  int acc = 0;
  if (kVec) {
    const int vecs = size_k / 16;
    for (int i = lane; i < vecs; i += 32) {
      const int4 v = reinterpret_cast<const int4*>(row)[i];
      acc = __dp4a(v.x, 0x01010101, acc);
      acc = __dp4a(v.y, 0x01010101, acc);
      acc = __dp4a(v.z, 0x01010101, acc);
      acc = __dp4a(v.w, 0x01010101, acc);
    }
    for (int k = vecs * 16 + lane; k < size_k; k += 32) acc += row[k];
  } else {
    for (int k = lane; k < size_k; k += 32) acc += row[k];
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) colsum[n] = acc;
}

// =============================================================================
// block_scale_swizzle (128x4 tiles), both directions
// =============================================================================

/// One CTA moves a (128 rows x 32 block columns) region = 8 consecutive 128x4 tiles of one tile
/// row.  The linear side is read/written as 32-byte row segments, the tiled side as one
/// contiguous 4 KB run; the transposition happens in shared memory.  `kUnswizzle` selects which
/// side is the tiled one.  The tiled side is always the padded [R_pad, C_pad] domain and the
/// linear side the logical [rows, cols] domain.  Shared word (row, cw) is stored at column
/// cw ^ (row >> 5) so the transposed access is nearly conflict free.
template <bool kUnswizzle>
__global__ void block_scale_swizzle_tiles_kernel(uint8_t* __restrict__ out,
                                                 const uint8_t* __restrict__ in, const int rows,
                                                 const int cols, const int k_tiles,
                                                 const int k_groups, const bool linear_aligned) {
  __shared__ uint32_t tile[128 * 8];
  const int m_tile = blockIdx.x / k_groups;
  const int kg = blockIdx.x - m_tile * k_groups;
  const int batch = blockIdx.y;
  const int64_t words_per_batch = static_cast<int64_t>(k_tiles) * ((rows + 127) / 128) * 128;
  uint8_t* linear_base = const_cast<uint8_t*>(kUnswizzle ? out : in) + static_cast<int64_t>(batch) * rows * cols;
  const int row0 = m_tile * 128;
  const int col0 = kg * 32;
  uint32_t* tiled_words = reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(kUnswizzle ? in : out)) +
                          static_cast<int64_t>(batch) * words_per_batch +
                          (static_cast<int64_t>(m_tile) * k_tiles + kg * 8) * 128;

  // Linear-side word (row, cw): bytes col0 + 4*cw .. +3 of row0 + row.
  auto linear_load = [&](int row, int cw) -> uint32_t {
    const int col = col0 + cw * 4;
    if (row0 + row >= rows || col >= cols) return 0u;
    const uint8_t* p = linear_base + static_cast<int64_t>(row0 + row) * cols + col;
    if (linear_aligned && col + 4 <= cols) return *reinterpret_cast<const uint32_t*>(p);
    uint32_t v = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (col + j < cols) v |= static_cast<uint32_t>(p[j]) << (8 * j);
    }
    return v;
  };
  auto linear_store = [&](int row, int cw, uint32_t v) {
    const int col = col0 + cw * 4;
    if (row0 + row >= rows || col >= cols) return;
    uint8_t* p = linear_base + static_cast<int64_t>(row0 + row) * cols + col;
    if (linear_aligned && col + 4 <= cols) {
      *reinterpret_cast<uint32_t*>(p) = v;
      return;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (col + j < cols) p[j] = static_cast<uint8_t>(v >> (8 * j));
    }
  };

  if (!kUnswizzle) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int w = threadIdx.x + i * 256;
      const int row = w >> 3;
      const int cw = w & 7;
      tile[row * 8 + (cw ^ (row >> 5))] = linear_load(row, cw);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int o = threadIdx.x + i * 256;
      const int tl = o >> 7;            // tile within the group
      const int rem = o & 127;
      const int row = (rem >> 2) + 32 * (rem & 3);
      if (kg * 8 + tl < k_tiles) tiled_words[o] = tile[row * 8 + (tl ^ (row >> 5))];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int o = threadIdx.x + i * 256;
      const int tl = o >> 7;
      const int rem = o & 127;
      const int row = (rem >> 2) + 32 * (rem & 3);
      tile[row * 8 + (tl ^ (row >> 5))] = (kg * 8 + tl < k_tiles) ? tiled_words[o] : 0u;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int w = threadIdx.x + i * 256;
      const int row = w >> 3;
      const int cw = w & 7;
      linear_store(row, cw, tile[row * 8 + (cw ^ (row >> 5))]);
    }
  }
}

// =============================================================================
// fp8_scale_to_tma_layout
// =============================================================================

/// Plain fp32 transpose [M, KB] -> [KB, m_pad]; rows >= valid_m (the TMA alignment padding and,
/// in the grouped form, every row past the last expert) are written as zero.
__global__ void fp8_scale_transpose_kernel(float* __restrict__ out, const float* __restrict__ in,
                                           const int64_t* __restrict__ expert_offsets,
                                           const int num_experts, const int m, const int k_blocks,
                                           const int m_pad) {
  __shared__ float tile[kTile][kTile + 1];
  int valid_m = m;
  if (expert_offsets != nullptr) {
    const int64_t total = expert_offsets[num_experts];
    valid_m = static_cast<int>(total < 0 ? 0 : (total > m ? m : total));
  }
  const int m0 = blockIdx.x * kTile;
  const int kb0 = blockIdx.y * kTile;
  #pragma unroll
  for (int i = 0; i < kTile; i += kTileRows) {
    const int row = m0 + threadIdx.y + i;
    const int kb = kb0 + threadIdx.x;
    float v = 0.0f;
    if (row < valid_m && kb < k_blocks) v = in[static_cast<int64_t>(row) * k_blocks + kb];
    tile[threadIdx.y + i][threadIdx.x] = v;
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < kTile; i += kTileRows) {
    const int kb = kb0 + threadIdx.y + i;
    const int row = m0 + threadIdx.x;
    if (kb < k_blocks && row < m_pad) out[static_cast<int64_t>(kb) * m_pad + row] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/// Packed UE8M0: one thread per (row, group of 4 K-blocks); the four exponent bytes of
/// in[row][4g .. 4g+3] become one int32 at out[g][row].
__global__ void fp8_scale_pack_ue8m0_kernel(int32_t* __restrict__ out, const float* __restrict__ in,
                                            const int64_t* __restrict__ expert_offsets,
                                            const int num_experts, const int m, const int k_blocks,
                                            const int m_pad) {
  int valid_m = m;
  if (expert_offsets != nullptr) {
    const int64_t total = expert_offsets[num_experts];
    valid_m = static_cast<int>(total < 0 ? 0 : (total > m ? m : total));
  }
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int g = blockIdx.y;
  if (row >= m_pad) return;
  uint32_t packed = 0;
  if (row < valid_m) {
    const float4 v = *reinterpret_cast<const float4*>(in + static_cast<int64_t>(row) * k_blocks + 4 * g);
    packed = ((__float_as_uint(v.x) >> 23) & 0xFFu) | (((__float_as_uint(v.y) >> 23) & 0xFFu) << 8) |
             (((__float_as_uint(v.z) >> 23) & 0xFFu) << 16) |
             (((__float_as_uint(v.w) >> 23) & 0xFFu) << 24);
  }
  out[static_cast<int64_t>(g) * m_pad + row] = static_cast<int32_t>(packed);
}

// =============================================================================
// machete_prepack:  [N, K] low-bit weight -> Machete prepacked layout
// =============================================================================
// Provenance (3rd/vllm/csrc/quantization/machete/, Apache-2.0): the tensor is the permutation written by
// `prepack_B_kernel` (machete_prepack_kernel.cuh) under `PrepackedLayoutBTemplate`
// (machete_prepacked_layout.cuh).  It is a pure permutation of the N * K elements:
//
//   block          the weight is cut into PPBlockShape_NK = 128 (N) x 64 (K) blocks; block
//                  (n_tile, k_tile) starts at element 8192 * (n_tile + k_tile * N_tiles)
//                  (bNbKL_to_offset: column-major over blocks, n_tile fastest).  N % 128 == 0 and
//                  K % 64 == 0 are required, so there is no partial block.
//   within block   offset(n, k) = ppblock_ilvd_NK_to_offset()(n, k), which for 0 <= n < 128 and
//                  0 <= k < 64 reduces to the bit-field maps below (each destination bit range is
//                  disjoint, so both directions are exact bit permutations; verified against CUTE
//                  `print(PrepackedLayoutBTemplate<...>::ppblock_ilvd_NK_to_offset())` for every
//                  (n, k) and every atom of MacheteAtomLayout):
//
//   kInt4Mma16  n[0,3)->off[7,10)  n[3]->off[0]  n[4,3)->off[10,13)
//               k[0]->off[2]       k[1,2)->off[5,7]  k[3]->off[1]  k[4,2)->off[3,5)
//   kInt8Mma16  n[0,3)->off[7,10)  n[3]->off[1]  n[4,3)->off[10,13)
//               k[0]->off[0]       k[1,2)->off[5,7]  k[3,3)->off[2,5)
//   kInt4Mma8   n[0,3)->off[7,10)  n[3]->off[2]  n[4,3)->off[10,13)
//   kInt8Mma8   (same as kInt4Mma8)
//               k[0,2)->off[0,2)   k[2,2)->off[5,7)  k[4,2)->off[3,5)
//
// (n[a,b) is bits [a, b) of n.)  Every atom gives one MMA thread value 32 consecutive storage elements
// (16 B for a 4-bit payload, 32 B for 8-bit), puts n[0,3) on the 128-element stride (64 B / 128 B) and
// n[4,3) on 1024, which is why a 128x64 block is 4 KiB / 8 KiB.  kInt4Mma16
// is the only interleaved one: its first eight elements are {n0k0, n8k0, n0k8, n8k8, n0k1, n8k1, n0k9,
// n8k9}, the `get_interleaved_blk_layout<int4, 16, 32>` block Machete's prmt upconvert consumes.
//
// Packing: element 2i is the low nibble of byte i (the shared convention, `WeightPacking::
// kInt4LowNibbleFirst`); 8-bit payloads are one element per byte.  The tensor is therefore exactly as
// large as the unpacked weight and the whole map is a permutation in ELEMENT units — which is what the
// Machete mainloop's own `TVbNbKL_to_offset` model of B assumes.  (A literal transcription of
// `prepack_B_kernel` over a byte-addressed subbyte pointer would scale every block and within-block
// offset by the element size; that is not a layout, so this file keeps the element semantics.)

constexpr int kMacheteBlockN = 128;
constexpr int kMacheteBlockK = 64;
constexpr int kMacheteBlockElems = kMacheteBlockN * kMacheteBlockK;   // 8192 elements per block

/// Storage element offset inside a prepacked block -> (n, k): the field-for-field inverse of the bit
/// maps documented above (each (n, k) bit range lands in its own offset bit range, so this is exact).
__device__ __forceinline__ void machete_unpack_offset(int atom, int o, int* n, int* k) {
  switch (atom) {
    case static_cast<int>(MacheteAtomLayout::kInt4Mma16):
      *n = ((o >> 7) & 7) | ((o & 1) << 3) | (((o >> 10) & 7) << 4);
      *k = ((o >> 2) & 1) | (((o >> 5) & 3) << 1) | (((o >> 1) & 1) << 3) | (((o >> 3) & 3) << 4);
      break;
    case static_cast<int>(MacheteAtomLayout::kInt8Mma16):
      *n = ((o >> 7) & 7) | (((o >> 1) & 1) << 3) | (((o >> 10) & 7) << 4);
      *k = (o & 1) | (((o >> 5) & 3) << 1) | (((o >> 2) & 7) << 3);
      break;
    default:
      *n = ((o >> 7) & 7) | (((o >> 2) & 1) << 3) | (((o >> 10) & 7) << 4);
      *k = (o & 3) | (((o >> 5) & 3) << 2) | (((o >> 3) & 3) << 4);
      break;
  }
}

// =============================================================================
// fp8_block2d_quant_weight
// =============================================================================

/// s = max(amax, floor) / 448 (optionally rounded up to a power of two), q = sat_e4m3(w / s).
/// The floor and the ceil-to-UE8M0 rule follow DeepGEMM per_block_cast_to_fp8; the quotient
/// is a true fp32 division (the DeepGEMM reference multiplies by the reciprocal).
constexpr float kWeightAmaxFloor = 1e-4f;

__device__ __forceinline__ float block_scale(float amax, bool ue8m0) {
  float s = fmaxf(amax, kWeightAmaxFloor) / 448.0f;
  if (ue8m0) s = __uint_as_float(ue8m0_exponent_ceil(s) << 23);
  return s;
}

/// Eight consecutive elements as fp32 through 16-byte loads (fp32 rows are only 16 B aligned).
template <typename in_t>
__device__ __forceinline__ void load8(const in_t* __restrict__ src, float (&x)[8]) {
  if constexpr (sizeof(in_t) == 4) {
    const float4 a = reinterpret_cast<const float4*>(src)[0];
    const float4 b = reinterpret_cast<const float4*>(src)[1];
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w;
    x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  } else {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const in_t* e = reinterpret_cast<const in_t*>(&raw);
    #pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = to_f<in_t>(e[j]);
  }
}

/// Generic block kernel, one CTA per (block_n x block_k) block; two passes through L1.
template <typename in_t>
__global__ void fp8_block2d_quant_weight_kernel(
    uint8_t* __restrict__ out, const int64_t out_ld, float* __restrict__ scales,
    const int64_t scales_ld, const in_t* __restrict__ weight, const int64_t weight_ld,
    const int size_n, const int size_k, const int block_n, const int block_k, const bool ue8m0) {
  __shared__ float smem[32];
  const int n0 = blockIdx.y * block_n;
  const int k0 = blockIdx.x * block_k;
  const int n1 = min(n0 + block_n, size_n);
  const int k1 = min(k0 + block_k, size_k);
  float amax = 0.0f;
  for (int n = n0; n < n1; ++n) {
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
      amax = fmaxf(amax, fabsf(to_f<in_t>(weight[static_cast<int64_t>(n) * weight_ld + k])));
    }
  }
  amax = kops::block_reduce_max<float, 8>(amax, smem);
  __shared__ float s_shared;
  if (threadIdx.x == 0) {
    s_shared = block_scale(amax, ue8m0);
    scales[static_cast<int64_t>(blockIdx.y) * scales_ld + blockIdx.x] = s_shared;
  }
  __syncthreads();
  const float s = s_shared;
  for (int n = n0; n < n1; ++n) {
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
      out[static_cast<int64_t>(n) * out_ld + k] =
          e4m3_from_float(__fdiv_rn(to_f<in_t>(weight[static_cast<int64_t>(n) * weight_ld + k]), s));
    }
  }
}

/// Fast path: 8-element vectors, the whole block (<= 8 vector rows per thread) held in
/// registers so the weight is read once.  Requires block_k % 8 == 0, 16-byte aligned rows and
/// block_k <= 8 * 128 columns of threads; kRowsPerThread bounds block_n / (threads / VPR).
template <typename in_t, int kBlockThreads, int kMaxRowIters>
__global__ void fp8_block2d_quant_weight_vec_kernel(
    uint8_t* __restrict__ out, const int64_t out_ld, float* __restrict__ scales,
    const int64_t scales_ld, const in_t* __restrict__ weight, const int64_t weight_ld,
    const int size_n, const int size_k, const int block_n, const int block_k, const bool ue8m0) {
  __shared__ float smem[32];
  __shared__ float s_shared;
  const int n0 = blockIdx.y * block_n;
  const int k0 = blockIdx.x * block_k;
  const int vpr = block_k / 8;                 // vectors per block row
  const int rows_per_iter = kBlockThreads / vpr;
  const int tr = threadIdx.x / vpr;            // row lane within an iteration
  const int tv = threadIdx.x - tr * vpr;       // vector column
  const int k = k0 + tv * 8;
  float x[kMaxRowIters][8];
  float amax = 0.0f;
  #pragma unroll
  for (int it = 0; it < kMaxRowIters; ++it) {
    const int n = n0 + tr + it * rows_per_iter;
    const bool live = (tr < rows_per_iter) && (n < min(n0 + block_n, size_n)) && (k + 8 <= size_k);
    if (live) {
      load8<in_t>(weight + static_cast<int64_t>(n) * weight_ld + k, x[it]);
      #pragma unroll
      for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(x[it][j]));
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j) x[it][j] = 0.0f;
    }
  }
  amax = kops::block_reduce_max<float, kBlockThreads / 32>(amax, smem);
  if (threadIdx.x == 0) {
    s_shared = block_scale(amax, ue8m0);
    scales[static_cast<int64_t>(blockIdx.y) * scales_ld + blockIdx.x] = s_shared;
  }
  __syncthreads();
  const float s = s_shared;
  #pragma unroll
  for (int it = 0; it < kMaxRowIters; ++it) {
    const int n = n0 + tr + it * rows_per_iter;
    const bool live = (tr < rows_per_iter) && (n < min(n0 + block_n, size_n)) && (k + 8 <= size_k);
    if (live) {
      uint2 q;
      uint8_t b[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) b[j] = e4m3_from_float(__fdiv_rn(x[it][j], s));
      q = *reinterpret_cast<const uint2*>(b);
      *reinterpret_cast<uint2*>(out + static_cast<int64_t>(n) * out_ld + k) = q;
    }
  }
}

}  // namespace

/// One CTA per prepacked block (blockIdx.x = n tile, blockIdx.y = k tile), 128 threads.  The block's
/// input rows are staged in shared memory (coalesced 16-byte loads: a 4-bit block is 128 rows x 32 B,
/// an 8-bit block 128 x 64 B) and the output is written as 32-bit words in element order.
__global__ void machete_prepack_kernel(void* __restrict__ out_weight, const void* __restrict__ weight,
                                       const int size_n, const int size_k, const int weight_bits,
                                       const int atom_layout_id) {
  const bool nib4 = weight_bits == 4;
  const int row_bytes = size_k * weight_bits / 8;             // one [K] row, nibble-packed for 4-bit
  const int tile_row_bytes = kMacheteBlockK * weight_bits / 8;   // 32 (4-bit) or 64 (8-bit)
  const int n_tiles = size_n / kMacheteBlockN;
  const int64_t block_bytes = static_cast<int64_t>(kMacheteBlockElems) * weight_bits / 8;
  const int64_t block = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(blockIdx.y) * n_tiles;

  __shared__ __align__(16) uint8_t tile[kMacheteBlockElems];   // elements of this block (4 KiB / 8 KiB)
  const uint8_t* src = static_cast<const uint8_t*>(weight) +
                       static_cast<int64_t>(blockIdx.x) * kMacheteBlockN * row_bytes +
                       static_cast<int64_t>(blockIdx.y) * tile_row_bytes;
  const int chunks = kMacheteBlockN * (tile_row_bytes / 16);
  for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
    const int row = c / (tile_row_bytes / 16);
    const int sub = c - row * (tile_row_bytes / 16);
    *reinterpret_cast<uint4*>(tile + row * tile_row_bytes + sub * 16) =
        *reinterpret_cast<const uint4*>(src + static_cast<int64_t>(row) * row_bytes + sub * 16);
  }
  __syncthreads();

  uint8_t* dst = static_cast<uint8_t*>(out_weight) + block * block_bytes;
  const int words = static_cast<int>(block_bytes / 4);
  for (int w = threadIdx.x; w < words; w += blockDim.x) {
    uint32_t word = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int o = 4 * w + j;                     // storage element / byte of this block
      int n = 0, k = 0;
      if (nib4) {
        // A byte of the output holds elements 2i (low) and 2i + 1; the two differ in exactly one k bit.
        uint32_t packed = 0;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          machete_unpack_offset(atom_layout_id, 2 * o + half, &n, &k);
          const uint8_t b = tile[n * tile_row_bytes + (k >> 1)];
          packed |= static_cast<uint32_t>((b >> ((k & 1) * 4)) & 0xF) << (4 * half);
        }
        word |= packed << (8 * j);
      } else {
        machete_unpack_offset(atom_layout_id, o, &n, &k);
        word |= static_cast<uint32_t>(tile[n * tile_row_bytes + k]) << (8 * j);
      }
    }
    reinterpret_cast<uint32_t*>(dst)[w] = word;
  }
}

// =============================================================================
// Host launchers
// =============================================================================

Status weight_transpose(const Tensor& out, const Tensor& in, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_rows(in, 2, "weight_transpose.in");
  if (s != Status::kSuccess) return s;
  s = check_rows(out, 2, "weight_transpose.out");
  if (s != Status::kSuccess) return s;
  if (in.dtype != out.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "weight_transpose: in %s != out %s",
                             dtype_name(in.dtype), dtype_name(out.dtype));
  }
  const int64_t rows_k = in.shape[0];
  const int64_t cols_n = in.shape[1];
  if (out.shape[0] != cols_n || out.shape[1] != rows_k) {
    return detail::set_error(Status::kInvalidArgument,
                             "weight_transpose: out [%lld, %lld] != transpose of in [%lld, %lld]",
                             static_cast<long long>(out.shape[0]), static_cast<long long>(out.shape[1]),
                             static_cast<long long>(rows_k), static_cast<long long>(cols_n));
  }
  if (in.stride[0] < cols_n || out.stride[0] < rows_k) {
    return detail::set_error(Status::kUnsupportedLayout, "weight_transpose: row stride < row length");
  }
  if (rows_k == 0 || cols_n == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  const dim3 grid32(static_cast<unsigned>(ceil_div(cols_n, kTile)),
                    static_cast<unsigned>(ceil_div(rows_k, kTile)));
  const dim3 block32(kTile, kTileRows);

  if (is_nibble(in.dtype)) {
    if ((rows_k & 1) != 0 || (cols_n & 1) != 0 || (in.stride[0] & 1) != 0 || (out.stride[0] & 1) != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "weight_transpose: packed 4-bit needs even K, N and row strides");
    }
    const dim3 grid64(static_cast<unsigned>(ceil_div(cols_n, 64)),
                      static_cast<unsigned>(ceil_div(rows_k, 64)));
    weight_transpose_nibble_kernel<<<grid64, 256, 0, stream>>>(
        mut_ptr<uint8_t>(out), out.stride[0] / 2, const_ptr<uint8_t>(in), in.stride[0] / 2, rows_k,
        cols_n);
  } else if (byte_dtype(in.dtype)) {
    const bool fast = (rows_k % 4 == 0) && (cols_n % 4 == 0) && (in.stride[0] % 4 == 0) &&
                      (out.stride[0] % 4 == 0) && aligned(data_of(in), 4) && aligned(data_of(out), 4);
    if (fast) {
      const dim3 grid64(static_cast<unsigned>(ceil_div(cols_n, 64)),
                        static_cast<unsigned>(ceil_div(rows_k, 64)));
      weight_transpose_bytes_kernel<<<grid64, 256, 0, stream>>>(
          mut_ptr<uint8_t>(out), out.stride[0], const_ptr<uint8_t>(in), in.stride[0], rows_k, cols_n);
    } else {
      weight_transpose_tile_kernel<uint8_t><<<grid32, block32, 0, stream>>>(
          mut_ptr<uint8_t>(out), out.stride[0], const_ptr<uint8_t>(in), in.stride[0], rows_k, cols_n);
    }
  } else {
    switch (elem_bytes(in.dtype)) {
      case 2:
        weight_transpose_tile_kernel<uint16_t><<<grid32, block32, 0, stream>>>(
            mut_ptr<uint16_t>(out), out.stride[0], const_ptr<uint16_t>(in), in.stride[0], rows_k, cols_n);
        break;
      case 4:
        weight_transpose_tile_kernel<uint32_t><<<grid32, block32, 0, stream>>>(
            mut_ptr<uint32_t>(out), out.stride[0], const_ptr<uint32_t>(in), in.stride[0], rows_k, cols_n);
        break;
      case 8:
        weight_transpose_tile_kernel<uint64_t><<<grid32, block32, 0, stream>>>(
            mut_ptr<uint64_t>(out), out.stride[0], const_ptr<uint64_t>(in), in.stride[0], rows_k, cols_n);
        break;
      default:
        return detail::set_error(Status::kUnsupportedDtype, "weight_transpose: dtype %s",
                                 dtype_name(in.dtype));
    }
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("weight_transpose", err);
}

Status glu_weight_interleave(const Tensor& out_weight, const Tensor& out_scales,
                             const Tensor& out_bias, const Tensor& weight, const Tensor& scales,
                             const Tensor& bias, int32_t src_interleave, int32_t dst_interleave,
                             const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (weight.ndim != 2 && weight.ndim != 3) {
    return detail::set_error(Status::kInvalidArgument, "glu_weight_interleave: weight must be [2I, K] or [E, 2I, K]");
  }
  s = check_tensor(weight, DType::kInvalid, weight.ndim, 1, "glu_weight_interleave.weight");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out_weight, weight.dtype, weight.ndim, 1, "glu_weight_interleave.out_weight");
  if (s != Status::kSuccess) return s;
  const bool contiguous_ok = tensor_is_contiguous(weight) && tensor_is_contiguous(out_weight);
  if (!contiguous_ok) {
    return detail::set_error(Status::kUnsupportedLayout, "glu_weight_interleave: contiguous tensors only");
  }
  const int64_t experts = weight.ndim == 3 ? weight.shape[0] : 1;
  const int64_t rows = weight.shape[weight.ndim - 2];
  const int64_t k = weight.shape[weight.ndim - 1];
  if ((rows & 1) != 0 || rows == 0) {
    return detail::set_error(Status::kInvalidArgument, "glu_weight_interleave: rows %lld must be even and > 0",
                             static_cast<long long>(rows));
  }
  const int64_t inter = rows / 2;
  for (int d = 0; d < weight.ndim; ++d) {
    if (out_weight.shape[d] != weight.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "glu_weight_interleave: out_weight shape mismatch");
    }
  }
  for (const int32_t il : {src_interleave, dst_interleave}) {
    if (il < 0 || (il > 0 && inter % il != 0)) {
      return detail::set_error(Status::kUnsupportedShape,
                               "glu_weight_interleave: interleave %d must be 0 or divide I = %lld", il,
                               static_cast<long long>(inter));
    }
  }
  const int64_t row_bytes = (k * dtype_bits(weight.dtype)) / 8;
  if (row_bytes * 8 != k * dtype_bits(weight.dtype)) {
    return detail::set_error(Status::kUnsupportedShape, "glu_weight_interleave: row is not a whole number of bytes");
  }

  int64_t scale_row_bytes = 0;
  const bool has_scales = tensor_present(scales);
  if (has_scales) {
    s = check_tensor(scales, DType::kInvalid, weight.ndim, 1, "glu_weight_interleave.scales");
    if (s != Status::kSuccess) return s;
    s = check_tensor(out_scales, scales.dtype, scales.ndim, 1, "glu_weight_interleave.out_scales");
    if (s != Status::kSuccess) return s;
    if (!tensor_is_contiguous(scales) || !tensor_is_contiguous(out_scales) ||
        scales.shape[scales.ndim - 2] != rows || (weight.ndim == 3 && scales.shape[0] != experts) ||
        out_scales.shape[out_scales.ndim - 2] != rows) {
      return detail::set_error(Status::kInvalidArgument,
                               "glu_weight_interleave: scales must be contiguous with one row per weight row");
    }
    scale_row_bytes = (scales.shape[scales.ndim - 1] * dtype_bits(scales.dtype)) / 8;
  }
  int bias_elem_bytes = 0;
  const bool has_bias = tensor_present(bias);
  if (has_bias) {
    s = check_tensor(bias, DType::kInvalid, weight.ndim - 1, 1, "glu_weight_interleave.bias");
    if (s != Status::kSuccess) return s;
    s = check_tensor(out_bias, bias.dtype, bias.ndim, 1, "glu_weight_interleave.out_bias");
    if (s != Status::kSuccess) return s;
    bias_elem_bytes = elem_bytes(bias.dtype);
    if (bias_elem_bytes == 0 || !tensor_is_contiguous(bias) || !tensor_is_contiguous(out_bias) ||
        bias.shape[bias.ndim - 1] != rows || out_bias.shape[out_bias.ndim - 1] != rows) {
      return detail::set_error(Status::kInvalidArgument,
                               "glu_weight_interleave: bias must be contiguous [.., 2I] with a byte-sized dtype");
    }
  }

  const int64_t total_rows = experts * rows;
  if (total_rows > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "glu_weight_interleave: too many rows");
  }
  const dim3 grid(static_cast<unsigned>(total_rows), (has_scales || has_bias) ? 3u : 1u);
  glu_weight_interleave_rows_kernel<<<grid, 256, 0, stream_of(ctx)>>>(
      mut_ptr<uint8_t>(out_weight), has_scales ? mut_ptr<uint8_t>(out_scales) : nullptr,
      has_bias ? mut_ptr<uint8_t>(out_bias) : nullptr, const_ptr<uint8_t>(weight),
      has_scales ? const_ptr<uint8_t>(scales) : nullptr, has_bias ? const_ptr<uint8_t>(bias) : nullptr,
      row_bytes, scale_row_bytes, bias_elem_bytes, static_cast<int>(inter), src_interleave,
      dst_interleave);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("glu_weight_interleave", err);
}

Status machete_prepack(const Tensor& out, const Tensor& weight, int32_t weight_bits,
                       int32_t atom_layout_id, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (atom_layout_id < static_cast<int32_t>(MacheteAtomLayout::kInt4Mma16) ||
      atom_layout_id > static_cast<int32_t>(MacheteAtomLayout::kInt8Mma8)) {
    return detail::set_error(Status::kInvalidArgument,
                             "machete_prepack: atom_layout_id %d outside [0, 3] (MacheteAtomLayout)",
                             atom_layout_id);
  }
  s = check_rows(weight, 2, "machete_prepack.weight");
  if (s != Status::kSuccess) return s;
  const int64_t n = weight.shape[0];
  const int64_t k = weight.shape[1];
  const int bits = dtype_bits(weight.dtype);
  if (bits != 4 && bits != 8) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "machete_prepack: weight dtype %s is neither packed 4-bit nor 8-bit",
                             dtype_name(weight.dtype));
  }
  if (weight_bits != bits) {
    return detail::set_error(Status::kInvalidArgument,
                             "machete_prepack: weight_bits %d disagrees with dtype %s (%d bits)",
                             weight_bits, dtype_name(weight.dtype), bits);
  }
  const bool atom4 = atom_layout_id == static_cast<int32_t>(MacheteAtomLayout::kInt4Mma16) ||
                     atom_layout_id == static_cast<int32_t>(MacheteAtomLayout::kInt4Mma8);
  if (atom4 != (bits == 4)) {
    return detail::set_error(Status::kInvalidArgument,
                             "machete_prepack: atom_layout_id %d is a %d-bit payload layout, weight is %d-bit",
                             atom_layout_id, atom4 ? 4 : 8, bits);
  }
  if (weight.stride[0] != k) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "machete_prepack: weight must be contiguous ([N, K] with row stride K)");
  }
  s = check_tensor(out, weight.dtype, -1, 4, "machete_prepack.out");
  if (s != Status::kSuccess) return s;
  const int64_t elems = n * k;
  if (out.ndim == 2) {
    if (out.shape[0] != n || out.shape[1] != k || out.stride[1] != 1 || out.stride[0] != k) {
      return detail::set_error(Status::kInvalidArgument,
                               "machete_prepack: out must be a contiguous [%lld, %lld] view",
                               static_cast<long long>(n), static_cast<long long>(k));
    }
  } else if (out.ndim == 1) {
    if (tensor_numel(out) < elems) {
      return detail::set_error(Status::kCapacityExceeded,
                               "machete_prepack: out holds %lld elements, the pack needs %lld",
                               static_cast<long long>(tensor_numel(out)), static_cast<long long>(elems));
    }
  } else {
    return detail::set_error(Status::kInvalidArgument, "machete_prepack: out must be [N, K] or a flat capacity view");
  }
  if ((n & (kMacheteBlockN - 1)) != 0 || (k & (kMacheteBlockK - 1)) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "machete_prepack: N %d must be a multiple of 128 and K %d a multiple of 64 "
                             "(the prepacked block is 128 x 64)", static_cast<int>(n), static_cast<int>(k));
  }
  if (n == 0 || k == 0) return Status::kSuccess;
  if (n > INT32_MAX || k > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "machete_prepack: N and K must fit in int32");
  }
  if (!aligned(data_of(weight), 16) || !aligned(data_of(out), 4)) {
    return detail::set_error(Status::kMisaligned,
                             "machete_prepack: weight needs 16-byte alignment (16-byte staging loads), out 4-byte");
  }
  const dim3 grid(static_cast<unsigned>(n / kMacheteBlockN), static_cast<unsigned>(k / kMacheteBlockK));
  machete_prepack_kernel<<<grid, 128, 0, stream_of(ctx)>>>(mut_ptr<uint8_t>(out), data_of(weight),
                                                           static_cast<int>(n), static_cast<int>(k),
                                                           weight_bits, atom_layout_id);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("machete_prepack", err);
}

Status router_weight_split_bf16x2(const Tensor& w_hi, const Tensor& w_lo, const Tensor& w_fp32,
                                  const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(w_fp32, DType::kFloat32, -1, 4, "router_weight_split.w_fp32");
  if (s != Status::kSuccess) return s;
  s = check_tensor(w_hi, DType::kBFloat16, w_fp32.ndim, 2, "router_weight_split.w_hi");
  if (s != Status::kSuccess) return s;
  s = check_tensor(w_lo, DType::kBFloat16, w_fp32.ndim, 2, "router_weight_split.w_lo");
  if (s != Status::kSuccess) return s;
  for (int d = 0; d < w_fp32.ndim; ++d) {
    if (w_hi.shape[d] != w_fp32.shape[d] || w_lo.shape[d] != w_fp32.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "router_weight_split: shape mismatch");
    }
  }
  if (!tensor_is_contiguous(w_fp32) || !tensor_is_contiguous(w_hi) || !tensor_is_contiguous(w_lo)) {
    return detail::set_error(Status::kUnsupportedLayout, "router_weight_split: contiguous tensors only");
  }
  const int64_t numel = tensor_numel(w_fp32);
  if (numel == 0) return Status::kSuccess;
  const bool vec_ok = aligned(data_of(w_fp32), 16) && aligned(data_of(w_hi), 8) && aligned(data_of(w_lo), 8);
  if (!vec_ok) {
    return detail::set_error(Status::kMisaligned, "router_weight_split: w_fp32 needs 16 B, w_hi/w_lo 8 B alignment");
  }
  const int sms = ctx.compute_units > 0 ? ctx.compute_units : 32;
  const int64_t work = std::max<int64_t>(numel / 4, 1);
  const int grid = static_cast<int>(std::min<int64_t>(ceil_div(work, 256), static_cast<int64_t>(sms) * 16));
  router_weight_split_vec_kernel<<<std::max(grid, 1), 256, 0, stream_of(ctx)>>>(
      mut_ptr<__nv_bfloat16>(w_hi), mut_ptr<__nv_bfloat16>(w_lo), const_ptr<float>(w_fp32), numel);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("router_weight_split", err);
}

Status int8_weight_colsum(const Tensor& colsum, const Tensor& weight, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_rows(weight, 2, "int8_weight_colsum.weight");
  if (s != Status::kSuccess) return s;
  if (weight.dtype != DType::kInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "int8_weight_colsum: weight dtype %s",
                             dtype_name(weight.dtype));
  }
  s = check_tensor(colsum, DType::kInt32, 1, 4, "int8_weight_colsum.colsum");
  if (s != Status::kSuccess) return s;
  const int64_t n = weight.shape[0];
  const int64_t k = weight.shape[1];
  if (colsum.shape[0] != n) {
    return detail::set_error(Status::kInvalidArgument, "int8_weight_colsum: colsum has %lld entries, want %lld",
                             static_cast<long long>(colsum.shape[0]), static_cast<long long>(n));
  }
  if (n == 0) return Status::kSuccess;
  if (n > INT32_MAX || k > INT32_MAX / 128) {
    return detail::set_error(Status::kUnsupportedShape, "int8_weight_colsum: K too large for int32 accumulation");
  }
  const bool vec = (weight.stride[0] % 16 == 0) && aligned(data_of(weight), 16);
  const int warps = 8;
  const int grid = static_cast<int>(ceil_div(n, warps));
  if (vec) {
    int8_weight_colsum_rows_kernel<true><<<grid, warps * 32, 0, stream_of(ctx)>>>(
        mut_ptr<int32_t>(colsum), const_ptr<int8_t>(weight), weight.stride[0], static_cast<int>(n),
        static_cast<int>(k));
  } else {
    int8_weight_colsum_rows_kernel<false><<<grid, warps * 32, 0, stream_of(ctx)>>>(
        mut_ptr<int32_t>(colsum), const_ptr<int8_t>(weight), weight.stride[0], static_cast<int>(n),
        static_cast<int>(k));
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("int8_weight_colsum", err);
}

Status block_scale_swizzle(const Tensor& out, const Tensor& in, bool unswizzle,
                           const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (in.ndim != 2 && in.ndim != 3) {
    return detail::set_error(Status::kInvalidArgument, "block_scale_swizzle: expected [rows, cols] or [batch, rows, cols]");
  }
  s = check_tensor(in, DType::kInvalid, in.ndim, 1, "block_scale_swizzle.in");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, in.dtype, in.ndim, 1, "block_scale_swizzle.out");
  if (s != Status::kSuccess) return s;
  if (!byte_dtype(in.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "block_scale_swizzle: scale dtype %s must be 1 byte",
                             dtype_name(in.dtype));
  }
  if (!tensor_is_contiguous(in) || !tensor_is_contiguous(out)) {
    return detail::set_error(Status::kUnsupportedLayout, "block_scale_swizzle: contiguous tensors only");
  }
  const Tensor& linear = unswizzle ? out : in;
  const Tensor& tiled = unswizzle ? in : out;
  const int64_t batch = linear.ndim == 3 ? linear.shape[0] : 1;
  const int64_t rows = linear.shape[linear.ndim - 2];
  const int64_t cols = linear.shape[linear.ndim - 1];
  const int64_t rows_pad = round_up(rows, 128);
  const int64_t cols_pad = round_up(cols, 4);
  if ((tiled.ndim == 3 ? tiled.shape[0] : 1) != batch || tiled.shape[tiled.ndim - 2] != rows_pad ||
      tiled.shape[tiled.ndim - 1] != cols_pad) {
    return detail::set_error(Status::kInvalidArgument,
                             "block_scale_swizzle: tiled side must be [%lld, %lld, %lld]",
                             static_cast<long long>(batch), static_cast<long long>(rows_pad),
                             static_cast<long long>(cols_pad));
  }
  if (rows == 0 || cols == 0 || batch == 0) return Status::kSuccess;
  if (!aligned(data_of(tiled), 4)) {
    return detail::set_error(Status::kMisaligned, "block_scale_swizzle: tiled side needs 4-byte alignment");
  }
  const int64_t k_tiles = cols_pad / 4;
  const int64_t k_groups = ceil_div(k_tiles, 8);
  const int64_t m_tiles = rows_pad / 128;
  if (batch > 65535 || m_tiles * k_groups > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "block_scale_swizzle: batch or tile count too large");
  }
  const Tensor& lin = unswizzle ? out : in;
  const bool linear_aligned = (cols % 4 == 0) && aligned(data_of(lin), 4);
  const dim3 grid(static_cast<unsigned>(m_tiles * k_groups), static_cast<unsigned>(batch));
  if (unswizzle) {
    block_scale_swizzle_tiles_kernel<true><<<grid, 256, 0, stream_of(ctx)>>>(
        mut_ptr<uint8_t>(out), const_ptr<uint8_t>(in), static_cast<int>(rows), static_cast<int>(cols),
        static_cast<int>(k_tiles), static_cast<int>(k_groups), linear_aligned);
  } else {
    block_scale_swizzle_tiles_kernel<false><<<grid, 256, 0, stream_of(ctx)>>>(
        mut_ptr<uint8_t>(out), const_ptr<uint8_t>(in), static_cast<int>(rows), static_cast<int>(cols),
        static_cast<int>(k_tiles), static_cast<int>(k_groups), linear_aligned);
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("block_scale_swizzle", err);
}

Status fp8_scale_to_tma_layout(const Tensor& out, const Tensor& in, const Tensor& expert_offsets,
                               bool pack_ue8m0, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(in, DType::kFloat32, 2, 4, "fp8_scale_to_tma_layout.in");
  if (s != Status::kSuccess) return s;
  if (in.stride[1] != 1 || in.stride[0] != in.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout, "fp8_scale_to_tma_layout: input must be contiguous row-major");
  }
  const int64_t m = in.shape[0];
  const int64_t k_blocks = in.shape[1];
  const int64_t m_pad = round_up(m, 4);
  const int64_t out_rows = pack_ue8m0 ? k_blocks / 4 : k_blocks;
  if (pack_ue8m0 && (k_blocks % 4 != 0 || !aligned(data_of(in), 16))) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fp8_scale_to_tma_layout: packed UE8M0 needs K/128 %% 4 == 0 and 16-byte input");
  }
  s = check_tensor(out, pack_ue8m0 ? DType::kInt32 : DType::kFloat32, 2, 4, "fp8_scale_to_tma_layout.out");
  if (s != Status::kSuccess) return s;
  if (out.shape[0] != out_rows || out.shape[1] != m_pad || out.stride[1] != 1 || out.stride[0] != m_pad) {
    return detail::set_error(Status::kInvalidArgument,
                             "fp8_scale_to_tma_layout: out must be contiguous [%lld, %lld]",
                             static_cast<long long>(out_rows), static_cast<long long>(m_pad));
  }
  const int64_t* offsets = nullptr;
  int num_experts = 0;
  if (tensor_present(expert_offsets)) {
    s = check_tensor(expert_offsets, DType::kInt64, 1, 8, "fp8_scale_to_tma_layout.expert_offsets");
    if (s != Status::kSuccess) return s;
    if (expert_offsets.shape[0] < 2) {
      return detail::set_error(Status::kInvalidArgument, "fp8_scale_to_tma_layout: expert_offsets needs G + 1 entries");
    }
    offsets = const_ptr<int64_t>(expert_offsets);
    num_experts = static_cast<int>(expert_offsets.shape[0] - 1);
  }
  if (m == 0 || k_blocks == 0) return Status::kSuccess;
  if (pack_ue8m0) {
    const dim3 grid(static_cast<unsigned>(ceil_div(m_pad, 128)), static_cast<unsigned>(k_blocks / 4));
    fp8_scale_pack_ue8m0_kernel<<<grid, 128, 0, stream_of(ctx)>>>(
        mut_ptr<int32_t>(out), const_ptr<float>(in), offsets, num_experts, static_cast<int>(m),
        static_cast<int>(k_blocks), static_cast<int>(m_pad));
  } else {
    const dim3 grid(static_cast<unsigned>(ceil_div(m_pad, kTile)), static_cast<unsigned>(ceil_div(k_blocks, kTile)));
    fp8_scale_transpose_kernel<<<grid, dim3(kTile, kTileRows), 0, stream_of(ctx)>>>(
        mut_ptr<float>(out), const_ptr<float>(in), offsets, num_experts, static_cast<int>(m),
        static_cast<int>(k_blocks), static_cast<int>(m_pad));
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("fp8_scale_to_tma_layout", err);
}

Status fp8_block2d_quant_weight(const Tensor& out, const Tensor& scales, const Tensor& weight,
                                int32_t block_n, int32_t block_k, bool ue8m0,
                                const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_rows(weight, 2, "fp8_block2d_quant_weight.weight");
  if (s != Status::kSuccess) return s;
  if (weight.dtype != DType::kBFloat16 && weight.dtype != DType::kFloat16 && weight.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_block2d_quant_weight: weight dtype %s",
                             dtype_name(weight.dtype));
  }
  s = check_rows(out, 2, "fp8_block2d_quant_weight.out");
  if (s != Status::kSuccess) return s;
  if (out.dtype != DType::kFloat8E4M3) {
    return detail::set_error(Status::kUnsupportedDtype, "fp8_block2d_quant_weight: out dtype %s",
                             dtype_name(out.dtype));
  }
  if (block_n <= 0 || block_k <= 0 || block_n > 1024 || block_k > 1024) {
    return detail::set_error(Status::kUnsupportedShape, "fp8_block2d_quant_weight: block %d x %d outside (0, 1024]",
                             block_n, block_k);
  }
  const int64_t n = weight.shape[0];
  const int64_t k = weight.shape[1];
  if (out.shape[0] != n || out.shape[1] != k) {
    return detail::set_error(Status::kInvalidArgument, "fp8_block2d_quant_weight: out shape != weight shape");
  }
  const int64_t sn = ceil_div(n, block_n);
  const int64_t sk = ceil_div(k, block_k);
  s = check_tensor(scales, DType::kFloat32, 2, 4, "fp8_block2d_quant_weight.scales");
  if (s != Status::kSuccess) return s;
  if (scales.shape[0] != sn || scales.shape[1] != sk || scales.stride[1] != 1) {
    return detail::set_error(Status::kInvalidArgument, "fp8_block2d_quant_weight: scales must be [%lld, %lld]",
                             static_cast<long long>(sn), static_cast<long long>(sk));
  }
  if (n == 0 || k == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  const dim3 grid(static_cast<unsigned>(sk), static_cast<unsigned>(sn));

  // The register-resident path covers block_k in {8*j}: 256 threads give 256 / (block_k / 8)
  // vector rows per iteration, and kMaxRowIters iterations must span block_n rows.
  const int vpr = block_k / 8;
  const bool vec_shape = (block_k % 8 == 0) && vpr <= 256 && (256 % vpr == 0) &&
                         (weight.stride[0] % 8 == 0) && (out.stride[0] % 8 == 0) && (k % 8 == 0) &&
                         aligned(data_of(weight), 16) && aligned(data_of(out), 8);
  const int rows_per_iter = vec_shape ? 256 / vpr : 0;
  const bool vec_rows = vec_shape && block_n <= rows_per_iter * 8;

#define KOPS_FP8_BLOCK(T)                                                                          \
  do {                                                                                             \
    if (vec_rows) {                                                                                \
      fp8_block2d_quant_weight_vec_kernel<T, 256, 8><<<grid, 256, 0, stream>>>(                    \
          mut_ptr<uint8_t>(out), out.stride[0], mut_ptr<float>(scales), scales.stride[0],          \
          const_ptr<T>(weight), weight.stride[0], static_cast<int>(n), static_cast<int>(k),        \
          block_n, block_k, ue8m0);                                                                \
    } else {                                                                                       \
      fp8_block2d_quant_weight_kernel<T><<<grid, 256, 0, stream>>>(                        \
          mut_ptr<uint8_t>(out), out.stride[0], mut_ptr<float>(scales), scales.stride[0],          \
          const_ptr<T>(weight), weight.stride[0], static_cast<int>(n), static_cast<int>(k),        \
          block_n, block_k, ue8m0);                                                                \
    }                                                                                              \
  } while (0)
  if (weight.dtype == DType::kBFloat16) KOPS_FP8_BLOCK(__nv_bfloat16);
  else if (weight.dtype == DType::kFloat16) KOPS_FP8_BLOCK(__half);
  else KOPS_FP8_BLOCK(float);
#undef KOPS_FP8_BLOCK
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error("fp8_block2d_quant_weight", err);
}

}  // namespace kops::gemm
