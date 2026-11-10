// =============================================================================
// kernel-ops | csrc/kvcache/kvcache.cu
// B2 native KV-cache implementation:
//   K01  reshape_and_cache (NHD/HND + VllmLegacy), append_paged_kv,
//        ring_kv_append, concat_and_cache_mla (+ grouped)
//   K02  gather_pages / scatter_pages, copy_pages / zero_pages,
//        cp_gather_cache, gather_and_maybe_dequant_cache
//   K03  convert_cache, kv_relayout_pages
// Host copies (swap_pages / copy_batch_async) are driven from the same file.
//
// B3 kernels declared next to these (per-token-head quantization, ds_mla FP8,
// indexer cache, requantize) stay kNotImplemented here; their declarations are
// still linked so the family API is complete.
//
// Slot-mapping contract (kvcache_kernels.cuh): num_tokens = len(slot_mapping);
// source tensors may carry extra padded rows, slot < 0 means skip.
//
// Numerics: native copies are bit-preserving; FP8 writes use e4m3 satfinite and
// scales are dequant multipliers (stored = x / scale). The kernels are
// architecture-generic (no arch-specific instructions), graph-capturable and
// never allocate or synchronize on the host.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

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
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"

namespace kops::kv {
namespace {

using dev::from_f;
using dev::to_f;

// -----------------------------------------------------------------------------
// Device scalar helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

template <KVCodec kCodec>
__host__ __device__ constexpr bool is_fp8_codec() {
  return kCodec == KVCodec::kFP8PerTensor || kCodec == KVCodec::kFP8PerHead;
}

/// Read one cache element as fp32, applying the dequant scale for FP8 codecs.
template <typename cache_t, KVCodec kCodec>
__device__ __forceinline__ float cache_load(cache_t v, float scale) {
  if constexpr (kCodec == KVCodec::kNative) {
    return to_f<cache_t>(v);
  } else {
    return e4m3_to_float(static_cast<uint8_t>(v)) * scale;
  }
}

/// Quantize one fp32 value into the cache element.
template <typename cache_t, KVCodec kCodec>
__device__ __forceinline__ cache_t cache_store(float x, float scale) {
  if constexpr (kCodec == KVCodec::kNative) {
    cache_t y;
    from_f<cache_t>(y, x);
    return y;
  } else {
    const float q = (scale != 0.0f) ? (x / scale) : 0.0f;
    return static_cast<cache_t>(e4m3_from_float(q));
  }
}

/// Scale of one head. Per-tensor scales use stride 0.
__device__ __forceinline__ float head_scale(const float* p, int head, int stride) {
  if (p == nullptr) return 1.0f;
  return p[static_cast<int64_t>(head) * stride];
}

/// Standard element strides of a dense NHD/HND cache when the descriptor left
/// them zero. Kernels only see a single K stride triple; V strides are inferred
/// from the layout and the V head size.
__device__ __forceinline__ bool layout_is_hnd(int64_t page_stride, int64_t head_stride,
                                              int head_size) {
  return head_stride != head_size;
}

/// Byte copy through 16-byte chunks with a scalar tail. `n` bytes.
__device__ __forceinline__ void copy_bytes(void* __restrict__ dst, const void* __restrict__ src,
                                           int64_t n, int tid, int nthreads) {
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15u) == 0u) {
    const int64_t words = n >> 4;
    for (int64_t i = tid; i < words; i += nthreads) {
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
    }
    for (int64_t i = (words << 4) + tid; i < n; i += nthreads) {
      static_cast<uint8_t*>(dst)[i] = static_cast<const uint8_t*>(src)[i];
    }
  } else {
    for (int64_t i = tid; i < n; i += nthreads) {
      static_cast<uint8_t*>(dst)[i] = static_cast<const uint8_t*>(src)[i];
    }
  }
}

__device__ __forceinline__ void zero_bytes(void* __restrict__ dst, int64_t n, int tid,
                                           int nthreads) {
  if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
    const int64_t words = n >> 4;
    for (int64_t i = tid; i < words; i += nthreads) {
      reinterpret_cast<uint4*>(dst)[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int64_t i = (words << 4) + tid; i < n; i += nthreads) {
      static_cast<uint8_t*>(dst)[i] = 0u;
    }
  } else {
    for (int64_t i = tid; i < n; i += nthreads) {
      static_cast<uint8_t*>(dst)[i] = 0u;
    }
  }
}

// -----------------------------------------------------------------------------
// B3 device helpers: per-token-head codecs and record layouts
// -----------------------------------------------------------------------------

constexpr int kMaxRowPairs = 8;
constexpr int kMaxRowDim = 2 * 32 * kMaxRowPairs;

__host__ __device__ __forceinline__ bool token_codec(KVCodec c) {
  return c == KVCodec::kFP8PerTokenHead || c == KVCodec::kInt8PerTokenHead ||
         c == KVCodec::kInt4PerTokenHead;
}

/// Byte/element offset of (slot, head) inside one cache. Byte strides for
/// record codecs and element strides for dense caches share the same formula:
/// cache_t is one byte for every per-token-head codec.
__device__ __forceinline__ int64_t cache_offset(const PagedKVDesc& d, int64_t slot, int head,
                                                bool is_value) {
  const int64_t page = slot / d.page_size;
  const int64_t off = slot - page * d.page_size;
  const int64_t stride_page = is_value ? d.v_stride_page : d.k_stride_page;
  const int64_t stride_token = is_value ? d.v_stride_token : d.k_stride_token;
  const int64_t stride_head = is_value ? d.v_stride_head : d.k_stride_head;
  return page * stride_page + off * stride_token + static_cast<int64_t>(head) * stride_head;
}

template <int P, typename T>
__device__ __forceinline__ void load_row_tokens(const T* src, int pairs, int lane,
                                                float (&x)[2 * P]) {
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int u = lane + 32 * i;
    x[2 * i] = u < pairs ? to_f(src[2 * u]) : 0.0f;
    x[2 * i + 1] = u < pairs ? to_f(src[2 * u + 1]) : 0.0f;
  }
}

__device__ __forceinline__ uint8_t int4_from_padded(float x) {
  const int q = __float2int_rn(x);
  const int c = q < 0 ? 0 : (q > 15 ? 15 : q);
  return static_cast<uint8_t>(c);
}

template <KVCodec kCodec, int P>
__device__ __forceinline__ void quantize_token_row(const float (&x)[2 * P], int pairs, int lane,
                                                   uint8_t* dst, float* meta) {
  constexpr bool kInt4 = kCodec == KVCodec::kInt4PerTokenHead;
  float scale = 1.0f, zero = 0.0f;
  (void)zero;
  if constexpr (kInt4) {
    float mx = -__int_as_float(0x7f800000), mn = __int_as_float(0x7f800000);
#pragma unroll
    for (int i = 0; i < P; ++i) {
      if (lane + 32 * i >= pairs) continue;
      mx = fmaxf(mx, fmaxf(x[2 * i], x[2 * i + 1]));
      mn = fminf(mn, fminf(x[2 * i], x[2 * i + 1]));
    }
    mx = warp_reduce_max(mx);
    mn = -warp_reduce_max(-mn);
    scale = __fdiv_rn(mx - mn, 15.0f);
    scale = (scale == scale) ? fmaxf(scale, 1e-10f) : scale;
    zero = static_cast<float>(__float2int_rn(-__fdiv_rn(mn, scale)));
    scale = __shfl_sync(0xffffffffu, scale, 0);
    zero = __shfl_sync(0xffffffffu, zero, 0);
  } else {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < P; ++i) {
      if (lane + 32 * i >= pairs) continue;
      amax = fmaxf(amax, fmaxf(fabsf(x[2 * i]), fabsf(x[2 * i + 1])));
    }
    amax = warp_reduce_max(amax);
    const float qmax = kCodec == KVCodec::kFP8PerTokenHead ? kFp8E4M3Max : kInt8Max;
    scale = fmaxf(amax, 1e-10f) / qmax;
  }
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int u = lane + 32 * i;
    if (u >= pairs) continue;
    if constexpr (kCodec == KVCodec::kFP8PerTokenHead) {
      const float lo = (x[2 * i] == 0.0f) ? 0.0f : x[2 * i] / scale;
      const float hi = (x[2 * i + 1] == 0.0f) ? 0.0f : x[2 * i + 1] / scale;
      dst[2 * u] = e4m3_from_float(lo);
      dst[2 * u + 1] = e4m3_from_float(hi);
    } else if constexpr (kCodec == KVCodec::kInt8PerTokenHead) {
      dst[2 * u] = static_cast<uint8_t>(
          static_cast<int8_t>(max(-127, min(127, __float2int_rn(x[2 * i] / scale)))));
      dst[2 * u + 1] = static_cast<uint8_t>(
          static_cast<int8_t>(max(-127, min(127, __float2int_rn(x[2 * i + 1] / scale)))));
    } else {
      const uint8_t lo = int4_from_padded(__fadd_rn(rintf(x[2 * i] / scale), zero));
      const uint8_t hi = int4_from_padded(__fadd_rn(rintf(x[2 * i + 1] / scale), zero));
      dst[u] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  if (lane == 0) {
    meta[0] = scale;
    if constexpr (kInt4) meta[1] = zero;
  }
}

template <int P>
__device__ __forceinline__ void quantize_token_row_as(KVCodec codec, const float (&x)[2 * P],
                                                      int pairs, int lane, uint8_t* dst,
                                                      float* meta) {
  if (codec == KVCodec::kFP8PerTokenHead) {
    quantize_token_row<KVCodec::kFP8PerTokenHead, P>(x, pairs, lane, dst, meta);
  } else if (codec == KVCodec::kInt8PerTokenHead) {
    quantize_token_row<KVCodec::kInt8PerTokenHead, P>(x, pairs, lane, dst, meta);
  } else {
    quantize_token_row<KVCodec::kInt4PerTokenHead, P>(x, pairs, lane, dst, meta);
  }
}

/// Decode one element of a cache row (native, FP8 per tensor/head, or a
/// per-token-head codec). Used by the requantizer.
template <int P>
__device__ __forceinline__ void load_decoded_row(const PagedKVDesc& s, bool is_value, int64_t slot,
                                                 int head, int lane, float (&x)[2 * P]) {
  const KVCodec codec = is_value ? s.v_codec : s.k_codec;
  const DType dtype = is_value ? s.v_dtype : s.k_dtype;
  const int dim = is_value ? s.head_dim_v : s.head_dim_k;
  const int pairs = dim / 2;
  const void* data = is_value ? s.v_data : s.k_data;
  const int64_t base = cache_offset(s, slot, head, is_value);
  const float* tensor_scale = is_value ? s.v_scale : s.k_scale;
  const float* meta = static_cast<const float*>(is_value ? s.v_token_scales : s.k_token_scales);
  const int64_t row = slot * s.num_kv_heads + head;
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int u = lane + 32 * i;
    float lo = 0.0f, hi = 0.0f;
    if (u < pairs) {
      if (codec == KVCodec::kNative) {
        if (dtype == DType::kFloat32) {
          lo = static_cast<const float*>(data)[base + 2 * u];
          hi = static_cast<const float*>(data)[base + 2 * u + 1];
        } else if (dtype == DType::kFloat16) {
          lo = __half2float(static_cast<const __half*>(data)[base + 2 * u]);
          hi = __half2float(static_cast<const __half*>(data)[base + 2 * u + 1]);
        } else {
          lo = __bfloat162float(static_cast<const __nv_bfloat16*>(data)[base + 2 * u]);
          hi = __bfloat162float(static_cast<const __nv_bfloat16*>(data)[base + 2 * u + 1]);
        }
      } else if (codec == KVCodec::kFP8PerTensor || codec == KVCodec::kFP8PerHead) {
        const float* sc = tensor_scale + (codec == KVCodec::kFP8PerHead ? head : 0);
        lo = e4m3_to_float(static_cast<const uint8_t*>(data)[base + 2 * u]) * sc[0];
        hi = e4m3_to_float(static_cast<const uint8_t*>(data)[base + 2 * u + 1]) * sc[0];
      } else if (codec == KVCodec::kFP8PerTokenHead) {
        lo = e4m3_to_float(static_cast<const uint8_t*>(data)[base + 2 * u]) * meta[row];
        hi = e4m3_to_float(static_cast<const uint8_t*>(data)[base + 2 * u + 1]) * meta[row];
      } else if (codec == KVCodec::kInt8PerTokenHead) {
        lo = static_cast<float>(static_cast<const int8_t*>(data)[base + 2 * u]) * meta[row];
        hi = static_cast<float>(static_cast<const int8_t*>(data)[base + 2 * u + 1]) * meta[row];
      } else {
        const uint8_t byte = static_cast<const uint8_t*>(data)[base + u];
        const float sc = meta[2 * row], zero = meta[2 * row + 1];
        lo = __fmul_rn(static_cast<float>(byte & 15u) - zero, sc);
        hi = __fmul_rn(static_cast<float>(byte >> 4) - zero, sc);
      }
    }
    x[2 * i] = lo;
    x[2 * i + 1] = hi;
  }
}

/// Store one decoded row into a cache by its destination codec.
template <int P>
__device__ __forceinline__ void store_decoded_row(const PagedKVDesc& d, bool is_value,
                                                  int64_t slot, int head, int lane,
                                                  const float (&x)[2 * P], float eps) {
  const KVCodec codec = is_value ? d.v_codec : d.k_codec;
  const int dim = is_value ? d.head_dim_v : d.head_dim_k;
  const int pairs = dim / 2;
  void* out = is_value ? d.v_data : d.k_data;
  const int64_t base = cache_offset(d, slot, head, is_value);
  if (token_codec(codec)) {
    uint8_t* dst = static_cast<uint8_t*>(out) + base;
    float* meta = static_cast<float*>(is_value ? d.v_token_scales : d.k_token_scales);
    const int64_t row = slot * d.num_kv_heads + head;
    quantize_token_row_as<P>(codec, x, pairs, lane, dst,
                             meta + (codec == KVCodec::kInt4PerTokenHead ? 2 * row : row));
    return;
  }
  const DType dtype = is_value ? d.v_dtype : d.k_dtype;
  const float* scales = is_value ? d.v_scale : d.k_scale;
  const float sc = (codec == KVCodec::kFP8PerTensor || codec == KVCodec::kFP8PerHead)
                       ? scales[codec == KVCodec::kFP8PerHead ? head : 0]
                       : 1.0f;
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int u = lane + 32 * i;
    if (u >= pairs) continue;
    for (int j = 0; j < 2; ++j) {
      const float v = x[2 * i + j];
      const int64_t idx = base + 2 * u + j;
      if (codec == KVCodec::kFP8PerTensor || codec == KVCodec::kFP8PerHead) {
        static_cast<uint8_t*>(out)[idx] = e4m3_from_float(__fdiv_rn(v, sc));
      } else if (dtype == DType::kFloat32) {
        static_cast<float*>(out)[idx] = v;
      } else if (dtype == DType::kFloat16) {
        static_cast<__half*>(out)[idx] = __float2half_rn(v);
      } else {
        static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
      }
    }
  }
  (void)eps;
}

__device__ __forceinline__ void rotate_pair_v(float a, float b, float c, float s, float& oa,
                                              float& ob) {
  oa = __fmaf_rn(a, c, -b * s);
  ob = __fmaf_rn(b, c, a * s);
}

template <bool kNeoX>
__device__ __forceinline__ void rope_pair_offsets(int i, int half, int& a, int& b) {
  if constexpr (kNeoX) {
    a = i;
    b = i + half;
  } else {
    a = 2 * i;
    b = 2 * i + 1;
  }
}

}  // namespace

// The kernels below are declared in kvcache_kernels.cuh; defining them at
// namespace scope keeps the definitions tied to those declarations.
using dev::from_f;
using dev::to_f;

// =============================================================================
// K01  standard K/V writes
// =============================================================================

template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache, cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int64_t block_stride,
    const int64_t page_stride, const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int head_size_v, const int block_size, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const int kv_scale_stride, const int64_t num_tokens) {
  constexpr int kWarps = 8;   // one warp per token
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  const int64_t page = slot / block_size;
  const int64_t off = slot % block_size;
  const int64_t k_base = page * block_stride + off * page_stride;
  const bool hnd = layout_is_hnd(page_stride, head_stride, head_size);
  const int64_t v_head_stride = hnd ? static_cast<int64_t>(block_size) * head_size_v
                                    : static_cast<int64_t>(head_size_v);
  const int64_t v_page_stride = hnd ? static_cast<int64_t>(head_size_v)
                                    : static_cast<int64_t>(num_heads) * head_size_v;
  // The V tensor has its own block stride when head_size_v != head_size.
  const int64_t v_block_stride = static_cast<int64_t>(block_size) * num_heads * head_size_v;
  const int64_t v_base = page * v_block_stride + off * v_page_stride;

  // 32-byte packs for a native 16-bit cache (two uint4), 16 bytes otherwise.
  constexpr int kVecElems = (kCodec == KVCodec::kNative && sizeof(scalar_t) == 2)
                                ? 16
                                : static_cast<int>(16 / sizeof(scalar_t));
  const int kvec = (head_size % kVecElems == 0) ? kVecElems : 1;
  const int vvec = (head_size_v % kVecElems == 0) ? kVecElems : 1;
  const int k_packs = head_size / kvec;
  const int v_packs = head_size_v / vvec;

  for (int p = lane; p < num_heads * k_packs; p += 32) {
    const int h = p / k_packs;
    const int d = (p % k_packs) * kvec;
    const float scale = head_scale(k_scale, h, kv_scale_stride);
    const scalar_t* src = key + token * key_stride +
                          static_cast<int64_t>(h) * head_size + d;
    cache_t* dst = key_cache + k_base + static_cast<int64_t>(h) * head_stride + d;
    if constexpr (kCodec == KVCodec::kNative) {
      if (kvec == 16 && (reinterpret_cast<uintptr_t>(src) & 31u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 31u) == 0) {
        const uint4 a = *reinterpret_cast<const uint4*>(src), b = *reinterpret_cast<const uint4*>(src + 8);
        *reinterpret_cast<uint4*>(dst) = a;
        *reinterpret_cast<uint4*>(dst + 8) = b;
        continue;
      }
      if (kvec == 8 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        continue;
      }
    }
    for (int j = 0; j < kvec; ++j) dst[j] = cache_store<cache_t, kCodec>(to_f(src[j]), scale);
  }
  for (int p = lane; p < num_heads * v_packs; p += 32) {
    const int h = p / v_packs;
    const int d = (p % v_packs) * vvec;
    const float scale = head_scale(v_scale, h, kv_scale_stride);
    const scalar_t* src = value + token * value_stride +
                          static_cast<int64_t>(h) * head_size_v + d;
    cache_t* dst = value_cache + v_base + static_cast<int64_t>(h) * v_head_stride + d;
    if constexpr (kCodec == KVCodec::kNative) {
      if (vvec == 16 && (reinterpret_cast<uintptr_t>(src) & 31u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 31u) == 0) {
        const uint4 a = *reinterpret_cast<const uint4*>(src), b = *reinterpret_cast<const uint4*>(src + 8);
        *reinterpret_cast<uint4*>(dst) = a;
        *reinterpret_cast<uint4*>(dst + 8) = b;
        continue;
      }
      if (vvec == 8 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        continue;
      }
    }
    for (int j = 0; j < vvec; ++j) dst[j] = cache_store<cache_t, kCodec>(to_f(src[j]), scale);
  }
}

template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void reshape_and_cache_legacy_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache, cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size, const int x,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale) {
  const int64_t slot = slot_mapping[blockIdx.x];
  if (slot < 0) return;
  const int64_t page = slot / block_size;
  const int64_t off = slot % block_size;
  const int64_t page_elems = static_cast<int64_t>(num_heads) * head_size * block_size;
  const int64_t key_row = static_cast<int64_t>(blockIdx.x) * key_stride;
  const int64_t value_row = static_cast<int64_t>(blockIdx.x) * value_stride;
  const float ks = k_scale != nullptr ? k_scale[0] : 1.0f;
  const float vs = v_scale != nullptr ? v_scale[0] : 1.0f;

  const int chunks_per_head = head_size / x;
  for (int idx = threadIdx.x; idx < num_heads * chunks_per_head; idx += blockDim.x) {
    const int h = idx / chunks_per_head;
    const int chunk = idx % chunks_per_head;
    for (int j = 0; j < x; ++j) {
      const int d = chunk * x + j;
      const scalar_t v = key[key_row + static_cast<int64_t>(h) * head_size + d];
      const int64_t dst = page * page_elems + static_cast<int64_t>(h) * head_size * block_size +
                          static_cast<int64_t>(chunk) * block_size * x +
                          static_cast<int64_t>(off) * x + j;
      key_cache[dst] = cache_store<cache_t, kCodec>(to_f(v), ks);
    }
  }
  for (int idx = threadIdx.x; idx < num_heads * head_size; idx += blockDim.x) {
    const int h = idx / head_size;
    const int d = idx % head_size;
    const scalar_t v = value[value_row + static_cast<int64_t>(h) * head_size + d];
    const int64_t dst = page * page_elems + static_cast<int64_t>(h) * head_size * block_size +
                        static_cast<int64_t>(d) * block_size + off;
    value_cache[dst] = cache_store<cache_t, kCodec>(to_f(v), vs);
  }
}

/// Warps per token for append_paged_kv, shared by the launcher and the kernel so both
/// agree on the thread -> (token, chunk) mapping.
///
/// `chunks` is the number of 16-byte K+V chunks of one token (K chunks first, then V).
/// Two chunks per lane is the shape that reaches the two-stream copy floor on the dev GPU:
/// with 32 threads per token each lane owns four dependent chunk copies and the DRAM stream
/// stays latency bound (measured 13.2 us vs 11.3 us for the same bytes on a 4090), while one
/// chunk per lane leaves too few loads in flight per thread. The result is a power-of-two
/// lanes-per-token count so the kernel can split threadIdx.x with a shift instead of an
/// integer division (~1 us of the append case).
///
/// Returns log2(threads per token / 32), i.e. threads per token = 32 << result.
__host__ __device__ __forceinline__ int append_token_shift(int chunks, int block) {
  int k = 0;
  while (k < 3 && chunks > 2 * (32 << k)) ++k;
  while (k > 0 && (32 << k) > block) --k;
  return k;
}

/// Warps per token implied by append_token_shift.
__host__ __device__ __forceinline__ int append_threads_per_token(int chunks, int block) {
  return 32 << append_token_shift(chunks, block);
}

/// One 16-byte chunk of K/V: a raw copy for a native cache, or VEC values packed into one 16/8/4-byte
/// store for the FP8 codecs. `dst` and `src` are aligned because d is a multiple of VEC.
template <typename scalar_t, typename cache_t, KVCodec kCodec, int VEC>
__device__ __forceinline__ void append_vec(cache_t* dst, const scalar_t* src, float scale) {
  if constexpr (kCodec == KVCodec::kNative) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  } else {
    uint8_t q[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) q[j] = static_cast<uint8_t>(cache_store<cache_t, kCodec>(to_f(src[j]), scale));
    if constexpr (VEC == 8) {
      *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(q);
    } else if constexpr (VEC == 4) {
      *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(q);
    } else {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) dst[j] = static_cast<cache_t>(q[j]);
    }
  }
}

/// A group of `threads_per_token` lanes owns one token and walks K and V chunks in one index space
/// (the first `k_chunks` chunks are K, the rest V), so K and V are issued together and
/// page/offset are computed once.
///
/// The native fast path issues its source loads *before* the CSR page-table chain
/// (batch_indices -> kv_indptr -> kv_page_indices) resolves, so the three dependent metadata
/// loads overlap the data fetch instead of preceding it, and it uses the chunk index itself as the
/// row offset when the heads are dense (no integer division on the load path). Both are worth
/// ~1 us of this 12 us case on the dev GPU. The generic path keeps the codec/alignment fallbacks:
/// FP8 packing, per-head scales, non 16-byte-aligned strides and head dims, HND rows, and more
/// than two chunks per lane.
template <typename scalar_t, typename cache_t, KVCodec kCodec, KVLayout kLayout>
__global__ void append_paged_kv_kernel(
    const PagedKVDesc desc, const scalar_t* __restrict__ append_key,
    const scalar_t* __restrict__ append_value, const int32_t* __restrict__ batch_indices,
    const int32_t* __restrict__ positions, const int64_t key_stride_n,
    const int64_t key_stride_h, const int64_t value_stride_n, const int64_t value_stride_h,
    const int nnz) {
  constexpr int VEC = 16 / static_cast<int>(sizeof(scalar_t));
  const bool native_vec = kCodec == KVCodec::kNative && (sizeof(scalar_t) == 2 || sizeof(scalar_t) == 4);
  const int block = static_cast<int>(blockDim.x);
  const int heads = desc.num_kv_heads;
  const int dk = desc.head_dim_k;
  const int dv = desc.head_dim_v;
  cache_t* kc = static_cast<cache_t*>(desc.k_data);
  cache_t* vc = static_cast<cache_t*>(desc.v_data);
  const bool v_live = append_value != nullptr && desc.v_data != nullptr && dv > 0;
  const bool vec_k = native_vec && dk > 0 && (dk % VEC) == 0;
  const bool vec_v = vec_k && v_live && (dv % VEC) == 0;
  const int nvec_k = vec_k ? dk / VEC : 0;
  const int k_chunks = vec_k ? heads * nvec_k : 0;
  const int v_chunks = vec_v ? heads * (dv / VEC) : 0;

  const int chunks = k_chunks + v_chunks;
  const int shift = append_token_shift(chunks, block);
  const int tpt = 32 << shift;                  // threads per token
  const int tpc = block >> (5 + shift);         // tokens per CTA
  const int64_t token = static_cast<int64_t>(blockIdx.x) * tpc + (threadIdx.x >> (5 + shift));
  if (token >= nnz) return;
  const int slot = static_cast<int>(threadIdx.x) & (tpt - 1);

  const scalar_t* ksrc = append_key + token * key_stride_n;
  const scalar_t* vsrc = append_value != nullptr ? append_value + token * value_stride_n : nullptr;

  // Fast path: a raw 16-byte copy for both components, at most two chunks per lane. Every stride
  // and base pointer must be 16-byte aligned so the per-token offset stays aligned too.
  const bool aligned =
      (reinterpret_cast<uintptr_t>(append_key) & 15u) == 0u &&
      (reinterpret_cast<uintptr_t>(kc) & 15u) == 0u &&
      (key_stride_n % VEC) == 0 && (key_stride_h % VEC) == 0 &&
      (desc.k_stride_page % VEC) == 0 && (desc.k_stride_token % VEC) == 0 &&
      (desc.k_stride_head % VEC) == 0 &&
      (!v_live || ((reinterpret_cast<uintptr_t>(append_value) & 15u) == 0u &&
                   (reinterpret_cast<uintptr_t>(vc) & 15u) == 0u &&
                   (value_stride_n % VEC) == 0 && (value_stride_h % VEC) == 0 &&
                   (desc.v_stride_page % VEC) == 0 && (desc.v_stride_token % VEC) == 0 &&
                   (desc.v_stride_head % VEC) == 0));
  // Dense heads (NHD with contiguous source rows) make the chunk index itself the row offset, so
  // the copy needs no integer division and no per-chunk index math: exactly two 16-byte loads and
  // two stores per lane. vec_k also rejects head_dim_k 0 / head dims that are not a multiple of
  // the 16-byte chunk, which keep the generic path (it alone can write V when K has no chunks).
  const bool flat = desc.k_stride_head == dk && key_stride_h == dk &&
                    (!v_live || (desc.v_stride_head == dv && value_stride_h == dv));
  const bool store_path = vec_k && flat && aligned && chunks <= tpt * 2;

  // Resolve the CSR page id of the logical page that holds `positions[token]` into the K/V
  // element bases. Kept separate so the fast path can issue its loads before it runs.
  const int32_t b = batch_indices[token];
  const int32_t pos = positions[token];
  auto resolve = [&](int64_t* k_base, int64_t* v_base) {
    if (b < 0 || pos < 0 || desc.kv_indptr == nullptr || desc.kv_page_indices == nullptr) return false;
    const int64_t first = desc.kv_indptr[b];
    const int64_t last = desc.kv_indptr[b + 1];
    const int64_t logical = pos / desc.page_size;
    if (logical < 0 || first + logical >= last) return false;
    const int32_t page = desc.kv_page_indices[first + logical];
    if (page < 0) return false;
    const int64_t off = pos % desc.page_size;
    *k_base = static_cast<int64_t>(page) * desc.k_stride_page + off * desc.k_stride_token;
    *v_base = static_cast<int64_t>(page) * desc.v_stride_page + off * desc.v_stride_token;
    return true;
  };

  // 1. Fast path: two 16-byte chunks per lane, raw copy, page table resolved last.
  if (store_path) {
    uint4 r0, r1;
    int64_t row_off0 = 0, row_off1 = 0;
    bool is_v0 = false, is_v1 = false;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = slot + tpt * i;
      if (c >= chunks) continue;
      const bool is_v = c >= k_chunks;
      const int64_t row_off = static_cast<int64_t>(c - (is_v ? k_chunks : 0)) * VEC;
      const uint4 v = *reinterpret_cast<const uint4*>(is_v ? (vsrc + row_off) : (ksrc + row_off));
      if (i == 0) {
        r0 = v;
        row_off0 = row_off;
        is_v0 = is_v;
      } else {
        r1 = v;
        row_off1 = row_off;
        is_v1 = is_v;
      }
    }
    int64_t k_base = 0, v_base = 0;
    if (!resolve(&k_base, &v_base)) return;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (slot + tpt * i >= chunks) continue;
      const bool is_v = i == 0 ? is_v0 : is_v1;
      cache_t* dst = (is_v ? vc + v_base : kc + k_base) + (i == 0 ? row_off0 : row_off1);
      *reinterpret_cast<uint4*>(dst) = (i == 0 ? r0 : r1);
    }
    return;
  }

  // 2. Generic path: any codec, scale layout, alignment and head dim.
  int64_t k_base = 0, v_base = 0;
  if (!resolve(&k_base, &v_base)) return;

  auto stream = [&](const scalar_t* src, cache_t* dst_cache, int64_t base, int64_t src_head_stride,
                    int64_t dst_head_stride, int dim, const float* scale_ptr) {
    const bool vec_ok = (reinterpret_cast<uintptr_t>(src) & 15u) == 0 && src_head_stride % VEC == 0 &&
                        (reinterpret_cast<uintptr_t>(dst_cache) & 15u) == 0 && dst_head_stride % VEC == 0 &&
                        base % VEC == 0;
    if (!vec_ok) {
      for (int i = slot; i < heads * dim; i += tpt) {
        const int h = i / dim;
        const int d = i - h * dim;
        const float s = head_scale(scale_ptr, h, desc.scale_head_stride);
        dst_cache[base + static_cast<int64_t>(h) * dst_head_stride + d] =
            cache_store<cache_t, kCodec>(to_f(src[static_cast<int64_t>(h) * src_head_stride + d]), s);
      }
      return;
    }
    const int nvec = dim / VEC;
    const int tail = dim - nvec * VEC;
    for (int i = slot; i < heads * nvec; i += tpt) {
      const int h = i / nvec;
      const int d = (i - h * nvec) * VEC;
      const float s = head_scale(scale_ptr, h, desc.scale_head_stride);
      append_vec<scalar_t, cache_t, kCodec, VEC>(dst_cache + base + static_cast<int64_t>(h) * dst_head_stride + d,
                                                 src + static_cast<int64_t>(h) * src_head_stride + d, s);
    }
    if (tail != 0) {
      for (int i = slot; i < heads * tail; i += tpt) {
        const int h = i / tail;
        const int d = nvec * VEC + (i - h * tail);
        const float s = head_scale(scale_ptr, h, desc.scale_head_stride);
        dst_cache[base + static_cast<int64_t>(h) * dst_head_stride + d] =
            cache_store<cache_t, kCodec>(to_f(src[static_cast<int64_t>(h) * src_head_stride + d]), s);
      }
    }
  };
  stream(ksrc, kc, k_base, key_stride_h, desc.k_stride_head, dk, desc.k_scale);
  if (vc != nullptr && desc.v_data != nullptr && vsrc != nullptr) {
    stream(vsrc, vc, v_base, value_stride_h, desc.v_stride_head, dv, desc.v_scale);
  }
  (void)kLayout;
}

template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void ring_kv_append_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    cache_t* __restrict__ ring_k, cache_t* __restrict__ ring_v,
    const int32_t* __restrict__ token_slot, const int64_t* __restrict__ positions,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, const int window,
    const int num_kv_heads, const int head_dim_k, const int head_dim_v, const int num_tokens) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= num_tokens) return;
  const int32_t slot = token_slot[token];
  if (slot < 0) return;
  const int64_t my_pos = positions[token];
  const int64_t ring_pos = my_pos % window;
  // Two tokens of one sequence can target the same ring cell when the batch
  // spans at least one window. Only the newest position may win; ties go to the
  // later token index. One lane performs the bounded scan and broadcasts the
  // decision, so the whole CTA does not issue the same loads.
  __shared__ int superseded;
  if (threadIdx.x < 32) {
    int mine = 0;
    if (num_tokens > window) {
      const int j_end = token + window + 1 < num_tokens ? token + window + 1 : num_tokens;
      for (int j = token + 1 + static_cast<int>(threadIdx.x); j < j_end; j += 32) {
        if (token_slot[j] == slot && positions[j] >= my_pos &&
            (positions[j] % window) == ring_pos) {
          mine = 1;
          break;
        }
      }
    }
    const unsigned any = __any_sync(0xffffffffu, mine);
    if (threadIdx.x == 0) superseded = any != 0u ? 1 : 0;
  }
  __syncthreads();
  if (superseded) return;
  const int64_t base = (static_cast<int64_t>(slot) * window + ring_pos) * num_kv_heads;
  const float ks = k_scale != nullptr ? k_scale[0] : 1.0f;
  const float vs = v_scale != nullptr ? v_scale[0] : 1.0f;
  const scalar_t* ksrc = key + static_cast<int64_t>(token) * num_kv_heads * head_dim_k;
  const scalar_t* vsrc = value != nullptr
                             ? value + static_cast<int64_t>(token) * num_kv_heads * head_dim_v
                             : nullptr;
  for (int idx = threadIdx.x; idx < num_kv_heads * head_dim_k; idx += blockDim.x) {
    const int h = idx / head_dim_k;
    const int d = idx % head_dim_k;
    ring_k[(base + h) * head_dim_k + d] =
        cache_store<cache_t, kCodec>(to_f(ksrc[h * head_dim_k + d]), ks);
  }
  if (ring_v != nullptr && vsrc != nullptr) {
    for (int idx = threadIdx.x; idx < num_kv_heads * head_dim_v; idx += blockDim.x) {
      const int h = idx / head_dim_v;
      const int d = idx % head_dim_v;
      ring_v[(base + h) * head_dim_v + d] =
          cache_store<cache_t, kCodec>(to_f(vsrc[h * head_dim_v + d]), vs);
    }
  }
}

// =============================================================================
// K01  MLA latent writes
// =============================================================================

template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c, const scalar_t* __restrict__ k_pe,
    cache_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const int block_stride, const int entry_stride, const int kv_c_stride, const int k_pe_stride,
    const int kv_lora_rank, const int pe_dim, const int block_size,
    const float* __restrict__ scale) {
  const int64_t slot = slot_mapping[blockIdx.x];
  if (slot < 0) return;
  const int64_t page = slot / block_size;
  const int64_t off = slot % block_size;
  cache_t* dst = kv_cache + page * block_stride + off * entry_stride;
  const float s = scale != nullptr ? scale[0] : 1.0f;
  const scalar_t* csrc = kv_c + static_cast<int64_t>(blockIdx.x) * kv_c_stride;
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    dst[i] = cache_store<cache_t, kCodec>(to_f(csrc[i]), s);
  }
  if (k_pe != nullptr && pe_dim > 0) {
    const scalar_t* psrc = k_pe + static_cast<int64_t>(blockIdx.x) * k_pe_stride;
    for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
      dst[kv_lora_rank + i] = cache_store<cache_t, kCodec>(to_f(psrc[i]), s);
    }
  } else {
    for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
      dst[kv_lora_rank + i] = cache_store<cache_t, kCodec>(0.0f, s);
    }
  }
}

template <typename scalar_t, typename cache_t, KVCodec kCodec>
__global__ void concat_and_cache_mla_grouped_kernel(
    const scalar_t* __restrict__ kv_c, const scalar_t* __restrict__ k_pe,
    const int64_t* __restrict__ kv_cache_ptrs, const float* __restrict__ kv_scales,
    const int64_t* __restrict__ slot_mapping, const int64_t kv_c_layer_stride,
    const int64_t kv_c_token_stride, const int64_t k_pe_layer_stride,
    const int64_t k_pe_token_stride, const int64_t slot_layer_stride,
    const int64_t block_stride, const int64_t entry_stride, const int kv_lora_rank,
    const int pe_dim, const int block_size) {
  const int64_t token = blockIdx.x;
  const int layer = blockIdx.y;
  cache_t* base = reinterpret_cast<cache_t*>(static_cast<uintptr_t>(kv_cache_ptrs[layer]));
  if (base == nullptr) return;
  const int64_t slot = slot_mapping[layer * slot_layer_stride + token];
  if (slot < 0) return;
  const int64_t page = slot / block_size;
  const int64_t off = slot % block_size;
  cache_t* dst = base + page * block_stride + off * entry_stride;
  const float s = kv_scales != nullptr ? kv_scales[layer] : 1.0f;
  const scalar_t* csrc = kv_c + static_cast<int64_t>(layer) * kv_c_layer_stride +
                         token * kv_c_token_stride;
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    dst[i] = cache_store<cache_t, kCodec>(to_f(csrc[i]), s);
  }
  if (k_pe != nullptr && pe_dim > 0) {
    const scalar_t* psrc = k_pe + static_cast<int64_t>(layer) * k_pe_layer_stride +
                           token * k_pe_token_stride;
    for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
      dst[kv_lora_rank + i] = cache_store<cache_t, kCodec>(to_f(psrc[i]), s);
    }
  }
}

// =============================================================================
// K01/B3  per-token-head cache writes and requantization
// =============================================================================

template <typename scalar_t, typename cache_t, KVCodec kCodec, KVLayout kLayout>
__global__ void reshape_and_cache_per_token_quant_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    const PagedKVDesc cache, const int64_t* __restrict__ slot_mapping,
    const int64_t key_stride, const int64_t value_stride, const int num_tokens) {
  constexpr int P = 8;
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int pairs_k = cache.head_dim_k / 2;
  const int pairs_v = cache.head_dim_v / 2;
  for (int component = 0; component < 2; ++component) {
    const bool is_value = component == 1;
    const KVCodec codec = is_value ? cache.v_codec : cache.k_codec;
    if (!token_codec(codec)) continue;
    const int dim = is_value ? cache.head_dim_v : cache.head_dim_k;
    if (dim <= 0 || dim > kMaxRowDim) continue;
    for (int h = threadIdx.x >> 5; h < cache.num_kv_heads; h += warps) {
      const scalar_t* src = is_value ? value + token * value_stride + h * cache.head_dim_v
                                     : key + token * key_stride + h * cache.head_dim_k;
      float x[2 * P];
      if (is_value) {
        load_row_tokens<P>(src, pairs_v, lane, x);
      } else {
        load_row_tokens<P>(src, pairs_k, lane, x);
      }
      cache_t* dst = static_cast<cache_t*>(is_value ? cache.v_data : cache.k_data) +
                     cache_offset(cache, slot, h, is_value);
      float* meta = static_cast<float*>(is_value ? cache.v_token_scales : cache.k_token_scales);
      const int64_t row = slot * cache.num_kv_heads + h;
      quantize_token_row_as<P>(codec, x, is_value ? pairs_v : pairs_k, lane,
                               reinterpret_cast<uint8_t*>(dst),
                               meta + (codec == KVCodec::kInt4PerTokenHead ? 2 * row : row));
    }
  }
}

template <typename src_t, typename dst_t, KVCodec kSrcCodec, KVCodec kDstCodec>
__global__ void kv_requantize_pages_kernel(const PagedKVDesc src_desc,
                                           const PagedKVDesc dst_desc,
                                           const int64_t* __restrict__ page_pairs,
                                           const int num_pairs) {
  constexpr int P = 8;
  const int64_t pair = blockIdx.x;
  if (pair >= num_pairs) return;
  const int64_t from = page_pairs[2 * pair];
  const int64_t to = page_pairs[2 * pair + 1];
  if (from < 0 || to < 0) return;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int per_component = src_desc.page_size * src_desc.num_kv_heads;
  for (int r = blockIdx.y * 32 + (threadIdx.x >> 5); r < 2 * per_component;
       r += warps * static_cast<int>(gridDim.y)) {
    const bool is_value = r >= per_component;
    const int rr = is_value ? r - per_component : r;
    const int token = rr / src_desc.num_kv_heads;
    const int h = rr - token * src_desc.num_kv_heads;
    const int64_t src_slot = from * src_desc.page_size + token;
    const int64_t dst_slot = to * dst_desc.page_size + token;
    float x[2 * P];
    load_decoded_row<P>(src_desc, is_value, src_slot, h, lane, x);
    store_decoded_row<P>(dst_desc, is_value, dst_slot, h, lane, x, 1e-10f);
  }
}

template <typename scalar_t>
__global__ void kv_scale_update_kernel(float* __restrict__ k_absmax,
                                       float* __restrict__ v_absmax,
                                       const scalar_t* __restrict__ key,
                                       const scalar_t* __restrict__ value, const int num_tokens,
                                       const int num_kv_heads, const int head_dim_k,
                                       const int head_dim_v) {
  const bool is_value = blockIdx.x >= num_kv_heads;
  const int h = is_value ? blockIdx.x - num_kv_heads : blockIdx.x;
  const scalar_t* src = is_value ? value : key;
  const int dim = is_value ? head_dim_v : head_dim_k;
  const int64_t stride = static_cast<int64_t>(num_kv_heads) * dim;
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  uint32_t keymax = 0;
  for (int64_t t = static_cast<int64_t>(blockIdx.y) * warps + (threadIdx.x >> 5);
       t < num_tokens; t += static_cast<int64_t>(gridDim.y) * warps) {
    const scalar_t* row = src + t * stride + static_cast<int64_t>(h) * dim;
    for (int i = lane; i < dim; i += 32) {
      const uint32_t k = __float_as_uint(fabsf(to_f<scalar_t>(row[i])));
      keymax = (k > keymax) ? k : keymax;
    }
  }
  keymax = warp_reduce_max(keymax);
  __shared__ uint32_t partial[32];
  if (lane == 0) partial[threadIdx.x >> 5] = keymax;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t m = 0;
    for (int w = 0; w < warps; ++w) m = (partial[w] > m) ? partial[w] : m;
    if (m != 0u) atomicMax(reinterpret_cast<unsigned*>(is_value ? v_absmax + h : k_absmax + h), m);
  }
}

// =============================================================================
// K01/B3  ds_mla latent records (656 / V4.0 584 / V4.1 528 / V4.1 FP4 288)
// =============================================================================

/// Decode a UE8M0 byte (2^(e-127)); 0xFF is NaN by spec.
__device__ __forceinline__ float ue8m0_to_scale(uint8_t e) {
  if (e == 0xffu) return __int_as_float(0x7fc00000);
  return ldexpf(1.0f, static_cast<int>(e) - 127);
}

/**
 * Power-of-two tile scale for the FP8 records: 2^ceil(log2(max(amax / 448, 1e-4))) as UE8M0
 * (FlashMLA _cast_scale_inv_to_ue8m0). A NaN in the tile keeps NaN (byte 0xFF) and an Inf
 * tile scale stays +Inf (also 0xFF), matching the reference's clamp/log2/pow chain.
 */
__device__ __forceinline__ float fp8_tile_scale(float amax, unsigned nan_flag, uint8_t* byte) {
  float x = __fdiv_rn(amax, kFp8E4M3Max);
  if (x < 1e-4f) x = 1e-4f;  // false for NaN, so NaN survives
  if (nan_flag != 0u) {
    *byte = 0xffu;
    return __int_as_float(0x7fc00000);
  }
  if (isinf(x)) {
    *byte = 0xffu;
    return x;
  }
  int e = 0;
  frexpf(x, &e);
  e -= 1;
  if (ldexpf(1.0f, e) < x) ++e;
  e = e < -127 ? -127 : (e > 127 ? 127 : e);
  *byte = static_cast<uint8_t>(e + 127);
  return ldexpf(1.0f, e);
}

/**
 * E4M3 block scale of one FP4 tile: clamp(amax / 6, 2^-9, 448) rounded to e4m3. A NaN or Inf
 * element poisons the tile (the reference nan_to_num + where(isinf(amax)) chain), so the scale
 * byte is the e4m3 NaN 0x7F and every code in the tile is zero.
 */
__device__ __forceinline__ float fp4_tile_scale(float amax, unsigned bad_flag, uint8_t* byte) {
  if (bad_flag != 0u) {
    *byte = 0x7fu;
    return __int_as_float(0x7fc00000);
  }
  float x = __fdiv_rn(amax, kFp4E2M1Max);
  x = x < 0x1p-9f ? 0x1p-9f : x;
  x = x > kFp8E4M3Max ? kFp8E4M3Max : x;
  *byte = e4m3_from_float(x);
  return e4m3_to_float(*byte);
}

/// fp32 -> e2m1 code with FlashMLA's semantics: nearest, ties to even code, saturate at +-6,
/// NaN -> zero magnitude (sign kept). `__nv_cvt_float_to_fp4` implements the finite cases.
__device__ __forceinline__ uint8_t fp4_from_float(float x, float scale) {
  const float q = (scale != 0.0f) ? __fdiv_rn(x, scale) : 0.0f;
  if (isnan(q)) return static_cast<uint8_t>((__float_as_uint(q) >> 28) & 0x8u);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp4(q, __NV_E2M1, cudaRoundNearest));
}

template <typename scalar_t, KVCodec kCodec>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c, const scalar_t* __restrict__ k_pe,
    uint8_t* __restrict__ kv_cache, uint8_t* __restrict__ scale_pages,
    const int64_t* __restrict__ slot_mapping, const int64_t block_stride,
    const int64_t entry_stride, const int64_t scale_block_stride,
    const int64_t scale_token_stride, const int kv_c_stride, const int k_pe_stride,
    const int kv_lora_rank, const int pe_dim, const int block_size) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  const int64_t page = slot / block_size;
  const int64_t off = slot - page * block_size;
  uint8_t* record = kv_cache + page * block_stride + off * entry_stride;
  uint8_t* scale_row = scale_pages != nullptr
                           ? scale_pages + page * scale_block_stride + off * scale_token_stride
                           : nullptr;
  if constexpr (kCodec == KVCodec::kMLAFp8DS) {
    if (threadIdx.x >= 64) {
      const int i = (threadIdx.x - 64) * 2;
      if (i + 1 < pe_dim && k_pe != nullptr) {
        const int64_t src = token * k_pe_stride + i;
        const __nv_bfloat16 a = __float2bfloat16_rn(to_f<scalar_t>(k_pe[src]));
        const __nv_bfloat16 b = __float2bfloat16_rn(to_f<scalar_t>(k_pe[src + 1]));
        uint16_t* dst = reinterpret_cast<uint16_t*>(record + kv_lora_rank + 16 + 2 * i);
        dst[0] = *reinterpret_cast<const uint16_t*>(&a);
        dst[1] = *reinterpret_cast<const uint16_t*>(&b);
      } else {
        *reinterpret_cast<uint32_t*>(record + kv_lora_rank + 16 + 2 * i) = 0u;
      }
      return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int tile = tid >> 4;
    const int lane16 = tid & 15;
    const int tile_count = kv_lora_rank / 128;
    if (tile >= tile_count) return;
    const scalar_t* src = kv_c + token * kv_c_stride + tile * 128 + lane16 * 8;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = to_f<scalar_t>(src[j]);
    float amax = fmaxf(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))),
                       fmaxf(fmaxf(fabsf(v[4]), fabsf(v[5])), fmaxf(fabsf(v[6]), fabsf(v[7]))));
    for (int offs = 8; offs > 0; offs >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offs, 16));
    }
    const float unrounded = fmaxf(__fdiv_rn(amax, kFp8E4M3Max), 1e-4f);
    const uint32_t bits = __float_as_uint(unrounded);
    const float tile_scale = __uint_as_float((bits + 0x7fffffu) & 0x7f800000u);
    if (lane16 == 0) {
      reinterpret_cast<float*>(record + kv_lora_rank)[tile] = tile_scale;
    }
    uint8_t codes[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) codes[j] = e4m3_from_float(__fdiv_rn(v[j], tile_scale));
    uint64_t packed;
    memcpy(&packed, codes, 8);
    *reinterpret_cast<uint64_t*>(record + tile * 128 + lane16 * 8) = packed;
    return;
  }
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  if constexpr (kCodec == KVCodec::kMLAFp8DSV40) {
    // 7 NoPE tiles of 64 (warps 0..6, two values per lane) + one bf16 RoPE warp.
    if (warp < 7) {
      const int base = warp * 64;
      const float v0 = to_f<scalar_t>(kv_c[token * kv_c_stride + base + lane]);
      const float v1 = to_f<scalar_t>(kv_c[token * kv_c_stride + base + lane + 32]);
      const float a0 = fabsf(v0);
      const float a1 = fabsf(v1);
      unsigned nan_flag = (isnan(a0) || isnan(a1)) ? 1u : 0u;
      float amax = fmaxf(isnan(a0) ? 0.0f : a0, isnan(a1) ? 0.0f : a1);
      amax = fmaxf(amax, (isinf(a0) || isinf(a1)) ? __int_as_float(0x7f800000) : 0.0f);
      amax = warp_reduce_max(amax);
      unsigned any_nan = nan_flag;
      for (int offs = 16; offs > 0; offs >>= 1) {
        any_nan |= __shfl_xor_sync(0xffffffffu, any_nan, offs);
      }
      uint8_t sb = 0;
      const float s = fp8_tile_scale(amax, any_nan, &sb);
      if (lane == 0) scale_row[warp] = sb;
      record[base + lane] = e4m3_from_float(__fdiv_rn(v0, s));
      record[base + lane + 32] = e4m3_from_float(__fdiv_rn(v1, s));
    } else if (warp == 7) {
      const int j = lane * 2;
      const float p0 = (k_pe != nullptr && j < pe_dim)
                           ? to_f<scalar_t>(k_pe[token * k_pe_stride + j])
                           : 0.0f;
      const float p1 = (k_pe != nullptr && j + 1 < pe_dim)
                           ? to_f<scalar_t>(k_pe[token * k_pe_stride + j + 1])
                           : 0.0f;
      const __nv_bfloat16 a = __float2bfloat16_rn(p0);
      const __nv_bfloat16 b = __float2bfloat16_rn(p1);
      uint16_t* dst = reinterpret_cast<uint16_t*>(record + 448 + 2 * j);
      dst[0] = *reinterpret_cast<const uint16_t*>(&a);
      dst[1] = *reinterpret_cast<const uint16_t*>(&b);
    }
    if (threadIdx.x == 0) scale_row[7] = 0u;  // unused pad byte of the 8 B scale record
    return;
  }
  if constexpr (kCodec == KVCodec::kMLAFp8DSV41) {
    // 16 tiles of 32 over the 512 logical dims (448 NoPE from kv_c, 64 RoPE from k_pe).
    for (int t = warp; t < 16; t += warps) {
      const int dim = t * 32 + lane;
      float v = 0.0f;
      if (dim < 448) {
        v = to_f<scalar_t>(kv_c[token * kv_c_stride + dim]);
      } else if (k_pe != nullptr && dim - 448 < pe_dim) {
        v = to_f<scalar_t>(k_pe[token * k_pe_stride + dim - 448]);
      }
      const float av = fabsf(v);
      const unsigned nan_flag = isnan(av) ? 1u : 0u;
      float amax = (nan_flag != 0u) ? 0.0f : av;
      amax = fmaxf(amax, isinf(av) ? __int_as_float(0x7f800000) : 0.0f);
      amax = warp_reduce_max(amax);
      unsigned any_nan = nan_flag;
      for (int offs = 16; offs > 0; offs >>= 1) {
        any_nan |= __shfl_xor_sync(0xffffffffu, any_nan, offs);
      }
      uint8_t sb = 0;
      const float s = fp8_tile_scale(amax, any_nan, &sb);
      if (lane == 0) record[512 + t] = sb;
      record[dim] = e4m3_from_float(__fdiv_rn(v, s));
    }
    return;
  }
  // kMLAFp8DSV41Fp4: 32 tiles of 16 packed two-per-byte; one 8-lane group per tile.
  const int group = static_cast<int>(threadIdx.x) >> 3;
  const int sub = static_cast<int>(threadIdx.x) & 7;
  if (group >= 32) return;
  const int dim0 = group * 16 + sub * 2;
  float v0 = 0.0f, v1 = 0.0f;
  if (dim0 < 448) {
    v0 = to_f<scalar_t>(kv_c[token * kv_c_stride + dim0]);
    v1 = (dim0 + 1 < 448) ? to_f<scalar_t>(kv_c[token * kv_c_stride + dim0 + 1]) : 0.0f;
  } else if (k_pe != nullptr && dim0 - 448 < pe_dim) {
    v0 = to_f<scalar_t>(k_pe[token * k_pe_stride + dim0 - 448]);
    v1 = (dim0 + 1 - 448 < pe_dim) ? to_f<scalar_t>(k_pe[token * k_pe_stride + dim0 + 1 - 448])
                                   : 0.0f;
  }
  const float a0 = fabsf(v0);
  const float a1 = fabsf(v1);
  unsigned bad = (isnan(a0) || isnan(a1) || isinf(a0) || isinf(a1)) ? 1u : 0u;
  float amax = fmaxf(isnan(a0) ? 0.0f : a0, isnan(a1) ? 0.0f : a1);
  for (int offs = 4; offs > 0; offs >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offs, 8));
    bad |= __shfl_xor_sync(0xffffffffu, bad, offs, 8);
  }
  uint8_t sb = 0;
  const float s = fp4_tile_scale(amax, bad, &sb);
  if (sub == 0) record[256 + group] = sb;
  const uint8_t c0 = fp4_from_float(v0, s);
  const uint8_t c1 = fp4_from_float(v1, s);
  record[group * 8 + sub] = static_cast<uint8_t>(c0 | (c1 << 4));
}

// =============================================================================
// K01/B3  indexer split-page records
// =============================================================================

template <typename scalar_t>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k, uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, const int head_dim,
    const int quant_block_size, const int cache_block_size, const int64_t cache_block_stride,
    const bool use_ue8m0) {
  const int64_t token = blockIdx.x;
  const int lane = threadIdx.x;
  const int warp_in_row = threadIdx.y;
  const int head_dim_idx = (blockIdx.y * blockDim.y * 32 + warp_in_row * 32 + lane) * 4;
  const int64_t slot = slot_mapping[token];
  const bool live = slot >= 0 && head_dim_idx < head_dim;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (live) {
    const scalar_t* src = k + token * head_dim + head_dim_idx;
    if (head_dim_idx + 4 <= head_dim) {
      const dev::Pack<scalar_t, 4> r = dev::load_pack<4>(src);
      #pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = to_f<scalar_t>(r.x[j]);
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (head_dim_idx + j < head_dim) v[j] = to_f<scalar_t>(src[j]);
      }
    }
  }
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  const int lanes_per_block = quant_block_size / 4;
  for (int off = lanes_per_block / 2; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  if (!live) return;
  float scale = __fdiv_rn(fmaxf(amax, 1e-4f), kFp8E4M3Max);
  if (use_ue8m0) {
    int e = 0;
    frexpf(scale, &e);
    e -= 1;
    if (ldexpf(1.0f, e) < scale) ++e;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    scale = ldexpf(1.0f, e);
  }
  uint8_t* page = kv_cache + (slot / cache_block_size) * cache_block_stride;
  const int64_t offset_in_page = slot % cache_block_size;
  uint8_t codes[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) codes[j] = e4m3_from_float(__fdiv_rn(v[j], scale));
  uint32_t packed;
  memcpy(&packed, codes, 4);
  *reinterpret_cast<uint32_t*>(page + offset_in_page * head_dim + head_dim_idx) = packed;
  if ((lane % lanes_per_block) == 0) {
    const int groups = head_dim / quant_block_size;
    reinterpret_cast<float*>(page + static_cast<int64_t>(cache_block_size) * head_dim)
        [offset_in_page * groups + head_dim_idx / quant_block_size] = scale;
  }
}

// =============================================================================
// K01/B3  write fusions: QK-norm + RoPE + append, RoPE + MLA append
// =============================================================================

template <typename scalar_t, typename cache_t, int HEAD_DIM, KVCodec kCodec, KVLayout kLayout,
          bool kHasNorm>
__global__ void fused_qkv_norm_rope_append_kernel(
    const __grid_constant__ FusedQKVAppendParams params) {
  const int64_t token = blockIdx.x;
  if (token >= params.num_tokens) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int heads = params.num_q_heads + params.num_kv_heads;
  const int half = params.rot_dim / 2;
  const int64_t pos = params.positions != nullptr ? params.positions[token] : 0;
  const float* cos_sin = static_cast<const float*>(params.cos_sin_cache);
  const float* row_cs = cos_sin != nullptr
                            ? cos_sin + pos * static_cast<int64_t>(params.rot_dim)
                            : nullptr;
  scalar_t* qkv = static_cast<scalar_t*>(params.qkv) + token * params.qkv_token_stride;
  const int64_t slot =
      (params.slot_mapping != nullptr) ? params.slot_mapping[token] : -1;

  for (int task = warp; task < heads; task += warps) {
    const bool is_q = task < params.num_q_heads;
    const int h = is_q ? task : task - params.num_q_heads;
    scalar_t* row = qkv + static_cast<int64_t>(task) * HEAD_DIM;
    const scalar_t* weight = is_q ? static_cast<const scalar_t*>(params.q_norm_weight)
                                  : static_cast<const scalar_t*>(params.k_norm_weight);
    (void)weight;
    float x[HEAD_DIM / 32];
    float ss = 0.0f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; ++c) {
      x[c] = to_f<scalar_t>(row[lane + 32 * c]);
      ss = __fmaf_rn(x[c], x[c], ss);
    }
    ss = warp_reduce_sum(ss);
    const float rstd = rsqrtf(ss / static_cast<float>(HEAD_DIM) + params.norm_eps);
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; ++c) {
      const int j = lane + 32 * c;
      float v = x[c] * rstd;
      if constexpr (kHasNorm) {
        if (weight != nullptr) v *= to_f<scalar_t>(weight[j]) + params.norm_weight_offset;
      }
      from_f<scalar_t>(row[j], v);
    }
    __syncwarp();
    if (row_cs != nullptr) {
      const int lo = params.rot_offset;
      for (int i = lane; i < half; i += 32) {
        int a = 0, b = 0;
        if (params.rope_style == RopeStyle::kNeoX) {
          a = i;
          b = i + half;
        } else {
          a = 2 * i;
          b = 2 * i + 1;
        }
        const float c = row_cs[i];
        const float s = row_cs[half + i];
        float oa, ob;
        rotate_pair_v(to_f<scalar_t>(row[lo + a]), to_f<scalar_t>(row[lo + b]), c, s, oa, ob);
        from_f<scalar_t>(row[lo + a], oa);
        from_f<scalar_t>(row[lo + b], ob);
      }
      __syncwarp();
    }
    if (is_q && params.q_out != nullptr) {
      scalar_t* qout = static_cast<scalar_t*>(params.q_out) + token * params.q_out_token_stride +
                       static_cast<int64_t>(h) * HEAD_DIM;
      if (params.q_out_scale != nullptr) {
        uint8_t* q8 = reinterpret_cast<uint8_t*>(qout);
        const float s = params.q_out_scale[0];
        for (int j = lane; j < HEAD_DIM; j += 32) q8[j] = e4m3_from_float(__fdiv_rn(to_f<scalar_t>(row[j]), s));
      } else {
        for (int j = lane; j < HEAD_DIM; j += 32) qout[j] = row[j];
      }
    }
    if (!is_q && slot >= 0 && params.cache.k_data != nullptr) {
      const int64_t off = cache_offset(params.cache, slot, h, false);
      cache_t* dst = static_cast<cache_t*>(params.cache.k_data) + off;
      const float scale = (kCodec == KVCodec::kNative || params.cache.k_scale == nullptr)
                              ? 1.0f
                              : params.cache.k_scale[params.cache.scale_head_stride == 0 ? 0 : h];
      for (int j = lane; j < HEAD_DIM; j += 32) {
        dst[j] = cache_store<cache_t, kCodec>(to_f<scalar_t>(row[j]), scale);
      }
    }
    if (!is_q && slot >= 0 && params.cache.v_data != nullptr) {
      scalar_t* vrow = qkv + static_cast<int64_t>(params.num_q_heads + params.num_kv_heads + h) *
                                 HEAD_DIM;
      const int64_t off = cache_offset(params.cache, slot, h, true);
      cache_t* dst = static_cast<cache_t*>(params.cache.v_data) + off;
      const float scale = (kCodec == KVCodec::kNative || params.cache.v_scale == nullptr)
                              ? 1.0f
                              : params.cache.v_scale[params.cache.scale_head_stride == 0 ? 0 : h];
      for (int j = lane; j < HEAD_DIM; j += 32) {
        dst[j] = cache_store<cache_t, kCodec>(to_f<scalar_t>(vrow[j]), scale);
      }
    }
  }
}

template <typename qk_t, typename cos_sin_t, bool kNeoX, typename raw_kv_t, typename cache_t,
          KVCodec kCodec>
__global__ void concat_and_cache_mla_rope_fused_kernel(
    const int64_t* __restrict__ positions, qk_t* __restrict__ q_pe, qk_t* __restrict__ k_pe,
    const qk_t* __restrict__ kv_c, const cos_sin_t* __restrict__ rope_cos_sin_cache,
    const int rot_dim, const int64_t q_pe_stride_token, const int64_t q_pe_stride_head,
    const int64_t k_pe_stride, const int64_t kv_c_stride, const int num_q_heads,
    cache_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const int block_stride, const int entry_stride, const int kv_lora_rank, const int block_size,
    const float* __restrict__ kv_cache_quant_scale) {
  const int64_t token = blockIdx.x;
  const int half = rot_dim / 2;
  const int64_t pos = positions != nullptr ? positions[token] : 0;
  const cos_sin_t* row = rope_cos_sin_cache + pos * static_cast<int64_t>(rot_dim);
  for (int task = threadIdx.x; task < num_q_heads * half; task += blockDim.x) {
    const int h = task / half;
    const int i = task - h * half;
    qk_t* base = q_pe + token * q_pe_stride_token + static_cast<int64_t>(h) * q_pe_stride_head;
    int a = 0, b = 0;
    rope_pair_offsets<kNeoX>(i, half, a, b);
    float oa, ob;
    rotate_pair_v(to_f<qk_t>(base[a]), to_f<qk_t>(base[b]), to_f<cos_sin_t>(row[i]),
                  to_f<cos_sin_t>(row[half + i]), oa, ob);
    from_f<qk_t>(base[a], oa);
    from_f<qk_t>(base[b], ob);
  }
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    qk_t* base = k_pe + token * k_pe_stride;
    int a = 0, b = 0;
    rope_pair_offsets<kNeoX>(i, half, a, b);
    float oa, ob;
    rotate_pair_v(to_f<qk_t>(base[a]), to_f<qk_t>(base[b]), to_f<cos_sin_t>(row[i]),
                  to_f<cos_sin_t>(row[half + i]), oa, ob);
    from_f<qk_t>(base[a], oa);
    from_f<qk_t>(base[b], ob);
  }
  const int64_t slot = slot_mapping != nullptr ? slot_mapping[token] : -1;
  if (slot < 0) return;
  __syncthreads();
  cache_t* dst = kv_cache + (slot / block_size) * block_stride +
                 (slot % block_size) * entry_stride;
  const float scale = (kCodec == KVCodec::kNative || kv_cache_quant_scale == nullptr)
                          ? 1.0f
                          : kv_cache_quant_scale[0];
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    dst[i] = cache_store<cache_t, kCodec>(to_f<qk_t>(kv_c[token * kv_c_stride + i]), scale);
  }
  for (int i = threadIdx.x; i < rot_dim; i += blockDim.x) {
    dst[kv_lora_rank + i] = cache_store<cache_t, kCodec>(to_f<qk_t>(k_pe[token * k_pe_stride + i]),
                                                         scale);
  }
}

// =============================================================================
// K02  index-driven page gathers / scatters
// =============================================================================

__host__ __device__ __forceinline__ int entry_elems(const PagedKVDesc& d) {
  if (d.layout == KVLayout::kLatent || d.entry_bytes != 0) {
    return d.head_dim_k > 0 ? d.head_dim_k : 1;
  }
  const int heads = d.num_kv_heads > 0 ? d.num_kv_heads : 1;
  return heads * (d.head_dim_k > 0 ? d.head_dim_k : 1);
}

template <typename cache_t, typename out_t, KVCodec kCodec>
__global__ void gather_pages_kernel(out_t* __restrict__ out, const cache_t* __restrict__ cache,
                                    const int64_t* __restrict__ indices,
                                    const float* __restrict__ scale, const PagedKVDesc desc,
                                    const bool gather_by_token, const int num_items) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int item = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (item >= num_items) return;
  const int64_t index = indices[item];
  if (index < 0) return;
  const int elems = entry_elems(desc);
  const int64_t first_off = gather_by_token ? (index % desc.page_size) : 0;
  const int64_t page = gather_by_token ? (index / desc.page_size) : index;
  const int count = gather_by_token ? 1 : desc.page_size;
  const int64_t out_row = static_cast<int64_t>(item) * count * elems;
  const int heads = desc.num_kv_heads > 0 ? desc.num_kv_heads : 1;
  const bool latent = desc.layout == KVLayout::kLatent || desc.entry_bytes != 0;
  for (int t = 0; t < count; ++t) {
    const int64_t off = first_off + t;
    const int64_t token_base = page * desc.k_stride_page + off * desc.k_stride_token;
    out_t* dst = out + out_row + static_cast<int64_t>(t) * elems;
    for (int e = lane; e < elems; e += 32) {
      int64_t src_idx = 0;
      float s = scale != nullptr ? scale[0] : 1.0f;
      if (latent) {
        src_idx = token_base + e;
      } else {
        const int h = e / desc.head_dim_k;
        const int d = e % desc.head_dim_k;
        (void)heads;
        src_idx = token_base + static_cast<int64_t>(h) * desc.k_stride_head + d;
        s = head_scale(scale, h, desc.scale_head_stride);
      }
      const float v = cache_load<cache_t, kCodec>(cache[src_idx], s);
      out_t y;
      from_f<out_t>(y, v);
      dst[e] = y;
    }
  }
}

template <typename in_t, typename cache_t, KVCodec kCodec>
__global__ void scatter_pages_kernel(cache_t* __restrict__ cache,
                                     const in_t* __restrict__ src,
                                     const int64_t* __restrict__ indices,
                                     const float* __restrict__ scale, const PagedKVDesc desc,
                                     const bool scatter_by_token, const int num_items) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int item = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (item >= num_items) return;
  const int64_t index = indices[item];
  if (index < 0) return;
  const int elems = entry_elems(desc);
  const int64_t first_off = scatter_by_token ? (index % desc.page_size) : 0;
  const int64_t page = scatter_by_token ? (index / desc.page_size) : index;
  const int count = scatter_by_token ? 1 : desc.page_size;
  const int64_t src_row = static_cast<int64_t>(item) * count * elems;
  const bool latent = desc.layout == KVLayout::kLatent || desc.entry_bytes != 0;
  for (int t = 0; t < count; ++t) {
    const int64_t off = first_off + t;
    const int64_t token_base = page * desc.k_stride_page + off * desc.k_stride_token;
    const in_t* s = src + src_row + static_cast<int64_t>(t) * elems;
    for (int e = lane; e < elems; e += 32) {
      int64_t dst_idx = 0;
      float sc = scale != nullptr ? scale[0] : 1.0f;
      if (latent) {
        dst_idx = token_base + e;
      } else {
        const int h = e / desc.head_dim_k;
        const int d = e % desc.head_dim_k;
        dst_idx = token_base + static_cast<int64_t>(h) * desc.k_stride_head + d;
        sc = head_scale(scale, h, desc.scale_head_stride);
      }
      cache[dst_idx] = cache_store<cache_t, kCodec>(to_f<in_t>(s[e]), sc);
    }
  }
}

// =============================================================================
// K02  page movement
// =============================================================================

__global__ void copy_pages_kernel(void* const* __restrict__ layer_caches,
                                  const int64_t* __restrict__ page_pairs,
                                  const int64_t page_bytes, const int caches_per_layer,
                                  const int num_pairs, const int num_layers) {
  const int pair = static_cast<int>(blockIdx.x);
  const int layer = static_cast<int>(blockIdx.y);
  if (pair >= num_pairs || layer >= num_layers) return;
  const int64_t src_page = page_pairs[static_cast<int64_t>(pair) * 2];
  const int64_t dst_page = page_pairs[static_cast<int64_t>(pair) * 2 + 1];
  if (src_page < 0 || dst_page < 0) return;
  for (int c = 0; c < caches_per_layer; ++c) {
    void* const* base_ptrs = layer_caches + static_cast<int64_t>(layer) * caches_per_layer;
    const uint8_t* src = static_cast<const uint8_t*>(base_ptrs[c]) + src_page * page_bytes;
    uint8_t* dst = static_cast<uint8_t*>(base_ptrs[c]) + dst_page * page_bytes;
    copy_bytes(dst, src, page_bytes, static_cast<int>(threadIdx.x),
               static_cast<int>(blockDim.x));
  }
}

__global__ void zero_pages_kernel(void* const* __restrict__ layer_caches,
                                  const int64_t* __restrict__ pages, const int64_t page_bytes,
                                  const int caches_per_layer, const int num_pages,
                                  const int num_layers) {
  const int item = static_cast<int>(blockIdx.x);
  const int layer = static_cast<int>(blockIdx.y);
  if (item >= num_pages || layer >= num_layers) return;
  const int64_t page = pages[item];
  if (page < 0) return;
  for (int c = 0; c < caches_per_layer; ++c) {
    void* const* base_ptrs = layer_caches + static_cast<int64_t>(layer) * caches_per_layer;
    uint8_t* dst = static_cast<uint8_t*>(base_ptrs[c]) + page * page_bytes;
    zero_bytes(dst, page_bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
  }
}

// =============================================================================
// K02  ragged gathers out of the paged cache
// =============================================================================

__device__ __forceinline__ int find_request(const GatherSpec& spec, int token) {
  if (spec.cu_seq_lens == nullptr || spec.num_reqs <= 0) return -1;
  if (token >= spec.cu_seq_lens[spec.num_reqs]) return -1;
  int lo = 0;
  int hi = spec.num_reqs;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (spec.cu_seq_lens[mid + 1] <= token) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < spec.num_reqs ? lo : -1;
}

__device__ __forceinline__ bool resolve_gather_row(const GatherSpec& spec, int token, int block_size,
                                                   int64_t* src_row_page, int64_t* src_row_off) {
  const int r = find_request(spec, token);
  if (r < 0 || spec.block_table == nullptr) return false;
  const int64_t pos = (spec.seq_starts != nullptr ? spec.seq_starts[r] : 0) +
                      (token - spec.cu_seq_lens[r]);
  const int64_t logical = pos / block_size;
  if (logical < 0 || logical >= spec.block_table_stride) return false;
  const int32_t page = spec.block_table[static_cast<int64_t>(r) * spec.block_table_stride + logical];
  if (page < 0) return false;
  *src_row_page = page;
  *src_row_off = pos % block_size;
  return true;
}

template <bool kContiguousEntries, bool kVectorized>
__global__ void cp_gather_cache_kernel(const uint8_t* __restrict__ src_cache,
                                       uint8_t* __restrict__ dst, const GatherSpec spec,
                                       const int32_t block_size, const int32_t entry_size_bytes,
                                       const int64_t cache_block_stride,
                                       const int64_t cache_entry_stride,
                                       const int64_t dst_entry_stride) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (token >= spec.num_tokens) return;
  int64_t page = 0;
  int64_t page_off = 0;
  if (!resolve_gather_row(spec, token, block_size, &page, &page_off)) return;
  const uint8_t* src = src_cache + page * cache_block_stride + page_off * cache_entry_stride;
  uint8_t* out = dst + static_cast<int64_t>(token) * dst_entry_stride;
  if constexpr (kVectorized) {
    if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(out)) & 15u) == 0u &&
        (entry_size_bytes & 15) == 0) {
      const int words = entry_size_bytes >> 4;
      for (int i = lane; i < words; i += 32) {
        reinterpret_cast<uint4*>(out)[i] = reinterpret_cast<const uint4*>(src)[i];
      }
      return;
    }
  }
  for (int i = lane; i < entry_size_bytes; i += 32) out[i] = src[i];
  (void)kContiguousEntries;
}

// B3: MLA latent records -> bf16. One warp per token. The 656 B path (lane i dequantizes the
// 16 NoPE codes of tile i/8 and copies one RoPE word) is unchanged; the V4 records follow.
template <KVCodec kCodec>
__global__ void cp_gather_and_upconvert_ds_mla_fp8_kernel(
    const uint8_t* __restrict__ src_cache, bf16_t* __restrict__ dst, const GatherSpec spec,
    const int32_t block_size, const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride, const uint8_t* __restrict__ host_cache,
    const int32_t* __restrict__ host_row_ids, const int32_t* __restrict__ device_row_ids,
    const int64_t host_entry_stride, const uint8_t* __restrict__ scale_pages,
    const int64_t scale_block_stride, const int64_t scale_token_stride, const int dst_cols) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * warps + warp;
  if (token >= spec.num_tokens) return;
  int64_t page = 0, off = 0;
  if (!resolve_gather_row(spec, static_cast<int>(token), block_size, &page, &off)) return;
  const uint8_t* record = nullptr;
  if (device_row_ids == nullptr) {
    record = src_cache + page * cache_block_stride + off * cache_entry_stride;
  } else {
    const int64_t r = page * block_size + off;
    const int32_t device_row = device_row_ids[r];
    if (device_row >= 0) {
      record = src_cache + (device_row / block_size) * cache_block_stride +
               (device_row % block_size) * cache_entry_stride;
    } else {
      const int32_t host_row = host_row_ids[r];
      if (host_row >= 0) record = host_cache + static_cast<int64_t>(host_row) * host_entry_stride;
    }
  }
  if (record == nullptr) return;
  bf16_t* out = dst + token * dst_entry_stride;
  if constexpr (kCodec == KVCodec::kMLAFp8DS) {
    const uint2 lo = reinterpret_cast<const uint2*>(record)[2 * lane];
    const uint2 hi = reinterpret_cast<const uint2*>(record)[2 * lane + 1];
    const float scale = reinterpret_cast<const float*>(record + 512)[lane >> 3];
    const uint32_t words[4] = {lo.x, lo.y, hi.x, hi.y};
    uint32_t packed[8];
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      #pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint8_t code0 = (words[w] >> (8 * (2 * k))) & 0xff;
        const uint8_t code1 = (words[w] >> (8 * (2 * k + 1))) & 0xff;
        const float f0 = e4m3_to_float(code0) * scale;
        const float f1 = e4m3_to_float(code1) * scale;
        const __nv_bfloat162 pair = __floats2bfloat162_rn(f0, f1);
        packed[2 * w + k] = *reinterpret_cast<const uint32_t*>(&pair);
      }
    }
    uint4* nope = reinterpret_cast<uint4*>(out) + 2 * lane;
    nope[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    nope[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    reinterpret_cast<uint32_t*>(out + 512)[lane] = reinterpret_cast<const uint32_t*>(record + 528)[lane];
    return;
  }
  if constexpr (kCodec == KVCodec::kMLAFp8DSV40) {
    // Scale rows are addressed by the block-table page and offset, exactly like the write path.
    // They must stay device-resident when only the data rows are offloaded (host_cache carries
    // entry_bytes-wide data records, not the 8 B scale region).
    const uint8_t* scale_row = scale_pages + page * scale_block_stride + off * scale_token_stride;
    for (int t = 0; t < 7; ++t) {
      const float s = ue8m0_to_scale(scale_row[t]);
      for (int j = lane; j < 64; j += 32) {
        out[t * 64 + j] = __float2bfloat16_rn(e4m3_to_float(record[t * 64 + j]) * s);
      }
    }
    for (int j = lane; j < 64; j += 32) {
      out[448 + j] = *reinterpret_cast<const bf16_t*>(record + 448 + 2 * j);
    }
  } else if constexpr (kCodec == KVCodec::kMLAFp8DSV41) {
    for (int t = 0; t < 16; ++t) {
      const float s = ue8m0_to_scale(record[512 + t]);
      const int idx = t * 32 + lane;
      out[idx] = __float2bfloat16_rn(e4m3_to_float(record[idx]) * s);
    }
  } else {
    // FP4: 32 tiles of 16; four tiles per warp, 8 lanes per tile, even element low nibble.
    const int sub = lane & 7;
    for (int t = lane >> 3; t < 32; t += 4) {
      const uint8_t by = record[t * 8 + sub];
      const float s = e4m3_to_float(record[256 + t]);
      const float f0 = dev::e2m1_decode(by & 0x0fu) * s;
      const float f1 = dev::e2m1_decode(by >> 4) * s;
      const __nv_bfloat162 pair = __floats2bfloat162_rn(f0, f1);
      *reinterpret_cast<uint32_t*>(out + t * 16 + 2 * sub) =
          *reinterpret_cast<const uint32_t*>(&pair);
    }
  }
  if (dst_cols > 512) {
    for (int j = 512 + lane; j < dst_cols; j += 32) out[j] = __float2bfloat16_rn(0.0f);
  }
}

// B3: indexer split-page gather. Block (32, BLOCK_Y) with BLOCK_Y tokens per
// CTA; one warp per token copies head_dim code bytes and the scale bytes.
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const uint8_t* __restrict__ kv_cache, uint8_t* __restrict__ dst_k,
    uint8_t* __restrict__ dst_scale, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens, const int batch_size, const int64_t token_stride,
    const int64_t head_dim, const int64_t block_stride, const int64_t cache_token_stride,
    const int64_t cache_block_size, const int num_blocks_per_seq, const int num_tokens,
    const int quant_block_size) {
  GatherSpec spec{};
  spec.block_table = block_table;
  spec.block_table_stride = num_blocks_per_seq;
  spec.cu_seq_lens = cu_seq_lens;
  spec.seq_starts = nullptr;
  spec.num_reqs = batch_size;
  spec.num_tokens = num_tokens;
  const int lane = threadIdx.x;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * BLOCK_Y_SIZE + threadIdx.y;
  int64_t page = 0, off = 0;
  if (!resolve_gather_row(spec, static_cast<int>(t), static_cast<int>(cache_block_size), &page,
                          &off)) {
    return;
  }
  const uint8_t* src = kv_cache + page * block_stride + off * cache_token_stride;
  const uint8_t* page_base = kv_cache + page * block_stride;
  uint8_t* out = dst_k + t * token_stride;
  for (int64_t i = lane * 4; i < head_dim; i += 128) {
    if (i + 4 <= head_dim) {
      *reinterpret_cast<uint32_t*>(out + i) = *reinterpret_cast<const uint32_t*>(src + i);
    } else {
      for (int64_t j = i; j < head_dim; ++j) out[j] = src[j];
    }
  }
  const int groups = static_cast<int>(head_dim) / quant_block_size;
  const int64_t scale_bytes = static_cast<int64_t>(groups) * 4;
  const uint8_t* scales = page_base + static_cast<int64_t>(cache_block_size) * head_dim +
                          off * scale_bytes;
  const int64_t scale_stride = (quant_block_size == 0) ? 0 : scale_bytes;
  uint8_t* scale_out = dst_scale + t * scale_stride;
  for (int64_t i = lane; i < scale_bytes; i += 32) scale_out[i] = scales[i];
  (void)batch_size;
}

template <typename scalar_t, typename cache_t, KVCodec kCodec, int ENTRY_SIZE>
__global__ void gather_and_maybe_dequant_cache_kernel(
    const cache_t* __restrict__ src_cache, scalar_t* __restrict__ dst, const GatherSpec spec,
    const int32_t block_size, const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride, const float* __restrict__ scale) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (token >= spec.num_tokens) return;
  int64_t page = 0;
  int64_t page_off = 0;
  if (!resolve_gather_row(spec, token, block_size, &page, &page_off)) return;
  const cache_t* src = src_cache + page * cache_block_stride + page_off * cache_entry_stride;
  scalar_t* out = dst + static_cast<int64_t>(token) * dst_entry_stride;
  const float s = scale != nullptr ? scale[0] : 1.0f;
  for (int i = lane; i < ENTRY_SIZE; i += 32) {
    const float v = cache_load<cache_t, kCodec>(src[i], s);
    scalar_t y;
    from_f<scalar_t>(y, v);
    out[i] = y;
  }
}

// =============================================================================
// K03  conversion / relayout
// =============================================================================

template <typename out_t, typename in_t, KVCodec kCodec>
__global__ void convert_cache_kernel(const in_t* __restrict__ src_cache,
                                     out_t* __restrict__ dst_cache, const float scale,
                                     const int64_t block_stride) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * block_stride;
  for (int64_t i = threadIdx.x; i < block_stride; i += blockDim.x) {
    if constexpr (kCodec == KVCodec::kNative) {
      const float v = to_f<in_t>(src_cache[base + i]) * scale;
      out_t o;
      from_f<out_t>(o, v);
      dst_cache[base + i] = o;
    } else {
      // Decoding (1-byte source) or encoding (1-byte destination).
      if constexpr (sizeof(in_t) == 1 && sizeof(out_t) != 1) {
        const float v = e4m3_to_float(static_cast<uint8_t>(src_cache[base + i])) * scale;
        out_t o;
        from_f<out_t>(o, v);
        dst_cache[base + i] = o;
      } else if constexpr (sizeof(out_t) == 1 && sizeof(in_t) != 1) {
        const float q = scale != 0.0f ? to_f<in_t>(src_cache[base + i]) / scale : 0.0f;
        dst_cache[base + i] = static_cast<out_t>(e4m3_from_float(q));
      } else {
        dst_cache[base + i] = src_cache[base + i];
      }
    }
  }
}

__device__ __forceinline__ int64_t relayout_src_page(const PagedKVDesc& d,
                                                     const int32_t* indptr, int seq, int j) {
  if (d.kv_indptr != nullptr && d.kv_page_indices != nullptr) {
    return d.kv_page_indices[d.kv_indptr[seq] + j];
  }
  return static_cast<int64_t>(indptr[seq]) + j;
}

template <typename cache_t>
__global__ void kv_relayout_pages_kernel(const PagedKVDesc src_desc,
                                         const PagedKVDesc dst_desc,
                                         const int32_t* __restrict__ src_seq_page_indptr,
                                         const int32_t* __restrict__ dst_seq_page_indptr,
                                         const int batch_size) {
  const int seq = static_cast<int>(blockIdx.x);
  if (seq >= batch_size) return;
  const int tile = static_cast<int>(blockIdx.y);
  const int tiles = static_cast<int>(gridDim.y);
  const int src_pages = src_seq_page_indptr[seq + 1] - src_seq_page_indptr[seq];
  const int dst_pages = dst_seq_page_indptr[seq + 1] - dst_seq_page_indptr[seq];
  if (src_pages < 0 || dst_pages < 0) return;
  const int64_t src_tokens = static_cast<int64_t>(src_pages) * src_desc.page_size;
  const int64_t dst_tokens = static_cast<int64_t>(dst_pages) * dst_desc.page_size;
  const int64_t tokens = src_tokens < dst_tokens ? src_tokens : dst_tokens;
  const bool latent = src_desc.layout == KVLayout::kLatent ||
                      src_desc.entry_bytes != 0 || dst_desc.entry_bytes != 0;
  const int heads = src_desc.num_kv_heads > 0 ? src_desc.num_kv_heads : 1;
  const int64_t dense_elems = static_cast<int64_t>(heads) * src_desc.head_dim_k;
  const int64_t elems = latent ? (src_desc.entry_bytes != 0
                                      ? src_desc.entry_bytes / static_cast<int64_t>(sizeof(cache_t))
                                      : src_desc.head_dim_k)
                               : dense_elems;
  const cache_t* src = static_cast<const cache_t*>(src_desc.k_data);
  cache_t* dst = static_cast<cache_t*>(dst_desc.k_data);
  for (int64_t t = tile; t < tokens; t += tiles) {
    const int64_t sp = t / src_desc.page_size;
    const int64_t so = t % src_desc.page_size;
    const int64_t dp = t / dst_desc.page_size;
    const int64_t doff = t % dst_desc.page_size;
    const int64_t spage = relayout_src_page(src_desc, src_seq_page_indptr, seq, static_cast<int>(sp));
    const int64_t dpage = relayout_src_page(dst_desc, dst_seq_page_indptr, seq, static_cast<int>(dp));
    if (spage < 0 || dpage < 0) continue;
    const int64_t sbase = spage * src_desc.k_stride_page + so * src_desc.k_stride_token;
    const int64_t dbase = dpage * dst_desc.k_stride_page + doff * dst_desc.k_stride_token;
    constexpr int kV = 16 / static_cast<int>(sizeof(cache_t));
    const bool vectorized =
        (elems % kV) == 0 && (src_desc.head_dim_k % kV) == 0 &&
        (sbase % kV) == 0 && (dbase % kV) == 0 &&
        (latent || ((src_desc.k_stride_head % kV) == 0 && (dst_desc.k_stride_head % kV) == 0)) &&
        ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(src) & 15u) == 0u);
    if (vectorized) {
      const int packs = static_cast<int>(elems / kV);
      const int packs_per_head =
          latent ? packs : static_cast<int>(src_desc.head_dim_k / kV);
      for (int p = static_cast<int>(threadIdx.x); p < packs; p += static_cast<int>(blockDim.x)) {
        int64_t si = sbase;
        int64_t di = dbase;
        int off = p * kV;
        if (!latent) {
          const int h = p / packs_per_head;
          off = (p % packs_per_head) * kV;
          si += static_cast<int64_t>(h) * src_desc.k_stride_head;
          di += static_cast<int64_t>(h) * dst_desc.k_stride_head;
        }
        *reinterpret_cast<uint4*>(dst + di + off) =
            *reinterpret_cast<const uint4*>(src + si + off);
      }
    } else {
      for (int64_t e = threadIdx.x; e < elems; e += blockDim.x) {
        int64_t si = 0;
        int64_t di = 0;
        if (latent) {
          si = sbase + e;
          di = dbase + e;
        } else {
          const int h = static_cast<int>(e / src_desc.head_dim_k);
          const int d = static_cast<int>(e % src_desc.head_dim_k);
          si = sbase + static_cast<int64_t>(h) * src_desc.k_stride_head + d;
          const int dh = static_cast<int>(e / dst_desc.head_dim_k);
          const int dd = static_cast<int>(e % dst_desc.head_dim_k);
          di = dbase + static_cast<int64_t>(dh) * dst_desc.k_stride_head + dd;
        }
        dst[di] = src[si];
      }
    }
  }
}

namespace {

// =============================================================================
// Host helpers
// =============================================================================

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
    return detail::set_error(Status::kUnsupportedBackend, "kvcache: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool supported_scalar(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

bool fp8_storage(DType t) { return dtype_is_fp8(t) || t == DType::kUInt8; }

/// B2 compiles the OCP e4m3 conversion path only.
bool fp8_payload_unsupported(DType t) {
  return t == DType::kFloat8E5M2 || t == DType::kFloat8E4M3FNUZ ||
         t == DType::kFloat8E5M2FNUZ;
}

int threads_for(int64_t work, int minimum = 32, int maximum = 512) {
  int n = static_cast<int>(work > maximum ? maximum : (work < minimum ? minimum : work));
  n = ((n + 31) / 32) * 32;
  return n;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Convert byte strides of a descriptor into element strides for cache_t.
/// Fills standard defaults when a stride field is zero.
template <typename cache_t>
Status element_strides(const PagedKVDesc& d, int64_t default_token, int64_t default_page,
                       int64_t default_head, PagedKVDesc* out) {
  *out = d;
  const int64_t es = static_cast<int64_t>(sizeof(cache_t));
  if (d.stride_unit == StrideUnit::kByte) {
    if ((d.k_stride_page % es) != 0 || (d.k_stride_token % es) != 0 ||
        (d.k_stride_head % es) != 0 || (d.v_stride_page % es) != 0 ||
        (d.v_stride_token % es) != 0 || (d.v_stride_head % es) != 0) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "kvcache: byte strides not divisible by cache element size");
    }
    out->k_stride_page /= es;
    out->k_stride_token /= es;
    out->k_stride_head /= es;
    out->v_stride_page /= es;
    out->v_stride_token /= es;
    out->v_stride_head /= es;
    out->stride_unit = StrideUnit::kElement;
  }
  if (out->k_stride_page == 0) out->k_stride_page = default_page;
  if (out->k_stride_token == 0) out->k_stride_token = default_token;
  if (out->k_stride_head == 0) out->k_stride_head = default_head;
  return Status::kSuccess;
}

/// Element-strided copy of a relayout descriptor, including latent records whose
/// byte strides must be divided by sizeof(cache_t).
template <typename cache_t>
Status relayout_element_desc(const PagedKVDesc& in, PagedKVDesc* out) {
  const int heads = in.num_kv_heads > 0 ? in.num_kv_heads : 1;
  const int64_t entry = in.entry_bytes != 0 ? in.entry_bytes / static_cast<int64_t>(sizeof(cache_t))
                                            : in.head_dim_k;
  const bool latent = in.layout == KVLayout::kLatent || in.entry_bytes != 0;
  const int64_t token = latent ? entry
                               : (in.layout == KVLayout::kHND
                                      ? in.head_dim_k
                                      : static_cast<int64_t>(heads) * in.head_dim_k);
  const int64_t page = latent
                           ? entry * in.page_size
                           : static_cast<int64_t>(heads) * in.head_dim_k * in.page_size;
  const int64_t head = latent ? entry
                              : (in.layout == KVLayout::kHND
                                     ? static_cast<int64_t>(in.page_size) * in.head_dim_k
                                     : in.head_dim_k);
  return element_strides<cache_t>(in, token, page, head, out);
}

Status check_slot_mapping(const Tensor& slots, int64_t* num_tokens) {
  Status s = check_tensor(slots, DType::kInt64, 1, 8, "kvcache.slot_mapping");
  if (s != Status::kSuccess) return s;
  if (slots.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "kvcache.slot_mapping: last stride must be 1");
  }
  *num_tokens = slots.shape[0];
  return Status::kSuccess;
}

Status check_scale(const float* scale, KVCodec codec, int scale_head_stride, const char* name) {
  if (codec == KVCodec::kFP8PerTensor || codec == KVCodec::kFP8PerHead) {
    if (scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "%s: FP8 codec needs a scale", name);
    }
    if (codec == KVCodec::kFP8PerTensor && scale_head_stride != 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: per-tensor scale must have scale_head_stride 0", name);
    }
    if (codec == KVCodec::kFP8PerHead && scale_head_stride != 1) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: per-head scale must have scale_head_stride 1", name);
    }
  }
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// reshape_and_cache dispatchers
// -----------------------------------------------------------------------------

template <typename scalar_t, typename cache_t, KVCodec kCodec>
Status launch_reshape_flash(const Tensor& key, const Tensor& value, const PagedKVDesc& d,
                            const Tensor& slots, int64_t num_tokens, cudaStream_t stream) {
  PagedKVDesc e{};
  const int heads = static_cast<int>(key.shape[1]);
  const int dk = static_cast<int>(key.shape[2]);
  const int dv = static_cast<int>(value.shape[2]);
  const bool hnd = d.layout == KVLayout::kHND;
  const int64_t default_token = hnd ? dk : static_cast<int64_t>(heads) * dk;
  const int64_t default_page = hnd ? static_cast<int64_t>(heads) * d.page_size * dk
                                   : static_cast<int64_t>(heads) * dk * d.page_size;
  const int64_t default_head = hnd ? static_cast<int64_t>(d.page_size) * dk : dk;
  Status s = element_strides<cache_t>(d, default_token, default_page, default_head, &e);
  if (s != Status::kSuccess) return s;
  // One warp per token, eight tokens per CTA: the kernel works in 16-byte packs, not elements, and a
  // 512-thread block for 512 elements left 448 threads idle.
  constexpr int kWarps = 8;
  const int block = 32 * kWarps;
  if (num_tokens > INT32_MAX) return detail::set_error(Status::kCapacityExceeded, "reshape: %lld tokens",
                                                       static_cast<long long>(num_tokens));
  const int64_t grid = (num_tokens + kWarps - 1) / kWarps;
  reshape_and_cache_flash_kernel<scalar_t, cache_t, kCodec>
      <<<static_cast<int>(grid), block, 0, stream>>>(
          const_ptr<scalar_t>(key), const_ptr<scalar_t>(value), static_cast<cache_t*>(d.k_data),
          static_cast<cache_t*>(d.v_data), const_ptr<int64_t>(slots), e.k_stride_page,
          e.k_stride_token, e.k_stride_head, key.stride[0], value.stride[0], heads, dk, dv,
          d.page_size, d.k_scale, d.v_scale, d.scale_head_stride, num_tokens);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("reshape_and_cache_flash", err);
  return Status::kSuccess;
}

template <typename scalar_t, typename cache_t, KVCodec kCodec>
Status launch_reshape_legacy(const Tensor& key, const Tensor& value, const PagedKVDesc& d,
                             const Tensor& slots, int64_t num_tokens, cudaStream_t stream) {
  const int heads = static_cast<int>(key.shape[1]);
  const int dk = static_cast<int>(key.shape[2]);
  const int x = static_cast<int>(16 / sizeof(cache_t));
  if (dk % x != 0) {
    return detail::set_error(Status::kUnsupportedShape, "reshape legacy: head_size %d not / %d", dk,
                             x);
  }
  const int block = threads_for(static_cast<int64_t>(heads) * (dk / x), 32, 1024);
  reshape_and_cache_legacy_kernel<scalar_t, cache_t, kCodec>
      <<<static_cast<int>(num_tokens), block, 0, stream>>>(
          const_ptr<scalar_t>(key), const_ptr<scalar_t>(value), static_cast<cache_t*>(d.k_data),
          static_cast<cache_t*>(d.v_data), const_ptr<int64_t>(slots),
          static_cast<int>(key.stride[0]), static_cast<int>(value.stride[0]), heads, dk,
          d.page_size, x, d.k_scale, d.v_scale);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("reshape_and_cache_legacy", err);
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// append_paged_kv / ring dispatchers
// -----------------------------------------------------------------------------

template <typename scalar_t, typename cache_t, KVCodec kCodec, KVLayout kLayout>
Status launch_append(const Tensor& key, const Tensor& value, const PagedKVDesc& d,
                     const Tensor& batch_indices, const Tensor& positions, cudaStream_t stream) {
  const int64_t nnz = key.shape[0];
  const int heads = static_cast<int>(key.shape[1]);
  const int dk = static_cast<int>(key.shape[2]);
  const int dv = static_cast<int>(value.shape[2]);
  const bool hnd = d.layout == KVLayout::kHND;
  PagedKVDesc e{};
  const int64_t default_token = hnd ? dk : static_cast<int64_t>(heads) * dk;
  const int64_t default_page = static_cast<int64_t>(heads) * d.page_size * dk;
  const int64_t default_head = hnd ? static_cast<int64_t>(d.page_size) * dk : dk;
  Status es = element_strides<cache_t>(d, default_token, default_page, default_head, &e);
  if (es != Status::kSuccess) return es;
  if (e.v_stride_page == 0) e.v_stride_page = static_cast<int64_t>(heads) * d.page_size * dv;
  if (e.v_stride_token == 0) e.v_stride_token = hnd ? dv : static_cast<int64_t>(heads) * dv;
  if (e.v_stride_head == 0) e.v_stride_head = hnd ? static_cast<int64_t>(d.page_size) * dv : dv;
  constexpr int kAppendThreads = 256;
  // Thread mapping must match append_paged_kv_kernel: warps per token from the 16-byte chunk
  // count of one token (K chunks then V), tokens per CTA from the CTA size.
  constexpr int kVec = 16 / static_cast<int>(sizeof(scalar_t));
  const bool v_present = tensor_present(value) && dv > 0;
  const int k_chunks = (dk > 0 && dk % kVec == 0) ? heads * (dk / kVec) : 0;
  const int v_chunks = (v_present && dv % kVec == 0) ? heads * (dv / kVec) : 0;
  const int tpt = append_threads_per_token(k_chunks + v_chunks, kAppendThreads);
  const int tpc = kAppendThreads / tpt > 0 ? kAppendThreads / tpt : 1;
  const int64_t grid64 = (nnz + tpc - 1) / tpc;
  if (grid64 > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "append_paged_kv: %lld tokens",
                             static_cast<long long>(nnz));
  }
  append_paged_kv_kernel<scalar_t, cache_t, kCodec, kLayout>
      <<<static_cast<int>(grid64), kAppendThreads, 0, stream>>>(
          e, const_ptr<scalar_t>(key), const_ptr<scalar_t>(value),
          const_ptr<int32_t>(batch_indices), const_ptr<int32_t>(positions), key.stride[0],
          key.stride[1], value.stride[0], value.stride[1], static_cast<int>(nnz));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("append_paged_kv", err);
  return Status::kSuccess;
}

template <typename scalar_t, typename cache_t, KVCodec kCodec>
Status launch_ring(const Tensor& key, const Tensor& value, const Tensor& ring_k,
                   const Tensor& ring_v, const Tensor& token_slot, const Tensor& positions,
                   const Tensor& k_scale, const Tensor& v_scale, int32_t window,
                   cudaStream_t stream) {
  const int heads = static_cast<int>(key.shape[1]);
  const int dk = static_cast<int>(key.shape[2]);
  const int dv = tensor_present(value) ? static_cast<int>(value.shape[2]) : 0;
  const int64_t num_tokens = key.shape[0];
  const int block = threads_for(static_cast<int64_t>(heads) * (dk > dv ? dk : dv), 32, 512);
  ring_kv_append_kernel<scalar_t, cache_t, kCodec><<<static_cast<int>(num_tokens), block, 0, stream>>>(
      const_ptr<scalar_t>(key), tensor_present(value) ? const_ptr<scalar_t>(value) : nullptr,
      static_cast<cache_t*>(ring_k.data), static_cast<cache_t*>(ring_v.data),
      const_ptr<int32_t>(token_slot), const_ptr<int64_t>(positions),
      tensor_present(k_scale) ? const_ptr<float>(k_scale) : nullptr,
      tensor_present(v_scale) ? const_ptr<float>(v_scale) : nullptr, window, heads, dk, dv,
      static_cast<int>(num_tokens));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("ring_kv_append", err);
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// MLA dispatchers
// -----------------------------------------------------------------------------

template <typename scalar_t, typename cache_t, KVCodec kCodec>
Status launch_concat_mla(const Tensor& kv_c, const Tensor& k_pe, const PagedKVDesc& d,
                         const Tensor& slots, int64_t num_tokens, cudaStream_t stream) {
  const int rank = static_cast<int>(kv_c.shape[1]);
  const int pe = tensor_present(k_pe) ? static_cast<int>(k_pe.shape[1]) : 0;
  PagedKVDesc e{};
  Status s = element_strides<cache_t>(d, rank + pe,
                                      static_cast<int64_t>(rank + pe) * d.page_size, rank + pe, &e);
  if (s != Status::kSuccess) return s;
  if (e.k_stride_page > INT32_MAX || e.k_stride_token > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "concat_and_cache_mla: strides > int32");
  }
  const int block = threads_for(rank, 32, 512);
  concat_and_cache_mla_kernel<scalar_t, cache_t, kCodec>
      <<<static_cast<int>(num_tokens), block, 0, stream>>>(
          const_ptr<scalar_t>(kv_c), tensor_present(k_pe) ? const_ptr<scalar_t>(k_pe) : nullptr,
          static_cast<cache_t*>(d.k_data), const_ptr<int64_t>(slots),
          static_cast<int>(e.k_stride_page), static_cast<int>(e.k_stride_token),
          static_cast<int>(kv_c.stride[0]),
          tensor_present(k_pe) ? static_cast<int>(k_pe.stride[0]) : 0, rank, pe, d.page_size,
          d.k_scale);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("concat_and_cache_mla", err);
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// descriptors
// -----------------------------------------------------------------------------

}  // namespace

// =============================================================================
// Host API
// =============================================================================

Status kv_desc_validate(const PagedKVDesc& desc) {
  if (desc.k_data == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: k_data is null");
  }
  if (desc.page_size <= 0) {
    return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: page_size %d", desc.page_size);
  }
  if (desc.num_kv_heads < 0 || desc.head_dim_k < 0 || desc.head_dim_v < 0 ||
      desc.entry_bytes < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "PagedKVDesc: negative head/entry sizes (%d, %d, %d, %d)",
                             desc.num_kv_heads, desc.head_dim_k, desc.head_dim_v, desc.entry_bytes);
  }
  if (desc.entry_bytes != 0 && desc.stride_unit != StrideUnit::kByte) {
    return detail::set_error(Status::kInvalidArgument,
                             "PagedKVDesc: entry_bytes != 0 requires byte stride_unit");
  }
  if (desc.entry_bytes == 0 && desc.stride_unit != StrideUnit::kElement) {
    return detail::set_error(Status::kInvalidArgument,
                             "PagedKVDesc: dense layout requires element stride_unit");
  }
  const bool k_fp8 = dtype_is_fp8(desc.k_dtype);
  switch (desc.k_codec) {
    case KVCodec::kNative:
      if (k_fp8) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: native codec with FP8 k_dtype %s",
                                 dtype_name(desc.k_dtype));
      }
      break;
    case KVCodec::kFP8PerTensor:
      if (!k_fp8) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: FP8 codec requires an FP8 k_dtype, got %s",
                                 dtype_name(desc.k_dtype));
      }
      break;
    case KVCodec::kFP8PerHead:
      if (!k_fp8) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: FP8 per-head requires an FP8 k_dtype, got %s",
                                 dtype_name(desc.k_dtype));
      }
      if (desc.scale_head_stride != 1) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: per-head codec needs scale_head_stride 1");
      }
      break;
    case KVCodec::kMLAFp8DS:
      if (desc.entry_bytes != 656) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DS record must be 656 B, got %d",
                                 desc.entry_bytes);
      }
      if (desc.head_dim_k != 576) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DS head_dim_k must be 576, got %d",
                                 desc.head_dim_k);
      }
      if (desc.fp8 != Fp8Encoding::kOCP) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: kMLAFp8DS requires OCP FP8");
      }
      break;
    case KVCodec::kMLAFp8DSFNUZ:
      if (desc.entry_bytes != 656) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DSFNUZ record must be 656 B");
      }
      if (desc.fp8 != Fp8Encoding::kFNUZ) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: kMLAFp8DSFNUZ requires FNUZ FP8");
      }
      break;
    case KVCodec::kMLAFp8DSV40: {
      if (desc.layout != KVLayout::kLatent) {
        return detail::set_error(Status::kUnsupportedLayout,
                                 "PagedKVDesc: kMLAFp8DSV40 is a latent record codec");
      }
      if (desc.entry_bytes != 576) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DSV40 data record must be 576 B, got %d",
                                 desc.entry_bytes);
      }
      if (desc.head_dim_k != 512 || desc.head_dim_v != 512) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DSV40 head dims must be 512/512, got %d/%d",
                                 desc.head_dim_k, desc.head_dim_v);
      }
      if (desc.num_kv_heads > 1) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: kMLAFp8DSV40 has one latent head, got %d",
                                 desc.num_kv_heads);
      }
      if (desc.k_dtype != DType::kFloat8E4M3 || desc.fp8 != Fp8Encoding::kOCP) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: kMLAFp8DSV40 requires an OCP e4m3 payload, got %s",
                                 dtype_name(desc.k_dtype));
      }
      if (desc.scale_bytes_per_token != 8) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: kMLAFp8DSV40 needs scale_bytes_per_token 8, got %d",
                                 desc.scale_bytes_per_token);
      }
      if (desc.k_scale_pages == nullptr || desc.k_scale_stride_page <= 0 ||
          desc.k_scale_stride_token < 8) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: kMLAFp8DSV40 needs k_scale_pages with positive "
                                 "page/token strides");
      }
      if (desc.k_stride_token != 0 && desc.k_stride_token < 576) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: kMLAFp8DSV40 token stride %lld < 576 B",
                                 static_cast<long long>(desc.k_stride_token));
      }
      if (desc.v_codec != KVCodec::kNative) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: kMLAFp8DSV40 has no V cache (v_codec must be "
                                 "native)");
      }
      break;
    }
    case KVCodec::kMLAFp8DSV41:
    case KVCodec::kMLAFp8DSV41Fp4: {
      const bool fp4 = desc.k_codec == KVCodec::kMLAFp8DSV41Fp4;
      const int want_bytes = fp4 ? 288 : 528;
      if (desc.layout != KVLayout::kLatent) {
        return detail::set_error(Status::kUnsupportedLayout,
                                 "PagedKVDesc: codec %d is a latent record codec",
                                 static_cast<int>(desc.k_codec));
      }
      if (desc.entry_bytes != want_bytes) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: codec %d record must be %d B, got %d",
                                 static_cast<int>(desc.k_codec), want_bytes, desc.entry_bytes);
      }
      if (desc.head_dim_k != 512 || desc.head_dim_v != 512) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: codec %d head dims must be 512/512, got %d/%d",
                                 static_cast<int>(desc.k_codec), desc.head_dim_k, desc.head_dim_v);
      }
      if (desc.num_kv_heads > 1) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: codec %d has one latent head, got %d",
                                 static_cast<int>(desc.k_codec), desc.num_kv_heads);
      }
      if (desc.scale_bytes_per_token != 0 || desc.k_scale_pages != nullptr) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: codec %d stores its scales inline",
                                 static_cast<int>(desc.k_codec));
      }
      if (fp4) {
        if ((desc.k_dtype != DType::kFloat4E2M1x2 && desc.k_dtype != DType::kFloat8E4M3) ||
            desc.fp8 != Fp8Encoding::kOCP) {
          return detail::set_error(Status::kUnsupportedDtype,
                                   "PagedKVDesc: kMLAFp8DSV41Fp4 needs an e2m1 payload (or the "
                                   "e4m3 record view) with OCP scales, got %s",
                                   dtype_name(desc.k_dtype));
        }
      } else if (desc.k_dtype != DType::kFloat8E4M3 || desc.fp8 != Fp8Encoding::kOCP) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: codec %d requires an OCP e4m3 payload, got %s",
                                 static_cast<int>(desc.k_codec), dtype_name(desc.k_dtype));
      }
      if (desc.k_stride_token != 0 && desc.k_stride_token < want_bytes) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: codec %d token stride %lld < %d B",
                                 static_cast<int>(desc.k_codec),
                                 static_cast<long long>(desc.k_stride_token), want_bytes);
      }
      if (desc.v_codec != KVCodec::kNative) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "PagedKVDesc: codec %d has no V cache (v_codec must be native)",
                                 static_cast<int>(desc.k_codec));
      }
      break;
    }
    case KVCodec::kFP8PerTokenHead:
    case KVCodec::kInt8PerTokenHead:
    case KVCodec::kInt4PerTokenHead:
    case KVCodec::kNVFP4:
    case KVCodec::kLowBitWHT:
      if (desc.k_token_scales == nullptr) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: codec %d needs k_token_scales",
                                 static_cast<int>(desc.k_codec));
      }
      break;
    case KVCodec::kIndexerFP8:
    case KVCodec::kIndexerFP4:
      break;
    default:
      return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: unknown codec %d",
                               static_cast<int>(desc.k_codec));
  }
  if (desc.v_data != nullptr) {
    switch (desc.v_codec) {
      case KVCodec::kNative:
      case KVCodec::kFP8PerTensor:
      case KVCodec::kFP8PerHead:
      case KVCodec::kFP8PerTokenHead:
      case KVCodec::kInt8PerTokenHead:
      case KVCodec::kInt4PerTokenHead:
        break;
      default:
        return detail::set_error(Status::kUnsupportedShape,
                                 "PagedKVDesc: unsupported V codec %d",
                                 static_cast<int>(desc.v_codec));
    }
  }
  if (desc.k_codec != KVCodec::kNative && desc.k_codec != KVCodec::kMLAFp8DS &&
      desc.k_codec != KVCodec::kMLAFp8DSFNUZ && desc.k_codec != KVCodec::kIndexerFP8 &&
      desc.k_codec != KVCodec::kIndexerFP4 && desc.k_codec != KVCodec::kMLAFp8DSV40 &&
      desc.k_codec != KVCodec::kMLAFp8DSV41 && desc.k_codec != KVCodec::kMLAFp8DSV41Fp4 &&
      desc.k_scale == nullptr &&
      desc.k_codec != KVCodec::kFP8PerTokenHead &&
      desc.k_codec != KVCodec::kInt8PerTokenHead &&
      desc.k_codec != KVCodec::kInt4PerTokenHead) {
    return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: k_scale is required");
  }
  if (desc.layout == KVLayout::kLatent) {
    if (desc.v_data != nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "PagedKVDesc: latent layout has one shared cache (v_data null)");
    }
    if (desc.head_dim_k <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "PagedKVDesc: latent layout needs head_dim_k");
    }
  } else if (desc.k_codec == KVCodec::kNative && desc.entry_bytes == 0) {
    if (desc.num_kv_heads <= 0 || desc.head_dim_k <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "PagedKVDesc: dense layout needs positive head dims");
    }
    if (desc.v_data != nullptr && desc.head_dim_v <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "PagedKVDesc: v_data needs positive head_dim_v");
    }
  } else if (desc.layout == KVLayout::kVllmLegacy) {
    if (desc.legacy_x <= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "PagedKVDesc: legacy layout needs legacy_x");
    }
  }
  switch (desc.table_format) {
    case PageTableFormat::kCSR:
      if (desc.kv_indptr == nullptr) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: CSR table needs kv_indptr");
      }
      break;
    case PageTableFormat::kBlockTable:
      if (desc.block_table == nullptr || desc.block_table_stride <= 0) {
        return detail::set_error(Status::kInvalidArgument,
                                 "PagedKVDesc: block table needs block_table/stride");
      }
      break;
    case PageTableFormat::kSlotMapping:
      break;
    default:
      return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: unknown table format");
  }
  if (desc.k_stride_page < 0 || desc.k_stride_token < 0 || desc.k_stride_head < 0 ||
      desc.v_stride_page < 0 || desc.v_stride_token < 0 || desc.v_stride_head < 0) {
    return detail::set_error(Status::kInvalidArgument, "PagedKVDesc: negative strides");
  }
  return Status::kSuccess;
}

int64_t kv_physical_bytes(const PagedKVDesc& desc, int64_t num_pages) {
  if (desc.page_size <= 0 || num_pages <= 0) return 0;
  auto align16 = [](int64_t n) { return (n + 15) & ~static_cast<int64_t>(15); };
  const int64_t ps = desc.page_size;
  // V4 MLA records own their page geometry: the format fixes the per-token width, so the ledger
  // does not depend on a caller-supplied stride. V4.0 keeps the FlashMLA page padding rule
  // (the whole page is rounded up to a multiple of the 576 B data row) and counts its 8 B/token
  // out-of-record scale region once, inside that padded page.
  switch (desc.k_codec) {
    case KVCodec::kMLAFp8DSV40: {
      int64_t page = ((ps * 584 + 575) / 576) * 576;
      if (desc.stride_unit == StrideUnit::kByte && desc.k_stride_page > page) {
        page = desc.k_stride_page;  // caller padded further; never double count the scales
      }
      return align16(page) * num_pages;
    }
    case KVCodec::kMLAFp8DSV41: {
      int64_t page = ps * 528;
      if (desc.stride_unit == StrideUnit::kByte && desc.k_stride_page > page) {
        page = desc.k_stride_page;
      }
      return align16(page) * num_pages;
    }
    case KVCodec::kMLAFp8DSV41Fp4: {
      int64_t page = ps * 288;
      if (desc.stride_unit == StrideUnit::kByte && desc.k_stride_page > page) {
        page = desc.k_stride_page;
      }
      return align16(page) * num_pages;
    }
    default:
      break;
  }
  auto page_bytes = [&](DType dtype, KVCodec codec, int32_t entry_bytes, int32_t heads,
                        int32_t head_dim, int64_t stride_page, int64_t stride_token,
                        int64_t stride_head, StrideUnit unit) -> int64_t {
    if (stride_page > 0 && unit == StrideUnit::kByte) return stride_page;
    if (entry_bytes > 0) return static_cast<int64_t>(desc.page_size) * entry_bytes;
    const int64_t elem_bytes = dtype_bits(dtype) > 0 ? (dtype_bits(dtype) + 7) / 8 : 0;
    int64_t elems = 0;
    if (stride_page > 0 && unit == StrideUnit::kElement && heads > 0 && head_dim > 0) {
      elems = stride_page;
    } else if (entry_bytes == 0 && heads > 0 && head_dim > 0) {
      elems = static_cast<int64_t>(desc.page_size) * heads * head_dim;
    }
    (void)stride_token;
    (void)stride_head;
    (void)codec;
    return elems * elem_bytes;
  };
  int64_t bytes = page_bytes(desc.k_dtype, desc.k_codec, desc.entry_bytes, desc.num_kv_heads,
                             desc.head_dim_k, desc.k_stride_page, desc.k_stride_token,
                             desc.k_stride_head, desc.stride_unit);
  if (desc.v_data != nullptr) {
    bytes += page_bytes(desc.v_dtype, desc.v_codec, desc.entry_bytes, desc.num_kv_heads,
                        desc.head_dim_v, desc.v_stride_page, desc.v_stride_token,
                        desc.v_stride_head, desc.stride_unit);
  }
  if (desc.k_token_scales != nullptr && desc.token_scale_stride_page > 0) {
    bytes += desc.token_scale_stride_page;
  }
  if (desc.v_token_scales != nullptr && desc.token_scale_stride_page > 0) {
    bytes += desc.token_scale_stride_page;
  }
  return bytes * num_pages;
}

Status reshape_and_cache(const Tensor& key, const Tensor& value, const PagedKVDesc& cache,
                         const Tensor& slot_mapping, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (key.data == nullptr || value.data == nullptr || slot_mapping.data == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "reshape_and_cache: key/value/slot_mapping is null");
  }
  if (cache.layout == KVLayout::kLatent) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "reshape_and_cache: use concat_and_cache_mla for latent caches");
  }
  if (!supported_scalar(key.dtype) || key.dtype != value.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "reshape_and_cache: key/value dtype %s/%s",
                             dtype_name(key.dtype), dtype_name(value.dtype));
  }
  s = check_tensor(key, key.dtype, 3, 16, "reshape_and_cache.key");
  if (s != Status::kSuccess) return s;
  s = check_tensor(value, value.dtype, 3, 16, "reshape_and_cache.value");
  if (s != Status::kSuccess) return s;
  if (key.stride[2] != 1 || value.stride[2] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "reshape_and_cache: key/value last dim must be contiguous");
  }
  int64_t num_tokens = 0;
  s = check_slot_mapping(slot_mapping, &num_tokens);
  if (s != Status::kSuccess) return s;
  if (key.shape[0] < num_tokens || value.shape[0] < num_tokens) {
    return detail::set_error(Status::kInvalidArgument,
                             "reshape_and_cache: sources have %lld/%lld rows, slot_mapping %lld",
                             static_cast<long long>(key.shape[0]),
                             static_cast<long long>(value.shape[0]),
                             static_cast<long long>(num_tokens));
  }
  if (cache.num_kv_heads > 0 && key.shape[1] != cache.num_kv_heads) {
    return detail::set_error(Status::kInvalidArgument,
                             "reshape_and_cache: %lld heads vs descriptor %d",
                             static_cast<long long>(key.shape[1]), cache.num_kv_heads);
  }
  if (cache.head_dim_k > 0 && key.shape[2] != cache.head_dim_k) {
    return detail::set_error(Status::kInvalidArgument,
                             "reshape_and_cache: head_dim_k %lld vs descriptor %d",
                             static_cast<long long>(key.shape[2]), cache.head_dim_k);
  }
  if (cache.v_data != nullptr && cache.head_dim_v > 0 && value.shape[2] != cache.head_dim_v) {
    return detail::set_error(Status::kInvalidArgument,
                             "reshape_and_cache: head_dim_v %lld vs descriptor %d",
                             static_cast<long long>(value.shape[2]), cache.head_dim_v);
  }
  if (cache.layout == KVLayout::kVllmLegacy && key.shape[2] != value.shape[2]) {
    return detail::set_error(Status::kUnsupportedShape,
                             "reshape_and_cache: legacy layout requires Dk == Dv");
  }
  if (cache.v_data != nullptr && cache.v_codec != cache.k_codec) {
    return detail::set_error(Status::kUnsupportedShape,
                             "reshape_and_cache: B2 requires K and V to share one codec");
  }
  if (token_codec(cache.k_codec)) {
    if (!token_codec(cache.v_codec)) {
      return detail::set_error(Status::kUnsupportedShape,
                               "reshape_and_cache: K and V must both be per-token-head");
    }
    if (cache.layout != KVLayout::kNHD && cache.layout != KVLayout::kHND) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "reshape_and_cache: per-token-head needs NHD or HND");
    }
    if (cache.k_token_scales == nullptr || cache.v_token_scales == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "reshape_and_cache: per-token-head needs k/v token scales");
    }
    if (cache.head_dim_k < 2 || cache.head_dim_v < 2 || (cache.head_dim_k & 1) ||
        (cache.head_dim_v & 1) || cache.head_dim_k > kMaxRowDim ||
        cache.head_dim_v > kMaxRowDim) {
      return detail::set_error(Status::kUnsupportedShape,
                               "reshape_and_cache: per-token-head dims must be even and <= %d",
                               kMaxRowDim);
    }
    if (cache.k_codec == KVCodec::kFP8PerTokenHead) {
      if (!fp8_storage(cache.k_dtype) || fp8_payload_unsupported(cache.k_dtype) ||
          cache.fp8 != Fp8Encoding::kOCP) {
        return detail::set_error(Status::kUnsupportedDtype,
                                 "reshape_and_cache: per-token-head FP8 must be OCP e4m3");
      }
    } else if (cache.k_dtype != DType::kUInt8) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "reshape_and_cache: INT8/INT4 payload dtype must be uint8");
    }
    if (num_tokens == 0) return Status::kSuccess;
    const cudaStream_t stream = stream_of(ctx);
    const int block = 128;
    const dim3 grid(static_cast<unsigned>(num_tokens));
#define KOPS_KV_TOKEN_DTYPE(S)                                                                     \
  do {                                                                                             \
    reshape_and_cache_per_token_quant_kernel<S, uint8_t, KVCodec::kFP8PerTokenHead,                \
                                             KVLayout::kNHD><<<grid, block, 0, stream>>>(           \
        const_ptr<S>(key), const_ptr<S>(value), cache, const_ptr<int64_t>(slot_mapping),           \
        key.stride[0], value.stride[0], static_cast<int>(num_tokens));                             \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("reshape_and_cache per-token-head", err);          \
    return Status::kSuccess;                                                                       \
  } while (0)
    if (key.dtype == DType::kBFloat16) KOPS_KV_TOKEN_DTYPE(__nv_bfloat16);
    if (key.dtype == DType::kFloat16) KOPS_KV_TOKEN_DTYPE(__half);
    if (key.dtype == DType::kFloat32) KOPS_KV_TOKEN_DTYPE(float);
#undef KOPS_KV_TOKEN_DTYPE
    return detail::set_error(Status::kUnsupportedDtype, "reshape_and_cache per-token-head dtype");
  }
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor || cache.k_codec == KVCodec::kFP8PerHead;
  if (fp8) {
    if (!fp8_storage(cache.k_dtype) || fp8_payload_unsupported(cache.k_dtype)) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "reshape_and_cache: only e4m3 FP8 payload is compiled, got %s",
                               dtype_name(cache.k_dtype));
    }
    if (cache.fp8 != Fp8Encoding::kOCP) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "reshape_and_cache: FNUZ encoding is not compiled");
    }
    s = check_scale(cache.k_scale, cache.k_codec, cache.scale_head_stride, "reshape_and_cache.k");
    if (s != Status::kSuccess) return s;
    if (cache.v_data != nullptr) {
      s = check_scale(cache.v_scale, cache.v_codec, cache.scale_head_stride,
                      "reshape_and_cache.v");
      if (s != Status::kSuccess) return s;
    }
  } else if (cache.k_codec == KVCodec::kFP8PerTokenHead ||
             cache.k_codec == KVCodec::kInt8PerTokenHead ||
             cache.k_codec == KVCodec::kInt4PerTokenHead ||
             cache.k_codec == KVCodec::kNVFP4 || cache.k_codec == KVCodec::kLowBitWHT) {
    return detail::set_error(Status::kNotImplemented,
                             "reshape_and_cache: codec %d lands in B3", static_cast<int>(cache.k_codec));
  } else if (cache.k_codec != KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedShape,
                             "reshape_and_cache: codec %d", static_cast<int>(cache.k_codec));
  }
  if (!fp8 && (cache.k_dtype != key.dtype || (cache.v_data != nullptr && cache.v_dtype != key.dtype))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "reshape_and_cache: native cache dtype %s/%s vs source %s",
                             dtype_name(cache.k_dtype), dtype_name(cache.v_dtype),
                             dtype_name(key.dtype));
  }
  if (num_tokens == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);

#define KOPS_KV_RAC_LAUNCH(SCALAR, CACHE_T, CODEC)                                            \
  do {                                                                                         \
    if (cache.layout == KVLayout::kVllmLegacy) {                                               \
      return launch_reshape_legacy<SCALAR, CACHE_T, CODEC>(key, value, cache, slot_mapping,    \
                                                           num_tokens, stream);                \
    }                                                                                          \
    return launch_reshape_flash<SCALAR, CACHE_T, CODEC>(key, value, cache, slot_mapping,       \
                                                        num_tokens, stream);                   \
  } while (0)

#define KOPS_KV_RAC_CODEC(SCALAR, CACHE_T)                                                      \
  do {                                                                                          \
    if (cache.k_codec == KVCodec::kNative) {                                                    \
      KOPS_KV_RAC_LAUNCH(SCALAR, CACHE_T, KVCodec::kNative);                                    \
    }                                                                                           \
    if (cache.k_codec == KVCodec::kFP8PerTensor) {                                              \
      KOPS_KV_RAC_LAUNCH(SCALAR, uint8_t, KVCodec::kFP8PerTensor);                              \
    }                                                                                           \
    if (cache.k_codec == KVCodec::kFP8PerHead) {                                                \
      KOPS_KV_RAC_LAUNCH(SCALAR, uint8_t, KVCodec::kFP8PerHead);                                \
    }                                                                                           \
    return detail::set_error(Status::kNotImplemented, "reshape_and_cache codec");               \
  } while (0)

  if (fp8) {
    switch (key.dtype) {
      case DType::kBFloat16:
        if (cache.layout == KVLayout::kVllmLegacy) {
          return launch_reshape_legacy<__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor>(
              key, value, cache, slot_mapping, num_tokens, stream);
        }
        return launch_reshape_flash<__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor>(
            key, value, cache, slot_mapping, num_tokens, stream);
      case DType::kFloat16:
        if (cache.layout == KVLayout::kVllmLegacy) {
          return launch_reshape_legacy<__half, uint8_t, KVCodec::kFP8PerTensor>(
              key, value, cache, slot_mapping, num_tokens, stream);
        }
        return launch_reshape_flash<__half, uint8_t, KVCodec::kFP8PerTensor>(
            key, value, cache, slot_mapping, num_tokens, stream);
      case DType::kFloat32:
        if (cache.layout == KVLayout::kVllmLegacy) {
          return launch_reshape_legacy<float, uint8_t, KVCodec::kFP8PerTensor>(
              key, value, cache, slot_mapping, num_tokens, stream);
        }
        return launch_reshape_flash<float, uint8_t, KVCodec::kFP8PerTensor>(
            key, value, cache, slot_mapping, num_tokens, stream);
      default:
        return detail::set_error(Status::kUnsupportedDtype, "reshape_and_cache source dtype");
    }
  }

  switch (key.dtype) {
    case DType::kBFloat16: KOPS_KV_RAC_CODEC(__nv_bfloat16, __nv_bfloat16);
    case DType::kFloat16: KOPS_KV_RAC_CODEC(__half, __half);
    case DType::kFloat32: KOPS_KV_RAC_CODEC(float, float);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "reshape_and_cache source dtype");
  }
#undef KOPS_KV_RAC_CODEC
#undef KOPS_KV_RAC_CACHE
}

Status append_paged_kv(const Tensor& append_key, const Tensor& append_value,
                       const PagedKVDesc& cache, const Tensor& batch_indices,
                       const Tensor& positions, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (cache.table_format != PageTableFormat::kCSR || cache.kv_page_indices == nullptr) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "append_paged_kv: needs a CSR page table with kv_page_indices");
  }
  if (!supported_scalar(append_key.dtype) || append_key.dtype != append_value.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "append_paged_kv: key/value dtype");
  }
  s = check_tensor(append_key, append_key.dtype, 3, 16, "append_paged_kv.key");
  if (s != Status::kSuccess) return s;
  s = check_tensor(append_value, append_value.dtype, 3, 16, "append_paged_kv.value");
  if (s != Status::kSuccess) return s;
  s = check_tensor(batch_indices, DType::kInt32, 1, 4, "append_paged_kv.batch_indices");
  if (s != Status::kSuccess) return s;
  s = check_tensor(positions, DType::kInt32, 1, 4, "append_paged_kv.positions");
  if (s != Status::kSuccess) return s;
  if (batch_indices.shape[0] != append_key.shape[0] || positions.shape[0] != append_key.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "append_paged_kv: per-token arrays disagree");
  }
  if (append_key.stride[2] != 1 || append_value.stride[2] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "append_paged_kv: last dim must be contiguous");
  }
  if (cache.v_data != nullptr && cache.v_codec != cache.k_codec) {
    return detail::set_error(Status::kUnsupportedShape,
                             "append_paged_kv: B2 requires K and V to share one codec");
  }
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor || cache.k_codec == KVCodec::kFP8PerHead;
  if (fp8 && (cache.fp8 != Fp8Encoding::kOCP || fp8_payload_unsupported(cache.k_dtype) ||
              (cache.v_data != nullptr && fp8_payload_unsupported(cache.v_dtype)))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "append_paged_kv: only OCP e4m3 is compiled");
  }
  if (!fp8 && (cache.k_dtype != append_key.dtype || cache.v_dtype != append_key.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "append_paged_kv: native dtype mismatch");
  }
  if (append_key.shape[0] == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);

#define KOPS_KV_APPEND(SCALAR, CACHE_T, CODEC)                                                    \
  do {                                                                                            \
    if (cache.layout == KVLayout::kHND) {                                                         \
      return launch_append<SCALAR, CACHE_T, CODEC, KVLayout::kHND>(append_key, append_value,      \
                                                                   cache, batch_indices,          \
                                                                   positions, stream);            \
    }                                                                                             \
    return launch_append<SCALAR, CACHE_T, CODEC, KVLayout::kNHD>(append_key, append_value, cache, \
                                                                 batch_indices, positions, stream); \
  } while (0)

#define KOPS_KV_APPEND_DTYPE(SCALAR)                                                              \
  do {                                                                                            \
    if (fp8) KOPS_KV_APPEND(SCALAR, uint8_t, KVCodec::kFP8PerTensor);                             \
    KOPS_KV_APPEND(SCALAR, SCALAR, KVCodec::kNative);                                             \
  } while (0)

  switch (append_key.dtype) {
    case DType::kBFloat16: KOPS_KV_APPEND_DTYPE(__nv_bfloat16);
    case DType::kFloat16: KOPS_KV_APPEND_DTYPE(__half);
    case DType::kFloat32: KOPS_KV_APPEND_DTYPE(float);
    default: return detail::set_error(Status::kUnsupportedDtype, "append_paged_kv dtype");
  }
#undef KOPS_KV_APPEND_DTYPE
#undef KOPS_KV_APPEND
}

Status fused_qkv_norm_rope_append(const FusedQKVAppendParams& params, DType qkv_dtype,
                                  const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (params.qkv == nullptr || params.positions == nullptr || params.slot_mapping == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "fused_qkv_norm_rope_append: null qkv/positions/slots");
  }
  if (!supported_scalar(qkv_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "fused_qkv_norm_rope_append: dtype %s",
                             dtype_name(qkv_dtype));
  }
  if (params.num_tokens < 0 || params.num_q_heads < 0 || params.num_kv_heads <= 0 ||
      params.head_dim <= 0 || params.rot_dim <= 0 || params.rot_dim % 2 != 0 ||
      params.rot_offset < 0 || params.rot_offset + params.rot_dim > params.head_dim) {
    return detail::set_error(Status::kInvalidArgument,
                             "fused_qkv_norm_rope_append: extents/rotary window");
  }
  s = kv_desc_validate(params.cache);
  if (s != Status::kSuccess) return s;
  if (params.cache.layout != KVLayout::kNHD && params.cache.layout != KVLayout::kHND) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "fused_qkv_norm_rope_append: NHD/HND cache only");
  }
  if (params.cache.num_kv_heads != params.num_kv_heads ||
      params.cache.head_dim_k != params.head_dim ||
      params.cache.head_dim_v != params.head_dim) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fused_qkv_norm_rope_append: cache extents differ from qkv");
  }
  const bool fp8 = params.cache.k_codec == KVCodec::kFP8PerTensor ||
                   params.cache.k_codec == KVCodec::kFP8PerHead;
  if (!fp8 && params.cache.k_codec != KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fused_qkv_norm_rope_append: codec %d",
                             static_cast<int>(params.cache.k_codec));
  }
  if (fp8 && (params.cache.v_codec != params.cache.k_codec ||
              params.cache.fp8 != Fp8Encoding::kOCP ||
              fp8_payload_unsupported(params.cache.k_dtype))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "fused_qkv_norm_rope_append: OCP e4m3 K/V FP8 only");
  }
  if (!fp8 && (params.cache.k_dtype != qkv_dtype || params.cache.v_dtype != qkv_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "fused_qkv_norm_rope_append: native cache dtype mismatch");
  }
  if (params.head_dim != 64 && params.head_dim != 128 && params.head_dim != 256) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fused_qkv_norm_rope_append: head_dim must be 64/128/256");
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  const bool has_norm = params.q_norm_weight != nullptr || params.k_norm_weight != nullptr;
#define KOPS_FUSED_APPEND(S, CACHE_T, CODEC, LAYOUT, D, HAS_N)                                     \
  do {                                                                                             \
    fused_qkv_norm_rope_append_kernel<S, CACHE_T, D, CODEC, LAYOUT, HAS_N>                         \
        <<<static_cast<unsigned>(params.num_tokens), 256, 0, stream>>>(params);                     \
    const cudaError_t err = cudaGetLastError();                                                     \
    if (err != cudaSuccess) return launch_error("fused_qkv_norm_rope_append", err);                 \
    return Status::kSuccess;                                                                        \
  } while (0)
#define KOPS_FUSED_APPEND_LAYOUT(S, CACHE_T, CODEC, D, HAS_N)                                       \
  do {                                                                                              \
    if (params.cache.layout == KVLayout::kHND) {                                                    \
      KOPS_FUSED_APPEND(S, CACHE_T, CODEC, KVLayout::kHND, D, HAS_N);                               \
    } else {                                                                                        \
      KOPS_FUSED_APPEND(S, CACHE_T, CODEC, KVLayout::kNHD, D, HAS_N);                               \
    }                                                                                               \
  } while (0)
#define KOPS_FUSED_APPEND_DTYPE(S)                                                                  \
  do {                                                                                              \
    if (params.cache.k_codec == KVCodec::kNative) {                                                 \
      if (params.head_dim == 64) {                                                                  \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 64, true);                   \
        KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 64, false);                                \
      } else if (params.head_dim == 128) {                                                          \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 128, true);                  \
        KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 128, false);                               \
      } else {                                                                                      \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 256, true);                  \
        KOPS_FUSED_APPEND_LAYOUT(S, S, KVCodec::kNative, 256, false);                               \
      }                                                                                             \
    } else {                                                                                        \
      if (params.head_dim == 64) {                                                                  \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 64, true);       \
        KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 64, false);                    \
      } else if (params.head_dim == 128) {                                                          \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 128, true);      \
        KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 128, false);                   \
      } else {                                                                                      \
        if (has_norm) KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 256, true);      \
        KOPS_FUSED_APPEND_LAYOUT(S, uint8_t, KVCodec::kFP8PerTensor, 256, false);                   \
      }                                                                                             \
    }                                                                                               \
  } while (0)
  if (qkv_dtype == DType::kBFloat16) KOPS_FUSED_APPEND_DTYPE(__nv_bfloat16);
  if (qkv_dtype == DType::kFloat16) KOPS_FUSED_APPEND_DTYPE(__half);
  if (qkv_dtype == DType::kFloat32) KOPS_FUSED_APPEND_DTYPE(float);
#undef KOPS_FUSED_APPEND_DTYPE
#undef KOPS_FUSED_APPEND_LAYOUT
#undef KOPS_FUSED_APPEND
  return detail::set_error(Status::kUnsupportedDtype, "fused_qkv_norm_rope_append dtype");
}

Status ring_kv_append(const Tensor& key, const Tensor& value, const Tensor& ring_k,
                      const Tensor& ring_v, const Tensor& token_slot, const Tensor& positions,
                      const Tensor& k_scale, const Tensor& v_scale, int32_t window,
                      const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(key.dtype) || key.dtype != ring_k.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "ring_kv_append: key/ring dtype");
  }
  s = check_tensor(key, key.dtype, 3, 16, "ring_kv_append.key");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ring_k, ring_k.dtype, 4, 16, "ring_kv_append.ring_k");
  if (s != Status::kSuccess) return s;
  s = check_tensor(token_slot, DType::kInt32, 1, 4, "ring_kv_append.token_slot");
  if (s != Status::kSuccess) return s;
  s = check_tensor(positions, DType::kInt64, 1, 8, "ring_kv_append.positions");
  if (s != Status::kSuccess) return s;
  const bool ring_packed = ring_k.ndim == 4 && ring_k.stride[3] == 1 &&
                           ring_k.stride[2] == ring_k.shape[3] &&
                           ring_k.stride[1] == ring_k.shape[2] * ring_k.shape[3] &&
                           ring_k.stride[0] == ring_k.shape[1] * ring_k.shape[2] * ring_k.shape[3];
  if (key.stride[2] != 1 || !ring_packed) {
    return detail::set_error(Status::kUnsupportedLayout, "ring_kv_append: contiguous views only");
  }
  if (window <= 0 || ring_k.shape[1] != window) {
    return detail::set_error(Status::kInvalidArgument, "ring_kv_append: window %d", window);
  }
  const int64_t tokens = key.shape[0];
  if (token_slot.shape[0] != tokens || positions.shape[0] != tokens) {
    return detail::set_error(Status::kInvalidArgument, "ring_kv_append: per-token arrays disagree");
  }
  const bool has_v = tensor_present(value);
  if (has_v) {
    s = check_tensor(value, value.dtype, 3, 16, "ring_kv_append.value");
    if (s != Status::kSuccess) return s;
    s = check_tensor(ring_v, ring_v.dtype, 4, 16, "ring_kv_append.ring_v");
    if (s != Status::kSuccess) return s;
    if (value.dtype != key.dtype || ring_v.dtype != key.dtype ||
        value.shape[2] != ring_v.shape[3]) {
      return detail::set_error(Status::kUnsupportedDtype, "ring_kv_append: value/ring_v dtype");
    }
  }
  if (tokens == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_RING(SCALAR)                                                                  \
  return launch_ring<SCALAR, SCALAR, KVCodec::kNative>(key, value, ring_k, ring_v, token_slot, \
                                                       positions, k_scale, v_scale, window,   \
                                                       stream)
  switch (key.dtype) {
    case DType::kBFloat16: KOPS_KV_RING(__nv_bfloat16);
    case DType::kFloat16: KOPS_KV_RING(__half);
    case DType::kFloat32: KOPS_KV_RING(float);
    default: return detail::set_error(Status::kUnsupportedDtype, "ring_kv_append dtype");
  }
#undef KOPS_KV_RING
}

Status concat_and_cache_mla(const Tensor& kv_c, const Tensor& k_pe, const PagedKVDesc& cache,
                            const Tensor& slot_mapping, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(kv_c.dtype) || (tensor_present(k_pe) && k_pe.dtype != kv_c.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla: kv_c/k_pe dtype");
  }
  s = check_tensor(kv_c, kv_c.dtype, 2, 16, "concat_and_cache_mla.kv_c");
  if (s != Status::kSuccess) return s;
  if (tensor_present(k_pe)) {
    s = check_tensor(k_pe, k_pe.dtype, 2, 16, "concat_and_cache_mla.k_pe");
    if (s != Status::kSuccess) return s;
  }
  if (kv_c.stride[1] != 1 || (tensor_present(k_pe) && k_pe.stride[1] != 1)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "concat_and_cache_mla: last dim must be contiguous");
  }
  int64_t num_tokens = 0;
  s = check_slot_mapping(slot_mapping, &num_tokens);
  if (s != Status::kSuccess) return s;
  if (kv_c.shape[0] < num_tokens || (tensor_present(k_pe) && k_pe.shape[0] < num_tokens)) {
    return detail::set_error(Status::kInvalidArgument, "concat_and_cache_mla: source too short");
  }
  const bool record_codec = cache.k_codec == KVCodec::kMLAFp8DS ||
                            cache.k_codec == KVCodec::kMLAFp8DSV40 ||
                            cache.k_codec == KVCodec::kMLAFp8DSV41 ||
                            cache.k_codec == KVCodec::kMLAFp8DSV41Fp4;
  if (record_codec) {
    const int rank = static_cast<int>(kv_c.shape[1]);
    const int pe = tensor_present(k_pe) ? static_cast<int>(k_pe.shape[1]) : 0;
    const int want_rank = cache.k_codec == KVCodec::kMLAFp8DS ? 512 : 448;
    if (!((rank == want_rank && pe == 64) || (rank == want_rank && pe == 0))) {
      return detail::set_error(Status::kUnsupportedShape,
                               "concat_and_cache_mla: codec %d needs kv_c %d + pe 64 (or NoPE), "
                               "got %d + %d",
                               static_cast<int>(cache.k_codec), want_rank, rank, pe);
    }
    if (num_tokens == 0) return Status::kSuccess;
    const cudaStream_t stream = stream_of(ctx);
    const bool v40 = cache.k_codec == KVCodec::kMLAFp8DSV40;
    const int64_t block_stride =
        cache.k_stride_page > 0 ? cache.k_stride_page
                                : static_cast<int64_t>(cache.page_size) * cache.entry_bytes;
    const int64_t entry_stride =
        cache.k_codec == KVCodec::kMLAFp8DS
            ? cache.entry_bytes
            : (cache.k_stride_token > 0 ? cache.k_stride_token : cache.entry_bytes);
    const int64_t scale_block_stride = v40 ? cache.k_scale_stride_page : 0;
    const int64_t scale_token_stride = v40 ? cache.k_scale_stride_token : 0;
    const int block = cache.k_codec == KVCodec::kMLAFp8DS ? 96 : 256;
    if (kv_c.stride[0] > INT32_MAX || (tensor_present(k_pe) && k_pe.stride[0] > INT32_MAX)) {
      return detail::set_error(Status::kCapacityExceeded, "concat_and_cache_mla: strides");
    }
#define KOPS_KV_DS_MLA(S, CODEC)                                                                   \
  do {                                                                                             \
    concat_and_cache_ds_mla_kernel<S, CODEC><<<static_cast<unsigned>(num_tokens), block, 0,        \
                                               stream>>>(                                          \
        const_ptr<S>(kv_c), tensor_present(k_pe) ? const_ptr<S>(k_pe) : nullptr,                    \
        reinterpret_cast<uint8_t*>(cache.k_data),                                                   \
        v40 ? reinterpret_cast<uint8_t*>(cache.k_scale_pages) : nullptr,                            \
        const_ptr<int64_t>(slot_mapping), block_stride, entry_stride, scale_block_stride,           \
        scale_token_stride, static_cast<int>(kv_c.stride[0]),                                      \
        tensor_present(k_pe) ? static_cast<int>(k_pe.stride[0]) : 0, rank, pe, cache.page_size);     \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("concat_and_cache_ds_mla", err);                   \
    return Status::kSuccess;                                                                       \
  } while (0)
#define KOPS_KV_DS_MLA_DTYPE(S)                                                                    \
  do {                                                                                             \
    if (cache.k_codec == KVCodec::kMLAFp8DS) KOPS_KV_DS_MLA(S, KVCodec::kMLAFp8DS);                \
    if (cache.k_codec == KVCodec::kMLAFp8DSV40) KOPS_KV_DS_MLA(S, KVCodec::kMLAFp8DSV40);          \
    if (cache.k_codec == KVCodec::kMLAFp8DSV41) KOPS_KV_DS_MLA(S, KVCodec::kMLAFp8DSV41);          \
    KOPS_KV_DS_MLA(S, KVCodec::kMLAFp8DSV41Fp4);                                                   \
  } while (0)
    if (kv_c.dtype == DType::kBFloat16) KOPS_KV_DS_MLA_DTYPE(__nv_bfloat16);
    if (kv_c.dtype == DType::kFloat16) KOPS_KV_DS_MLA_DTYPE(__half);
    if (kv_c.dtype == DType::kFloat32) KOPS_KV_DS_MLA_DTYPE(float);
#undef KOPS_KV_DS_MLA_DTYPE
#undef KOPS_KV_DS_MLA
    return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_ds_mla dtype");
  }
  if (cache.k_codec == KVCodec::kMLAFp8DSFNUZ || cache.k_codec == KVCodec::kMLANvfp4DS) {
    return detail::set_error(Status::kNotImplemented,
                             "concat_and_cache_mla: codec %d needs the AMD / Blackwell record",
                             static_cast<int>(cache.k_codec));
  }
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor || cache.k_codec == KVCodec::kFP8PerHead;
  if (!fp8 && cache.k_dtype != kv_c.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla: native dtype mismatch");
  }
  if (fp8) {
    if (cache.fp8 != Fp8Encoding::kOCP || fp8_payload_unsupported(cache.k_dtype)) {
      return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla: e4m3 OCP only");
    }
    s = check_scale(cache.k_scale, cache.k_codec, cache.scale_head_stride,
                    "concat_and_cache_mla");
    if (s != Status::kSuccess) return s;
  }
  if (num_tokens == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_MLA(SCALAR, CACHE_T, CODEC) \
  return launch_concat_mla<SCALAR, CACHE_T, CODEC>(kv_c, k_pe, cache, slot_mapping, num_tokens, stream)
  if (fp8) {
    switch (kv_c.dtype) {
      case DType::kBFloat16: KOPS_KV_MLA(__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor);
      case DType::kFloat16: KOPS_KV_MLA(__half, uint8_t, KVCodec::kFP8PerTensor);
      case DType::kFloat32: KOPS_KV_MLA(float, uint8_t, KVCodec::kFP8PerTensor);
      default: return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla dtype");
    }
  }
  switch (kv_c.dtype) {
    case DType::kBFloat16: KOPS_KV_MLA(__nv_bfloat16, __nv_bfloat16, KVCodec::kNative);
    case DType::kFloat16: KOPS_KV_MLA(__half, __half, KVCodec::kNative);
    case DType::kFloat32: KOPS_KV_MLA(float, float, KVCodec::kNative);
    default: return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla dtype");
  }
#undef KOPS_KV_MLA
}

Status concat_and_cache_mla_grouped(const Tensor& kv_c, const Tensor& k_pe,
                                    const Tensor& cache_ptrs, const Tensor& scales,
                                    const Tensor& slot_mapping, const PagedKVDesc& cache_template,
                                    const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache_template);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(kv_c.dtype) || (tensor_present(k_pe) && k_pe.dtype != kv_c.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla_grouped: dtype");
  }
  s = check_tensor(kv_c, kv_c.dtype, 3, 16, "concat_and_cache_mla_grouped.kv_c");
  if (s != Status::kSuccess) return s;
  if (tensor_present(k_pe)) {
    s = check_tensor(k_pe, k_pe.dtype, 3, 16, "concat_and_cache_mla_grouped.k_pe");
    if (s != Status::kSuccess) return s;
  }
  s = check_tensor(cache_ptrs, DType::kInt64, 1, 8, "concat_and_cache_mla_grouped.cache_ptrs");
  if (s != Status::kSuccess) return s;
  s = check_tensor(slot_mapping, DType::kInt64, 2, 8, "concat_and_cache_mla_grouped.slots");
  if (s != Status::kSuccess) return s;
  if (tensor_present(scales)) {
    s = check_tensor(scales, DType::kFloat32, 1, 4, "concat_and_cache_mla_grouped.scales");
    if (s != Status::kSuccess) return s;
  }
  const int64_t layers = kv_c.shape[0];
  if (cache_ptrs.shape[0] != layers || slot_mapping.shape[0] != layers) {
    return detail::set_error(Status::kInvalidArgument, "concat_and_cache_mla_grouped: layer counts");
  }
  if (slot_mapping.stride[1] != 1 || slot_mapping.stride[0] < slot_mapping.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "concat_and_cache_mla_grouped: slots must be row-major packed");
  }
  const int rank = static_cast<int>(kv_c.shape[2]);
  const int pe = tensor_present(k_pe) ? static_cast<int>(k_pe.shape[2]) : 0;
  if (rank <= 0 || layers <= 0) return Status::kSuccess;
  const bool fp8 = cache_template.k_codec == KVCodec::kFP8PerTensor ||
                   cache_template.k_codec == KVCodec::kFP8PerHead;
  if (!fp8 && cache_template.k_codec != KVCodec::kNative) {
    return detail::set_error(Status::kNotImplemented,
                             "concat_and_cache_mla_grouped: codec %d lands in B3",
                             static_cast<int>(cache_template.k_codec));
  }
  if (fp8 && (cache_template.fp8 != Fp8Encoding::kOCP ||
              fp8_payload_unsupported(cache_template.k_dtype))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "concat_and_cache_mla_grouped: e4m3 OCP only");
  }
  if (!fp8 && cache_template.k_dtype != kv_c.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "concat_and_cache_mla_grouped: native dtype mismatch");
  }
  const cudaStream_t stream = stream_of(ctx);
  const int block = threads_for(rank, 32, 512);
  const dim3 grid(static_cast<unsigned>(slot_mapping.shape[1]), static_cast<unsigned>(layers));
#define KOPS_KV_MLA_GROUPED(SCALAR, CACHE_T, CODEC)                                                \
  do {                                                                                             \
    PagedKVDesc e{};                                                                               \
    const Status es = element_strides<CACHE_T>(                                                    \
        cache_template, rank + pe, static_cast<int64_t>(rank + pe) * cache_template.page_size,     \
        rank + pe, &e);                                                                            \
    if (es != Status::kSuccess) return es;                                                         \
    concat_and_cache_mla_grouped_kernel<SCALAR, CACHE_T, CODEC><<<grid, block, 0, stream>>>(       \
        const_ptr<SCALAR>(kv_c), tensor_present(k_pe) ? const_ptr<SCALAR>(k_pe) : nullptr,          \
        const_ptr<int64_t>(cache_ptrs),                                                             \
        tensor_present(scales) ? const_ptr<float>(scales) : nullptr,                                \
        const_ptr<int64_t>(slot_mapping), kv_c.stride[0], kv_c.stride[1],                           \
        tensor_present(k_pe) ? k_pe.stride[0] : 0, tensor_present(k_pe) ? k_pe.stride[1] : 0,       \
        slot_mapping.stride[0], e.k_stride_page, e.k_stride_token, rank, pe,                        \
        cache_template.page_size);                                                                   \
    const cudaError_t err = cudaGetLastError();                                                     \
    if (err != cudaSuccess) return launch_error("concat_and_cache_mla_grouped", err);               \
    return Status::kSuccess;                                                                        \
  } while (0)
  if (fp8) {
    switch (kv_c.dtype) {
      case DType::kBFloat16: KOPS_KV_MLA_GROUPED(__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor);
      case DType::kFloat16: KOPS_KV_MLA_GROUPED(__half, uint8_t, KVCodec::kFP8PerTensor);
      case DType::kFloat32: KOPS_KV_MLA_GROUPED(float, uint8_t, KVCodec::kFP8PerTensor);
      default: return detail::set_error(Status::kUnsupportedDtype, "mla grouped dtype");
    }
  }
  switch (kv_c.dtype) {
    case DType::kBFloat16: KOPS_KV_MLA_GROUPED(__nv_bfloat16, __nv_bfloat16, KVCodec::kNative);
    case DType::kFloat16: KOPS_KV_MLA_GROUPED(__half, __half, KVCodec::kNative);
    case DType::kFloat32: KOPS_KV_MLA_GROUPED(float, float, KVCodec::kNative);
    default: return detail::set_error(Status::kUnsupportedDtype, "mla grouped dtype");
  }
#undef KOPS_KV_MLA_GROUPED
}

Status concat_and_cache_mla_rope_fused(const Tensor& positions, const Tensor& q_pe,
                                       const Tensor& k_pe, const Tensor& kv_c,
                                       const Tensor& cos_sin_cache, RopeStyle rope_style,
                                       const Tensor& slot_mapping, const PagedKVDesc& cache,
                                       const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(q_pe.dtype) || k_pe.dtype != q_pe.dtype || kv_c.dtype != q_pe.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "concat_and_cache_mla_rope_fused: q_pe/k_pe/kv_c dtype");
  }
  s = check_tensor(q_pe, q_pe.dtype, 3, 1, "concat_and_cache_mla_rope_fused.q_pe");
  if (s != Status::kSuccess) return s;
  s = check_tensor(k_pe, k_pe.dtype, 2, 1, "concat_and_cache_mla_rope_fused.k_pe");
  if (s != Status::kSuccess) return s;
  s = check_tensor(kv_c, kv_c.dtype, 2, 1, "concat_and_cache_mla_rope_fused.kv_c");
  if (s != Status::kSuccess) return s;
  s = check_tensor(positions, DType::kInt64, 1, 8, "concat_and_cache_mla_rope_fused.positions");
  if (s != Status::kSuccess) return s;
  s = check_tensor(cos_sin_cache, DType::kFloat32, 2, 1,
                   "concat_and_cache_mla_rope_fused.cos_sin_cache");
  if (s != Status::kSuccess) return s;
  if (q_pe.stride[2] != 1 || k_pe.stride[1] != 1 || kv_c.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "concat_and_cache_mla_rope_fused: contiguous last dims");
  }
  int64_t num_tokens = 0;
  s = check_slot_mapping(slot_mapping, &num_tokens);
  if (s != Status::kSuccess) return s;
  const int64_t heads = q_pe.shape[1];
  const int64_t rot = q_pe.shape[2];
  const int64_t rank = kv_c.shape[1];
  if (k_pe.shape[0] < num_tokens || kv_c.shape[0] < num_tokens || q_pe.shape[0] < num_tokens ||
      positions.shape[0] < num_tokens || k_pe.shape[1] != rot || rot <= 0 || rot % 2 != 0 ||
      rot > 512 || cos_sin_cache.shape[1] != rot) {
    return detail::set_error(Status::kUnsupportedShape,
                             "concat_and_cache_mla_rope_fused: rotary window/rows");
  }
  if (num_tokens == 0) return Status::kSuccess;
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor;
  if (!fp8 && cache.k_codec != KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedShape,
                             "concat_and_cache_mla_rope_fused: codec %d",
                             static_cast<int>(cache.k_codec));
  }
  if (fp8) {
    if (cache.fp8 != Fp8Encoding::kOCP || fp8_payload_unsupported(cache.k_dtype) ||
        cache.k_scale == nullptr) {
      return detail::set_error(Status::kUnsupportedDtype,
                               "concat_and_cache_mla_rope_fused: OCP e4m3 + scale");
    }
  } else if (cache.k_dtype != q_pe.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "concat_and_cache_mla_rope_fused: native dtype mismatch");
  }
  const int64_t entry = fp8 ? cache.entry_bytes : rank + rot;
  if (entry != rank + rot) {
    return detail::set_error(Status::kUnsupportedShape,
                             "concat_and_cache_mla_rope_fused: entry width %lld vs %lld",
                             static_cast<long long>(entry), static_cast<long long>(rank + rot));
  }
  int64_t block_stride = cache.k_stride_page;
  int64_t entry_stride = cache.k_stride_token;
  if (block_stride == 0) block_stride = entry * cache.page_size;
  if (entry_stride == 0) entry_stride = entry;
  if (block_stride > INT32_MAX || entry_stride > INT32_MAX || q_pe.stride[0] > INT64_MAX ||
      k_pe.stride[0] > INT32_MAX || kv_c.stride[0] > INT32_MAX || heads > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "concat_and_cache_mla_rope_fused: extents");
  }
  const cudaStream_t stream = stream_of(ctx);
  const int block = threads_for(heads * rot / 2, 64, 512);
#define KOPS_MLA_ROPE_FUSED(QK, CACHE_T, CODEC, NEOX, RAW)                                          \
  do {                                                                                             \
    concat_and_cache_mla_rope_fused_kernel<QK, float, NEOX, RAW, CACHE_T, CODEC>                   \
        <<<static_cast<unsigned>(num_tokens), block, 0, stream>>>(                                 \
            const_ptr<int64_t>(positions), mut_ptr<QK>(q_pe), mut_ptr<QK>(k_pe),                   \
            const_ptr<QK>(kv_c), const_ptr<float>(cos_sin_cache), static_cast<int>(rot),           \
            q_pe.stride[0], q_pe.stride[1], k_pe.stride[0], kv_c.stride[0],                        \
            static_cast<int>(heads), static_cast<CACHE_T*>(cache.k_data),                          \
            const_ptr<int64_t>(slot_mapping), static_cast<int>(block_stride),                      \
            static_cast<int>(entry_stride), static_cast<int>(rank), cache.page_size,               \
            cache.k_scale);                                                                        \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("concat_and_cache_mla_rope_fused", err);           \
    return Status::kSuccess;                                                                       \
  } while (0)
#define KOPS_MLA_ROPE_STYLE(QK, CACHE_T, CODEC)                                                     \
  do {                                                                                             \
    if (rope_style == RopeStyle::kNeoX) {                                                          \
      KOPS_MLA_ROPE_FUSED(QK, CACHE_T, CODEC, true, uint16_t);                                     \
    } else {                                                                                       \
      KOPS_MLA_ROPE_FUSED(QK, CACHE_T, CODEC, false, uint16_t);                                    \
    }                                                                                              \
  } while (0)
  if (fp8) {
    if (q_pe.dtype == DType::kBFloat16) KOPS_MLA_ROPE_STYLE(__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor);
    if (q_pe.dtype == DType::kFloat16) KOPS_MLA_ROPE_STYLE(__half, uint8_t, KVCodec::kFP8PerTensor);
    if (q_pe.dtype == DType::kFloat32) KOPS_MLA_ROPE_STYLE(float, uint8_t, KVCodec::kFP8PerTensor);
  } else {
    if (q_pe.dtype == DType::kBFloat16) KOPS_MLA_ROPE_STYLE(__nv_bfloat16, __nv_bfloat16, KVCodec::kNative);
    if (q_pe.dtype == DType::kFloat16) KOPS_MLA_ROPE_STYLE(__half, __half, KVCodec::kNative);
    if (q_pe.dtype == DType::kFloat32) KOPS_MLA_ROPE_STYLE(float, float, KVCodec::kNative);
  }
#undef KOPS_MLA_ROPE_STYLE
#undef KOPS_MLA_ROPE_FUSED
  return detail::set_error(Status::kUnsupportedDtype, "concat_and_cache_mla_rope_fused dtype");
}

Status indexer_k_quant_and_cache(const Tensor& k, const PagedKVDesc& index_cache,
                                 const Tensor& slot_mapping, int32_t quant_block_size,
                                 bool scale_ue8m0, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(index_cache);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(k.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "indexer_k_quant_and_cache: k dtype");
  }
  s = check_tensor(k, k.dtype, 2, 1, "indexer_k_quant_and_cache.k");
  if (s != Status::kSuccess) return s;
  if (k.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "indexer_k_quant_and_cache: contiguous last dim");
  }
  if (index_cache.k_codec != KVCodec::kIndexerFP8) {
    return detail::set_error(Status::kUnsupportedShape,
                             "indexer_k_quant_and_cache: kIndexerFP8 cache required");
  }
  if (quant_block_size != 32 && quant_block_size != 64 && quant_block_size != 128) {
    return detail::set_error(Status::kUnsupportedShape,
                             "indexer_k_quant_and_cache: quant_block_size 32/64/128");
  }
  const int64_t head_dim = k.shape[1];
  if (head_dim <= 0 || head_dim % quant_block_size != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "indexer_k_quant_and_cache: head_dim %% quant_block_size");
  }
  const int64_t groups = head_dim / quant_block_size;
  if (index_cache.entry_bytes != head_dim + groups * 4) {
    return detail::set_error(Status::kUnsupportedShape,
                             "indexer_k_quant_and_cache: entry_bytes %d vs %lld",
                             index_cache.entry_bytes,
                             static_cast<long long>(head_dim + groups * 4));
  }
  int64_t num_tokens = 0;
  s = check_slot_mapping(slot_mapping, &num_tokens);
  if (s != Status::kSuccess) return s;
  if (k.shape[0] < num_tokens) {
    return detail::set_error(Status::kInvalidArgument, "indexer_k_quant_and_cache: source rows");
  }
  if (num_tokens == 0) return Status::kSuccess;
  const int rows = static_cast<int>(std::min<int64_t>(4, (head_dim + 127) / 128));
  const dim3 grid(static_cast<unsigned>(num_tokens),
                  static_cast<unsigned>((head_dim + 128 * rows - 1) / (128 * rows)));
  const dim3 block(32, rows);
  const cudaStream_t stream = stream_of(ctx);
  const int64_t block_stride = index_cache.k_stride_page > 0
                                   ? index_cache.k_stride_page
                                   : static_cast<int64_t>(index_cache.page_size) *
                                         index_cache.entry_bytes;
#define KOPS_INDEXER_WRITE(T)                                                                      \
  do {                                                                                             \
    indexer_k_quant_and_cache_kernel<T><<<grid, block, 0, stream>>>(                               \
        const_ptr<T>(k), static_cast<uint8_t*>(index_cache.k_data),                                \
        const_ptr<int64_t>(slot_mapping), static_cast<int>(head_dim), quant_block_size,            \
        index_cache.page_size, block_stride, scale_ue8m0);                                         \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("indexer_k_quant_and_cache", err);                 \
    return Status::kSuccess;                                                                       \
  } while (0)
  if (k.dtype == DType::kBFloat16) KOPS_INDEXER_WRITE(__nv_bfloat16);
  if (k.dtype == DType::kFloat16) KOPS_INDEXER_WRITE(__half);
  if (k.dtype == DType::kFloat32) KOPS_INDEXER_WRITE(float);
#undef KOPS_INDEXER_WRITE
  return detail::set_error(Status::kUnsupportedDtype, "indexer_k_quant_and_cache dtype");
}

// -----------------------------------------------------------------------------
// K02 gathers
// -----------------------------------------------------------------------------

template <typename scalar_t, typename cache_t, KVCodec kCodec, int ENTRY>
Status launch_gather_dequant(const Tensor& dst, const PagedKVDesc& cache, const GatherSpec& spec,
                             const float* scale, int32_t block_size, cudaStream_t stream) {
  PagedKVDesc e{};
  Status s = element_strides<cache_t>(cache, ENTRY, static_cast<int64_t>(ENTRY) * block_size,
                                      ENTRY, &e);
  if (s != Status::kSuccess) return s;
  const int64_t grid = ceil_div(spec.num_tokens, 8) + 2 * static_cast<int64_t>(spec.num_reqs);
  gather_and_maybe_dequant_cache_kernel<scalar_t, cache_t, kCodec, ENTRY>
      <<<static_cast<int>(grid), 256, 0, stream>>>(
          static_cast<const cache_t*>(cache.k_data), mut_ptr<scalar_t>(dst), spec, block_size,
          e.k_stride_page, e.k_stride_token, dst.stride[0], scale);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("gather_and_maybe_dequant_cache", err);
  return Status::kSuccess;
}

Status gather_and_maybe_dequant_cache(const Tensor& dst, const PagedKVDesc& cache,
                                      const GatherSpec& spec, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(dst.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "gather_and_maybe_dequant_cache dtype");
  }
  s = check_tensor(dst, dst.dtype, 2, 16, "gather_and_maybe_dequant_cache.dst");
  if (s != Status::kSuccess) return s;
  if (dst.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "gather_and_maybe_dequant_cache: last dim must be contiguous");
  }
  if (spec.block_table == nullptr || spec.cu_seq_lens == nullptr || spec.num_reqs <= 0 ||
      spec.block_table_stride <= 0 || spec.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "gather_and_maybe_dequant_cache: incomplete GatherSpec");
  }
  if (cache.table_format == PageTableFormat::kBlockTable && cache.block_table == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "gather_and_maybe_dequant_cache: descriptor has no block table");
  }
  if (cache.k_codec != KVCodec::kNative && cache.k_codec != KVCodec::kFP8PerTensor &&
      cache.k_codec != KVCodec::kFP8PerHead) {
    return detail::set_error(Status::kUnsupportedShape,
                             "gather_and_maybe_dequant_cache: codec %d",
                             static_cast<int>(cache.k_codec));
  }
  const int entry = static_cast<int>(dst.shape[1]);
  const cudaStream_t stream = stream_of(ctx);
  const int block_size = cache.page_size;
#define KOPS_KV_GATHER_ENTRY(SCALAR, CACHE_T, CODEC)                                                 \
  do {                                                                                                \
    switch (entry) {                                                                                  \
      case 576: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 576>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 512: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 512>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 320: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 320>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 256: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 256>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 192: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 192>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 128: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 128>(dst, cache, spec, cache.k_scale,    \
                                                                          block_size, stream);        \
      case 96: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 96>(dst, cache, spec, cache.k_scale,      \
                                                                        block_size, stream);          \
      case 64: return launch_gather_dequant<SCALAR, CACHE_T, CODEC, 64>(dst, cache, spec, cache.k_scale,      \
                                                                        block_size, stream);          \
      default:                                                                                        \
        return detail::set_error(Status::kUnsupportedShape,                                           \
                                 "gather_and_maybe_dequant_cache: ENTRY_SIZE %d", entry);             \
    }                                                                                                 \
  } while (0)

  if (cache.k_codec == KVCodec::kNative) {
    if (cache.k_dtype == DType::kBFloat16) {
      if (dst.dtype == DType::kBFloat16) KOPS_KV_GATHER_ENTRY(__nv_bfloat16, __nv_bfloat16, KVCodec::kNative);
      if (dst.dtype == DType::kFloat16) KOPS_KV_GATHER_ENTRY(__half, __nv_bfloat16, KVCodec::kNative);
      if (dst.dtype == DType::kFloat32) KOPS_KV_GATHER_ENTRY(float, __nv_bfloat16, KVCodec::kNative);
    } else if (cache.k_dtype == DType::kFloat16) {
      if (dst.dtype == DType::kBFloat16) KOPS_KV_GATHER_ENTRY(__nv_bfloat16, __half, KVCodec::kNative);
      if (dst.dtype == DType::kFloat16) KOPS_KV_GATHER_ENTRY(__half, __half, KVCodec::kNative);
      if (dst.dtype == DType::kFloat32) KOPS_KV_GATHER_ENTRY(float, __half, KVCodec::kNative);
    } else if (cache.k_dtype == DType::kFloat32) {
      if (dst.dtype == DType::kBFloat16) KOPS_KV_GATHER_ENTRY(__nv_bfloat16, float, KVCodec::kNative);
      if (dst.dtype == DType::kFloat16) KOPS_KV_GATHER_ENTRY(__half, float, KVCodec::kNative);
      if (dst.dtype == DType::kFloat32) KOPS_KV_GATHER_ENTRY(float, float, KVCodec::kNative);
    }
  } else {
    if (dst.dtype == DType::kBFloat16) KOPS_KV_GATHER_ENTRY(__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor);
    if (dst.dtype == DType::kFloat16) KOPS_KV_GATHER_ENTRY(__half, uint8_t, KVCodec::kFP8PerTensor);
    if (dst.dtype == DType::kFloat32) KOPS_KV_GATHER_ENTRY(float, uint8_t, KVCodec::kFP8PerTensor);
  }
#undef KOPS_KV_GATHER_ENTRY
  return detail::set_error(Status::kUnsupportedDtype,
                           "gather_and_maybe_dequant_cache: cache %s -> dst %s",
                           dtype_name(cache.k_dtype), dtype_name(dst.dtype));
}

Status cp_gather_cache(const Tensor& dst, const Tensor& src_cache, const GatherSpec& spec,
                       const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(dst, dst.dtype, 2, 1, "cp_gather_cache.dst");
  if (s != Status::kSuccess) return s;
  s = check_tensor(src_cache, src_cache.dtype, 3, 1, "cp_gather_cache.src");
  if (s != Status::kSuccess) return s;
  if (spec.block_table == nullptr || spec.cu_seq_lens == nullptr || spec.num_reqs <= 0 ||
      spec.block_table_stride <= 0 || spec.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument, "cp_gather_cache: incomplete GatherSpec");
  }
  const int64_t dst_elem = (dtype_bits(dst.dtype) + 7) / 8;
  const int64_t src_elem = (dtype_bits(src_cache.dtype) + 7) / 8;
  if (dst_elem <= 0 || src_elem <= 0) {
    return detail::set_error(Status::kUnsupportedDtype, "cp_gather_cache dtype");
  }
  const int64_t entry_bytes = dst.shape[1] * dst_elem;
  if (entry_bytes != src_cache.shape[2] * src_elem) {
    return detail::set_error(Status::kInvalidArgument, "cp_gather_cache: record width mismatch");
  }
  const int32_t block_size = static_cast<int32_t>(src_cache.shape[1]);
  const int64_t dst_stride = dst.stride[0] * dst_elem;
  const int64_t src_block_stride = src_cache.stride[0] * src_elem;
  const int64_t src_entry_stride = src_cache.stride[1] * src_elem;
  if (entry_bytes <= 0) return Status::kSuccess;
  const int64_t grid = ceil_div(spec.num_tokens, 8) + 2 * static_cast<int64_t>(spec.num_reqs);
  const bool vectorized = (entry_bytes % 16) == 0;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(const_ptr<uint8_t>(src_cache));
  uint8_t* out = reinterpret_cast<uint8_t*>(mut_ptr<uint8_t>(dst));
  const cudaStream_t stream = stream_of(ctx);
  if (vectorized) {
    cp_gather_cache_kernel<true, true><<<static_cast<int>(grid), 256, 0, stream>>>(
        src, out, spec, block_size, static_cast<int32_t>(entry_bytes), src_block_stride,
        src_entry_stride, dst_stride);
  } else {
    cp_gather_cache_kernel<true, false><<<static_cast<int>(grid), 256, 0, stream>>>(
        src, out, spec, block_size, static_cast<int32_t>(entry_bytes), src_block_stride,
        src_entry_stride, dst_stride);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("cp_gather_cache", err);
  return Status::kSuccess;
}

Status cp_gather_and_upconvert_ds_mla(const Tensor& dst, const PagedKVDesc& cache,
                                      const GatherSpec& spec, const Tensor& host_cache,
                                      const Tensor& host_row_ids, const Tensor& device_row_ids,
                                      const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (dst.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "cp_gather_and_upconvert_ds_mla: dst must be bfloat16");
  }
  s = check_tensor(dst, DType::kBFloat16, 2, 1, "cp_gather_and_upconvert_ds_mla.dst");
  if (s != Status::kSuccess) return s;
  const kops::kv::MLAKVFormat fmt = mla_kv_format(cache.k_codec);
  const bool record = fmt.data_bytes != 0;
  if (!record) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_and_upconvert_ds_mla: codec %d is not an MLA record",
                             static_cast<int>(cache.k_codec));
  }
  const int logical_cols = cache.k_codec == KVCodec::kMLAFp8DS ? 576 : 512;
  if (dst.stride[1] != 1 ||
      (dst.shape[1] != 576 && dst.shape[1] != 512) || dst.shape[1] < logical_cols) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_and_upconvert_ds_mla: dst must be [T, %d (or 576)]",
                             logical_cols);
  }
  if (cache.entry_bytes != fmt.data_bytes + (fmt.inline_scales ? fmt.scale_bytes : 0)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_and_upconvert_ds_mla: codec %d record width",
                             static_cast<int>(cache.k_codec));
  }
  if (spec.block_table == nullptr || spec.cu_seq_lens == nullptr || spec.num_reqs <= 0 ||
      spec.block_table_stride <= 0 || spec.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "cp_gather_and_upconvert_ds_mla: incomplete GatherSpec");
  }
  const bool any_host = host_cache.data != nullptr || host_row_ids.data != nullptr ||
                        device_row_ids.data != nullptr;
  const bool all_host = host_cache.data != nullptr && host_row_ids.data != nullptr &&
                        device_row_ids.data != nullptr;
  if (any_host && !all_host) {
    return detail::set_error(Status::kInvalidArgument,
                             "cp_gather_and_upconvert_ds_mla: host_cache/host_row_ids/"
                             "device_row_ids must be given together");
  }
  const uint8_t* host_ptr = nullptr;
  int64_t host_stride = 0;
  if (all_host) {
    s = check_tensor(host_cache, DType::kUInt8, 2, 1, "cp_gather_and_upconvert_ds_mla.host_cache");
    if (s != Status::kSuccess) return s;
    s = check_tensor(host_row_ids, DType::kInt32, 1, 4,
                     "cp_gather_and_upconvert_ds_mla.host_row_ids");
    if (s != Status::kSuccess) return s;
    s = check_tensor(device_row_ids, DType::kInt32, 1, 4,
                     "cp_gather_and_upconvert_ds_mla.device_row_ids");
    if (s != Status::kSuccess) return s;
    if (host_cache.device >= 0) {
      return detail::set_error(Status::kInvalidArgument,
                               "cp_gather_and_upconvert_ds_mla: host_cache must be host memory");
    }
    if (host_cache.shape[1] != cache.entry_bytes ||
        host_row_ids.shape[0] < static_cast<int64_t>(cache.page_size)) {
      return detail::set_error(Status::kUnsupportedShape,
                               "cp_gather_and_upconvert_ds_mla: host table extents");
    }
    void* device_ptr = nullptr;
    const cudaError_t herr =
        cudaHostGetDevicePointer(&device_ptr, const_cast<void*>(host_cache.data), 0);
    host_ptr = herr == cudaSuccess ? static_cast<const uint8_t*>(device_ptr)
                                   : const_ptr<uint8_t>(host_cache);
    host_stride = host_cache.stride[0];
  }
  if (spec.num_tokens == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  const int64_t block_stride = cache.k_stride_page > 0
                                   ? cache.k_stride_page
                                   : static_cast<int64_t>(cache.page_size) * cache.entry_bytes;
  const int64_t entry_stride = cache.k_stride_token > 0 ? cache.k_stride_token
                                                        : cache.entry_bytes;
  const int grid = static_cast<int>(ceil_div(spec.num_tokens, 8));
  const uint8_t* scale_pages = cache.k_scale_pages != nullptr
                                   ? static_cast<const uint8_t*>(cache.k_scale_pages)
                                   : nullptr;
  const int64_t scale_block_stride = cache.k_scale_stride_page;
  const int64_t scale_token_stride = cache.k_scale_stride_token;
  const int dst_cols = static_cast<int>(dst.shape[1]);
#define KOPS_KV_DS_GATHER(CODEC)                                                                   \
  do {                                                                                             \
    cp_gather_and_upconvert_ds_mla_fp8_kernel<CODEC><<<grid, 256, 0, stream>>>(                    \
        static_cast<const uint8_t*>(cache.k_data), mut_ptr<bf16_t>(dst), spec, cache.page_size,    \
        block_stride, entry_stride, dst.stride[0], host_ptr,                                       \
        all_host ? const_ptr<int32_t>(host_row_ids) : nullptr,                                     \
        all_host ? const_ptr<int32_t>(device_row_ids) : nullptr, host_stride, scale_pages,         \
        scale_block_stride, scale_token_stride, dst_cols);                                         \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("cp_gather_and_upconvert_ds_mla", err);            \
    return Status::kSuccess;                                                                       \
  } while (0)
  switch (cache.k_codec) {
    case KVCodec::kMLAFp8DS: KOPS_KV_DS_GATHER(KVCodec::kMLAFp8DS);
    case KVCodec::kMLAFp8DSV40: KOPS_KV_DS_GATHER(KVCodec::kMLAFp8DSV40);
    case KVCodec::kMLAFp8DSV41: KOPS_KV_DS_GATHER(KVCodec::kMLAFp8DSV41);
    case KVCodec::kMLAFp8DSV41Fp4: KOPS_KV_DS_GATHER(KVCodec::kMLAFp8DSV41Fp4);
    default:
      return detail::set_error(Status::kUnsupportedShape,
                               "cp_gather_and_upconvert_ds_mla: codec %d",
                               static_cast<int>(cache.k_codec));
  }
#undef KOPS_KV_DS_GATHER
}

Status cp_gather_indexer_k_quant_cache(const Tensor& dst_k, const Tensor& dst_scale,
                                       const PagedKVDesc& index_cache, const GatherSpec& spec,
                                       const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(index_cache);
  if (s != Status::kSuccess) return s;
  if (dst_k.dtype != DType::kUInt8 && dst_k.dtype != DType::kFloat8E4M3) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "cp_gather_indexer_k_quant_cache: dst_k must be uint8/e4m3");
  }
  if (dst_scale.dtype != DType::kUInt8 && dst_scale.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "cp_gather_indexer_k_quant_cache: dst_scale must be uint8/f32");
  }
  s = check_tensor(dst_k, dst_k.dtype, 2, 1, "cp_gather_indexer_k_quant_cache.dst_k");
  if (s != Status::kSuccess) return s;
  s = check_tensor(dst_scale, dst_scale.dtype, 2, 1, "cp_gather_indexer_k_quant_cache.dst_scale");
  if (s != Status::kSuccess) return s;
  if (dst_k.stride[1] != 1 || dst_scale.stride[1] != 1 ||
      dst_k.shape[0] != dst_scale.shape[0]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "cp_gather_indexer_k_quant_cache: unit last stride, equal rows");
  }
  if (index_cache.k_codec != KVCodec::kIndexerFP8) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_indexer_k_quant_cache: kIndexerFP8 cache required");
  }
  const int64_t head_dim = dst_k.shape[1];
  const int64_t scale_bytes = dst_scale.shape[1] * ((dtype_bits(dst_scale.dtype) + 7) / 8);
  if (head_dim <= 0 || scale_bytes <= 0 || head_dim * 4 % scale_bytes != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_indexer_k_quant_cache: scale width");
  }
  const int quant_block_size = static_cast<int>(head_dim * 4 / scale_bytes);
  if (quant_block_size != 32 && quant_block_size != 64 && quant_block_size != 128) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_indexer_k_quant_cache: quant_block_size");
  }
  if (head_dim % quant_block_size != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_indexer_k_quant_cache: head_dim %% quant_block_size");
  }
  const int64_t groups = head_dim / quant_block_size;
  if (index_cache.entry_bytes != head_dim + groups * 4) {
    return detail::set_error(Status::kUnsupportedShape,
                             "cp_gather_indexer_k_quant_cache: entry_bytes");
  }
  if (spec.block_table == nullptr || spec.cu_seq_lens == nullptr || spec.num_reqs <= 0 ||
      spec.block_table_stride <= 0 || spec.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "cp_gather_indexer_k_quant_cache: incomplete GatherSpec");
  }
  if (spec.num_tokens == 0) return Status::kSuccess;
  const int64_t block_stride = index_cache.k_stride_page > 0
                                   ? index_cache.k_stride_page
                                   : static_cast<int64_t>(index_cache.page_size) *
                                         index_cache.entry_bytes;
  const int64_t token_stride = dst_k.stride[0];
  const int64_t scale_stride = dst_scale.stride[0] * ((dtype_bits(dst_scale.dtype) + 7) / 8);
  const int64_t tokens = spec.num_tokens;
  const cudaStream_t stream = stream_of(ctx);
  const auto run = [&](auto rows) {
    constexpr int kRows = decltype(rows)::value;
    cp_gather_indexer_k_quant_cache_kernel<kRows>
        <<<dim3(static_cast<unsigned>(ceil_div(tokens, kRows))), dim3(32, kRows), 0, stream>>>(
            static_cast<const uint8_t*>(index_cache.k_data),
            reinterpret_cast<uint8_t*>(mut_ptr<uint8_t>(dst_k)),
            reinterpret_cast<uint8_t*>(mut_ptr<uint8_t>(dst_scale)), spec.block_table,
            spec.cu_seq_lens, spec.num_reqs, token_stride, head_dim, block_stride,
            index_cache.k_stride_token > 0 ? index_cache.k_stride_token
                                           : index_cache.entry_bytes,
            index_cache.page_size, spec.block_table_stride, static_cast<int>(tokens),
            quant_block_size);
    const cudaError_t err = cudaGetLastError();
    return err;
  };
  cudaError_t err;
  if (tokens < 32) err = run(std::integral_constant<int, 1>{});
  else if (tokens < 64) err = run(std::integral_constant<int, 2>{});
  else if (tokens < 128) err = run(std::integral_constant<int, 4>{});
  else if (tokens < 256) err = run(std::integral_constant<int, 8>{});
  else if (tokens < 512) err = run(std::integral_constant<int, 16>{});
  else err = run(std::integral_constant<int, 32>{});
  if (err != cudaSuccess) return launch_error("cp_gather_indexer_k_quant_cache", err);
  (void)scale_stride;
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// gather_pages / scatter_pages
// -----------------------------------------------------------------------------

template <typename cache_t, typename out_t, KVCodec kCodec>
Status launch_gather_pages(const Tensor& out, const PagedKVDesc& cache, const Tensor& indices,
                           bool by_token, int64_t items, cudaStream_t stream) {
  const int block = 256;
  const int64_t grid = ceil_div(items, 8);
  const int heads = cache.num_kv_heads > 0 ? cache.num_kv_heads : 1;
  const int elems = entry_elems(cache);
  const bool latent = cache.layout == KVLayout::kLatent || cache.entry_bytes != 0;
  const bool hnd = cache.layout == KVLayout::kHND;
  const int64_t default_token = latent ? elems : (hnd ? cache.head_dim_k : heads * cache.head_dim_k);
  const int64_t default_page = static_cast<int64_t>(elems) * cache.page_size;
  const int64_t default_head =
      latent ? elems : (hnd ? static_cast<int64_t>(cache.page_size) * cache.head_dim_k : cache.head_dim_k);
  PagedKVDesc e{};
  const Status es = element_strides<cache_t>(cache, default_token, default_page, default_head, &e);
  if (es != Status::kSuccess) return es;
  gather_pages_kernel<cache_t, out_t, kCodec><<<static_cast<int>(grid), block, 0, stream>>>(
      mut_ptr<out_t>(out), static_cast<const cache_t*>(cache.k_data), const_ptr<int64_t>(indices),
      cache.k_scale, e, by_token, static_cast<int>(items));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("gather_pages", err);
  return Status::kSuccess;
}

Status gather_pages(const Tensor& out, const PagedKVDesc& cache, const Tensor& indices,
                    bool by_token, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, out.dtype, 2, 16, "gather_pages.out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(indices, DType::kInt64, 1, 8, "gather_pages.indices");
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(out.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "gather_pages out dtype");
  }
  const int64_t items = indices.shape[0];
  const int elems = entry_elems(cache);
  const int64_t expected = by_token ? elems : static_cast<int64_t>(elems) * cache.page_size;
  if (out.shape[1] != expected) {
    return detail::set_error(Status::kInvalidArgument,
                             "gather_pages: out width %lld, expected %lld",
                             static_cast<long long>(out.shape[1]), static_cast<long long>(expected));
  }
  if (items == 0) return Status::kSuccess;
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor || cache.k_codec == KVCodec::kFP8PerHead;
  if (!fp8 && !supported_scalar(cache.k_dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "gather_pages cache dtype");
  }
  if (fp8 && (cache.fp8 != Fp8Encoding::kOCP || fp8_payload_unsupported(cache.k_dtype))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "gather_pages: only OCP e4m3 is compiled");
  }
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_GP(CACHE_T, CODEC)                                                                 \
  do {                                                                                             \
    if (out.dtype == DType::kBFloat16) {                                                           \
      return by_token ? launch_gather_pages<CACHE_T, __nv_bfloat16, CODEC>(out, cache, indices,     \
                                                                           true, items, stream)    \
                      : launch_gather_pages<CACHE_T, __nv_bfloat16, CODEC>(out, cache, indices,     \
                                                                           false, items, stream);   \
    }                                                                                              \
    if (out.dtype == DType::kFloat16) {                                                            \
      return by_token ? launch_gather_pages<CACHE_T, __half, CODEC>(out, cache, indices, true,      \
                                                                    items, stream)                 \
                      : launch_gather_pages<CACHE_T, __half, CODEC>(out, cache, indices, false,     \
                                                                    items, stream);                \
    }                                                                                              \
    if (out.dtype == DType::kFloat32) {                                                            \
      return by_token ? launch_gather_pages<CACHE_T, float, CODEC>(out, cache, indices, true,       \
                                                                   items, stream)                  \
                      : launch_gather_pages<CACHE_T, float, CODEC>(out, cache, indices, false,      \
                                                                   items, stream);                 \
    }                                                                                              \
  } while (0)
  if (fp8) KOPS_KV_GP(uint8_t, KVCodec::kFP8PerTensor);
  if (cache.k_dtype == DType::kBFloat16) KOPS_KV_GP(__nv_bfloat16, KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat16) KOPS_KV_GP(__half, KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat32) KOPS_KV_GP(float, KVCodec::kNative);
#undef KOPS_KV_GP
  return detail::set_error(Status::kUnsupportedDtype, "gather_pages cache dtype");
}

template <typename in_t, typename cache_t, KVCodec kCodec>
Status launch_scatter_pages(const PagedKVDesc& cache, const Tensor& src, const Tensor& indices,
                            bool by_token, int64_t items, cudaStream_t stream) {
  const int64_t grid = ceil_div(items, 8);
  const int heads = cache.num_kv_heads > 0 ? cache.num_kv_heads : 1;
  const int elems = entry_elems(cache);
  const bool latent = cache.layout == KVLayout::kLatent || cache.entry_bytes != 0;
  const bool hnd = cache.layout == KVLayout::kHND;
  const int64_t default_token = latent ? elems : (hnd ? cache.head_dim_k : heads * cache.head_dim_k);
  const int64_t default_page = static_cast<int64_t>(elems) * cache.page_size;
  const int64_t default_head =
      latent ? elems : (hnd ? static_cast<int64_t>(cache.page_size) * cache.head_dim_k : cache.head_dim_k);
  PagedKVDesc e{};
  const Status es = element_strides<cache_t>(cache, default_token, default_page, default_head, &e);
  if (es != Status::kSuccess) return es;
  scatter_pages_kernel<in_t, cache_t, kCodec><<<static_cast<int>(grid), 256, 0, stream>>>(
      static_cast<cache_t*>(cache.k_data), const_ptr<in_t>(src), const_ptr<int64_t>(indices),
      cache.k_scale, e, by_token, static_cast<int>(items));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("scatter_pages", err);
  return Status::kSuccess;
}

Status scatter_pages(const PagedKVDesc& cache, const Tensor& src, const Tensor& indices,
                     bool by_token, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  s = check_tensor(src, src.dtype, 2, 16, "scatter_pages.src");
  if (s != Status::kSuccess) return s;
  s = check_tensor(indices, DType::kInt64, 1, 8, "scatter_pages.indices");
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(src.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "scatter_pages src dtype");
  }
  const int64_t items = indices.shape[0];
  const int elems = entry_elems(cache);
  const int64_t expected = by_token ? elems : static_cast<int64_t>(elems) * cache.page_size;
  if (src.shape[1] != expected) {
    return detail::set_error(Status::kInvalidArgument, "scatter_pages: src width %lld, expected %lld",
                             static_cast<long long>(src.shape[1]), static_cast<long long>(expected));
  }
  if (items == 0) return Status::kSuccess;
  const bool fp8 = cache.k_codec == KVCodec::kFP8PerTensor || cache.k_codec == KVCodec::kFP8PerHead;
  if (fp8 && (cache.fp8 != Fp8Encoding::kOCP || fp8_payload_unsupported(cache.k_dtype))) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "scatter_pages: only OCP e4m3 is compiled");
  }
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_SP(CACHE_T, CODEC)                                                                 \
  do {                                                                                             \
    if (src.dtype == DType::kBFloat16) {                                                           \
      return by_token ? launch_scatter_pages<__nv_bfloat16, CACHE_T, CODEC>(cache, src, indices,    \
                                                                            true, items, stream)    \
                      : launch_scatter_pages<__nv_bfloat16, CACHE_T, CODEC>(cache, src, indices,    \
                                                                            false, items, stream);  \
    }                                                                                              \
    if (src.dtype == DType::kFloat16) {                                                            \
      return by_token ? launch_scatter_pages<__half, CACHE_T, CODEC>(cache, src, indices, true,     \
                                                                     items, stream)                \
                      : launch_scatter_pages<__half, CACHE_T, CODEC>(cache, src, indices, false,    \
                                                                     items, stream);               \
    }                                                                                              \
    if (src.dtype == DType::kFloat32) {                                                            \
      return by_token ? launch_scatter_pages<float, CACHE_T, CODEC>(cache, src, indices, true,      \
                                                                    items, stream)                 \
                      : launch_scatter_pages<float, CACHE_T, CODEC>(cache, src, indices, false,     \
                                                                    items, stream);                \
    }                                                                                              \
  } while (0)
  if (fp8) KOPS_KV_SP(uint8_t, KVCodec::kFP8PerTensor);
  if (cache.k_dtype == DType::kBFloat16) KOPS_KV_SP(__nv_bfloat16, KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat16) KOPS_KV_SP(__half, KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat32) KOPS_KV_SP(float, KVCodec::kNative);
#undef KOPS_KV_SP
  return detail::set_error(Status::kUnsupportedDtype, "scatter_pages cache dtype");
}

// -----------------------------------------------------------------------------
// page movement
// -----------------------------------------------------------------------------

Status copy_pages(const Tensor& layer_cache_ptrs, const Tensor& page_pairs, int64_t page_bytes,
                  int32_t caches_per_layer, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(layer_cache_ptrs, DType::kInt64, 1, 8, "copy_pages.layer_cache_ptrs");
  if (s != Status::kSuccess) return s;
  s = check_tensor(page_pairs, DType::kInt64, 2, 8, "copy_pages.page_pairs");
  if (s != Status::kSuccess) return s;
  if (page_pairs.shape[1] != 2) {
    return detail::set_error(Status::kInvalidArgument, "copy_pages: page_pairs must be [N, 2]");
  }
  if (page_bytes <= 0 || caches_per_layer <= 0) {
    return detail::set_error(Status::kInvalidArgument, "copy_pages: page_bytes/caches_per_layer");
  }
  if (layer_cache_ptrs.shape[0] % caches_per_layer != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "copy_pages: pointer count %lld not divisible by caches_per_layer %d",
                             static_cast<long long>(layer_cache_ptrs.shape[0]), caches_per_layer);
  }
  const int64_t num_layers = layer_cache_ptrs.shape[0] / caches_per_layer;
  const int64_t pairs = page_pairs.shape[0];
  if (pairs == 0 || num_layers == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(pairs), static_cast<unsigned>(num_layers));
  copy_pages_kernel<<<grid, 256, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(layer_cache_ptrs)),
      const_ptr<int64_t>(page_pairs), page_bytes, caches_per_layer, static_cast<int>(pairs),
      static_cast<int>(num_layers));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("copy_pages", err);
  return Status::kSuccess;
}

Status zero_pages(const Tensor& layer_cache_ptrs, const Tensor& pages, int64_t page_bytes,
                  int32_t caches_per_layer, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(layer_cache_ptrs, DType::kInt64, 1, 8, "zero_pages.layer_cache_ptrs");
  if (s != Status::kSuccess) return s;
  s = check_tensor(pages, DType::kInt64, 1, 8, "zero_pages.pages");
  if (s != Status::kSuccess) return s;
  if (page_bytes <= 0 || caches_per_layer <= 0 ||
      layer_cache_ptrs.shape[0] % caches_per_layer != 0) {
    return detail::set_error(Status::kInvalidArgument, "zero_pages: byte/pointer geometry");
  }
  const int64_t num_layers = layer_cache_ptrs.shape[0] / caches_per_layer;
  const int64_t n = pages.shape[0];
  if (n == 0 || num_layers == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(n), static_cast<unsigned>(num_layers));
  zero_pages_kernel<<<grid, 256, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(layer_cache_ptrs)), const_ptr<int64_t>(pages),
      page_bytes, caches_per_layer, static_cast<int>(n), static_cast<int>(num_layers));
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("zero_pages", err);
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// host-driven copies
// -----------------------------------------------------------------------------

Status swap_pages(const Tensor& src, const Tensor& dst, int64_t page_bytes,
                  const Tensor& page_mapping_host, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (src.data == nullptr || dst.data == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "swap_pages: null src/dst");
  }
  if (page_bytes <= 0) {
    return detail::set_error(Status::kInvalidArgument, "swap_pages: page_bytes %lld",
                             static_cast<long long>(page_bytes));
  }
  if (page_mapping_host.data == nullptr || page_mapping_host.dtype != DType::kInt64 ||
      page_mapping_host.ndim != 2 || page_mapping_host.shape[1] != 2) {
    return detail::set_error(Status::kInvalidArgument,
                             "swap_pages: page_mapping_host must be a host int64 [N, 2]");
  }
  if (page_mapping_host.stride[0] != 2 || page_mapping_host.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "swap_pages: mapping must be packed");
  }
  const int64_t* mapping = const_ptr<int64_t>(page_mapping_host);
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(const_ptr<uint8_t>(src));
  uint8_t* dp = reinterpret_cast<uint8_t*>(mut_ptr<uint8_t>(dst));
  const cudaStream_t stream = stream_of(ctx);
  for (int64_t i = 0; i < page_mapping_host.shape[0]; ++i) {
    const int64_t s_page = mapping[i * 2];
    const int64_t d_page = mapping[i * 2 + 1];
    if (s_page < 0 || d_page < 0) {
      return detail::set_error(Status::kInvalidArgument, "swap_pages: negative page id at %lld",
                               static_cast<long long>(i));
    }
    const cudaError_t err = cudaMemcpyAsync(dp + d_page * page_bytes, sp + s_page * page_bytes,
                                            static_cast<size_t>(page_bytes), cudaMemcpyDefault,
                                            stream);
    if (err != cudaSuccess) return launch_error("swap_pages", err);
  }
  return Status::kSuccess;
}

Status copy_batch_async(const BatchCopyDesc& desc, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (desc.count < 0) {
    return detail::set_error(Status::kInvalidArgument, "copy_batch_async: negative count");
  }
  if (desc.count == 0) return Status::kSuccess;
  if (desc.src_ptrs_host == nullptr || desc.dst_ptrs_host == nullptr ||
      desc.sizes_host == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "copy_batch_async: null descriptor array");
  }
  for (int64_t i = 0; i < desc.count; ++i) {
    if (desc.sizes_host[i] == 0) {
      return detail::set_error(Status::kInvalidArgument, "copy_batch_async: zero-size copy %lld",
                               static_cast<long long>(i));
    }
  }
  const cudaStream_t stream = stream_of(ctx);
  const int64_t chunk_limit =
      desc.max_descriptors_per_call > 0 ? desc.max_descriptors_per_call : desc.count;
  // The batch API rejects the legacy default stream; use the documented fallback.
  const bool batch_ok = stream != nullptr;
  for (int64_t base = 0; base < desc.count; base += chunk_limit) {
    const int64_t n = std::min(chunk_limit, desc.count - base);
    if (batch_ok) {
#if CUDART_VERSION >= 12080
      std::vector<void*> dsts(static_cast<size_t>(n));
      std::vector<void*> srcs(static_cast<size_t>(n));
      std::vector<size_t> sizes(static_cast<size_t>(n));
      std::vector<cudaMemcpyAttributes> attrs(static_cast<size_t>(n));
      std::vector<size_t> attr_idxs(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) {
        dsts[static_cast<size_t>(i)] =
            reinterpret_cast<void*>(static_cast<uintptr_t>(desc.dst_ptrs_host[base + i]));
        srcs[static_cast<size_t>(i)] =
            reinterpret_cast<void*>(static_cast<uintptr_t>(desc.src_ptrs_host[base + i]));
        sizes[static_cast<size_t>(i)] = static_cast<size_t>(desc.sizes_host[base + i]);
        attrs[static_cast<size_t>(i)] = cudaMemcpyAttributes{};
        attrs[static_cast<size_t>(i)].srcAccessOrder =
            desc.src_access_order_any ? cudaMemcpySrcAccessOrderAny : cudaMemcpySrcAccessOrderStream;
        attr_idxs[static_cast<size_t>(i)] = static_cast<size_t>(i);
      }
      size_t fail_idx = 0;
#if CUDART_VERSION >= 13000   // CUDA 13 dropped the failIdx out-parameter
      (void)fail_idx;
      const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                   static_cast<size_t>(n), attrs.data(),
                                                   attr_idxs.data(), static_cast<size_t>(n), stream);
#else
      const cudaError_t err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                                   static_cast<size_t>(n), attrs.data(),
                                                   attr_idxs.data(), static_cast<size_t>(n),
                                                   &fail_idx, stream);
#endif
      if (err == cudaSuccess) continue;
      if (err != cudaErrorCallRequiresNewerDriver && err != cudaErrorNotSupported) {
        return launch_error("copy_batch_async", err);
      }
#endif
    }
    for (int64_t i = 0; i < n; ++i) {
      const cudaError_t err = cudaMemcpyAsync(
          reinterpret_cast<void*>(static_cast<uintptr_t>(desc.dst_ptrs_host[base + i])),
          reinterpret_cast<const void*>(static_cast<uintptr_t>(desc.src_ptrs_host[base + i])),
          static_cast<size_t>(desc.sizes_host[base + i]), cudaMemcpyDefault, stream);
      if (err != cudaSuccess) return launch_error("copy_batch_async fallback", err);
    }
  }
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// K03 conversion / relayout
// -----------------------------------------------------------------------------

Status convert_cache(const Tensor& dst_cache, const Tensor& src_cache, float scale, KVCodec codec,
                     const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(dst_cache, dst_cache.dtype, 2, 1, "convert_cache.dst");
  if (s != Status::kSuccess) return s;
  s = check_tensor(src_cache, src_cache.dtype, 2, 1, "convert_cache.src");
  if (s != Status::kSuccess) return s;
  if (dst_cache.shape[0] != src_cache.shape[0] || dst_cache.shape[1] != src_cache.shape[1]) {
    return detail::set_error(Status::kInvalidArgument, "convert_cache: shape mismatch");
  }
  if (scale == 0.0f) {
    return detail::set_error(Status::kInvalidArgument, "convert_cache: zero scale");
  }
  if (fp8_payload_unsupported(src_cache.dtype) || fp8_payload_unsupported(dst_cache.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "convert_cache: only OCP e4m3 is compiled");
  }
  const int64_t blocks = src_cache.shape[0];
  const int64_t block_stride = src_cache.shape[1];
  if (blocks == 0 || block_stride == 0) return Status::kSuccess;
  const int block = threads_for(block_stride, 64, 512);
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_CONVERT(OUT, IN, CODEC)                                                          \
  do {                                                                                           \
    convert_cache_kernel<OUT, IN, CODEC><<<static_cast<int>(blocks), block, 0, stream>>>(         \
        const_ptr<IN>(src_cache), mut_ptr<OUT>(dst_cache), scale, block_stride);                   \
    const cudaError_t err = cudaGetLastError();                                                  \
    if (err != cudaSuccess) return launch_error("convert_cache", err);                            \
    return Status::kSuccess;                                                                     \
  } while (0)
  const bool src_fp8 = dtype_is_fp8(src_cache.dtype) || src_cache.dtype == DType::kUInt8;
  const bool dst_fp8 = dtype_is_fp8(dst_cache.dtype) || dst_cache.dtype == DType::kUInt8;
  if (codec == KVCodec::kNative) {
    if (src_cache.dtype == DType::kFloat32 && dst_cache.dtype == DType::kFloat32) KOPS_KV_CONVERT(float, float, KVCodec::kNative);
    if (src_cache.dtype == DType::kFloat32 && dst_cache.dtype == DType::kBFloat16) KOPS_KV_CONVERT(__nv_bfloat16, float, KVCodec::kNative);
    if (src_cache.dtype == DType::kFloat32 && dst_cache.dtype == DType::kFloat16) KOPS_KV_CONVERT(__half, float, KVCodec::kNative);
    if (src_cache.dtype == DType::kBFloat16 && dst_cache.dtype == DType::kFloat32) KOPS_KV_CONVERT(float, __nv_bfloat16, KVCodec::kNative);
    if (src_cache.dtype == DType::kBFloat16 && dst_cache.dtype == DType::kBFloat16) KOPS_KV_CONVERT(__nv_bfloat16, __nv_bfloat16, KVCodec::kNative);
    if (src_cache.dtype == DType::kBFloat16 && dst_cache.dtype == DType::kFloat16) KOPS_KV_CONVERT(__half, __nv_bfloat16, KVCodec::kNative);
    if (src_cache.dtype == DType::kFloat16 && dst_cache.dtype == DType::kFloat32) KOPS_KV_CONVERT(float, __half, KVCodec::kNative);
    if (src_cache.dtype == DType::kFloat16 && dst_cache.dtype == DType::kBFloat16) KOPS_KV_CONVERT(__nv_bfloat16, __half, KVCodec::kNative);
    if (src_cache.dtype == DType::kFloat16 && dst_cache.dtype == DType::kFloat16) KOPS_KV_CONVERT(__half, __half, KVCodec::kNative);
  } else if (codec == KVCodec::kFP8PerTensor || codec == KVCodec::kFP8PerHead) {
    if (src_fp8 && !dst_fp8) {
      if (dst_cache.dtype == DType::kBFloat16) KOPS_KV_CONVERT(__nv_bfloat16, uint8_t, KVCodec::kFP8PerTensor);
      if (dst_cache.dtype == DType::kFloat16) KOPS_KV_CONVERT(__half, uint8_t, KVCodec::kFP8PerTensor);
      if (dst_cache.dtype == DType::kFloat32) KOPS_KV_CONVERT(float, uint8_t, KVCodec::kFP8PerTensor);
    }
    if (!src_fp8 && dst_fp8) {
      if (src_cache.dtype == DType::kBFloat16) KOPS_KV_CONVERT(uint8_t, __nv_bfloat16, KVCodec::kFP8PerTensor);
      if (src_cache.dtype == DType::kFloat16) KOPS_KV_CONVERT(uint8_t, __half, KVCodec::kFP8PerTensor);
      if (src_cache.dtype == DType::kFloat32) KOPS_KV_CONVERT(uint8_t, float, KVCodec::kFP8PerTensor);
    }
    if (src_fp8 && dst_fp8) KOPS_KV_CONVERT(uint8_t, uint8_t, KVCodec::kFP8PerTensor);
  }
#undef KOPS_KV_CONVERT
  return detail::set_error(Status::kUnsupportedDtype, "convert_cache: %s -> %s (codec %d)",
                           dtype_name(src_cache.dtype), dtype_name(dst_cache.dtype),
                           static_cast<int>(codec));
}

Status requantize_pages(const PagedKVDesc& src, const PagedKVDesc& dst, const Tensor& page_pairs,
                        const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(src);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(dst);
  if (s != Status::kSuccess) return s;
  s = check_tensor(page_pairs, DType::kInt64, 2, 8, "requantize_pages.page_pairs");
  if (s != Status::kSuccess) return s;
  if (page_pairs.shape[1] != 2) {
    return detail::set_error(Status::kInvalidArgument, "requantize_pages: page_pairs [N, 2]");
  }
  if (src.page_size != dst.page_size || src.num_kv_heads != dst.num_kv_heads ||
      src.head_dim_k != dst.head_dim_k || src.head_dim_v != dst.head_dim_v) {
    return detail::set_error(Status::kUnsupportedShape, "requantize_pages: extents differ");
  }
  if (src.layout != dst.layout ||
      (src.layout != KVLayout::kNHD && src.layout != KVLayout::kHND)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "requantize_pages: equal NHD/HND layouts required");
  }
  if (dst.k_codec == KVCodec::kNative && dst.v_codec == KVCodec::kNative) {
    return detail::set_error(Status::kUnsupportedShape,
                             "requantize_pages: destination must be quantized");
  }
  if (src.head_dim_k < 2 || src.head_dim_v < 2 || (src.head_dim_k & 1) ||
      (src.head_dim_v & 1) || src.head_dim_k > kMaxRowDim || src.head_dim_v > kMaxRowDim) {
    return detail::set_error(Status::kUnsupportedShape,
                             "requantize_pages: head dims even and <= %d", kMaxRowDim);
  }
  const int64_t pairs = page_pairs.shape[0];
  if (pairs == 0) return Status::kSuccess;
  const int64_t rows = static_cast<int64_t>(2) * src.page_size * src.num_kv_heads;
  const dim3 grid(static_cast<unsigned>(pairs),
                  static_cast<unsigned>(ceil_div(rows, 32)));
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_REQUANT(SRC_T, DST_T)                                                                 \
  do {                                                                                             \
    kv_requantize_pages_kernel<SRC_T, DST_T, KVCodec::kNative, KVCodec::kNative>                    \
        <<<grid, 256, 0, stream>>>(src, dst, const_ptr<int64_t>(page_pairs),                         \
                                   static_cast<int>(pairs));                                        \
    const cudaError_t err = cudaGetLastError();                                                     \
    if (err != cudaSuccess) return launch_error("requantize_pages", err);                           \
    return Status::kSuccess;                                                                        \
  } while (0)
  const auto src_u8 = src.k_dtype == DType::kFloat8E4M3 || src.k_dtype == DType::kUInt8 ||
                      src.k_codec == KVCodec::kFP8PerTokenHead ||
                      src.k_codec == KVCodec::kInt8PerTokenHead ||
                      src.k_codec == KVCodec::kInt4PerTokenHead;
  const auto dst_u8 = dst.k_dtype == DType::kFloat8E4M3 || dst.k_dtype == DType::kUInt8 ||
                      dst.k_codec == KVCodec::kFP8PerTokenHead ||
                      dst.k_codec == KVCodec::kInt8PerTokenHead ||
                      dst.k_codec == KVCodec::kInt4PerTokenHead;
  if (src_u8 && dst_u8) KOPS_REQUANT(uint8_t, uint8_t);
  if (src_u8 && !dst_u8) {
    if (dst.k_dtype == DType::kBFloat16) KOPS_REQUANT(uint8_t, __nv_bfloat16);
    if (dst.k_dtype == DType::kFloat16) KOPS_REQUANT(uint8_t, __half);
    if (dst.k_dtype == DType::kFloat32) KOPS_REQUANT(uint8_t, float);
  }
  if (!src_u8 && dst_u8) {
    if (src.k_dtype == DType::kBFloat16) KOPS_REQUANT(__nv_bfloat16, uint8_t);
    if (src.k_dtype == DType::kFloat16) KOPS_REQUANT(__half, uint8_t);
    if (src.k_dtype == DType::kFloat32) KOPS_REQUANT(float, uint8_t);
  }
  if (!src_u8 && !dst_u8) {
    if (src.k_dtype == DType::kBFloat16 && dst.k_dtype == DType::kBFloat16) KOPS_REQUANT(__nv_bfloat16, __nv_bfloat16);
    if (src.k_dtype == DType::kBFloat16 && dst.k_dtype == DType::kFloat16) KOPS_REQUANT(__nv_bfloat16, __half);
    if (src.k_dtype == DType::kBFloat16 && dst.k_dtype == DType::kFloat32) KOPS_REQUANT(__nv_bfloat16, float);
    if (src.k_dtype == DType::kFloat16 && dst.k_dtype == DType::kBFloat16) KOPS_REQUANT(__half, __nv_bfloat16);
    if (src.k_dtype == DType::kFloat16 && dst.k_dtype == DType::kFloat16) KOPS_REQUANT(__half, __half);
    if (src.k_dtype == DType::kFloat16 && dst.k_dtype == DType::kFloat32) KOPS_REQUANT(__half, float);
    if (src.k_dtype == DType::kFloat32 && dst.k_dtype == DType::kBFloat16) KOPS_REQUANT(float, __nv_bfloat16);
    if (src.k_dtype == DType::kFloat32 && dst.k_dtype == DType::kFloat16) KOPS_REQUANT(float, __half);
    if (src.k_dtype == DType::kFloat32 && dst.k_dtype == DType::kFloat32) KOPS_REQUANT(float, float);
  }
#undef KOPS_REQUANT
  return detail::set_error(Status::kUnsupportedDtype,
                           "requantize_pages: %s -> %s", dtype_name(src.k_dtype),
                           dtype_name(dst.k_dtype));
}

Status kv_scale_update(const Tensor& k_absmax, const Tensor& v_absmax, const Tensor& key,
                       const Tensor& value, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!supported_scalar(key.dtype) || value.dtype != key.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "kv_scale_update: key/value dtype");
  }
  s = check_tensor(key, key.dtype, 3, 1, "kv_scale_update.key");
  if (s != Status::kSuccess) return s;
  s = check_tensor(value, key.dtype, 3, 1, "kv_scale_update.value");
  if (s != Status::kSuccess) return s;
  s = check_tensor(k_absmax, DType::kFloat32, 1, 4, "kv_scale_update.k_absmax");
  if (s != Status::kSuccess) return s;
  s = check_tensor(v_absmax, DType::kFloat32, 1, 4, "kv_scale_update.v_absmax");
  if (s != Status::kSuccess) return s;
  const int64_t tokens = key.shape[0];
  const int64_t heads = key.shape[1];
  const int64_t dim_k = key.shape[2];
  const int64_t dim_v = value.shape[2];
  if (value.shape[0] != tokens || value.shape[1] != heads || heads <= 0 || dim_k <= 0 ||
      dim_v <= 0) {
    return detail::set_error(Status::kInvalidArgument, "kv_scale_update: extents");
  }
  if (key.stride[2] != 1 || value.stride[2] != 1 || key.stride[1] != dim_k ||
      value.stride[1] != dim_v || key.stride[0] != heads * dim_k ||
      value.stride[0] != heads * dim_v) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "kv_scale_update: key/value must be dense [T, H, D]");
  }
  if (k_absmax.shape[0] < heads || v_absmax.shape[0] < heads) {
    return detail::set_error(Status::kUnsupportedShape,
                             "kv_scale_update: absmax buffers must have at least H entries");
  }
  if (tokens == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(2 * heads),
                  static_cast<unsigned>(std::min<int64_t>(ceil_div(tokens, 8), 512)));
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_SCALE(T)                                                                           \
  do {                                                                                             \
    kv_scale_update_kernel<T><<<grid, 256, 0, stream>>>(                                            \
        mut_ptr<float>(k_absmax), mut_ptr<float>(v_absmax), const_ptr<T>(key),                     \
        const_ptr<T>(value), static_cast<int>(tokens), static_cast<int>(heads),                     \
        static_cast<int>(dim_k), static_cast<int>(dim_v));                                         \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("kv_scale_update", err);                           \
    return Status::kSuccess;                                                                       \
  } while (0)
  if (key.dtype == DType::kBFloat16) KOPS_KV_SCALE(__nv_bfloat16);
  if (key.dtype == DType::kFloat16) KOPS_KV_SCALE(__half);
  if (key.dtype == DType::kFloat32) KOPS_KV_SCALE(float);
#undef KOPS_KV_SCALE
  return detail::set_error(Status::kUnsupportedDtype, "kv_scale_update dtype");
}

Status relayout_pages(const PagedKVDesc& src, const PagedKVDesc& dst,
                      const Tensor& src_seq_page_indptr, const Tensor& dst_seq_page_indptr,
                      const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(src);
  if (s != Status::kSuccess) return s;
  s = kv_desc_validate(dst);
  if (s != Status::kSuccess) return s;
  s = check_tensor(src_seq_page_indptr, DType::kInt32, 1, 4, "relayout_pages.src_indptr");
  if (s != Status::kSuccess) return s;
  s = check_tensor(dst_seq_page_indptr, DType::kInt32, 1, 4, "relayout_pages.dst_indptr");
  if (s != Status::kSuccess) return s;
  if (src_seq_page_indptr.shape[0] != dst_seq_page_indptr.shape[0] ||
      src_seq_page_indptr.shape[0] < 1) {
    return detail::set_error(Status::kInvalidArgument, "relayout_pages: batch mismatch");
  }
  if (src.k_dtype != dst.k_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "relayout_pages: dtype mismatch");
  }
  if (src.layout != KVLayout::kLatent && dst.layout != KVLayout::kLatent &&
      src.head_dim_k != dst.head_dim_k) {
    return detail::set_error(Status::kUnsupportedShape, "relayout_pages: head_dim mismatch");
  }
  if (src.k_codec != KVCodec::kNative || dst.k_codec != KVCodec::kNative) {
    return detail::set_error(Status::kNotImplemented, "relayout_pages: native codec only");
  }
  if ((src.layout == KVLayout::kLatent) != (dst.layout == KVLayout::kLatent)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "relayout_pages: latent and dense layouts cannot be mixed");
  }
  if (src.layout == KVLayout::kLatent && src.entry_bytes != dst.entry_bytes) {
    return detail::set_error(Status::kUnsupportedShape,
                             "relayout_pages: latent record width mismatch");
  }
  const int batch = static_cast<int>(src_seq_page_indptr.shape[0] - 1);
  if (batch <= 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_KV_RELAYOUT(T)                                                                        \
  do {                                                                                             \
    PagedKVDesc src_e{};                                                                            \
    PagedKVDesc dst_e{};                                                                            \
    Status es = relayout_element_desc<T>(src, &src_e);                                              \
    if (es != Status::kSuccess) return es;                                                          \
    es = relayout_element_desc<T>(dst, &dst_e);                                                     \
    if (es != Status::kSuccess) return es;                                                          \
    kv_relayout_pages_kernel<T><<<dim3(batch, 256), 256, 0, stream>>>(                             \
        src_e, dst_e, const_ptr<int32_t>(src_seq_page_indptr), const_ptr<int32_t>(dst_seq_page_indptr), \
        batch);                                                                                     \
    const cudaError_t err = cudaGetLastError();                                                     \
    if (err != cudaSuccess) return launch_error("kv_relayout_pages", err);                          \
    return Status::kSuccess;                                                                        \
  } while (0)
  if (src.k_dtype == DType::kBFloat16) KOPS_KV_RELAYOUT(__nv_bfloat16);
  if (src.k_dtype == DType::kFloat16) KOPS_KV_RELAYOUT(__half);
  if (src.k_dtype == DType::kFloat32) KOPS_KV_RELAYOUT(float);
#undef KOPS_KV_RELAYOUT
  return detail::set_error(Status::kUnsupportedDtype, "relayout_pages dtype %s",
                           dtype_name(src.k_dtype));
}

}  // namespace kops::kv
