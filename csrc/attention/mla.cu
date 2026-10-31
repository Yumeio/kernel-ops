// =============================================================================
// kernel-ops | csrc/attention/mla.cu
// B2 A12 copy-side helpers: concat_mla_q and mla_context_gather.
//
//   concat_mla_q        q_out[t,h] = [ ql_nope[t,h,:] | q_pe[t,h,:] ]
//   mla_context_gather  gather cached latent entries of a cached prefix back
//                       into contiguous c_kv / k_pe buffers for up-projection.
//
// Both are plain index copies (no MMA); they accept strided query views and a
// paged latent cache. FP8 latent caches (per-tensor e4m3) are dequantized on the
// fly; the B3 ds_mla record codecs are rejected here.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstring>

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
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace kops::attn {
namespace {

using dev::from_f;
using dev::to_f;

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

template <typename cache_t, kv::KVCodec kCodec>
__device__ __forceinline__ float cache_load(cache_t v, float scale) {
  if constexpr (kCodec == kv::KVCodec::kNative) {
    return to_f<cache_t>(v);
  } else {
    return e4m3_to_float(static_cast<uint8_t>(v)) * scale;
  }
}

}  // namespace

using dev::from_f;
using dev::to_f;

// -----------------------------------------------------------------------------
// concat_mla_q
// -----------------------------------------------------------------------------

template <typename scalar_t, int NOPE_CHUNK>
__global__ void concat_mla_q_kernel(scalar_t* __restrict__ q_out,
                                    const scalar_t* __restrict__ ql_nope,
                                    const scalar_t* __restrict__ q_pe, const int num_tokens,
                                    const int num_heads, const int nope_dim,
                                    const int64_t out_stride_token,
                                    const int64_t out_stride_head,
                                    const int64_t nope_stride_token,
                                    const int64_t nope_stride_head,
                                    const int64_t pe_stride_token, const int64_t pe_stride_head) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t task = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  const int64_t total = static_cast<int64_t>(num_tokens) * num_heads;
  if (task >= total) return;
  const int t = static_cast<int>(task / num_heads);
  const int h = static_cast<int>(task % num_heads);
  const scalar_t* nsrc =
      ql_nope + static_cast<int64_t>(t) * nope_stride_token + static_cast<int64_t>(h) * nope_stride_head;
  const scalar_t* psrc =
      q_pe + static_cast<int64_t>(t) * pe_stride_token + static_cast<int64_t>(h) * pe_stride_head;
  scalar_t* dst =
      q_out + static_cast<int64_t>(t) * out_stride_token + static_cast<int64_t>(h) * out_stride_head;
  (void)NOPE_CHUNK;
  for (int i = lane; i < nope_dim; i += 32) dst[i] = nsrc[i];
  for (int i = lane; i < 64; i += 32) dst[nope_dim + i] = psrc[i];
}

// -----------------------------------------------------------------------------
// mla_context_gather
// -----------------------------------------------------------------------------

__device__ __forceinline__ int find_ctx_request(const int32_t* indptr, int batch_size,
                                                int token) {
  int lo = 0;
  int hi = batch_size;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indptr[mid + 1] <= token) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < batch_size ? lo : -1;
}

template <typename cache_t, typename out_t, kv::KVCodec kCodec>
__global__ void mla_context_gather_kernel(
    out_t* __restrict__ c_kv_out, out_t* __restrict__ k_pe_out, const cache_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ ctx_token_indptr,
    const int32_t* __restrict__ ctx_start, const float* __restrict__ scale,
    const kv::PagedKVDesc desc, const int batch_size) {
  constexpr int kWarps = 8;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
  out_t* c_base = c_kv_out;
  if (c_base == nullptr) return;
  if (token < 0) return;
  const int r = find_ctx_request(ctx_token_indptr, batch_size, token);
  if (r < 0) return;
  const int64_t pos = static_cast<int64_t>(ctx_start[r]) + (token - ctx_token_indptr[r]);
  const int64_t logical = pos / desc.page_size;
  if (logical < 0 || logical >= desc.block_table_stride) return;
  const int32_t page =
      block_table[static_cast<int64_t>(r) * desc.block_table_stride + logical];
  if (page < 0) return;
  const int64_t off = pos % desc.page_size;
  const cache_t* src =
      kv_cache + static_cast<int64_t>(page) * desc.k_stride_page + off * desc.k_stride_token;
  const int rank = desc.head_dim_v;
  const int pe = desc.head_dim_k - desc.head_dim_v;
  const float s = scale != nullptr ? scale[0] : 1.0f;
  out_t* c_dst = c_kv_out + static_cast<int64_t>(token) * rank;
  out_t* p_dst = k_pe_out + static_cast<int64_t>(token) * pe;
  for (int i = lane; i < rank; i += 32) {
    const float v = cache_load<cache_t, kCodec>(src[i], s);
    out_t y;
    from_f<out_t>(y, v);
    c_dst[i] = y;
  }
  for (int i = lane; i < pe; i += 32) {
    const float v = cache_load<cache_t, kCodec>(src[rank + i], s);
    out_t y;
    from_f<out_t>(y, v);
    p_dst[i] = y;
  }
}

namespace {

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
    return detail::set_error(Status::kUnsupportedBackend, "attention.mla: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool scalar_dtype(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Build an element-strided copy of an MLA descriptor. Native byte strides are
/// divided by sizeof(cache_t); FP8 (uint8) strides are already elements.
template <typename cache_t>
Status element_desc(const kv::PagedKVDesc& in, kv::PagedKVDesc* out) {
  *out = in;
  const int64_t es = static_cast<int64_t>(sizeof(cache_t));
  if (in.stride_unit == StrideUnit::kByte) {
    if ((in.k_stride_page % es) != 0 || (in.k_stride_token % es) != 0 ||
        (in.k_stride_head % es) != 0) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "mla_context_gather: byte strides not divisible by element size");
    }
    out->k_stride_page /= es;
    out->k_stride_token /= es;
    out->k_stride_head /= es;
    out->stride_unit = StrideUnit::kElement;
  }
  if (out->k_stride_page == 0) {
    out->k_stride_page = static_cast<int64_t>(out->page_size) * (in.head_dim_k > 0 ? in.head_dim_k : 1);
  }
  if (out->k_stride_token == 0) out->k_stride_token = in.head_dim_k > 0 ? in.head_dim_k : 1;
  return Status::kSuccess;
}

}  // namespace

Status concat_mla_q(const Tensor& q_out, const Tensor& ql_nope, const Tensor& q_pe,
                    const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (!scalar_dtype(q_out.dtype) || q_out.dtype != ql_nope.dtype || q_out.dtype != q_pe.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "concat_mla_q: dtype mismatch");
  }
  s = check_tensor(q_out, q_out.dtype, 3, 16, "concat_mla_q.q_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ql_nope, ql_nope.dtype, 3, 16, "concat_mla_q.ql_nope");
  if (s != Status::kSuccess) return s;
  s = check_tensor(q_pe, q_pe.dtype, 3, 16, "concat_mla_q.q_pe");
  if (s != Status::kSuccess) return s;
  if (q_out.stride[2] != 1 || ql_nope.stride[2] != 1 || q_pe.stride[2] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "concat_mla_q: last dim must be contiguous");
  }
  const int64_t tokens = q_out.shape[0];
  const int64_t heads = q_out.shape[1];
  const int64_t nope = ql_nope.shape[2];
  const int64_t pe = q_pe.shape[2];
  if (ql_nope.shape[0] != tokens || ql_nope.shape[1] != heads || q_pe.shape[0] != tokens ||
      q_pe.shape[1] != heads || q_out.shape[2] != nope + pe || pe != 64) {
    return detail::set_error(Status::kInvalidArgument,
                             "concat_mla_q: expected q_out [T,H,nope+64] and q_pe [T,H,64]");
  }
  if (tokens == 0 || heads == 0) return Status::kSuccess;
  if (tokens * heads > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "concat_mla_q: too many tasks");
  }
  const int64_t grid = ceil_div(tokens * heads, 8);
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_MLA_CONCAT(T)                                                                     \
  do {                                                                                         \
    concat_mla_q_kernel<T, 1><<<static_cast<int>(grid), 256, 0, stream>>>(                     \
        mut_ptr<T>(q_out), const_ptr<T>(ql_nope), const_ptr<T>(q_pe), static_cast<int>(tokens), \
        static_cast<int>(heads), static_cast<int>(nope), q_out.stride[0], q_out.stride[1],      \
        ql_nope.stride[0], ql_nope.stride[1], q_pe.stride[0], q_pe.stride[1]);                 \
    const cudaError_t err = cudaGetLastError();                                                \
    if (err != cudaSuccess) return launch_error("concat_mla_q", err);                          \
    return Status::kSuccess;                                                                   \
  } while (0)
  if (q_out.dtype == DType::kBFloat16) KOPS_MLA_CONCAT(__nv_bfloat16);
  if (q_out.dtype == DType::kFloat16) KOPS_MLA_CONCAT(__half);
  if (q_out.dtype == DType::kFloat32) KOPS_MLA_CONCAT(float);
#undef KOPS_MLA_CONCAT
  return detail::set_error(Status::kUnsupportedDtype, "concat_mla_q dtype");
}

Status mla_context_gather(const Tensor& c_kv_out, const Tensor& k_pe_out,
                          const kv::PagedKVDesc& cache, const Tensor& ctx_token_indptr,
                          const Tensor& ctx_start, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = kv::kv_desc_validate(cache);
  if (s != Status::kSuccess) return s;
  if (!scalar_dtype(c_kv_out.dtype) || k_pe_out.dtype != c_kv_out.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather: dtype mismatch");
  }
  s = check_tensor(c_kv_out, c_kv_out.dtype, 2, 16, "mla_context_gather.c_kv_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(k_pe_out, k_pe_out.dtype, 2, 16, "mla_context_gather.k_pe_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ctx_token_indptr, DType::kInt32, 1, 4, "mla_context_gather.ctx_token_indptr");
  if (s != Status::kSuccess) return s;
  s = check_tensor(ctx_start, DType::kInt32, 1, 4, "mla_context_gather.ctx_start");
  if (s != Status::kSuccess) return s;
  if (c_kv_out.stride[1] != 1 || k_pe_out.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mla_context_gather: outputs must be contiguous rows");
  }
  const int64_t tokens = c_kv_out.shape[0];
  const int64_t rank = c_kv_out.shape[1];
  const int64_t pe = k_pe_out.shape[1];
  if (k_pe_out.shape[0] < tokens || rank <= 0 || pe <= 0) {
    return detail::set_error(Status::kInvalidArgument, "mla_context_gather: output shapes");
  }
  if (cache.head_dim_v != rank || cache.head_dim_k != rank + pe) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mla_context_gather: descriptor entry %d+%d, outputs %lld+%lld",
                             cache.head_dim_v, cache.head_dim_k - cache.head_dim_v,
                             static_cast<long long>(rank), static_cast<long long>(pe));
  }
  if (cache.block_table == nullptr || cache.block_table_stride <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "mla_context_gather: a block table is required");
  }
  const int batch = static_cast<int>(ctx_token_indptr.shape[0] - 1);
  if (batch <= 0 || ctx_start.shape[0] < batch) {
    return detail::set_error(Status::kInvalidArgument, "mla_context_gather: batch mismatch");
  }
  if (cache.k_codec != kv::KVCodec::kNative && cache.k_codec != kv::KVCodec::kFP8PerTensor &&
      cache.k_codec != kv::KVCodec::kFP8PerHead) {
    return detail::set_error(Status::kNotImplemented,
                             "mla_context_gather: codec %d lands in B3",
                             static_cast<int>(cache.k_codec));
  }
  const bool fp8 = cache.k_codec != kv::KVCodec::kNative;
  if (fp8 && (cache.fp8 != Fp8Encoding::kOCP || cache.k_dtype == DType::kFloat8E5M2 ||
              cache.k_dtype == DType::kFloat8E5M2FNUZ)) {
    return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather: e4m3 OCP only");
  }
  if (!fp8 && cache.k_dtype != c_kv_out.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mla_context_gather: native cache/output dtype mismatch");
  }
  if (tokens == 0) return Status::kSuccess;
  const int64_t grid = ceil_div(tokens, 8);
  const cudaStream_t stream = stream_of(ctx);

#define KOPS_MLA_GATHER(CACHE_T, OUT_T, CODEC)                                                    \
  do {                                                                                             \
    kv::PagedKVDesc e{};                                                                           \
    Status es = element_desc<CACHE_T>(cache, &e);                                                   \
    if (es != Status::kSuccess) return es;                                                          \
    mla_context_gather_kernel<CACHE_T, OUT_T, CODEC><<<static_cast<int>(grid), 256, 0, stream>>>(   \
        mut_ptr<OUT_T>(c_kv_out), mut_ptr<OUT_T>(k_pe_out), static_cast<const CACHE_T*>(cache.k_data), \
        cache.block_table, const_ptr<int32_t>(ctx_token_indptr),                                      \
        const_ptr<int32_t>(ctx_start), cache.k_scale, e, batch);                                    \
    const cudaError_t err = cudaGetLastError();                                                    \
    if (err != cudaSuccess) return launch_error("mla_context_gather", err);                         \
    return Status::kSuccess;                                                                        \
  } while (0)

#define KOPS_MLA_GATHER_OUT(CACHE_T, CODEC)                                                        \
  do {                                                                                              \
    if (c_kv_out.dtype == DType::kBFloat16) KOPS_MLA_GATHER(CACHE_T, __nv_bfloat16, CODEC);          \
    if (c_kv_out.dtype == DType::kFloat16) KOPS_MLA_GATHER(CACHE_T, __half, CODEC);                  \
    if (c_kv_out.dtype == DType::kFloat32) KOPS_MLA_GATHER(CACHE_T, float, CODEC);                   \
  } while (0)

  if (fp8) KOPS_MLA_GATHER_OUT(uint8_t, kv::KVCodec::kFP8PerTensor);
  if (cache.k_dtype == DType::kBFloat16) KOPS_MLA_GATHER_OUT(__nv_bfloat16, kv::KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat16) KOPS_MLA_GATHER_OUT(__half, kv::KVCodec::kNative);
  if (cache.k_dtype == DType::kFloat32) KOPS_MLA_GATHER_OUT(float, kv::KVCodec::kNative);
#undef KOPS_MLA_GATHER_OUT
#undef KOPS_MLA_GATHER
  return detail::set_error(Status::kUnsupportedDtype, "mla_context_gather cache dtype");
}

}  // namespace kops::attn
