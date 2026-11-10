// =============================================================================
// kernel-ops | csrc/model_ops/residual.cu
// B0: residual_add and cast. Both view their operands as [rows, d] with a row
// stride in elements and unit stride along d. Arithmetic is fp32 and rounded
// once at the store.
//
// Launch shape: a grid-stride loop over the flattened chunk space
// [0, rows * (d / N)) instead of one CTA per row, so the launch is decoupled
// from `rows`: a narrow row no longer wastes CTAs and a very wide row (rows=1)
// no longer runs on one CTA. A chunk is one 16 B pack of the OUTPUT dtype (N =
// 16 / sizeof(out_t) elements); a thread handles two of them, one CTA-stride
// apart, so every store instruction covers 16 B per lane contiguously. Two
// block-strided packs per thread is what "32 B per thread" has to mean for a
// store-bound kernel: two *adjacent* 16 B stores per lane leave the warp's store
// 50% dense per instruction, which measured 723 vs 793 GB/s on the bf16 -> fp32
// cast (t1024 x h4096, cold L2).
// The input side of a chunk is N * sizeof(in_t) bytes: 8, 16 or 32 B, i.e. one
// or two 16 B loads (a narrow 8 B load when the output is twice as wide).
// A scalar path (N = 1) handles every hidden size and row stride that the
// vector path cannot cover, with the same expression and one rounding.
//
// Cache policy: the operands are read once and written once, so the accesses go
// through ld.global.cs / st.global.cs (evict-first in L2). Interleaved A/B on
// RTX 4090 with the benchmark's cold-L2 protocol, bf16 add of two 4096-wide
// rows (24 MB touched) and 8192 rows (192 MB touched):
//   24 MB   plain 724.7 GB/s   cs loads/stores 743.7 GB/s
//   192 MB  plain 862.6 GB/s   cs loads/stores 847.5 GB/s
// Evict-first reads pay below L2 (a small streaming op must not evict the work
// resident in L2) and cost ~2% above it. kStreamLoads/kStreamStores flip the
// policy in one place.
//
// Grid size: ceil(chunks / (2 * block)) CTAs, capped at 32 per SM, i.e. the grid
// follows the work instead of a one-wave persistent cap. On the bf16 shapes a
// grid that leaves each thread four chunks runs ~10% slower than one that leaves
// two, so more resident threads beat a bounded grid; the cap only bounds
// multi-GB tensors, where every CTA is many waves deep anyway.
//
// Alias: an exact in-place view is supported because every output element is
// produced by the single thread that reads the corresponding inputs, so no
// thread ever reads a value another thread has already overwritten.
//
// Reference: the kernel shape and alias rules follow
// 3rd/csrc/model_ops/residual.cu (Apache-2.0); see THIRD_PARTY_NOTICES.md.
// =============================================================================
#include <cstdint>

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
#include "csrc/model_ops/residual_kernels.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace kops::model_ops {
namespace {

using dev::Pack;
using dev::from_f;
using dev::to_f;

/// Streaming (evict-first) 16 B accesses; see the header comment.
constexpr bool kStreamLoads = true;
constexpr bool kStreamStores = true;
/// 256 threads, two chunks per thread, grid capped at 32 CTAs per SM.
constexpr int kBlock = 256;
constexpr int kMaxCtasPerSm = 32;

/// 16 B load, plain or ld.global.cs (evict-first).
template <typename T>
__device__ __forceinline__ uint4 load16(const T* __restrict__ p) {
  if constexpr (kStreamLoads) {
    uint4 u;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
        : "l"(p));
    return u;
  } else {
    return __ldg(reinterpret_cast<const uint4*>(p));
  }
}

/// N elements of in_t (8, 16 or 32 B) into `dst`; 16 B pieces use load16.
template <typename in_t, int N>
__device__ __forceinline__ void load_elems(const in_t* __restrict__ p, in_t (&dst)[N]) {
  constexpr int kBytes = N * static_cast<int>(sizeof(in_t));
  constexpr int kPer16 = 16 / static_cast<int>(sizeof(in_t));
  static_assert(kBytes == 8 || kBytes == 16 || kBytes == 32, "chunk is 8, 16 or 32 B");
  if constexpr (kBytes == 8) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    __builtin_memcpy(dst, &u, sizeof(u));
  } else {
#pragma unroll
    for (int v = 0; v < kBytes / 16; ++v) {
      const uint4 u = load16(p + v * kPer16);
      __builtin_memcpy(dst + v * kPer16, &u, sizeof(u));
    }
  }
}

/// One 16 B output pack of VEC = 16 / sizeof(T) elements.
template <typename T, int VEC>
__device__ __forceinline__ void store_pack(T* __restrict__ p, const Pack<T, VEC>& v) {
  static_assert(sizeof(Pack<T, VEC>) == 16, "a pack is one 16-byte access");
  if constexpr (kStreamStores) {
    const uint4 u = *reinterpret_cast<const uint4*>(&v);
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(u.x), "r"(u.y),
                 "r"(u.z), "r"(u.w)
                 : "memory");
  } else {
    *reinterpret_cast<Pack<T, VEC>*>(p) = v;
  }
}

/// One chunk of N elements: N = 16 / sizeof(out_t) for the vector path, 1 for
/// the scalar path. `p` is the chunk index in [0, rows * cpr).
template <typename out_t, typename in_t, int N, bool HasC>
__device__ __forceinline__ void residual_chunk(out_t* __restrict__ out, const int64_t out_stride,
                                               const in_t* __restrict__ a, const int64_t a_stride,
                                               const float a_scale, const in_t* __restrict__ b,
                                               const int64_t b_stride, const float b_scale,
                                               const in_t* __restrict__ c, const int64_t c_stride,
                                               const float c_scale, const int64_t cpr,
                                               const int64_t p) {
  const int64_t row = p / cpr;
  const int64_t col = (p - row * cpr) * N;
  const in_t* ap = a + row * a_stride + col;
  const in_t* bp = b + row * b_stride + col;
  const in_t* cp = HasC ? (c + row * c_stride + col) : nullptr;
  out_t* op = out + row * out_stride + col;
  if constexpr (N == 1) {
    float x = a_scale * to_f<in_t>(*ap) + b_scale * to_f<in_t>(*bp);
    if (HasC) x += c_scale * to_f<in_t>(*cp);
    from_f<out_t>(*op, x);
  } else {
    in_t av[N], bv[N];
    load_elems<in_t, N>(ap, av);
    load_elems<in_t, N>(bp, bv);
    [[maybe_unused]] in_t cv[HasC ? N : 1];
    if constexpr (HasC) load_elems<in_t, N>(cp, cv);
    Pack<out_t, N> py;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      float x = a_scale * to_f<in_t>(av[j]) + b_scale * to_f<in_t>(bv[j]);
      if constexpr (HasC) x += c_scale * to_f<in_t>(cv[j]);
      from_f<out_t>(py.x[j], x);
    }
    store_pack<out_t, N>(op, py);
  }
}

/// residual_add: out = a_scale * a + b_scale * b (+ c_scale * c).
template <typename out_t, typename in_t, int N, bool HasC>
__global__ void residual_add_kernel(out_t* __restrict__ out, const int64_t out_stride,
                                    const in_t* __restrict__ a, const int64_t a_stride,
                                    const float a_scale, const in_t* __restrict__ b,
                                    const int64_t b_stride, const float b_scale,
                                    const in_t* __restrict__ c, const int64_t c_stride,
                                    const float c_scale, const int rows, const int d) {
  const int64_t cpr = (N == 1) ? d : (d / N);
  const int64_t total = static_cast<int64_t>(rows) * cpr;
  const int64_t bid = blockIdx.x;
  const int64_t bdim = blockDim.x;
  const int64_t step = 2 * gridDim.x * bdim;
  for (int64_t p = bid * bdim * 2 + threadIdx.x; p < total; p += step) {
    residual_chunk<out_t, in_t, N, HasC>(out, out_stride, a, a_stride, a_scale, b, b_stride,
                                         b_scale, c, c_stride, c_scale, cpr, p);
    const int64_t q = p + bdim;
    if (q < total) {
      residual_chunk<out_t, in_t, N, HasC>(out, out_stride, a, a_stride, a_scale, b, b_stride,
                                           b_scale, c, c_stride, c_scale, cpr, q);
    }
  }
}

/// cast: same launch shape and chunk definition as residual_add_kernel.
template <typename out_t, typename in_t, int N>
__device__ __forceinline__ void cast_chunk(out_t* __restrict__ out, const int64_t out_stride,
                                           const in_t* __restrict__ in, const int64_t in_stride,
                                           const float scale, const int64_t cpr, const int64_t p) {
  const int64_t row = p / cpr;
  const int64_t col = (p - row * cpr) * N;
  const in_t* src = in + row * in_stride + col;
  out_t* dst = out + row * out_stride + col;
  if constexpr (N == 1) {
    const float x = to_f<in_t>(*src);
    from_f<out_t>(*dst, scale == 1.0f ? x : scale * x);
  } else {
    in_t xv[N];
    load_elems<in_t, N>(src, xv);
    Pack<out_t, N> py;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float x = to_f<in_t>(xv[j]);
      from_f<out_t>(py.x[j], scale == 1.0f ? x : scale * x);
    }
    store_pack<out_t, N>(dst, py);
  }
}

template <typename out_t, typename in_t, int N>
__global__ void cast_kernel(out_t* __restrict__ out, const int64_t out_stride,
                            const in_t* __restrict__ in, const int64_t in_stride, const float scale,
                            const int rows, const int d) {
  const int64_t cpr = (N == 1) ? d : (d / N);
  const int64_t total = static_cast<int64_t>(rows) * cpr;
  const int64_t bid = blockIdx.x;
  const int64_t bdim = blockDim.x;
  const int64_t step = 2 * gridDim.x * bdim;
  for (int64_t p = bid * bdim * 2 + threadIdx.x; p < total; p += step) {
    cast_chunk<out_t, in_t, N>(out, out_stride, in, in_stride, scale, cpr, p);
    const int64_t q = p + bdim;
    if (q < total) cast_chunk<out_t, in_t, N>(out, out_stride, in, in_stride, scale, cpr, q);
  }
}

struct RowView {
  const void* data;
  int64_t rows;
  int64_t d;
  int64_t stride;
  int64_t elem_bytes;
  uintptr_t begin;
  uintptr_t end;
};

int64_t element_bytes(DType t) { return (dtype_bits(t) + 7) / 8; }

bool is_supported_float(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

Status make_row_view(const Tensor& t, const char* name, RowView* out) {
  // Alignment is checked on the fast path; the scalar path accepts any row
  // stride, so the contract validation here must not reject a legal tail.
  Status s = check_tensor(t, t.dtype, -1, 1, name);
  if (s != Status::kSuccess) return s;
  if (!is_supported_float(t.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s", name, dtype_name(t.dtype));
  }
  if (t.ndim <= 0) return detail::set_error(Status::kUnsupportedShape, "%s: ndim %d", name, t.ndim);
  const int64_t d = t.shape[t.ndim - 1];
  if (d <= 0) return detail::set_error(Status::kUnsupportedShape, "%s: last dim %lld", name,
                                       static_cast<long long>(d));
  if (t.ndim > 1 && t.stride[t.ndim - 1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last dim must be contiguous", name);
  }
  // Collapse the leading dimensions only when they are row-major contiguous;
  // the kernel has one row stride.
  int64_t stride = (t.ndim > 1) ? t.stride[t.ndim - 2] : 0;
  for (int32_t i = 0; i + 2 < t.ndim; ++i) {
    if (t.stride[i] != t.stride[i + 1] * t.shape[i + 1]) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "%s: leading dimensions do not collapse to one row stride", name);
    }
  }
  int64_t rows = 1;
  for (int32_t i = 0; i < t.ndim - 1; ++i) rows *= t.shape[i];
  const int64_t elem = element_bytes(t.dtype);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
  if ((begin & 15u) != 0) {
    return detail::set_error(Status::kMisaligned, "%s: base address violates 16-byte alignment", name);
  }
  out->data = reinterpret_cast<const void*>(begin);
  out->rows = rows;
  out->d = d;
  out->stride = stride;
  out->elem_bytes = elem;
  out->begin = begin;
  out->end = begin + static_cast<uintptr_t>((rows > 0 ? (rows - 1) * stride + d : 0) * elem);
  return Status::kSuccess;
}

Status check_same_shape(const RowView& a, const RowView& b, const char* what) {
  if (a.rows != b.rows || a.d != b.d) {
    return detail::set_error(Status::kInvalidArgument, "%s: shape mismatch", what);
  }
  return Status::kSuccess;
}

Status check_alias(const RowView& out, const RowView& in, const char* what) {
  const bool disjoint = out.end <= in.begin || in.end <= out.begin;
  if (disjoint) return Status::kSuccess;
  const bool exact = out.begin == in.begin && out.elem_bytes == in.elem_bytes &&
                     (out.rows <= 1 || out.stride == in.stride);
  if (exact) return Status::kSuccess;
  return detail::set_error(Status::kUnsupportedLayout,
                           "%s: output partially overlaps an input; use an exact in-place view", what);
}

/// Elements per chunk: one 16 B pack of the output dtype on the vector path, one
/// element otherwise. Every row start must stay 16 B aligned, which holds when
/// the base address is 16 B aligned (checked by make_row_view) and every row
/// stride is a multiple of N.
int chunk_elements(int64_t d, const RowView* views, int nviews, int pack) {
  for (int i = 0; i < nviews; ++i) {
    if (views[i].rows > 1 && views[i].stride % pack != 0) return 1;
  }
  return (d % pack == 0) ? pack : 1;
}

/// Chunks per row for a chunk of `n` elements (n == 1 is the scalar path).
int64_t chunks_per_row(int64_t d, int n) { return (n == 1) ? d : d / n; }

/// One CTA per two chunks, capped: the grid follows the work (see the header
/// comment) but never explodes for multi-GB tensors.
int grid_for(int64_t work_chunks, int sms) {
  int64_t g = (work_chunks + 2 * kBlock - 1) / (2 * kBlock);
  if (g < 1) g = 1;
  if (sms > 0) {
    const int64_t cap = static_cast<int64_t>(sms) * kMaxCtasPerSm;
    if (g > cap) g = cap;
  }
  return static_cast<int>(g > INT32_MAX ? INT32_MAX : g);
}

template <typename out_t, typename in_t, bool HasC>
Status launch_residual(const RowView& out, const RowView& a, float as, const RowView& b, float bs,
                       const RowView& c, float cs, const LaunchContext& ctx) {
  if (out.rows == 0) return Status::kSuccess;
  if (out.rows > INT32_MAX || out.d > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "residual_add: rows %lld, d %lld exceed int32",
                             static_cast<long long>(out.rows), static_cast<long long>(out.d));
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  constexpr int PACK = 16 / static_cast<int>(sizeof(out_t));
  const RowView views[4] = {out, a, b, c};
  const int n = chunk_elements(out.d, views, HasC ? 4 : 3, PACK);
  const int64_t chunks = out.rows * chunks_per_row(out.d, n);
  const int grid = grid_for(chunks, ctx.compute_units);
  const auto* a_ptr = static_cast<const in_t*>(a.data);
  const auto* b_ptr = static_cast<const in_t*>(b.data);
  const auto* c_ptr = HasC ? static_cast<const in_t*>(c.data) : nullptr;
  auto* out_ptr = static_cast<out_t*>(const_cast<void*>(out.data));
  const int rows = static_cast<int>(out.rows);
  const int d = static_cast<int>(out.d);
#define KOPS_LAUNCH_RES(N, HC)                                                                       \
  residual_add_kernel<out_t, in_t, N, HC><<<grid, kBlock, 0, stream>>>(                              \
      out_ptr, out.stride, a_ptr, a.stride, as, b_ptr, b.stride, bs, c_ptr,                          \
      HasC ? c.stride : 0, cs, rows, d)
  if (n == PACK) {
    if (HasC) KOPS_LAUNCH_RES(PACK, true);
    else KOPS_LAUNCH_RES(PACK, false);
  } else {
    if (HasC) KOPS_LAUNCH_RES(1, true);
    else KOPS_LAUNCH_RES(1, false);
  }
#undef KOPS_LAUNCH_RES
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "residual_add launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename out_t, typename in_t>
Status launch_cast(const RowView& out, const RowView& in, float scale, const LaunchContext& ctx) {
  if (out.rows == 0) return Status::kSuccess;
  if (out.rows > INT32_MAX || out.d > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "cast: rows %lld, d %lld exceed int32",
                             static_cast<long long>(out.rows), static_cast<long long>(out.d));
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  constexpr int PACK = 16 / static_cast<int>(sizeof(out_t));
  const RowView views[2] = {out, in};
  const int n = chunk_elements(out.d, views, 2, PACK);
  const int64_t chunks = out.rows * chunks_per_row(out.d, n);
  const int grid = grid_for(chunks, ctx.compute_units);
  auto* out_ptr = static_cast<out_t*>(const_cast<void*>(out.data));
  const auto* in_ptr = static_cast<const in_t*>(in.data);
  const int rows = static_cast<int>(out.rows);
  const int d = static_cast<int>(out.d);
#define KOPS_LAUNCH_CAST(N)                                                                          \
  cast_kernel<out_t, in_t, N><<<grid, kBlock, 0, stream>>>(out_ptr, out.stride, in_ptr, in.stride,    \
                                                           scale, rows, d)
  if (n == PACK) KOPS_LAUNCH_CAST(PACK);
  else KOPS_LAUNCH_CAST(1);
#undef KOPS_LAUNCH_CAST
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "cast launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

}  // namespace

Status residual_add(const Tensor& out, const Tensor& a, float a_scale, const Tensor& b,
                    float b_scale, const Tensor& c, float c_scale, const LaunchContext& ctx) {
  RowView vo{}, va{}, vb{}, vc{};
  Status s = make_row_view(out, "residual_add.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(a, "residual_add.a", &va);
  if (s != Status::kSuccess) return s;
  s = make_row_view(b, "residual_add.b", &vb);
  if (s != Status::kSuccess) return s;
  s = check_same_shape(vo, va, "residual_add a");
  if (s != Status::kSuccess) return s;
  s = check_same_shape(vo, vb, "residual_add b");
  if (s != Status::kSuccess) return s;
  const bool has_c = tensor_present(c);
  if (has_c) {
    s = make_row_view(c, "residual_add.c", &vc);
    if (s != Status::kSuccess) return s;
    s = check_same_shape(vo, vc, "residual_add c");
    if (s != Status::kSuccess) return s;
  }
  if (a.dtype != b.dtype || (has_c && c.dtype != a.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "residual_add: inputs must share one dtype, got %s / %s / %s",
                             dtype_name(a.dtype), dtype_name(b.dtype),
                             has_c ? dtype_name(c.dtype) : "absent");
  }
  s = check_alias(vo, va, "residual_add");
  if (s != Status::kSuccess) return s;
  s = check_alias(vo, vb, "residual_add");
  if (s != Status::kSuccess) return s;
  if (has_c) {
    s = check_alias(vo, vc, "residual_add");
    if (s != Status::kSuccess) return s;
  }
  // Dispatch over the input dtype; the output dtype is independent.
#define KOPS_DISPATCH_IN(IN)                                                                        \
  do {                                                                                              \
    if (out.dtype == DType::kFloat32) {                                                             \
      return has_c ? launch_residual<float, IN, true>(vo, va, a_scale, vb, b_scale, vc, c_scale, ctx) \
                   : launch_residual<float, IN, false>(vo, va, a_scale, vb, b_scale, vc, c_scale, ctx); \
    }                                                                                               \
    if (out.dtype == a.dtype) {                                                                     \
      return has_c ? launch_residual<IN, IN, true>(vo, va, a_scale, vb, b_scale, vc, c_scale, ctx)  \
                   : launch_residual<IN, IN, false>(vo, va, a_scale, vb, b_scale, vc, c_scale, ctx); \
    }                                                                                               \
    return detail::set_error(Status::kUnsupportedDtype, "residual_add: out %s, in %s",              \
                             dtype_name(out.dtype), dtype_name(a.dtype));                           \
  } while (0)
  switch (a.dtype) {
    case DType::kFloat32: KOPS_DISPATCH_IN(float);
    case DType::kFloat16: KOPS_DISPATCH_IN(__half);
    case DType::kBFloat16: KOPS_DISPATCH_IN(__nv_bfloat16);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "residual_add: %s", dtype_name(a.dtype));
  }
#undef KOPS_DISPATCH_IN
}

Status cast(const Tensor& out, const Tensor& in, float scale, const LaunchContext& ctx) {
  RowView vo{}, vi{};
  Status s = make_row_view(out, "cast.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(in, "cast.in", &vi);
  if (s != Status::kSuccess) return s;
  s = check_same_shape(vo, vi, "cast");
  if (s != Status::kSuccess) return s;
  s = check_alias(vo, vi, "cast");
  if (s != Status::kSuccess) return s;
#define KOPS_DISPATCH_CAST(OUT, IN) return launch_cast<OUT, IN>(vo, vi, scale, ctx)
  if (in.dtype == DType::kFloat32) {
    if (out.dtype == DType::kFloat32) KOPS_DISPATCH_CAST(float, float);
    if (out.dtype == DType::kFloat16) KOPS_DISPATCH_CAST(__half, float);
    if (out.dtype == DType::kBFloat16) KOPS_DISPATCH_CAST(__nv_bfloat16, float);
  } else if (in.dtype == DType::kFloat16) {
    if (out.dtype == DType::kFloat32) KOPS_DISPATCH_CAST(float, __half);
    if (out.dtype == DType::kFloat16) KOPS_DISPATCH_CAST(__half, __half);
    if (out.dtype == DType::kBFloat16) KOPS_DISPATCH_CAST(__nv_bfloat16, __half);
  } else if (in.dtype == DType::kBFloat16) {
    if (out.dtype == DType::kFloat32) KOPS_DISPATCH_CAST(float, __nv_bfloat16);
    if (out.dtype == DType::kFloat16) KOPS_DISPATCH_CAST(__half, __nv_bfloat16);
    if (out.dtype == DType::kBFloat16) KOPS_DISPATCH_CAST(__nv_bfloat16, __nv_bfloat16);
  }
#undef KOPS_DISPATCH_CAST
  return detail::set_error(Status::kUnsupportedDtype, "cast: %s -> %s", dtype_name(in.dtype),
                           dtype_name(out.dtype));
}

}  // namespace kops::model_ops
