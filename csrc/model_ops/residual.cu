// =============================================================================
// kernel-ops | csrc/model_ops/residual.cu
// B0: residual_add and cast. Both view their operands as [rows, d] with a row
// stride in elements and unit stride along d. Arithmetic is fp32 and rounded
// once at the store. The wide path uses 16-byte bundles; a scalar path handles
// every tail and every layout that is not bundle-aligned.
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

template <typename out_t, typename in_t, int N, bool HasC>
__global__ void residual_add_kernel(out_t* __restrict__ out, const int64_t out_stride,
                                    const in_t* __restrict__ a, const int64_t a_stride, const float a_scale,
                                    const in_t* __restrict__ b, const int64_t b_stride, const float b_scale,
                                    const in_t* __restrict__ c, const int64_t c_stride, const float c_scale,
                                    const int64_t rows, const int d) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const in_t* ap = a + row * a_stride;
  const in_t* bp = b + row * b_stride;
  const in_t* cp = HasC ? (c + row * c_stride) : nullptr;
  out_t* op = out + row * out_stride;
  const int nvec = d / N;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * N;
    const Pack<in_t, N> pa = dev::load_pack<N>(ap + k);
    const Pack<in_t, N> pb = dev::load_pack<N>(bp + k);
    Pack<in_t, N> pc;
    if (HasC) pc = dev::load_pack<N>(cp + k);
    Pack<out_t, N> py;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      float x = a_scale * to_f<in_t>(pa.x[j]) + b_scale * to_f<in_t>(pb.x[j]);
      if (HasC) x += c_scale * to_f<in_t>(pc.x[j]);
      from_f<out_t>(py.x[j], x);
    }
    dev::store_pack<N>(op + k, py);
  }
  for (int i = nvec * N + threadIdx.x; i < d; i += blockDim.x) {
    float x = a_scale * to_f<in_t>(ap[i]) + b_scale * to_f<in_t>(bp[i]);
    if (HasC) x += c_scale * to_f<in_t>(cp[i]);
    from_f<out_t>(op[i], x);
  }
}

template <typename out_t, typename in_t, int N>
__global__ void cast_kernel(out_t* __restrict__ out, const int64_t out_stride,
                            const in_t* __restrict__ in, const int64_t in_stride, const float scale,
                            const int64_t rows, const int d) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const in_t* src = in + row * in_stride;
  out_t* dst = out + row * out_stride;
  const int nvec = d / N;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * N;
    const Pack<in_t, N> px = dev::load_pack<N>(src + k);
    Pack<out_t, N> py;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float x = to_f<in_t>(px.x[j]);
      from_f<out_t>(py.x[j], scale == 1.0f ? x : scale * x);
    }
    dev::store_pack<N>(dst + k, py);
  }
  for (int i = nvec * N + threadIdx.x; i < d; i += blockDim.x) {
    const float x = to_f<in_t>(src[i]);
    from_f<out_t>(dst[i], scale == 1.0f ? x : scale * x);
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

int block_for(int64_t d, int vec) {
  int n = static_cast<int>((d + vec - 1) / vec);
  n = ((n + 31) / 32) * 32;
  if (n < 64) n = 64;
  if (n > 256) n = 256;
  return n;
}

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename out_t, typename in_t>
bool can_vector(int64_t d, int64_t out_stride, int64_t a_stride, int64_t b_stride, int64_t c_stride,
                bool has_c, const void* out, const void* a, const void* b, const void* c) {
  constexpr int N = 16 / static_cast<int>(sizeof(out_t) > sizeof(in_t) ? sizeof(out_t) : sizeof(in_t));
  if (N <= 1 || d % N != 0) return false;
  auto ok = [&](int64_t s) { return s % N == 0; };
  return ok(out_stride) && ok(a_stride) && ok(b_stride) && (!has_c || ok(c_stride)) &&
         is_aligned16(out) && is_aligned16(a) && is_aligned16(b) && (!has_c || is_aligned16(c));
}

template <typename out_t, typename in_t, bool HasC>
Status launch_residual(const RowView& out, const RowView& a, float as, const RowView& b, float bs,
                       const RowView& c, float cs, const LaunchContext& ctx) {
  if (out.rows == 0) return Status::kSuccess;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  constexpr int Wide = 16 / static_cast<int>(sizeof(out_t) > sizeof(in_t) ? sizeof(out_t) : sizeof(in_t));
  const int n = (Wide > 1 &&
                 can_vector<out_t, in_t>(out.d, out.stride, a.stride, b.stride, c.stride, HasC,
                                         out.data, a.data, b.data, c.data))
                    ? Wide
                    : 1;
  const int block = block_for(out.d, n);
  const auto* a_ptr = static_cast<const in_t*>(a.data);
  const auto* b_ptr = static_cast<const in_t*>(b.data);
  const auto* c_ptr = HasC ? static_cast<const in_t*>(c.data) : nullptr;
  auto* out_ptr = static_cast<out_t*>(const_cast<void*>(out.data));
#define KOPS_LAUNCH_RES(OUT, IN, N, HC)                                                          \
  residual_add_kernel<OUT, IN, N, HC><<<static_cast<unsigned>(out.rows), block, 0, stream>>>(    \
      out_ptr, out.stride, a_ptr, a.stride, as, b_ptr, b.stride, bs, c_ptr, c.stride, cs,        \
      out.rows, static_cast<int>(out.d))
  if (n == Wide) {
    if (HasC) KOPS_LAUNCH_RES(out_t, in_t, Wide, true);
    else KOPS_LAUNCH_RES(out_t, in_t, Wide, false);
  } else {
    if (HasC) KOPS_LAUNCH_RES(out_t, in_t, 1, true);
    else KOPS_LAUNCH_RES(out_t, in_t, 1, false);
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
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  constexpr int Wide = 16 / static_cast<int>(sizeof(out_t) > sizeof(in_t) ? sizeof(out_t) : sizeof(in_t));
  const bool vec = Wide > 1 && out.d % Wide == 0 && out.stride % Wide == 0 && in.stride % Wide == 0 &&
                   is_aligned16(out.data) && is_aligned16(in.data);
  const int n = vec ? Wide : 1;
  const int block = block_for(out.d, n);
  auto* out_ptr = static_cast<out_t*>(const_cast<void*>(out.data));
  const auto* in_ptr = static_cast<const in_t*>(in.data);
#define KOPS_LAUNCH_CAST(OUT, IN, N)                                                             \
  cast_kernel<OUT, IN, N><<<static_cast<unsigned>(out.rows), block, 0, stream>>>(                \
      out_ptr, out.stride, in_ptr, in.stride, scale, out.rows, static_cast<int>(out.d))
  if (n == Wide) KOPS_LAUNCH_CAST(out_t, in_t, Wide);
  else KOPS_LAUNCH_CAST(out_t, in_t, 1);
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
