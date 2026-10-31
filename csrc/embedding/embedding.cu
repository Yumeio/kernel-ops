// =============================================================================
// kernel-ops | csrc/embedding/embedding.cu
// B1.2 first implementations: embedding_lookup, embedding_with_position,
// gather_rows, scatter_rows and logits_postprocess.
//
// All five kernels are memory-bound row copies or element-wise fix-ups: one CTA
// per output row, 16-byte bundles when every address allows it, and a scalar
// tail that covers `hidden % VEC`. Arithmetic (scale, position add, softcap) is
// fp32 with one rounding at the store; gather/scatter move bits and are exact.
//
// Contracts kept by this file
//   * Lookup masks instead of clamping. An id inside [vocab_start, vocab_end)
//     copies and scales the shard row; every other id writes a zero row and the
//     caller all-reduces across TP ranks. An id outside [0, vocab_size_global)
//     is a caller error: ids are device data, so it cannot be checked here
//     without a synchronisation.
//   * gather_rows / scatter_rows take int32 row indices. The device signatures
//     carry no table row count, so an out-of-range index cannot be bounds
//     checked without a sync; it is a caller contract. Duplicate *gather*
//     indices are harmless (the input is read-only and every duplicate reads
//     the same row). Duplicate *scatter* destinations race: the last CTA to
//     store wins, so a repeated destination row is not deterministic and the
//     caller must not produce one.
//   * logits_postprocess may run in place when logits is fp32: the map is
//     one-to-one over (row, col) and each element is loaded before it is
//     stored. A partial overlap (different row strides / offsets) is rejected
//     because a store could then clobber a value another thread still needs.
//
// Target support: arch-generic (no cp.async, no TMA, no wgmma); exact-target
// selection is the registry's job.
//
// Reference: 3rd/csrc/embedding/embedding.cu (Apache-2.0) for the kernel shape,
// the vocab-shard masking rule and the softcap ordering. The at::Tensor entry
// points, framework dispatch and the byte-word copy helpers are not ported;
// this file uses kops::Tensor / LaunchContext / Status.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/embedding/embedding_kernels.cuh"
#include "csrc/embedding/ops.h"

namespace kops::embed {

// Shorthand for the device scalar helpers in common/device_impl.cuh (this
// translation unit only).
using dev::from_f;
using dev::to_f;

namespace {

constexpr int32_t kIntMax = std::numeric_limits<int32_t>::max();
/// Columns per thread in logits_postprocess; 4 x 4 B = one 16-byte store.
constexpr int kLogitsVec = 4;
constexpr int kLogitsBlock = 256;
constexpr int kMaxGridY = 65535;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

/// Index load: token_ids is int32 or int64 (index_t), positions is int64.
template <typename index_t>
__device__ __forceinline__ int64_t load_index(const index_t* __restrict__ ids, int64_t i) {
  return static_cast<int64_t>(ids[i]);
}

/// x = logit * scale, then the optional final softcap. cap <= 0 is the identity,
/// so the common path is one correctly rounded product.
__device__ __forceinline__ float postprocess(float x, float scale, float cap) {
  const float y = __fmul_rn(x, scale);
  return softcap(y, cap);
}

}  // namespace

// -----------------------------------------------------------------------------
// Kernels (definitions of the contracts in csrc/embedding/embedding_kernels.cuh)
// -----------------------------------------------------------------------------

template <typename scalar_t, typename index_t, int VEC_SIZE>
__global__ void embedding_lookup_kernel(
    scalar_t* __restrict__ out, const int64_t out_stride,
    const index_t* __restrict__ token_ids,
    const scalar_t* __restrict__ weight, const int64_t weight_stride,
    const int64_t vocab_start, const int64_t vocab_end,
    const float scale, const int num_tokens, const int hidden) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  scalar_t* dst = out + static_cast<int64_t>(token) * out_stride;
  const int64_t id = load_index<index_t>(token_ids, token);
  const bool in_shard = id >= vocab_start && id < vocab_end;
  const int nvec = hidden / VEC_SIZE;

  if (!in_shard) {
    // Vocab-parallel rule: a token outside this rank's shard contributes zeros
    // and the TP all-reduce supplies the owning rank's row.
    scalar_t z;
    from_f<scalar_t>(z, 0.0f);
    Vec<scalar_t, VEC_SIZE> zero;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) zero.data[j] = z;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      store_vec<scalar_t, VEC_SIZE>(dst + static_cast<int64_t>(i) * VEC_SIZE, zero);
    }
    for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden; i += blockDim.x) dst[i] = z;
    return;
  }

  const scalar_t* src = weight + (id - vocab_start) * weight_stride;
  const bool do_scale = scale != 1.0f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * VEC_SIZE;
    Vec<scalar_t, VEC_SIZE> v = load_vec<scalar_t, VEC_SIZE>(src + k);
    if (do_scale) {
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        from_f<scalar_t>(v.data[j], __fmul_rn(to_f<scalar_t>(v.data[j]), scale));
      }
    }
    store_vec<scalar_t, VEC_SIZE>(dst + k, v);
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden; i += blockDim.x) {
    const float x = to_f<scalar_t>(src[i]);
    from_f<scalar_t>(dst[i], do_scale ? __fmul_rn(x, scale) : x);
  }
}

template <typename scalar_t, typename index_t, int VEC_SIZE>
__global__ void embedding_with_position_kernel(
    scalar_t* __restrict__ out,
    const index_t* __restrict__ token_ids,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ token_weight,
    const scalar_t* __restrict__ pos_weight,
    const int64_t vocab_start, const int64_t vocab_end, const int position_offset,
    const int is_first_tp_rank,
    const int num_tokens, const int hidden) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  const int64_t id = load_index<index_t>(token_ids, token);
  const bool in_shard = id >= vocab_start && id < vocab_end;
  // position_offset shifts the table (OPT stores position 0 at row 2).
  const int64_t position = positions[token] + static_cast<int64_t>(position_offset);
  // Positions are added on one TP rank only, before the all-reduce. The ABI has
  // no position-table row count, so the upper bound is a caller contract; only
  // a negative position is masked here.
  const bool has_pos = is_first_tp_rank != 0 && position >= 0;
  const scalar_t* tok_row = token_weight + (in_shard ? (id - vocab_start) : 0) * hidden;
  const scalar_t* pos_row = has_pos ? (pos_weight + position * hidden) : pos_weight;
  scalar_t* dst = out + static_cast<int64_t>(token) * hidden;
  const int nvec = hidden / VEC_SIZE;

  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * VEC_SIZE;
    Vec<scalar_t, VEC_SIZE> a;
    Vec<scalar_t, VEC_SIZE> b;
    Vec<scalar_t, VEC_SIZE> r;
    if (in_shard) a = load_vec<scalar_t, VEC_SIZE>(tok_row + k);
    if (has_pos) b = load_vec<scalar_t, VEC_SIZE>(pos_row + k);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = in_shard ? to_f<scalar_t>(a.data[j]) : 0.0f;
      const float y = has_pos ? to_f<scalar_t>(b.data[j]) : 0.0f;
      // Exactly one term present keeps the bit pattern; both present is one add.
      const float v = (in_shard && has_pos) ? __fadd_rn(x, y) : (in_shard ? x : y);
      from_f<scalar_t>(r.data[j], v);
    }
    store_vec<scalar_t, VEC_SIZE>(dst + k, r);
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden; i += blockDim.x) {
    const float x = in_shard ? to_f<scalar_t>(tok_row[i]) : 0.0f;
    const float y = has_pos ? to_f<scalar_t>(pos_row[i]) : 0.0f;
    from_f<scalar_t>(dst[i], (in_shard && has_pos) ? __fadd_rn(x, y) : (in_shard ? x : y));
  }
}

/// out[i] = input[rows[i]]. The input is read-only, so repeated indices are
/// deterministic. No bounds check: the ABI carries no input row count.
template <typename scalar_t, int VEC_SIZE>
__global__ void gather_rows_kernel(
    scalar_t* __restrict__ out, const int64_t out_stride,
    const scalar_t* __restrict__ input, const int64_t input_stride,
    const int32_t* __restrict__ rows, const int num_rows, const int hidden) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int64_t r = static_cast<int64_t>(rows[row]);
  scalar_t* dst = out + static_cast<int64_t>(row) * out_stride;
  const scalar_t* src = input + r * input_stride;
  const int nvec = hidden / VEC_SIZE;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * VEC_SIZE;
    store_vec<scalar_t, VEC_SIZE>(dst + k, load_vec<scalar_t, VEC_SIZE>(src + k));
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

/// out[dst_rows[i]] = src[i]. Rows of `out` that no index names keep their
/// previous value. Repeated destination rows race (last store wins), so the
/// caller must pass unique indices. No bounds check: see gather_rows.
template <typename scalar_t, int VEC_SIZE>
__global__ void scatter_rows_kernel(
    scalar_t* __restrict__ out, const int64_t out_stride,
    const scalar_t* __restrict__ src, const int64_t src_stride,
    const int32_t* __restrict__ dst_rows, const int num_src_rows, const int hidden) {
  const int row = blockIdx.x;
  if (row >= num_src_rows) return;
  const int64_t r = static_cast<int64_t>(dst_rows[row]);
  scalar_t* dst = out + r * out_stride;
  const scalar_t* from = src + static_cast<int64_t>(row) * src_stride;
  const int nvec = hidden / VEC_SIZE;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int64_t k = static_cast<int64_t>(i) * VEC_SIZE;
    store_vec<scalar_t, VEC_SIZE>(dst + k, load_vec<scalar_t, VEC_SIZE>(from + k));
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden; i += blockDim.x) dst[i] = from[i];
}

/// grid.x tiles the columns, grid.y strides over rows. The 16-byte path is a
/// uniform (warp-invariant) decision, so it never introduces divergence.
template <typename in_t>
__global__ void logits_postprocess_kernel(
    float* __restrict__ out, const int64_t out_stride,
    const in_t* __restrict__ logits, const int64_t logits_stride,
    const float scale, const float final_softcap,
    const int64_t vocab_start, const int64_t vocab_size,
    const int num_rows, const int v_shard) {
  const bool wide = v_shard % kLogitsVec == 0 && out_stride % kLogitsVec == 0 &&
                    logits_stride % kLogitsVec == 0 &&
                    (reinterpret_cast<uintptr_t>(out) % (kLogitsVec * sizeof(float))) == 0 &&
                    (reinterpret_cast<uintptr_t>(logits) % (kLogitsVec * sizeof(in_t))) == 0;
  for (int64_t row = blockIdx.y; row < num_rows; row += gridDim.y) {
    float* dst = out + row * out_stride;
    const in_t* src = logits + row * logits_stride;
    if (wide) {
      for (int64_t col = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kLogitsVec;
           col < v_shard; col += static_cast<int64_t>(gridDim.x) * blockDim.x * kLogitsVec) {
        const Vec<in_t, kLogitsVec> x = load_vec<in_t, kLogitsVec>(src + col);
        Vec<float, kLogitsVec> r;
#pragma unroll
        for (int j = 0; j < kLogitsVec; ++j) {
          r.data[j] = (vocab_start + col + j >= vocab_size)
                          ? neg_inf()
                          : postprocess(to_f<in_t>(x.data[j]), scale, final_softcap);
        }
        store_vec<float, kLogitsVec>(dst + col, r);
      }
      continue;
    }
    for (int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; col < v_shard;
         col += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      dst[col] = (vocab_start + col >= vocab_size)
                     ? neg_inf()
                     : postprocess(to_f<in_t>(src[col]), scale, final_softcap);
    }
  }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

namespace {

/// [rows, hidden] view over a 2-D tensor with a row stride in elements.
struct RowView {
  const void* data = nullptr;
  DType dtype = DType::kInvalid;
  int64_t rows = 0;
  int64_t hidden = 0;
  int64_t stride = 0;  // elements between consecutive rows
  int64_t elem_bytes = 0;
  uintptr_t begin = 0;
  uintptr_t end = 0;
};

int64_t element_bytes(DType t) { return (dtype_bits(t) + 7) / 8; }

bool is_supported_float(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

Status make_row_view(const Tensor& t, DType expected, const char* name, RowView* out) {
  Status s = check_tensor(t, expected, 2, 16, name);
  if (s != Status::kSuccess) return s;
  if (!is_supported_float(t.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s", name, dtype_name(t.dtype));
  }
  const int64_t hidden = t.shape[1];
  if (hidden > 1 && t.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: column stride %lld != 1", name,
                             static_cast<long long>(t.stride[1]));
  }
  const int64_t elem = element_bytes(t.dtype);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
  out->data = reinterpret_cast<const void*>(begin);
  out->dtype = t.dtype;
  out->rows = t.shape[0];
  out->hidden = hidden;
  out->stride = t.stride[0];
  out->elem_bytes = elem;
  out->begin = begin;
  const uintptr_t bytes = static_cast<uintptr_t>((t.shape[0] > 0 ? (t.shape[0] - 1) * t.stride[0] + hidden : 0) *
                                                 elem);
  out->end = begin + bytes;
  return Status::kSuccess;
}

/// int32/int64 index vector of `count` entries; returns the base pointer.
Status check_indices(const Tensor& t, int64_t count, const char* name, const void** data,
                     bool* is_int64) {
  const size_t alignment = (t.dtype == DType::kInt64) ? 8 : 4;
  Status s = check_tensor(t, t.dtype, 1, alignment, name);
  if (s != Status::kSuccess) return s;
  if (t.dtype == DType::kInt64) {
    *is_int64 = true;
  } else if (t.dtype == DType::kInt32) {
    *is_int64 = false;
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s is not int32/int64", name,
                             dtype_name(t.dtype));
  }
  if (t.shape[0] != count) {
    return detail::set_error(Status::kInvalidArgument, "%s: %lld indices for %lld rows", name,
                             static_cast<long long>(t.shape[0]), static_cast<long long>(count));
  }
  *data = static_cast<const char*>(t.data) + t.byte_offset;
  return Status::kSuccess;
}

Status check_no_overlap(const RowView& out, const RowView& other, const char* what,
                        const char* other_name) {
  if (out.end <= other.begin || other.end <= out.begin) return Status::kSuccess;
  return detail::set_error(Status::kUnsupportedLayout, "%s overlaps %s; use separate buffers", what,
                           other_name);
}

/// In-place is allowed only for the exact same view (logits_postprocess in fp32);
/// a partial overlap could let one store destroy a value another thread reads.
Status check_exact_or_disjoint(const RowView& out, const RowView& in, const char* what) {
  if (out.end <= in.begin || in.end <= out.begin) return Status::kSuccess;
  const bool exact = out.begin == in.begin && out.dtype == in.dtype && out.rows == in.rows &&
                     out.hidden == in.hidden && out.stride == in.stride;
  if (exact) return Status::kSuccess;
  return detail::set_error(Status::kUnsupportedLayout,
                           "%s: out partially overlaps logits; pass the same view for an in-place "
                           "update or a disjoint buffer",
                           what);
}

int block_for(int64_t width, int vec) {
  int64_t n = (width + vec - 1) / vec;
  n = ((n + 31) / 32) * 32;
  if (n < 64) n = 64;
  if (n > 256) n = 256;
  return static_cast<int>(n);
}

Status check_dim_range(const RowView& v, const char* what) {
  if (v.rows > kIntMax || v.hidden > kIntMax) {
    return detail::set_error(Status::kUnsupportedShape, "%s: [%lld, %lld] exceeds the int range",
                             what, static_cast<long long>(v.rows), static_cast<long long>(v.hidden));
  }
  return Status::kSuccess;
}

Status check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s launch: %s",
                                   what, cudaGetErrorString(err));
}

/// 16-byte bundle: 4 fp32, 8 fp16/bf16, 16 int8.
template <typename T>
constexpr int bundle() {
  return 16 / static_cast<int>(sizeof(T));
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------

template <typename T, typename index_t>
Status launch_lookup(const RowView& out, const void* ids, const RowView& weight,
                     int64_t vocab_start, int64_t vocab_end, float scale,
                     cudaStream_t stream) {
  constexpr int kVec = bundle<T>();
  auto* y = static_cast<T*>(const_cast<void*>(out.data));
  const auto* w = static_cast<const T*>(weight.data);
  const auto* id = static_cast<const index_t*>(ids);
  const int tokens = static_cast<int>(out.rows);
  const int hidden = static_cast<int>(out.hidden);
  embedding_lookup_kernel<T, index_t, kVec>
      <<<static_cast<unsigned>(tokens), block_for(hidden, kVec), 0, stream>>>(
          y, out.stride, id, w, weight.stride, vocab_start, vocab_end, scale, tokens, hidden);
  return check_launch("embedding_lookup");
}

template <typename T>
Status dispatch_lookup(const RowView& out, const void* ids, bool ids64, const RowView& weight,
                       int64_t vocab_start, int64_t vocab_end, float scale,
                       cudaStream_t stream) {
  if (ids64) {
    return launch_lookup<T, int64_t>(out, ids, weight, vocab_start, vocab_end, scale, stream);
  }
  return launch_lookup<T, int32_t>(out, ids, weight, vocab_start, vocab_end, scale, stream);
}

template <typename T, typename index_t>
Status launch_with_position(const RowView& out, const void* ids, const Tensor& positions,
                            const RowView& token_weight, const RowView& pos_weight,
                            int64_t vocab_start, int64_t vocab_end, int32_t position_offset,
                            bool is_first_tp_rank, cudaStream_t stream) {
  constexpr int kVec = bundle<T>();
  auto* y = static_cast<T*>(const_cast<void*>(out.data));
  const auto* id = static_cast<const index_t*>(ids);
  const auto* pos = reinterpret_cast<const int64_t*>(static_cast<const char*>(positions.data) +
                                                     positions.byte_offset);
  const auto* tok = static_cast<const T*>(token_weight.data);
  const auto* pw = static_cast<const T*>(pos_weight.data);
  const int tokens = static_cast<int>(out.rows);
  const int hidden = static_cast<int>(out.hidden);
  embedding_with_position_kernel<T, index_t, kVec>
      <<<static_cast<unsigned>(tokens), block_for(hidden, kVec), 0, stream>>>(
          y, id, pos, tok, pw, vocab_start, vocab_end, position_offset,
          is_first_tp_rank ? 1 : 0, tokens, hidden);
  return check_launch("embedding_with_position");
}

template <typename T>
Status dispatch_with_position(const RowView& out, const void* ids, bool ids64,
                              const Tensor& positions, const RowView& token_weight,
                              const RowView& pos_weight, int64_t vocab_start, int64_t vocab_end,
                              int32_t position_offset, bool is_first_tp_rank,
                              cudaStream_t stream) {
  if (ids64) {
    return launch_with_position<T, int64_t>(out, ids, positions, token_weight, pos_weight,
                                            vocab_start, vocab_end, position_offset,
                                            is_first_tp_rank, stream);
  }
  return launch_with_position<T, int32_t>(out, ids, positions, token_weight, pos_weight,
                                          vocab_start, vocab_end, position_offset, is_first_tp_rank,
                                          stream);
}

template <typename T>
Status dispatch_gather(const RowView& out, const RowView& input, const Tensor& rows,
                       cudaStream_t stream) {
  constexpr int kVec = bundle<T>();
  auto* y = static_cast<T*>(const_cast<void*>(out.data));
  const auto* x = static_cast<const T*>(input.data);
  const auto* r =
      reinterpret_cast<const int32_t*>(static_cast<const char*>(rows.data) + rows.byte_offset);
  const int n = static_cast<int>(out.rows);
  gather_rows_kernel<T, kVec><<<static_cast<unsigned>(n), block_for(out.hidden, kVec), 0, stream>>>(
      y, out.stride, x, input.stride, r, n, static_cast<int>(out.hidden));
  return check_launch("gather_rows");
}

template <typename T>
Status dispatch_scatter(const RowView& out, const RowView& src, const Tensor& dst_rows,
                        cudaStream_t stream) {
  constexpr int kVec = bundle<T>();
  auto* y = static_cast<T*>(const_cast<void*>(out.data));
  const auto* s = static_cast<const T*>(src.data);
  const auto* r = reinterpret_cast<const int32_t*>(
      static_cast<const char*>(dst_rows.data) + dst_rows.byte_offset);
  const int n = static_cast<int>(src.rows);
  scatter_rows_kernel<T, kVec><<<static_cast<unsigned>(n), block_for(src.hidden, kVec), 0, stream>>>(
      y, out.stride, s, src.stride, r, n, static_cast<int>(out.hidden));
  return check_launch("scatter_rows");
}

template <typename in_t>
Status dispatch_logits(const RowView& out, const RowView& logits, float scale,
                       float final_softcap, int64_t vocab_start, int64_t vocab_size,
                       cudaStream_t stream) {
  auto* y = static_cast<float*>(const_cast<void*>(out.data));
  const auto* x = static_cast<const in_t*>(logits.data);
  const int64_t col_blocks =
      (out.hidden + static_cast<int64_t>(kLogitsBlock) * kLogitsVec - 1) /
      (kLogitsBlock * kLogitsVec);
  const int grid_x = static_cast<int>(col_blocks < 1 ? 1 : (col_blocks < 4096 ? col_blocks : 4096));
  const int64_t row_blocks = out.rows < kMaxGridY ? out.rows : kMaxGridY;
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(row_blocks));
  logits_postprocess_kernel<in_t><<<grid, kLogitsBlock, 0, stream>>>(
      y, out.stride, x, logits.stride, scale, final_softcap, vocab_start, vocab_size,
      static_cast<int>(out.rows), static_cast<int>(out.hidden));
  return check_launch("logits_postprocess");
}

}  // namespace

// -----------------------------------------------------------------------------
// Public host API (csrc/embedding/ops.h)
// -----------------------------------------------------------------------------

Status embedding_lookup(const Tensor& out, const Tensor& token_ids, const Tensor& weight,
                        int64_t vocab_start, int64_t vocab_end, float scale,
                        const LaunchContext& ctx) {
  RowView vo{}, vw{};
  Status s = make_row_view(out, DType::kInvalid, "embedding_lookup.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(weight, DType::kInvalid, "embedding_lookup.weight", &vw);
  if (s != Status::kSuccess) return s;
  if (vo.dtype != vw.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "embedding_lookup: out %s, weight %s",
                             dtype_name(vo.dtype), dtype_name(vw.dtype));
  }
  if (vo.hidden != vw.hidden) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_lookup: weight width %lld != out width %lld",
                             static_cast<long long>(vw.hidden), static_cast<long long>(vo.hidden));
  }
  const void* ids = nullptr;
  bool ids64 = false;
  s = check_indices(token_ids, vo.rows, "embedding_lookup.token_ids", &ids, &ids64);
  if (s != Status::kSuccess) return s;
  if (vocab_start < 0 || vocab_end < vocab_start || vocab_end - vocab_start > vw.rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_lookup: vocab range [%lld, %lld) does not fit %lld weight rows",
                             static_cast<long long>(vocab_start), static_cast<long long>(vocab_end),
                             static_cast<long long>(vw.rows));
  }
  if (!std::isfinite(scale)) {
    return detail::set_error(Status::kInvalidArgument, "embedding_lookup: scale %g is not finite",
                             static_cast<double>(scale));
  }
  s = check_no_overlap(vo, vw, "embedding_lookup.out", "weight");
  if (s != Status::kSuccess) return s;
  s = check_dim_range(vo, "embedding_lookup.out");
  if (s != Status::kSuccess) return s;
  if (vo.rows == 0 || vo.hidden == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (vo.dtype) {
    case DType::kFloat32:
      return dispatch_lookup<float>(vo, ids, ids64, vw, vocab_start, vocab_end, scale, stream);
    case DType::kFloat16:
      return dispatch_lookup<__half>(vo, ids, ids64, vw, vocab_start, vocab_end, scale, stream);
    case DType::kBFloat16:
      return dispatch_lookup<__nv_bfloat16>(vo, ids, ids64, vw, vocab_start, vocab_end, scale,
                                            stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "embedding_lookup: dtype %s",
                               dtype_name(vo.dtype));
  }
}

Status embedding_with_position(const Tensor& out, const Tensor& token_ids, const Tensor& positions,
                               const Tensor& token_weight, const Tensor& pos_weight,
                               int64_t vocab_start, int64_t vocab_end, int32_t position_offset,
                               bool is_first_tp_rank, const LaunchContext& ctx) {
  RowView vo{}, vtok{}, vpos{};
  Status s = make_row_view(out, DType::kInvalid, "embedding_with_position.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(token_weight, DType::kInvalid, "embedding_with_position.token_weight", &vtok);
  if (s != Status::kSuccess) return s;
  s = make_row_view(pos_weight, DType::kInvalid, "embedding_with_position.pos_weight", &vpos);
  if (s != Status::kSuccess) return s;
  if (vo.dtype != vtok.dtype || vo.dtype != vpos.dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "embedding_with_position: out %s, token_weight %s, pos_weight %s",
                             dtype_name(vo.dtype), dtype_name(vtok.dtype), dtype_name(vpos.dtype));
  }
  if (vo.hidden != vtok.hidden || vo.hidden != vpos.hidden) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_with_position: widths out %lld, token %lld, pos %lld",
                             static_cast<long long>(vo.hidden),
                             static_cast<long long>(vtok.hidden),
                             static_cast<long long>(vpos.hidden));
  }
  // The device contract has no stride arguments: all three operands are dense
  // [rows, hidden] tables.
  if (vo.stride != vo.hidden || vtok.stride != vtok.hidden || vpos.stride != vpos.hidden) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "embedding_with_position: rows must be dense (stride == hidden)");
  }
  const void* ids = nullptr;
  bool ids64 = false;
  s = check_indices(token_ids, vo.rows, "embedding_with_position.token_ids", &ids, &ids64);
  if (s != Status::kSuccess) return s;
  s = check_tensor(positions, DType::kInt64, 1, 8, "embedding_with_position.positions");
  if (s != Status::kSuccess) return s;
  if (positions.shape[0] != vo.rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_with_position: %lld positions for %lld rows",
                             static_cast<long long>(positions.shape[0]),
                             static_cast<long long>(vo.rows));
  }
  if (vocab_start < 0 || vocab_end < vocab_start || vocab_end - vocab_start > vtok.rows) {
    return detail::set_error(
        Status::kInvalidArgument,
        "embedding_with_position: vocab range [%lld, %lld) does not fit %lld weight rows",
        static_cast<long long>(vocab_start), static_cast<long long>(vocab_end),
        static_cast<long long>(vtok.rows));
  }
  if (position_offset < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_with_position: position_offset %d is negative",
                             position_offset);
  }
  if (is_first_tp_rank && vpos.rows == 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "embedding_with_position: pos_weight is empty but this rank adds "
                             "positions");
  }
  s = check_no_overlap(vo, vtok, "embedding_with_position.out", "token_weight");
  if (s != Status::kSuccess) return s;
  s = check_no_overlap(vo, vpos, "embedding_with_position.out", "pos_weight");
  if (s != Status::kSuccess) return s;
  s = check_dim_range(vo, "embedding_with_position.out");
  if (s != Status::kSuccess) return s;
  if (vo.rows == 0 || vo.hidden == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (vo.dtype) {
    case DType::kFloat32:
      return dispatch_with_position<float>(vo, ids, ids64, positions, vtok, vpos, vocab_start,
                                           vocab_end, position_offset, is_first_tp_rank, stream);
    case DType::kFloat16:
      return dispatch_with_position<__half>(vo, ids, ids64, positions, vtok, vpos, vocab_start,
                                            vocab_end, position_offset, is_first_tp_rank, stream);
    case DType::kBFloat16:
      return dispatch_with_position<__nv_bfloat16>(vo, ids, ids64, positions, vtok, vpos,
                                                   vocab_start, vocab_end, position_offset,
                                                   is_first_tp_rank, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "embedding_with_position: dtype %s",
                               dtype_name(vo.dtype));
  }
}

Status gather_rows(const Tensor& out, const Tensor& input, const Tensor& rows,
                   const LaunchContext& ctx) {
  RowView vo{}, vi{};
  Status s = make_row_view(out, DType::kInvalid, "gather_rows.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(input, DType::kInvalid, "gather_rows.input", &vi);
  if (s != Status::kSuccess) return s;
  if (vo.dtype != vi.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "gather_rows: out %s, input %s",
                             dtype_name(vo.dtype), dtype_name(vi.dtype));
  }
  if (vo.hidden != vi.hidden) {
    return detail::set_error(Status::kInvalidArgument, "gather_rows: width %lld != %lld",
                             static_cast<long long>(vo.hidden),
                             static_cast<long long>(vi.hidden));
  }
  s = check_tensor(rows, DType::kInt32, 1, 4, "gather_rows.rows");
  if (s != Status::kSuccess) return s;
  if (rows.shape[0] != vo.rows) {
    return detail::set_error(Status::kInvalidArgument, "gather_rows: %lld rows indices for %lld rows",
                             static_cast<long long>(rows.shape[0]),
                             static_cast<long long>(vo.rows));
  }
  if (vi.rows > kIntMax) {
    return detail::set_error(Status::kUnsupportedShape,
                             "gather_rows: %lld input rows exceed int32 indices",
                             static_cast<long long>(vi.rows));
  }
  s = check_no_overlap(vo, vi, "gather_rows.out", "input");
  if (s != Status::kSuccess) return s;
  s = check_dim_range(vo, "gather_rows.out");
  if (s != Status::kSuccess) return s;
  if (vo.rows == 0 || vo.hidden == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (vo.dtype) {
    case DType::kFloat32: return dispatch_gather<float>(vo, vi, rows, stream);
    case DType::kFloat16: return dispatch_gather<__half>(vo, vi, rows, stream);
    case DType::kBFloat16: return dispatch_gather<__nv_bfloat16>(vo, vi, rows, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "gather_rows: dtype %s",
                               dtype_name(vo.dtype));
  }
}

Status scatter_rows(const Tensor& out, const Tensor& src, const Tensor& dst_rows,
                    const LaunchContext& ctx) {
  RowView vo{}, vs{};
  Status s = make_row_view(out, DType::kInvalid, "scatter_rows.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(src, DType::kInvalid, "scatter_rows.src", &vs);
  if (s != Status::kSuccess) return s;
  if (vo.dtype != vs.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "scatter_rows: out %s, src %s",
                             dtype_name(vo.dtype), dtype_name(vs.dtype));
  }
  if (vo.hidden != vs.hidden) {
    return detail::set_error(Status::kInvalidArgument, "scatter_rows: width %lld != %lld",
                             static_cast<long long>(vo.hidden),
                             static_cast<long long>(vs.hidden));
  }
  s = check_tensor(dst_rows, DType::kInt32, 1, 4, "scatter_rows.dst_rows");
  if (s != Status::kSuccess) return s;
  if (dst_rows.shape[0] != vs.rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "scatter_rows: %lld destination indices for %lld source rows",
                             static_cast<long long>(dst_rows.shape[0]),
                             static_cast<long long>(vs.rows));
  }
  if (vo.rows > kIntMax) {
    return detail::set_error(Status::kUnsupportedShape,
                             "scatter_rows: %lld output rows exceed int32 indices",
                             static_cast<long long>(vo.rows));
  }
  s = check_no_overlap(vo, vs, "scatter_rows.out", "src");
  if (s != Status::kSuccess) return s;
  s = check_dim_range(vs, "scatter_rows.src");
  if (s != Status::kSuccess) return s;
  if (vs.rows == 0 || vs.hidden == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (vo.dtype) {
    case DType::kFloat32: return dispatch_scatter<float>(vo, vs, dst_rows, stream);
    case DType::kFloat16: return dispatch_scatter<__half>(vo, vs, dst_rows, stream);
    case DType::kBFloat16: return dispatch_scatter<__nv_bfloat16>(vo, vs, dst_rows, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "scatter_rows: dtype %s",
                               dtype_name(vo.dtype));
  }
}

Status logits_postprocess(const Tensor& out, const Tensor& logits, float scale,
                          float final_softcap, int64_t vocab_start, int64_t vocab_size,
                          const LaunchContext& ctx) {
  RowView vo{}, vl{};
  Status s = make_row_view(out, DType::kFloat32, "logits_postprocess.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(logits, DType::kInvalid, "logits_postprocess.logits", &vl);
  if (s != Status::kSuccess) return s;
  if (vo.rows != vl.rows || vo.hidden != vl.hidden) {
    return detail::set_error(Status::kInvalidArgument,
                             "logits_postprocess: out [%lld, %lld] != logits [%lld, %lld]",
                             static_cast<long long>(vo.rows), static_cast<long long>(vo.hidden),
                             static_cast<long long>(vl.rows), static_cast<long long>(vl.hidden));
  }
  if (vl.dtype != DType::kFloat32 && vl.dtype != DType::kFloat16 &&
      vl.dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "logits_postprocess: logits %s",
                             dtype_name(vl.dtype));
  }
  if (vocab_start < 0 || vocab_size < 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "logits_postprocess: vocab_start %lld and vocab_size %lld must be "
                             "nonnegative",
                             static_cast<long long>(vocab_start),
                             static_cast<long long>(vocab_size));
  }
  if (!std::isfinite(scale)) {
    return detail::set_error(Status::kInvalidArgument, "logits_postprocess: scale %g is not finite",
                             static_cast<double>(scale));
  }
  if (!std::isfinite(final_softcap) || final_softcap < 0.0f) {
    return detail::set_error(Status::kInvalidArgument,
                             "logits_postprocess: final_softcap %g must be finite and nonnegative",
                             static_cast<double>(final_softcap));
  }
  s = check_exact_or_disjoint(vo, vl, "logits_postprocess");
  if (s != Status::kSuccess) return s;
  s = check_dim_range(vo, "logits_postprocess.out");
  if (s != Status::kSuccess) return s;
  if (vo.rows == 0 || vo.hidden == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (vl.dtype) {
    case DType::kFloat32:
      return dispatch_logits<float>(vo, vl, scale, final_softcap, vocab_start, vocab_size, stream);
    case DType::kFloat16:
      return dispatch_logits<__half>(vo, vl, scale, final_softcap, vocab_start, vocab_size, stream);
    case DType::kBFloat16:
      return dispatch_logits<__nv_bfloat16>(vo, vl, scale, final_softcap, vocab_start, vocab_size,
                                            stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "logits_postprocess: logits %s",
                               dtype_name(vl.dtype));
  }
}

}  // namespace kops::embed
