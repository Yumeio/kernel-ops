// =============================================================================
// kernel-ops | csrc/attention/merge.cu
// A04/A14 first implementations: merge_state, merge_states and merge_split_kv
// (attention-state algebra on (o, lse) pairs). Deterministic order: state i is
// consumed in increasing i, so the result does not depend on scheduling.
//
// LSE convention (attention_params.cuh): natural log, fp32. A state with
// lse = -inf contributes nothing even when its o is not finite; if every state
// (and the optional sink) is -inf, the output is 0 and lse = -inf.
//
// Geometry: one CTA owns a tile of (row, head) tasks. A task is one head of one
// row; thread (x = d / VEC_SIZE, y = head slot) walks the state list, so the
// merge needs no block-wide reduction. Head-dim traffic uses 16-byte vector
// bundles (VEC_SIZE = 4 fp32 elements). Tasks are independent, and every state
// index is visited in increasing order by the single thread that owns an
// element, so the result is batch-invariant.
//
// The public ops.h entry points are merge_state / merge_states / merge_split_kv.
// Optional descriptors that ops.h does not carry yet (row_mask for cascade
// take-a, total_rows_ptr for padded graph batches) are wired through the
// internal entry points declared at the bottom of this file.
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
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace kops::attn {
namespace {

using dev::from_f;
using dev::to_f;

constexpr int kVec = 4;  // 16 B of fp32 per lane; 8 B stores for 16-bit outputs

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

/// Weight of a state with log-sum-exp `l` relative to the running max `m`.
/// A masked (-inf) state weighs exactly zero, even when its o is not finite.
__device__ __forceinline__ float merge_weight(float l, float m) {
  return (l == neg_inf() || m == neg_inf()) ? 0.0f : __expf(l - m);
}

/// Closed-form log-sum-exp of the merged states; no mass gives -inf.
__device__ __forceinline__ float merge_lse_from(float m, float denom) {
  return denom > 0.0f ? m + __logf(denom) : neg_inf();
}

template <typename T, int VEC>
__device__ __forceinline__ void load_span(const T* __restrict__ p, float (&v)[VEC]) {
  const Vec<T, VEC> x = load_vec<T, VEC>(p);
#pragma unroll
  for (int i = 0; i < VEC; ++i) v[i] = to_f<T>(x.data[i]);
}

template <typename T, int VEC>
__device__ __forceinline__ void store_span(T* __restrict__ p, const float (&v)[VEC]) {
  Vec<T, VEC> x;
#pragma unroll
  for (int i = 0; i < VEC; ++i) from_f<T>(x.data[i], v[i]);
  store_vec<T, VEC>(p, x);
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_state_kernel(DTypeO* __restrict__ o, float* __restrict__ lse,
                                   const DTypeIn* __restrict__ o_a, const float* __restrict__ lse_a,
                                   const DTypeIn* __restrict__ o_b, const float* __restrict__ lse_b,
                                   const uint8_t* __restrict__ row_mask, const int num_rows,
                                   const int num_heads) {
  const int heads_per_cta = static_cast<int>(blockDim.y);
  const int head_groups = (num_heads + heads_per_cta - 1) / heads_per_cta;
  const int task = static_cast<int>(blockIdx.x);
  const int row = task / head_groups;
  if (row >= num_rows) return;
  const int head = (task % head_groups) * heads_per_cta + static_cast<int>(threadIdx.y);
  if (head >= num_heads) return;

  const int64_t idx = static_cast<int64_t>(row) * num_heads + head;
  const int64_t base = idx * HEAD_DIM;
  const int dstride = static_cast<int>(blockDim.x) * VEC_SIZE;
  const bool write_lse = (threadIdx.x == 0);

  // row_mask = 0: b is absent, so a is copied verbatim (no arithmetic at all).
  if (row_mask != nullptr && row_mask[row] == 0) {
    for (int d = static_cast<int>(threadIdx.x) * VEC_SIZE; d < HEAD_DIM; d += dstride) {
      float v[VEC_SIZE];
      load_span<DTypeIn, VEC_SIZE>(o_a + base + d, v);
      store_span<DTypeO, VEC_SIZE>(o + base + d, v);
    }
    if (lse != nullptr && write_lse) lse[idx] = lse_a[idx];
    return;
  }

  const float la = lse_a[idx];
  const float lb = lse_b[idx];
  const float m = fmaxf(la, lb);
  const float wa = merge_weight(la, m);
  const float wb = merge_weight(lb, m);
  const float denom = wa + wb;
  const float inv = denom > 0.0f ? __frcp_rn(denom) : 0.0f;
  for (int d = static_cast<int>(threadIdx.x) * VEC_SIZE; d < HEAD_DIM; d += dstride) {
    float va[VEC_SIZE];
    float vb[VEC_SIZE];
    float v[VEC_SIZE];
    load_span<DTypeIn, VEC_SIZE>(o_a + base + d, va);
    load_span<DTypeIn, VEC_SIZE>(o_b + base + d, vb);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float x = wa != 0.0f ? wa * va[j] : 0.0f;  // masked states may hold NaN
      x = wb != 0.0f ? __fmaf_rn(wb, vb[j], x) : x;
      v[j] = x * inv;
    }
    store_span<DTypeO, VEC_SIZE>(o + base + d, v);
  }
  if (lse != nullptr && write_lse) lse[idx] = merge_lse_from(m, denom);
}

template <typename DTypeIn, typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_states_kernel(DTypeO* __restrict__ o, float* __restrict__ lse,
                                    const DTypeIn* __restrict__ o_states,
                                    const float* __restrict__ lse_states, const int num_rows,
                                    const int num_states, const int num_heads) {
  const int heads_per_cta = static_cast<int>(blockDim.y);
  const int head_groups = (num_heads + heads_per_cta - 1) / heads_per_cta;
  const int task = static_cast<int>(blockIdx.x);
  const int row = task / head_groups;
  if (row >= num_rows) return;
  const int head = (task % head_groups) * heads_per_cta + static_cast<int>(threadIdx.y);
  if (head >= num_heads) return;

  const int64_t idx = static_cast<int64_t>(row) * num_heads + head;
  const int64_t lse_base = static_cast<int64_t>(row) * num_states * num_heads + head;
  const int64_t head_stride = static_cast<int64_t>(num_heads) * HEAD_DIM;

  float m = neg_inf();
  for (int s = 0; s < num_states; ++s) {
    m = fmaxf(m, lse_states[lse_base + static_cast<int64_t>(s) * num_heads]);
  }
  float denom = 0.0f;
  for (int s = 0; s < num_states; ++s) {
    denom += merge_weight(lse_states[lse_base + static_cast<int64_t>(s) * num_heads], m);
  }
  const float inv = denom > 0.0f ? __frcp_rn(denom) : 0.0f;

  const int64_t base =
      static_cast<int64_t>(row) * num_states * head_stride + static_cast<int64_t>(head) * HEAD_DIM;
  const int dstride = static_cast<int>(blockDim.x) * VEC_SIZE;
  for (int d = static_cast<int>(threadIdx.x) * VEC_SIZE; d < HEAD_DIM; d += dstride) {
    float acc[VEC_SIZE];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) acc[j] = 0.0f;
    for (int s = 0; s < num_states; ++s) {
      const float w = merge_weight(lse_states[lse_base + static_cast<int64_t>(s) * num_heads], m);
      if (w == 0.0f) continue;  // masked state; its o is never read into the sum
      float v[VEC_SIZE];
      load_span<DTypeIn, VEC_SIZE>(o_states + base + static_cast<int64_t>(s) * head_stride + d, v);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) acc[j] = __fmaf_rn(w, v[j], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) acc[j] *= inv;
    store_span<DTypeO, VEC_SIZE>(o + idx * HEAD_DIM + d, acc);
  }
  if (lse != nullptr && threadIdx.x == 0) {
    lse[idx] = merge_lse_from(m, denom);
  }
}

template <typename DTypeO, int HEAD_DIM, int VEC_SIZE, int HEADS_PER_CTA>
__global__ void merge_split_kv_kernel(
    DTypeO* __restrict__ o, const int64_t o_stride_n, const int64_t o_stride_h,
    float* __restrict__ lse, const float* __restrict__ o_partial,
    const float* __restrict__ lse_partial, const int32_t* __restrict__ merge_indptr,
    const float* __restrict__ sink_logits, const uint32_t* __restrict__ total_rows_ptr,
    const int total_rows, const int num_heads) {
  int rows = total_rows;
  if (total_rows_ptr != nullptr && *total_rows_ptr < static_cast<uint32_t>(rows)) {
    rows = static_cast<int>(*total_rows_ptr);
  }
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;

  constexpr int kLanes = HEAD_DIM / VEC_SIZE;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int head =
      static_cast<int>(blockIdx.y) * HEADS_PER_CTA + static_cast<int>(threadIdx.x) / kLanes;
  if (head >= num_heads) return;
  const int d = lane * VEC_SIZE;

  // Row r owns partial rows [merge_indptr[r], merge_indptr[r + 1]). Negative or
  // inverted bounds are normalized to an empty range; the caller guarantees the
  // upper bound stays inside o_partial / lse_partial (the kernel signature
  // carries no capacity).
  int32_t first = merge_indptr[row];
  int32_t last = merge_indptr[row + 1];
  if (first < 0) first = 0;
  if (last < first) last = first;

  const float sink = sink_logits != nullptr ? sink_logits[head] : neg_inf();
  float m = sink;
  for (int32_t s = first; s < last; ++s) {
    m = fmaxf(m, lse_partial[static_cast<int64_t>(s) * num_heads + head]);
  }
  float denom = sink_logits != nullptr ? merge_weight(sink, m) : 0.0f;
  for (int32_t s = first; s < last; ++s) {
    denom += merge_weight(lse_partial[static_cast<int64_t>(s) * num_heads + head], m);
  }
  const float inv = denom > 0.0f ? __frcp_rn(denom) : 0.0f;

  float acc[VEC_SIZE];
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) acc[j] = 0.0f;
  for (int32_t s = first; s < last; ++s) {
    const float w = merge_weight(lse_partial[static_cast<int64_t>(s) * num_heads + head], m);
    if (w == 0.0f) continue;
    float v[VEC_SIZE];
    load_span<float, VEC_SIZE>(
        o_partial + (static_cast<int64_t>(s) * num_heads + head) * HEAD_DIM + d, v);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) acc[j] = __fmaf_rn(w, v[j], acc[j]);
  }
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) acc[j] *= inv;
  store_span<DTypeO, VEC_SIZE>(
      o + static_cast<int64_t>(row) * o_stride_n + static_cast<int64_t>(head) * o_stride_h + d, acc);

  // The sink is folded into the denominator here, exactly once; lane 0 of each
  // head owns the lse slot.
  if (lse != nullptr && lane == 0) {
    lse[static_cast<int64_t>(row) * num_heads + head] = merge_lse_from(m, denom);
  }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
namespace {

template <typename T>
inline const T* typed_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

template <typename T>
inline T* typed_mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

/// Threads per head for the 128-thread tile used by merge_state / merge_states.
inline int heads_per_cta_for(int head_dim) {
  const int lanes = head_dim / kVec;
  int hpb = 128 / (lanes > 0 ? lanes : 1);
  if (hpb < 1) hpb = 1;
  if (hpb > 8) hpb = 8;
  return hpb;
}

inline bool supported_out_dtype(DType t) {
  return t == DType::kBFloat16 || t == DType::kFloat16 || t == DType::kFloat32;
}

/// Packed [rows, heads, head_dim]: the layout the merge kernels index directly.
bool packed_3d(const Tensor& t, int64_t heads, int64_t head_dim) {
  return t.ndim == 3 && t.stride[2] == 1 && t.stride[1] == head_dim &&
         t.stride[0] == heads * head_dim;
}

/// Packed [rows, states, heads, head_dim].
bool packed_4d(const Tensor& t, int64_t states, int64_t heads, int64_t head_dim) {
  return t.ndim == 4 && t.stride[3] == 1 && t.stride[2] == head_dim &&
         t.stride[1] == heads * head_dim && t.stride[0] == states * heads * head_dim;
}

/// Packed [rows, states, heads].
bool packed_lse_states(const Tensor& t, int64_t states, int64_t heads) {
  return t.ndim == 3 && t.stride[2] == 1 && t.stride[1] == heads &&
         t.stride[0] == states * heads;
}

Status check_lse_2d(const Tensor& t, int rows, int heads, const char* name) {
  Status s = check_tensor(t, DType::kFloat32, 2, 4, name);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != rows || t.shape[1] != heads) {
    return detail::set_error(Status::kInvalidArgument, "%s: shape [%lld, %lld], expected [%d, %d]",
                             name, static_cast<long long>(t.shape[0]),
                             static_cast<long long>(t.shape[1]), rows, heads);
  }
  if (t.stride[1] != 1 || t.stride[0] != heads) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: must be packed [rows, heads]", name);
  }
  return Status::kSuccess;
}

Status check_range(int64_t rows, int64_t heads, int64_t head_dim) {
  if (rows > INT32_MAX || heads > INT32_MAX || head_dim > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "merge: dimensions exceed int32");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

// ---- merge_state -------------------------------------------------------------

template <typename Out, int HEAD>
Status run_merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                       const Tensor& o_b, const Tensor& lse_b, const Tensor& row_mask, int rows,
                       int heads, cudaStream_t stream) {
  int hpb = heads_per_cta_for(HEAD);
  if (hpb > heads) hpb = heads;
  const int groups = (heads + hpb - 1) / hpb;
  const int64_t blocks = static_cast<int64_t>(rows) * groups;
  if (blocks > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "merge_state: %lld CTAs", blocks);
  }
  merge_state_kernel<float, Out, HEAD, kVec><<<static_cast<int>(blocks), dim3(HEAD / kVec, hpb), 0,
                                                stream>>>(
      typed_mut_ptr<Out>(o), tensor_present(lse) ? typed_mut_ptr<float>(lse) : nullptr,
      typed_ptr<float>(o_a), typed_ptr<float>(lse_a), typed_ptr<float>(o_b),
      typed_ptr<float>(lse_b), tensor_present(row_mask) ? typed_ptr<uint8_t>(row_mask) : nullptr,
      rows, heads);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("merge_state launch", err);
  return Status::kSuccess;
}

template <typename Out>
Status dispatch_merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a,
                            const Tensor& lse_a, const Tensor& o_b, const Tensor& lse_b,
                            const Tensor& row_mask, int rows, int heads, int head_dim,
                            cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return run_merge_state<Out, 64>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows, heads, stream);
    case 128:
      return run_merge_state<Out, 128>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows, heads, stream);
    case 256:
      return run_merge_state<Out, 256>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows, heads, stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "merge_state: head_dim %d", head_dim);
  }
}

// ---- merge_states ------------------------------------------------------------

template <typename Out, int HEAD>
Status run_merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                        const Tensor& lse_states, int rows, int num_states, int heads,
                        cudaStream_t stream) {
  int hpb = heads_per_cta_for(HEAD);
  if (hpb > heads) hpb = heads;
  const int groups = (heads + hpb - 1) / hpb;
  const int64_t blocks = static_cast<int64_t>(rows) * groups;
  if (blocks > INT32_MAX) {
    return detail::set_error(Status::kCapacityExceeded, "merge_states: %lld CTAs", blocks);
  }
  merge_states_kernel<float, Out, HEAD, kVec><<<static_cast<int>(blocks), dim3(HEAD / kVec, hpb), 0,
                                                 stream>>>(
      typed_mut_ptr<Out>(o), tensor_present(lse) ? typed_mut_ptr<float>(lse) : nullptr,
      typed_ptr<float>(o_states), typed_ptr<float>(lse_states), rows, num_states, heads);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("merge_states launch", err);
  return Status::kSuccess;
}

template <typename Out>
Status dispatch_merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                             const Tensor& lse_states, int rows, int num_states, int heads,
                             int head_dim, cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return run_merge_states<Out, 64>(o, lse, o_states, lse_states, rows, num_states, heads,
                                       stream);
    case 128:
      return run_merge_states<Out, 128>(o, lse, o_states, lse_states, rows, num_states, heads,
                                        stream);
    case 256:
      return run_merge_states<Out, 256>(o, lse, o_states, lse_states, rows, num_states, heads,
                                        stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "merge_states: head_dim %d", head_dim);
  }
}

// ---- merge_split_kv ----------------------------------------------------------

template <typename Out, int HEAD, int HEADS_PER_CTA>
Status run_merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                          const Tensor& sink_logits, const Tensor& total_rows, int rows,
                          int heads, cudaStream_t stream) {
  if (heads % HEADS_PER_CTA != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "merge_split_kv: %d heads is not a multiple of %d", heads,
                             HEADS_PER_CTA);
  }
  const int head_groups = heads / HEADS_PER_CTA;
  if (head_groups > 65535) {
    return detail::set_error(Status::kCapacityExceeded, "merge_split_kv: %d head groups",
                             head_groups);
  }
  const int block = (HEAD / kVec) * HEADS_PER_CTA;
  const dim3 grid(rows, head_groups);
  merge_split_kv_kernel<Out, HEAD, kVec, HEADS_PER_CTA><<<grid, block, 0, stream>>>(
      typed_mut_ptr<Out>(o), o.stride[0], o.stride[1],
      tensor_present(lse) ? typed_mut_ptr<float>(lse) : nullptr, split.o_partial,
      split.lse_partial, split.merge_indptr,
      tensor_present(sink_logits) ? typed_ptr<float>(sink_logits) : nullptr,
      tensor_present(total_rows) ? typed_ptr<uint32_t>(total_rows) : nullptr, rows, heads);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("merge_split_kv launch", err);
  return Status::kSuccess;
}

template <typename Out, int HEAD>
Status dispatch_merge_split_kv_hpb(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                                   const Tensor& sink_logits, const Tensor& total_rows, int rows,
                                   int heads, cudaStream_t stream) {
  if (heads % 4 == 0) {
    return run_merge_split_kv<Out, HEAD, 4>(o, lse, split, sink_logits, total_rows, rows, heads,
                                            stream);
  }
  if (heads % 2 == 0) {
    return run_merge_split_kv<Out, HEAD, 2>(o, lse, split, sink_logits, total_rows, rows, heads,
                                            stream);
  }
  return run_merge_split_kv<Out, HEAD, 1>(o, lse, split, sink_logits, total_rows, rows, heads,
                                          stream);
}

template <typename Out>
Status dispatch_merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                               const Tensor& sink_logits, const Tensor& total_rows, int rows,
                               int heads, int head_dim, cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return dispatch_merge_split_kv_hpb<Out, 64>(o, lse, split, sink_logits, total_rows, rows,
                                                  heads, stream);
    case 80:  // B8 attention head dims (Phi-2 / StableLM, Phi-3, MLA 192/128 prefill rows)
      return dispatch_merge_split_kv_hpb<Out, 80>(o, lse, split, sink_logits, total_rows, rows,
                                                  heads, stream);
    case 96:
      return dispatch_merge_split_kv_hpb<Out, 96>(o, lse, split, sink_logits, total_rows, rows,
                                                  heads, stream);
    case 192:
      return dispatch_merge_split_kv_hpb<Out, 192>(o, lse, split, sink_logits, total_rows, rows,
                                                   heads, stream);
    case 128:
      return dispatch_merge_split_kv_hpb<Out, 128>(o, lse, split, sink_logits, total_rows, rows,
                                                   heads, stream);
    case 256:
      return dispatch_merge_split_kv_hpb<Out, 256>(o, lse, split, sink_logits, total_rows, rows,
                                                   heads, stream);
    case 512:
      return dispatch_merge_split_kv_hpb<Out, 512>(o, lse, split, sink_logits, total_rows, rows,
                                                   heads, stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "merge_split_kv: head_dim %d", head_dim);
  }
}

// ---- merge_hybrid_branches ---------------------------------------------------

template <typename DTypeO, int HEAD_DIM, int VEC_SIZE>
__global__ void merge_hybrid_branches_kernel(DTypeO* __restrict__ o, const int64_t o_stride_n,
                                             const int64_t o_stride_h, float* __restrict__ lse,
                                             const int64_t lse_stride,
                                             const HybridBranchTable branches,
                                             const float* __restrict__ sink_logits,
                                             const int num_rows, const int num_heads) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;
  for (int h = 0; h < num_heads; ++h) {
    const int64_t task = static_cast<int64_t>(row) * num_heads + h;
    float m = sink_logits ? sink_logits[h] : neg_inf();
    for (int b = 0; b < branches.count; ++b) {
      if (branches.o[b] != nullptr) m = fmaxf(m, branches.lse[b][task]);
    }
    float denom = sink_logits ? merge_weight(sink_logits[h], m) : 0.0f;
    for (int b = 0; b < branches.count; ++b) {
      if (branches.o[b] != nullptr) denom += merge_weight(branches.lse[b][task], m);
    }
    const float inv = denom > 0.0f ? __frcp_rn(denom) : 0.0f;
    for (int d = threadIdx.x * VEC_SIZE; d < HEAD_DIM; d += blockDim.x * VEC_SIZE) {
      float acc[VEC_SIZE];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) acc[j] = 0.0f;
      for (int b = 0; b < branches.count; ++b) {
        if (branches.o[b] == nullptr) continue;
        const float w = merge_weight(branches.lse[b][task], m);
        float x[VEC_SIZE];
        load_span<float, VEC_SIZE>(branches.o[b] + task * HEAD_DIM + d, x);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) acc[j] = w != 0.0f ? __fmaf_rn(w, x[j], acc[j]) : acc[j];
      }
      float y[VEC_SIZE];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) y[j] = acc[j] * inv;
      store_span<DTypeO, VEC_SIZE>(o + row * o_stride_n + h * o_stride_h + d, y);
    }
    if (lse != nullptr && threadIdx.x == 0) {
      lse[row * lse_stride + h] = denom > 0.0f ? merge_lse_from(m, denom) : neg_inf();
    }
  }
}

template <typename Out, int HEAD>
Status run_merge_hybrid(const Tensor& o, const Tensor& lse, const HybridBranches& branches,
                        const Tensor& sink_logits, int rows, int heads, cudaStream_t stream) {
  const int block = 128;
  HybridBranchTable table{};
  table.count = branches.num_branches;
  for (int b = 0; b < branches.num_branches; ++b) {
    table.o[b] = branches.o[b];
    table.lse[b] = branches.lse[b];
  }
  merge_hybrid_branches_kernel<Out, HEAD, kVec><<<rows, block, 0, stream>>>(
      typed_mut_ptr<Out>(o), o.stride[0], o.stride[1],
      tensor_present(lse) ? typed_mut_ptr<float>(lse) : nullptr,
      tensor_present(lse) ? lse.stride[0] : 0, table,
      tensor_present(sink_logits) ? typed_ptr<float>(sink_logits) : nullptr, rows, heads);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("merge_hybrid_branches launch", err);
  return Status::kSuccess;
}

template <typename Out>
Status dispatch_merge_hybrid(const Tensor& o, const Tensor& lse, const HybridBranches& branches,
                             const Tensor& sink_logits, int rows, int heads, int head_dim,
                             cudaStream_t stream) {
  switch (head_dim) {
    case 64: return run_merge_hybrid<Out, 64>(o, lse, branches, sink_logits, rows, heads, stream);
    case 128: return run_merge_hybrid<Out, 128>(o, lse, branches, sink_logits, rows, heads, stream);
    case 256: return run_merge_hybrid<Out, 256>(o, lse, branches, sink_logits, rows, heads, stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "merge_hybrid_branches: head_dim %d",
                               head_dim);
  }
}

}  // namespace
}  // namespace

// -----------------------------------------------------------------------------
// Internal entry points: optional descriptors not carried by ops.h yet.
// A caller outside this translation unit must use the declarations verbatim.
// -----------------------------------------------------------------------------
namespace internal {

Status merge_state_run(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                       const Tensor& o_b, const Tensor& lse_b, const Tensor& row_mask,
                       const LaunchContext& ctx) {
  Status s = check_tensor(o, o.dtype, 3, 16, "merge_state.o");
  if (s != Status::kSuccess) return s;
  if (!supported_out_dtype(o.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_state: out dtype %s",
                             dtype_name(o.dtype));
  }
  s = check_tensor(o_a, o_a.dtype, 3, 16, "merge_state.o_a");
  if (s != Status::kSuccess) return s;
  s = check_tensor(o_b, o_b.dtype, 3, 16, "merge_state.o_b");
  if (s != Status::kSuccess) return s;
  if (o_a.dtype != o_b.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_state: o_a %s vs o_b %s",
                             dtype_name(o_a.dtype), dtype_name(o_b.dtype));
  }
  for (int d = 0; d < 3; ++d) {
    if (o.shape[d] != o_a.shape[d] || o.shape[d] != o_b.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "merge_state: shape mismatch at dim %d", d);
    }
  }
  if (o_a.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "merge_state: partial states must be fp32, got %s",
                             dtype_name(o_a.dtype));
  }
  if (!packed_3d(o, o.shape[1], o.shape[2]) || !packed_3d(o_a, o.shape[1], o.shape[2]) ||
      !packed_3d(o_b, o.shape[1], o.shape[2])) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "merge_state: o, o_a and o_b must be packed [rows, heads, dim]");
  }
  s = check_range(o.shape[0], o.shape[1], o.shape[2]);
  if (s != Status::kSuccess) return s;
  const int rows = static_cast<int>(o.shape[0]);
  const int heads = static_cast<int>(o.shape[1]);
  const int head_dim = static_cast<int>(o.shape[2]);
  s = check_lse_2d(lse_a, rows, heads, "merge_state.lse_a");
  if (s != Status::kSuccess) return s;
  s = check_lse_2d(lse_b, rows, heads, "merge_state.lse_b");
  if (s != Status::kSuccess) return s;
  if (tensor_present(lse)) {
    s = check_lse_2d(lse, rows, heads, "merge_state.lse");
    if (s != Status::kSuccess) return s;
  }
  if (tensor_present(row_mask)) {
    s = check_tensor(row_mask, DType::kUInt8, 1, 1, "merge_state.row_mask");
    if (s != Status::kSuccess) return s;
    if (row_mask.shape[0] != rows || row_mask.stride[0] != 1) {
      return detail::set_error(Status::kInvalidArgument, "merge_state: row_mask must be [%d]", rows);
    }
  }
  if (rows == 0 || heads == 0) return Status::kSuccess;
  if (head_dim <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "merge_state: head_dim %d", head_dim);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (o.dtype) {
    case DType::kBFloat16:
      return dispatch_merge_state<__nv_bfloat16>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows,
                                                 heads, head_dim, stream);
    case DType::kFloat16:
      return dispatch_merge_state<__half>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows, heads,
                                          head_dim, stream);
    case DType::kFloat32:
      return dispatch_merge_state<float>(o, lse, o_a, lse_a, o_b, lse_b, row_mask, rows, heads,
                                         head_dim, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "merge_state: out dtype %s",
                               dtype_name(o.dtype));
  }
}

Status merge_split_kv_run(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                          const Tensor& sink_logits, const Tensor& total_rows,
                          const LaunchContext& ctx) {
  Status s = check_tensor(o, o.dtype, 3, 16, "merge_split_kv.o");
  if (s != Status::kSuccess) return s;
  if (!supported_out_dtype(o.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_split_kv: out dtype %s",
                             dtype_name(o.dtype));
  }
  if (o.stride[2] != 1 || o.stride[0] < 0 || o.stride[1] < 0) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "merge_split_kv: o must have unit last stride and nonnegative strides");
  }
  s = check_range(o.shape[0], o.shape[1], o.shape[2]);
  if (s != Status::kSuccess) return s;
  const int rows = static_cast<int>(o.shape[0]);
  const int heads = static_cast<int>(o.shape[1]);
  const int head_dim = static_cast<int>(o.shape[2]);
  if (split.merge_indptr == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "merge_split_kv: merge_indptr is null");
  }
  if (split.o_partial == nullptr || split.lse_partial == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "merge_split_kv: o_partial and lse_partial are required");
  }
  if (split.max_splits < 1) {
    return detail::set_error(Status::kInvalidArgument, "merge_split_kv: max_splits %d < 1",
                             split.max_splits);
  }
  if (reinterpret_cast<uintptr_t>(split.o_partial) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "merge_split_kv: o_partial violates 16-byte alignment");
  }
  if (reinterpret_cast<uintptr_t>(split.lse_partial) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(split.merge_indptr) % 4 != 0) {
    return detail::set_error(Status::kMisaligned,
                             "merge_split_kv: lse_partial / merge_indptr violate 4-byte alignment");
  }
  if (heads <= 0 || head_dim <= 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "merge_split_kv: heads %d, head_dim %d", heads, head_dim);
  }
  if (tensor_present(lse)) {
    s = check_lse_2d(lse, rows, heads, "merge_split_kv.lse");
    if (s != Status::kSuccess) return s;
  }
  if (tensor_present(sink_logits)) {
    s = check_tensor(sink_logits, DType::kFloat32, 1, 4, "merge_split_kv.sink_logits");
    if (s != Status::kSuccess) return s;
    if (sink_logits.shape[0] != heads || sink_logits.stride[0] != 1) {
      return detail::set_error(Status::kInvalidArgument,
                               "merge_split_kv: sink_logits must be [%d]", heads);
    }
  }
  if (tensor_present(total_rows)) {
    s = check_tensor(total_rows, DType::kUInt32, 1, 4, "merge_split_kv.total_rows");
    if (s != Status::kSuccess) return s;
    if (total_rows.shape[0] != 1 || total_rows.stride[0] != 1) {
      return detail::set_error(Status::kInvalidArgument,
                               "merge_split_kv: total_rows must be a uint32 [1]");
    }
  }
  if (rows == 0) return Status::kSuccess;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (o.dtype) {
    case DType::kBFloat16:
      return dispatch_merge_split_kv<__nv_bfloat16>(o, lse, split, sink_logits, total_rows, rows,
                                                    heads, head_dim, stream);
    case DType::kFloat16:
      return dispatch_merge_split_kv<__half>(o, lse, split, sink_logits, total_rows, rows, heads,
                                             head_dim, stream);
    case DType::kFloat32:
      return dispatch_merge_split_kv<float>(o, lse, split, sink_logits, total_rows, rows, heads,
                                            head_dim, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "merge_split_kv: out dtype %s",
                               dtype_name(o.dtype));
  }
}

}  // namespace internal

// -----------------------------------------------------------------------------
// Public API (csrc/attention/ops.h)
// -----------------------------------------------------------------------------

Status merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                   const Tensor& o_b, const Tensor& lse_b, const LaunchContext& ctx) {
  return internal::merge_state_run(o, lse, o_a, lse_a, o_b, lse_b, Tensor{}, ctx);
}

Status merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                    const Tensor& lse_states, const LaunchContext& ctx) {
  Status s = check_tensor(o, o.dtype, 3, 16, "merge_states.o");
  if (s != Status::kSuccess) return s;
  if (!supported_out_dtype(o.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_states: out dtype %s",
                             dtype_name(o.dtype));
  }
  s = check_tensor(o_states, DType::kFloat32, 4, 16, "merge_states.o_states");
  if (s != Status::kSuccess) return s;
  s = check_tensor(lse_states, DType::kFloat32, 3, 4, "merge_states.lse_states");
  if (s != Status::kSuccess) return s;
  const int64_t rows64 = o.shape[0];
  const int64_t heads64 = o.shape[1];
  const int64_t head_dim64 = o.shape[2];
  const int64_t states64 = o_states.shape[1];
  if (o_states.shape[0] != rows64 || o_states.shape[2] != heads64 ||
      o_states.shape[3] != head_dim64) {
    return detail::set_error(Status::kInvalidArgument,
                             "merge_states: o_states must be [rows, N, heads, head_dim]");
  }
  if (lse_states.shape[0] != rows64 || lse_states.shape[1] != states64 ||
      lse_states.shape[2] != heads64) {
    return detail::set_error(Status::kInvalidArgument,
                             "merge_states: lse_states must be [rows, N, heads]");
  }
  if (!packed_3d(o, heads64, head_dim64) || !packed_4d(o_states, states64, heads64, head_dim64) ||
      !packed_lse_states(lse_states, states64, heads64)) {
    return detail::set_error(
        Status::kUnsupportedLayout,
        "merge_states: o, o_states and lse_states must be packed [rows, N, heads, dim]");
  }
  s = check_range(rows64, heads64, head_dim64);
  if (s != Status::kSuccess) return s;
  s = check_range(states64, 1, 1);
  if (s != Status::kSuccess) return s;
  const int rows = static_cast<int>(rows64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const int num_states = static_cast<int>(states64);
  if (num_states < 1) {
    return detail::set_error(Status::kInvalidArgument, "merge_states: num_states %d < 1",
                             num_states);
  }
  if (tensor_present(lse)) {
    s = check_lse_2d(lse, rows, heads, "merge_states.lse");
    if (s != Status::kSuccess) return s;
  }
  if (rows == 0 || heads == 0) return Status::kSuccess;
  if (head_dim <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "merge_states: head_dim %d", head_dim);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (o.dtype) {
    case DType::kBFloat16:
      return dispatch_merge_states<__nv_bfloat16>(o, lse, o_states, lse_states, rows, num_states,
                                                  heads, head_dim, stream);
    case DType::kFloat16:
      return dispatch_merge_states<__half>(o, lse, o_states, lse_states, rows, num_states, heads,
                                           head_dim, stream);
    case DType::kFloat32:
      return dispatch_merge_states<float>(o, lse, o_states, lse_states, rows, num_states, heads,
                                          head_dim, stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "merge_states: out dtype %s",
                               dtype_name(o.dtype));
  }
}

Status merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                      const Tensor& sink_logits, const LaunchContext& ctx) {
  return internal::merge_split_kv_run(o, lse, split, sink_logits, Tensor{}, ctx);
}

Status merge_hybrid_branches(const Tensor& o, const Tensor& lse, const HybridBranches& branches,
                             const Tensor& sink_logits, const LaunchContext& ctx) {
  Status s = check_tensor(o, o.dtype, 3, 16, "merge_hybrid_branches.o");
  if (s != Status::kSuccess) return s;
  if (!supported_out_dtype(o.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "merge_hybrid_branches: out dtype %s",
                             dtype_name(o.dtype));
  }
  const int64_t rows64 = o.shape[0];
  const int64_t heads64 = o.shape[1];
  const int64_t head_dim64 = o.shape[2];
  if (rows64 < 0 || heads64 <= 0 || head_dim64 <= 0 || rows64 > INT32_MAX ||
      heads64 > INT32_MAX || head_dim64 > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape,
                             "merge_hybrid_branches: invalid shape [%lld, %lld, %lld]",
                             static_cast<long long>(rows64), static_cast<long long>(heads64),
                             static_cast<long long>(head_dim64));
  }
  if (!packed_3d(o, heads64, head_dim64)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "merge_hybrid_branches: o must be packed [rows, heads, dim]");
  }
  if (tensor_present(lse)) {
    s = check_lse_2d(lse, static_cast<int>(rows64), static_cast<int>(heads64),
                     "merge_hybrid_branches.lse");
    if (s != Status::kSuccess) return s;
  }
  if (tensor_present(sink_logits)) {
    s = check_tensor(sink_logits, DType::kFloat32, 1, 4, "merge_hybrid_branches.sink_logits");
    if (s != Status::kSuccess) return s;
    if (sink_logits.shape[0] != heads64) {
      return detail::set_error(Status::kInvalidArgument,
                               "merge_hybrid_branches: sink_logits must be [%lld]",
                               static_cast<long long>(heads64));
    }
  }
  if (branches.o == nullptr || branches.lse == nullptr || branches.num_branches <= 0 ||
      branches.num_branches > kMaxHybridBranches) {
    return detail::set_error(Status::kInvalidArgument,
                             "merge_hybrid_branches: branch count %d not in [1, %d]",
                             branches.num_branches, kMaxHybridBranches);
  }
  for (int b = 0; b < branches.num_branches; ++b) {
    if (branches.o[b] != nullptr &&
        (reinterpret_cast<uintptr_t>(branches.o[b]) & 15u) != 0) {
      return detail::set_error(Status::kMisaligned,
                               "merge_hybrid_branches: branch %d o pointer is not 16-byte aligned",
                               b);
    }
    if (branches.o[b] != nullptr && branches.lse[b] == nullptr) {
      return detail::set_error(Status::kInvalidArgument,
                               "merge_hybrid_branches: branch %d has o but no lse", b);
    }
  }
  if (rows64 == 0) return Status::kSuccess;
  const int rows = static_cast<int>(rows64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (o.dtype) {
    case DType::kBFloat16:
      return dispatch_merge_hybrid<__nv_bfloat16>(o, lse, branches, sink_logits, rows, heads,
                                                  head_dim, stream);
    case DType::kFloat16:
      return dispatch_merge_hybrid<__half>(o, lse, branches, sink_logits, rows, heads, head_dim,
                                           stream);
    case DType::kFloat32:
      return dispatch_merge_hybrid<float>(o, lse, branches, sink_logits, rows, heads, head_dim,
                                          stream);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "merge_hybrid_branches: out dtype %s",
                               dtype_name(o.dtype));
  }
}

}  // namespace kops::attn
