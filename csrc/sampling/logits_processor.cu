// =============================================================================
// kernel-ops | csrc/sampling/logits_processor.cu
// B1 first implementation: in-place logit processors (S02).
//   token_bincount        counts[b, v] and prompt_mask bits for a ragged request
//   apply_penalties       repetition / frequency / presence, in place
//   apply_token_bitmask   packed int32 allow-mask (xgrammar layout), -inf
//   apply_logit_bias      sparse additive bias, one thread per row (sequential)
//
// Shape/stride contract (the kernel signatures carry no strides for the
// auxiliary arrays, so the launchers require them row-contiguous):
//   counts        int32  [B, V]                row stride V
//   prompt_mask   uint32 [B, ceil(V / 32)]     row stride ceil(V / 32)
//   output_counts int32  [B, V]                row stride V
//   bitmask       int32  [M, ceil(V / 32)]     row stride ceil(V / 32)
//   logits        fp16/bf16/fp32 [R, V]        any row stride >= V
//
// The engine owns the token history: token_indptr values must be
// nondecreasing and inside [0, num_ids], and rows/ids carried inside device
// tensors are assumed in range. The ABI passes neither num_ids nor nnz nor
// vocab_size to the counting/bias kernels, so those bounds cannot be checked
// without a device read, and the launchers do not synchronize.
//
// Penalty formula (HF/OpenAI semantics, fp32 math, one rounding at the store):
//   repetition (seen in the prompt OR in the output):
//       l = l > 0 ? l / r : l * r
//   frequency (output count c):  l -= f * c
//   presence  (c > 0):           l -= p
// The three are evaluated in that order, so a 16-bit logit is rounded once.
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
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/logits_processor_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"

namespace kops::sample {
namespace {

constexpr int kThreads = 256;
constexpr int kBincountSplits = 8;  // CTAs that split the ids of one request
constexpr int kElemVec = 4;         // logical elements per thread of one tile step
constexpr int kMaxGridY = 65535;

int row_grid(int64_t rows) {
  return static_cast<int>(rows > kMaxGridY ? kMaxGridY : rows);
}

int column_grid(int vocab) {
  const int64_t per_block = static_cast<int64_t>(kThreads) * kElemVec;
  const int64_t tiles = (static_cast<int64_t>(vocab) + per_block - 1) / per_block;
  return static_cast<int>(tiles < 1 ? 1 : (tiles > 1024 ? 1024 : tiles));
}

}  // namespace

// -----------------------------------------------------------------------------
// S02 kernels
// -----------------------------------------------------------------------------

__global__ void token_bincount_kernel(int32_t* __restrict__ counts,
                                      uint32_t* __restrict__ prompt_mask,
                                      const int32_t* __restrict__ token_ids,
                                      const int32_t* __restrict__ token_indptr,
                                      const uint8_t* __restrict__ is_prompt, const int batch_size,
                                      const int vocab_size) {
  const int words = (vocab_size + 31) / 32;
  for (int row = blockIdx.y; row < batch_size; row += gridDim.y) {
    int start = token_indptr[row];
    int end = token_indptr[row + 1];
    if (start < 0) start = 0;   // a negative indptr entry carries no token
    if (end < start) end = start;
    for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end;
         i += gridDim.x * blockDim.x) {
      const int32_t id = token_ids[i];
      if (id < 0 || id >= vocab_size) continue;  // padded / out-of-shard ids
      if (is_prompt != nullptr && is_prompt[i] != 0) {
        if (prompt_mask != nullptr) {
          atomicOr(reinterpret_cast<unsigned int*>(
                       &prompt_mask[static_cast<int64_t>(row) * words + (id >> 5)]),
                   1u << (id & 31));
        }
      } else {
        atomicAdd(&counts[static_cast<int64_t>(row) * vocab_size + id], 1);
      }
    }
  }
}

template <typename logits_t, int VEC_SIZE>
__global__ void apply_penalties_kernel(logits_t* __restrict__ logits,
                                       const int64_t logits_stride,
                                       const int32_t* __restrict__ output_counts,
                                       const uint32_t* __restrict__ prompt_mask,
                                       const float* __restrict__ repetition,
                                       const float* __restrict__ frequency,
                                       const float* __restrict__ presence, const int batch_size,
                                       const int vocab_size) {
  const int words = (vocab_size + 31) / 32;
  for (int row = blockIdx.y; row < batch_size; row += gridDim.y) {
    const float rep = repetition != nullptr ? repetition[row] : 1.0f;
    const float freq = frequency != nullptr ? frequency[row] : 0.0f;
    const float pres = presence != nullptr ? presence[row] : 0.0f;
    logits_t* __restrict__ lr = logits + static_cast<int64_t>(row) * logits_stride;
    const int32_t* __restrict__ counts =
        output_counts + static_cast<int64_t>(row) * vocab_size;
    const uint32_t* __restrict__ mask =
        prompt_mask != nullptr ? prompt_mask + static_cast<int64_t>(row) * words : nullptr;
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x * VEC_SIZE;
         base < vocab_size; base += static_cast<int64_t>(gridDim.x) * blockDim.x * VEC_SIZE) {
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const int64_t v = base + static_cast<int64_t>(j) * blockDim.x + threadIdx.x;
        if (v >= vocab_size) break;
        const int32_t count = counts[v];
        const bool seen = count > 0 ||
                          (mask != nullptr && ((mask[v >> 5] >> (v & 31)) & 1u) != 0u);
        float l = dev::to_f<logits_t>(lr[v]);
        if (seen) l = l > 0.0f ? l / rep : l * rep;
        if (count > 0) {
          l = __fsub_rn(l, __fmul_rn(freq, static_cast<float>(count)));
          l = __fsub_rn(l, pres);
        }
        dev::from_f<logits_t>(lr[v], l);
      }
    }
  }
}

template <typename logits_t, int VEC_SIZE>
__global__ void apply_token_bitmask_kernel(logits_t* __restrict__ logits,
                                           const int64_t logits_stride,
                                           const int32_t* __restrict__ bitmask,
                                           const int32_t* __restrict__ rows,
                                           const int num_mask_rows, const int vocab_size) {
  const int words = (vocab_size + 31) / 32;
  for (int m = blockIdx.y; m < num_mask_rows; m += gridDim.y) {
    const int target = rows != nullptr ? rows[m] : m;
    if (target < 0) continue;
    logits_t* __restrict__ lr = logits + static_cast<int64_t>(target) * logits_stride;
    const int32_t* __restrict__ mr = bitmask + static_cast<int64_t>(m) * words;
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x * VEC_SIZE;
         base < vocab_size; base += static_cast<int64_t>(gridDim.x) * blockDim.x * VEC_SIZE) {
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const int64_t v = base + static_cast<int64_t>(j) * blockDim.x + threadIdx.x;
        if (v >= vocab_size) break;
        // Bits past vocab_size in the last word are ignored: v < vocab_size.
        if (((static_cast<uint32_t>(mr[v >> 5]) >> (v & 31)) & 1u) == 0u) {
          dev::from_f<logits_t>(lr[v], __int_as_float(0xff800000));
        }
      }
    }
  }
}

template <typename logits_t>
__global__ void apply_logit_bias_kernel(logits_t* __restrict__ logits,
                                        const int64_t logits_stride,
                                        const int32_t* __restrict__ bias_indptr,
                                        const int32_t* __restrict__ bias_token_ids,
                                        const float* __restrict__ bias_values,
                                        const int batch_size) {
  // One thread owns a request and walks its segment in indptr order, so two
  // bias entries that name the same token add deterministically (no race).
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch_size) return;
  int start = bias_indptr[row];
  const int end = bias_indptr[row + 1];
  if (start < 0) start = 0;
  if (end <= start) return;
  logits_t* __restrict__ lr = logits + static_cast<int64_t>(row) * logits_stride;
  for (int i = start; i < end; ++i) {
    const int32_t id = bias_token_ids[i];
    if (id < 0) continue;  // the ABI does not carry vocab_size; negative ids are skipped
    const float updated = __fadd_rn(dev::to_f<logits_t>(lr[id]), bias_values[i]);
    dev::from_f<logits_t>(lr[id], updated);
  }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

namespace {

template <typename T>
const T* as_const(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

template <typename T>
T* as_mutable(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

size_t element_bytes(DType dtype) {
  const int bits = dtype_bits(dtype);
  return bits > 0 ? static_cast<size_t>((bits + 7) / 8) : 1u;
}

bool is_supported_logits(DType dtype) {
  return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16;
}

Status check_logits_matrix(const Tensor& logits, const char* name, int* rows, int* vocab) {
  const Status s = check_tensor(logits, logits.dtype, 2, element_bytes(logits.dtype), name);
  if (s != Status::kSuccess) return s;
  if (!is_supported_logits(logits.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: %s", name, dtype_name(logits.dtype));
  }
  if (logits.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last dim must be contiguous", name);
  }
  if (logits.shape[0] > INT32_MAX || logits.shape[1] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: shape exceeds int32", name);
  }
  *rows = static_cast<int>(logits.shape[0]);
  *vocab = static_cast<int>(logits.shape[1]);
  return Status::kSuccess;
}

Status check_vector(const Tensor& t, DType dtype, int64_t length, size_t alignment,
                    const char* name) {
  Status s = check_tensor(t, dtype, 1, alignment, name);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != length) {
    return detail::set_error(Status::kInvalidArgument, "%s: length %lld, expected %lld", name,
                             static_cast<long long>(t.shape[0]), static_cast<long long>(length));
  }
  if (length > 1 && t.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: stride[0] %lld", name,
                             static_cast<long long>(t.stride[0]));
  }
  return Status::kSuccess;
}

Status check_row_matrix(const Tensor& t, DType dtype, int64_t rows, int64_t cols,
                        const char* name) {
  Status s = check_tensor(t, dtype, 2, element_bytes(dtype), name);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != rows || t.shape[1] != cols) {
    return detail::set_error(Status::kInvalidArgument, "%s: shape [%lld, %lld], expected [%lld, %lld]",
                             name, static_cast<long long>(t.shape[0]),
                             static_cast<long long>(t.shape[1]), static_cast<long long>(rows),
                             static_cast<long long>(cols));
  }
  if (t.stride[1] != 1 || (rows > 1 && t.stride[0] != cols)) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: rows must be contiguous", name);
  }
  return Status::kSuccess;
}

template <typename logits_t>
Status launch_penalties(const Tensor& logits, const Tensor& output_counts, const Tensor& prompt_mask,
                        const Tensor& repetition, const Tensor& frequency, const Tensor& presence,
                        int batch, int vocab, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  apply_penalties_kernel<logits_t, kElemVec><<<dim3(column_grid(vocab), row_grid(batch)), kThreads,
                                              0, stream>>>(
      as_mutable<logits_t>(logits), logits.stride[0], as_const<int32_t>(output_counts),
      tensor_present(prompt_mask) ? as_const<uint32_t>(prompt_mask) : nullptr,
      tensor_present(repetition) ? as_const<float>(repetition) : nullptr,
      tensor_present(frequency) ? as_const<float>(frequency) : nullptr,
      tensor_present(presence) ? as_const<float>(presence) : nullptr, batch, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "apply_penalties launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename logits_t>
Status launch_bitmask(const Tensor& logits, const Tensor& bitmask, const Tensor& rows, int num_rows,
                      int num_mask_rows, int vocab, const LaunchContext& ctx) {
  (void)num_rows;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  apply_token_bitmask_kernel<logits_t, kElemVec><<<
      dim3(column_grid(vocab), row_grid(num_mask_rows)), kThreads, 0, stream>>>(
      as_mutable<logits_t>(logits), logits.stride[0], as_const<int32_t>(bitmask),
      tensor_present(rows) ? as_const<int32_t>(rows) : nullptr, num_mask_rows, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "apply_token_bitmask launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename logits_t>
Status launch_bias(const Tensor& logits, const Tensor& bias_indptr, const Tensor& token_ids,
                   const Tensor& values, int batch, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int blocks = (batch + kThreads - 1) / kThreads;
  apply_logit_bias_kernel<logits_t><<<blocks, kThreads, 0, stream>>>(
      as_mutable<logits_t>(logits), logits.stride[0], as_const<int32_t>(bias_indptr),
      as_const<int32_t>(token_ids), as_const<float>(values), batch);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "apply_logit_bias launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

}  // namespace

// -----------------------------------------------------------------------------
// S02 public API
// -----------------------------------------------------------------------------

Status token_bincount(const Tensor& counts, const Tensor& prompt_mask, const Tensor& token_ids,
                      const Tensor& token_indptr, const Tensor& is_prompt,
                      const LaunchContext& ctx) {
  Status s = check_tensor(counts, DType::kInt32, 2, 4, "token_bincount.counts");
  if (s != Status::kSuccess) return s;
  if (counts.shape[0] > INT32_MAX || counts.shape[1] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "token_bincount.counts: shape exceeds int32");
  }
  const int batch = static_cast<int>(counts.shape[0]);
  const int vocab = static_cast<int>(counts.shape[1]);
  if (vocab <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "token_bincount: vocab %d", vocab);
  }
  if (counts.stride[1] != 1 || (batch > 1 && counts.stride[0] != vocab)) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "token_bincount.counts: rows must be contiguous (stride[0] = V)");
  }
  const int words = (vocab + 31) / 32;
  const bool has_mask = tensor_present(prompt_mask);
  if (has_mask) {
    s = check_row_matrix(prompt_mask, DType::kUInt32, batch, words, "token_bincount.prompt_mask");
    if (s != Status::kSuccess) return s;
  }
  s = check_tensor(token_ids, DType::kInt32, 1, 4, "token_bincount.token_ids");
  if (s != Status::kSuccess) return s;
  const int64_t num_ids = token_ids.shape[0];
  if (num_ids > 1 && token_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "token_bincount.token_ids: stride[0] %lld",
                             static_cast<long long>(token_ids.stride[0]));
  }
  s = check_vector(token_indptr, DType::kInt32, static_cast<int64_t>(batch) + 1, 4,
                   "token_bincount.token_indptr");
  if (s != Status::kSuccess) return s;
  if (tensor_present(is_prompt)) {
    s = check_vector(is_prompt, DType::kUInt8, num_ids, 1, "token_bincount.is_prompt");
    if (s != Status::kSuccess) return s;
    if (!has_mask) {
      return detail::set_error(Status::kInvalidArgument,
                               "token_bincount: is_prompt needs a prompt_mask to record bits");
    }
  }
  if (batch == 0 || num_ids == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  token_bincount_kernel<<<dim3(kBincountSplits, row_grid(batch)), kThreads, 0, stream>>>(
      as_mutable<int32_t>(counts), has_mask ? as_mutable<uint32_t>(prompt_mask) : nullptr,
      as_const<int32_t>(token_ids), as_const<int32_t>(token_indptr),
      tensor_present(is_prompt) ? as_const<uint8_t>(is_prompt) : nullptr, batch, vocab);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "token_bincount launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

Status apply_penalties(const Tensor& logits, const Tensor& output_counts, const Tensor& prompt_mask,
                       const Tensor& repetition, const Tensor& frequency, const Tensor& presence,
                       const LaunchContext& ctx) {
  int batch = 0;
  int vocab = 0;
  Status s = check_logits_matrix(logits, "apply_penalties.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  if (vocab <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "apply_penalties: vocab %d", vocab);
  }
  s = check_row_matrix(output_counts, DType::kInt32, batch, vocab, "apply_penalties.output_counts");
  if (s != Status::kSuccess) return s;
  const int words = (vocab + 31) / 32;
  if (tensor_present(prompt_mask)) {
    s = check_row_matrix(prompt_mask, DType::kUInt32, batch, words, "apply_penalties.prompt_mask");
    if (s != Status::kSuccess) return s;
  }
  for (const Tensor* t : {&repetition, &frequency, &presence}) {
    if (!tensor_present(*t)) continue;
    const char* name = t == &repetition    ? "apply_penalties.repetition"
                       : t == &frequency   ? "apply_penalties.frequency"
                                           : "apply_penalties.presence";
    s = check_vector(*t, DType::kFloat32, batch, 4, name);
    if (s != Status::kSuccess) return s;
  }
  if (batch == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_penalties<float>(logits, output_counts, prompt_mask, repetition, frequency,
                                     presence, batch, vocab, ctx);
    case DType::kFloat16:
      return launch_penalties<__half>(logits, output_counts, prompt_mask, repetition, frequency,
                                      presence, batch, vocab, ctx);
    case DType::kBFloat16:
      return launch_penalties<__nv_bfloat16>(logits, output_counts, prompt_mask, repetition,
                                             frequency, presence, batch, vocab, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "apply_penalties.logits: %s",
                               dtype_name(logits.dtype));
  }
}

Status apply_token_bitmask(const Tensor& logits, const Tensor& bitmask, const Tensor& rows,
                           const LaunchContext& ctx) {
  int num_rows = 0;
  int vocab = 0;
  Status s = check_logits_matrix(logits, "apply_token_bitmask.logits", &num_rows, &vocab);
  if (s != Status::kSuccess) return s;
  if (vocab <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "apply_token_bitmask: vocab %d", vocab);
  }
  if (bitmask.shape[0] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "apply_token_bitmask.bitmask: rows exceed int32");
  }
  const int num_mask_rows = static_cast<int>(bitmask.shape[0]);
  const int words = (vocab + 31) / 32;
  s = check_row_matrix(bitmask, DType::kInt32, num_mask_rows, words, "apply_token_bitmask.bitmask");
  if (s != Status::kSuccess) return s;
  if (tensor_present(rows)) {
    s = check_vector(rows, DType::kInt32, num_mask_rows, 4, "apply_token_bitmask.rows");
    if (s != Status::kSuccess) return s;
  } else if (num_mask_rows != num_rows) {
    return detail::set_error(Status::kInvalidArgument,
                             "apply_token_bitmask: %d mask rows for %d logits rows without a map",
                             num_mask_rows, num_rows);
  }
  if (num_mask_rows == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_bitmask<float>(logits, bitmask, rows, num_rows, num_mask_rows, vocab, ctx);
    case DType::kFloat16:
      return launch_bitmask<__half>(logits, bitmask, rows, num_rows, num_mask_rows, vocab, ctx);
    case DType::kBFloat16:
      return launch_bitmask<__nv_bfloat16>(logits, bitmask, rows, num_rows, num_mask_rows, vocab,
                                           ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "apply_token_bitmask.logits: %s",
                               dtype_name(logits.dtype));
  }
}

Status apply_logit_bias(const Tensor& logits, const Tensor& bias_indptr, const Tensor& token_ids,
                        const Tensor& values, const LaunchContext& ctx) {
  int batch = 0;
  int vocab = 0;
  Status s = check_logits_matrix(logits, "apply_logit_bias.logits", &batch, &vocab);
  if (s != Status::kSuccess) return s;
  s = check_vector(bias_indptr, DType::kInt32, static_cast<int64_t>(batch) + 1, 4,
                   "apply_logit_bias.bias_indptr");
  if (s != Status::kSuccess) return s;
  s = check_tensor(token_ids, DType::kInt32, 1, 4, "apply_logit_bias.token_ids");
  if (s != Status::kSuccess) return s;
  const int64_t nnz = token_ids.shape[0];
  if (nnz > 1 && token_ids.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "apply_logit_bias.token_ids: stride[0] %lld",
                             static_cast<long long>(token_ids.stride[0]));
  }
  s = check_vector(values, DType::kFloat32, nnz, 4, "apply_logit_bias.values");
  if (s != Status::kSuccess) return s;
  if (batch == 0 || nnz == 0) return Status::kSuccess;

  switch (logits.dtype) {
    case DType::kFloat32:
      return launch_bias<float>(logits, bias_indptr, token_ids, values, batch, ctx);
    case DType::kFloat16:
      return launch_bias<__half>(logits, bias_indptr, token_ids, values, batch, ctx);
    case DType::kBFloat16:
      return launch_bias<__nv_bfloat16>(logits, bias_indptr, token_ids, values, batch, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "apply_logit_bias.logits: %s",
                               dtype_name(logits.dtype));
  }
}

}  // namespace kops::sample
