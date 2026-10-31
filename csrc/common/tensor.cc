// =============================================================================
// kernel-ops | csrc/common/tensor.cc
// Host-side tensor validation and device probing. This file touches the CUDA
// runtime (cuda_runtime.h), so it is compiled with CUDA language enabled; the
// public header stays backend-neutral.
// =============================================================================
#include <cstdint>
#include <cstring>

#include "csrc/common/macros.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"

namespace kops {
namespace {

int32_t kElementBytes(DType t) { return (dtype_bits(t) + 7) / 8; }

bool element_aligned(const Tensor& t, int32_t elem_bytes, size_t alignment) {
  if (alignment == 0) return true;
  const auto base = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
  if (base % alignment != 0) return false;
  if (t.stride_unit == StrideUnit::kElement && elem_bytes > 0) {
    // Only the strides that start a row must be aligned. The innermost dim is
    // element-contiguous (stride 1), and requiring it to be 16 B aligned would
    // reject every 2-byte dtype.
    for (int32_t d = 0; d + 1 < t.ndim; ++d) {
      if (t.shape[d] <= 1) continue;
      if ((t.stride[d] * elem_bytes) % static_cast<int64_t>(alignment) != 0) return false;
    }
  }
  return true;
}

}  // namespace

bool tensor_present(const Tensor& t) noexcept { return t.data != nullptr; }

int64_t tensor_numel(const Tensor& t) noexcept {
  if (t.ndim <= 0 || t.ndim > kMaxTensorDims) return 0;
  int64_t n = 1;
  for (int32_t d = 0; d < t.ndim; ++d) {
    if (t.shape[d] < 0) return 0;
    n *= t.shape[d];
  }
  return n;
}

bool tensor_is_contiguous(const Tensor& t) noexcept {
  if (t.ndim <= 0 || t.ndim > kMaxTensorDims) return false;
  const int32_t bits = dtype_bits(t.dtype);
  if (bits <= 0) return false;
  // one logical element = bits/8 bytes; work in eighths of a byte to stay exact
  // for packed 4-bit dtypes.
  int64_t expected = 1;  // in bytes * 8
  for (int32_t d = t.ndim - 1; d >= 0; --d) {
    const int64_t stride_bytes_x8 =
        (t.stride_unit == StrideUnit::kByte) ? t.stride[d] * 8 : t.stride[d] * bits;
    if (t.shape[d] > 1 && stride_bytes_x8 != expected) return false;
    expected *= t.shape[d] * bits;
  }
  return true;
}

Status check_tensor(const Tensor& t, DType expected, int32_t expected_ndim, size_t alignment_bytes,
                    const char* arg_name) noexcept {
  const char* name = (arg_name != nullptr) ? arg_name : "tensor";
  if (t.data == nullptr) return detail::set_error(Status::kInvalidArgument, "%s: null data", name);
  if (expected != DType::kInvalid && t.dtype != expected) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s, expected %s", name,
                             dtype_name(t.dtype), dtype_name(expected));
  }
  if (expected_ndim >= 0 && t.ndim != expected_ndim) {
    return detail::set_error(Status::kInvalidArgument, "%s: ndim %d, expected %d", name, t.ndim,
                             expected_ndim);
  }
  if (t.ndim <= 0 || t.ndim > kMaxTensorDims) {
    return detail::set_error(Status::kInvalidArgument, "%s: ndim %d out of range", name, t.ndim);
  }
  if (t.device < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: device %d is not a device tensor", name,
                             t.device);
  }
  for (int32_t d = 0; d < t.ndim; ++d) {
    if (t.shape[d] < 0) {
      return detail::set_error(Status::kInvalidArgument, "%s: shape[%d] = %lld < 0", name, d,
                               static_cast<long long>(t.shape[d]));
    }
  }
  const int32_t elem_bytes = kElementBytes(t.dtype);
  if (elem_bytes == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: invalid dtype", name);
  }
  const bool packed = dtype_is_packed_subbyte(t.dtype);
  if (t.stride_unit == StrideUnit::kByte && !packed) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: byte strides are only valid for packed sub-byte dtypes", name);
  }
  if (alignment_bytes > 1 && !element_aligned(t, elem_bytes, alignment_bytes)) {
    return detail::set_error(Status::kMisaligned, "%s: violates %zu-byte alignment", name,
                             alignment_bytes);
  }
  return Status::kSuccess;
}

Status make_launch_context(int32_t device, kops_stream_t stream, LaunchContext* out) noexcept {
  if (out == nullptr) return detail::set_error(Status::kInvalidArgument, "null LaunchContext");
  cudaDeviceProp prop{};
  const cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return detail::set_error(Status::kCudaError, "cudaGetDeviceProperties(%d): %s", device,
                             cudaGetErrorString(err));
  }
  out->backend = BackendKind::kCuda;
  out->arch_id = prop.major * 100 + prop.minor * 10;
  out->wave_size = 32;
  out->compute_units = prop.multiProcessorCount;
  out->lds_bytes = static_cast<int32_t>(prop.sharedMemPerBlock);
  out->max_ctas_per_cu = prop.maxBlocksPerMultiProcessor;
  out->stream = stream;
  out->device = device;
  return Status::kSuccess;
}

}  // namespace kops
