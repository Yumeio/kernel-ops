// =============================================================================
// kernel-ops | csrc/state/state_cache.cu
// B2 L03 byte-preserving recurrent/conv state cache primitives:
//   state_copy, state_gather, state_scatter, state_zero.
//
// Policy (which slot, when to snapshot, prefix identity, eviction) belongs to
// the engine; these kernels only move bytes between slots of the pools it
// names. Every launch is graph-capturable: no allocation, no host sync.
//
// Layout: pools[num_pools] is a device array of base addresses; slot_bytes[p]
// is the byte size of one slot of pool p; slots are addressed by index.
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
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

namespace kops::state {
namespace {

constexpr int kBlock = 256;

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

}  // namespace

// Declared in state_cache_kernels.cuh.
__global__ void state_copy_kernel(void* const* __restrict__ pools,
                                  const int64_t* __restrict__ slot_bytes,
                                  const int32_t* __restrict__ slot_pairs, const int num_pairs,
                                  const int num_pools) {
  const int pair = static_cast<int>(blockIdx.x);
  const int pool = static_cast<int>(blockIdx.y);
  if (pair >= num_pairs || pool >= num_pools) return;
  const int32_t src = slot_pairs[static_cast<int64_t>(pair) * 2];
  const int32_t dst = slot_pairs[static_cast<int64_t>(pair) * 2 + 1];
  if (src < 0 || dst < 0) return;
  const int64_t bytes = slot_bytes[pool];
  if (bytes <= 0) return;
  const uint8_t* sp = static_cast<const uint8_t*>(pools[pool]) + static_cast<int64_t>(src) * bytes;
  uint8_t* dp = static_cast<uint8_t*>(pools[pool]) + static_cast<int64_t>(dst) * bytes;
  copy_bytes(dp, sp, bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

__device__ __forceinline__ int64_t pool_offset(const int64_t* slot_bytes, int pool) {
  int64_t offset = 0;
  for (int i = 0; i < pool; ++i) offset += slot_bytes[i];
  return offset;
}

__global__ void state_gather_kernel(void* __restrict__ out, void* const* __restrict__ pools,
                                    const int64_t* __restrict__ slot_bytes,
                                    const int32_t* __restrict__ slots, const int num_items,
                                    const int num_pools) {
  const int item = static_cast<int>(blockIdx.x);
  const int pool = static_cast<int>(blockIdx.y);
  if (item >= num_items || pool >= num_pools) return;
  const int32_t slot = slots[item];
  if (slot < 0) return;
  const int64_t bytes = slot_bytes[pool];
  if (bytes <= 0) return;
  const uint8_t* src =
      static_cast<const uint8_t*>(pools[pool]) + static_cast<int64_t>(slot) * bytes;
  uint8_t* dst = static_cast<uint8_t*>(out) + static_cast<int64_t>(item) * pool_offset(slot_bytes, num_pools) +
                 pool_offset(slot_bytes, pool);
  copy_bytes(dst, src, bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

__global__ void state_scatter_kernel(void* const* __restrict__ pools,
                                     const void* __restrict__ src,
                                     const int64_t* __restrict__ slot_bytes,
                                     const int32_t* __restrict__ slots, const int num_items,
                                     const int num_pools) {
  const int item = static_cast<int>(blockIdx.x);
  const int pool = static_cast<int>(blockIdx.y);
  if (item >= num_items || pool >= num_pools) return;
  const int32_t slot = slots[item];
  if (slot < 0) return;
  const int64_t bytes = slot_bytes[pool];
  if (bytes <= 0) return;
  const uint8_t* sp = static_cast<const uint8_t*>(src) +
                      static_cast<int64_t>(item) * pool_offset(slot_bytes, num_pools) +
                      pool_offset(slot_bytes, pool);
  uint8_t* dst = static_cast<uint8_t*>(pools[pool]) + static_cast<int64_t>(slot) * bytes;
  copy_bytes(dst, sp, bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

__global__ void state_zero_kernel(void* const* __restrict__ pools,
                                  const int64_t* __restrict__ slot_bytes,
                                  const int32_t* __restrict__ slots, const int num_items,
                                  const int num_pools) {
  const int item = static_cast<int>(blockIdx.x);
  const int pool = static_cast<int>(blockIdx.y);
  if (item >= num_items || pool >= num_pools) return;
  const int32_t slot = slots[item];
  if (slot < 0) return;
  const int64_t bytes = slot_bytes[pool];
  if (bytes <= 0) return;
  uint8_t* dst = static_cast<uint8_t*>(pools[pool]) + static_cast<int64_t>(slot) * bytes;
  zero_bytes(dst, bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

// B10: after speculative verification, adopt the intermediate state that
// corresponds to the number of accepted drafts.  The live slot already holds
// the state after the full draft forward, so a row that accepted every draft
// needs no copy.  intermediate[b, j] is the state after j accepted drafts.
__global__ void state_commit_accepted_kernel(
    void* const* __restrict__ pools, const int64_t* __restrict__ slot_bytes,
    const int32_t* __restrict__ live_slots, const int32_t* __restrict__ intermediate_slots,
    const int32_t* __restrict__ num_accepted, const int max_draft_plus_one,
    const int batch_size, const int num_pools) {
  const int b = static_cast<int>(blockIdx.x);
  const int pool = static_cast<int>(blockIdx.y);
  if (b >= batch_size || pool >= num_pools) return;
  const int accepted = num_accepted[b];
  if (accepted < 0 || accepted >= max_draft_plus_one - 1) return;
  const int32_t live = live_slots[b];
  const int32_t src = intermediate_slots[static_cast<int64_t>(b) * max_draft_plus_one + accepted];
  if (live < 0 || src < 0) return;
  const int64_t bytes = slot_bytes[pool];
  if (bytes <= 0) return;
  uint8_t* dst = static_cast<uint8_t*>(pools[pool]) + static_cast<int64_t>(live) * bytes;
  const uint8_t* sp = static_cast<const uint8_t*>(pools[pool]) + static_cast<int64_t>(src) * bytes;
  copy_bytes(dst, sp, bytes, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

namespace {

// ---- host helpers ------------------------------------------------------------

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
    return detail::set_error(Status::kUnsupportedBackend, "state: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

Status check_pools(const Tensor& pool_ptrs, const Tensor& slot_bytes, int* num_pools,
                   int64_t* total_bytes) {
  Status s = check_tensor(pool_ptrs, DType::kInt64, 1, 8, "state.pool_ptrs");
  if (s != Status::kSuccess) return s;
  s = check_tensor(slot_bytes, DType::kInt64, 1, 8, "state.slot_bytes");
  if (s != Status::kSuccess) return s;
  if (pool_ptrs.shape[0] != slot_bytes.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "state: pool/slot_bytes count mismatch");
  }
  if (pool_ptrs.shape[0] <= 0 || pool_ptrs.shape[0] > INT32_MAX) {
    return detail::set_error(Status::kInvalidArgument, "state: num_pools %lld",
                             static_cast<long long>(pool_ptrs.shape[0]));
  }
  *num_pools = static_cast<int>(pool_ptrs.shape[0]);
  *total_bytes = 0;
  return Status::kSuccess;
}

Status check_1d_i32(const Tensor& t, const char* name, int64_t* n) {
  Status s = check_tensor(t, DType::kInt32, 1, 4, name);
  if (s != Status::kSuccess) return s;
  *n = t.shape[0];
  return Status::kSuccess;
}

}  // namespace

Status state_copy(const Tensor& pool_ptrs, const Tensor& slot_bytes, const Tensor& slot_pairs,
                  const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  int pools = 0;
  int64_t total = 0;
  s = check_pools(pool_ptrs, slot_bytes, &pools, &total);
  if (s != Status::kSuccess) return s;
  s = check_tensor(slot_pairs, DType::kInt32, 2, 4, "state.slot_pairs");
  if (s != Status::kSuccess) return s;
  if (slot_pairs.shape[1] != 2 || slot_pairs.stride[0] != 2 || slot_pairs.stride[1] != 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "state_copy: slot_pairs must be packed [N, 2]");
  }
  const int64_t pairs = slot_pairs.shape[0];
  if (pairs == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(pairs), static_cast<unsigned>(pools));
  state_copy_kernel<<<grid, kBlock, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(pool_ptrs)), const_ptr<int64_t>(slot_bytes),
      const_ptr<int32_t>(slot_pairs), static_cast<int>(pairs), pools);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("state_copy", err);
  return Status::kSuccess;
}

Status state_gather(const Tensor& out, const Tensor& pool_ptrs, const Tensor& slot_bytes,
                    const Tensor& slots, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  int pools = 0;
  int64_t total = 0;
  s = check_pools(pool_ptrs, slot_bytes, &pools, &total);
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, out.dtype, 2, 1, "state_gather.out");
  if (s != Status::kSuccess) return s;
  int64_t items = 0;
  s = check_1d_i32(slots, "state_gather.slots", &items);
  if (s != Status::kSuccess) return s;
  if (out.shape[0] < items) {
    return detail::set_error(Status::kInvalidArgument, "state_gather: out rows %lld < %lld",
                             static_cast<long long>(out.shape[0]), static_cast<long long>(items));
  }
  const int64_t elem = (dtype_bits(out.dtype) + 7) / 8;
  if (elem <= 0 || out.shape[1] <= 0 || out.stride[1] != 1 ||
      out.stride[0] != out.shape[1]) {
    return detail::set_error(Status::kInvalidArgument,
                             "state_gather: staging must be packed [items, bytes]");
  }
  (void)total;
  if (items == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(items), static_cast<unsigned>(pools));
  state_gather_kernel<<<grid, kBlock, 0, stream_of(ctx)>>>(
      out.data, reinterpret_cast<void* const*>(const_ptr<int64_t>(pool_ptrs)),
      const_ptr<int64_t>(slot_bytes), const_ptr<int32_t>(slots), static_cast<int>(items), pools);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("state_gather", err);
  return Status::kSuccess;
}

Status state_scatter(const Tensor& pool_ptrs, const Tensor& src, const Tensor& slot_bytes,
                     const Tensor& slots, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  int pools = 0;
  int64_t total = 0;
  s = check_pools(pool_ptrs, slot_bytes, &pools, &total);
  if (s != Status::kSuccess) return s;
  s = check_tensor(src, src.dtype, 2, 1, "state_scatter.src");
  if (s != Status::kSuccess) return s;
  int64_t items = 0;
  s = check_1d_i32(slots, "state_scatter.slots", &items);
  if (s != Status::kSuccess) return s;
  if (src.shape[0] < items) {
    return detail::set_error(Status::kInvalidArgument, "state_scatter: src rows %lld < %lld",
                             static_cast<long long>(src.shape[0]), static_cast<long long>(items));
  }
  const int64_t elem = (dtype_bits(src.dtype) + 7) / 8;
  if (elem <= 0 || src.shape[1] <= 0 || src.stride[1] != 1 ||
      src.stride[0] != src.shape[1]) {
    return detail::set_error(Status::kInvalidArgument,
                             "state_scatter: staging must be packed [items, bytes]");
  }
  (void)total;
  if (items == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(items), static_cast<unsigned>(pools));
  state_scatter_kernel<<<grid, kBlock, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(pool_ptrs)), src.data,
      const_ptr<int64_t>(slot_bytes), const_ptr<int32_t>(slots), static_cast<int>(items), pools);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("state_scatter", err);
  return Status::kSuccess;
}

Status state_zero(const Tensor& pool_ptrs, const Tensor& slot_bytes, const Tensor& slots,
                  const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  int pools = 0;
  int64_t total = 0;
  s = check_pools(pool_ptrs, slot_bytes, &pools, &total);
  if (s != Status::kSuccess) return s;
  int64_t items = 0;
  s = check_1d_i32(slots, "state_zero.slots", &items);
  if (s != Status::kSuccess) return s;
  if (items == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(items), static_cast<unsigned>(pools));
  state_zero_kernel<<<grid, kBlock, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(pool_ptrs)), const_ptr<int64_t>(slot_bytes),
      const_ptr<int32_t>(slots), static_cast<int>(items), pools);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("state_zero", err);
  return Status::kSuccess;
}

Status state_commit_accepted(const Tensor& pool_ptrs, const Tensor& slot_bytes,
                             const Tensor& live_slots, const Tensor& intermediate_slots,
                             const Tensor& num_accepted, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  int pools = 0;
  int64_t total = 0;
  s = check_pools(pool_ptrs, slot_bytes, &pools, &total);
  if (s != Status::kSuccess) return s;
  s = check_tensor(live_slots, DType::kInt32, 1, 4, "state_commit_accepted.live_slots");
  if (s != Status::kSuccess) return s;
  s = check_tensor(intermediate_slots, DType::kInt32, 2, 4,
                   "state_commit_accepted.intermediate_slots");
  if (s != Status::kSuccess) return s;
  s = check_tensor(num_accepted, DType::kInt32, 1, 4, "state_commit_accepted.num_accepted");
  if (s != Status::kSuccess) return s;
  const int64_t batch = live_slots.shape[0];
  const int64_t max_draft_plus_one = intermediate_slots.shape[1];
  if (max_draft_plus_one <= 0 || intermediate_slots.shape[0] != batch ||
      num_accepted.shape[0] != batch || live_slots.stride[0] != 1 ||
      num_accepted.stride[0] != 1 || intermediate_slots.stride[0] != max_draft_plus_one ||
      intermediate_slots.stride[1] != 1) {
    return detail::set_error(Status::kInvalidArgument,
                             "state_commit_accepted: live/intermediate/num_accepted shapes");
  }
  if (batch <= 0) return Status::kSuccess;
  if (batch > INT32_MAX || max_draft_plus_one > INT32_MAX) {
    return detail::set_error(Status::kInvalidArgument,
                             "state_commit_accepted: batch or draft table too large");
  }
  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(pools));
  state_commit_accepted_kernel<<<grid, kBlock, 0, stream_of(ctx)>>>(
      reinterpret_cast<void* const*>(const_ptr<int64_t>(pool_ptrs)), const_ptr<int64_t>(slot_bytes),
      const_ptr<int32_t>(live_slots), const_ptr<int32_t>(intermediate_slots),
      const_ptr<int32_t>(num_accepted), static_cast<int>(max_draft_plus_one),
      static_cast<int>(batch), pools);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return launch_error("state_commit_accepted", err);
  return Status::kSuccess;
}

}  // namespace kops::state
