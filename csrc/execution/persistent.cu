// =============================================================================
// kernel-ops | csrc/execution/persistent.cu
// Host-visible reset kernel for the epoch-tagged persistent counters. The device
// primitives (task_queue_next / task_wait / task_signal / last_cta_arrive /
// scheduler_next) are inline in persistent.cuh so every kernel TU sees the same
// progress-rule-safe implementation; this TU only contains the reset kernel.
// =============================================================================
#include <cstdint>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/sync.cuh"
#include "csrc/execution/persistent.cuh"

namespace kops::exec {

__global__ void task_queue_reset_kernel(uint32_t* __restrict__ cursor,
                                        uint32_t* __restrict__ counters,
                                        const int num_counters) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && cursor != nullptr) *cursor = 0u;
  for (int i = static_cast<int>(threadIdx.x); i < num_counters; i += blockDim.x) {
    counters[i] = 0u;
  }
}

}  // namespace kops::exec
