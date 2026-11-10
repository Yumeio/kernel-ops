#pragma once
// =============================================================================
// kernel-ops | csrc/execution/persistent.cuh                    Catalog: E02, E03, I02
// Persistent-CTA task runtime used by persistent GEMM/attention, dynamic decode,
// grouped GEMM and every Megakernel.
// Progress rule (enforced by plans, documented here): a CTA may only wait on a
// dependency whose producer is either (a) already finished, (b) resident because
// the grid was launched cooperatively / with grid <= max active CTAs, or (c) on a
// task that is dequeued strictly earlier by a monotonic counter. Never spin on
// work that might not be scheduled.
// Requires: common/macros.cuh, common/sync.cuh.
// =============================================================================

namespace kops::exec {

/// One unit of work in a prepared DAG.
struct TileTask {
  uint16_t op_id;          // stage in the plan's operator DAG (e.g. 0 = gate_up, 1 = down)
  uint16_t flags;          // bit0 last-k-split, bit1 needs-reduction, bit2 publishes-output
  int32_t tile_m;          // tile coordinates
  int32_t tile_n;
  int32_t k_begin, k_end;  // K range (split-K / stream-K)
  int32_t group;           // expert / request / head group
  int32_t wait_counter;    // dependency counter id (-1 none)
  int32_t wait_target;     // count that must be reached (within the current epoch)
  int32_t signal_counter;  // counter incremented on completion (-1 none)
  int32_t out_slot;        // intermediate buffer slot (workspace)
};

/// Device-resident queue (static order; dynamic fetch via an atomic cursor).
struct TaskQueue {
  const TileTask* tasks;   // [num_tasks] topologically ordered
  int32_t num_tasks;
  uint32_t* cursor;        // epoch-tagged atomic fetch counter
  uint32_t* counters;      // [num_counters] epoch-tagged dependency counters
  int32_t num_counters;
  uint32_t epoch;          // launch tag for every epoch-tagged counter (no host reset between replays)
};

/// Fetch the next task index (or -1 when exhausted). Monotonic -> satisfies progress rule (c).
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
KOPS_DEVICE int32_t task_queue_next(const TaskQueue& q) {
  const uint32_t old = atomicAdd(q.cursor, 1u);
  if (static_cast<int32_t>(old) >= q.num_tasks) return -1;
  return static_cast<int32_t>(old);
}

namespace detail {
constexpr uint32_t kCountMask = (1u << 20) - 1u;
constexpr uint32_t kEpochShift = 20;
KOPS_DEVICE uint32_t count_of(uint32_t v) { return v & kCountMask; }
KOPS_DEVICE uint32_t epoch_of(uint32_t v) { return v >> kEpochShift; }
KOPS_DEVICE void signal_epoch(uint32_t* counter, uint32_t epoch) {
  uint32_t old = *reinterpret_cast<volatile uint32_t*>(counter);
  for (;;) {
    const uint32_t next = (epoch_of(old) == epoch)
                              ? ((epoch << kEpochShift) | (count_of(old) + 1u))
                              : ((epoch << kEpochShift) | 1u);
    const uint32_t prev = atomicCAS(counter, old, next);
    if (prev == old) return;
    old = prev;
  }
}
KOPS_DEVICE void wait_epoch(uint32_t* counter, uint32_t epoch, uint32_t target) {
  for (;;) {
    const uint32_t v = *reinterpret_cast<volatile uint32_t*>(counter);
    if (epoch_of(v) == epoch && count_of(v) >= target) return;
    __nanosleep(32);
  }
}
}  // namespace detail

/// Block until counters[task.wait_counter] reaches task.wait_target in q.epoch (acquire, gpu scope).
KOPS_DEVICE void task_wait(const TaskQueue& q, const TileTask& task) {
  if (task.wait_counter < 0 || q.counters == nullptr) return;
  detail::wait_epoch(q.counters + task.wait_counter, q.epoch,
                     static_cast<uint32_t>(task.wait_target));
}

/// Publish completion with release semantics after all output stores (and TMA stores) are complete.
KOPS_DEVICE void task_signal(const TaskQueue& q, const TileTask& task) {
  if (task.signal_counter < 0 || q.counters == nullptr) return;
  __threadfence();
  detail::signal_epoch(q.counters + task.signal_counter, q.epoch);
}

/// Split-K / stream-K last-CTA arrival on an epoch-tagged counter (common/sync.cuh).
/// Every participating CTA calls this once after it has published its partial tile with release
/// ordering. Exactly one caller -- the last arrival of `epoch` -- gets `true` and must fold the
/// partials in fixed slice order, never in arrival order. The epoch tag lives in the counter
/// value, so an arrival left over from an earlier graph replay is never mistaken for a current
/// one: there is no host reset between replays, and `task_queue_reset_kernel` only runs on the
/// 20-bit count wrap. This is the progress-rule-safe substitute for a cluster barrier on
/// targets without clusters (ROADMAP B13/B15, split-K reduction).
KOPS_DEVICE bool last_cta_arrive(uint32_t* counter, uint32_t epoch, uint32_t num_arrivals) {
  if (counter == nullptr || num_arrivals == 0u) return false;
  uint32_t old = *reinterpret_cast<volatile uint32_t*>(counter);
  for (;;) {
    const bool same_epoch = detail::epoch_of(old) == epoch;
    const uint32_t next =
        (epoch << detail::kEpochShift) | (same_epoch ? (detail::count_of(old) + 1u) : 1u);
    const uint32_t prev = atomicCAS(counter, old, next);
    if (prev == old) {
      __threadfence();
      return same_epoch && (detail::count_of(old) + 1u) >= num_arrivals;
    }
    old = prev;
  }
}
#else
KOPS_DEVICE int32_t task_queue_next(const TaskQueue& q);
KOPS_DEVICE void task_wait(const TaskQueue& q, const TileTask& task);
KOPS_DEVICE void task_signal(const TaskQueue& q, const TileTask& task);
KOPS_DEVICE bool last_cta_arrive(uint32_t* counter, uint32_t epoch, uint32_t num_arrivals);
#endif

/// Static persistent tile scheduler for single-operator persistent kernels (GEMM / attention):
/// linear tile id -> (m, n, group) with swizzle; CTA c processes tiles c, c + grid, ...
struct PersistentTileScheduler {
  int32_t num_tiles_m, num_tiles_n, num_groups;
  int32_t swizzle_log;     // raster swizzle for L2 reuse
  int32_t grid_size;
};
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
KOPS_DEVICE bool scheduler_next(const PersistentTileScheduler& s, int32_t& linear_tile,
                                int32_t& tile_m, int32_t& tile_n, int32_t& group) {
  if (linear_tile < 0 || linear_tile >= s.num_tiles_m * s.num_tiles_n * s.num_groups) {
    return false;
  }
  const int32_t tiles_per_group = s.num_tiles_m * s.num_tiles_n;
  group = linear_tile / tiles_per_group;
  const int32_t in_group = linear_tile - group * tiles_per_group;
  tile_m = in_group / s.num_tiles_n;
  tile_n = in_group - tile_m * s.num_tiles_n;
  linear_tile += s.grid_size > 0 ? s.grid_size : 1;
  return true;
}
#else
KOPS_DEVICE bool scheduler_next(const PersistentTileScheduler& s, int32_t& linear_tile,
                                int32_t& tile_m, int32_t& tile_n, int32_t& group);
#endif

/**
 * task_queue_reset_kernel — only needed when the epoch counter would wrap (every 2^12 launches with
 * the default 20-bit count field); plans schedule it outside the hot path.
 */
__global__ void task_queue_reset_kernel(uint32_t* __restrict__ cursor,
                                        uint32_t* __restrict__ counters, const int num_counters);

}  // namespace kops::exec
