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
  uint32_t epoch;          // incremented per launch by the plan (no host reset between replays)
};

/// Fetch the next task index (or -1 when exhausted). Monotonic -> satisfies progress rule (c).
KOPS_DEVICE int32_t task_queue_next(const TaskQueue& q);
/// Block until counters[task.wait_counter] reaches task.wait_target in q.epoch (acquire, gpu scope).
KOPS_DEVICE void task_wait(const TaskQueue& q, const TileTask& task);
/// Publish completion with release semantics after all output stores (and TMA stores) are complete.
KOPS_DEVICE void task_signal(const TaskQueue& q, const TileTask& task);

/// Static persistent tile scheduler for single-operator persistent kernels (GEMM / attention):
/// linear tile id -> (m, n, group) with swizzle; CTA c processes tiles c, c + grid, ...
struct PersistentTileScheduler {
  int32_t num_tiles_m, num_tiles_n, num_groups;
  int32_t swizzle_log;     // raster swizzle for L2 reuse
  int32_t grid_size;
};
KOPS_DEVICE bool scheduler_next(const PersistentTileScheduler& s, int32_t& linear_tile,
                                int32_t& tile_m, int32_t& tile_n, int32_t& group);

/**
 * task_queue_reset_kernel — only needed when the epoch counter would wrap (every 2^12 launches with
 * the default 20-bit count field); plans schedule it outside the hot path.
 */
__global__ void task_queue_reset_kernel(uint32_t* __restrict__ cursor,
                                        uint32_t* __restrict__ counters, const int num_counters);

}  // namespace kops::exec
