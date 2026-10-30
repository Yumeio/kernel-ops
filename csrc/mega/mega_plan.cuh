#pragma once
// =============================================================================
// kernel-ops | csrc/mega/mega_plan.cuh                           Catalog: E04-E08
// MegaPlan: bounded operator DAG + tile schedule + dependency counters + workspace
// layout for one Megakernel instantiation. A Megakernel is a candidate execution
// strategy; it is only selected when it beats optimized modular + CUDA Graph (+PDL)
// on the declared domain (see docs, benchmark gates).
// Requires: common/*, execution/persistent.cuh, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::mega {

/// Operators that may appear as Megakernel stages.
enum class StageOp : uint8_t {
  kRMSNorm = 0, kResidualAdd = 1, kGemm = 2, kGluActQuant = 3, kRope = 4, kKVAppend = 5,
  kAttentionDecode = 6, kRoute = 7, kExpertGemm = 8, kCombine = 9, kDispatch = 10,
  kAllReduce = 11, kHCMix = 12, kAttnResMix = 13,
};

struct StageDesc {
  StageOp op;
  uint8_t num_inputs;
  uint8_t num_outputs;
  uint8_t reserved;
  int32_t input_slots[4];          // workspace slot ids (or -1 = external tensor)
  int32_t output_slots[2];
  int32_t first_task, num_tasks;   // range in the TaskQueue
};

/// Device-visible part of a prepared MegaPlan.
struct MegaPlanDevice {
  const StageDesc* stages;         // [num_stages] topological order
  int32_t num_stages;
  exec::TaskQueue queue;           // tiles of all stages, dependency counters, epoch
  void* const* slots;              // [num_slots] intermediate buffers (global, L2-resident when small)
  const int64_t* slot_bytes;
  int32_t num_slots;
  int32_t max_m;                   // capacity bound (tokens); run() rejects larger batches before enqueue
  uint32_t capture_generation;
};

/// Host-side summary used by registry / benchmarks.
struct MegaPlanInfo {
  uint64_t kernel_id;
  int32_t num_device_launches;     // including any zero/reset/copy launches (must be reported)
  int32_t num_stages;
  int32_t num_tasks;
  int32_t ctas;                    // persistent grid size
  int32_t regs_per_thread;
  int32_t smem_per_cta;
  WorkspaceRequirements workspace;
};

}  // namespace kops::mega
