#pragma once
// =============================================================================
// kernel-ops | csrc/execution/launch.h                          Catalog: E01, E02, I02
// Host helpers that every launcher uses: launch attributes (PDL, clusters,
// cooperative), graph-capture guards, occupancy queries. The engine owns graph
// capture itself; the library guarantees run() is capture-safe (no allocation,
// no sync, no JIT, no host reads of device data).
// Requires: common/*, execution/persistent.cuh.
// =============================================================================

namespace kops::exec {

/// Launch configuration resolved by a plan.
struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t dynamic_smem_bytes = 0;
  dim3 cluster;                     // {1,1,1} = no cluster
  bool programmatic_serialization = false;  // PDL attribute (SM90+, ctx.enable_pdl)
  bool cooperative = false;         // cudaLaunchCooperativeKernel semantics (grid-wide sync legal)
  int32_t max_active_ctas = 0;      // occupancy result used to size persistent grids
};

/// True if `stream` is currently capturing (cudaStreamIsCapturing); used to reject prepare()/JIT.
bool stream_is_capturing(cudaStream_t stream) noexcept;

/// Occupancy-based persistent grid: min(num_sms * max_active_per_sm, work_items).
Status persistent_grid(const void* kernel_fn, int32_t block_threads, size_t dynamic_smem_bytes,
                       const LaunchContext& ctx, int64_t work_items, LaunchConfig* out);

/// Launch any declared kernel with the attributes in cfg (PDL / cluster / cooperative).
Status launch_kernel(const void* kernel_fn, const LaunchConfig& cfg, void** args,
                     const LaunchContext& ctx);

/// Graph helpers for callers that let the library record a self-contained sub-graph (optional;
/// engines usually capture whole steps themselves).
struct GraphHandle {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  uint32_t capture_generation = 0;
};
Status graph_begin_capture(cudaStream_t stream);
Status graph_end_capture(cudaStream_t stream, uint32_t capture_generation, GraphHandle* out);
Status graph_replay(const GraphHandle& g, const LaunchContext& ctx);
Status graph_destroy(GraphHandle* g);

}  // namespace kops::exec
