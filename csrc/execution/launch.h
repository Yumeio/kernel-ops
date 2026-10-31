#pragma once
// =============================================================================
// kernel-ops | csrc/execution/launch.h                          Catalog: E01, E02, I02
// Host helpers that every launcher uses: launch attributes (PDL, clusters,
// cooperative), graph-capture guards, occupancy queries. The engine owns graph
// capture itself; the library guarantees run() is capture-safe (no allocation,
// no sync, no JIT, no host reads of device data).
// Requires: common/*, execution/persistent.cuh.
//
// Backend neutrality: streams/graphs are opaque handles. PDL and clusters are
// CUDA SM90+ features; the deferred HIP path must reject them with
// kUnsupportedBackend rather than silently downgrade (a silent downgrade would
// change overlap and ordering guarantees the caller relies on).
// =============================================================================

namespace kops::exec {

/// Launch configuration resolved by a plan.
struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t dynamic_smem_bytes = 0;
  dim3 cluster;                     // {1,1,1} = no cluster (CUDA SM90+ only)
  bool programmatic_serialization = false;  // PDL attribute (CUDA SM90+, ctx.enable_pdl)
  bool cooperative = false;         // cooperative-launch semantics (grid-wide sync legal)
  int32_t max_active_ctas = 0;      // occupancy result used to size persistent grids
};

/// True if `stream` is currently capturing (cudaStreamIsCapturing / hipStreamIsCapturing);
/// used to reject prepare()/JIT inside a capture.
bool stream_is_capturing(kops_stream_t stream) noexcept;

/// Occupancy-based persistent grid: min(compute_units * max_active_per_cu, work_items).
Status persistent_grid(const void* kernel_fn, int32_t block_threads, size_t dynamic_smem_bytes,
                       const LaunchContext& ctx, int64_t work_items, LaunchConfig* out);

/// Launch any declared kernel with the attributes in cfg (PDL / cluster / cooperative).
Status launch_kernel(const void* kernel_fn, const LaunchConfig& cfg, void** args,
                     const LaunchContext& ctx);

/// Graph helpers for callers that let the library record a self-contained sub-graph (optional;
/// engines usually capture whole steps themselves).
struct GraphHandle {
  kops_graph_t graph = nullptr;
  kops_graph_t exec = nullptr;
  uint32_t capture_generation = 0;
};
Status graph_begin_capture(kops_stream_t stream);
Status graph_end_capture(kops_stream_t stream, uint32_t capture_generation, GraphHandle* out);
Status graph_replay(const GraphHandle& g, const LaunchContext& ctx);
/// Update one node's parameters in place (cudaGraphExecKernelNodeSetParams equivalent).
/// Returns kNotImplemented on backends without it, so callers can fall back to recapture.
Status graph_update_node(const GraphHandle& g, uint64_t node_id, void** args);
Status graph_destroy(GraphHandle* g);

}  // namespace kops::exec
