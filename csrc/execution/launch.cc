// =============================================================================
// kernel-ops | csrc/execution/launch.cc
// Host launch helpers: capture detection, occupancy-based persistent grids,
// CUDA launch attributes (PDL/cluster/cooperative) and caller-owned graphs.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/execution/launch.h"

namespace kops::exec {
namespace {
cudaStream_t cuda_stream(kops_stream_t stream) { return reinterpret_cast<cudaStream_t>(stream); }
}  // namespace

bool stream_is_capturing(kops_stream_t stream) noexcept {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaError_t err = cudaStreamIsCapturing(cuda_stream(stream), &status);
  return err != cudaSuccess || status != cudaStreamCaptureStatusNone;
}

Status persistent_grid(const void* kernel_fn, int32_t block_threads, size_t dynamic_smem_bytes,
                       const LaunchContext& ctx, int64_t work_items, LaunchConfig* out) {
  if (out == nullptr || kernel_fn == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "persistent_grid: null argument");
  }
  if (block_threads <= 0 || block_threads > 1024 || work_items < 0) {
    return detail::set_error(Status::kInvalidArgument, "persistent_grid: invalid block/work");
  }
  int active = 0;
  const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel_fn, block_threads, dynamic_smem_bytes);
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "persistent_grid occupancy: %s", cudaGetErrorString(err));
  }
  int sms = ctx.compute_units;
  if (sms <= 0) {
    int device = ctx.device >= 0 ? ctx.device : 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      return detail::set_error(Status::kCudaError, "persistent_grid: device query failed");
    }
    sms = prop.multiProcessorCount;
  }
  *out = LaunchConfig{};
  out->block = dim3(static_cast<unsigned>(block_threads));
  out->dynamic_smem_bytes = dynamic_smem_bytes;
  out->max_active_ctas = sms * active;
  const int64_t grid = std::min<int64_t>(work_items, out->max_active_ctas);
  out->grid = dim3(static_cast<unsigned>(grid));
  return Status::kSuccess;
}

Status launch_kernel(const void* kernel_fn, const LaunchConfig& cfg, void** args,
                     const LaunchContext& ctx) {
  if (kernel_fn == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "launch_kernel: null kernel");
  }
  if (ctx.backend == BackendKind::kHip) {
    return detail::set_error(Status::kUnsupportedBackend, "launch_kernel: HIP is not built");
  }
  const uint64_t threads = uint64_t(cfg.block.x) * cfg.block.y * cfg.block.z;
  if (threads == 0 || threads > 1024) {
    return detail::set_error(Status::kInvalidArgument, "launch_kernel: invalid block");
  }
  if (cfg.grid.x == 0 || cfg.grid.y == 0 || cfg.grid.z == 0) return Status::kSuccess;
  dim3 cluster = cfg.cluster;
  if (cluster.x == 0 || cluster.y == 0 || cluster.z == 0) cluster = dim3(1, 1, 1);
  const bool cluster_mode = cluster.x != 1 || cluster.y != 1 || cluster.z != 1;
  if ((cluster_mode || cfg.programmatic_serialization) && ctx.arch_id < 900) {
    return detail::set_error(Status::kUnsupportedArch,
                             "launch_kernel: cluster/PDL requires SM90+, got arch %d", ctx.arch_id);
  }
  if (cfg.grid.x % cluster.x != 0 || cfg.grid.y % cluster.y != 0 ||
      cfg.grid.z % cluster.z != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "launch_kernel: grid is not divisible by cluster");
  }
  cudaLaunchAttribute attrs[3]{};
  unsigned count = 0;
  if (cluster_mode) {
    attrs[count].id = cudaLaunchAttributeClusterDimension;
    attrs[count++].val.clusterDim = {cluster.x, cluster.y, cluster.z};
  }
  if (cfg.programmatic_serialization) {
    attrs[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[count++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (cfg.cooperative) {
    attrs[count].id = cudaLaunchAttributeCooperative;
    attrs[count++].val.cooperative = 1;
  }
  cudaLaunchConfig_t launch{};
  launch.gridDim = cfg.grid;
  launch.blockDim = cfg.block;
  launch.dynamicSmemBytes = cfg.dynamic_smem_bytes;
  launch.stream = cuda_stream(ctx.stream);
  launch.attrs = attrs;
  launch.numAttrs = count;
  const cudaError_t err = cudaLaunchKernelExC(&launch, kernel_fn, args);
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "launch_kernel: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

Status graph_begin_capture(kops_stream_t stream) {
  if (stream_is_capturing(stream)) {
    return detail::set_error(Status::kInvalidArgument, "graph_begin_capture: stream already capturing");
  }
  const cudaError_t err = cudaStreamBeginCapture(cuda_stream(stream), cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "graph_begin_capture: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

Status graph_end_capture(kops_stream_t stream, uint32_t capture_generation, GraphHandle* out) {
  if (out == nullptr || out->graph != nullptr || out->exec != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "graph_end_capture: output not empty");
  }
  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamEndCapture(cuda_stream(stream), &graph);
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "graph_end_capture: %s", cudaGetErrorString(err));
  }
  cudaGraphExec_t exec = nullptr;
  err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "graph instantiate: %s", cudaGetErrorString(err));
  }
  out->graph = reinterpret_cast<kops_graph_t>(graph);
  out->exec = reinterpret_cast<kops_graph_t>(exec);
  out->capture_generation = capture_generation;
  return Status::kSuccess;
}

Status graph_replay(const GraphHandle& g, const LaunchContext& ctx) {
  if (g.exec == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "graph_replay: empty graph");
  }
  const cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(g.exec),
                                          cuda_stream(ctx.stream));
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "graph_replay: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

Status graph_update_node(const GraphHandle& g, uint64_t node_id, void** args) {
  (void)g;
  (void)node_id;
  (void)args;
  return detail::set_error(Status::kNotImplemented, "graph_update_node: recapture instead");
}

Status graph_destroy(GraphHandle* g) {
  if (g == nullptr) return detail::set_error(Status::kInvalidArgument, "graph_destroy: null");
  if (g->exec != nullptr) {
    cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(g->exec));
    g->exec = nullptr;
  }
  if (g->graph != nullptr) {
    cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(g->graph));
    g->graph = nullptr;
  }
  g->capture_generation = 0;
  return Status::kSuccess;
}

}  // namespace kops::exec
