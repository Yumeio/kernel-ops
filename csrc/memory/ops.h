#pragma once
// =============================================================================
// kernel-ops | csrc/memory/ops.h                                 Catalog: X04
// Optional memory services for engines that want them. The core library still
// never allocates inside run(); everything here happens in create/prepare or in
// an explicit sleep/offload phase, and every entry point says which.
//
// What this family covers (and why a kernel library needs it at all):
//   * VMM-backed pool          -> stable virtual addresses so captured graphs
//                                 survive offload/onload and KV growth
//   * sleep mode / offload     -> vLLM "sleep" and weight streaming
//   * IPC export/import        -> custom allreduce graph buffers, multi-process
//                                 tensor sharing (SGLang get_graph_buffer_ipc_meta)
//   * green context            -> CUDA SM partitioning for draft/comm kernels
//                                 (SGLang create_greenctx_stream_by_value)
//   * copy without the CE      -> SM-driven H2D/D2H when the copy engine is busy
//                                 (SGLang copy_to_gpu_no_ce)
//
// Requires: common/macros.cuh, common/status.h, common/tensor.h.
// =============================================================================

namespace kops::mem {

enum class MemKind : uint8_t {
  kDevice = 0,      // device local (VMM)
  kPinnedHost = 1,  // pinned, device accessible
  kUnified = 2,     // managed (only where the backend supports it)
  kPeer = 3,        // mapped from another device / process
};

/// Pool creation parameters. `granularity` is the backend allocation granularity
/// (VMM: cuMemGetAllocationGranularity / hipMemGetAllocationGranularity).
struct MemPoolDesc {
  BackendKind backend = BackendKind::kCuda;
  int32_t device = 0;
  uint64_t size_bytes = 0;
  uint64_t alignment = 256;
  uint64_t granularity = 0;      // 0 = query the backend
  MemKind kind = MemKind::kDevice;
  bool peer_accessible = false;  // map into other devices in the group
  bool shareable_ipc = false;    // exportable to another process
  bool multicast = false;        // NVLS multicast object (CUDA SM90+, ignored on HIP)
};

struct MemPoolStats {
  uint64_t reserved_bytes;   // virtual reservation
  uint64_t mapped_bytes;     // physically backed now
  uint64_t peak_mapped_bytes;
  uint64_t free_bytes;
  uint32_t num_blocks;
  uint32_t num_fragments;
};

using MemPool = void*;       // opaque, owned by the caller's handle

/// Create/destroy a pool. Allocation happens here (host, outside capture).
Status mem_pool_create(const MemPoolDesc& desc, MemPool* out);
Status mem_pool_destroy(MemPool pool);
Status mem_pool_stats(MemPool pool, MemPoolStats* out);
/// Reserve [offset, offset+bytes) inside the pool. The pool keeps one stable base
/// pointer, so pointers derived from it stay valid across offload/onload.
Status mem_pool_alloc(MemPool pool, uint64_t bytes, uint64_t* offset);
Status mem_pool_free(MemPool pool, uint64_t offset, uint64_t bytes);
/// Base device pointer of the reservation (stable for the pool's lifetime).
void* mem_pool_base(MemPool pool);

// ---- IPC ---------------------------------------------------------------------
/// Export/import a pool (or a single allocation) to another process. The handle is
/// backend-opaque bytes; both sides must use the same ABI version.
Status mem_pool_export_ipc(MemPool pool, void* handle_out, size_t* handle_bytes);
Status mem_pool_import_ipc(BackendKind backend, int32_t device, const void* handle,
                           size_t handle_bytes, MemPool* out);
Status mem_handle_close(void* handle);

// ---- Sleep mode / offload ----------------------------------------------------
/// Move the *contents* of resident allocations to host and back, without
/// unmapping the virtual addresses. Captured graphs and plan pointers stay valid.
/// `discard_on_offload` frees the host copy after the upload (weight re-read path).
struct OffloadDesc {
  void* const* device_ptrs;      // [num_regions] stable device addresses
  const uint64_t* bytes;         // [num_regions]
  int32_t num_regions;
  void* host_staging;            // pinned staging buffer
  uint64_t host_staging_bytes;
  bool discard_on_offload;
};

Status offload_workspace_requirements(const OffloadDesc& desc, WorkspaceRequirements* out);
/// Enqueue D2H (offload) or H2D (onload) on ctx.stream. Both are capture-safe when
/// the staging buffer is pinned and preallocated; never allocate here.
Status offload_run(const OffloadDesc& desc, const LaunchContext& ctx);
Status onload_run(const OffloadDesc& desc, const LaunchContext& ctx);

// ---- Green context / SM partitioning ----------------------------------------
/// Reserve a contiguous slice of CUs (CUDA green context; HIP: kUnsupportedBackend
/// unless the runtime exposes an equivalent partition API).
struct GreenCtxDesc {
  int32_t device = 0;
  int32_t min_compute_units = 1;
  int32_t max_compute_units = 0;   // 0 = as many as available
};
using GreenCtx = void*;

Status greenctx_create(const GreenCtxDesc& desc, GreenCtx* out);
/// Stream bound to the reserved slice, for the draft model / comm kernel.
Status greenctx_stream(GreenCtx ctx, kops_stream_t* out_stream);
Status greenctx_destroy(GreenCtx ctx);
/// Number of CUs actually reserved (may be less than requested).
Status greenctx_compute_units(GreenCtx ctx, int32_t* out);

// ---- Copies ------------------------------------------------------------------
/// SM-driven copy that does not use the copy engine. Required when a captured graph
/// already saturates the CE (weight prefetch, KV offload) or when the build has no
/// batch-copy API. Alignment: 16 B preferred, 4 B minimum; falls back to a byte loop.
Status copy_no_ce(void* dst, const void* src, uint64_t bytes, const LaunchContext& ctx);

}  // namespace kops::mem
