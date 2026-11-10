#pragma once
// =============================================================================
// kernel-ops | csrc/communication/ops.h                            Catalog: C01, C02
// Requires: common/*, gemm/gemm_params.cuh, communication/*.cuh.
// =============================================================================

namespace kops::comm {

/// Validate that a PeerBuffers / EPBuffers set matches the topology the kernels assume
/// (peer access enabled, multicast object bound, buffer sizes). Host, outside capture.
Status validate_peer_buffers(const PeerBuffers& peers, size_t required_bytes);

/// Zero this rank's signal area (flags[rank], kSignalWords words) and set the per-wait timeout
/// (0 = 20 s default). Host, synchronous, outside capture; call it on every rank before the first
/// collective and again after a CommError (the engine re-synchronises the ranks around it).
Status comm_configure(const PeerBuffers& peers, uint32_t timeout_us);
/// Read this rank's error word (CommError, 0 = none). Host, synchronous; call after the stream finished.
Status comm_error(const PeerBuffers& peers, int32_t* error_code);

/// Staging bytes (PeerBuffers::buffer_bytes) the collectives need per rank for a message.
/// 0 means the combination is unsupported (see the launcher for the reason).
size_t allreduce_staging_bytes(AllReduceAlgo algo, DType dtype, int64_t numel, int32_t world_size);
size_t allgather_staging_bytes(DType dtype, int64_t numel_per_rank, int32_t world_size);
size_t reduce_scatter_staging_bytes(DType dtype, int64_t numel, int32_t world_size);
size_t allreduce_fusion_staging_bytes(DType dtype, int32_t num_tokens, int32_t hidden, int32_t world_size);
Status validate_ep_buffers(const EPBuffers& buffers, int32_t hidden, int32_t max_tokens_per_rank,
                           int32_t num_experts);

/// Epoch-tagged barrier across the communicator (peer_barrier_kernel): every rank returns only
/// after every rank has entered. `epoch` of all entry points below is the communicator generation
/// (equal on all ranks, 0 for a fresh communicator); the per-call epoch advances on the device, so a
/// captured CUDA graph replays any number of times without a host update.
Status peer_barrier(const PeerBuffers& peers, uint32_t epoch, const LaunchContext& ctx);

Status allreduce(const Tensor& out, const Tensor& in, const PeerBuffers& peers, AllReduceAlgo algo,
                 uint32_t epoch, const LaunchContext& ctx);
Status allreduce_fusion(const AllReduceFusionParams& params, AllReduceAlgo algo, DType dtype,
                        const LaunchContext& ctx);
Status allgather(const Tensor& out, const Tensor& in, const PeerBuffers& peers, uint32_t epoch,
                 const LaunchContext& ctx);
Status reduce_scatter(const Tensor& out, const Tensor& in, const PeerBuffers& peers,
                      uint32_t epoch, const LaunchContext& ctx);

Status ep_dispatch_layout(const Tensor& num_tokens_per_rank, const Tensor& num_tokens_per_expert,
                          const Tensor& is_token_in_rank, const Tensor& topk_idx,
                          int32_t num_experts, int32_t num_ranks, const LaunchContext& ctx);
Status ep_dispatch(const EPDispatchParams& params, DType dtype, const LaunchContext& ctx);
Status ep_combine(const EPCombineParams& params, DType dtype, const LaunchContext& ctx);

/// Overlapped TP GEMMs (comm_gemm_kernels.cuh). Shapes in params.gemm are local shard shapes.
Status allgather_gemm(const CommGemmParams& params, gemm::GemmRecipe recipe, const LaunchContext& ctx);
Status gemm_reduce_scatter(const CommGemmParams& params, gemm::GemmRecipe recipe,
                           const LaunchContext& ctx);
Status gemm_allreduce(const CommGemmParams& params, gemm::GemmRecipe recipe, AllReduceAlgo algo,
                      const LaunchContext& ctx);

}  // namespace kops::comm
