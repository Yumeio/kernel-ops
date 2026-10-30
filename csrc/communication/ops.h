#pragma once
// =============================================================================
// kernel-ops | csrc/communication/ops.h                            Catalog: C01, C02
// Requires: common/*, gemm/gemm_params.cuh, communication/*.cuh.
// =============================================================================

namespace kops::comm {

/// Validate that a PeerBuffers / EPBuffers set matches the topology the kernels assume
/// (peer access enabled, multicast object bound, buffer sizes). Host, outside capture.
Status validate_peer_buffers(const PeerBuffers& peers, size_t required_bytes);
Status validate_ep_buffers(const EPBuffers& buffers, int32_t hidden, int32_t max_tokens_per_rank,
                           int32_t num_experts);

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
