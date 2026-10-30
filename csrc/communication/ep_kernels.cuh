#pragma once
// =============================================================================
// kernel-ops | csrc/communication/ep_kernels.cuh                       Catalog: C02
// Expert-parallel dispatch / combine (DeepEP-style), normal (high-throughput,
// prefill) and low-latency (decode, masked layout, hook-mode overlap).
// Transport (NVLink peer memory, NVSHMEM/IBGDA RDMA) is described by EPBuffers;
// session setup and rank lifecycle stay in the engine.
// Requires: common/*, communication/allreduce_kernels.cuh (kMaxRanks).
// =============================================================================

namespace kops::comm {

enum class EPTransport : uint8_t { kNvlink = 0, kNvshmemRdma = 1, kHybrid = 2 };

/// Symmetric buffers of an EP group.
struct EPBuffers {
  void* nvl_buffers[kMaxRanks];     // intra-node peer buffers
  void* rdma_buffer;                // symmetric (NVSHMEM) buffer, same offset on all ranks
  uint32_t* signals;                // epoch-tagged receive counters
  int32_t rank, num_ranks, num_local_ranks;
  EPTransport transport;
  size_t nvl_bytes, rdma_bytes;
};

/**
 * ep_dispatch_layout_kernel — from topk_idx compute per-rank / per-expert token counts and the
 * token->rank membership bitmap (input to dispatch, and to allocate receive buffers).
 */
__global__ void ep_dispatch_layout_kernel(
    int32_t* __restrict__ num_tokens_per_rank,    // [num_ranks]
    int32_t* __restrict__ num_tokens_per_expert,  // [num_experts]
    uint8_t* __restrict__ is_token_in_rank,       // [num_tokens, num_ranks]
    const int32_t* __restrict__ topk_idx,         // [num_tokens, topk] (-1 = dropped)
    const int num_tokens, const int topk, const int num_experts, const int num_ranks);

/// Arguments of EP dispatch.
struct EPDispatchParams {
  const void* x;                    // [num_tokens, hidden] (bf16) — quantized on the fly if quant != kNone
  const int32_t* topk_idx;          // [num_tokens, topk]
  const float* topk_weights;        // [num_tokens, topk]
  const uint8_t* is_token_in_rank;  // normal mode
  const int32_t* num_tokens_per_rank;
  ActQuantMode payload_quant;       // kNone | kPerGroup (FP8 128) | kMXFP8 | kNVFP4
  EPBuffers buffers;
  // Receive side
  void* recv_x;                     // normal: [num_recv_tokens, hidden]; LL: [E_local, max_tokens_per_expert, hidden]
  void* recv_scales;                // payload scales
  int32_t* recv_topk_idx;           // normal mode: local expert ids of received tokens
  float* recv_topk_weights;
  int32_t* recv_count_per_expert;   // [E_local] (LL masked_m)
  int64_t* recv_src_info;           // [..] (source rank, source token) for combine
  int32_t num_tokens, hidden, topk, num_experts, max_tokens_per_rank;
  uint32_t epoch;
  bool low_latency;                 // masked layout, per-expert slots, no layout pass
  bool hook_mode;                   // split send/recv phases for compute overlap (LL)
  int32_t phase;                    // hook mode: 0 = send, 1 = receive
};

template <typename T, int NUM_RANKS_PER_NODE>
__global__ void ep_dispatch_kernel(const KOPS_GRID_CONST EPDispatchParams params);

/// Arguments of EP combine (weighted reduction back at the source rank).
struct EPCombineParams {
  const void* expert_out;           // same layout as recv_x (bf16)
  const float* topk_weights;        // weights applied during combine (or already applied)
  bool weights_already_applied;
  const int64_t* src_info;          // from dispatch
  EPBuffers buffers;
  void* out;                        // [num_tokens, hidden]
  const void* residual;             // optional fused residual add
  int32_t num_tokens, hidden, topk;
  uint32_t epoch;
  bool low_latency, hook_mode;
  int32_t phase;
};

template <typename T, int NUM_RANKS_PER_NODE>
__global__ void ep_combine_kernel(const KOPS_GRID_CONST EPCombineParams params);

}  // namespace kops::comm
