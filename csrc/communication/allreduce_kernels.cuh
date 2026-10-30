#pragma once
// =============================================================================
// kernel-ops | csrc/communication/allreduce_kernels.cuh                Catalog: C01
// Intra-node collectives over peer-mapped (IPC / symmetric) buffers, with fused
// residual + RMSNorm (+ quantization). NCCL is the baseline; these are opt-in per
// topology (NVLink P2P, NVSwitch multicast). Process groups, handle exchange,
// timeouts and recovery are the engine's job.
// Requires: common/*.
// =============================================================================

namespace kops::comm {

inline constexpr int kMaxRanks = 16;

/// Peer-mapped buffers of one communicator (prepared by the engine's handle exchange).
struct PeerBuffers {
  void* data[kMaxRanks];           // each rank's staging buffer, mapped into this process
  uint32_t* flags[kMaxRanks];      // per-rank signal arrays (epoch-tagged)
  void* multicast_ptr;             // NVLS multicast address (SM90+ with NVSwitch) or null
  int32_t rank;
  int32_t world_size;
  size_t buffer_bytes;
};

enum class AllReduceAlgo : uint8_t {
  kOneShot = 0,        // every rank reads all peers (small messages)
  kTwoShot = 1,        // reduce-scatter + all-gather (medium)
  kLamportOneShot = 2, // flag-free Lamport (-0.0 sentinel) triple buffering (TRT-LLM style, lowest latency)
  kMultimem = 3,       // multimem.ld_reduce / st on NVSwitch (H100/B200 NVL)
};

/**
 * allreduce_kernel — out = sum_r in_r over the communicator. epoch increments every call so
 * signal arrays never need resetting (graph-replay safe). Launch needs all ranks to launch the
 * same kernel with the same epoch; co-residency of CTAs across ranks is NOT assumed.
 */
template <typename T, int WORLD_SIZE, AllReduceAlgo kAlgo>
__global__ void allreduce_kernel(
    T* __restrict__ out,                     // [numel]
    const T* __restrict__ in,                // [numel] (copied into the staging buffer by the kernel)
    const PeerBuffers peers,
    const int64_t numel, const uint32_t epoch);

/// Arguments of allreduce_fusion_kernel (C01).
struct AllReduceFusionParams {
  const void* input;               // [num_tokens, hidden] partial sums (row-parallel output)
  void* residual_inout;            // [num_tokens, hidden] residual stream (updated)
  const void* norm_weight;         // [hidden]
  float norm_eps;
  float weight_offset;
  void* norm_out;                  // [num_tokens, hidden] normalized output (bf16) or null
  ActQuantOut quant;               // optional FP8 / NVFP4 / MXFP8 quantized normalized output
  PeerBuffers peers;
  uint32_t epoch;
  int32_t num_tokens, hidden;
};

/**
 * allreduce_fusion_kernel — AR(input) + residual -> RMSNorm -> (quant), one kernel per TP layer
 * boundary (HPC-Ops fused allreduce-RMSNorm, TRT-LLM allreduce fusion). Supports PDL.
 */
template <typename T, int WORLD_SIZE, AllReduceAlgo kAlgo, ActQuantMode kQuant>
__global__ void allreduce_fusion_kernel(const __grid_constant__ AllReduceFusionParams params);

/// all-gather along dim 0 (vocab-parallel logits, sequence parallel, AttnRes SP sharding).
template <typename T, int WORLD_SIZE>
__global__ void allgather_kernel(
    T* __restrict__ out,                     // [world_size * numel_per_rank]
    const T* __restrict__ in,                // [numel_per_rank]
    const PeerBuffers peers, const int64_t numel_per_rank, const uint32_t epoch);

/// reduce-scatter along dim 0 (sequence-parallel residual streams).
template <typename T, int WORLD_SIZE>
__global__ void reduce_scatter_kernel(
    T* __restrict__ out,                     // [numel / world_size]
    const T* __restrict__ in,                // [numel]
    const PeerBuffers peers, const int64_t numel, const uint32_t epoch);

/// Device-side barrier across ranks (epoch-tagged flags).
template <int WORLD_SIZE>
__global__ void peer_barrier_kernel(const PeerBuffers peers, const uint32_t epoch);

}  // namespace kops::comm
