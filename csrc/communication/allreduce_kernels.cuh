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

// ---- Signal area and scratch (ABI 6, B12) ---------------------------------------------------------
// Every rank owns three kinds of memory, all passed in PeerBuffers and all caller-allocated:
//   data[r]     staging buffer, written only by rank r, read by the peers (device memory, a P2P mapping
//               or, without P2P, pinned *write-combined* host memory: SM stores to write-back host memory
//               run at half the PCIe rate);
//   flags[r]    kSignalWords words of arrival flags, written only by rank r (device memory or pinned host
//               memory, 16 B aligned, zeroed by comm_configure);
//   scratch     kScratchWords words of rank-local *device* memory (never shared): error word, timeout,
//               call counter and the CTA arrival counters. It is what makes a call independent of how many
//               CTAs run and what keeps the epoch on the device.
// The pull protocol of every collective: stage, publish one flag with a system-scope release, wait for
// the peers' flags with system-scope acquire, read their staging. A system fence costs about 1.5 us and
// the GPU serialises them across CTAs, so each call posts exactly one flag per phase: the last CTA to
// finish staging (device-scope arrival counter in scratch) posts it. The epoch of a call is
// `epoch + calls + 1`; the last CTA to finish advances `calls`, so a captured graph replays any number of
// times with no host update. The staging slot of a call is its epoch & 1.
inline constexpr int kCommCtas = 16;      // upper bound on CTAs of one call (host picks 1..16 by size)
inline constexpr int kCommThreads = 256;
inline constexpr int kSignalWords = 256;
inline constexpr int kSigReady1 = 0;      // phase-1 arrival flag (value = call epoch); own 128 B line
inline constexpr int kSigReady2 = 32;     // phase-2 arrival flag; own 128 B line
inline constexpr int kScratchWords = 64;
inline constexpr int kScrError = 0;       // CommError of the last failed call on this rank
inline constexpr int kScrTimeoutUs = 1;   // per-wait timeout; 0 selects the 20 s default
inline constexpr int kScrCalls = 2;       // completed calls (epoch base)
inline constexpr int kScrArrive1 = 3;     // CTAs that finished phase-1 staging
inline constexpr int kScrArrive2 = 4;     // CTAs that finished phase-2 staging
inline constexpr int kScrDone = 5;        // CTAs that finished the call

/// Value stored in the error word (kScrError) by the kernel that gave up.
enum class CommError : int32_t {
  kNone = 0,
  kPeerTimeout = 1,    // a peer flag did not reach the call epoch within the timeout
  kDataTimeout = 2,    // Lamport packets did not arrive within the timeout
  kBadArgument = 3,
};

/// Staging layout and capacity: two slots (call parity) of `slot_bytes` each. The kernels need
/// PeerBuffers::buffer_bytes >= 2 * slot_bytes; see the *_staging_bytes() helpers in ops.h.
inline constexpr size_t kStagingAlign = 128;

/// Peer-mapped buffers of one communicator (prepared by the engine's handle exchange).
/// kMaxRanks is the maximum ranks *in one group* (intra-node NVLink domain or one
/// multi-node group). Multi-node groups also set num_nodes/node_rank; the transport
/// (NVLink P2P, NVSHMEM/IBGDA RDMA, mscclpp) is described by EPBuffers/RankGroup, not here.
struct PeerBuffers {
  void* data[kMaxRanks];           // each rank's staging buffer, mapped into this process
  uint32_t* flags[kMaxRanks];      // per-rank signal arrays (epoch-tagged)
  void* multicast_ptr;             // NVLS multicast address (SM90+ with NVSwitch) or null
  int32_t rank;
  int32_t world_size;
  int32_t num_nodes;               // 1 = intra-node only
  int32_t node_rank;
  size_t buffer_bytes;
  uint32_t* scratch;               // this rank's private device scratch (kScratchWords words, zeroed by
                                   //   comm_configure); see the protocol note above
};

enum class AllReduceAlgo : uint8_t {
  kOneShot = 0,        // every rank reads all peers (small messages)
  kTwoShot = 1,        // reduce-scatter + all-gather (medium)
  kLamportOneShot = 2, // flag-free Lamport (-0.0 sentinel) triple buffering (TRT-LLM style, lowest latency)
                       //   Requires a floating-point T: the -0.0 sentinel is not representable for integers.
  kMultimem = 3,       // multimem.ld_reduce / st on NVSwitch (H100/B200 NVL); CUDA only
  kQuickReduce = 4,    // block-scaled quantized reduction (FP8/INT8 payload + fp32 block scale)
  kDeterministic = 5,  // fixed reduction order regardless of arrival order (batch-invariant rollouts)
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
