#pragma once
// =============================================================================
// kernel-ops | csrc/communication/comm_gemm_kernels.cuh            Catalog: C01 x G01-G05
// Tensor-parallel GEMMs overlapped with intra-node communication (Flux /
// async-TP / TRT-LLM GEMM+AllReduce style). Tiles start as soon as the data they
// need has arrived (AG+GEMM) or push their output to peers as soon as they finish
// (GEMM+RS, GEMM+AR). Any GemmRecipe of the dense family (BF16, FP8, FP4) can be
// combined; the communication protocol is epoch-tagged and graph-replay safe.
// Requires: common/*, gemm/gemm_params.cuh, communication/allreduce_kernels.cuh.
// =============================================================================

namespace kops::comm {

/// Arguments of the overlapped GEMM kernels.
struct CommGemmParams {
  gemm::GemmTmaParams gemm;         // the local GEMM (shapes are the LOCAL shard shapes)
  PeerBuffers peers;                // peer-mapped staging buffers + signal arrays, multicast address
  int32_t num_chunks;               // communication granularity along M (rows per chunk = M / num_chunks)
  uint32_t* chunk_ready;            // [world_size * num_chunks] arrival flags (epoch-tagged)
  uint32_t epoch;
  bool fuse_residual_rmsnorm;       // GEMM+AR only: residual add + RMSNorm after the reduction
  const void* residual;             // [M, N] in/out when fused
  const void* norm_weight;          // [N]
  float norm_eps;
  ActQuantOut norm_quant;           // optional quantized normalized output
};

/**
 * allgather_gemm_kernel — sequence-parallel input: A is sharded along M across ranks; each rank
 * pulls remote A chunks (copy-engine or SM-driven) while computing tiles of chunks already present.
 * Output [M_total, N_local] (column-parallel QKV / gate_up under SP).
 */
template <typename Traits, gemm::GemmRecipe kRecipe, int WORLD_SIZE>
__global__ void allgather_gemm_kernel(const KOPS_GRID_CONST CommGemmParams params);

/**
 * gemm_reduce_scatter_kernel — row-parallel output (o_proj / down_proj under SP): each finished
 * tile is reduced into the owner rank's M slice through peer stores (or multimem.st on NVLS).
 * Output [M_total / world_size, N].
 */
template <typename Traits, gemm::GemmRecipe kRecipe, int WORLD_SIZE>
__global__ void gemm_reduce_scatter_kernel(const KOPS_GRID_CONST CommGemmParams params);

/**
 * gemm_allreduce_kernel — row-parallel output without SP: tiles are reduced across ranks in the
 * epilogue (one-shot peer reads, or multimem.ld_reduce on NVSwitch systems), optionally followed by
 * residual + RMSNorm (+ quant) for the next layer's input. Decode-latency oriented.
 */
template <typename Traits, gemm::GemmRecipe kRecipe, int WORLD_SIZE, AllReduceAlgo kAlgo>
__global__ void gemm_allreduce_kernel(const KOPS_GRID_CONST CommGemmParams params);

}  // namespace kops::comm
