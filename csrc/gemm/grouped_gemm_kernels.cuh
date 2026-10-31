#pragma once
// =============================================================================
// kernel-ops | csrc/gemm/grouped_gemm_kernels.cuh                    Catalog: G06
// Grouped / segmented / masked / indexed GEMM for MoE experts and multi-LoRA.
// One launch serves all groups; empty groups produce no tiles; tail tiles are
// masked (no out-of-bounds reads). Every precision recipe of the dense family
// exists here per architecture:
//   grouped_gemm_mma_sync_kernel   Sm80/86/89/120  cp.async (+ FP8 on Sm89)  — small M per expert, HPC-Ops cp_async path
//   grouped_gemm_wgmma_kernel      Sm90a           BF16, FP8 tensor/row/blockwise(+UE8M0), INT8 — DeepGEMM / CUTLASS ptr-array
//   grouped_gemm_tcgen05_kernel    Sm100a/103a     BF16, FP8, MXFP8, NVFP4, MXFP4, MXFP4xMXFP8 — 1-SM / 2-SM
//   grouped_gemm_sm120_kernel      Sm120a/121a     BF16, FP8, MXFP8, NVFP4, MXFP4, MXFP4xMXFP8 (block-scaled mma.sync)
//   grouped_mixed_input_gemm_kernel Sm90a/Sm100a   W4A16 int4 / MXFP4 / NVFP4, W4A8 (int4 or FP4 x FP8)
//   grouped_fp4_to_fp8_gemm_wgmma_kernel Sm90a     FP4 experts on Hopper (DeepSeek-V4 / GPT-OSS / Kimi-K3 on H100/H200/H20)
//   grouped_gemv_kernel            any             decode: tiny M per expert, any weight format
// MoE Marlin (W4A16 with sorted_token_ids) is MarlinParams' MoE mode (mixed_input_gemm_kernels.cuh).
// Requires: common/*, gemm/gemm_params.cuh.
// =============================================================================

namespace kops::gemm {

/// How rows of A are assigned to groups (experts).
enum class GroupedLayout : uint8_t {
  kContiguous = 0,          // A rows sorted by expert; group g = rows [offsets[g], offsets[g+1]); segments padded
                            //   to the tile M when the kernel requires it (DeepGEMM contiguous: 128)
  kContiguousRowIndex = 1,  // sorted A plus m_indices[row] = expert (or -1 for padding rows) instead of offsets
  kMasked = 2,              // A is [G, max_m, K]; group g uses its first masked_m[g] rows (DeepEP low-latency)
  kIndexed = 3,             // rows gathered on the fly: A_row = A[gather_idx[r]] (no permute copy; HPC-Ops indexed)
  kPtrArray = 4,            // independent problems with per-group pointers/shapes (LoRA, heterogeneous experts)
};

/// Device-resident per-group problem (ptr-array form; built by grouped_gemm_prepare_kernel).
struct GroupedProblem {
  int32_t m, n, k;
  int32_t tile_offset;       // first linear tile id of this group in the persistent schedule
  const void* a;             // group's A rows
  const void* b;             // group's weight
  void* d;
  const void* sfa;           // group's A scales (layout per recipe)
  const void* sfb;           // group's B scales
  float alpha;               // per-group dequant product: FP8 tensorwise sA*sB_g, NVFP4 1/(gA*gB_g)
  int64_t lda, ldb, ldd;
};

/// Arguments of every grouped kernel.
struct GroupedGemmParams {
  GroupedLayout layout;
  GemmRecipe recipe;
  int32_t num_groups;                 // local experts (or LoRA problems)
  // A (activations)
  const void* a;                      // [total_rows, K] | [G, max_m, K] (masked) | gather source [T, K] (indexed)
  int64_t lda;
  const void* a_scale;                // per-token / 1x128 (col-major TMA) / MX / NVFP4 scales of A
  ScaleLayout a_scale_layout;
  const float* a_global_scale;        // NVFP4 activation global scale ([1] or [G])
  const int64_t* expert_offsets;      // kContiguous: [G + 1]
  const int32_t* m_indices;           // kContiguousRowIndex: [total_rows] expert per row, -1 = padding
  const int32_t* masked_m;            // kMasked: [G] valid rows per group
  int32_t max_m;                      // kMasked: rows reserved per group
  int32_t expected_m;                 // kMasked: typical rows per group (tactic hint; not a bound)
  const int32_t* gather_idx;          // kIndexed: [total_rows] source row in A (token id) per routed row
  const void* const* a_ptrs;          // kPtrArray: [G]
  // B (expert weights), D
  const void* b;                      // [G, N, K] (packed for fp4/int4; backend layouts per b_packing)
  int64_t ldb;
  int64_t b_group_stride;             // elements (bytes for packed) between experts
  WeightPacking b_packing;
  const void* b_scale;                // [G, ...] per-expert scales (per-tensor [G], per-channel [G, N],
                                      //   128x128 [G, N/128, K/128], MX/NVFP4 swizzled per expert)
  int64_t b_scale_group_stride;
  const float* b_global_scale;        // NVFP4 per-expert global scales [G]
  const void* b_zeros;                // int4 group zeros (W4A16 / W4A8) or null
  const void* const* b_ptrs;          // kPtrArray
  void* d;                            // [total_rows, N] | [G, max_m, N] | scattered (see epilogue)
  int64_t ldd;
  void* const* d_ptrs;                // kPtrArray
  const int32_t* problem_m;           // kPtrArray: [G] rows per problem
  // Shapes / epilogue
  int32_t n, k;
  EpilogueParams epi;                 // per-expert bias (bias_group_stride), GLU (+ glu_interleave), routing
                                      // weight, quant-out for the next stage, scatter + atomic finalize
  // Prepared device metadata (grouped_gemm_prepare_kernel)
  const GroupedProblem* problems;     // [G] or null (computed on the fly from the layout)
  const KOPS_TMA_DESC* group_tma;       // [G * 4] per-group A/B/SFA/SFB tensor maps (Sm90a/Sm100a) or null
  const int32_t* tile_task_map;       // [num_tasks, 3] (group, m_tile, n_tile) or null
  int32_t* tile_counter;              // dynamic persistent scheduling (epoch-tagged)
  uint32_t epoch;
  bool swap_ab;                       // D^T = B . A^T per group (decode: tiny M per expert)
};

/**
 * grouped_gemm_prepare_kernel — device-side argument preparation (graph-safe, no host sync):
 * from expert_offsets / m_indices / masked_m computes GroupedProblem[g] (shapes, pointers, per-group
 * alpha from per-expert scales), patches per-group tensor maps from templates (tensormap.replace on
 * Sm90a/Sm100a), and builds the persistent tile list. Groups with m == 0 get no tiles.
 */
__global__ void grouped_gemm_prepare_kernel(
    GroupedProblem* __restrict__ problems,         // [G]
    KOPS_TMA_DESC* __restrict__ group_tma,           // [G * 4] or null
    const KOPS_TMA_DESC* __restrict__ tma_templates, // [4] A/B/SFA/SFB templates or null
    int32_t* __restrict__ tile_task_map,           // [capacity, 3]
    int32_t* __restrict__ num_tasks,               // [1]
    int32_t* __restrict__ overflow_flag,           // [1]
    const GroupedGemmParams params,
    const int tile_m, const int tile_n, const int task_capacity);

/// grouped_gemm_task_map_kernel — task list only (when per-group pointers are derivable in-kernel).
__global__ void grouped_gemm_task_map_kernel(
    int32_t* __restrict__ task_map,          // [capacity, 3]
    int32_t* __restrict__ num_tasks,         // [1]
    int32_t* __restrict__ overflow_flag,     // [1]
    const int64_t* __restrict__ expert_offsets,  // [G + 1] or null
    const int32_t* __restrict__ masked_m,    // [G] or null
    const int num_groups, const int tile_m, const int n_tiles, const int capacity);

/**
 * grouped_gemm_mma_sync_kernel — cp.async + mma.sync grouped GEMM (Sm80/86/89/120). Occupancy-
 * oriented small tiles (M 16..64) for decode-sized expert batches; kIndexed gathers A rows directly
 * from token ids (HPC-Ops cp_async / indexed path). Recipes: kBF16, kFP16, kFP8Tensorwise,
 * kFP8Rowwise, kFP8Blockwise (Sm89 / Sm120), kINT8Rowwise.
 */
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout>
__global__ void grouped_gemm_mma_sync_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * grouped_gemm_wgmma_kernel — Hopper persistent grouped GEMM (TMA + WGMMA, warp-specialized).
 * Recipes: kBF16, kFP8Tensorwise (per-expert alpha), kFP8Rowwise, kFP8Blockwise / kFP8BlockwiseUE8M0
 * (DeepGEMM m-grouped contiguous & masked semantics, Fp8Accum::kPromoted), kINT8Rowwise.
 * Models : DeepSeek-V3/V3.2 & Kimi K2 FP8 experts, Qwen3-MoE FP8, GLM-4.5 FP8, Mixtral/Qwen BF16.
 */
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout, Fp8Accum kAccum>
__global__ void grouped_gemm_wgmma_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * grouped_gemm_tcgen05_kernel — Blackwell datacenter grouped GEMM. Recipes: kBF16, kFP8Tensorwise,
 * kFP8Rowwise, kFP8BlockwiseUE8M0, kMXFP8, kNVFP4, kMXFP4, kMXFP4xMXFP8 (GPT-OSS / Kimi-K3 W4A8-MX
 * experts), with per-expert block-scale tensor maps and per-expert NVFP4 global scales folded into
 * alpha. 1-SM / 2-SM; weight tile multicast in 2-SM mode.
 * Models : DeepSeek-V4 FP4 experts, GPT-OSS MXFP4, Kimi K3 MXFP4, NVFP4 MoE checkpoints, FP8 MoEs.
 */
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout>
__global__ void grouped_gemm_tcgen05_kernel(const __grid_constant__ GroupedGemmParams params);

/// grouped_gemm_sm120_kernel — RTX Blackwell grouped GEMM (block-scaled mma.sync for FP4/MX recipes).
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout>
__global__ void grouped_gemm_sm120_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * grouped_mixed_input_gemm_kernel — MoE weight-only / mixed-input on tensor cores (TRT-LLM / CUTLASS
 * mixed grouped style): weights converted in smem/registers. Recipes: kW4A16Int (int4 group + zeros),
 * kW4A16MXFP4 (GPT-OSS MXFP4 x BF16), kW4A16NVFP4, kW4A8IntFP8, kW4A8FP4xFP8. ArchTag Sm90a or Sm100a.
 */
template <typename Traits, GemmRecipe kRecipe, GroupedLayout kLayout>
__global__ void grouped_mixed_input_gemm_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * grouped_fp4_to_fp8_gemm_wgmma_kernel — Hopper grouped GEMM for FP4 experts: per-expert MXFP4/NVFP4
 * tiles converted to e4m3 in smem (block scales folded into promotion factors), FP8 activations,
 * wgmma with promoted accumulation.
 */
template <typename Traits, GroupedLayout kLayout>
__global__ void grouped_fp4_to_fp8_gemm_wgmma_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * grouped_gemv_kernel — decode MoE where each expert sees only a few rows: one CTA per
 * (expert, N tile) streams that expert's weight once for all its rows (rows located via
 * expert_offsets / masked_m / m_indices). Weight formats: bf16, fp8 (all FP8 recipes), MXFP4/NVFP4,
 * int4 groupwise. The weight-bandwidth-optimal path when tokens * topk is small.
 */
template <typename a_t, GemmRecipe kRecipe, GroupedLayout kLayout, int ROWS_PER_WARP, int WARPS>
__global__ void grouped_gemv_kernel(const __grid_constant__ GroupedGemmParams params);

/**
 * Gated W13 and finalize fusions (epilogue configuration of the kernels above, no extra kernels):
 *   W13 : epi.ops |= kEpiGLU with GateLayout::kInterleaved and epi.glu_interleave = TILE_N / 2 after
 *         gemm_prep_kernels.cuh::glu_weight_interleave_kernel -> writes [rows, I] (+ kEpiQuantOut for
 *         the W2 activation: FP8 1x128, MXFP8, NVFP4) — removes the act_and_mul launch.
 *   W2  : epi.ops |= kEpiRoutingWeight | kEpiScatterRows (| kEpiAtomicAdd) writes the weighted expert
 *         output straight into [T, H] (TRT-LLM "finalize fusion"). With ctx.deterministic the atomic
 *         form is rejected and moe::moe_unpermute_reduce_kernel is used instead.
 */

// -----------------------------------------------------------------------------
// Multi-LoRA (Punica SGMV / BGMV semantics)
// -----------------------------------------------------------------------------

struct LoraSgmvParams {
  const void* x;                // [T, in_dim]
  int64_t ldx;
  const void* const* lora_a;    // per adapter slot: [rank, in_dim]  (shrink) or [out_dim, rank] (expand)
  void* y;                      // shrink: [T, rank] fp32 buffer; expand: [T, out_dim] accumulate
  int64_t ldy;
  const int32_t* seg_indptr;    // [num_segments + 1] token ranges sharing one adapter
  const int32_t* seg_lora_ids;  // [num_segments] adapter slot, -1 = no LoRA
  int32_t num_segments;
  int32_t in_dim, out_dim, rank;
  int32_t out_slice_offset;     // column offset inside a fused qkv / gate_up output
  float scaling;                // alpha / rank
};

/**
 * lora_shrink_kernel  : y[t, :rank]  = x[t] . A[id]^T                (fp32 accumulate)
 * lora_expand_kernel  : out[t, off:off+out] += scaling * y[t] . B[id]^T
 * Serving-time multi-LoRA for every dense architecture; MoE LoRA uses kPtrArray grouped GEMM.
 */
template <typename scalar_t, int TILE_T, int TILE_R>
__global__ void lora_shrink_kernel(const __grid_constant__ LoraSgmvParams params);

template <typename scalar_t, int TILE_T, int TILE_N>
__global__ void lora_expand_kernel(const __grid_constant__ LoraSgmvParams params);

}  // namespace kops::gemm
