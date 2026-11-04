#pragma once
// =============================================================================
// kernel-ops | csrc/sampling/fused_sampler_kernels.cuh            Catalog: S01/S03
// Fused sampler device contract: greedy + unfiltered categorical draw without
// materializing a [B, V] fp32 probability row. Filtered top-k/top-p paths are
// added by later milestones behind the same host plan.
// Requires: common/macros.cuh, common/dtype.cuh, common/rng.cuh,
//           sampling/sampling_kernels.cuh.
// =============================================================================

namespace kops::sample {

/// Optional processor inputs fused into the sampler's logits load. Every pointer is nullable;
/// null means "disabled for this call". Arrays are device-resident. Per-request state
/// (penalty_mask, token_counts) is indexed by slot_id[b], or by the batch row b when slot_id
/// is null; per-row scalars are indexed by b. Applied in this order, before temperature:
///   repetition (bit set in penalty_mask): x > 0 ? x / rp : x * rp
///   frequency / presence (token_counts):  x -= freq * count + pres * (count > 0)
struct FusedSamplingParams {
  SamplingParams sampling;              // per-row temperature/top-k/top-p/min-p + RNG
  uint8_t* penalty_mask;                // [slots, ceil(V/8)] tokens seen so far (bit v&7 of byte v>>3), or null
  const int32_t* slot_id;               // [B] state row of batch row b, or null (row b)
  const float* repetition_penalty;      // [B] per-row repetition penalty, or null
  float default_repetition_penalty;     // used when repetition_penalty == null (1 = off)
  int32_t update_penalty_mask;          // nonzero: set the sampled token's bit after the draw
  const int32_t* token_counts;          // [slots, V] output-token counts, or null
  const float* frequency_penalty;       // [B] or null (0)
  const float* presence_penalty;        // [B] or null (0)
};

/// Split-row launch state (Path B). `splits == 1` is the one-CTA-per-row path and ignores the
/// pointers. With `splits > 1` (a power of two <= 16) the grid is (splits, batch): CTA x of row y
/// owns the x-th contiguous shard of the row, the CTAs of a row exchange partial results through
/// `scratch` and meet at barriers on `barriers[y]`. The launch is cooperative so every CTA is
/// resident; results are bit-identical to splits == 1.
struct FusedSplitDevice {
  int32_t splits;
  int32_t reserved;
  uint8_t* scratch;                 // [batch] per-row exchange areas (workspace scratch)
  unsigned long long* barriers;     // [batch] counters, zero-filled once (workspace persistent)
};

/// Every kernel takes SPLIT = (split.splits > 1) as a template flag so the one-CTA-per-row code
/// carries none of the exchange logic.
/// Greedy path: one pass, one CTA per row. Temperature is ignored (the host
/// routes non-positive/non-finite rows here). Ties resolve to the lowest id.
template <typename in_t, bool SPLIT>
__global__ void fused_greedy_kernel(int32_t* __restrict__ out_ids,
                                    const in_t* __restrict__ logits, int64_t logits_stride,
                                    FusedSamplingParams params, int batch_size, int vocab_size,
                                    int penalty_row_bytes, FusedSplitDevice split);

/// Unfiltered categorical path: two passes (row max, then fixed-point inverse
/// CDF) with one uniform draw per row. Per-row temperature <= 0 falls back to
/// the greedy branch inside the same kernel.
template <typename in_t, bool SPLIT>
__global__ void fused_categorical_kernel(int32_t* __restrict__ out_ids,
                                         const in_t* __restrict__ logits, int64_t logits_stride,
                                         FusedSamplingParams params, int batch_size, int vocab_size,
                                         int penalty_row_bytes, FusedSplitDevice split);

/// Filtered path (top-k with the whole tie group at the k-th value, min-p, top-p sequential or
/// joint): pass 1 counts the processed logits into 4096 linear bins of width 1/32 on [-64, 64)
/// and finds the row max; the counts bound every bin's mass, which fixes the top-k bin and a top-p
/// bin range. Pass 2 streams a Gumbel race over the tokens kept for sure, sums the exact domain
/// mass and collects the boundary bins (<= 2048 tokens) for an exact refine in shared memory.
/// Larger boundary bins fall back to in-kernel radix selects (about 10 passes). Rows with only
/// top-p active (no top-k, no min-p) take a cheaper branch: pass 1 builds a 2048-bin fixed-point
/// mass histogram overlaid on the count bins, the exact boundary bin follows without a full-domain
/// exp or an extra pass, and pass 2 races the bins above it and refines the boundary candidates.
/// Rows without an active filter take the categorical / greedy branches.
template <typename in_t, bool SPLIT, bool P_ONLY>
__global__ void fused_filtered_kernel(int32_t* __restrict__ out_ids,
                                      const in_t* __restrict__ logits, int64_t logits_stride,
                                      FusedSamplingParams params, int batch_size, int vocab_size,
                                      int penalty_row_bytes, FusedSplitDevice split);

}  // namespace kops::sample
