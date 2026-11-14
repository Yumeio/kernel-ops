#pragma once
// =============================================================================
// kernel-ops | csrc/common/occupancy.h
// Host-only split / occupancy heuristics shared by the family launchers.
//
// Every function here is pure: it reads only the caller-supplied work shape and
// the probed LaunchContext (compute_units / max_ctas_per_cu) and never touches
// the device, allocates, or needs a replay-time query. A launcher may therefore
// call them inside a capture-replay path as long as the context it was built
// with is the same; the values are exactly what `plan()` saw.
//
// Split counts are "how many CTAs share one independent work item" (split-KV,
// split rows, split key ranges), so a context that was never probed
// (compute_units == 0) must still produce a deterministic answer: it plans for
// kOccupancyFallbackUnits SMs, matching the smallest discrete partition this
// library targets rather than "1 CTA".
// Requires: common/tensor.h.
// =============================================================================

namespace kops {

/// Compute units assumed when the caller's context was not probed. Chosen as a
/// small discrete GPU partition so an unprobed context never plans a single CTA
/// for a wide row; it is a plan input, not a claim about the device.
inline constexpr int32_t kOccupancyFallbackUnits = 32;

/// SM count a plan should use: the probed partition, or the fallback.
inline int32_t occupancy_compute_units(const LaunchContext& ctx) noexcept {
  return ctx.compute_units > 0 ? ctx.compute_units : kOccupancyFallbackUnits;
}

/// CTAs that cover `waves` full waves of the partition. `waves <= 0` means one.
inline int32_t occupancy_target_ctas(int32_t waves, const LaunchContext& ctx) noexcept {
  const int32_t w = waves > 0 ? waves : 1;
  return w * occupancy_compute_units(ctx);
}

/// Uniform split count for an operator whose natural grid is `base_ctas` CTAs and
/// whose work is `work_len` independent items.
///
///   splits = clamp(ceil(target_ctas / base_ctas), 1, max_splits)
///   splits <= work_len / min_chunk      (when min_chunk > 0)
///
/// The second bound keeps every split worth its own launch: a split with fewer
/// than `min_chunk` items does not pay for the extra CTAs and the later merge.
/// `work_len` is a length the caller already knows on the host (a host-side
/// capacity, a token count, a uniform key count) - never a device value read
/// back during planning. The result is 1 when splitting cannot help.
inline int32_t occupancy_split_count(int32_t base_ctas, int64_t work_len, int64_t min_chunk,
                                     int32_t waves, int32_t max_splits,
                                     const LaunchContext& ctx) noexcept {
  if (base_ctas <= 0 || max_splits <= 1 || work_len <= 0) return 1;
  const int64_t target = occupancy_target_ctas(waves, ctx);
  int64_t splits = (target + base_ctas - 1) / base_ctas;  // ceil
  if (splits < 1) splits = 1;
  if (splits > max_splits) splits = max_splits;
  if (min_chunk > 0) {
    const int64_t by_work = work_len / min_chunk;
    if (by_work < 1) return 1;
    if (splits > by_work) splits = by_work;
  }
  return splits < 1 ? 1 : static_cast<int32_t>(splits);
}

}  // namespace kops
