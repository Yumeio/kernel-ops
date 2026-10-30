#pragma once
// =============================================================================
// kernel-ops | csrc/common/rng.cuh
// Counter-based RNG. Every random draw is a pure function of
// (seed, request_id, token_position, draw_index) so results do not depend on
// batch composition, stream, launch order, split count or graph replay.
// Requires: common/macros.cuh.
// =============================================================================

namespace kops {

/// Per-batch RNG descriptor (device-resident arrays so graphs can replay with new seeds).
struct RngParams {
  uint64_t global_seed = 0;                    // used when per_request_seed == nullptr
  const uint64_t* per_request_seed = nullptr;  // [B] request seed (user "seed" or engine generated)
  const int64_t* per_request_offset = nullptr; // [B] token position / step counter for this draw
  uint32_t stream_id = 0;                      // distinguishes independent uses (sampling=0, spec-accept=1, ...)
};

/// Philox-4x32-10 block: 4 uniform uint32 for (key, counter).
KOPS_DEVICE uint4 philox4x32_10(uint2 key, uint4 counter);
/// Uniform in (0, 1] from the canonical counter tuple; never returns 0 (safe for log).
KOPS_DEVICE float uniform_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                       uint32_t draw_index, uint32_t stream_id);
/// Standard Gumbel sample -log(-log(u)) for Gumbel-max sampling.
KOPS_DEVICE float gumbel_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                      uint32_t draw_index, uint32_t stream_id);
/// Exponential(1) sample -log(u) (used by exponential-race sampling, equivalent to Gumbel-max).
KOPS_DEVICE float exponential_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                           uint32_t draw_index, uint32_t stream_id);

}  // namespace kops
