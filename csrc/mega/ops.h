#pragma once
// =============================================================================
// kernel-ops | csrc/mega/ops.h                                   Catalog: E04-E08
// prepare (outside capture) -> run (capture-safe). Fallback is chosen BEFORE enqueue;
// a failed Megakernel launch is reported, never silently re-run modularly after KV/state writes.
// Requires: common/*, mega/*.cuh.
// =============================================================================

namespace kops::mega {

enum class MegaKind : uint8_t { kMLP = 0, kAttentionBlock = 1, kMoELocal = 2, kMoEEP = 3, kDecoder = 4 };

/// Opaque prepared plan (host side owns StageDesc / TileTask arrays in persistent workspace).
struct MegaPlan {
  MegaKind kind;
  MegaPlanDevice device;
  MegaPlanInfo info;
  uint32_t plan_generation;
};

Status mega_workspace_requirements(MegaKind kind, const void* params, const LaunchContext& ctx,
                                   WorkspaceRequirements* out);
Status mega_prepare(MegaKind kind, const void* params, const Workspace& ws,
                    const LaunchContext& ctx, MegaPlan* plan);
Status mega_run_mlp(const MegaPlan& plan, const MegaMLPParams& params, const LaunchContext& ctx);
Status mega_run_attention_block(const MegaPlan& plan, const MegaAttentionBlockParams& params,
                                const LaunchContext& ctx);
Status mega_run_moe_local(const MegaPlan& plan, const MegaMoELocalParams& params,
                          const LaunchContext& ctx);
Status mega_run_moe_ep(const MegaPlan& plan, const MegaMoEEPParams& params, const LaunchContext& ctx);

}  // namespace kops::mega
