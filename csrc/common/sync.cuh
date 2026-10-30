#pragma once
// =============================================================================
// kernel-ops | csrc/common/sync.cuh
// Memory-ordering primitives for cross-CTA / cross-GPU signalling used by
// split-KV combine, persistent schedulers, Megakernels and custom collectives.
// Every flag protocol carries an epoch so counters never need a host reset
// between graph replays.
// Requires: common/macros.cuh.
// =============================================================================

namespace kops {

/// Memory scope of an acquire/release operation.
enum class MemScope : uint8_t { kCta = 0, kCluster = 1, kGpu = 2, kSystem = 3 };

/// ld.acquire.<scope>.u32
template <MemScope kScope> KOPS_DEVICE uint32_t ld_acquire(const uint32_t* ptr);
/// st.release.<scope>.u32
template <MemScope kScope> KOPS_DEVICE void st_release(uint32_t* ptr, uint32_t v);
/// atom.add.release.<scope>.u32, returns old value.
template <MemScope kScope> KOPS_DEVICE uint32_t atomic_add_release(uint32_t* ptr, uint32_t v);
/// atom.add.acq_rel.<scope>.u32, returns old value.
template <MemScope kScope> KOPS_DEVICE uint32_t atomic_add_acq_rel(uint32_t* ptr, uint32_t v);

/// Spin until *flag >= target (acquire). Caller MUST guarantee the producer is
/// resident or already finished (see execution/persistent.cuh progress rules);
/// never use between CTAs that may not be co-scheduled.
template <MemScope kScope> KOPS_DEVICE void wait_flag_geq(const uint32_t* flag, uint32_t target);
/// Epoch-tagged counter: value = (epoch << 20) | count. Readers wait for
/// (epoch, count >= target); writers never reset, so graph replay is safe.
KOPS_DEVICE void wait_epoch_count(const uint32_t* counter, uint32_t epoch, uint32_t target);
KOPS_DEVICE void signal_epoch_count(uint32_t* counter, uint32_t epoch);

/// fence.proxy.async (make generic-proxy writes visible to TMA / async proxy).
KOPS_DEVICE void fence_proxy_async();
/// Cluster-wide barrier (barrier.cluster.arrive + wait), SM90+.
KOPS_DEVICE void cluster_sync();

// ---- Programmatic Dependent Launch (SM90+, no-op elsewhere) --------------------
/// griddepcontrol.wait: block until the primary grid's memory is visible.
/// Must precede the first read of data produced by the preceding kernel.
KOPS_DEVICE void pdl_wait_primary();
/// griddepcontrol.launch_dependents: allow the next kernel to start its prologue.
KOPS_DEVICE void pdl_launch_dependents();

}  // namespace kops
