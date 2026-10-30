#pragma once
// =============================================================================
// kernel-ops | csrc/runtime/trace.h                                 Catalog: I03
// Low-overhead tracing and reproducible benchmark records. Tracing never inserts
// host syncs into captured graphs; GPU timestamps come from events recorded on
// the caller's stream and are read back outside the hot path.
// Requires: common/*.
// =============================================================================

namespace kops::trace {

struct TraceConfig {
  bool enabled = false;
  bool nvtx_ranges = false;        // push/pop NVTX ranges around every launcher
  bool record_events = false;      // cudaEventRecord before/after each launch (not inside capture)
  bool log_dispatch = false;       // log registry selection + rejected candidates
  const char* output_path = nullptr;
};

/// One measured sample (raw, never pre-aggregated).
struct TraceRecord {
  const char* op_id;
  uint64_t kernel_id;
  const char* shape_key;           // canonical shape / dtype / layout string; owned by the tracer
                                   // (interned at trace_configure, stable until the next configure)
  int32_t num_launches;
  float plan_ms;                   // host planning time
  float kernel_ms;                 // GPU time of the kernel(s)
  float total_ms;                  // incl. reset / copy / routing / merge launches
  size_t scratch_bytes, persistent_bytes, pinned_bytes;
  double max_abs_err, rms_err;     // filled by numerics harness, NaN otherwise
  // Roofline inputs: what the kernel actually moved and computed, so a benchmark
  // can report achieved bandwidth / TFLOPS instead of only elapsed time.
  int64_t bytes_moved;             // logical traffic (weights + KV + IO), 0 = unknown
  double flops;                    // 0 = unknown
  double achieved_bytes_per_s;     // 0 when bytes_moved or kernel_ms is 0
  double achieved_flops;           // 0 when flops or kernel_ms is 0
};

Status trace_configure(const TraceConfig& cfg);
Status trace_flush();
/// Environment record attached to every benchmark file: git commit, provider commits, GPU SKU,
/// SM, clocks, driver, CUDA runtime, nvcc version, compile flags hash.
Status write_environment_record(const char* path);

}  // namespace kops::trace
