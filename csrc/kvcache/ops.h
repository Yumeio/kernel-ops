#pragma once
// =============================================================================
// kernel-ops | csrc/kvcache/ops.h                                Catalog: K01-K03
// Host launchers for KV-cache writes / gathers / movement / conversion, plus the
// host-driven copies (swap / batched memcpy) used for CPU offload and PD transfer.
// Requires: common/*, kvcache/kv_layout.cuh, kvcache/kvcache_kernels.cuh.
// =============================================================================

namespace kops::kv {

// ---- K01 writes -------------------------------------------------------------------------
/// Dispatches to reshape_and_cache_flash_kernel (NHD/HND) or reshape_and_cache_legacy_kernel
/// (KVLayout::kVllmLegacy) or the per-token-quant kernel, by cache.layout / cache.k_codec.
Status reshape_and_cache(const Tensor& key, const Tensor& value, const PagedKVDesc& cache,
                         const Tensor& slot_mapping, const LaunchContext& ctx);
Status append_paged_kv(const Tensor& append_key, const Tensor& append_value,
                       const PagedKVDesc& cache, const Tensor& batch_indices,
                       const Tensor& positions, const LaunchContext& ctx);
Status fused_qkv_norm_rope_append(const FusedQKVAppendParams& params, DType qkv_dtype,
                                  const LaunchContext& ctx);
Status ring_kv_append(const Tensor& key, const Tensor& value, const Tensor& ring_k,
                      const Tensor& ring_v, const Tensor& token_slot, const Tensor& positions,
                      const Tensor& k_scale, const Tensor& v_scale, int32_t window,
                      const LaunchContext& ctx);

/// MLA latent writes. cache.k_codec selects kNative / kFP8PerTensor / kMLAFp8DS / kMLANvfp4DS.
Status concat_and_cache_mla(const Tensor& kv_c, const Tensor& k_pe, const PagedKVDesc& cache,
                            const Tensor& slot_mapping, const LaunchContext& ctx);
/// Multi-layer variant: kv_c/k_pe [L, T, *], cache_ptrs [L] int64 device addresses,
/// scales [L] (or absent), slot_mapping [L, T].
Status concat_and_cache_mla_grouped(const Tensor& kv_c, const Tensor& k_pe, const Tensor& cache_ptrs,
                                    const Tensor& scales, const Tensor& slot_mapping,
                                    const PagedKVDesc& cache_template, const LaunchContext& ctx);
/// RoPE(q_pe, k_pe) in place + latent cache write in one launch.
Status concat_and_cache_mla_rope_fused(const Tensor& positions, const Tensor& q_pe, const Tensor& k_pe,
                                       const Tensor& kv_c, const Tensor& cos_sin_cache,
                                       RopeStyle rope_style, const Tensor& slot_mapping,
                                       const PagedKVDesc& cache, const LaunchContext& ctx);
Status indexer_k_quant_and_cache(const Tensor& k, const PagedKVDesc& index_cache,
                                 const Tensor& slot_mapping, int32_t quant_block_size,
                                 bool scale_ue8m0, const LaunchContext& ctx);

// ---- K02 gathers ---------------------------------------------------------------------------
Status gather_and_maybe_dequant_cache(const Tensor& dst, const PagedKVDesc& cache,
                                      const GatherSpec& spec, const LaunchContext& ctx);
Status cp_gather_cache(const Tensor& dst, const Tensor& src_cache, const GatherSpec& spec,
                       const LaunchContext& ctx);
/// host_cache / host_row_ids / device_row_ids must be given together (partially offloaded cache);
/// host_cache must be pinned (cudaMemoryTypeHost) and match the device record size.
Status cp_gather_and_upconvert_ds_mla(const Tensor& dst, const PagedKVDesc& cache,
                                      const GatherSpec& spec, const Tensor& host_cache,
                                      const Tensor& host_row_ids, const Tensor& device_row_ids,
                                      const LaunchContext& ctx);
Status cp_gather_indexer_k_quant_cache(const Tensor& dst_k, const Tensor& dst_scale,
                                       const PagedKVDesc& index_cache, const GatherSpec& spec,
                                       const LaunchContext& ctx);
Status gather_pages(const Tensor& out, const PagedKVDesc& cache, const Tensor& indices,
                    bool by_token, const LaunchContext& ctx);
Status scatter_pages(const PagedKVDesc& cache, const Tensor& src, const Tensor& indices,
                     bool by_token, const LaunchContext& ctx);

// ---- K02 movement ---------------------------------------------------------------------------
Status copy_pages(const Tensor& layer_cache_ptrs, const Tensor& page_pairs, int64_t page_bytes,
                  int32_t caches_per_layer, const LaunchContext& ctx);
Status zero_pages(const Tensor& layer_cache_ptrs, const Tensor& pages, int64_t page_bytes,
                  int32_t caches_per_layer, const LaunchContext& ctx);

/**
 * swap_pages — host-driven page copies between device and host (or device and device):
 * one cudaMemcpyAsync per (src, dst) pair of `page_mapping` ([N, 2] int64 on the HOST, so no
 * device->host sync is needed to read it). Direction is inferred from the tensors' devices.
 * Not graph-capturable when host memory is pageable.
 */
Status swap_pages(const Tensor& src, const Tensor& dst, int64_t page_bytes,
                  const Tensor& page_mapping_host, const LaunchContext& ctx);

/// Descriptor-batched copies (KV offload / PD transfer).
struct BatchCopyDesc {
  const int64_t* src_ptrs_host = nullptr;   // [n] source addresses (device or pinned host)
  const int64_t* dst_ptrs_host = nullptr;   // [n] destination addresses
  const int64_t* sizes_host = nullptr;      // [n] bytes per copy
  int64_t count = 0;
  bool src_access_order_any = false;        // allow the copy engine to read sources out of stream order
                                            // (only when no stream is concurrently writing the sources)
  int64_t max_descriptors_per_call = 0;     // 0 = unlimited; chunk calls when > 0
};

/**
 * copy_batch_async — submit all copies with one cuMemcpyBatchAsync call (resolved at runtime through
 * cuGetProcAddress, CUDA >= 12.8 driver), chunked by max_descriptors_per_call. Falls back to one
 * cudaMemcpyAsync(cudaMemcpyDefault) per descriptor on older drivers and on the legacy default
 * stream (which the batch API rejects).
 */
Status copy_batch_async(const BatchCopyDesc& desc, const LaunchContext& ctx);

// ---- K03 conversion ---------------------------------------------------------------------------
Status convert_cache(const Tensor& dst_cache, const Tensor& src_cache, float scale, KVCodec codec,
                     const LaunchContext& ctx);
Status requantize_pages(const PagedKVDesc& src, const PagedKVDesc& dst, const Tensor& page_pairs,
                        const LaunchContext& ctx);
Status kv_scale_update(const Tensor& k_absmax, const Tensor& v_absmax, const Tensor& key,
                       const Tensor& value, const LaunchContext& ctx);
Status relayout_pages(const PagedKVDesc& src, const PagedKVDesc& dst,
                      const Tensor& src_seq_page_indptr, const Tensor& dst_seq_page_indptr,
                      const LaunchContext& ctx);

}  // namespace kops::kv
