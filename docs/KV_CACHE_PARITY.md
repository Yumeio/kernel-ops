# KV cache parity — B2 (K01–K03)

Status: **B2 implemented on SM86** (20/20 kernels have bodies, tests, and a
benchmark case where noted). The contract is the C++ API in
`csrc/kvcache/ops.h` plus the layouts in `csrc/kvcache/kv_layout.cuh`; vLLM,
FlashInfer, SGLang and attention.rs are provenance, not runtime oracles.

## 1. What B2 covers

| Catalog | Public entry | Kernel(s) | Test | Benchmark |
|---|---|---|---|---|
| K01 | `reshape_and_cache` | `reshape_and_cache_flash_kernel` (NHD/HND), `reshape_and_cache_legacy_kernel` (`kVllmLegacy`) | `test_kvcache` | `bench_kvcache` (flash + legacy) |
| K01 | `append_paged_kv` | `append_paged_kv_kernel` | `test_kvcache` | `bench_kvcache` |
| K01 | `ring_kv_append` | `ring_kv_append_kernel` | `test_kvcache` | `bench_kvcache` |
| K01 | `concat_and_cache_mla` | `concat_and_cache_mla_kernel` | `test_kvcache` | `bench_kvcache` |
| K01 | `concat_and_cache_mla_grouped` | `concat_and_cache_mla_grouped_kernel` | `test_kvcache` | `bench_kvcache` |
| K02 | `gather_pages` / `scatter_pages` | `gather_pages_kernel`, `scatter_pages_kernel` | `test_kvcache` | `bench_kvcache` (`gather_pages`, `scatter_pages`) |
| K02 | `copy_pages` / `zero_pages` | `copy_pages_kernel`, `zero_pages_kernel` | `test_kvcache` | `bench_kvcache` |
| K02 | `cp_gather_cache` | `cp_gather_cache_kernel` | `test_kvcache` | `bench_kvcache` |
| K02 | `gather_and_maybe_dequant_cache` | `gather_and_maybe_dequant_cache_kernel` | `test_kvcache` | `bench_kvcache` |
| K03 | `convert_cache` | `convert_cache_kernel` | `test_kvcache` | `bench_kvcache` (`bf16 -> f32`) |
| K03 | `relayout_pages` | `kv_relayout_pages_kernel` | `test_kvcache` | `bench_kvcache` (`NHD -> HND`) |
| A12 | `concat_mla_q` | `concat_mla_q_kernel` | `test_merge` | `bench_merge` |
| A12 | `mla_context_gather` | `mla_context_gather_kernel` | `test_merge` | `bench_merge` |
| L03 | `state_copy` / `state_gather` | `state_copy_kernel`, `state_gather_kernel` | `test_state` | `bench_state` |
| L03 | `state_scatter` / `state_zero` | `state_scatter_kernel`, `state_zero_kernel` | `test_state` | `bench_state` |

Host-only entries in the same API:

| Entry | Behaviour in B2 |
|---|---|
| `kv_desc_validate` | Checks page size, dtype/codec, stride unit, latent/v_data, CSR/block-table fields and scale pointers before launch. |
| `swap_pages` | One `cudaMemcpyAsync(cudaMemcpyDefault)` per page pair; the mapping is a **host** `int64 [N,2]`, so no D2H sync is needed. |
| `copy_batch_async` | `cudaMemcpyBatchAsync` (CUDA >= 12.8) with per-copy attributes, chunked by `max_descriptors_per_call`; falls back to one `cudaMemcpyAsync` per descriptor on the legacy default stream or an older driver. |

B3 entries are declared and linked but return `kNotImplemented`:
`fused_qkv_norm_rope_append`, `concat_and_cache_mla_rope_fused`,
`indexer_k_quant_and_cache`, `cp_gather_and_upconvert_ds_mla`,
`cp_gather_indexer_k_quant_cache`, `requantize_pages`, `kv_scale_update`.
`state_commit_accepted` is B10 and returns `kNotImplemented`.

## 2. Layout and indexing contract

`slot = page * page_size + offset`; `slot < 0` skips. `num_tokens` is
`slot_mapping.size(0)`; source tensors may be longer (graph padding) and the
extra rows are never read.

| Layout | K record | V record |
|---|---|---|
| `kNHD` | `page*k_stride_page + off*k_stride_token + h*k_stride_head + d` | same formula with `v_stride_*` |
| `kHND` | same formula with `k_stride_token = Dk`, `k_stride_head = page_size*Dk` | `v_stride_token = Dv`, `v_stride_head = page_size*Dv` |
| `kVllmLegacy` | `block*H*D*PS + h*D*PS + (d/x)*PS*x + off*x + d%x`, `x = 16/sizeof(cache_t)` | `block*H*D*PS + h*D*PS + d*PS + off` |
| `kLatent` | one record per token: `page*k_stride_page + off*k_stride_token + e` | shared K/V (`v_data == nullptr`) |

Zero stride fields are filled with the standard values for the layout, so a
descriptor only needs the fields that differ from a dense cache. When
`entry_bytes != 0`, `stride_unit` must be `kByte` and the strides are divided by
`sizeof(cache_t)` at the launcher boundary.

`GatherSpec` maps output token `o` of request `r` to position
`seq_starts[r] + (o - cu_seq_lens[r])`; the physical page is
`block_table[r, position / page_size]`. `num_tokens` may be an allocation upper
bound: rows beyond `cu_seq_lens[num_reqs]` are left untouched. Gather kernels
launch `ceil(num_tokens / 8) + 2*num_reqs` CTAs; CTAs without a row exit.

## 3. Codecs in B2

| Codec | Status | Notes |
|---|---|---|
| `kNative` (bf16/f16/f32) | implemented | bit-preserving copy |
| `kFP8PerTensor` (e4m3 OCP) | implemented | `stored = satfinite(x / scale)`, `scale` is a device dequant multiplier |
| `kFP8PerHead` (e4m3 OCP) | implemented | `scale_head_stride == 1`; `reshape` uses the same code path as per-tensor with a per-head scale |
| `kMLAFp8DS` / `kMLANvfp4DS` | B3 | record layouts and upconversion land with B3 |
| per-token-head FP8/INT8/INT4, NVFP4, low-bit WHT | B3 | declared, rejected with `kNotImplemented` by the B2 entry points |
| `kIndexerFP8`/`kIndexerFP4` | B3 | split-page codec |
| e5m2 / FNUZ payloads | not compiled in B2 | rejected explicitly; B3/B8 add the paths and twin tests |

If a scale is per-tensor, `scale_head_stride` is 0; per-head requires 1 and a
non-null `k_scale`/`v_scale`. Graph replay with a new scale is supported because
the scale pointer is read from device memory by the kernel.

## 4. Test matrix

`tests/unit/test_kvcache.cc` checks:

* NHD, HND and legacy writes with negative slots and padded source rows.
* FP8 per-tensor/per-head write → `gather_pages` dequant round trip.
* CSR `append_paged_kv` with two requests and a short last page.
* Ring writes, including two tokens of one sequence colliding in the same ring
  cell (the newest position wins, ties by token index).
* Native and grouped MLA latent writes (`rank + rope` record).
* Page copy/zero across layers and K/V caches.
* Token and whole-page gather/scatter.
* `cp_gather_cache` with a block table, `seq_starts`, an upper-bound
  `num_tokens`, and untouched sentinel rows.
* `gather_and_maybe_dequant_cache` with `ENTRY_SIZE = 576`.
* Native conversion and FP8 round trip.
* NHD→HND relayout and page-size change (4→2).
* Host `swap_pages` and both `copy_batch_async` paths.
* CUDA-graph capture/replay of `reshape_and_cache` with different source data.
* Status contract: null pointers, dtype mismatch, bad descriptor, FP8 without
  scales, B3 codecs, empty gather spec, zero conversion scale, relayout batch
  mismatch.

`tests/unit/test_merge.cc` adds `concat_mla_q` (exact copy, `pe == 64`
contract) and `mla_context_gather` (two requests, shared block table, graph
replay, missing block table, entry-width mismatch). `tests/unit/test_state.cc`
covers two pools with different slot sizes, gather/scatter round trip, zeroing
only the selected slots, graph replay and the L03 contract.

Page sizes in the suite are {2, 4, 16}; the production set {16, 32, 64, 128} is
a shape parameter, not a separate code path. D ∈ {4, 8, 128, 512, 576}; the
Dk != Dv path is exercised by the MLA and generic cache records. A larger
`mem_gb` tier for 2k+ sequences is not required by B2 on the 4 GB dev GPU.

## 5. Graph and determinism

* All B2 kernels allocate nothing, query nothing and synchronize nothing.
* `reshape`/`append`/`ring`/MLA writes are independent per token. Where two
  ring writes collide, the kernel deterministically keeps the newest position.
* `copy_pages`/`zero_pages`/`gather`/`scatter`/`cp_gather` are index-driven and
  do not use atomics.
* `swap_pages` and the `copy_batch_async` fallback issue host-side
  `cudaMemcpyAsync` calls; passing pageable host memory into a captured graph is
  the caller's responsibility (the API documents it).
* `cudaMemcpyBatchAsync` is used only when the launch stream is not the legacy
  default stream, matching the API restriction.

## 6. SM86 benchmark snapshot (2026-10-31)

`RTX 3050 Laptop, 16 SMs, peak 192 GB/s`, medians from the C++ benchmark:

| Case | ms | GB/s |
|---|---:|---:|
| reshape NHD bf16, T=2048, H=4, D=128 | 0.110 | 76 |
| reshape legacy bf16, T=2048 | 0.325 | 26 |
| append CSR bf16, T=2048 | 0.114 | 74 |
| ring bf16, T=1024, window=256 | 0.068 | 31 |
| concat MLA, T=2048, rank=512, rope=64 | 0.072 | 65 |
| concat MLA grouped, 2 layers × 1024 | 0.079 | 60 |
| gather_pages bf16, T=4096 | 0.055 | 153 |
| scatter_pages bf16, T=4096 | 0.052 | 160 |
| cp_gather_cache, T=4096, 576 B records | 0.020 | 235 |
| copy_pages, 512 pages × 4 KiB × 4 caches | 0.101 | 166 |
| zero_pages, same geometry | 0.049 | 170 |
| convert bf16→f32, 256×4096 | 0.042 | 150 |
| gather_and_maybe_dequant, 576-wide, T=4096 | 0.058 | 162 |
| relayout NHD→HND, 512 pages | 0.140 | 120 |
| concat_mla_q bf16, T=1024, H=8, nope=512 | 0.107 | 175 |
| mla_context_gather bf16, T=2048, rank=512 | 0.032 | 149 |

Master controls are small (one CTA per token) and do not saturate DRAM; the
large copy/gather cases are the relevant ones for the B1 bandwidth target. The
`ring` number is dominated by the deterministic collision check for a synthetic
one-sequence prefill; decode batches (`T <= window`) skip it entirely. The
relayout case is a fallback path (layout conversion), not a hot inference path.

## 7. Known deviations / follow-ups

1. **e5m2 and FNUZ are not compiled in B2.** `PagedKVDesc::fp8` is validated and
   an unsupported encoding is rejected before launch. B3 adds the conversion
   variants; the AMD FNUZ reader is part of the deferred HIP path.
2. **`concat_and_cache_ds_mla_kernel` is implemented in B3.** The 656 B ds_mla
   record, its `cp_gather_and_upconvert` reader, and the indexer split-page
   codecs now run; the AMD FNUZ variant and the provider-owned 352 B NVFP4
   record remain out of scope. B3 also adds per-token-head FP8/INT8/INT4 writes,
   requantization, running per-head absmax, and the QK-norm/RoPE append fusions.
3. **`relayout_pages` uses `desc.kv_indptr`/`kv_page_indices` when present;
   otherwise it treats `*_seq_page_indptr` as cumulative page counts and maps
   page `j` of a sequence to `indptr[seq] + j`.** This is sufficient for dense
   page pools; a caller with non-contiguous physical pages must pass the CSR
   arrays.
4. **Ring collision check is bounded to `window + 1` tokens ahead.** That is
   exact for the documented request-major ordering (positions advance by at
   least one per token of a sequence). A caller that interleaves one sequence's
   tokens arbitrarily with more than `window` other tokens per position step
   would need an external epoch/counter; the engine owns that policy.

## 8. Evidence

* `ctest --test-dir build/dev -R "kvcache|state|merge"`: pass on SM86.
* `./build/bench/benchmark/bench_kvcache --json out/b2-kvcache.json`
* `./build/bench/benchmark/bench_state --json out/b2-state.json`
* `./build/bench/benchmark/bench_merge --json out/b2-mla.json`
* B3: `kops_test_kvcache` covers per-token-head FP8/INT8/INT4, ds_mla 656 B
  write/gather, indexer split-page write/gather, native→INT8 requantization,
  per-head absmax and both RoPE append fusions. `bench_kvcache` adds
  `reshape_per_token_head`, `ds_mla`, `indexer_fp8`, `fused_qkv_append`,
  `requantize_pages`, `kv_scale_update` and `mla_rope_fused` cases.
* Compiled for the native SM86 target with CUDA 12.9; the code is
  architecture-generic and compiles for every exact SM in `KOPS_CUDA_ARCHS`.
