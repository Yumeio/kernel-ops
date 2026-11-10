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
| `kMLAFp8DS` | implemented (B3) | 656 B V3.2 MLA record (512 e4m3 + 4 fp32 scales + 64 bf16 RoPE) |
| `kMLAFp8DSV40` | implemented (B14) | 584 B/token DeepSeek-V4.0 record, side scale array |
| `kMLAFp8DSV41` | implemented (B14) | 528 B/token DeepSeek-V4.1 FP8 record, inline UE8M0 scales |
| `kMLAFp8DSV41Fp4` | implemented (B14) | 288 B/token V4.1 FP4 `extra_kv` record, inline E4M3 scales |
| `kMLAFp8DSFNUZ` / `kMLANvfp4DS` | B3/B8 | AMD FNUZ 656 B twin and the provider-owned 352 B NVFP4 record still return `kNotImplemented` |
| per-token-head FP8/INT8/INT4, NVFP4, low-bit WHT | B3 | declared, rejected with `kNotImplemented` by the B2 entry points |
| `kIndexerFP8`/`kIndexerFP4` | B3 | split-page codec |
| e5m2 / FNUZ payloads | not compiled in B2 | rejected explicitly; B3/B8 add the paths and twin tests |

If a scale is per-tensor, `scale_head_stride` is 0; per-head requires 1 and a
non-null `k_scale`/`v_scale`. Graph replay with a new scale is supported because
the scale pointer is read from device memory by the kernel.

### 3.1 DeepSeek-V4 MLA records (`mla_kv_format`)

The V4 formats are **not** backward compatible with the 656 B V3.2 record, so the
`KVCodec` value carries the version. All three are latent, single-head caches
(`layout == kLatent`, `num_kv_heads <= 1`, `head_dim_k == head_dim_v == 512`,
`v_data == nullptr`, `v_codec == kNative`).

| | V4.0 `kMLAFp8DSV40` | V4.1 `kMLAFp8DSV41` | V4.1 FP4 `kMLAFp8DSV41Fp4` |
|---|---|---|---|
| `entry_bytes` | 576 (data only) | 528 | 288 |
| `scale_bytes_per_token` | 8, in `k_scale_pages` | 0 (inline) | 0 (inline) |
| data layout | `[0,448)` 7×64 e4m3 NoPE, `[448,576)` 64 bf16 RoPE | `[0,512)` 16×32 e4m3 covering NoPE + RoPE | `[0,256)` 512 e2m1, element `2i` in the low nibble of byte `i` |
| scale layout | 7 UE8M0 + 1 zero pad at `k_scale_pages + page*k_scale_stride_page + off*k_scale_stride_token` | 16 UE8M0 at `[512,528)` | 32 E4M3 at `[256,288)`, one per 16 dims |
| FP8 tile scale | `2^ceil(log2(max(amax/448, 1e-4)))`, byte = exponent + 127 | same | n/a |
| FP4 tile scale | n/a | n/a | `clamp(amax/6, 2^-9, 448)` cast to e4m3 |
| page bytes | `align16(round_up(page_size*584, 576))` | `align16(page_size*528)` | `align16(page_size*288)` |

The 576-byte V4.0 page padding is FlashMLA's MODEL1 rule: the page (data region
`page_size*576` followed by the `page_size*8` scale region) is rounded up to a
multiple of 576, so a 16-token page occupies 9792 B and a 64-token page 37440 B.
`kv_physical_bytes` counts the out-of-record scale region **once** inside that
padded page; a `k_stride_page` larger than the formula wins but is never summed
with the scale array. `k_scale_stride_page`/`k_scale_stride_token` describe where
`k_scale_pages` lives, not extra bytes to add.

Scale semantics:

* UE8M0 is `2^(byte - 127)`; `0xFF` is NaN and is never emitted for finite input.
* V4.1 FP8 scales every 32 consecutive logical dims over the full 512 (tiles
  0..13 are NoPE, tiles 14..15 are RoPE). V4.0 scales only the 448 NoPE dims in
  7 tiles of 64 and keeps RoPE in bf16.
* FP4 quantizes with `cvt.rn.satfinite.e2m1x2.f32` semantics: nearest, ties to
  the **even code**, saturating at ±6, NaN mapped to zero magnitude. One NaN or
  Inf element poisons the whole 16-value tile: the scale byte becomes the E4M3
  NaN `0x7F` and every code in the tile is zero (FlashMLA `tests/quant.py`).
* A tile whose `amax` is zero floors at `1e-4` (FP8 → `2^-13`, byte 114) or at
  the `2^-9` E4M3 minimum (FP4 → byte `0x01`).

Writers: `concat_and_cache_mla` dispatches to `concat_and_cache_ds_mla_kernel`
with the record version as a template parameter (one CTA per token, 256 threads;
V4.0 warps 0..6 quantize the seven NoPE tiles and warp 7 copies the bf16 RoPE,
V4.1 walks 16 tiles over 8 warps, FP4 uses one 8-lane group per tile). Readers:
`cp_gather_and_upconvert_ds_mla` uses `cp_gather_and_upconvert_ds_mla_fp8_kernel`
for all four records — V4.0 produces `[T,512]` bf16 (448 dequantized + 64 RoPE
copied), V4.1/FP4 produce `[T,512]`; a 576-wide destination is accepted and its
`[512,576)` tail is zeroed. The 656 B path is unchanged.

`kMLAFp8DSV41Fp4` is only a valid *pair* next to a 528 B V4.1 FP8 cache
(`is_valid_kv_format_pair` in FlashMLA `kv_cache_format.h`); the descriptor sees
one cache at a time, so `kv_desc_validate` cannot enforce the pairing and the
engine owns it.


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
* V4 MLA records: byte-exact V4.0 (584), V4.1 (528) and V4.1 FP4 (288) writes
  for `page_size` 16 and 64 against a host oracle built from the formulas above
  (tile scales, e4m3 payload, bf16 RoPE, the zero scale-pad byte, and untouched
  page padding / unwritten slots), `kv_physical_bytes` equality, 16-byte
  alignment and the stride-override path, block-table gather back to bf16 with
  exact bit comparison, the 576-wide zero tail, and the V4.1 FP4 E2M1 midpoint
  ties (`0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`) plus NaN/Inf tile poisoning
  and the 448 E4M3 scale clamp.
* V4 descriptor contract: wrong record width, missing V4.0 scale array or
  `scale_bytes_per_token`, side scales on an inline codec, non-e4m3 payload,
  576 head dim, non-latent layout, non-native V codec, more than one latent
  head, short token stride, and a wrong `kv_c` width.

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

V4 MLA record cases on `RTX 5060 Ti (36 SMs, 448 GB/s)`, same build, page 64:

| Case | ms | GB/s |
|---|---:|---:|
| V4.0 write, T=256, rank=448 | 0.0030 | 139 |
| V4.1 FP8 write, T=1024, rank=448 | 0.0120 | 133 |
| V4.1 FP4 write, T=1024, rank=448 | 0.0078 | 173 |
| V4.0 gather → bf16, T=256 | 0.0040 | 112 |
| V4.1 FP8 gather → bf16, T=1024 | 0.0031 | 520 |
| V4.1 FP4 gather → bf16, T=1024 | 0.0030 | 454 |

The two small write cases are latency-bound (256 and 1024 CTAs of 256 threads,
one token each); the V4.1 gather cases reach about half of peak because the
dequantized bf16 destination is twice the record width.

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
5. **The generic page entries do not decode MLA records.** `gather_pages`,
   `scatter_pages`, `relayout_pages` and `convert_cache` treat a `kLatent`
   record as `head_dim_k` elements of `k_dtype`; they are not valid for the
   656/584/528/288 records (as before this change for 656). Use
   `cp_gather_cache` for byte-exact record movement and
   `cp_gather_and_upconvert_ds_mla` for the bf16 read path. A guard that rejects
   these codecs there is a follow-up.
6. **V4.1 FP4 must be paired with a V4.1 FP8 cache.** `kv_desc_validate` checks
   one descriptor; the `(V41, V41_FP4)` pair rule from FlashMLA is enforced by
   the engine that allocates both caches.
7. **V4.0 partial host offload keeps its scale rows on device.**
   `cp_gather_and_upconvert_ds_mla` addresses `k_scale_pages` by the block-table
   page and offset; `host_cache` rows are `entry_bytes` (576 B) wide and do not
   carry the 8 B scale region. Offloading only the data rows is supported, but
   the engine must keep the whole scale array resident.

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
* B14: `kops_test_kvcache` adds `test_ds_mla_v4` / `test_ds_mla_v4_contracts`
  (byte-exact 584/528/288 writes, block-table upconvert, physical bytes and the
  descriptor contract); `bench_kvcache` adds the six `ds_mla_v4*` cases.
  Passed on `build/dev-sm120a` (RTX 5060 Ti, CUDA 13.4, sm_120a); header, ABI
  layout and `check_roadmap.py` checks pass with `g++ -std=c++17`.
* Compiled for the native SM86 target with CUDA 12.9; the code is
  architecture-generic and compiles for every exact SM in `KOPS_CUDA_ARCHS`.
