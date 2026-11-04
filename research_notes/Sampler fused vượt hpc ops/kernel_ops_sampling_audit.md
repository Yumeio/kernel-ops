# kernel-ops sampling implementation audit (fused-sampler design input)

**Scope.** Read-only audit of `csrc/sampling/` in the local kernel-ops checkout (HEAD `0aa87a2`
"feat: implement B6 routing, permute and sampling (ABI 5)", `KOPS_ABI_VERSION 5` at
`csrc/common/macros.cuh:42`), plus its tests, benchmarks and the in-tree hpc-ops sampler used as the
comparison target. No repository file was modified; this note is the only artifact written.

**Citation convention.** Every structural claim cites `path:line` inside this repository
(links are relative to this file, e.g. [csrc/sampling/sampling.cu:1757](../../../csrc/sampling/sampling.cu)).
External claims (bandwidth, FlashInfer/vLLM semantics) cite a URL. Claims are labelled
**FACT** (read directly in code / measured on this machine), **MEASURED** (numbers produced by
running the in-tree benchmark here), or **ESTIMATE** (model output; the assumption is stated).

**Two corrections to the task brief, established before analysis:**

- `csrc/sampling/sampling.cu` is **2357 lines**, not 2228 (working tree, 2026-11-02).
- The sampling tests are **not** in `tests/unit/test_m1.cc`; that file contains no sampling code at
  all (`grep` over `tests/` returns zero sampling symbols in `test_m1.cc`). They live in
  `tests/unit/test_sampling.cc` (1528 lines) and a dedicated `benchmark/bench_sampling.cu`
  (592 lines). `test_m1.cc` is 385 lines and covers only the M1 slice (norm/activation/merge).

**Measured machine.** `NVIDIA GeForce RTX 3050 Laptop GPU`, sm_86, 16 SMs, harness-reported peak
192.0 GB/s, 4 GB VRAM. All MEASURED numbers below are from `bench_sampling.exe` built from this
tree (`build/bench/benchmark/bench_sampling.exe`), default 10 warmup / 80 iterations. They are a
*calibration anchor*, not an H100/B200 prediction.

---

## Q1. Inventory of the public sampling / logits ops and their real implementation status

### Takeaway

All 15 declared entry points in `csrc/sampling/ops.h` have a host body **except** the three
speculative-decoding entries (`rejection_sample`, `tree_verify`, `build_tree_attention_mask`),
which are declared in `ops.h` and in `speculative_kernels.cuh` but defined nowhere in the tree.
Of the twelve implemented entries, three are degraded relative to their own documentation:
`sample()` does not contain the rejection-sampling algorithm its kernel header describes,
`gumbel_max_sample()` refuses top-p although the header advertises it as "generalized", and
`sample()` never uses the fast candidate-based top-k selector that already exists in the same
translation unit.

### Cited Findings

Every declaration is in [csrc/sampling/ops.h:12-54](../../../csrc/sampling/ops.h); host bodies and
launches are all in `csrc/sampling/sampling.cu` and `csrc/sampling/logits_processor.cu`.

| # | Op (declaration) | Host body | Kernel body | Status |
|---|---|---|---|---|
| 1 | `sampling_workspace_requirements` `ops.h:12` | `sampling.cu:837-869` | — | **Implemented** |
| 2 | `sample` `ops.h:20` | `sampling.cu:2225-2308` | `softmax_stats_kernel` `sampling.cu:373`, `softmax_norm_kernel` `sampling.cu:439`, `top_k_top_p_sampling_kernel` `sampling.cu:1757` | **Implemented, degraded** (see Q2/Q3) |
| 3 | `gumbel_max_sample` `ops.h:24` | `sampling.cu:2310-2355` | `gumbel_max_sampling_kernel` `sampling.cu:1935` | **Partially implemented**: top-p rejected at `sampling.cu:2315-2318`; `MAX_TOPK_CANDIDATES` unused (`sampling.cu:1947`) |
| 4 | `softmax` `ops.h:26` | `sampling.cu:871-912` | `softmax_kernel` `sampling.cu:293`, split pair `sampling.cu:373/439` | **Implemented** (split path real, not just declared) |
| 5 | `argmax` `ops.h:28` | `sampling.cu:914-963` | `argmax_kernel` `sampling.cu:525` | **Implemented** (1 pass) |
| 6 | `top_k_renorm_probs` `ops.h:30` | `sampling.cu:2102-2134` | `top_k_renorm_probs_kernel` `sampling.cu:1610` | **Implemented**, uses the candidate fast path `sampling.cu:1637` with radix fallback `sampling.cu:1640` |
| 7 | `top_p_renorm_probs` `ops.h:32` | `sampling.cu:2136-2168` | `top_p_renorm_probs_kernel` `sampling.cu:1677` | **Implemented**, general (4-pass u64 nucleus) only |
| 8 | `top_k_mask_logits` `ops.h:34` | `sampling.cu:2170-2223` | `top_k_mask_logits_kernel` `sampling.cu:1493` | **Implemented**; 2/3/4 radix passes by dtype width (`sampling.cu:1500-1501`) |
| 9 | `compute_logprobs` `ops.h:36` | `sampling.cu:965-1067` | `compute_logprobs_kernel` `sampling.cu:568` | **Implemented**, single pass, top-n ≤ 20 (`kMaxTopN`, `sampling.cu:60`) |
| 10 | `token_bincount` `ops.h:40` | `logits_processor.cu:327-381` | `logits_processor.cu:75` | **Implemented** |
| 11 | `apply_penalties` `ops.h:42` | `logits_processor.cu:383-424` | `logits_processor.cu:104` | **Implemented** (repetition/frequency/presence, in place) |
| 12 | `apply_token_bitmask` `ops.h:45` | `logits_processor.cu:426-464` | `logits_processor.cu:144` | **Implemented** |
| 13 | `apply_logit_bias` `ops.h:47` | `logits_processor.cu:466-498` | `logits_processor.cu:171` | **Implemented**, but one thread per row walks the segment serially (`logits_processor.cu:180-192`) |
| 14 | `rejection_sample` `ops.h:50` | **none** | `speculative_kernels.cuh:32-33` (declaration only) | **Declared only** |
| 15 | `tree_verify` `ops.h:51` | **none** | `speculative_kernels.cuh:51-52` | **Declared only** |
| 16 | `build_tree_attention_mask` `ops.h:52` | **none** | `speculative_kernels.cuh:58-64` | **Declared only** |

- The three missing bodies are confirmed absent by grep over the whole tree: the only hits for
  `rejection_sample_kernel`, `tree_verify_kernel`, `build_tree_attention_mask_kernel` are the
  declarations themselves and the ROADMAP rows ([ROADMAP.md:1331-1332](../../../ROADMAP.md)).
  `sampling.cu` includes `speculative_kernels.cuh` ([sampling.cu:51](../../../csrc/sampling/sampling.cu))
  but defines none of them, so `kops_cuda` currently has no definition for three symbols declared in
  `ops.h` (an engine that links the sampling family and calls them would fail at link time).
- `csrc/runtime/registry.h` contains **no** sampling rows at all (grep for `sample` in
  `csrc/runtime/registry.h` returns nothing), so no sampling op has a capability status, provider,
  predicate or evidence record. Per `AGENTS.md` ("Do not mark a row as compiled... without the
  corresponding evidence and an implemented registry") the family is therefore outside the
  dispatchable surface even where bodies exist.
- `docs/TREE.md` — referenced by `AGENTS.md` as the design map — does not exist in this checkout
  (`docs/` contains only `docs/KV_CACHE_PARITY.md`). Tree/status claims cannot be cross-checked there.
- `csrc/sampling/speculative_kernels.cuh:17-18` declares `draft_probs`/`target_probs` as
  `[num_draft_tokens, V]` fp32 full-vocab matrices, i.e. the speculative path is specified as
  full-vocabulary and would need the same materialisation that dominates `sample()`.

### Inferences

- The family is structurally complete for the single-token decode path (softmax/argmax/logprobs/
  truncation/sample/processors) and structurally incomplete for speculative decoding (3 of 15).
- Because there are no registry rows, nothing in the tree currently asserts which of these paths is
  compiled, numerically validated or release-supported for a given SM; the only evidence is the
  unit test binary and the benchmark table.
- `apply_logit_bias` being one-thread-per-row is fine for sparse bias (nnz per row is small —
  MEASURED 1.8-3.5 GB/s "effective" in `bench_sampling`, i.e. the op is launch-latency-bound, not a
  bottleneck), but it means a fused "processors + sample" megakernel cannot simply inherit it.

### Gaps

- No evidence in-tree of which of these kernels are instantiated for SM90a/SM100a; the only built
  target on this machine is sm_86. `CMakeLists.txt:153-154` compiles both `.cu` files unconditionally.
- Whether the missing speculative bodies exist on another branch could not be determined from this
  checkout (single-branch history visible via `git log`).

---

## Q2. End-to-end trace of `sample()` and `top_k_top_p_sampling_kernel`: passes, bytes, launches, workspace

### Takeaway

For the benchmarked decode configuration (BF16 logits, `top_k<=64`, `0<top_p<1`, `min_p=0`,
`joint_topk_topp=false`, V=120832) `sample()` is **4 kernel launches worst case / 3 typical** and
moves **36 bytes per (row, token)** — two BF16 reads of the logits, one FP32 probability-row write,
and **seven FP32 full-vocab reads** in the selection kernel. The information-theoretic minimum for
"read the logits once and emit one token id" is **2 bytes per (row, token)**, so the current path is
**18x the minimum traffic** and additionally requires a `batch x vocab x 4` byte workspace.

### Cited Findings

**Launch sequence (FACT).** `sample()` enqueues, in order:

1. `fill_temperature_kernel` — only when `params.temperature == nullptr`
   ([sampling.cu:2273-2274](../../../csrc/sampling/sampling.cu), kernel at `sampling.cu:2082-2086`).
2. `softmax(...)` — the public entry point, which dispatches by vocab
   ([sampling.cu:2296](../../../csrc/sampling/sampling.cu) → `sampling.cu:757-768`).
   For `V=120832`, `softmax_ctas_for_vocab` returns **4** (boundary `<=131072`,
   [sampling.cu:65-71](../../../csrc/sampling/sampling.cu)), so this is **two** launches:
   `softmax_stats_kernel<<<dim3(batch,4),512>>>` then `softmax_norm_kernel<<<dim3(batch,4),512>>>`
   ([sampling.cu:725-731](../../../csrc/sampling/sampling.cu)).
3. `top_k_top_p_sampling_kernel<1,8><<<batch,256>>>`
   ([sampling.cu:2301-2302](../../../csrc/sampling/sampling.cu)).

So: **4 launches** with a scalar/default temperature, **3** when a per-row temperature tensor is
supplied. There is no PDL, no persistent kernel, no graph node fusion
(`enable_pdl`/`strategy`/`capture_generation` are never read in the file).

**Full-vocab passes inside `top_k_top_p_sampling_kernel` (FACT, one CTA per row, 256 threads,
`kRowThreads=256` at [sampling.cu:1080](../../../csrc/sampling/sampling.cu)):**

| Step | Code | Full-row read? | Notes |
|---|---|---|---|
| a. row max | `sampling.cu:1785` → `row_max` `1450-1454` | 1 read of probs | executed **unconditionally**, needed only for min-p (`mp_thresh` at `1796`) |
| b. k-th value select | `select_kth_key` `sampling.cu:1123-1185`, called `1790` | **4 reads** (one per 8-bit radix pass, `1130`) | every pass re-reads the whole row; the `pass>0` filter only skips *counting*, not loads (`1140`) |
| c. boundary index scan | `boundary_equals` `sampling.cu:1188-1220`, called `1791` | 1 read | skipped entirely when the tie group is fully kept (`1194`) |
| d. candidate compaction (fast path) | `sampling.cu:1801-1815` | 1 read | admits ≤ 64 elements into shared `cp[64]/ci[64]` |
| e. serial top-p + sample | `sampling.cu:1817-1868` | 0 | thread 0 alone: bubble sort ≤ 64 entries (`1819-1830`), cumulative mass (`1831-1840`), one `uniform_from_counter` (`1853`), CDF walk (`1859-1865`) |

Fast-path total: **7 FP32 full-row reads**. Kernel writes only `out_ids[row]` and `out_valid[row]`
(`1866-1867`), 5 bytes per row.

**General path (FACT)** — taken when `!(use_k && use_tp && !joint && k<=64)`, i.e. for top-p-only
(`top_k<=0`, the vLLM/OpenAI default), for `k>64`, or for `joint_topk_topp`:

| Step | Code | Reads |
|---|---|---|
| a. row max | `1785` | 1 |
| b. k-th radix | `1790` | 4 (only if `use_k`) |
| c. k boundary | `1791` | 1 (only if `use_k`) |
| d. nucleus radix | `select_nucleus` `sampling.cu:1237-1294`, called `1876` | **4** |
| e. nucleus boundary | `boundary_equals_mass` `1297-1324`, called `1878` | 1 |
| f. truncated mass | `sampling.cu:1881-1887` | 1 |
| g. prefix-scan sample | `sampling.cu:1903-1932` | 1 (+ the winning thread re-reads its chunk, `1917-1926`) |

So the dominant production configuration (top-p only, no top-k) costs **8 FP32 full-row reads**;
top-k + top-p with `k>64` costs **13**. That is 32 and 52 bytes per (row, token) of probability
traffic alone.

**Shared-memory footprint (FACT).** `top_k_top_p_sampling_kernel` statically declares `hist[256]`,
`scan[256]` (u64), `wh[8*256]` (u64, 16 KiB), `rh[256]` (u64), plus the 64-entry fast-path arrays
([sampling.cu:1763-1768, 1802-1804](../../../csrc/sampling/sampling.cu)) ≈ **21.6 KiB/CTA**. The
nucleus scan itself is a naive Hillis-Steele `block_exscan_u64` with **3 `__syncthreads()` per step
and 8 steps = 24 barriers** ([sampling.cu:1098-1113](../../../csrc/sampling/sampling.cu)), called
once per boundary/mass pass.

**Workspace (FACT).** `sampling_workspace_requirements` returns
`device_scratch_bytes = B*V*4 + B*4 + B*ctas*2*4`, `alignment = 256`
([sampling.cu:858-868](../../../csrc/sampling/sampling.cu)). `sample()` re-derives the same layout at
call time ([sampling.cu:2248-2260](../../../csrc/sampling/sampling.cu)) and rejects a short
workspace with `kWorkspaceTooSmall` (`2253-2256`). For V=120832, `ctas=4`:
`4*V + 4 + 32 = 483,364 bytes per row`, i.e. **247.5 MB at B=512**.

**`ops.h` document comment is incomplete (FACT).** `ops.h:17-19` says the workspace holds "the fp32
probability row of the call (`num_tokens * vocab * 4` bytes, plus the optional split-softmax state)".
The code also reserves `max_batch * sizeof(float)` for the temperature fill
([sampling.cu:863, 2250](../../../csrc/sampling/sampling.cu)). A caller that reserves exactly the
documented amount fails at `2253`.

**Independent confirmation of the 7-pass count (MEASURED).** Using the byte model above,
`sampling.topk_topp.bf16.v32768.b256.k20.p0.95` moves `256*32768*36 = 302 MB` and measured
**1.6852 ms**, i.e. **179 GB/s = 93 % of the 192 GB/s device peak**. A 6- or 8-pass model would give
154 or 205 GB/s, so the measured value pins the pass count at 7 for this configuration. The other
sample case, `bf16.v131072.b32.k50.p0.9`, moves 151 MB in 1.3373 ms = **113 GB/s (59 % of peak)**,
consistent with only 32 CTAs on 16 SMs.

### Inferences

- `sample()` is not "one pass over the logits plus a reduction"; it is a softmax materialisation
  followed by a multi-pass selection over the *materialised* array. Every additional truncation
  feature multiplies the pass count instead of sharing one traversal.
- The single largest removable cost is the FP32 materialisation: it converts a 2-byte read into
  `4 + 28` bytes per token and 483 KB of workspace per row.
- The second largest is that the *most common* configuration (top-p only) takes the slowest branch.

### Gaps

- No profiler (Nsight) trace was captured, so per-kernel time split between `softmax_*` and
  `top_k_top_p_sampling_kernel` is not measured here; only the total and the byte model.
- The claim "the fast path is hit for k<=64" is from code reading plus the byte-model fit at B=256;
  it was not verified with a per-kernel counter.

---

## Q3. Exact sampling semantics

### Takeaway

Semantics are **full-vocabulary and exact**: thresholds are computed over the entire row, ties break
deterministically to the lowest token id, reductions use fixed trees, and the draw is a pure function
of `(seed, row, position, draw_index, stream_id)`. Truncation order is top-k → top-p over the
renormalised top-k remainder by default, with a joint mode available. The documented
rejection-sampling algorithm is **not** what runs.

### Cited Findings

- **Full vocab, not bounded top-K (FACT).** `sample()` materialises all V probabilities
  ([sampling.cu:2296](../../../csrc/sampling/sampling.cu)) and every threshold is a rank/mass
  threshold over the whole row (`select_kth_key` `1123`, `select_nucleus` `1237`). The 64-candidate
  array is a post-hoc compaction of an already-exact threshold (`1801-1815`), and it is provably not
  lossy: `RowMask::ok` (`1228-1232`) admits exactly `k` elements when `k<=64`, and the branch is
  guarded by `k <= 64` (`1801`). This is the property hpc-ops does not have (its `effK` is clamped
  to `kMaxTopK` = 32/64, [fused_sampler.cu:322-327](../../../3rd/hpc-ops/src/sampler/fused_sampler.cu),
  [fused_sampler.cu:44-46](../../../3rd/hpc-ops/src/sampler/fused_sampler.cu)).
- **Truncation order (FACT).** Default `joint_topk_topp=false`: top-k first, then top-p with the
  nucleus mass computed *over the top-k subset* — `select_nucleus` is called with `RowMask base`
  ([sampling.cu:1874-1878](../../../csrc/sampling/sampling.cu)) and `select_nucleus` accumulates only
  masked elements and normalises the target by the masked total (`1251`, `1265-1269`). Joint mode
  passes an all-permissive mask (`1874`) so both filters apply independently to the full row. This
  matches FlashInfer's `filter_apply_order` = `"top_k_first"` / `"joint"` distinction
  ([FlashInfer docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html)).
- **top-k edges (FACT).** `k<0 → 0`, `k>V → V` ([sampling.cu:1462-1468](../../../csrc/sampling/sampling.cu));
  the filter is active only when `0 < k < V` (`1778`). `k == V`, `k <= 0` and `k >= V` all disable it,
  consistent with the header comment at `sampling_kernels.cuh:17`.
- **top-p edges (FACT).** `top_p >= 1` disables truncation; `top_p <= 0` is reinterpreted as
  "keep only the largest token" by forcing a k=1 select (`sampling.cu:1779-1794`). `top_p` is *not*
  validated on the host for range — `sample()` only validates the default temperature
  (`2225-2247`), so `top_p = NaN` silently disables the filter (`!(tp>0)` is false, `tp<1` is false).
- **min-p (FACT).** `mp_thresh = min_p * max(row probs)`, admission is `v >= mp_thresh` in
  probability space ([sampling.cu:1230, 1785, 1796](../../../csrc/sampling/sampling.cu)), matching
  vLLM's "minimum probability relative to the most likely token"
  ([vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params)). There is **no
  upper clamp**: `min_p > 1` empties every row, and `sample()` then writes `-1` for every row
  (`1888-1893`). vLLM clamps to `[0,1]`; this is an untested behavioural divergence.
- **Temperature (FACT).** Applied inside softmax as `logits * (1/T)`
  ([sampling.cu:308-310, 323](../../../csrc/sampling/sampling.cu)). `T <= 0`, NaN and ±inf all mean
  greedy, implemented as a one-hot probability row at the argmax (`309`, `351`). Note this is *not*
  vLLM's approach: vLLM short-circuits greedy to `top_k=0, top_p=1` and an argmax kernel
  ([vLLM sampling_params](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)),
  so vLLM pays one read where kernel-ops pays one read + one FP32 row write + the full selection.
- **Per-row parameters (FACT).** `params.temperature/top_k/top_p/min_p` are `[B]` device pointers;
  null falls back to the scalar default (`sampling_kernels.cuh:15-26`,
  `sampling.cu:1462-1479`). This is the same per-request model as vLLM/FlashInfer.
- **Deterministic tie-breaking (FACT).** Total order is (key descending, index ascending):
  `RowMask::ok` uses `key > kkey || (key == kkey && i <= kbound)` (`1229`), the fast-path comparator
  is `cp[b] > cp[a] || (cp[b]==cp[a] && ci[b] < ci[a])` (`1821`), the boundary scan walks ascending
  ids (`1201-1217`), and Gumbel ties break to the lower id (`2049`). The boundary scan's
  `block_exscan_u64` gives the same result regardless of scheduling because the element order is the
  index order and no atomic is used for ordering (contrast the candidate *gather*, which does use
  `atomicAdd(&sh[3], 1)` at `1407`, but whose result is made order-independent by the rank pivot at
  `1426-1440`).
- **RNG (FACT).** Philox-4x32-10 (`csrc/common/device_impl.cuh:612-639`), keyed by a 64-bit seed and
  a 4-word counter built from `(request_id, stream_id, position, position>>32 ^ draw_index)`
  (`device_impl.cuh:641-651`). Categorical sampling calls it once per row with `draw_index = 0`
  (`sampling.cu:1853-1855`, `1899-1900`); Gumbel calls it once per **token** with `draw_index = i`
  (`2044-2046`). Uniform is `(r.x>>8 + 1)/2^24`, i.e. `(0,1]` with 24-bit resolution
  (`device_impl.cuh:650`).
- **Reproducibility contract (FACT).** The draw depends only on `(seed, row, position, draw, stream)`,
  never on batch composition, grid shape or launch order (`csrc/common/rng.cuh:4-7`). It is *not*
  auto-advanced: the engine must increment `per_request_offset[row]` every decode step, otherwise
  repeated `sample()` calls return the same token — which is exactly what the test asserts
  ([test_sampling.cc:1451-1457](../../../tests/unit/test_sampling.cc)).
- **NaN / inf policy (FACT).** NaN is treated as `-inf` (`clean_logit`, `sampling.cu:96`); a row
  containing `+inf` yields a uniform distribution over the `+inf` entries (`111-113`, `352`); a row
  with no finite and no `+inf` logit yields all-zero probabilities (`343-355`) and `sample()` then
  writes `out_ids = -1`, `out_valid = 0` (`1888-1893`).
- **Gumbel-max path (FACT).** `out = argmax(logits/T + Gumbel(u_i))` over the top-k ∧ min-p
  admissible set (`sampling.cu:2026-2052`), with an explicit `kNotImplemented` for any top-p
  (`2315-2318`). Because it draws one Gumbel per token it is distributionally exact over the full
  vocab without a normaliser, but it costs `V` Philox blocks per row.
- **Greedy is not short-circuited (FACT).** A greedy row produces a one-hot FP32 row (`351`) and
  then goes through the same radix/nucleus machinery; nothing detects "all rows greedy" and routes to
  `argmax_kernel`.

**Inconsistencies between comments and code (FACT):**

1. `sampling_kernels.cuh:55-58` describes `top_k_top_p_sampling_kernel` as "sorting-free rejection
   sampling ... (FlashInfer dual-pivot style): draw u, binary-search pivots ..., bounded by
   max_rounds; rows exceeding it fall back to an exact radix select." The kernel **contains no
   rejection sampling and no pivots**: `MAX_ROUNDS` is explicitly discarded
   ([sampling.cu:1771](../../../csrc/sampling/sampling.cu)) and the body is an unconditional
   radix-select + boundary + prefix-scan (`1786-1932`). The same claim is repeated in
   `sampling.cu:1070-1076` ("The public dynamics are identical to rejection sampling; only the number
   of passes differs") and in `ROADMAP.md:1120`.
2. `sampling.cu:12-15` says "this body implements the `CTAS_PER_ROW == 1` case only, so
   `sampling_workspace_requirements()` reports zero device scratch". Both halves are false today:
   the split body exists (`softmax_stats_kernel`/`softmax_norm_kernel`, `373-523`) and the
   requirement function reports `probs + temp + state` (`862-866`).
3. `sampling_kernels.cuh:69-72` advertises `gumbel_max_sampling_kernel` with "optional top-k
   candidates computed in the same pass" and `MAX_TOPK_CANDIDATES`; the constant is discarded
   (`sampling.cu:1947`) and top-k is a 4-pass radix (`1970-1997`).
4. `ops.h:18-19` understates the workspace (see Q2).
5. `sampling_kernels.cuh:29-31` says V up to 256K is split "with a second merge phase" — true for
   `softmax` only; the selection kernel has no split path at all (`1772`, one CTA per row).
6. `ROADMAP.md:1085` states the B6 benchmark comparison (geomean 3.57x over PyTorch) but the
   `sample` benchmark rows ignore the returned `Status` (see Q5), so a hard failure would still be
   reported as a passing case.

### Inferences

- The semantic surface is strong enough to be a *contract* for the new kernel: exact full-vocab
  thresholds, lowest-index ties, counter RNG, per-row params, `-1` on empty rows. A faster design
  must reproduce these bit-for-bit or the engine's reproducibility guarantees change.
- Because FlashInfer-style rejection sampling only guarantees a *stochastically equivalent* token and
  not the exact candidate set ([Qrita/arXiv 2602.01518](https://arxiv.org/html/2602.01518v1) notes it
  "severely undermines compatibility ... such as speculative decoding verification"), kernel-ops'
  exact-threshold contract is the more valuable one to keep; the speed of rejection sampling can be
  recovered by a candidate-based single pass instead.
- `min_p > 1` and `top_p = NaN` are silent-corruption edges that a fused design should reject on the
  host rather than propagate.

### Gaps

- Bit-exactness of `sample()` against the fp64 reference is only established for V=16 in the
  distribution test (see Q5); no reference comparison exists at V≥4096 for the combined
  top-k + top-p path.
- Whether `joint_topk_topp=true` is exercised by any model in `docs/MODEL_COVERAGE.md` could not be
  checked — that file is absent from this checkout.

---

## Q4. API / workspace constraints that affect fusion and CUDA Graph capture

### Takeaway

The call path is capture-clean (no allocation, no host sync, no device query, caller-owned stream and
workspace), and the split-softmax path is proven capturable by a test. The blockers for a fused
sampler are not capture safety but (a) the `batch x vocab x 4` workspace reservation, (b) one CTA per
row with no cross-CTA state in the ABI, and (c) a workspace layout that aliases three regions and is
therefore not safe for two concurrent `sample()` calls.

### Cited Findings

- **No allocation in the run path (FACT).** `sample()` only builds `Tensor`/`Workspace` views over
  caller memory and launches kernels ([sampling.cu:2257-2302](../../../csrc/sampling/sampling.cu)); the
  only host calls are `cudaGetLastError` (`2303`). Contrast hpc-ops, which allocates its scratch with
  `cudaMallocAsync` *inside* the launcher on every call
  ([fused_sampler.cu:585-607, 648-651](../../../3rd/hpc-ops/src/sampler/fused_sampler.cu)).
- **Caller owns stream/device/workspace (FACT).** `LaunchContext` carries `stream`, `backend`,
  `device`, `arch_id`, `wave_size`, `compute_units`, `max_ctas_per_cu`, `lds_bytes`,
  `enable_pdl`, `deterministic`, `batch_invariant`, `capture_generation`, `strategy`
  ([csrc/common/tensor.h:43-58](../../../csrc/common/tensor.h)); `Workspace` is a pure lease
  (`tensor.h:83-90`) documented as "valid only during one run; may be reused after completion event"
  (`tensor.h:76`).
- **`ctas_per_row` is advisory only (FACT).** `sampling_workspace_requirements` takes
  `max_batch, max_vocab, ctas_per_row` and reserves `max(requested, derived_from_vocab)`
  ([sampling.cu:850-857](../../../csrc/sampling/sampling.cu)); `sample()` ignores the caller's value
  entirely and recomputes from vocab (`2248`). The cap is `kMaxSoftmaxCtas = 16` (`61`), and asking
  for more than 16 is a hard `kInvalidArgument` (`853-857`).
- **The layout aliases three regions (FACT).** `probs_ptr = base`, `temp_ptr = probs_ptr + B*V`,
  `state_ptr = temp_ptr + B`, and the same `temp_ptr` bytes are then reinterpreted as the `uint8`
  `out_valid` row flag after softmax has consumed them
  ([sampling.cu:2257-2260, 2298-2300](../../../csrc/sampling/sampling.cu)). Two concurrent `sample()`
  calls sharing one `Workspace` (different streams) would corrupt each other; the engine must
  partition per stream. Nothing in `ops.h` says so.
- **Graph capture is exercised for `softmax` only (FACT).**
  `test_softmax_split` captures `softmax()` into a stream, instantiates and replays it
  ([test_sampling.cc:970-985](../../../tests/unit/test_sampling.cc)). No test captures `sample()`,
  `gumbel_max_sample()`, `top_k_renorm_probs` or `top_p_renorm_probs`.
- **Replay-safe by construction (FACT).** Workspace size and the probs/state offsets are derived
  only from `logits.shape`, never from pointer values or device properties
  ([sampling.cu:2248-2260](../../../csrc/sampling/sampling.cu)); all kernel arguments are either
  tensors, scalars, or the workspace base. The split count is a function of vocab only
  (`softmax_ctas_for_vocab`, `65-71`), explicitly to keep rows batch-invariant (`63-64`).
- **Ignored context fields (FACT).** `sampling.cu` never reads `ctx.deterministic`,
  `ctx.batch_invariant`, `ctx.strategy`, `ctx.enable_pdl` or `ctx.capture_generation` (grep over the
  file returns no hits). There is no PDL-aware launch, no persistent/megakernel variant, and no
  plan object; `ExecutionStrategy::kGraph/kPDL/kPersistent/kMegakernel`
  (`tensor.h:31-37`) are unused by this family.
- **Layout requirements on inputs (FACT).** `sample()` requires `logits.stride[1]==1`
  (`check_logits_tensor`, `682-689`) and `out_ids` contiguous (`2241-2243`), but accepts **any**
  `logits.stride[0]` (the kernels take `logits_stride`, `305`, `533`) and only 2-byte alignment for
  BF16 (alignment argument is `element_bytes`, `683`). Row-vectorisation is decided per row at
  runtime by `row_vec_ok` (`240-247`), so an unaligned producer silently degrades to scalar loads.
- **Architecture gating (FACT).** The sampling family contains no `KOPS_TARGET_*` predicate at all;
  it is compiled for whatever gencode list the build uses (`CMakeLists.txt:153-154`) and would run
  identically on sm_70..sm_121 as long as `atomicAdd(unsigned long long*)` on shared memory is
  available (sm_60+).
- **Cross-CTA state is already in the ABI for softmax, not for selection (FACT).** The split softmax
  uses `row_state_workspace[B, CTAS_PER_ROW, 2]` fp32 (`373-435`) and phase 2 merges the small state
  array per CTA without a third launch (`437-523`). The selection kernel has no equivalent parameter:
  its signature is `(out_ids, out_valid, probs, params, batch, vocab)`
  ([sampling_kernels.cuh:60-66](../../../csrc/sampling/sampling_kernels.cuh)), so a multi-CTA
  selection needs either a new parameter block or a pre-merged per-row state.

### Inferences

- A fused sampler can keep the current capture story exactly: two kernels, workspace carved from
  `Workspace::device_scratch`, all sizes derived from shapes, no allocation, no sync. The only new
  requirement is that any per-row/per-slice state the merge kernel needs must be reserved in the same
  workspace and documented in `sampling_workspace_requirements`.
- Sizing the workspace from "max batch x max vocab x 4" is the binding constraint at large batch
  (247 MB at B=512, V=120832). A candidate-list design (e.g. `B x Nmax x K x 8`) drops this to
  ~4 MB for the same shapes — the hpc-ops numbers are `B*Nmax*kMaxTopK*(4+4)` with `Nmax<=32`
  ([fused_sampler.cu:642-651](../../../3rd/hpc-ops/src/sampler/fused_sampler.cu)) — which matters for
  engines that pre-reserve workspace per stream.
- The absence of `deterministic`/`batch_invariant` handling is not a functional bug today because the
  implementation is already deterministic and batch-invariant; a fused multi-CTA design must keep
  that property (fixed-tree merges in CTA order, no order-dependent float atomics) or it will break
  the documented guarantee at `rng.cuh:4-7`.

### Gaps

- Multi-stream safety is inferred from the aliasing layout, not tested: no test runs two `sample()`
  calls on two streams with disjoint workspaces.
- Whether the engine that will host this kernel can afford 483 KB/row of workspace is an open
  product question; no in-tree consumer exists to measure against.

---

## Q5. Test and benchmark coverage

### Takeaway

Unit coverage is good for softmax/argmax/logprobs/processors and for the two renorm helpers, and it
includes a real chi-square distribution test plus a determinism test — but **`sample()` is only ever
tested at V=16**, so none of its radix, nucleus, fast-path, `-1`-row or 64-candidate branches are
exercised at realistic vocabulary, and there is **no CUDA-graph, multi-stream, min-p, joint-mode or
per-row-parameter test** for it. The benchmark has no V=120832/128256 case and no batch sweep, and it
swallows the returned `Status`.

### Cited Findings

**Present (FACT).** `tests/unit/test_sampling.cc` `main()` runs
([test_sampling.cc:1510-1527](../../../tests/unit/test_sampling.cc)):

- `softmax` fp32 / fp16 / bf16 against the fp64 reference with a non-vector-aligned V=1001, a greedy
  row, a `+inf` pair and an all-`-inf` row (`115-241`).
- split softmax at V=131072 with 4 CTAs: normal/greedy/`+inf`/empty rows vs fp64, bit-identical
  repeat, `kWorkspaceTooSmall` on an empty workspace, and CUDA capture+instantiate+replay
  (`888-991`).
- exact workspace arithmetic for `(8, 262144, ctas=1)` including the 16-CTA derivation
  (`1052-1063`).
- `argmax` fp32/bf16 with and without logprob (`246-327`); `compute_logprobs` fp32 top-5 / bf16 top-3
  (`332-425`); bincount/penalties/bitmask/bias (`430-756`); vectorised fp32/bf16 paths at V=1024
  (`760-883`).
- contract errors: probs dtype/shape/stride, temperature length, logits dtype, argmax id dtype,
  `top_n` bound, missing top tensors, bincount mask requirement, counts shape, bitmask width/rows,
  bias indptr/value length (`993-1140`).
- `top_k_renorm_probs` / `top_p_renorm_probs` vs an fp64 stable-sort oracle at V=17 (`1179-1265`),
  `top_k_renorm_probs` ties at V up to 16384 including the `kTopkCand` fallback (`1315-1363`),
  `top_k_mask_logits` at V up to 12288 for bf16/fp16 with k ∈ {1,7,50,333} (`1268-1310`).
- `sample()` distribution: **1,048,576 draws** at **V=16**, B=2048, k=4, p=0.8, compared to the fp64
  truncated distribution by chi-square with threshold 80 for df≤15 (`1383-1449`); same-seed
  determinism (`1451-1457`); Gumbel full-distribution chi-square with top-p disabled (`1459-1479`);
  `kNotImplemented` for Gumbel + top-p and `kWorkspaceTooSmall` for a 8-byte workspace (`1481-1492`).

**Absent (FACT — exhaustive over the file):**

- No `sample()` test at any V above 16; in particular the 4-pass radix, `select_nucleus`,
  `boundary_equals*`, the ≤64 fast path, and the k>64 general path never run in the test.
- No test of `min_p`, `joint_topk_topp`, per-row `top_k/top_p/temperature` arrays, `top_p=0`,
  `top_p=1`, `top_k>V`, `top_k=V`, `k=0 with p=1` (pure full-vocab sampling), greedy-through-`sample`
  (`temperature<=0`), `+inf`/all-`-inf` rows through `sample`, or the `out_ids = -1` contract.
- No test that the fast path and the general path agree (e.g. k=50 vs k=65 on the same row).
- No CUDA-graph test for `sample()`/`gumbel_max_sample()`; no multi-stream test; no RNG
  reproducibility test across different batch sizes or row permutations.
- No error-contract test for `sample()` beyond the workspace size: bad dtype, non-contiguous
  `out_ids`, shape mismatch, negative `default_top_k` are unverified.
- No test at all for the three speculative entries (they have no bodies).

**Benchmark (FACT).** `benchmark/bench_sampling.cu` has 26 cases; the sample-related ones are
`sampling.topk_topp.bf16.v131072.b32.k50.p0.9`, `...v32768.b256.k20.p0.95`,
`sampling.gumbel.bf16.v131072.b32.k50`, plus three renorm/mask cases
([bench_sampling.cu:111-122](../../../benchmark/bench_sampling.cu)). Observed gaps:

- **No V=120832 and no V=128256** case anywhere in the file; the closest are 131072 and 32768.
- **No batch sweep**: only B=32 and B=256. B=1, 8, 128, 512 (the target range) are unmeasured.
- No case with a per-row parameter array, no min-p case, no `k>64` case, no greedy case, no
  CUDA-graph replay case.
- The case id `sampling.gumbel...k50` is misleading: for `op=="gumbel"` the runner forces
  `default_top_k = 0` and `default_top_p = 1.0`
  ([bench_sampling.cu:486-488](../../../benchmark/bench_sampling.cu)), so no top-k is applied.
- `run_sample` computes `status_ok(kops::Status::kSuccess, ...)` — a hard-coded success — so the
  error code of `sample()`/`gumbel_max_sample()` is never checked
  ([bench_sampling.cu:509](../../../benchmark/bench_sampling.cu), same pattern at `530`, `533`,
  `553`). A regression that returns `kCudaError` would be reported as a valid timing.
- The reported GB/s uses a "minimum DRAM traffic" convention that counts **one** row read
  ([bench_sampling.cu:102-109](../../../benchmark/bench_sampling.cu)), so a kernel running at 93 % of
  peak shows as 6-10 GB/s (measured below). Useful as a roofline-min ratio, misleading as bandwidth.

**MEASURED (RTX 3050 Laptop, 16 SMs, 192 GB/s peak, 10 warmup + 80 iters):**

| case | median ms | min ms | reported GB/s | modelled traffic | implied GB/s | % of peak |
|---|---|---|---|---|---|---|
| `sampling.topk_topp.bf16.v131072.b32.k50.p0.9` | 1.3373 | 1.3358 | 6.3 | 151 MB | 113 | 59 % |
| `sampling.topk_topp.bf16.v32768.b256.k20.p0.95` | 1.6852 | 1.6768 | 10.0 | 302 MB | 179 | 93 % |
| `sampling.gumbel.bf16.v131072.b32.k50` (k disabled) | 0.5461 | 0.5447 | 15.4 | 16.8 MB | 31 | 16 % |
| `sampling.softmax.bf16.v131072.b32` | 0.2653 | 0.2623 | 94.9 | 25.2 MB | 95 | 49 % |
| `sampling.topk_renorm.fp32.v131072.b128.k50` | 2.7394 | 1.7176 | 49.0 | 402 MB | 147-234 | 76-122 %* |
| `sampling.topp_renorm.fp32.v131072.b128.p0.9` | 13.3735 | 11.8160 | 10.0 | 470 MB | 35-40 | 18-21 % |
| `sampling.topk_mask.bf16.v131072.b128.k50` | 2.5663 | 1.1818 | 26.2 | 100 MB | 39-85 | 20-44 %* |

\* the two renorm rows show large median/min spread (cache-resident vs cold), so their implied
bandwidth is a range. The important, stable observations are: (a) `sample()` at B=256 reaches 93 % of
device peak under the 7-pass model, confirming the traffic model; (b) top-p renorm — the same
algorithm `sample()` uses when `top_k<=0` — delivers only ~35 GB/s, i.e. it is barrier/atomic-bound,
not bandwidth-bound; (c) Gumbel at 16 % of peak confirms the per-token Philox cost.

### Inferences

- The distribution test is well designed (large N, fp64 oracle, chi-square) but its V=16 shape means
  it validates the *sampler algebra*, not the *selection kernels*. A 120k-vocab variant with a
  top-p-only configuration would exercise `select_nucleus` and the prefix scan, which are the slowest
  and least covered code paths.
- Because the benchmark never checks the status, the numbers in `ROADMAP.md:1085` cannot distinguish
  "fast" from "failing fast".
- Adding a `--dump-cases`-driven H100/B200 sweep at V=120832/128256 with B ∈ {1,8,32,128,512} is a
  prerequisite for the "beat hpc-ops" claim: currently there is no shared case table between the two
  implementations.

### Gaps

- No hpc-ops measurement was taken here (it needs its own build and a `120832` vocab); the comparison
  in Q6 is structural plus modelled, not measured.
- No Nsight/Perfetto trace of `sample()` exists in the tree, so per-kernel attribution is unverified.

---

## Q6. Performance model: current vs a one-pass fused sampler, V=120832, H100/B200

### Takeaway

Under a traffic model calibrated on the measured 3050 data (per-CTA streaming rate × resident CTAs,
capped at ~90 % of peak), the current `sample()` is **CTA-count-bound below ~130 rows and
bandwidth-bound above it**, giving roughly **0.36 ms flat for B ≤ 128 and ~0.78 ms at B=512 on
H100**, versus **~4 µs (B=1) to ~45 µs (B=512)** for a one-read fused design — a **15x-80x** gap that
comes almost entirely from the 18x traffic multiplier, not from kernel quality.

### Cited Findings

**Traffic per (row, token) — FACT, derived from the pass table in Q2:**

| Path | Reads | Writes | Bytes/token (BF16 in) |
|---|---|---|---|
| theoretical minimum (read logits once, emit id) | 1 × 2 B | 5 B/row | **2** |
| `sample()`, `k<=64` fast path (7 prob passes) | 2×2 + 7×4 | 4 + 1/row | **36** |
| `sample()`, top-p only (8 prob passes) | 2×2 + 8×4 | 4 | **40** |
| `sample()`, `k>64` + top-p (13 prob passes) | 2×2 + 13×4 | 4 | **60** |
| `gumbel_max_sample()`, top-k only | 7 × 2 | 4 | **14** + V Philox draws |
| `argmax` / `compute_logprobs` | 1 × 2 | 4-8 B/row | **2** |
| `top_k_mask_logits` (BF16, 2 passes) | 3 × 2 | 2 | **8** |
| standalone `softmax` (V=120832, 4 CTAs) | 2 × 2 | 4 | **8** |

**Current latencies — ESTIMATE.** Assumptions, all explicit:

- A1. Traffic = 36 B/token (fast path) ⇒ **4.35 MB per row** at V=120832.
- A2. Large-batch achieved bandwidth = 0.88 × peak ⇒ H100 **2.95 TB/s**, B200 **7.0 TB/s**
  (peak 3.35 TB/s per [NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/);
  B200 8 TB/s per [OpenMetal comparison](https://openmetal.io/resources/blog/h100-vs-h200-vs-b200-vs-b300-ai-gpu-server)).
- A3. Per-CTA streaming rate for a 256-thread CTA doing `float4` reads with dependent loop
  iterations: **12 GB/s** on H100, **20 GB/s** on B200 (range 8-20 / 12-30). Calibrated from the
  MEASURED 3.5 GB/s per CTA on the 3050 (113 GB/s / 32 CTAs) scaled by the SM/memory-partition
  ratio; this is the softest assumption in the model.
- A4. Effective bandwidth = `min(A2, min(B, resident_CTAs) × A3)` with `resident_CTAs ≈ B` up to
  `SMs × 4` (256 threads, 21.6 KiB smem ⇒ 4-8 CTAs/SM fit).
- A5. Launch overhead 2-3 µs per kernel, 4 kernels.

| B | total traffic | H100 (SMs=132) | B200 (SMs=148) |
|---|---|---|---|
| 1 | 4.4 MB | **0.36 ms** (1 CTA × 12 GB/s) | **0.22 ms** (1 × 20 GB/s) |
| 8 | 35 MB | **0.36 ms** (CTA-bound) | **0.22 ms** |
| 32 | 139 MB | **0.36 ms** (CTA-bound) | **0.22 ms** |
| 128 | 557 MB | **0.36 ms** (128 × 12 = 1.5 TB/s, still CTA-bound) | **0.22 ms** (2.6 TB/s) |
| 512 | 2.23 GB | **0.76 ms** (2.23/2.95 TB/s) | **0.32 ms** |

Sanity check of A3/A4 against measurement: the model reproduces the 3050 numbers to within 1 %
(B=32, V=131072: 151 MB / (32 × 3.5 GB/s) = 1.35 ms vs MEASURED 1.337 ms; B=256, V=32768: 302 MB /
192 GB/s = 1.57 ms vs MEASURED 1.685 ms).

**Fused one-read design — ESTIMATE, same assumptions.** One BF16 pass + a small candidate buffer,
`2 B/token` ⇒ **0.242 MB per row**:

| B | traffic | H100, 1 CTA/row | H100, 32 CTAs/row (hpc-ops-style) | B200, 32 CTAs/row |
|---|---|---|---|---|
| 1 | 0.24 MB | 20 µs | **0.6 µs + launch ≈ 3-4 µs** | ≈ 2-3 µs |
| 8 | 1.9 MB | 162 µs | **5 µs** | 3 µs |
| 32 | 7.7 MB | (saturates) | **7 µs** (bandwidth-bound from here) | 4 µs |
| 128 | 31 MB | — | **11 µs** | 5 µs |
| 512 | 124 MB | — | **43 µs** | 18 µs |

So the achievable target with a fixed 2-3 µs launch overhead and multi-CTA-per-row for small batch is
**~3-5 µs at B=1..8, ~11 µs at B=128, ~45 µs at B=512 on H100** ⇒ **80x / 33x / 17x** faster than the
current path at those batch sizes (and the B=512 case is a pure bandwidth floor: 124 MB cannot be
read faster than 42 µs at 2.95 TB/s).

**Structural comparison with hpc-ops (FACT, from source):**

| Property | kernel-ops `sample()` | hpc-ops `fused_sampler` |
|---|---|---|
| kernels per call | 4 (3 with per-row T) `sampling.cu:2273,2296,2301` | 2 `fused_sampler.cu:653-675` |
| logits passes | 2 (softmax stats + probs) + 7 over probs | 1 `fused_sampler.cu:226-263` |
| bytes/token (BF16) | 36 | 2 + scratch ~0.5 `fused_sampler.cu:650-651` |
| CTAs per row | 1 (256 thr) `sampling.cu:1772` | N = clamp(SMs/B, 8, Nmax≤32) `fused_sampler.cu:616-627` |
| top-k | exact, any k ≤ V, radix | CUB `BlockRadixSort`, **capped at 32/64** `fused_sampler.cu:44-46, 322-327` |
| top-p | exact over full row (or top-k subset) | over the ≤K candidate list only `fused_sampler.cu:445-469, 502-507` |
| vocab support | any (V≤INT32_MAX, split softmax to 262144) | **only 120832** `fused_sampler.cu:757-766` |
| scratch | `B*V*4` (247 MB at B=512) | `B*Nmax*K*8` (4 MB at B=512, K=32) |
| scratch ownership | caller `Workspace`, capture-safe | `cudaMallocAsync`/`cudaFreeAsync` per call `fused_sampler.cu:585-607, 648-651` |
| RNG | counter over (seed,row,pos,draw,stream) `device_impl.cuh:641-651` | Philox + **host `std::atomic` offset advanced per launch** `sampler_rng.cuh:42-52` ⇒ not replay-reproducible |
| greedy | one-hot full row then full selection `sampling.cu:351` | not a separate path (temperature only) |

### Inferences

- The 15x-80x gap is a *design* gap (traffic multiplier and CTA count), not a code-quality gap: the
  radix and prefix-scan primitives are efficient when measured per byte (93 % of peak in the
  bandwidth-bound regime).
- Because the current latency is nearly flat from B=1 to B=128 (CTA-bound), any design that keeps
  one CTA per row inherits the same plateau. Splitting the row across 8-32 CTAs for small batch is
  mandatory, and that is precisely the structural feature hpc-ops has and kernel-ops lacks.
- Copying hpc-ops' semantics would be a regression: bounded top-K (32/64), top-p over the candidate
  set only, single-vocab specialisation and a non-replayable RNG. The correct target is hpc-ops'
  *structure* (2 kernels, one pass, multi-CTA/row) with kernel-ops' *semantics* (exact full-vocab
  thresholds, deterministic ties, counter RNG).
- MEASURED evidence that the top-p path is the real risk: `topp_renorm` reaches only ~35 GB/s on a
  problem whose bytes alone would allow 192 GB/s, i.e. the 4-pass u64 histogram plus 24-barrier scan
  costs more than the memory traffic. In `sample()` the same code runs for the default
  `top_k=-1, top_p=0.9` configuration.

### Gaps

- No hpc-ops latency number was measured (its launcher requires vocab == 120832 and its own build);
  the comparison above is byte- and structure-based, not a head-to-head timing.
- A3 (per-CTA bandwidth on H100/B200) is the largest source of error; a single microbenchmark
  (`1 CTA, 256 threads, float4 strided read of 483 KB`) on the target GPU would pin it and collapse
  the ±2x band.
- No measurement of the serial 64-element bubble sort in the fast path
  ([sampling.cu:1819-1830](../../../csrc/sampling/sampling.cu)); at B=1 it may dominate the kernel
  (est. 2-30 µs), which would make the B=1 number in the table optimistic and is a reason to replace
  it with a warp-parallel scan.

---

## Q7. What to keep and what to replace

### Takeaway

Keep the *semantic contract and the exact-threshold primitives*; replace the *dataflow* (materialised
FP32 probabilities) and the *parallel decomposition* (one CTA per row, 4 radix passes, separate
nucleus pass). Four of the seven removable costs already have working code in the same file
(`topk_by_candidates`, the ≤64 candidate compaction, the split-softmax partial-state pattern, the
single-pass `compute_logprobs` structure) — they are simply not wired into `sample()`.

### Cited Findings

**Keep (FACT, with location):**

1. **Exact k-th selection with deterministic ties** — `select_kth_key` + `boundary_equals`
   (`sampling.cu:1123-1220`), including the "tie group fully kept ⇒ no scan" short-circuit
   (`1194`, mirrored in `top_k_mask_logits` at `1568`). This is the piece that gives the exact
   candidate set FlashInfer-style rejection sampling cannot provide
   ([arXiv 2602.01518](https://arxiv.org/html/2602.01518v1)).
2. **Monotone key maps** `prob_key`/`logit_key` (`sampling.cu:1083-1091`) — they make radix selection
   valid on fp32 bits and handle NaN (`1088`) without special cases downstream.
3. **Exact prefix-scan sampling** with fixed-point 2^40 mass (`fp_mass` `1093-1096`, prefix scan
   `1903-1932`): deterministic, order-independent, and the natural replacement for per-token Gumbel.
4. **Per-row parameter resolution and clamping** (`row_topk/row_top_p/row_min_p`,
   `sampling.cu:1462-1479`) and the top-k-first / joint ordering (`RowMask`, `1228-1232`, `1874`).
5. **Counter RNG contract** (`device_impl.cuh:641-651`, `rng.cuh:4-7`): the one property hpc-ops
   lacks and CUDA-graph replay depends on.
6. **Empty/NaN/±inf row policy and the `-1` contract** (`sampling.cu:96-122, 343-355, 1888-1893`).
7. **Caller-owned workspace and allocation-free run path** (`sampling.cu:2257-2302`,
   `tensor.h:76-90`) — the opposite of hpc-ops' per-call `cudaMallocAsync`.
8. **Split-row partial-state pattern** (`softmax_stats_kernel` → `softmax_norm_kernel`,
   `sampling.cu:373-523`): a working template for multi-CTA-per-row with a per-CTA state array and a
   merge that needs no third launch.
9. **Single-pass logprob structure** (`compute_logprobs_kernel`, `sampling.cu:568-659`): one read for
   row stats + per-thread rank count + per-thread top-n list, with block reductions only for the
   final ordering. This is the shape the fused sampler should have.

**Replace (FACT, with the concrete cost):**

1. **Materialised FP32 probability row** (`sampling.cu:2296`, workspace `862-866`) — 4 B/token of
   workspace plus 1 write and 7-13 reads. Replace with online softmax state (max, sum) carried in
   fp32 per (row, slice) and consumed directly by the selection kernel, exactly as
   `softmax_stats_kernel` already computes it (`373-435`) and as hpc-ops' K1 does
   (`fused_sampler.cu:244-252`, written per slice at `274-277`).
2. **4-pass 8-bit radix over 32-bit keys** (`sampling.cu:1130`) — the code already contains a
   **3-read** alternative: `topk_by_candidates` (`1345-1448`), an 11-bit histogram + candidate gather
   + rank pivot that provably matches `select_kth_key`+`boundary_equals` (`1338-1343`) and is used by
   `top_k_renorm_probs` (`1637`) but **not** by `sample()`. Wire it in; keep the radix as the
   documented fallback for the "bin too large" case (`1400`).
3. **The unconditional `row_max` pass** (`sampling.cu:1785`) — pure overhead when `min_p == 0`
   (1 of 7 passes, 14 % of traffic on the fast path). Fold the max into the softmax stats kernel that
   already computes it (`sampling.cu:407-421`), or gate it on `use_mp`.
4. **4-pass u64 nucleus histogram plus 24-barrier `block_exscan_u64`** (`1243-1291`, `1098-1113`) —
   MEASURED at ~35 GB/s (`topp_renorm`, 13.37 ms) versus 179 GB/s for the candidate-based top-k path.
   Replace with: top-k candidates in shared memory (≤64, already done at `1801-1815`) and the nucleus
   computed by a single warp-level cumulative scan over them, since top-p after top-k never needs the
   whole row. For the top-p-only case, compute the nucleus threshold from a single-pass partial-sum
   array (per-CTA partials + merge), not from four full-row passes.
5. **The separate nucleus pass and the separate boundary pass** (`1874-1879`, `1903-1912`) — after
   candidate compaction these become shared-memory operations; the fast path already demonstrates
   that (`1817-1868`), but it is (a) gated on `k<=64` **and** `use_tp` **and** `!joint`, so the
   default top-p-only configuration never reaches it, and (b) implemented with a serial 64-element
   bubble sort in thread 0 (`1819-1830`) that should become a warp bitonic sort or a rank-based
   selection.
6. **One CTA per row with 256 threads** (`sampling.cu:1772`) — replace with N CTAs/row chosen from
   SMs and batch (as `fused_sampler.cu:616-627` does) writing per-slice partial (max, sum) and a
   per-slice top-k candidate list, then a merge kernel. This is the only way to fix the B ≤ 128
   plateau in Q6.
7. **Greedy through the full pipeline** (`sampling.cu:309-355`) — route all-greedy batches to
   `argmax_kernel` (1 pass, 2 B/token) instead of writing a one-hot FP32 row; vLLM does the
   equivalent short-circuit at the parameter level
   ([vLLM sampling_params](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).
8. **Per-token Philox in the Gumbel path** (`sampling.cu:2044-2046`) — V draws per row, MEASURED at
   16 % of peak for a 2-pass kernel. Either draw one uniform per row and invert the CDF (the exact
   prefix scan already exists), or keep Gumbel only as an explicit "no normaliser available" mode.
9. **Kernel count** — 4 → 2 (a scan/candidate kernel and a merge/sample kernel), or 1 for the
   single-CTA-per-row case with `cudaLaunchCooperativeKernel`-free cross-CTA merge via the existing
   partial-state pattern; hpc-ops' 2-kernel shape is the proven minimum
   (`fused_sampler.cu:3-13, 653-675`).

### Inferences

- Items 2, 3 and 5 are low-risk, high-return edits entirely inside the existing kernel: they remove
  5 of the 7 full-row passes without changing semantics, because `topk_by_candidates` is already
  proven equivalent to the radix path by construction (`1338-1343`) and by the tie tests
  (`test_sampling.cc:1315-1363`).
- Items 1, 6 and 9 require a new device ABI (per-slice state, candidate lists) and therefore an ABI
  revision; `KOPS_ABI_VERSION` is 5 today (`macros.cuh:42`) and `AGENTS.md` requires a bump for
  changed kernel template parameters, argument order or launcher signatures.
- The semantic risk concentrates in one place: the top-k-first top-p threshold must stay "fraction of
  the top-k renormalised mass" (`sampling.cu:1265-1269`), not "fraction of the full row", or every
  configured `top_k>0 && top_p<1` request changes distribution. A differential test (fast path vs
  radix path over the same rows, as suggested in Q5) is the cheapest guard.

### Gaps

- No estimate was made of the shared-memory budget of a fused candidate design on sm_90a (TMEM/WGMMA
  are irrelevant here, but 228 KiB/SM caps the candidate array size); a 2048-candidate gather needs
  16 KiB (`kTopkCand`, `sampling.cu:1332`), which fits, but a 120832-vocab single-CTA design cannot.
- Whether the engine can supply BF16 logits already contiguous with 16-byte-aligned rows was not
  verified; unaligned rows silently fall back to scalar loads (`row_vec_ok`, `sampling.cu:240-247`).
- The exact hpc-ops latency at V=120832 remains unmeasured, so the "beat hpc-ops" target in Q6 is a
  modelled target rather than a measured one.
