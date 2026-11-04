# Fused LLM Sampling Pipelines: SOTA Survey and Concrete Design Options for kernel-ops

Scope: end-to-end sampling pipeline design (not single-kernel micro-optimisation) for
`V ∈ {120832, 128256, 151936}`, `B ∈ [1, 512]`, BF16/FP32 logits, H100 (SM90) / B200 (SM100),
CUDA-graph-captured decode loops, optional speculative decoding. Target to beat:
`3rd/hpc-ops/src/sampler` (2-kernel, bounded top-32/64, vocab specialised to 120832).

All numbers below are labelled **[measured]** (someone ran it and published it), **[derived]**
(arithmetic from a cited quantity, no measurement), or **[estimate]** (a model that still needs
a benchmark to confirm). Nothing in this note was produced by running GPU code: no repository
code was modified and no benchmark was executed.

---

## Q1. How do vLLM (V1), SGLang, TensorRT-LLM and FlashInfer structure their sampling pipelines?

### Takeaway
Every production stack converged on the same shape — *penalties → temperature → filter → draw* —
but they disagree on the two things that decide cost: **whether an fp32 probability row is
materialised**, and **whether the filter is implemented as a sort, a radix threshold, or
rejection sampling**. Only TensorRT-LLM runs the whole chain in **one kernel launch**; vLLM's
newest GPU path avoids probabilities entirely by drawing Gumbel noise per token from a
counter-based hash.

### Cited Findings

**vLLM V1 (`vllm/v1/sample/`)**

- The V1 sampler pipeline is documented in-order in the module docstring: logprobs capture,
  fp32 conversion, allowed-token mask, bad-words, non-argmax-invariant processors (min-tokens,
  logit bias), penalties (repetition/frequency/presence), then greedy-or-random with
  temperature → min-p → top-k/top-p → draw — `3rd/vllm/vllm/v1/sample/sampler.py:22-60`.
- Each stage is a separate PyTorch op on a `[B, V]` tensor: `logits.div_(temp.unsqueeze(1))`
  (in-place), then processor `.apply(logits)` calls, then the fused top-k/top-p stage —
  `3rd/vllm/vllm/v1/sample/sampler.py:278-294`.
- The fused stage (`TopKTopPSampler`) has **four backends**: native, CUDA, ROCm/aiter, XPU,
  plus CPU — `3rd/vllm/vllm/v1/sample/ops/topk_topp_sampler.py:296-447`. The CUDA path calls
  `flashinfer_sample`, which requires contiguous logits and **rejects rows that need
  `processed_logits`/`processed_logprobs`** — `topk_topp_sampler.py:307-313`.
- The PyTorch-native path is `apply_top_k_top_p` → sort the whole vocab → mask by the k-th
  value → softmax → cumsum → mask by `cumsum <= 1-p` → scatter back, then `random_sample` —
  `topk_topp_sampler.py:473-514`, `546-577`. This is O(V log V) per row and materialises probs.
- vLLM's own PR replacing that sort with FlashInfer's kernel states the sort "can take ~20%
  performance overhead in high-throughput settings" and reports ShareGPT throughput going from
  40.20 reqs/s (with top-p, main) to 47.15 reqs/s (with the fused kernel) against 47.40 reqs/s
  with no top-p/top-k at all — [vLLM PR #11394](https://github.com/vllm-project/vllm/pull/11394).
- `random_sample` is the exponential race, not inverse-CDF: `q.exponential_()`, then
  `probs.div_(q).argmax(-1)` — `topk_topp_sampler.py:546-552, 555-577`. The docstring says
  this exists "instead of torch.multinomial because torch.multinomial causes CPU-GPU
  synchronization" — `topk_topp_sampler.py:560-564`.
- Newest path (`vllm/v1/worker/gpu/sample/gumbel.py`): temperature is a separate in-place
  Triton kernel (`_temperature_kernel`, one program per `(token, 8192-wide block)`),
  then `_gumbel_sample_kernel` computes a **per-block** `argmax(logits/T + Gumbel)` with a
  1024-element block, and the final token is picked by `local_max.argmax(-1)` + `gather` —
  [vllm/v1/worker/gpu/sample/gumbel.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py).
  That is **4 kernel launches, zero probability materialisation, exact full-vocab Gumbel-max**.
- That path's noise is `murmur3_hash32(seed, pos, token_offset)` → `double hashing` for fp64 —
  i.e. **one hash per (request seed, step position, token id)**, no RNG state
  ([gumbel.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py)).
  A `_DRAFT_NOISE_SALT = 1<<30` keeps a draft's noise disjoint from the target's so that
  "a proposal and the residual it is resampled from must not share a noise vector".
- vLLM's spec-decode verification is a separate Triton kernel `rejection_random_sample_kernel`
  doing the standard `target_prob / draft_prob >= uniform` test, with a pre-generated
  `uniform_probs` buffer and a `recovered_token_ids` fallback —
  `3rd/vllm/vllm/v1/sample/rejection_sampler.py:829-901`.

**SGLang**

- `Sampler.forward` preprocesses logits, early-exits to `argmax` when all rows are greedy, and
  special-cases "no top-k/top-p/min-p at all" into a plain multinomial over probs —
  `3rd/sglang/python/sglang/srt/layers/sampler.py:103-166`, `189-225`.
- The standard path is `logits.div_(temperatures)` **in place**, then `torch.softmax(logits,
  dim=-1)` writing an fp32 `[B, V]` probability tensor, then a fused FlashInfer kernel
  `top_k_top_p_sampling_from_probs` (or `top_k_renorm_prob` + `top_p_renorm_prob` +
  `min_p_sampling_from_probs` when min-p is on) —
  `3rd/sglang/python/sglang/srt/layers/sampler.py:154-166`, `213-222`.
- So SGLang is **≥3 kernels and always materialises probs**; the batch-level need flags
  (`need_top_p_sampling`, `need_top_k_sampling`, `need_min_p_sampling`) are computed by scanning
  the request set on the host —
  `3rd/sglang/python/sglang/srt/sampling/sampling_batch_info.py:177-179`.
- SGLang's SGL kernel tree has **no sampling kernel of its own**; `sgl-kernel/.../topk.cu` is an
  Indexer/DeepSeek-V3.2 sparse-attention top-k
  (`3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu:90-215`, `254-268`) and
  `speculative/speculative_sampling.cu` is the spec-decode path. Sampling proper is delegated
  to FlashInfer (`3rd/sglang/sgl-kernel/python/sgl_kernel/sampling.py:7-11, 53-59`).

**TensorRT-LLM (`cpp/tensorrt_llm/kernels/fusedSampling/`)**

- One fused kernel applies temperature, min-p, top-k **and** top-p together, deciding per row on
  device which filters need work — `3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/fusedSampling/fusedSamplingKernels.h:35-48`.
- The unifying observation is stated in the header: after the softmax's max-reduce,
  `w = exp(l/T - max) = p / p_max`, so all three filters are thresholds on `w` — min-p is a
  constant, top-k is a threshold found *by rank*, top-p is a threshold found *by mass* — same
  file `:43-48`.
- Threshold search is MSB-first radix over the float bit pattern, **8 bits × 4 passes, resolving
  the boundary value exactly** — `fusedSamplingKernelsCommon.cuh:285-307`.
- Contention is handled with **16 private histogram copies** (`kHistCopies = 16`,
  `fusedSamplingKernelsCommon.cuh:48`) because "w lies in (0,1], so the top 8 bits only ever
  take ~64 of the 256 values, and a softmax concentrates most of a 128k-element row into a
  handful of those" — `:322-331`.
- Once the crossing bucket holds ≤ `kCandCap = 2048` elements, the survivors are **compacted
  into shared memory and the remaining radix passes never touch the row again** —
  `fusedSamplingKernelsCommon.cuh:51, 491-523`.
- For tokens-only calls the exact cutoff is skipped entirely in favour of **rejection sampling**:
  "The exact cutoff exists to *materialize* the filtered distribution. A token drawn from that
  distribution does not need it: sample from any superset of the kept set, test whether the draw
  landed inside, and on a rejection shrink the superset to what the test just measured" —
  `fusedSamplingKernelsCommon.cuh:756-772`, loop at `:800-900`, capped at
  `kMaxRejectRounds = 32` (`:61`). Pure top-k is deliberately kept out because "on a flat 131k
  row, drawing one of the top 50 would almost never succeed" (`:765-769`).
- RNG is Philox-4x32-10 with `curand_init(seed, subsequence, offset)`; when per-row seed/offset
  tensors are supplied the row's **own subsequence** is used (`kPerRowSubsequenceBase = 1<<63 |
  offset`) so draws "depend only on its own seed/offset, not on its position in the batch" —
  `fusedSamplingKernelsCommon.cuh:132-146`, `fusedSamplingKernels.h:29-33, 65-75`.
- The seed/offset are **tensors, not scalars**, explicitly "so the call is legal inside a
  captured CUDA graph" — `fusedSamplingKernels.h:65-67`.
- Launch policy: one CTA per row normally (`kNarrowBlock = 512`, `kWideBlock = 1024`); a
  multi-CTA family (8 CTAs/row, 512 threads) only when `outputProbs != nullptr && numRows <= 32
  && vocabSize >= 65536` — `fusedSamplingKernelsInternal.h:28-47`.
- Optional `outputProbs` writes the filtered, renormalised distribution — "what the rejection
  path needs and what a sort-free sampler cannot hand back" —
  `fusedSamplingKernels.h:79-82`.

**FlashInfer**

- Sampling under top-k/top-p/min-p is done **sorting-free** with *Dual Pivot Rejection Sampling*
  in a single fused kernel; the pivot range `[low, high]` is halved each round, giving
  `O(log(1/eps))` rounds in the worst case —
  [FlashInfer blog, 2025-03-10](https://flashinfer.ai/2025/03/10/sampling.html).
- The blog states the exactness theorem: `Pr[output = j] = p_j / Z` over the top-k set
  (identical to discarding then sampling categorically) —
  [same](https://flashinfer.ai/2025/03/10/sampling.html).
- It warns that the parallel prefix sum is the hard part: "due to the non-associative and
  non-commutative nature of floating-point arithmetic, parallel prefix-sum cannot guarantee
  monotonic outputs even with non-negative inputs. This can lead to invalid token generation"
  — [same](https://flashinfer.ai/2025/03/10/sampling.html).
- Rounds are bounded in practice, not in theory: "expected to stop within log(N) rounds … in my
  experience it converges much faster (4-6 rounds)" — maintainer comment,
  [flashinfer#710](https://github.com/flashinfer-ai/flashinfer/issues/710).
- The API exposes a `deterministic: bool = True` switch, `seed`/`offset` **tensor** parameters,
  and an `indices` argument so one probability row can serve several outputs; the docs note
  "Using torch.Tensor is required for CUDA graph compatibility" —
  [flashinfer.sampling docs](https://docs.flashinfer.ai/api/sampling.html),
  [top_k_top_p_sampling_from_probs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html).
- A logits-input variant exists (`top_k_top_p_sampling_from_logits`) that avoids materialising
  probs — [docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html).
- NVIDIA's own summary: "Traditional implementations sort the whole vocabulary, which is
  wasteful work when only a handful of logits matter. FlashInfer replaces global sorts with a
  rejection-based, sorting-free sampler" —
  [NVIDIA developer blog, 2025-06-13](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/).

**hpc-ops (the baseline to beat)**

- 2 kernels. K1 `grid=(N,B) block=1024`: one vocab pass doing rp+temp → online max/sum →
  **`cub::BlockRadixSort` top-K inside the load loop**; K2 `grid=(B,) block=Nmax*K`: merge,
  softmax/topp, Gumbel-max, penalty writeback — `3rd/hpc-ops/src/sampler/fused_sampler.cu:1-13`.
- It is explicitly a bounded sampler: "the kernel always collapses the vocab to the top-`max_topk`
  candidates before sampling. topk=0 (or unset) does NOT mean full-vocab sampling … Low-probability
  tokens outside the top-`max_topk` are never drawn" — `sampler.h:17-20`.
- Vocab is a compile-time template argument with a single accepted value (`case 120832`, else
  `throw`) — `fused_sampler.cu:757-766`.
- Scratch is `cudaMallocAsync`/`cudaFreeAsync` **per call** (4 buffers) —
  `fused_sampler.cu:585-607, 648-651`.
- RNG offset is a **host-side process-global atomic counter advanced once per launch** —
  `sampler_rng.cuh:6-13, 42-52`.

### Inferences
- The field has split into two camps: **"materialise probs, then sample"** (vLLM native,
  SGLang, FlashInfer-from-probs) and **"never materialise, threshold or race directly on
  logits"** (TRT-LLM, vLLM's new Gumbel path, hpc-ops). The second camp is the one that scales
  with `B·V`, and it is where kernel-ops must land.
- Kernel-count ranking at fixed semantics: TRT-LLM (1) < hpc-ops (2) < SGLang (3-4) ≈
  vLLM-gumbel (4) < vLLM-native (5-6, plus a sort).
- Only TRT-LLM attempts *exact, full-vocab, all-filters* in one kernel; everyone else either
  sorts, materialises probs, or bounds the support.

### Gaps
- No public, like-for-like latency table exists for {vLLM V1 gumbel, vLLM+FlashInfer, SGLang,
  TRT-LLM fused, hpc-ops} at `V=120832, B∈{1..512}` on H100/B200. Any comparison in the final
  report must be one we run, not one we quote.
- vLLM's `gumbel.py` path does not show top-k/top-p being applied; how the new GPU runner
  reconciles `top_p < 1` with a pure Gumbel argmax is not visible from the file alone. Treat
  "vLLM dropped top-p" as **unverified** — it is more likely that filters are applied earlier
  as a `-inf` mask, which would make the Gumbel argmax exact over the masked set (see Q2).

---

## Q2. Sampling methods compared: Gumbel-max, rejection, inverse-CDF, exponential race

### Takeaway
All four are **exact** when run to completion; they differ in what they require (a normaliser, a
sorted support, a threshold) and in per-element cost. Gumbel-max and the exponential race are
algebraically identical and need **no normaliser and no prefix sum**, which makes them the only
methods that are simultaneously exact, order-independent and cheap on the full vocab.
Inverse-CDF is the only one that yields the sampled token's own probability for free
(needed by spec-decode acceptance and logprobs) but is order-sensitive.

### Cited Findings
- **Gumbel-max ≡ categorical.** vLLM's CPU path documents the exponential form as exactly
  equivalent: `argmax(p_j / q_j)` with `q ~ Exp(1)` "selects j with probability p_j
  (== softmax)" — comment in the mudler/vllm.cpp port of vLLM's sampler,
  [src/vt/cpu/cpu_sample.cpp](https://github.com/mudler/vllm.cpp/blob/fc636c76/src/vt/cpu/cpu_sample.cpp)
  (secondary source; the same identity is what `topk_topp_sampler.py:546-552` implements).
- **Gumbel noise numerics.** vLLM draws the winning tail from `u → 0`: "Draw the large-noise
  tail (which decides the argmax winner) from u → 0, where fp32 has fine resolution", and uses
  a degree-8 `log1p` series for `-log(1-u)` with absolute error < 6e-7 on `[0, 0.25]` —
  [gumbel.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py).
  fp64 is offered but "~1/32-1/64x the throughput on H100/Ada/Blackwell and empirically
  indistinguishable for Gumbel-max".
- **hpc-ops clamps the other tail**: `inner = max(-log(U), 1e-20)` then `-log(inner)`, "guard
  U→1" — `3rd/hpc-ops/src/sampler/sampler_rng.cuh:34-40`. This caps the maximum Gumbel at
  ≈ 46.05, i.e. a token more than ~46 logits below the max can never win. That is a *safe*
  bound for fp32 logits in practice but it is an approximation that must be documented.
- **Rejection sampling is exact and sorting-free but unbounded in theory**: dual-pivot halves
  the range each round ⇒ `O(log(1/eps))`; observed 4-6 rounds
  ([FlashInfer blog](https://flashinfer.ai/2025/03/10/sampling.html),
  [flashinfer#710](https://github.com/flashinfer-ai/flashinfer/issues/710)).
- **Rejection's numerical hazard is the prefix sum**, not the acceptance test
  ([FlashInfer blog](https://flashinfer.ai/2025/03/10/sampling.html)). TRT-LLM works around it
  by scanning with the *same traversal and the same order* for the mass and for the crossing
  ("Same traversal as the accumulation above, so the two float sums agree term for term") —
  `fusedSamplingKernelsCommon.cuh:836-841`, and still has an explicit `sCandIdx < 0` fallback
  when the target lands past the last element (`:846-850`).
- **kernel-ops already solves the prefix-sum determinism problem** with fixed-point mass:
  `fp_mass()` scales by `2^40` into `uint64_t` and scans with an exact `uint64` block scan, so
  the sum is exact and therefore order-independent — `csrc/sampling/sampling.cu:1093-1119`.
- **k-th selection tie semantics**: vLLM's mask uses `logits < top_k_mask` (strictly less), so
  **all elements tied at the k-th value are kept** — `topk_topp_sampler.py:500-502`. kernel-ops
  matches this with `select_kth_key` returning `need` (how many elements share the boundary key)
  plus `boundary_equals` — `csrc/sampling/sampling.cu:1124-1126 (select_kth_key) / 1189 (boundary_equals)`.
- **Cost per row (derived).** With `V = 120832` and 2 B/elem BF16, one row is 236 KiB. Boundary
  SFU throughput on H100 (132 SM × 16 MUFU/clk × ~1.8 GHz) is ≈ `3.8e12` ops/s, so
  `V = 120832` elements cost ≈ **0.032 µs per transcendental-op per element, per row**, i.e.
  at `B = 512` one `exp`/`log` per element ≈ **16 µs** of pure SFU time versus **37 µs** of
  DRAM time at peak. Two transcendentals per element per pass ≈ 32 µs ≈ at parity with the
  memory pass; three or more makes the kernel SFU-bound. **[derived, must be measured]**

### Inferences
- Because `w = exp(l/T - max) ∝ p` and any filter is a `w ≥ t` test, **Gumbel-max over the
  masked logits is exactly categorical over the renormalised truncated distribution** — no
  renormalisation, no normaliser, no second read is needed *for the draw*. This is why
  TRT-LLM's `outputProbs` comment says a sort-free sampler "cannot hand back" the distribution.
- Therefore the only reason to compute `Z` at all is **top-p/min-p threshold search**, and the
  only reason to compute a prefix sum is **producing `p` for logprobs or spec-decode**. A
  design that disentangles "draw" from "report p" is strictly better than one that always
  materialises.
- The exponential race (`argmax w_j / q_j`, `q = -log u`) is preferable to the Gumbel form
  because it never forms `log p` (no `-inf` from underflow: `w` can be 1e-38 and still divide
  fine) and it consumes one uniform per *candidate*, not per *token in the vocab* when the
  candidate set has already been compacted.
- Rejection sampling's real cost is not the acceptance test, it is that **each round re-reads
  the row** unless the support has been compacted; that is why TRT-LLM only uses it for
  tokens-only rows and only after `findThreshold` has already narrowed the support.

### Gaps
- No published per-row cost breakdown (µs) for Gumbel vs rejection vs inverse-CDF at
  `V = 120832` on H100. The 4-6-round figure is anecdotal and distribution-dependent.
- Whether `-log(-log(u))` in fp32 introduces measurable distribution bias at `V = 120832` is
  not published; needs a chi-square test (see Q7).

---

## Q3. Design space for kernel-ops, option by option

### Takeaway
Five options were evaluated. Option (a) — *one CTA per row, two passes over the logits,
threshold in pass 1, compact-and-race in pass 2, no fp32 probs* — dominates on the
traffic/robustness/implementability frontier. hpc-ops' one-pass-plus-sort (option b) is
attractive only at very small `B`, where a split-row variant of (a) recovers the same
traffic anyway. Sorting (option d) is strictly dominated.

### Cited Findings — baseline reality check

- **hpc-ops K1 does one `BlockRadixSort` per load loop**, not one per launch:
  `__syncthreads(); BlockRadixSortT(...).SortDescending(...); __syncthreads();` sits *inside*
  `for (iloop < num_load_loops)` — `3rd/hpc-ops/src/sampler/fused_sampler.cu:225-263`.
- The loop count follows `elements_per_load_loop = N * kLoaderThreads * kItemsPerThread` with
  `kLoaderThreads = 1024 - kMaxTopK/4 = 1016` — `fused_sampler.cu:177-178, 222-223`. For
  `V = 120832, N = 8`: `8·1016·4 = 32512`, so `ceil(120832/32512) = 4` load loops ⇒
  **4 full 4096-key block radix sorts per block** **[derived]**.
- `N` is chosen as `clamp(SMs/B, 8, Nmax)` with `Nmax = min(1024/K, 32)` —
  `fused_sampler.cu:44-46, 616-627`. At `B = 512` and 132 SMs this gives `N = 8` ⇒
  `4096` blocks, and the source states K1 is "limited to 1 block/SM" ⇒ **31 waves**
  (`fused_sampler.cu:610-614`) **[derived]**.
- K2 merges with a `BlockRadixSort` over `Nmax*K = 1024` keys and then **only warp 0 samples**
  (`if (tid >= 32) return;`) — `fused_sampler.cu:355-402`.
- **Current kernel-ops** materialises the fp32 row: `softmax(probs, logits, temperature, …)`
  then `top_k_top_p_sampling_kernel<1,8><<<batch, kRowThreads, …>>>` —
  `csrc/sampling/sampling.cu:2296-2302`, with `kRowThreads = 256`
  (`csrc/sampling/sampling.cu:1080`). Workspace is `B·V·4` bytes plus state —
  `csrc/sampling/sampling.cu:859-864`.
- It already has the two hard pieces: deterministic fixed-point mass (`fp_mass`, 2^40 scale,
  `uint64` scan — `sampling.cu:1093-1119`) and a **compaction fast path (≤64 candidates) that
  samples entirely in shared memory** — `sampling.cu:1799-1801`.
- AIR Top-K is an *iteration-fused* radix design with an adaptive strategy that "eliminates
  unnecessary device memory traffic automatically under various data distributions", claiming
  1.98-21.48× (B=1) and 8.01-574.78× (B=100) over previous radix top-k —
  [SC'23 paper page](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html).
- RadiK targets scalability in `k` with 2.5×/4.8× over prior art and +2.7× on adversarial
  distributions via input scaling — [ICS'24](https://dl.acm.org/doi/10.1145/3650200.3656596).
- RTop-K's survey notes Dr. Top-k / AIR Top-k / RadiK are "optimized for limited batches of
  large vectors … vector size on the order of 2^20 … batch size does not exceed 100", which is
  **not** the decode-sampling regime (`V ≈ 2^17`, `B` up to 512) —
  [arXiv 2409.00822 §2.3](https://arxiv.org/html/2409.00822v3).
- TRT-LLM proves the "one global read + on-chip descent" structure is viable: after the first
  histogram pass, if `count(chosen bucket) ≤ kCandCap = 2048` the survivors are copied to
  shared memory and the remaining passes never touch the row again —
  `fusedSamplingKernelsCommon.cuh:491-523`.

### Option table

`V = 120832`, BF16 logits — row = 236 KiB, 0.2417 MB.

| Option | Passes over logits | Bytes/row moved | Shmem | Launches | Exactness | Graph safety | Risk |
|---|---|---|---|---|---|---|---|
| **(a) 2-pass online-softmax + 8-bit radix threshold + smem compact + race** | 2 | 0.483 MB (2×236 KiB) + `V/8` B bitmask | 32-64 KiB (16 hist copies × 256 × (int+float)) | **1** | exact (filters), tie-group = vLLM semantics | high (no allocs, fixed grid) | medium: needs fallback when the crossing bucket > `C` |
| (a') same, but the 2nd pass hits L2 | 2 (2nd from L2) | ≈ 0.25-0.30 MB effective | same | 1 | exact | high | low |
| **(b) hpc-ops-style block-topK + merge** | 1 | 0.242 MB + 4 MB scratch @B=512 | BlockRadixSort temp (~32-64 KiB) | 2 | **bounded to K≤64** | medium (per-call `cudaMallocAsync`, cross-stream unsafe) | low algorithmic, high semantic |
| (c) persistent/cooperative single kernel | 1-2 | same as (a) | same + grid-sync cost | 1 | exact | cooperative launch not capturable as a normal graph node in all toolchains | high |
| (d) CUB BlockRadixSort / DeviceSegmentedRadixSort | 1 + sort | 0.242 MB + `B·V·8` for sorted (k,v) pairs = 990 MB @B=512 | 32-64 KiB | 1-2 | exact but O(V log V) | ok | low, but dominated |
| (e) top-k-bounded sampling | 1 | 0.242 MB | small | 1-2 | **not** full-vocab | ok | **unacceptable as a default** |

**Notes on the table**

- (a) is the only row that is exact, single-launch, allocation-free, and within 2× of the
  single-pass byte floor. Its 2nd pass is the whole design's cost and the whole design's
  opportunity: for `B ≤ ~128` the row set (≤ 31 MB) fits in H100's 50 MB L2, so the second pass
  is an L2 read, not a DRAM read. **[derived; L2 size is spec, hit-rate must be measured]**
- (b) matches (a) on traffic but pays 4 × `BlockRadixSort(4096 keys)` per block. A CUB block
  radix sort of 4096 keys is roughly 1-3 µs of SM time **[estimate]**; at `B = 512` that is
  `31 waves × 4 sorts × 1-3 µs ≈ 124-372 µs` of pure sort time on top of 37 µs of memory time
  **[estimate, needs `ncu` confirmation]**. If that estimate is even half right, hpc-ops is
  sort-bound at large `B`, which is the single biggest lever kernel-ops has.
- (c) is rejected: `cudaLaunchCooperativeKernel` + `grid.sync()` is not cheaper than a second
  launch, and capture support is toolchain-dependent. The *useful* part of (c) — "last CTA does
  the merge" — is already proven in hpc-ops' temperature kernel (`atomicAdd` ticket, last block
  reduces N partials — `3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu:8-13`) and should
  be borrowed without the cooperative-launch machinery.
- (d) is dominated: sorting writes back `B·V·(4+4)` bytes. At `B=512` that is 990 MB ≈ 300 µs
  on H100 **[derived]**, 8× the read floor.
- (e) is what hpc-ops does and what we must not do by default; it is still worth keeping as an
  explicit "bounded/fast" mode so that the benchmark can compare like-for-like (see Q7).

### Inferences
- The cheapest correct decomposition is: **pass 1** = `max`, online `sum`, 8-bit histogram of
  the monotone logit key, and argmax (1 `exp`/element, 2 shared-memory atomics/element);
  **descent** = arithmetic on 256 counters (free); **pass 2** = recompute `w`, keep survivors of
  the crossing bucket in shared memory, build the sub-histogram (1 `exp`/element); **finish** =
  two more 8-bit rounds over ≤ 2048 shared-memory items, then a race over the final tie-group.
  Total: **2 `exp`/element**, 2 DRAM reads, 1 launch, 0 global scratch for `B ≥ 132`.
- For `B < 132`, splitting a row across `S` CTAs and merging in the last CTA (atomic ticket)
  keeps the launch count at 1 while restoring occupancy; each split CTA writes its survivors to
  the caller's workspace, so the merge CTA never re-reads the row. This is hpc-ops' K1→K2 shape
  with a threshold instead of a sort, and it is *one* launch instead of two.
- Because a row's `w` values live in `(0,1]`, the top-8-bit histogram only uses ~64 of 256 bins
  (`fusedSamplingKernelsCommon.cuh:322-327`). Two consequences: (i) 8 bits is coarse for the
  *first* mass bucket, so a second 8-bit pass is usually needed; (ii) histogram contention is
  the bottleneck, not histogram size, so 8-16 private copies are mandatory.
- BF16 logits carry only 7 mantissa bits, so an exact threshold on the BF16 key needs at most
  16 bits = **2 radix passes**, versus 4 for FP32. A BF16 fast path that skips the last two
  rounds is a real (if modest) win, and is only correct if the tie-group semantics above are
  preserved. **[derived]**

### Gaps
- No measurement of hpc-ops K1 alone (kernel time, sort fraction, achieved bandwidth). Until
  that exists, "hpc-ops is sort-bound" is an estimate and must be labelled as such.
- No measurement of how often the crossing bucket exceeds 2048 entries on real model logits
  (this determines whether the shared-memory compaction path is the common case or the
  exception). TRT-LLM's `kCandCap = 2048` suggests 2048 is a reasonable bet but is not evidence.

---

## Q4. Memory-traffic lower bound and achievable latency

### Takeaway
At `V = 120832` BF16 a row is 236 KiB. The single-pass floor is 37 µs at `B = 512` on H100
(15.5 µs on B200) and 0.07 µs at `B = 1` — i.e. **at `B ≤ 8` everything is launch-latency and
occupancy bound, and at `B ≥ 128` everything is bandwidth bound**. The current kernel-ops path
moves ~5× the floor because it materialises fp32 probs; hpc-ops moves ~1.04× the floor but pays
for it with a bounded support and a sort inside the read loop.

### Cited Findings

- H100 SXM HBM3 bandwidth 3.35 TB/s and B200 8 TB/s are the figures given in the assignment
  brief (treated as spec, not measured here).
- H100 L2 is 50 MB; B200/GB100 L2 is ~126 MB (vendor spec; must be re-confirmed against the
  exact SKU used for benchmarking). **[estimate — verify against the SKU]**
- kernel-ops' `sample()` requires `B·V·4` bytes of workspace for the probability row plus
  `B·ctas·2·4` for split state — `csrc/sampling/sampling.cu:859-864`.
- The implemented `softmax_kernel` body is `CTAS_PER_ROW == 1` only (the split path is declared
  but not built) — `csrc/sampling/sampling.cu:11-16, 293-298`.
- `top_k_top_p_sampling_kernel` runs one block of **256 threads per row**
  (`kRowThreads = 256`, `sampling.cu:1080`; launch at `sampling.cu:2301`).
- hpc-ops launches `grid=(N,B) block=1024` for K1 and `grid=(B,) block=Nmax*K` for K2 —
  `fused_sampler.cu:654-675`.

### Traffic model (BF16 logits, `V = 120832`)

Row bytes = `V·2 = 241 664 B = 0.2417 MB`.
`1 MB` costs **0.2985 µs** at 3.35 TB/s, **0.3317 µs** at 90 % of 3.35, **0.125 µs** at 8 TB/s,
**0.1389 µs** at 90 % of 8.

| B | MB (1 pass) | H100 @3.35 | H100 @90 % | B200 @8 | B200 @90 % | 2-pass H100 @90 % | current kernel-ops (≥5 passes) H100 @90 % |
|---|---|---|---|---|---|---|---|
| 1 | 0.242 | 0.07 µs | 0.08 µs | 0.03 µs | 0.03 µs | 0.16 µs | 0.82 µs |
| 8 | 1.93 | 0.58 | 0.64 | 0.24 | 0.27 | 1.28 | 6.4 |
| 32 | 7.73 | 2.31 | 2.56 | 0.97 | 1.07 | 5.13 | 25.7 |
| 128 | 30.9 | 9.23 | 10.3 | 3.87 | 4.30 | 20.5 | 103 |
| 512 | 123.7 | 36.9 | 41.0 | 15.5 | 17.2 | 82.1 | 411 |

All µs values **[derived]** from the byte counts above; "current kernel-ops" assumes one
rejection round (read logits + write probs + read probs = 5 × the single-pass bytes).

Same table for the other vocabularies at `B = 512` (1 pass, BF16, **[derived]**):

| V | row bytes | B=512 MB | H100 @3.35 | B200 @8 |
|---|---|---|---|---|
| 120832 | 236 KiB | 123.7 | 36.9 µs | 15.5 µs |
| 128256 | 250.5 KiB | 131.3 | 39.2 µs | 16.4 µs |
| 151936 | 296.8 KiB | 155.6 | 46.4 µs | 19.4 µs |

### Occupancy / SFU reality at small `B` **[derived, must be measured]**

- One CTA of 1024 threads issuing 16 B per thread has ~16 KiB in flight. At ~600 ns DRAM
  latency that is ~27 GB/s per CTA. A single CTA reading one 236 KiB row therefore takes
  **≈ 8.7 µs**, not 0.07 µs. To reach the bandwidth floor at `B = 1` you need **≥ ~120 CTAs**,
  i.e. row splitting.
- Current kernel-ops uses 256 threads per row ⇒ ~4 KiB in flight ⇒ ~6.8 GB/s per CTA ⇒
  **≈ 35 µs just to read one row at `B = 1`**, before any probs write. hpc-ops at `B = 1` uses
  `N = 32` blocks × 1024 threads, i.e. 32× more parallelism. **This is the single worst case in
  the current implementation and a 10-40× gap versus hpc-ops at `B = 1`.** **[derived]**
- SFU budget: `V·B` elements, 1 transcendental each = 61.9 M ops at `B = 512`; H100 MUFU
  throughput ≈ 3.8e12/s ⇒ 16 µs per transcendental-op-per-element. The 37 µs memory time thus
  affords **two** transcendentals per element; three makes the kernel compute-bound.
  **[derived]**
- Launch overhead: assume 1.5-2.5 µs per kernel launch and 0.5-1 µs per node when replayed from
  a CUDA graph. **[estimate — nobody published a number for this box; measure it]** With that
  assumption the `B = 1` floor is ~2-3 µs for a 1-launch design, ~4-6 µs for 2 launches, and
  ~8-10 µs for 4 launches. This is why launch count matters more than bytes at `B ≤ 8`.

### Inferences
- Ranking at `B = 512` on traffic alone: hpc-ops 1.04× floor < option (a) 2.0× < current
  kernel-ops 5.0×. Traffic alone therefore does **not** let (a) beat hpc-ops at large `B`; the
  win must come from (i) hpc-ops' sort cost, (ii) hpc-ops' K1 being 1 block/SM across 31 waves,
  (iii) exactness. Those must be measured, not asserted.
- At `B ≤ 128`, (a)'s second pass is an L2 read, so its *effective* traffic is ~1.05-1.3× the
  floor and it beats hpc-ops outright.
- At `B ≤ 8`, byte counts are irrelevant; the winner is whoever has the fewest launches and the
  most CTAs. A 1-launch split-row kernel with `S ≈ min(ceil(132/B), 32)` splits is the right
  answer and should land at ~2-3 µs.
- B200 shifts the crossover: its 8 TB/s makes `B = 512` a 15.5 µs memory problem, so any
  per-element compute (sorts, 2+ transcendentals) becomes the bottleneck earlier. A design that
  is compute-light wins more on B200 than on H100.

### Gaps
- Achievable (not peak) HBM bandwidth for a pure streaming read at `B = 512, V = 120832` on the
  target H100 and B200 is unmeasured here; 90 % is an assumption.
- Actual L2 hit rate for the second pass at each `B` is unmeasured.
- Launch overhead on the target machines (bare and graph-replayed) is unmeasured.

---

## Q5. RNG and determinism

### Takeaway
The correct primitive is a **counter-based hash of (request seed, absolute step position, token
id)** with no RNG state, so the draw is a pure function of the request's own stream and never of
batch composition, grid size, split count, launch order or graph replay. hpc-ops violates nearly
every one of those properties; kernel-ops already declares the right contract but only uses it
on part of the surface.

### Cited Findings

- **hpc-ops**: `curand_init(seed, sequence, offset)` where `offset` comes from a **host-side
  process-global `std::atomic<uint64_t>` advanced once per launch** (`kPerLaunchOffsetIncrement
  = 128`); the comment says the offset "makes the same seed give fresh samples" —
  `3rd/hpc-ops/src/sampler/sampler_rng.cuh:6-13, 30-52`. Consequences: (i) the result depends
  on how many sampler launches the process has already done, so it is **not reproducible across
  runs**; (ii) under CUDA-graph capture the offset is frozen at capture time, so **every replay
  reuses the same Gumbel vector**; (iii) the counter is shared across streams, on top of the
  documented "not safe to call concurrently on multiple streams" scratch restriction
  (`sampler.h:38-39`).
- hpc-ops' K2 derives `sequence = ibatch*32 + ilane` — `fused_sampler.cu:480-483` — so the
  *subsequence* is row-local (good) but the offset is not (bad). Its own tests sidestep the
  problem by passing an external `gumbel_noise` buffer to be bit-exact
  (`sampler.h:32-34`), which is a tell that the self-drawn path is not reproducible.
- **TRT-LLM**: per-row seed/offset **tensors** with `kPerRowSubsequenceBase = 1ULL<<63`; the
  header explains "A per-row offset instead names the row's own subsequence: its draws then
  depend only on its own seed/offset, not on its position in the batch" —
  `3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/fusedSampling/fusedSamplingKernels.h:29-33, 65-75`,
  implementation at `fusedSamplingKernelsCommon.cuh:132-146`.
- **vLLM**: `murmur3_hash32(seed, pos, offset)` mixes the request seed, the step position and
  the token offset into one 32-bit uniform, and `murmur3_uniform64` double-hashes with
  `domain = 0x9E3779B9` for an fp64 path —
  [gumbel.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py).
  The comment "`keys` indexes the noise, so the same token draws the same noise wherever it
  appears; `pos` and `seed` place the draw in the request's stream, which is what lets a draft
  and its verification agree" is the design statement for spec-decode correctness.
  A `_DRAFT_NOISE_SALT = 1<<30` separates the draft's stream from the target's.
- **FlashInfer** exposes `seed`/`offset` as **tensors**, explicitly "required for CUDA graph
  compatibility", and a `deterministic: bool = True` switch —
  [docs](https://docs.flashinfer.ai/api/sampling.html).
- **kernel-ops already declares the right thing**: `RngParams { global_seed, per_request_seed[B],
  per_request_offset[B], stream_id }` with the header contract "Every random draw is a pure
  function of (seed, request_id, token_position, draw_index) so results do not depend on batch
  composition, stream, launch order, split count or graph replay" — `csrc/common/rng.cuh:1-22`,
  with `philox4x32_10`, `uniform_from_counter`, `gumbel_from_counter`,
  `exponential_from_counter` primitives (`:24-38`).
- Speculative decoding needs the *same* noise for a token at the same position across the draft
  and the verification pass (vLLM's salt comment above) and a probability *ratio* test
  `target/draft >= u` — `3rd/vllm/vllm/v1/sample/rejection_sampler.py:870-889`.

### Inferences — recommended RNG contract

1. **One hash per draw, no state.** `u = uniform_from_counter(seed_b, request_id=b,
   position=step_b, draw_index=token_id, stream_id=0)`. `draw_index = token_id` (not a running
   counter) is what makes the noise *positional*: the same token always draws the same noise at
   the same step, which is what spec decoding needs, and it makes the draw independent of how
   many elements each thread touches.
2. **`position` is the absolute decode step**, supplied as a device tensor the engine already
   owns (KV length / decode index). No extra kernel, no host counter, graph-replay safe because
   the tensor is updated between replays, and it makes `seed + position` the only thing that
   changes per step.
3. **`stream_id`** separates sampling (0) from spec-accept (1) from draft noise (2) — the
   kernel-ops analogue of vLLM's `_DRAFT_NOISE_SALT`.
4. **Draw count**: an inverse-CDF path needs exactly **one** uniform per row; a Gumbel/race path
   needs one per *candidate*. After compaction to ≤ 2048 candidates that is 2048 hashes per row
   instead of 120 832 — a 59× reduction in RNG work and, more importantly, it makes the result
   **independent of the summation order** (argmax is associative/commutative with a
   deterministic tie-break), so it stays bitwise-identical under any split count. This is the
   key reason to prefer "compact then race" over "prefix-scan then invert".
5. **Tie-break** must be explicit and total: larger score wins; on exact equality the smaller
   token id wins. hpc-ops already does this in K2 (`fused_sampler.cu:528-531, 540-557`) and
   kernel-ops should state it in the ABI.
6. **Determinism classes to publish** (this is the part hpc-ops cannot offer):
   - D1 *bitwise, same shape*: same seed + same logits + same `B`/`V` ⇒ identical output, every
     run, every stream, every replay.
   - D2 *bitwise, any shape*: identical output regardless of `B` (each row sees the same noise
     vector and the same reduction order). Achievable if the per-row work is independent of `B`
     (fix the split count as a function of `V` only, or make the sampling rule order-free as in
     point 4).
   - D3 *distributional*: matches the reference distribution within a chi-square bound (Q7).

### Gaps
- The cost of one Philox-4x32-10 block vs one murmur3 hash per element is not measured on
  H100/B200. Philox needs a 10-round 4×32 permutation; murmur3 is ~6 multiplies. At one draw
  per element murmur3 is likely 2-4× cheaper **[estimate]**. Since the recommendation is to draw
  only per *candidate*, either is affordable, so this is a low-risk choice.
- No published standard for "RNG reproducibility across tensor-parallel ranks". TRT-LLM's
  `combineOnlineSoftmax` comment says the merge is "associative and commutative … which is what
  keeps two TP ranks holding identical logits in agreement" (`fusedSamplingKernelsCommon.cuh:219-222`)
  — kernel-ops should adopt the same argument for its fixed-point mass.

---

## Q6. CUDA Graphs: what breaks, and the rules to adopt

### Takeaway
Graph capture fails on four things: allocations, host-side RNG state, host-side `.item()`/
`.min()` syncs, and grid shapes derived from host state that changes between replays. hpc-ops
breaks the first two outright and is documented as cross-stream unsafe. kernel-ops must adopt a
"plan before capture, no allocation in replay, everything that varies is a device tensor"
contract, which its `Workspace`/`LaunchContext` design already implies.

### Cited Findings

- hpc-ops allocates four scratch buffers with `cudaMallocAsync` in the launcher and frees them
  with `cudaFreeAsync` in the RAII destructor — `fused_sampler.cu:585-607, 648-651`. Under
  capture these become graph memory nodes; they are legal but they serialise the stream, they
  pin the pool, and the free must be ordered before reuse.
- hpc-ops' own header: "Concurrency: not safe to call concurrently on multiple streams of the
  same device (the per-device scratch workspace is shared across streams)" — `sampler.h:38-39`.
- hpc-ops' RNG offset is a host atomic advanced per launch — `sampler_rng.cuh:44-52` — so the
  captured graph freezes the offset (see Q5).
- TRT-LLM is explicit that seed/offset are **tensors** "so the call is legal inside a captured
  CUDA graph" and that min-p is clamped **on device** rather than on the host because "a value
  check would need a D2H sync that is illegal under CUDA graph capture" —
  `fusedSamplingKernels.h:65-67`, `fusedSamplingKernelsCommon.cuh:155-162`.
- FlashInfer requires `top_k`/`top_p` to be tensors, not scalars, "for CUDA graph
  compatibility" — [docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html).
- vLLM notes `torch.multinomial` "causes CPU-GPU synchronization" and replaces it with
  `probs.div_(q).argmax()` for exactly that reason — `topk_topp_sampler.py:560-577`.
- kernel-ops' `WorkspaceRequirements`/`Workspace` are caller-owned with a declared alignment
  (`csrc/sampling/sampling.cu:859-866`), and `LaunchContext` carries stream/arch — i.e. the
  contract is already graph-friendly; the problem is only that `sample()` asks for
  `B·V·4` bytes.

### Inferences — the rules to encode

1. **No allocation, ever, in the sample path.** All scratch comes from the caller's `Workspace`,
   sized by `sampling_workspace_requirements()`. This alone makes hpc-ops' four
   `cudaMallocAsync` calls disappear.
2. **No D2H sync.** No `.item()`, no `.min()`, no early-exit decisions on host-side data.
   hpc-ops' entry point already honours this (`entry.cc` validates shape/dtype only); the
   danger is in kernel-ops' Python-facing wrappers.
3. **Grid is a pure function of `(B, V, dtype)`** and is computed at plan time, not per replay.
   For dynamic batch, either capture per bucket (normal practice) or launch a max-size grid and
   pass `B` in device memory, with an in-kernel early exit. hpc-ops' `pick_n_blocks_per_row`
   already reads `get_sm_count()` at plan time, which is fine, but it makes the *numerics*
   device-dependent (partial max/sum merge order changes with `N`) — kernel-ops should make the
   merge order a function of `V` only, or accept D1-but-not-D2.
4. **`capture_generation` propagation**: when the workspace base address or the split count
   changes, the plan must be re-resolved. kernel-ops already documents a
   `capture_generation` concept in `csrc/common/tensor.h`; the sampler must actually bump it when
   `ctas_per_row` changes (currently `sample()` re-derives it from `vocab` every call —
   `sampling.cu:2248`).
5. **Multi-stream**: all state must live in the caller's workspace, or be per-stream. hpc-ops
   fails this by construction.
6. **Determinism under replay**: with a counter RNG whose `position` advances, replay is safe by
   construction; with a *host* offset it is silently wrong. This is the strongest argument for
   the Q5 contract.

### Gaps
- Whether a graph node that both allocates and frees (hpc-ops' pattern) is measurably slower
  than a pre-allocated workspace in a 5-node decode graph is unmeasured. Expect a few µs
  **[estimate]**.
- No published data on `cudaGraphExecKernelNodeSetParams` cost for per-step `pos`/`seed` updates
  versus updating a device tensor outside the graph; both are viable, neither is measured here.

---

## Q7. Benchmark and correctness plan

### Takeaway
"Better than hpc-ops" must be defined in a way hpc-ops cannot wriggle out of: same output
distribution, same or fewer launches, strictly lower p50/p99 on the shared (bounded) cases, plus
cases hpc-ops cannot run at all (full-vocab, min-p, 128256/151936, `B=512` with 512 concurrent
streams). Correctness needs a chi-square/TV test against an fp64 reference and a determinism
matrix, not just a golden-token diff.

### Cited Findings
- The statistical target is well defined because the intended semantics coincide with
  FlashInfer's theorem `Pr[output = j] = p_j / Z` over the filtered set —
  [FlashInfer blog](https://flashinfer.ai/2025/03/10/sampling.html).
- vLLM's mask semantics keep the whole tie-group at the k-th value (`logits < top_k_mask`) —
  `topk_topp_sampler.py:500-502` — so the reference must be built the same way.
- hpc-ops' bounded support is documented, so it must be excluded from distributional comparison
  on the tail: `sampler.h:17-20`.
- FlashInfer's `deterministic=True` flag and seed/offset tensors give an independent,
  reproducible baseline to diff against —
  [docs](https://docs.flashinfer.ai/api/sampling.html).
- hpc-ops supports an **external `gumbel_noise` buffer** "for bit-exact tests"
  (`sampler.h:32-34`) — use it: it lets us compare kernels with the noise held constant, which
  isolates algorithmic from RNG differences.
- AIR Top-K's evaluation protocol (8 open-source implementations, wide range of sizes *and*
  distributions, including adversarial) is the right template for the top-k selection part —
  [SC'23](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html);
  RadiK adds adversarial-distribution robustness as a first-class metric
  ([ICS'24](https://dl.acm.org/doi/10.1145/3650200.3656596)).

### Proposed case matrix

| Axis | Values |
|---|---|
| V | 120832, 128256, 151936 (plus 1000, 32000, 262144 as robustness) |
| B | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| dtype | BF16, FP32 |
| params | greedy; T=0.7 only; T=1.0; top-k=1/50/1000; top-k=50 + top-p=0.9; top-p=0.95 only; min-p=0.05; joint vs sequential k→p |
| state | no penalty; repetition penalty with bitmask; frequency/presence; draft mask (spec) |
| adversarial logits | all-equal; one-hot; two-way tie at the boundary; 1e6 dynamic range; ±inf; NaN; V not a multiple of the vector width; row-stride ≠ V |

### Metrics

- **Latency**: p50 / p90 / p99 over ≥ 1000 iterations with the decode loop captured in a CUDA
  graph; report both end-to-end and `nsys` per-kernel GPU time.
- **Traffic**: `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum` per
  launch; report achieved GB/s and the ratio to the Q4 floor.
- **Occupancy**: `sm__warps_active.avg.pct_of_peak_sustained_active`, plus achieved CTAs/SM.
- **Launch count**: count graph nodes; this is a first-class metric at `B ≤ 8`.
- **SFU pressure**: `sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active` — the Q4 model
  predicts the design goes compute-bound above ~2 transcendentals/element.

### Statistical correctness

1. **Reference**: fp64 `softmax`, filter with vLLM tie semantics, renormalise, cumulative sum,
   in a CPU/PyTorch-fp64 script. Store the reference CDF.
2. **Draw N = 1e6 samples** for one fixed logits row (fixed seed sweep so the noise differs),
   build the empirical histogram over the support, compute:
   - chi-square statistic over the support (merge bins with expected count < 5), compare against
     the `χ²(K-1)` 99.9 % quantile;
   - total-variation distance, accept `TV < 5·sqrt(K/(2N))` (statistical noise scale);
   - for the tail, a separate test on the top-100 tokens only (this is where bounded samplers
     fail and where we must *not* be compared against hpc-ops).
3. **Exactness (not statistical)**: with the noise held constant (external Gumbel buffer, or a
   fixed `uniform_from_counter` table), the chosen token must equal a reference argmax computed
   in fp64 — this catches threshold bugs that a chi-square test would take 1e6 samples to see.
4. **Determinism matrix**: run each case twice in the same process, twice in fresh processes,
   with `B` decomposed as `{8}` vs `{4,4}` vs `{1}×8`, on two streams, and under graph replay —
   expect D1 always; D2 as declared.
5. **Edge cases**: `top_k >= V` (must equal full-vocab sampling), `top_p = 1` (no-op),
   `min_p = 1` (argmax only), `temperature → 0` (greedy), all `-inf` row (`out_ids = -1`, as the
   current `ops.h` already specifies at `csrc/sampling/ops.h:19`), `V = 1`.
6. **Spec-decode**: acceptance-rate equality with the analytic value
   `sum_j min(p_j, q_j)` over 1e5 trials, plus a tree case checking that a token's noise is
   identical in the draft and verify passes (vLLM's salt property).

### What "better than hpc-ops" should mean (falsifiable)

1. **Shared cases** (top-k ≤ 64 only, `V = 120832`, BF16/FP32): p50 and p99 no worse at every
   `B`, and **≥ 1.3× faster at `B ≥ 64`** where bytes dominate. If we cannot win here, we have
   not beaten it — we have only out-featured it.
2. **Launch count**: 1 at every `B` (hpc-ops: 2).
3. **Workspace**: `O(B·(V/8 + C))` with `C ≤ 2048` candidates — i.e. ≤ ~10 MB at `B = 512`
   versus hpc-ops' `B·Nmax·K·8 ≈ 4 MB` and kernel-ops' current **247 MB**. (hpc-ops wins here;
   we should match it, not beat it.)
4. **Semantics**: exact full-vocab (no top-k bound), min-p, 128256/151936, `top_k = 0` meaning
   *disabled* not *truncated*.
5. **Determinism**: D1 always, D2 declared; bitwise identical across graph replay and across
   stream count. hpc-ops cannot pass this today.
6. **No allocation and no D2H sync in the replay path.**

Fairness rules: identical logits tensor and allocation, same warmup (≥ 50 iters), locked clocks,
same CUDA/driver, 5 independent runs, report median-of-p50 and worst-of-p99, and always report
the **DRAM bytes actually moved** next to the latency so a win cannot come from skipping work.

### Gaps
- No agreed public baseline number for hpc-ops at these shapes; the report must publish one.
- Fair comparison with vLLM/SGLang is service-level (they include scheduling), so the only fair
  head-to-head is kernel-level: call `flashinfer.sampling.top_k_top_p_sampling_from_probs` and
  vLLM's `torch.ops._C.fused_gumbel_argmax` directly, on the same tensors, from the same process.

---

## Q8. Concrete recommendation

### Takeaway
Build **one fused kernel** (`kops::sample::sample_fused`) that reads the logits **twice** for
filtered sampling and **once** for unfiltered sampling, never materialises fp32 probabilities,
compacts survivors into shared memory, and draws with an order-independent exponential race over
the compacted set using a counter RNG keyed on `(seed, request, step, token)`.

### Proposed architecture

```
                    ┌─────────────── caller-owned Workspace (no alloc in replay) ──────────────┐
row seed[B] ────────┤ seed/pos tensors (device)                                                │
step pos[B] ────────┤                                                                          │
penalty bitmask ────┤ [MAX_BS, ceil(V/8)] uint8  (1 atomicOr writeback, as hpc-ops)            │
topk/topp/minp/temp ┤ [B] each                                                               │
logits [B,V] ───────┤ read-only                                                                │
                    └──────────────────────────────────────────────────────────────────────────┘

grid = (S, B), block = 1024, S = 1 if B >= 132 else min(ceil(132/B) rounded up to 2^k, 32)

CTA (row b, split s) ── P1 (1 exp/elem) ─────────────────────────────────────────────
    stream vocab slice, apply penalty-bitmask + temperature in registers
    accumulate: online (max, sum) · 8-bit histogram of the monotone logit key (16 copies) · argmax
    block-reduce (CUB BlockReduce over OnlineSoftmax, associative+commutative)
    write partial {max, sum, hist[256], argmax} to workspace slot (b,s)

CTA (row b, split S-1) ── P2 merge (arithmetic only) ────────────────────────────────
    reduce S partials in *split-index order* (deterministic) → m, Z, hist[256], argmax
    descent: pick the bucket b* where count crosses k (top-k) and/or mass crosses p·Z (top-p)
    publish {threshold_prefix, b*} to the row's workspace slot

CTA (row b, split s) ── P3 (1 exp/elem) ─────────────────────────────────────────────
    re-read the same slice (L2-resident for B <= ~128), recompute w = exp(l/T - m)
    append w into a per-row candidate buffer when the key is in bucket b* (atomic ticket)
    build the next-byte sub-histogram of those candidates

CTA (row b, S-1) ── P4/P5 (shared memory only) ─────────────────────────────────────
    two more 8-bit rounds over the <= C candidates → exact threshold (tie-group preserved)
    filter w >= t  (plus min-p floor, plus top-k boundary ties)
    exponential race: argmax_j (w_j / (-log u_j)) with u_j = uniform_from_counter(seed_b, b, step, token_j)
      tie-break: larger ratio wins, then smaller token id
    write out_ids[b]; atomicOr the penalty bit; optional out_logprob / chosen_rank

fallback (rare): candidate buffer overflow or S == 1 with a flat row
    → rejection sampling in a bounded loop with the *same* RNG (TRT-LLM's pattern), or
    → one extra global pass with the pinned threshold (exact, 3 reads)
```

**Why each choice**

| Decision | Reason | Evidence |
|---|---|---|
| No fp32 probs by default | 5× traffic and 247 MB workspace today | `sampling.cu:859-864`, Q4 table |
| 8-bit radix histogram, 16 copies | contention, not capacity, is the cost | `fusedSamplingKernelsCommon.cuh:322-331` |
| Compact survivors to smem at ≤ 2048 | remaining radix rounds stop touching DRAM | `fusedSamplingKernelsCommon.cuh:491-523` |
| Descent over count *and* mass | one structure serves top-k, top-p, min-p | `fusedSamplingKernelsCommon.cuh:43-48` |
| Exponential race over the compacted set | order-independent ⇒ D2 determinism; 1 hash per candidate; no `log p` | `rng.cuh:36-37`, `topk_topp_sampler.py:546-552` |
| Keep the whole tie group at the k-th value | matches vLLM/reference semantics | `topk_topp_sampler.py:500-502`, `sampling.cu:1124-1126 (select_kth_key) / 1189 (boundary_equals)` |
| Fixed-point (2^40) mass if a prefix sum is ever needed | exact ⇒ order-independent | `sampling.cu:1093-1119` (already there) |
| Last-CTA-per-row merge via atomic ticket | 1 launch at every `B` | pattern proven in `fused_sampler_temperature.cu:8-13` |
| Split rows when `B < 132` | 1 CTA cannot exceed ~27 GB/s | Q4 occupancy model |

**Expected traffic and latency** (`V = 120832`, BF16, **[estimate]** from Q4)

| B | design bytes | H100 | B200 | hpc-ops (measured? no — estimate) | current kernel-ops |
|---|---|---|---|---|---|
| 1 | 0.48 MB (2 rows-worth) | ~2-3 µs (launch bound) | ~2-3 µs | ~5-7 µs | ~40-100 µs |
| 8 | 3.9 MB | ~2-3 µs | ~2-3 µs | ~7-9 µs | ~10 µs |
| 32 | 15.5 MB | ~6 µs | ~3 µs | ~25-40 µs | ~28 µs |
| 128 | 61.9 MB (2nd pass from L2) | ~15-18 µs | ~8 µs | ~70-100 µs | ~105 µs |
| 512 | 247 MB | ~80-90 µs | ~35 µs | ~120-370 µs (sort-bound) | ~410 µs |

The `B = 512` row is the one that decides the project: if hpc-ops measures at ~40 µs there,
option (a) does **not** beat it on latency and the argument must shift to exactness +
`B ≤ 128` + launch count. If it measures at > 100 µs (as the 4-sorts × 31-waves model predicts),
option (a) wins outright. **Measure hpc-ops' K1 with `ncu` before committing to a story.**

**Milestones**

| M | Deliverable | Exit criterion |
|---|---|---|
| M0 | bench harness + baselines (hpc-ops full/temperature, current `sample()`, FlashInfer-from-logits, current `gumbel_max_sample`) + `ncu` counters | hpc-ops K1 time and bytes captured at `B ∈ {1,32,512}`; the estimate table above is replaced by measured numbers |
| M1 | 1-pass exact path: temperature/greedy/unfiltered through the counter RNG + exponential race, penalties from the bitmask | matches hpc-ops `fused_sampler_temperature` within 10 % at `B ≥ 32`; beats it at `B ≤ 8`; D1+D2 pass; 1 launch |
| M2 | pass-1 histogram + descent + `top_k_mask`-equivalent threshold; no sampling yet | threshold bitwise-equal to `select_kth_key` on every adversarial row; 2 reads, 1 launch |
| M3 | smem compaction + race + overflow fallback | chi-square/TV pass on `top-k ∈ {1,50,1000}`; TV within noise at `N=1e6` |
| M4 | top-p/min-p mass descent, joint vs sequential modes | chi-square pass on `top-p ∈ {0.9,0.95}`, `min-p ∈ {0.05}`; `top_p=1` is a no-op |
| M5 | split-row + last-CTA merge for `B < 132`; graph capture and the determinism matrix | 1 launch at every `B`; replay bitwise stable; no `cudaMalloc` in the capture window |
| M6 | spec decode: rejection + tree verification reusing the same noise | acceptance rate matches `Σ min(p,q)` analytically; draft/verify noise agreement |
| M7 | logprobs / `chosen_rank` / optional `outputProbs` as a *separate, opt-in* kernel | does not regress the default path; keeps TRT-LLM's "sort-free cannot hand back the distribution" separation |

**Risks**

1. **hpc-ops may be faster than modelled at `B = 512`.** Mitigation: M0 measures it first;
   fall back to "win at `B ≤ 128` + exactness + launches" if the sort cost is small.
2. **Crossing-bucket overflow** (flat distributions, high temperature) forces a 3rd read.
   Mitigation: TRT-LLM's boundary — `C = 2048` — plus rejection fallback; measure the overflow
   rate on real logits at `T ∈ {0.7, 1.0}` and on the adversarial rows.
3. **The 2nd pass is only cheap when L2-resident.** At `B = 512` it is a full DRAM read.
   Mitigation: accept it (2× floor) or implement the 1-pass adaptive candidate filter later.
4. **SFU saturation** if pass 2 also computes `log` for the race: keep the race on the compacted
   set only (≤ 2048 logs per row).
5. **Determinism regression** if the split count depends on `B`. Mitigation: make the sampling
   rule order-free (the race) so splits cannot change the answer.
6. **ABI churn**: adding `min_p` to the sampler's `SamplingParams` is already there
   (`sampling_kernels.cuh:15-26`), but adding a workspace slot for `step` and the candidate
   buffer changes `sampling_workspace_requirements` — that is an ABI change per
   `AGENTS.md`/`KOPS_ABI_VERSION` and must be versioned.

### Gaps
- The recommended `S` (splits) policy, `C` (candidate capacity), and block size (512 vs 1024)
  are all chosen from first principles here, not from a sweep. M0-M1 must sweep them.
- Whether the last-CTA merge can absorb the P4/P5 work without becoming the critical path at
  `B = 1` (one CTA doing 2048 candidate comparisons) is unverified; it should be ~2048/32 warp
  iterations ≈ sub-µs, but measure.

---

## Q9. Claims that still need measurement (consolidated)

1. hpc-ops K1 kernel time, sort fraction of that time, and achieved DRAM bandwidth at
   `B ∈ {1, 8, 32, 128, 512}` — **the single most important missing number**.
2. Achievable (not peak) HBM read bandwidth for these shapes on the target H100 and B200.
3. Launch overhead bare and graph-replayed, and the cost of a `cudaMallocAsync`/`cudaFreeAsync`
   pair inside a captured graph.
4. Second-pass L2 hit rate as a function of `B` (the whole "2 passes is nearly free at `B ≤ 128`"
   argument rests on it).
5. Crossing-bucket overflow rate at `C = 2048` on real model logits.
6. SFU (MUFU) utilisation: is the kernel bandwidth-bound or XU-bound at `B = 512`?
7. Rejection-round distribution for the fallback path (FlashInfer's "4-6" is anecdotal).
8. Gumbel/race numerical bias at `V = 120832` in fp32 (chi-square with `N = 1e6`).
9. Determinism across tensor-parallel ranks (does the fixed-point mass really make two ranks
   holding identical logits agree bit-for-bit?).
10. Whether vLLM's newest `gumbel.py` path applies top-k/top-p as a mask upstream (Q1 gap) —
    it changes what "vLLM's pipeline" means in the final comparison table.
