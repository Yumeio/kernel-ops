# Fast Top-K / Top-P Selection Kernels for LLM Inference (local `3rd/` tree + leading external projects)

Scope note: this file surveys exact top-k/top-p *selection* algorithms and their concrete implementations, aimed at a fused sampler in `csrc/sampling` for V ≈ 120832 / 128256 / 151936, K = 32/64 (occasionally 256), B = 1..512, BF16/FP32 logits, H100/B200.

Convention used below:
- **[MEASURED]** = a number reported by the source's own benchmark.
- **[COMPUTED]** = arithmetic I performed from cited constants (traffic = bytes × passes ÷ bandwidth). Treat as an estimate.
- **[SPEC]** = vendor hardware spec, not benchmarked here.

Baseline traffic constant used throughout (fp32 logits, one full read of one row):
`V = 128256 → 513,024 B ≈ 501 KiB/row`. `B = 512 → 256.5 MB per full pass`.
At H100 SXM5 ≈ 3.35 TB/s [SPEC] one pass ≈ **76.6 µs**; at B200 ≈ 8 TB/s [SPEC] one pass ≈ **32.1 µs** [COMPUTED]. With BF16 logits halve both.

---

## Q1. attention.rs `fast_topk`: exact algorithm, block config, per-pass work, tie handling, output ordering, memory traffic, limits, tests, usage, integration

### Takeaway
`3rd/attention.rs/src/kernels/src/fast_topk.cu` is a *two-path* selector: an **iterative argmax+invalidate loop** for `seq_len ≤ 4096` (misleadingly named "bitonic") and a **32-bit MSB→LSB radix scan with no candidate compaction** for longer rows. The radix path re-reads the whole row from global memory on every one of up to 32 bit iterations, so its worst-case traffic is ~33 full passes — roughly 16× worse than the SGLang design and ~6–12× worse than TensorRT-LLM's production radix, per the roofline numbers in Q4. It returns **unordered** indices, pads with `-1`, and its membership under exact ties is **non-deterministic**. It is compiled and FFI-declared but has **no caller** in the Rust tree at this checkout.

### Cited Findings

**Algorithm, block config, limits**
- Header comment states the policy explicitly: `len <= topk` or `len <= 4096` → "bitonic-sort approach"; longer → "radix-based bucket filtering", MSB→LSB, parallel histogram, threshold bucket, recurse on remaining bits, final pass collects indices — `3rd/attention.rs/src/kernels/src/fast_topk.cu:5-11`.
- Fixed launch geometry: grid `(batch_size, 1, 1)` — **one block per batch element**; block **256 threads** — `fast_topk.cu:19-20`, constants at `fast_topk.cu:34-35` (`TOPK_WARP_SIZE 32`, `TOPK_NUM_THREADS 256`).
- Public C API: `fast_topk_select(scores, indices_out, batch, seq_len, topk, stream)`, `scores: [batch, seq_len] F32`, `indices_out: [batch, topk] I32` — `fast_topk.cu:22-26`, definition `fast_topk.cu:250-283`.
- `topk` is clamped to `seq_len` (`if (topk > seq_len) topk = seq_len;`) — `fast_topk.cu:260`. There is **no explicit max-K cap**; the limits are implicit (see below).
- Path switch: `bitonic_smem = seq_len*8 + 2*256*4` bytes; take the small path if `bitonic_smem <= 48*1024 && seq_len <= 4096` — `fast_topk.cu:265-271`. Solving the first inequality gives `seq_len ≤ 5888`, so the `seq_len <= 4096` clause is the binding one; **effective crossover is exactly 4096**.
- No dtype support beyond F32: the signature is `const float*` only (`fast_topk.cu:64-68`, `144-148`, `251`). BF16/FP16 must be up-converted by the caller.

**Small path (`fast_topk_bitonic_kernel`) — not actually a bitonic sort**
- Despite the name and header comment, the body is a **route-by-route argmax with invalidation**: `for (route = 0; route < topk; ++route)` does a strided scan of shared memory, writes `(best, best_idx)` to shared per-thread slots, does a `log2(256)=8`-step shared-memory tree reduction, the winner writes its index and then poisons the slot (`s_vals[idx] = -FLT_MAX`) — `fast_topk.cu:91-125`, invalidation at `fast_topk.cu:122`.
- Work: `topk` full sweeps of `seq_len` per block ⇒ **O(topk · seq_len)** comparisons, plus `topk × 9` `__syncthreads()` (one after the scan, one per reduction step) — `fast_topk.cu:103`, `116`, `124`. For `topk=256, seq_len=4096` that is 256 sweep rounds per block.
- Tie-break: prefers the **smaller index** at equal value, both in the per-thread scan (`if (v > best || (v == best && i < best_idx))`, `fast_topk.cu:96`) and in the tree merge (`fast_topk.cu:109-114`). This is a *deterministic* tie rule.
- Output ordering: **descending by value** (route 0 is the largest) — `fast_topk.cu:119-123`.
- Wasted shared memory: `s_idxs` is written at `fast_topk.cu:82` but never read; the argmax uses the loop variable `i` directly (`fast_topk.cu:97`). 8 B/element is allocated where 4 B would do, halving the effective row length that fits in 48 KiB.

**Large path (`fast_topk_radix_kernel`)**
- Monotone float→uint32 key: `u ^ (sign ? 0xFFFFFFFF : 0x80000000)` — `fast_topk.cu:137-142`.
- Bit loop `for (bit = 31; bit >= 0 && remaining_k > 0; --bit)` — `fast_topk.cu:169`. Note that `remaining_k` only decreases in the "descend into 0-branch" case (`fast_topk.cu:205`), so in the common case the loop runs **all 32 iterations**.
- **Every bit iteration scans the entire row from global memory** and tests `(sv & prefix) == prefix_val` with no candidate list and no compaction — `fast_topk.cu:174-179`. This is the single most important fact about this kernel: there is no shared-memory staging of survivors.
- Reduction: intra-warp `__shfl_xor_sync` sum (`fast_topk.cu:182`, helper `37-42`), per-warp counts into `s_count[NUM_WARPS]` (`185`), second-level reduce by warp 0 (`189-193`). `NUM_WARPS = 8` (`fast_topk.cu:155`).
- Decision: `if (ones_count >= remaining_k)` set the bit in `prefix_val` (descend into the "1" half); else `remaining_k -= ones_count` and descend into the "0" half — `fast_topk.cu:199-208`.
- Threshold is `threshold_sortable = prefix_val` (`fast_topk.cu:222`). The `float threshold` computed at `fast_topk.cu:219-220` is **dead code** (immediately superseded, comment: "Simpler: just compare sortable values").
- Collection: one final full scan with `atomicAdd` on a shared counter; a candidate is stored only `if (pos < topk)` — `fast_topk.cu:224-232`. **This silently drops excess candidates**, so when duplicate values sit exactly at the threshold the chosen subset is whichever threads win the atomic race ⇒ **membership is not deterministic under exact ties** (contrast with FlashInfer's `deterministic=True` radix collect, Q5).
- Output ordering: **unordered** (atomic arrival order) — same as SGLang and FlashInfer's default; only the small path is sorted.
- Under-fill handling: `collected = min(s_count[0], topk)`, remaining slots padded with `-1` — `fast_topk.cu:237-240`.
- Shared memory for this path is only `(NUM_WARPS+1)*4 + 4 = 40` bytes (`fast_topk.cu:275`), i.e. the kernel is designed to keep occupancy maximal by not caching anything — which is exactly why it must re-stream the row 32 times.

**Memory traffic**
- Worst case (`remaining_k` never reaches 0 before bit 0): **32 histogram passes + 1 collect pass = 33 full reads** of `seq_len × 4 B` per row [COMPUTED from `fast_topk.cu:169-179` and `224-232`].
- For V = 128256, B = 512: 256.5 MB × 33 ≈ **8.5 GB** of logits traffic per sampling step ⇒ ~2.53 ms at 3.35 TB/s (H100) or ~1.06 ms at 8 TB/s (B200) [COMPUTED]. That is 2–3 orders of magnitude above the useful-work bound.

**Integration**
- FFI declaration (not a definition): `pub fn fast_topk_select(scores: *const f32, indices_out: *mut i32, batch: c_int, seq_len: c_int, topk: c_int, stream: i64) -> c_int;` — `3rd/attention.rs/src/kernels/src/ffi.rs:4676-4683`.
- Build wiring: `println!("cargo:rerun-if-changed=src/fast_topk.cu");` — `3rd/attention.rs/src/kernels/build.rs:96`, i.e. the file is in the `cc` source list and always compiled into the kernels crate.
- **No Rust caller.** A repo-wide grep over `*.rs` for `fast_topk_select` returns exactly one hit, the `ffi.rs` declaration; there is no `ffi::fast_topk_select(...)` call site.
- The MoE top-k that attention.rs actually uses is a *different* symbol: `topk_select` (softmax + top-k weights), declared at `3rd/attention.rs/src/kernels/src/ffi.rs:786`, defined at `3rd/attention.rs/src/kernels/src/moe/topk_softmax_kernels.cu:629`, called from Rust at `3rd/attention.rs/src/topk.rs:199` (CUDA path) with the Metal/D Flash path at `topk.rs:222-244`.
- The stated intended uses are a doc comment, not observed call sites: "DSA lightning indexer top-k selection / DeepSeek V4 compressed-KV indexer / MoE expert routing / any future sparse attention token selection" — `fast_topk.cu:14-17`.
- The same file also carries an unrelated op, `dflash_select_candidates` (DFlash2 token selection, `topk ≤ 32`, `rank` reduction, single CTA of 256 threads) — `fast_topk.cu:285-390`.

**Tests / benchmarks**
- No test and no benchmark exist for `fast_topk` anywhere in the tree (the only nearby test, `topk.rs:716 dflash_topk_select_orders_scores_and_ties`, exercises the Metal DFlash2 path, not this kernel).

### Inferences
- The doc comment promises "radix-based bucket filtering ... recurse on remaining bits", but the implementation never compacts: the `prefix` test only *filters* within a still-full-row scan. So the algorithm has the pass count of a naive radix select without its traffic advantage. Any real deployment of this kernel at V ≥ 100K would be bandwidth-dominated.
- Because the small/large switch is at 4096 and the small path is O(K·V), a caller asking for K = 256 at V = 4096 gets ~1M comparisons/block; at V = 4097..∞ it gets the 32-pass path. There is no "sweet spot" regime in between — the two paths were clearly written for different use cases (V ≤ 4K indexer scores vs. very long compressed-KV indexer) and not tuned for a 128K-vocab sampler.
- The `-1` padding and unordered output mean the kernel cannot be dropped into a sampler that expects exactly-K valid ids without an extra compaction step.

### Gaps
- No published or in-repo performance numbers exist for this kernel; all traffic figures here are computed, not measured. I did not run it.
- Whether an external (non-`3rd`) consumer calls `fast_topk_select` through the `kops`-style C ABI cannot be determined from this checkout — only that the Rust tree does not.

---

## Q2. SGLang `sgl-kernel` `topk.cu`: algorithm, API, dtype support, block config, tests, performance

### Takeaway
`3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu` is the **best-engineered of the three local top-k kernels**: an 8-bit coarse histogram over an fp16-derived key, followed by a *single* collect pass that simultaneously stages boundary-bin candidates into shared memory and builds the next histogram, then 4 more 8-bit radix rounds entirely on the staged ≤4096 candidates. Net cost ≈ **2 full passes + ≤4 small gathers**, not 33. It is hard-wired to `TopK = 2048`, `float32` input, 1024 threads/block, and it is a **DSA sparse-attention indexer**, not a sampler.

### Cited Findings
- Provenance: adapted from `tile-ai/tilelang` `examples/deepseek_v32/topk_selector.py`, "adapt from tilelang to pure cuda / optimize the performance a little / fix the potential illegal memory access" — `3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu:1-8`.
- Compile-time constants: `TopK = 2048`, `kThreadsPerBlock = 1024` — `topk.cu:23-24`. `kSmem = 8*1024*sizeof(uint32_t) = 32 KiB` on CUDA, `48 KiB` default on ROCm (arch-injected `SGL_TOPK_DYNAMIC_SMEM_BYTES`) — `topk.cu:26-39`.
- Occupancy rationale is documented in-source: "Reduced from 128KB to 32KB to improve occupancy. Each radix pass needs at most ~TopK candidates in the threshold bin, so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient" — `topk.cu:35-38`. `SMEM_INPUT_SIZE = kSmem / (2*sizeof(int)) = 4096` — `topk.cu:96`.
- Key transform: `convert_to_uint8` casts float→half, applies the monotone-uint16 trick, then keeps the **top 8 bits** — `topk.cu:78-83`; `convert_to_uint32` is the standard fp32 sortable transform — `topk.cu:85-88`. Using fp16 for the coarse pass means the first histogram is only an *approximation* of the fp32 ordering, but it is used purely to bracket; exactness is recovered by the fp32 refinement rounds.
- Stage 1: zero 257 bins, histogram the whole row with `atomicAdd` into shared (`topk.cu:110-117`); a reverse cumulative sum (`value += hist[tx+j]`, so bin `t` holds "count ≥ bin t") over 256 bins in 8 unrolled steps with double-buffered shared memory (`topk.cu:119-134`).
- Threshold bin: the unique `tx` with `hist[tx] > topk && hist[tx+1] <= topk` (`topk.cu:137-141`); `topk -= hist[threshold_bin+1]` converts the global k into "how many still needed from the boundary bin" (`topk.cu:145`).
- Fast exit: if `topk == 0` after subtracting the strictly-greater population, just emit those indices and return (`topk.cu:147-156`) — one pass total in the lucky case.
- Otherwise the collect pass does double duty: indices with `bin > threshold_bin` go straight to the output via `atomicAdd(&s_counter,1)`, while indices with `bin == threshold_bin` are appended to `s_input_idx[0][pos]` **and** immediately contribute to the next histogram via `atomicAdd(&s_histogram[sub_bin],1)` — `topk.cu:164-181`, with the "fuse the histogram computation here" note at `topk.cu:172`.
- Stage 2: `for (round = 0; round < 4; ++round)` refines with 8-bit slices at `offset = 24 - round*8`, ping-ponging between `s_input_idx[0]` and `s_input_idx[1]` so each round only touches the ≤4096 staged candidates — `topk.cu:186-251`. Round 3 resolves the final boundary by counting down with `s_last_remain` into the tail of the output array (`index[TopK - pos]`) — `topk.cu:232-236`.
- Overflow guard: `num_input = min(s_num_input[r_idx], SMEM_INPUT_SIZE)` with a comment "clip here to prevent overflow" (`topk.cu:190-192`) — i.e. pathological duplicate-heavy rows silently truncate the candidate set, which is where the test's tolerated error comes from.
- Total traffic [COMPUTED from the structure]: 1 histogram pass + 1 collect/next-histogram pass over `length`, then 4 rounds each re-reading only `num_input ≤ 4096` raw values from global (`topk.cu:209`, `225`). So ≈ **2·length + ≤16K gathered floats**, versus 33·length for attention.rs.
- API surface: `fast_topk_interface` (`topk.cu:437-454`), `fast_topk_transform_interface` (page-table gather fused, `topk.cu:456-510`), `fast_topk_transform_ragged_interface` (`topk.cu:512-546`). Params struct carries `input, row_starts, indices, lengths, input_stride` (`topk.cu:41-47`), so rows are **ragged-capable** with per-row `length` and `row_start`.
- Fused transform variants: decode / prefill / prefill-ragged kernels each declare `__shared__ int s_indices[TopK]` and then gather the page table — `topk.cu:286-297`, `338-349`, `369-380`. Note the manual unroll `static_assert(TopK/kThreadsPerBlock == 2)`.
- Short-row fast path: `if (length <= TopK) naive_topk_cuda(...)` writes `indice[i] = (i < length) ? i : -1` — `topk.cu:50-55`, dispatch at `topk.cu:262-266`. No sorting at all in that regime.
- Dtype: **float32 only** (`score.data_ptr<float>()`, `topk.cu:407`); the key path internally uses `__half` for the coarse bin (`topk.cu:79`).
- Tests: `3rd/sglang/sgl-kernel/tests/test_topk.py` — `bs ∈ {1,132,256,4096}`, `k = 2048` only ("we only support 2048 now", line 104), `seq_len ∈ {2048,4096,16384,65536}`, `MAX_SEQ_LEN = 131072` (line 67). The check is *set equality with a tie exemption*: differing index sets are accepted iff the sorted **values** match, and up to **5** wrong indices are tolerated (`max_permit_error=5`, line 129). So the kernel is **exact up to ties at the boundary**, not bit-exact vs `torch.topk`.
- Test harness allocates `score = torch.randn(bs, MAX_SEQ_LEN)` — i.e. 2 GB at `bs=4096` — confirming the target regime is long rows, few-to-moderate batch (`test_topk.py:113`).

### Inferences
- The two-pass + smem-staging structure is the single most transferable idea in the local tree: after pass 2 the working set is bounded by `SMEM_INPUT_SIZE`, so refinement cost is independent of `V`. That is what makes the design viable at V = 128K.
- Because `TopK` is a template/`constexpr` 2048, this kernel **cannot** be reused as-is for K = 32/64: the output array, the `s_indices[TopK]` staging buffer, and the `static_assert(TopK % kThreadsPerBlock == 0)` all assume 2048.
- Emitting `-1` for short rows means downstream attention code must mask; a sampler would need to reject `-1` explicitly.

### Gaps
- No benchmark numbers ship with the kernel (no `benchmark_*` file for `fast_topk` in this checkout); only the qualitative "reduced 128 KB → 32 KB to improve occupancy" note.
- I found no SGLang doc page stating the throughput of `fast_topk_v2`; performance claims would have to come from an external benchmark run.

---

## Q3. vLLM: which top-k/top-p kernels exist, their algorithms, and how V1 fuses them

### Takeaway
vLLM V1 ships **two completely different fast paths plus two fallbacks**. For *sampling* it prefers **FlashInfer's sorting-free dual-pivot rejection sampler** (`top_k_top_p_sampling_from_logits`), and its own Triton path is now **Qrita** (pivot-based truncation+selection, arXiv 2602.01518) which is documented in-source as the default GPU sampler for top-k/top-p *masking*. The PyTorch `sort`-based path survives as the reference fallback. Separately, for **DSA indexer** top-k, vLLM has a three-tier persistent radix-select (`csrc/libtorch_stable/persistent_topk.cuh`) with tier thresholds at 8192 / 32768 elements.

### Cited Findings

**The sampler module**
- `TopKTopPSampler` docstring: "performs optional top-k and top-p filtering followed by weighted random sampling of logits. Implementations may update the logits tensor in-place." — `3rd/vllm/vllm/v1/sample/ops/topk_topp_sampler.py:200-206`.
- Backend selection at construction: `self.forward = self.forward_cuda if can_use_flashinfer else self.forward_native` — `topk_topp_sampler.py:220-225`.
- `flashinfer_sampler_supported()` gates on: `VLLM_USE_FLASHINFER_SAMPLER`, CUDA platform, compute capability supported by `FlashInferBackend`, **> 16 SMs**, and FlashInfer being able to JIT for the arch — docstring `topk_topp_sampler.py:77-95`; the SM>16 rule is restated inline: "FlashInfer 0.7+ rejects multi-CTA top-k masking on <=16 SMs".
- `forward_cuda` → `flashinfer_sample(logits.contiguous(), k, p, generators)` — `topk_topp_sampler.py:286-313`; that helper calls `flashinfer.sampling.top_k_top_p_sampling_from_logits(...)` when both k and p are active, else the top-p-only / top-k-only entry points — `topk_topp_sampler.py:596-618`.
- Fallback trigger is explicit: "(k is None and p is None) or generators" → PyTorch-native, because "FlashInfer 0.2.3+ does not support per-request generators" — `topk_topp_sampler.py:288-300`.
- `apply_top_k_top_p` dispatches `apply_top_k_top_p_triton` when Triton is available, else `apply_top_k_top_p_pytorch` — `topk_topp_sampler.py:460-470`.
- `apply_top_k_top_p_pytorch` is the sorting reference: `logits.sort(dim=-1)`, gather the k-th value, `masked_fill_(-inf)`, softmax, `cumsum`, top-p mask — `topk_topp_sampler.py:473-515`.
- `apply_top_k_only` does one `logits.topk(max_top_k, dim=1).values` and then a threshold mask — `topk_topp_sampler.py:517-536`. Single top-k, then a broadcast compare; no renormalization.
- ROCm path uses AITER's `top_k_renorm_probs` — `topk_topp_sampler.py:427`.

**The Triton (Qrita) kernel**
- File header: "Combined Top-K and Top-P Triton kernels. Based on the paper 'Qrita: High-performance Top-k and Top-p Algorithm for GPUs using Pivot-based Truncation and Selection' By Park et al. (https://arxiv.org/abs/2602.01518)" — `3rd/vllm/vllm/v1/sample/ops/topk_topp_triton.py:3-8`.
- Algorithm primitives visible in the source: a `_update_min_larger_stats` merge rule that tracks the smallest value strictly above a pivot and its multiplicity across tiles (the "duplication handling") — `topk_topp_triton.py:112-135`; a 256-entry `_NORMAL_CDF_TO_SIGMA_TABLE` and `_PERCENTILE_TO_STD_TABLE` used for the Gaussian σ-truncation — `topk_topp_triton.py:38-70`.
- Combined top-k/top-p semantics: "Top-k is applied first (by logit value), then top-p is applied to the remaining k values (by probability)" — `topk_topp_triton.py:1690-1694` region (`apply_top_k_top_p_triton` docstring).
- **Measured, in-source performance note:** "The monolithic kernel above assigns one program per row, so at small batch sizes nearly all SMs idle and every full-vocab sweep is latency-bound (**~25us each on SM100**)" — `topk_topp_triton.py:1500-1505` region (split-pipeline docstring). This is the only hard latency number in the local vLLM sampler code.
- Consequence, implemented: a **split-row pipeline** (`_topp_split_stats`, `_topp_split_step`, `_topp_split_mask`) splits each row across `S` programs, used only for rows with an active top-p and **no** active top-k; rows with active top-k stay on the monolithic kernel because "its truncation path is already fast" — `topk_topp_triton.py:1500-1520` region.
- Determinism/robustness: the split pipeline "tracks the tightest fallback: the largest evaluated pivot whose above-mass is >= p, which guarantees the top-p mass invariant if the search fails to converge" — `topk_topp_triton.py` docstring for `_topp_split_stats`.

**The C++ persistent top-k (DSA indexer, not the sampler)**
- File header "Persistent TopK Scheduler for DSA Indexer" — `3rd/vllm/csrc/libtorch_stable/persistent_topk.cuh:1-3`.
- Tier constants: `kThreadsPerBlock = 1024`, `RADIX = 256` (`:23-24`); medium-tier smem header 3200 B + 2×4096 ints = **35,968 B** (`:29-35`); `RADIX_THRESHOLD = 32768` (`:36`); decode tier uses `kDecodeBins = 2048` with `HIST2048_THRESHOLD = 8192` (`:39-40`).
- Three tiers by `seq_len` (dispatch at `:932-941`): ≤8192 → 2048-bin decode histogram; ≤32768 → "coarse FP16 histogram + 4-pass FP32 radix refinement" (comment `:415`); >32768 → "multi-CTA radix select for sequences > 64K ... each CTA loads its chunk into shared memory as ordered uint32, then participates in 4 rounds of coordinated radix select via global-memory histograms and barriers" (comment `:644-650`).
- `FilteredTopK` tier: "single CTA per row for bs > 32", 1024 threads, `RADIX=256`, `NUM_REFINE_ROUNDS = 4`, `FIRST_REFINE_SHIFT = 24` (`:1009-1010`, `:1027-1052`), explicitly "Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215" (`:962-970`), and the medium path is attributed to SGLang: "Adapted from .../sglang/.../topk.cu#L87 by: DarkSharpness ... an optimized topk kernel copied from tilelang kernel" (`:415` region).

### Inferences
- The vLLM V1 sampler is **not** a radix select at all in its fast path — it is rejection sampling, which is distribution-exact without ever sorting or finding the k-th value. That is a fundamentally cheaper contract than what a "sampler that must produce the top-k list" needs.
- vLLM's own code admits the monolithic one-program-per-row design is latency-bound at small batch (25 µs per full-vocab sweep) and therefore switches to a **split-row** layout — i.e. the same conclusion as TensorRT-LLM's multi-CTA tier. For our B = 1..512 target this means the kernel must have *both* a split-row path (B < #SMs) and a one-CTA-per-row path (B ≥ #SMs).
- The presence of both `deterministic` semantics and "tightest fallback pivot" tracking suggests the Qrita authors treat bitwise reproducibility as a hard requirement for serving — worth copying.

### Gaps
- The local `topk_topp_triton.py` is 1724 lines and I did not read the full monolithic kernel body; the exact σ-truncation arithmetic and the number of tiles (`NUM_TILES`, `BLOCK_SIZE`) are not quoted here.
- No measured latency table ships in the vLLM tree for the sampler; the 25 µs figure is a single anecdotal in-source comment, not a benchmark.

---

## Q4. TensorRT-LLM: topK/topP sampling kernels, indexer radix select, and perf claims

### Takeaway
TensorRT-LLM has **four distinct selection stacks**: (a) `kernels/fusedSampling/` for token sampling, (b) `kernels/indexerTopK.cu` — a **4-pass 11/11/11/10-bit histogram radix select** with 2048 bins for DSA indexer scores, (c) `kernels/glmMoeDsa/glmDsaHistTopK.cu` for GLM-MoE-DSA, and (d) **GVR V2** (2026), a self-sampling + multi-threshold streaming core that is the current state of the art here. The published roofline puts production radix CUDA at only **7.7–8.3 % average / 16.4–17.8 % peak** of the ideal one-read bandwidth bound, versus **39–41.6 % / 66.5–77.8 %** for GVR V2 — the strongest available evidence that pass count, not instruction count, dominates this problem.

### Cited Findings

**`indexerTopK.cu` structure**
- `constexpr int kNumBins = 2048;` with the rationale "2048 = 2^11 maps directly to the fp16 fast path's 11-bit bin index (sign + exponent + top mantissa bits, mask 0x7FF). All file-scope dispatch thresholds below derive from this" — `3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/indexerTopK.cu:41-46`.
- Per-step bin extraction (`extractBinIdx<step>`, `indexerTopK.cu:49-73`): `step 0` = fp16 bits `>>5` (11 bits); `step 1` = `bits >> 21`; `step 2` = `(bits >> 10) & 0x7ff`; `step 3` = `bits & 0x3ff`. So the pass widths are **11 / 11 / 11 / 10** over the fp32 sortable key — coarse bins, few passes, in deliberate contrast to SGLang's 8-bit × 4.
- `processHistogramStep` (`indexerTopK.cu:172-383`): zero 2048 smem bins (`:194-200`), histogram with `atomicAdd` (`:219`), CUB `BlockScan::ExclusiveSum` per 256-bin chunk (`:241-262`), locate the bin where the running prefix crosses `topK` (`:268-280`), then a collect pass that writes survivors to `smemOutput` and, if the boundary bin size ≤ `kNumFinalItems`, stages it in `smemFinal` (`:303-360`); returns `smemFinalBinSize[0] > kNumFinalItems` to decide whether another step is required (`:383`).
- `kNumFinalItems = 2048` (`indexerTopK.cu:393`) — the boundary bin is finished in shared memory when it fits in 2048 items.
- Launch config: **512 threads/block** (`kNumThreadsPerBlock = 512`, `indexerTopK.cu:927`, `981`), `gridDim = numRows`, `dynamicSmemBytes = topK * sizeof(int32_t)` (`:944-945`), plus PDL via `cudaLaunchAttributeProgrammaticStreamSerialization` (`:946-948`, `:971-973`).
- Dispatch by row width (`indexerTopK.cu:903-913`): `numColumns < 12288` → **insertion sort**; `12288 ≤ numColumns < splitWorkThreshold` (default **200K**) → **radix sort**; `≥ 200K` → unsupported for bf16/fp16 (`:978-984`), handled only by the fp32 split-work multi-CTA entry.
- bf16/fp16 support is explicit and documented as numerically equivalent to upcasting: "the histogram and sort steps operate on float keys after a `static_cast<float>(InputT)` at HBM-read sites, so accuracy is identical to casting input to fp32 before the kernel" — `indexerTopK.cu:907-910`.
- Prefill entry launches a separate insertion-sort kernel for the first 12288 **rows** and radix for the remainder (`:1000-1015`) — a heuristic on block count, not on length.

**GVR V2 (the 2026 production design)**
- Problem statement with the key numbers: "Selecting 1,024 INT32 indices from 131,072 FP32 scores writes just **4 KiB** of output, yet one complete read of the scores moves **512 KiB**. Every additional full-row pass pays that input cost again." — `3rd/TensorRT-LLM/docs/source/blogs/tech_blog/blog29_GVR_V2_Self_Sampling_Exact_TopK_for_Sparse_Attention.md:11-13`.
- Headline result **[MEASURED]**: "On B200, this design delivers **5.05× geometric-mean speedup over TensorRT LLM radix CUDA**, with **1.46× over GVR V1**. Both comparisons use the same 9,746 workloads spanning DeepSeek-V3.2, DeepSeek-V4 Flash, and DeepSeek-V4 Pro indexers." — same file, lines 15-17.
- Benchmark grid **[MEASURED]**: K = 512 (V4 Flash, 4× compression), 1024 (V4 Pro, 4×), 2048 (V3.2, 1×); valid row lengths 1,027–262,127; batch (rows) 1..1024; 275 distinct shapes; cold-L2 kernel time excluding compilation and Python — `blog29...md`, "Benchmark Setup" table.
- Per-shape extremes **[MEASURED]**: best shape-average 19.85× (V4 Flash) / 20.18× (V4 Pro) at N = 4099, B = 1024, and 10.96× for V3.2 at N = 4111; worst shape-average still 1.55× / 1.52× / 3.50×; minimum over individual cases 1.34× vs radix — `blog29...md`, "The Gains Extend Beyond an Average".
- Roofline **[MEASURED]**: ideal traffic `Q_min = 4B(N+K)` bytes, intensity `I = N/(4(N+K))`, calibrated B200 roof **6.912 TB/s sustained read** and **37.047 Tcompare/s**. Reachable rate (fraction of the one-read bound actually achieved), average / peak:
  - GVR V2 — 41.6 %/77.8 % (Flash), 39.0 %/68.4 % (Pro), 41.5 %/66.5 % (V3.2)
  - GVR V1 — 26.1 %/63.3 %, 25.4 %/58.9 %, 27.8 %/51.1 %
  - **TensorRT-LLM radix CUDA — 7.7 %/16.9 %, 7.8 %/16.4 %, 8.3 %/17.8 %**
  — `blog29...md`, "Compare Pareto Curves and Reachable Rates" table.
- Algorithm: replace the temporal hint of V1 with **self-sampling of the current row** (packed sample windows spread across the row → a primary threshold, a lower safety floor, an upper anchor) plus **multi-thresholding** (one classification pass that yields exact population counts at many thresholds), then exact refinement in the **crossing bin** containing rank K. Exactness contract is unchanged: "Guess–Verify–Refine", returns "an exact value multiset through K distinct indices; ties can choose different positions" — `blog29...md`, "The operator contract" and the V1↔V2 comparison table.
- Design lineage: "GVR V2's self-sampling was inspired by Floyd and Rivest's 1975 paper, *Expected Time Bounds for Selection*" — `blog29...md`, "From Floyd–Rivest SELECT to GPU Top-K".
- Honest limitation, stated by the authors: P5 speedup vs V1 is 1.10× but the **minimum is 0.689×** (V2 ~45 % slower than V1 in the worst measured case), attributed to V2 under-admitting and needing a rescan at a lower threshold — `blog29...md`, table after Figure 3.

**Other stacks**
- `cpp/tensorrt_llm/kernels/fusedSampling/` contains `fusedSamplingKernels.cu` (52 lines — dispatcher), `fusedSamplingSingleCta.cu`, `fusedSamplingMultiCta.cu`, `fusedSamplingKernelsCommon.cuh`; entry via `cpp/tensorrt_llm/thop/fusedSamplingOp.cpp` and Python `cpp/tensorrt_llm/executor/samplingConfig.cpp`.
- `cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.cu` (1,968 lines) with `glmDsaHistTopK.h` / `thop/glmMoeDsa/glmDsaHistTopKOp.cpp`.
- There is also a full engineering casebook entry for this exact problem, `3rd/TensorRT-LLM/.claude/skills/perf-optimization-casebook/references/kernel-and-fusion/specialize-topk-selection-kernel.md`, which records the PR sequence (`#8882` custom topk, `#9255` coarse fp16 short-circuit, `#12236` CuteDSL single-pass multi-CTA chosen by "does `num_rows·ctas_per_group ≤ num_sms`", `#14268` single-block vs multi-pass radix with per-row DRAM ping-pong scratch, `#12385` temporal warm-start, `#15381` primitives) and states the regime rule verbatim: **"single-CTA vs single-pass multi-CTA (CuteDSL, chosen by whether the launch fits one SM wave `num_rows·ctas_per_group ≤ num_sms`, folding dtype-dependent sync cost — fp32 = 4 radix rounds vs fp16/bf16 = 2 — into the crossover)"**. It reports "PR #12236 reports ~1.14× over always-single-CTA for the CuteDSL path; treat as the author's figure".
- That casebook also states the accuracy contract plainly: "**lossless** — every variant returns the exact top-k (the coarse pass only brackets and falls back to exact; radix-select is exact). One flag: the temporal warm-start (#12385) has a bounded refinement loop that *targets* exactness ... but could deviate under pathological non-convergence" — same file.

### Inferences
- The 7.7–8.3 % reachable rate of production radix is the single most useful calibration point for our design: it means the incumbent does **~12 full-pass-equivalents** of real traffic where 1 is ideal [COMPUTED from 1/0.083 ≈ 12]. Attention.rs's 33-pass design would land near 3 % [COMPUTED], i.e. off the bottom of that chart.
- GVR V2's 66–78 % peak at B = 1024 shows the ceiling for a *streaming* design is a little over 1.3 passes of real traffic [COMPUTED from 1/0.778]. That is the number to beat in `csrc/sampling`.
- The 11/11/11/10 split is deliberately chosen so that the *first* pass can be done on fp16-derived keys (11 bits fit a `__half` exponent+mantissa slice), meaning a bf16-logits sampler can run its entire first histogram with 16-bit loads and no conversion — directly applicable to our BF16 input case.
- The "one SM wave" rule (`num_rows·ctas_per_group ≤ num_sms`) is the cleanest formulation I found for the B-vs-parallelism switch, and it maps exactly onto our B = 1..512 range: at B = 1 on B200 (148 SMs) you want ~100+ CTAs per row; at B = 512 you want 1 CTA per row and no cross-CTA barriers at all.

### Gaps
- No absolute microsecond numbers are published for `indexerTopK` alone; only ratios and reachable-rate percentages. I could not find a "µs at V=128K, K=64, B=256" datapoint anywhere.
- The `fusedSampling` top-k/top-p token sampler does not expose its algorithm in a header comment I could cite; I did not read the 1–2 kLOC of `fusedSamplingMultiCta.cu` closely enough to characterize it. Treat TRT-LLM's *sampler* (as opposed to its *indexer top-k*) as uncharacterized here.

---

## Q5. FlashInfer, and the GPU top-k literature (AIR TopK, Dr. Top-k, RadiK, Qrita, RTop-K)

### Takeaway
Two distinct lines of work matter. **Selection** (return the top-k *set*) is dominated by radix/bucket methods: AIR TopK and RadiK are the strongest published exact algorithms, and FlashInfer's `top_k` is the best-documented production instance (three backends: native FilteredTopK radix, SM100 clusters, CUB `DeviceBatchedTopK`). **Sampling** (draw one token) is dominated by **sorting-free rejection sampling**: FlashInfer's dual-pivot rejection sampler has a proven `O(log(1/ε))` round bound and is distribution-exact. For a fused sampler, the rejection approach sidesteps top-k selection entirely — but each rejection round is a full-vocabulary pass, which is the wrong trade at B = 512.

### Cited Findings

**FlashInfer `top_k` (selection)**
- Dispatches between "the native radix-based kernels, the clusters kernel (SM100), and CUB's `DeviceBatchedTopK`, based on shape, dtype, and the requested modes. Set `FLASHINFER_TOPK_ALGO` (`"default"`, `"clusters"`, `"cub"`) to force a specific backend." — https://docs.flashinfer.ai/api/topk.html
- Complexity claim: "The radix-based algorithm is O(n) in vocabulary size, compared to O(n log k) for heap-based methods, making it faster for large vocabularies. ... For small vocabularies (< 1000), `torch.topk` may be faster." — same page.
- `FilteredTopK` resource contract: "requires **128 KB dynamic shared memory**", checked by `can_implement_filtered_topk()`; native path additionally requires **k ≤ 2048** — https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/topk.py and https://docs.flashinfer.ai/api/topk.html
- Tie handling is a first-class API: `tie_break` ∈ {0 = none, 1 = prefer smaller index, 2 = prefer larger index}, with the explicit caveat that "Tie-breaking controls which boundary elements are selected; it does not imply deterministic output ordering. Set `deterministic=True` when repeatable output ordering is also required." — https://docs.flashinfer.ai/generated/flashinfer.top_k_ragged_transform.html
- CUDA-graph safety: `dsa_graph_safe=True` "forces FilteredTopK with graph-safe vectorization (`VEC_SIZE=1`)" — same page. (vLLM's copy exposes the same knob: `ComputeFilteredTopKVecSize`, `MAX_VEC = 16/sizeof(DType)` → 4 for fp32, 8 for fp16/bf16 — `3rd/vllm/csrc/libtorch_stable/persistent_topk.cuh:1320-1358`.)
- Sorted output is opt-in: "Unlike `torch.topk`, the default behavior returns unsorted results for better performance" — https://docs.flashinfer.ai/generated/flashinfer.top_k.html
- The same kernel has fused index-transform variants (`top_k_page_table_transform`, `top_k_ragged_transform`, `top_k_varlen`) that add the page-table/offset in the same launch — https://docs.flashinfer.ai/api/topk.html

**FlashInfer sampling (rejection, not selection)**
- "In FlashInfer, we show that sampling under filtering can be done in sorting-free manner, and we introduce the **Dual Pivot Rejection Sampling** algorithm ... ultimately achieving **logarithmic (in worst case) time complexity**." — https://flashinfer.ai/2025/03/10/sampling.html
- Dual-pivot rule: `low ← 0`, `high ← max p_i`; inverse-transform-sample over `(low, ∞)`; on drawing token `j`, set `pivot1 ← p_j`, `pivot2 ← (pivot1+high)/2`; accept if `f(pivot1)=1`; else shrink `[low,high]` by at least half each round ⇒ "the number of rounds is `O(log(1/ε))`" — same page.
- Correctness proof sketch: the returned distribution is `Pr[output=j] = p_j / Z` where `Z` is the mass of the top-k set — i.e. **exactly** equivalent to filter-then-sample, established by a contraction argument on `W(τ)/S(τ) < 1` — same page.
- Practical round count: "the number of rounds for returning a sampled token is usually small" — same page (no distribution given).
- Implementation warning worth heeding: "Due to the non-associative and non-commutative nature of floating-point arithmetic, parallel prefix-sum **cannot guarantee monotonic outputs** even with non-negative inputs. This can lead to invalid token generation if not handled properly." — same page.
- API: `top_k_top_p_sampling_from_logits(logits, top_k, top_p, ..., filter_apply_order='top_k_first'|'joint', deterministic=True)` — "the multiple rounds of rejection sampling are implemented in a single CUDA kernel" — https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html
- Adoption: "MLC-LLM, sglang, and vLLM (PR #7137)" — https://flashinfer.ai/2025/03/10/sampling.html
- End-to-end claim **[MEASURED, qualitative]**: "In the vLLM 1xH100 configuration, our kernels reduce the overall sampling time by more than 50% across all three models." — same page, Figure 1 caption.

**AIR TopK (SC'23)**
- "AIR Top-K employs an iteration-fused design to minimize CPU-GPU communication and device data access" **[MEASURED]**: "For batch sizes 1 and 100, respectively, AIR Top-K shows **1.98–21.48×** and **8.01–574.78×** speedup over previous radix top-K algorithm, and **1.44–7.34×** and **1.38–31.91×** speedup over state-of-the-art" — https://dl.acm.org/doi/abs/10.1145/3581784.3607062
- Shipped in RAFT: "GPU-efficient exact method from the RAFT library ... which implements Air Top-K and GridSelect as developed by Zhang et al. (2023)" — https://arxiv.org/html/2412.04358v1

**Dr. Top-k (SC'21)**
- Delegate-centric: "maximum delegate, delegate-based filtering, and β delegate mechanisms to help reduce the workload for top-k **up to more than 99%**" — https://arxiv.org/abs/2109.08219
- Applicability caveat from a later paper: "Dr. Top-k, AIR Top-k, and RadiK are designed for scenarios where the vector size is on the order of 2^20 (about one million elements), and the batch size does not exceed 100" — https://arxiv.org/html/2409.00822v1

**RadiK (ICS'24)**
- "a massively parallel radix top-k algorithm that not only supports much greater k than the prior art, but also preserves high efficiency for large k and input lengths, **regardless of the batch size**"; its benefit is "insensitive to k" whereas Dr. Top-k's "benefit decays as k increases" — https://dl.acm.org/doi/fullHtml/10.1145/3650200.3656596

**RTop-K (2024)**
- Row-wise top-k for neural networks, pivot-based truncation; argues that warp-level primitives can beat shared memory for the merge step — https://arxiv.org/html/2409.00822v1

**Qrita (2026)**
- "Gaussian-based σ-truncation, which greatly reduces the search space of the vocabulary" + "**Quaternary pivot search with duplication handling, which halves the number of pivot search iterations and guarantees deterministic output**"; implemented in Triton **[MEASURED]**: "improving end-to-end serving throughput **up to 1.4 times with half the memory usage**, while providing the same output as the sorting-based algorithms"; "Qrita is now the default Top-k and Top-p sampler for the GPU execution path of vLLM" — https://arxiv.org/abs/2602.01518
- **Conflicting claim:** the v1 abstract as summarized by a third party states "up to **2 times** throughput and half memory use" — https://tldr.takara.ai/p/2602.01518 — while the v2 abstract says 1.4× end-to-end. The two are probably different baselines (kernel-level vs end-to-end); prefer the v2/arXiv figure (1.4×) and treat 2× as superseded.

**Cross-cutting measurement**
- "our experiments ... show that top-k consumes **>10 % of the time to generate each token**" in a long-context setting — https://arxiv.org/html/2412.04358v1 (motivating the approximate-top-k paper; note this is the *authors'* framing for a different workload).

### Inferences
- The literature splits cleanly by *contract*: if you need the **set** (for renorm, for MoE routing, for DSA), radix/bucket selection is right and AIR TopK / RadiK are the reference algorithms; if you need only the **draw**, rejection sampling is asymptotically better and provably exact, and is what three of the four major engines actually ship.
- FlashInfer's `tie_break` API existing at all is evidence that boundary-tie behavior is a real correctness/compat issue in production, not a theoretical nicety.
- Qrita's determinism claim ("guarantees deterministic output" via duplication handling) is the only one in this survey that ties exactness *and* reproducibility together in a pivot-search design; that is the strongest candidate to copy for a sampler that must be graph-replayable.

### Gaps
- No head-to-head measured comparison of AIR TopK vs RadiK vs FlashInfer FilteredTopK on the same V/K/B grid exists in any source I found; the speedups are all relative to each paper's own baseline.
- FlashInfer's blog gives no absolute latency numbers (only Figures 5/6 which I could not read numerically from the fetched text).
- I did not find a 2024–2026 paper that benchmarks top-k selection specifically at **V ≈ 128K with B = 512 and K = 32/64** — the regime our sampler targets. All published grids are either (V ≈ 2^20, B ≤ 100) for the selection papers or (V ≈ 128K, B = 1..1024) for the TRT-LLM indexer blog, which is K = 512..2048 rather than K = 32..64.

---

## Q6. Algorithm families: passes, traffic, shared-memory footprint, best regime, exactness, determinism

### Takeaway
Every credible design reduces to the same skeleton — **one coarse histogram pass → bracket the boundary bin → finish inside shared memory** — and the differences that matter are (i) how many bits per pass, (ii) whether survivors are *compacted* or merely *filtered*, and (iii) whether the last step sorts. Full bitonic sort loses by a wide margin at any V ≥ 4K. Pure rejection sampling wins on traffic-per-draw but loses when batch is large.

### Cited Findings

Measured / documented anchors for the table:
- TensorRT-LLM 11/11/11/10-bit, 2048 bins, 512 threads, boundary bin finished in smem if ≤ 2048 items — `indexerTopK.cu:41-46`, `:172-383`, `:393`.
- SGLang 8-bit coarse (fp16 key) + 4 × 8-bit refinement on ≤ 4096 staged candidates, 1024 threads, 32 KiB smem — `topk.cu:23-39`, `:186-251`.
- vLLM/FlashInfer FilteredTopK: 256 bins, 4 refine rounds from shift 24, 1024 threads, 128 KiB dynamic smem, k ≤ 2048 — `persistent_topk.cuh:1009-1052`, https://docs.flashinfer.ai/api/topk.html
- attention.rs: 32 × 1-bit passes with no compaction, 256 threads, 40 B smem — `fast_topk.cu:169-179`, `:275`.
- GVR V2: self-sampling + multi-threshold, one classification pass yielding many exact counts, refinement in the crossing bin — `blog29...md`.
- Qrita: σ-truncation + quaternary pivot, Triton, no sort — https://arxiv.org/abs/2602.01518
- Dual-pivot rejection: `O(log(1/ε))` full-vocab rounds, exact distribution — https://flashinfer.ai/2025/03/10/sampling.html

| Family | Full-row passes (typical) | Traffic @V=128K, B=512, fp32 [COMPUTED] | Smem footprint | Best regime | Exact? | Tie determinism |
|---|---|---|---|---|---|---|
| **Full bitonic sort** (whole row) | 1 read + `O(V log²V)` smem compare-exchange | 1 pass read + heavy smem work | must hold whole row: 4V B = 501 KiB @V=128K → **does not fit** | V ≤ 2–4 K only | yes | yes (index tie-break) |
| **Partial bitonic / iterative argmax** (attention.rs small path) | K sequential sweeps of smem | n/a (V ≤ 4096 only) | 8 B/elem (4 wasted) — `fast_topk.cu:76-77` | V ≤ 4K, K ≤ 8 | yes | yes (smaller index) — `fast_topk.cu:96` |
| **Naive radix, no compaction** (attention.rs) | up to **33** | ≈ 8.5 GB → 2.5 ms H100 / 1.1 ms B200 | 40 B — `fast_topk.cu:275` | none at V ≥ 100K | yes (modulo ties) | **no** (atomic race) — `fast_topk.cu:224-232` |
| **Radix + smem staging, 8-bit** (SGLang) | ≈ **2** + ≤4 small gathers | ≈ 513 MB → 153 µs H100 / 64 µs B200 | 32 KiB — `topk.cu:38` | V ≥ 8K, K = 2048 fixed | yes (ties exempt in test) | **no** (atomic) — `topk.cu:168` |
| **Radix + smem staging, 11-bit ×4** (TRT-LLM) | ≈ 2–4 (first on fp16 key) | ≈ 513 MB–1.0 GB → 150–310 µs H100 | 8 KiB hist + `topK×4` out — `indexerTopK.cu:944` | 12 288 ≤ V < 200 K, K ≤ 2048 | yes | **no** by default; deterministic-collect variant exists in FlashInfer |
| **Bucket / mass-histogram** (GVR V2 style) | **1** classification + refinement in crossing bin | ≈ 1.0–1.5 passes → 77–115 µs H100 | histogram + sample windows + crossing-bin candidates | any V, B ≥ #SMs | yes (exact counts) | yes if collect is deterministic |
| **Pivot search / σ-truncation** (Qrita) | 1 truncation pass + a few pivot sweeps over a *truncated* row | < 1 pass if σ-fit is good | low (per-row tiles) | sampling (top-k **and** top-p) | yes (duplication-handled) | **yes** (claimed) |
| **Dual-pivot rejection** (FlashInfer) | `O(log(1/ε))` full-vocab rounds, typically a few | worst case ≈ 3–5 passes | low | B small, no top-k list needed | yes (proved `p_j/Z`) | distribution-level; per-draw RNG |
| **Delegate / hierarchical** (Dr. Top-k, AIR TopK, RadiK) | 1 delegate pass + per-subrange selection | n/a (targets V ≈ 2^20, B ≤ 100) | subrange-sized | V ≥ 2^20, B ≤ 100 | yes | algorithm-dependent |
| **Heap / priority queue** (`O(n log k)`) | 1 | 1 pass | `k` entries per thread/warp | K tiny (≤ 8), V small | yes | yes |

Additional cited specifics for the table:
- Full bitonic cannot hold a 128K row in shared memory: 128256 × 4 B = 501 KiB per row vs 228 KiB/SM opt-in on H100 and 227 KiB on B200 — https://docs.flashinfer.ai/api/topk.html states FilteredTopK needs 128 KiB and `can_implement_filtered_topk()` exists precisely because that does not fit on SM120/SM121.
- "For small vocabularies (< 1000), `torch.topk` may be faster" than radix — https://docs.flashinfer.ai/api/topk.html
- "For larger n, computing top-k can be slow" and hierarchical/merge designs "limit parallelism", while bucketed-approximate designs trade exactness for it — https://arxiv.org/html/2412.04358v1
- Pass-count-to-efficiency mapping is empirically anchored: 7.7–8.3 % reachable rate for radix CUDA vs 66.5–77.8 % peak for a ~1-pass streaming design — `blog29...md`, reachable-rate table.

### Inferences
- The table makes the design choice almost forced: at V ≥ 100K **only** compaction-based (SGLang / TRT-LLM / FilteredTopK) or histogram-with-mass (GVR V2) designs are viable; anything that re-streams the row per bit is dead.
- "8-bit × 4 passes" and "11-bit × 4 passes" have identical asymptotic traffic but 11-bit is better for us: (a) the first 11 bits of an fp16/bf16 key need no up-conversion, (b) 2048 bins × 4 B = 8 KiB of histogram is a third of SGLang's 256-bin × 2 buffers + staging, leaving more smem for candidate staging, and (c) 2048 bins make the boundary bin ~V/2048 ≈ 63 elements at V = 128K *before* any refinement — for K = 32/64 that bin is often already smaller than K.
- The "finished in shared memory if the boundary bin ≤ 2048" rule (`kNumFinalItems`, `indexerTopK.cu:393`) means for K = 32/64 at V = 128K the **second pass is usually unnecessary**: after an 11-bit histogram the crossing bin holds ~63 candidates on average, so you can finish with a 63-element sort. That is a genuinely different regime from TRT-LLM's K = 512..2048, where the boundary bin can exceed the staging buffer.
- Determinism must come from a **deterministic collect** (ordered position via prefix-sum over the survivor mask, not `atomicAdd`), otherwise the sampler is not CUDA-graph-replay-safe in the bitwise sense. None of the three local kernels does this; FlashInfer exposes it as `deterministic=True`.

### Gaps
- Shared-memory capacity per SM for H100/B200 is not stated in any of the local sources I read; the 128 KiB FilteredTopK requirement and the SM120 exclusion are the only hard data points.
- No source gives a measured "passes vs. reachable-rate" curve for K = 32/64 specifically; the 12×-passes-at-8 % inference above is my arithmetic on the TRT-LLM table, not a published result.

---

## Q7. Best fit for a fused sampler that reads logits once and samples exactly from the full vocab (top-p included)

### Takeaway
Use a **single-pass mass-histogram (bucket) selector** as the core: one CTA per row when B ≥ #SMs, split-row when B < #SMs; accumulate **both `count` and `Σexp(x−max)` per bin** so the same pass yields the top-k boundary *and* the top-p mass boundary; finish the crossing bin in shared memory. Then either sort the ≤K survivors (deterministic, sorted output) or run a CDF draw directly. Do **not** copy attention.rs's radix (33 passes) or a full bitonic sort. Rejection sampling is the right alternative only at small B and when you do not need the k-list.

### Cited Findings
- The fused-sampler contract already exists in this repo and mirrors FlashInfer's decomposition: `sample()` — "greedy rows via argmax, others via top-k/top-p/min-p rejection sampling or Gumbel-max, chosen per capability"; workspace "holds the fp32 probability row of the call (`num_tokens * vocab * 4` bytes, plus the optional split-softmax state)"; "A row with no admissible token writes `out_ids = -1`" — `csrc/sampling/ops.h:9-20`. Sibling entry points `top_k_renorm_probs`, `top_p_renorm_probs`, `top_k_mask_logits`, `gumbel_max_sample`, `rejection_sample` are already declared — `csrc/sampling/ops.h:22-51`.
- Top-p is a **mass** threshold, so counts alone are insufficient: TRT-LLM's admission function is explicitly a monotone count `C(T)=Σ 1[x_i ≥ T]` (`blog29...md`), whereas FlashInfer's rejection sampler verifies the *mass* `q ≥ top_p` each round (https://flashinfer.ai/2025/03/10/sampling.html). A per-bin `Σ exp` histogram gives you the mass version of `C(T)` for free in the same sweep.
- "Every additional full-row pass pays that input cost again" — the reason to fuse the mass accumulation into pass 1 rather than doing a second sweep for top-p (`blog29...md:11-13`).
- Selective-parallelism rule from production: single-pass multi-CTA is chosen when `num_rows·ctas_per_group ≤ num_sms`, "folding dtype-dependent sync cost — fp32 = 4 radix rounds vs fp16/bf16 = 2 — into the crossover" — `3rd/TensorRT-LLM/.claude/skills/.../specialize-topk-selection-kernel.md`. That is the exact rule to encode in a `csrc/sampling` policy: at B = 512 on B200 (148 SMs) use 1 CTA/row; at B = 1 use ~128 CTAs/row.
- vLLM independently reached the same conclusion and added a split-row pipeline because "at small batch sizes nearly all SMs idle and every full-vocab sweep is latency-bound (~25us each on SM100)" — `topk_topp_triton.py` split-pipeline docstring.
- Determinism knobs exist and are cheap to copy: FlashInfer `deterministic=True` + `tie_break ∈ {none, smaller-index, larger-index}` (https://docs.flashinfer.ai/generated/flashinfer.top_k_ragged_transform.html); Qrita claims determinism "by multi-pivot bookkeeping of duplicate logit values" (https://arxiv.org/abs/2602.01518); and none of the three local CUDA kernels achieves it (`fast_topk.cu:224-232`, `topk.cu:168`, `indexerTopK.cu:303-312` all use unordered `atomicAdd` slots).
- CUDA-graph safety is an explicit constraint in the existing contract — "Plan and workspace requirements are resolved before capture ... Keep plan capacity, backing addresses, and `capture_generation` valid across CUDA Graph replay; replan or recapture after those resources change" — `AGENTS.md` (Ownership and execution contracts). FlashInfer honors the same with `dsa_graph_safe` forcing `VEC_SIZE=1` (https://docs.flashinfer.ai/api/topk.html); vLLM has a `graph_safe` tier in `persistent_topk.cuh` (`ComputeFilteredTopKVecSize`, `:1320-1358`).
- BF16 is the right input dtype to optimize for: TRT-LLM's 11-bit first pass maps directly to the fp16/bf16 bit layout (`indexerTopK.cu:41-46`), and their dispatcher notes fp16/bf16 needs only **2** radix rounds vs **4** for fp32 in the CuteDSL tier (`specialize-topk-selection-kernel.md`).
- Numeric hazard to respect during the CDF draw: "parallel prefix-sum cannot guarantee monotonic outputs even with non-negative inputs. This can lead to invalid token generation if not handled properly" — https://flashinfer.ai/2025/03/10/sampling.html. The repo's own rule is presumably the same as FlashInfer's: do the final CDF walk with a stable, sequentially-consistent scan or an explicit carry.

### Recommended design (inference, not a cited fact)
1. **Pass 1 — fused histogram + mass.** One CTA (1024 threads) per row, `float4`/`uint4` loads, no `__syncthreads` inside the sweep. Build a 2048-bin histogram over the top 11 bits of the monotone key, and per-bin `Σ exp(x − running_max)` (or use a fixed `max` obtained from a cheap warp/CTA max reduction first — the max is needed for `exp` stability). Cost: **1 read of V**.
2. **Bracket.** Suffix-sum the 2048 bins. For top-k: find the bin where the cumulative count crosses K. For top-p: find the bin where the cumulative *mass* crosses `p·Z`. For top-k-then-top-p (the default order in both vLLM and FlashInfer), take the min of the two thresholds.
3. **Pass 2 — compact survivors.** Re-read only elements above the threshold *and* stage the crossing bin into smem, capped at ~4096 entries (SGLang's `SMEM_INPUT_SIZE`). For V = 128K and 2048 bins the crossing bin averages ~63 elements, so the staging buffer is never the constraint and the survivors above-threshold are ≤ K + 63.
4. **Finish.** If survivors ≤ 128, a single warp does a bitonic sort (deterministic, index tie-break) and writes descending indices; then a CDF draw over the exact mass. No third kernel, no full-vocab sweep.
5. **Determinism.** Write output slots via a prefix-sum over the survivor mask (stable index order), not `atomicAdd`; expose `tie_break` and `deterministic` as plan-time flags.
6. **Regime policy.** `ctas_per_group = clamp(floor(num_sms / batch), 1, ceil(V / 16384))`, resolved at plan time and stored in the workspace descriptor (so it survives graph capture). This is the TRT-LLM "one SM wave" rule and the vLLM split-row rule unified.
7. **What to avoid.** Full bitonic (does not fit: 501 KiB/row), iterative argmax (O(K) sweeps), and the local 32×1-bit radix (≈8.5 GB of traffic at B=512).

### Inferences
- Because K = 32/64 is *much* smaller than the K = 512..2048 that TensorRT-LLM and SGLang target, the boundary-bin refinement is essentially free for us, and a design optimized for their regime will over-provision `s_indices` staging (SGLang's `__shared__ int s_indices[2048]` = 8 KiB, `topk.cu:286`) and under-provision on parallelism.
- An 11-bit single-pass design at V = 128K would land at roughly the "1.0–1.3 passes" traffic of GVR V2 [COMPUTED], which is 5–8× better than the 7.7–8.3 % reachable rate of radix CUDA (`blog29...md`). That is the realistic target.
- The softmax is already needed for sampling, so its `max` and `Z` are free byproducts of the same pass — the sampler should never do a separate softmax kernel before the histogram, which is what `csrc/sampling/ops.h:26` (`softmax`) currently implies as a separate stage.

### Gaps
- I did not find a published measurement for a **mass-histogram** (count+Σexp per bin) top-p design; GVR V2 verifies counts, Qrita does pivot search on mass, FlashInfer does rejection on mass. Whether a single histogram can serve both exact top-k and exact top-p within the same pass is my design proposal, not a cited result.
- No source quantifies the cost of deterministic (prefix-sum) collect vs `atomicAdd` collect; the FlashInfer `deterministic=True` path is documented as existing but not benchmarked against the nondeterministic one.
- Whether BF16 logits should be histogrammed directly on the 16-bit key (2 × 8-bit passes) or up-converted to fp32 (11/11/11/10) is unresolved: TRT-LLM chose up-conversion for *accuracy parity* (`indexerTopK.cu:907-910`) but its 11-bit *first* pass already operates on the 16-bit representation. I found no measurement of the accuracy delta for a bf16-key-only 2-pass design at V = 128K.
- I have no measurement of the actual footprint/latency of any of these designs on **B200**; all B200 figures in this document are either quoted from the TRT-LLM blog (ratios and reachable rates) or my arithmetic against an assumed 8 TB/s spec bandwidth.
