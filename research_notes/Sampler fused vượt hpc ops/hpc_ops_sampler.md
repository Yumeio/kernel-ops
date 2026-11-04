# hpc-ops fused sampler: mechanics, speed, semantics and limits

Evidence base: the vendored snapshot at `3rd/hpc-ops/` in this repo (all paths below are
relative to that directory; `:Lnnn` = line number in the local file). No measured
sampler numbers are committed anywhere in the tree, so every latency figure below is
either **MEASURED-BY-REPO** (i.e. a repo claim, not a raw measurement) or
**ESTIMATE** (my model, assumptions stated inline). Structural claims are all
**CODE-FACT** with line citations.

Upstream project for context: Tencent/hpc-ops, an LLM inference operator library
optimized for Hopper/H20 — [github.com/Tencent/hpc-ops](https://github.com/Tencent/hpc-ops).

---

## Kernel map and pipeline: what K1 and K2 actually do

### Takeaway
K1 is a **single full-vocab streaming pass** with a *rolling in-register top-K* per CTA,
implemented by re-sorting all 4096 of the block's register values with
`cub::BlockRadixSort` **on every load iteration**. K2 merges the N per-block top-K lists
with a second block radix sort, then one warp does softmax/top-p/Gumbel-max/writeback.
Grid is `(N, B)` for K1 and `(B,)` for K2; `Nmax = min(1024/K, 32)`; K2's block is always
1024 threads.

### Cited Findings
- Pipeline is documented in the file header as
  `rp -> temp -> [softmax1] -> topk -> [softmax2] -> topp -> Gumbel-max -> penalty writeback`,
  with K1 = `fused_scan_topk_kernel`, K2 = `stage2_kernel` — [fused_sampler.cu:1-13](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- K1: `__launch_bounds__(1024)`, `grid=(n_blocks_per_row, batch_size)`, template
  `<DType, kVocabSize, kMaxTopK, kSoftmaxPolicy>` — [fused_sampler.cu:161-172](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:654-660](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Thread roles: `kKeeperThreads = kMaxTopK/4` (8 for K=32, 16 for K=64),
  `kLoaderThreads = 1024 - kKeeperThreads` (1016 / 1008); keepers hold the rolling top-K
  in `thread_logits[4]`/`thread_tokens[4]` initialised to `-inf/-1` —
  [fused_sampler.cu:176-180](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:203-214](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- K1 shared memory: a **union** of `cub::BlockRadixSort<float,1024,4,int>::TempStorage`
  and `float warp_scratch[32]` — i.e. the sort dominates smem, and the max/sum reduce
  reuses the same bytes — [fused_sampler.cu:196-201](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:115-155](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Load loop is runtime-bounded and not unrolled: `elements_per_load_loop =
  N * kLoaderThreads * 4`, `num_load_loops = ceil(V / elements_per_load_loop)`,
  `#pragma unroll 1` — [fused_sampler.cu:221-229](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- **The sort is inside the loop**: `__syncthreads(); BlockRadixSort.SortDescending(...);
  __syncthreads();` executes `num_load_loops` times over all 1024 threads × 4 items
  (4096 items), which is what makes keepers retain the running top-K —
  [fused_sampler.cu:260-263](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- `Nmax` is compile-time and derived from the 1024-thread K2 block:
  `kMaxBlocksPerBatch = min(1024/kMaxTopK, 32)` → **32 for K=32, 16 for K=64** —
  [fused_sampler.cu:42-46](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). K2's block is
  `kNmax*kMaxTopK` = 1024 in both cases — [fused_sampler.cu:307-308](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:663](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Runtime `N = clamp(floor(SMs/B), kNMin=8, Nmax)`; the comment states K1 is limited to
  **1 block/SM**, so `floor` keeps the launch to a single wave — [fused_sampler.cu:610-627](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). `kNMin = 8` — [fused_sampler.cu:616](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- K2: `grid=(B,)`, block 1024, `cub::BlockRadixSort<float,1024,1,int>`; candidates from
  block ids `>= N` are masked to `-inf/-1`; after the merge sort **only warp 0 continues**
  (`if (tid >= 32) return;`) — [fused_sampler.cu:354-401](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- K2 mechanics after the merge: load top-`kMaxTopK` into 32 lanes (`kItemsPerLane =
  kMaxTopK/32` → 1 or 2), optional softmax, optional top-p prefix scan, Gumbel-max,
  warp argmax with tie-break, 1 thread writes the token — [fused_sampler.cu:403-579](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Effective K is resolved **inside K2**, not on the host: `user_k > 0 → effK =
  min(user_k, kMaxTopK)`, else `effK = kMaxTopK` — [fused_sampler.cu:313-327](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Template axes are `DType ∈ {float, bf16}`, `kVocabSize`, `kMaxTopK ∈ {32,64}`,
  `kSoftmaxPolicy ∈ {NONE=0, BEFORE_TOPK=1, AFTER_TOPK=2}`, `kHasTopP ∈ {true,false}`,
  dispatched by nested lambdas — [fused_sampler.cu:37-40](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:704-772](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).

### Inferences
- The rolling-top-K invariant is: after each sort, threads 0..K/4-1 hold the K largest of
  *(everything seen so far) ∪ (this iteration's 4·1016 values)*, because previously-held
  keepers' values are re-inserted into the sort. It is correct, but it means **one full
  4096-item radix sort per 4064 loaded elements** (1016 loaders × 4) — i.e. ≈1.008
  sort-items per vocabulary element, independent of N, K and batch. Arithmetic:
  4096/4064 = 1.0079 (K=32); 4096/4032 = 1.016 (K=64).
- Because CUB's `BlockRadixSort` defaults to `RADIX_BITS = 4` (verified in the local
  toolkit header, `cuda12.9/include/cub/block/block_radix_sort.cuh:247`), a 32-bit float key
  costs **8 radix passes**, each with a block-wide rank + scatter over 4096 (key,value)
  pairs. That is ~8× the register/smem traffic of a single-pass max.
- The "1 block/SM" assumption is a *design* assumption, not an enforced one: K1 declares
  `__launch_bounds__(1024)` with a one-argument form, so the compiler may target 2 blocks
  of 1024 (needs ≤32 regs/thread). Either way, `grid = N*B` blocks means the *launch* is
  sized for one wave; occupancy only changes whether the wave is under-filled.
- K2's `if (tid >= 32) return;` means **31 of 32 warps idle** after the merge sort; K2 is
  a serial 1-warp tail (softmax + scan + 32-64 Gumbel scores) behind a 1024-thread sort.

### Gaps
- No SASS/`-Xptxas -v` output, register count, or achieved occupancy is committed, so the
  true blocks/SM for K1 (1 vs 2) is unverified; the code comment asserts 1.
- No profiler trace (nsys/ncu) is committed, so the split between sort time and memory
  time inside K1 is not measured anywhere in the repo.

---

## Full-vocab traffic per call, and where each stage runs

### Takeaway
**Logits are read exactly once** (K1 only) — the design's best property. Penalties and
temperature are fused into that same load. Softmax/top-p/Gumbel/writeback never touch the
vocab again; the only full-vocab extra read is the `ceil(V/8)`-byte penalty bitmask. Per
call there are 2 kernel launches, 4 `cudaMallocAsync` + 4 `cudaFreeAsync`, and ~8 KB/row of
intermediate traffic for K=32.

### Cited Findings
- Load + penalty + temperature are one fused per-element step in K1:
  `load_logits_safe` (16 B `float4` for fp32, 8 B `uint2` for bf16) → `apply_rp_temp` —
  [fused_sampler.cu:52-113](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:231-234](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Penalty is applied in K1 only, pre-temperature, on the raw loaded value:
  `vals[i] *= (vals[i] > 0.f) ? inv_rp : rp` when the bitmask bit is set —
  [fused_sampler.cu:94-106](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). The mask row is
  addressed by `slot_id[ibatch]` with stride `penalty_row_bytes = (vocab+7)/8` —
  [fused_sampler.cu:186-189](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:639](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Temperature is applied immediately after, in the same kernel, `vals[i] *= inv_t`, with
  `inv_t = rcp.approx.ftz(t)` — [fused_sampler.cu:107-112](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:191-193](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- `BEFORE_TOPK` softmax statistics are accumulated **on the fly during the same pass**
  (per-thread online max+sum, `thread_sum *= expf(thread_max - x)`), then block-reduced
  with a rebase and written as one `(partial_max, partial_sum)` per block —
  [fused_sampler.cu:244-252](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:265-278](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- K2 reduces the N partials with warp 0 and converts candidates to full-vocab
  probabilities `expf(x - global_max) * inv_sum` —
  [fused_sampler.cu:329-352](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:377-385](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- `AFTER_TOPK` softmax is computed **only over the surviving ≤64 candidates**, in warp 0,
  using sorted position 0 as the max — [fused_sampler.cu:417-434](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Top-p runs in warp 0: per-lane sum, `__shfl_up` inclusive scan over 32 lanes, exclusive
  `lane_run`, then `keep = keep && ((idx==0) || (running_cum < topp))` —
  [fused_sampler.cu:445-469](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:502-507](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Gumbel noise is generated in K2 warp 0, one `curand_uniform4` per lane
  (`sequence = ibatch*32 + ilane`), i.e. **≤128 draws per row**; the external-noise path
  instead gathers `gumbel_noise[ibatch*V + tok]` per candidate —
  [fused_sampler.cu:471-489](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:518-525](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Penalty writeback is **one `atomicOr` per row** on a 32-bit-aligned word of the mask
  (`word_addr = byte_addr & ~3`), executed by lane 0 of warp 0 in K2 —
  [fused_sampler.cu:559-579](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Scratch is allocated **per call** with `cudaMallocAsync` and freed with `cudaFreeAsync`
  in an RAII struct: 4 buffers (partial_max, partial_sum, mid_logits, mid_tokens) —
  [fused_sampler.cu:585-607](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:648-651](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). Return codes are not checked.

### Inferences
Traffic per call (bf16, V=120832, K=32, Nmax=32):

| item | bytes/row | B=1 | B=512 |
|---|---|---|---|
| logits read (once) | 241,664 | 236 KB | 118 MB |
| penalty bitmask read (only if rp enabled) | 15,104 | 14.8 KB | 7.4 MB |
| `mid_logits`+`mid_tokens` write | `Nmax*K*8` = 8,192 | 8 KB | 4.0 MB |
| same read by K2 | 8,192 | 8 KB | 4.0 MB |
| `partial_max`+`partial_sum` (BEFORE only) | 256 | 0.25 KB | 0.13 MB |
| token out | 4 | 4 B | 2 KB |
| **total** | ~273 KB | **~267 KB** | **~134 MB** |

At H100's 3.35 TB/s that is 0.08 µs (B=1) and 40 µs (B=512) of pure HBM time — so
**the kernel is nowhere near memory-bound**; everything above that bound is sort, sync and
launch cost. This is the single most important number for "can we beat it": a
single-pass sampler that does O(1) work per element is bounded by ~40 µs at B=512 and
~0.1 µs at B=1, i.e. hpc-ops leaves most of the memory roofline unused.
- The per-call `cudaMallocAsync`/`cudaFreeAsync` pair-set is *stream-ordered* and therefore
  legal to capture as CUDA graph memory nodes (allocation nodes were added in CUDA 11.4
  and `cudaMallocAsync` is documented as capturable; the pointer is fixed at capture) —
  [NVIDIA CUDA Programming Guide, Graph Memory Nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
  [NVIDIA forums: captured cudaMallocAsync gets fixed virtual addresses](https://forums.developer.nvidia.com/t/how-to-use-cuda-graph-with-cudamallocasync/215789).
  So graph capture works, but each replay still executes 8 allocator nodes and the
  addresses are frozen.

### Gaps
- I did not measure the allocator node cost; on a warm pool it is typically sub-µs, but
  that is an assumption, not a repo fact.

---

## Bounded top-K semantics: what it actually samples

### Takeaway
The sampler **always** collapses the vocabulary to `max_topk` (32 or 64) candidates before
drawing. `topk=0` means "do not tighten below 32/64", not "full vocab". Within that
truncated support the top-K selection is *exact* (union of per-block exact top-K), and
`AFTER_TOPK` reproduces vLLM's top-k→top-p→renormalised semantics exactly; `BEFORE_TOPK`
changes only the top-p *threshold arithmetic* (it uses unnormalised full-vocab mass), and
`NONE` reproduces the truncated softmax exactly because Gumbel-max is scale-invariant.
No min-p, no typical-p, no exact full-vocab fallback.

### Cited Findings
- Header contract: "This is a top-`max_topk` (32/64) bounded sampler… `topk=0` (or unset)
  does NOT mean full-vocab sampling — it only means 'do not tighten below max_topk'.
  Low-probability tokens outside the top-`max_topk` are never drawn." —
  [sampler.h:13-20](../../3rd/hpc-ops/src/sampler/sampler.h).
- Python doc repeats it and says "Every feature except the final Gumbel-max is optional…
  this is NOT a full-vocabulary argmax / softmax sample" —
  [hpc/sampler.py:57-68](../../3rd/hpc-ops/hpc/sampler.py), [hpc/sampler.py:90-94](../../3rd/hpc-ops/hpc/sampler.py).
- `max_topk` must be 32 or 64, else the op refuses — [entry.cc:125](../../3rd/hpc-ops/src/sampler/entry.cc),
  [hpc/sampler.py:127-128](../../3rd/hpc-ops/hpc/sampler.py).
- `topk` is **not** validated against `max_topk` on the host; the kernel silently clamps
  with `min(user_k, kMaxTopK)` — [fused_sampler.cu:313-327](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). (The test reference mirrors
  the clamp, `k_b > max_topk → max_topk` — [tests/test_sampler.py:119-122](../../3rd/hpc-ops/tests/test_sampler.py).)
- Softmax policies: `NONE` = no softmax, Gumbel-max on raw logits; `BEFORE_TOPK` =
  "joint topk-topp", full-vocab softmax then topk on probabilities; `AFTER_TOPK` =
  "topk-first", small softmax over survivors — [hpc/sampler.py:8-28](../../3rd/hpc-ops/hpc/sampler.py).
- Host gating rejects nonsensical combinations: top-p requires top-k enabled and
  `softmax_policy != NONE`; softmax without top-p is rejected as "silent overhead" —
  [entry.cc:110-124](../../3rd/hpc-ops/src/sampler/entry.cc).
- Top-p keeps position 0 unconditionally and otherwise keeps while the **exclusive**
  cumulative mass is `< topp` — the standard nucleus definition — [fused_sampler.cu:502-507](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Top-p is evaluated only over the ≤64 ranked candidates (`idx < effK`), never over the
  vocab — [fused_sampler.cu:453-458](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:502](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- No min-p / typical / eta filtering exists anywhere: a grep for `min_p|minp|typical|eta`
  over `src/sampler/*` and `hpc/sampler.py` returns nothing.
- Degenerate inputs: fully masked rows produce zero output / `-inf` LSE by convention in
  kernel-ops, but hpc-ops instead falls back to `tok = (best_tok < 0) ? 0 : best_tok`, i.e.
  **token 0** — [fused_sampler.cu:560-562](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler_temperature.cu:320-322](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu).
- Tests are bit-exact against a torch reference that **also truncates at max_topk**
  (`ref_fused_sampler` docstring: "the kernel always collapses the vocab to the
  top-`max_topk`… the reference mirrors that") — [tests/test_sampler.py:60-68](../../3rd/hpc-ops/tests/test_sampler.py); the
  only distributional test is for the temperature fast path (TV distance < 0.1 over 2000
  draws) — [tests/test_sampler.py:599-625](../../3rd/hpc-ops/tests/test_sampler.py).

### Inferences
- **Exactness of the candidate set:** each block's rolling sort yields that block's exact
  top-32 of its contiguous slice; the union of per-block top-32 contains the global top-32,
  and K2 re-sorts the union, so `effK` is the true global top-K. Ties exactly at the K-th
  boundary are resolved by CUB's (unstable) radix ranking, so *which* of several equal
  logits survives is implementation-defined, though deterministic for a fixed build.
  The final Gumbel comparison adds an explicit "smaller token id wins" tie-break —
  [fused_sampler.cu:528](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:539-557](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- **Scale invariance makes `NONE` == `AFTER_TOPK` in distribution** when top-p is off:
  `argmax(log p + g) = argmax(log p − log Z + g)` for any constant Z, so sampling the
  truncated set with raw logits is identical to renormalised softmax over the truncated
  set. The host actually forbids softmax-without-top-p, so `NONE` is the normal fast mode.
- **`BEFORE_TOPK` only changes top-p.** With top-p off it is mathematically the same
  distribution as `AFTER_TOPK`, at the price of a full-vocab online max+sum (one `expf`
  per element, sometimes two). With top-p on, the cumulative mass is the *unnormalised*
  full-vocab mass restricted to the top-K, so if the top-K mass is below `topp` **nothing
  is ever truncated** (e.g. random N(0,1) logits with V=128k: the top-32 hold ≈0.8 % of
  the mass, so `topp=0.9` degenerates to "keep all 32"). That is a semantic divergence from
  HF/vLLM, which renormalise over the top-k survivors before applying top-p.
  Arithmetic check for the 0.8 %: `Σ_{top32} e^x ≈ 32·e^{4.0} ≈ 1748` vs
  `Z ≈ V·E[e^X] = 128256·1.6487 ≈ 211,470` → 0.83 %.
- **Statistical consequence at high temperature:** the support is capped at 32 tokens
  regardless of T. For T ≫ 1 (or flat logits) the sampler returns essentially a uniform
  draw over the top-32 instead of a near-uniform draw over 128k tokens. Any benchmark that
  only checks top-20 membership (as `test_curand_smoke` does —
  [tests/test_sampler.py:326-337](../../3rd/hpc-ops/tests/test_sampler.py)) will not catch this.
- **NaN / all-`-inf` behaviour** (code reading, not tested): CUB's float twiddle maps
  `+NaN` above `+inf` (CUB `RadixSortTwiddle<float>::In` XORs `0x80000000` into
  non-negative bit patterns, so `0x7FC00000 → 0xFFC00000` sorts above `+inf`); a NaN logit
  therefore becomes candidate #0; `AFTER_TOPK` then computes
  `local_max = NaN`, `sum = NaN`, `inv = 0` (guarded by `sum > 0.f`), all probabilities 0,
  all `value = -inf`, `best_tok = -1` → output token 0. For `NONE`, `NaN + g` never
  satisfies `k > best_key`, so it is skipped. All-`-inf` rows likewise fall out as token 0.
  There is no explicit contract for either case.
- **`topp=0` is a clean disable** (the `topp_b > 0` guard, and `kHasTopP` is only true when
  a tensor is passed or the scalar is > 0 — [fused_sampler.cu:704](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)), but passing an all-zero
  top-p *tensor* still instantiates the `kHasTopP=true` kernel and runs the scan.

### Gaps
- No test covers `topk > max_topk`, `topk` tensor with mixed values, all-`-inf` rows, NaN
  logits, or `topp=1.0`; those behaviours are inferred from code only.
- Whether upstream (non-vendored) hpc-ops has a full-vocab mode could not be checked; the
  vendored snapshot has none.

---

## RNG: generator, seeding, reproducibility, graphs and streams

### Takeaway
cuRAND **Philox-4x32-10**, `curand_init(seed, sequence, offset)` with `sequence` a pure
function of (batch, lane/thread) and `offset` taken from a **host-side global atomic
counter advanced once per launch**. That makes samples irreproducible across processes and
**frozen under CUDA graph replay** (the offset is baked into the captured kernel node).
The fused path is multi-stream-safe (all scratch is per-call); the temperature fast path is
**not**, because it uses a per-device persistent scratch + arrival counter.

### Cited Findings
- Generator, seeding contract and Gumbel transform are centralised:
  `seed` forwarded unchanged; `offset` = host atomic advanced once per launch;
  `curand_init(seed, sequence, offset)`; `curand_uniform4`; noise
  `-log(max(-log(U), 1e-20))` — [sampler_rng.cuh:1-40](../../3rd/hpc-ops/src/sampler/sampler_rng.cuh).
- Offset increment is a fixed 128 per launch, reserved with
  `fetch_add(..., memory_order_relaxed)` — [sampler_rng.cuh:30-52](../../3rd/hpc-ops/src/sampler/sampler_rng.cuh).
- Fused path: `sequence = ibatch*32 + ilane`, **one Philox state per (row, lane)**,
  re-initialised every call; `rng_base_offset = next_launch_offset()` is computed on the
  host at launch — [fused_sampler.cu:478-489](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:665-669](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
- Temperature path: `sequence = (ibatch*scratch_stride + bid)*1024 + tid` (per row, block,
  thread), same host offset — [fused_sampler_temperature.cu:120-127](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu), [fused_sampler_temperature.cu:488-492](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu).
- An external `[B,V]` fp32 Gumbel buffer can replace the RNG entirely, and then results are
  bit-exact vs. the torch reference (tested) — [fused_sampler.cu:521-525](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [tests/test_sampler.py:171-183](../../3rd/hpc-ops/tests/test_sampler.py), [tests/test_sampler.py:443-482](../../3rd/hpc-ops/tests/test_sampler.py).
- Tie-breaking is explicit everywhere: "prefer valid token, then smaller ID"
  ([fused_sampler.cu:551](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)), `take_other` in the temperature kernel
  ([fused_sampler_temperature.cu:55-73](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)).
- Multi-stream prohibition is documented twice for the temperature path only:
  "not safe to call concurrently on multiple streams of the same device (the per-device
  scratch workspace is shared across streams)" — [sampler.h:38-39](../../3rd/hpc-ops/src/sampler/sampler.h), and
  in code: "The counter==0 invariant and the last-block reduction only hold under
  single-stream serial use" — [fused_sampler_temperature.cu:352-372](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu); the Python doc says the same —
  [hpc/sampler.py:118-123](../../3rd/hpc-ops/hpc/sampler.py).
- The workspace is `static WorkspaceCache g_ws[16]` per device, holding `scratch_score`,
  `scratch_tok`, `counter`, grown 2× under a mutex with `cudaStreamSynchronize` +
  `cudaFree` on growth; the counter is zeroed once by `cudaMemsetAsync` and then
  self-reset by the last block — [fused_sampler_temperature.cu:364-433](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu), [fused_sampler_temperature.cu:320-324](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu).
- Verified by reading `fused_sampler.cu`: it has **no** static device state; all scratch is
  per-call `cudaMallocAsync`, so the fused (2-kernel) path really is stream-safe —
  [fused_sampler.cu:585-607](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:648-651](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).

### Inferences
- **Graph capture hazard (the important one):** the per-launch RNG advance happens in host
  code (`next_launch_offset()`) at launch time. During `cudaStreamBeginCapture` the offset
  is read once and baked into the kernel node, so every `cudaGraphLaunch` replays the
  *same* Philox offsets and the *same* seed. A graph-captured decode loop therefore draws
  the identical Gumbel vector for a given (row, rank) at every step — if the logits repeat,
  the sampled token repeats, and in general the noise stream is never advanced. hpc-ops
  documents graph requirements nowhere. For kernel-ops this must be redesigned: derive the
  RNG counter from a *device-resident* step counter (or from token position, i.e.
  counter-based noise = f(seed, row, token_block, step_read_from_device)) so replay
  advances naturally.
- **Cross-process / re-run reproducibility is impossible** by construction: the counter
  starts at 0 per process, so reproducibility holds only if the *sequence* of launches and
  their batch sizes is identical. Any error path that skips a launch shifts every later
  sample.
- **Multi-stream:** the fused path is safe but pays allocator contention; the temperature
  path can silently produce a *stale or missing* token rather than an error. Failure mode:
  if two streams' `atomicAdd`s interleave, no block sees `s_prev == N-1`, the final reduce
  never runs, and `token_ids_out[ibatch]` keeps its previous (uninitialised) value —
  a silent wrong token, not a crash ([fused_sampler_temperature.cu:271-280](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)). This is exactly the class of bug
  we must exclude in `csrc/sampling`.
- The external-Gumbel path is a genuinely good testability idea: it makes the sampler
  bit-exactly testable without touching the RNG, and it is the only way their test suite
  can assert equality at all.

### Gaps
- Philox draw-count bookkeeping: `kPerLaunchOffsetIncrement = 128` is asserted to cover
  "max iterations × 4 draws" for the temperature path (16×4 = 64 ≤ 128,
  [fused_sampler_temperature.cu:346-350](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)), but for the fused path only ≤4 draws per
  lane are needed, so the same 128 is conservative. I found no analysis of Philox
  *subsequence* collisions between rows for large B (sequences are `row*32+lane`, dense and
  unique, so no collision — but that is my inference, not a documented proof).

---

## Performance evidence: what is measured, what is claimed, what is missing

### Takeaway
**There are zero committed sampler measurements** (no CSV/JSONL/nsys-rep anywhere in the
tree). The only numbers are a README marketing claim of "up to 8.5×" vs "vLLM-style
PyTorch, FlashInfer" and a benchmark script whose baseline is a naive multi-kernel torch
pipeline (full 120k-wide sort, full softmax, per-call index-buffer rebuild). From first
principles the sampler is sort-bound at ~1 radix item per vocab element, uses 24 % of the
SMs at B=1 (and 12 % with `max_topk=64`), and runs 2 kernels + 8 allocator calls per call.

### Cited Findings
- README performance table: "Fused sampler | BF16 logits, vocab size 120832, batch size
  1-512 | vLLM-style PyTorch, FlashInfer | **Up to 8.5×**" with the caveat
  "Performance varies across cases" — [README.md:169-181](../../3rd/hpc-ops/README.md).
- README prose claims "optimizes small top-k cases with **local heap/merge-style
  selection**" — [README.md:145-150](../../3rd/hpc-ops/README.md). The code contains no heap or merge: it is
  `cub::BlockRadixSort` ([fused_sampler.cu:196](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:355](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)). Test comments also say
  "cluster topk" ([tests/test_sampler.py:30](../../3rd/hpc-ops/tests/test_sampler.py), [tests/test_sampler.py:64-66](../../3rd/hpc-ops/tests/test_sampler.py)) while no cluster
  launch (`__cluster_dims__`, `cudaLaunchAttributeClusterDimension`) exists in either
  sampler TU. **Docs and code have drifted.**
- Benchmark methodology: `--timing nsys` = eager calls inside NVTX ranges, median of
  "Projected Duration (ns)" from `nsys stats -r nvtx_gpu_proj_trace`; no per-iteration
  synchronisation in the timed loop — [benchmark_sampler.py:202-213](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py),
  [benchmark_sampler.py:308-334](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py). `--timing event` = CUDA events bracketing each
  eager call — [benchmark_sampler.py:216-228](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py).
- Defaults: `warmup=20`, `iters=100`, batches `1,8,16,32,64,128,256,512`, providers
  `hpc/torch/flashinfer`, scenes `temperature`/`full`, V=120832, BF16, T=1.05, rp=1.1,
  TOPK=20, TOPP=0.9, seed=1 — [benchmark_sampler.py:36-46](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py), [benchmark_sampler.py:440-456](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py),
  [benchmark/sampler/README.md:14-16](../../3rd/hpc-ops/benchmark/sampler/README.md).
- hpc "full" cell: `softmax_policy=AFTER_TOPK, topk=20, topp=0.9, max_topk=32`
  — [benchmark_sampler.py:185-197](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py).
- The torch baseline for "full" is **not** vLLM's fused sampler: it rebuilds two
  `[B, V+1]` bool buffers and scatters 65 536 indices per call, calls vLLM's
  `apply_repetition_penalties`, then does a full `work.sort(dim=-1)` over 120 832 fp32
  values, a full softmax, a full `cumsum`, two `masked_fill`s, another full softmax and a
  full exponential/div/argmax — [benchmark_sampler.py:134-170](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py).
- The torch "temperature" baseline does a full fp32 softmax + full-vocab exponential RNG +
  full-vocab divide + argmax — [benchmark_sampler.py:115-120](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py). FlashInfer's path calls
  `sampling.softmax` over the full vocab then `sampling_from_probs`
  — [benchmark_sampler.py:122-129](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py), [benchmark_sampler.py:172-183](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py).
- No result artefacts: `find` over `3rd/hpc-ops` for `*.csv`, `*.jsonl`, `*.nsys-rep`,
  `*latency*` returns nothing; the benchmark only writes files when `--csv/--jsonl` are
  passed and defaults its nsys output to a *gitignored* `log/` dir
  ([benchmark_sampler.py:469-486](../../3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py)).

### Inferences
**Geometry table** (CODE-FACT arithmetic from `pick_n_blocks_per_row`, H100 = 132 SM,
V=120832, `max_topk=32` ⇒ `Nmax=32`, `kNMin=8`):

| B | N | K1 blocks | load loops | sort items K1 | K2 blocks | waves (132 SM) |
|---|---|---|---|---|---|---|
| 1 | 32 | 32 | 1 | 131,072 | 1 | 1 (24 % of SMs) |
| 2 | 32 | 64 | 1 | 262,144 | 2 | 1 |
| 4 | 32 | 128 | 1 | 524,288 | 4 | 1 |
| 8 | 16 | 128 | 2 | 1,048,576 | 8 | 1 |
| 16 | 8 | 128 | 4 | 2,097,152 | 16 | 1 |
| 32 | 8 | 256 | 4 | 4,194,304 | 32 | 2 |
| 64 | 8 | 512 | 4 | 8,388,608 | 64 | 4 |
| 128 | 8 | 1,024 | 4 | 16,777,216 | 128 | 8 |
| 256 | 8 | 2,048 | 4 | 33,554,432 | 256 | 16 |
| 512 | 8 | 4,096 | 4 | 67,108,864 | 512 | 31 |

With `max_topk=64`, `Nmax=16`, so B=1 launches **16 blocks = 12 % of an H100** and
`num_load_loops` doubles; total sort items stay ~1.08×B×V. **Asking for a larger K makes
small-batch latency worse** — an inversion no sane sampler should have.

**Latency model (ESTIMATE).** Assumptions: CUB block radix sort ≈ 8 passes, ~16 B of
smem traffic per item-pass, ~1.5–2.5 G sort-items/s/SM on a 1.8 GHz Hopper SM (i.e. a
4096-item (key,value) sort ≈ 1.6–2.7 µs on one SM); HBM 3.35 TB/s; kernel launch gap on
the GPU timeline ~0.5–1 µs.

| case | bound by | ESTIMATE GPU time | ESTIMATE per-call wall |
|---|---|---|---|
| temperature, B=1 | 241 KB over 132 blocks + 1 launch | 1.2–2.5 µs | 2–4 µs |
| temperature, B=512 | 124 MB HBM + RNG (≤4 draws/elem) | 45–70 µs | 50–80 µs |
| full, B=1 (K1 32 blocks + K2 1 block) | 1 sort on 32 SMs, then 1 serial block | 3–5 µs | 6–10 µs (incl. 8 allocator calls) |
| full, B=8 | 128 blocks, 2 sorts each | 6–9 µs | 10–15 µs |
| full, B=512 | 67.1 M sort items | 200–350 µs | 210–360 µs |

Cross-check that the model is self-consistent with the "8.5×" claim: the torch baseline at
B=512 sorts 61.9 M fp32 keys (plus a second full softmax), which is a ~1.5–3 ms pipeline, so
~0.25 ms for hpc-ops lands at 6–12× — i.e. the headline 8.5× is plausible **and is mostly a
statement about how slow the baseline is**, not about how close hpc-ops is to the roofline
(134 MB / 40 µs). Two design choices, not the baseline, define its ceiling: the repeated
block radix sort (~1 item per element) and `Nmax` capped by the K2 block size.

- **Temperature path vs full path**: the temperature kernel is a completely different,
  much better shape — 256-thread blocks, `grid=(ceil(SM/B), B)` (so B=1 uses *all* SMs,
  [fused_sampler_temperature.cu:435-445](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)), no `cudaMallocAsync` (persistent workspace),
  **one** kernel launch, block-arrival counter instead of a second kernel, and RNG/noise
  fused into the single pass — [fused_sampler_temperature.cu:147-217](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu), [fused_sampler_temperature.cu:268-326](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu).
  At B=1 I estimate it is **2–4× faster than the full path**; at B=512 the gap narrows
  because both become throughput-bound, though the temperature path never pays sort cost.
  Its one self-inflicted wound is a **true IEEE division per element** (`tmp[i] / t`,
  [fused_sampler_temperature.cu:199](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)) where the fused kernel uses `rcp.approx.ftz`
  ([fused_sampler.cu:187,193](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)) — ~10–20 cycles per element, i.e. potentially a
  large fraction of the ALU budget of an otherwise memory-only kernel.

### Gaps
- No measured numbers at all: no per-batch latency, no breakdown between K1 and K2, no
  nsys/ncu artefacts, no comparison against vLLM's *actual* fused sampler
  (`Sampler`/`apply_top_k_top_p` Triton kernels), only against a naive torch pipeline.
- Whether the 8.5× is at B=1 or B=512 is not stated.
- My per-SM sort throughput (1.5–2.5 G items/s) is an assumption; a ±2× error moves the
  B=512 estimate between ~0.1 and ~0.6 ms. Only a real measurement on H100/B200 settles it.

---

## Limitations and attack surface

### Takeaway
Structurally: 2 launches + 4 alloc/free pairs per call, `Nmax ≤ 32` tied to K2's block size,
one block per SM by design, a single hard-coded vocab, penalty information limited to a
1-bit "seen" mask, no min-p, top-p only over ≤64 candidates, no exact full-vocab mode, an
RNG whose offset is host-side (graph-hostile), and a shared per-device workspace that makes
the fast path single-stream-only. Some are fixable; `Nmax`, the vocab specialisation and the
bitmask penalty are architectural.

### Cited Findings
1. **Per-call scratch**: 4 × `cudaMallocAsync` + 4 × `cudaFreeAsync` per call, return codes
   unchecked — [fused_sampler.cu:585-607](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), [fused_sampler.cu:648-651](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). Capture-legal but
   adds 8 graph nodes and freezes addresses ([CUDA graph memory nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)).
2. **`Nmax` cap**: `min(1024/K, 32)` — 32 (K=32) / **16** (K=64) —
   [fused_sampler.cu:44-46](../../3rd/hpc-ops/src/sampler/fused_sampler.cu), so B=1 fills 24 % / 12 % of an H100 and less of a B200
   (148 SM). `kNMin=8` also means B>16 runs multiple waves with only 8 blocks/row
   ([fused_sampler.cu:616-627](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)).
3. **Two kernels**: K1 → K2 via global memory; K2 then does a redundant 1024-candidate sort
   even when only `N*K` slots are valid (`cand_bid >= N` masked to -inf,
   [fused_sampler.cu:365-375](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)) and 31/32 warps exit
   ([fused_sampler.cu:399-401](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)).
4. **Vocabulary specialisation**: the dispatcher accepts exactly `vocab_size == 120832` and
   throws otherwise — [fused_sampler.cu:757-766](../../3rd/hpc-ops/src/sampler/fused_sampler.cu); the temperature path is the same —
   [fused_sampler_temperature.cu:538-547](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu). `128256` (Llama-3) or `152064` (Qwen) cannot run
   at all; `kVocabSizeMax = 120832` even sizes the temperature path's `kNMin`
   ([fused_sampler_temperature.cu:339-344](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)).
5. **Penalty = 1 bit per token**: only "seen / not seen", applied uniformly as ×1/rp or
   ×rp — [fused_sampler.cu:94-106](../../3rd/hpc-ops/src/sampler/fused_sampler.cu). No frequency/presence
   distinction (HF has both), no penalty window, no per-token counts, no DRY/no-repeat-ngram.
6. **No min-p / typical-p / XTC / eta**, and no "no-repeat-ngram" — grep-verified absence.
7. **Top-p only over ≤64 candidates** — [fused_sampler.cu:453-458](../../3rd/hpc-ops/src/sampler/fused_sampler.cu); and `BEFORE_TOPK`
   compares against unnormalised full-vocab mass ([fused_sampler.cu:378-385](../../3rd/hpc-ops/src/sampler/fused_sampler.cu) +
   [fused_sampler.cu:504-506](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)), so top-p silently becomes a no-op when
   the top-32 mass < topp.
8. **No exact full-vocab fallback**: `topk=0, topp=0` still truncates at 32/64 —
   [sampler.h:16-20](../../3rd/hpc-ops/src/sampler/sampler.h), [hpc/sampler.py:90-94](../../3rd/hpc-ops/hpc/sampler.py). `topk > max_topk` silently clamps
   instead of erroring ([fused_sampler.cu:322-324](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)).
9. **Multi-stream**: temperature path uses `static WorkspaceCache g_ws[16]` +
   per-device counter; concurrent calls on two streams can lose the last-block reduce and
   emit a stale token — [fused_sampler_temperature.cu:352-372](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu), [sampler.h:38-39](../../3rd/hpc-ops/src/sampler/sampler.h).
   The workspace also grows 2× under a mutex with a `cudaStreamSynchronize` +
   `cudaFree` on the growth path ([fused_sampler_temperature.cu:380-433](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)) — a latency spike
   and a replay hazard if it ever reallocates inside a capture.
10. **RNG offset is host-side** (graph-frozen) — [sampler_rng.cuh:42-52](../../3rd/hpc-ops/src/sampler/sampler_rng.cuh),
    [fused_sampler.cu:665-669](../../3rd/hpc-ops/src/sampler/fused_sampler.cu).
11. **Vector width**: bf16 loads are 8 B per thread (`vec_t<__nv_bfloat16,4>` → `uint2` in
    `load`, [src/utils/utils.cuh:171-189](../../3rd/hpc-ops/src/utils/utils.cuh)), not the 16 B the temperature
    kernel's comment claims ([fused_sampler_temperature.cu:109-111](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)). fp32 gets full 16 B.
12. **No error checking** on the two kernel launches in `launch_pipeline`
    ([fused_sampler.cu:653-675](../../3rd/hpc-ops/src/sampler/fused_sampler.cu)) — the temperature path does check
    ([fused_sampler_temperature.cu:465-469](../../3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu)). A launch failure surfaces as a stale token.
13. **`get_sm_count()` always reads device 0** ([src/utils/utils.cc:19-31](../../3rd/hpc-ops/src/utils/utils.cc)) while
    `cudaGetDevice` is used for the workspace — on a heterogeneous multi-GPU node `N` is
    computed from the wrong device.

### Inferences
- The `Nmax ≤ 32` cap is a *derived* constant, not a requirement: it exists because K2's
  block is `Nmax*K ≤ 1024` and its partial reduce uses one lane per block. Decoupling those
  (segment the merge over multiple blocks, or use a warp-level multiway merge with a
  device-side arrival counter like the temperature kernel) would let B=1 use all SMs.
- **Reuse from `src/topk/` is not possible for sampling**: `topk_filtered` is an exact
  FP32-only, persistent, `topk ∈ {512, 2048}` row top-k for KV/index selection with
  `RADIX=2048`, `BLOCKS_PER_SM=3` and a caller-supplied counters+workspace ABI —
  [src/topk/topk_filtered.cu:17-45](../../3rd/hpc-ops/src/topk/topk_filtered.cu), [src/topk/entry.cc:43](../../3rd/hpc-ops/src/topk/entry.cc) — and
  the sampler never calls it (grep-verified). Its *technique*, however, is the right one
  for us: coarse 11-bit ordered-key histogram → derive an exact boundary → single
  classification pass, i.e. two cheap passes instead of a per-element sort.
- **The best ideas worth copying** (ranked): (a) exactly-one full-vocab read with
  penalty+temperature fused into the load; (b) candidate-indexed Gumbel noise instead of
  vocab-indexed (≤64 RNG draws/row, not 128k); (c) penalty-mask writeback on device with a
  single 32-bit `atomicOr` per row; (d) *policy* template axes so top-p/softmax variants
  compile to dead code; (e) an external-Gumbel injection path for bit-exact tests;
  (f) a temperature-only specialisation that drops a whole kernel (their fastest path);
  (g) explicit deterministic tie-breaking; (h) row-stride support for padded logits
  ([entry.cc:29-37](../../3rd/hpc-ops/src/sampler/entry.cc)).
- **What to avoid**: rolling block radix sort per load-loop (≈1 radix item/element, 8
  passes); a merge block sized so it caps `N`; block-wide sorting when only one warp
  consumes the result; per-call `cudaMallocAsync`; a host-atomic RNG offset; hard-coded
  vocab; `x/t` instead of `x*rcp(t)`; silent `min(topk, 32)` clamping instead of an error.
- **A concrete alternative for `csrc/sampling`** that preserves exact full-vocab semantics:
  Gumbel-max is a single-pass max over `logit/T + G(seed,row,token)`, so if the noise is
  counter-based and token-indexed (Philox counter = f(row, token_block), step read from a
  device scalar so graph replay advances) you get *exact* full-vocab sampling with **one
  read, one pass, one launch, no sort, no candidate buffer, no host state**. Cost per
  element is ~1 log + 1 log + add + max (or a cheaper exponential sampler). For
  top-k/top-p you then need the threshold, which is where a 2-pass or histogram approach
  comes in — and note that at B ≤ ~64 the bf16 logits (241 KB/row × B ≤ 15 MB) stay
  **inside the 50 MB L2 of an H100**, so a second pass is nearly free at exactly the batch
  sizes where the first pass cannot saturate HBM. That is the structural advantage hpc-ops
  gives away by sorting instead of re-reading.

### Gaps
- I did not verify H100/B200 L2 capacities or per-SM smem bandwidth against NVIDIA docs in
  this pass; the L2-residency argument uses the widely known ~50 MB figure and should be
  re-checked before it goes into a design doc.
- hpc-ops' `BEFORE_TOPK` intent ("joint topk-topp") suggests they believe it matches HF's
  order; I found no upstream issue/PR discussing the unnormalised-mass consequence, so the
  question "is this a bug or a deliberate choice" is open.
