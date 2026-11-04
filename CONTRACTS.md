# CONTRACTS — B0/B1 numerical and performance rules

This file is the single place for tolerances, rounding rules and the benchmark
protocol used by the B0/B1 gates. It is intentionally short; the kernel headers
remain the ABI and shape authority.

## 1. Numerical rules

- All reductions accumulate in fp32. The output is rounded once, at the store,
  to the output dtype. No kernel stores an intermediate fp32 copy unless its
  public contract says so.
- Normalization uses biased variance (`sum(x^2)/N` for RMS, `sum((x-mean)^2)/N`
  for LayerNorm). `epsilon` is added inside the square root.
- `weight_offset` is added to the weight before the affine multiply:
  `(w + weight_offset)`. Gemma-style zero-centred weights use `1.0`.
- Fused residual+norm writes the residual in its declared dtype first, then
  normalizes the written residual. This matches the vLLM rounding rule and is
  tested bit-for-bit where the dtype allows.
- RoPE uses the cache layout `[position, 0:rot/2] = cos`,
  `[position, rot/2:rot] = sin`; the cache angle is formed in fp64 and rounded
  once to the cache dtype.
- LSE is natural-log fp32. A fully masked row has `lse = -inf` and `o = 0`.
  A sink is added to the denominator exactly once.
- Argmax ties resolve to the lowest token index. Scatter/gather require unique
  destination rows; duplicate destinations are a caller error.
- Index operations (gather, scatter, bincount, bitmask, argmax, top-k indices)
  are compared exactly.
- Floating-point comparison against the fp64 oracle:
  - fp32 output: `atol = 1e-5`, `rtol = 1e-5`;
  - fp16/bf16 output: `atol = 2e-2`, `rtol = 2e-2` for elementwise ops, and
    `atol = 1e-4`, `rtol = 1e-4` for fp32 intermediates in merge operations.
  These are the defaults; a test may tighten them when it proves the tighter
  bound.

## 1b. Quantization rules (B3)

- Scales are dequant multipliers: `x ~= q * s`; INT4 per-token-head uses
  `x ~= (q - z) * s` with an integer zero point stored beside the scale.
- FP8 is OCP e4m3 with `satfinite`; INT8 is symmetric `[-127, 127]` or
  asymmetric `[-128, 127]`; INT4 is packed two per byte, element `2i` in the low
  nibble.
- Dynamic per-token FP8 scale is `max(amax, 1e-10) / 448`, clamped by
  `scale_upper_bound` when given. Dynamic per-token INT8 asymmetric uses
  `s = (max - min) / 255`, `z = round(-128 - min / s)`.
- Per-group scales are 1×group (32/64/128) row-major or column-major TMA
  (`[groups, round_up(rows, 4)]`). UE8M0 rounds the scale up to a power of two.
- MX uses block 32 with a UE8M0 byte; NVFP4 uses block 16 with an E4M3 byte and
  one fp32 global scale. Block-scale layouts follow the 128×4 swizzle
  `[row/128, ceil(groups/4), 32, 4, 4]`.
- The fused `norm_quant` / `act_and_mul_quant` keep the normalized/activated
  value in fp32 and quantize it directly; `norm_out` is the same value rounded
  once to the activation dtype. A test may compare against the unfused chain
  with a one-quantization-step tolerance, not bitwise.
- `kMLAFp8DS` records are 656 B: four 128-value e4m3 tiles, four fp32 tile
  scales `2^ceil(log2(max(amax/448, 1e-4)))`, then 64 bf16 RoPE values.
- The indexer split page is `[page_size × head_dim e4m3 | page_size ×
  (head_dim/quant_block) fp32 scales]`; scales follow all keys of the page.

## 1c. Routing, permutation and sampling rules (B6)

- Router selection uses the score (`softmax`, `sigmoid`, `sqrt(softplus)`, raw logits) plus
  `correction_bias`; the bias never enters the weight. Ties resolve to the lower expert index.
  `softmax-after-topk` normalizes the selected raw logits, `softmax` normalizes over all experts.
  `renormalize` divides the selected weights by their sum, then `routed_scaling_factor` is applied.
- Group-limited routing uses group score `max` or `sum(top-2)` of the biased scores, keeps
  `topk_groups` groups (ties -> lower group id), and selects the top-k inside the kept groups.
  There must be at least `topk` experts in the kept groups.
- EPLB: a selected logical expert maps to replica `token_index % replica_count[e]`; the output id
  is the physical id. `expert_load` counts logical experts. Fused shared experts are appended in
  order with weight `shared_weight` and id `num_experts + s`.
- Hash routing uses the checkpoint table when present (multiplicative fallback otherwise), uniform
  weights `routed_scaling_factor / topk`, and maps ids through `expert_map` when supplied. The
  DeepSeek-V4 gate-scored hash layer is not represented by this kernel.
- `align_block_size` is stable by slot id inside each expert, pads each expert to `block_size` with
  the sentinel `numel`, and emits local expert ids (`expert_map < 0` produces no blocks). It needs
  `(num_experts + 1) * 4` bytes of workspace. `permute` reuses the same stable order and requires a
  caller-zeroed `write_cursor`; the copy phase is order-free.
- `unpermute_reduce` sums in fixed k order (no atomics): `routed_scale * sum(w * expert) + shared +
  residual`; rows of non-local experts contribute zero.
- Sampling truncation order is min-p, then top-k, then top-p (the default); with
  `joint_topk_topp` the top-k and top-p masks both apply to the original distribution before
  renormalization. Tokens at a threshold tie are taken in ascending token id. `top_p >= 1`
  disables top-p, `top_p <= 0` keeps only the largest token, `top_k <= 0` or `>= vocab` disables
  top-k. Randomness is a pure function of `(seed, request_id, position, draw_index, stream_id)`;
  the graph replay never advances host state. A row with no admissible mass writes `out_ids = -1`.
- `sample()` materializes the fp32 probability row in `Workspace::device_scratch`
  (`batch * vocab * 4` bytes plus the temperature and split-softmax regions reported by
  `sampling_workspace_requirements`). `gumbel_max_sample()` deliberately rejects top-p
  (`kNotImplemented`) because it works from logits; use `sample()` for nucleus truncation.
- `sample_fused()` samples straight from logits in one launch and never materializes the fp32
  probability row. Its semantics differ from `sample()` in two declared ways:
  - **top-k keeps the whole tie group** at the k-th largest value (vLLM `apply_top_k` convention:
    every token with `x >= x_(k)` survives), so `top_k == 1` with `T > 0` samples among the tied
    maxima and is not greedy. The comparison uses the processed logits before temperature (the
    order is the same; only values that `x / T` would round together can differ). min-p keeps
    `y >= max(y) + log(min_p)` with `y = x * (1 / T)` (`min_p > 1` keeps nothing: `out_ids = -1`,
    as in `sample()`). top-p orders its domain (the min-p/top-k base, or every finite token with
    `joint_topk_topp`, intersected with the base afterwards) by value descending, then token id
    ascending, and keeps a token while the fixed-point (2^40) mass before it is below
    `max(1, round(p * Z))`, so ties at the boundary keep the lower ids and `top_p <= 0` (or a NaN
    per-row value) keeps only the first token; a NaN scalar `default_top_p` is rejected. `T <= 0` (or non-finite per-row `T`) is greedy:
    lowest id among the maxima. A row with top-p active and no top-k/min-p uses a 2048-bin
    fixed-point mass histogram to locate the boundary bin exactly; the histogram changes only how
    the boundary is found, not the nucleus or the draw, and a boundary bin over the candidate
    buffer still takes the exact multi-pass fallback.
  - **draws are equal in distribution, not token-for-token**: the unfiltered path uses a 64-bit
    uniform for the inverse CDF and the filtered paths a Gumbel race keyed by token id (64-bit
    uniform per kept token), so the same
    seed does not reproduce `sample()`'s token. `request_id` is the batch row, as in `sample()`:
    a row reproduces its token across batch sizes, other rows' contents, split counts and graph
    replays only when it keeps its row index.
  Processors (repetition via `penalty_mask`, frequency/presence via `token_counts`) are applied in
  the load, before temperature; with `update_penalty_mask` the sampled token's bit is set after the
  draw, so the rows of one call must use distinct slots (another row would read the mask while it
  changes). `SamplingPlan` is resolved by `sampling_plan()` before capture and is invalidated by a new
  `capture_generation`; a workspace serves one in-flight call (concurrent streams need their own).
  Filters that a plan cannot run yet return `kNotImplemented` from `sampling_plan()`. The plan and
  `FusedSamplingParams` are C++ only (no C mirror).
- **Split rows (`SamplingPlan::splits > 1`)**: `sampling_plan()` splits a row over up to 16 CTAs
  (a power of two) when the batch leaves the GPU idle, the rows are 16-byte vectorizable and every
  CTA keeps >= 4096 tokens. The grid (splits, batch) is launched cooperatively, so all CTAs are
  resident or the launch fails; the CTAs of a row combine integer sums, maxima and histograms in
  CTA order, so the tokens are bit-identical to `splits == 1` for every filter. Callers may lower
  `splits` (1 forces one CTA per row) but not raise it past what fits; non-vectorizable rows with
  `splits > 1` are `kPlanInvalidated`. The workspace holds a per-row exchange area
  (`device_scratch`) and one 8-byte barrier counter per row (`device_persistent`, zero-filled once
  before the first call, then valid for any split count). A workspace serves one call at a time;
  concurrent streams need their own. The whole call is one kernel node and is graph-capturable.

## 1d. Tensor-core GEMM and epilogue fusions (B7)

- `gemm_mma_sync_kernel` (bf16 / fp16, `mma.sync.m16n8k16`, fp32 accumulate) rounds once, at the end of
  the epilogue. For one output element the K chain is the same sequence of k16 steps in every tile
  shape and with or without `swap_ab`, so tactics with the same `split_k_slices` agree bitwise
  (`test_mma_invariance` checks it). Split-K changes the chain: the slices are summed in slice order
  `0..S-1` starting from 0.0f, both for `kSerial` (the last CTA of a tile sums the fp32 partials in the
  workspace) and `kParallel` (a reduce kernel), so the two give identical bits and repeated runs match.
- Nothing depends on arrival order: `kEpiAtomicAdd` is the only atomic and is refused when
  `ctx.deterministic` is set. For a 16-bit D the atomic adds the addend already rounded to the output
  dtype (`atomicAdd` on `__nv_bfloat16` / `__half`), so it rounds twice; fp32 D rounds once. `ctx.batch_invariant` disables the heuristic's split-K and `swap_ab`.
- Operands: K-major A and B, 16 B aligned pointers and leading dimensions, `K % 8 == 0`
  (`kMisaligned` otherwise). MN-major operands and `kTF32` return `kNotImplemented` here.
- Epilogue order (fp32): `alpha` -> row_scale -> col_scale -> bias -> clamp -> softcap -> act or GLU ->
  routing_weight -> sigmoid gate multiply -> `+ beta * residual` -> round to D -> store -> row_sumsq ->
  KV append. Residual, gate_src and the KV destination are read / written in the output dtype.
- **Epilogue fusions.** `kEpiRowSumSq` and `kEpiKvAppend` (ABI 6) do not change D; they hand data to
  a later op and run after D is written.
  - `kEpiRowSumSq`: `row_sumsq[m] = sum_k A[m, k]^2`, fp32, the squares of the rounded input values
    (DeepGEMM `tf32_hc_prenorm_gemm` `sqr_sum`). It is written, not accumulated, so the buffer needs no
    zeroing. The CTAs of the first N tile sum a row with one warp (8 elements per lane, xor butterfly),
    so the value depends on K only and never on the tile, split-K or `swap_ab`.
  - `kEpiKvAppend`: for every row `m` with `kv_slots[m] >= 0` and every output column `n` in
    `[kv_col_begin, kv_col_begin + kv_cols)`, `kv_dst[kv_slots[m] * kv_stride + (n - kv_col_begin)]`
    receives exactly the value stored to `D[m, n]` (after rounding, same dtype; `n` counts output columns,
    `N / 2` after GLU). A row dropped by `kEpiScatterRows` is still appended (its `kv_slots` entry
    decides). It cannot be combined with `kEpiAtomicAdd` (no single stored value) and needs `batch == 1`.
- `kEpiGLU` on the tile GEMM needs `GateLayout::kInterleaved` with `glu_interleave` 1..64 and
  `N % (2 * glu_interleave) == 0`; D has `N / 2` columns. The epilogue stages the fp32 accumulators in
  shared memory one pass at a time and pairs (gate block, up block) inside a pass, so the tile must have
  a pass of at least `2 * glu_interleave` columns (128 for the 128-wide tiles, 64 for the x64 tiles and the
  64-token `swap_ab` tile, 32 for the 32-token `swap_ab` tile). Both gate and up columns see
  bias / scales / clamp / softcap before the GLU. Split halves (`kGateUp` / `kUpGate`) belong to
  `dual_gemm_kernel` or `glu_weight_interleave`.
- `dual_gemm_kernel` (bf16 / fp16) computes `gate = A . B0^T`, `up = A . B1^T` in one pass over A and returns
  `GLU(gate, up)` with `DualGemmParams::act` / `glu` exactly as documented on `GluParams` (D has N columns,
  same dtype as the operands). Order (fp32): `* alpha` (or `*alpha_ptr`) -> `* row_scale[m]` -> clamp ->
  softcap (gate and up alike) -> GLU -> `* routing_weight[m]` -> `+ beta * residual` -> round. Bias
  (`bias0` / `bias1`), column scales, scatter, atomics, quantized output and FP8 operands are
  `kNotImplemented`; scale pointers with 16-bit operands are `kInvalidArgument`. A row alone and the same
  row in a larger M give identical bits.
- `bmm_kernel` / `bmm()` (bf16 / fp16, output the operand dtype or fp32): `D[b] = A[b] . B[b]`, B read as
  `[N, K]` (`b_k_major`) or `[K, N]`. `lda`, `ldd` and the batch strides may describe head-strided views of
  `[T, H, *]` (`lda = H * D`, `stride_a_batch = D`); batches beyond 65535 are split over launches. Needs 16 B
  aligned pointers, `lda`, `ldb` and the A / B batch strides, `K % 8 == 0`, and `N % 8 == 0` when B is `[K, N]`.
  The K chain per element is the one of `gemm_mma_sync_kernel`, so `bmm` (k-major B) and the batched tile
  GEMM agree bitwise. FP8 operands return `kNotImplemented` (use `fp8_bmm`).
- Not yet on this path: `kEpiQuantOut` and `kEpiAzpCorrection` (`kNotImplemented`), stream-K, MN-major
  operands.
- Tactic cache key: (recipe, N, K, M bucket = next power of two, `arch_id`, `CUDART_VERSION`). An
  entry is a function of that key alone, so the same conditions give the same tactic; the text file
  written by `tactic_cache_save` carries the ABI version and is refused on any other. `autotune`
  skips M <= 16 (the CUDA-core GEMV owns decode) and returns `kInvalidArgument` while the stream is
  being captured.

## 1e. Native attention on SM80-class GPUs (B8)

- **Score order** (prefill and decode, fp32): `s = sm_scale * k_scale * q.k` (`sm_scale = 0` means
  `1 / sqrt(Dqk)`; `k_scale` only for FP8 K/V) -> softcap `cap * tanh(s / cap)` -> ALiBi
  `+ slope[h] * (k_pos - q_pos)` -> mask -> softmax. The softmax runs in base 2 on `s * log2 e`.
  A sink adds `exp(sink[h] - m)` to the denominator once: in the final normalization of an
  unsplit row, or in `merge_split_kv` for split rows (partials never see it). `lse` includes the
  sink. FP8 V is rescaled by `v_scale` after normalization. P is rounded to the compute dtype for
  the PV product, the row sum uses the fp32 P (FA2 convention). Oracle: `ref::attention_f64`.
- **Mask** (bottom-right causal, `attention_params.cuh`): query row i of a request with `qo_len`
  rows and `kv_len` keys sits at `q_pos = q_pos_offset[b]` (else `kv_len - qo_len`) `+ i`. Key j
  is visible when every enabled rule holds: causal `j <= q_pos`; custom bit
  `mask_indptr[b] + i * kv_len + j`; window `j >= q_pos - window_left` unless `j < num_sink_tokens`;
  `j <= q_pos + window_right`; chunk `floor(q_pos / c) == floor(j / c)`. A row with nothing visible
  and no sink gives `o = 0`, `lse = -inf`. Decode is causal among its `q_len` rows; custom masks
  and `q_pos_offset` go through `paged_prefill_run`.
- **Mask mode and modifiers are runtime values** (ABI 6): `RaggedPrefillParams::mask` /
  `PagedPrefillParams::mask` (set by `*_run` from the plan) replace the `kMask` template parameter,
  and the B8 kernels are instantiated with every `AttentionVariant` flag on; the runtime value
  switches a modifier on, and `*_run` neutralizes the ones the `AttentionSpec` did not enable. The
  per-flag instantiation matrix does not fit the 90 s TU budget (ptxas ~2 s per FA2 kernel).
- **Compiled cells.** Prefill (`prefill.cu`): (Dqk, Dvo) in {64, 80, 96, 128, 256, 192/128} x
  {bf16, fp16} x {ragged, paged}; FP8 e4m3 / e5m2 paged K/V for 64, 128, 256. Decode (`decode.cu`):
  CUDA-core D in {64, 80, 96, 128, 192, 256}, tensor-core and dynamic D in {64, 128, 256}, each
  with FP8 K/V. Pages: NHD or HND (strides), CSR or block table, any page size. q/o are the same
  16-bit dtype. Other cells return `kUnsupportedShape` / `kUnsupportedDtype` from
  `attention_supports` / `decode_plan`.
- **Plans and workspace.** `prefill_plan` / `decode_plan` fill `plan->workspace` and return
  `kWorkspaceTooSmall` when the caller's workspace is smaller (call once with an empty one); the
  plan uploads its tables into `device_persistent` with one copy on `ctx.stream`; `*_run` fills
  `params.plan` and `params.split` and refuses a workspace other than the planned one
  (`kPlanInvalidated`). `ctx.strategy == kGraph`: the host lengths are capacities and
  `prefill_plan_kernel` re-plans every replay from the device lengths (same item order as the host
  planner); overflow sets the flag and invalidates every item.
- **Split-KV** (`plan.split_kv`): partial rows `merge_indptr[row] + chunk` hold normalized fp32
  `o` and `lse`; `merge_split_kv` combines them in chunk order. Prefill splits when the q tiles fill
  less than one wave (towards two CTAs per SM), decode when the grid is below 4 (CUDA core) or 2
  (tensor core) CTAs per SM. `merge_split_kv` accepts head dims 64, 80, 96, 128, 192, 256.
- **Determinism / batch invariance.** Every sum has a fixed order (per thread in key order, then
  quad lanes, token groups or warps, then splits in index order); no atomics touch values, so
  repeated runs are bit-identical. With `ctx.batch_invariant` the kv chunk is fixed (prefill
  4096, decode 2048 keys) and every row goes through `merge_split_kv`, so a request's bits do not
  depend on its neighbours or on the plan's capacity. The dynamic decode is bit-reproducible (a
  task's math does not depend on its CTA) but not batch-invariant (task length follows the batch).
- **Decode paths.** `decode_plan` picks the tensor-core kernel when the packed rows of a kv head
  (`G * q_len`) are at least 4 and D is 64/128/256, else the CUDA-core kernel;
  `use_device_planner = true` selects `assign_decode_tasks_kernel` + `paged_decode_dynamic_kernel`
  (tasks of about `total / (SMs x resident CTAs)` keys in units of 256, at most 32 splits per
  request; every task writes partials, `merge_split_kv` finishes). The task counter is
  `uint32[2]` and resets itself at the end of each launch.

## 2. Benchmark protocol

- Same GPU, driver, clocks, dtype and shape for kernel-ops and PyTorch.
- PyTorch baseline uses `torch.inference_mode()`, no autograd, no
  `torch.compile`, no Triton. The HF-style baseline uses the same sequence of
  eager ops a model implementation uses; the built-in baseline uses the
  strongest fused primitive (`F.rms_norm`, `F.silu`, `F.softmax`, ...).
- Warm up both sides. Time a group of launches between two CUDA events; never
  record an event around a single short kernel.
- Interleave kernel-ops and PyTorch blocks in the same session. Report median,
  p10, p90 and min over at least three sessions.
- Record GPU clock, memory clock, temperature and commit in the JSON. A dirty
  build (`git.dirty = true`) is diagnostic-only and cannot promote a registry
  row to `benchmarked` or `release_supported`.
- Speedup is `torch_median / kops_median`. The primary KPI is the geometric
  mean over the inference-chain suite. Isolated single-op cases are reported
  separately; a bandwidth-bound one-pass kernel is expected to be near parity,
  not 2x.
- Every chain result must name the exact PyTorch sequence it replaces.

## 3. File and ABI rules

- B0/B1 kernels live in the family `.cu` named in ROADMAP Appendix E.2.
- No `_partN`, `_vN`, per-kernel or per-dtype files.
- `ops.h` is the only public host API. Version 4 carries the frequency table
  and hybrid branch pointers by value in the kernel parameter; host signatures
  remain unchanged. Version 6 adds `kEpiRowSumSq` / `kEpiKvAppend` with their `EpilogueParams`
  fields and the runtime tactic fields of `ScaledGemmArgs`, and (B8) the runtime `mask` field of
  `RaggedPrefillParams` / `PagedPrefillParams` (the `kMask` template parameter of the two prefill
  kernels is gone) plus the kv-length / capacity arguments of `prefill_plan_kernel` and
  `assign_decode_tasks_kernel` (§1e). `KOPS_C_ABI_VERSION` and `KOPS_ABI_VERSION` must move
  together.
- Split-row softmax uses `Workspace.device_scratch` as `[B, CTAS_PER_ROW, 2]`
  fp32. `sampling_workspace_requirements` reserves the policy maximum for the
  vocabulary; a smaller workspace returns `kWorkspaceTooSmall`.
- Tests and benchmarks are C++ (`.cc`/`.cu`); PyTorch lives only under
  `benchmark/compare/` and is never linked into the library.
