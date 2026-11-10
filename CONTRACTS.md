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
- `kMLAFp8DSV40` is 584 B/token: 448 e4m3 NoPE in seven 64-value tiles, then 64
  bf16 RoPE values (576 B of data), plus an 8 B/token scale region outside the
  record (seven UE8M0 bytes + one pad). The page data region and scale region
  are each padded by the FlashMLA 576-byte rule; `k_scale_pages` describes the
  second region.
- `kMLAFp8DSV41` is 528 B/token: 512 e4m3 values covering NoPE and RoPE in 16
  tiles of 32, followed by 16 inline UE8M0 scale bytes.
- `kMLAFp8DSV41Fp4` is 288 B/token: 256 B e2m1 (512 values, even element in the
  low nibble) followed by 32 inline e4m3 scale bytes with block 16. It is valid
  only as the extra cache next to a V4.1 FP8 cache.
- The indexer split page is `[page_size × head_dim e4m3 | page_size ×
  (head_dim/quant_block) fp32 scales]`; scales follow all keys of the page.

## 1b2. Sparse MLA / DSA conventions (B14)

- Sparse MLA uses two KV ranges, `primary` and `extra`, each with its own
  `indices` and optional per-query `topk_length`. `-1` entries are invalid;
  entries at or beyond `topk_length[t]` are invalid. Both ranges contribute to
  one online softmax; a sink adds one denominator term and is included in LSE.
- Top-k indices are exact, unordered, and ties resolve to the lower logical
  index. `to_page_slots` converts through the block table; invalid or
  out-of-range entries become `-1`.
- V4.1 fused variants use the same attention body: Q-norm + Q-RoPE before the
  score, O-RoPE conjugate + FP8 1×32 UE8M0 cast after. The unfused chain must
  agree within the FP8 output tolerance.
- A query row with no valid key writes `o = 0`; LSE is `-inf` without a sink and
  the sink logit with a sink.

## 1b3. DeepSeek-V4 mHC and Kimi-K3 AttnRes (B15)

- mHC follows `deepgemm:tests/test_mega_mhc.py::mhc_reference`: `new_residual =
  bf16(x*post_mix + comb^T*residual)`, projection `flatten(new_residual) @ fn^T`,
  `rsqrt(mean(x^2)+hc_norm_eps)`, `mix_scales`/`mix_bases`, `pre = sigmoid+eps`,
  `post = sigmoid*scale`, `comb = Sinkhorn(softmax+eps)`, then `sum(pre*X)` and
  RMSNorm. The projection is TF32 on the reference kernel and fp32 in the fp64
  oracle; tests use the TF32 tolerance documented by the row, never bitwise.
- AttnRes follows the Kimi-K3 block residual contract: sources are completed
  block representations plus the running partial, logits use a learned query and
  cached inverse RMS, softmax over depth, weighted sum, optional output RMSNorm
  and MXFP8 quant. Batch composition must not change a row's bits.

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
  (`kMisaligned` otherwise). `kTF32` runs `m16n8k8` on fp32 operands rounded with `cvt.rna.tf32`
  and needs `ctx.allow_tf32`; MN-major operands still return `kNotImplemented`.
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
- `kEpiQuantOut` writes OCP e4m3 or int8 with static per-tensor or dynamic per-group scaling; the
  per-token amax spans CTAs and is `kNotImplemented`. `kEpiAzpCorrection` is the asymmetric-INT8
  epilogue (`acc -= azp[m] * colsum_b[n]`). Not yet on this path: stream-K and MN-major operands.
- **B7 variants on the shared body.** `fp8_gemm` / `fp8_bmm` / `fp8_block_gemm_small_m` run OCP e4m3
  `mma.sync.m16n8k32` on SM89+ with fp32 accumulators (the hardware keeps an effective 11-bit
  mantissa, so the fp64 oracle allows `2^-11 * sum|p|`); scales are tensorwise, rowwise or
  `1x128` x `128x128` blockwise (fp32 or UE8M0), and the blockwise small-M path is `swap_ab`.
  `int8_gemm` / `w4a8_int_gemm` use s8 `m16n8k32` with s32 accumulators and the azp epilogue.
  `grouped_gemm` maps tiles to (group, m, n) for `kContiguous`, `kContiguousRowIndex`, `kMasked`
  and `kIndexed` (empty experts emit no tile; the task map matches the in-kernel scan bitwise);
  `moe_gemm` reads the block-sorted slots of `align_block_size` (block 16–256, expert `-1` skips
  the block, padding slots are untouched). `lora_sgmv` is SGMV shrink/expand over
  `seg_indptr` / `seg_lora_ids`, with shrink in fp32 and expand accumulating in the output dtype.
  `marlin_gemm` covers U4/U4B8/U8B128/FE4M3/FE2M1 dense and the MoE `sorted_token_ids` mode
  (act-order is dense-only); `gguf_mmq` is the int8-mma prefill path (IQ* stay `kNotImplemented`).
  Operand scale and bias values are read exactly as passed; destination rounding happens once.
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

## 1f. Recurrent state kernels (B9)

- **Host API.** The B9 entries of `state/ops.h` take the kernel parameter structs (device
  pointers, element strides) and validate every host-visible invariant before any launch (dims,
  dtypes, strides >= 0, slot stride, snapshot rules, in-place aliasing); device indices and
  lengths are never read on the host. No allocation, no synchronization: graph-capturable.
- **Slots.** A sequence's slot is `cache_indices / state_indices / state_batch_indices[b]` (null:
  slot b). A slot `< 0`, `>= num_slots` or `== pad_slot_id` skips the sequence: its outputs and
  state stay untouched, except `ssd_chunked_scan`, which computes the output from a zero state and
  writes no state. `has_initial_state[b] = 0` starts from zeros (conv: zero history); null means
  "from the slot" for SSM / linear / delta and "zero history" for `causal_conv1d_fwd`.
- **Snapshots** (prefix caching): the state after `(j + 1) * snapshot_interval` tokens goes to
  `snapshot_slots[b * max_snapshots + j]`; destinations outside `[0, num_slots)` are dropped.
  The interval is any positive value for conv1d and the recurrent delta rule, a multiple of 64
  for lightning and the chunked delta rule, a multiple of `chunk_size` for SSD.
  **Intermediate states** (speculative decoding): `intermediate_state_slots[b, i]` receives the
  state after token i (conv update, SSM update, recurrent delta rule); B10's
  `state_commit_accepted` picks one.
- **Numerics.** States are fp32; every product is an fp32 FMA in a fixed order that does not
  depend on the batch, so a sequence's result is independent of the others. Gates use the fast
  exponential (`dev::sigmoid_f`, `softplus_f`: torch threshold 20). Every decay factor is the exp
  of a difference of cumulative sums with the later position first (<= 0): nothing overflows,
  KDA pairs are formed per channel. Conv histories are raw storage bits. Lightning, selective scan
  (128-token tiles), SSD and the chunked delta rule reproduce the one-call result bitwise when a
  call is cut at a block / tile / chunk boundary and resumed from the saved state.
- **Delta rule.** `S = exp(g) S; S += k (beta (v - S^T k))^T; o = S^T (scale q)`, q and k
  L2-normalized as `x / sqrt(sum x^2 + 1e-6)` when `use_qk_l2norm`. Gating:
  `g = -exp(A_log) softplus(a + dt_bias)` (Kimi Linear, Qwen3-Next) or
  `lower_bound * sigmoid(exp(A_log) (a + dt_bias))` when `lower_bound < 0` (FLA bounded variant),
  `beta = sigmoid(b)`. `fused_delta_decode` = conv update (SiLU) -> gating -> L2 -> rule ->
  `RMSNorm(o) * w * act(out_gate)` with the conv output and the rule output rounded to the
  input dtype at the same boundaries as the unfused chain (Kimi Linear: sigmoid gate, eps 1e-5).
  Projections (q/k/v, f_a/f_b, b, g_a/g_b, o_proj) stay outside.
- **Chunked delta rule.** 64-token chunks from each sequence start; `h` row of a chunk =
  `(first token >> 6) + b`. `ChunkMath::kFp32` (default) or `kTensorCore` (bf16 mma.sync for
  kkt, w/u, state, output; cumsum and `solve_tril` stay fp32; bf16 inputs only, SM80+); the
  tensor-core result differs from the fp32 one by the bf16 rounding of the operands (~4e-3 of
  the oracle). `delta_rule_prefill` always runs the chunked pipeline; the recurrent kernel is
  `delta_rule_decode`.
- **Compiled cells.** conv1d width 2..4, x fp32/fp16/bf16 with weights in x's dtype or fp32,
  SiLU / Identity. SSM update dstate 8..256 (powers of two); selective scan dstate 8/16/32
  (`dim / ngroups` a multiple of 16, 8 for dstate 32); SSD chunk 64/128/256, P and N 1..256.
  Linear attention Dk, Dv 1..256. Delta rule Dk 16/32/64/128, Dv 1..256; fused decode Dk 64/128,
  Dv 16..128, Hv / Hk <= 8, (Hv / Hk) Dv <= 1024. Oracles: `ref::causal_conv1d_f64`,
  `ssm_recurrence_f64`, `linear_attention_f64`, `delta_rule_f64`, `delta_rule_gating_f64`.

## 1g. Speculative decoding (B10)

- **Chain rows.** `cu_num_draft[b]..cu_num_draft[b+1]` addresses the ragged draft
  rows. `target_probs[j]` / `draft_probs[j]` predict draft `j`; the bonus row of
  request `b` is `num_draft_tokens + b` (`num_draft_tokens = cu_num_draft[B]`).
  `target_argmax` uses the same layout. Rows outside a request's `cu_num_draft`
  range are never read.
- **Chain rules.** Greedy rows accept iff `target_argmax[j] == draft[j]`; the
  first mismatch emits that argmax. Stochastic rows accept with
  `min(1, p/q)` when `draft_probs != null`; `draft_probs == null` is the
  deterministic-drafter (target-only) rule, `q = one-hot(draft)`, so acceptance
  is `p(draft)` and the recovery draw is `norm(max(0, p - e_draft))`. The first
  rejection emits the corrected token from the full-vocabulary residual and
  stops. If every draft is accepted, the bonus token is drawn from the bonus
  row. `num_accepted` counts accepted drafts; emitted tokens = accepted + 1.
  `output_token_ids[b, 0..k_b]` holds the accepted drafts, the recovered/bonus
  token, then `-1` padding to `max_draft + 1`. Probability entries are finite
  non-negative floats; negative/NaN entries contribute zero. A draft id outside
  `[0, V)` forces a rejection; an out-of-range argmax emits `-1`.
- **Chain RNG.** `request_id = b`, `position = per_request_offset[b]` (null = 0),
  `stream_id` from `RngParams`. Draw indices: acceptance of draft `i` = `2*i`,
  residual at draft `i` = `2*i + 1`, bonus = `2*k_b + 1`. Greedy rows draw
  nothing. No host state is read or advanced; graph replay is a pure function of
  the device RNG fields.
- **Tree nodes and rows.** Node 0 is the current/root token (not emitted).
  `first_child`/`next_sibling` enumerate the tree; `candidates[node]` is the
  node token. `target_probs[b*N + retrieve_index[b,node]]` is the target
  distribution **after** that node; the same row supplies its argmax in greedy
  mode. `draft_probs[b*N + node]` is the draft distribution after the node.
  A child is verified with its parent's target/draft row. `max_depth` is the
  accepted-draft cap and must be `<= num_nodes - 1`.
- **Tree rules.** Greedy: follow the first child whose token equals the parent's
  target argmax; a miss emits that argmax. Probabilistic: normalize `q` over the
  current children, sample a child by `q`, accept with `min(1, p/q_norm)`, and
  on rejection sample the full-vocabulary `norm(max(0, p - q))`; `q` has support
  only on the children. Target-only (`draft_probs == null`): the first child is
  the proposal with `q = one-hot`, accepted with `p(child)`; a rejection draws
  `norm(max(0, p - e_child))`; other siblings are ignored. A leaf, a
  `max_depth` stop, or an accepted path gets one bonus token from the target row
  of the stopping node. `accept_length` counts accepted drafts;
  `output_token_ids` holds those candidates, then the recovered/bonus token, then
  `-1`. `accept_index` holds the accepted node ids, then the recovered node id
  when the residual token is a tree node (else `-1`), then `-1` padding.
- **Tree RNG.** Draw indices at depth `d` (0-based): q candidate = `4*d`,
  acceptance = `4*d + 1`, residual = `4*d + 2`, bonus = `4*d + 3`. Greedy rows
  draw nothing. Target-only uses the q/acceptance slot for the one-hot test and
  the residual slot on rejection.
- **Tree mask (B10).** `parent` describes the draft nodes only; `-1` means the
  parent is the committed prefix. Columns are `[prefix | nodes]` with
  `kv_len = prefix_lens[b] + num_nodes`; row `node` starts at
  `mask_indptr[b] + node * kv_len` and uses **no padding between query rows**.
  Bits are little-endian (`bit j` in `byte[j>>3]`, mask `1 << (j&7)`), matching
  the B8 `MaskMode::kCustom` consumer. A query sees every prefix bit plus its
  ancestor-or-self bits. `mask_indptr[0] = 0` and
  `mask_indptr[b+1] = mask_indptr[b] + num_nodes * kv_len`; the builder writes
  it. `position_offsets[b,node]` is the depth (root draft = 0); the engine adds
  `prefix_lens[b]` for the absolute position. The buffer is zero-filled and each
  32-bit word is set with `atomicOr`, so an unaligned row base is supported; the
  engine owns the buffer capacity and must provide a 4-byte aligned base.
- **Speculative state commit.** `intermediate_slots[b, j]` is the state that
  must be live after `j` accepted drafts (state after the prefix + `j` draft
  tokens). `state_commit_accepted` copies it into `live_slots[b]` for
  `0 <= j < max_draft`; `j == max_draft` performs no copy because the live slot
  already holds the state after the complete draft forward. Negative slots skip.
  This is the rollback half of the hybrid B9 contract; the test composes it with
  `causal_conv1d_update`.

## 1h. Low-bit rotated KV (B17, experimental)

Opt-in only: the entry points live in `kops::experimental::lowbit` and no
`kops::attn` plan selects them. A low-bit cache is a `kv::PagedKVDesc` with
`k_codec = v_codec = kLowBitWHT`, `k_dtype = v_dtype = kUInt8`,
`stride_unit = kByte`, 4-byte strides and `entry_bytes` equal to the **K**
record size (V strides may address a different record size). There is no
ragged (non-paged) path.

- **Codecs.** `kUniform3/4` = signed uniform mid-rise levels
  `2i - (2^b - 1)`, `scale = absmax / (2^b - 1)`, K rotated, V *not* rotated
  (attention.rs turbo4/turbo3). `kTurboMse2/3/4` = TurboQuant_mse
  (arXiv:2504.19874 Alg. 1): signed Hadamard, b-bit Lloyd-Max code,
  `scale = ||x|| / sqrt(D)`; K and V rotated. `kTurboProd2/3/4` = TurboQuant_prod
  (Alg. 2): `kTurboMse<b-1>` codes plus one QJL sign per coordinate of the
  residual and one fp32 gain; K only. `kFP8` and `kNVFP4` are reserved and
  return `kUnsupportedDtype` (not compiled).
- **Transform.** `x_rot = Hn (s .* x)`, `Hn` the Sylvester Hadamard divided by
  `sqrt(D)`, `s` the sign stream of `lowbit_sign_word(seed, stream, word)`
  (`bit 1` negates). Streams: 0 = K and Q, 1 = QJL sketch, 2 = V. The seed is
  `LowBitCodec::transform_seed`; Q and K must use the same one. D is 64, 128 or
  256 (`kUnsupportedShape` otherwise).
- **Record** (`packing_version = 1`, little-endian, 4-byte aligned):
  `[codes: D*b/8][signs: D/8, kTurboProd][scale: fp32][gain: fp32, kTurboProd]`.
  Element `i` is bits `[i*b, (i+1)*b)` of the code stream; the code is the index
  into the ascending level table (`code = #{thresholds < x_rot / scale}`), and
  `thresholds` are the midpoints. The scale is rounded to fp32 first and the
  codes are taken against the stored scale. A non-finite input poisons the
  record scale (NaN) instead of producing an out-of-range code.
  `lowbit_entry_bytes` is the declared size; physical cache bytes still come
  from `kv::kv_physical_bytes` on a byte-stride descriptor.
- **Attention.** The query is rotated once per CTA, the score is
  `scale * <q_rot, levels[code]> + gain * <q_sk, sgn>` with
  `q_sk = Hn (s' .* q_rot)`, and V is accumulated in its rotated basis and
  rotated back once per output row. Mask / window / chunk / softcap / ALiBi /
  sink / LSE follow §1e exactly. Sink is applied once, in the unsplit row or in
  `merge_split_kv` for split rows. Rows with no visible key give `o = 0`,
  `lse = -inf`. Decode is fp32; paged prefill dequantizes K/V to fp16 in shared
  memory and runs QK/PV on `mma.sync` fp16, so dequantized magnitudes must stay
  below 65504 (the same bound as an fp16 cache).
- **Compiled cells.** D 64/128/256; q/o bf16 or fp16 (same type); NHD or HND
  pages via byte strides; CSR or block table; any page size; decode `q_len`
  1..8; prefill causal, window, chunk, softcap, ALiBi, sinks and custom masks.
  Decode refuses custom masks (`kUnsupportedShape`); use paged prefill for
  them. TurboProd V, FP8/NVFP4 records and ragged KV are not compiled.
- **Plans and workspace.** `lowbit_decode_plan` and `lowbit_prefill_plan` follow
  the two-call workspace protocol of §1e: an empty workspace returns
  `kWorkspaceTooSmall` with `plan->workspace` filled, then the same call with a
  256-byte aligned workspace uploads the tables. Prefill lengths are host
  inputs: the plan is valid for those `qo_indptr` / `kv_lens` and must be
  rebuilt when they change (`lowbit_prefill_run` returns `kCapacityExceeded`
  for a different batch). Decode carries the planned capacities and fixed
  split count in `kernel_id`; a different codec returns `kPlanInvalidated`.
  Runs never allocate or synchronize, so both paths are graph-capturable.
- **Oracle and tolerances.** `ref::lowbit_encode_f64` / `lowbit_decode_f64`
  (`reference/ops.h`) are the fp64 reader/writer. `test_merge` checks (1) store:
  every named slot decodes to the fp64-encoded record within one code cell and
  unwritten slots keep their bytes; (2) decode/prefill attention against
  `ref::attention_f64` over the fp64-decoded records, with `rtol = atol = 2e-2`
  for the fp16 prefill path and `lse` within 2e-3; (3) bit-identical repeat
  launches. Quality (perplexity, long-context eval) is a model-level gate and
  is not claimed by these numerics.

## 1i. Communication: TP collectives and EP (B12)

**Transport.** A communicator is a `PeerBuffers` set: one single-writer staging
buffer per rank (`data[r]`, `buffer_bytes`, two parity slots of
`(buffer_bytes / 2) & ~127` bytes) and one signal array per rank
(`flags[r]`, `kSignalWords` words). The kernels only take pointers, so the same
body runs over NVLink/IPC peer mappings, over pinned mapped host memory when the
GPUs have no P2P, and over virtual ranks (N regions of one allocation on one
device, one stream per rank). `scratch` is rank-private device memory:
`kScrCalls` is the on-device call counter, `kScrError` records a `CommError`,
`kScrTimeoutUs` overrides the 20 s wait. `comm_configure` zeros the signal area
and scratch and must run on every rank before the first call and after a
`CommError`.

**Epoch.** The call epoch is `generation + calls + 1`, read on the device; the
last CTA advances `calls`. A captured graph therefore replays with a fresh epoch
without a host update. One flag per phase is posted by the last staging CTA
(device-scope arrival counter) with a system-scope release; consumers acquire
the peers' flags. Waits are bounded and write `CommError::kPeerTimeout` instead
of trapping. A message above 128 KiB is chunked. The chunked launcher uses one
data CTA plus one poster CTA per rank: a rank's data CTAs and poster must be
co-resident, and two CTAs per rank remain progress-safe on every supported
device. (Four data CTAs plus a poster produced a mixed poster/chunk view at
world 4/8 on the dev GPU.)

**Numerics.** `allreduce`, `allgather` and `reduce_scatter` convert to fp32,
accumulate in ascending rank order and round once. This makes every algorithm
and arrival order bit-identical (`kOneShot == kTwoShot == kDeterministic`).
`allreduce_fusion` applies the chain `AR -> residual add -> RMSNorm`; the residual
is rounded to the output dtype before the sum of squares, matching the separate
`norm::fused_add_rms_norm_out` chain. The optional per-token FP8 output
(`ActQuantOut::mode = kDynamicPerToken`) is produced in the same kernel: the
normalized value `y = r' * rstd * (w + offset)` is kept in fp32 for the scale and
the payload, while `norm_out` is the rounded output dtype. The scale is
`max(amax(y)/448, 1/(448*512))` (clamped by `scale_upper_bound` when set),
`q = e4m3(y / scale)` uses saturating OCP conversion, and `scales[row] = scale`. `kMXFP8`, `kNVFP4` and the
other quant modes return `kNotImplemented`; the `q` pointer may be byte-aligned,
`scales` must be 4-byte aligned.
`kMultimem`, `kLamportOneShot` and `kQuickReduce` are reserved and return
`kUnsupportedBackend` / `kNotImplemented`; no multicast object is used.

**EP.** `ep_dispatch_layout` writes per-rank/per-expert counts and the
`[tokens, ranks]` membership bitmap; the caller zeroes the outputs (the kernel
also zeroes them). `ep_dispatch`/`ep_combine` use `EPBuffers`: a peer-mapped
NVLink buffer per rank and a symmetric signal array of **two words per rank**
(`signals[2r]` arrival counter, `signals[2r+1]` ready epoch). A source region is
`(nvl_bytes / num_ranks) & ~127` bytes. Records are fixed-size and epoch-tagged;
stale records are ignored. Normal dispatch compacts in `(source rank, source
token)` order and writes `recv_src_info = (source_rank << 32) | token`,
`recv_topk_idx` (local expert ids, -1 for unused) and `recv_topk_weights`.
Low-latency dispatch scatters a `(token, local expert)` assignment into the
masked `[E_local, max_tokens_per_rank, hidden]` layout with atomics; the slot
order inside an expert is intentionally unordered. `max_tokens_per_rank` is the
masked row capacity **per local expert**. The expert id → rank map is
`expert / (num_experts / num_ranks)`. Combine sums contributions in ascending
source-rank order; when `weights_already_applied == false`, a normal row uses the
sum of its local routing weights and a low-latency slot uses its own weight.
Unused `src_info` entries must be `-1`; the dispatch leaves them untouched, so
the caller clears the array before the first call. Payload quantization is not
implemented (`payload_quant != kNone` → `kNotImplemented`).

**GEMM + communication.** `CommGemmParams::gemm` carries the local GEMM
(`ScaledGemmArgs`). The implemented body is `mma.sync` m16n8k16 BF16/FP16 with
fp32 accumulation and a 64x32 CTA tile; A/B are K-major, D is the operand dtype.
`allgather_gemm` treats `args.m` as the local M shard and writes
`[m_local * world, N]`; `gemm_reduce_scatter` and `gemm_allreduce` treat
`args.m` as the total M. `num_chunks` must divide M and be a multiple of the
world size. A chunk is owned by a small CTA group (up to 8 CTAs at world 2); the
last local CTA posts `chunk_ready[rank * num_chunks + chunk]`
is set to the epoch tag when the rank's data for that chunk is staged. The
staging slot is `[m_local * K * sizeof(T)]` for allgather and
`[M * N * sizeof(float)]` for the output-side collectives (fp32 partials).
The chunk call counter is the same on-device `kScrCalls` used by the
collectives, so graph replay is safe. The GEMM epilogue accepts `kEpiNone` and
`kEpiBias`; `fuse_residual_rmsnorm` is GEMM+AR only and adds residual + RMSNorm
after the reduction (the norm runs as a second kernel launch on the same
stream). Quantized or packed operands and MN-major operands return
`kNotImplemented` / `kUnsupportedLayout`.

**Test model and tolerances.** `test_communication` runs virtual ranks (2/4/8
regions on one GPU) and a host-staged 2-GPU world. Collectives are compared
bit-for-bit with an ascending-rank fp32 oracle; EP is compared with a host
reader of the normal compact order and with the masked slots; comm-GEMM is
compared with an fp64 matrix product (`atol = rtol = 3e-2` for BF16). Real P2P,
NVLink multicast and RDMA are not exercised by the dev machine; those cells stay
`authored` until the `multi` handoff kit runs them.

## 1i. Experimental W2 experts (B17, opt-in only)

`kops::experimental::moe_w2` is not selected by `moe::route`, `moe::moe_forward`
or the registry. It exists to evaluate the attention.rs Moet idea (2-bit expert
weights) on this repo's block-sorted MoE path.

- **Format.** One 2-bit code per element, four codes per byte (little-endian:
  element `4i+j` in bits `2j..2j+1`), into the fixed levels `{-4, -1, +1, +4}`.
  One fp32 dequant multiplier per `GROUP_SIZE` consecutive K elements;
  `GROUP_SIZE` is 16, 32, 64 or 128 and `K % GROUP_SIZE == 0`.
- **Pack.** `pack(packed, scales, weight, source, group_size)` writes
  `[E, N, K/4]` u8 codes and `[E, N, K/group_size]` fp32 scales.
  `source = WeightPacking::kNone` reads bf16 `[E, N, K]`;
  `source = kFP4E2M1` reads packed e2m1 `[E, N, K/2]` (element `2i` in the low
  nibble). The scale is `amax(group) / 4` (0 for an all-zero group), and each
  element takes the nearest level using the midpoints `-2.5, 0, +2.5`.
  No input scale array is consumed; this is a self-contained 2-bit re-encode.
- **GEMM.** `moe_w2::gemm` consumes `FusedMoEGemmParams` with
  `w_quant.packing = kW2`, `granularity = kPerGroup`, `group_k = GROUP_SIZE`,
  `w` = packed codes (byte stride `w_expert_stride = N*K/4`), `w_scale` = fp32
  scales (`w_scale_expert_stride = N*K/GROUP_SIZE`). Block-sorted semantics are
  the B7 ones: `sorted_token_ids`/`expert_ids`/`num_tokens_post_pad` from
  `moe::align_block_size`, `block_size = 16` (small-M WMMA) or `64` (one expert
  tile), A row = slot when `a_rows_are_slots` else `slot / top_k`, C row = the
  padded slot, and padding slots or `expert_id < 0` are untouched. `bias` is
  `[E, N]` in `c_dtype`; `mul_routed_weight` uses `topk_weights[slot_id]` (the
  original slot id, not the padded row), matching `fused_moe`/`moe_gemv`.
  `a_scale` and `w_zeros` are not implemented (`kNotImplemented`). A/C are
  bf16, fp16 or fp32; the 16-bit path uses WMMA `m16n16k16` tiles
  (16×64 for block 16, 64×64 for block 64; group 32 only) after dequantizing the 2-bit tile
  to 16-bit shared memory, and falls back to a scalar fp32 kernel for fp32
  activations or K not divisible by 64.
- **Quality.** `benchmark/compare/lowbit_quality.py` measures weight NMSE and
  expert-output error against the attention.rs Moet MXFP4-LUT format. On the
  synthetic N(0, 0.1) study the kops nearest-level pack is better on both
  (weight NMSE 0.180 vs 0.197, output relative Frobenius 0.414 vs 0.441).
  A model-level quality gate (PPL / long-context) still needs an engine and a
  checkpoint and is not claimed here.

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
  `assign_decode_tasks_kernel` (§1e), and (B9) the state parameter structs and kernel template
  lists of `conv1d_kernels.cuh`, `ssm_kernels.cuh`, `linear_attention_kernels.cuh`,
  `delta_rule_kernels.cuh` (head strides, slot counts, snapshot / intermediate tables,
  `ChunkMath`, `DeltaGatingParams`; `delta_rule_gating` takes the struct) (§1f).
  `KOPS_C_ABI_VERSION` and `KOPS_ABI_VERSION` must move together.
- Split-row softmax uses `Workspace.device_scratch` as `[B, CTAS_PER_ROW, 2]`
  fp32. `sampling_workspace_requirements` reserves the policy maximum for the
  vocabulary; a smaller workspace returns `kWorkspaceTooSmall`.
- Tests and benchmarks are C++ (`.cc`/`.cu`); PyTorch lives only under
  `benchmark/compare/` and is never linked into the library.
