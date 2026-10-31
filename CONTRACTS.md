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
  remain unchanged. `KOPS_C_ABI_VERSION` and `KOPS_ABI_VERSION` must move
  together.
- Split-row softmax uses `Workspace.device_scratch` as `[B, CTAS_PER_ROW, 2]`
  fp32. `sampling_workspace_requirements` reserves the policy maximum for the
  vocabulary; a smaller workspace returns `kWorkspaceTooSmall`.
- Tests and benchmarks are C++ (`.cc`/`.cu`); PyTorch lives only under
  `benchmark/compare/` and is never linked into the library.
