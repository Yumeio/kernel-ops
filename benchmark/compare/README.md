# benchmark/compare

Dev-only comparison against PyTorch. Nothing here is built, linked, tested by
`ctest`, or shipped.

```bash
cmake --preset bench
cmake --build --preset bench

build/bench/benchmark/bench_norm --dump-cases > out/cases.json
build/bench/benchmark/bench_norm --json out/kops-norm.json
python benchmark/compare/torch_baseline.py --cases out/cases.json --json out/torch-norm.json
python benchmark/compare/compare_results.py out/kops-norm.json out/torch-norm.json
```

Use `--style builtin` (default) for the strongest fused PyTorch primitive
(`F.rms_norm`, `F.silu`, `F.softmax`, ...) and `--style hf` for the eager
multi-op path a model implementation actually runs. The acceptance report
uses `--style hf` for the 2x claim and reports both.

Softmax with vocab > 32768 uses split-row CTAs; the C++ benchmark queries
`sampling_workspace_requirements` and passes `Workspace.device_scratch`. A
caller that passes an undersized workspace gets `kWorkspaceTooSmall`.

The inference-chain suite is in `benchmark/bench_chains.cu`:

```bash
build/bench/benchmark/bench_chains --dump-cases > out/cases-chains.json
build/bench/benchmark/bench_chains --json out/kops-chains.json
python benchmark/compare/torch_baseline.py --cases out/cases-chains.json \
    --json out/torch-chains.json --style hf
python benchmark/compare/compare_results.py out/kops-chains.json out/torch-chains.json
```

## B3 quantization comparison

```bash
build/bench/benchmark/bench_quantization --dump-cases > out/cases-quant.json
build/bench/benchmark/bench_quantization --json out/kops-quant.json
python benchmark/compare/torch_baseline.py --cases out/cases-quant.json \
    --json out/torch-quant.json
python benchmark/compare/compare_results.py out/kops-quant.json out/torch-quant.json
```

The PyTorch environment pinned by `pyproject.toml` / `requirements.txt` is
`torch==2.14.1+cu126`. `torch_baseline.py` maps every B3 op (including
`quantize_per_token_int8`, `norm_quant`, `act_and_mul_quant`) and the two fused
chains `chain_pre_norm_quant` / `chain_glu_quant`. MXFP4/NVFP4 baselines cast to
FP8 because PyTorch has no portable e2m1 cast; that makes their comparison
conservative for the native FP4 path.

## B4 weight preparation comparison

```bash
build/bench/benchmark/bench_gemm --dump-cases > out/cases-gemm.json
build/bench/benchmark/bench_gemm --json out/kops-gemm.json
python benchmark/compare/torch_baseline.py --cases out/cases-gemm.json     --json out/torch-gemm.json --style hf
python benchmark/compare/compare_results.py out/kops-gemm.json out/torch-gemm.json

# The B4 cases that live in bench_quantization (the B3 cases are filtered out):
python benchmark/compare/compare_results.py out/kops-quant.json out/torch-quant.json     --only quant.gptq quant.awq quant.gguf quant.fp4 quant.fp8_to quant.smooth
```

The baselines are what a Python checkpoint loader runs today, not a strawman:

- Marlin repack (`gptq_to_marlin`, `awq_to_marlin`, `fp4_to_marlin`): the vLLM `marlin_utils_test`
  formulation (unpack to int32, tile permutation, column packing). vLLM's production path is a CUDA op,
  so these four ratios (tens of x) are against the pure-PyTorch reference, not against that op.
- `awq_dequantize`, `fp8_block_quant`, `fp8_block_scale_to_ue8m0`: AWQ / DeepGEMM `per_block_cast_to_fp8`
  formulas, skipping the padding copy when the shape is already aligned.
- `weight_transpose`, `glu_interleave`, `int8_colsum`, `smoothquant_fold`: the single PyTorch primitive
  (`.t().contiguous()`, `torch.stack`, `.sum(dtype=int32)`, in-place `mul_`/`div_`), so cases at DRAM
  roofline show ~1x.
- `block_scale_swizzle`: torchao `to_blocked`.

Each baseline was checked against an explicit per-lane / per-block Python loop on small shapes before
timing. `smoothquant_compute_scales` has no benchmark case (K elements, launch bound).

## Fairness checklist

- Same GPU, driver, dtype and shape; pre-allocate tensors; `torch.inference_mode()`.
- Warm up before timing; report median/p10/p90, not a single run.
- Interleave kops and PyTorch blocks in the same session; repeat at least three times.
- Record clocks/temperature (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv`)
  next to the result; a laptop that throttles makes the number `diagnostic_only`.
- Kernel-only numbers come from the C++ benchmark; end-to-end numbers (TTFT,
  TPOT, tokens/s) come from vLLM/SGLang scripts run in their own environment.
- A PyTorch result is a comparison, not evidence for the registry; the evidence
  is the C++ JSON with its provenance fields.
