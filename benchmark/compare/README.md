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

### Communication (B12)

```bash
build/bench/benchmark/bench_communication --dump-cases > out/b12/cases.json
build/bench/benchmark/bench_communication --json out/b12/kops-comm.json
python benchmark/compare/communication_torch.py --cases out/b12/cases.json \
    --json out/b12/torch-comm.json --calls 8 --groups 8
python benchmark/compare/compare_results.py out/b12/kops-comm.json out/b12/torch-comm.json
```

`communication_torch.py` runs two NCCL ranks with `torch.distributed` and the
same ids/shapes (allreduce, allgather, reduce-scatter, the fused residual+ RMSNorm
chain including dynamic per-token FP8, and the W=2 comm-GEMMs with `torch.mm` +
NCCL). It skips `peer_barrier`/EP and W=4/W=8 virtual-rank cases; DeepEP is not a
PyTorch package here and the node has two GPUs. The kops JSON may be measured in
graph mode while the PyTorch baseline is eager; run the C++ benchmark with
`--mode eager` for a launch-model-matched comparison, and report both.

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

## Third-party adapters (SM89 Stage 1)

`thirdparty.py` runs the same `--dump-cases` table through a library adapter and
writes the shared schema plus an `impl` field and a `skipped` list:

```bash
build/bench/benchmark/bench_norm --dump-cases > out/cases-norm.json
build/bench/benchmark/bench_norm --flush-l2 256 --rounds 3 --json out/kops-norm.json

python benchmark/compare/thirdparty.py --cases out/cases-norm.json --impl vllm \
    --flush-l2 256 --rounds 3 --json out/vllm-norm.json
python benchmark/compare/thirdparty.py --cases out/cases-norm.json --impl flashinfer \
    --flush-l2 256 --rounds 3 --json out/flashinfer-norm.json

python benchmark/compare/compare_results.py out/kops-norm.json \
    out/torch-norm.json out/vllm-norm.json out/flashinfer-norm.json
```

Current adapters: `vllm` (rms/fused norm, rope, fp8/int8 quant, reshape/append
KV, Marlin repack, topk/align/sum; the source build also exposes
`torch.ops._C.silu_and_mul` and `per_token_group_fp8_quant`), `flashinfer`
(norm, silu_and_mul, softmax, top-k mask, paged append, prefill/decode
attention), `sdpa` (prefill attention), `fa2` (dense prefill when the
`flash_attn` package is installed), `sglang` (sgl_kernel norm/rope/act/moe from
the v0.5.15 clone), `bnb` (NF4 GEMV M=1). Cases a library cannot express are
listed under `skipped` instead of aborting the run. `calibrate.py` records the
D2D copy and cuBLAS roofline; `report.py` renders the per-family markdown tables.

## llama.cpp GGUF GEMV

`llamacpp_mmvq.cpp` is a dev-only exact-shape `ggml_mul_mat` harness. It links
the local llama.cpp build, quantizes the weight with `ggml_quantize_chunk`, and
times a group of 20 launches after a 256 MB copy:

```bash
g++ -O2 -std=c++17 -I/usr/local/cuda/include \
    -I3rd/llama.cpp/ggml/include -I3rd/llama.cpp/ggml/src \
    benchmark/compare/llamacpp_mmvq.cpp -o /tmp/llamacpp_mmvq \
    -L3rd/llama.cpp/build-sm89/bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
    -L/usr/local/cuda/lib64 -lcudart -lcuda \
    -Wl,-rpath,$PWD/3rd/llama.cpp/build-sm89/bin \
    -Wl,-rpath,/usr/local/cuda/lib64
/tmp/llamacpp_mmvq
```

The harness flushes once per group, not once per launch, so its numbers are
**hot-L2** and only comparable to a kops run without `--flush-l2`. Use the same
protocol on both sides or add a per-launch flush before quoting a ratio.
