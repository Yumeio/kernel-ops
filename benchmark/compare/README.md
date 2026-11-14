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

## B14 compressed attention (CSA/HCA)

`bench_sparse` carries the DeepSeek-V4 compressed-attention cases (selected
compressed entries / HCA dense range + SWA + sink). The PyTorch baseline gathers
the same key set per row and runs the eager fp32-softmax / bf16 matmul step:

```bash
build/bench/benchmark/bench_sparse --only compressed_attention --dump-cases > out/cases-cattn.json
build/bench/benchmark/bench_sparse --only compressed_attention --rounds 3 --json out/kops-cattn.json
python benchmark/compare/csa_torch.py --cases out/cases-cattn.json \
    --json out/torch-cattn.json --only compressed_attention --rounds 3
python benchmark/compare/compare_results.py out/kops-cattn.json out/torch-cattn.json
```

`csa_torch.py` also implements the `kv_compress` reference math for the same
case table; this repository has no `kv_compress` benchmark case yet, so those
rows are reported under `skipped`.

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

`deepep_baseline.py` covers the EP rows (`ep_layout`, normal and low-latency
`ep_dispatch_combine`) through DeepEP v1.2.1. It needs an NVLink host with at
least `--world-size` GPUs, and NVSHMEM for the low-latency row; otherwise it
prints why and exits 77. Build steps, the API mapping and the 1x3050 status are
in `reports/deepep-baseline.md`.

The PyTorch environment pinned by `pyproject.toml` / `requirements.txt` is
`torch==2.14.1+cu126`. `torch_baseline.py` maps every B3 op (including
`quantize_per_token_int8`, `norm_quant`, `act_and_mul_quant`) and the two fused
chains `chain_pre_norm_quant` / `chain_glu_quant`. MXFP4/NVFP4 baselines cast to
FP8 because PyTorch has no portable e2m1 cast; that makes their comparison
conservative for the native FP4 path.

## Sampling matrix (B1/B6)

`bench_sampling` emits one matrix with three id families that must be reported
separately: `sampling.fused.*` (the one-launch sampler), `sampling.sample.*`
(the public `sample()` entry point) and the individual logits ops
(`softmax`, `argmax`, `bincount`, speculative decoding, ...). The eager
baseline covers all of them, but the aggregate geomean mixes a 15x legacy
traffic gap with 200-2000x small-op ratios and is not a sampler number. Use
`--groups 2` to print the per-family summary, and quote the fused geomean as
the fused-sampler result:

```bash
python benchmark/compare/compare_results.py out/kops-sampling.json \
    out/torch-sampling.json --groups 2 --json out/compare-sampling.json
```

`sample()` runs the fused body for greedy, unfiltered categorical, top-p-only
and min-p rows (top-k stays on the legacy probability path for the declared
tie-group semantics), so those rows are expected to match `sampling.fused.*`
within noise. A run of the old binary still shows the legacy numbers; record
the commit next to the table.

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

## One-command third-party benchmark

`run_thirdparty.py` is the dev-only orchestrator for every adapter that is
already installed or built. It does not clone or build a third-party library;
it records missing references with a reason.

```bash
python benchmark/compare/run_thirdparty.py --all --rounds 3 --flush-l2 256
```

On Linux/WSL, `tools/dev/bench_thirdparty.sh` is a thin wrapper that picks
`~/venv-3rd/bin/python` when present:

```bash
bash tools/dev/bench_thirdparty.sh --all --rounds 3 --flush-l2 256 \
    --report out/thirdparty/report.md
```

SGLang has two local options on this machine:

- `~/venv-3rd` has a **partial** `sgl_kernel` 0.4.5 build (norm/activation/
  rotary) that matches the adapter's norm/activation/position paths;
- `~/venv-sgl` has the **full** official 0.4.5 wheel with torch 2.11.0. Select
  it with `PY=~/venv-sgl/bin/python` when you need SGLang GEMM/MoE/KV ops.
  See `reports/sglang-sgl-kernel-wsl.md`.

Useful variants:

```bash
# availability and the family -> adapter matrix
python benchmark/compare/run_thirdparty.py --check
python benchmark/compare/run_thirdparty.py --list

# one family, one library
python benchmark/compare/run_thirdparty.py --families norm --impl vllm --rounds 1

# also run the PyTorch baseline (otherwise only third-party is compared)
python benchmark/compare/run_thirdparty.py --all --torch --torch-style hf

# reuse kops JSONs from a previous run, only run third-party
python benchmark/compare/run_thirdparty.py --all --kops-dir out/bench/kops --no-kops

# reuse both dumped cases and kops JSONs (no benchmark binary needed)
python benchmark/compare/run_thirdparty.py --all \
    --cases-dir out/bench/cases --kops-dir out/bench/kops --no-kops

# spot-instance friendly: skip completed JSONs
python benchmark/compare/run_thirdparty.py --all --resume
```

VRAM below 8 GiB automatically enables `--skip-large` (1M/128k/256k cases and
cases whose traffic exceeds ~55% of VRAM). Use `--no-skip-large` to force every
case.

Output goes to `out/thirdparty/<arch>-<gpu>-<date>/`:

```text
env.json  summary.json  report.md
cases/<family>.json
kops/<family>.json
thirdparty/<impl>/<family>.json
torch/<style>/<family>.json
external/<name>/<family>.json
compare/<family>.json
logs/
```

The report includes a coverage table (`wired`, `not_wired`, `needs_hardware`,
`reference_only`), the per-case best third-party ratio, and the known
references. Optional adapters now exist for:
`fla`/`mamba_ssm`/`causal_conv1d` (B9 state), `deepgemm` (SM90/SM100 FP8/FP4
GEMM), `flashmla` (SM90/SM100 MLA/sparse) and `trtllm` (TensorRT-LLM/FlashInfer
trtllm-*). They load only when the package and the module are present; missing
packages/architectures are reported as skip reasons. `DeepEP` has a separate
multi-GPU harness (`deepep_baseline.py`). DeepGEMM Mega MoE, TensorRT-LLM
containers and full FlashMLA sparse still need their own hardware/package
setup. The third-party numbers are comparisons, not registry evidence.

## B17 low-bit: cold protocol and interleaved A/B

`tools/dev/build_lowbit_baselines.sh` builds two dev-only harnesses under
`build/compare/`: `attention_rs_lowbit` (TurboQuant store/decode/prefill) and
`attention_rs_moe_w2` (Moet W2 pack). Both link nothing from kops.

The low-bit harness launches the upstream kernels with the exact grids from
`3rd/attention.rs/src/kernels/src/flash/flash_instantiate.cu`:

| op | grid | threads |
|---|---|---|
| store | `(num_tokens, num_kv_heads)` | 32 |
| decode | `(num_q_heads, num_seqs)` | 256 |
| prefill | `(num_q_heads, ceil(q_len/32), num_seqs)` | 128 (head_dim 128) |

Timing follows `benchmark/harness.h` (`ext_timing.h`): in cold mode
(`--flush-mb N > 0`) each sample is one launch after an N MiB D2D L2 flush and
one event pair; hot mode (`--flush-mb 0`) times groups of 20 launches. The JSON
carries `flush_l2_mb`, `rounds`, `protocol`, per-row
`median/p10/p90/min/samples/gbs`, and a `provenance` object with the kops commit
and the `3rd/attention.rs` commit. `run_thirdparty.py` passes its
`--flush-l2/--rounds/--only/--skip` to both harnesses, so a per-case run is:

```bash
build/compare/attention_rs_lowbit --only lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4 \
    --flush-mb 256 --rounds 3 --json out/b17/one-case.json
```

To cancel clock drift between the two sides, `interleave.py` runs A/B/B/A in one
process window and merges the per-run medians:

```bash
# attention.rs low-bit vs kops bench_merge, per lowbit case
python benchmark/compare/interleave.py \
    --a-bin build/bench/benchmark/bench_merge \
    --b-bin build/compare/attention_rs_lowbit \
    --family merge --only lowbit \
    --a-args "--rounds 3 --flush-l2 256" \
    --b-args "--flush-mb 256 --rounds 3" \
    --repeats 3 --out out/b17/interleave-lowbit.json

# old kops binary vs new kops binary (before/after)
python benchmark/compare/interleave.py \
    --a-bin /path/to/old/bench/benchmark/bench_merge \
    --b-bin build/bench/benchmark/bench_merge \
    --family merge --only lowbit.decode \
    --repeats 3 --out out/b17/interleave-before-after.json
```

`--a-cmd/--b-cmd` templates with `{case}` and `{out}` are supported for any
binary that writes the shared JSON schema. The merged file records both sides'
protocols and provenance, a `protocol-mismatch` warning when they differ, and
per-case `ratio = B/A` (plus `speedup_b_over_a`). Raw per-run JSONs and logs are
kept under `<out>.runs/`. Nothing is built by this script.

The kops JSON now also carries `protocol` and per-row `samples`; older binaries
simply omit those fields and the interleaver falls back to comparing medians
only.

## Clone the pinned 3rd-party trees

`tools/dev/clone_3rd.sh` reads `tools/dev/3rd_manifest.tsv` and clones/updates
the pinned checkouts into `3rd/`. It is separate from the benchmark runner and
is never called by it.

```bash
bash tools/dev/clone_3rd.sh --all --jobs 4
bash tools/dev/clone_3rd.sh --only sglang --force       # SGLang v0.5.16
bash tools/dev/clone_3rd.sh --check                     # verify HEAD against pins
bash tools/dev/clone_3rd.sh --list
bash tools/dev/clone_3rd.sh --root 3rd-full --all       # do not touch 3rd/
bash tools/dev/clone_3rd.sh --all --quiet               # file logs only
```

Every git command is written to `<root>/.clone-logs/<name>.log` and, unless
`--quiet` is passed, streamed to the terminal as `[<name>] ...`, so a parallel
run still shows what failed. Existing non-git directories are moved to
`3rd/.backup/<name>-<stamp>` before a fresh clone; existing git trees are
updated in place and are left untouched if dirty unless `--force` is given
(`--force` discards local changes there and repairs a checkout that failed
halfway). The run writes `3rd/MANIFEST.lock` with the resolved SHAs. `3rd/` is
gitignored.

On Windows, `sglang` and `TensorRT-LLM` contain paths longer than `MAX_PATH`
(260). The script sets `core.longpaths` for its own git calls and stores it in
each checkout, so a plain `git status` inside those trees works too.

