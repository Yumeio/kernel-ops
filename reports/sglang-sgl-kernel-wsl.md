# SGLang sgl-kernel 0.4.5 on WSL2 (RTX 3050 Laptop, SM86)

**Status: working.** The existing `benchmark/compare/thirdparty.py --impl sglang`
adapter now runs real norm / activation / position case tables with sgl-kernel
0.4.5.

No repository file was changed except this report. The adapter scripts were not
edited.

## Environment

| item | value |
|---|---|
| host | WSL2 Ubuntu 24.04, RTX 3050 Laptop 4 GB, SM 8.6, driver 591.59 |
| CPU / RAM | 4 cores / 5 GB |
| CUDA toolkit | 12.9 (`/usr/local/cuda`) |
| sglang checkout | `~/work/3rd-test/sglang`, tag `v0.5.16`, commit `fdebc93` |
| sgl-kernel version | `0.4.5` (`sgl-kernel/pyproject.toml`, matches `sgl_kernel.version`) |
| `~/venv-3rd` | torch `2.14.0+cu132`, vllm, flashinfer; **no pip module** (uv used) |
| `~/venv-sgl` (new) | torch `2.11.0+cu130`, sglang-kernel `0.4.5` |
| official wheel | `sglang_kernel-0.4.5-cp310-abi3-manylinux2014_x86_64.whl`, 366 MiB, sha256 `5927a14dfff92ced8b1f7c7d5844144bc574ec58e604d4fd53e6f53c709003a8` |

## What was installed, and where

### A. `~/venv-sgl` — official full 0.4.5 wheel (recommended)

The PyPI wheel links `libcudart.so.13` / `libtorch.so` and is built for the
`torch==2.11.0` pinned by SGLang v0.5.16, so it was installed in a fresh venv
with exactly that torch:

```bash
uv venv --python 3.12 ~/venv-sgl
uv pip install --python ~/venv-sgl/bin/python --index-url https://pypi.org/simple torch==2.11.0
# 4 m 28 s (torch 2.11.0+cu130 + nvidia-cu13 runtime wheels, ~5.7 GB venv)

uv pip install --python ~/venv-sgl/bin/python --index-url https://pypi.org/simple \
    --no-deps sglang-kernel==0.4.5
# 35 s download (365 MiB) + install

~/venv-sgl/bin/python -c "import sgl_kernel; print(sgl_kernel.__version__)"
# 0.4.5
```

This is the complete package (all ops, `sm90/` + `sm100/` `common_ops.so`,
`flash_ops`, `flashmla_ops`, `spatial_ops`, `infllm_ops`).

### B. `~/venv-3rd` — reduced source build from the local checkout

The official wheel **cannot** load under `~/venv-3rd`'s torch 2.14
(see "Why not directly" below). A full source build from
`3rd/sglang/sgl-kernel` was not attempted: its CMakeLists compiles two full
sm90 + sm100 copies (~60 translation units each) including CUTLASS MLA/GEMM and
FA3 (a multi-hour build on 4 cores / 5 GB, with OOM risk).

Instead the translation units used by the adapter's norm / activation /
position paths were built from the local tree and registered directly:

```bash
# build helper (outside the repo)
#   ~/work/sglbuild/mini/build_mini_sgl.py
#   ~/work/sglbuild/mini/mini_extension.cc
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 \
  ~/venv-3rd/bin/python ~/work/sglbuild/mini/build_mini_sgl.py
# ~6.5 min wall clock, 5 TUs (sm_80 cubin, runs on sm_86)

SP=~/venv-3rd/lib/python3.12/site-packages
cp -r 3rd/sglang/sgl-kernel/python/sgl_kernel "$SP/"
mkdir -p "$SP/sgl_kernel/sm90" "$SP/sgl_kernel/sm100"
cp ~/work/sglbuild/mini/build/common_ops.so "$SP/sgl_kernel/sm90/"
cp ~/work/sglbuild/mini/build/common_ops.so "$SP/sgl_kernel/sm100/"

~/venv-3rd/bin/python -c "import sgl_kernel; print(sgl_kernel.__version__)"
# 0.4.5
```

Built sources: flashinfer `csrc/norm.cu` (pinned flashinfer
`bc29697`), `csrc/elementwise/{activation,fused_add_rms_norm_kernel,pos_enc}.cu`,
plus a shim registering `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`,
`gelu_and_mul`, `gelu_tanh_and_mul`, `rotary_embedding`.

**This is a partial package.** gemm / kvcache / attention / sampling / moe
ops are not registered. Use `~/venv-sgl` for a complete 0.4.5. To remove:
`rm -rf ~/venv-3rd/lib/python3.12/site-packages/sgl_kernel`.

Source-build fixes required (found by compiling):

1. `-std=c++20` — torch 2.14 headers `#error C++20 or later compatible
   compiler is required`; sgl-kernel 0.4.5's CMake still sets C++17.
2. Undefine torch's forced CUDA macros in the CUDA pass:
   `-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
   -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__`.
   `torch.utils.cpp_extension` injects them, flashinfer `vec_dtypes.cuh`
   `(float)half` then fails.
3. `PYBIND11_MODULE(common_ops, m) {}` in the shim — scikit-build normally
   generates the module entry point; a raw cpp_extension build needs it.

## Why the official wheel is not usable in `~/venv-3rd`

Testing the extracted wheel with `PYTHONPATH` under `~/venv-3rd` fails at
`import sgl_kernel` (nothing was installed into `venv-3rd` for this test):

```text
ImportError: .../sm100/common_ops.abi3.so: undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

The wheel is built for torch 2.11 / CUDA 13; `venv-3rd` has torch 2.14.0+cu132.
Downgrading torch there would break vllm/flashinfer, so it was not done.
A full source build against torch 2.14 is possible but takes hours (C++20 flag
above is the first blocker), hence the reduced build.

## Validation

Exact command from the task, now against `~/venv-3rd` (reduced build):

```bash
cd /mnt/c/Users/Tuand/Downloads/kernel-ops-headers/kernel-ops
~/venv-3rd/bin/python benchmark/compare/thirdparty.py \
  --cases out/tp-win/cases/norm.raw.json --impl sglang \
  --json /tmp/sglang-norm.json --rounds 1
# wrote /tmp/sglang-norm.json: 4 results, 10 skipped   (6.4 s including startup)
```

Same run with the full official wheel in `~/venv-sgl` gives the same numbers
within noise.

### norm (rounds=1, no L2 flush)

| case | venv-3rd (reduced) | venv-sgl (official 0.4.5) |
|---|---|---|
| `norm.rms_norm.bf16.h1024.b1024` | 0.0297 ms / 141.3 GB/s | 0.0307 ms / 136.6 GB/s |
| `norm.rms_norm.bf16.h4096.b1024` | 0.0973 ms / 172.5 GB/s | 0.0993 ms / 169.0 GB/s |
| `norm.fused_add_rms_norm.fp32.h4096.b1024` | 0.3727 ms / 180.1 GB/s | 0.3717 ms / 180.6 GB/s |
| `norm.fused_add_rms_norm.bf16.h4096.b1024` | 0.1874 ms / 179.1 GB/s | 0.1905 ms / 176.2 GB/s |

### activation

| case | time | GB/s |
|---|---|---|
| `act_and_mul.silu.fp32.t2048.i4096` | 0.5499 ms | 183.1 |
| `act_and_mul.silu.bf16.t2048.i4096` | 0.2804 ms | 179.5 |

### position (fp32 only; see skips)

| case | time | GB/s |
|---|---|---|
| `position.apply_rotary.fp32.t1` | 0.0266 ms | 1.6 |
| `position.apply_rotary.fp32.t16` | 0.0256 ms | 25.9 |
| `position.apply_rotary.fp32.t512` | 0.1239 ms | 171.4 |
| `position.apply_rotary.fp32.t4096` | 0.9336 ms | 181.9 |

### Numerics spot check (RTX 3050, bf16 unless noted)

| op | result |
|---|---|
| `rmsnorm` | mean rel. err 1.41e-3 (bf16 rounding) |
| `fused_add_rmsnorm` | out 1.41e-3, residual max err 0.0117 (bf16); fp32 case exact |
| `silu_and_mul` | 1.41e-3 |
| `rotary_embedding` (fp32, NeoX) | 0.0 |

## Skips (install is fine; these are adapter/library limits)

- norm: 2 x `rmsnorm` fp32 → `failed to dispatch data type Float`
  (FlashInfer norm supports fp16/bf16 only); 8 ops not exposed by the adapter
  (`layer_norm`, `l2_norm`, `gated_rms_norm`, `qk_norm`, `norm_quant`,
  `fused_add_layer_norm`, `fused_add_rms_norm_out`).
- activation: 7 non-`act_and_mul` cases unsupported (gate_mul, quantized).
- position: 4 bf16 `apply_rotary` → `expected scalar type BFloat16 but found
  Float`; the adapter always builds a **fp32** `cos_sin_cache` (`cache =
  torch.randn(4096, d, dtype=torch.float32)`) while sgl-kernel requires cache
  dtype == q/k dtype. 14 other position ops unsupported.

## Suggested runner fixes (not applied)

1. `sglang_adapter`, `apply_rotary`: use `cache = torch.randn(4096, d,
   device=device, dtype=dt)` (or cast q/k to fp32) so bf16 cases run.
2. `sglang_adapter`, `rms_norm`: skip fp32 with a reason
   (`dt not in (torch.float16, torch.bfloat16)`) instead of a runtime dispatch
   error per case.
3. `run_thirdparty.py` probes `sgl_kernel` in the current interpreter and looks
   for `3rd/sglang` relative to the repo root; in this WSL layout the clone is
   `~/work/3rd-test/sglang`, so the reason string is misleading. Point it at an
   env var (e.g. `SGLANG_ROOT`) or the active venv.
4. To use the full package, run the adapter with `~/venv-sgl/bin/python`; to
   stay in `~/venv-3rd`, the reduced build covers norm/act/rotary only.

## Cleanup

- Background build finished; no build/benchmark process from this task remains.
- GPU idle (`nvidia-smi` compute apps: none, 0 MiB used) after validation.
  A separate full-matrix run started by another session afterwards may appear
  in `nvidia-smi`; it is not part of this task.
- Artifacts: wheel `/home/tuand/work/sglbuild/whl/sglang_kernel-0.4.5.whl`,
  reduced build `/home/tuand/work/sglbuild/mini/`, JSONs
  `/tmp/sglang-{norm,activation,position}.json` and
  `/home/tuand/work/sglbuild/results/`.
