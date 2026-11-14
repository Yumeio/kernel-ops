# Third-party benchmark runner — integration and WSL2 smoke results

Status: dev-only tooling; no kernel, ABI or build dependency was changed.

## One command

```bash
# Linux/WSL, auto-selects ~/venv-3rd/bin/python
bash tools/dev/bench_thirdparty.sh --all --rounds 3 --flush-l2 256

# or directly
python benchmark/compare/run_thirdparty.py --all --rounds 3 --flush-l2 256

# reuse kops numbers/cases from a previous build (e.g. Windows build, same GPU)
python benchmark/compare/run_thirdparty.py --all --rounds 3 --flush-l2 256 \
    --cases-dir out/tp-win/cases --kops-dir out/tp-win/kops --no-kops \
    --torch --impl all --external auto
```

The report is written to `out/thirdparty/<arch>-<gpu>-<date>/report.md`
(`--json-dir` / `--report` override it).

## Files

| File | Role |
|---|---|
| `benchmark/compare/run_thirdparty.py` | orchestrator: cases → kops → adapters → external → compare → report |
| `benchmark/compare/thirdparty.py` | base adapters + optional-module loader |
| `benchmark/compare/thirdparty_state.py` | FLA / mamba_ssm / causal_conv1d (B9) |
| `benchmark/compare/thirdparty_deepgemm.py` | DeepGEMM FP8/FP4/grouped (SM90/SM100) |
| `benchmark/compare/thirdparty_flashmla.py` | FlashMLA dense/sparse (SM90/SM100) |
| `benchmark/compare/thirdparty_trtllm.py` | TensorRT-LLM/FlashInfer `trtllm_*` |
| `benchmark/compare/deepep_baseline.py` | DeepEP EP dispatch/combine (multi-GPU) |
| `benchmark/compare/llamacpp_csv.py` | llama.cpp MMVQ CSV → shared JSON |
| `benchmark/compare/thirdparty_matrix.py` | family → impl/external matrix + known references |
| `tools/dev/3rd_manifest.tsv` | pinned 3rd-party checkouts |
| `tools/dev/clone_3rd.sh` | multiclone / check / lock |
| `tools/dev/bench_thirdparty.sh` | Linux/WSL wrapper |

## Adapter coverage (WSL2, RTX 3050 4 GB, SM86)

| Adapter | Status on this host | Notes |
|---|---|---|
| vLLM | runs | norm/activation/position/kvcache/quantization/gemm/moe subset |
| FlashInfer | runs | norm/activation/kvcache/quantization/sampling subset |
| SGLang sgl-kernel 0.4.5 | partial | `~/venv-3rd` has norm/activation/rotary; `~/venv-sgl` has the full official wheel |
| SDPA | runs | attention/merge subset |
| FLA | runs | GDN/KDA/lightning state cases; B9 |
| mamba_ssm | skip | package not installable here (binary vs torch/CUDA mismatch) |
| causal_conv1d | skip | same |
| DeepGEMM | skip | package missing + SM90/SM100 only |
| FlashMLA | skip | package missing + SM90/SM100 only; current `bench_mla` page/record shape also mismatches |
| TRT-LLM | skip | FlashInfer `trtllm_*` present but SM86/hardware mismatch |
| DeepEP | skip | needs ≥2 NVLink GPUs + NVSHMEM; harness exits 77 cleanly on 1 GPU |
| llama.cpp | skip | harness not built here |
| attention.rs lowbit/W2 | runs | SM86 binaries built; B17 case-id mapping is separate |
| NCCL | skip | needs multi-GPU |

## WSL2 smoke results

Rounds=1, no L2 flush, 10 families with kops cases; third-party ran on WSL2,
kops JSONs came from the Windows build on the **same** RTX 3050.

| Family | Cases | With 3rd ref | Geomean (best 3rd / kops) |
|---|---:|---:|---:|
| norm | 14 | 14 | 3.46× |
| activation | 9 | 9 | 1.64× |
| position | 22 | 22 | 3.43× |
| sampling | 127 | 127 | 2.72× |
| kvcache | 21 | 2 | 1.41× |
| quantization | 22 | 22 | 4.51× |
| gemm | 96 | 82 | 6.68× |
| moe | 13 | 13 | 3.06× |
| merge | 6 | 4 | 7.21× |
| state (FLA, T4096 skipped on 4 GB) | 11 | 3 | 0.75× |
| **Total** | **330** | **295** | **3.73×** |

`ratio > 1` means kernel-ops is faster. The `state` number is FLA-only and
uses the 3 small decode/linear cases; mamba_ssm/causal_conv1d were unavailable.

External attention.rs smoke: 19 cases, 17 refs, geomean 4.53×, 15 wins / 2 losses.

## Final rounds=3 run

`out/tp-wsl-final/` — `--rounds 3 --flush-l2 256` on WSL2, same 10 families.
Third-party adapters: vLLM, FlashInfer, SGLang 0.4.5 (partial in `venv-3rd`),
SDPA, FLA, TRT-LLM (skip on SM86), attention.rs external. kops JSONs came from
the Windows build on the same RTX 3050; no command errors.

| Family | Cases | With 3rd ref | Geomean (best 3rd / kops) |
|---|---:|---:|---:|
| norm | 14 | 14 | 3.33× |
| activation | 9 | 9 | 1.60× |
| position | 22 | 22 | 2.44× |
| sampling | 127 | 127 | 2.40× |
| kvcache | 21 | 2 | 0.76× |
| quantization | 22 | 22 | 3.84× |
| gemm | 96 | 82 | 6.18× |
| moe | 13 | 13 | 2.11× |
| merge | 6 | 4 | 4.38× |
| state (FLA, T4096 skipped on 4 GB) | 11 | 3 | 0.67× |
| **Total** | **341** | **298** | **3.18×** |

**223 wins / 75 losses / 0 ties**, 0 skipped commands, 0 errors.

Coverage highlights:

- norm: vLLM 6, FlashInfer 6, SGLang 4 results.
- activation: vLLM 2, FlashInfer 1, SGLang 2.
- position: vLLM 8, SGLang 8.
- gemm: PyTorch 82; vLLM/SGLang/TRT-LLM 0 (all skip on this adapter/hardware).
- moe: vLLM 3, PyTorch 13; SGLang/TRT 0.
- state: FLA 3; mamba_ssm/causal_conv1d missing.
- merge: PyTorch 4; SDPA 0 (non-attention cases).
- attention.rs external ran, but its fixed B17 ids do not align with the
  `bench_merge`/`bench_moe` ids; use `compare_b17.py` for B17.

Reports:

- `out/tp-wsl-final/report.md`
- `out/tp-wsl-final/summary.json`
- `out/tp-wsl-ext/report.md` (attention.rs external smoke)
- `out/tp-wsl-state2/report.md` (FLA state smoke)

`ratio > 1` means kernel-ops is faster. Use the rounds=3 report, not the smoke
table above, when quoting final WSL numbers.

## Caveats

- The worktree is dirty; all numbers are comparisons, not registry evidence.
- The 4 GB card OOMs on FLA's chunked delta-rule prefill (`T4096`); the runner
  skips those state cases automatically when `--skip-large`/VRAM < 8 GB.
- `kvcache` has only 2/21 third-party comparisons; it is not a full parity
  table.
- The vLLM GEMM adapter produces 0 results / 96 skips; PyTorch carries that
  family.
- SGLang in `~/venv-3rd` is the reduced norm/activation/rotary build. Use
  `PY=~/venv-sgl/bin/python` for the full 0.4.5 wheel.
- DeepGEMM, FlashMLA, TRT-LLM and DeepEP remain hardware/package gated; their
  adapters/harness exist and skip with a reason, but no kernel was executed.
