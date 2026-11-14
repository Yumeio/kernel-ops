# State adapters (B2 state cache, B9 recurrent)

Status: **module landed and discovered by `thirdparty.py`** (its
`_load_optional_impls()` merges `<family>_*` adapter modules automatically).
File: `benchmark/compare/thirdparty_state.py`.
Adapter contract, timing protocol and JSON schema are the ones already used by
`benchmark/compare/thirdparty.py`:

```python
adapter(case: dict, device: torch.device) -> (fn, note) | (None, reason)
```

`fn` is a zero-argument callable that performs exactly one library invocation;
all tensors are built before it is returned, so the timed region is the library
call plus the output allocation the library performs internally.

`IMPLS = {"fla": fla_adapter, "mamba_ssm": mamba_adapter, "causal_conv1d": causal_conv1d_adapter}`

`thirdparty.py`, `run_thirdparty.py`, `thirdparty_matrix.py` and every kernel are
untouched by this work; `thirdparty_matrix.py` already lists
`"impls": ["fla", "mamba_ssm", "causal_conv1d"]` for `state`, and the loader in
`thirdparty.py` picks the module up by name.

## Case → library mapping

`benchmark/bench_state.cu` emits 22 cases (21 in the stale local
`build/bench/benchmark/bench_state.exe`, which predates `state_commit`). All B9
cases are bf16 with fp32 recurrent state; the B2 cases are byte copies.

| case id | op | library | API | note |
|---|---|---|---|---|
| `conv1d_fwd.bf16.qwen3next.T4096` | conv1d_fwd | causal_conv1d | `causal_conv1d_fn(..., activation="silu", return_final_states=True)` | 2 × 2048 tokens, width 4, channel-last buffers |
| `conv1d_update.bf16.qwen3next.B64` | conv1d_update | causal_conv1d | `causal_conv1d_update(..., conv_state_indices=...)` | 64 rows, one token, slot-indexed state |
| `ssm_update.bf16.mamba2.H32P64N128.B64` | ssm_update | mamba_ssm | `ops.triton.selective_state_update.selective_state_update` | TIE_HDIM path (A per head), `dt_softplus=True`, z gate |
| `selective_scan.bf16.falcon_mamba.D8192N16.T4096` | selective_scan | mamba_ssm | `ops.selective_scan_interface.selective_scan_fn` | CUDA extension, 2 × 2048, `return_last_state=True` |
| `ssd_chunked_scan.bf16.mamba2.H32P64N128.T4096` | ssd_chunked_scan | mamba_ssm | `ops.triton.ssd_combined.mamba_chunk_scan_combined` | chunk 256, `dt_softplus`, initial + final state |
| `lightning_prefill.bf16.minimax.H64D128.T4096` | lightning_prefill | fla | `fla.ops.lightning_attn.chunk_lightning_attn` | slope from `(layer_idx, num_layers)`, not an explicit vector |
| `linear_decode.bf16.minimax.H64D128.B64` | linear_decode | fla | `fla.ops.lightning_attn.fused_recurrent_lightning_attn` | same decay caveat, one token per row |
| `delta_decode.gdn.bf16.B64` | delta_decode | fla | `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` | log-decay `g[B,T,Hv]`, `use_qk_l2norm_in_kernel=True` |
| `delta_decode.kda.bf16.B64` | delta_decode | fla | `fla.ops.kda.fused_recurrent_kda` | per-channel `g[B,T,Hv,Dk]`, same flags |
| `delta_prefill.gdn.fp32math.bf16.T4096` | delta_prefill | fla | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | WY/chunk, 2 × 2048 |
| `delta_prefill.gdn.tc.bf16.T4096` | delta_prefill_tc | fla | same `chunk_gated_delta_rule` | FLA has one chunked algorithm; the kops fp32math/tc split is internal |
| `delta_prefill.kda.fp32math.bf16.T4096` | delta_prefill | fla | `fla.ops.kda.chunk_kda` | per-channel decay, 2 × 2048 |
| `delta_prefill.kda.tc.bf16.T4096` | delta_prefill_tc | fla | same `chunk_kda` | as above |
| `fused_delta_decode.{gdn,kda}.bf16.B64` | fused_delta_decode | — | none | short conv + gate + delta recurrence + gated RMSNorm in one launch; FLA needs 4+ kernels |
| `delta_gating.{gdn,kda}.bf16.T4096` | delta_gating | — | none | FLA only gates inside its delta-rule kernels; `gdn_gate_fwd` covers `g` alone (per-head only) |
| `state_copy/gather/scatter/zero/commit` | state cache | — | none | engine-owned paged pool; no third-party equivalent |

Supported case counts: **fla 8/22**, **mamba_ssm 3/22**, **causal_conv1d 2/22**
(a case is one row of the dumped table).

## Geometry

The case id carries only part of the geometry; the rest is fixed by
`benchmark/bench_state.cu` and encoded in the module:

| family | defaults | parsed from the id |
|---|---|---|
| conv1d | dim = `cols`, width 4, SiLU, prefill 2 × `rows/2` | — |
| ssm_update | G = 8, H 32, P 64, N 128 | `.H<H>P<P>N<N>` |
| selective_scan | ngroups 1, N 16 | `.D<dim>N<N>` |
| ssd_chunked_scan | G = 8, chunk 256, H 32, P 64, N 128 | `.H<H>P<P>N<N>` |
| lightning | H 64, Dk = Dv = 128 | `.H<H>D<D>` |
| delta rule | Hk 16 (GDN) / 32 (KDA), Hv = `cols/D`, D 128 | `.D<D>`, `.gdn.` / `.kda.` |

Prefill cases are run as **2 sequences** of `rows/2` tokens (the bench's own
batching; prefill state is zero-initialized, decode state is a zeroed fp32
pool). Decode cases use `rows` as the batch.

## Required packages

| library | pin (`tools/dev/3rd_manifest.tsv`) | import | requirement |
|---|---|---|---|
| flash-linear-attention | v0.6.0 | `fla.ops.*` | the PyPI/GitHub 0.6.0 wheel ships only `fla/layers` + `fla/models`; `fla.ops` lives in **`fla-core==0.6.0`** and must be installed too. `torch>=2.7`, `triton>=3.3`, `einops`, `transformers>=5.6.2,<6` |
| mamba_ssm | v2.3.2 (package reports `__version__ = 2.3.1`) | `mamba_ssm.ops.*` | `mamba_ssm/__init__.py` imports `selective_scan_cuda`, so **any** `mamba_ssm.ops.*` import needs the CUDA extension built by `pip install mamba-ssm==2.3.2` (nvcc matching the torch CUDA build). `selective_scan_fn` itself is that extension; `selective_state_update` and `mamba_chunk_scan_combined` are Triton |
| causal-conv1d | v1.7.0 | `causal_conv1d` | CUDA extension (`causal_conv1d_cuda`); v1.7.0 wraps the kernels in `torch.library.custom_op`, so it needs a recent torch |

Install used for validation (WSL, venv `~/venv-3rd`, nothing system-wide):

```bash
uv pip install --python ~/venv-3rd/bin/python --no-deps fla-core==0.6.0
uv pip install --python ~/venv-3rd/bin/python --no-deps flash-linear-attention==0.6.0
```

`mamba_ssm` and `causal_conv1d` were **not** installed: both build CUDA
extensions and the venv's torch is `2.14.0+cu132` while the WSL toolkit is
CUDA 12.9, so `torch.utils.cpp_extension` refuses the build (major-version
mismatch). Their adapter paths were validated by the missing-package skip path.

The `3rd/` checkouts are gitignored and absent in this working tree; the API
surface used here was read from the pinned upstream tags (`fla-org/
flash-linear-attention` v0.6.0, `state-spaces/mamba` v2.3.2,
`Dao-AILab/causal-conv1d` v1.7.0). Only the installed packages matter at run
time.

## Deliberate differences (read before using the numbers)

1. **Lightning decay.** kops takes an explicit per-head `slope[H]` buffer; FLA
   derives the slope from `(layer_idx, num_layers)`. The adapter passes
   `layer_idx=0, num_layers=1`. The bench fills the kops slope with zeros
   (`b.get<float>(H)`), i.e. the kops case has no decay at all while the FLA
   call has geometric decay. Same operator family, different decay values; the
   arithmetic and memory traffic are unchanged.
2. **`delta_prefill_tc` vs `delta_prefill`** map to the same FLA call. FLA has no
   fp32math switch; the two kops cases differ only in their internal math mode.
3. **`selective_scan` initial state.** mamba_ssm v2.3.2 `selective_scan_fn` has no
   initial-state argument. The kops case zero-initializes its state pool, so the
   omission is numerically equal, but kops also writes the final state and mamba
   only returns it (`return_last_state=True`, no store).
4. **conv1d state layout.** `causal_conv1d_fn` requires channel-last buffers: the
   `(batch, dim, width-1)` conv state must have `stride(1) == 1`. kops stores
   `[slot][dim][width-1]` (width contiguous). The adapter allocates
   causal_conv1d's layout; a production engine would need a repack per step.
5. **Output allocation.** The kops benchmark reuses preallocated buffers; every
   library call here allocates its output (and, for FLA, runs autograd
   `Function`/`input_guard` Python dispatch and autotune lookups) inside the
   timed region. That overhead is part of the measured library call.
6. **State pool vs dense state.** kops indexes a slot pool (`state_indices`); the
   adapters pass a dense `(batch, heads, Dk, Dv)` fp32 state. Work is identical,
   addressing differs.

## Unsupported cases and why

| case op | all libraries | reason |
|---|---|---|
| `fused_delta_decode` | — | short conv + decay gate + delta recurrence + gated RMSNorm in one launch; no library has a one-call equivalent (FLA would be 4+ kernels) |
| `delta_gating` | — | FLA fuses gating into its delta-rule kernels (`use_gate_in_kernel`); `fla.ops.gated_delta_rule.gate.gdn_gate_fwd` computes `g` only (per-head, no `beta`), so it is not the same one-call op |
| `state_copy/gather/scatter/zero/commit` | — | engine-owned paged byte movement; no third-party kernel exists |

Per library, the remaining cases are "owned" by another adapter and return a
reason instead of a benchmark row (`fla` refuses conv1d/Mamba, `mamba_ssm`
refuses GDN/KDA/lightning, `causal_conv1d` refuses everything else).

## Test results

Environment: Windows 11 host + WSL2, Python 3.10.11 (host) and 3.12.3 (WSL
`~/venv-3rd`), `torch 2.14.0+cu132`, `triton 3.8.0`, `fla 0.6.0` +
`fla-core 0.6.0`, RTX 3050 Laptop (SM86, 4 GB).

| check | command | result |
|---|---|---|
| syntax | `python -m py_compile benchmark/compare/thirdparty_state.py` | pass |
| mapping, no packages | `python benchmark/compare/thirdparty_state.py` | pass; 22-case table, all three packages `missing` |
| no heavy import at module load | `import thirdparty_state; "torch" in sys.modules` | `False` |
| mapping on the dumped table | `thirdparty_state.py --cases <state.raw.json> --impl fla` | pass; 8 `yes` / 13 `-` |
| adapter setup, every case × every impl, `device=meta` | script in test log | fla `ready=8 skipped=13`, mamba_ssm `ready=0 skipped=21`, causal_conv1d `ready=0 skipped=21`; `torch.cuda.memory_allocated() == 0` |
| missing-package skip path | `--synthetic --impl mamba_ssm,causal_conv1d` | every row `skip <reason>`, exit 0 |
| tiny GPU run, FLA | `--synthetic --impl fla` | `chunk_lightning_attn`, `fused_recurrent_lightning_attn`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent_kda`, `chunk_kda` all `ok` on SM86 |
| runner integration | `import thirdparty; sorted(thirdparty.IMPLS)` | `fla`, `mamba_ssm`, `causal_conv1d` present; `IMPLS["fla"].__module__ == "thirdparty_state"` |
| runner end-to-end, 3 tiny cases | `thirdparty.py --impl fla --rounds 1 --iters 1` | 2 results (`delta_decode.gdn`, `linear_decode`), 1 skip (`ssm_update` → mamba_ssm), exit 0 |
| runner skip-only path | same cases, `--impl mamba_ssm` / `--impl causal_conv1d` | 0 results, 3 skips with reasons, exit 0 |
| mamba_ssm / causal_conv1d numerics | not run | packages not installable in this venv (CUDA 12.9 vs torch cu132); needs a matching nvcc |

The FLA synthetic run is the only GPU work performed; no full benchmark was run
and the GPU is left idle. Triton autotuning dominates the first run (a few
minutes for all six entry points); the second run is cache-warm.

## Reproduce

```bash
# mapping only (no GPU, no packages)
python benchmark/compare/thirdparty_state.py
python benchmark/compare/thirdparty_state.py --cases out/tp-win/cases/state.raw.json

# tiny GPU validation (FLA on SM86+)
python benchmark/compare/thirdparty_state.py --synthetic --impl fla

# through the shared runner (the module is auto-discovered)
python benchmark/compare/thirdparty.py --cases out/tp-win/cases/state.raw.json \
    --impl fla --json out/fla-state.json --rounds 3 --flush-l2 256
```
