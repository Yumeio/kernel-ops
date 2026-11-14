# DeepGEMM / FlashMLA / TensorRT-LLM adapters

Date: 2026-11-14
Scope: three dev-only adapters for `benchmark/compare`, following the
`thirdparty.py` contract `adapter(case, device) -> (fn, note) | (None, reason)`.

## Files

| file | `IMPLS` key | families |
|---|---|---|
| `benchmark/compare/thirdparty_deepgemm.py` | `deepgemm` | `bench_gemm` (dense/blockwise FP8, scale layout) |
| `benchmark/compare/thirdparty_flashmla.py` | `flashmla` | `bench_mla`, `bench_sparse` (MLA decode, sparse MLA, indexer) |
| `benchmark/compare/thirdparty_trtllm.py` | `trtllm` | `bench_gemm`, `bench_moe` (FP8/INT8 scaled GEMM, NVFP4) |

`thirdparty.py`, `run_thirdparty.py`, `thirdparty_matrix.py` and every kernel are
untouched; the three new modules are standalone, so `run_thirdparty.py --impl`
does not see them yet (wiring them into `PY_IMPLS`/`FAMILIES` is a one-line change
in files this task was asked not to edit).

Each module is importable without its heavy package and without `torch` at all
(`torch` is imported behind `try/except` because it is the benchmark runtime, not
the adapter target). `python benchmark/compare/thirdparty_<x>.py [--cases ...]`
prints the rule table and, with `--cases`, the concrete case-id -> API decision;
`--run --cases ... --json out.json` reuses `thirdparty.bench` and writes the
shared result schema with `impl = deepgemm|flashmla|trtllm`.

```bash
python benchmark/compare/thirdparty_deepgemm.py --cases out/thirdparty/cases/gemm.json
python benchmark/compare/thirdparty_deepgemm.py --run --cases ... --json out/deepgemm-gemm.json --flush-l2 256 --rounds 3
```

## Mapping

### DeepGEMM (`3rd/DeepGEMM` @ `5081f85`, `deep_gemm.__version__ == 2.8.2`)

| kops case / op | DeepGEMM call | condition |
|---|---|---|
| `gemm.mma_bf16.*`, `gemm.mma_bf16_tuned.*` (op `mma_bf16[_tuned]`) | `deep_gemm.bf16_gemm_nt(a, b, d)`, `D = A @ B^T` | dtype bf16 |
| `gemm.dense_gemv.bf16.m1.*` (op `dense_gemv`) | `bf16_gemm_nt` at `M = 1` | M must be 1 |
| `gemm.fp8_mma.blockwise.m*` (op `fp8_mma_blockwise`) | `fp8_gemm_nt((a,sfa), (b,sfb), d, recipe_a=(1,128), recipe_b=(128,128))` | N, K multiples of 128 |
| `gemm.scale_tma.f32.32768x56` (op `fp8_scale_to_tma`) | `get_mn_major_tma_aligned_tensor(x)` | K % 4 == 0 |
| `gemm.scale_tma.ue8m0.32768x56` (op `fp8_scale_to_tma_ue8m0`) | `get_mn_major_tma_aligned_packed_ue8m0_tensor(ceil_to_ue8m0(x))` | K % 4 == 0 |

Scale recipes are built with the library's own helpers so the layouts are the
ones DeepGEMM asserts on: `deep_gemm.utils.per_token_cast_to_fp8(..., gran_k=128)`
for A (per-token) and `per_block_cast_to_fp8(..., gran_k=128)` for B (128x128).
On SM100 the same fp32 scale factors are packed to UE8M0 by
`transform_sf_pair_into_required_layout`, so `disable_ue8m0_cast` stays `False`
and the case dtype/layout is unchanged; `mma_bf16_tuned` maps to the same call
because DeepGEMM selects its own tactic (the kops autotune is not forced).

Requires **SM90 or SM100/SM103** (DeepGEMM has only wgmma and tcgen05 bodies, no
`mma.sync` path), CUDA >= 12.9, torch >= 2.3, CUTLASS 4.0 submodule, C++20
`<format>`. Kernels are compiled at runtime by DeepJIT, so the first call of a
shape compiles inside the harness warm-up.

### FlashMLA (`3rd/FlashMLA` @ `395cb3a`, `flash_mla.__version__ == 1.0.0`, 2026-09-30 release)

| kops case / op | FlashMLA call | condition |
|---|---|---|
| `mla_decode.*` with a 576-wide latent cache (576 bf16 or 656 B fp8 rows) | `flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, 512, sched, num_splits, softmax_scale, causal=True, is_fp8_kvcache=...)` | page 64/128 **and** a dense-era build (<= 2025.09) |
| `mla_decode.*` with a 512-wide cache and `topkN` in the id | same call with `indices=[b, s_q, topk]` (DSA sparse decode, V4.1 528 B rows) | page 64/128 **and** the 2026.09.30 build **and** SM100 |
| `mla_metadata.*` | - | neither build exposes a bounded kernel here |
| `sparse_mla.*` | - | `flash_mla_sparse_fwd` instantiates `d_qk = 512`; kops uses 576 |
| `lightning_indexer.*` | - | FlashMLA has no indexer kernel |

The adapter probes the installed build instead of pinning one API:
`flash_mla_with_kvcache` gaining an `indices` parameter is the 2026.09.30
sparse-only build, otherwise it is the dense-era build. This matters because
the two releases are not compatible: 2026.09.30 removed dense MLA decode, the
656-byte FP8 KV format and Hopper support, and changed the KV layout to
512 data + scales per token. `flash_mla.get_mla_metadata` is a placeholder
object in the new build (the split schedule is produced by the first
`flash_mla_with_kvcache` call), so timing it would measure nothing.

### TensorRT-LLM / FlashInfer (`3rd/TensorRT-LLM` @ `1.4.0rc3`, pins `flashinfer-python==0.7.2`)

| kops case / op | call | condition |
|---|---|---|
| `gemm.fp8_mma.tensorwise.m*` | `torch.ops.trtllm.cublas_scaled_mm(a, b.t(), 1.0, 1.0, None, bf16)` | SM89+ |
| `gemm.fp8_mma.rowwise.m*` | `cublas_scaled_mm(a, b.t(), s_row[m], s_col[n], None, bf16)` | SM89+ |
| `gemm.int8_mma.m*` | `cublas_scaled_mm(a_int8, b_int8.t(), s_row[m], s_col[n], None, bf16)` | SM75+ |
| `gemm.fp4_gemv.nv.m*` | `flashinfer.gemm.mm_bf16_fp4(a_bf16, b_nvfp4, b_scale, ...)` | SM100+, FlashInfer must expose it |
| everything else | - | see below |

Provider order is FlashInfer first, then a full TensorRT-LLM install, as the
review note (`research_notes/review-B0-B8-va-chien-luoc-bench-gpu-thue.md`,
section 6.5) suggests: most trtllm-gen kernels are reachable through FlashInfer,
and only pull the NGC container for ops FlashInfer does not have. The
`torch.ops.trtllm.*` kernels exist only after `import tensorrt_llm`, so the
adapter imports the runtime lazily and reports the import error as the skip
reason.

`cublas_scaled_mm` is TRT-LLM's binding to **cuBLASLt**, not trtllm-gen; the note
string says so. It is used because it is the only entry that takes exactly the
kops FP8/INT8 dequant layouts (per-tensor, or per-row x per-column). FlashInfer's
`mm_bf16_fp4` is called through a signature filter (`inspect.signature`) so a
different FlashInfer minor version degrades to a skip instead of a wrong call.

## Required versions and hardware

| library | version used for the mapping | hardware | notes |
|---|---|---|---|
| DeepGEMM | 2.8.2 (`3rd/DeepGEMM` 5081f85) | SM90, SM100, SM103 | DeepJIT; CUDA >= 12.9, torch >= 2.3 |
| FlashMLA | 1.0.0, 2026-09-30 (`3rd/FlashMLA` 395cb3a) | SM100 (DSA); SM90/SM100 for <= 2025.09 builds | sparse-only release |
| TensorRT-LLM | 1.4.0rc3 (`3rd/TensorRT-LLM`) | SM80..SM120 (FP8 needs SM89+, FP4 SM100) | needs the container/compiled `libnvinfer` + `thop` |
| FlashInfer | 0.7.2 pin (docs 0.7.4) | trtllm-gen FP4 paths SM100 | `flashinfer-python==0.7.2` |

## Unsupported cases and why

DeepGEMM:

* `fp8_mma.rowwise`, `fp8_mma.tensorwise`, `grouped_fp8`, `moe_gemv_fp8` -
  kops applies a per-output-row x per-output-column dequant epilogue; DeepGEMM
  1D1D/1D2D group the second scale along K. Running unit scales through 1D1D
  would benchmark a different scale layout, so these skip.
* `fp4_gemv_*`, `fp4_marlin_*`, `w4a16_*` - DeepGEMM FP4 GEMM needs both
  operands quantized (`fp4 x fp4` or `fp8 x fp4`); these cases keep A in bf16.
* `mma_fp16*` - DeepGEMM's dense GEMM is bf16 only.
* `int8_mma`, `w4a8_*` - no INT8/W4A8 GEMM in DeepGEMM.
* GGUF (`q4_0`/`q8_0`/`q4_k`), LoRA SGMV, `dual_bf16`, `bmm_mla_*`, Machete
  prepack, `grouped_prepare`, `weight_transpose`, `router_*`, `gemv_argmax`,
  `int8_colsum`, `block_scale_swizzle`, `fp8_block_quant*` - no API, or the only
  exported helper is a PyTorch function (timing it would report torch).
* Any `gemm.mma_sm120.*` - DeepGEMM has no SM120 body.

FlashMLA (all of `bench_mla` today):

* Every `mla_decode` case is **page = 16** (`bench_mla.cu` uses `page = 16`) and
  FlashMLA accepts 64 (or 128) token blocks. Re-paging the cache would change
  the layout the kernel reads, not just the launch, so the adapter refuses
  instead of copying.
* The same cases are 576-wide (512 nope + 64 rope). The 2026.09.30 build only
  decodes 512-wide V4.1 records (528 B fp8 / 288 B fp4) through DSA, so even with
  page 64 they would need a build <= 2025.09 for the 656-byte format. The reason
  string says which of the two mismatches fired.
* `mla_metadata.*` - `get_mla_metadata` is a placeholder in the installed build.
* `sparse_mla.*` - kops qk_dim 576 vs `flash_mla_sparse_fwd` d_qk 512.
* `lightning_indexer.*` - no FlashMLA kernel; the reference is DeepGEMM
  `fp8_fp4_mqa_logits`, which needs fp8/fp4 operands while the case is bf16.
* `sparse_topk`, `compressed_attention` - no FlashMLA kernel.

TensorRT-LLM:

* `fp8_mma.blockwise` - trtllm-gen block-scaled FP8 uses 128x128 blocks on both
  operands; kops keeps A per-token (1x128).
* `fp4_gemv_mx`, `fp4_marlin_*`, `w4a16_*` - NVFP4 (16-element, e4m3 scales) vs
  MXFP4 (32-element, ue8m0) and Marlin packing; the bf16 x fp4 path is NVFP4
  only, and `marlin_nvfp4_gemm` takes FP4 activations.
* MoE: `moe_route`, `moe_grouped`, `moe_fused_router`, `moe_hash`,
  `moe_count_offsets`, `moe_align`, `moe_permute*`, `moe_unpermute`, `moe_sum`,
  `moe_latent`, `chain_moe_route_permute` are routing/sort/permute/reduce
  micro-ops with no standalone trtllm-gen kernel; `grouped_fp8`, `moe_gemv_fp8`,
  `moe_w2_gemm` hand over pre-computed sorted slots, while the trtllm-gen MoE
  kernels route inside the kernel, so the timed workload would differ.
* `mma_bf16*`, `mma_fp16*`, `dense_gemv` - not a trtllm-gen target here; use
  FlashInfer `mm_bf16`/cuBLAS.
* `fp8_scale_to_tma*`, `machete_prepack*`, MLA/sparse ops - no kernel.

## Test results

Environment: Windows 11, RTX 3050 Laptop (SM86, cc 8.6), `torch 2.14.1+cu126`,
Python 3.13.5 (`.venv`); WSL2 Ubuntu 24.04 with Python 3.12.3 (no torch).
None of `deep_gemm`, `flash_mla`, `tensorrt_llm`, `flashinfer`, `vllm` is
installed, which is exactly the skip path under test.

1. Compile and import (no heavy package pulled in):

```text
$ .venv/Scripts/python.exe -m py_compile benchmark/compare/thirdparty_{deepgemm,flashmla,trtllm}.py   -> OK
$ python -m py_compile ...                                     (system Python 3.13, no torch)        -> OK
$ wsl python3 -m py_compile ...                                (Python 3.12.3)                       -> OK

$ python -c "import thirdparty_deepgemm, thirdparty_flashmla, thirdparty_trtllm"
thirdparty_deepgemm      IMPLS=['deepgemm'] adapter=True main=True
thirdparty_flashmla      IMPLS=['flashmla'] adapter=True main=True
thirdparty_trtllm        IMPLS=['trtllm']   adapter=True main=True
heavy modules imported at import time: []
```

2. Mapping table (`--cases`, synthetic table built from the kops case ids:
`gemm`, `mla`, `sparse`, `moe` representatives, 25 cases). The fixtures and the
two test scripts are in `out/thirdparty-adapters-tests/` (`out/` is gitignored):

```text
$ python benchmark/compare/thirdparty_deepgemm.py --cases out/thirdparty-adapters-tests/cases_synthetic.json
6/25 cases have a DeepGEMM mapping   (mma_bf16 x2, dense_gemv, fp8_mma_blockwise, scale_tma x2)

$ python benchmark/compare/thirdparty_flashmla.py --cases out/thirdparty-adapters-tests/cases_synthetic.json
0/25 cases have a FlashMLA mapping   (page=16 / qk 576 / placeholder metadata)

$ python benchmark/compare/thirdparty_trtllm.py --cases out/thirdparty-adapters-tests/cases_synthetic.json
4/25 cases have a trtllm mapping     (fp8 rowwise, fp8 tensorwise, int8_mma, fp4_gemv_nv)
```

3. Skip path on the real device (every adapter, no exception escapes):

```text
deepgemm : deep_gemm requires SM90/SM100 (DeepJIT has no mma.sync body); device is sm86
flashmla : FlashMLA decode blocks are 64 (or 128) tokens; this case is page=16 ...
trtllm   : float8_e4m3fn scaled GEMM needs sm89+; device is sm86      (int8 falls through to:)
trtllm   : neither FlashInfer nor TensorRT-LLM is importable: No module named 'tensorrt_llm'; install ...
```

4. Package gate with the arch gate stepped over (`_device_cc` monkeypatched to
(9,0)/(10,0), the only way to reach it without renting an SM90/SM100 box):

```text
deepgemm bf16      : deep_gemm not importable: ModuleNotFoundError: No module named 'deep_gemm'; install DeepGEMM from 3rd/DeepGEMM ...
deepgemm fp8 block : deep_gemm not importable: ... (same)
flashmla dense p64 : flash_mla not importable: ModuleNotFoundError: No module named 'flash_mla'; install FlashMLA from 3rd/FlashMLA ...
flashmla dsa p64   : flash_mla not importable: ... (same)
trtllm fp8 row     : neither FlashInfer nor TensorRT-LLM is importable: No module named 'tensorrt_llm'; ...
trtllm nvfp4       : neither FlashInfer nor TensorRT-LLM is importable: No module named 'flashinfer'; ...
```

5. Layout/decision gates (pure functions, no package, no GPU):

```text
mla page 16        -> skip   (page must be 64/128)
mla dense p64      -> mla_decode_dense
mla dsa p64 topk   -> mla_decode_dsa
mla dsa p64 no topk-> skip   (needs topkNN in the id)
deepgemm fp16      -> skip   (bf16_gemm_nt only)
```

Not verified: no DeepGEMM/FlashMLA/TRT-LLM kernel was executed, because none of
the packages is installed and this GPU is SM86. The mapped calls are checked
against the vendored headers/READMEs (`3rd/DeepGEMM/csrc/apis/gemm.hpp`,
`csrc/apis/layout.hpp`, `csrc/utils/layout.hpp`; `3rd/FlashMLA/README.md`,
`flash_mla/flash_mla_interface.py`; `3rd/TensorRT-LLM/.../custom_ops/*.py`),
not against a run.

## How to enable

```bash
# DeepGEMM (SM90/SM100, CUDA >= 12.9)
cd 3rd/DeepGEMM && git submodule update --init --recursive && ./install.sh
python benchmark/compare/thirdparty_deepgemm.py --run --cases out/thirdparty/cases/gemm.json \
    --json out/deepgemm-gemm.json --flush-l2 256 --rounds 3

# FlashMLA (SM100 for the 2026.09.30 release; SM90 only with a <= 2025.09 checkout)
cd 3rd/FlashMLA && pip install -e . --no-build-isolation
python benchmark/compare/thirdparty_flashmla.py --run --cases ... --json ...

# TensorRT-LLM (NGC container is the supported path; needs libnvinfer + thop)
# or FlashInfer only, for the trtllm-gen FP4 paths:
pip install flashinfer-python==0.7.2
python benchmark/compare/thirdparty_trtllm.py --run --cases ... --json ...
```

Expected on a B200/H100 with the packages built: DeepGEMM maps the bf16,
blockwise-FP8 and scale-layout cases; FlashMLA still skips `bench_mla` unless
the case table moves to 64/128-token pages (and, for the 2026.09.30 build, to
512-wide V4.1 caches with DSA indices); TensorRT-LLM maps the FP8 tensorwise/
rowwise, INT8 and NVFP4 cases.

## Notes for the next step

* `bench_mla` and FlashMLA disagree on two contracts at once (page size 16 vs
  64/128, record width 576 vs 512). Either the MLA case table adds a
  `page` field with 64/128 and a 512-wide V4.1 variant, or the FlashMLA entry
  stays a documented "needs hardware" reference. The adapter already handles
  both shapes once such a case exists (`.p64`, `.topkN`).
* `fp8_mma.blockwise` is the one op where DeepGEMM and kops share a scale
  recipe exactly; it is the natural B11 gate on H100. `fp8_mma.rowwise` is a
  different recipe on all three libraries (DeepGEMM 1D1D, trtllm-gen 128x128,
  kops per-row x per-col) and needs a per-library mapping before it can be
  compared.
* The three modules are not in `PY_IMPLS`/`FAMILIES`, so `run_thirdparty.py`
  does not schedule them yet. Adding `"deepgemm"/"flashmla"/"trtllm"` to
  `PY_IMPLS` and to the `gemm`/`mla`/`sparse`/`moe` rows of `thirdparty_matrix.py`
  is all that is missing.
