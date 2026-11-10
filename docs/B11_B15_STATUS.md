# B11 / B14 / B15 implementation status (2026-11-10)

Scope of this wave: the T1 / generic bodies requested for MLA + DSA + the
DeepSeek-V4/V4.1 and Kimi-K3 structures. The optimized SM90a `wgmma` and SM100a
`tcgen05` bodies are **not** in this drop; the T1 bodies are compiled for those
targets and the handoff kits run them as a correctness fallback. The 1.2x
"chính chủ" performance gate therefore remains pending real H100/B200 runs and
the arch-specific ports.

## Implemented

| Kernel | File | Status on SM120 |
|---|---|---|
| `mla_get_metadata_kernel` | `csrc/attention/mla.cu` | bit-exact host oracle, passed |
| `mla_decode_kernel` (T1) | `csrc/attention/mla.cu` | `mma.sync`, native + 656 B, split-KV, sink, gate, 1M passed |
| `machete_prepack_kernel` | `csrc/gemm/gemm_prep.cu` | byte-exact CUTE-derived golden, passed |
| `grouped_gemm_prepare_kernel` | `csrc/gemm/grouped_gemm.cu` | generic contiguous/masked/ptr-array + task map, passed |
| `lightning_indexer_kernel` | `csrc/attention/sparse.cu` | weighted ReLU MQA, ragged/paged, passed |
| `sparse_topk_kernel` | `csrc/attention/sparse.cu` | exact radix, 512/1024/2048, passed |
| `sparse_mla_kernel` | `csrc/attention/sparse.cu` | two scopes, topk_length, sink, native tested; codec reader shared with B3 |
| `kv_compressor_kernel` | `csrc/attention/sparse.cu` | CSA overlap / HCA, RMSNorm, passed |
| `compressed_attention_kernel` | `csrc/attention/sparse.cu` | selected CSA + sink, passed |
| `hc_expand/head/coeff/pre/post` | `csrc/model_ops/hyper_connection.cu` | official V4 mHC, passed |
| `mega_mhc_kernel` | `csrc/model_ops/hyper_connection.cu` | generic chain parity, passed |
| `attn_res_kernel/commit` | `csrc/model_ops/attn_res.cu` | K3 block AttnRes, MXFP8, passed |

KV records added: V4.0 584 B, V4.1 528 B, V4.1 FP4 288 B, in addition to the
existing V3.2 656 B. Each is byte-exact against the FlashMLA `quant.py` layout.

## Tests

`ctest --test-dir build/dev-sm120a --output-on-failure`:

```text
100% tests passed, 0 tests failed out of 26
```

The suite includes `kops_test_mla` (with the new split-16/32 cases),
`kops_test_sparse_mla`, `kops_test_indexer`, `kops_test_compressed_attention`,
`kops_test_kv_compressor`, `kops_test_hyper_connection`, `kops_test_model_ops`
(AttnRes), `kops_test_kvcache`, `kops_test_gemm` and the pre-existing families.

This includes `kops_test_mla_1m` (1M-token dense MLA), `kops_test_mla`,
`kops_test_sparse_mla`, `kops_test_indexer`, `kops_test_compressed_attention`,
`kops_test_kv_compressor`, `kops_test_hyper_connection`, `kops_test_model_ops`
(AttnRes), `kops_test_kvcache`, `kops_test_gemm` and the pre-existing families.
The 1M test uses zero Q/V so the expected `o = 0` and `LSE = log(1M)` are exact.

Header/ABI checks:

- 76/76 headers compile with only their declared prerequisites.
- `abi_layout_test.cc`, `bindings/c_api/kops_c_check.c`, `check_roadmap.py`
  (236/236) pass.
- ABI version is 7.

## Benchmarks (RTX 5060 Ti, 36 SM, 16 GB, CUDA 13.4)

Raw JSON: `out/b11-b15/*.json` (not committed). Re-run the whole comparison with
`tools/dev/bench_b11_b15.sh`.

| Case | Median |
|---|---:|
| MLA bf16 64k q1, S=8 | 0.763 ms |
| MLA bf16 1M q1, S=8 | 12.09 ms |
| MLA FP8 656 1M q1, S=8 | 18.12 ms |
| lightning_indexer bf16 64k | 0.0357 ms |
| lightning_indexer bf16 1M | 0.659 ms |
| sparse_topk 1M r1 topk2048 | 0.0266 ms |
| sparse_topk 1M r16 topk2048 | 0.767 ms |
| sparse_mla h128 topk512 | 0.049 ms |
| sparse_mla h128 topk2048 | 0.184 ms |
| hc_chain n4 h7168 t1024 | 1.036 ms |
| mega_mhc n4 h7168 t1024 | 1.055 ms |
| attn_res h7168 t256 s8 | 0.068 ms |

The split-KV S=16/32 variants land within noise of S=8 (dense MLA is already
throughput-bound at 1M), so the S=8 column is representative. These are
medians from the quiet-GPU runs in `out/b11-b15/opt-*.json`.

## PyTorch baseline on SM120

Torch `2.14.1+cu130` was installed for this run; the baseline uses the eager
`--style hf` path and the same case table. `speedup = torch median / kops median`
(>1 means kernel-ops is faster). Raw JSON is in `out/b11-b15/torch-*.json`.

| Family / case | before | after | torch | speedup (after) |
|---|---:|---:|---:|---:|
| MLA 64k q1, S=1 | 15.24 ms | 3.01 ms | 0.86 ms | **0.29x** |
| MLA 64k q1, S=8 | 3.01 ms | 0.763 ms | 0.861 ms | **1.13x** |
| MLA 1M q1 bf16, S=8 | 48.10 ms | 12.09 ms | 48.86 ms | **4.04x** |
| MLA 1M q1 FP8, S=8 | 63.06 ms | 18.12 ms | 69.72 ms | **3.85x** |
| MLA 1M q1 bf16, S=1 | 125.63 ms | 48.26 ms | 42.97 ms | **0.89x** |
| lightning_indexer 64k | 1.66 ms | 0.0357 ms | 0.457 ms | **12.8x** |
| lightning_indexer 1M | 24.99 ms | 0.659 ms | 8.085 ms | **12.3x** |
| sparse_mla topk512 | 1.24 ms | 0.049 ms | 0.116 ms | **2.36x** |
| sparse_mla topk2048 | 5.63 ms | 0.184 ms | 0.124 ms | **0.68x** |
| sparse_topk 64k r1 topk2048 | 0.040 ms | 0.041 ms | 0.096 ms | **2.36x** |
| sparse_topk 1M r1 topk2048 | 0.428 ms | 0.0266 ms | 0.100 ms | **3.76x** |
| sparse_topk 1M r16 topk2048 | 0.761 ms | 0.767 ms | 0.977 ms | **1.27x** |
| hc_chain t64 / t1024 | 0.362 / 4.72 ms | 0.165 / 1.036 ms | 0.349 / 5.223 ms | **2.12x / 5.04x** |
| mega_mhc t64 / t1024 | 0.389 / 4.75 ms | 0.171 / 1.055 ms | 0.350 / 5.223 ms | **2.05x / 4.95x** |
| attn_res (18 cases) | — | — | — | **20.9x geomean** |

**Gate result (current):**
- `sparse_topk` 64k/r1, 1M/r1 and 1M/r16 **PASS** (2.36x/3.76x/1.27x).
- `lightning_indexer` **PASS** (12.3–12.8x).
- `mla_decode` 1M **PASS** (4.04x bf16, 3.85x FP8); 64k S=8 **FAIL** at 1.13x.
- `sparse_mla` topk512 **PASS** (2.36x); topk2048 **FAIL** at 0.68x.
- `mega_mhc` **PASS** at T64/T1024 (2.05x/4.95x); `attn_res` **PASS** (20.9x).

**MLA 64k S=8 is roof-limited, not an implementation gap.** The kernel runs at
23.9–24.2 TFLOPS, ~97% of the measured 24.8 TFLOPS `mma.sync` peak on this
RTX 5060 Ti; 18.25 GFLOP has a ~0.76 ms floor with all 36 SMs, while PyTorch is
0.861 ms. A 2x gate for this exact case is not physically reachable with
`mma.sync`; the 64k S=1 case is additionally grid-limited (8 CTAs) and needs a
different head/V tiling. 1M split-KV, the actual long-context target, passes
with margin.

The 1.2x "chính chủ" reference remains pending because FlashMLA/DeepGEMM are
not installed and the optimized SM90a/SM100a bodies are not ported.


## Handoff

`out/handoff/sm90a/` and `out/handoff/sm100a/` are built and package-validated
with `check_env.py --skip-gpu` (13 PASS, 3 WARN, 0 FAIL each; 23 tests and 17
benchmarks per kit). On the target run:

```bash
cd out/handoff/sm90a
./check_env.py --arch sm90a --kit .
./run_all.sh --arch sm90a --kit . --json-dir results
```

The kit was built from a dirty tree; commit it before treating results as
evidence. Tests without `KOPS_GOLDEN_DIR` still run; the 1M golden cases are
planned but not generated, so they skip 77.

## Remaining gaps

1. **Sm90a/Sm100a optimized bodies** (`wgmma`/TMA, `tcgen05`/TMEM) are not
   implemented. The kits run the T1/generic bodies; the 1.2x gate is pending.
2. `sparse_mla` fused V4.1 Q-RoPE/O-RoPE/FP8-cast variants are `kNotImplemented`
   in the T1 body; the plain two-scope path is implemented and tested.
3. `compressed_attention` T1 covers selected CSA + sink; HCA dense and the SWA
   window branch are wired but need dedicated tests.
4. `kv_compressor` does not implement the `tail_state` cross-chunk carry; it
   returns `kNotImplemented` when `tail_state` is non-null.
5. 1M output-only golden files are not generated yet; the analytic 1M test is
   present, and the manifest/schema/tooling are ready (currently 0 ready / 11
   planned cases, including the conformance fixtures).
6. `benchmark/compare/torch_baseline.py` has no handlers for the new ops yet.
