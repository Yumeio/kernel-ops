# Third-party notices

kernel-ops is licensed under **Apache-2.0** (see `LICENSE`).

This file records, per area, which upstream project a contract was derived from and under which
license. It exists because `csrc/` is a **contract-first** codebase: most headers describe the
semantics of an existing production kernel (vLLM, SGLang, hpc-ops, FlashMLA, DeepGEMM,
attention.rs, TRT-LLM, CUTLASS) before any body is written here.

## Status

**The only upstream data in `csrc/` are the GGUF codebook tables ported from llama.cpp** (see
"Ported code" below); everything else is an independent implementation. Every header is an
independent declaration; the "Reference" column below names the design source only. When
implementations are ported (M1 onward), each ported file MUST:

1. add a header comment `// Ported from <project> <path> @ <commit> (<license>)`;
2. add a row to the table below with the exact commit and file list;
3. keep the upstream license text available under `third_party/licenses/<project>.txt`;
4. be reviewed for license compatibility (MIT / BSD-3 / Apache-2.0 are compatible with
   Apache-2.0 distribution; GPL/AGPL are not accepted).

## Design references

| Project | License | Referenced for | kernel-ops files |
|---|---|---|---|
| vLLM | Apache-2.0 | KV-cache write/gather/convert contracts; `copy_batch_async` (`cuMemcpyBatchAsync`); batch-invariant constraints (`csrc/core/batch_invariant.hpp`); fused allreduce design; op naming for drop-in parity | `kvcache/*`, `docs/KV_CACHE_PARITY.md`, `docs/CONTRACTS.md` §3 |
| SGLang + sgl-kernel | Apache-2.0 | KV transfer (`kvcacheio/transfer.cu`); grammar bitmask apply; expert specialization; deterministic allreduce; mscclpp integration; GDN/KDA op surface (Triton) | `kvcache/*`, `sampling/*`, `communication/*` |
| FlashInfer | Apache-2.0 | paged KV layouts (`NHD`/`HND`/CSR append); dual-pivot rejection sampling; plan/workspace split | `kvcache/kv_layout.cuh`, `sampling/sampling_kernels.cuh` |
| FlashMLA | MIT | MLA decode + scheduler metadata layout; `ds_mla` 656 B FP8 record; sparse MLA; fused norm+rope+attn | `attention/mla_kernels.cuh`, `kvcache/kv_layout.cuh` (`kMLAFp8DS`, `kMLAFp8DSFNUZ`) |
| DeepGEMM | MIT | FP8/FP4 block-scale layouts (`128x4`, UE8M0 packing); contiguous/masked/k-grouped GEMM; mega-MoE scheduling concepts; MQA-logits metadata | `gemm/*`, `kvcache/kv_layout.cuh`, `mega/*` |
| hpc-ops (Tencent) | MIT | masked `act_mul_and_quant`; dynamic decode task scheduling; fused allreduce + RMSNorm; gated MLA GEMM epilogue; filtered radix top-k; BF16x2 router decomposition | `activation/*`, `attention/decode_kernels.cuh`, `communication/allreduce_kernels.cuh`, `gemm/*`, `attention/sparse_kernels.cuh` |
| attention.rs | MIT | low-bit WHT KV codecs (K8V4/K4V4/K3V4); TurboQuant turbo3/turbo4 store/decode/prefill structure; paged attention structure; experimental 2-bit MoE | `attention/lowbit_kernels.cuh`, `attention/lowbit.cu`, `kvcache/kv_layout.cuh` (`kLowBitWHT`), `moe/fused_moe_kernels.cuh` (M04) |
| TurboQuant (Zandieh et al., arXiv:2504.19874) | GPL-3.0 (reference checkout only) | `TurboQuant_mse` / `TurboQuant_prod` semantics (signed rotation, Lloyd-Max code, QJL sign plane and gain). No upstream code was copied: the level tables were regenerated independently from the published Lloyd-Max fixed point by `benchmark/compare/lowbit_codebook.py` and are re-derived in fp64 by `tests/unit/test_merge.cc`. | `attention/lowbit_kernels.cuh`, `attention/lowbit.cu` |
| TensorRT-LLM | Apache-2.0 | Lamport one-shot allreduce; allreduce fusion epilogue; TRT-LLM-gen shuffled weight layout; `kEpiSigmoidGateMul` | `communication/allreduce_kernels.cuh`, `common/quant.cuh` (`kTrtllmShuffled`), `gemm/gemm_params.cuh`, `quantization/weight_repack.cu` (`fp4_backend_shuffle`) |
| CUTLASS / CuTe | BSD-3-Clause | tile/cluster/stage vocabulary, `GemmTraits`, block-scaled MMA kinds (no code yet) | `gemm/gemm_params.cuh`, `docs/GEMM_MATRIX.md` |
| Local read-only reference checkout (`3rd/csrc`, commit `d99cb25`) | Apache-2.0 (same project lineage) | device-body and validation reference for the B0/B1 first slice (`residual`, `rope`, `embedding`, `sampling`, `merge`, `norm`); adapted to the `kops` host API, no framework types | `model_ops/residual.cu`, `position/rope.cu`, `embedding/embedding.cu`, `sampling/*.cu`, `attention/merge.cu`, `norm/norm.cu` |
| FlashAttention (Dao-AILab) | BSD-3-Clause | FA2 (sm80) / FA3 (sm90) / FA4 (sm100) mainloop structure | `attention/prefill_kernels.cuh`, `attention/decode_kernels.cuh` |
| ggml | MIT | GGUF block type numbering (`GGML_TYPE_*`) | `common/quant.cuh` (`GGUFType`) |
| llm-awq | MIT | AWQ packing and zero-point convention | `common/quant.cuh` (`kAWQ`), `quantization/weight_repack_kernels.cuh` |
| mistral.rs | MIT | hybrid cache (attention + recurrent) layout | `kvcache/kv_layout.cuh` (hybrid family, planned) |

## Derived implementations (B4 weight preparation, 2026-10-31)

These bodies are independent CUDA implementations of formats that an upstream project defines.
Layouts, constants and formulas were read from the upstream source in `3rd/` and re-expressed; no
upstream text was copied. They are listed here because the output is only correct if it matches
the upstream format bit for bit.

| Area | Derived from | License | Commit | What was taken | kernel-ops files |
|---|---|---|---|---|---|
| Marlin repack (GPTQ / AWQ / FP4) | vLLM `csrc/libtorch_stable/quantization/marlin/{gptq,awq}_marlin_repack.cu` (Marlin by IST-DASLab) | Apache-2.0 | unknown (vendored without `.git`) | 16x64 tile, lane -> (n, k rows) mapping, nibble/byte pack orders, AWQ column order | `quantization/weight_repack.cu` |
| Marlin scale and zero-point permutation | vLLM `marlin_utils.py` (`marlin_permute_scales`, `awq_to_marlin_zero_points`; python, not vendored, re-derived from the kernel layout and checked against the numpy formulation in the test) | Apache-2.0 | unknown | 64-/32-wide chunk permutations, zero-point interleave | `quantization/weight_repack.cu`, `tests/unit/test_quantization.cc` |
| GGUF dequantization (20 block types: Q4_0 ... IQ4_XS, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S) | llama.cpp `ggml/src/ggml-quants.c` `dequantize_row_*` and `ggml-common.h` block structs (first read through `3rd/sglang/sgl-kernel/csrc/quantization/gguf`) | MIT (ggml) / Apache-2.0 (sglang) | llama.cpp `0ec4027`; sglang `96a04cb` | block structs, per-type element formulas, `kvalues_iq4nl`; the output is checked bit for bit against `dequantize_row_*` built from the llama.cpp source | `quantization/weight_repack.cu`, `tests/unit/test_quantization.cc` |
| TRT-LLM-gen weight shuffle (`fp4_backend_shuffle`, `kTrtllmShuffled`) | TensorRT-LLM `tensorrt_llm/quantization/utils/fp4_utils.py` (`shuffle_matrix_a`, `srcToDstBlk32RowMap`), `cpp/tensorrt_llm/thop/moe/trtllmGen/fp4BlockScaleMoe.cpp` (`shuffleMatrix` is `index_select`), `cpp/tensorrt_llm/kernels/quantization.cu` + `thop/fp8Op.h` (`block_scale_interleave`, `computeSFIndex`), `_torch/moe/fused_moe/quantization.py` (call order, `epilogue_tile_m = 128`) | Apache-2.0 | TensorRT-LLM `09198b7` | 32-row block row map (kernel: `(i % 4) * 8 + i / 4`), 128x4 scale offset formula; the test oracle carries the 32-entry row map and the `computeSFIndex` expression | `quantization/weight_repack.cu`, `tests/unit/test_quantization.cc` |
| AWQ dequantization | llm-awq / vLLM `awq/dequantize.cuh` | MIT / Apache-2.0 | llm-awq vendored | nibble order `{0,2,4,6,1,3,5,7}` | `quantization/weight_repack.cu` |
| FP8 block quantization, UE8M0 | DeepGEMM `deep_gemm/utils/math.py` (`per_block_cast_to_fp8`, `ceil_to_ue8m0`, `pack_ue8m0_to_int`) | MIT | `5081f85` | amax floor, scale rounding, exponent packing | `gemm/gemm_prep.cu`, `quantization/weight_repack.cu` |
| 128x4 block-scale swizzle | CUTLASS `Sm1xxBlockScaledConfig::SfAtom` layout as exposed by torchao `to_blocked` | BSD-3-Clause | not vendored | tile shape and in-tile order (formula only) | `gemm/gemm_prep.cu`, `tests/unit/test_gemm.cc` |
| SmoothQuant | SmoothQuant (MIT-HAN-Lab) | MIT | not vendored | scale formula, 1e-5 floors | `quantization/weight_repack.cu` |

GGUF types that are not quantized block formats (`F32`, `F16`, `BF16`, `Q8_1`) return
`kNotImplemented` from `gguf_dequantize` by design.

## Derived implementations (B5 decode GEMV, 2026-11-01)

Independent CUDA-core bodies; formats and semantics were read from the upstream source and
re-expressed, no upstream text was copied.

| Area | Derived from | License | Commit | What was taken | kernel-ops files |
|---|---|---|---|---|---|
| GGUF mmvq (10 block types x q8_1 activations) | llama.cpp `ggml/src/ggml-common.h` block structs and `ggml-cuda/{vecdotq.cuh,quantize.cu}` (first read through `3rd/sglang/sgl-kernel/csrc/quantization/gguf`) | MIT (ggml) / Apache-2.0 (sglang) | llama.cpp `0ec4027`; sglang `96a04cb` | block layouts, per-type unpacking order, `quantize_q8_1` rounding (`d = amax / 127`, `roundf(x / d)`, fp16 `d`); the weight side of the test oracle is B4's `gguf_dequantize`, itself checked against `dequantize_row_*` | `gemm/gguf_gemm.cu`, `tests/unit/test_gemm.cc` |
| W4A16 AWQ / GPTQ GEMV | AutoAWQ / llm-awq nibble order `{0,2,4,6,1,3,5,7}`; AutoGPTQ / GPTQModel `qweight [K/8, N]` layout, `g_idx` act-order | MIT | not vendored | packed-word layouts only; the zero-point convention (stored zero, no `-1`) is documented in the file header | `gemm/mixed_input_gemm.cu` |
| FP4 weights (MXFP4 / NVFP4) | OCP MX v1.0 (UE8M0, block 32) and NVIDIA NVFP4 (E4M3 per 16, fp32 global scale); 128x4 swizzle as in the B4 row above | open spec | n/a | e2m1 value table, scale encodings, swizzle offset | `gemm/fp4_gemm.cu` |

## Ported code

Unlike the rows above, this data was copied (mechanically) and is therefore redistributed under
the upstream license. The license text is in `third_party/licenses/`.

| Area | Ported from | License | Commit | What was copied | kernel-ops files | License text |
|---|---|---|---|---|---|---|
| GGUF codebooks for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S/IQ1_M | llama.cpp `ggml/src/ggml-common.h`: `iq2xxs_grid` (256 x u64), `iq2xs_grid` (512 x u64), `iq2s_grid` (1024 x u64), `iq3xxs_grid` (256 x u32), `iq3s_grid` (512 x u32), `iq1s_grid_gpu` (2048 x u32) | MIT, Copyright (c) 2023-2026 The ggml authors | `0ec4027` (checkout at `llama.cpp/`, repository root) | the numbers of the six tables, 4 608 words | `quantization/weight_repack.cu`, block marked `adapted-from-llama.cpp` | `third_party/licenses/llama.cpp.txt` |

How it was adapted: a script reads the numbers from `ggml-common.h` (nothing is typed by hand)
and writes them as `__device__ const` arrays with fixed-width literals. The script also checks
that the table sizes match, that `ksigns_iq2xs` equals "seven sign bits plus an even-parity bit"
for all 128 entries (so the kernel computes it instead of storing it), and that the nibbles of
`iq1s_grid_gpu` equal the int8 values of the CPU table `iq1s_grid` plus one for all 2048 entries
(so the CPU table is not copied). The decode logic is a re-expression of `dequantize_row_*`, not
a copy. The first lines of the block carry a SHA-256 prefix of each table.

Evidence that the port is exact: `dequantize_row_*` of llama.cpp `0ec4027` (`ggml-quants.c`,
gcc 13 with `-O2 -ffp-contract=off`) was run on 1024 pseudo-random blocks per type, which
reference every codebook entry of every table. The kops output is bit-identical for all 20
supported types, and `tests/unit/test_quantization.cc` pins the FNV-1a-64 hash of that output.

## Dev-only external baselines (B10)

`benchmark/compare/external/` is dev tooling: it is never built by CMake, ctest or the shipped
library. It contains one vendored SGLang header and one harness that fetches hpc-ops at the pinned
commit for a local comparison. Neither implementation is linked into `kops_cuda`.

| Area | Source | License | Commit | What is present | kernel-ops file |
|---|---|---|---|---|---|
| SGLang tree target-only baseline | `sgl-kernel/csrc/speculative/speculative_sampling.cuh` (`TreeSpeculativeSamplingTargetOnly`, FlashInfer/CUB based) | Apache-2.0, Copyright (c) 2025 SGLang team and 2024-2025 FlashInfer team | `96a04cb` (vendored header) | the header is copied verbatim under `benchmark/compare/external/sglang/`; the harness only calls it | `benchmark/compare/external/sglang/speculative_sampling.cuh`, `benchmark/compare/external/sglang_tree.cu` |
| hpc-ops single-draw baseline | `src/sampler/fused_sampler_temperature.cu` / `sampler.h` | MIT (Tencent) | `2a2e265` | fetched by `tools/dev/build_external_baselines.sh` into the gitignored `3rd/hpc-ops`; the sampler TU is compiled for SM89 with `HPC_TARGET_ARCH=90` (the source officially lists 90/100/103) | `benchmark/compare/external/hpc_sampler.cu` |

The SGLang kernel is target-only/threshold-based and hpc-ops has no speculative verifier; the
report labels both as partial external references, not token-for-token oracles.

## Vendor trees in this checkout

`3rd/` contains read-only reference checkouts used for comparison. They are **not** part of the
build and must not be linked. Their licenses remain their own; do not copy from them without
following the four steps above.

| Path | License |
|---|---|
| `3rd/vllm` | Apache-2.0 |
| `3rd/sglang` | Apache-2.0 |
| `3rd/DeepGEMM` | MIT |
| `3rd/FlashMLA` | MIT |
| `3rd/hpc-ops` | MIT (bundled CUTLASS under BSD-3-Clause) |
| `3rd/attention.rs` | MIT |
| `3rd/llm-awq` | MIT |
| `3rd/mistral.rs` | MIT |
| `3rd/TensorRT-LLM` | Apache-2.0 |
| `llama.cpp` (repository root, untracked, not under `3rd/`) | MIT |
