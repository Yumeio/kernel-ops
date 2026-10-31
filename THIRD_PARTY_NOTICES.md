# Third-party notices

kernel-ops is licensed under **Apache-2.0** (see `LICENSE`).

This file records, per area, which upstream project a contract was derived from and under which
license. It exists because `csrc/` is a **contract-first** codebase: most headers describe the
semantics of an existing production kernel (vLLM, SGLang, hpc-ops, FlashMLA, DeepGEMM,
attention.rs, TRT-LLM, CUTLASS) before any body is written here.

## Status

**No upstream source code has been copied into `csrc/` yet.** Every header is an independent
declaration; the "Reference" column below names the design source only. When implementations are
ported (M1 onward), each ported file MUST:

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
| attention.rs | MIT | low-bit WHT KV codecs (K8V4/K4V4/K3V4); paged attention structure; experimental 2-bit MoE | `attention/lowbit_kernels.cuh`, `kvcache/kv_layout.cuh` (`kLowBitWHT`), `moe/fused_moe_kernels.cuh` (M04) |
| TensorRT-LLM | Apache-2.0 | Lamport one-shot allreduce; allreduce fusion epilogue; TRT-LLM-gen shuffled weight layout; `kEpiSigmoidGateMul` | `communication/allreduce_kernels.cuh`, `common/quant.cuh` (`kTrtllmShuffled`), `gemm/gemm_params.cuh` |
| CUTLASS / CuTe | BSD-3-Clause | tile/cluster/stage vocabulary, `GemmTraits`, block-scaled MMA kinds (no code yet) | `gemm/gemm_params.cuh`, `docs/GEMM_MATRIX.md` |
| Local read-only reference checkout (`3rd/csrc`, commit `d99cb25`) | Apache-2.0 (same project lineage) | device-body and validation reference for the B0/B1 first slice (`residual`, `rope`, `embedding`, `sampling`, `merge`, `norm`); adapted to the `kops` host API, no framework types | `model_ops/residual.cu`, `position/rope.cu`, `embedding/embedding.cu`, `sampling/*.cu`, `attention/merge.cu`, `norm/norm.cu` |
| FlashAttention (Dao-AILab) | BSD-3-Clause | FA2 (sm80) / FA3 (sm90) / FA4 (sm100) mainloop structure | `attention/prefill_kernels.cuh`, `attention/decode_kernels.cuh` |
| ggml | MIT | GGUF block type numbering (`GGML_TYPE_*`) | `common/quant.cuh` (`GGUFType`) |
| llm-awq | MIT | AWQ packing and zero-point convention | `common/quant.cuh` (`kAWQ`), `quantization/weight_repack_kernels.cuh` |
| mistral.rs | MIT | hybrid cache (attention + recurrent) layout | `kvcache/kv_layout.cuh` (hybrid family, planned) |

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
