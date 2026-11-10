# MLA / DSA / V4 / K3 parity notes

Status: implementation in progress. This document records the upstream source of
each kernel, the exact convention differences, and where the oracle/golden vector
comes from. It is not evidence by itself; the registry row points at the C++ test
and benchmark JSON.

## Pinned upstream sources

| Project | Commit | Used for |
|---|---|---|
| FlashMLA | `395cb3a` | V4.1 KV formats, sparse decode/prefill, fused Q-norm/RoPE/O-RoPE/cast |
| FlashMLA | `ba89a34` | V3.2/V4.0 KV formats, dense split-KV MLA, sparse SM90 |
| DeepGEMM | `5081f85` | SM90/SM100 MQA logits, Mega mHC, Mega Gate |
| vLLM | `07aea9d` | DeepSeek-V4/V4.1 model semantics, Kimi-K3 AttnRes kernel |
| hpc-ops | local clone | cross-checks for task maps, top-k, iHC |

Read-only clones live in `3rd/`. Ported code must be recorded in
`THIRD_PARTY_NOTICES.md`; the source-study template is
`docs/SOURCE_STUDY_TEMPLATE.md`.

## Convention map

| Topic | Upstream | kernel-ops | Test conversion |
|---|---|---|---|
| Attention LSE | FlashMLA uses base-2 internally; dense combine may use log2 | natural-log fp32, `lse = m + log(d)` | multiply/divide by `ln 2` at the test boundary only |
| Sink | FlashMLA sparse decode scales output by `1/(1+exp(sink-lse))` | sink is one extra term in the softmax denominator; LSE includes it | compare `lse` and `o` after converting the upstream convention |
| Split-KV | FlashMLA scheduler emits `DecodingSchedMeta` and cumulative `num_splits` | `mla_get_metadata` keeps the same metadata layout; T1 split uses the same cumulative counts | exact metadata bytes; partial rows in split order |
| DSA indexer | DeepGEMM returns compressed `[T, max_seqlen_k]` starting at column 0 | `row_starts/row_ends` select the visible range; outside = `-inf` | map compressed columns to the row range |
| Sparse MLA | FlashMLA `topk_length` and two scopes (`kv`, `extra_kv`) | same two scopes and per-query `topk_length`; `-1` skipped | gather valid entries and call `ref::attention_f64` per row |
| KV V3.2 656 B | 512 e4m3 NoPE (4x128), 4 fp32 tile scales, 64 bf16 RoPE | same | byte-exact `quantize_k_cache` at `ba89a34` |
| KV V4.0 584 B | 448 e4m3 NoPE (7x64) + 64 bf16 RoPE; 7 UE8M0 + 1 pad per page-token, out of record | same, `k_scale_pages` | byte-exact at `ba89a34` |
| KV V4.1 528 B | 512 e4m3 including RoPE; 16 UE8M0 inline (block 32) | same | byte-exact at `395cb3a` |
| KV V4.1 FP4 288 B | 256 B e2m1 (even low nibble), 32 e4m3 inline (block 16) | same, extra_kv only | byte-exact at `395cb3a` |
| mHC | DeepGEMM `mhc_reference` uses FP32 projection (TF32 off) | TF32 GEMM + `kEpiRowSumSq`; T1 generic fp32 | tolerance from CONTRACTS, documented per row |
| mHC Sinkhorn | softmax(row) + eps, alternating column/row normalisation | same, fixed iteration order | exact iteration order, tolerance |
| K3 AttnRes | vLLM kernel H=7168, up to 8 blocks, SM100 tcgen05/TMEM | T1 generic CUDA-core; Sm100a port of the same kernel | pure-Python/fp64 golden; batch-invariance |
| Gated MLA (K3) | gate projection epilogue `kEpiSigmoidGateMul` | optional `output_gate` on dense MLA | sigmoid gate in fp32, output dtype rounding |

## Per-kernel checklist

| Kernel | T1 | SM90a | SM100a | Oracle / golden | Status |
|---|---|---|---|---|---|
| `mla_get_metadata` | generic | generic | generic | host port of `get_decoding_sched_meta` | in progress |
| `mla_decode` | `mma.sync` | FlashMLA splitkv | vLLM CUTLASS SM100 | `ref::attention_f64` + 1M golden | in progress |
| `machete_prepack` | generic | generic | generic | vLLM layout bytes | in progress |
| `grouped_gemm_prepare` | generic | generic + TMA patch | generic + TMA patch | host task-map reference | in progress |
| `lightning_indexer` | `mma.sync` | DeepGEMM SM90 | DeepGEMM SM100 | `ref_mqa_logits`, exact top-k | in progress |
| `sparse_topk` | generic | generic | generic | host exact top-k | in progress |
| `sparse_mla` | `mma.sync` | FlashMLA sparse | FlashMLA sparse + fused V4.1 | `ref_sparse_attn_*`, `attention_f64` | in progress |
| `kv_compressor` | CUDA core | CUDA core | CUDA core | V4 compressor formula golden | in progress |
| `compressed_attention` | `mma.sync` | optional body | V4 SM100 body | two-scope `attention_f64` | in progress |
| `hc_expand/head/pre/post/coeff` | CUDA core | CUDA core | CUDA core | DeepGEMM `mhc_reference` | in progress |
| `mega_mhc` | generic split-K | generic | DeepGEMM `sm100_mega_mhc` | `mhc_reference` | in progress |
| `attn_res/commit` | generic | generic | vLLM K3 port | K3 formula golden | in progress |

## 1M context test policy

- Dense MLA: 64k/128k compare directly with `ref::attention_f64`; 256k/512k/1M
  load output-only golden from `KOPS_GOLDEN_DIR`. The KV cache is regenerated
  from the manifest seed; no 656 MB cache is committed.
- Sparse MLA: gather the valid top-k entries and call `ref::attention_f64` with
  one request per query row. For 1M indexer runs, only the selected top-k rows
  reach the attention oracle.
- Handoff kits contain golden output/LSE files and the manifest; the target only
  runs `ctest`/benchmarks, it does not regenerate or download data.
