# Golden vectors

Golden vectors are the repo's offline oracle for layer conformance and for
kernels whose expected result is too expensive to recompute in the test. They
are plain files plus a JSON manifest: no Python, no torch, no network and no
input cache is required to use them.

A test that needs golden data reads `KOPS_GOLDEN_DIR`, which points at the
directory containing `manifest.json` (normally `tests/golden/`, or
`<kit>/golden/` inside a handoff kit). If the directory or manifest is absent,
the test returns `77` so CTest reports **Skipped** instead of failing. A test
must never fall back to downloading data or to a PyTorch reference.

## Directory layout

```text
tests/golden/
  README.md                 this document
  manifest.json             committed index of all cases
  manifest.schema.json      draft-07 schema for manifest.json
  cases/<case-id>.kopsgold  committed output-only golden file (small cases)
```

`cases/` is flat. The manifest path is relative to `KOPS_GOLDEN_DIR` and may not
contain `..`, an absolute path or a subdirectory. Inputs are **not** stored in
`cases/`. A generated input cache, when a test chooses to make one, goes under
`${KOPS_GOLDEN_CACHE:-${TMPDIR:-/tmp}/kops-golden-cache}/<case-id>/` and must
never be written into the repository.

Inside a handoff kit the same tree is `<kit>/golden/`; `run_all.sh` exports
`KOPS_GOLDEN_DIR=<kit>/golden` automatically.

## Two storage modes

| Mode | Manifest | Golden payload | Input |
| --- | --- | --- | --- |
| `on_demand_input` | `"on_demand_input": true` | output, plus LSE when `has_lse` | Deterministically regenerated from `seed` by the test. The test may cache it locally under `KOPS_GOLDEN_CACHE`; the cache is disposable. |
| `stored_output` | `"on_demand_input": false` | output, plus LSE when `has_lse` | Supplied by the caller. It is not part of the golden file. |

Both modes are **output-only** on disk. The seed, dtype, shapes, codec, topk and
sink in the manifest fully describe the input to an `on_demand_input` case.

Large 1M-token cases are always `on_demand_input` and output-only. A 1M-token KV
cache (for example a 656-byte DeepSeek-MLA record per token) is never committed,
never copied into a handoff kit and never checked into `cases/`. The KV cache is
regenerated on the target GPU from `seed`, or produced by the caller in
`stored_output` mode.

### Deterministic input generators

Every `on_demand_input` operation documents its generator. Unless the operation
says otherwise, use SplitMix64 with `seed` as the initial state and map each
64-bit output to a value in `[-1, 1)` before casting to the input dtype. Padding
and unused lanes are zero-filled. If the generator changes, add a new case id;
do not silently change the stream behind an existing id.

## Layer conformance vectors (P8)

`tests/unit/test_conformance.cc` composes the implemented families into the
ROADMAP §3.5 layers and compares every result against an fp64 oracle. The first
version keeps those expected values **in the test translation unit** and
computes them at runtime, because `csrc/reference/` has no `gemm`, `route`,
`permute`, `residual` or `sample` entry points yet:

| Source | Used for |
| --- | --- |
| `csrc/reference/reference.cc` | `rms_norm`, `attention`, `merge_states`, `delta_rule`, `softmax`, `linear_attention`, `ssm`, `causal_conv1d` |
| local fp64 scalar loops in the test | GEMM / GEMV, softmax+sigmoid routers, residual add, W4A16 dequant, permute/unpermute, the V4.1 record recipe |

The `layer*` cases in `manifest.json` are `planned` and reserve the ids and
tolerances those checks use. They carry `sha256: null` and no file: nothing is
committed for them. When a `.kopsgold` is produced later it must come from the
same fp64 oracle, carry the real hash, and switch `status` to `ready`; the
runtime comparison then becomes a hash/segment comparison against the file. The
`on_demand_input` flag is `true` because the test regenerates the inputs from
SplitMix64 streams (`0x1001..` for layer 1, `0x2001..` for layer 2, and so on).

The layer suite depends only on CUDA and the built `kops_cuda` library, never on
Python, PyTorch or the network. It returns `77` when the exact SM120a device is
absent and reports per-sub-layer `VERIFIED` / `GAP` / `DEFECT` / `FAIL` lines
plus a summary table; `--strict` turns known defects into failures. The whole run
takes about a second, so `--tier` is accepted but does not gate any case.

### Size budget

Keep a committed `.kopsgold` at or below 8 MiB, and all committed goldens at or
below 32 MiB. A case whose output tensor alone would exceed that budget stays
`planned` until a reduced shape or a future chunked-digest format exists. The
16 KB output of a 1M-token decode case is fine; a 1M x 7168 activation output is
not, and must be reduced or kept as a planned large-memory case.

## `KOPSGOLD` binary format

All integers are little-endian. Floats are IEEE-754 little-endian. A file is
self-describing enough to be validated without the manifest; the manifest adds
the whole-file SHA-256 and generation provenance.

### Header

| Offset | Size | Type | Field | Notes |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | `char[8]` | `magic` | Exactly `KOPSGOLD` (no NUL). |
| 8 | 2 | `u16` | `version` | `1` for this document. |
| 10 | 2 | `u16` | `header_bytes` | `96 + 8 * rank`. |
| 12 | 4 | `u32` | `dtype` | `0` fp32, `1` bf16, `2` fp16, `3` u8. |
| 16 | 4 | `u32` | `rank` | `0..8`; number of entries in `shape[]`. |
| 20 | 4 | `u32` | `flags` | Bit 0 `HAS_LSE`, bit 1 `HAS_SINK`; bits 2..31 zero. |
| 24 | 8 | `u64` | `seed` | Seed used for the deterministic input generator. |
| 32 | 4 | `u32` | `codec` | Numeric `KVCodec`, see the table below. |
| 36 | 4 | `u32` | `topk` | Sparse top-k; `0` means dense. |
| 40 | 4 | `f32` | `sink` | Attention sink logit; `0.0` when `HAS_SINK` is clear. |
| 44 | 4 | `u32` | `reserved0` | Must be zero. |
| 48 | 8 | `u64` | `output_offset` | Absolute file offset of the output tensor. |
| 56 | 8 | `u64` | `output_bytes` | `product(shape) * dtype_size`. |
| 64 | 8 | `u64` | `lse_offset` | Absolute offset of fp32 LSE, or `0`. |
| 72 | 8 | `u64` | `lse_bytes` | `product(shape[0:-1]) * 4`, or `0`. |
| 80 | 8 | `u64` | `meta_offset` | Absolute offset of optional UTF-8 JSON metadata, or `0`. |
| 88 | 8 | `u64` | `meta_bytes` | Metadata length in bytes, or `0`. |
| 96 | `8 * rank` | `u64[]` | `shape` | Row-major logical extents. |

`header_bytes` is exactly `96 + 8 * rank`, so the shape array is always
8-byte-aligned. The output and metadata sections may start immediately after the
header or at a later 256-byte-aligned offset; unused padding bytes must be zero.
The LSE section, when present, is fp32 and has the output shape with the last
dimension removed: for an output `[B, S, H, D]`, LSE is `[B, S, H]`; for
`[B*H, D]`, LSE is `[B*H]`. For `rank == 0`, LSE is one fp32 value.

The file may end after the last section or carry zero padding only. Section
ranges must not overlap, must lie inside the file, and must satisfy
`output_offset >= header_bytes` (likewise for LSE and metadata).

### `dtype` enum

| Value | Manifest name | Bytes per element |
| ---: | --- | ---: |
| 0 | `fp32` | 4 |
| 1 | `bf16` | 2 |
| 2 | `fp16` | 2 |
| 3 | `u8` | 1 |

Values 4 and above are reserved; append new dtypes, never renumber these.

### `codec` enum

The numeric value mirrors `kops::kv::KVCodec` in `csrc/kvcache/kv_layout.cuh` and
must not be renumbered. The manifest uses the stable string names.

| Value | Manifest name |
| ---: | --- |
| 0 | `native` |
| 1 | `fp8_per_tensor` |
| 2 | `fp8_per_head` |
| 3 | `fp8_per_token_head` |
| 4 | `int8_per_token_head` |
| 5 | `int4_per_token_head` |
| 6 | `nvfp4` |
| 7 | `mla_fp8_ds` |
| 8 | `mla_nvfp4_ds` |
| 9 | `lowbit_wht` |
| 10 | `indexer_fp8` |
| 11 | `indexer_fp4` |
| 12 | `mla_fp8_ds_fnuz` |
| 13 | `mla_fp8_ds_v40` |
| 14 | `mla_fp8_ds_v41` |
| 15 | `mla_fp8_ds_v41_fp4` |

`codec` describes the storage of the inputs; it does not change the output
dtype. For `stored_output` cases the codec is still recorded so the caller can
construct the same input representation.

### Optional metadata

When `meta_bytes > 0`, the bytes at `meta_offset` are a UTF-8 JSON object. It may
carry codec parameters (scales, page size, record size), the generator name, or
a human-readable note. The metadata is additive: a reader that does not know a
key must still be able to consume the output and LSE sections.

## Manifest

`manifest.json` is validated by `manifest.schema.json` (JSON Schema draft-07).
Each case records:

| Field | Meaning |
| --- | --- |
| `id` | Stable lowercase id; the file name derives from it. |
| `op` | kernel-ops operation name. |
| `dtype` | `fp32`, `bf16`, `fp16` or `u8`; must match the binary header. |
| `shapes` | Named logical shapes; `output` is required. |
| `codec` | `KVCodec` name from the table above. |
| `topk` | Sparse top-k, `0` for dense. |
| `sink` | Attention sink logit, `0.0` for none. |
| `seed` | Seed for the on-demand generator. |
| `golden` | Relative path under `cases/`. |
| `sha256` | Lowercase SHA-256 of the whole `.kopsgold` file; `null` only while `planned`. |
| `tolerance` | `atol`, `rtol`, optional `lse_atol` / `max_ulp`. |
| `tier` | `smoke`, `standard` or `full`. |
| `on_demand_input` | Storage mode selector, see above. |
| `status` | `planned` (no file/hash) or `ready` (real file and hash). |
| `has_lse` | Optional; true when an LSE section is present. |
| `notes` | Optional provenance or regeneration note. |

Comparison is `abs(got - want) <= atol + rtol * abs(want)`. LSE uses
`lse_atol` when present. A NaN compares equal only to a NaN; `+inf`/`-inf`
compare equal only to the same infinity. Attention LSE is fp32 natural-log
log-sum-exp; a fully masked row has zero output and `-inf` LSE. Sink logits are
part of the denominator and are not written into the output.

### `sha256` rules

- `ready` cases must have a 64-character lowercase hex `sha256` and a real file;
  the hash is over the entire file, including header, padding and metadata.
- `planned` cases must have `"sha256": null`. Do not invent a hash and do not
  commit a placeholder binary.
- The manifest itself is not hashed. `source_commit` records the clean commit
  that produced the ready files; it is `"pending"` while no ready case exists.

## Producing a case

1. Generate the output on the target (or with the fp64 reference for a small
   shape) from a documented deterministic input.
2. Write the `.kopsgold` file with the layout above; use `status: "planned"`
   while it is not yet generated.
3. Compute `sha256sum cases/<id>.kopsgold`, put the lowercase digest in the
   manifest, set `"status": "ready"`, and regenerate `generated_at` /
   `source_commit`.
4. Keep inputs out of the repo. For a 1M case, verify that the file contains
   output/LSE only and that its size is within the budget.
5. Validate the JSON and, when available, a ready file's header and hash with
   `tools/handoff/check_env.py`.

```bash
python3 -m json.tool tests/golden/manifest.schema.json >/dev/null
python3 -m json.tool tests/golden/manifest.json >/dev/null
KOPS_GOLDEN_DIR=tests/golden ctest --test-dir build/dev --output-on-failure
```

## Handoff kits

`tools/handoff/build_kit.sh` copies `tests/golden/` into the kit when it exists.
`run_all.sh` exports `KOPS_GOLDEN_DIR=<kit>/golden`, so tests find the manifest
without any machine-specific path. A kit built before a golden case is ready
still runs; the dependent test returns `77` and the kit's JSON summary records a
skip, not a pass.
