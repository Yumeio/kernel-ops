#!/usr/bin/env python3
"""FlashMLA adapter for the kernel-ops benchmark cases.

Dev-only, like `thirdparty.py`. Adapter contract:

    adapter(case: dict, device: torch.device) -> (fn, note) | (None, reason)

FlashMLA changed shape twice, so the adapter probes the installed build instead
of assuming one API:

  * `3rd/FlashMLA` (checkout 395cb3a, 2026.09.30 release) is DeepSeek-V4.1 only:
    DSA sparse decode (`flash_mla_with_kvcache` with `indices`, 528-byte FP8 /
    288-byte FP4 KV rows, head_dim 512, h_q in {64, 128}) and DSA sparse prefill
    (`flash_mla_sparse_fwd`, d_qk 512). Dense MLA decode and the real
    `get_mla_metadata` kernel were removed, and Hopper support was dropped.
  * builds up to 2025.09 keep the dense absorbed MLA decode: `k_cache` rows are
    576 bf16 or 656 B of FP8 (512 e4m3 + 64x2 bf16 rope + 16 scales), `h_q` 128,
    page 64/128, on SM90 or SM100.

Both formats are mapped, each behind an exact layout check (page size, record
width, head count). With the current case table every `bench_mla` case is
page=16 and 576-wide, which FlashMLA accepts only at page 64/128 and 512 wide,
so the mapping currently reports a skip for all of them - the reason string
names the mismatch instead of silently re-paging the cache.

    python benchmark/compare/thirdparty_flashmla.py
    python benchmark/compare/thirdparty_flashmla.py --cases out/thirdparty/cases/mla.json
    python benchmark/compare/thirdparty_flashmla.py --run --cases ... --json out/flashmla-mla.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys

try:
    import torch
except Exception:  # noqa: BLE001 - a missing torch must not break --cases
    torch = None

IMPLS = {"flashmla": None}

# Constants of the kernel-ops case tables (benchmark/bench_mla.cu, bench_sparse.cu).
NOPE = 512          # kv_lora_rank / head_dim_v
PE = 64             # rope part of the latent record
ROW = NOPE + PE     # 576: native bf16 latent record, in elements
FP8_ROW_BYTES = 656  # 512 e4m3 + 64 bf16 rope + 16 e4m3 scales (kMLAFp8DS)
MLA_HEADS = 128
SPARSE_QK = 576      # kops sparse MLA qk_dim
V4_FP8_ROW_BYTES = 528  # FlashMLA 2026.09.30 V4.1 format: 512 e4m3 + 16 ue8m0 scales

HINT = ("install FlashMLA from 3rd/FlashMLA (`pip install -e . --no-build-isolation`; "
        "SM100 for the 2026.09.30 DSA release, SM90/SM100 for builds up to 2025.09)")

RULES = (
    ("mla_decode.* (dense absorbed MLA, 576 bf16 / 656 B fp8 rows, page 64/128)",
     "flash_mla.flash_mla_with_kvcache (dense, builds <= 2025.09)"),
    ("mla_decode.*.topkN (V4.1 528 B fp8 rows, head_dim 512)",
     "flash_mla.flash_mla_with_kvcache (DSA sparse decode, 2026.09.30)"),
    ("mla_metadata.*", "flash_mla.get_mla_metadata (real kernel only in builds <= 2025.09)"),
    ("sparse_mla.bf16.h128.topk*", "none: kops qk_dim is 576, flash_mla_sparse_fwd instantiates 512"),
    ("lightning_indexer.*", "none: FlashMLA has no indexer kernel"),
)


def _id_int(case_id: str, key: str, default: int) -> int:
    marker = "." + key
    if marker not in case_id:
        return default
    digits = ""
    for ch in case_id.split(marker, 1)[1]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else default


def _device_cc(device) -> tuple:
    if torch is None or not torch.cuda.is_available():
        return (0, 0)
    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()
    return tuple(torch.cuda.get_device_capability(index))


def _parse_kv(case_id: str) -> int:
    """`kv64k` / `kv256k` / `kv1m` -> token count."""
    m = re.search(r"\.kv(\d+)([km])", case_id)
    if m is None:
        return 0
    scale = 1024 if m.group(2) == "k" else 1024 * 1024
    return int(m.group(1)) * scale


def _parse_decode(case_id: str) -> dict:
    """`mla_decode.{bf16|fp8}.kv64k.q{1|4}[.sN]` -> decode shape."""
    parts = case_id.split(".")
    fp8 = len(parts) > 1 and parts[1] == "fp8"
    return {
        "kv": _parse_kv(case_id),
        "q_len": _id_int(case_id, "q", 1),
        "splits": _id_int(case_id, "s", 1),
        "fp8": fp8,
        "page": _id_int(case_id, "p", 16),  # the current table always uses 16
    }


def decide(case: dict) -> dict:
    """Pure mapping decision: no allocation, no package import."""
    op = case.get("op", "")
    cid = case.get("id", "")

    if op == "mla_decode":
        d = _parse_decode(cid)
        if d["kv"] <= 0:
            return {"api": None, "note": "unparsed mla_decode case id"}
        if d["page"] not in (64, 128):
            return {"api": None, "note":
                    f"FlashMLA decode blocks are 64 (or 128) tokens; this case is page={d['page']} "
                    "(re-paging would change the cache layout, not just the launch)"}
        cols = int(case["cols"])
        if cols == ROW:
            return {"api": "mla_decode_dense", **d,
                    "note": "flash_mla.flash_mla_with_kvcache (dense absorbed MLA; needs a build <= 2025.09)"}
        if cols == NOPE:
            topk = _id_int(cid, "topk", 0)
            if topk <= 0:
                return {"api": None, "note":
                        "a 512-wide latent cache is decoded through DSA sparse attention, which needs the "
                        "case to name its top-k (`topkNN` in the id)"}
            return {"api": "mla_decode_dsa", **d, "topk": topk,
                    "note": "flash_mla.flash_mla_with_kvcache (DSA sparse decode, V4.1 528 B rows)"}
        return {"api": None, "note":
                f"head_dim {cols} does not match a FlashMLA decode format (dense {ROW}, V4.1 {NOPE})"}

    if op == "mla_metadata":
        return {"api": None, "note":
                "get_mla_metadata is a placeholder in the 2026.09.30 build (the split schedule is built "
                "inside the first flash_mla_with_kvcache call), and timing a no-op would be meaningless"}

    if op == "sparse_mla":
        heads = _id_int(cid, "h", MLA_HEADS)
        topk = _id_int(cid, "topk", 0)
        if heads <= 0 or topk <= 0:
            return {"api": None, "note": "unparsed sparse_mla case id"}
        # bench_sparse.cu fixes qk_dim at 576 (512 nope + 64 rope) and head_dim_v at 512;
        # flash_mla_sparse_fwd instantiates d_qk = 512 only, so the record width differs.
        return {"api": None, "note":
                f"flash_mla_sparse_fwd instantiates d_qk = {NOPE}; these cases use qk_dim {SPARSE_QK} "
                f"({NOPE} nope + {PE} rope), so the kernel would read a different record"}

    if op == "lightning_indexer":
        return {"api": None, "note":
                "FlashMLA has no lightning-indexer kernel; the indexer scoring kernel is DeepGEMM's "
                "fp8_fp4_mqa_logits, and this case is bf16 so that one does not match either"}

    return {"api": None, "note": f"no FlashMLA expression for op '{op}'"}


def _load():
    import flash_mla  # noqa: F401  (loads the C++ extension through torch.ops.load_library)
    return flash_mla


def _api_kind(flash_mla) -> str:
    """'dsa' for the 2026.09.30 sparse-only build, 'legacy' for the dense-era build."""
    fn = getattr(flash_mla, "flash_mla_with_kvcache", None)
    if fn is None:
        return "missing"
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "unknown"
    # The sparse build added `indices`; the dense build has no such parameter.
    return "dsa" if "indices" in params else "legacy"


def _build_dsa_decode(dec: dict, flash_mla, device):
    """V4.1 sparse decode: 528-byte rows, head_dim 512, explicit top-k indices."""
    page = dec["page"]
    kv = dec["kv"]
    seq = dec["q_len"]
    topk = dec["topk"]
    npages = (kv + page - 1) // page
    q = torch.randn(1, seq, MLA_HEADS, NOPE, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(npages, page, 1, V4_FP8_ROW_BYTES, device=device, dtype=torch.uint8)
    seq_lens = torch.full((1,), kv, device=device, dtype=torch.int32)
    # indices_in_kvcache = page_block * page_block_size + offset within the page;
    # the synthetic cache has an identity block table, so token t maps to itself.
    tokens = torch.arange(topk, device=device, dtype=torch.int32) % kv
    idx = ((tokens // page) * page + (tokens % page))
    idx = idx.view(1, 1, topk).expand(1, seq, topk).contiguous()
    sched, _ = flash_mla.get_mla_metadata()
    scale = 1.0 / (NOPE ** 0.5)

    def run():
        flash_mla.flash_mla_with_kvcache(
            q, k_cache, None, seq_lens, NOPE, sched, None,
            softmax_scale=scale, causal=False, is_fp8_kvcache=True, indices=idx)
    return run, dec["note"]


def _build_dense_decode(dec: dict, flash_mla, device):
    """Dense absorbed MLA decode of the 2025.09-era API (576 bf16 or 656 B fp8 rows)."""
    page = dec["page"]
    kv = dec["kv"]
    seq = dec["q_len"]
    b = 1
    npages = (kv + page - 1) // page
    q = torch.randn(b, seq, MLA_HEADS, ROW, device=device, dtype=torch.bfloat16)
    if dec["fp8"]:
        k_cache = torch.zeros(npages, page, 1, FP8_ROW_BYTES, device=device, dtype=torch.uint8)
    else:
        k_cache = torch.randn(npages, page, 1, ROW, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(npages, device=device, dtype=torch.int32).view(b, npages)
    seq_lens = torch.full((b,), kv, device=device, dtype=torch.int32)
    # The dense-era metadata kernel is a real kernel; the split count it returns is
    # FlashMLA's own schedule, so the kops `.sN` split factor is not forced.
    sched, num_splits = flash_mla.get_mla_metadata(seq_lens, MLA_HEADS, 1)
    scale = 1.0 / (ROW ** 0.5)

    def run():
        flash_mla.flash_mla_with_kvcache(
            q, k_cache, block_table, seq_lens, NOPE, sched, num_splits,
            softmax_scale=scale, causal=True, is_fp8_kvcache=dec["fp8"])
    return run, dec["note"]


def _build(dec: dict, flash_mla, device, cc: tuple):
    api = dec["api"]
    if torch is None:
        return None, "torch is required to build FlashMLA operands"

    kind = _api_kind(flash_mla)
    if api == "mla_decode_dsa":
        if kind != "dsa":
            return None, "the installed FlashMLA has no DSA decode (needs the 2026.09.30 build)"
        if cc[0] < 10:
            return None, "the DSA build supports SM100/SM103 only (Hopper support was removed)"
        return _build_dsa_decode(dec, flash_mla, device)
    if api == "mla_decode_dense":
        if kind != "legacy":
            return None, ("the installed FlashMLA is the 2026.09.30 sparse-only build: dense MLA decode and "
                          "the 656-byte KV format were removed; pin a build <= 2025.09 for this case")
        return _build_dense_decode(dec, flash_mla, device)
    return None, f"internal: no builder for '{api}'"


def adapter(case: dict, device):
    """thirdparty.py adapter entry point."""
    dec = decide(case)
    if dec["api"] is None:
        return None, dec["note"]

    cc = _device_cc(device)
    if cc[0] not in (9, 10):
        return None, f"flash_mla requires SM90 (dense-era builds) or SM100 (DSA); device is sm{cc[0]}{cc[1]}"

    try:
        flash_mla = _load()
    except Exception as exc:  # ImportError, missing .so, no torch_npu, ...
        return None, f"flash_mla not importable: {type(exc).__name__}: {exc}; {HINT}"

    try:
        return _build(dec, flash_mla, device, cc)
    except Exception as exc:  # noqa: BLE001 - capability probe: unsupported shape/dtype
        return None, f"flash_mla build: {type(exc).__name__}: {exc}"


IMPLS["flashmla"] = adapter


# ---------------------------------------------------------------------------
# CLI: the case-id -> API mapping, with or without FlashMLA installed
# ---------------------------------------------------------------------------
def print_rules() -> None:
    print("FlashMLA adapter rules (exact match only; everything else is skipped)")
    print(f"  hardware: SM90 for dense-era builds, SM100/SM103 for the 2026.09.30 DSA build ({HINT})")
    print("")
    for pattern, api in RULES:
        print(f"  {pattern:<74} -> {api}")
    print("")


def describe(case: dict) -> None:
    dec = decide(case)
    tag = "SKIP " if dec["api"] is None else "MAP  "
    api = "-" if dec["api"] is None else dec["api"]
    print(f"  {tag}{case.get('id', '?'):<48} {api:<20} {dec['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="case JSON from `bench_<family> --dump-cases`")
    ap.add_argument("--run", action="store_true", help="benchmark the mapped cases (needs flash_mla)")
    ap.add_argument("--json", default=None, help="result JSON for --run")
    ap.add_argument("--flush-l2", type=int, default=0, metavar="MB")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    print_rules()
    cases = []
    if args.cases:
        with open(args.cases, "r", encoding="utf-8") as f:
            cases = json.load(f).get("cases", [])
        print(f"case mapping ({args.cases})")
        for case in cases:
            describe(case)
        mapped = sum(1 for c in cases if decide(c)["api"] is not None)
        print(f"\n{mapped}/{len(cases)} cases have a FlashMLA mapping")

    if not args.run:
        return 0
    if not cases:
        print("--run needs --cases", file=sys.stderr)
        return 2
    if not args.json:
        print("--run needs --json", file=sys.stderr)
        return 2
    if torch is None or not torch.cuda.is_available():
        print("--run needs torch with CUDA", file=sys.stderr)
        return 1

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from thirdparty import bench  # noqa: E402 - dev harness, imported only for --run

    device = torch.device("cuda", 0)
    results, skipped = [], []
    for case in cases:
        try:
            fn, note = adapter(case, device)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": case["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if fn is None:
            skipped.append({"id": case["id"], "reason": note or "unsupported"})
            continue
        try:
            st = bench(fn, int(case["warmup"]), args.iters or int(case["iters"]),
                       args.flush_l2, args.rounds)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": case["id"], "reason": f"run: {type(exc).__name__}: {exc}"})
            continue
        ms = st["median_ms"]
        results.append({
            "id": case["id"], "op": case["op"], "dtype": case["dtype"],
            "rows": case["rows"], "cols": case["cols"], "bytes": case["bytes"],
            "flops": case["flops"], **st,
            "gbs": case["bytes"] / (ms * 1e6) if ms > 0 else 0.0,
            "tflops": case["flops"] / (ms * 1e9) if ms > 0 else 0.0,
            "impl": "flashmla", "note": note,
        })
        print(f"{case['id']:<48} {ms:10.4f} ms  {results[-1]['gbs']:9.1f} GB/s  {note}")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": "flashmla (thirdparty_flashmla.py)",
        "provenance": {"commit": "thirdparty_flashmla", "dirty": 1, "flags_hash": "",
                       "archs": f"sm{props.major}{props.minor}", "toolkit": torch.version.cuda or ""},
        "device": {"name": torch.cuda.get_device_name(0), "cc": f"{props.major}.{props.minor}",
                   "sms": props.multi_processor_count, "driver": 0, "runtime": 0, "peak_gbs": 0.0},
        "skipped": skipped,
        "results": results,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.json}: {len(results)} results, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
