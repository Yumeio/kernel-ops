#!/usr/bin/env python3
"""Convert the llama.cpp GGUF MMVQ CSV into the shared benchmark JSON schema.

Dev-only. `benchmark/compare/llamacpp_mmvq.cpp` prints:

    type,n,k,m,median_ms,min_ms,GB/s,TFLOP/s
    q4_0,14336,4096,1,0.0963,0.0801,123.4,...

The C++ GGUF cases use ids of the form
`gemm.gguf_mmvq.<type>.m<m>.<n>x<k>`; this script maps a row to that id so
`compare_results.py` can diff it against bench_gemm.

    python benchmark/compare/llamacpp_csv.py --csv out/llamacpp.csv \
        --json out/llamacpp.json [--only gguf_mmvq] [--impl llamacpp]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys


def row_to_result(row: dict[str, str], impl: str) -> dict | None:
    try:
        typ = row["type"].strip()
        n = int(float(row["n"]))
        k = int(float(row["k"]))
        m = int(float(row["m"]))
        median_ms = float(row["median_ms"])
        min_ms = float(row.get("min_ms") or median_ms)
        gbs = float(row.get("GB/s") or 0.0)
        tflops = float(row.get("TFLOP/s") or 0.0)
    except (KeyError, ValueError):
        return None
    if median_ms <= 0:
        return None
    cid = f"gemm.gguf_mmvq.{typ}.m{m}.{n}x{k}"
    bytes_ = int(round(gbs * median_ms * 1e6))
    return {
        "id": cid,
        "op": f"gguf_mmvq_{typ}",
        "dtype": "bfloat16",
        "rows": m,
        "cols": n,
        "bytes": bytes_,
        "flops": int(round(tflops * median_ms * 1e9)),
        "median_ms": median_ms,
        "min_ms": min_ms,
        "p10_ms": median_ms,
        "p90_ms": median_ms,
        "p99_ms": median_ms,
        "gbs": gbs,
        "tflops": tflops,
        "impl": impl,
        "over_peak": False,
        "note": "llama.cpp ggml_mul_mat; hot-L2 (one flush per group, not per launch)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--impl", default="llamacpp")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    args = ap.parse_args()

    results, skipped = [], []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        # The harness prints `device: CUDA0` (and init lines) to stdout before
        # the CSV header, so start the reader at the real header row.
        lines = [ln for ln in f if ln.strip()]
        header = next((i for i, ln in enumerate(lines) if ln.startswith("type,")), None)
        if header is None:
            print("llamacpp_csv: no CSV header found", file=sys.stderr)
            return 2
        reader = csv.DictReader(lines[header:])
        for row in reader:
            r = row_to_result(row, args.impl)
            if r is None:
                skipped.append({"id": row.get("type", "?"), "reason": "malformed CSV row"})
                continue
            cid = r["id"]
            if args.only and not any(tok.strip() and tok.strip() in cid
                                     for tok in args.only.split(",")):
                continue
            if args.skip and any(tok.strip() and tok.strip() in cid
                                 for tok in args.skip.split(",")):
                continue
            results.append(r)

    out = {
        "schema": 1,
        "backend": "llama.cpp ggml_mul_mat (dev baseline)",
        "impl": args.impl,
        "results": results,
        "skipped": skipped,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"llamacpp_csv: {len(results)} results, {len(skipped)} skipped -> {args.json}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
