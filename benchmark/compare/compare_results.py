#!/usr/bin/env python3
"""Compare kernel-ops results with one or more baseline JSON files.

    python benchmark/compare/compare_results.py out/kops-norm.json \
        out/torch-norm.json out/vllm-norm.json out/flashinfer-norm.json

For every case the best baseline is the smallest median. `speedup` is
best_baseline_median / kops_median, so > 1 means kernel-ops is faster. The
script also reports the kernel-ops and baseline share of the measured memory
roofline when the JSON carries `device.peak_gbs`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec, {r["id"]: r for r in spec.get("results", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kops", help="results JSON from benchmark/bench_<family>")
    ap.add_argument("refs", nargs="+", help="one or more baseline JSON files")
    ap.add_argument("--names", nargs="+", help="display names for refs (default: file stem)")
    ap.add_argument("--only", nargs="+", metavar="PREFIX")
    ap.add_argument("--json", help="write the per-case comparison as JSON")
    args = ap.parse_args()

    kops_spec, kops = load(args.kops)
    refs = []
    names = args.names or [os.path.splitext(os.path.basename(p))[0] for p in args.refs]
    if len(names) != len(args.refs):
        print("--names must match --refs", file=sys.stderr)
        return 2
    for path, name in zip(args.refs, names):
        spec, rows = load(path)
        refs.append((name, spec, rows))

    ids = sorted(set(kops) | {i for _, _, rows in refs for i in rows})
    if args.only:
        ids = [i for i in ids if i.startswith(tuple(args.only))]

    header = f"{'case':<44} {'kops ms':>9} {'best ref':>16} {'ratio':>7} {'kops GB/s':>10} {'best GB/s':>10}"
    print(header)
    print("-" * len(header))
    rows_out, ratios = [], []
    peak = float(kops_spec.get("device", {}).get("peak_gbs", 0.0) or 0.0)
    for case_id in ids:
        k = kops.get(case_id)
        if k is None:
            print(f"{case_id:<44} missing in kops")
            continue
        best = None
        for name, spec, rows in refs:
            r = rows.get(case_id)
            if r is None:
                continue
            if best is None or r["median_ms"] < best[1]["median_ms"]:
                best = (name, r)
        kms = float(k["median_ms"])
        if best is None:
            print(f"{case_id:<44} {kms:9.4f} {'(no ref)':>16}")
            rows_out.append({"id": case_id, "kops_ms": kms, "ref": None,
                             "ratio": None, "kops_gbs": k.get("gbs"),
                             "best_gbs": None})
            continue
        name, r = best
        rms = float(r["median_ms"])
        ratio = rms / kms if kms > 0 else 0.0
        ratios.append(ratio)
        kgbs = float(k.get("gbs", 0.0) or 0.0)
        rgbs = float(r.get("gbs", 0.0) or 0.0)
        flag = ""
        if peak > 0.0 and max(kgbs, rgbs) > peak * 1.05:
            flag = " !byte-count"
        print(f"{case_id:<44} {kms:9.4f} {name:>9}:{rms:6.3f} {ratio:7.2f} "
              f"{kgbs:10.1f} {rgbs:10.1f}{flag}")
        rows_out.append({"id": case_id, "kops_ms": kms, "ref": name, "ref_ms": rms,
                         "ratio": ratio, "kops_gbs": kgbs, "ref_gbs": rgbs,
                         "ref_note": r.get("note")})

    if ratios:
        gm = math.exp(sum(math.log(max(r, 1e-12)) for r in ratios) / len(ratios))
        wins = sum(1 for r in ratios if r < 1.0)
        print(f"\ngeometric mean speedup: {gm:.2f}x over {len(ratios)} cases")
        print(f"kops slower than best ref: {wins}/{len(ratios)} cases")
        print("speedup = best baseline median / kops median (>1 means kernel-ops is faster)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"kops": args.kops, "refs": args.refs, "peak_gbs": peak,
                       "geomean": (math.exp(sum(math.log(max(r, 1e-12)) for r in ratios) /
                                            len(ratios)) if ratios else None),
                       "cases": rows_out}, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
