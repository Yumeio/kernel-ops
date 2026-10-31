#!/usr/bin/env python3
"""Diff a kernel-ops benchmark JSON against a PyTorch baseline JSON."""
from __future__ import annotations

import argparse
import json
import math


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["results"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kops", help="results JSON from benchmark/bench_<family>")
    ap.add_argument("torch", help="results JSON from torch_baseline.py")
    args = ap.parse_args()

    kops = load(args.kops)
    torch_ref = load(args.torch)
    ids = sorted(set(kops) | set(torch_ref))

    print(f"{'case':<42} {'kops ms':>10} {'torch ms':>10} {'speedup':>8} "
          f"{'kops GB/s':>10} {'torch GB/s':>11}")
    print("-" * 96)
    ratios = []
    for case_id in ids:
        k = kops.get(case_id)
        t = torch_ref.get(case_id)
        if k is None or t is None:
            side = "kops" if k is None else "torch"
            print(f"{case_id:<42} missing in {side}")
            continue
        kms = k["median_ms"]
        tms = t["median_ms"]
        speedup = tms / kms if kms > 0 else 0.0
        if speedup > 0:
            ratios.append(speedup)
        print(f"{case_id:<42} {kms:10.4f} {tms:10.4f} {speedup:8.2f}x "
              f"{k['gbs']:10.1f} {t['gbs']:11.1f}")
    if ratios:
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"\ngeometric mean speedup: {gm:.2f}x over {len(ratios)} cases")
    print("speedup = torch median / kops median (>1 means kernel-ops is faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
