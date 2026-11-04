#!/usr/bin/env python3
"""Normalize the hpc-ops sampler benchmark JSONL (dev tool, M5).

hpc-ops only accepts V=120832 and clamps top-k to 32/64, so compare it with the
kernel-ops fused cases on the same V/batch and state the semantic differences in
the report. This script prints the hpc-ops rows and, optionally, the kernel-ops
fused p0.95/k50 rows from a --json file next to them.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def hpc_table(rows):
    keep = [r for r in rows if r.get("provider") == "hpc"]
    print(f"{'scene':>14} {'batch':>6} {'timing':>24} {'us':>10} {'us/row':>9}")
    for r in sorted(keep, key=lambda x: (x.get("scene", ""), x.get("batch", 0))):
        us = r.get("latency_us")
        if us is None:
            continue
        b = r.get("batch", 0) or 1
        print(
            f"{r.get('scene', ''):>14} {r.get('batch', 0):6d} "
            f"{r.get('timing', ''):>24} {us:10.3f} {us / b:9.3f}"
        )


def kops_table(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    print()
    print(f"{'case':>52} {'median_ms':>10} {'p99_ms':>10} {'GB/s':>8}")
    for r in data.get("results", []):
        cid = r.get("id", "")
        if "sampling.fused." not in cid or "v120832" not in cid:
            continue
        print(
            f"{cid:>52} {r.get('median_ms', 0.0):10.4f} "
            f"{r.get('p99_ms', 0.0):10.4f} {r.get('gbs', 0.0):8.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="hpc-ops --jsonl output")
    ap.add_argument("--kops", default="", help="kernel-ops bench_sampling --json output")
    args = ap.parse_args()
    hpc_table(read_jsonl(Path(args.jsonl)))
    if args.kops:
        kops_table(Path(args.kops))


if __name__ == "__main__":
    main()
