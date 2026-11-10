#!/usr/bin/env python3
"""Render the kernel-ops benchmark tables from the result JSONs.

Dev-only. Reads one kops JSON per family and any number of baseline JSONs
(torch, vLLM, FlashInfer, SDPA) and emits markdown with the best baseline per
case, the speedup ratio and the share of the measured roofline.
"""
from __future__ import annotations

import argparse
import json
import math
import os


FAMILIES = ["model_ops", "norm", "activation", "position", "embedding", "chains",
            "kvcache", "quantization", "gemm", "moe", "sampling", "merge"]


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d, {r["id"]: r for r in d.get("results", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kops-dir", default="out/bench/kops")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    refs = []
    for item in args.ref:
        name, _, path = item.partition("=")
        spec, rows = load(path)
        refs.append((name, spec, rows))

    lines = []
    for family in FAMILIES:
        kpath = os.path.join(args.kops_dir, family + ".json")
        if not os.path.exists(kpath):
            continue
        kspec, kops = load(kpath)
        peak = float(kspec.get("device", {}).get("peak_gbs", 0.0) or 0.0)
        lines.append(f"\n### {family}\n")
        lines.append("| case | kops ms | best baseline | ratio | kops GB/s | best GB/s | note |")
        lines.append("|---|---:|---|---:|---:|---:|---|")
        ratios = []
        rows = sorted(kops.values(), key=lambda r: r["id"])
        for r in rows:
            best = None
            for name, spec, ref_rows in refs:
                rr = ref_rows.get(r["id"])
                if rr is not None and (best is None or rr["median_ms"] < best[1]["median_ms"]):
                    best = (name, rr)
            kms = float(r["median_ms"])
            if best is None:
                lines.append(f"| `{r['id']}` | {kms:.4f} | — | — | {r['gbs']:.1f} | — | no baseline |")
                continue
            name, rr = best
            rms = float(rr["median_ms"])
            ratio = rms / kms if kms > 0 else 0.0
            ratios.append(ratio)
            note = rr.get("note", "")
            if peak > 0 and max(r["gbs"], rr.get("gbs", 0.0)) > peak * 1.05:
                note = (note + "; byte-count suspect").strip("; ")
            lines.append(f"| `{r['id']}` | {kms:.4f} | {name} {rms:.4f} | {ratio:.2f}× | "
                         f"{r['gbs']:.1f} | {rr.get('gbs', 0.0):.1f} | {note} |")
        if ratios:
            gm = math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios))
            lines.append(f"\n**{family}: geomean {gm:.2f}× over {len(ratios)} cases; "
                         f"kops slower in {sum(1 for x in ratios if x < 1.0)} cases.**\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
