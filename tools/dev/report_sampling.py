#!/usr/bin/env python3
"""Render a sampling comparison as the two tables the sampler claim needs.

The benchmark matrix mixes three things that must not share a geomean:
`sampling.fused.*` (the one-launch sampler), `sampling.sample.*` (the public
entry point) and the individual logits ops (softmax, argmax, bincount, ...).
This script reads the JSON written by `compare_results.py --groups 2` and
prints (or writes) a markdown report with the per-path summary, the per-mode
breakdown of both sampler paths, and every case that is slower than eager.

    python benchmark/compare/compare_results.py out/kops-sampling.json \
        out/torch-sampling.json --groups 2 --json out/compare-sampling.json
    python tools/dev/report_sampling.py out/compare-sampling.json \
        --write out/report-sampling.md

Dev-only: never built, linked, tested or shipped.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def geomean(values):
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


def mode_of(case_id: str, path: str) -> str:
    # sampling.<path>.<mode>.<dtype>... ; the mode itself may contain dots (p0.95).
    parts = case_id.split(".")
    if len(parts) < 4 or parts[1] != path:
        return "other"
    rest = parts[2:]
    for i, token in enumerate(rest):
        if token in ("bf16", "fp32", "fp16"):
            return ".".join(rest[:i])
    return ".".join(rest)


def table(title: str, rows) -> list[str]:
    out = [f"### {title}", "", "| case group | cases | slower | geomean | worst |",
           "|---|---:|---:|---:|---:|"]
    for row in rows:
        out.append(f"| `{row['group']}` | {row['cases']} | {row['slower']} | "
                   f"{row['geomean']:.2f}x | {row['worst']:.2f}x |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare", help="compare_results.py --json output")
    ap.add_argument("--write", help="write markdown here instead of stdout")
    args = ap.parse_args()

    data = json.loads(Path(args.compare).read_text(encoding="utf-8"))
    cases = [c for c in data.get("cases", []) if c.get("ratio") is not None]
    groups = data.get("groups") or []
    if not groups:
        raise SystemExit("compare JSON has no 'groups'; rerun compare_results.py with --groups 2")

    by_group = {g["group"]: g for g in groups}
    overall = data.get("geomean")
    overall_text = f"{overall:.2f}x" if overall is not None else "n/a"
    lines = ["# Sampling benchmark", "",
             f"- kernel-ops: `{data.get('kops')}`",
             f"- references: {', '.join(f'`{r}`' for r in data.get('refs', []))}",
             f"- cases with a reference: {len(cases)}",
             f"- overall geomean: {overall_text}", ""]

    sampler = [by_group[k] for k in ("sampling.fused", "sampling.sample") if k in by_group]
    if sampler:
        fused = by_group.get("sampling.fused")
        sample = by_group.get("sampling.sample")
        lines += ["## Two sampler paths, reported separately", ""]
        if fused:
            lines.append(f"- `sampling.fused.*`: **{fused['geomean']:.2f}x** over {fused['cases']} "
                         f"cases, {fused['slower']} slower than eager.")
        if sample:
            lines.append(f"- `sampling.sample.*`: **{sample['geomean']:.2f}x** over {sample['cases']} "
                         f"cases, {sample['slower']} slower than eager.")
        lines += ["", "Only the `sampling.fused` row is a fused-sampler claim. "
                      "The aggregate mixes 15x-traffic legacy rows and 200-2000x "
                      "small logits ops and must not be quoted as one number.", ""]
        lines += table("Per-group summary", sorted(groups, key=lambda g: g["group"]))
        lines += table("Sampler paths", sampler)

    for path in ("fused", "sample"):
        rows = [c for c in cases if f".{path}." in c["id"]]
        if not rows:
            continue
        modes = {}
        for c in rows:
            modes.setdefault(mode_of(c["id"], path), []).append(c["ratio"])
        mode_rows = [{"group": f"sampling.{path}.{m}", "cases": len(v),
                      "slower": sum(1 for r in v if r < 1.0),
                      "geomean": geomean(v), "worst": min(v)}
                     for m, v in sorted(modes.items())]
        lines += table(f"`{path}` by mode", mode_rows)

    losses = sorted((c for c in cases if c["ratio"] < 1.0), key=lambda c: c["ratio"])
    lines += ["## Slower than eager", "",
              "| case | kernel-ops ms | eager ms | ratio |", "|---|---:|---:|---:|"]
    for c in losses:
        lines.append(f"| `{c['id']}` | {c['kops_ms']:.4f} | {c['ref_ms']:.4f} | "
                     f"{c['ratio']:.2f}x |")
    lines.append("")

    text = "\n".join(lines)
    if args.write:
        Path(args.write).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.write} ({len(cases)} cases, {len(losses)} slower)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
