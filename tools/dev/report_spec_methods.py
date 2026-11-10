#!/usr/bin/env python3
"""Summarize the dev-only vLLM method matrix (B10)."""
from __future__ import annotations

import argparse
import json
import pathlib


def load_manifest(path: pathlib.Path):
    p = path / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/b10/methods")
    args = ap.parse_args()
    root = pathlib.Path(args.dir)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = load_manifest(d)
        if m is None:
            continue
        tokens = sum(len(o["token_ids"]) for o in m.get("outputs", []))
        elapsed = max(float(m.get("elapsed_s", 0.0)), 1e-9)
        traces = m.get("traces", [])
        accepted = [a for t in traces for a in t.get("accepted", [])]
        acc_len = (sum(accepted) / len(accepted) + 1.0) if accepted else None
        runner = d / "runner.json"
        kops_acc = kops_ms = None
        if runner.exists():
            r = json.loads(runner.read_text())
            kops_acc = r.get("kops_accept")
            kops_ms = r.get("median_ms")
        rows.append({
            "tag": d.name,
            "tokens": tokens,
            "elapsed_s": elapsed,
            "tok_s": tokens / elapsed,
            "accept_len": acc_len,
            "kops_accept": kops_acc,
            "verifier_ms": kops_ms,
            "model": pathlib.Path(str(m.get("model", ""))).name,
            "spec": m.get("speculative_config"),
        })
    print(f"{'tag':<24} {'tok/s':>9} {'accept':>8} {'kops_acc':>9} {'verify ms':>10} {'tokens':>7}")
    print("-" * 75)
    for r in rows:
        print(f"{r['tag']:<24} {r['tok_s']:9.1f} "
              f"{(f'{r['accept_len']:.3f}' if r['accept_len'] is not None else '—'):>8} "
              f"{(f'{r['kops_accept']:.3f}' if r['kops_accept'] is not None else '—'):>9} "
              f"{(f'{r['verifier_ms']:.4f}' if r['verifier_ms'] is not None else '—'):>10} "
              f"{r['tokens']:7d}")
    # Baseline speedups use the matching Qwen target.
    base = {r["tag"]: r for r in rows}
    for r in rows:
        if "qwen35" in r["tag"]:
            b = base.get("baseline_qwen35")
        elif "llama" in r["tag"]:
            b = base.get("baseline_llama8")
        else:
            b = base.get("baseline_qwen8")
        if b and b is not r and b["tok_s"] > 0:
            print(f"{r['tag']:<24} speedup vs baseline: {r['tok_s'] / b['tok_s']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
