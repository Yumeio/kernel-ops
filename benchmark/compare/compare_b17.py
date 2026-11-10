#!/usr/bin/env python3
"""Merge the B17 kernel-ops benchmark and the dev-only 3rd-party baselines.

Inputs (all written by dev commands; none is part of the build):
  out/b17/lowbit-bench.json        bench_merge --only lowbit --json ...
  out/b17/attention-rs-lowbit.json build/compare/attention_rs_lowbit --json ...
  out/b17/w2-bench.json            bench_moe --only moe.w2 --json ...
  out/b17/attention-rs-w2.json     build/compare/attention_rs_moe_w2 --json ...
  out/b17/lowbit-quality.json      benchmark/compare/lowbit_quality.py --json ...

Output: Markdown on stdout (and --markdown PATH).
"""
from __future__ import annotations

import argparse
import json
import os


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def by_id(doc):
    if doc is None:
        return {}
    out = {}
    for r in doc.get("results", []):
        out[r["id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/b17")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    d = args.dir
    kops = by_id(load(os.path.join(d, "lowbit-bench.json")))
    rs = by_id(load(os.path.join(d, "attention-rs-lowbit.json")))
    w2 = by_id(load(os.path.join(d, "w2-bench.json")))
    w2rs = by_id(load(os.path.join(d, "attention-rs-w2.json")))
    quality = load(os.path.join(d, "lowbit-quality.json"))

    lines = ["# B17 benchmark: kernel-ops vs 3rd-party references", "",
             "GPU: RTX 5060 Ti (SM120), CUDA events, 50-100 iterations, median. "
             "The attention.rs entries are dev-only builds of `3rd/attention.rs` "
             "(MIT) through `tools/dev/build_lowbit_baselines.sh`; they are never "
             "linked into kernel-ops.", ""]
    lines += ["## Low-bit KV (same shape)", "",
              "| case | kops ms | attention.rs ms | kops / 3rd (>1 = slower) | note |",
              "|---|---:|---:|---:|---|"]
    for cid in ("lowbit.store.bf16.t4096.h8.d128.turbo4",
                "lowbit.store.bf16.t4096.h8.d128.turbo3",
                "lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4",
                "lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3",
                "lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4"):
        a, b = kops.get(cid), rs.get(cid)
        if a is None:
            continue
        ratio = f"{a['median_ms'] / b['median_ms']:.2f}x" if b and b.get("median_ms") else "—"
        note = "kops faster" if b and a["median_ms"] < b["median_ms"] else ("attention.rs faster" if b else "no 3rd case")
        lines.append(f"| `{cid}` | {a['median_ms']:.4f} | "
                     f"{b['median_ms']:.4f} | {ratio} | {note} |" if b else
                     f"| `{cid}` | {a['median_ms']:.4f} | — | — | {note} |")
    lines += ["", "The kops decode also returns LSE, supports split-KV, sinks, window/chunk/softcap/"
              "ALiBi and a packed per-record scale; the attention.rs kernels return only O and use "
              "separate absmax arrays, so this is an implementation reference, not a like-for-like "
              "feature comparison.", ""]

    lines += ["## W2 experts", "",
              "| case | kops ms | 3rd ms | note |", "|---|---:|---:|---|"]
    p, q = w2.get("moe.w2.pack.bf16.e8.n2048.k4096.g32"), w2rs.get("moe.w2.pack.mxfp4.e8.n2048.k4096")
    if p:
        third = f"{q['median_ms']:.4f}" if q else "—"
        note = ("attention.rs input is already packed MXFP4 + UE8M0 scales; kops reads bf16 and "
                "computes fp32 group scales") if q else ""
        lines.append(f"| `moe.w2.pack` | {p['median_ms']:.4f} | {third} | {note} |")
    g = w2.get("moe.w2.gemm.bf16.e8.n2048.k4096.slots2048")
    if g:
        lines.append(f"| `moe.w2.gemm` | {g['median_ms']:.4f} | — | "
                     "attention.rs ships unpack + an external FP8 GEMM stub, no W2 GEMM to time |")
    if g:
        lines.append(f"\nW2 GEMM: {2 * 2048 * 2048 * 4096 / (g['median_ms'] * 1e9):.2f} TFLOP/s "
                     f"(packed weights 16 MiB, A 16 MiB, C 8 MiB).")
    lines += ["", "## Quality", "",
              "See `out/b17/quality.md` for the full table. It covers KV NMSE/cosine/SQNR, "
              "attention output/logit/LSE error and an exp(KL) perplexity proxy for every codec, "
              "plus W2 weight/output error vs attention.rs Moet. `exp(KL)` is a proxy; model PPL "
              "needs an engine and checkpoint and is outside this repo.", ""]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
