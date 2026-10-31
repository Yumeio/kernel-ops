#!/usr/bin/env python3
"""Minimal static probe for a compiled CUDA object.

Usage:
  python tools/sass_probe.py --sass <file.sass> [--expect HGMMA] [--forbid LDL,STL]
  python tools/sass_probe.py --ptxas <file.txt> [--max-spill 0]

The build-tree logs are inputs; this script is a dev check, not part of the
release library.
"""
from __future__ import annotations

import argparse
import re
import sys


def probe_sass(path: str, expect: list[str], forbid: list[str]) -> int:
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    missing = [m for m in expect if m not in text]
    present = [m for m in forbid if m in text]
    for m in missing:
        print(f"sass: missing expected mnemonic {m}", file=sys.stderr)
    for m in present:
        print(f"sass: forbidden mnemonic present {m}", file=sys.stderr)
    return 1 if missing or present else 0


def probe_ptxas(path: str, max_spill: int) -> int:
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    spills = [int(x) for x in re.findall(r"(\d+) bytes spill stores", text)]
    if not spills:
        print("ptxas: no spill lines found (kernel may be optimized out)", file=sys.stderr)
        return 0
    worst = max(spills)
    if worst > max_spill:
        print(f"ptxas: spill {worst} B exceeds budget {max_spill} B", file=sys.stderr)
        return 1
    print(f"ptxas: max spill {worst} B (budget {max_spill} B)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass")
    ap.add_argument("--ptxas")
    ap.add_argument("--expect", default="")
    ap.add_argument("--forbid", default="")
    ap.add_argument("--max-spill", type=int, default=0)
    args = ap.parse_args()
    code = 0
    if args.sass:
        code |= probe_sass(args.sass, [x for x in args.expect.split(",") if x],
                           [x for x in args.forbid.split(",") if x])
    if args.ptxas:
        code |= probe_ptxas(args.ptxas, args.max_spill)
    if not args.sass and not args.ptxas:
        ap.error("pass --sass and/or --ptxas")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
