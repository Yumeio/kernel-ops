#!/usr/bin/env python3
"""Check the B11/B14/B15 PyTorch gate from compare_results.py text files.

Usage:
  python3 tools/dev/gate_b11_b15.py out/b11-b15

The expected files are compare-mla.txt, compare-sparse.txt, compare-hc.txt and
compare-modelops.txt, each produced by benchmark/compare/compare_results.py.
Core cases are checked at >=2.0x; the rest are reported for context.
"""
from __future__ import annotations

import pathlib
import re
import sys

CORE = {
    "mla_decode.bf16.kv1m.q1.s8": 2.0,
    "mla_decode.bf16.kv64k.q1.s8": 2.0,
    "mla_decode.fp8.kv1m.q1.s8": 2.0,
    "lightning_indexer.bf16.r1.kv1m": 2.0,
    "sparse_mla.bf16.h128.topk2048": 2.0,
    "sparse_topk.fp32.r1.kv1m.topk2048": 2.0,
}
CORE_PREFIX = ("model_ops.mega_mhc.bf16.t1024", "model_ops.attn_res.bf16.t256.h7168.s8")

ROW = re.compile(r"^(\S+)\s+[\d.]+\s+\S+:\s*([\d.]+)\s+([\d.]+)")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/b11-b15")
    ratios = {}
    for name in ("compare-mla.txt", "compare-sparse.txt", "compare-hc.txt",
                 "compare-modelops.txt", "compare-topk.txt"):
        path = root / name
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            m = ROW.match(line)
            if m:
                ratios[m.group(1)] = float(m.group(3))
    failures = 0
    print(f"{'case':<44} {'ratio':>7}  gate")
    for case, ratio in sorted(ratios.items()):
        gate = CORE.get(case)
        if gate is None and case.startswith(CORE_PREFIX):
            gate = 2.0
        if gate is None:
            continue
        ok = ratio >= gate
        failures += 0 if ok else 1
        print(f"{case:<44} {ratio:7.2f}  {'PASS' if ok else 'FAIL'} (>= {gate:.1f}x)")
    print(f"\ncore gate: {'PASS' if failures == 0 else 'FAIL'} ({failures} case(s) below target)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
