#!/usr/bin/env python3
"""Check that docs/ROADMAP.md assigns every __global__ kernel in csrc/ to exactly one step.

Rules
  * A step is a level-2 heading `## B<n> ...`. Any other level-2 heading ends the step.
  * Kernel tables have the header row `| Header | Kernel | ...`. Backticked names ending in
    `_kernel` in the second column are the kernels that step introduces (first native version).
  * Instantiation tables (other header rows) and prose are ignored, so a kernel may be
    mentioned anywhere else without counting.

Exit code 1 on: a declared kernel not assigned, a kernel assigned twice, or an assigned name
that no header declares. Prints per-step counts (used for the overview table).
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL_RE = re.compile(
    r"__global__\s+(?:void|[\w:<>]+)\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\(")
STEP_RE = re.compile(r"^##\s+(B\d+)\b")


def declared_kernels():
    found = {}
    for path in sorted((ROOT / "csrc").rglob("*.cuh")):
        for name in KERNEL_RE.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(str(path.relative_to(ROOT)))
    return found


def assigned_kernels(markdown):
    step, in_table, result, order = None, False, {}, []
    for line in markdown.splitlines():
        match = STEP_RE.match(line)
        if match:
            step, in_table = match.group(1), False
            order.append(step)
            continue
        if line.startswith("## "):
            step, in_table = None, False
            continue
        if line.startswith("| Header | Kernel |"):
            in_table = step is not None
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        for name in re.findall(r"`(\w+_kernel)`", cells[1]):
            result.setdefault(name, []).append(step)
    return result, order


def main():
    roadmap = ROOT / "docs" / "ROADMAP.md"
    declared = declared_kernels()
    assigned, order = assigned_kernels(roadmap.read_text(encoding="utf-8"))

    missing = sorted(set(declared) - set(assigned))
    unknown = sorted(set(assigned) - set(declared))
    duplicated = {k: v for k, v in assigned.items() if len(v) > 1}

    counts = {s: 0 for s in order}
    for name, steps in assigned.items():
        if name in declared:
            for s in steps:
                counts[s] += 1
    print("kernels per step:", ", ".join(f"{s}={counts[s]}" for s in order))
    print(f"declared={len(declared)} assigned={len(set(assigned) & set(declared))}")

    ok = True
    if missing:
        ok = False
        print("NOT ASSIGNED:")
        for name in missing:
            print(f"  {name}  ({', '.join(declared[name])})")
    if duplicated:
        ok = False
        print("ASSIGNED MORE THAN ONCE:")
        for name, steps in sorted(duplicated.items()):
            print(f"  {name}: {', '.join(steps)}")
    if unknown:
        ok = False
        print("NOT DECLARED IN ANY HEADER:")
        for name in unknown:
            print(f"  {name}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
