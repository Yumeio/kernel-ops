#!/usr/bin/env python3
"""Check the ROADMAP assigns every __global__ kernel in csrc/ to exactly one step, and that
kernel source files follow the naming policy.

Rules
  * A step is a level-2 heading `## B<n> ...`. Any other level-2 heading ends the step.
  * Kernel tables have the header row `| Header | Kernel | ...`. Backticked names ending in
    `_kernel` in the second column are the kernels that step introduces (first native version).
  * Instantiation tables (other header rows) and prose are ignored, so a kernel may be
    mentioned anywhere else without counting.
  * The roadmap is ROADMAP.md at the repo root when present, else docs/ROADMAP.md.
  * File naming (§1.5): a `.cu` name must not end in `_part<N>` / `_v<N>`, and its stem must
    appear in Phụ lục E.2 (the closed catalog). This is what keeps the "no part2/part3"
    rule enforceable instead of aspirational.

Exit code 1 on: a declared kernel not assigned, a kernel assigned twice, an assigned name
that no header declares, a banned file name, or a `.cu` outside E.2.

`--slice KEY` prints only part of the roadmap instead of checking it, so a session can read
one step without loading the whole file:
  * `--slice B8`: the B8 row of the §0.3 overview, the §1.1 beat cycle and the `## B8` section.
  * `--slice 2.5`, `--slice E.2`, `--slice "Phụ lục B"`: the heading whose title starts with
    KEY, up to the next heading of the same or a higher level.
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
        if line.startswith(("| Header | Kernel |", "| File | Kernel |")):
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


ORDINAL_RE = re.compile(r"_(part|v)[0-9]+$")


def source_name_problems(roadmap_text):
    """Enforce the §1.5 naming policy on every .cu under csrc/."""
    problems = []
    if "### E.2" in roadmap_text and "### E.3" in roadmap_text:
        catalog = roadmap_text.split("### E.2")[1].split("### E.3")[0]
    else:
        catalog = roadmap_text
    for path in sorted((ROOT / "csrc").rglob("*.cu")):
        stem = path.stem
        rel = path.relative_to(ROOT).as_posix()
        if ORDINAL_RE.search(stem):
            problems.append(f"{rel}: ordinal suffix is banned; use a concern name (ROADMAP 1.5)")
        if f"`{stem}`" not in catalog:
            problems.append(f"{rel}: stem '{stem}' is not in ROADMAP Appendix E.2")
    return problems


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def heading_block(lines, key):
    """Lines of the first heading whose title starts with `key`, up to the next heading of the
    same or a higher level; None when no heading matches."""
    for i, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match or not re.match(re.escape(key) + r"\b", match.group(2)):
            continue
        level = len(match.group(1))
        end = len(lines)
        for j in range(i + 1, len(lines)):
            other = HEADING_RE.match(lines[j])
            if other and len(other.group(1)) <= level:
                end = j
                break
        return lines[i:end]
    return None


def slice_roadmap(text, key):
    lines = text.splitlines()
    out = []
    if re.fullmatch(r"B\d+", key):
        overview = heading_block(lines, "0.3") or []
        rows = [l for l in overview if l.startswith("|")]
        out += ["### 0.3 (row)"] + rows[:2] + [l for l in rows if f"| {key} |" in l] + [""]
        beat = heading_block(lines, "1.1") or []
        start = next((i for i, l in enumerate(beat) if l.startswith("Mỗi nhịp theo cùng")), None)
        stop = next((i for i, l in enumerate(beat) if l.startswith("Nhịp xong khi")), None)
        if start is not None and stop is not None:
            out += ["### 1.1 (vòng nhịp)"] + beat[start:stop + 1] + [""]
        block = heading_block(lines, key)
    else:
        block = heading_block(lines, key)
    if block is None:
        print(f"no roadmap heading starts with '{key}'", file=sys.stderr)
        return 1
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out + block))
    return 0


def main():
    roadmap = ROOT / "ROADMAP.md"
    if not roadmap.exists():
        roadmap = ROOT / "docs" / "ROADMAP.md"
    if len(sys.argv) == 3 and sys.argv[1] == "--slice":
        return slice_roadmap(roadmap.read_text(encoding="utf-8"), sys.argv[2])
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
    name_problems = source_name_problems(roadmap.read_text(encoding="utf-8"))
    if name_problems:
        ok = False
        print("BAD SOURCE FILE NAMES:")
        for problem in name_problems:
            print(f"  {problem}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
