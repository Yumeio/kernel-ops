#!/usr/bin/env python3
"""Compile every header with ONLY the prerequisites its `Requires:` comment declares.

Verifies that the documented include order is complete (no hidden dependency on a
header that merely happened to be included earlier in check_all_headers.cc).
Usage: python3 tools/header_check/check_requires.py [--cxx g++]
"""
import argparse, glob, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER_FILE = os.path.join(ROOT, "tools/header_check/check_all_headers.cc")
ORDER = re.findall(r'#include "csrc/([^"]+)"', open(ORDER_FILE).read())
TOKEN = re.compile(r'([a-z_]+/(?:\*|[a-z_*]+)(?:\.cuh|\.h|\.\*)?(?:\(\.cuh\))?)')

def requires_of(path):
    lines = open(path).read().splitlines()
    deps, active = [], False
    for ln in lines[:40]:
        if "Requires:" in ln:
            active = True
            ln = ln.split("Requires:", 1)[1]
        elif active and not re.match(r"//\s{4,}\S", ln):
            active = False
        if active:
            deps += TOKEN.findall(ln)
    out = []
    for d in deps:
        d = d.replace("(.cuh)", "")
        if d.endswith("/*") or "*" in d:
            pat = d if d.endswith((".cuh", ".h")) else d.rstrip("*") + "*"
            out += [h for h in ORDER if glob.fnmatch.fnmatch(h, pat)]
        else:
            out.append(d)
    return out

def closure(h, seen=None):
    """Requires are transitive: a family header may rely on what its prerequisites require."""
    seen = set() if seen is None else seen
    for d in requires_of(os.path.join(ROOT, "csrc", h)):
        if d in ORDER and d not in seen:
            seen.add(d)
            closure(d, seen)
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    args = ap.parse_args()
    failures = 0
    for h in ORDER:
        deps = closure(h)
        unknown = [d for d in requires_of(os.path.join(ROOT, "csrc", h)) if d not in ORDER]
        incs = [d for d in ORDER if d in deps and d != h] + [h]
        src = '#include "tools/header_check/cuda_host_shim.h"\n' + "".join(
            f'#include "csrc/{i}"\n' for i in incs)
        with tempfile.NamedTemporaryFile("w", suffix=".cc", delete=False) as f:
            f.write(src)
        r = subprocess.run([args.cxx, "-std=c++17", "-fsyntax-only", "-I", ROOT, f.name],
                           capture_output=True, text=True)
        os.unlink(f.name)
        status = "ok" if r.returncode == 0 and not unknown else "FAIL"
        if status == "FAIL":
            failures += 1
            print(f"[FAIL] {h}  unknown_requires={unknown}")
            print("\n".join(r.stderr.splitlines()[:6]))
        else:
            print(f"[ ok ] {h}  ({len(incs) - 1} prerequisites)")
    print(f"\n{len(ORDER) - failures}/{len(ORDER)} headers compile with only their declared prerequisites")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
