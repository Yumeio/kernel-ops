#!/usr/bin/env python3
"""Dev loop for this repo on the Windows dev box: build, test, bench, mutate with terse output.

Full logs go to out/dev/; the console only shows what a session needs to decide the next step
(errors, FAIL lines, slow translation units, a bench table). Approved 2026-11-02 as dev tooling
outside the ROADMAP file budget (like benchmark/compare/*.py); it is not built or shipped.

  python tools/dev/kops.py build <target>...      cmake --build --preset bench -j 2 (+ slow TU report)
  python tools/dev/kops.py test <family> [args]   build kops_test_<family>, run it, print FAILs
  python tools/dev/kops.py bench <family> [--rounds N] [--tag T]
                                                  build bench_<family>, run N rounds under the GPU
                                                  lock, print median per case + spread + ceiling check
  python tools/dev/kops.py mutate <family> [--only i,j]
                                                  apply each mutation of tools/dev/mutations.json,
                                                  rebuild, expect the test to go red, restore
  python tools/dev/kops.py gpu-run -- <command>   run any GPU command (e.g. a WSL torch baseline)
                                                  under the same GPU lock

Traps this encodes (see memory/dev-machine-build-and-bench): -j 2 because of 5-6 GB RAM; never
trust a test binary after a failed build; restore mutated files with fresh mtime; benchmark only
on AC power with no other compute process on the GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "bench"
LOGS = ROOT / "out" / "dev"
LOCK = LOGS / "gpu.lock"
VSWHERE = pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
VS_FALLBACK = pathlib.Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community")
SLOW_TU_S = 60.0   # warn; the hard limit in CONTRACTS/ROADMAP §1.5 is 90 s
TU_LIMIT_S = 90.0


def out(msg: str = "") -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------------------------
# environment
_ENV: dict | None = None


def vs_env() -> dict:
    """os.environ + vcvars64 + tools/ninja + tools/ccache, computed once per invocation."""
    global _ENV
    if _ENV is not None:
        return _ENV
    vs = VS_FALLBACK
    if VSWHERE.exists():
        found = subprocess.run([str(VSWHERE), "-latest", "-property", "installationPath"],
                               capture_output=True, text=True).stdout.strip()
        if found:
            vs = pathlib.Path(found)
    vcvars = vs / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    dump = subprocess.run(f'cmd /s /c ""{vcvars}" >nul && set"', capture_output=True, text=True,
                          shell=True)
    env = dict(os.environ)
    for line in dump.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value
    env["PATH"] = os.pathsep.join([str(ROOT / "tools" / "ninja"), str(ROOT / "tools" / "ccache"),
                                   env.get("PATH", "")])
    _ENV = env
    return env


def run_logged(cmd: list[str], log: pathlib.Path, env: dict | None = None) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def log_lines(log: pathlib.Path) -> list[str]:
    return log.read_text(encoding="utf-8", errors="replace").splitlines()


# ---------------------------------------------------------------------------------------------
# GPU lock + guard
def pid_alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(handle, ctypes.byref(code))
    k32.CloseHandle(handle)
    return code.value == 259  # STILL_ACTIVE


class GpuLock:
    """One GPU user at a time across sessions/worktrees of this checkout."""

    def __enter__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 1800
        while True:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {' '.join(sys.argv[1:])}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = LOCK.read_text().split(" ", 1)
                    pid = int(holder[0])
                except (OSError, ValueError):
                    pid = -1
                if pid > 0 and not pid_alive(pid):
                    LOCK.unlink(missing_ok=True)
                    continue
                if time.time() > deadline:
                    raise SystemExit(f"GPU lock held by {LOCK.read_text()!r} for 30 min; giving up")
                time.sleep(5)

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)


def gpu_guard() -> list[str]:
    """Reasons a timing taken now would be untrustworthy (empty list = fine)."""
    issues = []
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if apps:
        issues.append(f"other compute process on the GPU: {apps}")
    battery = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery).BatteryStatus"],
                             capture_output=True, text=True).stdout.strip()
    if battery and battery != "2":
        issues.append(f"laptop not on AC power (BatteryStatus={battery}); kernels run 15-40% slower")
    return issues


# ---------------------------------------------------------------------------------------------
# build
def ninja_log_len() -> int:
    log = BUILD / ".ninja_log"
    return len(log.read_text(errors="replace").splitlines()) if log.exists() else 0


def slow_units(since: int) -> list[tuple[float, str]]:
    log = BUILD / ".ninja_log"
    if not log.exists():
        return []
    lines = log.read_text(errors="replace").splitlines()[since:]
    result = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].isdigit():
            secs = (int(parts[1]) - int(parts[0])) / 1000.0
            if secs >= SLOW_TU_S and parts[3].endswith((".obj", ".o")):
                result.append((secs, parts[3]))
    return sorted(result, reverse=True)


def build(targets: list[str], jobs: int = 2, quiet: bool = False) -> int:
    env = vs_env()
    if not (BUILD / "build.ninja").exists():
        rc = run_logged(["cmake", "--preset", "bench"], LOGS / "configure.log", env)
        if rc:
            out(f"configure FAILED (rc={rc}); see {LOGS / 'configure.log'}")
            return rc
    before = ninja_log_len()
    log = LOGS / "build.log"
    t0 = time.time()
    rc = run_logged(["cmake", "--build", "--preset", "bench", "-j", str(jobs), "--target", *targets],
                    log, env)
    lines = log_lines(log)
    errors = [l for l in lines if " error" in l or "error:" in l or l.startswith("FAILED")]
    if rc or errors:
        out(f"BUILD FAILED rc={rc} ({time.time() - t0:.0f} s); first errors (full log {log}):")
        for l in errors[:25]:
            out("  " + l.strip()[:240])
        return rc or 1
    if not quiet:
        last = next((l for l in reversed(lines) if l.startswith("[")), "(nothing to do)")
        out(f"build ok ({time.time() - t0:.0f} s) {last.strip()[:120]}")
        for secs, obj in slow_units(before):
            flag = "OVER 90 s LIMIT" if secs > TU_LIMIT_S else "slow"
            out(f"  {flag}: {secs:.0f} s  {obj}")
    return 0


# ---------------------------------------------------------------------------------------------
# test
def test_exe(family: str) -> pathlib.Path:
    return BUILD / f"kops_test_{family}.exe"


def run_test(family: str, extra: list[str], log: pathlib.Path) -> tuple[int, list[str], int]:
    rc = run_logged([str(test_exe(family)), *extra], log)
    lines = log_lines(log)
    fails = [l for l in lines if l.startswith("FAIL") or " FAIL " in l]
    passes = sum(1 for l in lines if l.startswith("PASS"))
    return rc, fails, passes


def cmd_test(args) -> int:
    if not args.no_build and build([f"kops_test_{args.family}"], quiet=True):
        return 1
    log = LOGS / f"test_{args.family}.log"
    with GpuLock():
        rc, fails, passes = run_test(args.family, args.extra, log)
    out(f"test_{args.family}: rc={rc} PASS={passes} FAIL={len(fails)} (log {log})")
    for l in fails[:30]:
        out("  " + l[:240])
    if len(fails) > 30:
        out(f"  ... {len(fails) - 30} more")
    return rc


# ---------------------------------------------------------------------------------------------
# bench
def cmd_bench(args) -> int:
    if not args.no_build and build([f"bench_{args.family}"], quiet=True):
        return 1
    exe = BUILD / "benchmark" / f"bench_{args.family}.exe"
    tag = args.tag or time.strftime("%Y%m%d-%H%M")
    rounds = []
    with GpuLock():
        for issue in gpu_guard():
            out("WARNING: " + issue)
        for r in range(1, args.rounds + 1):
            path = LOGS / f"bench_{args.family}-{tag}-r{r}.json"
            rc = run_logged([str(exe), "--json", str(path), *args.extra],
                            LOGS / f"bench_{args.family}-{tag}-r{r}.log")
            if rc:
                out(f"bench round {r} FAILED rc={rc}; see {LOGS}")
                return rc
            rounds.append(json.loads(path.read_text(encoding="utf-8")))
    dev = rounds[0].get("device", {})
    peak = float(dev.get("peak_gbs") or 0.0)
    out(f"{dev.get('name', '')}  dirty={rounds[0].get('provenance', {}).get('dirty', '?')}"
        f"  rounds={len(rounds)}  json={LOGS}/bench_{args.family}-{tag}-r*.json")
    out(f"{'case':<52} {'median ms':>10} {'spread':>7} {'GB/s':>8} {'TFLOP/s':>8}")
    by_id: dict[str, list[dict]] = {}
    for rnd in rounds:
        for res in rnd["results"]:
            by_id.setdefault(res["id"], []).append(res)
    for cid, rs in by_id.items():
        meds = [r["median_ms"] for r in rs]
        med = statistics.median(meds)
        spread = max(meds) / min(meds) if min(meds) > 0 else 0.0
        gbs = statistics.median(r.get("gbs", 0.0) for r in rs)
        tf = statistics.median(r.get("tflops", 0.0) for r in rs)
        note = ""
        if peak and gbs > 0.97 * peak:
            note = "  <-- above DRAM peak: byte count wrong or L2-resident"
        elif spread > 1.05:
            note = "  <-- noisy, rerun"
        out(f"{cid[:52]:<52} {med:10.4f} {spread:7.3f} {gbs:8.1f} {tf:8.2f}{note}")
    return 0


# ---------------------------------------------------------------------------------------------
# mutate
def cmd_mutate(args) -> int:
    table = json.loads((ROOT / "tools" / "dev" / "mutations.json").read_text(encoding="utf-8"))
    muts = table.get(args.family)
    if not muts:
        out(f"no mutations for '{args.family}'; families: {', '.join(table)}")
        return 1
    picks = range(len(muts)) if not args.only else [int(i) for i in args.only.split(",")]
    target = f"kops_test_{args.family}"
    if build([target], quiet=True):
        out("baseline build failed; fix the tree before mutating")
        return 1
    results = []
    for i in picks:
        m = muts[i]
        path = ROOT / m["file"]
        original = path.read_bytes()
        text = original.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        old, new = m["old"].replace("\n", nl), m["new"].replace("\n", nl)
        if text.count(old) != 1:
            results.append((i, m["name"], f"STALE (old text found {text.count(old)}x)"))
            continue
        try:
            path.write_bytes(text.replace(old, new).encode("utf-8"))
            if build([target], quiet=True):
                verdict = "build failed (not a test result)"
            else:
                with GpuLock():
                    rc, fails, _ = run_test(args.family, [], LOGS / f"mutate_{args.family}_{i}.log")
                verdict = f"red ({len(fails)} FAIL)" if rc else "SURVIVED"
        finally:
            path.write_bytes(original)   # fresh mtime, so ninja rebuilds the restored file
        results.append((i, m["name"], verdict))
        out(f"  [{i}] {m['name']}: {verdict}")
    build([target], quiet=True)
    survived = [r for r in results if r[2] == "SURVIVED"]
    out(f"mutate {args.family}: {len(results)} run, {len(survived)} survived"
        + (": " + ", ".join(f"[{i}] {n}" for i, n, _ in survived) if survived else ""))
    return 1 if survived else 0


def cmd_gpu_run(args) -> int:
    with GpuLock():
        for issue in gpu_guard():
            out("WARNING: " + issue)
        return subprocess.run(args.command, cwd=ROOT).returncode


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("targets", nargs="+"); p.add_argument("-j", type=int, default=2)
    p = sub.add_parser("test"); p.add_argument("family"); p.add_argument("extra", nargs="*")
    p.add_argument("--no-build", action="store_true")
    p = sub.add_parser("bench"); p.add_argument("family"); p.add_argument("extra", nargs="*")
    p.add_argument("--rounds", type=int, default=3); p.add_argument("--tag")
    p.add_argument("--no-build", action="store_true")
    p = sub.add_parser("mutate"); p.add_argument("family"); p.add_argument("--only")
    p = sub.add_parser("gpu-run"); p.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.cmd == "build":
        return build(args.targets, args.j)
    if args.cmd == "gpu-run" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return {"test": cmd_test, "bench": cmd_bench, "mutate": cmd_mutate, "gpu-run": cmd_gpu_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
