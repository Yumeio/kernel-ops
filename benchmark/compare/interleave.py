#!/usr/bin/env python3
"""Interleaved A/B comparison of two benchmark commands (dev-only, stdlib only).

Thermal drift and clock changes on a shared GPU can move a median by several
percent between two back-to-back runs. This script removes that bias by running
both sides in the same process window, alternating the order:

    repeat 1: A B B A      (default, --order abba)
    repeat 2: A B B A
    ...
    --order abab: A B, then B A, then A B, ...  (starting side alternates)

Every run is one process that writes JSON; the script reads the case row from
each file, keeps the per-run medians, and writes a merged JSON with the median
of the medians for each side plus the ratio.

It never builds or imports a third-party project: it only executes the command
templates you give it.

Typical uses
------------
(a) attention.rs low-bit baseline vs `bench_merge --only`:

    python benchmark/compare/interleave.py \\
        --a-bin build/bench/benchmark/bench_merge \\
        --b-bin build/compare/attention_rs_lowbit \\
        --family merge --only lowbit \\
        --a-args "--rounds 3 --flush-l2 256" \\
        --b-args "--flush-mb 256 --rounds 3" \\
        --repeats 3 --out out/b17/interleave-lowbit.json

(b) old kops binary vs new kops binary (before/after A/B):

    python benchmark/compare/interleave.py \\
        --a-bin /path/to/old/bench/benchmark/bench_merge \\
        --b-bin build/bench/benchmark/bench_merge \\
        --family merge --only lowbit.decode \\
        --repeats 3 --out out/b17/interleave-before-after.json

(c) explicit command templates ({case} and {out} are replaced per run; if {out}
    is absent the script appends `--json <path>`, if {case} is absent it appends
    `--only <id>` unless the template already contains `--only`):

    python benchmark/compare/interleave.py \\
        --a-cmd "build/bench/benchmark/bench_merge --only {case} --json {out} --rounds 3 --flush-l2 256" \\
        --b-cmd "build/compare/attention_rs_lowbit --only {case} --json {out} --flush-mb 256 --rounds 3" \\
        --cases out/b17/merge-cases.json --only lowbit.decode \\
        --repeats 3 --out out/b17/interleave-lowbit.json

Case lists come from `--case`, `--cases FILE` (kops `--dump-cases`, a result
JSON, or a plain list), or `--family FAMILY` by running `--a-bin --dump-cases`.
`--only` / `--skip` are comma-separated substring filters.

Exit status: 0 when at least one case produced a comparison, 1 when none did,
2 on a usage error, 1 (with --strict) when any requested case failed or was
missing on either side. Raw per-run JSONs and logs stay under `<out>.runs/`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(msg, flush=True)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "case"


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def rows_by_id(doc) -> dict[str, dict]:
    if doc is None:
        return {}
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict) and "results" in doc:
        rows = doc["results"]
    elif isinstance(doc, dict) and "cases" in doc and isinstance(doc["cases"], list):
        rows = doc["cases"]
    elif isinstance(doc, dict) and "id" in doc:
        rows = [doc]
    else:
        rows = []
    return {r["id"]: r for r in rows if isinstance(r, dict) and "id" in r}


def case_ids_from_doc(doc) -> list[str]:
    if doc is None:
        return []
    if isinstance(doc, list):
        items = doc
    elif isinstance(doc, dict):
        items = doc.get("cases") or doc.get("results") or []
        if not items and "id" in doc:
            items = [doc]
    else:
        items = []
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "id" in item:
            out.append(item["id"])
    return out


def match_csv(case_id: str, csv_value: str | None) -> bool:
    if not csv_value:
        return False
    return any(tok.strip() and tok.strip() in case_id for tok in csv_value.split(","))


def filter_ids(ids: list[str], only: str | None, skip: str | None) -> list[str]:
    seen, out = set(), []
    for case_id in ids:
        if only and not match_csv(case_id, only):
            continue
        if skip and match_csv(case_id, skip):
            continue
        if case_id not in seen:
            seen.add(case_id)
            out.append(case_id)
    return out


def run_capture(cmd: list[str], timeout: int, env: dict, dry_run: bool, log_path: Path | None):
    started = time.time()
    if dry_run:
        line = "+ " + " ".join(shlex.quote(c) for c in cmd)
        log(line)
        return 0, "", "", 0.0
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                              env=env)
        elapsed = time.time() - started
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
                f.write(f"# rc={proc.returncode} elapsed={elapsed:.2f}s\n")
                f.write(proc.stdout)
                if proc.stderr:
                    f.write("\n--- stderr ---\n")
                    f.write(proc.stderr)
        return proc.returncode, proc.stdout, proc.stderr, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        return 124, exc.stdout or "", f"timeout after {timeout}s", elapsed
    except FileNotFoundError as exc:
        return 127, "", str(exc), time.time() - started
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}", time.time() - started


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    return {"commit": run("rev-parse", "HEAD") or "unknown",
            "dirty": bool(run("status", "--porcelain"))}


def thirdparty_pin(name: str) -> str:
    manifest = REPO_ROOT / "tools" / "dev" / "3rd_manifest.tsv"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if cols and cols[0] == name:
                return cols[2] if len(cols) > 2 else ""
    except OSError:
        pass
    return ""


def thirdparty_git(name: str) -> dict:
    root = REPO_ROOT / "3rd" / name
    out = {"commit": thirdparty_pin(name), "dirty": None, "source": "manifest"}
    if not (root / ".git").exists():
        return out
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                 "--untracked-files=no"], capture_output=True, text=True,
                                timeout=30).stdout.strip()
        if commit:
            out.update({"commit": commit, "dirty": bool(status), "source": "git"})
    except Exception:
        pass
    return out


def provenance_env(git: dict) -> dict:
    env = {
        "PYTHONUNBUFFERED": "1",
        "KOPS_BENCH_COMMIT": git.get("commit", "") or "unknown",
        "KOPS_BENCH_DIRTY": "1" if git.get("dirty") else "0",
    }
    tp = thirdparty_git("attention.rs")
    if tp["commit"]:
        env["KOPS_THIRDPARTY"] = "attention.rs"
        env["KOPS_THIRDPARTY_COMMIT"] = tp["commit"]
        if tp["dirty"] is not None:
            env["KOPS_THIRDPARTY_DIRTY"] = "1" if tp["dirty"] else "0"
    return env


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    except Exception:
        return {"name": "unknown"}
    if not out:
        return {"name": "unknown"}
    parts = [x.strip() for x in out[0].split(",")]
    info = {"name": parts[0] if parts else "unknown"}
    if len(parts) > 1:
        info["cc"] = parts[1]
    if len(parts) > 2:
        info["clock_sm_mhz"] = parts[2]
    if len(parts) > 3:
        info["temperature_c"] = parts[3]
    return info


_protocol_keys = ("mode", "flush_l2_mb", "rounds", "samples_per_launch", "iters")


def protocol_key(doc) -> str:
    """Normalized timing protocol so a kops JSON and an external JSON compare equal."""
    if not isinstance(doc, dict):
        return ""
    proto = doc.get("protocol")
    detail = doc.get("protocol_detail")
    if isinstance(detail, dict):
        return json.dumps({k: detail.get(k) for k in _protocol_keys}, sort_keys=True)
    if isinstance(proto, dict):
        return json.dumps({k: proto.get(k) for k in _protocol_keys}, sort_keys=True)
    if isinstance(proto, str):
        return json.dumps({"protocol": proto}, sort_keys=True)
    return ""


def source_provenance(doc) -> dict:
    if not isinstance(doc, dict):
        return {}
    out = {}
    prov = doc.get("provenance")
    if isinstance(prov, dict):
        out.update(prov)
    if "backend" in doc:
        out.setdefault("backend", doc["backend"])
    if "gpu" in doc:
        out.setdefault("gpu", doc.get("gpu"))
    return out


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def aggregate(results: list[dict]) -> dict:
    """Median of the per-run medians, plus the per-run spread."""
    ok = [r for r in results if r.get("ok") and isinstance(r.get("row"), dict)]
    failures = [r for r in results if not r.get("ok")]
    if not ok:
        statuses = {r.get("status", "failed") for r in failures}
        status = "missing" if statuses == {"missing"} else "failed"
        reasons = [r.get("reason", "") for r in failures if r.get("reason")]
        return {"status": status, "n_runs": len(results), "n_ok": 0,
                "reason": "; ".join(dict.fromkeys(reasons))[:600]}
    rows = [r["row"] for r in ok]
    medians = [float(r["median_ms"]) for r in rows if r.get("median_ms") is not None]
    if not medians:
        return {"status": "failed", "n_runs": len(results), "n_ok": len(ok),
                "reason": "rows have no median_ms"}

    def field(name):
        return median_or_none([r.get(name) for r in rows])

    samples_med = median_or_none([r.get("samples") for r in rows])
    agg = {
        "status": "ok" if not failures else "partial",
        "n_runs": len(results),
        "n_ok": len(ok),
        "median_ms": statistics.median(medians),
        "p10_ms": field("p10_ms"),
        "p90_ms": field("p90_ms"),
        "min_ms": min((float(r["min_ms"]) for r in rows if r.get("min_ms") is not None),
                      default=None),
        "samples": int(samples_med) if samples_med is not None else None,
        "gbs": field("gbs"),
        "tflops": field("tflops"),
        "run_medians_ms": medians,
        "protocol": protocol_key(ok[0].get("doc")),
        "provenance": source_provenance(ok[0].get("doc")),
    }
    if len(medians) > 1 and agg["median_ms"]:
        agg["spread_pct"] = 100.0 * (max(medians) - min(medians)) / agg["median_ms"]
    else:
        agg["spread_pct"] = 0.0
    if failures:
        reasons = [r.get("reason", "") for r in failures if r.get("reason")]
        agg["reason"] = "; ".join(dict.fromkeys(reasons))[:600]
    return agg


def template_command(template: str, side: str, case_id: str, out_path: Path, runs_dir: Path,
                     no_only: bool) -> list[str]:
    has_out = "{out}" in template
    has_case = "{case}" in template
    text = (template.replace("{case}", case_id)
                    .replace("{out}", str(out_path))
                    .replace("{runs}", str(runs_dir))
                    .replace("{side}", side))
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        raise SystemExit(f"bad --{side}-cmd template: {exc}")
    if not has_out:
        parts += ["--json", str(out_path)]
    if not has_case and not no_only and "--only" not in template:
        parts += ["--only", case_id]
    return parts


def bin_command(binary: str, args_text: str | None, case_id: str, out_path: Path,
                no_only: bool) -> list[str]:
    parts = [binary]
    if args_text:
        try:
            parts += shlex.split(args_text)
        except ValueError as exc:
            raise SystemExit(f"bad args: {exc}")
    if not no_only and "--only" not in parts:
        parts += ["--only", case_id]
    parts += ["--json", str(out_path)]
    return parts


def discover_cases(args) -> list[str]:
    ids: list[str] = []
    if args.case:
        ids = list(args.case)
    elif args.cases:
        doc = load_json(Path(args.cases))
        if doc is None:
            raise SystemExit(f"cannot read --cases {args.cases}")
        ids = case_ids_from_doc(doc)
    elif args.a_bin:
        # Case discovery is a read-only query, so run it even with --dry-run.
        env = os.environ.copy()
        rc, out, err, _ = run_capture([args.a_bin, "--dump-cases"], 300, env, False, None)
        if rc != 0:
            raise SystemExit(f"--dump-cases failed for {args.a_bin}: {err or 'rc=' + str(rc)}")
        ids = case_ids_from_doc(json.loads(out))
    if not ids:
        raise SystemExit("no cases: pass --case, --cases FILE, or --a-bin with --family")
    ids = filter_ids(ids, args.only, args.skip)
    if not ids:
        raise SystemExit("--only/--skip filtered out every case")
    return ids


def run_one(side: str, case_id: str, index: int, repeat: int, args, env: dict) -> dict:
    runs_dir = args.runs_dir
    case_dir = runs_dir / sanitize(case_id)
    out_path = case_dir / f"{side}-r{repeat + 1}-{index + 1}.json"
    log_path = out_path.with_suffix(".log")
    if not args.dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)
    try:
        if side == "a":
            if args.a_cmd:
                cmd = template_command(args.a_cmd, side, case_id, out_path, runs_dir, args.no_only)
            else:
                cmd = bin_command(args.a_bin, args.a_args, case_id, out_path, args.no_only)
        else:
            if args.b_cmd:
                cmd = template_command(args.b_cmd, side, case_id, out_path, runs_dir, args.no_only)
            else:
                cmd = bin_command(args.b_bin, args.b_args, case_id, out_path, args.no_only)
    except SystemExit:
        raise
    if out_path.exists() and not args.dry_run:
        out_path.unlink()
    rc, stdout, stderr, elapsed = run_capture(cmd, args.timeout, env, args.dry_run, log_path)
    if args.dry_run:
        return {"ok": False, "status": "dry_run", "reason": "dry run", "cmd": cmd,
                "seconds": 0.0}
    if rc != 0:
        reason = (stderr or stdout).strip().splitlines()
        return {"ok": False, "status": "failed", "reason": f"rc={rc}: " + (reason[-1] if reason else ""),
                "cmd": cmd, "seconds": elapsed}
    doc = load_json(out_path)
    if doc is None:
        return {"ok": False, "status": "failed", "reason": f"no JSON at {out_path}",
                "cmd": cmd, "seconds": elapsed}
    row = rows_by_id(doc).get(case_id)
    if row is None:
        return {"ok": False, "status": "missing", "reason": f"{case_id} not in {out_path.name}",
                "cmd": cmd, "seconds": elapsed, "doc": doc}
    return {"ok": True, "status": "ok", "row": row, "doc": doc, "cmd": cmd, "seconds": elapsed}


def side_meta(side: str, args, docs: list) -> dict:
    protocols = sorted({protocol_key(d) for d in docs if protocol_key(d)})
    provenance = {}
    for doc in docs:
        for key, value in source_provenance(doc).items():
            provenance.setdefault(key, value)
    if side == "a":
        source = args.a_cmd or args.a_bin
        extra = args.a_args
    else:
        source = args.b_cmd or args.b_bin
        extra = args.b_args
    return {"label": args.a_label if side == "a" else args.b_label,
            "command": source, "args": extra or "",
            "protocols": protocols, "provenance": provenance}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-cmd", default=None, help="command template for A ({case}, {out}, {runs})")
    ap.add_argument("--b-cmd", default=None, help="command template for B")
    ap.add_argument("--a-bin", default=None, help="A benchmark binary (uses --only/--json)")
    ap.add_argument("--b-bin", default=None, help="B benchmark binary")
    ap.add_argument("--a-args", default=None, help="extra A binary arguments, shell-quoted")
    ap.add_argument("--b-args", default=None, help="extra B binary arguments, shell-quoted")
    ap.add_argument("--a-label", default="a", help="display label for A")
    ap.add_argument("--b-label", default="b", help="display label for B")
    ap.add_argument("--cases", default=None, help="case/result JSON with the case ids")
    ap.add_argument("--family", default=None,
                    help="only used in --a-bin mode: run `<a-bin> --dump-cases`")
    ap.add_argument("--case", action="append", default=[], help="explicit case id (repeatable)")
    ap.add_argument("--only", default=None, help="comma-separated case-id substrings to keep")
    ap.add_argument("--skip", default=None, help="comma-separated case-id substrings to drop")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--order", choices=("abba", "abab"), default="abba")
    ap.add_argument("--out", required=True, help="merged JSON path")
    ap.add_argument("--runs-dir", default=None, help="raw per-run directory (default <out>.runs)")
    ap.add_argument("--timeout", type=int, default=900, help="per-run timeout in seconds")
    ap.add_argument("--device", type=int, default=None, help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--no-only", action="store_true",
                    help="do not append --only; the command must select the case itself")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any requested case fails or is missing on a side")
    ap.add_argument("--quiet", action="store_true")
    return ap


def validate(args) -> None:
    for side in ("a", "b"):
        cmd = getattr(args, f"{side}_cmd")
        binary = getattr(args, f"{side}_bin")
        if bool(cmd) == bool(binary):
            raise SystemExit(f"pass exactly one of --{side}-cmd or --{side}-bin")
        if cmd and getattr(args, f"{side}_args"):
            raise SystemExit(f"--{side}-args only applies to --{side}-bin")
        if binary:
            path = Path(binary)
            if not args.dry_run and not path.exists():
                raise SystemExit(f"--{side}-bin not found: {binary}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.runs_dir is None:
        out = Path(args.out)
        args.runs_dir = out.with_suffix(out.suffix + ".runs")
    else:
        args.runs_dir = Path(args.runs_dir)
    if not args.dry_run:
        Path(args.runs_dir).mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    validate(args)
    cases = discover_cases(args)
    git = git_info()
    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env.update(provenance_env(git))

    if not args.quiet:
        log(f"[interleave] {len(cases)} case(s), {args.repeats} repeat(s), order={args.order}")
        log(f"[interleave] A = {args.a_label}: {args.a_cmd or args.a_bin}")
        log(f"[interleave] B = {args.b_label}: {args.b_cmd or args.b_bin}")
        log(f"[interleave] runs -> {args.runs_dir}")

    order = ["a", "b", "b", "a"] if args.order == "abba" else ["a", "b"]
    merged_cases = []
    all_docs = {"a": [], "b": []}
    for case_index, case_id in enumerate(cases):
        results = {"a": [], "b": []}
        for repeat in range(args.repeats):
            sides = order if args.order == "abba" else (
                ["a", "b"] if repeat % 2 == 0 else ["b", "a"])
            for index, side in enumerate(sides):
                results[side].append(run_one(side, case_id, index, repeat, args, env))
        agg_a = aggregate(results["a"])
        agg_b = aggregate(results["b"])
        for side in ("a", "b"):
            all_docs[side] += [r["doc"] for r in results[side] if r.get("doc") is not None]
        row = {"id": case_id, "a": agg_a, "b": agg_b}
        if agg_a.get("median_ms") and agg_b.get("median_ms"):
            row["a_ms"] = agg_a["median_ms"]
            row["b_ms"] = agg_b["median_ms"]
            row["ratio"] = agg_b["median_ms"] / agg_a["median_ms"]
            row["speedup_b_over_a"] = agg_a["median_ms"] / agg_b["median_ms"]
            row["status"] = "ok"
            row["partial"] = agg_a["status"] != "ok" or agg_b["status"] != "ok"
        elif agg_a.get("median_ms"):
            row["status"] = "missing_in_b" if agg_b["status"] == "missing" else "failed_b"
        elif agg_b.get("median_ms"):
            row["status"] = "missing_in_a" if agg_a["status"] == "missing" else "failed_a"
        else:
            row["status"] = "failed"
        merged_cases.append(row)
        if not args.quiet:
            if "ratio" in row:
                a_proto = agg_a.get("protocol", "")
                b_proto = agg_b.get("protocol", "")
                mismatch = " PROTOCOL-MISMATCH" if a_proto and b_proto and a_proto != b_proto else ""
                log(f"[interleave] {case_id:<52} A {row['a_ms']:.4f} ms  B {row['b_ms']:.4f} ms  "
                    f"B/A {row['ratio']:.3f}{mismatch}")
            else:
                log(f"[interleave] {case_id:<52} {row['status']}")

    compared = [r for r in merged_cases if "ratio" in r]
    ratios = [r["ratio"] for r in compared]
    totals = {
        "cases": len(merged_cases),
        "compared": len(compared),
        "a_faster": sum(1 for r in ratios if r > 1.0),
        "b_faster": sum(1 for r in ratios if r < 1.0),
        "ties": sum(1 for r in ratios if r == 1.0),
        "ratio_geomean": (math.exp(sum(math.log(max(r, 1e-12)) for r in ratios) / len(ratios))
                          if ratios else None),
        "a_median_ms": median_or_none([r["a_ms"] for r in compared]),
        "b_median_ms": median_or_none([r["b_ms"] for r in compared]),
    }
    warnings = []
    if totals["compared"] < len(merged_cases):
        warnings.append(f"{len(merged_cases) - totals['compared']} case(s) have no comparison")
    a_meta = side_meta("a", args, all_docs["a"])
    b_meta = side_meta("b", args, all_docs["b"])
    if a_meta["protocols"] and b_meta["protocols"] and a_meta["protocols"] != b_meta["protocols"]:
        warnings.append("A and B measured with different protocols: "
                        f"A={a_meta['protocols']} B={b_meta['protocols']}")

    payload = {
        "schema": 1,
        "kind": "interleave",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "order": args.order,
        "repeats": args.repeats,
        "only": args.only,
        "skip": args.skip,
        "a": a_meta,
        "b": b_meta,
        "warnings": warnings,
        "cases": merged_cases,
        "totals": totals,
        "env": {"gpu": gpu_info(), "git": git,
                "thirdparty": {"attention.rs": thirdparty_git("attention.rs")},
                "python": sys.version.split()[0], "argv": sys.argv},
    }
    out = Path(args.out)
    dump_json(out, payload)
    if not args.quiet:
        log("")
        if totals["ratio_geomean"] is not None:
            log(f"[interleave] compared {totals['compared']}/{totals['cases']} cases; "
                f"B/A geomean {totals['ratio_geomean']:.3f}x")
        else:
            log("[interleave] no comparable cases")
        log(f"[interleave] wrote {out}")
        for warning in warnings:
            log(f"[interleave][warn] {warning}")
    if args.dry_run:
        return 0
    if not compared:
        return 1
    if args.strict and any(r["status"] != "ok" for r in merged_cases):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
