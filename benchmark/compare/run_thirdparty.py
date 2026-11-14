#!/usr/bin/env python3
"""One-command third-party benchmark for kernel-ops.

Dev-only. Runs the C++ `bench_<family>` output through the third-party adapters
in `thirdparty.py`, the external harnesses under `benchmark/compare/external`,
and compares each case against the best available third-party median.

    python benchmark/compare/run_thirdparty.py --all --rounds 3 --flush-l2 256
    python benchmark/compare/run_thirdparty.py --families norm,gemm --impl vllm --torch
    python benchmark/compare/run_thirdparty.py --check
    python benchmark/compare/run_thirdparty.py --all --resume

This script never clones or builds a third-party library. It only checks what
is already installed/built and records missing references with a reason. The
clone shell is separate: `tools/dev/clone_3rd.sh`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_DIR = REPO_ROOT / "benchmark" / "compare"
sys.path.insert(0, str(COMPARE_DIR))

from thirdparty_matrix import (  # noqa: E402
    EXTERNALS,
    FAMILIES,
    KNOWN_REFERENCES,
    PY_IMPLS,
)

FAMILY_ORDER = list(FAMILIES.keys())


@dataclass
class CmdResult:
    name: str
    cmd: list[str]
    returncode: int
    seconds: float
    log: Path
    error: str = ""


@dataclass
class RunState:
    out_dir: Path
    logs: Path
    cases: Path
    kops: Path
    thirdparty: Path
    torch: Path
    external: Path
    compare: Path
    errors: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    commands: list[CmdResult] = field(default_factory=list)


def log(msg: str) -> None:
    print(msg, flush=True)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "gpu"


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


def detect_gpu() -> dict:
    info = {"name": "unknown", "cc": "", "vram_gb": 0.0, "arch": "unknown", "sms": 0}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
    except Exception:
        out = []
    if out:
        parts = [x.strip() for x in out[0].split(",")]
        if len(parts) >= 3:
            info["name"] = parts[0]
            info["vram_gb"] = float(parts[1]) / 1024.0
            info["cc"] = parts[2]
    cc = info["cc"]
    mapping = {"8.6": "sm86", "8.9": "sm89", "9.0": "sm90a",
               "10.0": "sm100a", "10.3": "sm103a", "12.0": "sm120a", "12.1": "sm121a"}
    info["arch"] = mapping.get(cc, "sm" + cc.replace(".", "") if cc else "unknown")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        info["smi"] = smi
    except Exception:
        pass
    return info


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "status": run("status", "--porcelain")[:2000],
    }


def python_impls() -> dict[str, dict]:
    """Return {impl: {available, reason}} for the adapters in thirdparty.py."""
    def spec(name):
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    has_torch = spec("torch")
    out: dict[str, dict] = {}
    if not has_torch:
        for impl in PY_IMPLS:
            out[impl] = {"available": False, "reason": "torch not installed"}
        return out
    out["sdpa"] = {"available": True, "reason": "torch"}
    out["fa2"] = {"available": spec("flash_attn"), "reason": "flash_attn"}
    out["vllm"] = {"available": spec("vllm"), "reason": "vllm"}
    out["flashinfer"] = {"available": spec("flashinfer"), "reason": "flashinfer"}
    sgl_reason = "sgl_kernel"
    if not spec("sgl_kernel") and (REPO_ROOT / "3rd" / "sglang").exists():
        sgl_reason = "sgl_kernel not installed (3rd/sglang cloned)"
    out["sglang"] = {"available": spec("sgl_kernel"), "reason": sgl_reason}
    out["bnb"] = {"available": spec("bitsandbytes"), "reason": "bitsandbytes"}
    return out


def external_paths() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, meta in EXTERNALS.items():
        found = None
        for rel in meta["paths"]:
            p = REPO_ROOT / rel
            if p.exists():
                found = p
                break
            if sys.platform == "win32" and (p.with_suffix(".exe")).exists():
                found = p.with_suffix(".exe")
                break
        out[name] = {"available": found is not None, "path": str(found) if found else "",
                     "build": meta["build"]}
    return out


def find_build_dir(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        REPO_ROOT / "build-wsl" / "bench",
        REPO_ROOT / "build" / "bench-sm120a",
        REPO_ROOT / "build" / "bench-sm90a",
        REPO_ROOT / "build" / "bench-sm100a",
        REPO_ROOT / "build" / "bench",
        REPO_ROOT / "build" / "dev",
    ]
    for d in candidates:
        if not d.is_absolute():
            d = REPO_ROOT / d
        if (d / "benchmark" / "bench_norm").exists() or (d / "benchmark" / "bench_norm.exe").exists():
            return d.resolve()
    return None


def kops_bin(build_dir: Path, family: str) -> Path | None:
    p = build_dir / "benchmark" / f"bench_{family}"
    if p.exists():
        return p
    exe = p.with_suffix(".exe")
    if exe.exists():
        return exe
    return None


def subprocess_env(device: int | None) -> dict:
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_cmd(state: RunState, name: str, cmd: list[str], log_path: Path,
            timeout: int, dry_run: bool, env: dict, capture_stdout: Path | None = None) -> CmdResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if dry_run:
        line = "+ " + " ".join(shlex.quote(c) for c in cmd)
        print(line, flush=True)
        log_path.write_text(line + "\n", encoding="utf-8")
        return CmdResult(name, cmd, 0, 0.0, log_path)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        logf.flush()
        try:
            if capture_stdout is not None:
                capture_stdout.parent.mkdir(parents=True, exist_ok=True)
                with open(capture_stdout, "w", encoding="utf-8") as out:
                    p = subprocess.run(cmd, cwd=REPO_ROOT, stdout=out, stderr=logf,
                                       timeout=timeout, env=env)
            else:
                p = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=env)
            rc, err = p.returncode, ""
        except subprocess.TimeoutExpired:
            rc, err = 124, f"timeout after {timeout}s"
            logf.write("\n" + err + "\n")
        except FileNotFoundError as exc:
            rc, err = 127, str(exc)
            logf.write("\n" + err + "\n")
        except Exception as exc:  # noqa: BLE001
            rc, err = 1, f"{type(exc).__name__}: {exc}"
            logf.write("\n" + err + "\n")
    elapsed = time.time() - started
    result = CmdResult(name, cmd, rc, elapsed, log_path, err)
    state.commands.append(result)
    if rc != 0:
        state.errors.append({"name": name, "rc": rc, "error": err,
                             "log": str(log_path), "cmd": cmd})
    return result


def matches(case_id: str, csv_value: str | None) -> bool:
    if not csv_value:
        return False
    return any(tok.strip() and tok.strip() in case_id for tok in csv_value.split(","))


def is_large_case(case: dict) -> bool:
    cid = case.get("id", "").lower()
    if re.search(r"(^|[._-])(1m|128k|256k|512k|kv1m|t1m|m1m)([._-]|$)", cid):
        return True
    return False


def filter_cases(raw_cases: dict, args, vram_gb: float) -> tuple[list[dict], int, int]:
    cases = raw_cases.get("cases", [])
    skip_large = args.skip_large
    if skip_large is None:
        skip_large = vram_gb < 8.0
    max_bytes = args.max_case_bytes
    if skip_large and max_bytes is None:
        max_bytes = int(vram_gb * (1024 ** 3) * 0.55) if vram_gb > 0 else 0
    selected, dropped = [], 0
    for c in cases:
        cid = c.get("id", "")
        if args.only and not matches(cid, args.only):
            continue
        if args.skip and matches(cid, args.skip):
            continue
        if skip_large and (is_large_case(c) or (max_bytes and int(c.get("bytes", 0)) > max_bytes)):
            dropped += 1
            continue
        selected.append(c)
    return selected, dropped, len(cases)


def write_cases(state: RunState, family: str, selected: list[dict]) -> Path:
    path = state.cases / f"{family}.json"
    dump_json(path, {"schema": 1, "backend": "kops", "cases": selected})
    return path


def normalize_baseline(path: Path, impl: str) -> dict | None:
    doc = load_json(path)
    if doc is None:
        return None
    if isinstance(doc, list):
        doc = {"results": doc}
    if "results" not in doc and "id" in doc:
        doc = {"results": [doc]}
    if "results" not in doc:
        return None
    for r in doc["results"]:
        r.setdefault("impl", impl)
    doc.setdefault("impl", impl)
    return doc


def filter_results_inplace(doc: dict, case_ids: set[str]) -> dict:
    doc["results"] = [r for r in doc.get("results", []) if r.get("id") in case_ids]
    return doc


def run_kops(state: RunState, family: str, binary: Path, cases_path: Path,
             args, env: dict, case_ids: set[str], skip_ids: set[str]) -> Path | None:
    out = state.kops / f"{family}.json"
    if args.resume and out.exists():
        doc = load_json(out)
        if doc is not None:
            filter_results_inplace(doc, case_ids)
            dump_json(out, doc)
            return out
    cmd = [str(binary), "--json", str(out), "--rounds", str(args.rounds),
           "--flush-l2", str(args.flush_l2)]
    if args.iters:
        cmd += ["--iters", str(args.iters)]
    if args.only:
        cmd += ["--only", args.only]
    skip_terms = []
    if args.skip:
        skip_terms.append(args.skip)
    if skip_ids:
        skip_terms.extend(sorted(skip_ids))
    if skip_terms:
        cmd += ["--skip", ",".join(skip_terms)]
    if family == "communication":
        cmd += ["--mode", args.mode]
    r = run_cmd(state, f"kops/{family}", cmd, state.logs / f"kops-{family}.log",
                args.timeout, args.dry_run, env)
    if r.returncode != 0 or not out.exists():
        state.skipped.append({"family": family, "impl": "kops",
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    doc = load_json(out)
    if doc is not None:
        filter_results_inplace(doc, case_ids)
        dump_json(out, doc)
    return out


def run_python_baseline(state: RunState, family: str, impl: str, cases_path: Path,
                        args, env: dict) -> Path | None:
    out = state.thirdparty / impl / f"{family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists() and load_json(out) is not None:
        return out
    cmd = [sys.executable, str(COMPARE_DIR / "thirdparty.py"),
           "--cases", str(cases_path), "--impl", impl, "--json", str(out),
           "--rounds", str(args.rounds), "--flush-l2", str(args.flush_l2)]
    r = run_cmd(state, f"thirdparty/{impl}/{family}", cmd,
                state.logs / f"thirdparty-{impl}-{family}.log", args.timeout,
                args.dry_run, env)
    if r.returncode != 0:
        state.skipped.append({"family": family, "impl": impl,
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    doc = normalize_baseline(out, impl)
    if doc is None:
        state.skipped.append({"family": family, "impl": impl, "reason": "no results JSON"})
        return None
    dump_json(out, doc)
    return out


def run_torch_baseline(state: RunState, family: str, style: str, cases_path: Path,
                       args, env: dict) -> Path | None:
    out = state.torch / style / f"{family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists() and load_json(out) is not None:
        return out
    cmd = [sys.executable, str(COMPARE_DIR / "torch_baseline.py"),
           "--cases", str(cases_path), "--json", str(out), "--style", style,
           "--rounds", str(args.rounds), "--flush-l2", str(args.flush_l2)]
    r = run_cmd(state, f"torch/{style}/{family}", cmd,
                state.logs / f"torch-{style}-{family}.log", args.timeout,
                args.dry_run, env)
    if r.returncode != 0:
        state.skipped.append({"family": family, "impl": f"torch-{style}",
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    return out


def run_csa(state: RunState, family: str, cases_path: Path, args, env: dict) -> Path | None:
    out = state.external / "csa_torch" / f"{family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists() and load_json(out) is not None:
        return out
    cmd = [sys.executable, str(COMPARE_DIR / "csa_torch.py"),
           "--cases", str(cases_path), "--json", str(out),
           "--rounds", str(args.rounds)]
    if args.flush_l2:
        cmd += ["--flush-l2", str(args.flush_l2)]
    r = run_cmd(state, f"csa_torch/{family}", cmd,
                state.logs / f"csa-{family}.log", args.timeout, args.dry_run, env)
    if r.returncode != 0:
        state.skipped.append({"family": family, "impl": "csa_torch",
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    return out


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
        return len([x for x in out.splitlines() if x.startswith("GPU ")])
    except Exception:
        return 0


def run_comm_torch(state: RunState, family: str, cases_path: Path, args, env: dict) -> Path | None:
    out = state.external / "communication_torch" / f"{family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists() and load_json(out) is not None:
        return out
    if gpu_count() < 2:
        state.skipped.append({"family": family, "impl": "communication_torch",
                              "reason": "requires >=2 GPUs"})
        return None
    cmd = [sys.executable, str(COMPARE_DIR / "communication_torch.py"),
           "--cases", str(cases_path), "--json", str(out),
           "--calls", str(args.comm_calls), "--groups", str(args.comm_groups)]
    r = run_cmd(state, f"communication_torch/{family}", cmd,
                state.logs / f"comm-torch-{family}.log", args.timeout, args.dry_run, env)
    if r.returncode != 0:
        state.skipped.append({"family": family, "impl": "communication_torch",
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    return out


def run_external(state: RunState, family: str, name: str, path: Path, cases_path: Path,
                 args, env: dict, case_ids: set[str]) -> Path | None:
    out = state.external / name / f"{family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists() and load_json(out) is not None:
        return out
    if name == "llamacpp":
        csv_path = state.external / name / f"{family}.csv"
        r = run_cmd(state, f"external/{name}", [str(path)], state.logs / f"external-{name}.log",
                    args.timeout, args.dry_run, env, capture_stdout=csv_path)
        if r.returncode != 0:
            state.skipped.append({"family": family, "impl": name,
                                  "reason": r.error or f"rc={r.returncode}"})
            return None
        cmd = [sys.executable, str(COMPARE_DIR / "llamacpp_csv.py"),
               "--csv", str(csv_path), "--json", str(out)]
        if args.only:
            cmd += ["--only", args.only]
        if args.skip:
            cmd += ["--skip", args.skip]
        r2 = run_cmd(state, f"external/{name}/convert", cmd,
                     state.logs / f"external-{name}-convert.log", 120, args.dry_run, env)
        return out if r2.returncode == 0 and out.exists() else None
    if name in ("attention_rs_lowbit", "attention_rs_w2"):
        cmd = [str(path), "--json", str(out)]
    elif name in ("hpc_sampler", "sglang_tree"):
        cmd = [str(path), "--json", str(out), "--flush-mb", str(args.flush_l2)]
    elif name == "nccl":
        cmd = [str(path), "--cases", str(cases_path), "--json", str(out),
               "--mode", args.mode]
        if args.only:
            cmd += ["--only", args.only]
    else:
        state.skipped.append({"family": family, "impl": name, "reason": "unknown external"})
        return None
    r = run_cmd(state, f"external/{name}/{family}", cmd,
                state.logs / f"external-{name}-{family}.log", args.timeout,
                args.dry_run, env)
    if r.returncode != 0 or not out.exists():
        state.skipped.append({"family": family, "impl": name,
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    doc = normalize_baseline(out, name)
    if doc is None:
        state.skipped.append({"family": family, "impl": name, "reason": "no results JSON"})
        return None
    filter_results_inplace(doc, case_ids)
    dump_json(out, doc)
    return out


def compare_family(state: RunState, family: str, kops_path: Path | None,
                   refs: list[tuple[str, Path]], args, env: dict) -> Path | None:
    if kops_path is None:
        return None
    out = state.compare / f"{family}.json"
    if family == "mega":
        return compare_mega(state, family, kops_path, refs, out)
    ref_paths = [(label, p) for label, p in refs if p.exists()]
    if not ref_paths:
        doc = {"family": family, "kops": str(kops_path), "refs": [], "cases": [],
               "note": "no third-party baseline available"}
        dump_json(out, doc)
        return out
    cmd = [sys.executable, str(COMPARE_DIR / "compare_results.py"), str(kops_path)]
    cmd += [str(p) for _, p in ref_paths]
    if ref_paths:
        cmd += ["--names"] + [label for label, _ in ref_paths]
    if family == "sampling":
        cmd += ["--groups", "2"]
    cmd += ["--json", str(out)]
    r = run_cmd(state, f"compare/{family}", cmd, state.logs / f"compare-{family}.log",
                args.timeout, args.dry_run, env)
    if r.returncode != 0 or not out.exists():
        state.skipped.append({"family": family, "impl": "compare",
                              "reason": r.error or f"rc={r.returncode}"})
        return None
    return out


def compare_mega(state: RunState, family: str, kops_path: Path,
                 refs: list[tuple[str, Path]], out: Path) -> Path | None:
    doc = load_json(kops_path) or {}
    rows = {r["id"]: r for r in doc.get("results", [])}
    cases = []
    for base in ("mega.mlp", "mega.attn", "mega.moe", "mega.decoder"):
        for cand in (base + ".mega", base + ".mega_graph"):
            k = rows.get(cand)
            if k is None:
                continue
            best = None
            for name, _ in refs:
                for ref_id in (base + ".modular_graph", base + ".modular"):
                    r = rows.get(ref_id)
                    if r is not None and (best is None or r["median_ms"] < best[1]["median_ms"]):
                        best = (name + ":" + ref_id, r)
            kms = float(k["median_ms"])
            if best is None:
                cases.append({"id": cand, "kops_ms": kms, "ref": None, "ratio": None})
            else:
                name, r = best
                rms = float(r["median_ms"])
                cases.append({"id": cand, "kops_ms": kms, "ref": name,
                              "ref_ms": rms, "ratio": rms / kms if kms else None})
    ratios = [c["ratio"] for c in cases if c.get("ratio") is not None]
    geomean = math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios)) if ratios else None
    payload = {"family": family, "kops": str(kops_path), "cases": cases,
               "geomean": geomean,
               "note": "mega vs modular_graph; ratio >1 means mega is faster"}
    dump_json(out, payload)
    return out


def coverage_for(state: RunState, family: str, impls: list[str],
                 externals: list[str], args) -> dict:
    rows = {}
    for impl in impls:
        p = state.thirdparty / impl / f"{family}.json"
        doc = load_json(p)
        if doc is None:
            rows[impl] = {"results": 0, "skipped": 0, "status": "missing"}
        else:
            rows[impl] = {"results": len(doc.get("results", [])),
                          "skipped": len(doc.get("skipped", [])), "status": "ok"}
    for style in args.torch_styles:
        p = state.torch / style / f"{family}.json"
        doc = load_json(p)
        if doc is not None:
            rows[f"torch-{style}"] = {"results": len(doc.get("results", [])),
                                      "skipped": len(doc.get("skipped", [])), "status": "ok"}
    for name in externals:
        p = state.external / name / f"{family}.json"
        doc = load_json(p)
        rows[name] = ({"results": len(doc.get("results", [])),
                       "skipped": len(doc.get("skipped", [])), "status": "ok"}
                      if doc is not None else {"results": 0, "skipped": 0, "status": "missing"})
    return rows


def build_summary(state: RunState, env_info: dict, args, selected_families: list[str],
                  coverage: dict, cases_per_family: dict) -> dict:
    fam_summaries = {}
    all_ratios = []
    total_cases = 0
    total_with_ref = 0
    for family in selected_families:
        doc = load_json(state.compare / f"{family}.json") or {}
        cases = doc.get("cases", [])
        ratios = [c["ratio"] for c in cases if c.get("ratio") is not None]
        all_ratios += ratios
        total_cases += cases_per_family.get(family, 0)
        total_with_ref += len(ratios)
        gm = math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios)) if ratios else None
        fam_summaries[family] = {
            "cases": cases_per_family.get(family, 0),
            "with_ref": len(ratios),
            "geomean": gm,
            "wins": sum(1 for x in ratios if x > 1.0),
            "losses": sum(1 for x in ratios if x < 1.0),
            "ties": sum(1 for x in ratios if x == 1.0),
            "coverage": coverage.get(family, {}),
            "status": FAMILIES[family].get("status", ""),
            "note": FAMILIES[family].get("note", ""),
        }
    gm_all = math.exp(sum(math.log(max(x, 1e-12)) for x in all_ratios) / len(all_ratios)) if all_ratios else None
    return {
        "schema": 1,
        "run_id": state.out_dir.name,
        "env": env_info,
        "args": {
            "rounds": args.rounds, "flush_l2": args.flush_l2, "torch": args.torch,
            "torch_styles": args.torch_styles, "impl": args.impl,
            "external": args.external, "mode": args.mode,
            "skip_large": args.skip_large,
        },
        "families": fam_summaries,
        "totals": {
            "cases": total_cases,
            "with_ref": total_with_ref,
            "geomean": gm_all,
            "wins": sum(1 for x in all_ratios if x > 1.0),
            "losses": sum(1 for x in all_ratios if x < 1.0),
            "ties": sum(1 for x in all_ratios if x == 1.0),
        },
        "errors": state.errors,
        "skipped": state.skipped,
        "known_references": [
            {"family": f, "name": n, "status": s, "hardware": h, "note": note}
            for f, n, s, h, note in KNOWN_REFERENCES
        ],
    }


def render_report(state: RunState, summary: dict, args) -> Path:
    out = state.out_dir / "report.md"
    env = summary["env"]
    geo = f"{summary['totals']['geomean']:.2f}x" if summary["totals"]["geomean"] else "n/a"
    lines = [
        "# Third-party benchmark report",
        "",
        f"- run: `{summary['run_id']}`",
        f"- GPU: **{env['gpu']['name']}** ({env['gpu']['arch']}, cc {env['gpu']['cc']}, "
        f"{env['gpu']['vram_gb']:.1f} GiB)",
        f"- commit: `{env['git']['commit'][:12]}` dirty={env['git']['dirty']}",
        f"- python: `{env['python']}`",
        f"- rounds: {args.rounds}, flush_l2: {args.flush_l2} MiB, mode: {args.mode}",
        "",
        "> Third-party numbers are comparisons, not registry evidence. "
        "Registry evidence is the C++ JSON with `dirty=false`.",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {summary['totals']['cases']} |",
        f"| cases with third-party ref | {summary['totals']['with_ref']} |",
        f"| geomean (best 3rd / kops) | {geo} |",
        f"| wins / losses / ties | {summary['totals']['wins']} / "
        f"{summary['totals']['losses']} / {summary['totals']['ties']} |",
        "",
        "`ratio = best third-party median / kops median`; >1 means kops is faster.",
        "",
        "## Coverage",
        "",
        "| family | cases | with ref | geomean | status | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for family, s in summary["families"].items():
        gm = f"{s['geomean']:.2f}x" if s["geomean"] else "—"
        lines.append(f"| `{family}` | {s['cases']} | {s['with_ref']} | {gm} | "
                     f"{s['status']} | {s['note']} |")
    lines += ["", "## Per-family results", ""]
    for family in summary["families"]:
        doc = load_json(state.compare / f"{family}.json") or {}
        cases = doc.get("cases", [])
        if not cases:
            lines += [f"### {family}", "", "No comparison rows.", ""]
            continue
        lines += [f"### {family}", "",
                  "| case | kops ms | best 3rd | ratio | kops GB/s | 3rd GB/s | note |",
                  "|---|---:|---|---:|---:|---:|---|"]
        for c in cases:
            ratio = f"{c['ratio']:.2f}x" if c.get("ratio") is not None else "—"
            ref = c.get("ref") or "—"
            ref_ms = f"{c['ref_ms']:.4f}" if c.get("ref_ms") is not None else "—"
            lines.append(f"| `{c['id']}` | {c.get('kops_ms', 0):.4f} | {ref} {ref_ms} | "
                         f"{ratio} | {c.get('kops_gbs', 0) or 0:.1f} | "
                         f"{c.get('ref_gbs', 0) or 0:.1f} | {c.get('ref_note', '') or ''} |")
        lines.append("")
    lines += ["## Known references not wired", "",
              "| family | reference | status | hardware | note |",
              "|---|---|---|---|---|"]
    for r in summary["known_references"]:
        lines.append(f"| `{r['family']}` | {r['name']} | {r['status']} | "
                     f"{r['hardware']} | {r['note']} |")
    lines += ["", "## Skipped / errors", ""]
    if summary["skipped"]:
        lines += ["| family | impl | reason |", "|---|---|---|"]
        for s in summary["skipped"]:
            lines.append(f"| `{s.get('family','')}` | `{s.get('impl','')}` | "
                         f"{s.get('reason','')} |")
    else:
        lines.append("None.")
    if summary["errors"]:
        lines += ["", "### Command errors", "", "| name | rc | error | log |",
                  "|---|---:|---|---|"]
        for e in summary["errors"]:
            lines.append(f"| `{e['name']}` | {e['rc']} | {e['error']} | `{e['log']}` |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_kops(build_dir: Path, args) -> int:
    cmake = shutil.which("cmake")
    if cmake is None:
        log("[run][error] cmake not found on PATH")
        return 127
    arch = args.arch if args.arch != "auto" else "native"
    cmd = [cmake, "-S", str(REPO_ROOT), "-B", str(build_dir), "-G", "Ninja",
           "-DCMAKE_BUILD_TYPE=Release", "-DKOPS_BUILD_BENCHMARKS=ON",
           "-DKOPS_BUILD_TESTS=OFF", f"-DKOPS_CUDA_ARCHS={arch}",
           "-DKOPS_USE_CCACHE=OFF"]
    log("[run] configure: " + " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        return r.returncode
    cmd2 = [cmake, "--build", str(build_dir), "-j", str(args.build_jobs)]
    log("[run] build: " + " ".join(shlex.quote(c) for c in cmd2))
    return subprocess.run(cmd2, cwd=REPO_ROOT).returncode


def print_check(gpu: dict, build_dir: Path | None, py: dict, ext: dict, args) -> int:
    log(f"GPU: {gpu['name']} ({gpu['cc']}, {gpu['vram_gb']:.1f} GiB, {gpu['arch']})")
    log(f"repo: {REPO_ROOT}")
    log(f"build dir: {build_dir or 'NOT FOUND'}")
    log("")
    log("Python adapters:")
    for name in PY_IMPLS:
        info = py.get(name, {})
        log(f"  {name:<12} {'OK' if info.get('available') else 'missing':<8} {info.get('reason','')}")
    log("")
    log("External harnesses:")
    for name in EXTERNALS:
        info = ext[name]
        log(f"  {name:<18} {'OK' if info['available'] else 'missing':<8} "
            f"{info['path'] or 'build: ' + info['build']}")
    log("")
    log("Family matrix:")
    for family, meta in FAMILIES.items():
        log(f"  {family:<20} {meta['status']:<22} impls={meta['impls']} external={meta['external']}")
    missing = []
    if args.impl != "all":
        for name in [x.strip() for x in args.impl.split(",") if x.strip()]:
            if not py.get(name, {}).get("available"):
                missing.append(f"impl:{name}")
    if args.external not in ("auto", "none", "all"):
        for name in [x.strip() for x in args.external.split(",") if x.strip()]:
            if not ext.get(name, {}).get("available"):
                missing.append(f"external:{name}")
    if missing:
        log("")
        log("REQUESTED BUT MISSING: " + ", ".join(missing))
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every family (default)")
    ap.add_argument("--families", default=None, help="comma-separated family names")
    ap.add_argument("--only", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--skip", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--impl", default="all",
                    help="all or comma list: " + ",".join(PY_IMPLS))
    ap.add_argument("--external", default="auto",
                    help="auto, none, all, or comma list of external harness names")
    ap.add_argument("--torch", action="store_true", help="also run PyTorch baseline")
    ap.add_argument("--torch-style", default="hf", choices=("builtin", "hf", "both"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--flush-l2", type=int, default=256)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--mode", default="graph", choices=("graph", "eager"))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--build", action="store_true", help="configure/build kops benchmarks")
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--build-jobs", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--kops-dir", default=None, help="reuse kops JSONs from this directory")
    ap.add_argument("--cases-dir", default=None, help="reuse dumped case JSONs from this directory")
    ap.add_argument("--no-kops", action="store_true")
    ap.add_argument("--json-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--skip-large", dest="skip_large", action="store_true", default=None)
    ap.add_argument("--no-skip-large", dest="skip_large", action="store_false")
    ap.add_argument("--max-case-bytes", type=int, default=None)
    ap.add_argument("--comm-calls", type=int, default=8)
    ap.add_argument("--comm-groups", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    gpu = detect_gpu()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    build_dir = find_build_dir(args.build_dir)
    py = python_impls()
    ext = external_paths()

    if args.list:
        for family, meta in FAMILIES.items():
            log(f"{family:<20} {meta['status']:<22} target={meta['target']} "
                f"impls={meta['impls']} external={meta['external']}")
        return 0
    if args.check:
        return print_check(gpu, build_dir, py, ext, args)

    if args.families:
        selected = [x.strip() for x in args.families.split(",") if x.strip()]
        unknown = [x for x in selected if x not in FAMILIES]
        if unknown:
            log(f"[run][error] unknown families: {unknown}")
            return 2
    else:
        selected = FAMILY_ORDER

    if args.build:
        if build_dir is None:
            build_dir = REPO_ROOT / "build-wsl" / "bench"
        rc = build_kops(build_dir, args)
        if rc != 0:
            log(f"[run][error] build failed rc={rc}")
            return rc
        build_dir = find_build_dir(str(build_dir))

    if build_dir is None:
        log("[run][error] no kops benchmark build found. Use --build or --build-dir.")
        return 2

    # Requested adapters.
    if args.impl == "all":
        requested_impls = [x for x in PY_IMPLS if py.get(x, {}).get("available")]
    else:
        requested_impls = [x.strip() for x in args.impl.split(",") if x.strip()]
        unknown = [x for x in requested_impls if x not in PY_IMPLS]
        if unknown:
            log(f"[run][error] unknown impls: {unknown}")
            return 2
        missing = [x for x in requested_impls if not py.get(x, {}).get("available")]
        if missing and args.strict:
            log(f"[run][error] requested impls missing: {missing}")
            return 2

    if args.external == "auto":
        requested_external = [x for x in EXTERNALS if ext[x]["available"]]
    elif args.external == "none":
        requested_external = []
    elif args.external == "all":
        requested_external = list(EXTERNALS.keys())
        if args.strict:
            missing = [x for x in requested_external if not ext[x]["available"]]
            if missing:
                log(f"[run][error] requested externals missing: {missing}")
                return 2
    else:
        requested_external = [x.strip() for x in args.external.split(",") if x.strip()]
        unknown = [x for x in requested_external if x not in EXTERNALS]
        if unknown:
            log(f"[run][error] unknown externals: {unknown}")
            return 2
        missing = [x for x in requested_external if not ext[x]["available"]]
        if missing and args.strict:
            log(f"[run][error] requested externals missing: {missing}")
            return 2

    if args.torch_style == "both":
        args.torch_styles = ["builtin", "hf"]
    else:
        args.torch_styles = [args.torch_style]

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_root = REPO_ROOT / "out" / "thirdparty" / f"{gpu['arch']}-{sanitize(gpu['name'])}-{run_id}"
    out_dir = Path(args.json_dir).resolve() if args.json_dir else default_root
    state = RunState(
        out_dir=out_dir,
        logs=out_dir / "logs",
        cases=out_dir / "cases",
        kops=out_dir / "kops",
        thirdparty=out_dir / "thirdparty",
        torch=out_dir / "torch",
        external=out_dir / "external",
        compare=out_dir / "compare",
    )
    for d in (state.logs, state.cases, state.kops, state.thirdparty,
              state.torch, state.external, state.compare):
        d.mkdir(parents=True, exist_ok=True)

    env_info = {
        "gpu": gpu,
        "git": git_info(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "platform": sys.platform,
        "cwd": str(REPO_ROOT),
        "build_dir": str(build_dir),
        "adapters": py,
        "externals": ext,
        "argv": sys.argv,
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    dump_json(out_dir / "env.json", env_info)
    log(f"[run] output: {out_dir}")
    log(f"[run] GPU: {gpu['name']} ({gpu['arch']}, {gpu['vram_gb']:.1f} GiB)")

    env = subprocess_env(args.device)
    coverage: dict[str, dict] = {}
    cases_per_family: dict[str, int] = {}
    all_case_ids: dict[str, set[str]] = {}

    for family in selected:
        meta = FAMILIES[family]
        binary = kops_bin(build_dir, family)
        if binary is None and not args.cases_dir:
            state.skipped.append({"family": family, "impl": "kops",
                                  "reason": f"binary missing: {meta['target']}"})
            continue
        log(f"\n[run] {family}")
        if args.cases_dir:
            raw_cases_path = Path(args.cases_dir) / f"{family}.raw.json"
            if not raw_cases_path.exists():
                raw_cases_path = Path(args.cases_dir) / f"{family}.json"
        else:
            raw_cases_path = state.cases / f"{family}.raw.json"
        if (not args.cases_dir) and (not raw_cases_path.exists() or args.force):
            cmd = [str(binary), "--dump-cases"]
            if args.only:
                cmd += ["--only", args.only]
            if args.skip:
                cmd += ["--skip", args.skip]
            r = run_cmd(state, f"cases/{family}", cmd,
                        state.logs / f"cases-{family}.log", 300, args.dry_run,
                        env, capture_stdout=raw_cases_path)
            if r.returncode != 0:
                state.skipped.append({"family": family, "impl": "cases",
                                      "reason": r.error or f"rc={r.returncode}"})
                continue
        if args.dry_run and not raw_cases_path.exists():
            dump_json(raw_cases_path, {"schema": 1, "backend": "dry-run", "cases": [
                {"id": f"dry.{family}.case", "op": "dry", "dtype": "float32",
                 "rows": 1, "cols": 1, "bytes": 8, "flops": 0, "warmup": 1, "iters": 1}
            ]})
        raw = load_json(raw_cases_path)
        if raw is None:
            state.skipped.append({"family": family, "impl": "cases",
                                  "reason": "could not dump cases"})
            continue
        selected_cases, dropped, total = filter_cases(raw, args, gpu["vram_gb"])
        cases_per_family[family] = len(selected_cases)
        case_ids = {c["id"] for c in selected_cases}
        raw_ids = {c.get("id", "") for c in raw.get("cases", [])}
        skip_ids = raw_ids - case_ids
        all_case_ids[family] = case_ids
        cases_path = write_cases(state, family, selected_cases)
        if dropped:
            log(f"[run] {family}: {len(selected_cases)}/{total} cases "
                f"({dropped} large cases skipped)")

        # kops
        kops_path = None
        if args.kops_dir:
            src = Path(args.kops_dir) / f"{family}.json"
            doc = load_json(src)
            if doc is not None:
                filter_results_inplace(doc, case_ids)
                kops_path = state.kops / f"{family}.json"
                dump_json(kops_path, doc)
        elif not args.no_kops and binary is not None:
            kops_path = run_kops(state, family, binary, cases_path, args, env,
                                 case_ids, skip_ids)

        refs: list[tuple[str, Path]] = []
        for impl in meta["impls"]:
            if impl not in requested_impls:
                continue
            p = run_python_baseline(state, family, impl, cases_path, args, env)
            if p is not None:
                refs.append((f"thirdparty:{impl}", p))
        if args.torch:
            for style in args.torch_styles:
                p = run_torch_baseline(state, family, style, cases_path, args, env)
                if p is not None:
                    refs.append((f"torch:{style}", p))
        if meta.get("special") == "csa":
            p = run_csa(state, family, cases_path, args, env)
            if p is not None:
                refs.append(("torch:csa", p))
        if meta.get("special") == "comm":
            p = run_comm_torch(state, family, cases_path, args, env)
            if p is not None:
                refs.append(("torch:nccl", p))
        for name in meta["external"]:
            if name not in requested_external or not ext[name]["available"]:
                continue
            p = run_external(state, family, name, Path(ext[name]["path"]),
                             cases_path, args, env, case_ids)
            if p is not None:
                refs.append((f"external:{name}", p))

        coverage[family] = coverage_for(state, family, meta["impls"],
                                        meta["external"], args)
        compare_family(state, family, kops_path, refs, args, env)

    summary = build_summary(state, env_info, args, selected, coverage, cases_per_family)
    dump_json(out_dir / "summary.json", summary)
    report = Path(args.report) if args.report else (out_dir / "report.md")
    if not report.is_absolute():
        report = (REPO_ROOT / report).resolve() if str(report).startswith("out") else (out_dir / report)
    report.parent.mkdir(parents=True, exist_ok=True)
    out_report = render_report(state, summary, args)
    if out_report != report:
        shutil.copyfile(out_report, report)
    log(f"\n[run] summary: {out_dir / 'summary.json'}")
    log(f"[run] report:  {report}")
    if summary["totals"]["geomean"]:
        log(f"[run] geomean best-3rd/kops: {summary['totals']['geomean']:.2f}x over "
            f"{summary['totals']['with_ref']}/{summary['totals']['cases']} cases")
    if args.strict and (state.errors or state.skipped):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
