#!/usr/bin/env python3
"""Preflight checks for a kernel-ops handoff kit.

Stdlib only: no torch, no third-party packages, no network. The script checks
the GPU, toolkit, CTest, kit binaries and the golden manifest, prints one
PASS/WARN/FAIL line per check, and exits nonzero on a hard mismatch.

    ./check_env.py --arch sm90a --kit .
    ./check_env.py --arch sm100a --kit /data/kits/sm100a --require-golden
    ./check_env.py --arch sm90a --kit . --json results/check-env.json

Hard failures: wrong compute capability, ctest missing, missing/invalid kit
binaries, missing shared libraries, a present nvcc that is too old or lacks the
exact target, and a malformed golden manifest or a ready case whose hash/header
is wrong. A missing golden directory is a warning unless --require-golden is
passed, because the C++ tests intentionally return 77 (CTest skip) without
goldens. A missing nvcc is also a warning: a prebuilt kit can still run, but it
cannot be rebuilt on that machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ARCH_INFO = {
    "sm90a": {
        "cc": (9, 0),
        "gpu": "H100/H200/H20",
        "min_cuda": (12, 0),
        "rec_cuda": (12, 9),
        "min_driver": 525,
        "elf": "sm_90a",
    },
    "sm100a": {
        "cc": (10, 0),
        "gpu": "B200/GB200",
        "min_cuda": (12, 8),
        "rec_cuda": (13, 0),
        "min_driver": 570,
        "elf": "sm_100a",
    },
}

DTYPE_SIZE = {"fp32": 4, "bf16": 2, "fp16": 2, "u8": 1}
DTYPE_BY_CODE = {0: "fp32", 1: "bf16", 2: "fp16", 3: "u8"}
CODEC_BY_CODE = {
    0: "native",
    1: "fp8_per_tensor",
    2: "fp8_per_head",
    3: "fp8_per_token_head",
    4: "int8_per_token_head",
    5: "int4_per_token_head",
    6: "nvfp4",
    7: "mla_fp8_ds",
    8: "mla_nvfp4_ds",
    9: "lowbit_wht",
    10: "indexer_fp8",
    11: "indexer_fp4",
    12: "mla_fp8_ds_fnuz",
    13: "mla_fp8_ds_v40",
    14: "mla_fp8_ds_v41",
    15: "mla_fp8_ds_v41_fp4",
}
CODEC_BY_NAME = {name: code for code, name in CODEC_BY_CODE.items()}
TIERS = {"smoke", "standard", "full"}
MAGIC = b"KOPSGOLD"
HEADER_FIXED = 96


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failures = 0
        self.warnings = 0

    def add(self, status: str, name: str, detail: str) -> None:
        self.checks.append({"name": name, "status": status, "detail": detail})
        if status == "FAIL":
            self.failures += 1
        elif status == "WARN":
            self.warnings += 1
        print(f"[{status}] {name:<22} {detail}")

    def ok(self, name: str, detail: str) -> None:
        self.add("PASS", name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add("WARN", name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add("FAIL", name, detail)


def run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timed out after {timeout}s"
    except OSError as exc:
        return 126, f"{cmd[0]}: {exc}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt_version(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def parse_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_kit(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve()


def detect_arch(rep: Report, kit: Path, explicit: str | None) -> str | None:
    kit_arch = None
    manifest = kit / "manifest.json"
    if manifest.is_file():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            value = doc.get("arch")
            if isinstance(value, str):
                kit_arch = value
        except (OSError, ValueError) as exc:
            rep.warn("kit.manifest", f"cannot read {manifest.name}: {exc}")

    if explicit and kit_arch and explicit != kit_arch:
        rep.fail(
            "kit.arch",
            f"kit manifest says arch={kit_arch}, but --arch={explicit}",
        )
        return explicit
    if explicit:
        return explicit
    if kit_arch:
        return kit_arch
    if kit.name in ARCH_INFO:
        return kit.name
    rep.fail(
        "kit.arch",
        "cannot determine arch from manifest or directory name; pass --arch",
    )
    return None


def check_kit_manifest(rep: Report, kit: Path, arch: str) -> dict:
    path = kit / "manifest.json"
    if not path.is_file():
        rep.warn(
            "kit.manifest",
            f"{path} absent (checking binaries and golden directly)",
        )
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        rep.fail("kit.manifest", f"cannot parse {path.name}: {exc}")
        return {}
    if not isinstance(doc, dict):
        rep.fail("kit.manifest", "manifest root is not an object")
        return {}

    rep.ok(
        "kit.manifest",
        f"schema={doc.get('schema')} preset={doc.get('preset')} "
        f"commit={str(doc.get('source_commit'))[:12]}",
    )
    if doc.get("schema") != 1:
        rep.fail("kit.schema", f"schema={doc.get('schema')!r}, expected 1")
    if doc.get("arch") != arch:
        rep.fail("kit.arch", f"manifest arch={doc.get('arch')!r}, expected {arch!r}")
    if kit.name in ARCH_INFO and kit.name != arch:
        rep.warn("kit.path", f"directory is named {kit.name}, manifest says {arch}")
    dirty = doc.get("source_dirty")
    if dirty is True:
        rep.warn(
            "kit.clean",
            "kit was built from a dirty tree; commit before using results as evidence",
        )
    elif dirty is False:
        rep.ok("kit.clean", "source_dirty=False")
    else:
        rep.warn("kit.clean", "source_dirty unknown; build from a git checkout")

    files = doc.get("files")
    if not isinstance(files, list) or not files:
        rep.warn("kit.hashes", "manifest has no file hash list")
        return doc

    bad: list[str] = []
    checked = 0
    for item in files:
        if not isinstance(item, dict):
            bad.append("non-object file entry")
            continue
        rel = item.get("path")
        expected = item.get("sha256")
        if not isinstance(rel, str) or not isinstance(expected, str):
            bad.append(f"invalid entry {rel!r}")
            continue
        candidate = (kit / rel).resolve()
        try:
            candidate.relative_to(kit)
        except ValueError:
            bad.append(f"{rel}: escapes kit")
            continue
        if not candidate.is_file():
            bad.append(f"{rel}: missing")
            continue
        actual = sha256_file(candidate)
        checked += 1
        if actual != expected:
            bad.append(f"{rel}: sha256 {actual[:12]} != {expected[:12]}")
    if bad:
        rep.fail(
            "kit.hashes",
            f"{len(bad)} of {len(files)} file(s) bad: " + "; ".join(bad[:3]),
        )
    else:
        rep.ok("kit.hashes", f"{checked} file(s) match their SHA-256")

    tests = doc.get("test_executables")
    benches = doc.get("benchmark_executables")
    if isinstance(tests, list):
        rep.ok("kit.tests.listed", f"{len(tests)} test executable(s) recorded")
    if isinstance(benches, list):
        rep.ok("kit.benches.listed", f"{len(benches)} benchmark executable(s) recorded")
    return doc


def parse_gpu_rows(text: str, with_cc: bool) -> list[dict]:
    rows: list[dict] = []
    for raw in csv.reader(text.splitlines()):
        row = [cell.strip() for cell in raw]
        if not row or not any(row):
            continue
        if with_cc and len(row) >= 4:
            # The GPU name may itself contain commas; the last three fields
            # are compute capability, memory and driver version.
            name = ", ".join(row[:-3])
            cc, memory, driver = row[-3], row[-2], row[-1]
            match = re.search(r"(\d+)\.(\d+)", cc)
            cc_tuple = (int(match.group(1)), int(match.group(2))) if match else None
        elif len(row) >= 3:
            name = ", ".join(row[:-2])
            cc_tuple = None
            memory, driver = row[-2], row[-1]
        else:
            continue
        try:
            memory_mib = int(float(memory))
        except ValueError:
            memory_mib = 0
        rows.append(
            {
                "name": name,
                "cc": cc_tuple,
                "cc_text": f"{cc_tuple[0]}.{cc_tuple[1]}" if cc_tuple else "unknown",
                "memory_mib": memory_mib,
                "driver": driver,
            }
        )
    return rows


def check_gpu(rep: Report, info: dict, toolkit_text: str) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        rep.fail("gpu.nvidia-smi", "nvidia-smi not found (no NVIDIA driver?)")
        return

    rc, out = run(
        [
            nvidia_smi,
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = parse_gpu_rows(out, with_cc=True) if rc == 0 else []
    if not rows:
        rc2, out2 = run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = parse_gpu_rows(out2, with_cc=False) if rc2 == 0 else []
        if rows:
            for row in rows:
                row["cc"] = None
                row["cc_text"] = "unknown"
        else:
            rep.fail("gpu.query", f"nvidia-smi query failed: {out or out2}")
            return

    summary = "; ".join(
        f"{row['name']} ({row['cc_text']}, {row['memory_mib']} MiB)" for row in rows
    )
    rep.ok("gpu.detect", f"{len(rows)} GPU(s): {summary}")

    expected = info["cc"]
    matches = [row for row in rows if row["cc"] == expected]
    if matches:
        gpu = matches[0]
        rep.ok(
            "gpu.arch",
            f"{gpu['name']}: compute capability {gpu['cc_text']} "
            f"(expected {expected[0]}.{expected[1]})",
        )
    else:
        found = ", ".join(f"{row['name']}={row['cc_text']}" for row in rows)
        rep.fail(
            "gpu.arch",
            f"no GPU at compute capability {expected[0]}.{expected[1]} "
            f"({info['gpu']}); found: {found}",
        )

    driver_text = rows[0]["driver"]
    driver_match = re.search(r"(\d+)", driver_text)
    if not driver_match:
        rep.warn("gpu.driver", f"cannot parse driver version {driver_text!r}")
    else:
        driver_major = int(driver_match.group(1))
        required = info["min_driver"]
        toolkit = parse_version(toolkit_text)
        if toolkit is not None:
            if toolkit[0] >= 13:
                required = max(required, 580)
            elif toolkit[0] == 12:
                required = max(required, 525)
        if driver_major < required:
            rep.fail(
                "gpu.driver",
                f"driver {driver_text} is older than the {required} minimum "
                f"for {info['gpu']} with CUDA {toolkit_text or 'unknown'}",
            )
        else:
            rep.ok(
                "gpu.driver",
                f"driver {driver_text} (>= {required} required)"
                + (f" for the kit's CUDA {toolkit_text}" if toolkit else ""),
            )

    max_memory = max(row["memory_mib"] for row in rows)
    if max_memory and max_memory < 16 * 1024:
        rep.warn(
            "gpu.memory",
            f"{max_memory} MiB visible; target cases may need more VRAM",
        )
    elif max_memory:
        rep.ok("gpu.memory", f"{max_memory} MiB visible")


def probe_nvcc_arch(nvcc: str, target: str) -> tuple[bool, str]:
    """Compile a trivial kernel for the exact `a` target.

    `nvcc --list-gpu-arch` lists base targets such as compute_90/compute_100 but
    not the architecture-specific `compute_90a`/`compute_100a` spellings, so the
    only reliable support check is a small compile.
    """
    source = "__global__ void kops_arch_probe() {}\n"
    try:
        with tempfile.TemporaryDirectory(prefix="kops-arch-") as tmp:
            src = Path(tmp) / "probe.cu"
            cubin = Path(tmp) / "probe.cubin"
            src.write_text(source, encoding="utf-8")
            rc, out = run(
                [nvcc, f"-arch={target}", "-cubin", "-o", str(cubin), str(src)],
                timeout=120,
            )
            if rc == 0 and cubin.is_file() and cubin.stat().st_size > 0:
                return True, f"{target} compiles"
            detail = out.strip().splitlines()
            tail = detail[-1] if detail else f"nvcc exited {rc}"
            return False, f"{target} compile probe failed: {tail[:200]}"
    except OSError as exc:
        return False, f"cannot create the probe workspace: {exc}"


def check_nvcc(rep: Report, arch: str, info: dict) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for env_name in ("CUDA_HOME", "CUDA_PATH"):
            candidate = os.environ.get(env_name)
            if candidate and (Path(candidate) / "bin" / "nvcc").is_file():
                nvcc = str(Path(candidate) / "bin" / "nvcc")
                break
    if nvcc is None:
        rep.warn(
            "nvcc",
            "nvcc not found; the prebuilt kit can run, but it cannot be rebuilt here",
        )
        return

    rc, out = run([nvcc, "--version"])
    if rc != 0:
        rep.fail("nvcc.version", f"nvcc --version failed: {out}")
    else:
        match = re.search(r"release\s+(\d+)\.(\d+)", out)
        if not match:
            rep.warn("nvcc.version", "cannot parse the release line")
        else:
            version = (int(match.group(1)), int(match.group(2)))
            if version < info["min_cuda"]:
                rep.fail(
                    "nvcc.version",
                    f"CUDA {fmt_version(version)} is older than the "
                    f"{fmt_version(info['min_cuda'])} minimum for {arch}",
                )
            elif version < info["rec_cuda"]:
                rep.warn(
                    "nvcc.version",
                    f"CUDA {fmt_version(version)}; {fmt_version(info['rec_cuda'])} "
                    f"or newer is recommended for {arch}",
                )
            else:
                rep.ok("nvcc.version", f"CUDA {fmt_version(version)} ({nvcc})")

    wanted = arch[2:]  # sm90a -> 90a, sm100a -> 100a
    target = f"sm_{wanted}"
    rc, out = run([nvcc, "--list-gpu-arch"], timeout=60)
    listed = set(re.findall(r"compute_(\d+[a-z]?)", out)) if rc == 0 else set()
    base = re.sub(r"[a-z]$", "", wanted)
    ok, detail = probe_nvcc_arch(nvcc, target)
    if ok:
        if base in listed:
            detail += f" (list-gpu-arch reports compute_{base})"
        rep.ok("nvcc.arch", detail)
    else:
        rep.fail("nvcc.arch", detail)


def check_ctest(rep: Report) -> None:
    ctest = shutil.which("ctest")
    if ctest is None:
        rep.fail("ctest", "ctest not found (install CMake >= 3.24)")
        return
    rc, out = run([ctest, "--version"])
    line = out.splitlines()[0] if out else "unknown"
    if rc == 0:
        rep.ok("ctest", f"{line} ({ctest})")
    else:
        rep.fail("ctest", f"ctest --version failed: {out}")


def inspect_golden_header(path: Path, case: dict) -> list[str]:
    errors: list[str] = []
    size = path.stat().st_size
    if size < HEADER_FIXED:
        return [f"file is only {size} bytes, shorter than the 96-byte header"]
    with path.open("rb") as handle:
        header = handle.read(HEADER_FIXED)
        if len(header) < HEADER_FIXED:
            return ["short read of the fixed header"]

        magic = header[:8]
        version, header_bytes = struct.unpack_from("<HH", header, 8)
        dtype_code, rank, flags = struct.unpack_from("<III", header, 12)
        seed = struct.unpack_from("<Q", header, 24)[0]
        codec, topk, sink = struct.unpack_from("<IIf", header, 32)
        reserved = struct.unpack_from("<I", header, 44)[0]
        (
            output_offset,
            output_bytes,
            lse_offset,
            lse_bytes,
            meta_offset,
            meta_bytes,
        ) = struct.unpack_from("<QQQQQQ", header, 48)

        if magic != MAGIC:
            errors.append(f"magic is {magic!r}, expected {MAGIC!r}")
        if version != 1:
            errors.append(f"version {version} is not 1")
        if rank > 8:
            errors.append(f"rank {rank} exceeds 8")
        if header_bytes != HEADER_FIXED + 8 * rank:
            errors.append(
                f"header_bytes {header_bytes} != 96 + 8 * rank ({HEADER_FIXED + 8 * rank})"
            )
        if dtype_code not in DTYPE_BY_CODE:
            errors.append(f"dtype code {dtype_code} is not fp32/bf16/fp16/u8")
        if flags & ~0x3:
            errors.append(f"flags 0x{flags:x} use reserved bits")
        if reserved != 0:
            errors.append(f"reserved0 is {reserved}, expected 0")
        if codec not in CODEC_BY_CODE:
            errors.append(f"codec {codec} is not a known KVCodec")
        if not math.isfinite(sink):
            errors.append(f"sink {sink} is not finite")
        if header_bytes > size:
            errors.append(f"header_bytes {header_bytes} exceeds file size {size}")

        shape: tuple[int, ...] = ()
        if rank <= 8:
            handle.seek(HEADER_FIXED)
            raw_shape = handle.read(8 * rank)
            if len(raw_shape) != 8 * rank:
                errors.append("short read of shape[]")
            else:
                shape = struct.unpack(f"<{rank}Q", raw_shape)
                if any(extent == 0 for extent in shape):
                    errors.append("shape contains a zero extent")

        if not errors and dtype_code in DTYPE_BY_CODE:
            expected_bytes = math.prod(shape) * DTYPE_SIZE[DTYPE_BY_CODE[dtype_code]]
            if output_bytes != expected_bytes:
                errors.append(
                    f"output_bytes {output_bytes} != product(shape)*dtype_size "
                    f"({expected_bytes})"
                )

        if output_offset < header_bytes:
            errors.append(f"output_offset {output_offset} < header_bytes {header_bytes}")
        if output_offset + output_bytes > size:
            errors.append("output section extends past end of file")
        if flags & 0x1:
            expected_lse = (math.prod(shape[:-1]) if shape else 1) * 4
            if lse_bytes != expected_lse:
                errors.append(f"lse_bytes {lse_bytes} != {expected_lse}")
            if lse_offset < header_bytes:
                errors.append(f"lse_offset {lse_offset} < header_bytes {header_bytes}")
            if lse_offset + lse_bytes > size:
                errors.append("LSE section extends past end of file")
        else:
            if lse_bytes != 0 or lse_offset != 0:
                errors.append("LSE offsets are set but HAS_LSE is clear")
        if meta_bytes:
            if meta_offset < header_bytes:
                errors.append(f"meta_offset {meta_offset} < header_bytes {header_bytes}")
            if meta_offset + meta_bytes > size:
                errors.append("metadata section extends past end of file")
            else:
                handle.seek(meta_offset)
                raw_meta = handle.read(meta_bytes)
                try:
                    json.loads(raw_meta.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    errors.append(f"metadata is not UTF-8 JSON: {exc}")

        sections = [("output", output_offset, output_bytes)]
        if flags & 0x1:
            sections.append(("LSE", lse_offset, lse_bytes))
        if meta_bytes:
            sections.append(("metadata", meta_offset, meta_bytes))
        for left in range(len(sections)):
            name_a, start_a, bytes_a = sections[left]
            for right in range(left + 1, len(sections)):
                name_b, start_b, bytes_b = sections[right]
                if start_a < start_b + bytes_b and start_b < start_a + bytes_a:
                    errors.append(f"{name_a} and {name_b} sections overlap")

        # Cross-check the self-describing header against the manifest so a
        # stale manifest cannot point at a valid file with different physics.
        if case:
            if dtype_code in DTYPE_BY_CODE and DTYPE_BY_CODE[dtype_code] != case.get("dtype"):
                errors.append(
                    f"header dtype {DTYPE_BY_CODE[dtype_code]} != manifest "
                    f"{case.get('dtype')}"
                )
            if seed != case.get("seed"):
                errors.append(f"header seed {seed} != manifest {case.get('seed')}")
            if topk != case.get("topk"):
                errors.append(f"header topk {topk} != manifest {case.get('topk')}")
            if CODEC_BY_CODE.get(codec) != case.get("codec"):
                errors.append(
                    f"header codec {CODEC_BY_CODE.get(codec)} != manifest "
                    f"{case.get('codec')}"
                )
            output_shape = (case.get("shapes") or {}).get("output")
            if isinstance(output_shape, list) and list(shape) != output_shape:
                errors.append(
                    f"header shape {list(shape)} != manifest output {output_shape}"
                )
            expected_sink = case.get("sink")
            if isinstance(expected_sink, (int, float)) and abs(sink - expected_sink) > 1e-6:
                errors.append(f"header sink {sink} != manifest {expected_sink}")
            if "has_lse" in case and bool(case["has_lse"]) != bool(flags & 0x1):
                errors.append(
                    f"header HAS_LSE={bool(flags & 0x1)} != manifest "
                    f"has_lse={case['has_lse']}"
                )
    return errors


def check_golden(rep: Report, kit: Path, require: bool) -> None:
    golden_dir = kit / "golden"
    manifest = golden_dir / "manifest.json"
    if not manifest.is_file():
        detail = f"no golden manifest at {manifest}"
        if require:
            rep.fail("golden.manifest", detail)
        else:
            rep.warn("golden.manifest", detail + "; golden tests skip 77")
        return

    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        rep.fail("golden.manifest", f"cannot parse {manifest}: {exc}")
        return
    if not isinstance(doc, dict):
        rep.fail("golden.manifest", "manifest root is not an object")
        return

    version = doc.get("version")
    if version != 1:
        rep.fail("golden.version", f"version {version!r}, expected 1")
    else:
        rep.ok(
            "golden.version",
            f"version 1, generated_by={doc.get('generated_by')!r}, "
            f"source_commit={doc.get('source_commit')!r}",
        )
    generated_by = doc.get("generated_by")
    source_commit = doc.get("source_commit")
    if not isinstance(generated_by, str) or not generated_by:
        rep.fail("golden.provenance", "generated_by must be a non-empty string")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"pending|[0-9a-f]{7,40}", source_commit
    ):
        rep.fail("golden.provenance", f"source_commit {source_commit!r} is invalid")
    cases = doc.get("cases")
    if not isinstance(cases, list):
        rep.fail("golden.cases", "cases is not an array")
        return
    if not cases:
        rep.warn("golden.cases", "manifest has no cases")
        return

    errors: list[str] = []
    ready = 0
    planned = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"case[{index}] is not an object")
            continue
        case_id = case.get("id", f"case[{index}]")
        status = case.get("status")
        if status not in ("ready", "planned"):
            errors.append(f"{case_id}: status {status!r} is not ready/planned")
            continue

        for field in ("id", "op", "dtype", "shapes", "codec", "topk", "sink",
                      "seed", "golden", "sha256", "tolerance", "tier",
                      "on_demand_input"):
            if field not in case:
                errors.append(f"{case_id}: missing field {field}")
        if case.get("dtype") not in DTYPE_SIZE:
            errors.append(f"{case_id}: dtype {case.get('dtype')!r} is not supported")
        if case.get("codec") not in CODEC_BY_NAME:
            errors.append(f"{case_id}: codec {case.get('codec')!r} is not known")
        if case.get("tier") not in TIERS:
            errors.append(f"{case_id}: tier {case.get('tier')!r} is not smoke/standard/full")
        if not isinstance(case.get("on_demand_input"), bool):
            errors.append(f"{case_id}: on_demand_input must be boolean")
        shapes = case.get("shapes")
        if not isinstance(shapes, dict) or "output" not in shapes:
            errors.append(f"{case_id}: shapes.output is required")

        rel = case.get("golden")
        if not isinstance(rel, str):
            errors.append(f"{case_id}: golden path is not a string")
            continue
        if rel.startswith("/") or ".." in Path(rel).parts:
            errors.append(f"{case_id}: golden path {rel!r} is not relative/safe")
            continue
        path = (golden_dir / rel).resolve()
        try:
            path.relative_to(golden_dir.resolve())
        except ValueError:
            errors.append(f"{case_id}: golden path {rel!r} escapes the golden dir")
            continue

        if status == "planned":
            planned += 1
            if case.get("sha256") is not None:
                errors.append(f"{case_id}: planned case must not carry a sha256")
            continue

        ready += 1
        digest = case.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{case_id}: ready case needs a 64-char lowercase sha256")
            continue
        if not path.is_file():
            errors.append(f"{case_id}: missing golden file {rel}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            errors.append(f"{case_id}: sha256 {actual[:12]} != manifest {digest[:12]}")
            continue
        errors.extend(f"{case_id}: {item}" for item in inspect_golden_header(path, case))

    if ready and source_commit == "pending":
        errors.append("ready case(s) exist but source_commit is still 'pending'")

    label = f"{ready} ready / {planned} planned / {len(cases)} total"
    if errors:
        rep.fail(
            "golden.cases",
            f"{len(errors)} error(s) in {label}: " + "; ".join(errors[:4]),
        )
    elif ready == 0:
        rep.warn(
            "golden.cases",
            f"{label}; no ready case, so golden tests skip 77",
        )
    else:
        rep.ok("golden.cases", label)


def check_binaries(rep: Report, kit: Path, info: dict, manifest: dict) -> None:
    bin_dir = kit / "bin"
    if not bin_dir.is_dir():
        rep.fail("kit.bin", f"{bin_dir} is missing")
        return

    tests = manifest.get("test_executables")
    benches = manifest.get("benchmark_executables")
    if isinstance(tests, list):
        tests = [name for name in tests if isinstance(name, str)]
    else:
        tests = sorted(p.name for p in bin_dir.glob("kops_test_*") if p.is_file())
    if isinstance(benches, list):
        benches = [name for name in benches if isinstance(name, str)]
    else:
        benches = sorted(p.name for p in bin_dir.glob("bench_*") if p.is_file())

    bad: list[str] = []
    for name in tests:
        path = bin_dir / name
        if not path.is_file():
            bad.append(f"{name}: missing")
        elif not os.access(path, os.X_OK):
            bad.append(f"{name}: not executable")
    if not tests:
        rep.fail("kit.tests", "no kops_test_* executables")
    elif bad:
        rep.fail("kit.tests", "; ".join(bad[:4]))
    else:
        rep.ok("kit.tests", f"{len(tests)} executable(s), first={tests[0]}")

    bad_bench: list[str] = []
    for name in benches:
        path = bin_dir / name
        if not path.is_file():
            bad_bench.append(f"{name}: missing")
        elif not os.access(path, os.X_OK):
            bad_bench.append(f"{name}: not executable")
    if not benches:
        rep.warn("kit.benches", "no bench_* executables")
    elif bad_bench:
        rep.fail("kit.benches", "; ".join(bad_bench[:4]))
    else:
        rep.ok("kit.benches", f"{len(benches)} executable(s)")

    if not tests:
        return
    first = bin_dir / tests[0]
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        rep.warn("kit.cubin", "cuobjdump not found; cannot verify the exact-SM cubin")
    else:
        rc, out = run([cuobjdump, "--list-elf", str(first)], timeout=60)
        target = info["elf"]  # sm_90a / sm_100a
        if rc != 0:
            rep.warn("kit.cubin", f"cuobjdump --list-elf failed: {out[:160]}")
        else:
            found = re.findall(r"sm_\d+[a-z]?", out)
            if target in found:
                rep.ok("kit.cubin", f"{first.name} contains {target}")
            else:
                rep.fail(
                    "kit.cubin",
                    f"{first.name} does not contain {target}; found "
                    + ", ".join(sorted(set(found))[:8]),
                )

    ldd = shutil.which("ldd")
    if ldd is None:
        rep.warn("kit.libs", "ldd not found; cannot check shared-library resolution")
    else:
        rc, out = run([ldd, str(first)], timeout=30)
        missing = [line.strip() for line in out.splitlines() if "not found" in line]
        if missing:
            rep.fail("kit.libs", "missing shared library: " + "; ".join(missing[:4]))
        elif "not a dynamic executable" in out:
            rep.ok("kit.libs", "static executable")
        else:
            rep.ok("kit.libs", "all shared libraries resolve")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight a kernel-ops handoff kit (stdlib only, no torch).",
    )
    parser.add_argument("--arch", choices=sorted(ARCH_INFO), help="target exact SM")
    parser.add_argument(
        "--kit",
        default=str(Path(__file__).resolve().parent),
        help="kit root (default: the directory containing this script)",
    )
    parser.add_argument("--json", type=Path, help="write the check report as JSON")
    parser.add_argument(
        "--require-golden",
        action="store_true",
        help="treat a missing golden manifest as a hard failure",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="skip nvidia-smi checks (packaging validation only)",
    )
    args = parser.parse_args(argv)

    kit = resolve_kit(args.kit)
    print("== kernel-ops handoff environment ==")
    print(f"kit:    {kit}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print(f"os:     {platform.platform()}")
    if not kit.is_dir():
        print(f"[FAIL] kit.path               {kit} is not a directory")
        return 1

    rep = Report()
    arch = detect_arch(rep, kit, args.arch)
    if arch is None:
        print("")
        print(f"RESULT: FAIL ({rep.failures} hard failure(s), {rep.warnings} warning(s))")
        return 1
    info = ARCH_INFO[arch]
    print(f"arch:   {arch} ({info['gpu']}, compute {info['cc'][0]}.{info['cc'][1]})")
    print("")

    rep.ok(
        "python.deps",
        "standard library only; torch is not imported",
    )
    env_golden = os.environ.get("KOPS_GOLDEN_DIR")
    if env_golden:
        if Path(env_golden).is_dir():
            rep.ok("env.golden", f"KOPS_GOLDEN_DIR={env_golden}")
        else:
            rep.warn("env.golden", f"KOPS_GOLDEN_DIR={env_golden} does not exist")

    manifest = check_kit_manifest(rep, kit, arch)
    if args.skip_gpu:
        rep.warn("gpu.arch", "GPU checks skipped (--skip-gpu)")
    else:
        check_gpu(rep, info, str(manifest.get("cuda_toolkit") or ""))
    check_nvcc(rep, arch, info)
    check_ctest(rep)
    check_binaries(rep, kit, info, manifest)
    check_golden(rep, kit, args.require_golden)

    result = "PASS" if rep.failures == 0 else "FAIL"
    print("")
    print(
        f"== check_env: {sum(1 for c in rep.checks if c['status'] == 'PASS')} pass, "
        f"{rep.warnings} warn, {rep.failures} fail =="
    )
    print(f"RESULT: {result}")
    if rep.failures:
        print("Fix the FAIL lines before spending a handoff slot.")

    if args.json:
        doc = {
            "schema": 1,
            "arch": arch,
            "kit": str(kit),
            "result": result,
            "failures": rep.failures,
            "warnings": rep.warnings,
            "checks": rep.checks,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "kops_golden_dir": env_golden,
            },
        }
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"report: {args.json}")
        except OSError as exc:
            print(f"[FAIL] report.write             {args.json}: {exc}", file=sys.stderr)
            return 1

    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
