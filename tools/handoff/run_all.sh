#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/handoff/run_all.sh
# Run the tests and benchmarks of a handoff kit and write JSON evidence under
# <kit>/results/ (or --json-dir). The script is copied into every kit and can
# also be run from tools/handoff with an explicit --kit.
#
#   ./run_all.sh --arch sm90a --kit . --json-dir results
#   ./run_all.sh --arch sm100a --kit /data/kits/sm100a
#   ./run_all.sh --kit . --no-bench
#
# It exports KOPS_GOLDEN_DIR=<kit>/golden so tests load committed goldens or
# return 77 when the manifest is absent. Python is used for stdlib-only JSON
# aggregation; no third-party packages and no torch are required.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
usage: run_all.sh [options]

options:
  --arch <sm90a|sm100a>  target SM; inferred from <kit>/manifest.json or the
                         kit directory name when omitted
  --kit <dir>            kit root (default: the directory containing this script)
  --json-dir <dir>       result directory; a relative path is resolved against
                         the kit root (default: <kit>/results)
  --golden <dir>         override the golden directory (default: <kit>/golden)
  --jobs <n>             ctest parallelism (default: 1; GPU tests often conflict)
  --no-ctest             skip ctest
  --no-bench             skip benchmark executables
  -h, --help             show this help

outputs (under --json-dir):
  ctest-<arch>.json/.xml/.log
  <arch>-<family>.json/.log       one pair per benchmark executable
  summary-<arch>.json             aggregate result and exit status

exit status: 0 when ctest and every benchmark pass; 1 otherwise.
EOF
}

ARCH=""
KIT_DIR="$SCRIPT_DIR"
JSON_DIR=""
GOLDEN_DIR=""
JOBS="${KOPS_CTEST_JOBS:-1}"
DO_CTEST=1
DO_BENCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      [[ $# -ge 2 ]] || { echo "[run][error] --arch needs a value" >&2; exit 2; }
      ARCH="$2"
      shift 2
      ;;
    --kit)
      [[ $# -ge 2 ]] || { echo "[run][error] --kit needs a value" >&2; exit 2; }
      KIT_DIR="$2"
      shift 2
      ;;
    --json-dir)
      [[ $# -ge 2 ]] || { echo "[run][error] --json-dir needs a value" >&2; exit 2; }
      JSON_DIR="$2"
      shift 2
      ;;
    --golden)
      [[ $# -ge 2 ]] || { echo "[run][error] --golden needs a value" >&2; exit 2; }
      GOLDEN_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "[run][error] --jobs needs a value" >&2; exit 2; }
      JOBS="$2"
      shift 2
      ;;
    --no-ctest)
      DO_CTEST=0
      shift
      ;;
    --no-bench)
      DO_BENCH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run][error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$KIT_DIR" ]]; then
  echo "[run][error] kit directory does not exist: ${KIT_DIR}" >&2
  exit 2
fi
KIT_DIR="$(cd "$KIT_DIR" && pwd)"

if [[ ! -d "${KIT_DIR}/bin" ]]; then
  echo "[run][error] ${KIT_DIR}/bin is missing; is this a handoff kit?" >&2
  echo "[run][hint]  build one with: bash tools/handoff/build_kit.sh sm90a" >&2
  exit 2
fi

KIT_ARCH=""
if [[ -f "${KIT_DIR}/manifest.json" ]]; then
  KIT_ARCH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("arch", ""))' "${KIT_DIR}/manifest.json" 2>/dev/null || true)"
fi
if [[ -z "$ARCH" ]]; then
  ARCH="$KIT_ARCH"
fi
if [[ -z "$ARCH" ]]; then
  case "$(basename "$KIT_DIR")" in
    sm90a|sm100a) ARCH="$(basename "$KIT_DIR")" ;;
  esac
fi
if [[ "$ARCH" != "sm90a" && "$ARCH" != "sm100a" ]]; then
  echo "[run][error] cannot determine target arch; pass --arch sm90a or --arch sm100a" >&2
  exit 2
fi
if [[ -n "$KIT_ARCH" && "$KIT_ARCH" != "$ARCH" ]]; then
  echo "[run][error] kit manifest says arch=${KIT_ARCH}, but --arch=${ARCH}" >&2
  exit 2
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run][error] --jobs must be a positive integer, got '${JOBS}'" >&2
  exit 2
fi
if [[ "$DO_CTEST" -eq 0 && "$DO_BENCH" -eq 0 ]]; then
  echo "[run][error] --no-ctest and --no-bench leave nothing to run" >&2
  exit 2
fi

if [[ -z "$JSON_DIR" ]]; then
  JSON_DIR="${KIT_DIR}/results"
else
  case "$JSON_DIR" in
    /*) ;;
    *) JSON_DIR="${KIT_DIR}/${JSON_DIR}" ;;
  esac
fi
mkdir -p "$JSON_DIR"
JSON_DIR="$(cd "$JSON_DIR" && pwd)"

if [[ -z "$GOLDEN_DIR" ]]; then
  GOLDEN_DIR="${KIT_DIR}/golden"
else
  case "$GOLDEN_DIR" in
    /*) ;;
    *) GOLDEN_DIR="${KIT_DIR}/${GOLDEN_DIR}" ;;
  esac
fi
command -v python3 >/dev/null 2>&1 || {
  echo "[run][error] python3 not found; run_all.sh uses the standard library to aggregate JSON" >&2
  exit 2
}

export KOPS_GOLDEN_DIR="$GOLDEN_DIR"
export KOPS_HANDOFF_ARCH="$ARCH"
export KOPS_HANDOFF_KIT="$KIT_DIR"
export KOPS_HANDOFF_RESULTS="$JSON_DIR"
export KOPS_CTEST_JOBS="$JOBS"

GOLDEN_STATE="absent"
GOLDEN_COUNTS=""
if [[ -f "${GOLDEN_DIR}/manifest.json" ]]; then
  GOLDEN_STATE="present"
  GOLDEN_COUNTS="$(python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
cases = doc.get("cases", [])
ready = sum(1 for c in cases if isinstance(c, dict) and c.get("status") == "ready")
print(f"{ready} ready / {len(cases) - ready} planned")
' "${GOLDEN_DIR}/manifest.json" 2>/dev/null || true)"
else
  echo "[run][warn] no golden manifest at ${GOLDEN_DIR}/manifest.json; golden tests will skip 77"
fi

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[run] kernel-ops handoff"
echo "[run]   arch:   ${ARCH}"
echo "[run]   kit:    ${KIT_DIR}"
echo "[run]   golden: ${GOLDEN_DIR} (${GOLDEN_STATE}${GOLDEN_COUNTS:+, ${GOLDEN_COUNTS}})"
echo "[run]   results:${JSON_DIR}"

CTEST_MODE="disabled"
CTEST_RC=0
DIRECT_TSV="${JSON_DIR}/.direct-tests.tsv"
: > "$DIRECT_TSV"

if [[ "$DO_CTEST" -eq 1 ]]; then
  rm -f "${JSON_DIR}/ctest-${ARCH}.xml" "${JSON_DIR}/ctest-${ARCH}.log"
  if command -v ctest >/dev/null 2>&1 && [[ -f "${KIT_DIR}/CTestTestfile.cmake" ]]; then
    CTEST_MODE="ctest"
    echo "[run] ctest --test-dir ${KIT_DIR} --parallel ${JOBS}"
    set +e
    ctest --test-dir "$KIT_DIR" --output-on-failure --parallel "$JOBS" \
      --output-junit "${JSON_DIR}/ctest-${ARCH}.xml" \
      2>&1 | tee "${JSON_DIR}/ctest-${ARCH}.log"
    CTEST_RC=${PIPESTATUS[0]}
    set -e
  else
    CTEST_MODE="direct"
    echo "[run][warn] ctest or CTestTestfile.cmake missing; running test executables directly"
    set +e
    : > "${JSON_DIR}/ctest-${ARCH}.log"
    shopt -s nullglob
    TEST_EXES=("${KIT_DIR}"/bin/kops_test_*)
    shopt -u nullglob
    if [[ "${#TEST_EXES[@]}" -eq 0 ]]; then
      echo "[run][error] no test executables in ${KIT_DIR}/bin" >&2
      CTEST_RC=1
    else
      for exe in "${TEST_EXES[@]}"; do
        name="$(basename "$exe")"
        echo "RUN ${name}" | tee -a "${JSON_DIR}/ctest-${ARCH}.log"
        RUNNER=()
        if command -v timeout >/dev/null 2>&1; then
          RUNNER=(timeout 300)
        fi
        "${RUNNER[@]}" "$exe" 2>&1 | tee -a "${JSON_DIR}/ctest-${ARCH}.log"
        test_rc=${PIPESTATUS[0]}
        printf '%s\t%s\n' "$name" "$test_rc" >> "$DIRECT_TSV"
        if [[ "$test_rc" -ne 0 && "$test_rc" -ne 77 ]]; then
          CTEST_RC=1
        fi
      done
    fi
    set -e
  fi
else
  echo "[run] ctest skipped (--no-ctest)"
fi

BENCH_TSV="${JSON_DIR}/.benchmarks.tsv"
: > "$BENCH_TSV"
BENCH_TOTAL=0
BENCH_FAIL=0

if [[ "$DO_BENCH" -eq 1 ]]; then
  shopt -s nullglob
  BENCH_EXES=("${KIT_DIR}"/bin/bench_*)
  shopt -u nullglob
  if [[ "${#BENCH_EXES[@]}" -eq 0 ]]; then
    echo "[run][warn] no benchmark executables in ${KIT_DIR}/bin"
  else
    for exe in "${BENCH_EXES[@]}"; do
      name="$(basename "$exe")"
      family="${name#bench_}"
      out="${JSON_DIR}/${ARCH}-${family}.json"
      log="${JSON_DIR}/${ARCH}-${family}.log"
      rm -f "$out"
      echo "[run] ${name} -> ${out}"
      set +e
      ( cd "$KIT_DIR" && "$exe" --json "$out" ) 2>&1 | tee "$log"
      bench_rc=${PIPESTATUS[0]}
      set -e
      BENCH_TOTAL=$((BENCH_TOTAL + 1))
      if [[ "$bench_rc" -ne 0 ]]; then
        BENCH_FAIL=$((BENCH_FAIL + 1))
      fi
      printf '%s\t%s\t%s\t%s\n' "$name" "$bench_rc" "$out" "$log" >> "$BENCH_TSV"
    done
  fi
else
  echo "[run] benchmarks skipped (--no-bench)"
fi

FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
python3 - "$ARCH" "$KIT_DIR" "$JSON_DIR" "$CTEST_MODE" "$CTEST_RC" \
  "$STARTED_UTC" "$FINISHED_UTC" "$DIRECT_TSV" "$BENCH_TSV" <<'PY'
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

arch, kit, out_dir, ctest_mode, ctest_rc_s, started, finished, direct_tsv, bench_tsv = sys.argv[1:11]
out_dir = Path(out_dir)
ctest_rc = int(ctest_rc_s)


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_manifest(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


kit_manifest = read_manifest(Path(kit) / "manifest.json")

# ---- ctest results ---------------------------------------------------------
tests = []
parse_note = None
junit = out_dir / f"ctest-{arch}.xml"
if ctest_mode == "ctest" and junit.is_file():
    try:
        tree = ET.parse(junit)
        for tc in tree.iter():
            if localname(tc.tag) != "testcase":
                continue
            name = tc.get("name") or "(unnamed)"
            try:
                seconds = float(tc.get("time") or 0.0)
            except ValueError:
                seconds = 0.0
            status = "passed"
            failure = None
            for child in tc:
                kind = localname(child.tag)
                if kind in ("failure", "error"):
                    status = "failed"
                    failure = (child.get("message") or (child.text or "").strip() or kind)
                elif kind == "skipped" and status == "passed":
                    status = "skipped"
            tests.append(
                {
                    "name": name,
                    "status": status,
                    "time_s": round(seconds, 6),
                    "failure": failure,
                }
            )
    except (OSError, ET.ParseError) as exc:
        parse_note = f"cannot parse {junit.name}: {exc}"
elif ctest_mode == "direct":
    try:
        for line in Path(direct_tsv).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, rc_s = line.split("\t", 1)
            rc = int(rc_s)
            tests.append(
                {
                    "name": name,
                    "status": "skipped" if rc == 77 else ("passed" if rc == 0 else "failed"),
                    "time_s": None,
                    "failure": None if rc in (0, 77) else f"exit code {rc}",
                }
            )
    except (OSError, ValueError) as exc:
        parse_note = f"cannot read direct test results: {exc}"
elif ctest_mode == "ctest":
    parse_note = "ctest did not produce a JUnit file"

passed = sum(1 for t in tests if t["status"] == "passed")
failed = sum(1 for t in tests if t["status"] == "failed")
skipped = sum(1 for t in tests if t["status"] == "skipped")

ctest_doc = {
    "schema": 1,
    "arch": arch,
    "kit": kit,
    "mode": ctest_mode,
    "exit_code": ctest_rc,
    "started_utc": started,
    "finished_utc": finished,
    "command": (
        f"ctest --test-dir {kit} --output-on-failure --parallel "
        f"{os.environ.get('KOPS_CTEST_JOBS', '1')}"
        if ctest_mode == "ctest"
        else None
    ),
    "golden_dir": os.environ.get("KOPS_GOLDEN_DIR"),
    "junit": junit.name if junit.is_file() else None,
    "log": f"ctest-{arch}.log",
    "note": parse_note,
    "totals": {
        "tests": len(tests),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    },
    "tests": tests,
}
(out_dir / f"ctest-{arch}.json").write_text(
    json.dumps(ctest_doc, indent=2) + "\n", encoding="utf-8"
)

# ---- benchmark results -----------------------------------------------------
benchmarks = []
try:
    for line in Path(bench_tsv).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, rc_s, json_path, log_path = line.split("\t", 3)
        rc = int(rc_s)
        benchmarks.append(
            {
                "name": name,
                "family": name[len("bench_") :] if name.startswith("bench_") else name,
                "exit_code": rc,
                "ok": rc == 0,
                "json": Path(json_path).name,
                "log": Path(log_path).name,
                "json_valid": False,
            }
        )
except (OSError, ValueError) as exc:
    parse_note = f"{parse_note}; " if parse_note else ""
    parse_note = f"{parse_note}cannot read benchmark results: {exc}"

for entry in benchmarks:
    path = out_dir / entry["json"]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        entry["json_valid"] = isinstance(doc, dict) and isinstance(doc.get("results"), list)
        if isinstance(doc, dict):
            entry["device"] = doc.get("device", {}).get("name")
            entry["cases"] = len(doc.get("results", []))
    except (OSError, ValueError):
        pass
    entry["evidence_ok"] = entry["ok"] and entry["json_valid"]

bench_ok = sum(1 for b in benchmarks if b["ok"] and b["json_valid"])
bench_failed = sum(1 for b in benchmarks if not (b["ok"] and b["json_valid"]))

result = "PASS"
if ctest_mode != "disabled" and ctest_rc != 0:
    result = "FAIL"
if bench_failed:
    result = "FAIL"
if parse_note:
    result = "FAIL"

golden_info = kit_manifest.get("golden") or {}
summary = {
    "schema": 1,
    "arch": arch,
    "kit": kit,
    "result": result,
    "started_utc": started,
    "finished_utc": finished,
    "source_commit": kit_manifest.get("source_commit"),
    "source_dirty": kit_manifest.get("source_dirty"),
    "golden_dir": os.environ.get("KOPS_GOLDEN_DIR"),
    "golden_manifest": golden_info.get("manifest"),
    "ctest": {
        "mode": ctest_mode,
        "exit_code": ctest_rc,
        "totals": ctest_doc["totals"],
        "json": f"ctest-{arch}.json",
        "note": parse_note,
    },
    "benchmarks": {
        "total": len(benchmarks),
        "ok": bench_ok,
        "failed": bench_failed,
        "items": benchmarks,
    },
    "environment": {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kops_golden_dir": os.environ.get("KOPS_GOLDEN_DIR"),
    },
}
(out_dir / f"summary-{arch}.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)

print("")
print("== kernel-ops handoff summary ==")
print(f"arch:        {arch}")
print(f"kit:         {kit}")
print(f"source:      {kit_manifest.get('source_commit')}"
      f"{' (dirty)' if kit_manifest.get('source_dirty') else ''}")
print(f"golden:      {os.environ.get('KOPS_GOLDEN_DIR')}")
if ctest_mode == "disabled":
    print("ctest:       skipped")
elif ctest_mode == "direct":
    print(
        f"ctest:       direct fallback: {passed} passed, {failed} failed, "
        f"{skipped} skipped ({len(tests)} total), exit {ctest_rc}"
    )
else:
    print(
        f"ctest:       {passed} passed, {failed} failed, {skipped} skipped "
        f"({len(tests)} total), exit {ctest_rc}"
    )
if benchmarks:
    print(
        f"benchmarks:  {bench_ok}/{len(benchmarks)} ok, "
        f"{bench_failed} failed"
    )
else:
    print("benchmarks:  none")
if parse_note:
    print(f"note:        {parse_note}")
print(f"results:     {out_dir}")
print(f"RESULT:      {result}")
sys.exit(0 if result == "PASS" else 3)
PY
SUMMARY_RC=$?
set -e

rm -f "$DIRECT_TSV" "$BENCH_TSV"

if [[ "$SUMMARY_RC" -ne 0 ]]; then
  echo "[run] FAIL: inspect ${JSON_DIR}/summary-${ARCH}.json and the logs"
  exit 1
fi
echo "[run] PASS: evidence written to ${JSON_DIR}"
exit 0
