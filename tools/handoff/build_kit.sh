#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/handoff/build_kit.sh
# Build a self-contained handoff kit for one exact-SM target.
#
#   bash tools/handoff/build_kit.sh sm90a     -> out/handoff/sm90a/
#   bash tools/handoff/build_kit.sh sm100a    -> out/handoff/sm100a/
#
# The kit carries the test and benchmark executables, `tests/golden/` when it
# exists, a generated CTestTestfile.cmake, run_all.sh, check_env.py, this
# directory's README and a SHA-256 kit manifest. Building needs no GPU; running
# the kit needs the matching GPU and a compatible CUDA runtime (see README.md).
#
# Only out/handoff/<sm90a|sm100a>/ is ever replaced. Nothing outside
# out/handoff is deleted, and the CMake build tree is reused as-is.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${ROOT}/out/handoff"

usage() {
  cat <<'EOF'
usage: bash tools/handoff/build_kit.sh <sm90a|sm100a> [options]

options:
  --arch <sm90a|sm100a>   target (also accepted as the first positional)
  --preset <name>         override the handoff-<arch> CMake preset
  --golden <dir>          copy this golden directory instead of tests/golden
  --jobs <n>              pass --parallel <n> to the build
  --skip-configure        reuse the existing build tree
  --skip-build            package existing executables without building
  -h, --help              show this help

examples:
  bash tools/handoff/build_kit.sh sm90a
  bash tools/handoff/build_kit.sh --arch sm100a --jobs 8
EOF
}

ARCH=""
PRESET=""
GOLDEN_SRC=""
JOBS=""
SKIP_CONFIGURE=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    sm90a|sm100a)
      if [[ -n "$ARCH" && "$ARCH" != "$1" ]]; then
        echo "[kit][error] conflicting arch arguments: ${ARCH} and $1" >&2
        exit 2
      fi
      ARCH="$1"
      shift
      ;;
    --arch)
      [[ $# -ge 2 ]] || { echo "[kit][error] --arch needs a value" >&2; exit 2; }
      ARCH="$2"
      shift 2
      ;;
    --preset)
      [[ $# -ge 2 ]] || { echo "[kit][error] --preset needs a value" >&2; exit 2; }
      PRESET="$2"
      shift 2
      ;;
    --golden)
      [[ $# -ge 2 ]] || { echo "[kit][error] --golden needs a value" >&2; exit 2; }
      GOLDEN_SRC="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "[kit][error] --jobs needs a value" >&2; exit 2; }
      JOBS="$2"
      shift 2
      ;;
    --skip-configure)
      SKIP_CONFIGURE=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[kit][error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  echo "[kit][error] target arch is required" >&2
  usage >&2
  exit 2
fi
if [[ "$ARCH" != "sm90a" && "$ARCH" != "sm100a" ]]; then
  echo "[kit][error] unsupported arch '${ARCH}' (expected sm90a or sm100a)" >&2
  exit 2
fi

PRESET="${PRESET:-handoff-${ARCH}}"
BUILD_DIR="${ROOT}/build/${PRESET}"
KIT_DIR="${OUT_ROOT}/${ARCH}"
if [[ -n "$JOBS" ]] && ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[kit][error] --jobs must be a positive integer, got '${JOBS}'" >&2
  exit 2
fi

case "$ARCH" in
  sm90a)  EXPECTED_CC="9.0" ;;
  sm100a) EXPECTED_CC="10.0" ;;
esac

if [[ ! -f "${ROOT}/CMakePresets.json" ]]; then
  echo "[kit][error] missing ${ROOT}/CMakePresets.json" >&2
  exit 1
fi
if ! grep -Eq "\"name\"[[:space:]]*:[[:space:]]*\"${PRESET}\"" "${ROOT}/CMakePresets.json"; then
  echo "[kit][error] CMake preset '${PRESET}' is not defined in CMakePresets.json" >&2
  echo "[kit][hint]  the handoff-sm90a / handoff-sm100a presets must exist before packaging" >&2
  exit 1
fi

command -v cmake >/dev/null 2>&1 || {
  echo "[kit][error] cmake not found (need >= 3.24)" >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "[kit][error] python3 not found (stdlib only; used for the kit manifest)" >&2
  exit 1
}
if [[ "$SKIP_CONFIGURE" -eq 0 || "$SKIP_BUILD" -eq 0 ]]; then
  command -v ninja >/dev/null 2>&1 || {
    echo "[kit][error] ninja not found (apt install ninja-build, or pip install ninja)" >&2
    exit 1
  }
fi
command -v ccache >/dev/null 2>&1 || {
  echo "[kit][warn] ccache not found: the build works but is not cached"
}

if [[ -n "$GOLDEN_SRC" ]]; then
  case "$GOLDEN_SRC" in
    /*) ;;
    *) GOLDEN_SRC="${ROOT}/${GOLDEN_SRC}" ;;
  esac
else
  GOLDEN_SRC="${ROOT}/tests/golden"
fi

cd "$ROOT"

if [[ "$SKIP_CONFIGURE" -eq 0 ]]; then
  echo "[kit] configure preset ${PRESET}"
  cmake --preset "$PRESET"
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[kit] build preset ${PRESET} (tests + benchmarks)"
  if [[ -n "$JOBS" ]]; then
    cmake --build --preset "$PRESET" --parallel "$JOBS"
  else
    cmake --build --preset "$PRESET"
  fi
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[kit][error] build directory ${BUILD_DIR} does not exist" >&2
  echo "[kit][hint]  run without --skip-configure/--skip-build, or configure the preset first" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
STAGE=""
cleanup() {
  if [[ -n "$STAGE" && -d "$STAGE" ]]; then
    rm -rf -- "$STAGE"
  fi
}
trap cleanup EXIT

STAGE="$(mktemp -d "${OUT_ROOT}/.stage-${ARCH}.XXXXXX")"
mkdir -p "${STAGE}/bin" "${STAGE}/golden" "${STAGE}/results"

echo "[kit] collect executables from ${BUILD_DIR}"
mapfile -t TEST_FILES < <(
  find "$BUILD_DIR" -maxdepth 2 -type f -name 'kops_test_*' -perm -u+x -print | sort
)
mapfile -t BENCH_FILES < <(
  find "$BUILD_DIR" -maxdepth 2 -type f -name 'bench_*' -perm -u+x -print | sort
)

if [[ "${#TEST_FILES[@]}" -eq 0 ]]; then
  echo "[kit][error] no test executables under ${BUILD_DIR}" >&2
  echo "[kit][hint]  the build must produce kops_test_* (KOPS_BUILD_TESTS=ON)" >&2
  exit 1
fi
if [[ "${#BENCH_FILES[@]}" -eq 0 ]]; then
  echo "[kit][error] no benchmark executables under ${BUILD_DIR}/benchmark" >&2
  echo "[kit][hint]  the handoff presets set KOPS_BUILD_BENCHMARKS=ON" >&2
  exit 1
fi

for f in "${TEST_FILES[@]}"; do
  name="$(basename "$f")"
  if [[ -e "${STAGE}/bin/${name}" ]]; then
    echo "[kit][error] duplicate executable name ${name}" >&2
    exit 1
  fi
  cp -p "$f" "${STAGE}/bin/${name}"
done
for f in "${BENCH_FILES[@]}"; do
  name="$(basename "$f")"
  if [[ -e "${STAGE}/bin/${name}" ]]; then
    echo "[kit][error] duplicate executable name ${name}" >&2
    exit 1
  fi
  cp -p "$f" "${STAGE}/bin/${name}"
done

echo "[kit] golden source: ${GOLDEN_SRC}"
if [[ -d "$GOLDEN_SRC" ]]; then
  cp -a "${GOLDEN_SRC}/." "${STAGE}/golden/"
  if [[ -f "${STAGE}/golden/manifest.json" ]]; then
    echo "[kit] copied golden manifest"
  else
    echo "[kit][warn] ${GOLDEN_SRC} has no manifest.json; tests will skip 77"
  fi
else
  echo "[kit][warn] no golden directory at ${GOLDEN_SRC}; tests will skip 77"
fi

install -m 0755 "${ROOT}/tools/handoff/run_all.sh" "${STAGE}/run_all.sh"
install -m 0755 "${ROOT}/tools/handoff/check_env.py" "${STAGE}/check_env.py"
if [[ -f "${ROOT}/tools/handoff/README.md" ]]; then
  install -m 0644 "${ROOT}/tools/handoff/README.md" "${STAGE}/README.md"
fi

# CTest metadata lives in the kit, so `ctest --test-dir <kit>` works without
# the original build tree. CMAKE_CURRENT_LIST_DIR resolves to the kit root.
CTEST_FILE="${STAGE}/CTestTestfile.cmake"
{
  echo "# Generated by tools/handoff/build_kit.sh. Do not edit."
  echo "# Tests are executed from <kit>/bin; 77 is the CTest skip code."
  for f in "${TEST_FILES[@]}"; do
    name="$(basename "$f")"
    printf 'add_test([=[%s]=] "${CMAKE_CURRENT_LIST_DIR}/bin/%s")\n' "$name" "$name"
    printf 'set_tests_properties([=[%s]=] PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)\n' "$name"
  done
} > "$CTEST_FILE"

COMMIT="unknown"
DIRTY="unknown"
if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null || true)" ]]; then
    DIRTY="true"
  else
    DIRTY="false"
  fi
fi

NVCC_VERSION="unknown"
if command -v nvcc >/dev/null 2>&1; then
  NVCC_VERSION="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
  [[ -n "$NVCC_VERSION" ]] || NVCC_VERSION="unknown"
fi

BUILD_TYPE="unknown"
CUDA_ARCHS="unknown"
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  BUILD_TYPE="$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1 || true)"
  CUDA_ARCHS="$(sed -n 's/^KOPS_CUDA_ARCHS:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1 || true)"
  [[ -n "$BUILD_TYPE" ]] || BUILD_TYPE="unknown"
  [[ -n "$CUDA_ARCHS" ]] || CUDA_ARCHS="unknown"
fi

export KOPS_KIT_ARCH="$ARCH"
export KOPS_KIT_PRESET="$PRESET"
export KOPS_KIT_EXPECTED_CC="$EXPECTED_CC"
export KOPS_KIT_COMMIT="$COMMIT"
export KOPS_KIT_DIRTY="$DIRTY"
export KOPS_KIT_NVCC="$NVCC_VERSION"
export KOPS_KIT_BUILD_TYPE="$BUILD_TYPE"
export KOPS_KIT_CUDA_ARCHS="$CUDA_ARCHS"
export KOPS_KIT_HOST="$(uname -srm 2>/dev/null || echo unknown)"
export KOPS_KIT_BUILD_DIR="build/${PRESET}"

echo "[kit] write kit manifest"
python3 - "$STAGE" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
dirty_raw = os.environ.get("KOPS_KIT_DIRTY", "unknown")
dirty = None if dirty_raw == "unknown" else dirty_raw == "true"

files = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    if path == root / "manifest.json":
        continue
    files.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )

bin_dir = root / "bin"
tests = sorted(p.name for p in bin_dir.glob("kops_test_*") if p.is_file())
benches = sorted(p.name for p in bin_dir.glob("bench_*") if p.is_file())

golden_manifest = root / "golden" / "manifest.json"
golden = {
    "manifest": golden_manifest.relative_to(root).as_posix()
    if golden_manifest.is_file()
    else None,
    "sha256": sha256(golden_manifest) if golden_manifest.is_file() else None,
    "ready_cases": 0,
    "planned_cases": 0,
}
if golden_manifest.is_file():
    try:
        doc = json.loads(golden_manifest.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("manifest root is not an object")
        cases = doc.get("cases", [])
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict) and case.get("status") == "ready":
                    golden["ready_cases"] += 1
                else:
                    golden["planned_cases"] += 1
        golden["version"] = doc.get("version")
        golden["source_commit"] = doc.get("source_commit")
    except (OSError, ValueError) as exc:
        golden["error"] = str(exc)

doc = {
    "schema": 1,
    "generated_by": "kernel-ops/tools/handoff/build_kit.sh",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ),
    "arch": os.environ["KOPS_KIT_ARCH"],
    "preset": os.environ["KOPS_KIT_PRESET"],
    "expected_compute_capability": os.environ["KOPS_KIT_EXPECTED_CC"],
    "source_commit": os.environ.get("KOPS_KIT_COMMIT", "unknown"),
    "source_dirty": dirty,
    "host": os.environ.get("KOPS_KIT_HOST", "unknown"),
    "cuda_toolkit": os.environ.get("KOPS_KIT_NVCC", "unknown"),
    "cuda_archs": os.environ.get("KOPS_KIT_CUDA_ARCHS", "unknown"),
    "build_type": os.environ.get("KOPS_KIT_BUILD_TYPE", "unknown"),
    "build_dir": os.environ.get("KOPS_KIT_BUILD_DIR", "unknown"),
    "python": sys.version.split()[0],
    "test_executables": tests,
    "benchmark_executables": benches,
    "golden": golden,
    "files": files,
}

(root / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
print(
    f"[kit] manifest: {len(files)} file(s), {len(tests)} test(s), "
    f"{len(benches)} benchmark(s), golden ready/planned "
    f"{golden['ready_cases']}/{golden['planned_cases']}"
)
PY

# Replace only the exact kit directory under out/handoff. The guard is
# deliberately narrow: a typo must never turn into `rm -rf` somewhere else.
case "$KIT_DIR" in
  "${OUT_ROOT}/sm90a"|"${OUT_ROOT}/sm100a") ;;
  *)
    echo "[kit][error] refusing to replace unexpected path ${KIT_DIR}" >&2
    exit 1
    ;;
esac
if [[ -e "$KIT_DIR" || -L "$KIT_DIR" ]]; then
  echo "[kit] replace existing ${KIT_DIR}"
  rm -rf -- "$KIT_DIR"
fi
mv -- "$STAGE" "$KIT_DIR"
STAGE=""
trap - EXIT

echo "[kit] ready: ${KIT_DIR}"
echo "[kit]   bin/       ${#TEST_FILES[@]} test(s), ${#BENCH_FILES[@]} benchmark(s)"
echo "[kit]   golden/    $(if [[ -f "${KIT_DIR}/golden/manifest.json" ]]; then echo "manifest present"; else echo "absent (tests skip 77)"; fi)"
echo "[kit]   manifest   ${KIT_DIR}/manifest.json"
echo "[kit] run on target:"
echo "[kit]   cd ${KIT_DIR}"
echo "[kit]   ./check_env.py --arch ${ARCH} --kit ."
echo "[kit]   ./run_all.sh --arch ${ARCH} --kit . --json-dir results"
echo "[kit] archive: tar -C ${OUT_ROOT} -czf ${ROOT}/out/handoff-${ARCH}.tar.gz ${ARCH}"
