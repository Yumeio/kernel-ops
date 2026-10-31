#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/build-linux.sh
# Configure + build + test through the CMake presets (Ninja + ccache).
#
#   tools/build-linux.sh                 -> dev preset (native arch, Release)
#   tools/build-linux.sh dev-debug       -> RelWithDebInfo
#   tools/build-linux.sh ci-sm90a        -> compile-only sm90a (no GPU needed)
#
# Requirements: cmake >= 3.24, CUDA toolkit, ninja, ccache.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-dev}"

command -v cmake >/dev/null || { echo "[error] cmake not found"; exit 1; }
command -v ninja >/dev/null || {
  echo "[error] ninja not found (apt install ninja-build, or pip install ninja)"
  exit 1
}
if ! command -v ccache >/dev/null; then
  echo "[warn] ccache not found: builds work but are not cached (apt install ccache)"
fi

cd "${ROOT}"
echo "[info] configure preset ${PRESET}"
cmake --preset "${PRESET}"

echo "[info] build preset ${PRESET}"
cmake --build --preset "${PRESET}"

if [[ -f "build/${PRESET}/CTestTestfile.cmake" ]]; then
  echo "[info] test preset ${PRESET}"
  ctest --test-dir "build/${PRESET}" --output-on-failure
fi
