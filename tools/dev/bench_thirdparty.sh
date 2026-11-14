#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/dev/bench_thirdparty.sh
#
# One-command wrapper for the third-party benchmark on Linux/WSL. It does not
# clone or build anything; it only selects a Python interpreter and calls
# benchmark/compare/run_thirdparty.py.
#
#   bash tools/dev/bench_thirdparty.sh --all --rounds 3 --flush-l2 256 \
#       --report out/thirdparty/report.md
#
# Interpreter selection (first match wins):
#   1. $PY
#   2. ~/venv-3rd/bin/python
#   3. python3
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SELF_DIR}/../.." && pwd)"

if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "${HOME}/venv-3rd/bin/python" ]]; then
  PYTHON="${HOME}/venv-3rd/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "[bench_thirdparty][error] no Python interpreter found (set PY=...)" >&2
  exit 2
fi

exec "$PYTHON" "${REPO_ROOT}/benchmark/compare/run_thirdparty.py" "$@"
