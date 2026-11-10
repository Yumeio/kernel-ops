#!/usr/bin/env bash
# Dev-only NCCL / cuBLAS baseline for B12 (never built by CMake, ctest or the shipped library).
#
#   bash tools/dev/build_comm_baselines.sh                # -> out/b12/nccl_baseline
#   ARCH=120a bash tools/dev/build_comm_baselines.sh      # override the target arch
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/out/b12"
mkdir -p "$OUT"
cd "$ROOT"
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d . | sed 's/^120$/120a/')}
"$NVCC" -std=c++17 -O3 -arch="sm_${ARCH}" benchmark/compare/external/nccl_baseline.cu -o "$OUT/nccl_baseline" -lnccl -lcublas
echo "built $OUT/nccl_baseline (sm_${ARCH})"
