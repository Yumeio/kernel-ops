#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/run_sampling_bench.sh
# Reproducible H100/H20 sampler run: kernel-ops benchmark, optional hpc-ops
# baseline and ncu counters. Run from any directory; artifacts go to out/.
#
#   tools/run_sampling_bench.sh                 # bench-sm90a + kernel-ops JSON
#   RUN_HPC_OPS=1 tools/run_sampling_bench.sh   # also build/run 3rd/hpc-ops
#   NCU=1 tools/run_sampling_bench.sh           # ncu on the B=512 fused cases
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${PRESET:-bench-sm90a}"
OUT="${ROOT}/out"
mkdir -p "${OUT}"

cd "${ROOT}"
echo "[info] configure ${PRESET}"
cmake --preset "${PRESET}"
echo "[info] build bench_sampling"
cmake --build --preset "${PRESET}" --target bench_sampling

BENCH="${ROOT}/build/${PRESET}/benchmark/bench_sampling"
[ -x "${BENCH}" ] || { echo "[error] missing ${BENCH}"; exit 1; }

echo "[info] kernel-ops sampler matrix"
EXTRA=()
if [[ "${SPLIT_SWEEP:-0}" == "1" ]]; then EXTRA+=(--split-sweep); fi
"${BENCH}" "${EXTRA[@]}" --json "${OUT}/kops-sampler-${PRESET}.json"

if [[ "${RUN_HPC_OPS:-0}" == "1" ]]; then
  HPC="${ROOT}/3rd/hpc-ops"
  if [[ ! -d "${HPC}" ]]; then
    echo "[warn] 3rd/hpc-ops not present; skipping the hpc-ops baseline"
  else
    echo "[info] build hpc-ops sm90"
    (cd "${HPC}" && make sm90)
    echo "[info] hpc-ops sampler baseline (V=120832)"
    (cd "${HPC}" && python3 benchmark/sampler/benchmark_sampler.py \
      --providers hpc --scenes temperature full \
      --batches 1 32 128 512 --timing event --warmup 20 --iters 100 \
      --jsonl "${OUT}/hpcops-sampler.jsonl")
    python3 "${ROOT}/benchmark/compare/hpcops_sampler.py" \
      --jsonl "${OUT}/hpcops-sampler.jsonl" | tee "${OUT}/hpcops-sampler.txt"
  fi
fi

if [[ "${NCU:-0}" == "1" ]]; then
  command -v ncu >/dev/null || { echo "[warn] ncu not found"; exit 0; }
  for mode in greedy k50 p0.95; do
    echo "[info] ncu fused.${mode} b512"
    ncu --set full --kernel-name "regex:fused_" --launch-count 3 \
      "${BENCH}" --only "sampling.fused.${mode}.bf16.v120832.b512" --iters 6 \
      > "${OUT}/ncu-sampler-${mode}.txt" 2>&1 || true
  done
fi

echo "[info] artifacts: ${OUT}/kops-sampler-${PRESET}.json"
