#!/usr/bin/env bash
# Build and compare the B11/B14/B15 benchmark cases against the PyTorch eager
# baseline on the same GPU. Dev-only; results go under out/b11-b15/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-out/b11-b15}"
mkdir -p "$OUT"

cmake --preset bench-sm120a >/dev/null
cmake --build build/bench-sm120a --target bench_mla bench_sparse bench_hyper_connection bench_model_ops -j 4 >/dev/null

for fam in mla sparse hc modelops; do
  case "$fam" in
    mla) bin=bench_mla ;;
    sparse) bin=bench_sparse ;;
    hc) bin=bench_hyper_connection ;;
    modelops) bin=bench_model_ops ;;
  esac
  build/bench-sm120a/benchmark/$bin --dump-cases > "$OUT/cases-$fam.json"
  build/bench-sm120a/benchmark/$bin --iters 10 --json "$OUT/kops-$fam.json" > "$OUT/kops-$fam.log"
  python3 benchmark/compare/torch_baseline.py --cases "$OUT/cases-$fam.json" \
      --json "$OUT/torch-$fam.json" --style hf > "$OUT/torch-$fam.log"
done

python3 benchmark/compare/compare_results.py "$OUT/kops-mla.json" "$OUT/torch-mla.json" \
    > "$OUT/compare-mla.txt"
python3 benchmark/compare/compare_results.py "$OUT/kops-sparse.json" "$OUT/torch-sparse.json" \
    > "$OUT/compare-sparse.txt"
python3 benchmark/compare/compare_results.py "$OUT/kops-hc.json" "$OUT/torch-hc.json" \
    > "$OUT/compare-hc.txt"
python3 benchmark/compare/compare_results.py "$OUT/kops-modelops.json" "$OUT/torch-modelops.json" \
    > "$OUT/compare-modelops.txt"

for f in compare-mla.txt compare-sparse.txt compare-hc.txt compare-modelops.txt; do
  echo "===== $f ====="
  tail -3 "$OUT/$f"
done
