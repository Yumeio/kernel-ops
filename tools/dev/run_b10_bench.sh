#!/usr/bin/env bash
# Dev-only B10 benchmark/comparison loop.
#
#   bash tools/dev/run_b10_bench.sh            # 3 interleaved rounds
#   ROUNDS=5 bash tools/dev/run_b10_bench.sh
#
# Assumes the `bench` CMake preset has been built (build/bench) and the external
# baselines were built with tools/dev/build_external_baselines.sh.  The tree is
# dirty during development; JSON therefore stays diagnostic_only per CONTRACTS §2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ROUNDS=${ROUNDS:-3}
KOPS_DIR=${KOPS_DIR:-build/b10/benchmark}
OUT=out/b10
mkdir -p "$OUT"

"$KOPS_DIR/bench_sampling" --dump-cases --only spec. > "$OUT/cases-spec.json"
"$KOPS_DIR/bench_state" --dump-cases --only state_commit > "$OUT/cases-commit.json"

for r in $(seq 1 "$ROUNDS"); do
  "$KOPS_DIR/bench_sampling" --only spec. --flush-l2 256 --rounds 1 \
    --json "$OUT/kops-spec-r$r.json" >/dev/null
  "$KOPS_DIR/bench_state" --only state_commit --flush-l2 256 --rounds 1 \
    --json "$OUT/kops-commit-r$r.json" >/dev/null
  python3 benchmark/compare/torch_baseline.py --cases "$OUT/cases-spec.json" --only spec. \
    --flush-l2 256 --rounds 1 --json "$OUT/torch-spec-r$r.json" >/dev/null
  python3 benchmark/compare/torch_baseline.py --cases "$OUT/cases-commit.json" --only state_commit \
    --flush-l2 256 --rounds 1 --json "$OUT/torch-commit-r$r.json" >/dev/null
done

# External baselines: SGLang tree target-only and hpc-ops single draw.
for r in $(seq 1 "$ROUNDS"); do
  out/b10/sglang_tree --flush-mb 256 --json "$OUT/sglang-tree-r$r.json" >/dev/null
  out/b10/hpc_sampler --flush-mb 256 --json "$OUT/hpc-sampler-r$r.json" >/dev/null
done

echo "== kops vs PyTorch, round $ROUNDS =="
python3 benchmark/compare/compare_results.py "$OUT/kops-spec-r$ROUNDS.json" \
  "$OUT/torch-spec-r$ROUNDS.json" --json "$OUT/compare-spec-r$ROUNDS.json"
echo "== state commit =="
python3 benchmark/compare/compare_results.py "$OUT/kops-commit-r$ROUNDS.json" \
  "$OUT/torch-commit-r$ROUNDS.json"
echo "== SGLang tree target-only =="
python3 - "$OUT/kops-spec-r$ROUNDS.json" "$OUT/sglang-tree-r$ROUNDS.json" <<'PY'
import json, sys
k = {r["id"]: r for r in json.load(open(sys.argv[1]))["results"]}
s = json.load(open(sys.argv[2]))
r = k[s["id"]]
print(f"{s['id']}: kops {r['median_ms']:.4f} ms, SGLang {s['median_ms']:.4f} ms, "
      f"ratio {s['median_ms']/r['median_ms']:.2f}x")
PY
echo "== hpc-ops single draw =="
python3 - "$OUT/kops-spec-r$ROUNDS.json" "$OUT/hpc-sampler-r$ROUNDS.json" <<'PY'
import json, sys
k = {r["id"]: r for r in json.load(open(sys.argv[1]))["results"]}
s = json.load(open(sys.argv[2]))
r = k[s["id"]]
print(f"{s['id']}: kops {r['median_ms']:.4f} ms, hpc-ops {s['median_ms']:.4f} ms, "
      f"ratio {s['median_ms']/r['median_ms']:.2f}x (single full-vocab draw; not a verifier)")
PY
