#!/usr/bin/env bash
# Dev-only method matrix for B10.  Models are local paths; missing targets are
# skipped rather than downloaded implicitly.
#
# Working methods:
#   Qwen3.5-9B + built-in MTP      baseline / MTP / EAGLE-3 / ngram
#   Qwen3-8B + EAGLE-3 / draft     baseline / EAGLE-3 / draft / ngram
#   Llama-3.1-8B + EAGLE-2/3    baseline / EAGLE-2 / EAGLE-3 (optional)
#
# This script is not built by CMake and is not part of the shipped library.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT=${OUT:-out/b10/methods}
QWEN8=${QWEN8:-/tmp/opencode/qwen3-8b}
QWEN06=${QWEN06:-/tmp/opencode/qwen06}
QWEN35=${QWEN35:-/tmp/opencode/qwen35-local}
EAGLE3_QWEN8=${EAGLE3_QWEN8:-/tmp/opencode/eagle3-qwen3-8b}
EAGLE3_QWEN35=${EAGLE3_QWEN35:-/tmp/opencode/eagle3-qwen35}
LLAMA8=${LLAMA8:-/tmp/opencode/llama31-8b}
EAGLE2_LLAMA=${EAGLE2_LLAMA:-/tmp/opencode/eagle2-llama8b}
EAGLE3_LLAMA=${EAGLE3_LLAMA:-/tmp/opencode/eagle3-llama8b}
COMMON=(--steps 8 --max-tokens 128 --batch-size 1 --temperature 0 --enforce-eager)
mkdir -p "$OUT"

run() {
  local tag=$1; shift
  echo "== $tag =="
  python3 tools/dev/export_vllm_spec_trace.py --tag "$tag" --output "$OUT/$tag" "$@"
}

if [ -d "$QWEN35" ]; then
  run baseline_qwen35 --model "$QWEN35" --baseline --gpu-memory-utilization 0.92 "${COMMON[@]}"
  run mtp_qwen35_9b --model "$QWEN35" --gpu-memory-utilization 0.92 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' "${COMMON[@]}"
  run ngram_qwen35_9b --model "$QWEN35" --gpu-memory-utilization 0.92 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    "${COMMON[@]}"
  if [ -d "$EAGLE3_QWEN35" ]; then
    run eagle3_qwen35_9b --model "$QWEN35" --gpu-memory-utilization 0.92 \
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE3_QWEN35\",\"num_speculative_tokens\":4,\"draft_tensor_parallel_size\":1}" \
      "${COMMON[@]}"
  fi
fi

if [ -d "$QWEN8" ]; then
  run baseline_qwen8 --model "$QWEN8" --baseline --gpu-memory-utilization 0.92 "${COMMON[@]}"
  run ngram_qwen3_8b --model "$QWEN8" --gpu-memory-utilization 0.92 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    "${COMMON[@]}"
  if [ -d "$EAGLE3_QWEN8" ]; then
    run eagle3_qwen3_8b --model "$QWEN8" --gpu-memory-utilization 0.92 \
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE3_QWEN8\",\"num_speculative_tokens\":4,\"draft_tensor_parallel_size\":1}" \
      "${COMMON[@]}"
  fi
  if [ -d "$QWEN06" ]; then
    run draft_qwen3_8b --model "$QWEN8" --gpu-memory-utilization 0.92 \
      --speculative-config "{\"method\":\"draft_model\",\"model\":\"$QWEN06\",\"num_speculative_tokens\":4}" \
      "${COMMON[@]}"
  fi
fi

if [ -d "$LLAMA8" ]; then
  run baseline_llama8 --model "$LLAMA8" --baseline --gpu-memory-utilization 0.92 "${COMMON[@]}"
  if [ -d "$EAGLE2_LLAMA" ]; then
    run eagle2_llama8b --model "$LLAMA8" --gpu-memory-utilization 0.92 \
      --speculative-config "{\"method\":\"eagle\",\"model\":\"$EAGLE2_LLAMA\",\"num_speculative_tokens\":4,\"draft_tensor_parallel_size\":1}" \
      "${COMMON[@]}"
  fi
  if [ -d "$EAGLE3_LLAMA" ]; then
    run eagle3_llama8b --model "$LLAMA8" --gpu-memory-utilization 0.92 \
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE3_LLAMA\",\"num_speculative_tokens\":4,\"draft_tensor_parallel_size\":1}" \
      "${COMMON[@]}"
  fi
fi

echo "== verifier parity / latency =="
for d in "$OUT"/*; do
  [ -d "$d" ] || continue
  if ls "$d"/step_*.bin >/dev/null 2>&1; then
    out/b10/spec_trace_runner --dir "$d" --json "$d/runner.json" || true
  fi
done
python3 tools/dev/report_spec_methods.py --dir "$OUT"
