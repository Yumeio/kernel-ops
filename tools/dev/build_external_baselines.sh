#!/usr/bin/env bash
# Dev-only external baselines for B10 (speculative decoding).
#
#   bash tools/dev/build_external_baselines.sh          # build both
#   bash tools/dev/build_external_baselines.sh sglang   # SGLang tree target-only
#   bash tools/dev/build_external_baselines.sh hpc      # hpc-ops single-draw sampler
#
# Neither target is built by CMake, ctest or the shipped library.  SGLang's
# kernel header is vendored under benchmark/compare/external/sglang (Apache-2.0,
# commit 96a04cb); hpc-ops is fetched at commit 2a2e265 into the gitignored 3rd/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/out/b10"
mkdir -p "$OUT"
cd "$ROOT"

FLASHINFER_INC=$(python3 -c "import flashinfer, os; print(os.path.join(os.path.dirname(flashinfer.__file__), 'data/include'))")
NVCC=${NVCC:-nvcc}

build_traces() {
  echo "[b10] building vLLM trace runner"
  "$NVCC" -std=c++17 -O3 -arch=sm_89 -I. \
    benchmark/compare/external/spec_trace_runner.cu \
    build/b10/libkops_cuda.a build/b10/libkops_host.a -lcudart \
    -o "$OUT/spec_trace_runner"
}

build_sglang() {
  echo "[b10] building SGLang tree target-only baseline"
  "$NVCC" -std=c++17 -O3 -arch=sm_89 \
    -Ibenchmark/compare/external/sglang -I"$FLASHINFER_INC" \
    benchmark/compare/external/sglang_tree.cu -o "$OUT/sglang_tree"
}

build_hpc() {
  echo "[b10] building hpc-ops sampler baseline"
  if [ ! -d 3rd/hpc-ops ]; then
    mkdir -p 3rd
    tmp=$(mktemp -d)
    curl -L -sS https://github.com/Tencent/hpc-ops/archive/2a2e265.tar.gz -o "$tmp/hpc.tgz"
    tar xzf "$tmp/hpc.tgz" -C "$tmp"
    mv "$tmp"/hpc-ops-* 3rd/hpc-ops
    rm -rf "$tmp"
  fi
  "$NVCC" -std=c++20 -O3 -arch=sm_89 --expt-relaxed-constexpr -DHPC_TARGET_ARCH=90 \
    -I3rd/hpc-ops -I3rd/hpc-ops/3rd/cutlass/include \
    benchmark/compare/external/hpc_sampler.cu \
    3rd/hpc-ops/src/sampler/fused_sampler_temperature.cu \
    -o "$OUT/hpc_sampler"
}

case "${1:-all}" in
  sglang) build_sglang ;;
  traces) build_traces ;;
  hpc) build_hpc ;;
  all) build_sglang; build_hpc; build_traces ;;
  *) echo "usage: $0 [sglang|hpc|all]" >&2; exit 2 ;;
esac
