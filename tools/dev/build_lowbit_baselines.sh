#!/usr/bin/env bash
# Dev-only 3rd-party baselines for B17 (never built by CMake, never shipped):
#   attention.rs TurboQuant low-bit KV store/decode and Moet W2 pack.
# The source is read-only in 3rd/; binaries land in build/compare/.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"
mkdir -p build/compare out/b17
arch="${KOPS_CUDA_ARCHS:-native}"
if [[ "${arch}" == "native" ]]; then
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
fi
case "${arch}" in
  86) nvarch=sm_86 ;;
  89) nvarch=sm_89 ;;
  90a) nvarch=sm_90a ;;
  100a) nvarch=sm_100a ;;
  103a) nvarch=sm_103a ;;
  120a) nvarch=sm_120a ;;
  121a) nvarch=sm_121a ;;
  *) nvarch="sm_${arch}" ;;
esac
flash="${root}/3rd/attention.rs/src/kernels/src/flash"
nvcc -std=c++17 -O2 -I "${flash}" -arch="${nvarch}" \
  benchmark/compare/external/attention_rs_lowbit_baseline.cu \
  -o build/compare/attention_rs_lowbit
nvcc -std=c++17 -O2 -arch="${nvarch}" \
  benchmark/compare/external/attention_rs_moe_w2_baseline.cu \
  3rd/attention.rs/src/kernels/src/moe_w2_pack.cu \
  -o build/compare/attention_rs_moe_w2
echo "built build/compare/attention_rs_lowbit and build/compare/attention_rs_moe_w2 (${nvarch})"
