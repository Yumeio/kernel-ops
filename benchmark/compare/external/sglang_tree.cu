// =============================================================================
// benchmark/compare/external/sglang_tree.cu                     Dev-only baseline
//
// Runs SGLang's TreeSpeculativeSamplingTargetOnly (sgl-kernel commit 96a04cb,
// Apache-2.0; vendored in ./sglang/speculative_sampling.cuh) on the same tree
// shape as `spec.tree.target_only.n64.d8.v32000.b8`.  This kernel implements
// SGLang's threshold/target-only rule, not the p/q rejection rule of S03, so it
// is a tree-construction/latency baseline, not a token-for-token oracle.
//
// Build:
//   bash tools/dev/build_external_baselines.sh sglang
// It is never built by CMake, ctest or the shipped library.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "speculative_sampling.cuh"

namespace {

void* alloc(size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc(%zu) failed\n", bytes);
    std::exit(2);
  }
  return p;
}

template <typename T>
void upload(T* dst, const std::vector<T>& src) {
  cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

/// One normalized row of pseudo-random probabilities.
void fill_probs(float* dst, int rows, int cols, uint32_t seed, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<float> all(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    double z = 0.0;
    for (int c = 0; c < cols; ++c) {
      const float e = std::exp(sigma * nd(rng));
      all[static_cast<size_t>(r) * cols + c] = e;
      z += e;
    }
    const float inv = static_cast<float>(1.0 / z);
    for (int c = 0; c < cols; ++c) all[static_cast<size_t>(r) * cols + c] *= inv;
  }
  upload(dst, all);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const int B = 8, N = 64, V = 32000, D = 8;
  const char* json_path = nullptr;
  int flush_mb = 0;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) json_path = argv[i + 1];
    if (std::strcmp(argv[i], "--flush-mb") == 0) flush_mb = std::atoi(argv[i + 1]);
  }

  int32_t* predicts = static_cast<int32_t*>(alloc(sizeof(int32_t) * B * N));
  int32_t* accept_index = static_cast<int32_t*>(alloc(sizeof(int32_t) * B * (D + 1)));
  int32_t* accept_num = static_cast<int32_t*>(alloc(sizeof(int32_t) * B));
  int64_t* candidates = static_cast<int64_t*>(alloc(sizeof(int64_t) * B * N));
  int64_t* retrieve_index = static_cast<int64_t*>(alloc(sizeof(int64_t) * B * N));
  int64_t* next_token = static_cast<int64_t*>(alloc(sizeof(int64_t) * B * N));
  int64_t* next_sibling = static_cast<int64_t*>(alloc(sizeof(int64_t) * B * N));
  float* uniforms = static_cast<float*>(alloc(sizeof(float) * B * N));
  float* final_uniform = static_cast<float*>(alloc(sizeof(float) * B));
  float* target = static_cast<float*>(alloc(sizeof(float) * B * N * V));
  float* draft = static_cast<float*>(alloc(sizeof(float) * B * N * V));

  fill_probs(target, B * N, V, 123u, 3.0f);
  fill_probs(draft, B * N, V, 456u, 2.5f);

  std::vector<int64_t> hcand(B * N), hridx(B * N), hnext(B * N, -1), hsib(B * N, -1);
  std::vector<float> hu(B * N), hf(B);
  std::mt19937 rng(789u);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) {
      const size_t k = static_cast<size_t>(b) * N + i;
      hcand[k] = (i * 13 + 1) % V;
      hridx[k] = i;
      if (2 * i + 1 < N) {
        hnext[k] = 2 * i + 1;
        if (2 * i + 2 < N) hsib[static_cast<size_t>(b) * N + 2 * i + 1] = 2 * i + 2;
      }
      hu[k] = ud(rng);
    }
    hf[b] = ud(rng);
  }
  upload(candidates, hcand);
  upload(retrieve_index, hridx);
  upload(next_token, hnext);
  upload(next_sibling, hsib);
  upload(uniforms, hu);
  upload(final_uniform, hf);

  auto call = [&] {
    auto st = flashinfer::sampling::TreeSpeculativeSamplingTargetOnly<float, int32_t, int64_t>(
        predicts, accept_index, accept_num, candidates, retrieve_index, next_token, next_sibling,
        uniforms, final_uniform, target, draft, B, D + 1, N, V, 1.0f, 1.0f, true, nullptr);
    if (st != cudaSuccess) {
      std::fprintf(stderr, "SGLang kernel: %s\n", cudaGetErrorString(st));
      std::exit(3);
    }
  };

  void* flush_src = nullptr;
  void* flush_dst = nullptr;
  const size_t flush_bytes = static_cast<size_t>(flush_mb) * 1024u * 1024u;
  if (flush_bytes > 0) {
    cudaMalloc(&flush_src, flush_bytes);
    cudaMalloc(&flush_dst, flush_bytes);
    cudaMemset(flush_src, 0x5a, flush_bytes);
  }
  for (int i = 0; i < 5; ++i) call();
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "SGLang warmup failed\n");
    return 3;
  }
  std::vector<double> ms;
  for (int i = 0; i < 20; ++i) {
    cudaEvent_t a = nullptr, b = nullptr;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    if (flush_dst != nullptr) cudaMemcpyAsync(flush_dst, flush_src, flush_bytes, cudaMemcpyDeviceToDevice, 0);
    cudaEventRecord(a);
    call();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t = 0.0f;
    cudaEventElapsedTime(&t, a, b);
    ms.push_back(t);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  const double med = median(ms);
  const double bytes = static_cast<double>(B) * V * 4 + static_cast<double>(B) * N * 8;
  const double gbs = bytes / (med * 1.0e6);
  std::printf("sglang_tree median_ms=%.6f gbs=%.1f\n", med, gbs);
  if (json_path != nullptr) {
    FILE* f = std::fopen(json_path, "w");
    if (f != nullptr) {
      std::fprintf(f,
                   "{\"id\":\"spec.tree.target_only.n64.d8.v32000.b8\","
                   "\"op\":\"spec_tree_target\",\"dtype\":\"float32\",\"rows\":%d,\"cols\":%d,"
                   "\"bytes\":%.0f,\"flops\":0,\"median_ms\":%.6f,\"min_ms\":%.6f,"
                   "\"p10_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,\"gbs\":%.2f,"
                   "\"tflops\":0,\"over_peak\":false}\n",
                   B, V, bytes, med, ms.front(), ms.front(), ms.back(), ms.back(), gbs);
      std::fclose(f);
    }
  }

  cudaFree(predicts);
  cudaFree(accept_index);
  cudaFree(accept_num);
  cudaFree(candidates);
  cudaFree(retrieve_index);
  cudaFree(next_token);
  cudaFree(next_sibling);
  cudaFree(uniforms);
  cudaFree(final_uniform);
  cudaFree(flush_src);
  cudaFree(flush_dst);
  cudaFree(target);
  cudaFree(draft);
  return 0;
}
