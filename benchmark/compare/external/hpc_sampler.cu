// =============================================================================
// benchmark/compare/external/hpc_sampler.cu                    Dev-only baseline
//
// hpc-ops (commit 2a2e265) has no speculative-verification kernel.  Its
// full-vocabulary temperature sampler is the closest overlapping operation:
// one draw from a [B, V] distribution, which is exactly the bonus/recovery step
// of S03.  We run it on the `spec.reject.bonus.k0.v120832.b128` shape (V is the
// only vocabulary hpc-ops accepts).
//
// The source officially targets SM90/100/103; for an SM89 comparison we compile
// its sampler translation unit with HPC_TARGET_ARCH=90.  Its sampler has no
// architecture dispatch, but this is still an off-label build and is labelled
// as such in the report.
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

#include "src/sampler/sampler.h"

namespace hpc {
int get_sm_count() {
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  return n;
}
int get_sm_major_version() { return 8; }
int get_sm_arch() { return 89; }
[[noreturn]] void throw_arch_not_supported(const char* op, const char* impl) {
  std::fprintf(stderr, "hpc-ops arch dispatch: %s (%s)\n", op, impl);
  std::abort();
}
}  // namespace hpc

namespace {

template <typename T>
void upload(T* dst, const std::vector<T>& src) {
  cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const int B = 128, V = 120832;
  const char* json_path = nullptr;
  int flush_mb = 0;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) json_path = argv[i + 1];
    if (std::strcmp(argv[i], "--flush-mb") == 0) flush_mb = std::atoi(argv[i + 1]);
  }
  float* logits = nullptr;
  int32_t* out = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&logits), static_cast<size_t>(B) * V * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&out), static_cast<size_t>(B) * 4) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 2;
  }
  std::mt19937 rng(123u);
  std::normal_distribution<float> nd;
  std::vector<float> h(static_cast<size_t>(B) * V);
  for (float& x : h) x = 3.0f * nd(rng);
  upload(logits, h);

  auto call = [&] {
    hpc::sampler::fused_sampler_temperature_async(out, logits, 0, V, nullptr, 1.0f, nullptr,
                                                  nullptr, B, V, 7u, nullptr);
  };
  void* flush_src = nullptr;
  void* flush_dst = nullptr;
  const size_t flush_bytes = static_cast<size_t>(flush_mb) * 1024u * 1024u;
  if (flush_bytes > 0) {
    cudaMalloc(&flush_src, flush_bytes);
    cudaMalloc(&flush_dst, flush_bytes);
    cudaMemset(flush_src, 0x5a, flush_bytes);
  }
  for (int i = 0; i < 10; ++i) call();
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "hpc-ops warmup failed\n");
    return 3;
  }
  std::vector<double> ms;
  for (int i = 0; i < 50; ++i) {
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
  const double bytes = static_cast<double>(B) * V * 4 + static_cast<double>(B) * 4;
  const double gbs = bytes / (med * 1.0e6);
  std::printf("hpc_sampler median_ms=%.6f gbs=%.1f\n", med, gbs);
  if (json_path != nullptr) {
    FILE* f = std::fopen(json_path, "w");
    if (f != nullptr) {
      std::fprintf(f,
                   "{\"id\":\"spec.reject.bonus.k0.v120832.b128\",\"op\":\"spec_reject_bonus\","
                   "\"dtype\":\"float32\",\"rows\":%d,\"cols\":%d,\"bytes\":%.0f,\"flops\":0,"
                   "\"median_ms\":%.6f,\"min_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
                   "\"p99_ms\":%.6f,\"gbs\":%.2f,\"tflops\":0,\"over_peak\":false}\n",
                   B, V, bytes, med, ms.front(), ms.front(), ms.back(), ms.back(), gbs);
      std::fclose(f);
    }
  }
  cudaFree(flush_src);
  cudaFree(flush_dst);
  cudaFree(logits);
  cudaFree(out);
  return 0;
}
