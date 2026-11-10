// Dev-only exact-shape GGUF GEMV benchmark against llama.cpp/ggml.
// Not part of the kernel-ops build. Compile against 3rd/llama.cpp:
//   g++ -O2 -std=c++17 -I3rd/llama.cpp/ggml/include -I3rd/llama.cpp/ggml/src \
//       benchmark/compare/llamacpp_mmvq.cpp -o /tmp/llamacpp_mmvq \
//       -L3rd/llama.cpp/build-sm89/ggml/src -lggml -lggml-base -lggml-cuda \
//       -lcudart -lcuda -Wl,-rpath,<abs path>/build-sm89/ggml/src
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct Result {
  std::string type;
  int64_t n;
  int64_t k;
  int64_t m;
  double median_ms;
  double min_ms;
  double gbs;
  double tflops;
};

static void fill(std::vector<float>& v, int seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ static_cast<uint32_t>(seed);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    v[i] = static_cast<float>(h & 0xFFFF) / 32768.0f - 1.0f;
  }
}

static bool run_case(ggml_backend_t backend, ggml_type type, const char* name,
                     int64_t n, int64_t k, int64_t m, int rounds, Result* out) {
  ggml_init_params params = {
      /* .mem_size   = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
      /* .mem_buffer = */ nullptr,
      /* .no_alloc   = */ true,
  };
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return false;

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_tensor* w = ggml_new_tensor_2d(ctx, type, k, n);
  ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
  ggml_tensor* d = ggml_mul_mat(ctx, w, a);
  ggml_build_forward_expand(gf, d);

  struct ggml_backend_buffer* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_ANY);

  std::vector<float> src(static_cast<size_t>(n) * static_cast<size_t>(k));
  fill(src, 1);
  const size_t wbytes = ggml_nbytes(w);
  std::vector<unsigned char> q(wbytes);
  const size_t qbytes = ggml_quantize_chunk(type, src.data(), q.data(), 0, n, k, nullptr);
  if (qbytes == 0 || qbytes > wbytes) {
    std::fprintf(stderr, "quantize %s failed (%zu > %zu)\n", name, qbytes, wbytes);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return false;
  }
  ggml_backend_tensor_set(w, q.data(), 0, qbytes);

  std::vector<float> in(static_cast<size_t>(k) * static_cast<size_t>(m));
  fill(in, 2);
  ggml_backend_tensor_set(a, in.data(), 0, in.size() * sizeof(float));

  // Warm up and validate that the graph runs.
  if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "compute %s failed\n", name);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return false;
  }
  ggml_backend_synchronize(backend);

  void* flush_a = nullptr;
  void* flush_b = nullptr;
  const size_t flush_bytes = 256ull * 1024 * 1024;
  cudaMalloc(&flush_a, flush_bytes);
  cudaMalloc(&flush_b, flush_bytes);
  cudaMemset(flush_a, 0x5a, flush_bytes);
  cudaMemset(flush_b, 0xa5, flush_bytes);

  constexpr int kInner = 20;
  const bool cold = std::getenv("LLAMACPP_COLD") != nullptr;
  std::vector<float> samples;
  if (cold) {
    for (int i = 0; i < kInner * rounds; ++i) {
      cudaMemcpy(flush_a, flush_b, flush_bytes, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      ggml_backend_synchronize(backend);
      const auto t0 = std::chrono::steady_clock::now();
      if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) break;
      ggml_backend_synchronize(backend);
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  } else {
    for (int r = 0; r < rounds; ++r) {
      cudaMemcpy(flush_a, flush_b, flush_bytes, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      ggml_backend_synchronize(backend);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kInner; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) break;
      }
      ggml_backend_synchronize(backend);
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / kInner;
      samples.push_back(ms);
    }
  }
  cudaFree(flush_a);
  cudaFree(flush_b);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  if (samples.empty()) return false;
  std::sort(samples.begin(), samples.end());
  out->type = name;
  out->n = n;
  out->k = k;
  out->m = m;
  out->median_ms = samples[samples.size() / 2];
  out->min_ms = samples.front();
  const double bytes = static_cast<double>(qbytes) + static_cast<double>(k * m * 4) +
                       static_cast<double>(n * m * 4);
  out->gbs = bytes / (out->median_ms * 1e6);
  out->tflops = 2.0 * static_cast<double>(n) * k * m / (out->median_ms * 1e9);
  return true;
}

int main() {
  ggml_backend_t backend = ggml_backend_cuda_init(0);
  if (backend == nullptr) {
    std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
    return 1;
  }
  std::printf("device: %s\n", ggml_backend_name(backend));
  std::printf("type,n,k,m,median_ms,min_ms,GB/s,TFLOP/s\n");
  const struct { ggml_type t; const char* name; } types[] = {
      {GGML_TYPE_Q4_0, "q4_0"},
      {GGML_TYPE_Q8_0, "q8_0"},
      {GGML_TYPE_Q4_K, "q4_k"},
  };
  const struct { int64_t n, k; } shapes[] = {{4096, 4096}, {14336, 4096}, {4096, 14336}};
  for (const auto& s : shapes) {
    for (const auto& t : types) {
      Result r{};
      if (run_case(backend, t.t, t.name, s.n, s.k, 1, 3, &r)) {
        std::printf("%s,%lld,%lld,%lld,%.6f,%.6f,%.2f,%.4f\n", r.type.c_str(),
                    static_cast<long long>(r.n), static_cast<long long>(r.k),
                    static_cast<long long>(r.m), r.median_ms, r.min_ms, r.gbs, r.tflops);
      }
    }
  }
  ggml_backend_free(backend);
  return 0;
}
