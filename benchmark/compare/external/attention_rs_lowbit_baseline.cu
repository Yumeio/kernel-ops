// Dev-only baseline: attention.rs TurboQuant low-bit KV store/decode (3rd/attention.rs,
// MIT). Never built by CMake, never linked into kops. It exists only to time the upstream
// flash_tq4/flash_tq3 kernels on the same GPU and shape as `bench_merge --only lowbit`.
//
//   tools/dev/build_lowbit_baselines.sh
//   build/compare/attention_rs_lowbit --json out/b17/attention-rs-lowbit.json
//
// Shape: B=8, q=1, kv=4096, Hq=32, Hkv=8, D=128, page 16 (matches the kops case
// lowbit.decode.bf16.d128.b8.kv4096.h32x8.*). The upstream format stores K absmax and
// V absmax as separate float arrays and packs 4-bit K/V (or 3-bit K + 4-bit V).
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <tuple>

#define FLASH_HDIM 128
#include "flash_turboquant.cuh"
#include "flash_turboquant_lowbit.cuh"
#include "flash_prefill_tq4.cuh"
#include "flash_prefill_tq3.cuh"

namespace {

constexpr int kB = 8, kKV = 4096, kHq = 32, kHkv = 8, kD = FLASH_HDIM, kPage = 16;
constexpr int kPagesPerSeq = kKV / kPage;
constexpr int kPages = kB * kPagesPerSeq;
constexpr int kTokens = kB * kKV;
constexpr int kIters = 50, kWarmup = 10;

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  Buf(size_t n, size_t elem = 1) : bytes(n * elem) {
    if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc(%zu) failed\n", bytes);
      std::exit(1);
    }
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

void fill_bf16(__nv_bfloat16* p, size_t n, unsigned seed) {
  std::vector<__nv_bfloat16> h(n);
  unsigned x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    h[i] = __float2bfloat16(static_cast<float>(x >> 8) / 8388608.0f - 1.0f);
  }
  cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice);
}

float time_ms(const std::function<void()>& fn) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < kWarmup; ++i) fn();
  cudaEventRecord(a);
  for (int i = 0; i < kIters; ++i) fn();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / kIters;
}

void json(const char* path, const std::vector<std::tuple<const char*, double, double>>& rows) {
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) return;
  std::fprintf(f, "{\"backend\":\"attention.rs TurboQuant (dev baseline)\",\"gpu\":\"");
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::fprintf(f, "%s\",\"cc\":\"%d.%d\",\"results\":[", prop.name, prop.major, prop.minor);
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& [id, ms, gbs] = rows[i];
    std::fprintf(f,
                 "%s{\"id\":\"%s\",\"median_ms\":%.6f,\"gbs\":%.3f}",
                 i ? "," : "", id, ms, gbs);
  }
  std::fprintf(f, "]}\n");
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  const char* json_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
  }
  Buf q(kTokens / kKV * kHq * kD, 2), o(kB * kHq * kD, 2);
  Buf key(kTokens * kHkv * kD, 2), val(kTokens * kHkv * kD, 2);
  fill_bf16(static_cast<__nv_bfloat16*>(q.p), static_cast<size_t>(kB) * kHq * kD, 7);
  fill_bf16(static_cast<__nv_bfloat16*>(key.p), static_cast<size_t>(kTokens) * kHkv * kD, 11);
  fill_bf16(static_cast<__nv_bfloat16*>(val.p), static_cast<size_t>(kTokens) * kHkv * kD, 13);

  constexpr int kQ = 512, kQTokens = kB * kQ;
  Buf qf(kQTokens * kHq * kD, 2), of(kQTokens * kHq * kD, 2);
  Buf cu(kB + 1, sizeof(unsigned)), ctxl(kB, sizeof(unsigned));
  {
    std::vector<unsigned> hc(kB + 1), hl(kB);
    for (int b = 0; b <= kB; ++b) hc[b] = static_cast<unsigned>(b) * kQ;
    for (int b = 0; b < kB; ++b) hl[b] = kKV;
    cudaMemcpy(cu.p, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ctxl.p, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice);
  }
  fill_bf16(static_cast<__nv_bfloat16*>(qf.p), static_cast<size_t>(kQTokens) * kHq * kD, 17);
  Buf slot(kTokens, sizeof(long long));
  {
    std::vector<long long> h(kTokens);
    for (int b = 0; b < kB; ++b)
      for (int t = 0; t < kKV; ++t)
        h[static_cast<size_t>(b) * kKV + t] =
            static_cast<long long>(b * kPagesPerSeq + t / kPage) * kPage + t % kPage;
    cudaMemcpy(slot.p, h.data(), h.size() * sizeof(long long), cudaMemcpyHostToDevice);
  }
  Buf block_table(kB * kPagesPerSeq, sizeof(int));
  Buf seq_lens(kB, sizeof(int));
  {
    std::vector<int> bt(kB * kPagesPerSeq), sl(kB, kKV);
    for (int b = 0; b < kB; ++b)
      for (int i = 0; i < kPagesPerSeq; ++i) bt[static_cast<size_t>(b) * kPagesPerSeq + i] = b * kPagesPerSeq + i;
    cudaMemcpy(block_table.p, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(seq_lens.p, sl.data(), sl.size() * 4, cudaMemcpyHostToDevice);
  }

  // Upstream layout: [pages, page, Hkv] absmax + [pages, page, Hkv, D/2] payload.
  const size_t am_n = static_cast<size_t>(kPages) * kPage * kHkv;
  const size_t q4_n = am_n * (kD / 2);
  const size_t q3_n = am_n * ((kD * 3 + 7) / 8);
  Buf k_am(am_n, 4), v_am(am_n, 4), k4(q4_n), v4(q4_n), k3(q3_n), v3(q4_n);

  std::vector<std::tuple<const char*, double, double>> rows;
  auto store4 = [&] {
    flash_tq4_store<__nv_bfloat16><<<kTokens, 32>>>(
        static_cast<const __nv_bfloat16*>(key.p), static_cast<const __nv_bfloat16*>(val.p),
        static_cast<float*>(k_am.p), static_cast<unsigned char*>(k4.p),
        static_cast<float*>(v_am.p), static_cast<unsigned char*>(v4.p),
        static_cast<const long long*>(slot.p), kTokens, kHkv, kD, kPage);
  };
  auto store3 = [&] {
    flash_tq3_store<__nv_bfloat16><<<kTokens, 32>>>(
        static_cast<const __nv_bfloat16*>(key.p), static_cast<const __nv_bfloat16*>(val.p),
        static_cast<float*>(k_am.p), static_cast<unsigned char*>(k3.p),
        static_cast<float*>(v_am.p), static_cast<unsigned char*>(v3.p),
        static_cast<const long long*>(slot.p), kTokens, kHkv, kD, kPage);
  };
  auto decode4 = [&] {
    flash_tq4_decode<__nv_bfloat16><<<dim3(kHq, kB), 256>>>(
        static_cast<const __nv_bfloat16*>(q.p), static_cast<const float*>(k_am.p),
        static_cast<const unsigned char*>(k4.p), static_cast<const float*>(v_am.p),
        static_cast<const unsigned char*>(v4.p), static_cast<__nv_bfloat16*>(o.p),
        static_cast<const int*>(block_table.p), static_cast<const int*>(seq_lens.p),
        kPagesPerSeq, kHq, kHkv, kD, kPage, 1.0f / std::sqrt(static_cast<float>(kD)), kB,
        kHq * kD, 0.0f, 0u);
  };
  auto decode3 = [&] {
    flash_tq3_decode<__nv_bfloat16><<<dim3(kHq, kB), 256>>>(
        static_cast<const __nv_bfloat16*>(q.p), static_cast<const float*>(k_am.p),
        static_cast<const unsigned char*>(k3.p), static_cast<const float*>(v_am.p),
        static_cast<const unsigned char*>(v3.p), static_cast<__nv_bfloat16*>(o.p),
        static_cast<const int*>(block_table.p), static_cast<const int*>(seq_lens.p),
        kPagesPerSeq, kHq, kHkv, kD, kPage, 1.0f / std::sqrt(static_cast<float>(kD)), kB,
        kHq * kD, 0.0f, 0u);
  };

  auto prefill4 = [&] {
    flash_tq4_prefill<__nv_bfloat16><<<dim3(kHq, (kQ + 31) / 32, 1), 128>>>(
        static_cast<const __nv_bfloat16*>(qf.p), static_cast<const float*>(k_am.p),
        static_cast<const unsigned char*>(k4.p), static_cast<const float*>(v_am.p),
        static_cast<const unsigned char*>(v4.p), static_cast<__nv_bfloat16*>(of.p),
        static_cast<const int*>(block_table.p), kPagesPerSeq,
        static_cast<const unsigned*>(cu.p), static_cast<const unsigned*>(ctxl.p), kHq, kHkv, kD,
        kPage, 0u, 1u, 1.0f / std::sqrt(static_cast<float>(kD)), 0.0f);
  };
  auto prefill3 = [&] {
    flash_tq3_prefill<__nv_bfloat16><<<dim3(kHq, (kQ + 31) / 32, 1), 128>>>(
        static_cast<const __nv_bfloat16*>(qf.p), static_cast<const float*>(k_am.p),
        static_cast<const unsigned char*>(k3.p), static_cast<const float*>(v_am.p),
        static_cast<const unsigned char*>(v3.p), static_cast<__nv_bfloat16*>(of.p),
        static_cast<const int*>(block_table.p), kPagesPerSeq,
        static_cast<const unsigned*>(cu.p), static_cast<const unsigned*>(ctxl.p), kHq, kHkv, kD,
        kPage, 0u, 1u, 1.0f / std::sqrt(static_cast<float>(kD)), 0.0f);
  };
  store4();
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "store4 launch failed\n");
    return 1;
  }
  const double k4_bytes = static_cast<double>(kTokens) * kHkv * (kD / 2 + 4);  // codes + absmax
  const double v4_bytes = static_cast<double>(kTokens) * kHkv * (kD / 2 + 4);
  const double k3_bytes = static_cast<double>(kTokens) * kHkv * (kD * 3 / 8.0 + 4);
  const double store4_bytes = static_cast<double>(kTokens) * kHkv * kD * 2 * 2 + k4_bytes + v4_bytes;
  const double store3_bytes = static_cast<double>(kTokens) * kHkv * kD * 2 * 2 + k3_bytes + v4_bytes;
  const double dec4_bytes = k4_bytes + v4_bytes + static_cast<double>(kB) * kHq * kD * 2 * 2;
  const double dec3_bytes = k3_bytes + v4_bytes + static_cast<double>(kB) * kHq * kD * 2 * 2;

  double ms = time_ms(store4);
  rows.emplace_back("lowbit.store.bf16.t4096.h8.d128.turbo4", ms, store4_bytes / (ms * 1e6));
  ms = time_ms(store3);
  rows.emplace_back("lowbit.store.bf16.t4096.h8.d128.turbo3", ms, store3_bytes / (ms * 1e6));
  ms = time_ms(decode4);
  rows.emplace_back("lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4", ms, dec4_bytes / (ms * 1e6));
  ms = time_ms(decode3);
  rows.emplace_back("lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3", ms, dec3_bytes / (ms * 1e6));
  {
    const double pairs = static_cast<double>(1) * kQ * (2.0 * kKV - kQ + 1.0) / 2.0;
    const double flops = 4.0 * pairs * kD * kHq;
    ms = time_ms(prefill4);
    std::printf("prefill turbo4 %.4f ms  %.2f TFLOP/s\n", ms, flops / (ms * 1e9));
    rows.emplace_back("lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4", ms, 0.0);
    ms = time_ms(prefill3);
    std::printf("prefill turbo3 %.4f ms  %.2f TFLOP/s\n", ms, flops / (ms * 1e9));
  }

  std::printf("attention.rs TurboQuant baseline (dev-only)\n");
  std::printf("%-52s %10s %10s\n", "case", "median_ms", "GB/s");
  for (const auto& [id, t, g] : rows) std::printf("%-52s %10.4f %10.1f\n", id, t, g);
  if (json_path != nullptr) json(json_path, rows);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
