// Dev-only baseline: attention.rs Moet W2 pack (`moe_w2_pack_from_mxfp4`, MIT).
// Never built by CMake. Compile together with
//   3rd/attention.rs/src/kernels/src/moe_w2_pack.cu
// via tools/dev/build_lowbit_baselines.sh. Input is already packed MXFP4 + UE8M0
// block scales; kops `moe_w2::pack` takes bf16 and computes fp32 group scales, so
// compare elements/s, not GB/s.
#include <cuda_runtime.h>
#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void moe_w2_pack_from_mxfp4(const uint8_t* packed, const uint8_t* scales_in,
                                       uint8_t* planes, uint8_t* scales_out, int E, int N, int K,
                                       int64_t stream_i64);

namespace {
constexpr int kE = 8, kN = 2048, kK = 4096;
constexpr int kIters = 50, kWarmup = 10;

struct Buf {
  void* p = nullptr;
  Buf(size_t n) {
    if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) std::exit(1);
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

float time_ms(void (*launch)(const uint8_t*, const uint8_t*, uint8_t*, uint8_t*, int, int, int, int64_t),
              const uint8_t* a, const uint8_t* b, uint8_t* c, uint8_t* d) {
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < kWarmup; ++i) launch(a, b, c, d, kE, kN, kK, 0);
  cudaEventRecord(s);
  for (int i = 0; i < kIters; ++i) launch(a, b, c, d, kE, kN, kK, 0);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return ms / kIters;
}
}  // namespace

int main(int argc, char** argv) {
  const char* json = nullptr;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json = argv[++i];
  const size_t packed_n = static_cast<size_t>(kE) * kN * kK / 2;
  const size_t scale_n = static_cast<size_t>(kE) * kN * kK / 32;
  const size_t plane_n = static_cast<size_t>(kE) * kN * kK / 4;
  Buf packed(packed_n), sin(scale_n), planes(plane_n), sout(scale_n);
  cudaMemset(packed.p, 0x44, packed_n);
  cudaMemset(sin.p, 0x7F, scale_n);
  const float ms = time_ms(moe_w2_pack_from_mxfp4, static_cast<const uint8_t*>(packed.p),
                           static_cast<const uint8_t*>(sin.p), static_cast<uint8_t*>(planes.p),
                           static_cast<uint8_t*>(sout.p));
  const double elems = static_cast<double>(kE) * kN * kK;
  const double gbs = (static_cast<double>(packed_n) + scale_n + plane_n + scale_n) / (ms * 1e6);
  std::printf("attention.rs Moet W2 pack (dev-only)\n");
  std::printf("%-42s %10s %10s %12s\n", "case", "median_ms", "GB/s", "Melem/s");
  std::printf("%-42s %10.4f %10.1f %12.1f\n", "moe.w2.pack.mxfp4.e8.n2048.k4096", ms, gbs,
              elems / (ms * 1e6));
  if (json != nullptr) {
    FILE* f = std::fopen(json, "w");
    if (f != nullptr) {
      std::fprintf(f,
                   "{\"backend\":\"attention.rs Moet W2 pack (dev baseline)\",\"results\":["
                   "{\"id\":\"moe.w2.pack.mxfp4.e8.n2048.k4096\",\"median_ms\":%.6f,"
                   "\"gbs\":%.3f,\"melem_s\":%.3f}]}\n",
                   ms, gbs, elems / (ms * 1e6));
      std::fclose(f);
    }
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
