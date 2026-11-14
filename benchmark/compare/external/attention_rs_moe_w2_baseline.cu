// Dev-only baseline: attention.rs Moet W2 pack (`moe_w2_pack_from_mxfp4`, MIT).
// Never built by CMake, never linked into kops. Compile together with
//   3rd/attention.rs/src/kernels/src/moe_w2_pack.cu
// via tools/dev/build_lowbit_baselines.sh. Input is already packed MXFP4 + UE8M0
// block scales; kops `moe_w2::pack` takes bf16 and computes fp32 group scales, so
// compare elements/s, not GB/s.
//
//   tools/dev/build_lowbit_baselines.sh
//   build/compare/attention_rs_moe_w2 --flush-mb 256 --rounds 3 --json out/b17/attention-rs-w2.json
//   build/compare/attention_rs_moe_w2 --only moe.w2.pack --json out/b17/w2-case.json
//
// Timing follows benchmark/harness.h through ext_timing.h: one event pair per launch in
// cold mode (after an optional L2 flush), groups of 20 launches in hot mode. The warmup
// and iteration counts match the kops `moe.w2.pack` case (10/100).
#include <cuda_runtime.h>
#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ext_timing.h"

extern "C" void moe_w2_pack_from_mxfp4(const uint8_t* packed, const uint8_t* scales_in,
                                       uint8_t* planes, uint8_t* scales_out, int E, int N, int K,
                                       int64_t stream_i64);

namespace {
constexpr int kE = 8, kN = 2048, kK = 4096;
constexpr int kWarmup = 10, kIters = 100;
constexpr const char* kCaseId = "moe.w2.pack.mxfp4.e8.n2048.k4096";

struct Buf {
  void* p = nullptr;
  Buf(size_t n) {
    if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) std::exit(1);
    cudaMemset(p, 0, n ? n : 1);
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};
}  // namespace

int main(int argc, char** argv) {
  const ext_bench::Options opt = ext_bench::parse_options(argc, argv);
  ext_bench::Flusher flush(opt.flush_mb);

  const size_t packed_n = static_cast<size_t>(kE) * kN * kK / 2;
  const size_t scale_n = static_cast<size_t>(kE) * kN * kK / 32;
  const size_t plane_n = static_cast<size_t>(kE) * kN * kK / 4;
  Buf packed(packed_n), sin(scale_n), planes(plane_n), sout(scale_n);
  cudaMemset(packed.p, 0x44, packed_n);
  cudaMemset(sin.p, 0x7F, scale_n);

  const double elems = static_cast<double>(kE) * kN * kK;
  const double bytes =
      static_cast<double>(packed_n) + static_cast<double>(scale_n) +
      static_cast<double>(plane_n) + static_cast<double>(scale_n);

  std::vector<ext_bench::Row> rows;
  if (ext_bench::wanted(opt, kCaseId)) {
    ext_bench::Row r;
    r.id = kCaseId;
    r.op = "moe_w2_pack";
    r.bytes = bytes;
    r.elems = elems;
    r.note = "input is packed MXFP4 + UE8M0; kops reads bf16 (compare Melem/s)";
    auto launch = [&] {
      moe_w2_pack_from_mxfp4(static_cast<const uint8_t*>(packed.p),
                             static_cast<const uint8_t*>(sin.p), static_cast<uint8_t*>(planes.p),
                             static_cast<uint8_t*>(sout.p), kE, kN, kK, 0);
    };
    if (!ext_bench::measure(launch, kWarmup, kIters, opt, flush, &r.st)) {
      std::fprintf(stderr, "%s: timing failed\n", kCaseId);
      return 1;
    }
    rows.push_back(r);
  } else {
    std::printf("%s: filtered out by --only/--skip\n", kCaseId);
  }

  ext_bench::print_rows("attention.rs Moet W2 pack (dev-only)", opt, rows);
  const double peak = ext_bench::peak_gbs();
  for (const auto& r : rows) {
    const double gbs = r.st.median_ms > 0.0 ? r.bytes / (r.st.median_ms * 1e6) : 0.0;
    if (peak > 0.0 && gbs > peak * 1.05) {
      std::fprintf(stderr, "warning: %s reports %.0f GB/s > device peak %.0f GB/s\n",
                   r.id.c_str(), gbs, peak);
    }
  }
  ext_bench::write_json(opt.json, "attention.rs Moet W2 pack (dev baseline)", opt, rows);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
