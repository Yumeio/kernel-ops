#pragma once
// =============================================================================
// kernel-ops | benchmark/harness.h
// Minimal C++ benchmark harness: CUDA-event timing, per-case stats, JSON output
// in the same schema the PyTorch comparison scripts read. No dependency beyond
// the CUDA toolkit and the kernel-ops headers.
//
// Case fields are the shared spec between the C++ benchmark and
// benchmark/compare/*.py:
//   rows / cols  generic shape; meaning per op:
//                  rms_norm            rows x cols
//                  fused_add_rms_norm  rows x cols
//                  activation          rows x cols (flat)
//                  act_and_mul         rows x cols, input is rows x 2*cols
//                  merge_state         (rows = B*H) x cols (= head_dim)
//   bytes / flops  used for GB/s and TFLOP/s; reported, not enforced
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"

#ifndef KOPS_BUILD_COMMIT
#define KOPS_BUILD_COMMIT "unknown"
#endif
#ifndef KOPS_BUILD_DIRTY
#define KOPS_BUILD_DIRTY 1
#endif
#ifndef KOPS_BUILD_FLAGS_HASH
#define KOPS_BUILD_FLAGS_HASH "unknown"
#endif
#ifndef KOPS_BUILD_ARCHS
#define KOPS_BUILD_ARCHS "unknown"
#endif
#ifndef KOPS_TOOLKIT_VERSION
#define KOPS_TOOLKIT_VERSION "unknown"
#endif

namespace kops_bench {

struct Case {
  std::string id;
  std::string op;
  kops::DType dtype;
  int64_t rows;
  int64_t cols;
  int64_t bytes;
  int64_t flops;
  int warmup;
  int iters;
};

struct Stats {
  double median_ms = 0.0;
  double p10_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
};

struct Result {
  const Case* c;
  Stats s;
};

struct DeviceInfo {
  char name[256] = {};
  int major = 0;
  int minor = 0;
  int sms = 0;
  int driver = 0;
  int runtime = 0;
  int memory_clock_khz = 0;
  int bus_width = 0;
  double peak_gbs = 0.0;
};

inline const char* dtype_label(kops::DType d) {
  switch (d) {
    case kops::DType::kFloat32: return "float32";
    case kops::DType::kBFloat16: return "bfloat16";
    case kops::DType::kFloat16: return "float16";
    case kops::DType::kInt32: return "int32";
    case kops::DType::kUInt32: return "uint32";
    case kops::DType::kInt64: return "int64";
    case kops::DType::kInt8: return "int8";
    case kops::DType::kUInt8: return "uint8";
    case kops::DType::kFloat8E4M3: return "float8_e4m3fn";
    default: return "other";
  }
}

inline int dtype_size(kops::DType d) {
  switch (d) {
    case kops::DType::kFloat32:
    case kops::DType::kInt32:
    case kops::DType::kUInt32: return 4;
    case kops::DType::kInt64: return 8;
    default: return 2;
  }
}

inline kops::Tensor tensor1d(void* data, kops::DType dtype, int64_t n) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

inline kops::Tensor tensor2d(void* data, kops::DType dtype, int64_t rows, int64_t cols) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = cols;
  t.stride[1] = 1;
  return t;
}

inline kops::Tensor tensor4d(void* data, kops::DType dtype, int64_t d0, int64_t d1, int64_t d2,
                             int64_t d3) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 4;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.shape[3] = d3;
  t.stride[0] = d1 * d2 * d3;
  t.stride[1] = d2 * d3;
  t.stride[2] = d3;
  t.stride[3] = 1;
  return t;
}

inline kops::Tensor tensor3d(void* data, kops::DType dtype, int64_t d0, int64_t d1, int64_t d2) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 3;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = d1 * d2;
  t.stride[1] = d2;
  t.stride[2] = 1;
  return t;
}

inline bool cuda_ok(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return true;
  std::fprintf(stderr, "[bench] %s: %s\n", what, cudaGetErrorString(err));
  return false;
}

inline bool status_ok(kops::Status s, const char* what) {
  if (s == kops::Status::kSuccess) return true;
  std::fprintf(stderr, "[bench] %s: %s\n", what, kops::last_error_message());
  return false;
}

inline bool query_device(DeviceInfo* out) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
  std::snprintf(out->name, sizeof(out->name), "%s", p.name);
  out->major = p.major;
  out->minor = p.minor;
  out->sms = p.multiProcessorCount;
  out->memory_clock_khz = 0;   // attribute query: CUDA 13 removed cudaDeviceProp::memoryClockRate
  cudaDeviceGetAttribute(&out->memory_clock_khz, cudaDevAttrMemoryClockRate, dev);
  out->bus_width = p.memoryBusWidth;
  out->driver = [] { int v = 0; cudaDriverGetVersion(&v); return v; }();
  out->runtime = [] { int v = 0; cudaRuntimeGetVersion(&v); return v; }();
  out->peak_gbs = 2.0 * static_cast<double>(out->memory_clock_khz) * 1.0e3 *
                  (static_cast<double>(out->bus_width) / 8.0) / 1.0e9;
  return true;
}

template <typename Fn>
inline bool measure(const Case& c, Fn fn, Stats* out) {
  for (int i = 0; i < c.warmup; ++i) fn();

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate") ||
      !cuda_ok(cudaEventCreate(&stop), "cudaEventCreate")) {
    return false;
  }

  // Time a small group of launches between two events. Per-iteration events
  // cost several microseconds and dominate kernels shorter than ~0.1 ms.
  constexpr int kInner = 20;
  const int groups = std::max(1, c.iters / kInner);
  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(groups));
  bool ok = true;
  for (int i = 0; i < groups; ++i) {
    cudaEventRecord(start);
    for (int j = 0; j < kInner; ++j) fn();
    cudaEventRecord(stop);
    if (cudaEventSynchronize(stop) != cudaSuccess) {
      ok = false;
      break;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    samples.push_back(ms / static_cast<float>(kInner));
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!ok || samples.empty()) return false;

  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  out->min_ms = samples.front();
  out->median_ms = samples[n / 2];
  out->p10_ms = samples[n / 10];
  out->p90_ms = samples[std::min(n - 1, (n * 9) / 10)];
  out->p99_ms = samples[std::min(n - 1, (n * 99) / 100)];
  return true;
}

inline void json_string(FILE* f, const char* s) {
  std::fputc('"', f);
  for (const char* p = s; *p != '\0'; ++p) {
    const unsigned char ch = static_cast<unsigned char>(*p);
    if (ch == '"' || ch == '\\') {
      std::fputc('\\', f);
      std::fputc(ch, f);
    } else if (ch < 0x20) {
      std::fprintf(f, "\\u%04x", ch);
    } else {
      std::fputc(ch, f);
    }
  }
  std::fputc('"', f);
}

inline void dump_cases(const std::vector<Case>& cases, const DeviceInfo& dev) {
  std::printf("{\"schema\":1,\"backend\":\"kops\",\"device\":{\"name\":");
  json_string(stdout, dev.name);
  std::printf(",\"cc\":\"%d.%d\",\"sms\":%d},", dev.major, dev.minor, dev.sms);
  std::printf("\"cases\":[");
  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    if (i != 0) std::printf(",");
    std::printf("{\"id\":");
    json_string(stdout, c.id.c_str());
    std::printf(",\"op\":");
    json_string(stdout, c.op.c_str());
    std::printf(",\"dtype\":\"%s\",\"rows\":%lld,\"cols\":%lld,\"bytes\":%lld,"
                "\"flops\":%lld,\"warmup\":%d,\"iters\":%d}",
                dtype_label(c.dtype), static_cast<long long>(c.rows),
                static_cast<long long>(c.cols), static_cast<long long>(c.bytes),
                static_cast<long long>(c.flops), c.warmup, c.iters);
  }
  std::printf("]}\n");
}

inline void write_results(const char* path, const std::vector<Result>& results,
                          const DeviceInfo& dev) {
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "[bench] cannot open %s\n", path);
    return;
  }
  std::fprintf(f, "{\"schema\":1,\"backend\":\"kops\",\"provenance\":{\"commit\":");
  json_string(f, KOPS_BUILD_COMMIT);
  std::fprintf(f, ",\"dirty\":%d,\"flags_hash\":", KOPS_BUILD_DIRTY);
  json_string(f, KOPS_BUILD_FLAGS_HASH);
  std::fprintf(f, ",\"archs\":");
  json_string(f, KOPS_BUILD_ARCHS);
  std::fprintf(f, ",\"toolkit\":");
  json_string(f, KOPS_TOOLKIT_VERSION);
  std::fprintf(f, "},\"device\":{\"name\":");
  json_string(f, dev.name);
  std::fprintf(f, ",\"cc\":\"%d.%d\",\"sms\":%d,\"driver\":%d,\"runtime\":%d,"
               "\"peak_gbs\":%.1f},\"results\":[",
               dev.major, dev.minor, dev.sms, dev.driver, dev.runtime, dev.peak_gbs);
  for (size_t i = 0; i < results.size(); ++i) {
    const Result& r = results[i];
    const Case& c = *r.c;
    const double gbs = r.s.median_ms > 0.0
                           ? static_cast<double>(c.bytes) / (r.s.median_ms * 1.0e6)
                           : 0.0;
    const double tflops = r.s.median_ms > 0.0
                              ? static_cast<double>(c.flops) / (r.s.median_ms * 1.0e9)
                              : 0.0;
    if (i != 0) std::fprintf(f, ",");
    std::fprintf(f, "{\"id\":");
    json_string(f, c.id.c_str());
    std::fprintf(f, ",\"op\":");
    json_string(f, c.op.c_str());
    std::fprintf(f, ",\"dtype\":\"%s\",\"rows\":%lld,\"cols\":%lld,\"bytes\":%lld,"
                 "\"flops\":%lld,\"median_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,"
                 "\"p99_ms\":%.6f,\"min_ms\":%.6f,\"gbs\":%.2f,\"tflops\":%.4f}",
                 dtype_label(c.dtype), static_cast<long long>(c.rows),
                 static_cast<long long>(c.cols), static_cast<long long>(c.bytes),
                 static_cast<long long>(c.flops), r.s.median_ms, r.s.p10_ms, r.s.p90_ms,
                 r.s.p99_ms, r.s.min_ms, gbs, tflops);
  }
  std::fprintf(f, "]}\n");
  std::fclose(f);
}

inline void print_table(const std::vector<Result>& results, const DeviceInfo& dev) {
  std::printf("device: %s (sm_%d%d, %d SMs, peak %.1f GB/s)\n", dev.name, dev.major,
              dev.minor, dev.sms, dev.peak_gbs);
  std::printf("%-42s %10s %10s %10s %9s\n", "case", "median_ms", "min_ms", "GB/s",
              "TFLOP/s");
  for (const Result& r : results) {
    const Case& c = *r.c;
    const double gbs = r.s.median_ms > 0.0
                           ? static_cast<double>(c.bytes) / (r.s.median_ms * 1.0e6)
                           : 0.0;
    const double tflops = r.s.median_ms > 0.0
                              ? static_cast<double>(c.flops) / (r.s.median_ms * 1.0e9)
                              : 0.0;
    std::printf("%-42s %10.4f %10.4f %10.1f %9.3f\n", c.id.c_str(), r.s.median_ms, r.s.min_ms,
                gbs, tflops);
  }
}

struct Options {
  bool dump_cases = false;
  bool split_sweep = false;
  int iters = 0;
  const char* only = nullptr;
  const char* json_path = nullptr;
};

inline Options parse_options(int argc, char** argv) {
  Options o{};
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-cases") == 0) {
      o.dump_cases = true;
    } else if (std::strcmp(argv[i], "--split-sweep") == 0) {
      o.split_sweep = true;
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      o.iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
      o.only = argv[++i];
    } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      o.json_path = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("usage: %s [--dump-cases] [--only substr] [--iters N] [--split-sweep] "
                  "[--json out.json]\n",
                  argv[0]);
    }
  }
  return o;
}

}  // namespace kops_bench
