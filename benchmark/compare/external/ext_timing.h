// Dev-only timing helper for the external 3rd-party harnesses in this directory
// (currently attention.rs low-bit / W2; other harnesses can adopt it the same way).
// Never built by CMake, never included from csrc/, never linked into kops.
//
// It reproduces the protocol of benchmark/harness.h `measure()` so a third-party median is
// comparable with the kops median of the same case:
//   * --flush-mb N > 0 (run_thirdparty.py passes --flush-l2, default 256): before every
//     sample a D2D copy of N MiB evicts L2, then one event pair times exactly one launch;
//   * --flush-mb 0: hot L2, one event pair around a group of 20 launches (sample = group/20);
//   * `warmup` launches first, `rounds * iters` launches timed, median/p10/p90 of the samples.
// Pass the stream the third-party code launches on; the flush is issued on that stream.
//
// `--only` / `--skip` are comma-separated substring filters with the same semantics as
// benchmark/harness.h, so interleave.py can run one case per process.
//
// The JSON output carries the kops row schema (median/p10/p90/min/samples/gbs/protocol)
// plus a `provenance` object. Commit/dirty state comes from the run environment
// (KOPS_BENCH_COMMIT / KOPS_BENCH_DIRTY / KOPS_THIRDPARTY*) which run_thirdparty.py and
// interleave.py set; individual flags override it. A standalone run reports "unknown".
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ext_bench {

inline const char* env_string(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? v : fallback;
}

/// -1 means "unknown" (no env var / flag); 0 and 1 are the recorded dirty states.
inline int env_int(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0) return 0;
  if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0) return 1;
  return fallback;
}

struct Options {
  const char* json = nullptr;
  int flush_mb = 0;
  int rounds = 1;
  int iters = 0;  // 0: per-case default (the kops case table value)
  const char* only = nullptr;
  const char* skip = nullptr;

  std::string commit = env_string("KOPS_BENCH_COMMIT", "unknown");
  int dirty = env_int("KOPS_BENCH_DIRTY", -1);
  std::string thirdparty = env_string("KOPS_THIRDPARTY", "");
  std::string thirdparty_commit = env_string("KOPS_THIRDPARTY_COMMIT", "unknown");
  int thirdparty_dirty = env_int("KOPS_THIRDPARTY_DIRTY", -1);
  std::string source = env_string("KOPS_BENCH_SOURCE", "");
};

inline Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) {
      o.json = argv[++i];
    } else if (std::strcmp(argv[i], "--flush-mb") == 0 || std::strcmp(argv[i], "--flush-l2") == 0) {
      o.flush_mb = std::max(0, std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--rounds") == 0) {
      o.rounds = std::max(1, std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      o.iters = std::max(0, std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--only") == 0) {
      o.only = argv[++i];
    } else if (std::strcmp(argv[i], "--skip") == 0) {
      o.skip = argv[++i];
    } else if (std::strcmp(argv[i], "--commit") == 0) {
      o.commit = argv[++i];
    } else if (std::strcmp(argv[i], "--dirty") == 0) {
      o.dirty = std::atoi(argv[++i]) ? 1 : 0;
    } else if (std::strcmp(argv[i], "--thirdparty") == 0) {
      o.thirdparty = argv[++i];
    } else if (std::strcmp(argv[i], "--thirdparty-commit") == 0) {
      o.thirdparty_commit = argv[++i];
    } else if (std::strcmp(argv[i], "--thirdparty-dirty") == 0) {
      o.thirdparty_dirty = std::atoi(argv[++i]) ? 1 : 0;
    } else if (std::strcmp(argv[i], "--source") == 0) {
      o.source = argv[++i];
    }
  }
  return o;
}

/// Comma-separated substring match, byte-for-byte the kops benchmark semantics.
inline bool match_csv(const std::string& id, const char* csv) {
  if (csv == nullptr || *csv == '\0') return false;
  std::string list(csv);
  size_t begin = 0;
  while (begin <= list.size()) {
    const size_t end = list.find(',', begin);
    std::string token =
        list.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    const size_t first = token.find_first_not_of(" \t");
    const size_t last = token.find_last_not_of(" \t");
    if (first != std::string::npos) {
      token = token.substr(first, last - first + 1);
      if (id.find(token) != std::string::npos) return true;
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return false;
}

inline bool wanted(const Options& o, const std::string& id) {
  if (o.only != nullptr && *o.only != '\0' && !match_csv(id, o.only)) return false;
  if (o.skip != nullptr && *o.skip != '\0' && match_csv(id, o.skip)) return false;
  return true;
}

struct Stats {
  double median_ms = 0.0, p10_ms = 0.0, p90_ms = 0.0, p99_ms = 0.0, min_ms = 0.0;
  int samples = 0;
};

/// Same roofline estimate as benchmark/harness.h DeviceInfo::peak_gbs.
inline double peak_gbs() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0.0;
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 0.0;
  return 2.0 * static_cast<double>(clock_khz) * 1.0e3 *
         (static_cast<double>(prop.memoryBusWidth) / 8.0) / 1.0e9;
}

class Flusher {
 public:
  explicit Flusher(int mb) : bytes_(static_cast<size_t>(mb) * 1024u * 1024u) {
    if (bytes_ == 0) return;
    if (cudaMalloc(&src_, bytes_) != cudaSuccess || cudaMalloc(&dst_, bytes_) != cudaSuccess) {
      std::fprintf(stderr, "ext_bench: flush buffer cudaMalloc(%zu) failed\n", bytes_);
      std::exit(2);
    }
    cudaMemset(src_, 0x5a, bytes_);
    cudaMemset(dst_, 0xa5, bytes_);
    cudaDeviceSynchronize();
  }
  ~Flusher() {
    if (src_ != nullptr) cudaFree(src_);
    if (dst_ != nullptr) cudaFree(dst_);
  }
  Flusher(const Flusher&) = delete;
  Flusher& operator=(const Flusher&) = delete;
  bool enabled() const { return bytes_ > 0; }
  void run(cudaStream_t s) const {
    if (bytes_ > 0) cudaMemcpyAsync(dst_, src_, bytes_, cudaMemcpyDeviceToDevice, s);
  }

 private:
  size_t bytes_ = 0;
  void* src_ = nullptr;
  void* dst_ = nullptr;
};

inline Stats summarize(std::vector<float> samples) {
  Stats st;
  if (samples.empty()) return st;
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  st.min_ms = samples.front();
  st.median_ms = samples[n / 2];
  st.p10_ms = samples[n / 10];
  st.p90_ms = samples[std::min(n - 1, (n * 9) / 10)];
  st.p99_ms = samples[std::min(n - 1, (n * 99) / 100)];
  st.samples = static_cast<int>(n);
  return st;
}

/// Same protocol as benchmark/harness.h measure(). `fn` enqueues one launch on `stream`.
template <typename Fn>
bool measure(Fn&& fn, int warmup, int iters, const Options& o, const Flusher& flush, Stats* out,
             cudaStream_t stream = nullptr) {
  for (int i = 0; i < warmup; ++i) fn();
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  cudaEvent_t a = nullptr, b = nullptr;
  if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return false;
  const int total = std::max(1, o.iters > 0 ? o.iters : iters);
  const int rounds = std::max(1, o.rounds);
  std::vector<float> samples;
  bool ok = true;
  if (flush.enabled()) {
    samples.reserve(static_cast<size_t>(total) * static_cast<size_t>(rounds));
    for (int r = 0; r < rounds && ok; ++r) {
      for (int i = 0; i < total; ++i) {
        flush.run(stream);
        cudaEventRecord(a, stream);
        fn();
        cudaEventRecord(b, stream);
        if (cudaEventSynchronize(b) != cudaSuccess) {
          ok = false;
          break;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(ms);
      }
    }
  } else {
    constexpr int kInner = 20;
    const int groups = std::max(1, total / kInner);
    samples.reserve(static_cast<size_t>(groups) * static_cast<size_t>(rounds));
    for (int r = 0; r < rounds && ok; ++r) {
      for (int g = 0; g < groups; ++g) {
        cudaEventRecord(a, stream);
        for (int j = 0; j < kInner; ++j) fn();
        cudaEventRecord(b, stream);
        if (cudaEventSynchronize(b) != cudaSuccess) {
          ok = false;
          break;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(ms / static_cast<float>(kInner));
      }
    }
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  if (!ok || samples.empty()) return false;
  *out = summarize(std::move(samples));
  return true;
}

inline std::string protocol(const Options& o) {
  char buf[128];
  if (o.flush_mb > 0) {
    std::snprintf(buf, sizeof(buf),
                  "cold: per-launch events after %d MiB L2 flush, rounds=%d", o.flush_mb, o.rounds);
  } else {
    std::snprintf(buf, sizeof(buf),
                  "hot: groups of 20 launches, rounds=%d", o.rounds);
  }
  return buf;
}

struct Row {
  std::string id, op;
  double bytes = 0.0, flops = 0.0, elems = 0.0;
  Stats st;
  std::string note;
};

inline void json_escape(FILE* f, const std::string& s) {
  std::fputc('"', f);
  for (char c : s) {
    const unsigned char ch = static_cast<unsigned char>(c);
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

inline void write_json(const char* path, const char* backend, const Options& o,
                       const std::vector<Row>& rows) {
  if (path == nullptr) return;
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "ext_bench: cannot write %s\n", path);
    return;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  int driver = 0;
  cudaDriverGetVersion(&driver);
  const std::string proto = protocol(o);
  std::fprintf(f, "{\"schema\":1,\"backend\":");
  json_escape(f, backend);
  std::fprintf(f,
               ",\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"device\":{\"name\":\"%s\",\"cc\":\"%d.%d\","
               "\"sms\":%d,\"driver\":%d},\"flush_l2_mb\":%d,\"rounds\":%d,\"protocol\":",
               prop.name, prop.major, prop.minor, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount, driver, o.flush_mb, o.rounds);
  json_escape(f, proto);
  std::fprintf(f, ",\"protocol_detail\":{\"mode\":\"%s\",\"flush_l2_mb\":%d,\"rounds\":%d,"
                  "\"samples_per_launch\":%d,\"iters\":%d}",
               o.flush_mb > 0 ? "cold" : "hot", o.flush_mb, o.rounds,
               o.flush_mb > 0 ? 1 : 20, o.iters);
  std::fprintf(f, ",\"provenance\":{\"commit\":");
  json_escape(f, o.commit);
  std::fprintf(f, ",\"dirty\":%s", o.dirty < 0 ? "null" : (o.dirty ? "true" : "false"));
  std::fprintf(f, ",\"thirdparty\":");
  json_escape(f, o.thirdparty);
  std::fprintf(f, ",\"thirdparty_commit\":");
  json_escape(f, o.thirdparty_commit);
  std::fprintf(f, ",\"thirdparty_dirty\":%s",
               o.thirdparty_dirty < 0 ? "null" : (o.thirdparty_dirty ? "true" : "false"));
  std::fprintf(f, ",\"source\":");
  json_escape(f, o.source);
  std::fprintf(f, "},\"results\":[");
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    const double gbs = r.st.median_ms > 0.0 ? r.bytes / (r.st.median_ms * 1e6) : 0.0;
    const double tflops = r.st.median_ms > 0.0 ? r.flops / (r.st.median_ms * 1e9) : 0.0;
    const double melem = r.st.median_ms > 0.0 ? r.elems / (r.st.median_ms * 1e6) : 0.0;
    std::fprintf(f,
                 "%s\n{\"id\":\"%s\",\"op\":\"%s\",\"bytes\":%.0f,\"flops\":%.0f,"
                 "\"median_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,"
                 "\"min_ms\":%.6f,\"samples\":%d,\"gbs\":%.3f,\"tflops\":%.4f,"
                 "\"melem_s\":%.3f,\"over_peak\":false,\"note\":",
                 i ? "," : "", r.id.c_str(), r.op.c_str(), r.bytes, r.flops, r.st.median_ms,
                 r.st.p10_ms, r.st.p90_ms, r.st.p99_ms, r.st.min_ms, r.st.samples, gbs, tflops,
                 melem);
    json_escape(f, r.note);
    std::fprintf(f, "}");
  }
  std::fprintf(f, "\n]}\n");
  std::fclose(f);
}

inline void print_rows(const char* title, const Options& o, const std::vector<Row>& rows) {
  std::printf("%s  [%s]\n", title, protocol(o).c_str());
  std::printf("%-52s %10s %10s %10s %9s %9s %9s %8s\n", "case", "median_ms", "p10_ms", "p90_ms",
              "GB/s", "TFLOP/s", "Melem/s", "samples");
  for (const Row& r : rows) {
    const double gbs = r.st.median_ms > 0.0 ? r.bytes / (r.st.median_ms * 1e6) : 0.0;
    const double tf = r.st.median_ms > 0.0 ? r.flops / (r.st.median_ms * 1e9) : 0.0;
    const double melem = r.st.median_ms > 0.0 ? r.elems / (r.st.median_ms * 1e6) : 0.0;
    std::printf("%-52s %10.4f %10.4f %10.4f %9.1f %9.3f %9.1f %8d\n", r.id.c_str(),
                r.st.median_ms, r.st.p10_ms, r.st.p90_ms, gbs, tf, melem, r.st.samples);
  }
}

}  // namespace ext_bench
