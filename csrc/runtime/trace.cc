// =============================================================================
// kernel-ops | csrc/runtime/trace.cc
// Environment record and trace lifecycle. Per-launch event tracing is a later
// milestone; the contract here is deliberately allocation-free in the hot path.
// =============================================================================
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/runtime/trace.h"

namespace kops::trace {
namespace {
TraceConfig g_config{};

const char* json_escape(char* out, size_t n, const char* s) {
  if (out == nullptr || n == 0) return "";
  size_t o = 0;
  for (const char* p = s; *p != '\0' && o + 2 < n; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (c == '"' || c == '\\') {
      out[o++] = '\\';
      out[o++] = static_cast<char>(c);
    } else if (c >= 0x20) {
      out[o++] = static_cast<char>(c);
    }
  }
  out[o] = '\0';
  return out;
}
}  // namespace

Status trace_configure(const TraceConfig& cfg) {
  if (cfg.enabled && cfg.output_path == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "trace_configure: output_path is required");
  }
  g_config = cfg;
  return Status::kSuccess;
}

Status trace_flush() { return Status::kSuccess; }

Status write_environment_record(const char* path) {
  if (path == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "write_environment_record: null path");
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return detail::set_error(Status::kCudaError, "write_environment_record: no CUDA device");
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, device) != cudaSuccess) {
    return detail::set_error(Status::kCudaError, "write_environment_record: device query failed");
  }
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "write_environment_record: cannot open %s", path);
  }
  char name[512];
  json_escape(name, sizeof(name), p.name);
  std::fprintf(f,
               "{\"gpu\":\"%s\",\"sm\":%d,\"num_sms\":%d,\"driver\":%d,\"runtime\":%d,"
               "\"toolkit\":%d,\"clock_khz\":%d,\"memory_clock_khz\":%d,"
               "\"bus_width\":%d,\"trace_enabled\":%d}\n",
               name, p.major * 100 + p.minor * 10, p.multiProcessorCount, driver, runtime,
               CUDART_VERSION, p.clockRate, p.memoryClockRate, p.memoryBusWidth,
               g_config.enabled ? 1 : 0);
  std::fclose(f);
  return Status::kSuccess;
}

}  // namespace kops::trace
