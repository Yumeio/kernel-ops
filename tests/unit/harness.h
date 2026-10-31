#pragma once
// =============================================================================
// kernel-ops | tests/unit/harness.h
// Minimal C++ test harness: device buffers, Tensor views, Status checks and
// fp64 comparison. Tests return 77 (CTest SKIP_RETURN_CODE) when the machine
// lacks the required exact SM, GPUs or VRAM. No Python, no framework.
// Requires: common/macros.cuh, common/status.h, common/dtype.cuh, common/tensor.h.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"

namespace kops_test {

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline void record_failure(const char* what, const char* detail, const char* file, int line) {
  ++failure_count();
  std::fprintf(stderr, "FAIL %s:%d: %s%s%s\n", file, line, what, detail ? " — " : "",
               detail ? detail : "");
}

#define KOPS_TEST_CHECK(cond)                                                       \
  do {                                                                              \
    if (!(cond)) kops_test::record_failure(#cond, nullptr, __FILE__, __LINE__);     \
  } while (0)

inline void expect_ok(kops::Status status, const char* what, const char* file, int line) {
  if (status != kops::Status::kSuccess) {
    record_failure(what, kops::last_error_message(), file, line);
  }
}

inline void expect_status(kops::Status got, kops::Status want, const char* what, const char* file,
                          int line) {
  if (got != want) {
    std::string detail = std::string("got ") + kops::status_name(got) + ", want " +
                         kops::status_name(want);
    record_failure(what, detail.c_str(), file, line);
  }
}

#define KOPS_EXPECT_OK(expr, what) kops_test::expect_ok((expr), (what), __FILE__, __LINE__)
#define KOPS_EXPECT_STATUS(expr, want, what) \
  kops_test::expect_status((expr), (want), (what), __FILE__, __LINE__)

template <typename T>
T* device_alloc(size_t count) {
  T* ptr = nullptr;
  const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
  if (err != cudaSuccess) {
    record_failure("cudaMalloc", cudaGetErrorString(err), __FILE__, __LINE__);
    return nullptr;
  }
  return ptr;
}

template <typename T>
void device_free(T* ptr) {
  if (ptr != nullptr) cudaFree(ptr);
}

template <typename T>
void copy_in(T* dst, const std::vector<T>& src) {
  const cudaError_t err =
      cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) record_failure("cudaMemcpy H2D", cudaGetErrorString(err), __FILE__, __LINE__);
}

template <typename T>
std::vector<T> copy_out(const T* src, size_t count) {
  std::vector<T> out(count);
  const cudaError_t err =
      cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) record_failure("cudaMemcpy D2H", cudaGetErrorString(err), __FILE__, __LINE__);
  return out;
}

inline float to_float(float x) { return x; }
inline float to_float(__half x) { return __half2float(x); }
inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

inline void from_float(float& dst, float src) { dst = src; }
inline void from_float(__half& dst, float src) { dst = __float2half_rn(src); }
inline void from_float(__nv_bfloat16& dst, float src) { dst = __float2bfloat16_rn(src); }

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

inline bool make_context(kops::LaunchContext* out) {
  const kops::Status status = kops::make_launch_context(0, nullptr, out);
  if (status != kops::Status::kSuccess) {
    record_failure("make_launch_context", kops::last_error_message(), __FILE__, __LINE__);
    return false;
  }
  return true;
}

template <typename T>
void compare(const std::vector<T>& got, const std::vector<double>& want, double atol, double rtol,
             const char* what) {
  if (got.size() != want.size()) {
    record_failure(what, "size mismatch", __FILE__, __LINE__);
    return;
  }
  size_t bad = 0;
  size_t worst = 0;
  double worst_err = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(to_float(got[i]));
    const double w = want[i];
    if (!std::isfinite(g) && !std::isfinite(w)) continue;
    const double err = std::fabs(g - w);
    if (err > atol + rtol * std::fabs(w)) {
      ++bad;
      if (err > worst_err) {
        worst_err = err;
        worst = i;
      }
    }
  }
  if (bad != 0) {
    char detail[256];
    std::snprintf(detail, sizeof(detail), "%zu/%zu mismatched, worst index %zu: got %.7g want %.7g",
                  bad, got.size(), worst,
                  static_cast<double>(to_float(got[worst])), want[worst]);
    record_failure(what, detail, __FILE__, __LINE__);
  }
}

inline int finish(const char* name) {
  if (failure_count() == 0) {
    std::printf("PASS %s\n", name);
    return 0;
  }
  std::printf("FAIL %s (%d checks)\n", name, failure_count());
  return 1;
}

}  // namespace kops_test
